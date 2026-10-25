#!/usr/bin/env python3
"""Hugging Face transformers: time to the complete response and peak GPU memory, four configurations in one process:
weights BF16 or state trees (statetree.convert) x KV cache transformers StaticCache + SDPA or statetree's
StateTreeKVCache (lossless state-tree pages read by the page decode-attention kernel). Greedy, B prompts of P tokens,
O new tokens each, decoded with statetree.bench.StaticGraphDecoder (the model's own forward, one step in a CUDA graph).

  ttft_s     prefill of the batch -> first token of every row
  full_s     until all O tokens of every row are out (prefill + O - 1 graph steps)
  peak_gib   peak allocated GPU memory (torch.cuda.max_memory_allocated) over set-up, prefill, capture and decode
  kv_gib     the KV cache's own memory

    python examples/hf_response.py --model Qwen/Qwen3-8B --out results/hf_response.json
(the state-tree KV cache needs head size 128)"""
import argparse
import gc
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import statetree
from statetree import bench
from statetree.kvcache import StateTreeKVCache, enable

SCENARIOS = {"chat_b1": (1, 256, 512), "chat_b8": (8, 512, 512), "chat_b32": (32, 512, 512),
             "long_b1": (1, 16000, 512), "long_b4": (4, 8000, 512)}
SENTENCES = ("Rivers carve valleys over millions of years, while glaciers grind mountains into fjords. ",
             "In 1687 Newton published the Principia, describing motion and universal gravitation. ",
             "Proteins fold into shapes determined by their amino-acid sequences and the cell's chaperones. ",
             "The stock market fell sharply on Tuesday after the central bank raised interest rates again. ",
             "A recipe for bread needs flour, water, salt and yeast, and patience while the dough rises. ",
             "Compilers translate source code into machine instructions through parsing and optimisation passes. ",
             "The violinist tuned her instrument before the concert began in the crowded hall. ",
             "Photosynthesis converts light energy into chemical energy stored in glucose molecules. ")
GiB = 2 ** 30


class TreeKVDecoder(bench.StaticGraphDecoder):
    def __init__(self, model, batch, max_len, graph=True):
        super().__init__(model, batch, max_len, graph)
        self.cache = StateTreeKVCache(model.config, batch_size=batch, max_cache_len=max_len)


def cache_bytes(c):
    if isinstance(c, StateTreeKVCache):
        return c.storage_bytes()
    return sum(t.numel() * t.element_size() for lay in c.layers for t in (getattr(lay, "keys", None),
                                                                         getattr(lay, "values", None))
               if isinstance(t, torch.Tensor))


def clean():
    torch.cuda.synchronize()
    gc.collect()                                   # caches and graphs of earlier runs hold reference cycles
    torch.cuda.empty_cache()


@torch.no_grad()
def run(model, ids, cls, B, P, O, reps=2):
    prompts = torch.stack([ids[1009 * b:1009 * b + P] for b in range(B)])
    clean()
    torch.cuda.reset_peak_memory_stats()
    r = cls(model, B, P + O + 8)
    r.prefill(prompts)
    r.capture()
    r.generate(prompts, 8)                                                     # warm
    best = None
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        first = r.prefill(prompts)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(O - 1):
            r.graph.replay()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if best is None or t2 - t0 < best[1]:
            best = (t1 - t0, t2 - t0)
    toks = torch.cat([first, r.out[:, :O - 1]], 1)
    res = dict(ttft_s=round(best[0], 4), full_s=round(best[1], 4), out_tps=round(B * O / best[1], 1),
               peak_gib=round(torch.cuda.max_memory_allocated() / GiB, 3), kv_gib=round(cache_bytes(r.cache) / GiB, 3))
    del r
    clean()
    return res, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--out", default="results/hf_response.json")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    ids = tok("".join(SENTENCES[(i * 3 + i // 8) % 8] for i in range(4000))).input_ids
    ids = torch.tensor(ids, device="cuda")
    names = args.scenarios.split(",")
    res = dict(model=args.model, scenarios={n: SCENARIOS[n] for n in names})
    toks = {}
    for wrep in ("bf16", "statetree"):
        if wrep == "statetree":
            statetree.convert(model, verify=True)
        clean()
        res[f"weights_{wrep}_gib"] = round(torch.cuda.memory_allocated() / GiB, 3)
        for kv in ("static", "statetree"):
            tag = f"w-{wrep}_kv-{kv}"
            if kv == "statetree":
                enable(model)
                cls = TreeKVDecoder
            else:
                model.set_attn_implementation("sdpa")
                cls = bench.StaticGraphDecoder
            res[tag] = {}
            for name in names:
                B, P, O = SCENARIOS[name]
                try:
                    r, t = run(model, ids, cls, B, P, O)
                except torch.OutOfMemoryError as e:
                    r, t = dict(error="OOM: " + str(e)[:120]), None
                    clean()
                res[tag][name], toks[(tag, name)] = r, t
                print(f"{tag} {name} B={B} P={P} O={O}: {r}", flush=True)
    base = "w-bf16_kv-static"
    for tag in [t for t in res if t.startswith("w-") and t != base]:
        cmp = {}
        for name in names:
            a, b = res[base][name], res[tag][name]
            if "full_s" in a and "full_s" in b:
                same = (toks[(base, name)] == toks[(tag, name)]).long().cumprod(1).sum(1).float().mean().item()
                cmp[name] = dict(ttft_speedup=round(a["ttft_s"] / b["ttft_s"], 3),
                                 full_response_speedup=round(a["full_s"] / b["full_s"], 3),
                                 peak_memory_ratio=round(b["peak_gib"] / a["peak_gib"], 3),
                                 identical_greedy_prefix=round(same, 1))
        res[f"{tag}_vs_{base}"] = cmp
        print(tag, "vs", base, json.dumps(cmp), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
