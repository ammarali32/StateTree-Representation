#!/usr/bin/env python3
"""Llama-3.2-1B end to end: load (HF), convert with statetree, check, save, reload, and measure tokens/s.

    python examples/llama_tps.py --model unsloth/Llama-3.2-1B --save-dir llama32_1b_statetree
    (--no-fuse / --no-lm-head / --no-embeddings for ablations)

Decode tokens/s come from a CUDA-graph greedy decoder (statetree.bench.DecodeRunner) run on the BF16 baseline and on
the converted model; 'model.generate' (eager) is timed too. Prefill is timed at batch 1 for a few prompt lengths."""
import argparse
import gc
import json
import os
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import statetree
from statetree import bench

TEXT = ("The history of mathematics deals with the origin of discoveries in mathematics and the mathematical methods "
        "and notation of the past. Before the modern age and the worldwide spread of knowledge, written examples of new "
        "mathematical developments have come to light only in a few locales. From 3000 BC the Mesopotamian states of "
        "Sumer, Akkad and Assyria, followed closely by Ancient Egypt and the Levantine state of Ebla began using "
        "arithmetic, algebra and geometry for purposes of taxation, commerce, trade and also in the patterns in nature, "
        "the field of astronomy and to record time and formulate calendars. ")


def agree(a, b):
    """mean fraction of each row generated identically before the first differing token, and rows fully equal"""
    eq = (a == b).long().cumprod(1)
    return round(float(eq.sum(1).float().mean() / a.shape[1]), 4), int((eq[:, -1] == 1).sum())


@torch.no_grad()
def logits_check(model, seq, ref):
    """teacher-forced: relative error of the logits and top-1 agreement against the BF16 model"""
    lg = model(seq).logits.float()
    return dict(rel_err=round(float((lg - ref).norm() / ref.norm()), 5),
                top1_agree=round(float((lg.argmax(-1) == ref.argmax(-1)).float().mean()), 4))


def gpu_bytes(model):
    seen, n = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            n += t.numel() * t.element_size()
    return n


def measure(model, prompts, batches, n_new, prefill_lens, tag, res):
    r = res.setdefault(tag, {})
    toks = {}
    ids1 = prompts[:1]
    out, tps = bench.hf_generate_tps(model, ids1, 32)
    r["hf_generate_b1_tps"] = round(tps, 1)
    toks["hf"] = out
    for B in batches:
        t, tps, step = bench.decode_tps(model, prompts[:B], n_new)
        toks[B] = t
        r[f"decode_b{B}_tps"] = round(tps, 1)
        r[f"decode_b{B}_ms_per_step"] = round(step * 1e3, 3)
        print(f"[{tag}] batch {B}: {tps:.1f} tok/s ({step * 1e3:.3f} ms/step)", flush=True)
    for T in prefill_lens:
        r[f"prefill_{T}_ms"] = round(bench.prefill_ms(model, 1, T), 2)
        print(f"[{tag}] prefill {T}: {r[f'prefill_{T}_ms']} ms", flush=True)
    r["gpu_bytes"] = gpu_bytes(model)
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--prefill", default="512,2048")
    ap.add_argument("--save-dir", default="llama32_1b_statetree")
    ap.add_argument("--no-fuse", action="store_true", help="keep q/k/v and gate/up as separate layers")
    ap.add_argument("--no-lm-head", action="store_true", help="keep lm_head (and a tied embedding) in BF16")
    ap.add_argument("--no-embeddings", action="store_true", help="keep the input embedding in BF16")
    ap.add_argument("--out", default="results/llama_tps.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    batches = [int(b) for b in args.batches.split(",")]
    prefill_lens = [int(t) for t in args.prefill.split(",") if t]
    dev = "cuda"
    res = dict(model=args.model, gpu=torch.cuda.get_device_name(), torch=torch.__version__, new_tokens=args.new,
               prompt_len=args.prompt_len)

    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.perf_counter()
    new_api = tuple(int(v) for v in transformers.__version__.split(".")[:2]) >= (4, 56)
    kw = dict(dtype=torch.bfloat16) if new_api else dict(torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).to(dev).eval()
    res["hf_load_s"] = round(time.perf_counter() - t0, 1)
    ids = tok(TEXT * 8, return_tensors="pt").input_ids[0].to(dev)
    prompts = torch.stack([ids[i * 3:i * 3 + args.prompt_len] for i in range(max(batches))])
    seq = ids[:512].view(4, 128)
    with torch.no_grad():
        ref_logits = model(seq).logits.float()

    # ---------------------------------------------------------------- baseline BF16 (cuBLAS)
    base = measure(model, prompts, batches, args.new, prefill_lens, "bf16", res)
    runner = bench.DecodeRunner(model, 1, args.prompt_len + 40, graph=False)
    rt, _ = runner.generate(prompts[:1], 32)
    res["runner_vs_hf_generate_bf16"] = agree(rt, base["hf"])
    del runner

    # ---------------------------------------------------------------- convert
    t0 = time.perf_counter()
    statetree.convert(model, exclude=("lm_head",) if args.no_lm_head else (),
                      fuse=None if args.no_fuse else statetree.DEFAULT_FUSE, embeddings=not args.no_embeddings,
                      verify=True)
    res["variant"] = dict(fuse=not args.no_fuse, lm_head=not args.no_lm_head, embeddings=not args.no_embeddings)
    res["convert_s"] = round(time.perf_counter() - t0, 1)
    res["size"] = statetree.size_report(model)
    t0 = time.perf_counter()
    statetree.prepare(model, batch_sizes=sorted(set(batches + prefill_lens)), verbose=True)
    res["prepare_s"] = round(time.perf_counter() - t0, 1)
    tree = measure(model, prompts, batches, args.new, prefill_lens, "statetree", res)
    res["tokens_vs_bf16"] = {str(k): agree(tree[k], base[k]) for k in base}
    res["logits_vs_bf16"] = logits_check(model, seq, ref_logits)

    # ---------------------------------------------------------------- save / reload
    t0 = time.perf_counter()
    statetree.save_pretrained(model, args.save_dir, tokenizer=tok)
    res["save_s"] = round(time.perf_counter() - t0, 1)
    res["saved_bytes"] = sum(os.path.getsize(os.path.join(args.save_dir, f)) for f in os.listdir(args.save_dir))
    ref_state = {k: v.clone() for k, v in model.state_dict().items() if "tree" in k or "esc_" in k}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    model = statetree.load_pretrained(args.save_dir, device=dev)
    res["load_s"] = round(time.perf_counter() - t0, 1)
    sd = model.state_dict()
    res["reload_bit_identical"] = all(torch.equal(sd[k], v) for k, v in ref_state.items())
    del ref_state
    t1, _, _ = bench.decode_tps(model, prompts[:batches[0]], args.new)
    res["reload_tokens_vs_converted"] = agree(t1, tree[batches[0]])

    # ---------------------------------------------------------------- summary
    for B in batches:
        res[f"speedup_decode_b{B}"] = round(res["statetree"][f"decode_b{B}_tps"] / res["bf16"][f"decode_b{B}_tps"], 3)
    for T in prefill_lens:
        res[f"speedup_prefill_{T}"] = round(res["bf16"][f"prefill_{T}_ms"] / res["statetree"][f"prefill_{T}_ms"], 3)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
