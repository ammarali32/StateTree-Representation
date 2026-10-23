#!/usr/bin/env python3
"""Any Hugging Face causal LM (tested: Qwen3.5-9B, a hybrid Gated-DeltaNet / attention model): BF16 vs statetree.

    python examples/hf_tps.py --model Qwen/Qwen3.5-9B --save-dir qwen35_9b_statetree

Measured for BF16, BF16 with the same layer fusion as statetree ("bf16_fused"), and statetree:
  decode tokens/s   statetree.bench.StaticGraphDecoder: the model's own forward + transformers StaticCache, one greedy
                    step captured in a CUDA graph
  prefill ms        one forward over a batch-1 prompt
  linear chain ms   all linear layers back to back in one CUDA graph (the part statetree changes)
plus exactness (bit-exact conversion, teacher-forced logits vs BF16) and a save / load round trip."""
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
    lg = model(input_ids=seq, use_cache=False).logits.float()
    return dict(rel_err=round(float((lg - ref).norm() / ref.norm()), 5),
                top1_agree=round(float((lg.argmax(-1) == ref.argmax(-1)).float().mean()), 4))


def gpu_bytes(model):
    seen, n = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            n += t.numel() * t.element_size()
    return n


def measure(model, prompts, batches, n_new, prefill_lens, tag, res, hf_generate=True):
    r = res.setdefault(tag, {})
    toks = {}
    if hf_generate:
        out, tps = bench.hf_generate_tps(model, prompts[:1], 32)
        r["hf_generate_b1_tps"] = round(tps, 2)
        toks["hf"] = out
    for B in batches:
        t, tps, step = bench.decode_tps_static(model, prompts[:B], n_new)
        toks[B] = t
        r[f"decode_b{B}_tps"] = round(tps, 1)
        r[f"decode_b{B}_ms_per_step"] = round(step * 1e3, 3)
        r[f"linears_b{B}_ms"] = round(bench.linear_chain_ms(model, B, include=lambda n: "visual" not in n), 3)
        print(f"[{tag}] batch {B}: {tps:.1f} tok/s ({step * 1e3:.3f} ms/step), linear layers {r[f'linears_b{B}_ms']} ms",
              flush=True)
    for T in prefill_lens:
        r[f"prefill_{T}_ms"] = round(bench.prefill_ms_hf(model, 1, T), 2)
        print(f"[{tag}] prefill {T}: {r[f'prefill_{T}_ms']} ms", flush=True)
    r["gpu_bytes"] = gpu_bytes(model)
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--prefill", default="512,2048")
    ap.add_argument("--save-dir", default="qwen35_9b_statetree")
    ap.add_argument("--no-fused-baseline", action="store_true")
    ap.add_argument("--out", default="results/hf_tps.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    batches = [int(b) for b in args.batches.split(",")]
    prefill_lens = [int(t) for t in args.prefill.split(",") if t]
    dev = "cuda"
    res = dict(model=args.model, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
               transformers=transformers.__version__, new_tokens=args.new, prompt_len=args.prompt_len)

    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dev).eval()
    res["hf_load_s"] = round(time.perf_counter() - t0, 1)
    res["model_class"] = type(model).__name__
    ids = tok(TEXT * 8, return_tensors="pt").input_ids[0].to(dev)
    prompts = torch.stack([ids[i * 3:i * 3 + args.prompt_len] for i in range(max(batches))])
    seq = ids[:256].view(2, 128)
    with torch.no_grad():
        ref_logits = model(input_ids=seq, use_cache=False).logits.float()

    # ---------------------------------------------------------------- BF16 baseline (+ decoder self-check)
    base = measure(model, prompts, batches, args.new, prefill_lens, "bf16", res)
    eager, _ = bench.StaticGraphDecoder(model, 1, args.prompt_len + 40, graph=False).generate(prompts[:1], 32)
    res["static_decoder_eager_vs_hf_generate"] = agree(eager, base["hf"])
    res["static_decoder_graph_vs_eager"] = agree(base[batches[0]][:1, :32], eager) if batches[0] == 1 else None

    if not args.no_fused_baseline:
        n = bench.fuse_bf16(model)
        res["bf16_fused_groups"] = n
        measure(model, prompts, batches, args.new, prefill_lens, "bf16_fused", res, hf_generate=False)

    # ---------------------------------------------------------------- statetree
    t0 = time.perf_counter()
    statetree.convert(model, verify=True)
    res["convert_s"] = round(time.perf_counter() - t0, 1)
    res["size"] = statetree.size_report(model)
    t0 = time.perf_counter()
    statetree.prepare(model, batch_sizes=sorted(set(batches + prefill_lens + [args.prompt_len])), verbose=True)
    res["prepare_s"] = round(time.perf_counter() - t0, 1)
    tree = measure(model, prompts, batches, args.new, prefill_lens, "statetree", res)
    res["tokens_vs_bf16"] = {str(k): agree(tree[k], base[k]) for k in base}
    res["logits_vs_bf16"] = logits_check(model, seq, ref_logits)
    print("logits vs bf16", res["logits_vs_bf16"], flush=True)

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
    res["reload_bit_identical"] = all(torch.equal(sd[k], v) for k, v in ref_state.items()) and len(ref_state) > 0
    del ref_state, sd
    t1, _, _ = bench.decode_tps_static(model, prompts[:batches[0]], args.new)
    res["reload_tokens_vs_converted"] = agree(t1, tree[batches[0]])

    # ---------------------------------------------------------------- summary
    for ref in ("bf16", "bf16_fused"):
        if ref not in res:
            continue
        for B in batches:
            res[f"speedup_decode_b{B}_vs_{ref}"] = round(res["statetree"][f"decode_b{B}_tps"] / res[ref][f"decode_b{B}_tps"], 3)
            res[f"speedup_linears_b{B}_vs_{ref}"] = round(res[ref][f"linears_b{B}_ms"] / res["statetree"][f"linears_b{B}_ms"], 3)
        for T in prefill_lens:
            res[f"speedup_prefill_{T}_vs_{ref}"] = round(res[ref][f"prefill_{T}_ms"] / res["statetree"][f"prefill_{T}_ms"], 3)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
