"""Why the per-layer speedup shrinks end to end: one decode step of BF16 (fused) vs statetree, broken down.

For each batch size:
  step_ms        CUDA-graph decode step (the tok/s benchmark's runner)
  linear_ms      all linear layers of the model back to back in one CUDA graph (statetree.bench.linear_chain_ms)
  per-type       the same chain restricted to qkv / o / gate_up / down / lm_head
  kernels        one eager decode step under torch.profiler, GPU time per kernel class
                 (tree: mv/tm/dec/rows; gemm: cuBLAS/cutlass; attention/recurrence; other)
  amdahl         1 / ((1 - f) + f / s_lin) with f = BF16 linear share of the step, s_lin = linear-chain speedup

    python examples/step_profile.py --model unsloth/Llama-3.2-1B --save-dir results/llama32_1b_statetree \
        --runner llama --out results/step_profile_llama1b.json"""
import argparse
import gc
import json
import time

import torch
from transformers import AutoModelForCausalLM

import statetree
from statetree import bench

TYPES = {"qkv": ("q_proj", "k_proj", "v_proj", "qkv", "in_proj"), "o": ("o_proj", "out_proj"),
         "gate_up": ("gate_proj", "up_proj", "gate_up"), "down": ("down_proj",), "lm_head": ("lm_head",)}


def kclass(name):
    n = name.lower()
    if any(k in n for k in ("mv7", "tm_kernel", "dec7", "rows7")):
        return "tree"
    if any(k in n for k in ("flash", "fmha", "attention", "attn", "delta", "recurrent", "conv", "chunk_", "fla_")):
        return "attention/recurrence"
    if any(k in n for k in ("gemv", "gemm", "xmma", "cutlass", "cublas", "s16816", "splitk")):
        return "gemm"
    return "other"


def runner(model, kind, B, max_len, graph):
    return bench.DecodeRunner(model, B, max_len, graph=graph) if kind == "llama" else \
        bench.StaticGraphDecoder(model, B, max_len, graph=graph)


@torch.no_grad()
def step_ms(model, kind, ids, n=64):
    B, T = ids.shape
    r = runner(model, kind, B, T + n + 16, True)
    r.prefill(ids)
    r.capture()
    r.generate(ids, 8)
    _, dt = r.generate(ids, n)
    del r
    return dt / (n - 1) * 1e3


@torch.no_grad()
def kernel_breakdown(model, kind, ids, steps=8):
    B, T = ids.shape
    r = runner(model, kind, B, T + steps + 16, False)
    r.prefill(ids)
    for _ in range(3):
        r._step()
    torch.cuda.synchronize()
    acts = [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=acts) as prof:
        for _ in range(steps):
            r._step()
        torch.cuda.synchronize()
    out, names = {}, {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            c = kclass(e.name)
            us = e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
            out[c] = out.get(c, 0.0) + us
            names[e.name[:60]] = names.get(e.name[:60], 0.0) + us
    del r
    top = sorted(names.items(), key=lambda kv: -kv[1])[:12]
    return {k: round(v / steps / 1e3, 4) for k, v in out.items()}, [(k, round(v / steps / 1e3, 4)) for k, v in top]


def per_type(model, B):
    res = {}
    for t, keys in TYPES.items():
        inc = (lambda keys: lambda n: any(k in n for k in keys))(keys)
        if not any(inc(n) for n, _ in bench.linear_modules(model)):
            continue
        res[t] = round(bench.linear_chain_ms(model, B, include=inc), 4)
    return res


def measure(model, kind, prompts, batches, tag, res):
    for B in batches:
        ids = prompts[:B]
        r = res.setdefault(str(B), {}).setdefault(tag, {})
        r["step_ms"] = round(step_ms(model, kind, ids), 4)
        r["linear_ms"] = round(bench.linear_chain_ms(model, B), 4)
        r["per_type_ms"] = per_type(model, B)
        r["kernels_ms"], r["top_kernels"] = kernel_breakdown(model, kind, ids)
        print(tag, B, {k: r[k] for k in ("step_ms", "linear_ms", "per_type_ms", "kernels_ms")}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--runner", choices=("llama", "static"), default="llama")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]
    dev = "cuda"
    res = dict(model=args.model, gpu=torch.cuda.get_device_name(), torch=torch.__version__, batches={})
    g = torch.Generator().manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dev).eval()
    prompts = torch.randint(100, model.config.vocab_size - 100, (max(batches), args.prompt_len), generator=g).to(dev)
    bench.fuse_bf16(model)
    measure(model, args.runner, prompts, batches, "bf16_fused", res["batches"])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = statetree.load_pretrained(args.save_dir, device=dev)
    statetree.prepare(model, batch_sizes=batches)
    measure(model, args.runner, prompts, batches, "statetree", res["batches"])
    for B, d in res["batches"].items():
        a, b = d["bf16_fused"], d["statetree"]
        f = a["linear_ms"] / a["step_ms"]
        s_lin = a["linear_ms"] / b["linear_ms"]
        d["summary"] = dict(e2e_speedup=round(a["step_ms"] / b["step_ms"], 3), linear_speedup=round(s_lin, 3),
                            linear_share_bf16=round(f, 3), amdahl_pred=round(1 / ((1 - f) + f / s_lin), 3),
                            nonlinear_bf16_ms=round(a["step_ms"] - a["linear_ms"], 4),
                            nonlinear_tree_ms=round(b["step_ms"] - b["linear_ms"], 4),
                            per_type_speedup={t: round(a["per_type_ms"][t] / b["per_type_ms"][t], 3)
                                              for t in a["per_type_ms"] if t in b["per_type_ms"]})
        print(B, d["summary"], flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
