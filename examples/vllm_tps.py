#!/usr/bin/env python3
"""vLLM: BF16 vs '--representation statetree' on one model (each engine in its own process).

    python examples/vllm_tps.py --model unsloth/Llama-3.2-1B

Per engine: weight memory, greedy decode throughput (batch 1 / 8 / 32, 128 new tokens, CUDA graphs on), prefill
latency (batch 1), and teacher-forced top-1 tokens on fixed sequences (exactness check against BF16)."""
import argparse
import json
import os
import subprocess
import sys
import time

TEXT = ("The history of mathematics deals with the origin of discoveries in mathematics and the mathematical methods "
        "and notation of the past. Before the modern age and the worldwide spread of knowledge, written examples of new "
        "mathematical developments have come to light only in a few locales. From 3000 BC the Mesopotamian states of "
        "Sumer, Akkad and Assyria, followed closely by Ancient Egypt and the Levantine state of Ebla began using "
        "arithmetic, algebra and geometry for purposes of taxation, commerce, trade and also in the patterns in nature, "
        "the field of astronomy and to record time and formulate calendars. ")


def _inspect(worker):
    """runs inside the vLLM worker: how many layers are trees and how many bytes the model holds"""
    m = worker.model_runner.model
    trees = _trees(m)
    total = sum(t.numel() * t.element_size() for t in list(m.parameters()) + list(m.buffers()))
    return dict(tree_layers=len(trees), tree_bytes=sum(t.storage_bytes() for t in trees), model_bytes=total,
                tree_embeddings=sum(int(type(mod).__name__.endswith(("Embedding", "LMHead")) and
                                        getattr(mod, "statetree", None) is not None) for mod in m.modules()))


def _trees(m):
    seen, out = set(), []                           # a tied embedding / lm_head pair shares one tree
    for mod in m.modules():
        st = getattr(mod, "statetree", None)
        if st is not None and id(st) not in seen:
            seen.add(id(st))
            out.append(st)
    return out


def _layer_bench(worker, batches=(1, 8, 32), reps=30):
    """runs inside the vLLM worker: GPU time (one CUDA graph) of all tree layers of each shape back to back, vs the same
    layers as BF16 F.linear on their decoded weights (vLLM's unquantized path), with the plan each batch uses"""
    import torch
    import torch.nn.functional as F
    groups = {}
    for t in _trees(worker.model_runner.model):
        groups.setdefault((t.out_features, t.in_features), []).append(t)

    def gtime(fn):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
            fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        return round(e0.elapsed_time(e1) / reps, 4)

    out = {}
    with torch.no_grad():
        for (M, N), ts in groups.items():
            Ws = [t.dequantize() for t in ts]
            for B in batches:
                x = torch.randn(B, N, device=Ws[0].device, dtype=torch.bfloat16)
                out[f"{M}x{N}x{len(ts)}_b{B}"] = dict(tree_ms=gtime(lambda: [t._forward(x) for t in ts]),
                                                      bf16_ms=gtime(lambda: [F.linear(x, W) for W in Ws]),
                                                      plan=str(ts[0]._plan(B)))
            del Ws
            torch.cuda.empty_cache()
    return out


def worker(spec):
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")       # lets collective_rpc ship _inspect
    import statetree.vllm_plugin  # noqa: F401  (adds the 'representation' engine argument)
    from vllm import LLM, SamplingParams
    t0 = time.perf_counter()
    llm = LLM(model=spec["model"], dtype="bfloat16", representation=spec["representation"], seed=0,
              gpu_memory_utilization=spec["gpu_mem"], max_model_len=spec["max_len"], enforce_eager=spec["eager"],
              enable_prefix_caching=False,                                  # time real prefills, not cache hits
              tensor_parallel_size=spec.get("tp", 1), pipeline_parallel_size=spec.get("pp", 1),
              distributed_executor_backend="mp" if spec.get("tp", 1) * spec.get("pp", 1) > 1 else None)
    out = dict(load_s=round(time.perf_counter() - t0, 1))
    try:
        ranks = llm.collective_rpc(_inspect)
        out.update(ranks[0])
        out["ranks"] = ranks
    except Exception as e:                                                  # noqa: BLE001
        out["inspect_error"] = repr(e)[:200]
    if spec.get("layer_bench") and out.get("tree_layers"):
        try:
            out["layer_bench"] = llm.collective_rpc(_layer_bench)[0]
        except Exception as e:                                              # noqa: BLE001
            out["layer_bench_error"] = repr(e)[:300]
    ids = llm.get_tokenizer()(TEXT * 40).input_ids
    P = spec["prompt_len"]
    prompts = [{"prompt_token_ids": ids[3 * i:3 * i + P]} for i in range(max(spec["batches"]))]
    # exactness: teacher-forced top-1 token at every position of fixed sequences
    sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1)
    tf = llm.generate([{"prompt_token_ids": ids[i * 128:(i + 1) * 128]} for i in range(4)], sp, use_tqdm=False)
    out["tf_top1"] = [[next((t for t, lp in d.items() if lp.rank == 1), -1) if d else -1 for d in r.prompt_logprobs]
                      for r in tf]
    # greedy generations (for a token-level comparison)
    sp = SamplingParams(max_tokens=32, temperature=0, ignore_eos=True)
    out["gen"] = [list(r.outputs[0].token_ids) for r in llm.generate(prompts[:8], sp, use_tqdm=False)]
    # decode throughput
    sp = SamplingParams(max_tokens=spec["new"], temperature=0, ignore_eos=True)
    for B in spec["batches"]:
        llm.generate(prompts[:B], sp, use_tqdm=False)                       # warm
        ts = []
        for _ in range(2):
            t0 = time.perf_counter()
            llm.generate(prompts[:B], sp, use_tqdm=False)
            ts.append(time.perf_counter() - t0)
        out[f"decode_b{B}_tps"] = round(B * spec["new"] / min(ts), 1)
    # prefill latency
    sp = SamplingParams(max_tokens=1, temperature=0)
    for T in spec["prefill"]:
        p = [{"prompt_token_ids": (ids * 4)[7:7 + T]}]
        llm.generate(p, sp, use_tqdm=False)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            llm.generate(p, sp, use_tqdm=False)
            ts.append(time.perf_counter() - t0)
        out[f"prefill_{T}_ms"] = round(1e3 * sorted(ts)[1], 2)
    print("RESULT " + json.dumps(out), flush=True)


def run_engine(spec, log=None):
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", json.dumps(spec)], capture_output=True,
                       text=True)
    if log:
        with open(log, "w") as fh:
            fh.write(p.stdout + "\n" + p.stderr)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError(f"engine {spec['representation']} failed:\n{p.stdout[-3000:]}\n{p.stderr[-6000:]}")


def top1_agree(a, b):
    n = s = 0
    for ra, rb in zip(a, b):
        for x, y in zip(ra[1:], rb[1:]):
            n += 1
            s += int(x == y)
    return round(s / max(n, 1), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--prefill", default="512,2048")
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--eager", action="store_true", help="disable CUDA graphs / torch.compile")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs")
    ap.add_argument("--pp", type=int, default=1, help="pipeline-parallel stages")
    ap.add_argument("--layer-bench", action="store_true", help="also time every tree-layer shape vs BF16 F.linear "
                                                                   "inside the statetree engine (needs a BF16 copy)")
    ap.add_argument("--out", default="results/vllm_tps.json")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        return worker(json.loads(args.worker))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    base = dict(model=args.model, batches=[int(b) for b in args.batches.split(",")], new=args.new,
                prompt_len=args.prompt_len, prefill=[int(t) for t in args.prefill.split(",") if t],
                gpu_mem=args.gpu_mem, max_len=args.max_len, eager=args.eager, layer_bench=args.layer_bench,
                tp=args.tp, pp=args.pp)
    res = dict(model=args.model)
    for tag in ("bf16", "statetree"):
        res[tag] = run_engine(dict(base, representation=tag), log=os.path.splitext(args.out)[0] + f"_{tag}.log")
        print(tag, {k: v for k, v in res[tag].items() if k not in ("tf_top1", "gen", "layer_bench")}, flush=True)
    a, b = res["bf16"], res["statetree"]
    res["tf_top1_agree"] = top1_agree(a["tf_top1"], b["tf_top1"])
    res["gen_first_token_equal"] = sum(int(x[0] == y[0]) for x, y in zip(a["gen"], b["gen"]))
    for B in base["batches"]:
        res[f"speedup_decode_b{B}"] = round(b[f"decode_b{B}_tps"] / a[f"decode_b{B}_tps"], 3)
    for T in base["prefill"]:
        res[f"speedup_prefill_{T}"] = round(a[f"prefill_{T}_ms"] / b[f"prefill_{T}_ms"], 3)
    for r in (a, b):
        r.pop("tf_top1", None)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
