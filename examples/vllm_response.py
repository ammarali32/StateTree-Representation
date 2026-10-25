#!/usr/bin/env python3
"""vLLM: time to the complete response, BF16 vs state trees (weights and / or KV cache), each engine in its own
process (prefix caching off: every prompt is really prefilled; CUDA graphs on; greedy, ignore_eos).

Per scenario (B requests of P prompt tokens, O new tokens each, all submitted at once):
  ttft_s   time to the first token of every request (a max_tokens=1 run of the same batch)
  full_s   time until every request's complete response is out (wall clock of the whole batch)
  out_tps  B * O / full_s
Per engine also: GPU memory of the weights and the KV-cache capacity (tokens) vLLM allocated in the same budget.
The 'capacity' scenario needs more KV memory than the BF16 cache has (vLLM queues / preempts), so it shows the
effect of the 1.28x denser cache (and of the smaller tree weights, which leave more memory to the cache).
    python examples/vllm_response.py --model Qwen/Qwen3-8B --out results/vllm_response.json
(the state-tree KV cache needs head size 128; --engines bf16:bf16,statetree:bf16 compares the weights alone)
"""
import argparse
import json
import os
import subprocess
import sys
import time

VARIED = ("Rivers carve valleys over millions of years, while glaciers grind mountains into fjords. ",
          "In 1687 Newton published the Principia, describing motion and universal gravitation. ",
          "Proteins fold into shapes determined by their amino-acid sequences and the cell's chaperones. ",
          "The stock market fell sharply on Tuesday after the central bank raised interest rates again. ",
          "A recipe for bread needs flour, water, salt and yeast, and patience while the dough rises. ",
          "Compilers translate source code into machine instructions through parsing and optimisation passes. ",
          "The violinist tuned her instrument before the concert began in the crowded hall. ",
          "Photosynthesis converts light energy into chemical energy stored in glucose molecules. ")
SCENARIOS = {                 # name: (B, P, O)
    "chat_b1": (1, 256, 512),
    "chat_b8": (8, 512, 512),
    "chat_b32": (32, 512, 512),
    "long_b1": (1, 16000, 512),
    "long_b8": (8, 8000, 512),
    "capacity_b96": (96, 1536, 512),
}


def _weights_gib(worker):
    m = worker.model_runner.model
    seen, n = set(), 0                                      # (state trees are buffers of registered submodules)
    for t in list(m.parameters()) + list(m.buffers()):
        if t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            n += t.numel() * t.element_size()
    return round(n / 2 ** 30, 3)


def worker(spec):
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")              # collective_rpc of _weights_gib
    import statetree.vllm_plugin  # noqa: F401
    from vllm import LLM, SamplingParams
    kw = dict(model=spec["model"], dtype="bfloat16", seed=0, gpu_memory_utilization=spec["gpu_mem"],
              max_model_len=spec["max_len"], enable_prefix_caching=False, max_num_seqs=spec["max_seqs"])
    if spec["rep"] != "bf16":
        kw["representation"] = spec["rep"]
    if spec["kv"] != "bf16":
        kw["kv_representation"] = spec["kv"]
    t0 = time.perf_counter()
    llm = LLM(**kw)
    cc = llm.llm_engine.vllm_config.cache_config
    out = dict(load_s=round(time.perf_counter() - t0, 1), num_gpu_blocks=cc.num_gpu_blocks,
               kv_tokens=cc.num_gpu_blocks * cc.block_size)
    try:
        out["weights_gib"] = llm.collective_rpc(_weights_gib)[0]
    except Exception as e:                                                   # noqa: BLE001
        out["weights_gib"] = repr(e)[:100]
    tok = llm.get_tokenizer()
    ids = tok("".join(VARIED[(i * 3 + i // 8) % 8] for i in range(16000))).input_ids
    sp = lambda n: SamplingParams(max_tokens=n, temperature=0, ignore_eos=True)   # noqa: E731
    llm.generate([{"prompt_token_ids": ids[:300]}] * 4, sp(16), use_tqdm=False)    # warm-up
    for name in spec["scenarios"]:
        B, P, O = SCENARIOS[name]
        prompts = [{"prompt_token_ids": ids[131 * b:131 * b + P]} for b in range(B)]
        r = {}
        ts = []
        for _ in range(2):
            t0 = time.perf_counter()
            llm.generate(prompts, sp(1), use_tqdm=False)
            ts.append(time.perf_counter() - t0)
        r["ttft_s"] = round(min(ts), 3)
        ts = []
        for _ in range(spec["reps"]):
            t0 = time.perf_counter()
            res = llm.generate(prompts, sp(O), use_tqdm=False)
            ts.append(time.perf_counter() - t0)
        r["full_s"] = round(min(ts), 3)
        r["out_tps"] = round(B * O / min(ts), 1)
        r["first_tokens"] = [list(x.outputs[0].token_ids[:24]) for x in res[:4]]
        out[name] = r
        print(f"{name}: B={B} P={P} O={O}: ttft {r['ttft_s']} s, full response {r['full_s']} s, "
              f"{r['out_tps']} tok/s", flush=True)
    print("RESULT " + json.dumps(out), flush=True)


def run_engine(spec, log):
    with open(log, "w") as fh:
        subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", json.dumps(spec)], stdout=fh,
                       stderr=subprocess.STDOUT, text=True)
    text = open(log).read()
    for line in text.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError(f"engine {spec['tag']} failed, see {log}:\n{text[-4000:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--engines", default="bf16:bf16,statetree:bf16,bf16:statetree,statetree:statetree",
                    help="weights:kv pairs, the first is the baseline")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-len", type=int, default=16640)
    ap.add_argument("--max-seqs", type=int, default=128)
    ap.add_argument("--out", default=None)
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        return worker(json.loads(args.worker))
    assert args.out and not os.path.exists(args.out), f"--out {args.out}: missing or exists (never overwritten)"
    base = dict(model=args.model, scenarios=args.scenarios.split(","), reps=args.reps, gpu_mem=args.gpu_mem,
                max_len=args.max_len, max_seqs=args.max_seqs)
    res = dict(model=args.model, scenarios={k: SCENARIOS[k] for k in base["scenarios"]})
    tags = []
    for e in args.engines.split(","):
        rep, kv = e.split(":")
        tag = f"w-{rep}_kv-{kv}"
        tags.append(tag)
        log = os.path.splitext(args.out)[0] + f"_{tag}.log"
        assert not os.path.exists(log), f"{log} exists"
        res[tag] = run_engine(dict(base, rep=rep, kv=kv, tag=tag), log)
        print(tag, {k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a != "first_tokens"})
                    for k, v in res[tag].items()}, flush=True)
    b = res[tags[0]]
    for t in tags[1:]:
        r = res[t]
        res[f"{t}_vs_{tags[0]}"] = {
            name: dict(ttft_speedup=round(b[name]["ttft_s"] / r[name]["ttft_s"], 3),
                       full_response_speedup=round(b[name]["full_s"] / r[name]["full_s"], 3),
                       first_tokens_equal=sum(int(x == y) for x, y in zip(b[name]["first_tokens"],
                                                                         r[name]["first_tokens"])))
            for name in base["scenarios"]}
        res[f"{t}_vs_{tags[0]}"]["kv_capacity_ratio"] = round(r["kv_tokens"] / b["kv_tokens"], 4)
        print(t, "vs", tags[0], json.dumps(res[f"{t}_vs_{tags[0]}"]), flush=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
