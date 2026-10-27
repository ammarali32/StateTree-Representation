"""vLLM plugin under vLLM's serving features: prefix caching, long prompts through the vocab-parallel lm_head (prompt
logprobs, > 512 rows at once), tensor parallelism and pipeline parallelism. Each engine runs in its own process;
the multi-GPU tests skip with fewer than 2 GPUs.

    python -m pytest tests/test_vllm_features.py -q
Model: STATETREE_TEST_MODEL (default unsloth/Llama-3.2-1B)."""
import importlib.util
import json
import os
import subprocess
import sys

os.environ.setdefault("STATETREE_MIN_PARAMS", "0")   # the tests expect every linear layer as a tree (small ones too)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))
MODEL = os.environ.get("STATETREE_TEST_MODEL", "unsloth/Llama-3.2-1B")
HAVE_VLLM = importlib.util.find_spec("vllm") is not None
LONG = 1500                                         # prompt tokens scored at once by the lm_head (> 512)


def _ngpu():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:                                                       # noqa: BLE001
        return 0


try:
    import pytest
    skip = pytest.mark.skipif(not HAVE_VLLM, reason="vllm not installed")
    two_gpus = pytest.mark.skipif(not HAVE_VLLM or _ngpu() < 2, reason="needs vllm and 2 GPUs")
    four_gpus = pytest.mark.skipif(not HAVE_VLLM or _ngpu() < 4, reason="needs vllm and 4 GPUs")
except ImportError:                                                         # plain: python tests/test_vllm_features.py
    def skip(f):
        return f

    def two_gpus(f):
        return f if _ngpu() >= 2 else (lambda: print("skip (needs 2 GPUs)", f.__name__))

    def four_gpus(f):
        return f if _ngpu() >= 4 else (lambda: print("skip (needs 4 GPUs)", f.__name__))


def worker(spec):
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    import statetree.vllm_plugin  # noqa: F401
    from vllm import LLM, SamplingParams
    import vllm_tps as V
    llm = LLM(model=spec["model"], dtype="bfloat16", representation=spec["representation"], seed=0,
              tensor_parallel_size=spec.get("tp", 1), pipeline_parallel_size=spec.get("pp", 1),
              enable_prefix_caching=spec.get("prefix", True), gpu_memory_utilization=spec.get("gpu_mem", 0.45),
              max_model_len=spec.get("max_len", 2048), disable_log_stats=False, enforce_eager=spec.get("eager", False),
              distributed_executor_backend="mp" if spec.get("tp", 1) * spec.get("pp", 1) > 1 else None)
    out = {"ranks": llm.collective_rpc(V._inspect)}
    ids = llm.get_tokenizer()(V.TEXT * 60).input_ids
    greedy = SamplingParams(max_tokens=16, temperature=0, ignore_eos=True)
    out["gen"] = [list(r.outputs[0].token_ids) for r in
                  llm.generate([{"prompt_token_ids": ids[5 * i:5 * i + 40]} for i in range(8)], greedy,
                               use_tqdm=False)]
    # prefix caching: six prompts share a 700-token prefix; the second pass reads it from the KV cache
    shared = ids[11:711]
    pp = [{"prompt_token_ids": shared + ids[900 + 25 * i:900 + 25 * i + 20]} for i in range(6)]

    def hits():
        try:
            return sum(m.value for m in llm.get_metrics() if m.name.endswith("prefix_cache_hits"))
        except Exception:                                                   # noqa: BLE001
            return None
    h0 = hits()
    out["prefix_gen1"] = [list(r.outputs[0].token_ids) for r in llm.generate(pp, greedy, use_tqdm=False)]
    out["prefix_gen2"] = [list(r.outputs[0].token_ids) for r in llm.generate(pp, greedy, use_tqdm=False)]
    h1 = hits()
    out["prefix_hits"] = None if h0 is None or h1 is None else h1 - h0
    # one long prompt: logits for all LONG positions in one pass (vocab-sized table, batch > 512)
    r = llm.generate([{"prompt_token_ids": ids[3:3 + LONG]}], SamplingParams(max_tokens=1, temperature=0,
                                                                           prompt_logprobs=1), use_tqdm=False)[0]
    out["long_top1"] = [next((t for t, lp in d.items() if lp.rank == 1), -1) if d else -1 for d in r.prompt_logprobs]
    print("RESULT " + json.dumps(out), flush=True)


def run(spec):
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", json.dumps(spec)],
                       capture_output=True, text=True)
    os.makedirs(os.path.join(HERE, "_out"), exist_ok=True)
    tag = f"{spec['representation']}_tp{spec.get('tp', 1)}_pp{spec.get('pp', 1)}"
    with open(os.path.join(HERE, "_out", f"vllm_features_{tag}.log"), "w") as fh:
        fh.write(p.stdout + "\n" + p.stderr)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError(f"engine {tag} failed:\n{p.stdout[-3000:]}\n{p.stderr[-6000:]}")


_CACHE = {}


def engine(rep, tp=1, pp=1):
    k = (rep, tp, pp)
    if k not in _CACHE:
        _CACHE[k] = run(dict(model=MODEL, representation=rep, tp=tp, pp=pp))
    return _CACHE[k]


def _agree(a, b):
    n = s = 0
    for x, y in zip(a, b):
        n += 1
        s += int(x == y)
    return s / max(n, 1)


def _check_against_bf16(st, bf16):
    assert all(r.get("tree_layers", 0) > 0 for r in st["ranks"]), st["ranks"]       # every rank / stage converted
    assert all(r.get("tree_layers", 0) == 0 for r in bf16["ranks"])
    assert sum(int(a[0] == b[0]) for a, b in zip(st["gen"], bf16["gen"])) >= 6, (st["gen"], bf16["gen"])
    assert len(st["long_top1"]) == LONG
    assert _agree(st["long_top1"][1:], bf16["long_top1"][1:]) >= 0.95


@skip
def test_long_prompt_logprobs_through_vocab_table():
    _check_against_bf16(engine("statetree"), engine("bf16"))


@skip
def test_prefix_caching():
    st, bf16 = engine("statetree"), engine("bf16")
    if st["prefix_hits"] is not None:
        assert st["prefix_hits"] >= 6 * 600, st["prefix_hits"]          # the shared prefix came from the KV cache
    assert all(a[0] == b[0] for a, b in zip(st["prefix_gen1"], st["prefix_gen2"]))
    assert sum(int(a == b) for a, b in zip(st["prefix_gen1"], st["prefix_gen2"])) >= 5
    assert sum(int(a[0] == b[0]) for a, b in zip(st["prefix_gen2"], bf16["prefix_gen2"])) >= 5


@two_gpus
def test_tensor_parallel_2():
    st = engine("statetree", tp=2)
    assert len(st["ranks"]) == 2
    one = engine("statetree")["ranks"][0]
    assert all(r["tree_layers"] == one["tree_layers"] for r in st["ranks"])   # every layer sharded and converted
    _check_against_bf16(st, engine("bf16", tp=2))


@two_gpus
def test_pipeline_parallel_2():
    st = engine("statetree", pp=2)
    assert len(st["ranks"]) == 2
    one = engine("statetree")["ranks"][0]
    # the stages split the decoder layers (a tied lm_head / embedding pair is one tree on each stage that holds it)
    assert sum(r["tree_layers"] for r in st["ranks"]) >= one["tree_layers"]
    _check_against_bf16(st, engine("bf16"))



@four_gpus
def test_tensor_parallel_4():
    st = engine("statetree", tp=4)
    assert len(st["ranks"]) == 4
    one = engine("statetree")["ranks"][0]
    assert all(r["tree_layers"] == one["tree_layers"] for r in st["ranks"])
    _check_against_bf16(st, engine("bf16"))

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(json.loads(sys.argv[2]))
    else:
        for name, fn in list(globals().items()):
            if name.startswith("test_"):
                fn()
                print("ok", name, flush=True)
