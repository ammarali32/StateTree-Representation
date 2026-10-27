"""vLLM with the state-tree KV cache ('--kv-representation statetree') against vLLM's BF16 KV cache: KV capacity,
greedy generations (decode path, CUDA graphs), prefix-cache hits read back from cached state-tree pages, and a long
prompt chunk-prefilled while short requests decode. Each engine runs in its own process.

    python -m pytest tests/test_vllm_kv.py -q
Model: STATETREE_TEST_KV_MODEL (default Qwen/Qwen3-8B; the KV kernels need head size 128).
GPU memory fraction: STATETREE_TEST_KV_GPU_MEM (default 0.85)."""
import importlib.util
import json
import os
import subprocess
import sys

os.environ.setdefault("STATETREE_MIN_PARAMS", "0")   # the tests expect every linear layer as a tree (small ones too)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("STATETREE_TEST_KV_MODEL", "Qwen/Qwen3-8B")
GPU_MEM = float(os.environ.get("STATETREE_TEST_KV_GPU_MEM", "0.85"))
HAVE_VLLM = importlib.util.find_spec("vllm") is not None
SENTENCES = ("Rivers carve valleys over millions of years, while glaciers grind mountains into fjords. ",
             "In 1687 Newton published the Principia, describing motion and universal gravitation. ",
             "Proteins fold into shapes determined by their amino-acid sequences and the cell's chaperones. ",
             "The stock market fell sharply on Tuesday after the central bank raised interest rates again. ",
             "Compilers translate source code into machine instructions through parsing and optimisation passes. ")

try:
    import pytest
    skip = pytest.mark.skipif(not HAVE_VLLM, reason="vllm not installed")
except ImportError:                                                         # plain: python tests/test_vllm_kv.py
    def skip(f):
        return f


def worker(spec):
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    import statetree.vllm_plugin  # noqa: F401
    from vllm import LLM, SamplingParams
    kw = dict(model=spec["model"], dtype="bfloat16", seed=0, gpu_memory_utilization=spec["gpu_mem"],
              max_model_len=4096, enable_prefix_caching=True, max_num_batched_tokens=1024, max_num_seqs=32)
    if spec["kv"] != "bf16":
        kw["kv_representation"] = spec["kv"]
    llm = LLM(**kw)
    cc = llm.llm_engine.vllm_config.cache_config
    out = dict(kv_tokens=cc.num_gpu_blocks * cc.block_size,
               backend=str(llm.llm_engine.vllm_config.attention_config.backend))
    ids = llm.get_tokenizer()("".join(SENTENCES[(i * 3) % 5] for i in range(1500))).input_ids
    greedy = lambda n: SamplingParams(max_tokens=n, temperature=0, ignore_eos=True)   # noqa: E731
    gp = [{"prompt_token_ids": ids[301 * i:301 * i + 200]} for i in range(8)]
    out["gen"] = [list(r.outputs[0].token_ids) for r in llm.generate(gp, greedy(48), use_tqdm=False)]
    pre = ids[5000:6000]                                                     # shared 1000-token prefix
    pp = [{"prompt_token_ids": pre + ids[7000 + 41 * i:7000 + 41 * i + 30]} for i in range(7)]
    llm.generate(pp[:1], greedy(2), use_tqdm=False)
    rs = llm.generate(pp[1:], greedy(32), use_tqdm=False)
    out["prefix_gen"] = [list(r.outputs[0].token_ids) for r in rs]
    out["prefix_cached"] = [getattr(r, "num_cached_tokens", None) for r in rs]
    cp = [{"prompt_token_ids": ids[9000:12000]}] + [{"prompt_token_ids": ids[13000 + 211 * i:13000 + 211 * i + 150]}
                                                    for i in range(3)]          # 3000 tokens in 1024-token chunks
    out["chunk_gen"] = [list(r.outputs[0].token_ids) for r in llm.generate(cp, greedy(32), use_tqdm=False)]
    print("RESULT " + json.dumps(out), flush=True)


def run(spec):
    os.makedirs(os.path.join(HERE, "_out"), exist_ok=True)
    log = os.path.join(HERE, "_out", f"vllm_kv_{spec['kv']}.log")
    with open(log, "w") as fh:
        subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", json.dumps(spec)], stdout=fh,
                       stderr=subprocess.STDOUT, text=True)
    text = open(log).read()
    for line in text.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError(f"engine kv={spec['kv']} failed, see {log}:\n{text[-4000:]}")


_CACHE = {}


def engine(kv):
    if kv not in _CACHE:
        _CACHE[kv] = run(dict(model=MODEL, kv=kv, gpu_mem=GPU_MEM))
    return _CACHE[kv]


def test_flag_selects_backend():
    if not HAVE_VLLM:
        return
    import statetree.vllm_plugin  # noqa: F401
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    ea = EngineArgs(model=MODEL, kv_representation="statetree")
    assert ea.attention_backend == AttentionBackendEnum.CUSTOM and ea.block_size == 16
    assert AttentionBackendEnum.CUSTOM.get_path().endswith("StateTreeKVBackend")


@skip
def test_capacity():
    st, bf = engine("statetree"), engine("bf16")
    assert st["backend"].endswith("CUSTOM"), st["backend"]
    assert st["kv_tokens"] >= 1.25 * bf["kv_tokens"], (st["kv_tokens"], bf["kv_tokens"])     # 8192 / 6400 = 1.28


@skip
def test_generations_match_bf16_cache():
    st, bf = engine("statetree"), engine("bf16")
    for key in ("gen", "prefix_gen", "chunk_gen"):
        same = sum(int(a == b) for a, b in zip(st[key], bf[key]))
        assert same >= len(bf[key]) - 1, (key, same)              # identical up to a near-tie in one sequence


@skip
def test_prefix_cache_hits():
    st = engine("statetree")
    if all(c is not None for c in st["prefix_cached"]):
        assert min(st["prefix_cached"]) >= 992                    # the shared prefix came from cached tree pages


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(json.loads(sys.argv[2]))
    else:
        for name, fn in list(globals().items()):
            if name.startswith("test_"):
                fn()
                print("ok", name, flush=True)
