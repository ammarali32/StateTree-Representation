"""vLLM plugin tests (need vllm and a CUDA GPU; skipped otherwise).

    python -m pytest tests/test_vllm.py -q      or      python tests/test_vllm.py
Model: STATETREE_TEST_MODEL (default unsloth/Llama-3.2-1B). Every engine runs in its own process. The 'vllm serve'
test uses port STATETREE_TEST_PORT (default 18473)."""
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request

os.environ.setdefault("STATETREE_MIN_PARAMS", "0")   # the tests expect every linear layer as a tree (small ones too)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))
MODEL = os.environ.get("STATETREE_TEST_MODEL", "unsloth/Llama-3.2-1B")
PORT = int(os.environ.get("STATETREE_TEST_PORT", "18473"))
HAVE_VLLM = importlib.util.find_spec("vllm") is not None
_BIN = os.path.join(os.path.dirname(sys.executable), "vllm")
VLLM_CLI = [_BIN] if os.path.exists(_BIN) else [sys.executable, "-m", "vllm.entrypoints.cli.main"]

try:
    import pytest
    skip = pytest.mark.skipif(not HAVE_VLLM, reason="vllm not installed")
except ImportError:                                                         # plain: python tests/test_vllm.py
    def skip(f):
        return f

_CACHE = {}


def _engines():
    """BF16 and statetree engine results (small run, CUDA graphs on), computed once."""
    if not _CACHE:
        import vllm_tps as V
        spec = dict(model=MODEL, batches=[1, 8], new=16, prompt_len=32, prefill=[], gpu_mem=0.45, max_len=1024,
                    eager=False)
        _CACHE["bf16"] = V.run_engine(dict(spec, representation="bf16"))
        _CACHE["statetree"] = V.run_engine(dict(spec, representation="statetree"))
    return _CACHE["bf16"], _CACHE["statetree"]


@skip
def test_flag_registered():
    """'--representation' is a vLLM engine argument (CLI) and an EngineArgs / LLM keyword; it is not a quantization."""
    out = subprocess.run(VLLM_CLI + ["serve", "--help=all"],
                         capture_output=True, text=True, timeout=600)
    assert "--representation" in out.stdout + out.stderr, (out.stdout[-2000:], out.stderr[-2000:])
    import statetree.vllm_plugin  # noqa: F401
    from vllm.engine.arg_utils import EngineArgs
    ea = EngineArgs(model=MODEL, representation="statetree")
    assert ea.additional_config.get("representation") == "statetree"
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    assert "statetree" not in QUANTIZATION_METHODS


@skip
def test_layers_converted_and_smaller():
    bf16, st = _engines()
    assert st.get("tree_layers", 0) > 0, st
    assert bf16.get("tree_layers", 0) == 0
    assert st.get("tree_embeddings", 0) >= 1, st            # input embedding / lm_head stored as trees too
    # every converted linear weight costs ~12.13 bits instead of 16
    assert st["model_bytes"] < bf16["model_bytes"]
    linear_bf16 = st["tree_bytes"] * 16 / 12.13
    assert abs((bf16["model_bytes"] - st["model_bytes"]) - (linear_bf16 - st["tree_bytes"])) < 0.02 * bf16["model_bytes"]


@skip
def test_outputs_match_bf16():
    import vllm_tps as V
    bf16, st = _engines()
    assert V.top1_agree(bf16["tf_top1"], st["tf_top1"]) >= 0.97       # teacher-forced top-1 tokens
    same_first = sum(int(a[0] == b[0]) for a, b in zip(bf16["gen"], st["gen"]))
    assert same_first >= 6, (same_first, bf16["gen"], st["gen"])
    assert all(len(g) == 32 for g in st["gen"])


@skip
def test_decode_runs_with_cuda_graphs():
    _, st = _engines()
    assert st["decode_b1_tps"] > 0 and st["decode_b8_tps"] > 0


@skip
def test_vllm_serve_cli():
    """vllm serve MODEL --representation statetree answers an OpenAI completion request."""
    log_path = os.path.join(HERE, "_out", "vllm_serve.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = VLLM_CLI + ["serve", MODEL, "--dtype", "bfloat16",
           "--representation", "statetree", "--port", str(PORT), "--gpu-memory-utilization", "0.45",
           "--max-model-len", "1024"]
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        url = f"http://127.0.0.1:{PORT}"
        t0 = time.time()
        while True:
            assert proc.poll() is None, open(log_path).read()[-4000:]
            try:
                urllib.request.urlopen(url + "/v1/models", timeout=5)
                break
            except OSError:
                assert time.time() - t0 < 900, "server did not start"
                time.sleep(5)
        req = urllib.request.Request(url + "/v1/completions", headers={"Content-Type": "application/json"},
                                     data=json.dumps(dict(model=MODEL, prompt="The history of mathematics",
                                                          max_tokens=16, temperature=0)).encode())
        body = json.load(urllib.request.urlopen(req, timeout=120))
        assert body["choices"][0]["text"].strip(), body
        assert "lossless state trees" in open(log_path).read()
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)
