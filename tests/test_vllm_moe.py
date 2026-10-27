"""vLLM with a Mixture-of-Experts model: --representation statetree stores every FusedMoE layer's routed experts as
state trees (StateTreeFusedMoEMethod). A tiny random Qwen2-MoE checkpoint (routed experts plus a gated shared expert,
which vLLM may run on a side CUDA stream) is written to tests/_out once, in the per-expert layout vLLM loads. Every
engine runs in its own process; the multi-GPU test skips with fewer than 2 GPUs.

Reference: the same statetree engine with the experts left in BF16 (STATETREE_VLLM_MOE=0), so the only difference is
the experts path (a random model's logits are nearly flat, so comparing with a plain BF16 engine would mostly measure
the dense layers' summation order). Scored teacher-forced (prompt logprobs of the same tokens in both engines), with
one prompt per request where the path must not depend on the batch (which requests share a step depends on when the
engine process receives them, and the experts path is chosen by the step's size):
    mvx   single 10-token prompts (40 (token, expert) pairs a step)
    mvg   single 50-token prompts (200 pairs, <= 8 per expert)
    mid   four 100-token prompts in one step (1600 pairs): the grouped tree GEMM gtt
    long  two 2000-token prompts, one per request, in 1024-token prefill chunks (about 4000 pairs): gtt
          only the experts' summation order differs, logprobs agree closely
A third engine runs with gtt off (STATETREE_MOE_GTT=0): mid runs mvg, and long decodes all experts into a BF16 scratch
and runs vLLM's own MoE kernel on it, bit-exact weights through the same kernel, so every logprob is identical.
Plus greedy decode at batch 1, 32 and 128 (CUDA graphs: mvx, mvg, gtt): along the common prefix every step's
logprob agrees.
Both engines run without autotuning or a tuning cache (STATETREE_AUTOTUNE=0, STATETREE_CACHE=0), so the dense
state-tree layers use the same kernel plans in both (with a shared cache, tensor-parallel ranks tuning at the same
time may settle on different split-K plans, a summation-order difference that would hide the bit-exact check).

    python -m pytest tests/test_vllm_moe.py -q"""
import importlib.util
import json
import os
import random
import subprocess
import sys

os.environ.setdefault("STATETREE_MIN_PARAMS", "0")   # the tests expect every linear layer as a tree (small ones too)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "_out", "tiny_qwen2_moe_vllm")
HAVE_VLLM = importlib.util.find_spec("vllm") is not None
VOCAB, HID, LAYERS, E, TOPK, MOE_I, SHARED_I = 2048, 512, 4, 32, 4, 128, 256
LONG = 2000


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
except ImportError:                                                         # plain: python tests/test_vllm_moe.py
    def skip(f):
        return f

    def two_gpus(f):
        return f if _ngpu() >= 2 else (lambda: print("skip (needs 2 GPUs)", f.__name__))


def make_model(path=MODEL_DIR):
    """random Qwen2-MoE checkpoint (decisive routers, so few near-ties between experts)"""
    if os.path.exists(os.path.join(path, "model.safetensors")):
        return path
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen2MoeConfig
    os.makedirs(path, exist_ok=True)
    cfg = Qwen2MoeConfig(vocab_size=VOCAB, hidden_size=HID, intermediate_size=1024, num_hidden_layers=LAYERS,
                         num_attention_heads=8, num_key_value_heads=4, moe_intermediate_size=MOE_I,
                         shared_expert_intermediate_size=SHARED_I, num_experts=E, num_experts_per_tok=TOPK,
                         decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=False,
                         max_position_embeddings=4096, tie_word_embeddings=False, use_sliding_window=False)
    cfg.architectures = ["Qwen2MoeForCausalLM"]
    g = torch.Generator().manual_seed(0)

    def rnd(*shape, std=0.05):
        return (torch.randn(*shape, generator=g) * std).to(torch.bfloat16)

    def ones(n):
        return torch.ones(n, dtype=torch.bfloat16)
    D = HID // 8
    sd = {"model.embed_tokens.weight": rnd(VOCAB, HID), "model.norm.weight": ones(HID), "lm_head.weight": rnd(VOCAB, HID)}
    for layer in range(LAYERS):
        p = f"model.layers.{layer}."
        sd[p + "input_layernorm.weight"], sd[p + "post_attention_layernorm.weight"] = ones(HID), ones(HID)
        for n, rows in (("q", 8 * D), ("k", 4 * D), ("v", 4 * D)):
            sd[f"{p}self_attn.{n}_proj.weight"] = rnd(rows, HID)
            sd[f"{p}self_attn.{n}_proj.bias"] = rnd(rows, std=0.01)
        sd[p + "self_attn.o_proj.weight"] = rnd(HID, 8 * D)
        sd[p + "mlp.gate.weight"] = rnd(E, HID, std=0.5)
        for e in range(E):
            sd[f"{p}mlp.experts.{e}.gate_proj.weight"] = rnd(MOE_I, HID)
            sd[f"{p}mlp.experts.{e}.up_proj.weight"] = rnd(MOE_I, HID)
            sd[f"{p}mlp.experts.{e}.down_proj.weight"] = rnd(HID, MOE_I)
        sd[p + "mlp.shared_expert.gate_proj.weight"] = rnd(SHARED_I, HID)
        sd[p + "mlp.shared_expert.up_proj.weight"] = rnd(SHARED_I, HID)
        sd[p + "mlp.shared_expert.down_proj.weight"] = rnd(HID, SHARED_I)
        sd[p + "mlp.shared_expert_gate.weight"] = rnd(1, HID)
    save_file(sd, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    cfg.save_pretrained(path)
    return path


def _inspect(worker):
    """per rank: FusedMoE layers whose experts are state trees / still BF16, and whether the direct kernels are on"""
    import torch
    trees = bf16 = 0
    direct, gtt, shared = [], [], 0
    for _, mod in worker.model_runner.model.named_modules():
        if getattr(mod, "shared_experts", None) is not None:
            shared += 1
        info = getattr(mod, "statetree_moe", None)
        if isinstance(info, dict):
            trees += 1
            direct.append(bool(info.get("mvx")))
            gtt.append(bool(info.get("gtt")))
            continue
        w = getattr(mod, "w13_weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 3 and w.numel() > 0:
            bf16 += 1
    return dict(moe_trees=trees, moe_bf16=bf16, direct_kernels=direct, gtt=gtt, with_shared_experts=shared)


def worker(spec):
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ["STATETREE_VLLM_MOE"] = "1" if spec.get("moe", True) else "0"
    os.environ["STATETREE_MOE_GTT"] = "1" if spec.get("gtt", True) else "0"
    os.environ["STATETREE_AUTOTUNE"], os.environ["STATETREE_CACHE"] = "0", "0"       # the same plans in both engines
    import statetree.vllm_plugin  # noqa: F401
    from vllm import LLM, SamplingParams
    tp = spec.get("tp", 1)
    llm = LLM(model=spec["model"], dtype="bfloat16", representation=spec["representation"], seed=0,
              tensor_parallel_size=tp, gpu_memory_utilization=0.3, max_model_len=2048, max_num_batched_tokens=1024,
              skip_tokenizer_init=True, enable_prefix_caching=False,
              distributed_executor_backend="mp" if tp > 1 else None)
    out = {"ranks": llm.collective_rpc(_inspect)}
    rng = random.Random(0)
    ids = [rng.randrange(VOCAB) for _ in range(24000)]
    forced = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1, detokenize=False)

    def score(prompts):
        res = []
        for r in llm.generate([{"prompt_token_ids": p} for p in prompts], forced, use_tqdm=False):
            p = r.prompt_token_ids
            res.append([[d[p[i]].logprob, next(t for t, lp in d.items() if lp.rank == 1)]
                        for i, d in enumerate(r.prompt_logprobs) if d])
        return res
    out["mvx"] = [score([ids[40 * i:40 * i + 10]])[0] for i in range(8)]          # 10 tokens x 4 = 40 pairs a step
    out["mvg"] = [score([ids[1000 + 60 * i:1000 + 60 * i + 50]])[0] for i in range(4)]  # 200 pairs a step
    out["mid"] = score([ids[2000 + 150 * i:2000 + 150 * i + 100] for i in range(4)])   # 400 tokens, 1600 pairs
    out["long"] = [score([ids[s:s + LONG]])[0] for s in (5000, 8000)]            # chunks of 1024 + 976 tokens: > 2048 pairs
    greedy = SamplingParams(max_tokens=24, temperature=0, ignore_eos=True, detokenize=False, logprobs=5)

    def gen(prompts):
        return [[[t, {str(k): v.logprob for k, v in lp.items()}] for t, lp in zip(o.token_ids, o.logprobs)]
                for o in (r.outputs[0] for r in llm.generate([{"prompt_token_ids": p} for p in prompts], greedy,
                                                            use_tqdm=False))]
    out["gen1"] = gen([ids[12000:12064]])                                          # decode at batch 1: mvx
    out["gen32"] = gen([ids[13000 + 70 * i:13000 + 70 * i + 64] for i in range(32)])   # batch 32: 128 pairs, mvg
    out["gen128"] = gen([ids[16000 + 40 * i:16000 + 40 * i + 32] for i in range(128)])  # batch 128: 512 pairs, gtt
    print("RESULT " + json.dumps(out), flush=True)


def run(spec):
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", json.dumps(spec)],
                       capture_output=True, text=True)
    tag = (f"{spec['representation']}{'' if spec.get('moe', True) else '_expertsbf16'}"
           f"{'' if spec.get('gtt', True) else '_nogtt'}_tp{spec.get('tp', 1)}")
    with open(os.path.join(HERE, "_out", f"vllm_moe_{tag}.log"), "w") as fh:
        fh.write(p.stdout + "\n" + p.stderr)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError(f"engine {tag} failed:\n{p.stdout[-3000:]}\n{p.stderr[-6000:]}")


_CACHE = {}


def engine(moe, tp=1, gtt=True):
    """statetree engine; moe=False keeps the experts in BF16 (the reference), gtt=False turns the grouped tree GEMM
    off (large batches: decode-all + vLLM's MoE kernel)"""
    if (moe, tp, gtt) not in _CACHE:
        _CACHE[(moe, tp, gtt)] = run(dict(model=make_model(), representation="statetree", moe=moe, tp=tp, gtt=gtt))
    return _CACHE[(moe, tp, gtt)]


def _forced(st, ref, key):
    """mean / max |logprob difference| of the scored tokens, and top-1 agreement"""
    d, m, agree, n = 0.0, 0.0, 0, 0
    for a, b in zip(st[key], ref[key]):
        assert len(a) == len(b) > 0
        for (la, ta), (lb, tb) in zip(a, b):
            d += abs(la - lb)
            m = max(m, abs(la - lb))
            agree += int(ta == tb)
            n += 1
    return d / n, m, agree / n


def _check(st, ref, nogtt):
    for eng, on in ((st, True), (nogtt, False)):
        for r in eng["ranks"]:
            assert r["moe_trees"] == LAYERS and r["moe_bf16"] == 0, eng["ranks"]  # every MoE layer, every rank
            assert all(r["direct_kernels"]), eng["ranks"]                         # mvx / mvg / gtt in use
            assert r["gtt"] == [on] * LAYERS, eng["ranks"]                        # STATETREE_MOE_GTT honoured
    for r in ref["ranks"]:
        assert r["moe_trees"] == 0 and r["moe_bf16"] == LAYERS, ref["ranks"]      # STATETREE_VLLM_MOE=0 honoured
    stats = {key: _forced(st, ref, key) for key in ("mvx", "mvg", "mid", "long")}
    stats.update({"nogtt_" + key: _forced(nogtt, ref, key) for key in ("mid", "long")})
    print("teacher-forced (mean |dlogprob|, max, top-1 agreement):", stats)
    assert stats["nogtt_long"][1] == 0.0 and stats["nogtt_long"][2] == 1.0, stats["nogtt_long"]   # decode-all:
    for key in ("mvx", "mvg", "mid", "long", "nogtt_mid"):    # bit-exact weights, same kernel. The others differ in
        mean, mx, agree = stats[key]    # summation order only: ~0.01 nats on this nearly flat random model; a wrong
        assert mean < 0.05 and mx < 0.5 and agree >= 0.85, (key, stats[key])     # expert, shard or race costs nats
    diffs, prefix = [], []
    for a, b in zip(st["gen1"] + st["gen32"] + st["gen128"], ref["gen1"] + ref["gen32"] + ref["gen128"]):
        j = 0
        while j < len(a) and a[j][0] == b[j][0]:
            diffs.append(abs(a[j][1][str(a[j][0])] - b[j][1][str(a[j][0])]))
            j += 1
        prefix.append(j)
    print("greedy decode: common prefix", prefix, "mean / max |dlogprob| along it",
          sum(diffs) / max(len(diffs), 1), max(diffs, default=0.0))
    assert sum(p >= 1 for p in prefix) >= 0.75 * len(prefix), prefix              # the engines decode alike
    assert sum(diffs) / len(diffs) < 0.05 and max(diffs) < 0.5, (sum(diffs) / len(diffs), max(diffs))


@skip
def test_moe_experts_as_trees_one_gpu():
    _check(engine(True), engine(False), engine(True, gtt=False))


@two_gpus
def test_moe_experts_as_trees_tensor_parallel_2():
    st, ref = engine(True, tp=2), engine(False, tp=2)
    assert len(st["ranks"]) == 2
    _check(st, ref, engine(True, tp=2, gtt=False))

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(json.loads(sys.argv[2]))
    else:
        for name, fn in list(globals().items()):
            if name.startswith("test_"):
                fn()
                print("ok", name, flush=True)
