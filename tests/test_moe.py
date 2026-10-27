"""MoE experts as state trees: Transformers 5 experts modules with 3-D parameters (Qwen3-MoE, Mixtral, and gpt-oss with
its transposed layout, biases and own gating), on tiny random models; the mvx / mvg / gtt / wsum kernels. Multi-GPU
tests skip on one GPU.

    python -m pytest tests/test_moe.py -q"""
import copy
import json
import os

import pytest
import torch

import statetree
from statetree import TreeExperts, TreeLinear
import statetree.experts as SE
from statetree.experts import (expert_tables, moe_align, moe_fused_decode, moe_gemv, moe_gemv_w, moe_group_gemv,
                               moe_group_gemv_w, moe_gtt, moe_wsum, stacked_gtt_tables)

DEV = "cuda"
NGPU = torch.cuda.device_count()
two_gpus = pytest.mark.skipif(NGPU < 2, reason="needs 2 GPUs")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out")
LAYERS = 3
KINDS = ["qwen3_moe", "mixtral", "gpt_oss"]


def _model(kind, seed=0):
    import transformers as T
    torch.manual_seed(seed)
    common = dict(hidden_size=256, num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=2,
                  vocab_size=1024, max_position_embeddings=512)
    try:
        if kind == "qwen3_moe":
            cfg = T.Qwen3MoeConfig(intermediate_size=512, moe_intermediate_size=128, head_dim=64, num_experts=16,
                                   num_experts_per_tok=4, decoder_sparse_step=1, mlp_only_layers=[], **common)
            cls = T.Qwen3MoeForCausalLM
        elif kind == "mixtral":
            cfg = T.MixtralConfig(intermediate_size=128, num_local_experts=8, num_experts_per_tok=2, **common)
            cls = T.MixtralForCausalLM
        else:
            cfg = T.GptOssConfig(intermediate_size=128, head_dim=64, num_local_experts=8, num_experts_per_tok=2,
                                 layer_types=["full_attention"] * LAYERS, **common)
            cls = T.GptOssForCausalLM
    except (AttributeError, TypeError) as e:
        pytest.skip(f"{kind} not available in this transformers: {e}")
    torch.set_default_dtype(torch.bfloat16)
    try:
        m = cls(cfg)
    finally:
        torch.set_default_dtype(torch.float32)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.05)
            if n.endswith("gate.weight") or n.endswith("router.weight"):
                p.normal_(0, 0.5)                              # decisive routers: few near-ties between experts
    return m.to(DEV).eval()


def _experts(m):
    return [(n, mod) for n, mod in m.named_modules() if type(mod).__name__.endswith("Experts")]


def _ids(n=40):
    return torch.randint(0, 1024, (2, n), generator=torch.Generator().manual_seed(1)).to(DEV)


def _mat(P, e, is_t):
    return P[e].t().contiguous() if is_t else P[e]


@torch.no_grad()
@pytest.mark.parametrize("kind", KINDS)
def test_every_expert_tensor_becomes_trees(kind):
    m = _model(kind)
    srcs = _experts(m)
    assert srcs, "no experts module"
    keep = {}
    for n, mod in srcs:
        is_t = bool(getattr(mod, "is_transposed", False))
        up = "gate_up_proj" if "gate_up_proj" in mod._parameters else "up_proj"
        for e in (0, mod._parameters[up].shape[0] - 1):
            keep[(n, up, e)] = _mat(mod._parameters[up], e, is_t).clone()
            keep[(n, "down_proj", e)] = _mat(mod._parameters["down_proj"], e, is_t).clone()
    statetree.convert(m, verbose=False)
    assert not [n for n, p in m.named_parameters() if p.dim() == 3], "3-D expert tensors left"
    for n, _ in srcs:
        assert isinstance(m.get_submodule(n), TreeExperts)
    for (n, which, e), W in keep.items():
        got = getattr(m.get_submodule(n), which)[e].dequantize()
        assert torch.equal(got.view(torch.int16), W.view(torch.int16)), (n, which, e)


@torch.no_grad()
@pytest.mark.parametrize("kind", KINDS)
def test_tree_experts_match_the_bf16_module(kind, monkeypatch):
    m = _model(kind)
    name, src = _experts(m)[0]
    ref = copy.deepcopy(src)
    statetree.convert(m, verbose=False)
    te = m.get_submodule(name)
    E, H = te.num_experts, 256
    K = 4 if kind == "qwen3_moe" else 2
    g = torch.Generator(device=DEV).manual_seed(2)
    calls = []
    real_gtt = SE.moe_gtt
    monkeypatch.setattr(SE, "moe_gtt", lambda *a: calls.append(a[-1]) or real_gtt(*a))
    fcalls = []
    real_fused = SE.moe_fused_decode
    monkeypatch.setattr(SE, "moe_fused_decode", lambda *a: fcalls.append(1) or real_fused(*a))
    wcalls = []
    real_w = SE.moe_group_gemv_w
    monkeypatch.setattr(SE, "moe_group_gemv_w", lambda *a: wcalls.append(1) or real_w(*a))
    for T, path in ((3, "fused"), (3, "mvx"), (200, "mvg"), (12, "mvw"), (200, "gtt"), (37, "gtt"), (200, "loop")):
        x = torch.randn(T, H, device=DEV, dtype=torch.bfloat16, generator=g)
        tk = torch.stack([torch.randperm(E, device=DEV, generator=g)[:K] for _ in range(T)])
        tw = torch.softmax(torch.randn(T, K, device=DEV, generator=g), -1).to(torch.bfloat16)
        monkeypatch.setenv("STATETREE_MOE_MVX_MAX", "64" if path == "mvx" else "0")
        monkeypatch.setenv("STATETREE_MOE_MVG_MAX", "0" if path == "loop" else str(64 * E))
        monkeypatch.setenv("STATETREE_MOE_GTT", "1" if path == "gtt" else "0")
        monkeypatch.setenv("STATETREE_MOE_GTT_MIN", "1")
        monkeypatch.setenv("STATETREE_MOE_FUSED_MAX", "64" if path == "fused" else "0")
        if path == "fused":
            monkeypatch.setenv("STATETREE_MOE_MVX_MAX", "64")
        n, nf, nw = len(calls), len(fcalls), len(wcalls)
        y, yr = te(x, tk, tw).float(), ref(x, tk, tw).float()
        assert len(wcalls) == nw + (1 if path == "mvw" else 0), (kind, path)    # down projection: mvwg (NT = 1)
        assert len(calls) == n + (2 if path == "gtt" else 0), (kind, path, calls)   # both projections
        assert len(fcalls) == nf + (1 if path == "fused" and kind != "gpt_oss" else 0), (kind, path)  # gpt-oss: own gate
        rel = ((y - yr).norm() / yr.norm()).item()
        assert rel < 1e-2, (kind, path, rel)


@torch.no_grad()
@pytest.mark.parametrize("kind", KINDS)
def test_model_outputs_close_to_bf16(kind):
    m = _model(kind)
    ids = _ids()
    want = m(ids).logits.float()
    statetree.convert(m, verbose=False)
    got = m(ids).logits.float()
    agree = (got.argmax(-1) == want.argmax(-1)).float().mean().item()
    assert agree >= 0.8, (kind, agree)


@torch.no_grad()
@pytest.mark.parametrize("kind", ["qwen3_moe", "gpt_oss"])
def test_save_load_round_trip(kind):
    m = _model(kind)
    statetree.convert(m, verbose=False)
    d = os.path.join(OUT, f"tiny_moe_{kind}")
    statetree.save_pretrained(m, d)
    info = json.load(open(os.path.join(d, "statetree.json")))
    assert len(info["experts"]) == LAYERS and info["architecture"] == type(m).__name__
    m2 = statetree.load_pretrained(d, device=DEV)
    assert type(m2) is type(m) and isinstance(_experts(m2)[0][1], TreeExperts) or any(
        isinstance(x, TreeExperts) for x in m2.modules())
    ids = _ids()
    assert torch.equal(m2(ids).logits, m(ids).logits)


@torch.no_grad()
@two_gpus
def test_two_gpus_equal_one_gpu():
    m = _model("qwen3_moe")
    statetree.convert(m, verbose=False)
    d = os.path.join(OUT, "tiny_moe_qwen3_moe_2gpu")
    statetree.save_pretrained(m, d)
    ids = _ids(160)                                  # 320 tokens: the tt prefill kernel runs on both GPUs
    want = m(ids).logits
    del m
    dm = {"model.embed_tokens": 0, "model.layers.0": 0, "model.layers.1": 1, "model.layers.2": 1, "model.norm": 1,
          "model.rotary_emb": 1, "lm_head": 1}
    m2 = statetree.load_pretrained(d, device_map=dm)
    assert {t.device.index for t in m2.buffers() if t.is_cuda} == {0, 1}
    assert torch.equal(m2(ids).logits.to(want.device), want)


@torch.no_grad()
def test_escape_heavy_tensors_stay_bf16():
    """tensors whose tree would be larger than BF16 (here: 30% of the weights replaced by a tiny 'fake zero', as in
    a few expert tensors of Qwen3.6-35B-A3B) stay BF16; everything else is converted and the model still runs"""
    m = _model("qwen3_moe")
    name, src = _experts(m)[0]
    lin_name = "model.layers.1.self_attn.o_proj"
    tiny = torch.tensor(7.8e-38, dtype=torch.bfloat16, device=DEV)
    for P in (src._parameters["down_proj"], m.get_submodule(lin_name).weight):
        mask = torch.rand(P.shape, device=DEV) < 0.3
        P.data[mask] = tiny
    ids = _ids()
    want = m(ids).logits.float()
    statetree.convert(m, verbose=False)
    assert not isinstance(m.get_submodule(name), TreeExperts)                 # kept as the BF16 experts module
    assert isinstance(m.get_submodule(lin_name), torch.nn.Linear)
    assert sum(isinstance(x, TreeExperts) for x in m.modules()) == LAYERS - 1
    got = m(ids).logits.float()
    assert (got.argmax(-1) == want.argmax(-1)).float().mean().item() >= 0.8


@torch.no_grad()
def test_mvx_mvg_kernels():
    torch.manual_seed(0)
    E, M, N = 32, 256, 512
    Ws = [(torch.randn(M, N, device=DEV) * 0.05).to(torch.bfloat16) for _ in range(E)]
    for W in Ws:
        W[torch.rand(M, N, device=DEV) < 0.003] *= 1e-6           # escapes
    mods = [TreeLinear.from_weight(W) for W in Ws]
    tabs = expert_tables(mods)
    Wall = torch.stack(Ws).float()
    for S, K in ((1, 1), (8, 4), (64, 8), (300, 4)):
        ids = torch.randint(0, E, (S,), device=DEV)
        x = torch.randn(S // K, N, device=DEV, dtype=torch.bfloat16)
        ref = torch.einsum("sn,smn->sm", x.repeat_interleave(K, 0).float(), Wall[ids])
        outs = {"mvx": moe_gemv(tabs, x, ids, M, K), "mvx_int32": moe_gemv(tabs, x, ids.int(), M, K)}
        for NT in (1, 2, 4):
            al = moe_align(ids, E, 8 * NT)
            sid = al[0][al[0] < S]
            assert torch.equal(sid.sort().values, torch.arange(S, device=DEV, dtype=torch.int32))   # every pair once
            outs[f"mvg{NT}"] = moe_group_gemv(tabs, x, al, S, M, K, NT, E)
        bad = ids.int().clone()                         # ids outside [0, E) (vLLM's dummy batches): clamped in-kernel
        bad[0] = E + 5
        bad[-1] = -2 if S > 1 else bad[-1]
        clamped = bad.clamp(0, E - 1)
        assert torch.equal(moe_gemv(tabs, x, bad, M, K), moe_gemv(tabs, x, clamped, M, K)), S
        for cfg in ((1, 4, 1, 2), (1, 2, 1, 4)):
            assert torch.equal(moe_gemv(tabs, x, bad, M, K, cfg), moe_gemv(tabs, x, clamped, M, K, cfg)), (S, cfg)
        for k, y in outs.items():
            rel = ((y.float() - ref).norm() / ref.norm()).item()
            assert rel < 4e-3, (S, k, rel)
    T, K, Hd = 5, 3, 64
    y = torch.randn(T * K, Hd, device=DEV, dtype=torch.bfloat16)
    for w in (torch.rand(T, K, device=DEV), torch.rand(T, K, device=DEV).to(torch.bfloat16)):
        got = moe_wsum(y, w).float()
        want = (y.view(T, K, Hd).float() * w.float().unsqueeze(-1)).sum(1)
        assert (got - want).abs().max().item() < 1e-2



@torch.no_grad()
def test_escape_patch_and_small_k_kernels_one_hot_exact():
    """one-hot inputs read single weights back: every MoE product kernel (mvx, mvg, the persistent small-K mvwx / mvwg)
    must return W[e][:, j] bit for bit, escapes included (exact zeros, tiny values, an expert with a fifth of its
    weights escaping: hundreds of entries per record, the per-warp escape image); mvwx / mvwg equal mvx / mvg with one
    warp per tile (same accumulation order) on random inputs; ids outside [0, E) clamp in mvwx"""
    torch.manual_seed(0)
    E, M = 12, 256
    for N in (256, 512):
        Ws = [(torch.randn(M, N, device=DEV) * 0.05).to(torch.bfloat16) for _ in range(E)]
        for W in Ws:
            W[torch.rand(M, N, device=DEV) < 0.003] *= 1e-6           # escapes
            W[torch.rand(M, N, device=DEV) < 0.001] = 0.0              # exact zeros
        Ws[3][torch.rand(M, N, device=DEV) < 0.2] *= 1e-6              # an escape-heavy expert
        Ws[5][:, :64][torch.rand(M, 64, device=DEV) < 0.5] = 0.0       # dense escape clusters
        mods = [TreeLinear.from_weight(W) for W in Ws]
        assert max(m.esc_rows.numel() for m in mods) > 0.15 * M * N
        tabs = expert_tables(mods)
        for S in (1, 8, 37, 200):
            ids = torch.randint(0, E, (S,), device=DEV)
            ids[: S // 3] = 3                                          # the heavy expert, several chunks
            cols = torch.randint(0, N, (S,), device=DEV)
            x = torch.zeros(S, N, device=DEV, dtype=torch.bfloat16)
            x[torch.arange(S, device=DEV), cols] = 1.0
            want = torch.stack([Ws[int(e)][:, int(c)] for e, c in zip(ids.tolist(), cols.tolist())])
            al8 = moe_align(ids, E, 8)
            outs = {"mvx": moe_gemv(tabs, x, ids, M, 1), "mvx_wk1": moe_gemv(tabs, x, ids, M, 1, (1, 1, 1, 2)),
                    "mvg1": moe_group_gemv(tabs, x, al8, S, M, 1, 1, E),
                    "mvg2": moe_group_gemv(tabs, x, moe_align(ids, E, 16), S, M, 1, 2, E)}
            for nb in (1, 4):
                outs[f"mvwx{nb}"] = moe_gemv_w(tabs, x, ids.int(), M, 1, nb)
                outs[f"mvwg{nb}"] = moe_group_gemv_w(tabs, x, al8, S, M, 1, nb)
            for k, y in outs.items():
                assert torch.equal(y, want), (N, S, k, (y.float() - want.float()).abs().max().item())
            xr = torch.randn(S, N, device=DEV, dtype=torch.bfloat16)
            assert torch.equal(moe_gemv_w(tabs, xr, ids, M, 1, 2), moe_gemv(tabs, xr, ids, M, 1, (1, 1, 1, 2))), (N, S)
            assert torch.equal(moe_group_gemv_w(tabs, xr, al8, S, M, 1, 3),
                               moe_group_gemv(tabs, xr, al8, S, M, 1, 1, E, (1, 1, 1))), (N, S)
            bad = ids.int().clone()
            bad[0] = E + 5
            assert torch.equal(moe_gemv_w(tabs, xr, bad, M, 1), moe_gemv_w(tabs, xr, bad.clamp(0, E - 1), M, 1)), S


@torch.no_grad()
def test_gtt_kernel():
    """the grouped tree GEMM against fp32 for every compiled tile: escapes (one expert with a fifth of its weights
    escaping, so its entry lists outgrow shared memory for some tiles), experts with no pairs and with several blocks,
    padded blocks, both input modes (one row per token with xdiv = k, one row per pair), the slot list as vLLM cuts
    it (not a whole number of blocks), repeat runs bit-identical"""
    torch.manual_seed(0)
    E, M, N = 12, 256, 320
    Ws = [(torch.randn(M, N, device=DEV) * 0.05).to(torch.bfloat16) for _ in range(E)]
    for W in Ws:
        W[torch.rand(M, N, device=DEV) < 0.003] *= 1e-6           # escapes
    Ws[3][torch.rand(M, N, device=DEV) < 0.2] *= 1e-6              # an escape-heavy expert
    mods = [TreeLinear.from_weight(W) for W in Ws]
    tp = expert_tables(mods)[0]
    tt = stacked_gtt_tables(mods)
    assert tt[3] >= 4096                                           # CAP: expert 3's blocks hold ~8000 entries each
    assert tt[1].numel() == sum(m.esc_rows.numel() for m in mods) + 4          # entries back to back, no padding
    Wall = torch.stack(Ws).float()
    cfgs = [(32, 2, 2), (64, 2, 2), (64, 4, 2), (128, 2, 1), (128, 4, 1)]
    for S, K in ((5, 1), (96, 4), (1000, 4), (777, 1)):
        ids = torch.randint(0, E - 1, (S,), device=DEV)            # expert E - 1 gets no pairs
        ids[:S // 3] = 7                                           # one expert with several blocks
        ids[S // 3:S // 3 + min(S, 50)] = 3
        x = torch.randn(-(-S // K), N, device=DEV, dtype=torch.bfloat16)
        ref = torch.einsum("sn,smn->sm", x.repeat_interleave(K, 0)[:S].float(), Wall[ids])
        for cfg in cfgs:
            al = moe_align(ids, E, cfg[0])
            y = moe_gtt(tp, tt, x, al, S, M, K, cfg)
            rel = ((y.float() - ref).norm() / ref.norm()).item()
            assert rel < 4e-3, (S, K, cfg, rel)
            assert torch.equal(moe_gtt(tp, tt, x, al, S, M, K, cfg), y), (S, cfg)       # deterministic
            n = S + E * (cfg[0] - 1)                               # vLLM's moe_align_block_size length
            if n < al[0].numel():
                assert torch.equal(moe_gtt(tp, tt, x, (al[0][:n].clone(), al[1], al[2]), S, M, K, cfg), y)


@torch.no_grad()
def test_fused_decode_kernel():
    """the fused MoE decode kernel (gate-up, silu * up, down and the routing-weighted sum in one launch) against fp32 and
    the unfused kernels; int32 / int64 ids, ids outside [0, E) clamped, k up to 8, repeat calls bit-identical (its
    counters reset themselves), escapes in both projections"""
    torch.manual_seed(0)
    E, H, I = 16, 256, 128
    W13 = [(torch.randn(2 * I, H, device=DEV) * 0.05).to(torch.bfloat16) for _ in range(E)]
    W2 = [(torch.randn(H, I, device=DEV) * 0.05).to(torch.bfloat16) for _ in range(E)]
    for W in W13 + W2:
        W[torch.rand(W.shape, device=DEV) < 0.003] *= 1e-6           # escapes
    t13 = expert_tables([TreeLinear.from_weight(W) for W in W13])
    t2 = expert_tables([TreeLinear.from_weight(W) for W in W2])
    A13, A2 = torch.stack(W13).float(), torch.stack(W2).float()
    for T, K in ((1, 8), (3, 4), (8, 8), (5, 2)):
        ids = torch.stack([torch.randperm(E, device=DEV)[:K] for _ in range(T)]).int()
        w = torch.softmax(torch.randn(T, K, device=DEV), -1)
        x = torch.randn(T, H, device=DEV, dtype=torch.bfloat16)
        e = ids.reshape(-1).long()
        h = torch.einsum("sh,soh->so", x.float().repeat_interleave(K, 0), A13[e])
        a = torch.nn.functional.silu(h[:, :I]) * h[:, I:]
        ref = (torch.einsum("si,shi->sh", a, A2[e]).view(T, K, H) * w.unsqueeze(-1)).sum(1)
        got = moe_fused_decode(t13, t2, x, ids, w, I)
        rel = ((got.float() - ref).norm() / ref.norm()).item()
        assert rel < 1e-2, (T, K, rel)
        hu = moe_gemv(t13, x, ids.reshape(-1), 2 * I, K)              # the unfused kernels
        au = torch.empty(T * K, I, device=DEV, dtype=torch.bfloat16)
        su = torch.nn.functional.silu(hu[:, :I].float()).to(torch.bfloat16)
        au.copy_((su.float() * hu[:, I:].float()).to(torch.bfloat16))
        unf = moe_wsum(moe_gemv(t2, au, ids.reshape(-1), H, 1), w).float()
        assert ((got.float() - unf).norm() / unf.norm()).item() < 1e-2, (T, K)
        assert torch.equal(moe_fused_decode(t13, t2, x, ids, w, I), got), (T, K)           # repeat: counters reset
        assert torch.equal(moe_fused_decode(t13, t2, x, ids.long(), w, I), got), (T, K)    # int64 ids
        bad = ids.clone()
        bad[0, 0] = E + 3
        assert torch.equal(moe_fused_decode(t13, t2, x, bad, w, I),
                           moe_fused_decode(t13, t2, x, bad.clamp(0, E - 1), w, I)), (T, K)

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
