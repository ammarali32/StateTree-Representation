"""Kernel and format tests (CUDA, sm_80+).  python -m pytest tests -q   or   python tests/test_kernels.py"""
import os

import torch
import torch.nn as nn

import statetree
from statetree import format as F
from statetree.linear import TreeEmbedding, TreeLinear, candidates, _state

DEV = "cuda"


def weights(M, N, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    W = (torch.randn(M, N, device=DEV, generator=g) * 0.02).to(torch.bfloat16)
    W[3, 5] = 0.0                                                           # exact zeros -> escapes
    W[17, 70:90] = 1e-9
    W[40, 128] = 3.0                                                        # outlier: its group's small weights escape
    W[100:104, 0:64:2] = 0.0                                                # a dense escape cluster in one unit
    return W


def test_format_roundtrip():
    W = weights(256, 512)
    enc = F.encode(W)
    assert enc["esc_rows"].numel() > 0
    back = F.decode_reference(enc["tree"], enc["esc_rows"], enc["esc_cols"], enc["esc_bits"])
    assert torch.equal(back.view(torch.int16), W.view(torch.int16))


def test_kernel_decode_exact():
    W = weights(512, 1024, 1)
    tl = TreeLinear.from_linear(_lin(W))
    assert torch.equal(tl.dequantize().view(torch.int16), W.view(torch.int16))


def _lin(W, bias=None):
    lin = nn.Linear(W.shape[1], W.shape[0], bias=bias is not None, device=DEV, dtype=torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_(W)
        if bias is not None:
            lin.bias.copy_(bias)
    return lin


def test_every_path_matches_reference():
    torch.manual_seed(0)
    for M, N in ((512, 1024), (1024, 512), (256, 2048)):
        W = weights(M, N, M + N)
        tl = TreeLinear.from_linear(_lin(W))
        st = _state(torch.device(DEV))
        for B in (1, 3, 8, 12, 16, 24, 32, 48, 64, 100, 128, 200, 256, 512, 600):
            x = torch.randn(B, N, device=DEV, dtype=torch.bfloat16)
            ref = x.double() @ W.double().t()
            tried = 0
            for path, prm in candidates(M, N, B, st) + ([("dec", None)] if B > 512 else []):
                if not tl._valid(path, prm, B):
                    continue
                y = torch.empty(B, M, device=DEV, dtype=torch.bfloat16)
                tl._run(x, y, (path, prm))
                err = float((y.double() - ref).norm() / ref.norm())
                assert err < 5e-3, (M, N, B, path, prm, err)
                for _ in range(3):                                          # bit-identical run to run
                    y2 = torch.empty_like(y)
                    tl._run(x, y2, (path, prm))
                    assert torch.equal(y2, y), (M, N, B, path, prm, "nondeterministic")
                tried += 1
            assert tried > 0
            y = tl(x.view(1, B, N))                                         # dispatcher, 3-D input
            assert y.shape == (1, B, M)
            assert float((y[0].double() - ref).norm() / ref.norm()) < 5e-3


def test_one_hot_exact():
    """x = one-hot rows: y must reproduce columns of W bit-exactly on the fused paths."""
    M, N = 512, 1024
    W = weights(M, N, 7)
    tl = TreeLinear.from_linear(_lin(W))
    st = _state(torch.device(DEV))
    B = 64
    x = torch.zeros(B, N, device=DEV, dtype=torch.bfloat16)
    x[torch.arange(B), torch.arange(B) * 7] = 1
    for path, prm in candidates(M, N, B, st):
        if path == "tm" and tl._valid(path, prm, B):
            y = torch.empty(B, M, device=DEV, dtype=torch.bfloat16)
            tl._run(x, y, (path, prm))
            assert torch.equal(y.t().contiguous().view(torch.int16), W[:, torch.arange(B) * 7].contiguous()
                               .view(torch.int16)), (path, prm)


def test_one_hot_exact_mv():
    """the matrix-vector paths (batch <= 32, every mv configuration) reproduce columns of W bit-exactly, escapes
    included: the fixture's outliers and zeros, and a matrix with a fifth of its weights escaping (hundreds of entries
    per record: the per-warp escape image)"""
    M, N = 256, 1024
    st = _state(torch.device(DEV))
    W1 = weights(M, N, 11)
    W2 = weights(M, N, 12)
    W2[torch.rand(M, N, device=DEV) < 0.2] *= 1e-6
    for W in (W1, W2):
        tl = TreeLinear.from_linear(_lin(W))
        for B in (1, 8, 32):
            x = torch.zeros(B, N, device=DEV, dtype=torch.bfloat16)
            cols = (torch.arange(B, device=DEV) * 37 + 5) % N
            x[torch.arange(B), cols] = 1
            want = W[:, cols].t().contiguous()
            n = 0
            for path, prm in candidates(M, N, B, st):
                if path == "mv" and tl._valid(path, prm, B):
                    y = torch.empty(B, M, device=DEV, dtype=torch.bfloat16)
                    tl._run(x, y, (path, prm))
                    assert torch.equal(y, want), (B, prm)
                    n += 1
            assert n > 0, B

def test_convert_save_load():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(256, 512, bias=True), nn.GELU(), nn.Linear(512, 256), nn.Linear(256, 10)) \
        .to(DEV, torch.bfloat16)
    x = torch.randn(5, 256, device=DEV, dtype=torch.bfloat16)
    ref = net(x)
    statetree.convert(net, exclude=(), verbose=False)
    assert isinstance(net[0], TreeLinear) and isinstance(net[3], nn.Linear)       # 10 outputs: not % 16
    y = net(x)
    assert float((y.float() - ref.float()).norm() / ref.float().norm()) < 1e-2
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out")
    os.makedirs(d, exist_ok=True)
    from safetensors.torch import load_file, save_file
    save_file({k: v.contiguous() for k, v in net.state_dict().items()}, os.path.join(d, "m.safetensors"))
    net2 = nn.Sequential(TreeLinear(256, 512, bias=True, device=DEV), nn.GELU(),
                         TreeLinear(512, 256, bias=True, device=DEV), nn.Linear(256, 10)).to(DEV, torch.bfloat16)
    net2.load_state_dict(load_file(os.path.join(d, "m.safetensors"), device=DEV))
    assert torch.equal(net2(x), y)


def test_rows_embedding():
    W = weights(512, 256, 3)
    tl = TreeLinear.from_weight(W)
    ids = torch.tensor([[0, 5, 17, 100], [511, 40, 3, 101]], device=DEV)
    assert torch.equal(tl.rows(ids).view(torch.int16), W[ids].view(torch.int16))


def test_hf_llama_fused_tied_save_load():
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        return
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=1024, tie_word_embeddings=True, max_position_embeddings=256)
    torch.set_default_dtype(torch.bfloat16)        # bf16 weights, fp32 RoPE tables (as from_pretrained / load)
    try:
        m = LlamaForCausalLM(cfg)
    finally:
        torch.set_default_dtype(torch.float32)
    m = m.to(DEV).eval()
    tied = m.lm_head.weight is m.model.embed_tokens.weight
    ids = torch.randint(0, 1024, (2, 24), device=DEV)
    with torch.no_grad():
        ref = m(ids).logits.float()
        statetree.convert(m, verbose=False)
        assert isinstance(m.model.layers[0].self_attn.qkv_proj, TreeLinear)
        assert isinstance(m.model.layers[0].mlp.gate_up_proj, TreeLinear)
        assert isinstance(m.lm_head, TreeLinear) and isinstance(m.model.embed_tokens, TreeEmbedding)
        assert (m.model.embed_tokens.source is m.lm_head) == tied
        out = m(ids).logits.float()
        assert float((out - ref).norm() / ref.norm()) < 2e-2
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out", "tiny_llama")
        statetree.save_pretrained(m, d)
        m2 = statetree.load_pretrained(d, device=DEV)
        assert torch.equal(m2(ids).logits, m(ids).logits)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
