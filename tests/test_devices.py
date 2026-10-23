"""Device placement: accelerate device maps (multi-GPU, CPU offload), convert() on a dispatched model, vocab-sized
tables at batches > 512 and inputs on another GPU. Multi-GPU tests skip on one GPU.

    python -m pytest tests/test_devices.py -q"""
import os

import pytest
import torch
import torch.nn as nn

import statetree
from statetree.linear import TreeEmbedding, TreeLinear

DEV = "cuda"
NGPU = torch.cuda.device_count()
two_gpus = pytest.mark.skipif(NGPU < 2, reason="needs 2 GPUs")
LAYERS = 6
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out")


def _tiny(tie=True, seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=LAYERS, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=1024, tie_word_embeddings=tie, max_position_embeddings=256)
    torch.set_default_dtype(torch.bfloat16)        # bf16 weights, fp32 RoPE tables (as from_pretrained / load)
    try:
        m = LlamaForCausalLM(cfg)
    finally:
        torch.set_default_dtype(torch.float32)
    return m.eval()


def _saved(tie):
    d = os.path.join(OUT, f"tiny_llama_dev_tie{int(tie)}")
    if not os.path.exists(os.path.join(d, "config.json")):
        m = _tiny(tie).to(DEV)
        statetree.convert(m, verbose=False)
        statetree.save_pretrained(m, d)
    return d


def _ids():
    return torch.randint(0, 1024, (2, 24), generator=torch.Generator().manual_seed(1)).to(DEV)


def _offload_map(gpu_layers=2, lm_head="cpu", embed=0):
    """first gpu_layers decoder layers on GPU 0, the rest on the CPU (offloaded)"""
    dm = {"model.embed_tokens": embed, "model.norm": 0, "model.rotary_emb": 0, "lm_head": lm_head}
    dm.update({f"model.layers.{i}": 0 if i < gpu_layers else "cpu" for i in range(LAYERS)})
    return dm


@torch.no_grad()
def _greedy(m, ids, n=8):
    return m.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=0)


@torch.no_grad()
@pytest.mark.parametrize("tie", [True, False])
def test_default_load_one_gpu(tie):
    d = _saved(tie)
    ref = statetree.load_pretrained(d, device=DEV)
    m = statetree.load_pretrained(d)                    # device_map="auto", as transformers' from_pretrained
    if NGPU == 1:                                       # one GPU, everything fits -> plain .to(), no hooks
        assert not hasattr(m, "hf_device_map")
    else:                                               # several GPUs: layers spread over them
        assert len(set(m.hf_device_map.values())) > 1
    assert all(t.device.type == "cuda" for t in m.state_dict().values())
    ids = _ids()
    assert torch.equal(m(ids).logits, ref(ids).logits)


@torch.no_grad()
@pytest.mark.parametrize("tie", [True, False])
def test_cpu_offload_explicit_map(tie):
    d = _saved(tie)
    ref = statetree.load_pretrained(d, device=DEV)
    ids = _ids()
    want, want_gen = ref(ids).logits, _greedy(ref, ids)
    del ref
    m = statetree.load_pretrained(d, device_map=_offload_map(lm_head="cpu", embed="cpu" if not tie else 0))
    hm = m.hf_device_map
    if tie:                                             # the tied input embedding reads lm_head's tree: kept on GPU
        assert hm["lm_head"] == 0
    assert hm["model.layers.5"] == "cpu"
    off = m.model.layers[5].mlp.gate_up_proj
    assert isinstance(off, TreeLinear) and off.tree.device.type == "meta"     # streamed in per forward
    assert m.model.layers[0].mlp.gate_up_proj.tree.device.type == "cuda"
    assert torch.equal(m(ids).logits, want)
    assert torch.equal(_greedy(m, ids), want_gen)
    assert off.tree.device.type == "meta"


@torch.no_grad()
def test_cpu_offload_max_memory():
    d = _saved(True)
    ref = statetree.load_pretrained(d, device=DEV)
    ids = _ids()
    want = ref(ids).logits
    del ref
    m = statetree.load_pretrained(d, device_map="sequential", max_memory={0: 2_000_000, "cpu": 8 << 30})
    assert "cpu" in m.hf_device_map.values()
    assert torch.equal(m(ids).logits, want)


@torch.no_grad()
def test_convert_dispatched_model():
    from accelerate import dispatch_model
    ids = _ids()
    m = _tiny(True)
    ref = _tiny(True).to(DEV)
    want = ref(ids).logits.float()
    statetree.convert(ref, exclude=[f"model.layers.{i}.*" for i in range(3, LAYERS)], verbose=False)
    want_tree = ref(ids).logits                         # the same layers converted, on one GPU
    del ref
    m = dispatch_model(m, device_map=_offload_map(gpu_layers=3, lm_head=0))
    assert torch.equal(m(ids).logits.float(), want)     # dispatch itself changes nothing
    statetree.convert(m, verbose=False)
    assert isinstance(m.model.layers[0].self_attn.qkv_proj, TreeLinear)
    assert isinstance(m.model.layers[2].mlp.gate_up_proj, TreeLinear)
    assert isinstance(m.model.layers[4].mlp.gate_proj, nn.Linear)       # offloaded (meta): stays BF16
    assert isinstance(m.lm_head, TreeLinear) and isinstance(m.model.embed_tokens, TreeEmbedding)
    out = m(ids).logits.float()
    assert float((out - want).norm() / want.norm()) < 2e-2
    assert torch.equal(m(ids).logits, want_tree)


@torch.no_grad()
def test_vocab_table_large_batch_chunks():
    g = torch.Generator(device=DEV).manual_seed(0)
    W = (torch.randn(4096, 256, device=DEV, generator=g) * 0.02).to(torch.bfloat16)
    tl = TreeLinear.from_weight(W)
    tl.allow_dec = False                               # as vLLM's lm_head / vocab-parallel tables
    x = torch.randn(1300, 256, device=DEV, dtype=torch.bfloat16)
    assert tl._plan(1300)[0] == "chunk"
    y = tl(x)
    parts = torch.cat([tl(x[i:i + 512]) for i in range(0, 1300, 512)])
    assert torch.equal(y, parts)
    ref = (x.float() @ W.float().t())
    assert float((y.float() - ref).norm() / ref.norm()) < 1e-2


@two_gpus
@torch.no_grad()
def test_inputs_from_another_gpu():
    g = torch.Generator(device="cuda:1").manual_seed(0)
    W = (torch.randn(512, 256, device="cuda:1", generator=g) * 0.02).to(torch.bfloat16)
    tl = TreeLinear.from_weight(W)
    x = torch.randn(7, 256, device="cuda:1", dtype=torch.bfloat16)
    y1 = tl(x)
    with torch.cuda.device(0):
        y0 = tl(x.to("cuda:0"))
    assert y0.device == torch.device("cuda:0") and torch.equal(y0.cpu(), y1.cpu())
    ids = torch.tensor([[0, 5, 511]], device="cuda:0")
    assert torch.equal(tl.rows(ids).cpu(), W[ids.to("cuda:1")].cpu())


@two_gpus
@torch.no_grad()
@pytest.mark.parametrize("tie", [True, False])
def test_multi_gpu_balanced(tie):
    d = _saved(tie)
    ref = statetree.load_pretrained(d, device=DEV)
    ids = _ids()
    want, want_gen = ref(ids).logits.cpu(), _greedy(ref, ids).cpu()
    del ref
    m = statetree.load_pretrained(d, device_map="balanced")
    assert len({v for v in m.hf_device_map.values() if v != "cpu"}) >= 2
    assert torch.equal(m(ids).logits.cpu(), want)
    assert torch.equal(_greedy(m, ids).cpu(), want_gen)


@two_gpus
@torch.no_grad()
def test_convert_multi_gpu_dispatched():
    from accelerate import dispatch_model
    ids = _ids()
    m = _tiny(True)
    dm = {"model.embed_tokens": 1, "model.norm": 1, "model.rotary_emb": 0, "lm_head": 1}   # tied pair: one GPU
    dm.update({f"model.layers.{i}": 0 if i < LAYERS // 2 else 1 for i in range(LAYERS)})
    m = dispatch_model(m, device_map=dm)
    want = m(ids).logits.float().cpu()
    statetree.convert(m, verbose=False)
    assert m.model.layers[-1].mlp.gate_up_proj.tree.device == torch.device("cuda:1")
    assert m.model.embed_tokens.source is m.lm_head
    out = m(ids).logits.float().cpu()
    assert float((out - want).norm() / want.norm()) < 2e-2


@two_gpus
@torch.no_grad()
def test_tied_table_on_another_gpu():
    """the input embedding (GPU 0) looks rows up in the tied lm_head tree that lives on GPU 1"""
    d = _saved(True)
    ref = statetree.load_pretrained(d, device=DEV)
    ids = _ids()
    want = ref(ids).logits.cpu()
    del ref
    dm = {"model.embed_tokens": 0, "model.norm": 1, "model.rotary_emb": 0, "lm_head": 1}
    dm.update({f"model.layers.{i}": 0 if i < LAYERS // 2 else 1 for i in range(LAYERS)})
    m = statetree.load_pretrained(d, device_map=dm)
    assert m.model.embed_tokens.source.tree.device == torch.device("cuda:1")
    assert torch.equal(m(ids).logits.cpu(), want)
