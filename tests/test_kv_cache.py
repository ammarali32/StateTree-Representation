"""State-tree KV cache tests (CUDA, sm_80+): the page kernels bit for bit (append over recycled pages, as vLLM reuses
blocks; gathered page decode), decode attention against an fp32 reference, and the Hugging Face cache
(StateTreeKVCache + the 'statetree' attention) against DynamicCache + SDPA on a small random Qwen3 model, and CUDA-graph
decode (statetree.bench.StaticGraphDecoder) with StaticCache and StateTreeKVCache against eager steps.
    python -m pytest tests/test_kv_cache.py -q   or   python tests/test_kv_cache.py"""
import math

import torch

from statetree.ops import kv_ext

DEV = "cuda"
H, HQ, PAGE, D = 8, 32, 16, 128
LENS, PRE = [600, 333, 17, 513], [400, 333, 5, 500]       # tokens per sequence, of which prefilled (rest decoded)
MAXP = 40                                                  # pages per sequence in the block table
NB = len(LENS) * MAXP + 5                                  # physical pages, reassigned every round


def _data(g, n, tiny):
    x = torch.randn(2, n, H, D, device=DEV, generator=g)
    x = torch.where(torch.rand(x.shape, device=DEV, generator=g) < tiny, x * 2.0 ** -30, x)    # escapes
    x = torch.where(torch.rand(x.shape, device=DEV, generator=g) < 0.01, torch.zeros_like(x), x)
    return x.to(torch.bfloat16)


def _write(kx, cache, pool, cnt, g, tiny, reuse=True):
    """chunked prefill (pages split across calls), then batched decode steps with a padding slot -1"""
    bt = torch.randperm(NB, device=DEV, generator=g)[:len(LENS) * MAXP].view(len(LENS), MAXP).to(torch.int32)
    data = [_data(g, L, tiny) for L in LENS]

    def slots(s, pos):
        pos = torch.as_tensor(pos, device=DEV, dtype=torch.long)
        return bt[s].long()[pos // PAGE] * PAGE + pos % PAGE

    for s, P0 in enumerate(PRE):
        c = min(P0, (P0 * 3) // 5 + 3)
        for a, b in ((0, c), (c, P0)):
            if b > a:
                kx.append(data[s][0][a:b], data[s][1][a:b], slots(s, torch.arange(a, b)), cache, pool, cnt, reuse)
    pos = list(PRE)
    while any(p < L for p, L in zip(pos, LENS)):
        ks, vs, sl = [], [], []
        for s, L in enumerate(LENS):
            if pos[s] < L:
                ks.append(data[s][0][pos[s]:pos[s] + 1])
                vs.append(data[s][1][pos[s]:pos[s] + 1])
                sl.append(slots(s, [pos[s]]))
                pos[s] += 1
        ks.append(torch.zeros(1, H, D, dtype=torch.bfloat16, device=DEV))
        vs.append(torch.zeros(1, H, D, dtype=torch.bfloat16, device=DEV))
        sl.append(torch.tensor([-1], device=DEV))
        kx.append(torch.cat(ks), torch.cat(vs), torch.cat(sl), cache, pool, cnt, reuse)
    return bt, data


def _wrong(kx, cache, pool, bt, data):
    n = 0
    for s, L in enumerate(LENS):
        npg = (L + PAGE - 1) // PAGE
        Kd = torch.empty(npg, PAGE, H, D, dtype=torch.bfloat16, device=DEV)
        Vd = torch.empty_like(Kd)
        kx.dequant_blocks(cache, pool, bt[s][:npg].contiguous(), Kd, Vd)
        n += int((Kd.view(-1, H, D)[:L].view(torch.int16) != data[s][0].view(torch.int16)).sum())
        n += int((Vd.view(-1, H, D)[:L].view(torch.int16) != data[s][1].view(torch.int16)).sum())
    return n


def _fresh():
    cache = torch.zeros(NB, H, 2, 2, 1600, dtype=torch.uint8, device=DEV)      # vLLM's cache starts zeroed
    pool = torch.zeros(1 << 17, 16, dtype=torch.int32, device=DEV)
    cnt = torch.zeros(2, dtype=torch.int32, device=DEV)
    return cache, pool, cnt


def test_page_size():
    assert kv_ext().page_bytes(1) == 6400                                         # BF16: 2 x 16 x 128 x 2 = 8192


def test_pages_exact_with_recycling():
    kx = kv_ext()
    g = torch.Generator(device=DEV).manual_seed(0)
    cache, pool, cnt = _fresh()
    used = []
    for tiny in (0.05, 0.12, 0.02, 0.12):                                         # 5-13 % escapes: chains needed
        bt, data = _write(kx, cache, pool, cnt, g, tiny)
        assert _wrong(kx, cache, pool, bt, data) == 0
        assert int(cnt[1]) == 0                                                   # pool never exhausted
        used.append(int(cnt[0]))
    assert used[0] > 0
    assert used[-1] < 1.6 * used[1], used                                        # recycled chains, no leak


def test_dequant_blocks_out_of_range_is_zero():
    kx = kv_ext()
    g = torch.Generator(device=DEV).manual_seed(1)
    cache, pool, cnt = _fresh()
    _write(kx, cache, pool, cnt, g, 0.01)
    blocks = torch.tensor([-1, NB, 5], dtype=torch.int32, device=DEV)
    Kd = torch.full((3, PAGE, H, D), 7.0, dtype=torch.bfloat16, device=DEV)
    Vd = torch.full_like(Kd, 7.0)
    kx.dequant_blocks(cache, pool, blocks, Kd, Vd)
    assert int((Kd[:2] != 0).sum()) == 0 and int((Vd[:2] != 0).sum()) == 0
    K5, V5 = torch.empty_like(Kd[:1]), torch.empty_like(Vd[:1])
    kx.dequant_blocks(cache, pool, blocks[2:].contiguous(), K5, V5)
    assert torch.equal(Kd[2:], K5) and torch.equal(Vd[2:], V5)


def test_decode_attention_matches_reference():
    kx = kv_ext()
    g = torch.Generator(device=DEV).manual_seed(2)
    cache, pool, cnt = _fresh()
    bt, data = _write(kx, cache, pool, cnt, g, 0.02)
    lens_l = LENS + [0]                                                          # a padded / empty sequence
    B = len(lens_l)
    bt = torch.cat([bt, bt[:1]]).contiguous()
    lens = torch.tensor(lens_l, dtype=torch.int32, device=DEV)
    q = torch.randn(B, HQ, D, device=DEV, generator=g).to(torch.bfloat16)
    scale = 1.0 / math.sqrt(D)
    ref = torch.zeros(B, HQ, D, device=DEV)
    for b, L in enumerate(lens_l[:-1]):
        k = data[b][0].float().repeat_interleave(HQ // H, dim=1)
        v = data[b][1].float().repeat_interleave(HQ // H, dim=1)
        p = torch.softmax(torch.einsum("hd,lhd->hl", q[b].float(), k) * scale, dim=-1)
        ref[b] = torch.einsum("hl,lhd->hd", p, v)
    for S, minpps in ((1, 1), (8, 4), (32, 4)):                                  # fixed split counts (CUDA graphs)
        out = torch.empty(B, HQ, D, dtype=torch.bfloat16, device=DEV)
        wo = torch.empty(B * HQ * S * D, dtype=torch.float32, device=DEV)
        wml = torch.empty(B * HQ * S * 2, dtype=torch.float32, device=DEV)
        kx.attn(q, cache, pool, bt, lens, out, wo, wml, scale, S, None, minpps)
        err = (out[:-1].float() - ref[:-1]).abs().max().item() / ref[:-1].abs().max().item()
        assert err < 1e-2, (S, err)                                               # P is rounded to BF16 (as FA-2)
        assert int((out[-1] != 0).sum()) == 0                                     # length 0 -> 0


def test_hf_cache_matches_sdpa():
    try:
        from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM
    except ImportError:
        print("skip (transformers with Qwen3 not installed)")
        return
    from statetree.kvcache import StateTreeKVCache, enable
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=1024, hidden_size=1024, intermediate_size=2048, num_hidden_layers=2,
                      num_attention_heads=16, num_key_value_heads=4, head_dim=128, max_position_embeddings=4096)
    m = Qwen3ForCausalLM(cfg).to(DEV, torch.bfloat16).eval()
    P, N = 300, 32
    seq = torch.randint(0, 1024, (1, P + N), device=DEV)

    @torch.no_grad()
    def teacher_forced(cache):
        out = m(input_ids=seq[:, :P], past_key_values=cache, use_cache=True)
        logs = [out.logits[:, -1].float()]
        for i in range(N - 1):
            out = m(input_ids=seq[:, P + i:P + i + 1], past_key_values=cache, use_cache=True)
            logs.append(out.logits[:, -1].float())
        return torch.stack(logs, 1)

    m.set_attn_implementation("sdpa")
    ref = teacher_forced(DynamicCache(config=cfg))
    enable(m)
    c = StateTreeKVCache(cfg, batch_size=1, max_cache_len=P + N + 8)
    got = teacher_forced(c)
    c.check()
    rel = ((got - ref).norm(dim=-1) / ref.norm(dim=-1)).max().item()
    assert rel < 2e-2, rel
    assert (got.argmax(-1) == ref.argmax(-1)).float().mean().item() >= 0.9



def test_hf_graph_decode():
    """statetree.bench.StaticGraphDecoder (one greedy decode step in a CUDA graph) with transformers' StaticCache and
    with StateTreeKVCache: graph replays give the same tokens as eager steps, also when replaying straight after
    capture(), and each cache's GPU-side token count ends at prompt + steps (Transformers >= 5.20 take positions from
    it)"""
    try:
        from transformers import Qwen3Config, Qwen3ForCausalLM
    except ImportError:
        print("skip (transformers with Qwen3 not installed)")
        return
    from statetree import bench
    from statetree.kvcache import StateTreeKVCache, enable

    class TreeKV(bench.StaticGraphDecoder):
        def __init__(self, model, batch, max_len, graph=True):
            super().__init__(model, batch, max_len, graph)
            self.cache = StateTreeKVCache(model.config, batch_size=batch, max_cache_len=max_len)

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=1024, hidden_size=1024, intermediate_size=2048, num_hidden_layers=2,
                      num_attention_heads=16, num_key_value_heads=4, head_dim=128, max_position_embeddings=4096)
    m = Qwen3ForCausalLM(cfg).to(DEV, torch.bfloat16).eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.05)                                                # varied greedy tokens
    B, P, N = 2, 64, 24
    ids = torch.randint(0, 1024, (B, P), device=DEV, generator=torch.Generator(device=DEV).manual_seed(1))
    for cls, attn in ((bench.StaticGraphDecoder, "sdpa"), (TreeKV, "statetree")):
        if attn == "sdpa":
            m.set_attn_implementation("sdpa")
        else:
            enable(m)
        eager, _ = cls(m, B, P + N + 8, graph=False).generate(ids, N)
        r = cls(m, B, P + N + 8, graph=True)
        first = r.prefill(ids)
        r.capture()
        for _ in range(N - 1):                                                    # straight after capture
            r.graph.replay()
        torch.cuda.synchronize()
        direct = torch.cat([first, r.out[:, :N - 1]], 1)
        again, _ = r.generate(ids, N)                                             # prefill again, then replays
        assert torch.equal(direct, eager), (attn, direct, eager)
        assert torch.equal(again, eager), (attn, again, eager)
        lay = r.cache.layers[0]
        count = getattr(lay, "cum", None) if attn == "statetree" else getattr(lay, "cumulative_length", None)
        if isinstance(count, torch.Tensor):
            assert int(count) == P + N - 1, (attn, int(count))
        if attn == "statetree":
            assert r.cache.lens[:B].tolist() == [P + N - 1] * B
            r.cache.check()
        assert len(set(eager.flatten().tolist())) > 4, "degenerate greedy output: the test would not see errors"
        del r


def test_hf_generate():
    """model.generate with a StateTreeKVCache (as in the README), batch 2 with one left-padded row, against
    transformers' default cache + SDPA: generate runs and starts with the same tokens; then, teacher-forced on the
    reference tokens (positions as generate computes them), every step's logits of both rows are within the SDPA
    tolerance and the padded row's first valid position is taken from the mask. (Free-running greedy outputs drift
    apart after the first near-tie, so they are not compared token for token.)"""
    try:
        from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM
    except ImportError:
        print("skip (transformers with Qwen3 not installed)")
        return
    from statetree.kvcache import StateTreeKVCache, enable
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=1024, hidden_size=1024, intermediate_size=2048, num_hidden_layers=2,
                      num_attention_heads=16, num_key_value_heads=4, head_dim=128, max_position_embeddings=4096,
                      pad_token_id=0, eos_token_id=None, bos_token_id=None)
    m = Qwen3ForCausalLM(cfg).to(DEV, torch.bfloat16).eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.05)
    P, PAD, N = 48, 8, 16
    ids = torch.randint(1, 1024, (2, P), device=DEV, generator=torch.Generator(device=DEV).manual_seed(2))
    mask = torch.ones_like(ids)
    ids[1, :PAD], mask[1, :PAD] = 0, 0                                            # row 1: left padding

    @torch.no_grad()
    def forced(cache, toks):
        am, pos = mask.clone(), (mask.cumsum(-1) - 1).clamp(min=0)
        out = m(input_ids=ids, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True)
        logs = [out.logits[:, -1].float()]
        for i in range(N - 1):
            am = torch.cat([am, torch.ones(2, 1, dtype=am.dtype, device=DEV)], 1)
            pos = pos[:, -1:] + 1
            out = m(input_ids=toks[:, i:i + 1], attention_mask=am, position_ids=pos, past_key_values=cache,
                    use_cache=True)
            logs.append(out.logits[:, -1].float())
        return torch.stack(logs, 1)

    kw = dict(attention_mask=mask, max_new_tokens=N, min_new_tokens=N, do_sample=False)
    with torch.no_grad():
        m.set_attn_implementation("sdpa")
        ref_tok = m.generate(ids, **kw)[:, P:]
        ref = forced(DynamicCache(config=cfg), ref_tok)
        enable(m)
        c = StateTreeKVCache(cfg, batch_size=2, max_cache_len=P + N + 8)
        got_tok = m.generate(ids, past_key_values=c, **kw)[:, P:]
        c.check()
        assert got_tok.shape == (2, N) and torch.equal(got_tok[:, :4], ref_tok[:, :4]), (got_tok, ref_tok)
        c = StateTreeKVCache(cfg, batch_size=2, max_cache_len=P + N + 8)
        got = forced(c, ref_tok)
        c.check()
    assert c.starts[:2].tolist() == [0, PAD], c.starts[:2].tolist()
    rel = ((got - ref).norm(dim=-1) / ref.norm(dim=-1)).max().item()
    assert rel < 2e-2, rel
    agree = (got.argmax(-1) == ref.argmax(-1)).float().mean(1)
    assert agree.min().item() >= 0.85, agree.tolist()

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)
