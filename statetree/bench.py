"""Throughput measurement for Hugging Face causal LMs.

DecodeRunner (Llama-style blocks) and StaticGraphDecoder (any model with transformers StaticCache support, e.g. the
hybrid Qwen3.5) capture one greedy decode step (all layers, lm_head, argmax, cache and position update) in a CUDA graph,
so the measured tokens/s reflect the GPU kernels rather than Python overhead. The same runners are used for the BF16
baseline (nn.Linear -> cuBLAS) and the converted model (TreeLinear). fuse_bf16 builds a fused BF16 baseline and
linear_chain_ms times the linear layers alone."""
import time

import torch
import torch.nn.functional as Fn


def _device(model):
    return next(t for t in list(model.parameters()) + list(model.buffers()) if t.device.type != "meta").device


def _rope_fn():
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    return apply_rotary_pos_emb


class DecodeRunner:
    def __init__(self, model, batch, max_len, graph=True):
        cfg = model.config
        self.model, self.m, self.lm_head = model, model.model, model.lm_head
        self.B, self.S = batch, max_len
        self.H, self.KVH = cfg.num_attention_heads, cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.rep = self.H // self.KVH
        self.scale = self.D ** -0.5
        dev = _device(model)
        L = len(self.m.layers)
        self.kc = torch.zeros(L, batch, self.KVH, max_len, self.D, device=dev, dtype=torch.bfloat16)
        self.vc = torch.zeros_like(self.kc)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.tok = torch.zeros(batch, 1, dtype=torch.long, device=dev)
        self.cnt = torch.zeros(1, dtype=torch.long, device=dev)
        self.out = torch.zeros(batch, max_len, dtype=torch.long, device=dev)
        self.ar = torch.arange(max_len, device=dev)
        self.rope = _rope_fn()
        self.use_graph = graph
        self.graph = None

    @torch.no_grad()
    def prefill(self, ids):
        """ids [B, T]: run the prompt, fill the cache, return the first generated token [B, 1]."""
        B, T = ids.shape
        assert B == self.B and T < self.S
        x = self.m.embed_tokens(ids)
        pid = torch.arange(T, device=ids.device)[None].expand(B, T)
        cos, sin = self.m.rotary_emb(x, pid)
        for l, layer in enumerate(self.m.layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_proj(h).view(B, T, self.H, self.D).transpose(1, 2)
            k = a.k_proj(h).view(B, T, self.KVH, self.D).transpose(1, 2)
            v = a.v_proj(h).view(B, T, self.KVH, self.D).transpose(1, 2)
            q, k = self.rope(q, k, cos, sin)
            self.kc[l, :, :, :T] = k
            self.vc[l, :, :, :T] = v
            kk = k[:, :, None].expand(B, self.KVH, self.rep, T, self.D).reshape(B, self.H, T, self.D)
            vv = v[:, :, None].expand(B, self.KVH, self.rep, T, self.D).reshape(B, self.H, T, self.D)
            o = Fn.scaled_dot_product_attention(q, kk, vv, is_causal=True)
            x = x + a.o_proj(o.transpose(1, 2).reshape(B, T, self.H * self.D))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        logits = self.lm_head(self.m.norm(x[:, -1:]))
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        self.tok.copy_(nxt)
        self.pos.fill_(T)
        self.cnt.zero_()
        return nxt

    def _step(self):
        B = self.B
        x = self.m.embed_tokens(self.tok)
        cos, sin = self.m.rotary_emb(x, self.pos.view(1, 1).expand(B, 1))
        valid = self.ar <= self.pos
        for l, layer in enumerate(self.m.layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_proj(h).view(B, 1, self.H, self.D).transpose(1, 2)
            k = a.k_proj(h).view(B, 1, self.KVH, self.D).transpose(1, 2)
            v = a.v_proj(h).view(B, 1, self.KVH, self.D).transpose(1, 2)
            q, k = self.rope(q, k, cos, sin)
            self.kc[l].index_copy_(2, self.pos, k)
            self.vc[l].index_copy_(2, self.pos, v)
            qg = q.reshape(B, self.KVH, self.rep, self.D)
            s = torch.matmul(qg, self.kc[l].transpose(-1, -2)).float() * self.scale       # [B, KVH, rep, S]
            p = s.masked_fill(~valid, float("-inf")).softmax(-1).to(torch.bfloat16)
            o = torch.matmul(p, self.vc[l]).reshape(B, 1, self.H * self.D)
            x = x + a.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        logits = self.lm_head(self.m.norm(x))
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        self.out.index_copy_(1, self.cnt, nxt)
        self.tok.copy_(nxt)
        self.pos += 1
        self.cnt += 1

    @torch.no_grad()
    def capture(self):
        """Warm up (resolves/tunes every layer's kernel for this batch) and capture one decode step."""
        save = (self.tok.clone(), self.pos.clone(), self.cnt.clone())
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.tok.copy_(save[0]); self.pos.copy_(save[1]); self.cnt.copy_(save[2])
        if self.use_graph:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._step()
            torch.cuda.synchronize()

    @torch.no_grad()
    def generate(self, ids, n_new):
        """Greedy: returns (tokens [B, n_new], decode seconds for the n_new - 1 graph steps)."""
        assert ids.shape[1] + n_new <= self.S
        first = self.prefill(ids)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_new - 1):
            if self.graph is not None:
                self.graph.replay()
            else:
                self._step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return torch.cat([first, self.out[:, :n_new - 1]], 1), dt


@torch.no_grad()
def decode_tps(model, ids, n_new=128, graph=True, max_len=None):
    B, T = ids.shape
    r = DecodeRunner(model, B, max_len or (T + n_new + 8), graph=graph)
    r.prefill(ids)
    r.capture()
    r.generate(ids, min(n_new, 8))                                            # warm
    toks, dt = r.generate(ids, n_new)
    del r
    return toks, B * (n_new - 1) / dt, dt / (n_new - 1)


@torch.no_grad()
def prefill_ms(model, B, T, iters=5, vocab=None):
    dev = _device(model)
    ids = torch.randint(0, vocab or model.config.vocab_size, (B, T), device=dev)
    r = DecodeRunner(model, B, T + 1, graph=False)
    r.prefill(ids)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        r.prefill(ids)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    del r
    return 1e3 * sorted(ts)[len(ts) // 2]


@torch.no_grad()
def hf_generate_tps(model, ids, n_new=32):
    """Plain 'model.generate' (eager, dynamic cache): what a drop-in user sees without CUDA graphs."""
    am = torch.ones_like(ids)
    kw = dict(max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False, attention_mask=am,
              pad_token_id=model.config.eos_token_id if isinstance(model.config.eos_token_id, int) else 0)
    model.generate(ids, **dict(kw, max_new_tokens=4, min_new_tokens=4))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(ids, **kw)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return out[:, ids.shape[1]:], ids.shape[0] * n_new / dt


# ----------------------------------------------------------------------------------------- any HF causal LM
class StaticGraphDecoder:
    """Greedy decoding through the model's own forward with a transformers StaticCache; one decode step (forward,
    argmax, cache and position update) is captured in a CUDA graph. Works for Hugging Face causal LMs whose cache
    supports StaticCache, including hybrid linear-attention models (Qwen3.5, Qwen3-Next)."""

    def __init__(self, model, batch, max_len, graph=True):
        from transformers import StaticCache
        dev = _device(model)
        self.model, self.B, self.S, self.use_graph = model, batch, max_len, graph
        self.cache = StaticCache(config=model.config, max_cache_len=max_len)
        self.tok = torch.zeros(batch, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.cnt = torch.zeros(1, dtype=torch.long, device=dev)
        self.out = torch.zeros(batch, max_len, dtype=torch.long, device=dev)
        self.graph = None
        self._used = False

    def _fwd(self, ids, cache_position):
        return self.model(input_ids=ids, past_key_values=self.cache, cache_position=cache_position, use_cache=True,
                          logits_to_keep=1).logits

    @torch.no_grad()
    def prefill(self, ids):
        B, T = ids.shape
        assert B == self.B and T < self.S
        if self._used:
            self.cache.reset()
        self._used = True
        logits = self._fwd(ids, torch.arange(T, device=ids.device))
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        self.tok.copy_(nxt)
        self.pos.fill_(T)
        self.cnt.zero_()
        return nxt

    def _step(self):
        logits = self._fwd(self.tok, self.pos)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        self.out.index_copy_(1, self.cnt, nxt)
        self.tok.copy_(nxt)
        self.pos += 1
        self.cnt += 1

    capture = DecodeRunner.capture
    generate = DecodeRunner.generate


@torch.no_grad()
def decode_tps_static(model, ids, n_new=128, graph=True):
    B, T = ids.shape
    r = StaticGraphDecoder(model, B, T + n_new + 8, graph=graph)
    r.prefill(ids)
    r.capture()
    r.generate(ids, min(n_new, 8))                                            # warm
    toks, dt = r.generate(ids, n_new)
    del r
    return toks, B * (n_new - 1) / dt, dt / (n_new - 1)


@torch.no_grad()
def prefill_ms_hf(model, B, T, iters=5):
    """time of one forward over a B x T prompt (no cache, logits of the last position only)"""
    ids = torch.randint(0, model.config.get_text_config().vocab_size, (B, T), device=_device(model))
    model(input_ids=ids, use_cache=False, logits_to_keep=1)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return 1e3 * sorted(ts)[len(ts) // 2]


class _FusedLinear(torch.nn.Linear):
    """BF16 counterpart of a fused TreeLinear (for a like-for-like baseline)."""
    _fcache = None

    def _fused_part(self, x, start, size, nparts):
        c = self._fcache
        if c is None or c[0] is not x:
            c = self._fcache = [x, self(x), 0]
        out = c[1].narrow(-1, start, size)
        c[2] += 1
        if c[2] >= nparts:
            self._fcache = None
        return out


@torch.no_grad()
def fuse_bf16(model, fuse=None):
    """Fuse sibling BF16 Linears exactly as statetree.convert does (same groups, same TreeSlice views), but keep them
    in BF16 / cuBLAS: the baseline that isolates the gain of the compressed format from the gain of fusion.
    statetree.convert(model) afterwards converts the fused layers and re-points the slices."""
    from .linear import TreeSlice
    from .model import DEFAULT_FUSE
    n = 0
    for _, parent in list(model.named_modules()):
        for fname, parts in (fuse or DEFAULT_FUSE).items():
            if hasattr(parent, fname) or not all(isinstance(getattr(parent, p, None), torch.nn.Linear) for p in parts):
                continue
            lins = [getattr(parent, p) for p in parts]
            if len({lin.in_features for lin in lins}) != 1 or any(lin.bias is not None for lin in lins):
                continue
            W = torch.cat([lin.weight for lin in lins], 0)
            f = _FusedLinear(W.shape[1], W.shape[0], bias=False, device=W.device, dtype=W.dtype)
            f.weight.copy_(W)
            del W
            setattr(parent, fname, f)
            start = 0
            for p, lin in zip(parts, lins):
                setattr(parent, p, TreeSlice(f, start, lin.out_features, len(parts)))
                start += lin.out_features
            n += 1
    return n


def linear_modules(model):
    """The weight-carrying matmul modules in execution order (fused layers once, slices skipped)."""
    from .linear import TreeEmbedding, TreeLinear, TreeSlice
    tables = {id(m.table) for m in model.modules() if isinstance(m, TreeEmbedding) and m.table is not None}
    out = []
    for name, m in model.named_modules():
        if isinstance(m, (TreeLinear, torch.nn.Linear)) and not isinstance(m, TreeSlice) and id(m) not in tables:
            out.append((name, m))
    return out


@torch.no_grad()
def linear_chain_ms(model, B, reps=50, include=None):
    """GPU time of all the model's linear layers run back to back in one CUDA graph (activations random)."""
    from .linear import TreeLinear
    mods = [m for n, m in linear_modules(model) if include is None or include(n)]
    xs = [torch.randn(B, m.in_features, device=_device(model), dtype=torch.bfloat16) for m in mods]
    for m in mods:
        if isinstance(m, TreeLinear):
            m.prepare([B])
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for m, x in zip(mods, xs):
                m(x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for m, x in zip(mods, xs):
            m(x)
    g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    del g
    return (time.perf_counter() - t0) / reps * 1e3
