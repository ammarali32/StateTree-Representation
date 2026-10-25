"""State-tree KV cache for Hugging Face transformers.

Keys and values are stored losslessly in 16-token pages of state-tree records (csrc/kvcache.cu: 0.781x of BF16 incl.
the escape slots, every value bit-exact). Decode steps (one new token per sequence) run the state-tree decode
attention straight over the pages; prefill runs the model's SDPA attention on the prompt's own K/V and appends them.

    from statetree.kvcache import StateTreeKVCache, enable
    enable(model)                                   # registers the 'statetree' attention implementation
    cache = StateTreeKVCache(model.config, batch_size=B, max_cache_len=prompt + new)
    model.generate(input_ids, attention_mask=mask, past_key_values=cache, max_new_tokens=new)

The cache is CUDA-graph safe for decode steps (positions come from the cache_position tensor, buffers are static), so
it also works with statetree.bench.StaticGraphDecoder(cache_factory=...). Left padding is supported (each sequence's
first valid position is taken from the prefill attention mask).
"""
import threading

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from .ops import kv_ext

PAGE = 16
RKV = 1600                                          # bytes per (page, KV head, K/V, dim half)
_CUR = threading.local()                            # the layer whose update() ran last (read by the attention fn)


def _splits(B, Hkv, max_len):
    """splits of each sequence's pages over CTAs: ~256 CTAs (measured best on A100), >= 4 pages per split"""
    pages = (max_len + PAGE - 1) // PAGE
    return max(1, min(-(-256 // (B * Hkv)), pages // 4))


class StateTreeKVLayer(CacheLayerMixin):
    is_compileable = False
    is_sliding = False

    def __init__(self, cache, layer_idx):
        super().__init__()
        self.cache, self.layer_idx = cache, layer_idx
        self._len = 0                               # tokens stored (host count: eager generate / prefill path)
        self.max_cache_len = cache.max_len
        self.max_batch_size = cache.batch_size

    def lazy_initialization(self, key_states, value_states):
        c = self.cache
        B, H, _, D = key_states.shape
        assert D == 128, "StateTreeKVCache: head dim must be 128"
        assert value_states.shape[-1] == 128
        c._ensure(B, H, key_states.device)
        self.dtype, self.device = key_states.dtype, key_states.device
        self.max_batch_size, self.num_heads = c.Bmax, H
        self.pages = torch.empty(c.Bmax * c.max_pages, H, 2, 2, RKV, dtype=torch.uint8, device=self.device)
        self.pool = torch.zeros(c.pool_blocks, 16, dtype=torch.int32, device=self.device)
        self.pcnt = torch.zeros(2, dtype=torch.int32, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        c = self.cache
        B, H, T, D = key_states.shape
        cp = cache_kwargs.get("cache_position") if cache_kwargs else None
        if cp is None:
            cp = torch.arange(self._len, self._len + T, device=self.device)
        slots = (c.bt64[:B, :][:, cp // PAGE] * PAGE + (cp % PAGE)).reshape(-1)
        K = key_states.transpose(1, 2).reshape(B * T, H, D)
        V = value_states.transpose(1, 2).reshape(B * T, H, D)
        if K.stride(1) != D or K.stride(2) != 1:
            K = K.contiguous()
        if V.stride(1) != D or V.stride(2) != 1:
            V = V.contiguous()
        kv_ext().append(K, V, slots, self.pages, self.pool, self.pcnt)
        _CUR.layer = self
        prev = self._len
        self._len += T
        if T == 1:                                  # decode: the attention function reads the pages
            if self.layer_idx == 0:
                c.lens[:B].copy_((cp[-1:] + 1).to(torch.int32).expand(B))
            return key_states, value_states
        if prev == 0:                               # fresh prefill: attention over the prompt's own K/V
            return key_states, value_states
        Kd, Vd = self.dense(prev)                   # continued prefill: the stored prefix + the new tokens
        return torch.cat([Kd, key_states], 2), torch.cat([Vd, value_states], 2)

    def dense(self, n):
        """the first n stored tokens as BF16 [B, H, n, 128] (exact)"""
        c = self.cache
        B, H = c.Bmax, self.num_heads
        Kd = torch.empty(self.pages.shape[0], PAGE, H, 128, dtype=torch.bfloat16, device=self.device)
        Vd = torch.empty_like(Kd)
        kv_ext().dequant(self.pages, self.pool, Kd, Vd)
        pos = torch.arange(n, device=self.device)
        blk = c.bt64[:B][:, pos // PAGE]                                   # [B, n]
        k = Kd[blk, (pos % PAGE)[None].expand(B, n)]                       # [B, n, H, 128]
        v = Vd[blk, (pos % PAGE)[None].expand(B, n)]
        return k.transpose(1, 2), v.transpose(1, 2)

    def get_mask_sizes(self, cache_position):
        return self._len + cache_position.shape[0], 0

    def get_seq_length(self):
        return self._len

    def get_max_cache_shape(self):
        return self.max_cache_len

    def reset(self):
        self._len = 0
        if self.is_initialized:
            self.pcnt.zero_()
        self.cache.starts_set = False

    def check(self):
        """raise if an escape could not be stored (extension pool exhausted): never silently lossy"""
        if self.is_initialized and int(self.pcnt[1].item()):
            raise RuntimeError("StateTreeKVCache: escape pool exhausted (pool_blocks too small)")

    def reorder_cache(self, beam_idx):
        raise NotImplementedError("StateTreeKVCache: beam search is not supported")

    def crop(self, max_length):
        raise NotImplementedError("StateTreeKVCache: crop is not supported")


class StateTreeKVCache(Cache):
    """Lossless state-tree KV cache (see module doc). batch_size and max_cache_len fix the page tables; pool_blocks
    sizes the per-layer escape-extension pool (64 B blocks, used by < 10% of records of the worst layer)."""

    def __init__(self, config, batch_size=None, max_cache_len=4096, pool_blocks=None, **kwargs):
        tc = config.get_text_config() if hasattr(config, "get_text_config") else config
        self.num_layers = tc.num_hidden_layers
        self.num_q_heads = tc.num_attention_heads
        self.num_kv_heads = getattr(tc, "num_key_value_heads", None) or tc.num_attention_heads
        self.max_len = int(max_cache_len)
        self.max_pages = (self.max_len + PAGE - 1) // PAGE
        self.batch_size = batch_size
        self.pool_blocks = pool_blocks
        self.starts_set = False
        self._ready = False
        super().__init__(layers=[StateTreeKVLayer(self, i) for i in range(self.num_layers)])

    def _ensure(self, B, H, dev):
        if self._ready:
            assert B <= self.Bmax, "StateTreeKVCache: batch larger than allocated"
            return
        self.Bmax = max(B, self.batch_size or 0)
        Bm = self.Bmax
        self.bt64 = torch.arange(Bm * self.max_pages, device=dev).view(Bm, self.max_pages)      # page ids
        self.bt32 = self.bt64.to(torch.int32)
        self.lens = torch.zeros(Bm, dtype=torch.int32, device=dev)
        self.starts = torch.zeros(Bm, dtype=torch.int32, device=dev)
        self.S = _splits(Bm, H, self.max_len)
        self.wo = torch.empty(Bm * self.num_q_heads * self.S * 128, dtype=torch.float32, device=dev)
        self.wml = torch.empty(Bm * self.num_q_heads * self.S * 2, dtype=torch.float32, device=dev)
        if self.pool_blocks is None:
            self.pool_blocks = max(4096, Bm * self.max_pages * H * 4 // 8)
        self._ready = True

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def check(self):
        for lay in self.layers:
            lay.check()

    def storage_bytes(self):
        return sum(l.pages.numel() + l.pool.numel() * 4 for l in self.layers if l.is_initialized)


def statetree_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """'statetree' attention implementation: decode steps over the state-tree pages, everything else SDPA."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    lay = getattr(_CUR, "layer", None)
    _CUR.layer = None
    if lay is not None and lay.layer_idx == getattr(module, "layer_idx", lay.layer_idx):
        c = lay.cache
        B, Hq, T, D = query.shape
        if T == 1 and lay._len > 1:
            q = query[:, :, 0, :]
            if q.stride(-1) != 1 or q.stride(1) != D:
                q = q.contiguous()
            out = torch.empty(B, Hq, D, dtype=query.dtype, device=query.device)
            kv_ext().attn(q, lay.pages, lay.pool, c.bt32[:B], c.lens[:B], out, c.wo, c.wml,
                          float(scaling if scaling is not None else D ** -0.5), c.S, c.starts[:B])
            return out.view(B, 1, Hq, D), None
        if T > 1 and not c.starts_set and attention_mask is not None and attention_mask.dim() == 4:
            m = attention_mask[:, 0, -1, :]                             # last query row: only left padding masked
            pad = (~m).sum(-1) if m.dtype == torch.bool else (m < 0).sum(-1)
            c.starts[:B].copy_(pad.to(torch.int32))
            c.starts_set = True
    return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)


def enable(model):
    """register the 'statetree' attention implementation (decode over state-tree pages, SDPA otherwise) and switch
    the model to it; use with a StateTreeKVCache."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    AttentionInterface.register("statetree", statetree_attention)
    AttentionMaskInterface.register("statetree", sdpa_mask)
    model.set_attn_implementation("statetree") if hasattr(model, "set_attn_implementation") else \
        setattr(model.config, "_attn_implementation", "statetree")
    return model
