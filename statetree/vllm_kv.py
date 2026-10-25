"""vLLM (0.32) attention backend that stores the KV cache losslessly in the state-tree page format (csrc/kvcache.cu).

Enable with the statetree plugin flag '--kv-representation statetree' (Python: LLM(..., kv_representation="statetree")).
The flag selects vLLM's CUSTOM attention backend, which the plugin registers as StateTreeKVBackend, and block size 16.
It composes with '--representation statetree' (tree weights), prefix caching, chunked prefill and CUDA graphs.

Page: one vLLM block (16 tokens) per KV head = 4 records (K / V x two 64-dim halves) of 1600 B = 6400 B, against 8192 B
in BF16. The backend declares 400 B per token per head (AttentionSpec.state_content_bytes), so vLLM's block manager fits
1.28x more tokens into the same memory. Every value decodes to its exact BF16 bits.
  write    do_kv_cache_update -> append(reuse=True) at vLLM's slot mapping (a recycled block recycles its escape
           extension chain, so the shared extension pool does not leak)
  decode   one query token per request: attention straight from the pages (records decode into mma fragments),
           split-K with a split count fixed per batch size, so vLLM captures it in full CUDA graphs
  prefill  fresh prompts: FlashAttention on their own K/V; requests with cached context (prefix caching, chunked
           prefill): their pages are decoded (exactly) into a BF16 scratch that FlashAttention reads paged
Escapes beyond a record's 11 slots go to one extension pool per process (STATETREE_KV_POOL_MB, default 128). Exhaustion
is flagged by the kernel and raises at the next step: never silently lossy.
"""
import os
from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import (AttentionBackend, AttentionCGSupport, AttentionImpl, AttentionMetadata,
                                       AttentionMetadataBuilder, AttentionType, CommonAttentionMetadata)
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, get_flash_attn_version
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_layout import KVCacheLayout

from .ops import kv_ext

logger = init_logger("vllm.statetree.kv")

PAGE, RKV, HD = 16, 1600, 128
TARGET_CTAS = 256              # decode split heuristic from the Stage A sweep: ~256 CTAs ...
MIN_PAGES_PER_SPLIT = 4        # ... and >= 4 pages per split (one per warp)
MAX_SPLITS = 32


def decode_splits(num_decodes, num_kv_heads):
    """split count of a decode batch: a function of the (padded) batch size only, so the CUDA graph of a batch size
    serves every context length (the kernel keeps >= MIN_PAGES_PER_SPLIT pages per split; idle splits exit)"""
    if num_decodes <= 0:
        return 1
    s = TARGET_CTAS // (num_decodes * num_kv_heads)
    if s <= 1:
        return 1
    return 1 << (min(s, MAX_SPLITS).bit_length() - 1)


class _Shared:
    """per process: the escape extension pool (all layers), the split-K workspace and the pool watch"""

    def __init__(self, device, num_q_heads, num_kv_heads):
        mb = float(os.environ.get("STATETREE_KV_POOL_MB", "128"))
        nblk = min(max(1024, int(mb * 2 ** 20) // 64), (1 << 24) - 1)
        self.device = device
        self.pool = torch.zeros(nblk, 16, dtype=torch.int32, device=device)
        self.cnt = torch.zeros(2, dtype=torch.int32, device=device)         # [0] blocks used, [1] exhausted flag
        self.host = torch.zeros(2, dtype=torch.int32, device="cpu").pin_memory()   # (model init runs under a cuda default device)
        self.num_q_heads, self.num_kv_heads = num_q_heads, num_kv_heads
        rows = max(TARGET_CTAS // num_kv_heads, MAX_SPLITS)                  # >= max over batch sizes of B * splits
        self.wo = torch.empty(rows * num_q_heads * HD, dtype=torch.float32, device=device)
        self.wml = torch.empty(rows * num_q_heads * 2, dtype=torch.float32, device=device)
        self.steps, self.warned = 0, False
        logger.info("statetree KV cache: lossless state-tree pages (6400 B per 16 tokens per KV head, BF16 8192 B); "
                    "extension pool %d blocks (%.0f MB)", nblk, nblk * 64 / 2 ** 20)

    def poll(self):
        """once per step (metadata builder, host): reads the pool counters copied at the previous step (no sync)"""
        if torch.cuda.is_current_stream_capturing():
            return
        used, err = int(self.host[0]), int(self.host[1])
        if err:
            raise RuntimeError(f"statetree KV: the escape extension pool ({self.pool.shape[0]} blocks) is exhausted; "
                               "restart with a larger STATETREE_KV_POOL_MB")
        self.steps += 1
        if not self.warned and used > 0.5 * self.pool.shape[0]:
            self.warned = True
            logger.warning("statetree KV: extension pool %.0f%% used (%d blocks)", 100 * used / self.pool.shape[0], used)
        elif self.steps % 5000 == 0:
            logger.info("statetree KV: extension pool %d / %d blocks used", used, self.pool.shape[0])
        self.host.copy_(self.cnt, non_blocking=True)


_SHARED = {}


def _shared(device, num_q_heads, num_kv_heads):
    key = (device.type, device.index)
    sh = _SHARED.get(key)
    if sh is None:
        sh = _SHARED[key] = _Shared(device, num_q_heads, num_kv_heads)
    elif (sh.num_q_heads, sh.num_kv_heads) != (num_q_heads, num_kv_heads):
        raise NotImplementedError("statetree KV: layers with different head counts in one model")
    return sh


def pool_usage():
    """(blocks used, blocks, exhausted) of this process's extension pool (synchronises; for tests)"""
    out = {}
    for k, sh in _SHARED.items():
        c = sh.cnt.tolist()
        out[k] = (c[0], sh.pool.shape[0], bool(c[1]))
    return out


def _pages(kv_cache, num_kv_heads):
    """vLLM's layer cache [blocks, H, 16, C] (LBHNC, C = 400 B) -> the kernels' view uint8 [blocks, H, 2, 2, 1600]"""
    c = kv_cache if kv_cache.dtype == torch.uint8 else kv_cache.view(torch.uint8)
    if not c.is_contiguous() or c.numel() != c.shape[0] * num_kv_heads * 4 * RKV:
        raise RuntimeError(f"statetree KV: unexpected layer cache {tuple(kv_cache.shape)} / strides {kv_cache.stride()}")
    return c.view(c.shape[0], num_kv_heads, 2, 2, RKV)


# ----------------------------------------------------------------------------------------------------- backend
class StateTreeKVBackend(AttentionBackend):
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list] = ["auto", "bfloat16"]

    @classmethod
    def customize_spec(cls, spec):
        if spec.state_content_bytes is not None:
            return spec
        return replace(spec, state_content_bytes=4 * RKV // PAGE)        # 400 B per token per KV head

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @classmethod
    def supported_kv_cache_layouts(cls):
        return (KVCacheLayout.LBHNC,)

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [PAGE]

    @classmethod
    def get_supported_head_sizes(cls):
        return [HD]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return capability.major >= 8

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls():
        return StateTreeKVImpl

    @staticmethod
    def get_builder_cls():
        return StateTreeKVMetadataBuilder


@dataclass
class StateTreeKVMetadata(AttentionMetadata):
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    seq_lens: torch.Tensor           # [reqs] int32 (GPU)
    block_table: torch.Tensor        # [reqs, max blocks] int32
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    splits: int = 1                  # decode split count
    # prefill requests (eager steps only)
    pf_fresh: bool = False           # no cached context: FlashAttention on the new K/V
    pf_max_q: int = 0
    pf_max_k: int = 0
    pf_qsl: torch.Tensor | None = None        # cu_seqlens of the prefill queries, from 0 (int32, GPU)
    pf_seq_lens: torch.Tensor | None = None   # [prefills] int32 (GPU)
    pf_blocks: torch.Tensor | None = None     # cache block of scratch page i (int32, GPU)
    pf_bt: torch.Tensor | None = None         # scratch block table [prefills, max pages] (int32, GPU)


class StateTreeKVMetadataBuilder(AttentionMetadataBuilder[StateTreeKVMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        self.num_kv_heads = kv_cache_spec.num_kv_heads

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self.build(0, common_attn_metadata)

    def build(self, common_prefix_len, common_attn_metadata: CommonAttentionMetadata, fast_build=False):
        cam = common_attn_metadata
        nd, npf, ndt, _ = split_decodes_and_prefills(cam, decode_threshold=self.reorder_batch_threshold)
        for sh in _SHARED.values():
            sh.poll()
        md = StateTreeKVMetadata(
            num_actual_tokens=cam.num_actual_tokens, num_decodes=nd, num_decode_tokens=ndt, num_prefills=npf,
            seq_lens=cam.seq_lens, block_table=cam.block_table_tensor, slot_mapping=cam.slot_mapping,
            query_start_loc=cam.query_start_loc, splits=decode_splits(min(nd, ndt), self.num_kv_heads))
        if npf > 0:
            self._prefill_tables(md, cam, nd, ndt, npf)
        return md

    def _prefill_tables(self, md, cam, nd, ndt, npf):
        qsl = cam.query_start_loc_cpu[nd:nd + npf + 1].to(torch.int64)
        ql = qsl[1:] - qsl[:-1]
        sl_cpu = cam.seq_lens_cpu_upper_bound                 # exact for prefill rows
        sl = (sl_cpu if sl_cpu is not None else cam.seq_lens.cpu())[nd:nd + npf].to(torch.int64)
        md.pf_max_q = int(ql.max())
        md.pf_qsl = cam.query_start_loc[nd:nd + npf + 1] - ndt
        md.pf_seq_lens = cam.seq_lens[nd:nd + npf]
        md.pf_fresh = bool(torch.equal(ql, sl))
        if md.pf_fresh:
            return
        npg = (sl + PAGE - 1) // PAGE                         # pages of each prefill request (cached + new tokens)
        tot = int(npg.sum())
        off = torch.cumsum(npg, 0) - npg
        cpu = torch.device("cpu")
        req = torch.repeat_interleave(torch.arange(npf, device=cpu), npg)
        pg = torch.arange(tot, device=cpu) - torch.repeat_interleave(off, npg)
        bt = torch.zeros(npf, int(npg.max()), dtype=torch.int32, device=cpu)
        bt[req, pg] = torch.arange(tot, dtype=torch.int32, device=cpu)
        dev = cam.block_table_tensor.device
        md.pf_blocks = cam.block_table_tensor[(req + nd).to(dev), pg.to(dev)].to(torch.int32).contiguous()
        md.pf_bt = bt.to(dev)
        md.pf_max_k = int(sl.max())


# ----------------------------------------------------------------------------------------------------- impl
class StateTreeKVImpl(AttentionImpl[StateTreeKVMetadata]):
    supports_quant_query_input: bool = False

    def __init__(self, num_heads, head_size, scale, num_kv_heads=None, alibi_slopes=None, sliding_window=None,
                 kv_cache_dtype="auto", logits_soft_cap=None, attn_type=AttentionType.DECODER,
                 kv_sharing_target_layer_name=None, **kwargs):
        if head_size != HD:
            raise NotImplementedError(f"statetree KV: head size {head_size} (supported: {HD})")
        if alibi_slopes is not None or sliding_window is not None or logits_soft_cap:
            raise NotImplementedError("statetree KV: ALiBi, sliding windows and soft-capping are not supported")
        if attn_type != AttentionType.DECODER or kv_sharing_target_layer_name is not None:
            raise NotImplementedError("statetree KV: decoder self-attention with its own cache only")
        if kwargs.get("sinks") is not None:
            raise NotImplementedError("statetree KV: attention sinks are not supported")
        self.num_heads, self.head_size, self.scale = num_heads, head_size, float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if num_heads % self.num_kv_heads or num_heads // self.num_kv_heads not in (1, 2, 4, 8):
            raise NotImplementedError("statetree KV: query heads per KV head must be 1, 2, 4 or 8")
        self.kv_cache_dtype = kv_cache_dtype
        self.kx = kv_ext()
        self.sh = _shared(torch.device("cuda", torch.cuda.current_device()), num_heads, self.num_kv_heads)
        try:
            self.fa_version = get_flash_attn_version(head_size=HD)
        except Exception:                                     # noqa: BLE001
            self.fa_version = None
        self._view = (None, None)

    def _layer_pages(self, kv_cache):
        key = (kv_cache.data_ptr(), tuple(kv_cache.shape))
        if self._view[0] != key:
            self._view = (key, _pages(kv_cache, self.num_kv_heads))
        return self._view[1]

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        n = slot_mapping.shape[0]
        if n <= 0 or kv_cache.numel() == 0:
            return
        k = key[:n].view(n, self.num_kv_heads, HD)
        v = value[:n].view(n, self.num_kv_heads, HD)
        self.kx.append(k, v, slot_mapping[:n], self._layer_pages(kv_cache), self.sh.pool, self.sh.cnt, True)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None, output_scale=None,
                output_block_scale=None):
        assert output is not None, "statetree KV: the output buffer is required"
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("statetree KV: fused output quantization is not supported")
        md = attn_metadata
        if md is None or kv_cache.numel() == 0:              # profiling run
            return output.fill_(0)
        n = md.num_actual_tokens
        q = query[:n].view(n, self.num_heads, HD)
        out = output[:n].view(n, self.num_heads, HD)
        pages = self._layer_pages(kv_cache)
        nb = min(md.num_decodes, md.num_decode_tokens)
        if nb > 0:
            self.kx.attn(q[:nb], pages, self.sh.pool, md.block_table[:nb], md.seq_lens[:nb], out[:nb], self.sh.wo,
                         self.sh.wml, self.scale, md.splits, None, MIN_PAGES_PER_SPLIT)
        if md.num_prefills > 0:
            t0 = md.num_decode_tokens
            self._prefill(md, q[t0:n], key, value, t0, n, pages, out[t0:n])
        return output

    def _prefill(self, md, q, key, value, t0, t1, pages, out):
        hkv = self.num_kv_heads
        if md.pf_fresh:
            k = key[t0:t1].view(t1 - t0, hkv, HD)
            v = value[t0:t1].view(t1 - t0, hkv, HD)
            self._fa(q, k, v, md.pf_qsl, md.pf_max_q, md.pf_max_q, out, cu_seqlens_k=md.pf_qsl)
            return
        n = md.pf_blocks.numel()
        Kd = torch.empty(n, PAGE, hkv, HD, dtype=q.dtype, device=q.device)
        Vd = torch.empty_like(Kd)
        self.kx.dequant_blocks(pages, self.sh.pool, md.pf_blocks, Kd, Vd)
        self._fa(q, Kd, Vd, md.pf_qsl, md.pf_max_q, md.pf_max_k, out, seqused_k=md.pf_seq_lens, block_table=md.pf_bt)

    def _fa(self, q, k, v, cu_q, max_q, max_k, out, **kw):
        if self.fa_version is not None:
            kw["fa_version"] = self.fa_version
        flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu_q, max_seqlen_q=max_q, max_seqlen_k=max_k,
                               softmax_scale=self.scale, causal=True, out=out, **kw)
