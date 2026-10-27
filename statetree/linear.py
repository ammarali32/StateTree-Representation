"""TreeLinear: a drop-in nn.Linear whose BF16 weight is stored losslessly in the state-tree format.

Forward paths (chosen per (shape, batch bucket), autotuned on first use or by 'statetree.prepare'):
  mv   batch <= 32     warp-per-tile matrix-vector kernel, decode in registers, split-K merged by the last CTA
  tm   batch 2..512    Marlin-style fused decode + mma.sync GEMM, stream-K over (row block, 64-col group) units
  tt   batch 65..1024  one-token-tile GEMM (prefill): 128 rows x TT tokens per tile, each 16 x 64 record decoded once
                       per tile into shared memory; data-parallel rounds + stream-K, or balanced split-K (small layers)
  dec  any batch       decode to a BF16 scratch matrix, then cuBLAS (large batches / prefill; the next layer's
                       decode overlaps this layer's GEMM on a side stream)
Outputs are computed from the exact BF16 weights; only the fp32 accumulation order differs from cuBLAS.
"""
import contextlib
import functools
import json
import os
import weakref

import torch
import torch.nn as nn

from . import format as F
from .ops import ext, tgx_ext

__all__ = ["TreeLinear", "TreeSlice", "TreeEmbedding", "BUCKETS", "EscapeHeavy", "MAX_ESCAPE_FRACTION"]

# tm kernel configurations: cfg -> (R, NTW, WR, WB, ST, OCC); rows per CTA = 16 * WR * R, batch per CTA = 8 * WB * NTW
TM_CFGS = {0: (1, 2, 8, 1, 6, 1), 1: (1, 2, 8, 1, 4, 2), 2: (1, 4, 8, 1, 6, 1), 3: (1, 4, 8, 1, 4, 2),
           4: (1, 8, 8, 1, 6, 1), 5: (1, 8, 8, 1, 3, 2), 6: (1, 16, 8, 1, 4, 1), 7: (1, 16, 8, 1, 2, 2),
           8: (2, 8, 8, 1, 2, 2), 9: (1, 16, 4, 1, 2, 3), 10: (2, 8, 4, 1, 2, 3), 14: (1, 8, 4, 1, 2, 4)}
BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512, 640, 768, 1024)
BIG = 1 << 30                                       # batches above 1024: decode + cuBLAS with prefetch
# matrices with more escapes than this fraction stay BF16: each escape costs 80 bits and a patch in every kernel, so
# such a tree is larger and slower than BF16 (seen on a few MoE expert tensors with a very wide exponent range)
MAX_ESCAPE_FRACTION = float(os.environ.get("STATETREE_MAX_ESCAPES", "0.01"))


class EscapeHeavy(ValueError):
    """the matrix has more escapes than max_escape_fraction allows: keep it in BF16"""
TUNE_VERSION = 4                  # 4: tt stream-K fix (small problems); plans tuned before it are re-tuned
# tt: main-loop cycles per 64-k tile per CTA by (TT, NWT) (A100, whole chip busy): the cost model that ranks tile shapes
TT_CYC = {(96, 2): 1350, (128, 2): 1630, (160, 2): 1880, (192, 2): 2250, (224, 2): 2640, (256, 2): 2945,
          (128, 4): 1861, (192, 4): 2111, (256, 4): 2745}


def tt_shapes(B, top=3):
    """the 'top' (TT, NWT) of the tt cost model for B tokens: (# token tiles) x (cycles per k-tile)"""
    return sorted(TT_CYC, key=lambda c: (-(-B // c[0]) * TT_CYC[c], -c[0]))[:top]


def _tt_cfgs(M, B, sms):
    if M % 128 or B <= 64:
        return
    for TT, NWT in tt_shapes(B):
        NT = (M // 128) * -(-B // TT)
        yield TT, NWT, 0                            # data-parallel rounds + stream-K
        for s in (1, 2, 3, 4):                      # balanced split-K (s = 1: whole tiles): layers with few tiles
            if NT * s <= 2 * sms:
                yield TT, NWT, s


@functools.lru_cache(None)
def tm_geom(cfg):
    R, NTW, WR, WB, ST, OCC = TM_CFGS[cfg]
    nth = 32 * WR * WB
    return dict(R=R, NTW=NTW, RT=WR * R, BB=WB * NTW * 8, NTH=nth, PE=nth * R * NTW * 4, ST=ST, OCC=OCC,
                SMEM=ST * (WB * NTW * 8 * 128 + WR * R * 1552 + 64))


def _cta_of(u, U, C):
    return ((u + 1) * C + U - 1) // U - 1


@functools.lru_cache(None)
def _pmax(M, NG, RT, C):
    """Most CTAs sharing one row block (sizes the stream-K partial-sum workspace)."""
    nRB = M // (16 * RT)
    U = nRB * NG
    return max(_cta_of(rb * NG + NG - 1, U, C) - _cta_of(rb * NG, U, C) + 1 for rb in range(nRB))


def bucket_of(B):
    for b in BUCKETS:
        if B <= b:
            return b
    return BIG


def _grid_options(M, NG, B, cfg, sms):
    gm = tm_geom(cfg)
    nRB = M // (16 * gm["RT"])
    U = nRB * NG
    nBB = -(-B // gm["BB"])
    opts = set()
    for k in (1, 2):
        c = max(1, sms * gm["OCC"] * k // nBB)
        if c <= U:
            opts.add(c)
    for ks in (1, 2, 4):
        c = nRB * ks
        if c <= U and c * nBB <= 3 * sms * gm["OCC"]:
            opts.add(c)
    if not opts:
        opts.add(U)
    return sorted(opts)


def _tm_cfgs(M, B, smem_max):
    for cfg in TM_CFGS:
        gm = tm_geom(cfg)
        if M % (16 * gm["RT"]) or gm["SMEM"] > smem_max:
            continue
        if gm["BB"] > max(16, 2 * B) or -(-B // gm["BB"]) > 8:
            continue
        yield cfg


def _mv_cfgs(M, NG):
    for WM in (1, 2):
        for WK in (1, 2, 4, 8):
            for GY in (1, 2, 4, 8, 16):
                for PF in (2, 4):
                    if WM * WK <= 8 and M % (16 * WM) == 0 and NG >= GY * WK * 2 and (M // (16 * WM)) * GY <= 4096:
                        yield WM, WK, GY, PF


def candidates(M, N, B, st):
    NG = N // 64
    out = []
    if B <= 32:
        out += [("mv", c) for c in _mv_cfgs(M, NG)]
    if 2 <= B <= 512:
        for cfg in _tm_cfgs(M, B, st.smem_max):
            out += [("tm", (cfg, C)) for C in _grid_options(M, NG, B, cfg, st.sms)]
    out += [("tt", c) for c in _tt_cfgs(M, B, st.sms)]
    if B > 32:
        out.append(("dec", None))
    return out


def heuristic(M, N, B, st):
    """Default plans (no tuning): a ranked list, the first valid one is used."""
    NG = N // 64
    out = []
    if 64 < B and M % 128 == 0:                     # prefill batches: tt with the cost model's tile shape
        out.append(("tt", (*tt_shapes(B, 1)[0], 0)))
    if B <= 8 or (B <= 32 and M < 1024):
        tiles = M // 16
        WK = next((w for w in (8, 4, 2, 1) if NG >= 2 * w), 1)
        GY = 1
        while GY < 16 and tiles * GY * 2 <= 2 * st.sms and NG >= 2 * GY * WK * 2:
            GY *= 2
        out.append(("mv", (1, WK, GY, 2)))
    if 2 <= B <= 512:
        pref = (1, 3, 9) if B <= 16 else (3, 9) if B <= 32 else (5, 9) if B <= 64 else (9, 10) if B <= 128 \
            else (10, 9)
        for cfg in pref:
            gm = tm_geom(cfg)
            if M % (16 * gm["RT"]) == 0 and gm["SMEM"] <= st.smem_max:
                out.append(("tm", (cfg, _grid_options(M, NG, B, cfg, st.sms)[0])))
    if B <= 32:
        out.append(("mv", (1, 1, 1, 2)))
    out.append(("dec", None))
    return out


# ----------------------------------------------------------------------------------------- large-batch prefetch
_PF_OFF = [0]                                      # > 0 while tuning: no side-stream work


def _prefetch_on():
    return os.environ.get("STATETREE_PREFETCH", "1") != "0" and _PF_OFF[0] == 0


def _no_prefetch(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        _PF_OFF[0] += 1
        try:
            return fn(*a, **k)
        finally:
            _PF_OFF[0] -= 1
    return w


class _Prefetch:
    """decode + cuBLAS with prefetch: while layer i's GEMM runs, a low-priority side stream decodes the weight of the
    layer expected next into the other of two BF16 buffers, so the decode overlaps the GEMM (its tail waves) instead
    of adding to it. The order is learned from the first pass (the decode-path calls up to the first repeat); a wrong
    guess only costs that decode. The buffers grow to the largest layer seen."""

    def __init__(self, dev):
        self.dev = dev
        self.side = torch.cuda.Stream(device=dev, priority=torch.cuda.Stream.priority_range()[0])
        self.buf, self.who, self.ready, self.free = [None, None], [None, None], [None, None], [None, None]
        self.last = 1
        self.seq, self.pos, self.closed = [], {}, False

    def _buf(self, i, n):
        b = self.buf[i]
        if b is None or b.numel() < n:
            for e in (self.ready[i], self.free[i]):
                if e is not None:
                    e.synchronize()                    # nothing in flight uses the old buffer when it is dropped
            self.buf[i] = b = torch.empty(int(n), dtype=torch.bfloat16, device=self.dev)
        return b

    def _next(self, m):
        k = id(m)
        if not self.closed:
            if k not in self.pos:
                self.pos[k] = len(self.seq)
                self.seq.append(weakref.ref(m))
                return None
            self.closed = len(self.seq) > 1
            if not self.closed:
                return None
        i = self.pos.get(k)
        if i is None:
            return None
        n = self.seq[(i + 1) % len(self.seq)]()
        return n if n is not None and n.allow_dec and n.tree.device == m.tree.device else None

    @staticmethod
    def _key(m):
        return id(m), m.tree.data_ptr(), m.tree._version

    def run(self, m, x, y):
        M, N = m.out_features, m.in_features
        cur = torch.cuda.current_stream(self.dev)
        k = self._key(m)
        if k in self.who:
            b = self.who.index(k)
            cur.wait_event(self.ready[b])
        else:
            b = 1 - self.last
            buf = self._buf(b, M * N)
            if self.ready[b] is not None:
                cur.wait_event(self.ready[b])          # a prefetch into b may still be running
            eo, el = m._esc_lists()
            ext().dec(m.tree, eo, el, buf[:M * N].view(M, N))
            self.who[b] = k
            self.ready[b] = torch.cuda.Event()
            self.ready[b].record(cur)
        n = self._next(m)
        o = 1 - b
        if n is not None and self.who[o] != self._key(n):
            n_eo, n_el = n._esc_lists()                # side tables are built on this stream, before the GEMM
            go = torch.cuda.Event()
            go.record(cur)
        else:
            n = None
        torch.matmul(x, self.buf[b][:M * N].view(M, N).t(), out=y)
        self.free[b] = torch.cuda.Event()
        self.free[b].record(cur)
        self.last = b
        if n is not None:
            Mn, Nn = n.out_features, n.in_features
            buf = self._buf(o, Mn * Nn)
            s = self.side
            s.wait_event(go)
            if self.free[o] is not None:
                s.wait_event(self.free[o])             # the GEMM that last read buffer o
            with torch.cuda.stream(s):
                ext().dec(n.tree, n_eo, n_el, buf[:Mn * Nn].view(Mn, Nn))
            self.ready[o] = torch.cuda.Event()
            self.ready[o].record(s)
            self.who[o] = self._key(n)


# ----------------------------------------------------------------------------------------- per-device state
class _DevState:
    def __init__(self, dev):
        self.dev = dev
        p = torch.cuda.get_device_properties(dev)
        self.name, self.sms = p.name, p.multi_processor_count
        self.l2 = int(getattr(p, "L2_cache_size", 40 << 20))
        self.smem_max = int(getattr(p, "shared_memory_per_block_optin", 99 << 10))
        self.cnt = torch.zeros(1 << 16, dtype=torch.int32, device=dev)     # semaphores, kernels reset them to 0
        self.ws = torch.empty(1, dtype=torch.float32, device=dev)
        self.scratch = torch.empty(1, dtype=torch.bfloat16, device=dev)
        self._keep = []                          # superseded buffers stay alive: captured CUDA graphs may use them
        self._flush = None

    def get_ws(self, n):
        if self.ws.numel() < n:
            self._keep.append(self.ws)
            self.ws = torch.empty(int(n), dtype=torch.float32, device=self.dev)
        return self.ws

    def get_mfd(self, n_act, n_cnt):
        """activation scratch and counters of the fused MoE decode kernel (counters stay zero between calls)"""
        a = getattr(self, "mfd_act", None)
        if a is None or a.numel() < n_act or self.mfd_cnt.numel() < n_cnt:
            if a is not None:
                self._keep += [a, self.mfd_cnt]
            self.mfd_act = torch.empty(max(int(n_act), 1 << 17), dtype=torch.bfloat16, device=self.dev)
            self.mfd_cnt = torch.zeros(max(int(n_cnt), 1 << 12), dtype=torch.int32, device=self.dev)
        return self.mfd_act, self.mfd_cnt

    def get_tgx(self, M, N):
        """ring buffer (R strips of 128 x N BF16, about <= 64 MB) and counters of the large-batch kernel"""
        NS = M // 128
        R = max(2, min(NS, 48, (64 << 20) // (256 * N)))
        if getattr(self, "tgx_ring", None) is None or self.tgx_ring.numel() < R * 128 * N:
            if getattr(self, "tgx_ring", None) is not None:
                self._keep.append(self.tgx_ring)
            self.tgx_ring = torch.empty(R * 128 * N, dtype=torch.bfloat16, device=self.dev)
        if getattr(self, "tgx_cnt", None) is None or self.tgx_cnt.numel() < 2 * NS + 1:
            self.tgx_cnt = torch.zeros(2 * NS + 1, dtype=torch.int32, device=self.dev)
        return self.tgx_ring[:R * 128 * N], self.tgx_cnt

    def prefetcher(self):
        if getattr(self, "pf", None) is None:
            self.pf = _Prefetch(self.dev)
        return self.pf

    def get_tt_ws(self):
        """tt: partial accumulators of split tiles (2 G slots of 128 x 256 fp32) and their counters (zero between
        launches: the last arriving piece resets its counter)"""
        if getattr(self, "tt_ws", None) is None:
            self.tt_ws = torch.empty(self.sms * 2 * 256 * 128, dtype=torch.float32, device=self.dev)
            self.tt_cnt = torch.zeros(4 * self.sms, dtype=torch.int32, device=self.dev)
        return self.tt_ws, self.tt_cnt

    def get_scratch(self, n):
        if self.scratch.numel() < n:
            self._keep.append(self.scratch)
            self.scratch = torch.empty(int(n), dtype=torch.bfloat16, device=self.dev)
        return self.scratch

    def snapshot(self):
        """the growable buffers, taken before an autotuning run"""
        return dict(ws=self.ws, scratch=self.scratch, keep=len(self._keep), flush=self._flush,
                    tgx=getattr(self, "tgx_ring", None))

    def restore(self, snap):
        """drop what an autotuning run grew: losing candidates would otherwise pin their workspaces and decode
        scratch matrices for the life of the process (3.6 GiB on Qwen3-8B, taken from vLLM's KV cache). The winning
        plans grow them again on first use; buffers that existed before tuning stay (a captured CUDA graph may use them)."""
        self.ws, self.scratch, self._flush = snap["ws"], snap["scratch"], snap["flush"]
        if snap["tgx"] is not None or getattr(self, "tgx_ring", None) is not None:
            self.tgx_ring = snap["tgx"]
        del self._keep[snap["keep"]:]

    def flush(self):
        if self._flush is None:
            self._flush = torch.empty(max(2 * self.l2, 64 << 20), dtype=torch.uint8, device=self.dev)
        self._flush.max()                         # a read: evicts L2 without leaving dirty lines to write back


_STATES = {}


def _state(dev):
    """workspaces, split-K semaphores and decode scratch of the current CUDA stream on dev: layers that run
    concurrently on two streams (e.g. vLLM's MoE shared expert on its side stream while the router and the routed
    experts run on the main one) must not share them"""
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    key = (idx, torch.cuda.current_stream(idx).cuda_stream)
    st = _STATES.get(key)
    if st is None:
        st = _STATES[key] = _DevState(torch.device("cuda", idx))
    return st


# ----------------------------------------------------------------------------------------- tuning table
_TUNED = {}
_TUNED_LOADED = False


def _cache_path():
    p = os.environ.get("STATETREE_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "statetree", "tune.json"))
    return None if p in ("0", "") else p


def _tkey(st, M, N, bk, tag=""):
    return f"v{TUNE_VERSION}{tag}|{st.name}|{M}|{N}|{bk}"


def _tuned_get(st, M, N, bk, tag=""):
    global _TUNED_LOADED
    if not _TUNED_LOADED:
        _TUNED_LOADED = True
        p = _cache_path()
        if p and os.path.exists(p):
            try:
                with open(p) as fh:
                    _TUNED.update(json.load(fh))
            except (OSError, ValueError):
                pass
    r = _TUNED.get(_tkey(st, M, N, bk, tag))
    return None if r is None else [(path, tuple(prm) if prm is not None else None) for path, prm in r]


def _tuned_put(st, M, N, bk, ranked, tag=""):
    _TUNED[_tkey(st, M, N, bk, tag)] = [[p, list(q) if q is not None else None] for p, q in ranked]
    p = _cache_path()
    if p:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = f"{p}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump(_TUNED, fh)
            os.replace(tmp, p)
        except OSError:
            pass


def _autotune_on():
    return os.environ.get("STATETREE_AUTOTUNE", "1") != "0"


def _device_ctx(dev):
    """Make dev the current CUDA device while the kernels launch (layers spread over GPUs by a device map)."""
    if dev.type == "cuda" and dev.index is not None and dev.index != torch.cuda.current_device():
        return torch.cuda.device(dev)
    return contextlib.nullcontext()


def _capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _compiling():
    f = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    return bool(f and f())


# ----------------------------------------------------------------------------------------- the module
_REGISTRY = weakref.WeakValueDictionary()        # handle -> module, for the torch.compile custom op


def tt_escape_tables(M, N, esc_rows, esc_cols, esc_bits):
    """escape tables of the tt / gtt kernels for an M x N tree (M % 128 == 0), per 128-row block rb: koff[rb, kt] =
    start of k-tile kt's escapes (KS ints per row) and ent[rb, :CAP] = bf16 bits << 16 | record << 10 | row << 6 |
    col, sorted by k-tile. Returns (koff, ent, KS, CAP), int32. One host sync (CAP)."""
    dev = esc_rows.device
    er, ec = esc_rows.long(), esc_cols.long()
    eb = esc_bits.long() & 0xFFFF
    NG, nrb = N // 64, M // 128
    KS = (NG + 1 + 3) // 4 * 4
    rb, kt = er // 128, ec // 64
    key = rb * NG + kt
    cnt = (torch.bincount(key, minlength=nrb * NG).view(nrb, NG) if er.numel()
           else torch.zeros(nrb, NG, dtype=torch.long, device=dev))
    per = cnt.sum(1)
    CAP = max(4, (int(per.max().item()) + 3) // 4 * 4)
    koff = torch.zeros(nrb, KS, dtype=torch.long, device=dev)
    koff[:, 1:NG + 1] = cnt.cumsum(1)
    koff[:, NG + 1:] = koff[:, NG:NG + 1]
    ent = torch.zeros(nrb, CAP, dtype=torch.long, device=dev)
    if er.numel():
        order = torch.argsort(key, stable=True)
        er_, ec_, eb_, rb_ = er[order], ec[order], eb[order], rb[order]
        start = torch.zeros(nrb + 1, dtype=torch.long, device=dev)
        start[1:] = per.cumsum(0)
        pos = torch.arange(er_.numel(), device=dev) - start[rb_]
        ent[rb_, pos] = (eb_ << 16) | (((er_ % 128) // 16) << 10) | ((er_ % 16) << 6) | (ec_ % 64)
    return F._i32(koff.view(-1)), F._i32(ent.view(-1)), KS, CAP


def gtt_escape_tables(M, N, esc_rows, esc_cols, esc_bits):
    """escape tables of the grouped tree GEMM (csrc/gtt.cu) for an M x N tree (M % 128 == 0; MoE experts stacked one
    after the other): the tt entries without padding, ent holding every 128-row block's entries back to back and
    koff[rb, kt] the absolute start of k-tile kt's entries of block rb (koff[rb, N / 64] = the block's end). CAP = the
    largest block's entry count (the kernel copies a block's list to shared memory when it fits). With padding to a
    common CAP, one escape-dense block would size every block's list (1.3 GiB per GPU for Qwen3.6-35B-A3B at TP = 2).
    Returns (koff, ent, KS, CAP), int32. One host sync."""
    dev = esc_rows.device
    er, ec = esc_rows.long(), esc_cols.long()
    eb = esc_bits.long() & 0xFFFF
    NG, nrb = N // 64, M // 128
    KS = (NG + 1 + 3) // 4 * 4
    key = (er // 128) * NG + ec // 64
    cnt = (torch.bincount(key, minlength=nrb * NG) if er.numel()
           else torch.zeros(nrb * NG, dtype=torch.long, device=dev))
    start = torch.zeros(nrb * NG + 1, dtype=torch.long, device=dev)
    start[1:] = cnt.cumsum(0)
    idx = torch.arange(nrb, device=dev).unsqueeze(1) * NG + torch.arange(NG + 1, device=dev)
    koff = torch.empty(nrb, KS, dtype=torch.long, device=dev)
    koff[:, :NG + 1] = start[idx]
    koff[:, NG + 1:] = koff[:, NG:NG + 1]
    CAP = max(4, (int(cnt.view(nrb, NG).sum(1).max().item()) + 3) // 4 * 4)
    ent = torch.zeros(er.numel() + 4, dtype=torch.long, device=dev)
    if er.numel():
        order = torch.argsort(key, stable=True)
        er_, ec_, eb_ = er[order], ec[order], eb[order]
        ent[:er.numel()] = (eb_ << 16) | (((er_ % 128) // 16) << 10) | ((er_ % 16) << 6) | (ec_ % 64)
    return F._i32(koff.view(-1)), F._i32(ent), KS, CAP


class TreeLinear(nn.Module):
    """y = x W^T (+ b) with W [out, in] BF16 stored as a lossless state tree (~12.1 bits/weight).

    Buffers (saved in the state dict): tree [out/16, in/64, 1552] uint8, esc_rows / esc_cols int32 and esc_bits int16
    (exact bits of the ~0.01% weights the tree cannot hold), bias (optional, bf16)."""

    TUNE_TAG = ""                                   # tuning-table key suffix (subclasses running other kernels)

    def __init__(self, in_features, out_features, bias=False, device=None, n_esc=0):
        super().__init__()
        assert out_features % 16 == 0 and in_features % 64 == 0, "TreeLinear needs out % 16 == 0 and in % 64 == 0"
        self.in_features, self.out_features = in_features, out_features
        dev = torch.device(device) if device is not None else None
        self.register_buffer("tree", torch.empty(out_features // 16, in_features // 64, F.RECORD_BYTES,
                                                 dtype=torch.uint8, device=dev))
        self.register_buffer("esc_rows", torch.empty(n_esc, dtype=torch.int32, device=dev))
        self.register_buffer("esc_cols", torch.empty(n_esc, dtype=torch.int32, device=dev))
        self.register_buffer("esc_bits", torch.empty(n_esc, dtype=torch.int16, device=dev))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.bfloat16, device=dev))
        else:
            self.bias = None
        self.set_handle(f"tl{id(self)}")
        self.allow_dec = True               # False: batches <= 512 never take the decode + cuBLAS path (no M*N scratch)
        self._reset_runtime()

    def set_handle(self, name):
        """The string key under which the torch.compile custom op finds this layer (a stable name, e.g. the module
        path, keeps compiled graphs valid across processes)."""
        self._handle = str(name)
        _REGISTRY[self._handle] = self

    # ---------------------------------------------------------------- construction / state dict
    @classmethod
    def from_linear(cls, lin, verify=True, max_escape_fraction=None):
        return cls.from_weight(lin.weight, lin.bias, verify=verify, max_escape_fraction=max_escape_fraction)

    @classmethod
    @torch.no_grad()
    def from_weight(cls, W, bias=None, verify=True, max_escape_fraction=None):
        """W [out, in] bf16 (e.g. a Linear weight, an embedding table, or several weights concatenated).
        max_escape_fraction: raise EscapeHeavy (before any side table is built) when more weights escape."""
        W = W.detach()
        assert W.dtype == torch.bfloat16, f"TreeLinear: weight dtype {W.dtype}, expected bfloat16"
        work = W if W.is_cuda or not torch.cuda.is_available() else W.cuda()
        if max_escape_fraction is not None:
            k = F.escape_count(work)
            if k > max_escape_fraction * work.numel():
                raise EscapeHeavy(f"{k / work.numel():.2%} of the weights escape")
        enc = F.encode(work)
        n = int(enc["esc_rows"].numel())
        self = cls(W.shape[1], W.shape[0], bias=bias is not None, device="meta", n_esc=n)
        for name in ("tree", "esc_rows", "esc_cols", "esc_bits"):     # adopt the encoded tensors (no copies; peak
            self._buffers[name] = enc[name].to(W.device)                # memory: the tree once)
        del enc
        if bias is not None:
            self._buffers["bias"] = bias.detach().to(device=W.device, dtype=torch.bfloat16).clone()
        self._reset_runtime()
        if verify:
            got = self.dequantize() if self.tree.is_cuda else F.decode_reference(self.tree, self.esc_rows,
                                                                                  self.esc_cols, self.esc_bits)
            if not torch.equal(got.view(torch.int16), W.view(torch.int16).to(got.device)):
                raise RuntimeError("TreeLinear: decoded weights differ from the original (format bug)")
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        for name in ("esc_rows", "esc_cols", "esc_bits"):        # data-dependent sizes: adopt the stored size
            v = state_dict.get(prefix + name)
            if v is not None and v.shape != self._buffers[name].shape:
                self._buffers[name] = torch.empty(v.shape, dtype=v.dtype, device=self._buffers[name].device)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)
        self._reset_runtime()

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        self._reset_runtime()
        return out

    def _reset_runtime(self):
        self._rt = {}                                   # escape side tables per (device, kind)
        self._plans = {}                                # batch bucket -> (path, params)
        self._fcache = None                             # fused groups: (input, output) shared by the slices

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
                f"escapes={self.esc_rows.numel()}, bits_per_weight={self.bits_per_weight:.3f}")

    @property
    def bits_per_weight(self):
        return F.bits_per_weight(self.out_features, self.in_features, self.esc_rows.numel())

    def storage_bytes(self):
        return sum(b.numel() * b.element_size() for b in (self.tree, self.esc_rows, self.esc_cols, self.esc_bits))

    # ---------------------------------------------------------------- escape side tables (built lazily, cached)
    def _esc_lists(self):
        k = (self.tree.device, "lists")
        if k not in self._rt:
            self._rt[k] = F.escape_lists(self.out_features, self.in_features, self.esc_rows, self.esc_cols,
                                         self.esc_bits)
        return self._rt[k]

    def _esc_blocks(self, RT):
        k = (self.tree.device, "blocks", RT)
        if k not in self._rt:
            self._rt[k] = F.escape_blocks(self.out_features, self.in_features, RT, self.esc_rows, self.esc_cols,
                                          self.esc_bits)
        return self._rt[k]

    def _tt_tables(self):
        """tt escape tables, per 128-row block rb: koff[rb, kt] = start of k-tile kt's escapes (KS ints per row) and
        ent[rb, :CAP] = bf16 bits << 16 | record << 10 | row << 6 | col, sorted by k-tile"""
        k = (self.tree.device, "tt")
        if k not in self._rt:
            self._rt[k] = tt_escape_tables(self.out_features, self.in_features, self.esc_rows, self.esc_cols,
                                           self.esc_bits)
        return self._rt[k]

    def _valid(self, path, prm, B):
        M, N = self.out_features, self.in_features
        if path == "tt":
            st = _state(self.tree.device)
            if M % 128 or B <= 0 or (M // 128) * -(-B // prm[0]) * prm[2] > 2 * st.sms:
                return False
            self._tt_tables()                       # build the side tables now (not inside a CUDA-graph capture)
            return True
        if path == "mv":
            return B <= 32 and M % (16 * prm[0]) == 0
        if path == "tm":
            RT = tm_geom(prm[0])["RT"]
            if M % (16 * RT):
                return False
            self._esc_blocks(RT)                    # build the side table now (not inside a CUDA-graph capture)
        return True

    @torch.no_grad()
    def dequantize(self):
        """The exact BF16 weight [out, in] (decoded by the CUDA kernel)."""
        W = torch.empty(self.out_features, self.in_features, dtype=torch.bfloat16, device=self.tree.device)
        eo, el = self._esc_lists()
        ext().dec(self.tree, eo, el, W)
        return W

    @torch.no_grad()
    def rows(self, ids):
        """W[ids] (exact BF16 rows), decoded on the fly: the embedding lookup of a tree-stored table."""
        dev = self.tree.device
        flat = ids.reshape(-1)
        if flat.dtype != torch.int64 or not flat.is_contiguous() or flat.device != dev:
            flat = flat.to(device=dev, dtype=torch.int64).contiguous()
        out = torch.empty(flat.numel(), self.in_features, dtype=torch.bfloat16, device=dev)
        eo, el = self._esc_lists()
        with _device_ctx(dev):
            ext().rows(self.tree, eo, el, flat, out)
        out = out.view(*ids.shape, self.in_features)
        # a tied table may live on another GPU than the embedding module (device maps): answer on the ids' GPU
        return out.to(ids.device) if ids.is_cuda and ids.device != dev else out

    def _fused_part(self, x, start, size, nparts):
        """Output columns [start, start + size) of this (fused) layer for input x. Parts called with the same input
        object share one GEMM; the cached output is dropped once every part has taken its slice."""
        c = self._fcache
        if c is None or c[0] is not x:
            c = self._fcache = [x, self(x), 0]
        out = c[1].narrow(-1, start, size)
        c[2] += 1
        if c[2] >= nparts:
            self._fcache = None
        return out

    # ---------------------------------------------------------------- dispatch
    def _plan(self, B):
        bk = bucket_of(B)
        p = self._plans.get(bk)
        if p is None:
            p = self._plans[bk] = self._resolve(bk)
        return p

    def _resolve(self, bk):
        M, N = self.out_features, self.in_features
        st = _state(self.tree.device)
        if bk == BIG:
            # large batches (prefill): CUTLASS GEMM with a concurrent decode kernel, no M x N scratch matrix
            if M % 128 == 0 and N % 64 == 0 and tgx_ext() is not None:
                return ("tgx", None)
            # otherwise decode + cuBLAS; a vocab-sized table never decodes to an M x N scratch matrix, it runs in
            # 512-row slices instead
            return ("dec", None) if self.allow_dec else ("chunk", None)
        ranked = _tuned_get(st, M, N, bk, self.TUNE_TAG)
        if ranked is None and _autotune_on() and not _capturing():
            ranked = self.tune(bk)
        for path, prm in (ranked or []) + heuristic(M, N, bk, st):
            if (path != "dec" or self.allow_dec) and self._valid(path, prm, bk):
                return (path, prm)
        return ("dec", None)

    def _run(self, x, y, plan):
        path, prm = plan
        M, N, B = self.out_features, self.in_features, x.shape[0]
        if path == "chunk":
            for i in range(0, B, 512):
                n = min(512, B - i)
                self._run(x[i:i + n], y[i:i + n], self._plan(n))
            return
        st = _state(x.device)
        E = ext()
        if path == "mv":
            WM, WK, GY, PF = prm
            eo, el = self._esc_lists()
            nbc = 8 if B <= 8 else 16 if B <= 16 else 32
            ws = st.get_ws((M // (16 * WM)) * GY * 16 * WM * nbc if GY > 1 else 1)
            E.mv(self.tree, eo, el, x, y, ws, st.cnt, WM, WK, GY, PF)
        elif path == "tm":
            cfg, C = prm
            gm = tm_geom(cfg)
            eb, eou, eall = self._esc_blocks(gm["RT"])
            P = _pmax(M, N // 64, gm["RT"], C)
            nbb = -(-B // gm["BB"])
            ws = st.get_ws(nbb * (M // (16 * gm["RT"])) * P * gm["PE"])
            E.tm(self.tree, eb, eou, eall, x, y, ws, st.cnt, cfg, C, P, 1 if nbb == 1 else 0, 0)
        elif path == "tt":
            TT, NWT, sp = prm
            koff, ent, KS, CAP = self._tt_tables()
            ws, cnt = st.get_tt_ws()
            E.tt(TT, NWT, sp, x, self.tree, koff, ent, y, ws, cnt, KS, CAP)
        elif path == "tgx":
            eo, el = self._esc_lists()
            ring, cnt = st.get_tgx(M, N)
            tgx_ext().tgx(self.tree, eo, el, x, y, ring, cnt, 0)
        else:
            if _prefetch_on() and not _capturing():
                st.prefetcher().run(self, x, y)
                return
            eo, el = self._esc_lists()
            W = st.get_scratch(M * N)[:M * N].view(M, N)
            E.dec(self.tree, eo, el, W)
            torch.matmul(x, W.t(), out=y)

    def forward(self, x):
        if _compiling() and hasattr(torch.ops, "statetree") and hasattr(torch.ops.statetree, "linear"):
            return torch.ops.statetree.linear(x, self.tree, self._handle)
        return self._forward(x)

    def _forward(self, x):
        dev, src = self.tree.device, x.device
        if src != dev:                          # e.g. a device map without a hook on this module: compute where the
            x = x.to(dev)                       # tree lives, return on the caller's device
        shp = x.shape
        x2 = x.reshape(-1, self.in_features)
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        if not x2.is_contiguous() or x2.data_ptr() % 16:
            x2 = x2.clone(memory_format=torch.contiguous_format)
        B = x2.shape[0]
        y = torch.empty(B, self.out_features, dtype=torch.bfloat16, device=x2.device)
        if B:
            with _device_ctx(dev):
                self._run(x2, y, self._plan(B))
        if self.bias is not None:
            y += self.bias
        y = y.view(*shp[:-1], self.out_features)
        y = y if x.dtype == torch.bfloat16 else y.to(x.dtype)
        return y if src == dev else y.to(src)

    # ---------------------------------------------------------------- tuning
    @_no_prefetch
    @torch.no_grad()
    def tune(self, bk, iters=12, verbose=False, group=None):
        """Autotune batch bucket bk (see _tune); the buffers the candidates grow are released afterwards."""
        st = _state(self.tree.device)
        snap = st.snapshot()
        try:
            return self._tune(bk, iters, verbose, group)
        finally:
            st.restore(snap)

    def _tune(self, bk, iters=12, verbose=False, group=None):
        """Time every candidate path for batch bucket bk and store the ranking for this shape. With group (layers of
        the same shape, e.g. one per transformer block) each timing runs the whole group back to back after an L2
        flush, as the layers run in a model; otherwise this layer alone after an L2 flush."""
        group = [m for m in (group or [self]) if m.tree.device == self.tree.device] or [self]
        M, N = self.out_features, self.in_features
        st = _state(self.tree.device)
        B = bk if bk != BIG else 1024
        dev = self.tree.device
        x = torch.randn(B, N, device=dev, dtype=torch.bfloat16)
        y = torch.empty(B, M, device=dev, dtype=torch.bfloat16)
        ref = (x.float() @ self.dequantize().float().t())
        scale = ref.norm()
        res = []
        for path, prm in candidates(M, N, B, st):
            if not all(m._valid(path, prm, B) for m in group):
                continue
            plan = (path, prm)
            try:
                self._run(x, y, plan)
                torch.cuda.synchronize(dev)
                if float((y.float() - ref).norm() / scale) > 1e-2:
                    continue
                evs = []
                for i in range(iters + 2):
                    st.flush()
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    for m in group:
                        m._run(x, y, plan)
                    b.record()
                    if i >= 2:
                        evs.append((a, b))
                torch.cuda.synchronize(dev)
                ts = sorted(a.elapsed_time(b) for a, b in evs)
                res.append((ts[len(ts) // 2] / len(group), plan))
            except RuntimeError as e:                       # e.g. a config the GPU cannot launch
                if verbose:
                    print("statetree tune: skip", plan, str(e)[:80])
                torch.cuda.synchronize(dev)
        res.sort(key=lambda r: r[0])
        ranked = [p for _, p in res[:4]]
        if verbose and res:
            print(f"statetree tune {M}x{N} b{bk}: " + ", ".join(f"{p[0]}{p[1]} {t * 1e3:.1f}us" for t, p in res[:3]))
        _tuned_put(st, M, N, bk, ranked, self.TUNE_TAG)
        return ranked

    def prepare(self, batch_sizes=(1,)):
        """Build the side tables and resolve the plans for these batch sizes (do it before CUDA-graph capture)."""
        self._esc_lists()
        for B in batch_sizes:
            self._plan(B)


class TreeSlice(nn.Module):
    """One of several Linear layers that read the same input (q/k/v, gate/up), fused into one TreeLinear on the parent:
    the first part computes the whole fused output once, every part returns its column slice (a view)."""

    def __init__(self, fused, start, size, nparts):
        super().__init__()
        object.__setattr__(self, "fused", fused)       # not a submodule: its buffers belong to the parent's fused layer
        self.in_features, self.out_features = fused.in_features, size
        self.start, self.nparts = start, nparts

    def forward(self, x):
        return self.fused._fused_part(x, self.start, self.out_features, self.nparts)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, columns from {self.start} of fused"


class TreeEmbedding(nn.Module):
    """nn.Embedding whose table lives in the state-tree format: rows are decoded exactly on lookup. The table is either
    its own TreeLinear ('table') or, for tied input/output embeddings, the lm_head TreeLinear ('source')."""

    def __init__(self, num_embeddings, embedding_dim, table=None, source=None):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.table = table                              # registered submodule when the table is owned
        object.__setattr__(self, "source", source)     # tied: the lm_head TreeLinear (not a submodule)

    @classmethod
    def from_embedding(cls, emb, verify=True):
        return cls(emb.num_embeddings, emb.embedding_dim, table=TreeLinear.from_weight(emb.weight, verify=verify))

    def forward(self, ids):
        src = self.table if self.table is not None else self.source
        if _compiling() and hasattr(torch.ops, "statetree") and hasattr(torch.ops.statetree, "rows"):
            return torch.ops.statetree.rows(ids, src.tree, src._handle)
        hook = getattr(src, "_hf_hook", None)
        if hook is not None and src.tree.device.type == "meta":    # table offloaded by an accelerate device map: rows()
            hook.pre_forward(src)                                   # bypasses the module call, so stream it in here
            try:
                return src.rows(ids)
            finally:
                hook.post_forward(src, None)
        return src.rows(ids)

    def extra_repr(self):
        return f"{self.num_embeddings}, {self.embedding_dim}, " + ("own table" if self.table is not None else "tied to lm_head")


# ----------------------------------------------------------------------------------------- torch.compile support
def _register_op():
    lib = getattr(torch, "library", None)
    if lib is None or not hasattr(lib, "custom_op"):
        return

    @lib.custom_op("statetree::linear", mutates_args=())
    def _linear(x: torch.Tensor, tree: torch.Tensor, handle: str) -> torch.Tensor:
        return _REGISTRY[handle]._forward(x)

    @_linear.register_fake
    def _(x, tree, handle):
        return x.new_empty(*x.shape[:-1], tree.shape[0] * 16)

    @lib.custom_op("statetree::rows", mutates_args=())
    def _rows(ids: torch.Tensor, tree: torch.Tensor, handle: str) -> torch.Tensor:
        return _REGISTRY[handle].rows(ids)

    @_rows.register_fake
    def _(ids, tree, handle):
        return (ids if ids.is_cuda else tree).new_empty(*ids.shape, tree.shape[1] * 64, dtype=torch.bfloat16)


try:
    _register_op()
except Exception:                                                         # noqa: BLE001  (old torch)
    pass
