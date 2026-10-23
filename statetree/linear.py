"""TreeLinear: a drop-in nn.Linear whose BF16 weight is stored losslessly in the state-tree format.

Forward paths (chosen per (shape, batch bucket), autotuned on first use or by 'statetree.prepare'):
  mv   batch <= 32     warp-per-tile matrix-vector kernel, decode in registers, split-K merged by the last CTA
  tm   batch 2..512    Marlin-style fused decode + mma.sync GEMM, stream-K over (row block, 64-col group) units
  dec  any batch       decode to a BF16 scratch matrix, then cuBLAS (large batches / prefill)
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
from .ops import ext

__all__ = ["TreeLinear", "TreeSlice", "TreeEmbedding", "BUCKETS"]

# tm kernel configurations: cfg -> (R, NTW, WR, WB, ST, OCC); rows per CTA = 16 * WR * R, batch per CTA = 8 * WB * NTW
TM_CFGS = {0: (1, 2, 8, 1, 6, 1), 1: (1, 2, 8, 1, 4, 2), 2: (1, 4, 8, 1, 6, 1), 3: (1, 4, 8, 1, 4, 2),
           4: (1, 8, 8, 1, 6, 1), 5: (1, 8, 8, 1, 3, 2), 6: (1, 16, 8, 1, 4, 1), 7: (1, 16, 8, 1, 2, 2),
           8: (2, 8, 8, 1, 2, 2), 9: (1, 16, 4, 1, 2, 3), 10: (2, 8, 4, 1, 2, 3), 14: (1, 8, 4, 1, 2, 4)}
BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
BIG = 1 << 30                                       # batches above 512
TUNE_VERSION = 2


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
    if B > 32:
        out.append(("dec", None))
    return out


def heuristic(M, N, B, st):
    """Default plans (no tuning): a ranked list, the first valid one is used."""
    NG = N // 64
    out = []
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

    def get_scratch(self, n):
        if self.scratch.numel() < n:
            self._keep.append(self.scratch)
            self.scratch = torch.empty(int(n), dtype=torch.bfloat16, device=self.dev)
        return self.scratch

    def flush(self):
        if self._flush is None:
            self._flush = torch.empty(max(2 * self.l2, 64 << 20), dtype=torch.uint8, device=self.dev)
        self._flush.max()                         # a read: evicts L2 without leaving dirty lines to write back


_STATES = {}


def _state(dev):
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    st = _STATES.get(idx)
    if st is None:
        st = _STATES[idx] = _DevState(torch.device("cuda", idx))
    return st


# ----------------------------------------------------------------------------------------- tuning table
_TUNED = {}
_TUNED_LOADED = False


def _cache_path():
    p = os.environ.get("STATETREE_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "statetree", "tune.json"))
    return None if p in ("0", "") else p


def _tkey(st, M, N, bk):
    return f"v{TUNE_VERSION}|{st.name}|{M}|{N}|{bk}"


def _tuned_get(st, M, N, bk):
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
    r = _TUNED.get(_tkey(st, M, N, bk))
    return None if r is None else [(path, tuple(prm) if prm is not None else None) for path, prm in r]


def _tuned_put(st, M, N, bk, ranked):
    _TUNED[_tkey(st, M, N, bk)] = [[p, list(q) if q is not None else None] for p, q in ranked]
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


class TreeLinear(nn.Module):
    """y = x W^T (+ b) with W [out, in] BF16 stored as a lossless state tree (~12.1 bits/weight).

    Buffers (saved in the state dict): tree [out/16, in/64, 1552] uint8, esc_rows / esc_cols int32 and esc_bits int16
    (exact bits of the ~0.01% weights the tree cannot hold), bias (optional, bf16)."""

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
    def from_linear(cls, lin, verify=True):
        return cls.from_weight(lin.weight, lin.bias, verify=verify)

    @classmethod
    @torch.no_grad()
    def from_weight(cls, W, bias=None, verify=True):
        """W [out, in] bf16 (e.g. a Linear weight, an embedding table, or several weights concatenated)."""
        W = W.detach()
        assert W.dtype == torch.bfloat16, f"TreeLinear: weight dtype {W.dtype}, expected bfloat16"
        work = W if W.is_cuda or not torch.cuda.is_available() else W.cuda()
        enc = F.encode(work)
        n = int(enc["esc_rows"].numel())
        self = cls(W.shape[1], W.shape[0], bias=bias is not None, device=W.device, n_esc=n)
        self.tree.copy_(enc["tree"])
        self.esc_rows.copy_(enc["esc_rows"])
        self.esc_cols.copy_(enc["esc_cols"])
        self.esc_bits.copy_(enc["esc_bits"])
        if bias is not None:
            self.bias.copy_(bias.detach().to(torch.bfloat16))
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

    def _valid(self, path, prm, B):
        M, N = self.out_features, self.in_features
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
        if bk == BIG:                           # a vocab-sized table never decodes to an M x N scratch matrix:
            return ("dec", None) if self.allow_dec else ("chunk", None)     # it runs in 512-row slices instead
        ranked = _tuned_get(st, M, N, bk)
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
        else:
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
    @torch.no_grad()
    def tune(self, bk, iters=12, verbose=False, group=None):
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
        _tuned_put(st, M, N, bk, ranked)
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
