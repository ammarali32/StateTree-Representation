"""MoE experts stored as state trees.

Transformers 5.x keeps the routed experts of an MoE block as 3-D parameters of a single module: the classes decorated
with transformers.integrations.moe.use_experts_implementation (Qwen3-MoE, Qwen3.5 / 3.6-MoE, Qwen3-Next, Mixtral,
DeepSeek-V3, gpt-oss, ...). Layout: gate_up_proj [E, 2I, H] (up_proj [E, I, H] without a gate) and down_proj [E, H, I],
optional biases [E, out], or the transposed layout [E, in, out] (is_transposed). TreeExperts stores each expert's two
matrices as TreeLinear layers (the exact BF16 weights, ~12.1 bits/weight; the BF16 tensors are freed as they are
encoded) and computes the block like transformers' grouped_mm path: gating, routing weights and the per-token sum are
unchanged. The products run straight from the trees, with no host sync and no BF16 copy of the experts:
  mvx  up to 64 (token, expert) pairs: one matrix-vector product per pair, one launch per projection
  mvg  up to 8 pairs per expert on average: the pairs sorted by expert into blocks of 8 / 16 (vLLM's
       moe_align_block_size layout), each block reading its expert's tree once
  gtt  beyond: the grouped tree GEMM, blocks of 32 / 64 / 128 pairs of one expert, each block decoding its expert's
       128-row tiles into shared memory once and running tensor-core MMAs on them (csrc/gtt.cu)
  mvg also runs any batch when gtt does not fit the shapes (128-row blocks); the per-expert loop (every active expert
  runs its own TreeLinear, one host sync) is the fallback beyond mvg's grid limit
STATETREE_MOE_MVX_MAX / STATETREE_MOE_MVG_MAX / STATETREE_MOE_GTT_MIN override the limits (pairs), STATETREE_MOE_GTT=0
turns gtt off."""
import os

import torch
import torch.nn as nn

from .linear import TreeLinear, _device_ctx, _state, gtt_escape_tables
from .ops import ext

UP_NAMES = ("gate_up_proj", "up_proj")


def mvx_cfg(M, N, S):
    """(WM, WK, GY, PF) of the mvx kernel for S slots of M x N expert matrices: the 64-column groups of a 16-row tile
    are split over enough warps to have ~2048 in flight, and matrices with <= 8 groups (the down projection at TP = 2,
    K = 256) always one record per warp; prefetch depth 2, no split-K across blocks (slower in every case). A100 sweep
    on the real decode routing of Qwen3.6-35B-A3B (moe_kbench10/11, TP = 2 shapes, batch 1-4): the best configuration
    at every size, 10-25% faster than the 0.5.1 rule on the down projection."""
    NG = N // 64
    WK = 1
    while WK < min(8, NG) and (S * (M // 16) * WK < 2048 or NG <= 8):
        WK *= 2
    return 1, WK, 1, 2


def _aligned(x):
    """BF16, contiguous and 16-byte aligned (the kernels load x in 4-byte words and Y / weights in 16-byte vectors)"""
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    if not x.is_contiguous() or x.data_ptr() % 16:
        x = x.clone(memory_format=torch.contiguous_format)
    return x


def moe_gemv(tables, x, ids, M, xdiv, cfg=None):
    """y[s] = W[ids[s]] x[s // xdiv] for every slot s: x [rows, N] BF16, ids [S] int32 / int64 expert of each slot (clamped
    to [0, E) in the kernel), tables =
    (tree, escape-offset, escape-list) device pointers per expert (int64 tensors), W of expert e: M x N"""
    x = _aligned(x)
    if ids.dtype not in (torch.int32, torch.int64):
        ids = ids.to(torch.int64)
    ids = ids.reshape(-1).contiguous()                  # int32 / int64 as given (the kernel clamps to [0, E))
    y = torch.empty(ids.numel(), M, dtype=torch.bfloat16, device=x.device)
    WM, WK, GY, PF = cfg or mvx_cfg(M, x.shape[1], ids.numel())
    st = _state(x.device)
    with _device_ctx(x.device):
        ext().mvx(*tables, x, ids, y, st.ws, st.cnt, xdiv, WM, WK, GY, PF)
    return y


class PtrTables(tuple):
    """(TP, EOP, ELP) int64 device pointer tables; .keep holds the tensors they point into, so the memory stays valid
    even if a layer later rebuilds its side tables"""


def expert_tables(mods):
    """(TP, EOP, ELP) device pointer tables of a list of per-expert TreeLinear layers (escape side tables built now)"""
    dev = mods[0].tree.device
    tp, eop, elp, keep = [], [], [], []
    for m in mods:
        eo, el = m._esc_lists()
        tp.append(m.tree.data_ptr())
        eop.append(eo.data_ptr())
        elp.append(el.data_ptr())
        keep += [m.tree, eo, el]
    t = PtrTables(torch.tensor(v, dtype=torch.int64, device=dev) for v in (tp, eop, elp))
    t.keep = keep
    return t


def mvx_max_slots(num_experts):
    """(token, expert) pairs up to which the mvx kernel runs a block (one expert matrix read per pair; no alignment
    kernel). Real routing reuses experts across tokens (Qwen3.6-35B-A3B decode: 44 distinct experts for 64 pairs), so
    from 64 pairs the grouped mvg, which reads each expert once, is faster despite its alignment step (moe_kbench11)."""
    v = os.environ.get("STATETREE_MOE_MVX_MAX")
    return int(v) if v is not None else 32


def mvg_max_slots(num_experts):
    """(token, expert) pairs up to which the grouped mvg kernel runs a block (above: the per-expert loop in
    Transformers, decode-all + the BF16 MoE kernel in vLLM)"""
    v = os.environ.get("STATETREE_MOE_MVG_MAX")
    return int(v) if v is not None else 64 * num_experts


def tree_experts_mvg_max(num_experts):
    """(token, expert) pairs up to which TreeExperts (Transformers) runs the grouped mvg kernel: every batch by default
    (one Qwen3.6-35B-A3B layer on an A100 at 4096 / 8192 tokens: mvg 7.0 / 13.6 ms, the per-expert tree loop 54 / 57
    ms, BF16 grouped_mm 7.2 / 8.6 ms), unless STATETREE_MOE_MVG_MAX is set; beyond the mvg grid limit, the loop"""
    v = os.environ.get("STATETREE_MOE_MVG_MAX")
    return int(v) if v is not None else 1 << 62


def _mvg_fits(S, E, NT):
    """the mvg grid has one z-block per block of 8 NT slots (moe_align layout), at most 65535"""
    return S // (8 * NT) + min(S, E) + 1 <= 65535


def mvg_nt(S, E):
    """mma column tiles per mvg block (8 slots each) for S pairs over E experts (A100 sweep, Qwen3.6-35B-A3B TP=2
    shapes, 16 to 2048 tokens: NT = 1 up to 8 pairs per expert on average, 2 up to 40, then 4)"""
    avg = S / max(E, 1)
    return 1 if avg <= 8 else (2 if avg <= 40 else 4)


def moe_align(ids, E, bs):
    """vLLM's moe_align_block_size layout computed with torch ops (no host sync): the pair ids sorted by expert, each
    expert padded to a multiple of bs with the value S, the expert of every block of bs entries, and the number of
    used entries ([1] int32). ids: [S] expert of every pair, all < E."""
    S = ids.numel()
    dev = ids.device
    ids = ids.reshape(-1).to(torch.int64)
    counts = torch.bincount(ids, minlength=E)[:E]
    padded = (counts + bs - 1) // bs * bs
    ends = torch.cumsum(padded, 0)
    nblk = S // bs + min(S, E) + 1
    order = torch.argsort(ids, stable=True)
    sid = ids[order]
    first = torch.cumsum(counts, 0) - counts
    dest = (ends - padded)[sid] + torch.arange(S, device=dev) - first[sid]
    sorted_ids = torch.full((nblk * bs,), S, dtype=torch.int32, device=dev)
    sorted_ids[dest] = order.to(torch.int32)
    bexp = torch.searchsorted(ends, torch.arange(nblk, device=dev) * bs, right=True).clamp(max=E - 1).to(torch.int32)
    return sorted_ids, bexp, ends[-1:].to(torch.int32)


def mvg_cfg(M, N, blocks):
    """(WM, WK, GY) of the mvg kernel: four warps per 16-row tile (fewer for matrices with < 4 groups), no split-K
    across blocks: the best configuration for both expert projections at every decode size on the real routing of
    Qwen3.6-35B-A3B (moe_kbench10/11, TP = 2 shapes, batch 8-32; up to 20% faster than the 0.5.1 rule)"""
    NG = N // 64
    return 1, min(4, NG), 1


def moe_group_gemv(tables, x, aligned, S, M, xdiv, NT, E, cfg=None):
    """y[s] = W[expert of s] x[s // xdiv] for the S pairs, grouped by expert (aligned = moe_align(..., bs = 8 NT))"""
    sorted_ids, bexp, npost = (t.contiguous() for t in aligned)
    x = _aligned(x)
    y = torch.empty(S, M, dtype=torch.bfloat16, device=x.device)
    WM, WK, GY = cfg or mvg_cfg(M, x.shape[1], min(bexp.numel(), S // (8 * NT) + min(S, E)))
    st = _state(x.device)
    with _device_ctx(x.device):
        ext().mvg(*tables, x, sorted_ids, bexp, npost, y, st.ws, st.cnt, xdiv, NT, WM, WK, GY)
    return y


def gtt_ok(M, N):
    """the grouped tree GEMM needs whole 128-row blocks and 64-column k-tiles per expert matrix (M x N)"""
    return M % 128 == 0 and N % 64 == 0 and N >= 64


def gtt_enabled():
    return os.environ.get("STATETREE_MOE_GTT", "1") != "0"


def gtt_min_slots(num_experts):
    """(token, expert) pairs from which the grouped tree GEMM gtt runs a block instead of mvg: more than 8 pairs per
    expert on average (A100, one Qwen3.6-35B-A3B layer: the two tie at 256 tokens = 2048 pairs, gtt is 1.15x faster
    at 512 and 1.6x at 1024). STATETREE_MOE_GTT_MIN overrides it."""
    v = os.environ.get("STATETREE_MOE_GTT_MIN")
    return int(v) if v is not None else 8 * num_experts + 1


def gtt_cfg(S, E):
    """(TT, NWT, MINB) of gtt for S pairs over E experts: TT tokens per block (vLLM's moe_align_block_size layout with
    block size TT), NWT warp groups over the tokens, MINB blocks per SM. More pairs per expert, larger blocks: each
    block decodes its expert's 128-row tile once for TT tokens, but pads the expert's last block (A100 sweep, Qwen3.6
    shapes: 32 tokens up to 40 pairs per expert, 64 up to 192, then 128). STATETREE_MOE_GTT_CFG="TT_NWT_MINB"."""
    v = os.environ.get("STATETREE_MOE_GTT_CFG")
    if v:
        return tuple(int(c) for c in v.split("_"))
    avg = S / max(E, 1)
    return (32, 2, 2) if avg <= 40 else ((64, 2, 2) if avg <= 192 else (128, 4, 1))


def gtt_tables(t):
    """gtt escape tables (koff, ent, KS, CAP) of one TreeLinear holding E experts stacked [E * M, N] (vLLM)"""
    return gtt_escape_tables(t.out_features, t.in_features, t.esc_rows, t.esc_cols, t.esc_bits)


def stacked_gtt_tables(mods):
    """gtt escape tables of a list of same-shape TreeLinear layers (one per expert), built in one pass as those of the
    experts stacked into one [E * M, N] matrix: (koff, ent, KS, CAP), the experts' 128-row blocks one after the
    other (one host sync for all experts)"""
    M, N = mods[0].out_features, mods[0].in_features
    rows = torch.cat([m.esc_rows.long() + e * M for e, m in enumerate(mods)])
    cols = torch.cat([m.esc_cols for m in mods])
    bits = torch.cat([m.esc_bits for m in mods])
    return gtt_escape_tables(len(mods) * M, N, rows, cols, bits)


def moe_gtt(tp, tt, x, aligned, S, M, xdiv, cfg):
    """y[s] = W[expert of s] x[s // xdiv] for the S pairs with the grouped tree GEMM (csrc/gtt.cu): tp = the experts'
    record base pointers ([E] int64), tt = (koff, ent, KS, CAP) escape tables of the stacked experts, aligned =
    moe_align(ids, E, TT) or vLLM's moe_align_block_size, cfg = (TT, NWT, MINB). No workspace, no host sync,
    deterministic (whole K per tile)."""
    sorted_ids, bexp, npost = (t.contiguous() for t in aligned)
    koff, ent, KS, CAP = tt
    x = _aligned(x)
    y = torch.empty(S, M, dtype=torch.bfloat16, device=x.device)
    TT, NWT, MINB = cfg
    with _device_ctx(x.device):
        ext().gtt(TT, NWT, MINB, tp, koff, ent, x, sorted_ids, bexp, npost, y, xdiv, KS, CAP)
    return y


def mfd_max_slots(num_experts):
    """(token, expert) pairs up to which a decode step's routed-expert block runs as one fused launch (mfd: gate-up,
    silu * up, down and the routing-weighted sum). STATETREE_MOE_FUSED_MAX overrides it, 0 turns it off."""
    v = os.environ.get("STATETREE_MOE_FUSED_MAX")
    return int(v) if v is not None else 0


def moe_fused_decode(t13, t2, x, ids, w, I):
    """out[t] = sum_j w[t, j] W2[e] (silu(W13gate[e] x[t]) * W13up[e] x[t]), e = ids[t k + j], in one launch (csrc mfd):
    t13 / t2 = per-expert pointer tables of the [2I, H] gate-up (gate rows first) and [H, I] down trees, x [T, H] BF16,
    ids [T k] int32 / int64 (clamped to [0, E) in the kernel), w [T, k] routing weights, k <= 8. Deterministic."""
    x = _aligned(x)
    T, H = x.shape
    K = w.shape[-1]
    if ids.dtype not in (torch.int32, torch.int64):
        ids = ids.to(torch.int64)
    ids = ids.reshape(-1).contiguous()
    w = w.reshape(T, K)
    if w.dtype != torch.float32 or not w.is_contiguous():
        w = w.float().contiguous()
    st = _state(x.device)
    act, cnt = st.get_mfd(T * K * 2 * I, T * K + T)
    out = torch.empty(T, H, dtype=torch.bfloat16, device=x.device)
    with _device_ctx(x.device):
        ext().mfd(*t13, *t2, x, ids, w, act, cnt, out, I)
    return out


def mvw_ok(N):
    """The persistent small-K kernels (mvwx / mvwg) take matrices with an even number of 64-column groups, at most 16"""
    return N % 128 == 0 and N <= 1024


def mvw_enabled():
    """The grouped decode path runs the down projection (K = I <= 1024) with the persistent mvwg kernel instead of mvg
    when the slots are grouped 8 per block (NT = 1). A100, Qwen3.6-35B-A3B TP = 2 (w2 2048 x 256), real routing
    (moe_kbench14b): 1.13x / 1.20x / 1.37x faster than mvg at batch 8 / 16 / 32 (each warp computes whole 16-row
    tiles, no block barrier, no empty blocks). Env STATETREE_MVW=0 turns it off."""
    return os.environ.get("STATETREE_MVW", "1") != "0"


def mvw_nbps():
    """blocks of 4 warps per SM of the mvw kernels (env STATETREE_MVW_NBPS)"""
    return int(os.environ.get("STATETREE_MVW_NBPS", "4"))


def moe_gemv_w(tables, x, ids, M, xdiv, nbps=None):
    """moe_gemv for small K (the down projection): the persistent warp-per-tile mvwx kernel, same slot layout"""
    x = _aligned(x)
    if ids.dtype not in (torch.int32, torch.int64):
        ids = ids.to(torch.int64)
    ids = ids.reshape(-1).contiguous()
    y = torch.empty(ids.numel(), M, dtype=torch.bfloat16, device=x.device)
    with _device_ctx(x.device):
        ext().mvwx(*tables, x, ids, y, xdiv, nbps or mvw_nbps())
    return y


def moe_group_gemv_w(tables, x, aligned, S, M, xdiv, nbps=None):
    """moe_group_gemv for small K: the persistent mvwg kernel on moe_align(..., bs = 8) (rows of other ranks' experts
    are not written, as in mvg)"""
    sorted_ids, bexp, npost = (t.contiguous() for t in aligned)
    x = _aligned(x)
    y = torch.empty(S, M, dtype=torch.bfloat16, device=x.device)
    with _device_ctx(x.device):
        ext().mvwg(*tables, x, sorted_ids, bexp, npost, y, xdiv, nbps or mvw_nbps())
    return y

def moe_wsum(y, w):
    """out[t] = sum_j w[t, j] y[t * k + j] (fp32 sum, BF16 out); w [T, k] float32 or BF16"""
    T, K = w.shape
    out = torch.empty(T, y.shape[1], dtype=torch.bfloat16, device=y.device)
    with _device_ctx(y.device):
        ext().wsum(_aligned(y), w.contiguous(), out)
    return out


def expert_layout(mod):
    """(up name, is_transposed, has_bias) for a Transformers experts module (3-D up / down projections and the layout
    flags set by use_experts_implementation), else None. Modules without those flags are left alone: their layout
    cannot be known for sure. Transformers 5.0's decorator sets is_transposed / has_bias but not has_gate; there every
    experts module is gated with a gate_up_proj, so a missing has_gate is accepted only for that name."""
    if isinstance(mod, TreeExperts) or not hasattr(mod, "is_transposed"):
        return None
    p = mod._parameters
    up = next((n for n in UP_NAMES if isinstance(p.get(n), torch.Tensor) and p[n].dim() == 3), None)
    down = p.get("down_proj")
    if up is None or not isinstance(down, torch.Tensor) or down.dim() != 3 or down.shape[0] != p[up].shape[0]:
        return None
    if not hasattr(mod, "has_gate") and up != "gate_up_proj":
        return None
    has_bias = bool(getattr(mod, "has_bias", False))
    if has_bias and not (isinstance(p.get(up + "_bias"), torch.Tensor) and isinstance(p.get("down_proj_bias"),
                                                                                    torch.Tensor)):
        return None
    return up, bool(mod.is_transposed), has_bias


def _mat_shape(P, is_t):
    """(out, in) of one expert's matrix"""
    return (int(P.shape[2]), int(P.shape[1])) if is_t else (int(P.shape[1]), int(P.shape[2]))


def experts_eligible(mod, lay):
    """BF16, on a real device (not offloaded to meta by a device map) and every matrix out % 16 == 0, in % 64 == 0"""
    up, is_t, _ = lay
    U, D = mod._parameters[up], mod._parameters["down_proj"]
    if U.dtype != torch.bfloat16 or D.dtype != torch.bfloat16 or "meta" in (U.device.type, D.device.type):
        return False
    (uo, ui), (do, di) = _mat_shape(U, is_t), _mat_shape(D, is_t)
    return uo % 16 == 0 and ui % 64 == 0 and do % 16 == 0 and di % 64 == 0


class TreeExperts(nn.Module):
    """Drop-in for a Transformers 5 experts module, forward(hidden_states [T, H], top_k_index [T, k], top_k_weights
    [T, k]) -> [T, H], whose routed experts are state trees: '<up>.<e>' and 'down_proj.<e>' are TreeLinear layers
    (biases inside them). Flags and scalars of the source module (has_gate, act_fn, alpha, limit, post_expert_norm, ...)
    are kept, and its class's _apply_gate does the gating."""

    def __init__(self, src, up_name, num_experts, up_shape, down_shape, has_bias, device="meta"):
        super().__init__()
        for k, v in vars(src).items():                 # flags / scalars of the source class and of the decorator
            if not k.startswith("_") and k != "training" and isinstance(v, (bool, int, float, str)):
                setattr(self, k, v)
        self.config = getattr(src, "config", None)
        for n, m in src._modules.items():             # act_fn, post_expert_norm, ...
            if m is not None:
                self.add_module(n, m)
        self._gate_fn = getattr(type(src), "_apply_gate", None)
        self.src_class = f"{type(src).__module__}.{type(src).__qualname__}"
        self.up_name, self.num_experts, self.has_bias = up_name, int(num_experts), bool(has_bias)
        self.has_gate = bool(getattr(src, "has_gate", up_name == "gate_up_proj"))
        self.has_post_expert_norm = bool(getattr(src, "has_post_expert_norm", False))
        (uo, ui), (do, di) = up_shape, down_shape
        self.add_module(up_name, nn.ModuleList(TreeLinear(ui, uo, bias=has_bias, device=device)
                                               for _ in range(num_experts)))
        self.down_proj = nn.ModuleList(TreeLinear(di, do, bias=has_bias, device=device) for _ in range(num_experts))
        self._mvx = {}                                  # pointer tables / stacked biases, rebuilt when trees move

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        self._mvx = {}
        return out

    def _tables(self, which):
        mods = getattr(self, self.up_name) if which == "up" else self.down_proj
        key = (which, mods[0].tree.data_ptr(), mods[-1].tree.data_ptr())
        t = self._mvx.get(key)
        if t is None:
            t = self._mvx[key] = (expert_tables(list(mods)),
                                  torch.stack([m.bias for m in mods]) if self.has_bias else None)
        return t

    def _fused_ok(self):
        """the fused decode kernel computes silu(gate) * up from a [2I, H] gate_up_proj (gate rows first), no biases: the
        default gating (no _apply_gate, or the one Transformers' experts decorator adds), not e.g. gpt-oss's own"""
        u, d = getattr(self, self.up_name)[0], self.down_proj[0]
        act = getattr(self, "act_fn", None)
        default_gate = self._gate_fn is None or getattr(self._gate_fn, "__name__", "") == "_default_apply_gate"
        return (self.has_gate and not self.has_bias and not self.has_post_expert_norm and default_gate and
                self.up_name == "gate_up_proj" and (isinstance(act, nn.SiLU) or "SiLU" in type(act).__name__) and
                u.out_features == 2 * d.in_features and u.in_features == d.out_features and
                u.in_features % 64 == 0 and d.in_features % 64 == 0)

    def _gtt_ok(self):
        u, d = getattr(self, self.up_name)[0], self.down_proj[0]
        return gtt_enabled() and gtt_ok(u.out_features, u.in_features) and gtt_ok(d.out_features, d.in_features)

    def _gtt_for(self, S):
        return S >= gtt_min_slots(self.num_experts) and self._gtt_ok()

    def _gtt_tables(self, which):
        mods = getattr(self, self.up_name) if which == "up" else self.down_proj
        key = ("gtt", which, mods[0].tree.data_ptr(), mods[-1].tree.data_ptr())
        t = self._mvx.get(key)
        if t is None:
            t = self._mvx[key] = stacked_gtt_tables(list(mods))
        return t

    def spec(self):
        """what load_pretrained needs to rebuild this module from the model's config"""
        u, d = getattr(self, self.up_name)[0], self.down_proj[0]
        return dict(up=self.up_name, num_experts=self.num_experts, up_shape=[u.out_features, u.in_features],
                    down_shape=[d.out_features, d.in_features], bias=self.has_bias, source=self.src_class)

    @classmethod
    @torch.no_grad()
    def from_experts(cls, src, verify=True):
        """Encode every expert of a Transformers experts module (see expert_layout); the source module's 3-D BF16
        tensors are released as soon as their trees exist."""
        lay = expert_layout(src)
        assert lay is not None and experts_eligible(src, lay), "TreeExperts: not a supported BF16 experts module"
        up, is_t, has_bias = lay
        p = src._parameters
        E = int(p[up].shape[0])
        self = cls(src, up, E, _mat_shape(p[up], is_t), _mat_shape(p["down_proj"], is_t), has_bias, device="meta")
        for name, dst in ((up, getattr(self, up)), ("down_proj", self.down_proj)):
            P = p[name]
            B = p.get(name + "_bias") if has_bias else None
            for e in range(E):
                W = P[e].t().contiguous() if is_t else P[e]
                dst[e] = TreeLinear.from_weight(W, None if B is None else B[e], verify=verify)
            del P, B, W
            p.pop(name)                                # free the BF16 experts tensor now (peak: one tensor's trees)
            p.pop(name + "_bias", None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return self

    @classmethod
    def skeleton(cls, src, spec):
        """empty (meta) TreeExperts in place of src, filled by load_state_dict"""
        return cls(src, spec["up"], spec["num_experts"], spec["up_shape"], spec["down_shape"], spec["bias"],
                   device="meta")

    def storage_bytes(self):
        return sum(m.storage_bytes() for m in self.modules() if isinstance(m, TreeLinear))

    def _apply_gate(self, x):
        if self._gate_fn is not None:
            return self._gate_fn(self, x)
        gate, up = x.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    def forward(self, hidden_states, top_k_index, top_k_weights):
        K = top_k_index.size(-1)
        T = hidden_states.size(0)
        ids = top_k_index.reshape(-1)
        w = top_k_weights.reshape(-1)
        S, E = ids.numel(), self.num_experts
        if S > 0 and (S <= mvx_max_slots(E) or self._gtt_for(S)
                      or (S <= tree_experts_mvg_max(E) and _mvg_fits(S, E, mvg_nt(S, E)))):
            return self._forward_mvx(hidden_states, ids, w, T, K)
        ids_g, perm = torch.sort(ids)                  # pairs grouped by expert (expert-parallel sentinels last)
        x = hidden_states[perm // K]
        counts = torch.bincount(ids_g, minlength=self.num_experts + 1)[:self.num_experts].tolist()
        ups, downs = getattr(self, self.up_name), self.down_proj
        out = hidden_states.new_zeros(x.shape[0], downs[0].out_features)   # sentinel rows stay 0
        s = 0
        for e, n in enumerate(counts):
            if n:
                h = ups[e](x[s:s + n])
                h = self._apply_gate(h) if self.has_gate else self.act_fn(h)
                out[s:s + n] = downs[e](h)
                s += n
        if self.has_post_expert_norm:
            out = self.post_expert_norm(out)
        out = out * w[perm].unsqueeze(-1)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=perm.device)
        return out[inv].view(T, K, -1).sum(dim=1).to(hidden_states.dtype)

    def _forward_mvx(self, hidden_states, ids, w, T, K):
        """every (token, expert) pair's products straight from the trees, one launch per projection (mvx for few
        pairs, the grouped mvg up to 8 pairs per expert, the grouped tree GEMM gtt beyond)"""
        E, S = self.num_experts, ids.numel()
        ids64 = ids.to(torch.int64)
        valid = ids64 < E                                    # expert-parallel sentinels contribute nothing
        ids_c = ids64.clamp(max=E - 1)
        (tu, bu), (td, bd) = self._tables("up"), self._tables("down")
        ups, downs = getattr(self, self.up_name), self.down_proj
        if K <= 8 and S <= mfd_max_slots(E) and self._fused_ok():    # one launch; sentinel pairs get weight 0
            return moe_fused_decode(tu, td, hidden_states, ids_c, (w * valid).view(T, K),
                                    downs[0].in_features).to(hidden_states.dtype)
        run_d = None                                         # the down projection, when it runs another kernel
        if S <= mvx_max_slots(E):
            run = lambda tab, x, M, xdiv: moe_gemv(tab, x, ids_c, M, xdiv)                       # noqa: E731
        elif self._gtt_for(S):
            cfg = gtt_cfg(S, E)
            al = moe_align(ids_c, E, cfg[0])
            gt = {id(tu): self._gtt_tables("up"), id(td): self._gtt_tables("down")}
            run = lambda tab, x, M, xdiv: moe_gtt(tab[0], gt[id(tab)], x, al, S, M, xdiv, cfg)  # noqa: E731
        else:
            NT = mvg_nt(S, E)
            al = moe_align(ids_c, E, 8 * NT)
            run = lambda tab, x, M, xdiv: moe_group_gemv(tab, x, al, S, M, xdiv, NT, E)          # noqa: E731
            if NT == 1 and mvw_enabled() and mvw_ok(downs[0].in_features):
                run_d = lambda tab, x, M, xdiv: moe_group_gemv_w(tab, x, al, S, M, xdiv)          # noqa: E731
        h = run(tu, hidden_states, ups[0].out_features, K)
        if bu is not None:
            h = h + bu[ids_c]
        h = self._apply_gate(h) if self.has_gate else self.act_fn(h)
        out = (run_d or run)(td, h, downs[0].out_features, 1)
        if bd is not None:
            out = out + bd[ids_c]
        if self.has_post_expert_norm:
            out = self.post_expert_norm(out)
        if out.dtype == torch.bfloat16 and out.shape[1] % 8 == 0:
            return moe_wsum(out, (w * valid).view(T, K)).to(hidden_states.dtype)
        out = out * (w * valid).unsqueeze(-1)
        return out.view(T, K, -1).sum(dim=1).to(hidden_states.dtype)

    def extra_repr(self):
        u, d = getattr(self, self.up_name)[0], self.down_proj[0]
        return (f"{self.num_experts} experts as state trees: {self.up_name} {u.out_features}x{u.in_features}, "
                f"down_proj {d.out_features}x{d.in_features}, gate={self.has_gate}, bias={self.has_bias}")
