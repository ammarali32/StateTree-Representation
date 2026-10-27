"""vLLM internals for '--representation statetree' (imported by statetree.vllm_plugin.register inside vLLM processes)."""
import os
import weakref

import torch

from .dense16 import Dense16Linear
from . import format as _F
from .linear import _REGISTRY, BUCKETS, MAX_ESCAPE_FRACTION, EscapeHeavy, TreeLinear

PLAN_BATCHES = tuple(BUCKETS) + (1024,)          # every CUDA-graph batch bucket, plus the large-batch (prefill) path
_INSTALLED = False
_ANNOUNCED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    _register_kv_backend()
    _patch_engine_args()
    _patch_linear_layers()
    if os.environ.get("STATETREE_VLLM_EMBEDDINGS", "1") != "0":
        _patch_embeddings()
    if os.environ.get("STATETREE_VLLM_MOE", "1") != "0":
        _patch_moe()


# ----------------------------------------------------------------------------------------- the flag
def _check(rep):
    from .vllm_plugin import REPRESENTATIONS
    if rep not in REPRESENTATIONS:
        raise ValueError(f"representation={rep!r}: expected one of {REPRESENTATIONS}")
    return rep


def _set_representation(engine_args, rep):
    cfg = dict(engine_args.additional_config or {})
    cfg["representation"] = _check(rep)
    engine_args.additional_config = cfg


KV_BACKEND = "statetree.vllm_kv.StateTreeKVBackend"


def _register_kv_backend():
    """the state-tree KV cache is vLLM's CUSTOM attention backend (selected by --kv-representation statetree); only a
    class path is registered here, the backend module is imported when vLLM selects it"""
    from vllm.v1.attention.backends import registry as R
    if R.AttentionBackendEnum.CUSTOM not in R._ATTN_OVERRIDES:
        R.register_backend(R.AttentionBackendEnum.CUSTOM, KV_BACKEND)


def _set_kv_representation(engine_args, rep):
    if rep not in ("bf16", "statetree"):
        raise ValueError(f"kv_representation={rep!r}: expected 'bf16' or 'statetree'")
    if rep == "statetree":
        from vllm.v1.attention.backends.registry import AttentionBackendEnum as E
        cur = getattr(engine_args, "attention_backend", None)
        if cur not in (None, E.CUSTOM) and str(cur).upper() not in ("CUSTOM", "ATTENTIONBACKENDENUM.CUSTOM"):
            raise ValueError("--kv-representation statetree selects the attention backend itself: drop --attention-backend")
        engine_args.attention_backend = E.CUSTOM
        if engine_args.block_size is None:
            engine_args.block_size = 16
        elif engine_args.block_size != 16:
            raise ValueError("--kv-representation statetree needs --block-size 16")
    cfg = dict(engine_args.additional_config or {})
    cfg["kv_representation"] = rep
    engine_args.additional_config = cfg


def _patch_engine_args():
    from vllm.engine import arg_utils as au
    EA = au.EngineArgs

    orig_add = EA.add_cli_args

    def add_cli_args(parser, *args, **kwargs):
        parser = orig_add(parser, *args, **kwargs)
        if not any("--representation" in a.option_strings for a in parser._actions):
            parser.add_argument(
                "--representation", type=str, default=None, choices=["bf16", "statetree", "dense16"],
                help="Weight representation (statetree plugin). 'statetree' stores every BF16 linear layer losslessly "
                     "as a state tree (~12.1 bits/weight, decoded inside fused kernels, bit-exact weights); 'bf16' "
                     "(default) keeps the dense weights; 'dense16' (ablation control) runs the statetree kernels on plain BF16. Independent of --quantization.")
        if not any("--kv-representation" in a.option_strings for a in parser._actions):
            parser.add_argument(
                "--kv-representation", type=str, default=None, choices=["bf16", "statetree"],
                help="KV-cache representation (statetree plugin). 'statetree' stores the paged KV cache losslessly as "
                     "state-tree pages (6400 B instead of 8192 B per 16 tokens per KV head: 1.28x more tokens in the "
                     "same memory, exact BF16 values) with its own decode-attention kernel; selects the CUSTOM "
                     "attention backend and block size 16. Independent of --representation.")
        return parser
    EA.add_cli_args = staticmethod(add_cli_args)

    orig_from = EA.from_cli_args.__func__

    def from_cli_args(cls, args):
        engine_args = orig_from(cls, args)
        rep = getattr(args, "representation", None)
        if rep:
            _set_representation(engine_args, rep)
        kv = getattr(args, "kv_representation", None)
        if kv:
            _set_kv_representation(engine_args, kv)
        return engine_args
    EA.from_cli_args = classmethod(from_cli_args)

    for C in {EA, getattr(au, "AsyncEngineArgs", EA)}:
        def make(orig_init):
            def __init__(self, *args, representation=None, kv_representation=None, **kwargs):
                orig_init(self, *args, **kwargs)
                if representation:
                    _set_representation(self, representation)
                if kv_representation:
                    _set_kv_representation(self, kv_representation)
            return __init__
        C.__init__ = make(C.__init__)


def representation():
    """The representation requested for the model being built (from the current vLLM config, or the env var)."""
    rep = None
    try:
        from vllm.config import get_current_vllm_config_or_none
        cfg = get_current_vllm_config_or_none()
    except ImportError:                                                     # older vLLM
        try:
            from vllm.config import get_current_vllm_config
            cfg = get_current_vllm_config()
        except Exception:                                                   # noqa: BLE001
            cfg = None
    extra = getattr(cfg, "additional_config", None)
    if isinstance(extra, dict):
        rep = extra.get("representation")
    return _check(rep or os.environ.get("VLLM_REPRESENTATION", "bf16"))


# ----------------------------------------------------------------------------------------- the layers
from vllm.model_executor.layers import linear as _L  # noqa: E402  (only imported inside vLLM processes)


def _patch_linear_layers():
    orig_init = _L.LinearBase.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        rep = representation()
        if type(getattr(self, "quant_method", None)) is _L.UnquantizedLinearMethod and rep != "bf16":
            self.quant_method = StateTreeLinearMethod(getattr(self, "prefix", ""), rep)
    _L.LinearBase.__init__ = __init__


def _unique_handle(prefix):
    name = f"vllm:{prefix}"
    k = 1
    while name in _REGISTRY:
        k += 1
        name = f"vllm:{prefix}#{k}"
    return name


def _supported(w):
    return w.dim() == 2 and w.dtype == torch.bfloat16 and w.is_cuda and w.shape[0] % 16 == 0 and w.shape[1] % 64 == 0


def min_params():
    """linear layers with fewer weights stay BF16 (STATETREE_MIN_PARAMS, default 2^21; 0 converts every layer): a tree
    kernel's fixed cost (a few µs) exceeds the time it takes to read such a matrix, so trees run at half cuBLAS speed
    there while saving little memory (Qwen3.6-35B-A3B at TP = 2: routers, shared experts and the linear-attention ba
    projections, 150 matrices of <= 1M weights per decode step, 0.7 ms per step slower as trees; about 40 MB per GPU)"""
    return int(os.environ.get("STATETREE_MIN_PARAMS", str(1 << 21)))


def _announce(verify):
    global _ANNOUNCED
    if not _ANNOUNCED:
        _ANNOUNCED = True
        from vllm.logger import init_logger
        init_logger("vllm.statetree").info("statetree: storing BF16 linear layers as lossless state trees (verify=%s)",
                                      verify)


def _encode(layer, prefix, verify, rep="statetree"):
    """layer.weight (BF16, loaded by vLLM) -> TreeLinear registered as layer.statetree; the BF16 weight is freed."""
    _announce(verify)
    w = layer.weight.data
    if rep == "dense16":
        tl = Dense16Linear.from_weight(w, None, verify=verify)
    else:                                         # EscapeHeavy: the caller keeps this layer in BF16
        tl = TreeLinear.from_weight(w, None, verify=verify, max_escape_fraction=MAX_ESCAPE_FRACTION)
    tl.set_handle(_unique_handle(prefix))
    _release(layer, tl)
    return tl


def _heavy(prefix, why):
    from vllm.logger import init_logger
    init_logger("vllm.statetree").warning("statetree: %s stays BF16 (escape-heavy: %s)", prefix, why)


def _release(layer, tl):
    w = layer.weight
    layer.weight = torch.nn.Parameter(torch.empty(0, dtype=w.dtype, device=w.device), requires_grad=False)
    layer.register_module("statetree", tl)


def _prepare(tl, batches):
    """side tables, kernel plans (autotuned once per shape, cached) and workspaces for every batch bucket: nothing is
    allocated, tuned or synchronised later inside vLLM's CUDA-graph capture"""
    tl.prepare(batches)
    with torch.no_grad():
        for b in batches:
            tl._forward(torch.zeros(b, tl.in_features, dtype=torch.bfloat16, device=tl.tree.device))


class StateTreeLinearMethod(_L.UnquantizedLinearMethod):
    """Loads the BF16 weight exactly like the unquantized method (tensor-parallel shards, merged q/k/v and gate/up),
    then replaces it by a TreeLinear ('layer.statetree') that apply() runs."""

    def __init__(self, prefix="", rep="statetree"):
        super().__init__()
        self.prefix, self.rep = prefix, rep
        self.verify = os.environ.get("STATETREE_VLLM_VERIFY", "1") != "0"

    def process_weights_after_loading(self, layer):
        w = layer.weight.data
        if not _supported(w) or w.numel() < min_params():   # conv weights held by a linear layer, tiny matrices
            layer.statetree = None
            return super().process_weights_after_loading(layer)
        try:
            tl = _encode(layer, self.prefix, self.verify, self.rep)
        except EscapeHeavy as e:                       # its tree would be larger and slower than BF16
            _heavy(self.prefix, e)
            layer.statetree = None
            return super().process_weights_after_loading(layer)
        _prepare(tl, PLAN_BATCHES)

    def apply(self, layer, x, bias=None):
        tl = getattr(layer, "statetree", None)
        if tl is None:
            return super().apply(layer, x, bias)
        y = torch.ops.statetree.linear(x, tl.tree, tl._handle)
        return y if bias is None else y + bias


# ----------------------------------------------------------------------------------------- embeddings and lm_head
from vllm.model_executor.layers import vocab_parallel_embedding as _V  # noqa: E402

_TIED = []                                       # (weakref to a BF16 table Parameter, its TreeLinear)


def _patch_embeddings():
    orig_init = _V.VocabParallelEmbedding.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        rep = representation()
        if type(getattr(self, "quant_method", None)) is _V.UnquantizedEmbeddingMethod and rep != "bf16":
            self.quant_method = StateTreeEmbeddingMethod(getattr(self, "prefix", ""),
                                                         lm_head=isinstance(self, _V.ParallelLMHead), rep=rep)
    _V.VocabParallelEmbedding.__init__ = __init__


class StateTreeEmbeddingMethod(_V.UnquantizedEmbeddingMethod):
    """Input embedding (rows decoded on lookup) and lm_head (the tree GEMV/GEMM) stored as one state tree; a tied
    embedding / lm_head pair shares a single tree."""

    def __init__(self, prefix="", lm_head=False, rep="statetree"):
        super().__init__()
        self.prefix, self.lm_head, self.rep = prefix, lm_head, rep
        self.verify = os.environ.get("STATETREE_VLLM_VERIFY", "1") != "0"

    def process_weights_after_loading(self, layer):
        p = layer.weight
        if not _supported(p.data):
            layer.statetree = None
            return super().process_weights_after_loading(layer)
        tl = next((t for r, t in _TIED if r() is p), None)
        if tl is None:
            try:
                tl = _encode(layer, self.prefix, self.verify, self.rep)
            except EscapeHeavy as e:
                _heavy(self.prefix, e)
                layer.statetree = None
                return super().process_weights_after_loading(layer)
            _TIED.append((weakref.ref(p), tl))
        else:
            _release(layer, tl)
        del p
        tl.allow_dec = False                         # a vocab-sized table: no M*N decode scratch for batches <= 512
        # vLLM's fused TP > 1 lookup (ops.vocab_parallel_embedding) reads layer.weight directly: take the masked
        # quant_method.embedding path instead (out-of-shard rows are zeroed, then all-reduced)
        layer.use_fused_embedding = False
        # plans for every CUDA-graph bucket (the lm_head batch is the number of sequences sampled per step); also for a
        # plain embedding, since some models reuse the embedding module itself as their lm_head
        _prepare(tl, tuple(BUCKETS))

    def apply(self, layer, x, bias=None):
        tl = getattr(layer, "statetree", None)
        if tl is None:
            return super().apply(layer, x, bias)
        y = torch.ops.statetree.linear(x, tl.tree, tl._handle)
        return y if bias is None else y + bias

    def embedding(self, layer, input_):
        tl = getattr(layer, "statetree", None)
        if tl is None:
            return super().embedding(layer, input_)
        return torch.ops.statetree.rows(input_, tl.tree, tl._handle)


# ----------------------------------------------------------------------------------------- MoE experts (FusedMoE)
_MOE_SCRATCH = {}                                # device index -> BF16 buffer of the decode-all path (shared by layers)


def _moe_scratch(dev, n):
    b = _MOE_SCRATCH.get(dev.index)
    if b is None or b.numel() < n:
        b = _MOE_SCRATCH[dev.index] = torch.empty(int(n), dtype=torch.bfloat16, device=dev)
    return b


def _stacked_tables(t, E, M, N):
    """mvx pointer tables of a stacked [E * M, N] tree: expert e's records and escape offsets start e * (M/16)(N/64)
    records / entries in; the escape list is shared (its offsets are absolute)"""
    from .experts import PtrTables
    from .format import RECORD_BYTES
    eo, el = t._esc_lists()
    rec = (M // 16) * (N // 64)
    e = torch.arange(E, dtype=torch.int64, device=t.tree.device)
    tabs = PtrTables((t.tree.data_ptr() + e * (rec * RECORD_BYTES), eo.data_ptr() + e * (rec * eo.element_size()),
                      torch.full_like(e, el.data_ptr())))
    tabs.keep = [t.tree, eo, el]                   # the pointed-to memory stays valid whatever happens to t's caches
    return tabs


def _make_moe_method(Base):
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder

    from .experts import (gtt_cfg, gtt_enabled, gtt_min_slots, gtt_ok, gtt_tables, mfd_max_slots, moe_align,
                          moe_fused_decode, moe_gemv, moe_group_gemv, moe_group_gemv_w, moe_gtt, moe_wsum,
                          mvg_max_slots, mvg_nt, mvw_enabled, mvw_ok, mvx_max_slots)
    from .ops import ext
    try:
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    except ImportError:
        moe_align_block_size = None

    def _align(ids, E, bs):
        """ids: [S] int32 / int64 (vLLM's alignment skips ids outside [0, E); the torch fallback needs them clamped)"""
        if moe_align_block_size is not None:
            s, b, n = moe_align_block_size(ids.view(-1, 1), bs, E)
            return s.to(torch.int32), b.to(torch.int32), n.to(torch.int32)
        return moe_align(ids.clamp(0, E - 1), E, bs)

    class StateTreeFusedMoEMethod(Base):
        """Loads the routed experts exactly like the unquantized method (tensor-parallel shards, expert parallelism),
        then stores w13 [E, 2I, H] and w2 [E, H, I] as two stacked state trees (exact BF16 bits, the BF16 copies are
        freed). The products run straight from the trees at every batch size (CUDA-graph safe; shared experts run
        where the MoE kernel would run them): mvx for few (token, expert) pairs, the grouped mvg on vLLM's
        moe_align_block_size layout up to 8 pairs per expert, the grouped tree GEMM gtt beyond. Only when gtt does
        not fit the shapes (2I or H not a multiple of 128) or STATETREE_MOE_GTT=0, batches above mvg_max_slots decode
        all experts into one BF16 scratch shared by the MoE layers and run vLLM's MoE kernel on it."""

        def __init__(self, moe, prefix=""):
            super().__init__(moe)
            self.st_prefix = prefix
            self.st_verify = os.environ.get("STATETREE_VLLM_VERIFY", "1") != "0"

        def process_weights_after_loading(self, layer):
            super().process_weights_after_loading(layer)          # backend format and the BF16 MoE kernel
            layer.statetree_moe = None
            w13, w2 = layer.w13_weight.data, layer.w2_weight.data
            if self.is_monolithic or not all(w.dim() == 3 and w.dtype == torch.bfloat16 and w.is_cuda for w in (w13, w2)):
                return
            E, N1, H = w13.shape
            I = w2.shape[2]
            if w2.shape[:2] != (E, H) or N1 % 16 or H % 64 or I % 64:
                return
            frac = max(_F.escape_count(w) / w.numel() for w in (w13, w2))
            if frac > MAX_ESCAPE_FRACTION:                         # e.g. a few MoE layers with a very wide exponent
                _heavy(self.st_prefix, f"{frac:.2%} of the expert weights escape")      # range: BF16 is smaller
                return
            _announce(self.st_verify)
            t13 = TreeLinear.from_weight(w13.reshape(E * N1, H), verify=self.st_verify)
            t2 = TreeLinear.from_weight(w2.reshape(E * H, I), verify=self.st_verify)
            dev = w13.device
            for t, n in ((t13, "w13"), (t2, "w2")):
                t.set_handle(_unique_handle(f"{self.st_prefix}.{n}"))
                t.allow_dec = False
            for n in ("w13_weight", "w2_weight"):
                setattr(layer, n, torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16, device=dev), requires_grad=False))
            del w13, w2
            layer.register_module("statetree_w13", t13)
            layer.register_module("statetree_w2", t2)
            act = getattr(layer, "activation", None)
            backend = getattr(getattr(self, "unquantized_backend", None), "name", "")
            layer.statetree_moe = dict(
                E=E, N1=N1, H=H, I=I, t13=_stacked_tables(t13, E, N1, H), t2=_stacked_tables(t2, E, H, I),
                ptrs=(t13.tree.data_ptr(), t2.tree.data_ptr()),
                mvx=(str(getattr(act, "value", act)) == "silu" and not self.moe.has_bias and backend == "TRITON"
                     and not getattr(layer, "apply_router_weight_on_input", False)))
            info = layer.statetree_moe
            info["fused"] = info["mvx"] and N1 == 2 * I and H % 64 == 0 and I % 64 == 0
            info["gtt"] = info["mvx"] and gtt_enabled() and gtt_ok(N1, H) and gtt_ok(H, I)
            if info["gtt"]:                                        # escape tables now, not inside a graph capture
                info["tt13"], info["tt2"] = gtt_tables(t13), gtt_tables(t2)
            torch.cuda.empty_cache()                               # (the decode-all scratch is allocated in the
                                                                   # profiling run, which vLLM counts before it sizes
                                                                   # the KV cache)

        def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
            info = getattr(layer, "statetree_moe", None)
            if info is None:
                return super().apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
            E, N1, H, I = info["E"], info["N1"], info["H"], info["I"]
            t13, t2 = layer.statetree_w13, layer.statetree_w2
            if (t13.tree.data_ptr(), t2.tree.data_ptr()) != info["ptrs"]:      # trees moved (e.g. .to()): new tables
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(f"statetree: {self.st_prefix} trees moved during CUDA-graph capture")
                info["t13"], info["t2"] = _stacked_tables(t13, E, N1, H), _stacked_tables(t2, E, H, I)
                info["ptrs"] = (t13.tree.data_ptr(), t2.tree.data_ptr())
                if info["gtt"]:
                    info["tt13"], info["tt2"] = gtt_tables(t13), gtt_tables(t2)
            T, k = topk_ids.shape
            S = T * k
            if info["mvx"] and layer.expert_map is None and S > 0 and (
                    info["gtt"] or S <= max(mvx_max_slots(E), mvg_max_slots(E))):
                # the router's int32 ids as they are, no extra kernels: mvx clamps them to [0, E) itself and
                # vLLM's alignment (mvg / gtt) skips ids outside it (the dummy batches vLLM runs while profiling /
                # capturing CUDA graphs need not route to valid experts)
                ids = topk_ids.reshape(-1)
                if ids.dtype not in (torch.int32, torch.int64) or not ids.is_contiguous():
                    ids = ids.to(torch.int64).contiguous()
                if info["fused"] and k <= 8 and S <= mfd_max_slots(E):    # the whole block in one launch
                    out = moe_fused_decode(info["t13"], info["t2"], x, ids, topk_weights.view(T, k), I).to(x.dtype)
                    if shared_experts is not None:
                        shared_experts(shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
                    return out
                run2 = None                                       # the down projection, when another kernel runs it
                if S <= mvx_max_slots(E):
                    run = lambda w, v, M, xdiv: moe_gemv(info["t" + w], v, ids, M, xdiv)            # noqa: E731
                elif info["gtt"] and (S >= gtt_min_slots(E) or S > mvg_max_slots(E)):
                    cfg = gtt_cfg(S, E)
                    al = _align(ids, E, cfg[0])
                    run = lambda w, v, M, xdiv: moe_gtt(info["t" + w][0], info["tt" + w], v, al, S, M, xdiv,  # noqa: E731
                                                        cfg)
                else:
                    NT = mvg_nt(S, E)
                    al = _align(ids, E, 8 * NT)
                    run = lambda w, v, M, xdiv: moe_group_gemv(info["t" + w], v, al, S, M, xdiv, NT, E)  # noqa: E731
                    if NT == 1 and mvw_enabled() and mvw_ok(I):         # the down projection: persistent mvwg
                        run2 = lambda v: moe_group_gemv_w(info["t2"], v, al, S, H, 1)                # noqa: E731
                h = run("13", x, N1, k)
                a = torch.empty(S, N1 // 2, dtype=h.dtype, device=h.device)
                try:
                    torch.ops._C.silu_and_mul(a, h)
                except (AttributeError, RuntimeError):
                    a = torch.nn.functional.silu(h[:, :N1 // 2]) * h[:, N1 // 2:]
                y = run2(a) if run2 is not None else run("2", a, H, 1)
                out = moe_wsum(y, topk_weights.view(T, k)).to(x.dtype)
                if shared_experts is not None:                    # runs only where the MoE kernel would run it
                    shared_experts(shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
                return out
            buf = _moe_scratch(x.device, E * N1 * H + E * H * I)
            w1 = buf[:E * N1 * H].view(E, N1, H)
            w2 = buf[E * N1 * H:E * N1 * H + E * H * I].view(E, H, I)
            for t, w in ((layer.statetree_w13, w1), (layer.statetree_w2, w2)):
                eo, el = t._esc_lists()
                ext().dec(t.tree, eo, el, w.view(-1, w.shape[-1]))
            return self.moe_kernel.apply(
                hidden_states=x, w1=w1, w2=w2, topk_weights=topk_weights, topk_ids=topk_ids,
                activation=layer.activation, apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=layer.global_num_experts, expert_map=layer.expert_map,
                shared_experts=shared_experts, shared_experts_input=shared_experts_input)

    return StateTreeFusedMoEMethod


StateTreeFusedMoEMethod = None


def _patch_moe():
    """--representation statetree also stores the routed experts of vLLM's FusedMoE layers as state trees"""
    global StateTreeFusedMoEMethod
    try:
        from vllm.model_executor.layers.fused_moe import routed_experts as _R
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
        StateTreeFusedMoEMethod = _make_moe_method(UnquantizedFusedMoEMethod)
    except (ImportError, AttributeError):                 # other vLLM versions: the experts stay BF16
        return
    orig = _R.RoutedExperts._get_quant_method

    def _get_quant_method(self, prefix, quant_config, moe_config):
        qm = orig(self, prefix, quant_config, moe_config)
        if type(qm) is UnquantizedFusedMoEMethod and representation() == "statetree":
            qm = StateTreeFusedMoEMethod(moe_config, prefix)
        return qm
    _R.RoutedExperts._get_quant_method = _get_quant_method
