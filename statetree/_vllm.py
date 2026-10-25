"""vLLM internals for '--representation statetree' (imported by statetree.vllm_plugin.register inside vLLM processes)."""
import os
import weakref

import torch

from .dense16 import Dense16Linear
from .linear import _REGISTRY, BUCKETS, TreeLinear

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
    tl = (Dense16Linear if rep == "dense16" else TreeLinear).from_weight(w, None, verify=verify)
    tl.set_handle(_unique_handle(prefix))
    _release(layer, tl)
    return tl


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
        if not _supported(layer.weight.data):          # e.g. conv weights held by a linear layer stay BF16
            layer.statetree = None
            return super().process_weights_after_loading(layer)
        _prepare(_encode(layer, self.prefix, self.verify, self.rep), PLAN_BATCHES)

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
            tl = _encode(layer, self.prefix, self.verify, self.rep)
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
