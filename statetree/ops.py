"""Loads the CUDA extension: the prebuilt 'statetree._C' (pip install) or a JIT build of csrc/ as a fallback."""
import os

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        try:
            from . import _C as mod
            if getattr(mod, "ABI", 0) < 53:             # a prebuilt extension from before 0.5.2 (mvw)
                raise ImportError("stale statetree._C")
        except ImportError:
            from torch.utils.cpp_extension import load
            src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
            import torch
            tv = torch.__version__.split("+")[0].replace(".", "_")             # one build per torch version (venvs)
            mod = load(name=f"statetree_C_0_5_2b_t{tv}",
                       sources=[os.path.join(src, f) for f in ("kernels.cu", "tt.cu", "gtt.cu", "bindings.cpp")],
                       extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-std=c++17"],
                       verbose=os.environ.get("STATETREE_VERBOSE", "0") == "1")
        _EXT = mod
    return _EXT


_TGX = None
_TGX_TRIED = False


def cutlass_include():
    """CUTLASS headers for the large-batch kernel: STATETREE_CUTLASS_INCLUDE, else the copy bundled with flashinfer."""
    cand = [os.environ.get("STATETREE_CUTLASS_INCLUDE", "")]
    try:
        import importlib.util
        spec = importlib.util.find_spec("flashinfer")
        if spec is not None and spec.origin:
            cand.append(os.path.join(os.path.dirname(spec.origin), "data", "cutlass", "include"))
    except Exception:                                                           # noqa: BLE001
        pass
    for c in cand:
        if c and os.path.exists(os.path.join(c, "cutlass", "cutlass.h")):
            return c
    return None


def tgx_ext():
    """JIT build of csrc/tgx.cu (large-batch GEMM with a concurrent decode kernel). None when CUTLASS headers are
    missing, the build fails, or STATETREE_TGX=0; callers then use decode + cuBLAS."""
    global _TGX, _TGX_TRIED
    if _TGX_TRIED:
        return _TGX
    _TGX_TRIED = True
    if os.environ.get("STATETREE_TGX", "0") != "1":                 # opt-in: slower than decode + cuBLAS on A100
        return None
    inc = cutlass_include()
    if inc is None:
        return None
    try:
        import torch
        from torch.utils.cpp_extension import load
        major, minor = torch.cuda.get_device_capability()
        if major < 8:
            return None
        arch = f"{major}{minor}"
        tv = torch.__version__.split("+")[0].replace(".", "_")             # one build per torch version (venvs)
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "tgx.cu")
        _TGX = load(name=f"statetree_tgx_sm{arch}_t{tv}", sources=[src], extra_include_paths=[inc],
                    extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}",
                                       "--expt-relaxed-constexpr", "-std=c++17", "-maxrregcount=224"],
                    verbose=os.environ.get("STATETREE_VERBOSE", "0") == "1")
    except Exception as e:                                                      # noqa: BLE001
        import warnings
        warnings.warn(f"statetree: large-batch kernel unavailable ({type(e).__name__}: {e}); using decode + cuBLAS")
        _TGX = None
    return _TGX


_KV = None


def kv_ext():
    """The state-tree KV-cache extension (csrc/kvcache.cu: page append, page decode, decode attention): the prebuilt
    'statetree._kv' (pip install) or a JIT build of csrc/kvcache.cu as a fallback."""
    global _KV
    if _KV is None:
        try:
            from . import _kv as mod
        except ImportError:
            import torch
            from torch.utils.cpp_extension import load
            major, minor = torch.cuda.get_device_capability()
            arch = f"{major}{minor}"
            tv = torch.__version__.split("+")[0].replace(".", "_")
            src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "kvcache.cu")
            mod = load(name=f"statetree_kvcache_sm{arch}_t{tv}", sources=[src],
                       extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-std=c++17"],
                       verbose=os.environ.get("STATETREE_VERBOSE", "0") == "1")
        _KV = mod
    return _KV
