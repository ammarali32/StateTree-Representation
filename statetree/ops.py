"""Loads the CUDA extension: the prebuilt 'statetree._C' (pip install) or a JIT build of csrc/ as a fallback."""
import os

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        try:
            from . import _C as mod
        except ImportError:
            from torch.utils.cpp_extension import load
            src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
            mod = load(name="statetree_C", sources=[os.path.join(src, "kernels.cu"), os.path.join(src, "bindings.cpp")],
                       extra_cflags=["-O3"], extra_cuda_cflags=["-O3"],
                       verbose=os.environ.get("STATETREE_VERBOSE", "0") == "1")
        _EXT = mod
    return _EXT
