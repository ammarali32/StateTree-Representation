import os

from setuptools import find_packages, setup

ext_modules, cmdclass = [], {}
if os.environ.get("STATETREE_NO_EXT", "0") != "1":
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        if not torch.cuda.is_available():
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0a;9.0+PTX"  # sm_80+ (cp.async, bf16 mma); 9.0a: th
        else:                                       # the GPUs present; Hopper as 9.0a (the th kernel needs wgmma)
            caps = sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
            os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(f"{a}.{b}a" if a == 9 else f"{a}.{b}" for a, b in caps)
    flags = {"cxx": ["-O3", "-std=c++17"], "nvcc": ["-O3", "-std=c++17"]}
    ext_modules = [
        CUDAExtension("statetree._C",                                        # weight kernels (mv, tm, tt, dec, rows)
                      ["statetree/csrc/kernels.cu", "statetree/csrc/tt.cu", "statetree/csrc/gtt.cu",
                       "statetree/csrc/hop.cu", "statetree/csrc/bindings.cpp"],
                      extra_compile_args=flags),
        CUDAExtension("statetree._kv", ["statetree/csrc/kvcache.cu"],      # KV-cache pages + decode attention
                      extra_compile_args=flags),
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="statetree",
    version="0.5.3",
    description="Lossless state-tree BF16 weights (12.1 bits/weight) and KV cache (12.5 bits/value) with fused CUDA "
                "kernels for PyTorch, Hugging Face Transformers and vLLM",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Ammar Ali",
    license="Apache-2.0",
    packages=find_packages(exclude=("tests", "examples")),
    package_data={"statetree": ["csrc/*.cu", "csrc/*.cpp"]},
    python_requires=">=3.9",
    install_requires=["torch>=2.1", "safetensors"],
    extras_require={"hf": ["transformers>=5.20"], "vllm": ["vllm>=0.10"]},      # tested: Transformers 5.21.0, 5.20.2
    entry_points={"vllm.general_plugins": ["statetree = statetree.vllm_plugin:register"]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
