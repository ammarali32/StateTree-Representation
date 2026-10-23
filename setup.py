import os

from setuptools import find_packages, setup

ext_modules, cmdclass = [], {}
if os.environ.get("STATETREE_NO_EXT", "0") != "1":
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    if "TORCH_CUDA_ARCH_LIST" not in os.environ and not torch.cuda.is_available():
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0+PTX"         # the kernels need sm_80+ (cp.async, bf16 mma)
    ext_modules = [CUDAExtension(
        "statetree._C", ["statetree/csrc/kernels.cu", "statetree/csrc/bindings.cpp"],
        extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]})]
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="statetree",
    version="0.3.0",
    description="Lossless state-tree BF16 weights (12.1 bits/weight) with fused CUDA GEMV/GEMM kernels for PyTorch",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Ammar Ali",
    license="Apache-2.0",
    packages=find_packages(exclude=("tests", "examples")),
    package_data={"statetree": ["csrc/*.cu", "csrc/*.cpp"]},
    python_requires=">=3.9",
    install_requires=["torch>=2.1", "safetensors"],
    extras_require={"hf": ["transformers>=4.43"], "vllm": ["vllm>=0.10"]},
    entry_points={"vllm.general_plugins": ["statetree = statetree.vllm_plugin:register"]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
