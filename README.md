<div align="center">

# statetree 🌳

**Lossless compression of LLM weights *and* KV cache, with fused CUDA kernels for vLLM and 🤗 Transformers.**

Every weight and every cached key / value decodes back to its exact BF16 bits.
No quantization, no calibration, no accuracy trade-off. Just fewer bytes, and faster tokens.
Dense and Mixture-of-Experts models, on one GPU or several.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-0.32-30A2FF)](#with-vllm-two-flags)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-5.21-FFD21E)](#with--transformers)
[![GPU](https://img.shields.io/badge/NVIDIA-A100%20%C2%B7%20H100%20%C2%B7%20H200-76B900)](#installation)
[![CUDA](https://img.shields.io/badge/CUDA-12.6%20%7C%2013.0-76B900)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%20%7C%202.13-EE4C2C)](#installation)
[![Lossless](https://img.shields.io/badge/weights%20%26%20KV%20cache-bit--exact-2EA44F)](RESULTS.md#exactness)

<img src="assets/demo.gif" alt="BF16 and statetree answering the same question side by side in vLLM; statetree finishes first with the identical answer" width="100%">

<sub>Qwen3-8B in vLLM 0.32 on one A100-40GB, the same question at batch 1, greedy. Both engines' real answers,
token-for-token identical (256 tokens), replayed at their measured speeds: 3.35 s with BF16, 2.63 s with statetree (1.27× sooner).</sub>

</div>

---

LLM generation is memory-bound: every new token re-reads all the weights and the whole KV cache. statetree stores
both at about **12 bits per value instead of 16**, losslessly, and its kernels read that format directly, decoding
straight into the tensor cores. Fewer bytes per token means faster tokens, and the memory it frees holds more
conversations at once.

**New: NVIDIA Hopper (H100 / H200 family).** statetree now supports Hopper GPUs with kernels designed for them:
small-batch decode kernels that stream the compressed weights in memory order (with TMA bulk copies and programmatic
dependent launch), a fused tree-decode + `wgmma` GEMM for larger batches, and a tuner that ranks the kernels inside CUDA
graphs, the way vLLM runs them. The A100 kernels are unchanged.

## Performance

Decode speed of statetree weights vs BF16 in vLLM 0.32 (CUDA graphs, greedy):

| | batch 1 | batch 8 | batch 32 | batch 64 |
|---|---:|---:|---:|---:|
| Qwen3-8B, A100-40GB | **1.26×** | **1.21×** | 1.01× | 1.00× |
| Qwen3-8B, H100 SXM | **1.23×** | **1.21×** | 1.06× | 1.00× |
| Qwen3.6-35B-A3B (MoE), 2 × A100-40GB | **1.11×** | **1.13×** | 1.09× | |
| Qwen3.6-35B-A3B (MoE), H100 SXM | **1.11×** | **1.13×** | 1.01× | |

With the state-tree KV cache as well (Qwen3-8B, A100): complete answers **1.10–1.28× sooner**, decode attention
**1.21–2.46× faster** than FlashAttention-2, and **1.49× more tokens** in the same KV-cache budget. Weights take 24%
less memory. All tables, setups and caveats: **[RESULTS.md](RESULTS.md)**.

### Complete answers, up to 1.28× sooner

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/response_dark.png">
  <img alt="Seconds until every 512-token answer is complete, BF16 vs statetree, six scenarios: chat with 1, 8 and 32 requests, long context 1 x 16K and 8 x 8K, and 96 requests beyond BF16 KV capacity; statetree is 1.10 to 1.28 times faster" src="assets/response_light.png" width="100%">
</picture>

### Decode attention faster than FlashAttention-2

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/attention_dark.png">
  <img alt="Decode attention speed-up over FlashAttention-2 for batch 1, 8, 32 and 64 at 1K, 4K and 16K context: 1.21x to 2.46x" src="assets/attention_light.png" width="100%">
</picture>

### Faster decoding at every batch size and context length

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/decode_dark.png">
  <img alt="Decode throughput gain over BF16 for six batch and context combinations: KV cache only 2 to 14 percent, weights plus KV cache 9 to 29 percent" src="assets/decode_light.png" width="100%">
</picture>

<sub>Figures: Qwen3-8B on one A100-40GB, vLLM 0.32.</sub>

---

## Installation

Requirements: an NVIDIA GPU with compute capability 8.0 or newer (A100, RTX 30xx/40xx, L4, H100, H200), PyTorch ≥ 2.1
with CUDA, a CUDA toolkit (`nvcc`) with the same major version as torch's CUDA, and `safetensors`. `transformers` 5.20
or newer (tested with 5.21.0, `pip install "statetree[hf]"`) is needed for `load_pretrained`, the Transformers KV
cache, MoE models and the examples.

```bash
pip install wheel ninja
pip install --no-build-isolation .
# one GPU type only (faster build):
#   A100:         TORCH_CUDA_ARCH_LIST=8.0  pip install --no-build-isolation .
#   H100 / H200:  TORCH_CUDA_ARCH_LIST=9.0a pip install --no-build-isolation .   (the "a" is required)
```

This builds two extensions, `statetree._C` (weight kernels) and `statetree._kv` (KV-cache pages and decode
attention). Without `TORCH_CUDA_ARCH_LIST`, the build targets the GPUs it finds (Hopper as 9.0a). Without a build, both
are compiled on first use.

<details>
<summary><b>vLLM environment</b></summary>

```bash
pip install vllm==0.32.0              # brings torch 2.13 (CUDA 13.0 build; CUDA 12.9 wheels also exist)
pip install --no-build-isolation .
python -m pytest tests/test_vllm.py tests/test_vllm_kv.py -q
```

The `nvcc` that builds statetree must have torch's CUDA major version (13 for the default torch 2.13 wheels). If you
take `nvcc` from NVIDIA's pip wheels, pin `nvidia-nvvm` and `nvidia-cuda-crt` to the same version as
`nvidia-cuda-nvcc`; mixed versions make ptxas reject the generated PTX.

If vLLM 0.32 fails to load `libcudart.so.13` with a driver older than 580 (through torchcodec), run:
```bash
pip install --target cu13libs "nvidia-cuda-runtime==13.*" "nvidia-cuda-nvrtc==13.*"
export LD_LIBRARY_PATH=$PWD/cu13libs/nvidia/cu13/lib
```
</details>

---

## Usage

### With vLLM: two flags

```bash
vllm serve Qwen/Qwen3-8B --dtype bfloat16 --representation statetree --kv-representation statetree
```

`--representation statetree` converts the weights on load (checked bit-exact) and gives the freed memory to the KV
cache. `--kv-representation statetree` stores the paged KV cache itself as state trees, with its own
decode-attention kernel. Each flag works on its own; prefix caching, chunked prefill and CUDA graphs keep working.

```python
import statetree.vllm_plugin
from vllm import LLM
llm = LLM(model="Qwen/Qwen3-8B", dtype="bfloat16", representation="statetree", kv_representation="statetree")
```

<details>
<summary><b>More vLLM options</b></summary>

Other ways to enable the weights: `--additional-config '{"representation": "statetree"}'` or
`VLLM_REPRESENTATION=statetree`. statetree composes with quantization: layers claimed by a `--quantization` method
stay quantized, and the remaining BF16 layers become state trees. Tensor and pipeline parallelism, chunked prefill,
prefix caching and CUDA graphs all work with state-tree weights.

The KV cache is vLLM's CUSTOM attention backend, registered by the plugin; `--kv-representation statetree` selects it
and block size 16. It needs vLLM 0.32, BF16 and head size 128 (Qwen3, Llama-3-8B and larger, ...). Escapes beyond
the per-page slots use a shared pool (`STATETREE_KV_POOL_MB`, default 128); an exhausted pool raises instead of
losing values.

Kernel tuning results are cached in `~/.cache/statetree/tune.json` (`STATETREE_CACHE` to move it,
`STATETREE_AUTOTUNE=0` for heuristics only).
</details>

### With 🤗 Transformers

```python
import torch, statetree
from transformers import AutoModelForCausalLM
from statetree.kvcache import StateTreeKVCache, enable

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16).cuda()
statetree.convert(model)        # every weight -> state tree, checked bit-exact
model.generate(...)             # works exactly as before

enable(model)                   # optional: the KV cache as state trees too
cache = StateTreeKVCache(model.config, batch_size=1, max_cache_len=4096)
model.generate(**inputs, past_key_values=cache, max_new_tokens=512)

statetree.save_pretrained(model, "qwen3_8b_statetree", tokenizer)       # 24% smaller on disk
model = statetree.load_pretrained("qwen3_8b_statetree")                  # loads directly, no BF16 copy
```

<details>
<summary><b>Multiple GPUs and CPU offload</b></summary>

```python
model = statetree.load_pretrained(path)                          # device_map="auto"
model = statetree.load_pretrained(path, device_map="balanced")
model = statetree.load_pretrained(path, max_memory={0: "10GiB", 1: "10GiB", "cpu": "64GiB"})
```

Offloaded layers keep their trees in CPU memory and cross PCIe at 12 bits per weight instead of 16.
</details>

<details>
<summary><b>API</b></summary>

* `statetree.convert(model, include=None, exclude=(), fuse=DEFAULT_FUSE, embeddings=True, experts=True, verify=True,
  max_escape_fraction=None)`: `experts` also converts MoE experts held as 3-D tensors (`statetree.TreeExperts`)
* `statetree.prepare(model, batch_sizes)`: tunes kernels ahead of time (run before CUDA-graph capture)
* `statetree.save_pretrained(model, path, tokenizer=None)` / `statetree.load_pretrained(path, ...)`
* `statetree.size_report(model)`, `TreeLinear.dequantize()` (exact BF16 weight)
* `statetree.kvcache.enable(model)`, `statetree.kvcache.StateTreeKVCache(config, batch_size, max_cache_len)`: the
  Transformers KV cache (`generate()`, left padding, CUDA-graph decode)
* `statetree.encode` / `statetree.decode_reference` (pure PyTorch reference)
</details>

<details>
<summary><b>Benchmarks and tests</b></summary>

* `examples/vllm_response.py`, `examples/hf_response.py`: time to the complete answer and memory, BF16 / state-tree
  weights × BF16 / state-tree KV cache
* `examples/vllm_tps.py`, `examples/hf_tps.py`: decode throughput, prefill latency and exactness
* `python -m pytest tests -q`: kernels, KV cache, devices, MoE, vLLM (`tests/test_vllm*.py`)
</details>

How the format works, its exactness, limitations and the roadmap: [RESULTS.md](RESULTS.md) and
[docs/DESIGN.md](docs/DESIGN.md).

---

## Paper

statetree implements one application of our weight-representation <a href="https://huggingface.co/papers/2610.31931">paper</a>:

```bibtex
@article{ali2026every,
  title   = {Every Weight Is a Path of Shared Discrete Decisions},
  author  = {Ammar Ali and Stamatis Lefkimmiatis},
  year    = {2026},
  journal = {arXiv preprint arXiv: 2610.31931}
}
```

## License

Apache-2.0. See `LICENSE` and `NOTICE`. The fused GEMM follows the structure of
[Marlin](https://github.com/IST-DASLab/marlin) (Apache-2.0).
