# statetree 🌳

**Do you want to serve your LLM with 24% less GPU memory, 24% smaller checkpoints on disk and up to
1.2× faster generation, without changing a single weight?**

If yes, you're in the right place.

statetree stores BF16 weights as **state trees**: a lossless format in which weights that belong
together share their exponent. Every weight decodes back to its **exact BF16 bits**, so there is no
quantization, no calibration, no accuracy loss and no retraining. Your model just reads about
**12.1 bits per weight instead of 16** from memory, and since token generation is memory-bound,
fewer bytes means faster tokens.

| | BF16 | statetree |
|---|---|---|
| Weights in GPU memory | 100% | **76%** |
| Checkpoint on disk | 100% | **76%** |
| Generation speed (vLLM, batch 1) | 1.00× | **1.23×** |
| Weights | original | **bit-identical** |

*Llama-3.2-1B and Qwen3.5-9B on an A100-40GB, vLLM 0.32. Details below.*

---

## Quick start

### With vLLM: one flag

```bash
pip install statetree
vllm serve Qwen/Qwen3.5-9B --dtype bfloat16 --representation statetree
```

That's it. The weights are converted on load, checked bit-exact, and the freed memory goes to the KV
cache.

### With 🤗 Transformers: one line

```python
import torch, statetree
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B", dtype=torch.bfloat16).cuda()
statetree.convert(model)        # every weight -> state tree, checked bit-exact
model.generate(...)             # works exactly as before

statetree.save_pretrained(model, "llama32_1b_statetree", tokenizer)    # 24% smaller on disk
model = statetree.load_pretrained("llama32_1b_statetree")               # loads directly, no BF16 copy
```

---

## How it works, in 30 seconds

A BF16 weight is a sign, an 8-bit exponent and a 7-bit mantissa. Neighbouring weights in a trained
layer live within a few powers of two of each other, yet BF16 stores the full exponent for every one
of them. statetree stores it **once per group** of 64 weights:

```
            group node: largest exponent E          (8 bits, once per 64 weights)
           /        |         |          \
     state D₁   state D₂   state D₃ ...   D₆₄       (4 bits each: how far below E)
         |          |          |           |
     sign+mant  sign+mant  sign+mant ... sign+mant  (8 bits each, unchanged)
```

**8 + 4 + 8/64 = 12.125 bits per weight**, and every weight is reproduced exactly:
`bf16 = (sign << 15) | ((E − D) << 7) | mantissa`.

The rare weights that don't fit, such as exact zeros (about 0.01% in LLMs), are kept in a small side
list. Every field sits at a fixed offset, laid out in tensor-core register order, so a GPU decodes
the weights **straight into the tensor cores** with two integer operations per weight. There are no
pointers, no lookup tables and no BF16 copy in memory.

This is the simplest case of a more general idea, writing every weight as a path of shared
decisions. See the paper below.

---

## Results

### Where statetree shines: generation (decode)

vLLM 0.32, A100-40GB, CUDA graphs and `torch.compile` on, 32-token prompt, 128 new tokens:

| | Llama-3.2-1B | Qwen3.5-9B |
|---|---:|---:|
| Weights in GPU memory | 2.49 → **1.89 GB** (−24%) | 18.85 → **14.37 GB** (−24%) |
| Decode, batch 1 | **1.23×** | **1.23×** |
| Decode, batch 8 | **1.21×** | **1.17×** |
| Decode, batch 32 | 1.00× | 1.04× |

On Qwen3.5-9B, the linear layers alone run **1.29× faster** than fused BF16 at batch 1, close to the
16 / 12.13 = 1.32× limit that fewer bytes allow.

### Where it doesn't (yet): prefill

| | Llama-3.2-1B | Qwen3.5-9B |
|---|---:|---:|
| Prefill, 512 tokens | 0.80× | 0.71× |
| Prefill, 2048 tokens | 0.91× | 0.89× |

Prefill is compute-bound. Today we decode each layer to BF16 and then call cuBLAS, so prefill is
**slower than BF16**. In principle the format is tensor-core friendly and should match cuBLAS for large
batches while keeping its advantage in generation. Closing this gap is our top open problem, and
**contributions are very welcome** (see the roadmap).

### About exactness

The **weights** are bit-identical to BF16: this is checked at conversion and after save → load. The
**outputs** match BF16 up to the order of floating-point additions, the same kind of difference you
get between two cuBLAS algorithms. Teacher-forced top-1 tokens agree with BF16 at 98–99% of positions,
and the differences fall on near-ties.

---

## A note on the kernels

The CUDA kernels were written with substantial help from AI coding assistants, because our team's
background is in model compression rather than GPU kernel engineering. They are correct (bit-exact
and tested), but they are **not optimal**. Faster kernels are clearly possible, especially for larger
batch sizes and prefill. If you know CUDA and tensor cores, this is the best place to help.

---

## Roadmap

- [ ] **Faster kernels for large batches and prefill.** State trees are tensor-core friendly, so
  in theory they should match cuBLAS at large batch sizes while staying faster in generation. That
  would carry the savings beyond LLM decoding to any model.
- [ ] **LoRA fine-tuning on state-tree models.** Train adapters on top of the frozen, compressed base
  to reduce training memory.
- [ ] **State trees for whole models, including activations and the KV cache**, not just weights.
- [ ] **State trees for convolution layers.**
- [ ] Better speed-ups under tensor parallelism for small models.
- [ ] Wider tuning beyond the A100 (H100, L4, consumer GPUs).

Pick one, open an issue, and let's talk. 🙌

---

## Install

Requirements: an NVIDIA GPU with compute capability 8.0 or newer (A100, RTX 30xx/40xx, L4, H100),
PyTorch ≥ 2.1 with CUDA, `nvcc` matching torch's CUDA, and `safetensors`. `transformers` is needed
for `load_pretrained` and the examples.

```bash
pip install wheel ninja
pip install --no-build-isolation .
# faster build for one GPU type:  TORCH_CUDA_ARCH_LIST=8.0 pip install --no-build-isolation .
```

Tested with torch 2.7.1+cu126 (Transformers) and torch 2.13.0+cu129 with vLLM 0.32.0.

<details>
<summary><b>vLLM installation details</b></summary>

```bash
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu129
pip install https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl \
    --extra-index-url https://download.pytorch.org/whl/cu129
pip install --no-build-isolation .
python tests/test_vllm.py
```

Other ways to enable it: `--additional-config '{"representation": "statetree"}'`, or
`VLLM_REPRESENTATION=statetree`. From Python:

```python
import statetree.vllm_plugin
from vllm import LLM
llm = LLM(model="unsloth/Llama-3.2-1B", dtype="bfloat16", representation="statetree")
```

statetree composes with quantization: layers claimed by a `--quantization` method stay quantized,
and the remaining BF16 layers become state trees. Tensor and pipeline parallelism, chunked prefill,
prefix caching and CUDA graphs all work.

If vLLM 0.32 fails to load `libcudart.so.13` with a driver older than 580 (through torchcodec), run:
```bash
pip install --target cu13libs "nvidia-cuda-runtime==13.*" "nvidia-cuda-nvrtc==13.*"
export LD_LIBRARY_PATH=$PWD/cu13libs/nvidia/cu13/lib
```
</details>

<details>
<summary><b>Multiple GPUs and CPU offload (Transformers)</b></summary>

```python
model = statetree.load_pretrained(path)                          # device_map="auto"
model = statetree.load_pretrained(path, device_map="balanced")
model = statetree.load_pretrained(path, max_memory={0: "10GiB", 1: "10GiB", "cpu": "64GiB"})
```

Offloaded layers keep their trees in CPU memory and cross PCIe at 12 bits per weight instead of 16.
</details>

<details>
<summary><b>API</b></summary>

* `statetree.convert(model, include=None, exclude=(), fuse=DEFAULT_FUSE, embeddings=True, verify=True)`
* `statetree.prepare(model, batch_sizes)`: tunes kernels ahead of time (run before CUDA-graph capture)
* `statetree.save_pretrained(model, path, tokenizer=None)` / `statetree.load_pretrained(path, ...)`
* `statetree.size_report(model)`, `TreeLinear.dequantize()` (exact BF16 weight)
* `statetree.encode` / `statetree.decode_reference` (pure PyTorch reference)
</details>

---

## Limitations

* Inference only; there is no backward pass yet (see LoRA on the roadmap).
* Requires sm_80 or newer; tuned on A100.
* Prefill is 0.7–0.93× of BF16 (see above).
* Small matrices at batch 16–64, and tensor parallelism on small models, give little or no gain.
* Only weights are compressed; the KV cache is not.

---

## Citation

statetree-repo implements one application of our proposed weight representation paper (soon):

```bibtex
@article{ali2026twig,
  title   = {Every Weight Is a Path of Shared Discrete Decisions},
  author  = {Ali, Ammar and Lefkimmiatis, Stamatis},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

Apache-2.0. See `LICENSE` and `NOTICE`. The fused GEMM follows the structure of
[Marlin](https://github.com/IST-DASLab/marlin) (Apache-2.0).