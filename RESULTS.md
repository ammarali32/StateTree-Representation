# statetree: detailed results

All the measurements behind the summary in the [README](README.md): setups, tables and caveats.

- [A100 results](#a100-results) (Qwen3-8B on one A100-40GB, Qwen3.6-35B-A3B on two)
- [H100 results](#h100-results) (H100 SXM, Hopper kernels)
- [Tested environments](#tested-environments)
- [Exactness](#exactness)
- [Limitations](#limitations)

---

## A100 results

### At a glance

| Qwen3-8B · one A100-40GB · vLLM 0.32 | BF16 | **statetree** |
|---|---:|---:|
| Weights in GPU memory | 15.3 GiB | **12.8 GiB** (−24%) |
| KV cache per token | 144 KiB | **112.5 KiB** (−22%) |
| Tokens the KV cache holds (same 0.85 GPU budget) | 119K | **178K (1.49×)** |
| Decode attention vs FlashAttention-2 | 1.00× | **1.21–2.46×** |
| Generation speed, batch 1 | 75.6 tok/s | **97.2 tok/s (1.29×)** |
| Time to the complete answer | 1.00× | **1.10–1.28× faster** |
| Weights and cached values | original | **bit-identical** |

| Qwen3.6-35B-A3B (MoE, 256 experts) · 2 × A100-40GB, TP = 2 · vLLM 0.32 | BF16 | **statetree** |
|---|---:|---:|
| Weights per GPU | 32.35 GiB | **25.26 GiB** (−22%) |
| Tokens the KV cache holds (same 0.93 GPU budget) | 199.5K | **596.8K (3.0×)** |
| Generation speed, batch 1 / 8 / 32 | 166 / 675 / 1671 tok/s | **185 / 762 / 1828 tok/s (1.11× / 1.13× / 1.09×)** |
| Prefill (16 prompts of 3000 tokens) | 30.8K tok/s | **29.1K tok/s (0.95×)** |
| Weights | original | **bit-identical** |

<sub>The MoE model nearly fills two 40 GB GPUs in BF16, so the 7.2 GiB per GPU that the experts free go straight to the
KV cache. Both engines with a warm torch.compile cache, run back to back.
Greedy answers are not token-for-token identical: two BF16 runs of this model in vLLM already disagree on 2 of our 4
chat prompts (after 27 and 50 tokens), and statetree's answers diverge from BF16's at about the same rate.</sub>

**Setup.** Qwen3-8B, one A100-SXM4-40GB. vLLM 0.32 (torch 2.13, CUDA graphs and `torch.compile` on,
`gpu_memory_utilization` 0.85) and Transformers 5.0 (torch 2.7.1, static-cache decoder in a CUDA graph). Greedy
decoding. Scripts: `examples/vllm_response.py`, `examples/hf_response.py`, `examples/vllm_tps.py`. Exact versions:
[Tested environments](#tested-environments).

### Complete answers (vLLM)

Seconds until every request's 512-token answer is complete, prompts prefilled (prefix caching off), and speed-up vs
BF16 weights + BF16 KV cache:

| scenario | BF16 | weights | KV cache | weights + KV cache |
|---|---:|---:|---:|---:|
| chat, 1 request (256-token prompt) | 6.76 s | 1.25× | 1.02× | **5.28 s (1.28×)** |
| chat, 8 requests (512-token prompts) | 7.26 s | 1.19× | 1.02× | **5.93 s (1.23×)** |
| chat, 32 requests | 9.55 s | 1.03× | 1.07× | **8.70 s (1.10×)** |
| long context, 1 × 16K-token prompt | 8.97 s | 1.17× | 1.02× | **7.47 s (1.20×)** |
| long context, 8 × 8K-token prompts | 14.97 s | 1.07× | 1.04× | **13.29 s (1.13×)** |
| 96 requests × (1.5K + 512) tokens | 36.0 s | 1.02× | 1.05× | **28.2 s (1.28×)** |
| KV-cache tokens | 119K | 140K | 152K | **178K** |

The two halves help in different places, and together in all of them. State-tree **weights** speed up small batches
(the linear layers read 24% fewer bytes). The state-tree **KV cache** speeds up large batches and long contexts, and
above all helps when memory runs out: with 96 requests, BF16 can keep about 58 of them in its KV cache at once,
statetree about 87, so the whole batch finishes 1.28× sooner.

### Decode (vLLM)

Tokens/s after the prompt:

| batch × context | BF16 | KV cache | weights + KV cache |
|---|---:|---:|---:|
| 1 × 512 | 75.6 | 77.0 (1.02×) | **97.2 (1.29×)** |
| 1 × 16K | 67.5 | 68.8 (1.02×) | **84.4 (1.25×)** |
| 8 × 8K | 402 | 423 (1.05×) | **487 (1.21×)** |
| 16 × 4K | 797 | 838 (1.05×) | **914 (1.15×)** |
| 32 × 2K | 1433 | 1636 (1.14×) | **1706 (1.19×)** |
| 64 × 1K | 2989 | 3207 (1.07×) | **3257 (1.09×)** |

Weights only (`examples/vllm_tps.py`, 32-token prompts, 128 new tokens), tokens/s:

| batch | BF16 | statetree weights |
|---:|---:|---:|
| 1 | 76.2 | **95.8 (1.26×)** |
| 8 | 599.8 | **727.2 (1.21×)** |
| 32 | 2213 | 2239 (1.01×) |
| 64 | 4114 | 4107 (1.00×) |

### Decode attention

One query token per sequence, Qwen3-8B heads (32 query / 8 KV, head size 128), µs per call, L2 flushed before every
call, vs vLLM 0.32's paged FlashAttention-2 (block 16) on the same values:

| batch | context 1K | 4K | 16K |
|---:|---|---|---|
| 1 | 53.8 → **21.9 (2.46×)** | 57.0 → **34.1 (1.67×)** | 95.2 → **78.0 (1.22×)** |
| 8 | 72.4 → **47.0 (1.54×)** | 150.3 → **118.0 (1.27×)** | 440.7 → **363.7 (1.21×)** |
| 32 | 171.5 → **116.5 (1.47×)** | 538.1 → **361.8 (1.49×)** | 2001.7 → **1345.8 (1.49×)** |
| 64 | 263.6 → **207.5 (1.27×)** | 914.4 → **740.6 (1.23×)** | 3276.3 → **2610.8 (1.25×)** |

The kernel reads the state-tree KV pages directly: decoding a key record yields exactly the tensor-core fragments of
Kᵀ, values are transposed in registers, and an online softmax with split-K covers long contexts. It reads 22% fewer
bytes than FlashAttention-2, with the split count vLLM's CUDA graphs use. Outputs agree with FlashAttention-2 to within
1e-3 (the softmax weights are rounded to BF16 in both kernels). In a full decode step attention is only part of the
time, hence the 1.02–1.14× from the KV cache alone in the decode table.

### Memory

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/capacity_dark.png">
  <img alt="Tokens the vLLM KV cache holds in the same 0.85 GPU budget: BF16 119K, statetree KV cache 152K, statetree weights 140K, both 178K" src="assets/capacity_light.png" width="100%">
</picture>

Weights: 15.27 → 12.80 GiB in vLLM (including kernel workspaces and the 128 MB escape pool), 15.26 → 11.71 GiB in
Transformers. KV cache: 144 → 112.5 KiB per token.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/hf_memory_dark.png">
  <img alt="Peak GPU memory in Transformers, BF16 with StaticCache vs statetree, five scenarios: 15 to 20 percent less" src="assets/hf_memory_light.png" width="100%">
</picture>

Peak GPU memory in Transformers (`torch.cuda.max_memory_allocated` over prefill, CUDA-graph capture and 512 decode
steps):

| requests × (prompt + new) | BF16 + StaticCache | statetree weights + KV cache |
|---|---:|---:|
| 1 × (256 + 512) | 15.44 GiB | **12.37 GiB (0.80×)** |
| 8 × (512 + 512) | 16.83 GiB | **13.64 GiB (0.81×)** |
| 32 × (512 + 512) | 21.46 GiB | **17.84 GiB (0.83×)** |
| 1 × (16K + 512) | 19.42 GiB | **16.00 GiB (0.82×)** |
| 4 × (8K + 512) | 23.24 GiB | **19.86 GiB (0.85×)** |

### Transformers speed

With the weights as state trees, decode is 1.27× faster at batch 1, 1.15× at 8, 1.07× at 32 and 1.04× at 64. With
the state-tree KV cache as well, complete answers in the scenarios above are 1.46× faster at batch 1 and 2.1–3.9×
faster at batch 8–32 and 8K–16K contexts than transformers' StaticCache + SDPA. Most of that gap is transformers'
static-cache path itself (with a mask it copies every KV head per query head and attends over the whole static length
at every step), so read it as "statetree's cache inside transformers", not as the format alone; the vLLM comparison
against FlashAttention-2 is the like-for-like one.

### Prefill

Prefill is compute-bound, so fewer bytes don't help there. A fused tensor-core GEMM decodes the trees on the fly for
65–1024 tokens per step (decode + cuBLAS above). Prefill latency vs BF16, batch 1:

| prompt tokens | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|---:|
| vLLM | 1.03× | 0.87× | 0.87× | 0.87× | 0.92× | 0.95× |
| Transformers | 1.00× | 0.99× | 0.90× | 0.91× | 0.94× | 0.96× |

Time to first token is therefore 3–15% longer with state-tree weights (the KV cache adds < 1%).

---

## H100 results

Measured on one H100 80GB HBM3 (SXM, 700 W) with the Hopper kernels, vLLM 0.32, CUDA graphs, a fresh tuning cache,
greedy decoding, weights only (`examples/vllm_tps.py`: 32-token prompts, 128 new tokens). H200 uses the same Hopper
(sm_90) kernels and was not measured separately.

### Qwen3-8B

Weights 16.4 GB → 12.4 GB; teacher-forced top-1 agreement with BF16 98.4%.

| decode, tokens/s | BF16 | statetree | speedup |
|---|---:|---:|---:|
| batch 1 | 152.1 | 187.2 | **1.23×** |
| batch 8 | 1166 | 1410 | **1.21×** |
| batch 32 | 4388 | 4653 | 1.06× |
| batch 64 | 7975 | 7944 | 1.00× |

| prefill, batch 1 | BF16 | statetree | speedup |
|---|---:|---:|---:|
| 512 tokens | 20.2 ms | 23.9 ms | 0.84× |
| 2048 tokens | 57.3 ms | 66.5 ms | 0.86× |

Per layer kind (all 36 layers of a kind back to back in a CUDA graph, batch 1, 3 s sustained):

| layer | cuBLAS BF16 | statetree | speedup |
|---|---:|---:|---:|
| gate_up | 2.453 ms | 1.960 ms | 1.25× |
| down | 1.358 ms | 1.079 ms | 1.26× |
| qkv | 0.723 ms | 0.581 ms | 1.24× |
| o | 0.574 ms | 0.446 ms | 1.29× |

### Qwen3.6-35B-A3B (MoE)

TP = 1, `--gpu-mem 0.95` (the BF16 model needs nearly all of the 80 GB); weights 70.2 GB → 54.2 GB; teacher-forced
top-1 agreement 95.3%. The expert kernels are the A100 ones on this card.

| decode | batch 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| BF16 tok/s | 215.7 | 372.1 | 608.6 | 913.2 | 1482 | 2306 |
| statetree tok/s | 238.4 | 393.7 | 624.0 | 1029 | 1517 | 2328 |
| speedup | **1.11×** | 1.06× | 1.03× | **1.13×** | 1.02× | 1.01× |

Prefill, batch 1: 0.90× at 512 tokens, 0.89× at 2048.

### Power-limited cards

Decode at small batch is a sustained streaming load, so the card's power limit matters. On an H100 NVL capped at
310 W the SM clock fell to 345–750 MHz and Qwen3-8B gained only 1.06× at batch 1. On the 700 W SXM card the clock stays
at 1980 MHz. Check `nvidia-smi --query-gpu=power.limit,power.max_limit --format=csv` before benchmarking.

### KV cache on H100

The state-tree KV cache runs on Hopper with the A100 decode-attention kernel (its tests pass); its speed on H100 has
not been measured yet.

---

## Tested environments

**A100** (every A100 number above; the MoE results on two of these GPUs):

| | Transformers runs | vLLM runs |
|---|---|---|
| GPU | 1 × NVIDIA A100-SXM4-40GB (Ampere, sm_80) | same |
| NVIDIA driver | 535.129.03 | same |
| CUDA toolkit (`nvcc`, builds statetree's kernels) | 12.6 (V12.6.85) | same |
| PyTorch | 2.7.1+cu126 | 2.13.0+cu129 (the build vLLM 0.32 requires) |
| 🤗 Transformers | 5.0.0 | 5.19.0 |
| vLLM | – | 0.32.0 |
| Triton | 3.3.1 | 3.7.1 |
| Python | 3.11.13 | 3.11.13 |
| OS · host compiler | Ubuntu 22.04 · GCC 11.4.0 | same |
| Model | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), BF16 | same |

Both environments compile statetree's kernels with the same CUDA 12.6 `nvcc`. In the vLLM environment, PyTorch
brings its own CUDA 12.9 runtime, and CUDA's minor-version compatibility lets the two work together.

**H100** (every H100 number above):

| | vLLM runs |
|---|---|
| GPU | 1 × NVIDIA H100 80GB HBM3 (SXM, Hopper, sm_90), power limit 700 W |
| NVIDIA driver | 595.91.07 |
| CUDA toolkit (`nvcc`) | 13.0 (V13.0.88) |
| PyTorch | 2.13.0 (CUDA 13.0 build) |
| vLLM · flashinfer · Triton | 0.32.0 · 0.7.0.post1 · 3.7.1 |
| 🤗 Transformers | 5.21.0 |
| Python | 3.12.3 |
| Models | Qwen/Qwen3-8B, Qwen/Qwen3.6-35B-A3B, BF16 |

**Transformers versions.** statetree follows the current Transformers interfaces (the cache layers of 5.20+, the 3-D
experts modules) and needs Transformers 5.20 or newer; the test suite passes with 5.21.0 and 5.20.2. vLLM 0.32 pins
`transformers>=5.16.1,<5.20` for its own use.

---

## Exactness

The **weights** and every **cached key / value** are bit-identical to BF16: checked at conversion, after save → load,
and in the KV-cache tests (pages written and read back over recycled blocks). Every kernel is tested bit-exact with
one-hot inputs, escapes included, on the A100 and on Hopper. The **outputs** match BF16 up to the order of
floating-point additions, the same kind of difference you get between two cuBLAS or attention kernels. In our vLLM
tests, greedy generations, prefix-cache hits and chunk-prefilled long prompts produced the same tokens as vLLM's BF16
cache; teacher-forced top-1 tokens agree with BF16 at 95–99.8% of positions, and the differences fall on near-ties.
Outputs are bit-identical run to run.

---

## Limitations

* Inference only; there is no backward pass yet.
* Requires sm_80 or newer. Tuned and measured on the A100 and the H100 SXM.
* Prefill is 0.84–1.03× of BF16 in vLLM (A100 0.87–1.03×, H100 0.84–0.86×).
* KV cache: BF16 and head size 128 only; no speculative decoding, sliding windows or attention sinks yet. With prefix
  caching or chunked prefill, the cached part of a prompt is decoded into a temporary BF16 buffer for prefill
  attention, outside vLLM's memory budget. Not yet measured on H100.
* Small matrices at batch 16–64, and tensor parallelism on small models, gain little from the weights alone.
* MoE: decode is 1.01–1.13× of BF16 on Qwen3.6-35B-A3B in vLLM (batch 1–32), and the KV cache holds 3× the tokens.
  Prefill is 0.89–0.95× of BF16 end to end: the expert layers are faster than vLLM's BF16 Triton kernel up to 2048
  tokens per step and slower beyond (one full layer: 0.86× / 0.79× at 4096 / 8192 tokens). Which kernel runs depends
  on the size of the step, so a token's output can differ in the last bits with the batch it was scheduled in (as with
  most fused kernels). Routers stay BF16 in Transformers. Experts modules without the Transformers 5 layout flags are
  left alone. In vLLM, expert parallelism, biased or non-SiLU experts use the decode-to-BF16 path at every batch size,
  as do experts whose matrices are not whole 128-row blocks above 64 pairs per expert.
