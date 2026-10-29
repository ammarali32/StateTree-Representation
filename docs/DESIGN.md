# statetree: design notes

## How it works

A BF16 number is a sign, an 8-bit exponent and a 7-bit mantissa. Neighbouring weights in a trained layer, and
neighbouring channels of one token's key or value, live within a few powers of two of each other, yet BF16 stores the
full exponent for every one of them. statetree stores it **once per group** of 64 values:

```
            group node: largest exponent E          (8 bits, once per 64 values)
           /        |         |          \
     state D₁   state D₂   state D₃ ...   D₆₄       (4 bits each: how far below E)
         |          |          |           |
     sign+mant  sign+mant  sign+mant ... sign+mant  (8 bits each, unchanged)
```

**8 + 4 + 8/64 = 12.125 bits per value**, and every value is reproduced exactly:
`bf16 = (sign << 15) | ((E − D) << 7) | mantissa`. The rare values that don't fit, such as exact zeros (about 0.01%
of LLM weights, 0.01–0.07% of KV values), are kept in a small side list.

Every field sits at a fixed offset, laid out in tensor-core register order, so a GPU decodes the values **straight
into the tensor cores** with two integer operations per value: no pointers, no lookup tables, no BF16 copy in memory.

- **Weights.** Fused GEMV / GEMM kernels for every batch size: decode, mid-size batches, and a one-token-tile
  tensor-core GEMM for prefill. On Hopper (H100 / H200), dedicated kernels stream the records in memory order (one
  of them through TMA bulk copies into a shared-memory ring), start loading weights while the previous kernel drains
  (programmatic dependent launch), and a fused tree-decode + `wgmma` GEMM serves larger batches. A tuner times the
  candidate kernels per shape and batch size, inside CUDA graphs on Hopper, and caches the choice.
- **KV cache.** One vLLM block (16 tokens) per KV head holds its keys and values as four 16 × 64 records with 11
  escape slots each: **6400 bytes instead of 8192**. New tokens are appended in place; the decode-attention kernel
  reads the pages directly. Escapes beyond the slots go to a small shared pool, recycled whenever vLLM reuses a block.

This is the simplest case of a more general idea, writing every weight as a path of shared decisions. See the
[paper](https://huggingface.co/papers/2610.31931).

## Roadmap

- [ ] **Prefill at parity with cuBLAS.** The fused GEMM closed most of the gap (0.84–1.03× in vLLM); the format is
  tensor-core friendly, so parity should be reachable.
- [x] **State trees for the KV cache** (vLLM paged cache and Transformers, lossless).
- [x] **Mixture-of-Experts models** (Transformers and vLLM), with a grouped tree GEMM for MoE prefill.
- [x] **Hopper kernels** (H100 / H200).
- [ ] **MoE on Hopper:** tune the expert kernels for H100 (batch 16–32 gains less than on the A100).
- [ ] **KV cache on Hopper:** measure and tune the decode-attention kernel.
- [ ] **MoE prefill at parity** at 4096+ tokens per step.
- [ ] **State trees for activations.**
- [ ] **LoRA fine-tuning on state-tree models.** Train adapters on top of the frozen, compressed base.
- [ ] **State trees for convolution layers.**
- [ ] KV cache: speculative decoding, sliding windows, attention sinks, head sizes other than 128.
- [ ] Tuning for more GPUs (L4, consumer cards). Tell us which hardware you need.

Pick one, open an issue, and let's talk.

## A note on the kernels

The CUDA kernels were written with substantial help from AI coding assistants, because our team's background is in
model compression rather than GPU kernel engineering. They are correct (bit-exact and tested), but not optimal: faster
kernels are clearly possible, especially for prefill. If you know CUDA and tensor cores, this is the best place to
help.

**Need statetree on specific hardware?** Whether it is a particular GPU or custom hardware, open an issue or contact
us with the device and your workload.
