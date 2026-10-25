// kvcache.cu: the KV cache stored losslessly in the state-tree format (statetree.kvcache, statetree.vllm_kv).
//
// Page = 16 tokens. Per (page, KV head, K/V, dim half) one record of 16 tokens x 64 dims in the weight format v2
// (token = row: the node is the token's largest exponent over 64 dims; 4-bit state D = node - exponent, leaf =
// mantissa << 1 | sign of the pair partner), followed by its escape area (D >= 15: the exact bf16 bits):
//   [0, 1552)     record (leaves / states / nodes in mma fragment order, see statetree/format.py)
//   [1552, 1556)  header: bits 0-3 slots used, bit 4 has extension, bits 8-31 first extension block
//   [1556, 1600)  11 slots: bf16 bits << 16 | row << 6 | col
// Extension blocks (pool, 16 u32 each, rarely used): word 0 = slots used | (next block + 1) << 8, words 1-15 slots.
// Cache tensor: uint8 [num_blocks, H, 2, 2, 1600] = 6400 B per page per KV head (BF16: 8192 B) -> 0.781 x.
// Slot mapping as in vLLM: slot = block * 16 + offset; -1 = padding token (skipped).
// vLLM: append(reuse=True) recycles a page's extension chain when the page is rewritten from row 0 (vLLM reuses
// freed blocks; its cache starts zeroed), so the pool never leaks; dequant_blocks gathers listed pages (prefill reads);
// attn(minpps) keeps >= minpps pages per split, so one fixed split count (CUDA graphs) suits every context length.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {
constexpr int PAGE = 16;
constexpr int HD = 128;
constexpr int REC = 1552;
constexpr int RKV = 1600;
constexpr int NSLOT = 11;
constexpr int XSLOT = 15;
constexpr unsigned FULL = 0xFFFFFFFFu;

// byte offsets of the two leaves / two states of cols 2l, 2l+1 (a bf16 pair) of row r in a record
__device__ __forceinline__ int leaf_off(int r, int l) {
  const int s = l >> 3, t = l & 3, hi = (l >> 2) & 1;
  return (s >> 1) * 512 + ((r & 7) * 4 + t) * 16 + ((s & 1) * 2 + (r >> 3)) * 4 + 2 * hi;
}
__device__ __forceinline__ int state_off(int r, int l) {
  const int s = l >> 3, t = l & 3, hi = (l >> 2) & 1;
  return 1024 + ((r & 7) * 4 + t) * 16 + s * 4 + 2 * hi;
}
__device__ __forceinline__ void dec4r(uint32_t lf, uint32_t e4, uint32_t& p01, uint32_t& p23) {
  const uint32_t a = __byte_perm(lf, e4, 0x5140), b = __byte_perm(lf, e4, 0x7362);
  p01 = __funnelshift_r(a, a, 1);
  p23 = __funnelshift_r(b, b, 1);
}
__device__ __forceinline__ uint8_t* record(uint8_t* kv, long long blk, int H, int h, int kvi, int half) {
  return kv + ((((size_t)blk * H + h) * 2 + kvi) * 2 + half) * RKV;
}

// ------------------------------------------------------------------------------------------------ append
struct AppendArgs {
  const __nv_bfloat16* K;
  const __nv_bfloat16* V;
  long long ks, vs;         // token strides (elements) of K / V; heads at stride 128 inside a token
  const int64_t* slot;      // [N]
  uint8_t* kv;
  uint32_t* pool;           // [P][16]
  int* pool_cnt;            // [0] next free extension block, [1] error (pool exhausted)
  int N, H, P, reuse;
};

// one warp per (segment start token, head, K/V, half): a segment = consecutive tokens of one page in this call; the
// warp owns the record for the whole call (no races on the state bytes shared by rows g / g + 8 or on the header)
__global__ void __launch_bounds__(256) kv_append_kernel(const AppendArgs a) {
  const int lane = threadIdx.x & 31;
  const long long w = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
  const int q = (int)(w % (a.H * 4));
  const long long n = w / (a.H * 4);
  if (n >= a.N) return;
  const int h = q >> 2, kvi = (q >> 1) & 1, half = q & 1;
  const long long s0 = a.slot[n];
  if (s0 < 0) return;
  const long long blk = s0 / PAGE;
  if (n > 0) {
    const long long sp = a.slot[n - 1];
    if (sp >= 0 && sp + 1 == s0 && sp / PAGE == blk) return;    // not the start of a segment
  }
  uint8_t* const rec = record(a.kv, blk, a.H, h, kvi, half);
  uint32_t* const hdr = (uint32_t*)(rec + REC);
  uint32_t* const slots = hdr + 1;
  uint32_t head;
  if (s0 % PAGE == 0) {                             // fresh page: unwritten rows decode to +-0, no stale escapes
    int keep = -1;                                  // reuse: the page's old extension chain, recycled in order
    if (a.reuse) {
      const uint32_t oh = *(volatile uint32_t*)hdr;
      if ((oh & 16u) && (int)(oh >> 8) < a.P) keep = (int)(oh >> 8);
    }
    __syncwarp();
    for (int i = lane; i < RKV / 16; i += 32) ((uint4*)rec)[i] = make_uint4(0, 0, 0, 0);
    if (keep >= 0 && lane == 0) {                   // empty every block of the kept chain (the links stay)
      int hops = 0;
      for (int b = keep; b >= 0 && b < a.P && hops < 4096; ++hops) {
        const uint32_t w0 = a.pool[(size_t)b * 16];
        a.pool[(size_t)b * 16] = w0 & ~0xFFu;
        b = (int)(w0 >> 8) - 1;
      }
    }
    head = keep >= 0 ? (16u | ((uint32_t)keep << 8)) : 0u;
    __syncwarp();
  } else {
    head = *(volatile uint32_t*)hdr;
  }
  int cnt = head & 15;
  int xh = (head & 16) ? (int)(head >> 8) : -1;     // first extension block
  int xt = -1, xc = 0;                              // tail block and its count (lane 0, found on first use)
  auto put = [&](uint32_t e) {                      // lane 0 only: append one escape entry
    if (cnt < NSLOT) {                              // slots first (entries are independent patches: any order)
      slots[cnt++] = e;
      return;
    }
    if (xh >= 0 && xt < 0) {                        // continue an existing chain: its first non-full block
      xt = xh;
      for (int hops = 0; hops < 4096; ++hops) {
        const uint32_t w0 = a.pool[(size_t)xt * 16];
        const int nx = (int)(w0 >> 8) - 1;
        xc = (int)(w0 & 0xFF);
        if (xc < XSLOT || nx < 0 || nx >= a.P) break;
        xt = nx;
      }
    }
    if (xt >= 0 && xc == XSLOT) {                   // full: move on to a recycled block of the chain, if any
      const int nx = (int)(a.pool[(size_t)xt * 16] >> 8) - 1;
      if (nx >= 0 && nx < a.P) {
        xt = nx;
        xc = (int)(a.pool[(size_t)xt * 16] & 0xFF);
      }
    }
    if (xt < 0 || xc == XSLOT) {
      const int nb = atomicAdd(a.pool_cnt, 1);
      if (nb >= a.P) {
        atomicExch(a.pool_cnt + 1, 1);              // pool exhausted: the host raises (never silently lossy)
        return;
      }
      a.pool[(size_t)nb * 16] = 0;
      if (xt < 0) xh = nb;
      else a.pool[(size_t)xt * 16] = (a.pool[(size_t)xt * 16] & 0xFFu) | ((uint32_t)(nb + 1) << 8);
      xt = nb;
      xc = 0;
    }
    a.pool[(size_t)xt * 16 + 1 + xc] = e;
    ++xc;
    a.pool[(size_t)xt * 16] = (a.pool[(size_t)xt * 16] & ~0xFFu) | (uint32_t)xc;
  };
  for (long long m = n; m < a.N; ++m) {
    const long long sm = a.slot[m];
    if (m > n && !(sm >= 0 && sm == a.slot[m - 1] + 1 && sm / PAGE == blk)) break;   // warp-uniform
    const int r = (int)(sm % PAGE);
    const __nv_bfloat16* src = (kvi ? a.V + m * a.vs : a.K + m * a.ks) + h * HD + half * 64;
    const uint32_t pr = *(const uint32_t*)(src + 2 * lane);
    const uint32_t b0 = pr & 0xFFFFu, b1 = pr >> 16;
    const int e0 = (int)((b0 >> 7) & 0xFF), e1 = (int)((b1 >> 7) & 0xFF);
    int E = max(e0, e1);
#pragma unroll
    for (int o = 16; o; o >>= 1) E = max(E, __shfl_xor_sync(FULL, E, o));
    const int D0 = E - e0, D1 = E - e1;
    const bool x0 = D0 > 14, x1 = D1 > 14;
    const uint32_t l0 = ((x0 ? 0u : (b0 & 0x7Fu)) << 1) | (x1 ? 0u : (b1 >> 15));
    const uint32_t l1 = ((x1 ? 0u : (b1 & 0x7Fu)) << 1) | (x0 ? 0u : (b0 >> 15));
    *(uint16_t*)(rec + leaf_off(r, lane)) = (uint16_t)(l0 | (l1 << 8));
    uint16_t* const st = (uint16_t*)(rec + state_off(r, lane));
    const int sh = 4 * (r >> 3);
    const uint32_t d0 = x0 ? 15u : (uint32_t)D0, d1 = x1 ? 15u : (uint32_t)D1;
    uint32_t old = *st;
    old = (old & ~((0xFu << sh) | (0xFu << (8 + sh)))) | (d0 << sh) | (d1 << (8 + sh));
    *st = (uint16_t)old;
    if (lane == 0) rec[1536 + 2 * (r & 7) + (r >> 3)] = (uint8_t)E;
    const unsigned bl0 = __ballot_sync(FULL, x0), bl1 = __ballot_sync(FULL, x1);
    const int tot = __popc(bl0) + __popc(bl1);
    if (tot) {                                      // warp-uniform
      const uint32_t en0 = (b0 << 16) | ((uint32_t)r << 6) | (uint32_t)(2 * lane);
      const uint32_t en1 = (b1 << 16) | ((uint32_t)r << 6) | (uint32_t)(2 * lane + 1);
      if (cnt + tot <= NSLOT) {
        const unsigned below = (1u << lane) - 1;
        const int pre = __popc(bl0 & below) + __popc(bl1 & below);
        if (x0) slots[cnt + pre] = en0;
        if (x1) slots[cnt + pre + (x0 ? 1 : 0)] = en1;
        cnt += tot;
      } else {                                      // rare: lane 0 appends one by one (slots, then extensions)
        for (int sl = 0; sl < 32; ++sl) {
          const uint32_t f0 = __shfl_sync(FULL, en0, sl), f1 = __shfl_sync(FULL, en1, sl);
          if (lane == 0) {
            if ((bl0 >> sl) & 1) put(f0);
            if ((bl1 >> sl) & 1) put(f1);
          }
        }
        cnt = __shfl_sync(FULL, cnt, 0);
        xh = __shfl_sync(FULL, xh, 0);
        xt = __shfl_sync(FULL, xt, 0);
        xc = __shfl_sync(FULL, xc, 0);
      }
    }
  }
  if (lane == 0) *hdr = (uint32_t)cnt | (xh >= 0 ? 16u | ((uint32_t)xh << 8) : 0u);
}

// ------------------------------------------------------------------------------------------------ record decode
// lane (g, t) of a warp decodes its fragment of a record: A[s][0..3] = k-step s, rows g / g + 8, cols 16 s + 2 t (+1)
// and 16 s + 8 + 2 t (+1) (bf16 pairs), escapes patched
__device__ __forceinline__ void patch_one(uint32_t (&A)[4][4], uint32_t e, int lane) {
  const int row = (int)((e >> 6) & 15), col = (int)(e & 63);
  const int s = col >> 4, cc = col & 15, t = (cc & 7) >> 1, hi = cc >> 3, bit = cc & 1;
  if (lane == (row & 7) * 4 + t) {
    const int reg = (row >> 3) + 2 * hi;
    const uint32_t sh = 16u * (uint32_t)bit, v = (e >> 16) << sh, msk = 0xFFFFu << sh;
#pragma unroll
    for (int ss = 0; ss < 4; ++ss)
#pragma unroll
      for (int q = 0; q < 4; ++q)
        if (ss == s && q == reg) A[ss][q] = (A[ss][q] & ~msk) | v;
  }
}
__device__ __forceinline__ void decode_record(const uint8_t* rec, const uint32_t* pool, int P, int lane,
                                              uint32_t (&A)[4][4]) {
  const uint4 c0 = *(const uint4*)(rec + lane * 16);
  const uint4 c1 = *(const uint4*)(rec + 512 + lane * 16);
  const uint4 c2 = *(const uint4*)(rec + 1024 + lane * 16);
  const uint32_t ee = *(const uint16_t*)(rec + 1536 + 2 * (lane >> 2));
  const uint32_t E0 = (ee & 0xFFu) * 0x01010101u, E1 = ((ee >> 8) & 0xFFu) * 0x01010101u;
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const uint4& c = s < 2 ? c0 : c1;
    const uint32_t lw0 = (s & 1) ? c.z : c.x, lw1 = (s & 1) ? c.w : c.y;
    const uint32_t nw = s == 0 ? c2.x : s == 1 ? c2.y : s == 2 ? c2.z : c2.w;
    dec4r(lw0, E0 - (nw & 0x0F0F0F0Fu), A[s][0], A[s][2]);
    dec4r(lw1, E1 - ((nw >> 4) & 0x0F0F0F0Fu), A[s][1], A[s][3]);
  }
  const uint32_t head = *(const uint32_t*)(rec + REC);
  const int cnt = min((int)(head & 15), NSLOT);    // (guards: a never-written page holds garbage)
  const uint32_t* sl = (const uint32_t*)(rec + REC + 4);
  for (int i = 0; i < cnt; ++i) patch_one(A, sl[i], lane);
  if (head & 16) {
    int hops = 0;
    for (int b = (int)(head >> 8); b >= 0 && b < P && hops < 4096; ++hops) {
      const uint32_t w0 = pool[(size_t)b * 16];
      const int xc = min((int)(w0 & 0xFF), XSLOT);
      for (int i = 0; i < xc; ++i) patch_one(A, pool[(size_t)b * 16 + 1 + i], lane);
      b = (int)(w0 >> 8) - 1;
    }
  }
}

// dequant: every record -> dense paged BF16 (vLLM flash layout [num_blocks, 16, H, 128] for K and for V)
__global__ void __launch_bounds__(256) kv_dequant_kernel(const uint8_t* kv, const uint32_t* pool, int P,
                                                         __nv_bfloat16* Kd, __nv_bfloat16* Vd, long long nrec, int H) {
  const int lane = threadIdx.x & 31;
  const long long w = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
  if (w >= nrec) return;
  const int half = (int)(w & 1), kvi = (int)((w >> 1) & 1);
  const long long bh = w >> 2;
  const int h = (int)(bh % H);
  const long long blk = bh / H;
  uint32_t A[4][4];
  decode_record(kv + (size_t)w * RKV, pool, P, lane, A);
  __nv_bfloat16* out = kvi ? Vd : Kd;
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int s = 0; s < 4; ++s)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int row = g + 8 * (q & 1), col = 16 * s + 8 * (q >> 1) + 2 * t;
      *(uint32_t*)(out + (((size_t)blk * PAGE + row) * H + h) * HD + half * 64 + col) = A[s][q];
    }
}

// dequant_blocks: listed pages -> dense pages (output page i = cache block blocks[i]; an id outside the cache gives 0)
__global__ void __launch_bounds__(256) kv_dequant_blocks_kernel(const uint8_t* kv, const uint32_t* pool, int P,
                                                                const int* blocks, long long nbc, long long n, int H,
                                                                __nv_bfloat16* Kd, __nv_bfloat16* Vd) {
  const int lane = threadIdx.x & 31;
  const long long w = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
  if (w >= n * H * 4) return;
  const int half = (int)(w & 1), kvi = (int)((w >> 1) & 1);
  const long long ih = w >> 2;
  const int h = (int)(ih % H);
  const long long i = ih / H;
  const long long blk = blocks[i];
  uint32_t A[4][4];
  if (blk >= 0 && blk < nbc) {
    decode_record(kv + (((size_t)blk * H + h) * 4 + kvi * 2 + half) * RKV, pool, P, lane, A);
  } else {
#pragma unroll
    for (int s = 0; s < 4; ++s)
#pragma unroll
      for (int q = 0; q < 4; ++q) A[s][q] = 0u;
  }
  __nv_bfloat16* out = kvi ? Vd : Kd;
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int s = 0; s < 4; ++s)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int row = g + 8 * (q & 1), col = 16 * s + 8 * (q >> 1) + 2 * t;
      *(uint32_t*)(out + (((size_t)i * PAGE + row) * H + h) * HD + half * 64 + col) = A[s][q];
    }
}

int64_t page_bytes(int64_t H) { return H * 2 * 2 * RKV; }

void kv_append(torch::Tensor K, torch::Tensor V, torch::Tensor slot, torch::Tensor kv, torch::Tensor pool,
               torch::Tensor pool_cnt, bool reuse) {
  const at::cuda::OptionalCUDAGuard guard(K.device());
  TORCH_CHECK(K.dtype() == torch::kBFloat16 && V.dtype() == torch::kBFloat16 && K.dim() == 3 && V.dim() == 3, "kv_append: K/V");
  TORCH_CHECK(K.size(2) == HD && V.size(2) == HD && K.stride(2) == 1 && V.stride(2) == 1 && K.stride(1) == HD &&
              V.stride(1) == HD, "kv_append: [N, H, 128] with heads contiguous");
  TORCH_CHECK(slot.dtype() == torch::kInt64 && slot.is_contiguous() && slot.numel() == K.size(0), "kv_append: slots");
  TORCH_CHECK(kv.dtype() == torch::kUInt8 && kv.is_contiguous() && kv.dim() == 5 && kv.size(1) == K.size(1) &&
              kv.size(4) == RKV, "kv_append: cache [blocks, H, 2, 2, 1600]");
  TORCH_CHECK(pool.dtype() == torch::kInt32 && pool.dim() == 2 && pool.size(1) == 16, "kv_append: pool [P, 16]");
  AppendArgs a;
  a.K = (const __nv_bfloat16*)K.data_ptr();
  a.V = (const __nv_bfloat16*)V.data_ptr();
  a.ks = K.stride(0);
  a.vs = V.stride(0);
  a.slot = slot.data_ptr<int64_t>();
  a.kv = kv.data_ptr<uint8_t>();
  a.pool = (uint32_t*)pool.data_ptr<int>();
  a.pool_cnt = pool_cnt.data_ptr<int>();
  a.N = (int)K.size(0);
  a.H = (int)K.size(1);
  a.P = (int)pool.size(0);
  a.reuse = reuse ? 1 : 0;
  const long long warps = (long long)a.N * a.H * 4;
  if (warps == 0) return;
  kv_append_kernel<<<(unsigned)((warps + 7) / 8), 256, 0, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void kv_dequant(torch::Tensor kv, torch::Tensor pool, torch::Tensor Kd, torch::Tensor Vd) {
  const at::cuda::OptionalCUDAGuard guard(kv.device());
  const long long nb = kv.size(0), H = kv.size(1);
  TORCH_CHECK(Kd.is_contiguous() && Vd.is_contiguous() && Kd.numel() == nb * PAGE * H * HD && Vd.numel() == Kd.numel(),
              "kv_dequant: outputs");
  const long long nrec = nb * H * 4;
  kv_dequant_kernel<<<(unsigned)((nrec + 7) / 8), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      kv.data_ptr<uint8_t>(), (const uint32_t*)pool.data_ptr<int>(), (int)pool.size(0), (__nv_bfloat16*)Kd.data_ptr(),
      (__nv_bfloat16*)Vd.data_ptr(), nrec, (int)H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void kv_dequant_blocks(torch::Tensor kv, torch::Tensor pool, torch::Tensor blocks, torch::Tensor Kd, torch::Tensor Vd) {
  const at::cuda::OptionalCUDAGuard guard(kv.device());
  TORCH_CHECK(kv.dtype() == torch::kUInt8 && kv.is_contiguous() && kv.dim() == 5 && kv.size(4) == RKV,
              "kv_dequant_blocks: cache [blocks, H, 2, 2, 1600]");
  TORCH_CHECK(blocks.dtype() == torch::kInt32 && blocks.is_contiguous() && blocks.dim() == 1,
              "kv_dequant_blocks: blocks int32 [n]");
  const long long n = blocks.numel(), H = kv.size(1);
  TORCH_CHECK(Kd.dtype() == torch::kBFloat16 && Vd.dtype() == torch::kBFloat16 && Kd.is_contiguous() &&
              Vd.is_contiguous() && Kd.numel() >= n * PAGE * H * HD && Vd.numel() >= n * PAGE * H * HD,
              "kv_dequant_blocks: outputs [n, 16, H, 128]");
  if (n == 0) return;
  const long long nw = n * H * 4;
  kv_dequant_blocks_kernel<<<(unsigned)((nw + 7) / 8), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      kv.data_ptr<uint8_t>(), (const uint32_t*)pool.data_ptr<int>(), (int)pool.size(0), blocks.data_ptr<int>(),
      kv.size(0), n, (int)H, (__nv_bfloat16*)Kd.data_ptr(), (__nv_bfloat16*)Vd.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------------------------------------ decode attention
// One query token per sequence, GQA (G query heads per KV head), pages read straight from the state-tree cache:
//   S = q K^T  : A = q (rows = the G heads, padded to 16), B = K^T: decoding a K record yields exactly the B fragments
//                (lane (g, t): token g / g + 8, dims 2t, 2t+1 of each k-step) -> mma.m16n8k16, no shared memory
//   online softmax over the 16 tokens of a page (quad reductions), P from the S accumulators (FlashAttention-2)
//   O += P V   : B = V (k = tokens, n = dims): the V record's fragments transposed by movmatrix (8 x 8, in registers)
// Grid: B x Hkv x S splits of the sequence's pages; NW warps per CTA interleave the split's pages and are merged in
// shared memory; with S > 1 the partials (m, l, unnormalised O) go to a workspace and kv_merge combines them.
__device__ __forceinline__ uint32_t movtrans(uint32_t a) {
  uint32_t d;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(d) : "r"(a));
  return d;
}
__device__ __forceinline__ void hmma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                     uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *(const uint32_t*)&v;
}

struct AttnArgs {
  const __nv_bfloat16* q;   // [B, Hq, 128], token stride qs
  long long qs;
  const uint8_t* kv;
  const uint32_t* pool;
  int P;
  const int* bt;            // block table [B, bts]
  int bts;
  const int* lens;          // [B] tokens in the cache (incl. this step's)
  const int* starts;        // [B] first valid token (left padding), or null: 0
  __nv_bfloat16* out;       // [B, Hq, 128], token stride os
  long long os;
  float* wo;                // [B, Hq, S, 128]
  float* wml;               // [B, Hq, S, 2]
  float sl2;                // softmax scale * log2(e)
  int Hkv, S, minpps;
};

template <int G, int NW>
__global__ void __launch_bounds__(NW * 32) kv_attn_kernel(const AttnArgs a) {
  extern __shared__ float smf[];                    // NW x (G x 128 + 2 G)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int j = blockIdx.x % a.S, bh = blockIdx.x / a.S;
  const int h = bh % a.Hkv, b = bh / a.Hkv, Hq = a.Hkv * G;
  const int len = a.lens[b], st0 = a.starts ? a.starts[b] : 0;
  const int pfirst = st0 / PAGE, npg = (len + PAGE - 1) / PAGE, pps = max(a.minpps, (npg - pfirst + a.S - 1) / a.S);
  const int p0 = pfirst + j * pps, p1 = min(npg, p0 + pps);
  uint32_t qa[8][2];
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    if (g < G) {
      const __nv_bfloat16* qp = a.q + (size_t)b * a.qs + (size_t)(h * G + g) * HD;
      qa[ks][0] = *(const uint32_t*)(qp + 16 * ks + 2 * t);
      qa[ks][1] = *(const uint32_t*)(qp + 16 * ks + 8 + 2 * t);
    } else {
      qa[ks][0] = qa[ks][1] = 0u;
    }
  }
  float o[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m = -INFINITY, l = 0.f;
  for (int p = p0 + warp; p < p1; p += NW) {
    const int blk = a.bt[(size_t)b * a.bts + p];
    const uint8_t* base = a.kv + ((size_t)blk * a.Hkv + h) * 4 * RKV;   // K half 0, K half 1, V half 0, V half 1
    float s0[4] = {0.f, 0.f, 0.f, 0.f}, s1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      uint32_t A[4][4];
      decode_record(base + hf * RKV, a.pool, a.P, lane, A);
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        hmma(s0, qa[hf * 4 + s][0], 0u, qa[hf * 4 + s][1], 0u, A[s][0], A[s][2]);   // tokens 0..7
        hmma(s1, qa[hf * 4 + s][0], 0u, qa[hf * 4 + s][1], 0u, A[s][1], A[s][3]);   // tokens 8..15
      }
    }
    const int hiv = len - p * PAGE, lov = st0 - p * PAGE;          // valid tokens of this page: [lov, hiv)
    const float x0 = (2 * t < hiv && 2 * t >= lov) ? s0[0] * a.sl2 : -INFINITY;
    const float x1 = (2 * t + 1 < hiv && 2 * t + 1 >= lov) ? s0[1] * a.sl2 : -INFINITY;
    const float x2 = (8 + 2 * t < hiv && 8 + 2 * t >= lov) ? s1[0] * a.sl2 : -INFINITY;
    const float x3 = (9 + 2 * t < hiv && 9 + 2 * t >= lov) ? s1[1] * a.sl2 : -INFINITY;
    float mp = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
    mp = fmaxf(mp, __shfl_xor_sync(FULL, mp, 1));
    mp = fmaxf(mp, __shfl_xor_sync(FULL, mp, 2));
    const float mn = fmaxf(m, mp);                  // finite: every page in [pfirst, npg) has >= 1 valid token
    const float al = exp2f(m - mn);
    const float e0 = exp2f(x0 - mn), e1 = exp2f(x1 - mn), e2 = exp2f(x2 - mn), e3 = exp2f(x3 - mn);
    float ps = (e0 + e1) + (e2 + e3);
    ps += __shfl_xor_sync(FULL, ps, 1);
    ps += __shfl_xor_sync(FULL, ps, 2);
    l = l * al + ps;
    m = mn;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      o[i][0] *= al;
      o[i][1] *= al;
    }
    const uint32_t pa0 = pack_bf16(e0, e1), pa2 = pack_bf16(e2, e3);
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      uint32_t A[4][4];
      decode_record(base + (2 + hf) * RKV, a.pool, a.P, lane, A);
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int nt = 2 * (4 * hf + s);
        hmma(o[nt], pa0, 0u, pa2, 0u, movtrans(A[s][0]), movtrans(A[s][1]));      // dims 16 s .. + 7
        hmma(o[nt + 1], pa0, 0u, pa2, 0u, movtrans(A[s][2]), movtrans(A[s][3]));  // dims 16 s + 8 .. + 15
      }
    }
  }
  // merge the warps of this CTA (each covered a subset of the split's pages)
  constexpr int WS = G * 128 + 2 * G;
  float* so = smf + warp * WS;
  if (g < G) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      so[g * 128 + i * 8 + 2 * t] = o[i][0];
      so[g * 128 + i * 8 + 2 * t + 1] = o[i][1];
    }
    if (t == 0) {
      so[G * 128 + 2 * g] = m;
      so[G * 128 + 2 * g + 1] = l;
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < G * 128; i += NW * 32) {
    const int r = i >> 7, d = i & 127;
    float M = -INFINITY;
#pragma unroll
    for (int w = 0; w < NW; ++w) M = fmaxf(M, smf[w * WS + G * 128 + 2 * r]);
    float L = 0.f, O = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float mw = smf[w * WS + G * 128 + 2 * r];
      if (mw == -INFINITY) continue;
      const float f = exp2f(mw - M);
      L += f * smf[w * WS + G * 128 + 2 * r + 1];
      O += f * smf[w * WS + r * 128 + d];
    }
    if (a.S == 1) {
      a.out[(size_t)b * a.os + (size_t)(h * G + r) * HD + d] = __float2bfloat16(L > 0.f ? O / L : 0.f);
    } else {
      const size_t ix = (size_t)(b * Hq + h * G + r) * a.S + j;
      a.wo[ix * HD + d] = O;
      if (d == 0) {
        a.wml[ix * 2] = M;
        a.wml[ix * 2 + 1] = L;
      }
    }
  }
}

__global__ void __launch_bounds__(128) kv_merge_kernel(const float* wo, const float* wml, __nv_bfloat16* out,
                                                       long long os, int Hq, int S) {
  const int bq = blockIdx.x, d = threadIdx.x;
  float M = -INFINITY;
  for (int j = 0; j < S; ++j) M = fmaxf(M, wml[((size_t)bq * S + j) * 2]);
  float L = 0.f, O = 0.f;
  for (int j = 0; j < S; ++j) {
    const float mj = wml[((size_t)bq * S + j) * 2];
    if (mj == -INFINITY) continue;
    const float f = exp2f(mj - M);
    L += f * wml[((size_t)bq * S + j) * 2 + 1];
    O += f * wo[((size_t)bq * S + j) * HD + d];
  }
  out[(size_t)(bq / Hq) * os + (size_t)(bq % Hq) * HD + d] = __float2bfloat16(L > 0.f ? O / L : 0.f);
}

template <int G>
void attn_launch(const AttnArgs& a, int B, cudaStream_t st) {
  constexpr int NW = 4;
  const int smem = NW * (G * 128 + 2 * G) * 4;
  kv_attn_kernel<G, NW><<<(unsigned)(B * a.Hkv * a.S), NW * 32, smem, st>>>(a);
}

void kv_attn(torch::Tensor q, torch::Tensor kv, torch::Tensor pool, torch::Tensor bt, torch::Tensor lens,
             torch::Tensor out, torch::Tensor wo, torch::Tensor wml, double scale, int64_t S,
             c10::optional<torch::Tensor> starts, int64_t minpps) {
  const at::cuda::OptionalCUDAGuard guard(q.device());
  TORCH_CHECK(q.dtype() == torch::kBFloat16 && q.dim() == 3 && q.size(2) == HD && q.stride(2) == 1 && q.stride(1) == HD,
              "kv_attn: q [B, Hq, 128]");
  TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.dim() == 3 && out.size(1) == q.size(1) && out.stride(2) == 1 &&
              out.stride(1) == HD, "kv_attn: out [B, Hq, 128]");
  TORCH_CHECK(kv.dtype() == torch::kUInt8 && kv.dim() == 5 && kv.size(4) == RKV, "kv_attn: cache");
  TORCH_CHECK(bt.dtype() == torch::kInt32 && bt.dim() == 2 && bt.stride(1) == 1 && lens.dtype() == torch::kInt32,
              "kv_attn: block table / lens int32");
  const int B = (int)q.size(0), Hq = (int)q.size(1), Hkv = (int)kv.size(1);
  TORCH_CHECK(Hq % Hkv == 0 && S >= 1, "kv_attn: heads / splits");
  TORCH_CHECK(S == 1 || (wo.numel() >= (int64_t)B * Hq * S * HD && wml.numel() >= (int64_t)B * Hq * S * 2), "kv_attn: workspace");
  AttnArgs a;
  a.q = (const __nv_bfloat16*)q.data_ptr();
  a.qs = q.stride(0);
  a.kv = kv.data_ptr<uint8_t>();
  a.pool = (const uint32_t*)pool.data_ptr<int>();
  a.P = (int)pool.size(0);
  a.bt = bt.data_ptr<int>();
  a.bts = (int)bt.stride(0);
  a.lens = lens.data_ptr<int>();
  a.starts = nullptr;
  if (starts.has_value() && starts->defined()) {
    TORCH_CHECK(starts->dtype() == torch::kInt32 && starts->numel() >= q.size(0), "kv_attn: starts int32 [B]");
    a.starts = starts->data_ptr<int>();
  }
  a.out = (__nv_bfloat16*)out.data_ptr();
  a.os = out.stride(0);
  a.wo = wo.data_ptr<float>();
  a.wml = wml.data_ptr<float>();
  a.sl2 = (float)(scale * 1.4426950408889634);
  a.Hkv = Hkv;
  a.S = (int)S;
  a.minpps = (int)(minpps < 1 ? 1 : minpps);
  auto st = at::cuda::getCurrentCUDAStream();
  switch (Hq / Hkv) {
    case 1: attn_launch<1>(a, B, st); break;
    case 2: attn_launch<2>(a, B, st); break;
    case 4: attn_launch<4>(a, B, st); break;
    case 8: attn_launch<8>(a, B, st); break;
    default: TORCH_CHECK(false, "kv_attn: query heads per KV head must be 1, 2, 4 or 8");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (S > 1) {
    kv_merge_kernel<<<(unsigned)(B * Hq), 128, 0, st>>>(a.wo, a.wml, a.out, a.os, Hq, (int)S);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("append", &kv_append, "append tokens (K, V [N, H, 128] bf16) at slots into the state-tree page cache",
        py::arg("K"), py::arg("V"), py::arg("slot"), py::arg("kv"), py::arg("pool"), py::arg("pool_cnt"),
        py::arg("reuse") = false);
  m.def("dequant_blocks", &kv_dequant_blocks, "listed pages -> dense BF16 K and V [n, 16, H, 128]");
  m.def("version", []() { return 2; });
  m.def("dequant", &kv_dequant, "every page -> dense paged BF16 K and V [blocks, 16, H, 128]");
  m.def("page_bytes", &page_bytes);
  m.def("attn", &kv_attn, "decode attention (one query token per sequence) over state-tree pages", py::arg("q"),
        py::arg("kv"), py::arg("pool"), py::arg("bt"), py::arg("lens"), py::arg("out"), py::arg("wo"), py::arg("wml"),
        py::arg("scale"), py::arg("S"), py::arg("starts") = py::none(), py::arg("minpps") = 1);
}
