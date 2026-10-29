// statetree: lossless state-tree BF16 linear kernels (A100 / sm_80+).
// Format: per 16-row x 64-column block one 1552-byte record (leaves: sign + 7-bit mantissa; 4-bit exponent states;
// per-row 8-bit group exponent node), escapes (|w| < 2^-14 of the group max) patched to their exact bits.
//   mv   batch <= 32 : warp-per-tile matrix-vector kernel, split-K merged by the last CTA (semaphore)
//   tm   batch 12-512: Marlin-style fused decode + mma.m16n8k16 GEMM (stream-K, cp.async pipeline)
//   dec  any batch   : tree -> BF16 (for decode + cuBLAS)
// The tm kernel structure follows Marlin (IST-DASLab, Apache-2.0) with the INT4 dequant replaced by the tree decode.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>

__device__ __forceinline__ uint4 ldw(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N_> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N_)); }
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t wsel(const uint4& v, int s) {
  return s == 0 ? v.x : (s == 1 ? v.y : (s == 2 ? v.z : v.w));
}
// 4 weights: leaf bytes (format v2: m << 1 | sign of the pair partner), states (bytes 0..15), node exponent *
// 0x01010101 -> bf16 pairs (w0,w1), (w2,w3): bytes [leaf0, exp0, leaf1, exp1] rotated right by one bit put w1's sign
// (leaf0 bit 0) on bit 31 and w0's sign (leaf1 bit 0) on bit 15
__device__ __forceinline__ void dec4(uint32_t lf, uint32_t d, uint32_t E4, uint32_t& p01, uint32_t& p23) {
  const uint32_t e4 = E4 - d;
  const uint32_t a = __byte_perm(lf, e4, 0x5140), b = __byte_perm(lf, e4, 0x7362);
  p01 = __funnelshift_r(a, a, 1);
  p23 = __funnelshift_r(b, b, 1);
}
// A fragment (a0..a3) of k-step s from a lane's record chunks c0, c1 (leaves) and c2 (states)
__device__ __forceinline__ void dec_step5(const uint4& c0, const uint4& c1, const uint4& c2, uint32_t E0, uint32_t E1,
                                          int s, uint32_t* A) {
  const uint4& c = (s < 2) ? c0 : c1;
  const uint32_t lw0 = (s & 1) ? c.z : c.x, lw1 = (s & 1) ? c.w : c.y;
  const uint32_t nw = wsel(c2, s);
  dec4(lw0, nw & 0x0F0F0F0Fu, E0, A[0], A[2]);
  dec4(lw1, (nw >> 4) & 0x0F0F0F0Fu, E1, A[1], A[3]);
}


struct A7 {
  const uint8_t* T; const int* EO; const uint32_t* EL; const uint16_t* X; __nv_bfloat16* Y; float* WS; int* CNT;
  int M, N, B, WM, WK;
};

// patch escapes of (tile-group) list [lo, hi) for this lane and k-step s into the A fragment registers
__device__ __forceinline__ void patch_regs(const uint32_t* EL, int lo, int hi, int lane, int s, uint32_t* A) {
  for (int e = lo; e < hi; ++e) {
    const uint32_t v = __ldg(EL + e);
    if ((int)((v >> 5) & 31) == lane && (int)((v >> 3) & 3) == s) {
      const uint32_t sh = (v & 1u) * 16u, mask = 0xFFFFu << sh, val = (v >> 16) << sh;
      const int r = (v >> 1) & 3;
#pragma unroll
      for (int q = 0; q < 4; ++q) A[q] = (r == q) ? ((A[q] & ~mask) | val) : A[q];
    }
  }
}

// Escape patching for records with many escapes (the down projections of some MoE layers: up to ~430 of a record's
// 1024 weights): the warp loads the record's entries once, 32 at a time, and scatters them into a per-warp shared
// image (wv: 32 halves per lane, wm: one bit mask per lane, bit = s * 8 + q * 2 + half = entry & 31); each lane then
// patches only its own set bits at every k-step. patch_regs above re-reads the whole list at each of the 4 k-steps,
// one entry at a time (O(4 n) dependent loads per warp). lo / hi are warp-uniform; the image is reused per record.
constexpr int ESC_IMG_BYTES = 32 * 4 + 1024 * 2;                // one warp's escape image (mask words + halves)
__device__ __forceinline__ uint32_t esc_scatter(const uint32_t* EL, int lo, int hi, int lane, uint32_t* wm,
                                                uint16_t* wv) {
  __syncwarp();                                                  // every lane is done with the previous image
  wm[lane] = 0u;
  __syncwarp();
#pragma unroll 2
  for (int e = lo + lane; e < hi; e += 32) {
    const uint32_t v = __ldg(EL + e);
    const int L = (v >> 5) & 31, b = v & 31;
    wv[L * 32 + b] = (uint16_t)(v >> 16);
    atomicOr(wm + L, 1u << b);
  }
  __syncwarp();
  return wm[lane];
}
__device__ __forceinline__ void esc_apply(uint32_t m, const uint16_t* wvl, int s, uint32_t* A) {
  uint32_t bits = (m >> (8 * s)) & 0xFFu;
  while (bits) {
    const int b = __ffs(bits) - 1;
    bits &= bits - 1;
    const int r = b >> 1;
    const uint32_t sh = (uint32_t)(b & 1) * 16u, mask = 0xFFFFu << sh, val = (uint32_t)wvl[8 * s + b] << sh;
#pragma unroll
    for (int q = 0; q < 4; ++q) A[q] = (r == q) ? ((A[q] & ~mask) | val) : A[q];
  }
}

__device__ __forceinline__ bool merge7(const A7& a, float* out, int elems, int tile_id, int ks, int KS) {
  __shared__ int s_last;
  float* ws = a.WS + (size_t)tile_id * KS * elems;
  for (int i = threadIdx.x; i < elems; i += blockDim.x) ws[(size_t)ks * elems + i] = out[i];
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) s_last = (atomicAdd(&a.CNT[tile_id], 1) == KS - 1);
  __syncthreads();
  if (!s_last) return false;
  __threadfence();
  for (int i = threadIdx.x; i < elems; i += blockDim.x) {
    float s = 0.f;
    for (int k = 0; k < KS; ++k) s += __ldcg(ws + (size_t)k * elems + i);
    out[i] = s;
  }
  __syncthreads();
  if (threadIdx.x == 0) a.CNT[tile_id] = 0;
  return true;
}

// ---------------------------------------------------------------------------------------------- mv7
template <int NT>
struct F7 { uint4 c0, c1, c2; uint32_t e; int lo, hi; uint32_t x[NT][8]; };

template <int NT>
__device__ __forceinline__ void load_f7(F7<NT>& f, const A7& a, int tile, int G, int lane, int NG) {
  const int g = lane >> 2, t = lane & 3;
  const size_t tg = (size_t)tile * NG + G;
  const uint8_t* base = a.T + tg * 1552;
  f.c0 = ldw(base + lane * 16);
  f.c1 = ldw(base + 512 + lane * 16);
  f.c2 = ldw(base + 1024 + lane * 16);
  f.e = __ldg((const uint16_t*)(base + 1536) + g);
  f.lo = __ldg(a.EO + tg);
  f.hi = __ldg(a.EO + tg + 1);
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int n = nt * 8 + g;
    if (n < a.B) {
      const uint32_t* xp = (const uint32_t*)(a.X + (size_t)n * a.N + G * 64 + 2 * t);
#pragma unroll
      for (int s = 0; s < 4; ++s) { f.x[nt][2 * s] = __ldg(xp + 8 * s); f.x[nt][2 * s + 1] = __ldg(xp + 8 * s + 4); }
    } else {
#pragma unroll
      for (int q = 0; q < 8; ++q) f.x[nt][q] = 0u;
    }
  }
}

template <int NT, int PF>
__global__ void __launch_bounds__(256) mv7_kernel(A7 a) {
  extern __shared__ float smem[];
  uint32_t (*s_em)[32] = (uint32_t (*)[32])(smem + (a.WK + 1) * a.WM * 16 * NT * 8);      // per-warp escape images (esc_scatter) after the
  uint16_t (*s_ev)[1024] = (uint16_t (*)[1024])(s_em + a.WM * a.WK);  // reduction buffers: ESC_IMG_BYTES per warp
  const int NG = a.N / 64;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int wm = warp % a.WM, wk = warp / a.WM;
  const int tile = blockIdx.x * a.WM + wm;
  const int GY = gridDim.y;
  const int cg0 = NG * blockIdx.y / GY, cg1 = NG * (blockIdx.y + 1) / GY;
  const int G0 = cg0 + (cg1 - cg0) * wk / a.WK, G1 = cg0 + (cg1 - cg0) * (wk + 1) / a.WK;
  float c[NT][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) c[nt][0] = c[nt][1] = c[nt][2] = c[nt][3] = 0.f;
  F7<NT> f[PF];
#pragma unroll
  for (int i = 0; i < PF; ++i)
    if (G0 + i < G1) load_f7<NT>(f[i], a, tile, G0 + i, lane, NG);
  for (int G = G0; G < G1; G += PF) {
#pragma unroll
    for (int i = 0; i < PF; ++i) {
      if (G + i < G1) {
        const uint32_t E0 = (f[i].e & 0xFFu) * 0x01010101u, E1 = ((f[i].e >> 8) & 0xFFu) * 0x01010101u;
        const uint32_t pm = (f[i].lo < f[i].hi) ? esc_scatter(a.EL, f[i].lo, f[i].hi, lane, s_em[warp], s_ev[warp]) : 0u;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          uint32_t A[4];
          dec_step5(f[i].c0, f[i].c1, f[i].c2, E0, E1, s, A);
          if (pm) esc_apply(pm, s_ev[warp] + lane * 32, s, A);
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) mma16816(c[nt], A, f[i].x[nt][2 * s], f[i].x[nt][2 * s + 1]);
        }
        if (G + i + PF < G1) load_f7<NT>(f[i], a, tile, G + i + PF, lane, NG);
      }
    }
  }
  const int R = a.WM * 16, NBC = NT * 8, EL = R * NBC;
  float* red = smem;
  float* out = smem + a.WK * EL;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int rr = wm * 16 + g, n = nt * 8 + 2 * t;
    red[wk * EL + n * R + rr] = c[nt][0];
    red[wk * EL + (n + 1) * R + rr] = c[nt][1];
    red[wk * EL + n * R + rr + 8] = c[nt][2];
    red[wk * EL + (n + 1) * R + rr + 8] = c[nt][3];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < EL; idx += blockDim.x) {
    float s = 0.f;
    for (int k = 0; k < a.WK; ++k) s += red[k * EL + idx];
    out[idx] = s;
  }
  __syncthreads();
  if (GY > 1 && !merge7(a, out, EL, blockIdx.x, blockIdx.y, GY)) return;
  for (int idx = threadIdx.x; idx < EL; idx += blockDim.x) {
    const int n = idx / R, r = idx % R;
    if (n < a.B) a.Y[(size_t)n * a.M + (size_t)blockIdx.x * R + r] = __float2bfloat16_rn(out[idx]);
  }
}

template <int NT, int PF>
void launch_mv7(const A7& a, int GY, cudaStream_t st) {
  dim3 grid(a.M / (16 * a.WM), GY);
  const size_t sm = (size_t)(a.WK + 1) * a.WM * 16 * NT * 8 * sizeof(float) + (size_t)a.WM * a.WK * ESC_IMG_BYTES;
  mv7_kernel<NT, PF><<<grid, 32 * a.WM * a.WK, sm, st>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static A7 make_a7(torch::Tensor T, torch::Tensor EO, torch::Tensor EL, torch::Tensor X, torch::Tensor Y,
                  torch::Tensor WS, torch::Tensor CNT) {
  A7 a;
  a.T = T.data_ptr<uint8_t>(); a.EO = EO.data_ptr<int>(); a.EL = (const uint32_t*)EL.data_ptr<int>();
  a.X = (const uint16_t*)X.data_ptr<at::BFloat16>(); a.Y = (__nv_bfloat16*)Y.data_ptr<at::BFloat16>();
  a.WS = WS.data_ptr<float>(); a.CNT = CNT.data_ptr<int>();
  a.M = Y.size(1); a.N = X.size(1); a.B = X.size(0); a.WM = 0; a.WK = 0;
  return a;
}

void mv7(torch::Tensor T, torch::Tensor EO, torch::Tensor EL, torch::Tensor X, torch::Tensor Y, torch::Tensor WS,
         torch::Tensor CNT, int64_t WM, int64_t WK, int64_t GY, int64_t PF) {
  const c10::cuda::OptionalCUDAGuard guard(T.device());
  A7 a = make_a7(T, EO, EL, X, Y, WS, CNT);
  a.WM = WM; a.WK = WK;
  TORCH_CHECK(a.B >= 1 && a.B <= 32 && a.M % (16 * a.WM) == 0 && a.N % 64 == 0 && a.WM * a.WK <= 8, "mv7 config");
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const int G = (int)GY;
  if (a.B <= 8) { if (PF == 4) launch_mv7<1, 4>(a, G, st); else launch_mv7<1, 2>(a, G, st); }
  else if (a.B <= 16) { if (PF == 4) launch_mv7<2, 4>(a, G, st); else launch_mv7<2, 2>(a, G, st); }
  else { if (PF == 4) launch_mv7<4, 4>(a, G, st); else launch_mv7<4, 2>(a, G, st); }
}

// ---------------------------------------------------------------------------------------------- mvs (Hopper)
// mv7 with the activations staged in shared memory. mv7 loads each record's activation fragments with 4-byte loads at
// the record's prefetch: at batch > 1 a warp's 8 loads per record touch 8 rows each (64 L1 wavefronts per 1552-byte
// record) and hold 8 registers per prefetched record. On the H100, whose HBM feeds each SM ~1.5x more bytes per cycle
// than the A100's, that costs 13-20% at batch 8 and more at 16-32 (Qwen3-8B layers). Here the block copies its slice
// of X (rows < B, the columns of its split-K range; padded rows: conflict-free) to shared memory once with cp.async,
// issued after the first records' loads so it overlaps them, and every k-step pair reads its B fragments with one
// ldmatrix.x4 per 8 batch rows. Rows >= B are not copied: they only feed output columns that are never stored.
// Records, decode, escapes (EO / EL lists, per-warp image) and mma order are mv7's: outputs are bit-identical to mv7 on
// the same (WM, WK, GY). Ranked by the tuner only on sm_90+ (statetree.linear.candidates); the A100 keeps mv7.
__device__ __forceinline__ void ldsm_x4s(uint32_t addr, uint32_t* r) { ldsm_x4(addr, r[0], r[1], r[2], r[3]); }

template <int NT>
struct FS { uint4 c0, c1, c2; uint32_t e; int lo, hi; };

__device__ __forceinline__ void load_fs(FS<1>& f, const A7& a, const uint8_t* rec, const int* eo, int lane) {
  f.c0 = ldw(rec + lane * 16);
  f.c1 = ldw(rec + 512 + lane * 16);
  f.c2 = ldw(rec + 1024 + lane * 16);
  f.e = __ldg((const uint16_t*)(rec + 1536) + (lane >> 2));
  f.lo = __ldg(eo);
  f.hi = __ldg(eo + 1);
}

static inline int mvs_xrow(int N, int GY) { return (N / 64 + GY - 1) / GY * 64 + 8; }   // bf16 per staged row

template <int NT, int PF>
__global__ void __launch_bounds__(256) mvs_kernel(A7 a) {
  extern __shared__ __align__(16) float smem[];
  const int NG = a.N / 64;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int wm = warp % a.WM, wk = warp / a.WM;
  const int tile = blockIdx.x * a.WM + wm;
  const int GY = gridDim.y;
  const int cg0 = NG * blockIdx.y / GY, cg1 = NG * (blockIdx.y + 1) / GY;
  const int G0 = cg0 + (cg1 - cg0) * wk / a.WK, G1 = cg0 + (cg1 - cg0) * (wk + 1) / a.WK;
  const int R = a.WM * 16, ELN = R * NT * 8;
  uint32_t (*s_em)[32] = (uint32_t (*)[32])(smem + (a.WK + 1) * ELN);   // per-warp escape images, as mv7
  uint16_t (*s_ev)[1024] = (uint16_t (*)[1024])(s_em + a.WM * a.WK);
  uint16_t* xs = (uint16_t*)(s_ev + a.WM * a.WK);                        // [8 NT rows][XROW]: rows < B staged
  const int XROW = (NG + GY - 1) / GY * 64 + 8;
  const uint8_t* rec0 = a.T + ((size_t)tile * NG + G0) * 1552;
  const int* eo0 = a.EO + (size_t)tile * NG + G0;
  float c[NT][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) c[nt][0] = c[nt][1] = c[nt][2] = c[nt][3] = 0.f;
  FS<1> f[PF];
#pragma unroll
  for (int i = 0; i < PF; ++i)
    if (G0 + i < G1) load_fs(f[i], a, rec0 + (size_t)i * 1552, eo0 + i, lane);
  {                                                                      // stage X's slice (overlaps the loads above)
    const int ch = (cg1 - cg0) * 8;                                      // 16-byte chunks per row
    for (int i = threadIdx.x; i < a.B * ch; i += blockDim.x) {
      const int n = i / ch, q = i - n * ch;
      cp_async16(smem_u32(xs + n * XROW + q * 8), a.X + (size_t)n * a.N + cg0 * 64 + q * 8, 16);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
  }
  // ldmatrix.x4 row addresses: lane l feeds matrix j = l / 8 (k-step s + j / 2, half j % 2), batch row nt * 8 + l % 8
  const uint32_t xbase = smem_u32(xs) + (uint32_t)(((lane & 7) * XROW + ((lane >> 4) & 1) * 16 + ((lane >> 3) & 1) * 8) * 2);
  for (int G = G0; G < G1; G += PF) {
#pragma unroll
    for (int i = 0; i < PF; ++i) {
      if (G + i < G1) {
        const uint32_t E0 = (f[i].e & 0xFFu) * 0x01010101u, E1 = ((f[i].e >> 8) & 0xFFu) * 0x01010101u;
        const uint32_t pm = (f[i].lo < f[i].hi) ? esc_scatter(a.EL, f[i].lo, f[i].hi, lane, s_em[warp], s_ev[warp]) : 0u;
        const uint32_t xg = xbase + (uint32_t)((G + i - cg0) * 64 * 2);
#pragma unroll
        for (int sp = 0; sp < 2; ++sp) {                                 // k-steps 2 sp, 2 sp + 1
          uint32_t xb[NT][4];
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) ldsm_x4s(xg + (uint32_t)((nt * 8 * XROW + 32 * sp) * 2), xb[nt]);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int s = 2 * sp + h;
            uint32_t A[4];
            dec_step5(f[i].c0, f[i].c1, f[i].c2, E0, E1, s, A);
            if (pm) esc_apply(pm, s_ev[warp] + lane * 32, s, A);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) mma16816(c[nt], A, xb[nt][2 * h], xb[nt][2 * h + 1]);
          }
        }
        if (G + i + PF < G1) load_fs(f[i], a, rec0 + (size_t)(G + i + PF - G0) * 1552, eo0 + G + i + PF - G0, lane);
      }
    }
  }
  float* red = smem;
  float* out = smem + a.WK * ELN;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int rr = wm * 16 + g, n = nt * 8 + 2 * t;
    red[wk * ELN + n * R + rr] = c[nt][0];
    red[wk * ELN + (n + 1) * R + rr] = c[nt][1];
    red[wk * ELN + n * R + rr + 8] = c[nt][2];
    red[wk * ELN + (n + 1) * R + rr + 8] = c[nt][3];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < ELN; idx += blockDim.x) {
    float s = 0.f;
    for (int k = 0; k < a.WK; ++k) s += red[k * ELN + idx];
    out[idx] = s;
  }
  __syncthreads();
  if (GY > 1 && !merge7(a, out, ELN, blockIdx.x, blockIdx.y, GY)) return;
  for (int idx = threadIdx.x; idx < ELN; idx += blockDim.x) {
    const int n = idx / R, r = idx % R;
    if (n < a.B) a.Y[(size_t)n * a.M + (size_t)blockIdx.x * R + r] = __float2bfloat16_rn(out[idx]);
  }
}

static size_t mvs_smem(int NT, int WM, int WK, int GY, int N) {     // ldmatrix reads 8 NT rows (rows >= B: unused)
  return (size_t)(WK + 1) * WM * 16 * NT * 8 * sizeof(float) + (size_t)WM * WK * ESC_IMG_BYTES +
         (size_t)NT * 8 * mvs_xrow(N, GY) * 2;
}

template <int NT, int PF>
void launch_mvs(const A7& a, int GY, cudaStream_t st) {
  dim3 grid(a.M / (16 * a.WM), GY);
  const size_t sm = mvs_smem(NT, a.WM, a.WK, GY, a.N);
  if (sm > 48 * 1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(mvs_kernel<NT, PF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
  mvs_kernel<NT, PF><<<grid, 32 * a.WM * a.WK, sm, st>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mvs(torch::Tensor T, torch::Tensor EO, torch::Tensor EL, torch::Tensor X, torch::Tensor Y, torch::Tensor WS,
         torch::Tensor CNT, int64_t WM, int64_t WK, int64_t GY, int64_t PF) {
  const c10::cuda::OptionalCUDAGuard guard(T.device());
  A7 a = make_a7(T, EO, EL, X, Y, WS, CNT);
  a.WM = WM; a.WK = WK;
  TORCH_CHECK(a.B >= 1 && a.B <= 32 && a.M % (16 * a.WM) == 0 && a.N % 64 == 0 && a.WM * a.WK <= 8 && GY >= 1 &&
              GY <= a.N / 64 && X.data_ptr() != nullptr && ((uintptr_t)X.data_ptr() & 15) == 0, "mvs config");
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const int G = (int)GY;
  if (a.B <= 8) { if (PF == 4) launch_mvs<1, 4>(a, G, st); else launch_mvs<1, 2>(a, G, st); }
  else if (a.B <= 16) { if (PF == 4) launch_mvs<2, 4>(a, G, st); else launch_mvs<2, 2>(a, G, st); }
  else { if (PF == 4) launch_mvs<4, 4>(a, G, st); else launch_mvs<4, 2>(a, G, st); }
}

int64_t mvs_smem_bytes(int64_t B, int64_t N, int64_t WM, int64_t WK, int64_t GY) {
  return (int64_t)mvs_smem(B <= 8 ? 1 : (B <= 16 ? 2 : 4), (int)WM, (int)WK, (int)GY, (int)N);
}

// ---------------------------------------------------------------------------------------------- mvh (Hopper, B <= 8)
// mv7 with a lean inner loop, for GPUs whose SMs rather than HBM limit the matrix-vector product. On the H100, above all
// under a power cap (the SM clock falls to 345-1100 MHz while HBM keeps streaming), mv7's ~175 instructions per
// 1552-byte record (8 activation loads, 2 escape-offset loads, divergence handling) bound it, not the bytes. Here, per
// record: 3 + 1 loads, the 4-k-step decode into 16 A registers, 2 ldmatrix.x4 of the staged activations (rows < B
// staged, row addresses clamped to B - 1), 4 mma. Escapes: one escape-offset load per lane per 32 records; a ballot
// marks the records that have entries (warp-uniform test), and only those patch (entries fetched 32 at a time, shuffled
// to their lane, register selects). mma order as mv7: outputs are bit-identical to mv7 for the same split.
__device__ __forceinline__ void patch16(uint32_t (&A)[4][4], const uint32_t* EL, int lo, int hi, int lane) {
#pragma unroll 1
  for (int base = lo; base < hi; base += 32) {
    const int e = base + lane;
    const uint32_t v = (e < hi) ? __ldg(EL + e) : 0u;
    uint32_t mask = __ballot_sync(0xFFFFFFFFu, e < hi);
#pragma unroll 1
    while (mask) {
      const int b = __ffs(mask) - 1;
      mask &= mask - 1;
      const uint32_t vv = __shfl_sync(0xFFFFFFFFu, v, b);
      const int f = ((int)((vv >> 5) & 31u) == lane) ? (int)((vv >> 1) & 15u) : -1;      // k-step * 4 + register
      const uint32_t sh = (vv & 1u) * 16u, keep = ~(0xFFFFu << sh), val = (vv >> 16) << sh;
#pragma unroll
      for (int k = 0; k < 4; ++k)
#pragma unroll
        for (int q = 0; q < 4; ++q) A[k][q] = (f == k * 4 + q) ? ((A[k][q] & keep) | val) : A[k][q];
    }
  }
}

template <int PF>
__global__ void __launch_bounds__(256) mvh_kernel(A7 a) {
  extern __shared__ __align__(16) float smem[];
  const int NG = a.N / 64;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int wm = warp % a.WM, wk = warp / a.WM;
  const int tile = blockIdx.x * a.WM + wm;
  const int GY = gridDim.y;
  const int cg0 = NG * blockIdx.y / GY, cg1 = NG * (blockIdx.y + 1) / GY;
  const int G0 = cg0 + (cg1 - cg0) * wk / a.WK, G1 = cg0 + (cg1 - cg0) * (wk + 1) / a.WK;
  const int n = G1 - G0;                                                 // this warp's records
  const int R = a.WM * 16, ELN = R * 8;
  uint16_t* xs = (uint16_t*)(smem + (a.WK + 1) * ELN);                   // [B rows][XROW]
  const int XROW = (NG + GY - 1) / GY * 64 + 8;
  const size_t r0 = (size_t)tile * NG + G0;
  const uint8_t* rp = a.T + r0 * 1552 + lane * 16;
  const uint16_t* ep = (const uint16_t*)(a.T + r0 * 1552 + 1536) + g;
  const int* eo = a.EO + r0;
  uint4 c0[PF], c1[PF], c2[PF];
  uint32_t ee[PF];
  auto reload = [&](int i, int j) {
    const uint8_t* p = rp + (size_t)j * 1552;
    c0[i] = ldw(p);
    c1[i] = ldw(p + 512);
    c2[i] = ldw(p + 1024);
    ee[i] = __ldg(ep + (size_t)j * 776);
  };
#pragma unroll
  for (int i = 0; i < PF; ++i)
    if (i < n) reload(i, i);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // launched as a programmatic dependent (sm_90+): the first records are in flight while the previous kernel drains;
  // X, Y, WS and CNT are touched only after its completion (no-op when there is no programmatic dependency)
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  {                                                                      // stage X's rows < B (overlaps the loads)
    const int ch = (cg1 - cg0) * 8;
    for (int i = threadIdx.x; i < a.B * ch; i += blockDim.x) {
      const int r = i / ch, q = i - r * ch;
      cp_async16(smem_u32(xs + r * XROW + q * 8), a.X + (size_t)r * a.N + cg0 * 64 + q * 8, 16);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
  }
  // ldmatrix.x4 row addresses as mvs (matrix j = lane / 8: k-step 2 sp + j / 2, half j % 2), batch row min(lane % 8, B - 1)
  const int xrow = min(lane & 7, a.B - 1);
  const uint32_t xa = smem_u32(xs) + (uint32_t)((xrow * XROW + ((lane >> 4) & 1) * 16 + ((lane >> 3) & 1) * 8 +
                                                 (G0 - cg0) * 64) * 2);
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  int eol = 0, eoh = 0;
  uint32_t em = 0u;
  auto window = [&](int j) {                                             // escape lists of records j .. j + 31
    const int jj = j + lane;
    eol = (jj < n) ? __ldg(eo + jj) : 0;
    eoh = (jj < n) ? __ldg(eo + jj + 1) : 0;
    em = __ballot_sync(0xFFFFFFFFu, eoh > eol);
  };
  auto step = [&](int i, int j) {                                        // record j, held in slot i
    const uint32_t E0 = (ee[i] & 0xFFu) * 0x01010101u, E1 = (ee[i] >> 8) * 0x01010101u;
    uint32_t A[4][4];
#pragma unroll
    for (int s = 0; s < 4; ++s) dec_step5(c0[i], c1[i], c2[i], E0, E1, s, A[s]);
    if ((em >> (j & 31)) & 1u)
      patch16(A, a.EL, __shfl_sync(0xFFFFFFFFu, eol, j & 31), __shfl_sync(0xFFFFFFFFu, eoh, j & 31), lane);
    uint32_t xb[2][4];
    ldsm_x4s(xa + (uint32_t)(j * 128), xb[0]);
    ldsm_x4s(xa + (uint32_t)(j * 128 + 64), xb[1]);
#pragma unroll
    for (int s = 0; s < 4; ++s) mma16816(c, A[s], xb[s >> 1][2 * (s & 1)], xb[s >> 1][2 * (s & 1) + 1]);
  };
  int j0 = 0;
  for (; j0 + 2 * PF <= n; j0 += PF) {                                   // steady state: no bounds tests
    if ((j0 & 31) == 0) window(j0);
#pragma unroll
    for (int i = 0; i < PF; ++i) {
      step(i, j0 + i);
      reload(i, j0 + i + PF);
    }
  }
  for (; j0 < n; j0 += PF) {                                             // the last 1 .. 2 PF - 1 records
    if ((j0 & 31) == 0) window(j0);
#pragma unroll
    for (int i = 0; i < PF; ++i)
      if (j0 + i < n) {
        step(i, j0 + i);
        if (j0 + i + PF < n) reload(i, j0 + i + PF);
      }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");   // the next kernel may start its prologue
#endif
  float* red = smem;
  float* out = smem + a.WK * ELN;
  {
    const int rr = wm * 16 + g, nn = 2 * t;
    red[wk * ELN + nn * R + rr] = c[0];
    red[wk * ELN + (nn + 1) * R + rr] = c[1];
    red[wk * ELN + nn * R + rr + 8] = c[2];
    red[wk * ELN + (nn + 1) * R + rr + 8] = c[3];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < ELN; idx += blockDim.x) {
    float s = 0.f;
    for (int k = 0; k < a.WK; ++k) s += red[k * ELN + idx];
    out[idx] = s;
  }
  __syncthreads();
  if (GY > 1 && !merge7(a, out, ELN, blockIdx.x, blockIdx.y, GY)) return;
  for (int idx = threadIdx.x; idx < ELN; idx += blockDim.x) {
    const int nn = idx / R, r = idx % R;
    if (nn < a.B) a.Y[(size_t)nn * a.M + (size_t)blockIdx.x * R + r] = __float2bfloat16_rn(out[idx]);
  }
}

static size_t mvh_smem(int B, int WM, int WK, int GY, int N) {          // reduction buffers + rows < B of X
  return (size_t)(WK + 1) * WM * 16 * 8 * sizeof(float) + (size_t)B * mvs_xrow(N, GY) * 2;
}

// programmatic dependent launch (sm_90+; STATETREE_PDL=0 disables): the kernel may start while the previous kernel in
// the stream drains, and griddepcontrol.wait orders its reads of X after that kernel's writes
static bool pdl_on() {
  static int dev_pdl[64] = {0};                                          // 0 unknown, 1 off, 2 on
  int dev = 0;
  cudaGetDevice(&dev);
  int& v = dev_pdl[dev & 63];
  if (v == 0) {
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    const char* e = getenv("STATETREE_PDL");
    v = (major >= 9 && !(e && e[0] == '0')) ? 2 : 1;
  }
  return v == 2;
}

template <int PF>
void launch_mvh(const A7& a, int GY, cudaStream_t st) {
  const size_t sm = mvh_smem(a.B, a.WM, a.WK, GY, a.N);
  if (sm > 48 * 1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(mvh_kernel<PF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(a.M / (16 * a.WM), GY);
  cfg.blockDim = dim3(32 * a.WM * a.WK);
  cfg.dynamicSmemBytes = sm;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl_on() ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, mvh_kernel<PF>, a));
}

void mvh(torch::Tensor T, torch::Tensor EO, torch::Tensor EL, torch::Tensor X, torch::Tensor Y, torch::Tensor WS,
         torch::Tensor CNT, int64_t WM, int64_t WK, int64_t GY, int64_t PF) {
  const c10::cuda::OptionalCUDAGuard guard(T.device());
  A7 a = make_a7(T, EO, EL, X, Y, WS, CNT);
  a.WM = WM; a.WK = WK;
  TORCH_CHECK(a.B >= 1 && a.B <= 8 && a.M % (16 * a.WM) == 0 && a.N % 64 == 0 && a.WM * a.WK <= 8 && GY >= 1 &&
              GY <= a.N / 64 && X.is_contiguous() && ((uintptr_t)X.data_ptr() & 15) == 0, "mvh config");
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (PF == 4) launch_mvh<4>(a, (int)GY, st);
  else launch_mvh<2>(a, (int)GY, st);
}

int64_t mvh_smem_bytes(int64_t B, int64_t N, int64_t WM, int64_t WK, int64_t GY) {
  return (int64_t)mvh_smem((int)B, (int)WM, (int)WK, (int)GY, (int)N);
}

// ---------------------------------------------------------------------------------------------- mvc (Hopper, B <= 8)
// The matrix-vector product as a sweep over the records in memory order. In mv7 / mvh every warp streams its own run of
// one tile's records, so ~2000 concurrent streams sit 25-100 KB apart: on the H100 that read pattern alone costs ~15%
// against a linear read in short bursts and ~35% under a power cap (more DRAM energy per byte: the GPU then lowers the
// SM clock further). Here a CTA owns TC consecutive tiles (or 1 / GY of one tile's columns) and its WK warps take the
// records round-robin: at every step the CTA reads WK consecutive records, and all CTAs are resident at once. A warp
// keeps one accumulator per tile (parked in shared memory when its tile ends); the CTA sums the warps' partials in a fixed
// order (and merges GY column splits as mv7): deterministic, a different summation order than mv7.
// Escapes: a warp reads the escape offsets of its next 32 records one window ahead; when a record is loaded, its first
// 32 entries are fetched with it (one per lane), so patching never waits on a dependent load.
__device__ __forceinline__ void patch_pre(uint32_t (&A)[4][4], uint32_t ev, int cnt, int lo, const uint32_t* EL,
                                          int lane) {
  uint32_t mask = cnt >= 32 ? 0xFFFFFFFFu : ((1u << cnt) - 1u);
#pragma unroll 1
  for (int base = 0;;) {
#pragma unroll 1
    while (mask) {
      const int b = __ffs(mask) - 1;
      mask &= mask - 1;
      const uint32_t vv = __shfl_sync(0xFFFFFFFFu, ev, b);
      const int f = ((int)((vv >> 5) & 31u) == lane) ? (int)((vv >> 1) & 15u) : -1;      // k-step * 4 + register
      const uint32_t sh = (vv & 1u) * 16u, keep = ~(0xFFFFu << sh), val = (vv >> 16) << sh;
#pragma unroll
      for (int k = 0; k < 4; ++k)
#pragma unroll
        for (int q = 0; q < 4; ++q) A[k][q] = (f == k * 4 + q) ? ((A[k][q] & keep) | val) : A[k][q];
    }
    base += 32;
    if (base >= cnt) break;
    ev = (base + lane < cnt) ? __ldg(EL + lo + base + lane) : 0u;     // more than 32 entries: the next 32
    mask = (cnt - base >= 32) ? 0xFFFFFFFFu : ((1u << (cnt - base)) - 1u);
  }
}

template <int PF>
__global__ void __launch_bounds__(256, PF == 2 ? 3 : 2) mvc_kernel(A7 a, int TC) {   // 3 resident CTAs at PF = 2
  extern __shared__ __align__(16) float smem[];
  const int NG = a.N / 64;
  const int lane = threadIdx.x & 31, wk = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int WK = a.WK;
  const int GY = gridDim.y;
  const int cg0 = NG * blockIdx.y / GY, cg1 = NG * (blockIdx.y + 1) / GY;
  const int per = (cg1 - cg0) / WK;                                      // records per warp and tile
  const int n = TC * per;                                                // records of this warp
  const int tile0 = blockIdx.x * TC;
  float* red = smem;                                                     // [TC][WK][128]: per-warp partial tiles
  float* outb = red + TC * WK * 128;                                     // [TC][128]
  uint16_t* xs = (uint16_t*)(outb + TC * 128);                           // [B][XROW]
  const int XROW = (cg1 - cg0) * 64 + 8;
  const size_t r0 = (size_t)tile0 * NG + cg0 + wk;                       // record q of this warp: r0 + q WK
  const size_t RS = (size_t)WK * 1552;
  const uint8_t* rp = a.T + r0 * 1552 + lane * 16;
  const uint16_t* ep = (const uint16_t*)(a.T + r0 * 1552 + 1536) + g;
  const int* eo = a.EO + r0;
  uint4 c0[PF], c1[PF], c2[PF];
  uint32_t ee[PF], ev[PF];
  int ecnt[PF], elo[PF];
  int wl = 0, wh = 0, nl = 0, nh = 0;
  uint32_t wmask = 0u;
  auto wload = [&](int w, int& lo, int& hi) {                            // escape offsets of window w (records 32 w ..)
    const int q = w * 32 + lane;
    lo = (q < n) ? __ldg(eo + (size_t)q * WK) : 0;
    hi = (q < n) ? __ldg(eo + (size_t)q * WK + 1) : 0;
  };
  const uint8_t* np = rp;                                                // the next record to load (record q)
  const uint16_t* nep = ep;
  auto reload = [&](int i, int q) {                                      // record q into slot i (q = 0, 1, 2, ..)
    if ((q & 31) == 0) {                                                 // window q / 32 (fetched a window ago)
      wl = nl;
      wh = nh;
      wmask = __ballot_sync(0xFFFFFFFFu, wh > wl);
      wload((q >> 5) + 1, nl, nh);
    }
    c0[i] = ldw(np);
    c1[i] = ldw(np + 512);
    c2[i] = ldw(np + 1024);
    ee[i] = __ldg(nep);
    np += RS;
    nep += RS / 2;
    ecnt[i] = 0;
    if ((wmask >> (q & 31)) & 1u) {
      const int lo = __shfl_sync(0xFFFFFFFFu, wl, q & 31), hi = __shfl_sync(0xFFFFFFFFu, wh, q & 31);
      ecnt[i] = hi - lo;
      elo[i] = lo;
      ev[i] = (lane < hi - lo) ? __ldg(a.EL + lo + lane) : 0u;
    }
  };
  wload(0, nl, nh);
#pragma unroll
  for (int i = 0; i < PF; ++i)
    if (i < n) reload(i, i);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");                  // X / WS / CNT after the previous kernel
#endif
  {
    const int ch = (cg1 - cg0) * 8;
    for (int i = threadIdx.x; i < a.B * ch; i += blockDim.x) {
      const int r = i / ch, q = i - r * ch;
      cp_async16(smem_u32(xs + r * XROW + q * 8), a.X + (size_t)r * a.N + cg0 * 64 + q * 8, 16);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
  }
  const int xrow = min(lane & 7, a.B - 1);
  const uint32_t xa0 = smem_u32(xs) + (uint32_t)((xrow * XROW + ((lane >> 4) & 1) * 16 + ((lane >> 3) & 1) * 8 +
                                                  wk * 64) * 2);
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  int qq = 0, tc = 0;
  uint32_t xq = xa0;
  auto step = [&](int i) {
    const uint32_t E0 = (ee[i] & 0xFFu) * 0x01010101u, E1 = (ee[i] >> 8) * 0x01010101u;
    uint32_t A[4][4];
#pragma unroll
    for (int s = 0; s < 4; ++s) dec_step5(c0[i], c1[i], c2[i], E0, E1, s, A[s]);
    if (ecnt[i]) patch_pre(A, ev[i], ecnt[i], elo[i], a.EL, lane);
    uint32_t xb[2][4];
    ldsm_x4s(xq, xb[0]);
    ldsm_x4s(xq + 64, xb[1]);
#pragma unroll
    for (int s = 0; s < 4; ++s) mma16816(c, A[s], xb[s >> 1][2 * (s & 1)], xb[s >> 1][2 * (s & 1) + 1]);
    xq += (uint32_t)(WK * 128);
    if (++qq == per) {                                                   // this warp's share of the tile is done
      float* rw = red + (tc * WK + wk) * 128;
      rw[(2 * t) * 16 + g] = c[0];
      rw[(2 * t + 1) * 16 + g] = c[1];
      rw[(2 * t) * 16 + g + 8] = c[2];
      rw[(2 * t + 1) * 16 + g + 8] = c[3];
      c[0] = c[1] = c[2] = c[3] = 0.f;
      qq = 0;
      ++tc;
      xq = xa0;
    }
  };
  int j0 = 0;
  for (; j0 + 2 * PF <= n; j0 += PF) {
#pragma unroll
    for (int i = 0; i < PF; ++i) {
      step(i);
      reload(i, j0 + i + PF);
    }
  }
  for (; j0 < n; j0 += PF) {
#pragma unroll
    for (int i = 0; i < PF; ++i)
      if (j0 + i < n) {
        step(i);
        if (j0 + i + PF < n) reload(i, j0 + i + PF);
      }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
  __syncthreads();
  for (int idx = threadIdx.x; idx < TC * 128; idx += blockDim.x) {
    const int tt = idx >> 7, e = idx & 127;
    float s = 0.f;
    for (int k = 0; k < WK; ++k) s += red[(tt * WK + k) * 128 + e];
    outb[idx] = s;
  }
  __syncthreads();
  if (GY > 1 && !merge7(a, outb, 128, blockIdx.x, blockIdx.y, GY)) return;   // GY > 1: TC = 1, tile = blockIdx.x
  for (int idx = threadIdx.x; idx < TC * 128; idx += blockDim.x) {
    const int tt = idx >> 7, e = idx & 127, nn = e >> 4, r = e & 15;
    if (nn < a.B) a.Y[(size_t)nn * a.M + (size_t)(tile0 + tt) * 16 + r] = __float2bfloat16_rn(outb[idx]);
  }
}

static size_t mvc_smem(int B, int WK, int GY, int TC, int N) {
  return (size_t)TC * (WK + 1) * 128 * sizeof(float) + (size_t)B * ((N / 64) / GY * 64 + 8) * 2;
}

template <int PF>
void launch_mvc(const A7& a, int GY, int TC, cudaStream_t st) {
  const size_t sm = mvc_smem(a.B, a.WK, GY, TC, a.N);
  if (sm > 48 * 1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(mvc_kernel<PF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(a.M / 16 / TC, GY);
  cfg.blockDim = dim3(32 * a.WK);
  cfg.dynamicSmemBytes = sm;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl_on() ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, mvc_kernel<PF>, a, TC));
}

void mvc(torch::Tensor T, torch::Tensor EO, torch::Tensor EL, torch::Tensor X, torch::Tensor Y, torch::Tensor WS,
         torch::Tensor CNT, int64_t WK, int64_t GY, int64_t TC, int64_t PF) {
  const c10::cuda::OptionalCUDAGuard guard(T.device());
  A7 a = make_a7(T, EO, EL, X, Y, WS, CNT);
  a.WM = 1; a.WK = WK;
  const int NG = a.N / 64, tiles = a.M / 16;
  TORCH_CHECK(a.B >= 1 && a.B <= 8 && a.M % 16 == 0 && a.N % 64 == 0 && WK >= 1 && WK <= 8 && GY >= 1 && TC >= 1 &&
              NG % (GY * WK) == 0 && tiles % TC == 0 && (TC == 1 || GY == 1) && X.is_contiguous() &&
              ((uintptr_t)X.data_ptr() & 15) == 0, "mvc config");
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (PF == 4) launch_mvc<4>(a, (int)GY, (int)TC, st);
  else launch_mvc<2>(a, (int)GY, (int)TC, st);
}

int64_t mvc_smem_bytes(int64_t B, int64_t N, int64_t WK, int64_t GY, int64_t TC) {
  return (int64_t)mvc_smem((int)B, (int)WK, (int)GY, (int)TC, (int)N);
}

// ---------------------------------------------------------------------------------------------- mvx (MoE)
// Mixture-of-experts matrix-vector products without a BF16 copy of the experts: slot s multiplies row s / xdiv of X by
// the M x N matrix of expert ids[s] (its tree, escape offsets and escape list at TP / EOP / ELP[ids[s]]: separate
// per-expert trees, or pointers into one stacked [E * M, N] tree) and writes row s of Y. Grid (M / 16WM, GY, S); one
// input row per slot, so only column 0 of the m16n8k16 B fragment carries data (the kernel is bandwidth bound).
template <int PF, typename IT>
__global__ void __launch_bounds__(256) mvx_kernel(A7 a, const int64_t* TP, const int64_t* EOP, const int64_t* ELP,
                                                  const IT* ids, int xdiv, int nexp) {
  extern __shared__ float smem[];
  uint32_t (*s_em)[32] = (uint32_t (*)[32])(smem + (a.WK + 1) * a.WM * 16 * 8);      // per-warp escape images (esc_scatter) after the
  uint16_t (*s_ev)[1024] = (uint16_t (*)[1024])(s_em + a.WM * a.WK);  // reduction buffers: ESC_IMG_BYTES per warp
  const int NG = a.N / 64;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int wm = warp % a.WM, wk = warp / a.WM;
  const int slot = blockIdx.z;
  long long e = (long long)ids[slot];
  e = e < 0 ? 0 : (e >= nexp ? nexp - 1 : e);                 // vLLM's dummy batches need not route to valid experts
  A7 b = a;
  b.T = (const uint8_t*)TP[e];
  b.EO = (const int*)EOP[e];
  b.EL = (const uint32_t*)ELP[e];
  b.X = a.X + (size_t)(slot / xdiv) * a.N;
  b.B = 1;
  const int tile = blockIdx.x * a.WM + wm;
  const int GY = gridDim.y;
  const int cg0 = NG * blockIdx.y / GY, cg1 = NG * (blockIdx.y + 1) / GY;
  const int G0 = cg0 + (cg1 - cg0) * wk / a.WK, G1 = cg0 + (cg1 - cg0) * (wk + 1) / a.WK;
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  F7<1> f[PF];
#pragma unroll
  for (int i = 0; i < PF; ++i)
    if (G0 + i < G1) load_f7<1>(f[i], b, tile, G0 + i, lane, NG);
  for (int G = G0; G < G1; G += PF) {
#pragma unroll
    for (int i = 0; i < PF; ++i) {
      if (G + i < G1) {
        const uint32_t E0 = (f[i].e & 0xFFu) * 0x01010101u, E1 = ((f[i].e >> 8) & 0xFFu) * 0x01010101u;
        const uint32_t pm = (f[i].lo < f[i].hi) ? esc_scatter(b.EL, f[i].lo, f[i].hi, lane, s_em[warp], s_ev[warp]) : 0u;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          uint32_t A[4];
          dec_step5(f[i].c0, f[i].c1, f[i].c2, E0, E1, s, A);
          if (pm) esc_apply(pm, s_ev[warp] + lane * 32, s, A);
          mma16816(c, A, f[i].x[0][2 * s], f[i].x[0][2 * s + 1]);
        }
        if (G + i + PF < G1) load_f7<1>(f[i], b, tile, G + i + PF, lane, NG);
      }
    }
  }
  const int R = a.WM * 16, EL = R * 8;
  float* red = smem;
  float* out = smem + a.WK * EL;
  {
    const int rr = wm * 16 + g, n = 2 * t;
    red[wk * EL + n * R + rr] = c[0];
    red[wk * EL + (n + 1) * R + rr] = c[1];
    red[wk * EL + n * R + rr + 8] = c[2];
    red[wk * EL + (n + 1) * R + rr + 8] = c[3];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < R; idx += blockDim.x) {      // column 0: this slot's input row
    float s = 0.f;
    for (int k = 0; k < a.WK; ++k) s += red[k * EL + idx];
    out[idx] = s;
  }
  __syncthreads();
  if (GY > 1 && !merge7(a, out, R, slot * gridDim.x + blockIdx.x, blockIdx.y, GY)) return;
  for (int idx = threadIdx.x; idx < R; idx += blockDim.x)
    a.Y[(size_t)slot * a.M + (size_t)blockIdx.x * R + idx] = __float2bfloat16_rn(out[idx]);
}

template <int PF, typename IT>
static void launch_mvx(const A7& a, const int64_t* TP, const int64_t* EOP, const int64_t* ELP, const IT* ids,
                       int S, int xdiv, int GY, int nexp, cudaStream_t st) {
  dim3 grid(a.M / (16 * a.WM), GY, S);
  const size_t sm = (size_t)(a.WK + 1) * a.WM * 16 * 8 * sizeof(float) + (size_t)a.WM * a.WK * ESC_IMG_BYTES;
  mvx_kernel<PF, IT><<<grid, 32 * a.WM * a.WK, sm, st>>>(a, TP, EOP, ELP, ids, xdiv, nexp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mvx(torch::Tensor TP, torch::Tensor EOP, torch::Tensor ELP, torch::Tensor X, torch::Tensor ids, torch::Tensor Y,
         torch::Tensor WS, torch::Tensor CNT, int64_t xdiv, int64_t WM, int64_t WK, int64_t GY, int64_t PF) {
  const c10::cuda::OptionalCUDAGuard guard(X.device());
  auto i64 = [](const torch::Tensor& v) { return v.is_cuda() && v.scalar_type() == at::kLong && v.is_contiguous(); };
  const bool ids32 = ids.is_cuda() && ids.scalar_type() == at::kInt && ids.is_contiguous();
  TORCH_CHECK(i64(TP) && i64(EOP) && i64(ELP) && (i64(ids) || ids32) && TP.numel() == EOP.numel() &&
              TP.numel() == ELP.numel(),
              "mvx: pointer tables (int64) of equal sizes and ids (int32 or int64) must be contiguous CUDA tensors");
  TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kBFloat16 && X.is_contiguous() && X.dim() == 2, "mvx: X");
  TORCH_CHECK(Y.is_cuda() && Y.dtype() == torch::kBFloat16 && Y.is_contiguous() && Y.dim() == 2, "mvx: Y");
  const int64_t S = ids.numel();
  TORCH_CHECK(Y.size(0) == S && xdiv >= 1 && X.size(0) * xdiv >= S && S <= 65535, "mvx: rows");
  A7 a;
  a.T = nullptr; a.EO = nullptr; a.EL = nullptr;
  a.X = (const uint16_t*)X.data_ptr<at::BFloat16>(); a.Y = (__nv_bfloat16*)Y.data_ptr<at::BFloat16>();
  a.WS = WS.data_ptr<float>(); a.CNT = CNT.data_ptr<int>();
  a.M = (int)Y.size(1); a.N = (int)X.size(1); a.B = 1; a.WM = (int)WM; a.WK = (int)WK;
  TORCH_CHECK(a.WM >= 1 && a.WK >= 1 && a.WM * a.WK <= 8 && a.M % (16 * a.WM) == 0 && a.N % 64 == 0 && GY >= 1 &&
              GY <= a.N / 64, "mvx config");
  const int64_t tiles = a.M / (16 * a.WM);
  TORCH_CHECK(GY == 1 || (WS.numel() >= S * tiles * GY * a.WM * 16 && CNT.numel() >= S * tiles), "mvx: split-K workspace");
  if (S == 0) return;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const int64_t* tp = TP.data_ptr<int64_t>(); const int64_t* eop = EOP.data_ptr<int64_t>();
  const int64_t* elp = ELP.data_ptr<int64_t>();
  const int nexp = (int)TP.numel();
  if (ids32) {
    const int* idp = ids.data_ptr<int>();
    if (PF == 4) launch_mvx<4, int>(a, tp, eop, elp, idp, (int)S, (int)xdiv, (int)GY, nexp, st);
    else launch_mvx<2, int>(a, tp, eop, elp, idp, (int)S, (int)xdiv, (int)GY, nexp, st);
  } else {
    const int64_t* idp = ids.data_ptr<int64_t>();
    if (PF == 4) launch_mvx<4, int64_t>(a, tp, eop, elp, idp, (int)S, (int)xdiv, (int)GY, nexp, st);
    else launch_mvx<2, int64_t>(a, tp, eop, elp, idp, (int)S, (int)xdiv, (int)GY, nexp, st);
  }
}

// ---------------------------------------------------------------------------------------------- mvg (grouped MoE)
// Grouped MoE products: the (token, expert) slots sorted by expert into blocks of BS = 8 * NT (vLLM's
// moe_align_block_size layout: sorted[blk * BS + j] = slot id, >= S for padding; bexp[blk] = the block's expert, < 0
// for another rank's expert; *npost = used entries of sorted). Block blk multiplies the X rows of its slots (row
// slot / xdiv) by its expert's M x N matrix, so the expert's tree is read once for up to BS tokens, and writes rows
// 'slot' of Y. Grid (M / 16WM, GY, blocks); blocks past *npost exit at once.
template <int NT>
__device__ __forceinline__ void load_fg(F7<NT>& f, const uint8_t* T, const int* EO, const uint16_t* const (&xr)[NT],
                                        int tile, int G, int lane, int NG) {
  const int t = lane & 3;
  const size_t tg = (size_t)tile * NG + G;
  const uint8_t* base = T + tg * 1552;
  f.c0 = ldw(base + lane * 16);
  f.c1 = ldw(base + 512 + lane * 16);
  f.c2 = ldw(base + 1024 + lane * 16);
  f.e = __ldg((const uint16_t*)(base + 1536) + (lane >> 2));
  f.lo = __ldg(EO + tg);
  f.hi = __ldg(EO + tg + 1);
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    if (xr[nt] != nullptr) {
      const uint32_t* xp = (const uint32_t*)(xr[nt] + G * 64 + 2 * t);
#pragma unroll
      for (int s = 0; s < 4; ++s) { f.x[nt][2 * s] = __ldg(xp + 8 * s); f.x[nt][2 * s + 1] = __ldg(xp + 8 * s + 4); }
    } else {
#pragma unroll
      for (int q = 0; q < 8; ++q) f.x[nt][q] = 0u;
    }
  }
}

template <int NT, int PF>
__global__ void __launch_bounds__(256) mvg_kernel(A7 a, const int64_t* TP, const int64_t* EOP, const int64_t* ELP,
                                                  const int* sorted, const int* bexp, const int* npost, int S,
                                                  int xdiv, int nexp) {
  extern __shared__ float smem[];
  uint32_t (*s_em)[32] = (uint32_t (*)[32])(smem + (a.WK + 1) * a.WM * 16 * NT * 8);      // per-warp escape images (esc_scatter) after the
  uint16_t (*s_ev)[1024] = (uint16_t (*)[1024])(s_em + a.WM * a.WK);  // reduction buffers: ESC_IMG_BYTES per warp
  constexpr int BS = NT * 8;
  __shared__ int s_slot[BS];
  const int blk = blockIdx.z;
  if (blk * BS >= __ldg(npost)) return;
  const int e = __ldg(bexp + blk);
  if (e < 0 || e >= nexp) return;                               // another rank's expert / an invalid id
  if (threadIdx.x < BS) s_slot[threadIdx.x] = __ldg(sorted + (size_t)blk * BS + threadIdx.x);
  __syncthreads();
  const int NG = a.N / 64;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int wm = warp % a.WM, wk = warp / a.WM;
  const uint8_t* T = (const uint8_t*)TP[e];
  const int* EO = (const int*)EOP[e];
  const uint32_t* EL = (const uint32_t*)ELP[e];
  const uint16_t* xr[NT];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int sid = s_slot[nt * 8 + g];
    xr[nt] = (sid >= 0 && sid < S) ? a.X + (size_t)(sid / xdiv) * a.N : nullptr;
  }
  const int tile = blockIdx.x * a.WM + wm;
  const int GY = gridDim.y;
  const int cg0 = NG * blockIdx.y / GY, cg1 = NG * (blockIdx.y + 1) / GY;
  const int G0 = cg0 + (cg1 - cg0) * wk / a.WK, G1 = cg0 + (cg1 - cg0) * (wk + 1) / a.WK;
  float c[NT][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) c[nt][0] = c[nt][1] = c[nt][2] = c[nt][3] = 0.f;
  F7<NT> f[PF];
#pragma unroll
  for (int i = 0; i < PF; ++i)
    if (G0 + i < G1) load_fg<NT>(f[i], T, EO, xr, tile, G0 + i, lane, NG);
  for (int G = G0; G < G1; G += PF) {
#pragma unroll
    for (int i = 0; i < PF; ++i) {
      if (G + i < G1) {
        const uint32_t E0 = (f[i].e & 0xFFu) * 0x01010101u, E1 = ((f[i].e >> 8) & 0xFFu) * 0x01010101u;
        const uint32_t pm = (f[i].lo < f[i].hi) ? esc_scatter(EL, f[i].lo, f[i].hi, lane, s_em[warp], s_ev[warp]) : 0u;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          uint32_t A[4];
          dec_step5(f[i].c0, f[i].c1, f[i].c2, E0, E1, s, A);
          if (pm) esc_apply(pm, s_ev[warp] + lane * 32, s, A);
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) mma16816(c[nt], A, f[i].x[nt][2 * s], f[i].x[nt][2 * s + 1]);
        }
        if (G + i + PF < G1) load_fg<NT>(f[i], T, EO, xr, tile, G + i + PF, lane, NG);
      }
    }
  }
  const int R = a.WM * 16, ELN = R * BS;
  float* red = smem;
  float* out = smem + a.WK * ELN;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int rr = wm * 16 + g, n = nt * 8 + 2 * t;
    red[wk * ELN + n * R + rr] = c[nt][0];
    red[wk * ELN + (n + 1) * R + rr] = c[nt][1];
    red[wk * ELN + n * R + rr + 8] = c[nt][2];
    red[wk * ELN + (n + 1) * R + rr + 8] = c[nt][3];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < ELN; idx += blockDim.x) {
    float s = 0.f;
    for (int k = 0; k < a.WK; ++k) s += red[k * ELN + idx];
    out[idx] = s;
  }
  __syncthreads();
  if (GY > 1 && !merge7(a, out, ELN, blk * gridDim.x + blockIdx.x, blockIdx.y, GY)) return;
  for (int idx = threadIdx.x; idx < ELN; idx += blockDim.x) {
    const int n = idx / R, r = idx % R;
    const int sid = s_slot[n];
    if (sid >= 0 && sid < S) a.Y[(size_t)sid * a.M + (size_t)blockIdx.x * R + r] = __float2bfloat16_rn(out[idx]);
  }
}

template <int NT>
static void launch_mvg(const A7& a, const int64_t* TP, const int64_t* EOP, const int64_t* ELP, const int* sorted,
                       const int* bexp, const int* npost, int S, int xdiv, int nblk, int GY, int nexp,
                       cudaStream_t st) {
  dim3 grid(a.M / (16 * a.WM), GY, nblk);
  const size_t sm = (size_t)(a.WK + 1) * a.WM * 16 * NT * 8 * sizeof(float) + (size_t)a.WM * a.WK * ESC_IMG_BYTES;
  mvg_kernel<NT, 2><<<grid, 32 * a.WM * a.WK, sm, st>>>(a, TP, EOP, ELP, sorted, bexp, npost, S, xdiv, nexp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mvg(torch::Tensor TP, torch::Tensor EOP, torch::Tensor ELP, torch::Tensor X, torch::Tensor sorted,
         torch::Tensor bexp, torch::Tensor npost, torch::Tensor Y, torch::Tensor WS, torch::Tensor CNT, int64_t xdiv,
         int64_t NT, int64_t WM, int64_t WK, int64_t GY) {
  const c10::cuda::OptionalCUDAGuard guard(X.device());
  auto i64 = [](const torch::Tensor& v) { return v.is_cuda() && v.scalar_type() == at::kLong && v.is_contiguous(); };
  auto i32 = [](const torch::Tensor& v) { return v.is_cuda() && v.scalar_type() == at::kInt && v.is_contiguous(); };
  TORCH_CHECK(i64(TP) && i64(EOP) && i64(ELP) && TP.numel() == EOP.numel() && TP.numel() == ELP.numel(),
              "mvg: pointer tables must be contiguous int64 CUDA tensors of one size");
  TORCH_CHECK(i32(sorted) && i32(bexp) && i32(npost) && npost.numel() >= 1, "mvg: sorted / bexp / npost int32");
  TORCH_CHECK(NT == 1 || NT == 2 || NT == 4, "mvg: NT 1, 2 or 4");
  TORCH_CHECK(sorted.numel() >= (bexp.numel() - 1) * 8 * NT + 1 || bexp.numel() == 0, "mvg: sorted too short");
  TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kBFloat16 && X.is_contiguous() && X.dim() == 2, "mvg: X");
  TORCH_CHECK(Y.is_cuda() && Y.dtype() == torch::kBFloat16 && Y.is_contiguous() && Y.dim() == 2, "mvg: Y");
  const int64_t S = Y.size(0), nblk = bexp.numel();
  TORCH_CHECK(xdiv >= 1 && X.size(0) * xdiv >= S && nblk <= 65535, "mvg: rows / blocks");
  A7 a;
  a.T = nullptr; a.EO = nullptr; a.EL = nullptr;
  a.X = (const uint16_t*)X.data_ptr<at::BFloat16>(); a.Y = (__nv_bfloat16*)Y.data_ptr<at::BFloat16>();
  a.WS = WS.data_ptr<float>(); a.CNT = CNT.data_ptr<int>();
  a.M = (int)Y.size(1); a.N = (int)X.size(1); a.B = 0; a.WM = (int)WM; a.WK = (int)WK;
  TORCH_CHECK(a.WM >= 1 && a.WK >= 1 && a.WM * a.WK <= 8 && a.M % (16 * a.WM) == 0 && a.N % 64 == 0 && GY >= 1 &&
              GY <= a.N / 64, "mvg config");
  const int64_t tiles = a.M / (16 * a.WM);
  TORCH_CHECK(GY == 1 || (WS.numel() >= nblk * tiles * GY * a.WM * 16 * 8 * NT && CNT.numel() >= nblk * tiles),
              "mvg: split-K workspace");
  if (S == 0 || nblk == 0) return;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const int64_t* tp = TP.data_ptr<int64_t>(); const int64_t* eop = EOP.data_ptr<int64_t>();
  const int64_t* elp = ELP.data_ptr<int64_t>();
  const int* so = sorted.data_ptr<int>(); const int* be = bexp.data_ptr<int>(); const int* np = npost.data_ptr<int>();
  const int ne = (int)TP.numel();
  if (NT == 1) launch_mvg<1>(a, tp, eop, elp, so, be, np, (int)S, (int)xdiv, (int)nblk, (int)GY, ne, st);
  else if (NT == 2) launch_mvg<2>(a, tp, eop, elp, so, be, np, (int)S, (int)xdiv, (int)nblk, (int)GY, ne, st);
  else launch_mvg<4>(a, tp, eop, elp, so, be, np, (int)S, (int)xdiv, (int)nblk, (int)GY, ne, st);
}

// ---------------------------------------------------------------------------------------------- mvw (MoE, small K)
// Persistent warp-per-unit expert products for matrices with few 64-column groups (NG = N / 64 even, <= 16: the down
// projection, N = I). A unit is (chunk, 16-row tile). Each warp computes whole units (all NG groups accumulate in its
// mma registers: no cross-warp reduction, no block barrier) and streams its units (u, u + NW, ...) through a ring of
// two records, so the next unit's records load while the current one is multiplied; the next unit's expert pointers
// and the unit after that's chunk metadata are fetched one iteration ahead. Grid: a few blocks of 4 warps per SM
// (mvx / mvg launch one block per tile and chunk, most of them tiny or empty at decode sizes).
//   MODE 0 (mvx layout): chunk = slot s: expert ids[s] (clamped), input row s / xdiv, output row s (B column 0).
//   MODE 1 (mvg layout, moe_align_block_size with block size 8): chunk c = 8 sorted slots of expert bexp[c] (< 0 or
//   >= E: another rank's expert, not written), ceil(*npost / 8) chunks; padding slots (>= S) are zero columns.
template <typename IT>
struct MW {
  const int64_t* TP; const int64_t* EOP; const int64_t* ELP;
  const uint16_t* X; __nv_bfloat16* Y;
  const IT* ids; const int* sorted; const int* bexp; const int* npost;
  int S, M, N, xdiv, nexp, nsorted;
};
struct MWA { long long e; int sg, s0, s1; };                     // chunk metadata of a unit (sg < 0 in MODE 0: none)
struct MWB { const uint8_t* T; const int* EO; const uint32_t* EL; const uint16_t* xr; int o0, o1; };  // T null: none

template <int MODE, typename IT>
__device__ __forceinline__ MWA mw_fetch(const MW<IT>& p, int u, int U, int TILES, int g, int t) {
  MWA a; a.e = -1; a.sg = a.s0 = a.s1 = -1;
  if (u < U) {
    const int c = u / TILES;
    if (MODE == 0) {
      a.e = (long long)p.ids[c]; a.sg = c;
    } else {
      a.e = __ldg(p.bexp + c);
      const int b = c * 8;
      a.sg = (b + g < p.nsorted) ? __ldg(p.sorted + b + g) : -1;
      a.s0 = (b + 2 * t < p.nsorted) ? __ldg(p.sorted + b + 2 * t) : -1;
      a.s1 = (b + 2 * t + 1 < p.nsorted) ? __ldg(p.sorted + b + 2 * t + 1) : -1;
    }
  }
  return a;
}

template <int MODE, typename IT>
__device__ __forceinline__ MWB mw_resolve(const MW<IT>& p, const MWA& a, int g, int t) {
  MWB b; b.T = nullptr; b.EO = nullptr; b.EL = nullptr; b.xr = nullptr; b.o0 = b.o1 = -1;
  long long e = a.e;
  bool ok;
  if (MODE == 0) { ok = a.sg >= 0; e = e < 0 ? 0 : (e >= p.nexp ? p.nexp - 1 : e); }
  else ok = e >= 0 && e < p.nexp;
  if (ok) {
    b.T = (const uint8_t*)p.TP[e]; b.EO = (const int*)p.EOP[e]; b.EL = (const uint32_t*)p.ELP[e];
    if (MODE == 0) {
      b.xr = (g == 0) ? p.X + (size_t)(a.sg / p.xdiv) * p.N : nullptr;
      b.o0 = (t == 0) ? a.sg : -1;
    } else {
      b.xr = (a.sg >= 0 && a.sg < p.S) ? p.X + (size_t)(a.sg / p.xdiv) * p.N : nullptr;
      b.o0 = (a.s0 >= 0 && a.s0 < p.S) ? a.s0 : -1;
      b.o1 = (a.s1 >= 0 && a.s1 < p.S) ? a.s1 : -1;
    }
  }
  return b;
}

__device__ __forceinline__ void mw_load(F7<1>& f, const MWB& b, int tile, int G, int lane, int NG) {
  if (b.T != nullptr) {
    const uint16_t* xr[1] = {b.xr};
    load_fg<1>(f, b.T, b.EO, xr, tile, G, lane, NG);
  } else {
    f.c0 = f.c1 = f.c2 = make_uint4(0u, 0u, 0u, 0u);
    f.e = 0u; f.lo = f.hi = 0;
#pragma unroll
    for (int q = 0; q < 8; ++q) f.x[0][q] = 0u;
  }
}

template <int MODE, typename IT>
__global__ void __launch_bounds__(128, 4) mvw_kernel(MW<IT> p) {
  __shared__ uint32_t s_em[4][32];                            // escape images (esc_scatter), one per warp
  __shared__ uint16_t s_ev[4][1024];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int TILES = p.M / 16, NG = p.N / 64;
  const int U = (MODE == 0 ? p.S : (__ldg(p.npost) + 7) / 8) * TILES;
  const int NW = gridDim.x * 4;
  int u = blockIdx.x * 4 + warp;
  if (u >= U) return;
  MWA a1 = mw_fetch<MODE>(p, u, U, TILES, g, t);
  const MWA a2 = mw_fetch<MODE>(p, u + NW, U, TILES, g, t);
  MWB b0 = mw_resolve<MODE>(p, a1, g, t);
  a1 = a2;                                                       // chunk metadata of the next unit
  F7<1> f[2];
  mw_load(f[0], b0, u % TILES, 0, lane, NG);
  mw_load(f[1], b0, u % TILES, 1, lane, NG);
  for (;;) {
    const int un = u + NW;
    const MWB b1 = mw_resolve<MODE>(p, a1, g, t);               // the next unit's expert pointers
    a1 = mw_fetch<MODE>(p, un + NW, U, TILES, g, t);             // the unit after that's chunk metadata
    const int tile = u % TILES, tilen = un % TILES;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    for (int G = 0; G < NG; G += 2) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint32_t E0 = (f[i].e & 0xFFu) * 0x01010101u, E1 = ((f[i].e >> 8) & 0xFFu) * 0x01010101u;
        const uint32_t pm = (f[i].lo < f[i].hi) ? esc_scatter(b0.EL, f[i].lo, f[i].hi, lane, s_em[warp], s_ev[warp]) : 0u;
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t A[4];
          dec_step5(f[i].c0, f[i].c1, f[i].c2, E0, E1, st, A);
          if (pm) esc_apply(pm, s_ev[warp] + lane * 32, st, A);
          mma16816(c, A, f[i].x[0][2 * st], f[i].x[0][2 * st + 1]);
        }
        const int Gn = G + i + 2;
        if (Gn < NG) mw_load(f[i], b0, tile, Gn, lane, NG);
        else if (un < U) mw_load(f[i], b1, tilen, Gn - NG, lane, NG);
      }
    }
    if (b0.T != nullptr) {
      __nv_bfloat16* y = p.Y + (size_t)tile * 16 + g;
      if (b0.o0 >= 0) {
        y[(size_t)b0.o0 * p.M] = __float2bfloat16_rn(c[0]);
        y[(size_t)b0.o0 * p.M + 8] = __float2bfloat16_rn(c[2]);
      }
      if (MODE == 1 && b0.o1 >= 0) {
        y[(size_t)b0.o1 * p.M] = __float2bfloat16_rn(c[1]);
        y[(size_t)b0.o1 * p.M + 8] = __float2bfloat16_rn(c[3]);
      }
    }
    if (un >= U) break;
    u = un;
    b0 = b1;
  }
}

static int mvw_blocks(int64_t units, int64_t nbps, const torch::Tensor& like) {
  const int sms = at::cuda::getDeviceProperties(like.device().index())->multiProcessorCount;
  const int64_t need = (units + 3) / 4;
  return (int)std::max<int64_t>(1, std::min<int64_t>(need, std::max<int64_t>(1, nbps) * sms));
}

static void mvw_check(const torch::Tensor& TP, const torch::Tensor& EOP, const torch::Tensor& ELP,
                      const torch::Tensor& X, const torch::Tensor& Y, const char* who) {
  auto i64 = [](const torch::Tensor& v) { return v.is_cuda() && v.scalar_type() == at::kLong && v.is_contiguous(); };
  TORCH_CHECK(i64(TP) && i64(EOP) && i64(ELP) && TP.numel() == EOP.numel() && TP.numel() == ELP.numel(), who,
              ": pointer tables must be contiguous int64 CUDA tensors of one size");
  TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kBFloat16 && X.is_contiguous() && X.dim() == 2, who, ": X");
  TORCH_CHECK(Y.is_cuda() && Y.dtype() == torch::kBFloat16 && Y.is_contiguous() && Y.dim() == 2, who, ": Y");
  TORCH_CHECK(Y.size(1) % 16 == 0 && X.size(1) % 128 == 0 && X.size(1) <= 1024, who,
              ": M % 16 == 0 and N % 128 == 0, N <= 1024 (an even number of 64-column groups, at most 16)");
}

void mvwx(torch::Tensor TP, torch::Tensor EOP, torch::Tensor ELP, torch::Tensor X, torch::Tensor ids, torch::Tensor Y,
          int64_t xdiv, int64_t nbps) {
  const c10::cuda::OptionalCUDAGuard guard(X.device());
  mvw_check(TP, EOP, ELP, X, Y, "mvwx");
  const bool ids32 = ids.is_cuda() && ids.scalar_type() == at::kInt && ids.is_contiguous();
  const bool ids64 = ids.is_cuda() && ids.scalar_type() == at::kLong && ids.is_contiguous();
  TORCH_CHECK(ids32 || ids64, "mvwx: ids int32 / int64 contiguous");
  const int64_t S = ids.numel();
  TORCH_CHECK(Y.size(0) == S && xdiv >= 1 && X.size(0) * xdiv >= S, "mvwx: rows");
  if (S == 0) return;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const int nb = mvw_blocks(S * (Y.size(1) / 16), nbps, X);
  if (ids32) {
    MW<int> p{TP.data_ptr<int64_t>(), EOP.data_ptr<int64_t>(), ELP.data_ptr<int64_t>(),
              (const uint16_t*)X.data_ptr<at::BFloat16>(), (__nv_bfloat16*)Y.data_ptr<at::BFloat16>(),
              ids.data_ptr<int>(), nullptr, nullptr, nullptr,
              (int)S, (int)Y.size(1), (int)X.size(1), (int)xdiv, (int)TP.numel(), 0};
    mvw_kernel<0, int><<<nb, 128, 0, st>>>(p);
  } else {
    MW<int64_t> p{TP.data_ptr<int64_t>(), EOP.data_ptr<int64_t>(), ELP.data_ptr<int64_t>(),
                  (const uint16_t*)X.data_ptr<at::BFloat16>(), (__nv_bfloat16*)Y.data_ptr<at::BFloat16>(),
                  ids.data_ptr<int64_t>(), nullptr, nullptr, nullptr,
                  (int)S, (int)Y.size(1), (int)X.size(1), (int)xdiv, (int)TP.numel(), 0};
    mvw_kernel<0, int64_t><<<nb, 128, 0, st>>>(p);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mvwg(torch::Tensor TP, torch::Tensor EOP, torch::Tensor ELP, torch::Tensor X, torch::Tensor sorted,
          torch::Tensor bexp, torch::Tensor npost, torch::Tensor Y, int64_t xdiv, int64_t nbps) {
  const c10::cuda::OptionalCUDAGuard guard(X.device());
  mvw_check(TP, EOP, ELP, X, Y, "mvwg");
  auto i32 = [](const torch::Tensor& v) { return v.is_cuda() && v.scalar_type() == at::kInt && v.is_contiguous(); };
  TORCH_CHECK(i32(sorted) && i32(bexp) && i32(npost) && npost.numel() >= 1, "mvwg: sorted / bexp / npost int32");
  const int64_t S = Y.size(0), nblk = bexp.numel();
  TORCH_CHECK(xdiv >= 1 && X.size(0) * xdiv >= S, "mvwg: rows");
  if (S == 0 || nblk == 0) return;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const int nb = mvw_blocks(std::min<int64_t>(nblk, S) * (Y.size(1) / 16), nbps, X);   // <= min(blocks, S) chunks
  MW<int> p{TP.data_ptr<int64_t>(), EOP.data_ptr<int64_t>(), ELP.data_ptr<int64_t>(),
            (const uint16_t*)X.data_ptr<at::BFloat16>(), (__nv_bfloat16*)Y.data_ptr<at::BFloat16>(),
            nullptr, sorted.data_ptr<int>(), bexp.data_ptr<int>(), npost.data_ptr<int>(),
            (int)S, (int)Y.size(1), (int)X.size(1), (int)xdiv, (int)TP.numel(), (int)sorted.numel()};
  mvw_kernel<1, int><<<nb, 128, 0, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------------------------- mfd (fused MoE decode)
// The whole routed-expert block of a decode step in one launch, for few (token, expert) pairs:
//   out[t] = sum_j w[t, j] * W2[e] . (silu(W13gate[e] x[t]) * (W13up[e] x[t])),  e = ids[t k + j], j = 0 .. k-1
// (vLLM's w13 layout: gate rows [0, I), up rows [I, 2I)). Phase A, blocks [0, S I/16): block (slot s, pair p) computes
// rows p*16 .. +16 of the gate and of the up projection of slot s (4 warps split each tile's K), rounds both to BF16 and
// applies silu * up as silu_and_mul does, writes act[s, p*16 .. +16] and counts the slot's finished pairs. Phase B,
// blocks [S I/16, S I/16 + T H/16): block (token t, tile r) loads the down-projection records of its k experts first
// (warp j: slot j, the whole K = I), so they arrive while phase A runs, then waits until every pair of those slots is
// counted, multiplies, rounds each expert's rows to BF16 and sums them with the routing weights in fp32 in expert order,
// as wsum does: out[t, r*16 .. +16]. Deterministic; the last block of every token resets the counters (CUDA graphs).
// Blocks start in index order, so all phase A blocks have started before any phase B block waits on them.
struct MFD {
  const int64_t *TP13, *EOP13, *ELP13, *TP2, *EOP2, *ELP2;
  const uint16_t* X;          // [T, H]
  const float* W;             // [T, K] routing weights
  __nv_bfloat16* act;         // [S, I] scratch
  int* cnt;                   // [S] finished pairs per slot, then [T] finished tiles per token (zero between calls)
  __nv_bfloat16* out;         // [T, H]
  int T, K, H, I, nexp, nA;
};

struct RecB { uint4 c0, c1, c2; uint32_t e; int lo, hi; };

__device__ __forceinline__ void load_recb(RecB& f, const uint8_t* T, const int* EO, int tile, int G, int lane, int NG) {
  const size_t tg = (size_t)tile * NG + G;
  const uint8_t* base = T + tg * 1552;
  f.c0 = ldw(base + lane * 16);
  f.c1 = ldw(base + 512 + lane * 16);
  f.c2 = ldw(base + 1024 + lane * 16);
  f.e = __ldg((const uint16_t*)(base + 1536) + (lane >> 2));
  f.lo = __ldg(EO + tg);
  f.hi = __ldg(EO + tg + 1);
}

__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

template <typename IT>
__device__ __forceinline__ long long clamp_id(const IT* ids, int s, int nexp) {
  long long e = (long long)ids[s];
  return e < 0 ? 0 : (e >= nexp ? nexp - 1 : e);
}

__device__ __forceinline__ uint32_t silu_mul2(uint32_t gp, uint32_t up) {   // silu_and_mul on a bf16 pair
  const float2 gf = __bfloat1622float2(*(const __nv_bfloat162*)&gp), uf = __bfloat1622float2(*(const __nv_bfloat162*)&up);
  const float s0 = __bfloat162float(__float2bfloat16_rn(gf.x / (1.0f + expf(-gf.x))));
  const float s1 = __bfloat162float(__float2bfloat16_rn(gf.y / (1.0f + expf(-gf.y))));
  const __nv_bfloat162 r = __floats2bfloat162_rn(s0 * uf.x, s1 * uf.y);
  return *(const uint32_t*)&r;
}

template <int PF, typename IT>
__global__ void __launch_bounds__(256, 2) mfd_kernel(MFD p, const IT* ids) {
  __shared__ float red[8][16];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tid = threadIdx.x;
  const int g = lane >> 2, t = lane & 3;
  const int MT13 = 2 * p.I / 16;                              // 16-row tiles of one slot's gate (first half) and up rows
  if ((int)blockIdx.x < p.nA) {                               // ---- phase A: one 16-row tile of one slot, 8 warps split K
    const int s = blockIdx.x / MT13, tile = blockIdx.x % MT13;
    const int NG = p.H / 64, G0 = NG * warp / 8, G1 = NG * (warp + 1) / 8;
    const long long e = clamp_id(ids, s, p.nexp);
    const uint8_t* T = (const uint8_t*)p.TP13[e];
    const int* EO = (const int*)p.EOP13[e];
    const uint32_t* EL = (const uint32_t*)p.ELP13[e];
    const uint16_t* xr[1] = {(g == 0) ? p.X + (size_t)(s / p.K) * p.H : nullptr};
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    F7<1> f[PF];
#pragma unroll
    for (int i = 0; i < PF; ++i)
      if (G0 + i < G1) load_fg<1>(f[i], T, EO, xr, tile, G0 + i, lane, NG);
    for (int G = G0; G < G1; G += PF) {
#pragma unroll
      for (int i = 0; i < PF; ++i) {
        if (G + i < G1) {
          const uint32_t E0 = (f[i].e & 0xFFu) * 0x01010101u, E1 = ((f[i].e >> 8) & 0xFFu) * 0x01010101u;
#pragma unroll
          for (int st = 0; st < 4; ++st) {
            uint32_t A[4];
            dec_step5(f[i].c0, f[i].c1, f[i].c2, E0, E1, st, A);
            if (f[i].lo < f[i].hi) patch_regs(EL, f[i].lo, f[i].hi, lane, st, A);
            mma16816(c, A, f[i].x[0][2 * st], f[i].x[0][2 * st + 1]);
          }
          if (G + i + PF < G1) load_fg<1>(f[i], T, EO, xr, tile, G + i + PF, lane, NG);
        }
      }
    }
    if (t == 0) { red[warp][g] = c[0]; red[warp][g + 8] = c[2]; }   // column 0 = this slot's input row
    __syncthreads();
    if (tid < 16) {
      float sum = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) sum += red[k][tid];
      p.act[(size_t)s * 2 * p.I + tile * 16 + tid] = __float2bfloat16_rn(sum);   // h[s] (BF16, as the unfused path)
      __threadfence();
    }
    __syncthreads();
    if (tid == 0) atomicAdd(p.cnt + s, 1);
    return;
  }
  // ---- phase B: one 16-row tile of the down projection of one token, all k experts; silu(gate) * up on load
  const int b = blockIdx.x - p.nA, MT2 = p.H / 16;
  const int tk = b / MT2, r = b % MT2;
  const int S = p.T * p.K, NG2 = p.I / 64;
  const int j = warp;
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  if (j < p.K) {
    const int s = tk * p.K + j;
    const long long e = clamp_id(ids, s, p.nexp);
    const uint8_t* T = (const uint8_t*)p.TP2[e];
    const int* EO = (const int*)p.EOP2[e];
    const uint32_t* EL = (const uint32_t*)p.ELP2[e];
    RecB rb[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (i < NG2) load_recb(rb[i], T, EO, r, i, lane, NG2);   // the weights do not wait for phase A
    if (lane == 0)
      while (ld_acquire(p.cnt + s) < MT13) __nanosleep(32);
    __syncwarp();
    const uint32_t* hg = (const uint32_t*)(p.act + (size_t)s * 2 * p.I);
    const uint32_t* hu = hg + p.I / 2;
    for (int G0 = 0; G0 < NG2; G0 += 4) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int G = G0 + i;
        if (G < NG2) {
          if (G0 > 0) load_recb(rb[i], T, EO, r, G, lane, NG2);
          uint32_t xv[8];
#pragma unroll
          for (int st = 0; st < 4; ++st) {
            const int o = G * 32 + t + 8 * st;                    // uint32 index: bf16 column G*64 + 2t + 16 st
            xv[2 * st] = (g == 0) ? silu_mul2(__ldcg(hg + o), __ldcg(hu + o)) : 0u;
            xv[2 * st + 1] = (g == 0) ? silu_mul2(__ldcg(hg + o + 4), __ldcg(hu + o + 4)) : 0u;
          }
          const uint32_t E0 = (rb[i].e & 0xFFu) * 0x01010101u, E1 = ((rb[i].e >> 8) & 0xFFu) * 0x01010101u;
#pragma unroll
          for (int st = 0; st < 4; ++st) {
            uint32_t A[4];
            dec_step5(rb[i].c0, rb[i].c1, rb[i].c2, E0, E1, st, A);
            if (rb[i].lo < rb[i].hi) patch_regs(EL, rb[i].lo, rb[i].hi, lane, st, A);
            mma16816(c, A, xv[2 * st], xv[2 * st + 1]);
          }
        }
      }
    }
    if (t == 0) {                                              // this expert's rows, rounded as Y is in the unfused path
      red[j][g] = __bfloat162float(__float2bfloat16_rn(c[0]));
      red[j][g + 8] = __bfloat162float(__float2bfloat16_rn(c[2]));
    }
  }
  __syncthreads();
  if (tid < 16) {
    float acc = 0.f;
    for (int jj = 0; jj < p.K; ++jj) acc += p.W[(size_t)tk * p.K + jj] * red[jj][tid];
    p.out[(size_t)tk * p.H + r * 16 + tid] = __float2bfloat16_rn(acc);
  }
  if (tid == 0) {                                              // the token's last tile resets its counters
    if (atomicAdd(p.cnt + S + tk, 1) == MT2 - 1) {
      for (int jj = 0; jj < p.K; ++jj) p.cnt[tk * p.K + jj] = 0;
      p.cnt[S + tk] = 0;
    }
  }
}

void mfd(torch::Tensor TP13, torch::Tensor EOP13, torch::Tensor ELP13, torch::Tensor TP2, torch::Tensor EOP2,
         torch::Tensor ELP2, torch::Tensor X, torch::Tensor ids, torch::Tensor W, torch::Tensor act, torch::Tensor cnt,
         torch::Tensor out, int64_t I) {
  const c10::cuda::OptionalCUDAGuard guard(X.device());
  auto i64 = [](const torch::Tensor& v) { return v.is_cuda() && v.scalar_type() == at::kLong && v.is_contiguous(); };
  const bool ids32 = ids.is_cuda() && ids.scalar_type() == at::kInt && ids.is_contiguous();
  TORCH_CHECK(i64(TP13) && i64(EOP13) && i64(ELP13) && i64(TP2) && i64(EOP2) && i64(ELP2) &&
              TP13.numel() == TP2.numel() && EOP13.numel() == TP13.numel() && ELP13.numel() == TP13.numel() &&
              EOP2.numel() == TP2.numel() && ELP2.numel() == TP2.numel(), "mfd: pointer tables (int64, one size)");
  TORCH_CHECK(i64(ids) || ids32, "mfd: ids int32 / int64, contiguous");
  TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kBFloat16 && X.is_contiguous() && X.dim() == 2, "mfd: X");
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kBFloat16 && out.is_contiguous() && out.sizes() == X.sizes(),
              "mfd: out");
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == at::kFloat && W.is_contiguous() && W.dim() == 2 &&
              W.size(0) == X.size(0), "mfd: W float32 [T, k]");
  const int64_t T = X.size(0), H = X.size(1), K = W.size(1), S = T * K;
  TORCH_CHECK(ids.numel() == S && K >= 1 && K <= 8, "mfd: ids [T k], k <= 8");
  TORCH_CHECK(H % 64 == 0 && I % 64 == 0 && I >= 64, "mfd: H and I multiples of 64");
  TORCH_CHECK(act.is_cuda() && act.dtype() == torch::kBFloat16 && act.numel() >= S * 2 * I, "mfd: h scratch [S, 2I]");
  TORCH_CHECK(cnt.is_cuda() && cnt.scalar_type() == at::kInt && cnt.numel() >= S + T, "mfd: counters");
  if (T == 0) return;
  MFD p;
  p.TP13 = TP13.data_ptr<int64_t>(); p.EOP13 = EOP13.data_ptr<int64_t>(); p.ELP13 = ELP13.data_ptr<int64_t>();
  p.TP2 = TP2.data_ptr<int64_t>(); p.EOP2 = EOP2.data_ptr<int64_t>(); p.ELP2 = ELP2.data_ptr<int64_t>();
  p.X = (const uint16_t*)X.data_ptr<at::BFloat16>(); p.W = W.data_ptr<float>();
  p.act = (__nv_bfloat16*)act.data_ptr<at::BFloat16>(); p.cnt = cnt.data_ptr<int>();
  p.out = (__nv_bfloat16*)out.data_ptr<at::BFloat16>();
  p.T = (int)T; p.K = (int)K; p.H = (int)H; p.I = (int)I; p.nexp = (int)TP13.numel(); p.nA = (int)(S * (2 * I / 16));
  const int64_t nb = S * (2 * I / 16) + T * (H / 16);
  TORCH_CHECK(nb < (1ll << 31), "mfd: grid");
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (ids32) mfd_kernel<2, int><<<(unsigned)nb, 256, 0, st>>>(p, ids.data_ptr<int>());
  else mfd_kernel<2, int64_t><<<(unsigned)nb, 256, 0, st>>>(p, ids.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out[t, :] = sum_j w[t, j] * Y[t * K + j, :] (fp32 sum, BF16 out): the routing-weighted sum of the expert outputs
template <typename WT>
__device__ __forceinline__ float wval(WT v);
template <> __device__ __forceinline__ float wval<float>(float v) { return v; }
template <> __device__ __forceinline__ float wval<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename WT>
__global__ void __launch_bounds__(256) wsum_kernel(const __nv_bfloat16* Y, const WT* W, __nv_bfloat16* out, int T,
                                                   int K, int H) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int H8 = H / 8;
  if (i >= (long long)T * H8) return;
  const int t = (int)(i / H8), h = (int)(i % H8) * 8;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int j = 0; j < K; ++j) {
    const float w = wval<WT>(W[(size_t)t * K + j]);
    const uint4 v = *(const uint4*)(Y + ((size_t)t * K + j) * H + h);
    const __nv_bfloat162* p = (const __nv_bfloat162*)&v;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float2 f = __bfloat1622float2(p[q]);
      acc[2 * q] += w * f.x;
      acc[2 * q + 1] += w * f.y;
    }
  }
  uint4 o;
  __nv_bfloat162* po = (__nv_bfloat162*)&o;
#pragma unroll
  for (int q = 0; q < 4; ++q) po[q] = __floats2bfloat162_rn(acc[2 * q], acc[2 * q + 1]);
  *(uint4*)(out + (size_t)t * H + h) = o;
}

void wsum(torch::Tensor Y, torch::Tensor W, torch::Tensor out) {
  const c10::cuda::OptionalCUDAGuard guard(Y.device());
  TORCH_CHECK(Y.is_cuda() && Y.dtype() == torch::kBFloat16 && Y.is_contiguous() && Y.dim() == 2, "wsum: Y");
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kBFloat16 && out.is_contiguous() && out.dim() == 2, "wsum: out");
  TORCH_CHECK(W.is_cuda() && W.is_contiguous() && W.dim() == 2, "wsum: W");
  const int64_t T = out.size(0), K = W.size(1), Hd = out.size(1);
  TORCH_CHECK(W.size(0) == T && Y.size(0) == T * K && Y.size(1) == Hd && Hd % 8 == 0, "wsum: shapes");
  if (T == 0) return;
  const long long n = T * (Hd / 8);
  const unsigned grid = (unsigned)((n + 255) / 256);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (W.scalar_type() == at::kFloat)
    wsum_kernel<float><<<grid, 256, 0, st>>>((const __nv_bfloat16*)Y.data_ptr<at::BFloat16>(), W.data_ptr<float>(),
                                            (__nv_bfloat16*)out.data_ptr<at::BFloat16>(), (int)T, (int)K, (int)Hd);
  else if (W.scalar_type() == at::kBFloat16)
    wsum_kernel<__nv_bfloat16><<<grid, 256, 0, st>>>((const __nv_bfloat16*)Y.data_ptr<at::BFloat16>(),
                                                     (const __nv_bfloat16*)W.data_ptr<at::BFloat16>(),
                                                     (__nv_bfloat16*)out.data_ptr<at::BFloat16>(), (int)T, (int)K, (int)Hd);
  else TORCH_CHECK(false, "wsum: W must be float32 or bfloat16");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------------------------- dec7
// one warp per (16-row tile, group): decode to registers, patch escapes, stage the 16 x 64 block in shared memory,
// write 16 rows x 128 B with 16-byte stores.
__global__ void __launch_bounds__(256) dec7_kernel(A7 a, uint16_t* W) {
  __shared__ __align__(16) uint32_t buf[8][16 * 36];
  const int NG = a.N / 64;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const long long tg = (long long)blockIdx.x * 8 + warp;
  if (tg >= (long long)(a.M / 16) * NG) return;
  const int tile = (int)(tg / NG), G = (int)(tg % NG);
  const int g = lane >> 2, t = lane & 3;
  const uint8_t* base = a.T + tg * 1552;
  const uint4 c0 = ldw(base + lane * 16), c1 = ldw(base + 512 + lane * 16), c2 = ldw(base + 1024 + lane * 16);
  const uint32_t e = __ldg((const uint16_t*)(base + 1536) + g);
  const uint32_t E0 = (e & 0xFFu) * 0x01010101u, E1 = ((e >> 8) & 0xFFu) * 0x01010101u;
  const int lo = __ldg(a.EO + tg), hi = __ldg(a.EO + tg + 1);
  uint32_t* b = buf[warp];                       // row r: 36 words (64 bf16 + pad, conflict-free)
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    uint32_t A[4];
    dec_step5(c0, c1, c2, E0, E1, s, A);
    if (lo < hi) patch_regs(a.EL, lo, hi, lane, s, A);
    b[g * 36 + 8 * s + t] = A[0];
    b[g * 36 + 8 * s + 4 + t] = A[2];
    b[(g + 8) * 36 + 8 * s + t] = A[1];
    b[(g + 8) * 36 + 8 * s + 4 + t] = A[3];
  }
  __syncwarp();
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int idx = k * 32 + lane, r = idx >> 3, q = idx & 7;    // row r, 16-byte chunk q
    const uint4 v = *(const uint4*)(b + r * 36 + q * 4);
    *(uint4*)(W + (size_t)(tile * 16 + r) * a.N + G * 64 + q * 8) = v;
  }
}

void dec7(torch::Tensor T, torch::Tensor EO, torch::Tensor EL, torch::Tensor W) {
  const c10::cuda::OptionalCUDAGuard guard(T.device());
  A7 a;
  a.T = T.data_ptr<uint8_t>(); a.EO = EO.data_ptr<int>(); a.EL = (const uint32_t*)EL.data_ptr<int>();
  a.M = W.size(0); a.N = W.size(1);
  const long long n = (long long)(a.M / 16) * (a.N / 64);
  dec7_kernel<<<(unsigned)((n + 7) / 8), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      a, (uint16_t*)W.data_ptr<at::BFloat16>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------------------------- rows (embedding)
// out[i, :] = W[ids[i], :]: one warp per (id, 64-column group) decodes the 16 x 64 record holding that row (the dec7
// body, escapes patched) and copies the requested row with 16-byte stores.
__global__ void __launch_bounds__(256) rows7_kernel(A7 a, const int64_t* ids, int n, uint16_t* out) {
  __shared__ __align__(16) uint32_t buf[8][16 * 36];
  const int NG = a.N / 64;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const long long w = (long long)blockIdx.x * 8 + warp;
  if (w >= (long long)n * NG) return;
  const int i = (int)(w / NG), G = (int)(w % NG);
  const long long id = ids[i];
  const long long tg = (id >> 4) * NG + G;
  const int row = (int)(id & 15);
  const int g = lane >> 2, t = lane & 3;
  const uint8_t* base = a.T + tg * 1552;
  const uint4 c0 = ldw(base + lane * 16), c1 = ldw(base + 512 + lane * 16), c2 = ldw(base + 1024 + lane * 16);
  const uint32_t e = __ldg((const uint16_t*)(base + 1536) + g);
  const uint32_t E0 = (e & 0xFFu) * 0x01010101u, E1 = ((e >> 8) & 0xFFu) * 0x01010101u;
  const int lo = __ldg(a.EO + tg), hi = __ldg(a.EO + tg + 1);
  uint32_t* b = buf[warp];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    uint32_t A[4];
    dec_step5(c0, c1, c2, E0, E1, s, A);
    if (lo < hi) patch_regs(a.EL, lo, hi, lane, s, A);
    b[g * 36 + 8 * s + t] = A[0];
    b[g * 36 + 8 * s + 4 + t] = A[2];
    b[(g + 8) * 36 + 8 * s + t] = A[1];
    b[(g + 8) * 36 + 8 * s + 4 + t] = A[3];
  }
  __syncwarp();
  if (lane < 8) *(uint4*)(out + (size_t)i * a.N + G * 64 + lane * 8) = *(const uint4*)(b + row * 36 + lane * 4);
}

void rows7(torch::Tensor T, torch::Tensor EO, torch::Tensor EL, torch::Tensor ids, torch::Tensor out) {
  const c10::cuda::OptionalCUDAGuard guard(T.device());
  TORCH_CHECK(ids.scalar_type() == at::kLong && ids.is_contiguous(), "rows: ids must be contiguous int64");
  A7 a;
  a.T = T.data_ptr<uint8_t>(); a.EO = EO.data_ptr<int>(); a.EL = (const uint32_t*)EL.data_ptr<int>();
  a.M = (int)T.size(0) * 16; a.N = (int)out.size(1);
  const long long n = ids.numel();
  if (n == 0) return;
  const long long nw = n * (a.N / 64);
  rows7_kernel<<<(unsigned)((nw + 7) / 8), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      a, ids.data_ptr<int64_t>(), (int)n, (uint16_t*)out.data_ptr<at::BFloat16>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__device__ __forceinline__ void cp_async16_ef(uint32_t dst, const void* src) {
  asm volatile(
      "{\n"
      "   .reg .b64 p;\n"
      "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;\n"
      "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, p;\n"
      "}\n" :: "r"(dst), "l"(src));
}

struct ATM {
  const uint8_t* T; const uint32_t* EB; const int* EOU; const uint32_t* EALL;
  const uint16_t* X; __nv_bfloat16* Y; float* WS; int* CNT;
  int M, N, B, nRB, PMAX, ef, flags;
};

__device__ __forceinline__ void patch2(uint32_t* A, const uint32_t* esc, int n, int rec, int s, int lane) {
#pragma unroll 1
  for (int e = 0; e < n; ++e) {
    const uint32_t v = esc[e];
    if ((int)((v >> 10) & 31u) == rec && (int)((v >> 3) & 3u) == s && (int)((v >> 5) & 31u) == lane) {
      const uint32_t sh = (v & 1u) * 16u, mask = 0xFFFFu << sh, val = (v >> 16) << sh;
      const int q = (v >> 1) & 3;
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) A[qq] = (q == qq) ? ((A[qq] & ~mask) | val) : A[qq];
    }
  }
}

__device__ __forceinline__ void cpa_p(uint32_t dst, const void* src, bool p) {
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %0, 0;\n @q cp.async.cg.shared.global [%1], [%2], 16;\n}\n"
               :: "r"((int)p), "r"(dst), "l"(src));
}
__device__ __forceinline__ void cpa_p_ef(uint32_t dst, const void* src, bool p) {
  asm volatile("{\n .reg .pred q;\n .reg .b64 pol;\n setp.ne.b32 q, %0, 0;\n"
               " createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n"
               " @q cp.async.cg.shared.global.L2::cache_hint [%1], [%2], 16, pol;\n}\n"
               :: "r"((int)p), "r"(dst), "l"(src));
}

__device__ __forceinline__ int cta_of(int u, int U, int C) {
  return (int)(((u + 1) * C + U - 1) / U) - 1;
}

template <int R, int NTW, int WR, int WB, int ST, int OCC>
__global__ void __launch_bounds__(32 * WR * WB, OCC) tm_kernel(ATM a) {
  constexpr int NW = WR * WB, NTH = 32 * NW, RT = WR * R, BB = WB * NTW * 8;
  constexpr int XB = BB * 128, TB = RT * 1552, SB = XB + TB + 64;
  constexpr int KX = (BB * 8 + NTH - 1) / NTH, KR = (RT * 97 + NTH - 1) / NTH;
  constexpr int PE = NTH * R * NTW * 4;
  static_assert(NTW % 2 == 0 && RT <= 16 && SB % 16 == 0 && NTH % 8 == 0 && ST >= 2 && (8 % R == 0), "tm cfg");
  extern __shared__ __align__(128) uint8_t sm[];
  __shared__ int s_last;
  const int NG = a.N >> 6;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wr = warp % WR, wb = warp / WR;
  const int g = lane >> 2, t = lane & 3;
  const int n0 = blockIdx.y * BB;
  const int C = gridDim.x, cta = blockIdx.x;
  const int U = (int)a.nRB * NG;
  const int u0 = U * cta / C, u1 = U * (cta + 1) / C;
  const int nIt = (int)(u1 - u0);
  const uint32_t sbase = smem_u32(sm);
  const bool do_x = !(a.flags & 12), do_w = !(a.flags & 8), do_c = !(a.flags & 2), do_e = !(a.flags & 9);

  // per-thread copy plans (constant over the kernel)
  const int xr = tid >> 3, xc = tid & 7;
  const uint32_t xdst0 = (uint32_t)((xr * 8 + (xc ^ (xr & 7))) * 16);
  int xo[KX];
  uint32_t xmask = 0, xact = 0;
#pragma unroll
  for (int k = 0; k < KX; ++k) {
    const int row = n0 + xr + k * (NTH / 8);
    xo[k] = (row < a.B ? row : a.B - 1) * a.N + xc * 8;
    if (row < a.B) xmask |= 1u << k;
    if (tid + k * NTH < BB * 8) xact |= 1u << k;
  }
  int rsrc[KR];
  uint32_t ract = 0;
#pragma unroll
  for (int k = 0; k < KR; ++k) {
    const int idx = tid + k * NTH, i = idx / 97, ch = idx - i * 97;
    rsrc[k] = i * NG * 1552 + ch * 16;
    if (idx < RT * 97) ract |= 1u << k;
  }
  // load-side stage state (advanced incrementally)
  int lrb = (int)(u0 / NG), lG = (int)(u0 - (int)lrb * NG), ls = 0, nload = 0;
  const uint8_t* wst = a.T + ((size_t)lrb * RT * NG + lG) * 1552;
  const uint32_t* est = a.EB + (size_t)u0 * 16;

  auto load_next = [&]() {
    if (nload < nIt) {
      const uint32_t st = sbase + (uint32_t)(ls * SB);
      if (do_x) {
#pragma unroll
        for (int k = 0; k < KX; ++k) {
          if ((BB * 8) % NTH == 0 || ((xact >> k) & 1u))
            cp_async16(st + xdst0 + k * (NTH / 8) * 128, a.X + xo[k] + lG * 64, ((xmask >> k) & 1u) ? 16 : 0);
        }
      }
      if (do_w) {
        if (a.ef) {
#pragma unroll
          for (int k = 0; k < KR; ++k) cpa_p_ef(st + XB + (tid + k * NTH) * 16, wst + rsrc[k], (ract >> k) & 1u);
        } else {
#pragma unroll
          for (int k = 0; k < KR; ++k) cpa_p(st + XB + (tid + k * NTH) * 16, wst + rsrc[k], (ract >> k) & 1u);
        }
        if (tid < 4) cp_async16(st + XB + TB + tid * 16, est + tid * 4, 16);
      }
      ++nload;
      ls = (ls + 1 == ST) ? 0 : ls + 1;
      est += 16;
      if (++lG == NG) { lG = 0; ++lrb; wst = a.T + (size_t)lrb * RT * NG * 1552; }
      else wst += 1552;
    }
  };

  const uint32_t lbase = (uint32_t)((wb * NTW * 8 + (lane >> 4) * 8 + (lane & 7)) * 128);
  uint32_t loff[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) loff[s] = (uint32_t)(((2 * s + ((lane >> 3) & 1)) ^ (lane & 7)) * 16);

  float acc[R][NTW][4];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int j = 0; j < NTW; ++j) acc[r][j][0] = acc[r][j][1] = acc[r][j][2] = acc[r][j][3] = 0.f;

  const int rec0 = wr * R;

  auto compute = [&](int slot, int u) {
    const uint8_t* sp = sm + slot * SB;
    const uint32_t* esc = (const uint32_t*)(sp + XB + TB);
    const uint32_t hw = (rec0 < 8) ? esc[0] : esc[1];       // a warp's records never straddle record 8 (R | 8)
    const uint32_t hm = do_e ? (hw >> ((rec0 & 7) * 4)) : 0u;  // bit 4 r + s: record rec0 + r, k-step s
    const int nesc = (int)esc[2];                           // > 13: the unit's full list is in global memory
    const uint32_t* elist = (nesc > 13) ? a.EALL + __ldg(a.EOU + u) : esc + 3;
    const uint32_t xs = sbase + (uint32_t)(slot * SB) + lbase;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint32_t A[R][4];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint8_t* rp = sp + XB + (rec0 + r) * 1552;
        const uint2 lw = *(const uint2*)(rp + (s >> 1) * 512 + lane * 16 + (s & 1) * 8);
        const uint32_t sw = *(const uint32_t*)(rp + 1024 + lane * 16 + s * 4);
        const uint32_t ew = *(const uint16_t*)(rp + 1536 + g * 2);
        dec4(lw.x, sw & 0x0F0F0F0Fu, (ew & 0xFFu) * 0x01010101u, A[r][0], A[r][2]);
        dec4(lw.y, (sw >> 4) & 0x0F0F0F0Fu, (ew >> 8) * 0x01010101u, A[r][1], A[r][3]);
        if ((hm >> (4 * r + s)) & 1u) patch2(A[r], elist, nesc, rec0 + r, s, lane);
      }
      const uint32_t xa = xs + loff[s];
#pragma unroll
      for (int jj = 0; jj < NTW / 2; ++jj) {
        uint32_t b0, b1, b2, b3;
        ldsm_x4(xa + jj * 2048, b0, b1, b2, b3);
#pragma unroll
        for (int r = 0; r < R; ++r) {
          mma16816(acc[r][2 * jj], A[r], b0, b1);
          mma16816(acc[r][2 * jj + 1], A[r], b2, b3);
        }
      }
    }
  };

  const int odd = g & 1;
  __nv_bfloat16* const ybase = a.Y + (size_t)(n0 + wb * NTW * 8 + 2 * t + odd) * a.M + rec0 * 16 + g - odd;
  const int nlane = n0 + wb * NTW * 8 + 2 * t + odd;
  auto finish = [&](int rb, bool full) {
    bool out = full;
    if (!full) {
      const int tile_id = blockIdx.y * a.nRB + rb;
      const int cf = cta_of(rb * NG, U, C), cl = cta_of(rb * NG + NG - 1, U, C);
      const int P = cl - cf + 1, p = cta - cf;
      float4* ws = (float4*)(a.WS + (size_t)tile_id * a.PMAX * PE) + warp * R * NTW * 32 + lane;
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
          ws[(size_t)p * (PE / 4) + (r * NTW + j) * 32] = make_float4(acc[r][j][0], acc[r][j][1], acc[r][j][2], acc[r][j][3]);
      __threadfence();
      __syncthreads();
      if (tid == 0) s_last = (atomicAdd(&a.CNT[tile_id], 1) == P - 1);
      __syncthreads();
      out = s_last;
      if (out) {
        __threadfence();
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
          for (int j = 0; j < NTW; ++j) acc[r][j][0] = acc[r][j][1] = acc[r][j][2] = acc[r][j][3] = 0.f;
        for (int q = 0; q < P; ++q) {               // fixed order, own partial included: run-to-run deterministic
          const float4* wq = ws + (size_t)q * (PE / 4);
#pragma unroll
          for (int r = 0; r < R; ++r)
#pragma unroll
            for (int j = 0; j < NTW; ++j) {
              const float4 v = __ldcg(wq + (r * NTW + j) * 32);
              acc[r][j][0] += v.x; acc[r][j][1] += v.y; acc[r][j][2] += v.z; acc[r][j][3] += v.w;
              asm volatile("" ::: "memory");
            }
        }
        if (tid == 0) a.CNT[tile_id] = 0;
      }
    }
    if (out) {
      __nv_bfloat16* yb = ybase + (size_t)rb * RT * 16;
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const float v0 = acc[r][j][2 * h], v1 = acc[r][j][2 * h + 1];
            const float recv = __shfl_xor_sync(0xFFFFFFFFu, odd ? v0 : v1, 4);
            const __nv_bfloat162 pr = __floats2bfloat162_rn(odd ? recv : v0, odd ? v1 : recv);
            if (nlane + j * 8 < a.B) *(__nv_bfloat162*)(yb + (size_t)j * 8 * a.M + r * 16 + 8 * h) = pr;
          }
    }
  };

#pragma unroll
  for (int q = 0; q < ST - 1; ++q) { load_next(); cp_commit(); }
  int cs = 0, crb = (int)(u0 / NG), cG = (int)(u0 - (int)crb * NG);
  bool segfull = (cG == 0);
  for (int it = 0; it < nIt; ++it) {
    cp_wait<ST - 2>();
    __syncthreads();
    load_next();
    cp_commit();
    if (do_c) compute(cs, u0 + it);
    if (cG == NG - 1 || it == nIt - 1) {
      finish(crb, segfull && cG == NG - 1);
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j) acc[r][j][0] = acc[r][j][1] = acc[r][j][2] = acc[r][j][3] = 0.f;
      segfull = true;
    }
    if (++cG == NG) { cG = 0; ++crb; }
    cs = (cs + 1 == ST) ? 0 : cs + 1;
  }
  cp_wait<0>();
}

template <int R, int NTW, int WR, int WB, int ST, int OCC>
void launch_tm(ATM a, int C, cudaStream_t st) {
  constexpr int RT = WR * R, BB = WB * NTW * 8, NTH = 32 * WR * WB;
  constexpr int SB = BB * 128 + RT * 1552 + 64;
  const size_t smem = (size_t)ST * SB;
  static bool attr[64] = {false};                    // the attribute is per device
  int dev = 0;
  cudaGetDevice(&dev);
  if (!attr[dev & 63]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(tm_kernel<R, NTW, WR, WB, ST, OCC>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    attr[dev & 63] = true;
  }
  TORCH_CHECK(a.M % (16 * RT) == 0 && a.N % 64 == 0 && C >= 1, "tm: shape / grid");
  a.nRB = a.M / (16 * RT);
  TORCH_CHECK((long)a.nRB * (a.N / 64) * ((long)C + 1) < (1L << 31), "tm: 32-bit stream-K overflow");
  const int nBB = (a.B + BB - 1) / BB;
  tm_kernel<R, NTW, WR, WB, ST, OCC><<<dim3((unsigned)C, (unsigned)nBB), NTH, smem, st>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tm(torch::Tensor T, torch::Tensor EB, torch::Tensor EOU, torch::Tensor EALL, torch::Tensor X, torch::Tensor Y,
         torch::Tensor WS, torch::Tensor CNT, int64_t cfg, int64_t C, int64_t PMAX, int64_t ef, int64_t flags) {
  const c10::cuda::OptionalCUDAGuard guard(T.device());
  ATM a;
  a.T = T.data_ptr<uint8_t>(); a.EB = (const uint32_t*)EB.data_ptr<int>(); a.EOU = EOU.data_ptr<int>();
  a.EALL = (const uint32_t*)EALL.data_ptr<int>(); a.X = (const uint16_t*)X.data_ptr<at::BFloat16>();
  a.Y = (__nv_bfloat16*)Y.data_ptr<at::BFloat16>(); a.WS = WS.data_ptr<float>(); a.CNT = CNT.data_ptr<int>();
  a.M = Y.size(1); a.N = X.size(1); a.B = X.size(0); a.nRB = 0; a.PMAX = (int)PMAX; a.ef = (int)ef;
  a.flags = (int)flags;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const int c = (int)C;
  switch (cfg) {
    case 0: launch_tm<1, 2, 8, 1, 6, 1>(a, c, st); break;
    case 1: launch_tm<1, 2, 8, 1, 4, 2>(a, c, st); break;
    case 2: launch_tm<1, 4, 8, 1, 6, 1>(a, c, st); break;
    case 3: launch_tm<1, 4, 8, 1, 4, 2>(a, c, st); break;
    case 4: launch_tm<1, 8, 8, 1, 6, 1>(a, c, st); break;
    case 5: launch_tm<1, 8, 8, 1, 3, 2>(a, c, st); break;
    case 6: launch_tm<1, 16, 8, 1, 4, 1>(a, c, st); break;
    case 7: launch_tm<1, 16, 8, 1, 2, 2>(a, c, st); break;
    case 8: launch_tm<2, 8, 8, 1, 2, 2>(a, c, st); break;
    case 9: launch_tm<1, 16, 4, 1, 2, 3>(a, c, st); break;
    case 10: launch_tm<2, 8, 4, 1, 2, 3>(a, c, st); break;
    case 14: launch_tm<1, 8, 4, 1, 2, 4>(a, c, st); break;
    default: TORCH_CHECK(false, "tm: bad cfg");
  }
}

// ---------------------------------------------------------------------------------------------- dense16 (ablation)
// Control for the format: the mv kernel above, reading plain BF16 instead of tree records. Per (16-row tile, 64-column
// group) one 2048-byte block in mma-fragment order: k-step s, lane l -> uint4 {a0, a1, a2, a3} at byte s * 512 + l * 16.
// Same tiling, split-K, prefetch depth and reduction as mv7; no decode, no escapes. statetree vs dense16 isolates the
// format (bytes read), dense16 vs cuBLAS isolates the kernel.
struct AD {
  const uint8_t* T; const uint16_t* X; __nv_bfloat16* Y; float* WS; int* CNT;
  int M, N, B, WM, WK;
};

template <int NT>
struct FD { uint4 c[4]; uint32_t x[NT][8]; };

template <int NT>
__device__ __forceinline__ void load_fd(FD<NT>& f, const AD& a, int tile, int G, int lane, int NG) {
  const int g = lane >> 2, t = lane & 3;
  const uint8_t* base = a.T + ((size_t)tile * NG + G) * 2048 + lane * 16;
#pragma unroll
  for (int s = 0; s < 4; ++s) f.c[s] = ldw(base + s * 512);
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int n = nt * 8 + g;
    if (n < a.B) {
      const uint32_t* xp = (const uint32_t*)(a.X + (size_t)n * a.N + G * 64 + 2 * t);
#pragma unroll
      for (int s = 0; s < 4; ++s) { f.x[nt][2 * s] = __ldg(xp + 8 * s); f.x[nt][2 * s + 1] = __ldg(xp + 8 * s + 4); }
    } else {
#pragma unroll
      for (int q = 0; q < 8; ++q) f.x[nt][q] = 0u;
    }
  }
}

template <int NT, int PF>
__global__ void __launch_bounds__(256) mvd_kernel(AD a) {
  extern __shared__ float smem[];
  const int NG = a.N / 64;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int wm = warp % a.WM, wk = warp / a.WM;
  const int tile = blockIdx.x * a.WM + wm;
  const int GY = gridDim.y;
  const int cg0 = NG * blockIdx.y / GY, cg1 = NG * (blockIdx.y + 1) / GY;
  const int G0 = cg0 + (cg1 - cg0) * wk / a.WK, G1 = cg0 + (cg1 - cg0) * (wk + 1) / a.WK;
  float c[NT][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) c[nt][0] = c[nt][1] = c[nt][2] = c[nt][3] = 0.f;
  FD<NT> f[PF];
#pragma unroll
  for (int i = 0; i < PF; ++i)
    if (G0 + i < G1) load_fd<NT>(f[i], a, tile, G0 + i, lane, NG);
  for (int G = G0; G < G1; G += PF) {
#pragma unroll
    for (int i = 0; i < PF; ++i) {
      if (G + i < G1) {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const uint32_t A[4] = {f[i].c[s].x, f[i].c[s].y, f[i].c[s].z, f[i].c[s].w};
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) mma16816(c[nt], A, f[i].x[nt][2 * s], f[i].x[nt][2 * s + 1]);
        }
        if (G + i + PF < G1) load_fd<NT>(f[i], a, tile, G + i + PF, lane, NG);
      }
    }
  }
  const int R = a.WM * 16, NBC = NT * 8, EL = R * NBC;
  float* red = smem;
  float* out = smem + a.WK * EL;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int rr = wm * 16 + g, n = nt * 8 + 2 * t;
    red[wk * EL + n * R + rr] = c[nt][0];
    red[wk * EL + (n + 1) * R + rr] = c[nt][1];
    red[wk * EL + n * R + rr + 8] = c[nt][2];
    red[wk * EL + (n + 1) * R + rr + 8] = c[nt][3];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < EL; idx += blockDim.x) {
    float s = 0.f;
    for (int k = 0; k < a.WK; ++k) s += red[k * EL + idx];
    out[idx] = s;
  }
  __syncthreads();
  A7 m;
  m.WS = a.WS; m.CNT = a.CNT;
  if (GY > 1 && !merge7(m, out, EL, blockIdx.x, blockIdx.y, GY)) return;
  for (int idx = threadIdx.x; idx < EL; idx += blockDim.x) {
    const int n = idx / R, r = idx % R;
    if (n < a.B) a.Y[(size_t)n * a.M + (size_t)blockIdx.x * R + r] = __float2bfloat16_rn(out[idx]);
  }
}

template <int NT, int PF>
void launch_mvd(const AD& a, int GY, cudaStream_t st) {
  dim3 grid(a.M / (16 * a.WM), GY);
  const size_t sm = (size_t)(a.WK + 1) * a.WM * 16 * NT * 8 * sizeof(float);
  mvd_kernel<NT, PF><<<grid, 32 * a.WM * a.WK, sm, st>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mvd(torch::Tensor T, torch::Tensor X, torch::Tensor Y, torch::Tensor WS, torch::Tensor CNT, int64_t WM,
         int64_t WK, int64_t GY, int64_t PF) {
  const c10::cuda::OptionalCUDAGuard guard(T.device());
  AD a;
  a.T = T.data_ptr<uint8_t>();
  a.X = (const uint16_t*)X.data_ptr<at::BFloat16>(); a.Y = (__nv_bfloat16*)Y.data_ptr<at::BFloat16>();
  a.WS = WS.data_ptr<float>(); a.CNT = CNT.data_ptr<int>();
  a.M = Y.size(1); a.N = X.size(1); a.B = X.size(0); a.WM = WM; a.WK = WK;
  TORCH_CHECK(a.B >= 1 && a.B <= 32 && a.M % (16 * a.WM) == 0 && a.N % 64 == 0 && a.WM * a.WK <= 8, "mvd config");
  TORCH_CHECK(T.numel() == (int64_t)a.M * a.N * 2, "mvd: weight blocks do not match the shapes");
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const int G = (int)GY;
  if (a.B <= 8) { if (PF == 4) launch_mvd<1, 4>(a, G, st); else launch_mvd<1, 2>(a, G, st); }
  else if (a.B <= 16) { if (PF == 4) launch_mvd<2, 4>(a, G, st); else launch_mvd<2, 2>(a, G, st); }
  else { if (PF == 4) launch_mvd<4, 4>(a, G, st); else launch_mvd<4, 2>(a, G, st); }
}
