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
// 4 weights: leaf bytes, states (bytes 0..15), node exponent * 0x01010101 -> bf16 pairs (w0,w1), (w2,w3)
__device__ __forceinline__ void dec4(uint32_t sm, uint32_t d, uint32_t E4, uint32_t& p01, uint32_t& p23) {
  const uint32_t e4 = E4 - d;
  const uint32_t hi = (sm & 0x80808080u) | ((e4 >> 1) & 0x7F7F7F7Fu);
  const uint32_t lo = ((e4 << 7) & 0x80808080u) | (sm & 0x7F7F7F7Fu);
  p01 = __byte_perm(lo, hi, 0x5140);
  p23 = __byte_perm(lo, hi, 0x7362);
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
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          uint32_t A[4];
          dec_step5(f[i].c0, f[i].c1, f[i].c2, E0, E1, s, A);
          if (f[i].lo < f[i].hi) patch_regs(a.EL, f[i].lo, f[i].hi, lane, s, A);
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
  const size_t sm = (size_t)(a.WK + 1) * a.WM * 16 * NT * 8 * sizeof(float);
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
