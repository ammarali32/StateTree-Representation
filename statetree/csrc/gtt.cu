// gtt.cu: grouped (MoE) state-tree GEMM on A100 (sm_80): Y[s, :] = W[e(s)] . X[s / xdiv]^T for every (token, expert)
// slot s, the slots grouped by expert in vLLM's moe_align_block_size layout with block size TT (sorted[blk * TT + j] =
// slot id, >= S for padding; bexp[blk] = the block's expert; npost = used entries). W[e] is M x K in the lossless
// state-tree format v2 (expert e's records start at TP[e]). Escapes: the tt tables of the experts stacked into one
// [E * M, K] matrix, without padding (linear.gtt_escape_tables): ent holds every row block's entries back to back,
// koff[e * M / 128 + rb, kt] is the absolute start of k-tile kt's entries (koff[., K / 64] = the block's end), CAP the
// largest block's count (it sizes the shared-memory copy).
//
// One CTA per (token block, 128-row block): the tile of tt.cu (TT tokens x 128 rows, 8 warps = NWT token groups x
// 8/NWT row groups). Every warp LDGs its own 16x64 record two k-tiles ahead into registers, decodes it once for the
// whole CTA into a double-buffered swizzled BF16 tile and patches its escapes; all warps read W fragments with ldmatrix
// like a dense GEMM. The TT token rows are gathered through a 3-stage cp.async pipeline (padding slots re-read row 0;
// their outputs are never stored). Every tile runs the whole K: no workspace, no counters, the same sums every run.
// The last block of an expert is partly padding; skipping its MMAs (GTT_PADSKIP=1) breaks the unrolled pipeline and
// was slower, so padding rows are computed and dropped.
// Consecutive CTAs (blockIdx.x = token block, fastest) share a row block of one expert, so its records come from DRAM
// about once per row block.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace {
constexpr int NST = 3;           // activation stages
constexpr int RB = 128;          // weight rows per tile
constexpr int REC = 1552;        // bytes per 16 x 64 record
constexpr int WB = RB * 128;     // decoded W tile: 128 rows x 64 bf16
constexpr int SMEM_MAX = 166912;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cpa16(uint32_t d, const void* s) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(s));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void waitg() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void hmma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 ldg128(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}
// cheaper exact decode: lf = 4 leaves (m << 1 | partner sign), e4 = 4 exponents -> bf16 pairs (cols 2t, 2t+1)
__device__ __forceinline__ void dec4r(uint32_t lf, uint32_t e4, uint32_t& p01, uint32_t& p23) {
  const uint32_t a = __byte_perm(lf, e4, 0x5140), b = __byte_perm(lf, e4, 0x7362);
  p01 = __funnelshift_r(a, a, 1);
  p23 = __funnelshift_r(b, b, 1);
}
// 128-byte rows (64 bf16), 16-byte chunks XOR-swizzled by (row & 7): conflict-free for cp.async, STS and ldmatrix
__device__ __forceinline__ uint32_t swz(int row, int c) { return row * 128 + ((c ^ (row & 7)) << 4); }

struct GArgs {
  const __nv_bfloat16* X;   // [rows, K]
  const int64_t* TP;        // [E]: device address of expert e's records [M/16, K/64, 1552]
  const int* koff;          // [E * M/128, KS] absolute offsets into ent
  const int* ent;           // every row block's entries back to back (no padding)
  const int* sorted;        // [nblk * TT]
  const int* bexp;          // [nblk]
  const int* npost;         // [1]
  __nv_bfloat16* Y;         // [S, M]
  int S, M, K, KS, CAP, xdiv, E, esc_smem;
  int64_t nsorted;          // length of sorted (vLLM does not round it up to whole blocks for every block size)
};

#ifndef GTT_PADSKIP
#define GTT_PADSKIP 0   // 1: skip the MMAs of padding m-tiles (A100, Qwen3.6 shapes: 5-17% slower, it breaks the unrolled pipeline)
#endif

template <int TT, int NWT, int MINB>
__global__ void __launch_bounds__(256, MINB) gtt_kernel(const GArgs a) {
  constexpr int NTH = 256;                          // 8 warps = NWT token groups x NWR row groups
  constexpr int NWR = 8 / NWT;
  constexpr int WTOK = TT / NWT, WROW = 128 / NWR;  // warp tile: WTOK tokens x WROW weight rows
  constexpr int MW = WTOK / 16, NN = WROW / 8;      // m-tiles (16 tokens) and n-tiles (8 rows) per warp
  constexpr int XB = TT * 128;                      // bytes of one activation stage (TT x 64 bf16)
  constexpr int XCH = XB / 16 / NTH;                // 16-B activation chunks per thread per stage (rows r0 + 32 j)
  static_assert((NWT == 2 || NWT == 4) && WTOK % 16 == 0 && XCH >= 1, "gtt: tile shape");
  extern __shared__ __align__(128) uint8_t smem[];
  const int blk = blockIdx.x, rb = blockIdx.y;
  if (blk * TT >= __ldg(a.npost)) return;           // CTA-uniform: past the used blocks
  const int e = __ldg(a.bexp + blk);
  if (e < 0 || e >= a.E) return;
  const uint32_t sX = smem_u32(smem);
  const uint32_t sW = sX + NST * XB;                // 2 decoded W buffers
  uint8_t* const sWp = smem + NST * XB;
  int* const sSlot = (int*)(smem + NST * XB + 2 * WB);   // the block's TT slot ids
  int& sNval = sSlot[TT];                           // real slots of this block (they come first, then padding)
  int* const sKoff = sSlot + TT + 4;                // escape tables of this (expert, row block)
  int* const sEnt = sKoff + a.KS;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wt = warp / NWR, wr = warp % NWR;
  const int g = lane >> 2, t = lane & 3;
  const int rp = warp;                              // record (16-row piece of the 128-row tile) this warp decodes
  const std::true_type GU{};
  const std::false_type NG{};
  const int NKT = a.K >> 6;
  const int rbg = e * (a.M / RB) + rb;              // row block in the stacked escape tables
  const int ebase = __ldg(a.koff + (size_t)rbg * a.KS);   // this row block's entries: ent[ebase, koff[rbg, NKT])
  const int* const Es = a.esc_smem ? sEnt : a.ent;
  const int eoff = a.esc_smem ? ebase : 0;                // Es[i - eoff] = entry i

  for (int i = tid; i < TT; i += NTH) {
    const int64_t q = (int64_t)blk * TT + i;
    sSlot[i] = q < a.nsorted ? __ldg(a.sorted + q) : a.S;
  }
  for (int i = tid; i < (a.KS >> 2); i += NTH) cpa16(smem_u32(sKoff) + i * 16, a.koff + (size_t)rbg * a.KS + i * 4);
  if (a.esc_smem) {                                       // (<= CAP entries; plain loads, once per CTA)
    const int n = __ldg(a.koff + (size_t)rbg * a.KS + NKT) - ebase;
    for (int i = tid; i < n; i += NTH) sEnt[i] = __ldg(a.ent + ebase + i);
  }
  if (tid == 0) sNval = 0;
  __syncthreads();                                  // slot ids (plain stores) and sNval = 0 visible
  if (tid < TT && sSlot[tid] < a.S) atomicMax(&sNval, tid + 1);
  constexpr bool XREG = !GTT_PADSKIP || XCH <= 4;   // gathered row offsets in registers (else recomputed per stage)
  uint32_t xoff[XREG ? XCH : 1];
  auto xsrc = [&](int j) -> uint32_t {              // byte offset of this thread's j-th gathered row (chunk col c)
    const int s = sSlot[(tid + j * NTH) >> 3];
    const int src = s < a.S ? s / a.xdiv : 0;       // padding re-reads row 0 (its outputs are never stored)
    return (uint32_t)src * (uint32_t)(a.K * 2) + (uint32_t)((tid & 7) * 16);
  };
  if constexpr (XREG) {
#pragma unroll
    for (int j = 0; j < XCH; ++j) xoff[j] = xsrc(j);
  }
  __syncthreads();                                  // sNval final
  const int nval = sNval;
  const bool wlive = !GTT_PADSKIP || wt * WTOK < nval;   // warp-uniform: this warp's token group holds real tokens
  const char* const Xc = (const char*)a.X;
  const uint8_t* const Rw = (const uint8_t*)(uintptr_t)__ldg(a.TP + e) + (size_t)(rb * 8 + rp) * NKT * REC;

  float acc[MW][NN][4];
  uint32_t af[MW][4], bf[2][2 * NN];
  uint4 rc0[2], rc1[2], rc2[2];                    // records, two k-tiles ahead: leaves k-steps 0-1, 2-3, states
  uint32_t ree[2];                                  // nodes of rows g, g + 8
#pragma unroll
  for (int i = 0; i < MW; ++i)
#pragma unroll
    for (int j = 0; j < NN; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
  const int kb = 0, ke = NKT;

  auto copy_stage = [&](int kt, int st, auto guard) {   // gathered activations of k-tile kt -> stage st
    if (decltype(guard)::value && kt >= ke) return;
    const char* const xk = Xc + (size_t)kt * 128;
#pragma unroll
    for (int j = 0; j < XCH; ++j) {
      const int row = (tid + j * NTH) >> 3, c = tid & 7;
      if constexpr (XREG) cpa16(sX + st * XB + swz(row, c), xk + xoff[XREG ? j : 0]);
      else cpa16(sX + st * XB + swz(row, c), xk + xsrc(j));
    }
  };
  auto ldg_rec = [&](int kt, int b, auto guard) {   // this warp's record of k-tile kt -> register buffer b
    if (decltype(guard)::value && kt >= ke) return;
    const uint8_t* rec = Rw + (size_t)kt * REC;
    rc0[b] = ldg128(rec + lane * 16);
    rc1[b] = ldg128(rec + 512 + lane * 16);
    rc2[b] = ldg128(rec + 1024 + lane * 16);
    ree[b] = __ldg((const uint16_t*)(rec + 1536) + g);
  };
  auto dec_slice = [&](int buf, int b, int s) {   // k-step s of the record in register buffer b -> W buffer buf
    const uint32_t ee = ree[b];
    const uint32_t E0 = (ee & 0xFFu) * 0x01010101u, E1 = ((ee >> 8) & 0xFFu) * 0x01010101u;
    const uint4& c = (s < 2) ? rc0[b] : rc1[b];
    const uint32_t lw0 = (s & 1) ? c.z : c.x, lw1 = (s & 1) ? c.w : c.y;
    const uint32_t nw = s == 0 ? rc2[b].x : s == 1 ? rc2[b].y : s == 2 ? rc2[b].z : rc2[b].w;
    uint8_t* wb = sWp + buf * WB;
    const int r0 = 16 * rp + g, r1 = r0 + 8;
    uint32_t A0, A1, A2, A3;
    dec4r(lw0, E0 - (nw & 0x0F0F0F0Fu), A0, A2);
    dec4r(lw1, E1 - ((nw >> 4) & 0x0F0F0F0Fu), A1, A3);
    *(uint32_t*)(wb + swz(r0, 2 * s) + 4 * t) = A0;
    *(uint32_t*)(wb + swz(r0, 2 * s + 1) + 4 * t) = A2;
    *(uint32_t*)(wb + swz(r1, 2 * s) + 4 * t) = A1;
    *(uint32_t*)(wb + swz(r1, 2 * s + 1) + 4 * t) = A3;
  };
  auto esc_patch = [&](int buf, int kr) {          // escapes of k-tile kr in this warp's record, after its slices
    const int lo = sKoff[kr], hi = sKoff[kr + 1];  // CTA-uniform
    if (lo < hi) {
      uint8_t* wb = sWp + buf * WB;
      __syncwarp();
#pragma unroll 1
      for (int i = lo + lane; i < hi; i += 32) {
        const uint32_t v = (uint32_t)Es[i - eoff];
        if ((int)((v >> 10) & 7) == rp) {
          const int r = 16 * rp + (int)((v >> 6) & 15), col = (int)(v & 63);
          *(uint16_t*)(wb + swz(r, col >> 3) + (col & 7) * 2) = (uint16_t)(v >> 16);
        }
      }
    }
  };
  auto decode = [&](int buf, int b, int kr) {
#pragma unroll
    for (int s = 0; s < 4; ++s) dec_slice(buf, b, s);
    esc_patch(buf, kr);
  };
  auto load_A0 = [&](int st, int ks) {
#pragma unroll
    for (int m = 0; m < MW; ++m) {
      if (GTT_PADSKIP && wt * WTOK + m * 16 >= nval) continue;   // padding m-tile: no MMAs
      const int row = wt * WTOK + m * 16 + (lane & 15), c = 2 * ks + (lane >> 4);
      ldsm4(sX + st * XB + swz(row, c), af[m][0], af[m][1], af[m][2], af[m][3]);
    }
  };
  auto load_B = [&](uint32_t base, int ks) {
    if (!wlive) return;
#pragma unroll
    for (int j = 0; j < NN / 2; ++j) {
      const int row = wr * WROW + j * 16 + (lane & 7) + ((lane >> 4) << 3), c = 2 * ks + ((lane >> 3) & 1);
      ldsm4(base + swz(row, c), bf[ks & 1][4 * j], bf[ks & 1][4 * j + 1], bf[ks & 1][4 * j + 2], bf[ks & 1][4 * j + 3]);
    }
  };
  auto mma = [&](int ks, int nst, int nks) {
#pragma unroll
    for (int m = 0; m < MW; ++m) {
      if (GTT_PADSKIP && wt * WTOK + m * 16 >= nval) continue;   // padding m-tile
#pragma unroll
      for (int n = 0; n < NN; ++n) hmma(acc[m][n], af[m], bf[ks & 1][2 * n], bf[ks & 1][2 * n + 1]);
      const int row = wt * WTOK + m * 16 + (lane & 15), c = 2 * nks + (lane >> 4);
      ldsm4(sX + nst * XB + swz(row, c), af[m][0], af[m][1], af[m][2], af[m][3]);
    }
  };

  // prologue: activation stages 0, 1 (the escape tables ride in group 0), records 0, 1
  copy_stage(kb, 0, NG);
  commit();
  copy_stage(kb + 1, 1, GU);
  commit();
  ldg_rec(kb, 0, NG);
  ldg_rec(kb + 1, 1, GU);
  waitg<1>();
  __syncthreads();
  decode(0, 0, kb);
  __syncthreads();
  int rst = 0, wst = NST - 1;
  load_A0(0, 0);
  load_B(sW, 0);
  auto step = [&](const int kt, const int b, auto guard) {
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      if (ks < 3) load_B(sW + (b ^ 1) * WB, ks + 1);
      else load_B(sW + b * WB, 0);
      if (ks == 0) ldg_rec(kt + 2, b ^ 1, guard);
      mma(ks, rst, ks == 3 ? 0 : ks + 1);
      if (ks == 2) {
        if (!decltype(guard)::value || kt + 1 < ke) decode(b, b, kt + 1);
        copy_stage(kt + NST - 1, wst, guard);
        commit();
        waitg<NST - 2>();
        __syncthreads();
        rst = rst + 1 == NST ? 0 : rst + 1;
        wst = wst + 1 == NST ? 0 : wst + 1;
      }
    }
  };
  int kt = kb;
  for (; kt + 3 < ke; kt += 2) {
    step(kt, 1, NG);
    step(kt + 1, 0, NG);
  }
  if (kt + 1 < ke) {
    step(kt, 1, GU);
    step(kt + 1, 0, GU);
    kt += 2;
  }
  if (kt < ke) step(kt, 1, GU);
  waitg<0>();
  __syncthreads();                                  // all warps done with the stages / W buffers

  {   // bf16 tile -> shared (16-B chunks swizzled by token) -> Y rows of the block's real slots
    uint8_t* const sC = smem;
#pragma unroll
    for (int m = 0; m < MW; ++m)
#pragma unroll
      for (int n = 0; n < NN; ++n) {
        const int tok = wt * WTOK + m * 16 + g, col = wr * WROW + n * 8 + 2 * t;
        const int off = (((col >> 3) ^ (tok & 7)) << 4) + (col & 7) * 2;
        *(__nv_bfloat162*)(sC + tok * 256 + off) = __floats2bfloat162_rn(acc[m][n][0], acc[m][n][1]);
        *(__nv_bfloat162*)(sC + (tok + 8) * 256 + off) = __floats2bfloat162_rn(acc[m][n][2], acc[m][n][3]);
      }
    __syncthreads();
    for (int i = tid; i < TT * 16; i += NTH) {
      const int tok = i >> 4, ch = i & 15;
      const int s = sSlot[tok];
      if (s < a.S)
        *(uint4*)(a.Y + (size_t)s * a.M + rb * RB + ch * 8) = *(const uint4*)(sC + tok * 256 + ((ch ^ (tok & 7)) << 4));
    }
  }
}

template <int TT, int NWT, int MINB>
void launch(GArgs a, int nblk) {
  const int base = NST * TT * 128 + 2 * WB + TT * 4 + 16 + a.KS * 4;
  a.esc_smem = base + a.CAP * 4 <= SMEM_MAX / MINB - 1024;   // rare escape-heavy matrices: entries read from global
  const int smem = base + (a.esc_smem ? a.CAP * 4 : 0);
  TORCH_CHECK(smem <= SMEM_MAX, "gtt: shared memory ", smem, " too large");
  static bool set[64] = {false};
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  if (!set[dev & 63]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(gtt_kernel<TT, NWT, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
    set[dev & 63] = true;
  }
  const dim3 grid(nblk, a.M / RB);
  gtt_kernel<TT, NWT, MINB><<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// (TT, NWT, MINB) compiled: experts.gtt_cfg picks 32 / 64 / 128 tokens per block by the pairs per expert (A100 sweep
// of the Qwen3.6-35B-A3B expert shapes, 64 to 16384 tokens; 96 / 192 / 256 and mixed tiles per GEMM were never faster)
#define GTTLIST(X) X(32, 2, 2) X(64, 2, 2) X(64, 4, 2) X(128, 2, 1) X(128, 4, 1)

void run(int64_t TT, int64_t NWT, int64_t MINB, torch::Tensor TP, torch::Tensor koff, torch::Tensor ent, torch::Tensor X,
         torch::Tensor sorted, torch::Tensor bexp, torch::Tensor npost, torch::Tensor Y, int64_t xdiv, int64_t KS,
         int64_t CAP) {
  auto i32 = [](const torch::Tensor& v) { return v.is_cuda() && v.scalar_type() == at::kInt && v.is_contiguous(); };
  TORCH_CHECK(TP.is_cuda() && TP.scalar_type() == at::kLong && TP.is_contiguous(), "gtt: TP int64");
  TORCH_CHECK(i32(koff) && i32(ent) && i32(sorted) && i32(bexp) && i32(npost) && npost.numel() >= 1, "gtt: int32 tables");
  TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kBFloat16 && X.is_contiguous() && X.dim() == 2, "gtt: X");
  TORCH_CHECK(Y.is_cuda() && Y.dtype() == torch::kBFloat16 && Y.is_contiguous() && Y.dim() == 2, "gtt: Y");
  const int64_t S = Y.size(0), M = Y.size(1), K = X.size(1), E = TP.numel(), nblk = bexp.numel();
  TORCH_CHECK(M % RB == 0 && K % 64 == 0 && K >= 64, "gtt: shapes (M % 128, K % 64)");
  TORCH_CHECK(xdiv >= 1 && X.size(0) * xdiv >= S && X.size(0) * K * 2 < (1ll << 32), "gtt: rows");
  TORCH_CHECK(nblk <= 2147483647 / TT && sorted.numel() >= 1, "gtt: block count");
  TORCH_CHECK(KS % 4 == 0 && KS >= K / 64 + 1 && CAP % 4 == 0 && CAP >= 4, "gtt: escape strides");
  TORCH_CHECK(koff.numel() == E * (M / RB) * KS && ent.numel() >= 1, "gtt: escape tables");
  if (S == 0 || nblk == 0) return;
  GArgs a;
  a.X = (const __nv_bfloat16*)X.data_ptr();
  a.TP = TP.data_ptr<int64_t>();
  a.koff = koff.data_ptr<int>();
  a.ent = ent.data_ptr<int>();
  a.sorted = sorted.data_ptr<int>();
  a.bexp = bexp.data_ptr<int>();
  a.npost = npost.data_ptr<int>();
  a.Y = (__nv_bfloat16*)Y.data_ptr();
  a.nsorted = sorted.numel();
  a.S = (int)S; a.M = (int)M; a.K = (int)K; a.KS = (int)KS; a.CAP = (int)CAP; a.xdiv = (int)xdiv; a.E = (int)E;
#define GTTC(T_, N_, B_) if (TT == T_ && NWT == N_ && MINB == B_) { launch<T_, N_, B_>(a, (int)nblk); return; }
  GTTLIST(GTTC)
#undef GTTC
  TORCH_CHECK(false, "gtt: TT ", TT, " NWT ", NWT, " MINB ", MINB, " not compiled");
}
std::vector<int64_t> attrs(int64_t TT, int64_t NWT, int64_t MINB) {
  cudaFuncAttributes f;
#define GTTA(T_, N_, B_) if (TT == T_ && NWT == N_ && MINB == B_) { C10_CUDA_CHECK(cudaFuncGetAttributes(&f, gtt_kernel<T_, N_, B_>)); return {f.numRegs, (int64_t)f.localSizeBytes}; }
  GTTLIST(GTTA)
#undef GTTA
  return {-1, -1};
}
}  // namespace

void gtt_run(int64_t TT, int64_t NWT, int64_t MINB, torch::Tensor TP, torch::Tensor koff, torch::Tensor ent,
             torch::Tensor X, torch::Tensor sorted, torch::Tensor bexp, torch::Tensor npost, torch::Tensor Y,
             int64_t xdiv, int64_t KS, int64_t CAP) {
  const at::cuda::OptionalCUDAGuard guard(X.device());
  run(TT, NWT, MINB, TP, koff, ent, X, sorted, bexp, npost, Y, xdiv, KS, CAP);
}
std::vector<int64_t> gtt_attrs(int64_t TT, int64_t NWT, int64_t MINB) { return attrs(TT, NWT, MINB); }

#ifdef GTT_STANDALONE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gtt", &gtt_run, "grouped MoE GEMM, experts as state trees (moe_align_block_size layout, block size TT)");
  m.def("gtt_attrs", &gtt_attrs, "registers / local bytes of a compiled gtt configuration");
}
#endif
