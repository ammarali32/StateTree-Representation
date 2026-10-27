// tt.cu: one-token-tile state-tree GEMM for prefill (T > 64) on A100 (sm_80): Y[T, M] = X[T, K] . W[M, K]^T, with W in
// the lossless state-tree format v2 (leaf bytes pre-permuted to m << 1 | partner sign for the cheaper decode) and the
// escapes (D = 15) patched exactly.
//
// Tile = TT tokens x 128 weight rows (8 records of 16 rows); 8 warps = NWT token groups x 8/NWT row groups (NWT = 2:
// warp tile TT/2 tokens x 32 rows; NWT = 4: TT/4 x 64, cuBLAS's 64x64 shape for TT = 256, less fragment traffic).
// Every warp LDGs its own 16x64 record two k-tiles ahead into registers, decodes it once for the whole CTA into a
// double-buffered, swizzled BF16 shared tile, patches its escapes (after __syncwarp), and all warps read W fragments
// with ldmatrix exactly like a dense GEMM. Activations stream
// through a 3-stage cp.async pipeline (token rows >= T zero-filled).
//
// Work distribution: persistent CTAs, one per SM. Tiles = 128-row blocks x token tiles (token tile fastest). With
// NT >= 2 G tiles, (NT / G - 1) rounds of whole tiles go data-parallel (consecutive CTAs share a row block, so its
// records come from DRAM once); the remaining tiles are stream-K: their k-iterations are split evenly over all CTAs.
// Every piece stores its fp32 accumulators fragment-major (coalesced) into its workspace slot and bumps the tile's
// counter; the last arriver sums the pieces in a fixed order (bit-identical run to run, whichever piece arrives
// last), writes Y and resets the counter (no waiting: deadlock free, and counters are zero again after every launch).
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
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }
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
__device__ __forceinline__ uint2 ldg64(const void* p) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
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

struct Args {
  const __nv_bfloat16* X;   // [T, K]
  const uint8_t* R;         // [M/16, K/64, 1552], format v2
  const int* koff;          // [M/128, KS]: escapes of (row block rb, k-tile kt) are ent[rb][koff[rb][kt] .. koff[rb][kt+1])
  const int* ent;           // [M/128, CAP]: bf16 bits << 16 | record << 10 | row << 6 | col (record-local)
  __nv_bfloat16* Y;         // [T, M]
  float* ws;                // [2 G slots][128 * TT floats]: partial accumulators of split tiles
  int* cnt;                 // [>= split tiles], zero between launches
  int T, M, K, NTT, NT, KS, CAP, esc_smem, G, dp_tiles;   // esc_smem: entries staged in shared (else read from global)
  int split;                // 0: data-parallel rounds + stream-K; s >= 1: every tile cut into s equal k-ranges
  long long sk_iters;
};

template <int TT, int NWT>
__global__ void __launch_bounds__(256, 1) tt_kernel(const Args a) {
  constexpr int NTH = 256;                          // 8 warps = NWT token groups x NWR row groups
  constexpr int NWR = 8 / NWT;
  constexpr int WTOK = TT / NWT, WROW = 128 / NWR;  // warp tile: WTOK tokens x WROW weight rows
  constexpr int MW = WTOK / 16, NN = WROW / 8;      // m-tiles (16 tokens) and n-tiles (8 rows) per warp
  constexpr int XB = TT * 128;                      // bytes of one activation stage (TT x 64 bf16)
  constexpr int PF = MW * NN * 4;                   // accumulator floats per thread
  constexpr int XCH = XB / 16 / NTH;                // 16-B activation chunks per thread per stage (rows r0 + 32 j)
  static_assert((NWT == 2 || NWT == 4) && WTOK % 16 == 0, "tt: tile shape");
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sX = smem_u32(smem);
  const uint32_t sW = sX + NST * XB;                // 2 decoded W buffers
  uint8_t* const sWp = smem + NST * XB;
  int* const sFlag = (int*)(smem + NST * XB + 2 * WB);   // 64 B at a compile-time offset: flag + scheduler state
  int* const sKoff = sFlag + 16;                    // escape tables of the current row block
  int* const sEnt = sKoff + a.KS;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wt = warp / NWR, wr = warp % NWR;
  const int g = lane >> 2, t = lane & 3;
  const int rp = warp;                              // record (16-row piece of the 128-row tile) this warp decodes
  const std::true_type GU{};                        // bounds-checked (last k-tiles of a segment)
  const std::false_type NG{};                       // steady state: every k-tile touched exists
  const int NKT = a.K >> 6;
  const int cta = blockIdx.x;
  const int ndp = a.dp_tiles / a.G;
  const long long S = a.sk_iters;
  // scheduler state lives in shared memory, so the main loop keeps its registers for addressing:
  // sSeg = {next work index w, tile, kb, ke}, sS = {next SK iteration, end of this CTA's SK range}, sEs = escape list
  int* const sSeg = sFlag + 4;
  long long* const sS = (long long*)(sFlag + 8);
  const int** const sEs = (const int**)(sFlag + 12);
  if (tid == 0) {
    sSeg[0] = 0;
    sS[0] = (long long)cta * S / a.G;
    sS[1] = (long long)(cta + 1) * S / a.G;
  }
  __syncthreads();

  float acc[MW][NN][4];
  uint32_t af[MW][4], bf[2][2 * NN];
  uint4 rc0[2], rc1[2], rc2[2];                    // records, two k-tiles ahead: leaves k-steps 0-1, 2-3, states
  uint32_t ree[2];                                  // nodes of rows g, g + 8

  for (;;) {
    {                                               // next segment: whole tile (data-parallel) or stream-K piece
      const int w = sSeg[0];
      int tile, kb, ke;
      if (a.split > 0) {                            // balanced split: piece u = w * G + cta of NT * split
        const int u = w * a.G + cta;
        if (u >= a.NT * a.split) break;             // CTA-uniform
        tile = u / a.split;
        const int j = u - tile * a.split;
        kb = j * NKT / a.split;
        ke = (j + 1) * NKT / a.split;
      } else if (w < ndp) {
        tile = w * a.G + cta;
        kb = 0;
        ke = NKT;
      } else {
        const long long s0 = sS[0], s1 = sS[1];
        if (s0 >= s1) break;                        // CTA-uniform
        const int ts = (int)(s0 / NKT);
        tile = a.dp_tiles + ts;
        kb = (int)(s0 - (long long)ts * NKT);
        ke = (int)min((long long)NKT, (long long)kb + (s1 - s0));
      }
      __syncthreads();                              // everyone has read the state
      if (tid == 0) {
        sSeg[0] = w + 1;
        sSeg[1] = tile;
        sSeg[2] = kb;
        sSeg[3] = ke;
        if (a.split == 0 && w >= ndp) sS[0] += ke - kb;
        sFlag[14] = w * a.G + cta;                  // piece index (balanced split)
        const int rb = tile / a.NTT;
        sEs[0] = a.esc_smem ? sEnt : a.ent + (size_t)rb * a.CAP;
      }
      __syncthreads();
    }
    const int kb = sSeg[2], ke = sSeg[3];
    const int rb = sSeg[1] / a.NTT;
    const int tok0 = (sSeg[1] - rb * a.NTT) * TT;
    const uint8_t* const Rw = a.R + (size_t)(rb * 8 + rp) * NKT * REC;     // this warp's record row
    const char* const Xc = (const char*)a.X + (size_t)tok0 * a.K * 2 + (tid & 7) * 16;   // this thread's chunk column
    const int xlast = a.T - 1 - tok0;                 // tile rows > xlast re-read row xlast (outputs never stored)

#pragma unroll
    for (int i = 0; i < MW; ++i)
#pragma unroll
      for (int j = 0; j < NN; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

    auto copy_stage = [&](int kt, int st, auto guard) {   // activations of k-tile kt -> stage st
      if (decltype(guard)::value && kt >= ke) return;
      const char* const xk = Xc + (size_t)kt * 128;
#pragma unroll
      for (int j = 0; j < XCH; ++j) {
        const int row = (tid + j * NTH) >> 3, c = tid & 7;
        cpa16(sX + st * XB + swz(row, c), xk + (uint32_t)min(row, xlast) * (uint32_t)(a.K * 2));
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
      const uint32_t lw0 = (s & 1) ? c.z : c.x, lw1 = (s & 1) ? c.w : c.y;   // leaves of rows r0 / r1
      const uint32_t nw = s == 0 ? rc2[b].x : s == 1 ? rc2[b].y : s == 2 ? rc2[b].z : rc2[b].w;   // states r0 | r1
      uint8_t* wb = sWp + buf * WB;
      const int r0 = 16 * rp + g, r1 = r0 + 8;
      uint32_t A0, A1, A2, A3;
      dec4r(lw0, E0 - (nw & 0x0F0F0F0Fu), A0, A2);
      dec4r(lw1, E1 - ((nw >> 4) & 0x0F0F0F0Fu), A1, A3);
      // A0: row r0, cols 16s+2t,+1   A2: row r0, cols 16s+8+2t,+1   A1/A3: same columns, row r1
      *(uint32_t*)(wb + swz(r0, 2 * s) + 4 * t) = A0;
      *(uint32_t*)(wb + swz(r0, 2 * s + 1) + 4 * t) = A2;
      *(uint32_t*)(wb + swz(r1, 2 * s) + 4 * t) = A1;
      *(uint32_t*)(wb + swz(r1, 2 * s + 1) + 4 * t) = A3;
    };
    auto esc_patch = [&](int buf, int kr) {          // escapes of k-tile kr in this warp's record, after its slices
      const int lo = sKoff[kr], hi = sKoff[kr + 1];  // CTA-uniform
      if (lo < hi) {
        uint8_t* wb = sWp + buf * WB;
        __syncwarp();                                 // this warp's decoded stores before its escape stores
        const int* const Es = sEs[0];
#pragma unroll 1
        for (int i = lo + lane; i < hi; i += 32) {
          const uint32_t e = (uint32_t)Es[i];
          if ((int)((e >> 10) & 7) == rp) {
            const int r = 16 * rp + (int)((e >> 6) & 15), col = (int)(e & 63);
            *(uint16_t*)(wb + swz(r, col >> 3) + (col & 7) * 2) = (uint16_t)(e >> 16);
          }
        }
      }
    };
    auto decode = [&](int buf, int b, int kr) {     // whole record of k-tile kr (register buffer b) + its escapes
#pragma unroll
      for (int s = 0; s < 4; ++s) dec_slice(buf, b, s);
      esc_patch(buf, kr);
    };
    auto load_A0 = [&](int st, int ks) {
#pragma unroll
      for (int m = 0; m < MW; ++m) {
        const int row = wt * WTOK + m * 16 + (lane & 15), c = 2 * ks + (lane >> 4);
        ldsm4(sX + st * XB + swz(row, c), af[m][0], af[m][1], af[m][2], af[m][3]);
      }
    };
    auto load_B = [&](uint32_t base, int ks) {       // NN n-tiles: each ldmatrix.x4 covers 16 rows x 16 k
#pragma unroll
      for (int j = 0; j < NN / 2; ++j) {
        const int row = wr * WROW + j * 16 + (lane & 7) + ((lane >> 4) << 3), c = 2 * ks + ((lane >> 3) & 1);
        ldsm4(base + swz(row, c), bf[ks & 1][4 * j], bf[ks & 1][4 * j + 1], bf[ks & 1][4 * j + 2], bf[ks & 1][4 * j + 3]);
      }
    };
    auto mma = [&](int ks, int nst, int nks) {        // k-step ks; then reload each A m-tile for k-step nks of stage nst
#pragma unroll
      for (int m = 0; m < MW; ++m) {
#pragma unroll
        for (int n = 0; n < NN; ++n) hmma(acc[m][n], af[m], bf[ks & 1][2 * n], bf[ks & 1][2 * n + 1]);
        const int row = wt * WTOK + m * 16 + (lane & 15), c = 2 * nks + (lane >> 4);
        ldsm4(sX + nst * XB + swz(row, c), af[m][0], af[m][1], af[m][2], af[m][3]);
      }
    };

    // prologue: escape tables of this row block + activation stages kb, kb+1 (group 0 / 1), records kb, kb+1
    for (int i = tid; i < (a.KS >> 2); i += NTH) cpa16(smem_u32(sKoff) + i * 16, a.koff + (size_t)rb * a.KS + i * 4);
    if (a.esc_smem)
      for (int i = tid; i < (a.CAP >> 2); i += NTH) cpa16(smem_u32(sEnt) + i * 16, a.ent + (size_t)rb * a.CAP + i * 4);
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
    // one k-tile. Record register buffer and decoded W buffer of k-tile k are both (k - kb) & 1, so with
    // b = (kt + 1 - kb) & 1 (compile-time after inlining) k-tile kt uses buffer b ^ 1 and kt + 1 uses b.
    auto step = [&](const int kt, const int b, auto guard) {
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        if (ks < 3) load_B(sW + (b ^ 1) * WB, ks + 1);
        else load_B(sW + b * WB, 0);                  // k-step 0 of k-tile kt + 1 (decoded + synced at ks = 2)
        if (ks == 0) ldg_rec(kt + 2, b ^ 1, guard);  // two k-tiles ahead into the other register buffer
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
    for (; kt + 3 < ke; kt += 2) {                    // steady state: k-tiles up to kt + 3 exist, no bounds checks
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

    {   // prefetch the next segment's first two k-tiles (records, activations) into L2 while this one finishes
      const int nw = sSeg[0];
      int ntile = -1, nkb = 0;
      if (a.split > 0) {
        const int u = nw * a.G + cta;
        if (u < a.NT * a.split) {
          ntile = u / a.split;
          nkb = (u - ntile * a.split) * NKT / a.split;
        }
      } else if (nw < ndp) {
        ntile = nw * a.G + cta;
      } else if (sS[0] < sS[1]) {
        const int ts = (int)(sS[0] / NKT);
        ntile = a.dp_tiles + ts;
        nkb = (int)(sS[0] - (long long)ts * NKT);
      }
      if (ntile >= 0) {
        const int nrb = ntile / a.NTT, ntok0 = (ntile - nrb * a.NTT) * TT;
        const int nb = min(2, NKT - nkb) * REC;       // record bytes of k-tiles nkb, nkb + 1 (contiguous per row)
        for (int i = tid; i < 8 * 25; i += NTH) {
          const int p = i / 25, off = (i % 25) * 128;
          if (off < nb) prefetch_l2(a.R + ((size_t)(nrb * 8 + p) * NKT + nkb) * REC + off);
        }
        for (int i = tid; i < 2 * TT; i += NTH) {
          const int row = i >> 1, k = nkb + (i & 1);
          if (ntok0 + row < a.T && k < NKT) prefetch_l2((const char*)a.X + ((size_t)(ntok0 + row) * a.K + k * 64) * 2);
        }
      }
    }
    const int tile = sSeg[1], fkb = sSeg[2], fke = sSeg[3];   // segment again (reloaded after the barriers)
    const int frb = tile / a.NTT, ftok0 = (tile - frb * a.NTT) * TT;
    bool out = (fkb == 0 && fke == NKT);
    if (!out) {                                       // one of np pieces of a split tile (counter ts)
      // stream-K: CTAs cf..cl own consecutive k-ranges of SK tile ts, slot 2 cta + (first / last segment);
      // balanced split: piece j0 of a.split, workspace slot = piece index.
      int ts, np, myslot, j0 = 0, cf = 0;
      long long i0 = 0;
      if (a.split > 0) {
        const int u = sFlag[14];
        ts = tile;
        np = a.split;
        myslot = u;
        j0 = u - tile * a.split;
      } else {
        ts = tile - a.dp_tiles;
        i0 = (long long)ts * NKT;
        cf = (int)(((i0 + 1) * a.G - 1) / S);         // owners of i0 and i0 + NKT - 1
        const int cl = (int)(((i0 + NKT) * a.G - 1) / S);
        np = cl - cf + 1;
        myslot = cta * 2 + ((long long)cta * S / a.G >= i0 ? 0 : 1);
      }
      // The sum of the np pieces is the fixed chain 0 + 1 + ... + (np - 1) in k-range order: Y is bit-identical run to
      // run whichever piece arrives last. Piece 0 (the tile's first k-range) is usually last: under stream-K it is its
      // CTA's second segment. When it is last, it adds the others to its own accumulators (and never stores its own
      // partial if every other piece had already arrived); any other last piece runs the same chain from -0 over the
      // workspace, its own partial included.
      const int me = a.split > 0 ? j0 : cta - cf;     // this piece's index, in k-range order
      auto piece_slot = [&](int k) {                  // workspace slot of piece k (k < np)
        if (a.split > 0) return tile * a.split + k;
        const int q = cf + k;
        return q * 2 + ((long long)q * S / a.G >= i0 ? 0 : 1);
      };
      if (tid == 0) sFlag[0] = ld_acquire(a.cnt + ts);
      __syncthreads();
      bool last = sFlag[0] == np - 1;
      const bool root = me == 0;
      if (!last || !root) {                           // store the partial (the root that is last at once keeps it)
        float* p = a.ws + (size_t)myslot * NTH * PF;
#pragma unroll
        for (int m = 0; m < MW; ++m)
#pragma unroll
          for (int n = 0; n < NN; ++n)
            __stcg((float4*)(p + ((m * NN + n) * NTH + tid) * 4),
                   make_float4(acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3]));
        __threadfence();
      }
      if (!last) {
        __syncthreads();
        if (tid == 0) sFlag[0] = atomicAdd(a.cnt + ts, 1);
        __syncthreads();
        last = sFlag[0] == np - 1;
      }
      if (last) {                                     // every other piece is in the workspace
        __threadfence();
        if (!root) {
#pragma unroll
          for (int m = 0; m < MW; ++m)
#pragma unroll
            for (int n = 0; n < NN; ++n) acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = -0.f;
        }
        auto chain_slot = [&](int c) {                // c-th term of the chain after the start value
          return piece_slot(root ? c + 1 : c);
        };
        // other pieces' partials, in halves, through shared memory (stages + W buffers >= TT * 512 B): half h of
        // a partial lands in half h of the staging area while the other half is being added. Every thread copies
        // and reads back only its own fragments, so no barriers are needed and no registers hold loads in flight.
        constexpr int HJ = MW * NN / 2;               // float4 per thread per half
        const int np1 = root ? np - 1 : np;           // terms to add: 1..np-1 onto piece 0, or the whole chain
        auto issue = [&](int k, int h) {              // half h of the k-th other piece (empty group past the end)
          if (k < np1) {
            const float* pq = a.ws + (size_t)chain_slot(k) * NTH * PF;
#pragma unroll
            for (int jj = 0; jj < HJ; ++jj) {
              const int j = h * HJ + jj;
              cpa16(sX + (j * NTH + tid) * 16, pq + (j * NTH + tid) * 4);
            }
          }
          commit();
        };
        auto add = [&](auto hc) {                     // add staged half hc (compile-time) into the accumulators
          constexpr int h = decltype(hc)::value;
          asm volatile("" ::: "memory");
#pragma unroll
          for (int jj = 0; jj < HJ; ++jj) {
            const int j = h * HJ + jj;
            const float4 v = *(const float4*)(smem + (j * NTH + tid) * 16);
            acc[j / NN][j % NN][0] += v.x;
            acc[j / NN][j % NN][1] += v.y;
            acc[j / NN][j % NN][2] += v.z;
            acc[j / NN][j % NN][3] += v.w;
          }
          asm volatile("" ::: "memory");              // reads done before this half is refilled
        };
        issue(0, 0);
        issue(0, 1);
        for (int k = 0; k < np1; ++k) {
          waitg<1>();
          add(std::integral_constant<int, 0>{});
          issue(k + 1, 0);
          waitg<1>();
          add(std::integral_constant<int, 1>{});
          issue(k + 1, 1);
        }
        waitg<0>();
        if (tid == 0) a.cnt[ts] = 0;
        out = true;
        __syncthreads();                              // staging area -> epilogue tile
      }
    }
    if (out) {                                        // bf16 tile -> shared (16-B chunks swizzled by token) -> Y rows
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
        if (ftok0 + tok < a.T)
          *(uint4*)(a.Y + (size_t)(ftok0 + tok) * a.M + frb * RB + ch * 8) =
              *(const uint4*)(sC + tok * 256 + ((ch ^ (tok & 7)) << 4));
      }
    }
    __syncthreads();                                  // shared memory is reused by the next segment
  }
}

int num_sms() {                                     // per device: GPUs of a device map may differ
  static int n[64] = {0};
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  if (!n[dev & 63]) n[dev & 63] = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return n[dev & 63];
}
// schedule: (G, dp_tiles, sk_tiles, sk_iters). Stream-K (split 0) counts a tile's pieces as the contiguous range of
// CTAs that own its k-range, so every CTA must own >= 1 k-iteration: problems with fewer k-iterations than SMs run on
// that many CTAs.
std::vector<int64_t> sched(int64_t T, int64_t M, int64_t K, int64_t TT, int64_t split) {
  int64_t G = num_sms();
  const int64_t NTT = (T + TT - 1) / TT, NT = (M / RB) * NTT;
  const int64_t dp = NT >= 2 * G ? (NT / G - 1) * G : 0;
  if (split == 0 && dp == 0 && NT * (K / 64) < G) G = NT * (K / 64);
  return {G, dp, NT - dp, (NT - dp) * (K / 64)};
}
template <int TT, int NWT>
void launch(Args a) {
  static_assert(NST * TT * 128 + 2 * WB >= TT * 512, "tt: partial staging buffer");
  const int base = NST * TT * 128 + 2 * WB + 64 + a.KS * 4;
  a.esc_smem = base + a.CAP * 4 <= SMEM_MAX;          // rare escape-heavy matrices: entries read from global
  const int smem = base + (a.esc_smem ? a.CAP * 4 : 0);
  TORCH_CHECK(smem <= SMEM_MAX, "tt: shared memory ", smem, " too large");
  static bool set[64] = {false};                     // the attribute is per device (layers spread over GPUs)
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  if (!set[dev & 63]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(tt_kernel<TT, NWT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
    set[dev & 63] = true;
  }
  tt_kernel<TT, NWT><<<a.G, 256, smem, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#define TTLIST(X) X(96, 2) X(128, 2) X(160, 2) X(192, 2) X(224, 2) X(256, 2) X(128, 4) X(192, 4) X(256, 4)

void run(int64_t TT, int64_t NWT, int64_t split, torch::Tensor X, torch::Tensor R, torch::Tensor koff, torch::Tensor ent, torch::Tensor Y,
         torch::Tensor ws, torch::Tensor cnt, int64_t KS, int64_t CAP) {
  TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kBFloat16 && X.is_contiguous() && X.dim() == 2, "tt: X");
  TORCH_CHECK(Y.is_cuda() && Y.dtype() == torch::kBFloat16 && Y.is_contiguous() && Y.dim() == 2, "tt: Y");
  const int64_t T = X.size(0), K = X.size(1), M = Y.size(1);
  TORCH_CHECK(Y.size(0) == T && T > 0 && M % RB == 0 && K % 64 == 0, "tt: shapes");
  TORCH_CHECK(T * K * 2 < (1ll << 32), "tt: X must be < 4 GB (32-bit offsets)");
  TORCH_CHECK(R.dtype() == torch::kUInt8 && R.is_contiguous() && R.numel() == (M / 16) * (K / 64) * REC, "tt: records");
  TORCH_CHECK(KS % 4 == 0 && KS >= K / 64 + 1 && CAP % 4 == 0 && CAP >= 4, "tt: escape strides");
  TORCH_CHECK(koff.dtype() == torch::kInt32 && koff.is_contiguous() && koff.numel() == (M / RB) * KS, "tt: koff");
  TORCH_CHECK(ent.dtype() == torch::kInt32 && ent.is_contiguous() && ent.numel() == (M / RB) * CAP, "tt: ent");
  const auto s = sched(T, M, K, TT, split);
  const int64_t NT = (M / RB) * ((T + TT - 1) / TT);
  TORCH_CHECK(split >= 0 && split <= 8 && NT * split <= 2 * s[0], "tt: split ", split, " (NT ", NT, ")");
  TORCH_CHECK(ws.dtype() == torch::kFloat32 && ws.numel() >= s[0] * 2 * 128 * TT, "tt: workspace");
  TORCH_CHECK(cnt.dtype() == torch::kInt32 && cnt.numel() >= (split ? NT : s[2]), "tt: counters");
  Args a;
  a.X = (const __nv_bfloat16*)X.data_ptr();
  a.R = R.data_ptr<uint8_t>();
  a.koff = koff.data_ptr<int>();
  a.ent = ent.data_ptr<int>();
  a.Y = (__nv_bfloat16*)Y.data_ptr();
  a.ws = ws.data_ptr<float>();
  a.cnt = cnt.data_ptr<int>();
  a.T = (int)T; a.M = (int)M; a.K = (int)K; a.NTT = (int)((T + TT - 1) / TT); a.KS = (int)KS; a.CAP = (int)CAP;
  a.G = (int)s[0]; a.NT = (int)NT; a.split = (int)split;
  a.dp_tiles = split ? 0 : (int)s[1];
  a.sk_iters = split ? 0 : s[3];
#define TTC(T_, N_) if (TT == T_ && NWT == N_) { launch<T_, N_>(a); return; }
  TTLIST(TTC)
#undef TTC
  TORCH_CHECK(false, "tt: TT ", TT, " NWT ", NWT, " not compiled");
}
std::vector<int64_t> attrs(int64_t TT, int64_t NWT) {
  cudaFuncAttributes f;
#define TTA(T_, N_) if (TT == T_ && NWT == N_) { C10_CUDA_CHECK(cudaFuncGetAttributes(&f, tt_kernel<T_, N_>)); return {f.numRegs, (int64_t)f.localSizeBytes}; }
  TTLIST(TTA)
#undef TTA
  return {-1, -1};
}
}  // namespace

// entry points (registered in bindings.cpp)
void tt_run(int64_t TT, int64_t NWT, int64_t split, torch::Tensor X, torch::Tensor R, torch::Tensor koff,
            torch::Tensor ent, torch::Tensor Y, torch::Tensor ws, torch::Tensor cnt, int64_t KS, int64_t CAP) {
  const at::cuda::OptionalCUDAGuard guard(X.device());
  run(TT, NWT, split, X, R, koff, ent, Y, ws, cnt, KS, CAP);
}
std::vector<int64_t> tt_attrs(int64_t TT, int64_t NWT) { return attrs(TT, NWT); }
