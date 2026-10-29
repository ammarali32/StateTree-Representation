// statetree Hopper kernel "th" (sm_90a): fused tree decode + wgmma GEMM for H100 / H200, any batch.
//
// Per CTA (one per SM, persistent, stream-K over units = (128 R-row block, 64-column group)):
//   warp 8 (producer)  : one thread issues the TMA copies of a unit into a ring of ST stages: the 8 R records of the
//                        row block (one tensor-map copy: the tree seen as [tiles][groups][194 x u64]), its 64-byte
//                        escape block, and the activation tile [NTOK tokens x 64] (tensor-map copy, 128-byte swizzle =
//                        the wgmma K-major B layout; rows past the batch are zero-filled by the TMA unit).
//   warps 0-7 (2 warpgroups, consumers): warp w decodes records w, 8 + w, .. (R of them) straight into the register A
//                        fragments of wgmma.m64nNk16 (a record is stored in the mma A-fragment order, which is also
//                        wgmma's register-A layout), patches their escapes, and its warpgroup multiplies each 64-row
//                        slice by the activation tile read from shared memory by the tensor cores.
// R = 2 (small token tiles): two independent decode chains per warp and unit hide the latency of the per-unit chain
// (barrier wait, shared loads, decode, wgmma issue) that bounds R = 1 at small batch.
// Row blocks split between CTAs are merged by the last CTA in a fixed order (deterministic). Exact BF16 weights; only
// the fp32 summation order differs from cuBLAS. Compiled only for sm_90a; th_ok() reports whether it is available.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <map>
#include <tuple>

#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define TH_ON 1
#else
#define TH_ON 0
#endif

namespace th {

constexpr int REC = 1552;                  // one 16 x 64 record
constexpr int NCW = 256;                   // consumer threads

struct Args {
  const uint8_t* T; const uint32_t* EB; const int* EOU; const uint32_t* EALL;
  __nv_bfloat16* Y; float* WS; int* CNT;
  int M, N, B, nRB, flags;                 // flags: diagnostics, honoured only in builds with TH_DIAG
};

__host__ __device__ constexpr int rstage(int R) {          // records + escape block, padded to 128 B (TMA tensor)
  return (8 * R * REC + 64 + 127) / 128 * 128;
}
__host__ __device__ constexpr int dtile(int W) { return W == 2 ? 2 * 16384 : 0; }   // W = 2: 2 decoded 128 x 64 tiles
__host__ __device__ constexpr int stages(int ntok, int R, int W) {
  return (232448 - 2048 - dtile(W)) / (ntok * 128 + rstage(R)) < 12 ? (232448 - 2048 - dtile(W)) / (ntok * 128 + rstage(R))
                                                                     : 12;
}
__host__ __device__ constexpr size_t smem_bytes(int ntok, int R, int W) {
  return (size_t)stages(ntok, R, W) * (ntok * 128 + rstage(R)) + dtile(W) + 1024 + 2 * stages(ntok, R, W) * 8 + 32 + 16;
}

template <int N> struct WG;
template <int N> struct WGS;

#if TH_ON
template <> struct WG<8> {
  static __device__ __forceinline__ void mma(float* d, const uint32_t* a, uint64_t desc, int sd) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(sd));
  }
};
template <> struct WG<16> {
  static __device__ __forceinline__ void mma(float* d, const uint32_t* a, uint64_t desc, int sd) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(sd));
  }
};
template <> struct WG<32> {
  static __device__ __forceinline__ void mma(float* d, const uint32_t* a, uint64_t desc, int sd) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(sd));
  }
};
template <> struct WG<64> {
  static __device__ __forceinline__ void mma(float* d, const uint32_t* a, uint64_t desc, int sd) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(sd));
  }
};
template <> struct WG<128> {
  static __device__ __forceinline__ void mma(float* d, const uint32_t* a, uint64_t desc, int sd) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(sd));
  }
};
template <> struct WG<256> {
  static __device__ __forceinline__ void mma(float* d, const uint32_t* a, uint64_t desc, int sd) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %133, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, {%128,%129,%130,%131}, %132, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(sd));
  }
};
template <> struct WGS<128> {
  static __device__ __forceinline__ void mma(float* d, uint64_t da, uint64_t db, int sd) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1, 0, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(da), "l"(db), "r"(sd));
  }
};

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(su32(b)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(su32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive_if(uint64_t* b, bool p) {
  asm volatile("{\n.reg .pred q;\nsetp.ne.b32 q, %1, 0;\n@q mbarrier.arrive.shared::cta.b64 _, [%0];\n}\n"
               :: "r"(su32(b)), "r"((int)p) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE;\n"
      "bra LAB_WAIT;\n"
      "DONE:\n"
      "}\n" :: "r"(su32(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(su32(dst)), "l"(src), "r"(bytes), "r"(su32(bar)) : "memory");
}
__device__ __forceinline__ void tma3d_ef(void* dst, const CUtensorMap* map, int c0, int c1, int c2, uint64_t* bar,
                                         uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%2, %3, %4}], [%5], %6;\n"
               :: "r"(su32(dst)), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(su32(bar)), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(su32(dst)), "l"(map), "r"(c0), "r"(c1), "r"(su32(bar)) : "memory");
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N_> __device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N_) : "memory");
}
__device__ __forceinline__ void cbar() { asm volatile("bar.sync 1, 256;\n" ::: "memory"); }
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
__device__ __forceinline__ void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n"
               :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void red_release_add(int* p, int v) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
template <int NA> __device__ __forceinline__ void fence_acc(float* acc) {
#pragma unroll
  for (int i = 0; i < NA; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");
}
// wgmma shared-memory descriptor: K-major tile with 128-byte swizzle, 8-row groups 1024 B apart
__device__ __forceinline__ uint64_t desc_sw128(uint32_t saddr) {
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}

// --- the record decode (as in kernels.cu: dec4 / dec_step5)
__device__ __forceinline__ uint32_t wsel(const uint4& v, int s) {
  return s == 0 ? v.x : (s == 1 ? v.y : (s == 2 ? v.z : v.w));
}
__device__ __forceinline__ void dec4(uint32_t lf, uint32_t d, uint32_t E4, uint32_t& p01, uint32_t& p23) {
  const uint32_t e4 = E4 - d;
  const uint32_t a = __byte_perm(lf, e4, 0x5140), b = __byte_perm(lf, e4, 0x7362);
  p01 = __funnelshift_r(a, a, 1);
  p23 = __funnelshift_r(b, b, 1);
}
__device__ __forceinline__ void dec_step5(const uint4& c0, const uint4& c1, const uint4& c2, uint32_t E0, uint32_t E1,
                                          int s, uint32_t* A) {
  const uint4& c = (s < 2) ? c0 : c1;
  const uint32_t lw0 = (s & 1) ? c.z : c.x, lw1 = (s & 1) ? c.w : c.y;
  const uint32_t nw = wsel(c2, s);
  dec4(lw0, nw & 0x0F0F0F0Fu, E0, A[0], A[2]);
  dec4(lw1, (nw >> 4) & 0x0F0F0F0Fu, E1, A[1], A[3]);
}
// Escapes of warp w's records (record i = 8 j + w, slot j): entries bits << 16 | i << 10 | lane << 5 | k-step << 3 |
// reg << 1 | half. One load per 32 entries, a ballot for this warp's ones, then for each: every lane replaces the 16-bit
// half at (slot, k-step, reg) if the entry is its own (register selects: no indexing, no divergence).
template <int R>
__device__ __forceinline__ void patch_escapes(uint32_t (&A)[R][4][4], const uint32_t* el, int nesc, int w, int lane) {
#pragma unroll 1
  for (int base = 0; base < nesc; base += 32) {
    const int e = base + lane;
    const uint32_t v = (e < nesc) ? el[e] : 0u;
    uint32_t mask = __ballot_sync(0xFFFFFFFFu, e < nesc && (int)((v >> 10) & 7u) == w);
#pragma unroll 1
    while (mask) {
      const int b = __ffs(mask) - 1;
      mask &= mask - 1;
      const uint32_t vv = __shfl_sync(0xFFFFFFFFu, v, b);
      const int f = ((int)((vv >> 5) & 31u) == lane) ? (int)(((vv >> 13) & 3u) * 16 + ((vv >> 3) & 3u) * 4 + ((vv >> 1) & 3u)) : -1;
      const uint32_t sh = (vv & 1u) * 16u, keep = ~(0xFFFFu << sh), val = (vv >> 16) << sh;
#pragma unroll
      for (int jj = 0; jj < R; ++jj)
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq)
            A[jj][kk][qq] = (f == jj * 16 + kk * 4 + qq) ? ((A[jj][kk][qq] & keep) | val) : A[jj][kk][qq];
    }
  }
}
__device__ __forceinline__ int cta_of(int u, int U, int C) {
  return (int)(((long long)(u + 1) * C + U - 1) / U) - 1;
}
#endif  // TH_ON

// threads: 2 consumer warpgroups + the producer warp; NTOK = 256 (128 accumulators per thread) also takes a whole
// producer warpgroup, whose registers it hands to the consumers (setmaxnreg: 40 producer, 232 consumer)
__host__ __device__ constexpr int threads(int ntok) { return ntok == 256 ? 384 : 288; }

template <int NTOK, int R, int W>
__global__ void __launch_bounds__(threads(NTOK), 1) th_kernel(const __grid_constant__ CUtensorMap xmap,
                                                              const __grid_constant__ CUtensorMap tmap, const Args a) {
#if TH_ON
  constexpr int RT = 8 * R;                         // records per unit (128 R rows)
  constexpr int RS_ = rstage(R);
  constexpr int ST = stages(NTOK, R, W);
  constexpr int XT = NTOK * 128;                    // activation tile bytes (NTOK rows of 128 B)
  constexpr int NA = NTOK / 2;                      // accumulators per consumer thread and slot
  constexpr int PE = NCW * NA * R;                  // floats of one partial tile
  // W = 0: mma.sync per warp (B fragments by ldmatrix from the swizzled activation tile): cheap to issue, warps
  // independent (an escape patch delays one warp, not its warpgroup), stage released right after it is read.
  // W = 1: wgmma per warpgroup (activations read once per 64 rows; large token tiles).
  // W = 2 (token tiles 128 / 256): transposed: every warp stores its decoded record to a 128 x 64 bf16 tile in shared
  // memory (stmatrix, 128-byte swizzle), and each warpgroup multiplies its 64 / 128 tokens (A, the activation tile) by
  // all 128 rows (B, the decoded tile) with wgmma m64n128k16: twice the work per instruction of W = 1.
  constexpr bool WGM = W == 1;
  constexpr int MT = NTOK / 128;                    // W = 2: 64-token tiles per warpgroup
  // independent accumulator sets (k-step k accumulates into set k % NS): at small token tiles a wgmma is short and
  // its latency, not the tensor pipe, bounds a single accumulation chain; the sets are summed in a fixed order
  constexpr int NS = R == 1 ? (NTOK <= 32 ? 4 : (NTOK == 64 ? 2 : 1)) : (NTOK <= 8 ? 4 : (NTOK == 16 ? 2 : 1));
  extern __shared__ uint8_t smraw[];
  uint8_t* sm = smraw + ((1024 - (su32(smraw) & 1023u)) & 1023u);     // 1024-aligned: the swizzled tiles need it
  uint8_t* xs = sm;                                 // [ST][XT]
  uint8_t* dts = sm + ST * XT;                      // W = 2: [2][128 rows][128 B] decoded tiles (1024-aligned)
  uint8_t* rs = dts + dtile(W);                     // [ST][RS_]
  uint64_t* full = (uint64_t*)(rs + ST * RS_);
  uint64_t* empty = full + ST;
  uint64_t* sready = empty + ST;                    // W = 2: decoded tile q written (8 warps)
  uint64_t* sfree = sready + 2;                     // W = 2: decoded tile q consumed (2 warpgroups)
  int* s_flag = (int*)(sfree + 2);

  // warp index through a shuffle: provably warp-uniform for ptxas (as CUTLASS does); with tid >> 5 the role branch
  // below counts as divergent and every wgmma is serialized (ptxas C7518)
  const int tid = threadIdx.x, warp = __shfl_sync(0xFFFFFFFFu, tid >> 5, 0), lane = tid & 31;
  const int NG = a.N >> 6, U = a.nRB * NG;
  const int C = gridDim.x, cta = blockIdx.x, tt = blockIdx.y;
  const int u0 = (int)((long long)U * cta / C), u1 = (int)((long long)U * (cta + 1) / C);
  const int nIt = u1 - u0;
  if (tid == 0) {
#pragma unroll 1
    for (int s = 0; s < ST; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], W == 2 ? 10 : 8); }
    for (int q = 0; q < 2; ++q) { mbar_init(&sready[q], 8); mbar_init(&sfree[q], 2); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (nIt <= 0) return;

  if (warp >= 8) {                                  // ------------------------------------------------ producer
    if constexpr (NTOK == 256) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    if (warp == 8 && lane == 0) {
      asm volatile("prefetch.tensormap [%0];\n" :: "l"(&xmap) : "memory");
      asm volatile("prefetch.tensormap [%0];\n" :: "l"(&tmap) : "memory");
      uint64_t pol;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
      int rb = u0 / NG, G = u0 - rb * NG;
#pragma unroll 1
      for (int it = 0; it < nIt; ++it) {
        const int s = it % ST;
        if (it >= ST) mbar_wait(&empty[s], ((it / ST) + 1) & 1);
        mbar_expect_tx(&full[s], RT * REC + 64 + XT);
        uint8_t* rd = rs + s * RS_;
        tma3d_ef(rd, &tmap, 0, G, rb * RT, &full[s], pol);       // the unit's RT records, back to back
        bulk_g2s(rd + RT * REC, a.EB + (size_t)(rb * NG + G) * 16, 64, &full[s]);
        tma2d(xs + s * XT, &xmap, G * 64, tt * NTOK, &full[s]);
        if (++G == NG) { G = 0; ++rb; }
      }
    }
    return;
  }

  // ------------------------------------------------------------------------------------------------- consumers
  if constexpr (NTOK == 256) asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
  const int w = warp, wg = warp >> 2;
  const int g = lane >> 2, t = lane & 3;
  float acc[R][NA], accx[R][NS > 1 ? NS - 1 : 1][NA];
#pragma unroll
  for (int j = 0; j < R; ++j) {
#pragma unroll
    for (int i = 0; i < NA; ++i) acc[j][i] = 0.f;
#pragma unroll
    for (int q = 0; q < NS - 1; ++q)
#pragma unroll
      for (int i = 0; i < NA; ++i) accx[j][q][i] = 0.f;
  }
  const int first_rb = u0 / NG;                     // the first row block of this CTA's units
  const uint32_t xs0 = su32(xs);
#ifdef TH_PROF
  long long tp[8] = {0, 0, 0, 0, 0, 0, 0, 0};       // cycles: wait full, loads + decode, escapes, wgmma issue, wait<1>,
  long long tq = clock64();                          // segment end + epilogue, units, total
#define TP(i) { const long long _t = clock64(); tp[i] += _t - tq; tq = _t; }
#else
#define TP(i)
#endif
  auto fence_all = [&]() {
#pragma unroll
    for (int j = 0; j < R; ++j) {
      fence_acc<NA>(acc[j]);
#pragma unroll
      for (int q = 0; q < NS - 1; ++q) fence_acc<NA>(accx[j][q]);
    }
  };

  // A row block split between CTAs cf..cl (contiguous unit ranges): cf processes it at the end of its range, the
  // others at the start of theirs. Those write their partial sums (slot per CTA) and signal with a release add;
  // cf waits for the P - 1 signals, adds the partials in CTA order (deterministic) and writes the output.
  // Requires all CTAs resident at once (C * token tiles <= SMs: checked on the host).
  auto finish = [&](int rbx, bool full_) {
    if (!full_) {
      const int tile_id = tt * a.nRB + rbx;
      const int cf = cta_of(rbx * NG, U, C), cl = cta_of(rbx * NG + NG - 1, U, C);
      if (cta != cf) {
        float4* wsl = (float4*)(a.WS + (size_t)(tt * C + cta) * PE) + tid;
#pragma unroll
        for (int j = 0; j < R; ++j)
#pragma unroll
          for (int i = 0; i < NA / 4; ++i)
            wsl[(j * (NA / 4) + i) * NCW] = make_float4(acc[j][4 * i], acc[j][4 * i + 1], acc[j][4 * i + 2], acc[j][4 * i + 3]);
        cbar();
        if (tid == 0) red_release_add(&a.CNT[tile_id], 1);
        return;
      }
      if (tid == 0) {
        while (ld_acquire(&a.CNT[tile_id]) != cl - cf) __nanosleep(64);
        a.CNT[tile_id] = 0;                         // every signal has arrived: reset for the next launch
      }
      cbar();
#pragma unroll 1
      for (int q = cf + 1; q <= cl; ++q) {          // fixed order: own (cf) + cf + 1 + ... + cl
        const float4* wq = (const float4*)(a.WS + (size_t)(tt * C + q) * PE) + tid;
#pragma unroll
        for (int j = 0; j < R; ++j)
#pragma unroll
          for (int i = 0; i < NA / 4; ++i) {
            const float4 v = __ldcg(wq + (j * (NA / 4) + i) * NCW);
            acc[j][4 * i] += v.x; acc[j][4 * i + 1] += v.y; acc[j][4 * i + 2] += v.z; acc[j][4 * i + 3] += v.w;
          }
      }
    }
    if constexpr (W == 2) {                         // D = tokens x rows: (token 16 wi + g (+8), rows 8 i + 2 t, +1)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int tok = tt * NTOK + wg * (NTOK / 2) + mt * 64 + (w & 3) * 16 + g;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const int row = rbx * 128 + 8 * i + 2 * t;
          const float* v = &acc[0][mt * 64 + 4 * i];
          if (tok < a.B) *(__nv_bfloat162*)(a.Y + (size_t)tok * a.M + row) = __floats2bfloat162_rn(v[0], v[1]);
          if (tok + 8 < a.B) *(__nv_bfloat162*)(a.Y + (size_t)(tok + 8) * a.M + row) = __floats2bfloat162_rn(v[2], v[3]);
        }
      }
      return;
    }
    const int odd = g & 1;
#pragma unroll
    for (int j = 0; j < R; ++j) {
      const int row0 = rbx * 128 * R + j * 128 + wg * 64 + (w & 3) * 16 + g - odd;
#pragma unroll
      for (int i = 0; i < NTOK / 8; ++i) {
        const int tok = tt * NTOK + 8 * i + 2 * t + odd;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float v0 = acc[j][4 * i + 2 * h], v1 = acc[j][4 * i + 2 * h + 1];
          const float recv = __shfl_xor_sync(0xFFFFFFFFu, odd ? v0 : v1, 4);
          const __nv_bfloat162 pr = __floats2bfloat162_rn(odd ? recv : v0, odd ? v1 : recv);
          if (tok < a.B) *(__nv_bfloat162*)(a.Y + (size_t)tok * a.M + row0 + 8 * h) = pr;
        }
      }
    }
  };

  // one unit: wait for its stage, decode records w, 8 + w, .. into A, patch escapes, R x 4 wgmma (slots x k-steps) on
  // the activation tile, then wait for the previous unit's products and release that unit's stage
  const uint32_t dt0 = su32(dts);
  // W = 2 unit: decode record w, release the records, wait until tile q is free (the products of unit it - 2 are done),
  // store the record's 16 rows into it, signal; then each warpgroup multiplies its tokens by the 128 rows of tile q and,
  // once the previous unit's products are done, frees that unit's tile and activation stage
  auto unit2 = [&](int it, int u, uint32_t (&A)[R][4][4], int first) {
    const int s = it % ST, q = it & 1;
    mbar_wait(&full[s], (it / ST) & 1);
    const uint8_t* stg = rs + s * RS_;
    const uint32_t* esc = (const uint32_t*)(stg + RT * REC);
    const uint8_t* rp = stg + w * REC;
    const uint4 c0 = *(const uint4*)(rp + lane * 16);
    const uint4 c1 = *(const uint4*)(rp + 512 + lane * 16);
    const uint4 c2 = *(const uint4*)(rp + 1024 + lane * 16);
    const uint32_t ew = *(const uint16_t*)(rp + 1536 + g * 2);
    const uint32_t hm = (esc[0] >> (4 * w)) & 15u;
    const int nesc = (int)esc[2];
    const uint32_t E0 = (ew & 0xFFu) * 0x01010101u, E1 = (ew >> 8) * 0x01010101u;
#pragma unroll
    for (int k = 0; k < 4; ++k) dec_step5(c0, c1, c2, E0, E1, k, A[0][k]);
    if (hm) patch_escapes<R>(A, (nesc > 13) ? a.EALL + __ldg(a.EOU + u) : esc + 3, nesc, w, lane);
    mbar_arrive_if(&empty[s], lane == 0);           // records read (the activation tile is released below)
    if (it >= 2) mbar_wait(&sfree[q], ((it >> 1) - 1) & 1);
    const uint32_t dq = dt0 + q * 16384;
    {
      const int m = lane >> 3, r = 16 * w + (lane & 7) + 8 * (m & 1);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int c = 2 * k + (m >> 1);
        stsm_x4(dq + r * 128 + ((c ^ (lane & 7)) << 4), A[0][k][0], A[0][k][1], A[0][k][2], A[0][k][3]);
      }
    }
    fence_proxy_async();
    __syncwarp();
    mbar_arrive_if(&sready[q], lane == 0);
    mbar_wait(&sready[q], (it >> 1) & 1);
    fence_all();
    wg_fence();
    const uint64_t db = desc_sw128(dq);
    const uint32_t xa = xs0 + s * XT + wg * (NTOK / 2) * 128;
#pragma unroll
    for (int k = 0; k < 4; ++k)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
        WGS<128>::mma(&acc[0][mt * 64], desc_sw128(xa + mt * 64 * 128) + 2 * k, db + 2 * k, (first && k == 0) ? 0 : 1);
    wg_commit();
    wg_wait<1>();
    fence_all();
    const bool rel = it > 0 && (w & 3) == 0 && lane == 0;
    mbar_arrive_if(&sfree[(it + 1) & 1], rel);      // unit it - 1: its tile ...
    mbar_arrive_if(&empty[(it + ST - 1) % ST], rel); // ... and its activation stage
  };

  auto unit = [&](int it, int u, uint32_t (&A)[R][4][4], int first) {
    if constexpr (W == 2) {
      unit2(it, u, A, first);
    } else {
    const int s = it % ST;
    TP(7);
    mbar_wait(&full[s], (it / ST) & 1);
    TP(0);
    const uint8_t* stg = rs + s * RS_;
    const uint32_t* esc = (const uint32_t*)(stg + RT * REC);
#ifdef TH_DIAG
    if (a.flags & 1) {                              // diagnostics: stream only
      mbar_arrive_if(&empty[(it + ST - 1) % ST], it > 0 && lane == 0);
      return;
    }
#endif
    uint4 c0[R], c1[R], c2[R];
    uint32_t ew[R];
#pragma unroll
    for (int j = 0; j < R; ++j) {
      const uint8_t* rp = stg + (j * 8 + w) * REC;
      c0[j] = *(const uint4*)(rp + lane * 16);
      c1[j] = *(const uint4*)(rp + 512 + lane * 16);
      c2[j] = *(const uint4*)(rp + 1024 + lane * 16);
      ew[j] = *(const uint16_t*)(rp + 1536 + g * 2);
    }
    uint32_t hm = 0;                                // escape k-steps of this warp's records (bit 4 j + k)
#pragma unroll
    for (int j = 0; j < R; ++j) hm |= ((esc[j] >> (4 * w)) & 15u) << (4 * j);
#pragma unroll
    for (int j = 0; j < R; ++j) {
      const uint32_t E0 = (ew[j] & 0xFFu) * 0x01010101u, E1 = (ew[j] >> 8) * 0x01010101u;
#pragma unroll
      for (int k = 0; k < 4; ++k) dec_step5(c0[j], c1[j], c2[j], E0, E1, k, A[j][k]);
    }
#ifdef TH_PROF
    { uint32_t z = 0;
#pragma unroll
      for (int j = 0; j < R; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k) z ^= A[j][k][0] ^ A[j][k][3];
      asm volatile("" :: "r"(z)); }
#endif
    TP(1);
    if (hm) {
      const int nesc = (int)esc[2];
      patch_escapes<R>(A, (nesc > 13) ? a.EALL + __ldg(a.EOU + u) : esc + 3, nesc, w, lane);
    }
    TP(2);
    if constexpr (WGM) {
      fence_all();
      wg_fence();
      const uint64_t d0 = desc_sw128(xs0 + s * XT);
#pragma unroll
      for (int j = 0; j < R; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int sd = (first && k < NS) ? 0 : 1; // the first product of each set in a segment starts it
          if (k % NS == 0) WG<NTOK>::mma(acc[j], A[j][k], d0 + 2 * k, sd);
          else WG<NTOK>::mma(accx[j][(k % NS) - 1], A[j][k], d0 + 2 * k, sd);
        }
      wg_commit();
      TP(3);
      wg_wait<1>();
      fence_all();
      TP(4);
      mbar_arrive_if(&empty[(it + ST - 1) % ST], it > 0 && lane == 0);
    } else {
      // B fragments of the swizzled tile: row n = token, 16-byte chunk c (k 8c .. 8c + 7) stored at chunk c ^ (n & 7);
      // ldmatrix.x4 per 8-token tile and pair of k-steps: lanes 0-7 / 8-15 / 16-23 / 24-31 = (k-step, half)
      uint32_t xb[NTOK / 8][8];
      const uint32_t xt = xs0 + s * XT;
#pragma unroll
      for (int nt = 0; nt < NTOK / 8; ++nt)
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int n = nt * 8 + (lane & 7), c = 4 * p + (lane >> 3);
          ldsm_x4(xt + n * 128 + ((c ^ (n & 7)) << 4), xb[nt][4 * p], xb[nt][4 * p + 1], xb[nt][4 * p + 2],
                  xb[nt][4 * p + 3]);
        }
      mbar_arrive_if(&empty[s], lane == 0);         // this warp is done with the stage (records and tile in registers)
      TP(3);
#pragma unroll
      for (int j = 0; j < R; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k)
#pragma unroll
          for (int nt = 0; nt < NTOK / 8; ++nt) {
            float* c = (k % NS == 0) ? &acc[j][4 * nt] : &accx[j][(k % NS) - 1][4 * nt];
            mma16816(c, A[j][k], xb[nt][2 * k], xb[nt][2 * k + 1]);
          }
      TP(4);
    }
#ifdef TH_PROF
    tp[6] += 1;
#endif
    }
  };

  uint32_t A0[R][4][4], A1[R][4][4];
  int it = 0, rb = first_rb, G = u0 - first_rb * NG;
#pragma unroll 1
  while (it < nIt) {                                // one row-block segment per pass: units G .. G + n - 1
    const int n = min(NG - G, nIt - it);
    const bool full_seg = (G == 0 && n == NG);
    const int ub = rb * NG + G;
    int k = 0;
    if (n & 1) { unit(it, ub, A0, 1); k = 1; }      // odd count: one unit first, then pairs (static A buffers)
#pragma unroll 1
    for (; k < n; k += 2) {
      unit(it + k, ub + k, A1, k == 0);
      unit(it + k + 1, ub + k + 1, A0, 0);
    }
    TP(7);
    if constexpr (W != 0) {
      wg_wait<0>();
      fence_all();
    }
#pragma unroll
    for (int j = 0; j < R; ++j)
#pragma unroll
      for (int q = 0; q < NS - 1; ++q)              // fixed order: set 0 + set 1 + ... (deterministic)
#pragma unroll
        for (int i = 0; i < NA; ++i) acc[j][i] += accx[j][q][i];
    finish(rb, full_seg);
    if constexpr (W == 0) {                         // mma.sync accumulates: start the next segment from zero
#pragma unroll
      for (int j = 0; j < R; ++j) {
#pragma unroll
        for (int i = 0; i < NA; ++i) acc[j][i] = 0.f;
#pragma unroll
        for (int q = 0; q < NS - 1; ++q)
#pragma unroll
          for (int i = 0; i < NA; ++i) accx[j][q][i] = 0.f;
      }
    }
    TP(5);
    it += n; G = 0; ++rb;
  }
#ifdef TH_PROF
  if (lane == 0 && cta < 4 && tt == 0) {             // per warp of CTAs 0-3: 8 counters (cycles / 64, units)
#pragma unroll
    for (int i = 0; i < 8; ++i) a.CNT[60000 + (cta * 8 + w) * 8 + i] = (int)(i == 6 ? tp[i] : tp[i] >> 6);
  }
#endif
#endif
}

__global__ void th_probe(int* f) { *f = TH_ON; }

// ------------------------------------------------------------------------------------------------- mvt (batch <= 8)
// Matrix-vector products as a TMA sweep over the records in memory order. CTA c (C of them, all resident: 1-3 per SM)
// owns the chunks [c U / C, (c + 1) U / C) of 8 consecutive records (12416 B; a tile = 16 rows x N is N / 512 chunks).
// Warp 8 issues one bulk copy per chunk (records + the chunk's 64-byte escape block) into an S-stage ring; consumer warp
// w takes record w of every chunk: decode from shared memory, escape patch, 2 ldmatrix of the staged activations (rows
// < B, row addresses clamped), 4 mma into its tile accumulator. When the sweep crosses a tile boundary the 8 warps park
// their partial tiles, and one warp sums them in a fixed order and stores the tile (or, for a tile shared with the
// neighbouring CTA, writes its part and the last of the CTAs sums the parts in order): deterministic.
// Why: on the H100 the mv kernels' ~2000 per-warp streams 25-100 KB apart cost DRAM row locality (15% in short bursts,
// 35% under a power cap, where the SM clock then drops further); a sweep with few contiguous streams reads at the floor,
// and the in-flight bytes live in shared memory, not in registers, so they do not depend on occupancy.
struct VArgs {
  const uint8_t* T; const uint32_t* EB; const int* EOU; const uint32_t* EALL;
  const __nv_bfloat16* X; __nv_bfloat16* Y; float* WS; int* CNT;
  int M, N, B, U, C;                       // U chunks, C CTAs
};
constexpr int CHB = 8 * REC;               // one chunk of records (12416 B)
constexpr int VSTG = CHB + 64;             // + its escape block
__host__ __device__ constexpr size_t mvt_smem(int S, int B, int N) {
  return 128 + (size_t)S * VSTG + (size_t)B * (N + 8) * 2 + 2 * 8 * 128 * 4 + 2 * S * 8;
}

template <int S>
__global__ void __launch_bounds__(288, 2) mvt_kernel(const VArgs a) {
#if TH_ON
  extern __shared__ uint8_t smraw[];
  uint8_t* ring = smraw + ((128 - (su32(smraw) & 127u)) & 127u);      // [S][VSTG]
  uint16_t* xs = (uint16_t*)(ring + S * VSTG);                        // [B][N + 8]
  const int XROW = a.N + 8;
  float* red = (float*)(xs + a.B * XROW);                             // [2][8][128]: parked partial tiles
  uint64_t* full = (uint64_t*)(red + 2 * 8 * 128);
  uint64_t* empty = full + S;
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = __shfl_sync(0xFFFFFFFFu, tid >> 5, 0);
  const int cpt = a.N / 512;                                          // chunks per tile
  const int c = blockIdx.x;
  const int k0 = (int)((long long)c * a.U / a.C), k1 = (int)((long long)(c + 1) * a.U / a.C);
  const int nk = k1 - k0;
  if (tid == 0) {
#pragma unroll 1
    for (int s = 0; s < S; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], 8); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (nk <= 0) return;
  if (warp == 8) {                                                    // producer: weights need no wait (PDL)
    if (lane == 0) {
      uint64_t pol;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
#pragma unroll 1
      for (int j = 0; j < nk; ++j) {
        const int s = j % S;
        if (j >= S) mbar_wait(&empty[s], ((j / S) - 1) & 1);
        mbar_expect_tx(&full[s], VSTG);
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
                     :: "r"(su32(ring + s * VSTG)), "l"(a.T + (size_t)(k0 + j) * CHB), "r"(CHB), "r"(su32(&full[s])), "l"(pol)
                     : "memory");
        bulk_g2s(ring + s * VSTG + CHB, a.EB + (size_t)(k0 + j) * 16, 64, &full[s]);
      }
    }
    return;
  }
  // ---------------------------------------------------------------------------------------------- consumers
  asm volatile("griddepcontrol.wait;\n" ::: "memory");               // X / WS / CNT after the previous kernel
  {
    const int ch = a.N / 8;
    for (int i = tid; i < a.B * ch; i += NCW) {
      const int r = i / ch, q = i - r * ch;
      *(uint4*)(xs + r * XROW + q * 8) = *(const uint4*)(a.X + (size_t)r * a.N + q * 8);
    }
    cbar();
  }
  const int g = lane >> 2, t = lane & 3;
  const int xrow = min(lane & 7, a.B - 1);
  const uint32_t xa = su32(xs) + (uint32_t)((xrow * XROW + ((lane >> 4) & 1) * 16 + ((lane >> 3) & 1) * 8 + warp * 64) * 2);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  int tile = k0 / cpt, kt = k0 - tile * cpt;                          // current tile, chunk within it
  int nfl = 0;
  auto cta_of_chunk = [&](int k) { return (int)(((long long)(k + 1) * a.C + a.U - 1) / a.U) - 1; };
  auto flush = [&]() {                                                // the sweep leaves this tile
    float* rw = red + ((nfl & 1) * 8 + warp) * 128;
    rw[(2 * t) * 16 + g] = acc[0];
    rw[(2 * t + 1) * 16 + g] = acc[1];
    rw[(2 * t) * 16 + g + 8] = acc[2];
    rw[(2 * t + 1) * 16 + g + 8] = acc[3];
    acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
    cbar();
    if (warp == (nfl & 7)) {                                          // one warp sums the 8 partials (fixed order)
      const float* rs = red + (nfl & 1) * 8 * 128;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float sum = 0.f;
#pragma unroll
        for (int w8 = 0; w8 < 8; ++w8) sum += rs[w8 * 128 + lane + 32 * e];
        v[e] = sum;
      }
      const int ka = tile * cpt, kb = ka + cpt - 1;
      bool store = true;
      if (ka < k0 || kb >= k1) {                                      // shared with a neighbour: merge the parts
        const int ca = cta_of_chunk(ka), np = cta_of_chunk(kb) - ca + 1;
        float* part = a.WS + ((size_t)tile * 4 + (c - ca)) * 128;
#pragma unroll
        for (int e = 0; e < 4; ++e) part[lane + 32 * e] = v[e];
        __threadfence();
        __syncwarp();
        int last = 0;
        if (lane == 0) last = (atomicAdd(&a.CNT[tile], 1) == np - 1);
        last = __shfl_sync(0xFFFFFFFFu, last, 0);
        store = last != 0;
        if (last) {
          __threadfence();
          const float* p0 = a.WS + (size_t)tile * 4 * 128;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            float sum = 0.f;
            for (int p = 0; p < np; ++p) sum += __ldcg(p0 + p * 128 + lane + 32 * e);
            v[e] = sum;
          }
          if (lane == 0) a.CNT[tile] = 0;
        }
      }
      if (store) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int idx = lane + 32 * e, nn = idx >> 4, r = idx & 15;
          if (nn < a.B) a.Y[(size_t)nn * a.M + (size_t)tile * 16 + r] = __float2bfloat16_rn(v[e]);
        }
      }
    }
    ++nfl;
  };
#pragma unroll 1
  for (int j = 0; j < nk; ++j) {
    const int s = j % S;
    mbar_wait(&full[s], (j / S) & 1);
    const uint8_t* rp = ring + s * VSTG + warp * REC;
    const uint32_t* esc = (const uint32_t*)(ring + s * VSTG + CHB);
    const uint4 c0 = *(const uint4*)(rp + lane * 16);
    const uint4 c1 = *(const uint4*)(rp + 512 + lane * 16);
    const uint4 c2 = *(const uint4*)(rp + 1024 + lane * 16);
    const uint32_t ew = *(const uint16_t*)(rp + 1536 + g * 2);
    const uint32_t hm = (esc[0] >> (4 * warp)) & 15u;
    const int nesc = (int)esc[2];
    const uint32_t E0 = (ew & 0xFFu) * 0x01010101u, E1 = (ew >> 8) * 0x01010101u;
    uint32_t A[1][4][4];
#pragma unroll
    for (int k = 0; k < 4; ++k) dec_step5(c0, c1, c2, E0, E1, k, A[0][k]);
    if (hm) patch_escapes<1>(A, (nesc > 13) ? a.EALL + __ldg(a.EOU + k0 + j) : esc + 3, nesc, warp, lane);
    __syncwarp();
    mbar_arrive_if(&empty[s], lane == 0);                             // stage read
    uint32_t xb[2][4];
    const uint32_t xg = xa + (uint32_t)(kt * 8 * 128);
    ldsm_x4(xg, xb[0][0], xb[0][1], xb[0][2], xb[0][3]);
    ldsm_x4(xg + 64, xb[1][0], xb[1][1], xb[1][2], xb[1][3]);
#pragma unroll
    for (int k = 0; k < 4; ++k) mma16816(acc, A[0][k], xb[k >> 1][2 * (k & 1)], xb[k >> 1][2 * (k & 1) + 1]);
    if (++kt == cpt) {
      flush();
      kt = 0;
      ++tile;
    }
  }
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if (kt != 0) flush();                                               // the sweep ends inside a tile
#endif
}

}  // namespace th

// ------------------------------------------------------------------------------------------------------------- host
static PFN_cuTensorMapEncodeTiled_v12000 th_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q));
    TORCH_CHECK(q == cudaDriverEntryPointSuccess && p != nullptr, "th: cuTensorMapEncodeTiled unavailable");
    fn = (PFN_cuTensorMapEncodeTiled_v12000)p;
  }
  return fn;
}

// tensor maps are encoded on the host; caching them keeps eager calls cheap (in CUDA graphs they are captured once)
static const CUtensorMap& th_xmap(void* X, int B, int N, int ntok) {
  static std::map<std::tuple<uintptr_t, int, int, int>, CUtensorMap> cache;
  const auto key = std::make_tuple((uintptr_t)X, B, N, ntok);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  if (cache.size() > 1024) cache.clear();
  CUtensorMap map;
  cuuint64_t dims[2] = {(cuuint64_t)N, (cuuint64_t)B};
  cuuint64_t strides[1] = {(cuuint64_t)N * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)ntok};
  cuuint32_t es[2] = {1, 1};
  const CUresult r = th_encode()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, X, dims, strides, box, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "th: tensor map encode failed (", (int)r, ")");
  return cache.emplace(key, map).first->second;
}

// the tree as [M/16 tiles][N/64 groups][194 x u64]: one box = the RT records of a unit, stored back to back
static const CUtensorMap& th_tmap(void* T, int M, int N, int RT) {
  static std::map<std::tuple<uintptr_t, int, int, int>, CUtensorMap> cache;
  const auto key = std::make_tuple((uintptr_t)T, M, N, RT);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  if (cache.size() > 4096) cache.clear();
  CUtensorMap tmap;
  cuuint64_t tdims[3] = {(cuuint64_t)(th::REC / 8), (cuuint64_t)(N / 64), (cuuint64_t)(M / 16)};
  cuuint64_t tstr[2] = {(cuuint64_t)th::REC, (cuuint64_t)th::REC * (N / 64)};
  cuuint32_t tbox[3] = {(cuuint32_t)(th::REC / 8), 1, (cuuint32_t)RT};
  cuuint32_t tes[3] = {1, 1, 1};
  const CUresult r = th_encode()(&tmap, CU_TENSOR_MAP_DATA_TYPE_UINT64, 3, T, tdims, tstr, tbox, tes,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "th: tree tensor map encode failed (", (int)r, ")");
  return cache.emplace(key, tmap).first->second;
}

template <int NTOK, int R, int W>
static void launch_th(const CUtensorMap& map, const CUtensorMap& tmap, const th::Args& a, int C, int nTT,
                      cudaStream_t st) {
  const size_t smem = th::smem_bytes(NTOK, R, W);
  static bool attr[64] = {false};
  int dev = 0;
  cudaGetDevice(&dev);
  if (!attr[dev & 63]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(th::th_kernel<NTOK, R, W>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    attr[dev & 63] = true;
  }
  th::th_kernel<NTOK, R, W><<<dim3((unsigned)C, (unsigned)nTT), th::threads(NTOK), smem, st>>>(map, tmap, a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void th_run(torch::Tensor T, torch::Tensor EB, torch::Tensor EOU, torch::Tensor EALL, torch::Tensor X, torch::Tensor Y,
            torch::Tensor WS, torch::Tensor CNT, int64_t ntok, int64_t C, int64_t R, int64_t W, int64_t flags) {
  const c10::cuda::OptionalCUDAGuard guard(T.device());
  TORCH_CHECK(X.scalar_type() == at::kBFloat16 && X.is_contiguous() && X.dim() == 2, "th: X must be contiguous [B, N] bf16");
  TORCH_CHECK(((uintptr_t)X.data_ptr() & 15) == 0, "th: X must be 16-byte aligned");
  TORCH_CHECK(T.is_contiguous() && ((uintptr_t)T.data_ptr() & 15) == 0, "th: tree must be contiguous, 16-byte aligned");
  TORCH_CHECK(R == 1 || (R == 2 && ntok <= 32), "th: R = 1, or 2 with ntok <= 32");
  TORCH_CHECK((W == 0 && ntok <= 64) || (W == 1 && ntok >= 64 && R == 1) || (W == 2 && ntok >= 128 && R == 1),
              "th: W = 0 (ntok <= 64), 1 (ntok >= 64) or 2 (ntok >= 128)");
  th::Args a;
  a.T = T.data_ptr<uint8_t>(); a.EB = (const uint32_t*)EB.data_ptr<int>(); a.EOU = EOU.data_ptr<int>();
  a.EALL = (const uint32_t*)EALL.data_ptr<int>();
  a.Y = (__nv_bfloat16*)Y.data_ptr<at::BFloat16>(); a.WS = WS.data_ptr<float>(); a.CNT = CNT.data_ptr<int>();
  a.M = (int)Y.size(1); a.N = (int)X.size(1); a.B = (int)X.size(0); a.flags = (int)flags;
  TORCH_CHECK(a.M % (128 * R) == 0 && a.N % 64 == 0 && a.B >= 1, "th: M % (128 R), N % 64");
  a.nRB = a.M / (128 * (int)R);
  const int U = a.nRB * (a.N / 64);
  TORCH_CHECK(C >= 1 && C <= U, "th: 1 <= C <= units");
  const int nTT = (int)((a.B + ntok - 1) / ntok);
  TORCH_CHECK((long long)nTT * a.nRB <= CNT.numel(), "th: counters");
  TORCH_CHECK(WS.numel() >= (long long)nTT * C * th::NCW * (ntok / 2) * R, "th: workspace");
  int sms = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, T.get_device()));
  TORCH_CHECK((long long)C * nTT <= sms, "th: C x token tiles must not exceed the SMs (co-resident CTAs)");
  const CUtensorMap& map = th_xmap(X.data_ptr(), a.B, a.N, (int)ntok);
  const CUtensorMap& tmap = th_tmap(T.data_ptr(), a.M, a.N, 8 * (int)R);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const int c = (int)C;
  if (R == 2) {
    switch (ntok) {
      case 8: launch_th<8, 2, 0>(map, tmap, a, c, nTT, st); break;
      case 16: launch_th<16, 2, 0>(map, tmap, a, c, nTT, st); break;
      case 32: launch_th<32, 2, 0>(map, tmap, a, c, nTT, st); break;
      default: TORCH_CHECK(false, "th: ntok");
    }
    return;
  }
  if (W == 2) {
    switch (ntok) {
      case 128: launch_th<128, 1, 2>(map, tmap, a, c, nTT, st); break;
      case 256: launch_th<256, 1, 2>(map, tmap, a, c, nTT, st); break;
      default: TORCH_CHECK(false, "th: ntok");
    }
    return;
  }
  if (W == 1) {
    switch (ntok) {
      case 64: launch_th<64, 1, 1>(map, tmap, a, c, nTT, st); break;
      case 128: launch_th<128, 1, 1>(map, tmap, a, c, nTT, st); break;
      case 256: launch_th<256, 1, 1>(map, tmap, a, c, nTT, st); break;
      default: TORCH_CHECK(false, "th: ntok");
    }
    return;
  }
  switch (ntok) {
    case 8: launch_th<8, 1, 0>(map, tmap, a, c, nTT, st); break;
    case 16: launch_th<16, 1, 0>(map, tmap, a, c, nTT, st); break;
    case 32: launch_th<32, 1, 0>(map, tmap, a, c, nTT, st); break;
    case 64: launch_th<64, 1, 0>(map, tmap, a, c, nTT, st); break;
    default: TORCH_CHECK(false, "th: ntok");
  }
}

int64_t th_ok() {
  // 1 when this build carries the sm_90a code and the current GPU runs it (H100 / H200), else 0
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  if (p.major != 9) return 0;
  int* f = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&f, sizeof(int)));
  C10_CUDA_CHECK(cudaMemset(f, 0, sizeof(int)));
  th::th_probe<<<1, 1>>>(f);
  const cudaError_t e = cudaGetLastError();
  int h = 0;
  if (e == cudaSuccess) C10_CUDA_CHECK(cudaMemcpy(&h, f, sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(f);
  return e == cudaSuccess ? h : 0;
}

int64_t th_smem(int64_t ntok, int64_t R) {
  if (R != 1 && R != 2) return -1;
  switch (ntok) {
    case 8: return th::smem_bytes(8, (int)R, 0);
    case 16: return th::smem_bytes(16, (int)R, 0);
    case 32: return th::smem_bytes(32, (int)R, 0);
    case 64: return th::smem_bytes(64, (int)R, 1);
    case 128: return th::smem_bytes(128, (int)R, 2);
    case 256: return th::smem_bytes(256, (int)R, 2);
    default: return -1;
  }
}

// ------------------------------------------------------------------------------------------------- mvt host
static bool th_pdl_on() {                  // programmatic dependent launch (sm_90+; STATETREE_PDL=0 disables)
  static int v = 0;
  if (v == 0) {
    const char* e = getenv("STATETREE_PDL");
    v = (e && e[0] == '0') ? 1 : 2;
  }
  return v == 2;
}

int64_t mvt_smem(int64_t S, int64_t B, int64_t N) { return (int64_t)th::mvt_smem((int)S, (int)B, (int)N); }

template <int S>
static void launch_mvt(const th::VArgs& a, cudaStream_t st) {
  const size_t smem = th::mvt_smem(S, a.B, a.N);
  static size_t set[64] = {0};
  int dev = 0;
  cudaGetDevice(&dev);
  if (smem > set[dev & 63]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(th::mvt_kernel<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    set[dev & 63] = smem;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)a.C);
  cfg.blockDim = dim3(288);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = th_pdl_on() ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, th::mvt_kernel<S>, a));
}

void mvt_run(torch::Tensor T, torch::Tensor EB, torch::Tensor EOU, torch::Tensor EALL, torch::Tensor X, torch::Tensor Y,
             torch::Tensor WS, torch::Tensor CNT, int64_t C, int64_t S) {
  const c10::cuda::OptionalCUDAGuard guard(T.device());
  TORCH_CHECK(X.scalar_type() == at::kBFloat16 && X.is_contiguous() && X.dim() == 2, "mvt: X must be contiguous [B, N] bf16");
  TORCH_CHECK(((uintptr_t)X.data_ptr() & 15) == 0, "mvt: X must be 16-byte aligned");
  TORCH_CHECK(T.is_contiguous() && ((uintptr_t)T.data_ptr() & 15) == 0, "mvt: tree must be contiguous, 16-byte aligned");
  th::VArgs a;
  a.T = T.data_ptr<uint8_t>(); a.EB = (const uint32_t*)EB.data_ptr<int>(); a.EOU = EOU.data_ptr<int>();
  a.EALL = (const uint32_t*)EALL.data_ptr<int>();
  a.X = (const __nv_bfloat16*)X.data_ptr<at::BFloat16>(); a.Y = (__nv_bfloat16*)Y.data_ptr<at::BFloat16>();
  a.WS = WS.data_ptr<float>(); a.CNT = CNT.data_ptr<int>();
  a.M = (int)Y.size(1); a.N = (int)X.size(1); a.B = (int)X.size(0);
  TORCH_CHECK(a.B >= 1 && a.B <= 8 && a.M % 16 == 0 && a.N % 512 == 0, "mvt: B <= 8, M % 16, N % 512");
  a.U = (a.M / 16) * (a.N / 64) / 8;
  a.C = (int)C;
  const int cpt = a.N / 512, per = a.U / (int)C;
  TORCH_CHECK(C >= 1 && C <= a.U && per >= 1 && 1 + (cpt - 1 + per - 1) / per <= 4, "mvt: C (a tile spans at most 4 CTAs)");
  TORCH_CHECK(WS.numel() >= (int64_t)(a.M / 16) * 4 * 128 && CNT.numel() >= a.M / 16, "mvt: workspace");
  TORCH_CHECK(EB.numel() >= (int64_t)a.U * 16, "mvt: escape blocks (escape_chunks)");
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  switch (S) {
    case 2: launch_mvt<2>(a, st); break;
    case 4: launch_mvt<4>(a, st); break;
    case 8: launch_mvt<8>(a, st); break;
    default: TORCH_CHECK(false, "mvt: S = 2, 4 or 8");
  }
}
