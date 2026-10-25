// tgx: large-batch state-tree GEMM (prefill / large batches) = CUTLASS GEMM + a decode kernel on the same SMs.
//   Y[T, M] = X[T, K] W[M, K]^T with W given as state-tree records. Two kernels run concurrently:
//   * decode: persistent, 2 warps per CTA, <= 64 registers. Strip by strip (strip = 128 weight rows) it decodes the
//     records into a ring of R row-major BF16 strips (small enough to live mostly in L2) and counts finished chunks.
//     A warp decodes a record into a shared-memory tile and writes it out as full 128-byte lines.
//   * GEMM: CUTLASS's multistage main loop + epilogue (A = X, B = the strip's ring slot), CTA tile 256 tokens x 128
//     weight rows in strip-major order, built for <= 224 registers so that a decode CTA fits beside it on every SM.
//     A GEMM CTA whose strip is not decoded yet decodes chunks of its own strip (or earlier ones) itself instead of
//     waiting, so the call completes even if the decode kernel never gets an SM.
//   Each weight is decoded once per call; the BF16 weights never exist as a whole matrix. Results match a BF16 GEMM
//   with the exact weights (escapes included).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>

static constexpr int kStripRows = 128;
static constexpr int kChunk = 32;            // records per decode work item
static constexpr int kStageWords = 16 * 36;  // one record (16 x 64 BF16) in shared memory, rows padded to 144 bytes

struct XArgs {
  const uint8_t* T;      // records [M / 16][K / 64][1552]
  const int* EO;         // escape offsets per record
  const uint32_t* EL;    // escape entries (value << 16 | lane << 5 | kstep << 3 | reg << 1 | half)
  uint16_t* ring;        // R slots x 128 x K BF16 (row-major strip)
  int* ready;            // [NS] decoded chunks per strip
  int* done;             // [NS] token tiles that finished reading the strip
  int* next;             // decode work-item counter
  int NG, K, MT, NS, R, CH, items;
};

__device__ __forceinline__ void xdec4(uint32_t lf, uint32_t d, uint32_t E4, uint32_t& p01, uint32_t& p23) {
  const uint32_t e4 = E4 - d;                       // leaf format v2 (see kernels.cu dec4)
  const uint32_t a = __byte_perm(lf, e4, 0x5140), b = __byte_perm(lf, e4, 0x7362);
  p01 = __funnelshift_r(a, a, 1);
  p23 = __funnelshift_r(b, b, 1);
}
__device__ __forceinline__ uint4 xldg128(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// Decode the records of work item j (chunk j % CH of strip j / CH) into its ring slot with nw warps of the calling
// CTA (this is warp w). stage: this warp's shared tile. The caller makes sure the slot is free and signals afterwards.
__device__ __noinline__ void decode_item(const XArgs a, int j, int w, int nw, uint32_t* stage) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int sd = j / a.CH, c = j % a.CH;
  const int nrec = (kStripRows / 16) * a.NG;
  const int q0 = c * kChunk, q1 = min(q0 + kChunk, nrec);
  uint16_t* slot = a.ring + (size_t)(sd % a.R) * kStripRows * a.K;
  uint16_t* st16 = reinterpret_cast<uint16_t*>(stage);
  constexpr int U = 2;
  for (int qb = q0 + w; qb < q1; qb += nw * U) {
    uint4 c0[U], c1[U], sw[U];
    uint32_t ex[U];
    int lo[U], hi[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int q = qb + u * nw;
      if (q < q1) {
        const size_t r = (size_t)sd * nrec + q;            // record (row block q / NG, column group q % NG)
        const uint8_t* rec = a.T + r * 1552;
        c0[u] = xldg128(rec + lane * 16);
        c1[u] = xldg128(rec + 512 + lane * 16);
        sw[u] = xldg128(rec + 1024 + lane * 16);
        ex[u] = __ldg((const uint16_t*)(rec + 1536) + g);
        lo[u] = __ldg(a.EO + r);
        hi[u] = __ldg(a.EO + r + 1);
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int q = qb + u * nw;
      if (q < q1) {
        // fragments -> shared tile (row g / g + 8, column pairs 2t and 2t + 8 of each k-step)
        const uint32_t E0 = (ex[u] & 0xFFu) * 0x01010101u, E1 = ((ex[u] >> 8) & 0xFFu) * 0x01010101u;
        const uint32_t L0[4] = {c0[u].x, c0[u].z, c1[u].x, c1[u].z}, L1[4] = {c0[u].y, c0[u].w, c1[u].y, c1[u].w};
        const uint32_t S[4] = {sw[u].x, sw[u].y, sw[u].z, sw[u].w};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          uint32_t A0, A1, A2, A3;
          xdec4(L0[s], S[s] & 0x0F0F0F0Fu, E0, A0, A2);
          xdec4(L1[s], (S[s] >> 4) & 0x0F0F0F0Fu, E1, A1, A3);
          stage[g * 36 + 8 * s + t] = A0;
          stage[g * 36 + 8 * s + 4 + t] = A2;
          stage[(g + 8) * 36 + 8 * s + t] = A1;
          stage[(g + 8) * 36 + 8 * s + 4 + t] = A3;
        }
        __syncwarp();
        if (lo[u] < hi[u]) {                                  // escapes: exact bits over the decoded values
          for (int e = lo[u] + lane; e < hi[u]; e += 32) {
            const uint32_t v = __ldg(a.EL + e);
            const int el = (v >> 5) & 31, es = (v >> 3) & 3, er = (v >> 1) & 3;
            const int row = (el >> 2) + 8 * (er & 1), col = 16 * es + 8 * (er >> 1) + 2 * (el & 3) + (int)(v & 1);
            st16[row * 72 + col] = (uint16_t)(v >> 16);
          }
          __syncwarp();
        }
        // shared tile -> ring: 16 rows x 128 bytes, each quarter-warp store is one full 128-byte line
        const int rr = q / a.NG, G = q - rr * a.NG;
        uint16_t* dst = slot + (size_t)(rr * 16) * a.K + G * 64;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int idx = lane + 32 * k, row = idx >> 3, ch = idx & 7;
          const uint4 v = *reinterpret_cast<const uint4*>(stage + row * 36 + ch * 4);
          *reinterpret_cast<uint4*>(dst + (size_t)row * a.K + ch * 8) = v;
        }
        __syncwarp();
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------- decode kernel
__global__ void __launch_bounds__(64, 16) tgx_decode(XArgs a) {
  __shared__ int item;
  __shared__ __align__(16) uint32_t stage[2][kStageWords];
  const int warp = threadIdx.x >> 5;
  while (true) {
    if (threadIdx.x == 0) item = atomicAdd(a.next, 1);
    __syncthreads();
    const int j = item;
    if (j >= a.items) break;
    const int sd = j / a.CH;
    if (sd >= a.R && threadIdx.x == 0)
      while (ld_acquire(a.done + (sd - a.R)) < a.MT) __nanosleep(256);
    __syncthreads();
    decode_item(a, j, warp, 2, stage[warp]);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(a.ready + sd, 1);
  }
}

// ------------------------------------------------------------------------------------------------- GEMM kernel
template <typename Mma_, typename Epilogue_>
struct StripGemm {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using WarpCount = typename Mma::WarpCount;
  static int const kThreads = 32 * WarpCount::kCount;
  static int const kWarps = WarpCount::kCount;
  static_assert(Mma::Shape::kN == kStripRows, "CTA tile N is one strip");

  struct Params {
    cutlass::gemm::GemmCoord problem;
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorB::Params params_B;
    typename Epilogue::OutputTileIterator::Params params_D;
    const cutlass::bfloat16_t* X;
    cutlass::bfloat16_t* Y;
    typename OutputOp::Params output_op;
    XArgs tree;
  };
  struct SharedStorage {
    union {
      typename Mma::SharedStorage main_loop;
      typename Epilogue::SharedStorage epilogue;
      uint32_t stage[kWarps][kStageWords];
    } u;
    int cmd;
  };

  CUTLASS_DEVICE void operator()(Params const& p, SharedStorage& ss) {
    const XArgs& a = p.tree;
    const int tid = threadIdx.x, warp = cutlass::canonical_warp_idx_sync(), lane = tid & 31;
    const int s = blockIdx.x / a.MT, m = blockIdx.x % a.MT;
    // wait for the strip; meanwhile decode chunks of strips <= s (every wait inside is on earlier strips)
    while (true) {
      if (tid == 0) {
        int cmd = -1;
        if (ld_acquire(a.ready + s) < a.CH) {
          cmd = -2;
          const int nxt = *reinterpret_cast<volatile int*>(a.next);
          if (nxt < (s + 1) * a.CH && nxt < a.items && atomicCAS(a.next, nxt, nxt + 1) == nxt) cmd = nxt;
        }
        ss.cmd = cmd;
      }
      __syncthreads();
      const int cmd = ss.cmd;
      __syncthreads();
      if (cmd == -1) break;
      if (cmd == -2) {
        if (tid == 0) __nanosleep(256);
        continue;
      }
      const int sd = cmd / a.CH;
      if (sd >= a.R && tid == 0)
        while (ld_acquire(a.done + (sd - a.R)) < a.MT) __nanosleep(256);
      __syncthreads();
      decode_item(a, cmd, warp, kWarps, ss.u.stage[warp]);
      __threadfence();
      __syncthreads();
      if (tid == 0) atomicAdd(a.ready + sd, 1);
    }
    typename Mma::IteratorA iterator_A(p.params_A, const_cast<cutlass::bfloat16_t*>(p.X),
                                       {p.problem.m(), p.problem.k()}, tid, {m * Mma::Shape::kM, 0});
    typename Mma::IteratorB iterator_B(
        p.params_B, reinterpret_cast<cutlass::bfloat16_t*>(a.ring + (size_t)(s % a.R) * kStripRows * a.K),
        {p.problem.k(), kStripRows}, tid, {0, 0});
    Mma mma(ss.u.main_loop, tid, warp, lane);
    typename Mma::FragmentC accum;
    accum.clear();
    mma((p.problem.k() + Mma::Shape::kK - 1) / Mma::Shape::kK, accum, iterator_A, iterator_B, accum);
    if (tid == 0) {
      __threadfence();
      atomicAdd(a.done + s, 1);
    }
    OutputOp output_op(p.output_op);
    typename Epilogue::OutputTileIterator iterator_D(p.params_D, p.Y, p.problem.mn(), tid,
                                                     {m * Mma::Shape::kM, s * kStripRows});
    Epilogue epilogue(ss.u.epilogue, tid, warp, lane);
    epilogue(output_op, iterator_D, accum, iterator_D);
  }
};

// no launch bounds: -maxrregcount=224 applies (2 GEMM warps x 224 + 1 decode warp x 64 registers per sub-partition)
template <typename K>
__global__ void tgx_gemm(typename K::Params p) {
  extern __shared__ int4 tgx_smem[];
  K op;
  op(p, *reinterpret_cast<typename K::SharedStorage*>(tgx_smem));
}

struct XCfg {
  using E = cutlass::bfloat16_t;
  using TB = cutlass::gemm::GemmShape<256, kStripRows, 32>;
  using WS = cutlass::gemm::GemmShape<64, 64, 32>;
  using IS = cutlass::gemm::GemmShape<16, 8, 16>;
  using Mma = typename cutlass::gemm::threadblock::DefaultMma<
      E, cutlass::layout::RowMajor, 8, E, cutlass::layout::ColumnMajor, 8, float, cutlass::layout::RowMajor,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS, IS, 4, cutlass::arch::OpMultiplyAdd>::ThreadblockMma;
  using OutOp = cutlass::epilogue::thread::LinearCombination<E, 8, float, float>;
  using Epi = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<TB, typename Mma::Operator, 1, OutOp,
                                                                               OutOp::kCount>::Epilogue;
  using K = StripGemm<Mma, Epi>;
};

namespace {
struct DevRes {
  bool init = false;
  cudaStream_t side = nullptr;
  cudaEvent_t e0 = nullptr, e1 = nullptr;
  int sms = 0, smem = 0;
};
DevRes g_res[64];
}  // namespace

// Y = X W^T. ring: >= 2 strips of 128 x K BF16 (any dtype; bytes count), cnt: int32 >= 2 * M / 128 + 1 (scratch).
// dgrid: decode CTAs (0: one per SM). Returns (strips, ring slots, decode chunks per strip).
std::vector<int64_t> tgx(torch::Tensor T, torch::Tensor EO, torch::Tensor EL, torch::Tensor X, torch::Tensor Y,
                         torch::Tensor ring, torch::Tensor cnt, int64_t dgrid) {
  using C = XCfg;
  using K = C::K;
  const c10::cuda::OptionalCUDAGuard guard(X.device());
  const int Tk = X.size(0), Kd = X.size(1), Mw = Y.size(1);
  TORCH_CHECK(X.scalar_type() == torch::kBFloat16 && Y.scalar_type() == torch::kBFloat16, "tgx: bf16 activations");
  TORCH_CHECK(Kd % 64 == 0 && Mw % kStripRows == 0 && X.is_contiguous() && Y.is_contiguous() && Y.size(0) == Tk,
              "tgx: shape");
  TORCH_CHECK(T.numel() == (int64_t)(Mw / 16) * (Kd / 64) * 1552 && T.is_contiguous(), "tgx: tree shape");
  const int MT = (Tk + C::TB::kM - 1) / C::TB::kM, NS = Mw / kStripRows;
  const int R = (int)std::min<int64_t>(ring.numel() * ring.element_size() / ((int64_t)kStripRows * Kd * 2), NS);
  TORCH_CHECK(R >= 2 && cnt.numel() >= 2 * NS + 1 && cnt.scalar_type() == torch::kInt32, "tgx: ring / counters");
  const int nrec = (kStripRows / 16) * (Kd / 64), CH = (nrec + kChunk - 1) / kChunk;
  K::Params p;
  p.problem = cutlass::gemm::GemmCoord(Tk, Mw, Kd);
  p.params_A = C::Mma::IteratorA::Params(cutlass::layout::RowMajor(Kd));
  p.params_B = C::Mma::IteratorB::Params(cutlass::layout::ColumnMajor(Kd));
  p.params_D = C::Epi::OutputTileIterator::Params(cutlass::layout::RowMajor(Mw));
  p.X = (const cutlass::bfloat16_t*)X.data_ptr();
  p.Y = (cutlass::bfloat16_t*)Y.data_ptr();
  p.output_op = C::OutOp::Params(1.f, 0.f);
  int* c = cnt.data_ptr<int>();
  p.tree = XArgs{T.data_ptr<uint8_t>(), EO.data_ptr<int>(), (const uint32_t*)EL.data_ptr<int>(),
                 (uint16_t*)ring.data_ptr(), c, c + NS, c + 2 * NS, Kd / 64, Kd, MT, NS, R, CH, NS * CH};
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  DevRes& r = g_res[dev & 63];
  if (!r.init) {
    r.smem = (int)sizeof(K::SharedStorage);
    C10_CUDA_CHECK(cudaFuncSetAttribute(tgx_gemm<K>, cudaFuncAttributeMaxDynamicSharedMemorySize, r.smem));
    // both kernels ask for the largest shared-memory carve-out: an SM hosting a decode CTA must still take a GEMM CTA
    C10_CUDA_CHECK(cudaFuncSetAttribute(tgx_gemm<K>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(tgx_decode, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&r.sms, cudaDevAttrMultiProcessorCount, dev));
    int lo = 0, hi = 0;
    C10_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    C10_CUDA_CHECK(cudaStreamCreateWithPriority(&r.side, cudaStreamNonBlocking, hi));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&r.e0, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&r.e1, cudaEventDisableTiming));
    r.init = true;
  }
  auto cur = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(c, 0, sizeof(int) * (2 * NS + 1), cur));
  C10_CUDA_CHECK(cudaEventRecord(r.e0, cur));
  C10_CUDA_CHECK(cudaStreamWaitEvent(r.side, r.e0, 0));
  tgx_decode<<<(int)(dgrid > 0 ? dgrid : r.sms), 64, 0, r.side>>>(p.tree);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  tgx_gemm<K><<<NS * MT, K::kThreads, r.smem, cur>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaEventRecord(r.e1, r.side));
  C10_CUDA_CHECK(cudaStreamWaitEvent(cur, r.e1, 0));
  return {NS, R, CH};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("tgx", &tgx); }
