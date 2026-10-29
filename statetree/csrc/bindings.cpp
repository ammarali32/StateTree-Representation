#include <torch/extension.h>
void mv7(torch::Tensor T, torch::Tensor EO, torch::Tensor EL, torch::Tensor X, torch::Tensor Y, torch::Tensor WS,
         torch::Tensor CNT, int64_t WM, int64_t WK, int64_t GY, int64_t PF);
void mvs(torch::Tensor T, torch::Tensor EO, torch::Tensor EL, torch::Tensor X, torch::Tensor Y, torch::Tensor WS,
         torch::Tensor CNT, int64_t WM, int64_t WK, int64_t GY, int64_t PF);
int64_t mvs_smem_bytes(int64_t B, int64_t N, int64_t WM, int64_t WK, int64_t GY);
void mvh(torch::Tensor T, torch::Tensor EO, torch::Tensor EL, torch::Tensor X, torch::Tensor Y, torch::Tensor WS,
         torch::Tensor CNT, int64_t WM, int64_t WK, int64_t GY, int64_t PF);
int64_t mvh_smem_bytes(int64_t B, int64_t N, int64_t WM, int64_t WK, int64_t GY);
void mvc(torch::Tensor T, torch::Tensor EO, torch::Tensor EL, torch::Tensor X, torch::Tensor Y, torch::Tensor WS,
         torch::Tensor CNT, int64_t WK, int64_t GY, int64_t TC, int64_t PF);
int64_t mvc_smem_bytes(int64_t B, int64_t N, int64_t WK, int64_t GY, int64_t TC);
void th_run(torch::Tensor T, torch::Tensor EB, torch::Tensor EOU, torch::Tensor EALL, torch::Tensor X, torch::Tensor Y,
            torch::Tensor WS, torch::Tensor CNT, int64_t ntok, int64_t C, int64_t R, int64_t W, int64_t flags);
int64_t th_ok();
void mvt_run(torch::Tensor T, torch::Tensor EB, torch::Tensor EOU, torch::Tensor EALL, torch::Tensor X, torch::Tensor Y,
             torch::Tensor WS, torch::Tensor CNT, int64_t C, int64_t S);
int64_t mvt_smem(int64_t S, int64_t B, int64_t N);
int64_t th_smem(int64_t ntok, int64_t R);
void dec7(torch::Tensor T, torch::Tensor EO, torch::Tensor EL, torch::Tensor W);
void mvx(torch::Tensor TP, torch::Tensor EOP, torch::Tensor ELP, torch::Tensor X, torch::Tensor ids, torch::Tensor Y,
         torch::Tensor WS, torch::Tensor CNT, int64_t xdiv, int64_t WM, int64_t WK, int64_t GY, int64_t PF);
void mvg(torch::Tensor TP, torch::Tensor EOP, torch::Tensor ELP, torch::Tensor X, torch::Tensor sorted,
         torch::Tensor bexp, torch::Tensor npost, torch::Tensor Y, torch::Tensor WS, torch::Tensor CNT, int64_t xdiv,
         int64_t NT, int64_t WM, int64_t WK, int64_t GY);
void wsum(torch::Tensor Y, torch::Tensor W, torch::Tensor out);
void mvwx(torch::Tensor TP, torch::Tensor EOP, torch::Tensor ELP, torch::Tensor X, torch::Tensor ids, torch::Tensor Y,
          int64_t xdiv, int64_t nbps);
void mvwg(torch::Tensor TP, torch::Tensor EOP, torch::Tensor ELP, torch::Tensor X, torch::Tensor sorted,
          torch::Tensor bexp, torch::Tensor npost, torch::Tensor Y, int64_t xdiv, int64_t nbps);
void mfd(torch::Tensor TP13, torch::Tensor EOP13, torch::Tensor ELP13, torch::Tensor TP2, torch::Tensor EOP2,
         torch::Tensor ELP2, torch::Tensor X, torch::Tensor ids, torch::Tensor W, torch::Tensor act, torch::Tensor cnt,
         torch::Tensor out, int64_t I);
void rows7(torch::Tensor T, torch::Tensor EO, torch::Tensor EL, torch::Tensor ids, torch::Tensor out);
void mvd(torch::Tensor T, torch::Tensor X, torch::Tensor Y, torch::Tensor WS, torch::Tensor CNT, int64_t WM,
         int64_t WK, int64_t GY, int64_t PF);
void tm(torch::Tensor T, torch::Tensor EB, torch::Tensor EOU, torch::Tensor EALL, torch::Tensor X, torch::Tensor Y,
        torch::Tensor WS, torch::Tensor CNT, int64_t cfg, int64_t C, int64_t PMAX, int64_t ef, int64_t flags);
void tt_run(int64_t TT, int64_t NWT, int64_t split, torch::Tensor X, torch::Tensor R, torch::Tensor koff,
            torch::Tensor ent, torch::Tensor Y, torch::Tensor ws, torch::Tensor cnt, int64_t KS, int64_t CAP);
std::vector<int64_t> tt_attrs(int64_t TT, int64_t NWT);
void gtt_run(int64_t TT, int64_t NWT, int64_t MINB, torch::Tensor TP, torch::Tensor koff, torch::Tensor ent,
             torch::Tensor X, torch::Tensor sorted, torch::Tensor bexp, torch::Tensor npost, torch::Tensor Y,
             int64_t xdiv, int64_t KS, int64_t CAP);
std::vector<int64_t> gtt_attrs(int64_t TT, int64_t NWT, int64_t MINB);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "statetree: lossless state-tree BF16 linear kernels";
  m.attr("ABI") = 58;   // 0.5.3: mvs, th (Hopper TMA + wgmma), mvh / mvc / mvt (B <= 8); ranked on sm_90+ only
  m.def("mv", &mv7, "matrix-vector kernel, batch <= 32");
  m.def("mvs", &mvs, "matrix-vector kernel with activations staged in shared memory (Hopper), batch <= 32");
  m.def("mvs_smem", &mvs_smem_bytes, "dynamic shared memory of an mvs launch");
  m.def("mvh", &mvh, "lean matrix-vector kernel (few instructions per record: Hopper, power-capped GPUs), batch <= 8");
  m.def("mvh_smem", &mvh_smem_bytes, "dynamic shared memory of an mvh launch");
  m.def("mvc", &mvc, "matrix-vector kernel sweeping the records in memory order (CTA-contiguous), batch <= 8");
  m.def("mvc_smem", &mvc_smem_bytes, "dynamic shared memory of an mvc launch");
  m.def("th", &th_run, "Hopper (sm_90a) fused decode + wgmma GEMM: TMA producer warp, 2 consumer warpgroups, stream-K");
  m.def("th_ok", &th_ok, "1 if the sm_90a th kernel is built and runs on the current GPU");
  m.def("th_smem", &th_smem, "dynamic shared memory of a th launch (token tile ntok, R records per warp)");
  m.def("mvt", &mvt_run, "Hopper (sm_90a) matrix-vector products: TMA sweep of the records in memory order, B <= 8");
  m.def("mvt_smem", &mvt_smem, "dynamic shared memory of an mvt launch (S stages, batch B, N columns)");
  m.def("dec", &dec7, "decode a tree matrix to BF16");
  m.def("mvx", &mvx, "MoE matrix-vector products: one (input row, expert) slot per GEMV, experts as trees");
  m.def("mvg", &mvg, "grouped MoE products: up to 32 slots of one expert per block (moe_align_block_size layout)");
  m.def("mfd", &mfd, "fused MoE decode: gate-up, silu * up, down and the routing-weighted sum in one launch");
  m.def("mvwx", &mvwx, "MoE products for small K (down projection): persistent warp-per-tile, mvx slot layout");
  m.def("mvwg", &mvwg, "MoE products for small K (down projection): persistent warp-per-tile, moe_align layout bs 8");
  m.def("wsum", &wsum, "routing-weighted sum of the k expert outputs of every token (fp32 sum, BF16 out)");
  m.def("rows", &rows7, "decode selected rows (embedding lookup)");
  m.def("mvd", &mvd, "dense16 ablation: the mv kernel on BF16 blocks in fragment order");
  m.def("tm", &tm, "fused decode + mma GEMM (Marlin-style), batch 12..512");
  m.def("tt", &tt_run, "one-token-tile GEMM for prefill (batch > 64): data-parallel + stream-K or balanced split-K");
  m.def("tt_attrs", &tt_attrs, "registers / local bytes of a compiled tt configuration");
  m.def("gtt", &gtt_run, "grouped MoE GEMM, experts as state trees (moe_align_block_size layout, block size TT)");
  m.def("gtt_attrs", &gtt_attrs, "registers / local bytes of a compiled gtt configuration");
}
