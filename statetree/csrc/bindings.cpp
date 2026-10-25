#include <torch/extension.h>
void mv7(torch::Tensor T, torch::Tensor EO, torch::Tensor EL, torch::Tensor X, torch::Tensor Y, torch::Tensor WS,
         torch::Tensor CNT, int64_t WM, int64_t WK, int64_t GY, int64_t PF);
void dec7(torch::Tensor T, torch::Tensor EO, torch::Tensor EL, torch::Tensor W);
void rows7(torch::Tensor T, torch::Tensor EO, torch::Tensor EL, torch::Tensor ids, torch::Tensor out);
void mvd(torch::Tensor T, torch::Tensor X, torch::Tensor Y, torch::Tensor WS, torch::Tensor CNT, int64_t WM,
         int64_t WK, int64_t GY, int64_t PF);
void tm(torch::Tensor T, torch::Tensor EB, torch::Tensor EOU, torch::Tensor EALL, torch::Tensor X, torch::Tensor Y,
        torch::Tensor WS, torch::Tensor CNT, int64_t cfg, int64_t C, int64_t PMAX, int64_t ef, int64_t flags);
void tt_run(int64_t TT, int64_t NWT, int64_t split, torch::Tensor X, torch::Tensor R, torch::Tensor koff,
            torch::Tensor ent, torch::Tensor Y, torch::Tensor ws, torch::Tensor cnt, int64_t KS, int64_t CAP);
std::vector<int64_t> tt_attrs(int64_t TT, int64_t NWT);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "statetree: lossless state-tree BF16 linear kernels";
  m.def("mv", &mv7, "matrix-vector kernel, batch <= 32");
  m.def("dec", &dec7, "decode a tree matrix to BF16");
  m.def("rows", &rows7, "decode selected rows (embedding lookup)");
  m.def("mvd", &mvd, "dense16 ablation: the mv kernel on BF16 blocks in fragment order");
  m.def("tm", &tm, "fused decode + mma GEMM (Marlin-style), batch 12..512");
  m.def("tt", &tt_run, "one-token-tile GEMM for prefill (batch > 64): data-parallel + stream-K or balanced split-K");
  m.def("tt_attrs", &tt_attrs, "registers / local bytes of a compiled tt configuration");
}
