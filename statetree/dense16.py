"""Ablation control for the state-tree format.

Dense16Linear runs the same batch <= 32 kernel as TreeLinear (warp-per-tile mma, split-K merged by the last CTA,
prefetch depth, reduction; autotuned over the same configurations) on plain BF16 weights, 16 bits per weight, stored in
the mma-fragment order: one 2048-byte block per 16 x 64 tile, no decode, no escapes.

    statetree vs dense16   the format alone (12.1 instead of 16 bits read, plus the decode work)
    dense16   vs cuBLAS    the kernel alone (same bytes)

Batches above 32 unpack the weight to a BF16 scratch matrix and call cuBLAS (the tree uses its tm kernel or decode +
cuBLAS there), so dense16 is a control for the decode regime (batch <= 32), not for prefill. In vLLM it is
'--representation dense16'."""
import torch

from .linear import TreeLinear, _state
from .ops import ext

__all__ = ["Dense16Linear"]

BLOCK = 2048                                     # bytes per 16 x 64 tile
_POS = {}


def _pos(N, dev):
    """[16 * N] int64: element (r, c) of a 16-row tile (row-major) -> its BF16 offset inside the tile's blocks."""
    key = (N, str(dev))
    p = _POS.get(key)
    if p is None:
        c = torch.arange(N)
        G, cc = c // 64, c % 64
        s, q = cc // 16, cc % 16
        hi, qq = q // 8, q % 8
        t, b = qq // 2, qq % 2
        rr = torch.arange(16)[:, None]
        g, half = rr % 8, rr // 8
        lane = g * 4 + t
        p = _POS[key] = (G * 1024 + s * 256 + lane * 8 + (half + 2 * hi) * 2 + b).reshape(-1).to(dev)
    return p


class Dense16Linear(TreeLinear):
    """nn.Linear with BF16 weights in mma-fragment order, run by the TreeLinear batch <= 32 kernel without decoding."""

    TUNE_TAG = "|dense16"

    def __init__(self, in_features, out_features, bias=False, device=None, n_esc=0):
        super().__init__(in_features, out_features, bias=bias, device=device, n_esc=0)
        self.tree = torch.empty(out_features // 16, in_features // 64, BLOCK, dtype=torch.uint8,
                                device=self.tree.device)

    def _packed(self):
        return self.tree.reshape(self.out_features // 16, -1).view(torch.bfloat16)          # [M / 16, 16 N]

    @classmethod
    @torch.no_grad()
    def from_weight(cls, W, bias=None, verify=True):
        W = W.detach()
        assert W.dtype == torch.bfloat16, f"Dense16Linear: weight dtype {W.dtype}, expected bfloat16"
        work = W if W.is_cuda or not torch.cuda.is_available() else W.cuda()
        M, N = work.shape
        self = cls(N, M, bias=bias is not None, device=work.device)
        self._packed().index_copy_(1, _pos(N, work.device), work.reshape(M // 16, 16 * N))
        if bias is not None:
            self.bias.copy_(bias.detach().to(torch.bfloat16))
        if verify and not torch.equal(self.dequantize().view(torch.int16), work.view(torch.int16)):
            raise RuntimeError("Dense16Linear: unpacked weights differ from the original (layout bug)")
        return self

    @property
    def bits_per_weight(self):
        return 16.0

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
                f"dense16 (ablation control)")

    def _esc_lists(self):
        return None, None

    def _valid(self, path, prm, B):
        if path == "tm":
            return False
        if path == "mv":
            return B <= 32 and self.out_features % (16 * prm[0]) == 0
        return True

    @torch.no_grad()
    def dequantize(self):
        M, N = self.out_features, self.in_features
        return torch.index_select(self._packed(), 1, _pos(N, self.tree.device)).view(M, N)

    @torch.no_grad()
    def rows(self, ids):
        dev, N = self.tree.device, self.in_features
        flat = ids.reshape(-1).to(device=dev, dtype=torch.int64)
        P = self._packed().reshape(-1)
        pos = _pos(N, dev).view(16, N)
        out = torch.empty(flat.numel(), N, dtype=torch.bfloat16, device=dev)
        for i in range(0, flat.numel(), 1024):
            f = flat[i:i + 1024]
            out[i:i + 1024] = P[(f // 16 * (16 * N))[:, None] + pos[f % 16]]
        out = out.view(*ids.shape, N)
        return out.to(ids.device) if ids.is_cuda and ids.device != dev else out

    def _resolve(self, bk):
        if bk > 32 and not self.allow_dec:                  # vocabulary tables: 32-row slices of the mv kernel
            return ("chunk", 32)
        return super()._resolve(bk)

    def _run(self, x, y, plan):
        path, prm = plan
        M, N, B = self.out_features, self.in_features, x.shape[0]
        if path == "chunk":
            step = prm or 512
            for i in range(0, B, step):
                n = min(step, B - i)
                self._run(x[i:i + n], y[i:i + n], self._plan(n))
            return
        st = _state(x.device)
        if path == "mv":
            WM, WK, GY, PF = prm
            nbc = 8 if B <= 8 else 16 if B <= 16 else 32
            ws = st.get_ws((M // (16 * WM)) * GY * 16 * WM * nbc if GY > 1 else 1)
            ext().mvd(self.tree, x, y, ws, st.cnt, WM, WK, GY, PF)
        else:                                               # unpack to a BF16 scratch matrix, then cuBLAS
            W = st.get_scratch(M * N)[:M * N].view(M // 16, 16 * N)
            torch.index_select(self._packed(), 1, _pos(N, x.device), out=W)
            torch.matmul(x, W.view(M, N).t(), out=y)
