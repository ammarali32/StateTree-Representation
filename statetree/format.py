"""Lossless state-tree format for BF16 matrices.

A BF16 weight is sign | 8-bit exponent | 7-bit mantissa. Every row is cut into groups of 64 weights; the group's
largest exponent E is the tree node, each weight keeps a 4-bit state D = E - exponent and an 8-bit leaf (sign and
mantissa; leaf format v2: mantissa << 1 | sign of its pair partner, the two weights of a bf16 pair holding each
other's sign, which the kernels decode with two byte permutes and two 1-bit rotates). D = 15 marks an escape (|w| < 2^-14 of the group's largest weight, ~0.01% of LLM weights) whose exact bits
are kept in a side list. 12.125 bits/weight + escapes, every weight decodes to its exact BF16 bits.

Storage unit: one 1552-byte record per 16-row x 64-column block, laid out in mma.m16n8k16 A-fragment order:
  [0, 1024)     leaves: 2 chunks x 32 lanes x 16 B; lane (g, t), chunk c: rows g / g+8, k-steps 2c, 2c+1
  [1024, 1536)  states: 32 lanes x 16 B; word s byte b = D(row g) | D(row g+8) << 4
  [1536, 1552)  nodes: 8 x (E(row g) | E(row g+8) << 8)
"""
import torch

RECORD_BYTES = 1552
LEAF_FORMAT = 2                     # 1: s << 7 | m (statetree <= 0.3), 2: m << 1 | partner sign (>= 0.4)
PERM = [16 * (p // 4) + 2 * t + (p % 2) + 8 * ((p // 2) % 2) for t in range(4) for p in range(16)]


def _v2(lv):
    """leaf bytes s << 7 | m -> m << 1 | sign of the pair partner (bytes 0-1 and 2-3 of every 4-byte leaf word)"""
    w = lv.reshape(*lv.shape[:-1], -1, 4)
    part = w[..., [1, 0, 3, 2]]
    return (((w & 0x7F) << 1) | (part >> 7)).reshape(lv.shape)


def _v1(lv):
    """inverse of _v2"""
    w = lv.reshape(*lv.shape[:-1], -1, 4)
    part = w[..., [1, 0, 3, 2]]
    return (((part & 1) << 7) | (w >> 1)).reshape(lv.shape)


@torch.no_grad()
def leaves_v1_to_v2(tree):
    """a tree stored with leaf format 1 (statetree <= 0.3 checkpoints) -> leaf format 2 (states and nodes unchanged)"""
    t = tree.clone()
    t[..., :1024] = _v2(t[..., :1024].to(torch.int32)).to(torch.uint8)
    return t


def _encode_rows(W, perm):
    M, N = W.shape
    T, NG = M // 16, N // 64
    b = W.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = (b >> 15) & 1, (b >> 7) & 0xFF, b & 0x7F
    eg = e.view(M, NG, 64)
    E = eg.amax(-1)
    D = E[..., None] - eg
    esc = D > 14
    Dq = torch.where(esc, torch.full_like(D, 15), D)[..., perm]
    SM = torch.where(esc, torch.zeros_like(D), ((s << 7) | m).view(M, NG, 64))[..., perm]
    leaf = _v2(SM.reshape(T, 2, 8, NG, 4, 2, 2, 4).permute(0, 3, 5, 2, 4, 6, 1, 7).reshape(T, NG, 1024))
    Dv = Dq.reshape(T, 2, 8, NG, 4, 4, 4)
    nib = (Dv[:, 0] | (Dv[:, 1] << 4)).permute(0, 2, 1, 3, 4, 5).reshape(T, NG, 512)
    Ev = E.view(T, 2, 8, NG)
    Eb = torch.stack([Ev[:, 0], Ev[:, 1]], -1).permute(0, 2, 1, 3).reshape(T, NG, 16)
    rec = torch.cat([leaf, nib, Eb], 2).to(torch.uint8).contiguous()
    er, ec = esc.view(M, N).nonzero(as_tuple=True)
    bits = W.contiguous().view(torch.int16)[er, ec]
    return rec, er, ec, bits


@torch.no_grad()
def encode(W, chunk_rows=4096):
    """W [M, N] bf16 (M % 16 == 0, N % 64 == 0) -> dict(tree [M/16, N/64, 1552] uint8, esc_rows/esc_cols int32,
    esc_bits int16). Runs on W's device, in row chunks to bound memory."""
    assert W.dim() == 2 and W.dtype == torch.bfloat16, "encode expects a 2-D bf16 matrix"
    M, N = W.shape
    assert M % 16 == 0 and N % 64 == 0, f"shape {tuple(W.shape)}: rows must be a multiple of 16, cols of 64"
    perm = torch.tensor(PERM, device=W.device)
    chunk_rows = max(16, chunk_rows // 16 * 16)
    tree = torch.empty(M // 16, N // 64, RECORD_BYTES, dtype=torch.uint8, device=W.device)   # filled in place: no
    rows, cols, bits = [], [], []                                                          # second copy of the tree
    for r0 in range(0, M, chunk_rows):
        rec, er, ec, eb = _encode_rows(W[r0:r0 + chunk_rows], perm)
        tree[r0 // 16:r0 // 16 + rec.shape[0]] = rec
        del rec
        rows.append((er + r0).to(torch.int32))
        cols.append(ec.to(torch.int32))
        bits.append(eb)
    return dict(tree=tree, esc_rows=torch.cat(rows), esc_cols=torch.cat(cols), esc_bits=torch.cat(bits))


@torch.no_grad()
def escape_count(W, chunk_rows=8192):
    """number of weights of W [M, N] (or [..., M, N], a stack of matrices) that the tree cannot hold (exponent more
    than 14 below the largest of its 64-group): the escapes encode() would list. Chunked: a few MB of scratch."""
    assert W.dtype == torch.bfloat16 and W.shape[-1] % 64 == 0
    W2 = W.reshape(-1, W.shape[-1])
    n = 0
    for r0 in range(0, W2.shape[0], chunk_rows):
        b = W2[r0:r0 + chunk_rows].contiguous().view(torch.int16).to(torch.int32)
        e = ((b >> 7) & 0xFF).view(b.shape[0], -1, 64)
        n += int(((e.amax(-1, keepdim=True) - e) > 14).sum())
    return n


@torch.no_grad()
def decode_reference(tree, esc_rows, esc_cols, esc_bits):
    """Pure-torch decode (slow reference; the CUDA 'dec' kernel is the fast path). Returns the exact bf16 matrix."""
    Tn, NG, _ = tree.shape
    M, N = Tn * 16, NG * 64
    rec = tree.to(torch.int32)
    SMl = _v1(rec[..., :1024]).reshape(Tn, NG, 2, 8, 4, 2, 2, 4).permute(0, 6, 3, 1, 4, 2, 5, 7).reshape(M, NG, 64)
    nib = rec[..., 1024:1536].reshape(Tn, NG, 8, 4, 4, 4)
    Dl = torch.stack([nib & 15, nib >> 4], 1).permute(0, 1, 3, 2, 4, 5, 6).reshape(M, NG, 64)
    E = rec[..., 1536:].reshape(Tn, NG, 8, 2).permute(0, 3, 2, 1).reshape(M, NG)
    perm = torch.tensor(PERM, device=tree.device)
    SM = torch.empty_like(SMl)
    Dq = torch.empty_like(Dl)
    SM[..., perm] = SMl
    Dq[..., perm] = Dl
    SM, Dq = SM.reshape(M, N), Dq.reshape(M, N)
    ex = (E.repeat_interleave(64, 1) - Dq).clamp(min=0)
    bits = ((SM & 128) << 8) | (ex << 7) | (SM & 127)
    bits = torch.where(bits >= 32768, bits - 65536, bits).to(torch.int16)
    if esc_rows.numel():
        bits[esc_rows.long(), esc_cols.long()] = esc_bits
    return bits.view(torch.bfloat16)


def _i32(x):
    return torch.where(x >= 2 ** 31, x - 2 ** 32, x).to(torch.int32).contiguous()


def _esc_fields(M, N, esc_rows, esc_cols, esc_bits):
    rows, cols = esc_rows.long(), esc_cols.long()
    bits = esc_bits.to(torch.int64) & 0xFFFF
    tile, rr = rows // 16, rows % 16
    h, g = rr // 8, rr % 8
    G, c = cols // 64, cols % 64
    s, kk = c // 16, c % 16
    tq, hi, half = (kk % 8) // 2, kk // 8, kk % 2
    lane, reg = g * 4 + tq, h + 2 * hi
    return bits, tile, G, s, lane, reg, half


@torch.no_grad()
def escape_lists(M, N, esc_rows, esc_cols, esc_bits):
    """Per (16-row tile, 64-col group) escape lists for the mv / dec kernels: EO [tiles*NG + 1] offsets,
    EL entries (bf16 bits << 16 | lane << 5 | k-step << 3 | register << 1 | half)."""
    NG = N // 64
    dev = esc_rows.device
    bits, tile, G, s, lane, reg, half = _esc_fields(M, N, esc_rows, esc_cols, esc_bits)
    key = tile * NG + G
    order = torch.argsort(key, stable=True)
    ent = ((bits << 16) | (lane << 5) | (s << 3) | (reg << 1) | half)[order]
    eo = torch.zeros((M // 16) * NG + 1, dtype=torch.int64, device=dev)
    if key.numel():
        eo[1:] = torch.bincount(key, minlength=(M // 16) * NG).cumsum(0)
    el = _i32(ent) if ent.numel() else torch.zeros(1, dtype=torch.int32, device=dev)
    return eo.to(torch.int32).contiguous(), el


@torch.no_grad()
def escape_blocks(M, N, RT, esc_rows, esc_cols, esc_bits):
    """Escape blocks for the fused tm kernel with row blocks of RT tiles: EB [U, 16] int32 (w0/w1: 64-bit mask over
    (record, k-step), w2: count, w3..15: entries), EOU [U + 1] offsets and EALL (the full list, read by the kernel
    for units with more than 13 escapes)."""
    NG = N // 64
    dev = esc_rows.device
    bits, tile, G, s, lane, reg, half = _esc_fields(M, N, esc_rows, esc_cols, esc_bits)
    rb, i = tile // RT, tile % RT
    u = rb * NG + G
    nU = (M // 16 // RT) * NG
    ent = (bits << 16) | (i << 10) | (lane << 5) | (s << 3) | (reg << 1) | half
    order = torch.argsort(u, stable=True)
    us, es = u[order], ent[order]
    cnt = torch.bincount(u, minlength=nU) if u.numel() else torch.zeros(nU, dtype=torch.int64, device=dev)
    eou = torch.zeros(nU + 1, dtype=torch.int64, device=dev)
    eou[1:] = cnt.cumsum(0)
    m = torch.zeros(nU * 64, dtype=torch.bool, device=dev)
    if u.numel():
        m[u * 64 + i * 4 + s] = True
    words = (m.view(nU, 2, 32).long() << torch.arange(32, device=dev)).sum(-1)
    eb = torch.zeros(nU, 16, dtype=torch.int64, device=dev)
    eb[:, 0:2] = words
    eb[:, 2] = cnt
    if u.numel():
        pos = torch.arange(us.numel(), device=dev) - eou[us]
        keep = pos < 13
        eb[us[keep], 3 + pos[keep]] = es[keep]
    eall = _i32(es) if es.numel() else torch.zeros(1, dtype=torch.int32, device=dev)
    return _i32(eb.view(-1)), eou.to(torch.int32).contiguous(), eall


def bits_per_weight(M, N, n_esc, esc_bytes=10):
    """Stored size: records + escape list (row, col: 4 B each, bits: 2 B)."""
    return ((M // 16) * (N // 64) * RECORD_BYTES * 8 + n_esc * esc_bytes * 8) / (M * N)
