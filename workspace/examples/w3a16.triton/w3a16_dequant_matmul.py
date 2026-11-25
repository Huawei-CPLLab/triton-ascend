import sys
from pathlib import Path

from utils import compile_npu_kernel
import triton
import triton.language as tl


@triton.jit
def dequantize_w3_g32(
    qh,
    qs,
    scales,
    n: tl.constexpr,
    K,
    tile_k: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = 32
    QH_SIZE: tl.constexpr = 4
    QS_SIZE: tl.constexpr = 8
    ng_total = K >> 5
    NG_TILE: tl.constexpr = tile_k >> 5

    rows = tl.arange(0, n)
    cols = tl.arange(0, tile_k)

    group = cols >> 5
    in_group = (cols & 31).to(tl.uint8)

    block = rows[:, None] * ng_total + group[None, :]
    row_mask = rows[:, None] < n

    # Per-group fp16 scales (placed first so all 3 GM->UB loads precede the VFs)
    g_offs = tl.arange(0, NG_TILE)
    scale_val = tl.load(
        scales + rows[:, None] * ng_total + g_offs[None, :],
        mask=row_mask,
        other=0.0,
    )

    # 1. qh: 1 bit per weight (8 weights per byte)
    qh_byte = tl.load(
        qh + block * QH_SIZE + (in_group[None, :] >> 3),
        mask=row_mask,
        other=0,
    )
    shift_h = in_group[None, :] & 7
    qh_bit = (qh_byte >> shift_h) & 1

    # 2. qs: 2 bits per weight (4 weights per byte)
    # Direct byte offset = in_group >> 2. Eliminates second load and tl.where!
    qs_byte = tl.load(
        qs + block * QS_SIZE + (in_group[None, :] >> 2),
        mask=row_mask,
        other=0,
    )
    shift_s = (in_group[None, :] & 3) << 1
    qs_low2 = (qs_byte >> shift_s) & 3

    # 3. Combine bits into 3-bit integer [0..7] in uint8 (pure i8 VF -> 128-wide State::B8_2VL)
    q = (qh_bit << 2) | qs_low2

    # 4. Broadcast fp16 scales across the 32 elements of each group in 3D
    q_3d = tl.reshape(q, (n, NG_TILE, GROUP_SIZE))
    scale_3d = tl.reshape(scale_val, (n, NG_TILE, 1))
    w_3d = (q_3d.to(tl.float16) - 4.0) * scale_3d
    return tl.reshape(w_3d, (n, tile_k))


@triton.jit
def w3a16_dequant_matmul(
    x_ptr,
    qh_ptr,
    qs_ptr,
    scale_ptr,
    shape_ptr,
    out_ptr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tl.static_assert(
        GROUP_SIZE == 32,
        "dequantize_w3_g32 requires GROUP_SIZE == 32",
    )
    tl.static_assert(
        BLOCK_SIZE_N == 16 or BLOCK_SIZE_N == 32,
        "BLOCK_SIZE_N must be 16 or 32 for Cube fractal alignment",
    )

    M = tl.load(shape_ptr + 0)
    N = tl.load(shape_ptr + 1)
    K = tl.load(shape_ptr + 2)

    num_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_k = tl.cdiv(K, BLOCK_SIZE_K)
    ng_total = K >> 5
    NG_TILE: tl.constexpr = BLOCK_SIZE_K >> 5

    k_range = tl.arange(0, BLOCK_SIZE_K)

    for pid_m in range(0, num_m):
        m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        m_mask = m[:, None] < M

        for n_idx in range(0, num_n):
            is_odd_m = (pid_m & 1) == 1
            pid_n = (num_n - 1 - n_idx) if is_odd_m else n_idx

            qh_n_base = qh_ptr + (pid_n * BLOCK_SIZE_N) * (ng_total * 4)
            qs_n_base = qs_ptr + (pid_n * BLOCK_SIZE_N) * (ng_total * 8)
            scale_n_base = scale_ptr + (pid_n * BLOCK_SIZE_N * ng_total)

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for k_idx in range(0, num_k):
                k_cols = k_idx * BLOCK_SIZE_K + k_range
                x_tile = tl.load(
                    x_ptr + m[:, None] * K + k_cols[None, :],
                    mask=m_mask,
                    other=0.0,
                )

                qh_tile = qh_n_base + k_idx * (NG_TILE * 4)
                qs_tile = qs_n_base + k_idx * (NG_TILE * 8)
                scale_tile = scale_n_base + k_idx * NG_TILE
                w = dequantize_w3_g32(
                    qh_tile, qs_tile, scale_tile, BLOCK_SIZE_N, K, BLOCK_SIZE_K
                )
                acc = tl.dot(x_tile, tl.trans(w), acc)

            n_out = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            tl.store(
                out_ptr + m[:, None] * N + n_out[None, :],
                acc.to(tl.float16),
                mask=m_mask,
            )


if __name__ == "__main__":
    signature = {
        "x_ptr": "*fp16",
        "qh_ptr": "*u8",
        "qs_ptr": "*u8",
        "scale_ptr": "*fp16",
        "shape_ptr": "*i32",
        "out_ptr": "*fp16",
    }
    constexprs = {
        "GROUP_SIZE": 32,
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_K": 512,
    }

    out_dir = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "output"
    compile_npu_kernel(w3a16_dequant_matmul, signature, constexprs, "w3a16_dequant_matmul", out_dir)
