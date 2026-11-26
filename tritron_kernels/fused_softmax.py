import math

import torch
import triton
import triton.language as tl

try:
    from _compat import prepare_metal_backend
except ImportError:  # imported as triton_kernels.fused_softmax
    from ._compat import prepare_metal_backend

prepare_metal_backend()


@triton.jit
def fused_softmat_kernel(
    x_ptr,
    out_ptr,
    col_size: tl.constexpr,
    ROWS_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offset = pid * ROWS_SIZE * col_size

    # Unroll at compile time; a runtime loop is unnecessary here.
    for i in tl.static_range(ROWS_SIZE):
        row = row_offset + i * col_size
        # A row occupies [row, row + col_size). BLOCK_SIZE >= col_size, so the
        # whole row fits in one block and the padding lanes are masked off.
        offset = row + tl.arange(0, BLOCK_SIZE)
        mask = offset < row + col_size
        x = tl.load(x_ptr + offset, mask=mask, other=-float("inf"))

        # Subtract the maximum value and exponentiate
        x_exp = tl.exp(x - tl.max(x))

        # Compute the sum of exponentials
        sum_exp = tl.sum(x_exp)

        # Normalize to get softmax
        softmax = x_exp / sum_exp

        # Store the result
        tl.store(out_ptr + offset, softmax, mask=mask)


def fused_softmat(
    x: torch.Tensor, rows_per_program: int = 8, num_reads: int = 4
) -> torch.Tensor:
    """Row-wise softmax over the last dimension.

    ``BLOCK_SIZE`` must be a power of two ("2N purity") because ``tl.arange``
    only accepts power-of-two lengths. A row is therefore rounded up to the
    next power of two and the extra lanes are masked out by the kernel.
    """
    if x.dim() < 2:
        raise ValueError("fused_softmat expects at least a 2-D tensor")

    col_size = x.shape[-1]
    num_rows = x.numel() // col_size
    if col_size == 0 or num_rows == 0:
        return torch.empty_like(x)

    # 2N purity: tl.arange requires a power-of-two block size.
    block_size = triton.next_power_of_2(col_size)

    # The kernel masks within a row but not against the global end, so keep
    # rows_per_program a divisor of num_rows to avoid out-of-bounds rows.
    rows_per_program = max(1, min(rows_per_program, num_rows))
    while num_rows % rows_per_program != 0:
        rows_per_program -= 1

    out = torch.empty_like(x)
    grid = (triton.cdiv(num_rows, rows_per_program),)
    wraps = math.ceil(math.ceil(block_size / num_reads) / 32)
    fused_softmat_kernel[grid](
        x,
        out,
        col_size=col_size,
        ROWS_SIZE=rows_per_program,
        BLOCK_SIZE=block_size,
        num_warps=max(wraps, 1),
    )
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    for shape in ((1024, 512), (33, 1000), (7, 3)):
        x = torch.randn(*shape, device="mps")
        out = fused_softmat(x)
        ref = torch.softmax(x, dim=-1)
        print(shape, "max err:", (out - ref).abs().max().item())
