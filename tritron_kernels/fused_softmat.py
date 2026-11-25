import torch
import triton
import triton.language as tl


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

    for row in tl.range(row_offset, row_offset + ROWS_SIZE * col_size, col_size):
        # Load a block of rows
        offset = row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offset, mask=offset < col_size, other=-float("inf"))

        # Subtract the maximum value and exponentiate
        x_exp = tl.exp(x - tl.max(x))

        # Compute the sum of exponentials
        sum_exp = tl.sum(x_exp)

        # Normalize to get softmax
        softmax = x_exp / sum_exp

        # Store the result
        tl.store(out_ptr + offset, softmax, mask=offset < col_size)
