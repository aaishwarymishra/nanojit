import math

import torch
import triton
import triton.language as tl

try:
    from _compat import prepare_metal_backend
except ImportError:  # imported as triton_kernels.binary_ops
    from ._compat import prepare_metal_backend

prepare_metal_backend()


@triton.jit
def binary_kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr, OP: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offset)
    y = tl.load(y_ptr + offset)

    if OP == "add":
        out = x + y
    elif OP == "sub":
        out = x - y
    elif OP == "mul":
        out = x * y
    else:
        out = x / y

    tl.store(out_ptr + offset, out)


def binary_op(op: str, x: torch.tensor, y: torch.tensor) -> torch.tensor:
    # 16-byte loads: 16 / element_size = 4 elements per thread for fp32, with
    # 256 threads per group (the same decomposition as the Metal kernel).
    rows = 16 // x.element_size()
    threads = 256
    block_size = rows * threads
    assert x.numel() % block_size == 0, "size must be a multiple of block_size"

    out = torch.empty_like(x)
    grid = math.ceil(x.numel() / block_size)
    binary_kernel[(grid,)](
        x,
        y,
        out,
        BLOCK_SIZE=block_size,
        OP=op,
        num_warps=threads // 32,
    )
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1024, device="mps")
    y = torch.randn(1024, device="mps")
    for op in ("add", "sub", "mul", "div"):
        out = binary_op(op, x, y)
        ref = {"add": x + y, "sub": x - y, "mul": x * y, "div": x / y}[op]
        print(op, "max err:", (out - ref).abs().max().item())
