import math

import torch
import triton
import triton.language as tl


@triton.jit
def add(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # A tensor offset, not a scalar one: each thread owns its own element.
    # The AppleGPU emitter declines stores whose address is shared by several
    # threads ("several threads address one element").
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offset < n

    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)

    tl.store(out_ptr + offset, x + y, mask=mask)


def triton_add(x: torch.tensor, y: torch.tensor, block_size=1024) -> torch.tensor:
    out = torch.empty_like(x)
    grid = math.ceil(x.numel() / block_size)

    add[(grid,)](x, y, out, x.numel(), BLOCK_SIZE=block_size)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1024, device="mps")
    y = torch.randn(1024, device="mps")
    out = triton_add(x, y)
    print("max err:", (out - (x + y)).abs().max().item())
