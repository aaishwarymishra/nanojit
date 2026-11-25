import math

import torch
import triton
import triton.language as tl


@triton.jit
def add(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if EVEN:
        # No mask: same work per thread as the Metal kernel.
        x = tl.load(x_ptr + offset)
        y = tl.load(y_ptr + offset)
        tl.store(out_ptr + offset, x + y)
    else:
        mask = offset < n
        x = tl.load(x_ptr + offset, mask=mask)
        y = tl.load(y_ptr + offset, mask=mask)
        tl.store(out_ptr + offset, x + y, mask=mask)


def triton_add(x: torch.tensor, y: torch.tensor, block_size=256) -> torch.tensor:
    # 256 elements / 8 warps = 1 element per thread. The AppleGPU emitter uses
    # scalar loads, so BLOCK_SIZE must equal 32 * num_warps; otherwise each
    # thread owns several adjacent elements and every load strides -> uncoalesced.
    out = torch.empty_like(x)
    grid = math.ceil(x.numel() / block_size)

    add[(grid,)](
        x,
        y,
        out,
        x.numel(),
        BLOCK_SIZE=block_size,
        EVEN=x.numel() % block_size == 0,
        num_warps=block_size // 32,
    )
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1024, device="mps")
    y = torch.randn(1024, device="mps")
    out = triton_add(x, y)
    print("max err:", (out - (x + y)).abs().max().item())
