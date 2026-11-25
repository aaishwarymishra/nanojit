"""``add``: elementwise ``out = x + y`` for float32.

Same math in four places: MLX's built-in ``x + y``, ``metal_kernels.add``,
``tritron_kernels.add`` and torch MPS. Sizes start at 1M so the kernel, not
Python dispatch, is what the table compares.
"""

import mlx.core as mx
import torch

from benchmarks.harness import Case
from metal_kernels.add import metal_add
from tritron_kernels.add import triton_add

TITLE = "add (float32, x + y)"
SIZES = (1_048_576, 8_388_608, 33_554_432)


def _mlx_inputs(n):
    return (
        mx.random.normal((n,), dtype=mx.float32),
        mx.random.normal((n,), dtype=mx.float32),
    )


def _mlx_add(x, y):
    out = x + y
    mx.async_eval(out)  # enqueue, don't eval
    return out


def _check_mlx(out, x, y):
    err = mx.max(mx.abs(out - (x + y))).item()
    assert err < 1e-6, f"mlx add max err {err}"


def _metal_add(x, y):
    out = metal_add(x, y)
    mx.async_eval(out)
    return out


def _torch_inputs(n):
    return (
        torch.randn(n, device="mps"),
        torch.randn(n, device="mps"),
    )


def _triton_add(x, y):
    return triton_add(x, y)


def _torch_add(x, y):
    return torch.add(x, y)


def _check_torch(out, x, y):
    err = (out - (x + y)).abs().max().item()
    assert err < 1e-6, f"torch/triton add max err {err}"


def cases() -> list[Case]:
    return [
        Case("mlx", _mlx_inputs, _mlx_add, mx.synchronize, _check_mlx),
        Case("metal", _mlx_inputs, _metal_add, mx.synchronize, _check_mlx),
        Case("triton", _torch_inputs, _triton_add, torch.mps.synchronize, _check_torch),
        Case("torch", _torch_inputs, _torch_add, torch.mps.synchronize, _check_torch),
    ]
