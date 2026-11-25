"""Elementwise binary ops (add/sub/mul/div) across MLX, Metal, Triton and torch.

Each backend loads a 16-byte vector per thread (4 fp32 elements), so the
decomposition matches across frameworks.
"""

import operator

import mlx.core as mx
import torch

from metal_kernels.binary_ops import binary_op as metal_binary_op
from tritron_kernels.binary_ops import binary_op as triton_binary_op

from benchmarks.harness import Case

SIZES = (1_048_576, 8_388_608, 33_554_432)
_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_OPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
}
OPS = tuple(_OPS)


class Kernel:
    """One op of the binary-ops family, registered under its own name."""

    def __init__(self, op: str):
        self.op = op

    @property
    def TITLE(self) -> str:
        return f"{self.op} (float32, x {_SYMBOLS[self.op]} y)"

    SIZES = SIZES

    def cases(self) -> list[Case]:
        return cases(self.op)


def _mlx_inputs(n):
    return (
        mx.random.normal((n,), dtype=mx.float32),
        mx.random.normal((n,), dtype=mx.float32),
    )


def _torch_inputs(n):
    return (
        torch.randn(n, device="mps"),
        torch.randn(n, device="mps"),
    )


def cases(op: str) -> list[Case]:
    fn = _OPS[op]

    def mlx_run(x, y):
        out = fn(x, y)
        mx.async_eval(out)
        return out

    def metal_run(x, y):
        out = metal_binary_op(op, x, y)
        mx.async_eval(out)
        return out

    def triton_run(x, y):
        return triton_binary_op(op, x, y)

    def torch_run(x, y):
        return fn(x, y)

    def check_mlx(out, x, y):
        ref = fn(x, y)
        # Division can produce inf/nan (a 0 denominator); treat inf==inf as equal.
        assert mx.all(mx.isfinite(out) == mx.isfinite(ref)).item(), f"mlx {op} non-finite mismatch"
        finite = mx.isfinite(ref)
        err = mx.max(mx.where(finite, mx.abs(out - ref), 0.0)).item()
        scale = max(1.0, mx.max(mx.where(finite, mx.abs(ref), 0.0)).item())
        assert err / scale < 1e-5, f"mlx {op} err {err}"

    def check_torch(out, x, y):
        ref = fn(x, y)
        assert torch.isfinite(out).eq(torch.isfinite(ref)).all().item(), f"torch {op} non-finite mismatch"
        finite = torch.isfinite(ref)
        err = (out - ref).abs().masked_fill(~finite, 0).max().item()
        scale = max(1.0, ref.abs().masked_fill(~finite, 0).max().item())
        assert err / scale < 1e-5, f"torch {op} err {err}"

    return [
        Case("mlx", _mlx_inputs, mlx_run, mx.synchronize, check_mlx),
        Case("metal", _mlx_inputs, metal_run, mx.synchronize, check_mlx),
        Case("triton", _torch_inputs, triton_run, torch.mps.synchronize, check_torch),
        Case("torch", _torch_inputs, torch_run, torch.mps.synchronize, check_torch),
    ]
