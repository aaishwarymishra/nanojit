"""Row-wise softmax (last axis) across MLX, Metal, Triton and torch MPS.

Shapes follow the Triton softmax tutorial: 4096 rows, with the row width swept
from 128 to 4096. The harness's ``elements`` axis is therefore ``rows * width``,
so each table row is one width. A softmax row must fit in registers, which is
why the width, not the total size, is the interesting axis.

Triton and Metal are each launched one row per program/threadgroup plus outer
loops:

- `triton-rpp2/4/8` / `metal-rows2/4/8` — `rows_per_program` / ``ROWS``, which
  batch several rows per program or threadgroup. For Triton that *unrolls* the
  row body with ``tl.static_range`` (shader size grows with the row count).
- `triton-persist` — the tutorial's persistent loop (``tl.range``), body
  compiled once, each program striding over rows. It is given 8 rows per
  program, the same batch as `triton-rpp8` and `metal-rows8`, so the two are
  comparable.
"""

import mlx.core as mx
import torch

from benchmarks.harness import Case
from metal_kernels.softmax import softmax as metal_softmax
from tritron_kernels.fused_softmax import (
    fused_softmat as triton_softmax,
)
from tritron_kernels.fused_softmax import (
    fused_softmat_persistent as triton_softmax_persistent,
)

ROWS = 4096
WIDTHS = (128, 256, 512, 1024, 2048, 4096)
SIZES = tuple(ROWS * width for width in WIDTHS)


class Kernel:
    TITLE = "softmax (float32, row-wise, last axis)"
    SIZES = SIZES

    def cases(self) -> list[Case]:
        return cases()


def _shape(elements: int) -> tuple[int, int]:
    # Default sizes are exact multiples; custom --sizes pick the nearest width.
    return (ROWS, max(1, round(elements / ROWS)))


def _mlx_inputs(elements):
    return (mx.random.normal(_shape(elements), dtype=mx.float32), None)


def _torch_inputs(elements):
    return (torch.randn(_shape(elements), device="mps"), None)


def cases() -> list[Case]:
    def mlx_run(x, _):
        out = mx.softmax(x, axis=-1)
        mx.async_eval(out)
        return out

    def metal_run(rows_per_threadgroup):
        def run(x, _):
            out = metal_softmax(x, rows_per_threadgroup=rows_per_threadgroup)
            mx.async_eval(out)
            return out

        return run

    def triton_run(rows_per_program):
        def run(x, _):
            return triton_softmax(x, rows_per_program=rows_per_program)

        return run

    def triton_persist_run(x, _):
        return triton_softmax_persistent(x, num_programs=max(1, x.shape[0] // 8))

    def torch_run(x, _):
        return torch.softmax(x, dim=-1)

    def check_mlx(out, x, _):
        err = mx.abs(out - mx.softmax(x, axis=-1)).max().item()
        assert err < 1e-5, f"mlx softmax err {err}"

    def check_torch(out, x, _):
        err = (out - torch.softmax(x, dim=-1)).abs().max().item()
        assert err < 1e-5, f"torch softmax err {err}"

    # 1 read + 1 write per element.
    bpe = 8
    return [
        Case("mlx", _mlx_inputs, mlx_run, mx.synchronize, check_mlx, bpe),
        Case("metal", _mlx_inputs, metal_run(1), mx.synchronize, check_mlx, bpe),
        *[
            Case(
                f"metal-rows{rpg}",
                _mlx_inputs,
                metal_run(rpg),
                mx.synchronize,
                check_mlx,
                bpe,
            )
            for rpg in (2, 4, 8)
        ],
        Case(
            "triton",
            _torch_inputs,
            triton_run(1),
            torch.mps.synchronize,
            check_torch,
            bpe,
        ),
        *[
            Case(
                f"triton-rpp{rpp}",
                _torch_inputs,
                triton_run(rpp),
                torch.mps.synchronize,
                check_torch,
                bpe,
            )
            for rpp in (2, 4, 8)
        ],
        Case(
            "triton-persist",
            _torch_inputs,
            triton_persist_run,
            torch.mps.synchronize,
            check_torch,
            bpe,
        ),
        Case(
            "torch", _torch_inputs, torch_run, torch.mps.synchronize, check_torch, bpe
        ),
    ]
