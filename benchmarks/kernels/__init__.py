"""Benchmark kernels. Each module exports ``OPS``/``cases()`` and ``SIZES``."""

from benchmarks.kernels import binary_ops

KERNELS = {op: binary_ops.Kernel(op) for op in binary_ops.OPS}
