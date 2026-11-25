"""Benchmark kernels. Add a module here and expose ``cases()`` + ``SIZES``."""

from benchmarks.kernels import add

KERNELS = {
    "add": add,
}
