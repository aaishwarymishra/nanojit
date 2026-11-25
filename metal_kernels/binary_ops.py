import math

import mlx.core as mx

# 16-byte loads: each thread handles READ_ROWS = 16 / dtype bytes elements
# (4 for fp32), so the vectorised load matches Triton's BLOCK_SIZE layout.
SOURCE = """
    auto pid = thread_position_in_grid.x;
    auto offset = pid * {n};

    const vec<T, {n}> a = reinterpret_cast<device const vec<T, {n}>&>(x[offset]);
    const vec<T, {n}> b = reinterpret_cast<device const vec<T, {n}>&>(y[offset]);

    reinterpret_cast<device vec<T, {n}>&>(out[offset]) = a {op} b;
"""

OP_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_kernels = {}


def _kernel(op: str, rows: int):
    key = (op, rows)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(
            name=f"binary_{op}_{rows}",
            source=SOURCE.format(op=OP_SYMBOLS[op], n=rows),
            input_names=["x", "y"],
            output_names=["out"],
        )
    return _kernels[key]


def binary_op(op: str, x: mx.array, y: mx.array) -> mx.array:
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    if x.dtype == mx.float64:
        raise ValueError("Metal does not support float64")

    rows = 16 // x.dtype.size  # 4 for fp32
    assert x.size % rows == 0, "size must be a multiple of 16 / dtype bytes"

    threads = x.size // rows
    (out,) = _kernel(op, rows)(
        inputs=[x, y],
        template=[("T", x.dtype)],
        grid=(threads, 1, 1),
        threadgroup=(min(threads, 256), 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out


if __name__ == "__main__":
    x = mx.random.normal((1024,), dtype=mx.float32)
    y = mx.random.normal((1024,), dtype=mx.float32)
    for op in OP_SYMBOLS:
        out = binary_op(op, x, y)
        ref = {"add": x + y, "sub": x - y, "mul": x * y, "div": x / y}[op]
        print(op, "max err:", (out - ref).abs().max().item())
