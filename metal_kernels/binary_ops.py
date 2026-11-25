import math

import mlx.core as mx

# 16-byte loads: each thread handles READ_ROWS = 16 / dtype bytes elements
# (4 for fp32). A tail thread (offset + READ_ROWS > size) falls back to
# element-wise loads so any size works, not just multiples of READ_ROWS.
SOURCE = """
    uint pid = thread_position_in_grid.x;
    uint offset = pid * {n};

    if (offset + {n} <= (uint)size) {
        const vec<T, {n}> a = reinterpret_cast<device const vec<T, {n}>&>(x[offset]);
        const vec<T, {n}> b = reinterpret_cast<device const vec<T, {n}>&>(y[offset]);
        reinterpret_cast<device vec<T, {n}>&>(out[offset]) = a {op} b;
    } else {
        for (uint i = offset; i < (uint)size; i++) {
            out[i] = x[i] {op} y[i];
        }
    }
"""

OP_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_kernels = {}


def _kernel(op: str, rows: int):
    key = (op, rows)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(
            name=f"binary_{op}_{rows}",
            source=SOURCE.replace("{op}", OP_SYMBOLS[op]).replace("{n}", str(rows)),
            input_names=["x", "y", "size"],
            output_names=["out"],
        )
    return _kernels[key]


def binary_op(op: str, x: mx.array, y: mx.array) -> mx.array:
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    if x.dtype == mx.float64:
        raise ValueError("Metal does not support float64")

    rows = 16 // x.dtype.size  # 4 for fp32
    threads = math.ceil(x.size / rows)
    (out,) = _kernel(op, rows)(
        inputs=[x, y, x.size],
        template=[("T", x.dtype)],
        grid=(threads, 1, 1),
        threadgroup=(min(threads, 256), 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out


if __name__ == "__main__":
    for size in (1024, 1000):
        x = mx.random.normal((size,), dtype=mx.float32)
        y = mx.random.normal((size,), dtype=mx.float32)
        for op in OP_SYMBOLS:
            out = binary_op(op, x, y)
            ref = {"add": x + y, "sub": x - y, "mul": x * y, "div": x / y}[op]
            print(size, op, "max err:", (out - ref).abs().max().item())
