import mlx.core as mx

# One element per thread, 256 threads per group: the same decomposition as the
# Triton kernel, so the benchmark compares backends rather than schedules.
source = """
    auto pid = thread_position_in_grid.x;
    out[pid] = x[pid] + y[pid];
"""

_kernel = mx.fast.metal_kernel(
    name="add",
    source=source,
    input_names=["x", "y"],
    output_names=["out"],
)


def metal_add(x: mx.array, y: mx.array) -> mx.array:
    assert x.shape == y.shape
    assert x.dtype == y.dtype

    if x.dtype == mx.float64:
        raise ValueError("Metal does not support float64")

    (out,) = _kernel(
        inputs=[x, y],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out


if __name__ == "__main__":
    x = mx.random.normal((1024,), dtype=mx.float32)
    y = mx.random.normal((1024,), dtype=mx.float32)
    out = metal_add(x, y)
    print("max err:", (out - (x + y)).abs().max().item())
