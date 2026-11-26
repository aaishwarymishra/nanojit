"""Row-wise softmax over the last axis, in raw Metal.

One threadgroup per row. Each thread owns ``READ`` elements (16 bytes for
fp32), reduces inside its simdgroup with ``simd_max`` / ``simd_sum``, and the
simdgroups combine their partials through threadgroup memory. Values stay in
registers, so each element is read from memory once and written once.

``READ`` is a Metal template argument and the threadgroup size is a launch
parameter, so the source is static: no per-shape string manipulation.
``ROWS`` is a template argument too: each threadgroup can process several
rows in a serial loop (one row per threadgroup is the default, ``ROWS=1``).
"""

import math

import mlx.core as mx

READS = (2, 4, 8, 16)
MAX_THREADS = 1024

SOURCE = """
    uint tg = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint groups = simdgroups_per_threadgroup;
    threadgroup float partial[32];

    for (int r = 0; r < ROWS; r++) {
        uint row = tg * (uint)ROWS + (uint)r;
        uint base = row * (uint)cols;
        if (base >= (uint)size) {
            break;
        }

        uint offset = tid * (uint)READ;
        float vals[READ];
        float local_max = -INFINITY;
        for (int j = 0; j < READ; j++) {
            uint idx = offset + uint(j);
            vals[j] = idx < (uint)cols ? float(x[base + idx]) : -INFINITY;
            local_max = max(local_max, vals[j]);
        }

        local_max = simd_max(local_max);
        if (lane == 0) {
            partial[sg] = local_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float row_max = -INFINITY;
        for (uint g = 0; g < groups; g++) {
            row_max = max(row_max, partial[g]);
        }
        // The sum overwrites the maxima, so everyone must read them first.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_sum = 0.0f;
        for (int j = 0; j < READ; j++) {
            vals[j] = exp(vals[j] - row_max);
            local_sum += vals[j];
        }

        local_sum = simd_sum(local_sum);
        if (lane == 0) {
            partial[sg] = local_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float row_sum = 0.0f;
        for (uint g = 0; g < groups; g++) {
            row_sum += partial[g];
        }
        // One reciprocal per thread instead of one division per element.
        float inv = 1.0f / row_sum;

        for (int j = 0; j < READ; j++) {
            uint idx = offset + uint(j);
            if (idx < (uint)cols) {
                out[base + idx] = T(vals[j] * inv);
            }
        }

        // The next row reuses partial[]; order its writes against this row's
        // reads. Folded away when ROWS == 1, which needs no trailing barrier.
        if (ROWS > 1) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
"""

_kernel = mx.fast.metal_kernel(
    name="softmax",
    source=SOURCE,
    input_names=["x", "size", "cols"],
    output_names=["out"],
)


def softmax(x: mx.array, num_reads: int = 4, rows_per_threadgroup: int = 1) -> mx.array:
    """Softmax over the last axis of ``x``.

    ``rows_per_threadgroup`` is the serial outer-loop batch (``ROWS``): how
    many rows one threadgroup processes back to back. The default of 1 is one
    row per threadgroup, which keeps the kernel fully parallel.
    """
    if x.dtype == mx.float64:
        raise ValueError("Metal does not support float64")
    if x.size == 0:
        return mx.array(x)
    if num_reads not in READS:
        raise ValueError(f"num_reads must be one of {READS}")
    if rows_per_threadgroup < 1:
        raise ValueError("rows_per_threadgroup must be >= 1")

    cols = x.shape[-1]
    rows = x.size // cols
    threads = min(MAX_THREADS, max(32, math.ceil(cols / num_reads / 32) * 32))
    if threads * num_reads < cols:
        raise ValueError(
            f"row length {cols} exceeds {threads * num_reads} elements; "
            "pass a larger num_reads"
        )

    threadgroups = math.ceil(rows / rows_per_threadgroup)
    (out,) = _kernel(
        inputs=[x, x.size, cols],
        template=[("T", x.dtype), ("READ", num_reads), ("ROWS", rows_per_threadgroup)],
        grid=(threadgroups * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out


if __name__ == "__main__":
    for shape in ((1024, 512), (33, 1000), (7, 3), (1, 1), (5, 130)):
        x = mx.random.normal(shape, dtype=mx.float32)
        out = softmax(x)
        ref = mx.softmax(x, axis=-1)
        print(shape, "max err:", (out - ref).abs().max().item())

    for rows, cols, reads in ((9, 4096, 8), (4, 9000, 16), (11, 77, 2)):
        x = mx.random.normal((rows, cols), dtype=mx.float32)
        out = softmax(x, num_reads=reads)
        ref = mx.softmax(x, axis=-1)
        print((rows, cols), f"read={reads}", "max err:", (out - ref).abs().max().item())

    x = mx.random.normal((64, 256), dtype=mx.float16)
    print("float16 max err:", (softmax(x) - mx.softmax(x, axis=-1)).abs().max().item())
