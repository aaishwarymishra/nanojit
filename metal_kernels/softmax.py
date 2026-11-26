"""Row-wise softmax over the last axis, in raw Metal.

One threadgroup covers ``ROWS`` rows. Each thread holds ``READ`` elements of a
row (16 bytes for fp32), reduces inside its simdgroup with ``simd_max`` /
``simd_sum``, and the simdgroups combine their partials through threadgroup
memory. Values stay in registers, so each element is read from memory once and
written once.
"""

import math

import mlx.core as mx

SOURCE = """
    uint tg = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    threadgroup float partial[{GROUPS}];

    for (uint r = 0; r < {ROWS}; r++) {
        uint row = tg * {ROWS} + r;
        uint base = row * (uint)cols;
        if (base >= (uint)size) {
            break;
        }

        uint offset = tid * {READ};
        float vals[{READ}];
        float local_max = -INFINITY;

        if (offset + {READ} <= (uint)cols && (cols & ({READ} - 1)) == 0) {
            {LOAD}
            for (uint j = 0; j < {READ}; j++) {
                vals[j] = float(v[j]);
                local_max = max(local_max, vals[j]);
            }
        } else {
            for (uint j = 0; j < {READ}; j++) {
                uint idx = offset + j;
                vals[j] = idx < (uint)cols ? float(x[base + idx]) : -INFINITY;
                local_max = max(local_max, vals[j]);
            }
        }

        local_max = simd_max(local_max);
        if (lane == 0) {
            partial[sg] = local_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float row_max = -INFINITY;
        for (uint g = 0; g < {GROUPS}; g++) {
            row_max = max(row_max, partial[g]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_sum = 0.0f;
        for (uint j = 0; j < {READ}; j++) {
            vals[j] = exp(vals[j] - row_max);
            local_sum += vals[j];
        }

        local_sum = simd_sum(local_sum);
        if (lane == 0) {
            partial[sg] = local_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float row_sum = 0.0f;
        for (uint g = 0; g < {GROUPS}; g++) {
            row_sum += partial[g];
        }
        float inv = 1.0f / row_sum;

        if (offset + {READ} <= (uint)cols && (cols & ({READ} - 1)) == 0) {
            vec<T, {READ}> o;
            for (uint j = 0; j < {READ}; j++) {
                o[j] = T(vals[j] * inv);
            }
            reinterpret_cast<device vec<T, {READ}>&>(out[base + offset]) = o;
        } else {
            for (uint j = 0; j < {READ}; j++) {
                uint idx = offset + j;
                if (idx < (uint)cols) {
                    out[base + idx] = T(vals[j] * inv);
                }
            }
        }

        // Everyone must finish reading `partial` before the next row reuses it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

MAX_THREADS = 1024
# MLX binds arrays with fewer than 8 elements to the constant address space
# (mlx/backend/common/custom_kernel.cpp), so they cannot be cast to device vec.
MIN_DEVICE_ELEMENTS = 8
_kernels = {}


def _kernel(read: int, threads: int, rows: int, vector: bool):
    key = (read, threads, rows, vector)
    if key not in _kernels:
        load = (
            "vec<T, {READ}> v = reinterpret_cast<device const vec<T, {READ}>&>"
            "(x[base + offset]);"
            if vector
            else "vec<T, {READ}> v; "
            "for (uint j = 0; j < {READ}; j++) v[j] = x[base + offset + j];"
        )
        source = SOURCE.replace("{LOAD}", load)
        for name, value in (
            ("{GROUPS}", threads // 32),
            ("{ROWS}", rows),
            ("{READ}", read),
        ):
            source = source.replace(name, str(value))
        _kernels[key] = mx.fast.metal_kernel(
            name=f"softmax_{read}_{threads}_{rows}_{'v' if vector else 's'}",
            source=source,
            input_names=["x", "size", "cols"],
            output_names=["out"],
        )
    return _kernels[key]


def softmax(x: mx.array, num_reads: int = 4, rows_per_threadgroup: int = 1) -> mx.array:
    """Softmax over the last axis of ``x`` (contiguous by construction)."""
    if x.dtype == mx.float64:
        raise ValueError("Metal does not support float64")
    if x.size == 0:
        return mx.array(x)

    cols = x.shape[-1]
    rows = x.size // cols
    if num_reads not in (2, 4, 8, 16):
        raise ValueError("num_reads must be one of 2, 4, 8, 16")

    needed = math.ceil(cols / num_reads)
    threads = min(MAX_THREADS, max(32, math.ceil(needed / 32) * 32))
    if threads * num_reads < cols:
        raise ValueError(
            f"row length {cols} exceeds {threads * num_reads} elements; "
            "pass a larger num_reads"
        )
    rows_per_tg = max(1, min(rows_per_threadgroup, rows))
    threadgroups = math.ceil(rows / rows_per_tg)
    vector = x.size >= MIN_DEVICE_ELEMENTS

    (out,) = _kernel(num_reads, threads, rows_per_tg, vector)(
        inputs=[x, x.size, cols],
        template=[("T", x.dtype)],
        grid=(threadgroups * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out


if __name__ == "__main__":
    shapes = ((1024, 512), (33, 1000), (7, 3))
    for shape in shapes:
        x = mx.random.normal(shape, dtype=mx.float32)
        out = softmax(x)
        ref = mx.softmax(x, axis=-1)
        print(shape, "max err:", (out - ref).abs().max().item())

    for cols, rows in ((130, 5), (4096, 2), (100, 17)):
        x = mx.random.normal((rows, cols), dtype=mx.float32)
        out = softmax(x, rows_per_threadgroup=4)
        ref = mx.softmax(x, axis=-1)
        print((rows, cols), "max err:", (out - ref).abs().max().item())

    x = mx.random.normal((64, 256), dtype=mx.float16)
    out = softmax(x)
    ref = mx.softmax(x, axis=-1)
    print("float16", "max err:", (out - ref).abs().max().item())
