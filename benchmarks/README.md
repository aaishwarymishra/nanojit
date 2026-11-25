# benchmarks

Elementwise binary ops (`add`, `sub`, `mul`, `div`) across MLX, raw Metal
(`mx.fast.metal_kernel`), Triton (AppleGPU) and torch MPS — one table per op.

```bash
uv run python -m benchmarks                    # every op, every backend
uv run python -m benchmarks add                # one op
uv run python -m benchmarks add --sizes 1048576,33554432
uv run python -m benchmarks add --backends mlx,metal,triton,torch
```

Default sizes: 1M, 8M, 32M float32 elements. Smaller is Python dispatch, not
the kernel.

## Same work per thread

Metal and Triton both load a 16-byte vector per thread —
`read_rows = 16 / dtype bytes` = 4 elements for fp32 — with 256 threads per
group, so the decomposition matches across frameworks:

- `metal_kernels/binary_ops.py` — `vec<T,4>` load/add/store, one kernel per op.
- `tritron_kernels/binary_ops.py` — `BLOCK_SIZE = read_rows * 256`, `OP`
  constexpr, one kernel for all ops.

The MLX/torch rows are the vendor built-ins; their internals aren't ours to
control, so they're reference points.

Caveat: the AppleGPU Triton backend emits **scalar** loads, so its
4-elements-per-thread is uncoalesced (each instruction strides by 4), while
Metal's is a real `float4` load. Same source-level structure, different
codegen — that gap is the backend, not the kernel.

## Method

Warm up, then enqueue a batch of launches (`mx.async_eval` for MLX, plain
launches for Triton/torch), sync once, divide by the batch size. Repeat 5 times
and report mean/median/min. GB/s assumes 12 bytes per element (2 reads + 1
write, fp32).

Per-call sync is deliberately not used: `torch.mps.synchronize()` costs
~0.2-0.4 ms, which is 10x a 64K add, so the table would report the sync. It also
makes MLX and torch look identical. Batching amortises it. `do_bench` is not used
either: it times the Triton driver queue and reports ~0.07 ms for MLX at every
size.

100 ms idle between tests (`COOLDOWN_MS`) keeps each measurement's thermal state
similar. Do not sleep between calls: idle GPU clocks drop and the next call is
slower (16M add: 2.0 ms → 3.2 ms after 2 ms idle).

## Adding an op

Add it to `_OPS` in `metal_kernels/binary_ops.py` (symbol), `OP_SYMBOLS`
(Metal source), the Triton `if/elif` chain, and `_OPS` in
`benchmarks/kernels/binary_ops.py` (reference). The registry picks it up
automatically.

## Notes

- Compare `vs torch` ratios within one run; absolute ms drift with GPU clocks.
- The mean can be skewed by a rare slow trial — the median/min columns are the
  robust ones.
- At 1M the tensors are cache-resident; 32M is the memory-bandwidth number.
