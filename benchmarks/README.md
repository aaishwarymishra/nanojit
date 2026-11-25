# benchmarks

Same kernel across MLX, raw Metal (`mx.fast.metal_kernel`), Triton (AppleGPU)
and torch MPS, one table.

```bash
uv run python -m benchmarks                    # all kernels, all backends
uv run python -m benchmarks add                # one kernel
uv run python -m benchmarks add --sizes 1048576,33554432
uv run python -m benchmarks add --backends mlx,metal,triton,torch
```

Default sizes for `add`: 1M, 8M, 32M float32 elements. Anything smaller is
Python dispatch, not the kernel.

## Method

Warm up, then enqueue a batch of launches (`mx.async_eval` for MLX, plain
launches for Triton/torch), sync once, divide by the batch size. Repeat 5 times
and report mean/median/min. GB/s assumes 12 bytes per element.

Per-call sync is deliberately not used: `torch.mps.synchronize()` costs
~0.2-0.4 ms, which is 10x a 64K add, so the table would report the sync. It also
makes MLX and torch look identical. Batching amortises it. `do_bench` is not used
either: it times the Triton driver queue and reports ~0.07 ms for MLX at every
size. Sync-based numbers are fine for single-op latency, just not for ranking
kernels.

100 ms idle between tests (`COOLDOWN_MS`) keeps each measurement's thermal state
similar. Do not sleep between calls: idle GPU clocks drop and the next call is
slower (16M add: 2.0 ms → 3.2 ms after 2 ms idle).

## Adding a kernel

Create `benchmarks/kernels/myop.py` with `TITLE`, `SIZES` and `cases()`
returning `Case` objects, then register it in `benchmarks/kernels/__init__.py`.
`Case.run` must enqueue only — no `mx.eval`, no `synchronize`.

## Notes

- Compare `vs torch` ratios within one run; absolute ms drift with GPU clocks.
- At 1M the tensors are cache-resident; 32M is the memory-bandwidth number.
- `metal_kernels/add.py` builds a new `mx.fast.metal_kernel` per call, which
  costs ~7-9 µs. Hoist it to module level if you care about small sizes.
