# nanojit

A tiny ML compiler written in pure Python. It symbolically traces ordinary Python
functions into a small SSA-style intermediate representation (IR) — the first step
toward optimization and code generation for machine-learning workloads.

## How it works

1. `Expr` nodes form a graph. Python operators (`+`, `-`, `*`, and their reflected
   forms) build new nodes instead of computing values.
2. `trace(fn, *args)` runs the function on the arguments you pass. Plain Python
   values (scalars, lists, dicts, defaults) stay constants; `Expr(op="input")`
   marks a value symbolic. Required arguments you omit become inputs.
3. `print_ir` walks the resulting graph, assigns virtual registers (`%0`, `%1`, ...)
   and prints the IR.

## Example

```python
from nanojit import trace


def model(x):
    return x * 2 + 3


trace(model)
```

Output:

```
%0 = input x
%1 = 2
%2 = mul %0 %1
%3 = 3
%4 = add %2 %3
return %4
```

## Roadmap

Early prototype / learning project.

- [x] Symbolic tracing of scalar expressions
- [x] SSA-style IR printing with register reuse
- [ ] Shape and dtype inference — prerequisite for every backend
- [ ] Array / tensor ops and a reference lowering to NumPy (eager execution)
- [ ] Optimization passes: constant folding, CSE, dead-code elimination
- [ ] Fusion pass + kernel selection (pattern match subgraphs → dispatch)
- [ ] Triton backend for fused elementwise kernels
- [ ] Tests and CI

## Backends

The plan is a `Backend`/`Target` interface so a new target plugs in without touching
the frontend. Selection, not generation: the optimizer pattern-matches IR subgraphs
against a registry of prebuilt kernels and emits a call — the same trick MLX and XLA
use (fuse elementwise chains, dispatch matmul to vendor BLAS/cuDNN).

1. **NumPy (reference)** — correctness baseline, runs everywhere including this Mac.
2. **Triton (GPU)** — prewritten kernels (`add`, `mul`, `fused_elementwise`,
   `reduce_sum`, `matmul`) in a registry; a pass fuses op chains and dispatches.
   Writing Triton source *from* the IR comes last, if at all.
3. **MLX / Metal (optional, Apple Silicon)** — the macOS-native equivalent; Triton
   cannot run on Apple GPUs.

Two notes: matmul/conv are almost always better off calling into BLAS than generating
Triton, and Triton needs a Linux GPU + PyTorch — so GPU CI is optional, not blocking.
A GPU machine (or CI runner) becomes useful once milestone "Triton backend" lands.

## Development

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync              # create the environment
uv run python -m nanojit   # run the demo
```

## Layout

```
src/nanojit/
  __init__.py   package entry point
  __main__.py   demo entry point (`python -m nanojit`)
  compiler.py   Expr graph, tracer, and IR printer
```
