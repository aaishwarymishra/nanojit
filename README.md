# nanojit

A tiny ML compiler written in pure Python. It symbolically traces ordinary Python
functions into a small SSA-style intermediate representation (IR) — the first step
toward optimization and code generation for machine-learning workloads.

## How it works

1. `Expr` nodes form a graph. Python operators (`+`, `-`, `*`, and their reflected
   forms) build new nodes instead of computing values.
2. `trace(fn, *examples)` runs the function on example arguments. Array arguments
   become symbolic inputs (shape/dtype are the spec); other values (scalars,
   defaults, lists, dicts) stay constants.
3. `lower_to_ir` walks the resulting graph, assigns virtual registers (`%0`, `%1`,
   ...) and returns a `Program` of SSA `Instruction`s.

## Example

```python
import numpy as np

from nanojit import trace


def model(x):
    return x * 2 + 3


print(trace(model, np.array(2.0)))
```

Output:

```
inputs:
    %0: float64[]

instructions:
    %1 = const 2 : int64[]
    %2 = mul %0, %1 : float64[]
    %3 = const 3 : int64[]
    %4 = add %2, %3 : float64[]
return %4
```

## Roadmap

Early prototype / learning project.

- [x] Symbolic tracing of scalar expressions
- [x] SSA-style IR (`Program` / `Instruction`) with register reuse
- [x] Shape and dtype inference — prerequisite for every backend
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
  compiler.py   Expr graph, tracer, and IR lowering
```
