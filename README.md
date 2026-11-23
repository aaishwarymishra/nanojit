# nanojit

A tiny ML compiler written in pure Python. It symbolically traces ordinary Python
functions into a small SSA-style intermediate representation (IR) — the first step
toward optimization and code generation for machine-learning workloads.

## How it works

1. `Expr` nodes form a graph. Python operators (`+`, `-`, `*`, and their reflected
   forms) build new nodes instead of computing values.
2. `trace(fn)` binds every function parameter to an `Expr(op="input")` and runs the
   function on those symbolic values.
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

## Status

Early prototype / learning project.

- [x] Symbolic tracing of scalar expressions
- [x] SSA-style IR printing with register reuse
- [ ] Shape and dtype inference
- [ ] IR optimization passes (constant folding, CSE, dead-code elimination)
- [ ] Array / tensor operations backed by NumPy
- [ ] Native code generation
- [ ] Tests and CI

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
