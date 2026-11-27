import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nanojit.compiler import Program, bind_arguments, trace

_OPS: dict[str, Callable[..., Any]] = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "matmul": np.matmul,
}


def execute(program: Program, *args: NDArray[Any]) -> NDArray[Any]:
    """Execute a lowered SSA ``Program`` on NumPy arrays."""
    input_insts = [inst for inst in program.instructions if inst.op == "input"]
    if len(args) != len(input_insts):
        raise ValueError(f"Expected {len(input_insts)} inputs, got {len(args)}.")

    env: dict[str, NDArray[Any]] = {}
    for inst, value in zip(input_insts, args):
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Input {inst.output} must be numpy.ndarray, got {type(value).__name__}."
            )
        expected_dtype = np.dtype(inst.spec.dtype)
        if value.shape != inst.spec.shape or value.dtype != expected_dtype:
            raise ValueError(
                f"Input {inst.output} expected {(inst.spec.shape, expected_dtype)}, "
                f"got {(value.shape, value.dtype)}."
            )
        env[inst.output] = value

    for inst in program.instructions:
        if inst.op == "input":
            continue
        if inst.op == "const":
            env[inst.output] = np.asarray(inst.inputs[0])
            continue
        if inst.op not in _OPS:
            raise ValueError(f"Unsupported op: {inst.op!r}")
        env[inst.output] = np.asarray(
            _OPS[inst.op](*(env[str(r)] for r in inst.inputs))
        )

    return env[program.output]


def compile(
    fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Callable[..., NDArray[Any]]:
    """Trace ``fn`` on example arguments and return a callable executor."""
    sig = inspect.signature(fn)
    example_args = bind_arguments(sig, *args, **kwargs)

    program = trace(fn, *args, **kwargs)
    input_names = [
        inst.name
        for inst in program.instructions
        if inst.op == "input" and inst.name is not None
    ]
    input_name_set = set(input_names)
    static_args = {
        name: value
        for name, value in example_args.items()
        if name not in input_name_set
    }

    def wrapper(*call_args: Any, **call_kwargs: Any) -> NDArray[Any]:
        bound_args = bind_arguments(sig, *call_args, **call_kwargs)
        for name, expected in static_args.items():
            actual = bound_args[name]
            if actual != expected:
                raise ValueError(
                    f"Static argument {name!r} changed from {expected!r} to {actual!r}; "
                    "re-compile to trace with the new value."
                )
        return execute(program, *(bound_args[name] for name in input_names))

    return wrapper
