import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nanojit.compiler import Program, trace

ops = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "matmul": np.matmul,
}


def executor(program: Program, **arguments):
    input_nodes = [i for i in program.instructions if i.op == "input"]
    if len(input_nodes) != len(arguments):
        raise ValueError(
            f"Expected {len(input_nodes)} inputs, but got {len(arguments)}."
        )
    for i in input_nodes:
        if i.name not in arguments:
            raise ValueError(f"Input '{i.name}' not in arguments.")
        value = arguments[i.name]
        if not np.array_equal(value.shape, i.spec.shape):
            raise ValueError(
                f"Input '{i.name}' has shape {value.shape}, but expected {i.spec.shape}."
            )
        if value.dtype != i.spec.dtype:
            raise ValueError(
                f"Input '{i.name}' has dtype {value.dtype}, but expected {i.spec.dtype}."
            )

    env = {}
    for i in input_nodes:
        env[i.output] = arguments[i.name]
    for instr in program.instructions:
        if instr.op == "input":
            continue
        if instr.op == "const":
            env[instr.output] = instr.inputs[0]
            continue
        input_values = [env[i] for i in instr.inputs]
        op_func = ops[instr.op]
        env[instr.output] = op_func(*input_values)
    return env[program.output]


def compile(fn: Callable, *args: Any, **kwargs: Any) -> Program:
    """Compile ``fn`` with example arguments.

    This is a convenience wrapper around ``trace`` that also returns the lowered
    ``Program``.
    """
    program = trace(fn, *args, **kwargs)

    def wrapper(*args: Any, **kwargs: Any) -> NDArray[Any]:
        print(program)
        bound_args = inspect.signature(fn).bind(*args, **kwargs)
        bound_args.apply_defaults()
        return executor(program, **bound_args.arguments)

    return wrapper
