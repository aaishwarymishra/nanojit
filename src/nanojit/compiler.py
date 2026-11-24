import inspect
from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)
class TensorSpec:
    shape: tuple
    dtype: np.dtype = np.float32


@dataclass(eq=False)
class Expr:
    op: str
    inputs: tuple = ()
    value: any = None
    name: str | None = None
    spec: TensorSpec | None = None

    @staticmethod
    def from_numpy(array, name=None) -> "Expr":

        if not isinstance(array, np.ndarray):
            raise TypeError("Input must be a numpy array")
        return Expr(
            op="const",
            value=array,
            name=name,
            spec=TensorSpec(shape=array.shape, dtype=array.dtype),
        )

    @staticmethod
    def as_expr(value) -> "Expr":
        if isinstance(value, Expr):
            return value
        if not isinstance(value, (int, float, np.ndarray)):
            raise TypeError(
                f"Cannot convert {type(value)} to Expr. Must be int, float, or np.ndarray."
            )
        array = np.array(value)
        return Expr.from_numpy(array)

    def __add__(self, other):
        return Expr(op="add", inputs=(self, self.as_expr(other)))

    def __radd__(self, other):
        return Expr(op="add", inputs=(self.as_expr(other), self))

    def __sub__(self, other):
        return Expr(op="sub", inputs=(self, self.as_expr(other)))

    def __rsub__(self, other):
        return Expr(op="sub", inputs=(self.as_expr(other), self))

    def __mul__(self, other):
        return Expr(op="mul", inputs=(self, self.as_expr(other)))

    def __rmul__(self, other):
        return Expr(op="mul", inputs=(self.as_expr(other), self))


def print_ir(root_node: Expr, indent=0):
    lines = []
    registers = {}

    def visit(node: Expr) -> str:

        node_id = id(node)

        if node_id in registers:
            return registers[node_id]

        args = [visit(arg) for arg in node.inputs]

        result = f"%{len(registers)}"
        registers[node_id] = result

        if node.op == "const":
            lines.append(f"{result} = {node.value}")
        elif node.op == "input":
            lines.append(f"{result} = input {node.name}")
        else:
            lines.append(f"{result} = {node.op} " + " ".join(args))

        return result

    output = visit(root_node)
    lines.append(f"return {output}")
    print("\n".join(lines))


def trace(fn: callable, *args, **kwargs):
    """Trace ``fn`` by calling it with the arguments you give it.

    Arguments are used as-is, so plain Python values (scalars, lists, dicts,
    defaults) stay constant in the IR. Pass ``Expr(op="input", name=...)`` for a
    value you want to become a symbolic input. Required arguments you leave out
    are treated as inputs, so ``trace(model)`` still works for ``def model(x)``.
    """

    signature = inspect.signature(fn)
    bound = signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()

    arguments = {}
    for name, param in signature.parameters.items():
        if param.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise ValueError(
                f"Function {fn.__name__} has unsupported parameter kind: {param.kind}"
            )
        arguments[name] = bound.arguments.get(name, Expr(op="input", name=name))

    result = fn(*arguments.values())
    print_ir(result)
