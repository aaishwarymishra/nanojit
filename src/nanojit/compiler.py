import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

type ExprLike = Expr | int | float | NDArray[Any]


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: DTypeLike = np.float32

    def infer_broadcast_shape(self, other: TensorSpec) -> tuple[int, ...]:
        """Infer the broadcasted shape of two tensor specs."""
        if len(self.shape) > len(other.shape):
            longer, shorter = self.shape, other.shape
        else:
            longer, shorter = other.shape, self.shape

        # Pad the shorter shape with ones on the left
        padded_shorter = (1,) * (len(longer) - len(shorter)) + shorter

        broadcast_shape = [1] * len(longer)
        for index, (dim_longer, dim_shorter) in enumerate(zip(longer, padded_shorter)):
            if dim_longer == dim_shorter or dim_shorter == 1:
                broadcast_shape[index] = dim_longer
            elif dim_longer == 1:
                broadcast_shape[index] = dim_shorter
            else:
                raise ValueError(
                    f"Shapes {self.shape} and {other.shape} are not broadcastable."
                )

        return tuple(broadcast_shape)

    def infer_matmul_shape(self, other: TensorSpec) -> tuple[int, ...]:
        """Infer the shape of the result of a matrix multiplication."""
        if len(self.shape) < 2 or len(other.shape) < 2:
            raise ValueError(
                f"Both tensors must be at least 2D for matmul. Got {self.shape} and {other.shape}."
            )
        if self.shape[-1] != other.shape[-2]:
            raise ValueError(
                f"Incompatible shapes for matmul: {self.shape} and {other.shape}."
            )

        # The resulting shape is the broadcasted shape of the batch dimensions
        # followed by the last two dimensions of the result
        batch_shape_self = self.shape[:-2]
        batch_shape_other = other.shape[:-2]
        broadcasted_batch_shape = TensorSpec(batch_shape_self).infer_broadcast_shape(
            TensorSpec(batch_shape_other)
        )
        return broadcasted_batch_shape + (self.shape[-2], other.shape[-1])

    def broadcast(self, other: TensorSpec) -> TensorSpec:
        return TensorSpec(
            shape=self.infer_broadcast_shape(other),
            dtype=np.result_type(self.dtype, other.dtype),
        )

    def matmul(self, other: TensorSpec) -> TensorSpec:
        return TensorSpec(
            shape=self.infer_matmul_shape(other),
            dtype=np.result_type(self.dtype, other.dtype),
        )


@dataclass(eq=False)
class Expr:
    op: str
    inputs: tuple[Expr, ...] = ()
    value: NDArray[Any] | None = None
    name: str | None = None
    spec: TensorSpec | None = None

    @staticmethod
    def from_numpy(array: NDArray[Any], op: str, name: str | None = None) -> Expr:
        if not isinstance(array, np.ndarray):
            raise TypeError("Input must be a numpy array")
        return Expr(
            op=op,
            value=array,
            name=name,
            spec=TensorSpec(shape=array.shape, dtype=array.dtype),
        )

    @staticmethod
    def as_expr(value: Any) -> Expr:
        if isinstance(value, Expr):
            return value
        if not isinstance(value, (int, float, np.ndarray)):
            raise TypeError(
                f"Cannot convert {type(value)} to Expr. Must be int, float, or np.ndarray."
            )
        array = np.array(value)
        return Expr.from_numpy(array, op="const", name=None)

    def _binary(self, other: ExprLike, op: str, *, reflected: bool = False) -> Expr:
        other_expr = self.as_expr(other)
        lhs, rhs = (other_expr, self) if reflected else (self, other_expr)
        if lhs.spec is None or rhs.spec is None:
            raise ValueError("Expr must have a TensorSpec for shape inference.")
        spec = (
            lhs.spec.matmul(rhs.spec)
            if op == "matmul"
            else lhs.spec.broadcast(rhs.spec)
        )
        return Expr(op=op, inputs=(lhs, rhs), spec=spec)

    def __add__(self, other: ExprLike) -> Expr:
        return self._binary(other, "add")

    def __radd__(self, other: ExprLike) -> Expr:
        return self._binary(other, "add", reflected=True)

    def __sub__(self, other: ExprLike) -> Expr:
        return self._binary(other, "sub")

    def __rsub__(self, other: ExprLike) -> Expr:
        return self._binary(other, "sub", reflected=True)

    def __mul__(self, other: ExprLike) -> Expr:
        return self._binary(other, "mul")

    def __rmul__(self, other: ExprLike) -> Expr:
        return self._binary(other, "mul", reflected=True)

    def __matmul__(self, other: ExprLike) -> Expr:
        return self._binary(other, "matmul")

    def __rmatmul__(self, other: ExprLike) -> Expr:
        return self._binary(other, "matmul", reflected=True)


@dataclass
class Instruction:
    output: str
    op: str
    inputs: tuple[str | NDArray[Any], ...]
    spec: TensorSpec
    name: str | None = None


@dataclass
class Program:
    instructions: list[Instruction]
    output: str

    def __str__(self) -> str:
        inputs = [i for i in self.instructions if i.op == "input"]
        lines = ["inputs:"]
        lines += [f"    {i.output}: {_format_spec(i.spec)}" for i in inputs]
        lines += ["", "instructions:"]
        for i in self.instructions:
            if i.op == "input":
                continue
            if i.op == "const":
                rhs = "const " + ", ".join(str(v) for v in i.inputs)
            else:
                rhs = f"{i.op} " + ", ".join(str(v) for v in i.inputs)
            lines.append(f"    {i.output} = {rhs} : {_format_spec(i.spec)}")
        lines.append(f"return {self.output}")
        return "\n".join(lines)


def _format_spec(spec: TensorSpec) -> str:
    dims = ", ".join(str(d) for d in spec.shape)
    return f"{np.dtype(spec.dtype).name}[{dims}]"


def _require_spec(node: Expr) -> TensorSpec:
    if node.spec is None:
        raise ValueError(f"Expr {node.op!r} is missing a TensorSpec.")
    return node.spec


def lower_to_ir(root_node: Expr, input_nodes: Sequence[Expr] | None = None) -> Program:
    instructions: list[Instruction] = []
    registers: dict[int, str] = {}

    for node in input_nodes or ():
        node_id = id(node)
        if node_id not in registers:
            output = f"%{len(registers)}"
            registers[node_id] = output
            instructions.append(
                Instruction(
                    output=output,
                    op=node.op,
                    inputs=(),
                    spec=_require_spec(node),
                    name=node.name,
                )
            )

    def visit(node: Expr) -> str:
        node_id = id(node)
        if node_id in registers:
            return registers[node_id]

        preds = tuple(visit(arg) for arg in node.inputs)
        output = f"%{len(registers)}"
        registers[node_id] = output

        if node.op == "const":
            if node.value is None:
                raise ValueError("Const Expr is missing its value.")
            inputs: tuple[str | NDArray[Any], ...] = (node.value,)
        else:
            inputs = preds
        instructions.append(
            Instruction(
                output=output,
                op=node.op,
                inputs=inputs,
                spec=_require_spec(node),
                name=node.name,
            )
        )
        return output

    output = visit(root_node)
    return Program(instructions=instructions, output=output)


def _to_input(name: str, value: Any, input_nodes: list[Expr]) -> Any:
    if isinstance(value, Expr):
        raise TypeError(f"Function {name} has an argument that is already an Expr.")
    value = np.array(value) if not isinstance(value, np.ndarray) else value
    expr = Expr.from_numpy(value, op="input", name=name)
    input_nodes.append(expr)
    return expr


def trace(fn: Callable[..., Expr], *args: Any, **kwargs: Any) -> Program:
    """Trace ``fn`` with example arguments.

    Array arguments become symbolic inputs; everything else (scalars, lists,
    dicts, defaults) stays a constant. Literals in the body stay constants too.
    Returns the lowered ``Program``.
    """

    signature = inspect.signature(fn)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()

    arguments: dict[str, Any] = {}
    input_nodes: list[Expr] = []
    for name, param in signature.parameters.items():
        if param.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            raise ValueError(
                f"Function {fn.__name__} has unsupported parameter kind: {param.kind}"
            )
        arguments[name] = _to_input(name, bound.arguments[name], input_nodes)

    return lower_to_ir(fn(**arguments), input_nodes)
