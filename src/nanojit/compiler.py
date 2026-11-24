import inspect
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple
    dtype: np.dtype = np.float32

    def infer_broadcast_shape(self, other: "TensorSpec") -> tuple:
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

    def infer_matmul_shape(self, other: "TensorSpec") -> tuple:
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

    def broadcast(self, other: "TensorSpec") -> "TensorSpec":
        return TensorSpec(
            shape=self.infer_broadcast_shape(other),
            dtype=np.result_type(self.dtype, other.dtype),
        )

    def matmul(self, other: "TensorSpec") -> "TensorSpec":
        return TensorSpec(
            shape=self.infer_matmul_shape(other),
            dtype=np.result_type(self.dtype, other.dtype),
        )


@dataclass(eq=False)
class Expr:
    op: str
    inputs: tuple = ()
    value: any = None
    name: str | None = None
    spec: TensorSpec | None = None

    @staticmethod
    def from_numpy(array, op: str, name=None) -> "Expr":

        if not isinstance(array, np.ndarray):
            raise TypeError("Input must be a numpy array")
        return Expr(
            op=op,
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
        return Expr.from_numpy(array, op="const", name=None)

    def __add__(self, other):
        other = self.as_expr(other)
        return Expr(
            op="add", inputs=(self, other), spec=self.spec.broadcast(other.spec)
        )

    def __radd__(self, other):
        other = self.as_expr(other)
        return Expr(
            op="add", inputs=(other, self), spec=other.spec.broadcast(self.spec)
        )

    def __sub__(self, other):
        other = self.as_expr(other)
        return Expr(
            op="sub", inputs=(self, other), spec=self.spec.broadcast(other.spec)
        )

    def __rsub__(self, other):
        other = self.as_expr(other)
        return Expr(
            op="sub", inputs=(other, self), spec=other.spec.broadcast(self.spec)
        )

    def __mul__(self, other):
        other = self.as_expr(other)
        return Expr(
            op="mul", inputs=(self, other), spec=self.spec.broadcast(other.spec)
        )

    def __rmul__(self, other):
        other = self.as_expr(other)
        return Expr(
            op="mul", inputs=(other, self), spec=other.spec.broadcast(self.spec)
        )

    def __matmul__(self, other):
        other = self.as_expr(other)
        return Expr(
            op="matmul", inputs=(self, other), spec=self.spec.matmul(other.spec)
        )

    def __rmatmul__(self, other):
        other = self.as_expr(other)
        return Expr(
            op="matmul", inputs=(other, self), spec=other.spec.matmul(self.spec)
        )


@dataclass
class Instruction:
    output: str
    op: str
    inputs: tuple
    spec: TensorSpec


@dataclass
class Program:
    instructions: list
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
                rhs = f"{i.op} " + ", ".join(i.inputs)
            lines.append(f"    {i.output} = {rhs} : {_format_spec(i.spec)}")
        lines.append(f"return {self.output}")
        return "\n".join(lines)


def _format_spec(spec: TensorSpec) -> str:
    dims = ", ".join(str(d) for d in spec.shape)
    return f"{np.dtype(spec.dtype).name}[{dims}]"


def lower_to_ir(root_node: Expr) -> Program:
    instructions = []
    registers = {}

    def visit(node: Expr) -> str:
        node_id = id(node)
        if node_id in registers:
            return registers[node_id]

        preds = tuple(visit(arg) for arg in node.inputs)
        output = f"%{len(registers)}"
        registers[node_id] = output

        inputs = (node.value,) if node.op == "const" else preds
        instructions.append(
            Instruction(output=output, op=node.op, inputs=inputs, spec=node.spec)
        )
        return output

    output = visit(root_node)
    return Program(instructions=instructions, output=output)


def _to_input(name: str, value):
    if isinstance(value, Expr):
        return value
    if isinstance(value, np.ndarray):
        return Expr.from_numpy(value, op="input", name=name)
    return value


def trace(fn: callable, *args, **kwargs) -> Program:
    """Trace ``fn`` with example arguments.

    Array arguments become symbolic inputs; everything else (scalars, lists,
    dicts, defaults) stays a constant. Literals in the body stay constants too.
    Returns the lowered ``Program``.
    """

    signature = inspect.signature(fn)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()

    arguments = {}
    for name, param in signature.parameters.items():
        if param.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            raise ValueError(
                f"Function {fn.__name__} has unsupported parameter kind: {param.kind}"
            )
        arguments[name] = _to_input(name, bound.arguments[name])

    return lower_to_ir(fn(**arguments))
