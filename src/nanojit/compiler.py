import inspect
from dataclasses import dataclass


@dataclass(eq=False)
class Expr:
    op: str
    inputs: tuple = ()
    value: any = None
    name: str | None = None

    @staticmethod
    def as_expr(value) -> "Expr":
        if isinstance(value, Expr):
            return value
        return Expr(op="const", value=value)

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


def trace(fn: callable):

    signature = inspect.signature(fn)
    arguments = {}

    for name, param in signature.parameters.items():
        if param.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise ValueError(
                f"Function {fn.__name__} has unsupported parameter kind: {param.kind}"
            )
        arguments[name] = Expr(op="input", name=name)

    result = fn(**arguments)
    print_ir(result)



