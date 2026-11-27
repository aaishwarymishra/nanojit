from .backend import compile, execute
from .compiler import (
    Expr,
    Instruction,
    Program,
    TensorSpec,
    lower_to_ir,
    trace,
)

__all__ = [
    "Expr",
    "Instruction",
    "Program",
    "TensorSpec",
    "compile",
    "execute",
    "lower_to_ir",
    "trace",
]


def main() -> None:
    print("nanojit: a tiny ML compiler")
