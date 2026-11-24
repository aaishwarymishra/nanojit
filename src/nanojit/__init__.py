from .compiler import Expr, TensorSpec, print_ir, trace

__all__ = ["Expr", "TensorSpec", "print_ir", "trace"]


def main() -> None:
    print("nanojit: a tiny ML compiler")
