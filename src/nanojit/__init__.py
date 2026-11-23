from .compiler import Expr, print_ir, trace

__all__ = ["Expr", "print_ir", "trace"]


def main() -> None:
    print("nanojit: a tiny ML compiler")
