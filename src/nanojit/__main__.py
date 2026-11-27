import numpy as np

from .compiler import Expr, trace


def model(x: Expr, y: Expr) -> Expr:
    return x * 2 + 3 * y


if __name__ == "__main__":
    x = np.ones((2, 3), dtype=np.float32)
    y = np.ones((3,), dtype=np.float32)
    print(trace(model, x, y))
