import numpy as np

from .compiler import trace


def model(x, y):
    return x * 2 + 3 * y


if __name__ == "__main__":
    trace(model, np.array(2.0), 5)
