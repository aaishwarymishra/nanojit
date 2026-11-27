from typing import Any

import numpy as np

from nanojit.backend import compile


def model(x: Any, y: Any) -> Any:
    return x * 2 + 3 * y


if __name__ == "__main__":
    x = np.ones((2, 3), dtype=np.float32)
    y = np.ones((3,), dtype=np.float32)
    fn = compile(model, x, y)
    result = fn(x, y)
    assert np.allclose(result, model(x, y))
    print(result)
