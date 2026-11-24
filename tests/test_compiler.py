import contextlib
import io

import numpy as np

from nanojit import Expr, TensorSpec, trace


def tensor(name, shape, dtype=np.float32):
    return Expr(op="input", name=name, spec=TensorSpec(shape=shape, dtype=dtype))


def ir(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        trace(fn, *args, **kwargs)
    return buf.getvalue()


def test_example_array_becomes_input_with_spec():
    captured = {}

    def model(x):
        captured["spec"] = x.spec
        return x * 2

    ir(model, np.zeros((2, 3), dtype=np.float32))
    assert captured["spec"].shape == (2, 3)
    assert captured["spec"].dtype == np.float32


def test_default_stays_constant():
    def model(x, scale=2):
        return x * scale

    out = ir(model, np.zeros((2, 3), dtype=np.float32))
    assert "input x" in out
    assert "input scale" not in out
    assert "= 2" in out


def test_scalar_and_container_args_stay_constant():
    def model(x, n, cfg):
        return x * cfg["scale"] + n

    out = ir(model, np.zeros((2, 3)), 10, {"scale": 4})
    assert "input x" in out
    assert "input n" not in out
    assert "input cfg" not in out
    assert "= 10" in out and "= 4" in out


def test_missing_example_raises():
    def model(x, y):
        return x + y

    try:
        ir(model, np.zeros(3))
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for missing example argument")


def test_add_broadcasts():
    a, b = tensor("a", (2, 3)), tensor("b", (3,))
    c = a + b
    assert c.op == "add"
    assert c.spec.shape == (2, 3)


def test_mul_broadcasts_scalar():
    a = tensor("a", (2, 3))
    c = a * 2
    assert c.op == "mul"
    assert c.spec.shape == (2, 3)


def test_reflected_mul_broadcasts():
    a = tensor("a", (4, 1))
    c = 3 * a
    assert c.op == "mul"
    assert c.spec.shape == (4, 1)


def test_mul_broadcasts_leading_dims():
    a, b = tensor("a", (5, 1, 3)), tensor("b", (2, 1))
    c = a * b
    assert c.spec.shape == (5, 2, 3)


def test_incompatible_broadcast_raises():
    a, b = tensor("a", (2, 3)), tensor("b", (4,))
    try:
        a + b
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for incompatible shapes")


def test_matmul_broadcasts_batch_dims():
    a, b = tensor("a", (5, 2, 3)), tensor("b", (3, 4))
    c = a @ b
    assert c.op == "matmul"
    assert c.spec.shape == (5, 2, 4)


def test_matmul_dtype_is_promoted():
    a = tensor("a", (2, 3), np.float32)
    b = tensor("b", (3, 4), np.float64)
    assert (a @ b).spec.dtype == np.float64


if __name__ == "__main__":
    test_example_array_becomes_input_with_spec()
    test_default_stays_constant()
    test_scalar_and_container_args_stay_constant()
    test_missing_example_raises()
    test_add_broadcasts()
    test_mul_broadcasts_scalar()
    test_reflected_mul_broadcasts()
    test_mul_broadcasts_leading_dims()
    test_incompatible_broadcast_raises()
    test_matmul_broadcasts_batch_dims()
    test_matmul_dtype_is_promoted()
    print("ok")
