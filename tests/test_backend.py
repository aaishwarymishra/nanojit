from typing import Any

import numpy as np

from nanojit import (
    Expr,
    Instruction,
    Program,
    TensorSpec,
    compile,
    execute,
    lower_to_ir,
    trace,
)


def test_execute_ops_and_broadcasting() -> None:
    def model(x: Any, y: Any, z: Any) -> Any:
        return (x @ y) * 2 + 3 * z - 1

    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    y = np.arange(12, dtype=np.float32).reshape(3, 4)
    z = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)

    np.testing.assert_allclose(execute(trace(model, x, y, z), x, y, z), model(x, y, z))


def test_execute_reused_unused_reordered_and_zero_d() -> None:
    def reordered(x: Any, y: Any) -> Any:
        sub = y * 2
        return sub + sub + x

    a = np.array([[1.0, 2.0]], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(execute(trace(reordered, a, b), a, b), reordered(a, b))

    def unused(x: Any, y: Any) -> Any:
        return y * 2

    np.testing.assert_allclose(execute(trace(unused, a, b), a, b), unused(a, b))

    s = np.array(2.5, dtype=np.float32)
    out = execute(trace(lambda v: v + 1.0, s), s)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, s + 1.0)

    expr = Expr(op="input", spec=TensorSpec((2,), np.float32))
    arr = np.array([2.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(execute(lower_to_ir(3 * expr), arr), 3 * arr)


def test_compile_binds_keywords_and_defaults() -> None:
    def model(x: Any, cfg: dict[str, int], scale: int = 2) -> Any:
        return x * scale + cfg["bias"]

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    fn = compile(model, x, {"bias": 5})

    np.testing.assert_allclose(fn(x, {"bias": 5}), model(x, {"bias": 5}))
    np.testing.assert_allclose(fn(cfg={"bias": 5}, x=x * 2), model(x * 2, {"bias": 5}))


def test_execute_validates_inputs_and_ops() -> None:
    x = np.zeros((2, 3), dtype=np.float32)
    program = trace(lambda a: a + 1, x)

    for bad_args, exc_type in [
        ((), ValueError),
        ((x, x), ValueError),
        (([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],), TypeError),
        ((np.zeros((3, 2), dtype=np.float32),), ValueError),
        ((np.zeros((2, 3), dtype=np.float64),), ValueError),
    ]:
        try:
            execute(program, *bad_args)
        except exc_type:
            pass
        else:
            raise AssertionError(f"expected {exc_type.__name__} for {bad_args!r}")

    bad_program = Program(
        instructions=[
            Instruction(
                output="%0", op="input", inputs=(), spec=TensorSpec((2,), np.float32)
            ),
            Instruction(
                output="%1",
                op="unknown_op",
                inputs=("%0",),
                spec=TensorSpec((2,), np.float32),
            ),
        ],
        output="%1",
    )
    try:
        execute(bad_program, np.zeros((2,), dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unsupported op")
