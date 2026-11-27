from dataclasses import FrozenInstanceError

import numpy as np
from numpy.typing import DTypeLike

from nanojit import Expr, Program, TensorSpec, lower_to_ir, trace


def tensor(name: str, shape: tuple[int, ...], dtype: DTypeLike = np.float32) -> Expr:
    return Expr(op="input", name=name, spec=TensorSpec(shape=shape, dtype=dtype))


# --- trace: inputs vs constants ------------------------------------------------


def test_trace_array_is_input() -> None:
    program = trace(lambda x: x * 2, np.zeros((2, 3), dtype=np.float32))
    assert isinstance(program, Program)
    assert program.instructions[0].op == "input"
    assert program.instructions[0].spec == TensorSpec((2, 3), np.float32)


def test_trace_default_is_constant() -> None:
    def model(x: Expr, scale: int = 2) -> Expr:
        return x * scale

    program = trace(model, np.zeros((2, 3), dtype=np.float32))
    assert sum(i.op == "input" for i in program.instructions) == 1
    consts = [i.inputs[0] for i in program.instructions if i.op == "const"]
    assert any(int(c) == 2 for c in consts)


def test_trace_passed_scalar_is_constant() -> None:
    def model(x: Expr, scale: int) -> Expr:
        return x * scale

    program = trace(model, np.zeros(3), 3)
    assert sum(i.op == "input" for i in program.instructions) == 1
    consts = [i.inputs[0] for i in program.instructions if i.op == "const"]
    assert any(int(c) == 3 for c in consts)


def test_trace_list_and_dict_are_constants() -> None:
    def model(x: Expr, n: int, cfg: dict[str, int]) -> Expr:
        return x * cfg["scale"] + n

    program = trace(model, np.zeros((2, 3)), 10, {"scale": 4})
    assert sum(i.op == "input" for i in program.instructions) == 1
    consts = [int(i.inputs[0]) for i in program.instructions if i.op == "const"]
    assert 10 in consts and 4 in consts


def test_trace_missing_example_raises() -> None:
    def model(x: Expr, y: Expr) -> Expr:
        return x + y

    try:
        trace(model, np.zeros(3))
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for missing example argument")


def test_trace_rejects_unsupported_parameter_kinds() -> None:
    def model(x: Expr, *, scale: int = 2) -> Expr:
        return x * scale

    try:
        trace(model, np.zeros(3))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for keyword-only parameter")


def test_trace_preserves_parameter_register_order() -> None:
    def model(x: Expr, y: Expr) -> Expr:
        return y + x

    program = trace(
        model, np.zeros((2, 3), dtype=np.float32), np.zeros((3,), dtype=np.float64)
    )
    assert program.instructions[0].op == "input"
    assert program.instructions[0].output == "%0"
    assert program.instructions[0].spec == TensorSpec((2, 3), np.float32)
    assert program.instructions[1].op == "input"
    assert program.instructions[1].output == "%1"
    assert program.instructions[1].spec == TensorSpec((3,), np.float64)
    assert program.instructions[2].op == "add"
    assert program.instructions[2].inputs == ("%1", "%0")


def test_trace_includes_unused_array_inputs() -> None:
    def model(x: Expr, y: Expr) -> Expr:
        return y * 2

    program = trace(
        model, np.zeros((2,), dtype=np.float32), np.zeros((3,), dtype=np.float32)
    )
    inputs = [i for i in program.instructions if i.op == "input"]
    assert [i.output for i in inputs] == ["%0", "%1"]
    assert [i.spec.shape for i in inputs] == [(2,), (3,)]


def test_trace_rejects_expr_argument() -> None:
    def model(x: Expr) -> Expr:
        return x + 1

    try:
        trace(model, tensor("x", (2,)))
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError when passing Expr to trace")


# --- shape / dtype inference ---------------------------------------------------


def test_tensor_spec_is_frozen_and_comparable() -> None:
    spec = TensorSpec((2, 3), np.float32)
    assert spec == TensorSpec((2, 3), np.float32)
    try:
        spec.shape = (4,)  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("expected TensorSpec to be immutable")


def test_add_broadcasts() -> None:
    a, b = tensor("a", (2, 3)), tensor("b", (3,))
    c = a + b
    assert c.op == "add"
    assert c.spec is not None and c.spec.shape == (2, 3)


def test_mul_broadcasts_scalar() -> None:
    a = tensor("a", (2, 3))
    c = a * 2
    assert c.op == "mul"
    assert c.spec is not None and c.spec.shape == (2, 3)


def test_reflected_mul_broadcasts() -> None:
    a = tensor("a", (4, 1))
    c = 3 * a
    assert c.op == "mul"
    assert c.spec is not None and c.spec.shape == (4, 1)


def test_mul_broadcasts_leading_dims() -> None:
    a, b = tensor("a", (5, 1, 3)), tensor("b", (2, 1))
    spec = (a * b).spec
    assert spec is not None and spec.shape == (5, 2, 3)


def test_incompatible_broadcast_raises() -> None:
    a, b = tensor("a", (2, 3)), tensor("b", (4,))
    try:
        a + b
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for incompatible shapes")


def test_matmul_broadcasts_batch_dims() -> None:
    a, b = tensor("a", (5, 2, 3)), tensor("b", (3, 4))
    c = a @ b
    assert c.op == "matmul"
    assert c.spec is not None and c.spec.shape == (5, 2, 4)


def test_matmul_dtype_is_promoted() -> None:
    a = tensor("a", (2, 3), np.float32)
    b = tensor("b", (3, 4), np.float64)
    spec = (a @ b).spec
    assert spec is not None and spec.dtype == np.float64


# --- lowering to SSA IR --------------------------------------------------------


def test_from_numpy_sets_op() -> None:
    expr = Expr.from_numpy(np.zeros(3), op="input", name="x")
    assert expr.op == "input"
    assert expr.name == "x"


def test_lower_to_ir_builds_ssa_program() -> None:
    x = tensor("x", (2, 3))
    program = lower_to_ir(x * 2 + 3)
    assert [i.op for i in program.instructions] == [
        "input",
        "const",
        "mul",
        "const",
        "add",
    ]
    assert [i.output for i in program.instructions] == ["%0", "%1", "%2", "%3", "%4"]
    assert program.output == "%4"
    assert program.instructions[0].spec == TensorSpec((2, 3), np.float32)
    assert program.instructions[-1].spec.shape == (2, 3)


def test_lower_to_ir_reuses_subexpressions() -> None:
    x = tensor("x", (2, 3))
    y = x * 2
    program = lower_to_ir(y + y)
    assert [i.op for i in program.instructions] == ["input", "const", "mul", "add"]
    assert program.instructions[-1].inputs == ("%2", "%2")


def test_const_instruction_stores_literal_not_result() -> None:
    x = tensor("x", (2,))
    program = lower_to_ir(x + 5)
    const = program.instructions[1]
    assert const.op == "const"
    assert int(const.inputs[0]) == 5
    add = program.instructions[2]
    assert add.op == "add"
    assert add.inputs == ("%0", "%1")


def test_program_str_renders_inputs_and_instructions() -> None:
    x = tensor("x", (2, 3))
    text = str(lower_to_ir(x * 2))
    assert "inputs:" in text
    assert "%0: float32[2, 3]" in text
    assert "const 2 : int64[]" in text
    assert "mul %0, %1 : float64[2, 3]" in text
    assert text.rstrip().endswith("return %2")


if __name__ == "__main__":
    test_trace_array_is_input()
    test_trace_default_is_constant()
    test_trace_passed_scalar_is_constant()
    test_trace_list_and_dict_are_constants()
    test_trace_missing_example_raises()
    test_trace_rejects_unsupported_parameter_kinds()
    test_trace_preserves_parameter_register_order()
    test_trace_includes_unused_array_inputs()
    test_trace_rejects_expr_argument()
    test_tensor_spec_is_frozen_and_comparable()
    test_add_broadcasts()
    test_mul_broadcasts_scalar()
    test_reflected_mul_broadcasts()
    test_mul_broadcasts_leading_dims()
    test_incompatible_broadcast_raises()
    test_matmul_broadcasts_batch_dims()
    test_matmul_dtype_is_promoted()
    test_from_numpy_sets_op()
    test_lower_to_ir_builds_ssa_program()
    test_lower_to_ir_reuses_subexpressions()
    test_const_instruction_stores_literal_not_result()
    test_program_str_renders_inputs_and_instructions()
    print("ok")
