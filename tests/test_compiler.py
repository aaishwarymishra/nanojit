import contextlib
import io

from nanojit import Expr, trace


def ir(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        trace(fn, *args, **kwargs)
    return buf.getvalue()


def test_default_stays_constant():
    def model(x, scale=2):
        return x * scale

    out = ir(model, Expr(op="input", name="x"))
    assert "input x" in out
    assert "input scale" not in out
    assert "= 2" in out


def test_plain_and_container_args_stay_constant():
    def model(x, n, cfg):
        return x * cfg["scale"] + n

    out = ir(model, Expr(op="input", name="x"), 10, {"scale": 4})
    assert "input x" in out
    assert "input n" not in out
    assert "input cfg" not in out
    assert "= 10" in out and "= 4" in out


def test_missing_required_becomes_input():
    def model(x, scale=2):
        return x * scale

    out = ir(model)
    assert "input x" in out
    assert "input scale" not in out
