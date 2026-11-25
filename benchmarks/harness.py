"""Batched cross-framework timing: enqueue N calls, sync once, divide.

Per-call sync costs ~0.2-0.4 ms here, which swamps every kernel below ~1M
elements, so the harness batches and syncs once per trial. That is also the only
number comparable across MLX, Metal, Triton and torch MPS:
``triton.testing.do_bench`` reads the Triton driver queue, so it reports ~0.07 ms
for MLX at every size.

Sleep only between tests (``COOLDOWN_MS``), never between calls: an idle GPU
downclocks and the next call starts slower.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

WARMUP_MS = 100
REP_MS = 150
TRIALS = 5


@dataclass(frozen=True)
class Case:
    """One backend's version of a kernel."""

    name: str
    make_inputs: Callable[[int], tuple[Any, Any]]
    # Enqueue the work. Must not synchronize: the harness syncs once per trial.
    run: Callable[[Any, Any], Any]
    sync: Callable[[], None]
    check: Callable[[Any, Any, Any], None] | None = None
    bytes_per_element: int = 12  # 2 reads + 1 write, fp32
    align: int = 1  # some kernels need a size multiple of this


@dataclass(frozen=True)
class Measurement:
    elements: int
    name: str
    mean_ms: float
    median_ms: float
    min_ms: float
    gbps: float


def measure(case: Case, elements: int) -> Measurement | None:
    """Time ``case`` on ``elements`` elements. Returns None if unsupported."""
    if elements % case.align:
        return None

    x, y = case.make_inputs(elements)
    out = case.run(x, y)  # also compiles JIT kernels
    case.sync()
    if case.check is not None:
        case.check(out, x, y)

    # Batch size: aim for REP_MS per trial. Probe a few calls so launch cost is
    # amortised before the sync; otherwise a tiny kernel looks like pure sync.
    case.sync()
    start = time.perf_counter()
    for _ in range(5):
        case.run(x, y)
    case.sync()
    per_call_ms = max((time.perf_counter() - start) / 5 * 1e3, 1e-5)
    iters = max(5, min(int(REP_MS / per_call_ms), 5000))

    deadline = time.perf_counter() + WARMUP_MS / 1e3
    while time.perf_counter() < deadline:
        for _ in range(iters):
            case.run(x, y)
        case.sync()

    samples = []
    for _ in range(TRIALS):
        case.sync()
        start = time.perf_counter()
        for _ in range(iters):
            case.run(x, y)
        case.sync()
        samples.append((time.perf_counter() - start) / iters * 1e3)

    samples.sort()
    mean_ms = sum(samples) / len(samples)
    return Measurement(
        elements=elements,
        name=case.name,
        mean_ms=mean_ms,
        median_ms=samples[len(samples) // 2],
        min_ms=samples[0],
        gbps=elements * case.bytes_per_element / 1e9 / (mean_ms / 1e3),
    )


def _ms(value: float) -> str:
    return f"{value:.4f}" if value < 1 else f"{value:.3f}"


def print_table(title: str, rows: list[Measurement]) -> None:
    """Markdown table, fastest mean per size in bold."""
    if not rows:
        print(f"\n{title}: nothing to measure\n")
        return

    best = {r.elements: min(m.mean_ms for m in rows if m.elements == r.elements) for r in rows}
    torch_ms = {r.elements: r.mean_ms for r in rows if r.name == "torch"}
    has_torch = bool(torch_ms)

    header = "| elements | backend | mean ms | median ms | min ms | GB/s |"
    rule = "|---:|---|---:|---:|---:|---:|"
    if has_torch:
        header += " vs torch |"
        rule += "---:|"

    print(f"\n### {title}\n")
    print(header)
    print(rule)
    for r in rows:
        mean = f"**{_ms(r.mean_ms)}**" if r.mean_ms == best[r.elements] else _ms(r.mean_ms)
        line = (
            f"| {r.elements:,} | `{r.name}` | {mean} | {_ms(r.median_ms)} "
            f"| {_ms(r.min_ms)} | {r.gbps:.1f} |"
        )
        if has_torch:
            line += f" {torch_ms[r.elements] / r.mean_ms:.2f}x |"
        print(line)
    print()
