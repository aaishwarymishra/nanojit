"""Run the cross-framework benchmarks.

uv run python -m benchmarks
uv run python -m benchmarks add --sizes 1048576,8388608
uv run python -m benchmarks softmax --backends mlx,metal,triton,torch
"""

from __future__ import annotations

import argparse
import time

from benchmarks.harness import measure, print_table
from benchmarks.kernels import KERNELS

COOLDOWN_MS = 100  # idle between tests, so each starts from a similar state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m benchmarks")
    parser.add_argument(
        "kernel", nargs="?", choices=sorted(KERNELS), help="default: all"
    )
    parser.add_argument("--sizes", help="comma-separated element counts")
    parser.add_argument("--backends", help="comma-separated backend names")
    args = parser.parse_args(argv)

    selected = [args.kernel] if args.kernel else sorted(KERNELS)
    backends = set(args.backends.split(",")) if args.backends else None

    for name in selected:
        module = KERNELS[name]
        sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else module.SIZES
        rows = []
        for elements in sorted(sizes):
            for case in module.cases():
                if backends is not None and case.name not in backends:
                    continue
                rows.append(measure(case, elements))
                time.sleep(COOLDOWN_MS / 1e3)
        print_table(module.TITLE, rows)


if __name__ == "__main__":
    main()
