"""Compatibility shims for the Triton Metal (triton-msl) backend."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys


def prepare_metal_backend() -> None:
    """Seed triton-msl's toolchain identity cache on macOS 27+.

    triton-msl 0.3.0 fingerprints the Metal toolchain by parsing Mach-O load
    commands. macOS 27 ships load commands and frameworks it cannot resolve, so
    its identity check refuses every kernel ("unsupported native load command"
    / "ambiguous native rpath dependency provider"). The resulting digest is
    cached in ``_toolchain_contract._snapshot``; derive it from the compiler and
    SDK version text instead, so codegen can proceed.

    No-op when triton-msl is absent or on older macOS.
    """
    # If a second Metal backend (e.g. triton-apple-backend) is still installed,
    # Triton refuses to choose between active drivers. This project uses
    # triton-msl, so pin it explicitly; an existing setting still wins.
    if sys.platform == "darwin":
        os.environ.setdefault("TRITON_DEFAULT_BACKEND", "metal")

    major = platform.mac_ver()[0].split(".")[0]
    if not major.isdigit() or int(major) < 27:
        return
    try:
        import triton_msl.backend._toolchain_contract as tc
    except Exception:
        return
    if getattr(tc, "_snapshot", None) is not None:
        return
    try:
        version = subprocess.check_output(
            ["xcrun", "-sdk", "macosx", "metal", "--version"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        sdk = subprocess.check_output(
            ["xcrun", "-sdk", "macosx", "--show-sdk-build-version"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return
    digest = hashlib.sha256((version + "\0" + sdk).encode()).hexdigest()
    selection = tuple(os.environ.get(key) for key in tc._SELECTION)
    tc._snapshot = (selection, digest)
