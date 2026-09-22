"""Regression guard: the package must not shadow stdlib modules.

Root cause (issue #61): ``voiceclone_sidecar/secrets.py`` shared its name
with the stdlib ``secrets`` module. When a worker subprocess got the package
directory injected into ``sys.path`` (e.g. running from a git worktree), the
package module shadowed the stdlib one and ``secrets.token_hex`` vanished.

Fix: the module was renamed to ``key_store``. This test keeps any stdlib-
shadowing module name from coming back.
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "voiceclone_sidecar"


def test_no_package_module_shadows_the_stdlib() -> None:
    # sys.stdlib_module_names is the authoritative top-level stdlib set for
    # the running interpreter — it reflects THIS Python version, so e.g.
    # `context` (removed from the stdlib in Python 3.0) is correctly absent
    # and a module of that name is not a shadow.
    stdlib = set(sys.stdlib_module_names)
    # Every package directory counts: a shadowing name in engines/ or
    # routers/ pollutes sys.path just as hard as a top-level one.
    names = {p.stem for p in PACKAGE_DIR.rglob("*.py")}
    clash = sorted(names & stdlib)
    assert clash == [], (
        f"module(s) {clash} shadow stdlib modules; subprocess sys.path "
        "injection makes the shadowing real (issue #61)"
    )


def test_key_store_imports_by_its_own_name() -> None:
    # The renamed module must import cleanly with the package directory
    # prepended to sys.path — exactly the worker-subprocess condition that
    # used to break on ``secrets``.
    import subprocess

    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import secrets; assert secrets.token_hex(4); "
        "from voiceclone_sidecar.key_store import KeyStore; print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(PACKAGE_DIR)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
