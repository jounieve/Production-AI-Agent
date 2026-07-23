"""
conftest.py (repo root) - makes `src/` importable for every test module,
regardless of which directory `pytest` is invoked from.

Without this file at the repo root, `python -m pytest tests/test_security.py`
run from the repository root fails with:
    ModuleNotFoundError: No module named 'guardrails'
because pytest only auto-discovers conftest.py files that live in the test
file's own directory or one of its ANCESTOR directories. `tests/` and `src/`
are sibling directories, so a conftest.py placed inside `src/` (as it
previously was) is invisible to modules under `tests/`. Placing it here, at
the repository root (a common ancestor of both `tests/` and `src/`), fixes
that for every current and future test module.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that load ML models (deselect with -m 'not slow')",
    )