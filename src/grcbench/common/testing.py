"""`./gb test [paths] [pytest args]` — run the unit tests with the pinned interpreter.

With no path arguments the whole suite in tests/ runs; otherwise only the given files/directories.
"""

from __future__ import annotations

from pathlib import Path

from .paths import REPO


def main(argv: list[str]) -> int:
    import pytest
    has_path = any(not a.startswith("-") and (Path(a).exists() or "::" in a) for a in argv)
    targets = [] if has_path else [str(REPO / "tests")]
    return pytest.main([*targets, "-q", "-p", "no:cacheprovider", *argv])
