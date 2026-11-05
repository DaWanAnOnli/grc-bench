"""Tests for ./setup.sh, the setup check of a new machine (README, "Set up a new machine")."""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from grcbench.common.paths import GNURADIO_VERSION, GRC_AGENT_COMMIT, REPO
from grcbench.tier_a import isolation

SETUP = REPO / "setup.sh"


def _var(name: str) -> str:
    m = re.search(rf"^{name}=(\S+)$", SETUP.read_text(encoding="utf-8"), re.M)
    assert m, f"setup.sh sets no {name}"
    return m.group(1)


def test_setup_script_checks_the_pins_gb_enforces():
    assert _var("PIN_AGENT") == GRC_AGENT_COMMIT
    assert GNURADIO_VERSION.startswith(_var("PIN_GR"))


def test_setup_script_builds_every_database_the_runs_copy():
    text = SETUP.read_text(encoding="utf-8")
    for name in isolation.LEXICAL_DBS:
        assert name in text, name
    assert "GRC_EMBED_BACKEND=lexical" in text


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_setup_script_is_valid_bash_with_unix_line_endings():
    assert b"\r" not in SETUP.read_bytes()
    assert subprocess.run(["bash", "-n", str(SETUP)], capture_output=True).returncode == 0


def test_a_missing_knowledge_database_points_to_the_setup_script(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation, "VECTORS_SRC", tmp_path / "vectors")
    monkeypatch.setattr(isolation, "VECTORS_CACHE", tmp_path / "cache")
    with pytest.raises(isolation.IsolationError, match=r"\./setup\.sh"):
        isolation._ensure_vectors_cache()
