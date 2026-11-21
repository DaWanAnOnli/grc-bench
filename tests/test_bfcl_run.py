"""The BFCL runs (pre-registration §6, eval plan Table 6): the pinned package, the categories and settings the
pre-registration states, and the run list's BFCL configurations. bfcl-eval itself lives in its own environment
(./gb bfcl setup); these tests need only GRC-Bench."""

from __future__ import annotations

import re

import pytest
import yaml

from grcbench import bfcl_run
from grcbench.stats import prereg

TEXT = " ".join(prereg.PREREG.read_text(encoding="utf-8").split())


def test_the_pre_registration_states_the_pinned_bfcl_run():
    s6 = TEXT.split("## 6.", 1)[1].split("## 7.", 1)[0]
    assert f"`bfcl-eval` {bfcl_run.BFCL_VERSION} (BFCL v4)" in s6
    assert "non-live and live categories" in s6 and "temperature 0.001" in s6
    assert bfcl_run.CATEGORIES == ("non_live", "live") and bfcl_run.TEMPERATURE == 0.001
    assert f"{bfcl_run.REPLY_TOKENS:,} tokens" in s6 and bfcl_run.REPLY_TOKENS == 16384
    assert "not tested" in s6


def test_bfcl_runs_only_the_marked_configurations(tmp_path):
    marked = bfcl_run.bfcl_configurations()
    assert {"nb-bf16-b0", "nb-bf16-s1", "nmc-q8-b0", "nmc-q8-final"} <= set(marked)
    main = {k for k, c in marked.items() if not c.get("later")}
    assert main == {"nb-bf16-b0", "nb-bf16-s1", "nmc-q8-final"}   # Amendment 1: nmc-q8-b0 (and its BFCL) later
    with pytest.raises(bfcl_run.BfclError, match="BFCL mark"):
        bfcl_run.check_configuration("nmc-q4-b0")
    other = tmp_path / "run_list.yaml"
    other.write_text(yaml.safe_dump({"configurations": [{"id": "x", "tiers": {"K": {"n": 5}, "BFCL": True}}]}))
    assert bfcl_run.check_configuration("x", other)["id"] == "x"


def test_the_registered_model_uses_function_calling_through_the_server():
    fields = bfcl_run.model_config_fields("nb-bf16-b0", "nanbeige42")
    assert fields["model_name"] == "nanbeige42" and fields["is_fc_model"] is True
    assert bfcl_run.registry_name("nb-bf16-b0") == "grcbench-nb-bf16-b0-FC"
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", bfcl_run.registry_name("nmc-q8-final"))


def test_gb_knows_the_bfcl_command():
    from grcbench.__main__ import COMMANDS
    assert COMMANDS["bfcl"][0] == "grcbench.bfcl_run:main"
