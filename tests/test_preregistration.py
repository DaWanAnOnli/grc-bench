"""The G2 pre-registration draft: generated appendices in sync with the code, and the run list."""

from __future__ import annotations

import re

import numpy as np
import pytest
import yaml

from grcbench.runlog import log as runlog
from grcbench.stats import functions as F
from grcbench.stats import prereg, proposals
from grcbench.tier_k import judge_prompt

TEXT = prereg.PREREG.read_text(encoding="utf-8")
RUN_LIST = yaml.safe_load(runlog.RUN_LIST.read_text(encoding="utf-8"))
HYPOTHESES = ["H1a", "H1b", "H1c", "H2a", "H2b", "H3", "H4a", "H4b"]


def _block(name: str) -> str:
    m = re.search(rf"<!-- BEGIN {name} .*?-->\n(.*?)\n<!-- END {name} -->", TEXT, re.DOTALL)
    assert m, f"markers for {name} missing"
    return m.group(1)


def test_proposals_appendix_matches_the_code():
    assert _block("stats-proposals") == prereg.proposals_block()
    for p in proposals.PROPOSALS:
        assert f"### {p['id']} ({p['kind']})" in TEXT


def test_judge_appendix_names_the_current_prompt():
    block = _block("judge-prompt")
    assert f"`{judge_prompt.PROMPT_VERSION}`" in block
    assert f"`{judge_prompt.prompt_sha256()}`" in block


def test_every_hypothesis_has_a_decision_rule():
    table = TEXT.split("## 3.", 1)[1].split("\n## 4.", 1)[0]
    for h in HYPOTHESES:
        assert re.search(rf"^\| \*\*{h}\*\* ", table, re.MULTILINE), h


def test_run_list_is_well_formed_and_still_a_draft():
    configs = RUN_LIST["configurations"]
    ids = [c["id"] for c in configs]
    assert len(ids) == len(set(ids))
    for c in configs:
        assert set(c) == {"id", "model", "precision", "stage", "tiers", "serves"}, c["id"]
        assert set(c["tiers"]) <= {"K", "F", "A", "BFCL"} and c["tiers"], c["id"]
        for tier in ("K", "F"):
            if tier in c["tiers"]:
                assert c["tiers"][tier]["n"] in (1, 3, 5), c["id"]
        if "A" in c["tiers"]:
            assert c["tiers"]["A"]["k"] in (2, 4), c["id"]
    assert sum(i.startswith("panel-") for i in ids) == 11          # 6 small, 4 teachers, 1 frontier
    if RUN_LIST["status"] != "declared":                              # signed off at G2 by the team
        assert runlog.declared_config_ids() is None
        with pytest.raises(runlog.RunRefused):
            runlog.guard(split="test", config_id=ids[0])


def test_h4a_decision_rule():
    rng = np.random.default_rng(1)
    k_full = rng.uniform(0.3, 0.9, 285)
    k_comp = k_full - 0.05
    a_full = rng.uniform(0.4, 1.0, 30)
    same = F.compression_cost_difference(a_full, a_full - 0.05, k_full, k_comp, B=2000)
    assert abs(same["difference"]) < 1e-12 and same["lo"] <= 0 <= same["hi"] and not same["holds"]
    a_comp = np.clip(a_full - 0.35 + rng.normal(0, 0.1, 30), 0, 1)
    worse = F.compression_cost_difference(a_full, a_comp, k_full, k_comp, B=2000)
    assert worse["difference"] > 0.2 and worse["holds"]
    clustered = F.compression_cost_difference(a_full, a_comp, k_full, k_comp,
                                              a_clusters=[i // 3 for i in range(30)],
                                              k_clusters=[i // 5 for i in range(285)], B=2000)
    assert clustered["n_clusters"] == {"A": 10, "K": 57}
    assert clustered == F.compression_cost_difference(a_full, a_comp, k_full, k_comp,
                                                      a_clusters=[i // 3 for i in range(30)],
                                                      k_clusters=[i // 5 for i in range(285)], B=2000)


def test_every_proposal_points_to_code_that_exists():
    import importlib
    for p in proposals.PROPOSALS:
        for where in p["where"]:
            module, _, name = where.rpartition(".")
            mod = importlib.import_module(f"grcbench.stats.{module}" if "." not in module
                                          else f"grcbench.{module}")
            assert hasattr(mod, name), where


def test_document_points_to_its_companions():
    for needle in ("run_list.yaml", "judge_prompt_snapshot.json", "./gb stats proposals", "grcbench.stats"):
        assert needle in TEXT, needle
