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


def test_run_list_after_decision_1():
    ids = {c["id"] for c in RUN_LIST["configurations"]}
    assert len(ids) == 38 and "b2-deepseek-v4.1-flash" in ids
    assert not ({"nb-bf16-b2", "nmc-q4-b2"} & ids) and not any(i.startswith("b3-") for i in ids)
    stages = {c["id"]: c["stage"] for c in RUN_LIST["configurations"]}
    assert stages["b2-deepseek-v4.1-flash"] == "B2 cloud teacher upper bound"
    for cid in ("nb-bf16-b0", "nmc-bf16-b0", "nmc-q4-b0"):
        assert stages[cid] == "B0 base model (server default: tool grammar on)"


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


def section(text, heading):                  # the text from a heading to the next one of its level
    level = heading.split(" ")[0]
    start = text.index(heading)
    nxt = re.search(rf"^{level} ", text[start + len(heading):], re.M)
    return text[start:start + len(heading) + nxt.start()] if nxt else text[start:]


def test_section_8_records_decisions_1_to_7():
    s8 = " ".join(section(TEXT, "## 8.").split())
    s51 = " ".join(section(TEXT, "### 5.1").split()).lower()
    assert "human route" not in s51 and "human annotator" not in s51 and "`human`" not in s51
    assert "human labels on 150 answers" in s51               # the judge calibration stays
    for n, d in ((1, "B0 is the base model served with the default tool grammar"),
                 (2, "accepted as written, including the strict bootstrap rule of H1a (95 % of pairs"),
                 (3, "4K, 16K, 32K and 64K"), (4, "design + gnuradio_usage, of 50 frozen test items"),
                 (5, "(155 items)"), (6, "F5"), (7, "scored automatically")):
        item = re.search(rf"(?:^|\s){n}\. (.*?)(?=\s{n + 1}\. \*\*|$)", s8)
        assert item and d in item[1], (n, d)
    assert "stays open" in re.search(r"\s6\. (.*?)\s7\. \*\*", s8)[1]
    assert "judge runner" in s8 and re.search(r"\s11\. \*\*The code version", s8)
    assert "context-cap failure" in s8                         # the Tier-A harness work
    assert "38 configurations" in " ".join(section(TEXT, "## 6.").split())
    assert "North-Mini-Code's BF16 B0 (k = 2)" in " ".join(section(TEXT, "## 6.").split())


def test_tier_k_dbv_band_and_masked_names():
    s51 = " ".join(section(TEXT, "### 5.1").split())
    assert "about 0.44 dB" in s51 and "dBV" in s51
    assert "listed model and vendor names are masked" in s51 and "masked names" in s51


def test_tier_k_count_nouns_and_unread_units():
    s51 = " ".join(section(TEXT, "### 5.1").split())
    assert '"34 vectors"' in s51 and '"34 bytes"' in s51
    assert "a unit the rules do not read" in s51 and "scored 0 for a unit the rules do not read" in s51


def test_tool_grammar_consequences_are_stated():
    s52 = " ".join(section(TEXT, "### 5.2").split())
    s53 = " ".join(section(TEXT, "### 5.3").split())
    s10 = " ".join(section(TEXT, "## 10.").split())
    assert "mostly check the server" in s52 and "hallucinated-block rate stays informative" in s52
    assert "V2–V5" in s52 and "mostly checks the server" in s53
    assert "or whether fine-tuning alone makes tool calls well formed" in s10
    assert "numbered 3 in the approved plans" in s10
