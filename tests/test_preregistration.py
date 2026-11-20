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
    assert len(ids) == 39 and "b2-deepseek-v4.1-flash" in ids
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
        assert set(c) - {"later"} == {"id", "model", "precision", "stage", "tiers", "serves"}, c["id"]
        assert c.get("later", True) is True, c["id"]                  # `later: true` or absent
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
    assert "reference labelling of 150 answers by three annotator agents" in s51   # the judge calibration stays
    from collections import Counter

    from grcbench.common.io import read_jsonl
    from grcbench.common.paths import SPLITS
    counts = Counter(r["clean"] for r in read_jsonl(SPLITS / "task_type_labels.jsonl") if r["split"] == "eval")
    merged = counts["design"] + counts["gnuradio_usage"]
    for n, d in ((1, "B0 is the base model served with the default tool grammar"),
                 (2, "accepted as written, including the strict bootstrap rule of H1a (95 % of pairs"),
                 (3, "4K, 16K, 32K and 64K"),
                 (4, f"design + gnuradio_usage, of {merged} frozen test items"),
                 (5, "(155 items)"), (6, "F5"), (7, "scored automatically")):
        item = re.search(rf"(?:^|\s){n}\. (.*?)(?=\s{n + 1}\. \*\*|$)", s8)
        assert item and d in item[1], (n, d)
    assert "stays open" in re.search(r"\s6\. (.*?)\s7\. \*\*", s8)[1]
    assert re.search(r"\s11\. \*\*The code version", s8)
    assert "judge calibration" in s8 and "./gb tier-k judge" in s8     # the work that remains
    assert "context-cap failure" not in s8 and "the judge runner, which" not in s8
    assert "39 configurations" in " ".join(section(TEXT, "## 6.").split())
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
    assert "numbered 3 in the methodology proposals and in the evaluation plan as approved at gate G1" in s10
    assert "Eval plan Table 1" not in s10 and "numbered 3 in the approved plans" not in s10


def test_check_covers_the_judge_snapshot(tmp_path, monkeypatch, capsys):
    stale = tmp_path / "judge_prompt_snapshot.json"
    stale.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(prereg, "SNAPSHOT", stale)
    assert prereg.main(["--check"]) == 1
    assert "judge_prompt_snapshot.json: OUT OF DATE" in capsys.readouterr().out
    monkeypatch.setattr(prereg, "SNAPSHOT", tmp_path / "missing.json")
    assert prereg.main(["--check"]) == 1


def test_a_failed_request_is_not_a_failed_reply():
    s51 = " ".join(section(TEXT, "### 5.1").split())
    assert "never counts as a failed reply" in s51 and "is not scored and is listed" in s51


def test_the_tier_k_answer_limit_is_16384_tokens():
    from grcbench.tier_k import answer_prompt
    s51 = " ".join(section(TEXT, "### 5.1").split())
    s10 = " ".join(section(TEXT, "## 10.").split())
    assert "A reply is limited to 16,384 tokens, reasoning included" in s51
    assert "Answers longer than 16,384 tokens are cut" in s51 and "4K tokens are cut" not in s51
    assert "The evaluation plan as approved at gate G1 limits Tier-K answers to 4K tokens (§7.1)" in s10
    assert "9 finished" in s10
    assert RUN_LIST["defaults"]["max_answer_tokens"]["K"] == answer_prompt.MAX_ANSWER_TOKENS == 16384


def test_the_tier_a_context_cap_is_the_models_own_if_shorter():
    s53 = " ".join(section(TEXT, "### 5.3").split())
    assert "64K tokens of context, or the model's own context if shorter" in s53


def test_the_wrong_form_count_matches_the_code():
    s51 = " ".join(section(TEXT, "### 5.1").split())
    assert "with a number scored 0 for the wrong form," in s51 and "wrong form alone" not in s51


def test_the_mdes_of_section_7_are_at_the_v1_test_items():
    from grcbench.review.items import active_dirs
    from grcbench.stats import mde as md
    from grcbench.tier_a.task import task_dirs as a_dirs
    from grcbench.tier_f.taskio import task_dirs as f_dirs
    from grcbench.tier_k.run import load_split
    assert md.V1_ITEMS == {"K": len(load_split("test")), "F": len(active_dirs(f_dirs("test"))),
                           "A": len(active_dirs(a_dirs("test")))}
    s7 = section(TEXT, "## 7.")
    for r in md.v1_table():
        _, lo, hi = md.TABLE8[r["tier"]]
        assert f"| {r['tier']} | {r['planned_items']} | {lo}–{hi} | {r['n_items']} | {r['points'].replace('-', '–')} |" in s7
    a = md.v1_table()[2]
    pts = a["points"].replace("-", "–")
    from scipy import stats as st
    z = st.norm.ppf(0.975) + st.norm.ppf(0.8)           # MDE(n, p) <= p exactly when p >= z^2 / n
    s7_flat = " ".join(s7.split())
    assert f"At its {a['n_items']} test tasks, Tier A's MDE is" in s7_flat
    assert f"reaches 80 % power only when more than about {round(100 * z * z / a['n_items'])} % of its tasks" in s7_flat
    assert "a non-significant Tier-A comparison is not read as evidence of no difference" in s7_flat
    assert f"Tier A's MDE at its {a['n_items']} test tasks ({pts} points) is printed beside the result." in TEXT
    h4a = next(p for p in proposals.PROPOSALS if p["id"] == "H4a-decision")
    assert f"Tier A's MDE at its {a['n_items']} test tasks ({a['points']} points) is printed beside it." in h4a["proposal"]


def test_the_run_list_names_the_freeze_date():
    head = runlog.RUN_LIST.read_text(encoding="utf-8").split("status:", 1)[0]
    assert "at the freeze (20 Nov 2026)" in head and "13 Nov" not in head and "17 Nov" not in head
    assert "G2" not in head and "30 Oct" not in head


OUTSIDE = {"qwen3-30b-a3b-q4", "b2-deepseek-v4.1-flash"}
MAIN = {"nb-bf16-b0", "nb-bf16-b1", "nb-bf16-s1", "nb-q8-final", "nb-q5km-final", "nb-q4-final",   # decision 9:
        "nmc-q8-b0", "nmc-q4-b0", "nmc-q4-b1", "nmc-q8-final", "nmc-q5km-final",  # base and SFT on all data;
        "nmc-q4-final"}                                                         # decision 15: NMC's anchor Q8_0


def test_the_main_runs_are_the_base_and_fine_tuned_models():
    configs = RUN_LIST["configurations"]
    panel = {c["id"] for c in configs if c["id"].startswith("panel-")}
    later = {c["id"] for c in configs if c.get("later")}
    assert {c["id"] for c in configs} - later == MAIN and len(later) == 27
    assert panel | OUTSIDE <= later
    assert {c["model"] for c in configs if c["id"] in panel} == {"chosen later"}
    assert runlog.slots_without_member() == panel                    # the guard refuses them until entered
    stages = {c["id"]: c["stage"] for c in configs}
    assert "knowledge-only" in stages["nb-arm-1"] and "trajectories-only" in stages["nb-arm-2"]
    assert "named at sign-off" not in " ".join(stages.values())
    head = runlog.RUN_LIST.read_text(encoding="utf-8").split("status:", 1)[0]
    assert "`later: true`" in head and "one dated amendment" in head and "12 configurations" in head


def test_section_8_records_decisions_8_to_11():
    s8 = " ".join(section(TEXT, "## 8.").split())
    assert "are open" not in s8 and "this draft proposes" not in s8 and "*proposed:*" not in s8
    for n, d in ((8, "L1 has at most 5 functional blocks"), (9, "supervised fine-tuning (SFT) on all the data"),
                 (10, "re-run up to twice"), (11, "the commit tagged `v1.0`")):
        item = re.search(rf"(?:^|\s){n}\. (.*?)(?=\s{n + 1}\. \*\*|$)", s8)
        assert item and d in item[1], (n, d)
    assert "L1: at most 5 functional blocks" in " ".join(section(TEXT, "### 5.2").split())
    s10 = section(TEXT, "## 10.")
    for entry in ("**Training variants.**", "**Where the Tier-F tasks come from.**", "**The canary.**"):
        assert entry in s10, entry
    for h in ("H2a", "H2b", "H3"):
        row = next(line for line in TEXT.splitlines() if line.startswith(f"| **{h}**"))
        assert f"{h} is decided only if" in row, h
    assert "commit named at sign-off" not in TEXT


def test_the_hypotheses_that_need_the_outside_models_wait_for_them():
    rows = {h: next(line for line in TEXT.splitlines() if line.startswith(f"| **{h}**"))
            for h in ("H1a", "H1b", "H1c")}
    assert "H1a is decided once the outside models have run (§6)" in rows["H1a"]
    assert "once the panel has run (§6)" in rows["H1b"]
    assert "H1c is decided once the outside models have run (§6)" in rows["H1c"]
    section6 = " ".join(TEXT.split("## 6. The declared run list", 1)[1].split("## 7.", 1)[0].split())
    assert "The main runs are 12 configurations" in section6
    assert "one dated amendment before the first panel run" in section6
    assert "chosen and added at the freeze" not in TEXT
    section9 = TEXT.split("## 9. Fixed at the freeze", 1)[1].split("## 10.", 1)[0]
    assert "panel's members, entered" not in section9
    assert "**Outside models after the main runs.**" in TEXT.split("## 10.", 1)[1]


NANBEIGE_TEMPLATE_SHA256 = "3d5fb57d879e88549d48211ba990eb7471084891b0af9cd8e6576351964981b2"


def test_nanbeige_is_served_with_the_teams_chat_template():
    """Nanbeige's own template refuses GRC-Agent's two leading system messages (HTTP 500), so §6 names the
    team's template by its hash; where the team's repository is cloned, the file has that hash."""
    import hashlib

    from grcbench.common.paths import CACHE
    engines = " ".join(TEXT.split("**Engines and settings**", 1)[1].split("\n\n", 1)[0].split())
    assert "Nanbeige4.2-3B is served with the team's chat template" in engines
    assert f"SHA-256 `{NANBEIGE_TEMPLATE_SHA256}`" in engines
    assert "With one system message the prompt is unchanged." in engines
    template = CACHE / "convert" / "grc-model-eval" / "serve" / "templates" / "nanbeige42.jinja"
    if template.is_file():
        assert hashlib.sha256(template.read_bytes()).hexdigest() == NANBEIGE_TEMPLATE_SHA256


def test_section_8_records_decisions_12_to_15():
    """20 Nov: B1 on Tier F is oracle grounding; the harness's resource limits; parallel server slots;
    North-Mini-Code's anchor is Q8_0. The stated numbers are the code's."""
    import inspect

    from grcbench.tier_a import isolation
    from grcbench.tier_k import run as krun
    s8 = " ".join(section(TEXT, "## 8.").split())
    assert "decisions 12–16 on 20 November 2026" in s8
    for n, d in ((12, "**B1 on Tier F** is oracle grounding"), (13, "8 GiB of memory and one hour of CPU time"),
                 (14, "**Parallel server slots**"), (15, "**North-Mini-Code's anchor is Q8_0**"),
                 (16, "**BFCL** runs on each target model's base model and on its fine-tuned model")):
        item = re.search(rf"(?:^|\s){n}\. (.*?)(?=\s{n + 1}\. \*\*|Work that remains|$)", s8)
        assert item and d in item[1], (n, d)
    assert "B1 on Tier F is oracle grounding" in " ".join(section(TEXT, "### 5.2").split())
    s53 = " ".join(section(TEXT, "### 5.3").split())
    assert ("limited to 8 GiB of memory and one hour of CPU time, and each shell command to 4 GiB and ten "
            "minutes of CPU time") in s53
    defaults = inspect.signature(isolation.run_episode_process).parameters
    assert defaults["mem_bytes"].default == 8 << 30 and defaults["cpu_seconds"].default == 3600
    assert isolation.SHELL_MEM_BYTES == 4 << 30 and isolation.SHELL_CPU_SECONDS == 600
    s6 = " ".join(section(TEXT, "## 6.").split())
    assert "`--n-cpu-moe 4`" in s6 and "several server slots at once" in s6
    assert "at least the answer limit plus 4,096 tokens" in s6 and krun.CONTEXT_MARGIN == 4096
    assert "with Q8_0 as the anchor" in s6 and "The other 27 configurations" in s6
    s10 = section(TEXT, "## 10.")
    assert "**B1 on Tier F.**" in s10 and "**North-Mini-Code's anchor.**" in s10
    h4a = next(line for line in TEXT.splitlines() if line.startswith("| **H4a**"))
    assert "Q8_0 for North-Mini-Code" in h4a
    by_id = {c["id"]: c for c in RUN_LIST["configurations"]}
    assert by_id["nmc-q8-b0"]["precision"] == "Q8_0" and not by_id["nmc-q8-b0"].get("later")
    assert by_id["nmc-bf16-b0"].get("later") and by_id["nmc-bf16-final"].get("later")
    assert "RQ4 anchor" in by_id["nmc-q8-b0"]["serves"] and "RQ4 anchor" in by_id["nmc-q8-final"]["serves"]
