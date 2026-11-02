"""Tests for `./gb e2e`: the report is rendered from a summary (no server, no model)."""

from __future__ import annotations

from grcbench import e2e

FIXTURE_SUMMARY = {
    "model": "Qwen3-4B-Q4_K_M", "started": "2026-10-28T10:00:00Z", "finished": "2026-10-28T12:00:00Z",
    "server": {"context": 40960, "llama_cpp_build": "b10333"},
    "tiers": {
        "K": {"items": 200, "answers": 200, "seconds": 1800.0, "n_rules": 0, "n_judge": 0, "n_no_reference": 200,
              "n_final_answer_line": 190, "n_request_failed": 0, "by_reason": {"no_reference": 200}, "out": "/w/tier_k"},
        "F": {"tasks": 10, "samples": 10, "seconds": 900.0, "passed": 3, "funnel": {"V1": 9, "V2": 8, "V3": 6, "V4": 4, "V5": 3},
              "out": "/w/tier_f"},
        "A": {"tasks": 12, "episodes": 12, "seconds": 2400.0, "passed": 2, "valid": 12, "out": "/w/tier_a"},
        "E": {"contexts": [4096], "runs": 5, "seconds": 120.0, "generation_tps": 150.0, "prompt_tps": 4000.0,
              "out": "/w/tier_e"},
    },
}


def test_report_names_every_tier_and_the_dev_split():
    text = e2e.render_report(FIXTURE_SUMMARY)
    for s in ("Tier K", "Tier F", "Tier A", "Tier E", "dev split", "Qwen3-4B"):
        assert s in text, s
    assert "test split" not in text.replace("no test-split item", "")
    assert "no test-split item" in text


def test_report_states_what_the_dev_run_cannot_measure():
    text = e2e.render_report(FIXTURE_SUMMARY)
    assert "not scored" in text and "deferral rate" in text          # empty dev rubrics (finding F5 open)
    assert "V5" in text and "GPU-hours" in text


def test_a_tier_that_did_not_run_is_reported_as_such():
    s = {**FIXTURE_SUMMARY, "tiers": {**FIXTURE_SUMMARY["tiers"], "E": {"error": "llama-bench missing"}}}
    assert "did not complete: llama-bench missing" in e2e.render_report(s)


def test_failed_requests_are_reported():
    assert "failed requests 0" in e2e.render_report(FIXTURE_SUMMARY)


def test_the_funnel_counts_rungs_in_a_row():
    grades = [{"highest_passed": "V5", "rungs": {}}, {"highest_passed": "V2", "rungs": {"V5": {"passed": True}}},
              {"highest_passed": None}, {}]
    assert e2e._funnel(grades) == {"V1": 2, "V2": 2, "V3": 1, "V4": 1, "V5": 1}


def test_the_tier_e_record_is_parsed_with_its_hardware_note():
    rec = {"contexts": [{"context": 4096, "generation": {"median_ts": 151.2}, "prompt": {"median_ts": 4021.0}}],
           "hardware": {"gpu_used": True, "mismatches": ["not an RTX 3090"]}}
    e = e2e._parse_e(rec)
    assert e["generation_tps"] == 151.2 and e["prompt_tps"] == 4021.0 and e["gpu_used"] is True
    assert e["mismatches"] == ["not an RTX 3090"]


def test_the_report_names_the_open_finding_and_the_context_and_label():
    text = e2e.render_report(FIXTURE_SUMMARY)
    assert "F5" in text and "stays open" in text and "40960" in text.replace(",", "") and "H3" in text


def test_an_error_with_a_pipe_does_not_break_the_table():
    s = {**FIXTURE_SUMMARY, "tiers": {**FIXTURE_SUMMARY["tiers"], "A": {"error": "bad | thing\nsecond line"}}}
    row = next(l for l in e2e.render_report(s).splitlines() if l.startswith("| Tier A"))
    assert row.count("|") == 5 and row.rstrip().endswith("|") and "bad / thing" in row


def test_unknown_counts_and_the_judge_line_follow_the_summary():
    k = {**FIXTURE_SUMMARY["tiers"]["K"]}
    k.pop("n_request_failed")
    s = {**FIXTURE_SUMMARY, "tiers": {**FIXTURE_SUMMARY["tiers"], "K": {**k, "n_judge": 3}}}
    text = e2e.render_report(s)
    assert "failed requests n/a" in text and "No answer reaches the judge" not in text
    assert "No answer reaches the judge" in e2e.render_report(FIXTURE_SUMMARY)


def test_report_only_rerenders_from_the_summary(tmp_path, monkeypatch):
    import json
    (tmp_path / "summary.json").write_text(json.dumps(FIXTURE_SUMMARY), encoding="utf-8")
    monkeypatch.setattr(e2e, "REPORT", tmp_path / "report.md")
    assert e2e.main(["--out", str(tmp_path), "--report-only"]) == 0
    assert "Tier K" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_notes_are_rendered():
    s = {**FIXTURE_SUMMARY, "notes": ["Two episodes ended with server errors."]}
    assert "## Notes" in e2e.render_report(s) and "Two episodes ended" in e2e.render_report(s)


def test_limits_rubrics_machine_and_report_reach_the_tiers(tmp_path, monkeypatch):
    """The options a run on another machine (Colab) needs: fewer tasks, a rubric file, its own Tier-E
    record name and machine label, and a report outside the repository."""
    from grcbench.tier_a import smoke
    seen = {}
    monkeypatch.setattr(smoke, "native_context", lambda model: 40960)
    monkeypatch.setattr(smoke, "LocalServer", lambda *a, **kw: type("S", (), {
        "load_seconds": 1.0, "__enter__": lambda self: "http://127.0.0.1:9/v1", "__exit__": lambda self, *e: None})())
    monkeypatch.setattr(e2e, "_tier_k", lambda *a, **kw: seen.setdefault("K", (a, kw)) and {"items": 1})
    monkeypatch.setattr(e2e, "_tier_f", lambda *a, **kw: seen.setdefault("F", (a, kw)) and {"tasks": 1})
    monkeypatch.setattr(e2e, "_tier_a", lambda *a, **kw: seen.setdefault("A", (a, kw)) and {"tasks": 1})
    monkeypatch.setattr(e2e, "_tier_e", lambda *a, **kw: seen.setdefault("E", (a, kw)) and {"runs": 5})
    monkeypatch.setattr(e2e, "REPORT", tmp_path / "must_not_be_written.md")
    report, rubrics = tmp_path / "colab_report.md", tmp_path / "placeholder.jsonl"
    assert e2e.main(["--out", str(tmp_path / "run"), "--limit-k", "3", "--limit-f", "2", "--limit-a", "1",
                     "--rubrics", str(rubrics), "--report", str(report), "--e-name", "colab_t4",
                     "--machine-label", "Colab T4"]) == 0
    assert seen["K"][1]["rubrics"] == rubrics and seen["K"][0][4] == 3
    assert seen["F"][1]["limit"] == 2 and seen["A"][1]["limit"] == 1
    assert seen["E"][1] == {"name": "colab_t4", "machine_label": "Colab T4"}
    assert report.exists() and not (tmp_path / "must_not_be_written.md").exists()
