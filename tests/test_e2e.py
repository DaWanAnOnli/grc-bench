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


PROPS = {"build_info": "b12020-0ec4027a2", "model_path": "/content/models/nanbeige42/Nanbeige4.2-3B-Q8_0.gguf",
         "total_slots": 1,
         "default_generation_settings": {"n_ctx": 65536, "params": {"temperature": 0.6, "top_k": 20, "top_p": 0.95,
                                                                    "min_p": 0.05, "n_predict": -1}}}


def test_a_running_servers_build_context_and_sampling_are_read_from_its_props():
    s = e2e.props_summary(PROPS)
    assert s == {"llama_cpp_build": "b12020", "build_info": "b12020-0ec4027a2", "context": 65536, "slots": 1,
                 "model_path": PROPS["model_path"],
                 "sampling": {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.05}}
    assert e2e.props_summary({})["llama_cpp_build"] is None
    f32 = {"default_generation_settings": {"params": {"temperature": 0.6000000238418579, "top_p": 0.949999988079071}}}
    assert e2e.props_summary(f32)["sampling"] == {"temperature": 0.6, "top_p": 0.95}     # the server's float32 values


def _external(tmp_path, monkeypatch, props, only="K,F,A"):
    import pytest
    from grcbench.tier_a import smoke
    seen = {}
    monkeypatch.setattr(smoke, "LocalServer", lambda *a, **kw: pytest.fail("--base-url: no server of its own"))
    monkeypatch.setattr(e2e, "server_props", lambda url: seen.setdefault("props_url", url) and props)
    monkeypatch.setattr(e2e, "_tier_k", lambda *a, **kw: seen.setdefault("K", (a, kw)) and {"items": 1})
    monkeypatch.setattr(e2e, "_tier_f", lambda *a, **kw: seen.setdefault("F", (a, kw)) and {"tasks": 1})
    monkeypatch.setattr(e2e, "_tier_a", lambda *a, **kw: seen.setdefault("A", (a, kw)) and {"tasks": 1})
    monkeypatch.setattr(e2e, "_tier_e", lambda *a, **kw: seen.setdefault("E", (a, kw)) and {"runs": 5})
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    code = e2e.main(["--out", str(tmp_path / "run"), "--only", only, "--base-url", "http://127.0.0.1:8181/v1",
                     "--model", "nanbeige42", "--gguf", str(gguf), "--report", str(tmp_path / "report.md"),
                     "--machine-label", "Colab T4"])
    return code, seen, gguf


def test_a_running_model_server_serves_tiers_k_f_and_a(tmp_path, monkeypatch):
    """Another notebook's llama-server (Joy's evaluation notebook) answers Tiers K, F and A; the run records
    the server's build, context and sampling."""
    import json
    code, seen, gguf = _external(tmp_path, monkeypatch, PROPS)
    assert code == 0 and seen["props_url"] == "http://127.0.0.1:8181/v1"
    for tier in "KFA":
        assert seen[tier][0][:2] == ("http://127.0.0.1:8181/v1", "nanbeige42")
    assert seen["K"][0][5] == {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.05}   # the server's own
    assert "E" not in seen
    s = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert s["model"] == "nanbeige42" and s["server"]["base_url"] == "http://127.0.0.1:8181/v1"
    assert s["server"]["llama_cpp_build"] == "b12020" and s["server"]["context"] == 65536
    assert s["server"]["model_file"] == str(gguf) and s["machine_label"] == "Colab T4"
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "nanbeige42" in text and "Colab T4" in text and "smoke model" not in text and "RTX 5090" not in text


def test_a_running_model_server_of_another_llama_cpp_build_is_refused(tmp_path, monkeypatch):
    props = {**PROPS, "build_info": "b10333-08659901c"}
    code, seen, _ = _external(tmp_path, monkeypatch, props)
    assert code == 2 and not any(t in seen for t in "KFAE")


def test_tier_e_does_not_run_beside_a_running_model_server(tmp_path, monkeypatch):
    """Tier E measures with no other llama.cpp process running (it would wait up to an hour for the server to
    stop), so with --base-url it runs alone, after the server has stopped; asked together, nothing runs."""
    import pytest
    with pytest.raises(SystemExit):
        _external(tmp_path, monkeypatch, PROPS, only="K,F,A,E")
    assert not (tmp_path / "run").exists()


def test_a_running_model_server_needs_its_model_name():
    import pytest
    with pytest.raises(SystemExit):
        e2e.main(["--out", "/nonexistent", "--base-url", "http://127.0.0.1:8181/v1"])


def test_tier_e_alone_after_the_running_server_stopped_keeps_the_servers_record(tmp_path, monkeypatch):
    """On a 16 GB T4 the notebook stops its model server before Tier E: `--only E` with --base-url asks the
    stopped server nothing and keeps the server's record from the earlier run of Tiers K, F and A."""
    import json
    import pytest
    from grcbench.tier_a import smoke
    code, seen, gguf = _external(tmp_path, monkeypatch, PROPS)
    assert code == 0
    monkeypatch.setattr(e2e, "server_props", lambda url: pytest.fail("the server is stopped: no /props"))
    monkeypatch.setattr(smoke, "native_context", lambda model: pytest.fail("no smoke model with --base-url"))
    seen.clear()
    assert e2e.main(["--out", str(tmp_path / "run"), "--only", "E", "--base-url", "http://127.0.0.1:8181/v1",
                     "--model", "nanbeige42", "--gguf", str(gguf), "--report", str(tmp_path / "report.md"),
                     "--machine-label", "Colab T4"]) == 0
    assert list(seen) == ["E"] and seen["E"][0][0] == gguf
    s = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert set(s["tiers"]) == {"K", "F", "A", "E"} and s["server"]["llama_cpp_build"] == "b12020"
    assert s["server"]["context"] == 65536


def test_a_run_with_a_rubric_file_says_its_tier_k_answers_are_scored_with_it():
    k = {**FIXTURE_SUMMARY["tiers"]["K"], "n_judge": 2, "n_no_reference": 0,
         "rubrics": {"file": "colab/dev_placeholder.jsonl", "n_placeholder": 107, "n_records": 107}}
    text = e2e.render_report({**FIXTURE_SUMMARY, "tiers": {**FIXTURE_SUMMARY["tiers"], "K": k}})
    assert "colab/dev_placeholder.jsonl" in text and "107 of 107 placeholder" in text
    assert "not scored" not in text
