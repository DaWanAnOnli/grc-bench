"""The generator-wide changes the review protocol decided on Tier F (reviews/tier_f.jsonl, the expert's
`note: generator` lines), each checked on the reviewed task it was decided on. A change never makes a reviewed
task another task: the same attempt gives the same blocks, numbers and draw, and only the decided parts differ."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from grcbench.tier_f import generate as G
from grcbench.tier_f import golden

FX = Path(__file__).parent / "fixtures" / "tier_f_task" / "fx_modify_tone"


def _cand(task_id: str, attempt: int) -> dict:
    split = task_id.split("-")[1]
    family, level = G.plan(split)[task_id]
    cand = G.assemble(task_id, split, family, level, attempt)
    assert cand is not None, (task_id, attempt)
    return cand


def _rec(task_id: str, attempt: int) -> dict:
    return _cand(task_id, attempt)["record"]


def _groups(rec: dict) -> tuple[dict, dict]:
    """{(feature, lo or expect): group} of the signal checks, {(kind, via ids): [groups]} of the graph checks."""
    sig = {(c["feature"], c.get("lo", c.get("expect"))): c["group"] for c in rec["oracle"]["checks"]}
    graph: dict = {}
    for c in rec["oracle"]["graph_checks"]:
        graph.setdefault((c["kind"], tuple(c.get("via_ids", ()))), []).append(c["group"])
    return sig, graph


# ------------------------------------------------------------------------- golden lookup (dev-01, test-10, ...)
def test_a_task_can_give_its_own_natural_golden_lookup():
    ref = {"blocks": [{"name": "samp_rate", "id": "variable"}, {"name": "src", "id": "analog_sig_source_x"},
                      {"name": "hpf", "id": "high_pass_filter"}, {"name": "out", "id": "blocks_file_sink"}]}
    assert golden.lookup_args(ref) == {"query": "analog_sig_source_x high_pass_filter blocks_file_sink",
                                       "domain": "catalog", "k": 6}
    natural = {"query": "high pass filter block", "domain": "catalog", "k": 8}
    assert golden.lookup_args(ref, natural) == natural
    assert golden.lookup_args(ref, {"query": "high pass filter block"}) == {
        "query": "high pass filter block", "domain": "catalog", "k": 6}            # domain and k as derived
    for bad in ({"query": ""}, {"query": "x", "k": 0}, {"query": "x", "extra": 1}, "a string"):
        with pytest.raises(golden.GoldenCallsError):
            golden.lookup_args(ref, bad)


def test_the_golden_derivation_reads_the_natural_lookup_from_task_json(tmp_path):
    task_dir = tmp_path / "fx_modify_tone"
    shutil.copytree(FX, task_dir)
    rec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    derived = golden.derive_golden_calls(task_dir)["query_knowledge"]
    assert derived["query"].startswith("analog_sig_source_x")                   # no natural query: block ids
    rec[golden.NATURAL_LOOKUP] = {"query": "signal source frequency", "domain": "catalog", "k": 8}
    (task_dir / "task.json").write_text(json.dumps(rec), encoding="utf-8")
    calls = golden.derive_golden_calls(task_dir)
    assert calls["query_knowledge"] == {"query": "signal source frequency", "domain": "catalog", "k": 8}


# ------------------------------------------------------------------------- tone sources (dev-21, dev-34, test-32)
def test_tone_sources_are_named_by_position_not_by_frequency():
    g = G.Graph("t", 32000.0)
    names, out = G.Source().build(g, "", {"kind": "complex", "tones": [(9500, 0.5), (4200, 0.4), (11200, 1.0)]},
                                  G.Sig("float", 32000.0, ()))
    assert names == ["tone_1", "tone_2", "tone_3", "sum_of_tones"] and out == "sum_of_tones"
    g = G.Graph("t", 48000.0)
    names, out = G.Source().build(g, "", {"kind": "float", "wave": "saw", "tones": [(1200, 1.0)], "noise": 0.0},
                                  G.Sig("float", 48000.0, ()))
    assert names == ["saw"] and out == "saw"


@pytest.mark.gnuradio
def test_a_tuned_source_no_longer_names_its_target_frequency():
    import yaml
    for task_id, attempt, block, start_freq in (("gbf-dev-21", 71, "tone_1", "4700"), ("gbf-dev-34", 8, "tone_2", "5500"),
                                                ("gbf-test-32", 2003, "saw", "1600")):
        cand = _cand(task_id, attempt)
        start = yaml.safe_load(cand["files"]["start.grc"]) if isinstance(cand["files"]["start.grc"], str) \
            else cand["files"]["start.grc"]
        names = {b["name"]: b for b in start["blocks"]}
        assert names[block]["parameters"]["freq"] == start_freq, task_id
        assert not any(n.startswith(("tone_", "saw_")) and n not in ("tone_1", "tone_2", "tone_3") for n in names)


# ------------------------------------------------------------------------- structural repairs (dev-14, dev-20, ...)
@pytest.mark.gnuradio
def test_a_structural_repair_labels_p2p_what_the_fault_does_not_touch():
    """nonexistent_block: the checks that still pass with the faulted stage bypassed; missing_connection: the
    path checks that already pass on the start graph. Everything else stays f2p."""
    sig, graph = _groups(_rec("gbf-dev-14", 3))
    assert sig == {("band_fraction", -2500.0): "f2p", ("band_fraction", 2100.0): "f2p", ("rms", 0.5657): "p2p"}
    assert graph == {("path", ("channels_channel_model",)): ["f2p"], ("path", ("blocks_conjugate_cc",)): ["p2p"]}
    sig, graph = _groups(_rec("gbf-dev-35", 2578))
    assert [g for k, g in sig.items() if k[0] == "rms"] == ["p2p"]
    assert {k[0] for k, g in sig.items() if g == "f2p"} == {"peak_freq", "band_fraction"}
    assert graph == {("path", ("channels_channel_model",)): ["f2p"], ("path", ("blocks_conjugate_cc",)): ["p2p"]}
    sig, graph = _groups(_rec("gbf-test-107", 111))
    assert {k[0] for k, g in sig.items() if g == "p2p"} == {"peak_freq", "band_fraction", "rms"}
    assert {k[0] for k, g in sig.items() if g == "f2p"} == {"max_int_dev"}
    assert graph == {("path", ("rational_resampler_xxx",)): ["p2p"], ("block_present", ()): ["p2p"],
                     ("path", ("blocks_float_to_short",)): ["f2p"]}           # the two Resamplers: one path, min 2
    for task_id, attempt, more in (("gbf-dev-20", 20, {("block_present", ()): ["p2p"]}),   # its two Channel Models
                                   ("gbf-dev-24", 2001, {})):
        sig, graph = _groups(_rec(task_id, attempt))
        assert set(sig.values()) == {"f2p"}, task_id
        assert {k: g for k, g in graph.items() if "p2p" in g} == {("path", ("blocks_multiply_xx",)): ["p2p"],
                                                                 **more}, task_id


# ------------------------------------------------------------------------- modify tap point (dev-22, dev-27, ...)
@pytest.mark.gnuradio
def test_a_modify_tap_is_named_when_two_stages_share_its_label():
    assert "take the signal at the output of the first AGC block (agc, the one before the Channel Model), shift it " \
           "up by 3 kHz" in _rec("gbf-dev-27", 33)["instruction"]
    assert "take the signal at the output of the second mixer (mixer_2, the one fed by the 4.5 kHz tone), bring it " \
           "to a magnitude of 2" in _rec("gbf-test-103", 7)["instruction"]
    assert "take the signal at the output of the High Pass Filter, round it to steps of 0.01" in \
        _rec("gbf-test-141", 57)["instruction"]
    assert "take the signal at the output of the Add block, take its complex conjugate" in \
        _rec("gbf-dev-22", 488)["instruction"]                                       # a tone and the noise: the Add


@pytest.mark.gnuradio
def test_a_modify_task_checks_the_path_from_its_tap_point():
    rec = _rec("gbf-dev-22", 488)
    assert {"kind": "path", "from_ids": ["blocks_add_xx"], "via_ids": ["blocks_conjugate_cc"],
            "to_ids": ["blocks_file_sink"], "to_params": {"file": "second_mirrored.dat"},
            "group": "f2p"} in rec["oracle"]["graph_checks"]
    assert any("from the Add block" in p for p in rec["spec_properties"])
    rec = _rec("gbf-dev-27", 33)
    assert {"kind": "path", "from_ids": ["analog_agc_xx", "analog_agc2_xx", "analog_agc3_xx"],
            "via_ids": ["blocks_multiply_xx"], "to_ids": ["blocks_file_sink"], "to_params": {"file": "second_shifted.dat"},
            "group": "f2p"} in rec["oracle"]["graph_checks"]
    rec = _rec("gbf-dev-43", 1678)                                                  # tapped after the Conjugate
    assert {"kind": "path", "from_ids": ["blocks_conjugate_cc"], "via_ids": ["analog_agc_xx", "analog_agc2_xx",
            "analog_agc3_xx"], "to_ids": ["blocks_file_sink"], "to_params": {"file": "second_levelled.dat"},
            "group": "f2p"} in rec["oracle"]["graph_checks"]
    rec = _rec("gbf-dev-13", 70)                          # a tap at the only tone source: anchored on the main output
    taps = [c for c in rec["oracle"]["graph_checks"] if c.get("from_upstream_of")]
    assert taps == [{"kind": "path", "from_ids": ["analog_sig_source_x"],
                     "from_upstream_of": {"ids": ["blocks_file_sink"], "params": {"file": rec["observation"]["points"][0]
                                                                                  ["sink"]["params"]["file"]}},
                     "via_ids": taps[0]["via_ids"], "to_ids": ["blocks_file_sink"],
                     "to_params": taps[0]["to_params"], "group": "f2p"}]


# ------------------------------------------------------------------------- the strobe tune (test-105, test-133, ...)
@pytest.mark.gnuradio
def test_a_strobe_tune_describes_the_start_graph_and_not_the_answer():
    text = _rec("gbf-test-105", 417)["instruction"]
    assert "use a Signal Source to make a 5.6 kHz cosine with amplitude 1;" in text
    assert "sends pmt.to_pmt({'freq': 5100}) every 100 ms to that Signal Source's message input cmd, so that the " \
           "5.6 kHz tone is retuned to 5.1 kHz" in text
    assert "3900" not in text and "3.9 kHz" not in text
    text = _rec("gbf-test-133", 625)["instruction"]
    assert "use a Signal Source to make a 5.9 kHz complex tone" in text
    assert "sends pmt.to_pmt({'freq': 10800})" in text and "so that the 5.9 kHz tone is retuned to 10.8 kHz" in text
    assert "8300" not in text and "8.3 kHz" not in text


# ------------------------------------------------------------------------- the stated rate (test-121, test-34, ...)
@pytest.mark.gnuradio
def test_tune_and_wrong_value_repairs_pin_the_stated_sample_rate():
    for task_id, attempt, rate, words in (("gbf-test-121", 423, "48000", "48 kHz"), ("gbf-test-140", 2754, "32000", "32 kHz"),
                                          ("gbf-test-32", 2003, "48000", "48 kHz"), ("gbf-test-34", 2004, "32000", "32 kHz")):
        rec = _rec(task_id, attempt)
        assert {"kind": "block_present", "ids": ["variable"], "params": {"value": rate},
                "group": "p2p"} in rec["oracle"]["graph_checks"], task_id
        assert f"the sample rate stays at {words} (samp_rate {rate})" in rec["spec_properties"], task_id
        assert "structure: same blocks and connections as the start graph" in rec["spec_properties"], task_id
    for task_id, attempt in (("gbf-dev-14", 3), ("gbf-dev-22", 488)):            # structural repair, modify: no pin
        assert all(c["kind"] != "block_present" for c in _rec(task_id, attempt)["oracle"]["graph_checks"])


# ------------------------------------------------------------------------- rules for every new attempt (dev-40, dev-13)
@pytest.mark.gnuradio
def test_the_search_applies_the_hilbert_band_and_the_l1_size_to_every_attempt():
    """A new search keeps no candidate whose Hilbert block gets a tone outside fs/20 .. 0.45 fs, or whose L1
    reference holds more than five blocks; a reviewed task's recorded attempt still builds as it was."""
    misses = G.search_rule_misses(_cand("gbf-dev-40", 99))
    assert len(misses) == 1 and "Hilbert" in misses[0] and "600 Hz" in misses[0]
    misses = G.search_rule_misses(_cand("gbf-dev-13", 70))
    assert len(misses) == 1 and "L1" in misses[0] and "6 blocks" in misses[0]
    assert G.search_rule_misses(_cand("gbf-dev-21", 71)) == []
    assert G.search_rule_misses(_cand("gbf-test-141", 57)) == []                  # Hilbert tones inside the band


# ------------------------------------------------------------------------- the per-task patches
def test_every_tier_f_patch_implements_a_decided_edit():
    """reviews/tier_f_patches.json: each patch names a task decided `approve with edits` and quotes one of the
    expert's merged edits, and its op is one review.patches applies. A patch marked `post_review` is an edit after
    the review (pre-registration §10, "Edits after the review"): it names a task the review kept, states its edit,
    and the tasks with such patches are the seven build tasks §10 names."""
    from grcbench.common.paths import REPO
    from grcbench.review import patches
    path = REPO / "reviews" / "tier_f_patches.json"
    if not path.exists():
        pytest.skip("no Tier-F patches yet")
    recs = {json.loads(l)["key"][0]: json.loads(l)
            for l in (REPO / "reviews" / "tier_f.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    post = set()
    for p in json.loads(path.read_text(encoding="utf-8")):
        rec = recs.get(p["item"])
        assert p["op"] in patches.OPS and p.get("file", "task.json") == "task.json", p
        if p.get("post_review"):
            assert rec is not None and rec["final"] in ("approve", "approve with edits") and p["edit"], p["item"]
            post.add(p["item"])
            continue
        assert rec is not None and rec["final"] == "approve with edits", p["item"]
        assert p["edit"] in (rec.get("ex") or rec.get("audit"))["edits"], (p["item"], p["edit"][:60])
    assert post == {"gbf-dev-33", *(f"gbf-test-{i}" for i in (24, 37, 94, 114, 134, 135))}


def test_the_search_skips_a_candidate_that_misses_a_rule(monkeypatch, tmp_path):
    good = {"record": {"level": "L1"}, "files": {}, "cell": "a|b", "rule_misses": []}
    bad = {"record": {"level": "L1"}, "files": {}, "cell": "a|c", "rule_misses": ["L1: 7 blocks besides samp_rate"]}
    monkeypatch.setattr(G, "TF", tmp_path)
    monkeypatch.setattr(G, "used_cells", lambda exclude=(): {})
    monkeypatch.setattr(G, "cell_counts", lambda exclude=(): {})
    monkeypatch.setattr(G, "assemble", lambda tid, split, fam, lev, attempt: bad if attempt == 0 else good)
    written = []

    def fake_write(task_dir, cand):
        Path(task_dir).mkdir(parents=True, exist_ok=True)
        written.append(cand["cell"])
        return []
    monkeypatch.setattr(G, "write", fake_write)
    assert G.generate(["gbf-dev-11"]) == ["gbf-dev-11"]
    assert written == ["a|b"]
    assert "attempt 1" in (tmp_path / "dev" / "gbf-dev-11" / "build.py").read_text(encoding="utf-8").splitlines()[0]
