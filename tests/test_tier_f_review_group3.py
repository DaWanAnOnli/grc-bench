"""The generator-wide changes the review protocol decided on Tier F's third group (gbf-test-35 .. gbf-test-99;
reviews/tier_f.jsonl, the expert's `note: generator` lines), each checked on a reviewed task it was decided on. As in
test_tier_f_review_changes.py, a change never makes a reviewed task another task: the same attempt gives the same
blocks, numbers and draw. The rules a new search applies (search_rule_misses, the shared-cell rules of generate) only
skip candidates; a reviewed task's recorded attempt still builds as it was."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grcbench.tier_f import generate as G
from grcbench.tier_f import golden


def _cand(task_id: str, attempt: int) -> dict:
    split = task_id.split("-")[1]
    family, level = G.plan(split)[task_id]
    cand = G.assemble(task_id, split, family, level, attempt)
    assert cand is not None, (task_id, attempt)
    return cand


def _rec(task_id: str, attempt: int) -> dict:
    return _cand(task_id, attempt)["record"]


def _structure(rec: dict) -> str:
    return next(p for p in rec["spec_properties"] if p.startswith("structure: a path from a Signal Source"))


# ------------------------------------------------------------------------- natural golden lookup (test-35, test-37, ...)
@pytest.mark.gnuradio
def test_assemble_writes_the_catalog_labels_of_the_blocks_the_edit_adds():
    """golden_lookup: the catalog labels of the blocks the edit adds (all of a build's, a modify task's new ones, the
    block whose id a nonexistent-block repair corrects), or of the block whose parameter a value edit changes."""
    want = {("gbf-test-37", 13): "Signal Source Add Low Pass Filter Complex Conjugate File Sink",          # build
            ("gbf-test-40", 12): "Signal Source Frequency Mod Quadrature Demod Hilbert File Sink",
            ("gbf-test-36", 13): "Abs File Sink",                                                         # modify
            ("gbf-test-64", 1753): "Signal Source Multiply File Sink",
            ("gbf-test-68", 5): "Complex to Mag^2 Moving Average Log10 File Sink",
            ("gbf-test-35", 13): "Hilbert",                                                               # wrong id
            ("gbf-test-92", 3610): "Complex Conjugate",
            ("gbf-test-81", 2086): "Channel Model",                                                       # wrong value
            ("gbf-test-66", 468): "Message Strobe"}                                                       # strobe tune
    for (task_id, attempt), query in want.items():
        assert _rec(task_id, attempt)[golden.NATURAL_LOOKUP] == {"query": query, "domain": "catalog"}, task_id
    assert golden.NATURAL_LOOKUP not in _rec("gbf-test-39", 428)        # a missing connection adds no block
    assert golden.catalog_label("hilbert_fc") == "Hilbert"
    assert golden.catalog_label("no_such_block") == "no_such_block"


# ------------------------------------------------------------------------- repeated stages (test-35, test-40, ...)
@pytest.mark.gnuradio
def test_a_repeated_stage_is_checked_as_often_as_it_is_stated():
    """The second identical path check gives way to block_present checks with min k (one per block the stage
    builds); a structural repair labels them p2p when the fault is not in that stage."""
    fm = ["analog_frequency_modulator_fc", "analog_quadrature_demod_cf"]
    rec = _rec("gbf-test-40", 12)                                                  # build: fm, fm, hilbert
    gc = rec["oracle"]["graph_checks"]
    assert [c["kind"] for c in gc] == ["path", "block_present", "block_present", "path"]
    assert gc[1:3] == [{"kind": "block_present", "ids": ["analog_frequency_modulator_fc"], "min": 2, "group": "main"},
                       {"kind": "block_present", "ids": ["analog_quadrature_demod_cf"], "min": 2, "group": "main"}]
    assert gc[0]["via_ids"] == fm and gc[3]["via_ids"] == ["hilbert_fc"]
    assert _structure(rec).endswith("to the File Sink; two Frequency Mod and two Quadrature Demod blocks")
    gc = _rec("gbf-test-35", 13)["oracle"]["graph_checks"]                         # wrong id in the Hilbert stage
    assert len(gc) == 3 and gc[2] == {"kind": "block_present", "ids": ["blocks_multiply_xx"], "min": 2, "group": "p2p"}
    rec = _rec("gbf-test-39", 428)                                                 # missing connection
    assert rec["oracle"]["graph_checks"][1] == {"kind": "block_present", "ids": ["channels_channel_model"], "min": 2,
                                                "group": "p2p"}
    assert _structure(rec).endswith("; both Channel Models are kept")
    rec = _rec("gbf-test-65", 8)                                                   # wrong id in the second Resampler
    assert rec["oracle"]["graph_checks"][2] == {"kind": "block_present", "ids": ["rational_resampler_xxx"], "min": 2,
                                                "group": "f2p"}
    assert _structure(rec).endswith("; two Rational Resamplers")
    for task_id, attempt, words in (("gbf-test-79", 2076, "; two Conjugates"), ("gbf-test-83", 2088, "; two Channel Models"),
                                    ("gbf-test-48", 28, "; two Channel Models")):
        rec = _rec(task_id, attempt)
        paths = [json.dumps(c, sort_keys=True) for c in rec["oracle"]["graph_checks"] if c["kind"] == "path"]
        assert len(paths) == len(set(paths)) and _structure(rec).endswith(words), task_id
    for task_id, attempt in (("gbf-test-35", 13), ("gbf-test-40", 12)):
        assert not any(k.startswith("_") for c in _rec(task_id, attempt)["oracle"]["graph_checks"] for k in c)


# ------------------------------------------------------------------------- phrases (test-40, test-36, ...)
@pytest.mark.gnuradio
def test_the_fm_and_rectify_phrases_give_what_grc_and_the_oracle_take():
    text = _rec("gbf-test-40", 12)["instruction"]
    for words in ("sensitivity 2*math.pi*1500/samp_rate", "gain samp_rate/(2*math.pi*3000)",
                  "sensitivity 2*math.pi*2500/samp_rate", "gain samp_rate/(2*math.pi*5000)"):
        assert words in text
    assert "2*pi" not in text
    assert "gain samp_rate*2/1/(2*math.pi*750)" in _rec("gbf-test-60", 41)["instruction"]
    assert "full-wave rectify it with an Abs block, so that each output sample is the absolute value of the input " \
           "sample" in _rec("gbf-test-36", 13)["instruction"]


# ------------------------------------------------------------------------- modify taps at k == 0 (test-44, test-61, ...)
@pytest.mark.gnuradio
def test_a_tap_at_the_tones_step_is_the_add_block_or_the_source_behind_the_main_output():
    assert "take the signal at the output of the Add block, take its complex conjugate" in \
        _rec("gbf-test-44", 596)["instruction"]                                     # one tone and the noise
    assert {"kind": "path", "from_ids": ["analog_sig_source_x"],
            "from_upstream_of": {"ids": ["blocks_file_sink"], "params": {"file": "levelled.dat"}},
            "via_ids": ["analog_frequency_modulator_fc", "analog_quadrature_demod_cf"], "to_ids": ["blocks_file_sink"],
            "to_params": {"file": "second_recovered.dat"}, "group": "f2p"} in _rec("gbf-test-61", 3)["oracle"]["graph_checks"]
    assert {"kind": "path", "from_ids": ["analog_sig_source_x"],
            "from_upstream_of": {"ids": ["blocks_file_sink"], "params": {"file": "level_db.dat"}},
            "via_ids": ["blocks_multiply_xx"], "to_ids": ["blocks_file_sink"], "to_params": {"file": "second_shifted.dat"},
            "group": "f2p"} in _rec("gbf-test-64", 1753)["oracle"]["graph_checks"]


# ------------------------------------------------------------------------- the stated waveform (test-42)
@pytest.mark.gnuradio
def test_a_square_saw_or_triangle_source_is_checked_for_its_waveform():
    rec = _rec("gbf-test-42", 2071)
    assert {"kind": "block_present", "ids": ["analog_sig_source_x"], "params": {"waveform": "analog.GR_SQR_WAVE"},
            "group": "main"} in rec["oracle"]["graph_checks"]
    assert all("waveform" not in json.dumps(c) for c in _rec("gbf-test-37", 13)["oracle"]["graph_checks"])


# ------------------------------------------------------------------------- search-only rules (test-44, test-139, test-22)
@pytest.mark.gnuradio
def test_the_search_skips_a_branch_that_repeats_the_next_stage_a_cancelling_pair_and_a_no_op_quantizer():
    misses = G.search_rule_misses(_cand("gbf-test-44", 596))
    assert any("repeats the main chain's next stage (conjugate)" in m for m in misses)
    misses = G.search_rule_misses(_cand("gbf-test-139", 2040))
    assert any("Conjugate right after a Conjugate" in m for m in misses)
    misses = G.search_rule_misses(_cand("gbf-test-22", 2))
    assert any("already on the grid" in m for m in misses)
    for task_id, attempt in (("gbf-test-36", 13), ("gbf-test-40", 12), ("gbf-test-61", 3)):
        assert not [m for m in G.search_rule_misses(_cand(task_id, attempt))
                    if "repeats" in m or "Conjugate right after" in m or "grid" in m], task_id


@pytest.mark.gnuradio
def test_a_candidate_carries_its_edit_key():
    assert _cand("gbf-test-139", 2040)["edit_key"] == ("build", ("tones", "conjugate", "conjugate"))
    key = _cand("gbf-test-128", 2018)["edit_key"]
    assert key[0] == "modify" and key[1].startswith("second output tapped after step")
    assert _cand("gbf-test-47", 2057)["edit_key"][0] == "parameter-tune"
    assert _cand("gbf-test-39", 428)["edit_key"] == ("repair", "missing_connection")


def test_an_edit_key_names_the_block_type_not_the_instance():
    """Second-round review (gbf-test-151 and gbf-test-51, gbf-test-153 and gbf-test-58): retuning or repairing tone_2
    instead of tone_1 is the same edit, so the shared-cell rule keys an edit on the block id and the parameter."""
    def graph(f1, f2):
        return {"blocks": [{"name": "tone_1", "id": "analog_sig_source_x", "parameters": {"freq": f1}},
                           {"name": "tone_2", "id": "analog_sig_source_x", "parameters": {"freq": f2}}]}
    tune_1 = G.edit_key({"family": "parameter-tune"}, graph("1000", "2000"), graph("1500", "2000"))
    tune_2 = G.edit_key({"family": "parameter-tune"}, graph("1000", "2000"), graph("1000", "2500"))
    assert tune_1 == tune_2 == ("parameter-tune", ("analog_sig_source_x", "freq"))
    fix_1 = G.edit_key({"family": "repair", "fault": {"type": "wrong_value", "block": "tone_1", "param": "freq"}},
                       graph("1000", "2000"), None)
    fix_2 = G.edit_key({"family": "repair", "fault": {"type": "wrong_value", "block": "tone_2", "param": "freq"}},
                       graph("1000", "2000"), None)
    assert fix_1 == fix_2 == ("repair", ("analog_sig_source_x", "freq"))


def test_a_quantizer_phrase_names_the_short_to_float_scale_of_the_graph_it_describes():
    """Second-round review (gbf-test-155): a parameter-tune task describes its start graph, whose Short To Float scale
    is the detuned one; the phrase names that scale, and claims a rounding step only when the two scales agree."""
    q = G.Quantize()
    assert q.phrase({"scale": 10, "back": 10}, None) == (
        "round it to steps of 0.1: convert it to short integers with a Float To Short block (scale 10) and back with "
        "a Short To Float block (scale 10)")
    assert q.phrase({"scale": 10, "back": 100}, None) == (
        "convert it to short integers with a Float To Short block (scale 10) and back with a Short To Float block "
        "(scale 100)")


def test_a_shared_cell_takes_no_second_task_of_one_family_and_edit_and_no_task_beside_an_l2_one(monkeypatch, tmp_path):
    """--share-cells: a candidate is skipped when a task of its cell has the same family and edit key (a build: its
    stage list), or when a task of its cell is written at a level that never shares (L2)."""
    def cand(cell, family, key, level="L1"):
        return {"record": {"level": level, "family": family}, "files": {}, "cell": cell, "rule_misses": [],
                "edit_key": (family, key)}
    seq = {G.SHARE_FROM: cand("c1", "build", ("tones", "conjugate")),            # the same build as task x in c1
           G.SHARE_FROM + 1: cand("c2", "build", ("tones", "filter")),            # c2 holds an L2 task
           G.SHARE_FROM + 2: cand("c1", "build", ("tones", "filter"))}            # another build: shares c1
    monkeypatch.setattr(G, "TF", tmp_path)
    monkeypatch.setattr(G, "plan", lambda split: {"gbf-test-21": ("build", "L1")})
    monkeypatch.setattr(G, "used_cells", lambda exclude=(): {"c1": "x", "c2": "y"})
    monkeypatch.setattr(G, "cell_counts", lambda exclude=(): {"c1": 1, "c2": 1})
    monkeypatch.setattr(G, "cell_tasks", lambda exclude=(): {"c1": [("L1", ("build", ("tones", "conjugate")))],
                                                            "c2": [("L2", ("modify", "x"))]})
    monkeypatch.setattr(G, "assemble", lambda tid, split, fam, lev, attempt: seq.get(attempt))
    written = []

    def fake_write(task_dir, c):
        Path(task_dir).mkdir(parents=True, exist_ok=True)
        written.append(c["cell"])
        return []
    monkeypatch.setattr(G, "write", fake_write)
    assert G.generate(["gbf-test-21"], share=True) == ["gbf-test-21"]
    assert written == ["c1"]
    first = (tmp_path / "test" / "gbf-test-21" / "build.py").read_text(encoding="utf-8").splitlines()[0]
    assert f"attempt {G.SHARE_FROM + 2}" in first
