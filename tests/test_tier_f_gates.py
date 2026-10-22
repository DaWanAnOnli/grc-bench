"""The Tier-F quality gates on a fixture task (eval plan §5.4), plus the composer and edit helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from grcbench.tier_f import compose, edit, grcfile

FX = Path(__file__).parent / "fixtures" / "tier_f_task" / "fx_modify_tone"


def test_signature_split_is_deterministic_and_covers_all_splits():
    # v2 (functional classes, the rule since 18 Oct) and v1 (ordered block ids, provenance only)
    sigs = [tuple(sorted(("tone", f"class_{i}"))) for i in range(300)]
    splits = [compose.signature_split(s) for s in sigs]
    assert splits == [compose.signature_split(s) for s in sigs]
    assert {"test", "dev", "train"} <= set(splits)
    v1 = [("analog_sig_source_x", f"blk_{i}", "blocks_file_sink") for i in range(300)]
    v1_splits = [compose.signature_split_v1(s) for s in v1]
    assert v1_splits == [compose.signature_split_v1(s) for s in v1]
    assert {"test", "dev", "train"} <= set(v1_splits)
    assert splits != [compose.signature_split_v1(s) for s in sigs]  # v2 uses its own seed


def test_composed_chain_signature_is_functional():
    g = compose.chain("probe", [compose.sig_source("src", "float", "1250", "0.5"),
                                compose.low_pass("lpf", "float", "3000", "500"),
                                compose.multiply_const("gain", "float", "2"),
                                compose.file_sink("out", "float")], {"samp_rate": "32000"})
    assert compose.signature(g) == ("filter", "tone")
    assert compose.signature_v1(g) == ("analog_sig_source_x", "low_pass_filter", "blocks_multiply_const_vxx",
                                       "blocks_file_sink")


def test_edit_operators_do_not_mutate_input():
    data = yaml.safe_load((FX / "reference.grc").read_text())
    before = grcfile.dump(data)
    edit.bypass(data, "gain")
    edit.scale_param(data, "src", "freq", 2)
    assert grcfile.dump(data) == before
    by = edit.bypass(data, "gain")
    assert ["thr", "0", "out", "0"] in by["connections"]
    assert edit.scale_param(data, "src", "freq", 0.5)["blocks"][1]["parameters"]["freq"] == "1000.0"


@pytest.mark.gnuradio
def test_composed_chain_passes_the_ladder():
    from grcbench.tier_f.ladder import check
    g = compose.chain("probe", [compose.sig_source("src", "float", "1250", "0.5"),
                                compose.low_pass("lpf", "float", "3000", "500"),
                                compose.file_sink("out", "float")], {"samp_rate": "32000"})
    task = {"observation": compose.file_observation(),
            "oracle": {"checks": [{"point": "out", "feature": "peak_freq", "fs": 32000, "expect": 1250, "tol": 20}]}}
    assert check(grcfile.dump(g), task)["first_failed"] is None


@pytest.mark.gnuradio
@pytest.mark.slow
def test_all_gates_pass_on_fixture_task():
    from grcbench.tier_f import gates
    res = gates.run(FX, reps=10, workers=8)
    assert res["passed"], res["gates"]
    assert res["gates"]["mutation"]["valid_mutants"] >= 5
    assert res["gates"]["start_fails"]["first_failed"] == "V6"
