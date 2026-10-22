"""Partial credit, the compact .grc form and the Tier-F prompt."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from grcbench.tier_f import grcfile, partial_credit, prompt

FIX = Path(__file__).parent / "fixtures" / "ladder"
BASE = yaml.safe_load((FIX / "good_tone.grc").read_text())


def test_partial_credit_identity_and_degradation():
    assert partial_credit.score(BASE, BASE)["score"] == 1.0
    renamed = copy.deepcopy(BASE)                     # block names do not matter, structure does
    for b in renamed["blocks"]:
        b["name"] = "x_" + b["name"]
    renamed["connections"] = [["x_" + s, sp, "x_" + d, dp] for s, sp, d, dp in renamed["connections"]]
    assert partial_credit.score(renamed, BASE)["score"] == 1.0
    worse = copy.deepcopy(BASE)
    next(b for b in worse["blocks"] if b["name"] == "src")["parameters"]["freq"] = "1500"
    s = partial_credit.score(worse, BASE)
    assert s["blocks"] == 1.0 and s["connections"] == 1.0 and s["parameters"] < 1.0
    empty = {"blocks": [], "connections": []}
    assert partial_credit.score(empty, BASE)["score"] == 0.0


def test_prompt_is_stable_and_never_contains_the_canary():
    from grcbench.common.canary import CANARY_GUID
    msgs = prompt.messages("Build a tone generator.", start_grc="blocks: []\n")
    assert msgs[0]["role"] == "system" and "```yaml" in msgs[1]["content"]
    assert all(CANARY_GUID not in m["content"] for m in msgs)
    assert prompt.prompt_sha256() == prompt.prompt_sha256()


@pytest.mark.gnuradio
def test_compact_form_loads_to_the_same_flowgraph():
    from grcbench.tier_f import grc
    full = copy.deepcopy(BASE)
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in full["blocks"]}
    for b in full["blocks"]:                          # make every default explicit, then compact
        for k, v in defaults[b["id"]].items():
            b["parameters"].setdefault(k, v)
    comp = grcfile.compact(full, defaults)
    assert sum(len(b["parameters"]) for b in comp["blocks"]) < sum(len(b["parameters"]) for b in full["blocks"])
    assert grc.compact_equivalent(full, comp) == []


def test_graph_checks_structure_and_parameter_only_rule():
    from grcbench.tier_f import graph_checks
    data = copy.deepcopy(BASE)
    res = graph_checks.evaluate([
        {"kind": "block_present", "ids": ["blocks_multiply_const_vxx"], "params": {"const": "2.0"}},
        {"kind": "path", "from_ids": ["analog_sig_source_x"], "to_ids": ["blocks_file_sink"],
         "via_ids": ["blocks_multiply_const_vxx"]},
        {"kind": "block_absent", "ids": ["low_pass_filter"]},
        {"kind": "same_blocks_as_start"},
        {"kind": "max_functional_blocks", "max": 4},
    ], data, start=BASE)
    assert all(r["passed"] for r in res), res
    shortcut = copy.deepcopy(BASE)                    # bypass the gain: path via multiply must fail
    shortcut["connections"] = [["src", "0", "thr", "0"], ["thr", "0", "out", "0"]]
    shortcut["blocks"] = [b for b in shortcut["blocks"] if b["name"] != "gain"]
    res = graph_checks.evaluate([
        {"kind": "path", "from_ids": ["analog_sig_source_x"], "to_ids": ["blocks_file_sink"],
         "via_ids": ["blocks_multiply_const_vxx"]},
        {"kind": "same_blocks_as_start"}], shortcut, start=BASE)
    assert [r["passed"] for r in res] == [False, False]


def test_band_fraction_counts_one_sided_power_twice():
    import numpy as np
    from grcbench.tier_f import oracles
    fs = 8000.0
    for n in (4096, 4095):                                   # even and odd lengths
        t = np.arange(n) / fs
        x = 1.0 + np.cos(2 * np.pi * 1000.0 * t)              # DC power 1, tone power 0.5
        assert abs(oracles.band_fraction(x, fs, 900, 1100) - 1 / 3) < 0.01
        assert abs(oracles.band_fraction(x, fs, -1, 1) - 2 / 3) < 0.01
        assert abs(oracles.band_fraction(x, fs, 0, fs / 2) - 1.0) < 1e-12


def test_normalise_fills_grc_hier_block_path_default():
    """An unknown block id makes GRC search the options block's hier_block_src_path while loading; in the
    isolated episode process that value otherwise reads None and import_data raises (gbf-dev-08 and
    gbf-test-10, whose fault is an unknown block). An answer's own value is kept."""
    from grcbench.tier_f._worker import _normalise
    assert _normalise({"options": {"parameters": {"id": "x"}}})["options"]["parameters"]["hier_block_src_path"] == ".:"
    assert _normalise({})["options"]["parameters"]["hier_block_src_path"] == ".:"
    kept = _normalise({"options": {"parameters": {"hier_block_src_path": "/opt/hier"}}})
    assert kept["options"]["parameters"]["hier_block_src_path"] == "/opt/hier"
