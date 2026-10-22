"""The V1-V6 ladder must fail each known failure class at exactly the right rung.

Failure classes come from proposal Table 5 and from the GRC behaviour verified on GNU Radio 3.10.12
(unknown block ids load as silent "dummy" blocks; dangling connections are only reported by
import_data; bad enum values are silently coerced).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from grcbench.tier_f import grcfile, oracles
from grcbench.tier_f.ladder import check

FIX = Path(__file__).parent / "fixtures" / "ladder"
BASE = yaml.safe_load((FIX / "good_tone.grc").read_text())
TASK = json.loads((FIX / "good_tone.task.json").read_text())


def _variant(mutate) -> str:
    data = copy.deepcopy(BASE)
    mutate(data)
    return grcfile.dump(data)


def _block(data, name):
    return next(b for b in data["blocks"] if b["name"] == name)


# ----------------------------------------------------------------------------- pure (fast) tests
def test_extract_first_yaml_block_and_strip_thinking():
    ans = "<think>```yaml\nblocks: [wrong]\n```</think>Here:\n```python\nprint(1)\n```\n```yaml\n" \
          + (FIX / "good_tone.grc").read_text() + "```"
    out = grcfile.extract_grc(ans)
    assert out is not None and "good_tone" in out
    assert grcfile.extract_grc("no flowgraph here") is None


def test_v1_schema_errors():
    ok, errs, _ = grcfile.v1_parse("options: [\n")
    assert not ok and "does not parse" in errs[0]
    ok, errs, _ = grcfile.v1_parse("blocks: []\n")
    assert not ok and any("metadata" in e for e in errs)


def test_oracle_features():
    import numpy as np
    fs = 32000
    t = np.arange(8192) / fs
    x = np.cos(2 * np.pi * 1000 * t).astype(np.float32)
    assert abs(oracles.peak_freq(x, fs) - 1000) < 5
    assert abs(oracles.rms(x) - 0.7071) < 0.01
    assert oracles.band_fraction(x, fs, 900, 1100) > 0.95
    z = np.exp(-2j * np.pi * 2000 * t)
    assert abs(oracles.peak_freq(z, fs) + 2000) < 5          # signed frequency for complex
    assert oracles.corr_ref(np.roll(x, 7), x, 64) > 0.99
    assert oracles.equal_prefix(np.r_[np.zeros(3), np.arange(100)], np.arange(100))
    ev = oracles.evaluate(TASK["oracle"], {"out": x})
    assert ev["passed"]
    ev = oracles.evaluate(TASK["oracle"], {})
    assert not ev["passed"] and ev["checks"][0]["error"] == "observation missing"


# ----------------------------------------------------------------------------- sandboxed ladder
pytestmark_gr = pytest.mark.gnuradio


def _rungs(res):
    return {k: v.get("passed") for k, v in res["rungs"].items()}


@pytest.mark.gnuradio
def test_good_graph_passes_all_rungs():
    res = check(grcfile.dump(BASE), TASK)
    assert res["first_failed"] is None, res
    assert res["highest_passed"] == "V6"


CASES = {
    # rung V2
    "unknown_block": ("V2", lambda d: _block(d, "gain").__setitem__("id", "blocks_multiply_magic")),
    "dangling_connection": ("V2", lambda d: d["connections"].append(["gain", "0", "ghost", "0"])),
    "bad_enum": ("V2", lambda d: _block(d, "src")["parameters"].__setitem__("type", "cmplx")),
    # rung V3
    "unconnected_port": ("V3", lambda d: d["connections"].remove(["gain", "0", "out", "0"])),
    "io_size_mismatch": ("V3", lambda d: _block(d, "src")["parameters"].__setitem__("type", "complex")),
    "failed_assert": ("V3", lambda d: _block(d, "thr")["parameters"].__setitem__("vlen", "0")),
    "undefined_variable": ("V3", lambda d: _block(d, "src")["parameters"].__setitem__("freq", "undefined_var*2")),
    "double_upstream": ("V3", lambda d: (d["blocks"].append({"name": "src2", "id": "analog_sig_source_x",
                                                              "parameters": {"type": "float", "samp_rate": "samp_rate"}}),
                                         d["connections"].append(["src2", "0", "gain", "0"]))),
    # rung V6
    "wrong_frequency": ("V6", lambda d: _block(d, "src")["parameters"].__setitem__("freq", "1500")),
}


@pytest.mark.gnuradio
@pytest.mark.parametrize("case", sorted(CASES))
def test_failure_class_fails_at_expected_rung(case):
    rung, mutate = CASES[case]
    res = check(_variant(mutate), TASK)
    assert res["first_failed"] == rung, (case, _rungs(res), res["rungs"].get(rung))
    if case == "unknown_block":
        assert res["hallucinated_blocks"] == ["blocks_multiply_magic"]


@pytest.mark.gnuradio
def test_liveness_failure_at_v5():
    def mutate(d):
        d["blocks"].append({"name": "hd", "id": "blocks_head", "parameters": {"type": "float", "num_items": "100"}})
        d["connections"].remove(["gain", "0", "out", "0"])
        d["connections"] += [["gain", "0", "hd", "0"], ["hd", "0", "out", "0"]]
    res = check(_variant(mutate), TASK)
    assert res["first_failed"] == "V5", _rungs(res)
    assert any("liveness" in e for e in res["rungs"]["V5"]["errors"])


@pytest.mark.gnuradio
def test_gui_sinks_are_converted_for_v5():
    def mutate(d):
        _block(d, "out").update({"id": "qtgui_time_sink_x", "parameters": {"type": "float", "nconnections": "1",
                                                                           "size": "1024", "srate": "samp_rate"}})
        d["options"]["parameters"]["generate_options"] = "qt_gui"
    task = {"observation": {"points": [{"name": "out", "sink": {"block_name": "out"}}]},
            "oracle": TASK["oracle"]}
    res = check(_variant(mutate), task)
    assert res["first_failed"] is None, _rungs(res)
