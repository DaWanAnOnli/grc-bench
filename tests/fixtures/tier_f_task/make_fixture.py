"""Regenerates the fixture Tier-F task used by tests/test_tier_f_gates.py (not a benchmark item).

    PYTHONPATH=src ~/grc-agent/.venv/bin/python tests/fixtures/tier_f_task/make_fixture.py
"""

import copy
import json
from pathlib import Path

import yaml

HERE = Path(__file__).parent
OUT = HERE / "fx_modify_tone"
base = yaml.safe_load((HERE.parent / "ladder" / "good_tone.grc").read_text())


def setp(d, blk, k, v):
    d = copy.deepcopy(d)
    next(b for b in d["blocks"] if b["name"] == blk)["parameters"][k] = v
    return d


def dump(d):
    return yaml.safe_dump(d, sort_keys=False)


ref = setp(base, "src", "freq", "2000")
(OUT / "variants").mkdir(parents=True, exist_ok=True)
(OUT / "start.grc").write_text(dump(base))
(OUT / "reference.grc").write_text(dump(ref))
(OUT / "variants" / "v1_scientific.grc").write_text(dump(setp(base, "src", "freq", "2e3")))
(OUT / "variants" / "v2_amp_instead_of_gain.grc").write_text(dump(setp(setp(ref, "src", "amp", "1.0"), "gain", "const", "1")))
task = {
    "id": "fx-modify-tone", "tier": "F", "split": "dev", "version": 1,
    "license": "CC-BY-4.0 (test fixture)",
    "source": {"kind": "test_fixture", "ref": "tests/fixtures/ladder/good_tone.grc"},
    "cluster_id": "fixture", "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "tests"},
    "review": {"status": "not-required", "notes": "unit-test fixture, not a benchmark item"},
    "family": "modify", "level": "L1",
    "instruction": "Change the tone frequency of this flowgraph to 2 kHz. Keep everything else, including "
                   "the output written to output.dat, unchanged.",
    "start_grc": "start.grc", "reference_grc": "reference.grc",
    "observation": {"points": [{"name": "out", "sink": {"block_id": "blocks_file_sink", "params": {"file": "output.dat"}}}],
                    "run": {"n_items": 8192, "min_items": 8192, "timeout_s": 10},
                    "contract": "samples written to output.dat"},
    "oracle": {"checks": [
        {"point": "out", "feature": "peak_freq", "fs": 32000, "expect": 2000, "tol": 20, "group": "f2p"},
        {"point": "out", "feature": "rms", "expect": 0.7071, "rel_tol": 0.05, "group": "p2p"}]},
    "mutations": [
        {"op": "scale_param", "block": "src", "param": "freq", "factor": 0.75, "breaks": "frequency"},
        {"op": "scale_param", "block": "src", "param": "freq", "factor": 1.5, "breaks": "frequency"},
        {"op": "bypass", "block": "gain", "breaks": "amplitude"},
        {"op": "set_param", "block": "src", "param": "amp", "value": "1.0", "breaks": "amplitude"},
        {"op": "rewire_input", "dst": "out", "dst_port": "0", "new_src": "thr", "breaks": "amplitude"},
        {"op": "set_param", "block": "samp_rate", "param": "value", "value": "48000", "breaks": "sample rate"}],
}
(OUT / "task.json").write_text(json.dumps(task, indent=1) + "\n")
print("written", OUT)
