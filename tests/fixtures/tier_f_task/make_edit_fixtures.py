"""Regenerates the edit-format Tier-F fixture fx_rewire_tone (not a benchmark item).

The golden change_graph of fx_modify_tone is a single update_params and that of fx_build_tone only
adds blocks; fx_rewire_tone needs every operation of tier_f.golden.edit_args: it removes a block
(the throttle) and its connections, replaces a block by one of another id under the same name (the
multiplier becomes a blocks_multiply_const_xx), resets a parameter the reference omits to its
catalog default (the file sink's `unbuffered`), enables a disabled block (a null sink) and adds
connections.

    PYTHONPATH=src ~/grc-agent/.venv/bin/python tests/fixtures/tier_f_task/make_edit_fixtures.py
"""

import copy
import json
from pathlib import Path

import yaml

HERE = Path(__file__).parent
OUT = HERE / "fx_rewire_tone"
base = yaml.safe_load((HERE.parent / "ladder" / "good_tone.grc").read_text())


def blk(d, name):
    return next(b for b in d["blocks"] if b["name"] == name)


start = copy.deepcopy(base)
start["options"]["parameters"]["id"] = "rewire_tone"
blk(start, "out")["parameters"]["unbuffered"] = "True"
start["blocks"].append({"name": "spare", "id": "blocks_null_sink", "parameters": {"type": "float"},
                        "states": {"state": "disabled"}})

ref = copy.deepcopy(start)
ref["blocks"] = [b for b in ref["blocks"] if b["name"] != "thr"]
blk(ref, "out")["parameters"].pop("unbuffered")
gain = blk(ref, "gain")
gain["id"] = "blocks_multiply_const_xx"
gain["parameters"] = {"type": "float", "const": "1"}
blk(ref, "spare")["states"] = {"state": "enabled"}
ref["connections"] = [["src", "0", "gain", "0"], ["gain", "0", "out", "0"], ["gain", "0", "spare", "0"]]

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "start.grc").write_text(yaml.safe_dump(start, sort_keys=False))
(OUT / "reference.grc").write_text(yaml.safe_dump(ref, sort_keys=False))
task = {
    "id": "fx-rewire-tone", "tier": "F", "split": "dev", "version": 1,
    "license": "CC-BY-4.0 (test fixture)",
    "source": {"kind": "test_fixture", "ref": "tests/fixtures/tier_f_task/make_edit_fixtures.py"},
    "cluster_id": "fixture", "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "tests"},
    "review": {"status": "not-required",
               "notes": "unit-test fixture for tier_f.golden (every change_graph operation), not a benchmark item"},
    "family": "modify", "level": "L1",
    "instruction": ("Remove the throttle so that the signal source feeds the multiplier directly, replace the "
                    "multiplier by a Multiply Const block whose constant is 1, and let the file sink buffer its "
                    "writes. Enable the disabled Null Sink and feed it from the multiplier too. Keep writing float "
                    "samples to output.dat."),
    "start_grc": "start.grc", "reference_grc": "reference.grc",
    "observation": {"points": [{"name": "out", "sink": {"block_id": "blocks_file_sink",
                                                         "params": {"file": "output.dat"}}}],
                    "run": {"n_items": 8192, "min_items": 8192, "timeout_s": 10},
                    "contract": "samples written to output.dat"},
    "oracle": {"checks": [
        {"point": "out", "feature": "peak_freq", "fs": 32000, "expect": 1000, "tol": 20, "group": "p2p"},
        {"point": "out", "feature": "rms", "expect": 0.3536, "rel_tol": 0.05, "group": "f2p"}],
        "graph_checks": [{"kind": "block_absent", "ids": ["blocks_throttle", "blocks_throttle2"], "group": "f2p"}]},
}
(OUT / "task.json").write_text(json.dumps(task, indent=1) + "\n")
print("written", OUT)
