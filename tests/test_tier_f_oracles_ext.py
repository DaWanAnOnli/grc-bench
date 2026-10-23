"""Tier-F tooling extensions of 18 Oct (reports/tier_f_reviews/fix_plan.json, tooling T1-T8).

T1 graph checks: same_connections_as_start; same_blocks_as_start variables / ignore_ids / bypass
T2 corr_ref: complex reference (ref_imag) and the signed, phase-sensitive mode
T3 integer-lattice features max_int_dev and int_fraction
T4 min (and max) sample features
T5 band_fraction remove_mean / window; peak_freq ignores DC
T6 ./gb ladder --task sets the task directory (start graph found from the CLI)
T7 observation taps never back-pressure the graph (two taps on one upstream block, different rates)
T8 message-port seeds are estimated L3
plus the task schema's feature / kind names
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from grcbench.tier_f import graph_checks, oracles

FIX = Path(__file__).parent / "fixtures"
BASE = yaml.safe_load((FIX / "ladder" / "good_tone.grc").read_text())
FX_TASK_DIR = FIX / "tier_f_task" / "fx_modify_tone"


def _gc(kind: str, data: dict, start: dict, **opts) -> dict:
    return graph_checks.evaluate([{"kind": kind, **opts}], data, start=start)[0]


def _rename(data: dict, prefix: str = "n_") -> dict:
    out = copy.deepcopy(data)
    for b in out["blocks"]:
        b["name"] = prefix + b["name"]
    out["connections"] = [[prefix + s, sp, prefix + d, dp] for s, sp, d, dp in out["connections"]]
    return out


def _block(data: dict, name: str) -> dict:
    return next(b for b in data["blocks"] if b["name"] == name)


# ----------------------------------------------------------------------------- T1 graph checks
def test_same_connections_accepts_value_changes_and_renaming():
    tuned = copy.deepcopy(BASE)
    _block(tuned, "src")["parameters"]["freq"] = "1500"
    _block(tuned, "gain")["parameters"]["const"] = "3"
    for mode in ("isomorphic", "multiset"):
        assert _gc("same_connections_as_start", tuned, BASE, mode=mode)["passed"]
        assert _gc("same_connections_as_start", _rename(tuned), BASE, mode=mode)["passed"]
    assert _gc("same_connections_as_start", BASE, BASE)["detail"] == "same connections"


def test_same_connections_rejects_rewiring_and_extra_blocks():
    rewired = copy.deepcopy(BASE)                     # same blocks, gain moved before the throttle
    rewired["connections"] = [["src", "0", "gain", "0"], ["gain", "0", "thr", "0"], ["thr", "0", "out", "0"]]
    r = _gc("same_connections_as_start", rewired, BASE)
    assert not r["passed"] and "added" in r["detail"]
    assert _gc("same_blocks_as_start", rewired, BASE)["passed"]      # the block check alone cannot see it
    inserted = copy.deepcopy(BASE)                    # an extra block in the chain
    inserted["blocks"].append({"name": "g2", "id": "blocks_multiply_const_vxx",
                               "parameters": {"type": "float", "const": "1"}})
    inserted["connections"] = [["src", "0", "thr", "0"], ["thr", "0", "gain", "0"], ["gain", "0", "g2", "0"],
                               ["g2", "0", "out", "0"]]
    assert not _gc("same_connections_as_start", inserted, BASE)["passed"]
    dropped = copy.deepcopy(BASE)
    dropped["connections"] = dropped["connections"][:-1]
    assert not _gc("same_connections_as_start", dropped, BASE)["passed"]
    disabled = copy.deepcopy(BASE)                    # disabling a block removes its connections
    _block(disabled, "gain")["states"] = {"state": "disabled"}
    assert not _gc("same_connections_as_start", disabled, BASE)["passed"]
    assert _gc("same_connections_as_start", BASE, None)["detail"] == "no start graph"


def _two_branch() -> dict:
    """Two sources of the same type; source a feeds two filters, source b a third block."""
    blk = lambda n, i: {"name": n, "id": i, "parameters": {}}  # noqa: E731
    return {"blocks": [blk("a", "src_t"), blk("b", "src_t"), blk("f1", "filt_t"), blk("f2", "filt_t"),
                       blk("k", "sink_t"), blk("s1", "sink_t"), blk("s2", "sink_t")],
            "connections": [["a", "0", "f1", "0"], ["a", "0", "f2", "0"], ["b", "0", "k", "0"],
                            ["f1", "0", "s1", "0"], ["f2", "0", "s2", "0"]]}


def test_same_connections_isomorphic_mode_catches_same_type_rewiring():
    start = _two_branch()
    swapped = copy.deepcopy(start)                    # b now feeds f2 and a feeds k: same id-level edges
    swapped["connections"] = [["a", "0", "f1", "0"], ["b", "0", "f2", "0"], ["a", "0", "k", "0"],
                              ["f1", "0", "s1", "0"], ["f2", "0", "s2", "0"]]
    assert _gc("same_connections_as_start", swapped, start, mode="multiset")["passed"]
    r = _gc("same_connections_as_start", swapped, start)
    assert not r["passed"] and "different blocks" in r["detail"]
    relabelled = _rename(start, "x")                  # renamed and listed in another order: still equal
    relabelled["blocks"].reverse()
    relabelled["connections"].reverse()
    assert _gc("same_connections_as_start", relabelled, start)["passed"]


def test_same_connections_message_ports_compare_by_name():
    blk = lambda n, i: {"name": n, "id": i, "parameters": {}}  # noqa: E731
    start = {"blocks": [blk("strobe", "blocks_message_strobe"), blk("src", "analog_sig_source_x"),
                        blk("snk", "blocks_null_sink")],
             "connections": [["strobe", "strobe", "src", "cmd"], ["src", "0", "snk", "0"]]}
    assert _gc("same_connections_as_start", copy.deepcopy(start), start)["passed"]
    other = copy.deepcopy(start)
    other["connections"][0] = ["strobe", "strobe", "src", "freq"]
    assert not _gc("same_connections_as_start", other, start)["passed"]


def test_same_blocks_variable_modes_and_ignore_ids():
    start = copy.deepcopy(BASE)
    start["blocks"].append({"name": "f0", "id": "variable", "parameters": {"value": "1000"}})
    added = copy.deepcopy(start)
    added["blocks"].append({"name": "helper", "id": "variable", "parameters": {"value": "2"}})
    revalued = copy.deepcopy(start)
    _block(revalued, "f0")["parameters"]["value"] = "1500"
    renamed = copy.deepcopy(start)
    _block(renamed, "f0")["name"] = "f_tone"
    removed = copy.deepcopy(start)
    removed["blocks"] = [b for b in removed["blocks"] if b["name"] != "f0"]

    def ok(d, **o):
        return _gc("same_blocks_as_start", d, start, **o)["passed"]

    assert ok(revalued) and ok(renamed) and not ok(added) and not ok(removed)            # "count" (default)
    assert ok(revalued, variables="value_only") and not ok(renamed, variables="value_only")
    assert not ok(added, variables="value_only") and not ok(removed, variables="value_only")
    assert all(ok(d, variables="ignore") for d in (revalued, renamed, added, removed))
    swapped_fn = copy.deepcopy(start)                 # a functional block swap still fails in every mode
    _block(swapped_fn, "gain")["id"] = "blocks_add_const_vxx"
    assert not any(ok(swapped_fn, variables=m) for m in graph_checks.VARIABLE_MODES)
    noted = copy.deepcopy(start)
    noted["blocks"].append({"name": "nb", "id": "note", "parameters": {"note": "x"}})
    assert not ok(noted) and ok(noted, ignore_ids=["note"])
    assert not _gc("same_blocks_as_start", start, start, variables="bogus")["passed"]


def test_same_blocks_counts_a_bypassed_block_as_different():
    bypassed = copy.deepcopy(BASE)
    _block(bypassed, "gain")["states"] = {"state": "bypassed"}
    r = _gc("same_blocks_as_start", bypassed, BASE)
    assert not r["passed"] and "bypassed" in r["detail"]
    assert _gc("same_blocks_as_start", bypassed, bypassed)["passed"]


def test_isomorphism_handles_many_identical_blocks():
    blk = lambda n, i: {"name": n, "id": i, "parameters": {}}  # noqa: E731
    n = 12                                            # a source fanning out to 12 identical filter -> sink chains
    start = {"blocks": [blk("src", "noise")] + [blk(f"f{i}", "fir") for i in range(n)]
             + [blk(f"s{i}", "file_sink") for i in range(n)],
             "connections": [["src", "0", f"f{i}", "0"] for i in range(n)]
             + [[f"f{i}", "0", f"s{i}", "0"] for i in range(n)]}
    perm = copy.deepcopy(start)                       # filters renamed in a permuted order
    names = {f"f{i}": f"g{(i * 5) % n}" for i in range(n)}
    for b in perm["blocks"]:
        b["name"] = names.get(b["name"], b["name"])
    perm["connections"] = [[names.get(s, s), sp, names.get(d, d), dp] for s, sp, d, dp in perm["connections"]]
    assert _gc("same_connections_as_start", perm, start)["passed"]
    chained = copy.deepcopy(start)                    # f0 -> f1 chain instead of parallel: rejected
    chained["connections"][1] = ["f0", "0", "f1", "0"]
    assert not _gc("same_connections_as_start", chained, start)["passed"]


def test_path_anchored_on_observation_points():
    """from_upstream_of / to_params (review round 2, gbf-dev-07): the source behind stream_a.dat must reach
    mux_out.dat through a merging block; a header-baked copy of stream A, or a fixed output through the mux,
    fails although some Vector Source still reaches some File Sink through the mux."""
    blk = lambda n, i, **p: {"name": n, "id": i, "parameters": p}  # noqa: E731
    sink = lambda n, f: blk(n, "blocks_file_sink", file=f, type="float")  # noqa: E731
    good = {"blocks": [blk("a", "blocks_vector_source_x"), blk("b", "blocks_vector_source_x"),
                       blk("mux", "blocks_stream_mux"), sink("out", "mux_out.dat"), sink("sa", "stream_a.dat")],
            "connections": [["a", "0", "mux", "0"], ["b", "0", "mux", "1"], ["mux", "0", "out", "0"],
                            ["a", "0", "sa", "0"]]}
    chk = {"from_ids": ["blocks_vector_source_x"], "via_ids": ["blocks_stream_mux"], "to_ids": ["blocks_file_sink"],
           "from_upstream_of": {"ids": ["blocks_file_sink"], "params": {"file": "stream_a.dat"}},
           "to_params": {"file": "mux_out.dat"}}
    assert _gc("path", good, None, **chk)["passed"]
    moved = copy.deepcopy(good)                       # absolute paths match by file name
    for b in moved["blocks"]:
        if "file" in b["parameters"]:
            b["parameters"]["file"] = "'/tmp/" + b["parameters"]["file"] + "'"
    assert _gc("path", moved, None, **chk)["passed"]
    doctored = copy.deepcopy(good)                    # stream A feeds only stream_a.dat; a copy feeds the mux
    doctored["blocks"].append(blk("a_copy", "blocks_vector_source_x"))
    doctored["connections"][0] = ["a_copy", "0", "mux", "0"]
    assert _gc("path", doctored, None, **{k: v for k, v in chk.items() if k != "from_upstream_of"})["passed"]
    assert not _gc("path", doctored, None, **chk)["passed"]
    other_sink = copy.deepcopy(good)                  # the mux output goes to another file
    other_sink["blocks"][3]["parameters"]["file"] = "frames.dat"
    assert not _gc("path", other_sink, None, **chk)["passed"]


# ----------------------------------------------------------------------------- T2 corr_ref
def _qpsk(n: int = 2048, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return ((2 * rng.integers(0, 2, n) - 1) + 1j * (2 * rng.integers(0, 2, n) - 1)) / np.sqrt(2)


def test_corr_ref_default_is_unchanged_and_phase_blind():
    sym = _qpsk()
    for wrong in (sym * 1j, sym * np.exp(1j * np.pi / 4), -sym):
        assert oracles.corr_ref(wrong, sym, 8) > 0.999
    x = np.cos(np.arange(4096) * 0.05)
    assert oracles.corr_ref(-x, x, 8) > 0.999                # sign inversion invisible by default
    assert oracles.corr_ref(np.roll(x, 7), x, 64) > 0.99


def test_corr_ref_signed_rejects_rotation_inversion_and_conjugate():
    sym = _qpsk()
    delayed = np.r_[np.zeros(5, complex), sym]
    assert oracles.corr_ref(delayed, sym, 16, signed=True) > 0.999
    assert abs(oracles.corr_ref(sym * 1j, sym, 0, signed=True)) < 1e-9
    assert abs(oracles.corr_ref(sym * np.exp(1j * np.pi / 4), sym, 0, signed=True) - np.cos(np.pi / 4)) < 1e-9
    assert oracles.corr_ref(-sym, sym, 0, signed=True) < -0.999
    assert oracles.corr_ref(np.conj(sym), sym, 0, signed=True) < 0.1
    x = np.cos(np.arange(4096) * 0.05)
    assert oracles.corr_ref(-x, x, 0, signed=True) < -0.999


def test_corr_ref_complex_reference_through_evaluate():
    sym = _qpsk(1024)
    chk = {"point": "sym", "feature": "corr_ref", "ref": sym.real.tolist(), "ref_imag": sym.imag.tolist(),
           "max_lag": 8, "signed": True, "min": 0.95}
    obs = {"sym": np.r_[np.zeros(3, np.complex64), sym.astype(np.complex64)]}
    ev = oracles.evaluate({"checks": [chk]}, obs)
    assert ev["passed"] and ev["checks"][0]["value"] > 0.999
    assert "ref" not in ev["checks"][0]["spec"] and "ref_imag" not in ev["checks"][0]["spec"]
    for wrong in (obs["sym"] * 1j, np.conj(obs["sym"]), -obs["sym"]):
        assert not oracles.evaluate({"checks": [chk]}, {"sym": wrong})["passed"]
    phase_blind = {k: v for k, v in chk.items() if k != "signed"}
    assert oracles.evaluate({"checks": [phase_blind]}, {"sym": obs["sym"] * 1j})["passed"]
    real_only = {k: v for k, v in chk.items() if k != "ref_imag"}      # real reference against a complex signal
    assert oracles.evaluate({"checks": [{**real_only, "min": 0.6}]}, obs)["passed"]
    bad = {**chk, "ref_imag": chk["ref_imag"][:-1]}
    r = oracles.evaluate({"checks": [bad]}, obs)["checks"][0]
    assert not r["passed"] and "ref_imag" in r["error"]


# ----------------------------------------------------------------------------- T3 integer lattice
def test_integer_lattice_features():
    rng = np.random.default_rng(1)
    codes = np.rint(rng.normal(0, 8, 4096)).astype(np.float32)
    assert oracles.max_int_dev(codes) == 0.0 and oracles.int_fraction(codes) == 1.0
    midrise = codes + 0.5
    assert abs(oracles.max_int_dev(midrise) - 0.5) < 1e-6 and oracles.int_fraction(midrise) == 0.0
    quantized = (codes / 32).astype(np.float32)               # multiples of 1/32
    assert oracles.max_int_dev(quantized, 32) < 1e-6
    assert oracles.max_int_dev(quantized) > 0.4
    noisy = codes + rng.uniform(-0.5, 0.5, codes.shape) / 64
    assert oracles.max_int_dev(noisy) > 1e-3 and oracles.int_fraction(noisy) < 0.05
    assert oracles.int_fraction(noisy, int_tol=0.01) == 1.0
    cplx = codes + 1j * (codes + 0.25)
    assert abs(oracles.max_int_dev(cplx) - 0.25) < 1e-6 and oracles.int_fraction(cplx) == 0.0
    chks = [{"point": "c", "feature": "max_int_dev", "max": 1e-4},
            {"point": "q", "feature": "max_int_dev", "scale": 32, "max": 1e-4},
            {"point": "c", "feature": "int_fraction", "min": 0.999, "skip": 10}]
    assert oracles.evaluate({"checks": chks}, {"c": codes, "q": quantized})["passed"]
    assert not oracles.evaluate({"checks": chks}, {"c": midrise, "q": quantized + 1 / 64})["passed"]


# ----------------------------------------------------------------------------- T4 min / max
def test_min_and_max_features():
    t = np.arange(8192) / 8000.0
    x = (0.5 + 0.35 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    assert abs(oracles.sample_min(x) - 0.15) < 1e-3 and abs(oracles.sample_max(x) - 0.85) < 1e-3
    chks = [{"point": "x", "feature": "min", "min": 0.149}, {"point": "x", "feature": "max", "max": 0.851}]
    assert oracles.evaluate({"checks": chks}, {"x": x})["passed"]
    sag = x.copy()
    sag[5000] = 0.1                                           # one sample below the stated lower bound
    assert not oracles.evaluate({"checks": chks}, {"x": sag})["passed"]
    z = np.array([1 + 5j, -2 - 9j, 3 + 0j])                   # complex: the real part decides
    assert oracles.sample_min(z) == -2.0 and oracles.sample_max(z) == 3.0
    assert np.isnan(oracles.sample_min(np.array([])))
    ev = oracles.evaluate({"checks": [{"point": "x", "feature": "min", "min": 0, "n": 0, "skip": 99999}]}, {"x": x})
    assert not ev["passed"]                                   # nothing captured: NaN fails


def test_complex_mean_is_the_real_part():
    z = np.full(64, 0.25 - 3j, np.complex64)                  # unchanged behaviour, now documented
    ev = oracles.evaluate({"checks": [{"point": "z", "feature": "mean", "expect": 0.25, "tol": 1e-6, "min": 0}]},
                          {"z": z})
    assert ev["passed"] and ev["checks"][0]["value"] == 0.25
    json.dumps(ev)


# ----------------------------------------------------------------------------- T5 band_fraction / peak_freq
def test_band_fraction_remove_mean_measures_ac_purity():
    fs, n = 24000.0, 12000                                   # 2 Hz bins: tone and spur sit on bins
    t = np.arange(n) / fs
    tone = 0.35 * np.sin(2 * np.pi * 5100 * t)
    spur = 0.35 * 10 ** (-28 / 20) * np.sin(2 * np.pi * 1500 * t)    # a -28 dBc spur
    clean, dirty = 0.5 + tone, 0.5 + tone + spur
    # default (unchanged): the DC level counts, so the fraction is ~0.2 and the spur barely moves it
    assert oracles.band_fraction(clean, fs, 5000, 5200) < 0.5
    assert abs(oracles.band_fraction(dirty, fs, 5000, 5200) - oracles.band_fraction(clean, fs, 5000, 5200)) < 1e-3
    pure = oracles.band_fraction(clean, fs, 5000, 5200, remove_mean=True)
    spurred = oracles.band_fraction(dirty, fs, 5000, 5200, remove_mean=True)
    assert pure > 0.999 and spurred < 0.999
    for const in (np.full(1024, 0.7), np.full(1000, 0.7, np.float32), np.zeros(64)):
        assert oracles.band_fraction(const, fs, 0, fs / 2, remove_mean=True) == 0.0
    chk = {"point": "x", "feature": "band_fraction", "fs": fs, "lo": 5000, "hi": 5200, "remove_mean": True,
           "min": 0.999}
    assert oracles.evaluate({"checks": [chk]}, {"x": clean})["passed"]
    assert not oracles.evaluate({"checks": [chk]}, {"x": dirty})["passed"]


def test_band_fraction_window_reduces_leakage():
    fs, n = 48000.0, 16384
    t = np.arange(n) / fs
    off_bin = np.sin(2 * np.pi * 2010 * t)                   # not on an FFT bin: leaks without a window
    plain = oracles.band_fraction(off_bin, fs, 1900, 2100)
    hann = oracles.band_fraction(off_bin, fs, 1900, 2100, window="hann")
    black = oracles.band_fraction(off_bin, fs, 1900, 2100, window="blackman")
    assert plain < 0.9999 and hann > 0.99999 and black > 0.99999
    with pytest.raises(ValueError):
        oracles.band_fraction(off_bin, fs, 1900, 2100, window="kaiser")
    ev = oracles.evaluate({"checks": [{"point": "x", "feature": "band_fraction", "fs": fs, "lo": 0, "hi": 1,
                                       "window": "bogus", "max": 1}]}, {"x": off_bin})
    assert not ev["passed"] and "window" in ev["checks"][0]["error"]


def test_peak_freq_excludes_dc_and_constant_has_no_peak():
    fs = 8000.0
    t = np.arange(4096) / fs
    x = 5.0 + 0.1 * np.cos(2 * np.pi * 1000 * t)             # a large DC level does not win
    assert abs(oracles.peak_freq(x, fs) - 1000) < 5
    assert np.isnan(oracles.peak_freq(np.full(4096, 2.0), fs))
    assert np.isnan(oracles.peak_freq(np.full(4096, 0.7, np.float32), fs))   # not exactly representable
    assert np.isnan(oracles.peak_freq(np.zeros(4096), fs))
    assert np.isnan(oracles.peak_freq(np.full(4096, 1 + 1j, np.complex64), fs))


# ----------------------------------------------------------------------------- schema names
def test_schema_lists_every_feature_and_check_kind():
    from grcbench.common import schemas
    assert tuple(schemas.TIER_F_FEATURES) == oracles.FEATURES
    assert tuple(schemas.TIER_F_GRAPH_CHECKS) == graph_checks.KINDS
    for name in oracles.FEATURES + graph_checks.KINDS:      # every name is documented in its module
        doc = oracles.__doc__ if name in oracles.FEATURES else graph_checks.__doc__
        assert f"  {name} " in doc, name


def test_schema_rejects_unknown_feature_or_kind():
    from grcbench.common import schemas
    from grcbench.tier_f import taskio
    from grcbench.common.canary import CANARY
    rec = taskio.load(FX_TASK_DIR)
    rec = {**{k: v for k, v in rec.items() if not k.startswith("_")}, "canary": CANARY}   # as taskio.save adds it
    assert schemas.validate(rec) == []
    bad = copy.deepcopy(rec)
    bad["oracle"]["checks"][0]["feature"] = "peak_frequency"
    assert any("peak_frequency" in e for e in schemas.validate(bad))
    bad = copy.deepcopy(rec)
    bad["oracle"]["graph_checks"] = [{"kind": "same_wires_as_start"}]
    assert any("same_wires_as_start" in e for e in schemas.validate(bad))
    good = copy.deepcopy(rec)
    good["oracle"]["checks"].append({"point": "out", "feature": "min", "min": -1})
    good["oracle"]["graph_checks"] = [{"kind": "same_connections_as_start", "group": "p2p"},
                                      {"kind": "same_blocks_as_start", "variables": "value_only"}]
    assert schemas.validate(good) == []


def test_every_tier_f_task_validates():
    from grcbench.common import schemas
    from grcbench.tier_f import taskio
    for d in taskio.task_dirs():
        rec = {k: v for k, v in taskio.load(d).items() if not k.startswith("_")}
        assert schemas.validate(rec) == [], d.name


# ----------------------------------------------------------------------------- T6 CLI task directory
def test_cli_task_loading_sets_the_task_directory():
    from grcbench.tier_f import ladder
    task = ladder.load_task_file(FX_TASK_DIR / "task.json")
    assert Path(task["_dir"]) == FX_TASK_DIR.resolve()
    start = ladder._start_data(task)
    assert start is not None and start == yaml.safe_load((FX_TASK_DIR / task["start_grc"]).read_text())
    plain = json.loads((FX_TASK_DIR / "task.json").read_text())       # what the CLI used to do
    assert ladder._start_data(plain) is None


# ----------------------------------------------------------------------------- T8 seed levels
def test_message_port_seeds_are_level_l3():
    from grcbench.tier_f import seeds
    blk = lambda n, i: {"name": n, "id": i, "parameters": {}}  # noqa: E731
    data = {"blocks": [blk("strobe", "blocks_message_strobe"), blk("rnd", "blocks_message_strobe_random"),
                       blk("src", "analog_sig_source_x"), blk("thr", "blocks_throttle2"), blk("snk", "blocks_null_sink")]}
    f = seeds.features(data)
    assert f["level_estimate"] == "L3" and {"message", "strobe"} <= set(f["l3_hints"])
    dbg = {"blocks": [blk("src", "analog_sig_source_x"), blk("dbg", "blocks_message_debug")]}
    assert seeds.features(dbg)["level_estimate"] == "L3"
    plain = {"blocks": [blk("src", "analog_sig_source_x"), blk("snk", "blocks_null_sink")]}
    assert seeds.features(plain)["level_estimate"] == "L1"


# ----------------------------------------------------------------------------- T7 taps never back-pressure
SHARED_UPSTREAM_GRC = """\
options:
  parameters:
    id: shared_upstream
    generate_options: no_gui
    output_language: python
blocks:
- name: samp_rate
  id: variable
  parameters:
    value: '32000'
- name: src
  id: analog_sig_source_x
  parameters:
    type: float
    samp_rate: samp_rate
    waveform: analog.GR_COS_WAVE
    freq: '10'
    amp: '1'
- name: keep
  id: blocks_keep_one_in_n
  parameters:
    type: float
    n: '64'
- name: fast_sink
  id: blocks_file_sink
  parameters:
    type: float
    file: fast.dat
- name: slow_sink
  id: blocks_file_sink
  parameters:
    type: float
    file: slow.dat
connections:
- [src, '0', fast_sink, '0']
- [src, '0', keep, '0']
- [keep, '0', slow_sink, '0']
metadata:
  file_format: 1
"""
# fast.dat fills after 8192 source samples; slow.dat (keep 1 in 64) needs 64 x 8192 = 524,288 of them
# from the same source buffer. A tap that stops reading once full would stall the shared source.
SHARED_UPSTREAM_TASK = {
    "observation": {"points": [
        {"name": "fast", "sink": {"block_id": "blocks_file_sink", "params": {"file": "fast.dat"}}},
        {"name": "slow", "sink": {"block_id": "blocks_file_sink", "params": {"file": "slow.dat"}}}],
        "run": {"n_items": 8192, "min_items": 8192, "timeout_s": 20}},
    "oracle": {"checks": [{"point": "fast", "feature": "max", "expect": 1.0, "tol": 1e-3},
                          {"point": "slow", "feature": "rms", "expect": 0.7071, "rel_tol": 0.01},
                          {"point": "slow", "feature": "peak_freq", "fs": 500, "expect": 10, "tol": 0.5}]},
}


@pytest.mark.gnuradio
def test_full_tap_does_not_stall_a_shared_upstream_block():
    from grcbench.tier_f.ladder import check
    res = check(SHARED_UPSTREAM_GRC, SHARED_UPSTREAM_TASK)
    v5 = res["rungs"]["V4"]
    assert res["first_failed"] is None, (v5.get("errors"), v5.get("items"), res["rungs"].get("V5"))
    assert v5["items"] == {"fast": 8192, "slow": 8192}       # each tap keeps exactly n_items


@pytest.mark.gnuradio
def test_taps_keep_liveness_and_capture_vectors():
    from grcbench.tier_f.ladder import check
    from grcbench.tier_f import grcfile
    data = yaml.safe_load(SHARED_UPSTREAM_GRC)
    # liveness still fails when a point cannot deliver min_items within the timeout (keep 1 in 10^7:
    # slow.dat would need 8.2e10 source samples), while the other point fills normally
    _block(data, "keep")["parameters"]["n"] = "10000000"
    task = {**SHARED_UPSTREAM_TASK, "observation": {**SHARED_UPSTREAM_TASK["observation"],
                                                    "run": {"n_items": 8192, "min_items": 8192, "timeout_s": 3}}}
    res = check(grcfile.dump(data), task)
    assert res["first_failed"] == "V4" and any("liveness" in e for e in res["rungs"]["V4"]["errors"])
    assert res["rungs"]["V4"]["items"]["fast"] == 8192 and res["rungs"]["V4"]["items"]["slow"] < 8192
    # complex vector items: an edge point on a stream_to_vector output (vlen 4) is captured as (n, 4)
    vec = yaml.safe_load(SHARED_UPSTREAM_GRC)
    vec["blocks"] = [b for b in vec["blocks"] if b["name"] not in ("keep", "slow_sink")]
    _block(vec, "src")["parameters"]["type"] = "complex"
    _block(vec, "fast_sink")["parameters"].update({"type": "complex", "vlen": "4"})
    vec["blocks"].append({"name": "s2v", "id": "blocks_stream_to_vector",
                          "parameters": {"type": "complex", "num_items": "4", "vlen": "1"}})
    vec["connections"] = [["src", "0", "s2v", "0"], ["s2v", "0", "fast_sink", "0"]]
    vtask = {"observation": {"points": [{"name": "v", "edge": {"block": "s2v", "port": "0"}}],
                             "run": {"n_items": 1000, "min_items": 1000, "timeout_s": 10}},
             "oracle": {"checks": [{"point": "v", "feature": "rms", "expect": 1.0, "rel_tol": 0.01}]}}
    res = check(grcfile.dump(vec), vtask, keep=True)
    try:
        assert res["first_failed"] is None, res["rungs"]
        arr = np.load(Path(res["run_dir"]) / "obs" / "v.npy")
        assert arr.shape == (1000, 4) and arr.dtype == np.complex64
        assert np.allclose(np.abs(arr), 1.0, atol=1e-4)
    finally:
        import shutil
        shutil.rmtree(res.get("run_dir", ""), ignore_errors=True)
