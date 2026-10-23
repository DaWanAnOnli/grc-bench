"""Authoring script for GRC-Bench Tier-F item gbf-test-04 (build, L1, composed chain), version 2.

Theme: a unipolar DAC test tone. In the empty flowgraph (samp_rate only) the user asks for a 1.25 kHz
sinusoid at 48 kHz that swings between 0.15 and 0.95 (peak amplitude 0.4 around a DC level of 0.55),
written as float samples to dac_test.dat. The instruction states the outcome only (round-1 fix: the
old block-by-block recipe contradicted the accepted variants); every tolerance it states is checked:

    frequency      peak_freq 1250 Hz +- 1 % (12.5 Hz)
    DC level       mean 0.55 +- 0.01
    swing          min 0.15 +- 0.01 and max 0.95 +- 0.01  (T4 features; together they pin the peak
                   amplitude to 0.4 +- 0.01 and keep the signal inside the DAC's 0..1 range)
    clean sinusoid band_fraction of the AC part (mean removed, Hann window; T5) within 50 Hz of
                   1.25 kHz >= 0.99, i.e. harmonics/noise/other components <= 1 % of the AC power.
                   A triangle wave keeps only 96/pi^4 = 98.55 % in its fundamental, a square wave
                   8/pi^2 = 81 %, a sawtooth 6/pi^2 = 61 %, so all three fail even with the right swing.

All checks run on the samples written from 0.1 s on (skip 4800 of 16384 captured items), so a correct
design whose filter or start-up transient settles within 0.1 s is not penalised. The run sentence
(fix plan R5) is recorded as a V4 liveness condition (16384 items), nothing more.

Signature v2: the reference and every variant except v5/v6 are {tone} -> test (cluster chain:tone).
v5 (Vector Source lookup table, {data_source}) and v6 (sine with a phase offset through a Rail clamp,
{nonlinear, tone}) are reviewer-shown structural alternatives whose v2 signatures hash to train; they
are listed in review.notes for the reserved-signature step (fix plan R1/R2).

Wrong answers that a single-edit mutant of the reference cannot express (a triangle or sawtooth with
the right swing, an amplitude error hidden by a DC shift, a Head-limited run) are kept as probes:

    PYTHONPATH=src ~/grc-agent/.venv/bin/python items/tier_f/test/gbf-test-04/build.py           # write
    PYTHONPATH=src ~/grc-agent/.venv/bin/python items/tier_f/test/gbf-test-04/build.py --probes  # check

Deterministic; needs GNU Radio (catalog defaults are used to write the compact files).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from grcbench.tier_f import compose as C          # noqa: E402
from grcbench.tier_f import grc, grcfile, taskio  # noqa: E402

TASK_ID = "gbf-test-04"
SPLIT = "test"
VERSION = 2

# ---- the specification (every number below is stated in the instruction) -----------------------
FS = 48000                 # sample rate, Hz
F_TONE = 1250              # tone frequency, Hz
AMP = 0.4                  # peak amplitude
DC = 0.55                  # DC level
LOW, HIGH = round(DC - AMP, 6), round(DC + AMP, 6)          # 0.15 .. 0.95
OUT_FILE = "dac_test.dat"

FREQ_TOL = round(0.01 * F_TONE, 6)          # 1 % -> 12.5 Hz
LEVEL_TOL = 0.01                            # mean, lowest and highest sample
PURITY_MIN = 0.99                           # >= 99 % of the AC power in the tone
PURITY_HALF_BAND = 50                       # Hz: everything within 50 Hz of 1.25 kHz counts as the tone
SETTLE_S = 0.1                              # checks start 0.1 s after the start
SKIP = int(SETTLE_S * FS)                   # 4800 samples
N_ITEMS = 16384                             # captured items (V4 liveness); 11584 are measured

INSTRUCTION = (
    "In this empty flowgraph, set the sample rate to 48 kHz and build a test signal for a DAC input that only accepts values "
    "between 0 and 1: a 1.25 kHz sinusoid that swings between 0.15 and 0.95, that is, a peak amplitude "
    "of 0.4 around a DC level of 0.55. Write it to a File Sink with the file name dac_test.dat and item "
    "type float. How you generate the signal is up to you. Measured on the samples written from 0.1 s "
    "after the start, it must meet these tolerances: frequency within 1 % of 1.25 kHz; mean (DC level) "
    "within 0.01 of 0.55; lowest sample within 0.01 of 0.15 and highest sample within 0.01 of 0.95; and "
    "a clean sinusoid, meaning that harmonics, noise and any other components together carry at most "
    "1 % of the power of the signal's AC part (the signal minus its mean), counting everything within "
    "50 Hz of 1.25 kHz as the tone. The flowgraph should keep producing samples until it is stopped; do "
    "not limit the number of samples."
)


def compact(data: dict) -> dict:
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in data["blocks"]}
    out = grcfile.compact(data, defaults)
    out["options"] = data["options"]
    diffs = grc.compact_equivalent(data, out)
    assert not diffs, f"compact form differs from the full form: {diffs}"
    return out


# ---- reference -----------------------------------------------------------------------------------
def reference_full() -> dict:
    """Unit cosine, scaled by 0.4 (Multiply Const), lifted by 0.55 (Add Const)."""
    blocks = [
        C.sig_source("src", "float", freq=F_TONE, amp=1),
        C.multiply_const("gain", "float", AMP),
        C.add_const("offset", "float", DC),
        C.file_sink("out", "float", OUT_FILE),
    ]
    return C.chain("dac_test_signal", blocks, {"samp_rate": str(FS)})


# ---- equivalent variants (genuinely different correct solutions) --------------------------------
def variant_source_params() -> dict:
    """Amplitude and DC offset set directly on the Signal Source (no Multiply/Add Const blocks)."""
    blocks = [
        C.sig_source("tone", "float", freq="1.25e3", amp="0.4", offset="0.55"),
        C.file_sink("dac_out", "float", OUT_FILE),
    ]
    return C.chain("dac_test_v1", blocks, {"samp_rate": "48e3"})


def variant_offset_then_scale() -> dict:
    """Sine instead of cosine, a Throttle, variables, and the offset added *before* scaling
    ((x + 0.55/0.4) * 0.4 = 0.4 x + 0.55)."""
    blocks = [
        C.sig_source("src", "float", freq="tone_freq", amp="1", waveform="analog.GR_SIN_WAVE"),
        C.block("thr", "blocks_throttle2", type="float", samples_per_second="samp_rate"),
        C.add_const("pre_offset", "float", "dc_level / scale"),
        C.multiply_const("scale_blk", "float", "scale"),
        C.file_sink("sink", "float", "/tmp/dac_test.dat"),
    ]
    return C.chain("dac_test_v2", blocks, {"samp_rate": "48000", "tone_freq": "1250",
                                           "scale": "0.4", "dc_level": "0.55"})


def variant_const_source_adder() -> dict:
    """DC level from a Constant Source summed in with an Add block."""
    blocks = [
        C.sig_source("src", "float", freq="1250", amp="1"),
        C.multiply_const("gain", "float", "0.4"),
        C.block("dc", "analog_const_source_x", type="float", const="0.55"),
        C.block("adder", "blocks_add_xx", type="float", num_inputs="2"),
        C.file_sink("out", "float", OUT_FILE),
    ]
    conns = [["src", "0", "gain", "0"], ["gain", "0", "adder", "0"], ["dc", "0", "adder", "1"],
             ["adder", "0", "out", "0"]]
    return C.graph("dac_test_v3", blocks, conns, {"samp_rate": "48000"})


def variant_complex_real_part() -> dict:
    """Reviewer alternative (item-quality c4, plus the quoted file-name expression): complex unit
    tone -> Complex to Real -> Multiply Const (blocks_multiply_const_xx) -> Add Const, legacy Throttle,
    the file name given as a Python string expression in an absolute path."""
    blocks = [
        C.sig_source("csrc", "complex", freq="1250", amp="1"),
        C.block("legacy_thr", "blocks_throttle", type="complex", samples_per_second="samp_rate"),
        C.block("re", "blocks_complex_to_real"),
        C.block("scale", "blocks_multiply_const_xx", type="float", const="0.4"),
        C.add_const("lift", "float", "0.55"),
        C.file_sink("snk", "float", "'/tmp/' + 'dac_test.dat'"),
    ]
    return C.chain("dac_test_v4", blocks, {"samp_rate": "48000"})


def variant_vector_table() -> dict:
    """Reviewer alternative (item-quality c5): a repeating Vector Source holding a lookup table of
    5 periods (192 samples = 5 x 38.4) of 0.55 + 0.4 cos(2 pi 1250 n / 48000)."""
    blocks = [
        C.block("table", "blocks_vector_source_x", type="float", repeat="True",
                vector="[0.55 + 0.4*math.cos(2*math.pi*1250*n/48000) for n in range(192)]"),
        C.block("thr", "blocks_throttle2", type="float", samples_per_second="samp_rate"),
        C.file_sink("dac", "float", OUT_FILE),
    ]
    g = C.chain("dac_test_v5", blocks, {"samp_rate": "48000"})
    g["blocks"].insert(0, {"name": "import_math", "id": "import", "parameters": {"imports": "import math"}})
    return g


def variant_rail_guard() -> dict:
    """Reviewer alternative (item-quality, 'Rail clip with sine and phase'): a sine with a phase
    offset, amplitude and offset on the source, then a Rail clamp to the DAC's 0..1 range (inactive
    here, a safety guard)."""
    blocks = [
        C.block("src", "analog_sig_source_x", type="float", samp_rate="samp_rate",
                waveform="analog.GR_SIN_WAVE", freq="1250", amp="0.4", offset="0.55", phase="0.7"),
        C.block("clamp", "analog_rail_ff", lo="0", hi="1"),
        C.file_sink("out", "float", OUT_FILE),
    ]
    return C.chain("dac_test_v6", blocks, {"samp_rate": "48000"})


VARIANTS = {
    "v1_source_amp_offset.grc": variant_source_params,
    "v2_offset_then_scale_sine.grc": variant_offset_then_scale,
    "v3_const_source_adder.grc": variant_const_source_adder,
    "v4_complex_real_legacy_throttle.grc": variant_complex_real_part,
    "v5_vector_lookup_table.grc": variant_vector_table,
    "v6_sine_phase_rail_guard.grc": variant_rail_guard,
}


# ---- probes: answers a single-edit mutant cannot express (expected verdicts) --------------------
def _source_only(gid: str, **src) -> dict:
    blocks = [C.block("src", "analog_sig_source_x", type="float", samp_rate="samp_rate", **src),
              C.file_sink("out", "float", OUT_FILE)]
    return C.chain(gid, blocks, {"samp_rate": "48000"})


def probes() -> dict[str, tuple[dict, bool, str]]:
    """name -> (flowgraph, expected to pass V5, what it shows)."""
    lpf = C.chain("probe_lpf", [
        C.sig_source("src", "float", freq="1250", amp="0.4", offset="0.55"),
        C.low_pass("lpf", "float", cutoff="3000", width="1000"),
        C.file_sink("out", "float", OUT_FILE)], {"samp_rate": "48000"})
    head = C.chain("probe_head", [
        C.sig_source("src", "float", freq="1250", amp="0.4", offset="0.55"),
        C.block("hd", "blocks_head", type="float", num_items="8192"),
        C.file_sink("out", "float", OUT_FILE)], {"samp_rate": "48000"})
    return {
        "triangle_right_swing": (_source_only("probe_tri", waveform="analog.GR_TRI_WAVE", freq="1250",
                                              amp="0.8", offset="0.15"), False,
                                 "round-1 compliance p2_tri / item-quality c2: triangle 0.15..0.95 (98.55 % "
                                 "in the fundamental) must fail the clean-sinusoid check"),
        "sawtooth_right_swing": (_source_only("probe_saw", waveform="analog.GR_SAW_WAVE", freq="1250",
                                              amp="0.8", offset="0.15"), False,
                                 "item-quality c1: sawtooth 0.15..0.95 must fail"),
        "square_right_swing": (_source_only("probe_sqr", waveform="analog.GR_SQR_WAVE", freq="1250",
                                            amp="0.8", offset="0.15"), False,
                                 "square wave 0.15..0.95 (mean 0.55) must fail the clean-sinusoid check"),
        "amp_0p37_dc_0p565": (_source_only("probe_c7", freq="1250", amp="0.37", offset="0.565"), False,
                              "item-quality c7: amplitude 7.5 % low hidden by a DC shift (0.195..0.935) "
                              "must fail the swing checks"),
        "lowpass_cleanup": (lpf, True, "a correct tone through a 3 kHz low-pass: the filter's start-up "
                            "transient lies inside the first 0.1 s, so it passes"),
        "head_8192": (head, False, "item-quality c3: a Head of 8192 items stops before the 16384 captured "
                      "items (V4 liveness). A Head of more than 16384 items would pass (documented limit)"),
    }


# ---- task record ---------------------------------------------------------------------------------
def record(sig: tuple[str, ...], variant_sigs: dict[str, tuple[str, ...]]) -> dict:
    point = "out"
    win = {"point": point, "skip": SKIP}
    rms_note = math.sqrt(DC ** 2 + AMP ** 2 / 2)
    other = {k: "|".join(v) for k, v in variant_sigs.items() if v != sig}
    notes = (
        "Round-1 fix (v2, 18 Oct): behavioural instruction (outcome only; the method is free), stated "
        "tolerances, swing checked with min/max (T4) and waveform purity with a mean-free band_fraction "
        "(T5), 0.1 s settle window, run sentence per R5, cluster_id from signature v2 (chain:tone). "
        "Old twins gbf-dev-03 and gbf-test-11 were re-themed in the same round; gbf-dev-01 too. "
        "Known limits: (1) the sample rate is observable only through the frequency/sample-rate ratio "
        "(a graph at another samp_rate with the tone moved proportionally passes); (2) 'do not limit the "
        "number of samples' is enforced only as V4 liveness (16384 items): a Head of more than 16384 items "
        "passes; (3) the first 0.1 s is not checked. Variant signatures that differ from the cell, for "
        "the reserved-signature step (R1/R2): "
        + "; ".join(f"{k}: {v}" for k, v in sorted(other.items()))
        + " (reviewer-shown alternatives; both hash to train). The v6 cell nonlinear|tone is also the cell "
        "of a gbf-test-01 variant (max(x, -x) rectifier) at the time of writing; both are test items, so "
        "this overlap stays inside the test split. Wrong answers that are not single-edit "
        "mutants are kept as probes in build.py (--probes): triangle, sawtooth and square with the right "
        "swing, amplitude error hidden by a DC shift and a Head of 8192 items fail; a low-pass clean-up "
        "stage passes. Role-table (round-1 F5) and canary-policy (F6) findings are team decisions "
        "(fix_plan team_decisions 4 and 5), not item edits."
    )
    return {
        "id": TASK_ID, "tier": "F", "split": SPLIT, "version": VERSION,
        "license": "CC-BY-4.0 (original GRC-Bench item; licence to be confirmed by the team)",
        "source": {"kind": "composed_chain", "ref": "|".join(sig)},
        "cluster_id": C.cluster_id(sig),
        "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring (round-1 fix)",
                       "generator_version": "0.2.0"},
        "review": {"status": "pending-team-review", "reviewers": [], "notes": notes},
        "family": "build", "level": "L1",
        "instruction": INSTRUCTION,
        "start_grc": None,
        "reference_grc": "reference.grc",
        "observation": {
            "points": [{"name": point, "sink": {"block_id": "blocks_file_sink",
                                                "params": {"file": OUT_FILE, "type": "float"}}}],
            "run": {"n_items": N_ITEMS, "min_items": N_ITEMS, "timeout_s": 10.0},
            "contract": f"the float samples written to the File Sink whose file name is {OUT_FILE} "
                        f"(item type float); checks use the samples from 0.1 s (sample {SKIP}) on",
        },
        "oracle": {"checks": [
            {**win, "feature": "peak_freq", "fs": FS, "expect": F_TONE, "tol": FREQ_TOL,
             "group": "main", "property": "tone frequency"},
            {**win, "feature": "mean", "expect": DC, "tol": LEVEL_TOL, "group": "main", "property": "DC level"},
            {**win, "feature": "min", "expect": LOW, "tol": LEVEL_TOL, "group": "main", "property": "swing"},
            {**win, "feature": "max", "expect": HIGH, "tol": LEVEL_TOL, "group": "main", "property": "swing"},
            {**win, "feature": "band_fraction", "fs": FS, "lo": F_TONE - PURITY_HALF_BAND,
             "hi": F_TONE + PURITY_HALF_BAND, "remove_mean": True, "window": "hann", "min": PURITY_MIN,
             "group": "main", "property": "clean sinusoid"},
        ]},
        "spec_properties": [
            "tone frequency: strongest component of the AC part at 1250 Hz +- 12.5 Hz (1 %), measured at "
            "fs = 48 kHz",
            "sample rate 48 kHz: observable only through the frequency/sample-rate ratio (a 1.25 kHz tone "
            "at another samp_rate lands on another frequency on the 48 kHz time base)",
            "DC level: mean 0.55 +- 0.01",
            "swing: lowest sample 0.15 +- 0.01 and highest sample 0.95 +- 0.01; together they pin the peak "
            "amplitude to 0.4 +- 0.01 and the mid level to 0.55 +- 0.01, and keep the signal inside the "
            "DAC's 0..1 range",
            "clean sinusoid: at least 99 % of the AC power (mean removed, Hann window) within 50 Hz of "
            "1.25 kHz; a triangle (98.55 %), square (81 %) or sawtooth (61 %) with the same swing fails",
            f"measurement window: samples {SKIP}..{N_ITEMS - 1} (from 0.1 s after the start); RMS of the "
            f"target signal for reference: sqrt(0.55^2 + 0.4^2/2) = {rms_note:.4f} (not checked separately)",
            "observation point: the File Sink dac_test.dat with item type float (a sink of another type "
            "does not match the observation point)",
            "run condition: the flowgraph keeps producing samples; checked only as V4 liveness (16384 "
            "items must arrive at the observation point)",
        ],
        "mutations": [
            {"op": "scale_param", "block": "src", "param": "freq", "factor": 0.8, "breaks": "tone frequency"},
            {"op": "scale_param", "block": "src", "param": "freq", "factor": 1.02,
             "breaks": "tone frequency (near miss: 1275 Hz, 2 % high)"},
            {"op": "set_param", "block": "samp_rate", "param": "value", "value": "32000",
             "breaks": "sample rate (tone lands on 1875 Hz on the 48 kHz time base)"},
            {"op": "scale_param", "block": "gain", "param": "const", "factor": 0.9,
             "breaks": "swing (amplitude 0.36)"},
            {"op": "scale_param", "block": "gain", "param": "const", "factor": 1.05,
             "breaks": "swing (near miss: amplitude 0.42, 0.13..0.97)"},
            {"op": "bypass", "block": "gain", "breaks": "swing (amplitude 1, outside 0..1)"},
            {"op": "set_param", "block": "src", "param": "amp", "value": "0.4",
             "breaks": "swing (scale applied twice: amplitude 0.16)"},
            {"op": "set_param", "block": "offset", "param": "const", "value": "0.565",
             "breaks": "DC level (near miss: +0.015)"},
            {"op": "set_param", "block": "offset", "param": "const", "value": "0.5", "breaks": "DC level"},
            {"op": "bypass", "block": "offset", "breaks": "DC level (no offset; negative samples)"},
            {"op": "rewire_input", "dst": "out", "dst_port": "0", "new_src": "src",
             "breaks": "DC level and swing (wrong tap: the raw unit tone, before scaling and offset)"},
            {"op": "set_param", "block": "src", "param": "waveform", "value": "analog.GR_SQR_WAVE",
             "breaks": "clean sinusoid (square wave)"},
            {"op": "set_param", "block": "src", "param": "waveform", "value": "analog.GR_TRI_WAVE",
             "breaks": "clean sinusoid (triangle wave)"},
            {"op": "set_param", "block": "src", "param": "waveform", "value": "analog.GR_SAW_WAVE",
             "breaks": "clean sinusoid (sawtooth wave)"},
        ],
        "perturbation": {"perturbed": False, "changes": []},
        "disjointness": (
            "A single 1.25 kHz sinusoid placed into a 0..1 DAC range (mean 0.55, swing 0.15..0.95) and written "
            "to a file. Nearest training-side specs (reports/reference/training_task_specs.json): GRC-Agent "
            "scenario 14_build_chain_from_scratch (from the same empty graph: a float Signal Source with "
            "freq 1000 and amp 0.5 -> Throttle -> Null Sink, every block and value dictated) and scenario "
            "02_update_sample_rate (set samp_rate to 48000); t01 dial tone (two summed audio tones to an "
            "audio sink), t08 noise-source level probe, the GRC-Agent dial-tone edit scenarios (amplitude "
            "variable, constant source in place of the noise source) and t19 RSSI meter. None states an "
            "outcome to reach (a DC-offset tone with a stated swing, purity and tolerances, written to a "
            "named file) and leaves the method open. "
            "Under signature v2 the cell {tone} is held by this item only (old dev twins gbf-dev-01 and "
            "gbf-dev-03 and the test sibling gbf-test-11 were re-themed in round 1)."),
    }


def build() -> tuple[dict, dict[str, str], tuple[str, ...]]:
    ref = compact(reference_full())
    sig = C.signature(ref)
    split = C.signature_split(sig)
    assert split == SPLIT, f"v2 signature {sig} belongs to split {split}, not {SPLIT}"
    files = {"reference.grc": grcfile.dump(ref)}
    variant_sigs = {}
    print(f"reference: {'|'.join(sig)} -> {split}")
    for name, fn in VARIANTS.items():
        v = compact(fn())
        vs = C.signature(v)
        variant_sigs[name] = vs
        print(f"  {name}: {'|'.join(vs)} -> {C.signature_split(vs)}")
        files[f"variants/{name}"] = grcfile.dump(v)
    return record(sig, variant_sigs), files, sig


def run_probes() -> int:
    from grcbench.tier_f.ladder import check
    task = taskio.load(HERE)
    bad = 0
    for name, (g, expect, why) in probes().items():
        res = check(grcfile.dump(compact(g)), task)
        ok = res["first_failed"] is None and res["highest_passed"] == "V5"
        failed = [f"{c['feature']}={c['value']}" for c in (res["rungs"].get("V5") or {}).get("checks", [])
                  if not c["passed"]]
        verdict = "as expected" if ok == expect else "UNEXPECTED"
        bad += ok != expect
        print(f"{name:24s} passed={ok!s:5s} first_failed={res['first_failed']} {failed} -> {verdict}\n    {why}")
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    if "--probes" in argv:
        return run_probes()
    rec, files, sig = build()
    errors = taskio.save(HERE, rec, files)
    if errors:
        print("NOT WRITTEN:", *errors, sep="\n  ")
        return 1
    for stale in (HERE / "variants").glob("*.grc"):
        if f"variants/{stale.name}" not in files:
            stale.unlink()
    print(f"written {HERE.relative_to(REPO)} (cluster {C.cluster_id(sig)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
