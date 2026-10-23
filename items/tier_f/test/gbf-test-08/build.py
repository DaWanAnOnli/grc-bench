"""Authoring script for GRC-Bench Tier-F item gbf-test-08 (family repair, level L1, composed chain).

Version 2 (re-themed in the 18 Oct fix round, reports/tier_f_reviews/fix_plan.json, group G4).
Version 1 repaired a square-to-sine wave shaper; it was one of five near-duplicate
square/sawtooth-to-sine items and its signature hashed to train under signature v2.

Theme: a phase comparator between two oscillators. At 8 kHz a reference oscillator (complex Signal
Source, 1 kHz, amplitude 1) and a test oscillator (complex Signal Source, 1.005 kHz, amplitude 1)
feed a Multiply Conjugate block (output = in0 * conj(in1) = test * conj(reference)); Complex To Arg
turns its output into the test oscillator's phase relative to the reference, in radians wrapped to
(-pi, pi], written as float samples to the File Sink phase.dat. Because the test oscillator runs
5 Hz fast, the relative phase starts at 0 and rises by 2 pi every 0.2 s: a rising sawtooth from -pi
to pi, five periods per second.

Fault (deleted_connection, expected rung V3): the connection reference oscillator -> Multiply
Conjugate input 1 is removed. GRC validation then reports two unconnected ports (the reference
oscillator's output and the comparator's input 1). The repair must restore the reference as the
CONJUGATED input: connecting it to input 0 instead (and the test oscillator to input 1) negates the
phase, a falling sawtooth, which the signed correlation check rejects.

Why 8 kHz (the fix plan sketched 48 kHz): the signed corr_ref check (tooling T2) compares phase.dat
with one full period of the ideal ramp; at 8 kHz that period is 1,600 samples instead of 9,600, so
the reference stored in task.json stays small. Nothing else depends on the rate.

Oracle groups (rule R8 of the fix plan). The start graph fails at V3 and never runs, so no signal
property "held" there: every signal check depends on the repaired connection and is f2p. The p2p
group holds the properties carried by the parts of the graph the fault does not touch, checked
structurally: both oscillators are still present with their stated frequency, amplitude and type,
and a path Signal Source -> Complex To Arg (or Complex to Mag/Phase) -> File Sink exists. build.py
evaluates the p2p graph checks on the start graph and asserts that they pass there.

Measurement: 8,000 items (1 s, five ramp periods) from the first sample. The relative phase starts
at exactly 0 (both oscillators start at phase 0), so the signed correlation with the ideal ramp
(1,600 samples, lag 0..16) is 1.0 for a correct repair and -0.94 for the inverted one (best lag 16).

Deterministic; regenerate with (inside WSL, from the repository root):

    PYTHONPATH=src QT_QPA_PLATFORM=offscreen ~/grc-agent/.venv/bin/python items/tier_f/test/gbf-test-08/build.py
    ./gb tier-f gates items/tier_f/test/gbf-test-08
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from grcbench.common.canary import CANARY        # noqa: E402
from grcbench.tier_f import compose as C          # noqa: E402
from grcbench.tier_f import edit, graph_checks, grc, grcfile, taskio  # noqa: E402

TASK_ID = "gbf-test-08"
SPLIT = "test"
VERSION = 2

# ---- the specification (every number the oracle depends on is stated in the instruction) --------
FS = 8000                 # sample rate, Hz
F_REF = 1000              # reference oscillator, Hz
F_TEST = 1005             # test oscillator, Hz (5 Hz fast)
AMP = 1                   # both oscillators' amplitude
F_PHASE = F_TEST - F_REF  # the relative phase rises by 2 pi five times per second
OUT_FILE = "phase.dat"

PERIOD = FS // F_PHASE                    # 1600 samples = 0.2 s
N_ITEMS = 8000                            # 1 s = five periods
MAX_LAG = 16                              # 2 ms
CORR_MIN = 0.95

# names in GRC's default style, as a user's own flowgraph would have them
REF, TEST, CMP, ARG, OUT = ("analog_sig_source_x_0", "analog_sig_source_x_1",
                            "blocks_multiply_conjugate_cc_0", "blocks_complex_to_arg_0", "blocks_file_sink_0")
FAULT = {"src": REF, "dst": CMP}
PHASE_BLOCKS = ["blocks_complex_to_arg", "blocks_complex_to_magphase"]
# compact form keeps these parameters even at their catalog defaults, so the graph checks below
# (which read parameters from the file) see them in reference.grc as they see them in GRC's exports
KEEP = {"analog_sig_source_x": ("freq", "amp")}

INSTRUCTION = (
    "This flowgraph is meant to compare the phase of two oscillators at a sample rate of 8 kHz: a "
    "reference oscillator at 1 kHz and a test oscillator at 1.005 kHz, both complex Signal Sources "
    "with amplitude 1. It should write the phase of the test oscillator relative to the reference "
    "oscillator, in radians and wrapped to the range -π to π, as float samples to the File Sink whose "
    "file name is phase.dat. Because the test oscillator runs 5 Hz fast, that relative phase starts "
    "at 0 and rises steadily by 2π every 0.2 s, so phase.dat should hold a rising sawtooth that climbs "
    "from -π to π five times per second. At the moment GRC refuses to generate the flowgraph: "
    "validation fails with \"Port is not connected\". Please repair it so that it does this, and keep "
    "both oscillators as they are. The flowgraph should keep producing samples until it is stopped; "
    "do not limit the number of samples."
)


def ideal_ramp(n: int) -> list[float]:
    """Phase of exp(j 2 pi 5 t) relative to t = 0, wrapped to (-pi, pi], at 8 kHz."""
    k = np.arange(n)
    return [round(float(v), 6) for v in np.angle(np.exp(2j * np.pi * F_PHASE * k / FS))]


# ---- task properties ---------------------------------------------------------------------------
P_VALID = "P0 the flowgraph validates and runs (every port connected)"
P_SIGN = ("P1 relative phase of the TEST oscillator against the REFERENCE: phase.dat follows the "
          "rising ramp that starts at 0 (signed correlation with one ideal 0.2 s period >= 0.95 at a "
          "lag of at most 2 ms); the inverted phase (reference relative to test) scores -0.94")
P_RATE = "P2 rate: the ramp repeats five times per second (strongest component 5 Hz, 1 Hz resolution)"
P_RANGE = ("P3 radians wrapped to -pi..pi: largest |value| is pi (within 1 %; every correct repair gives "
           "pi exactly, so the allowance only absorbs float rounding); degrees, a 0..2 pi convention or an "
           "unwrapped phase fail")
P_KEEP = ("P4 both oscillators kept as they are: complex Signal Sources at 1 kHz and 1.005 kHz with "
          "amplitude 1 are present (graph checks on the parameters)")
P_PATH = ("P5 the phase is computed from the oscillators: a path Signal Source -> Complex To Arg (or "
          "Complex to Mag/Phase) -> File Sink exists (rules out a sawtooth Signal Source written "
          "straight to phase.dat)")
P_OBS = ("P6 observation and run: float samples (8 kHz) written to the File Sink phase.dat (the sink's "
         "type is part of the observation point); the flowgraph runs until stopped, checked only by V4 "
         "liveness (at least 8,000 items)")
P_GROUPS = ("Groups (fix plan R8): the start graph fails at V3 and never runs, so every signal check "
            "depends on the repaired connection and is f2p; p2p holds the structural properties of the "
            "parts the fault does not touch (P4, P5), which pass on the start graph (asserted in "
            "build.py)")


def oracle() -> dict:
    pt = "out"
    return {
        "checks": [
            {"point": pt, "feature": "corr_ref", "ref": ideal_ramp(PERIOD), "signed": True,
             "max_lag": MAX_LAG, "min": CORR_MIN, "group": "f2p", "property": P_SIGN},
            {"point": pt, "feature": "peak_freq", "fs": FS, "expect": F_PHASE, "tol": 0.5, "n": N_ITEMS,
             "group": "f2p", "property": P_RATE},
            {"point": pt, "feature": "max_abs", "expect": round(math.pi, 6), "rel_tol": 0.01, "n": N_ITEMS,
             "group": "f2p", "property": P_RANGE},
        ],
        "graph_checks": [
            {"kind": "block_present", "ids": ["analog_sig_source_x"],
             "params": {"type": "complex", "freq": str(F_REF), "amp": str(AMP)},
             "group": "p2p", "property": P_KEEP},
            {"kind": "block_present", "ids": ["analog_sig_source_x"],
             "params": {"type": "complex", "freq": str(F_TEST), "amp": str(AMP)},
             "group": "p2p", "property": P_KEEP},
            {"kind": "path", "from_ids": ["analog_sig_source_x"], "via_ids": PHASE_BLOCKS,
             "to_ids": ["blocks_file_sink"], "group": "p2p", "property": P_PATH},
        ],
    }


OBSERVATION = {
    "points": [{"name": "out", "sink": {"block_id": "blocks_file_sink",
                                        "params": {"file": OUT_FILE, "type": "float"}}}],
    "run": {"n_items": N_ITEMS, "min_items": N_ITEMS, "timeout_s": 10.0},
    "contract": "the float samples (8 kHz) written to the File Sink whose file name is phase.dat",
}

MUTATIONS = [
    {"op": "set_param", "block": REF, "param": "freq", "value": "1010",
     "breaks": "P1/P4 (reference oscillator retuned to 1.01 kHz: the relative phase falls at 5 Hz)"},
    {"op": "set_param", "block": TEST, "param": "freq", "value": "1010",
     "breaks": "P1/P2/P4 (test oscillator retuned to 1.01 kHz: a 10 Hz ramp)"},
    {"op": "set_param", "block": TEST, "param": "freq", "value": "1001",
     "breaks": "P1/P2/P4 near miss (test oscillator at 1.001 kHz: a 1 Hz ramp)"},
    {"op": "set_param", "block": TEST, "param": "amp", "value": "0.5",
     "breaks": "P4 keep the oscillators (amplitude changed; Complex To Arg ignores it, so only the "
               "graph check sees it)"},
    {"op": "set_param", "block": "samp_rate", "param": "value", "value": "16000",
     "breaks": "P1/P2 (oscillators generated for 16 kHz: read at 8 kHz the ramp repeats 2.5 times per "
               "second)"},
    {"op": "set_id", "block": ARG, "new_id": "blocks_complex_to_real",
     "breaks": "P1/P3/P5 (real part instead of the phase: a 5 Hz cosine of amplitude 1)"},
    {"op": "set_id", "block": ARG, "new_id": "blocks_complex_to_imag",
     "breaks": "P1/P3/P5 (imaginary part instead of the phase: a 5 Hz sine of amplitude 1)"},
    {"op": "set_id", "block": ARG, "new_id": "blocks_complex_to_mag",
     "breaks": "P1/P2/P3/P5 (magnitude instead of the phase: constant 1)"},
    {"op": "set_id", "block": CMP, "new_id": "blocks_multiply_xx",
     "breaks": "P1/P2 (plain Multiply instead of Multiply Conjugate: phase of the 2.005 kHz sum "
               "frequency)"},
]


# ---- graphs ------------------------------------------------------------------------------------
def osc(name: str, freq) -> dict:
    return C.sig_source(name, "complex", freq, AMP)


def reference_full() -> dict:
    blocks = [osc(REF, F_REF), osc(TEST, F_TEST),
              C.block(CMP, "blocks_multiply_conjugate_cc"),
              C.block(ARG, "blocks_complex_to_arg"),
              C.file_sink(OUT, "float", OUT_FILE)]
    conns = [[TEST, "0", CMP, "0"], [REF, "0", CMP, "1"], [CMP, "0", ARG, "0"], [ARG, "0", OUT, "0"]]
    return C.graph("phase_comparator", blocks, conns, {"samp_rate": str(FS)})


def variant_swapped_negated() -> dict:
    """Reference on input 0 and test on input 1 (the opposite phase), then Multiply Const -1."""
    blocks = [osc(REF, F_REF), osc(TEST, F_TEST),
              C.block(CMP, "blocks_multiply_conjugate_cc"),
              C.block(ARG, "blocks_complex_to_arg"),
              C.multiply_const("blocks_multiply_const_vxx_0", "float", "-1"),
              C.file_sink(OUT, "float", OUT_FILE)]
    conns = [[REF, "0", CMP, "0"], [TEST, "0", CMP, "1"], [CMP, "0", ARG, "0"],
             [ARG, "0", "blocks_multiply_const_vxx_0", "0"], ["blocks_multiply_const_vxx_0", "0", OUT, "0"]]
    return C.graph("phase_comparator", blocks, conns, {"samp_rate": str(FS)})


def variant_conjugate_multiply() -> dict:
    """Conjugate the reference explicitly and use a plain complex Multiply."""
    blocks = [osc(REF, F_REF), osc(TEST, F_TEST),
              C.block("blocks_conjugate_cc_0", "blocks_conjugate_cc"),
              C.block("blocks_multiply_xx_0", "blocks_multiply_xx", type="complex", num_inputs="2"),
              C.block(ARG, "blocks_complex_to_arg"),
              C.file_sink(OUT, "float", OUT_FILE)]
    conns = [[REF, "0", "blocks_conjugate_cc_0", "0"], [TEST, "0", "blocks_multiply_xx_0", "0"],
             ["blocks_conjugate_cc_0", "0", "blocks_multiply_xx_0", "1"],
             ["blocks_multiply_xx_0", "0", ARG, "0"], [ARG, "0", OUT, "0"]]
    return C.graph("phase_comparator", blocks, conns, {"samp_rate": "8e3"})


def variant_magphase() -> dict:
    """The restored connection, with Complex to Mag/Phase (phase output) instead of Complex To Arg;
    the magnitude output goes to a Null Sink."""
    blocks = [osc(REF, F_REF), osc(TEST, F_TEST),
              C.block(CMP, "blocks_multiply_conjugate_cc"),
              C.block("blocks_complex_to_magphase_0", "blocks_complex_to_magphase"),
              C.block("blocks_null_sink_0", "blocks_null_sink", type="float"),
              C.file_sink(OUT, "float", OUT_FILE)]
    conns = [[TEST, "0", CMP, "0"], [REF, "0", CMP, "1"], [CMP, "0", "blocks_complex_to_magphase_0", "0"],
             ["blocks_complex_to_magphase_0", "0", "blocks_null_sink_0", "0"],
             ["blocks_complex_to_magphase_0", "1", OUT, "0"]]
    return C.graph("phase_comparator", blocks, conns, {"samp_rate": str(FS)})


VARIANTS = {
    "variants/v1_swapped_inputs_negated.grc": variant_swapped_negated,
    "variants/v2_conjugate_then_multiply.grc": variant_conjugate_multiply,
    "variants/v3_mag_phase_block.grc": variant_magphase,
}


def compact(data: dict) -> dict:
    defaults = {}
    for b in data["blocks"]:
        d = grc.param_defaults(b["id"])
        defaults[b["id"]] = {k: v for k, v in d.items() if k not in KEEP.get(b["id"], ())}
    out = grcfile.compact(copy.deepcopy(data), defaults)
    out["options"] = data["options"]
    diffs = grc.compact_equivalent(data, out)
    assert not diffs, f"compact form differs from the full form: {diffs}"
    return out


REVIEW_NOTES = (
    "Re-themed 18 Oct (fix plan G4): the old square-to-sine repair became a phase-comparator repair "
    "with the same fault type (deleted connection). Known oracle limits: (1) a 'repair' that keeps both "
    "oscillators (parked on Null Sinks) but builds the ramp from a NEW 5 Hz complex Signal Source -> "
    "Complex To Arg would pass (the path check cannot tell which Signal Source feeds the phase block); "
    "it ignores the instruction and is far more work than the one-connection fix. (2) Oscillator "
    "waveform, phase and sample-rate parameters are not compared; frequency, amplitude and type are. "
    "(3) The rate is measured at 1 Hz resolution (1 s); the one-period signed correlation also needs "
    "the period right to about 1 %. Sample rate 8 kHz instead of the planned 48 kHz keeps the corr_ref "
    "reference at 1,600 values. Fault-type overlap with training-side t04_broken_sink_repair is "
    "recorded under disjointness."
)


def record(sig: tuple[str, ...], variant_sigs: dict, p2p_on_start: list[dict]) -> dict:
    sig_str = "|".join(sig)
    return {
        "id": TASK_ID, "tier": "F", "split": SPLIT, "version": VERSION, "canary": CANARY,
        "license": "CC-BY-4.0 (original GRC-Bench item; licence to be confirmed by the team)",
        "source": {"kind": "composed_chain", "ref": sig_str, "signature_version": C.SIGNATURE_VERSION},
        "cluster_id": C.cluster_id(sig),
        "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring",
                       "generator_version": "0.2.0"},
        "review": {"status": "pending-team-review", "reviewers": [], "notes": REVIEW_NOTES},
        "family": "repair", "level": "L1",
        "instruction": INSTRUCTION,
        "start_grc": "start.grc",
        "reference_grc": "reference.grc",
        "observation": OBSERVATION,
        "oracle": oracle(),
        "spec_properties": [P_VALID, P_SIGN, P_RATE, P_RANGE, P_KEEP, P_PATH, P_OBS, P_GROUPS],
        "mutations": MUTATIONS,
        "perturbation": {"perturbed": False, "changes": []},
        "fault": {"type": "deleted_connection",
                  "description": (f"the connection {REF}:0 -> {CMP}:1 (reference oscillator to the "
                                  "Multiply Conjugate's conjugated input) was deleted; the oscillator's "
                                  "output and the comparator's input 1 are left unconnected, so GRC "
                                  "validation fails"),
                  "expected_rung": "V3"},
        "disjointness": (
            "Two-oscillator phase comparator (Multiply Conjugate -> Complex To Arg) with the reference "
            "input deleted. Nearest training-side spec (reports/reference/training_task_specs.json): "
            "t04_broken_sink_repair shares the fault type and its symptom ('something unconnected when "
            "I generate it'), but its graph is a tone -> Throttle -> Audio Sink chain with an unconnected "
            "sink, and its fix does not depend on port semantics; here the missing link must go to the "
            "conjugated input or the phase is inverted. GRC-Agent scenario 21 (type conversion and "
            "conjugate) inserts a Conjugate block on request; no training spec measures relative phase."),
        "authoring_notes": {
            "revision": "2: re-themed from the square-to-sine shaper repair (filter|harmonic_wave, train "
                        "under v2) to a phase comparator repair (mix|phase|tone, test)",
            "level": "L1: 5 functional blocks (two Signal Sources, Multiply Conjugate, Complex To Arg, "
                     "File Sink), one sample rate",
            "reference_signature_v2": f"{sig_str} ({C.signature_split(sig)})",
            "variant_signatures_v2": variant_sigs,
            "p2p_graph_checks_on_start": {f"{g['kind']}#{i}": g["passed"] for i, g in enumerate(p2p_on_start)},
            "related_items": "gbf-test-01 (rectifier build) and gbf-test-12 (Channel Model tune) are the "
                             "other re-themed members of the former square-to-sine group; gbf-test-02 "
                             "(PLL build) also uses Multiply Conjugate, as a frequency-error detector "
                             "inside a different function",
        },
    }


def main() -> int:
    ref = compact(reference_full())
    sig = C.signature(ref)
    split = C.signature_split(sig)
    assert split == SPLIT, f"v2 signature {sig} belongs to split {split}, not {SPLIT}"
    start = edit.delete_connection(ref, FAULT["src"], FAULT["dst"])
    assert len(start["connections"]) == len(ref["connections"]) - 1
    p2p = [g for g in oracle()["graph_checks"] if g["group"] == "p2p"]
    p2p_on_start = graph_checks.evaluate(p2p, start, start)
    assert all(g["passed"] for g in p2p_on_start), p2p_on_start
    files = {"reference.grc": grcfile.dump(ref), "start.grc": grcfile.dump(start)}
    variant_sigs = {}
    for path, fn in VARIANTS.items():
        v = compact(fn())
        files[path] = grcfile.dump(v)
        vs = C.signature(v)
        variant_sigs[path] = f"{'|'.join(vs)} ({C.signature_split(vs)})"
    for old in (HERE / "variants").glob("*.grc") if (HERE / "variants").exists() else []:
        if f"variants/{old.name}" not in files:
            old.unlink()
    errors = taskio.save(HERE, record(sig, variant_sigs, p2p_on_start), files)
    if errors:
        print("NOT WRITTEN:", *errors, sep="\n  ")
        return 1
    print(f"written {HERE.name}: reference {'|'.join(sig)} -> {split}; start {'|'.join(C.signature(start))}; "
          f"variants {variant_sigs}; {len(MUTATIONS)} mutations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
