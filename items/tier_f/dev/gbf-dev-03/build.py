"""Authoring script for GRC-Bench Tier-F item gbf-dev-03 (modify, L1, composed chain), version 3.

Re-themed in round 1 (fix plan 18 Oct): the old dev-03 (retune a unipolar tone) was a template twin of
gbf-test-04 in the {tone} cell (gbf-test-04 keeps it) and was really a parameter-tune task (round-1
item-quality F1). New theme, a genuine structural modify: add a matched filter to an NRZ stream.

Start graph (samp_rate 32000; sps = 8; symbols = 24 NRZ symbols of +-1):

    Vector Source (float, [s for s in symbols for _ in range(sps)], repeat) -> Throttle -> File Sink nrz.dat

The user asks for a matched filter for the rectangular pulses: for every input sample, the average of
that sample and the 7 before it (one symbol), one output per input sample with no extra delay, written as
float to a new File Sink mf.dat, nrz.dat unchanged. The reference adds a Moving Average (length 8,
scale 1/8). The ideal output is a sequence of ramps and plateaus that reach exactly +-1 at the end of
every full symbol; every value is a multiple of 1/8 (exact in float for direct-form filters).

Oracle (f2p on mf.dat, from sample 7 on, i.e. once the filter has filled):
    corr_ref  signed, max_lag 0, against the ideal output over two pattern periods (384 samples)
              >= 0.99999                                        (shape and alignment: no extra delay)
    rms       over the same 384 samples, within 0.1 % of the ideal  (scale)
    max_int_dev with scale 8 <= 0.008 over the whole capture     (every value within 0.001 of k/8)
    max / min  1 and -1 within 0.001 over the whole capture      (a full symbol gives +-1)
Over the corr window these imply the stated tolerance: with corr >= 1 - g and rms ratio 1 + d the RMS
error is at most rms(ideal) * sqrt(d^2 + 2g(1 + d)) ~ 3.2e-3, so no sample can sit on a wrong 1/8
lattice point (that alone would need an error of 0.124 on one sample), and the lattice check bounds the
remaining error by 0.008/8 = 0.001. An FFT filter (float rounding ~1e-7) therefore passes, while an
exact equal_prefix would reject it.
p2p: nrz.dat equals the original pattern from its first sample (two periods, exact).
Graph check (f2p): a path Vector Source -> filter block -> File Sink, so a hard-coded Vector Source
holding the expected output fails. Accepted filter blocks: every block that can realise the
rectangular matched filter exactly (Moving Average, FIR / FFT / interpolating FIR / IIR filters, the
Rational Resampler at 1:1, an Add of a delay line, Embedded Python).

Signature v2: start {data_source} (train, irrelevant for a modify start); reference and variants v1-v3
{data_source, filter} -> dev (cluster chain:data_source|filter). Variant v4, the delay-line adder, is
{combine, data_source, delay}, which hashes to train; it is shipped so the reserved list includes it.

    PYTHONPATH=src ~/grc-agent/.venv/bin/python items/tier_f/dev/gbf-dev-03/build.py           # write
    PYTHONPATH=src ~/grc-agent/.venv/bin/python items/tier_f/dev/gbf-dev-03/build.py --probes  # check

Deterministic; needs GNU Radio (catalog defaults are used to write the compact files).
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from grcbench.tier_f import compose as C          # noqa: E402
from grcbench.tier_f import grc, grcfile, taskio  # noqa: E402

TASK_ID = "gbf-dev-03"
SPLIT = "dev"
VERSION = 3

# ---- the specification ---------------------------------------------------------------------------
FS = 32000
SPS = 8                                     # samples per symbol (4 kBd)
SYMBOLS = (1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1)
P = SPS * len(SYMBOLS)                      # 192-sample period
NRZ_FILE, MF_FILE = "nrz.dat", "mf.dat"
SKIP = SPS - 1                              # the first 7 outputs (filter filling) are not checked
TOL = 0.001                                 # stated: every output within 0.001 of the average
CORR_WIN = 2 * P                            # 384 samples
CORR_MIN = 0.99999
RMS_REL = 0.001
LATTICE_MAX = round(TOL * SPS, 6)           # 0.008 on the 8x scale
N_ITEMS = 4096

INSTRUCTION = (
    "This flowgraph plays a repeating NRZ test pattern: a Vector Source sends +1/-1 symbols as "
    "rectangular pulses, 8 samples per symbol (4 kBd at 32 kHz), through a Throttle to a File Sink that "
    "writes nrz.dat. Add a matched filter for these rectangular pulses and write its output to a new File "
    "Sink with the file name mf.dat and item type float. For every input sample the filter output must be "
    "the average of that sample and the 7 samples before it (one symbol), so that it reaches exactly +1 "
    "or -1 at the end of every full symbol. Keep one output sample per input sample, aligned with the "
    "input (no decimation and no extra delay). Each output value must be within 0.001 of that average; "
    "the first 7 output samples, while the filter fills, do not matter. Keep nrz.dat exactly as it is. "
    "The flowgraph should keep producing samples until it is stopped; do not limit the number of samples."
)

# blocks that can realise the rectangular matched filter exactly (graph check)
FILTERS = ["blocks_moving_average_xx", "fir_filter_xxx", "fft_filter_xxx", "interp_fir_filter_xxx",
           "iir_filter_xxx", "rational_resampler_xxx", "blocks_add_xx", "epy_block"]

SYMBOLS_EXPR = "(" + ", ".join(str(s) for s in SYMBOLS) + ")"
VECTOR_EXPR = "[s for s in symbols for _ in range(sps)]"


def nrz(n: int) -> list[int]:
    one = [s for s in SYMBOLS for _ in range(SPS)]
    return (one * (n // P + 2))[:n]


def ideal(n: int) -> list[Fraction]:
    """Ideal matched-filter output: mean of the current and the 7 previous inputs (zero before start)."""
    x = nrz(n)
    return [Fraction(sum(x[j] for j in range(max(0, i - SPS + 1), i + 1)), SPS) for i in range(n)]


def _rms(v) -> float:
    return math.sqrt(sum(float(a) ** 2 for a in v) / len(v))


def _corr0(r, x) -> float:
    num = sum(float(a) * float(b) for a, b in zip(r, x))
    return num / math.sqrt(sum(float(a) ** 2 for a in r) * sum(float(b) ** 2 for b in x))


def margins() -> dict[str, tuple[float, float]]:
    """(corr, rms ratio) of typical wrong outputs over the corr window, to show the thresholds' margins."""
    n = SKIP + CORR_WIN + 32
    x = nrz(n)
    ref = ideal(n)[SKIP:SKIP + CORR_WIN]

    def box(length, scale, delay=0):
        out = []
        for i in range(n):
            j = i - delay
            out.append(scale * sum(x[k] for k in range(max(0, j - length + 1), j + 1)) if j >= 0 else 0.0)
        return out[SKIP:SKIP + CORR_WIN]
    cases = {"length 7": box(7, 1 / 8), "length 9": box(9, 1 / 8), "delay 1": box(8, 1 / 8, 1),
             "scale 0.12": box(8, 0.12), "no filter": x[SKIP:SKIP + CORR_WIN]}
    r0 = _rms(ref)
    return {k: (_corr0(ref, v), _rms(v) / r0) for k, v in cases.items()}


def compact(data: dict) -> dict:
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in data["blocks"]}
    out = grcfile.compact(data, defaults)
    out["options"] = data["options"]
    diffs = grc.compact_equivalent(data, out)
    assert not diffs, f"compact form differs from the full form: {diffs}"
    return out


VARS = {"samp_rate": str(FS), "sps": str(SPS), "symbols": SYMBOLS_EXPR}


def _base_blocks() -> list[dict]:
    return [
        C.block("src", "blocks_vector_source_x", type="float", vector=VECTOR_EXPR, repeat="True"),
        C.block("thr", "blocks_throttle2", type="float", samples_per_second="samp_rate"),
        C.file_sink("nrz_out", "float", NRZ_FILE),
    ]


BASE_CONNS = [["src", "0", "thr", "0"], ["thr", "0", "nrz_out", "0"]]


def start_full() -> dict:
    return C.graph("nrz_test_pattern", _base_blocks(), [list(c) for c in BASE_CONNS], dict(VARS))


def _with(gid: str, extra: list[dict], conns: list[list[str]]) -> dict:
    return C.graph(gid, _base_blocks() + extra, [list(c) for c in BASE_CONNS] + conns, dict(VARS))


# ---- reference -----------------------------------------------------------------------------------
def reference_full() -> dict:
    return _with("nrz_test_pattern", [
        C.block("mf", "blocks_moving_average_xx", type="float", length="8", scale="0.125"),
        C.file_sink("mf_out", "float", MF_FILE)],
        [["thr", "0", "mf", "0"], ["mf", "0", "mf_out", "0"]])


# ---- equivalent variants (genuinely different correct solutions, same v2 cell) ------------------
def variant_fir_taps() -> dict:
    """FIR filter with 8 taps of 1/8."""
    return _with("nrz_test_pattern", [
        C.block("matched", "fir_filter_xxx", type="fff", decim="1", taps="[0.125]*8"),
        C.file_sink("mf_sink", "float", MF_FILE)],
        [["thr", "0", "matched", "0"], ["matched", "0", "mf_sink", "0"]])


def variant_fft_from_source() -> dict:
    """FFT filter with taps from the sps variable, fed straight from the Vector Source (before the
    Throttle); its output carries float rounding (~1e-7), so it is not bit-identical to the ideal."""
    return _with("nrz_test_pattern", [
        C.block("mf_fft", "fft_filter_xxx", type="fff", decim="1", taps="[1.0/sps]*sps", nthreads="1"),
        C.file_sink("mf_file", "float", MF_FILE)],
        [["src", "0", "mf_fft", "0"], ["mf_fft", "0", "mf_file", "0"]])


def variant_sum_then_scale() -> dict:
    """Moving sum over one symbol (Moving Average with scale 1), then Multiply Const 1/sps."""
    return _with("nrz_test_pattern", [
        C.block("integrate", "blocks_moving_average_xx", type="float", length="sps", scale="1"),
        C.multiply_const("normalise", "float", "1/sps"),
        C.file_sink("mf_out", "float", MF_FILE)],
        [["thr", "0", "integrate", "0"], ["integrate", "0", "normalise", "0"], ["normalise", "0", "mf_out", "0"]])


def variant_delay_line_adder(gid: str = "nrz_test_pattern") -> dict:
    """Delay line (7 Delay blocks) + 8-input Add + Multiply Const 1/8. Its v2 cell
    combine|data_source|delay hashes to train, so it is shipped (and therefore reserved, R1/R2)."""
    delays = [C.block(f"d{k}", "blocks_delay", type="float", delay=str(k)) for k in range(1, SPS)]
    return _with(gid, delays + [
        C.block("acc", "blocks_add_xx", type="float", num_inputs=str(SPS)),
        C.multiply_const("norm", "float", "0.125"),
        C.file_sink("mf_out", "float", MF_FILE)],
        [["thr", "0", "acc", "0"]] + [["thr", "0", f"d{k}", "0"] for k in range(1, SPS)]
        + [[f"d{k}", "0", "acc", str(k)] for k in range(1, SPS)]
        + [["acc", "0", "norm", "0"], ["norm", "0", "mf_out", "0"]])


VARIANTS = {
    "v1_fir_eighth_taps.grc": variant_fir_taps,
    "v2_fft_filter_from_source.grc": variant_fft_from_source,
    "v3_moving_sum_then_scale.grc": variant_sum_then_scale,
}
# correct designs in another v2 cell that hashes to train: shipped so the reserved list includes them
OTHER_CELL_VARIANTS = {
    "v4_delay_line_adder.grc": variant_delay_line_adder,
}


# ---- probes: answers a single-edit mutant cannot express (expected verdicts) --------------------
def probes() -> dict[str, tuple[dict, bool, str]]:
    steady = ideal(2 * P)[P:]                       # one steady-state period of the ideal output
    hard = _with("p_hard", [
        C.block("fake", "blocks_vector_source_x", type="float", repeat="True",
                vector="(" + ", ".join(str(float(v)) for v in steady) + ")"),
        C.file_sink("mf_out", "float", MF_FILE)], [["fake", "0", "mf_out", "0"]])
    i_and_d = _with("p_iad", [
        C.block("iad", "blocks_integrate_xx", type="float", decim="8"),
        C.multiply_const("norm", "float", "0.125"),
        C.file_sink("mf_out", "float", MF_FILE)],
        [["thr", "0", "iad", "0"], ["iad", "0", "norm", "0"], ["norm", "0", "mf_out", "0"]])
    late = _with("p_late", [
        C.block("mf", "fir_filter_xxx", type="fff", decim="1", taps="[0] + [0.125]*8"),
        C.file_sink("mf_out", "float", MF_FILE)], [["thr", "0", "mf", "0"], ["mf", "0", "mf_out", "0"]])
    iir = _with("p_iir", [
        C.block("mf", "iir_filter_xxx", type="ffd", fftaps="[0.125]*8", fbtaps="[1]", oldstyle="True"),
        C.file_sink("mf_out", "float", MF_FILE)], [["thr", "0", "mf", "0"], ["mf", "0", "mf_out", "0"]])
    interp = _with("p_interp", [
        C.block("mf", "interp_fir_filter_xxx", type="fff", interp="1", taps="[0.125]*8"),
        C.file_sink("mf_out", "float", MF_FILE)], [["thr", "0", "mf", "0"], ["mf", "0", "mf_out", "0"]])
    line = variant_delay_line_adder("p_line")
    in_tol = _with("p_in_tol", [
        C.block("mf", "blocks_moving_average_xx", type="float", length="8", scale="0.1249"),
        C.file_sink("mf_out", "float", MF_FILE)], [["thr", "0", "mf", "0"], ["mf", "0", "mf_out", "0"]])
    return {
        "scale_0p1249_in_tolerance": (in_tol, True, "scale 0.1249: every value within 0.0008 of the average "
                                      "(full symbols +-0.9992), inside the stated 0.001, so it must pass"),
        "hardcoded_output": (hard, False, "mf.dat written from a Vector Source holding the expected output "
                             "(matches from sample 7 on): no filter on a path from the pattern (graph check)"),
        "integrate_and_dump": (i_and_d, False, "Integrate with decimation 8 (one output per symbol): wrong "
                               "rate, shape check fails"),
        "fir_one_sample_late": (late, False, "FIR with a leading zero tap: right values, one sample late "
                                "(alignment)"),
        "iir_as_fir": (iir, True, "IIR filter with feedback taps [1] used as the 8-tap average"),
        "interp_fir_unity": (interp, True, "Interpolating FIR filter with interpolation 1"),
        "delay_line_adder": (line, True, "delay line (7 Delay blocks) + 8-input Add + Multiply Const 1/8 "
                             "(v2 cell combine|data_source|delay, train hash: shipped as variant v4)"),
    }


# ---- task record ---------------------------------------------------------------------------------
def record(sig: tuple[str, ...], start_sig: tuple[str, ...]) -> dict:
    y = ideal(SKIP + CORR_WIN)[SKIP:]
    corr_ref = [float(v) for v in y]
    rms_ref = round(_rms(y), 6)
    nrz_ref = nrz(2 * P)
    shape_req = ("matched filter: from sample 7 on, mf.dat is the average of the current and the 7 previous "
                 "input samples, aligned with the input (no extra delay, one output per input), every value "
                 "within 0.001; checked exactly over samples 7-390 by corr_ref (signed, no lag) >= 0.99999 "
                 "with rms within 0.1 % plus the 1/8 lattice check (together: every value within 0.001)")
    lattice_req = ("output values: every sample from 7 on lies within 0.001 of a multiple of 1/8 (an average "
                   "of 8 values +-1) and between -1 and +1, reaching +-1 within 0.001 at full symbols")
    keep_req = "nrz.dat unchanged: the original pattern, 8 samples per symbol, from its first sample (exact)"
    m = margins()
    margin_txt = "; ".join(f"{k}: corr {c:.4f}, rms ratio {r:.3f}" for k, (c, r) in m.items())
    notes = (
        "Re-themed in round 1 (fix plan 18 Oct, group G1): old dev-03 (retune a unipolar tone) was a "
        "template twin of gbf-test-04 ({tone}, kept by test-04) and in substance a parameter-tune item "
        "(round-1 F1); the new task is a genuine structural modify (a block is added). Oracle: exact "
        "equality was not used because an FFT filter (accepted by the plan's graph check) is correct but "
        "not bit-identical; corr_ref + rms + 1/8-lattice imply the stated 0.001 tolerance over samples "
        "7-390, and the lattice/min/max checks cover the whole capture. Margins of typical wrong outputs "
        f"over the corr window: {margin_txt} (thresholds: corr >= {CORR_MIN}, rms within {RMS_REL:.1%}). "
        "Known limits: (1) the path check accepts any of the listed filter blocks (and Add, for delay-line "
        "designs) between a Vector Source and a File Sink, so a deliberate cheat that routes a hard-coded "
        "output through a one-tap filter or an Add would pass; "
        "(2) 'do not limit the number of samples' is enforced only as V5 liveness (4096 items at both "
        "points); (3) samples 391-4095 are checked for the 1/8 lattice and the +-1 range, not exactly. "
        "Concept relations: gbf-test-13 shapes NRZ data for transmission (low-pass interpolation x8), the "
        "receive-side counterpart; gbf-dev-01 (same fixer group) correlates a 0/1 bit stream with a 16-bit "
        "sync word and thresholds it (build), this item averages +-1 NRZ samples over one symbol (modify). "
        "Training spec t19 (RSSI meter) also uses a Moving Average, for level metering. Variant "
        "signatures for the reserved-signature step (R1/R2): v1-v3 share the reference cell "
        "data_source|filter (dev); v4 (delay line + Add + Multiply Const) is combine|data_source|delay, "
        "which hashes to train, and is shipped so the reserved list includes it. The Embedded Python "
        "alternative (custom|data_source) is dev."
    )
    return {
        "id": TASK_ID, "tier": "F", "split": SPLIT, "version": VERSION,
        "license": "CC-BY-4.0 (original GRC-Bench item; licence to be confirmed by the team)",
        "source": {"kind": "composed_chain", "ref": "|".join(sig)},
        "cluster_id": C.cluster_id(sig),
        "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring (round-1 re-theme)",
                       "generator_version": "0.2.0"},
        "review": {"status": "pending-team-review", "reviewers": [], "notes": notes},
        "family": "modify", "level": "L1",
        "instruction": INSTRUCTION,
        "start_grc": "start.grc",
        "reference_grc": "reference.grc",
        "observation": {
            "points": [
                {"name": "mf", "sink": {"block_id": "blocks_file_sink",
                                        "params": {"file": MF_FILE, "type": "float"}}},
                {"name": "nrz", "sink": {"block_id": "blocks_file_sink",
                                         "params": {"file": NRZ_FILE, "type": "float"}}},
            ],
            "run": {"n_items": N_ITEMS, "min_items": N_ITEMS, "timeout_s": 10.0},
            "contract": "the float samples written to the File Sinks whose file names are mf.dat and nrz.dat "
                        "(item type float)",
        },
        "oracle": {
            "checks": [
                {"point": "mf", "feature": "corr_ref", "ref": corr_ref, "signed": True, "max_lag": 0,
                 "skip": SKIP, "min": CORR_MIN, "group": "f2p", "property": "matched filter",
                 "requires": [shape_req]},
                {"point": "mf", "feature": "rms", "skip": SKIP, "n": CORR_WIN, "expect": rms_ref,
                 "rel_tol": RMS_REL, "group": "f2p", "property": "matched filter", "requires": [shape_req]},
                {"point": "mf", "feature": "max_int_dev", "scale": SPS, "skip": SKIP, "max": LATTICE_MAX,
                 "group": "f2p", "property": "output values", "requires": [lattice_req]},
                {"point": "mf", "feature": "max", "skip": SKIP, "expect": 1.0, "tol": TOL,
                 "group": "f2p", "property": "output values", "requires": [lattice_req]},
                {"point": "mf", "feature": "min", "skip": SKIP, "expect": -1.0, "tol": TOL,
                 "group": "f2p", "property": "output values", "requires": [lattice_req]},
                {"point": "nrz", "feature": "equal_prefix", "ref": nrz_ref, "max_offset": 0,
                 "group": "p2p", "property": "nrz.dat unchanged", "requires": [keep_req]},
            ],
            "graph_checks": [
                {"kind": "path", "from_ids": ["blocks_vector_source_x"], "to_ids": ["blocks_file_sink"],
                 "via_ids": FILTERS, "group": "f2p", "property": "matched filter added",
                 "requires": ["a filter block on a path from the Vector Source to a File Sink ("
                              + ", ".join(FILTERS) + ")"]},
            ],
        },
        "spec_properties": [
            shape_req,
            lattice_req,
            keep_req,
            "matched filter added: graph check path Vector Source -> filter block -> File Sink (a "
            "hard-coded output pattern fails)",
            "observation points: File Sinks mf.dat and nrz.dat with item type float (sinks of another type "
            "do not match); the start graph has no mf.dat, so it fails V6",
            "run condition: the flowgraph keeps producing samples; checked only as V5 liveness (4096 items "
            "at both observation points)",
        ],
        "mutations": [
            {"op": "set_param", "block": "mf", "param": "length", "value": "7",
             "breaks": "matched filter (7 samples, shorter than one symbol)"},
            {"op": "set_param", "block": "mf", "param": "length", "value": "9",
             "breaks": "matched filter (near miss: 9 samples)"},
            {"op": "set_param", "block": "mf", "param": "length", "value": "16",
             "breaks": "matched filter (two symbols)"},
            {"op": "set_param", "block": "mf", "param": "scale", "value": "1",
             "breaks": "output values (sum instead of average: +-8)"},
            {"op": "set_param", "block": "mf", "param": "scale", "value": "0.12",
             "breaks": "output values (scale rounded to 0.12: full symbols give +-0.96)"},
            {"op": "set_param", "block": "mf", "param": "scale", "value": "0.1248",
             "breaks": "output values (near miss on the 0.001 tolerance: full symbols give +-0.9984)"},
            {"op": "bypass", "block": "mf", "breaks": "matched filter (no filter: mf.dat is the raw NRZ)"},
            {"op": "rewire_input", "dst": "nrz_out", "dst_port": "0", "new_src": "mf",
             "breaks": "nrz.dat unchanged (nrz.dat records the filter output)"},
            {"op": "set_param", "block": "symbols", "param": "value",
             "value": "(" + ", ".join(str(-s if k == 5 else s) for k, s in enumerate(SYMBOLS)) + ")",
             "breaks": "nrz.dat unchanged (test pattern edited: symbol 5 inverted)"},
            {"op": "set_param", "block": "sps", "param": "value", "value": "4",
             "breaks": "nrz.dat unchanged (4 samples per symbol)"},
        ],
        "perturbation": {"perturbed": False, "changes": []},
        "disjointness": (
            "Receive-side rectangular matched filter (sliding one-symbol average) added to an NRZ test pattern. "
            "Nearest training-side specs (reports/reference/training_task_specs.json): t19 RSSI meter "
            "(Moving Average for level metering, then a probe), t02 QPSK BER chain and t21 GMSK pipe "
            "(modulator/demodulator chains with built-in filters), t11 vector-sink extension (adds a "
            "second stream); none adds a matched filter to a baseband NRZ stream with an exact per-sample "
            "spec. Within GRC-Bench: distinct from gbf-dev-01 (sync-word correlator + threshold, build) and "
            "gbf-test-13 (transmit-side low-pass interpolation of NRZ data)."),
    }


def build() -> tuple[dict, dict[str, str], tuple[str, ...]]:
    start = compact(start_full())
    ref = compact(reference_full())
    sig = C.signature(ref)
    start_sig = C.signature(start)
    split = C.signature_split(sig)
    assert split == SPLIT, f"v2 signature {sig} belongs to split {split}, not {SPLIT}"
    print(f"start: {'|'.join(start_sig)} -> {C.signature_split(start_sig)}")
    print(f"reference: {'|'.join(sig)} -> {split}")
    files = {"start.grc": grcfile.dump(start), "reference.grc": grcfile.dump(ref)}
    for name, fn in VARIANTS.items():
        v = compact(fn())
        vs = C.signature(v)
        print(f"  {name}: {'|'.join(vs)} -> {C.signature_split(vs)}")
        assert vs == sig, f"variant {name} leaves the cell: {vs}"
        files[f"variants/{name}"] = grcfile.dump(v)
    for name, fn in OTHER_CELL_VARIANTS.items():
        v = compact(fn())
        vs = C.signature(v)
        print(f"  {name}: {'|'.join(vs)} -> {C.signature_split(vs)} (other cell, reserved)")
        assert vs != sig and C.signature_split(vs) == "train", f"variant {name}: {vs} need not be listed apart"
        files[f"variants/{name}"] = grcfile.dump(v)
    for k, (c, r) in margins().items():
        assert c < CORR_MIN or abs(r - 1) > RMS_REL, f"wrong output '{k}' would pass: corr {c}, rms ratio {r}"
        print(f"  margin {k}: corr {c:.5f}, rms ratio {r:.4f}")
    return record(sig, start_sig), files, sig


def run_probes() -> int:
    from grcbench.tier_f.ladder import check
    task = taskio.load(HERE)
    bad = 0
    for name, (g, expect, why) in probes().items():
        cg = compact(g)
        s = C.signature(cg)
        res = check(grcfile.dump(cg), task)
        ok = res["first_failed"] is None and res["highest_passed"] == "V6"
        v6 = res["rungs"].get("V6") or {}
        failed = [f"{c['feature']}={c['value']}" for c in v6.get("checks", []) if not c["passed"]] + \
                 [f"graph:{g_['kind']}" for g_ in v6.get("graph_checks", []) if not g_["passed"]]
        verdict = "as expected" if ok == expect else "UNEXPECTED"
        bad += ok != expect
        print(f"{name:22s} passed={ok!s:5s} first_failed={res['first_failed']} {failed} -> {verdict}\n"
              f"    {why} [v2 {'|'.join(s)} -> {C.signature_split(s)}]")
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
