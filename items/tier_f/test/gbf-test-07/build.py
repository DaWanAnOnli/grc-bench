"""Authoring script for GRC-Bench Tier-F item gbf-test-07 (modify, L1, composed chain).

Re-themed on 18 Oct (fix plan: the old square-wave-to-sine item hashed to train under signature v2
and nearly duplicated gbf-test-01). New theme: group-delay compensation.

Start graph (48 kHz, float): a Signal Source (cosine, 1.5 kHz, amplitude 0.8) feeds
  (a) an FIR Filter (fir_filter_xxx, fff) whose taps are the variable lpf_taps, a literal list of 41
      symmetric (linear-phase) low-pass taps -> File Sink out.dat, and
  (b) directly a File Sink ref.dat.
The user sees out.dat lag ref.dat and asks to delay the stream written to ref.dat by exactly the
filter's group delay so the two line up sample for sample, leaving out.dat unchanged. The model has
to know that a linear-phase FIR with N taps delays by (N - 1) / 2 samples (here 20) and insert a
20-sample delay on the ref.dat branch (or compute it as (len(lpf_taps) - 1) // 2).

lpf_taps = firdes.low_pass(1, 48000, 4000, 2820, Hamming) rounded to 6 decimals: 41 taps, exactly
symmetric, gain 1.003 at 1.5 kHz, edge tap -0.0011.

V6 oracle (two observation points, both float File Sinks):
  f2p  ref.dat zeros      max |x| over the first 20 samples <= 1e-6 (the delay's zero samples; a
                          shorter delay puts the cosine there, a filtered copy puts the FIR transient
                          there: |x[0]| = 0.8 * 0.0011)
  f2p  ref.dat aligned    signed correlation at lag 0 of samples 20..531 with the source cosine
                          0.8 cos(2 pi 1500 m / 48000) >= 0.9999 (one sample off: cos(11.25 deg) = 0.981;
                          a longer delay leaves zeros in the window)
  p2p  ref.dat level      RMS of samples 20..4115 = 0.8 / sqrt(2) within 1 %
  p2p  out.dat unchanged  signed correlation at lag 0 of samples 0..511 with the start graph's FIR
                          output (np.convolve of the taps with the cosine, incl. the transient)
                          >= 0.9999, and RMS of samples 64..4159 within 1 % of its value
Tolerance-based (not bit-exact) on purpose: an FFT-filter delay is a valid answer with ~1e-7 rounding,
and FIR outputs are not bit-identical across VOLK machines.

Deterministic; regenerate with (inside WSL, from the repository root):

    PYTHONPATH=src ~/grc-agent/.venv/bin/python items/tier_f/test/gbf-test-07/build.py

Needs GNU Radio (catalog defaults are used to write the compact flowgraphs).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from grcbench.common.canary import CANARY             # noqa: E402
from grcbench.tier_f import compose as C              # noqa: E402
from grcbench.tier_f import grc, grcfile, taskio      # noqa: E402

TASK_ID = "gbf-test-07"
SPLIT = "test"

# ---- the specification -------------------------------------------------------------------------
FS = 48000           # sample rate, Hz
F_TONE = 1500        # cosine frequency, Hz (32 samples per period)
AMP = 0.8            # cosine amplitude
TAPS = [-0.001103, -0.000718, 0.0, 0.001219, 0.002893, 0.004558, 0.00531, 0.004055, 0.0, -0.006774,
        -0.014889, -0.021649, -0.023511, -0.01703, 0.0, 0.027549, 0.062877, 0.100808, 0.134761,
        0.158291, 0.166706, 0.158291, 0.134761, 0.100808, 0.062877, 0.027549, 0.0, -0.01703,
        -0.023511, -0.021649, -0.014889, -0.006774, 0.0, 0.004055, 0.00531, 0.004558, 0.002893,
        0.001219, 0.0, -0.000718, -0.001103]
NTAPS = len(TAPS)                     # 41
DELAY = (NTAPS - 1) // 2              # 20 samples: group delay of a linear-phase FIR
REF_FILE, OUT_FILE = "ref.dat", "out.dat"

ZERO_TOL = 1e-6      # "zero samples" (stated)
LEVEL_REL = 0.01     # levels within 1 % (stated)
CORR_MIN = 0.9999    # exact whole-sample alignment scores > 0.99999; one sample off scores 0.981
N_CORR = 512         # samples compared sample by sample
N_RMS = 4096         # samples for the level checks (128 periods: no leakage)
OUT_RMS_SKIP = 64    # skip the FIR transient (40 samples) for the out.dat level
N_ITEMS = 8192

assert NTAPS == 41 and TAPS == TAPS[::-1]

INSTRUCTION = (
    "This flowgraph runs at a sample rate of 48 kHz. A float Signal Source makes a 1.5 kHz cosine with "
    "amplitude 0.8. The cosine goes straight to a File Sink that writes ref.dat, and it also goes "
    "through an FIR Filter (a linear-phase low-pass whose taps are in the variable lpf_taps) to a File "
    "Sink that writes out.dat; both files store float samples. When I compare the two files sample by "
    "sample, out.dat lags ref.dat because of the filter's delay. Delay the stream written to ref.dat by "
    "exactly the FIR filter's group delay (a whole number of samples) so that the two files line up "
    "sample for sample: ref.dat should then start with that many zero samples (zero to within 1e-6), "
    "followed by the same 1.5 kHz cosine with amplitude 0.8. Leave the Signal Source, the FIR filter "
    "and out.dat unchanged (same samples, levels within 1 %). The flowgraph should keep producing "
    "samples until it is stopped; do not limit the number of samples."
)


# ---- start and reference --------------------------------------------------------------------------
def _variables() -> dict[str, str]:
    return {"samp_rate": str(FS), "lpf_taps": "[" + ", ".join(repr(t) for t in TAPS) + "]"}


def _src() -> dict:
    return C.sig_source("src", "float", freq=F_TONE, amp=AMP)


def _fir(name: str = "lpf", taps: str = "lpf_taps", bid: str = "fir_filter_xxx") -> dict:
    return C.block(name, bid, type="fff", decim="1", taps=taps)


def _graph(ref_branch: list[dict]) -> dict:
    """src -> lpf -> out_sink, and src -> <ref_branch...> -> ref_sink."""
    blocks = [_src(), _fir(), *ref_branch, C.file_sink("out_sink", "float", OUT_FILE),
              C.file_sink("ref_sink", "float", REF_FILE)]
    conns = [["src", "0", "lpf", "0"], ["lpf", "0", "out_sink", "0"]]
    prev = "src"
    for b in ref_branch:
        conns.append([prev, "0", b["name"], "0"])
        prev = b["name"]
    conns.append([prev, "0", "ref_sink", "0"])
    return C.graph("fir_delay_alignment", blocks, conns, _variables())


def start_full() -> dict:
    return _graph([])


def reference_full() -> dict:
    return _graph([C.block("align", "blocks_delay", type="float", delay=str(DELAY))])


def compact(data: dict) -> dict:
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in data["blocks"]}
    out = grcfile.compact(data, defaults)
    out["options"] = data["options"]
    diffs = grc.compact_equivalent(data, out)
    assert not diffs, f"compact form differs from the full form: {diffs}"
    return out


# ---- equivalent variants (genuinely different correct solutions) --------------------------------
def variant_delay_expression() -> dict:
    """Delay block whose length is computed from the taps: (len(lpf_taps) - 1) // 2."""
    return _graph([C.block("group_delay", "blocks_delay", type="float", delay="(len(lpf_taps) - 1) // 2")])


def variant_pure_delay_fir() -> dict:
    """A second Decimating FIR Filter whose taps are a pure 20-sample delay, [0] * 20 + [1]."""
    return _graph([_fir("delay_fir", "[0.0] * 20 + [1.0]")])


def variant_fft_delay_filter() -> dict:
    """An FFT Filter with pure-delay taps; its output carries FFT rounding (~1e-7), not bit-exact."""
    return _graph([_fir("delay_fft", "[0.0] * 20 + [1.0]", "fft_filter_xxx")])


# ---- expected signals ------------------------------------------------------------------------------
def source(n: int) -> np.ndarray:
    return AMP * np.cos(2 * np.pi * F_TONE / FS * np.arange(n))


def fir_output(n: int) -> np.ndarray:
    return np.convolve(np.asarray(TAPS), source(n))[:n]


def record(sig: tuple[str, ...], start_sig: tuple[str, ...], variant_sigs: dict[str, tuple]) -> dict:
    src_ref = [round(float(v), 7) for v in source(N_CORR)]
    out_ref = [round(float(v), 7) for v in fir_output(N_CORR)]
    out_rms = float(np.sqrt(np.mean(fir_output(OUT_RMS_SKIP + N_RMS)[OUT_RMS_SKIP:] ** 2)))
    ref_rms = AMP / math.sqrt(2)
    spec = {
        "zeros": f"ref.dat starts with exactly the {DELAY} zero samples of the delay (|x| <= 1e-6)",
        "aligned": (f"ref.dat from sample {DELAY} on is the source cosine sample for sample (a delay of exactly "
                    f"the group delay (N-1)/2 = {DELAY} samples of the {NTAPS}-tap linear-phase FIR): signed "
                    f"correlation at lag 0 with 0.8 cos(2 pi 1500 m / 48000) over {N_CORR} samples >= {CORR_MIN}"),
        "ref_level": f"ref.dat keeps the cosine's amplitude 0.8 (RMS {ref_rms:.6f} within 1 %)",
        "out_same": ("out.dat unchanged: the start graph's FIR output sample for sample (signed correlation at "
                     f"lag 0 over its first {N_CORR} samples, incl. the filter transient, >= {CORR_MIN}); this "
                     "also keeps the Signal Source's 1.5 kHz and the 48 kHz rate, and rejects a delay before the split"),
        "out_level": f"out.dat level unchanged (RMS {out_rms:.6f} within 1 %)",
        "observation": ("float samples written to the File Sinks ref.dat and out.dat (sink type float), "
                        f"running continuously (V5 liveness: at least {N_ITEMS} items at each)"),
    }
    return {
        "id": TASK_ID, "tier": "F", "split": SPLIT, "version": 2, "canary": CANARY,
        "license": "CC-BY-4.0 (original GRC-Bench item; licence to be confirmed by the team)",
        "source": {"kind": "composed_chain", "ref": "|".join(sig)},
        "cluster_id": C.cluster_id(sig),
        "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring",
                       "generator_version": "0.2.0"},
        "review": {"status": "pending-team-review", "reviewers": [], "notes": (
            "Re-themed 18 Oct (fix plan G5): the old item (low-pass a square wave into a sine) hashed to train "
            "under signature v2 and nearly duplicated gbf-test-01; its round-1 findings are moot. Modify L1 "
            "(5 functional blocks, one rate; R7). Known oracle limits: (1) alignment and 'unchanged' are "
            "checked with tolerances (correlation >= 0.9999, levels within 1 %, zeros within 1e-6), not "
            "bit-exact, so an FFT-based delay passes; a deliberate change below these tolerances would too. "
            "(2) A copy of the Signal Source plus a delay on the ref.dat branch writes the same samples and "
            "passes (behaviourally identical). Alternative-structure link: the pure-delay FIR / FFT variants "
            "have signature filter|tone, the cell of gbf-test-10 (same split, different function).")},
        "family": "modify", "level": "L1",
        "instruction": INSTRUCTION,
        "start_grc": "start.grc",
        "reference_grc": "reference.grc",
        "observation": {
            "points": [
                {"name": "ref", "sink": {"block_id": "blocks_file_sink", "params": {"file": REF_FILE, "type": "float"}}},
                {"name": "out", "sink": {"block_id": "blocks_file_sink", "params": {"file": OUT_FILE, "type": "float"}}},
            ],
            "run": {"n_items": N_ITEMS, "min_items": N_ITEMS, "timeout_s": 10.0},
            "contract": "the float samples written to the File Sinks whose file names are ref.dat and out.dat",
        },
        "oracle": {"checks": [
            {"point": "ref", "feature": "max_abs", "skip": 0, "n": DELAY, "max": ZERO_TOL, "group": "f2p",
             "property": "zeros", "requires": spec["zeros"]},
            {"point": "ref", "feature": "corr_ref", "skip": DELAY, "n": N_CORR, "ref": src_ref, "max_lag": 0,
             "signed": True, "min": CORR_MIN, "group": "f2p", "property": "aligned", "requires": spec["aligned"]},
            {"point": "ref", "feature": "rms", "skip": DELAY, "n": N_RMS, "expect": round(ref_rms, 6),
             "rel_tol": LEVEL_REL, "group": "p2p", "property": "ref_level", "requires": spec["ref_level"]},
            {"point": "out", "feature": "corr_ref", "skip": 0, "n": N_CORR, "ref": out_ref, "max_lag": 0,
             "signed": True, "min": CORR_MIN, "group": "p2p", "property": "out_same", "requires": spec["out_same"]},
            {"point": "out", "feature": "rms", "skip": OUT_RMS_SKIP, "n": N_RMS, "expect": round(out_rms, 6),
             "rel_tol": LEVEL_REL, "group": "p2p", "property": "out_level", "requires": spec["out_level"]},
        ]},
        "spec_properties": list(spec.values()),
        "mutations": [
            {"op": "set_param", "block": "align", "param": "delay", "value": "19", "breaks": spec["zeros"],
             "note": "one sample short"},
            {"op": "set_param", "block": "align", "param": "delay", "value": "21", "breaks": spec["aligned"],
             "note": "one sample long"},
            {"op": "set_param", "block": "align", "param": "delay", "value": "41", "breaks": spec["aligned"],
             "note": "the number of taps instead of (N - 1) / 2"},
            {"op": "set_param", "block": "align", "param": "delay", "value": "52", "breaks": spec["aligned"],
             "note": "group delay plus one cosine period (32 samples): same phase, wrong delay"},
            {"op": "bypass", "block": "align", "breaks": spec["zeros"], "note": "no delay (the start behaviour)"},
            {"op": "rewire_input", "dst": "lpf", "dst_port": "0", "new_src": "align", "breaks": spec["out_same"],
             "note": "the delay sits before the split, so out.dat is delayed too"},
            {"op": "rewire_input", "dst": "ref_sink", "dst_port": "0", "new_src": "lpf", "breaks": spec["zeros"],
             "note": "ref.dat gets a filtered copy (the delay's output is parked): it starts with the FIR transient"},
            {"op": "set_param", "block": "lpf_taps", "param": "value", "value": "[0.25, 0.25, 0.25, 0.25]",
             "breaks": spec["out_same"], "note": "the FIR filter changed"},
            {"op": "set_param", "block": "src", "param": "freq", "value": "1600", "breaks": spec["out_same"],
             "note": "the Signal Source changed"},
            {"op": "set_param", "block": "src", "param": "amp", "value": "0.84", "breaks": spec["ref_level"],
             "note": "near miss: the source level 5 % high (correlations are scale-free)"},
        ],
        "perturbation": {"perturbed": False, "changes": []},
        "disjointness": (
            "Group-delay compensation: delaying a reference branch by (N-1)/2 samples to line it up with a linear-"
            "phase FIR's output. Nearest training-side specs (reports/reference/training_task_specs.json): GRC-Agent "
            "scenarios 01_add_throttle and 23_lexical_conjugate_insert insert a named block into a path, and "
            "22_fm_rx_filter_squelch adds a filter; none is about filter delay or alignment, and no Qoherent task "
            "(t01-t25) uses a Delay block. "
            f"v2 signatures: reference {'|'.join(sig)} -> {SPLIT}; start {'|'.join(start_sig)} -> "
            f"{C.signature_split(start_sig)}; variants " + "; ".join(
                f"{k} {'|'.join(s)} -> {C.signature_split(s)}" for k, s in variant_sigs.items()) + "."
        ),
        "notes": (
            "lpf_taps = firdes.low_pass(1, 48000, 4000, 2820, window.WIN_HAMMING) rounded to 6 decimals (41 "
            "symmetric taps; gain 1.003 at 1.5 kHz). The pure-delay FIR and FFT-filter variants classify as "
            "filter|tone (gbf-test-10's cell, test split): a genuine structural alternative, listed so the "
            "reserved-signature list includes it."
        ),
    }


def main() -> int:
    start = compact(start_full())
    ref = compact(reference_full())
    variants = {
        "v1_delay_expression": compact(variant_delay_expression()),
        "v2_pure_delay_fir": compact(variant_pure_delay_fir()),
        "v3_fft_delay_filter": compact(variant_fft_delay_filter()),
    }
    sig, start_sig = C.signature(ref), C.signature(start)
    assert C.signature_split(sig) == SPLIT, f"reference signature {sig} -> {C.signature_split(sig)}, not {SPLIT}"
    variant_sigs = {k: C.signature(v) for k, v in variants.items()}
    print("v2 signatures:")
    print(f"  start      {'|'.join(start_sig)} -> {C.signature_split(start_sig)}")
    print(f"  reference  {'|'.join(sig)} -> {C.signature_split(sig)}")
    for k, s in variant_sigs.items():
        print(f"  {k:22s} {'|'.join(s)} -> {C.signature_split(s)}")
    files = {"start.grc": grcfile.dump(start), "reference.grc": grcfile.dump(ref)}
    files.update({f"variants/{k}.grc": grcfile.dump(v) for k, v in variants.items()})
    errors = taskio.save(HERE, record(sig, start_sig, variant_sigs), files)
    if errors:
        print("NOT WRITTEN:", *errors, sep="\n  ")
        return 1
    for stale in (HERE / "variants").glob("*.grc"):     # variants no longer produced by this script
        if f"variants/{stale.name}" not in files:
            stale.unlink()
    print(f"written {HERE.relative_to(REPO)} (signature {'|'.join(sig)} -> {SPLIT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
