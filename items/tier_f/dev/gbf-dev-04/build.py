"""Authoring script for GRC-Bench Tier-F item gbf-dev-04 (repair, L1, composed chain).

Theme (re-themed 18 Oct 2026, fix plan reports/tier_f_reviews/fix_plan.json): a heterodyne mixer that
converts an RF test carrier to a 2 kHz intermediate frequency (IF). At 48 kHz, the Signal Source `rf`
(10.7 kHz cosine, amplitude 0.8, float) is multiplied by a local-oscillator Signal Source `lo`
(amplitude 1, frequency = variable lo_freq = 8.7 kHz), a Low Pass Filter (cutoff 4 kHz, transition
width 1 kHz) removes the sum-frequency product (19.4 kHz), and the File Sink writes the IF as float
samples to if.dat. Five functional blocks, one sample rate: level L1 (R7; the team confirms the rule).

Fault (undefined_variable): the variable block lo_freq is deleted while the local oscillator's
frequency still refers to it. GRC loads the graph (V2 passes) but validation fails (V3): the LO's
frequency cannot be evaluated (name 'lo_freq' is not defined). The deleted value is visible nowhere,
so the repair has to work out the LO frequency from the stated behaviour: |f_RF - f_LO| = 2 kHz, i.e.
8.7 kHz (low-side LO, the reference) or 12.7 kHz (high-side LO, variant v1). Both are correct; the
instruction does not say which side the LO is on.

Signal: 0.8 cos(2 pi 10.7k t) * cos(2 pi f_LO t) = 0.4 cos(2 pi 2k t) + 0.4 cos(2 pi (10.7k + f_LO) t).
The filter keeps the 2 kHz term: amplitude 0.4, RMS 0.2828. The reference LPF (115 Hamming taps)
passes 2 kHz with gain 1.001 and attenuates 19.4 kHz by 66.7 dB (23.4 kHz for the high-side LO:
81.4 dB).

Observation points: the File Sink if.dat (item type float; the type is part of the match) and an
edge point on the output of block `rf` (no extra block, so the 5-block budget holds). Measurement
window: 16384 items are captured; the first 4096 are skipped (FIR start-up; covers filters of up to
~4000 taps), then exactly 9600 items are analysed (5 Hz FFT bins). band_fraction (power within
1.9-2.1 kHz >= 0.9999, i.e. everything else at least 40 dB below the IF tone) uses a Blackman window
(version 3): the instruction allows any IF within 1 % of 2 kHz, and such an IF need not sit on the
5 Hz bin grid. Without a window (version 2) a pure 2001 Hz IF leaked 0.34 % of its power outside the
band and failed (reviewer probe, reproduced: lo_freq 8699 gave 0.996593). With the window the
stated tolerance is what is graded: off-grid IFs of 1982-2018 Hz give 1.0 (probes), while the
bypassed filter (0.5) and the 18.8 kHz near-miss cutoff (sum product 36.6 dB down, about 0.99978)
still fail. A window scales every tone's power by the same factor, so the fraction still measures
the power ratio the instruction states. band_fraction (18 Oct Parseval correction) is used only on
the IF output, which has no DC or Nyquist component, so the correction does not change its value.

Version 3 (second review round, 18 Oct 2026; reports/tier_f_reviews/round1/gbf-dev-04.resolution.json,
"round2"): G3-dev04-band-fraction-tolerance accepted; window added, variant v5 (lo_freq 8697.5, IF
2002.5 Hz, half a bin off the grid: the worst case for the old check) and probes (--probes) added.

Oracle groups (fix-plan rule R8): the start graph fails at V3, so "p2p" means properties carried by
the parts of the graph the fault does not touch (the RF source and the mixer structure); every check
whose value depends on the repaired LO frequency is "f2p". scratch probe (18 Oct): the start graph
with lo_freq defined as 0 (the LO parked at DC) passes every p2p check and fails the f2p checks.

Graph checks (p2p): a path Signal Source -> Multiply -> File Sink and a path Multiply -> filter ->
File Sink, so an answer that deletes the mixer and writes a 2 kHz tone to if.dat fails. The filter
set accepts every filter block (named filters, FIR/FFT/IIR filters, moving average, ...); the
signal checks decide whether it removes the sum product. Known limit (review.notes): the checks are
by block type, so an answer that parks `rf` on a sink and mixes a different source would pass them.

Deterministic; regenerate with (inside WSL, from the repository root):

    PYTHONPATH=src ~/grc-agent/.venv/bin/python items/tier_f/dev/gbf-dev-04/build.py
    PYTHONPATH=src ~/grc-agent/.venv/bin/python items/tier_f/dev/gbf-dev-04/build.py --probes  # check

Needs GNU Radio (catalog defaults are used to write the compact files).
Drafted by Claude (Anthropic); pending team review.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from grcbench.tier_f import compose as C               # noqa: E402
from grcbench.tier_f import grc, grcfile, taskio       # noqa: E402

TASK_ID = "gbf-dev-04"
SPLIT = "dev"
VERSION = 3

# ---- the specification (every number the oracle depends on is stated in the instruction) --------
FS = 48000            # sample rate, Hz
RF_F = 10700          # RF test carrier, Hz
RF_AMP = 0.8          # RF amplitude
LO_F = 8700           # reference LO frequency (low-side), Hz: the deleted variable's value
LO_F_HIGH = 12700     # the high-side LO, equally correct
LO_AMP = 1.0          # LO amplitude (visible in the start graph)
IF_F = 2000           # intermediate frequency, Hz
IF_AMP = RF_AMP * LO_AMP / 2          # 0.4
LPF_CUTOFF, LPF_WIDTH = 4000, 1000    # reference low-pass filter (stays in the start graph)
OUT_FILE = "if.dat"
MIN_IF_FRACTION = 0.9999              # everything outside 1.9-2.1 kHz at least 40 dB below the IF tone
BF_WINDOW = "blackman"                # so an IF anywhere within 1 % (off the 5 Hz bin grid) is not penalised

IF_RMS = IF_AMP / math.sqrt(2)        # 0.2828
RF_RMS = RF_AMP / math.sqrt(2)        # 0.5657
FREQ_REL_TOL = 0.01                   # frequency within 1 %
AMP_REL_TOL = 0.05                    # amplitude / RMS within 5 %
N_ITEMS = 16384
SKIP, N = 4096, 9600                  # 5 Hz FFT bins; the nominal tones sit on bins, the window covers the rest
LO_OFF_GRID = 8697.5                  # variant v5: IF 2002.5 Hz, half a bin off the grid (worst case without a window)

RF, LO, MIX, LPF, OUT, LO_VAR = "rf", "lo", "mixer", "if_filter", "if_sink", "lo_freq"

INSTRUCTION = (
    "This flowgraph is a heterodyne mixer that converts a radio-frequency (RF) test carrier down to "
    "an intermediate frequency (IF). It runs at a sample rate of 48 kHz. The Signal Source block rf "
    "generates the RF carrier, a 10.7 kHz cosine with amplitude 0.8 (float samples). A second Signal "
    "Source, the local oscillator (amplitude 1), is multiplied with the carrier, and a low-pass filter "
    "after the multiplier keeps only the difference frequency. The result goes to the File Sink whose "
    "file name is if.dat (item type float).\n\n"
    "At the moment GNU Radio Companion refuses to generate the flowgraph: validation reports that a "
    "parameter value cannot be evaluated.\n\n"
    "Please repair it so that if.dat receives the RF carrier converted to a 2 kHz IF: a 2 kHz tone "
    "(within 1 %) with amplitude 0.4, which is half the RF amplitude, as an ideal multiplier with a "
    "unit-amplitude local oscillator gives (an RMS of 0.283, within 5 %). Everything else in the "
    "output, in particular the multiplier's sum-frequency product, must be at least 40 dB weaker than "
    "the IF tone. These figures apply after the first 4096 output samples, which cover the filter's "
    "start-up. Keep the mixer structure (the RF source and the local oscillator feeding the "
    "multiplier, then a low-pass filter, then the File Sink if.dat), and keep the RF source rf "
    "unchanged. The flowgraph should keep producing samples until it is stopped; do not limit the "
    "number of samples."
)

MULTIPLIERS = ["blocks_multiply_xx", "blocks_multiply_ff", "blocks_multiply_cc", "blocks_multiply_conjugate_cc"]
FILTERS = ["low_pass_filter", "band_pass_filter", "band_reject_filter", "high_pass_filter",
           "fir_filter_xxx", "fft_filter_xxx", "interp_fir_filter_xxx", "iir_filter_xxx",
           "single_pole_iir_filter_xx", "blocks_moving_average_xx", "dc_blocker_xx",
           "rational_resampler_xxx"]


# ---- reference -----------------------------------------------------------------------------------
def _mixer_graph(gid: str, lo_freq: str, variables: dict, lpf: dict | None = None,
                 lo_port: str = "1", extra_blocks: list | None = None, out_name: str = OUT) -> dict:
    blocks = [
        C.sig_source(RF, "float", freq=RF_F, amp=RF_AMP),
        C.sig_source(LO, "float", freq=lo_freq, amp=LO_AMP),
        C.block(MIX, "blocks_multiply_xx", type="float", num_inputs="2", vlen="1"),
        lpf or C.low_pass(LPF, "float", cutoff=LPF_CUTOFF, width=LPF_WIDTH),
        C.file_sink(out_name, "float", OUT_FILE),
    ]
    rf_port = "0" if lo_port == "1" else "1"
    filt = blocks[3]["name"]
    conns = [[RF, "0", MIX, rf_port], [LO, "0", MIX, lo_port], [MIX, "0", filt, "0"],
             [filt, "0", out_name, "0"]]
    g = C.graph(gid, blocks, conns, variables)
    for blk, (a, b) in extra_blocks or []:
        g["blocks"].append(blk)
        g["connections"] = [c for c in g["connections"] if not (c[0] == a and c[2] == b)]
        g["connections"] += [[a, "0", blk["name"], "0"], [blk["name"], "0", b, "0"]]
    return g


def reference_full() -> dict:
    return _mixer_graph("if_mixer", LO_VAR, {"samp_rate": str(FS), LO_VAR: str(LO_F)})


def compact(data: dict) -> dict:
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in data["blocks"]}
    out = grcfile.compact(data, defaults)
    out["options"] = data["options"]
    diffs = grc.compact_equivalent(data, out)
    assert not diffs, f"compact form differs from the full form: {diffs}"
    return out


# ---- the fault -----------------------------------------------------------------------------------
def inject_fault(ref: dict) -> dict:
    """Delete the variable block lo_freq; the local oscillator keeps referring to it."""
    start = copy.deepcopy(ref)
    before = len(start["blocks"])
    start["blocks"] = [b for b in start["blocks"] if b["name"] != LO_VAR]
    assert len(start["blocks"]) == before - 1, "variable block not found"
    lo = next(b for b in start["blocks"] if b["name"] == LO)
    assert lo["parameters"]["freq"] == LO_VAR
    # the name is now undefined everywhere: no other block defines or mentions it
    assert not any(LO_VAR in str(v) for b in start["blocks"] if b["name"] != LO
                   for v in (b.get("parameters") or {}).values())
    return start


# ---- equivalent variants (genuinely different correct repairs) ----------------------------------
def variant_high_side_lo() -> dict:
    """The deleted variable defined again with the high-side LO, 12.7 kHz (IF = 12.7 - 10.7 kHz)."""
    return _mixer_graph("if_mixer_v1", LO_VAR, {"samp_rate": str(FS), LO_VAR: str(LO_F_HIGH)})


def variant_literal_throttle() -> dict:
    """No variable: the LO frequency typed into the block as 8.7e3, the multiplier's inputs swapped
    (LO on port 0) and a Throttle inserted before the File Sink."""
    thr = C.block("throttle", "blocks_throttle", type="float", samples_per_second="samp_rate", vlen="1")
    return _mixer_graph("if_mixer_v2", "8.7e3", {"samp_rate": "48e3"}, lo_port="0",
                        extra_blocks=[(thr, (LPF, OUT))])


def variant_parameter_fir() -> dict:
    """lo_freq restored as a GRC Parameter block (12.7 kHz, high-side) and the low-pass filter
    realised as a generic FIR Filter with firdes.low_pass taps (3 kHz cutoff, 2 kHz width)."""
    lpf = C.block("lpf_fir", "fir_filter_xxx", type="fff", decim="1",
                  taps="firdes.low_pass(1.0, samp_rate, 3000, 2000)")
    g = _mixer_graph("if_mixer_v3", LO_VAR, {"samp_rate": str(FS)}, lpf=lpf, out_name="sink")
    g["blocks"].insert(1, C.block(LO_VAR, "parameter", label="LO frequency", type="eng_float",
                                  value="12.7e3", hide="none"))
    return g


def variant_expression() -> dict:
    """lo_freq written as an expression of two new variables (rf_freq - if_freq = 8.7 kHz)."""
    return _mixer_graph("if_mixer_v4", LO_VAR, {"samp_rate": str(FS), "rf_freq": str(RF_F),
                                                 "if_freq": str(IF_F), LO_VAR: "rf_freq - if_freq"})


def variant_lo_within_tolerance() -> dict:
    """lo_freq 8697.5 Hz: the IF is 2002.5 Hz (0.125 % off, inside the stated 1 %), half an FFT bin off
    the 5 Hz grid (review round 2: the rectangular-window band check of version 2 rejected it)."""
    return _mixer_graph("if_mixer_v5", LO_VAR, {"samp_rate": str(FS), LO_VAR: f"{LO_OFF_GRID:g}"})


# ---- probes: alternatives and wrong answers that are not shipped variants or single-edit mutants ---
def probes() -> dict[str, tuple[dict, bool, str]]:
    """name -> (flowgraph, expected to pass V6, what it shows)."""
    var = {"samp_rate": str(FS), LO_VAR: str(LO_F)}
    out = {}
    for lo, why in [("8699", "IF 2001 Hz (0.05 %)"), ("8701", "IF 1999 Hz"), ("8682", "IF 2018 Hz (0.9 %)"),
                    ("8718", "IF 1982 Hz (-0.9 %)"), ("12718", "high-side LO, IF 2018 Hz"),
                    ("8690", "IF 2010 Hz, on the bin grid"), ("8710", "IF 1990 Hz, on the bin grid")]:
        out[f"lo_{lo}"] = (_mixer_graph("probe_lo", LO_VAR, {"samp_rate": str(FS), LO_VAR: lo}), True,
                           f"round-2 G3: {why} lies within the stated 1 %, so it must pass (Blackman-windowed "
                           "band check)")
    tone = C.sig_source("if_tone", "float", freq=IF_F, amp=IF_AMP)
    null = C.block("park", "blocks_null_sink", type="float", vlen="1", num_inputs="1")
    blocks = [C.sig_source(RF, "float", freq=RF_F, amp=RF_AMP), C.sig_source(LO, "float", freq=LO_VAR, amp=LO_AMP),
              C.block(MIX, "blocks_multiply_xx", type="float", num_inputs="2", vlen="1"), null, tone,
              C.low_pass(LPF, "float", cutoff=LPF_CUTOFF, width=LPF_WIDTH), C.file_sink(OUT, "float", OUT_FILE)]
    parked = [[RF, "0", MIX, "0"], [LO, "0", MIX, "1"], [MIX, "0", "park", "0"]]
    out["shortcut_tone_through_lpf"] = (
        C.graph("probe_s1", blocks, parked + [["if_tone", "0", LPF, "0"], [LPF, "0", OUT, "0"]], var), False,
        "round-1 shortcut: a 2 kHz tone through the low-pass filter, mixer parked on a Null Sink; passes every "
        "signal check, fails the mixer-structure graph checks")
    out["shortcut_tone_direct"] = (
        C.graph("probe_s2", [b for b in blocks if b["name"] != LPF],
                parked + [["if_tone", "0", OUT, "0"]], var), False,
        "round-1 shortcut: a 2 kHz tone straight into if.dat; fails the mixer-structure graph checks")
    return out


def run_probes() -> int:
    from concurrent.futures import ThreadPoolExecutor

    import yaml

    from grcbench.tier_f.ladder import check
    task = taskio.load(HERE)
    todo = [(name, grcfile.dump(compact(g)), expect, why) for name, (g, expect, why) in probes().items()]
    # the start graph with the deleted variable defined as 0 (LO parked at DC): the R8 p2p/f2p split
    start = yaml.safe_load(taskio.read_grc(HERE, "start.grc"))
    assert not any(b["name"] == LO_VAR for b in start["blocks"])
    start["blocks"].insert(0, {"name": LO_VAR, "id": "variable", "parameters": {"value": "0"}})
    todo.append(("start_lo_freq_0", grcfile.dump(start), False,
                 "R8: the start graph with lo_freq defined as 0 passes every p2p check and fails f2p"))
    # calibration of the windowed band check: the sum-product mutants must still fail it
    from grcbench.tier_f import edit
    ref = yaml.safe_load(taskio.read_grc(HERE, "reference.grc"))
    for i, op in enumerate(task["mutations"]):
        if op["breaks"] == "sum_product_removed":
            todo.append((f"mutant_{i}", grcfile.dump(edit.apply(ref, op)), False, f"gate mutant: {op['note']}"))

    def one(item):
        name, text, expect, why = item
        return name, expect, why, check(text, task)

    bad = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        for name, expect, why, res in ex.map(one, todo):
            ok = res["first_failed"] is None and res["highest_passed"] == "V6"
            v6 = res["rungs"].get("V6") or {}
            bf = [c["value"] for c in v6.get("checks", []) if c["feature"] == "band_fraction"]
            failed = [f"{c['point']}.{c['feature']}={c['value']}" for c in v6.get("checks", []) if not c["passed"]]
            failed += [f"graph:{c['kind']}" for c in v6.get("graph_checks", []) if not c["passed"]]
            verdict = "as expected" if ok == expect else "UNEXPECTED"
            bad += ok != expect
            print(f"{name:26s} passed={ok!s:5s} first_failed={res['first_failed']} band_fraction={bf} "
                  f"groups={v6.get('groups')} failed={failed} -> {verdict}\n    {why}")
    return 1 if bad else 0


# ---- task record ---------------------------------------------------------------------------------
def observation() -> dict:
    obs = C.file_observation(OUT_FILE, n_items=N_ITEMS)
    obs["points"][0]["sink"]["params"]["type"] = "float"
    obs["points"].append({"name": "rf", "edge": {"block": RF, "port": "0"}})
    obs["contract"] = ("'out': the float samples written to the File Sink whose file name is if.dat (item "
                       "type float); 'rf': the output of the Signal Source block rf (edge tap, no extra "
                       "block). Both keep producing samples (V5 liveness: 16384 items at each point).")
    return obs


SPEC = [
    {"id": "validates", "text": "the flowgraph validates: every parameter value evaluates (the repaired fault)",
     "checked_by": "V3 (the start graph fails here)"},
    {"id": "output_point", "text": "float samples are written by a File Sink to if.dat",
     "checked_by": "observation 'out' (sink matched by file name and type float; a missing or mistyped point fails V6)"},
    {"id": "if_frequency", "text": "the output is a 2 kHz tone (within 1 %)",
     "checked_by": "f2p peak_freq at 'out'"},
    {"id": "if_amplitude", "text": "IF amplitude 0.4 = half the RF amplitude (RMS 0.283 within 5 %)",
     "checked_by": "f2p rms at 'out'"},
    {"id": "sum_product_removed",
     "text": "everything outside the IF tone (sum-frequency product, LO or RF leakage) at least 40 dB below it",
     "checked_by": "f2p band_fraction 1.9-2.1 kHz >= 0.9999 at 'out', Blackman window (an IF anywhere within "
                   "the stated 1 % is measured without leakage; the fraction is the stated power ratio)"},
    {"id": "rf_kept", "text": "the Signal Source rf is kept unchanged (10.7 kHz cosine, amplitude 0.8)",
     "checked_by": "p2p peak_freq and rms at edge point 'rf'"},
    {"id": "mixer_structure",
     "text": "the mixer structure is kept: a Signal Source feeds a multiplier whose output reaches if.dat "
             "through a filter",
     "checked_by": "p2p graph checks: path Signal Source -> multiplier -> File Sink and path multiplier -> "
                   "filter -> File Sink"},
    {"id": "run_length", "text": "the flowgraph keeps producing samples until it is stopped",
     "checked_by": "V5 liveness: at least 16384 items at both observation points; nothing more is claimed"},
    {"id": "context_only",
     "text": "the low-pass filter's cutoff and width, and which side of the carrier the LO is on, are not "
             "requirements; only the stated output properties are graded",
     "checked_by": "not checked by design (low-side 8.7 kHz and high-side 12.7 kHz LOs are both variants)"},
]


def record(sig: tuple[str, ...], variant_sigs: dict[str, tuple[str, ...]]) -> dict:
    win = {"skip": SKIP, "n": N}
    checks = [
        {"point": "out", "feature": "peak_freq", "fs": FS, "expect": IF_F, "tol": round(FREQ_REL_TOL * IF_F, 3),
         **win, "group": "f2p", "property": "if_frequency"},
        {"point": "out", "feature": "rms", "expect": round(IF_RMS, 4), "rel_tol": AMP_REL_TOL,
         **win, "group": "f2p", "property": "if_amplitude"},
        {"point": "out", "feature": "band_fraction", "fs": FS, "lo": IF_F - 100, "hi": IF_F + 100,
         "window": BF_WINDOW, "min": MIN_IF_FRACTION, **win, "group": "f2p", "property": "sum_product_removed"},
        {"point": "rf", "feature": "peak_freq", "fs": FS, "expect": RF_F, "tol": round(FREQ_REL_TOL * RF_F, 3),
         "n": N, "group": "p2p", "property": "rf_kept"},
        {"point": "rf", "feature": "rms", "expect": round(RF_RMS, 4), "rel_tol": AMP_REL_TOL,
         "n": N, "group": "p2p", "property": "rf_kept"},
    ]
    graph_checks = [
        {"kind": "path", "from_ids": ["analog_sig_source_x"], "via_ids": MULTIPLIERS,
         "to_ids": ["blocks_file_sink"], "group": "p2p", "property": "mixer_structure"},
        {"kind": "path", "from_ids": MULTIPLIERS, "via_ids": FILTERS,
         "to_ids": ["blocks_file_sink"], "group": "p2p", "property": "mixer_structure"},
    ]
    mutations = [
        {"op": "set_param", "block": LO_VAR, "param": "value", "value": "9700", "breaks": "if_frequency",
         "note": "LO 9.7 kHz: IF 1 kHz"},
        {"op": "set_param", "block": LO_VAR, "param": "value", "value": "8670", "breaks": "if_frequency",
         "note": "near miss: IF 2.03 kHz, 1.5 % off"},
        {"op": "set_param", "block": LO, "param": "samp_rate", "value": "32000", "breaks": "if_frequency",
         "note": "wrong sample rate in the LO: it runs at 13.05 kHz, IF 2.35 kHz"},
        {"op": "bypass", "block": LPF, "breaks": "sum_product_removed",
         "note": "missing filter: the 19.4 kHz sum product reaches the output"},
        {"op": "set_param", "block": LPF, "param": "cutoff_freq", "value": "18800",
         "breaks": "sum_product_removed", "note": "near miss: sum product only 36.6 dB down"},
        {"op": "set_param", "block": LPF, "param": "cutoff_freq", "value": "1500", "breaks": "if_amplitude",
         "note": "cutoff below the IF: the IF tone is suppressed"},
        {"op": "set_param", "block": LPF, "param": "cutoff_freq", "value": "2400", "breaks": "if_amplitude",
         "note": "near miss: the IF sits in the transition band (gain 0.91)"},
        {"op": "set_param", "block": LO, "param": "amp", "value": "0.5", "breaks": "if_amplitude",
         "note": "LO amplitude 0.5: IF amplitude 0.2"},
        {"op": "set_id", "block": MIX, "new_id": "blocks_add_xx", "breaks": "if_frequency",
         "note": "adder instead of multiplier: no mixing, the filter removes both inputs"},
        {"op": "rewire_input", "dst": MIX, "dst_port": "1", "new_src": RF, "breaks": "if_frequency",
         "note": "wrong tap: the RF is squared instead of mixed with the LO"},
        {"op": "set_param", "block": RF, "param": "amp", "value": "0.6", "breaks": "rf_kept",
         "note": "RF source changed"},
        {"op": "scale_param", "block": RF, "param": "freq", "factor": 1.1, "breaks": "rf_kept",
         "note": "RF source changed (11.77 kHz)"},
        {"op": "set_param", "block": OUT, "param": "file", "value": "if_out.dat", "breaks": "output_point",
         "note": "output written to another file"},
    ]
    notes = (
        "Re-themed on 18 Oct 2026 (fix plan): the old dev-04 (500 Hz tone + 60 Hz hum, high-pass) was a template "
        "twin of the old gbf-test-05 and hashed to train under signature v2; both were re-themed. "
        "Known limits: (1) the mixer-structure graph checks are by block type, so an answer that parks rf on a "
        "sink and mixes a different source into if.dat would pass them; (2) the LPF cutoff and width are not "
        "requirements (a repair may also retune or replace the filter, as long as the stated output holds); "
        "(3) an LO at 35.3 kHz (the alias of 12.7 kHz at 48 kHz) produces identical samples and passes; "
        "(4) the 40 dB purity requirement is measured as the power outside 1.9-2.1 kHz (band_fraction >= 0.9999, "
        "Blackman window since review round 2, so an IF anywhere within the stated 1 % passes; variant v5 and "
        "build.py --probes cover off-grid IFs); a spur within 100 Hz of the IF would count as IF power. "
        "Level L1 under the proposed rule R7 (5 functional blocks, single rate; a mixer is not a modulator); "
        "the team confirms R7. Relation to gbf-test-18 (CW receiver: complex down-conversion, decimation, BFO) "
        "and gbf-dev-06 (SSB mixing) is conceptual only. L1 repair conventions shared with gbf-test-10: "
        "symptom-only wording, p2p = properties of the parts the fault does not touch (R8), an edge "
        "observation point on the kept source. Signature-v2 cells of the variants: "
        + "; ".join(f"{k}: {'|'.join(v)}" for k, v in variant_sigs.items()) + "."
    )
    return {
        "id": TASK_ID, "tier": "F", "split": SPLIT, "version": VERSION,
        "license": "CC-BY-4.0 (original GRC-Bench item; licence to be confirmed by the team)",
        "source": {"kind": "composed_chain", "ref": "|".join(sig)},
        "cluster_id": C.cluster_id(sig),
        "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring",
                       "generator_version": "0.2.0"},
        "review": {"status": "pending-team-review", "reviewers": [], "notes": notes},
        "family": "repair", "level": "L1",
        "instruction": INSTRUCTION,
        "start_grc": "start.grc",
        "reference_grc": "reference.grc",
        "observation": observation(),
        "oracle": {"checks": checks, "graph_checks": graph_checks},
        "mutations": mutations,
        "spec_properties": SPEC,
        "perturbation": {"perturbed": False, "changes": []},
        "fault": {"type": "undefined_variable",
                  "description": (f"the variable block {LO_VAR} (value {LO_F}) was deleted while the local "
                                  f"oscillator's frequency ({LO}.freq) still refers to it, so GRC validation "
                                  f"fails: name '{LO_VAR}' is not defined. Fix: define the LO frequency so that "
                                  f"|10.7 kHz - f_LO| = 2 kHz (8.7 kHz or 12.7 kHz)."),
                  "expected_rung": "V3", "block": LO, "param": "freq", "undefined_name": LO_VAR},
        "disjointness": (
            "A composed heterodyne down-converter (RF and LO Signal Sources, Multiply, Low Pass Filter, File Sink) "
            "whose LO frequency variable was deleted; the repair needs the IF arithmetic |f_RF - f_LO| = f_IF. "
            "Nearest training-side spec: GRC-Agent scenario 06_query_knowledge_multiply (replace the adder of a "
            "two-tone dial tone with a multiplier on request): a multiplier of two tones too, but a requested "
            "block swap with no IF target, no filter and no fault. Other training specs repair other faults "
            "(t03/t17 FM receivers, t04 unconnected sink, t05 QAM rates/types, t15 load warnings, t20 resampler "
            "ratio); 04_add_and_remove_variable adds a variable on request rather than repairing an undefined "
            "name. v2 cell filter|mix|tone hashes to dev with either multiplier port order."),
        "notes": ("v2 signature of the reference: " + "|".join(sig) + "; variants: "
                  + "; ".join(f"{k}: {'|'.join(v)}" for k, v in variant_sigs.items())),
    }


def main(argv: list[str]) -> int:
    if "--probes" in argv:
        return run_probes()
    ref = compact(reference_full())
    sig = C.signature(ref)
    split = C.signature_split(sig)
    assert split == SPLIT, f"signature {sig} belongs to split {split}, not {SPLIT}"
    ids = grc.block_ids()
    variants = {
        "variants/v1_high_side_lo.grc": compact(variant_high_side_lo()),
        "variants/v2_literal_swapped_throttle.grc": compact(variant_literal_throttle()),
        "variants/v3_parameter_block_fir.grc": compact(variant_parameter_fir()),
        "variants/v4_expression_variables.grc": compact(variant_expression()),
        "variants/v5_lo_within_tolerance.grc": compact(variant_lo_within_tolerance()),
    }
    for d in [ref, *variants.values()]:
        unknown = [b["id"] for b in d["blocks"] if b["id"] not in ids]
        assert not unknown, f"unknown block ids {unknown}"
    n_func = sum(1 for b in ref["blocks"] if not b["id"].startswith(("variable", "parameter")))
    assert n_func <= 5, f"reference has {n_func} functional blocks (L1 allows 5)"
    start = inject_fault(ref)
    vsigs = {Path(k).stem: C.signature(v) for k, v in variants.items()}
    print(f"start     v2 {'|'.join(C.signature(start))}")
    print(f"reference v2 {'|'.join(sig)} -> {split}")
    for k, s in vsigs.items():
        print(f"{k:30s} v2 {'|'.join(s)} -> {C.signature_split(s)}")
    files = {"reference.grc": grcfile.dump(ref), "start.grc": grcfile.dump(start)}
    files.update({k: grcfile.dump(v) for k, v in variants.items()})
    errors = taskio.save(HERE, record(sig, vsigs), files)
    if errors:
        print("NOT WRITTEN:", *errors, sep="\n  ")
        return 1
    for stale in sorted((HERE / "variants").glob("*.grc")):     # drop variants of the old theme
        if f"variants/{stale.name}" not in files:
            stale.unlink()
            print(f"removed stale {stale.relative_to(REPO)}")
    print(f"written {HERE.relative_to(REPO)} (signature {'|'.join(sig)} -> {split})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
