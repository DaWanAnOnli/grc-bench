"""Authoring script for GRC-Bench Tier-F item gbf-test-09 (repair, L1, fault: type_mismatch).

Deterministic: re-running it rewrites task.json, reference.grc, start.grc and variants/*.grc
byte-identically. Run inside WSL from the repository root:

    PYTHONPATH=src ~/grc-agent/.venv/bin/python items/tier_f/test/gbf-test-09/build.py
then run the gates:
    ./gb tier-f gates items/tier_f/test/gbf-test-09

Source: the GNU Radio 3.10.12 example blocks/peak_detector2.grc (a test-split seed). The example
runs a sawtooth through a Throttle into a Peak Detector2 block, converts the detector's byte peak
indicator to float with Char To Float, and shows the input, the peak indicator and the detector's
running-average (debug) output on one Qt time sink. This item keeps the detector chain, replaces
the GUI sink with two File Sinks (peaks.dat for the indicator, average.dat for the running average),
drops the Throttle and the example's disabled File Sinks (5 functional blocks, single rate: L1), and
perturbs every number (see PERTURBATION below).

The fault: the Char To Float converter is replaced by Short To Float. Peak Detector2's output 0 is
byte (char), so GRC validation reports a type and size mismatch on that connection and the start
fails at rung V3. Drafted by Claude (Anthropic); pending team review.

Oracle (revised 18 Oct after review round 1):
  f2p  peaks.dat: exactly 192 flags in 192 sawtooth periods (mean 0.0125 +-0.2 %, i.e. no extra or
       missing flag), the flags exactly at the ramp tops (samples 39 + 80k; equal_prefix over all
       192 periods of the window), indicator values 1.0 / 0.0.
  p2p  properties of the parts the fault does not touch (R8): the sawtooth itself (an edge
       observation point on the Signal Source output: bottom, top and sawtooth shape), the running
       average in average.dat (mean within 5 % of 1.0 and every sample within 0.95-1.05 from 0.1 s
       on; tooling T4 'min'/'max'), and a graph check that the sawtooth reaches a File Sink through
       the Peak Detector2. Proved on the start graph with the faulted branch parked (detector out0 to a
       byte Null Sink): every p2p check passes, the f2p checks fail (observation missing).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # .../grc-bench
sys.path.insert(0, str(ROOT / "src"))

from grcbench.common.canary import CANARY                                 # noqa: E402
from grcbench.common.io import sha256_file                                # noqa: E402
from grcbench.common.paths import GNURADIO_EXAMPLES, SPLITS                # noqa: E402
from grcbench.tier_f import compose as C, edit, grc, grcfile, taskio      # noqa: E402

TASK_ID = "gbf-test-09"
SPLIT = "test"
SEED = "blocks/peak_detector2.grc"

FS = 48000            # sample rate (Hz)
SAW_F = 600           # sawtooth frequency (Hz)
SAW_AMP = 1.5         # sawtooth amplitude: ramps from OFFSET to OFFSET + SAW_AMP
OFFSET = 0.25         # sawtooth offset (bottom of the ramp)
FACTOR = 0.2          # Peak Detector2 rise threshold factor
ALPHA = 0.002         # Peak Detector2 averaging constant
LOOKAHEAD = "int(samp_rate/saw_freq/1.1)"   # = 72 samples, 0.9 of a sawtooth period
PEAKS_FILE = "peaks.dat"
AVG_FILE = "average.dat"

N_ITEMS = 16384       # observed items per point (0.34 s at 48 kHz)
PERIOD = FS // SAW_F  # 80 samples
PEAK_SKIP = 6 * PERIOD            # 480: skip the first 10 ms
PEAK_N = 192 * PERIOD             # 15360: a whole number of periods, so the count is exact
N_PEAKS = PEAK_N // PERIOD        # 192
TOP = PERIOD // 2 - 1             # 39: the Signal Source's sawtooth starts mid-ramp (phase 0), so the last
                                  # sample before every drop is sample 39 + 80k
POS_N = PEAK_N                    # 15360: exact flag positions over all 192 periods of the window
AVG_SKIP = FS // 10               # 4800: 0.1 s; the running average (alpha 0.002, ~500-sample time
                                  # constant) has settled to within 0.01 % by then
SAW_BOTTOM = OFFSET                               # 0.25
SAW_TOP = OFFSET + SAW_AMP * (PERIOD - 1) / PERIOD   # 1.73125: the last sample of a ramp
LEVEL_TOL = 0.05 * SAW_AMP                        # 0.075: 5 % of the sawtooth's 1.5 swing
SAW_H2 = 0.152318     # share of the sawtooth's AC power in its 2nd harmonic (1.2 kHz): (1/4) / sum(1/k^2)
                      # for the 80-sample discrete ramp (computed with oracles.band_fraction on the ideal
                      # ramp 0.25 + 1.5*((k+40) % 80)/80); a triangle or square wave has none

# ---------------------------------------------------------------- the seed and its original numbers
seed_path = GNURADIO_EXAMPLES / SEED
seed_sha = sha256_file(seed_path)
seed_split = json.loads((SPLITS / "flowgraph_seed_split.json").read_text())["seeds"][SEED]
assert seed_split["split"] == SPLIT, seed_split
assert seed_split["sha256"] == seed_sha, (seed_split["sha256"], seed_sha)
seed = yaml.safe_load(seed_path.read_text(encoding="utf-8"))
sb = {b["name"]: b for b in seed["blocks"]}
assert {b["id"] for b in seed["blocks"]} >= {"analog_sig_source_x", "blocks_throttle2", "blocks_peak_detector2_fb",
                                             "blocks_char_to_float", "qtgui_time_sink_x"}
src0 = sb["analog_sig_source_x_0"]["parameters"]
assert src0["waveform"] == "analog.GR_SAW_WAVE" and src0["type"] == "float"
ORIG = {"samp_rate": sb["samp_rate"]["parameters"]["value"], "freq": src0["freq"], "amp": src0["amp"],
        "offset": sb["offset"]["parameters"]["value"], "factor": sb["factor"]["parameters"]["value"],
        "alpha": sb["alpha"]["parameters"]["value"], "lookahead": sb["lookahead"]["parameters"]["value"]}
ORIG_OPTS = seed["options"]["parameters"]
assert ORIG_OPTS["id"] == "test_peak2" and ORIG_OPTS["generate_options"] == "qt_gui"

PERTURBATION = {"perturbed": True, "changes": [
    {"what": "sample rate (variable samp_rate)", "original": ORIG["samp_rate"], "new": str(FS)},
    {"what": "sawtooth frequency (Signal Source freq; now the variable saw_freq)", "original": ORIG["freq"],
     "new": str(SAW_F)},
    {"what": "sawtooth amplitude (Signal Source amp)", "original": ORIG["amp"], "new": str(SAW_AMP)},
    {"what": "sawtooth offset (variable offset; a Qt range widget in the example)", "original": ORIG["offset"],
     "new": str(OFFSET)},
    {"what": "Peak Detector2 rise threshold factor (variable factor)", "original": ORIG["factor"], "new": str(FACTOR)},
    {"what": "Peak Detector2 alpha (variable alpha)", "original": ORIG["alpha"], "new": str(ALPHA)},
    {"what": "Peak Detector2 look-ahead (variable lookahead)", "original": ORIG["lookahead"], "new": LOOKAHEAD},
    # conversion changes, recorded for completeness (R11)
    {"what": "Qt range widget offset", "original": "variable_qtgui_range", "new": "plain variable"},
    {"what": "Qt Time Sink qtgui_time_sink_x_0 (3 inputs: sawtooth, indicator, running average)",
     "original": "GUI display", "new": "File Sinks peaks.dat (indicator) and average.dat (running average); the "
                                       "sawtooth input display dropped"},
    {"what": "Throttle blocks_throttle2_0 and the three disabled File Sinks (in.data, avg.data, flag.data)",
     "original": "present", "new": "removed"},
    {"what": "options block", "original": "id test_peak2, generate_options qt_gui, category Custom, "
                                          "window_size/placement/sizing_mode set",
     "new": "id sawtooth_peaks, generate_options no_gui, output_language python (other options stripped)"},
]}
DERIVATION = ("Qt time sink (3 inputs) replaced by two File Sinks: peaks.dat (Char To Float output) and "
              "average.dat (Peak Detector2 debug output); the input display, the Throttle and the example's "
              "three disabled File Sinks dropped; the offset Qt range widget became a plain variable; GUI options "
              "stripped.")

# ---------------------------------------------------------------- reference (compact form)
CONV = "blocks_char_to_float_0"
DET = "blocks_peak_detector2_fb_0"
SRC = "analog_sig_source_x_0"
PEAKS_SINK = "blocks_file_sink_0"
AVG_SINK = "blocks_file_sink_1"

full = C.graph("sawtooth_peaks", [
    C.sig_source(SRC, "float", freq="saw_freq", amp=SAW_AMP, waveform="analog.GR_SAW_WAVE", offset="offset"),
    C.block(DET, "blocks_peak_detector2_fb", threshold_factor_rise="factor", look_ahead="lookahead", alpha="alpha"),
    C.block(CONV, "blocks_char_to_float", vlen="1", scale="1"),
    C.file_sink(PEAKS_SINK, "float", PEAKS_FILE),
    C.file_sink(AVG_SINK, "float", AVG_FILE),
], [[SRC, "0", DET, "0"], [DET, "0", CONV, "0"], [CONV, "0", PEAKS_SINK, "0"], [DET, "1", AVG_SINK, "0"]],
    {"samp_rate": FS, "saw_freq": SAW_F, "offset": OFFSET, "factor": FACTOR, "alpha": ALPHA, "lookahead": LOOKAHEAD})


def compact(data: dict) -> dict:
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in data["blocks"]}
    out = grcfile.compact(data, defaults)
    diffs = grc.compact_equivalent(data, out)
    assert not diffs, diffs
    return out


def replace_block(data: dict, old_name: str, new_block: dict) -> dict:
    """Swap one block for another (new name allowed); connections follow the new name."""
    d = copy.deepcopy(data)
    d["blocks"] = [copy.deepcopy(new_block) if b["name"] == old_name else b for b in d["blocks"]]
    d["connections"] = [[new_block["name"] if x == old_name else x for x in c] for c in d["connections"]]
    return d


ref = compact(full)
ids = grc.block_ids()
assert all(b["id"] in ids for b in ref["blocks"])
functional = [b for b in ref["blocks"] if not b["id"].startswith("variable")]
assert len(functional) == 5, functional          # L1: <= 5 functional blocks, single rate

# ---------------------------------------------------------------- starting flowgraph (the fault)
WRONG = "blocks_short_to_float"
start = replace_block(ref, CONV, C.block("blocks_short_to_float_0", WRONG))

# ---------------------------------------------------------------- equivalent variants
# v1: UChar To Float (unsigned byte -> float) also maps the 0/1 indicator to 0.0/1.0.
v1 = replace_block(ref, CONV, C.block("blocks_uchar_to_float_0", "blocks_uchar_to_float"))

# v2: keep the Short To Float of the start and put a Char To Short in front of it. Char To Short
# scales by 256 (volk 8i -> 16i), so the Short To Float divides by 256 to give 1.0 again.
v2 = copy.deepcopy(start)
v2 = edit.set_param(v2, "blocks_short_to_float_0", "scale", "256")
v2["blocks"].insert(len(v2["blocks"]) - 3, C.block("blocks_char_to_short_0", "blocks_char_to_short"))
v2["connections"] = [c for c in v2["connections"] if not (c[0] == DET and c[2] == "blocks_short_to_float_0")]
v2["connections"] += [[DET, "0", "blocks_char_to_short_0", "0"],
                      ["blocks_char_to_short_0", "0", "blocks_short_to_float_0", "0"]]

# v3: Char To Float, plus the example's Throttle between the source and the detector (real time).
v3 = copy.deepcopy(ref)
v3["blocks"].insert(len(v3["blocks"]) - 4, C.block("blocks_throttle2_0", "blocks_throttle2", type="float",
                                                   samples_per_second="samp_rate"))
v3["connections"] = [c for c in v3["connections"] if not (c[0] == SRC and c[2] == DET)]
v3["connections"] += [[SRC, "0", "blocks_throttle2_0", "0"], ["blocks_throttle2_0", "0", DET, "0"]]

# v4 (review round 1): literal numbers instead of the variables (samp_rate written 48e3) and absolute
# File Sink paths; the observation points match on the file's base name.
v4 = replace_block(ref, CONV, C.block("to_float", "blocks_char_to_float"))
v4 = edit.set_param(v4, "samp_rate", "value", "48e3")
v4 = edit.set_param(v4, SRC, "freq", "600")
v4 = edit.set_param(v4, SRC, "offset", "0.25")
v4 = edit.set_param(v4, DET, "threshold_factor_rise", "0.2")
v4 = edit.set_param(v4, DET, "look_ahead", "72")
v4 = edit.set_param(v4, DET, "alpha", "0.002")
v4 = edit.set_param(v4, PEAKS_SINK, "file", "/tmp/sawtooth/peaks.dat")
v4 = edit.set_param(v4, AVG_SINK, "file", "/tmp/sawtooth/average.dat")

# v5 (review round 1): the faulty Short To Float left in the graph but disabled, a Char To Float added.
v5 = copy.deepcopy(start)
for b in v5["blocks"]:
    if b["name"] == "blocks_short_to_float_0":
        b["states"] = {"state": "disabled"}
v5["connections"] = [c for c in v5["connections"] if "blocks_short_to_float_0" not in (c[0], c[2])]
v5["blocks"].insert(len(v5["blocks"]) - 2, C.block("blocks_char_to_float_0", "blocks_char_to_float"))
v5["connections"] += [[DET, "0", "blocks_char_to_float_0", "0"], ["blocks_char_to_float_0", "0", PEAKS_SINK, "0"]]

variants = {"v1_uchar_to_float.grc": v1, "v2_char_to_short_then_short_to_float.grc": v2,
            "v3_throttle_and_char_to_float.grc": v3, "v4_literal_numbers_absolute_paths.grc": v4,
            "v5_old_converter_disabled.grc": v5}
variants = {k: compact(v) for k, v in variants.items()}
assert any((b.get("states") or {}).get("state") == "disabled"         # compact() keeps non-enabled states
           for b in variants["v5_old_converter_disabled.grc"]["blocks"])

# ---------------------------------------------------------------- instruction
instruction = (
    "This flowgraph should mark the peaks of a sawtooth wave. Its Signal Source generates a 600 Hz sawtooth "
    "as float samples at a sample rate of 48 kHz, with amplitude 1.5 and offset 0.25: every period ramps up "
    "from 0.25 to about 1.75 and then drops back, so the wave's average level is about 1.0. The Peak "
    "Detector2 block should flag the top of every ramp (the last sample before the drop), giving exactly one "
    "detected peak per sawtooth period. Its peak indicator must be written to the File Sink with file name "
    "peaks.dat and item type float, as 1.0 at each detected peak and 0.0 at every other sample. Its second "
    "(debug) output, the detector's running average of the input, must be written to the File Sink with file "
    "name average.dat and item type float; from 0.1 s after the start it should stay within 5 % of the "
    "sawtooth's average level of 1.0, between 0.95 and 1.05.\n\n"
    "At the moment GNU Radio Companion refuses to generate the flowgraph: validation reports a connection "
    "whose source and sink IO types and IO sizes do not match. Please repair the flowgraph so that it "
    "validates, runs and behaves as described. Keep the sawtooth source and both output files as they are. "
    "The flowgraph should keep producing samples until it is stopped; do not limit the number of samples. "
    "The peaks are judged from 10 ms after the start: every flag must sit exactly on a ramp top, with no "
    "extra or missing flags; the sawtooth's bottom and top must stay within 0.075 (5 % of its 1.5 swing)."
)

# ---------------------------------------------------------------- observation and oracle
# The File Sink item type is part of each observation point: a sink of another type does not match,
# so the point is missing and V5 fails (e.g. a byte File Sink fed straight from the detector).
# The "saw" point taps the Signal Source's output directly (an edge point adds no block).
observation = {
    "points": [{"name": "peaks", "sink": {"block_id": "blocks_file_sink", "params": {"file": PEAKS_FILE, "type": "float"}}},
               {"name": "average", "sink": {"block_id": "blocks_file_sink", "params": {"file": AVG_FILE, "type": "float"}}},
               {"name": "saw", "edge": {"block": SRC, "port": "0"}}],
    "run": {"n_items": N_ITEMS, "min_items": N_ITEMS, "timeout_s": 10.0},
    "contract": ("the samples written to the File Sink whose file name is peaks.dat (the peak indicator) and to "
                 "the File Sink whose file name is average.dat (the detector's running average), both with "
                 f"item type float, and the output of the Signal Source block {SRC} (the sawtooth, kept from the "
                 "starting flowgraph)"),
}
RATE = SAW_F / FS                    # one peak per period: 0.0125 peaks per sample
POSITIONS = [1 if (PEAK_SKIP + i) % PERIOD == TOP else 0 for i in range(POS_N)]
assert sum(POSITIONS) == POS_N // PERIOD
checks = [
    # f2p: the repaired path (detector output 0 -> converter -> peaks.dat) delivers the indicator
    {"point": "peaks", "feature": "mean", "expect": RATE, "rel_tol": 0.002, "skip": PEAK_SKIP, "n": PEAK_N,
     "group": "f2p", "property": "peak_rate",
     "note": "exactly 192 flags in 192 whole periods: one extra or missing flag changes the mean by 0.52 %"},
    {"point": "peaks", "feature": "equal_prefix", "ref": POSITIONS, "max_offset": 0, "skip": PEAK_SKIP,
     "group": "f2p", "property": "peak_position",
     "note": "all 192 periods of the window equal the ideal indicator: 1.0 at samples 39 + 80k (the ramp "
             "tops of the unchanged sawtooth), 0.0 elsewhere (so no flag is extra, missing or misplaced)"},
    {"point": "peaks", "feature": "max_abs", "expect": 1.0, "tol": 0.001, "skip": PEAK_SKIP, "n": PEAK_N,
     "group": "f2p", "property": "peak_indicator", "note": "indicator value 1.0 at a peak"},
    {"point": "peaks", "feature": "rms", "expect": round(RATE ** 0.5, 6), "rel_tol": 0.002, "skip": PEAK_SKIP,
     "n": PEAK_N, "group": "f2p", "property": "peak_indicator",
     "note": "rms = sqrt(mean) holds only for a 0.0/1.0 indicator"},
    # p2p (R8): the parts the fault does not touch
    {"point": "average", "feature": "mean", "expect": 1.0, "rel_tol": 0.05, "skip": AVG_SKIP, "group": "p2p",
     "property": "average_level", "note": "settled running average = sawtooth mean (0.25 + 1.5 * 79/160 = 0.9906)"},
    {"point": "average", "feature": "min", "min": 0.95, "skip": AVG_SKIP, "group": "p2p",
     "property": "average_band", "note": "never below 0.95 from 0.1 s on (reference minimum 0.9806)"},
    {"point": "average", "feature": "max", "max": 1.05, "skip": AVG_SKIP, "group": "p2p",
     "property": "average_band", "note": "never above 1.05 from 0.1 s on (reference maximum 1.0107; the raw "
                                         "sawtooth reaches 1.73)"},
    {"point": "saw", "feature": "min", "expect": SAW_BOTTOM, "tol": LEVEL_TOL, "skip": PEAK_SKIP, "n": PEAK_N,
     "group": "p2p", "property": "sawtooth_kept", "note": "bottom of the ramp = offset 0.25 (+-5 % of the swing)"},
    {"point": "saw", "feature": "max", "expect": SAW_TOP, "tol": LEVEL_TOL, "skip": PEAK_SKIP, "n": PEAK_N,
     "group": "p2p", "property": "sawtooth_kept",
     "note": "top of the ramp = 0.25 + 1.5 * 79/80 = 1.731 (the last sample before the drop; +-5 % of the swing)"},
    {"point": "saw", "feature": "band_fraction", "fs": FS, "lo": 2 * SAW_F - 50, "hi": 2 * SAW_F + 50,
     "remove_mean": True, "expect": SAW_H2, "rel_tol": 0.10, "skip": PEAK_SKIP, "n": PEAK_N,
     "group": "p2p", "property": "sawtooth_kept",
     "note": "sawtooth shape: its 2nd harmonic (1.2 kHz) carries 15.2 % of the AC power (a triangle or square "
             "wave: 0)"},
]
graph_checks = [
    {"kind": "path", "from_ids": ["analog_sig_source_x"], "via_ids": ["blocks_peak_detector2_fb"],
     "to_ids": ["blocks_file_sink"], "group": "p2p", "property": "detector_used",
     "note": "the sawtooth reaches a File Sink through the Peak Detector2 (the instruction names that block)"},
]

P2P_MEANING = ("p2p here means properties of the parts the fault does not touch (R8): the start fails at V3 and never "
               "runs; with the faulted branch parked (detector out0 to a byte Null Sink, Short To Float and peaks.dat "
               "removed) every p2p check passes and the f2p checks fail")
spec_properties = [
    {"id": "validates", "text": "the repaired flowgraph passes GRC validation (the IO types of every connection "
                                "match), generates and runs", "checked_by": "ladder V3-V4"},
    {"id": "outputs", "text": "float items are written to the File Sinks peaks.dat and average.dat",
     "checked_by": "observation contract (file name and item type float; a missing or differently typed sink fails V5)"},
    {"id": "run_length", "text": "the flowgraph keeps producing samples until it is stopped",
     "checked_by": f"V4 liveness: at least {N_ITEMS} items at each observation point"},
    {"id": "peak_rate", "text": "exactly one detected peak per period of the 600 Hz sawtooth at 48 kHz: 192 flags in "
                                "192 whole periods (mean 0.0125 +-0.2 %, so no extra or missing flag)",
     "checked_by": "f2p mean on peaks.dat (also catches a wrong sample rate or sawtooth frequency)"},
    {"id": "peak_position", "text": "every flag sits on a ramp top: samples 39 + 80k of the unchanged sawtooth",
     "checked_by": "f2p equal_prefix on peaks.dat (all 192 periods of the window, no offset; a delayed indicator, "
                   "a shifted or reshaped wave fails)"},
    {"id": "peak_indicator", "text": "peaks.dat holds 1.0 at each detected peak and 0.0 at every other sample "
                                     "(maximum 1.0; RMS sqrt(0.0125) = 0.1118)",
     "checked_by": "f2p max_abs, f2p rms (and the exact values in equal_prefix)"},
    {"id": "average_level", "text": "average.dat holds the detector's running average, which from 0.1 s on sits at "
                                    "the sawtooth's average level 1.0 (mean within 5 %)",
     "checked_by": "p2p mean (catches changes that shift the mean: amplitude, offset, a sine centred on the offset; "
                   "the sawtooth shape itself is checked at the 'saw' point). " + P2P_MEANING},
    {"id": "average_band", "text": "from 0.1 s on every running-average sample lies between 0.95 and 1.05",
     "checked_by": "p2p min and max on average.dat (tooling T4)"},
    {"id": "sawtooth_kept", "text": "the Signal Source still produces the sawtooth: bottom 0.25 and top 1.731 within "
                                    "0.075 (5 % of the 1.5 swing), sawtooth shape (2nd-harmonic share 0.152 +-10 %); "
                                    "its frequency and phase are pinned by peak_rate and peak_position",
     "checked_by": "p2p min, max, band_fraction at the edge point 'saw' (the Signal Source output)"},
    {"id": "detector_used", "text": "the Peak Detector2 block does the detection",
     "checked_by": "p2p graph check: path Signal Source -> Peak Detector2 -> File Sink"},
]

mutations = [
    {"op": "scale_param", "block": "saw_freq", "param": "value", "factor": 1.25, "breaks": "peak_rate",
     "note": "750 Hz sawtooth"},
    {"op": "set_param", "block": "saw_freq", "param": "value", "value": "605", "breaks": "peak_rate",
     "note": "near miss: 605 Hz sawtooth (194 flags in the window)"},
    {"op": "set_param", "block": "samp_rate", "param": "value", "value": "32000", "breaks": "peak_rate",
     "note": "wrong sample rate (the example's 32 kHz)"},
    {"op": "set_param", "block": "lookahead", "param": "value", "value": "int(2.5*samp_rate/saw_freq)",
     "breaks": "peak_rate", "note": "look-ahead of 2.5 periods: most peaks are missed"},
    {"op": "set_param", "block": "lookahead", "param": "value", "value": "int(samp_rate/1e3/1.1)",
     "breaks": "peak_rate", "note": "near miss: the example's look-ahead formula (43 samples) gives 194 flags, some "
                                    "not at ramp tops"},
    {"op": "set_param", "block": "factor", "param": "value", "value": "7", "breaks": "peak_rate",
     "note": "the block's default rise factor 7: no peak is detected after start-up"},
    {"op": "set_param", "block": SRC, "param": "phase", "value": "1.0", "breaks": "peak_position",
     "note": "sawtooth phase changed: flags move 13 samples (the sawtooth is not kept as it was)"},
    {"op": "set_param", "block": CONV, "param": "scale", "value": "2", "breaks": "peak_indicator",
     "note": "indicator 0.5 instead of 1.0"},
    {"op": "rewire_input", "dst": PEAKS_SINK, "dst_port": "0", "new_src": DET, "new_src_port": "1",
     "breaks": "peak_indicator", "note": "wrong tap: the running average is written to peaks.dat"},
    {"op": "rewire_input", "dst": AVG_SINK, "dst_port": "0", "new_src": SRC, "breaks": "average_band",
     "note": "wrong tap: the raw sawtooth is written to average.dat"},
    {"op": "rewire_input", "dst": AVG_SINK, "dst_port": "0", "new_src": CONV, "breaks": "average_level",
     "note": "wrong tap: the peak indicator is written to average.dat"},
    {"op": "set_param", "block": SRC, "param": "amp", "value": "1", "breaks": "average_level",
     "note": "sawtooth amplitude 1 (average level 0.75)"},
    {"op": "set_param", "block": "offset", "param": "value", "value": ORIG["offset"], "breaks": "average_level",
     "note": "the example's offset 0.1 (average level 0.85)"},
    {"op": "set_param", "block": SRC, "param": "amp", "value": "1.65", "breaks": "sawtooth_kept",
     "note": "near miss: amplitude +10 % with the same offset (top 1.88)"},
    {"op": "set_param", "block": "alpha", "param": "value", "value": "0.02", "breaks": "average_band",
     "note": "averaging ten times faster: the running average ripples up to about 1.18"},
    {"op": "set_param", "block": SRC, "param": "waveform", "value": "analog.GR_SIN_WAVE", "breaks": "average_level",
     "note": "sine instead of sawtooth (average level 0.25)"},
    {"op": "set_param", "block": SRC, "param": "waveform", "value": "analog.GR_TRI_WAVE", "breaks": "sawtooth_kept",
     "note": "triangle instead of sawtooth: same mean, bottom and top; peaks move to the triangle tops"},
    {"op": "set_param", "block": PEAKS_SINK, "param": "file", "value": "peak.dat", "breaks": "outputs",
     "note": "indicator written to another file name"},
]

REVIEW_NOTES = (
    "Revised 18 Oct after review round 1 (see reports/tier_f_reviews/round1/gbf-test-09.resolution.json). The "
    "'keep the Peak Detector2 settings' clause was removed: settings the oracle cannot see do not change the "
    "behaviour, and the behaviour they set is checked (exact flag count and positions, the 0.95-1.05 band of the "
    "running average); alpha 0.001 (the example's value) or threshold factors 0.05-0.65 still pass because the "
    "outputs stay within the stated behaviour. Flag positions are compared sample by sample over the whole "
    "192-period window (15360 samples). Known oracle limits: (1) a deliberate cheat that writes "
    "peaks.dat from a Vector Source pattern while the detector still feeds average.dat would pass (the path graph "
    "check only requires the detector between the sawtooth and some File Sink); (2) the 'saw' point is an edge "
    "point on the block named analog_sig_source_x_0, so renaming the Signal Source fails V5. Licence: derived "
    "from the GNU Radio 3.10.12 example blocks/peak_detector2.grc (GPL-3.0-or-later, GNU Radio project); the "
    "licence decision is the team's."
)

record = {
    "id": TASK_ID, "tier": "F", "split": SPLIT, "version": 2, "canary": CANARY,
    "license": f"GPL-3.0-or-later (derived from GNU Radio 3.10.12 example {SEED})",
    "source": {"kind": "gnuradio_example", "ref": SEED, "sha256": seed_sha, "derivation": DERIVATION},
    "cluster_id": SEED,
    "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring",
                   "generator_version": "0.2.0"},
    "review": {"status": "pending-team-review", "reviewers": [], "notes": REVIEW_NOTES},
    "family": "repair", "level": "L1",
    "instruction": instruction,
    "start_grc": "start.grc", "reference_grc": "reference.grc",
    "observation": observation,
    "oracle": {"checks": checks, "graph_checks": graph_checks},
    "mutations": mutations,
    "spec_properties": spec_properties,
    "perturbation": PERTURBATION,
    "fault": {"type": "type_mismatch",
              "description": ("The Char To Float converter after Peak Detector2 is replaced by Short To Float "
                              "(block 'blocks_short_to_float_0'). Peak Detector2's output 0 is byte (char), Short "
                              "To Float's input is short, so GRC validation reports 'Source IO type \"byte\" does "
                              "not match sink IO type \"short\"' and 'Source IO size \"1\" does not match sink IO "
                              "size \"2\"' (V3). Fix: a byte-to-float converter (Char To Float or UChar To Float) "
                              "with unit scale, or a Char To Short in front of the Short To Float with scale 256."),
              "expected_rung": "V3", "block": "blocks_short_to_float_0", "wrong_id": WRONG,
              "correct_id": "blocks_char_to_float"},
    "disjointness": ("Repair of a byte-vs-short converter mismatch after a Peak Detector2 on a sawtooth, from the "
                     "test-split seed blocks/peak_detector2.grc; no training-side spec does peak detection (they "
                     "are dial tone, QPSK BER, FM/WBFM receiver repair, unconnected-sink repair, the QAM "
                     "types/rates debug, resampler repair, AM demod, spectrogram sweep, GMSK, PSK31, RSSI, "
                     "Embedded-Python blocks and GRC-Agent dial-tone scenarios). The nearest is the QAM types/rates "
                     "debug spec, which fixes type mismatches in a different graph with a different function."),
    "notes": ("Measured on the reference (deterministic, no noise): flags at samples 39 + 80k, 192 in the window; "
              "running average 0.9806-1.0107 from 0.1 s on; sawtooth 0.25-1.73125, 2nd-harmonic share 0.152318. "
              "No band_fraction threshold depends on a DC level (remove_mean is set), so the 18 Oct band_fraction "
              "correction (R12) does not apply. Distinct from gbf-test-07 (different group); the 0/1 indicator "
              "from a ramp stays unique to this item."),
}

files = {"reference.grc": grcfile.dump(ref), "start.grc": grcfile.dump(start)}
files.update({f"variants/{k}": grcfile.dump(v) for k, v in variants.items()})

if __name__ == "__main__":
    errors = taskio.save(HERE, record, files)
    if errors:
        raise SystemExit("schema errors:\n" + "\n".join(errors))
    print(f"written {HERE} (seed {SEED}, split {seed_split['split']}, sha256 {seed_sha[:12]})")
