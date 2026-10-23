"""Authoring script for GRC-Bench Tier-F item gbf-test-17 (repair, L2, fault: undefined_variable).

Deterministic: re-running it rewrites task.json, reference.grc, start.grc and variants/*.grc
byte-identically (and removes variant files that are no longer part of the item). Run inside WSL
from the repository root:

    PYTHONPATH=src ~/grc-agent/.venv/bin/python items/tier_f/test/gbf-test-17/build.py
then run the gates:
    ./gb tier-f gates items/tier_f/test/gbf-test-17

Seed: the GNU Radio 3.10.12 example digital/demod/pam_timing.grc (a test-split seed in
splits/flowgraph_seed_split.json). Random QPSK symbols (Random Source -> Chunks to Symbols) are
pulse-shaped by a Polyphase Arbitrary Resampler with root-raised-cosine (RRC) taps, pass a Channel
Model (noise, frequency offset, sample-clock offset) and a Throttle, and a Polyphase Clock Sync with
the matching RRC filter bank recovers one sample per symbol. In the example one variable `spb`
(samples per symbol) feeds both the transmitter's resampling rate and the receiver.

This item exports the seed headless with `./gb tier-f seed` (the five Qt time sinks become File
Sinks, renamed here to tx_signal.dat, rx_symbols.dat, timing_error.dat, timing_rate.dat and
timing_phase.dat), gives the transmitter its own symbol-rate variable (resampling rate
samp_rate/sym_rate) so that `spb` is the receiver's samples per symbol only, and perturbs every
number (see PERTURBATION).

The fault: the variable block `spb` is deleted, while the Polyphase Clock Sync's Samples/Symbol and
Taps still refer to it, so GRC validation fails (V3) with "name 'spb' is not defined". The repair is
to define spb = samp_rate/sym_rate = 5 (48 kHz / 9.6 kBd). Restoring the example's 4.2563, or any
value outside about 4.95-5.1, leaves the receiver mismatched to the transmitter beyond what the
timing loop can follow (maximum rate deviation 0.5): it never locks, which the oracle detects.
Drafted by Claude (Anthropic); pending team review.

Round-1 review fixes (18 Oct 2026; reports/tier_f_reviews/round1/gbf-test-17.resolution.json):
  * F1 "keep" sentence checked: a second observation point on tx_signal.dat with a phase-sensitive
    correlation against the exact transmitted waveform (this also makes the 9.6 kBd symbol rate and
    the 0.5 roll-off measurable), graph checks for the five File Sinks (file name and item type),
    for the Channel Model's parameters and the values 0.998 / 0.05, and a path check that the
    received signal reaches a File Sink through a timing-recovery block (kills the "no clock sync,
    no clock offset" shortcut).
  * F2 phase-sensitive symbol check: corr_ref against the complex transmitted symbols in signed mode
    (tooling T2), so a constant rotation (90 or 45 degrees) or an I/Q mirror (conjugate) fails.
  * F3 the receiver's 32 filters / loop bandwidth / maximum deviation are no longer stated as
    requirements (they are visible in the flowgraph); the 64-filter variant stays valid, and the
    reviewers' 32-filter alternatives were added as variants.
  * Compliance F1/F2: the survivable mutants were added; the recorded claims were corrected.

Design notes (measured while authoring):
  * Observation points. Until 18 Oct the harness tapped every point with Head + Vector Sink, and a
    full tap at the 48 kHz transmitter stalled the shared chain before the 5x slower symbol tap
    filled. Since tooling T7 the taps never back-pressure the graph, so tx_signal.dat (48 kHz) and
    rx_symbols.dat (9.6 kHz) are observed together.
  * The receiver runs at an integer 5 samples per symbol on purpose. With a non-integer value
    (4.8, or the example's 4.2563) the GNU Radio 3.10.12 Polyphase Clock Sync output is not
    reproducible from run to run (it reads past the input it was guaranteed, apparently because its
    forecast uses floor(sps)). With 5 samples/symbol 8 of 8 runs are bit-identical, with or without
    the Throttle.
  * The clock offset is 0.998 (the receiver sees 5.01 samples per symbol): an answer that
    "compensates" the offset writes spb = 5.01, which stays in the deterministic floor(sps) = 5
    regime (24/24 identical passes; variant v6). An answer that applies the offset in the wrong
    direction writes 4.99, which falls into the floor(sps) = 4 regime and is not reproducible: it
    passed 20 of 24 runs on 18 Oct and 21 of 24 in a G8 re-run (four different outputs; the round-1
    review had seen 64 of 64 before tooling T7 changed the observation taps, which changes the
    scheduling). It is not the matched value the instruction
    asks for, so it is not a variant; review.notes records it.
  * The clock sync's maximum rate deviation is 0.5 (default 1.5). With 1.5 the loop also followed a
    receiver set to 4.8 samples/symbol (-4 %), and that wrong answer then passed or failed depending
    on the run. With 0.5, 4.8, 4.85, 4.9 and 5.2 fail in every run and 4.95-5.1, 64 filters and loop
    bandwidths 0.03-0.1 pass in every run, each with bit-identical outputs.
  * GNU Radio 3.10.12's QPSK constellation (digital.qpsk_constellation(), the example's `const`)
    has the points +-1.414 +- 1.414j, i.e. magnitude 2; the instruction says so and the RMS check
    expects 2.0.
  * The worker seeds numpy with 0 before it builds the flowgraph, so the Random Source
    (numpy.random.randint) emits a fixed symbol sequence; the reference sequences below reproduce it
    (the transmitted waveform by running the transmitter blocks in-process, exactly as generated).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # .../grc-bench
sys.path.insert(0, str(ROOT / "src"))

from grcbench.common.canary import CANARY                         # noqa: E402
from grcbench.common.io import sha256_file                        # noqa: E402
from grcbench.common.paths import GNURADIO_EXAMPLES, SPLITS        # noqa: E402
from grcbench.tier_f import edit, grc, grcfile, taskio             # noqa: E402
from grcbench.tier_f.seedtools import headless_seed               # noqa: E402

TASK_ID = "gbf-test-17"
SPLIT = "test"
SEED = "digital/demod/pam_timing.grc"

# ---------------------------------------------------------------- numbers (perturbed)
FS = 48000                 # sample rate (example: 32000)
SYM_RATE = 9600            # transmitter symbol rate in Bd (example: implicit, 32000/4.2563 = 7518 Bd)
SPS = FS // SYM_RATE       # 5 samples per symbol at the receiver (example: spb = 4.2563)
assert FS / SYM_RATE == SPS == 5
ROLLOFF = "0.5"            # RRC roll-off of transmitter and receiver (example: .35)
TIME_BW = "0.0628"         # timing-loop bandwidth (example: 0, a Qt range)
MAX_DEV = "0.5"            # clock sync maximum rate deviation (example: 1.5, the default)
INTERPRATIO = "0.998"      # channel sample-clock offset: 0.2 % more samples per symbol at the receiver (example: 1)
NOISE = "0.05"             # channel noise voltage (example: 0, a Qt range)
NUM_SAMPS = "100000"       # Random Source length before it repeats (example: 10000000)
NFILTS = 32
QPSK_MAG = 2.0             # |point| of GNU Radio 3.10.12's QPSK constellation (+-1.414 +- 1.414j)

OUT_FILE = "rx_symbols.dat"
TX_FILE = "tx_signal.dat"

# ---------------------------------------------------------------- block names
CLOCK = "digital_pfb_clock_sync_xxx_0"
TXRS = "pfb_arb_resampler_xxx_0"
CHAN = "channels_channel_model_0"
THR = "blocks_throttle2_0"
RAND = "analog_random_source_x"
SYMS = "digital_chunks_to_symbols_xx"
RX_SINK = "rx_symbols_sink"
TX_SINK = "tx_signal_sink"
SINKS = {  # seed tool's File Sink name -> (new name, file, item type): what each Qt time sink showed
    "qtgui_time_sink_x_0_0": (TX_SINK, TX_FILE, "complex"),                     # "Resampled Signal" (transmitter)
    "qtgui_time_sink_x_0_0_0": (RX_SINK, OUT_FILE, "complex"),                  # "Scope Plot" (recovered symbols)
    "qtgui_time_sink_x_1_0": ("timing_error_sink", "timing_error.dat", "float"),  # "Error"
    "qtgui_time_sink_x_1_0_0": ("timing_rate_sink", "timing_rate.dat", "float"),  # "Rate"
    "qtgui_time_sink_x_1_1_0": ("timing_phase_sink", "timing_phase.dat", "float"),  # "Phase"
}
# every block that recovers symbol timing (the receiver may legitimately use any of them)
TIMING_IDS = ["digital_pfb_clock_sync_xxx", "digital_symbol_sync_xx", "digital_clock_recovery_mm_xx",
              "digital_msk_timing_recovery_cc"]

# ---------------------------------------------------------------- the seed
seed_path = GNURADIO_EXAMPLES / SEED
seed_sha = sha256_file(seed_path)
seed_split = json.loads((SPLITS / "flowgraph_seed_split.json").read_text())["seeds"][SEED]
assert seed_split["split"] == SPLIT, seed_split
assert seed_split["sha256"] == seed_sha, (seed_split["sha256"], seed_sha)
orig = {b["name"]: b for b in yaml.safe_load(seed_path.read_text(encoding="utf-8"))["blocks"]}
ORIG = {k: str(orig[k]["parameters"]["value"]) for k in
        ("samp_rate", "spb", "rolloff", "time_bw", "interpratio", "noise_amp")}
ORIG["num_samps"] = str(orig[RAND]["parameters"]["num_samps"])
ORIG["rrate"] = str(orig[TXRS]["parameters"]["rrate"])
ORIG["max_dev"] = str(orig[CLOCK]["parameters"]["max_dev"])

PERTURBATION = {"perturbed": True, "changes": [
    {"what": "sample rate (variable samp_rate)", "original": ORIG["samp_rate"], "new": str(FS)},
    {"what": "transmitter symbol rate (new variable sym_rate; the example has none: 32000/4.2563 = 7518 Bd)",
     "original": "32000/4.2563 (about 7518 Bd)", "new": str(SYM_RATE)},
    {"what": "transmitter resampling rate (Polyphase Arbitrary Resampler rrate)", "original": ORIG["rrate"],
     "new": "samp_rate/sym_rate (= 5)"},
    {"what": "receiver samples per symbol (variable spb; deleted in the starting flowgraph)",
     "original": ORIG["spb"], "new": "samp_rate/sym_rate (= 5)"},
    {"what": "RRC roll-off (variable rolloff)", "original": ORIG["rolloff"], "new": ROLLOFF},
    {"what": "timing-loop bandwidth (variable time_bw; a Qt range in the example)", "original": ORIG["time_bw"],
     "new": TIME_BW},
    {"what": "Polyphase Clock Sync maximum rate deviation (max_dev)", "original": ORIG["max_dev"], "new": MAX_DEV},
    {"what": "channel sample-clock offset (variable interpratio; a Qt range in the example)",
     "original": ORIG["interpratio"], "new": INTERPRATIO},
    {"what": "channel noise voltage (variable noise_amp; a Qt range in the example)", "original": ORIG["noise_amp"],
     "new": NOISE},
    {"what": "Random Source length (num_samps)", "original": ORIG["num_samps"], "new": NUM_SAMPS},
    {"what": "the five Qt time sinks became File Sinks (tx_signal.dat, rx_symbols.dat, timing_error.dat, "
             "timing_rate.dat, timing_phase.dat); Qt widget variables became plain variables; the flowgraph "
             "options' author/title/GUI settings were dropped (generate_options no_gui)",
     "original": "Qt GUI sinks and range widgets", "new": "File Sinks and plain variables"},
]}
DERIVATION = ("Headless export by `./gb tier-f seed`: the five Qt time sinks became File Sinks, renamed tx_signal_sink "
              "(tx_signal.dat), rx_symbols_sink (rx_symbols.dat), timing_error_sink (timing_error.dat), "
              "timing_rate_sink (timing_rate.dat) and timing_phase_sink (timing_phase.dat); Qt widget variables became "
              "plain variables; a variable sym_rate was added and the transmitter's resampling rate set to "
              "samp_rate/sym_rate, so that spb is the receiver's samples per symbol only; empty housekeeping parameters "
              "dropped.")


def tidy(data: dict) -> dict:
    """Compact form: no `states`, no empty housekeeping parameters, no catalog-default parameters."""
    d = copy.deepcopy(data)
    for b in d["blocks"]:
        b.pop("states", None)
        b["parameters"] = {k: v for k, v in (b.get("parameters") or {}).items() if str(v) != ""}
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in d["blocks"]}
    out = grcfile.compact(d, defaults)
    out["options"] = d["options"]
    diffs = grc.compact_equivalent(d, out)
    assert not diffs, diffs
    return out


def blk(data: dict, name: str) -> dict:
    return next(b for b in data["blocks"] if b["name"] == name)


def rename(data: dict, old: str, new: str) -> dict:
    d = copy.deepcopy(data)
    blk(d, old)["name"] = new
    d["connections"] = [[new if x == old else x for x in c] for c in d["connections"]]
    return d


def without(data: dict, names: set[str]) -> dict:
    d = copy.deepcopy(data)
    d["blocks"] = [b for b in d["blocks"] if b["name"] not in names]
    d["connections"] = [c for c in d["connections"] if c[0] not in names and c[2] not in names]
    return d


def build_reference() -> dict:
    seed, sinks = headless_seed(SEED)
    assert sorted(s["name"] for s in sinks) == sorted(SINKS), sinks
    d = copy.deepcopy(seed)
    d["options"]["parameters"] = {"id": "pam_timing", "generate_options": "no_gui", "output_language": "python"}
    for old, (new, fname, typ) in SINKS.items():
        d = rename(d, old, new)
        blk(d, new)["parameters"]["file"] = fname
        assert blk(d, new)["parameters"]["type"] == typ, (new, blk(d, new)["parameters"])
    i = next(k for k, b in enumerate(d["blocks"]) if b["name"] == "samp_rate")
    d["blocks"].insert(i + 1, {"name": "sym_rate", "id": "variable", "parameters": {"value": str(SYM_RATE)}})
    for var, val in {"samp_rate": FS, "spb": "samp_rate/sym_rate", "rolloff": ROLLOFF, "time_bw": TIME_BW,
                     "interpratio": INTERPRATIO, "noise_amp": NOISE}.items():
        blk(d, var)["parameters"]["value"] = str(val)
    blk(d, TXRS)["parameters"]["rrate"] = "samp_rate/sym_rate"
    blk(d, RAND)["parameters"]["num_samps"] = NUM_SAMPS
    blk(d, CLOCK)["parameters"]["max_dev"] = MAX_DEV
    # the receiver is the only user of spb; the transmitter's taps never used it
    assert blk(d, CLOCK)["parameters"]["sps"] == "spb"
    assert "spb" in blk(d, CLOCK)["parameters"]["taps"] and "spb" not in blk(d, TXRS)["parameters"]["taps"]
    return tidy(d)


ref = build_reference()
ids = grc.block_ids()
assert all(b["id"] in ids for b in ref["blocks"])
functional = [b for b in ref["blocks"] if not b["id"].startswith("variable") and b["id"] != "import"]
assert len(functional) == 11, [b["id"] for b in functional]   # L2: 6-15 functional blocks; rate change, modulation, filtering
users_of_spb = sorted(b["name"] for b in ref["blocks"] if b["name"] != "spb" and any(
    "spb" in str(v) for v in b["parameters"].values()))
assert users_of_spb == [CLOCK], users_of_spb
CHAN_PARAMS = {k: blk(ref, CHAN)["parameters"][k] for k in ("noise_voltage", "epsilon", "freq_offset")}
assert CHAN_PARAMS == {"noise_voltage": "noise_amp", "epsilon": "interpratio", "freq_offset": "freq_offset"}, CHAN_PARAMS
assert sum(b["id"] == "variable" and b["parameters"].get("value") in (INTERPRATIO, NOISE) for b in ref["blocks"]) == 2

# ---------------------------------------------------------------- starting flowgraph (the fault)
start = without(ref, {"spb"})

# ---------------------------------------------------------------- equivalent variants
# v1: spb as the literal 5 instead of an expression.
v1 = edit.set_param(ref, "spb", "value", str(SPS))

# v2: no spb variable at all: the clock sync's Samples/Symbol and taps written with samp_rate/sym_rate.
v2 = without(ref, {"spb"})
v2 = edit.set_param(v2, CLOCK, "sps", "samp_rate/sym_rate")
v2 = edit.set_param(v2, CLOCK, "taps", "firdes.root_raised_cosine(nfilts, nfilts*samp_rate/sym_rate, 1.0, rolloff, 44*nfilts)")

# v3: a finer receiver filter bank (64 filters, its own RRC prototype of 44 symbols), spb = 48000/9600.
# Only the clock sync's own filter count changes (the shared nfilts also sets the transmitter's taps).
v3 = edit.set_param(ref, "spb", "value", "48000/9600")
v3 = edit.set_param(v3, CLOCK, "filter_size", "64")
v3 = edit.set_param(v3, CLOCK, "taps", "firdes.root_raised_cosine(64, 64*spb, 1.0, rolloff, 64*44)")

# v4: spb = samp_rate/sym_rate and the Throttle removed (it only paces the simulation).
v4 = edit.bypass(ref, THR)

# v5 (round-1 review): the textbook taps form, 11 symbols per filter, length rounded to an integer.
v5 = edit.set_param(ref, CLOCK, "taps", "firdes.root_raised_cosine(nfilts, nfilts, 1.0/float(spb), rolloff, int(11*spb*nfilts))")

# v6 (round-1 review): spb compensates the channel's clock offset, samp_rate/sym_rate/interpratio = 5.01.
v6 = edit.set_param(ref, "spb", "value", "samp_rate/sym_rate/interpratio")

# Not a variant: spb = samp_rate/sym_rate*interpratio = 4.99 (the offset applied in the wrong direction).
# Below an integer samples/symbol the 3.10.12 clock sync is not reproducible: 4.99 passed 20 and 21 of 24 runs,
# 4.97 19 of 24 and 4.95 13 of 24 (8 in parallel, 18 Oct), while 5, 5.01, 5.1 and the 64-filter bank
# were bit-identical in 24 of 24. See review.notes.

# v7 (round-1 review): spb as an integer expression and the catalog-default initial filter phase nfilts/2.
v7 = edit.set_param(ref, "spb", "value", "int(samp_rate/sym_rate)")
v7 = edit.set_param(v7, CLOCK, "init_phase", "nfilts/2")

variants = {"v1_literal_spb.grc": v1, "v2_inline_rx_rate.grc": v2, "v3_64_filter_bank.grc": v3,
            "v4_no_throttle.grc": v4, "v5_textbook_taps.grc": v5, "v6_spb_compensates_offset.grc": v6,
            "v7_int_spb_init_phase.grc": v7}
variants = {k: tidy(v) for k, v in variants.items()}

# ---------------------------------------------------------------- instruction
RUN_SENTENCE = "The flowgraph should keep producing samples until it is stopped; do not limit the number of samples."
instruction = (
    "This flowgraph simulates a QPSK link to demonstrate symbol timing recovery. It runs at a sample rate of 48 kHz. "
    "The transmitter takes random QPSK symbols (GNU Radio's QPSK constellation, points ±1.414 ± 1.414j, i.e. "
    "magnitude 2) and pulse-shapes them with root-raised-cosine filtering (roll-off 0.5) in a polyphase arbitrary "
    "resampler, at a symbol rate of 9.6 kBd; its output is written as complex items to the File Sink whose file name "
    "is tx_signal.dat. A channel model adds Gaussian noise (noise voltage 0.05) and a sample-clock offset of 0.2 % "
    "(timing offset 0.998, so the receiver sees about 0.2 % more samples per symbol than nominal), with no frequency "
    "offset. The receiver is a Polyphase Clock Sync whose bank of root-raised-cosine filters is matched to the "
    "transmitter's pulse and symbol rate. It should lock onto the symbol timing and output one sample per symbol, "
    "written as complex items to the File Sink whose file name is rx_symbols.dat; its timing error, rate and phase "
    "outputs go to the File Sinks timing_error.dat, timing_rate.dat and timing_phase.dat (float items).\n\n"
    "Once the timing loop has settled, that is from the 2,049th recovered symbol on, the recovered symbols must be the "
    "transmitted symbol sequence, in order, without slipped or extra symbols and without any rotation or I/Q "
    "mirroring of the constellation (a fixed delay is fine): at the best fixed delay, their normalised correlation "
    "with the transmitted complex QPSK symbols must be at least 0.95. Each recovered symbol must sit on its QPSK "
    "point: none with a magnitude above 2.5, and an RMS magnitude of 2.0 (within 5 %). " + RUN_SENTENCE + "\n\n"
    "At the moment GNU Radio Companion refuses to generate the flowgraph: validation reports that two parameters "
    "of the Polyphase Clock Sync block (Samples/Symbol and Taps) cannot be evaluated. Please repair the "
    "flowgraph so that it works as described. Keep the transmitter (the signal written to tx_signal.dat must stay "
    "exactly as it is), the channel model and all five File Sinks as they are."
)

# ---------------------------------------------------------------- reference sequences
# The worker seeds numpy with 0 before it builds the top block; the Random Source is
# vector_source_b(list(numpy.random.randint(0, const.arity(), num_samps)), repeat=True).
from gnuradio import blocks, digital, gr  # noqa: E402
from gnuradio.filter import firdes, pfb  # noqa: E402

POINTS = np.array(digital.qpsk_constellation().points())
assert np.allclose(np.abs(POINTS), QPSK_MAG), POINTS
np.random.seed(0)
SYM_IDX = np.random.randint(0, len(POINTS), int(NUM_SAMPS))
TX_SYMBOLS = POINTS[SYM_IDX]
N_ITEMS = 8192             # items captured at each point
SKIP = 2048                # timing-loop acquisition (the reference locks within about 1,000 symbols)
LAG = 256                  # delay search range of corr_ref
PRE = 64                   # the reference window starts 64 symbols late, so delays D in [-64, 192] are found
NC = 4096                  # symbols compared (2048 .. 6400 of the capture)
# capture[SKIP + i] holds transmitted symbol SKIP + i - D (D = 28 for the reference); corr_ref aligns
# ref[0] = tx[SKIP + PRE] with capture[SKIP + lag], i.e. lag = D + PRE.
REF_SYM = TX_SYMBOLS[SKIP + PRE:SKIP + PRE + NC + LAG]
REF_RE = [round(float(v), 6) for v in REF_SYM.real]
REF_IM = [round(float(v), 6) for v in REF_SYM.imag]
assert len(REF_RE) == NC + LAG


def transmitted_waveform(n: int) -> np.ndarray:
    """The transmitter's output, built exactly as GRC generates it (pfb_arb_resampler_xxx: flt_size=32,
    atten=100; chunks_to_symbols_bc(const.points(), 1); the Random Source's seeded symbols)."""
    tb = gr.top_block()
    src = blocks.vector_source_b(list(map(int, SYM_IDX)), True)
    c2s = digital.chunks_to_symbols_bc(digital.qpsk_constellation().points(), 1)
    rs = pfb.arb_resampler_ccf(FS / SYM_RATE, taps=firdes.root_raised_cosine(NFILTS, NFILTS, 1.0, float(ROLLOFF),
                                                                              44 * NFILTS), flt_size=32, atten=100)
    head = blocks.head(gr.sizeof_gr_complex, n)
    snk = blocks.vector_sink_c()
    tb.connect(src, c2s, rs, head, snk)
    tb.run()
    return np.array(snk.data())


TX_SKIP = 1024             # past the resampler's start-up ramp
TX_N = 2048                # transmitted samples compared (about 410 symbols)
TX_LAG = 4
TXW = transmitted_waveform(TX_SKIP + TX_N + TX_LAG)[TX_SKIP:]
TX_RE = [round(float(v), 6) for v in TXW.real]
TX_IM = [round(float(v), 6) for v in TXW.imag]
assert len(TX_RE) == TX_N + TX_LAG and np.std(TXW) > 0.1

observation = {
    "points": [{"name": "symbols", "sink": {"block_id": "blocks_file_sink",
                                            "params": {"file": OUT_FILE, "type": "complex"}}},
               {"name": "tx", "sink": {"block_id": "blocks_file_sink",
                                       "params": {"file": TX_FILE, "type": "complex"}}}],
    "run": {"n_items": N_ITEMS, "min_items": N_ITEMS, "timeout_s": 10.0},
    "contract": ("the complex samples written to the File Sink whose file name is rx_symbols.dat (the recovered "
                 f"symbols, one per symbol), from the {SKIP + 1:,}th symbol on, and the complex samples written to the "
                 "File Sink whose file name is tx_signal.dat (the transmitter's output)"),
}
checks = [
    # f2p: the receiver works again (matched samples per symbol): locked, in order, no slips, no rotation
    {"point": "symbols", "feature": "corr_ref", "ref": REF_RE, "ref_imag": REF_IM, "signed": True,
     "max_lag": LAG, "skip": SKIP, "n": NC + LAG, "min": 0.95, "group": "f2p", "property": "symbol_sequence",
     "note": ("phase-sensitive normalised correlation Re<ref, x>/(|ref||x|) of 4,096 recovered symbols with the "
              "transmitted complex QPSK symbols at the best fixed delay (-64..192 symbols): reference 0.9994; a "
              "constant rotation by theta scores cos(theta) (90 degrees about 0, 45 degrees 0.71), an I/Q mirror "
              "about 0.04, a slip, a drifting rotation or another symbol sequence below 0.7")},
    {"point": "symbols", "feature": "max_abs", "skip": SKIP, "max": 2.5, "group": "f2p", "property": "timing_lock",
     "note": ("no recovered symbol more than 25 % above the constellation magnitude 2 (reference 2.09; an unlocked "
              "or mismatched receiver 2.8-3.0, from inter-symbol interference)")},
    # p2p: properties carried by the parts of the graph the fault does not touch
    {"point": "symbols", "feature": "rms", "skip": SKIP, "expect": QPSK_MAG, "rel_tol": 0.05, "group": "p2p",
     "property": "constellation_level",
     "note": "QPSK points of magnitude 2 through unity-gain RRC filters (reference 1.98)"},
    {"point": "tx", "feature": "corr_ref", "ref": TX_RE, "ref_imag": TX_IM, "signed": True, "max_lag": TX_LAG,
     "skip": TX_SKIP, "n": TX_N + TX_LAG, "min": 0.999, "group": "p2p", "property": "transmitter",
     "note": ("the transmitter is unchanged: phase-sensitive correlation of 2,048 samples of tx_signal.dat with the "
              "transmitted waveform (computed in build.py from the same blocks and the worker's numpy seed); "
              "reference 1.0. Another symbol rate, roll-off, constellation or symbol sequence scores well below")},
]
graph_checks = (
    [{"kind": "block_present", "ids": ["blocks_file_sink"], "params": {"file": f, "type": t}, "group": "p2p",
      "property": "file_sinks", "note": f"the File Sink {f} ({t} items) is kept"}
     for (_, f, t) in SINKS.values()]
    + [{"kind": "path", "from_ids": ["channels_channel_model"], "to_ids": ["blocks_file_sink"], "via_ids": TIMING_IDS,
        "group": "p2p", "property": "timing_recovery",
        "note": ("the received signal reaches a File Sink through a timing-recovery block (any of them); rules out "
                 "replacing the receiver by a fixed decimating matched filter")},
       {"kind": "block_present", "ids": ["channels_channel_model"], "params": CHAN_PARAMS, "group": "p2p",
        "property": "channel", "note": "the Channel Model still takes its noise, timing and frequency offsets from "
                                       "the variables noise_amp, interpratio and freq_offset"},
       {"kind": "block_present", "ids": ["variable"], "params": {"value": INTERPRATIO}, "group": "p2p",
        "property": "channel", "note": "a variable holds the timing offset 0.998 (interpratio)"},
       {"kind": "block_present", "ids": ["variable"], "params": {"value": NOISE}, "group": "p2p",
        "property": "channel", "note": "a variable holds the noise voltage 0.05 (noise_amp)"}]
)

spec_properties = [
    {"id": "loads", "text": "the flowgraph validates (every parameter evaluates), generates and runs",
     "checked_by": "ladder V3-V4"},
    {"id": "run", "text": "the flowgraph keeps producing samples until it is stopped (no sample limit)",
     "checked_by": "V4 liveness: at least 8,192 items at rx_symbols.dat and at tx_signal.dat"},
    {"id": "output", "text": "the recovered symbols, one complex item per symbol, are written to the File Sink "
                             "rx_symbols.dat",
     "checked_by": "observation contract (file name and item type complex); oversampled output or a wrong tap fails "
                   "the f2p checks"},
    {"id": "rx_sps", "text": "the Polyphase Clock Sync is matched to the transmitter: 48 kHz / 9.6 kBd = 5 samples per "
                             "symbol",
     "checked_by": ("f2p corr_ref and max_abs: a receiver outside about 4.95-5.1 samples/symbol cannot follow the "
                    "transmitter within the loop's maximum rate deviation of 0.5 and never locks (4.8, 4.9, 5.2 and the "
                    "example's 4.2563 fail in every run)")},
    {"id": "timing_lock", "text": "from the 2,049th recovered symbol on every symbol sits on its QPSK point: no "
                                  "magnitude above 2.5",
     "checked_by": "f2p max_abs"},
    {"id": "symbol_sequence", "text": "the recovered symbols are the transmitted sequence, in order, without slips, "
                                      "extra symbols, rotation or I/Q mirroring; a fixed delay is fine; normalised "
                                      "correlation with the transmitted complex symbols at least 0.95",
     "checked_by": "f2p corr_ref, signed, complex reference, 4,096 symbols, best fixed delay in -64..192 symbols"},
    {"id": "constellation_level", "text": "QPSK constellation of magnitude 2 (GNU Radio's QPSK points ±1.414 ± "
                                          "1.414j): RMS magnitude 2.0 within 5 %",
     "checked_by": "p2p rms"},
    {"id": "transmitter", "text": "the transmitter is kept: random QPSK symbols at 9.6 kBd, RRC roll-off 0.5, 48 kHz; "
                                  "the signal written to tx_signal.dat does not change",
     "checked_by": "p2p corr_ref (signed, complex) of tx_signal.dat against the transmitted waveform, at least 0.999"},
    {"id": "channel", "text": "the channel model is kept: noise voltage 0.05, timing offset 0.998, no frequency offset",
     "checked_by": ("p2p graph checks: the Channel Model's noise_voltage / epsilon / freq_offset still refer to "
                    "noise_amp / interpratio / freq_offset, and variables with the values 0.998 and 0.05 exist; a "
                    "frequency offset or a channel rotation also fails the signed symbol check")},
    {"id": "file_sinks", "text": "all five File Sinks are kept (tx_signal.dat and rx_symbols.dat complex; "
                                 "timing_error.dat, timing_rate.dat and timing_phase.dat float)",
     "checked_by": "p2p graph checks block_present with file name and item type"},
    {"id": "timing_recovery", "text": "the receiver recovers the symbol timing (a Polyphase Clock Sync, or any other "
                                      "timing-recovery block)",
     "checked_by": "p2p graph check path Channel Model -> timing-recovery block -> File Sink"},
    {"id": "receiver_design", "text": "a Polyphase Clock Sync with root-raised-cosine filters matched to the "
                                      "transmitter (its filter count, loop bandwidth and maximum deviation are not "
                                      "requirements)",
     "checked_by": ("not separately: consistent receiver settings (32 or 64 filters, the textbook taps form, loop "
                    "bandwidth 0.03-0.1) give the same recovered symbols; variants v3, v5 and v7")},
]

mutations = [
    {"op": "set_param", "block": "spb", "param": "value", "value": ORIG["spb"], "breaks": "rx_sps",
     "note": "the example's 4.2563 samples/symbol (what a memorised answer would restore)"},
    {"op": "set_param", "block": "spb", "param": "value", "value": "4", "breaks": "rx_sps"},
    {"op": "set_param", "block": "spb", "param": "value", "value": "4.8", "breaks": "rx_sps",
     "note": "receiver set for 10 kBd (-4 %)"},
    {"op": "set_param", "block": "spb", "param": "value", "value": "4.9", "breaks": "rx_sps", "note": "-2 %"},
    {"op": "set_param", "block": "spb", "param": "value", "value": "6", "breaks": "rx_sps"},
    {"op": "scale_param", "block": "spb", "param": "value", "factor": 1.04, "breaks": "rx_sps",
     "note": "5.2 samples/symbol (+4 %)"},
    {"op": "set_param", "block": TXRS, "param": "rrate", "value": "4.8", "breaks": "transmitter",
     "note": "transmitter at 10 kBd while the receiver expects 9.6 kBd"},
    {"op": "set_param", "block": "sym_rate", "param": "value", "value": "10000", "breaks": "transmitter",
     "note": "transmitter and receiver moved together to 10 kBd (4.8 samples/symbol): the receiver matches, but the "
             "transmitter is not kept"},
    {"op": "set_param", "block": "rolloff", "param": "value", "value": "0.35", "breaks": "transmitter",
     "note": "roll-off 0.35 at transmitter and receiver (the example's value): the symbols are still recovered"},
    {"op": "set_param", "block": "nfilts", "param": "value", "value": "64", "breaks": "transmitter",
     "note": "the shared nfilts set to 64: the resampler keeps 32 arms but gets taps for 64"},
    {"op": "set_param", "block": "const", "param": "value", "value": "digital.constellation_8psk()",
     "breaks": "constellation_level", "note": "8PSK (magnitude 1) instead of QPSK"},
    {"op": "set_param", "block": "time_bw", "param": "value", "value": "0", "breaks": "timing_lock",
     "note": "timing loop off (the example's slider position)"},
    {"op": "set_param", "block": "freq_offset", "param": "value", "value": "0.0001", "breaks": "symbol_sequence",
     "note": "4.8 Hz frequency offset: the constellation rotates"},
    {"op": "set_param", "block": CHAN, "param": "taps", "value": "1j", "breaks": "symbol_sequence",
     "note": "channel rotates the constellation by 90 degrees (every symbol on the wrong QPSK point)"},
    {"op": "set_param", "block": CHAN, "param": "taps", "value": "(0.7071067811865476+0.7071067811865476j)",
     "breaks": "symbol_sequence", "note": "constellation rotated by 45 degrees (symbols between QPSK points)"},
    {"op": "set_param", "block": CHAN, "param": "taps", "value": "0.5", "breaks": "constellation_level",
     "note": "channel gain 0.5"},
    {"op": "set_param", "block": "noise_amp", "param": "value", "value": "1.0", "breaks": "channel",
     "note": "noise voltage 1.0 instead of 0.05"},
    {"op": "set_param", "block": "noise_amp", "param": "value", "value": "0", "breaks": "channel",
     "note": "noise removed"},
    {"op": "set_param", "block": "interpratio", "param": "value", "value": "1.05", "breaks": "channel",
     "note": "5 % clock offset (1.05) instead of 0.2 % (0.998)"},
    {"op": "set_param", "block": "interpratio", "param": "value", "value": "1", "breaks": "channel",
     "note": "clock offset removed"},
    {"op": "set_param", "block": "interpratio", "param": "value", "value": "1.002", "breaks": "channel",
     "note": "clock offset in the other direction"},
    {"op": "set_param", "block": "timing_rate_sink", "param": "file", "value": "rate.dat", "breaks": "file_sinks",
     "note": ("the timing-rate File Sink writes rate.dat instead of timing_rate.dat. (Deleting a single timing sink "
              "is not a valid mutant: pfb_clock_sync_ccf accepts 1 or 4 connected outputs and fails at V4 otherwise; "
              "deleting all three is rejected by the File Sink checks, see the round-1 probe del_4_other_sinks.)")},
    {"op": "bypass", "block": TX_SINK, "breaks": "file_sinks",
     "note": "the tx_signal.dat File Sink deleted"},
    {"op": "rewire_input", "dst": RX_SINK, "dst_port": "0", "new_src": THR, "breaks": "output",
     "note": "wrong tap: the received 48 kHz samples instead of the recovered symbols"},
    {"op": "set_param", "block": CLOCK, "param": "osps", "value": "2", "breaks": "output",
     "note": "two output samples per symbol"},
]

record = {
    "id": TASK_ID, "tier": "F", "split": SPLIT, "version": 2, "canary": CANARY,
    "license": f"GPL-3.0-or-later (derived from GNU Radio 3.10.12 example {SEED})",
    "source": {"kind": "gnuradio_example", "ref": SEED, "sha256": seed_sha, "derivation": DERIVATION},
    "cluster_id": SEED,
    "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring",
                   "generator_version": "0.2.0"},
    "review": {"status": "pending-team-review", "reviewers": [], "notes": (
        "Round-1 review applied 18 Oct 2026 (see reports/tier_f_reviews/round1/gbf-test-17.resolution.json). Known "
        "limits: (1) the channel 'keep' is checked at string level: the Channel Model must still refer to the variables "
        "noise_amp / interpratio / freq_offset and variables with the values 0.998 and 0.05 must exist; an answer that "
        "rewrites these as equal literals would be rejected, and the checks do not tie a value to a variable name. The "
        "exact noise and clock-offset values cannot be measured at the outputs (the receiver hides them). (2) The File "
        "Sink checks pin file name and item type, not which output feeds each sink. (3) The path check accepts any "
        "timing-recovery block (Polyphase Clock Sync, Symbol Sync, M&M, MSK timing). (4) GNU Radio 3.10.12's Polyphase "
        "Clock Sync is not reproducible for samples/symbol just below an integer (its forecast uses floor(sps)): "
        "answers with spb in about 4.93-4.99, e.g. samp_rate/sym_rate*interpratio = 4.99 (the clock offset applied in "
        "the wrong direction), pass in only some runs (4.99: 20 and 21 of 24 in two trials, 4.97: 19 of 24, 4.95: 13 of 24, 8 in parallel). "
        "The matched value 5, the compensating 5.01 and everything up to 5.1 are bit-identical in every run. Such "
        "answers are not the matched receiver the instruction asks for; the item statistics should be read with this "
        "in mind. (5) Related item gbf-dev-10 (dev) "
        "also uses a Polyphase Clock Sync; after the round-1 fix it faults a different block (its Null Source), so "
        "the two items no longer train and test the same repair. (6) GPL-3.0-or-later attribution: derived from the "
        "GNU Radio 3.10.12 example digital/demod/pam_timing.grc (Free Software Foundation / GNU Radio project); the "
        "licence decision for seed-derived items is the team's.")},
    "family": "repair", "level": "L2",
    "instruction": instruction,
    "start_grc": "start.grc", "reference_grc": "reference.grc",
    "observation": observation,
    "oracle": {"checks": checks, "graph_checks": graph_checks},
    "mutations": mutations,
    "spec_properties": spec_properties,
    "perturbation": PERTURBATION,
    "fault": {"type": "undefined_variable",
              "description": ("the variable block spb (the receiver's samples per symbol, samp_rate/sym_rate = 5) was "
                              "deleted while the Polyphase Clock Sync's Samples/Symbol (sps: spb) and Taps "
                              "(firdes.root_raised_cosine(nfilts, nfilts*spb, 1.0, rolloff, 44*nfilts)) still refer to "
                              "it, so GRC validation fails: name 'spb' is not defined. Fix: define spb = "
                              "samp_rate/sym_rate (or 5), or write the receiver's parameters without it. Because the "
                              "start fails at V3, p2p here means the properties carried by the parts of the graph the "
                              "fault does not touch (transmitter, channel, sinks, constellation level)."),
              "expected_rung": "V3", "variable": "spb", "correct_value": "samp_rate/sym_rate (= 5)"},
    "disjointness": ("Repair of an undefined samples-per-symbol variable in the receiver of a QPSK symbol-timing-recovery "
                     "simulation (Polyphase Clock Sync), from the test-split seed digital/demod/pam_timing.grc. Nearest "
                     "training-side specs (reports/reference/training_task_specs.json): t05_qam_debug (fix sample "
                     "rates/types in a QAM demo) and t25_sync_chain_explain (explain a QAM sync/equalizer chain, "
                     "knowledge only) touch synchronisation but neither repairs an undefined variable or a timing loop; "
                     "t02 builds a QPSK BER chain from scratch, t03/t17 repair FM receivers, t04 an unconnected sink, "
                     "t20 a resampler ratio, and GRC-Agent scenario 04 adds a variable on request in the dial-tone "
                     "graph. Within GRC-Bench, gbf-dev-10 (dev) also contains a Polyphase Clock Sync but its fault is "
                     "a Null Source type mismatch in a burst BPSK transmitter (different seed, block and fault type)."),
}

files = {"reference.grc": grcfile.dump(ref), "start.grc": grcfile.dump(start)}
files.update({f"variants/{k}": grcfile.dump(v) for k, v in variants.items()})

if __name__ == "__main__":
    errors = taskio.save(HERE, record, files)
    if errors:
        raise SystemExit("schema errors:\n" + "\n".join(errors))
    for stale in sorted((HERE / "variants").glob("*.grc")):
        if f"variants/{stale.name}" not in files:
            stale.unlink()
            print(f"removed stale variant {stale.name}")
    print(f"written {HERE} (seed {SEED}, split {seed_split['split']}, sha256 {seed_sha[:12]}); "
          f"{len(variants)} variants, {len(mutations)} mutants, {len(checks)} signal and {len(graph_checks)} graph checks")
