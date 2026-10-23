"""Authoring script for GRC-Bench Tier-F item gbf-dev-10 (repair, L2, fault: type_mismatch).

Deterministic: re-running it rewrites task.json, reference.grc, start.grc and variants/*.grc
byte-identically (and removes variant files that are no longer part of the item). Run inside WSL
from the repository root:

    PYTHONPATH=src ~/grc-agent/.venv/bin/python items/tier_f/dev/gbf-dev-10/build.py
then run the gates:
    ./gb tier-f gates items/tier_f/dev/gbf-dev-10

Seed: the GNU Radio 3.10.12 example digital/demod/test_corr_est.grc (a dev-split seed in
splits/flowgraph_seed_split.json). A Vector Source repeats one packet (preamble + payload bytes); a
Constellation Modulator makes BPSK with root-raised-cosine pulses; a Stream Mux alternates one packet
with a run of zeros from a Null Source (bursts); after a Throttle a Channel Model adds impairments; a
Correlation Estimator looks for the modulated preamble and tags the burst starts; a Polyphase Clock
Sync and a Costas loop recover timing and carrier. The example shows the Costas output on a
constellation and a time display and |corr|, Re{corr}, Im{corr} on a second time display.

This item exports the seed headless with `./gb tier-f seed`, keeps the transmit/receive chain, writes
the Costas loop's symbols to symbols.dat and its frequency estimate (its second output) to freq.dat,
and drops the correlation displays (Complex To Mag / Complex To Float and their sinks). The
example's random payload (random.getrandbits) becomes a seeded numpy payload so the transmitted
symbols are known, a carrier frequency offset makes the Costas loop's work observable, and every
number is perturbed (see PERTURBATION). 11 functional blocks with BPSK modulation/demodulation,
pulse-shaping/matched filtering and a 5:1 rate change: L2.

The fault (since the round-1 review, 18 Oct 2026): the Null Source that supplies the zero samples
between the bursts has type 'float' instead of 'complex'. Its output feeds the Stream Mux's complex
input 1, so GRC validation reports an IO type and an IO size mismatch on that one connection and the
start fails at rung V3. The error names two blocks; only the Null Source can change (turning the
Stream Mux to float creates new mismatches with the complex modulator and Throttle). A Float To
Complex converter between the Null Source and the Stream Mux, or another source of complex zeros, is
an equally valid repair (variants v1, v2). Before the review the fault was a float-typed Polyphase
Clock Sync; it moved because gbf-test-17 (test) repairs that block too (review C1; fix plan G8).
Drafted by Claude (Anthropic); pending team review.

Oracle (all on the 8,192-sample captures, one sample per symbol). The transmitted stream is
continuous; the Stream Mux cuts 5,280 samples (1,056 symbol periods) of it per burst, and because
the modulator's RRC filter delays the symbols by 27 symbol periods, burst k >= 1 carries the last 27
symbols of the packet followed by the first 1,029 symbols of the next packet (review F4). On the
reference the bursts start at symbol 66 + 1856k (packet symbol 0 at 93 + 1856k); valid receivers sit
within +-62 symbols of that (the correlation windows allow +-128):
  p2p  bursts 1-3 correlate with that transmitted stream (phase-invariant, >= 0.95; valid receivers
       0.988-0.9996); the 64 preamble symbols of burst 2 correlate >= 0.95; the Costas frequency
       estimate averaged over symbols 300-1000 of bursts 1-3 is 2*pi*0.0005*5 rad/symbol +-1 % (valid
       loop bandwidths 0.01-0.5 are within 0.23 %; review F3: the value the loop holds in the gaps
       drifts up to +1.03 % with the loop bandwidth, so it is no longer measured there).
  f2p  the middle 500 of the 800 silent symbol periods after bursts 0-3 have RMS <= 0.01 (the
       repaired Null Source supplies complex zeros; the reference gives exactly 0).
  p2p  the transmitter is kept: an edge tap on the Constellation Modulator's output must equal the
       transmitted waveform (signed correlation >= 0.99999 against the waveform build.py computes with
       the same GNU Radio blocks; bit-identical to the harness's capture). Added by the G8 fixer:
       excess bandwidth 0.35 and an inverted bit mapping (bit 0 -> -1) passed every other check,
       because the receiver uses the same eb and the symbol checks are phase-invariant.
  graph (p2p) an order-2 Costas loop is present and the received signal reaches a File Sink through
       it (an order-4 loop also passed every signal check); the Channel Model still refers to
       freq_offset / time_offset / noise and a variable holds the timing offset 1 (review F2 and the
       "keep the channel" clause).
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # .../grc-bench
sys.path.insert(0, str(ROOT / "src"))

from grcbench.common.canary import CANARY                            # noqa: E402
from grcbench.common.io import sha256_file                           # noqa: E402
from grcbench.common.paths import GNURADIO_EXAMPLES, SPLITS           # noqa: E402
from grcbench.tier_f import compose as C, edit, grc, grcfile, taskio  # noqa: E402
from grcbench.tier_f.seedtools import headless_seed                  # noqa: E402

TASK_ID = "gbf-dev-10"
SPLIT = "dev"
SEED = "digital/demod/test_corr_est.grc"

# ---------------------------------------------------------------- numbers (perturbed)
FS = 250000                 # throttle rate in samples/s (example: 100000); it only paces the samples
SPS = 5                     # samples per symbol (example: 4)
EB = 0.4                    # RRC excess bandwidth (example: 0.35)
PREAMBLE = [0x2E, 0xA4, 0xAF, 0x33, 0x1F, 0x22, 0xF1, 0xF7]   # example: ac dd a4 e2 f2 8c 20 fc
# (chosen for low aperiodic autocorrelation sidelobes: peak sidelobe 7 of 64, the example's is 11)
PAYLOAD = 120               # random payload bytes (example: 992)
PAYLOAD_SEED = 7
GAP = 4000                  # zero samples between bursts (example: 20000)
FREQ_OFFSET = 0.0005        # channel carrier offset in cycles per sample (example: 0, a Qt slider)
MARK_DELAY = 4              # Correlation Estimator tag marking delay (example: 1, tuned for 4 sps)

PREAMBLE_TXT = "[" + ", ".join(f"0x{b:02x}" for b in PREAMBLE) + "]"
DATA_EXPR = f"[0]*4 + numpy.random.RandomState({PAYLOAD_SEED}).randint(0, 256, payload_size).tolist()"

# the transmitted packet as BPSK symbols (MSB first; bit 0 -> +1, bit 1 -> -1)
PACKET = PREAMBLE + [0] * 4 + np.random.RandomState(PAYLOAD_SEED).randint(0, 256, PAYLOAD).tolist()
BITS = [(b >> (7 - i)) & 1 for b in PACKET for i in range(8)]
SYMS = [1 - 2 * b for b in BITS]
N_SYM = len(SYMS)                         # 1056 symbols per burst
N_PRE = 8 * len(PREAMBLE)                 # 64 preamble symbols
GAP_SYM = GAP // SPS                      # 800 symbol periods of silence
PERIOD = N_SYM + GAP_SYM                  # 1856 symbols from burst to burst
BURST_SAMPLES = N_SYM * SPS               # 5280 samples per burst
MOD_DELAY = 27                            # RRC delay of the modulator in symbol periods (55-symbol filter)
assert (N_SYM, GAP_SYM, PERIOD, BURST_SAMPLES) == (1056, 800, 1856, 5280)
F_EST = 2 * math.pi * FREQ_OFFSET * SPS   # 0.0157080 rad/symbol: what the Costas loop must settle at
# what burst k >= 1 carries: the last 27 symbols of one packet, then the first 1,029 of the next
STREAM = SYMS[-MOD_DELAY:] + SYMS[:N_SYM - MOD_DELAY]

# ---------------------------------------------------------------- block names
VEC = "blocks_vector_source_x_0_0"
MOD = "digital_constellation_modulator_0"
NULLSRC = "blocks_null_source_0_0"
MUX = "blocks_stream_mux_0_0"
THR = "blocks_throttle2_0"
CHAN = "channels_channel_model_0"
CORR = "digital_corr_est_cc_0"
CS = "digital_pfb_clock_sync_xxx_0"
COSTAS = "digital_costas_loop_cc_0"
SYM_SINK = "symbols_sink"        # was qtgui_const_sink_x_0_0 (the seed tool's File Sink for the constellation)
FREQ_SINK = "freq_sink"
SYM_FILE = "symbols.dat"
FREQ_FILE = "freq.dat"
DROPPED = {"qtgui_time_sink_x_0_0", "qtgui_time_sink_x_1_0", "qtgui_time_sink_x_1_1", "qtgui_time_sink_x_1_2",
           "blocks_complex_to_float_0", "blocks_complex_to_mag_0", "import_0_0"}

# ---------------------------------------------------------------- the seed and its original numbers
seed_path = GNURADIO_EXAMPLES / SEED
seed_sha = sha256_file(seed_path)
seed_split = json.loads((SPLITS / "flowgraph_seed_split.json").read_text())["seeds"][SEED]
assert seed_split["split"] == SPLIT, seed_split
assert seed_split["sha256"] == seed_sha, (seed_split["sha256"], seed_sha)
orig = {b["name"]: b for b in yaml.safe_load(seed_path.read_text(encoding="utf-8"))["blocks"]}
ORIG = {k: orig[k]["parameters"]["value"] for k in ("samp_rate", "sps", "eb", "preamble", "payload_size", "data",
                                                     "gap", "freq_offset")}
ORIG["mark_delay"] = orig[CORR]["parameters"]["mark_delay"]
assert orig[NULLSRC]["parameters"]["type"] == "complex"

PERTURBATION = {"perturbed": True, "changes": [
    {"what": "throttle sample rate (variable samp_rate)", "original": ORIG["samp_rate"], "new": str(FS)},
    {"what": "samples per symbol (variable sps)", "original": ORIG["sps"], "new": str(SPS)},
    {"what": "RRC excess bandwidth (variable eb)", "original": ORIG["eb"], "new": str(EB)},
    {"what": "preamble / sync word bytes (variable preamble)", "original": ORIG["preamble"], "new": PREAMBLE_TXT},
    {"what": "payload length in bytes (variable payload_size)", "original": ORIG["payload_size"], "new": str(PAYLOAD)},
    {"what": "payload contents (variable data): unseeded random bytes became a seeded numpy sequence",
     "original": ORIG["data"], "new": DATA_EXPR},
    {"what": "zero samples between bursts (variable gap)", "original": ORIG["gap"], "new": str(GAP)},
    {"what": "channel carrier frequency offset (variable freq_offset; a Qt range widget at 0 in the example)",
     "original": "0", "new": str(FREQ_OFFSET)},
    {"what": "Correlation Estimator tag marking delay (retuned for 5 samples per symbol)",
     "original": ORIG["mark_delay"], "new": str(MARK_DELAY)},
    {"what": "headless conversion: the Qt range widgets freq_offset, noise and time_offset became plain variables "
             "(values 0.0005, 0, 1); the Qt constellation sink became the File Sink symbols.dat; the duplicate Qt "
             "time sink of the Costas output, the correlation displays (Complex To Mag, Complex To Float, three Qt "
             "time sinks) and 'import random' were dropped; a File Sink freq.dat was added on the Costas loop's "
             "frequency output; the flowgraph options became generate_options no_gui (the example's qt_gui window, "
             "run and placement options were dropped; the example has no author or copyright fields)",
     "original": "Qt GUI flowgraph", "new": "headless flowgraph with two File Sinks"},
]}
DERIVATION = ("Headless export by `./gb tier-f seed`; the Qt constellation sink's File Sink was renamed symbols_sink "
              "(symbols.dat); the duplicate Qt time sink of the Costas output, the correlation displays (Complex To Mag, "
              "Complex To Float and their three File Sinks) and the unused 'import random' were dropped; a File Sink "
              "freq_sink (freq.dat) was added on the Costas loop's frequency output; Qt widget variables became plain "
              "variables; empty housekeeping parameters dropped.")


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


def with_blocks(data: dict, new_blocks: list[dict], conns: list[list[str]]) -> dict:
    """Insert blocks just before the File Sinks and add connections."""
    d = copy.deepcopy(data)
    at = next(i for i, b in enumerate(d["blocks"]) if b["name"] in (SYM_SINK, FREQ_SINK))
    d["blocks"][at:at] = copy.deepcopy(new_blocks)
    d["connections"] += conns
    return d


def build_reference() -> dict:
    seed, sinks = headless_seed(SEED)
    assert sorted(s["name"] for s in sinks) == ["qtgui_const_sink_x_0_0", "qtgui_time_sink_x_0_0",
                                                 "qtgui_time_sink_x_1_0", "qtgui_time_sink_x_1_1",
                                                 "qtgui_time_sink_x_1_2"], sinks
    d = without(seed, DROPPED)
    d = rename(d, "qtgui_const_sink_x_0_0", SYM_SINK)
    blk(d, SYM_SINK)["parameters"]["file"] = SYM_FILE
    d["blocks"].append(C.block(FREQ_SINK, "blocks_file_sink", type="float", file=FREQ_FILE))
    d["connections"].append([COSTAS, "1", FREQ_SINK, "0"])
    for var, val in {"samp_rate": FS, "sps": SPS, "eb": EB, "preamble": PREAMBLE_TXT, "payload_size": PAYLOAD,
                     "data": DATA_EXPR, "gap": GAP, "freq_offset": FREQ_OFFSET}.items():
        blk(d, var)["parameters"]["value"] = str(val)
    blk(d, CORR)["parameters"]["mark_delay"] = str(MARK_DELAY)
    assert blk(d, CS)["parameters"]["type"] == "ccf" and blk(d, NULLSRC)["parameters"]["type"] == "complex"
    return tidy(d)


ref = build_reference()
ids = grc.block_ids()
assert all(b["id"] in ids for b in ref["blocks"])
functional = [b for b in ref["blocks"] if not b["id"].startswith("variable") and b["id"] != "import"]
assert len(functional) == 11, [b["id"] for b in functional]   # L2: 6-15 functional blocks
CHAN_PARAMS = {k: blk(ref, CHAN)["parameters"][k] for k in ("freq_offset", "epsilon", "noise_voltage")}
assert CHAN_PARAMS == {"freq_offset": "freq_offset", "epsilon": "time_offset", "noise_voltage": "noise"}, CHAN_PARAMS
assert blk(ref, "time_offset")["parameters"] == {"value": "1"}
assert [b["name"] for b in ref["blocks"] if b["id"] == "variable" and b["parameters"].get("value") == "1"] == ["time_offset"]
assert blk(ref, "noise")["parameters"].get("value", "0") == "0"            # catalog default 0: no noise

# ---------------------------------------------------------------- starting flowgraph (the fault)
start = edit.set_param(ref, NULLSRC, "type", "float")

# ---------------------------------------------------------------- equivalent variants (all repair the fault)
# v1: the Null Source stays float; a Float To Complex converter (imaginary input left open, it is optional)
# turns its zeros into complex zeros for the Stream Mux.
v1 = copy.deepcopy(start)
v1["connections"] = [c for c in v1["connections"] if c[0] != NULLSRC]
v1 = with_blocks(v1, [C.block("gap_to_complex", "blocks_float_to_complex")],
                 [[NULLSRC, "0", "gap_to_complex", "0"], ["gap_to_complex", "0", MUX, "1"]])

# v2: the Null Source replaced by a Constant Source of complex zeros.
v2 = without(start, {NULLSRC})
v2 = with_blocks(v2, [C.block("gap_zeros", "analog_const_source_x", type="complex", const="0")],
                 [["gap_zeros", "0", MUX, "1"]])

# v3: the direct fix (type complex), with literal numbers in the clock sync and the Costas loop and
# without the Throttle (it only paces the samples).
v3 = edit.set_param(start, NULLSRC, "type", "complex")
v3 = edit.set_param(v3, CS, "sps", str(SPS))
v3 = edit.set_param(v3, CS, "loop_bw", "0.0628")
v3 = edit.set_param(v3, CS, "filter_size", "32")
v3 = edit.set_param(v3, COSTAS, "w", "0.0628")
v3 = edit.bypass(v3, THR)

# v4: the direct fix, and Symbol Sync (Gardner timing error detector) with its own polyphase matched
# filter (the same RRC filter bank) instead of the Polyphase Clock Sync: the oracle accepts another
# timing-recovery block.
TIMING = "timing_recovery"
v4 = with_blocks(without(edit.set_param(start, NULLSRC, "type", "complex"), {CS}), [
    C.block(TIMING, "digital_symbol_sync_xx", type="cc", ted_type="digital.TED_GARDNER", constellation="constel",
            sps="sps", ted_gain="1.0", loop_bw="0.045", damping="1.0", max_dev="1.5", osps="1",
            resamp_type="digital.IR_PFB_MF", nfilters="nfilts", pfb_mf_taps="rrc_taps"),
], [[CORR, "0", TIMING, "0"], [TIMING, "0", COSTAS, "0"]])

# v5: the direct fix, an explicit root-raised-cosine matched filter (FIR Filter, ccf) followed by Symbol
# Sync with an 8-tap MMSE interpolator, and a narrower Costas loop (w = 0.03).
v5 = with_blocks(without(edit.set_param(start, NULLSRC, "type", "complex"), {CS}), [
    C.block("matched_filter", "fir_filter_xxx", type="ccf", decim="1",
            taps="firdes.root_raised_cosine(1.0, sps, 1.0, eb, 11*sps)", samp_delay="0"),
    C.block(TIMING, "digital_symbol_sync_xx", type="cc", ted_type="digital.TED_GARDNER", constellation="constel",
            sps="sps", ted_gain="1.0", loop_bw="0.045", damping="1.0", max_dev="1.5", osps="1",
            resamp_type="digital.IR_MMSE_8TAP"),
], [[CORR, "0", "matched_filter", "0"], ["matched_filter", "0", TIMING, "0"], [TIMING, "0", COSTAS, "0"]])
v5 = edit.set_param(v5, COSTAS, "w", "0.03")

variants = {"v1_float_to_complex_converter.grc": v1, "v2_constant_source_zeros.grc": v2,
            "v3_direct_fix_literals_no_throttle.grc": v3, "v4_direct_fix_symbol_sync_pfb_mf.grc": v4,
            "v5_direct_fix_rrc_filter_symbol_sync.grc": v5}
variants = {k: tidy(v) for k, v in variants.items()}

# ---------------------------------------------------------------- instruction
RUN_SENTENCE = "The flowgraph should keep producing samples until it is stopped; do not limit the number of samples."
instruction = (
    "This flowgraph simulates a burst BPSK link and its receiver. A Vector Source repeats one packet of bytes: the "
    "8-byte preamble 0x2E 0xA4 0xAF 0x33 0x1F 0x22 0xF1 0xF7, then a 124-byte payload (4 zero bytes followed by the "
    "120 pseudo-random bytes numpy.random.RandomState(7).randint(0, 256, 120)), sent most significant bit first. A "
    "Constellation Modulator turns each bit into one BPSK symbol (bit 0 -> +1, bit 1 -> -1, no differential coding) "
    "at 5 samples per symbol with root-raised-cosine pulses of excess bandwidth 0.4. A Stream Mux alternates 5,280 "
    "samples of the modulator's output (1,056 symbol periods) with 4,000 zero samples from a Null Source (800 symbol "
    "periods), so the receiver sees a burst of 1,056 symbols every 1,856 symbol periods. Because of the modulator's "
    "pulse-shaping delay, each burst after the first starts with the last 27 symbols of one packet, followed by the "
    "first 1,029 symbols of the next. After a Throttle (250,000 samples/s), a Channel Model adds a carrier frequency "
    "offset of 0.0005 cycles per sample (no noise, no timing offset). On the receive side a Correlation Estimator "
    "looks for the modulated preamble, a Polyphase Clock Sync recovers the symbol timing with a bank of matched "
    "root-raised-cosine filters and outputs one sample per symbol, and a Costas loop of order 2 removes the carrier "
    "offset.\n\n"
    "The recovered symbols must be written as complex samples to the File Sink with file name symbols.dat, one "
    "sample per symbol. From the second burst on, every burst must reproduce the transmitted symbol stream: over the "
    "burst, the normalised correlation between the recovered symbols and the transmitted ±1 symbols must be at least "
    "0.95, and so must the correlation over the 64 preamble symbols within it (a perfect receiver gives 1; a "
    "constant phase rotation, such as the 180-degree ambiguity of BPSK, and a fixed delay are fine). Between bursts "
    "the output must be silent: an RMS below 0.01 over the middle 500 of the 800 silent symbol periods. The Costas "
    "loop's frequency estimate (its second output) must be written as float samples to the File Sink with file name "
    "freq.dat; while a burst is being received, from its 300th to its 1,000th symbol, the estimate must average the "
    "carrier offset in radians per symbol, 2*pi*0.0005*5 = 0.0157, within 1 %. " + RUN_SENTENCE + "\n\n"
    "At the moment GNU Radio Companion refuses to generate the flowgraph. Validation reports 'Source IO type "
    "\"float\" does not match sink IO type \"complex\"' and 'Source IO size \"4\" does not match sink IO size \"8\"' "
    "on the connection from the Null Source to the Stream Mux. Please repair the flowgraph so that it validates, "
    "runs and behaves as described. Keep the transmitter (the modulator's output must stay exactly as it is), the "
    "burst timing, the channel and both output files as they are, and keep the order-2 Costas loop in the receive "
    "chain."
)

# ---------------------------------------------------------------- observation and oracle
N_ITEMS = 8192                 # symbols per point (4.4 burst periods; 40,960 samples, 0.16 s at 250 kHz)
B0 = 66                        # burst k starts at symbol B0 + 1856k on the reference (burst 0: 27-symbol ramp first)
LAG = 128                      # latency tolerance (+- symbols) of every window
GAP_MARGIN = 150               # gap windows: the middle 500 of the 800 silent symbol periods
IN_BURST = (300, 1000)         # carrier-estimate window inside a burst (symbols from the burst start)


def burst(k: int) -> int:
    return B0 + k * PERIOD     # 66, 1922, 3778, 5634


observation = {
    "points": [{"name": "symbols", "sink": {"block_id": "blocks_file_sink", "params": {"file": SYM_FILE, "type": "complex"}}},
               {"name": "freq", "sink": {"block_id": "blocks_file_sink", "params": {"file": FREQ_FILE, "type": "float"}}},
               {"name": "tx", "edge": {"block": MOD, "port": "0"}}],
    "run": {"n_items": N_ITEMS, "min_items": N_ITEMS, "timeout_s": 10.0},
    "contract": ("the complex samples written to the File Sink whose file name is symbols.dat (the recovered symbols, "
                 "one per symbol), the float samples written to the File Sink whose file name is freq.dat (the "
                 "Costas loop's frequency estimate), and an internal tap on the output of the Constellation Modulator "
                 f"{MOD} (the transmitter, which the repair keeps)"),
}

# ---------------------------------------------------------------- the transmitted waveform (edge tap "tx")
# The Constellation Modulator's output, built in-process exactly as GRC generates it (variable_constellation
# calcdist [1,-1] / [0,1] / rot_sym 2 / dims 1, amplitude normalisation, npwr 1; generic_mod with
# differential False, pre_diff_code True, excess_bw 0.4, 5 samples per symbol), fed by the repeating packet.
# It is bit-identical to the harness's capture of the reference (checked while authoring, 18 Oct).
from gnuradio import blocks, digital, gr  # noqa: E402


def transmitted_waveform(n: int) -> np.ndarray:
    constel = digital.constellation_calcdist([1, -1], [0, 1], 2, 1,
                                             digital.constellation.AMPLITUDE_NORMALIZATION).base()
    constel.set_npwr(1.0)
    tb = gr.top_block()
    src = blocks.vector_source_b(list(PACKET), True)
    mod = digital.generic_mod(constellation=constel, differential=False, samples_per_symbol=SPS, pre_diff_code=True,
                              excess_bw=EB, verbose=False, log=False, truncate=False)
    head = blocks.head(gr.sizeof_gr_complex, n)
    snk = blocks.vector_sink_c()
    tb.connect(src, mod, head, snk)
    tb.run()
    return np.array(snk.data())


TX_SKIP = 512                  # past the pulse-shaping ramp at the start
TX_N = 4096                    # samples compared (about 820 symbols: preamble and payload of the first packet)
TX_LAG = 8
TXW = transmitted_waveform(TX_SKIP + TX_N + TX_LAG)[TX_SKIP:]
assert np.max(np.abs(TXW.imag)) == 0.0 and np.std(TXW.real) > 0.1      # BPSK: real-valued
TX_RE = [round(float(v), 6) for v in TXW.real]
TX_MIN = 0.99999               # "exactly as it is": excess bandwidth 0.39 or 0.41 already gives 0.99998
STREAM_REF = STREAM + [0] * (2 * LAG)        # padding only fixes the compared length to the 1,056 burst symbols
PRE_REF = SYMS[:N_PRE] + [0] * (2 * LAG)     # compares the 64 preamble symbols
assert len(STREAM) == N_SYM and STREAM[MOD_DELAY:MOD_DELAY + N_PRE] == SYMS[:N_PRE]


def stream_check(k: int) -> dict:
    return {"point": "symbols", "feature": "corr_ref", "ref": STREAM_REF, "max_lag": 2 * LAG, "min": 0.95,
            "skip": burst(k) - LAG, "n": len(STREAM_REF), "group": "p2p", "property": "symbols",
            "note": (f"burst {k}: normalised, phase-invariant correlation with the 1,056 transmitted symbols it carries "
                     f"(the last 27 of one packet, then 1,029 of the next), expected at symbol {burst(k)} +-{LAG}; "
                     "reference 0.998, valid receivers 0.988-0.9996, no timing or carrier recovery about 0.1")}


def gap_check(k: int) -> dict:
    return {"point": "symbols", "feature": "rms", "max": 0.01, "skip": burst(k) + N_SYM + GAP_MARGIN,
            "n": GAP_SYM - 2 * GAP_MARGIN, "group": "f2p", "property": "silence",
            "note": f"silence in the middle 500 of the 800-symbol gap after burst {k} (reference 0.0; bursts carry RMS about 1)"}


def carrier_check(k: int) -> dict:
    return {"point": "freq", "feature": "mean", "expect": round(F_EST, 6), "rel_tol": 0.01,
            "skip": burst(k) + IN_BURST[0], "n": IN_BURST[1] - IN_BURST[0], "group": "p2p",
            "property": "carrier_estimate",
            "note": (f"Costas frequency estimate over symbols {IN_BURST[0]}-{IN_BURST[1]} of burst {k}: 2*pi*0.0005*5 "
                     "rad/symbol +-1 % (reference +0.001 %; Costas loop bandwidths 0.01-0.5 within 0.23 %)")}


def preamble_check(k: int) -> dict:
    return {"point": "symbols", "feature": "corr_ref", "ref": PRE_REF, "max_lag": 2 * LAG, "min": 0.95,
            "skip": burst(k) + MOD_DELAY - LAG, "n": len(PRE_REF), "group": "p2p", "property": "preamble",
            "note": (f"the 64 preamble symbols in burst {k} (27 symbols after its start); reference 0.998, the "
                     "example's preamble about 0.3")}


checks = (
    [stream_check(k) for k in (1, 2, 3)]
    + [preamble_check(k) for k in (1, 2, 3)]
    + [carrier_check(k) for k in (1, 2, 3)]
    + [gap_check(k) for k in (0, 1, 2, 3)]
    + [{"point": "tx", "feature": "corr_ref", "ref": TX_RE, "signed": True, "max_lag": TX_LAG, "skip": TX_SKIP,
        "n": TX_N + TX_LAG, "min": TX_MIN, "group": "p2p", "property": "transmitter",
        "note": ("the transmitter is unchanged: signed correlation of 4,096 samples of the modulator's output with the "
                 "transmitted waveform (reference 1.0). Excess bandwidth 0.35 gives 0.99943 and 0.45 0.99949; an "
                 "inverted bit mapping -1; differential coding, another payload or samples per symbol far lower")}]
)
graph_checks = [
    {"kind": "block_present", "ids": ["digital_costas_loop_cc"], "params": {"order": "2"}, "group": "p2p",
     "property": "receiver", "note": "the order-2 Costas loop is kept (freq.dat is its frequency estimate)"},
    {"kind": "path", "from_ids": ["channels_channel_model"], "to_ids": ["blocks_file_sink"],
     "via_ids": ["digital_costas_loop_cc"], "group": "p2p", "property": "receiver",
     "note": "the received signal reaches a File Sink through the Costas loop (rules out synthesised outputs)"},
    {"kind": "block_present", "ids": ["channels_channel_model"], "params": CHAN_PARAMS, "group": "p2p",
     "property": "channel", "note": "the Channel Model still takes its offsets and noise from freq_offset, time_offset and noise"},
    {"kind": "block_present", "ids": ["variable"], "params": {"value": "1"}, "group": "p2p", "property": "channel",
     "note": "a variable holds the timing offset 1 (time_offset: no timing offset)"},
]

spec_properties = [
    {"id": "validates", "text": "the repaired flowgraph passes GRC validation (the IO types of every connection match), "
                                "generates and runs", "checked_by": "ladder V3-V4"},
    {"id": "run", "text": "the flowgraph keeps producing samples until it is stopped (no sample limit)",
     "checked_by": "V4 liveness: at least 8,192 items at symbols.dat, at freq.dat and at the modulator's output"},
    {"id": "transmitter", "text": "the transmitter is kept: the packet (preamble 2E A4 AF 33 1F 22 F1 F7, 4 zero bytes, "
                                  "120 bytes from numpy.random.RandomState(7)), MSB first, BPSK with bit 0 -> +1 and "
                                  "bit 1 -> -1, no differential coding, 5 samples per symbol, root-raised-cosine pulses "
                                  "of excess bandwidth 0.4: the modulator's output stays exactly as it is",
     "checked_by": ("p2p corr_ref (signed) >= 0.99999 of 4,096 samples of the Constellation Modulator's output (edge "
                    "tap) against the waveform computed in build.py; the receiver shares eb and the symbol checks are "
                    "phase-invariant, so excess bandwidth and the bit mapping are only visible here")},
    {"id": "outputs", "text": "complex samples are written to the File Sink symbols.dat and float samples to the File "
                              "Sink freq.dat, both at one sample per symbol",
     "checked_by": "observation contract (file name and item type; a missing or differently typed sink fails V5)"},
    {"id": "silence", "text": "the Null Source supplies complex zeros: between bursts the output is silent, RMS below "
                              "0.01 over the middle 500 of the 800 silent symbol periods",
     "checked_by": "f2p rms <= 0.01 after bursts 0, 1, 2 and 3 (the repaired element)"},
    {"id": "symbols", "text": "symbols.dat holds one sample per symbol and every burst from the second on reproduces "
                              "the transmitted symbol stream (8-byte preamble 2E A4 AF 33 1F 22 F1 F7, 4 zero bytes, "
                              "120 bytes from numpy.random.RandomState(7), MSB first, BPSK, 5 samples per symbol, RRC "
                              "0.4; each burst carries the last 27 symbols of one packet and the first 1,029 of the "
                              "next) with a normalised correlation of at least 0.95, up to a constant phase rotation "
                              "and a fixed delay",
     "checked_by": "p2p corr_ref >= 0.95 on bursts 1, 2 and 3 (also catches a wrong samples-per-symbol, differential "
                   "coding, another payload, a changed burst length or gap, or missing timing/carrier recovery)"},
    {"id": "preamble", "text": "each burst carries the 64 preamble symbols of 2E A4 AF 33 1F 22 F1 F7 (correlation at "
                               "least 0.95)",
     "checked_by": "p2p corr_ref >= 0.95 over the preamble of bursts 1, 2 and 3"},
    {"id": "carrier_estimate", "text": "the channel adds a carrier offset of 0.0005 cycles per sample and the order-2 "
                                       "Costas loop removes it: its frequency output (freq.dat) averages "
                                       "2*pi*0.0005*5 = 0.0157 rad/symbol within 1 % over symbols 300-1,000 of a burst",
     "checked_by": "p2p mean of freq.dat over symbols 300-1000 of bursts 1, 2 and 3"},
    {"id": "framing", "text": "bursts of 1,056 symbol periods (5,280 samples) separated by 4,000 zero samples (800 "
                              "symbol periods), i.e. one burst every 1,856 symbol periods",
     "checked_by": "the burst, preamble, carrier and silence windows all sit at 66 + 1856k (+-128 symbols of latency "
                   "tolerance in the correlation windows); another burst length or gap moves later bursts out of them"},
    {"id": "receiver", "text": "the order-2 Costas loop is kept in the receive chain and the recovered symbols come "
                               "through it",
     "checked_by": "p2p graph checks block_present digital_costas_loop_cc with order 2 and path Channel Model -> "
                   "Costas loop -> File Sink (an order-4 loop passes every signal check)"},
    {"id": "channel", "text": "the channel is kept: carrier offset 0.0005 cycles/sample, no noise, no timing offset",
     "checked_by": ("p2p: the carrier estimate (offset), the silence checks (noise of 0.05 already gives gap RMS "
                    "0.022) and graph checks: the Channel Model still refers to freq_offset / time_offset / noise and "
                    "a variable holds the timing offset 1")},
]

mutations = [
    {"op": "rewire_input", "dst": SYM_SINK, "dst_port": "0", "new_src": CS, "breaks": "symbols",
     "note": "wrong tap: clock sync output without carrier recovery (symbols rotate with the offset)"},
    {"op": "set_param", "block": CS, "param": "sps", "value": "sps+1", "breaks": "symbols",
     "note": "clock sync expects 6 samples per symbol"},
    {"op": "set_param", "block": MOD, "param": "samples_per_symbol", "value": "sps+1", "breaks": "symbols",
     "note": "transmitter at 6 samples per symbol, receiver at 5"},
    {"op": "set_param", "block": MOD, "param": "differential", "value": "True", "breaks": "symbols",
     "note": "differential coding in the transmitter"},
    {"op": "set_param", "block": "eb", "param": "value", "value": "0.35", "breaks": "transmitter",
     "note": "the example's excess bandwidth 0.35 at transmitter and receiver (the symbols are still recovered)"},
    {"op": "set_param", "block": "constel", "param": "sym_map", "value": "[1,0]", "breaks": "transmitter",
     "note": "inverted bit mapping (bit 0 -> -1): the phase-invariant symbol checks cannot see it"},
    {"op": "set_param", "block": COSTAS, "param": "order", "value": "4", "breaks": "receiver",
     "note": "an order-4 Costas loop instead of order 2 (it still locks on this BPSK signal)"},
    {"op": "set_param", "block": "data", "param": "value",
     "value": "[0]*4 + numpy.random.RandomState(8).randint(0, 256, payload_size).tolist()", "breaks": "symbols",
     "note": "another payload"},
    {"op": "set_param", "block": "preamble", "param": "value", "value": ORIG["preamble"], "breaks": "preamble",
     "note": "the example's preamble"},
    {"op": "set_param", "block": "gap", "param": "value", "value": "3000", "breaks": "framing",
     "note": "600 silent symbol periods instead of 800"},
    {"op": "set_param", "block": "payload_size", "param": "value", "value": "100", "breaks": "framing",
     "note": "shorter bursts"},
    {"op": "set_param", "block": "sps", "param": "value", "value": "4", "breaks": "framing",
     "note": "the example's 4 samples per symbol throughout (gap of 1,000 symbol periods)"},
    {"op": "set_id", "block": NULLSRC, "new_id": "analog_noise_source_x", "breaks": "silence",
     "note": ("the gap source replaced by a complex Noise Source (amplitude 1) instead of a source of zeros. (Feeding "
              "the gap input from the modulator instead is not a valid mutant: both Stream Mux inputs then read one "
              "buffer alternately and the graph crawls, failing V4 liveness.)")},
    {"op": "set_param", "block": "noise", "param": "value", "value": "0.05", "breaks": "channel",
     "note": "channel noise 0.05 (gap RMS 0.022)"},
    {"op": "set_param", "block": "time_offset", "param": "value", "value": "1.002", "breaks": "channel",
     "note": "a 0.2 % timing offset in the channel (the receiver follows it)"},
    {"op": "set_param", "block": CHAN, "param": "epsilon", "value": "1.002", "breaks": "channel",
     "note": "the same timing offset written into the Channel Model"},
    {"op": "set_param", "block": "freq_offset", "param": "value", "value": "0", "breaks": "carrier_estimate",
     "note": "no carrier offset"},
    {"op": "scale_param", "block": "freq_offset", "param": "value", "factor": 2, "breaks": "carrier_estimate",
     "note": "twice the carrier offset"},
    {"op": "scale_param", "block": "freq_offset", "param": "value", "factor": 1.02, "breaks": "carrier_estimate",
     "note": "near miss: carrier offset 2 % too large (0.00051 cycles/sample)"},
    {"op": "set_param", "block": COSTAS, "param": "w", "value": "0.0005", "breaks": "carrier_estimate",
     "note": "Costas loop bandwidth too small to acquire the offset"},
    {"op": "rewire_input", "dst": FREQ_SINK, "dst_port": "0", "new_src": COSTAS, "new_src_port": "2",
     "breaks": "carrier_estimate", "note": "wrong tap: the Costas loop's phase output written to freq.dat (its error "
                                           "output, port 3, cannot be tapped alone: GNU Radio requires port 2 connected)"},
]

record = {
    "id": TASK_ID, "tier": "F", "split": SPLIT, "version": 2, "canary": CANARY,
    "license": f"GPL-3.0-or-later (derived from GNU Radio 3.10.12 example {SEED})",
    "source": {"kind": "gnuradio_example", "ref": SEED, "sha256": seed_sha, "derivation": DERIVATION},
    "cluster_id": SEED,
    "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring",
                   "generator_version": "0.2.0"},
    "review": {"status": "pending-team-review", "reviewers": [], "notes": (
        "Round-1 review applied 18 Oct 2026 (see reports/tier_f_reviews/round1/gbf-dev-10.resolution.json). The fault "
        "moved from the Polyphase Clock Sync (type fff) to the Null Source (type float) so that this dev item and "
        "gbf-test-17 (test) no longer repair the same block (fix plan G8; review C1); the team may move it back. Known "
        "limits: (1) freq.dat is required to hold the Costas loop's frequency output, but graph checks are block-level "
        "and cannot tie the freq.dat File Sink to Costas output port 1: an answer that feeds freq.dat from a Constant "
        "Source of 0.0157 while the Costas loop stays on the symbol path would pass (implausible in a one-edit repair). "
        "(2) The channel 'keep' is checked at string level for the timing offset: the Channel Model must still refer to "
        "freq_offset / time_offset / noise and a variable must hold 1; noise and the carrier offset are also checked "
        "by behaviour. (3) Timing recovery is not pinned to the Polyphase Clock Sync: Symbol Sync or M&M receivers pass "
        "(variants v4, v5). (4) The transmitter is observed by an internal edge tap on the block "
        f"{MOD} (the start graph's name): a repair that deletes and re-creates the Constellation "
        "Modulator under another name fails that observation. A one-edit repair of the Null Source never touches it. "
        "(5) The Costas loop's bandwidth is not a requirement (bandwidths 0.01-0.5 give the same symbols and carrier "
        "estimate); only its presence, its order 2 and its place in the receive chain are, which is what the "
        "instruction's keep sentence now says. (6) Round-1 F1 (tap stall) is fixed by tooling T7: the wrong repair "
        "'Costas loop ahead of a float clock sync' now reaches V5 and fails the carrier checks (freq.dat 0.00314) "
        "instead of failing V4 liveness. (7) GPL-3.0-or-later attribution: derived from the GNU Radio 3.10.12 "
        "example digital/demod/test_corr_est.grc (GNU Radio project); the licence decision for seed-derived items "
        "is the team's.")},
    "family": "repair", "level": "L2",
    "instruction": instruction,
    "start_grc": "start.grc", "reference_grc": "reference.grc",
    "observation": observation,
    "oracle": {"checks": checks, "graph_checks": graph_checks},
    "mutations": mutations,
    "spec_properties": spec_properties,
    "perturbation": PERTURBATION,
    "fault": {"type": "type_mismatch",
              "description": (f"The Null Source ('{NULLSRC}') that supplies the zero samples between bursts has type "
                              "'float' instead of 'complex'. It feeds input 1 of the complex Stream Mux, so GRC "
                              "validation reports 'Source IO type \"float\" does not match sink IO type \"complex\"' and "
                              "'Source IO size \"4\" does not match sink IO size \"8\"' on that connection (V3). Fix: "
                              "Null Source type complex, or convert its output (Float To Complex), or another source of "
                              "complex zeros. Turning the Stream Mux to float instead creates new mismatches with the "
                              "complex modulator and Throttle. Because the start fails at V3, p2p here means the "
                              "properties carried by the parts of the graph the fault does not touch (transmitter, "
                              "channel, receiver); f2p is the silence the repaired Null Source supplies."),
              "expected_rung": "V3", "block": NULLSRC, "param": "type", "wrong_value": "float",
              "correct_value": "complex"},
    "disjointness": ("Repair of a float-vs-complex type mismatch at the Null Source that fills the gaps of a burst BPSK "
                     "link with a correlation-based preamble detector and timing/carrier recovery, from the dev-split "
                     "seed digital/demod/test_corr_est.grc. Nearest training-side specs (reports/reference/"
                     "training_task_specs.json): t05_qam_debug ('suspect the sample rates or types are off between "
                     "blocks' in a QAM demo: also an IO-type repair, but another graph and a parameter task), "
                     "t04_broken_sink_repair (an unconnected port) and GRC-Agent scenario 21 (look up a float-to-"
                     "complex converter on request); t25 only explains a QAM sync chain. No training-side spec uses "
                     "this seed, a Correlation Estimator, a Polyphase Clock Sync or a Costas loop. Within GRC-Bench, "
                     "gbf-test-17 (test) repairs an undefined variable in a Polyphase Clock Sync of another seed."),
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
