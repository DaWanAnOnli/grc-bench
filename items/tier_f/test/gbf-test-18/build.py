"""Authoring script for GRC-Bench Tier-F item gbf-test-18 (repair, L2, composed graph, fault rate_mismatch).

    cd grc-bench && PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 ~/grc-agent/.venv/bin/python \
        items/tier_f/test/gbf-test-18/build.py
    ./gb tier-f gates items/tier_f/test/gbf-test-18

Theme: a CW (Morse) beacon receiver. A simulated complex baseband recording at 96 kHz holds the
beacon's carrier at +21 kHz (amplitude 0.5), keyed on and off by a 5 Hz square wave (100 ms off, 100 ms
on, ...), and two stronger unkeyed neighbouring carriers (amplitude 1.0) at +25 kHz and -17 kHz. The
receiver
  1. tunes: multiplies by a -19.5 kHz complex oscillator, so the beacon moves to +1.5 kHz and the
     neighbours to +5.5 kHz and -36.5 kHz;
  2. channel-filters and decimates: Low Pass Filter (cutoff 3 kHz, transition width 1 kHz) with
     decimation 8, giving a 12 kHz channel that is written (complex) to the File Sink channel.dat;
  3. beats the beacon down to an audible pitch: multiplies the 12 kHz channel by a -800 Hz complex
     beat-frequency oscillator (BFO) of amplitude 1.6, so the beacon sits at +700 Hz with amplitude 0.8;
  4. takes the real part, written (float, 12 kHz) to the File Sink audio.dat.
Fifteen functional blocks (beacon, keying square wave, Float To Complex, keying Multiply, 2 neighbours,
Add, tuning LO, mixer, Low Pass Filter, channel File Sink, BFO, BFO Multiply, Complex To Real, audio
File Sink) with a rate change (96 kHz -> 12 kHz), filtering and frequency conversion: level L2
(proposal 5.2: 6-15 blocks, rate changes, filtering).

Version 2 (18 Oct 2026, round-1 fix stage)
  * F1 (both lenses, medium): in version 1 the beacon was an unkeyed carrier, so any 700 Hz tone
    passed: a free-running 700 Hz oscillator wired to audio.dat with the faulty BFO left in place, or a
    graph with the neighbours zeroed and the filter replaced by Keep 1 in N. The beacon is now keyed by a
    5 Hz square-wave Signal Source (NOT a Vector Source: that would add the data_source class and move
    the cell to dev; fix_plan), and V6 checks the keying in both outputs: the level inside two "on" and
    two "off" periods of channel.dat and audio.dat. A free-running oscillator is never silent and
    fails. The channel filter the instruction names ("low-pass filtered and decimated by 8") is a graph
    check (p2p): a path Signal Source -> filter -> File Sink, accepting every GNU Radio 3.10 block that
    can low-pass filter a channel (Low Pass / Band Pass / RRC / FIR / FFT / FFT Low Pass / Interpolating
    FIR / Frequency Xlating FIR or FFT / Rational Resampler / PFB Arbitrary Resampler / PFB Decimator /
    channelizers / IIR / single-pole IIR / Moving Average / Integrate); whether it filters well enough is
    left to the signal checks (variant v12 uses an FFT Filter). To stay within 15 functional blocks (L2),
    the audio gain 1.6 moved from a Multiply Const into the BFO amplitude. Authoring probes: the
    reviewers' graphs now fail V6 (a free-running 700 Hz oscillator on audio.dat with the faulty BFO
    left in place: audio rms 0.567 in the off windows; neighbours zeroed and the filter replaced by Keep
    1 in N: no filter path; two oscillators straight into the two File Sinks: off windows and no filter
    path). A graph that rebuilds the keyed tone from its own oscillator times a copy of the key still
    passes (review.notes limit 1).
  * item-quality F2 (run length): the set run sentence is in the instruction (V5 liveness: 8400 items).
  * F2 compliance (split follows one path): moot under signature v2 (a set of functional classes of all
    blocks, independent of paths and connection order); the reference and every variant are printed
    below with their v2 signature. The keying source sits on the keying Multiply's port 1 and is
    absorbed as an oscillator; on port 0 it would add harmonic_wave, which also hashes to test.
  * F3 compliance (composed L2 is not a plan source category): recorded in review.notes for the team.
  * F4 compliance / F3 item-quality (disjointness incomplete): t05_qam_debug and t12_fivegnalizer_modify
    added.

Fault (rate_mismatch, expected rung V6): the BFO Signal Source, which runs after the decimate-by-8
filter and so produces samples at 12 kHz, has its Sample Rate set to samp_rate (96 kHz) instead of
samp_rate/decim. Its -800 Hz setting then advances the phase as -100 Hz does at 12 kHz, so the beacon
comes out of audio.dat at 1.4 kHz instead of 700 Hz. The graph validates and runs (V1-V5 pass); only
the behaviour is wrong. In start.grc every Signal Source and the Low Pass Filter state their Sample Rate
explicitly (samp_rate), so that the faulty block does not stand out by formatting; apart from the
fault this is exactly the reference (checked with grc.compact_equivalent).

Oracle (all numbers are stated in the instruction; tolerances: frequency +-1 %, amplitude +-5 %)
  Time base at 12 kHz: the key is off for output items [0, 1200), on [1200, 2400), off [2400, 3600), ...
  (the square wave starts low; the reference's channel filter delays by ~14 items).
  * spectral window: skip 1200 (0.1 s), 7200 items = 3 whole keying periods; 1.5 kHz, 700 Hz and every
    keying sideband (multiples of 5 Hz) fall on exact FFT bins (1.667 Hz). A 50 %-duty on/off key puts
    0.25 + sum_{k odd <= 39} 2/(pi k)^2 = 99.5 % of the keyed carrier's power within +-200 Hz (99.0 %
    within +-100 Hz, 98.0 % within +-50 Hz), so the purity bands are +-200 Hz with >= 98 %.
  * level windows: 400 items centred in a period, i.e. at least 400 items (33 ms) from each keying edge,
    which covers the group delay of any reasonable channel filter (a 100 Hz-wide transition at 96 kHz
    would be about the limit): on 1600-2000 and 4000-4400, off 2800-3200 and 5200-5600.
    channel on: rms 0.5 +-5 %; audio on: rms 0.8/sqrt2 +-5 %, peak 0.8 +-5 %; off: rms <= 5 % of the
    on level (0.025 channel, 0.0283 audio).
  * groups: f2p = the audio frequency (700 Hz, +-200 Hz purity); p2p = everything that held in the
    start graph (channel checks, the audio's keyed levels and peak, the channel-filter graph check).
Mutations: a free-running replacement for the audio path cannot be written as one edit of the
reference; its behaviour (a 700 Hz tone that is never silent) is covered by the "key held on" mutant.

Deterministic; needs GNU Radio (catalog defaults are used to write the compact files).
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from grcbench.common.canary import CANARY        # noqa: E402
from grcbench.tier_f import compose as C          # noqa: E402
from grcbench.tier_f import edit, grc, grcfile, taskio  # noqa: E402

TASK_ID = "gbf-test-18"
SPLIT = "test"
VERSION = 2

# ---- the specification (every number the oracle depends on is stated in the instruction) --------
FS_IN = 96000          # recording sample rate, Hz
DECIM = 8
FS_OUT = FS_IN // DECIM   # 12 kHz channel and audio rate
BEACON_F, BEACON_A = 21000, 0.5
KEY_F = 5              # on/off keying square wave, Hz (100 ms off, 100 ms on)
NEIGH = [(25000, 1.0), (-17000, 1.0)]
TUNE_F = 19500         # +19.5 kHz moves to 0 Hz
CH_F = BEACON_F - TUNE_F   # +1.5 kHz in the channel
CUTOFF, WIDTH = 3000, 1000
BFO_F = -800           # beat-frequency oscillator, at the channel rate
AUDIO_F = CH_F + BFO_F     # 700 Hz
AUDIO_PEAK = 0.8
GAIN = AUDIO_PEAK / BEACON_A   # 1.6, the BFO amplitude
CH_FILE, AU_FILE = "channel.dat", "audio.dat"

FAULT_AUDIO_F = CH_F + BFO_F * FS_OUT / FS_IN   # 1400 Hz: the symptom stated in the instruction
HALF_PERIOD = FS_OUT // (2 * KEY_F)             # 1200 items at 12 kHz
SKIP, N = HALF_PERIOD, 6 * HALF_PERIOD          # skip 0.1 s, then 3 whole keying periods
WIN = 400                                       # level windows: 400 items, >= 400 items from an edge
ON_WINDOWS = [c * HALF_PERIOD // 2 - WIN // 2 for c in (3, 7)]    # centred in [1200, 2400), [3600, 4800): 1600, 4000
OFF_WINDOWS = [c * HALF_PERIOD // 2 - WIN // 2 for c in (5, 9)]   # centred in [2400, 3600), [4800, 6000): 2800, 5200
RUN_ITEMS = SKIP + N                            # 8400
FREQ_REL = 0.01
AMP_REL = 0.05
OFF_REL = 0.05
MIN_FRACTION = 0.98
BAND_HALF = 200

# names in GRC's default style, as a user's own flowgraph would have them
BEACON, NB_HI, NB_LO = "analog_sig_source_x_0", "analog_sig_source_x_1", "analog_sig_source_x_2"
KEY, KEY_C, KEY_MIX = "analog_sig_source_x_5", "blocks_float_to_complex_0", "blocks_multiply_xx_2"
ADD = "blocks_add_xx_0"
LO, MIX = "analog_sig_source_x_3", "blocks_multiply_xx_0"
LPF, CH_SINK = "low_pass_filter_0", "blocks_file_sink_0"
BFO, BFO_MIX = "analog_sig_source_x_4", "blocks_multiply_xx_1"
TO_REAL, AU_SINK = "blocks_complex_to_real_0", "blocks_file_sink_1"

INSTRUCTION = (
    "This flowgraph is a receiver for a CW (Morse) beacon. Its input is a simulated complex baseband "
    "recording at a sample rate of 96 kHz. It holds the beacon's carrier at +21 kHz (amplitude 0.5), keyed "
    "on and off by a 5 Hz square wave (off for 100 ms, then on for 100 ms, repeating), and two stronger, "
    "unkeyed neighbouring carriers (amplitude 1.0) at +25 kHz and -17 kHz. The receiver should produce two "
    "outputs, both at a sample rate of 12 kHz. The first is the channel: the recording shifted so that "
    "+19.5 kHz moves to 0 Hz, low-pass filtered and decimated by 8, written as complex items to the File "
    "Sink whose file name is channel.dat. In the channel the beacon should sit at +1.5 kHz with amplitude "
    "0.5 while it is keyed on, and the neighbouring carriers should be gone: at least 98 % of the channel "
    "power within 200 Hz of +1.5 kHz, and the channel silent (below 5 % of the on level) while the beacon "
    "is off. The second is the audio: the beacon heard as a clean 700 Hz tone with a peak amplitude of 0.8 "
    "while it is keyed on (at least 98 % of the audio power within 200 Hz of 700 Hz) and silence (below "
    "5 % of that level) while it is off, written as float items to the File Sink whose file name is "
    "audio.dat. Tolerances: frequencies within 1 %, amplitudes within 5 %; levels are measured inside the "
    "on and off periods, away from the keying edges, and the first 0.1 s is ignored. The flowgraph runs "
    "without errors and channel.dat is correct, but in audio.dat the beacon comes out as a tone of about "
    "1.4 kHz instead of 700 Hz. Repair the flowgraph so that both outputs behave as described, and keep "
    "both File Sinks (channel.dat and audio.dat). The flowgraph should keep producing samples until it is "
    "stopped; do not limit the number of samples."
)


def csrc(name: str, freq, amp, samp_rate: str = "samp_rate", **extra) -> dict:
    return C.sig_source(name, "complex", freq=freq, amp=amp, samp_rate=samp_rate, **extra)


def keyed_recording(names: dict, beacon_f, rate: str, key_rate: str | None = None) -> tuple[list, list]:
    """Blocks and connections of the simulated recording (keyed beacon + two neighbours) ending in
    names['sum'] (an Add); used by the reference and by the full-rebuild variants."""
    blocks = [
        csrc(names["beacon"], beacon_f, BEACON_A, samp_rate=rate),
        C.sig_source(names["key"], "float", KEY_F, 1, waveform="analog.GR_SQR_WAVE", samp_rate=key_rate or rate),
        C.block(names["key_c"], "blocks_float_to_complex"),
        C.block(names["key_mix"], "blocks_multiply_xx", type="complex"),
        csrc(names["nb1"], NEIGH[0][0], NEIGH[0][1], samp_rate=rate),
        csrc(names["nb2"], NEIGH[1][0], NEIGH[1][1], samp_rate=rate),
        C.block(names["sum"], "blocks_add_xx", type="complex", num_inputs="3"),
    ]
    conns = [[names["beacon"], "0", names["key_mix"], "0"], [names["key"], "0", names["key_c"], "0"],
             [names["key_c"], "0", names["key_mix"], "1"], [names["key_mix"], "0", names["sum"], "0"],
             [names["nb1"], "0", names["sum"], "1"], [names["nb2"], "0", names["sum"], "2"]]
    return blocks, conns


# ---- reference -----------------------------------------------------------------------------------
def reference_full() -> dict:
    rec_b, rec_c = keyed_recording({"beacon": BEACON, "key": KEY, "key_c": KEY_C, "key_mix": KEY_MIX,
                                    "nb1": NB_HI, "nb2": NB_LO, "sum": ADD}, BEACON_F, "samp_rate")
    blocks = rec_b + [
        csrc(LO, -TUNE_F, 1),
        C.block(MIX, "blocks_multiply_xx", type="complex"),
        C.block(LPF, "low_pass_filter", type="fir_filter_ccf", decim="decim", gain="1", samp_rate="samp_rate",
                cutoff_freq=CUTOFF, width=WIDTH, win="window.WIN_HAMMING", beta="6.76"),
        C.file_sink(CH_SINK, "complex", CH_FILE),
        csrc(BFO, BFO_F, GAIN, samp_rate="samp_rate/decim"),
        C.block(BFO_MIX, "blocks_multiply_xx", type="complex"),
        C.block(TO_REAL, "blocks_complex_to_real"),
        C.file_sink(AU_SINK, "float", AU_FILE),
    ]
    conns = rec_c + [
        [ADD, "0", MIX, "0"], [LO, "0", MIX, "1"], [MIX, "0", LPF, "0"],
        [LPF, "0", CH_SINK, "0"], [LPF, "0", BFO_MIX, "0"], [BFO, "0", BFO_MIX, "1"],
        [BFO_MIX, "0", TO_REAL, "0"], [TO_REAL, "0", AU_SINK, "0"]]
    return C.graph("cw_beacon_rx", blocks, conns, {"samp_rate": str(FS_IN), "decim": str(DECIM)})


def compact(data: dict) -> dict:
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in data["blocks"]}
    out = grcfile.compact(data, defaults)
    out["options"] = data["options"]
    diffs = grc.compact_equivalent(data, out)
    assert not diffs, f"compact form differs from the full form: {diffs}"
    return out


def catalog_order(data: dict) -> dict:
    """Same flowgraph with every block's parameters listed in the block catalog's order."""
    out = copy.deepcopy(data)
    plat = grc.platform()
    for b in out["blocks"]:
        cat = plat.blocks.get(b["id"])
        order = [p["id"] for p in (getattr(cat, "parameters_data", None) or [])]
        rank = {pid: i for i, pid in enumerate(order)}
        b["parameters"] = dict(sorted(b["parameters"].items(), key=lambda kv: rank.get(kv[0], len(rank))))
    return out


def start_from(ref: dict) -> dict:
    """The fault (BFO at the 96 kHz rate) plus explicit, behaviour-neutral Sample Rate parameters,
    with parameters in catalog order so that the faulty block looks like every other Signal Source."""
    start = edit.set_param(ref, BFO, "samp_rate", "samp_rate")          # the injected rate mismatch
    for b in start["blocks"]:
        if b["id"] in ("analog_sig_source_x", "low_pass_filter"):
            b["parameters"].setdefault("samp_rate", "samp_rate")
    start = catalog_order(start)
    diffs = grc.compact_equivalent(ref, start)
    assert diffs == [f"{BFO}.samp_rate: 'samp_rate/decim' != 'samp_rate'"], diffs
    return start


# ---- equivalent variants: full rebuilds (genuinely different receivers) ------------------------------
def variant_rotator_bfo() -> dict:
    """BFO done with a Rotator (phase increment for -800 Hz at the 12 kHz channel rate) and a Multiply
    Const for the audio gain; a separate channel-rate variable; Blackman channel filter."""
    rec_b, rec_c = keyed_recording({"beacon": "beacon", "key": "keyer", "key_c": "key_cplx", "key_mix": "keying",
                                    "nb1": "neighbour_a", "nb2": "neighbour_b", "sum": "rf_sum"}, "21e3", "rf_rate")
    blocks = [{"name": "imports", "id": "import", "parameters": {"imports": "import math"}}] + rec_b + [
        csrc("tuner", "-19.5e3", "1", samp_rate="rf_rate"),
        C.block("tune_mixer", "blocks_multiply_xx", type="complex"),
        C.block("chan_filter", "low_pass_filter", type="fir_filter_ccf", decim="8", samp_rate="rf_rate",
                cutoff_freq="2500", width="1500", win="window.WIN_BLACKMAN"),
        C.file_sink("chan_out", "complex", CH_FILE),
        C.block("bfo", "blocks_rotator_cc", phase_inc="-2 * math.pi * 800 / ch_rate"),
        C.block("real_part", "blocks_complex_to_real"),
        C.multiply_const("volume", "float", "1.6"),
        C.file_sink("audio_out", "float", AU_FILE),
    ]
    conns = rec_c + [["rf_sum", "0", "tune_mixer", "0"], ["tuner", "0", "tune_mixer", "1"],
                     ["tune_mixer", "0", "chan_filter", "0"], ["chan_filter", "0", "chan_out", "0"],
                     ["chan_filter", "0", "bfo", "0"], ["bfo", "0", "real_part", "0"], ["real_part", "0", "volume", "0"],
                     ["volume", "0", "audio_out", "0"]]
    return C.graph("cw_rx_rotator", blocks, conns, {"rf_rate": "96e3", "ch_rate": "rf_rate / 8"})


def variant_xlating_filter() -> dict:
    """Tuning, channel filtering and decimation in one Frequency Xlating FIR Filter; the BFO Signal
    Source runs at a literal 12 kHz with amplitude 1.6."""
    rec_b, rec_c = keyed_recording({"beacon": "tx_beacon", "key": "tx_key", "key_c": "tx_key_c", "key_mix": "tx_keyer",
                                    "nb1": "tx_adj1", "nb2": "tx_adj2", "sum": "antenna"}, BEACON_F, "samp_rate")
    blocks = rec_b + [
        C.block("ddc", "freq_xlating_fir_filter_xxx", type="ccf", decim="8",
                taps="firdes.low_pass(1.0, samp_rate, 3000, 1000)", center_freq="19500", samp_rate="samp_rate"),
        C.file_sink("ch_file", "complex", "/tmp/" + CH_FILE),
        csrc("beat_osc", "-800", "1.6", samp_rate="12000"),
        C.block("product", "blocks_multiply_xx", type="complex"),
        C.block("re", "blocks_complex_to_real"),
        C.file_sink("au_file", "float", "/tmp/" + AU_FILE),
    ]
    conns = rec_c + [["antenna", "0", "ddc", "0"], ["ddc", "0", "ch_file", "0"], ["ddc", "0", "product", "0"],
                     ["beat_osc", "0", "product", "1"], ["product", "0", "re", "0"], ["re", "0", "au_file", "0"]]
    return C.graph("cw_rx_xlating", blocks, conns, {"samp_rate": "96000"})


def variant_rotator_tune_conjugate_bfo() -> dict:
    """Tuning with a Rotator at 96 kHz, a generic decimating FIR Filter with firdes taps, the BFO as a
    +800 Hz oscillator applied through Multiply Conjugate, and Complex To Float (imaginary part parked
    on a Null Sink)."""
    rec_b, rec_c = keyed_recording({"beacon": "src_beacon", "key": "src_key", "key_c": "key_to_c", "key_mix": "on_off",
                                    "nb1": "src_qrm1", "nb2": "src_qrm2", "sum": "sum3"}, "f_beacon", "samp_rate")
    blocks = [C.block("imp", "import", imports="import math"),
              C.block("imp_window", "import", imports="from gnuradio.fft import window")] + rec_b + [
        C.block("tune", "blocks_rotator_cc", phase_inc="-2 * math.pi * f_tune / samp_rate"),
        C.block("chan_fir", "fir_filter_xxx", type="ccf", decim="decim",
                taps="firdes.low_pass(1.0, samp_rate, 3000, 1000, window.WIN_HANN)"),
        C.file_sink("channel_sink", "complex", CH_FILE),
        csrc("bfo_conj", "800", "1.6", samp_rate="samp_rate / decim"),
        C.block("mix_conj", "blocks_multiply_conjugate_cc"),
        C.block("c2f", "blocks_complex_to_float"),
        C.block("imag_dump", "blocks_null_sink", type="float"),
        C.file_sink("audio_sink", "float", AU_FILE),
    ]
    conns = rec_c + [["sum3", "0", "tune", "0"], ["tune", "0", "chan_fir", "0"], ["chan_fir", "0", "channel_sink", "0"],
                     ["chan_fir", "0", "mix_conj", "0"], ["bfo_conj", "0", "mix_conj", "1"], ["mix_conj", "0", "c2f", "0"],
                     ["c2f", "0", "audio_sink", "0"], ["c2f", "1", "imag_dump", "0"]]
    return C.graph("cw_rx_conj", blocks, conns, {"samp_rate": "96000", "decim": "8", "f_beacon": "21000",
                                                 "f_tune": "19500"})


# ---- equivalent variants: one-edit repairs of start.grc (round-1 reviewer alternatives) --------------
def repair_literal_rate(start: dict) -> dict:
    """BFO Sample Rate set to the literal 12000."""
    return edit.set_param(start, BFO, "samp_rate", "12000")


def repair_frequency_hack(start: dict) -> dict:
    """BFO left at 96 kHz but set to -6400 Hz, the same phase step per sample (-800 Hz at 12 kHz)."""
    return edit.set_param(start, BFO, "freq", "-6400")


def repair_bfo_minus_2200(start: dict) -> dict:
    """BFO at -2200 Hz at the 12 kHz rate: the beacon at -700 Hz complex, whose real part is 700 Hz."""
    return edit.set_param(edit.set_param(start, BFO, "samp_rate", "samp_rate/decim"), BFO, "freq", "-2200")


def repair_audio_rate_variable(start: dict) -> dict:
    """A new variable audio_rate = samp_rate/decim used as the BFO's Sample Rate."""
    g = edit.set_param(start, BFO, "samp_rate", "audio_rate")
    g["blocks"].insert(2, {"name": "audio_rate", "id": "variable", "parameters": {"value": "samp_rate/decim"}})
    return g


def repair_with_throttle(start: dict) -> dict:
    """The BFO fixed and a Throttle (96 kHz) inserted after the Add."""
    g = edit.set_param(start, BFO, "samp_rate", "samp_rate/decim")
    g["blocks"].append(C.block("blocks_throttle2_0", "blocks_throttle2", type="complex", samples_per_second="samp_rate"))
    g["connections"] = [c for c in g["connections"] if not (c[0] == ADD and c[2] == MIX)]
    g["connections"] += [[ADD, "0", "blocks_throttle2_0", "0"], ["blocks_throttle2_0", "0", MIX, "0"]]
    return g


def repair_with_audio_lpf(start: dict) -> dict:
    """The BFO fixed and an extra audio Low Pass Filter (float, 12 kHz, cutoff 1.5 kHz) before audio.dat."""
    g = edit.set_param(start, BFO, "samp_rate", "samp_rate/decim")
    g["blocks"].append(C.block("low_pass_filter_1", "low_pass_filter", type="fir_filter_fff", decim="1", gain="1",
                               samp_rate="samp_rate/decim", cutoff_freq="1500", width="500",
                               win="window.WIN_HAMMING", beta="6.76"))
    g["connections"] = [c for c in g["connections"] if not (c[0] == TO_REAL and c[2] == AU_SINK)]
    g["connections"] += [[TO_REAL, "0", "low_pass_filter_1", "0"], ["low_pass_filter_1", "0", AU_SINK, "0"]]
    return g


def repair_two_branch(start: dict) -> dict:
    """The BFO left at 96 kHz but moved in front of a second decimate-by-8 filter: the audio branch is
    taken from the 96 kHz mixer output, beaten down there, then filtered and decimated."""
    g = copy.deepcopy(start)
    g["blocks"].append(C.block("low_pass_filter_1", "low_pass_filter", type="fir_filter_ccf", decim="decim", gain="1",
                               samp_rate="samp_rate", cutoff_freq="3000", width="1000",
                               win="window.WIN_HAMMING", beta="6.76"))
    g["connections"] = [c for c in g["connections"] if not (c[0] == LPF and c[2] == BFO_MIX)
                        and not (c[0] == BFO_MIX and c[2] == TO_REAL)]
    g["connections"] += [[MIX, "0", BFO_MIX, "0"], [BFO_MIX, "0", "low_pass_filter_1", "0"],
                         ["low_pass_filter_1", "0", TO_REAL, "0"]]
    return g


def repair_fft_channel_filter(start: dict) -> dict:
    """The BFO fixed, and the Low Pass Filter replaced by a decimating FFT Filter with the same taps
    (another block on the channel-filter graph check's list)."""
    g = edit.set_param(start, BFO, "samp_rate", "samp_rate/decim")
    for b in g["blocks"]:
        if b["name"] == LPF:
            b["name"], b["id"] = "fft_filter_xxx_0", "fft_filter_xxx"
            b["parameters"] = {"type": "ccc", "decim": "decim", "taps": "firdes.low_pass(1.0, samp_rate, 3000, 1000)"}
    g["connections"] = [[("fft_filter_xxx_0" if x == LPF else x) for x in c] for c in g["connections"]]
    return g


def repair_xlating_bfo(start: dict) -> dict:
    """The BFO Signal Source and its Multiply replaced by a Frequency Xlating FIR Filter (taps [1.6],
    centre 800 Hz at the 12 kHz rate: a -800 Hz shift with gain 1.6)."""
    g = copy.deepcopy(start)
    g["blocks"] = [b for b in g["blocks"] if b["name"] not in (BFO, BFO_MIX)]
    g["blocks"].append(C.block("freq_xlating_fir_filter_xxx_0", "freq_xlating_fir_filter_xxx", type="ccf", decim="1",
                               taps="[1.6]", center_freq="800", samp_rate="samp_rate/decim"))
    g["connections"] = [c for c in g["connections"] if BFO not in (c[0], c[2]) and BFO_MIX not in (c[0], c[2])]
    g["connections"] += [[LPF, "0", "freq_xlating_fir_filter_xxx_0", "0"], ["freq_xlating_fir_filter_xxx_0", "0", TO_REAL, "0"]]
    return g


# ---- task record ---------------------------------------------------------------------------------
P_CH_FREQ = "channel.dat: 12 kHz complex channel (96 kHz recording decimated by 8) with the beacon at +1.5 kHz (+19.5 kHz tuned to 0 Hz)"
P_CH_AMP = "channel.dat: beacon amplitude 0.5 while keyed on (unity-gain tuning and filtering)"
P_CH_CLEAN = ("channel.dat: neighbouring carriers removed (>= 98 % of the channel power within 200 Hz of +1.5 kHz; "
              "below 5 % of the on level while the beacon is off)")
P_CH_FILTER = "channel: the recording is low-pass filtered (a filter block on a path from a Signal Source to a File Sink)"
P_AU_FREQ = "audio.dat: the beacon heard as a 700 Hz tone at the 12 kHz audio rate (within 1 %)"
P_AU_CLEAN = "audio.dat: clean tone (>= 98 % of the audio power within 200 Hz of 700 Hz)"
P_AU_AMP = "audio.dat: real-valued (float) audio with peak amplitude 0.8 (RMS 0.566) while the beacon is keyed on"
P_AU_KEYED = "audio.dat: silent (below 5 % of the on level) while the beacon is keyed off: the audio follows the beacon's keying"
P_OBS = "outputs: complex items to the File Sink channel.dat and float items to the File Sink audio.dat"
P_RUN = ("run condition: the flowgraph keeps producing samples until stopped (V5 liveness: at least "
         f"{RUN_ITEMS} items at both observation points)")
P2P_NOTE = ("p2p checks hold in the start graph (its fault only changes the audio frequency) and must still hold; "
            "f2p checks are the audio frequency and purity")

FILTER_IDS = ["low_pass_filter", "band_pass_filter", "fir_filter_xxx", "fft_filter_xxx", "filter_fft_low_pass_filter",
              "interp_fir_filter_xxx", "freq_xlating_fir_filter_xxx", "freq_xlating_fft_filter_ccc",
              "rational_resampler_xxx", "pfb_arb_resampler_xxx", "pfb_decimator_ccf", "pfb_channelizer_ccf",
              "pfb_channelizer_hier_ccf", "iir_filter_xxx", "single_pole_iir_filter_xx", "blocks_moving_average_xx",
              "blocks_integrate_xx", "root_raised_cosine_filter"]


def oracle() -> dict:
    ch = {"point": "channel", "skip": SKIP, "n": N}
    au = {"point": "audio", "skip": SKIP, "n": N}
    au_rms = round(AUDIO_PEAK / math.sqrt(2), 4)
    checks = [
        {**ch, "feature": "peak_freq", "fs": FS_OUT, "expect": CH_F, "tol": FREQ_REL * CH_F,
         "group": "p2p", "property": P_CH_FREQ},
        {**ch, "feature": "band_fraction", "fs": FS_OUT, "lo": CH_F - BAND_HALF, "hi": CH_F + BAND_HALF,
         "min": MIN_FRACTION, "group": "p2p", "property": P_CH_CLEAN},
        {**au, "feature": "peak_freq", "fs": FS_OUT, "expect": AUDIO_F, "tol": FREQ_REL * AUDIO_F,
         "group": "f2p", "property": P_AU_FREQ},
        {**au, "feature": "band_fraction", "fs": FS_OUT, "lo": AUDIO_F - BAND_HALF, "hi": AUDIO_F + BAND_HALF,
         "min": MIN_FRACTION, "group": "f2p", "property": P_AU_CLEAN},
    ]
    for s in ON_WINDOWS:
        checks += [
            {"point": "channel", "skip": s, "n": WIN, "feature": "rms", "expect": BEACON_A, "rel_tol": AMP_REL,
             "group": "p2p", "property": P_CH_AMP},
            {"point": "audio", "skip": s, "n": WIN, "feature": "rms", "expect": au_rms, "rel_tol": AMP_REL,
             "group": "p2p", "property": P_AU_AMP},
            {"point": "audio", "skip": s, "n": WIN, "feature": "max_abs", "expect": AUDIO_PEAK, "rel_tol": AMP_REL,
             "group": "p2p", "property": P_AU_AMP},
        ]
    for s in OFF_WINDOWS:
        checks += [
            {"point": "channel", "skip": s, "n": WIN, "feature": "rms", "max": round(OFF_REL * BEACON_A, 4),
             "group": "p2p", "property": P_CH_CLEAN},
            {"point": "audio", "skip": s, "n": WIN, "feature": "rms", "max": round(OFF_REL * au_rms, 4),
             "group": "p2p", "property": P_AU_KEYED},
        ]
    graph_checks = [
        {"kind": "path", "from_ids": ["analog_sig_source_x"], "via_ids": FILTER_IDS, "to_ids": ["blocks_file_sink"],
         "group": "p2p", "property": P_CH_FILTER},
    ]
    return {"checks": checks, "graph_checks": graph_checks}


MUTATIONS = [
    {"op": "set_param", "block": LO, "param": "freq", "value": "-20000", "breaks": P_CH_FREQ,
     "note": "mistuned by 500 Hz: beacon at +1.0 kHz in the channel, 200 Hz in the audio"},
    {"op": "set_param", "block": LO, "param": "freq", "value": str(TUNE_F), "breaks": P_CH_FREQ,
     "note": "tuning sign error: the -17 kHz neighbour lands at +2.5 kHz, the beacon is filtered out"},
    {"op": "set_param", "block": LPF, "param": "cutoff_freq", "value": "6000", "breaks": P_CH_CLEAN,
     "note": "channel filter too wide: the neighbour at +5.5 kHz passes"},
    {"op": "bypass", "block": LPF, "breaks": P_CH_FREQ, "note": "missing filter and no decimation (96 kHz output)"},
    {"op": "set_param", "block": "decim", "param": "value", "value": "4", "breaks": P_CH_FREQ,
     "note": "wrong decimation: 24 kHz outputs"},
    {"op": "set_param", "block": "samp_rate", "param": "value", "value": "48000", "breaks": P_CH_FREQ,
     "note": "wrong recording sample rate"},
    {"op": "set_param", "block": BFO, "param": "samp_rate", "value": "samp_rate", "breaks": P_AU_FREQ,
     "note": "the injected fault: BFO at the 96 kHz rate behind the decimator"},
    {"op": "scale_param", "block": BFO, "param": "freq", "factor": 1.25, "breaks": P_AU_FREQ,
     "note": "BFO at -1 kHz: 500 Hz audio"},
    {"op": "set_param", "block": BFO, "param": "freq", "value": "-790", "breaks": P_AU_FREQ,
     "note": "near miss: BFO 10 Hz off, 710 Hz audio (700 Hz +-1 %)"},
    {"op": "set_param", "block": BFO, "param": "waveform", "value": "analog.GR_SQR_WAVE", "breaks": P_AU_CLEAN,
     "note": "square-wave BFO: DC term and harmonics in the audio"},
    {"op": "set_param", "block": BFO, "param": "amp", "value": "1.5", "breaks": P_AU_AMP,
     "note": "near miss: BFO amplitude 1.5, audio peak 0.75 (-6 %)"},
    {"op": "set_id", "block": TO_REAL, "new_id": "blocks_complex_to_mag", "breaks": P_AU_FREQ,
     "note": "envelope instead of the real part: the keying only, no beat note"},
    {"op": "rewire_input", "dst": CH_SINK, "dst_port": "0", "new_src": BFO_MIX, "breaks": P_CH_FREQ,
     "note": "wrong tap: channel.dat taken after the BFO (beacon at 700 Hz instead of +1.5 kHz)"},
    {"op": "set_param", "block": KEY, "param": "waveform", "value": "analog.GR_CONST_WAVE", "breaks": P_AU_KEYED,
     "note": "key held on: an unkeyed 700 Hz carrier, never silent (what a free-running 700 Hz oscillator gives)"},
    {"op": "set_param", "block": KEY, "param": "freq", "value": "10", "breaks": P_AU_KEYED,
     "note": "keying at 10 Hz: the beacon is on during the stated off periods"},
]

REVIEW_NOTES = (
    "Version 2: round-1 fixes (resolution in reports/tier_f_reviews/round1/gbf-test-18.resolution.json); the beacon "
    "is now keyed by a 5 Hz square wave and V6 checks the keyed levels. Composed L2 item: the approved plan says L2 "
    "tasks come from converted examples and eval plan 4.3 lists only hand-written L1 chains, so composed L2 items "
    "need the team's sign-off (fix_plan team_decisions); the brief's seed qt-gui/qtgui_multi_input.grc was not used "
    "(single-rate, no place for a rate fault). Known oracle limits: (1) the recording is generated inside the graph "
    "the model edits; editing the recording itself (e.g. zeroing the neighbours while keeping a filter) is not "
    "detected; a graph that reproduces the keyed beacon tone with its own oscillator and a copy of the key would "
    "pass. (2) The level windows sit at least 400 items (33 ms) from each keying edge; a channel filter with a "
    "group delay above that (transition width ~100 Hz at 96 kHz) would fail them. (3) The channel-filter graph "
    "check accepts any filter block on a Signal Source -> File Sink path; the filter's quality is checked by the "
    "signal checks. Version 1's L2 count (13 blocks) became 15 with the keying; the audio gain moved into the BFO "
    "amplitude to stay within L2's 15-block limit."
)


def record(sig: tuple[str, ...]) -> dict:
    return {
        "id": TASK_ID, "tier": "F", "split": SPLIT, "version": VERSION, "canary": CANARY,
        "license": "CC-BY-4.0 (original GRC-Bench item; licence to be confirmed by the team)",
        "source": {"kind": "composed_chain", "ref": "|".join(sig)},
        "cluster_id": C.cluster_id(sig),
        "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring (round-1 fix stage)",
                       "generator_version": "0.2.0"},
        "review": {"status": "pending-team-review", "reviewers": [], "notes": REVIEW_NOTES},
        "family": "repair", "level": "L2",
        "instruction": INSTRUCTION,
        "start_grc": "start.grc",
        "reference_grc": "reference.grc",
        "observation": {
            "points": [{"name": "channel", "sink": {"block_id": "blocks_file_sink",
                                                    "params": {"file": CH_FILE, "type": "complex"}}},
                       {"name": "audio", "sink": {"block_id": "blocks_file_sink",
                                                  "params": {"file": AU_FILE, "type": "float"}}}],
            "run": {"n_items": RUN_ITEMS, "min_items": RUN_ITEMS, "timeout_s": 10.0},
            "contract": ("the complex samples written to the File Sink whose file name is channel.dat (the 12 kHz "
                         "channel) and the float samples written to the File Sink whose file name is audio.dat "
                         "(the 12 kHz audio)")},
        "oracle": oracle(),
        "mutations": MUTATIONS,
        "spec_properties": [P_CH_FREQ, P_CH_AMP, P_CH_CLEAN, P_CH_FILTER, P_AU_FREQ, P_AU_CLEAN, P_AU_AMP, P_AU_KEYED,
                            P_OBS, P_RUN, P2P_NOTE],
        "perturbation": {"perturbed": False, "changes": []},
        "fault": {"type": "rate_mismatch",
                  "description": (f"the beat-frequency oscillator {BFO} (Signal Source, -800 Hz), which runs behind "
                                  "the decimate-by-8 channel filter and so produces samples at 12 kHz, has its "
                                  "Sample Rate set to samp_rate (96 kHz) instead of samp_rate/decim; its shift acts "
                                  "as -100 Hz, so the beacon comes out of audio.dat at 1.4 kHz instead of 700 Hz. "
                                  "The graph validates and runs; only its behaviour is wrong."),
                  "expected_rung": "V6"},
        "disjointness": ("A composed CW-beacon receiver (keyed three-carrier test recording, tuning mixer, decimating "
                         "channel filter, beat-frequency oscillator, real-part audio) whose BFO runs at the "
                         "pre-decimation rate. The training-side rate tasks are a 48k-to-44.1k resampler ratio fix "
                         "(t20_resample_ratio_repair), a wideband-FM receiver rate/filter repair "
                         "(t17_wbfm_tune_repair), a QAM demo whose sample rates or types mismatch between blocks "
                         "(t05_qam_debug) and a 5G analyser's FFT-rate bottleneck (t12_fivegnalizer_modify); the FM "
                         "receiver repair (t03) is broadcast FM, and the GRC-Agent scenario 22_fm_rx_filter_squelch "
                         "adds a channel low-pass filter and squelch to an FM receiver (a scripted modify step, not a "
                         "rate repair). None is a CW/BFO receiver or a post-decimation "
                         "oscillator rate fault, and no training spec (dial tone, QPSK BER, AM demod, spectrogram, "
                         "...) keys a beacon (reports/reference/training_task_specs.json)."),
    }


def main() -> int:
    ref = compact(reference_full())
    sig = C.signature(ref)
    split = C.signature_split(sig)
    assert split == SPLIT, f"signature {sig} belongs to split {split}, not {SPLIT}"
    assert abs(FAULT_AUDIO_F - 1400) < 1e-9
    assert ON_WINDOWS == [1600, 4000] and OFF_WINDOWS == [2800, 5200], (ON_WINDOWS, OFF_WINDOWS)
    functional = [b for b in ref["blocks"] if b["id"] not in ("variable", "import")]
    assert len(functional) == 15, len(functional)
    start = start_from(ref)
    variants = {
        "variants/v1_rotator_bfo_blackman.grc": compact(variant_rotator_bfo()),
        "variants/v2_xlating_filter_bfo_gain.grc": compact(variant_xlating_filter()),
        "variants/v3_rotator_tune_conjugate_bfo.grc": compact(variant_rotator_tune_conjugate_bfo()),
        "variants/v4_repair_literal_12000.grc": repair_literal_rate(start),
        "variants/v5_repair_freq_minus_6400_at_96k.grc": repair_frequency_hack(start),
        "variants/v6_repair_bfo_minus_2200.grc": repair_bfo_minus_2200(start),
        "variants/v7_repair_audio_rate_variable.grc": repair_audio_rate_variable(start),
        "variants/v8_repair_plus_throttle.grc": repair_with_throttle(start),
        "variants/v9_repair_plus_audio_lpf.grc": repair_with_audio_lpf(start),
        "variants/v10_repair_two_branch_bfo_at_96k.grc": repair_two_branch(start),
        "variants/v11_repair_xlating_fir_as_bfo.grc": repair_xlating_bfo(start),
        "variants/v12_repair_fft_channel_filter.grc": repair_fft_channel_filter(start),
    }
    files = {"reference.grc": grcfile.dump(ref), "start.grc": grcfile.dump(start)}
    print(f"reference  v2 signature {'|'.join(sig)} -> {split}")
    ss = C.signature(start)
    print(f"start      v2 signature {'|'.join(ss)} -> {C.signature_split(ss)}")
    for name, g in variants.items():
        vs = C.signature(g)
        print(f"{name:48s} v2 signature {'|'.join(vs)} -> {C.signature_split(vs)}")
        files[name] = grcfile.dump(g)
    errors = taskio.save(HERE, record(sig), files)
    if errors:
        print("NOT WRITTEN:", *errors, sep="\n  ")
        return 1
    for stale in sorted((HERE / "variants").glob("*.grc")):   # keep the directory equal to `files`
        if f"variants/{stale.name}" not in files:
            stale.unlink()
    print(f"written {HERE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
