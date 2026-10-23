"""Authoring script for GRC-Bench Tier-F item gbf-test-13 (build, L2, composed graph).

Re-themed on 18 Oct 2026 (fix plan group G6). The first version (an anti-aliasing decimation demo)
shared the signature combine|filter|resample|tone, which hashes to train under signature v2, with
gbf-test-20; both were re-themed. This version: build a baseband-to-IF data transmitter.

Regenerates task.json, reference.grc and variants/*.grc deterministically (no randomness; the inputs
are the constants below, the pinned GNU Radio 3.10.12 block catalog, used to drop default-valued
parameters, and GNU Radio's firdes.low_pass, used for the reference IF waveform). Run inside WSL from
the repository root:

    PYTHONPATH=src QT_QPA_PLATFORM=offscreen ~/grc-agent/.venv/bin/python items/tier_f/test/gbf-test-13/build.py
    ./gb tier-f gates items/tier_f/test/gbf-test-13

Task. In the empty flowgraph (samp_rate only) the user asks for: a Vector Source repeating 32 listed
NRZ symbols (+-1, float) at 4000 symbols per second; the symbol stream itself written to symbols.dat
(float, one item per symbol); the symbols interpolated to 32 kHz (8 samples per symbol) with
low-pass pulse shaping and shifted up to a complex IF at exactly +8 kHz, written to if.dat (complex).
The instruction states the requirements, not a design:
  - if.dat carries the symbol stream at 4 kBd on the +8 kHz IF (any carrier phase);
  - >= 99 % of its power within +-3 kHz of the IF, < 0.5 % at negative frequencies;
  - RMS between 0.7 and 1.2 (unit symbols, pulse shaping with unity gain at DC, LO amplitude 1).
Reference: Vector Source -> Repeat 8 -> Low Pass Filter (2 kHz, 1 kHz wide) -> Float To Complex ->
Multiply by a complex 8 kHz Signal Source -> Throttle -> File Sink if.dat; Vector Source -> File Sink
symbols.dat. 9 functional blocks, a rate change and a modulation (upconversion): L2.

Why two observation points (calibrated with a numpy model of 21 designs, scratch study). A
correlation of if.dat with the reference waveform cannot tell a single wrong symbol (0.918-0.938) from
a valid alternative pulse shape (0.926-1.0), so the symbol sequence is checked exactly where it is
exact: equal_prefix on symbols.dat (first two periods, max_offset 0: the Vector Source starts with
the first symbol). The IF waveform check is then a loose phase-invariant corr_ref (|<ref, x>|, complex
reference, lags 0..256 = one full data period, so any filter delay is covered): every valid design
scores >= 0.93 (Repeat + low-pass 1.5-3 kHz, zero-stuffing low-pass with gain 8, Rational Resampler
with default taps, RRC alpha 0.2-0.5, Gaussian), while 8 kBd (interpolation 4) scores 0.57, the
reversed sequence 0.48, a 6 kHz or 8.01 kHz IF 0.21 / 0.19, a -8 kHz IF 0.0 and a real IF 0.71.
MIN_CORR = 0.8.

Level range. The RMS of a unity-DC-gain design depends on its pulse shape (study: 0.73 Gaussian
BT 0.3, 0.77 Repeat + 1.5 kHz low-pass, 0.85 reference, 0.92 Rational Resampler, 1.0 RRC, 1.08
zero-stuffing with a 2.5 kHz cutoff), so the instruction states a range, 0.7-1.2, that every such
design meets and that rejects the usual gain errors: a zero-stuffing interpolator without the gain of
8 (0.12), a filter gain of 8 after Repeat (6.8), an LO amplitude of 0.5 (0.43), a real IF (0.6).

Spectral checks (band_fraction on the complex signal: two-sided spectrum, unaffected by the 18 Oct
real-signal fix). The waveform is periodic with 256 samples (32 symbols x 8; the 8 kHz LO has period
4), so the 4096-sample window holds 16 periods and every spectral line sits on an FFT bin: no window
needed. Valid designs: >= 0.9975 within 5-11 kHz and <= 0.0002 below 0 Hz; rectangular pulses
(Repeat only) 0.898 and 0.039.
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "src"))

from gnuradio.filter import firdes  # noqa: E402

from grcbench.common.canary import CANARY  # noqa: E402
from grcbench.tier_f import compose, grc, grcfile, taskio  # noqa: E402
from grcbench.tier_f.compose import block, file_sink, graph, signature, signature_split  # noqa: E402

TASK_ID = "gbf-test-13"
SPLIT = "test"
GID = "if_data_tx"
FS = 32000
BAUD = 4000
SPS = FS // BAUD                     # 8
F_IF = 8000
IF_FILE, SYM_FILE = "if.dat", "symbols.dat"
SKIP, N = 2048, 4096                 # if.dat window: 16 data periods after the filter start-up
PERIOD = 32 * SPS                    # 256 samples
CORR_LEN, CORR_LAG = 4 * PERIOD, PERIOD
MIN_CORR = 0.8
RMS_MIN, RMS_MAX = 0.7, 1.2
IN_BAND, MAX_NEG = 0.99, 0.005
RUN_ITEMS = SKIP + N                 # 6144 items at each point (symbols.dat: 6144 symbols = 1.5 s at 4 kBd)


def m_sequence() -> list[int]:
    """31-chip maximal-length sequence (x^5 + x^3 + 1, register 10000), then one extra 0."""
    reg, out = [1, 0, 0, 0, 0], []
    for _ in range(31):
        out.append(reg[-1])
        reg = [reg[4] ^ reg[2]] + reg[:-1]
    return out + [0]


SYMBOLS = [1 if b else -1 for b in m_sequence()]
assert len(SYMBOLS) == 32 and sum(SYMBOLS) == 0
VECTOR = "(" + ", ".join(str(s) for s in SYMBOLS) + ")"
SYMBOL_TEXT = ", ".join(f"{s:+d}" for s in SYMBOLS)

# ---- flowgraphs -------------------------------------------------------------------------------
SRC, HOLD, SHAPE, TOC, LO, MIX, THR, IF_SINK, SYM_SINK = (
    "symbols", "hold", "pulse_shaping", "to_complex", "if_lo", "upconvert", "throttle", "if_sink", "symbol_sink")


def source(vector: str = VECTOR) -> dict:
    return block(SRC, "blocks_vector_source_x", type="float", vector=vector, tags="[]", repeat="True", vlen="1")


def lo(freq=F_IF, amp="1") -> dict:
    return compose.sig_source(LO, "complex", freq, amp)


def tx(chain: list[dict], *, upconvert: str = "multiply", throttle: bool = True, gid: str = GID,
       extra: list[dict] | None = None, variables: dict | None = None) -> dict:
    """Vector Source -> chain (float, ends at 32 kHz) -> Float To Complex -> upconversion -> [Throttle]
    -> if.dat; Vector Source -> symbols.dat."""
    blocks = list(extra or []) + [source()] + chain + [block(TOC, "blocks_float_to_complex", vlen="1")]
    names = [SRC] + [b["name"] for b in chain] + [TOC]
    conns = [[a, "0", b, "0"] for a, b in zip(names, names[1:])]
    if upconvert == "multiply":
        blocks += [lo(), block(MIX, "blocks_multiply_xx", type="complex", num_inputs="2", vlen="1")]
        conns += [[TOC, "0", MIX, "0"], [LO, "0", MIX, "1"]]
        last = MIX
    else:                                            # Rotator: phase increment per sample
        blocks += [block(MIX, "blocks_rotator_cc", phase_inc=upconvert, tag_inc_update="False")]
        conns += [[TOC, "0", MIX, "0"]]
        last = MIX
    if throttle:
        blocks.append(block(THR, "blocks_throttle2", type="complex", samples_per_second="samp_rate", vlen="1",
                            ignoretag="True", limit="auto", maximum="0.1"))
        conns.append([last, "0", THR, "0"])
        last = THR
    blocks += [file_sink(IF_SINK, "complex", IF_FILE), file_sink(SYM_SINK, "float", SYM_FILE)]
    conns += [[last, "0", IF_SINK, "0"], [SRC, "0", SYM_SINK, "0"]]
    return graph(gid, blocks, conns, {"samp_rate": FS, **(variables or {})})


def lpf(name=SHAPE, cutoff=2000, width=1000, gain="1", ftype="fir_filter_fff", interp="1", window=None) -> dict:
    b = block(name, "low_pass_filter", type=ftype, decim="1", interp=interp, gain=gain, samp_rate="samp_rate",
              cutoff_freq=cutoff, width=width, win=window or "window.WIN_HAMMING", beta="6.76")
    return b


reference = tx([block(HOLD, "blocks_repeat", type="float", interp=str(SPS), vlen="1"), lpf()])

# v1: Interpolating FIR Filter (zero-stuffing, firdes low-pass taps with gain 8) and a Rotator; no Throttle.
v1 = tx([block(SHAPE, "interp_fir_filter_xxx", type="fff", interp=str(SPS), samp_delay="0",
               taps="firdes.low_pass(8, samp_rate, 2000, 1000)")],
        upconvert="2*math.pi*8000/samp_rate", throttle=False, gid="if_tx_v1",
        extra=[block("import_0", "import", imports="import math")])
# v2: Rational Resampler 8/1 with GNU Radio's default taps (fractional bandwidth 0.4), Multiply by the LO.
v2 = tx([block(SHAPE, "rational_resampler_xxx", type="fff", interp=str(SPS), decim="1", taps="[]", fbw="0")],
        gid="if_tx_v2")
# v3: root-raised-cosine pulse shaping (RRC Filter block, interpolating, alpha 0.35, 11 symbols of taps).
v3 = tx([block(SHAPE, "root_raised_cosine_filter", type="interp_fir_filter_fff", decim="1", interp=str(SPS),
               gain=str(SPS), samp_rate="samp_rate", sym_rate=str(BAUD), alpha="0.35", ntaps=str(11 * SPS))],
        gid="if_tx_v3")
# v4: the Low Pass Filter block in interpolating mode (gain 8, Blackman window, 2.4 kHz cutoff); a Rotator
# with phase increment pi/2; no Throttle.
v4 = tx([lpf(cutoff=2400, width=1200, gain=str(SPS), ftype="interp_fir_filter_fff", interp=str(SPS),
             window="window.WIN_BLACKMAN")], upconvert="3.141592653589793/2", throttle=False, gid="if_tx_v4")

VARIANT_GRAPHS = {"v1_interp_fir_rotator.grc": v1, "v2_rational_resampler_default_taps.grc": v2,
                  "v3_rrc_pulse_shaping.grc": v3, "v4_interpolating_low_pass_rotator.grc": v4}
SIG = signature(reference)
assert signature_split(SIG) == SPLIT, (SIG, signature_split(SIG))
VARIANT_SIGS = {k: signature(v) for k, v in VARIANT_GRAPHS.items()}
for k, s in VARIANT_SIGS.items():
    assert signature_split(s) == SPLIT, (k, s, signature_split(s))


# ---- reference IF waveform for the corr_ref check ---------------------------------------------------
def reference_if(n: int) -> np.ndarray:
    """if.dat of the reference design in steady state (numpy): Repeat 8, firdes.low_pass(1, 32 kHz, 2 kHz,
    1 kHz, Hamming), times exp(j*2*pi*8 kHz*t). Periodic with 256 samples."""
    reps = 4 + n // PERIOD + 4
    base = np.convolve(np.repeat(np.tile(np.array(SYMBOLS, float), reps), SPS),
                       np.array(firdes.low_pass(1, FS, 2000, 1000)))[: reps * PERIOD]
    steady = base[2 * PERIOD:]                      # filter (105 taps) settled
    t = np.arange(len(steady))
    return (steady * np.exp(2j * np.pi * F_IF / FS * t))[:n]


_ref = reference_if(CORR_LEN + CORR_LAG)
REF_RE = [round(float(v), 4) for v in _ref.real]
REF_IM = [round(float(v), 4) for v in _ref.imag]

# ---- task record --------------------------------------------------------------------------------
INSTRUCTION = (
    "Build a small data transmitter in this flowgraph, which is empty apart from the samp_rate variable. Use a "
    "sample rate of 32 kHz (samp_rate = 32000).\n"
    "1. Symbols: a Vector Source (float) that repeats these 32 NRZ symbols, in this order and starting with the "
    f"first: {SYMBOL_TEXT}. The symbol rate is 4000 symbols per second.\n"
    f"2. Write the symbol stream itself, one float per symbol, to a File Sink with file name {SYM_FILE} (item "
    "type float).\n"
    "3. Interpolate the symbols to 32 kHz (8 samples per symbol) with low-pass pulse shaping, so that the "
    "transmitted spectrum stays compact.\n"
    "4. Shift the shaped signal up to a complex IF of exactly +8 kHz and write it to a File Sink with file name "
    f"{IF_FILE} (item type complex).\n\n"
    f"Requirements for {IF_FILE}, measured from sample 2048 on (the phase of the IF carrier does not matter):\n"
    f"- it carries the symbol stream of {SYM_FILE} at 4000 symbols per second on the +8 kHz IF;\n"
    "- at least 99 % of its power lies within ±3 kHz of the IF, i.e. between +5 kHz and +11 kHz;\n"
    "- less than 0.5 % of its power lies at negative frequencies (a complex IF, not a real one);\n"
    "- its RMS is between 0.7 and 1.2, which is what unit-amplitude symbols, pulse shaping with a gain of 1 at "
    "DC (a long run of equal symbols comes out at amplitude 1) and an IF oscillator of amplitude 1 give.\n\n"
    "The flowgraph should keep producing samples until it is stopped; do not limit the number of samples."
)

SPEC_PROPERTIES = [
    {"id": "symbols", "text": f"{SYM_FILE}: the 32 listed symbols, one float per symbol, repeating in order from the "
                              "first", "checked_by": "f2p equal_prefix of the first 64 items (two periods), max_offset 0"},
    {"id": "carries_data", "text": f"{IF_FILE} carries that symbol stream at 4000 symbols per second (8 samples per "
                                   "symbol at 32 kHz) on a complex IF of exactly +8 kHz, any carrier phase",
     "checked_by": (f"f2p corr_ref (phase-invariant |<ref, x>|, complex reference = the reference design's IF "
                    f"waveform, {CORR_LEN} samples, lags 0..{CORR_LAG} = one data period) >= {MIN_CORR}; valid pulse "
                    "shapes >= 0.93, wrong rate / IF / sequence <= 0.71")},
    {"id": "compact_spectrum", "text": f"at least 99 % of the power of {IF_FILE} between +5 kHz and +11 kHz",
     "checked_by": "f2p band_fraction (complex, two-sided) >= 0.99"},
    {"id": "complex_if", "text": f"less than 0.5 % of the power of {IF_FILE} at negative frequencies",
     "checked_by": "f2p band_fraction -16 kHz..-1 Hz <= 0.005"},
    {"id": "level", "text": f"RMS of {IF_FILE} between 0.7 and 1.2 (unit symbols, unity DC gain pulse shaping, LO "
                            "amplitude 1)", "checked_by": "f2p rms min 0.7 max 1.2"},
    {"id": "sample_rate", "text": "the signals are at 32 kHz (if.dat) and 4 kHz (symbols.dat)",
     "checked_by": "all if.dat checks assume 32 kHz (another rate moves the IF and the symbol length)"},
    {"id": "outputs", "text": f"complex items to the File Sink {IF_FILE}, float items to the File Sink {SYM_FILE}",
     "checked_by": "observation contract (file name and item type)"},
    {"id": "run", "text": "the flowgraph keeps producing samples until it is stopped",
     "checked_by": f"V4 liveness: at least {RUN_ITEMS} items at each observation point"},
]

W = {"point": "if", "skip": SKIP, "n": N}
CHECKS = [
    {"point": "sym", "feature": "equal_prefix", "ref": [float(s) for s in SYMBOLS * 2], "max_offset": 0, "n": 64,
     "expect": True, "group": "main", "property": "symbols"},
    {"point": "if", "feature": "corr_ref", "ref": REF_RE, "ref_imag": REF_IM, "signed": False, "max_lag": CORR_LAG,
     "skip": SKIP, "n": CORR_LEN + 2 * CORR_LAG, "min": MIN_CORR, "group": "main", "property": "carries_data"},
    {**W, "feature": "band_fraction", "fs": FS, "lo": F_IF - 3000, "hi": F_IF + 3000, "min": IN_BAND,
     "group": "main", "property": "compact_spectrum"},
    {**W, "feature": "band_fraction", "fs": FS, "lo": -FS / 2, "hi": -1, "max": MAX_NEG, "group": "main",
     "property": "complex_if"},
    {**W, "feature": "rms", "min": RMS_MIN, "max": RMS_MAX, "group": "main", "property": "level"},
]

DATA_FLIP = list(SYMBOLS)
DATA_FLIP[9] = -DATA_FLIP[9]
MUTATIONS = [
    {"op": "bypass", "block": SHAPE, "breaks": "compact_spectrum",
     "note": "rectangular pulses (Repeat only): 0.898 in band, 0.039 at negative frequencies"},
    {"op": "set_param", "block": SHAPE, "param": "cutoff_freq", "value": "4500", "breaks": "compact_spectrum",
     "note": "pulse shaping too wide: the NRZ sidelobes beyond 3 kHz pass"},
    {"op": "set_param", "block": SHAPE, "param": "gain", "value": str(SPS), "breaks": "level",
     "note": "interpolation gain applied after Repeat: RMS about 6.8"},
    {"op": "set_param", "block": LO, "param": "amp", "value": "0.5", "breaks": "level", "note": "RMS about 0.43"},
    {"op": "set_param", "block": LO, "param": "freq", "value": "6000", "breaks": "carries_data", "note": "IF 6 kHz"},
    {"op": "set_param", "block": LO, "param": "freq", "value": "8100", "breaks": "carries_data",
     "note": "IF 100 Hz off (the spectrum still fits the mask)"},
    {"op": "set_param", "block": LO, "param": "freq", "value": "-8000", "breaks": "complex_if, carries_data",
     "note": "IF at -8 kHz"},
    {"op": "rewire_input", "dst": THR, "dst_port": "0", "new_src": TOC, "breaks": "carries_data, compact_spectrum",
     "note": "no upconversion: the complex baseband goes to if.dat"},
    {"op": "set_param", "block": HOLD, "param": "interp", "value": "4", "breaks": "carries_data",
     "note": "4 samples per symbol: 8000 symbols per second"},
    {"op": "set_param", "block": SRC, "param": "vector", "value": "(" + ", ".join(str(s) for s in DATA_FLIP) + ")",
     "breaks": "symbols", "note": "one symbol (the 10th) copied with the wrong sign"},
    {"op": "set_param", "block": SRC, "param": "vector", "value": "(" + ", ".join(str(s) for s in SYMBOLS[::-1]) + ")",
     "breaks": "symbols, carries_data", "note": "symbols in reverse order"},
    {"op": "set_param", "block": "samp_rate", "param": "value", "value": "48000", "breaks": "sample_rate",
     "note": "designed for 48 kHz: read at 32 kHz the IF is 5.3 kHz"},
    {"op": "set_param", "block": SYM_SINK, "param": "file", "value": "syms.dat", "breaks": "outputs",
     "note": "symbol stream written to another file name"},
]

OBSERVATION = {
    "points": [{"name": "if", "sink": {"block_id": "blocks_file_sink", "params": {"file": IF_FILE, "type": "complex"}}},
               {"name": "sym", "sink": {"block_id": "blocks_file_sink", "params": {"file": SYM_FILE, "type": "float"}}}],
    "run": {"n_items": RUN_ITEMS, "min_items": RUN_ITEMS, "timeout_s": 10.0},
    "contract": (f"the complex samples (32 kHz) written to the File Sink whose file name is {IF_FILE} and the float "
                 f"items (one per symbol) written to the File Sink whose file name is {SYM_FILE}; {RUN_ITEMS} items "
                 "are read from each"),
}

RECORD = {
    "id": TASK_ID, "tier": "F", "split": SPLIT, "version": 2, "canary": CANARY,
    "license": "CC-BY-4.0 (original GRC-Bench item; licence to be confirmed by the team)",
    "source": {"kind": "composed_chain", "ref": "|".join(SIG)},
    "cluster_id": compose.cluster_id(SIG),
    "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring",
                   "generator_version": "0.2.0"},
    "review": {"status": "pending-team-review", "reviewers": [],
               "notes": ("Re-themed 18 Oct (fix plan G6): the earlier anti-aliasing decimation demo hashed to train under "
                         "signature v2. Composed L2 item: a deviation from 'L2 from converted examples' (approved plan "
                         "Phase 3) and from eval plan 4.3's source list, recorded for the team (fix_plan team_decisions). "
                         "Oracle limits: (1) the symbol sequence is checked exactly on symbols.dat; if.dat is checked by a "
                         "correlation >= 0.8 with the reference waveform, which tolerates any sensible pulse shape but "
                         "cannot see a single wrong symbol, so an answer that feeds the IF path from a second, mistyped "
                         "Vector Source would pass. (2) The level is a range (0.7-1.2) because the RMS of a correct "
                         "unity-DC-gain design depends on its pulse shape (0.73-1.08 measured). (3) The sample rate is "
                         "checked through the signals (the IF frequency and the symbol length assume 32 kHz), not as the "
                         "samp_rate variable. Signatures (v2): reference and variants v1, v2, v4 "
                         f"{'|'.join(SIG)} (test); the RRC variant v3 {'|'.join(VARIANT_SIGS['v3_rrc_pulse_shaping.grc'])} "
                         f"({signature_split(VARIANT_SIGS['v3_rrc_pulse_shaping.grc'])}); a Constellation Modulator "
                         "design would be data_source|linear_mod|mix|pulse_shaping|resample (train) and belongs on the "
                         "reserved list.")},
    "family": "build", "level": "L2",
    "instruction": INSTRUCTION,
    "start_grc": None, "reference_grc": "reference.grc",
    "observation": OBSERVATION,
    "oracle": {"checks": CHECKS, "graph_checks": []},
    "mutations": MUTATIONS,
    "spec_properties": SPEC_PROPERTIES,
    "perturbation": {"perturbed": False, "changes": []},
    "disjointness": ("Not a training-side spec: building a pulse-shaped NRZ data transmitter on a complex IF (interpolation, "
                     "low-pass pulse shaping, frequency shift). The nearest training specs are t02_qpsk_ber_chain (QPSK "
                     "modulator with decoding and BER), t21_gmsk_pipe_build (GMSK modulate/demodulate chain) and "
                     "t16_am_demod_build (AM modulate and demodulate audio); none upconverts a pulse-shaped symbol stream "
                     "to an IF or states an occupied-bandwidth mask. Within the test split it differs from gbf-test-20 "
                     "(baseband PAM mapper retune), gbf-test-19 (GMSK retune) and gbf-test-05 (BPSK demapper build)."),
    "authoring_notes": {
        "level": ("L2: 9 functional blocks (Vector Source, Repeat, Low Pass Filter, Float To Complex, Signal Source, "
                  "Multiply, Throttle, 2 File Sinks); interpolation 4 kHz -> 32 kHz and upconversion"),
        "signature_v2": "|".join(SIG),
        "signature_split": signature_split(SIG),
        "variant_signatures": {k: "|".join(v) for k, v in VARIANT_SIGS.items()},
        "design": "see build.py docstring",
        "reference_ifwave": "numpy model of the reference design (agrees with the GNU Radio run, corr 1.0)",
    },
}


def compact_text(data: dict) -> str:
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in data["blocks"]}
    small = grcfile.compact(data, defaults)
    diffs = grc.compact_equivalent(data, small)
    if diffs:
        raise SystemExit(f"compact form differs from the full form: {diffs}")
    return grcfile.dump(small)


def main() -> int:
    if (HERE / "variants").exists():
        shutil.rmtree(HERE / "variants")
    for stale in ("start.grc",):
        if (HERE / stale).exists():
            (HERE / stale).unlink()
    files = {"reference.grc": compact_text(reference)}
    files.update({f"variants/{k}": compact_text(v) for k, v in VARIANT_GRAPHS.items()})
    errors = taskio.save(HERE, RECORD, files)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"written {HERE} (signature {SIG} -> {signature_split(SIG)}; variants "
          f"{ {k: signature_split(v) for k, v in VARIANT_SIGS.items()} })")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
