"""Authoring script for GRC-Bench Tier-F item gbf-dev-01 (build, L1, composed chain), version 3.

Re-themed in round 1 (fix plan 18 Oct): the old dev-01 (16 dB tone gain) was a template twin of
gbf-test-04 in the {tone} cell, which gbf-test-04 keeps. New theme: a sync-word detector on a bit stream.

The planned design (fix plan brief) used GNU Radio's "Correlate Access Code - Bit"
(digital_correlate_access_code_bb), whose output byte carries the data bit plus a +2 flag. That block is
NOT in the pinned GNU Radio 3.10.12 GRC catalog (only the deprecated "- Tag" and "- Tag Stream"
variants are, and their detections are stream tags or variable-rate packet payloads, which are not a
per-bit output). A GRC-Agent answer can only use catalog blocks, so the item uses the planner's listed
alternative, the textbook correlator: an FIR filter whose taps are the time-reversed sync word mapped to
+-1, followed by a threshold. On 0/1 input the correlator output is w - d (w = 8 ones in the sync word,
d = Hamming distance of the last 16 bits to it), so 8 marks an error-free copy and 7 the one-bit-wrong
copy; a threshold at 7.5 marks only the error-free one.

    Vector Source (float 0/1, 64-bit pattern, repeat) --+--> File Sink bits.dat  (float)
                                                       +--> FIR (16 taps) --> Threshold 7.5 --> File Sink flags.dat (float)

The pattern holds the sync word 1110 1011 0010 0001 once (bits 8-23) and a copy with bit 9 flipped once
(bits 40-55); every other 16-bit window, including the windows across the period boundary and the
zero-history start, is at least 6 bits from the sync word and 5 bits from its reversal (checked below).
bits.dat is a second observation point so the stated input pattern is checked (a different input with
the same flags, e.g. without the near-miss copy and with a one-error threshold, would otherwise pass).
A path check (Vector Source -> block with memory -> File Sink) rejects a hard-coded flags pattern. Round 2:
the list holds every kind of block with memory (filters, Delay, correlator blocks, Embedded Python), so
delay-line correlators (Delay + Add/Subtract, Delay + Multiply, Delay + NOT/AND on bytes) pass; stateless
blocks are left out, so a stored flags pattern gated by the bit stream fails.

Signature v2: data_source|filter|nonlinear -> dev (cluster chain:data_source|filter|nonlinear); the
variants v1-v4 share it. Correct alternatives in other cells (Binary Slicer, delay-line designs,
Rational Resampler + Delay, Embedded Python) are kept as passing probes; build() prints their v2
signatures and writes them into review.notes, and ships every one that hashes to train as a further
variant (Rational Resampler + Delay), so the reserved-signature step includes it.

    PYTHONPATH=src ~/grc-agent/.venv/bin/python items/tier_f/dev/gbf-dev-01/build.py           # write
    PYTHONPATH=src ~/grc-agent/.venv/bin/python items/tier_f/dev/gbf-dev-01/build.py --probes  # check

Deterministic; needs GNU Radio (catalog defaults are used to write the compact files).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from grcbench.tier_f import compose as C          # noqa: E402
from grcbench.tier_f import grc, grcfile, taskio  # noqa: E402

TASK_ID = "gbf-dev-01"
SPLIT = "dev"
VERSION = 3

# ---- the specification (every number below is stated in the instruction) -----------------------
PATTERN_TXT = "0011 0110 1110 1011 0010 0001 1000 1001 1111 0001 1110 1011 0110 0001 1100 0010"
SYNC_TXT = "1110 1011 0010 0001"
BITS = [int(c) for c in PATTERN_TXT.replace(" ", "")]
SYNC = [int(c) for c in SYNC_TXT.replace(" ", "")]
P = len(BITS)                       # 64
L = len(SYNC)                       # 16
W = sum(SYNC)                       # 8 ones: correlator output for an error-free copy
NEAR_MISS_BIT = 9                   # the second copy has this bit of the sync word flipped
BITS_FILE, FLAGS_FILE = "bits.dat", "flags.dat"

TAPS = [2 * s - 1 for s in reversed(SYNC)]          # time-reversed sync word as +-1
THRESH = W - 0.5                                    # 7.5
N_ITEMS = 4096
REF_PERIODS = 3                                     # each equal_prefix compares 3 periods (192 items)
LATE_SKIP = N_ITEMS - REF_PERIODS * P               # 3904 = 61 periods: end of the capture, in phase

INSTRUCTION = (
    "In this empty flowgraph, build a sync-word detector for a test bit stream. Use a repeating Vector "
    "Source to play this 64-bit pattern, one sample per bit, leftmost bit first, repeated for as long as "
    f"the flowgraph runs (the spaces only group the bits): {PATTERN_TXT}. Record the bit stream in a "
    "File Sink with the file name bits.dat and item type float, as 0.0 for a 0 bit and 1.0 for a 1 bit. "
    f"The pattern contains the 16-bit sync word {SYNC_TXT} once, and also a copy of it with one wrong "
    "bit. Detect the sync word with no bit errors allowed, computing the detection inside the flowgraph "
    "from the bit stream (do not store the expected result as a fixed pattern). Write the result to a "
    "second File Sink with the file name flags.dat and item type float: one sample per input bit, exactly "
    "1.0 on the bit that completes an error-free sync word (its last bit) and exactly 0.0 on every other "
    "bit, so the copy with the wrong bit is never marked. Both files must start with the first bit of the pattern, "
    "with no delay between them. The sample rate does not affect the result, so you can leave the "
    "samp_rate variable as it is. The flowgraph should keep producing samples until it is stopped; do not "
    "limit the number of samples."
)

# Graph check (round 2, finding G1-dev01-delay-line-correlator-rejected): a flag depends on 16 bits at
# different times, so every design that computes it from the bit stream has a block WITH MEMORY on the
# path from the source to flags.dat: a filter (FIR, FFT, interpolating FIR, IIR, Rational Resampler,
# Moving Average), a Delay (delay-line correlators: Delay + Add/Subtract, Delay + Multiply, Delay + NOT/AND
# on bytes), a correlator block, or custom code. A stateless stage (Threshold, Add, Multiply, AND, ...)
# cannot see more than one bit, so it is not listed: listing it would let "bit stream x a stored flags
# pattern" pass (probe stored_flags_times_bits), which the instruction forbids. A hard-coded flags pattern
# has no block with memory on its path.
DETECTORS = ["fir_filter_xxx", "fft_filter_xxx", "interp_fir_filter_xxx", "iir_filter_xxx",
             "rational_resampler_xxx", "blocks_moving_average_xx", "blocks_delay", "digital_corr_est_cc",
             "digital_correlate_access_code_tag_xx", "epy_block"]


def _seq(xs) -> str:
    return "(" + ", ".join(str(x) for x in xs) + ")"


def _list(xs) -> str:
    return "[" + ", ".join(str(x) for x in xs) + "]"


def ideal(n: int) -> tuple[list[int], list[int]]:
    """Bit stream and ideal flags for the first n samples (zero history before the first bit)."""
    x = (BITS * (n // P + 2))[:n]
    flags = []
    for i in range(n):
        win = [x[j] if j >= 0 else 0 for j in range(i - L + 1, i + 1)]
        flags.append(1 if win == SYNC else 0)
    return x, flags


def _hamming(a, b) -> int:
    return sum(u != v for u, v in zip(a, b))


def check_pattern() -> None:
    """The pattern's stated properties, and the margins the oracle and mutants rely on."""
    assert P == 64 and L == 16 and W == 8
    pp = BITS + BITS
    d = [_hamming(pp[i:i + L], SYNC) for i in range(P)]           # cyclic windows, by start bit
    assert [i for i, v in enumerate(d) if v == 0] == [8], d       # one error-free copy (bits 8-23)
    assert [i for i, v in enumerate(d) if v == 1] == [40], d      # one copy with one wrong bit (40-55)
    near = BITS[40:56]
    assert [k for k in range(L) if near[k] != SYNC[k]] == [NEAR_MISS_BIT]
    assert sorted(d)[2] >= 6, sorted(d)[:4]                       # every other window >= 6 bits away
    assert min(_hamming(pp[i:i + L], SYNC[::-1]) for i in range(P)) >= 5
    _, flags = ideal(4 * P)
    assert all(flags[i] == flags[i % P] for i in range(4 * P)), "zero-history start differs"
    assert [i for i in range(P) if flags[i]] == [23]
    assert LATE_SKIP % P == 0


def compact(data: dict) -> dict:
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in data["blocks"]}
    out = grcfile.compact(data, defaults)
    out["options"] = data["options"]
    diffs = grc.compact_equivalent(data, out)
    assert not diffs, f"compact form differs from the full form: {diffs}"
    return out


def _vsrc(name: str, vector: str, typ: str = "float") -> dict:
    return C.block(name, "blocks_vector_source_x", type=typ, vector=vector, repeat="True")


def _thresh(name: str, low, high) -> dict:
    return C.block(name, "blocks_threshold_ff", low=str(low), high=str(high), init="0")


# ---- reference -----------------------------------------------------------------------------------
def reference_full() -> dict:
    blocks = [
        _vsrc("src", _seq(BITS)),
        C.file_sink("bits_out", "float", BITS_FILE),
        C.block("corr", "fir_filter_xxx", type="fff", decim="1", taps=_list(TAPS)),
        _thresh("det", THRESH, THRESH),
        C.file_sink("flags_out", "float", FLAGS_FILE),
    ]
    conns = [["src", "0", "bits_out", "0"], ["src", "0", "corr", "0"], ["corr", "0", "det", "0"],
             ["det", "0", "flags_out", "0"]]
    return C.graph("sync_word_detector", blocks, conns, {"samp_rate": "32000"})


# ---- equivalent variants (genuinely different correct solutions, same v2 cell) ------------------
def variant_pm1_mapping() -> dict:
    """Pattern and sync word as variables; the bits are mapped to +-1 (x2, -1) before an FIR whose taps
    are computed from the sync variable; perfect match = 16, one error = 14, threshold 15; a Throttle."""
    blocks = [
        _vsrc("bitsrc", "pattern"),
        C.block("thr", "blocks_throttle2", type="float", samples_per_second="samp_rate"),
        C.file_sink("bits_sink", "float", BITS_FILE),
        C.multiply_const("x2", "float", "2"),
        C.add_const("minus1", "float", "-1"),
        C.block("correlator", "fir_filter_xxx", type="fff", decim="1",
                taps="[2*s - 1 for s in reversed(sync)]"),
        _thresh("decide", 15, 15),
        C.file_sink("flag_sink", "float", FLAGS_FILE),
    ]
    conns = [["bitsrc", "0", "thr", "0"], ["thr", "0", "bits_sink", "0"], ["thr", "0", "x2", "0"],
             ["x2", "0", "minus1", "0"], ["minus1", "0", "correlator", "0"], ["correlator", "0", "decide", "0"],
             ["decide", "0", "flag_sink", "0"]]
    return C.graph("sync_detect_v1", blocks, conns,
                   {"samp_rate": "32000", "sync": _seq(SYNC), "pattern": _seq(BITS)})


def variant_fft_hysteresis() -> dict:
    """FFT filter correlator (output rounded only to float precision) and a threshold with hysteresis
    (on above 7.8, off below 7.2)."""
    blocks = [
        _vsrc("v", _list(BITS)),
        C.file_sink("b", "float", BITS_FILE),
        C.block("xc", "fft_filter_xxx", type="fff", decim="1", taps=_list(TAPS), nthreads="1"),
        _thresh("th", 7.2, 7.8),
        C.file_sink("f", "float", FLAGS_FILE),
    ]
    conns = [["v", "0", "b", "0"], ["v", "0", "xc", "0"], ["xc", "0", "th", "0"], ["th", "0", "f", "0"]]
    return C.graph("sync_detect_v2", blocks, conns, {"samp_rate": "32000"})


def variant_byte_source() -> dict:
    """GNU Radio's usual unpacked-bit convention: a byte Vector Source, converted once to float
    (UChar To Float) for both the recording and the correlator."""
    blocks = [
        _vsrc("bytes_src", _seq(BITS), typ="byte"),
        C.block("to_float", "blocks_uchar_to_float"),
        C.file_sink("bits_file", "float", BITS_FILE),
        C.block("fir", "fir_filter_xxx", type="fff", decim="1", taps=_list(TAPS)),
        _thresh("thr", THRESH, THRESH),
        C.file_sink("flags_file", "float", FLAGS_FILE),
    ]
    conns = [["bytes_src", "0", "to_float", "0"], ["to_float", "0", "bits_file", "0"], ["to_float", "0", "fir", "0"],
             ["fir", "0", "thr", "0"], ["thr", "0", "flags_file", "0"]]
    return C.graph("sync_detect_v3", blocks, conns, {"samp_rate": "32000"})


def variant_rail_decision() -> dict:
    """The decision made by arithmetic instead of a Threshold block: the correlator value minus 7,
    clamped to 0..1 by a Rail (8 -> 1, 7 or less -> 0; the correlator values are exact integers)."""
    blocks = [
        _vsrc("pattern_src", _list(BITS)),
        C.block("thr", "blocks_throttle2", type="float", samples_per_second="samp_rate"),
        C.file_sink("bits_rec", "float", BITS_FILE),
        C.block("sync_corr", "fir_filter_xxx", type="fff", decim="1", taps=_list(TAPS)),
        C.add_const("minus7", "float", str(-(W - 1))),
        C.block("clamp", "analog_rail_ff", lo="0", hi="1"),
        C.file_sink("flags_rec", "float", FLAGS_FILE),
    ]
    conns = [["pattern_src", "0", "thr", "0"], ["thr", "0", "bits_rec", "0"], ["thr", "0", "sync_corr", "0"],
             ["sync_corr", "0", "minus7", "0"], ["minus7", "0", "clamp", "0"], ["clamp", "0", "flags_rec", "0"]]
    return C.graph("sync_detect_v4", blocks, conns, {"samp_rate": "32000"})


VARIANTS = {
    "v1_pm1_mapping_variables.grc": variant_pm1_mapping,
    "v2_fft_correlator_hysteresis.grc": variant_fft_hysteresis,
    "v3_byte_source_uchar_to_float.grc": variant_byte_source,
    "v4_rail_clamp_decision.grc": variant_rail_decision,
}


# ---- probes: answers a single-edit mutant cannot express (expected verdicts) --------------------
def _lag_bit(k: int) -> int:
    """The sync-word bit that the input delayed by k samples must equal when the flag is raised."""
    return SYNC[L - 1 - k]


def _delay_line(gid: str, combiner: dict, src_type: str = "float", inverted: str | None = None,
                head: list[dict] | None = None, head_conns: list[list[str]] | None = None,
                tail: list[dict] | None = None, tail_conns: list[list[str]] | None = None) -> dict:
    """A delay-line correlator: the input delayed by 0..15 samples feeds input k of `combiner`; lags whose
    sync bit is 0 are taken from the stream `inverted` (when given) instead of from `src`."""
    blocks = [_vsrc("src", _seq(BITS), typ=src_type)] + (head or []) + [combiner]
    conns = [list(c) for c in head_conns or []]
    for k in range(L):
        node = inverted if (inverted and _lag_bit(k) == 0) else "src"
        if k:
            blocks.append(C.block(f"d{k}", "blocks_delay", type=src_type, delay=str(k)))
            conns.append([node, "0", f"d{k}", "0"])
            node = f"d{k}"
        conns.append([node, "0", combiner["name"], str(k)])
    return C.graph(gid, blocks + (tail or []), conns + [list(c) for c in tail_conns or []],
                   {"samp_rate": "32000"})


EPY_DETECTOR = '''import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """Sync-word detector: 1.0 on the last bit of every error-free copy of the word, else 0.0."""

    def __init__(self, word='1110101100100001'):
        gr.sync_block.__init__(self, name='Sync word detector', in_sig=[np.float32], out_sig=[np.float32])
        self.word = word
        self.w = np.array([int(c) for c in word], dtype=np.float32)
        self.set_history(len(self.w))

    def work(self, input_items, output_items):
        x, out = input_items[0], output_items[0]
        n = len(self.w)
        for i in range(len(out)):
            out[i] = 1.0 if np.array_equal(x[i:i + n], self.w) else 0.0
        return len(out)
'''


def alternative_probes() -> dict[str, tuple[dict, str]]:
    """Correct designs that are not shipped as variants (their v2 cell differs from the reference's):
    name -> (flowgraph, what it shows). All must pass V5; build() asserts that none hashes to train
    (a train-hash alternative would have to be shipped as a variant so the reserved list includes it)."""
    slicer = C.graph("probe_slicer", [
        _vsrc("src", _seq(BITS)),
        C.file_sink("bits_out", "float", BITS_FILE),
        C.block("corr", "fir_filter_xxx", type="fff", decim="1", taps=_list(TAPS)),
        C.add_const("bias", "float", -THRESH),
        C.block("slicer", "digital_binary_slicer_fb"),
        C.block("u2f", "blocks_uchar_to_float"),
        C.file_sink("flags_out", "float", FLAGS_FILE)],
        [["src", "0", "bits_out", "0"], ["src", "0", "corr", "0"], ["corr", "0", "bias", "0"],
         ["bias", "0", "slicer", "0"], ["slicer", "0", "u2f", "0"], ["u2f", "0", "flags_out", "0"]],
        {"samp_rate": "32000"})
    # the round-2 reviewer's delay-line correlator: lags with +1 taps summed, minus lags with -1 taps
    plus = [k for k in range(L) if _lag_bit(k) == 1]
    minus = [k for k in range(L) if _lag_bit(k) == 0]
    blocks = [_vsrc("src", _seq(BITS)), C.file_sink("bits_out", "float", BITS_FILE)]
    blocks += [C.block(f"d{k}", "blocks_delay", type="float", delay=str(k)) for k in range(1, L)]
    blocks += [C.block("addp", "blocks_add_xx", type="float", num_inputs=str(len(plus))),
               C.block("addn", "blocks_add_xx", type="float", num_inputs=str(len(minus))),
               C.block("diff", "blocks_sub_xx", type="float", num_inputs="2"),
               _thresh("det", THRESH, THRESH), C.file_sink("flags_out", "float", FLAGS_FILE)]
    conns = [["src", "0", "bits_out", "0"]] + [["src", "0", f"d{k}", "0"] for k in range(1, L)]
    conns += [["src" if k == 0 else f"d{k}", "0", "addp", str(i)] for i, k in enumerate(plus)]
    conns += [["src" if k == 0 else f"d{k}", "0", "addn", str(i)] for i, k in enumerate(minus)]
    conns += [["addp", "0", "diff", "0"], ["addn", "0", "diff", "1"], ["diff", "0", "det", "0"],
              ["det", "0", "flags_out", "0"]]
    add_sub = C.graph("probe_delay_line", blocks, conns, {"samp_rate": "32000"})
    # product of the 16 match indicators (x for a 1 bit, 1 - x for a 0 bit): exactly 1.0 or 0.0
    product = _delay_line(
        "probe_product", C.block("prod", "blocks_multiply_xx", type="float", num_inputs=str(L)), inverted="one_minus",
        head=[C.file_sink("bits_out", "float", BITS_FILE), C.multiply_const("negate", "float", "-1"),
              C.add_const("one_minus", "float", "1")],
        head_conns=[["src", "0", "bits_out", "0"], ["src", "0", "negate", "0"], ["negate", "0", "one_minus", "0"]],
        tail=[C.file_sink("flags_out", "float", FLAGS_FILE)], tail_conns=[["prod", "0", "flags_out", "0"]])
    # bitwise on unpacked bytes: NOT for the 0 bits, AND of the 16 lags, keep the LSB, convert to float
    bitwise = _delay_line(
        "probe_bitwise", C.block("match_all", "blocks_and_xx", type="byte", num_inputs=str(L)), src_type="byte",
        inverted="inv",
        head=[C.block("u2f", "blocks_uchar_to_float"), C.file_sink("bits_out", "float", BITS_FILE),
              C.block("inv", "blocks_not_xx", type="byte")],
        head_conns=[["src", "0", "u2f", "0"], ["u2f", "0", "bits_out", "0"], ["src", "0", "inv", "0"]],
        tail=[C.block("lsb", "blocks_and_const_xx", type="byte", const="1"), C.block("f2", "blocks_uchar_to_float"),
              C.file_sink("flags_out", "float", FLAGS_FILE)],
        tail_conns=[["match_all", "0", "lsb", "0"], ["lsb", "0", "f2", "0"], ["f2", "0", "flags_out", "0"]])
    # a Rational Resampler 1:1 outputs the window x[n..n+15] (advanced by ntaps - 1 = 15), so a Delay of 15
    # restores the alignment
    rr_delay = C.graph("probe_rr_delay", [
        _vsrc("src", _seq(BITS)), C.file_sink("bits_out", "float", BITS_FILE),
        C.block("corr", "rational_resampler_xxx", type="fff", interp="1", decim="1", taps=_list(TAPS), fbw="0"),
        C.block("align", "blocks_delay", type="float", delay=str(L - 1)),
        _thresh("det", THRESH, THRESH), C.file_sink("flags_out", "float", FLAGS_FILE)],
        [["src", "0", "bits_out", "0"], ["src", "0", "corr", "0"], ["corr", "0", "align", "0"],
         ["align", "0", "det", "0"], ["det", "0", "flags_out", "0"]], {"samp_rate": "32000"})
    epy = C.graph("probe_epy", [
        _vsrc("src", _seq(BITS)), C.file_sink("bits_out", "float", BITS_FILE),
        C.block("det", "epy_block", word="'" + "".join(map(str, SYNC)) + "'", _source_code=EPY_DETECTOR),
        C.file_sink("flags_out", "float", FLAGS_FILE)],
        [["src", "0", "bits_out", "0"], ["src", "0", "det", "0"], ["det", "0", "flags_out", "0"]],
        {"samp_rate": "32000"})
    return {
        "binary_slicer_threshold": (slicer, "FIR correlator, bias -7.5, Binary Slicer, UChar To Float"),
        "delay_line_add_sub": (add_sub, "round-2 reviewer's design: Delay blocks 1..15, two 8-input Adds "
                               "(+1 and -1 lags), Subtract, Threshold 7.5"),
        "delay_line_product": (product, "Delay blocks, x or 1 - x per lag, 16-input Multiply (no threshold)"),
        "delay_line_bitwise_bytes": (bitwise, "byte stream: Delay blocks, NOT for the 0 bits, 16-input AND, "
                                     "AND Const 1, UChar To Float"),
        "rational_resampler_delay": (rr_delay, "Rational Resampler 1:1 with the correlator taps, Delay 15 "
                                     "(the resampler's output is 15 samples early), Threshold"),
        "embedded_python": (epy, "Embedded Python block comparing the last 16 bits with the word"),
    }


def probes() -> dict[str, tuple[dict, bool, str, bool]]:
    """name -> (flowgraph, expected to pass V5, what it shows, write in compact form)."""
    flags_period = ideal(P)[1]
    bb = C.graph("probe_bb", [
        _vsrc("src", _seq(BITS), typ="byte"),
        C.block("cac", "digital_correlate_access_code_bb", access_code="'" + "".join(map(str, SYNC)) + "'",
                threshold="0"),
        C.file_sink("out", "byte", FLAGS_FILE)],
        [["src", "0", "cac", "0"], ["cac", "0", "out", "0"]], {"samp_rate": "32000"})
    iir = C.graph("probe_iir", [
        _vsrc("src", _seq(BITS)),
        C.file_sink("bits_out", "float", BITS_FILE),
        C.block("corr", "iir_filter_xxx", type="ffd", fftaps=_list(TAPS), fbtaps="[1]", oldstyle="True"),
        _thresh("det", THRESH, THRESH),
        C.file_sink("flags_out", "float", FLAGS_FILE)],
        [["src", "0", "bits_out", "0"], ["src", "0", "corr", "0"], ["corr", "0", "det", "0"],
         ["det", "0", "flags_out", "0"]], {"samp_rate": "32000"})
    hard = C.graph("probe_hardcoded", [
        _vsrc("src", _seq(BITS)), C.file_sink("bits_out", "float", BITS_FILE),
        _vsrc("fake", _seq(flags_period)), C.file_sink("flags_out", "float", FLAGS_FILE)],
        [["src", "0", "bits_out", "0"], ["fake", "0", "flags_out", "0"]], {"samp_rate": "32000"})
    other = list(BITS)
    other[40:56] = [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0]   # near-miss copy replaced
    lenient = reference_full()
    for b in lenient["blocks"]:
        if b["name"] == "src":
            b["parameters"]["vector"] = _seq(other)
        if b["name"] == "det":
            b["parameters"]["high"] = str(W - 1.5)
    stored = C.graph("probe_stored", [
        _vsrc("src", _seq(BITS)), C.file_sink("bits_out", "float", BITS_FILE),
        _vsrc("expected", _seq(flags_period)), C.block("gate", "blocks_multiply_xx", type="float", num_inputs="2"),
        C.file_sink("flags_out", "float", FLAGS_FILE)],
        [["src", "0", "bits_out", "0"], ["src", "0", "gate", "0"], ["expected", "0", "gate", "1"],
         ["gate", "0", "flags_out", "0"]], {"samp_rate": "32000"})
    fft_rail = C.graph("probe_fft_rail", [
        _vsrc("src", _seq(BITS)), C.file_sink("bits_out", "float", BITS_FILE),
        C.block("corr", "fft_filter_xxx", type="fff", decim="1", taps=_list(TAPS), nthreads="1"),
        C.add_const("minus7", "float", str(-(W - 1))), C.block("clamp", "analog_rail_ff", lo="0", hi="1"),
        C.file_sink("flags_out", "float", FLAGS_FILE)],
        [["src", "0", "bits_out", "0"], ["src", "0", "corr", "0"], ["corr", "0", "minus7", "0"],
         ["minus7", "0", "clamp", "0"], ["clamp", "0", "flags_out", "0"]], {"samp_rate": "32000"})
    rr = C.graph("probe_rr", [
        _vsrc("src", _seq(BITS)), C.file_sink("bits_out", "float", BITS_FILE),
        C.block("corr", "rational_resampler_xxx", type="fff", interp="1", decim="1", taps=_list(TAPS), fbw="0"),
        _thresh("det", THRESH, THRESH), C.file_sink("flags_out", "float", FLAGS_FILE)],
        [["src", "0", "bits_out", "0"], ["src", "0", "corr", "0"], ["corr", "0", "det", "0"],
         ["det", "0", "flags_out", "0"]], {"samp_rate": "32000"})
    out = {
        "access_code_bb_block": (bb, False, "GNU Radio 3.7-style Correlate Access Code - Bit "
                                 "(digital_correlate_access_code_bb): not in the pinned 3.10.12 catalog, "
                                 "fails at V2 (why the item does not build on it)", False),
        "iir_as_fir_correlator": (iir, True, "correct alternative: IIR filter with feedback taps [1] used as "
                                  "the correlator (same v2 cell as the reference)", True),
        "hardcoded_flags": (hard, False, "flags written from a fixed Vector Source: right samples, but no "
                            "block with memory on the path (graph check)", True),
        "stored_flags_times_bits": (stored, False, "bit stream multiplied by a stored copy of the expected "
                                    "flags: right samples, but only a stateless block on the path (why Add, "
                                    "Multiply, AND, ... are not in the graph check's list)", True),
        "other_input_lenient": (lenient, False, "input without the one-bit-wrong copy and a detector that "
                                "allows one error: flags.dat is identical, bits.dat is not", True),
        "fft_correlator_rail_inexact": (fft_rail, False, "FFT correlator, Add Const -7, Rail 0..1: the FFT's "
                                        "rounding (about 1e-6) reaches flags.dat, so values such as 2.4e-07 and "
                                        "0.999999 appear instead of exactly 0.0 and 1.0", True),
        "rational_resampler_unaligned": (rr, False, "Rational Resampler 1:1 with the correlator taps and no "
                                         "Delay: its output is 15 samples early, so the flags are misaligned", True),
    }
    for name, (g, why) in alternative_probes().items():
        out[name] = (g, True, "correct alternative not shipped (v2 cell differs from the reference): " + why,
                     not _has_epy(g))
    return out


def _has_epy(g: dict) -> bool:
    return any(b["id"] == "epy_block" for b in g["blocks"])


def _form(g: dict) -> dict:
    """Compact form for catalog blocks; an Embedded Python block is kept in full form."""
    return g if _has_epy(g) else compact(g)


def alternative_signatures() -> dict[str, tuple[str, ...]]:
    return {name: C.signature(_form(g)) for name, (g, _) in alternative_probes().items()}


# ---- task record ---------------------------------------------------------------------------------
def record(sig: tuple[str, ...], alt_sigs: dict[str, tuple[str, ...]]) -> dict:
    bits_ref, flags_ref = ideal(REF_PERIODS * P)
    bits_late, flags_late = ideal(LATE_SKIP + REF_PERIODS * P)
    assert bits_late[LATE_SKIP:] == bits_ref and flags_late[LATE_SKIP:] == flags_ref
    bits_req = (f"bit stream: bits.dat holds the stated 64-bit pattern as float 0.0/1.0, from its first "
                f"sample, repeated without slips (samples 0-191 and {LATE_SKIP}-{N_ITEMS - 1} compared exactly)")
    flags_req = ("detection: flags.dat is 1.0 exactly on the last bit of every error-free sync word "
                 f"(sample 23 of each 64-bit period) and 0.0 elsewhere, aligned with bits.dat from the first "
                 f"sample; the one-bit-wrong copy is not marked (samples 0-191 and {LATE_SKIP}-{N_ITEMS - 1} "
                 "compared exactly)")
    notes = (
        "Re-themed in round 1 (fix plan 18 Oct, group G1): old dev-01 (16 dB tone gain) shared the {tone} "
        "cell with gbf-test-04, which keeps it. The planned block, Correlate Access Code - Bit "
        "(digital_correlate_access_code_bb, data bit + 2 x flag per byte), is not in the pinned GNU Radio "
        "3.10.12 GRC catalog (probe access_code_bb_block fails V2), so the item uses the planner's listed "
        "alternative (FIR correlator + threshold) with a float 0/1 flag output; the v2 cell is "
        "data_source|filter|nonlinear (dev; the planner's pre-check listed it as dev and unused). bits.dat "
        "is a second observation point so the stated input pattern is checked. Graph check (round 2): a "
        "path from a Vector Source through a block with memory (a filter, a Delay, a correlator block or "
        "an Embedded Python block) to a File Sink. A flag depends on 16 bits at different times, so every "
        "design that computes it from the bit stream has such a block on the path, including delay-line "
        "correlators built from Delay blocks with Add/Subtract, Multiply or byte NOT/AND. Stateless blocks "
        "(Add, Multiply, AND, Threshold, ...) are not in the list on purpose: with them, the bit stream "
        "multiplied by a stored copy of the expected flags would pass (probe stored_flags_times_bits). "
        "Known limits: (1) a deliberate cheat that routes a hard-coded flags pattern through a block with "
        "memory (e.g. a one-tap FIR or a Delay of 0) passes the path check; (2) 'do not limit the number "
        "of samples' is enforced only as V4 liveness (4096 items at both points); (3) samples 192-3903 are "
        "not compared (a periodic source cannot deviate there without also failing the late window or the "
        "slip check, but an adversarial time-varying design could); (4) the stream holds one near-miss copy "
        "(bit 9 wrong), so a detector that checks only part of the word including bit 9 (e.g. only its "
        "last 7 bits, which occur nowhere else in the pattern) writes the same flags. The flags must be "
        "exactly 0.0 and 1.0 (stated in the instruction): an FFT correlator whose rounding (about 1e-6) "
        "reaches flags.dat through arithmetic (Add Const + Rail) fails (probe fft_correlator_rail_inexact), "
        "while an FFT correlator followed by a Threshold (v2) or a Binary Slicer writes exact flags. A "
        "Rational Resampler at 1:1 outputs its window 15 samples early, so it needs a Delay of 15 (probes "
        "rational_resampler_unaligned and rational_resampler_delay). Relation to gbf-dev-03 (same fixer "
        "group): both filter float data, but dev-03 is a symbol-length rectangular matched filter on +-1 NRZ "
        "samples at 8 samples per symbol (modify), while this item detects a specific 16-bit word in a "
        "1-sample-per-bit 0/1 stream with a decision threshold (build). Shipped variants use a Threshold "
        "with hysteresis, a +-1 mapping, a byte source, or a Rail clamp as the decision stage (v4), all in "
        "the dev cell. Correct alternatives whose v2 cell differs are kept as passing probes in build.py "
        "(--probes); their v2 signatures, for the reserved-signature step (R1/R2): "
        + "; ".join(f"{k}: {'|'.join(v)} ({C.signature_split(v)})" for k, v in alt_sigs.items())
        + ". Those that hash to train are also shipped as variants (v5 onwards), so the reserved list "
        "includes them. The Embedded Python cell custom|data_source is also the cell of gbf-dev-03's "
        "Embedded Python alternative; both items are dev."
    )
    return {
        "id": TASK_ID, "tier": "F", "split": SPLIT, "version": VERSION,
        "license": "CC-BY-4.0 (original GRC-Bench item; licence to be confirmed by the team)",
        "source": {"kind": "composed_chain", "ref": "|".join(sig)},
        "cluster_id": C.cluster_id(sig),
        "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring (round-1 re-theme)",
                       "generator_version": "0.2.0"},
        "review": {"status": "pending-team-review", "reviewers": [], "notes": notes},
        "family": "build", "level": "L1",
        "instruction": INSTRUCTION,
        "start_grc": None,
        "reference_grc": "reference.grc",
        "observation": {
            "points": [
                {"name": "bits", "sink": {"block_id": "blocks_file_sink",
                                          "params": {"file": BITS_FILE, "type": "float"}}},
                {"name": "flags", "sink": {"block_id": "blocks_file_sink",
                                           "params": {"file": FLAGS_FILE, "type": "float"}}},
            ],
            "run": {"n_items": N_ITEMS, "min_items": N_ITEMS, "timeout_s": 10.0},
            "contract": "the float samples written to the File Sinks whose file names are bits.dat and flags.dat "
                        "(item type float)",
        },
        "oracle": {
            "checks": [
                {"point": "bits", "feature": "equal_prefix", "ref": bits_ref, "max_offset": 0,
                 "group": "main", "property": "bit stream", "requires": [bits_req]},
                {"point": "bits", "feature": "equal_prefix", "ref": bits_ref, "max_offset": 0, "skip": LATE_SKIP,
                 "group": "main", "property": "bit stream", "requires": [bits_req]},
                {"point": "flags", "feature": "equal_prefix", "ref": flags_ref, "max_offset": 0,
                 "group": "main", "property": "detection", "requires": [flags_req]},
                {"point": "flags", "feature": "equal_prefix", "ref": flags_ref, "max_offset": 0, "skip": LATE_SKIP,
                 "group": "main", "property": "detection", "requires": [flags_req]},
            ],
            "graph_checks": [
                {"kind": "path", "from_ids": ["blocks_vector_source_x"], "to_ids": ["blocks_file_sink"],
                 "via_ids": DETECTORS, "group": "main", "property": "detection computed in the flowgraph",
                 "requires": ["the detection is computed from the bit stream: a path from a Vector Source "
                              "through a block with memory, i.e. a filter, a Delay, a correlator block or "
                              "custom code (" + ", ".join(DETECTORS) + "), to a File Sink"]},
            ],
        },
        "spec_properties": [
            bits_req,
            flags_req,
            "no bit errors allowed: the copy with bit 9 of the sync word flipped (bits 40-55 of the period) "
            "gives correlator value 7 of 8 and must stay 0.0 (a one-error threshold marks it on sample 55)",
            "detection computed in the flowgraph: graph check path Vector Source -> block with memory "
            "(filter, Delay, correlator block or Embedded Python) -> File Sink (a fixed flags pattern, alone "
            "or gated by the bit stream through a stateless block, fails)",
            "observation points: File Sinks bits.dat and flags.dat with item type float (sinks of another "
            "type do not match)",
            "sample rate: stated as irrelevant; not checked",
            "run condition: the flowgraph keeps producing samples; checked only as V4 liveness (4096 items "
            "at both observation points)",
        ],
        "mutations": [
            {"op": "set_param", "block": "det", "param": "high", "value": str(W - 1.5),
             "breaks": "detection (one bit error allowed: the near-miss copy is marked)"},
            {"op": "set_param", "block": "det", "param": "high", "value": str(W),
             "breaks": "detection (threshold equal to the perfect-match value: nothing is marked)"},
            {"op": "set_param", "block": "corr", "param": "taps", "value": _list([2 * s - 1 for s in SYNC]),
             "breaks": "detection (taps not time-reversed)"},
            {"op": "set_param", "block": "corr", "param": "taps",
             "value": _list([2 * s - 1 for s in reversed([b ^ (k == NEAR_MISS_BIT) for k, b in enumerate(SYNC)])]),
             "breaks": "detection (wrong sync word: bit 9 flipped, the near-miss copy is marked instead)"},
            {"op": "set_param", "block": "corr", "param": "taps", "value": _list([0] + TAPS),
             "breaks": "detection (flag one bit late, on the bit after the sync word)"},
            {"op": "bypass", "block": "det", "breaks": "detection (raw correlator values written, not 0/1)"},
            {"op": "rewire_input", "dst": "flags_out", "dst_port": "0", "new_src": "src",
             "breaks": "detection (detector bypassed: flags.dat records the raw bits)"},
            {"op": "set_param", "block": "corr", "param": "decim", "value": "2",
             "breaks": "detection (one flag sample per two input bits: wrong rate)"},
            {"op": "set_param", "block": "src", "param": "vector",
             "value": _seq([b ^ (k == 3) for k, b in enumerate(BITS)]),
             "breaks": "bit stream (one bit of the pattern changed)"},
            {"op": "set_param", "block": "src", "param": "vector", "value": _seq(BITS[:40] + SYNC + BITS[56:]),
             "breaks": "bit stream (near-miss copy corrected: two error-free sync words)"},
            {"op": "rewire_input", "dst": "bits_out", "dst_port": "0", "new_src": "det",
             "breaks": "bit stream (bits.dat records the flags)"},
        ],
        "perturbation": {"perturbed": False, "changes": []},
        "disjointness": (
            "Sync-word (frame-sync) detection by correlation on a stated test bit stream, with an exact "
            "no-error threshold. Nearest training-side specs (reports/reference/training_task_specs.json): "
            "t18 PSK31 monitor (demodulates and writes a decoded stream), t02 QPSK BER chain (modulate, "
            "decode, compare with the source), t21 GMSK pipe and t19 RSSI meter (moving-average level "
            "meter); none searches a bit stream for a sync word or thresholds a correlator. Within "
            "GRC-Bench: distinct from gbf-dev-03 (rectangular matched filter on NRZ samples), gbf-test-05 "
            "(BPSK hard-decision demapper, data_source|linear_demod) and gbf-dev-10 (seed receiver with "
            "Correlation Estimator)."),
    }


def build() -> tuple[dict, dict[str, str], tuple[str, ...]]:
    check_pattern()
    ref = compact(reference_full())
    sig = C.signature(ref)
    split = C.signature_split(sig)
    assert split == SPLIT, f"v2 signature {sig} belongs to split {split}, not {SPLIT}"
    files = {"reference.grc": grcfile.dump(ref)}
    print(f"reference: {'|'.join(sig)} -> {split}")
    for name, fn in VARIANTS.items():
        v = compact(fn())
        vs = C.signature(v)
        print(f"  {name}: {'|'.join(vs)} -> {C.signature_split(vs)}")
        assert vs == sig, f"variant {name} leaves the cell: {vs}"
        files[f"variants/{name}"] = grcfile.dump(v)
    alt_sigs = alternative_signatures()
    alts = alternative_probes()
    n = len(VARIANTS)
    for name, s in alt_sigs.items():
        print(f"  alternative probe {name}: {'|'.join(s)} -> {C.signature_split(s)}")
        assert s != sig, f"alternative {name} is in the reference cell: ship it as a variant"
        if C.signature_split(s) == "train":     # shipped, so the reserved-signature step (R1/R2) includes it
            n += 1
            files[f"variants/v{n}_{name}.grc"] = grcfile.dump(_form(alts[name][0]))
    return record(sig, alt_sigs), files, sig


def run_probes() -> int:
    from grcbench.tier_f.ladder import check
    task = taskio.load(HERE)
    bad = 0
    for name, (g, expect, why, use_compact) in probes().items():
        form = compact(g) if use_compact else g
        text = grcfile.dump(form)
        if use_compact or _has_epy(g):
            s = C.signature(form)
            why += f" [v2 {'|'.join(s)} -> {C.signature_split(s)}]"
        res = check(text, task)
        ok = res["first_failed"] is None and res["highest_passed"] == "V5"
        v6 = res["rungs"].get("V5") or {}
        failed = [f"{c['feature']}" for c in v6.get("checks", []) if not c["passed"]] + \
                 [f"graph:{g_['kind']}" for g_ in v6.get("graph_checks", []) if not g_["passed"]]
        verdict = "as expected" if ok == expect else "UNEXPECTED"
        bad += ok != expect
        print(f"{name:26s} passed={ok!s:5s} first_failed={res['first_failed']} {failed} -> {verdict}\n    {why}")
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
