"""Authoring script for GRC-Bench Tier-F item gbf-dev-06 (build, L2, composed graph). Deterministic.

    cd grc-bench && PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 ~/grc-agent/.venv/bin/python \
        items/tier_f/dev/gbf-dev-06/build.py
    ./gb tier-f gates items/tier_f/dev/gbf-dev-06

Version 3 (22 Oct 2026): the 'made from the 8 kHz audio' path check ends at the carrier-shift stage.
Version 2 (18 Oct 2026, round-1 fix stage): re-themed. Version 1 (a two-channel digital down-converter)
shared the signature-v2 cell combine|filter|mix|resample|tone with gbf-test-18 (a cross-split twin);
gbf-test-18 keeps that cell (reports/tier_f_reviews/fix_plan.json).

The task: an upper-sideband (USB) SSB transmitter test bench built by the PHASING method.
  * audio: the sum of a 700 Hz sinusoid (amplitude 0.4) and a 2.4 kHz sinusoid (amplitude 0.25), float,
    at an 8 kHz audio rate, written to audio.dat (float, 8 kHz);
  * SSB: the audio is brought to 48 kHz (samp_rate), turned into its analytic signal with GNU Radio's
    Hilbert block (hilbert_fc: real part = the audio delayed by (N-1)/2, imaginary part = its Hilbert
    transform), multiplied by a +10 kHz complex carrier, and reduced to its real part:
        Re{(x + jH{x}) e^{+j 2 pi 10k t}} = 0.4 cos(2 pi 10.7k t) + 0.25 cos(2 pi 12.4k t)
    written to ssb.dat (float, 48 kHz). Lower sideband (9.3 kHz, 7.6 kHz) and carrier (10 kHz) must
    each be >= 30 dB below the weaker wanted tone.
Reference: 2 tones, Add, File Sink (audio), Rational Resampler x6, Hilbert (257 taps), carrier Signal
Source, Multiply, Complex To Real, File Sink (ssb) = 10 functional blocks; level L2 (6-15 blocks, a
rate change, modulation, filtering; proposal 5.2).

Why 2.4 kHz and not the classic 1.9 kHz of the two-tone test (fix stage, calibration)
  The oracle bounds each tone's share of the power plus the total rms (there is no per-band absolute
  power feature), so it can only see an amplitude RATIO error beyond about +-10 %. A Repeat x6 (zero-
  order-hold) interpolator droops by sinc(f/8 kHz): -1.25 % at 700 Hz and -9.0 % at 1.9 kHz, which
  violates "amplitudes within 5 %" but passed V5 (authoring probe). At 2.4 kHz the droop is -14.2 %:
  share 0.228 against the minimum 0.242, so the zero-order hold (with or without a low-pass after it) is
  rejected, while every answer with both amplitudes within 5 % still passes.

Graph checks (structural requirements stated in the instruction)
  * "use GNU Radio's Hilbert block to form the analytic signal": a Hilbert block is present and lies on
    a path from a source to a File Sink. This also keeps the item out of gbf-test-18's test cell: the
    filter method, a complex band-pass "analytic" filter or a Weaver modulator give
    combine|filter|mix|resample|tone (test).
  * "then turn this audio into ... at 48 kHz": the SSB is made from the 8 kHz audio, so a
    rate-increasing block lies on a path from a source to the block that shifts the signal to the
    carrier (Multiply, Multiply Conjugate, Rotator, Frequency Xlating filter, Embedded Python). Every
    correct design raises the rate before that stage, because a 10 kHz carrier does not exist at 8 kHz.
    Without this check a graph that writes the 8 kHz audio to audio.dat but builds the SSB from a second
    copy of the tones generated at 48 kHz passed V5 (authoring probe; its cell combine|hilbert|mix|tone is
    a test cell). Ending the path at the shift stage, not at any File Sink, also rejects the same graph
    when a decimator on the audio.dat branch is the only rate-changing block. The id list holds every
    GNU Radio 3.10 block that can raise the rate (resamplers, interpolating FIR / Low Pass / Band Pass /
    RRC filters, PFB interpolator / synthesizer, Repeat, stream interleavers).

Oracle derivation (every number is stated in the instruction)
  * Real signals, band_fraction after the 18 Oct Parseval fix: an on-bin cosine of amplitude A has
    band power A^2/2 of the total. Two tones: total 0.4^2/2 + 0.25^2/2 = 0.11125 (rms 0.33354).
    700 Hz / 10.7 kHz share 0.71910, 2.4 kHz / 12.4 kHz share 0.28090. With each amplitude allowed
    +-5 %, the shares range over [0.6769, 0.7578] and [0.2422, 0.3231] (computed below); together
    with rms 0.33354 +-5 % this never rejects an answer whose two amplitudes are both within 5 %.
  * "at least 30 dB below the 12.4 kHz tone": the share of a +-50 Hz band around 9.3 kHz, 7.6 kHz and
    10 kHz must be <= 1e-3 x (largest allowed 12.4 kHz share) = 3.23e-4.
  * "the two tones carry at least 99 % of the power": band 10.65-12.45 kHz >= 0.99 (images of the
    interpolation, sidebands and the carrier together <= 1 %).
  * frequencies: peak_freq within 1 % (10.7 kHz in ssb.dat at 48 kHz, 700 Hz in audio.dat at 8 kHz;
    the 8 kHz audio rate is checked here: the same samples read at another rate move the peak).
  * Window: skip 0.1 s (4800 / 800 items), analyse 0.2 s (9600 / 1600 items): 5 Hz bins, every tone,
    sideband and carrier frequency on an exact bin, no window (no leakage).
Hilbert length (calibration, measured with the ladder in authoring): the Hamming Hilbert's gain g at
  700 Hz sets the opposite-sideband level |1-g|/(1+g). At 48 kHz: 65 taps (the block default) leaves
  the 9.3 kHz LSB tone ~16 dB below the 12.4 kHz tone (fails 30 dB); 93 taps fail narrowly (share
  3.8e-4 > 3.23e-4); 97, 129 and 257 taps pass. At 8 kHz the default 65 taps pass. So a correct answer
  either places the Hilbert block at the 8 kHz audio rate or gives it >= ~97 taps (Hamming) at 48 kHz.
  The near-miss mutant "num_taps 65 at 48 kHz" pins this.

Deterministic; writes task.json, reference.grc and variants/*.grc through taskio.save.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from grcbench.common.canary import CANARY  # noqa: E402
from grcbench.tier_f import compose as C  # noqa: E402
from grcbench.tier_f import grc, grcfile, taskio  # noqa: E402
from grcbench.tier_f.compose import block, file_sink, sig_source  # noqa: E402

TASK_ID = "gbf-dev-06"
SPLIT = "dev"
VERSION = 3

# ---- the specification ---------------------------------------------------------------------------
FS = 48000                 # samp_rate: the SSB output rate
AUDIO_RATE = 8000
TONES = [(700, 0.4), (2400, 0.25)]
CARRIER = 10000
USB = [(CARRIER + f, a) for f, a in TONES]          # 10.7 kHz / 12.4 kHz
LSB = [CARRIER - f for f, _ in TONES]               # 9.3 kHz / 7.6 kHz
SUPPRESS_DB = 30
AMP_REL = 0.05
FREQ_REL = 0.01
HALF = 50                  # +-50 Hz bands
PURITY = 0.99
AUDIO_FILE, SSB_FILE = "audio.dat", "ssb.dat"
SKIP_S, WIN_S = 0.1, 0.2
SKIP_SSB, N_SSB = int(SKIP_S * FS), int(WIN_S * FS)                    # 4800, 9600
SKIP_AU, N_AU = int(SKIP_S * AUDIO_RATE), int(WIN_S * AUDIO_RATE)      # 800, 1600
RUN_ITEMS = SKIP_SSB + N_SSB                                            # 14400 at every point

P_TOT = sum(a * a / 2 for _, a in TONES)
RMS = math.sqrt(P_TOT)

# sources that can make the audio, and every GNU Radio 3.10 block that can raise the sample rate
SOURCE_IDS = ["analog_sig_source_x", "blocks_vector_source_x"]
RATE_UP_IDS = ["rational_resampler_xxx", "interp_fir_filter_xxx", "pfb_arb_resampler_xxx", "pfb_interpolator_ccf",
               "pfb_synthesizer_ccf", "mmse_resampler_xx", "blocks_repeat", "low_pass_filter", "band_pass_filter",
               "root_raised_cosine_filter", "blocks_interleave", "blocks_streams_to_stream", "blocks_stream_mux",
               "blocks_patterned_interleaver", "blocks_vector_to_stream"]
# blocks that can shift the analytic signal up to the carrier (every correct design raises the rate before
# this stage: a 10 kHz carrier does not exist at the 8 kHz audio rate)
SHIFT_IDS = ["blocks_multiply_xx", "blocks_multiply_conjugate_cc", "blocks_rotator_cc", "freq_xlating_fir_filter_xxx",
             "freq_xlating_fft_filter_ccc", "epy_block"]


def share_bounds(k: int) -> tuple[float, float]:
    """Range of tone k's power share when every tone amplitude is within +-AMP_REL."""
    lo_hi = []
    for own, other in ((1 - AMP_REL, 1 + AMP_REL), (1 + AMP_REL, 1 - AMP_REL)):
        p = [(a * (own if i == k else other)) ** 2 / 2 for i, (_, a) in enumerate(TONES)]
        lo_hi.append(p[k] / sum(p))
    return lo_hi[0], lo_hi[1]


SHARE = [share_bounds(0), share_bounds(1)]
SIDE_MAX = 10 ** (-SUPPRESS_DB / 10) * SHARE[1][1]   # 30 dB below the weaker tone, at its largest share


def _floor(x: float, d: int = 4) -> float:
    return math.floor(x * 10 ** d) / 10 ** d


def _ceil(x: float, d: int = 4) -> float:
    return math.ceil(x * 10 ** d) / 10 ** d


INSTRUCTION = (
    "In this empty flowgraph, build a test bench for a single-sideband (SSB) transmitter that uses the "
    "phasing method. Set samp_rate to 48 kHz; this is the sample rate of the SSB output. First make a "
    "two-tone audio test signal at an audio sample rate of 8 kHz: the sum of a 700 Hz sinusoid with "
    "amplitude 0.4 and a 2.4 kHz sinusoid with amplitude 0.25 (float samples). Write this audio to a File "
    "Sink with the file name audio.dat (float items, 8 kHz). Then turn this audio into an upper-sideband "
    "(USB) signal on a 10 kHz carrier at 48 kHz by the phasing method: use GNU Radio's Hilbert block to "
    "form the analytic (complex) signal of the audio, and shift that up to the carrier. Write the result "
    "as a real signal to a File Sink with the file name ssb.dat (float items, 48 kHz). In ssb.dat the "
    "700 Hz tone must appear at 10.7 kHz with amplitude 0.4 and the 2.4 kHz tone at 12.4 kHz with "
    "amplitude 0.25, and these two tones must carry at least 99 % of the power; the lower sideband (the "
    "tones at 9.3 kHz and 7.6 kHz) and the carrier (10 kHz) must each be at least 30 dB weaker than the "
    "12.4 kHz tone. In audio.dat the 700 Hz and 2.4 kHz tones must have the amplitudes given above. "
    "Tolerances: frequencies within 1 %, amplitudes within 5 %. Everything is measured from 0.1 s after "
    "the start, so filter start-up does not count. The flowgraph should keep producing samples until it "
    "is stopped; do not limit the number of samples."
)


# ---- helpers -------------------------------------------------------------------------------------
def compact(data: dict) -> dict:
    """Drop catalog-default parameters; assert GRC resolves the compact form to the same values."""
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in data["blocks"]}
    out = grcfile.compact(data, defaults)
    out["options"] = data["options"]
    diffs = grc.compact_equivalent(data, out)
    assert not diffs, f"compact form differs from the full form: {diffs}"
    return out


def tone(name: str, f, a, rate: str = "audio_rate", **kw) -> dict:
    return sig_source(name, "float", f, a, samp_rate=rate, **kw)


# ---- reference -----------------------------------------------------------------------------------
def reference() -> dict:
    blocks = [
        tone("tone_700", TONES[0][0], TONES[0][1]),
        tone("tone_2400", TONES[1][0], TONES[1][1]),
        block("audio_sum", "blocks_add_xx", type="float", num_inputs="2"),
        file_sink("audio_out", "float", AUDIO_FILE),
        block("upsample", "rational_resampler_xxx", type="fff", interp="int(samp_rate/audio_rate)", decim="1"),
        block("hilbert", "hilbert_fc", num_taps="257"),
        sig_source("carrier", "complex", "carrier_freq", "1"),
        block("mixer", "blocks_multiply_xx", type="complex"),
        block("to_real", "blocks_complex_to_real"),
        file_sink("ssb_out", "float", SSB_FILE),
    ]
    conns = [["tone_700", "0", "audio_sum", "0"], ["tone_2400", "0", "audio_sum", "1"],
             ["audio_sum", "0", "audio_out", "0"], ["audio_sum", "0", "upsample", "0"],
             ["upsample", "0", "hilbert", "0"], ["hilbert", "0", "mixer", "0"], ["carrier", "0", "mixer", "1"],
             ["mixer", "0", "to_real", "0"], ["to_real", "0", "ssb_out", "0"]]
    return C.graph("ssb_phasing_tx", blocks, conns,
                   {"samp_rate": str(FS), "audio_rate": str(AUDIO_RATE), "carrier_freq": str(CARRIER)})


# ---- variants: genuinely different correct solutions ------------------------------------------------
def variant_hilbert_at_audio_rate() -> dict:
    """Hilbert block at the 8 kHz audio rate (its default 65 taps suffice there), the complex analytic
    signal interpolated by a complex Rational Resampler, carrier applied with a Rotator."""
    blocks = [
        tone("a1", "700", "0.4", rate="8000"),
        tone("a2", "2.4e3", "0.25", rate="8000"),
        block("audio_add", "blocks_add_xx", type="float", num_inputs="2"),
        file_sink("fs_audio", "float", AUDIO_FILE),
        block("hb", "hilbert_fc"),
        block("interp", "rational_resampler_xxx", type="ccc", interp="6", decim="1"),
        block("shift", "blocks_rotator_cc", phase_inc="2*math.pi*10e3/samp_rate"),
        block("re", "blocks_complex_to_real"),
        file_sink("fs_ssb", "float", SSB_FILE),
    ]
    conns = [["a1", "0", "audio_add", "0"], ["a2", "0", "audio_add", "1"], ["audio_add", "0", "fs_audio", "0"],
             ["audio_add", "0", "hb", "0"], ["hb", "0", "interp", "0"], ["interp", "0", "shift", "0"],
             ["shift", "0", "re", "0"], ["re", "0", "fs_ssb", "0"]]
    g = C.graph("ssb_hilbert_8k", [block("imp", "import", imports="import math")] + blocks, conns,
                {"samp_rate": "48000"})
    return g


def variant_zero_stuff_lpf_classic_phasing() -> dict:
    """Zero-stuffing interpolation (Interpolating FIR Filter, taps [1]) plus a Low Pass Filter with
    gain 6; the classic phasing modulator in real arithmetic: I*cos(wc t) - Q*sin(wc t), with the
    Hilbert block's I and Q split by Complex To Float."""
    blocks = [
        tone("lo_tone", "700", "0.4"),
        tone("hi_tone", "2400", "0.25"),
        block("mix_audio", "blocks_add_xx", type="float", num_inputs="2"),
        file_sink("audio_file", "float", AUDIO_FILE),
        block("zero_stuff", "interp_fir_filter_xxx", type="fff", interp="6", taps="[1.0]"),
        block("anti_image", "low_pass_filter", type="fir_filter_fff", decim="1", gain="6", samp_rate="samp_rate",
              cutoff_freq="3400", width="1200", win="window.WIN_BLACKMAN", beta="6.76"),
        block("hilb", "hilbert_fc", num_taps="301", win="window.WIN_BLACKMAN"),
        block("iq", "blocks_complex_to_float"),
        sig_source("lo_cos", "float", "fc", "1", waveform="analog.GR_COS_WAVE"),
        sig_source("lo_sin", "float", "fc", "1", waveform="analog.GR_SIN_WAVE"),
        block("i_mix", "blocks_multiply_xx", type="float"),
        block("q_mix", "blocks_multiply_xx", type="float"),
        block("usb", "blocks_sub_xx", type="float"),
        file_sink("ssb_file", "float", SSB_FILE),
    ]
    conns = [["lo_tone", "0", "mix_audio", "0"], ["hi_tone", "0", "mix_audio", "1"],
             ["mix_audio", "0", "audio_file", "0"], ["mix_audio", "0", "zero_stuff", "0"],
             ["zero_stuff", "0", "anti_image", "0"], ["anti_image", "0", "hilb", "0"], ["hilb", "0", "iq", "0"],
             ["iq", "0", "i_mix", "0"], ["lo_cos", "0", "i_mix", "1"], ["iq", "1", "q_mix", "0"],
             ["lo_sin", "0", "q_mix", "1"], ["i_mix", "0", "usb", "0"], ["q_mix", "0", "usb", "1"],
             ["usb", "0", "ssb_file", "0"]]
    return C.graph("ssb_classic_phasing", blocks, conns,
                   {"samp_rate": "48000", "audio_rate": "8000", "fc": "10000"})


def variant_interp_fir_rotator() -> dict:
    """Interpolating FIR Filter with designed low-pass taps (gain 6), a 129-tap Hilbert at 48 kHz and a
    Rotator; the audio gain split between the sources and a Multiply Const."""
    blocks = [
        tone("t1", "700", "0.8", rate="fs_audio"),
        tone("t2", "2400", "0.5", rate="fs_audio"),
        block("adder", "blocks_add_xx", type="float", num_inputs="2"),
        block("half", "blocks_multiply_const_vxx", type="float", const="0.5"),
        file_sink("snk_audio", "float", AUDIO_FILE),
        block("interp", "interp_fir_filter_xxx", type="fff", interp="6",
              taps="firdes.low_pass(6, samp_rate, 3500, 1000)"),
        block("hilbert", "hilbert_fc", num_taps="129"),
        block("rot", "blocks_rotator_cc", phase_inc="2*3.141592653589793*10000/samp_rate"),
        block("real", "blocks_complex_to_real"),
        file_sink("snk_ssb", "float", SSB_FILE),
    ]
    conns = [["t1", "0", "adder", "0"], ["t2", "0", "adder", "1"], ["adder", "0", "half", "0"],
             ["half", "0", "snk_audio", "0"], ["half", "0", "interp", "0"], ["interp", "0", "hilbert", "0"],
             ["hilbert", "0", "rot", "0"], ["rot", "0", "real", "0"], ["real", "0", "snk_ssb", "0"]]
    return C.graph("ssb_interp_fir", blocks, conns, {"samp_rate": "48000", "fs_audio": "8000"})


def variant_pfb_conjugate() -> dict:
    """Polyphase Arbitrary Resampler (rate 6), Hilbert at 48 kHz (Kaiser, 401 taps), the carrier applied
    as Multiply Conjugate with a -10 kHz oscillator, real part taken by Complex To Float (imaginary
    part parked on a Null Sink); a Throttle at the audio rate and absolute File Sink paths."""
    blocks = [
        tone("s700", "700", "0.4", rate="audio_rate", waveform="analog.GR_COS_WAVE"),
        tone("s2400", "2400", "0.25", rate="audio_rate"),
        block("sum2", "blocks_add_xx", type="float", num_inputs="2"),
        block("thr", "blocks_throttle2", type="float", samples_per_second="audio_rate"),
        file_sink("out_audio", "float", "/tmp/" + AUDIO_FILE),
        block("arb", "pfb_arb_resampler_xxx", type="fff", rrate="samp_rate/audio_rate"),
        block("ht", "hilbert_fc", num_taps="401", win="window.WIN_KAISER", beta="8"),
        sig_source("lo_neg", "complex", "-10000", "1"),
        block("conj_mix", "blocks_multiply_conjugate_cc"),
        block("c2f", "blocks_complex_to_float"),
        block("imag_park", "blocks_null_sink", type="float"),
        file_sink("out_ssb", "float", "/tmp/" + SSB_FILE),
    ]
    conns = [["s700", "0", "sum2", "0"], ["s2400", "0", "sum2", "1"], ["sum2", "0", "thr", "0"],
             ["thr", "0", "out_audio", "0"], ["thr", "0", "arb", "0"], ["arb", "0", "ht", "0"],
             ["ht", "0", "conj_mix", "0"], ["lo_neg", "0", "conj_mix", "1"], ["conj_mix", "0", "c2f", "0"],
             ["c2f", "0", "out_ssb", "0"], ["c2f", "1", "imag_park", "0"]]
    return C.graph("ssb_pfb_conj", blocks, conns, {"samp_rate": "48e3", "audio_rate": "8e3"})


def variant_interp_lpf_xlating() -> dict:
    """One interpolating Low Pass Filter block (type interp_fir_filter_fff, gain 6) as the interpolator,
    a Blackman Hilbert at 48 kHz, and the shift up to the carrier done by a Frequency Xlating FIR
    Filter (taps [1], centre -10 kHz, i.e. multiplication by e^{+j 2 pi 10k t})."""
    blocks = [
        tone("low", "700", "0.4"),
        tone("high", "2400", "0.25"),
        block("audio", "blocks_add_xx", type="float", num_inputs="2"),
        file_sink("audio_rec", "float", AUDIO_FILE),
        block("interp_lpf", "low_pass_filter", type="interp_fir_filter_fff", interp="6", gain="6",
              samp_rate="samp_rate", cutoff_freq="3400", width="1000", win="window.WIN_HAMMING", beta="6.76"),
        block("analytic", "hilbert_fc", num_taps="201", win="window.WIN_BLACKMAN"),
        block("up_shift", "freq_xlating_fir_filter_xxx", type="ccc", decim="1", taps="[1]",
              center_freq="-carrier", samp_rate="samp_rate"),
        block("real_out", "blocks_complex_to_real"),
        file_sink("ssb_rec", "float", SSB_FILE),
    ]
    conns = [["low", "0", "audio", "0"], ["high", "0", "audio", "1"], ["audio", "0", "audio_rec", "0"],
             ["audio", "0", "interp_lpf", "0"], ["interp_lpf", "0", "analytic", "0"],
             ["analytic", "0", "up_shift", "0"], ["up_shift", "0", "real_out", "0"], ["real_out", "0", "ssb_rec", "0"]]
    return C.graph("ssb_interp_lpf_xlating", blocks, conns,
                   {"samp_rate": "48000", "audio_rate": "8000", "carrier": "10e3"})


# ---- task record ---------------------------------------------------------------------------------
P_AU_RATE = "audio.dat: float audio at 8 kHz with the 700 Hz tone as the strongest component (within 1 %)"
P_AU_AMP = "audio.dat: the 700 Hz and 2.4 kHz tones with amplitudes 0.4 and 0.25 (each within 5 %)"
P_USB_FREQ = "ssb.dat: real (float) signal at 48 kHz, strongest component the 700 Hz tone at 10.7 kHz (within 1 %)"
P_USB_AMP = "ssb.dat: upper-sideband tones at 10.7 kHz (amplitude 0.4) and 12.4 kHz (amplitude 0.25), each within 5 %"
P_USB_PURE = "ssb.dat: the two upper-sideband tones carry at least 99 % of the power"
P_LSB = "ssb.dat: lower sideband (9.3 kHz, 7.6 kHz) each at least 30 dB below the 12.4 kHz tone"
P_CARRIER = "ssb.dat: carrier (10 kHz) at least 30 dB below the 12.4 kHz tone"
P_HILBERT = ("phasing method: GNU Radio's Hilbert block forms the analytic signal on the path from the audio "
             "sources to a File Sink")
P_FROM_AUDIO = ("the SSB signal is made from the 8 kHz audio: a rate-increasing block (8 kHz -> 48 kHz) lies on a "
                "path from the audio sources to the block that shifts the signal to the carrier (Multiply, Multiply "
                "Conjugate, Rotator, Frequency Xlating filter or Embedded Python)")
P_OBS = "outputs: float items to the File Sink audio.dat (8 kHz) and float items to the File Sink ssb.dat (48 kHz)"
P_RUN = ("run condition: the flowgraph keeps producing samples until stopped (V4 liveness: at least "
         f"{RUN_ITEMS} items at every observation point)")


def oracle() -> dict:
    au = {"point": "audio", "skip": SKIP_AU, "n": N_AU}
    ssb = {"point": "ssb", "skip": SKIP_SSB, "n": N_SSB}
    (f1, a1), (f2, a2) = TONES
    (u1, _), (u2, _) = USB
    checks = [
        {**au, "feature": "peak_freq", "fs": AUDIO_RATE, "expect": f1, "tol": FREQ_REL * f1, "property": P_AU_RATE},
        {**au, "feature": "rms", "expect": round(RMS, 5), "rel_tol": AMP_REL, "property": P_AU_AMP},
        {**au, "feature": "band_fraction", "fs": AUDIO_RATE, "lo": f1 - HALF, "hi": f1 + HALF,
         "min": _floor(SHARE[0][0]), "max": _ceil(SHARE[0][1]), "property": P_AU_AMP},
        {**au, "feature": "band_fraction", "fs": AUDIO_RATE, "lo": f2 - HALF, "hi": f2 + HALF,
         "min": _floor(SHARE[1][0]), "max": _ceil(SHARE[1][1]), "property": P_AU_AMP},
        {**ssb, "feature": "peak_freq", "fs": FS, "expect": u1, "tol": FREQ_REL * u1, "property": P_USB_FREQ},
        {**ssb, "feature": "rms", "expect": round(RMS, 5), "rel_tol": AMP_REL, "property": P_USB_AMP},
        {**ssb, "feature": "band_fraction", "fs": FS, "lo": u1 - HALF, "hi": u1 + HALF,
         "min": _floor(SHARE[0][0]), "max": _ceil(SHARE[0][1]), "property": P_USB_AMP},
        {**ssb, "feature": "band_fraction", "fs": FS, "lo": u2 - HALF, "hi": u2 + HALF,
         "min": _floor(SHARE[1][0]), "max": _ceil(SHARE[1][1]), "property": P_USB_AMP},
        {**ssb, "feature": "band_fraction", "fs": FS, "lo": u1 - HALF, "hi": u2 + HALF, "min": PURITY,
         "property": P_USB_PURE},
    ]
    for f in LSB:
        checks.append({**ssb, "feature": "band_fraction", "fs": FS, "lo": f - HALF, "hi": f + HALF,
                       "max": _ceil(SIDE_MAX, 6), "property": P_LSB})
    checks.append({**ssb, "feature": "band_fraction", "fs": FS, "lo": CARRIER - HALF, "hi": CARRIER + HALF,
                   "max": _ceil(SIDE_MAX, 6), "property": P_CARRIER})
    graph_checks = [
        {"kind": "block_present", "ids": ["hilbert_fc"], "min": 1, "group": "main", "property": P_HILBERT},
        {"kind": "path", "from_ids": SOURCE_IDS, "via_ids": ["hilbert_fc"],
         "to_ids": ["blocks_file_sink"], "group": "main", "property": P_HILBERT},
        {"kind": "path", "from_ids": SOURCE_IDS, "via_ids": RATE_UP_IDS,
         "to_ids": SHIFT_IDS, "group": "main", "property": P_FROM_AUDIO},
    ]
    return {"checks": checks, "graph_checks": graph_checks}


MUTATIONS = [
    {"op": "set_param", "block": "carrier_freq", "param": "value", "value": "-10000", "breaks": P_LSB,
     "note": "carrier at -10 kHz: the real part is the LOWER sideband (tones at 9.3 and 7.6 kHz)"},
    {"op": "set_id", "block": "hilbert", "new_id": "blocks_float_to_complex", "breaks": P_HILBERT,
     "note": "Hilbert block replaced by Float To Complex (imaginary part 0): double sideband, no phasing"},
    {"op": "set_param", "block": "carrier", "param": "amp", "value": "0.9", "breaks": P_USB_AMP,
     "note": "near miss: carrier amplitude 0.9, both SSB tones 10 % low (audio.dat unchanged)"},
    {"op": "set_param", "block": "hilbert", "param": "num_taps", "value": "65", "breaks": P_LSB,
     "note": "near miss: the default 65-tap Hilbert at 48 kHz leaves the 9.3 kHz LSB tone ~16 dB below the 12.4 kHz tone"},
    {"op": "set_param", "block": "carrier_freq", "param": "value", "value": "9500", "breaks": P_USB_FREQ,
     "note": "carrier 9.5 kHz: tones at 10.2 and 11.9 kHz"},
    {"op": "set_param", "block": "upsample", "param": "interp", "value": "4", "breaks": P_USB_FREQ,
     "note": "interpolation by 4: the 48 kHz stream holds audio at the wrong rate"},
    {"op": "set_param", "block": "tone_700", "param": "freq", "value": "710", "breaks": P_AU_RATE,
     "note": "near miss: audio tone 1.4 % high (700 Hz +-1 %)"},
    {"op": "set_param", "block": "tone_700", "param": "amp", "value": "0.25", "breaks": P_AU_AMP,
     "note": "700 Hz tone with the 2.4 kHz tone's amplitude"},
    {"op": "set_param", "block": "tone_2400", "param": "amp", "value": "0.28", "breaks": P_USB_AMP,
     "note": "near miss: 2.4 kHz tone amplitude 12 % high"},
    {"op": "set_param", "block": "tone_700", "param": "offset", "value": "0.05", "breaks": P_CARRIER,
     "note": "DC in the audio: carrier leaks through (the Hilbert transform of DC is 0, the real branch passes it)"},
    {"op": "set_param", "block": "tone_700", "param": "samp_rate", "value": "samp_rate", "breaks": P_AU_RATE,
     "note": "700 Hz source run at 48 kHz: audio.dat no longer holds 8 kHz audio"},
    {"op": "set_id", "block": "to_real", "new_id": "blocks_complex_to_mag", "breaks": P_USB_FREQ,
     "note": "magnitude instead of the real part: the envelope, no carrier"},
    {"op": "rewire_input", "dst": "ssb_out", "dst_port": "0", "new_src": "upsample", "breaks": P_USB_FREQ,
     "note": "wrong tap: the 48 kHz audio before the Hilbert and the carrier"},
]

SPEC_PROPERTIES = [P_AU_RATE, P_AU_AMP, P_USB_FREQ, P_USB_AMP, P_USB_PURE, P_LSB, P_CARRIER, P_HILBERT, P_FROM_AUDIO,
                   P_OBS, P_RUN]

REVIEW_NOTES = (
    "Version 2: re-themed in the round-1 fix stage (version 1, a two-channel DDC, was a cross-split twin of "
    "gbf-test-18; resolution in reports/tier_f_reviews/round1/gbf-dev-06.resolution.json). Composed L2 item: the "
    "approved plan says L2 tasks come from converted examples and eval plan 4.3 lists only hand-written L1 chains, so "
    "composed L2 items need the team's sign-off (fix_plan team_decisions; only L3 seeds remain unused in dev). The "
    "upper test tone is 2.4 kHz, not the 1.9 kHz of the classic two-tone test, so that a zero-order-hold (Repeat) "
    "interpolator, whose droop violates the 5 % amplitude tolerance, is rejected (at 1.9 kHz it passed). Known "
    "oracle limits: (1) the Hilbert requirement is checked structurally (a Hilbert block on a source -> File Sink "
    "path); a contrived graph that adds a Hilbert block on a side branch to another File Sink and makes ssb.dat some "
    "other way would pass. (2) Only GNU Radio's Hilbert block (hilbert_fc) counts, as the instruction names it. "
    "Two other Hilbert-transformer constructions fail the graph check: the Filter Delay block (filter_delay_fc) with "
    "firdes.hilbert taps, and an FIR filter with firdes.hilbert taps plus a matching Delay. Team decision: keep the "
    "block named, or drop the structural requirement. Dropping it would also admit the filter and Weaver methods, "
    "whose v2 cell is gbf-test-18's test cell. The FIR + Delay form hashes to train "
    "(combine|delay|filter|hilbert|mix|resample|tone). The Filter Delay form gets the unmapped class "
    "other:filter_delay_fc, which hashes to test. (3) The default 65-tap Hilbert placed at 48 kHz misses the stated "
    "30 dB (about 16 dB below the 12.4 kHz tone); placing it at 8 kHz or giving it >= ~97 taps (Hamming) passes. (4) "
    "'Made from the 8 kHz audio' is checked as a rate-increasing block on a path from a source to the block that "
    "shifts the signal to the carrier (Multiply, Multiply Conjugate, Rotator, Frequency Xlating filter, Embedded "
    "Python). A graph that generates the tones at 48 kHz, builds the SSB from them and decimates only the copy for "
    "audio.dat fails it, whichever decimator it uses (Rational Resampler, Low Pass Filter or Keep 1 in N). The check "
    "cannot read rate parameters, so a contrived graph that puts a 1:1 resampler or a non-interpolating Low Pass "
    "Filter between 48 kHz tones and the mixer would pass. (5) Amplitude tolerance: there is no per-band absolute power feature, "
    "so the oracle bounds each tone's share of the power and the total rms. This accepts every answer with both "
    "amplitudes within 5 %, but also some with one tone further off (up to about 8 % for the 700 Hz tone and 12 % for "
    "the 2.4 kHz tone)."
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
        "family": "build", "level": "L2",
        "instruction": INSTRUCTION,
        "start_grc": None, "reference_grc": "reference.grc",
        "observation": {
            "points": [{"name": "audio", "sink": {"block_id": "blocks_file_sink",
                                                  "params": {"file": AUDIO_FILE, "type": "float"}}},
                       {"name": "ssb", "sink": {"block_id": "blocks_file_sink",
                                                "params": {"file": SSB_FILE, "type": "float"}}}],
            "run": {"n_items": RUN_ITEMS, "min_items": RUN_ITEMS, "timeout_s": 10.0},
            "contract": ("the float samples written to the File Sink whose file name is audio.dat (the 8 kHz "
                         "two-tone audio) and the float samples written to the File Sink whose file name is "
                         "ssb.dat (the 48 kHz USB signal)")},
        "oracle": oracle(),
        "mutations": MUTATIONS,
        "spec_properties": SPEC_PROPERTIES,
        "perturbation": {"perturbed": False, "changes": []},
        "disjointness": (
            "No training-side spec builds an SSB transmitter or uses a Hilbert transformer "
            "(reports/reference/training_task_specs.json). Nearest: t01_dial_tone / t24 (a two-tone audio "
            "build: here the tones are 700 Hz and 2.4 kHz, not the 350/440 Hz dial tone, and they are only the "
            "test input), t16_am_demod_build (AM modulate and demodulate: double sideband with carrier, no "
            "Hilbert or sideband suppression), 21_type_conversion_and_conjugate (a GRC-Agent scenario that "
            "converts float to complex and conjugates, with no analytic signal or frequency shift) and "
            "t10_param_tuning (centre-frequency retune of a given graph). Signature cell "
            "combine|filter|hilbert|mix|resample|tone (dev) differs from gbf-test-18's combine|filter|mix|resample|tone."),
    }


def main() -> int:
    ref = compact(reference())
    sig = C.signature(ref)
    split = C.signature_split(sig)
    assert split == SPLIT, f"signature {sig} belongs to split {split}, not {SPLIT}"
    functional = [b for b in ref["blocks"] if b["id"] not in ("variable", "import")]
    assert len(functional) == 10, len(functional)
    variants = {
        "variants/v1_hilbert_at_8k_rotator.grc": variant_hilbert_at_audio_rate(),
        "variants/v2_zero_stuff_classic_phasing.grc": variant_zero_stuff_lpf_classic_phasing(),
        "variants/v3_interp_fir_rotator.grc": variant_interp_fir_rotator(),
        "variants/v4_pfb_multiply_conjugate.grc": variant_pfb_conjugate(),
        "variants/v5_interp_lpf_xlating_shift.grc": variant_interp_lpf_xlating(),
    }
    files = {"reference.grc": grcfile.dump(ref)}
    print(f"reference  v2 signature {'|'.join(sig)} -> {split}")
    for name, g in variants.items():
        g = compact(g)
        vs = C.signature(g)
        print(f"{name:45s} v2 signature {'|'.join(vs)} -> {C.signature_split(vs)}")
        files[name] = grcfile.dump(g)
    errors = taskio.save(HERE, record(sig), files)
    if errors:
        print("NOT WRITTEN:", *errors, sep="\n  ")
        return 1
    for stale in sorted((HERE / "variants").glob("*.grc")):   # keep the directory equal to `files`
        if f"variants/{stale.name}" not in files:
            stale.unlink()
    print(f"written {HERE.relative_to(REPO)}; bounds: shares {SHARE}, side max {SIDE_MAX:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
