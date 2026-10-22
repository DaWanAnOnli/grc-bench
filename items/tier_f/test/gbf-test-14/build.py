"""Authoring script for GRC-Bench Tier-F item gbf-test-14 (family build, level L2, composed graph).

    cd grc-bench && PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 ~/grc-agent/.venv/bin/python \
        items/tier_f/test/gbf-test-14/build.py
    ./gb tier-f gates items/tier_f/test/gbf-test-14

Narrowband FM loopback test: a 1.2 kHz test tone (amplitude 1.0) is frequency-modulated onto a 40 kHz
complex baseband signal with 2.5 kHz peak deviation (input 1.0 = 2.5 kHz), demodulated, and brought
back to an 8 kHz audio rate. Two observation points: the FM signal (fm_iq.dat, complex, 40 kHz) and the
recovered audio (recovered.dat, float, 8 kHz).

Version 2 (18 Oct 2026, round-1 fix stage):
  * F1 (both lenses): in version 1 the receiver scale equalled the transmitter scale, so the recovered
    audio was the source tone itself and a graph that wrote the tone straight to recovered.dat passed.
    The receiver now reads the instantaneous deviation in kHz (1 kHz of deviation = 1.0), so the
    recovered tone has amplitude 2.5 (rms 1.7678) while the source tone has amplitude 1.0: the natural
    wrong tap (recovered.dat fed from the tone) fails, and it is now a mutant. The instruction's
    "demodulate" is also checked structurally: a path from an FM modulator through an FM demodulator to a
    File Sink, accepting every GNU Radio 3.10 block that frequency-modulates (Frequency Mod, VCO
    (complex), Phase Mod, NBFM/WBFM Transmit) or demodulates (Quadrature Demod, FM Demod, NBFM/WBFM
    Receive, WBFM Receive PLL, PLL Freq Det, or a hand-built discriminator ending in Complex to Arg;
    variant v9 is such a discriminator and passes). No noise source was added (that would move the
    signature cell to train; fix_plan). Authoring probes: the tone copied to recovered.dat, a graph
    without a demodulator and a second 1.2 kHz source of amplitude 2.5 all fail V6; NBFM Receive with
    its 75 us de-emphasis fails the amplitude check (rms 1.54 against 1.77 +-5 %).
  * F2 (run length): the set run sentence is in the instruction; V5 liveness needs 9000 items at both
    points (with the non-blocking taps of tooling T7, n_items = min_items = 9000).
  * F3 (tolerances): every tolerance and the measurement window are stated in the instruction.
  * F4 (8 kHz tone stage unobservable): the instruction now allows either generating the tone at 8 kHz
    and up-sampling, or generating it directly at 40 kHz; spec_properties no longer claim the 8 kHz
    tone stage. Variant v4 generates it at 40 kHz.
  * compliance F2 (split depends on connection order): moot under signature v2 (a sorted set of
    functional classes, independent of connection order); every variant's v2 signature is printed.
  * compliance F4 (default parameter kept in the compact reference): not changed. The reference is
    now compacted against the catalog defaults (and checked with grc.compact_equivalent), but the File
    Sinks keep "type" on purpose: the item type of each observation File Sink is part of the stated
    observation contract (R9; the observation points match on it), the catalog declares no explicit
    default for blocks_file_sink.type (GRC falls back to the first option), and it costs ~3 tokens.
  * build-task format (16 Oct decision, R3): the open flowgraph is empty except for samp_rate, so the
    FM rate is samp_rate (40 kHz) and the instruction says so.

Oracle derivation (all tolerances are stated in the instruction):
  * recovered audio: 1.2 kHz within +-1 % (peak_freq), amplitude 2.5 within +-5 % (rms 1.7678), clean
    tone: >= 98 % of the power within 1150-1250 Hz (real signal; band_fraction after the 18 Oct Parseval
    fix; a pure on-bin tone gives 1.0);
  * FM signal: constant unit amplitude (rms 1.0 within +-5 %, peak magnitude <= 1.05). A sinusoidal FM
    signal has its power in lines at n * 1.2 kHz with power J_n(beta)^2, beta = 2500/1200. The carrier
    line (J_0^2) and the first upper sideband (J_1^2) are bounded by the values for a deviation of
    2.5 kHz +-5 %, so they check the deviation, the 40 kHz rate (line position) and the absence of
    pre-emphasis (75 us / 50 us pre-emphasis raises the deviation at 1.2 kHz by 15 % / 7 %); de-emphasis
    lowers the recovered amplitude by 12 % / 6 % at 1.2 kHz; the Carson band +-3.7 kHz holds >= 98 %.
    Complex signals: band_fraction uses the two-sided spectrum (unchanged by the 18 Oct fix).
  * windows: skip 1000 items, analyse 8000 (1 Hz bins for the audio, 5 Hz bins for the FM signal; every
    line on an exact bin).
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))

from scipy.special import jv  # noqa: E402

from grcbench.common.canary import CANARY  # noqa: E402
from grcbench.tier_f import compose as C  # noqa: E402
from grcbench.tier_f import edit, grc, grcfile, taskio  # noqa: E402
from grcbench.tier_f.compose import block  # noqa: E402

TASK_ID = "gbf-test-14"
SPLIT = "test"
VERSION = 2

AUDIO_RATE = 8000
IQ_RATE = 40000            # samp_rate
TONE_HZ = 1200
TONE_AMP = 1.0
DEV_HZ = 2500
RX_HZ_PER_UNIT = 1000      # receiver scale: 1 kHz of deviation reads 1.0
RECOVERED_AMP = TONE_AMP * DEV_HZ / RX_HZ_PER_UNIT    # 2.5
SKIP = 1000
N = 8000
RUN_ITEMS = SKIP + N
FREQ_REL, AMP_REL, DEV_REL = 0.01, 0.05, 0.05
IQ_FILE, AU_FILE = "fm_iq.dat", "recovered.dat"

MODULATORS = ["analog_frequency_modulator_fc", "blocks_vco_c", "analog_phase_modulator_fc", "analog_nbfm_tx",
              "analog_wfm_tx"]
DEMODULATORS = ["analog_quadrature_demod_cf", "analog_fm_demod_cf", "analog_nbfm_rx", "analog_wfm_rcv",
                "analog_wfm_rcv_pll", "analog_pll_freqdet_cf", "blocks_complex_to_arg"]

INSTRUCTION = (
    "In this empty flowgraph, build a loopback test for a narrowband FM voice link: frequency-modulate a "
    "test tone and then demodulate it again. Set samp_rate to 40 kHz; this is the sample rate of the FM "
    "signal. The test tone is a 1.2 kHz sinusoid with amplitude 1.0 (float samples); you may generate it at "
    "an 8 kHz audio rate and bring it up to 40 kHz, or generate it directly at 40 kHz. Frequency-modulate "
    "it into a complex baseband FM signal at 40 kHz (carrier at 0 Hz, constant amplitude 1.0) with a peak "
    "frequency deviation of 2.5 kHz (an input of 1.0 corresponds to 2.5 kHz of deviation), and write it to "
    "a File Sink with the file name fm_iq.dat (complex items). Then demodulate the FM signal and bring the "
    "result down to an 8 kHz audio rate, scaled so that the output reads the frequency deviation in kHz "
    "(1 kHz of deviation gives 1.0): the recovered audio should be a clean 1.2 kHz tone of amplitude 2.5. "
    "Write it to a File Sink with the file name recovered.dat (float items, 8 kHz). Do not use "
    "pre-emphasis or de-emphasis. Tolerances: tone frequency within 1 %, recovered amplitude within 5 %, "
    "peak deviation within 5 %; at least 98 % of the recovered audio power within 50 Hz of 1.2 kHz; at "
    "least 98 % of the FM signal power within 3.7 kHz of the carrier, and the FM signal's magnitude never "
    "above 1.05. The first 1000 samples written to each file are ignored (start-up). The flowgraph should "
    "keep producing samples until it is stopped; do not limit the number of samples."
)


# ---------------------------------------------------------------- helpers
def graph(gid: str, variables: dict[str, str], blocks: list[dict], connections: list[list]) -> dict:
    var_blocks = [{"name": k, "id": "variable", "parameters": {"value": str(v)}} for k, v in variables.items()]
    return {"options": {"parameters": {"id": gid, "generate_options": "no_gui"}},
            "blocks": var_blocks + blocks,
            "connections": [[a, str(ap), b, str(bp)] for a, ap, b, bp in connections],
            "metadata": {"file_format": 1}}


def file_sink(name: str, typ: str, path: str) -> dict:
    return block(name, "blocks_file_sink", type=typ, file=path)


def compact(data: dict) -> dict:
    """Drop catalog-default parameters; assert GRC resolves the compact form to the same values."""
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in data["blocks"]}
    out = grcfile.compact(data, defaults)
    out["options"] = data["options"]
    diffs = grc.compact_equivalent(data, out)
    assert not diffs, f"compact form differs from the full form: {diffs}"
    return out


# ---------------------------------------------------------------- reference
def reference() -> dict:
    blocks = [
        block("tone", "analog_sig_source_x", type="float", samp_rate="audio_rate", freq=TONE_HZ, amp="1"),
        block("upsample", "rational_resampler_xxx", type="fff", interp="int(samp_rate/audio_rate)", decim="1"),
        block("fm_mod", "analog_frequency_modulator_fc", sensitivity="2*math.pi*deviation/samp_rate"),
        file_sink("fm_iq", "complex", IQ_FILE),
        block("fm_demod", "analog_quadrature_demod_cf", gain="samp_rate/(2*math.pi*rx_scale)"),
        block("audio_lpf", "low_pass_filter", type="fir_filter_fff", decim="int(samp_rate/audio_rate)",
              samp_rate="samp_rate", cutoff_freq="3000", width="1000"),
        file_sink("recovered", "float", AU_FILE),
    ]
    conns = [["tone", 0, "upsample", 0], ["upsample", 0, "fm_mod", 0], ["fm_mod", 0, "fm_iq", 0],
             ["fm_mod", 0, "fm_demod", 0], ["fm_demod", 0, "audio_lpf", 0], ["audio_lpf", 0, "recovered", 0]]
    return graph("nbfm_loopback", {"samp_rate": IQ_RATE, "audio_rate": AUDIO_RATE, "deviation": DEV_HZ,
                                   "rx_scale": RX_HZ_PER_UNIT}, blocks, conns)


# ---------------------------------------------------------------- variants (other correct solutions)
def variant_pfb_up_rational_down() -> dict:
    """Sine tone, literal constants, Polyphase Arbitrary Resampler (rate 5) up, Rational Resampler
    (decimation 5) down instead of a decimating low-pass filter."""
    blocks = [
        block("src", "analog_sig_source_x", type="float", samp_rate="8000", waveform="analog.GR_SIN_WAVE",
              freq="1.2e3", amp="1"),
        block("up", "pfb_arb_resampler_xxx", type="fff", rrate="5"),
        block("modulator", "analog_frequency_modulator_fc", sensitivity="2*math.pi*2.5e3/40e3"),
        file_sink("iq_out", "complex", IQ_FILE),
        block("demod", "analog_quadrature_demod_cf", gain="40e3/(2*math.pi*1e3)"),
        block("down", "rational_resampler_xxx", type="fff", decim="5"),
        file_sink("audio_out", "float", AU_FILE),
    ]
    conns = [["src", 0, "up", 0], ["up", 0, "modulator", 0], ["modulator", 0, "iq_out", 0],
             ["modulator", 0, "demod", 0], ["demod", 0, "down", 0], ["down", 0, "audio_out", 0]]
    return graph("fm_loop_v1", {"samp_rate": "40000"}, blocks, conns)


def variant_fm_demod_hier() -> dict:
    """Interpolating FIR filter with explicit taps (gain 5 makes up for zero stuffing) and the
    FM Demod hierarchical block (deviation 1 kHz -> 1.0) with de-emphasis disabled (tau = 0)."""
    blocks = [
        block("imp", "import", imports="import math"),
        block("tone_src", "analog_sig_source_x", type="float", samp_rate="audio_fs", freq="tone_f", amp="1"),
        block("interp", "interp_fir_filter_xxx", type="fff", interp="5",
              taps="firdes.low_pass(5, samp_rate, 3000, 1000)"),
        block("modulator", "analog_frequency_modulator_fc", sensitivity="2*math.pi*max_dev/samp_rate"),
        file_sink("iq_file", "complex", IQ_FILE),
        block("demodulator", "analog_fm_demod_cf", chan_rate="samp_rate", audio_decim="5", deviation="1000",
              audio_pass="3000", audio_stop="3600", tau="0"),
        file_sink("audio_file", "float", AU_FILE),
    ]
    conns = [["tone_src", 0, "interp", 0], ["interp", 0, "modulator", 0], ["modulator", 0, "iq_file", 0],
             ["modulator", 0, "demodulator", 0], ["demodulator", 0, "audio_file", 0]]
    return graph("fm_loop_v2", {"samp_rate": "40e3", "audio_fs": "8e3", "max_dev": "2500", "tone_f": "1200"},
                 blocks, conns)


def variant_throttled_keep_one() -> dict:
    """Throttled source, Rational Resampler up, Blackman FIR low-pass at 40 kHz followed by Keep 1 in 5."""
    blocks = [
        block("osc", "analog_sig_source_x", type="float", samp_rate="fs_audio", freq="1200", amp="1"),
        block("throttle", "blocks_throttle2", type="float", samples_per_second="fs_audio"),
        block("up5", "rational_resampler_xxx", type="fff", interp="5", decim="1"),
        block("fmod", "analog_frequency_modulator_fc", sensitivity="k_mod"),
        file_sink("save_iq", "complex", IQ_FILE),
        block("qdemod", "analog_quadrature_demod_cf", gain="1.0/k_rx"),
        block("lpf", "low_pass_filter", type="fir_filter_fff", decim="1", samp_rate="samp_rate", cutoff_freq="2500",
              width="1500", win="window.WIN_BLACKMAN"),
        block("keep", "blocks_keep_one_in_n", type="float", n="5"),
        file_sink("save_audio", "float", AU_FILE),
    ]
    conns = [["osc", 0, "throttle", 0], ["throttle", 0, "up5", 0], ["up5", 0, "fmod", 0],
             ["fmod", 0, "save_iq", 0], ["fmod", 0, "qdemod", 0], ["qdemod", 0, "lpf", 0],
             ["lpf", 0, "keep", 0], ["keep", 0, "save_audio", 0]]
    return graph("fm_loop_v3", {"fs_audio": "8000", "samp_rate": "fs_audio*5",
                                "k_mod": "2*3.141592653589793*2500/samp_rate",
                                "k_rx": "2*3.141592653589793*1000/samp_rate"}, blocks, conns)


def variant_tone_at_40k() -> dict:
    """The tone generated directly at 40 kHz (no up-sampling stage), which the instruction allows."""
    blocks = [
        block("tone40", "analog_sig_source_x", type="float", samp_rate="samp_rate", freq="1200", amp="1.0"),
        block("fm", "analog_frequency_modulator_fc", sensitivity="2*math.pi*2500/samp_rate"),
        file_sink("iq_sink", "complex", IQ_FILE),
        block("disc", "analog_quadrature_demod_cf", gain="samp_rate/(2*math.pi*1000)"),
        block("dec5", "low_pass_filter", type="fir_filter_fff", decim="5", samp_rate="samp_rate", cutoff_freq="3400",
              width="600", win="window.WIN_HANN"),
        file_sink("audio_sink", "float", AU_FILE),
    ]
    conns = [["tone40", 0, "fm", 0], ["fm", 0, "iq_sink", 0], ["fm", 0, "disc", 0], ["disc", 0, "dec5", 0],
             ["dec5", 0, "audio_sink", 0]]
    return graph("fm_loop_v4", {"samp_rate": "40000"}, blocks, conns)


def variant_vco_repeat() -> dict:
    """VCO (complex) as the modulator, Repeat x5 (zero-order hold) as the up-sampler (round-1 reviewer
    alternatives); the sinc droop of the hold is -3.5 % at 1.2 kHz, inside the 5 % tolerances."""
    blocks = [
        block("tone", "analog_sig_source_x", type="float", samp_rate="audio_rate", freq="1200", amp="1"),
        block("hold", "blocks_repeat", type="float", interp="5"),
        block("vco", "blocks_vco_c", samp_rate="samp_rate", sensitivity="2*math.pi*2500", amplitude="1"),
        file_sink("iq", "complex", IQ_FILE),
        block("quad", "analog_quadrature_demod_cf", gain="samp_rate/(2*math.pi*1000)"),
        block("aa", "low_pass_filter", type="fir_filter_fff", decim="5", samp_rate="samp_rate", cutoff_freq="3000",
              width="1000"),
        file_sink("audio", "float", AU_FILE),
    ]
    conns = [["tone", 0, "hold", 0], ["hold", 0, "vco", 0], ["vco", 0, "iq", 0], ["vco", 0, "quad", 0],
             ["quad", 0, "aa", 0], ["aa", 0, "audio", 0]]
    return graph("fm_loop_v5", {"samp_rate": "40000", "audio_rate": "8000"}, blocks, conns)


def variant_decimate_before_demod() -> dict:
    """The complex FM signal is channel-filtered (sharp low-pass, 3.8 kHz) and decimated to 8 kHz
    before a Quadrature Demod at 8 kHz (round-1 reviewer alternative)."""
    blocks = [
        block("tone", "analog_sig_source_x", type="float", samp_rate="audio_rate", freq="1200", amp="1"),
        block("up", "rational_resampler_xxx", type="fff", interp="5", decim="1"),
        block("mod", "analog_frequency_modulator_fc", sensitivity="2*math.pi*2500/samp_rate"),
        file_sink("iq_rec", "complex", IQ_FILE),
        block("chan", "low_pass_filter", type="fir_filter_ccf", decim="5", samp_rate="samp_rate", cutoff_freq="3800",
              width="200"),
        block("demod8k", "analog_quadrature_demod_cf", gain="audio_rate/(2*math.pi*1000)"),
        file_sink("au_rec", "float", AU_FILE),
    ]
    conns = [["tone", 0, "up", 0], ["up", 0, "mod", 0], ["mod", 0, "iq_rec", 0], ["mod", 0, "chan", 0],
             ["chan", 0, "demod8k", 0], ["demod8k", 0, "au_rec", 0]]
    return graph("fm_loop_v6", {"samp_rate": "40000", "audio_rate": "8000"}, blocks, conns)


def variant_mirrored_dc_block_throttle(ref: dict) -> dict:
    """The reference with a mirrored spectrum (negative sensitivity and negative receiver gain), a DC
    Blocker before the audio sink and a Throttle at 8 kHz on the 40 kHz IQ stream (round-1 reviewer
    alternatives; the throttled run takes about 6 s for the 9000 audio items)."""
    g = edit.set_param(ref, "fm_mod", "sensitivity", "-2*math.pi*deviation/samp_rate")
    g = edit.set_param(g, "fm_demod", "gain", "-samp_rate/(2*math.pi*rx_scale)")
    g = copy.deepcopy(g)
    g["blocks"] += [block("dcb", "dc_blocker_xx", type="ff", length="32", long_form="True"),
                    block("slow", "blocks_throttle2", type="complex", samples_per_second="8000")]
    g["connections"] = [c for c in g["connections"] if c[0] != "fm_mod" and c[2] != "recovered"]
    g["connections"] += [["fm_mod", "0", "slow", "0"], ["slow", "0", "fm_iq", "0"], ["slow", "0", "fm_demod", "0"],
                         ["audio_lpf", "0", "dcb", "0"], ["dcb", "0", "recovered", "0"]]
    g["options"] = {"parameters": {"id": "fm_loop_v7", "generate_options": "no_gui"}}
    return g


def variant_keep_one_in_five(ref: dict) -> dict:
    """The reference with the decimating audio low-pass replaced by Keep 1 in 5 and no filter (round-1
    reviewer alternative: the demodulated clean tone needs no anti-alias filter)."""
    g = copy.deepcopy(ref)
    for b in g["blocks"]:
        if b["name"] == "audio_lpf":
            b["name"], b["id"], b["parameters"] = "decimate", "blocks_keep_one_in_n", {"type": "float", "n": "5"}
    g["connections"] = [[("decimate" if x == "audio_lpf" else x) for x in c] for c in g["connections"]]
    g["options"] = {"parameters": {"id": "fm_loop_v8", "generate_options": "no_gui"}}
    return g


def variant_hand_discriminator(ref: dict) -> dict:
    """The Quadrature Demod replaced by a hand-built discriminator: x[n] * conj(x[n-1]) (Delay + Multiply
    Conjugate), Complex to Arg, and a Multiply Const that turns radians per sample into kHz of deviation.
    Its v2 cell (delay|filter|fm_mod|multiply|phase|resample|tone) differs from the reference's: a genuine
    structural alternative, recorded in review.notes for the reserved list."""
    g = copy.deepcopy(ref)
    g["blocks"] = [b for b in g["blocks"] if b["name"] != "fm_demod"]
    g["blocks"] += [block("imp", "import", imports="import math"),
                    block("one_back", "blocks_delay", type="complex", delay="1"),
                    block("phase_step", "blocks_multiply_conjugate_cc"),
                    block("angle", "blocks_complex_to_arg"),
                    block("to_khz", "blocks_multiply_const_vxx", type="float", const="samp_rate/(2*math.pi*rx_scale)")]
    g["connections"] = [c for c in g["connections"] if "fm_demod" not in (c[0], c[2])]
    g["connections"] += [["fm_mod", "0", "phase_step", "0"], ["fm_mod", "0", "one_back", "0"],
                         ["one_back", "0", "phase_step", "1"], ["phase_step", "0", "angle", "0"],
                         ["angle", "0", "to_khz", "0"], ["to_khz", "0", "audio_lpf", "0"]]
    g["options"] = {"parameters": {"id": "fm_loop_v9", "generate_options": "no_gui"}}
    return g


# ---------------------------------------------------------------- oracle
def _floor(x: float, d: int = 3) -> float:
    return math.floor(x * 10**d) / 10**d


def _ceil(x: float, d: int = 3) -> float:
    return math.ceil(x * 10**d) / 10**d


P_TONE = "test tone: 1.2 kHz sinusoid, amplitude 1.0, float (generated at 8 kHz and brought up to 40 kHz, or at 40 kHz)"
P_IQ = "FM signal: complex baseband, constant amplitude 1.0 (rms within 5 %, magnitude <= 1.05)"
P_DEV = ("FM signal at 40 kHz with peak frequency deviation 2.5 kHz within 5 % (input 1.0 = 2.5 kHz): the carrier "
         "line J0^2 at 0 Hz and the first sideband J1^2 at +1.2 kHz (line positions read at 40 kHz)")
P_CARSON = "FM signal centred at 0 Hz: >= 98 % of its power within +-3.7 kHz (Carson bandwidth)"
P_NO_EMPH = ("no pre-emphasis or de-emphasis: checked through the deviation bounds of the J0^2/J1^2 checks (75 us / "
             "50 us pre-emphasis raises the deviation at 1.2 kHz by 15 % / 7 %) and the recovered-amplitude check "
             "(de-emphasis lowers it by 12 % / 6 %)")
P_AU_FREQ = "recovered audio: float at 8 kHz, a 1.2 kHz tone (within 1 %)"
P_AU_AMP = "recovered audio: amplitude 2.5 within 5 % (receiver scale: 1 kHz of deviation = 1.0)"
P_AU_CLEAN = "recovered audio: clean tone (>= 98 % of the power within 50 Hz of 1.2 kHz)"
P_DEMOD = ("recovered audio obtained by demodulating the FM signal: a path from an FM modulator through an FM "
           "demodulator to a File Sink")
P_OBS = "observation points: File Sink fm_iq.dat (complex, 40 kHz) and File Sink recovered.dat (float, 8 kHz)"
P_RUN = ("run condition: the flowgraph keeps producing samples until stopped (V5 liveness: at least "
         f"{RUN_ITEMS} items at both observation points)")


def oracle() -> dict:
    beta = DEV_HZ * TONE_AMP / TONE_HZ
    b_lo, b_hi = beta * (1 - DEV_REL), beta * (1 + DEV_REL)
    j0 = [jv(0, b) ** 2 for b in (b_lo, b_hi)]        # J0^2 falls monotonically on [b_lo, b_hi]
    j1 = [jv(1, b) ** 2 for b in (b_lo, b_hi)]        # J1^2 falls monotonically on [b_lo, b_hi]
    assert j0[0] > j0[1] and j1[0] > j1[1]
    carson = DEV_HZ + TONE_HZ
    iq = {"point": "iq", "skip": SKIP, "n": N}
    au = {"point": "audio", "skip": SKIP, "n": N}
    checks = [
        {**au, "feature": "peak_freq", "fs": AUDIO_RATE, "expect": TONE_HZ, "tol": FREQ_REL * TONE_HZ,
         "property": P_AU_FREQ},
        {**au, "feature": "rms", "expect": round(RECOVERED_AMP / math.sqrt(2), 4), "rel_tol": AMP_REL,
         "property": P_AU_AMP},
        {**au, "feature": "band_fraction", "fs": AUDIO_RATE, "lo": TONE_HZ - 50, "hi": TONE_HZ + 50,
         "min": 0.98, "property": P_AU_CLEAN},
        {**iq, "feature": "rms", "expect": 1.0, "rel_tol": AMP_REL, "property": P_IQ},
        {**iq, "feature": "max_abs", "max": 1.05, "property": P_IQ},
        {**iq, "feature": "band_fraction", "fs": IQ_RATE, "lo": -50, "hi": 50,
         "min": _floor(j0[1]), "max": _ceil(j0[0]), "property": P_DEV},
        {**iq, "feature": "band_fraction", "fs": IQ_RATE, "lo": TONE_HZ - 50, "hi": TONE_HZ + 50,
         "min": _floor(j1[1]), "max": _ceil(j1[0]), "property": P_DEV},
        {**iq, "feature": "band_fraction", "fs": IQ_RATE, "lo": -carson, "hi": carson, "min": 0.98,
         "property": P_CARSON},
    ]
    graph_checks = [
        {"kind": "path", "from_ids": MODULATORS, "via_ids": DEMODULATORS, "to_ids": ["blocks_file_sink"],
         "group": "main", "property": P_DEMOD},
    ]
    return {"checks": checks, "graph_checks": graph_checks}


# ---------------------------------------------------------------- mutations (each breaks the spec)
MUTATIONS = [
    {"op": "scale_param", "block": "tone", "param": "freq", "factor": 1.25, "breaks": P_AU_FREQ,
     "note": "test tone at 1.5 kHz"},
    {"op": "set_param", "block": "tone", "param": "samp_rate", "value": "samp_rate", "breaks": P_AU_FREQ,
     "note": "tone generated for 40 kHz but fed to the x5 up-sampler: 240 Hz"},
    {"op": "set_param", "block": "tone", "param": "amp", "value": "0.8", "breaks": P_DEV,
     "note": "tone amplitude 0.8: deviation 2.0 kHz, recovered amplitude 2.0"},
    {"op": "set_param", "block": "tone", "param": "waveform", "value": "analog.GR_SQR_WAVE", "breaks": P_AU_CLEAN,
     "note": "square-wave test tone"},
    {"op": "set_param", "block": "deviation", "param": "value", "value": "2650", "breaks": P_DEV,
     "note": "near miss: peak deviation 6 % too high"},
    {"op": "set_param", "block": "deviation", "param": "value", "value": "2350", "breaks": P_DEV,
     "note": "near miss: peak deviation 6 % too low"},
    {"op": "scale_param", "block": "fm_mod", "param": "sensitivity", "factor": 1.15, "breaks": P_DEV,
     "note": "deviation 15 % high, which is what 75 us pre-emphasis does at 1.2 kHz"},
    {"op": "set_param", "block": "rx_scale", "param": "value", "value": "2500", "breaks": P_AU_AMP,
     "note": "receiver scaled like the transmitter (2.5 kHz = 1.0): recovered amplitude 1.0"},
    {"op": "set_param", "block": "rx_scale", "param": "value", "value": "940", "breaks": P_AU_AMP,
     "note": "near miss: recovered amplitude 6 % high"},
    {"op": "set_param", "block": "samp_rate", "param": "value", "value": "48000", "breaks": P_DEV,
     "note": "FM signal at 48 kHz instead of 40 kHz: read at 40 kHz its sidebands sit at 1.0 kHz, not 1.2 kHz"},
    {"op": "bypass", "block": "audio_lpf", "breaks": P_AU_FREQ,
     "note": "no decimation back to the audio rate: recovered.dat holds 40 kHz samples"},
    {"op": "set_param", "block": "audio_lpf", "param": "cutoff_freq", "value": "1000", "breaks": P_AU_AMP,
     "note": "audio filter cutoff below the tone: recovered tone attenuated"},
    {"op": "rewire_input", "dst": "recovered", "dst_port": "0", "new_src": "tone", "breaks": P_DEMOD,
     "note": "wrong tap: recovered.dat fed from the 8 kHz source tone (no demodulation)"},
    {"op": "rewire_input", "dst": "recovered", "dst_port": "0", "new_src": "upsample", "breaks": P_DEMOD,
     "note": "wrong tap: the 40 kHz tone before modulation"},
]

SPEC_PROPERTIES = [P_TONE, P_IQ, P_DEV, P_CARSON, P_NO_EMPH, P_AU_FREQ, P_AU_AMP, P_AU_CLEAN, P_DEMOD, P_OBS, P_RUN]

REVIEW_NOTES = (
    "Version 2: round-1 fixes (resolution in reports/tier_f_reviews/round1/gbf-test-14.resolution.json). Composed "
    "L2 item: the approved plan says L2 tasks come from converted examples and eval plan 4.3 lists only hand-written "
    "L1 chains, so composed L2 items need the team's sign-off (fix_plan team_decisions). Known oracle limits: (1) the "
    "test tone's own rate (8 kHz then up-sampled, or 40 kHz) is not observable, so the instruction allows both. (2) An "
    "FM loopback reproduces its input, so recovered.dat is a scaled copy of the tone; the different receiver scale "
    "(amplitude 2.5, tone 1.0) and the modulator -> demodulator -> File Sink path check make a shortcut fail, but a "
    "contrived graph that scales the tone by 2.5 into recovered.dat and also wires a demodulator to another File "
    "Sink would pass. (3) The demodulator list of the path check covers GNU Radio 3.10's FM demodulators and a "
    "hand-built discriminator ending in Complex to Arg (variant v9); any other demodulator design would fail it. "
    "(4) With a single test tone, phase modulation with index 2.5/1.2 = 2.08 rad has the same line spectrum and "
    "the same demodulated tone as the stated FM, so a Phase Mod block without an integrator passes (authoring "
    "probe); the Phase Mod stays on the modulator list because Phase Mod after an integrator is a valid FM "
    "modulator. (5) Variant v9 (hand-built discriminator) has the v2 cell "
    "delay|filter|fm_mod|multiply|phase|resample|tone (dev), a genuine structural alternative to the reference's "
    "filter|fm_demod|fm_mod|resample|tone (test); it is listed here so the reserved list can include it."
)


def main() -> int:
    ref = compact(reference())
    sig = C.signature(ref)
    split = C.signature_split(sig)
    if split != SPLIT:
        raise SystemExit(f"signature {sig} belongs to split {split}, not {SPLIT}")
    variants = {
        "variants/v1_pfb_up_rational_down.grc": variant_pfb_up_rational_down(),
        "variants/v2_interp_fir_fm_demod_block.grc": variant_fm_demod_hier(),
        "variants/v3_throttle_lpf_keep_one_in_n.grc": variant_throttled_keep_one(),
        "variants/v4_tone_at_40k.grc": variant_tone_at_40k(),
        "variants/v5_vco_repeat_hold.grc": variant_vco_repeat(),
        "variants/v6_decimate_before_demod.grc": variant_decimate_before_demod(),
        "variants/v7_mirrored_dc_blocker_throttle.grc": variant_mirrored_dc_block_throttle(ref),
        "variants/v8_keep_one_in_five_no_filter.grc": variant_keep_one_in_five(ref),
        "variants/v9_hand_built_discriminator.grc": variant_hand_discriminator(ref),
    }
    files = {"reference.grc": grcfile.dump(ref)}
    print(f"reference  v2 signature {'|'.join(sig)} -> {split}")
    for name, g in variants.items():
        g = compact(g)
        vs = C.signature(g)
        print(f"{name:48s} v2 signature {'|'.join(vs)} -> {C.signature_split(vs)}")
        files[name] = grcfile.dump(g)
    record = {
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
            "points": [{"name": "iq", "sink": {"block_id": "blocks_file_sink",
                                               "params": {"file": IQ_FILE, "type": "complex"}}},
                       {"name": "audio", "sink": {"block_id": "blocks_file_sink",
                                                  "params": {"file": AU_FILE, "type": "float"}}}],
            "run": {"n_items": RUN_ITEMS, "min_items": RUN_ITEMS, "timeout_s": 10.0},
            "contract": ("the complex samples written to the File Sink whose file name is fm_iq.dat (the 40 kHz "
                         "FM signal) and the float samples written to the File Sink whose file name is "
                         "recovered.dat (the 8 kHz recovered audio)")},
        "oracle": oracle(),
        "mutations": MUTATIONS,
        "spec_properties": SPEC_PROPERTIES,
        "perturbation": {"perturbed": False, "changes": []},
        "disjointness": ("Not a training-side spec (reports/reference/training_task_specs.json): the training FM "
                         "tasks repair broadcast/wideband FM receivers (t03_fm_receiver_repair, t17_wbfm_tune_repair) "
                         "and the modulate-demodulate builds are AM radio (t16_am_demod_build), GMSK "
                         "(t21_gmsk_pipe_build) and QPSK BER (t02_qpsk_ber_chain); the GRC-Agent scenarios "
                         "21_type_conversion_and_conjugate (removes an FM modulator from a chain) and "
                         "22_fm_rx_filter_squelch (adds a filter and squelch to an FM receiver) are scripted edits "
                         "without a modulate-demodulate loop. This is a narrowband-FM tone "
                         "loopback with a stated deviation, a kHz-calibrated receiver and an 8 kHz / 40 kHz rate "
                         "plan, checked on the FM spectrum and the recovered tone."),
    }
    errors = taskio.save(HERE, record, files)
    if errors:
        print("\n".join(errors))
        return 1
    for stale in sorted((HERE / "variants").glob("*.grc")):   # keep the directory equal to `files`
        if f"variants/{stale.name}" not in files:
            stale.unlink()
    print(f"wrote {HERE.relative_to(ROOT)}")
    for c in record["oracle"]["checks"]:
        print(" ", c["point"], c["feature"], {k: c[k] for k in ("lo", "hi", "min", "max", "expect") if k in c})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
