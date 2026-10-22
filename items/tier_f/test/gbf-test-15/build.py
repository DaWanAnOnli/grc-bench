"""Authoring script for GRC-Bench Tier-F item gbf-test-15 (family modify, level L2, split test).

Seed: GNU Radio 3.10.12 example channels/demo_spec_an.grc (a test-split seed in
splits/flowgraph_seed_split.json). A complex tone and complex Gaussian noise are added, throttled
and shown on a spectrum display; in parallel, two RMS blocks measure the clean tone and the noise
separately, and Divide + Log10 turn the two levels into an SNR reading in dB. Three number
displays show the tone level, the noise level and the SNR.

Task: the user wants the spectrum display zoomed in on the band around the tone: a low-pass
filter (6 kHz cutoff, 1.5 kHz transition centred on the cutoff as GNU Radio's filter design defines
it, stated as band edges: unity gain up to 5.25 kHz, noise above 6.75 kHz at least 30 dB down) that
also decimates by 4, inserted only in the spectrum path, while the three measurement outputs keep
measuring the unfiltered tone and noise. Any design that meets the band edges is accepted, windowed
or equiripple; the half-amplitude point may lie within about 5 % of 6 kHz.

Oracle (revised 18 Oct after review round 1). Expected values come from the stated numbers and from
the taps of the stated design as GNU Radio's Low Pass Filter block makes it (firdes.low_pass(1, 64e3,
6e3, 1.5e3), default Hamming window), including the aliasing of the decimation:
  f2p  tone at +3.1 kHz on the 16 kHz output (rate, no frequency shift); output RMS (unity gain);
       the noise share of the last 1.5 kHz of the passband (-5.25..-3.75 kHz: unity gain up to
       5.25 kHz); the noise share of the transition band on each side (5.25-6.75 kHz: -45 % to +58 %
       of the stated design's share, i.e. the half-amplitude point within about 5 % of 6 kHz, wide
       enough for every equiripple design with the stated edges); the noise share of the stopband on
       each side (6.75-8 kHz of the output: at most what a flat -30 dB stopband would pass).
  p2p  the three meters (tone 0.5, noise 0.794, SNR -4.02 dB) and the tone meter's settling over
       its first 4000 samples, which pins the RMS averaging constant 0.0005 (the perturbed alpha).
Complex signals: band_fraction uses the two-sided spectrum, unaffected by the 18 Oct correction.

Start graph = the seed exported by `./gb tier-f seed` with its numbers perturbed (see PERTURBATION).
Observation points (File Sinks created by the seed tool from the Qt sinks):
  spec  qtgui_freq_sink_x_0_0.dat                 complex   (former frequency display)
  sig   qtgui_number_sink_signal_amplitude_0.dat  float     (former "Signal Amplitude" number display)
  noise qtgui_number_sink_0_0.dat                 float     (former "Noise Amplitude" number display)
  snr   qtgui_number_sink_snr_0.dat               float     (former "SNR" number display)

Regenerate (WSL, from the grc-bench folder; deterministic):
    PYTHONPATH=src ~/grc-agent/.venv/bin/python items/tier_f/test/gbf-test-15/build.py
then run the gates:
    ./gb tier-f gates items/tier_f/test/gbf-test-15
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from gnuradio.filter import firdes

from grcbench.common.canary import CANARY
from grcbench.common.io import read_json, sha256_file
from grcbench.common.paths import GNURADIO_EXAMPLES, SPLITS
from grcbench.tier_f import grc, grcfile, taskio
from grcbench.tier_f.seedtools import headless_seed

TASK_ID = "gbf-test-15"
SPLIT = "test"
HERE = Path(__file__).resolve().parent
SEED = "channels/demo_spec_an.grc"

# ----------------------------------------------------------------------------- numbers
FS = 64000                     # sample rate (perturbed from 100000)
FFT_SIZE = 640                 # (perturbed from 1000); tone = FS / FFT_SIZE * TONE_BIN
TONE_BIN = 31                  # (perturbed from 26)
F_TONE = FS / FFT_SIZE * TONE_BIN          # 3100 Hz
A_TONE = 0.5                   # tone amplitude (perturbed from 1.0)
NOISE_DB = -2                  # "snr" variable = noise level in dB (perturbed from 0)
SIGMA = 10 ** (NOISE_DB / 20)  # noise RMS amplitude, 0.7943
SNR_DB = 20 * math.log10(A_TONE / SIGMA)   # -4.02 dB
ALPHA = "0.0005"               # RMS averaging constant (perturbed from .0001)

DECIM = 4
FS_OUT = FS // DECIM           # 16000
CUTOFF = 6000.0
TWIDTH = 1500.0
PASS_EDGE = CUTOFF - TWIDTH / 2    # 5250: GNU Radio's convention, transition band centred on the cutoff
STOP_EDGE = CUTOFF + TWIDTH / 2    # 6750
STOP_DB = 30                       # stated minimum noise suppression above the stopband edge

# ----------------------------------------------------------------------------- names
SINK_SPEC = "qtgui_freq_sink_x_0_0"
SINK_SIG = "qtgui_number_sink_signal_amplitude_0"
SINK_NOISE = "qtgui_number_sink_0_0"
SINK_SNR = "qtgui_number_sink_snr_0"
FILES = {k: f"{v}.dat" for k, v in
         {"spec": SINK_SPEC, "sig": SINK_SIG, "noise": SINK_NOISE, "snr": SINK_SNR}.items()}
SRC, NOISE, ADD, THR = ("analog_sig_source_x_0", "analog_noise_source_x_0", "blocks_add_xx_0",
                        "blocks_throttle2_0")
RMS_NOISE, RMS_SIG, DIV, LOG = ("blocks_rms_xx_0", "blocks_rms_xx_0_0", "blocks_divide_xx_0",
                                "blocks_nlog10_ff_0")
LPF = "zoom_filter"


# ----------------------------------------------------------------------------- spec-derived expectations
TAPS = np.array(firdes.low_pass(1, FS, CUTOFF, TWIDTH))     # the stated design (LPF block default: Hamming)
_NF = 1 << 18
_F = np.fft.fftfreq(_NF, 1 / FS)
_H2 = np.abs(np.fft.fft(TAPS, _NF)) ** 2
_G = ((_F + FS_OUT / 2) % FS_OUT) - FS_OUT / 2               # output frequency each input frequency aliases to
P_TONE_OUT = A_TONE ** 2 * float(np.interp(F_TONE, np.fft.fftshift(_F), np.fft.fftshift(_H2)))
P_NOISE_OUT = SIGMA ** 2 * float(np.sum(TAPS ** 2))          # noise power after the filter (decimation keeps it)
P_OUT = P_TONE_OUT + P_NOISE_OUT
SPEC_RMS = math.sqrt(P_OUT)                                  # 0.6037


def out_share(lo: float, hi: float) -> float:
    """Expected noise share of the 16 kHz output's power in [lo, hi] Hz (aliasing included; tone-free bands)."""
    assert not lo <= F_TONE <= hi
    m = (_G >= lo) & (_G <= hi)
    return SIGMA ** 2 * float(np.sum(_H2[m])) / _NF / P_OUT


PASS_BAND = (-PASS_EDGE, -PASS_EDGE + TWIDTH)            # -5250..-3750 Hz: the last 1.5 kHz of the passband
TRANS_BANDS = [(-STOP_EDGE, -PASS_EDGE), (PASS_EDGE, STOP_EDGE)]   # the transition band on each side
STOP_BANDS = [(-FS_OUT / 2, -STOP_EDGE), (STOP_EDGE, FS_OUT / 2)]   # 6.75-8 kHz on each side
PASS_FRAC = out_share(*PASS_BAND)                        # 0.0404
TRANS_FRAC = out_share(*TRANS_BANDS[0])                  # 0.0139
assert abs(out_share(*TRANS_BANDS[1]) / TRANS_FRAC - 1) < 1e-6
# a flat -30 dB stopband: every input frequency that lands in 6.75-8 kHz after decimation (4 x 1.25 kHz)
STOP_MAX = SIGMA ** 2 * DECIM * (FS_OUT / 2 - STOP_EDGE) / FS * 10 ** (-STOP_DB / 10) / P_OUT   # 1.35e-4
# Transition-band share: -45 % ... +58 % of the stated design's share. Derived from the taps of the designs
# the instruction admits (aliasing included): an equiripple design with the stated edges 5.25 / 6.75 kHz gives
# -35 % (0.5 dB ripple, 80 dB) to +46 % (0.01 dB ripple, 30 dB), because its half-amplitude point moves between
# 5.73 and 6.21 kHz with the ripple and attenuation chosen; GNU Radio's windows at the exact cutoff give -8 %
# (Hann) to +5 % (Kaiser); the noise realization adds about +-6 % (seed 42 reads -1.3 % / +4.2 %). For a
# Hamming design the bounds correspond to a cutoff of about 5.72-6.33 kHz (-5 % / +5.5 %); 5.6 and 6.4 kHz
# (+-6.7 %) give -60 % / +71 % and fail. Ripple above 0.5 dB is not covered: it breaks "unity gain within 5 %".
TRANS_MIN_REL, TRANS_MAX_REL = -0.45, 0.58
PASS_TOL = 0.10      # +-10 % in power = +-5 % in amplitude: unity gain up to 5.25 kHz


# ----------------------------------------------------------------------------- flowgraph helpers
def blk(data: dict, name: str) -> dict:
    return next(b for b in data["blocks"] if b["name"] == name)


def setp(data: dict, name: str, **params) -> dict:
    d = copy.deepcopy(data)
    blk(d, name)["parameters"].update({k: str(v) for k, v in params.items()})
    return d


def tidy(data: dict) -> dict:
    """Compact form: no `states`, no empty housekeeping parameters, no catalog-default parameters."""
    d = copy.deepcopy(data)
    for b in d["blocks"]:
        b.pop("states", None)
        b["parameters"] = {k: v for k, v in b["parameters"].items() if str(v) != ""}
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in d["blocks"]}
    out = grcfile.compact(d, defaults)
    out["options"] = d["options"]
    return out


class _Dumper(yaml.SafeDumper):
    pass


def _repr_list(dumper, data):
    flow = all(not isinstance(x, (list, dict)) for x in data)   # connections: one line each
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _repr_list)


def dump(data: dict) -> str:
    return yaml.dump(data, Dumper=_Dumper, sort_keys=False, default_flow_style=False, width=120,
                     allow_unicode=True)


def insert_after(data: dict, upstream: str, downstream: str, new_blocks: list[dict]) -> dict:
    """Replace the connection upstream:0 -> downstream:0 by upstream -> new_blocks... -> downstream."""
    d = copy.deepcopy(data)
    d["connections"] = [c for c in d["connections"] if not (c[0] == upstream and c[2] == downstream)]
    chain = [upstream] + [b["name"] for b in new_blocks] + [downstream]
    d["blocks"] += copy.deepcopy(new_blocks)
    d["connections"] += [[a, "0", b, "0"] for a, b in zip(chain, chain[1:])]
    return d


# ----------------------------------------------------------------------------- start / reference
def build_start() -> tuple[dict, list[dict]]:
    seed, sinks = headless_seed(SEED)
    assert sorted(s["file"] for s in sinks) == sorted(FILES.values()), sinks
    d = copy.deepcopy(seed)
    d["options"]["parameters"] = {"id": "demo_spec_an", "generate_options": "no_gui",
                                  "output_language": "python"}
    # the disabled Qt time sink of the example has no effect; drop it
    d["blocks"] = [b for b in d["blocks"] if b["name"] != "qtgui_time_sink_x_0"]
    d = setp(d, "samp_rate", value=FS)
    d = setp(d, "fft_size", value=FFT_SIZE)
    d = setp(d, "snr", value=NOISE_DB)
    d = setp(d, SRC, amp=A_TONE, freq=f"float(samp_rate)/float(fft_size)*{TONE_BIN}+freq_offset")
    d = setp(d, RMS_NOISE, alpha=ALPHA)
    d = setp(d, RMS_SIG, alpha=ALPHA)
    return tidy(d), sinks


PERTURBATION = {"perturbed": True, "changes": [
    {"what": "sample rate (variable samp_rate)", "original": "100000", "new": str(FS)},
    {"what": "variable fft_size (sets the tone frequency)", "original": "1000", "new": str(FFT_SIZE)},
    {"what": "Signal Source frequency expression (bin multiplier)",
     "original": "float(samp_rate)/float(fft_size)*26+freq_offset",
     "new": f"float(samp_rate)/float(fft_size)*{TONE_BIN}+freq_offset (= {F_TONE:.0f} Hz)"},
    {"what": "Signal Source amplitude", "original": "1.0", "new": str(A_TONE)},
    {"what": "noise level in dB (variable snr; noise amplitude = 10^(snr/20))", "original": "0",
     "new": str(NOISE_DB)},
    {"what": "RMS blocks averaging constant alpha (both)", "original": ".0001", "new": ALPHA},
    {"what": "Qt range widgets freq_offset (Frequency Offset) and snr (Noise Level)",
     "original": "variable_qtgui_range (GUI sliders)", "new": "plain variables (seed tool)"},
    {"what": "Qt GUI sinks (frequency display, three number displays)", "original": "qtgui_* (GUI)",
     "new": "File Sinks " + ", ".join(FILES.values()) + " (seed tool, observation points)"},
    {"what": "disabled Qt time sink qtgui_time_sink_x_0", "original": "present (disabled)", "new": "removed"},
    {"what": "options block", "original": "generate_options qt_gui, run_options prompt, author 'mettus', "
                                          "category Custom, window_size/placement/sizing_mode set",
     "new": "id demo_spec_an, generate_options no_gui, output_language python (other options stripped)"},
]}


def lpf_block(name: str = LPF, decim: int = DECIM, **over) -> dict:
    p = {"type": "fir_filter_ccf", "decim": str(decim), "gain": "1", "samp_rate": "samp_rate",
         "cutoff_freq": "6e3", "width": "1.5e3", "win": "window.WIN_HAMMING", "beta": "6.76"}
    p.update({k: str(v) for k, v in over.items()})
    return {"name": name, "id": "low_pass_filter", "parameters": p}


def build_reference(start: dict) -> dict:
    return tidy(insert_after(start, THR, SINK_SPEC, [lpf_block()]))


def build_variants(start: dict) -> dict[str, dict]:
    v = {}
    # 1) Rational Resampler (interpolation 1, decimation 4) with explicit low-pass taps (firdes default
    #    window, Hamming)
    v["v1_rational_resampler"] = insert_after(start, THR, SINK_SPEC, [
        {"name": "rational_resampler_xxx_0", "id": "rational_resampler_xxx",
         "parameters": {"type": "ccf", "interp": "1", "decim": "4",
                        "taps": "firdes.low_pass(1.0, samp_rate, 6000, 1500)"}}])
    # 2) full-rate Low Pass Filter (Blackman window) followed by Keep 1 in N
    v["v2_filter_then_keep_one_in_n"] = insert_after(start, THR, SINK_SPEC, [
        lpf_block("channel_lpf", decim=1, win="window.WIN_BLACKMAN"),
        {"name": "keep_one_in_4", "id": "blocks_keep_one_in_n", "parameters": {"type": "complex", "n": "4"}}])
    # 3) Frequency Xlating FIR Filter (centre 0) with taps and decimation held in variables
    d = insert_after(start, THR, SINK_SPEC, [
        {"name": "xlate_zoom", "id": "freq_xlating_fir_filter_xxx",
         "parameters": {"type": "ccc", "decim": "zoom_decim", "taps": "zoom_taps", "center_freq": "0",
                        "samp_rate": "samp_rate"}}])
    d["blocks"] = [{"name": "zoom_decim", "id": "variable", "parameters": {"value": "4"}},
                   {"name": "zoom_taps", "id": "variable",
                    "parameters": {"value": "firdes.low_pass(1, samp_rate, 6e3, 1.5e3)"}}] + d["blocks"]
    v["v3_xlating_filter_vars"] = d
    # 4) decimating filter (Kaiser window) placed between the adder and the Throttle; the Throttle then
    #    paces the 16 kHz stream (sample values identical, only wall-clock pacing differs)
    d = copy.deepcopy(start)
    d["connections"] = [c for c in d["connections"] if not (c[0] == ADD and c[2] == THR)]
    d["blocks"].append(lpf_block("decim_lpf", win="window.WIN_KAISER", beta="6.76"))
    d["connections"] += [[ADD, "0", "decim_lpf", "0"], ["decim_lpf", "0", THR, "0"]]
    d = setp(d, THR, samples_per_second="samp_rate/4")
    v["v4_filter_before_throttle"] = d
    # 5) two-stage decimation (review round 1): a wide anti-alias stage at 64 kHz (flat to 7 kHz,
    #    stopband from 25 kHz) decimating by 2, then the stated 6 kHz / 1.5 kHz filter at 32 kHz by 2
    v["v5_two_stage_decimation"] = insert_after(start, THR, SINK_SPEC, [
        lpf_block("stage1", decim=2, cutoff_freq="16e3", width="18e3"),
        lpf_block("stage2", decim=2, samp_rate="samp_rate/2")])
    # 6) Polyphase Decimator with the stated low-pass taps (review round 1)
    v["v6_polyphase_decimator"] = insert_after(start, THR, SINK_SPEC, [
        {"name": "pfb_decim", "id": "pfb_decimator_ccf",
         "parameters": {"decim": "4", "taps": "firdes.low_pass(1, samp_rate, 6e3, 1.5e3)", "channel": "0"}}])
    # 7) FFT Filter decimating by 4 (review round 1)
    v["v7_fft_filter"] = insert_after(start, THR, SINK_SPEC, [
        {"name": "fft_zoom", "id": "fft_filter_xxx",
         "parameters": {"type": "ccf", "decim": "4", "taps": "firdes.low_pass(1, samp_rate, 6e3, 1.5e3)"}}])
    # 8) Low-Pass Filter Taps variable feeding a Decimating FIR Filter (review round 1), Hann window
    d = insert_after(start, THR, SINK_SPEC, [
        {"name": "fir_zoom", "id": "fir_filter_xxx", "parameters": {"type": "ccf", "decim": "4", "taps": "lp_taps"}}])
    d["blocks"] = [{"name": "lp_taps", "id": "variable_low_pass_filter_taps",
                    "parameters": {"gain": "1.0", "samp_rate": "samp_rate", "cutoff_freq": "6000", "width": "1500",
                                   "win": "window.WIN_HANN", "beta": "6.76"}}] + d["blocks"]
    v["v8_taps_variable_hann_fir"] = d
    # 9) an equiripple (Parks-McClellan) design with the stated band edges 5.25 / 6.75 kHz, 40 dB stopband
    d = insert_after(start, THR, SINK_SPEC, [
        {"name": "remez_zoom", "id": "fir_filter_xxx",
         "parameters": {"type": "ccf", "decim": "4",
                        "taps": "optfir.low_pass(1, samp_rate, 5250, 6750, 0.1, 40)"}}])
    d["blocks"].append({"name": "imp_optfir", "id": "import",
                        "parameters": {"imports": "from gnuradio.filter import optfir"}})
    v["v9_equiripple_band_edges"] = d
    # 10) firdes.low_pass_2 with a 40 dB stopband (review round 1: one of the reviewer's correct designs)
    # 12-14) equiripple designs with the stated band edges across the ripple / attenuation range the instruction
    #        admits (review round 2): 0.1 dB / 30 dB, 0.01 dB / 30 dB (highest transition share), 0.5 dB / 80 dB
    #        (lowest)
    for name, ripple, atten in (("v12_equiripple_0p1db_30db", "0.1", "30"), ("v13_equiripple_0p01db_30db", "0.01", "30"),
                                ("v14_equiripple_0p5db_80db", "0.5", "80")):
        d = insert_after(start, THR, SINK_SPEC, [
            {"name": "remez_zoom", "id": "fir_filter_xxx",
             "parameters": {"type": "ccf", "decim": "4",
                            "taps": f"optfir.low_pass(1, samp_rate, 5250, 6750, {ripple}, {atten})"}}])
        d["blocks"].append({"name": "imp_optfir", "id": "import",
                            "parameters": {"imports": "from gnuradio.filter import optfir"}})
        v[name] = d
    v["v10_low_pass_2_40db"] = insert_after(start, THR, SINK_SPEC, [
        {"name": "lp2_zoom", "id": "fir_filter_xxx",
         "parameters": {"type": "ccf", "decim": "4", "taps": "firdes.low_pass_2(1, samp_rate, 6000, 1500, 40)"}}])
    # 11) review round 1: the Throttle removed (the filter fed straight from the adder) and another noise seed;
    #     neither the Throttle nor the noise realization is part of the request
    d = copy.deepcopy(start)
    d["blocks"] = [b for b in d["blocks"] if b["name"] != THR] + [lpf_block()]
    d["connections"] = [c for c in d["connections"] if THR not in (c[0], c[2])]
    d["connections"] += [[ADD, "0", LPF, "0"], [LPF, "0", SINK_SPEC, "0"]]
    v["v11_no_throttle_seed0"] = setp(d, NOISE, seed="0")
    return {k: tidy(x) for k, x in v.items()}


# ----------------------------------------------------------------------------- task record
INSTRUCTION = (
    "This flowgraph runs at a sample rate of 64 kHz (64,000 samples per second). It adds a complex tone "
    "at 3.1 kHz with amplitude 0.5 to complex Gaussian noise with an RMS amplitude of about 0.794 (-2 dB) "
    "and, after a Throttle, writes the noisy signal as complex samples to the File Sink whose file is "
    f"{FILES['spec']}; this is the signal we look at on a spectrum display. Three more File Sinks record "
    "float measurements taken on the clean tone and on the noise before the two are added, by RMS meters that "
    f"average with alpha 0.0005: the tone's RMS level in {FILES['sig']} (0.5), the noise's RMS level in "
    f"{FILES['noise']} (about 0.794) and the resulting SNR in dB in {FILES['snr']} (about -4 dB).\n\n"
    "I want the spectrum view zoomed in on the band around the tone. In the path to "
    f"{FILES['spec']} only, low-pass filter the noisy signal with a cutoff frequency of 6 kHz and a "
    "transition width of 1.5 kHz centred on the cutoff, as GNU Radio's filter design defines it: unity gain "
    "up to 5.25 kHz and the stopband from 6.75 kHz. Then decimate by 4, so that this File Sink receives "
    "complex samples at 16 kHz. In that 16 kHz output, the noise between 6.75 and 8 kHz on each side must be "
    "at least 30 dB weaker than it would be after decimating without a filter. Any filter design that meets "
    "these band edges is fine (for example a windowed or an equiripple design). "
    "Do not shift the signal in frequency: the tone must still appear at +3.1 kHz in the 16 kHz output. Keep "
    "the 64 kHz sample rate, the tone, the noise level and the three measurement outputs (including how their "
    "meters average) as they are; the meters must keep measuring the unfiltered tone and noise. "
    "The flowgraph should keep producing samples until it is stopped; do not limit the number of samples. "
    "The tone frequency must be right within 1 %, levels within 5 % (the SNR within 0.4 dB), and the "
    "filter's response must fall to half amplitude within about 5 % of 6 kHz."
)

# Taps never back-pressure the graph (tooling T7): the run lasts until every tap holds N_ITEMS. The
# spectrum tap runs at 16 kHz, so it needs 4 x 32768 source samples (2.05 s behind the 64 kHz Throttle).
N_ITEMS = 32768
SPEC_SKIP, SPEC_N = 512, 32000              # filter transient < 100 output items; 0.5 Hz bins, the 3.1 kHz tone
assert SPEC_SKIP + SPEC_N <= N_ITEMS and (F_TONE * SPEC_N) % FS_OUT == 0   # sits exactly on bin 6200
METER_SKIP, METER_N = 12000, 20000          # RMS averages settle with time constant 1/alpha = 2000 samples
SETTLE_N = 4000                             # the tone meter's first 4000 samples (two time constants)
_a = float(ALPHA)
SETTLE_MEAN = float(np.mean(A_TONE * np.sqrt(1 - (1 - _a) ** (np.arange(SETTLE_N) + 1))))   # 0.3639
_a0 = 0.0001
SETTLE_MEAN_ORIG = float(np.mean(A_TONE * np.sqrt(1 - (1 - _a0) ** (np.arange(SETTLE_N) + 1))))   # 0.1989

SPEC = {
    "rate": {"id": "rate_and_tone",
             "text": (f"the spectrum output {FILES['spec']} carries complex samples at 16 kHz (the 64 kHz signal "
                      "decimated by 4) and shows the tone at +3.1 kHz (+-1 %), not shifted in frequency"),
             "checked_by": "f2p peak_freq at fs 16 kHz (no decimation gives 775 Hz, decimation 2 gives 1550 Hz, a "
                           "filter centred on the tone gives ~0 Hz)"},
    "gain": {"id": "unity_gain",
             "text": ("unity gain: output RMS = sqrt(0.5^2 + 0.794^2 * sum(h^2)) = "
                      f"{SPEC_RMS:.4f} +-5 %, h = the stated filter's taps"),
             "checked_by": "f2p rms on the spectrum output"},
    "pass": {"id": "passband",
             "text": ("the passband reaches 5.25 kHz at unity gain: share of the output power in -5.25..-3.75 kHz "
                      f"(noise only) {PASS_FRAC:.4f} +-10 % (amplitude +-5 %)"),
             "checked_by": "f2p band_fraction -5250..-3750 Hz"},
    "trans": {"id": "cutoff",
              "text": ("cutoff 6 kHz with the 1.5 kHz transition centred on it (GNU Radio convention): share of the "
                       "output power in each transition band 5.25-6.75 kHz between "
                       f"{TRANS_FRAC * (1 + TRANS_MIN_REL):.4f} and {TRANS_FRAC * (1 + TRANS_MAX_REL):.4f} "
                       f"({TRANS_FRAC:.4f} -45 % / +58 %): the half-amplitude point within about 5 % of 6 kHz; "
                       "every equiripple design with the stated band edges and at most 0.5 dB ripple passes"),
              "checked_by": "f2p band_fraction -6750..-5250 Hz and 5250..6750 Hz"},
    "stop": {"id": "stopband",
             "text": ("in the 16 kHz output the noise between 6.75 and 8 kHz on each side is at least 30 dB weaker "
                      "than after decimating without a filter: share of the output power in 6.75-8 kHz on each side "
                      f"at most {STOP_MAX:.3g}, what a flat -30 dB stopband passes (4 x 1.25 kHz of input noise alias "
                      "into each band); this also bounds the transition width to about 2.45 kHz and rules out "
                      "reading 6 kHz as the passband edge"),
             "checked_by": "f2p band_fraction 6750..8000 Hz and -8000..-6750 Hz (max)"},
    "sig": {"id": "tone_meter", "text": f"the tone RMS measurement in {FILES['sig']} stays 0.5 (unfiltered tone)",
            "checked_by": "p2p mean"},
    "noise": {"id": "noise_meter",
              "text": f"the noise RMS measurement in {FILES['noise']} stays about 0.794 (unfiltered noise)",
              "checked_by": "p2p mean"},
    "snr": {"id": "snr_meter", "text": f"the SNR measurement in {FILES['snr']} stays about -4.0 dB (+-0.4 dB)",
            "checked_by": "p2p mean"},
    "settle": {"id": "meter_averaging",
               "text": ("the meters keep averaging with alpha 0.0005: the tone meter's first 4000 samples average "
                        f"{SETTLE_MEAN:.4f} +-5 % (0.5 * sqrt(1 - (1 - alpha)^(n+1)); the example's alpha 0.0001 "
                        f"gives {SETTLE_MEAN_ORIG:.4f})"),
               "checked_by": "p2p mean over the first 4000 samples of the tone meter"},
    "run": {"id": "run_length", "text": "the flowgraph keeps producing samples until it is stopped",
            "checked_by": f"V5 liveness: at least {N_ITEMS} items at every observation point"},
    "outputs": {"id": "outputs", "text": "the four File Sinks keep their file names and item types",
                "checked_by": "observation contract (a missing sink fails V6)"},
}

ORACLE = {"checks": [
    {"point": "spec", "skip": SPEC_SKIP, "n": SPEC_N, "feature": "peak_freq", "fs": FS_OUT, "expect": F_TONE,
     "tol": 0.01 * F_TONE, "group": "f2p", "property": "rate_and_tone"},
    {"point": "spec", "skip": SPEC_SKIP, "n": SPEC_N, "feature": "rms", "expect": round(SPEC_RMS, 4),
     "rel_tol": 0.05, "group": "f2p", "property": "unity_gain"},
    {"point": "spec", "skip": SPEC_SKIP, "n": SPEC_N, "feature": "band_fraction", "fs": FS_OUT,
     "lo": PASS_BAND[0], "hi": PASS_BAND[1], "expect": round(PASS_FRAC, 5), "rel_tol": PASS_TOL,
     "group": "f2p", "property": "passband"},
    *[{"point": "spec", "skip": SPEC_SKIP, "n": SPEC_N, "feature": "band_fraction", "fs": FS_OUT,
       "lo": lo, "hi": hi, "min": round(TRANS_FRAC * (1 + TRANS_MIN_REL), 5),
       "max": round(TRANS_FRAC * (1 + TRANS_MAX_REL), 5), "group": "f2p", "property": "cutoff"}
      for lo, hi in TRANS_BANDS],
    *[{"point": "spec", "skip": SPEC_SKIP, "n": SPEC_N, "feature": "band_fraction", "fs": FS_OUT,
       "lo": lo, "hi": hi, "max": float(f"{STOP_MAX:.3g}"), "group": "f2p", "property": "stopband"}
      for lo, hi in STOP_BANDS],
    {"point": "sig", "skip": METER_SKIP, "n": METER_N, "feature": "mean", "expect": A_TONE, "rel_tol": 0.05,
     "group": "p2p", "property": "tone_meter"},
    {"point": "noise", "skip": METER_SKIP, "n": METER_N, "feature": "mean", "expect": round(SIGMA, 4),
     "rel_tol": 0.05, "group": "p2p", "property": "noise_meter"},
    {"point": "snr", "skip": METER_SKIP, "n": METER_N, "feature": "mean", "expect": round(SNR_DB, 3),
     "tol": 0.4, "group": "p2p", "property": "snr_meter"},
    {"point": "sig", "skip": 0, "n": SETTLE_N, "feature": "mean", "expect": round(SETTLE_MEAN, 4),
     "rel_tol": 0.05, "group": "p2p", "property": "meter_averaging"},
]}

MUTATIONS = [
    {"op": "set_param", "block": LPF, "param": "decim", "value": "1", "breaks": "rate_and_tone"},
    {"op": "set_param", "block": LPF, "param": "decim", "value": "2", "breaks": "rate_and_tone"},
    {"op": "set_param", "block": "freq_offset", "param": "value", "value": "100", "breaks": "rate_and_tone",
     "note": "near miss: tone moved to 3.2 kHz (+3 %)"},
    {"op": "bypass", "block": LPF, "breaks": "rate_and_tone", "note": "no filter and no decimation"},
    {"op": "set_param", "block": LPF, "param": "cutoff_freq", "value": "12e3", "breaks": "cutoff"},
    {"op": "set_param", "block": LPF, "param": "cutoff_freq", "value": "3e3", "breaks": "passband",
     "note": "cutoff 3 kHz: the passband and the tone are cut"},
    {"op": "set_param", "block": LPF, "param": "cutoff_freq", "value": "5.6e3", "breaks": "cutoff",
     "note": "near miss: cutoff -6.7 %"},
    {"op": "set_param", "block": LPF, "param": "cutoff_freq", "value": "6.4e3", "breaks": "cutoff",
     "note": "near miss: cutoff +6.7 %"},
    {"op": "set_param", "block": LPF, "param": "cutoff_freq", "value": "6.75e3", "breaks": "stopband",
     "note": "6 kHz read as the passband edge (transition 6-7.5 kHz)"},
    {"op": "set_param", "block": LPF, "param": "gain", "value": "4", "breaks": "unity_gain",
     "note": "gain 4 (decimation 'compensation') instead of unity"},
    {"op": "set_param", "block": LPF, "param": "gain", "value": "1.1", "breaks": "unity_gain",
     "note": "near miss: gain 1.1"},
    {"op": "set_param", "block": LPF, "param": "width", "value": "6e3", "breaks": "stopband"},
    {"op": "set_param", "block": LPF, "param": "width", "value": "3e3", "breaks": "stopband",
     "note": "near miss: transition twice as wide"},
    {"op": "set_param", "block": LPF, "param": "win", "value": "window.WIN_RECTANGULAR", "breaks": "stopband",
     "note": "rectangular window: only ~21 dB of stopband suppression"},
    {"op": "set_param", "block": "samp_rate", "param": "value", "value": "48000", "breaks": "cutoff",
     "note": "48 kHz: the filter edges move relative to the output band"},
    {"op": "set_param", "block": SRC, "param": "amp", "value": "1.0", "breaks": "tone_meter"},
    {"op": "set_param", "block": "snr", "param": "value", "value": "0", "breaks": "noise_meter"},
    {"op": "rewire_input", "dst": RMS_NOISE, "dst_port": "0", "new_src": ADD, "breaks": "noise_meter",
     "note": "noise meter measures tone plus noise"},
    {"op": "rewire_input", "dst": RMS_SIG, "dst_port": "0", "new_src": ADD, "breaks": "tone_meter",
     "note": "tone meter measures tone plus noise"},
    {"op": "set_param", "block": RMS_SIG, "param": "alpha", "value": ".0001", "breaks": "meter_averaging",
     "note": "the example's averaging constant restored on the tone meter"},
]

DISJOINTNESS = (
    "Inserts a decimating low-pass 'zoom' filter into only the display path of a tone-plus-noise SNR-meter "
    "demo (test-split seed channels/demo_spec_an.grc) while its RMS/SNR measurement paths stay unfiltered; "
    "training-side specs that add or fix filters do so in other graphs and roles (22_fm_rx_filter_squelch: "
    "channel filter and squelch in an FM receiver; t12_fivegnalizer_modify: FFT load in a 5G graph; "
    "t20_resample_ratio_repair: 48k-to-44.1k resampler), and none uses this seed."
)

NOTES = (
    "Measured on the reference (noise seed 42, so every run is identical): see gates.json and the resolution "
    "record. Expected values are computed from the stated numbers and the stated design's taps "
    f"({len(TAPS)} taps, sum h^2 = {float(np.sum(TAPS ** 2)):.5f}); the RMS expectation uses the taps' noise "
    "bandwidth, not an ideal 2 x cutoff (review F2). Liveness: taps never back-pressure the graph (tooling T7); "
    f"every tap must hold {N_ITEMS} items, which the 16 kHz spectrum tap reaches after 2.05 s of the 64 kHz "
    "Throttle. The 64 kHz rate itself is observable only through frequencies relative to it; a 48 kHz mutant is "
    "caught because the filter edges move relative to the output band."
)

REVIEW_NOTES = (
    "Review rounds 1 and 2: see reports/tier_f_reviews/round1/gbf-test-15.resolution.json. The instruction "
    "names GNU Radio's cutoff convention as band edges (unity gain up to 5.25 kHz, stopband from 6.75 kHz), "
    "states the 30 dB figure as what is measured (the noise in 6.75-8 kHz of the 16 kHz output, against "
    "decimating without a filter), allows any design that meets the edges, and bounds the half-amplitude point "
    "to about 5 % of 6 kHz. An equiripple design with the stated edges puts its half-amplitude point anywhere "
    "from about 5.73 to 6.21 kHz depending on ripple and attenuation, so the transition-band check accepts -45 % "
    "to +58 % of the stated design's share; every such design with at most 0.5 dB ripple passes (variants v9, "
    "v12-v14 cover 0.01-0.5 dB and 30-80 dB). Consequences the team should confirm: a filter reading 6 kHz as "
    "the passband edge (6-7.5 kHz), one reading it as the stopband edge (4.5-6 kHz), a rectangular window "
    "(about 29 dB in the output band) and firdes.low_pass_2 with 30 dB (its real transition is about 3.6 kHz "
    "wide) fail the stopband or transition checks; a cascade of two identical 6 kHz / 1.5 kHz stages "
    "(half-amplitude point near 5.75 kHz) and a properly designed two-stage decimator (v5) pass. Known oracle "
    "limits: for a windowed design the half-amplitude point is bounded to about 5.72-6.33 kHz (5.6 and 6.4 kHz "
    "fail); a transition sharper than stated passes as long as it meets the band edges and is centred near "
    "6 kHz; ripple above 0.5 dB can pass the passband check although it breaks 'levels within 5 %'. The "
    "Throttle and the noise distribution are not required (the instruction keeps only the noise level). "
    "Licence: derived from the GNU Radio 3.10.12 example channels/demo_spec_an.grc (GPL-3.0-or-later, GNU Radio "
    "project; example author field 'mettus'); the licence decision is the team's."
)


def main() -> int:
    split_rec = read_json(SPLITS / "flowgraph_seed_split.json")["seeds"][SEED]
    assert split_rec["split"] == SPLIT, f"seed split is {split_rec['split']}, task split is {SPLIT}"
    sha = sha256_file(GNURADIO_EXAMPLES / SEED)
    assert sha == split_rec["sha256"], "example file differs from the inventoried seed"
    opts = yaml.safe_load((GNURADIO_EXAMPLES / SEED).read_text(encoding="utf-8"))["options"]["parameters"]
    assert opts["author"] == "mettus" and opts["generate_options"] == "qt_gui"

    start, _ = build_start()
    ref = build_reference(start)
    variants = build_variants(start)
    functional = [b for b in ref["blocks"] if not b["id"].startswith(("variable", "import"))]
    assert 6 <= len(functional) <= 15, len(functional)      # L2: 6-15 functional blocks, rate change, filtering

    record = {
        "id": TASK_ID, "tier": "F", "split": SPLIT, "version": 3, "canary": CANARY,
        "license": f"GPL-3.0-or-later (derived from GNU Radio 3.10.12 example {SEED})",
        "source": {"kind": "gnuradio_example", "ref": SEED, "sha256": sha,
                   "derivation": ("Headless export by `./gb tier-f seed`: the Qt frequency sink and the three Qt "
                                  "number sinks became File Sinks named after them; the Qt range widgets became plain "
                                  "variables; the disabled Qt time sink was dropped; GUI options stripped.")},
        "cluster_id": SEED,
        "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring",
                       "generator_version": "0.2.0"},
        "review": {"status": "pending-team-review", "reviewers": [], "notes": REVIEW_NOTES},
        "family": "modify", "level": "L2",
        "instruction": INSTRUCTION,
        "start_grc": "start.grc", "reference_grc": "reference.grc",
        "observation": {
            "points": [{"name": k, "sink": {"block_id": "blocks_file_sink",
                                            "params": {"file": f, "type": "complex" if k == "spec" else "float"}}}
                       for k, f in FILES.items()],
            "run": {"n_items": N_ITEMS, "min_items": N_ITEMS, "timeout_s": 10},
            "contract": (f"the complex samples written to the File Sink whose file name is {FILES['spec']}, "
                         f"and the float samples written to the File Sinks {FILES['sig']}, {FILES['noise']} "
                         f"and {FILES['snr']}"),
        },
        "oracle": ORACLE,
        "mutations": MUTATIONS,
        "spec_properties": list(SPEC.values()),
        "perturbation": PERTURBATION,
        "disjointness": DISJOINTNESS,
        "notes": NOTES,
    }
    files = {"start.grc": dump(start), "reference.grc": dump(ref)}
    files.update({f"variants/{k}.grc": dump(v) for k, v in variants.items()})
    errors = taskio.save(HERE, record, files)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"wrote {HERE} ({len(files)} flowgraphs); SPEC_RMS {SPEC_RMS:.5f} PASS {PASS_FRAC:.5f} TRANS {TRANS_FRAC:.5f} "
          f"STOP_MAX {STOP_MAX:.4g} SETTLE {SETTLE_MEAN:.4f} (orig alpha {SETTLE_MEAN_ORIG:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
