"""Authoring script for GRC-Bench Tier-F item gbf-test-03 (family build, level L1, composed chain).

Task: band-limited Gaussian noise. Float Gaussian noise (standard deviation 2.0) at a 40 kHz sample
rate goes through a band-pass filter (cutoffs 6 and 9 kHz, transition width at most 500 Hz, unity
passband gain) and is written as float samples to a File Sink "bandnoise.dat". The flowgraph runs until
it is stopped. v2 signature filter|noise -> test (this item keeps the cell; its former dev twin
gbf-dev-02 was re-themed in the 18 Oct fix plan).

Deterministic: regenerates task.json, reference.grc and variants/*.grc (gates.json is written by
`./gb tier-f gates`). Needs GNU Radio (catalog defaults for the compact form), so run it in WSL:

    cd grc-bench && PYTHONPATH=src QT_QPA_PLATFORM=offscreen ~/grc-agent/.venv/bin/python \
        items/tier_f/test/gbf-test-03/build.py            # write the task
    ... build.py --probes                                  # ladder verdicts of the counter-examples
    ./gb tier-f gates items/tier_f/test/gbf-test-03

Round-1 fixes (reports/tier_f_reviews/round1/gbf-test-03.resolution.json)
  * run length stated with the set sentence (R5); "so that every run writes the same samples" and the
    unobservable seed requirement are gone (the reference keeps a fixed seed only for gate determinism);
  * the design numbers (cutoffs 6 / 9 kHz, transition width at most 500 Hz) are checked through stated
    acceptance criteria on narrow strips at the band edges (below), with near-miss mutants;
  * "Gaussian" and "band-pass filter it" are checked structurally (graph checks), because band-limited
    noise is Gaussian whatever the source distribution and tones can imitate a power spectrum;
  * the level window is stated as RMS 0.69-0.80, which every Hamming, Hann, Blackman and Kaiser design
    meets on every seed (rectangular windows: see the RMS bullet below);
  * instruction rewritten as a request about the open (empty) flowgraph (16 Oct decision, R3).

Why the thresholds hold (GNU Radio 3.10.12 firdes; fixer calibration t03_calib.py: noise-free |H|^2
shares plus 300 noise seeds per design through oracles.band_fraction after the 18 Oct Parseval fix,
61440 analysed samples). Real zero-mean noise has no DC line, so the R12 change only fixes the
denominator (every non-DC bin counted twice) and the shares below are true power shares.
  * in-band 6-9 kHz: valid designs 0.984-0.999 in theory (>= 0.90 required);
  * edge strips 6.0-6.2 / 8.8-9.0 kHz (>= 2 % each, "the passband reaches both cutoffs"): valid designs
    0.033-0.064 in theory, worst seed 0.028 (rectangular window); a cutoff moved inward by 250 Hz
    (lo 6250 / hi 8750) gives 0.003;
  * skirt strips 5.5-5.8 / 9.2-9.5 kHz (<= 0.1 % each, "steep skirts"): 500 Hz designs 2.8e-5 (Kaiser)
    to 4.0e-4 (rectangular) in theory, worst seed 5.2e-4 (Hann); rectangular 400 Hz 7.7e-4; a doubled
    width (1 kHz) gives 1.5e-3 (Kaiser) to 4.8e-3 and fails 300/300 for every window; a cutoff moved
    outward by 250 Hz (lo 5750 / hi 9250) gives 1.1e-2;
  * stopbands below 5 kHz / above 10 kHz (< 1 % each): valid designs <= 8e-4;
  * RMS: sigma * sqrt(sum h^2) is 0.747-0.773 for Hamming, Hann and Blackman designs with transitions of
    50-500 Hz; the seed-to-seed spread is 0.7 % (1 sigma), so the stated window 0.69-0.80 is >= 4 sigma
    from each of them. Rectangular windows range from 0.716 (500 Hz) to 0.805 (350-360 Hz): GNU Radio
    normalises the band-pass gain at the band centre, and for those widths the centre sits in a ripple
    trough, so the average passband gain is about 4 % high and the design fails the stated RMS window
    on most seeds (round-2 check, 22 Oct).
    A 10 % gain error (gain 1.1: 0.812-0.845) and a 15 % level error (amplitude 1.7: 0.627-0.653) fail
    on every seed; 5 % errors (gain 1.05, amplitude 1.9) mostly pass (valid windows already differ by 8 %).
  * effective limits (300 seeds): transition width up to about 550 Hz (Hann; 600 Hz passes 66 %),
    700 Hz (Hamming; 800 Hz passes 6 %), 800 Hz (Blackman, Kaiser; 900 Hz 0 % / 22 %), 500 Hz
    (rectangular; 600 Hz 0 %); cutoff shifts of 100 Hz are borderline (lo 6100: 45 %, lo 5900: 21 %),
    250 Hz shifts fail 300/300. A Bartlett window (firdes taps only; not in the Band Pass Filter menu)
    at 500 Hz puts 3.0e-3 in each skirt strip and fails the stated skirt criterion.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))

from grcbench.common.canary import CANARY  # noqa: E402
from grcbench.tier_f import compose as C  # noqa: E402
from grcbench.tier_f import edit, grc, grcfile, taskio  # noqa: E402

TASK_ID = "gbf-test-03"
SPLIT = "test"

# ---- the specification (every number here is stated in the instruction) ----
FS = 40000               # sample rate, Hz (the output file is read at this rate)
SIGMA = 2.0              # standard deviation of the Gaussian noise
F_LO, F_HI = 6000, 9000  # band-pass cutoffs, Hz
WIDTH = 500              # transition width, Hz (at most)
OUT_FILE = "bandnoise.dat"
IN_BAND_MIN = 0.90                          # share of power in [F_LO, F_HI]
EDGE_STRIPS = ((6000, 6200), (8800, 9000))  # inside the band at each cutoff: >= EDGE_MIN each
EDGE_MIN = 0.02
SKIRT_STRIPS = ((5500, 5800), (9200, 9500))  # just outside the transition bands: <= SKIRT_MAX each
SKIRT_MAX = 0.001
STOP_LO, STOP_HI = 5000, 10000              # stopbands: < STOP_MAX each
STOP_MAX = 0.01
RMS_MIN, RMS_MAX = 0.69, 0.80               # output level (sigma 2.0 through a unity-gain 6-9 kHz band)
SEED = 2718              # reference only (gate determinism); not part of the specification

# ---- oracle analysis window ----
# 65536 captured items; the first 4096 are skipped (FIR start-up: a filter of up to ~4000 taps, e.g. a
# 50 Hz Blackman-Harris design, has filled its history) and the remaining 61440 are analysed.
N_ITEMS, SKIP = 65536, 4096

INSTRUCTION = (
    "In this flowgraph, make band-limited Gaussian noise at a sample rate of 40 kHz. Generate real-valued "
    "(float) Gaussian noise with a standard deviation of 2.0 and pass it through a band-pass filter with "
    "cutoffs at 6 kHz and 9 kHz, a transition width of no more than 500 Hz and unity gain in the passband. "
    "Write the filtered signal, still at 40 kHz, as float samples to a File Sink whose file name is "
    "bandnoise.dat. The flowgraph should keep producing samples until it is stopped; do not limit the "
    "number of samples. Measured on bandnoise.dat after its first 4096 samples, the output must meet all "
    "of these: at least 90 % of the power lies between 6 and 9 kHz; the passband reaches both cutoffs, "
    "with at least 2 % of the power between 6.0 and 6.2 kHz and at least 2 % between 8.8 and 9.0 kHz; the "
    "skirts are steep, with at most 0.1 % of the power between 5.5 and 5.8 kHz and at most 0.1 % between "
    "9.2 and 9.5 kHz; less than 1 % of the power lies below 5 kHz and less than 1 % above 10 kHz; and the "
    "RMS level is between 0.69 and 0.80."
)

SPEC_PROPERTIES = [
    "P1 sample rate: the output in bandnoise.dat is at 40 kHz (every spectral check is computed with "
    "fs = 40 kHz, so a graph at another rate, or a rate change before the file, puts the band elsewhere "
    "and fails P2-P5)",
    "P2 passband: at least 90 % of the output power lies between 6 and 9 kHz",
    "P3 cutoffs at 6 and 9 kHz (the passband reaches both cutoffs): at least 2 % of the power between 6.0 "
    "and 6.2 kHz and at least 2 % between 8.8 and 9.0 kHz",
    "P4 transition width at most 500 Hz (steep skirts): at most 0.1 % of the power between 5.5 and 5.8 kHz "
    "and at most 0.1 % between 9.2 and 9.5 kHz",
    "P5 stopbands: less than 1 % of the power below 5 kHz and less than 1 % above 10 kHz",
    "P6 level: output RMS between 0.69 and 0.80, the range that Gaussian noise of standard deviation 2.0 "
    "through a unity-gain 6-9 kHz band-pass with transitions of at most 500 Hz gives for any window "
    "choice, plus the estimation spread (noise level and passband gain are checked jointly through it)",
    "P7 band-pass filtered Gaussian noise (structural, graph checks): the File Sink is reached from a noise "
    "source through a filter block, and a noise source is set to Gaussian noise (band-limited noise is "
    "Gaussian whatever the source distribution, and tones can imitate a power spectrum, so neither is "
    "observable in the samples)",
    "P8 observation: the filtered signal is written as float samples to the File Sink whose file name is "
    "bandnoise.dat (the observation point requires that file name and item type); P2-P6 are measured "
    "after its first 4096 samples",
    "P9 run condition (V5): the flowgraph keeps producing samples until it is stopped; V5 liveness needs "
    f"{N_ITEMS} items at bandnoise.dat (1.64 s at 40 kHz), so a graph that stops itself earlier fails V5",
]


def _band(lo: float, hi: float, prop: str, **lim) -> dict:
    return {"point": "out", "feature": "band_fraction", "fs": FS, "lo": lo, "hi": hi, "skip": SKIP,
            "group": "main", "property": prop, **lim}


NOISE_IDS = ["analog_noise_source_x", "analog_fastnoise_source_x"]
FILTER_IDS = ["band_pass_filter", "low_pass_filter", "high_pass_filter", "band_reject_filter",
              "fir_filter_xxx", "fft_filter_xxx", "interp_fir_filter_xxx", "iir_filter_xxx",
              "filter_fft_low_pass_filter", "rational_resampler_xxx", "pfb_arb_resampler_xxx",
              "freq_xlating_fir_filter_xxx", "freq_xlating_fft_filter_ccc", "single_pole_iir_filter_xx",
              "dc_blocker_xx", "blocks_moving_average_xx"]

ORACLE = {
    "checks": [
        _band(F_LO, F_HI, "P2", min=IN_BAND_MIN),
        _band(*EDGE_STRIPS[0], "P3", min=EDGE_MIN),
        _band(*EDGE_STRIPS[1], "P3", min=EDGE_MIN),
        _band(*SKIRT_STRIPS[0], "P4", max=SKIRT_MAX),
        _band(*SKIRT_STRIPS[1], "P4", max=SKIRT_MAX),
        _band(0, STOP_LO, "P5", max=STOP_MAX),
        _band(STOP_HI, FS / 2, "P5", max=STOP_MAX),
        {"point": "out", "feature": "rms", "min": RMS_MIN, "max": RMS_MAX, "skip": SKIP, "group": "main",
         "property": "P6"},
    ],
    "graph_checks": [
        {"kind": "path", "from_ids": NOISE_IDS, "via_ids": FILTER_IDS, "to_ids": ["blocks_file_sink"],
         "group": "main", "property": "P7 noise source -> filter -> File Sink"},
        {"kind": "block_present", "ids": NOISE_IDS, "params": {"noise_type": "analog.GR_GAUSSIAN"},
         "group": "main", "property": "P7 Gaussian noise source"},
    ],
}

MUTATIONS = [
    {"op": "set_param", "block": "bpf", "param": "low_cutoff_freq", "value": "6250",
     "breaks": "P3 lower cutoff 6 kHz (moved up 250 Hz: 6.0-6.2 kHz holds about 0.3 %)"},
    {"op": "set_param", "block": "bpf", "param": "low_cutoff_freq", "value": "5750",
     "breaks": "P4 lower skirt (cutoff moved down 250 Hz: about 1 % in 5.5-5.8 kHz)"},
    {"op": "set_param", "block": "bpf", "param": "high_cutoff_freq", "value": "9250",
     "breaks": "P4 upper skirt (cutoff moved up 250 Hz: about 1 % in 9.2-9.5 kHz)"},
    {"op": "set_param", "block": "bpf", "param": "high_cutoff_freq", "value": "8750",
     "breaks": "P3 upper cutoff 9 kHz (moved down 250 Hz: 8.8-9.0 kHz holds about 0.3 %)"},
    {"op": "scale_param", "block": "bpf", "param": "width", "factor": 2.0,
     "breaks": "P4 transition width at most 500 Hz (doubled to 1 kHz: about 0.2 % in each skirt strip)"},
    {"op": "bypass", "block": "bpf",
     "breaks": "P2/P7 band-pass filtering (filter removed: white noise in the file)"},
    {"op": "set_param", "block": "bpf", "param": "gain", "value": "1.1",
     "breaks": "P6 unity passband gain (10 % too loud: RMS about 0.83)"},
    {"op": "set_param", "block": "noise", "param": "amp", "value": "1.7",
     "breaks": "P6 noise standard deviation 2.0 (15 % too quiet: RMS about 0.64)"},
    {"op": "set_param", "block": "noise", "param": "noise_type", "value": "analog.GR_UNIFORM",
     "breaks": "P7 Gaussian noise (uniform noise of amplitude 2; also P6: RMS about 0.43)"},
    {"op": "set_param", "block": "samp_rate", "param": "value", "value": "48000",
     "breaks": "P1 sample rate 40 kHz (read at 40 kHz the band sits at 5-7.5 kHz)"},
]


# ---- flowgraphs ----
def _noise(name: str = "noise", bid: str = "analog_noise_source_x", seed: int | None = SEED,
           noise_type: str = "analog.GR_GAUSSIAN", amp=SIGMA) -> dict:
    params = {"type": "float", "noise_type": noise_type, "amp": amp}
    if seed is not None:
        params["seed"] = seed
    return C.block(name, bid, **params)


def _throttle(name: str = "thr") -> dict:
    return C.block(name, "blocks_throttle2", type="float", samples_per_second="samp_rate")


def _bpf(name: str = "bpf", window: str = "window.WIN_HAMMING", width: int = WIDTH) -> dict:
    return C.block(name, "band_pass_filter", type="fir_filter_fff", decim="1", gain="1",
                   samp_rate="samp_rate", low_cutoff_freq=F_LO, high_cutoff_freq=F_HI, width=width,
                   win=window, beta="6.76")


def reference() -> dict:
    """noise -> throttle -> band-pass (Hamming, 500 Hz transitions) -> file sink."""
    return C.chain("band_limited_noise",
                   [_noise(), _throttle(), _bpf(), C.file_sink("snk", "float", OUT_FILE)],
                   {"samp_rate": str(FS)})


def variant_hpf_lpf_cascade() -> dict:
    """No throttle, default seed; the band-pass is a 6 kHz high-pass followed by a 9 kHz low-pass."""
    return C.chain("noise_band_cascade",
                   [_noise("src", seed=None),
                    C.high_pass("hpf", "float", F_LO, WIDTH),
                    C.low_pass("lpf", "float", F_HI, WIDTH),
                    C.file_sink("out_file", "float", OUT_FILE)],
                   {"samp_rate": "40e3"})


def _import_window() -> dict:
    """Import block that makes gnuradio.fft.window available to parameter expressions."""
    return C.block("import_0", "import", imports="from gnuradio.fft import window")


def variant_fir_taps_hann() -> dict:
    """Generic Decimating FIR Filter with Hann-window firdes.band_pass taps; cutoffs in variables."""
    g = C.chain("noise_fir_taps",
                [_noise("noise_src", seed=1),
                 C.block("throttle", "blocks_throttle2", type="float", samples_per_second="samp_rate"),
                 C.block("band_filter", "fir_filter_xxx", type="fff", decim="1",
                         taps="firdes.band_pass(1.0, samp_rate, f_low, f_high, 500, window.WIN_HANN)"),
                 C.file_sink("sink", "float", OUT_FILE)],
                {"samp_rate": "40e3", "f_low": "6e3", "f_high": "9e3"})
    g["blocks"].insert(0, _import_window())
    return g


def variant_fastnoise_blackman() -> dict:
    """Fast Noise Source (variate pool), Blackman-window band-pass, throttle after the filter."""
    src = _noise("fast_noise", "analog_fastnoise_source_x")
    src["parameters"]["samples"] = "16384"
    return C.chain("noise_fast_blackman",
                   [src, _bpf("bandpass", "window.WIN_BLACKMAN"),
                    C.block("rate_limit", "blocks_throttle2", type="float", samples_per_second="samp_rate"),
                    C.file_sink("file_out", "float", OUT_FILE)],
                   {"samp_rate": "40000"})


def variant_fft_filter_bp2() -> dict:
    """FFT Filter with firdes.band_pass_2 taps (60 dB stopband, 300 Hz transitions)."""
    return C.chain("noise_fft_bp2",
                   [_noise("ns", seed=7),
                    C.block("thr", "blocks_throttle2", type="float", samples_per_second="samp_rate"),
                    C.block("fftbp", "fft_filter_xxx", type="fff", decim="1",
                            taps="firdes.band_pass_2(1, samp_rate, 6000, 9000, 300, 60)"),
                    C.file_sink("snk", "float", OUT_FILE)],
                   {"samp_rate": "40000"})


def variant_rectangular_window() -> dict:
    """Band Pass Filter with a rectangular window (largest passband ripple of the window choices)."""
    return C.chain("noise_rect",
                   [_noise(seed=None), _throttle(), _bpf(window="window.WIN_RECTANGULAR"),
                    C.file_sink("snk", "float", OUT_FILE)],
                   {"samp_rate": "40000"})


def variant_kaiser_narrow_fullpath() -> dict:
    """Kaiser window (beta 8) with a 100 Hz transition; the file name is given as a full path."""
    bpf = _bpf(window="window.WIN_KAISER", width=100)
    bpf["parameters"]["beta"] = "8"
    return C.chain("noise_kaiser_narrow",
                   [_noise(seed=99), _throttle(), bpf,
                    C.file_sink("snk", "float", "/tmp/grcbench_out/" + OUT_FILE)],
                   {"samp_rate": "40000"})


def variant_blackman_harris_taps() -> dict:
    """FFT Filter with Blackman-Harris firdes.band_pass taps and a 100 Hz transition (1673 taps)."""
    g = C.chain("noise_bh_taps",
                [_noise("noise_0", seed=11),
                 C.block("fft_bp", "fft_filter_xxx", type="fff", decim="1",
                         taps="firdes.band_pass(1, samp_rate, 6000, 9000, 100, window.WIN_BLACKMAN_HARRIS)"),
                 C.block("throttle_0", "blocks_throttle2", type="float", samples_per_second="samp_rate"),
                 C.file_sink("file_sink_0", "float", OUT_FILE)],
                {"samp_rate": "40000"})
    g["blocks"].insert(0, _import_window())
    return g


def variant_fastnoise_rectangular() -> dict:
    """Fast Noise Source (default pool and seed) with a rectangular-window Band Pass Filter (reviewer:
    'a rectangular window ... on either noise source'); lowest RMS of the valid designs."""
    return C.chain("noise_fast_rect",
                   [_noise("fns", "analog_fastnoise_source_x", seed=None),
                    _bpf("bp", "window.WIN_RECTANGULAR"),
                    C.file_sink("fsink", "float", OUT_FILE)],
                   {"samp_rate": "40000"})


def variant_hamming_50hz() -> dict:
    """Band Pass Filter with a 50 Hz transition (1927 taps; reviewer: 'a 100 Hz or 50 Hz transition')."""
    return C.chain("noise_ham_50",
                   [_noise(seed=31), _throttle(), _bpf(width=50), C.file_sink("snk", "float", OUT_FILE)],
                   {"samp_rate": "40000"})


VARIANTS = {
    "v1_hpf_lpf_cascade.grc": variant_hpf_lpf_cascade,
    "v2_fir_filter_firdes_hann.grc": variant_fir_taps_hann,
    "v3_fastnoise_blackman.grc": variant_fastnoise_blackman,
    "v4_fft_filter_band_pass_2.grc": variant_fft_filter_bp2,
    "v5_rectangular_window.grc": variant_rectangular_window,
    "v6_kaiser_100hz_fullpath.grc": variant_kaiser_narrow_fullpath,
    "v7_fft_filter_blackman_harris_100hz.grc": variant_blackman_harris_taps,
    "v8_fastnoise_rectangular.grc": variant_fastnoise_rectangular,
    "v9_hamming_50hz.grc": variant_hamming_50hz,
}


# ---- probes: reviewer counter-examples and answers a single-edit mutant cannot express ----
def _with_head(n: int) -> dict:
    g = reference()
    g["blocks"].append(C.block("hd", "blocks_head", type="float", num_items=str(n)))
    g["connections"] = [c for c in g["connections"] if c[2] != "snk"] + \
        [["bpf", "0", "hd", "0"], ["hd", "0", "snk", "0"]]
    return g


def _ref_with(**bpf_params) -> dict:
    g = reference()
    for b in g["blocks"]:
        if b["name"] == "bpf":
            b["parameters"].update({k: str(v) for k, v in bpf_params.items()})
    return g


def _tone_comb() -> dict:
    """15 equal cosines at 6.1, 6.3, ..., 8.9 kHz summed to RMS 0.75: no noise, no filter."""
    freqs = [6100 + 200 * k for k in range(15)]
    amp = 0.75 / (len(freqs) / 2) ** 0.5
    blocks = [C.sig_source(f"tone{k}", "float", f, f"{amp:.6f}") for k, f in enumerate(freqs)]
    blocks += [C.block("adder", "blocks_add_xx", type="float", num_inputs=str(len(freqs))),
               C.file_sink("snk", "float", OUT_FILE)]
    conns = [[f"tone{k}", "0", "adder", str(k)] for k in range(len(freqs))] + [["adder", "0", "snk", "0"]]
    return C.graph("tone_comb", blocks, conns, {"samp_rate": str(FS)})


def _bartlett() -> dict:
    g = C.chain("noise_bartlett",
                [_noise(seed=5),
                 C.block("bp", "fir_filter_xxx", type="fff", decim="1",
                         taps="firdes.band_pass(1, samp_rate, 6000, 9000, 500, window.WIN_BARTLETT)"),
                 C.file_sink("snk", "float", OUT_FILE)],
                {"samp_rate": str(FS)})
    g["blocks"].insert(0, _import_window())
    return g


def _uniform_matched() -> dict:
    g = reference()
    for b in g["blocks"]:
        if b["name"] == "noise":
            b["parameters"].update({"noise_type": "analog.GR_UNIFORM", "amp": "2*3**0.5"})
    return g


def _default_seed() -> dict:
    g = reference()
    for b in g["blocks"]:
        if b["name"] == "noise":
            del b["parameters"]["seed"]
    return g


def probes() -> dict[str, tuple[dict, bool, str]]:
    """name -> (graph, expected to pass V6, why)."""
    return {
        "head_40000": (_with_head(40000), False,
                       "review F1: a 1 s file (Head of 40000) fails V5 liveness; the instruction now says "
                       "not to limit the number of samples"),
        "head_100000": (_with_head(100000), True,
                        "documented limit: a Head larger than the 65536-item capture is not detected"),
        "width_1000": (_ref_with(width=1000), False,
                       "review F2 (b_width1000_fullpath): a doubled transition width now fails P4"),
        "lo_6250": (_ref_with(low_cutoff_freq=6250), False, "compliance F2 probe: cutoff +250 Hz fails P3"),
        "hi_9250": (_ref_with(high_cutoff_freq=9250), False, "compliance F2 probe: cutoff +250 Hz fails P4"),
        "default_seed": (_default_seed(), True, "review F3/F5: the seed is free (GNU Radio's default 0)"),
        "uniform_rms_matched": (_uniform_matched(), False,
                                "uniform noise scaled to standard deviation 2.0 meets every signal check "
                                "(band-limited noise is Gaussian anyway) and fails the P7 Gaussian check"),
        "tone_comb": (_tone_comb(), False,
                      "15 tones across 6-9 kHz meet every signal check and fail the P7 path check"),
        "bartlett_500": (_bartlett(), False,
                         "documented: a Bartlett-window 500 Hz design misses the stated skirt criterion"),
    }


def run_probes() -> int:
    from grcbench.tier_f.ladder import check
    task = taskio.load(HERE)
    bad = 0
    for name, (g, expect, why) in probes().items():
        cg = compact(g, KEEP)
        res = check(grcfile.dump(cg), task)
        ok = res["first_failed"] is None and res["highest_passed"] == "V6"
        v6 = res["rungs"].get("V6") or {}
        failed = [f"{c['feature']}[{c['spec'].get('lo', '')}-{c['spec'].get('hi', '')}]={c['value']}"
                  for c in v6.get("checks", []) if not c["passed"]] + \
                 [f"graph:{g_['kind']}" for g_ in v6.get("graph_checks", []) if not g_["passed"]]
        if res["first_failed"] and res["first_failed"] != "V6":
            failed = (res["rungs"].get(res["first_failed"]) or {}).get("errors", [])[:1]
        verdict = "as expected" if ok == expect else "UNEXPECTED"
        bad += ok != expect
        print(f"{name:20s} passed={ok!s:5s} first_failed={res['first_failed']} {failed} -> {verdict}\n    {why}")
    return 1 if bad else 0


REVIEW_NOTES = (
    "Known oracle limits (drafted by Claude; for the team). (1) The design numbers are checked through the "
    "stated strip criteria, not tap by tap (300-seed calibration in build.py): cutoff shifts of about "
    "100 Hz are borderline, 250 Hz shifts fail for every window and seed; transition widths up to about "
    "550 Hz (Hann), 700 Hz (Hamming) or 800 Hz (Blackman, Kaiser) pass, a doubled width (1 kHz) fails for "
    "every window. Narrower transitions pass, as 'no more than 500 Hz' allows. A Bartlett window (firdes "
    "taps only; not offered by the Band Pass Filter block) at 500 Hz misses the stated skirt criterion. "
    "(2) Noise level and passband gain are checked jointly through the RMS window 0.69-0.80: amplitude 1 "
    "with gain 2 gives the same signal, and the window has to cover the 0.716-0.773 spread of valid window "
    "choices, so 5 % level errors are not reliably detected (10 % high and 15 % low are). (3) 'Gaussian' "
    "is checked on the noise source's noise_type parameter (GRC's writer exports every parameter, so the "
    "harness's final graph always carries it; the compact gate files keep it explicitly). (4) The path "
    "check proves that filtered noise reaches the File Sink; an answer that also adds tones on that path "
    "could pass. (5) The noise seed is not specified (any seed passes; the reference keeps seed 2718 so "
    "gate runs are bit-identical). (6) A Head block larger than the 65536-item capture is not detected "
    "(probe head_100000). (7) A rectangular-window Band Pass Filter with a 340-375 Hz transition has an "
    "expected RMS of 0.79-0.805 (its passband ripple raises the average gain about 4 %, because GNU Radio "
    "normalises the gain at the band centre), so it fails the stated RMS window on some or most seeds; "
    "widening the window would also admit a 10 % gain error (0.812-0.845). Team decision: keep, or state "
    "that the passband gain must be unity on average. Variant signatures (v2): all filter|noise (test). Former dev twin gbf-dev-02 was "
    "re-themed (envelope|noise); gbf-dev-05 (noise limiter, noise|nonlinear) and gbf-dev-08 (seed item "
    "with band-passed noise among other functions) are related noise items in dev."
)


def compact(data: dict, keep: dict[str, list[str]] | None = None) -> dict:
    """Compact form (only non-default parameters, plus the `keep` ones), verified equivalent in GRC."""
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in data["blocks"]}
    out = grcfile.compact(copy.deepcopy(data), defaults)
    full = {b["name"]: b.get("parameters", {}) for b in data["blocks"]}
    for blk in out["blocks"]:
        for pid in (keep or {}).get(blk["id"], []):
            if pid in full[blk["name"]]:
                blk["parameters"][pid] = full[blk["name"]][pid]
    diffs = grc.compact_equivalent(data, out)
    if diffs:
        raise SystemExit(f"compact form differs from the full form: {diffs}")
    return out


KEEP = {"analog_noise_source_x": ["noise_type"], "analog_fastnoise_source_x": ["noise_type"]}


def main(argv: list[str]) -> int:
    if "--probes" in argv:
        return run_probes()
    ref = compact(reference(), KEEP)
    sig = C.signature(ref)
    split = C.signature_split(sig)
    if split != SPLIT:
        raise SystemExit(f"v2 signature {sig} belongs to split {split}, not {SPLIT}")
    for op in MUTATIONS:
        edit.apply(ref, op)                      # every mutation applies to the reference
    files = {"reference.grc": grcfile.dump(ref)}
    print(f"reference: v2 signature {'|'.join(sig)} -> {split}")
    variant_sigs = {}
    for path, fn in VARIANTS.items():
        v = compact(fn(), KEEP)
        vsig = C.signature(v)
        variant_sigs[path] = f"{'|'.join(vsig)} ({C.signature_split(vsig)})"
        print(f"{path}: v2 signature {variant_sigs[path]}")
        files[f"variants/{path}"] = grcfile.dump(v)

    record = {
        "id": TASK_ID, "tier": "F", "split": SPLIT, "version": 2, "canary": CANARY,
        "license": "CC-BY-4.0 (original GRC-Bench item; licence to be confirmed by the team)",
        "source": {"kind": "composed_chain", "ref": "|".join(sig)},
        "cluster_id": C.cluster_id(sig),
        "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring",
                       "generator_version": "0.2.0"},
        "review": {"status": "pending-team-review", "reviewers": [], "notes": REVIEW_NOTES},
        "family": "build", "level": "L1",
        "instruction": INSTRUCTION,
        "start_grc": None, "reference_grc": "reference.grc",
        "observation": {
            "points": [{"name": "out", "sink": {"block_id": "blocks_file_sink",
                                                "params": {"file": OUT_FILE, "type": "float"}}}],
            "run": {"n_items": N_ITEMS, "min_items": N_ITEMS, "timeout_s": 20},
            "contract": f"the float samples written to the File Sink whose file name is {OUT_FILE}",
        },
        "oracle": ORACLE,
        "mutations": MUTATIONS,
        "spec_properties": SPEC_PROPERTIES,
        "perturbation": {"perturbed": False, "changes": []},
        "disjointness": (
            "Nearest training-side spec: t08_run_verify_probe (Qoherent collection and SFT session: build a "
            "noise source feeding a file plus a level probe, run it and say whether the measured level "
            "matches theory). This task shares the noise source and a level-versus-theory element (P6), but "
            "its core is band-pass filter design (cutoffs, transition width, unity gain) judged on where "
            "the output power lies (P2-P5); it has no probe block and no run-and-report step. It is none of "
            "the other training specs (dial tone, QPSK BER chain, FM receiver repair, AM demod, "
            "spectrogram sweep, RSSI meter) nor GRC-Agent's noise-source scenarios (03_disable_and_enable, "
            "05_full_rewire: toggling or replacing the dial-tone noise source). Its v2 signature "
            "filter|noise hashes to test and is reserved for test (fix_plan R2); every variant (Fast Noise "
            "Source, FIR/FFT filter with firdes taps, HPF+LPF cascade, inert Throttle anywhere) has the "
            "same signature, so the former train- and dev-hashed solution chains are closed."
        ),
        "authoring_notes": {
            "level": "L1: 4 functional blocks (noise source, throttle, band-pass filter, file sink), one rate",
            "signature_v2": "|".join(sig),
            "variant_signatures_v2": variant_sigs,
            "design": "see build.py docstring (calibration over 300 seeds per design)",
            "start_graph": ("build task: start_grc is null in this stage; in GRC-Agent's harness the start is "
                            "the empty graph with samp_rate = 40000 (the instruction's stated rate)"),
        },
    }
    errors = taskio.save(HERE, record, files)
    if errors:
        print("\n".join(errors))
        return 1
    for old in (HERE / "variants").glob("*.grc"):
        if f"variants/{old.name}" not in files:
            old.unlink()
    print(f"written {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
