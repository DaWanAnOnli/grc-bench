"""Authoring script for GRC-Bench Tier-F item gbf-dev-09 (family parameter-tune, level L2, split dev).

Deterministic: re-running it rewrites task.json, start.grc, reference.grc and variants/*.grc
byte-identically (no randomness; the inputs are the constants below, the seed file, whose split and
SHA-256 are asserted, the pinned GNU Radio 3.10.12 block catalog, used to drop default-valued
parameters, and one bounded run of the start graph through the ladder, whose rrc.dat samples become
the "unchanged" reference: the Fast Noise Source with seed 0 is bit-identical from run to run in
GNU Radio 3.10.12, and build.py runs the start graph twice and asserts that). Run inside WSL from
the grc-bench folder:

    PYTHONPATH=src QT_QPA_PLATFORM=offscreen ~/grc-agent/.venv/bin/python items/tier_f/dev/gbf-dev-09/build.py
    ./gb tier-f gates items/tier_f/dev/gbf-dev-09

Seed: the GNU Radio 3.10.12 example filter/filter_taps.grc (a dev-split seed in
splits/flowgraph_seed_split.json). A Fast Noise Source (float, Gaussian, amplitude 1) feeds five
FIR filters (fir_filter_xxx, type fff) whose taps come from filter-taps variables: low-pass
(lp_taps, cutoff = variable cutoff_high), high-pass (hp_taps, cutoff = variable cutoff_low),
band-pass and band-reject (bp_taps / br_taps, edges = variables bp_low / bp_high), all with the
shared transition width `transition`, and a root-raised-cosine filter (rrc_taps, symbol rate
samp_rate / sps). A five-input Qt frequency sink shows the five outputs. The seed tool turns that
sink into five File Sinks; this item names them by filter (lowpass.dat, highpass.dat,
bandpass.dat, bandreject.dat, rrc.dat).

Task idea (parameter-tune). The user wants to split the 0-24 kHz spectrum around a channel at
4-7 kHz with 500 Hz guard bands (3.5-4 and 7-7.5 kHz): low-pass = below the channel, high-pass =
above it, band-pass = the channel, band-reject = everything but the channel, each with unity
passband gain and rolling off only inside the guard bands; the RRC filter stays as it is. The
start graph (perturbed numbers) has the low-pass at 18 kHz, the high-pass at 3 kHz, the band-pass /
band-reject edges at 9-15 kHz and a 1.5 kHz transition width. Five variables have to move:
    cutoff_high  18000 -> 3750   (feeds the LOW-pass: the names describe the displayed band edges)
    cutoff_low    3000 -> 7250   (feeds the HIGH-pass)
    bp_low        9000 -> 3750,  bp_high 15000 -> 7250   (shared by band-pass and band-reject)
    transition    1500 -> 500    (shared by the four filters; 1500 leaks 0.9 % of the band-pass
                                  power outside 3.5-7.5 kHz and 0.39 % of the low-pass power above 4 kHz)
Assigning the cutoffs by variable name (cutoff_low -> low-pass) puts the low-pass at 7.25 kHz and
the high-pass at 3.75 kHz; both stopband checks fail.

Oracle (all numbers are stated in the instruction; 48 kHz; 262144 samples after skipping 4096):
  per retuned filter (f2p): stopband share <= 0.1 %; each 1 kHz passband slice next to a guard
  band carries >= 90 % of the share a flat passband with edges in the guard-band centres would
  give it (rounded down to 0.1 point); RMS within 5 % of the level of unit-power white noise
  through that flat passband, rounded down to two digits (nominal 0.395 / 0.835 / 0.382 / 0.924
  -> stated 0.39 / 0.83 / 0.38 / 0.92: windowed designs lose 1-4 % of the power in the roll-off).
  Noise source and RRC filter unchanged (p2p; round-1 review F2): rrc.dat must hold the same
  samples as the start graph's: correlation at zero lag with the start graph's samples 4096-8191
  >= 0.99999 and RMS within 1 % of the start graph's (numerical tolerance for "the same samples";
  RRC roll-off 0.3 or ntaps 88 score 0.9993 / 0.9999, uniform noise 0.07).
  Parameter values only (p2p; round-1 F1, rule R6): graph checks same_blocks_as_start with
  variables "value_only" (no block added or removed; every variable, including the filter-taps
  blocks, kept under its name and type; values may change) and same_connections_as_start
  (isomorphic: no rewiring).

Robustness (GNU Radio firdes + fastnoise_source_f, 40 noise seeds, measured before writing this):
  * The Fast Noise Source draws from a pool of `samples` Gaussian values; with the example's 8192
    the RMS of every output varies by 1.0 % from seed to seed (pool variance), which left some
    RMS checks only ~3.5 sigma from their bounds. The start graph therefore uses a 65536-sample
    pool (perturbation): RMS spread 0.3 %, and with 262144 analysed samples the tightest check is
    the band-pass RMS of off-centre or Hann designs at >= 5.4 sigma; all others >= 8 sigma. Seed 0
    (the default) is deterministic in GNU Radio 3.10.12, so the reference run is identical each time.
  * Designs that pass every check analytically: Hamming / Hann / Blackman / Kaiser windows with
    transition widths 100-700 Hz (Hann up to 500 Hz with margin, 600 Hz just passes). With
    Hamming and 500 Hz: low-pass cutoffs 3.375-3.9 kHz, high-pass cutoffs 7.025-7.525 kHz,
    band-pass/band-reject edges within +-100 Hz of the guard-band centres (+-150 Hz fails).
    Reference values (Hamming, 500 Hz; mean over 40 noise seeds): lowpass RMS 0.391, 0.002 % above
    4 kHz, 27.1 % in 2.5-3.5 kHz; highpass 0.833, 5.98 % in 7.5-8.5 kHz; bandpass 0.373, 99.996 %
    in 3.5-7.5 kHz, 29.9 % per slice; bandreject 0.921, 0.001 % in 4-7 kHz, 4.89 % per slice;
    rrc 0.499, 75.1 %, 0.011 %. (The seed-0 run of the gates: lowpass 0.389, highpass 0.835,
    bandpass 0.373, bandreject 0.922, rrc 0.497.)
  * Designs that fail: transition 1000 Hz or more (low-pass leaks 0.11 %, band-pass 0.24 %), a
    rectangular window (band-pass leaks 0.3 %), all cutoffs at the channel edges 4 / 7 kHz
    (low-pass leaks 0.45 % above 4 kHz, band-reject 0.17 % inside 4-7 kHz), band-pass/band-reject
    edges at 3.5 / 7.5 kHz (band-pass leaks 0.9 %), a low-pass below 3.375 kHz or a high-pass above
    7.525 kHz (edge slices), passband gain errors of 10 % or more (RMS), any change of the RRC's
    samples per symbol.
  * GNU Radio's "transition width" is nominal: a Hamming design with width 500 Hz rolls off over
    about 1.37 x 500 = 685 Hz, slightly more than the guard band. The instruction therefore turns
    "may roll off only inside the guard bands" into measurable criteria (<= 0.1 % stopband share,
    >= 90 % of the flat-passband share in the slices next to the guard bands), and those are what
    the oracle checks.

Known limits (for the reviewers, also in review.notes): the cutoff ranges above are accepted within
tolerance (the stopband share is relative to each output's own power, so the wide high-pass
tolerates a -6 dB point 25 Hz above the channel edge, the low-pass one 100 Hz below it); renaming a
non-variable block (e.g. a FIR filter) is neither forbidden nor checked.
"""

from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from grcbench.common.canary import CANARY                         # noqa: E402
from grcbench.common.io import read_json, sha256_file             # noqa: E402
from grcbench.common.paths import GNURADIO_EXAMPLES, SPLITS       # noqa: E402
from grcbench.tier_f import edit, grc, grcfile, taskio            # noqa: E402
from grcbench.tier_f.ladder import check                          # noqa: E402
from grcbench.tier_f.seedtools import headless_seed               # noqa: E402

TASK_ID = "gbf-dev-09"
SPLIT = "dev"
SEED = "filter/filter_taps.grc"

# ----------------------------------------------------------------------------- the specification
FS = 48000                         # sample rate (Hz), kept
CH_LO, CH_HI, GUARD = 4000, 7000, 500
LP_EDGE, HP_EDGE = CH_LO - GUARD / 2, CH_HI + GUARD / 2       # 3750, 7250: guard-band centres
POOL = 65536                       # Fast Noise Source pool (perturbed from the example's 8192)

# start (perturbed example numbers) and reference values of the retuned variables
START_VARS = {"samp_rate": "48000", "cutoff_low": "3000", "cutoff_high": "18000", "bp_low": "9000",
              "bp_high": "15000", "transition": "1500", "sps": "4"}
START_ALPHA = "0.25"
REF_VARS = {"cutoff_high": "3750", "cutoff_low": "7250", "bp_low": "3750", "bp_high": "7250",
            "transition": "500"}
ORIG_VARS = {"samp_rate": "32000", "cutoff_low": "2000", "cutoff_high": "14000", "bp_low": "6000",
             "bp_high": "10000", "transition": "1000", "sps": "2"}

# observation: File Sink names and files, by the filter that feeds them
SINKS = {"lp_filter": ("lowpass_sink", "lowpass.dat", "lp"),
         "hp_filter": ("highpass_sink", "highpass.dat", "hp"),
         "bp_filter": ("bandpass_sink", "bandpass.dat", "bp"),
         "br_filter": ("bandreject_sink", "bandreject.dat", "br"),
         "rrc_filter": ("rrc_sink", "rrc.dat", "rrc")}
SKIP, N = 4096, 262144
RUN_ITEMS = SKIP + N
TAPS = ("lp_taps", "hp_taps", "bp_taps", "br_taps")


# ----------------------------------------------------------------------------- flowgraph helpers
def blk(data: dict, name: str) -> dict:
    return next(b for b in data["blocks"] if b["name"] == name)


def setp(data: dict, name: str, **params) -> dict:
    d = copy.deepcopy(data)
    blk(d, name).setdefault("parameters", {}).update({k: str(v) for k, v in params.items()})
    return d


def setvars(data: dict, values: dict[str, str]) -> dict:
    d = copy.deepcopy(data)
    for k, v in values.items():
        b = blk(d, k)
        assert b["id"] == "variable", k
        b["parameters"]["value"] = str(v)
    return d


def tidy(data: dict) -> dict:
    """Compact form: no `states`, no empty housekeeping parameters, no catalog-default parameters.
    Checked: GRC resolves the compact form to exactly the same parameters as the full form."""
    d = copy.deepcopy(data)
    for b in d["blocks"]:
        b.pop("states", None)
        b["parameters"] = {k: v for k, v in b["parameters"].items() if str(v) != ""}
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in d["blocks"]}
    out = grcfile.compact(d, defaults)
    out["options"] = d["options"]
    diffs = grc.compact_equivalent(data, out)
    assert not diffs, f"compact form differs from the full form: {diffs}"
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


# ----------------------------------------------------------------------------- start / reference
def seed_record() -> tuple[str, dict]:
    split_rec = read_json(SPLITS / "flowgraph_seed_split.json")["seeds"][SEED]
    assert split_rec["split"] == SPLIT, f"seed split is {split_rec['split']}, task split is {SPLIT}"
    sha = sha256_file(GNURADIO_EXAMPLES / SEED)
    assert sha == split_rec["sha256"], "example file differs from the inventoried seed"
    return sha, split_rec


def build_start() -> dict:
    seed, sinks = headless_seed(SEED)
    assert len(sinks) == 5 and all(s["type"] == "float" for s in sinks), sinks
    d = copy.deepcopy(seed)
    d["options"]["parameters"] = {"id": "filter_taps", "generate_options": "no_gui", "output_language": "python"}
    # the example's own numbers (asserted, then perturbed)
    for k, v in ORIG_VARS.items():
        assert blk(d, k)["parameters"]["value"] == v, (k, blk(d, k)["parameters"]["value"])
    assert blk(d, "lp_taps")["parameters"]["cutoff_freq"] == "cutoff_high"
    assert blk(d, "hp_taps")["parameters"]["cutoff_freq"] == "cutoff_low"
    for t in ("bp_taps", "br_taps"):
        p = blk(d, t)["parameters"]
        assert (p["low_cutoff_freq"], p["high_cutoff_freq"], p["width"]) == ("bp_low", "bp_high", "transition")
    assert "alpha" not in blk(d, "rrc_taps")["parameters"]                   # example: default 0.35
    assert "samples" not in blk(d, "analog_fastnoise_source_x_0")["parameters"]   # example: default 8192
    # rename the five File Sinks by the filter that feeds them
    feeder = {c[2]: c[0] for c in d["connections"]}
    rename = {s["name"]: SINKS[feeder[s["name"]]][0] for s in sinks}
    files = {v[0]: v[1] for v in SINKS.values()}
    assert sorted(rename.values()) == sorted(files), rename
    for b in d["blocks"]:
        if b["name"] in rename:
            b["name"] = rename[b["name"]]
            b["parameters"]["file"] = files[b["name"]]
    d["connections"] = [[rename.get(c[0], c[0]), c[1], rename.get(c[2], c[2]), c[3]] for c in d["connections"]]
    # perturbed numbers
    d = setvars(d, START_VARS)
    d = setp(d, "rrc_taps", alpha=START_ALPHA)
    d = setp(d, "analog_fastnoise_source_x_0", samples=POOL)
    return tidy(d)


def reference_full(start: dict) -> dict:
    return setvars(start, REF_VARS)


def variants_full(start: dict) -> dict[str, dict]:
    v = {}
    # 1) same variables, Blackman windows and a 300 Hz transition width
    d = setvars(start, {**REF_VARS, "transition": "300"})
    for t in TAPS:
        d = setp(d, t, win="window.WIN_BLACKMAN")
    v["v1_blackman_300hz"] = d
    # 2) the band plan written as expressions inside the existing variables (channel edges and guard
    #    width; round 1 removed the three helper variables this variant used to add, which the
    #    value-only rule forbids), Kaiser windows (beta 8) and a transition of 80 % of the guard band
    d = setvars(start, {"cutoff_high": f"{CH_LO} - {GUARD}/2", "cutoff_low": f"{CH_HI} + {GUARD}/2",
                        "bp_low": f"{CH_LO} - {GUARD}/2", "bp_high": f"{CH_HI} + {GUARD}/2",
                        "transition": f"0.8*{GUARD}"})
    for t in TAPS:
        d = setp(d, t, win="window.WIN_KAISER", beta="8")
    v["v2_band_plan_expressions_kaiser"] = d
    # 3) literal values in each filter-taps block, Hann windows, a different transition width per
    #    filter and slightly off-centre edges; the five scalar variables are kept (now unused) with
    #    their starting values (round 1: this variant used to delete them)
    d = copy.deepcopy(start)
    d = setp(d, "lp_taps", cutoff_freq="3700", width="400", win="window.WIN_HANN")
    d = setp(d, "hp_taps", cutoff_freq="7300", width="450", win="window.WIN_HANN")
    d = setp(d, "bp_taps", low_cutoff_freq="3800", high_cutoff_freq="7200", width="300", win="window.WIN_HANN")
    d = setp(d, "br_taps", low_cutoff_freq="3750", high_cutoff_freq="7250", width="500", win="window.WIN_HANN")
    v["v3_literal_taps_hann"] = d
    # 4) taps written straight into the four FIR filters (default Hamming window); the filter-taps
    #    blocks and the scalar variables untouched (round-1 reviewer's alternative)
    d = copy.deepcopy(start)
    d = setp(d, "lp_filter", taps="firdes.low_pass(1.0, samp_rate, 3750, 500)")
    d = setp(d, "hp_filter", taps="firdes.high_pass(1.0, samp_rate, 7250, 500)")
    d = setp(d, "bp_filter", taps="firdes.band_pass(1.0, samp_rate, 3750, 7250, 500)")
    d = setp(d, "br_filter", taps="firdes.band_reject(1.0, samp_rate, 3750, 7250, 500)")
    v["v4_inline_firdes"] = d
    # 5...) the other designs the round-1 reviewer showed to pass: the reference's band edges with
    #    other windows and transition widths, other edge placements, and the "cutoff = passband edge"
    #    reading of the low-pass and high-pass
    alts = [("alt_hamming_250hz", None, "250", {}), ("alt_hamming_400hz", None, "400", {}),
            ("alt_hann_250hz", "window.WIN_HANN", "250", {}), ("alt_hann_500hz", "window.WIN_HANN", "500", {}),
            ("alt_kaiser_500hz", "window.WIN_KAISER", "500", {}),
            ("alt_blackman_100hz", "window.WIN_BLACKMAN", "100", {}),
            ("alt_blackman_50hz", "window.WIN_BLACKMAN", "50", {}),
            ("alt_blackman_20hz", "window.WIN_BLACKMAN", "20", {}),
            ("alt_edges_3600_7400_200hz", None, "200",
             {"cutoff_high": "3600", "cutoff_low": "7400", "bp_low": "3600", "bp_high": "7400"}),
            ("alt_edges_3900_7100_200hz", None, "200",
             {"cutoff_high": "3900", "cutoff_low": "7100", "bp_low": "3900", "bp_high": "7100"}),
            ("alt_passband_edge_reading", None, "500", {"cutoff_high": "3500", "cutoff_low": "7500"})]
    for name, win, width, extra in alts:
        d = setvars(start, {**REF_VARS, "transition": width, **extra})
        if win:
            for t in TAPS:
                d = setp(d, t, win=win)
        v[name] = d
    return v


PERTURBATION = {"perturbed": True, "changes": [
    {"what": "sample rate (variable samp_rate)", "original": ORIG_VARS["samp_rate"], "new": START_VARS["samp_rate"]},
    {"what": "high-pass cutoff (variable cutoff_low) in the starting flowgraph", "original": ORIG_VARS["cutoff_low"],
     "new": START_VARS["cutoff_low"]},
    {"what": "low-pass cutoff (variable cutoff_high) in the starting flowgraph", "original": ORIG_VARS["cutoff_high"],
     "new": START_VARS["cutoff_high"]},
    {"what": "band-pass / band-reject lower edge (variable bp_low) in the starting flowgraph",
     "original": ORIG_VARS["bp_low"], "new": START_VARS["bp_low"]},
    {"what": "band-pass / band-reject upper edge (variable bp_high) in the starting flowgraph",
     "original": ORIG_VARS["bp_high"], "new": START_VARS["bp_high"]},
    {"what": "shared transition width (variable transition) in the starting flowgraph",
     "original": ORIG_VARS["transition"], "new": START_VARS["transition"]},
    {"what": "RRC samples per symbol (variable sps; symbol rate 16 kbaud -> 12 kbaud)", "original": ORIG_VARS["sps"],
     "new": START_VARS["sps"]},
    {"what": "RRC roll-off (rrc_taps alpha)", "original": "0.35 (default)", "new": START_ALPHA},
    {"what": "Fast Noise Source pool size (samples), for run-to-run stability of the RMS levels",
     "original": "8192 (default)", "new": str(POOL)},
    {"what": "target of the task (reference): cutoff_high / cutoff_low / bp_low / bp_high / transition",
     "original": "14000 / 2000 / 6000 / 10000 / 1000 (example)",
     "new": " / ".join(REF_VARS[k] for k in ("cutoff_high", "cutoff_low", "bp_low", "bp_high", "transition"))},
    {"what": "Qt GUI frequency sink with five inputs",
     "original": "qtgui_freq_sink_x_0 (GUI)",
     "new": ("five File Sinks lowpass_sink (lowpass.dat), highpass_sink (highpass.dat), bandpass_sink "
             "(bandpass.dat), bandreject_sink (bandreject.dat), rrc_sink (rrc.dat); the seed tool's names "
             "qtgui_freq_sink_x_0_<i> replaced")},
    {"what": "generate_options", "original": "qt_gui", "new": "no_gui (seed tool)"},
    {"what": "options block metadata (conversion)",
     "original": "category 'Custom', window_size '1280, 1024', run_options 'prompt', placement, sizing_mode",
     "new": "dropped: the options block keeps only id, generate_options and output_language"},
    {"what": "catalog-default parameters and GUI states (conversion)", "original": "written out in the example",
     "new": "omitted (compact form; GRC fills in the same defaults, checked by grc.compact_equivalent)"},
]}

# ----------------------------------------------------------------------------- task record
INSTRUCTION = (
    "This flowgraph is a filter test bench. It runs at a sample rate of 48 kHz with real (float) samples "
    "throughout. A Fast Noise Source generates Gaussian white noise with unit power (standard deviation 1), "
    "spread evenly from 0 to 24 kHz, and five FIR filters, whose taps come from the filter-taps variables, "
    "shape it. Each filter output is written as float samples to its own File Sink: the low-pass filter to "
    "lowpass.dat, the high-pass filter to highpass.dat, the band-pass filter to bandpass.dat, the band-reject "
    "filter to bandreject.dat and the root-raised-cosine (RRC) filter to rrc.dat.\n\n"
    "I want to use the bench to split the spectrum around a channel that occupies 4 kHz to 7 kHz, with a "
    "500 Hz guard band on each side of it (3.5 to 4 kHz and 7 to 7.5 kHz). Retune the flowgraph so that the "
    "low-pass filter keeps only what lies below the channel, the high-pass filter only what lies above it, the "
    "band-pass filter only the channel, and the band-reject filter everything except the channel. Each of "
    "these four filters must have unity gain in its passband and may roll off only inside the guard bands. "
    "Leave the noise source and the RRC filter exactly as they are: rrc.dat must keep receiving the same "
    "samples as now.\n\n"
    "Measured on each file over 262144 samples after skipping its first 4096 samples, that means:\n"
    "- lowpass.dat: at most 0.1 % of its power above 4 kHz, at least 24.0 % of its power between 2.5 and "
    "3.5 kHz, and an RMS level of 0.39 (within 5 %);\n"
    "- highpass.dat: at most 0.1 % of its power below 7 kHz, at least 5.3 % of its power between 7.5 and "
    "8.5 kHz, and an RMS level of 0.83 (within 5 %);\n"
    "- bandpass.dat: at most 0.1 % of its power outside 3.5 to 7.5 kHz, at least 25.7 % of its power in each "
    "of the slices 4 to 5 kHz and 6 to 7 kHz, and an RMS level of 0.38 (within 5 %);\n"
    "- bandreject.dat: at most 0.1 % of its power between 4 and 7 kHz, at least 4.3 % of its power in each "
    "of the slices 2.5 to 3.5 kHz and 7.5 to 8.5 kHz, and an RMS level of 0.92 (within 5 %).\n"
    "I worked out these numbers for unit-power noise and passbands that are perfectly flat with every band edge "
    "in the middle of its guard band: each 1 kHz slice next to a guard band must carry at least 90 % of the "
    "power it would carry then, and the RMS levels allow a little for the roll-off. Keep the 48 kHz sample rate "
    "and the five float File Sinks. Change parameter values only: do not add, remove or rewire any block, and "
    "keep every variable under its current name. The flowgraph should keep producing samples until it is "
    "stopped; do not limit the number of samples."
)

SPEC = {
    "lp_stop": "low-pass stopband: the channel and everything above it blocked, at most 0.1 % of the lowpass.dat "
               "power above 4 kHz",
    "lp_edge": "low-pass passband edge: full strength up to the 3.5-4 kHz guard band, at least 24.0 % of the "
               "lowpass.dat power in 2.5-3.5 kHz (90 % of the share a flat 0-3.75 kHz passband gives that slice)",
    "lp_level": "low-pass unity passband gain: lowpass.dat RMS 0.39 within 5 % (unit-power white noise through a "
                "unity-gain passband ending in the middle of the guard band: sqrt(3.75/24) = 0.395, less roll-off)",
    "hp_stop": "high-pass stopband: the channel and everything below it blocked, at most 0.1 % of the highpass.dat "
               "power below 7 kHz",
    "hp_edge": "high-pass passband edge: full strength from the 7-7.5 kHz guard band up, at least 5.3 % of the "
               "highpass.dat power in 7.5-8.5 kHz (90 % of the share of a flat 7.25-24 kHz passband)",
    "hp_level": "high-pass unity passband gain: highpass.dat RMS 0.83 within 5 % (sqrt(16.75/24) = 0.835)",
    "bp_stop": "band-pass keeps only the channel: at most 0.1 % of the bandpass.dat power outside 3.5-7.5 kHz",
    "bp_edge": "band-pass passes the whole channel at full strength: at least 25.7 % of the bandpass.dat power in "
               "each of 4-5 kHz and 6-7 kHz (90 % of the share of a flat 3.75-7.25 kHz passband)",
    "bp_level": "band-pass unity passband gain: bandpass.dat RMS 0.38 within 5 % (sqrt(3.5/24) = 0.382)",
    "br_stop": "band-reject removes the channel: at most 0.1 % of the bandreject.dat power in 4-7 kHz",
    "br_edge": "band-reject passes everything outside the guard bands at full strength: at least 4.3 % of the "
               "bandreject.dat power in each of 2.5-3.5 kHz and 7.5-8.5 kHz (90 % of the share of flat passbands "
               "0-3.75 kHz and 7.25-24 kHz)",
    "br_level": "band-reject unity passband gain: bandreject.dat RMS 0.92 within 5 % (sqrt(20.5/24) = 0.924)",
    "rrc_same": "noise source and RRC filter unchanged: rrc.dat receives the same samples as in the starting "
                "flowgraph (samples 4096-8191 correlate with the starting flowgraph's at zero lag with a signed "
                "score >= 0.99999, and the RMS over the measured window is within 1 % of the starting flowgraph's; "
                "numerical tolerance for 'the same samples')",
    "values_only": "parameter values only: the same blocks as the starting flowgraph (block types counted), every "
                   "variable (filter-taps blocks included) kept under its name and type with only its value "
                   "changed, none added or removed (graph check same_blocks_as_start, variables value_only)",
    "no_rewire": "no rewiring: the connections equal the starting flowgraph's up to block renaming (graph check "
                 "same_connections_as_start, isomorphic)",
    "run": "run condition: the flowgraph keeps producing samples until it is stopped (no sample limit); checked "
           "only by V5 liveness (at least 266240 items at every observation point), not by V6",
    "obs": "observation points and rate: float samples written to the File Sinks lowpass.dat, highpass.dat, "
           "bandpass.dat, bandreject.dat and rrc.dat, read as 48 kHz (a graph at another sample rate puts every "
           "band elsewhere, so 'keep the 48 kHz sample rate' is checked by the band checks), 262144 samples after "
           "skipping 4096",
}
UNCHECKED = ("run", "obs")


def bf(point, lo, hi, group, prop, **cmp):
    return {"point": point, "feature": "band_fraction", "fs": FS, "lo": lo, "hi": hi, "skip": SKIP, "n": N,
            "group": group, "property": prop, **cmp}


def rms(point, expect, group, prop, rel_tol=0.05):
    return {"point": point, "feature": "rms", "skip": SKIP, "n": N, "expect": expect, "rel_tol": rel_tol,
            "group": group, "property": prop}


NYQ = FS / 2
REF_LEN = 4096                     # rrc.dat samples 4096-8191 of the start graph: the "unchanged" reference


def build_checks(rrc_ref: list[float], rrc_level: float) -> list[dict]:
    return [
        bf("lp", 4000, NYQ, "f2p", "lp_stop", max=0.001),
        bf("lp", 2500, 3500, "f2p", "lp_edge", min=0.240),
        rms("lp", 0.39, "f2p", "lp_level"),
        bf("hp", 0, 7000, "f2p", "hp_stop", max=0.001),
        bf("hp", 7500, 8500, "f2p", "hp_edge", min=0.053),
        rms("hp", 0.83, "f2p", "hp_level"),
        bf("bp", 3500, 7500, "f2p", "bp_stop", min=0.999),
        bf("bp", 4000, 5000, "f2p", "bp_edge", min=0.257),
        bf("bp", 6000, 7000, "f2p", "bp_edge", min=0.257),
        rms("bp", 0.38, "f2p", "bp_level"),
        bf("br", 4000, 7000, "f2p", "br_stop", max=0.001),
        bf("br", 2500, 3500, "f2p", "br_edge", min=0.043),
        bf("br", 7500, 8500, "f2p", "br_edge", min=0.043),
        rms("br", 0.92, "f2p", "br_level"),
        {"point": "rrc", "feature": "corr_ref", "skip": SKIP, "n": REF_LEN, "ref": rrc_ref, "max_lag": 0,
         "signed": True, "min": 0.99999, "group": "p2p", "property": "rrc_same"},
        rms("rrc", rrc_level, "p2p", "rrc_same", rel_tol=0.01),
    ]


GRAPH_CHECKS = [
    {"kind": "same_blocks_as_start", "variables": "value_only", "group": "p2p", "property": "values_only"},
    {"kind": "same_connections_as_start", "mode": "isomorphic", "group": "p2p", "property": "no_rewire"},
]

MUTATIONS = [
    {"op": "set_param", "block": "cutoff_high", "param": "value", "value": "7250", "breaks": SPEC["lp_stop"],
     "note": "variables assigned by name: the low-pass cutoff (cutoff_high) put at the upper guard band"},
    {"op": "set_param", "block": "cutoff_low", "param": "value", "value": "3750", "breaks": SPEC["hp_stop"],
     "note": "variables assigned by name: the high-pass cutoff (cutoff_low) put at the lower guard band"},
    {"op": "set_param", "block": "transition", "param": "value", "value": "1500", "breaks": SPEC["bp_stop"],
     "note": "transition width left at the starting value"},
    {"op": "set_param", "block": "transition", "param": "value", "value": "1000", "breaks": SPEC["bp_stop"],
     "note": "roll-off twice as wide as the guard band"},
    {"op": "set_param", "block": "bp_low", "param": "value", "value": "4250", "breaks": SPEC["br_stop"],
     "note": "lower band edge inside the channel"},
    {"op": "set_param", "block": "bp_high", "param": "value", "value": "7750", "breaks": SPEC["bp_stop"],
     "note": "upper band edge beyond the guard band"},
    {"op": "set_param", "block": "samp_rate", "param": "value", "value": "32000", "breaks": SPEC["lp_stop"],
     "note": "wrong sample rate: read at 48 kHz every band edge moves up by half"},
    {"op": "bypass", "block": "lp_filter", "breaks": SPEC["lp_stop"], "note": "low-pass filter removed"},
    {"op": "bypass", "block": "bp_filter", "breaks": SPEC["bp_stop"], "note": "band-pass filter removed"},
    {"op": "rewire_input", "dst": "highpass_sink", "dst_port": "0", "new_src": "br_filter", "breaks": SPEC["hp_stop"],
     "note": "wrong tap: highpass.dat records the band-reject output"},
    {"op": "rewire_input", "dst": "bandreject_sink", "dst_port": "0", "new_src": "lp_filter",
     "breaks": SPEC["br_edge"], "note": "wrong tap: bandreject.dat records the low-pass output"},
    {"op": "set_param", "block": "lp_taps", "param": "gain", "value": "0.8", "breaks": SPEC["lp_level"],
     "note": "low-pass passband gain 0.8"},
    {"op": "set_param", "block": "hp_taps", "param": "gain", "value": "0.9", "breaks": SPEC["hp_level"],
     "note": "high-pass passband gain 0.9"},
    {"op": "set_param", "block": "bp_taps", "param": "gain", "value": "1.25", "breaks": SPEC["bp_level"],
     "note": "band-pass passband gain 1.25"},
    {"op": "set_param", "block": "bp_taps", "param": "win", "value": "window.WIN_RECTANGULAR",
     "breaks": SPEC["bp_stop"], "note": "rectangular window: sidelobes leak past the guard bands"},
    {"op": "set_param", "block": "cutoff_low", "param": "value", "value": "8000", "breaks": SPEC["hp_edge"],
     "note": "high-pass passband starts 1 kHz above the guard band"},
    {"op": "set_param", "block": "cutoff_high", "param": "value", "value": "3250", "breaks": SPEC["lp_edge"],
     "note": "low-pass passband ends 500 Hz below the guard band"},
    {"op": "set_param", "block": "br_taps", "param": "high_cutoff_freq", "value": "8250", "breaks": SPEC["br_edge"],
     "note": "band-reject notch 1 kHz too wide at the top"},
    {"op": "set_param", "block": "sps", "param": "value", "value": "3", "breaks": SPEC["rrc_same"],
     "note": "RRC symbol rate changed (16 kbaud)"},
    {"op": "set_param", "block": "rrc_taps", "param": "alpha", "value": "0.3", "breaks": SPEC["rrc_same"],
     "note": "RRC roll-off 0.25 -> 0.3 (near miss: passed the round-1 RRC band checks)"},
    {"op": "set_param", "block": "rrc_taps", "param": "ntaps", "value": "22*sps", "breaks": SPEC["rrc_same"],
     "note": "RRC filter twice as long (near miss: passed the round-1 RRC checks)"},
    {"op": "set_param", "block": "analog_fastnoise_source_x_0", "param": "amp", "value": "0.5",
     "breaks": SPEC["rrc_same"], "note": "noise source no longer unit power (every RMS level halves)"},
    {"op": "set_param", "block": "analog_fastnoise_source_x_0", "param": "noise_type", "value": "analog.GR_UNIFORM",
     "breaks": SPEC["rrc_same"],
     "note": "uniform instead of Gaussian noise (round-1 probe, there with amp sqrt(3) so that every level was "
             "kept; the zero-lag correlation with the start graph's rrc.dat rejects it at any amplitude)"},
    {"op": "set_id", "block": "lp_filter", "new_id": "fft_filter_xxx", "breaks": SPEC["values_only"],
     "note": "the low-pass FIR filter swapped for an FFT filter with the same taps (round-1 probe): same output, "
             "only the block check sees it"},
    {"op": "bypass", "block": "len_taps", "breaks": SPEC["values_only"],
     "note": "an (unused) variable deleted: same output, only the value-only rule sees it"},
    {"op": "rewire_input", "dst": "lp_filter", "dst_port": "0", "new_src": "br_filter", "breaks": SPEC["no_rewire"],
     "note": "the low-pass filter fed from the band-reject output instead of the noise source: almost the same "
             "output, only the connection check sees it"},
]

DISJOINTNESS = (
    "Retunes a five-filter noise test bench (dev-split seed filter/filter_taps.grc) to split the spectrum around a "
    "4-7 kHz channel with guard bands, checked by band-power shares and RMS levels of the filtered noise. Nearest "
    "training-side specs (reports/reference/training_task_specs.json): t10_param_tuning retunes a receiver's centre "
    "frequency and bandwidth and renames variables (playground/new_dmeo.grc); t08_run_verify_probe and "
    "t19_rssi_meter_build probe noise or signal levels without filtering; t17_wbfm_tune_repair and "
    "t20_resample_ratio_repair repair an FM receiver's or a resampler's rates and filter; the GRC-Agent dial-tone "
    "scenarios (02_update_sample_rate, 04_add_and_remove_variable) edit tone chains; the train seed "
    "filter/filter_taps_loader.grc (taps loaded from a file into a single FIR) is a different file and task."
)

OBS = {
    "points": [{"name": key, "sink": {"block_id": "blocks_file_sink", "params": {"file": f, "type": "float"}}}
               for _, f, key in SINKS.values()],
    "run": {"n_items": RUN_ITEMS, "min_items": RUN_ITEMS, "timeout_s": 10.0},
    "contract": ("the float samples (48 kHz) written to the File Sinks whose file names are lowpass.dat, "
                 "highpass.dat, bandpass.dat, bandreject.dat and rrc.dat"),
}

AUTHORING_NOTES = {
    "level": ("L2: 11 functional blocks (Fast Noise Source, five FIR filters, five File Sinks; the filter-taps "
              "blocks are variables), filtering, one sample rate"),
    "design": "see build.py docstring (firdes analysis and a 40-seed Monte Carlo of every check)",
    "scope": ("'Change parameter values only: do not add, remove or rewire any block, and keep every variable under "
              "its current name' is checked by graph checks (p2p): same_blocks_as_start with variables value_only "
              "and same_connections_as_start (isomorphic). Round 1: this sentence used to be unchecked."),
    "rrc_unchanged": ("'Leave the noise source and the RRC filter exactly as they are' is checked on rrc.dat against "
                      "the start graph's own samples (the noise source with seed 0 is deterministic); the former "
                      "RRC level/band checks let roll-off 0.15-0.3, ntaps 88 and uniform noise pass (round-1 F2)."),
    "start_grc": ("the seed exported by `./gb tier-f seed filter/filter_taps.grc`, File Sinks renamed by filter, "
                  "numbers perturbed (see perturbation), compact form"),
    "related_items": ("gbf-test-03 (test) builds band-limited noise with a single 6-9 kHz band-pass and gbf-dev-08 "
                      "band-passes noise at 6-18 kHz; this item retunes a whole filter bank around a 4-7 kHz channel, "
                      "and its channel was moved away from 6-9 kHz on purpose"),
    "transition_width": ("GNU Radio's transition width is nominal (Hamming, 500 Hz: about 685 Hz of roll-off); "
                         "'may roll off only inside the guard bands' is made measurable by the stated stopband shares "
                         "and edge-slice shares, which are what the oracle checks"),
    "accepted_ranges": ("Hamming, 500 Hz: low-pass cutoff 3.375-3.9 kHz, high-pass cutoff 7.025-7.525 kHz, "
                        "band-pass/band-reject edges within +-100 Hz of 3.75 / 7.25 kHz; transition widths up to "
                        "about 700 Hz (Hann about 600 Hz)"),
    "variants": ("v1 Blackman windows with a 300 Hz transition; v2 the band plan as expressions inside the existing "
                 "variables, Kaiser windows (beta 8) and a 400 Hz transition; v3 literal values in each filter-taps "
                 "block (scalar variables kept, unused), Hann windows, per-filter transitions and slightly off-centre "
                 "edges; v4 firdes taps written straight into the four FIR filters; alt_* the other designs the "
                 "round-1 reviewer showed to pass (Hamming 250 / 400 Hz, Hann 250 / 500 Hz, Kaiser 6.76 500 Hz, "
                 "Blackman 100 / 50 / 20 Hz, edges 3.6 / 7.4 and 3.9 / 7.1 kHz with 200 Hz, the 'cutoff = passband "
                 "edge' reading of the low-pass and high-pass). All obey the value-only rule."),
}

REVIEW_NOTES = (
    "Known oracle limits: (1) the stated cutoff ranges are accepted within tolerance (the stopband share is relative "
    "to each output's own power, so the wide high-pass tolerates a -6 dB point 25 Hz above the channel edge and the "
    "low-pass one 100 Hz below it). (2) Renaming a non-variable block (e.g. a FIR filter) is neither forbidden by the "
    "instruction nor checked; variables must keep their names (value_only). (3) 'Keep producing samples until "
    "stopped' is checked only by V5 liveness (266240 items per observation point). (4) 'The same samples as now' on "
    "rrc.dat is checked with a numerical tolerance (signed zero-lag correlation >= 0.99999 against the start graph's "
    "samples 4096-8191, RMS within 1 %), which relies on the Fast Noise Source with seed 0 being deterministic in "
    "GNU Radio 3.10.12 (build.py asserts two runs are identical). Licence: derived from the GNU Radio 3.10.12 "
    "example filter/filter_taps.grc (GNU Radio project, GPL-3.0-or-later); the item keeps that attribution "
    "(licence decision is the team's)."
)


def start_rrc_reference(start_text: str) -> tuple[list[float], float]:
    """rrc.dat samples SKIP..SKIP+REF_LEN of the start graph and its RMS over the measured window, from two
    bounded ladder runs (V2-V5, the harness's own taps) that must be identical."""
    arrays = []
    for _ in range(2):
        res = check(start_text, {"observation": OBS}, max_rung=5, keep=True)
        run_dir = Path(res["run_dir"])
        try:
            assert res["first_failed"] is None and res["highest_passed"] == "V5", res["rungs"]
            arrays.append(np.load(run_dir / "obs" / "rrc.npy"))
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)
    assert np.array_equal(arrays[0], arrays[1]), "rrc.dat differs between two runs of the start graph"
    x = arrays[0].astype(np.float64)
    ref = [round(float(v), 7) for v in x[SKIP:SKIP + REF_LEN]]
    level = round(float(np.sqrt(np.mean(x[SKIP:SKIP + N] ** 2))), 4)
    return ref, level


def main() -> int:
    sha, _ = seed_record()
    start = build_start()
    ref = tidy(reference_full(start))
    variants = {k: tidy(v) for k, v in variants_full(start).items()}
    for op in MUTATIONS:
        edit.apply(ref, op)                      # every mutation applies to the reference
    rrc_ref, rrc_level = start_rrc_reference(dump(start))
    checks = build_checks(rrc_ref, rrc_level)
    used = {c["property"] for c in checks + GRAPH_CHECKS}
    assert used == set(SPEC) - set(UNCHECKED), used ^ (set(SPEC) - set(UNCHECKED))
    checks = [{**c, "property": SPEC[c["property"]]} for c in checks]
    graph_checks = [{**g, "property": SPEC[g["property"]]} for g in GRAPH_CHECKS]
    print(f"start rrc.dat RMS over the measured window: {rrc_level}")

    record = {
        "id": TASK_ID, "tier": "F", "split": SPLIT, "version": 2, "canary": CANARY,
        "license": f"GPL-3.0-or-later (derived from GNU Radio 3.10.12 example {SEED})",
        "source": {"kind": "gnuradio_example", "ref": SEED, "sha256": sha},
        "cluster_id": SEED,
        "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring",
                       "generator_version": "0.2.0"},
        "review": {"status": "pending-team-review", "reviewers": [], "notes": REVIEW_NOTES},
        "family": "parameter-tune", "level": "L2",
        "instruction": INSTRUCTION,
        "start_grc": "start.grc", "reference_grc": "reference.grc",
        "observation": OBS,
        "oracle": {"checks": checks, "graph_checks": graph_checks},
        "mutations": MUTATIONS,
        "spec_properties": list(SPEC.values()),
        "perturbation": PERTURBATION,
        "disjointness": DISJOINTNESS,
        "authoring_notes": AUTHORING_NOTES,
    }
    if (HERE / "variants").exists():
        shutil.rmtree(HERE / "variants")
    files = {"start.grc": dump(start), "reference.grc": dump(ref)}
    files.update({f"variants/{k}.grc": dump(v) for k, v in variants.items()})
    errors = taskio.save(HERE, record, files)
    if errors:
        print("NOT WRITTEN:", *errors, sep="\n  ")
        return 1
    print(f"wrote {HERE.relative_to(REPO)} ({len(files)} flowgraphs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
