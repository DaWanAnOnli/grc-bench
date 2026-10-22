"""Authoring script for GRC-Bench Tier-F item gbf-test-19 (parameter-tune, L2, seed-derived).

Deterministic: re-running it rewrites task.json, start.grc, reference.grc and variants/*.grc
byte-identically (no randomness; the inputs are the constants below, the seed file, whose split and
SHA-256 are asserted, the pinned GNU Radio 3.10.12 block catalog, used to drop default-valued
parameters, and GNU Radio's firdes.gaussian, used for the reference frequency trace). Run inside WSL
from the repository root:

    PYTHONPATH=src QT_QPA_PLATFORM=offscreen ~/grc-agent/.venv/bin/python items/tier_f/test/gbf-test-19/build.py
    ./gb tier-f gates items/tier_f/test/gbf-test-19

Source: the GNU Radio 3.10.12 example channels/demo_gmsk.grc (a test-split seed). The example sends
random bytes through a GMSK modulator (10 samples per symbol, BT 0.3, 32 kHz), a Throttle and the
HW Impairments block (channels_impairments, every impairment at zero except -100 dB phase noise),
then shows the impaired signal on a Qt constellation sink and its Quadrature Demod output (gain 1.0)
on a Qt time sink. This item keeps that chain, turns the two Qt sinks into File Sinks (gmsk_iq.dat,
complex; freq_trace.dat, float), turns the Qt range widgets into plain variables, drops the unused
variable const_size, and perturbs the numbers (PERTURBATION below).

Task idea (parameter-tune, value-only). The start graph sends 6 kbit/s GMSK (8 samples per symbol
at 48 kHz) with a Gaussian BT of 0.5, and its demodulator output is in radians per sample (gain 1.0).
The user states three targets in physical units and the model decides which parameter values
realise them; blocks, connections and variables stay as they are:
  bit rate 4 kbit/s               -> samples_per_symbol = 48000 / 4000 = 12
  >= 99.9 % of the power in +-2.5 kHz -> a smaller BT (0.5 leaves 0.14-0.15 % outside; BT 0.3 leaves
                                     0.03 %; BT <= 0.4 passes, measured below)
  trace calibrated in kHz         -> Quadrature Demod gain = samp_rate / (2*pi*1000) = 7.639
                                     (= GNU Radio's own gmsk_demod idiom 2*sps/pi for sps 12)
Fixing only two of the three still fails. The trace reads +-1.0 because GMSK (h = 0.5) deviates by
+-bit_rate/4 = +-1 kHz.

Trap avoided on purpose: channels_impairments' "Frequency Offset" does NOT move the signal (the
block multiplies by conj(LO) before its distortions and by LO after them; only the DC-offset and IQ
images move). A carrier-offset target would therefore need an extra block, so this item has none.

Measured with the seed data (numpy seed 0, as in the harness), 65536 samples after skipping 4096,
48 kHz, 4 kbit/s; the spread over numpy seeds 0-7 is given in brackets:
  BT    power within +-2.5 kHz        peak |trace| (kHz)   mean trace (kHz)
  0.30  0.99967 [0.99967-0.99972]     1.0000               -0.024 [-0.024..+0.027]
  0.40  0.99919 [0.99917-0.99927]     1.0000
  0.45  0.99886 [0.99884-0.99897]     1.0000
  0.50  0.99852 [0.99848-0.99864]     1.0000
  1.00  0.99570                       1.0000
  6 kbit/s, BT 0.5 (the start's modulation): 0.95524, peak 1.5 kHz.
The peak |trace| is bit_rate/4 for every BT (runs of >= 4 equal bits reach the full deviation).

Bit-rate and sign check (round-1 review, item-quality F1/F2). The peak |trace| alone does not pin
the bit rate: any samples_per_symbol passes it when the demodulator gain is re-fitted (gain = 2*sps/pi
normalises the peak to 1.0 for every sps). The trace is therefore also compared with the trace the
stated data must produce: the harness seeds numpy with 0 before the flowgraph is built, so the
Random Source (numpy.random.randint(0, 256, 100000)) emits a fixed byte sequence. reference_trace()
below models GMSK Mod + Quadrature Demod in numpy (MSB-first unpacking, NRZ +-1, the Gaussian taps
from firdes.gaussian convolved with a samples_per_symbol-long rectangle, interpolation, FM with
sensitivity pi/(2 sps), demodulator gain samp_rate/(2 pi 1000)); against the real reference run it
agrees within 1e-5 at lag 0 (the -100 dB phase noise is the only difference). The check is a SIGNED
corr_ref (Re<ref, x> / |ref||x|, best lag within +-128 samples), so a negated trace fails as well.
Scores (numpy model, same window): BT 0.5 0.983, 0.4 0.993, 0.3 1.000, 0.2 0.980, 0.1 0.871, 0.05
0.800, 0.03 0.781 (a Gaussian truncated to 4 symbols tends to a trapezoid, so no BT goes much lower);
wrong bit rates with the peak re-fitted (8, 10, 11, 13, 14, 16, 24 samples per symbol, BT 0.2/0.3)
0.06-0.21; negated trace 0.04; other random bytes (Random Source min 1) 0.43. MIN_CORR = 0.6 lies
between the lowest score of a correct bit rate and the highest score of a wrong one.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # .../grc-bench
sys.path.insert(0, str(ROOT / "src"))

from gnuradio.filter import firdes                                         # noqa: E402

from grcbench.common.canary import CANARY                                 # noqa: E402
from grcbench.common.io import sha256_file                                # noqa: E402
from grcbench.common.paths import GNURADIO_EXAMPLES, SPLITS                # noqa: E402
from grcbench.tier_f import compose as C, grc, grcfile, taskio            # noqa: E402

TASK_ID = "gbf-test-19"
SPLIT = "test"
SEED = "channels/demo_gmsk.grc"

FS = 48000              # sample rate (Hz)
START_SPS = 8           # start: 6 kbit/s
START_BT = "0.5"
REF_SPS = 12            # target: 4 kbit/s
REF_BT = "0.3"
BIT_RATE = 4000
DEV_KHZ = BIT_RATE / 4 / 1000          # GMSK (h = 0.5): +-1.0 kHz
KHZ_GAIN = "samp_rate/(2*math.pi*1000)"
IQ_FILE = "gmsk_iq.dat"
TRACE_FILE = "freq_trace.dat"
BAND_HZ = 2500
MIN_IN_BAND = 0.999
SKIP, N = 4096, 65536
RUN_ITEMS = SKIP + N + 2048    # 71680 items per point, about 1.5 s at the throttled 48 kHz
CORR_LEN, CORR_LAG = 4096, 256  # correlated samples; lag search 0..256 (= -128..+128 around the reference)
MIN_CORR = 0.6
RANDOM_BYTES = 100000           # the Random Source's num_samps (numpy.random.randint(0, 256, 100000))

# ---------------------------------------------------------------- the seed and its original numbers
seed_path = GNURADIO_EXAMPLES / SEED
seed_sha = sha256_file(seed_path)
seed_split = json.loads((SPLITS / "flowgraph_seed_split.json").read_text())["seeds"][SEED]
assert seed_split["split"] == SPLIT, seed_split
assert seed_split["sha256"] == seed_sha, (seed_split["sha256"], seed_sha)
seed = yaml.safe_load(seed_path.read_text(encoding="utf-8"))
sb = {b["name"]: b for b in seed["blocks"]}
assert {b["id"] for b in seed["blocks"]} >= {"analog_random_source_x", "digital_gmsk_mod", "blocks_throttle2",
                                             "channels_impairments", "analog_quadrature_demod_cf",
                                             "qtgui_const_sink_x", "qtgui_time_sink_x"}
ORIG = {"samp_rate": sb["samp_rate"]["parameters"]["value"],
        "sps": sb["digital_gmsk_mod_0"]["parameters"]["samples_per_symbol"],
        "bt": sb["digital_gmsk_mod_0"]["parameters"]["bt"],
        "gain": sb["analog_quadrature_demod_cf_0"]["parameters"]["gain"]}
assert ORIG == {"samp_rate": "32000", "sps": "10", "bt": "0.3", "gain": "1.0"}, ORIG

PERTURBATION = {"perturbed": True, "changes": [
    {"what": "sample rate (variable samp_rate)", "original": ORIG["samp_rate"], "new": str(FS)},
    {"what": "GMSK samples per symbol in the starting flowgraph (bit rate 3.2 kbit/s -> 6 kbit/s)",
     "original": ORIG["sps"], "new": str(START_SPS)},
    {"what": "GMSK Gaussian filter BT in the starting flowgraph", "original": ORIG["bt"], "new": START_BT},
    {"what": "target of the task: samples per symbol for 4 kbit/s (reference)", "original": ORIG["sps"],
     "new": str(REF_SPS)},
    {"what": "target of the task: Quadrature Demod gain calibrated in kHz (reference)", "original": ORIG["gain"],
     "new": KHZ_GAIN},
    # headless conversion (not a perturbation of numbers, recorded so the derivation is complete)
    {"what": "conversion: options block", "original": ("generate_options qt_gui, author 'mettus', window_size, "
                                                        "sizing_mode, placement, run_options prompt and other GUI options"),
     "new": "id demo_gmsk, generate_options no_gui, output_language python (everything else stripped)"},
    {"what": "conversion: Qt constellation sink qtgui_const_sink_x_0 (input from channels_impairments_0)",
     "original": "qtgui_const_sink_x", "new": f"File Sink blocks_file_sink_0, file {IQ_FILE}, type complex"},
    {"what": "conversion: Qt time sink qtgui_time_sink_x_0_0 (input from analog_quadrature_demod_cf_0)",
     "original": "qtgui_time_sink_x", "new": f"File Sink blocks_file_sink_1, file {TRACE_FILE}, type float"},
    {"what": "conversion: Qt range widgets beta, freq_offset, i_ofs, magbal, phase_noise_mag, phasebal, q_ofs",
     "original": "variable_qtgui_range (counter_slider)", "new": "variable with the widget's value (0, or -100 for "
                                                                 "phase_noise_mag)"},
    {"what": "conversion: unused variable const_size", "original": "4", "new": "removed"},
]}
DERIVATION = ("Qt constellation sink -> File Sink gmsk_iq.dat (complex); Qt time sink -> File Sink "
              "freq_trace.dat (float); the seven Qt range widgets (beta, freq_offset, i_ofs, magbal, "
              "phase_noise_mag, phasebal, q_ofs) became plain variables with their widget values; the unused "
              "variable const_size dropped.")

# ---------------------------------------------------------------- flowgraphs
SRC, MOD, THR, CHAN, DEMOD = ("analog_random_source_x_0", "digital_gmsk_mod_0", "blocks_throttle2_0",
                              "channels_impairments_0", "analog_quadrature_demod_cf_0")
IQ_SINK, TRACE_SINK = "blocks_file_sink_0", "blocks_file_sink_1"
IMPAIRMENT_VARS = {"beta": "0", "freq_offset": "0", "i_ofs": "0", "magbal": "0", "phase_noise_mag": "-100",
                   "phasebal": "0", "q_ofs": "0"}


def gmsk_graph(sps, bt, gain, *, gid="demo_gmsk", samp_rate=FS) -> dict:
    """The seed chain: random bytes -> GMSK -> Throttle -> impairments -> Quadrature Demod -> File Sinks.
    Every graph of this item has exactly these blocks, connections and variables (value-only task)."""
    variables = {"samp_rate": samp_rate, **IMPAIRMENT_VARS}
    blocks = [
        C.block("import_0", "import", imports="import math"),
        C.block(SRC, "analog_random_source_x", type="byte", min="0", max="256", num_samps=str(RANDOM_BYTES),
                repeat="True"),
        C.block(MOD, "digital_gmsk_mod", samples_per_symbol=sps, bt=bt, verbose="False", log="False",
                do_unpack="True"),
        C.block(THR, "blocks_throttle2", type="complex", samples_per_second="samp_rate", vlen="1",
                ignoretag="True", limit="auto", maximum="0.1"),
        C.block(CHAN, "channels_impairments", **{k: k for k in IMPAIRMENT_VARS}, gamma="0"),
        C.block(DEMOD, "analog_quadrature_demod_cf", gain=gain),
        C.file_sink(IQ_SINK, "complex", IQ_FILE),
        C.file_sink(TRACE_SINK, "float", TRACE_FILE),
    ]
    conns = [[SRC, "0", MOD, "0"], [MOD, "0", THR, "0"], [THR, "0", CHAN, "0"],
             [CHAN, "0", IQ_SINK, "0"], [CHAN, "0", DEMOD, "0"], [DEMOD, "0", TRACE_SINK, "0"]]
    return C.graph(gid, blocks, conns, variables)


def compact(data: dict, keep: dict[str, list[str]] | None = None) -> dict:
    """Compact form; `keep` {block: [param, ...]} re-adds default-valued parameters that should stay
    visible to the model (start graph only: the impairment variables' values, e.g. magbal = 0)."""
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in data["blocks"]}
    out = grcfile.compact(data, defaults)
    full = {b["name"]: b["parameters"] for b in data["blocks"]}
    for blk in out["blocks"]:
        for pid in (keep or {}).get(blk["name"], []):
            blk["parameters"][pid] = full[blk["name"]][pid]
    diffs = grc.compact_equivalent(data, out)
    assert not diffs, diffs
    return out


start = gmsk_graph(START_SPS, START_BT, "1.0")
reference = gmsk_graph(REF_SPS, REF_BT, KHZ_GAIN)

# Every variant changes parameter values only (same blocks, connections and variables as the start).
# v1: BT 0.25; samples per symbol and gain written in terms of samp_rate.
v1 = gmsk_graph("int(samp_rate/4000)", "0.25", "samp_rate/(2*math.pi*1e3)", gid="gmsk_tuned_v1")
# v2: GNU Radio's own gmsk_demod idiom for the gain, 2*sps/pi (= 7.639 for sps 12), BT 0.3.
v2 = gmsk_graph(REF_SPS, "0.3", "2*12/math.pi", gid="gmsk_tuned_v2")
# v3: BT 0.35, the gain as a literal number (48000 / (2 pi 1000)).
v3 = gmsk_graph(REF_SPS, "0.35", repr(FS / (2 * math.pi * 1000)), gid="gmsk_tuned_v3")
# v4: BT 0.4, the largest BT that meets the bandwidth limit (0.99919); sps as integer division.
v4 = gmsk_graph("samp_rate//4000", "0.4", "48/(2*math.pi)", gid="gmsk_tuned_v4")
# v5: BT 0.15, a much narrower Gaussian (more inter-symbol interference, still every requirement met).
v5 = gmsk_graph(REF_SPS, "0.15", KHZ_GAIN, gid="gmsk_tuned_v5")

ref_c = compact(reference)
ids = grc.block_ids()
assert all(b["id"] in ids for b in ref_c["blocks"])
functional = [b for b in ref_c["blocks"] if b["id"] not in ("variable", "import")]
assert len(functional) == 7, functional      # L2: 6-15 functional blocks, modulation + demodulation
start_c = compact(start, keep={k: ["value"] for k in IMPAIRMENT_VARS})
variants = {"v1_bt025_rate_from_samp_rate.grc": compact(v1),
            "v2_bt03_gmsk_demod_gain_idiom.grc": compact(v2),
            "v3_bt035_literal_gain.grc": compact(v3),
            "v4_bt04_band_edge.grc": compact(v4),
            "v5_bt015_narrow_gaussian.grc": compact(v5)}


# ---------------------------------------------------------------- reference trace (bit-rate/sign check)
def reference_trace(n: int, sps: int = REF_SPS, bt: float = float(REF_BT)) -> np.ndarray:
    """freq_trace.dat of the reference without phase noise: GMSK Mod + Quadrature Demod in numpy."""
    rng = np.random.RandomState(0)                     # the harness calls numpy.random.seed(0)
    data = rng.randint(0, 256, RANDOM_BYTES)           # analog_random_source_x's make template
    bits = np.unpackbits(data[: n // (8 * sps) + 2].astype(np.uint8))      # MSB first
    nrz = 2.0 * bits - 1.0
    taps = np.convolve(np.array(firdes.gaussian(1, sps, bt, 4 * sps)), np.ones(sps))
    up = np.zeros(len(nrz) * sps)
    up[::sps] = nrz
    freq = np.convolve(up, taps)[: len(up)]            # interp_fir_filter_fff(sps, taps)
    sensitivity = (math.pi / 2) / sps                  # frequency_modulator_fc
    gain = FS / (2 * math.pi * 1000)                   # Quadrature Demod gain (kHz)
    return (freq * sensitivity * gain)[:n]


_trace = reference_trace(SKIP + CORR_LAG + CORR_LEN + CORR_LAG)
REF_TRACE = [round(float(v), 4) for v in _trace[SKIP + CORR_LAG // 2: SKIP + CORR_LAG // 2 + CORR_LEN + CORR_LAG]]
assert len(REF_TRACE) == CORR_LEN + CORR_LAG and abs(max(map(abs, REF_TRACE)) - 1.0) < 1e-3

# ---------------------------------------------------------------- instruction
instruction = (
    "This flowgraph generates a GMSK test signal from random bytes and passes it through the HW "
    "Impairments block, which is set so that it changes nothing (phase noise at -100 dB, all other "
    "impairments 0). Everything runs at a sample rate of 48 kHz. "
    "The complex output of the HW Impairments block is written to the File Sink with file name gmsk_iq.dat "
    "(item type complex), and the output of the Quadrature Demod block, a trace of the signal's "
    "instantaneous frequency, is written to the File Sink with file name freq_trace.dat (item type float). "
    "At the moment it sends 6 kbit/s, its spectrum is too wide for my new receiver, and the trace is not "
    "in a convenient unit.\n\n"
    "Tune the flowgraph so that, measured after the first 4096 samples of each file:\n"
    "1. The GMSK bit rate is 4 kbit/s (4000 bits per second), so the frequency deviation is ±1 kHz. The "
    "bits still come from the same random byte source.\n"
    "2. At least 99.9 % of the power of the complex signal in gmsk_iq.dat lies between -2.5 kHz and "
    "+2.5 kHz, the passband of the receiver's channel filter.\n"
    "3. freq_trace.dat shows the instantaneous frequency in kHz, positive above 0 Hz and negative below: "
    "it swings between about -1.0 and +1.0, and its largest absolute value is 1.0 within 3 %.\n\n"
    "Change parameter values only: do not add, remove or rewire blocks, and do not add or remove "
    "variables. Keep everything else as it is: the 48 kHz sample rate (samp_rate = 48000), the random byte "
    "source, GMSK modulation, the HW Impairments block changing nothing (the complex signal keeps its "
    "constant amplitude of 1.0, RMS within 5 %, and stays centred at 0 Hz, so the trace averages to 0 "
    "within ±0.1 kHz), and both File Sinks with their file names and item types. The flowgraph should keep "
    "producing samples until it is stopped; do not limit the number of samples."
)

# ---------------------------------------------------------------- observation and oracle
observation = {
    "points": [{"name": "iq", "sink": {"block_id": "blocks_file_sink", "params": {"file": IQ_FILE, "type": "complex"}}},
               {"name": "trace", "sink": {"block_id": "blocks_file_sink",
                                          "params": {"file": TRACE_FILE, "type": "float"}}}],
    "run": {"n_items": RUN_ITEMS, "min_items": RUN_ITEMS, "timeout_s": 10.0},
    "contract": ("the complex samples written to the File Sink whose file name is gmsk_iq.dat and the float "
                 "samples written to the File Sink whose file name is freq_trace.dat, both at 48 kHz; "
                 f"{RUN_ITEMS} items are read from each"),
}
W = {"skip": SKIP, "n": N}
checks = [
    {"point": "trace", "feature": "max_abs", "expect": DEV_KHZ, "rel_tol": 0.03, **W, "group": "f2p",
     "property": "trace_scale",
     "note": "peak |trace| = bit_rate/4 in kHz = 1.0; 6 kbit/s -> 1.5, 4.8 kbit/s -> 1.2, rad/sample -> 0.13"},
    {"point": "trace", "feature": "corr_ref", "ref": REF_TRACE, "signed": True, "max_lag": CORR_LAG,
     "skip": SKIP, "n": CORR_LEN + 2 * CORR_LAG, "min": MIN_CORR, "group": "f2p",
     "property": "bit_rate, trace_sign",
     "note": ("signed correlation with the trace the seeded random bytes give at 4 kbit/s (reference starts "
              "128 samples after the skip; lags 0..256 = -128..+128 samples). Correct bit rate, any BT: >= 0.78; "
              "wrong bit rate with a re-fitted gain: <= 0.21; negated trace: 0.04")},
    {"point": "iq", "feature": "band_fraction", "fs": FS, "lo": -BAND_HZ, "hi": BAND_HZ, "min": MIN_IN_BAND, **W,
     "group": "f2p", "property": "occupied_bandwidth",
     "note": "BT 0.3: 0.99967; BT 0.4: 0.99919; BT 0.5: 0.99852 (fails); the start (6 kbit/s, BT 0.5): 0.955"},
    {"point": "iq", "feature": "rms", "expect": 1.0, "rel_tol": 0.05, **W, "group": "p2p",
     "property": "channel_unchanged", "note": "constant envelope 1.0 (no IQ imbalance, DC offset or nonlinearity)"},
    {"point": "trace", "feature": "mean", "expect": 0.0, "tol": 0.1, **W, "group": "p2p", "property": "centred",
     "note": "zero-mean random data: |mean| <= 0.03 kHz over numpy seeds 0-7"},
]
graph_checks = [
    {"kind": "same_blocks_as_start", "variables": "value_only", "ignore_ids": ["import", "note"], "group": "p2p",
     "property": "values_only",
     "note": "no block added, removed or bypassed; every variable keeps its name, none added or removed"},
    {"kind": "same_connections_as_start", "mode": "isomorphic", "group": "p2p", "property": "values_only",
     "note": "no rewiring"},
    {"kind": "block_present", "ids": ["variable"], "params": {"value": str(FS)}, "min": 1, "group": "p2p",
     "property": "sample_rate", "note": "samp_rate stays 48000 (a numeric spelling such as 48e3 also matches)"},
]

spec_properties = [
    {"id": "bit_rate", "text": "GMSK at 4 kbit/s (48 kHz / 12 samples per symbol), i.e. a +-1 kHz deviation, "
                               "carrying the bits of the unchanged random byte source",
     "checked_by": "f2p signed corr_ref of freq_trace.dat against the trace of the seeded random bytes at 4 kbit/s "
                   "(independent of the demodulator gain); f2p max_abs (peak = bit_rate/4 in kHz)"},
    {"id": "trace_scale", "text": "freq_trace.dat shows the instantaneous frequency in kHz: largest absolute "
                                  "value 1.0 within 3 %", "checked_by": "f2p max_abs on freq_trace.dat"},
    {"id": "trace_sign", "text": "the trace is positive above 0 Hz and negative below",
     "checked_by": "f2p signed corr_ref (a negated trace scores -1 at the aligned lag, 0.04 at best)"},
    {"id": "occupied_bandwidth", "text": "at least 99.9 % of the power of gmsk_iq.dat within -2.5 kHz..+2.5 kHz "
                                         "(at 48 kHz)", "checked_by": "f2p band_fraction on gmsk_iq.dat"},
    {"id": "values_only", "text": "only parameter values change: no block added, removed or rewired, no variable "
                                  "added or removed",
     "checked_by": "p2p graph checks same_blocks_as_start (variables value_only; import/note blocks ignored) and "
                   "same_connections_as_start (isomorphic)"},
    {"id": "sample_rate", "text": "the sample rate stays 48 kHz (samp_rate = 48000)",
     "checked_by": "p2p graph check block_present variable with value 48000; a consistent rate change with "
                   "rescaled samples per symbol also fails the f2p corr_ref"},
    {"id": "channel_unchanged", "text": "the HW Impairments block stays and changes nothing (phase noise -100 dB, "
                                        "all other impairments 0): the complex signal keeps constant amplitude 1.0 "
                                        "(RMS within 5 %)",
     "checked_by": "p2p rms on gmsk_iq.dat (magnitude imbalance, DC offset, nonlinearity); phase imbalance and "
                   "phase noise leave the RMS unchanged but raise the f2p trace peak above 1.03; removing or "
                   "bypassing the block fails same_blocks_as_start"},
    {"id": "centred", "text": "the signal stays centred at 0 Hz: the trace averages to 0 within ±0.1 kHz "
                              "(random, balanced data)", "checked_by": "p2p mean on freq_trace.dat"},
    {"id": "outputs", "text": "complex items to the File Sink gmsk_iq.dat and float items to the File Sink "
                              "freq_trace.dat", "checked_by": "observation contract (file name and item type)"},
    {"id": "run", "text": "the flowgraph keeps producing samples until it is stopped",
     "checked_by": f"V5 liveness: at least {RUN_ITEMS} items at each observation point"},
]

mutations = [
    {"op": "set_param", "block": MOD, "param": "samples_per_symbol", "value": str(START_SPS), "breaks": "bit_rate",
     "note": "start value left: 6 kbit/s (peak 1.5, bandwidth too wide)"},
    {"op": "set_param", "block": MOD, "param": "samples_per_symbol", "value": ORIG["sps"], "breaks": "bit_rate",
     "note": "the example's 10 samples per symbol: 4.8 kbit/s (peak 1.2)"},
    {"op": "set_param", "block": MOD, "param": "samples_per_symbol", "value": "16", "breaks": "bit_rate",
     "note": "3 kbit/s (peak 0.75)"},
    {"op": "set_param", "block": MOD, "param": "bt", "value": START_BT, "breaks": "occupied_bandwidth",
     "note": "start BT left: 0.148 % of the power outside +-2.5 kHz"},
    {"op": "set_param", "block": MOD, "param": "bt", "value": "1.0", "breaks": "occupied_bandwidth"},
    {"op": "set_param", "block": DEMOD, "param": "gain", "value": "1.0", "breaks": "trace_scale",
     "note": "start gain left: radians per sample"},
    {"op": "set_param", "block": DEMOD, "param": "gain", "value": "samp_rate/(2*math.pi)", "breaks": "trace_scale",
     "note": "trace in Hz instead of kHz"},
    {"op": "set_param", "block": DEMOD, "param": "gain", "value": "samp_rate/1000", "breaks": "trace_scale",
     "note": "2*pi missing"},
    {"op": "scale_param", "block": DEMOD, "param": "gain", "factor": 1.1, "breaks": "trace_scale"},
    {"op": "set_param", "block": DEMOD, "param": "gain", "value": "-samp_rate/(2*math.pi*1000)",
     "breaks": "trace_sign", "note": "negated trace: every amplitude check passes, only the signed corr_ref fails"},
    {"op": "set_param", "block": "samp_rate", "param": "value", "value": ORIG["samp_rate"], "breaks": "sample_rate",
     "note": "the example's 32 kHz: the kHz gain no longer matches the 48 kHz signal"},
    {"op": "set_param", "block": "magbal", "param": "value", "value": "3", "breaks": "channel_unchanged",
     "note": "3 dB IQ magnitude imbalance"},
    {"op": "set_param", "block": "i_ofs", "param": "value", "value": "0.5", "breaks": "channel_unchanged",
     "note": "DC offset on I"},
    {"op": "set_param", "block": "beta", "param": "value", "value": "-0.1", "breaks": "channel_unchanged",
     "note": "third-order nonlinearity"},
    {"op": "set_param", "block": "phasebal", "param": "value", "value": "10", "breaks": "channel_unchanged",
     "note": "10 degree IQ phase imbalance (RMS unchanged; caught by the trace peak, 1.19)"},
    {"op": "set_param", "block": "phase_noise_mag", "param": "value", "value": "0", "breaks": "channel_unchanged",
     "note": "'tidied' to 0, which is 0 dB of phase noise, not off (RMS unchanged; caught by the trace peak)"},
    {"op": "bypass", "block": CHAN, "breaks": "channel_unchanged, values_only",
     "note": "HW Impairments removed: the signals are unchanged, the block-set check fails"},
    {"op": "set_param", "block": SRC, "param": "max", "value": "2", "breaks": "centred",
     "note": "bytes 0/1 only: 7 of 8 bits are 0, the signal sits mostly at -1 kHz"},
    {"op": "set_param", "block": SRC, "param": "min", "value": "1", "breaks": "bit_rate",
     "note": "other random bytes (the random byte source changed): corr_ref 0.43, every other check passes"},
    {"op": "set_param", "block": TRACE_SINK, "param": "file", "value": "trace.dat", "breaks": "outputs",
     "note": "the trace is written to another file name: observation freq_trace.dat missing"},
]

ORACLE_LIMITS = (
    "Known oracle limits: (1) the bit-rate/sign check (signed corr_ref of freq_trace.dat) compares with the trace "
    "the Random Source must produce when the harness seeds numpy with 0 before the flowgraph is built; it relies "
    "on the stated 'keep the random byte source' and would fail a correct answer only if a block that draws from "
    "numpy's global random state were constructed before the Random Source (no block of this graph does). "
    "(2) samp_rate = 48000 is checked as a graph fact (a variable with value 48000); the Throttle's own rate "
    "parameter is not checked. (3) The HW Impairments 'Frequency Offset' does not move the signal, so a changed "
    "freq_offset value is invisible (it changes nothing observable).")

record = {
    "id": TASK_ID, "tier": "F", "split": SPLIT, "version": 2, "canary": CANARY,
    "license": f"GPL-3.0-or-later (derived from GNU Radio 3.10.12 example {SEED})",
    "source": {"kind": "gnuradio_example", "ref": SEED, "sha256": seed_sha, "derivation": DERIVATION},
    "cluster_id": SEED,
    "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring",
                   "generator_version": "0.2.0"},
    "review": {"status": "pending-team-review", "reviewers": [],
               "notes": (ORACLE_LIMITS + " Attribution: derived from the GNU Radio 3.10.12 example "
                         f"{SEED} (author field 'mettus'; GNU Radio is GPL-3.0-or-later); the conversion and number "
                         "changes are listed in perturbation.changes. The licence decision for seed-derived items is the "
                         "team's.")},
    "family": "parameter-tune", "level": "L2",
    "instruction": instruction,
    "start_grc": "start.grc", "reference_grc": "reference.grc",
    "observation": observation,
    "oracle": {"checks": checks, "graph_checks": graph_checks},
    "mutations": mutations,
    "spec_properties": spec_properties,
    "perturbation": PERTURBATION,
    "disjointness": ("Parameter-tune of the test-split seed channels/demo_gmsk.grc (bit rate, Gaussian BT for an "
                     "occupied-bandwidth limit, discriminator gain in kHz); not t21_gmsk_pipe_build (build a "
                     "GMSK modulate-unpack-demodulate data chain and run it) nor t10_param_tuning (retune a "
                     "receiver's centre frequency/bandwidth and rename variables), and none of the other "
                     "training-side specs (dial tone, QPSK BER, FM/WBFM receiver repair, QAM debug, AM demod, "
                     "spectrogram sweep, PSK31, RSSI, resampler repair, Embedded-Python blocks). Within the test "
                     "split it differs from gbf-test-20 (baseband PAM symbol mapping retune) and gbf-test-13 "
                     "(pulse-shaped data upconverted to an IF, build)."),
    "authoring_notes": {
        "level": "L2: 7 functional blocks (Random Source, GMSK Mod, Throttle, HW Impairments, Quadrature "
                 "Demod, 2 File Sinks) with modulation (and interpolation inside GMSK Mod) and demodulation",
        "parameters_to_tune": "samples_per_symbol 8 -> 12, bt 0.5 -> <= 0.4, Quadrature Demod gain 1.0 -> "
                              "samp_rate/(2*pi*1000); each alone or in pairs still fails",
        "start_grc": "compact form, except that the impairment variables keep their values visible "
                     "(e.g. magbal = 0) although they equal the variable default",
        "variants": "all five change parameter values only (value-only rule): BT 0.25/0.3/0.35/0.4/0.15, with "
                    "samples per symbol and gain written as literals, samp_rate expressions or the 2*sps/pi idiom",
        "design": "see build.py docstring",
    },
}

files = {"reference.grc": grcfile.dump(ref_c), "start.grc": grcfile.dump(start_c)}
files.update({f"variants/{k}": grcfile.dump(v) for k, v in variants.items()})

if __name__ == "__main__":
    if (HERE / "variants").exists():
        shutil.rmtree(HERE / "variants")
    errors = taskio.save(HERE, record, files)
    if errors:
        raise SystemExit("schema errors:\n" + "\n".join(errors))
    print(f"written {HERE} (seed {SEED}, split {seed_split['split']}, sha256 {seed_sha[:12]})")
