"""Authoring script for GRC-Bench Tier-F item gbf-test-16 (family modify, level L2, split test).

Seed: GNU Radio 3.10.12 example filter/polyphase_channelizer_demo.grc (a test-split seed in
splits/flowgraph_seed_split.json). Complex tones and a little complex Gaussian noise are summed;
the sum goes to a spectrum display and into a Hierarchical Polyphase Channelizer that splits it into
three channels (centred on 0, +fs/3 and -fs/3, each at fs/3), one display per channel.

Task: the user wants a 12 kHz channel grid instead of the 16 kHz one, i.e. the 48 kHz input split
into four channels at 12 kHz each, with the new fourth channel (centred on -12 kHz) written to a new
File Sink. The model has to change the channel count, wire and type a new File Sink, and know how a
4-channel polyphase channelizer orders its outputs (output k centred on k * 12 kHz, so output 3 is
the -12 kHz channel; round 1 removed that hint from the instruction, which now names the channels
by frequency only) - while keeping the tones, the noise and the wideband output as they are.

Start graph = the seed exported by `./gb tier-f seed` with its numbers perturbed, a fourth tone
added and its four exported File Sinks given readable names (see PERTURBATION). Round-1 changes:
  * tone frequencies are absolute (Hz) instead of fractions of samp_rate, so a changed sample rate
    moves the tones relative to the channel grid and is observable (compliance F3);
  * a fourth tone (23.04 kHz, amplitude 0.2) lies in the new channel 2, so every channel's 12 kHz
    rate is measured through a tone offset (compliance F4: channel 2 used to hold only noise);
  * the noise is checked on spectrum.dat (power share in the tone-free band 3-11 kHz, Hann window,
    within 30 % of that of complex noise with amplitude 0.02), so removing or rescaling it fails
    (F1 / F3). The noise source is deterministic from run to run; over 40 noise seeds the share
    varies by 5.1 % (relative std; numpy model 5.3 %, 1e-4 quantiles -17.5 % / +18.6 %), so 30 % is
    more than 5.7 sigma, while amplitudes 0.016 and 0.025 (power x0.64 / x1.56) fail;
  * the filter requirement is stated as what every valid design achieves and the oracle checks
    (signals within 2 kHz of a centre keep offset and amplitude; signals 10 kHz or more away are
    filtered out) instead of an impossible brick wall at the channel edge (F2);
  * 4096 items per observation point (taps no longer back-pressure the graph, tooling T7), channel
    checks after a 1024-item transient (prototype filters up to ~4000 taps), and FFT-based and very
    sharp designs among the variants (item-quality F1).
Round 2 (G10R-2): each channel file must also carry the input noise near its centre (power share in a
tone-free band within 2 kHz of the centre at least half that of the original noise), so an answer that
writes synthetic tones to the channel files without channelizing fails (build.py --probes).
Observation points: the complex samples written by the File Sinks to spectrum.dat (wideband sum,
48 kHz; pass-to-pass) and to ch0.dat .. ch3.dat (channel outputs, 12 kHz; fail-to-pass).

Tone placement (48 kHz): 1.2 kHz / 0.8, 13.92 kHz / 0.5, 34.56 kHz (= -13.44 kHz) / 0.3,
23.04 kHz / 0.2.
  4 channels (spacing 12 kHz): ch0 +1.2 kHz, ch1 +1.92 kHz, ch2 -0.96 kHz, ch3 -1.44 kHz; every
  tone is at least 10.08 kHz from every other channel's centre.
  3 channels (spacing 16 kHz, start): ch0 +1.2 kHz, ch1 -2.08 kHz and +7.04 kHz, ch2 +2.56 kHz.
Tone offsets and amplitudes differ between channels, so mis-wired outputs are detected.

Regenerate (WSL, from the grc-bench folder; deterministic):
    PYTHONPATH=src ~/grc-agent/.venv/bin/python items/tier_f/test/gbf-test-16/build.py
then run the gates:
    ./gb tier-f gates items/tier_f/test/gbf-test-16
"""

from __future__ import annotations

import copy
import math
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from grcbench.common.io import read_json, sha256_file           # noqa: E402
from grcbench.common.paths import GNURADIO_EXAMPLES, SPLITS      # noqa: E402
from grcbench.tier_f import edit, grc, grcfile, taskio           # noqa: E402
from grcbench.tier_f.seedtools import headless_seed              # noqa: E402

TASK_ID = "gbf-test-16"
SPLIT = "test"
SEED = "filter/polyphase_channelizer_demo.grc"

FS = 48000                         # input sample rate (perturbed from 32000)
TONES = [                          # (block, freq in Hz, amplitude); perturbed, absolute frequencies
    ("analog_sig_source_x_0", 1200, 0.8),
    ("analog_sig_source_x_1", 13920, 0.5),
    ("analog_sig_source_x_2", 34560, 0.3),
    ("analog_sig_source_x_3", 23040, 0.2),     # added: lies in the new channel 2
]
ORIG_FREQS = {"analog_sig_source_x_0": "0.03 * samp_rate", "analog_sig_source_x_1": "0.38 * samp_rate",
              "analog_sig_source_x_2": "0.70*samp_rate"}
NOISE_AMP = 0.02                   # perturbed from .01
NOISE, ADD, THR, CHAN = ("analog_noise_source_x_0", "blocks_add_xx_0", "blocks_throttle2_0",
                         "pfb_channelizer_hier_ccf_0")

N_OLD, N_NEW = 3, 4                # channels in the start graph / requested
SPACING = FS // N_NEW              # 12 kHz: channel spacing and channel output rate requested
CH_FS = FS / N_NEW
CENTRES = [0, 12000, 24000, -12000]  # channel k of the requested plan (the channelizer's own order)

# exported File Sinks (seed tool names) -> readable names used by the task
SINKS = {  # seed sink name: (new block name, file)
    "qtgui_freq_sink_x_0_0": ("spectrum_sink", "spectrum.dat"),     # was the "Spectrum" display
    "qtgui_freq_sink_x_0_2_0": ("ch0_sink", "ch0.dat"),            # was the "Ch0" display
    "qtgui_freq_sink_x_0_1_0": ("ch1_sink", "ch1.dat"),            # was the "Ch1" display
    "qtgui_freq_sink_x_0_0_0": ("ch2_sink", "ch2.dat"),            # was the "Ch2" display
}
CH3_SINK, CH3_FILE = "ch3_sink", "ch3.dat"


def wrap(f: float, fs: float) -> float:
    """Frequency folded into [-fs/2, fs/2)."""
    return (f + fs / 2) % fs - fs / 2


def chan_offset(f: float, centre: float) -> float:
    """Where a tone at f appears in the channel centred on `centre` (Hz, relative to the centre)."""
    return wrap(f - centre, FS)


# which tone lies in which requested channel (within 2 kHz of its centre); all others >= 10 kHz away
TONE_OF = {}
for _k, _c in enumerate(CENTRES):
    _near = [t for t in TONES if abs(chan_offset(t[1], _c)) <= 2000]
    assert len(_near) == 1, (_k, _near)
    assert all(abs(chan_offset(t[1], _c)) >= 10000 for t in TONES if t not in _near), _k
    TONE_OF[_k] = _near[0]


# --------------------------------------------------------------------------- flowgraph helpers
def blk(data: dict, name: str) -> dict:
    return next(b for b in data["blocks"] if b["name"] == name)


def setp(data: dict, name: str, **params) -> dict:
    d = copy.deepcopy(data)
    blk(d, name).setdefault("parameters", {}).update({k: str(v) for k, v in params.items()})
    return d


def rename(data: dict, old: str, new: str) -> dict:
    d = copy.deepcopy(data)
    blk(d, old)["name"] = new
    d["connections"] = [[new if c[0] == old else c[0], c[1], new if c[2] == old else c[2], c[3]]
                        for c in d["connections"]]
    return d


def file_sink(name: str, path: str) -> dict:
    return {"name": name, "id": "blocks_file_sink", "parameters": {"type": "complex", "file": path}}


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


# --------------------------------------------------------------------------- start / reference
def build_start() -> dict:
    seed, sinks = headless_seed(SEED)
    assert sorted(s["name"] for s in sinks) == sorted(SINKS), sinks
    d = copy.deepcopy(seed)
    d["options"]["parameters"] = {"id": "polyphase_channelizer_demo", "generate_options": "no_gui",
                                  "output_language": "python"}
    assert blk(d, "samp_rate")["parameters"]["value"] == "32000"
    d = setp(d, "samp_rate", value=FS)
    for name, expr in ORIG_FREQS.items():
        assert blk(d, name)["parameters"]["freq"] == expr, (name, blk(d, name)["parameters"])
    # the added fourth tone: a copy of the first tone source on a fifth Add input
    extra = copy.deepcopy(blk(d, TONES[0][0]))
    extra["name"] = TONES[3][0]
    d["blocks"].append(extra)
    assert blk(d, ADD)["parameters"]["num_inputs"] == "4"
    d = setp(d, ADD, num_inputs=5)
    d["connections"].append([TONES[3][0], "0", ADD, "4"])
    for name, freq, amp in TONES:
        d = setp(d, name, freq=freq, amp=amp)
    d = setp(d, NOISE, amp=NOISE_AMP)
    for old, (new, path) in SINKS.items():
        d = setp(rename(d, old, new), new, file=path)
    order = ["samp_rate", NOISE, *[t[0] for t in TONES], THR, ADD, CHAN,
             "spectrum_sink", "ch0_sink", "ch1_sink", "ch2_sink"]
    d["blocks"] = sorted(d["blocks"], key=lambda b: order.index(b["name"]))
    d["connections"] = sorted(d["connections"])
    return d


def build_reference(start: dict) -> dict:
    d = setp(start, CHAN, nchans=N_NEW)
    d["blocks"].append(file_sink(CH3_SINK, CH3_FILE))
    d["connections"].append([CHAN, "3", CH3_SINK, "0"])
    return d


PERTURBATION = {"perturbed": True, "changes": [
    {"what": "sample rate (variable samp_rate)", "original": "32000", "new": str(FS)},
    {"what": "tone 1 frequency (analog_sig_source_x_0)", "original": ORIG_FREQS[TONES[0][0]],
     "new": f"{TONES[0][1]} (Hz, absolute)"},
    {"what": "tone 2 frequency (analog_sig_source_x_1)", "original": ORIG_FREQS[TONES[1][0]],
     "new": f"{TONES[1][1]} (Hz, absolute)"},
    {"what": "tone 3 frequency (analog_sig_source_x_2)", "original": ORIG_FREQS[TONES[2][0]],
     "new": f"{TONES[2][1]} (Hz, absolute; = -13440 Hz)"},
    {"what": "tone amplitudes (sources 0 / 1 / 2)", "original": "1 / 1 / 1",
     "new": " / ".join(str(t[2]) for t in TONES[:3])},
    {"what": "added a fourth tone source analog_sig_source_x_3 (copy of source 0) on a fifth Add input",
     "original": "three tones, Add with 4 inputs",
     "new": f"tone {TONES[3][1]} Hz, amplitude {TONES[3][2]}; Add with 5 inputs (so that every channel of the "
            "requested plan carries a tone and its rate is observable)"},
    {"what": "noise amplitude (analog_noise_source_x_0)", "original": ".01", "new": str(NOISE_AMP)},
    {"what": "Qt GUI frequency sinks", "original": "qtgui_freq_sink_x (GUI): Spectrum, Ch0, Ch1, Ch2",
     "new": "File Sinks spectrum_sink / ch0_sink / ch1_sink / ch2_sink writing spectrum.dat / ch0.dat / "
            "ch1.dat / ch2.dat (exported by the seed tool, then renamed)"},
    {"what": "generate_options", "original": "qt_gui", "new": "no_gui (seed tool)"},
    {"what": "options block metadata (conversion)",
     "original": ("id 'top_block', category 'Custom', window_size '1280, 1024', run_options 'prompt', "
                  "catch_exceptions, placement, sizing_mode"),
     "new": "id 'polyphase_channelizer_demo'; the options block keeps only id, generate_options and output_language"},
    {"what": "catalog-default parameters and GUI states (conversion)", "original": "written out in the example",
     "new": "omitted (compact form; GRC fills in the same defaults, checked by grc.compact_equivalent)"},
]}


def with_chain(ref: dict, blocks: list[dict], conns: list[list[str]], drop_chan: bool = True) -> dict:
    """The reference with the channelizer replaced by other blocks feeding the four channel sinks."""
    d = copy.deepcopy(ref)
    if drop_chan:
        d["blocks"] = [b for b in d["blocks"] if b["name"] != CHAN]
        d["connections"] = [c for c in d["connections"] if CHAN not in (c[0], c[2])]
    d["blocks"] += copy.deepcopy(blocks)
    d["connections"] += [[str(x) for x in c] for c in conns]
    return d


def chan_block(block_id: str, **params) -> dict:
    return {"name": CHAN, "id": block_id, "parameters": {k: str(v) for k, v in params.items()}}


def per_channel(prefix: str, make) -> tuple[list[dict], list[list[str]]]:
    """Four parallel per-channel chains ADD -> ... -> ch<k>_sink; make(k, centre) -> list of blocks in order."""
    blocks, conns = [], []
    for k, centre in enumerate(CENTRES):
        chain = make(k, centre)
        for b in chain:
            b["name"] = f"{prefix}{b['name']}{k}"
        blocks += chain
        names = [ADD] + [b["name"] for b in chain] + [f"ch{k}_sink"]
        conns += [[a, "0", b, "0"] for a, b in zip(names, names[1:])]
    return blocks, conns


def build_variants(ref: dict) -> dict[str, dict]:
    v = {}
    sink_conns = [[CHAN, str(k), f"ch{k}_sink", "0"] for k in range(N_NEW)]
    taps_var = {"name": "chan_taps", "id": "variable",
                "parameters": {"value": "firdes.low_pass(1.0, samp_rate, 5000, 2000)"}}
    ctaps_var = {"name": "chan_taps_c", "id": "variable",
                 "parameters": {"value": "[complex(t) for t in firdes.low_pass(1.0, samp_rate, 5000, 2000)]"}}
    # 1) the C++ Polyphase Channelizer block with explicitly designed prototype taps
    v["v1_pfb_channelizer_ccf"] = with_chain(
        ref, [chan_block("pfb_channelizer_ccf", nchans=4, taps="firdes.low_pass_2(1.0, samp_rate, 5000, 2000, 80)")],
        [[ADD, "0", CHAN, "0"]] + sink_conns)
    # 2) no channelizer: one frequency-translating FIR filter per channel, each decimating by 4
    blocks, conns = per_channel("xlate_ch", lambda k, c: [{"name": "", "id": "freq_xlating_fir_filter_xxx",
                                                         "parameters": {"type": "ccf", "decim": "4",
                                                                        "taps": "chan_taps",
                                                                        "center_freq": f"{c:g}",
                                                                        "samp_rate": "samp_rate"}}])
    v["v2_xlating_filters"] = with_chain(ref, [taps_var] + blocks, conns)
    # 3) hierarchical channelizer again, channel count in a variable, other filter settings
    d = setp(ref, CHAN, nchans="n_chans", bw="0.4", atten="80", n_filterbanks="1")
    d["blocks"].insert(1, {"name": "n_chans", "id": "variable", "parameters": {"value": "int(samp_rate // 12e3)"}})
    v["v3_hier_variables"] = d
    # round-1 reviewers' alternatives (all correct; several used to fail V5 liveness or the old 128-item skip)
    for name, params in [("alt_hier_bw045_tb005", {"bw": "0.45", "tb": "0.05"}),
                         ("alt_hier_bw048_tb002", {"bw": "0.48", "tb": "0.02"}),
                         ("alt_hier_bw05_tb002", {"bw": "0.5", "tb": "0.02"})]:
        v[name] = setp(ref, CHAN, **params)
    # user taps in the hierarchical channelizer: its block does not import firdes (GRC's validation
    # rejects a firdes expression there), so these answers add an Import block for it
    imp = {"name": "import_firdes", "id": "import", "parameters": {"imports": "from gnuradio.filter import firdes"}}
    for name, params in [("alt_hier_user_taps", {"n_filterbanks": "1",
                                                 "taps": "firdes.low_pass(1, samp_rate, 6000, 1200)"}),
                         ("alt_hier_taps_lp2_100db", {"taps": "firdes.low_pass_2(1.0, samp_rate, 5800, 200, 100)"})]:
        d = setp(ref, CHAN, **params)
        d["blocks"].insert(1, copy.deepcopy(imp))
        v[name] = d
    for name, taps in [("alt_pfb_lp2_sharp", "firdes.low_pass_2(1, samp_rate, 5800, 200, 100)"),
                       ("alt_pfb_lp2_verysharp", "firdes.low_pass_2(1, samp_rate, 5900, 100, 100)"),
                       # 4363 taps: a start-up transient of about 1091 channel samples, longer than the
                       # 1024-sample skip; guards the transient margin of the channel RMS checks
                       ("alt_pfb_lp2_tw50_4363taps", "firdes.low_pass_2(1, samp_rate, 5900, 50, 100)"),
                       ("alt_pfb_lp_hamming", "firdes.low_pass(1, samp_rate, 5500, 500)"),
                       ("alt_pfb_lp2_6k_60db", "firdes.low_pass_2(1, samp_rate, 6000, 1000, 60)")]:
        v[name] = with_chain(ref, [chan_block("pfb_channelizer_ccf", nchans=4, taps=taps)],
                             [[ADD, "0", CHAN, "0"]] + sink_conns)
    # Polyphase Decimators, one per channel (default FFT rotator / FFT filters, and without them)
    for name, extra in [("alt_pfb_decimators", {}), ("alt_pfb_decimators_nofft", {"fft_rot": "False",
                                                                                  "fft_filts": "False"})]:
        blocks, conns = per_channel("dec", lambda k, c, extra=extra: [
            {"name": "", "id": "pfb_decimator_ccf",
             "parameters": {"decim": "4", "taps": "chan_taps", "channel": str(k), **extra}}])
        v[name] = with_chain(ref, [taps_var] + blocks, conns)
    # FFT-based frequency-translating filters (complex taps)
    blocks, conns = per_channel("xf", lambda k, c: [{"name": "", "id": "freq_xlating_fft_filter_ccc",
                                                    "parameters": {"decim": "4", "taps": "chan_taps_c",
                                                                   "center_freq": str(k * SPACING),
                                                                   "samp_rate": "samp_rate"}}])
    v["alt_xlating_fft_filters"] = with_chain(ref, [ctaps_var] + blocks, conns)
    # Rotator + FFT filter (decimating by 4), and Rotator + FIR filter
    blocks, conns = per_channel("", lambda k, c: [
        {"name": "rot", "id": "blocks_rotator_cc", "parameters": {"phase_inc": f"-2*3.141592653589793*{k}*12e3/samp_rate"}},
        {"name": "ff", "id": "fft_filter_xxx", "parameters": {"type": "ccc", "decim": "4", "taps": "chan_taps_c"}}])
    v["alt_rotator_fft_filter"] = with_chain(ref, [ctaps_var] + blocks, conns)
    blocks, conns = per_channel("", lambda k, c: [
        {"name": "rot", "id": "blocks_rotator_cc", "parameters": {"phase_inc": f"-2*math.pi*{k}*12e3/samp_rate"}},
        {"name": "fir", "id": "fir_filter_xxx", "parameters": {"type": "ccf", "decim": "4", "taps": "chan_taps"}}])
    v["alt_rotator_fir"] = with_chain(
        ref, [{"name": "import_math", "id": "import", "parameters": {"imports": "import math"}}, taps_var] + blocks,
        conns)
    # sharp frequency-translating FIR filters (channel 3 written as +36 kHz)
    for name, taps in [("alt_xlating_sharp_lp2", "firdes.low_pass_2(1.0, samp_rate, 5800, 200, 100)"),
                       ("alt_xlating_sharp_tw100", "firdes.low_pass(1.0, samp_rate, 5800, 100)")]:
        blocks, conns = per_channel("xl", lambda k, c: [{"name": "", "id": "freq_xlating_fir_filter_xxx",
                                                        "parameters": {"type": "ccf", "decim": "4",
                                                                       "taps": "sharp_taps",
                                                                       "center_freq": f"{k * 12}e3",
                                                                       "samp_rate": "samp_rate"}}])
        v[name] = with_chain(ref, [{"name": "sharp_taps", "id": "variable", "parameters": {"value": taps}}] + blocks,
                             conns)
    # the Throttle moved behind the Add (the sum it feeds to spectrum.dat and the channelizer is the same)
    d = copy.deepcopy(ref)
    d["connections"] = [c for c in d["connections"] if THR not in (c[0], c[2]) and c[0] != ADD]
    d["connections"] += [[NOISE, "0", ADD, "3"], [ADD, "0", THR, "0"], [THR, "0", CHAN, "0"],
                         [THR, "0", "spectrum_sink", "0"]]
    v["alt_throttle_after_add"] = d
    return {k: tidy(x) for k, x in v.items()}


# --------------------------------------------------------------------------- task record
def khz(f: float) -> str:
    return f"{f / 1000:g} kHz"


INSTRUCTION = (
    "This flowgraph runs at a sample rate of 48 kHz. It adds four complex tones (1.2 kHz with amplitude 0.8, "
    "13.92 kHz with amplitude 0.5, 23.04 kHz with amplitude 0.2, and 34.56 kHz, which is the same as -13.44 kHz, "
    "with amplitude 0.3) and complex Gaussian noise with amplitude 0.02. The sum is written as complex samples by a "
    "File Sink to spectrum.dat, and a polyphase channelizer splits it into three channels 16 kHz apart, whose "
    "outputs are written as complex samples to ch0.dat, ch1.dat and ch2.dat.\n\n"
    "Change the channel plan to four channels 12 kHz apart, centred on 0 Hz, +12 kHz, 24 kHz (the same as "
    "-24 kHz) and -12 kHz (the same as 36 kHz). Write the channel centred on 0 Hz to ch0.dat, the one on +12 kHz "
    "to ch1.dat and the one on 24 kHz to ch2.dat, using the existing File Sinks, and the one on -12 kHz to a new "
    "File Sink that writes complex samples to ch3.dat. Each channel must come out as a complex stream at 12 kHz "
    "with the channel centre moved to 0 Hz. Signals within 2 kHz of a channel's centre must keep their frequency "
    "offset and their amplitude, and signals 10 kHz or more from it must be filtered out of that channel, so "
    "that each channel file carries only the one tone that lies near its centre, plus a little noise. Measured on "
    "each channel file over 3072 samples after skipping its first 1024, that means: the strongest component is at "
    "that tone's offset from the channel centre (within 1 %), the RMS level equals the tone's amplitude "
    "(within 5 %), and the noise within 2 kHz of the centre keeps at least half of its original power.\n\n"
    "Keep the 48 kHz sample rate, the four tones, the noise and the sum written to spectrum.dat as they are. The "
    "flowgraph should keep producing samples until it is stopped; do not limit the number of samples."
)

# Behaviours the instruction requires; every oracle check names the one it tests ("requires").
POWER = sum(t[2] ** 2 for t in TONES) + NOISE_AMP ** 2
NOISE_BAND = (3000, 11000)          # tone-free band of the wideband sum (>= 1.8 kHz from every tone)
assert all(not (NOISE_BAND[0] - 1800 < wrap(t[1], FS) < NOISE_BAND[1] + 1800) for t in TONES)
NOISE_SHARE = NOISE_AMP ** 2 * (NOISE_BAND[1] - NOISE_BAND[0]) / FS / POWER
NOISE_REL_TOL = 0.30                # > 5.7 sigma of the seed-to-seed spread (5.1 %); kills amplitude x0.8 / x1.25


# Noise near each channel centre (round 2, G10R-2): a tone-free 1.8 kHz band within 2 kHz of the centre, on the
# side away from the channel's tone. Every valid design passes it unchanged, so its power share is the original
# noise density over the band; the check asks for at least half of it (variants measured -10 % ... +20 %), and
# synthetic tones written straight to the channel files give 0.
def noise_near(k: int) -> tuple[int, int]:
    return (-2000, -200) if chan_offset(TONE_OF[k][1], CENTRES[k]) > 0 else (200, 2000)


def noise_near_share(k: int) -> float:
    lo, hi = noise_near(k)
    a = TONE_OF[k][2]
    return NOISE_AMP ** 2 * (hi - lo) / FS / (a * a + NOISE_AMP ** 2 * CH_FS / FS)


NOISE_NEAR_MIN = 0.5


def _ch_spec(k: int) -> str:
    name, f, a = TONE_OF[k]
    off = chan_offset(f, CENTRES[k])
    lo, hi = noise_near(k)
    new = " (new File Sink, complex)" if k == 3 else ""
    return (f"ch{k}.dat{new}: the channel centred on {khz(CENTRES[k])} as a complex 12 kHz stream with its centre "
            f"at 0 Hz: strongest component at {off:+g} Hz (the {khz(f)} tone; within 1 %, measured at 12 kHz, so a "
            f"wrong rate or centre fails) and RMS {a} within 5 % (the tone kept at full amplitude, the tones 10 kHz "
            f"or more away filtered out), and the input noise near the centre passed: power share in {lo:+d} to "
            f"{hi:+d} Hz (tone-free, Hann window) at least half that of the original noise "
            f"({noise_near_share(k):.3g}), samples 1024-4095")


SPEC = {
    "wide": ("spectrum.dat still holds the unchanged 48 kHz sum (so the sample rate and the four tones are kept): "
             "strongest component at 1.2 kHz, the power share of each tone (amplitudes 0.8 / 0.5 / 0.2 / 0.3, "
             "+-480 Hz, Hann window) within 10 %, total RMS sqrt(1.0204) = 1.01 within 5 %, first 4096 samples"),
    "noise": (f"the noise is kept: spectrum.dat's power share in the tone-free band 3-11 kHz (Hann window) within "
              f"30 % of that of complex noise with amplitude 0.02 ({NOISE_SHARE:.3g}; amplitude within about 15 %); "
              f"Gaussian and uniform noise of the same power cannot be told apart"),
    "ch0": _ch_spec(0), "ch1": _ch_spec(1), "ch2": _ch_spec(2), "ch3": _ch_spec(3),
    "run": ("run condition: the flowgraph keeps producing samples until it is stopped (no sample limit); checked only "
            "by V5 liveness (at least 4096 items at every observation point), not by V6"),
    "obs": ("observation points: complex samples written to the File Sinks spectrum.dat (48 kHz) and ch0.dat - "
            "ch3.dat (12 kHz), 4096 items each"),
}

# Run length: 4096 items at every observation point (16384 input samples, about 0.34 s through the
# 48 kHz Throttle). The channel checks skip the first 1024 channel samples, the start-up transient
# of a prototype filter of up to about 4000 taps (the round-1 reviewer's 2181-tap design needs 545).
N_ITEMS = 4096
WIDE_N = N_ITEMS
CH_SKIP = 1024
CH_N = N_ITEMS - CH_SKIP
BAND = 480                          # +-480 Hz around each tone in the wideband spectrum (+-41 FFT bins)


def build_oracle() -> dict:
    wide = {"point": "spectrum", "skip": 0, "n": WIDE_N, "group": "p2p", "requires": SPEC["wide"]}
    f_a = TONES[0][1]
    checks = [{**wide, "feature": "peak_freq", "fs": FS, "expect": f_a, "tol": round(2 * FS / WIDE_N, 2)}]
    for _, f, a in TONES:
        fw = wrap(f, FS)
        checks.append({**wide, "feature": "band_fraction", "fs": FS, "lo": fw - BAND, "hi": fw + BAND,
                       "window": "hann", "expect": round(a * a / POWER, 4), "rel_tol": 0.10})
    checks.append({**wide, "feature": "rms", "expect": round(math.sqrt(POWER), 4), "rel_tol": 0.05})
    checks.append({"point": "spectrum", "skip": 0, "n": WIDE_N, "group": "p2p", "requires": SPEC["noise"],
                   "feature": "band_fraction", "fs": FS, "lo": NOISE_BAND[0], "hi": NOISE_BAND[1], "window": "hann",
                   "expect": float(f"{NOISE_SHARE:.4g}"), "rel_tol": NOISE_REL_TOL})
    for k in range(N_NEW):
        _, f, a = TONE_OF[k]
        off = chan_offset(f, CENTRES[k])
        base = {"point": f"ch{k}", "skip": CH_SKIP, "n": CH_N, "group": "f2p", "requires": SPEC[f"ch{k}"]}
        checks.append({**base, "feature": "peak_freq", "fs": CH_FS, "expect": off, "tol": round(0.01 * abs(off), 2)})
        checks.append({**base, "feature": "rms", "expect": a, "rel_tol": 0.05})
        lo, hi = noise_near(k)
        checks.append({**base, "feature": "band_fraction", "fs": CH_FS, "lo": lo, "hi": hi, "window": "hann",
                       "min": float(f"{NOISE_NEAR_MIN * noise_near_share(k):.3g}")})
    return {"checks": checks}


def probes(ref: dict) -> dict[str, dict]:
    """Wrong answers a single edit operator cannot express; build.py --probes checks that each fails V6.
    s1: the channelizer deleted and four Signal Sources at 12 kHz (each channel's tone offset and amplitude)
    written straight to ch0-ch3. s2: the channelizer kept for ch0 only, ch1-ch3 synthetic."""
    def tones(ks):
        blocks, conns = [], []
        for k in ks:
            _, f, a = TONE_OF[k]
            blocks.append({"name": f"fake{k}", "id": "analog_sig_source_x",
                           "parameters": {"type": "complex", "samp_rate": str(int(CH_FS)),
                                          "freq": str(chan_offset(f, CENTRES[k])), "amp": str(a)}})
            conns.append([f"fake{k}", "0", f"ch{k}_sink", "0"])
        return blocks, conns
    s1 = with_chain(ref, *tones(range(N_NEW)))
    s2 = with_chain(ref, *tones(range(1, N_NEW)), drop_chan=False)
    s2["connections"] = [c for c in s2["connections"] if not (c[0] == CHAN and c[2] != "ch0_sink")]
    s2["blocks"] += [{"name": f"park{k}", "id": "blocks_null_sink", "parameters": {"type": "complex"}}
                     for k in range(1, N_NEW)]                          # unused channelizer outputs parked
    s2["connections"] += [[CHAN, str(k), f"park{k}", "0"] for k in range(1, N_NEW)]
    return {"s1_synthetic_channels": s1, "s2_synthetic_ch1_to_ch3": s2}


def run_probes(ref: dict) -> int:
    from grcbench.tier_f.ladder import check
    task = taskio.load(HERE)
    bad = 0
    for name, g in probes(ref).items():
        res = check(dump(tidy(g)), task)
        passes = res["first_failed"] is None and res["highest_passed"] == "V6"
        v6 = res["rungs"].get("V6") or {}
        failed = [f"{c['point']}:{c['feature']}" for c in v6.get("checks", []) if not c["passed"]]
        bad += passes
        print(f"{name:26s} passes V6: {passes!s:5s} (expected False); first failed {res['first_failed']}; {failed}")
    return 1 if bad else 0


MUTATIONS = [
    {"op": "rewire_input", "dst": CH3_SINK, "dst_port": "0", "new_src": CHAN, "new_src_port": "1",
     "breaks": SPEC["ch3"], "note": "channel 1 written to ch3.dat"},
    {"op": "rewire_input", "dst": "ch2_sink", "dst_port": "0", "new_src": CHAN, "new_src_port": "3",
     "breaks": SPEC["ch2"], "note": "channel 3 written to ch2.dat"},
    {"op": "set_param", "block": CHAN, "param": "outchans", "value": "[0, 1, 3, 2]", "breaks": SPEC["ch2"],
     "note": "channels 2 and 3 written to each other's files"},
    {"op": "rewire_input", "dst": "ch0_sink", "dst_port": "0", "new_src": ADD, "new_src_port": "0",
     "breaks": SPEC["ch0"], "note": "the 48 kHz wideband sum (wrong rate, not channelized) written to ch0.dat"},
    {"op": "rewire_input", "dst": "ch2_sink", "dst_port": "0", "new_src": ADD, "new_src_port": "0",
     "breaks": SPEC["ch2"], "note": "the 48 kHz wideband sum written to ch2.dat (channel 2's rate, round-1 F4)"},
    {"op": "set_param", "block": CHAN, "param": "bw", "value": "1.4", "breaks": SPEC["ch2"],
     "note": "channel filter far too wide: tones 10 kHz away leak into channel 2"},
    {"op": "set_param", "block": CHAN, "param": "bw", "value": "0.1", "breaks": SPEC["ch1"],
     "note": "channel filter too narrow: the tone 1.92 kHz from the centre loses amplitude"},
    {"op": "set_param", "block": "samp_rate", "param": "value", "value": "32000", "breaks": SPEC["wide"],
     "note": "the example's original sample rate: the absolute tone frequencies move relative to the rate"},
    {"op": "set_param", "block": "analog_sig_source_x_1", "param": "freq", "value": "0.38 * samp_rate",
     "breaks": SPEC["wide"], "note": "tone 2 back at the example's original 0.38 * samp_rate"},
    {"op": "scale_param", "block": "analog_sig_source_x_2", "param": "freq", "factor": 1.05,
     "breaks": SPEC["ch3"], "note": "tone 3 frequency 5 % off"},
    {"op": "set_param", "block": "analog_sig_source_x_0", "param": "amp", "value": "1", "breaks": SPEC["ch0"],
     "note": "tone 1 amplitude 1 instead of 0.8"},
    {"op": "rewire_input", "dst": "spectrum_sink", "dst_port": "0", "new_src": "analog_sig_source_x_0",
     "breaks": SPEC["wide"], "note": "spectrum.dat records one tone instead of the sum"},
    {"op": "set_param", "block": NOISE, "param": "amp", "value": "0", "breaks": SPEC["noise"],
     "note": "noise removed (amplitude 0; round-1 F1 / F3)"},
    {"op": "set_param", "block": NOISE, "param": "amp", "value": "0.025", "breaks": SPEC["noise"],
     "note": "near miss: noise amplitude 25 % too high (power x1.56); subsumes the round-1 probes 0.05-0.08"},
    {"op": "set_param", "block": NOISE, "param": "amp", "value": "0.016", "breaks": SPEC["noise"],
     "note": "near miss: noise amplitude 20 % too low (power x0.64)"},
]

DISJOINTNESS = (
    "Re-plans a polyphase channelizer (3 -> 4 channels, new output and File Sink) in a test-split seed. Nearest "
    "training-side specs (reports/reference/training_task_specs.json): t10_param_tuning retunes the centre frequency "
    "and bandwidth of a different receiver graph (playground/new_dmeo.grc); t12_fivegnalizer_modify lightens the FFT "
    "load of a 5G graph; t06_spectrogram_display builds a waterfall view; GRC-Agent scenario 22_fm_rx_filter_squelch "
    "adds a low-pass filter and squelch to an FM receiver. None uses a channelizer or splits a band into channels."
)

NOTES = (
    "Checks: spectrum.dat (48 kHz, first 4096 items) pass-to-pass: strongest tone 1.2 kHz, power share of each tone "
    "within +-480 Hz (Hann window), total RMS, and the noise share in the tone-free band 3-11 kHz (Hann window) within "
    "30 % (the noise source is deterministic; seed-to-seed spread 5.1 %). ch0-ch3 (12 kHz, items 1024-4095 after the "
    "channel filter transient, i.e. prototype filters up to about 4000 taps; the 4363-tap variant "
    "alt_pfb_lp2_tw50_4363taps still passes) "
    "fail-to-pass: tone offset "
    "(within 1 %, measured at 12 kHz, so each channel's rate and centre are checked), RMS (within 5 %) and the noise "
    "near the centre (power share in a tone-free 1.8 kHz band within 2 kHz of the centre at least half the original "
    "noise's; synthetic tones written to the channel files fail it, probes s1 / s2 in build.py --probes) per channel. "
    "The tone frequencies are absolute, so a changed samp_rate moves them relative to the rate (mutant). Variants: "
    "v1 C++ Polyphase Channelizer with low_pass_2 taps; v2 four frequency-translating FIR filters; v3 the hierarchical "
    "channelizer with nchans in a variable and other filter settings; alt_* the round-1 reviewers' alternatives "
    "ported to this start graph (other hierarchical-channelizer settings and user taps, sharp and very sharp "
    "prototype filters of 1091 and 2181 taps, plus a 4363-tap one, Polyphase Decimators with and without FFT, "
    "FFT-based translating filters, "
    "Rotator + FFT filter, Rotator + FIR filter, the Throttle moved behind the Add). A near-brick-wall hierarchical "
    "setting (bw 0.495, tb 0.005) fails at V5 inside GNU Radio's remez design ('insufficient extremals'), so it is "
    "not a valid answer."
)

REVIEW_NOTES = (
    "Known oracle limits: (1) 'filtered out' is checked through each channel's RMS (within 5 % of its own tone) and "
    "peak, as the instruction spells out: a neighbouring tone that leaks in with up to about 10 % of the channel tone's "
    "power (about -10 dB relative to it) would pass. Measured: prototype taps firdes.low_pass(1, samp_rate, 9500, 2000) "
    "(stopband from 10.5 kHz; the 13.92 kHz tone leaks into ch2 at -7 dB) fails, low_pass(1, samp_rate, 9000, 2000) "
    "and low_pass(1, samp_rate, 6000, 8000) (stopband from 10 kHz; everything outside the ch2 tone, noise included, "
    "at -20 / -18 dB relative to it) pass. "
    "(2) Gaussian and uniform noise of the same power cannot be told apart (the noise check is a power share within "
    "30 %; uniform noise with the same amplitude has a third of the power and fails). (3) 'Keep producing samples "
    "until stopped' is checked only by V5 liveness (4096 items per "
    "observation point). (4) Discrimination (round-1 item-quality F4, team decision): the instruction now names the "
    "channels by frequency only, so the model must know that output 3 of a 4-channel polyphase channelizer is the "
    "-12 kHz channel; the file assignment still follows the channelizer's own output order, so the reference edit is "
    "still small (nchans 4, one File Sink, one connection). A non-native order (e.g. ch0.dat = -12 kHz) would need "
    "rewiring or outchans and is left to the team. Licence: derived from the GNU Radio 3.10.12 example "
    "filter/polyphase_channelizer_demo.grc (GNU Radio project, GPL-3.0-or-later); the item keeps that attribution "
    "(licence decision is the team's)."
)


def main() -> int:
    split_rec = read_json(SPLITS / "flowgraph_seed_split.json")["seeds"][SEED]
    assert split_rec["split"] == SPLIT, f"seed split is {split_rec['split']}, task split is {SPLIT}"
    sha = sha256_file(GNURADIO_EXAMPLES / SEED)
    assert sha == split_rec["sha256"], "example file differs from the inventoried seed"

    start_full = build_start()
    start = tidy(start_full)
    ref = tidy(build_reference(start))
    if "--probes" in sys.argv:
        return run_probes(ref)
    variants = build_variants(ref)
    for op in MUTATIONS:
        edit.apply(ref, op)                                  # every mutation applies to the reference
    oracle = build_oracle()
    props = {c["requires"] for c in oracle["checks"]}
    assert props == set(SPEC.values()) - {SPEC["run"], SPEC["obs"]}, "every stated property is checked"

    points = [{"name": "spectrum", "sink": {"block_id": "blocks_file_sink",
                                            "params": {"file": "spectrum.dat", "type": "complex"}}}]
    points += [{"name": f"ch{k}", "sink": {"block_id": "blocks_file_sink",
                                           "params": {"file": f"ch{k}.dat", "type": "complex"}}}
               for k in range(N_NEW)]
    record = {
        "id": TASK_ID, "tier": "F", "split": SPLIT, "version": 3,
        "license": f"GPL-3.0-or-later (derived from GNU Radio 3.10.12 example {SEED})",
        "source": {"kind": "gnuradio_example", "ref": SEED, "sha256": sha},
        "cluster_id": SEED,
        "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring",
                       "generator_version": "0.2.0"},
        "review": {"status": "pending-team-review", "reviewers": [], "notes": REVIEW_NOTES},
        "family": "modify", "level": "L2",
        "instruction": INSTRUCTION,
        "start_grc": "start.grc", "reference_grc": "reference.grc",
        "observation": {
            "points": points,
            "run": {"n_items": N_ITEMS, "min_items": N_ITEMS, "timeout_s": 10},
            "contract": ("the complex samples written by the File Sinks to spectrum.dat (the 48 kHz sum) and to "
                         "ch0.dat, ch1.dat, ch2.dat and ch3.dat (the channels centred on 0 Hz, +12 kHz, 24 kHz and "
                         "-12 kHz, 12 kHz each)"),
        },
        "oracle": oracle,
        "mutations": MUTATIONS,
        "spec_properties": list(SPEC.values()),
        "perturbation": PERTURBATION,
        "disjointness": DISJOINTNESS,
        "notes": NOTES,
    }
    if (HERE / "variants").exists():
        shutil.rmtree(HERE / "variants")
    files = {"start.grc": dump(start), "reference.grc": dump(ref)}
    files.update({f"variants/{k}.grc": dump(v) for k, v in variants.items()})
    errors = taskio.save(HERE, record, files)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"wrote {HERE.relative_to(REPO)} ({len(files)} flowgraphs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
