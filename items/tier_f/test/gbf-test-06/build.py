"""Authoring script for GRC-Bench Tier-F item gbf-test-06 (family parameter-tune, level L3, split test).

Seed: GNU Radio 3.10.12 example analog/sig_source_msg_ports.grc (a test-split seed in
splits/flowgraph_seed_split.json; GPL-3.0-or-later, Copyright 2020 Free Software Foundation, author
Marcus Mueller). A complex Signal Source feeds a Throttle and a sink; a Message Strobe periodically
replaces the message of a Message Strobe Random-Delay, which sends it to the Signal Source's `cmd`
message port. So the source plays its initial tone (with random phase resets), and once the first
periodic message has gone through, the source is retuned to the frequency/amplitude in that message.

Task (parameter-tune, round-1 fix F4 / plan R6): the user wants the retune to go to a different
frequency and amplitude and says "change parameter values only". The model has to find that the
retune target lives in the Message Strobe's PMT dict (not in the Signal Source) and change it without
disturbing the initial tone, the timing, the rate or the output. Level L3 (round-1 F2; proposal 5.2
lists message ports under L3); the level is pending team confirmation.

Start graph = the seed exported by `./gb tier-f seed` with its numbers perturbed (PERTURBATION) and
the seed's copyright/author notice kept in the options block (round-1 compliance F4).
Observation: the complex samples written by the File Sink qtgui_time_sink_x_0_0 (the exported time
sink) to qtgui_time_sink_x_0_0.dat.

Timing (round-1 F1 / compliance F1 and F3). The Message Strobe first fires 0.5 s (wall clock) after
start and the Random-Delay strobe forwards it at its next tick (Gaussian, 100 +- 20 ms). The source
runs ahead of the Throttle by its output buffer, so in the captured stream the switch lands on a
buffer boundary at or after 0.5 s: measured 0.6826 s in 24/24 runs at 8-way concurrency (18 Oct,
this build), 0.597-0.768 s over 40 runs at 8-12-way concurrency in the round-1 review. It cannot
appear before about 0.5 s, because the strobe has not fired yet. The instruction states the window
the switch must fall into (after 0.45 s, before 0.95 s of the recording) and the oracle checks it with
two windows that each have to hold a single pure tone:
    pre   0.00-0.45 s  1.5 kHz, amplitude 0.8   (p2p; any 4.2 kHz power in it fails the purity check)
    post  0.95-1.45 s  4.2 kHz, amplitude 1.2   (f2p; any 1.5 kHz power in it fails the purity check)
Margins: >= 0.05 s before the earliest possible switch (0.5 s) and >= 0.18 s after the latest observed
one (0.768 s). Timing mutants that leave this window: Message Strobe period 1000 ms (the unperturbed
seed value; switch at 1.19 s) and 100 ms (switch at 0.26-0.34 s). Timing changes that stay inside the
stated window (e.g. Random-Delay mean 400 ms, switch at 0.85-0.94 s) satisfy the stated spec and are
not mutants.

Regenerate (WSL, from the grc-bench folder; deterministic):
    PYTHONPATH=src ~/grc-agent/.venv/bin/python items/tier_f/test/gbf-test-06/build.py
then run the gates:
    ./gb tier-f gates items/tier_f/test/gbf-test-06
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from grcbench.common.io import read_json, sha256_file          # noqa: E402
from grcbench.common.paths import GNURADIO_EXAMPLES, SPLITS     # noqa: E402
from grcbench.tier_f import grc, grcfile, taskio                # noqa: E402
from grcbench.tier_f.seedtools import headless_seed             # noqa: E402

TASK_ID = "gbf-test-06"
SPLIT = "test"
SEED = "analog/sig_source_msg_ports.grc"
SEED_AUTHOR = "Marcus Müller"
SEED_COPYRIGHT = "2020 Free Software Foundation"

FS = 48000                      # perturbed sample rate (Hz)
F0, A0 = 1500, 0.8              # initial tone (perturbed from 1000 Hz / 1)
F_OLD, A_OLD = 3000, 0.5        # retune target in the start graph (perturbed from 2000 Hz / 1.5)
F_NEW, A_NEW = 4200, 1.2        # retune target the user asks for
PERIOD_MS = 500                 # Message Strobe period (perturbed from 1000 ms)
MEAN_MS, STD_MS = 100, 20       # random-delay strobe, Gaussian (perturbed from 200 / 40 ms)

SINK = "qtgui_time_sink_x_0_0"  # File Sink created by the seed tool from the Qt time sink
OUT_FILE = f"{SINK}.dat"
SRC, STROBE, RSTROBE, THR = ("analog_sig_source_x_0", "blocks_message_strobe_0",
                             "blocks_message_strobe_random_0", "blocks_throttle2_0")

# ---- windows and tolerances (all stated in the instruction) -------------------------------------
PRE_END_S = 0.45                 # the switch must appear after this time (s) ...
POST_START_S = 0.95              # ... and before this time (s); the new tone holds from here on
POST_LEN_S = 0.50
PRE_N = int(PRE_END_S * FS)      # 21600 items
POST_SKIP = int(POST_START_S * FS)   # 45600 items
POST_N = int(POST_LEN_S * FS)        # 24000 items
N_ITEMS = POST_SKIP + POST_N     # 69600 items = 1.45 s at 48 kS/s
FREQ_REL = 0.01                  # frequencies within 1 %
AMP_REL = 0.05                   # amplitudes (RMS) within 5 %
PURITY = 0.99                    # at least 99 % of the power ...
PURITY_HW = 50                   # ... within 50 Hz of the tone


def cmd(freq, ampl) -> str:
    return f'pmt.to_pmt({{"freq": {freq}, "ampl": {ampl}}})'


# --------------------------------------------------------------------------- flowgraph helpers
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


# --------------------------------------------------------------------------- start / reference
def build_start() -> dict:
    seed, sinks = headless_seed(SEED)
    assert [s["file"] for s in sinks] == [OUT_FILE], sinks
    d = copy.deepcopy(seed)
    d["options"]["parameters"] = {"id": "sig_source_msg_ports",
                                  "title": "Demonstration: Signal Source Message Ports",
                                  "author": SEED_AUTHOR, "copyright": SEED_COPYRIGHT,
                                  "generate_options": "no_gui", "output_language": "python"}
    d = setp(d, "samp_rate", value="48e3")
    d = setp(d, SRC, freq=F0, amp=A0)
    d = setp(d, STROBE, period=PERIOD_MS, msg=cmd(F_OLD, A_OLD),
             comment="sets the amplitude and frequency every half second")
    d = setp(d, RSTROBE, mean=MEAN_MS, std=STD_MS, comment="resets the phase at random points in time")
    return tidy(d)


PERTURBATION = {"perturbed": True, "changes": [
    {"what": "sample rate (parameter samp_rate)", "original": "100e3", "new": "48e3"},
    {"what": "Signal Source initial frequency (Hz)", "original": "1000", "new": str(F0)},
    {"what": "Signal Source initial amplitude", "original": "1", "new": str(A0)},
    {"what": "Message Strobe period (ms)", "original": "1000", "new": str(PERIOD_MS)},
    {"what": "Message Strobe message (retune target)", "original": 'pmt.to_pmt({"freq": 2000, "ampl":1.5})',
     "new": cmd(F_OLD, A_OLD)},
    {"what": "Message Strobe Random-Delay mean (ms)", "original": "200", "new": str(MEAN_MS)},
    {"what": "Message Strobe Random-Delay std. deviation (ms)", "original": "40", "new": str(STD_MS)},
    {"what": "Message Strobe comment", "original": "sets the amplitude and frequency every second",
     "new": "sets the amplitude and frequency every half second"},
    {"what": "Qt GUI time sink", "original": "qtgui_time_sink_x (GUI)",
     "new": f"File Sink {SINK} writing {OUT_FILE} (complex; seed tool, observation point)"},
    {"what": "options: flowgraph id", "original": "demonstrate_signal_source_message_ports",
     "new": "sig_source_msg_ports (seed tool: the example's file stem)"},
    {"what": "options: generate_options", "original": "qt_gui", "new": "no_gui (headless conversion)"},
    {"what": "options: GUI/build settings (window_size, qt_qss_theme, run_options, placement, sizing_mode, "
             "category, gen_cmake, gen_linking, catch_exceptions, ...)", "original": "seed values",
     "new": "dropped (GRC defaults; no effect on a headless run)"},
    {"what": "options: author and copyright notice", "original": f"{SEED_AUTHOR} / {SEED_COPYRIGHT}",
     "new": "kept unchanged in every released .grc file (GPL notice)"},
    {"what": "GUI coordinates and default-valued parameters", "original": "present", "new": "dropped (compact form)"},
    {"what": "modification record", "original": "-",
     "new": "converted and perturbed for GRC-Bench by Claude (Anthropic), October 2026"},
]}


def build_variants(ref: dict) -> dict[str, dict]:
    """Every variant changes parameter values only (no block, variable or connection added/removed)."""
    v = {}
    # 1) the same command built with the PMT dictionary API instead of pmt.to_pmt on a Python dict
    v["v1_pmt_dict_api"] = setp(ref, STROBE, msg=(
        "pmt.dict_add(pmt.dict_add(pmt.make_dict(), pmt.intern('freq'), pmt.from_double(4200.0)), "
        "pmt.intern('ampl'), pmt.from_double(1.2))"))
    # 2) single-quoted keys in the other order, float literals (round-1 reviewer's alternative)
    v["v2_quoted_float_keys"] = setp(ref, STROBE, msg="pmt.to_pmt({'ampl': 1.2, 'freq': 4.2e3})")
    # 3) target written relative to the sample rate parameter, File Sink renamed with a ./ path
    #    (round-1 reviewer's alternative: rename + relative path)
    d = setp(ref, STROBE, msg='pmt.to_pmt({"freq": samp_rate * 0.0875, "ampl": 6.0 / 5})')
    d = setp(d, SINK, file=f"./{OUT_FILE}")
    blk(d, SINK)["name"] = "tone_file"
    d["connections"] = [[("tone_file" if x == SINK else x) for x in c] for c in d["connections"]]
    v["v3_expression_renamed_sink"] = d
    return {k: tidy(x) for k, x in v.items()}


# --------------------------------------------------------------------------- task record
INSTRUCTION = (
    "This flowgraph runs at a sample rate of 48 kHz and writes a complex tone to a File Sink that "
    f"stores complex samples in the file {OUT_FILE}. When the flowgraph starts, the tone is at 1.5 kHz "
    "with amplitude 0.8. Half a second after the start it is switched to 3 kHz with amplitude 0.5 and "
    "stays there; in the file, the switch shows up about 0.7 s into the recording (time = sample index "
    "/ 48 kHz). Make that switch go to 4.2 kHz with amplitude 1.2 instead. Change parameter values "
    "only: do not add, remove or rewire blocks, and do not add or remove variables. Keep the 48 kHz "
    f"sample rate, the timing of the switch and the complex output to {OUT_FILE}. Measured in the file: "
    "the first 0.45 s must still hold the 1.5 kHz tone with amplitude 0.8, and the 0.5 s starting at "
    "0.95 s must hold the 4.2 kHz tone with amplitude 1.2; in each of these stretches the frequency must "
    "be within 1 %, the amplitude (RMS of the complex samples) within 5 %, and at least 99 % of the "
    "power within 50 Hz of the tone. The flowgraph should keep producing samples until it is stopped; "
    "do not limit the number of samples."
)

SPEC = {
    "values_only": "parameter values only: no block or variable added or removed, no connection changed",
    "rate": "sample rate stays 48 kHz (frequencies are measured on the sample index at 48 kHz)",
    "f0": "first 0.45 s of the file: the initial tone is still at 1.5 kHz (within 1 %)",
    "a0": "first 0.45 s of the file: the initial tone amplitude is still 0.8 (RMS within 5 %)",
    "pure0": "first 0.45 s of the file: a single tone, >= 99 % of the power within 1450-1550 Hz",
    "timing": ("the switch appears after 0.45 s and before 0.95 s of the recording (checked through the "
               "purity of both windows: power of the other tone in either window fails it)"),
    "f1": "0.95-1.45 s of the file: the tone is at 4.2 kHz (within 1 %)",
    "a1": "0.95-1.45 s of the file: the tone amplitude is 1.2 (RMS within 5 %)",
    "pure1": "0.95-1.45 s of the file: a single tone, >= 99 % of the power within 4150-4250 Hz",
    "out": f"complex output written by the File Sink to {OUT_FILE} (observation point; sink type complex)",
    "run": (f"run condition: the flowgraph keeps producing samples (V5 liveness: at least {N_ITEMS} items, "
            "1.45 s at 48 kS/s, reach the observation point)"),
}

PRE = {"point": "out", "skip": 0, "n": PRE_N}
POST = {"point": "out", "skip": POST_SKIP, "n": POST_N}
ORACLE = {
    "checks": [
        # pass-to-pass: the initial tone (first 0.45 s) is unchanged and pure (no early switch)
        {**PRE, "feature": "peak_freq", "fs": FS, "expect": F0, "tol": FREQ_REL * F0, "group": "p2p",
         "requires": SPEC["f0"]},
        {**PRE, "feature": "rms", "expect": A0, "rel_tol": AMP_REL, "group": "p2p", "requires": SPEC["a0"]},
        {**PRE, "feature": "band_fraction", "fs": FS, "lo": F0 - PURITY_HW, "hi": F0 + PURITY_HW,
         "min": PURITY, "group": "p2p", "requires": SPEC["pure0"] + "; " + SPEC["timing"]},
        # fail-to-pass: 0.95-1.45 s holds the requested tone, pure (no late switch)
        {**POST, "feature": "peak_freq", "fs": FS, "expect": F_NEW, "tol": FREQ_REL * F_NEW, "group": "f2p",
         "requires": SPEC["f1"]},
        {**POST, "feature": "rms", "expect": A_NEW, "rel_tol": AMP_REL, "group": "f2p", "requires": SPEC["a1"]},
        {**POST, "feature": "band_fraction", "fs": FS, "lo": F_NEW - PURITY_HW, "hi": F_NEW + PURITY_HW,
         "min": PURITY, "group": "f2p", "requires": SPEC["pure1"] + "; " + SPEC["timing"]},
    ],
    "graph_checks": [
        {"kind": "same_blocks_as_start", "variables": "value_only", "ignore_ids": ["import", "note"],
         "group": "p2p", "property": "values_only",
         "note": "no block added, removed or bypassed; every variable keeps its name, none added or removed"},
        {"kind": "same_connections_as_start", "mode": "isomorphic", "group": "p2p", "property": "values_only",
         "note": "no rewiring (stream and message connections)"},
    ],
}

MUTATIONS = [
    {"op": "set_param", "block": STROBE, "param": "msg", "value": cmd(4260, A_NEW), "breaks": SPEC["f1"],
     "note": "near miss: 60 Hz (1.4 %) off"},
    {"op": "set_param", "block": STROBE, "param": "msg", "value": cmd(F_OLD, A_NEW), "breaks": SPEC["f1"],
     "note": "frequency of the old target kept"},
    {"op": "set_param", "block": STROBE, "param": "msg", "value": cmd(F_NEW, 1.1), "breaks": SPEC["a1"],
     "note": "near miss: amplitude 8 % low"},
    {"op": "set_param", "block": STROBE, "param": "msg", "value": 'pmt.to_pmt({"freq": 4200, "amp": 1.2})',
     "breaks": SPEC["a1"], "note": "wrong dict key: the cmd port knows 'ampl', so the amplitude stays 0.8"},
    {"op": "set_param", "block": STROBE, "param": "msg",
     "value": 'pmt.to_pmt({"freq": 4200, "ampl": 1.2, "offset": 0.3})', "breaks": SPEC["pure1"],
     "note": "a DC offset added to the new tone (RMS 1.24 still within 5 %)"},
    {"op": "set_param", "block": SRC, "param": "freq", "value": "1540", "breaks": SPEC["f0"],
     "note": "near miss: initial tone 40 Hz (2.7 %) off"},
    {"op": "set_param", "block": SRC, "param": "amp", "value": "0.9", "breaks": SPEC["a0"]},
    {"op": "set_param", "block": SRC, "param": "waveform", "value": "analog.GR_SQR_WAVE",
     "breaks": SPEC["pure0"], "note": "square waveform: peak and RMS checks alone would pass it"},
    {"op": "set_param", "block": STROBE, "param": "period", "value": "1000", "breaks": SPEC["timing"],
     "note": "the unperturbed seed value: the switch appears at about 1.19 s"},
    {"op": "set_param", "block": STROBE, "param": "period", "value": "100", "breaks": SPEC["timing"],
     "note": "the switch appears at about 0.26-0.34 s"},
    {"op": "set_param", "block": RSTROBE, "param": "msg", "value": cmd(F_NEW, A_NEW), "breaks": SPEC["timing"],
     "note": "switch at the first random tick (~0.1 s) instead of after 0.5 s"},
    {"op": "delete_connection", "src": RSTROBE, "dst": SRC, "breaks": SPEC["f1"],
     "note": "the command never reaches the Signal Source (also a rewiring)"},
    {"op": "bypass", "block": THR, "breaks": SPEC["timing"],
     "note": "without the throttle the whole capture is produced before the switch (also a removed block)"},
    {"op": "set_param", "block": "samp_rate", "param": "value", "value": "96e3", "breaks": SPEC["rate"]},
]

REVIEW_NOTES = (
    "Family parameter-tune (round-1 F4, plan R6): the edit is a value change toward a stated target; "
    "the instruction says 'change parameter values only' and graph checks enforce it "
    "(same_blocks_as_start with variables value_only, same_connections_as_start isomorphic). "
    "Level L3 (round-1 F2: proposal 5.2 lists message ports under L3) is pending team confirmation; if "
    "the team keeps L1, record it as a deviation (5 functional blocks, but an asynchronous message "
    "side-graph). The seed inventory's L1 estimate comes from seeds.py L3_HINTS not matching "
    "'message'/'strobe' (tooling T8, not part of this item). "
    "Known oracle limits: (1) the switch time is pinned to the stated 0.45-0.95 s window, not to the "
    "start graph's 0.6-0.8 s; timing edits that stay inside the window are accepted by design. (2) The "
    "random phase resets before the switch are part of the kept structure (no block may be removed) but "
    "their rate is not measured. (3) A model that drops showports='True' from the Signal Source hides "
    "the cmd port; GRC then omits the message connection without an error, so that answer fails at V6, "
    "not V2 (harness issue, round-1 compliance F6). "
    "Alternatives from round 1 that are now invalid by the stated rule: wiring the strobe straight to "
    "cmd (Random-Delay removed), adding a second message path, holding the retune dict or its values "
    "in new variables. "
    "Licence: derived from a GPL-3.0-or-later GNU Radio example; the copyright/author notice is kept in "
    "every .grc file and in task.json (source.attribution); the licence decision for seed-derived items "
    "is the team's. Seed-derived item at L1/L3 vs the approved plan's 'L1 from the composer' (round-1 "
    "compliance F5): team decision."
)


def main() -> int:
    split_rec = read_json(SPLITS / "flowgraph_seed_split.json")["seeds"][SEED]
    assert split_rec["split"] == SPLIT, f"seed split is {split_rec['split']}, task split is {SPLIT}"
    sha = sha256_file(GNURADIO_EXAMPLES / SEED)
    assert sha == split_rec["sha256"], "example file differs from the inventoried seed"

    start = build_start()
    ref = setp(start, STROBE, msg=cmd(F_NEW, A_NEW))
    variants = build_variants(ref)

    record = {
        "id": TASK_ID, "tier": "F", "split": SPLIT, "version": 2,
        "license": (f"GPL-3.0-or-later (derived from GNU Radio 3.10.12 example {SEED}, Copyright "
                    f"{SEED_COPYRIGHT}, author {SEED_AUTHOR}; converted and modified for GRC-Bench, Oct 2026)"),
        "source": {"kind": "gnuradio_example", "ref": SEED, "sha256": sha,
                   "attribution": (f"GNU Radio example {SEED}: Copyright {SEED_COPYRIGHT}; author "
                                   f"{SEED_AUTHOR}; GPL-3.0-or-later. Modified for GRC-Bench (headless "
                                   "conversion, perturbed numbers), October 2026.")},
        "cluster_id": SEED,
        "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring",
                       "generator_version": "0.2.0"},
        "review": {"status": "pending-team-review", "reviewers": [], "notes": REVIEW_NOTES},
        "family": "parameter-tune", "level": "L3",
        "instruction": INSTRUCTION,
        "start_grc": "start.grc", "reference_grc": "reference.grc",
        "observation": {
            "points": [{"name": "out", "sink": {"block_id": "blocks_file_sink",
                                                "params": {"file": OUT_FILE, "type": "complex"}}}],
            "run": {"n_items": N_ITEMS, "min_items": N_ITEMS, "timeout_s": 10},
            "contract": f"the complex samples written to the File Sink whose file name is {OUT_FILE}",
        },
        "oracle": ORACLE,
        "mutations": MUTATIONS,
        "spec_properties": list(SPEC.values()),
        "perturbation": PERTURBATION,
        "disjointness": ("Retunes the target of a message-port command (a PMT dict a Message Strobe sends, "
                         "through a Random-Delay strobe, to a Signal Source's cmd port) in a test-split seed, "
                         "with values only. Nearest training-side specs (reports/reference/"
                         "training_task_specs.json): t10_param_tuning retunes a receiver's centre frequency and "
                         "bandwidth through variables and renames them; GRC-Agent scenario 02_update_sample_rate "
                         "changes a samp_rate variable. Neither touches message strobes, PMT dicts or command "
                         "ports, and t22_epy_integrator (message-triggered reset) is an Embedded-Python build."),
        "notes": ("Oracle windows: initial tone 0-0.45 s (p2p), new tone 0.95-1.45 s (f2p); each must be "
                  ">= 99 % pure, which pins the switch into the stated 0.45-0.95 s window. Measured switch in "
                  "the captured stream: 0.6826 s in 24/24 runs at 8-way concurrency (18 Oct); 0.597-0.768 s "
                  "in the round-1 review (40 runs, 8-12-way). The strobe cannot fire before 0.5 s, so the "
                  "margin is >= 0.05 s before and >= 0.18 s after the observed range. Every variant changes "
                  "values only (PMT dict API; quoted float keys; an expression of samp_rate with the File "
                  "Sink renamed and a ./ path)."),
    }
    files = {"start.grc": dump(start), "reference.grc": dump(ref)}
    files.update({f"variants/{k}.grc": dump(v) for k, v in variants.items()})
    errors = taskio.save(HERE, record, files)
    if errors:
        print("\n".join(errors))
        return 1
    for stale in (HERE / "variants").glob("*.grc"):     # variants no longer produced by this script
        if f"variants/{stale.name}" not in files:
            stale.unlink()
    print(f"wrote {HERE} ({len(files)} flowgraphs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
