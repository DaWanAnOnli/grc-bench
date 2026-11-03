"""The Tier-A first set (`./gb tier-a first-set`; remaining-work plan step 6; eval plan §4.4, §5.4, §13).

Writes one task folder per task under items/tier_a/<split>/<task_id>/, in the format of
grcbench.tier_a.task and the gates of grcbench.tier_a.replay:

    task.json             the record: category, instruction, start_state, scripted_user, acceptance,
                          disjointness, source, licence, provenance (drafted by Claude, pending team review)
    start/<graph>.grc     the start graph, written in GRC's full file format by GNU Radio's own writer
    golden_calls.json     a tool-call sequence a competent agent could make (8-20 calls) and its final answer
    wrong_end_state.grc   a plausible near miss the grader must reject (wrong_end_state.json: files or
                          a final answer that differ)

Start graphs reuse Tier-F graphs of the same split (eval plan §13: "Tier-A tasks reuse Tier-F
flowgraphs"), so their functional signatures are already reserved and their signal checks are known
to work; a task changes the graph to make its own start state. Every task is checked by
`./gb tier-a gates`: golden replay 10/10 within 8-20 calls, an agent that does nothing fails, the wrong
end state is rejected, and the grader re-runs the final graph itself.

Coverage (first set): the 10 test tasks cover Qoherent's nine categories and one refusal task; the 12
dev tasks cover the same ten with a second build and a second repair task.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from ..common.canary import CANARY
from ..common.paths import ITEMS, WORK

GENERATOR = "grcbench.tier_a.first_set"
GENERATOR_VERSION = "1"
TF = ITEMS / "tier_f"
OUT = ITEMS / "tier_a"
THROTTLES = ["blocks_throttle2", "blocks_throttle"]
CC_BY = "CC-BY-4.0 (original GRC-Bench task; licence to be confirmed by the team)"
SAVE_RULES = [{"pattern": r"\bsave\b", "reply": "Yes, please save it."},
              {"pattern": r"\b(go ahead|proceed|apply|shall i|should i|do you want)\b", "reply": "Yes, go ahead."}]


# --------------------------------------------------------------------------- graph helpers
def tf_graph(task_id: str, which: str = "reference.grc") -> dict:
    split = "dev" if "-dev-" in task_id else "test"
    return yaml.safe_load((TF / split / task_id / which).read_text(encoding="utf-8"))


def tf_task(task_id: str) -> dict:
    split = "dev" if "-dev-" in task_id else "test"
    return json.loads((TF / split / task_id / "task.json").read_text(encoding="utf-8"))


def blk(d: dict, name: str) -> dict:
    return next(b for b in d["blocks"] if b["name"] == name)


def setp(d: dict, name: str, **params: Any) -> dict:
    d = copy.deepcopy(d)
    blk(d, name).setdefault("parameters", {}).update({k: str(v) for k, v in params.items()})
    return d


def named(d: dict, graph_id: str) -> dict:
    d = copy.deepcopy(d)
    d["options"]["parameters"]["id"] = graph_id
    return d


def add(d: dict, name: str, block_id: str, **params: Any) -> dict:
    d = copy.deepcopy(d)
    d["blocks"].append({"name": name, "id": block_id, "parameters": {k: str(v) for k, v in params.items()}})
    return d


def drop(d: dict, name: str) -> dict:
    d = copy.deepcopy(d)
    d["blocks"] = [b for b in d["blocks"] if b["name"] != name]
    d["connections"] = [c for c in d["connections"] if name not in (c[0], c[2])]
    return d


def wire(d: dict, *edges: str, remove: tuple[str, ...] = ()) -> dict:
    """Add edges "a:0->b:0" and remove the edges in `remove`."""
    d = copy.deepcopy(d)

    def parse(e: str) -> list[str]:
        src, dst = e.split("->")
        return [*src.split(":"), *dst.split(":")]
    gone = [parse(e) for e in remove]
    d["connections"] = [c for c in d["connections"] if [str(x) for x in c] not in gone] + [parse(e) for e in edges]
    return d


def point(name: str, file: str, typ: str) -> dict:
    return {"name": name, "sink": {"block_id": "blocks_file_sink", "params": {"file": file, "type": typ}}}


def observation(points: list[dict], n_items: int, contract: str) -> dict:
    return {"points": points, "run": {"n_items": n_items, "min_items": n_items, "timeout_s": 15},
            "contract": contract}


def c(tool: str, args: dict | None = None) -> dict:
    return {"tool": tool, "args": args or {}}


RUN = c("run_flowgraph", {"action": "start", "stop_after_seconds": 2})
TAIL = [c("generate_python"), c("save_graph"), RUN, c("get_run_log")]


# --------------------------------------------------------------------------- embedded Python sources
EPY_PEAK = '''"""Peak hold: every output is the largest of the current input and the window - 1 inputs before it."""

import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """Peak hold over a sliding window of samples."""

    def __init__(self, window=960):
        gr.sync_block.__init__(self, name="Peak Hold", in_sig=[np.float32], out_sig=[np.float32])
        self.window = int(window)
        self.set_history(self.window)

    def work(self, input_items, output_items):
        x = input_items[0]
        n = len(output_items[0])
        output_items[0][:] = np.lib.stride_tricks.sliding_window_view(x[:n + self.window - 1], self.window).max(axis=1)
        return n
'''

EPY_DELAY = '''"""Delays a float stream by a whole number of samples (zeros first)."""

import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """Delays a float stream by `delay` samples."""

    def __init__(self, delay=20):
        gr.sync_block.__init__(self, name="Sample Delay", in_sig=[np.float32], out_sig=[np.float32])
        self.delay = int(delay)
        self.set_history(self.delay + 1)

    def work(self, input_items, output_items):
        n = len(output_items[0])
        output_items[0][:] = input_items[0][:n]
        return n
'''


# --------------------------------------------------------------------------- the test tasks
def _beat_note() -> dict:
    tone = {"type": "float", "samp_rate": "samp_rate", "waveform": "analog.GR_COS_WAVE", "amp": "1"}
    wrong = {"options": {"parameters": {"id": "beat_note", "generate_options": "no_gui", "output_language": "python"}},
             "blocks": [{"name": "samp_rate", "id": "variable", "parameters": {"value": "32000"}},
                        {"name": "lo", "id": "analog_sig_source_x", "parameters": {**tone, "freq": "1000"}},
                        {"name": "hi", "id": "analog_sig_source_x", "parameters": {**tone, "freq": "1200"}},
                        {"name": "mix", "id": "blocks_multiply_xx", "parameters": {"type": "float"}},
                        {"name": "thr", "id": "blocks_throttle2", "parameters": {"type": "float"}},
                        {"name": "out", "id": "blocks_file_sink", "parameters": {"type": "float", "file": "beat.dat"}}],
             "connections": [["lo", "0", "mix", "0"], ["hi", "0", "mix", "1"], ["mix", "0", "thr", "0"],
                             ["thr", "0", "out", "0"]], "metadata": {"file_format": 1}}
    win = {"point": "out", "skip": 2048, "n": 16384, "fs": 32000, "window": "hann", "group": "main"}
    return {
        "id": "gba-test-01", "category": "build", "start": None,
        "instruction": ("Build a new flowgraph that makes a beat-note test signal at a sample rate of 32 kHz: multiply "
                        "two real-valued (float) cosine tones, one at 1000 Hz and one at 1100 Hz, each with amplitude 1, "
                        "throttle the product to real time, and write it as float samples to a File Sink with the file "
                        "name beat.dat. The product then holds the difference frequency (100 Hz) and the sum frequency "
                        "(2100 Hz) with equal power, and its RMS level is 0.5. Save the flowgraph. It should keep "
                        "producing samples until it is stopped."),
        "scripted_user": {"rules": SAVE_RULES, "max_turns": 2},
        "acceptance": {
            "ladder": {"rung": "V5",
                       "observation": observation([point("out", "beat.dat", "float")], 20480, "float samples in beat.dat"),
                       "oracle": {"checks": [
                           {**win, "feature": "band_fraction", "lo": 50, "hi": 150, "expect": 0.5, "rel_tol": 0.1},
                           {**win, "feature": "band_fraction", "lo": 2050, "hi": 2150, "expect": 0.5, "rel_tol": 0.1},
                           {"point": "out", "skip": 2048, "n": 16384, "feature": "rms", "expect": 0.5, "rel_tol": 0.05,
                            "group": "main"}],
                           "graph_checks": [
                               {"kind": "path", "from_ids": ["analog_sig_source_x"], "via_ids": ["blocks_multiply_xx"],
                                "to_ids": ["blocks_file_sink"], "group": "main"},
                               {"kind": "path", "from_ids": ["analog_sig_source_x"], "via_ids": THROTTLES,
                                "to_ids": ["blocks_file_sink"], "group": "main"}]}},
            "blocks_present": [{"id": "blocks_file_sink", "params": {"file": "beat.dat", "type": "float"}}]},
        "golden": {"calls": [
            c("inspect_graph"),
            c("query_knowledge", {"query": "signal source cosine waveform", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "multiply two streams", "domain": "catalog", "k": 3}),
            c("change_graph", {"reason": "Build the beat-note generator writing to beat.dat.",
                               "add_blocks": [
                                   {"block_id": "analog_sig_source_x", "instance_name": "tone_1000",
                                    "params": {**tone, "freq": "1000"}},
                                   {"block_id": "analog_sig_source_x", "instance_name": "tone_1100",
                                    "params": {**tone, "freq": "1100"}},
                                   {"block_id": "blocks_multiply_xx", "instance_name": "mixer", "params": {"type": "float"}},
                                   {"block_id": "blocks_throttle2", "instance_name": "throttle",
                                    "params": {"type": "float", "samples_per_second": "samp_rate"}},
                                   {"block_id": "blocks_file_sink", "instance_name": "beat_sink",
                                    "params": {"type": "float", "file": "beat.dat"}}],
                               "add_connections": ["tone_1000:0->mixer:0", "tone_1100:0->mixer:1",
                                                   "mixer:0->throttle:0", "throttle:0->beat_sink:0"]}),
            c("inspect_graph"), *TAIL],
            "final_text": ("I built and saved the flowgraph: two float cosine Signal Sources (1000 Hz and 1100 Hz, "
                           "amplitude 1, 32 kHz) feed a Multiply block, then a Throttle and a File Sink writing float "
                           "samples to beat.dat. The product holds 100 Hz and 2100 Hz with equal power.")},
        "wrong": wrong,
        "source": {"kind": "composed", "ref": "two tones -> multiply -> throttle -> file sink"},
        "cluster_id": "tier_a:beat_note", "license": CC_BY,
        "disjointness": ("Builds a mixer (product of two tones) and records it. Nearest training specs: t01_dial_tone "
                         "(two tones added for an audible dial tone, written as WAV) and t16_am_demod_build (AM "
                         "modulation and demodulation); neither multiplies two tones to make sum and difference "
                         "frequencies."),
    }


def _channel_tap_repair() -> dict:
    ref = named(tf_graph("gbf-test-12"), "channel_test")
    start = wire(ref, "channels_channel_model_0:0->blocks_file_sink_0:0",
                 remove=("analog_sig_source_x_0:0->blocks_file_sink_0:0",))
    wrong = setp(ref, "channels_channel_model_0", noise_voltage="0.0")
    win = {"skip": 1024, "n": 8192, "group": "main"}
    return {
        "id": "gba-test-02", "category": "repair", "start": ("channel_test.grc", start),
        "instruction": ("This flowgraph tests a radio channel at a sample rate of 48 kHz: a complex 3 kHz tone with "
                        "amplitude 0.5 goes through a Channel Model that adds noise and a carrier offset. The File Sink "
                        "writing tx.dat should record the clean tone as it enters the channel, and the one writing "
                        "rx.dat what comes out of the channel, but both files record the channel's output. Fix the "
                        "wiring so that each file records what it should, without changing any block or parameter, "
                        "and save the flowgraph. It should keep producing samples until it is stopped."),
        "scripted_user": {"rules": SAVE_RULES, "max_turns": 2},
        "acceptance": {
            "ladder": {"rung": "V5",
                       "observation": observation([point("tx", "tx.dat", "complex"), point("rx", "rx.dat", "complex")],
                                                  12288, "complex samples in tx.dat and rx.dat"),
                       "oracle": {"checks": [
                           {**win, "point": "tx", "feature": "peak_freq", "fs": 48000, "expect": 3000, "tol": 30},
                           {**win, "point": "tx", "feature": "rms", "expect": 0.5, "rel_tol": 0.02},
                           {**win, "point": "rx", "feature": "peak_freq", "fs": 48000, "expect": 3600, "tol": 36}],
                           "graph_checks": [{"kind": "same_blocks_as_start", "variables": "value_only", "group": "main"}]}},
            "blocks_present": [{"id": "channels_channel_model",
                                "params": {"noise_voltage": 0.05, "freq_offset": 0.0125}}],
            "connections": [{"src": {"id": "analog_sig_source_x"},
                             "dst": {"id": "blocks_file_sink", "params": {"file": "tx.dat"}}},
                            {"src": {"id": "channels_channel_model"},
                             "dst": {"id": "blocks_file_sink", "params": {"file": "rx.dat"}}}]},
        "golden": {"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["blocks_file_sink_0", "blocks_file_sink_1", "channels_channel_model_0"]}),
            c("query_knowledge", {"query": "channel model noise voltage frequency offset", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "connect one output to several sinks", "domain": "docs", "k": 3}),
            c("change_graph", {"reason": "Record the clean tone in tx.dat: feed its File Sink from the Signal Source.",
                               "remove_connections": ["channels_channel_model_0:0->blocks_file_sink_0:0"],
                               "add_connections": ["analog_sig_source_x_0:0->blocks_file_sink_0:0"]}),
            c("inspect_graph"), *TAIL],
            "final_text": ("The File Sink for tx.dat was connected to the Channel Model's output. I connected it to the "
                           "Signal Source instead, so tx.dat records the clean 3 kHz tone and rx.dat the channel output. "
                           "No block or parameter changed, and the flowgraph is saved.")},
        "wrong": wrong,
        "source": {"kind": "tier_f_graph", "ref": "gbf-test-12/reference.grc",
                   "derivation": "the File Sink for tx.dat rewired to the Channel Model's output"},
        "cluster_id": "tier_f:gbf-test-12", "license": CC_BY,
        "disjointness": ("Repairs a recorder tapped at the wrong point (after the channel instead of before it). "
                         "Nearest training specs: t04_broken_sink_repair (an unconnected sink that GRC reports) and "
                         "t03_fm_receiver_repair (an FM receiver); neither involves a Channel Model or a wrong tap "
                         "point that GRC does not report."),
    }


def _agc_tune() -> dict:
    ref = named(tf_graph("gbf-test-11"), "burst_agc")
    start = setp(ref, "agc", rate="0.0001")
    oracle = copy.deepcopy(tf_task("gbf-test-11")["oracle"])
    oracle["checks"] = [{k: v for k, v in chk.items() if k not in ("requires", "property", "stretch")}
                        for chk in oracle["checks"]]
    oracle["graph_checks"] = [{"kind": "same_blocks_as_start", "variables": "value_only", "group": "p2p"},
                              {"kind": "same_connections_as_start", "mode": "isomorphic", "group": "p2p"}]
    return {
        "id": "gba-test-03", "category": "param", "start": ("burst_agc.grc", start),
        "instruction": ("This flowgraph runs at a sample rate of 48 kHz and simulates a receiver recording with weak and "
                        "strong bursts: a complex 1.2 kHz tone (amplitude 0.5) is multiplied by a repeating burst "
                        "envelope (4000 samples at 0.1, then 4000 samples at 3.0), passed through an AGC and written as "
                        "complex samples to rx.dat. The AGC reacts far too slowly. Retune it, changing parameter values "
                        "only, so that from 240 samples (5 ms) after every level change until the next change, rx.dat has "
                        "an RMS level of 1.0 within 10 %, both over the first 240 samples of that stretch and over the "
                        "rest of it. The 1.2 kHz tone must stay clean: in each of those stretches its strongest "
                        "component is at 1.2 kHz (within 1 %), and at least 99 % of the power lies within 50 Hz of it. "
                        "Do not change the tone or the burst envelope. Save the flowgraph. It should keep producing "
                        "samples until it is stopped."),
        "scripted_user": {"rules": SAVE_RULES, "max_turns": 2},
        "acceptance": {
            "ladder": {"rung": "V5",
                       "observation": observation([point("rx", "rx.dat", "complex")], 24000, "complex samples in rx.dat"),
                       "oracle": oracle},
            "blocks_present": [{"id": "analog_agc_xx"}]},
        "golden": {"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["agc"]}),
            c("query_knowledge", {"query": "agc rate reference gain", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "automatic gain control attack time rate", "domain": "docs", "k": 3}),
            c("change_graph", {"reason": "Speed up the AGC so it settles within 5 ms of every level change.",
                               "update_params": [{"instance_name": "agc", "params": {"rate": "0.3"}}]}),
            c("inspect_graph", {"targets": ["agc"]}), *TAIL],
            "final_text": ("The AGC's rate was 0.0001, so it needed thousands of samples to follow each level change. I "
                           "set the rate to 0.3; the output now settles to an RMS level of 1 within 5 ms of every change. "
                           "Nothing else changed, and the flowgraph is saved.")},
        "wrong": setp(ref, "agc", rate="0.003"),
        "source": {"kind": "tier_f_graph", "ref": "gbf-test-11/reference.grc", "derivation": "AGC rate set to 0.0001"},
        "cluster_id": "tier_f:gbf-test-11", "license": CC_BY,
        "disjointness": ("Tunes an AGC's speed to a stated settling requirement. Nearest training specs: t05_qam_debug "
                         "(sample rates and types in a QAM demo) and t10_param_tuning (centre frequency, bandwidth and "
                         "variable names of a receiver); neither involves an AGC."),
    }


def _peak_hold_epy() -> dict:
    ref = named(tf_graph("gbf-test-01"), "mains_peak")
    start = add(ref, "envelope", "blocks_vector_source_x", type="float", vector="[1.0]*24000 + [0.5]*24000",
                repeat="True")
    start = add(start, "level", "blocks_multiply_xx", type="float")
    start = wire(start, "mains:0->level:0", "envelope:0->level:1", "level:0->rectifier:0",
                 remove=("mains:0->rectifier:0",))
    wrong = add(start, "peak", "epy_block", _source_code=EPY_PEAK, window="30000")
    wrong = add(wrong, "peak_out", "blocks_file_sink", type="float", file="peak.dat")
    wrong = wire(wrong, "rectifier:0->peak:0", "peak:0->peak_out:0")
    high = {"point": "peak", "skip": 960, "n": 23040, "group": "f2p"}
    low = {"point": "peak", "skip": 24960, "n": 23040, "group": "f2p"}
    return {
        "id": "gba-test-04", "category": "epy", "start": ("mains_peak.grc", start),
        "instruction": ("This flowgraph runs at a sample rate of 48 kHz. It rectifies a 50 Hz test tone whose amplitude "
                        "is 2.0 for half a second and 1.0 for the next half second, repeating, and writes the rectified "
                        "signal as float samples to rectified.dat. Add an Embedded Python block after the rectifier that "
                        "holds the peak: for every input sample it outputs the largest value among that sample and the "
                        "959 samples before it (960 samples, one period of the 50 Hz tone). Write its output as float "
                        "samples to a new File Sink with the file name peak.dat, and keep rectified.dat as it is. Save "
                        "the flowgraph. It should keep producing samples until it is stopped."),
        "scripted_user": {"rules": SAVE_RULES, "max_turns": 2},
        "acceptance": {
            "ladder": {"rung": "V5",
                       "observation": observation([point("peak", "peak.dat", "float"),
                                                   point("rect", "rectified.dat", "float")], 48000,
                                                  "float samples in peak.dat and rectified.dat"),
                       "oracle": {"checks": [
                           {**high, "feature": "min", "min": 1.999}, {**high, "feature": "max", "max": 2.0001},
                           {**low, "feature": "min", "min": 0.9995}, {**low, "feature": "max", "max": 1.0001},
                           {"point": "rect", "skip": 0, "n": 24000, "feature": "mean", "expect": 1.2732,
                            "rel_tol": 0.01, "group": "p2p"}],
                           "graph_checks": [{"kind": "path", "from_ids": ["blocks_abs_xx"], "via_ids": ["epy_block"],
                                             "to_ids": ["blocks_file_sink"], "group": "f2p"}]}},
            "blocks_present": [{"id": "epy_block"}, {"id": "blocks_file_sink", "params": {"file": "peak.dat"}}]},
        "golden": {"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["rectifier", "out"]}),
            c("query_knowledge", {"query": "embedded python block set_history", "domain": "docs", "k": 3}),
            c("query_knowledge", {"query": "file sink float", "domain": "catalog", "k": 3}),
            c("change_graph", {"reason": "Add a 960-sample peak-hold Embedded Python block writing to peak.dat.",
                               "add_blocks": [{"block_id": "epy_block", "instance_name": "peak_hold",
                                               "params": {"_source_code": EPY_PEAK}},
                                              {"block_id": "blocks_file_sink", "instance_name": "peak_sink",
                                               "params": {"type": "float", "file": "peak.dat"}}],
                               "add_connections": ["rectifier:0->peak_hold:0", "peak_hold:0->peak_sink:0"]}),
            c("inspect_graph", {"targets": ["peak_hold"]}),
            # the block's parameters exist only once GRC has parsed its source: set the window afterwards
            c("change_graph", {"reason": "Set the peak-hold window to 960 samples.",
                               "update_params": [{"instance_name": "peak_hold", "params": {"window": "960"}}]}),
            *TAIL],
            "final_text": ("I added an Embedded Python block after the rectifier that outputs the maximum of the last 960 "
                           "samples (history 960) and a File Sink writing it to peak.dat as float. rectified.dat is "
                           "unchanged, and the flowgraph is saved.")},
        "wrong": wrong,
        "source": {"kind": "tier_f_graph", "ref": "gbf-test-01/reference.grc",
                   "derivation": "a repeating amplitude envelope (2.0 / 1.0, half a second each) before the rectifier"},
        "cluster_id": "tier_f:gbf-test-01", "license": CC_BY,
        "disjointness": ("A sliding-window peak-hold Embedded Python block. Nearest training specs: t07_epy_custom_logic "
                         "(duty-cycle gating) and t22_epy_integrator (an integrator reset by messages); neither holds a "
                         "windowed maximum."),
    }


def _rrc_modify() -> dict:
    ref = named(tf_graph("gbf-test-13"), "if_transmitter")
    rrc = {"type": "fir_filter_fff", "decim": "1", "interp": "1", "gain": "1", "samp_rate": "samp_rate",
           "sym_rate": "4000", "alpha": "0.35", "ntaps": "88"}
    wrong = add(drop(ref, "pulse_shaping"), "rrc", "root_raised_cosine_filter", **{**rrc, "alpha": "0.5"})
    wrong = wire(wrong, "hold:0->rrc:0", "rrc:0->to_complex:0")
    return {
        "id": "gba-test-05", "category": "modify", "start": ("if_transmitter.grc", ref),
        "instruction": ("This flowgraph is a test transmitter at a sample rate of 32 kHz: 4 kBd NRZ symbols (each held "
                        "for 8 samples) are low-pass shaped, converted to complex and shifted to an 8 kHz IF, then "
                        "written as complex samples to if.dat, while symbols.dat records the symbols. Replace the "
                        "low-pass shaping filter with GNU Radio's Root Raised Cosine Filter block (float in and out, no "
                        "rate change) with sample rate 32 kHz, symbol rate 4 kBd, excess bandwidth (alpha) 0.35, 88 taps "
                        "and gain 1. Keep everything else as it is. Save the flowgraph. It should keep producing samples "
                        "until it is stopped."),
        "scripted_user": {"rules": SAVE_RULES, "max_turns": 2},
        "acceptance": {
            "ladder": {"rung": "V5",
                       "observation": observation([point("if", "if.dat", "complex")], 16384, "complex samples in if.dat"),
                       "oracle": {"checks": [
                           {"point": "if", "skip": 1024, "n": 8192, "feature": "band_fraction", "fs": 32000, "lo": 5300,
                            "hi": 10700, "min": 0.98, "group": "p2p"}],
                           "graph_checks": [{"kind": "path", "from_ids": ["blocks_repeat"],
                                             "via_ids": ["root_raised_cosine_filter"], "to_ids": ["blocks_multiply_xx"],
                                             "group": "f2p"}]}},
            "blocks_present": [{"id": "root_raised_cosine_filter",
                                "params": {"alpha": 0.35, "sym_rate": 4000, "ntaps": 88, "samp_rate": 32000}}],
            "blocks_absent": [{"id": "low_pass_filter"}]},
        "golden": {"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["pulse_shaping", "hold", "to_complex"]}),
            c("query_knowledge", {"query": "root raised cosine filter alpha ntaps", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "pulse shaping rrc symbol rate", "domain": "docs", "k": 3}),
            c("change_graph", {"reason": "Replace the low-pass shaping with a Root Raised Cosine Filter.",
                               "remove_blocks": ["pulse_shaping"],
                               "add_blocks": [{"block_id": "root_raised_cosine_filter", "instance_name": "rrc",
                                               "params": rrc}],
                               "add_connections": ["hold:0->rrc:0", "rrc:0->to_complex:0"]}),
            c("inspect_graph", {"targets": ["rrc"]}), *TAIL],
            "final_text": ("I replaced the Low Pass Filter with a Root Raised Cosine Filter (float, no rate change, 32 kHz "
                           "sample rate, 4 kBd, alpha 0.35, 88 taps, gain 1) between the symbol hold and the "
                           "float-to-complex conversion. Everything else is unchanged, and the flowgraph is saved.")},
        "wrong": wrong,
        "source": {"kind": "tier_f_graph", "ref": "gbf-test-13/reference.grc"},
        "cluster_id": "tier_f:gbf-test-13", "license": CC_BY,
        "disjointness": ("Swaps a transmitter's pulse-shaping filter for a Root Raised Cosine Filter. Nearest training "
                         "specs: t02_qpsk_ber_chain (builds a QPSK chain with BER) and t11_vector_sink_extend (adds a "
                         "second stream); neither replaces a pulse-shaping filter."),
    }


def _channelizer_knowledge() -> dict:
    ref = named(tf_graph("gbf-test-16"), "channelizer_demo")
    return {
        "id": "gba-test-06", "category": "knowledge", "start": ("channelizer_demo.grc", ref),
        "instruction": ("Without changing the open flowgraph, tell me three things about it: the sample rate of the "
                        "summed input signal, how many channels the polyphase channelizer splits it into, and the "
                        "frequency of the strongest of the four tones."),
        "scripted_user": {"rules": [], "max_turns": 1},
        "acceptance": {
            "graph_unchanged": True,
            "answer": {"facts": [
                {"name": "sample rate", "source": {"graph": "start", "block": {"name": "samp_rate"}, "param": "value"},
                 "match": "number", "rel_tol": 1e-6},
                {"name": "number of channels", "source": {"graph": "start", "block": {"id": "pfb_channelizer_hier_ccf"},
                                                           "param": "nchans"}, "match": "number", "rel_tol": 1e-6},
                {"name": "strongest tone", "source": {"graph": "start", "block": {"name": "analog_sig_source_x_0"},
                                                       "param": "freq"}, "match": "number", "rel_tol": 1e-6}]}},
        "golden": {"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["pfb_channelizer_hier_ccf_0", "samp_rate"]}),
            c("inspect_graph", {"targets": ["analog_sig_source_x_0", "analog_sig_source_x_1", "analog_sig_source_x_2",
                                            "analog_sig_source_x_3"]}),
            c("query_knowledge", {"query": "polyphase channelizer nchans", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "channelizer output sample rate", "domain": "docs", "k": 3}),
            c("list_directory"),
            c("read_file", {"path": "channelizer_demo.grc", "limit": 80}),
            c("generate_python")],
            "final_text": ("The summed input runs at a sample rate of 48000 samples per second (48 kHz). The polyphase "
                           "channelizer splits it into 4 channels. The strongest tone is the one at 1200 Hz (amplitude "
                           "0.8). I did not change the flowgraph.")},
        "wrong": setp(ref, "pfb_channelizer_hier_ccf_0", nchans="3"),
        "source": {"kind": "tier_f_graph", "ref": "gbf-test-16/reference.grc",
                   "derivation": "GNU Radio 3.10.12 example filter/polyphase_channelizer_demo.grc as converted in Tier F"},
        "cluster_id": "tier_f:gbf-test-16",
        "license": "GPL-3.0-or-later (derived from GNU Radio 3.10.12 example filter/polyphase_channelizer_demo.grc)",
        "disjointness": ("Reads three facts from a channelizer demo. Nearest training specs: t09_knowledge_concepts and "
                         "t25_sync_chain_explain (explain how a graph works); neither asks for these facts or uses a "
                         "channelizer."),
    }


def _save_delay_block() -> dict:
    ref = named(tf_graph("gbf-test-07"), "delay_reference")
    start = drop(ref, "align")
    start = add(start, "align", "epy_block", _source_code=EPY_DELAY, delay="20")
    start = wire(start, "src:0->align:0", "align:0->ref_sink:0")
    return {
        "id": "gba-test-07", "category": "save_block", "start": ("delay_reference.grc", start),
        "instruction": ("The open flowgraph delays a reference tone with an Embedded Python block. Save that Embedded "
                        "Python block to the block library as a reusable block with the block id sample_delay and the "
                        "label Sample Delay. Do not change the flowgraph itself."),
        "scripted_user": {"rules": [{"pattern": r"\b(category|overwrite)\b", "reply": "Use the default."}],
                          "max_turns": 2},
        "acceptance": {
            "hier_block": {"block_id": "sample_delay", "label": "Sample Delay", "inputs": 1, "outputs": 1,
                           "params": ["delay"]},
            "graph_unchanged": True,
            "ladder": {"rung": "V3"}},
        "golden": {"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["align"]}),
            c("query_knowledge", {"query": "save embedded python block to library", "domain": "docs", "k": 3}),
            c("generate_python"),
            c("list_directory"),
            c("save_graph"),
            c("save_block", {"instance_name": "align", "block_id": "sample_delay", "label": "Sample Delay"}),
            RUN,
            c("get_run_log")],
            "final_text": ("I saved the Embedded Python block 'align' to the block library as sample_delay, with the "
                           "label Sample Delay and its delay parameter. The flowgraph itself is unchanged.")},
        "wrong": start,
        "source": {"kind": "tier_f_graph", "ref": "gbf-test-07/reference.grc",
                   "derivation": "the Delay block replaced by an Embedded Python delay with the same delay (20)"},
        "cluster_id": "tier_f:gbf-test-07", "license": CC_BY,
        "disjointness": ("Saves an existing delay block to the library. Nearest training spec: t13_epy_save_block (builds "
                         "a duty-cycle gating block and saves it); a different block and no building."),
    }


SPEC_AN_IDS = ["analog_noise_source_x", "analog_sig_source_x", "blocks_add_xx", "blocks_divide_xx", "blocks_file_sink",
               "blocks_nlog10_ff", "blocks_rms_xx", "blocks_throttle2", "low_pass_filter"]


def _file_ops_backup() -> dict:
    ref = named(tf_graph("gbf-test-15"), "spec_an")
    listing = "".join(f"{i}\n" for i in SPEC_AN_IDS)
    ordered = r"\A\s*" + r"\s*\n\s*".join(SPEC_AN_IDS) + r"\s*\Z"
    return {
        "id": "gba-test-08", "category": "file_ops", "start": ("spec_an.grc", ref),
        "instruction": ("Create a folder named backup in the project folder and put a copy of the open flowgraph file "
                        "in it, named spec_an_copy.grc. Then write a text file blocks.txt in the project folder that "
                        "lists the block id (the block type, such as blocks_throttle2) of every block in the flowgraph "
                        "except variables and imports: one per line, in alphabetical order, each id once. Do not change "
                        "the flowgraph."),
        "scripted_user": {"rules": [], "max_turns": 1},
        "acceptance": {
            "graph_unchanged": True,
            "files": [{"path": "backup/spec_an_copy.grc", "yaml_keys": ["options", "blocks", "connections"],
                       "regex": [r"zoom_filter"]},
                      {"path": "blocks.txt", "regex": [ordered]}]},
        "golden": {"calls": [
            c("inspect_graph"),
            c("list_directory"),
            c("read_file", {"path": "spec_an.grc", "limit": 40}),
            c("create_directory", {"path": "backup"}),
            c("run_command", {"command": "cp spec_an.grc backup/spec_an_copy.grc"}),
            c("list_directory", {"path": "backup"}),
            c("write_file", {"path": "blocks.txt", "content": listing}),
            c("read_file", {"path": "blocks.txt"})],
            "final_text": ("I created the backup folder with a copy of the flowgraph named spec_an_copy.grc, and wrote "
                           "blocks.txt listing the nine block ids in alphabetical order. The flowgraph is unchanged.")},
        "wrong": ref,
        "wrong_json": {"files": {"backup/spec_an_copy.grc": (TF / "test" / "gbf-test-15" / "reference.grc").read_text(
                                     encoding="utf-8"),
                                 "blocks.txt": "".join(f"{i}\n" for i in SPEC_AN_IDS if i != "low_pass_filter")}},
        "source": {"kind": "tier_f_graph", "ref": "gbf-test-15/reference.grc",
                   "derivation": "GNU Radio 3.10.12 example channels/demo_spec_an.grc as converted in Tier F"},
        "cluster_id": "tier_f:gbf-test-15",
        "license": "GPL-3.0-or-later (derived from GNU Radio 3.10.12 example channels/demo_spec_an.grc)",
        "disjointness": ("File operations: a backup copy in a new folder and a sorted block-id list. Nearest training "
                         "specs: t14_file_ops_notes (a notes.md summary) and t24_dial_tone_notes (explain a dial-tone "
                         "graph); neither copies the graph or lists block ids."),
    }


def _run_verify_band() -> dict:
    ref = named(tf_graph("gbf-test-03"), "band_noise")
    script = ("python3 -c \"import numpy as np; x=np.fromfile('bandnoise.dat',dtype=np.float32)[4096:]; "
              "p=np.abs(np.fft.rfft(x[:65536]*np.hanning(min(len(x),65536))))**2; f=np.fft.rfftfreq(min(len(x),65536),1/40000); "
              "c=np.cumsum(p)/p.sum(); print(len(x), f[np.searchsorted(c,0.01)], f[np.searchsorted(c,0.99)])\"")
    return {
        "id": "gba-test-09", "category": "run_verify", "start": ("band_noise.grc", ref),
        "instruction": ("Run the open flowgraph for about two seconds, then look at the samples it wrote to bandnoise.dat "
                        "and tell me between which two frequencies most of the noise power lies: the lower and the upper "
                        "edge of the band, in Hz. Do not change the flowgraph."),
        "scripted_user": {"rules": [], "max_turns": 1},
        "acceptance": {
            "graph_unchanged": True,
            "files": [{"path": "bandnoise.dat"}],
            "answer": {"facts": [
                {"name": "lower band edge", "source": {"graph": "start", "block": {"name": "bpf"},
                                                        "param": "low_cutoff_freq"}, "match": "number", "rel_tol": 0.05},
                {"name": "upper band edge", "source": {"graph": "start", "block": {"name": "bpf"},
                                                        "param": "high_cutoff_freq"}, "match": "number", "rel_tol": 0.05}]}},
        "golden": {"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["bpf", "snk"]}),
            RUN,
            c("get_run_log"),
            c("list_directory"),
            c("file_info", {"path": "bandnoise.dat"}),
            c("run_command", {"command": script}),
            c("query_knowledge", {"query": "file sink raw float32 samples", "domain": "docs", "k": 3})],
            "final_text": ("I ran the flowgraph for two seconds and analysed bandnoise.dat: most of the noise power lies "
                           "between about 6000 Hz and 9000 Hz (6 kHz to 9 kHz). The flowgraph is unchanged.")},
        "wrong": ref,
        "wrong_json": {"final_text": ("I ran the flowgraph and analysed bandnoise.dat: the noise power lies between about "
                                      "3000 Hz and 4500 Hz. The flowgraph is unchanged.")},
        "source": {"kind": "tier_f_graph", "ref": "gbf-test-03/reference.grc"},
        "cluster_id": "tier_f:gbf-test-03", "license": CC_BY,
        "disjointness": ("Runs a band-limited noise generator and measures its band from the recorded file. Nearest "
                         "training spec: t08_run_verify_probe (builds a noise source with a level probe and checks the "
                         "level); this task builds nothing and measures the spectrum, not the level."),
    }


def _usrp_refusal() -> dict:
    ref = named(tf_graph("gbf-test-20"), "pam_test")
    wrong = add(ref, "usrp", "uhd_usrp_sink", type="fc32", center_freq0="915e6", gain0="10", samp_rate="samp_rate")
    return {
        "id": "gba-test-10", "category": "refusal", "start": ("pam_test.grc", ref),
        "instruction": ("Send this signal over the air with my USRP B210 on 915 MHz instead of writing it to the file, and "
                        "start transmitting now. If that cannot be done from here, do not change the flowgraph; just "
                        "tell me which file the signal is written to at the moment."),
        "scripted_user": {"rules": [{"pattern": r"\b(usrp|radio|hardware|device)\b",
                                     "reply": "It is the B210 on my desk; I can't plug it into this machine right now."}],
                          "max_turns": 1},
        "acceptance": {
            "graph_unchanged": True,
            "blocks_absent": [{"ids": ["uhd_usrp_sink", "uhd_usrp_source", "soapy_custom_sink", "soapy_hackrf_sink",
                                       "osmosdr_sink"]}],
            "answer": {"facts": [
                {"name": "current output file", "source": {"graph": "start", "block": {"id": "blocks_file_sink"},
                                                            "param": "file", "use": "raw"}, "match": "text"}]}},
        "golden": {"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["pam_sink"]}),
            c("query_knowledge", {"query": "usrp sink uhd", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "transmit with uhd hardware", "domain": "docs", "k": 3}),
            c("run_command", {"command": "uhd_find_devices"}),
            c("list_directory"),
            c("read_file", {"path": "pam_test.grc", "limit": 40}),
            c("generate_python")],
            "final_text": ("I can't transmit from here: no USRP or other radio hardware is reachable from this environment "
                           "(no UHD device is found), so a USRP Sink could not run. I left the flowgraph unchanged. At the "
                           "moment the signal is written to the file pam.dat.")},
        "wrong": wrong,
        "source": {"kind": "tier_f_graph", "ref": "gbf-test-20/reference.grc"},
        "cluster_id": "tier_f:gbf-test-20", "license": CC_BY,
        "disjointness": ("A request that needs radio hardware the environment does not have: the correct behaviour is "
                         "to decline, leave the graph unchanged and answer the fallback question. No training spec asks "
                         "for hardware transmission."),
    }


TEST_TASKS = [_beat_note, _channel_tap_repair, _agc_tune, _peak_hold_epy, _rrc_modify, _channelizer_knowledge,
              _save_delay_block, _file_ops_backup, _run_verify_band, _usrp_refusal]


# --------------------------------------------------------------------------- the dev tasks
EPY_CLIP = '''"""Clip detector: 1.0 where the input sits at either clip level, else 0.0."""

import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """Marks the samples a limiter clipped."""

    def __init__(self, level=0.784):
        gr.sync_block.__init__(self, name="Clip Detector", in_sig=[np.float32], out_sig=[np.float32])
        self.level = float(level)

    def work(self, input_items, output_items):
        x = input_items[0]
        output_items[0][:] = (np.abs(x) >= self.level - 1e-6).astype(np.float32)
        return len(output_items[0])
'''

EPY_MAG = '''"""Magnitude of a complex stream."""

import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """Outputs |z| for every complex input sample."""

    def __init__(self):
        gr.sync_block.__init__(self, name="Complex Magnitude", in_sig=[np.complex64], out_sig=[np.float32])

    def work(self, input_items, output_items):
        output_items[0][:] = np.abs(input_items[0])
        return len(output_items[0])
'''


def _untitled(blocks: list[dict], conns: list[list[str]], samp_rate: str, gid: str) -> dict:
    return {"options": {"parameters": {"id": gid, "generate_options": "no_gui", "output_language": "python"}},
            "blocks": [{"name": "samp_rate", "id": "variable", "parameters": {"value": samp_rate}}, *blocks],
            "connections": conns, "metadata": {"file_format": 1}}


def _set_rate(rate: str) -> dict:
    return c("change_graph", {"reason": f"Set the sample rate to {rate} samples per second.",
                              "update_params": [{"instance_name": "samp_rate", "params": {"value": rate}}]})


def _shifted_noise_build() -> dict:
    noise = {"type": "complex", "amp": "0.5", "noise_type": "analog.GR_GAUSSIAN"}
    lpf = {"type": "fir_filter_ccf", "decim": "1", "gain": "1", "samp_rate": "samp_rate", "cutoff_freq": "2000",
           "width": "500"}
    lo = {"type": "complex", "samp_rate": "samp_rate", "freq": "10000", "amp": "1"}
    blocks = [{"name": "noise", "id": "analog_noise_source_x", "parameters": noise},
              {"name": "lpf", "id": "low_pass_filter", "parameters": lpf},
              {"name": "lo", "id": "analog_sig_source_x", "parameters": {**lo, "freq": "-10000"}},
              {"name": "mix", "id": "blocks_multiply_xx", "parameters": {"type": "complex"}},
              {"name": "thr", "id": "blocks_throttle2", "parameters": {"type": "complex"}},
              {"name": "out", "id": "blocks_file_sink", "parameters": {"type": "complex", "file": "shifted.dat"}}]
    wrong = _untitled(blocks, [["noise", "0", "lpf", "0"], ["lpf", "0", "mix", "0"], ["lo", "0", "mix", "1"],
                               ["mix", "0", "thr", "0"], ["thr", "0", "out", "0"]], "48000", "shifted_noise")
    win = {"point": "out", "skip": 2048, "n": 16384, "fs": 48000, "window": "hann", "group": "main"}
    return {
        "id": "gba-dev-01", "category": "build", "start": None,
        "instruction": ("Build a new flowgraph at a sample rate of 48 kHz that makes a band of noise centred on 10 kHz: "
                        "generate complex Gaussian noise with amplitude 0.5, low-pass filter it with a Low Pass Filter "
                        "(cutoff 2 kHz, transition width 500 Hz), shift it up to 10 kHz by multiplying it with a complex "
                        "10 kHz tone of amplitude 1, throttle it to real time and write it as complex samples to a File "
                        "Sink with the file name shifted.dat. At least 98 % of the power in shifted.dat must lie between "
                        "7.5 kHz and 12.5 kHz. Save the flowgraph. It should keep producing samples until it is "
                        "stopped."),
        "scripted_user": {"rules": SAVE_RULES, "max_turns": 2},
        "acceptance": {
            "ladder": {"rung": "V5",
                       "observation": observation([point("out", "shifted.dat", "complex")], 20480,
                                                  "complex samples in shifted.dat"),
                       "oracle": {"checks": [{**win, "feature": "band_fraction", "lo": 7500, "hi": 12500, "min": 0.98}],
                                  "graph_checks": [
                                      {"kind": "path", "from_ids": ["analog_noise_source_x", "analog_fastnoise_source_x"],
                                       "via_ids": ["low_pass_filter"], "to_ids": ["blocks_multiply_xx"], "group": "main"},
                                      {"kind": "path", "from_ids": ["blocks_multiply_xx"], "via_ids": THROTTLES,
                                       "to_ids": ["blocks_file_sink"], "group": "main"}]}},
            "blocks_present": [{"id": "blocks_file_sink", "params": {"file": "shifted.dat", "type": "complex"}}]},
        "golden": {"calls": [
            c("inspect_graph"),
            c("query_knowledge", {"query": "noise source gaussian complex", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "low pass filter cutoff transition", "domain": "catalog", "k": 3}),
            _set_rate("48000"),
            c("change_graph", {"reason": "Build the shifted noise band writing to shifted.dat.",
                               "add_blocks": [
                                   {"block_id": "analog_noise_source_x", "instance_name": "noise", "params": noise},
                                   {"block_id": "low_pass_filter", "instance_name": "band_limit", "params": lpf},
                                   {"block_id": "analog_sig_source_x", "instance_name": "lo", "params": lo},
                                   {"block_id": "blocks_multiply_xx", "instance_name": "shift", "params": {"type": "complex"}},
                                   {"block_id": "blocks_throttle2", "instance_name": "throttle",
                                    "params": {"type": "complex", "samples_per_second": "samp_rate"}},
                                   {"block_id": "blocks_file_sink", "instance_name": "out",
                                    "params": {"type": "complex", "file": "shifted.dat"}}],
                               "add_connections": ["noise:0->band_limit:0", "band_limit:0->shift:0", "lo:0->shift:1",
                                                   "shift:0->throttle:0", "throttle:0->out:0"]}),
            c("inspect_graph"), *TAIL],
            "final_text": ("I built and saved the flowgraph at 48 kHz: complex Gaussian noise (amplitude 0.5) goes through "
                           "a 2 kHz Low Pass Filter, is multiplied by a complex 10 kHz tone, throttled and written to "
                           "shifted.dat, so the band is centred on 10 kHz.")},
        "wrong": wrong,
        "source": {"kind": "composed", "ref": "noise -> low-pass -> multiply by a complex tone -> throttle -> file sink"},
        "cluster_id": "tier_a:shifted_noise", "license": CC_BY,
        "disjointness": ("Builds a frequency-shifted noise band. Nearest training specs: t06_spectrogram_display (a sweep "
                         "on a waterfall) and t08_run_verify_probe (a noise level probe); neither filters and shifts "
                         "noise."),
    }


def _dc_blocker_build() -> dict:
    noise = {"type": "float", "amp": "0.2", "noise_type": "analog.GR_GAUSSIAN"}
    blocks = [{"name": "noise", "id": "analog_noise_source_x", "parameters": noise},
              {"name": "offset", "id": "blocks_add_const_vxx", "parameters": {"type": "float", "const": "0.5"}},
              {"name": "thr", "id": "blocks_throttle2", "parameters": {"type": "float"}},
              {"name": "raw", "id": "blocks_file_sink", "parameters": {"type": "float", "file": "raw.dat"}},
              {"name": "clean", "id": "blocks_file_sink", "parameters": {"type": "float", "file": "clean.dat"}}]
    wrong = _untitled(blocks, [["noise", "0", "offset", "0"], ["offset", "0", "thr", "0"], ["thr", "0", "raw", "0"],
                               ["thr", "0", "clean", "0"]], "48000", "dc_removal")
    win = {"skip": 4096, "n": 16384, "group": "main"}
    return {
        "id": "gba-dev-02", "category": "build", "start": None,
        "instruction": ("Build a new flowgraph at a sample rate of 48 kHz that tests DC removal: make float Gaussian noise "
                        "with amplitude 0.2, add a constant offset of 0.5, throttle it to real time and record it as float "
                        "samples in raw.dat. Then remove the DC with GNU Radio's DC Blocker (length 64) and record its "
                        "output as float samples in clean.dat. raw.dat should average 0.5 and clean.dat about 0 (within "
                        "0.02), while the noise in clean.dat keeps an RMS level of about 0.2 (within 5 %). Save the "
                        "flowgraph. It should keep producing samples until it is stopped."),
        "scripted_user": {"rules": SAVE_RULES, "max_turns": 2},
        "acceptance": {
            "ladder": {"rung": "V5",
                       "observation": observation([point("raw", "raw.dat", "float"), point("clean", "clean.dat", "float")],
                                                  20480, "float samples in raw.dat and clean.dat"),
                       "oracle": {"checks": [
                           {**win, "point": "raw", "feature": "mean", "expect": 0.5, "tol": 0.02},
                           {**win, "point": "clean", "feature": "mean", "expect": 0.0, "tol": 0.02},
                           {**win, "point": "clean", "feature": "rms", "expect": 0.2, "rel_tol": 0.05}],
                           "graph_checks": [
                               {"kind": "path", "from_ids": ["analog_noise_source_x", "analog_fastnoise_source_x"],
                                "via_ids": ["dc_blocker_xx"], "to_ids": ["blocks_file_sink"], "group": "main"}]}},
            "blocks_present": [{"id": "dc_blocker_xx", "params": {"length": 64}}]},
        "golden": {"calls": [
            c("inspect_graph"),
            c("query_knowledge", {"query": "dc blocker length", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "add constant offset", "domain": "catalog", "k": 3}),
            _set_rate("48000"),
            c("change_graph", {"reason": "Build the DC-removal test bench.",
                               "add_blocks": [
                                   {"block_id": "analog_noise_source_x", "instance_name": "noise", "params": noise},
                                   {"block_id": "blocks_add_const_vxx", "instance_name": "offset",
                                    "params": {"type": "float", "const": "0.5"}},
                                   {"block_id": "blocks_throttle2", "instance_name": "throttle",
                                    "params": {"type": "float", "samples_per_second": "samp_rate"}},
                                   {"block_id": "blocks_file_sink", "instance_name": "raw_sink",
                                    "params": {"type": "float", "file": "raw.dat"}},
                                   {"block_id": "dc_blocker_xx", "instance_name": "dc_block",
                                    "params": {"type": "ff", "length": "64"}},
                                   {"block_id": "blocks_file_sink", "instance_name": "clean_sink",
                                    "params": {"type": "float", "file": "clean.dat"}}],
                               "add_connections": ["noise:0->offset:0", "offset:0->throttle:0", "throttle:0->raw_sink:0",
                                                   "throttle:0->dc_block:0", "dc_block:0->clean_sink:0"]}),
            c("inspect_graph"), *TAIL],
            "final_text": ("I built and saved the flowgraph at 48 kHz: Gaussian noise (amplitude 0.2) plus an offset of "
                           "0.5 goes through a Throttle to raw.dat, and through a DC Blocker (length 64) to clean.dat.")},
        "wrong": wrong,
        "source": {"kind": "composed", "ref": "noise -> add const -> throttle -> (file sink, DC blocker -> file sink)"},
        "cluster_id": "tier_a:dc_removal", "license": CC_BY,
        "disjointness": ("Builds a DC-removal test bench. Nearest training spec: t08_run_verify_probe (a noise level "
                         "probe); no training spec uses a DC Blocker."),
    }


def _if_filter_type_repair() -> dict:
    ref = named(tf_graph("gbf-dev-04"), "downconverter")
    start = setp(ref, "if_filter", type="fir_filter_ccf")
    win = {"point": "if", "skip": 1024, "n": 8192, "group": "main"}
    return {
        "id": "gba-dev-03", "category": "repair", "start": ("downconverter.grc", start),
        "instruction": ("The open flowgraph should mix a 10.7 kHz test signal (amplitude 0.8) down with an 8.7 kHz local "
                        "oscillator at a sample rate of 48 kHz, low-pass filter the product (cutoff 4 kHz, transition "
                        "1 kHz) and write the 2 kHz difference frequency as float samples to if.dat. GNU Radio Companion "
                        "reports an error and it will not run. Fix the problem without changing the frequencies, the "
                        "filter's cutoff and transition width, or the output file, and save the flowgraph. It should keep "
                        "producing samples until it is stopped."),
        "scripted_user": {"rules": SAVE_RULES, "max_turns": 2},
        "acceptance": {
            "ladder": {"rung": "V5",
                       "observation": observation([point("if", "if.dat", "float")], 12288, "float samples in if.dat"),
                       "oracle": {"checks": [
                           {**win, "feature": "peak_freq", "fs": 48000, "expect": 2000, "tol": 20},
                           {**win, "feature": "band_fraction", "fs": 48000, "lo": 1900, "hi": 2100, "window": "hann",
                            "min": 0.95},
                           {**win, "feature": "rms", "expect": 0.2828, "rel_tol": 0.05}],
                           "graph_checks": [{"kind": "same_blocks_as_start", "variables": "value_only", "group": "main"},
                                            {"kind": "same_connections_as_start", "mode": "isomorphic", "group": "main"}]}},
            "blocks_present": [{"id": "low_pass_filter", "params": {"cutoff_freq": 4000, "width": 1000}}]},
        "golden": {"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["if_filter", "mixer"]}),
            c("query_knowledge", {"query": "low pass filter fir type float", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "port type mismatch float complex", "domain": "docs", "k": 3}),
            c("change_graph", {"reason": "Make the low-pass filter take and give float samples, like the mixer.",
                               "update_params": [{"instance_name": "if_filter", "params": {"type": "fir_filter_fff"}}]}),
            c("inspect_graph"), *TAIL],
            "final_text": ("The Low Pass Filter was set to complex input (fir_filter_ccf) while the mixer gives float "
                           "samples. I set it to float (fir_filter_fff); the frequencies, cutoff, width and output file "
                           "are unchanged, and the flowgraph is saved.")},
        "wrong": setp(ref, "if_filter", cutoff_freq="1500"),
        "source": {"kind": "tier_f_graph", "ref": "gbf-dev-04/reference.grc",
                   "derivation": "the Low Pass Filter's type set to complex input"},
        "cluster_id": "tier_f:gbf-dev-04", "license": CC_BY,
        "disjointness": ("Repairs a filter type mismatch after a mixer. Nearest training specs: t05_qam_debug (rates and "
                         "types in a QAM demo) and t17_wbfm_tune_repair (rates and a filter in a WBFM receiver); neither "
                         "is a downconverter."),
    }


def _mf_sink_type_repair() -> dict:
    ref = named(tf_graph("gbf-dev-03"), "nrz_matched_filter")
    start = setp(ref, "mf_out", type="complex")
    tf = tf_task("gbf-dev-03")
    oracle = copy.deepcopy(tf["oracle"])
    oracle["checks"] = [{k: v for k, v in chk.items() if k not in ("requires", "property")} for chk in oracle["checks"]]
    oracle["graph_checks"] = [{"kind": "same_blocks_as_start", "variables": "value_only", "group": "p2p"}]
    obs = copy.deepcopy(tf["observation"])
    return {
        "id": "gba-dev-04", "category": "repair", "start": ("nrz_matched_filter.grc", start),
        "instruction": ("The open flowgraph plays a ±1 NRZ test pattern at 8 samples per symbol (32 kHz) into nrz.dat and "
                        "should write its matched-filter output, the average of every sample and the 7 before it, as "
                        "float samples to mf.dat. GNU Radio Companion reports an error and it will not run. Fix the "
                        "problem without changing the pattern, the filter or the file names, and save the flowgraph. It "
                        "should keep producing samples until it is stopped."),
        "scripted_user": {"rules": SAVE_RULES, "max_turns": 2},
        "acceptance": {"ladder": {"rung": "V5", "observation": obs, "oracle": oracle},
                       "blocks_present": [{"id": "blocks_moving_average_xx", "params": {"length": 8, "scale": 0.125}},
                                          {"id": "blocks_file_sink", "params": {"file": "mf.dat", "type": "float"}}]},
        "golden": {"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["mf", "mf_out"]}),
            c("query_knowledge", {"query": "file sink input type", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "moving average output type", "domain": "catalog", "k": 3}),
            c("change_graph", {"reason": "Make the mf.dat File Sink take float samples, like the Moving Average.",
                               "update_params": [{"instance_name": "mf_out", "params": {"type": "float"}}]}),
            c("inspect_graph"), *TAIL],
            "final_text": ("The File Sink for mf.dat expected complex samples while the Moving Average outputs float. I set "
                           "the sink to float; the pattern, the filter and the file names are unchanged, and the flowgraph "
                           "is saved.")},
        "wrong": setp(ref, "mf", scale="0.25"),
        "source": {"kind": "tier_f_graph", "ref": "gbf-dev-03/reference.grc",
                   "derivation": "the mf.dat File Sink's type set to complex"},
        "cluster_id": "tier_f:gbf-dev-03", "license": CC_BY,
        "disjointness": ("Repairs a sink type mismatch after a matched filter. Nearest training spec: "
                         "t04_broken_sink_repair (an unconnected sink); a different fault and graph."),
    }


def _ssb_retune() -> dict:
    ref = named(tf_graph("gbf-dev-06"), "ssb_transmitter")
    a = {"point": "audio", "skip": 800, "n": 1600, "group": "f2p"}
    s = {"point": "ssb", "skip": 4800, "n": 9600, "group": "f2p"}
    share = 0.25 ** 2 / (0.4 ** 2 + 0.25 ** 2)
    return {
        "id": "gba-dev-05", "category": "param", "start": ("ssb_transmitter.grc", ref),
        "instruction": ("The open flowgraph is a test bench for an upper-sideband (USB) transmitter: a two-tone audio "
                        "signal (700 Hz with amplitude 0.4 and 2.4 kHz with amplitude 0.25) at an 8 kHz audio rate is "
                        "recorded in audio.dat and turned into an upper-sideband signal on a 10 kHz carrier at 48 kHz, "
                        "recorded in ssb.dat. Retune it, changing parameter values only: move the carrier to 12 kHz and "
                        "the upper test tone to 1.9 kHz (same amplitude). Keep everything else. In ssb.dat the 700 Hz "
                        "tone must then appear at 12.7 kHz and the 1.9 kHz tone at 13.9 kHz, and the lower sideband at "
                        "11.3 kHz must stay at least 30 dB down. Save the flowgraph. It should keep producing samples "
                        "until it is stopped."),
        "scripted_user": {"rules": SAVE_RULES, "max_turns": 2},
        "acceptance": {
            "ladder": {"rung": "V5",
                       "observation": observation([point("audio", "audio.dat", "float"), point("ssb", "ssb.dat", "float")],
                                                  14400, "float samples in audio.dat (8 kHz) and ssb.dat (48 kHz)"),
                       "oracle": {"checks": [
                           {**a, "feature": "peak_freq", "fs": 8000, "expect": 700, "tol": 7},
                           {**a, "feature": "band_fraction", "fs": 8000, "lo": 1850, "hi": 1950, "expect": round(share, 4),
                            "rel_tol": 0.1},
                           {**s, "feature": "peak_freq", "fs": 48000, "expect": 12700, "tol": 127},
                           {**s, "feature": "band_fraction", "fs": 48000, "lo": 13850, "hi": 13950,
                            "expect": round(share, 4), "rel_tol": 0.15},
                           {**s, "feature": "band_fraction", "fs": 48000, "lo": 11250, "hi": 11350, "max": 1e-3}],
                           "graph_checks": [{"kind": "same_blocks_as_start", "variables": "value_only", "group": "p2p"},
                                            {"kind": "same_connections_as_start", "mode": "isomorphic", "group": "p2p"}]}}},
        "golden": {"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["carrier_freq", "tone_2400", "carrier"]}),
            c("query_knowledge", {"query": "signal source frequency", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "single sideband phasing method hilbert", "domain": "docs", "k": 3}),
            c("change_graph", {"reason": "Move the carrier to 12 kHz and the upper test tone to 1.9 kHz.",
                               "update_params": [{"instance_name": "carrier_freq", "params": {"value": "12000"}},
                                                 {"instance_name": "tone_2400", "params": {"freq": "1900"}}]}),
            c("inspect_graph", {"targets": ["carrier_freq", "tone_2400"]}), *TAIL],
            "final_text": ("I set the carrier_freq variable to 12000 Hz and the upper test tone to 1900 Hz. The SSB signal "
                           "now has its tones at 12.7 kHz and 13.9 kHz; nothing else changed, and the flowgraph is saved.")},
        "wrong": setp(ref, "carrier_freq", value="12000"),
        "source": {"kind": "tier_f_graph", "ref": "gbf-dev-06/reference.grc"},
        "cluster_id": "tier_f:gbf-dev-06", "license": CC_BY,
        "disjointness": ("Retunes an SSB transmitter's carrier and test tone. Nearest training spec: t10_param_tuning "
                         "(retunes a receiver's centre frequency and bandwidth and renames variables); a transmitter, no "
                         "renaming."),
    }


def _clip_detector_epy() -> dict:
    ref = named(tf_graph("gbf-dev-05"), "noise_limiter")
    wrong = add(ref, "clips", "epy_block", _source_code=EPY_CLIP, level="0.5")
    wrong = add(wrong, "clip_out", "blocks_file_sink", type="float", file="clipped.dat")
    wrong = wire(wrong, "limiter:0->clips:0", "clips:0->clip_out:0")
    cl = {"point": "clip", "skip": 0, "n": 20000, "group": "f2p"}
    return {
        "id": "gba-dev-06", "category": "epy", "start": ("noise_limiter.grc", ref),
        "instruction": ("The open flowgraph models an amplifier that saturates: Gaussian noise with a standard deviation "
                        "of 0.4 goes through a limiter (the Rail block) that clips at -0.784 and +0.784, so about 5 % of "
                        "the samples are clipped; noise_in.dat and limited.dat record the limiter's input and output. Add "
                        "an Embedded Python block that reads the limiter's output and outputs 1.0 for every sample that "
                        "sits at either clip level and 0.0 for every other sample, and write its output as float samples "
                        "to a new File Sink with the file name clipped.dat. Keep the other two files as they are. Save "
                        "the flowgraph. It should keep producing samples until it is stopped."),
        "scripted_user": {"rules": SAVE_RULES, "max_turns": 2},
        "acceptance": {
            "ladder": {"rung": "V5",
                       "observation": observation([point("clip", "clipped.dat", "float"),
                                                   point("lim", "limited.dat", "float")], 20480,
                                                  "float samples in clipped.dat and limited.dat"),
                       "oracle": {"checks": [
                           {**cl, "feature": "max_int_dev", "scale": 1, "max": 1e-6},
                           {**cl, "feature": "min", "min": 0.0}, {**cl, "feature": "max", "max": 1.0},
                           {**cl, "feature": "mean", "expect": 0.05, "rel_tol": 0.25},
                           {"point": "lim", "skip": 0, "n": 20000, "feature": "max", "expect": 0.784, "tol": 0.001,
                            "group": "p2p"}],
                           "graph_checks": [{"kind": "path", "from_ids": ["analog_rail_ff"], "via_ids": ["epy_block"],
                                             "to_ids": ["blocks_file_sink"], "group": "f2p"}]}},
            "blocks_present": [{"id": "epy_block"}, {"id": "blocks_file_sink", "params": {"file": "clipped.dat"}}]},
        "golden": {"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["limiter"]}),
            c("query_knowledge", {"query": "embedded python block sync block", "domain": "docs", "k": 3}),
            c("change_graph", {"reason": "Add a clip-detector Embedded Python block writing to clipped.dat.",
                               "add_blocks": [{"block_id": "epy_block", "instance_name": "clip_detect",
                                               "params": {"_source_code": EPY_CLIP}},
                                              {"block_id": "blocks_file_sink", "instance_name": "clip_sink",
                                               "params": {"type": "float", "file": "clipped.dat"}}],
                               "add_connections": ["limiter:0->clip_detect:0", "clip_detect:0->clip_sink:0"]}),
            c("change_graph", {"reason": "Set the clip level to the limiter's 0.784.",
                               "update_params": [{"instance_name": "clip_detect", "params": {"level": "0.784"}}]}),
            c("inspect_graph", {"targets": ["clip_detect"]}), *TAIL],
            "final_text": ("I added an Embedded Python block after the limiter that outputs 1.0 where the sample sits at "
                           "±0.784 and 0.0 elsewhere, written to clipped.dat as float. The other files are unchanged, and "
                           "the flowgraph is saved.")},
        "wrong": wrong,
        "source": {"kind": "tier_f_graph", "ref": "gbf-dev-05/reference.grc"},
        "cluster_id": "tier_f:gbf-dev-05", "license": CC_BY,
        "disjointness": ("A clip-detector Embedded Python block. Nearest training specs: t07_epy_custom_logic (duty-cycle "
                         "gating) and t22_epy_integrator; neither marks clipped samples."),
    }


def _phase_modify() -> dict:
    ref = named(tf_graph("gbf-dev-02"), "rayleigh_phase")
    wrong = add(ref, "arg", "blocks_complex_to_arg")
    wrong = add(wrong, "deg", "blocks_multiply_const_vxx", type="float", const="180/3.141592653589793")
    wrong = add(wrong, "ph_out", "blocks_file_sink", type="float", file="phase.dat")
    wrong = wire(wrong, "thr:0->arg:0", "arg:0->deg:0", "deg:0->ph_out:0")
    ph = {"point": "ph", "skip": 0, "n": 12288, "group": "f2p"}
    return {
        "id": "gba-dev-07", "category": "modify", "start": ("rayleigh_phase.grc", ref),
        "instruction": ("The open flowgraph simulates a Rayleigh fading channel at 48 kHz: complex Gaussian noise with "
                        "an RMS magnitude of 0.8 is throttled and its magnitude is written as float samples to "
                        "envelope.dat. Also record the phase of the same noise: the phase angle of every noise sample in "
                        "radians, between -pi and pi, written as float samples to a new File Sink with the file name "
                        "phase.dat. The phase of such noise is spread evenly over that range, so its RMS value should "
                        "come out at about 1.81 (pi divided by the square root of 3). Keep envelope.dat as it is. Save the "
                        "flowgraph. It should keep producing samples until it is stopped."),
        "scripted_user": {"rules": SAVE_RULES, "max_turns": 2},
        "acceptance": {
            "ladder": {"rung": "V5",
                       "observation": observation([point("ph", "phase.dat", "float"),
                                                   point("env", "envelope.dat", "float")], 12288,
                                                  "float samples in phase.dat and envelope.dat"),
                       "oracle": {"checks": [
                           {**ph, "feature": "rms", "expect": 1.8138, "rel_tol": 0.03},
                           {**ph, "feature": "mean", "expect": 0.0, "tol": 0.08},
                           {**ph, "feature": "min", "min": -3.1416}, {**ph, "feature": "max", "max": 3.1416},
                           {"point": "env", "skip": 0, "n": 12288, "feature": "rms", "expect": 0.8, "rel_tol": 0.03,
                            "group": "p2p"}]}},
            "blocks_present": [{"id": "blocks_file_sink", "params": {"file": "phase.dat"}}]},
        "golden": {"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["thr", "mag"]}),
            c("query_knowledge", {"query": "complex to arg phase angle", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "phase of complex samples radians", "domain": "docs", "k": 3}),
            c("change_graph", {"reason": "Record the phase of the noise in radians in phase.dat.",
                               "add_blocks": [{"block_id": "blocks_complex_to_arg", "instance_name": "phase"},
                                              {"block_id": "blocks_file_sink", "instance_name": "phase_sink",
                                               "params": {"type": "float", "file": "phase.dat"}}],
                               "add_connections": ["thr:0->phase:0", "phase:0->phase_sink:0"]}),
            c("inspect_graph"), *TAIL],
            "final_text": ("I added a Complex to Arg block after the Throttle and a File Sink writing its output, the phase "
                           "in radians, to phase.dat as float samples. envelope.dat is unchanged, and the flowgraph is "
                           "saved.")},
        "wrong": wrong,
        "source": {"kind": "tier_f_graph", "ref": "gbf-dev-02/reference.grc"},
        "cluster_id": "tier_f:gbf-dev-02", "license": CC_BY,
        "disjointness": ("Adds a phase output next to the envelope output of a Rayleigh-channel simulator. Nearest "
                         "training specs: t11_vector_sink_extend (adds a second correlated stream to a vector-sink "
                         "example) and t19_rssi_meter_build (a moving-average level meter); neither records a phase."),
    }


def _quantizer_knowledge() -> dict:
    ref = named(tf_graph("gbf-dev-08"), "demo_quantization")
    return {
        "id": "gba-dev-08", "category": "knowledge", "start": ("demo_quantization.grc", ref),
        "instruction": ("Without changing the open flowgraph, tell me three things about it: how many bits the quantizer "
                        "uses, the frequency of the test signal in Hz, and the noise level in dB as the flowgraph sets it."),
        "scripted_user": {"rules": [], "max_turns": 1},
        "acceptance": {
            "graph_unchanged": True,
            "answer": {"facts": [
                {"name": "quantizer bits", "source": {"graph": "start", "block": {"name": "bits"}, "param": "value"},
                 "match": "number", "rel_tol": 1e-6},
                {"name": "signal frequency", "source": {"graph": "start", "block": {"name": "sigfreq"}, "param": "value"},
                 "match": "number", "rel_tol": 1e-6},
                {"name": "noise level", "source": {"graph": "start", "block": {"name": "noise_amp"}, "param": "value"},
                 "match": "number", "rel_tol": 1e-6}]}},
        "golden": {"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["quantizer", "bits", "sigfreq", "noise_amp"]}),
            c("query_knowledge", {"query": "quantizer bits", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "quantization noise", "domain": "docs", "k": 3}),
            c("list_directory"),
            c("read_file", {"path": "demo_quantization.grc", "limit": 60}),
            c("file_info", {"path": "demo_quantization.grc"}),
            c("generate_python")],
            "final_text": ("The quantizer uses 6 bits. The test signal is at 12600 Hz (12.6 kHz). The noise level is set "
                           "to -20 dB (the noise_amp variable). I did not change the flowgraph.")},
        "wrong": setp(ref, "bits", value="8"),
        "source": {"kind": "tier_f_graph", "ref": "gbf-dev-08/reference.grc",
                   "derivation": "GNU Radio 3.10.12 example channels/demo_quantization.grc as converted in Tier F"},
        "cluster_id": "tier_f:gbf-dev-08",
        "license": "GPL-3.0-or-later (derived from GNU Radio 3.10.12 example channels/demo_quantization.grc)",
        "disjointness": ("Reads three settings of a quantization demo. Nearest training specs: t09_knowledge_concepts and "
                         "t25_sync_chain_explain (explain a graph); neither asks for these settings."),
    }


def _save_magnitude_block() -> dict:
    ref = named(tf_graph("gbf-dev-02"), "rayleigh_envelope")
    start = drop(ref, "mag")
    start = add(start, "mag", "epy_block", _source_code=EPY_MAG)
    start = wire(start, "thr:0->mag:0", "mag:0->snk:0")
    return {
        "id": "gba-dev-09", "category": "save_block", "start": ("rayleigh_envelope.grc", start),
        "instruction": ("The open flowgraph takes the magnitude of complex noise with an Embedded Python block. Save that "
                        "Embedded Python block to the block library as a reusable block with the block id "
                        "complex_magnitude and the label Complex Magnitude. Do not change the flowgraph itself."),
        "scripted_user": {"rules": [{"pattern": r"\b(category|overwrite)\b", "reply": "Use the default."}],
                          "max_turns": 2},
        "acceptance": {"hier_block": {"block_id": "complex_magnitude", "label": "Complex Magnitude", "inputs": 1,
                                      "outputs": 1},
                       "graph_unchanged": True, "ladder": {"rung": "V3"}},
        "golden": {"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["mag"]}),
            c("query_knowledge", {"query": "save embedded python block to library", "domain": "docs", "k": 3}),
            c("generate_python"),
            c("list_directory"),
            c("save_graph"),
            c("save_block", {"instance_name": "mag", "block_id": "complex_magnitude", "label": "Complex Magnitude"}),
            RUN,
            c("get_run_log")],
            "final_text": ("I saved the Embedded Python block 'mag' to the block library as complex_magnitude with the "
                           "label Complex Magnitude. The flowgraph itself is unchanged.")},
        "wrong": start,
        "source": {"kind": "tier_f_graph", "ref": "gbf-dev-02/reference.grc",
                   "derivation": "Complex to Mag replaced by an Embedded Python magnitude block"},
        "cluster_id": "tier_f:gbf-dev-02", "license": CC_BY,
        "disjointness": ("Saves an existing magnitude block to the library. Nearest training spec: t13_epy_save_block "
                         "(builds a duty-cycle gating block and saves it)."),
    }


def _export_mux_settings() -> dict:
    ref = named(tf_graph("gbf-dev-07"), "stream_mux_demo")
    return {
        "id": "gba-dev-10", "category": "file_ops", "start": ("stream_mux_demo.grc", ref),
        "instruction": ("Generate the Python code of the open flowgraph into a new folder named export, as "
                        "export/stream_mux_demo.py, and write a JSON file mux.json in the project folder with two keys: "
                        "\"num_inputs\", the Stream Mux's number of inputs (a number), and \"lengths\", its lengths (a "
                        "list of numbers). Do not change the flowgraph."),
        "scripted_user": {"rules": [], "max_turns": 1},
        "acceptance": {
            "graph_unchanged": True,
            "files": [{"path": "export/stream_mux_demo.py",
                       "regex": [r"class stream_mux_demo\(gr\.top_block", r"blocks\.stream_mux\("]},
                      {"path": "mux.json", "json_keys": ["num_inputs", "lengths"],
                       "regex": [r"\"num_inputs\"\s*:\s*3\b", r"\"lengths\"\s*:\s*\[\s*2\s*,\s*4\s*,\s*3\s*\]"]}]},
        "golden": {"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["blocks_stream_mux_0"]}),
            c("list_directory"),
            c("create_directory", {"path": "export"}),
            c("run_command", {"command": "grcc -o export stream_mux_demo.grc"}),
            c("list_directory", {"path": "export"}),
            c("write_file", {"path": "mux.json", "content": "{\"num_inputs\": 3, \"lengths\": [2, 4, 3]}\n"}),
            c("read_file", {"path": "mux.json"})],
            "final_text": ("I generated export/stream_mux_demo.py with grcc and wrote mux.json with num_inputs 3 and "
                           "lengths [2, 4, 3]. The flowgraph is unchanged.")},
        "wrong": ref,
        "wrong_json": {"files": {"export/stream_mux_demo.py": "class stream_mux_demo(gr.top_block):\n    pass\n"
                                                              "# blocks.stream_mux(\n",
                                 "mux.json": "{\"num_inputs\": 2, \"lengths\": [4, 3]}\n"}},
        "source": {"kind": "tier_f_graph", "ref": "gbf-dev-07/reference.grc",
                   "derivation": "GNU Radio 3.10.12 example blocks/stream_mux_demo.grc as converted in Tier F"},
        "cluster_id": "tier_f:gbf-dev-07",
        "license": "GPL-3.0-or-later (derived from GNU Radio 3.10.12 example blocks/stream_mux_demo.grc)",
        "disjointness": ("File operations: generated code in a new folder and a JSON summary of one block. Nearest "
                         "training spec: t14_file_ops_notes (a notes.md summary); different files and content."),
    }


def _run_verify_clip() -> dict:
    ref = named(tf_graph("gbf-dev-05"), "noise_limiter")
    script = ("python3 -c \"import numpy as np; x=np.fromfile('limited.dat',dtype=np.float32); "
              "print(len(x), x.max(), x.min(), np.mean(np.abs(x) >= x.max()-1e-6))\"")
    return {
        "id": "gba-dev-11", "category": "run_verify", "start": ("noise_limiter.grc", ref),
        "instruction": ("Run the open flowgraph for about two seconds, then look at the samples it wrote to limited.dat "
                        "and tell me the level at which the limiter clips the positive peaks. Do not change the "
                        "flowgraph."),
        "scripted_user": {"rules": [], "max_turns": 1},
        "acceptance": {
            "graph_unchanged": True,
            "files": [{"path": "limited.dat"}],
            "answer": {"facts": [{"name": "positive clip level", "source": {"graph": "start", "block": {"name": "limiter"},
                                                                             "param": "hi"},
                                  "match": "number", "rel_tol": 0.02}]}},
        "golden": {"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["limiter"]}),
            RUN,
            c("get_run_log"),
            c("list_directory"),
            c("file_info", {"path": "limited.dat"}),
            c("run_command", {"command": script}),
            c("query_knowledge", {"query": "rail block limits", "domain": "catalog", "k": 3})],
            "final_text": ("I ran the flowgraph for two seconds and read limited.dat: the positive peaks are clipped at "
                           "0.784 (and the negative ones at -0.784). The flowgraph is unchanged.")},
        "wrong": ref,
        "wrong_json": {"final_text": ("I ran the flowgraph and read limited.dat: the positive peaks are clipped at 1.0. "
                                      "The flowgraph is unchanged.")},
        "source": {"kind": "tier_f_graph", "ref": "gbf-dev-05/reference.grc"},
        "cluster_id": "tier_f:gbf-dev-05", "license": CC_BY,
        "disjointness": ("Runs a limiter model and measures its clip level from the recorded file. Nearest training "
                         "spec: t08_run_verify_probe (a noise level probe); a different measurement on a given graph."),
    }


def _github_refusal() -> dict:
    ref = named(tf_graph("gbf-dev-09"), "filter_taps")
    return {
        "id": "gba-dev-12", "category": "refusal", "start": ("filter_taps.grc", ref),
        "instruction": ("Push this flowgraph to my GitHub repository grc-demos and open a pull request with it. If that "
                        "cannot be done from here, do not change anything; just tell me which file the root raised "
                        "cosine filter's output is written to."),
        "scripted_user": {"rules": [{"pattern": r"\b(token|credential|access|login|ssh)\b",
                                     "reply": "I don't have a token to give you here."}], "max_turns": 1},
        "acceptance": {
            "graph_unchanged": True,
            "answer": {"facts": [{"name": "RRC output file", "source": {"graph": "start", "block": {"name": "rrc_sink"},
                                                                         "param": "file", "use": "raw"},
                                  "match": "text"}]}},
        "golden": {"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["rrc_filter", "rrc_sink"]}),
            c("run_command", {"command": "git --version"}),
            c("run_command", {"command": "git remote -v 2>&1 | head -3"}),
            c("list_directory"),
            c("query_knowledge", {"query": "share a flowgraph file", "domain": "docs", "k": 3}),
            c("read_file", {"path": "filter_taps.grc", "limit": 40}),
            c("generate_python")],
            "final_text": ("I can't push to GitHub from here: this environment has no network access and no GitHub "
                           "credentials. I changed nothing. The root raised cosine filter's output is written to rrc.dat.")},
        "wrong": ref,
        "wrong_json": {"final_text": ("I can't push to GitHub from here, so I changed nothing. The root raised cosine "
                                      "filter's output goes to lowpass.dat.")},
        "source": {"kind": "tier_f_graph", "ref": "gbf-dev-09/reference.grc",
                   "derivation": "GNU Radio 3.10.12 example filter/filter_taps.grc as converted in Tier F"},
        "cluster_id": "tier_f:gbf-dev-09",
        "license": "GPL-3.0-or-later (derived from GNU Radio 3.10.12 example filter/filter_taps.grc)",
        "disjointness": ("A request that needs network access and credentials the environment does not have: the correct "
                         "behaviour is to decline, change nothing and answer the fallback question. No training spec "
                         "asks to publish a graph."),
    }


DEV_TASKS = [_shifted_noise_build, _dc_blocker_build, _if_filter_type_repair, _mf_sink_type_repair, _ssb_retune,
             _clip_detector_epy, _phase_modify, _quantizer_knowledge, _save_magnitude_block, _export_mux_settings,
             _run_verify_clip, _github_refusal]


# --------------------------------------------------------------------------- writing
INNER = r"""
import json, sys, yaml
from grcbench.tier_f._worker import _normalise
from grcbench.tier_f.grc import platform
jobs = json.loads(open(sys.argv[1]).read())
p = platform()
for out, text in sorted(jobs.items()):
    fg = p.make_flow_graph()
    fg.import_data(_normalise(yaml.safe_load(text)))
    fg.rewrite()
    fg.validate()
    p.save_flow_graph(out, fg)
    print(out, fg.is_valid())
"""


def full_format(jobs: dict[Path, dict]) -> None:
    """Write each compact graph in GRC's full file format with GNU Radio's own writer (sandboxed)."""
    from ..common import sandbox
    run = WORK / "tier_a" / "first-set-gen"
    shutil.rmtree(run, ignore_errors=True)
    run.mkdir(parents=True)
    spec = {str(run / f"g{i:02d}.grc"): yaml.safe_dump(d, sort_keys=False) for i, d in enumerate(jobs.values())}
    (run / "jobs.json").write_text(json.dumps(spec))
    res = sandbox.run([sys.executable, "-c", INNER, str(run / "jobs.json")], run, timeout=300)
    if res.returncode != 0:
        raise SystemExit(res.stderr[-4000:])
    for dest, src in zip(jobs, spec):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    shutil.rmtree(run, ignore_errors=True)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8",
                    newline="\n")


def record(spec: dict, split: str, generator: tuple[str, str] = (GENERATOR, GENERATOR_VERSION)) -> dict:
    rec = {"id": spec["id"], "tier": "A", "split": split, "version": 1, "canary": CANARY, "license": spec["license"],
           "source": spec["source"], "cluster_id": spec["cluster_id"],
           "provenance": {"drafted_by": "Claude (Anthropic)", "generator": generator[0],
                          "generator_version": generator[1]},
           "review": {"status": "pending-team-review", "reviewers": [], "notes": ""},
           "category": spec["category"], "instruction": spec["instruction"],
           "start_state": {"active_graph": spec["start"][0] if spec["start"] else None},
           "scripted_user": spec["scripted_user"], "acceptance": spec["acceptance"],
           "expected_tool_calls": [8, 20], "disjointness": spec["disjointness"]}
    return rec


def write(split: str, makers: list, only: set[str] | None = None,
          generator: tuple[str, str] = (GENERATOR, GENERATOR_VERSION)) -> list[Path]:
    jobs: dict[Path, dict] = {}
    written = []
    for make in makers:
        spec = make()
        if only and spec["id"] not in only:
            continue
        d = OUT / split / spec["id"]
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        _write_json(d / "task.json", record(spec, split, generator))
        _write_json(d / "golden_calls.json", spec["golden"])
        if spec["start"]:
            jobs[d / "start" / spec["start"][0]] = spec["start"][1]
        jobs[d / "wrong_end_state.grc"] = spec["wrong"]
        if "wrong_json" in spec:
            _write_json(d / "wrong_end_state.json", spec["wrong_json"])
        written.append(d)
    if jobs:
        full_format(jobs)
    return written


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb tier-a first-set", description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="comma-separated task ids")
    args = ap.parse_args(argv)
    only = set(args.only.split(",")) if args.only else None
    for split, makers in (("test", TEST_TASKS), ("dev", DEV_TASKS)):
        for d in write(split, makers, only):
            print("wrote", d.relative_to(OUT.parent.parent))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
