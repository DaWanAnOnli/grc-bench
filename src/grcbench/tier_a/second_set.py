"""The other 20 Tier-A test tasks (`./gb tier-a second-set`; remaining-work plan step 6; eval plan §4.4, §5.4).

Builds the specs of grcbench.tier_a.spec_drafts (gba-test-11 to gba-test-30) into task folders in the
format of grcbench.tier_a.first_set: task.json, start/<graph>.grc, golden_calls.json, wrong_end_state.grc (and
wrong_end_state.json). The instruction, category, source, cluster, licence and disjointness note of every task
are the spec's, word for word; this module adds what the spec only outlines: the start graph, the acceptance
checks, the golden tool calls and the wrong end state. Every task is checked by `./gb tier-a gates` like the
first set.
"""

from __future__ import annotations

import argparse
import copy
import sys
from typing import Any

from ..common.paths import ITEMS
from .spec_drafts import draft_records
from .first_set import (RUN, SAVE_RULES, TAIL, THROTTLES, _set_rate, _untitled, add, c, drop, named, observation,
                        point, setp, tf_graph, tf_task, wire, write)

GENERATOR = ("grcbench.tier_a.second_set", "1")
SPECS = {r["id"]: r for r in draft_records()}


def _from_spec(task_id: str, **built: Any) -> dict:
    """The task: the spec's own fields, plus what the builder adds (start, acceptance, golden, wrong...)."""
    s = SPECS[task_id]
    return {"id": task_id, "category": s["category"], "instruction": s["instruction"], "source": s["source"],
            "cluster_id": s["cluster_id"], "license": s["license"], "disjointness": s["disjointness"], **built}


def _tier_f_oracle(task_id: str, groups: dict[str, str]) -> dict:
    """A Tier-F task's signal checks, regrouped by observation point (e.g. {"received": "p2p", "carrier": "f2p"})."""
    oracle = copy.deepcopy(tf_task(task_id)["oracle"])
    checks = [{**{k: v for k, v in chk.items() if k not in ("requires", "property", "stretch")},
               "group": groups[chk["point"]]} for chk in oracle["checks"] if chk["point"] in groups]
    return {"checks": checks}


SAME_GRAPH = [{"kind": "same_blocks_as_start", "variables": "value_only", "group": "p2p"},
              {"kind": "same_connections_as_start", "mode": "isomorphic", "group": "p2p"}]


# --------------------------------------------------------------------------- build
def _tone_in_noise_build() -> dict:
    tone = {"type": "complex", "samp_rate": "samp_rate", "waveform": "analog.GR_COS_WAVE", "freq": "5000", "amp": "0.1"}
    noise = {"type": "complex", "noise_type": "analog.GR_GAUSSIAN", "amp": "0.1", "seed": "0"}
    blocks = [{"name": "tone", "id": "analog_sig_source_x", "parameters": tone},
              {"name": "noise", "id": "analog_noise_source_x", "parameters": {**noise, "amp": "0.3"}},
              {"name": "adder", "id": "blocks_add_xx", "parameters": {"type": "complex"}},
              {"name": "thr", "id": "blocks_throttle2", "parameters": {"type": "complex"}},
              {"name": "out", "id": "blocks_file_sink", "parameters": {"type": "complex", "file": "tone_in_noise.dat"}}]
    wrong = _untitled(blocks, [["tone", "0", "adder", "0"], ["noise", "0", "adder", "1"], ["adder", "0", "thr", "0"],
                               ["thr", "0", "out", "0"]], "32000", "tone_in_noise")
    win = {"point": "out", "skip": 2048, "n": 16384, "fs": 32000, "window": "hann", "group": "main"}
    return _from_spec(
        "gba-test-11", start=None,
        scripted_user={"rules": SAVE_RULES, "max_turns": 2},
        acceptance={
            "ladder": {"rung": "V5",
                       "observation": observation([point("out", "tone_in_noise.dat", "complex")], 20480,
                                                  "complex samples in tone_in_noise.dat"),
                       "oracle": {"checks": [
                           {**win, "feature": "peak_freq", "expect": 5000, "tol": 10},
                           {**win, "feature": "band_fraction", "lo": 4900, "hi": 5100, "expect": 0.5, "rel_tol": 0.1},
                           {"point": "out", "skip": 2048, "n": 16384, "feature": "rms", "expect": 0.1414,
                            "rel_tol": 0.05, "group": "main"}],
                           "graph_checks": [
                               {"kind": "path", "from_ids": ["analog_sig_source_x"], "via_ids": THROTTLES,
                                "to_ids": ["blocks_file_sink"], "group": "main"}]}},
            "blocks_present": [{"id": "blocks_file_sink", "params": {"file": "tone_in_noise.dat", "type": "complex"}}]},
        golden={"calls": [
            c("inspect_graph"),
            c("query_knowledge", {"query": "signal source complex exponential", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "noise source gaussian", "domain": "catalog", "k": 3}),
            c("change_graph", {"reason": "Build a 5 kHz tone in equal-power noise writing to tone_in_noise.dat.",
                               "add_blocks": [
                                   {"block_id": "analog_sig_source_x", "instance_name": "tone", "params": tone},
                                   {"block_id": "analog_noise_source_x", "instance_name": "noise", "params": noise},
                                   {"block_id": "blocks_add_xx", "instance_name": "adder", "params": {"type": "complex"}},
                                   {"block_id": "blocks_throttle2", "instance_name": "throttle",
                                    "params": {"type": "complex", "samples_per_second": "samp_rate"}},
                                   {"block_id": "blocks_file_sink", "instance_name": "out",
                                    "params": {"type": "complex", "file": "tone_in_noise.dat"}}],
                               "add_connections": ["tone:0->adder:0", "noise:0->adder:1", "adder:0->throttle:0",
                                                   "throttle:0->out:0"]}),
            c("inspect_graph"), *TAIL],
            "final_text": ("I built and saved the flowgraph at 32 kHz: a complex 5 kHz tone (amplitude 0.1) and complex "
                           "Gaussian noise (amplitude 0.1) are added, throttled and written as complex samples to "
                           "tone_in_noise.dat, so tone and noise have equal power (0 dB SNR).")},
        wrong=wrong)


def _decimate_select_build() -> dict:
    tone = {"type": "complex", "samp_rate": "samp_rate", "waveform": "analog.GR_COS_WAVE", "amp": "0.5"}
    lpf = {"type": "fir_filter_ccf", "decim": "4", "gain": "1", "samp_rate": "samp_rate", "cutoff_freq": "4000",
           "width": "1000"}
    blocks = [{"name": "t2", "id": "analog_sig_source_x", "parameters": {**tone, "freq": "2000"}},
              {"name": "t9", "id": "analog_sig_source_x", "parameters": {**tone, "freq": "9000"}},
              {"name": "adder", "id": "blocks_add_xx", "parameters": {"type": "complex"}},
              {"name": "thr", "id": "blocks_throttle2", "parameters": {"type": "complex"}},
              {"name": "lpf", "id": "low_pass_filter", "parameters": {**lpf, "decim": "1"}},
              {"name": "out", "id": "blocks_file_sink", "parameters": {"type": "complex", "file": "narrow.dat"}}]
    wrong = _untitled(blocks, [["t2", "0", "adder", "0"], ["t9", "0", "adder", "1"], ["adder", "0", "thr", "0"],
                               ["thr", "0", "lpf", "0"], ["lpf", "0", "out", "0"]], "48000", "narrow_select")
    win = {"point": "out", "skip": 1024, "n": 8192, "fs": 12000, "window": "hann", "group": "main"}
    return _from_spec(
        "gba-test-12", start=None,
        scripted_user={"rules": SAVE_RULES, "max_turns": 2},
        acceptance={
            "ladder": {"rung": "V5",
                       "observation": observation([point("out", "narrow.dat", "complex")], 9216,
                                                  "complex samples in narrow.dat (12 kHz)"),
                       "oracle": {"checks": [
                           {**win, "feature": "peak_freq", "expect": 2000, "tol": 10},
                           {**win, "feature": "band_fraction", "lo": 1900, "hi": 2100, "min": 0.99},
                           {"point": "out", "skip": 1024, "n": 8192, "feature": "rms", "expect": 0.5, "rel_tol": 0.03,
                            "group": "main"}]}},
            "blocks_present": [{"id": "low_pass_filter", "params": {"decim": 4, "cutoff_freq": 4000, "width": 1000}},
                               {"id": "blocks_file_sink", "params": {"file": "narrow.dat", "type": "complex"}}]},
        golden={"calls": [
            c("inspect_graph"),
            c("query_knowledge", {"query": "low pass filter decimation", "domain": "catalog", "k": 3}),
            _set_rate("48000"),
            c("change_graph", {"reason": "Add two tones, keep the 2 kHz one and decimate to 12 kHz into narrow.dat.",
                               "add_blocks": [
                                   {"block_id": "analog_sig_source_x", "instance_name": "tone_2k",
                                    "params": {**tone, "freq": "2000"}},
                                   {"block_id": "analog_sig_source_x", "instance_name": "tone_9k",
                                    "params": {**tone, "freq": "9000"}},
                                   {"block_id": "blocks_add_xx", "instance_name": "adder", "params": {"type": "complex"}},
                                   {"block_id": "blocks_throttle2", "instance_name": "throttle",
                                    "params": {"type": "complex", "samples_per_second": "samp_rate"}},
                                   {"block_id": "low_pass_filter", "instance_name": "select", "params": lpf},
                                   {"block_id": "blocks_file_sink", "instance_name": "out",
                                    "params": {"type": "complex", "file": "narrow.dat"}}],
                               "add_connections": ["tone_2k:0->adder:0", "tone_9k:0->adder:1", "adder:0->throttle:0",
                                                   "throttle:0->select:0", "select:0->out:0"]}),
            c("inspect_graph"), *TAIL],
            "final_text": ("I built and saved the flowgraph at 48 kHz: two complex tones (2 kHz and 9 kHz, amplitude 0.5) "
                           "are added and throttled, and a Low Pass Filter (cutoff 4 kHz, transition 1 kHz, decimation 4) "
                           "keeps the 2 kHz tone and writes it at 12 kHz to narrow.dat.")},
        wrong=wrong)


# --------------------------------------------------------------------------- repair
def _hpf_cutoff_repair() -> dict:
    ref = named(tf_graph("gbf-test-10"), "clean_tone")
    start = setp(ref, "hpf", cutoff_freq="30000")
    out = {"point": "out", "skip": 1024, "n": 8192, "group": "f2p"}
    return _from_spec(
        "gba-test-13", start=("clean_tone.grc", start),
        scripted_user={"rules": SAVE_RULES, "max_turns": 2},
        acceptance={
            "ladder": {"rung": "V5",
                       "observation": observation([point("out", "clean_tone.dat", "float")], 9216,
                                                  "float samples in clean_tone.dat"),
                       "oracle": {"checks": [
                           {**out, "feature": "mean", "expect": 0.0, "tol": 0.01},
                           {**out, "feature": "rms", "expect": 0.7071, "rel_tol": 0.03},
                           {**out, "feature": "peak_freq", "fs": 32000, "expect": 1200, "tol": 10}],
                           "graph_checks": [{"kind": "same_blocks_as_start", "variables": "value_only",
                                             "group": "p2p"}]}},
            "blocks_present": [{"id": "analog_sig_source_x", "params": {"freq": 1200, "amp": 0.4, "offset": 0.75}},
                               {"id": "blocks_multiply_const_vxx", "params": {"const": 2.5}}]},
        golden={"calls": [
            c("inspect_graph"),
            RUN,
            c("get_run_log"),
            c("inspect_graph", {"targets": ["hpf"]}),
            c("query_knowledge", {"query": "high pass filter cutoff", "domain": "catalog", "k": 3}),
            c("change_graph", {"reason": "The cutoff (30 kHz) is above half the 32 kHz sample rate; set it to 300 Hz.",
                               "update_params": [{"instance_name": "hpf", "params": {"cutoff_freq": "300"}}]}),
            c("inspect_graph", {"targets": ["hpf"]}), *TAIL],
            "final_text": ("The High Pass Filter's cutoff was 30 kHz, above half the 32 kHz sample rate, so the filter "
                           "design failed when the flowgraph started. I set the cutoff to 300 Hz: the DC offset is "
                           "removed and the 1.2 kHz tone passes with amplitude 1.0. The tone, gain and output file are "
                           "unchanged, and the flowgraph is saved.")},
        wrong=setp(ref, "hpf", cutoff_freq="3000"))


def _pll_limits_repair() -> dict:
    ref = named(tf_graph("gbf-test-02"), "pll_carrier_recovery")
    limits = {"max_freq": "6.283185307179586*4000/samp_rate", "min_freq": "-6.283185307179586*4000/samp_rate"}
    start = setp(ref, "pll", max_freq=limits["min_freq"], min_freq=limits["max_freq"])
    oracle = _tier_f_oracle("gbf-test-02", {"received": "p2p", "carrier": "f2p"})
    oracle["graph_checks"] = copy.deepcopy(SAME_GRAPH)
    fft = ("python3 -c \"import numpy as np; x=np.fromfile('carrier.dat',dtype=np.complex64)[16000:32000]; "
           "f=np.fft.fftfreq(len(x),1/32000); print(len(x), f[np.argmax(np.abs(np.fft.fft(x)))])\"")
    return _from_spec(
        "gba-test-14", start=("pll_carrier_recovery.grc", start),
        scripted_user={"rules": SAVE_RULES, "max_turns": 2},
        acceptance={
            "ladder": {"rung": "V5", "observation": copy.deepcopy(tf_task("gbf-test-02")["observation"]),
                       "oracle": oracle},
            "blocks_present": [{"id": "analog_pll_refout_cc", "params": {"w": 0.02}},
                               {"id": "channels_channel_model", "params": {"noise_voltage": 0.7}}]},
        golden={"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["pll"]}),
            c("query_knowledge", {"query": "pll refout max_freq min_freq", "domain": "catalog", "k": 3}),
            RUN,
            c("get_run_log"),
            c("run_command", {"command": fft}),
            c("change_graph", {"reason": "The PLL's frequency limits are swapped; restore max +4 kHz and min -4 kHz.",
                               "update_params": [{"instance_name": "pll", "params": limits}]}),
            *TAIL],
            "final_text": ("The PLL's maximum and minimum frequencies were swapped (max -4 kHz, min +4 kHz), so its "
                           "frequency stayed pinned at a limit and it never locked to the 2.4 kHz carrier. I restored "
                           "max_freq to +4 kHz and min_freq to -4 kHz; the loop bandwidth, carrier, noise and file names "
                           "are unchanged, and the flowgraph is saved.")},
        wrong=setp(ref, "pll", max_freq=limits["max_freq"], min_freq=limits["max_freq"]))   # one limit restored


# --------------------------------------------------------------------------- parameter tuning
def _fm_deviation_tune() -> dict:
    ref = named(tf_graph("gbf-test-14"), "nbfm_loopback")
    iq = {"point": "iq", "skip": 1000, "n": 8000, "group": "f2p"}
    audio = {"point": "audio", "skip": 1000, "n": 8000, "group": "f2p"}
    return _from_spec(
        "gba-test-15", start=("nbfm_loopback.grc", ref),
        scripted_user={"rules": SAVE_RULES, "max_turns": 2},
        acceptance={
            "ladder": {"rung": "V5", "observation": copy.deepcopy(tf_task("gbf-test-14")["observation"]),
                       "oracle": {"checks": [
                           {**iq, "feature": "rms", "expect": 1.0, "rel_tol": 0.05},
                           {**iq, "feature": "band_fraction", "fs": 40000, "lo": -6200, "hi": 6200, "min": 0.97},
                           {**iq, "feature": "band_fraction", "fs": 40000, "lo": -3700, "hi": 3700, "max": 0.9},
                           {**audio, "feature": "peak_freq", "fs": 8000, "expect": 1200, "tol": 12},
                           {**audio, "feature": "rms", "expect": 0.7071, "rel_tol": 0.03}],
                           "graph_checks": copy.deepcopy(SAME_GRAPH)}}},
        golden={"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["fm_mod", "fm_demod", "deviation", "rx_scale"]}),
            c("query_knowledge", {"query": "frequency modulator sensitivity", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "quadrature demod gain", "domain": "catalog", "k": 3}),
            c("change_graph", {"reason": "Double the deviation to 5 kHz and scale the receiver to match.",
                               "update_params": [{"instance_name": "deviation", "params": {"value": "5000"}},
                                                 {"instance_name": "rx_scale", "params": {"value": "5000"}}]}),
            c("inspect_graph", {"targets": ["deviation", "rx_scale"]}), *TAIL],
            "final_text": ("I set the peak deviation to 5 kHz and the receiver scale to 5000, so 5 kHz of deviation "
                           "demodulates to 1.0: the recovered 1.2 kHz tone has amplitude 1.0. Only these two values "
                           "changed, and the flowgraph is saved.")},
        wrong=setp(ref, "deviation", value="5000"))


def _sawtooth_peaks_tune() -> dict:
    ref = named(tf_graph("gbf-test-09"), "sawtooth_peaks")
    peaks = {"point": "peaks", "skip": 480, "n": 15360, "group": "f2p"}       # 128 periods of 400 Hz
    return _from_spec(
        "gba-test-16", start=("sawtooth_peaks.grc", ref),
        scripted_user={"rules": SAVE_RULES, "max_turns": 2},
        acceptance={
            "ladder": {"rung": "V5", "observation": copy.deepcopy(tf_task("gbf-test-09")["observation"]),
                       "oracle": {"checks": [
                           {**peaks, "feature": "mean", "expect": 400 / 48000, "rel_tol": 0.02},
                           {**peaks, "feature": "max_abs", "expect": 1.0, "tol": 0.001},
                           {"point": "saw", "skip": 480, "n": 15360, "feature": "band_fraction", "fs": 48000, "lo": 350,
                            "hi": 450, "remove_mean": True, "expect": 0.6079, "rel_tol": 0.1, "group": "f2p"},
                           {"point": "average", "skip": 4800, "feature": "mean", "expect": 1.0, "rel_tol": 0.05,
                            "group": "p2p"}],
                           "graph_checks": copy.deepcopy(SAME_GRAPH)}},
            "blocks_present": [{"id": "blocks_peak_detector2_fb", "params": {"alpha": 0.01}}]},
        golden={"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["saw_freq", "lookahead", "alpha", "blocks_peak_detector2_fb_0"]}),
            c("query_knowledge", {"query": "peak detector2 look ahead alpha", "domain": "catalog", "k": 3}),
            c("change_graph", {"reason": "Retune for a 400 Hz sawtooth and a five times faster running average.",
                               "update_params": [{"instance_name": "saw_freq", "params": {"value": "400"}},
                                                 {"instance_name": "alpha", "params": {"value": "0.01"}}]}),
            c("inspect_graph", {"targets": ["saw_freq", "lookahead", "alpha"]}), *TAIL],
            "final_text": ("I set saw_freq to 400 Hz; the look-ahead follows it (int(samp_rate/saw_freq/1.1), now 109 "
                           "samples), so the detector still flags one peak per period. I set alpha to 0.01, five times "
                           "faster than before. Only these values changed, and the flowgraph is saved.")},
        wrong=setp(ref, "saw_freq", value="400"))


EPY_DAC = '''"""8-bit DAC codes: round(255 * x), limited to 0..255, output as float."""

import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """Converts samples in 0..1 to 8-bit DAC codes."""

    def __init__(self):
        gr.sync_block.__init__(self, name="DAC Codes", in_sig=[np.float32], out_sig=[np.float32])

    def work(self, input_items, output_items):
        output_items[0][:] = np.clip(np.round(input_items[0] * 255.0), 0, 255)
        return len(output_items[0])
'''


def _with_point(task_id: str, *points: dict) -> dict:
    """A Tier-F task's observation with further points."""
    obs = copy.deepcopy(tf_task(task_id)["observation"])
    obs["points"] += list(points)
    return obs


def _dac_codes_epy() -> dict:
    ref = named(tf_graph("gbf-test-04"), "dac_test_signal")
    oracle = _tier_f_oracle("gbf-test-04", {"out": "p2p"})
    codes = {"point": "codes", "skip": 4800, "group": "f2p"}
    oracle["checks"] += [{**codes, "feature": "max_int_dev", "max": 1e-6},
                         {**codes, "feature": "min", "min": 37.999}, {**codes, "feature": "max", "max": 242.001},
                         {**codes, "feature": "mean", "expect": 140.25, "tol": 0.15},
                         {**codes, "feature": "peak_freq", "fs": 48000, "expect": 1250, "tol": 12.5}]
    oracle["graph_checks"] = [{"kind": "path", "from_ids": ["blocks_add_const_vxx"], "via_ids": ["epy_block"],
                               "to_ids": ["blocks_file_sink"], "group": "f2p"}]
    wrong = add(ref, "codes", "epy_block", _source_code=EPY_DAC.replace("np.round(", "np.floor("))
    wrong = add(wrong, "codes_sink", "blocks_file_sink", type="float", file="codes.dat")
    wrong = wire(wrong, "offset:0->codes:0", "codes:0->codes_sink:0")
    return _from_spec(
        "gba-test-17", start=("dac_test_signal.grc", ref),
        scripted_user={"rules": SAVE_RULES, "max_turns": 2},
        acceptance={
            "ladder": {"rung": "V5", "observation": _with_point("gbf-test-04", point("codes", "codes.dat", "float")),
                       "oracle": oracle},
            "blocks_present": [{"id": "epy_block"}, {"id": "blocks_file_sink", "params": {"file": "codes.dat"}}]},
        golden={"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["offset", "out"]}),
            c("query_knowledge", {"query": "embedded python block", "domain": "docs", "k": 3}),
            c("change_graph", {"reason": "Add an Embedded Python block making 8-bit DAC codes, written to codes.dat.",
                               "add_blocks": [{"block_id": "epy_block", "instance_name": "dac_codes",
                                               "params": {"_source_code": EPY_DAC}},
                                              {"block_id": "blocks_file_sink", "instance_name": "codes_sink",
                                               "params": {"type": "float", "file": "codes.dat"}}],
                               "add_connections": ["offset:0->dac_codes:0", "dac_codes:0->codes_sink:0"]}),
            c("inspect_graph", {"targets": ["dac_codes"]}), *TAIL],
            "final_text": ("I added an Embedded Python block after the offset that outputs round(255 * x), limited to "
                           "0..255, as float, and a File Sink writing it to codes.dat. dac_test.dat is unchanged, and the "
                           "flowgraph is saved.")},
        wrong=wrong)


EPY_UNWRAP = '''"""Unwraps a phase in -pi..pi into a continuous phase, keeping its state between work() calls."""

import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """Phase unwrapper: removes the 2*pi jumps of a wrapped phase."""

    def __init__(self):
        gr.sync_block.__init__(self, name="Phase Unwrap", in_sig=[np.float32], out_sig=[np.float32])
        self.last = None
        self.offset = 0.0

    def work(self, input_items, output_items):
        x = input_items[0].astype(np.float64)
        prev = x[0] if self.last is None else self.last
        d = np.diff(np.concatenate(([prev], x)))
        steps = np.cumsum(np.where(d < -np.pi, 2 * np.pi, np.where(d > np.pi, -2 * np.pi, 0.0)))
        output_items[0][:] = x + self.offset + steps
        self.offset += steps[-1]
        self.last = x[-1]
        return len(output_items[0])
'''


def _phase_unwrap_epy() -> dict:
    ref = named(tf_graph("gbf-test-08"), "phase_comparator")
    oracle = _tier_f_oracle("gbf-test-08", {"out": "p2p"})
    unwrapped = {"point": "unwrapped", "skip": 0, "n": 8000, "group": "f2p"}
    oracle["checks"] += [{**unwrapped, "feature": "min", "expect": 0.0, "tol": 0.01},
                         {**unwrapped, "feature": "max", "expect": 31.412, "tol": 0.05}]
    oracle["graph_checks"] = [{"kind": "path", "from_ids": ["blocks_complex_to_arg"], "via_ids": ["epy_block"],
                               "to_ids": ["blocks_file_sink"], "group": "f2p"}]
    wrong_src = EPY_UNWRAP.replace("np.where(d < -np.pi, 2 * np.pi, np.where(d > np.pi, -2 * np.pi, 0.0))",
                                   "np.where(d < -np.pi, -2 * np.pi, np.where(d > np.pi, 2 * np.pi, 0.0))")
    wrong = add(ref, "unwrap", "epy_block", _source_code=wrong_src)
    wrong = add(wrong, "unwrapped_sink", "blocks_file_sink", type="float", file="unwrapped.dat")
    wrong = wire(wrong, "blocks_complex_to_arg_0:0->unwrap:0", "unwrap:0->unwrapped_sink:0")
    return _from_spec(
        "gba-test-18", start=("phase_comparator.grc", ref),
        scripted_user={"rules": SAVE_RULES, "max_turns": 2},
        acceptance={
            "ladder": {"rung": "V5",
                       "observation": _with_point("gbf-test-08", point("unwrapped", "unwrapped.dat", "float")),
                       "oracle": oracle},
            "blocks_present": [{"id": "epy_block"}, {"id": "blocks_file_sink", "params": {"file": "unwrapped.dat"}}]},
        golden={"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["blocks_complex_to_arg_0", "blocks_file_sink_0"]}),
            c("query_knowledge", {"query": "embedded python block keep state between work calls", "domain": "docs",
                                  "k": 3}),
            c("change_graph", {"reason": "Add a stateful phase unwrapper writing to unwrapped.dat.",
                               "add_blocks": [{"block_id": "epy_block", "instance_name": "unwrap",
                                               "params": {"_source_code": EPY_UNWRAP}},
                                              {"block_id": "blocks_file_sink", "instance_name": "unwrapped_sink",
                                               "params": {"type": "float", "file": "unwrapped.dat"}}],
                               "add_connections": ["blocks_complex_to_arg_0:0->unwrap:0",
                                                   "unwrap:0->unwrapped_sink:0"]}),
            c("inspect_graph", {"targets": ["unwrap"]}), *TAIL],
            "final_text": ("I added an Embedded Python block that unwraps the phase, keeping the last sample and the "
                           "accumulated 2*pi offset between work() calls, and a File Sink writing it to unwrapped.dat. "
                           "The phase now grows by 2*pi every 0.2 s without jumps; phase.dat is unchanged, and the "
                           "flowgraph is saved.")},
        wrong=wrong)


def _bits_from_bytes(packed: list[int]) -> list[int]:
    return [(b & 0xFF) >> (7 - i) & 1 for b in packed for i in range(8)]


def _decisions_modify() -> dict:
    ref = named(tf_graph("gbf-test-05"), "bpsk_demapper")
    oracle = _tier_f_oracle("gbf-test-05", {"out": "p2p"})
    packed = oracle["checks"][0]["ref"][:4]                      # one period of 32 bits, MSB first
    decisions = [2.0 * b - 1.0 for b in _bits_from_bytes(packed)] * 2
    dec = {"point": "decisions", "group": "f2p"}
    oracle["checks"] += [{**dec, "feature": "equal_prefix", "ref": decisions, "max_offset": 0, "n": 64},
                         {**dec, "feature": "max_abs", "expect": 1.0, "tol": 1e-6},
                         {**dec, "feature": "max_int_dev", "max": 1e-6}]
    chain = [{"block_id": "blocks_char_to_float", "instance_name": "to_float", "params": {"scale": "1", "vlen": "1"}},
             {"block_id": "blocks_multiply_const_vxx", "instance_name": "times_two",
              "params": {"type": "float", "const": "2"}},
             {"block_id": "blocks_add_const_vxx", "instance_name": "minus_one",
              "params": {"type": "float", "const": "-1"}},
             {"block_id": "blocks_file_sink", "instance_name": "decisions_sink",
              "params": {"type": "float", "file": "decisions.dat"}}]
    wrong = add(ref, "to_float", "blocks_char_to_float", scale="1", vlen="1")
    wrong = add(wrong, "decisions_sink", "blocks_file_sink", type="float", file="decisions.dat")
    wrong = wire(wrong, "slicer:0->to_float:0", "to_float:0->decisions_sink:0")
    return _from_spec(
        "gba-test-19", start=("bpsk_demapper.grc", ref),
        scripted_user={"rules": SAVE_RULES, "max_turns": 2},
        acceptance={
            "ladder": {"rung": "V5",
                       "observation": _with_point("gbf-test-05", point("decisions", "decisions.dat", "float")),
                       "oracle": oracle},
            "blocks_present": [{"id": "blocks_file_sink", "params": {"file": "decisions.dat"}}]},
        golden={"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["slicer", "pack"]}),
            c("query_knowledge", {"query": "char to float", "domain": "catalog", "k": 3}),
            c("change_graph", {"reason": "Record the hard decisions as +1/-1 float samples in decisions.dat.",
                               "add_blocks": chain,
                               "add_connections": ["slicer:0->to_float:0", "to_float:0->times_two:0",
                                                   "times_two:0->minus_one:0", "minus_one:0->decisions_sink:0"]}),
            c("inspect_graph", {"targets": ["to_float", "times_two", "minus_one"]}), *TAIL],
            "final_text": ("I tapped the slicer's output: Char To Float, multiply by 2 and add -1 map each bit to +1.0 "
                           "or -1.0, and a File Sink writes the decisions as float samples to decisions.dat. bits.dat is "
                           "unchanged, and the flowgraph is saved.")},
        wrong=wrong)


def _neighbour_channel_modify() -> dict:
    ref = named(tf_graph("gbf-test-18"), "cw_beacon_rx")
    oracle = _tier_f_oracle("gbf-test-18", {"channel": "p2p", "audio": "p2p"})
    nb = {"point": "neighbour", "skip": 1200, "n": 7200, "group": "f2p"}
    # the carrier at 0 Hz: peak_freq removes the mean first, so the band check (no mean removed) places it
    oracle["checks"] += [{**nb, "feature": "band_fraction", "fs": 12000, "lo": -100, "hi": 100, "min": 0.95},
                         {**nb, "feature": "rms", "expect": 1.0, "rel_tol": 0.05}]
    lo = {"type": "complex", "samp_rate": "samp_rate", "waveform": "analog.GR_COS_WAVE", "freq": "-25000", "amp": "1"}
    lpf = {"type": "fir_filter_ccf", "decim": "decim", "gain": "1", "samp_rate": "samp_rate", "cutoff_freq": "3000",
           "width": "1000"}
    wrong = add(ref, "lo_25k", "analog_sig_source_x", **{**lo, "freq": "25000"})
    wrong = add(wrong, "mix_25k", "blocks_multiply_xx", type="complex")
    wrong = add(wrong, "lpf_25k", "low_pass_filter", **lpf)
    wrong = add(wrong, "neighbour_sink", "blocks_file_sink", type="complex", file="neighbour.dat")
    wrong = wire(wrong, "blocks_add_xx_0:0->mix_25k:0", "lo_25k:0->mix_25k:1", "mix_25k:0->lpf_25k:0",
                 "lpf_25k:0->neighbour_sink:0")
    return _from_spec(
        "gba-test-20", start=("cw_beacon_rx.grc", ref),
        scripted_user={"rules": SAVE_RULES, "max_turns": 2},
        acceptance={
            "ladder": {"rung": "V5",
                       "observation": _with_point("gbf-test-18", point("neighbour", "neighbour.dat", "complex")),
                       "oracle": oracle},
            "blocks_present": [{"id": "blocks_file_sink", "params": {"file": "neighbour.dat", "type": "complex"}}]},
        golden={"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["blocks_multiply_xx_0", "low_pass_filter_0", "blocks_add_xx_0"]}),
            c("query_knowledge", {"query": "frequency xlating fir filter", "domain": "catalog", "k": 3}),
            c("change_graph", {"reason": "Add a second channel for the +25 kHz carrier, written to neighbour.dat.",
                               "add_blocks": [
                                   {"block_id": "analog_sig_source_x", "instance_name": "lo_25k", "params": lo},
                                   {"block_id": "blocks_multiply_xx", "instance_name": "mix_25k",
                                    "params": {"type": "complex"}},
                                   {"block_id": "low_pass_filter", "instance_name": "lpf_25k", "params": lpf},
                                   {"block_id": "blocks_file_sink", "instance_name": "neighbour_sink",
                                    "params": {"type": "complex", "file": "neighbour.dat"}}],
                               "add_connections": ["blocks_add_xx_0:0->mix_25k:0", "lo_25k:0->mix_25k:1",
                                                   "mix_25k:0->lpf_25k:0", "lpf_25k:0->neighbour_sink:0"]}),
            c("inspect_graph", {"targets": ["lo_25k", "lpf_25k"]}), *TAIL],
            "final_text": ("I added a second channel from the Add block: a complex -25 kHz tone shifts the neighbouring "
                           "carrier to 0 Hz, and a Low Pass Filter (cutoff 3 kHz, transition 1 kHz, decimation 8, as for "
                           "the beacon) writes it at 12 kHz to neighbour.dat. The existing outputs are unchanged, and the "
                           "flowgraph is saved.")},
        wrong=wrong)


# --------------------------------------------------------------------------- knowledge
def _start_fact(name: str, block: dict, param: str, match: str = "number", **extra: Any) -> dict:
    fact = {"name": name, "source": {"graph": "start", "block": block, "param": param, **extra}, "match": match}
    return {**fact, "rel_tol": 1e-6} if match == "number" else fact


def _fm_knowledge() -> dict:
    ref = named(tf_graph("gbf-test-14"), "nbfm_loopback")
    return _from_spec(
        "gba-test-21", start=("nbfm_loopback.grc", ref),
        scripted_user={"rules": [], "max_turns": 1},
        acceptance={"graph_unchanged": True, "answer": {"facts": [
            _start_fact("peak frequency deviation", {"name": "deviation"}, "value"),
            _start_fact("audio sample rate", {"name": "audio_rate"}, "value"),
            _start_fact("audio filter decimation", {"name": "audio_lpf"}, "decim")]}},
        golden={"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["deviation", "audio_rate", "audio_lpf"]}),
            c("query_knowledge", {"query": "low pass filter decimation", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "fm deviation", "domain": "docs", "k": 3}),
            c("list_directory"),
            c("read_file", {"path": "nbfm_loopback.grc", "limit": 80}),
            c("file_info", {"path": "nbfm_loopback.grc"}),
            c("generate_python")],
            "final_text": ("The FM signal has a peak frequency deviation of 2500 Hz (2.5 kHz). The audio side runs at "
                           "8000 Hz (8 kHz). The audio filter decimates by 5 (int(samp_rate/audio_rate) = 40000/8000). I "
                           "did not change the flowgraph.")},
        wrong=setp(ref, "deviation", value="5000"),
        wrong_json={"final_text": ("The peak deviation is 2500 Hz, the audio sample rate is 40000 Hz and the audio "
                                   "filter decimates by 8. I did not change the flowgraph.")})


def _command_port_knowledge() -> dict:
    ref = named(tf_graph("gbf-test-06"), "sig_source_msg_ports")
    msg = {"graph": "start", "block": {"name": "blocks_message_strobe_0"}, "param": "msg", "use": "raw"}
    return _from_spec(
        "gba-test-22", start=("sig_source_msg_ports.grc", ref),
        scripted_user={"rules": [], "max_turns": 1},
        acceptance={"graph_unchanged": True, "answer": {"facts": [
            {"name": "command port", "source": {"block_yaml": "analog_sig_source_x", "path": "inputs[domain=message].id"},
             "match": "text"},
            {"name": "commanded frequency", "source": {**msg, "regex": r'"freq"\s*:\s*([-+0-9.eE]+)'},
             "match": "number", "rel_tol": 1e-6},
            {"name": "commanded amplitude", "source": {**msg, "regex": r'"ampl"\s*:\s*([-+0-9.eE]+)'},
             "match": "number", "rel_tol": 1e-6},
            _start_fact("average command interval", {"name": "blocks_message_strobe_random_0"}, "mean")]}},
        golden={"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["blocks_message_strobe_0", "blocks_message_strobe_random_0",
                                            "analog_sig_source_x_0"]}),
            c("query_knowledge", {"query": "message strobe random", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "signal source cmd message port", "domain": "docs", "k": 3}),
            c("read_file", {"path": "sig_source_msg_ports.grc", "limit": 120}),
            c("list_directory"),
            c("file_info", {"path": "sig_source_msg_ports.grc"}),
            c("generate_python")],
            "final_text": ("The Signal Source receives the commands on its message input port cmd. A Message Strobe "
                           "sends {freq: 4200, ampl: 1.2} every 500 ms to a Message Strobe Random, which forwards it to "
                           "the Signal Source, so the commands set the frequency to 4200 Hz and the amplitude to 1.2. The "
                           "random strobe's mean period is 100 ms, so on average a command arrives every 100 ms. I did "
                           "not change the flowgraph.")},
        wrong=setp(ref, "blocks_message_strobe_random_0", mean="1000"),
        wrong_json={"final_text": ("The commands arrive on the Signal Source's port freq; they set the frequency to "
                                   "1500 Hz and the amplitude to 0.8, on average every 500 ms. I did not change the "
                                   "flowgraph.")})


# --------------------------------------------------------------------------- save a block
EPY_SOFT_LIMITER = '''"""Soft limiter: tanh(drive * x) / tanh(drive), so that +-1 stays +-1 and larger inputs saturate softly."""

import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """Soft limiter with an adjustable drive."""

    def __init__(self, drive=2.5):
        gr.sync_block.__init__(self, name="Soft Limiter", in_sig=[np.float32], out_sig=[np.float32])
        self.drive = float(drive)

    def work(self, input_items, output_items):
        output_items[0][:] = np.tanh(self.drive * input_items[0]) / np.tanh(self.drive)
        return len(output_items[0])
'''


def _save_soft_limiter() -> dict:
    ref = named(tf_graph("gbf-test-10"), "soft_limited_tone")
    start = drop(ref, "gain")
    start = add(start, "limiter", "epy_block", _source_code=EPY_SOFT_LIMITER, drive="2.5")
    start = wire(start, "hpf:0->limiter:0", "limiter:0->clean_sink:0")
    return _from_spec(
        "gba-test-23", start=("soft_limited_tone.grc", start),
        scripted_user={"rules": [{"pattern": r"\b(category|overwrite)\b", "reply": "Use the default."}],
                       "max_turns": 2},
        acceptance={
            "hier_block": {"block_id": "soft_limiter", "label": "Soft Limiter", "inputs": 1, "outputs": 1,
                           "params": ["drive"]},
            "graph_unchanged": True,
            "ladder": {"rung": "V3"}},
        golden={"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["limiter"]}),
            c("query_knowledge", {"query": "save embedded python block to library", "domain": "docs", "k": 3}),
            c("generate_python"),
            c("list_directory"),
            c("save_graph"),
            c("save_block", {"instance_name": "limiter", "block_id": "soft_limiter", "label": "Soft Limiter"}),
            RUN,
            c("get_run_log")],
            "final_text": ("I saved the Embedded Python block 'limiter' to the block library as soft_limiter, with the "
                           "label Soft Limiter and its drive parameter. The flowgraph itself is unchanged.")},
        wrong=start)


EPY_DB = '''"""Level in dB: 20 * log10(|x|), with a floor of -120 dB (for x = 0)."""

import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """Converts a signal into its level in dB."""

    def __init__(self):
        gr.sync_block.__init__(self, name="dB Level", in_sig=[np.float32], out_sig=[np.float32])

    def work(self, input_items, output_items):
        output_items[0][:] = 20.0 * np.log10(np.maximum(np.abs(input_items[0]), 1e-6))
        return len(output_items[0])
'''


def _db_level_block() -> dict:
    ref = named(tf_graph("gbf-test-04"), "dac_test_signal")
    oracle = _tier_f_oracle("gbf-test-04", {"out": "p2p"})
    level = {"point": "level", "skip": 4800, "group": "f2p"}
    oracle["checks"] += [{**level, "feature": "max", "min": -0.46, "max": -0.44},
                         {**level, "feature": "min", "min": -16.5, "max": -16.4}]
    oracle["graph_checks"] = [{"kind": "path", "from_ids": ["blocks_add_const_vxx"], "via_ids": ["epy_block"],
                               "to_ids": ["blocks_file_sink"], "group": "f2p"}]
    wrong = add(ref, "db", "epy_block", _source_code=EPY_DB.replace("20.0 * np.log10", "10.0 * np.log10"))
    wrong = add(wrong, "level_sink", "blocks_file_sink", type="float", file="level_db.dat")
    wrong = wire(wrong, "offset:0->db:0", "db:0->level_sink:0")
    return _from_spec(
        "gba-test-24", start=("dac_test_signal.grc", ref),
        scripted_user={"rules": [*SAVE_RULES, {"pattern": r"\b(category|overwrite)\b", "reply": "Use the default."}],
                       "max_turns": 2},
        acceptance={
            "hier_block": {"block_id": "db_level", "label": "dB Level", "inputs": 1, "outputs": 1},
            "ladder": {"rung": "V5", "observation": _with_point("gbf-test-04", point("level", "level_db.dat", "float")),
                       "oracle": oracle},
            "blocks_present": [{"id": "epy_block"}, {"id": "blocks_file_sink", "params": {"file": "level_db.dat"}}]},
        golden={"calls": [
            c("inspect_graph"),
            c("query_knowledge", {"query": "embedded python block", "domain": "docs", "k": 3}),
            c("change_graph", {"reason": "Add a dB-level Embedded Python block writing to level_db.dat.",
                               "add_blocks": [{"block_id": "epy_block", "instance_name": "db_level",
                                               "params": {"_source_code": EPY_DB}},
                                              {"block_id": "blocks_file_sink", "instance_name": "level_sink",
                                               "params": {"type": "float", "file": "level_db.dat"}}],
                               "add_connections": ["offset:0->db_level:0", "db_level:0->level_sink:0"]}),
            c("inspect_graph", {"targets": ["db_level"]}),
            c("generate_python"),
            c("save_graph"),
            c("save_block", {"instance_name": "db_level", "block_id": "db_level", "label": "dB Level"}),
            RUN,
            c("get_run_log")],
            "final_text": ("I added an Embedded Python block after the offset that outputs 20 * log10(|x|) with a -120 dB "
                           "floor, and a File Sink writing it to level_db.dat; the level runs from about -16.5 dB to "
                           "-0.45 dB. I saved the block to the library as db_level (label dB Level) and saved the "
                           "flowgraph. dac_test.dat is unchanged.")},
        wrong=wrong)


# --------------------------------------------------------------------------- file operations
PAM_SINKS = {"tx_signal_sink": "tx_signal.dat", "rx_symbols_sink": "rx_symbols.dat",
             "timing_error_sink": "timing_error.dat", "timing_rate_sink": "timing_rate.dat",
             "timing_phase_sink": "timing_phase.dat"}


def _move_outputs_file_ops() -> dict:
    ref = named(tf_graph("gbf-test-17"), "pam_timing")
    wrong = ref
    for name, file in list(PAM_SINKS.items())[:4]:                 # the fifth file left in the project folder
        wrong = setp(wrong, name, file=f"data/{file}")
    run_1s = c("run_flowgraph", {"action": "start", "stop_after_seconds": 1})
    return _from_spec(
        "gba-test-25", start=("pam_timing.grc", ref),
        scripted_user={"rules": SAVE_RULES, "max_turns": 2},
        acceptance={
            "files": [{"path": f"data/{file}"} for file in PAM_SINKS.values()],
            "blocks_present": [{"id": "blocks_file_sink", "params": {"file": f"data/{file}"}}
                               for file in PAM_SINKS.values()],
            "ladder": {"rung": "V4",
                       "observation": observation([point("symbols", "data/rx_symbols.dat", "complex"),
                                                   point("tx", "data/tx_signal.dat", "complex")], 8192,
                                                  "complex samples in data/rx_symbols.dat and data/tx_signal.dat")}},
        golden={"calls": [
            c("inspect_graph"),
            c("list_directory"),
            c("create_directory", {"path": "data"}),
            c("change_graph", {"reason": "Write the five output files into the data folder.",
                               "update_params": [{"instance_name": n, "params": {"file": f"data/{f}"}}
                                                 for n, f in PAM_SINKS.items()]}),
            c("inspect_graph", {"targets": list(PAM_SINKS)}),
            c("save_graph"),
            run_1s,
            c("get_run_log"),
            c("list_directory", {"path": "data"})],
            "final_text": ("I created the data folder and pointed all five File Sinks there, keeping their names "
                           "(data/tx_signal.dat, data/rx_symbols.dat, data/timing_error.dat, data/timing_rate.dat and "
                           "data/timing_phase.dat). I saved the flowgraph and ran it for one second; the five files are "
                           "now in data/.")},
        wrong=wrong)


def _copy_outputs_manifest() -> dict:
    ref = named(tf_graph("gbf-test-09"), "sawtooth_peaks")
    manifest = ("cd outputs && { echo file,bytes; for f in *.dat; do echo \"$f,$(stat -c %s \"$f\")\"; done; } "
                "> manifest.csv")
    return _from_spec(
        "gba-test-26", start=("sawtooth_peaks.grc", ref),
        scripted_user={"rules": [], "max_turns": 1},
        acceptance={
            "graph_unchanged": True,
            "files": [{"path": "outputs/peaks.dat"}, {"path": "outputs/average.dat"},
                      {"path": "outputs/manifest.csv",
                       "csv_sizes": {"dir": "outputs", "header": ["file", "bytes"],
                                     "names": ["peaks.dat", "average.dat"]}}]},
        golden={"calls": [
            c("inspect_graph"),
            c("run_flowgraph", {"action": "start", "stop_after_seconds": 1}),
            c("get_run_log"),
            c("list_directory"),
            c("create_directory", {"path": "outputs"}),
            c("run_command", {"command": "cp *.dat outputs/"}),
            c("run_command", {"command": manifest}),
            c("read_file", {"path": "outputs/manifest.csv"}),
            c("list_directory", {"path": "outputs"})],
            "final_text": ("I ran the flowgraph for one second; it wrote peaks.dat and average.dat. I created the "
                           "outputs folder, copied both files into it and wrote outputs/manifest.csv with the header "
                           "file,bytes and each copy's size in bytes. The flowgraph is unchanged.")},
        wrong=ref,
        wrong_json={"files": {"outputs/peaks.dat": "0" * 64, "outputs/average.dat": "0" * 64,
                              "outputs/manifest.csv": "file,bytes\npeaks.dat,64\n"}})


# --------------------------------------------------------------------------- run and verify
def _rms_measure_verify() -> dict:
    ref = named(tf_graph("gbf-test-14"), "nbfm_loopback")
    script = ("python3 -c \"import numpy as np; x=np.fromfile('recovered.dat',dtype=np.float32)[800:]; "
              "print(len(x), float(np.sqrt(np.mean(x*x))))\"")
    return _from_spec(
        "gba-test-27", start=("nbfm_loopback.grc", ref),
        scripted_user={"rules": [], "max_turns": 1},
        acceptance={
            "graph_unchanged": True,
            "files": [{"path": "recovered.dat"}],
            "answer": {"facts": [{"name": "rms of the recovered audio",
                                  "source": {"file": "recovered.dat", "dtype": "float32", "skip": 800,
                                             "feature": "rms"},
                                  "match": "number", "rel_tol": 0.03}]}},
        golden={"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["audio_lpf", "recovered"]}),
            RUN,
            c("get_run_log"),
            c("list_directory"),
            c("file_info", {"path": "recovered.dat"}),
            c("run_command", {"command": script}),
            c("query_knowledge", {"query": "file sink raw float32", "domain": "docs", "k": 3})],
            "final_text": ("I ran the flowgraph for two seconds and measured recovered.dat (float32 samples at 8 kHz), "
                           "skipping the first 0.1 s (800 samples): the RMS level of the recovered audio is about 1.77. "
                           "The flowgraph is unchanged.")},
        wrong=ref,
        wrong_json={"final_text": ("I ran the flowgraph and measured recovered.dat: its RMS level is about 2.5. The "
                                   "flowgraph is unchanged.")})


def _pll_settle_verify() -> dict:
    ref = named(tf_graph("gbf-test-02"), "pll_carrier_recovery")
    fft = ("python3 -c \"import numpy as np; x=np.fromfile('carrier.dat',dtype=np.complex64)[16000:32000]; "
           "f=np.fft.fftfreq(len(x),1/32000); print(len(x), f[np.argmax(np.abs(np.fft.fft(x)))])\"")
    return _from_spec(
        "gba-test-28", start=("pll_carrier_recovery.grc", ref),
        scripted_user={"rules": [], "max_turns": 1},
        acceptance={
            "graph_unchanged": True,
            "files": [{"path": "carrier.dat"}],
            "answer": {"facts": [_start_fact("settled PLL frequency", {"name": "carrier"}, "freq")
                                 | {"rel_tol": 0.01}]}},
        golden={"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["pll"]}),
            RUN,
            c("get_run_log"),
            c("list_directory"),
            c("file_info", {"path": "carrier.dat"}),
            c("run_command", {"command": fft}),
            c("query_knowledge", {"query": "pll refout", "domain": "catalog", "k": 3})],
            "final_text": ("I ran the flowgraph for two seconds and took the spectrum of carrier.dat after 0.5 s: the "
                           "PLL's output settles at 2400 Hz (2.4 kHz), the carrier's frequency. The flowgraph is "
                           "unchanged.")},
        wrong=ref,
        wrong_json={"final_text": ("I ran the flowgraph and analysed carrier.dat: the PLL's output settles at 4000 Hz. "
                                   "The flowgraph is unchanged.")})


# --------------------------------------------------------------------------- abstention
def _morse_block_abstention() -> dict:
    ref = named(tf_graph("gbf-test-18"), "cw_beacon_rx")
    return _from_spec(
        "gba-test-29", start=("cw_beacon_rx.grc", ref),
        scripted_user={"rules": [], "max_turns": 1},
        acceptance={
            "graph_unchanged": True,
            "blocks_absent": [{"ids": ["satellites_morse_decoder", "satellites_morse_deframer"]}],
            "answer": {"facts": [_start_fact("beacon carrier frequency", {"name": "analog_sig_source_x_0"}, "freq")]}},
        golden={"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["analog_sig_source_x_0"]}),
            c("query_knowledge", {"query": "morse decoder", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "satellites", "domain": "catalog", "k": 3}),
            c("run_command", {"command": "python3 -c 'import satellites'"}),
            c("list_directory"),
            c("read_file", {"path": "cw_beacon_rx.grc", "limit": 60}),
            c("query_knowledge", {"query": "signal source", "domain": "catalog", "k": 3})],
            "final_text": ("The Morse Decoder block of gr-satellites is not available here: the block catalog has no "
                           "gr-satellites blocks and Python cannot import the satellites module. So I left the flowgraph "
                           "unchanged. The beacon's carrier sits at 21000 Hz (21 kHz) in the recording.")},
        wrong=add(ref, "printer", "blocks_message_debug"),
        wrong_json={"final_text": ("gr-satellites is not installed, so I could not add the Morse Decoder. The beacon's "
                                   "carrier is at 19500 Hz.")})


def _zero_width_abstention() -> dict:
    ref = named(tf_graph("gbf-test-10"), "clean_tone")
    return _from_spec(
        "gba-test-30", start=("clean_tone.grc", ref),
        scripted_user={"rules": [], "max_turns": 1},
        acceptance={
            "graph_unchanged": True,
            "answer": {"facts": [_start_fact("current transition width", {"name": "hpf"}, "width")]}},
        golden={"calls": [
            c("inspect_graph"),
            c("inspect_graph", {"targets": ["hpf"]}),
            c("query_knowledge", {"query": "high pass filter transition width", "domain": "catalog", "k": 3}),
            c("query_knowledge", {"query": "fir filter design window transition", "domain": "docs", "k": 3}),
            c("list_directory"),
            c("read_file", {"path": "clean_tone.grc", "limit": 60}),
            c("generate_python"),
            c("file_info", {"path": "clean_tone.grc"})],
            "final_text": ("A transition width of exactly 0 Hz is not possible: an FIR filter would need infinitely many "
                           "taps, and GNU Radio's filter design refuses a width of 0. I left the flowgraph unchanged. Its "
                           "High Pass Filter's current transition width is 200 Hz.")},
        wrong=setp(ref, "hpf", width="1"),
        wrong_json={"final_text": "I set the transition width to 1 Hz, as close to 0 Hz as the filter allows."})


# Where a built task differs from what its spec outlines (shown to the reviewers on the review sheet).
BUILT_NOTES = {
    "gba-test-14": ("Wrong end state: one PLL limit restored and the other left (min = max = +4 kHz); the spec's "
                    "2 kHz cap still locked to the carrier."),
    "gba-test-15": ("The +/-6.2 kHz band check asks for at least 0.97 (the spec said 0.98): the Bessel power of a "
                    "1.2 kHz tone at 5 kHz deviation (beta 4.17) within +/-6.2 kHz is close to 0.98."),
    "gba-test-16": ("No min/max on average.dat: with alpha 0.01 the running average ripples by more than the "
                    "Tier-F bounds; its mean (1.0 within 5 %) is checked."),
    "gba-test-18": "Wrong end state: an unwrapper with the sign of the 2*pi steps reversed.",
    "gba-test-20": ("No peak_freq check on neighbour.dat: the feature removes the mean, so a carrier at 0 Hz is not "
                    "visible to it; the band check (>= 0.95 of the power within +/-100 Hz, mean kept) places it."),
    "gba-test-22": ("The frequency and amplitude facts are read out of the Message Strobe's message with a regex "
                    "(grader fact source option regex); the port fact from the Signal Source's block YAML."),
    "gba-test-26": "The manifest is checked by the files option csv_sizes: every listed size equals the copy's size.",
    "gba-test-27": ("The RMS fact is measured at grading time from the run's own recovered.dat (grader fact source "
                    "file: the oracle feature rms after 800 samples)."),
}


TASKS = [_tone_in_noise_build, _decimate_select_build, _hpf_cutoff_repair, _pll_limits_repair, _fm_deviation_tune,
         _sawtooth_peaks_tune, _dac_codes_epy, _phase_unwrap_epy, _decisions_modify, _neighbour_channel_modify,
         _fm_knowledge, _command_port_knowledge, _save_soft_limiter, _db_level_block, _move_outputs_file_ops,
         _copy_outputs_manifest, _rms_measure_verify, _pll_settle_verify, _morse_block_abstention,
         _zero_width_abstention]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb tier-a second-set", description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="comma-separated task ids")
    args = ap.parse_args(argv)
    only = set(args.only.split(",")) if args.only else None
    for d in write("test", TASKS, only, generator=GENERATOR):
        print("wrote", d.relative_to(ITEMS.parent))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
