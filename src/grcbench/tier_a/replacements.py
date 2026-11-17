"""Replacements for the Tier-A tasks the review rejected (`./gb tier-a replacements`; pre-registration §10).

The review rejected every knowledge task and three of the four save_block tasks: a competent agent finished
them in 1 to 3 tool calls, below the 8-call floor of eval plan §4.4, and the knowledge tasks passed an answer
that gave the right numbers to the wrong items. The replacements keep the categories and avoid both faults:

- knowledge: a data sheet of the open flowgraph, one line per item. The items come from the graph, from GNU
  Radio's block definitions (blocks the graph uses, at their defaults) and from what the flowgraph writes when it
  runs (measured by the grader from the run's own files). Every fact has a label (grader: answer.facts[].label),
  so a value given to the wrong item fails. A label ends with LABEL_END: it matches only an 'item: value' line
  (or a table cell, 'item = value', a bold or bracketed item name), so a sentence that names the items before the
  data sheet does not stand for the item's line. Where every item is a number and a graph setting is an
  expression, the instruction asks for each value as a single number.
- save_block: the agent writes a new Embedded Python block, wires it into the graph, checks its output and saves
  it to the block library; nothing of the block is in the start graph. The ladder's observation point holds the
  new File Sink by its evaluated file name, so a name set through a variable passes; the scripted user answers a
  question about the category before any other rule (SAVE_BLOCK_RULES).

Every start graph is a Tier-F test or dev reference graph that no other Tier-A task uses, with a Throttle so that
a run of a few seconds writes a few seconds of samples. Slots: 3 knowledge and 2 save_block test tasks, 1 of each
in dev; gba-test-36 (knowledge) and gba-test-37 (save_block) are spares. The first replacements of a category the
review accepts, in id order, fill its slots; any further accepted one is kept unscored (SLOTS).
"""

from __future__ import annotations

import argparse
import copy
import sys
from typing import Any

from ..common.paths import ITEMS
from .first_set import RUN, add, c, named, point, tf_graph, tf_task, wire, write

GENERATOR = ("grcbench.tier_a.replacements", "1")
CC_BY = "CC-BY-4.0 (original GRC-Bench task; licence to be confirmed by the team)"
SAVE_SPECS = ("t13_epy_save_block (builds a duty-cycle gating block and saves it) and GRC-Agent scenario "
              "25_save_epy_block_to_library (a scale-by-three block on an empty graph)")

# category -> split -> the ids that fill its slots, in order (the first accepted fill them; the rest are spares)
SLOTS = {"knowledge": {"test": (["gba-test-31", "gba-test-32", "gba-test-33", "gba-test-36"], 3),
                       "dev": (["gba-dev-13"], 1)},
         "save_block": {"test": (["gba-test-34", "gba-test-35", "gba-test-37"], 2),
                        "dev": (["gba-dev-14"], 1)}}
REPLACEMENT_IDS = frozenset(i for split in SLOTS.values() for ids, _ in split.values() for i in ids)
DATA_SHEET = "Give one line per item, in the form 'item: value', using the item names below."
NUMBERS_SHEET = ("Give one line per item, in the form 'item: value', with each value as a single number, using the "
                 "item names below.")
LABEL_END = r"[\s*_`]*(?:\([^)\n]*\)[\s*_`]*)?[:|=]"     # appended to every fact label: an 'item: value' line
NEAREST = ("Nearest training specs: t09_knowledge_concepts and t25_sync_chain_explain (explain a graph), which ask "
           "for none of these items and measure no run; for the measured items, t08_run_verify_probe (runs a "
           "level-probe chain briefly and reports the measured level)")
NEAREST_METER = (" and t19_rssi_meter_build (a moving-average level meter logged to a file), which build their own "
                 "chain and ask for no settings or block defaults of an open graph.")
NEAREST_PROBE = ", which builds its own chain and asks for no settings or block defaults of an open graph."
SAVE_BLOCK_RULES = [{"pattern": r"\bcategory\b", "reply": "Use the default."},
                    {"pattern": r"\bsave\b", "reply": "Yes, please save it."},
                    {"pattern": r"\b(go ahead|proceed|apply|shall i|should i|do you want|overwrite)\b",
                     "reply": "Yes, go ahead."}]


def _throttled(ref: dict, before: str, after: str, typ: str) -> dict:
    """The graph with a Throttle (samp_rate) on the edge before -> after."""
    g = add(ref, "throttle", "blocks_throttle2", type=typ, samples_per_second="samp_rate", vlen="1",
            ignoretag="True", limit="auto", maximum="0.1")
    return wire(g, f"{before}:0->throttle:0", f"throttle:0->{after}:0", remove=(f"{before}:0->{after}:0",))


def _graph_fact(name: str, label: str, block: str, param: str, rel_tol: float = 1e-6, **extra: Any) -> dict:
    return {"name": name, "label": label, "source": {"graph": "start", "block": {"name": block}, "param": param,
                                                      **extra}, "match": "number", "rel_tol": rel_tol}


def _yaml_fact(name: str, label: str, block_id: str, path: str, match: str = "number") -> dict:
    fact = {"name": name, "label": label, "source": {"block_yaml": block_id, "path": path}, "match": match}
    return {**fact, "rel_tol": 1e-9} if match == "number" else fact


def _file_fact(name: str, label: str, file: str, dtype: str, feature: str, skip: int, tol: dict,
               **opts: Any) -> dict:
    return {"name": name, "label": label, "match": "number", **tol,
            "source": {"file": file, "dtype": dtype, "skip": skip, "feature": feature, **opts}}


def _measure(file: str, dtype: str, skip: int, expr: str) -> dict:
    return c("run_command", {"command": (f"python3 -c \"import numpy as np; x=np.fromfile('{file}',dtype=np.{dtype})"
                                         f"[{skip}:]; print(len(x), {expr})\"")})


PEAK = "float(np.fft.fftfreq(len(x), 1/{fs})[np.argmax(np.abs(np.fft.fft(x)))])"
RMS = "float(np.sqrt(np.mean(np.abs(x)**2)))"
MEAN = "float(np.mean(x))"


def _source(tf_id: str, derivation: str) -> dict:
    return {"kind": "tier_f_graph", "ref": f"{tf_id}/reference.grc", "derivation": derivation}


def _knowledge(task_id: str, tf_id: str, graph_id: str, start: dict, items: list[str], facts: list[dict],
               calls: list[dict], final: str, wrong_final: str, files: list[str], what: str,
               numbers: bool = False, meter: bool = True) -> dict:
    """`numbers`: every item is a number, asked as a single number; `meter`: the graph has a moving-average power
    meter, so t19_rssi_meter_build is among the nearest training specs."""
    facts = [{**f, "label": f["label"] + LABEL_END} for f in facts]
    return {
        "id": task_id, "category": "knowledge", "start": (f"{graph_id}.grc", start),
        "instruction": ("Without changing the open flowgraph, write me a short data sheet of it. Run it for about two "
                        "seconds for the measured items, and ignore the first 0.1 s of each file. "
                        + (NUMBERS_SHEET if numbers else DATA_SHEET) + " "
                        + " ".join(f"{i + 1}. {item}" for i, item in enumerate(items))),
        "scripted_user": {"rules": [], "max_turns": 1},
        "acceptance": {"graph_unchanged": True, "files": [{"path": f} for f in files], "answer": {"facts": facts}},
        "golden": {"calls": calls, "final_text": final},
        "wrong": start, "wrong_json": {"final_text": wrong_final},
        "source": _source(tf_id, "the Tier-F reference graph with a Throttle (samp_rate) after its sources"),
        "cluster_id": f"tier_f:{tf_id}", "license": tf_task(tf_id)["license"],
        "disjointness": (f"A data sheet of {what}: settings, block-definition defaults and measured levels. "
                         + NEAREST + (NEAREST_METER if meter else NEAREST_PROBE)),
    }


# --------------------------------------------------------------------------- knowledge
def _agc_power_sheet() -> dict:
    tf = "gbf-test-149"
    start = _throttled(named(tf_graph(tf), "agc_power_meter"), "sum_of_tones", "hpf", "float")
    facts = [_graph_fact("high-pass cutoff", r"high.?pass cut.?off", "hpf", "cutoff_freq"),
             _graph_fact("AGC reference", r"agc reference", "agc", "reference"),
             _graph_fact("averaging length", r"averaging length", "average", "length"),
             _yaml_fact("AGC default maximum gain", r"agc default max(imum)? gain", "analog_agc_xx",
                        "parameters[id=max_gain].default"),
             _yaml_fact("Moving Average item types", r"moving average (item )?types", "blocks_moving_average_xx",
                        "parameters[id=type].options", match="all_of"),
             _file_fact("received RMS", r"received rms", "received.dat", "complex64", "rms", 4800, {"rel_tol": 0.03}),
             _file_fact("mean power", r"mean power", "second_power_db.dat", "float32", "mean", 4800, {"tol": 0.1})]
    calls = [c("inspect_graph"),
             c("inspect_graph", {"targets": ["hpf", "agc", "average", "to_db"]}),
             c("query_knowledge", {"query": "AGC max gain", "domain": "catalog", "k": 3}),
             c("query_knowledge", {"query": "moving average type", "domain": "catalog", "k": 3}),
             RUN, c("get_run_log"),
             _measure("received.dat", "complex64", 4800, RMS),
             _measure("second_power_db.dat", "float32", 4800, MEAN)]
    return _knowledge(
        "gba-test-31", tf, "agc_power_meter", start,
        ["high-pass cutoff: the high-pass filter's cutoff in Hz;",
         "AGC reference: the level the AGC holds;",
         "averaging length: how many samples the moving average spans;",
         "AGC default maximum gain: the Max Gain that GNU Radio's AGC block takes when it is left at its default;",
         "Moving Average types: every item type GNU Radio's Moving Average block offers;",
         "received RMS: the RMS level of received.dat;",
         "mean power: the mean of second_power_db.dat, in dB."],
        facts, calls,
        ("high-pass cutoff: 6150 Hz\nAGC reference: 0.5\naveraging length: 500\nAGC default maximum gain: 65536\n"
         "Moving Average types: complex, float, int, short\nreceived RMS: 0.5\nmean power: -6.02 dB\n"
         "I did not change the flowgraph."),
        ("high-pass cutoff: 6150 Hz\nAGC reference: 0.5\naveraging length: 500\nAGC default maximum gain: 1\n"
         "Moving Average types: complex, float, int, short\nreceived RMS: 1.0\nmean power: -6.02 dB\n"
         "I did not change the flowgraph."),
        ["received.dat", "second_power_db.dat"], "an AGC-levelled tone and its power meter")


def _shifted_power_sheet() -> dict:
    tf = "gbf-test-150"
    start = _throttled(named(tf_graph(tf), "shifted_power_meter"), "sum_of_tones", "conj", "complex")
    facts = [_graph_fact("low-pass cutoff", r"low.?pass cut.?off", "lpf", "cutoff_freq"),
             _graph_fact("mixer LO frequency", r"lo frequency", "lo", "freq"),
             _graph_fact("averaging length", r"averaging length", "average", "length"),
             _yaml_fact("Moving Average default length", r"moving average default length",
                        "blocks_moving_average_xx", "parameters[id=length].default"),
             _yaml_fact("Log10 default n", r"log10 default n", "blocks_nlog10_ff", "parameters[id=n].default"),
             _file_fact("shifted RMS", r"shifted rms", "shifted.dat", "complex64", "rms", 4800, {"rel_tol": 0.03}),
             _file_fact("mean power", r"mean power", "second_power_db.dat", "float32", "mean", 4800, {"tol": 0.1})]
    calls = [c("inspect_graph"),
             c("inspect_graph", {"targets": ["lpf", "lo", "average", "to_db"]}),
             c("query_knowledge", {"query": "moving average length", "domain": "catalog", "k": 3}),
             c("query_knowledge", {"query": "log10 block n k", "domain": "catalog", "k": 3}),
             RUN, c("get_run_log"),
             _measure("shifted.dat", "complex64", 4800, RMS),
             _measure("second_power_db.dat", "float32", 4800, MEAN)]
    return _knowledge(
        "gba-test-32", tf, "shifted_power_meter", start,
        ["low-pass cutoff: the low-pass filter's cutoff in Hz;",
         "LO frequency: the frequency of the mixer's local oscillator in Hz;",
         "averaging length: how many samples the moving average spans;",
         "Moving Average default length: the length GNU Radio's Moving Average block takes by default;",
         "Log10 default n: the n GNU Radio's Log10 block takes by default;",
         "shifted RMS: the RMS level of shifted.dat;",
         "mean power: the mean of second_power_db.dat, in dB."],
        facts, calls,
        ("low-pass cutoff: 7750 Hz\nLO frequency: 2250 Hz\naveraging length: 2000\nMoving Average default length: "
         "1000\nLog10 default n: 1\nshifted RMS: 0.6\nmean power: -4.44 dB\nI did not change the flowgraph."),
        ("low-pass cutoff: 7750 Hz\nLO frequency: 2250 Hz\naveraging length: 1000\nMoving Average default length: "
         "2000\nLog10 default n: 10\nshifted RMS: 0.6\nmean power: -4.44 dB\nI did not change the flowgraph."),
        ["shifted.dat", "second_power_db.dat"], "a filtered, shifted tone pair and its power meter")


def _fm_power_sheet() -> dict:
    tf = "gbf-test-112"
    start = _throttled(named(tf_graph(tf), "fm_power_meter"), "tone_1", "fm_mod", "float")
    facts = [_graph_fact("modulator sensitivity", r"modulator sensitivity", "fm_mod", "sensitivity", rel_tol=0.01),
             _graph_fact("demodulator gain", r"demodulator gain", "fm_demod", "gain", rel_tol=0.01),
             _graph_fact("averaging length", r"averaging length", "average", "length"),
             _graph_fact("quantizer scale", r"quantizer scale", "to_short", "scale"),
             _yaml_fact("Float To Short default scale", r"float to short default scale", "blocks_float_to_short",
                        "parameters[id=scale].default"),
             _file_fact("mean power", r"mean power", "power_db.dat", "float32", "mean", 4800, {"tol": 0.1}),
             _file_fact("quantized peak frequency", r"quantized peak", "second_quantized.dat", "float32",
                        "peak_freq", 4800, {"tol": 20}, fs=48000.0)]
    calls = [c("inspect_graph"),
             c("inspect_graph", {"targets": ["fm_mod", "fm_demod", "average", "to_short"]}),
             c("query_knowledge", {"query": "float to short scale", "domain": "catalog", "k": 3}),
             c("query_knowledge", {"query": "moving average length", "domain": "catalog", "k": 3}),
             RUN, c("get_run_log"),
             _measure("power_db.dat", "float32", 4800, MEAN),
             _measure("second_quantized.dat", "float32", 4800, PEAK.format(fs=48000))]
    return _knowledge(
        "gba-test-33", tf, "fm_power_meter", start,
        ["modulator sensitivity: the FM modulator's sensitivity in radians per sample;",
         "demodulator gain: the quadrature demodulator's gain;",
         "averaging length: how many samples the moving average spans;",
         "quantizer scale: the scale of the Float To Short block;",
         "Float To Short default scale: the scale GNU Radio's Float To Short block takes by default;",
         "mean power: the mean of power_db.dat, in dB;",
         "quantized peak: the frequency of the strongest component of second_quantized.dat, in Hz."],
        facts, calls,
        ("modulator sensitivity: 0.1309 rad/sample\ndemodulator gain: 3.8197\naveraging length: 1000\n"
         "quantizer scale: 1000\nFloat To Short default scale: 1\nmean power: -13.98 dB\nquantized peak: 6400 Hz\n"
         "I did not change the flowgraph."),
        ("modulator sensitivity: 0.1309 rad/sample\ndemodulator gain: 3.8197\naveraging length: 1000\n"
         "quantizer scale: 1\nFloat To Short default scale: 1000\nmean power: -13.98 dB\nquantized peak: 6400 Hz\n"
         "I did not change the flowgraph."),
        ["power_db.dat", "second_quantized.dat"], "an FM loopback with a power meter and a quantized copy",
        numbers=True)


def _received_sheet() -> dict:
    tf = "gbf-test-72"
    start = _throttled(named(tf_graph(tf), "shifted_receiver"), "sum_of_tones", "hpf", "float")
    facts = [_graph_fact("high-pass cutoff", r"high.?pass cut.?off", "hpf", "cutoff_freq"),
             _graph_fact("LO frequency", r"lo frequency", "lo", "freq"),
             _graph_fact("channel frequency offset", r"channel (frequency )?offset", "channel", "freq_offset",
                         rel_tol=1e-3),
             _yaml_fact("Channel Model default epsilon", r"channel model default epsilon", "channels_channel_model",
                        "parameters[id=epsilon].default"),
             _file_fact("received peak frequency", r"received peak", "received.dat", "complex64", "peak_freq", 3200,
                        {"tol": 20}, fs=32000.0),
             _file_fact("received RMS", r"received rms", "received.dat", "complex64", "rms", 3200, {"rel_tol": 0.03})]
    calls = [c("inspect_graph"),
             c("inspect_graph", {"targets": ["hpf", "lo", "channel"]}),
             c("query_knowledge", {"query": "channel model epsilon", "domain": "catalog", "k": 3}),
             c("query_knowledge", {"query": "channel model frequency offset normalized", "domain": "docs", "k": 3}),
             RUN, c("get_run_log"),
             _measure("received.dat", "complex64", 3200, PEAK.format(fs=32000)),
             _measure("received.dat", "complex64", 3200, RMS)]
    return _knowledge(
        "gba-test-36", tf, "shifted_receiver", start,
        ["high-pass cutoff: the high-pass filter's cutoff in Hz;",
         "LO frequency: the frequency of the mixer's local oscillator in Hz;",
         "channel offset: the channel model's frequency offset, as a fraction of the sample rate;",
         "Channel Model default epsilon: the epsilon GNU Radio's Channel Model block takes by default;",
         "received peak: the frequency of the strongest component of received.dat, in Hz;",
         "received RMS: the RMS level of received.dat, to three significant figures."],
        facts, calls,
        ("high-pass cutoff: 1100 Hz\nLO frequency: 5250 Hz\nchannel offset: 0.0859375\nChannel Model default "
         "epsilon: 1.0\nreceived peak: 11400 Hz\nreceived RMS: 1.03\nI did not change the flowgraph."),
        ("high-pass cutoff: 1100 Hz\nLO frequency: 5250 Hz\nchannel offset: 2750\nChannel Model default "
         "epsilon: 1.0\nreceived peak: 8650 Hz\nreceived RMS: 1.03\nI did not change the flowgraph."),
        ["received.dat"], "a shifted tone pair through a channel model", numbers=True, meter=False)


def _retuned_sheet() -> dict:
    tf = "gbf-dev-37"
    start = named(tf_graph(tf), "retuned_power_meter")       # an L3 graph: it has its Throttle
    msg = {"graph": "start", "block": {"name": "retune"}, "param": "msg", "use": "raw"}
    facts = [_graph_fact("strobe period", r"strobe period", "retune", "period"),
             {"name": "retune frequency", "label": r"retune frequency", "match": "number", "rel_tol": 1e-6,
              "source": {**msg, "regex": r"'freq'\s*:\s*([-+0-9.eE]+)"}},
             _graph_fact("AGC reference", r"agc reference", "agc", "reference"),
             _yaml_fact("Message Strobe input port", r"(strobe )?input port", "blocks_message_strobe",
                        "inputs[domain=message].id", match="text"),
             _file_fact("mean power", r"mean power", "power_db.dat", "float32", "mean", 6400, {"tol": 0.1}),
             _file_fact("levelled RMS", r"levelled rms", "second_levelled.dat", "complex64", "rms", 6400,
                        {"rel_tol": 0.03})]
    calls = [c("inspect_graph"),
             c("inspect_graph", {"targets": ["retune", "tone_1", "agc", "average"]}),
             c("query_knowledge", {"query": "message strobe input port", "domain": "catalog", "k": 3}),
             c("query_knowledge", {"query": "signal source cmd message freq", "domain": "docs", "k": 3}),
             RUN, c("get_run_log"),
             _measure("power_db.dat", "float32", 6400, MEAN),
             _measure("second_levelled.dat", "complex64", 6400, RMS)]
    return {**_knowledge(
        "gba-dev-13", tf, "retuned_power_meter", start,
        ["strobe period: how often the Message Strobe sends its message, in ms;",
         "retune frequency: the frequency the message sets, in Hz;",
         "AGC reference: the level the AGC holds;",
         "Message Strobe input port: the name of the message input port of GNU Radio's Message Strobe block;",
         "mean power: the mean of power_db.dat, in dB;",
         "levelled RMS: the RMS level of second_levelled.dat."],
        facts, calls,
        ("strobe period: 200 ms\nretune frequency: -7800 Hz\nAGC reference: 2\nMessage Strobe input port: set_msg\n"
         "mean power: -6.02 dB\nlevelled RMS: 2.0\nI did not change the flowgraph."),
        ("strobe period: 200 ms\nretune frequency: -4900 Hz\nAGC reference: 2\nMessage Strobe input port: strobe\n"
         "mean power: -6.02 dB\nlevelled RMS: 2.0\nI did not change the flowgraph."),
        ["power_db.dat", "second_levelled.dat"], "a retuned tone, its power meter and an AGC copy"),
        "source": _source(tf, "none (the Tier-F reference graph; it has a Throttle)")}


# --------------------------------------------------------------------------- save a block
def _epy(doc: str, name: str, body: str) -> str:
    return (f'"""{doc}"""\n\nimport numpy as np\nfrom gnuradio import gr\n\n\nclass blk(gr.sync_block):\n'
            f'    """{doc}"""\n\n    def __init__(self):\n        gr.sync_block.__init__(self, name="{name}", '
            f'in_sig=[np.float32], out_sig=[np.float32])\n\n    def work(self, input_items, output_items):\n'
            f'        x = input_items[0]\n        output_items[0][:] = {body}\n        return len(output_items[0])\n')


EPY_SQUARE = _epy("Square law: y = x * x.", "Square Law", "x * x")
EPY_HALF = _epy("Half-wave rectifier: y = max(x, 0).", "Half-Wave Rectifier", "np.maximum(x, 0.0)")
EPY_CLIP = _epy("Clipper: y = x limited to -0.3 .. 0.3.", "Clipper", "np.clip(x, -0.3, 0.3)")
EPY_SIGN = _epy("Sign: y = 1 where x >= 0, else -1.", "Sign", "np.where(x >= 0, 1.0, -1.0)")


def _save_block(task_id: str, tf: str, graph_id: str, start: dict, tap: str, name: str, block_id: str, label: str,
                code: str, wrong_code: str, out_file: str, what: str, checks: list[dict], instruction: str,
                measure: str, final: str, path_from: str | None = None) -> dict:
    """Write a new Embedded Python block after `tap`, record it to `out_file`, save it to the library. The path
    check starts at `path_from` (default `tap`): the block the instruction names before the new one."""
    oracle = copy.deepcopy(tf_task(tf)["oracle"])
    oracle["checks"] = [{**{k: v for k, v in chk.items() if k not in ("requires", "property", "stretch")},
                         "group": "p2p"} for chk in oracle["checks"]]
    oracle["checks"] += [{"point": "added", "skip": 4800, "group": "f2p", **chk} for chk in checks]
    oracle["graph_checks"] = [{"kind": "path", "from_ids": [start_id(start, path_from or tap)],
                               "via_ids": ["epy_block"],
                               "to_ids": ["blocks_file_sink"], "group": "f2p"}]
    obs = copy.deepcopy(tf_task(tf)["observation"])
    obs["points"] = obs["points"] + [point("added", out_file, "float")]
    wrong = add(start, name, "epy_block", _source_code=wrong_code)
    wrong = add(wrong, "added_sink", "blocks_file_sink", type="float", file=out_file)
    wrong = wire(wrong, f"{tap}:0->{name}:0", f"{name}:0->added_sink:0")
    return {
        "id": task_id, "category": "save_block", "start": (f"{graph_id}.grc", start),
        "instruction": instruction,
        "scripted_user": {"rules": copy.deepcopy(SAVE_BLOCK_RULES), "max_turns": 2},
        "acceptance": {
            "hier_block": {"block_id": block_id, "label": label, "inputs": 1, "outputs": 1},
            "ladder": {"rung": "V5", "observation": obs, "oracle": oracle},
            "blocks_present": [{"id": "epy_block"}]},
        "golden": {"calls": [
            c("inspect_graph"),
            c("query_knowledge", {"query": "embedded python block", "domain": "docs", "k": 3}),
            c("change_graph", {"reason": f"Add the {label} Embedded Python block writing to {out_file}.",
                               "add_blocks": [{"block_id": "epy_block", "instance_name": name,
                                               "params": {"_source_code": code}},
                                              {"block_id": "blocks_file_sink", "instance_name": "added_sink",
                                               "params": {"type": "float", "file": out_file}}],
                               "add_connections": [f"{tap}:0->{name}:0", f"{name}:0->added_sink:0"]}),
            c("inspect_graph", {"targets": [name]}),
            c("generate_python"),
            c("save_graph"),
            RUN,
            c("get_run_log"),
            c("run_command", {"command": measure}),
            c("save_block", {"instance_name": name, "block_id": block_id, "label": label})],
            "final_text": final},
        "wrong": wrong,
        "source": _source(tf, "the Tier-F reference graph with a Throttle (samp_rate) after its sources"),
        "cluster_id": f"tier_f:{tf}", "license": tf_task(tf)["license"],
        "disjointness": (f"Writes {what} as an Embedded Python block on an existing graph, checks its output in "
                         f"{out_file} and saves it to the block library. Nearest training specs: {SAVE_SPECS}; "
                         f"a different block, written into a running graph and checked before it is saved."),
    }


def start_id(start: dict, name: str) -> str:
    return next(b["id"] for b in start["blocks"] if b["name"] == name)


def _stats(file: str) -> str:
    return ("python3 -c \"import numpy as np; x=np.fromfile('" + file + "',dtype=np.float32)[4800:]; "
            "print(len(x), float(x.mean()), float(x.min()), float(x.max()), float(np.sqrt(np.mean(x*x))))\"")


def _instruction(what: str, out_file: str, block_id: str, label: str, keep: str) -> str:
    return (f"Add an Embedded Python block to the open flowgraph that {what}, and write its output as float samples "
            f"to a new File Sink with the file name {out_file}. Run the flowgraph to check that the block does what "
            f"it should. Then save the block to the block library as a reusable block with the block id {block_id} "
            f"and the label {label}, and save the flowgraph. Keep {keep} as {'they are' if ' and ' in keep else 'it is'}. "
            f"The flowgraph should keep "
            f"producing samples until it is stopped.")


def _square_law_block() -> dict:
    tf = "gbf-test-24"
    start = _throttled(named(tf_graph(tf), "quantized_tones"), "sum_of_tones", "lpf", "float")
    return _save_block(
        "gba-test-34", tf, "quantized_tones", start, "lpf", "square", "square_law", "Square Law", EPY_SQUARE,
        EPY_SQUARE.replace("x * x", "np.abs(x)"), "squared.dat", "a square-law detector",
        [{"feature": "mean", "min": 0.17, "max": 0.19}, {"feature": "max", "min": 0.34, "max": 0.38},
         {"feature": "min", "min": -0.001, "max": 0.01}],
        _instruction("squares the low-pass filter's output (y = x * x)", "squared.dat", "square_law", "Square Law",
                     "quantized.dat"),
        _stats("squared.dat"),
        ("I added an Embedded Python block after the low-pass filter that outputs x * x, and a File Sink writing it to "
         "squared.dat. A run gives a mean of about 0.18 and a maximum of about 0.36 (the 5.8 kHz tone at 0.6), never "
         "below 0. I saved the block to the library as square_law (label Square Law) and saved the flowgraph. "
         "quantized.dat is unchanged."))


def _half_wave_block() -> dict:
    tf = "gbf-test-46"
    start = _throttled(named(tf_graph(tf), "shifted_tones"), "sum_of_tones", "lpf", "float")
    return _save_block(
        "gba-test-35", tf, "shifted_tones", start, "lpf", "rectify", "half_wave", "Half-Wave Rectifier", EPY_HALF,
        EPY_HALF.replace("np.maximum(x, 0.0)", "np.abs(x)"), "rectified.dat", "a half-wave rectifier",
        [{"feature": "mean", "min": 0.25, "max": 0.275}, {"feature": "max", "min": 0.95, "max": 1.1},
         {"feature": "min", "min": -0.001, "max": 0.01}],
        _instruction("rectifies the low-pass filter's output to its positive half-waves (y = max(x, 0))",
                     "rectified.dat", "half_wave", "Half-Wave Rectifier", "shifted.dat"),
        _stats("rectified.dat"),
        ("I added an Embedded Python block after the low-pass filter that outputs max(x, 0), and a File Sink writing "
         "it to rectified.dat. A run gives a mean of about 0.27 and a maximum just above 1, never below 0. I saved the "
         "block to the library as half_wave (label Half-Wave Rectifier) and saved the flowgraph. shifted.dat is "
         "unchanged."))


def _clipper_block() -> dict:
    tf = "gbf-test-102"
    g = add(named(tf_graph(tf), "mirrored_tone"), "throttle", "blocks_throttle2", type="float",
            samples_per_second="samp_rate", vlen="1", ignoretag="True", limit="auto", maximum="0.1")
    start = wire(g, "tone_1:0->throttle:0", "throttle:0->hilbert:0", "throttle:0->to_short:0",
                 remove=("tone_1:0->hilbert:0", "tone_1:0->to_short:0"))
    return _save_block(
        "gba-test-37", tf, "mirrored_tone", start, "throttle", "clip", "clipper", "Clipper", EPY_CLIP,
        EPY_CLIP.replace("np.clip(x, -0.3, 0.3)", "np.clip(x, -0.5, 0.5)"), "clipped.dat", "a clipper",
        [{"feature": "max", "min": 0.299, "max": 0.301}, {"feature": "min", "min": -0.301, "max": -0.299},
         {"feature": "rms", "min": 0.25, "max": 0.265}],
        _instruction("limits the tone to the range -0.3 to 0.3 (values beyond it are set to the limit)", "clipped.dat",
                     "clipper", "Clipper", "mirrored.dat and second_quantized.dat"),
        _stats("clipped.dat"),
        ("I added an Embedded Python block after the tone that limits it to -0.3 .. 0.3, and a File Sink writing it to "
         "clipped.dat. A run gives a maximum of 0.3, a minimum of -0.3 and an RMS of about 0.257 for the 0.5 tone. I "
         "saved the block to the library as clipper (label Clipper) and saved the flowgraph. mirrored.dat and "
         "second_quantized.dat are unchanged."), path_from="tone_1")


def _sign_block() -> dict:
    tf = "gbf-dev-34"
    start = _throttled(named(tf_graph(tf), "resampled_tones"), "sum_of_tones", "resampler", "float")
    return _save_block(
        "gba-dev-14", tf, "resampled_tones", start, "resampler", "sign", "sign_detect", "Sign Detector", EPY_SIGN,
        EPY_SIGN.replace("np.where(x >= 0, 1.0, -1.0)", "np.tanh(4.0 * x)"), "sign.dat", "a sign detector",
        [{"feature": "max", "min": 0.999, "max": 1.001}, {"feature": "min", "min": -1.001, "max": -0.999},
         {"feature": "rms", "min": 0.999, "max": 1.001}, {"feature": "mean", "min": -0.1, "max": 0.1},
         {"feature": "peak_freq", "expect": 9500, "tol": 95.0, "fs": 96000.0, "n": 16384}],
        _instruction("outputs the sign of the resampler's output (1 where x >= 0, else -1)", "sign.dat",
                     "sign_detect", "Sign Detector", "quantized.dat"),
        _stats("sign.dat"),
        ("I added an Embedded Python block after the resampler that outputs 1 where the input is at least 0 and -1 "
         "elsewhere, and a File Sink writing it to sign.dat. A run gives only the values 1 and -1 (RMS 1). I saved the "
         "block to the library as sign_detect (label Sign Detector) and saved the flowgraph. quantized.dat is "
         "unchanged."))


TEST_TASKS = [_agc_power_sheet, _shifted_power_sheet, _fm_power_sheet, _square_law_block, _half_wave_block,
              _received_sheet, _clipper_block]
DEV_TASKS = [_retuned_sheet, _sign_block]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb tier-a replacements", description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="comma-separated task ids")
    args = ap.parse_args(argv)
    only = set(args.only.split(",")) if args.only else None
    for split, tasks in (("test", TEST_TASKS), ("dev", DEV_TASKS)):
        for d in write(split, tasks, only, generator=GENERATOR):
            print("wrote", d.relative_to(ITEMS.parent))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
