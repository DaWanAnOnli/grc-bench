"""Authoring script for GRC-Bench Tier-F item gbf-dev-07 (family modify, level L2, split dev).

Seed: GNU Radio 3.10.12 example blocks/stream_mux_demo.grc (a dev-split seed in
splits/flowgraph_seed_split.json). Two repeating Vector Sources (float) feed a Stream Mux that
takes a fixed number of samples from each input in turn (time-division multiplexing); the
multiplexed stream goes through a Throttle to a time display, and two more time displays show the
two input streams.

Task: the user wants every multiplexed frame to be preceded by a two-sample header (-1, -2), i.e.
the output becomes header | 4 samples of stream A | 3 samples of stream B, repeated, starting with
a header. The obvious edit is a third input on the Stream Mux (num_inputs 3, lengths [2, 4, 3], the
header on input 0 and the two streams moved to inputs 1 and 2); a cascaded second mux (before or
after the Throttle), a Patterned Interleaver, two Constant Sources, a Vector Insert (before or
after the Throttle) or a Tagged Stream Mux also work (see the variants). The data are exact small
integers, so the oracle compares the captured samples exactly (equal_prefix with no offset
allowed: the instruction states the output starts with the header):
  f2p  all 8190 captured samples of mux_out.dat (455 periods) equal the stated 18-sample pattern
       (-1, -2, 1..7, -1, -2, 8..14) repeated, from its first sample (round-1 review F1: the former
       start/tail windows plus a mean let a wrong header in the middle of the capture pass)
  p2p  stream_a.dat and stream_b.dat still receive streams A and B exactly, from their first sample;
       graph checks: the Vector Source behind stream_a.dat, and the one behind stream_b.dat, each
       reach the File Sink of mux_out.dat through a block that merges several streams ("built from
       streams A and B by multiplexing", round-1 F2 and round-2 G10R-1: a fixed output, with or
       without a merging block, and a header-baked copy of stream A both fail), and the Throttle
       still sits between that merging block and a File Sink ("keep the Throttle": F3)

Level L2: the reference has 8 functional blocks (three Vector Sources, Stream Mux, Throttle, three
File Sinks; variables not counted) and the Stream Mux is a multi-rate block (its output runs at 9/4
of stream A's item rate and 3 times stream B's).

Start graph = the seed exported by `./gb tier-f seed` with its numbers perturbed and its three
observation File Sinks renamed (see PERTURBATION). The seed tool names each File Sink
"<former Qt sink>_<i>"; for this example that collides with the next Qt sink's name, so the File Sink
fed by the mux output ended up named qtgui_time_sink_x_0_0, which was the example's "Input stream 0"
display. The three File Sinks are therefore given clear names:
  out  mux_out.dat   float  (former "Output stream" display, after the Throttle)
  a    stream_a.dat  float  (former "Input stream 0" display, stream A before the mux)
  b    stream_b.dat  float  (former "Input stream 1" display, stream B before the mux)

Regenerate (WSL, from the grc-bench folder; deterministic):
    PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen \
        ~/grc-agent/.venv/bin/python items/tier_f/dev/gbf-dev-07/build.py
then run the gates:
    ./gb tier-f gates items/tier_f/dev/gbf-dev-07
"""

from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from grcbench.common.io import read_json, sha256_file           # noqa: E402
from grcbench.common.paths import GNURADIO_EXAMPLES, SPLITS, WORK  # noqa: E402
from grcbench.tier_f import edit, grc, grcfile, taskio           # noqa: E402
from grcbench.tier_f.seedtools import headless_seed              # noqa: E402

TASK_ID = "gbf-dev-07"
SPLIT = "dev"
SEED = "blocks/stream_mux_demo.grc"

# ----------------------------------------------------------------------------- numbers
# (every number below is stated in the instruction)
STREAM_A = (1, 2, 3, 4, 8, 9, 10, 11)     # perturbed from (0, 1, 2, 5, 6, 7)
STREAM_B = (5, 6, 7, 12, 13, 14)          # perturbed from (3, 4, 8, 9)
LEN_A, LEN_B = 4, 3                       # mux lengths, perturbed from [3, 2]
HEADER = (-1, -2)                         # the requested frame header

assert len(STREAM_A) % LEN_A == 0 and len(STREAM_B) % LEN_B == 0
assert len(STREAM_A) // LEN_A == len(STREAM_B) // LEN_B == 2


def tdm(streams: list[tuple[int, ...]], lengths: list[int], n_out: int) -> list[int]:
    """What a Stream Mux with these (repeating) inputs and lengths outputs, first n_out items."""
    pos = [0] * len(streams)
    out: list[int] = []
    while len(out) < n_out:
        for i, (s, n) in enumerate(zip(streams, lengths)):
            for _ in range(n):
                out.append(s[pos[i] % len(s)])
                pos[i] += 1
    return out[:n_out]


PERIOD = len(HEADER) * 2 + len(STREAM_A) + len(STREAM_B)    # 18 output samples = two frames
START_OUT = tdm([STREAM_A, STREAM_B], [LEN_A, LEN_B], 14)
assert START_OUT == list(range(1, 15))                       # the start graph writes 1..14 repeated
TARGET = tdm([HEADER, STREAM_A, STREAM_B], [len(HEADER), LEN_A, LEN_B], PERIOD)
assert TARGET == [-1, -2, 1, 2, 3, 4, 5, 6, 7, -1, -2, 8, 9, 10, 11, 12, 13, 14]

N_ITEMS = 8192
WHOLE_N = (N_ITEMS // PERIOD) * PERIOD                       # 8190 samples = 455 whole periods
REF_OUT = TARGET * (WHOLE_N // PERIOD)                       # the whole capture, exact
REF_A = list(STREAM_A) * 8                                   # 64 samples
REF_B = list(STREAM_B) * 10                                  # 60 samples
assert len(REF_OUT) == WHOLE_N <= N_ITEMS

# ----------------------------------------------------------------------------- names
SRC_A, SRC_B = "blocks_vector_source_x_0", "blocks_vector_source_x_0_0"
MUX, THR = "blocks_stream_mux_0", "blocks_throttle2_0"
SINK_OUT, SINK_A, SINK_B = "mux_out", "stream_a", "stream_b"
FILES = {"out": "mux_out.dat", "a": "stream_a.dat", "b": "stream_b.dat"}
HDR = "frame_header"

# Blocks that merge two or more input streams into one output stream (GNU Radio 3.10.12 catalog,
# curated from every block with several stream inputs; sinks, hardware blocks and decoders left
# out). Any legitimate way to put the header and the samples of streams A and B into one stream
# goes through one of them (or through an Embedded Python block). The graph checks built on this list
# are anchored on the observation points (round 2, G10R-1): the source behind stream_a.dat (and the
# one behind stream_b.dat) must reach mux_out.dat through one of these blocks, so a fixed output routed
# through a merging block, or a header-baked copy of stream A, fails (probes x_doctored_copy_of_a,
# x_hardcoded_via_1in_mux, x_hardcoded_via_add_zero).
MERGERS = ["blocks_stream_mux", "blocks_tagged_stream_mux", "blocks_patterned_interleaver", "blocks_interleave",
           "blocks_streams_to_stream", "blocks_streams_to_vector", "blocks_selector", "blocks_add_xx",
           "blocks_sub_xx", "blocks_multiply_xx", "blocks_divide_xx", "blocks_max_xx", "blocks_min_xx",
           "blocks_and_xx", "blocks_or_xx", "blocks_xor_xx", "blocks_multiply_matrix_xx",
           "blocks_sample_and_hold_xx", "blocks_tag_share", "blocks_float_to_complex", "epy_block"]
THROTTLES = ["blocks_throttle2", "blocks_throttle"]


# ----------------------------------------------------------------------------- flowgraph helpers
def blk(data: dict, name: str) -> dict:
    return next(b for b in data["blocks"] if b["name"] == name)


def setp(data: dict, name: str, **params) -> dict:
    d = copy.deepcopy(data)
    blk(d, name)["parameters"].update({k: str(v) for k, v in params.items()})
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


def vec(values) -> str:
    return "(" + ", ".join(str(v) for v in values) + ")"


def vector_source(name: str, values, **extra) -> dict:
    p = {"type": "float", "vector": vec(values) if not isinstance(values, str) else values}
    p.update({k: str(v) for k, v in extra.items()})
    return {"name": name, "id": "blocks_vector_source_x", "parameters": p}


def without_conns(data: dict, pred) -> dict:
    d = copy.deepcopy(data)
    d["connections"] = [c for c in d["connections"] if not pred(c)]
    return d


# ----------------------------------------------------------------------------- start / reference
def build_start() -> dict:
    seed, sinks = headless_seed(SEED)
    assert len(sinks) == 3 and all(s["type"] == "float" for s in sinks), sinks
    d = copy.deepcopy(seed)
    d["options"]["parameters"] = {"id": "stream_mux_demo", "generate_options": "no_gui",
                                  "output_language": "python"}
    # identify the three observation File Sinks by what feeds them (not by their colliding names)
    feeder = {c[2]: c[0] for c in d["connections"]}
    rename = {}
    for s in sinks:
        src = feeder[s["name"]]
        rename[s["name"]] = {THR: SINK_OUT, SRC_A: SINK_A, SRC_B: SINK_B}[src]
    assert sorted(rename.values()) == sorted([SINK_OUT, SINK_A, SINK_B]), rename
    for b in d["blocks"]:
        if b["name"] in rename:
            b["name"] = rename[b["name"]]
            b["parameters"]["file"] = f"{b['name']}.dat"
    d["connections"] = [[rename.get(c[0], c[0]), c[1], rename.get(c[2], c[2]), c[3]] for c in d["connections"]]
    # perturbed numbers
    assert blk(d, SRC_A)["parameters"]["vector"] == "(0, 1, 2, 5, 6, 7)"
    assert blk(d, SRC_B)["parameters"]["vector"] == "(3, 4, 8, 9)"
    assert blk(d, MUX)["parameters"]["lengths"] == "[3, 2]"
    d = setp(d, SRC_A, vector=vec(STREAM_A))
    d = setp(d, SRC_B, vector=vec(STREAM_B))
    d = setp(d, MUX, lengths=f"[{LEN_A}, {LEN_B}]")
    return tidy(d)


PERTURBATION = {"perturbed": True, "changes": [
    {"what": "stream A (Vector Source blocks_vector_source_x_0, vector)", "original": "(0, 1, 2, 5, 6, 7)",
     "new": vec(STREAM_A)},
    {"what": "stream B (Vector Source blocks_vector_source_x_0_0, vector)", "original": "(3, 4, 8, 9)",
     "new": vec(STREAM_B)},
    {"what": "Stream Mux lengths (samples taken from A, then from B)", "original": "[3, 2]",
     "new": f"[{LEN_A}, {LEN_B}]"},
    {"what": "Qt GUI time sinks 'Output stream', 'Input stream 0', 'Input stream 1'",
     "original": "qtgui_time_sink_x_0, qtgui_time_sink_x_0_0, qtgui_time_sink_x_0_0_0 (GUI)",
     "new": ("File Sinks mux_out (mux_out.dat), stream_a (stream_a.dat), stream_b (stream_b.dat); the seed "
             "tool's own names for them collide with the Qt sinks' names and were replaced")},
    {"what": "generate_options", "original": "qt_gui", "new": "no_gui (seed tool)"},
    {"what": "options block metadata (conversion)",
     "original": ("author 'David Pi', copyright 'GNU GPL 3.0+', title 'Stream Mux demo', description "
                  "'Demonstration of the Stream Mux block', category '[GRC Hier Blocks]', run_options 'prompt', "
                  "placement, sizing_mode, catch_exceptions"),
     "new": ("dropped: the options block keeps only id, generate_options and output_language (authorship and "
             "licence are recorded in the task's license field and notes)")},
    {"what": "catalog-default parameters and GUI states (conversion)", "original": "written out in the example",
     "new": "omitted (compact form; GRC fills in the same defaults, checked by grc.compact_equivalent)"},
]}


def reference_full(start: dict) -> dict:
    """Third Stream Mux input: header on input 0, streams A and B moved to inputs 1 and 2."""
    d = without_conns(start, lambda c: c[2] == MUX)
    d["blocks"].append(vector_source(HDR, HEADER))
    d = setp(d, MUX, num_inputs=3, lengths=f"[{len(HEADER)}, {LEN_A}, {LEN_B}]")
    d["connections"] += [[HDR, "0", MUX, "0"], [SRC_A, "0", MUX, "1"], [SRC_B, "0", MUX, "2"]]
    return d


def variants_full(start: dict) -> dict[str, dict]:
    v = {}
    # 1) the original two-input mux left exactly as it was; a second two-input mux in front of the
    #    Throttle puts the 2 header samples before every 7 multiplexed samples
    d = without_conns(start, lambda c: c[0] == MUX and c[2] == THR)
    d["blocks"] += [vector_source("sync_word", HEADER),
                    {"name": "frame_mux", "id": "blocks_stream_mux",
                     "parameters": {"type": "float", "num_inputs": "2", "lengths": "(2, 7)"}}]
    d["connections"] += [["sync_word", "0", "frame_mux", "0"], [MUX, "0", "frame_mux", "1"],
                         ["frame_mux", "0", THR, "0"]]
    v["v1_cascaded_muxes"] = d
    # 2) the Stream Mux replaced by a Patterned Interleaver (pattern = input index of every output sample)
    d = without_conns(start, lambda c: MUX in (c[0], c[2]))
    d["blocks"] = [b for b in d["blocks"] if b["name"] != MUX]
    d["blocks"] += [vector_source("header_src", "[-1.0, -2.0]"),
                    {"name": "frame_interleaver", "id": "blocks_patterned_interleaver",
                     "parameters": {"type": "float", "pattern": "[0, 0, 1, 1, 1, 1, 2, 2, 2]"}}]
    d["connections"] += [["header_src", "0", "frame_interleaver", "0"], [SRC_A, "0", "frame_interleaver", "1"],
                         [SRC_B, "0", "frame_interleaver", "2"], ["frame_interleaver", "0", THR, "0"]]
    v["v2_patterned_interleaver"] = d
    # 3) the two header values from two Constant Sources on a four-input mux, one sample each
    d = without_conns(start, lambda c: c[2] == MUX)
    d["blocks"] += [{"name": "hdr_first", "id": "analog_const_source_x",
                     "parameters": {"type": "float", "const": "-1"}},
                    {"name": "hdr_second", "id": "analog_const_source_x",
                     "parameters": {"type": "float", "const": "-2.0"}}]
    d = setp(d, MUX, num_inputs=4, lengths="[1, 1, 4, 3]")
    d["connections"] += [["hdr_first", "0", MUX, "0"], ["hdr_second", "0", MUX, "1"],
                         [SRC_A, "0", MUX, "2"], [SRC_B, "0", MUX, "3"]]
    v["v3_constant_sources"] = d
    # 4) same structure as the reference, parameterised with variables (header vector and frame
    #    lengths) and a header source with an explicit repeat
    d = without_conns(start, lambda c: c[2] == MUX)
    d["blocks"] = [{"name": "header_word", "id": "variable", "parameters": {"value": "[-1, -2]"}},
                   {"name": "n_from_a", "id": "variable", "parameters": {"value": "4"}},
                   {"name": "n_from_b", "id": "variable", "parameters": {"value": "3"}}] + d["blocks"]
    d["blocks"].append(vector_source("hdr", "header_word", repeat="True"))
    d = setp(d, MUX, num_inputs="3", lengths="[len(header_word), n_from_a, n_from_b]")
    d["connections"] += [["hdr", "0", MUX, "0"], [SRC_A, "0", MUX, "1"], [SRC_B, "0", MUX, "2"]]
    v["v4_variables"] = d
    # 5) the original mux left as it was; a Vector Insert between the mux and the Throttle puts the
    #    two header samples at the start of every 9 output samples (period 9 includes the header)
    #    (round-1 reviewers' alternative)
    d = without_conns(start, lambda c: c[0] == MUX and c[2] == THR)
    d["blocks"].append({"name": "header_insert", "id": "blocks_vector_insert_x",
                        "parameters": {"type": "float", "vector": "[-1, -2]", "period": "9", "offset": "0"}})
    d["connections"] += [[MUX, "0", "header_insert", "0"], ["header_insert", "0", THR, "0"]]
    v["v5_vector_insert"] = d
    # 6) the same Vector Insert placed after the Throttle, in front of mux_out.dat (reviewers' alternative)
    d = without_conns(start, lambda c: c[0] == THR and c[2] == SINK_OUT)
    d["blocks"].append({"name": "header_insert", "id": "blocks_vector_insert_x",
                        "parameters": {"type": "float", "vector": "(-1, -2)", "period": "2 + 4 + 3", "offset": "0"}})
    d["connections"] += [[THR, "0", "header_insert", "0"], ["header_insert", "0", SINK_OUT, "0"]]
    v["v6_vector_insert_after_throttle"] = d
    # 7) tagged streams: each input cut into packets (2 / 4 / 3 items) and joined by a Tagged Stream
    #    Mux (reviewers' alternative)
    d = without_conns(start, lambda c: MUX in (c[0], c[2]))
    d["blocks"] = [b for b in d["blocks"] if b["name"] != MUX]
    d["blocks"].append(vector_source("hdr_words", HEADER))
    for name, n in (("pkt_hdr", 2), ("pkt_a", 4), ("pkt_b", 3)):
        d["blocks"].append({"name": name, "id": "blocks_stream_to_tagged_stream",
                            "parameters": {"type": "float", "packet_len": str(n), "len_tag_key": '"frame_len"'}})
    d["blocks"].append({"name": "frame_tsmux", "id": "blocks_tagged_stream_mux",
                        "parameters": {"type": "float", "ninputs": "3", "lengthtagname": '"frame_len"'}})
    d["connections"] += [["hdr_words", "0", "pkt_hdr", "0"], [SRC_A, "0", "pkt_a", "0"], [SRC_B, "0", "pkt_b", "0"],
                         ["pkt_hdr", "0", "frame_tsmux", "0"], ["pkt_a", "0", "frame_tsmux", "1"],
                         ["pkt_b", "0", "frame_tsmux", "2"], ["frame_tsmux", "0", THR, "0"]]
    v["v7_tagged_stream_mux"] = d
    # 8) a second two-input mux after the Throttle, right in front of mux_out.dat (reviewers' alternative)
    d = without_conns(start, lambda c: c[0] == THR and c[2] == SINK_OUT)
    d["blocks"] += [vector_source("sync_word", HEADER),
                    {"name": "frame_mux", "id": "blocks_stream_mux",
                     "parameters": {"type": "float", "num_inputs": "2", "lengths": "[2, 7]"}}]
    d["connections"] += [["sync_word", "0", "frame_mux", "0"], [THR, "0", "frame_mux", "1"],
                         ["frame_mux", "0", SINK_OUT, "0"]]
    v["v8_header_mux_after_throttle"] = d
    # 9) the header inserted into stream A itself before the mux: a Vector Insert puts -1, -2 in front
    #    of every 4 samples of A (period 6), and the mux takes 6 samples from that and 3 from B
    #    (round-2 reviewer's valid probe d7_p3)
    d = without_conns(start, lambda c: c[0] == SRC_A and c[2] == MUX)
    d["blocks"].append({"name": "hdr_ins", "id": "blocks_vector_insert_x",
                        "parameters": {"type": "float", "vector": "[-1, -2]", "period": "6", "offset": "0"}})
    d = setp(d, MUX, lengths="[6, 3]")
    d["connections"] += [[SRC_A, "0", "hdr_ins", "0"], ["hdr_ins", "0", MUX, "0"]]
    v["v9_vector_insert_on_stream_a"] = d
    return v


def probes_full(start: dict) -> dict[str, dict]:
    """Wrong answers kept out of the task files; main() --probes writes them to a scratch folder and
    re-checks the oracle's verdict on each (PROBE_PASSES_V6). They need more than one edit-operator
    call, so they cannot be mutation-gate mutants. Every one must fail V6."""
    p = {}
    # the round-1 shortcut: no multiplexing at all, one Vector Source holding the 18 output values
    d = without_conns(start, lambda c: MUX in (c[0], c[2]))
    d["blocks"] = [b for b in d["blocks"] if b["name"] != MUX]
    d["blocks"].append(vector_source("fixed_output", TARGET))
    d["connections"].append(["fixed_output", "0", THR, "0"])
    p["x_hardcoded_output"] = d
    # header placed after each frame instead of before it (third mux input, lengths [4, 3, 2])
    d = without_conns(start, lambda c: c[2] == MUX)
    d["blocks"].append(vector_source(HDR, HEADER))
    d = setp(d, MUX, num_inputs=3, lengths=f"[{LEN_A}, {LEN_B}, {len(HEADER)}]")
    d["connections"] += [[SRC_A, "0", MUX, "0"], [SRC_B, "0", MUX, "1"], [HDR, "0", MUX, "2"]]
    p["x_header_after_frame"] = d
    # Vector Insert with offset 7: the header lands after each frame
    d = copy.deepcopy(variants_full(start)["v5_vector_insert"])
    d = setp(d, "header_insert", offset="7")
    p["x_vector_insert_offset7"] = d
    # the Throttle removed (the output itself is right)
    d = reference_full(start)
    d["connections"] = [c for c in d["connections"] if THR not in (c[0], c[2])] + [[MUX, "0", SINK_OUT, "0"]]
    d["blocks"] = [b for b in d["blocks"] if b["name"] != THR]
    p["x_no_throttle"] = d
    # Round-2 shortcuts (G10R-1), each one change_graph edit; the anchored path checks reject them.
    # (a) stream B multiplexed with a hand-made copy of stream A that already holds the headers; stream A
    #     itself now feeds only stream_a.dat, so a later change of stream A would not reach mux_out.dat
    d = without_conns(start, lambda c: c[0] == SRC_A and c[2] == MUX)
    d["blocks"].append(vector_source("a_framed", HEADER + STREAM_A[:LEN_A] + HEADER + STREAM_A[LEN_A:]))
    d = setp(d, MUX, lengths=f"[{len(HEADER) + LEN_A}, {LEN_B}]")
    d["connections"].append(["a_framed", "0", MUX, "0"])
    p["x_doctored_copy_of_a"] = d
    # (b) the round-1 shortcut (one Vector Source holding the 18 output values) fed through the existing
    #     Stream Mux set to one input
    d = without_conns(start, lambda c: c[2] == MUX)
    d["blocks"].append(vector_source("fixed_output", TARGET))
    d = setp(d, MUX, num_inputs=1, lengths=f"[{PERIOD}]")
    d["connections"].append(["fixed_output", "0", MUX, "0"])
    p["x_hardcoded_via_1in_mux"] = d
    # (c) the same 18 values plus a zero source on an Add in front of the Throttle
    d = without_conns(start, lambda c: MUX in (c[0], c[2]))
    d["blocks"] = [b for b in d["blocks"] if b["name"] != MUX]
    d["blocks"] += [vector_source("fixed_output", TARGET), vector_source("zeros", (0,)),
                    {"name": "disguise", "id": "blocks_add_xx", "parameters": {"type": "float", "num_inputs": "2"}}]
    d["connections"] += [["fixed_output", "0", "disguise", "0"], ["zeros", "0", "disguise", "1"],
                         ["disguise", "0", THR, "0"]]
    p["x_hardcoded_via_add_zero"] = d
    return p


# expected V6 verdict of each probe (build.py --probes re-runs them through the ladder and asserts it)
PROBE_PASSES_V6 = {"x_hardcoded_output": False, "x_header_after_frame": False, "x_vector_insert_offset7": False,
                   "x_no_throttle": False, "x_doctored_copy_of_a": False, "x_hardcoded_via_1in_mux": False,
                   "x_hardcoded_via_add_zero": False}


# ----------------------------------------------------------------------------- task record
INSTRUCTION = (
    "This flowgraph demonstrates time-division multiplexing of two streams of real 32-bit float samples. "
    "Two repeating Vector Sources produce stream A (1, 2, 3, 4, 8, 9, 10, 11) and stream B (5, 6, 7, 12, 13, "
    "14). A Stream Mux takes 4 samples from stream A, then 3 samples from stream B, and repeats, so after the "
    "Throttle the File Sink writing mux_out.dat receives 1, 2, 3, ..., 14, 1, 2, 3, ... as float samples. Two "
    "more File Sinks record the streams before multiplexing: stream A in stream_a.dat and stream B in "
    "stream_b.dat.\n\n"
    "I want every 7-sample frame (the 4 samples from stream A followed by the 3 samples from stream B) to be "
    "preceded by a two-sample header: the value -1 followed by the value -2. The float samples written to "
    "mux_out.dat must therefore start with a header and read -1, -2, 1, 2, 3, 4, 5, 6, 7, -1, -2, 8, 9, 10, "
    "11, 12, 13, 14, -1, -2, 1, 2, ..., repeating this 18-sample pattern exactly, with no sample missing, added "
    "or out of place. The frames must still be built from streams A and B by multiplexing (I will change the "
    "streams later), and the Throttle must stay in front of mux_out.dat. Keep streams A and B as they are: "
    "stream_a.dat and stream_b.dat must still receive them exactly as before, from their first sample. The "
    "flowgraph should keep producing samples until it is stopped; do not limit the number of samples."
)

SPEC = {
    "header": "frame header: every 7-sample frame in mux_out.dat is preceded by the two samples -1, -2 (in "
              "that order)",
    "payload": "frame payload: after each header come 4 samples of stream A, then 3 samples of stream B, so "
               "mux_out.dat reads -1, -2, 1, 2, 3, 4, 5, 6, 7, -1, -2, 8, 9, 10, 11, 12, 13, 14 and repeats "
               "(period 18)",
    "start": "the output starts with a header: the first two samples written to mux_out.dat are -1, -2 "
             "(no offset allowed)",
    "continuous": "the pattern holds exactly and in phase over the whole capture: all 8190 captured samples of "
                  "mux_out.dat (455 periods) equal it, so no sample is missing, added or out of place",
    "multiplexed": "the frames are still built from streams A and B by multiplexing, not from a fixed list of "
                   "output values: the Vector Source behind stream_a.dat and the one behind stream_b.dat each "
                   "reach the File Sink of mux_out.dat through a block that merges several input streams "
                   "(graph checks)",
    "throttle": "the Throttle stays in front of mux_out.dat (graph check: a stream-merging block reaches a File "
                "Sink through a Throttle)",
    "a": "stream A recording unchanged: stream_a.dat receives 1, 2, 3, 4, 8, 9, 10, 11 repeated, from its "
         "first sample",
    "b": "stream B recording unchanged: stream_b.dat receives 5, 6, 7, 12, 13, 14 repeated, from its first "
         "sample",
    "run": "run condition: the flowgraph keeps producing samples until it is stopped (no sample limit); checked "
           "only by V5 liveness (at least 8192 items at every observation point), not by V6",
    "obs": "observation points: float samples written to the File Sinks mux_out.dat, stream_a.dat and "
           "stream_b.dat (8192 items each)",
}

ORACLE = {
    "checks": [
        {"point": "out", "feature": "equal_prefix", "ref": REF_OUT, "n": WHOLE_N, "max_offset": 0, "group": "f2p",
         "requires": [SPEC["header"], SPEC["payload"], SPEC["start"], SPEC["continuous"]]},
        {"point": "a", "feature": "equal_prefix", "ref": REF_A, "max_offset": 0, "group": "p2p",
         "requires": [SPEC["a"]]},
        {"point": "b", "feature": "equal_prefix", "ref": REF_B, "max_offset": 0, "group": "p2p",
         "requires": [SPEC["b"]]},
    ],
    "graph_checks": [
        *[{"kind": "path", "from_ids": ["blocks_vector_source_x"], "via_ids": MERGERS, "to_ids": ["blocks_file_sink"],
           "from_upstream_of": {"ids": ["blocks_file_sink"], "params": {"file": FILES[k]}},
           "to_params": {"file": FILES["out"]}, "group": "p2p", "requires": [SPEC["multiplexed"]]}
          for k in ("a", "b")],
        {"kind": "path", "from_ids": MERGERS, "via_ids": THROTTLES, "to_ids": ["blocks_file_sink"],
         "group": "p2p", "requires": [SPEC["throttle"]]},
    ],
}

MUTATIONS = [
    {"op": "set_param", "block": MUX, "param": "lengths", "value": "[1, 4, 3]", "breaks": SPEC["header"],
     "note": "one header sample per frame (alternately -1 and -2)"},
    {"op": "set_param", "block": HDR, "param": "vector", "value": "(-2, -1)", "breaks": SPEC["header"],
     "note": "header samples in the wrong order"},
    {"op": "set_param", "block": HDR, "param": "vector", "value": "(1, 2)", "breaks": SPEC["header"],
     "note": "header values without the minus sign"},
    {"op": "set_param", "block": MUX, "param": "lengths", "value": "[2, 3, 4]", "breaks": SPEC["payload"],
     "note": "3 samples of A and 4 of B per frame"},
    {"op": "rewire_input", "dst": MUX, "dst_port": "0", "new_src": SRC_A, "breaks": SPEC["header"],
     "note": "stream A on the header input; the header source is left on a Null Sink"},
    {"op": "rewire_input", "dst": MUX, "dst_port": "2", "new_src": SRC_A, "breaks": SPEC["payload"],
     "note": "stream A on both payload inputs"},
    {"op": "rewire_input", "dst": SINK_OUT, "dst_port": "0", "new_src": SRC_A, "breaks": SPEC["payload"],
     "note": "mux_out.dat records stream A instead of the multiplexed output"},
    {"op": "rewire_input", "dst": SINK_A, "dst_port": "0", "new_src": SRC_B, "breaks": SPEC["a"],
     "note": "stream_a.dat records stream B"},
    {"op": "rewire_input", "dst": SINK_B, "dst_port": "0", "new_src": THR, "breaks": SPEC["b"],
     "note": "stream_b.dat records the multiplexed output"},
    {"op": "set_param", "block": SRC_A, "param": "vector", "value": "(0, 1, 2, 5, 6, 7)", "breaks": SPEC["a"],
     "note": "the example's original (unperturbed) stream A"},
    {"op": "set_param", "block": SRC_B, "param": "vector", "value": "(3, 4, 8, 9)", "breaks": SPEC["b"],
     "note": "the example's original (unperturbed) stream B"},
    {"op": "set_param", "block": HDR, "param": "vector", "value": "[-1, -2] * 450 + [-2, -1] + [-1, -2] * 2000",
     "breaks": SPEC["continuous"],
     "note": "one swapped header in the middle of the capture (frame 451, samples 4050-4051), the round-1 "
             "reviewers' probe: the start and the end of the output are right, only the whole-capture "
             "comparison sees it"},
    {"op": "bypass", "block": THR, "breaks": SPEC["throttle"],
     "note": "the Throttle removed; the output itself is unchanged, only the graph check sees it"},
]

DISJOINTNESS = (
    "Inserts a two-sample frame header into the output of a two-stream time-division Stream Mux demo (dev-split "
    "seed blocks/stream_mux_demo.grc), checked sample-exactly. Nearest training-side specs "
    "(reports/reference/training_task_specs.json): t11_vector_sink_extend adds a second correlated stream next to "
    "a vector-sink display (no multiplexing, no header); t07_epy_custom_logic / t13_epy_save_block gate a signal "
    "on and off in fixed windows; GRC-Agent scenario 01_add_throttle inserts a Throttle into a dial-tone chain and "
    "05_full_rewire swaps a noise source for a constant source; t02/t05/t21 are modem chains. None multiplexes "
    "streams, adds frame headers or uses this seed."
)

NOTES = (
    "The data are small integers carried as float, so the oracle is exact: all 8190 captured samples of mux_out.dat "
    "(455 whole periods) must equal the stated 18-sample pattern from the first sample (no offset search, because "
    "the instruction states that the output starts with the header; a solution that appends the header after each "
    "frame writes 1, ..., 7, -1, -2, ... and fails). Round 1 replaced the former start window + tail window + mean "
    "by this single whole-capture comparison, so a wrong header anywhere in the capture fails. Graph checks (p2p): "
    "a Vector Source reaches a File Sink through a stream-merging block, and a merging block reaches a File Sink "
    "through a Throttle. Level L2: 8 functional blocks in the reference (three Vector Sources, Stream Mux, "
    "Throttle, three File Sinks) and the Stream Mux is multi-rate. The sample rate (32 kHz, Throttle only) is not "
    "observable and is not stated. Variants: v1 keeps the original mux and adds a second two-input mux (lengths "
    "(2, 7)); v2 replaces the mux by a Patterned Interleaver; v3 uses two Constant Sources on a four-input mux "
    "(lengths [1, 1, 4, 3]); v4 is the reference structure written with variables; v5 / v6 put a Vector Insert "
    "(period 9, offset 0) before / after the Throttle; v7 joins tagged packets of 2 / 4 / 3 items with a Tagged "
    "Stream Mux; v8 adds a second mux after the Throttle (v5-v8 are the round-1 reviewers' alternatives); v9 inserts the header into stream A with a Vector Insert (period 6) before the mux (round-2 reviewer's design). Probes "
    "(build.py --probes, not task files): x_hardcoded_output, x_header_after_frame and x_vector_insert_offset7 "
    "fail V6, x_no_throttle fails the Throttle graph check, and x_doctored_copy_of_a, x_hardcoded_via_1in_mux "
    "and x_hardcoded_via_add_zero fail the multiplexing graph checks."
)

REVIEW_NOTES = (
    "Known oracle limits: (1) 'the frames must still be built from streams A and B by multiplexing' is checked "
    "structurally: the Vector Source behind stream_a.dat, and the one behind stream_b.dat, must each reach the File "
    "Sink of mux_out.dat through one of the stream-merging blocks listed in build.py MERGERS (Embedded Python "
    "blocks included). This rejects a fixed output with or without a merging block (an Add with a zero source, "
    "the Stream Mux set to one input) and a header-baked copy of stream A multiplexed with stream B (probes "
    "x_hardcoded_*, x_doctored_copy_of_a). A deliberately contrived graph that still wires streams A and B into "
    "a merging block but takes the output values from elsewhere could pass; where the samples come from cannot "
    "be observed in a single run. Team decision (fix plan): keep this graph check or drop the sentence. "
    "(2) 'The Throttle must stay in front of mux_out.dat' is checked structurally only; its rate is not observable "
    "and not stated. (3) 'Keep producing samples until stopped' is checked only by V5 liveness (8192 items per "
    "observation point). (4) Stream tags (tag0 / tag1) are not checked. (5) Header-after-frame cannot be made by "
    "one mutation operator on this reference, so it is a probe (fails V6 f2p), not a gate mutant. Licence: derived "
    "from the GNU Radio 3.10.12 example blocks/stream_mux_demo.grc (author David Pi, GNU GPL 3.0+); the item keeps "
    "the GPL-3.0-or-later attribution (licence decision is the team's)."
)


def main() -> int:
    split_rec = read_json(SPLITS / "flowgraph_seed_split.json")["seeds"][SEED]
    assert split_rec["split"] == SPLIT, f"seed split is {split_rec['split']}, task split is {SPLIT}"
    sha = sha256_file(GNURADIO_EXAMPLES / SEED)
    assert sha == split_rec["sha256"], "example file differs from the inventoried seed"

    start = build_start()
    ref = tidy(reference_full(start))
    variants = {k: tidy(v) for k, v in variants_full(start).items()}
    for op in MUTATIONS:
        edit.apply(ref, op)                                  # every mutation applies to the reference
    props = {p for c in ORACLE["checks"] + ORACLE["graph_checks"] for p in c["requires"]}
    assert props == set(SPEC.values()) - {SPEC["run"], SPEC["obs"]}, "every stated property is checked"

    if "--probes" in sys.argv:
        from grcbench.tier_f.ladder import check             # needs the GNU Radio worker; probes only
        out = WORK / "probes" / TASK_ID
        out.mkdir(parents=True, exist_ok=True)
        task = taskio.load(HERE)                             # the written task (run build.py first)
        probes = probes_full(start)
        assert set(probes) == set(PROBE_PASSES_V6), set(probes) ^ set(PROBE_PASSES_V6)
        bad = []
        for k, v in probes.items():
            text = dump(tidy(v))
            (out / f"{k}.grc").write_text(text, encoding="utf-8")
            res = check(text, task)
            v6 = res["rungs"].get("V6") or {}
            passes = res["first_failed"] is None and res["highest_passed"] == "V6"
            fails = [f"{c['point']}:{c['feature']}" for c in v6.get("checks", []) if not c["passed"]]
            fails += [f"{g['kind']}: {g['detail']}" for g in v6.get("graph_checks", []) if not g["passed"]]
            print(f"{k:28s} passes V6: {passes!s:5s} (expected {PROBE_PASSES_V6[k]}); first failed "
                  f"{res['first_failed']}; groups {v6.get('groups')}; failing {fails}")
            if passes != PROBE_PASSES_V6[k]:
                bad.append(k)
        print(f"wrote probes to {out}")
        return 1 if bad else 0

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
            "points": [{"name": k, "sink": {"block_id": "blocks_file_sink", "params": {"file": f, "type": "float"}}}
                       for k, f in FILES.items()],
            "run": {"n_items": N_ITEMS, "min_items": N_ITEMS, "timeout_s": 10},
            "contract": ("the float samples written to the File Sink whose file name is mux_out.dat, and the float "
                         "samples written to the File Sinks stream_a.dat and stream_b.dat"),
        },
        "oracle": ORACLE,
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
        print("NOT WRITTEN:", *errors, sep="\n  ")
        return 1
    print(f"wrote {HERE.relative_to(REPO)} ({len(files)} flowgraphs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
