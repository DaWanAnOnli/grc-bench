"""Authoring script for GRC-Bench Tier-F item gbf-test-20 (parameter-tune, L2, composed graph).

Re-themed on 18 Oct 2026 (fix plan group G6). The first version (a two-stage decimator test bench)
shared the signature combine|filter|resample|tone, which hashes to train under signature v2, with
gbf-test-13; both were re-themed. This version: retune a baseband PAM test transmitter from 2-PAM to
Gray-coded 4-PAM by changing parameter values only.

Regenerates task.json, start.grc, reference.grc and variants/*.grc deterministically (no randomness;
the only inputs are the constants below and the pinned GNU Radio 3.10.12 block catalog, which is used
to drop default-valued parameters). Run inside WSL from the repository root:

    PYTHONPATH=src QT_QPA_PLATFORM=offscreen ~/grc-agent/.venv/bin/python items/tier_f/test/gbf-test-20/build.py
    ./gb tier-f gates items/tier_f/test/gbf-test-20

Start graph (48 kHz, samp_rate = 48000):
    Vector Source (byte, 8 bytes, repeat) -> Packed To Unpacked (1 bit per chunk, MSB first)
    -> Chunks To Symbols (byte -> float, table [-1, 1], dimension 1) -> Repeat (48: 1 kBd)
    -> Throttle (samp_rate) -> File Sink pam.dat (float)
Target: Gray-coded 4-PAM at 2.4 kBd from the same bit stream (pairs of bits in order, the first bit of
a pair the more significant one), levels 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3, held for whole
symbols, starting at the first sample. Reference edit (three values):
    Packed To Unpacked bits_per_chunk 1 -> 2 (MSB first: chunk = 2*first bit + second bit)
    Chunks To Symbols symbol_table [-1, 1] -> [-3, -1, 3, 1] (index order 00, 01, 10, 11)
    Repeat interp 48 -> 20 (48000 / 2400)
The trap is the table order: Gray order is about levels, the table is indexed by the chunk value, so
index 2 (bits 10) -> +3 and index 3 (bits 11) -> +1.

Genuinely different value-only solutions exist and are variants:
    v2  LSB-first unpacking with every data byte written bit-reversed: the same bit stream and the same
        chunks (GNU Radio's LSB-first mode still puts the first extracted bit in the chunk's MSB, measured:
        0x1B -> 11, 01, 10, 00), so the Gray table stays [-3, -1, 3, 1];
    v3  4 bits per chunk with dimension 2: Chunks To Symbols emits two levels per nibble from a
        32-entry table (written as a list comprehension).

Oracle. The output is a deterministic sequence of exact levels (-3, -1, 1, 3 are exact in float32), so
the f2p check is equal_prefix on the first three data periods (3 x 32 symbols x 20 samples = 1920
samples, max_offset 0: the chain has no delay, and the instruction says the first symbol starts at
the first sample). It pins bits per symbol, bit order, Gray table, levels, symbol rate and data at
once. The stated "parameter values only" rule and the 48 kHz sample rate are graph facts (no block's
output depends on samp_rate except the Throttle's pace): same_blocks_as_start (variables
value_only), same_connections_as_start and a variable with value 48000, all p2p.
Level L2 (R7): 6 functional blocks (Vector Source, Packed To Unpacked, Chunks To Symbols, Repeat,
Throttle, File Sink), with rate changes (unpacking, Repeat) and symbol mapping (modulation).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "src"))

from grcbench.common.canary import CANARY  # noqa: E402
from grcbench.tier_f import compose, grc, grcfile, taskio  # noqa: E402
from grcbench.tier_f.compose import block, file_sink, graph, signature, signature_split  # noqa: E402

TASK_ID = "gbf-test-20"
SPLIT = "test"
GID = "pam_test_tx"
FS = 48000
DATA = (0x1B, 0x9C, 0x4E, 0xD2, 0x37, 0xA5, 0x60, 0xF1)
DATA_TEXT = "(" + ", ".join(f"0x{b:02X}" for b in DATA) + ")"
START_SPS, BAUD = 48, 2400
SPS = FS // BAUD                                   # 20 samples per symbol
GRAY = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
OUT_FILE = "pam.dat"
PERIOD = len(DATA) * 8 // 2 * SPS                  # 640 samples per data period
N_CHECK = 3 * PERIOD                               # 1920
RUN_ITEMS = 4 * PERIOD                             # 2560

SRC, UNPACK, MAP, HOLD, THR, SINK = ("data_bytes", "unpack_bits", "symbol_mapper", "symbol_hold", "throttle",
                                     "pam_sink")


def bits_msb_first(data) -> list[int]:
    return [(b >> (7 - i)) & 1 for b in data for i in range(8)]


def expected_levels(n: int) -> list[float]:
    """pam.dat of a correct answer: Gray 4-PAM levels of bit pairs, each held for SPS samples."""
    bits = bits_msb_first(DATA)
    syms = [float(GRAY[(bits[i], bits[i + 1])]) for i in range(0, len(bits), 2)]
    period = [s for s in syms for _ in range(SPS)]
    return (period * (n // len(period) + 1))[:n]


def reverse_bits(b: int) -> int:
    return int(f"{b:08b}"[::-1], 2)


# ---- flowgraphs -------------------------------------------------------------------------------

def tx(*, data=DATA_TEXT, bits="1", endianness="gr.GR_MSB_FIRST", table="[-1, 1]", dimension="1",
       interp=str(START_SPS), gid=GID) -> dict:
    blocks = [
        # GRC evaluates Packed To Unpacked's endianness (gr.GR_MSB_FIRST) in the flowgraph namespace, which
        # only holds the blocks' template imports; none of these blocks imports gr, hence the Import block.
        block("import_0", "import", imports="from gnuradio import gr"),
        block(SRC, "blocks_vector_source_x", type="byte", vector=data, tags="[]", repeat="True", vlen="1"),
        block(UNPACK, "blocks_packed_to_unpacked_xx", type="byte", bits_per_chunk=bits, endianness=endianness,
              num_ports="1"),
        block(MAP, "digital_chunks_to_symbols_xx", in_type="byte", out_type="float", symbol_table=table,
              dimension=dimension, num_ports="1"),
        block(HOLD, "blocks_repeat", type="float", interp=interp, vlen="1"),
        block(THR, "blocks_throttle2", type="float", samples_per_second="samp_rate", vlen="1", ignoretag="True",
              limit="auto", maximum="0.1"),
        file_sink(SINK, "float", OUT_FILE),
    ]
    conns = [[SRC, "0", UNPACK, "0"], [UNPACK, "0", MAP, "0"], [MAP, "0", HOLD, "0"], [HOLD, "0", THR, "0"],
             [THR, "0", SINK, "0"]]
    return graph(gid, blocks, conns, {"samp_rate": FS})


start = tx()
reference = tx(bits="2", table="[-3, -1, 3, 1]", interp=str(SPS))

# v1: the same values spelled differently (float tuple, symbol rate from samp_rate).
v1 = tx(bits="2", table="(-3.0, -1.0, 3.0, 1.0)", interp="int(samp_rate/2400)", gid="pam4_v1")
# v2: LSB-first unpacking of bit-reversed bytes sends the same bit stream in the same 2-bit chunks.
v2 = tx(data="(" + ", ".join(f"0x{reverse_bits(b):02X}" for b in DATA) + ")", bits="2",
        endianness="gr.GR_LSB_FIRST", table="[-3, -1, 3, 1]", interp=str(SPS), gid="pam4_v2")
# v3: one nibble (two symbols) per chunk, dimension 2, 32-entry table; Repeat written as samp_rate//2400.
v3 = tx(bits="4", dimension="2",
        table="[lvl for n in range(16) for lvl in ((-3, -1, 3, 1)[n >> 2], (-3, -1, 3, 1)[n & 3])]",
        interp="samp_rate//2400", gid="pam4_v3")

SIG = signature(reference)
assert signature_split(SIG) == SPLIT, (SIG, signature_split(SIG))
VARIANT_GRAPHS = {"v1_float_table_rate_expression.grc": v1,
                  "v2_lsb_first_bit_reversed_bytes.grc": v2,
                  "v3_nibble_chunks_dimension_2.grc": v3}
for _name, _g in [("start", start), *VARIANT_GRAPHS.items()]:
    assert signature(_g) == SIG, (_name, signature(_g), SIG)

# ---- task record --------------------------------------------------------------------------------
INSTRUCTION = (
    "This flowgraph is a baseband test transmitter running at a sample rate of 48 kHz (samp_rate = 48000). "
    f"A Vector Source repeats eight data bytes, {DATA_TEXT.strip('()')}. Packed To Unpacked splits them into "
    "bits, most significant bit first; Chunks To Symbols maps each bit to a level (0 to -1, 1 to +1); and "
    "Repeat holds each level for 48 samples, so it sends 2-PAM at 1000 symbols per second. The float samples "
    f"go through a Throttle to the File Sink with file name {OUT_FILE} (item type float).\n\n"
    "Retune it to send Gray-coded 4-PAM at 2400 symbols per second instead:\n"
    "- Each symbol carries two bits, taken in order from the same bit stream as now (the eight bytes above in "
    "that order, each byte most significant bit first); the first bit of each pair is the more significant "
    "one.\n"
    "- Bit pairs map to levels with Gray coding: 00 to -3, 01 to -1, 11 to +1 and 10 to +3.\n"
    "- The sample rate stays 48 kHz, and each symbol's level is held constant for the whole symbol.\n"
    f"So {OUT_FILE} must contain exactly these levels, with the first symbol (the first two bits of the first "
    "byte) starting at the very first sample.\n\n"
    "Change parameter values only: do not add, remove or rewire blocks, and do not add or remove variables. "
    f"Keep the 48 kHz sample rate and the float output to {OUT_FILE}. The flowgraph should keep producing "
    "samples until it is stopped; do not limit the number of samples."
)

SPEC_PROPERTIES = [
    {"id": "pam4_sequence", "text": ("pam.dat holds the Gray 4-PAM levels of the unchanged bit stream: bit pairs in "
                                     "order (bytes in order, MSB first, first bit of a pair more significant), "
                                     "00 -3, 01 -1, 11 +1, 10 +3, 20 samples per symbol (2400 Bd at 48 kHz), "
                                     "first symbol at the first sample"),
     "checked_by": f"f2p equal_prefix of the first {N_CHECK} samples (three data periods), max_offset 0"},
    {"id": "values_only", "text": "only parameter values change: no block added, removed or rewired, no "
                                  "variable added or removed",
     "checked_by": "p2p graph checks same_blocks_as_start (variables value_only; import/note blocks ignored) "
                   "and same_connections_as_start (isomorphic)"},
    {"id": "sample_rate", "text": "the sample rate stays 48 kHz (samp_rate = 48000)",
     "checked_by": "p2p graph check block_present variable with value 48000 (the samples do not depend on "
                   "samp_rate; a different rate with a matching Repeat factor also fails the f2p sequence)"},
    {"id": "outputs", "text": "float items to the File Sink pam.dat",
     "checked_by": "observation contract (file name and item type)"},
    {"id": "run", "text": "the flowgraph keeps producing samples until it is stopped",
     "checked_by": f"V4 liveness: at least {RUN_ITEMS} items at the observation point"},
]

CHECKS = [
    {"point": "pam", "feature": "equal_prefix", "ref": expected_levels(N_CHECK), "max_offset": 0, "n": N_CHECK,
     "expect": True, "group": "f2p", "property": "pam4_sequence",
     "note": "the start (2-PAM, 48 samples per symbol) and every wrong table, bit order, symbol rate or level set fail"},
]
GRAPH_CHECKS = [
    {"kind": "same_blocks_as_start", "variables": "value_only", "ignore_ids": ["import", "note"], "group": "p2p",
     "property": "values_only"},
    {"kind": "same_connections_as_start", "mode": "isomorphic", "group": "p2p", "property": "values_only"},
    {"kind": "block_present", "ids": ["variable"], "params": {"value": str(FS)}, "min": 1, "group": "p2p",
     "property": "sample_rate"},
]

MUTATIONS = [
    {"op": "set_param", "block": MAP, "param": "symbol_table", "value": "[-3, -1, 1, 3]", "breaks": "pam4_sequence",
     "note": "natural binary order instead of Gray (10 -> +1, 11 -> +3)"},
    {"op": "set_param", "block": MAP, "param": "symbol_table", "value": "[1, 3, -1, -3]", "breaks": "pam4_sequence",
     "note": "Gray order written from the wrong end of the table"},
    {"op": "set_param", "block": MAP, "param": "symbol_table", "value": "[3, 1, -3, -1]", "breaks": "pam4_sequence",
     "note": "polarity inverted (00 -> +3)"},
    {"op": "set_param", "block": MAP, "param": "symbol_table", "value": "[-1.5, -0.5, 1.5, 0.5]",
     "breaks": "pam4_sequence", "note": "levels scaled by 1/2"},
    {"op": "set_param", "block": UNPACK, "param": "endianness", "value": "gr.GR_LSB_FIRST", "breaks": "pam4_sequence",
     "note": "bits taken LSB first: other pairs, other order"},
    {"op": "set_param", "block": UNPACK, "param": "bits_per_chunk", "value": "1", "breaks": "pam4_sequence",
     "note": "one bit per symbol with the 4-level table (only -3 and -1 used)"},
    {"op": "set_param", "block": HOLD, "param": "interp", "value": "24", "breaks": "pam4_sequence",
     "note": "2000 symbols per second"},
    {"op": "set_param", "block": HOLD, "param": "interp", "value": str(START_SPS), "breaks": "pam4_sequence",
     "note": "start symbol rate left (1000 Bd)"},
    {"op": "set_param", "block": SRC, "param": "vector", "value": DATA_TEXT.replace("0xF1", "0xF0"),
     "breaks": "pam4_sequence", "note": "data changed (last bit of the last byte)"},
    {"op": "set_param", "block": "samp_rate", "param": "value", "value": "57600", "breaks": "sample_rate",
     "note": "the samples are unchanged (24 samples per symbol would be 2400 Bd at 57.6 kHz); only the graph check sees it"},
    {"op": "bypass", "block": THR, "breaks": "values_only",
     "note": "Throttle removed: the samples are unchanged, the block-set check fails"},
    {"op": "set_param", "block": SINK, "param": "file", "value": "pam4.dat", "breaks": "outputs",
     "note": "written to another file name: observation pam.dat missing"},
]

obs = compose.file_observation(OUT_FILE, n_items=RUN_ITEMS, timeout_s=10.0)
obs["points"][0]["name"] = "pam"
obs["points"][0]["sink"]["params"]["type"] = "float"
obs["contract"] = (f"the float samples (48 kHz) written to the File Sink whose file name is {OUT_FILE}; "
                   f"{RUN_ITEMS} items are read")

RECORD = {
    "id": TASK_ID, "tier": "F", "split": SPLIT, "version": 2, "canary": CANARY,
    "license": "CC-BY-4.0 (original GRC-Bench item; licence to be confirmed by the team)",
    "source": {"kind": "composed_chain", "ref": "|".join(SIG)},
    "cluster_id": compose.cluster_id(SIG),
    "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring",
                   "generator_version": "0.2.0"},
    "review": {"status": "pending-team-review", "reviewers": [],
               "notes": ("Re-themed 18 Oct (fix plan G6): the earlier two-stage decimator bench hashed to train under "
                         "signature v2. Composed L2 item: a deviation from 'L2 from converted examples' (approved plan "
                         "Phase 3) and from eval plan 4.3's source list, recorded for the team (fix_plan team_decisions). "
                         "Level L2 under R7: 6 functional blocks with rate changes and symbol mapping. Oracle limits: "
                         "the sample rate is checked as a graph fact (a variable with value 48000), because no output "
                         "sample depends on samp_rate; the Throttle's own rate parameter is not checked. Accepted "
                         "alternatives (variants): LSB-first unpacking of bit-reversed bytes, and nibble chunks with "
                         "dimension 2; both change parameter values only and send the same samples.")},
    "family": "parameter-tune", "level": "L2",
    "instruction": INSTRUCTION,
    "start_grc": "start.grc", "reference_grc": "reference.grc",
    "observation": obs,
    "oracle": {"checks": CHECKS, "graph_checks": GRAPH_CHECKS},
    "mutations": MUTATIONS,
    "spec_properties": SPEC_PROPERTIES,
    "perturbation": {"perturbed": False, "changes": []},
    "disjointness": ("Not a training-side spec: retuning a baseband PAM symbol mapper (bits per symbol, Gray table "
                     "order, symbol rate) by values only. The nearest training specs are t02_qpsk_ber_chain (build a "
                     "QPSK modulator with decoding and BER) and t21_gmsk_pipe_build (build a GMSK modulate/demodulate "
                     "chain); neither retunes a Chunks To Symbols mapping, uses PAM or Gray coding, or changes the "
                     "symbol rate of an existing graph. Within the test split it differs from gbf-test-19 (GMSK "
                     "retune: bit rate, Gaussian BT, discriminator gain), gbf-test-13 (build: pulse-shaped NRZ data "
                     "upconverted to a complex IF) and gbf-test-05 (build: BPSK hard-decision demapper)."),
    "authoring_notes": {
        "level": "L2: 6 functional blocks; rate changes (Packed To Unpacked, Repeat) and symbol mapping",
        "signature_v2": "|".join(SIG),
        "signature_split": signature_split(SIG),
        "variant_signatures": "start, reference and all three variants share the signature",
        "reference_edit": "bits_per_chunk 1 -> 2, symbol_table [-1, 1] -> [-3, -1, 3, 1], Repeat 48 -> 20",
        "design": "see build.py docstring",
    },
}


def compact_text(data: dict) -> str:
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in data["blocks"]}
    small = grcfile.compact(data, defaults)
    diffs = grc.compact_equivalent(data, small)
    if diffs:
        raise SystemExit(f"compact form differs from the full form: {diffs}")
    return grcfile.dump(small)


def main() -> int:
    if (HERE / "variants").exists():
        shutil.rmtree(HERE / "variants")
    files = {"start.grc": compact_text(start), "reference.grc": compact_text(reference)}
    files.update({f"variants/{k}": compact_text(v) for k, v in VARIANT_GRAPHS.items()})
    errors = taskio.save(HERE, RECORD, files)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"written {HERE} (signature {SIG} -> {signature_split(SIG)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
