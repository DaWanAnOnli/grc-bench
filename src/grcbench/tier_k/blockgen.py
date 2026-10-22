"""GNU Radio behaviour questions for Tier K (`./gb blockq`; eval plan §4.2, Table 1).

About 100 test questions on test-side blocks of splits/block_partition.json, each asking for one number
that comes from what a block DOES, never from what its catalog entry SAYS:

  output_value   a block, set as the question states, receives a stated finite input sequence; what is
                 its k-th output item?
  output_count   it receives N items and the input ends; how many items (or vectors) does it produce?

Why execution: the agent traces Qoherent collected for training (reports/agent_traces.md) show the
catalog entries of almost every block (port types, parameter options, port counts), so questions
answered by such facts would not be held out. No trace ever ran a flowgraph, so a block's behaviour on
a given input is not in the training data. Tier K stays closed-book: a model under test gets only
the question text, with no GRC-Agent and no tools.

How an answer is made: the generator places the block in a flowgraph (finite Vector Source(s) ->
block -> File Sink(s)), runs it in the sandbox through the Tier-F ladder (V1-V5) and reads the
requested item from the recording; this is GNU Radio's own answer, the key. Each question also
carries an independent model of the block's documented behaviour (`expect`): a question is kept only
when the run and the model agree (within the key's tolerance), when two runs give identical output,
and when the rule cascade gives the key full credit and a wrong value less. Disagreements are dropped
and listed in the report. Blocks that a training trace built or ran in GNU Radio code
(splits/agent_traces.json, `./gb traces`) get no question, and a question is dropped when a bank
question (the 1,853 frozen questions) names the same block and asks about its outputs.

Settings and inputs come from SETTINGS, drawn with a seeded generator per (block, variant), so the
output is deterministic. Claude (Anthropic) wrote the settings table and the question templates; the
team reviews the wording (handoff/review_sheets/tier_k_block_questions.csv); the answers come from
GNU Radio 3.10.12. Scoring: the rule cascade (numbers), no judge rubric.

Outputs (`--out` root, default the repository):
  items/tier_k/test/block_questions.jsonl     the question records (no answers)
  items/tier_k/test/block_answers.jsonl       the answer key (reference for rules.score, evidence)
  handoff/review_sheets/tier_k_block_questions.csv   wording review sheet (two reviewers)
  reports/tier_k_block_questions.md           counts, drops, checks
"""

from __future__ import annotations

import argparse
import cmath
import csv
import hashlib
import io
import json
import math
import random
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from grcbench.common.canary import CANARY
from grcbench.common.io import write_text
from grcbench.common.paths import FROZEN, REPO, SPLITS, WORK
from grcbench.common.schemas import validate

SEED = "grc-bench-blockq-v2-2026-10-23"
VERSION = 2
GENERATOR = "grcbench.tier_k.blockgen"
OUT_QUESTIONS = "items/tier_k/test/block_questions.jsonl"
OUT_ANSWERS = "items/tier_k/test/block_answers.jsonl"
OUT_SHEET = "handoff/review_sheets/tier_k_block_questions.csv"
OUT_REPORT = "reports/tier_k_block_questions.md"
TRACES_MANIFEST = SPLITS / "agent_traces.json"

TARGET = 100                    # eval plan §4.2: "about 100" questions
MAX_PER_BLOCK = 4
VARIANT_ATTEMPTS = 12           # draws per variant before giving up on a new setting
PAD_ZEROS = 507                 # zeros after a short stated input to a block that works on large blocks
GR_IMPORT = {"name": "import_gr", "id": "import", "parameters": {"imports": "from gnuradio import gr"}}
WORKERS = 8
RUN_ITEMS = 200_000             # tap capacity per output port; every input here is far shorter
FLOAT_TOL = {"tolerance": 0.01, "abs_tolerance": 0.005}
INT_TOL = {"tolerance": 1e-9, "abs_tolerance": 0.01}
INT_TYPES = ("int", "short", "byte")
STD_PARAMS = {"id", "alias", "comment", "affinity", "minoutbuf", "maxoutbuf"}
KIND_WORDS = {"output_value": ("output value", "output item", "output sample", "what value", "outputs"),
              "output_count": ("how many output", "number of output", "how many items", "how many samples")}
MIN_LABEL_CHARS = 5             # shorter labels ("Add", "Not", "Or", "Max") are ordinary words: match the id only
LICENSE = ("CC-BY-4.0 for the question wording (GRC-Bench; licence to be confirmed by the team); the answers come "
           "from running GNU Radio 3.10.12 (GPL-3.0-or-later)")


# --------------------------------------------------------------------------- helpers
def _rng(block_id: str, variant: int) -> random.Random:
    h = hashlib.sha256(f"{SEED}\0{block_id}\0{variant}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _qid(block_id: str, variant: int) -> str:
    return "bq_" + hashlib.sha256(f"{SEED}\0{block_id}\0{variant}".encode()).hexdigest()[:16]


def _code(s: str) -> str:
    return f"`{s}`"


def _num(v: Any) -> str:
    """A value as the question prints it: integers without a decimal point, floats in short form."""
    if isinstance(v, complex):
        return f"{_num(v.real)}{'+' if v.imag >= 0 else '-'}{_num(abs(v.imag))}j"
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return f"{v:.1f}"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _bank() -> list[str]:
    out = []
    for name in ("eval_questions.jsonl", "train_questions.jsonl"):
        for line in (FROZEN / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line)["question"].lower())
    return out


def _in_bank(bank: list[str], label: str, block_id: str, kind: str) -> str | None:
    names = [n.lower() for n in (label, block_id) if n and (n == block_id or len(n) >= MIN_LABEL_CHARS)]
    pat = [re.compile(r"\b" + re.escape(n) + r"\b") for n in names]
    for q in bank:
        if any(p.search(q) for p in pat) and any(w in q for w in KIND_WORDS[kind]):
            return q[:160]
    return None


def _executed_blocks() -> set[str]:
    """Blocks a training trace built or ran in GNU Radio code (`./gb traces`)."""
    if not TRACES_MANIFEST.is_file():
        return set()
    return set(json.loads(TRACES_MANIFEST.read_text(encoding="utf-8")).get("executed_blocks", []))


class Grc:
    """GRC's Platform API: place one block in a fresh flowgraph, set parameters, rewrite, read ports."""

    def __init__(self):
        from grcbench.tier_k import catalog
        self.platform = catalog.load_platform()

    def place(self, block_id: str, params: dict[str, str] | None = None):
        fg = self.platform.make_flow_graph()
        fg.import_data({"options": {"parameters": {"id": "blockq", "generate_options": "no_gui"}, "states": {}},
                        "blocks": [{"name": "samp_rate", "id": "variable", "parameters": {"value": "32000"},
                                    "states": {}}, dict(GR_IMPORT, states={}),
                                   {"name": "b0", "id": block_id, "parameters": dict(params or {}), "states": {}}],
                        "connections": [], "metadata": {"file_format": 1}})
        fg.rewrite()
        blk = next(b for b in fg.blocks if b.name == "b0")
        return fg, blk

    @staticmethod
    def stream(ports) -> list:
        return [p for p in ports if p.domain != "message"]

    def valid(self, block_id: str, params: dict[str, str]) -> tuple[bool, list[str]]:
        fg, blk = self.place(block_id, params)
        fg.validate()
        errs = [e for e in blk.get_error_messages() if "not connected" not in e.lower()]
        return not errs, errs


def _setting_text(cls, params: dict[str, str]) -> str:
    """The setting in GRC's own words: "Type Float, M 2, N 5, Offset 0"."""
    parts = []
    for prm in cls.parameters_data:
        if prm["id"] not in params:
            continue
        value = params[prm["id"]]
        if prm.get("dtype") == "enum" or prm.get("options"):
            opts = [str(o) for o in prm.get("options") or []]
            labels = [str(x) for x in prm.get("option_labels") or opts]
            value = labels[opts.index(value)] if value in opts else value
        parts.append(f"{prm['label']} {value}")
    return ", ".join(parts)


# --------------------------------------------------------------------------- reference models
SEG_AEND = (0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF)
SEG_UEND = (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF)


def _seg(v: int, table: tuple[int, ...]) -> int:
    return next((i for i, end in enumerate(table) if v <= end), len(table))


def linear2alaw(pcm: int) -> int:
    """ITU-T G.711 A-law of a 16-bit sample (the Sun reference code GNU Radio's vocoder module uses)."""
    pcm >>= 3
    if pcm >= 0:
        mask = 0xD5
    else:
        mask, pcm = 0x55, -pcm - 1
    seg = _seg(pcm, SEG_AEND)
    if seg >= 8:
        return 0x7F ^ mask
    aval = seg << 4
    aval |= ((pcm >> 1) if seg < 2 else (pcm >> seg)) & 0xF
    return aval ^ mask


def linear2ulaw(pcm: int) -> int:
    """ITU-T G.711 u-law of a 16-bit sample (Sun reference code: 14-bit magnitude, bias 0x84, clip 8159)."""
    pcm >>= 2
    if pcm < 0:
        pcm, mask = -pcm, 0x7F
    else:
        mask = 0xFF
    pcm = min(pcm, 8159) + (0x84 >> 2)
    seg = _seg(pcm, SEG_UEND)
    if seg >= 8:
        return 0x7F ^ mask
    return (((seg << 4) | ((pcm >> (seg + 1)) & 0xF)) ^ mask) & 0xFF


def _c_div(a: int, b: int) -> int:
    """C integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _iir(ff: list[float], fb: list[float], x: list[float], oldstyle: bool) -> list[float]:
    y: list[float] = []
    for n in range(len(x)):
        acc = sum(ff[k] * x[n - k] for k in range(len(ff)) if n - k >= 0)
        fbs = sum(fb[k] * y[n - k] for k in range(1, len(fb)) if n - k >= 0)
        y.append(acc + fbs if oldstyle else acc - fbs)
    return y


# --------------------------------------------------------------------------- the settings table
@dataclass(frozen=True)
class Setting:
    block: str
    kind: str                                   # "output_value" | "output_count"
    variants: int
    make: Callable[[random.Random], dict]       # -> params, inputs, port, k, expect (+ optional fields)
    difficulty: str = "intermediate"


def _vals(rng: random.Random, n: int, choices: list) -> list:
    return [rng.choice(choices) for _ in range(n)]


def _pick_k(rng: random.Random, n: int, prefer: list[int] | None = None) -> int:
    return (rng.choice(prefer) if prefer else rng.randrange(n)) + 1


def _rail(r):
    lo, hi = -r.choice([0.25, 0.5, 0.75]), r.choice([0.25, 0.5, 0.75, 1.0])
    x = _vals(r, 6, [-1.5, -1.0, -0.6, -0.2, 0.0, 0.3, 0.6, 0.9, 1.25, 2.0])
    clipped = [i for i, v in enumerate(x) if v < lo or v > hi]
    k = _pick_k(r, 6, clipped)
    return {"params": {"lo": _num(lo), "hi": _num(hi)}, "inputs": [x], "port": 0, "k": k,
            "expect": min(max(x[k - 1], lo), hi)}


def _add_const(r):
    typ = r.choice(["float", "int", "short"])
    c = r.choice([2.5, -1.25, 0.75]) if typ == "float" else r.choice([3, -7, 12])
    x = _vals(r, 5, [-2.0, -0.5, 1.0, 3.5, 4.0] if typ == "float" else [-9, -2, 0, 5, 40])
    k = _pick_k(r, 5)
    return {"params": {"type": typ, "const": _num(c)}, "inputs": [x], "port": 0, "k": k, "expect": x[k - 1] + c}


def _add(r):
    typ = r.choice(["float", "int"])
    n_in = r.choice([2, 3])
    pool = [1.5, -2.0, 4.25, 10.0, -0.5] if typ == "float" else [3, -8, 11, 25, -1]
    xs = [_vals(r, 4, pool) for _ in range(n_in)]
    k = _pick_k(r, 4)
    return {"params": {"type": typ, "num_inputs": str(n_in)}, "inputs": xs, "port": 0, "k": k,
            "expect": sum(x[k - 1] for x in xs)}


def _sub(r):
    n_in = r.choice([2, 3])
    xs = [_vals(r, 4, [1.0, 2.5, -3.0, 7.0, 0.5]) for _ in range(n_in)]
    k = _pick_k(r, 4)
    return {"params": {"type": "float", "num_inputs": str(n_in)}, "inputs": xs, "port": 0, "k": k,
            "expect": xs[0][k - 1] - sum(x[k - 1] for x in xs[1:])}


def _divide(r):
    if r.random() < 0.5:
        a, b = _vals(r, 4, [9.0, -6.0, 1.5, 12.0]), _vals(r, 4, [2.0, -3.0, 4.0, 0.5])
        k = _pick_k(r, 4)
        return {"params": {"type": "float", "num_inputs": "2"}, "inputs": [a, b], "port": 0, "k": k,
                "expect": a[k - 1] / b[k - 1]}
    a, b = _vals(r, 4, [7, -7, 9, -11, 20]), _vals(r, 4, [2, 3, -4])
    k = _pick_k(r, 4)
    return {"params": {"type": "int", "num_inputs": "2"}, "inputs": [a, b], "port": 0, "k": k,
            "expect": _c_div(a[k - 1], b[k - 1])}


def _argmax(r):
    vals = r.sample([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 8)
    xs = [vals[:4], vals[4:]]
    best = max(vals)
    inp, idx = (0, xs[0].index(best)) if best in xs[0] else (1, xs[1].index(best))
    port = r.choice([0, 1])
    return {"params": {"type": "float", "num_inputs": "2", "vlen": "4"}, "inputs": xs, "port": port, "k": 1,
            "expect": idx if port == 0 else inp, "vector_in": 4}


def _blockinterleaver(r):
    size = r.choice([3, 4])
    idx = r.sample(range(size), size)
    while idx == sorted(idx):
        idx = r.sample(range(size), size)
    mode = r.choice(["interleave", "deinterleave"])
    x = [float(10 * (i + 1)) for i in range(2 * size)]
    out = []
    for b in range(2):
        blk = x[b * size:(b + 1) * size]
        if mode == "interleave":
            out += [blk[idx[i]] for i in range(size)]
        else:
            o = [0.0] * size
            for i in range(size):
                o[idx[i]] = blk[i]
            out += o
    k = _pick_k(r, len(x))
    return {"params": {"type": "float", "interleaver_indices": str(idx), "interleaver_mode": mode},
            "inputs": [x], "port": 0, "k": k, "expect": out[k - 1]}


def _magphase(r):
    pool = [3 + 4j, -1 + 0j, 0 + 2j, 1 + 1j, -2 - 2j, 0.5 - 0.5j]
    x = _vals(r, 4, pool)
    port = r.choice([0, 1])
    k = _pick_k(r, 4)
    return {"params": {}, "inputs": [x], "port": port, "k": k,
            "expect": abs(x[k - 1]) if port == 0 else cmath.phase(x[k - 1])}


def _c2real(r):
    x = _vals(r, 4, [1.5 - 2j, -0.5 + 1j, 3 + 0j, -2.25 - 0.75j])
    k = _pick_k(r, 4)
    return {"params": {}, "inputs": [x], "port": 0, "k": k, "expect": x[k - 1].real}


def _float_to_int(r):
    scale = r.choice([1, 10, 100])
    x = _vals(r, 5, [2.71, -2.71, 0.26, -1.04, 3.14, 7.49, -0.91])
    k = _pick_k(r, 5)
    return {"params": {"scale": str(scale)}, "inputs": [x], "port": 0, "k": k, "expect": round(x[k - 1] * scale)}


def _float_to_short(r):
    scale = r.choice([1, 100, 1000])
    x = _vals(r, 5, [1.234, -0.5, 40.0, -45.5, 400.0, 0.019])
    big = [i for i, v in enumerate(x) if abs(v * scale) > 32767]
    k = _pick_k(r, 5, big if big and r.random() < 0.5 else None)
    return {"params": {"scale": str(scale)}, "inputs": [x], "port": 0, "k": k,
            "expect": max(-32768, min(32767, round(x[k - 1] * scale)))}


def _keep_m_in_n(r):
    n = r.choice([3, 4, 5, 6])
    m = r.randint(1, n - 1)
    off = r.randint(0, n - m)
    x = [float(i + 1) for i in range(3 * n)]
    out = [x[w * n + off + j] for w in range(3) for j in range(m)]
    k = _pick_k(r, len(out))
    return {"params": {"type": "float", "m": str(m), "n": str(n), "offset": str(off)}, "inputs": [x], "port": 0,
            "k": k, "expect": out[k - 1]}


def _keep_one_in_n(r):
    n = r.choice([3, 4, 5, 6])
    x = [float(i + 1) for i in range(3 * n)]
    out = [x[w * n + n - 1] for w in range(3)]
    k = _pick_k(r, 3)
    return {"params": {"type": "float", "n": str(n)}, "inputs": [x], "port": 0, "k": k, "expect": out[k - 1]}


def _max(r):
    if r.random() < 0.5:
        xs = [_vals(r, 3, [1.0, 9.0, -4.0, 5.5, 2.0]) for _ in range(3)]
        k = _pick_k(r, 3)
        return {"params": {"type": "float", "num_inputs": "3", "vlen": "1", "vlen_out": "1"}, "inputs": xs,
                "port": 0, "k": k, "expect": max(x[k - 1] for x in xs)}
    x = _vals(r, 8, [2.0, 8.0, 1.0, 5.0, 7.0, 3.0, 9.0, 0.5])
    k = _pick_k(r, 2)
    return {"params": {"type": "float", "num_inputs": "1", "vlen": "4", "vlen_out": "1"}, "inputs": [x], "port": 0,
            "k": k, "expect": max(x[(k - 1) * 4:k * 4]), "vector_in": 4}


def _multiply_matrix(r):
    a = [[r.choice([1, 2, -1, 3, 0]) for _ in range(2)] for _ in range(2)]
    xs = [_vals(r, 3, [1.0, 2.0, -3.0, 0.5]) for _ in range(2)]
    port = r.choice([0, 1])
    k = _pick_k(r, 3)
    return {"params": {"type": "float", "A": str(tuple(tuple(row) for row in a)),
                       "tag_propagation_policy": "gr.TPP_ALL_TO_ALL"},
            "inputs": xs, "port": port, "k": k, "expect": a[port][0] * xs[0][k - 1] + a[port][1] * xs[1][k - 1]}


def _not(r):
    typ = r.choice(["int", "short"])
    x = _vals(r, 4, [5, -1, 0, 255, 1000, -32])
    k = _pick_k(r, 4)
    return {"params": {"type": typ}, "inputs": [x], "port": 0, "k": k, "expect": ~x[k - 1]}


def _or(r):
    typ = r.choice(["int", "short"])
    xs = [_vals(r, 4, [12, 10, 1, 2, 96, 7]) for _ in range(2)]
    k = _pick_k(r, 4)
    return {"params": {"type": typ, "num_inputs": "2"}, "inputs": xs, "port": 0, "k": k,
            "expect": xs[0][k - 1] | xs[1][k - 1]}


def _patterned(r):
    pattern = r.choice([[0, 0, 1, 2], [0, 1, 0, 2], [1, 0, 2, 0], [0, 1, 1, 2]])
    n_in = max(pattern) + 1
    reps = 2
    xs = [[float(100 * (p + 1) + i) for i in range(reps * pattern.count(p))] for p in range(n_in)]
    pos = [0] * n_in
    out = []
    for _ in range(reps):
        for p in pattern:
            out.append(xs[p][pos[p]])
            pos[p] += 1
    k = _pick_k(r, len(out))
    return {"params": {"type": "float", "pattern": str(pattern)}, "inputs": xs, "port": 0, "k": k,
            "expect": out[k - 1]}


def _short_to_char(r):
    x = _vals(r, 4, [1000, 513, 256, 32000, 4095, 300])
    k = _pick_k(r, 4)
    return {"params": {}, "inputs": [x], "port": 0, "k": k, "expect": x[k - 1] >> 8}


def _short_to_float(r):
    scale = r.choice([2, 4, 100])
    x = _vals(r, 4, [100, -6, 250, 1, -32768])
    k = _pick_k(r, 4)
    return {"params": {"scale": str(scale)}, "inputs": [x], "port": 0, "k": k, "expect": x[k - 1] / scale}


def _stream_to_vector(r):
    n = r.choice([4, 8, 16])
    m = r.choice([5, 10, 12])
    return {"params": {"type": "float", "num_items": str(n)}, "count_input": (n * m, 1.0), "port": 0, "expect": m,
            "vector_out": n}


def _vector_to_stream(r):
    n = r.choice([2, 4, 8])
    m = r.choice([5, 10, 12])
    return {"params": {"type": "float", "num_items": str(n)}, "count_input": (n * m, 1.0), "port": 0,
            "expect": n * m, "vector_in": n}


def _vector_to_streams(r):
    s = r.choice([2, 3, 4])
    x = [float(i + 1) for i in range(2 * s)]
    port = r.randrange(s)
    k = r.choice([1, 2])
    return {"params": {"type": "float", "num_streams": str(s)}, "inputs": [x], "port": port, "k": k,
            "expect": x[(k - 1) * s + port], "vector_in": s}


def _transcendental(r):
    name, pool, fn = r.choice([("log10", [1000.0, 0.01, 10.0, 0.5], math.log10), ("exp", [0.0, 1.0, -1.0, 2.0], math.exp),
                               ("tanh", [0.0, 0.5, -1.0, 2.0], math.tanh), ("cos", [0.0, 1.0, 3.14159, 0.5], math.cos)])
    x = _vals(r, 4, pool)
    k = _pick_k(r, 4)
    return {"params": {"type": "float", "name": name}, "inputs": [x], "port": 0, "k": k,
            "expect": fn(x[k - 1])}


def _unpacked_to_packed(r):
    bpc = r.choice([1, 2, 4])
    msb = r.random() < 0.6
    per_byte = 8 // bpc
    chunks = [r.randrange(1 << bpc) for _ in range(2 * per_byte)]
    bits = []
    for c in chunks:
        cb = [(c >> (bpc - 1 - i)) & 1 for i in range(bpc)] if msb else [(c >> i) & 1 for i in range(bpc)]
        bits += cb
    out = []
    for b in range(2):
        byte_bits = bits[b * 8:(b + 1) * 8]
        out.append(sum(bit << (7 - i) for i, bit in enumerate(byte_bits)) if msb
                   else sum(bit << i for i, bit in enumerate(byte_bits)))
    k = _pick_k(r, 2)
    return {"params": {"type": "byte", "bits_per_chunk": str(bpc),
                       "endianness": "gr.GR_MSB_FIRST" if msb else "gr.GR_LSB_FIRST"},
            "inputs": [chunks], "port": 0, "k": k, "expect": out[k - 1]}


def _puncture(r):
    size = r.choice([4, 6])
    pat = r.randrange(1, (1 << size) - 1)
    while bin(pat).count("1") < 2:
        pat = r.randrange(1, (1 << size) - 1)
    x = [float(i + 1) for i in range(2 * size)]
    out = [x[b * size + i] for b in range(2) for i in range(size) if (pat >> (size - 1 - i)) & 1]
    if r.random() < 0.5:
        k = _pick_k(r, len(out))
        return {"params": {"type": "float", "puncsize": str(size), "puncpat": str(pat), "delay": "0"},
                "inputs": [x], "port": 0, "k": k, "expect": out[k - 1]}
    n = size * r.choice([20, 30])
    return {"params": {"type": "float", "puncsize": str(size), "puncpat": str(pat), "delay": "0"},
            "count_input": (n, 1.0), "port": 0, "expect": n // size * bin(pat).count("1"), "kind": "output_count"}


def _fft_filter(r):
    """The FFT filter works on FFT-sized blocks, so the stated values are followed by zeros: without
    them a 5-item input ends before the first block and nothing comes out."""
    taps = r.choice([[0.5, 0.25], [1.0, -1.0], [0.25, 0.5, 0.25], [2.0, 0.0, 1.0]])
    x = _vals(r, 5, [4.0, 8.0, 0.0, -2.0, 1.0])
    y = _iir(taps, [1.0], x, True)
    k = _pick_k(r, 5)
    return {"params": {"type": "fff", "decim": "1", "taps": str(taps), "samp_delay": "0", "nthreads": "1"},
            "inputs": [x], "pad_zeros": PAD_ZEROS, "port": 0, "k": k, "expect": y[k - 1]}


def _iir_filter(r):
    ff = r.choice([[1.0], [0.5, 0.5], [1.0, 1.0]])
    a1 = r.choice([0.5, -0.5, 0.25])
    old = r.random() < 0.5
    x = [1.0, 0.0, 0.0, 0.0, 0.0]
    y = _iir(ff, [1.0, a1], x, old)
    k = r.choice([2, 3, 4])
    return {"params": {"type": "ffd", "fftaps": str(ff), "fbtaps": str([1.0, a1]), "oldstyle": str(old)},
            "inputs": [x], "port": 0, "k": k, "expect": y[k - 1]}


def _single_pole(r):
    alpha = r.choice([0.25, 0.5, 0.75])
    x = r.choice([[1.0, 1.0, 1.0, 1.0], [4.0, 0.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0]])
    y, prev = [], 0.0
    for v in x:
        prev = alpha * v + (1 - alpha) * prev
        y.append(prev)
    k = r.choice([2, 3, 4])
    return {"params": {"type": "float", "alpha": _num(alpha)}, "inputs": [x], "port": 0, "k": k, "expect": y[k - 1]}


def _alaw(r):
    x = _vals(r, 4, [0, 1000, -1000, 32767, -32768, 64, -8])
    k = _pick_k(r, 4)
    return {"params": {}, "inputs": [x], "port": 0, "k": k, "expect": linear2alaw(x[k - 1])}


def _ulaw(r):
    x = _vals(r, 4, [0, 1000, -1000, 32767, -32768, 64, -8])
    k = _pick_k(r, 4)
    return {"params": {}, "inputs": [x], "port": 0, "k": k, "expect": linear2ulaw(x[k - 1])}


def _count(params: Callable[[random.Random], dict], n_in: Callable[[dict, random.Random], int],
           n_out: Callable[[dict, int], int], value: Any = 1.0, **extra):
    def make(r):
        p = params(r)
        n = n_in(p, r)
        return {"params": p, "count_input": (n, value), "port": 0, "expect": n_out(p, n), **extra}
    return make


SETTINGS: list[Setting] = [
    Setting("analog_rail_ff", "output_value", 4, _rail, "basic"),
    Setting("analog_am_demod_cf", "output_count", 2, _count(
        lambda r: {"chan_rate": str(r.choice([32000, 48000, 64000])), "audio_decim": str(r.choice([2, 4, 8])),
                   "audio_pass": "5000", "audio_stop": "5500"},
        lambda p, r: int(p["audio_decim"]) * r.choice([60, 120]), lambda p, n: n // int(p["audio_decim"]), 1 + 0j)),
    Setting("analog_wfm_rcv", "output_count", 2, _count(
        lambda r: {"quad_rate": str(r.choice([192000, 240000])), "audio_decimation": str(r.choice([4, 5]))},
        lambda p, r: int(p["audio_decimation"]) * r.choice([100, 200]), lambda p, n: n // int(p["audio_decimation"]),
        1 + 0j)),
    Setting("blocks_add_const_vxx", "output_value", 3, _add_const, "basic"),
    Setting("blocks_add_xx", "output_value", 2, _add, "basic"),
    Setting("blocks_sub_xx", "output_value", 3, _sub, "basic"),
    Setting("blocks_divide_xx", "output_value", 3, _divide),
    Setting("blocks_argmax_xx", "output_value", 3, _argmax, "advanced"),
    Setting("blocks_blockinterleaver_xx", "output_value", 3, _blockinterleaver, "advanced"),
    Setting("blocks_complex_to_magphase", "output_value", 3, _magphase),
    Setting("blocks_complex_to_real", "output_value", 1, _c2real, "basic"),
    Setting("blocks_float_to_int", "output_value", 3, _float_to_int),
    Setting("blocks_float_to_short", "output_value", 3, _float_to_short),
    Setting("blocks_keep_m_in_n", "output_value", 4, _keep_m_in_n),
    Setting("blocks_keep_one_in_n", "output_value", 3, _keep_one_in_n),
    Setting("blocks_max_xx", "output_value", 3, _max),
    Setting("blocks_multiply_matrix_xx", "output_value", 3, _multiply_matrix),
    Setting("blocks_not_xx", "output_value", 2, _not),
    Setting("blocks_or_xx", "output_value", 2, _or, "basic"),
    Setting("blocks_patterned_interleaver", "output_value", 3, _patterned, "advanced"),
    Setting("blocks_short_to_char", "output_value", 2, _short_to_char, "advanced"),
    Setting("blocks_short_to_float", "output_value", 3, _short_to_float),
    Setting("blocks_stream_to_vector", "output_count", 2, _stream_to_vector, "basic"),
    Setting("blocks_vector_to_stream", "output_count", 1, _vector_to_stream, "basic"),
    Setting("blocks_vector_to_streams", "output_value", 3, _vector_to_streams),
    Setting("blocks_transcendental", "output_value", 3, _transcendental, "basic"),
    Setting("blocks_unpacked_to_packed_xx", "output_value", 4, _unpacked_to_packed, "advanced"),
    Setting("digital_gmskmod_bc", "output_count", 1, _count(
        lambda r: {"samples_per_symbol": str(r.choice([2, 4, 8])), "bt": "0.3", "L": "4"},
        lambda p, r: r.choice([64, 128]), lambda p, n: n * int(p["samples_per_symbol"]), 1)),
    Setting("dtv_catv_reed_solomon_enc_bb", "output_count", 1, _count(
        lambda r: {}, lambda p, r: 122 * r.choice([4, 6]), lambda p, n: n // 122 * 128, 0), "advanced"),
    Setting("fec_encode_ccsds_27_bb", "output_count", 1, _count(
        lambda r: {}, lambda p, r: r.choice([10, 25]), lambda p, n: 16 * n, 85), "advanced"),
    Setting("fec_puncture_xx", "output_value", 3, _puncture, "advanced"),
    Setting("goertzel_fc", "output_count", 1, _count(
        lambda r: {"rate": "8000", "len": str(r.choice([32, 64, 100])), "freq": "1000"},
        lambda p, r: int(p["len"]) * r.choice([10, 20]), lambda p, n: n // int(p["len"]), 0.5)),
    Setting("band_reject_filter", "output_count", 1, _count(
        lambda r: {"type": "fir_filter_fff", "decim": str(r.choice([2, 3, 4])), "gain": "1", "samp_rate": "48000",
                   "low_cutoff_freq": "5000", "high_cutoff_freq": "8000", "width": "1000"},
        lambda p, r: int(p["decim"]) * 200, lambda p, n: n // int(p["decim"]), 1.0)),
    Setting("fft_filter_xxx", "output_value", 3, _fft_filter),
    Setting("freq_xlating_fir_filter_xxx", "output_count", 1, _count(
        lambda r: {"type": "fcf", "decim": str(r.choice([2, 4, 5, 6])), "taps": "[0.125]*8", "center_freq": "0",
                   "samp_rate": "48000"},
        lambda p, r: int(p["decim"]) * 200, lambda p, n: n // int(p["decim"]), 1.0)),
    Setting("iir_filter_xxx", "output_value", 4, _iir_filter, "advanced"),
    Setting("rational_resampler_xxx", "output_count", 2, _count(
        lambda r: dict(zip(("interp", "decim"), map(str, r.choice([(3, 2), (2, 5), (4, 3), (5, 2)]))),
                       type="fff", taps="[]", fbw="0"),
        lambda p, r: int(p["decim"]) * 100, lambda p, n: n // int(p["decim"]) * int(p["interp"]), 1.0)),
    Setting("single_pole_iir_filter_xx", "output_value", 3, _single_pole),
    Setting("pfb_channelizer_ccf", "output_count", 1, _count(
        lambda r: (lambda n: {"nchans": str(n), "taps": f"[1.0]*{n}", "osr": "1.0", "atten": "100",
                              "samp_delay": "0", "ch_map": "[]", "bus_structure_source": "[[0,],]"})(
            r.choice([2, 4, 8])),
        lambda p, r: int(p["nchans"]) * 100, lambda p, n: n // int(p["nchans"]), 1 + 0j), "advanced"),
    Setting("vocoder_alaw_encode_sb", "output_value", 3, _alaw, "advanced"),
    Setting("vocoder_ulaw_encode_sb", "output_value", 3, _ulaw, "advanced"),
    Setting("vocoder_g721_decode_bs", "output_count", 1, _count(
        lambda r: {}, lambda p, r: r.choice([80, 160]), lambda p, n: n, 0)),
    Setting("vocoder_gsm_fr_decode_ps", "output_count", 1, _count(
        lambda r: {}, lambda p, r: 33 * r.choice([2, 3]), lambda p, n: n // 33 * 160, 0,
        vector_in=33, frame_head=208), "advanced"),
]


# --------------------------------------------------------------------------- running GNU Radio
def _ports(grc: Grc, block_id: str, params: dict[str, str]):
    _, blk = grc.place(block_id, params)
    ins = [(str(p.dtype), int(p.vlen or 1)) for p in grc.stream(blk.sinks)]
    outs = [(str(p.dtype), int(p.vlen or 1)) for p in grc.stream(blk.sources)]
    return ins, outs


def _vector_text(values: list) -> str:
    return "[" + ", ".join(repr(complex(v)) if isinstance(v, complex) else repr(v) for v in values) + "]"


def _input_vectors(spec: dict, ins: list[tuple[str, int]]) -> list[str]:
    """The Vector Source expression of every input port."""
    if "inputs" in spec:
        pad = f" + [0.0] * {spec['pad_zeros']}" if spec.get("pad_zeros") else ""
        return [_vector_text(x) + pad for x in spec["inputs"]]
    n, value = spec["count_input"]
    if spec.get("frame_head") is not None:       # GSM frames: a valid header byte, then zeros
        vlen = ins[0][1]
        frame = [spec["frame_head"]] + [0] * (vlen - 1)
        return [f"{frame!r} * {n // vlen}"]
    lit = repr(complex(value)) if isinstance(value, complex) else repr(value)
    return [f"[{lit}] * {n}" for _ in ins]


def probe_graph(block_id: str, spec: dict, ins, outs) -> tuple[str, dict] | str:
    """The probe flowgraph (sources -> block -> sinks) as .grc text and its observation task, or why not."""
    from grcbench.tier_f import grcfile
    vectors = _input_vectors(spec, ins)
    if len(vectors) != len(ins):
        return f"{len(vectors)} input vectors for {len(ins)} input ports"
    blocks = [{"name": "samp_rate", "id": "variable", "parameters": {"value": "32000"}}, dict(GR_IMPORT)]
    conns = []
    for i, ((dt, vl), vec) in enumerate(zip(ins, vectors)):
        blocks.append({"name": f"src{i}", "id": "blocks_vector_source_x",
                       "parameters": {"type": dt, "vector": vec, "repeat": "False", "vlen": str(vl)}})
        conns.append([f"src{i}", "0", "dut", str(i)])
    blocks.append({"name": "dut", "id": block_id, "parameters": dict(spec["params"])})
    points = []
    for j, (dt, vl) in enumerate(outs):
        blocks.append({"name": f"snk{j}", "id": "blocks_file_sink",
                       "parameters": {"type": dt, "file": f"out{j}.dat", "vlen": str(vl)}})
        conns.append(["dut", str(j), f"snk{j}", "0"])
        points.append({"name": f"out{j}", "sink": {"block_id": "blocks_file_sink",
                                                   "params": {"file": f"out{j}.dat", "type": dt}}})
    data = {"options": {"parameters": {"id": "blockq_probe", "generate_options": "no_gui", "output_language": "python"}},
            "blocks": blocks, "connections": conns, "metadata": {"file_format": 1}}
    task = {"observation": {"points": points, "run": {"n_items": RUN_ITEMS, "min_items": 0, "timeout_s": 60}},
            "oracle": {"checks": [{"point": "out0", "feature": "max_abs", "max": 1e30}]}}
    return grcfile.dump(data), task


def run_once(graph_text: str, task: dict, port: int, tag: str) -> tuple[list | None, str]:
    """Run the probe flowgraph once in the sandbox (thread-safe: its own process and run folder);
    returns the recorded items of the asked port (a vector port gives one list per vector) and a status."""
    import numpy as np

    from grcbench.tier_f import ladder
    root = WORK / "blockq" / tag
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    try:
        res = ladder.check(graph_text, task, keep=True, run_root=root, timeout=180)
        run_dirs = [d for d in root.iterdir() if d.is_dir()]
        obs = run_dirs[0] / "obs" / f"out{port}.npy" if len(run_dirs) == 1 else None
        if res.get("highest_passed") not in ("V5", "V6") or obs is None or not obs.exists():
            rung = res.get("first_failed") or res.get("highest_passed")
            errs = ((res.get("rungs") or {}).get(rung) or {}).get("errors") or []
            return None, f"run failed at {rung}: {'; '.join(map(str, errs))[:200]}"
        arr = np.load(obs)
        return arr.tolist(), "ok"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _answer(spec: dict, kind: str, items: list, out_dtype: str) -> float | None:
    if kind == "output_count":
        return float(len(items))
    k = spec["k"]
    if len(items) < k:
        return None
    v = items[k - 1]
    if isinstance(v, list):
        v = v[0]
    v = float(v.real if isinstance(v, complex) else v)
    if out_dtype == "byte":
        v = float(int(v) & 0xFF)
    return v


def _is_int(kind: str, out_dtype: str) -> bool:
    return kind == "output_count" or out_dtype in INT_TYPES


def _tol(kind: str, out_dtype: str, value: float) -> dict:
    """Integers exact; floats within 1 % or an absolute margin that shrinks near zero (so 0 never passes
    for a small non-zero answer)."""
    if _is_int(kind, out_dtype):
        return dict(INT_TOL)
    return {**FLOAT_TOL, "abs_tolerance": FLOAT_TOL["abs_tolerance"] if value == 0
            else round(min(FLOAT_TOL["abs_tolerance"], 0.1 * abs(value)), 6)}


def _agrees(a: float, b: float, tol: dict) -> bool:
    return abs(a - b) <= max(tol["abs_tolerance"], tol["tolerance"] * abs(b))


# --------------------------------------------------------------------------- question text
def _input_text(spec: dict, ins: list[tuple[str, int]]) -> str:
    if "inputs" in spec:
        def seq(x, dt, vl):
            vals = ", ".join(_num(v) for v in x)
            if vl > 1:
                n_vec = len(x) // vl
                return f"{'one vector' if n_vec == 1 else f'{n_vec} vectors'} of {vl} {dt} values ({vals}, in this order)"
            zeros = f", followed by {spec['pad_zeros']} zeros" if spec.get("pad_zeros") else ""
            return f"exactly these {len(x)} {dt} items, in this order: {vals}{zeros}"
        if len(ins) == 1:
            return f"Its input receives {seq(spec['inputs'][0], *ins[0])}, and then the input ends."
        parts = [f"input {i} receives {seq(x, *ins[i])}" for i, x in enumerate(spec["inputs"])]
        return "Its " + "; ".join(parts) + ". Then every input ends."
    n, value = spec["count_input"]
    dt, vl = ins[0]
    if spec.get("frame_head") is not None:
        return (f"Its input receives {n // vl} vectors of {vl} byte values (each vector starts with the value "
                f"{spec['frame_head']}, followed by zeros), and then the input ends.")
    if vl > 1:
        return f"Its input receives {n // vl} vectors of {vl} {dt} values, every value {_num(value)}, and then the input ends."
    each = f", every item {_num(value)}" if len(ins) == 1 else ""
    return f"Its input receives {n} {dt} items{each}, and then the input ends."


def question_text(cls, block_id: str, kind: str, spec: dict, ins, outs) -> str:
    setting = _setting_text(cls, spec["params"])
    shown = {p["id"] for p in cls.parameters_data} - STD_PARAMS
    others = bool(shown - set(spec["params"]))
    head = (f"In GNU Radio 3.10, take the {cls.label} block ({_code(block_id)}) set to {setting}"
            + (", with its other parameters at their defaults." if others else ".")) if setting else \
        f"In GNU Radio 3.10, take the {cls.label} block ({_code(block_id)}) with its default settings."
    port = spec["port"]
    on_port = f" on its output port {port}" if len(outs) > 1 else ""
    if kind == "output_count":
        dt, vl = outs[port]
        what = f"vectors of {vl} values" if vl > 1 else f"{dt} items"
        return f"{head} {_input_text(spec, ins)} How many {what} does it produce{on_port} in total?"
    unsigned = " Give the byte as an unsigned number from 0 to 255." if outs[port][0] == "byte" else ""
    return (f"{head} {_input_text(spec, ins)} What is the value of output item {spec['k']} (counting from 1)"
            f"{on_port}?{unsigned}")


# --------------------------------------------------------------------------- references and checks
def reference(q: dict) -> dict:
    return {"value": q["answer"], "unit": "", **q["tolerance"]}


def _rules_check(q: dict) -> str | None:
    from grcbench.tier_k import rules
    ref = reference(q)
    v = q["answer"]
    right = f"Final answer: {_num(v) if not float(v).is_integer() else int(v)}"
    wrong_v = v + 1 if q["integer"] else (v * 1.5 + 1)
    wrong = f"Final answer: {_num(wrong_v) if not float(wrong_v).is_integer() else int(wrong_v)}"
    r1, r0 = rules.score(right, ref), rules.score(wrong, ref)
    if r1["method"] == "judge" or r1["credit"] != 1.0:
        return f"reference answer scores {r1['method']} {r1['credit']} ({r1['details'].get('reason')})"
    if r0["method"] != "judge" and r0["credit"] >= 1.0:
        return f"wrong answer scores {r0['credit']}"
    return None


# --------------------------------------------------------------------------- build
def _candidates(grc: Grc, part: dict, executed: set[str], only: set[str] | None = None
                ) -> tuple[list[dict], list[dict]]:
    out, skipped = [], []
    for s in SETTINGS:
        if only and s.block not in only:
            continue
        if part.get(s.block, {}).get("split") != "test":
            skipped.append({"block": s.block, "reason": "not a test-side block"})
            continue
        if s.block in executed:
            skipped.append({"block": s.block, "reason": "built or run in GNU Radio code by a training trace"})
            continue
        seen: set[str] = set()
        for v in range(s.variants):
            for attempt in range(VARIANT_ATTEMPTS):     # a variant must not repeat an earlier setting and input
                spec = s.make(_rng(s.block, v + 1000 * attempt))
                keys = _variant_keys(spec)
                if not keys & seen:
                    break
            else:
                skipped.append({"block": s.block, "reason": f"variant {v}: no new setting in {VARIANT_ATTEMPTS} draws"})
                continue
            seen |= keys
            kind = spec.pop("kind", s.kind)
            out.append({"block": s.block, "variant": v, "kind": kind, "spec": spec, "difficulty": s.difficulty})
    return out, skipped


def _variant_keys(spec: dict) -> set[str]:
    """What makes two variants the same question: the setting with the input, and, for a value question
    on a single input, the setting with the asked input item."""
    keys = {json.dumps([spec["params"], spec.get("inputs"), spec.get("count_input")], sort_keys=True, default=str)}
    if "inputs" in spec and len(spec["inputs"]) == 1 and "k" in spec and len(spec["inputs"][0]) >= spec["k"]:
        keys.add(json.dumps([spec["params"], "item", spec["inputs"][0][spec["k"] - 1]], sort_keys=True, default=str))
    return keys


def _prepare(grc: Grc, c: dict) -> dict:
    """GRC work for one candidate (sequential: the Platform API is not shared across threads)."""
    block, spec = c["block"], c["spec"]
    ok, errs = grc.valid(block, spec["params"])
    if not ok:
        return {**c, "status": "invalid setting", "detail": "; ".join(errs)[:200]}
    ins, outs = _ports(grc, block, spec["params"])
    probe = probe_graph(block, spec, ins, outs)
    if isinstance(probe, str):
        return {**c, "status": "invalid setting", "detail": probe}
    return {**c, "status": "prepared", "ins": ins, "outs": outs, "graph": probe[0], "task": probe[1]}


def _run(c: dict) -> dict:
    """Two sandboxed runs of a prepared candidate (thread-safe)."""
    if c["status"] != "prepared":
        return c
    runs = [run_once(c["graph"], c["task"], c["spec"]["port"], f"{c['block']}-{c['variant']}-{i}") for i in range(2)]
    return {**c, "runs": runs}


def _evaluate(c: dict) -> dict:
    if c["status"] != "prepared":
        return c
    spec, kind, ins, outs, runs = c["spec"], c["kind"], c["ins"], c["outs"], c["runs"]
    if any(r[0] is None for r in runs):
        return {**c, "status": "run failed", "detail": next(r[1] for r in runs if r[0] is None)}
    if runs[0][0] != runs[1][0]:
        return {**c, "status": "not deterministic", "detail": "two runs differ"}
    out_dtype = outs[spec["port"]][0]
    got = _answer(spec, kind, runs[0][0], out_dtype)
    expect = float(spec["expect"])
    if got is None:
        return {**c, "status": "too few output items", "detail": f"{len(runs[0][0])} items"}
    integer = _is_int(kind, out_dtype)
    answer = float(round(got)) if integer else float(f"{got:.6g}")
    tol = _tol(kind, out_dtype, answer)
    if not _agrees(got, expect, tol):
        return {**c, "status": "run and model disagree", "detail": f"GNU Radio {got:g}, model {expect:g}"}
    return {**c, "status": "ok", "answer": answer, "integer": integer, "tolerance": tol, "ins": ins, "outs": outs}


def build(workers: int = WORKERS, only: set[str] | None = None) -> dict[str, Any]:
    from grcbench.tier_k import catalog as cat
    part = {b["id"]: b for b in json.loads((SPLITS / "block_partition.json").read_text(encoding="utf-8"))["blocks"]}
    grc = Grc()
    executed = _executed_blocks()
    cands, skipped = _candidates(grc, part, executed, only)
    prepared = [_prepare(grc, c) for c in cands]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        ran = list(pool.map(_run, prepared))
    results = [_evaluate(c) for c in ran]
    trees = cat.tree_modules()
    bank = _bank()
    questions, dropped = [], []
    for r in results:
        if r["status"] != "ok":
            dropped.append({"block": r["block"], "variant": r["variant"], "kind": r["kind"], "reason": r["status"],
                            "detail": r.get("detail", "")})
            continue
        cls = grc.platform.blocks[r["block"]]
        rec = cat.block_record(cls, trees)
        hit = _in_bank(bank, cls.label, r["block"], r["kind"])
        if hit:
            dropped.append({"block": r["block"], "variant": r["variant"], "kind": r["kind"], "reason": "bank duplicate",
                            "detail": hit})
            continue
        q = {"block": r["block"], "variant": r["variant"], "kind": r["kind"], "label": cls.label,
             "family": part[r["block"]]["family"], "yaml_file": rec["yaml_file"], "yaml_sha256": rec["yaml_sha256"],
             "difficulty": r["difficulty"], "answer": r["answer"], "tolerance": r["tolerance"], "integer": r["integer"],
             "setting": r["spec"]["params"],
             "question": question_text(cls, r["block"], r["kind"], r["spec"], r["ins"], r["outs"])}
        q["evidence"] = (f"GNU Radio 3.10.12 run (twice, identical): {q['answer']:g}; the reference model of the "
                         f"block's documented behaviour agrees")
        problem = _rules_check(q)
        if problem:
            dropped.append({"block": r["block"], "variant": r["variant"], "kind": r["kind"], "reason": "not scorable",
                            "detail": problem})
            continue
        questions.append(q)
    return {"questions": select(questions), "candidates": len(cands), "dropped": dropped, "skipped": skipped,
            "executed_blocks": sorted(executed)}


def select(questions: list[dict]) -> list[dict]:
    """At most MAX_PER_BLOCK questions per block (lowest variants first), then up to TARGET ranked by
    sha256(SEED + question id)."""
    per_block: dict[str, list[dict]] = {}
    for q in sorted(questions, key=lambda q: (q["block"], q["variant"])):
        if len(per_block.setdefault(q["block"], [])) < MAX_PER_BLOCK:
            per_block[q["block"]].append(q)
    pool = [q for qs in per_block.values() for q in qs]
    pool.sort(key=lambda q: hashlib.sha256(f"{SEED}\0{_qid(q['block'], q['variant'])}".encode()).hexdigest())
    return sorted(pool[:TARGET], key=lambda q: (q["block"], q["variant"]))


def records(doc: dict) -> tuple[list[dict], list[dict]]:
    items, keys = [], []
    for q in doc["questions"]:
        qid = _qid(q["block"], q["variant"])
        item = {"id": qid, "tier": "K", "split": "test", "version": VERSION, "canary": CANARY, "license": LICENSE,
                "source": {"kind": "gnuradio_run", "ref": f"gnuradio-3.10.12/grc/blocks/{q['yaml_file']}",
                           "sha256": q["yaml_sha256"], "block_id": q["block"]},
                "cluster_id": f"block_family:{q['family']}",
                "provenance": {"drafted_by": "script (grcbench.tier_k.blockgen); settings and question templates by "
                                             "Claude (Anthropic); answers from GNU Radio runs",
                               "generator": GENERATOR, "generator_version": f"{VERSION} ({SEED})"},
                "review": {"status": "pending-team-review", "reviewers": [], "notes": ""},
                "question": q["question"], "task_type": "gnuradio_usage", "difficulty": q["difficulty"],
                "item_kind": "block_question", "block_question_kind": q["kind"]}
        errors = validate(item)
        if errors:
            raise ValueError(f"{qid}: {errors}")
        items.append(item)
        keys.append({"question_id": qid, "canary": CANARY, "kind": q["kind"], "block_id": q["block"],
                     "setting": q["setting"], "reference": reference(q), "answer": q["answer"],
                     "evidence": q["evidence"]})
    return items, keys


def sheet(items: list[dict], keys: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["question_id", "kind", "block_id", "question", "answer", "evidence", "reviewer_1_wording",
                "reviewer_1_note", "reviewer_2_wording", "reviewer_2_note", "final_decision"])
    for it, k in zip(items, keys):
        w.writerow([it["id"], k["kind"], k["block_id"], it["question"], f"{k['answer']:g}", k["evidence"],
                    "", "", "", "", ""])
    return buf.getvalue()


def report(doc: dict, items: list[dict]) -> str:
    kinds = Counter(i["block_question_kind"] for i in items)
    blocks = Counter(i["source"]["block_id"] for i in items)
    reasons = Counter(d["reason"] for d in doc["dropped"])
    L = ["# Tier K: GNU Radio behaviour questions", "",
         f"Generated by `./gb blockq` ({GENERATOR}, {SEED}). Status: pending team review of the wording "
         "(`handoff/review_sheets/tier_k_block_questions.csv`) and the Qoherent expert review.", "",
         f"**{len(items)} test questions about {len(blocks)} of the 125 test-side blocks** "
         "(`splits/block_partition.json`), at most "
         f"{MAX_PER_BLOCK} per block. Each asks for one number that comes from what the block does on a stated input: "
         "the k-th output item, or how many items it produces. The answer key is GNU Radio's own output: the "
         "generator runs the block (finite Vector Source(s) -> block -> File Sink(s)) in the sandbox and reads the "
         "recording. A question is kept only when two runs agree, an independent model of the block's documented "
         "behaviour gives the same number, and the rule cascade gives the key full credit and a wrong value less.", "",
         "Why these questions: the agent traces used for training show the catalog entries of almost every block, "
         "but no trace ran a flowgraph (`reports/agent_traces.md`), so behaviour on a given input is held out. "
         "Tier K is closed-book: the model under test sees only the question text.", "",
         "| Kind | Questions | Answer format |", "|---|---:|---|",
         f"| output_value | {kinds.get('output_value', 0)} | one number (floats within 1 % or 0.005; integers exact) |",
         f"| output_count | {kinds.get('output_count', 0)} | a count (exact) |", "",
         f"Candidates: {doc['candidates']}. Dropped: {len(doc['dropped'])} ("
         + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())) + ").", ""]
    if doc["dropped"]:
        L += ["| Block | Variant | Reason | Detail |", "|---|--:|---|---|"]
        L += [f"| {d['block']} | {d['variant']} | {d['reason']} | {str(d['detail']).replace('|', '/')[:150]} |"
              for d in doc["dropped"]]
        L.append("")
    if doc["skipped"]:
        L += ["Blocks without questions:", ""]
        L += [f"- {s['block']}: {s['reason']}" for s in doc["skipped"]]
        L.append("")
    L += ["Questions per block: " + ", ".join(f"{b} {n}" for b, n in sorted(blocks.items())) + "."]
    return "\n".join(L) + "\n"


def outputs(doc: dict) -> dict[str, str]:
    items, keys = records(doc)
    return {OUT_QUESTIONS: "".join(json.dumps(i, ensure_ascii=False, sort_keys=True) + "\n" for i in items),
            OUT_ANSWERS: "".join(json.dumps(k, ensure_ascii=False, sort_keys=True) + "\n" for k in keys),
            OUT_SHEET: sheet(items, keys), OUT_REPORT: report(doc, items)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb blockq", description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO, help="root folder for the outputs (default: the repo)")
    ap.add_argument("--workers", type=int, default=WORKERS, help="GNU Radio runs in parallel")
    ap.add_argument("--check", action="store_true", help="compare a fresh build with the files on disk")
    ap.add_argument("--only", default=None, help="comma-separated block ids (development; use with --out)")
    args = ap.parse_args(argv)
    doc = build(workers=args.workers, only=set(args.only.split(",")) if args.only else None)
    files = outputs(doc)
    if args.check:
        stale = [rel for rel, text in files.items()
                 if not (REPO / rel).exists() or (REPO / rel).read_text(encoding="utf-8") != text]
        print("blockq --check: " + ("OK" if not stale else "differs: " + ", ".join(stale)))
        return 1 if stale else 0
    for rel, text in files.items():
        write_text(args.out / rel, text)
    print(f"wrote {len(doc['questions'])} behaviour questions from {doc['candidates']} candidates "
          f"({len(doc['dropped'])} dropped) under {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
