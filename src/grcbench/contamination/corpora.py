"""Loaders for the two sides of the contamination audit (eval plan §6; plan WS2 "compared sets").

Benchmark side (one "tier" each; a tier whose files do not exist yet is reported as absent):
  tier_k_test   the 185 frozen test questions, plus any items/tier_k/test/*.jsonl records
  tier_k_dev    the dev questions named in splits/tier_k_dev.json, plus items/tier_k/dev/*.jsonl
  tier_f_test / tier_f_dev   the `instruction` of items/tier_f/<split>/**/task.json
  tier_a_test / tier_a_dev   the `instruction` of items/tier_a/<split>/**/task.json; tier_a_test also
                             holds the test tasks drafted as specs only (items/tier_a/test_spec_drafts.jsonl)

Training side (every benchmark tier is compared with every corpus):
  train_questions  the 1,668 frozen training questions minus the dev ids
  teacher_answers  05_answers/<teacher>/answers.jsonl, answer + rationale; duplicate
                   (question_id, model) rows keep the last row (laguna-s21 has 120 re-runs)
  repaired         10_repaired/repaired_answers.jsonl (repaired_answer)
  repaired_glm     10_repaired_glm/repaired_answers.jsonl (repaired_answer)
  sft_sessions     grc-agent-sft-v0.1.0 sft_train.jsonl + quarantine.jsonl: all message text
                   (the session's system prompt, then content, thinking, tool-call names and
                   arguments of every message)
  task_specs       reports/reference/training_task_specs.json goals (Qoherent collection tasks,
                   GRC-Agent scenarios, the SFT sessions' tasks): what Tier-A instructions must avoid
  agent_traces     Qoherent's 1,853 agent traces of 15-18 Oct (new_data/, tier1 + tier2): every message's
                   text (user prompts, assistant content and reasoning, tool calls and tool results); the
                   system prompt, the same in every trace, is one document of its own

Dev questions are excluded from every training use (handoff to the training team), so answer and
repair rows whose question is a dev question are dropped as well, and the number dropped is
reported. All inputs are read-only; nothing here writes.

A document is compared as passages: a document of at most PASSAGE_SINGLE_MAX words is one passage
(its whole text, so a training question is embedded exactly as a test question is); a longer one is
cut into windows of PASSAGE_WORDS words every PASSAGE_STRIDE words (the last window ends at the last
word). Passages keep a short question inside a long answer or session visible to
the cosine signals; the 8-gram signal always reads whole documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from grcbench.common.io import iter_jsonl, read_json, sha256_file
from grcbench.common.paths import AGENT_TRACES, DATASET, FROZEN, ITEMS, REPO, REPORTS, SFT_RELEASE, SPLITS

PASSAGE_WORDS = 64
PASSAGE_STRIDE = 32
PASSAGE_SINGLE_MAX = PASSAGE_WORDS + PASSAGE_STRIDE // 2   # 80 words: every frozen question is one passage

TIERS = ("tier_k_test", "tier_k_dev", "tier_f_test", "tier_f_dev", "tier_a_test", "tier_a_dev")
CORPORA = ("train_questions", "teacher_answers", "repaired", "repaired_glm", "sft_sessions", "task_specs",
           "agent_traces")

DEV_SPLIT = SPLITS / "tier_k_dev.json"
TASK_SPECS = REPORTS / "reference" / "training_task_specs.json"

_WORD = re.compile(r"\w+")
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Doc:
    id: str
    text: str
    ref: str = ""                       # e.g. the question id an answer belongs to
    aliases: tuple[str, ...] = ()       # other item ids with the same text (benchmark side)


@dataclass
class Corpus:
    name: str
    description: str
    docs: list[Doc]
    sources: list[dict[str, str]] = field(default_factory=list)   # path (repo-relative) + sha256
    notes: list[str] = field(default_factory=list)
    embed_all: bool = False             # short texts: embed every passage (else TF-IDF candidates)


@dataclass(frozen=True)
class Passage:
    doc: int                            # index into Corpus.docs
    start: int                          # character span in the document text
    end: int


# --------------------------------------------------------------------------- helpers ------------

def clean(text: str) -> str:
    """Collapse whitespace; the form that is embedded, hashed for the cache and shown in snippets."""
    return _SPACE.sub(" ", text).strip()


def snippet(text: str, limit: int = 200) -> str:
    t = clean(text)
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.parent.resolve()).as_posix()
    except ValueError:
        return str(path)


def _source(path: Path) -> dict[str, str]:
    return {"path": _rel(path), "sha256": sha256_file(path)}


def passages(text: str, words: int = PASSAGE_WORDS, stride: int = PASSAGE_STRIDE,
             single_max: int = PASSAGE_SINGLE_MAX) -> list[tuple[int, int]]:
    """Character spans of the passages of one document (see module docstring)."""
    spans = [m.span() for m in _WORD.finditer(text)]
    if not spans:
        return []
    if len(spans) <= single_max:
        return [(0, len(text))]           # the whole text, embedded exactly like a benchmark item
    starts = list(range(0, len(spans) - words, stride)) + [len(spans) - words]
    return [(spans[s][0], spans[s + words - 1][1]) for s in starts]


def corpus_passages(corpus: Corpus) -> list[Passage]:
    return [Passage(i, a, b) for i, d in enumerate(corpus.docs) for a, b in passages(d.text)]


def passage_text(corpus: Corpus, p: Passage) -> str:
    return corpus.docs[p.doc].text[p.start:p.end]


# --------------------------------------------------------------------------- dev split ----------

def dev_ids(path: Path = DEV_SPLIT) -> set[str] | None:
    """Dev question ids from splits/tier_k_dev.json, or None while the dev split is not drawn."""
    if not path.exists():
        return None
    data = read_json(path)
    if isinstance(data, list):
        ids = data
    else:
        ids = next((data[k] for k in ("dev_ids", "ids", "question_ids") if k in data), [])
    return {i if isinstance(i, str) else i["id"] for i in ids}


# --------------------------------------------------------------------------- benchmark side -----

def _frozen(name: str) -> list[dict[str, Any]]:
    return list(iter_jsonl(FROZEN / name))


def _merge(base: list[Doc], extra: list[Doc]) -> list[Doc]:
    """Add item records to a question list; a record with an already-present text becomes an alias."""
    by_text = {clean(d.text): k for k, d in enumerate(base)}
    out = list(base)
    for d in extra:
        k = by_text.get(clean(d.text))
        if k is None:
            by_text[clean(d.text)] = len(out)
            out.append(d)
        elif d.id != out[k].id and d.id not in out[k].aliases:
            out[k] = Doc(out[k].id, out[k].text, out[k].ref, tuple(sorted(out[k].aliases + (d.id,))))
    return out


def _item_records(folder: Path) -> tuple[list[Doc], list[dict[str, str]]]:
    docs, sources = [], []
    for path in sorted(folder.glob("*.jsonl")):
        sources.append(_source(path))
        for r in iter_jsonl(path):
            if r.get("question"):
                docs.append(Doc(r["id"], r["question"], ref=str((r.get("source") or {}).get("ref", ""))))
    return docs, sources


def _task_records(folder: Path) -> tuple[list[Doc], list[dict[str, str]]]:
    docs, sources = [], []
    for path in sorted(folder.rglob("task.json")):
        r = read_json(path)
        if r.get("instruction"):
            docs.append(Doc(r.get("id") or path.parent.name, r["instruction"]))
            sources.append(_source(path))
    return docs, sources


def benchmark_tier(name: str, dev: set[str] | None = None) -> Corpus | None:
    """The items of one benchmark tier, or None if the tier has no items yet."""
    tier, split = name.split("_")[1], name.split("_")[2]
    if tier == "k":
        if split == "test":
            base = [Doc(r["id"], r["question"]) for r in _frozen("eval_questions.jsonl")]
            sources = [_source(FROZEN / "eval_questions.jsonl")]
            desc = "185 frozen test questions (04_eval_frozen) plus items/tier_k/test records"
        else:
            base, sources = [], []
            if dev:
                base = [Doc(r["id"], r["question"]) for r in _frozen("train_questions.jsonl") if r["id"] in dev]
                sources = [_source(DEV_SPLIT), _source(FROZEN / "train_questions.jsonl")]
            desc = "dev questions named in splits/tier_k_dev.json plus items/tier_k/dev records"
        extra, extra_src = _item_records(ITEMS / "tier_k" / split)
        docs = _merge(base, extra)
        sources += extra_src
    else:
        docs, sources = _task_records(ITEMS / f"tier_{tier}" / split)
        desc = f"instructions of items/tier_{tier}/{split}/**/task.json"
        drafts = ITEMS / f"tier_{tier}" / f"{split}_spec_drafts.jsonl"
        if drafts.is_file():                    # Tier-A test tasks drafted as specs only
            docs += [Doc(r["id"], r["instruction"]) for r in iter_jsonl(drafts) if r.get("instruction")]
            sources.append(_source(drafts))
            desc += f" and of the specs in items/tier_{tier}/{drafts.name}"
    if not docs:
        return None
    return Corpus(name, desc, docs, sources)


def absent_reason(name: str, dev: set[str] | None) -> str:
    if name == "tier_k_dev" and dev is None:
        return "splits/tier_k_dev.json does not exist yet (dev split not drawn)"
    tier, split = name.split("_")[1], name.split("_")[2]
    folder = f"items/tier_{tier}/{split}/"
    return f"no items under {folder} yet"


# --------------------------------------------------------------------------- training side ------

def _train_questions(dev: set[str] | None) -> Corpus:
    path = FROZEN / "train_questions.jsonl"
    rows = _frozen("train_questions.jsonl")
    keep = [r for r in rows if not dev or r["id"] not in dev]
    notes = [f"{len(rows)} training questions; {len(rows) - len(keep)} dev questions removed"
             if dev is not None else
             f"{len(rows)} training questions; dev split not drawn yet, so no dev questions removed"]
    return Corpus("train_questions", "frozen training questions minus the dev split",
                  [Doc(r["id"], r["question"]) for r in keep], [_source(path)], notes, embed_all=True)


def _teacher_answers(dev: set[str] | None) -> Corpus:
    files = sorted(p for p in (DATASET / "05_answers").glob("*/answers.jsonl"))
    docs: dict[tuple[str, str], Doc] = {}
    dupes = dev_dropped = empty = 0
    for path in files:
        for r in iter_jsonl(path):
            key = (r["question_id"], r["model"])
            if dev and r["question_id"] in dev:
                dev_dropped += 1
                continue
            text = "\n\n".join(t for t in ((r.get("answer") or "").strip(), (r.get("rationale") or "").strip()) if t)
            if not text:
                empty += 1
                continue
            if key in docs:
                dupes += 1
            docs[key] = Doc(f"{r['model']}/{r['question_id']}", text, ref=r["question_id"])   # last row wins
    notes = [f"{len(files)} teachers; {len(docs)} (question_id, model) answers",
             f"{dupes} duplicate (question_id, model) rows replaced by the later row (laguna-s21 re-runs)",
             f"{dev_dropped} rows for dev questions dropped", f"{empty} rows with no text skipped"]
    return Corpus("teacher_answers", "teacher answers and rationales (05_answers)",
                  [docs[k] for k in sorted(docs)], [_source(p) for p in files], notes)


def _repaired(folder: str, name: str, dev: set[str] | None) -> Corpus:
    path = DATASET / folder / "repaired_answers.jsonl"
    docs, dev_dropped = [], 0
    for r in iter_jsonl(path):
        if dev and r["question_id"] in dev:
            dev_dropped += 1
            continue
        if (r.get("repaired_answer") or "").strip():
            docs.append(Doc(f"{r['candidate_model']}/{r['question_id']}", r["repaired_answer"], ref=r["question_id"]))
    docs.sort(key=lambda d: d.id)
    return Corpus(name, f"repaired answers ({folder})", docs, [_source(path)],
                  [f"{len(docs)} repaired answers; {dev_dropped} rows for dev questions dropped"])


def _message_text(m: dict[str, Any]) -> Iterator[str]:
    for key in ("content", "thinking", "reasoning"):
        if isinstance(m.get(key), str) and m[key].strip():
            yield m[key]
    for call in m.get("tool_calls") or []:
        fn = call.get("function") or call
        yield f"{fn.get('name', '')} {fn.get('arguments', '')}".strip()


def session_text(row: dict[str, Any]) -> str:
    """All text of one SFT row: its system prompt (a top-level field, the first turn the model is
    trained with), then every message."""
    system = row.get("system") if isinstance(row.get("system"), str) else ""
    parts = ([system] if system.strip() else []) + [t for m in row.get("messages", []) for t in _message_text(m)]
    return "\n\n".join(parts)


def _sft_sessions(dev: set[str] | None) -> Corpus:
    docs, sources, counts = [], [], []
    for fname in ("sft_train.jsonl", "quarantine.jsonl"):
        path = SFT_RELEASE / fname
        sources.append(_source(path))
        n = 0
        for i, r in enumerate(iter_jsonl(path)):
            docs.append(Doc(f"{path.stem}/{i:02d}/{r.get('session_id', '')}", session_text(r),
                            ref=str(r.get("task_id", ""))))
            n += 1
        counts.append(f"{n} {path.stem}")
    return Corpus("sft_sessions", "SFT sessions and quarantine (grc-agent-sft-v0.1.0), all message text",
                  docs, sources, [f"{len(docs)} sessions ({' + '.join(counts)}); system prompt included"])


def _task_specs(dev: set[str] | None) -> Corpus:
    data = read_json(TASK_SPECS)
    docs = [Doc(f"{s.get('source')}/{s.get('id')}", s["goal"]) for s in data["specs"] if (s.get("goal") or "").strip()]
    by_source: dict[str, int] = {}
    for s in data["specs"]:
        by_source[s.get("source")] = by_source.get(s.get("source"), 0) + 1
    return Corpus("task_specs", "training-side agent task specs (Qoherent tasks, GRC-Agent scenarios, SFT tasks)",
                  sorted(docs, key=lambda d: d.id), [_source(TASK_SPECS)],
                  [f"{len(docs)} goals; " + ", ".join(f"{k} {v}" for k, v in sorted(by_source.items()))],
                  embed_all=True)


def _agent_traces(dev: set[str] | None) -> Corpus:
    docs, sources, counts, systems = [], [], [], set()
    for tier, path in AGENT_TRACES.items():
        sources.append(_source(path))
        n = 0
        for r in iter_jsonl(path):
            systems.add(r.get("system") or "")
            text = "\n\n".join(t for m in r.get("messages", []) if m.get("role") != "system" for t in _message_text(m))
            docs.append(Doc(f"{tier}/{r['scenario_id']}", text, ref=r["scenario_id"]))
            n += 1
        counts.append(f"{n} {tier}")
    first = min(systems) if systems else ""
    common = "\n".join(line for line in first.splitlines()
                       if not line.startswith(("Session ID", "Active flowgraph file path")))
    docs.append(Doc("system_prompt", common, ref="system_prompt"))
    return Corpus("agent_traces", "Qoherent's agent traces of 15-18 Oct (new_data/), all message text", docs, sources,
                  [f"{len(docs) - 1} traces ({' + '.join(counts)}); the shared system prompt is one document"])


LOADERS = {
    "train_questions": _train_questions,
    "teacher_answers": _teacher_answers,
    "repaired": lambda dev: _repaired("10_repaired", "repaired", dev),
    "repaired_glm": lambda dev: _repaired("10_repaired_glm", "repaired_glm", dev),
    "sft_sessions": _sft_sessions,
    "task_specs": _task_specs,
    "agent_traces": _agent_traces,
}


def training_corpus(name: str, dev: set[str] | None) -> Corpus:
    return LOADERS[name](dev)
