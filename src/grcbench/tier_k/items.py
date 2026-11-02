"""`./gb tier-k-items` writes the Tier-K test and dev item records (eval plan §4.1, §4.2, §5.1, §6, §9).

One schema-valid record (grcbench.common.schemas, tier K) per question, built only from files:

  items/tier_k/test/frozen_questions.jsonl   the 185 frozen test questions (04_eval_frozen), manifest order
  items/tier_k/dev/questions.jsonl           the dev questions (splits/tier_k_dev.json), dev_ids order
  reports/tier_k_items.md                    counts, cluster rule, near-twin reporting split, writers
  handoff/review_sheets/tier_k_test_clusters.csv   multi-member test clusters for the team to confirm

Fields beyond the shared ones (id, tier, split, version, canary, license, source, cluster_id,
provenance, review):

  question, difficulty, origin, primary_topic   copied unchanged from the frozen dataset files
  task_type            clean label from splits/task_type_labels.jsonl (eval plan §4.2 "Labels")
  task_type_raw        the raw dataset spelling, never overwritten
  task_type_labeller   who mapped it: the rule map, or Claude (Anthropic) for the 134 ambiguous rows
                       (test ones pending blind double labelling by the team)
  item_kind            frozen_question (test) | bank_question (dev)
  near_twin_in_train   eval plan §6 reporting split: the question shares a word 8-gram (tokenizer of
                       splits/twins.py) with a training question. Test: against the 1,668 frozen training
                       questions (26 test questions, the plan's count). Dev: against the training
                       questions minus dev (none, by construction of the dev split).
  near_twin_train_ids  those training questions
  train_neighbour_ids  flagged only, not counted in §6: training questions linked by semantic, lexical or
                       keep_both evidence (thresholds of splits/tier_k_twin_groups.json) without a shared
                       8-gram. Test: splits/tier_k_dev.json test_neighbours; dev: twin-group edges.
  subsets              dev only: "anchor" and/or "ci" membership (splits/tier_k_dev.json)

cluster_id (eval plan §9, clustered intervals):
  dev   the rule in splits/tier_k_twin_groups.json: a question in a twin group takes the group id
        (tg_...); every other question is a singleton with its own id. No dev question is in a group.
  test  the same evidence and thresholds as the training twin groups, applied to test<->test pairs:
        shared word 8-gram (template families), semantic score >= semantic_threshold, lexical TF-IDF
        >= lexical_threshold (pairs the semantic file did not score) and KEEP_BOTH decisions. A
        connected component of two or more test questions gets "tc_" + the hex part of its smallest
        id; every other test question is a singleton with its own id. The 8-gram-only clustering is
        reported beside it; the team confirms the clusters with the review sheet.

provenance (eval plan §5.1, Table 5: who wrote the questions):
  drafted_by           the question-writing model, readable name plus where it is recorded: source lane
                       "Mistral Large (source2_mistral_large.json)" from provenance[0].source_file; blind
                       lane "Gemma 4 (generator gemma4, model gemma)" from the generator/model fields
  writer_family        Mistral | Gemma | gpt-oss | Qwen | Nemotron (Table 5: a teacher's Tier-K score is
                       also reported without the questions its family wrote)
  writers              every distinct writer in the record's provenance (merged duplicates keep all)
  generator            "GRC_dataset_20261007"
  generator_version    first 16 hex digits of the SHA-256 of 03_questions_merged/questions_canonical.jsonl
                       (the eval manifest's source_canonical_sha256, checked against the file)

source.ref is relative to project/data/ (as in splits/tier_k_dev.json "sources"), source.sha256 is the
SHA-256 of that file. Question text is never changed, so the contamination audit's loader
(contamination/corpora.py) treats these records as the same documents as the frozen questions.

Similarity evidence never removes a question (dataset README); it only sets cluster ids and flags.
Inputs are read-only and nothing is sent to any model. Outputs are deterministic (sorted keys, stable
order, no timestamps); a re-run gives byte-identical files.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from grcbench.common.canary import CANARY
from grcbench.common.io import read_json, read_jsonl, sha256_file, write_jsonl, write_text
from grcbench.common.paths import DATA, FROZEN, REPO, SPLITS
from grcbench.common.schemas import validate
from grcbench.splits import twins
from grcbench.splits.frozen import CANONICAL, MANIFEST

VERSION = 1
GENERATOR = "GRC_dataset_20261007"
GENERATED_BY = "grcbench.tier_k.items (./gb tier-k-items)"

TEST_FILE = FROZEN / "eval_questions.jsonl"
TRAIN_FILE = FROZEN / "train_questions.jsonl"
DEV_SPLIT = SPLITS / "tier_k_dev.json"
TWIN_GROUPS = SPLITS / "tier_k_twin_groups.json"
LABELS = SPLITS / "task_type_labels.jsonl"

OUT_TEST = Path("items/tier_k/test/frozen_questions.jsonl")
OUT_DEV = Path("items/tier_k/dev/questions.jsonl")
OUT_REPORT = Path("reports/tier_k_items.md")
OUT_SHEET = Path("handoff/review_sheets/tier_k_test_clusters.csv")

TEST_CLUSTER_PREFIX = "tc_"
KIND = {"test": "frozen_question", "dev": "bank_question"}
LABEL_SPLIT = {"test": "eval", "dev": "train"}
REVIEW_NOTES = {
    "test": "frozen dataset question; rubric review happens separately",
    "dev": "frozen dataset question (training bank, held out as dev); rubric review happens separately",
}

# Licence: the dataset README states no licence for its question records (checked 15 Oct 2026).
LICENSE_BASE = ("licence: as in GRC_dataset_20261007 (Qoherent); to be confirmed before release. Question copied "
                "unchanged from the GRC_dataset_20261007 release by Qoherent (its README states no licence for "
                "question records); ")
LICENSE = {
    "source": LICENSE_BASE + ("source-lane question, model-generated from one licensed reference (dataset terms "
                              "include CC BY-SA, CC BY-NC-SA 4.0, per-post Stack Exchange and public-use) "
                              "without reproducing its text."),
    "blind": LICENSE_BASE + "blind-lane question, model-generated without source material.",
}

# Readable names of the question writers recorded in the dataset (eval plan §5.1, Table 5). Every value
# met in the data must be listed here; an unknown writer stops the build instead of guessing.
FAMILY = {"mistral": "Mistral", "gemma4": "Gemma", "gptoss": "gpt-oss", "qwen": "Qwen", "nemotron": "Nemotron"}
SOURCE_MODELS = {"mistral_large": "Mistral Large", "mistral_large2": "Mistral Large 2",
                 "mistral_large3": "Mistral Large 3", "mistral_vibe": "Mistral Vibe"}
BLIND_MODELS = {("mistral", "mistral_vibe"): "Mistral Vibe", ("mistral", "mistral_small"): "Mistral Small",
                ("gemma4", "gemma"): "Gemma 4", ("gptoss", "gptoss"): "gpt-oss", ("qwen", "qwen"): "Qwen",
                ("nemotron", "nemotron"): "Nemotron"}
_SOURCE_FILE = re.compile(r"source\d+_([a-z0-9_]+)\.json")


def _rel(path: Path) -> str:
    return path.relative_to(DATA).as_posix()


# ---------------------------------------------------------------- inputs -------------------------

@dataclass
class Inputs:
    test: list[dict[str, Any]]
    train: list[dict[str, Any]]
    manifest: dict[str, Any]
    dev_split: dict[str, Any]
    twin_groups: dict[str, Any]
    labels: dict[str, dict[str, Any]]
    hashes: dict[str, str]                    # data-relative or repo-relative path -> sha256


def load() -> Inputs:
    hashes = {_rel(p): sha256_file(p) for p in (TEST_FILE, TRAIN_FILE, MANIFEST, CANONICAL)}
    hashes.update({p.relative_to(REPO).as_posix(): sha256_file(p) for p in (DEV_SPLIT, TWIN_GROUPS, LABELS)})
    return Inputs(test=read_jsonl(TEST_FILE), train=read_jsonl(TRAIN_FILE), manifest=read_json(MANIFEST),
                  dev_split=read_json(DEV_SPLIT), twin_groups=read_json(TWIN_GROUPS),
                  labels={r["id"]: r for r in read_jsonl(LABELS)}, hashes=hashes)


def check_inputs(inp: Inputs) -> None:
    """Refuse to build from inconsistent inputs (stale split files, changed dataset)."""
    m, d, g = inp.manifest, inp.dev_split, inp.twin_groups
    problems = []
    if inp.hashes[_rel(CANONICAL)] != m["source_canonical_sha256"]:
        problems.append("questions_canonical.jsonl differs from the eval manifest's source hash")
    if [r["id"] for r in inp.test] != m["eval_ids"]:
        problems.append("eval_questions.jsonl ids differ from eval_manifest.json eval_ids")
    recorded = {s["path"]: s["sha256"] for s in d["sources"]}
    for p in (TEST_FILE, TRAIN_FILE, MANIFEST):
        if recorded.get(_rel(p)) != inp.hashes[_rel(p)]:
            problems.append(f"splits/tier_k_dev.json was drawn from another {_rel(p)}: re-run ./gb split-dev")
    train_ids = {r["id"] for r in inp.train}
    if len(set(d["dev_ids"])) != len(d["dev_ids"]) or not set(d["dev_ids"]) <= train_ids:
        problems.append("dev_ids are not distinct training questions")
    if (g["semantic_threshold"], g["lexical_threshold"]) != (d["twin_groups"]["semantic_threshold"],
                                                             d["twin_groups"]["lexical_threshold"]):
        problems.append("tier_k_twin_groups.json and tier_k_dev.json use different thresholds")
    missing = [r["id"] for r in inp.test + inp.train if r["id"] not in inp.labels]
    if missing:
        problems.append(f"{len(missing)} questions have no row in splits/task_type_labels.jsonl")
    if problems:
        raise SystemExit("tier-k-items: inconsistent inputs:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------- writers (§5.1) -----------------

def _blind_writer(generator: str, model: str) -> tuple[str, str]:
    """(family, readable model name) of a blind-lane generator/model pair."""
    if (generator, model) not in BLIND_MODELS or generator not in FAMILY:
        raise SystemExit(f"tier-k-items: unknown blind-lane writer {(generator, model)}; add it to BLIND_MODELS")
    return FAMILY[generator], BLIND_MODELS[(generator, model)]


def _source_writer(source_file: str) -> tuple[str, str]:
    """(family, readable model name) of a source-lane file such as source2_mistral_large.json."""
    m = _SOURCE_FILE.fullmatch(source_file)
    if not m or m.group(1) not in SOURCE_MODELS:
        raise SystemExit(f"tier-k-items: unknown source-lane writer {source_file!r}; add it to SOURCE_MODELS")
    return "Mistral", SOURCE_MODELS[m.group(1)]


def writer_info(r: Mapping[str, Any]) -> dict[str, Any]:
    """drafted_by, writer_family and writers of a frozen question record."""
    entries = []                                                  # (family, "Name (file)") per provenance entry
    for p in r["provenance"]:
        fam, name = (_source_writer(p["source_file"]) if r["origin"] == "source"
                     else _blind_writer(p["generator"], p["model"]))
        entries.append((fam, f"{name} ({p['source_file']})"))
    if r["origin"] == "source":
        drafted = entries[0][1]                                   # provenance[0].source_file
    else:
        if (r["generator"], r["model"]) != (r["provenance"][0]["generator"], r["provenance"][0]["model"]):
            raise SystemExit(f"tier-k-items: {r['id']} generator/model disagree with provenance[0]")
        drafted = f"{_blind_writer(r['generator'], r['model'])[1]} (generator {r['generator']}, model {r['model']})"
    return {"drafted_by": drafted, "writer_family": " + ".join(sorted({f for f, _ in entries})),
            "writers": sorted({label for _, label in entries})}


# ---------------------------------------------------------------- near twins (§6) ----------------

def invert(mapping: Mapping[str, Any]) -> dict[str, list[str]]:
    """{a: [b, ...]} or {a: {b: ...}} -> {b: sorted [a, ...]}."""
    out: dict[str, set[str]] = defaultdict(set)
    for a, bs in mapping.items():
        for b in bs:
            out[b].add(a)
    return {b: sorted(a) for b, a in sorted(out.items())}


def near_twins_of_test(inp: Inputs) -> dict[str, list[str]]:
    """Test id -> training ids sharing a word 8-gram (recomputed, then checked against tier_k_dev.json)."""
    hits = invert(twins.ngram_hits(inp.test, inp.train))
    recorded = invert(inp.dev_split["test_twins"]["by_training_id"])
    if hits != recorded:
        raise SystemExit("tier-k-items: recomputed test twins differ from splits/tier_k_dev.json: re-run ./gb split-dev")
    return hits


def near_twins_of_dev(inp: Inputs) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Dev id -> (training-minus-dev ids sharing an 8-gram; ids linked by other twin-group edges)."""
    dev = set(inp.dev_split["dev_ids"])
    dev_recs = [r for r in inp.train if r["id"] in dev]
    rest = [r for r in inp.train if r["id"] not in dev]
    ngram = invert(twins.ngram_hits(dev_recs, rest))
    other: dict[str, set[str]] = defaultdict(set)
    for a, b, via in inp.twin_groups["edges"]:
        if any(not v.startswith("8gram") for v in via):
            for x, y in ((a, b), (b, a)):
                if x in dev and y not in dev:
                    other[x].add(y)
    return ngram, {k: sorted(v) for k, v in sorted(other.items())}


# ---------------------------------------------------------------- clusters (§9) ------------------

def pair_links(inp: Inputs, semantic: float, lexical: float) -> dict[tuple[str, str], list[str]]:
    """Every test<->test pair with twin evidence (same kinds and thresholds as the training groups)."""
    ids = {r["id"] for r in inp.test}
    by_id = {r["id"]: r for r in inp.test}
    links: dict[tuple[str, str], list[str]] = defaultdict(list)
    for a, b in twins.ngram_pairs(inp.test):
        shared = sorted(twins.ngrams(by_id[a]["question"]) & twins.ngrams(by_id[b]["question"]))
        links[(a, b)].append("8gram:" + " ".join(shared[0]))
    sem, _ = twins.semantic_scores(inp.train + inp.test)
    for (a, b), s in sem.items():
        if a in ids and b in ids and s >= semantic:
            links[(a, b)].append(f"semantic:{s:.4f}")
    for (a, b), s in twins.lexical_scores(inp.test, inp.test, fit=inp.train, floor=lexical, same=True).items():
        links[(a, b)].append(f"lexical:{s:.4f}")
    kb, _ = twins.keep_both_pairs()
    for (a, b), cid in kb:
        if a in ids and b in ids:
            links[(a, b)].append(f"keep_both:{cid}")
    return {p: sorted(set(v)) for p, v in sorted(links.items())}


def components(ids: Sequence[str], pairs: Sequence[tuple[str, str]]) -> dict[str, str]:
    """Id -> cluster id: TEST_CLUSTER_PREFIX + hex of the smallest member, or the id itself if alone."""
    uf = twins.UnionFind(ids)
    for a, b in pairs:
        uf.union(a, b)
    out = {i: i for i in ids}
    for group in uf.groups():
        cid = TEST_CLUSTER_PREFIX + group[0].removeprefix("q_")
        out.update({m: cid for m in group})
    return out


def dev_clusters(inp: Inputs) -> dict[str, str]:
    member_of = {m: gid for gid, ms in inp.twin_groups["groups"].items() for m in ms}
    return {i: member_of.get(i, i) for i in inp.dev_split["dev_ids"]}


# ---------------------------------------------------------------- records ------------------------

def record(r: Mapping[str, Any], split: str, *, label: Mapping[str, Any], cluster_id: str, near: Sequence[str],
           neighbours: Sequence[str], source: Mapping[str, str], generator_version: str,
           subsets: Sequence[str] | None = None) -> dict[str, Any]:
    if label["raw"] != r["task_type"] or label["split"] != LABEL_SPLIT[split]:
        raise SystemExit(f"tier-k-items: label row of {r['id']} does not match the frozen record")
    writers = writer_info(r)
    rec: dict[str, Any] = {
        "id": r["id"], "tier": "K", "split": split, "version": VERSION, "canary": CANARY,
        "license": LICENSE[r["origin"]],
        "source": {"kind": KIND[split], "ref": source["ref"], "sha256": source["sha256"]},
        "cluster_id": cluster_id,
        "provenance": {"drafted_by": writers["drafted_by"], "generator": GENERATOR,
                       "generator_version": generator_version, "writer_family": writers["writer_family"],
                       "writers": writers["writers"]},
        "review": {"status": "not-required", "notes": REVIEW_NOTES[split]},
        "question": r["question"], "task_type": label["clean"], "task_type_raw": r["task_type"],
        "task_type_labeller": label["labeller"], "difficulty": r["difficulty"], "origin": r["origin"],
        "primary_topic": r["primary_topic"], "item_kind": KIND[split],
        "near_twin_in_train": bool(near), "near_twin_train_ids": list(near),
        "train_neighbour_ids": list(neighbours),
    }
    if subsets is not None:
        rec["subsets"] = list(subsets)
    return rec


@dataclass
class Build:
    test: list[dict[str, Any]]
    dev: list[dict[str, Any]]
    links: dict[tuple[str, str], list[str]]
    ngram_only: dict[str, str]                # test id -> cluster id under 8-gram evidence alone
    dev_ngram: dict[str, list[str]]
    dev_other: dict[str, list[str]]
    inputs: Inputs


def build() -> Build:
    inp = load()
    check_inputs(inp)
    d, g = inp.dev_split, inp.twin_groups
    generator_version = "questions_canonical.jsonl sha256:" + inp.manifest["source_canonical_sha256"][:16]
    src = {split: {"ref": _rel(p), "sha256": inp.hashes[_rel(p)]} for split, p in (("test", TEST_FILE), ("dev", TRAIN_FILE))}

    test_ids = [r["id"] for r in inp.test]
    links = pair_links(inp, g["semantic_threshold"], g["lexical_threshold"])
    clusters = components(test_ids, list(links))
    ngram_only = components(test_ids, [p for p, v in links.items() if any(x.startswith("8gram") for x in v)])
    near = near_twins_of_test(inp)
    neighbours = invert(d["test_neighbours"]["by_training_id"])
    test = [record(r, "test", label=inp.labels[r["id"]], cluster_id=clusters[r["id"]], near=near.get(r["id"], []),
                   neighbours=neighbours.get(r["id"], []), source=src["test"], generator_version=generator_version)
            for r in inp.test]

    by_id = {r["id"]: r for r in inp.train}
    dev_cl = dev_clusters(inp)
    dev_ngram, dev_other = near_twins_of_dev(inp)
    anchor, ci = set(d["anchor_ids"]), set(d["ci_ids"])
    dev = [record(by_id[i], "dev", label=inp.labels[i], cluster_id=dev_cl[i], near=dev_ngram.get(i, []),
                  neighbours=dev_other.get(i, []), source=src["dev"], generator_version=generator_version,
                  subsets=[s for s, ids in (("anchor", anchor), ("ci", ci)) if i in ids])
           for i in d["dev_ids"]]

    errors = [f"{rec['id']}: {e}" for rec in test + dev for e in validate(rec)]
    if errors:
        raise SystemExit("tier-k-items: schema errors:\n  " + "\n  ".join(errors[:20]))
    return Build(test=test, dev=dev, links=links, ngram_only=ngram_only, dev_ngram=dev_ngram,
                 dev_other=dev_other, inputs=inp)


# ---------------------------------------------------------------- report and sheet ---------------

def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def cluster_sizes(recs: Sequence[Mapping[str, Any]]) -> Counter:
    return Counter(Counter(r["cluster_id"] for r in recs).values())


def _split_table(recs: Sequence[Mapping[str, Any]], key: str) -> str:
    """Near-twin vs clean counts per value of `key` (eval plan §6 reporting split)."""
    c: dict[str, Counter] = defaultdict(Counter)
    for r in recs:
        x = c[r["provenance"][key] if key == "writer_family" else r[key]]
        x["near" if r["near_twin_in_train"] else "clean"] += 1
        if not r["near_twin_in_train"] and r["train_neighbour_ids"]:
            x["flag"] += 1
    rows = [[v, x["near"], x["clean"], x["flag"], x["near"] + x["clean"]]
            for v, x in sorted(c.items(), key=lambda kv: (-(kv[1]["near"] + kv[1]["clean"]), kv[0]))]
    tot = [sum(r[i] for r in rows) for i in (1, 2, 3, 4)]
    return _table([key, "near-twin", "clean", "clean but flagged neighbour", "total"], rows + [["**all**", *tot]])


def _count_table(recs: Mapping[str, Sequence[Mapping[str, Any]]], fn, name: str) -> str:
    counts = {s: Counter(fn(r) for r in rs) for s, rs in recs.items()}
    values = sorted({v for c in counts.values() for v in c}, key=lambda v: (-sum(c.get(v, 0) for c in counts.values()), v))
    return _table([name, *recs], [[v, *(counts[s].get(v, 0) for s in recs)] for v in values])


def report(b: Build, file_hashes: Mapping[str, str]) -> str:
    inp, d, g = b.inputs, b.inputs.dev_split, b.inputs.twin_groups
    t, v = b.test, b.dev
    near = [r for r in t if r["near_twin_in_train"]]
    flagged = [r for r in t if not r["near_twin_in_train"] and r["train_neighbour_ids"]]
    multi = sorted({r["cluster_id"] for r in t if r["cluster_id"].startswith(TEST_CLUSTER_PREFIX)})
    kinds = Counter(x.split(":")[0] for via in b.links.values() for x in via)
    only_other = [p for p, via in b.links.items() if not any(x.startswith("8gram") for x in via)]
    ngram_clusters = len({c for c in b.ngram_only.values() if c.startswith(TEST_CLUSTER_PREFIX)})
    test_sizes, dev_sizes = cluster_sizes(t), cluster_sizes(v)
    by_split = {"test": t, "dev": v}
    files = _table(["file", "records", "sha256"],
                   [[f"`{OUT_TEST.as_posix()}`", len(t), f"`{file_hashes[OUT_TEST.as_posix()]}`"],
                    [f"`{OUT_DEV.as_posix()}`", len(v), f"`{file_hashes[OUT_DEV.as_posix()]}`"]])
    inputs = _table(["input", "sha256"], [[f"`{p}`", f"`{h}`"] for p, h in sorted(inp.hashes.items())])
    pair_rows = [[a, b2, "; ".join(via), next(r["cluster_id"] for r in t if r["id"] == a)]
                 for (a, b2), via in b.links.items()]
    sub = Counter(s for r in v for s in r["subsets"])
    return f"""# Tier-K item records (eval plan §4.1, §4.2, §5.1, §6, §9)

Generated by `./gb tier-k-items` (`src/grcbench/tier_k/items.py`, GRC-Bench item version {VERSION}); do not
edit by hand. Status: draft for team review. The questions are the dataset's own; Claude (Anthropic) wrote
only this tooling and the labels it reuses (`splits/task_type_labels.jsonl`). Nothing was sent to any model.

## Files

{files}

Every record passes `grcbench.common.schemas.validate` (tier K) and carries the GRC-Bench canary. Test
records follow the manifest order of `eval_manifest.json`; dev records follow `dev_ids` of
`splits/tier_k_dev.json` (dev split v{d['version']}; its status: "{d['status']}"). Dev records name their
subsets: anchor {sub['anchor']}, ci {sub['ci']}.

Inputs (paths under `project/data/` or the repository):

{inputs}

## Fields

- `source`: kind `frozen_question` (test) or `bank_question` (dev); `ref` relative to `project/data/`;
  `sha256` of that file.
- `provenance.drafted_by`: the question-writing model as recorded in the dataset (source lane:
  `provenance[0].source_file`; blind lane: the `generator`/`model` fields). `writer_family` is the Table 5
  family, `writers` every distinct writer of a merged question. `generator` is `{GENERATOR}`;
  `generator_version` is the first 16 hex digits of the SHA-256 of `questions_canonical.jsonl`.
- `review`: `not-required` for the question text (frozen dataset question); rubrics are reviewed separately.
- `task_type` is the clean label, `task_type_raw` the dataset spelling, `task_type_labeller` who mapped it
  (`splits/task_type_labels.jsonl`). {sum(1 for r in t if 'Claude' in r['task_type_labeller'])} test and
  {sum(1 for r in v if 'Claude' in r['task_type_labeller'])} dev labels come from Claude's reading of an ambiguous
  spelling; the test ones await the team's blind double labelling (`reports/labels.md`).
- `near_twin_in_train`, `near_twin_train_ids`, `train_neighbour_ids`: see the reporting split below.
- `license`: the dataset README states no licence for question records, so every record says
  "licence: as in GRC_dataset_20261007 (Qoherent); to be confirmed before release", plus a note on the
  source lane (questions generated from licensed references, without reproducing them) or the blind lane.

## Clusters (eval plan §9: clustered intervals)

**Dev.** Rule of `splits/tier_k_twin_groups.json`: a training question in a twin group takes the group id,
every other question is a singleton with its own id. The dev split excludes every twin-group member, so all
{len(v)} dev questions are singletons (cluster sizes {dict(sorted(dev_sizes.items()))}). Checked again here:
{len(b.dev_ngram)} dev questions share an 8-gram with a training question outside dev, and
{len(b.dev_other)} have a semantic, lexical or keep_both edge to one.

**Test.** The training twin-group evidence and thresholds, applied to test<->test pairs: shared word
8-gram, semantic score >= {g['semantic_threshold']}, lexical TF-IDF >= {g['lexical_threshold']} (fitted on the
training questions, only for pairs the semantic file did not score) and KEEP_BOTH decisions. A connected
component of two or more test questions gets `{TEST_CLUSTER_PREFIX}` + the hex part of its smallest id;
the others keep their own id. Linked pairs: {len(b.links)} (evidence: {', '.join(f'{k} {n}' for k, n in sorted(kinds.items()))}).
Result: {len(set(r['cluster_id'] for r in t))} clusters for {len(t)} questions; {len(multi)} have more than one
member (sizes {dict(sorted(test_sizes.items()))}). With 8-gram evidence alone there would be {ngram_clusters}
multi-member clusters; {len(only_other)} pairs are linked only by the other evidence.

{_table(['test id', 'test id', 'evidence', 'cluster_id'], pair_rows)}

The team confirms or splits these clusters in `{OUT_SHEET.as_posix()}`. A shared 8-gram can be a generic
phrase ("gnu radio what is the role of the"); keeping such a pair in one cluster only widens the interval
slightly, so the rule is applied as written.

## Near-twin reporting split (eval plan §6)

A test question is a **near-twin** when it shares a word 8-gram with one of the 1,668 frozen training
questions (tokenizer: lowercase, every non-word character to a space, split on whitespace); the others are
**clean**. Scores are reported on both groups separately. {len(near)} of {len(t)} test questions are
near-twins (eval plan §6 counts 26), linked to {len({x for r in near for x in r['near_twin_train_ids']})}
training questions; the 8-gram recount matches `splits/tier_k_dev.json`. A further {len(flagged)} clean test
questions have a flagged training neighbour (semantic, lexical or keep_both evidence, no shared 8-gram);
they stay clean under §6 and can be dropped as a sensitivity check. No dev question is a near-twin of the
training questions left after holding dev out.

By clean task type:

{_split_table(t, 'task_type')}

By origin:

{_split_table(t, 'origin')}

By question writer (Table 5):

{_split_table(t, 'writer_family')}

## Who wrote the questions (eval plan §5.1, Table 5)

{_count_table(by_split, lambda r: r['provenance']['writer_family'], 'writer family')}

Mistral wrote {sum(1 for r in t if r['provenance']['writer_family'] == 'Mistral')} of the {len(t)} test questions
(eval plan: 145). Per model, as recorded:

{_count_table(by_split, lambda r: r['provenance']['drafted_by'].split(' (')[0], 'model')}

Model names are the dataset's labels made readable (`mistral_large2` -> Mistral Large 2, `gemma4/gemma` ->
Gemma 4); the question records do not state exact model versions.

## Composition

{_count_table(by_split, lambda r: r['task_type'], 'task_type (clean)')}

{_count_table(by_split, lambda r: r['difficulty'], 'difficulty')}

{_count_table(by_split, lambda r: r['origin'], 'origin')}

## Open points for the team

- Confirm the test clusters (review sheet), in particular the generic-phrase 8-gram pair and the pairs
  linked only by lexical or keep_both evidence.
- Confirm the licence of the question records with Qoherent before release (source-lane questions were
  generated from references that include CC BY-SA and CC BY-NC-SA 4.0 material).
- The clean task types of the ambiguous test items change only after the team's double labelling; re-run
  `./gb labels` and then `./gb tier-k-items`.
"""


def sheet(b: Build) -> str:
    """One row per test question in a multi-member cluster, with its links and the team's columns."""
    by_id = {r["id"]: r for r in b.test}
    linked: dict[str, list[str]] = defaultdict(list)
    for (a, c), via in b.links.items():
        linked[a].append(f"{c}: {'; '.join(via)}")
        linked[c].append(f"{a}: {'; '.join(via)}")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["cluster_id", "id", "linked_to (evidence)", "origin", "task_type", "question",
                "team_decision (same family / split out)", "reviewer", "notes"])
    rows = sorted((r for r in b.test if r["cluster_id"].startswith(TEST_CLUSTER_PREFIX)),
                  key=lambda r: (r["cluster_id"], r["id"]))
    for r in rows:
        w.writerow([r["cluster_id"], r["id"], " | ".join(sorted(linked[r["id"]])), r["origin"], r["task_type"],
                    by_id[r["id"]]["question"], "", "", ""])
    return buf.getvalue()


# ---------------------------------------------------------------- main ---------------------------

def write(b: Build, root: Path) -> dict[str, str]:
    """Write every output under `root`; return {relative path: sha256}."""
    write_jsonl(root / OUT_TEST, b.test)
    write_jsonl(root / OUT_DEV, b.dev)
    hashes = {p.as_posix(): sha256_file(root / p) for p in (OUT_TEST, OUT_DEV)}
    write_text(root / OUT_SHEET, sheet(b))
    write_text(root / OUT_REPORT, report(b, hashes))
    for p in (OUT_SHEET, OUT_REPORT):
        hashes[p.as_posix()] = sha256_file(root / p)
    return hashes


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb tier-k-items", description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO, help="root folder for the outputs (default: the repo)")
    args = ap.parse_args(argv)
    b = build()
    hashes = write(b, args.out)
    near = sum(r["near_twin_in_train"] for r in b.test)
    multi = len({r["cluster_id"] for r in b.test if r["cluster_id"].startswith(TEST_CLUSTER_PREFIX)})
    print(f"test {len(b.test)} (near-twin {near}, clean {len(b.test) - near}, multi-member clusters {multi}) | "
          f"dev {len(b.dev)}")
    for rel, h in hashes.items():
        print(f"  {rel:55s} sha256 {h[:16]}")
    return 0
