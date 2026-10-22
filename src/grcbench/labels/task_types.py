"""Task-type label normalization: 23 raw spellings -> the 6 canonical types.

Implements eval plan §4.2 ("Labels": report by cleaned labels only; no group under 20 items; merges
decided before the final runs) and proposal §4.2 ("Label normalization", 10% human spot-check and
inter-annotator agreement). The canonical types and their definitions come from
`00_dataset_spec/task_types.json`; `debugging` -> `diagnosis` follows the Data Summary (§3).

  * 16 unambiguous raw spellings are mapped by RULE_MAP (the 6 canonical names map to themselves).
  * The 134 rows with one of the 7 ambiguous spellings (`conceptual`, the `compare` variants and the
    missing-information variants) were each read and classified by Claude (Anthropic) against
    task_types.json. The decisions are stored as data in `ambiguous_decisions.json`, so the mapping
    is reproducible. This is labelling, not reference drafting (eval plan Table 5).
  * Raw labels are kept beside the clean ones and are never overwritten.

Outputs (deterministic; a re-run gives byte-identical files):
  splits/task_type_labels.jsonl                         one row per question (1,853)
  splits/task_type_review_key.json                      review-sheet rows -> Claude / rule labels
  reports/labels.md                                     counts, small groups, merge proposals
  handoff/review_sheets/task_type_ambiguous_test.csv    blind double labelling of ambiguous test items
  handoff/review_sheets/task_type_spotcheck_train.csv   seeded 10% spot-check of training rows
  handoff/review_sheets/task_type_instructions.md       how to fill in the two sheets

Commands:  ./gb labels [build] [--out DIR]   |   ./gb labels agreement [...]  (see agreement.py)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from grcbench.common.io import read_json, read_jsonl, sha256_file, write_json, write_jsonl, write_text
from grcbench.common.paths import DATASET, FROZEN, HANDOFF, REPO, REPORTS, SPLITS

SPEC_FILE = DATASET / "00_dataset_spec" / "task_types.json"
EVAL_FILE = FROZEN / "eval_questions.jsonl"
TRAIN_FILE = FROZEN / "train_questions.jsonl"
MANIFEST_FILE = FROZEN / "eval_manifest.json"
CANONICAL_FILE = DATASET / "03_questions_merged" / "questions_canonical.jsonl"
DECISIONS_FILE = Path(__file__).with_name("ambiguous_decisions.json")

# Output locations, relative to the output root (the repository by default).
LABELS_REL = SPLITS.relative_to(REPO) / "task_type_labels.jsonl"
KEY_REL = SPLITS.relative_to(REPO) / "task_type_review_key.json"
REPORT_REL = REPORTS.relative_to(REPO) / "labels.md"
SHEETS_REL = HANDOFF.relative_to(REPO) / "review_sheets"
TEST_SHEET_REL = SHEETS_REL / "task_type_ambiguous_test.csv"
TRAIN_SHEET_REL = SHEETS_REL / "task_type_spotcheck_train.csv"
INSTRUCTIONS_REL = SHEETS_REL / "task_type_instructions.md"

CANONICAL = ("calculation", "design", "diagnosis", "factual", "gnuradio_usage", "reasoning")

# raw spelling -> (canonical type, why)
RULES: dict[str, tuple[str, str]] = {
    "calculation": ("calculation", "already the canonical spelling"),
    "calculations": ("calculation", "plural spelling of calculation"),
    "design": ("design", "already the canonical spelling"),
    "system design": ("design", "system-level design is design (task_types.json: select parameters, "
                                "architecture or methods)"),
    "system_design": ("design", "system-level design is design (task_types.json: select parameters, "
                                "architecture or methods)"),
    "diagnosis": ("diagnosis", "already the canonical spelling"),
    "diagnosis from symptoms": ("diagnosis", "spelling of diagnosis (task_types.json: reason from symptoms)"),
    "diagnosis_from_symptoms": ("diagnosis", "spelling of diagnosis (task_types.json: reason from symptoms)"),
    "debugging": ("diagnosis", "Data Summary §3: 'debugging' is cleaned up into diagnosis; curriculum "
                               "capability 3 is 'debugging and diagnosis'"),
    "factual": ("factual", "already the canonical spelling"),
    "factual knowledge": ("factual", "spelling of factual (curriculum capability 4: factual knowledge)"),
    "factual_knowledge": ("factual", "spelling of factual (curriculum capability 4: factual knowledge)"),
    "gnuradio_usage": ("gnuradio_usage", "already the canonical spelling"),
    "gnu_radio_usage": ("gnuradio_usage", "spelling of gnuradio_usage (Data Summary §2: the official "
                                          "spelling is gnuradio_usage)"),
    "GNU Radio usage": ("gnuradio_usage", "spelling of gnuradio_usage (Data Summary §2: the official "
                                          "spelling is gnuradio_usage)"),
    "reasoning": ("reasoning", "already the canonical spelling"),
}
RULE_MAP: dict[str, str] = {raw: clean for raw, (clean, _) in RULES.items()}

# Spellings that do not name one canonical type; every row with one of these is classified by hand.
AMBIGUOUS = ("ask_for_info", "asking for missing information", "compare", "compare/choose",
             "compare_choose", "conceptual", "information_gathering")

LABELLER_CLAUDE = "Claude (Anthropic)"
LABELLER_RULE = "rule map (grcbench.labels.task_types)"
MIN_GROUP = 20                      # eval plan §4.2: no group with fewer than 20 items is reported
SPOTCHECK_FRACTION = 0.10           # proposal §4.2: 10% human spot-check
SPOTCHECK_SEED = "grc-bench-labels-spotcheck-v1-2026-10-14"
ORDER_SEED = "grc-bench-labels-blind-order-v1-2026-10-14"
TEST_SHEET_COLUMNS = ["id", "question", "raw_label", "annotator_1", "annotator_2", "notes"]
TRAIN_SHEET_COLUMNS = ["id", "question", "raw_label", "method", "claude_or_rule_label", "reason",
                       "reviewer_label", "notes"]

# Conceptual nearest neighbour of each type (used beside the lexical check for merge proposals).
CONCEPT_NEIGHBOUR: dict[str, tuple[str, str]] = {
    "calculation": ("design", "design questions also choose numeric parameters; both need worked quantities"),
    "design": ("reasoning", "design weighs trade-offs, which is reasoning applied to a requirement"),
    "diagnosis": ("reasoning", "diagnosis infers causes from behaviour, a special case of reasoning"),
    "factual": ("reasoning", "both explain concepts; reasoning applies them to a situation"),
    "gnuradio_usage": ("design", "both are about building and configuring a system: choosing blocks, "
                                 "parameters and methods. Most of the test items read 'how would you "
                                 "implement / configure X'"),
    "reasoning": ("factual", "both explain concepts; factual stops at recall"),
}


# ----------------------------------------------------------------------------------------------
# Loading and checking inputs
# ----------------------------------------------------------------------------------------------

def load_spec(path: Path = SPEC_FILE) -> dict[str, Any]:
    spec = read_json(path)
    names = tuple(sorted(spec["task_types"]))
    if names != CANONICAL:
        raise ValueError(f"task_types.json lists {names}, expected {CANONICAL}")
    return spec


def load_questions() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the frozen split and check it against its manifest and the canonical bank."""
    eval_rows, train_rows = read_jsonl(EVAL_FILE), read_jsonl(TRAIN_FILE)
    manifest = read_json(MANIFEST_FILE)
    eval_ids = {r["id"] for r in eval_rows}
    train_ids = {r["id"] for r in train_rows}
    if eval_ids != set(manifest["eval_ids"]) or len(eval_rows) != manifest["eval_count"]:
        raise ValueError("eval_questions.jsonl does not match eval_manifest.json")
    if len(train_rows) != manifest["train_count"] or eval_ids & train_ids:
        raise ValueError("train_questions.jsonl does not match the manifest or overlaps eval")
    bank_ids = {r["id"] for r in read_jsonl(CANONICAL_FILE)}
    if bank_ids != eval_ids | train_ids:
        raise ValueError("eval + train ids differ from questions_canonical.jsonl")
    return eval_rows, train_rows


def load_decisions(path: Path = DECISIONS_FILE) -> dict[str, Any]:
    doc = read_json(path)
    for qid, d in doc["decisions"].items():
        if d["clean"] not in CANONICAL:
            raise ValueError(f"{qid}: '{d['clean']}' is not a canonical task type")
        if d["raw"] not in AMBIGUOUS:
            raise ValueError(f"{qid}: decision recorded for unambiguous spelling '{d['raw']}'")
        if not d["reason"].strip() or "\n" in d["reason"]:
            raise ValueError(f"{qid}: the reason must be one non-empty line")
    return doc


# ----------------------------------------------------------------------------------------------
# Labelling
# ----------------------------------------------------------------------------------------------

def label_rows(eval_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]],
               decisions: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    """One label row per question, eval first, then train; each part sorted by id."""
    out: list[dict[str, str]] = []
    used: set[str] = set()
    for split, rows in (("eval", eval_rows), ("train", train_rows)):
        for r in sorted(rows, key=lambda r: r["id"]):
            raw = r["task_type"]
            if raw in RULES:
                clean, why = RULES[raw]
                row = {"method": "rule", "clean": clean, "reason": f"rule map: {why}",
                       "labeller": LABELLER_RULE}
            elif raw in AMBIGUOUS:
                d = decisions.get(r["id"])
                if d is None:
                    raise ValueError(f"{r['id']}: ambiguous spelling '{raw}' has no recorded decision")
                if d["raw"] != raw:
                    raise ValueError(f"{r['id']}: decision was made for raw '{d['raw']}', data says '{raw}'")
                used.add(r["id"])
                row = {"method": "classified", "clean": d["clean"], "reason": d["reason"],
                       "labeller": LABELLER_CLAUDE}
            else:
                raise ValueError(f"{r['id']}: unknown raw task_type spelling '{raw}'")
            out.append({"id": r["id"], "split": split, "raw": raw, **row})
    stale = sorted(set(decisions) - used)
    if stale:
        raise ValueError(f"decisions for ids that are not ambiguous questions: {stale[:5]}")
    return out


def rank_key(seed: str, item_id: str) -> str:
    """Deterministic pseudo-random rank, as in the frozen split (sha256 of seed NUL id)."""
    return hashlib.sha256(f"{seed}\0{item_id}".encode("utf-8")).hexdigest()


def largest_remainder(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation of n slots; leftover slots to the largest remainders, ties by key."""
    total = sum(sizes.values())
    quotas = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: math.floor(q) for k, q in quotas.items()}
    order = sorted(sizes, key=lambda k: (-(quotas[k] - alloc[k]), k))
    for k in order[: n - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def spotcheck_sample(train_labels: list[dict[str, str]], fraction: float = SPOTCHECK_FRACTION,
                     seed: str = SPOTCHECK_SEED) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Seeded sample of training rows, stratified by raw spelling, returned in rank order.

    `fraction` of the rows (rounded) is allocated across spellings by largest remainder; a spelling
    that would get no row still gets one, so every raw spelling's mapping is checked at least once.
    """
    by_raw: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train_labels:
        by_raw[row["raw"]].append(row)
    n = int(math.floor(fraction * len(train_labels) + 0.5))
    alloc = largest_remainder({k: len(v) for k, v in by_raw.items()}, n)
    alloc = {k: max(v, 1) for k, v in alloc.items()}
    picked = []
    for raw in sorted(by_raw):
        ranked = sorted(by_raw[raw], key=lambda r: rank_key(seed, r["id"]))
        picked.extend(ranked[: alloc[raw]])
    picked.sort(key=lambda r: rank_key(seed, r["id"]))
    return picked, dict(sorted(alloc.items()))


# ----------------------------------------------------------------------------------------------
# Counting and the lexical nearest-neighbour check
# ----------------------------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Same tokenizer as the 8-gram overlap check: lowercase, non-word characters to spaces."""
    return re.sub(r"\W", " ", text.lower()).split()


def lexical_similarity(docs: Iterable[tuple[str, str]]) -> dict[str, dict[str, float]]:
    """Cosine similarity between TF-IDF centroids of each label's questions (label, text) pairs."""
    docs = [(label, Counter(tokenize(text))) for label, text in docs]
    df: Counter[str] = Counter()
    for _, tf in docs:
        df.update(tf.keys())
    n = len(docs)
    idf = {w: math.log(n / c) + 1.0 for w, c in df.items()}
    parts: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for label, tf in docs:
        vec = {w: c * idf[w] for w, c in tf.items()}
        norm = math.sqrt(math.fsum(x * x for x in vec.values()))
        for w, x in vec.items():
            parts[label][w].append(x / norm)
    cent = {lab: {w: math.fsum(xs) for w, xs in ws.items()} for lab, ws in parts.items()}

    def cos(a: dict[str, float], b: dict[str, float]) -> float:
        dot = math.fsum(a[w] * b[w] for w in sorted(a.keys() & b.keys()))
        na = math.sqrt(math.fsum(x * x for x in a.values()))
        nb = math.sqrt(math.fsum(x * x for x in b.values()))
        return round(dot / (na * nb), 3)

    labels = sorted(cent)
    return {a: {b: cos(cent[a], cent[b]) for b in labels if b != a} for a in labels}


def nearest(sims: dict[str, float]) -> str:
    return sorted(sims, key=lambda k: (-sims[k], k))[0]


# ----------------------------------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------------------------------

def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    """UTF-8 with a byte-order mark (so Excel shows symbols such as π correctly), LF line ends."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def render_instructions(spec: dict[str, Any], n_test: int, n_train: int) -> str:
    types = "\n".join(f"- `{name}`: {spec['task_types'][name]['description']}" for name in CANONICAL)
    rules = "\n".join(f"- {r}" for r in spec["classification_rules"])
    return f"""# Task-type review sheets: instructions

Two sheets, both UTF-8 CSV (open in Excel or LibreOffice). Background and counts: `reports/labels.md`,
but **annotators of the test sheet must not open that report, `splits/task_type_labels.jsonl`,
`splits/task_type_review_key.json` or `src/grcbench/labels/ambiguous_decisions.json` until their
labels are returned**: those files show Claude's labels.

The test sheet holds frozen test questions. Keep it inside the team, and never paste its questions
into ChatGPT, Claude or any other model or online tool (frozen-split rule: eval questions are never
sent to any model).

## Allowed labels (from `00_dataset_spec/task_types.json`)

{types}

Classification rules from the same file:

{rules}

Write exactly one of the six labels above in each label cell (lower case, as spelled).

## Sheet 1: `task_type_ambiguous_test.csv` ({n_test} test questions, blind double labelling)

1. Two team members label independently. Each works on **their own copy** of the sheet and fills in only
   their own column (`annotator_1` or `annotator_2`). Do not discuss items until both are done.
2. Read the question and pick the one type that best represents the main capability needed to answer it.
   `raw_label` is the question generator's own spelling, which is ambiguous by construction.
3. Use `notes` for doubts. Return both copies; agreement (Cohen's κ, each annotator against the other and
   against Claude) is computed with
   `./gb labels agreement --test <copy_1.csv> <copy_2.csv>`.
4. Disagreements are settled by discussion and the agreed label goes into the G2 pre-registration.

## Sheet 2: `task_type_spotcheck_train.csv` ({n_train} training questions, spot-check)

A seeded 10% sample of training rows, stratified by raw spelling, with at least one row for every
spelling. `method` says whether the label came
from the rule map (`rule`) or from Claude's reading of the question (`classified`). This check is not
blind: read the question and the proposed label, then write the label you think is right in
`reviewer_label` (the same label if you agree). Agreement is computed with
`./gb labels agreement --train <sheet.csv>`.
"""


def render_report(ctx: dict[str, Any]) -> str:
    spec, labels = ctx["spec"], ctx["labels"]
    by_split = {s: [r for r in labels if r["split"] == s] for s in ("eval", "train")}
    n_eval, n_train, n_all = len(by_split["eval"]), len(by_split["train"]), len(labels)
    classified = [r for r in labels if r["method"] == "classified"]
    amb_eval = [r for r in classified if r["split"] == "eval"]
    out: list[str] = []
    add = out.append

    add("# Task-type labels: 23 raw spellings to 6 canonical types")
    add("")
    add("> **Annotators of `handoff/review_sheets/task_type_ambiguous_test.csv`: return your labels before "
        "reading this report.** It shows how Claude labelled the ambiguous test items.")
    add("")
    add("Implements eval plan §4.2 (Labels) and proposal §4.2 (Label normalization). Generated by "
        "`./gb labels` (`src/grcbench/labels/task_types.py`); do not edit by hand. Raw labels are kept "
        "beside the clean ones in `splits/task_type_labels.jsonl` and are never overwritten.")
    add("")
    add("## Summary")
    add("")
    n_rule = n_all - len(classified)
    add(f"- {n_all:,} questions ({n_eval} test, {n_train:,} train) carry 23 raw `task_type` spellings. "
        f"They are mapped to the 6 canonical types of `00_dataset_spec/task_types.json`.")
    add(f"- **Rule map:** {len(RULES)} unambiguous spellings (the 6 canonical names plus 10 variants such as "
        f"`debugging`, `system_design`, `factual knowledge`) cover {n_rule:,} rows.")
    add(f"- **Classified one by one:** {len(classified)} rows ({len(amb_eval)} test, "
        f"{len(classified) - len(amb_eval)} train) use one of {len(AMBIGUOUS)} ambiguous spellings "
        f"({', '.join(f'`{a}`' for a in AMBIGUOUS)}). Claude (Anthropic) read each question and chose a "
        "type against the task_types.json definitions, with a one-line reason. The decisions are data "
        "in `src/grcbench/labels/ambiguous_decisions.json`. This is labelling, not reference drafting "
        "(eval plan Table 5: Claude never drafts references).")
    small = ctx["small_groups"]
    if small:
        names = ", ".join(f"`{g}` ({ctx['test_counts'][g]})" for g in small)
        add(f"- **Small test groups:** {names} below {MIN_GROUP} items. Merge proposals are below, "
            "**for team decision; to be fixed in the G2 pre-registration before any test run**.")
    add(f"- **Team review:** all {len(amb_eval)} ambiguous test items go to two team members for blind "
        f"double labelling, and a seeded {int(SPOTCHECK_FRACTION * 100)}% sample of training rows "
        f"({len(ctx['spot_rows'])} rows, every raw spelling included) is spot-checked. The sheets are in "
        "`handoff/review_sheets/`, and the instructions are in `task_type_instructions.md`.")
    add("")

    add("## Inputs")
    add("")
    add(_table(["file", "sha256"], [[f"`{name}`", f"`{digest}`"] for name, digest in ctx["inputs"]]))
    add("")

    add("## Decision rules for the ambiguous rows")
    add("")
    doc = ctx["decisions_doc"]
    for code, text in sorted(doc["rules"].items()):
        add(f"- `{code}`: {text}")
    add("")
    add(doc["rule_application"])
    add("")
    rule_counts = Counter((d["raw"], d["rule"]) for d in doc["decisions"].values())
    add(_table(["raw spelling", "rule used", "rows"],
               [[f"`{raw}`", f"`{rule}`", n] for (raw, rule), n in sorted(rule_counts.items())]))
    add("")

    add("## Raw to clean counts")
    add("")
    pair_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for r in labels:
        pair_counts[(r["raw"], r["method"], r["clean"])][r["split"]] += 1
    rows = []
    for (raw, method, clean), c in sorted(pair_counts.items(), key=lambda kv: (kv[0][2], kv[0][0])):
        rows.append([f"`{clean}`", f"`{raw}`", method, c["eval"], c["train"], c["eval"] + c["train"]])
    rows.append(["**total**", "", "", n_eval, n_train, n_all])
    add(_table(["clean", "raw spelling", "method", "test (eval)", "train", "all"], rows))
    add("")

    add("## Clean type counts")
    add("")
    cnt = {s: Counter(r["clean"] for r in by_split[s]) for s in by_split}
    rows = []
    for t in CANONICAL:
        a = cnt["eval"][t] + cnt["train"][t]
        rows.append([f"`{t}`", f"{spec['task_types'][t]['target_share']:.0%}", cnt["eval"][t],
                     cnt["train"][t], a, f"{a / n_all:.1%}"])
    rows.append(["**total**", "", n_eval, n_train, n_all, "100%"])
    add(_table(["type", "target share (task_types.json)", "test (eval)", "train", "all", "share of all"],
               rows))
    add("")

    add(f"## The {n_eval} test questions by clean type")
    add("")
    add(f"Eval plan §4.2: results are reported by cleaned label only, and no group with fewer than "
        f"{MIN_GROUP} items is reported. The {len(amb_eval)} classified test items can still move when the "
        "team's double labels come back, so the last column gives each group's possible range "
        "(all of them moving in or out).")
    add("")
    rows = []
    for t in CANONICAL:
        n_rule_t = sum(1 for r in by_split["eval"] if r["clean"] == t and r["method"] == "rule")
        n_cls_t = cnt["eval"][t] - n_rule_t
        total = cnt["eval"][t]
        status = ("below 20: merge needed" if total < MIN_GROUP
                  else "at the threshold" if total == MIN_GROUP else "ok")
        rows.append([f"`{t}`", n_rule_t, n_cls_t, total, status,
                     f"{n_rule_t}–{n_rule_t + len(amb_eval)}"])
    rows.append(["**total**", n_eval - len(amb_eval), len(amb_eval), n_eval, "", ""])
    add(_table(["type", "rule-mapped", "classified", "total", f"status (< {MIN_GROUP})",
                "range after team labels"], rows))
    add("")
    for t in CANONICAL:
        if cnt["eval"][t] == MIN_GROUP:
            add(f"`{t}` has exactly {MIN_GROUP} test items, so it can be reported. None of its items is "
                "among the classified ones, so team relabelling can only add to it.")
            add("")

    add(f"## Groups with fewer than {MIN_GROUP} test items: merge proposals")
    add("")
    add("**For team decision; to be fixed in the G2 pre-registration before any test run** "
        "(eval plan §4.2: smaller groups are merged with their nearest neighbour, and the merge is decided "
        "before the final runs).")
    add("")
    sims = ctx["similarity"]
    if not small:
        add(f"No group is below {MIN_GROUP}.")
        add("")
    for g in small:
        concept, why = CONCEPT_NEIGHBOUR[g]
        lex = nearest(sims[g])
        add(f"### `{g}`: {ctx['test_counts'][g]} test items")
        add("")
        merged = ctx["test_counts"][g] + ctx["test_counts"][concept]
        add(f"- **Proposal: merge into `{concept}`** and report one group "
            f"\"{concept} + {g}\" ({merged} test items). Reason: {why}.")
        agree = "agrees" if lex == concept else "differs"
        add(f"- **Lexical check ({agree}):** on the training questions only, the TF-IDF centroid of `{g}` "
            f"is closest to `{lex}` (cosine {sims[g][lex]:.3f}). "
            + ", ".join(f"`{k}` {v:.3f}" for k, v in sorted(sims[g].items(), key=lambda kv: -kv[1]))
            + ".")
        if g == "gnuradio_usage":
            add("- **Alternative A (no merge):** Tier K v1.0 adds about 100 block-catalog questions (eval plan "
                "§4.2; WS3b), all about GNU Radio blocks, ports, types and rates, which is the "
                "gnuradio_usage definition. If the group rule is applied to Tier K as a whole (285 items) "
                f"rather than to the {n_eval} frozen questions, `gnuradio_usage` has about "
                f"{ctx['test_counts'][g] + 100} items and needs no merge. GRC-Agent needs this skill most "
                "(Data Summary §6), so keeping it visible is worth considering.")
            add("- **Alternative B:** merge into `factual` if the team sees the frozen usage questions as "
                "procedural recall. That would conflict with reporting factual recall separately (next "
                "section).")
            add("- Note for the decision: several of the frozen `gnuradio_usage` test items are not about "
                "GNU Radio itself (USRP VM setup, Python implementations, Phaser). They were labelled "
                "that way by the generator, and the rule map keeps that label.")
        add("")
    add("Lexical similarity between all clean types (TF-IDF centroids over the training questions; the "
        "same tokenizer as the 8-gram check). It is a supporting check only; the proposal rests on the "
        "definitions.")
    add("")
    types = sorted(sims)
    add(_table(["type"] + [f"`{t}`" for t in types],
               [[f"`{a}`"] + [("—" if a == b else f"{sims[a][b]:.3f}") for b in types] for a in types]))
    add("")

    add("## Factual recall and reasoning are reported separately")
    add("")
    fam = ctx["families"]
    add("Proposal §5.1 and eval plan §4.2 report factual recall and reasoning separately, because looped "
        "models such as Nanbeige4.2 may trade recall for reasoning [14]. Proposed mapping, "
        "**for team decision; to be fixed in the G2 pre-registration before any test run**:")
    add("")
    add(f"- **Recall:** `factual` ({fam['recall']} test items), plus the WS3b block-catalog questions, whose "
        "answers are block properties read from the YAML (pure recall).")
    add(f"- **Reasoning:** `reasoning`, `calculation`, `design` and `diagnosis` ({fam['reasoning']} test "
        "items).")
    add(f"- **`gnuradio_usage` ({fam['usage']} frozen test items):** put them in the reasoning family, in "
        "line with the merge into `design` proposed above (they ask how to implement or configure "
        "something). The alternative is to leave them out of the two-way split.")
    add("- The split goes by item, not only by task type: a block-catalog question counts as recall even "
        "if its task_type is `gnuradio_usage`.")
    add(f"- The recall group is much smaller than the reasoning group ({fam['recall']} vs "
        f"{fam['reasoning'] + fam['usage']} frozen items), so its intervals are wider. Print the MDE beside "
        "it (eval plan §9).")
    add("")

    add("## Team review")
    add("")
    add(f"- `handoff/review_sheets/task_type_ambiguous_test.csv`: all {len(amb_eval)} ambiguous test items, "
        "for blind double labelling by two team members (columns `annotator_1`, `annotator_2`). Claude's "
        f"label is not shown. Row order is shuffled with seed `{ORDER_SEED}`.")
    n_prop = int(math.floor(SPOTCHECK_FRACTION * n_train + 0.5))
    add(f"- `handoff/review_sheets/task_type_spotcheck_train.csv`: {len(ctx['spot_rows'])} training rows: "
        f"10% of {n_train:,} ({n_prop}) allocated across raw spellings by largest remainder, plus one row "
        f"for each of the {len(ctx['spot_rows']) - n_prop} spellings that would otherwise get none, so "
        f"every spelling is checked. Seed `{SPOTCHECK_SEED}`. Of these, "
        f"{sum(1 for r in ctx['spot_rows'] if r['method'] == 'classified')} were classified by Claude and "
        f"{sum(1 for r in ctx['spot_rows'] if r['method'] == 'rule')} came from the rule map (the "
        "`method` column says which).")
    add("- `splits/task_type_review_key.json` maps every sheet row to Claude's or the rule's label.")
    add("- When the sheets come back: `./gb labels agreement --test <copy_1.csv> <copy_2.csv> --train "
        "<spotcheck.csv>` reports Cohen's κ (annotator 1 vs 2, each vs Claude) and spot-check agreement in "
        f"`reports/labels_agreement.md`. With only {len(amb_eval)} items, κ has a wide interval. Report "
        "it, but settle disagreements by discussion.")
    add("")

    add("## Notes and deviations")
    add("")
    add(f"- The plan counts \"14 unambiguous spellings\". The data has {len(RULES)} unambiguous spellings "
        f"(including the 6 canonical names) and {len(AMBIGUOUS)} ambiguous ones, 23 in all. The "
        f"ambiguous rows number {len(classified)}, as planned.")
    add("- Proposal §4.2 describes an embedding-plus-LLM classifier. Only 7 spellings are ambiguous, so a "
        "rule map plus a reasoned reading of each of the 134 questions replaces it. Every decision has a "
        "reason, and none uses embeddings. Topic normalization (930 to 14 topics) is the data team's work "
        "and is not done here.")
    add("- Rule-mapped labels are taken as given and not re-judged. Some raw `factual` items are about "
        "GNU Radio (e.g. 'What is throttling in GNU Radio?'), and some raw `gnuradio_usage` items are not "
        "about GNU Radio. The spot-check sheet lets the team flag such cases.")
    add("- Claude read the 11 ambiguous test questions from the files in order to label them. No test "
        "question was sent to any other model.")
    add("")
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------------------------

def _rel(path: Path) -> str:
    try:
        return path.relative_to(REPO.parent).as_posix()
    except ValueError:
        return path.as_posix()


def build(out: Path = REPO) -> dict[str, Any]:
    """Compute every label output and write it under `out` (the repository by default)."""
    spec = load_spec()
    eval_rows, train_rows = load_questions()
    decisions_doc = load_decisions()
    labels = label_rows(eval_rows, train_rows, decisions_doc["decisions"])
    questions = {r["id"]: r["question"] for r in eval_rows + train_rows}

    # Blind double-labelling sheet: every ambiguous test item, shuffled, without Claude's label.
    amb_eval = sorted((r for r in labels if r["split"] == "eval" and r["method"] == "classified"),
                      key=lambda r: rank_key(ORDER_SEED, r["id"]))
    test_sheet = [{"id": r["id"], "question": questions[r["id"]], "raw_label": r["raw"]} for r in amb_eval]

    spot_rows, strata = spotcheck_sample([r for r in labels if r["split"] == "train"])
    train_sheet = [{"id": r["id"], "question": questions[r["id"]], "raw_label": r["raw"],
                    "method": r["method"], "claude_or_rule_label": r["clean"], "reason": r["reason"]}
                   for r in spot_rows]

    key = {
        "about": "Maps each task-type review-sheet row to the label Claude or the rule map gave it. "
                 "Do not show to annotators before their labels are returned.",
        "ambiguous_test": {
            "sheet": TEST_SHEET_REL.as_posix(), "order_seed": ORDER_SEED, "n": len(amb_eval),
            "rows": [{"row": i, "id": r["id"], "raw": r["raw"], "claude_label": r["clean"]}
                     for i, r in enumerate(amb_eval, 1)],
        },
        "spotcheck_train": {
            "sheet": TRAIN_SHEET_REL.as_posix(), "seed": SPOTCHECK_SEED, "fraction": SPOTCHECK_FRACTION,
            "stratified_by": "raw task_type spelling (largest remainder; at least one row per spelling)",
            "strata": strata, "n": len(spot_rows),
            "rows": [{"row": i, "id": r["id"], "raw": r["raw"], "method": r["method"], "label": r["clean"]}
                     for i, r in enumerate(spot_rows, 1)],
        },
    }

    test_counts = Counter(r["clean"] for r in labels if r["split"] == "eval")
    small = [t for t in CANONICAL if test_counts[t] < MIN_GROUP]
    similarity = lexical_similarity((r["clean"], questions[r["id"]]) for r in labels if r["split"] == "train")
    families = {"recall": test_counts["factual"], "usage": test_counts["gnuradio_usage"],
                "reasoning": sum(test_counts[t] for t in ("reasoning", "calculation", "design", "diagnosis"))}
    inputs = [(_rel(p), sha256_file(p)) for p in (SPEC_FILE, EVAL_FILE, TRAIN_FILE, MANIFEST_FILE)]
    inputs.append(("grc-bench/src/grcbench/labels/ambiguous_decisions.json", sha256_file(DECISIONS_FILE)))
    ctx = {"spec": spec, "labels": labels, "decisions_doc": decisions_doc, "spot_rows": spot_rows,
           "test_counts": test_counts, "small_groups": small, "similarity": similarity,
           "families": families, "inputs": inputs}

    paths = {name: out / rel for name, rel in (
        ("labels", LABELS_REL), ("key", KEY_REL), ("report", REPORT_REL), ("test_sheet", TEST_SHEET_REL),
        ("train_sheet", TRAIN_SHEET_REL), ("instructions", INSTRUCTIONS_REL))}
    write_jsonl(paths["labels"], labels)
    write_json(paths["key"], key)
    write_text(paths["report"], render_report(ctx))
    write_csv(paths["test_sheet"], TEST_SHEET_COLUMNS, test_sheet)
    write_csv(paths["train_sheet"], TRAIN_SHEET_COLUMNS, train_sheet)
    write_text(paths["instructions"], render_instructions(spec, len(test_sheet), len(train_sheet)))
    return {"paths": paths, "labels": labels, "test_counts": dict(test_counts), "small_groups": small,
            "n_classified": sum(1 for r in labels if r["method"] == "classified"),
            "n_test_sheet": len(test_sheet), "n_train_sheet": len(train_sheet)}


def main(argv: list[str]) -> int:
    if argv and argv[0] == "agreement":
        from grcbench.labels.agreement import main as agreement_main
        return agreement_main(argv[1:])
    parser = argparse.ArgumentParser(prog="./gb labels",
                                     description="Map raw task-type spellings to the 6 canonical types.")
    parser.add_argument("action", nargs="?", default="build", choices=["build"])
    parser.add_argument("--out", type=Path, default=REPO, help="output root (default: the repository)")
    args = parser.parse_args(argv)
    res = build(args.out)
    print(f"labelled {len(res['labels'])} questions ({res['n_classified']} classified, "
          f"{len(res['labels']) - res['n_classified']} by rule)")
    print("test counts: " + ", ".join(f"{t}={res['test_counts'].get(t, 0)}" for t in CANONICAL))
    print(f"groups below {MIN_GROUP} in test: {', '.join(res['small_groups']) or 'none'}")
    print(f"review sheets: {res['n_test_sheet']} ambiguous test rows, {res['n_train_sheet']} spot-check rows")
    for name, path in res["paths"].items():
        print(f"  {name:12s} {_rel(path)}")
    return 0
