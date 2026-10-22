"""`./gb split-dev` draws the Tier-K dev split and its anchor and CI subsets (eval plan §4.1, Table 4; §6).

Eval plan §4.1 holds 200 of the 1,668 training questions back as the Tier-K dev split. The dev split
drives every choice: checkpoint selection, early stopping and the CI gate. Its questions and
everything derived from them are excluded from every training use. The anchor subset that compares
checkpoints during training is drawn from dev (§4.1), and so is the 20-question Tier-K part of the CI
smoke subset. The 10 Tier-F dev tasks of the CI subset are added after Phase 3.

Draw (`frozen.allocate`, algorithm stratified-largest-remainder-sha256-v1):
  * strata: the exact raw (origin, task_type, difficulty) values, as in the frozen split;
  * quotas: proportional to the FULL training pool (1,668), so the exclusions below cannot skew the
    dev mix; a stratum with too few eligible questions is capped and its slots re-apportioned
    (capacity rule in frozen.py);
  * eligible: training questions that are NOT (i) in a multi-member twin group (twins.py), (ii) a
    test twin (shares an 8-gram with a frozen test question), (iii) a test neighbour (semantic,
    lexical or keep_both evidence with a test question), or (iv) all-REJECT (every teacher answer
    rejected);
  * rank: sha256("grc-bench-dev-v1-2026-10-14" + "\\0" + id).
Anchor (60) and CI (20) apply the same algorithm to the 200 dev questions, with their own seeds.

Outputs are deterministic (sorted keys, stable order, no timestamps); running the command twice gives
byte-identical files. No item records are written here; `./gb tier-k-items` writes those.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from grcbench.common.io import iter_jsonl, read_jsonl, sha256_file, sha256_text, write_text
from grcbench.common.paths import DATA, DATASET, FROZEN, REPO

from . import frozen, twins

VERSION = 1
DEV_SEED = "grc-bench-dev-v1-2026-10-14"
ANCHOR_SEED = "grc-bench-anchor-v1-2026-10-14"
CI_SEED = "grc-bench-ci-v1-2026-10-14"
DEV_N, ANCHOR_N, CI_N = 200, 60, 20
DATE = "2026-10-14"

TRAIN = FROZEN / "train_questions.jsonl"
TEST = FROZEN / "eval_questions.jsonl"
JUDGMENTS = DATASET / "08_judging" / "judgments.jsonl"
REPAIRED = (DATASET / "10_repaired" / "repaired_answers.jsonl",
            DATASET / "10_repaired_glm" / "repaired_answers.jsonl")

OUT_DEV = Path("splits/tier_k_dev.json")
OUT_GROUPS = Path("splits/tier_k_twin_groups.json")
OUT_EXCLUDE = Path("handoff/training_team/exclude_question_ids.txt")
OUT_TWINS_TSV = Path("handoff/training_team/test_twins.tsv")
OUT_README = Path("handoff/training_team/README.md")
OUT_SHEET = Path("handoff/review_sheets/tier_k_twin_threshold_review.csv")
OUT_REPORT = Path("reports/tier_k_dev_split.md")

REASONS = ("twin_group_member", "test_twin_8gram", "test_neighbour_semantic", "test_neighbour_lexical",
           "test_neighbour_keep_both", "all_reject")


def _rel(path: Path) -> str:
    return path.relative_to(DATA).as_posix()


def _json(obj: Any) -> str:
    """Same canonical form as common.io.write_json."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1) + "\n"


# ---------------------------------------------------------------- inputs -------------------------

def judgment_status(ids: set[str]) -> dict[str, Counter]:
    status: dict[str, Counter] = defaultdict(Counter)
    for row in iter_jsonl(JUDGMENTS):
        if row["question_id"] in ids:
            status[row["question_id"]][row["status"]] += 1
    return status


def all_reject(status: Mapping[str, Counter]) -> list[str]:
    return sorted(q for q, c in status.items() if c and set(c) == {"REJECT"})


def repaired_rows(ids: set[str]) -> dict[str, Counter]:
    """Rows per question in each repaired-answer file (each row is one DPO repair pair)."""
    out: dict[str, Counter] = {}
    for path in REPAIRED:
        c = Counter(r["question_id"] for r in iter_jsonl(path) if r["question_id"] in ids)
        out[_rel(path)] = c
    return out


def source_files() -> list[dict[str, str]]:
    paths = [frozen.CANONICAL, frozen.MANIFEST, TRAIN, TEST, twins.SEMANTIC_CANDIDATES, JUDGMENTS, *REPAIRED]
    for p in twins.decision_files():
        paths += [p, p.with_name(p.name.replace("dedupe_decisions", "duplicate_clusters"))]
    return [{"path": _rel(p), "sha256": sha256_file(p)} for p in sorted(set(paths))]


# ---------------------------------------------------------------- exclusions ---------------------

def exclusions(g: Mapping[str, Any], ev: twins.Evidence, rejects: Sequence[str]) -> dict[str, list[str]]:
    """Training id -> sorted exclusion reasons (only excluded ids)."""
    out: dict[str, set[str]] = defaultdict(set)
    for m in g["members"]:
        out[m].add("twin_group_member")
    for t in ev.ngram_test:
        out[t].add("test_twin_8gram")
    for t, d in g["test_neighbours"].items():
        for via in d.values():
            for v in via:
                out[t].add("test_neighbour_" + v.split(":")[0])
    for q in rejects:
        out[q].add("all_reject")
    return {q: sorted(r) for q, r in sorted(out.items())}


def sensitivity(ev: twins.Evidence, train: Sequence[Mapping[str, Any]], rejects: Sequence[str],
                dev_ids: Sequence[str], chosen_excl: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Twin-group and eligibility statistics across thresholds (None = source not used).

    For each setting the dev draw is repeated: `redraw_overlap` is how many of its 200 ids are also
    in the chosen dev split, `redraw_rejected_by_chosen` how many of them the chosen setting excludes
    (i.e. would have a twin or test neighbour by the chosen evidence), and `chosen_dev_excluded` how
    many chosen dev ids this setting would exclude."""
    settings: list[tuple[float | None, float | None]] = [(None, None), (twins.SEMANTIC_THRESHOLD, None)]
    settings += [(s, twins.LEXICAL_THRESHOLD) for s in twins.SEMANTIC_SWEEP]
    settings += [(twins.SEMANTIC_THRESHOLD, x) for x in twins.LEXICAL_SWEEP if x != twins.LEXICAL_THRESHOLD]
    pool = Counter(frozen.stratum_of(r) for r in train)
    quota = frozen.largest_remainder(pool, DEV_N)
    dev = set(dev_ids)
    rows = []
    for sem, lex in settings:
        g = twins.graph(ev, sem, lex)
        excl = exclusions(g, ev, rejects)
        eligible_ids = [r["id"] for r in train if r["id"] not in excl]
        eligible = Counter(frozen.stratum_of(r) for r in train if r["id"] not in excl)
        try:
            redraw = set(frozen.allocate(train, DEV_N, DEV_SEED, eligible=eligible_ids).ids)
        except ValueError:
            redraw = set()
        sizes = sorted((len(m) for m in g["groups"].values()), reverse=True)
        rows.append({
            "semantic_threshold": sem, "lexical_threshold": lex,
            "chosen": sem == twins.SEMANTIC_THRESHOLD and lex == twins.LEXICAL_THRESHOLD,
            "edges": twins.edge_kind_counts(g["edges"]), "twin_groups": len(g["groups"]),
            "group_members": len(g["members"]), "largest_groups": sizes[:3],
            "test_neighbours": len(g["test_neighbours"]), "excluded": len(excl),
            "eligible": len(eligible_ids),
            "strata_short": sum(1 for s, q in quota.items() if eligible.get(s, 0) < q),
            "chosen_dev_excluded": len(dev & set(excl)),
            "redraw_overlap": len(redraw & dev),
            "redraw_rejected_by_chosen": len(redraw & set(chosen_excl)),
        })
    return rows


# ---------------------------------------------------------------- composition --------------------

def composition(pools: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    fields: dict[str, list[dict[str, Any]]] = {}
    for f in frozen.FIELDS:
        count = {name: Counter(str(r[f]) for r in recs) for name, recs in pools.items()}
        values = sorted({v for c in count.values() for v in c},
                        key=lambda v: (-count["training_pool"].get(v, 0), v))
        rows = []
        for v in values:
            row: dict[str, Any] = {"value": v}
            for name, recs in pools.items():
                row[name] = count[name].get(v, 0)
                row[name + "_pct"] = round(100 * count[name].get(v, 0) / len(recs), 1)
            rows.append(row)
        fields[f] = rows
    diff = {f: round(max(abs(r["dev_pct"] - r["test_pct"]) for r in rows), 1) for f, rows in fields.items()}
    return {"pools": {name: len(recs) for name, recs in pools.items()}, "fields": fields,
            "max_abs_share_difference_pp_dev_vs_test": diff}


# ---------------------------------------------------------------- build --------------------------

def build() -> dict[Path, str]:
    """Compute everything and return {relative output path: file text}."""
    check = frozen.reproduce_frozen()
    if not frozen.frozen_ok(check):
        raise SystemExit(f"frozen split does not reproduce; refusing to draw dev: {check}")
    train, test = read_jsonl(TRAIN), read_jsonl(TEST)
    by_id = {r["id"]: r for r in train}
    ev = twins.collect(train, test)
    g = twins.graph(ev)
    status = judgment_status(set(by_id))
    rejects = all_reject(status)
    excl = exclusions(g, ev, rejects)
    eligible = [i for i in by_id if i not in excl]

    dev = frozen.allocate(train, DEV_N, DEV_SEED, eligible=eligible)
    dev_records = [by_id[i] for i in dev.ids]
    anchor = frozen.allocate(dev_records, ANCHOR_N, ANCHOR_SEED)
    ci = frozen.allocate(dev_records, CI_N, CI_SEED)
    assert set(dev.ids).isdisjoint(excl) and set(anchor.ids) <= set(dev.ids) and set(ci.ids) <= set(dev.ids)

    test_strata = Counter(frozen.stratum_of(r) for r in test)
    a_by, c_by = anchor.by_stratum(), ci.by_stratum()
    strata = []
    for row in dev.strata:
        s = tuple(row["stratum"])
        strata.append({"stratum": row["stratum"], "training_pool": row["quota_population"],
                       "eligible": row["eligible"], "dev_proportional_quota": row["proportional_quota"],
                       "dev": row["allocated"], "anchor": a_by.get(s, {}).get("allocated", 0),
                       "ci": c_by.get(s, {}).get("allocated", 0), "test": test_strata.get(s, 0)})

    reason_counts = Counter(r for rs in excl.values() for r in rs)
    group_sizes = Counter(len(m) for m in g["groups"].values())
    neighbours = g["test_neighbours"]
    repaired = repaired_rows(set(by_id))
    comp = composition({"test": test, "dev": dev_records, "training_pool": train,
                        "anchor": [by_id[i] for i in anchor.ids], "ci": [by_id[i] for i in ci.ids]})
    sens = sensitivity(ev, train, rejects, dev.ids, excl)

    def affected(ids: Sequence[str]) -> dict[str, Any]:
        st = Counter()
        for i in ids:
            st.update(status.get(i, Counter()))
        return {"questions": len(ids), "judged_teacher_answers": sum(st.values()),
                "by_status": dict(sorted(st.items())),
                "repaired_answer_rows": {f: sum(c.get(i, 0) for i in ids) for f, c in sorted(repaired.items())}}

    dev_json = {
        "name": "GRC-Bench Tier-K dev split",
        "version": VERSION,
        "date": DATE,
        "generated_by": "grcbench.splits.dev (./gb split-dev)",
        "implements": ["eval plan §4.1 Table 4: 200 dev questions held back from the 1,668 training questions",
                       "eval plan §4.1: the anchor subset and the CI gate use the dev split",
                       "eval plan §6: near-twins of test questions"],
        "status": "draft: twin thresholds are Claude's reading, pending team review",
        "algorithm": {
            "name": frozen.ALGORITHM,
            "description": ("Largest-remainder proportional allocation over exact raw (origin, task_type, "
                            "difficulty) strata (ties: ascending stratum tuple, exact fractions); within a "
                            "stratum ids ranked by ascending sha256(seed + NUL + id).hexdigest(); ids listed "
                            "in training-file order."),
            "dev_quota_population": "the full training pool (1,668 questions), before exclusions",
            "capacity_rule": ("A stratum whose quota exceeds its eligible count takes all eligible members and "
                              "is fixed; the remaining slots are re-apportioned by the same largest-remainder "
                              "rule over the unfixed strata (still weighted by training-pool sizes), repeated "
                              "until no quota exceeds capacity."),
            "capacity_rounds": {"dev": dev.capped_rounds, "anchor": anchor.capped_rounds, "ci": ci.capped_rounds},
            "subsets": "anchor and ci: the same algorithm over the 200 dev questions (quotas from dev)",
        },
        "seeds": {"dev": DEV_SEED, "anchor": ANCHOR_SEED, "ci": CI_SEED, "frozen_test": check["seed"]},
        "sources": source_files(),
        "frozen_split_check": check,
        "counts": {"training_pool": len(train), "test": len(test), "excluded": len(excl),
                   "eligible": len(eligible), "dev": len(dev.ids), "anchor": len(anchor.ids), "ci_tier_k": len(ci.ids)},
        "exclusions": {
            "rule": "dev candidates exclude every training question with at least one of these reasons",
            "by_reason": {r: reason_counts.get(r, 0) for r in REASONS},
            "unique_questions": len(excl),
        },
        "twin_groups": {
            "file": OUT_GROUPS.as_posix(),
            "semantic_threshold": twins.SEMANTIC_THRESHOLD,
            "lexical_threshold": twins.LEXICAL_THRESHOLD,
            "threshold_rule": ("lower edge of the lowest score band in which at least half of the training "
                               "pairs Claude read (no shared 8-gram) were twins; checked against union-find "
                               "chaining"),
            "band_reading": {k: twins.band_counts(ev, k) for k in ("semantic", "lexical")},
            "edges_by_kind": twins.edge_kind_counts(g["edges"]),
            "groups": len(g["groups"]),
            "members": len(g["members"]),
            "group_size_histogram": {str(k): v for k, v in sorted(group_sizes.items())},
            "semantic_candidates_mapping": ev.semantic_stats,
            "keep_both_decisions": ev.keep_both_summary,
            "coverage_note": ("semantic_duplicate_candidates.jsonl scores only source-lane questions (floor "
                              "0.82); pairs with a blind-lane question are covered by the lexical TF-IDF "
                              "check, 8-grams and adjudicated KEEP_BOTH decisions."),
            "sensitivity": sens,
        },
        "test_twins": {
            "definition": "training questions sharing a word 8-gram with a frozen test question (eval plan §6)",
            "tokenizer": "lowercase; every non-word character (regex \\W) to a space; split on whitespace; n = 8",
            "test_questions_with_training_twin": len({e for v in ev.ngram_test.values() for e in v}),
            "training_questions": len(ev.ngram_test),
            "by_training_id": {t: sorted(v) for t, v in sorted(ev.ngram_test.items())},
        },
        "test_neighbours": {
            "definition": ("training questions linked by semantic, lexical or keep_both evidence (thresholds "
                           "above) to a frozen test question they share no 8-gram with; flagged, not counted "
                           "in eval plan §6"),
            "training_questions": len(neighbours),
            "also_test_twins": len(set(neighbours) & set(ev.ngram_test)),
            "by_training_id": neighbours,
        },
        "all_reject_ids": rejects,
        "strata": strata,
        "dev_ids": dev.ids,
        "anchor_ids": anchor.ids,
        "ci_ids": ci.ids,
        "ci_note": "The CI smoke subset is these 20 Tier-K dev questions plus 10 Tier-F dev tasks added after Phase 3.",
        "composition": comp,
        "training_artifacts_affected": {
            "dev": affected(dev.ids), "test_twins": affected(sorted(ev.ngram_test)),
            "test_neighbours_not_test_twins": affected(sorted(set(neighbours) - set(ev.ngram_test)))},
    }

    groups_json = {
        "name": "GRC-Bench Tier-K twin groups over the 1,668 training questions",
        "version": VERSION,
        "date": DATE,
        "generated_by": "grcbench.splits.dev (./gb split-dev)",
        "semantic_threshold": twins.SEMANTIC_THRESHOLD,
        "lexical_threshold": twins.LEXICAL_THRESHOLD,
        "evidence": {
            "8gram": "the two questions share a word 8-gram (tokenizer in tier_k_dev.json test_twins)",
            "semantic": "semantic_duplicate_candidates.jsonl similarity (source lane only) >= semantic_threshold",
            "lexical": "TF-IDF cosine >= lexical_threshold, only for pairs the semantic file never scored",
            "keep_both": "KEEP_BOTH decision in a *semantic_dedupe_decisions.jsonl (cluster id given)",
        },
        "cluster_rule": ("Tier-K cluster_id: a training question in a group takes the group id; every other "
                         "question (every dev question among them) is a singleton whose cluster_id is its own id."),
        "groups": g["groups"],
        "edges": [[a, b, via] for (a, b), via in g["edges"].items()],
    }

    files = {
        OUT_DEV: _json(dev_json),
        OUT_GROUPS: _json(groups_json),
        OUT_EXCLUDE: exclude_text(dev.ids),
        OUT_TWINS_TSV: twins_tsv(ev.ngram_test, neighbours),
        OUT_SHEET: review_sheet(ev, by_id),
    }
    files[OUT_README] = readme(dev_json, sha256_text(files[OUT_EXCLUDE]))
    files[OUT_REPORT] = report(dev_json)
    return files


# ---------------------------------------------------------------- writers ------------------------

def exclude_text(dev_ids: Sequence[str]) -> str:
    head = [f"# GRC-Bench Tier-K dev split v{VERSION} ({DATE}): {len(dev_ids)} training-question ids, sorted.",
            "# Exclude these questions, and everything derived from them, from EVERY training use.",
            "# Details: README.md in this folder. Source: splits/tier_k_dev.json (dev_ids)."]
    return "\n".join(head + sorted(dev_ids)) + "\n"


def near_test(ngram_test: Mapping[str, set[str]],
              neighbours: Mapping[str, Mapping[str, Sequence[str]]]) -> dict[str, dict[str, list[str]]]:
    """Training id -> {test id: evidence}, merging 8-gram twins and flagged neighbours."""
    out: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for t, ids in ngram_test.items():
        for e in ids:
            out[t][e] = ["8gram"]
    for t, d in neighbours.items():
        for e, via in d.items():
            out[t][e] = sorted(out[t].get(e, []) + list(via))
    return {t: dict(sorted(d.items())) for t, d in sorted(out.items())}


def twins_tsv(ngram_test: Mapping[str, set[str]], neighbours: Mapping[str, Mapping[str, Sequence[str]]]) -> str:
    lines = [f"# GRC-Bench Tier-K v{VERSION} ({DATE}): training questions near a frozen test question, one row each.",
             "# kind=test_twin_8gram: eval plan §6 near-twin (shares a word 8-gram with a test question);"
             " remove its answers from training where possible.",
             "# kind=test_neighbour: no shared 8-gram; flagged by semantic/lexical/keep_both evidence; the team decides.",
             "# evidence: test_id=kind[:score or cluster], several joined by '+', test questions separated by ';'.",
             "training_id\tkind\ttest_ids\tevidence"]
    for t, d in near_test(ngram_test, neighbours).items():
        kind = "test_twin_8gram" if t in ngram_test else "test_neighbour"
        evidence = ";".join(f"{e}={'+'.join(via)}" for e, via in d.items())
        lines.append(f"{t}\t{kind}\t{','.join(d)}\t{evidence}")
    return "\n".join(lines) + "\n"


def review_sheet(ev: twins.Evidence, by_id: Mapping[str, Mapping[str, Any]]) -> str:
    """Every training pair Claude read to set the thresholds (training questions only)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["source", "band", "score", "linked_now", "id_a", "id_b", "origin_a", "origin_b",
                "question_a", "question_b", "team_label (twin / not twin)", "reviewer", "notes"])
    train = set(by_id)
    for kind, scores, thr in (("semantic", ev.semantic, twins.SEMANTIC_THRESHOLD),
                              ("lexical", ev.lexical, twins.LEXICAL_THRESHOLD)):
        for band in twins.BAND_READING[kind]:
            lo, hi = band["band"]
            rows = [(s, a, b) for (a, b), s in scores.items()
                    if a in train and b in train and lo <= s < hi and (a, b) not in ev.ngram_train]
            for s, a, b in sorted(rows, key=lambda x: (-x[0], x[1], x[2])):
                w.writerow([kind, f"[{lo:.2f}, {min(hi, 1.0):.2f})", f"{s:.4f}", "yes" if s >= thr else "no",
                            a, b, by_id[a]["origin"], by_id[b]["origin"], by_id[a]["question"],
                            by_id[b]["question"], "", "", ""])
    return buf.getvalue()


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _thr(x: float | None) -> str:
    return "off" if x is None else f"{x:.2f}"


def sensitivity_table(rows: Sequence[Mapping[str, Any]]) -> str:
    return _table(
        ["semantic", "lexical", "8gram / sem / lex / keep_both edges", "groups", "members", "largest",
         "test neighbours", "excluded", "eligible", "strata short", "chosen dev excluded", "redraw overlap",
         "redraw rejected by chosen"],
        [[_thr(r["semantic_threshold"]) + (" **(chosen)**" if r["chosen"] else ""), _thr(r["lexical_threshold"]),
          " / ".join(str(r["edges"].get(k, 0)) for k in ("8gram", "semantic", "lexical", "keep_both")),
          r["twin_groups"], r["group_members"], ", ".join(map(str, r["largest_groups"])), r["test_neighbours"],
          r["excluded"], r["eligible"], r["strata_short"], r["chosen_dev_excluded"], r["redraw_overlap"],
          r["redraw_rejected_by_chosen"]] for r in rows])


def composition_tables(comp: Mapping[str, Any]) -> str:
    parts = []
    for f, rows in comp["fields"].items():
        parts.append(f"**{f}** (max |dev - test| = {comp['max_abs_share_difference_pp_dev_vs_test'][f]} pp)\n")
        parts.append(_table([f, "test n", "test %", "dev n", "dev %", "training pool n", "training pool %"],
                            [[r["value"], r["test"], r["test_pct"], r["dev"], r["dev_pct"], r["training_pool"],
                              r["training_pool_pct"]] for r in rows]))
        parts.append("")
    return "\n".join(parts)


def readme(d: Mapping[str, Any], exclude_sha: str) -> str:
    a = d["training_artifacts_affected"]
    dev_a, tw_a, nb_a = a["dev"], a["test_twins"], a["test_neighbours_not_test_twins"]
    rep = lambda x: ", ".join(f"{v} in `{k}`" for k, v in x["repaired_answer_rows"].items())
    st = lambda x: ", ".join(f"{k} {v}" for k, v in x["by_status"].items())
    tt = d["test_twins"]
    twin_rows = [[t, ", ".join(e)] for t, e in tt["by_training_id"].items()]
    nb = d["test_neighbours"]["by_training_id"]
    nb_only = {t: m for t, m in nb.items() if t not in tt["by_training_id"]}
    nb_also = sorted(t for t in nb if t in tt["by_training_id"])
    nb_rows = [[t, ", ".join(f"{e} ({' + '.join(v)})" for e, v in m.items())] for t, m in nb_only.items()]
    return f"""# Tier-K dev split: what the training team must exclude

GRC-Bench v{d['version']}, {d['date']}. Generated by `./gb split-dev`; do not edit by hand.
Status: {d['status']}. If the team changes a threshold, the dev split gets a new version, so always check
the version line at the top of `exclude_question_ids.txt`.

## 1. The 200 dev questions: exclude from every training use

`exclude_question_ids.txt` lists {d['counts']['dev']} question ids (SHA-256 of the file:
`{exclude_sha}`). They come from `04_eval_frozen/train_questions.jsonl`, and eval plan §4.1 holds them
back as the Tier-K dev split. The rule for these questions **and everything derived from them**:

- **no** teacher answers in SFT (any teacher in `05_answers/`, including duplicate rows such as laguna-s21's);
- **no** repaired answers (`10_repaired/`, `10_repaired_glm/`) and **no** DPO/DPOP repair pairs built from them;
- **no** RAFT retrieval-aware examples built on them (oracle or distractor documents);
- **no** SliCK known/unknown tagging of them (it would make them part of the training recipe);
- **no** use as few-shot examples, and no new teacher generations for them.

What is allowed: scoring checkpoints on them through GRC-Bench, for early stopping, checkpoint
selection and the CI gate. That is what the dev split is for (eval plan §4.1).

What this removes from the current data: {dev_a['judged_teacher_answers']} judged teacher answers
({st(dev_a)}) and these repaired-answer rows (each one a DPO repair pair): {rep(dev_a)}.

The dev questions were chosen so that none of them has a twin left in training. Twin groups are
built from shared 8-word sequences, the dataset's semantic-similarity scores, a lexical TF-IDF check
and the dataset team's KEEP_BOTH decisions. No other training question has to be removed to protect
the dev split.

## 2. Test twins: remove their answers from training where possible (eval plan §6)

{tt['training_questions']} training questions share an 8-word sequence with {tt['test_questions_with_training_twin']}
of the 185 frozen test questions. Most are template twins, such as the same question asked about a
different signal. GRC-Bench reports those test questions separately from clean ones. Eval plan §6
asks that these training questions' answers be removed from training **where possible**. They cover
{tw_a['judged_teacher_answers']} judged teacher answers ({st(tw_a)}); repaired-answer rows: {rep(tw_a)}.
The ids are below, and also in `test_twins.tsv` (kind `test_twin_8gram`).

{_table(['training question', 'frozen test question(s) it shares an 8-gram with'], twin_rows)}

## 3. Flagged test neighbours: the team decides

These {len(nb_only)} training questions share no 8-word sequence with any test question, but are close
to one by the semantic score (>= {d['twin_groups']['semantic_threshold']}), the lexical TF-IDF check
(>= {d['twin_groups']['lexical_threshold']}) or a KEEP_BOTH decision. They are kept out of the dev
split. Removing their answers from training is optional; the team decides. They cover
{nb_a['judged_teacher_answers']} judged teacher answers ({st(nb_a)}). They are also listed in
`test_twins.tsv` (kind `test_neighbour`). A further {len(nb_also)} test twins from section 2 are also
close to another test question; `test_twins.tsv` shows that evidence.

{_table(['training question', 'test question (evidence)'], nb_rows)}

## 4. Other notes

- The {len(d['all_reject_ids'])} training questions whose every teacher answer was rejected are not in dev. Nothing to do.
- Claude (Anthropic) wrote this tooling and these lists. Claude never judges, repairs or generates training data.
- Full details (algorithm, seeds, source hashes, sensitivity tables): `splits/tier_k_dev.json` and `reports/tier_k_dev_split.md`.
"""


def report(d: Mapping[str, Any]) -> str:
    c, x, t = d["counts"], d["exclusions"]["by_reason"], d["twin_groups"]
    fz = d["frozen_split_check"]
    bands = lambda k: _table(["band", "pairs read", "judged twins", "share"],
                             [[f"[{b['band'][0]:.2f}, {min(b['band'][1], 1.0):.2f})", b["pairs_now"], b["twins"],
                               b["twin_share"]] for b in t["band_reading"][k]])
    strata = [s for s in d["strata"] if s["dev"] or s["test"]]
    capped = d["algorithm"]["capacity_rounds"]["dev"]
    return f"""# Tier-K dev split (eval plan §4.1, §6)

Generated by `./gb split-dev` (GRC-Bench v{d['version']}, {d['date']}). Status: {d['status']}.

## Frozen split check
The 185 frozen test ids were regenerated from `questions_canonical.jsonl` with `{fz['algorithm']}`
and seed `{fz['seed']}`: source SHA-256 matches = {fz['source_sha256_matches']}; ids match the
manifest in order = {fz['ids_match_in_order']} ({fz['reproduced_ids']}/{fz['manifest_ids']}); the train
file is the complement = {fz['train_file_is_complement']}.

## Draw
- Pool: {c['training_pool']} training questions. Quotas are proportional to the full pool over the raw
  (origin, task_type, difficulty) strata; ranking uses sha256(`{d['seeds']['dev']}` + NUL + id).
- Excluded from dev ({c['excluded']} unique questions, so {c['eligible']} eligible):
  twin-group members {x['twin_group_member']}; test twins (8-gram) {x['test_twin_8gram']};
  test neighbours (semantic {x['test_neighbour_semantic']}, lexical {x['test_neighbour_lexical']},
  keep_both {x['test_neighbour_keep_both']}); all-REJECT {x['all_reject']}. The reasons overlap.
- Capacity rule triggered: {'no, every stratum had enough eligible questions' if not capped else capped}.
- Dev {c['dev']}; anchor {c['anchor']} (seed `{d['seeds']['anchor']}`); CI {c['ci_tier_k']} Tier-K questions
  (seed `{d['seeds']['ci']}`), plus 10 Tier-F dev tasks after Phase 3.

## Twin groups and thresholds
Edges: {', '.join(f'{k} {v}' for k, v in t['edges_by_kind'].items())}. Groups: {t['groups']}
({t['members']} questions); size histogram {t['group_size_histogram']}.

`semantic_duplicate_candidates.jsonl` uses ids from `questions_extracted.jsonl`. They are mapped to
canonical ids through (source_file, source_index) provenance:
{', '.join(f'{k} {v}' for k, v in t['semantic_candidates_mapping'].items())}. The file scores only
source-lane questions, so pairs with a blind-lane question get a lexical TF-IDF check instead (a
deviation from the WS1 brief, which lists only 8-grams, semantic candidates and KEEP_BOTH).

Threshold rule: {t['threshold_rule']}. Claude read every pair in these bands (14 Oct 2026; the
team's check is in `handoff/review_sheets/tier_k_twin_threshold_review.csv`).

Semantic similarity (training pairs without a shared 8-gram):

{bands('semantic')}

Lexical TF-IDF cosine (training pairs with a blind-lane question, without a shared 8-gram):

{bands('lexical')}

Sensitivity. Columns: "strata short" = strata whose dev quota exceeds their eligible count; "chosen
dev excluded" = chosen dev questions this setting would exclude; "redraw overlap" = how many of the
200 questions drawn at this setting are in the chosen dev split; "redraw rejected by chosen" = how
many of those 200 have a twin or test neighbour under the chosen evidence.

{sensitivity_table(t['sensitivity'])}

## Dev vs test composition (raw labels)
{composition_tables(d['composition'])}
## Strata with a dev or test question
{_table(['origin', 'task_type', 'difficulty', 'training pool', 'eligible', 'quota', 'dev', 'anchor', 'ci', 'test'],
        [[*s['stratum'], s['training_pool'], s['eligible'], s['dev_proportional_quota'], s['dev'], s['anchor'],
          s['ci'], s['test']] for s in strata])}

## Open points for the team
- Confirm or change the two thresholds using the review sheet. A change gives a new dev version.
- The lexical check is an addition to the brief. Drawn without it (semantic {_thr(t['sensitivity'][1]['semantic_threshold'])},
  lexical off), {t['sensitivity'][1]['redraw_rejected_by_chosen']} of the 200 dev questions would have a blind-lane
  near-twin left in training or a lexical test neighbour.
- The WS2 embedding audit (EmbeddingGemma) should re-check the dev ids against training; anything new is flagged, never auto-removed.
"""


def write(files: Mapping[Path, str], root: Path) -> None:
    for rel, text in files.items():
        write_text(root / rel, text)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb split-dev", description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO, help="root folder for the outputs (default: the repo)")
    args = ap.parse_args(argv)
    files = build()
    write(files, args.out)
    d = json.loads(files[OUT_DEV])
    print(f"dev {d['counts']['dev']} | anchor {d['counts']['anchor']} | ci {d['counts']['ci_tier_k']} | "
          f"excluded {d['counts']['excluded']} | eligible {d['counts']['eligible']}")
    for rel in files:
        print(f"  {rel.as_posix():55s} sha256 {sha256_text(files[rel])[:16]}")
    return 0
