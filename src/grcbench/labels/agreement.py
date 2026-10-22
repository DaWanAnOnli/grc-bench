"""Agreement on the task-type labels: Cohen's kappa for the returned review sheets.

Proposal §4.2 promises "a 10% human spot-check and inter-annotator agreement reported"; the WS2 plan
adds blind double labelling of every ambiguous test item. This module reads the sheets written by
`task_types.py` once the team has filled them in, and reports:
  * ambiguous test items: annotator 1 vs annotator 2, and each annotator vs Claude (Cohen's κ);
  * training spot-check: reviewer vs the rule map / Claude label, overall and by method.

Command:  ./gb labels agreement [--test CSV [CSV ...]] [--train CSV] [--key JSON] [--out DIR]
Annotators may return one merged sheet or one copy each; copies are merged by question id.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from grcbench.common.io import read_json, write_json, write_text
from grcbench.common.paths import REPO, REPORTS

from .task_types import CANONICAL, KEY_REL, RULE_MAP

AGREEMENT_MD_REL = REPORTS.relative_to(REPO) / "labels_agreement.md"
AGREEMENT_JSON_REL = REPORTS.relative_to(REPO) / "labels_agreement.json"


def observed_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty label lists of equal length")
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's κ = (p_o - p_e) / (1 - p_e) for two raters over the same items.

    Returns NaN when chance agreement is 1 (both raters used one and the same category for every
    item), where κ is undefined.
    """
    p_o = observed_agreement(a, b)
    n = len(a)
    ca, cb = Counter(a), Counter(b)
    p_e = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / (n * n)
    if p_e == 1.0:
        return float("nan")
    return (p_o - p_e) / (1.0 - p_e)


def pair_stats(a: Sequence[str], b: Sequence[str]) -> dict[str, Any]:
    """κ and raw agreement over the items both raters labelled ('' means not labelled)."""
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return {"n": 0, "agreement": None, "kappa": None}
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    kappa = cohen_kappa(xs, ys)
    return {"n": len(pairs), "agreement": round(observed_agreement(xs, ys), 4),
            "kappa": None if kappa != kappa else round(kappa, 4)}


def normalise_label(text: str) -> str:
    """Map an annotator's entry to a canonical type; '' if blank. Raises ValueError if unknown."""
    t = " ".join((text or "").strip().split())
    if not t:
        return ""
    for cand in (t, t.lower(), t.lower().replace(" ", "_").replace("-", "_")):
        if cand in CANONICAL:
            return cand
        if cand in RULE_MAP:
            return RULE_MAP[cand]
    raise ValueError(f"'{text}' is not one of the six task types")


def read_sheet(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return [{k: (v or "") for k, v in row.items()} for row in csv.DictReader(fh)]


def _merge_column(sheets: list[list[dict[str, str]]], column: str, problems: list[str]) -> dict[str, str]:
    """Collect one label column across copies of a sheet; flag conflicts and invalid entries."""
    merged: dict[str, str] = {}
    for rows in sheets:
        for row in rows:
            try:
                value = normalise_label(row.get(column, ""))
            except ValueError as exc:
                problems.append(f"{row.get('id')}: {column}: {exc}")
                continue
            if not value:
                continue
            old = merged.get(row["id"])
            if old and old != value:
                problems.append(f"{row['id']}: {column} differs between copies ({old} vs {value})")
            merged[row["id"]] = value
    return merged


def ambiguous_test_agreement(sheet_paths: Sequence[Path], key: dict[str, Any]) -> dict[str, Any]:
    rows_key = key["ambiguous_test"]["rows"]
    ids = [r["id"] for r in rows_key]
    claude = {r["id"]: r["claude_label"] for r in rows_key}
    sheets = [read_sheet(p) for p in sheet_paths]
    problems: list[str] = []
    for path, rows in zip(sheet_paths, sheets):
        extra = sorted({r.get("id", "") for r in rows} - set(ids))
        if extra:
            problems.append(f"{path.name}: ids not in the key: {', '.join(extra)}")
    a1 = _merge_column(sheets, "annotator_1", problems)
    a2 = _merge_column(sheets, "annotator_2", problems)
    col1 = [a1.get(i, "") for i in ids]
    col2 = [a2.get(i, "") for i in ids]
    colc = [claude[i] for i in ids]
    disagreements = [{"id": i, "annotator_1": x, "annotator_2": y, "claude": c}
                     for i, x, y, c in zip(ids, col1, col2, colc) if len({v for v in (x, y, c) if v}) > 1]
    return {
        "n_items": len(ids),
        "labelled": {"annotator_1": sum(1 for x in col1 if x), "annotator_2": sum(1 for x in col2 if x)},
        "pairs": {
            "annotator_1 vs annotator_2": pair_stats(col1, col2),
            "annotator_1 vs claude": pair_stats(col1, colc),
            "annotator_2 vs claude": pair_stats(col2, colc),
        },
        "disagreements": disagreements,
        "problems": problems,
    }


def spotcheck_agreement(sheet_path: Path, key: dict[str, Any]) -> dict[str, Any]:
    rows_key = key["spotcheck_train"]["rows"]
    given = {r["id"]: r["label"] for r in rows_key}
    method = {r["id"]: r["method"] for r in rows_key}
    problems: list[str] = []
    reviewer = _merge_column([read_sheet(sheet_path)], "reviewer_label", problems)
    unknown = sorted(set(reviewer) - set(given))
    if unknown:
        problems.append(f"ids not in the key: {', '.join(unknown)}")
    result: dict[str, Any] = {"n_items": len(given), "labelled": sum(1 for i in given if reviewer.get(i)),
                              "by_method": {}, "disagreements": [], "problems": problems}
    groups = {"all": list(given)} | {m: [i for i in given if method[i] == m] for m in ("classified", "rule")}
    for name, ids in groups.items():
        result["by_method"][name] = pair_stats([reviewer.get(i, "") for i in ids], [given[i] for i in ids])
    result["disagreements"] = [{"id": i, "method": method[i], "given": given[i], "reviewer": reviewer[i]}
                               for i in given if reviewer.get(i) and reviewer[i] != given[i]]
    return result


def _fmt(stats: dict[str, Any]) -> list[Any]:
    if not stats["n"]:
        return [0, "–", "–"]
    kappa = "undefined (one category)" if stats["kappa"] is None else f"{stats['kappa']:.3f}"
    return [stats["n"], f"{stats['agreement']:.1%}", kappa]


def render(result: dict[str, Any]) -> str:
    out = ["# Task-type labels: agreement", "",
           "Cohen's κ for the returned task-type review sheets (proposal §4.2; see `reports/labels.md`). "
           "Generated by `./gb labels agreement`; do not edit by hand.", ""]
    head = ["pair", "items labelled by both", "raw agreement", "Cohen's κ"]
    test = result.get("ambiguous_test")
    if test:
        out += ["## Ambiguous test items (blind double labelling)", "",
                f"{test['n_items']} items; annotator 1 labelled {test['labelled']['annotator_1']}, "
                f"annotator 2 labelled {test['labelled']['annotator_2']}.", ""]
        out += ["| " + " | ".join(head) + " |", "|---|---|---|---|"]
        out += ["| " + " | ".join(str(c) for c in [name, *_fmt(s)]) + " |" for name, s in test["pairs"].items()]
        out += ["", "With few items, κ has a wide interval: report it, but settle disagreements by discussion.", ""]
        if test["disagreements"]:
            out += ["Items where the labels differ:", "", "| id | annotator 1 | annotator 2 | Claude |",
                    "|---|---|---|---|"]
            out += [f"| `{d['id']}` | {d['annotator_1'] or '–'} | {d['annotator_2'] or '–'} | {d['claude']} |"
                    for d in test["disagreements"]]
            out.append("")
    spot = result.get("spotcheck_train")
    if spot:
        out += ["## Training spot-check", "",
                f"{spot['n_items']} rows; the reviewer labelled {spot['labelled']}.", "",
                "| rows | items labelled by both | raw agreement | Cohen's κ |", "|---|---|---|---|"]
        out += ["| " + " | ".join(str(c) for c in [name, *_fmt(s)]) + " |" for name, s in spot["by_method"].items()]
        out.append("")
        if spot["disagreements"]:
            out += ["Rows where the reviewer disagrees:", "", "| id | method | given | reviewer |",
                    "|---|---|---|---|"]
            out += [f"| `{d['id']}` | {d['method']} | {d['given']} | {d['reviewer']} |" for d in spot["disagreements"]]
            out.append("")
    problems = (test or {}).get("problems", []) + (spot or {}).get("problems", [])
    if problems:
        out += ["## Problems in the returned sheets", ""] + [f"- {p}" for p in problems] + [""]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="./gb labels agreement",
                                     description="Cohen's kappa for the returned task-type review sheets.")
    parser.add_argument("--test", type=Path, nargs="+", default=[],
                        help="returned ambiguous-test sheet(s): one merged sheet or one copy per annotator")
    parser.add_argument("--train", type=Path, help="returned training spot-check sheet")
    parser.add_argument("--key", type=Path, default=REPO / KEY_REL, help="review key (splits/...)")
    parser.add_argument("--out", type=Path, default=REPO, help="output root (default: the repository)")
    args = parser.parse_args(argv)
    if not args.test and not args.train:
        parser.error("give --test and/or --train")
    key = read_json(args.key)
    result: dict[str, Any] = {}
    if args.test:
        result["ambiguous_test"] = ambiguous_test_agreement(args.test, key)
    if args.train:
        result["spotcheck_train"] = spotcheck_agreement(args.train, key)
    write_json(args.out / AGREEMENT_JSON_REL, result)
    text = render(result)
    write_text(args.out / AGREEMENT_MD_REL, text)
    print(text)
    return 0
