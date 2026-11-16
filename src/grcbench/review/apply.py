"""Applying decided records to the files the benchmark code reads.

Twins (handoff/review_sheets/similarity_sheets_instructions.md, section B): the decided labels are
counted per score band, and a threshold is the lower edge of the lowest band in which at least half of
the pairs are twins.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable, Mapping

_BAND = re.compile(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")


def _band(text: str) -> tuple[float, float]:
    m = _BAND.match(text.strip())
    if not m:
        raise ValueError(f"not a band: {text!r}")
    return float(m.group(1)), float(m.group(2))


def band_counts(sheet_rows: Iterable[Mapping[str, str]], recs: Iterable[Mapping[str, Any]]
                ) -> dict[str, list[dict[str, Any]]]:
    """{source: [{"band": [lo, hi], "pairs": n, "twins": n}, ...]} in ascending band order."""
    final = {tuple(r["key"]): r["final"] for r in recs}
    counts: dict[str, dict[tuple[float, float], list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in sheet_rows:
        k = (row["source"], row["id_a"], row["id_b"])
        if k not in final:
            raise ValueError(f"twin pair {k} has no decision")
        c = counts[row["source"]][_band(row["band"])]
        c[0] += 1
        c[1] += final[k] == "twin"
    return {src: [{"band": [lo, hi], "pairs": p, "twins": t} for (lo, hi), (p, t) in sorted(bands.items())]
            for src, bands in counts.items()}


RUBRIC_FIELDS = ("required_points", "disqualifying_errors", "numeric", "numeric_tolerance", "sources")
_STATUS = {"approve": "approved", "approve with edits": "approved-with-edits", "reject": "rejected",
           "underspecified": "rejected"}


def final_fields(rec: Mapping[str, Any], writer: Mapping[str, Any] | None = None) -> Mapping[str, Any] | None:
    """The row whose rubric fields are final for a decided question: the writer's where the checkers (or
    the expert, unchanged) approved them, the expert's where the expert decided edits; None for a
    question rejected or underspecified."""
    final = rec["final"]
    if final not in ("approve", "approve with edits"):
        return None
    src = writer if writer is not None else rec["writer"]
    if rec["decided_by"].startswith("EX"):
        ex = rec.get("ex") or rec.get("audit") or {}
        if final == "approve with edits" or any(f in ex for f in RUBRIC_FIELDS):
            missing = [f for f in RUBRIC_FIELDS if f not in ex]
            if missing:
                raise ValueError(f"{rec['key'][0]}: the expert's decision gives no final {missing}")
            src = ex
    elif final == "approve with edits":
        raise ValueError(f"{rec['key'][0]}: `approve with edits` without the expert's final fields")
    return src


def calc_status_text(task_type: str, calc: Mapping[str, Any] | None, rubric: Mapping[str, Any], date: str,
                     calc_date: str | None = None) -> str:
    """The rubric's calc_status (docs/guidelines/rubric_writing.md): who confirmed a calculation question's
    value. `calc` is the question's decided record of the blind recomputation (test questions), or None
    for a calculation question outside that sheet, whose value the rubric's checkers or expert checked.
    `date` is the rubric's decision date, `calc_date` the recomputation's (default: `date`)."""
    if task_type != "calculation":
        return ""
    checked = (f"value decided by EX {date}" if rubric["decided_by"].startswith("EX")
               else f"value checked by C1;C2 {date}")
    if calc is None:
        return checked
    cd = calc_date or date
    if calc["decided_by"].startswith("EX"):
        return f"decided by EX {cd}"
    return {"agrees": f"recomputed by R1;R2 {cd}", "underspecified": f"underspecified by R1;R2 {cd}",
            "no value": f"no script value (R1;R2 {cd}); {checked}"}[calc["final"]]


def rubric_record(meta: Mapping[str, Any], writer: Mapping[str, Any], rec: Mapping[str, Any],
                  calc_status: str) -> dict[str, Any]:
    """The judge's rubric record (judge_prompt.normalize_rubric) for one decided question: the writer's
    fields where the checkers approved them, the expert's final fields where the expert decided edits."""
    import hashlib

    from grcbench.common.canary import CANARY
    from grcbench.review.validate import parse_lines
    from grcbench.tier_k.rubrics import TEMPLATE_VERSION
    final = rec["final"]
    src = final_fields({**rec, "key": (meta["question_id"],)}, writer)
    reviewers = ["W", "C1", "C2"] + (["EX"] if rec["decided_by"].startswith("EX") else [])
    note = f"decided by {rec['decided_by']}"
    if final in ("reject", "underspecified"):
        why = " ".join(((rec.get("ex") or rec.get("audit") or {}).get("notes") or writer.get("notes") or "").split())
        note = f"{final}: {why}" if final == "underspecified" else f"rejected ({note}): {why}"
    usable = src is not None
    src = src or {}
    return {
        "question_id": meta["question_id"], "split": meta["split"], "task_type": meta["task_type"],
        "question_sha256": hashlib.sha256(meta["question"].encode("utf-8")).hexdigest(),
        "required_points": parse_lines(src.get("required_points") or [], "P") if usable else [],
        "disqualifying_errors": parse_lines(src.get("disqualifying_errors") or [], "D") if usable else [],
        "numeric": src.get("numeric") if usable else None,
        "numeric_tolerance": src.get("numeric_tolerance") if usable else None,
        "sources": list(src.get("sources") or []) if usable else [],
        "calc_status": calc_status,
        "review": {"status": _STATUS[final], "reviewers": reviewers, "notes": note},
        "template_version": TEMPLATE_VERSION, "canary": CANARY,
    }


def rubric_file(questions: list[Mapping[str, Any]], recs: Iterable[Mapping[str, Any]],
                calc: Mapping[str, Mapping[str, Any]], date: str, calc_date: str | None = None,
                patches: list[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """One judge's rubric record per question, in the question file's order. Every question needs its
    decided record (with the writer's row); `calc` holds the blind recomputation's decided records by id.
    `patches` (review.patches format, `item` = question id) are the expert review's (H1b) edits on a
    decided rubric: applied to the judge's record, which then says so in its review field."""
    from grcbench.review import patches as patch_ops
    by_id = {r["key"][0]: r for r in recs}
    missing = [q["id"] for q in questions if q["id"] not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} question(s) with no decided rubric, e.g. {missing[:3]}")
    out = []
    for q in questions:
        rec = by_id[q["id"]]
        meta = {"question_id": q["id"], "split": q["split"], "task_type": q["task_type"], "question": q["question"]}
        r = rubric_record(meta, rec["writer"], rec,
                          calc_status_text(q["task_type"], calc.get(q["id"]), rec, date, calc_date))
        mine = [p for p in patches or [] if p["item"] == q["id"]]
        if mine:
            edits = [p for p in mine if "path" in p]       # an entry without a path records a decision, no change
            if edits and r["review"]["status"] == "rejected":
                raise ValueError(f"{q['id']}: an edit patch on a rejected rubric; a rewrite goes through the rubric sheet")
            r = patch_ops.apply(r, edits) if edits else r
            r["review"] = {"status": "approved-with-edits" if edits else r["review"]["status"],
                           "reviewers": r["review"]["reviewers"] + ([] if "EX" in r["review"]["reviewers"] else ["EX"]),
                           "notes": r["review"]["notes"] + "".join(f"; expert review (H1b): {p['edit']}" for p in mine)}
        out.append(r)
    return out


def threshold(bands: list[Mapping[str, Any]]) -> float | None:
    """The lower edge of the lowest band in which at least half of the pairs are twins."""
    for b in sorted(bands, key=lambda b: b["band"][0]):
        if b["pairs"] and 2 * b["twins"] >= b["pairs"]:
            return b["band"][0]
    return None
