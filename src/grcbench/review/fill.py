"""The filled sheets: decided records written into each sheet's own columns.

The decided records (reviews/<sheet>.jsonl) are the full record; a filled sheet shows, per row, what
its guide asks for: the reviewers' values where the sheet has columns for them, the final value, who
decided, and the notes.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Callable, Mapping

from grcbench.review.inputs import SHEET_DIR
from grcbench.review.sheets import SHEETS, key_of

Rec = Mapping[str, Any]


def _who(rec: Rec) -> str:
    spec = SHEETS[rec["sheet"]]
    base = ";".join(spec.roles)
    return base + ";EX" if rec["decided_by"].startswith("EX") else base


def _lines(out: Mapping[str, Any] | None) -> str:
    if not out:
        return ""
    lines = list(out.get("edits") or [])
    notes = (out.get("notes") or "").strip()
    if notes:
        lines += [n if n.startswith("note:") else f"note: {n}" for n in notes.splitlines() if n.strip()]
    return "\n".join(lines)


def _ex_line(rec: Rec) -> str:
    ex = rec.get("ex") or rec.get("audit")
    if not rec["decided_by"].startswith("EX"):
        return ""
    reason = " ".join((ex or {}).get("notes", "").split())
    lines = [f"note: decided by EX ({rec['route'] if rec['decided_by'] == 'EX' else 'audit'})"
             + (f" -- {reason}" if reason else "")]
    lines += [f"note: decided edit -- {e}" for e in (ex or {}).get("edits") or []]
    return "\n".join(lines)


def _summary(rec: Rec) -> str:
    spec = SHEETS[rec["sheet"]]
    parts = []
    for role, out in zip(spec.roles, (rec["r1"], rec["r2"])):
        note = " ".join((out.get("notes") or "").split())
        parts.append(f"{role}: {out.get('value', '')}" + (f" -- {note}" if note else ""))
    if rec["decided_by"].startswith("EX"):
        ex = rec.get("ex") or rec.get("audit") or {}
        note = " ".join((ex.get("notes") or "").split())
        parts.append(f"EX: {rec['final']}" + (f" -- {note}" if note else ""))
    return "; ".join(parts)


def _decision_sheet(row: dict, rec: Rec) -> None:
    for n, out in ((1, rec["r1"]), (2, rec["r2"])):
        row[f"reviewer_{n}_decision"] = out.get("value", "")
        row[f"reviewer_{n}_edits"] = _lines(out)
    extra = _ex_line(rec)
    if extra:
        row["reviewer_1_edits"] = (row["reviewer_1_edits"] + "\n" + extra).strip()
    row["final_decision"] = rec["final"]


def _block(row: dict, rec: Rec) -> None:
    for n, out in ((1, rec["r1"]), (2, rec["r2"])):
        row[f"reviewer_{n}_wording"] = out.get("value", "")
        row[f"reviewer_{n}_note"] = _lines(out)
    extra = _ex_line(rec)
    if extra:
        row["reviewer_1_note"] = (row["reviewer_1_note"] + "\n" + extra).strip()
    row["final_decision"] = rec["final"]


def _single(column: str) -> Callable[[dict, Rec], None]:
    def f(row: dict, rec: Rec) -> None:
        row[column] = rec["final"]
        row["reviewer"] = _who(rec)
        row["notes"] = _summary(rec)
    return f


def _task_type_test(row: dict, rec: Rec) -> None:
    row["annotator_1"], row["annotator_2"] = rec["r1"].get("value", ""), rec["r2"].get("value", "")
    row["notes"] = f"final: {rec['final']} ({rec['decided_by']}); " + _summary(rec)


def _task_type_train(row: dict, rec: Rec) -> None:
    row["reviewer_label"] = rec["final"]
    row["notes"] = f"decided by {rec['decided_by']}; " + _summary(rec)


def _calc(row: dict, rec: Rec) -> None:
    row["annotator"] = _who(rec)
    ex = rec.get("ex") or {}
    if rec["final"] == "agrees":
        row["value"], row["unit"] = rec["r1"].get("value", ""), rec["r1"].get("unit", "")
    elif rec["final"] == "disagrees":
        row["value"], row["unit"] = ex.get("confirmed_value", ""), ex.get("confirmed_unit", "")
    else:
        row["value"], row["unit"] = "", ""
    row["notes"] = (f"final: {rec['final']} ({rec['decided_by']}); R1: {rec['r1'].get('value', '')} "
                    f"{rec['r1'].get('unit', '')}; R2: {rec['r2'].get('value', '')} {rec['r2'].get('unit', '')}"
                    + (f"; EX: {' '.join((ex.get('notes') or '').split())}" if ex else ""))


def _rubric(row: dict, rec: Rec) -> None:
    """The final rubric fields (the writer's, or the expert's where it decided edits; the writer's as
    written for a rejected question), both checks, and who decided (`checkers` or the expert)."""
    import json

    from grcbench.review.apply import final_fields
    src = final_fields(rec) or rec.get("writer") or {}

    def text(v: Any) -> str:
        return "" if v is None else v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)

    for col in ("required_points", "disqualifying_errors", "sources"):
        row[col] = "\n".join(src.get(col) or [])
    row["numeric"], row["numeric_tolerance"] = text(src.get("numeric")), text(src.get("numeric_tolerance"))
    if "calc_status" in rec:
        row["calc_status"] = rec["calc_status"]
    row["writer"] = "W"
    for n, out in ((1, rec["r1"]), (2, rec["r2"])):
        row[f"checker_{n}_initials"] = f"C{n}"
        row[f"checker_{n}_verdict"] = out.get("value", "")
        row[f"checker_{n}_source"] = out.get("source", "")
        row[f"checker_{n}_note"] = _lines(out)
    extra = "\n".join([_ex_line(rec)] + [f"note: expert review (H1b) -- {e}" for e in rec.get("h1b_edits") or []])
    if extra.strip():
        row["checker_1_note"] = (row["checker_1_note"] + "\n" + extra).strip()
    row["final_decision"] = rec["final"]
    row["decided_by"] = rec["decided_by"] if rec["decided_by"].startswith("EX") else "checkers"


FILLERS: dict[str, Callable[[dict, Rec], None]] = {
    "tier_f": _decision_sheet, "tier_a": _decision_sheet, "block_questions": _block,
    "clusters": _single("team_decision (same family / split out)"),
    "twins": _single("team_label (twin / not twin)"),
    "task_type_test": _task_type_test, "task_type_train": _task_type_train, "calc": _calc,
    **{f"contamination_{t}": _single("decision") for t in ("tier_k_test", "tier_k_dev", "tier_f_test", "tier_f_dev")},
    "rubric_test": _rubric, "rubric_dev": _rubric,
}


def filled(sheet: str, recs: list[Rec], require_all: bool = True) -> str:
    raw = (SHEET_DIR / SHEETS[sheet].file).read_text(encoding="utf-8")
    return _fill(sheet, raw, recs, require_all)


def filled_text(sheet: str, raw: str, reviews_dir=None) -> str:
    """A generator's sheet text with the decided columns filled from reviews/<sheet>.jsonl (unchanged
    while the sheet has no decided records). Generators call this so a re-run keeps the decisions."""
    from grcbench.review.records import decided
    recs = list(decided(sheet, reviews_dir=reviews_dir).values())
    return _fill(sheet, raw, recs, require_all=False) if recs else raw


def with_new_rows(sheet: str, generated: str, reviews_dir=None, path=None) -> str:
    """A generator's sheet text once the sheet has decided records: the current sheet file, its reviewed rows
    exactly as they were reviewed, with the generator's rows for new keys appended (a second round: the
    replacement items). Before any decision, the generated text itself."""
    from grcbench.review.records import decided, redact
    spec = SHEETS[sheet]
    path = path or SHEET_DIR / spec.file
    if not decided(sheet, reviews_dir=reviews_dir) or not path.exists():
        return generated
    current = path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(current.lstrip("﻿")))
    have = {key_of(spec, r) for r in reader}
    new = [r for r in csv.DictReader(io.StringIO(generated.lstrip("﻿"))) if key_of(spec, r) not in have]
    buf = io.StringIO()
    csv.DictWriter(buf, fieldnames=reader.fieldnames or [], lineterminator="\n", extrasaction="ignore").writerows(new)
    return (current if current.endswith("\n") else current + "\n") + redact(buf.getvalue())


def _fill(sheet: str, raw: str, recs: list[Rec], require_all: bool) -> str:
    spec = SHEETS[sheet]
    bom = raw.startswith("﻿")
    reader = csv.DictReader(io.StringIO(raw.lstrip("﻿")))
    header, rows = list(reader.fieldnames or []), list(reader)
    by_key = {tuple(r["key"]): r for r in recs}
    missing = [key_of(spec, r) for r in rows if key_of(spec, r) not in by_key]
    if require_all and missing:
        raise ValueError(f"{sheet}: {len(missing)} row(s) with no decision, e.g. {missing[:3]}")
    for r in rows:
        rec = by_key.get(key_of(spec, r))
        if rec is not None:
            FILLERS[sheet](r, rec)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    from grcbench.review.records import redact
    return ("﻿" if bom else "") + redact(buf.getvalue())
