"""The automatic checks on a written rubric (docs/guidelines/rubric_writing.md, "Checking a rubric"; review
protocol, "Evidence"). A rubric that fails one goes to the expert agent with the failure, whatever the
checkers say (coordinate.rubric_router).

  - the lines parse into the judge's rubric record, which judge_prompt.normalize_rubric and the rule
    cascade (rules.parse_reference) accept;
  - every `<path> :: "<quote>"` source line is found in the file it names;
  - a calculation question's `numeric` gives full credit to the confirmed value, and a question the blind
    recomputation found underspecified is marked underspecified;
  - no point or error shares a word 8-gram with a teacher answer to the question.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from grcbench.review import validate


def record(w: Mapping[str, Any]) -> dict[str, Any]:
    """The judge's fields of a writer row (raises ValueError on a line that is not `P<n>: ...`)."""
    return {"required_points": validate.parse_lines(w.get("required_points") or [], "P"),
            "disqualifying_errors": validate.parse_lines(w.get("disqualifying_errors") or [], "D"),
            "numeric": w.get("numeric"), "numeric_tolerance": w.get("numeric_tolerance"),
            "sources": list(w.get("sources") or []), "review": {"status": "approved"}}


def _parts(x: Any) -> list[Any]:
    return list(x) if isinstance(x, (list, tuple)) else [x]


def numeric_problem(numeric: Any, confirmed: Mapping[str, Any] | None) -> str | None:
    """Whether a rubric's `numeric` gives full credit to the blind recomputation's confirmed value(s)."""
    if confirmed is None or confirmed.get("confirmed") is None:
        return None
    c = confirmed["confirmed"]
    values, units = _parts(c["value"]), _parts(c["unit"])
    units = units * len(values) if len(units) == 1 else units
    want = [f"{v} {u}".strip() for v, u in zip(values, units)]
    if numeric is None or not validate.numeric_matches(numeric, want):
        return f"numeric does not give full credit to the confirmed value(s) {want}"
    return None


def problems(w: Mapping[str, Any], confirmed: Mapping[str, Any] | None, teacher: Sequence[str]) -> list[str]:
    out: list[str] = []
    under = w.get("value") == "underspecified"
    if confirmed is not None and confirmed.get("outcome") == "underspecified" and not under:
        out.append("the blind recomputation found the question underspecified, but the rubric is written")
    if under:
        return out
    try:
        rec = record(w)
    except ValueError as e:
        return out + [str(e)]
    out += validate.rubric_problems(rec)
    out += validate.quote_problems(rec["sources"])
    found = numeric_problem(rec["numeric"], confirmed)
    if found:
        out.append(found)
    lines = list(w.get("required_points") or []) + list(w.get("disqualifying_errors") or [])
    shared = validate.shared_8grams(lines, teacher)
    if shared:
        out.append(f"{shared} word 8-gram(s) shared with a teacher answer")
    return out


def final_problems(rec: Mapping[str, Any], confirmed: Mapping[str, Any] | None,
                   teacher: Sequence[str]) -> list[str]:
    """The automatic checks on a decided question's final rubric (the writer's, or the expert's merged
    one): a rubric the judge will read passes the same checks as a written one. None for a rejected one."""
    from grcbench.review.apply import final_fields
    src = final_fields(rec)
    return [] if src is None else problems({**src, "value": "written"}, confirmed, teacher)


def teacher_answers(question_ids: Iterable[str], answers_dir: Path) -> dict[str, list[str]]:
    """Teacher answers and rationales by question id (05_answers/<teacher>/answers.jsonl; read-only)."""
    want, out = set(question_ids), {}
    for path in sorted(answers_dir.glob("*/answers.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("question_id") in want:
                text = "\n\n".join(t for t in ((r.get("answer") or "").strip(), (r.get("rationale") or "").strip()) if t)
                if text:
                    out.setdefault(r["question_id"], []).append(text)
    return out
