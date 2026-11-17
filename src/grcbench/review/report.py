"""reports/review_summary.md: what the review protocol decided, per sheet (docs/guidelines/review_protocol.md).

Per sheet:
- the rows;
- how many the two reviewers agreed on, and how many went to the expert and why;
- the expert's audit of the agreements;
- the agreement of the two reviewers (observed and Cohen's κ, which are two runs of one model, not
  agreement between people);
- the final values;
- the removals with their reasons.

Then the declared agent runs.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from grcbench.labels.agreement import cohen_kappa, observed_agreement
from grcbench.review import decide
from grcbench.review.sheets import SHEETS

REMOVED = ("reject", "underspecified")


def _kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Cohen's kappa, None where it is undefined (every review gave one and the same value)."""
    k = cohen_kappa(a, b)
    return None if math.isnan(k) else round(k, 4)


def sheet_summary(sheet: str, recs: Sequence[Mapping[str, Any]],
                  calc_keys: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """One sheet's counts. For the blind recomputation the two reviewers are compared on their outcome
    against the key (agrees, disagrees, underspecified, no value), not on the numbers as written."""
    spec = SHEETS[sheet]
    if spec.kind == "calc":
        from grcbench.review import compare
        if calc_keys is None:
            from grcbench.review.coordinate import calc_keys as load
            calc_keys = load()
        a = [compare.calc_status(calc_keys[r["key"][0]], r["r1"]) for r in recs]
        b = [compare.calc_status(calc_keys[r["key"][0]], r["r2"]) for r in recs]
    else:
        a = [str(r["r1"].get("value", "")) for r in recs]
        b = [str(r["r2"].get("value", "")) for r in recs]
    def judged(rs: list) -> list:
        return [(r["key"], r.get("agreed", r["r1"].get("value")), decide.expert_value(spec, r["audit"])) for r in rs]

    sample = judged([r for r in recs if r.get("audit") and not r["audit"].get("full_review")])
    full = judged([r for r in recs if r.get("audit") and r["audit"].get("full_review")])
    routes = Counter(r["route"] for r in recs if r["decided_by"] == "EX")

    def why(r: Mapping[str, Any]) -> str:
        """The decider's reason: a task reviewer's reject reason is its edit line, a rubric's its notes."""
        by = (r.get("ex") or r.get("audit") or {}) if r["decided_by"].startswith("EX") else r["r1"]
        edits = [e for e in by.get("edits") or [] if not e.startswith("note:")] if spec.kind == "decision" else []
        return " ".join((edits[0] if edits else by.get("notes") or "").split())[:300]

    removed = [{"key": list(r["key"]), "final": r["final"], "decided_by": r["decided_by"], "why": why(r)}
               for r in recs if r["final"] in REMOVED and spec.kind in ("decision", "rubric")]
    from grcbench.review import replacements
    second: dict[str, Any] = {}
    if replacements.groups(sheet):       # a sheet with a second round: its reserves and each round's audit
        new = replacements.all_ids()
        second = {"reserves": sorted(replacements.pick(replacements.groups(sheet),
                                                       {r["key"][0]: r["final"] for r in recs})),
                  "audit_rounds": {"first": decide.audit_outcome([j for j in sample if j[0][0] not in new]),
                                   "second": decide.audit_outcome([j for j in sample if j[0][0] in new])}}
    return {**second,
        "sheet": sheet, "rows": len(recs),
        "agreed": sum(r["route"] == "agreement" for r in recs),
        "expert": sum(r["decided_by"] == "EX" for r in recs),
        "expert_reasons": dict(sorted(routes.items())),
        "audit": decide.audit_outcome(sample),
        "full_review": {"n": len(full), "changed": sum(a != e for _, a, e in full)} if full else None,
        "audit_changed": sum(r["decided_by"] == "EX (audit)" for r in recs),
        "observed_agreement": round(observed_agreement(a, b), 4) if recs else None,
        "kappa": _kappa(a, b) if recs else None,
        "final": dict(sorted(Counter(r["final"] for r in recs).items())),
        "removed": removed,
    }


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def render(summaries: Sequence[Mapping[str, Any]], runs: Sequence[Mapping[str, Any]],
           notes: Sequence[str] = (), h1b: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    rows = []
    for s in summaries:
        au = s["audit"]
        rows.append([s["sheet"], s["rows"], s["agreed"], s["expert"],
                     f"{au['disagreements']} of {au['n']}"
                     + (f" (full review: {s['full_review']['changed']} of {s['full_review']['n']} changed)"
                        if s.get("full_review")
                        else (" (full review: every agreed row is in the sample)" if au["n"] == s["agreed"]
                              else " (full review required, not done)") if au["full_review"] else ""),
                     s["observed_agreement"], "–" if s["kappa"] is None else s["kappa"],
                     "; ".join(f"{k} {v}" for k, v in s["final"].items())])
    lines = [
        "# Review summary",
        "",
        "Generated by `./gb review summary` from the decided records in `reviews/`; do not edit by hand. "
        "The protocol is `docs/guidelines/review_protocol.md`: two review agents per sheet, each alone; the "
        "expert agent decides every disagreement and audits a seeded sample of the agreements.",
        "",
        "**Read the agreement columns with care.** The two reviewers are two runs of one model "
        "(`claude-opus-5-5`): their agreement is two runs of one model, not agreement between people, and it "
        "overstates the agreement two people would reach (pre-registration §10).",
        "",
        _table(["sheet", "rows", "agreed", "decided by the expert", "audit: expert disagreed",
                "observed agreement", "Cohen's κ", "final values"], rows),
        "",
        "## Why rows went to the expert",
        "",
        _table(["sheet", "reasons"], [[s["sheet"], "; ".join(f"{k}: {v}" for k, v in s["expert_reasons"].items())
                                       or "–"] for s in summaries]),
        "",
        "## Removed or underspecified items (eval plan §5.3)",
        "",
    ]
    removed = [(s["sheet"], r) for s in summaries for r in s["removed"]]
    lines.append(_table(["sheet", "item", "decision", "decided by", "reason"],
                        [[sh, " / ".join(r["key"]), r["final"], r["decided_by"], r["why"]] for sh, r in removed])
                 if removed else "None.")
    second = [s for s in summaries if "audit_rounds" in s]
    if second:
        lines += ["", "## The second round: the replacement tasks", "",
                  "The tasks that replace the ones the first round rejected were reviewed in a second round, with "
                  "its own seeded audit sample (`docs/guidelines/review_protocol.md`). A reserve is an accepted "
                  "replacement beyond the slots of the tasks it replaces: kept, counted under its final value above, "
                  "and not scored.", "",
                  _table(["sheet", "audit, first round: expert disagreed", "audit, second round: expert disagreed",
                          "reserves"],
                         [[s["sheet"], f"{s['audit_rounds']['first']['disagreements']} of "
                                       f"{s['audit_rounds']['first']['n']}",
                           f"{s['audit_rounds']['second']['disagreements']} of {s['audit_rounds']['second']['n']}",
                           ", ".join(s["reserves"]) or "none"] for s in second])]
    if h1b:
        lines += ["", "## Expert review (H1b)", "",
                  "Every GNU Radio question and every Tier-F and Tier-A task, reviewed by the expert agent "
                  "(`docs/checklists/expert_review.md`). H1b holds for a tier when at least 90 % of its reviewed "
                  "items are accepted (first pass; the items the two-reviewer review removed count too); this is the "
                  "expert agent's acceptance, not a GNU Radio engineer's (pre-registration §10). H1b counts the "
                  "original pool of items; the tasks that replace the rejected ones are counted beside it. The "
                  "columns after 'H1b holds' describe; they are not the test.", "",
                  _table(["tier", "items", "accepted", "accepted with no edit", "unsure", "share", "H1b holds",
                          "items kept", "share of the kept", "replacements accepted"],
                         [[t, c["items"], c["accepted"], c["accepted_no_edit"], c["unsure"], c["share"],
                           "yes" if c["holds"] else "no", c.get("kept", "–"), c.get("share_kept", "–"),
                           f"{c['replacements_accepted']} of {c['replacements']}" if "replacements" in c else "–"]
                          for t, c in sorted(h1b.items())])]
    if notes:
        lines += ["", "## Coordinator notes (`reviews/coordinator_notes.json`)", ""] + [f"- {n}" for n in notes]
    lines += ["", "## Declared agent runs (`reviews/review_log.jsonl`)", ""]
    lines.append(_table(["agent", "role", "sheets", "rows", "model", "date", "rows written"],
                        [[f"a{e['agent']:02d}", e["role"], ", ".join(e["sheets"]), e["rows"], e["model"], e["date"],
                          "–" if e.get("rows_written") is None else e["rows_written"]] for e in runs])
                 if runs else "None.")
    return "\n".join(lines) + "\n"
