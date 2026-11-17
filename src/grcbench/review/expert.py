"""The expert review of eval plan §5.3 and hypothesis H1b (docs/checklists/expert_review.md).

The expert agent gives one verdict per item in scope (scope.expert_items): correct, relevant, verdict,
reason_code, comment, suggested_fix. This module checks each row against the checklist's rules, writes
handoff/review_sheets/expert_review.csv from the template's columns, and counts H1b per tier:

  an item counts as accepted when `correct` and `relevant` are both `yes` and the verdict is `accept` or
  `accept with edits`; an item marked `unsure` counts as not accepted; H1b holds for a tier when at least
  90 % of its reviewed items are accepted (pre-registration §3).
"""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from grcbench.common.paths import HANDOFF

TEMPLATE = HANDOFF / "review_sheets" / "templates" / "expert_review.csv"
SHEET = HANDOFF / "review_sheets" / "expert_review.csv"
YES_NO = ("yes", "no", "unsure")
VERDICTS = ("accept", "accept with edits", "reject")
REASONS = ("wrong_answer", "ambiguous_wording", "broken_oracle", "not_relevant", "outdated", "other")
TIER = {"gnuradio_block_question": "K", "gnuradio_frozen_question": "K", "gnuradio_dev_question": "K",
        "tier_f_task": "F", "tier_a_task": "A"}
H1B_SHARE = 0.90


def problems(row: Mapping[str, Any]) -> list[str]:
    """What the checklist forbids in one expert row (expert_review.md, "How to record your verdict")."""
    out = []
    c, r, v = row.get("correct", ""), row.get("relevant", ""), row.get("verdict", "")
    codes = [x.strip() for x in (row.get("reason_code") or "").split(";") if x.strip()]
    if c not in YES_NO or r not in YES_NO:
        out.append("correct and relevant must be yes, no or unsure")
    if "unsure" in (c, r):
        if v:
            out.append("verdict must be empty while unsure")
        if not (row.get("comment") or "").strip():
            out.append("comment required: say what information is missing")
    elif v not in VERDICTS:
        out.append("verdict must be accept, accept with edits or reject")
    if any(x not in REASONS for x in codes):
        out.append("unknown reason_code")
    if (c == "no" or r == "no" or v in ("accept with edits", "reject")) and not codes:
        out.append("reason_code required")
    if v in ("accept with edits", "reject") and not (row.get("comment") or "").strip():
        out.append("comment required")
    return out


def accepted(row: Mapping[str, Any]) -> bool:
    return row["correct"] == "yes" and row["relevant"] == "yes" and row["verdict"] in ("accept", "accept with edits")


def h1b(items: Sequence[Mapping[str, str]], rows: Mapping[tuple, Mapping[str, Any]],
        removed: frozenset[str] | set[str] = frozenset(),
        replacements: frozenset[str] | set[str] = frozenset()) -> dict[str, dict[str, Any]]:
    """H1b per tier on the first-pass rows: every item in scope counts, the ones the two-reviewer review
    removed included (the rule fixed before the expert review). `removed` gives, beside it and as a
    description only, the share over the items the review kept. The replacement items (review.replacements)
    are not in the test: H1b is counted on the original pool, and a tier with replacements reports how many
    the expert accepted beside it."""
    by_tier: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    repl: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for it in items:
        (repl[TIER[it["item_kind"]]].append(rows[(it["item_id"],)]) if it["item_id"] in replacements
         else by_tier[TIER[it["item_kind"]]].append((it["item_id"], rows[(it["item_id"],)])))
    out = {}
    for tier, pairs in sorted(by_tier.items()):
        rs = [r for _, r in pairs]
        kept = [r for i, r in pairs if i not in removed]
        n, acc = len(rs), sum(accepted(r) for r in rs)
        acc_kept = sum(accepted(r) for r in kept)
        out[tier] = {"items": n, "accepted": acc, "accepted_no_edit": sum(accepted(r) and r["verdict"] == "accept"
                                                                         for r in rs),
                     "unsure": sum("unsure" in (r["correct"], r["relevant"]) for r in rs),
                     "share": round(acc / n, 4) if n else None, "holds": bool(n) and acc / n >= H1B_SHARE,
                     "kept": len(kept), "accepted_kept": acc_kept,
                     "share_kept": round(acc_kept / len(kept), 4) if kept else None}
        if repl.get(tier):
            out[tier].update({"replacements": len(repl[tier]),
                              "replacements_accepted": sum(accepted(r) for r in repl[tier])})
    return out


def first_pass(work: Path) -> tuple[dict[tuple, dict[str, Any]], dict[tuple, list[dict[str, Any]]]]:
    """(the first expert_review row per item, the later rows per item): EX files in name order, lines in
    order. H1b counts the first pass; a later row is a re-check, kept but not counted."""
    from grcbench.review import coordinate
    first: dict[tuple, dict[str, Any]] = {}
    later: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for path in coordinate.role_files(work).get("EX", []):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("sheet") != "expert_review":
                continue
            k = (str(r["key"]["item_id"]),)
            if k in first:
                later[k].append(r)
            else:
                first[k] = r
    return first, dict(later)


def sheet_text(items: Sequence[Mapping[str, str]], rows: Mapping[tuple, Mapping[str, Any]], date: str,
               dates: Mapping[str, str] | None = None) -> str:
    """The filled sheet; `dates` keeps an earlier round's date for its items."""
    header = next(csv.reader(io.StringIO(TEMPLATE.read_text(encoding="utf-8-sig"))))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
    w.writeheader()
    for it in items:
        r = rows.get((it["item_id"],))
        if r is None:
            raise ValueError(f"{it['item_id']}: no expert row")
        w.writerow({**it, **{k: r.get(k, "") for k in ("correct", "relevant", "verdict", "reason_code", "comment",
                                                        "suggested_fix")},
                    "minutes": "", "expert": "EX", "date": (dates or {}).get(it["item_id"]) or date,
                    "team_action": r.get("team_action", ""), "team_action_by": r.get("team_action_by", "")})
    return buf.getvalue()


def write(work: Path, items: Sequence[Mapping[str, str]], date: str, reviews_dir: Path | None = None,
          sheet: Path | None = None, removed: frozenset[str] | set[str] = frozenset(), merge: bool = False,
          replacements: frozenset[str] | set[str] = frozenset()) -> dict[str, dict[str, Any]]:
    """The expert's rows (WORK/out/*-EX-*.jsonl, sheet `expert_review`) for every item in scope, checked
    against the checklist's rules: reviews/expert_review.jsonl (the records, in scope order),
    reviews/expert_review_h1b.json (H1b per tier) and the filled sheet. Returns the H1b counts. With `merge` (a
    second round: the replacement items), WORK holds only the new items; the earlier records give the rest,
    with their own dates, and an item reviewed before is refused."""
    from grcbench.common.paths import REPO
    from grcbench.review import records
    rows, later = first_pass(work)
    out = Path(reviews_dir or REPO / "reviews")
    dates: dict[str, str] = {}
    lines: dict[str, str] = {}            # an earlier round's records, written again exactly as they were
    if merge:
        keep_ = ("correct", "relevant", "verdict", "reason_code", "comment", "suggested_fix", "notes", "evidence")
        earlier = [json.loads(x) for x in (out / "expert_review.jsonl").read_text(encoding="utf-8").splitlines()
                   if x.strip()]
        again = sorted({r["item_id"] for r in earlier} & {k[0] for k in rows})
        if again:
            raise ValueError(f"{len(again)} item(s) already reviewed in an earlier round, e.g. {again[:3]}")
        gone = sorted({r["item_id"] for r in earlier} - {it["item_id"] for it in items})
        if gone:
            raise ValueError(f"{len(gone)} earlier record(s) no longer in scope, e.g. {gone[:3]}")
        old_sheet = Path(sheet or SHEET)
        team: dict[str, dict[str, str]] = {}
        if old_sheet.exists():
            old_rows = list(csv.DictReader(io.StringIO(old_sheet.read_text(encoding="utf-8-sig"))))
            dates = {r["item_id"]: r.get("date", "") for r in old_rows}
            team = {r["item_id"]: {k: r.get(k) or "" for k in ("team_action", "team_action_by")} for r in old_rows}
        for r in earlier:
            rows[(r["item_id"],)] = {**{k: r.get(k, "") for k in keep_}, **team.get(r["item_id"], {})}
            later[(r["item_id"],)] = list(r.get("rechecks") or [])
        lines = {json.loads(x)["item_id"]: x for x in (out / "expert_review.jsonl").read_text(encoding="utf-8")
                 .splitlines() if x.strip()}
    missing = [it["item_id"] for it in items if (it["item_id"],) not in rows]
    if missing:
        raise ValueError(f"{len(missing)} item(s) with no expert row, e.g. {missing[:3]}")
    bad = {it["item_id"]: found for it in items if (found := problems(rows[(it["item_id"],)]))}
    if bad:
        raise ValueError(f"expert rows the checklist refuses: {bad}")
    out.mkdir(parents=True, exist_ok=True)
    keep = ("correct", "relevant", "verdict", "reason_code", "comment", "suggested_fix", "notes", "evidence")
    (out / "expert_review.jsonl").write_text("".join(
        (lines[it["item_id"]] if it["item_id"] in lines else
         json.dumps(records.redact({**dict(it), **{k: rows[(it["item_id"],)].get(k, "") for k in keep},
                                    "removed_by_review": it["item_id"] in removed,
                                    "replacement": it["item_id"] in replacements,
                                    "rechecks": [{k: r.get(k, "") for k in keep}
                                                 for r in later.get((it["item_id"],), [])]}),
                    ensure_ascii=False, sort_keys=True)) + "\n" for it in items), encoding="utf-8")
    counts = h1b(items, rows, removed, replacements)
    (out / "expert_review_h1b.json").write_text(json.dumps(counts, indent=1, sort_keys=True) + "\n",
                                                encoding="utf-8")
    Path(sheet or SHEET).write_text(records.redact(sheet_text(items, rows, date, dates)), encoding="utf-8")
    return counts
