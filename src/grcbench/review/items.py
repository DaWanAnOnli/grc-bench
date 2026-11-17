"""The review protocol's decisions on Tier-F and Tier-A tasks, applied where a task is written.

The task writers (tier_f.taskio.save for every Tier-F task; tier_a.first_set.record and write for Tier A)
call apply(): the decided record (reviews/<sheet>.jsonl) sets the task's `review` field (its status and
reviewers, and a note of who decided, followed by the task's own notes as the decided edits leave them), and the decided
edits, written as patches (reviews/<sheet>_patches.json, see review.patches), change task.json
(file "task.json") or another file of the task (file "<name>", a text patch on path ["text"]). A task
whose decided edits are applied gets version 2. A patch marked `post_review` is an edit made after the
task's review (pre-registration §10): it, too, makes an approved task approved-with-edits, and the notes
name it as a post-review edit. A rejected task is kept, with status `rejected`, and
active() leaves it out of every benchmark run (eval plan §5.3: removed and counted). An accepted replacement
beyond the slots of the tasks it replaces (review.replacements) has status `reserve`: kept, and not run either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from grcbench.common.paths import REPO
from grcbench.review import patches
from grcbench.review.records import decided

STATUS = {"approve": "approved", "approve with edits": "approved-with-edits", "reject": "rejected"}
NOT_RUN = ("rejected", "reserve")
RESERVE_NOTE = ("an accepted replacement beyond the slots of the tasks it replaces: kept and not scored "
                "(pre-registration §10)")


def _dir(reviews_dir: Path | None) -> Path:
    return Path(reviews_dir) if reviews_dir else REPO / "reviews"


def review_field(sheet: str, item_id: str, reviews_dir: Path | None = None) -> dict | None:
    """The `review` field from the decided record (None while the item is undecided)."""
    rec = decided(sheet, reviews_dir=_dir(reviews_dir)).get((item_id,))
    if rec is None:
        return None
    by = rec["decided_by"]
    notes = f"decided by {by}" + (f" ({rec['route']})" if by == "EX" else "")
    if rec["final"] == "approve with edits":
        notes += "; decided edits applied"
    status = STATUS[rec["final"]]
    if status != "rejected":
        from grcbench.review import replacements
        if item_id in replacements.reserve(sheet, reviews_dir=_dir(reviews_dir)):
            status, notes = "reserve", f"{notes}; {RESERVE_NOTE}"
    return {"status": status, "reviewers": ["R1", "R2"] + (["EX"] if by.startswith("EX") else []), "notes": notes}


def review_after(sheet: str, item_id: str, applied: list[Mapping[str, Any]],
                 reviews_dir: Path | None = None) -> dict | None:
    """The `review` field from the decided record once the patches `applied` to the task are written (before the
    task's own notes): an approved task that gets an expert-review patch, or a post-review patch, is
    approved-with-edits, and the notes name each such edit once. None while the item is undecided."""
    review = review_field(sheet, item_id, reviews_dir)
    if review is None:
        return None
    post = _distinct([p for p in applied if p.get("post_review")])           # edits made after the review (§10)
    expert_edits = _distinct([p for p in applied if not p.get("post_review")])
    if expert_edits and review["status"] == "approved":
        # an approved task that gets a patch (an expert-review edit) is a changed task
        review = {"status": "approved-with-edits",
                  "reviewers": review["reviewers"] + ([] if "EX" in review["reviewers"] else ["EX"]),
                  "notes": review["notes"] + "".join(f"; expert review (H1b) edit: {p.get('edit', '')}"
                                                     for p in expert_edits)}
    if post:
        review = {**review, "status": "approved-with-edits" if review["status"] == "approved" else review["status"],
                  "notes": review["notes"] + "".join(f"; post-review edit ({p['post_review']}): {p.get('edit', '')}"
                                                     for p in post)}
    return review


def apply(sheet: str, item_id: str, record: Mapping[str, Any], files: Mapping[str, str],
          reviews_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    patch_file = _dir(reviews_dir) / f"{sheet}_patches.json"
    import json
    every = json.loads(patch_file.read_text(encoding="utf-8")) if patch_file.exists() else []
    unknown = sorted({p.get("file", "task.json") for p in every if p["item"] == item_id}
                     - {"task.json", "spec", *files})
    if unknown:
        raise ValueError(f"{sheet}: patches for {item_id} name files this writer does not write: {unknown}")
    rec = patches.apply(record, patches.for_item(patch_file, item_id, "task.json"))
    out_files = dict(files)
    for name in files:
        ps = patches.for_item(patch_file, item_id, name)
        if ps:
            out_files[name] = patches.apply({"text": files[name]}, ps)["text"]
    applied = patches.for_item(patch_file, item_id, "task.json") + [
        p for name in files for p in patches.for_item(patch_file, item_id, name)]
    review = review_after(sheet, item_id, applied, reviews_dir)
    if review is not None:
        own = " ".join(str((rec.get("review") or {}).get("notes") or "").split())   # the task's own notes, patched
        if own and not own.startswith("decided by"):
            review["notes"] += f". {own}"
        rec["review"] = review
        # version: 1 as drafted, +1 for the review's edits (decided, or an expert-review patch on an approved task;
        # a reserve, too), +1 for edits after the review (post_review patches)
        final = (decided(sheet, reviews_dir=_dir(reviews_dir)).get((item_id,)) or {}).get("final")
        expert = any(not p.get("post_review") for p in applied)
        version = (1 + (final == "approve with edits" or (final == "approve" and expert))
                   + any(p.get("post_review") for p in applied))
        if version > 1:
            rec["version"] = max(version, int(rec.get("version") or 1))
    return rec, out_files


def _distinct(ps: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """One patch per decided edit: an edit applied as several patches is named once in the notes."""
    seen: set[str] = set()
    out = []
    for p in ps:
        if p.get("edit", "") not in seen:
            seen.add(p.get("edit", ""))
            out.append(p)
    return out


def refuse_rejected(records: Iterable[Mapping[str, Any]]) -> None:
    """A benchmark run never uses a task the review rejected or keeps as a reserve (eval plan §5.3), nor a
    replacement whose group the review has not fully decided (review.replacements)."""
    from grcbench.review import replacements
    from grcbench.runlog.log import RunRefused
    records = list(records)
    pending, unsettled = replacements.all_ids(), replacements.unsettled_ids()
    bad = [r.get("id", "?") for r in records if (r.get("review") or {}).get("status") in NOT_RUN
           or r.get("id") in unsettled
           or (r.get("id") in pending and (r.get("review") or {}).get("status") == "pending-team-review")]
    if bad:
        raise RunRefused(f"tasks {bad} were rejected or kept as a reserve by the review (reviews/), or are replacements "
                         f"the review has not settled, and are not part of the benchmark")


def active_dirs(dirs: Iterable[Path]) -> list[Path]:
    """The task folders a split-wide run uses: every one but those the review rejected or keeps as a reserve,
    and a replacement the review has not decided yet (it is not part of the benchmark until it is accepted)."""
    import json

    from grcbench.review import replacements
    pending, unsettled = replacements.all_ids(), replacements.unsettled_ids()
    out = []
    for d in dirs:
        t = json.loads((Path(d) / "task.json").read_text(encoding="utf-8"))
        status, tid = (t.get("review") or {}).get("status"), t.get("id", Path(d).name)
        if status in NOT_RUN or tid in unsettled or (status == "pending-team-review" and tid in pending):
            continue
        out.append(Path(d))
    return out


def active(sheet: str, item_ids: Iterable[str], reviews_dir: Path | None = None) -> list[str]:
    """The items a benchmark run uses: every item but the rejected ones, the reserves and the replacements of a
    group the review has not fully decided."""
    from grcbench.review import replacements
    gone = {k[0] for k, r in decided(sheet, reviews_dir=_dir(reviews_dir)).items() if r["final"] == "reject"}
    gone |= replacements.reserve(sheet, reviews_dir=_dir(reviews_dir))
    gone |= replacements.unsettled(sheet, reviews_dir=_dir(reviews_dir))
    return [i for i in item_ids if i not in gone]
