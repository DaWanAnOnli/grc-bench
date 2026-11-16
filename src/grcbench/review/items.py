"""The review protocol's decisions on Tier-F and Tier-A tasks, applied where a task is written.

The task writers (tier_f.taskio.save for every Tier-F task; tier_a.first_set.record and write for Tier A)
call apply(): the decided record (reviews/<sheet>.jsonl) sets the task's `review` field (its status and
reviewers, and a note of who decided, followed by the task's own notes as the decided edits leave them), and the decided
edits, written as patches (reviews/<sheet>_patches.json, see review.patches), change task.json
(file "task.json") or another file of the task (file "<name>", a text patch on path ["text"]). A task
whose decided edits are applied gets version 2. A rejected task is kept, with status `rejected`, and
active() leaves it out of every benchmark run (eval plan §5.3: removed and counted).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from grcbench.common.paths import REPO
from grcbench.review import patches
from grcbench.review.records import decided

STATUS = {"approve": "approved", "approve with edits": "approved-with-edits", "reject": "rejected"}


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
    return {"status": STATUS[rec["final"]], "reviewers": ["R1", "R2"] + (["EX"] if by.startswith("EX") else []),
            "notes": notes}


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
    review = review_field(sheet, item_id, reviews_dir)
    applied = patches.for_item(patch_file, item_id, "task.json") + [
        p for name in files for p in patches.for_item(patch_file, item_id, name)]
    if review is not None and applied and review["status"] == "approved":
        # an approved task that gets a patch (an expert-review edit) is a changed task
        review = {"status": "approved-with-edits",
                  "reviewers": review["reviewers"] + ([] if "EX" in review["reviewers"] else ["EX"]),
                  "notes": review["notes"] + "".join(f"; expert review (H1b) edit: {p.get('edit', '')}"
                                                     for p in applied)}
    if review is not None:
        own = " ".join(str((rec.get("review") or {}).get("notes") or "").split())   # the task's own notes, patched
        if own and not own.startswith("decided by"):
            review["notes"] += f". {own}"
        rec["review"] = review
        if review["status"] == "approved-with-edits":
            rec["version"] = max(2, int(rec.get("version") or 1))
    return rec, out_files


def refuse_rejected(records: Iterable[Mapping[str, Any]]) -> None:
    """A benchmark run never uses a task the review rejected (eval plan §5.3: removed and counted)."""
    from grcbench.runlog.log import RunRefused
    bad = [r.get("id", "?") for r in records if (r.get("review") or {}).get("status") == "rejected"]
    if bad:
        raise RunRefused(f"tasks {bad} were rejected by the review (reviews/) and are not part of the benchmark")


def active_dirs(dirs: Iterable[Path]) -> list[Path]:
    """The task folders a split-wide run uses: every one but those whose task.json the review rejected."""
    import json
    return [Path(d) for d in dirs
            if (json.loads((Path(d) / "task.json").read_text(encoding="utf-8")).get("review") or {}).get("status")
            != "rejected"]


def active(sheet: str, item_ids: Iterable[str], reviews_dir: Path | None = None) -> list[str]:
    """The items a benchmark run uses: every item but the rejected ones."""
    rejected = {k[0] for k, r in decided(sheet, reviews_dir=_dir(reviews_dir)).items() if r["final"] == "reject"}
    return [i for i in item_ids if i not in rejected]
