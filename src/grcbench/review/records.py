"""Agent outputs in, decided records out.

An agent writes one JSON object per row to its output file, as soon as the row is done:
  {"sheet": <spec name>, "key": {<key column>: <value>, ...}, "value": <allowed value>,
   "edits": [<edit lines>], "notes": "...", "evidence": [<file :: "quote"> or <command and result>], ...}
Sheet-specific fields (a recomputed `unit`, a `blind_value`, a rubric writer's fields) ride along.
The last row for a key wins, so an agent may correct itself by writing the row again.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from grcbench.common.privacy import redact          # the decided records are published
from grcbench.review import decide
from grcbench.review.sheets import SHEETS, Sheet, key_of

Key = tuple[str, ...]
Outputs = Mapping[Key, Mapping[str, Any]]


def load_outputs(paths: Iterable[Path]) -> dict[str, dict[Key, dict[str, Any]]]:
    got: dict[str, dict[Key, dict[str, Any]]] = defaultdict(dict)
    for path in paths:
        for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("sheet") == "expert_review":          # the expert review (H1b): one row per item
                if not isinstance(row.get("key"), dict) or "item_id" not in row["key"]:
                    raise ValueError(f"{path}:{n}: expert_review row without key.item_id")
                got["expert_review"][(str(row["key"]["item_id"]),)] = row
                continue
            spec = SHEETS.get(row.get("sheet"))
            if spec is None:
                raise ValueError(f"{path}:{n}: unknown sheet {row.get('sheet')!r}")
            if not isinstance(row.get("key"), dict):
                raise ValueError(f"{path}:{n}: no key mapping")
            try:
                k = key_of(spec, row["key"])
            except ValueError as e:
                raise ValueError(f"{path}:{n}: {e}") from None
            got[spec.name][k] = row
    return dict(got)


def check_groups(spec: Sheet, rows: list[Mapping[str, Any]], outputs: Outputs) -> None:
    """Rows sharing spec.group (a test cluster) must carry one decision."""
    if not spec.group:
        return
    groups: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        k = key_of(spec, r)
        if k in outputs:
            groups[str(r[spec.group])].add((outputs[k].get("value") or "").strip())
    split = {g: v for g, v in groups.items() if len(v) > 1}
    if split:
        raise ValueError(f"{spec.name}: cluster(s) with different decisions on their rows: {split}")


Router = Callable[[Key, Mapping[str, Any], Mapping[str, Any]], decide.Route]


def build(spec: Sheet, r1: Outputs, r2: Outputs, ex: Outputs, audit: Outputs | None = None,
          router: Router | None = None) -> list[dict[str, Any]]:
    """One decided record per row; refuses a missing reviewer row or a missing expert decision."""
    audit = audit or {}
    recs, to_expert = [], set()
    for k in sorted(set(r1) | set(r2)):
        for role, outs in zip(spec.roles, (r1, r2)):
            if k not in outs:
                raise ValueError(f"{spec.name}: {role} has no row {k}")
        rt = router(k, r1[k], r2[k]) if router else decide.route(spec, r1[k], r2[k])
        rec: dict[str, Any] = {"sheet": spec.name, "key": k, "r1": dict(r1[k]), "r2": dict(r2[k]),
                               "route": rt.reason, "ex": None, "audit": None}
        if rt.to == "expert":
            to_expert.add(k)
            if k not in ex:
                raise ValueError(f"{spec.name}: row {k} needs the expert's decision ({rt.reason})")
            rec["ex"] = dict(ex[k])
            rec["final"], rec["decided_by"] = decide.expert_value(spec, ex[k]), "EX"
        else:
            rec["agreed"] = rt.value             # the rule's value; the audit is compared with it
            rec["final"], rec["decided_by"] = rt.value, "R1;R2" if spec.roles == ("R1", "R2") else ";".join(spec.roles)
            if k in audit:
                rec["audit"] = dict(audit[k])
                a = decide.expert_value(spec, audit[k])
                if a != rt.value:
                    rec["final"], rec["decided_by"] = a, "EX (audit)"
        recs.append(rec)
    stray = sorted(set(ex) - to_expert)
    if stray:                            # an expert row the rules did not ask for is never dropped silently
        raise ValueError(f"{spec.name}: expert decision rows for rows not sent to the expert (an audit row needs "
                         f'"pass": "audit"): {stray[:5]}')
    return recs


def write_records(path: Path, recs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(redact({**r, "key": list(r["key"])}), ensure_ascii=False, sort_keys=True)
                            + "\n" for r in recs), encoding="utf-8")


def read_records(path: Path) -> list[dict[str, Any]]:
    return [{**json.loads(line), "key": tuple(json.loads(line)["key"])}
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def decided(sheet: str, reviews_dir: Path | None = None) -> dict[Key, dict[str, Any]]:
    """The decided records of one sheet by row key ({} before the sheet is decided). Generators read
    this to apply the review's decisions to the files they write."""
    from grcbench.common.paths import REPO
    path = Path(reviews_dir or REPO / "reviews") / f"{sheet}.jsonl"
    return {r["key"]: r for r in read_records(path)} if path.exists() else {}
