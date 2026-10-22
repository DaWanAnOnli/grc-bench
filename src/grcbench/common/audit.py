"""`./gb audit`: schema, canary, licence and disjointness checks on every benchmark item.

The automated part of the final check (remaining-work plan step 10; approved plan, Phase 6
"Automated gates"). Each check prints PASS or FAIL with a short reason, and the full result goes to
reports/audit.json; the command exits 1 if any check fails.

  schema         every item record (Tier-K questions, Tier-F and Tier-A task.json, Tier-A specs) is
                 valid against grcbench.common.schemas
  canary         every record of every items/**/*.jsonl and task.json carries the canary, and nothing
                 a model sees (Tier-K questions, instructions, scripted replies, start graphs and
                 start files) contains the canary GUID
  licence        tasks built from GNU Radio examples, and block questions, say GPL; a Tier-A task on a
                 Tier-F graph has that graph's licence identifier (the first word, e.g. GPL-3.0-or-later)
  unique ids     no id is used twice across tiers and splits
  tier-k splits  dev and test questions are disjoint; the training team's exclusion list holds every
                 dev id; every block question is about a test-side block of the block partition
  tier-f splits  every task's split agrees with its signature rule or seed file (compose.audit_items)
  tier-a splits  every Tier-A start graph comes from a Tier-F graph of the same split
  reserved list  handoff/training_team/flowgraph_reserved_signatures.txt holds the current reserved set

The read-only inputs are checked separately by `./gb baseline check`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterator

from .canary import CANARY, CANARY_GUID
from .io import iter_jsonl, read_json, write_json
from .paths import HANDOFF, ITEMS, REPO, REPORTS, SPLITS
from .schemas import BASE, validate

OUT = REPORTS / "audit.json"
ITEM_FILES_K = ("frozen_questions.jsonl", "block_questions.jsonl", "questions.jsonl")
SPEC_DRAFTS = ITEMS / "tier_a" / "test_spec_drafts.jsonl"


def _records() -> Iterator[tuple[str, dict[str, Any], str]]:
    """(path relative to the repo, record, kind) for every record under items/."""
    for path in sorted(ITEMS.rglob("*.jsonl")):
        kind = "item" if path.name in ITEM_FILES_K or path == SPEC_DRAFTS else "key"
        for rec in iter_jsonl(path):
            yield path.relative_to(REPO).as_posix(), rec, kind
    for path in sorted(ITEMS.rglob("task.json")):
        yield path.relative_to(REPO).as_posix(), read_json(path), "item"


def _check(name: str, failures: list[str], detail: str) -> dict[str, Any]:
    return {"check": name, "passed": not failures, "detail": detail, "failures": failures[:50],
            "n_failures": len(failures)}


def check_schema(records: list[tuple[str, dict, str]]) -> dict[str, Any]:
    bad, n = [], 0
    for path, rec, kind in records:
        if kind != "item":
            continue
        n += 1
        if path == SPEC_DRAFTS.relative_to(REPO).as_posix():
            import jsonschema
            errs = [e.message for e in jsonschema.Draft202012Validator(BASE).iter_errors(rec)]
        else:
            errs = validate(rec)
        bad += [f"{path} {rec.get('id')}: {e}" for e in errs]
    return _check("schema", bad, f"{n} item records")


def _visible(path: str, rec: dict) -> list[str]:
    """Texts of a record that a model under test sees."""
    out = [rec.get("question") or "", rec.get("instruction") or ""]
    out.append(json.dumps(rec.get("scripted_user") or {}))
    return out


def check_canary(records: list[tuple[str, dict, str]]) -> dict[str, Any]:
    bad = [f"{p} {r.get('id')}: canary missing or different" for p, r, _ in records if r.get("canary") != CANARY]
    bad += [f"{p} {r.get('id')}: the canary GUID is in a model-visible field" for p, r, kind in records
            if kind == "item" and any(CANARY_GUID in t for t in _visible(p, r))]
    files = 0
    for tier in ("tier_f", "tier_a"):
        for task in sorted((ITEMS / tier).glob("*/*/task.json")):
            seen = [task.parent / "start.grc"] + sorted((task.parent / "start").rglob("*"))
            for f in (f for f in seen if f.is_file()):
                files += 1
                if CANARY_GUID in f.read_text(encoding="utf-8", errors="replace"):
                    bad.append(f"{f.relative_to(REPO).as_posix()}: the canary GUID is in a start file")
    return _check("canary", bad, f"{len(records)} records; {files} start files read")


def check_licence(records: list[tuple[str, dict, str]]) -> dict[str, Any]:
    bad = []
    f_licence = {}
    for path, rec, _ in records:
        if rec.get("tier") == "F" and path.endswith("task.json"):
            f_licence[rec["id"]] = rec.get("license", "")
            if rec.get("source", {}).get("kind") == "gnuradio_example" and "GPL" not in rec.get("license", ""):
                bad.append(f"{rec['id']}: built from a GNU Radio example but the licence does not say GPL")
        if rec.get("item_kind") == "block_question" and "GPL" not in rec.get("license", ""):
            bad.append(f"{rec['id']}: block question without the GPL notice of the block YAML")
    for path, rec, _ in records:
        src = rec.get("source") or {}
        if rec.get("tier") == "A" and src.get("kind") == "tier_f_graph":
            tf = str(src.get("ref", "")).split("/")[0]
            if tf not in f_licence:
                bad.append(f"{rec['id']}: start graph {tf} is not a Tier-F task")
            elif str(rec.get("license", "")).split(" ")[0] != f_licence[tf].split(" ")[0]:
                bad.append(f"{rec['id']}: licence {rec.get('license', '').split(' ')[0]!r} differs from "
                           f"its start graph {tf} ({f_licence[tf].split(' ')[0]!r})")
    return _check("licence", bad, f"{len(f_licence)} Tier-F tasks and their Tier-A reuse")


def check_unique_ids(records: list[tuple[str, dict, str]]) -> dict[str, Any]:
    seen: dict[str, str] = {}
    bad = []
    for path, rec, kind in records:
        if kind != "item":
            continue
        rid = rec.get("id")
        if rid in seen:
            bad.append(f"{rid}: in {seen[rid]} and {path}")
        seen[rid] = path
    return _check("unique ids", bad, f"{len(seen)} ids")


def check_tier_k_splits(records: list[tuple[str, dict, str]]) -> dict[str, Any]:
    dev = set(read_json(SPLITS / "tier_k_dev.json")["dev_ids"])
    test = {r["id"] for p, r, _ in records if r.get("tier") == "K" and r.get("split") == "test"}
    dev_items = {r["id"] for p, r, _ in records if r.get("tier") == "K" and r.get("split") == "dev"}
    bad = [f"{i}: in both dev and test" for i in sorted(dev & test)]
    bad += [f"{i}: dev item not in splits/tier_k_dev.json" for i in sorted(dev_items - dev)]
    excluded = {line.strip() for line in (HANDOFF / "training_team" / "exclude_question_ids.txt")
                .read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}
    bad += [f"{i}: dev id missing from the training team's exclusion list" for i in sorted(dev - excluded)]
    split_of = {b["id"]: b.get("split") for b in read_json(SPLITS / "block_partition.json")["blocks"]}
    blockq = [r for _, r, _ in records if r.get("item_kind") == "block_question"]
    bad += [f"{r['id']}: block {r['source'].get('block_id')} is not a test-side block" for r in blockq
            if split_of.get(r["source"].get("block_id")) != "test"]
    return _check("tier-k splits", bad, f"{len(dev)} dev ids, {len(test)} test items, {len(blockq)} block questions")


def check_tier_f_splits() -> dict[str, Any]:
    from ..tier_f import compose
    audit = compose.audit_items(REPO)
    bad = [f"{r['task']}: split {r['split']} but its rule gives {r['rule_split']}" for r in audit["rows"]
           if r["conflict"]]
    cross = [g for g in audit["twin_groups"] if len(g["splits"]) > 1]
    res = _check("tier-f splits", bad, f"{len(audit['rows'])} tasks; {len(cross)} reference cells shared "
                                       "across splits (reported in reports/tier_f_first_set.md)")
    res["reserved"] = sorted(r["digest"] for r in audit["reserved"])
    return res


def check_tier_a_splits(records: list[tuple[str, dict, str]]) -> dict[str, Any]:
    bad = []
    for _, rec, _ in records:
        if rec.get("tier") != "A":
            continue
        src = rec.get("source") or {}
        start = (rec.get("start") or {}).get("tier_f_task") if isinstance(rec.get("start"), dict) else None
        ref = start or (str(src.get("ref", "")).split("/")[0] if src.get("kind") == "tier_f_graph" else None)
        if ref and not ref.startswith(f"gbf-{rec['split']}-"):
            bad.append(f"{rec['id']} ({rec['split']}): start graph {ref} is from another split")
    return _check("tier-a splits", bad, "start graphs of built tasks and specs")


def check_reserved(reserved: list[str]) -> dict[str, Any]:
    path = HANDOFF / "training_team" / "flowgraph_reserved_signatures.txt"
    listed = sorted(line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.startswith("#"))
    bad = [] if listed == reserved else [f"{path.name} lists {len(listed)} digests, the items reserve "
                                         f"{len(reserved)}: run ./gb tier-f reserved"]
    return _check("reserved list", bad, f"{len(reserved)} reserved signatures")


def run() -> list[dict[str, Any]]:
    records = list(_records())
    f_splits = check_tier_f_splits()
    reserved = f_splits.pop("reserved")
    return [check_schema(records), check_canary(records), check_licence(records), check_unique_ids(records),
            check_tier_k_splits(records), f_splits, check_tier_a_splits(records), check_reserved(reserved)]


def main(argv: list[str]) -> int:
    results = run()
    for r in results:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['check']:14s} {r['detail']}")
        for f in r["failures"][:10]:
            print(f"        {f}")
    write_json(OUT, {"passed": all(r["passed"] for r in results), "checks": results})
    print(f"wrote {OUT.relative_to(REPO).as_posix()}")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
