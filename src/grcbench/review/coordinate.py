"""The coordinator's steps between the agent runs (docs/guidelines/review_protocol.md).

Agent output files are WORK/out/<agent id>-<role>-<what>.jsonl; the role in the name (R1, R2, C1, C2,
W, EX) says which column of the decision rules the rows fill.

  route(WORK, sheet)   applies the decision rules to the two reviews and writes
                       WORK/expert/<sheet>.queue.jsonl  the rows the expert decides, with both reviews
                       WORK/expert/<sheet>.audit.jsonl  the seeded audit sample of agreed rows (keys only)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from grcbench.common.io import read_jsonl
from grcbench.common.paths import ITEMS
from grcbench.review import compare, decide, inputs, records
from grcbench.review.sheets import SHEETS

FILE = re.compile(r"^(a\d+)-(R1|R2|C1|C2|W|EX)(?:-.*)?\.jsonl$")
FIRST, SECOND = {"R1", "C1"}, {"R2", "C2"}


def role_files(work: Path) -> dict[str, list[Path]]:
    by_role: dict[str, list[Path]] = {}
    for p in sorted((Path(work) / "out").glob("*.jsonl")):
        m = FILE.match(p.name)
        if m:
            by_role.setdefault(m.group(2), []).append(p)
    return by_role


def check_independence(work: Path, sheet: str) -> None:
    """The protocol's roles: R1 and R2 (C1 and C2) of a sheet are different agents, and the expert is
    neither (review_protocol.md, "Roles")."""
    agents: dict[str, set[str]] = {}
    for role, paths in role_files(work).items():
        for p in paths:
            rows = records.load_outputs([p]).get(sheet, {})
            if rows:
                agents.setdefault(role, set()).add(FILE.match(p.name).group(1))
    first = agents.get("R1", set()) | agents.get("C1", set())
    second = agents.get("R2", set()) | agents.get("C2", set())
    both = first & second
    expert = agents.get("EX", set()) & (first | second)
    if both or expert:
        raise ValueError(f"{sheet}: an agent holds two roles: {sorted(both | expert)}")


def collect(work: Path, sheet: str, roles: tuple[set[str], set[str]] = (FIRST, SECOND)):
    """(first reviewer's rows, second reviewer's rows) of one sheet, by row key."""
    files = role_files(work)
    out = []
    for wanted in roles:
        paths = [p for r in sorted(wanted) for p in files.get(r, [])]
        out.append(records.load_outputs(paths).get(sheet, {}) if paths else {})
    return out[0], out[1]


def calc_keys() -> dict[str, dict[str, Any]]:
    return {r["question_id"]: r for r in read_jsonl(ITEMS / "tier_k" / "test" / "calc_answers.jsonl")}


def block_keys() -> dict[str, dict[str, Any]]:
    return {r["question_id"]: r for r in read_jsonl(ITEMS / "tier_k" / "test" / "block_answers.jsonl")}


def calc_router(keys: Mapping[str, Mapping[str, Any]]):
    def router(k, a, b):
        key = keys[k[0]]
        s1, s2 = compare.calc_status(key, a), compare.calc_status(key, b)
        if compare.calc_route(key["specification"], s1, s2) == "final":
            return decide.Route("final", s1, "agreement")
        return decide.Route("expert", None, f"recomputations: {s1} / {s2}; key: {key['specification']}")
    return router


def block_router(keys: Mapping[str, Mapping[str, Any]]):
    spec = SHEETS["block_questions"]

    def router(k, a, b):
        ref = keys[k[0]]["reference"]
        if not (compare.block_matches(ref, a.get("blind_value", "")) and
                compare.block_matches(ref, b.get("blind_value", ""))):
            decide.route(spec, a, b)                       # still refuses a value outside the list
            return decide.Route("expert", None, "blind value misses the key")
        return decide.route(spec, a, b)
    return router


def rubric_router(writer: Mapping[tuple, Mapping[str, Any]], problems):
    """Checkers' verdicts by the decision rules, except that a rubric whose writer row fails an automatic
    check (problems(key, writer_row) -> list) goes to the expert whatever the checkers said."""
    spec = SHEETS["rubric_test"]

    def router(k, a, b):
        found = problems(k, writer[k]) if k in writer else ["no writer row"]
        if found:
            decide.route(spec, a, b)                       # still refuses a value outside the list
            return decide.Route("expert", None, "checks failed: " + "; ".join(found))
        return decide.route(spec, a, b)
    return router


def router_for(sheet: str):
    if sheet == "calc":
        return calc_router(calc_keys())
    if sheet == "block_questions":
        return block_router(block_keys())
    return None


def _key_dict(sheet: str, k: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(SHEETS[sheet].key, k))


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def route(work: Path, sheet: str, router=None, use_default_router: bool = True) -> dict[str, Any]:
    spec = SHEETS[sheet]
    r1, r2 = collect(work, sheet)
    if spec.group:
        _, rows = inputs.read_sheet(sheet)
        records.check_groups(spec, rows, r1)
        records.check_groups(spec, rows, r2)
    if router is None and use_default_router:
        router = router_for(sheet)
    queue, agreed = [], []
    both = sorted(set(r1) & set(r2))
    for k in both:
        rt = router(k, r1[k], r2[k]) if router else decide.route(spec, r1[k], r2[k])
        if rt.to == "expert":
            queue.append({"sheet": sheet, "key": _key_dict(sheet, k), "reason": rt.reason,
                          spec.roles[0]: r1[k], spec.roles[1]: r2[k]})
        else:
            agreed.append(k)
    sample = decide.audit_sample(sheet, agreed)
    _write(Path(work) / "expert" / f"{sheet}.queue.jsonl", queue)
    _write(Path(work) / "expert" / f"{sheet}.audit.jsonl", [{"sheet": sheet, "key": _key_dict(sheet, k)}
                                                              for k in sample])
    summary = {"rows": len(both), "final": len(agreed), "expert": len(queue), "audit": len(sample)}
    only1, only2 = sorted(set(r1) - set(r2)), sorted(set(r2) - set(r1))
    if only1:
        summary[f"only_{spec.roles[0]}"] = len(only1)
    if only2:
        summary[f"only_{spec.roles[1]}"] = len(only2)
    return summary


H1B_VALUE = {"reject": "reject", "accept with edits": "approve with edits"}
FULL_REVIEW_BY_H1B = ("block_questions", "tier_f", "tier_a")      # sheets whose every item the expert reviews


def full_review_rows(spec, r1: Mapping, r2: Mapping, h1b: Mapping, router=None) -> dict[tuple, dict[str, Any]]:
    """The expert review (H1b) as the full review of a sheet's agreed rows: one audit row per agreed row
    whose item the expert reviewed. Its `reject` or `accept with edits` changes the decision; `accept`
    keeps the agreed value; an empty verdict takes the sheet's conservative default. An agreed `reject`
    (a duplicate, for example) is not re-opened by an item-quality verdict."""
    out = {}
    for k in sorted(set(r1) & set(r2)):
        rt = router(k, r1[k], r2[k]) if router else decide.route(spec, r1[k], r2[k])
        if rt.to != "final" or k not in h1b:
            continue
        h = h1b[k]
        verdict = (h.get("verdict") or "").strip()
        value = ("reject" if rt.value == "reject"
                 else H1B_VALUE.get(verdict, rt.value if verdict == "accept" else spec.default))
        out[k] = {"sheet": spec.name, "key": _key_dict(spec.name, k), "value": value, "pass": "audit",
                  "full_review": True,
                  "notes": f"expert review (H1b): {verdict or 'unsure'} -- {' '.join((h.get('comment') or '').split())}",
                  "edits": [h["suggested_fix"]] if verdict == "accept with edits" and h.get("suggested_fix") else [],
                  "evidence": list(h.get("evidence") or [])}
    return out


def check_audit(spec, recs: list[Mapping[str, Any]], sample_keys) -> None:
    """Every seeded key has its audit row, and a sample that calls for a full review has an audit row on
    every agreed row (review_protocol.md, "Audit of the agreements")."""
    agreed = [r for r in recs if "agreed" in r]
    audited = {tuple(r["key"]): r for r in agreed if r.get("audit")}
    missing = [tuple(k) for k in sample_keys if tuple(k) not in audited]
    if missing:
        raise ValueError(f"{spec.name}: seeded audit rows missing: {missing[:5]}")
    sample = [(k, audited[tuple(k)]["agreed"], decide.expert_value(spec, audited[tuple(k)]["audit"]))
              for k in sample_keys]
    if decide.audit_outcome(sample)["full_review"]:
        not_done = [tuple(r["key"]) for r in agreed if not r.get("audit")]
        if not_done:
            raise ValueError(f"{spec.name}: the audit sample calls for a full review, and {len(not_done)} agreed "
                             f"rows have no audit row, e.g. {not_done[:3]}")


def expert_rows(work: Path, sheet: str) -> tuple[dict, dict]:
    """(the expert's decisions, the expert's audit rows) for one sheet; audit rows carry "pass": "audit"."""
    paths = role_files(work).get("EX", [])
    records.load_outputs(paths)          # refuses an unknown sheet or a row without its key columns
    decided, audit = {}, {}
    for p in paths:                      # one key can have a decision row and an audit row: keep both
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("sheet") != sheet:
                continue
            k = tuple(str(r["key"][c]) for c in SHEETS[sheet].key)
            (audit if r.get("pass") == "audit" else decided)[k] = r
    return decided, audit


def decide_sheet(work: Path, sheet: str, reviews_dir: Path | None = None, sheet_dir: Path | None = None,
                 require_all: bool = True, merge: bool = False) -> list[dict]:
    """Decide one sheet from WORK's outputs. With `merge` (a second round: the replacement items), the rows
    of WORK are decided alone, with their own audit sample, and added to the sheet's decided records; a row
    that is already decided is refused, never decided again."""
    from grcbench.common.paths import REPO
    from grcbench.review import fill
    spec = SHEETS[sheet]
    earlier = records.decided(sheet, reviews_dir=Path(reviews_dir or REPO / "reviews")) if merge else {}
    check_independence(work, sheet)
    r1, r2 = collect(work, sheet)
    ex, audit = expert_rows(work, sheet)
    router = router_for(sheet)
    if sheet in FULL_REVIEW_BY_H1B:      # the expert review of every item serves as the sheet's full review
        from grcbench.review import expert
        h1b, _ = expert.first_pass(work)
        audit = {**full_review_rows(spec, r1, r2, h1b, router=router), **audit}
    recs = records.build(spec, r1, r2, ex, audit, router=router)
    again = sorted(set(earlier) & {r["key"] for r in recs})
    if again:
        raise ValueError(f"{sheet}: {len(again)} row(s) already decided in an earlier round, e.g. {again[:3]}")
    check_audit(spec, recs, decide.audit_sample(sheet, [r["key"] for r in recs if "agreed" in r]))
    if not merge:                        # a re-run of a round covers its records; a second round needs --merge
        dropped = sorted(set(records.decided(sheet, reviews_dir=Path(reviews_dir or REPO / "reviews")))
                         - {r["key"] for r in recs})
        if dropped:
            raise ValueError(f"{sheet}: {len(dropped)} decided row(s) would be dropped, e.g. {dropped[:3]}; "
                             f"a second round is decided with --merge")
    recs = list(earlier.values()) + recs
    if spec.group:                       # the final values, too, carry one decision per group
        _, rows = inputs.read_sheet(sheet)
        records.check_groups(spec, rows, {r["key"]: {"value": r["final"]} for r in recs})
    text = fill.filled(sheet, recs, require_all=require_all)          # before any write: a failing fill writes nothing
    records.write_records(Path(reviews_dir or REPO / "reviews") / f"{sheet}.jsonl", recs)
    out = Path(sheet_dir or inputs.SHEET_DIR) / spec.file
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return recs


def decide_rubrics(work: Path, sheet: str, problems, calc: Mapping[str, Mapping[str, Any]] | None = None,
                   date: str | None = None, reviews_dir: Path | None = None, sheet_dir: Path | None = None,
                   require_all: bool = True, calc_date: str | None = None) -> list[dict]:
    """decide_sheet for a rubric sheet: the checkers' verdicts by rubric_router (a writer row that fails
    an automatic check, problems(key, writer_row) -> list, goes to the expert), each decided record with
    the writer's row ("writer") and, given a date, the question's calc_status (apply.calc_status_text)."""
    from grcbench.common.paths import REPO
    from grcbench.review import apply, fill
    spec = SHEETS[sheet]
    writer = records.load_outputs(role_files(work).get("W", [])).get(sheet, {})
    c1, c2 = collect(work, sheet)
    ex, audit = expert_rows(work, sheet)
    recs = records.build(spec, c1, c2, ex, audit, router=rubric_router(writer, problems))
    _, rows = inputs.read_sheet(sheet)
    types = {r["question_id"]: r["task_type"] for r in rows}
    for rec in recs:
        if rec["key"] not in writer:
            raise ValueError(f"{sheet}: {rec['key']} has checker rows but no writer row")
        rec["writer"] = dict(writer[rec["key"]])
        if date is not None:
            rec["calc_status"] = apply.calc_status_text(types[rec["key"][0]], (calc or {}).get(rec["key"][0]),
                                                        rec, date, calc_date)
    records.write_records(Path(reviews_dir or REPO / "reviews") / f"{sheet}.jsonl", recs)
    out = Path(sheet_dir or inputs.SHEET_DIR) / spec.file
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(fill.filled(sheet, recs, require_all=require_all), encoding="utf-8")
    return recs


def unreviewed(work: Path, sheet: str, reviews_dir: Path | None = None) -> dict[str, list[tuple[str, ...]]]:
    """Sheet rows still to decide that one or both reviewers have not written (a row decided in an earlier
    round is not expected again)."""
    from grcbench.common.paths import REPO
    spec = SHEETS[sheet]
    _, rows = inputs.read_sheet(sheet)
    done = set(records.decided(sheet, reviews_dir=Path(reviews_dir or REPO / "reviews")))
    keys = {tuple(str(r[c]) for c in spec.key) for r in rows} - done
    r1, r2 = collect(work, sheet)
    return {spec.roles[0]: sorted(keys - set(r1)), spec.roles[1]: sorted(keys - set(r2)),
            "unknown": sorted((set(r1) | set(r2)) - keys)}
