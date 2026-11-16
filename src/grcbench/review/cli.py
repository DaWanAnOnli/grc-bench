"""./gb review: the review protocol's coordinator commands (docs/guidelines/review_protocol.md).

  inputs WORK SHEET... [--undecided]  write each reviewer's blinded input copy to WORK/inputs/ (--undecided:
                                      only the rows with no decided record, a second round)
  declare --agent N --role R --sheets S... --rows N --brief FILE
                                      declare an agent run in reviews/review_log.jsonl (cap: 30)
  finish --agent N OUTPUT...          record the run's output hash and row count
  route WORK SHEET...                 apply the decision rules; write the expert's queue and audit sample
  decide WORK SHEET... [--merge]      with the expert's rows: write reviews/<sheet>.jsonl and the filled sheet
                                      (--merge: WORK is a second round, added to the decided records)
  rubrics WORK [--date D]             decide both rubric sheets (rubric router, the writer's rows kept), check
                                      every final rubric, write the filled sheets and items/tier_k/<split>/rubrics.jsonl
  expert WORK [--date D] [--merge]    the expert review (H1b): reviews/expert_review.jsonl, the H1b counts (on
                                      the original pool; the replacements beside it) and
                                      handoff/review_sheets/expert_review.csv (--merge: a second round)
  summary                             write reports/review_summary.md from reviews/
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

from grcbench.review import inputs, log
from grcbench.review.sheets import SHEETS


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb review", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inputs")
    p.add_argument("work", type=Path)
    p.add_argument("sheets", nargs="+", choices=sorted(SHEETS))
    p.add_argument("--undecided", action="store_true")
    p = sub.add_parser("declare")
    p.add_argument("--agent", type=int, required=True)
    p.add_argument("--role", required=True)
    p.add_argument("--sheets", nargs="+", required=True)
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--brief", type=Path, required=True)
    p = sub.add_parser("finish")
    p.add_argument("--agent", type=int, required=True)
    p.add_argument("outputs", type=Path, nargs="+")
    for name in ("route", "decide"):
        p = sub.add_parser(name)
        p.add_argument("work", type=Path)
        p.add_argument("sheets", nargs="+", choices=sorted(SHEETS))
        if name == "decide":
            p.add_argument("--merge", action="store_true")
    p = sub.add_parser("rubrics")
    p.add_argument("work", type=Path)
    p.add_argument("--date", default=datetime.date.today().isoformat())
    p = sub.add_parser("expert")
    p.add_argument("work", type=Path)
    p.add_argument("--date", default=datetime.date.today().isoformat())
    p.add_argument("--merge", action="store_true")
    sub.add_parser("summary")
    args = ap.parse_args(argv)

    if args.cmd == "expert":
        from grcbench.review import expert, replacements, scope
        counts = expert.write(args.work, scope.expert_items(), date=args.date, removed=scope.removed_ids(),
                              merge=args.merge, replacements=replacements.all_ids())
        for tier, c in sorted(counts.items()):
            print(f"H1b tier {tier}: {c['accepted']} of {c['items']} accepted ({c['share']}); holds: {c['holds']}; "
                  f"kept by the review: {c['accepted_kept']} of {c['kept']} ({c['share_kept']})"
                  + (f"; replacements accepted: {c['replacements_accepted']} of {c['replacements']}"
                     if "replacements" in c else ""))
        return 0

    if args.cmd == "rubrics":
        return _rubrics(args.work, args.date)

    if args.cmd == "summary":
        import json

        from grcbench.common.io import write_text
        from grcbench.common.paths import REPO
        from grcbench.review import records, report
        summaries = [report.sheet_summary(n, list(records.decided(n).values()))
                     for n in SHEETS if records.decided(n)]
        runs = ([json.loads(x) for x in log.LOG.read_text(encoding="utf-8").splitlines() if x.strip()]
                if log.LOG.exists() else [])
        notes_file = REPO / "reviews" / "coordinator_notes.json"
        notes = json.loads(notes_file.read_text(encoding="utf-8")) if notes_file.exists() else []
        h1b_file = REPO / "reviews" / "expert_review_h1b.json"
        h1b = json.loads(h1b_file.read_text(encoding="utf-8")) if h1b_file.exists() else None
        write_text(REPO / "reports" / "review_summary.md",
                   report.render(summaries, sorted(runs, key=lambda e: e["agent"]), notes, h1b=h1b))
        print(f"reports/review_summary.md: {len(summaries)} decided sheets, {len(runs)} declared runs")
        return 0

    if args.cmd == "route":
        from grcbench.review import coordinate
        for name in args.sheets:
            missing = coordinate.unreviewed(args.work, name)
            print(name, coordinate.route(args.work, name),
                  {k: len(v) for k, v in missing.items() if v})
        return 0
    if args.cmd == "decide":
        from grcbench.review import coordinate
        for name in args.sheets:
            recs = coordinate.decide_sheet(args.work, name, merge=args.merge)
            print(f"{name}: {len(recs)} decided rows -> reviews/{name}.jsonl and the filled sheet")
        return 0

    if args.cmd == "inputs":
        for name in args.sheets:
            print(inputs.write_blinded(name, args.work / "inputs", undecided_only=args.undecided))
        return 0
    if args.cmd == "declare":
        e = log.declare(log.LOG, agent=args.agent, role=args.role, sheets=args.sheets, rows=args.rows,
                        brief_text=args.brief.read_text(encoding="utf-8"),
                        date=datetime.date.today().isoformat())
        print(f"declared agent {e['agent']} ({e['role']}, {', '.join(e['sheets'])}, {e['rows']} rows)")
        return 0
    if args.cmd == "finish":
        e = log.finish(log.LOG, agent=args.agent, outputs=args.outputs)
        print(f"agent {e['agent']}: {e['rows_written']} rows, output sha256 {e['output_sha256'][:12]}")
        return 0
    return 2


def _rubrics(work: Path, date: str) -> int:
    """Decide both rubric sheets and write the judge's rubric files. A final rubric (the writer's, or the
    expert's merged one) that fails an automatic check stops its split: nothing is written to its
    rubrics.jsonl, and the failures are listed for the expert."""
    import json

    from grcbench.common.io import read_jsonl, write_text
    from grcbench.common.paths import DATASET, REPO
    from grcbench.review import apply, coordinate, fill, records, rubric, validate
    from grcbench.tier_k import calc as kcalc
    from grcbench.tier_k.rubrics import OUT, SOURCES
    keys = coordinate.calc_keys()
    calc_recs = {k[0]: r for k, r in kcalc.decided_calc().items()}
    confirmed = {q: {"outcome": r["final"], "confirmed": kcalc.confirmed(keys[q], r)} for q, r in calc_recs.items()}
    runs = ([json.loads(x) for x in log.LOG.read_text(encoding="utf-8").splitlines() if x.strip()]
            if log.LOG.exists() else [])
    calc_date = max((e["date"] for e in runs if "calc" in e["sheets"]), default=None)   # the recomputation's runs
    status = 0
    for split in ("test", "dev"):
        sheet = f"rubric_{split}"
        questions = list(read_jsonl(SOURCES[split]))
        teach = rubric.teacher_answers({q["id"] for q in questions}, DATASET / "05_answers")

        def problems(k, w, teach=teach):
            return rubric.problems(w, confirmed.get(k[0]), teach.get(k[0], []))
        recs = coordinate.decide_rubrics(work, sheet, problems, calc=calc_recs, date=date, calc_date=calc_date)
        bad = {r["key"][0]: found for r in recs
               if (found := rubric.final_problems(r, confirmed.get(r["key"][0]), teach.get(r["key"][0], [])))}
        if bad:
            status = 1
            for qid, found in sorted(bad.items()):
                print(f"{sheet}: {qid}: the final rubric fails a check: {' || '.join(found)}")
            print(f"{sheet}: {len(bad)} final rubric(s) fail a check; {OUT[split]} not written")
            continue
        patch_file = REPO / "reviews" / f"{sheet}_patches.json"
        pats = json.loads(patch_file.read_text(encoding="utf-8")) if patch_file.exists() else []
        out = apply.rubric_file(questions, recs, calc_recs, date, calc_date, patches=pats)
        for rec in recs:                  # the sheet and the records name the expert review's edits too
            rec["h1b_edits"] = [p["edit"] for p in pats if p["item"] == rec["key"][0]]
        records.write_records(REPO / "reviews" / f"{sheet}.jsonl", recs)
        (inputs.SHEET_DIR / SHEETS[sheet].file).write_text(fill.filled(sheet, recs), encoding="utf-8")
        patched = {p["item"] for p in pats}
        bad = {}
        for r in out:                     # an expert-review edit passes the same checks as a written rubric
            if r["question_id"] in patched:
                found = validate.rubric_problems(r) + validate.quote_problems(r["sources"])
                if (numeric := rubric.numeric_problem(r["numeric"], confirmed.get(r["question_id"]))):
                    found.append(numeric)
                texts = [x["text"] for x in r["required_points"] + r["disqualifying_errors"]]
                if (n := validate.shared_8grams(texts, teach.get(r["question_id"], []))):
                    found.append(f"{n} word 8-gram(s) shared with a teacher answer")
                if found:
                    bad[r["question_id"]] = found
        if bad:
            status = 1
            for qid, found in sorted(bad.items()):
                print(f"{sheet}: {qid}: the expert review's edit fails a check: {' || '.join(found)}")
            print(f"{sheet}: {OUT[split]} not written")
            continue
        write_text(REPO / OUT[split], "".join(json.dumps(records.redact(r), ensure_ascii=False, sort_keys=True) + "\n"
                                              for r in out))
        counts: dict[str, int] = {}
        for r in out:
            counts[r["review"]["status"]] = counts.get(r["review"]["status"], 0) + 1
        by_ex = sum(r["decided_by"].startswith("EX") for r in recs)
        print(f"{sheet}: {len(out)} rubrics -> {OUT[split]} {counts}; decided by the expert: {by_ex}")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
