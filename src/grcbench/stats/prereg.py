"""`./gb prereg`: fill the generated appendices of the G2 pre-registration (eval plan §9).

docs/preregistration/preregistration.md is written by hand, except two appendices that must match the
code exactly, each kept between BEGIN/END markers:

  stats-proposals   the analysis proposals and approved definitions (stats.proposals.as_markdown)
  judge-prompt      the Tier-K question prompt's version and hash (tier_k.answer_prompt), then the
                    judge prompt version and hash, the rule-cascade version and the entity index hash
                    (tier_k.judge_prompt.snapshot, also written in full to
                    docs/preregistration/judge_prompt_snapshot.json)

    ./gb prereg            rewrite both appendices and the snapshot
    ./gb prereg --check    exit 1 if an appendix or the snapshot differs from what the code would write
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ..common.io import read_json, write_json
from ..common.paths import DOCS
from . import proposals

PREREG = DOCS / "preregistration" / "preregistration.md"
SNAPSHOT_NAME = "docs/preregistration/judge_prompt_snapshot.json"      # as the appendix names it
SNAPSHOT = DOCS.parent / SNAPSHOT_NAME
MARK = "<!-- {edge} {name}{note} -->"
NOTE = " (generated; do not edit by hand)"


def proposals_block() -> str:
    lines = proposals.as_markdown().splitlines()
    if lines and lines[0].startswith("## "):        # the appendix heading is the document's own
        lines = lines[1:]
    return "\n".join(lines).strip("\n")


def judge_block(snap: dict, answer: dict | None = None) -> str:
    agg = snap["aggregation"]
    ent = snap["entity_check"]
    judges = snap["judges"]
    judges = (f"{judges['primary']} (fallback: {judges['fallback']})" if isinstance(judges, dict)
              else ", ".join(judges))
    rows = []
    if answer:
        rows = [f"| Tier-K question prompt version | `{answer['prompt_version']}` |",
                f"| Tier-K question prompt SHA-256 | `{answer['prompt_sha256']}` |",
                f"| Tier-K answer limit | {answer['max_answer_tokens']:,} tokens, reasoning included |"]
    return "\n".join([
        "| Item | Value |", "|---|---|", *rows,
        f"| Judge prompt version | `{snap['prompt_version']}` |",
        f"| Judge prompt SHA-256 | `{snap['prompt_sha256']}` |",
        f"| Judges | {judges} |",
        f"| Rule cascade | `{agg['rules_version']}` |",
        f"| Entity index | {ent['index_names']:,} names, SHA-256 `{ent['index_sha256']}` |",
        f"| Probe requests in the hash | {snap['probes']['items']} toy items x {snap['probes']['answers']} answers |",
        "",
        "Tier K is closed-book: a model under test gets one message, the question prompt's instruction and "
        "the question (`grcbench.tier_k.answer_prompt`); B1 places the retrieved documentation before it. "
        "The fallback judge uses the judge template unchanged, so both judges share one version and one "
        "hash. The full template, rubric rendering, verdict schema and aggregation rules are in "
        f"`{SNAPSHOT_NAME}`."])


def _replace(text: str, name: str, body: str) -> str:
    begin, end = MARK.format(edge="BEGIN", name=name, note=NOTE), MARK.format(edge="END", name=name, note="")
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"{PREREG}: markers for {name!r} not found")
    return pattern.sub(lambda _: f"{begin}\n{body}\n{end}", text, count=1)


def render(text: str, snap: dict, answer: dict | None = None) -> str:
    return _replace(_replace(text, "stats-proposals", proposals_block()), "judge-prompt", judge_block(snap, answer))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb prereg", description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="only report whether the appendices are current")
    args = ap.parse_args(argv)
    from ..tier_k import answer_prompt
    from ..tier_k.judge_prompt import snapshot
    snap = snapshot()
    text = PREREG.read_text(encoding="utf-8")
    new = render(text, snap, answer_prompt.snapshot())
    if args.check:
        current = new == text
        # the snapshot file as JSON, against what write_json would write
        snap_current = SNAPSHOT.is_file() and read_json(SNAPSHOT) == json.loads(json.dumps(snap))
        print(f"{PREREG.name}: generated appendices {'current' if current else 'OUT OF DATE (run ./gb prereg)'}")
        print(f"{SNAPSHOT.name}: {'current' if snap_current else 'OUT OF DATE (run ./gb prereg)'}")
        return 0 if current and snap_current else 1
    PREREG.write_text(new, encoding="utf-8", newline="\n")
    write_json(SNAPSHOT, snap)
    print(f"wrote {PREREG} and {SNAPSHOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
