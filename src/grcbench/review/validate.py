"""The coordinator's checks of the agents' claims (docs/guidelines/review_protocol.md, "Evidence").

  parse_lines        the sheet's "P1 (w=2): ..." / "D1: ..." lines -> the judge's rubric entries
  rubric_problems    what the judge (judge_prompt.normalize_rubric, numeric_reference) and the rule
                     cascade (rules.parse_reference) refuse in a rubric record
  quote_problems     every `<path> :: "<quote>"` source line must be found, whitespace-normalised, in
                     the file it names
  numeric_matches    a rubric's numeric reference gives full credit to the confirmed calculation
  shared_8grams      word 8-grams a rubric shares with teacher answers (none allowed)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from grcbench.review import compare
from grcbench.splits.twins import ngrams
from grcbench.tier_k import judge_prompt, rules

_LINE = {"P": re.compile(r"^\s*(P\d+)\s*(?:\(\s*w\s*=\s*([0-9.]+)\s*\))?\s*:\s*(.+?)\s*$"),
         "D": re.compile(r"^\s*(D\d+)\s*:\s*(.+?)\s*$")}
_QUOTE = re.compile(r'^\s*(?P<path>[^"]+?)\s*::\s*"(?P<quote>.*)"\s*$')


def parse_lines(lines: Sequence[str], kind: str) -> list[dict[str, Any]]:
    out = []
    for i, line in enumerate((x for x in lines if x.strip()), 1):
        m = _LINE[kind].match(line)
        if not m:
            raise ValueError(f"{kind}{i}: not a '{kind}<n>: text' line: {line!r}")
        if kind == "P":
            out.append({"id": m.group(1), "weight": float(m.group(2) or 1), "text": m.group(3)})
        else:
            out.append({"id": m.group(1), "text": m.group(2)})
    return out


def rubric_problems(record: Mapping[str, Any]) -> list[str]:
    try:
        judge_prompt.normalize_rubric(record)
        if record.get("numeric") is not None:
            judge_prompt.numeric_reference(record)
            rules.parse_reference(record["numeric"])
    except Exception as e:                         # every refusal is a problem to report
        return [f"{type(e).__name__}: {e}"]
    return []


def _norm(text: str) -> str:
    return " ".join(text.split())


def quote_problems(source_lines: Iterable[str], read=lambda p: Path(p).expanduser().read_text(
        encoding="utf-8", errors="replace")) -> list[str]:
    probs = []
    for line in source_lines:
        m = _QUOTE.match(line)
        if not m:
            continue                               # a named reference (a textbook) is not a quote
        path, quote = m.group("path"), _norm(m.group("quote"))
        try:
            text = _norm(read(path))
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
            probs.append(f"{path}: no such file")
            continue
        if quote not in text:
            probs.append(f"{path}: quote not found: {quote[:80]!r}")
    return probs


def numeric_matches(numeric: Any, confirmed: Sequence[str]) -> bool:
    """Every part of the rubric's reference gives full credit to the confirmed value(s), in order."""
    parts = [numeric] if isinstance(numeric, Mapping) else list(numeric or [])
    if len(parts) != len(confirmed):
        return False
    return all(compare.full_credit(c, p) for c, p in zip(confirmed, parts))


def shared_8grams(rubric_lines: Iterable[str], answers: Iterable[str]) -> int:
    mine: set = set()
    for line in rubric_lines:
        mine |= ngrams(re.sub(r"^\s*[PD]\d+\s*(?:\([^)]*\))?\s*:\s*", "", line))
    theirs: set = set()
    for a in answers:
        theirs |= ngrams(a)
    return len(mine & theirs)
