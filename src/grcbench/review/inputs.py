"""The reviewers' input copies: the sheet without the columns its guide hides from them."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from grcbench.common.paths import HANDOFF
from grcbench.review.sheets import SHEETS

SHEET_DIR = HANDOFF / "review_sheets"


def read_sheet(name: str) -> tuple[list[str], list[dict[str, str]]]:
    text = (SHEET_DIR / SHEETS[name].file).read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader.fieldnames or []), list(reader)


def write_blinded(name: str, out_dir: Path) -> Path:
    spec = SHEETS[name]
    header, rows = read_sheet(name)
    keep = [c for c in header if c not in spec.hidden]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keep, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    out = Path(out_dir) / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(buf.getvalue(), encoding="utf-8")
    return out
