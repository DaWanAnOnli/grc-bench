"""`./gb colab-package`: the zip that runs `./gb` on a Google Colab GPU, for a pipeline test on the dev split.

    ./gb colab-package [--out DIR]

Contents (top folder grc-bench-colab/), so that `./gb e2e`, `./gb tier-k`, `./gb tier-e` and `./gb env`
run there as they do here:
  gb, pyproject.toml, src/grcbench/ (no bytecode)
  items/tier_k/dev/, items/tier_f/dev/, items/tier_a/dev/      the dev split only
  reports/block_catalog.json (the rule cascade reads it), docs/preregistration/run_list.yaml (the run log
  reads it)
  colab/dev_placeholder.jsonl     placeholder rubrics for the dev questions (tier_k/placeholder_rubrics.py)
  colab/grcbench_colab_test.ipynb, colab/README.md, colab/u2604.py         from handoff/colab/ (u2604.py runs
                                  ./gb inside an Ubuntu 26.04 root filesystem: the pinned GNU Radio)
  tests/fixtures/ and SANDBOX_TESTS  GRC-Bench's synthetic fixture tasks and the sandbox tests the notebook's
                                  step 7b runs (the sandbox inside a Tier-A episode)
  VERSION.txt                     the GRC-Bench commit, the pins, the build time, the SHA-256 of every file

No test item is packaged: no path under items/*/test*, and the build fails (PackageRefused) if any
packaged file contains the text of a test item (a Tier-K test question, a Tier-F or Tier-A test
instruction or Tier-A spec draft). The dev records carry the canary, as in the repository. The zip
goes outside the repository (default $GRC_BENCH_WORK/colab/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

from .paths import GNURADIO_VERSION, GRC_AGENT_COMMIT, HANDOFF, ITEMS, LLAMA_CPP_BUILD, REPO, WORK

NAME = "grc-bench-colab"
OUT_DIR = WORK / "colab"
RUBRICS = "colab/dev_placeholder.jsonl"
NOTEBOOK = "colab/grcbench_colab_test.ipynb"
COLAB_HANDOFF = HANDOFF / "colab"
SANDBOX_TESTS = ("tests/test_sandbox_masks.py", "tests/test_common.py", "tests/test_tier_a_runner.py")
FILES = ("gb", "pyproject.toml", "reports/block_catalog.json", "docs/preregistration/run_list.yaml", *SANDBOX_TESTS)
TREES = ("src/grcbench", "items/tier_k/dev", "items/tier_f/dev", "items/tier_a/dev", "tests/fixtures")
MIN_TEXT = 40            # shorter test texts are not searched for (they would match ordinary words)


class PackageRefused(RuntimeError):
    """A file for the package would carry test-split content."""


def test_texts() -> list[str]:
    """The texts of every test item: Tier-K questions, Tier-F and Tier-A instructions, Tier-A spec drafts."""
    texts = []
    for name in ("frozen_questions.jsonl", "block_questions.jsonl"):
        path = ITEMS / "tier_k" / "test" / name
        texts += [json.loads(line)["question"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for tier in ("tier_f", "tier_a"):
        texts += [json.loads(p.read_text(encoding="utf-8"))["instruction"] for p in sorted((ITEMS / tier / "test").rglob("task.json"))]
    drafts = ITEMS / "tier_a" / "test_spec_drafts.jsonl"
    texts += [json.loads(line)["instruction"] for line in drafts.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [t.strip() for t in texts if len(t.strip()) >= MIN_TEXT]


def check_content(files: dict[str, bytes], texts: list[str]) -> None:
    """Refuse a package with a test-split path or a file containing a test item's text."""
    bad = [rel for rel in files if rel.startswith("items/") and "/test" in rel]
    for rel, data in files.items():
        content = data.decode("utf-8", errors="replace")
        bad += [rel for t in texts if t in content][:1]
    if bad:
        raise PackageRefused(f"files with test item content: {sorted(set(bad))[:10]}")


def _collect() -> dict[str, bytes]:
    from grcbench.splits import release
    from grcbench.tier_k import placeholder_rubrics
    from grcbench.tier_k import run as krun
    files = {rel: (REPO / rel).read_bytes() for rel in FILES}
    for tree in TREES:
        for p in sorted((REPO / tree).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                files[p.relative_to(REPO).as_posix()] = p.read_bytes()
    rows, _ = placeholder_rubrics.records(krun.load_split("dev"), release.validation_reference_answers())
    files[RUBRICS] = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")
    for name in ("grcbench_colab_test.ipynb", "README.md", "u2604.py"):
        files[f"colab/{name}"] = (COLAB_HANDOFF / name).read_bytes()
    return files


def build(out_dir: Path = OUT_DIR) -> Path:
    from grcbench.runlog import log as runlog
    files = _collect()
    check_content(files, test_texts())
    lines = [f"GRC-Bench commit {runlog.git_commit()} (uncommitted changes: {runlog.git_dirty()})",
             f"pins: GNU Radio {GNURADIO_VERSION}, GRC-Agent {GRC_AGENT_COMMIT}, llama.cpp {LLAMA_CPP_BUILD}",
             f"built {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", "dev split only; SHA-256 per file:"]
    lines += [f"{hashlib.sha256(data).hexdigest()}  {rel}" for rel, data in sorted(files.items())]
    files["VERSION.txt"] = ("\n".join(lines) + "\n").encode("utf-8")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{NAME}.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel, data in sorted(files.items()):
            info = zipfile.ZipInfo(f"{NAME}/{rel}", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if rel == "gb" or rel.endswith(".sh") else 0o644) << 16
            zf.writestr(info, data)
    return target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gb colab-package", description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR, help=f"output folder (default {OUT_DIR})")
    a = ap.parse_args(argv)
    try:
        target = build(a.out)
    except PackageRefused as exc:
        print(f"refused: {exc}")
        return 1
    with zipfile.ZipFile(target) as zf:
        n = len(zf.namelist())
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB, {n} files, SHA-256 "
          f"{hashlib.sha256(target.read_bytes()).hexdigest()[:16]}...)")
    print("upload it to Google Drive (My Drive/grc-bench-colab/) and open colab/grcbench_colab_test.ipynb in Colab")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
