"""Where everything lives.

The repository sits on the Windows drive (visible to the team); code runs in WSL. Large downloads
and per-run working directories live on the WSL filesystem, outside the repository.
Inputs under `project/data/` are read-only: nothing in GRC-Bench writes there.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]          # .../project/grc-bench
PROJECT = REPO.parent                                # .../project
DATA = PROJECT / "data"
DATASET = DATA / "GRC_dataset_20261007" / "dataset"
SFT_RELEASE = DATA / "grc-agent-sft-v0.1.0" / "grc-agent-sft-v0.1.0"
FROZEN = DATASET / "04_eval_frozen"

ITEMS = REPO / "items"
SPLITS = REPO / "splits"
REPORTS = REPO / "reports"
HANDOFF = REPO / "handoff"
DOCS = REPO / "docs"
RUNS = REPO / "runs"
FIXTURES = REPO / "tests" / "fixtures"

HOME = Path(os.environ.get("HOME", str(Path.home())))
CACHE = Path(os.environ.get("GRC_BENCH_CACHE", str(HOME / "grc-bench-cache")))
WORK = Path(os.environ.get("GRC_BENCH_WORK", str(HOME / "grc-bench-work")))
GRC_AGENT = Path(os.environ.get("GRC_AGENT_REPO", str(HOME / "grc-agent")))

GNURADIO_BLOCKS = Path("/usr/share/gnuradio/grc/blocks")
GNURADIO_EXAMPLES = Path("/usr/share/gnuradio/examples")

# Pinned versions (checked by `gb` before anything runs).
GRC_AGENT_COMMIT = "6c98128"
GNURADIO_VERSION = "3.10.12.0"
LLAMA_CPP_BUILD = "b10333"   # the build GRC-Agent's embed_runtime.py pins
# SHA-256 over the sorted (file name, content SHA-256) of the 580 *.block.yml files of GNU Radio
# 3.10.12 (grcbench.tier_f.grc.catalog_hash); V2 fails if the installed catalog differs.
GNURADIO_CATALOG_SHA256 = "ee88541933fc92b0061a3a6b900d7964b7cd17f61f01831b560e94d4bb08a277"


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
