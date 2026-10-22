"""`./gb reference` — index the training-side agent task specifications that GRC-Bench must avoid.

Tier-A tasks must be "disjoint from every task specification used to collect or synthesize training
sessions" (proposal §5.3). The known training-side specs are:
  * Qoherent's session-collection tasks (GRC-Agent commit 4bd0436^, experiments/dataset_collection),
  * the tasks behind the 12 trainable and 20 quarantined sessions of grc-agent-sft-v0.1.0,
  * GRC-Agent's own integration scenarios (tests/scenarios/harness.py at the pinned commit).
The recovered files stay in ~/grc-bench-cache/reference (AGPL code, not part of any release); this
writes only an index with commit, path and SHA-256, plus the goal texts needed by the overlap check.
"""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

from .io import iter_jsonl, sha256_file, write_json
from .paths import CACHE, GRC_AGENT, REPORTS, SFT_RELEASE

REF = CACHE / "reference"
HARNESS_COMMIT = "4bd0436^"


def _scenarios() -> list[dict]:
    src = subprocess.run(["git", "-C", str(GRC_AGENT), "show", "HEAD:tests/scenarios/harness.py"],
                         capture_output=True, text=True, check=True).stdout
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "SCENARIOS" for t in node.targets):
            out = []
            for elt in node.value.elts:
                d = {}
                for k, v in zip(elt.keys, elt.values):
                    key = ast.literal_eval(k)
                    if key in ("name", "fixture", "prompt"):
                        try:
                            d[key] = ast.literal_eval(v)
                        except ValueError:
                            d[key] = ast.unparse(v)
                out.append(d)
            return out
    return []


def main(argv: list[str]) -> int:
    specs = []
    files = []
    task_dir = REF / "experiments" / "dataset_collection" / "tasks"
    for path in sorted(task_dir.glob("t*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        specs.append({"source": "qoherent_collection_task", "id": d.get("id"), "category": d.get("category"),
                      "seed_grc": d.get("seed_grc"), "goal": d.get("goal", "")})
    for path in sorted((REF / "experiments" / "dataset_collection").rglob("*")):
        if path.is_file():
            files.append({"path": str(path.relative_to(REF)), "sha256": sha256_file(path)})
    for fname, kind in (("sft_train.jsonl", "sft_trainable_session"), ("quarantine.jsonl", "sft_quarantined_session")):
        for row in iter_jsonl(SFT_RELEASE / fname):
            first_user = next((m.get("content", "") for m in row.get("messages", []) if m.get("role") == "user"), "")
            specs.append({"source": kind, "id": row.get("task_id"), "category": row.get("category"),
                          "seed_grc": None, "goal": first_user if isinstance(first_user, str) else json.dumps(first_user)})
    for sc in _scenarios():
        specs.append({"source": "grc_agent_scenario", "id": sc.get("name"), "category": None,
                      "seed_grc": sc.get("fixture"), "goal": sc.get("prompt", "")})
    head = subprocess.run(["git", "-C", str(GRC_AGENT), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    out = {
        "note": "Training-side agent task specs; GRC-Bench Tier-A tasks must be disjoint from all of them.",
        "grc_agent_head": head,
        "collection_harness_commit": HARNESS_COMMIT,
        "recovered_files": files,
        "specs": specs,
    }
    write_json(REPORTS / "reference" / "training_task_specs.json", out)
    by = {}
    for s in specs:
        by[s["source"]] = by.get(s["source"], 0) + 1
    print(json.dumps(by, indent=1))
    return 0
