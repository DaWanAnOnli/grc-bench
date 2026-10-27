"""Sign-off decision 1: the grammar-constrained setting is B0 (the server default) and the cloud
teacher is B2. No file of the repository (tracked or new, not ignored) outside the excluded paths may
use the old names."""

from __future__ import annotations

import subprocess

from grcbench.common.paths import REPO

OLD_NAMES = (r"\bB3\b|b3-deepseek|nb-bf16-b2|nmc-q4-b2|B0_candidate|B0_request_candidate"
             r"|the B2 server|B0 vs\. B2|B0/B2|B0 candidate|unconstrained B0"
             r"|B0 and B2|behaviou?r is B2|B2 grammar")   # the old meaning of B2 and B0
EXCLUDED = ("tests/", "reports/scoring_reviews/",      # the 18 Oct review records stay as written
            "src/grcbench/tier_a/legacy_keys.py")       # the old smoke keys, for loading stored summaries


def test_no_old_baseline_name_left():
    out = subprocess.run(["git", "grep", "--untracked", "-nIE", OLD_NAMES, "--", "."] + [f":(exclude){e}" for e in EXCLUDED],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 1 and out.stdout == "", out.stdout[:2000]   # 1: no match
