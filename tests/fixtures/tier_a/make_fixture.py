"""Regenerates the Tier-A probe flowgraph used by the conformance selftest (not a benchmark item).

GRC-Agent loads flowgraphs with GRC's own parser, which needs the full .grc file format (block
`states`, `options.states`); the Tier-F fixtures are in the compact form. This script loads the
compact tests/fixtures/ladder/good_tone.grc into GNU Radio 3.10.12 and saves it with GRC's own
writer, inside the bubblewrap sandbox (HOME in a scratch run directory under ~/grc-bench-work).

    ./gb has no command for it; run in WSL from the repository root:
    PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 ~/grc-agent/.venv/bin/python tests/fixtures/tier_a/make_fixture.py
"""

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "ladder" / "good_tone.grc"
OUT = HERE / "probe_tone.grc"

INNER = r"""
import sys, yaml
from grcbench.tier_f._worker import _normalise
from grcbench.tier_f.grc import platform
data = _normalise(yaml.safe_load(open(sys.argv[1]).read()))
data["options"]["parameters"]["id"] = "probe_tone"
p = platform()
fg = p.make_flow_graph()
fg.import_data(data)
fg.rewrite()
fg.validate()
assert fg.is_valid(), list(fg.iter_error_messages())
p.save_flow_graph(sys.argv[2], fg)
"""


def main() -> int:
    from grcbench.common import sandbox
    from grcbench.common.paths import WORK
    run = WORK / "tier_a" / "fixture-gen"
    shutil.rmtree(run, ignore_errors=True)
    run.mkdir(parents=True)
    res = sandbox.run([sys.executable, "-c", INNER, str(SRC), str(run / OUT.name)], run, timeout=120)
    if res.returncode != 0:
        print(res.stderr)
        return 1
    shutil.copyfile(run / OUT.name, OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
