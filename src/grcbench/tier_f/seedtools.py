"""Turn a GNU Radio example (a usable seed) into a headless, compact task flowgraph.

Every Qt GUI or audio sink becomes one File Sink per stream input, named "<sink>_<i>" and writing
"<sink>_<i>.dat", so a task can name a concrete observation point; Qt widget variables become plain
variables; generate_options becomes no_gui; GUI coordinates and default-valued parameters are dropped.
Runs in-process: only trusted example files go through here, never model output.

    ./gb tier-f seed <example path relative to /usr/share/gnuradio/examples> [--out file.grc]
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..common.paths import GNURADIO_EXAMPLES, WORK
from . import grc, grcfile
from ._worker import _convert_headless, _normalise


def headless_seed(example_rel: str) -> tuple[dict, list[dict]]:
    """Returns (compact flowgraph dict, file sinks [{name, file, type}] available as observation points)."""
    src = GNURADIO_EXAMPLES / example_rel
    data = _normalise(yaml.safe_load(src.read_text(encoding="utf-8")))
    plat = grc.platform()
    fg = plat.make_flow_graph()
    fg.import_data(data)
    fg.rewrite()
    _convert_headless(plat, fg, sink_mode="file")
    exported = json.loads(json.dumps(fg.export_data(), default=str))
    exported["options"]["parameters"]["id"] = Path(example_rel).stem
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in exported["blocks"]}
    compact = grcfile.compact(exported, defaults)
    keep_opts = {k: v for k, v in exported["options"]["parameters"].items()
                 if k in ("id", "title", "generate_options", "output_language") and v not in (None, "")}
    compact["options"] = {"parameters": keep_opts}
    sinks = [{"name": b["name"], "file": b["parameters"].get("file"), "type": b["parameters"].get("type")}
             for b in compact["blocks"] if b["id"] == "blocks_file_sink"]
    return compact, sinks


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="gb tier-f seed")
    ap.add_argument("example")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    data, sinks = headless_seed(args.example)
    out = Path(args.out) if args.out else WORK / "seeds" / (Path(args.example).stem + ".grc")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(grcfile.dump(data), encoding="utf-8")
    print(json.dumps({"written": str(out), "observation_points": sinks}, indent=1))
    return 0
