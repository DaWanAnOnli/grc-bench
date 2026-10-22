"""What GNU Radio makes of a flowgraph, for the Tier-A grader (eval plan §4.4 "Grading", §5.4).

The grader never trusts the agent's own reading of its graph: it loads the end state with GNU
Radio 3.10.12's own platform (grcbench.tier_f.grc.platform, no side effects on any home directory)
inside the bubblewrap sandbox (GRC evaluates parameters as Python, and Embedded Python blocks run
on load), one OS process per call:

    python -m grcbench.tier_a._grc_view <run_dir>/spec.json     (run by `views`, never by hand)

For every graph it reports a canonical VIEW:
    {"ok": bool, "error": str | None, "valid": bool, "errors": [GRC validation messages],
     "blocks": {name: {"key", "state", "dummy", "params": {param: raw text as GRC resolves it,
                       defaults filled in}, "evaluated": {param: JSON value or None}}},
     "connections": [[src name, src port key, dst name, dst port key], ...] (sorted)}
`params` are GRC's resolved raw values (str(param.get_value())); `evaluated` are the values GRC
evaluates them to (numbers as numbers, complex as [re, im], strings, lists of those; anything else
None), used for numeric tolerances.

For saved hierarchical / library blocks it reports whether each id is in GRC's block catalog when
the run's library directory (the episode's GRC_HIER_PATH, <run>/home/.grc_gnuradio) is on GRC's
block path: the files are copied into the worker's own GRC_HIER_PATH first.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ..common import sandbox
from ..common.paths import WORK, ensure

VIEW_TIMEOUT = 180.0


def _jsonable(value: Any, depth: int = 0) -> Any:
    import numbers
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        f = float(value)
        return f if f == f and abs(f) != float("inf") else None
    if isinstance(value, numbers.Complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, (list, tuple)) and depth < 2 and len(value) <= 4096:
        return [_jsonable(v, depth + 1) for v in value]
    try:                                          # numpy scalars and arrays
        import numpy as np
        if isinstance(value, np.ndarray) and value.size <= 4096 and depth < 2:
            return [_jsonable(v, depth + 1) for v in value.tolist()]
        if isinstance(value, np.generic):
            return _jsonable(value.item(), depth)
    except ImportError:
        pass
    return None


def view_flow_graph(fg: Any) -> dict[str, Any]:
    """The canonical view of a loaded GRC FlowGraph (rewritten and validated by the caller)."""
    blocks: dict[str, Any] = {}
    for b in fg.blocks:
        params, evaluated = {}, {}
        for k, p in b.params.items():
            params[k] = str(p.get_value())
            try:
                evaluated[k] = _jsonable(p.get_evaluated())
            except Exception:                    # noqa: BLE001 - an unevaluable parameter
                evaluated[k] = None
        blocks[b.name] = {"key": b.key, "state": str((getattr(b, "states", None) or {}).get("state", "enabled")),
                          "dummy": bool(getattr(b, "is_dummy_block", False)), "params": params,
                          "evaluated": evaluated}
    conns = sorted([c.source_block.name, str(c.source_port.key), c.sink_block.name, str(c.sink_port.key)]
                   for c in fg.connections)
    try:
        valid = bool(fg.is_valid())
        errors = list(dict.fromkeys(str(m) for m in fg.get_error_messages()))[:50]
    except Exception as exc:                     # noqa: BLE001
        valid, errors = False, [f"{type(exc).__name__}: {exc}"]
    return {"ok": True, "error": None, "valid": valid, "errors": errors, "blocks": blocks, "connections": conns}


def load_view(platform: Any, text: str) -> dict[str, Any]:
    """Load .grc text (full or compact form) with `platform` and return its view."""
    import yaml

    from ..tier_f._worker import _normalise
    try:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("not a YAML mapping")
        fg = platform.make_flow_graph()
        fg.import_data(_normalise(data))
        fg.rewrite()
        fg.validate()
    except Exception as exc:                     # noqa: BLE001 - reported, graded as unloadable
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000], "valid": False, "errors": [],
                "blocks": {}, "connections": []}
    return view_flow_graph(fg)


def _main(spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    out: dict[str, Any] = {"graphs": {}, "library_blocks": {}}
    from ..tier_f.grc import platform
    plat = platform()
    for name, path in sorted(spec.get("graphs", {}).items()):
        if path is None:
            out["graphs"][name] = {"ok": False, "error": "file missing", "valid": False, "errors": [],
                                   "blocks": {}, "connections": []}
            continue
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            out["graphs"][name] = {"ok": False, "error": f"cannot read: {exc}", "valid": False, "errors": [],
                                   "blocks": {}, "connections": []}
            continue
        out["graphs"][name] = load_view(plat, text)
    for block_id in spec.get("library_blocks", []):
        blk = plat.blocks.get(block_id)
        out["library_blocks"][block_id] = {"in_catalog": blk is not None,
                                           "label": getattr(blk, "label", None) if blk is not None else None}
    Path(spec["out"]).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return 0


def views(graphs: dict[str, Path | str | None], *, library_dir: Path | None = None,
          library_blocks: list[str] | tuple[str, ...] = (), timeout: float = VIEW_TIMEOUT) -> dict[str, Any]:
    """Views of several graphs ({key: path}) and catalog membership of `library_blocks` (with the
    files of `library_dir` on GRC's hier-block path), computed in one sandboxed GRC process."""
    from ..tier_f.grc import install_warm_cache
    run_dir = Path(tempfile.mkdtemp(prefix=f"view-{uuid.uuid4().hex[:6]}-", dir=ensure(WORK / "tier_a" / "grader")))
    try:
        inputs = run_dir / "in"
        inputs.mkdir()
        spec_graphs = {}
        for key, path in graphs.items():
            dest = inputs / f"{len(spec_graphs):02d}.grc"
            if path is not None and Path(path).is_file():
                shutil.copyfile(path, dest)
                spec_graphs[key] = str(dest)
            else:
                spec_graphs[key] = None
        install_warm_cache(run_dir / "home")
        hier = run_dir / "home" / ".grc_gnuradio"
        hier.mkdir(parents=True, exist_ok=True)
        if library_dir is not None and Path(library_dir).is_dir():
            for f in sorted(Path(library_dir).iterdir()):
                if f.is_file() and not f.is_symlink():
                    shutil.copyfile(f, hier / f.name)
        spec = {"graphs": spec_graphs, "library_blocks": list(library_blocks), "out": str(run_dir / "views.json")}
        (run_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        res = sandbox.run([sys.executable, "-m", "grcbench.tier_a._grc_view", str(run_dir / "spec.json")],
                          run_dir, timeout=timeout, cpu_seconds=int(timeout * 2))
        out_path = run_dir / "views.json"
        if not out_path.is_file():
            reason = "timed out" if res.timed_out else f"exit code {res.returncode}"
            raise RuntimeError(f"GRC view worker failed ({reason}): {res.stderr[-2000:]}")
        return json.loads(out_path.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1]))
