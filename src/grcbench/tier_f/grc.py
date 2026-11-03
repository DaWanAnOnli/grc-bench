"""Headless access to the GNU Radio Companion platform (GNU Radio 3.10.12).

Loads the block catalog without a GUI and without side effects on the user's home directory: the
library cache is never saved, hierarchical blocks are never auto-built, and (unless documentation
is needed) the C++ docstring extractor subprocess is not started. A warm library cache, built once
into ~/grc-bench-cache/grc-home, is copied into each sandboxed run so loading takes about a second.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from ..common.paths import CACHE

BLOCKS_DIR = Path("/usr/share/gnuradio/grc/blocks")
WARM_HOME = CACHE / "grc-home"
CACHE_REL = Path(".cache") / "gnuradio" / "grc" / "cache_v2.json"
STREAM_TYPES = {  # GRC port dtype -> (null/vector sink type key, numpy dtype, bytes per item)
    "complex": ("complex", "complex64", 8), "fc32": ("complex", "complex64", 8),
    "float": ("float", "float32", 4), "f32": ("float", "float32", 4),
    "int": ("int", "int32", 4), "s32": ("int", "int32", 4),
    "short": ("short", "int16", 2), "s16": ("short", "int16", 2), "sc16": ("int", "int32", 4),
    "byte": ("byte", "int8", 1), "s8": ("byte", "int8", 1), "bits": ("byte", "int8", 1),
}


@lru_cache(maxsize=2)
def platform(docs: bool = False, save_cache: bool = False):
    from gnuradio import gr
    import gnuradio.grc.core.cache as cache
    if not save_cache:
        cache.Cache.save = lambda self: None                     # never rewrite a library cache
    from gnuradio.grc.core.platform import Platform
    from gnuradio.grc.core.FlowGraph import FlowGraph
    # Never auto-build hierarchical blocks from .grc files found on disk: an unknown block id must
    # load as an (unknown) dummy block, not trigger a file search and code generation.
    Platform.load_and_generate_flow_graph = lambda self, *a, **k: (None, None)
    FlowGraph._build_depending_hier_block = lambda self, block_id: False
    if not docs:
        import gnuradio.grc.core.utils.extract_docs as ed
        ed.SubprocessLoader.start = lambda self: None
        ed.SubprocessLoader.finish = lambda self: None
        ed.SubprocessLoader.query = lambda self, *a, **k: None
    p = Platform(name="grc-bench", prefs=gr.prefs(), version=gr.version(),
                 version_parts=(gr.major_version(), gr.api_version(), gr.minor_version()))
    p.build_library()
    return p


def catalog_hash(blocks_dir: Path = BLOCKS_DIR) -> str:
    """SHA-256 over the sorted (file name, content hash) of every *.block.yml in the catalog."""
    h = hashlib.sha256()
    for path in sorted(blocks_dir.glob("*.block.yml")):
        h.update(path.name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).hexdigest().encode() + b"\n")
    return h.hexdigest()


def warm_cache_file() -> Path:
    """Build (once) a GRC library cache in WARM_HOME; returns the cache file path. Callers that start together
    (parallel episodes on a machine without the cache yet) take turns on a lock file beside WARM_HOME: one
    builds it, the others wait and then use it. The stamp is written last, so a cache is used only complete."""
    target = WARM_HOME / CACHE_REL
    stamp = WARM_HOME / "catalog_hash.txt"
    current = catalog_hash()

    def ready() -> bool:
        return target.exists() and stamp.exists() and stamp.read_text().strip() == current

    if ready():
        return target
    WARM_HOME.parent.mkdir(parents=True, exist_ok=True)
    with open(WARM_HOME.parent / f"{WARM_HOME.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)                   # released when the file closes
        if ready():
            return target
        shutil.rmtree(WARM_HOME, ignore_errors=True)
        WARM_HOME.mkdir(parents=True)
        env = {**os.environ, "HOME": str(WARM_HOME), "XDG_CACHE_HOME": str(WARM_HOME / ".cache"),
               "PYTHONDONTWRITEBYTECODE": "1"}
        res = subprocess.run([sys.executable, "-c",
                              "from grcbench.tier_f.grc import platform; platform(docs=False, save_cache=True)"],
                             env=env, cwd=str(WARM_HOME), capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"GRC could not build its library cache (exit {res.returncode}): {res.stderr[-2000:]}")
        if not target.exists():
            raise RuntimeError("GRC did not write a library cache")
        stamp.write_text(current + "\n")
    return target


def install_warm_cache(run_home: Path) -> None:
    dest = run_home / CACHE_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(warm_cache_file(), dest)


def enum_options(block_key: str) -> dict[str, list[str]]:
    """{param id: allowed option keys} for parameters whose options are a literal list."""
    blk = platform().blocks.get(block_key)
    out: dict[str, list[str]] = {}
    if blk is None:
        return out
    for prm in getattr(blk, "parameters_data", []) or []:
        opts = prm.get("options")
        if prm.get("dtype") == "enum" and isinstance(opts, list):
            out[prm["id"]] = [str(o) for o in opts]
    return out


def block_ids() -> set[str]:
    return set(platform().blocks.keys())


def param_defaults(block_key: str) -> dict[str, str]:
    """{param id: default value} as declared in the block's catalog entry."""
    blk = platform().blocks.get(block_key)
    out: dict[str, str] = {}
    for prm in getattr(blk, "parameters_data", []) or []:
        if "default" in prm:
            out[prm["id"]] = str(prm["default"])
    return out


def loaded_params(data: dict) -> dict[str, dict[str, str]]:
    """Load a (trusted) flowgraph dict in GRC and return every block's resolved raw parameter values."""
    import copy
    data = copy.deepcopy(data)
    data.setdefault("options", {}).setdefault("states", {})
    for blk in data.get("blocks", []):
        blk.setdefault("states", {})
        blk.setdefault("parameters", {})
    fg = platform().make_flow_graph()
    fg.import_data(data)
    fg.rewrite()
    return {b.name: {k: str(p.get_value()) for k, p in b.params.items()} for b in fg.blocks}


def compact_equivalent(full: dict, compacted: dict) -> list[str]:
    """Differences between the GRC-resolved parameters of a flowgraph and its compact form."""
    a, b = loaded_params(full), loaded_params(compacted)
    diffs = []
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            diffs.append(f"block '{name}' missing in one form")
            continue
        for k in sorted(set(a[name]) | set(b[name])):
            if a[name].get(k) != b[name].get(k):
                diffs.append(f"{name}.{k}: {a[name].get(k)!r} != {b[name].get(k)!r}")
    return diffs
