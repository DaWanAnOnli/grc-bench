"""GNU Radio block catalog (`./gb blocks`; plan WS3a).

The pinned GNU Radio 3.10.12 catalog with each block's module, family and eligibility, for the Tier-K
GNU Radio questions (tier_k/blockgen.py) and the trace audit (contamination/agent_traces.py). There is
no test/train block split: training data may use any block, and Tier K's GNU Radio questions come from
a seed, so every evaluation can draw a fresh set with the same composition.

  load_platform()  headless GRC load of the pinned catalog. It never reads or writes ~/.cache/gnuradio
                   (the JSON cache is bypassed and its save disabled), never auto-builds hier blocks,
                   and does not start GRC's docstring subprocess.
  catalog()        one record per loaded block (585 on GNU Radio 3.10.12).
  catalog_hash()   SHA-256 over the sorted (file name, SHA-256) of every *.block.yml; the same bytes
                   as tier_f/grc.py, so the ladder's V2 rung and this catalog pin the same catalog.

Rules (also written into reports/block_catalog.json):

  module       GRC's category paths all start with "Core", so they cannot tell modules apart. The
               module is (1) the stem of the `<module>.tree.yml` file that lists the block id
               (GNU Radio installs one tree file per module); else (2) the first `_`-separated word
               of the block's YAML file name (audio, network, trellis, zeromq and the RFNoC uhd
               blocks have no tree file); else (3) "grc" for GRC's built-ins without a YAML file.
  eligibility  A block is excluded, with every reason recorded, if it is
                 non-block   a GRC entry, not a signal-processing block: options, import, note,
                             snippet, parameter, pad_source/sink, virtual_source/sink, epy_block,
                             epy_module, _dummy, every id starting with "variable", and variable-like
                             entries (a `value` expression and no ports: json_config, yaml_config);
                 deprecated  the YAML flags contain "deprecated", or GRC files it under a
                             "Deprecated" category;
                 hardware-or-network-module  module in uhd, soapy, iio, zeromq, network, audio.
  family       Blocks that differ only in item type or version form one family. The key of a name
               is: lower-case; drop one trailing type-signature word made only of the GNU Radio type
               letters b c f i s v x (_xx, _vxx, _ccf, _fc, _vcvcf ...); drop a trailing `_hier`;
               drop one trailing version digit (agc2 -> agc, channel_model2 -> channel_model). Two
               blocks share a family when their ids, or their YAML file names (without .block.yml),
               have the same key (union of both links). A family is named by its smallest key; the
               Tier-K block questions use it as their cluster.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from grcbench.common.io import sha256_file, write_json, write_text
from grcbench.common.paths import GNURADIO_BLOCKS, REPO

EXCLUDED_MODULES = ("audio", "iio", "network", "soapy", "uhd", "zeromq")
NON_BLOCK_IDS = ("_dummy", "epy_block", "epy_module", "import", "note", "options", "pad_sink",
                 "pad_source", "parameter", "snippet", "virtual_sink", "virtual_source")
REASONS = ("non-block", "deprecated", "hardware-or-network-module")   # order = priority
AUTO_PARAMS = {"id", "alias", "affinity", "minoutbuf", "maxoutbuf", "comment"}   # added by GRC itself
GENERATOR = "grcbench.tier_k.catalog"

_TYPE_WORD = re.compile(r"_[bcfisvx]{1,5}$")
_VERSION_DIGIT = re.compile(r"(?<=[a-z])\d$")
_RATE_PARAM = re.compile(r"^(audio_)?(decim|decimation|interp|interpolation)$", re.I)
_TEMPLATE_NAME = re.compile(r"\$\{\s*([A-Za-z_]\w*)")


# --------------------------------------------------------------------------- loading


class _NoDocstrings:
    """Stands in for GRC's docstring extractor, which starts a Python subprocess and fills block
    documentation asynchronously (non-deterministic). `has_documentation` uses the YAML text only."""

    def start(self): pass
    def query(self, *args, **kwargs): pass
    def finish(self): pass
    def wait(self): pass
    def terminate(self): pass


class _FileOnlyCache:
    """Stands in for GRC's JSON cache (~/.cache/gnuradio/grc/cache_v2.json): every YAML file is parsed
    from disk, and nothing is read from, created in or written to the user's cache directory."""

    def __init__(self, *args, **kwargs):
        from gnuradio.grc.core.io import yaml as grc_yaml
        self._load = grc_yaml.safe_load

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_or_load(self, filename):
        with open(filename, encoding="utf-8") as fh:
            return self._load(fh)


def load_platform(blocks_dir: Path = GNURADIO_BLOCKS):
    """Load the GRC block library headlessly from the system catalog directory only.

    User block paths (~/.local/state/gnuradio, GRC_BLOCKS_PATH) are ignored so the catalog depends
    on the pinned GNU Radio install alone."""
    from gnuradio import gr
    import gnuradio.grc.core.cache as grc_cache
    grc_cache.Cache.save = lambda self: None                     # never rewrite ~/.cache/gnuradio
    from gnuradio.grc.core import platform as grc_platform
    grc_platform.Platform.load_and_generate_flow_graph = lambda self, *a, **k: (None, None)  # no hier build
    plat = grc_platform.Platform(name="grc-bench", prefs=gr.prefs(), version=gr.version(),
                                 version_parts=(gr.major_version(), gr.api_version(), gr.minor_version()))
    plat._docstring_extractor = _NoDocstrings()
    original = grc_platform.Cache
    grc_platform.Cache = _FileOnlyCache
    try:
        plat.build_library(path=[str(blocks_dir)])
    finally:
        grc_platform.Cache = original
    return plat


def gnuradio_version() -> str:
    from gnuradio import gr
    return gr.version()


def catalog_hash(blocks_dir: Path = GNURADIO_BLOCKS) -> str:
    """SHA-256 over the sorted (file name, SHA-256) of every *.block.yml (same bytes as tier_f/grc.py)."""
    h = hashlib.sha256()
    for path in sorted(blocks_dir.glob("*.block.yml")):
        h.update(path.name.encode() + b"\0" + sha256_file(path).encode() + b"\n")
    return h.hexdigest()


# --------------------------------------------------------------------------- module rule


def tree_modules(blocks_dir: Path = GNURADIO_BLOCKS) -> dict[str, list[str]]:
    """{block id: sorted modules whose <module>.tree.yml lists it}, walking trees as GRC does."""
    found: dict[str, set[str]] = defaultdict(set)

    def walk(node: Any, module: str) -> None:
        if isinstance(node, str):
            found[node].add(module)
        elif isinstance(node, list):
            for item in node:
                walk(item, module)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value, module)

    for path in sorted(blocks_dir.glob("*.tree.yml")):
        with open(path, encoding="utf-8") as fh:
            walk(yaml.safe_load(fh), path.name[: -len(".tree.yml")])
    return {k: sorted(v) for k, v in sorted(found.items())}


def module_of(block_id: str, yaml_file: str | None, trees: dict[str, list[str]]) -> str:
    if block_id in trees:
        return trees[block_id][0]
    if yaml_file:
        return yaml_file[: -len(".block.yml")].split("_")[0] or "grc"
    return "grc"


# --------------------------------------------------------------------------- catalog records


def _ports(cls) -> list[dict]:
    return list(cls.inputs_data or []) + list(cls.outputs_data or [])


def _is_message(port: dict) -> bool:
    return port.get("domain") == "message"


def _templated(value: Any) -> bool:
    return isinstance(value, str) and "${" in value


def _yaml_file(cls, blocks_dir: Path) -> str | None:
    loaded = str(getattr(cls, "loaded_from", "") or "")
    if loaded.endswith(".block.yml"):
        return Path(loaded).name
    candidate = blocks_dir / f"{cls.key}.block.yml"        # built-in with a YAML twin (options)
    return candidate.name if candidate.exists() else None


def block_record(cls, trees: dict[str, list[str]], blocks_dir: Path = GNURADIO_BLOCKS) -> dict:
    """Catalog record of one GRC block class (deterministic; no GNU Radio objects inside).

    Ports are counted as declared in the YAML (before multiplicity expansion); n_params excludes the
    parameters GRC adds itself (AUTO_PARAMS). type_params are the enum parameters a templated stream
    port dtype refers to (`${ type }`, `${ type.fcn }`); has_type_param also counts an enum named
    "type". rate_params are decim/interp(olation) parameters (optionally `audio_`-prefixed) on blocks
    with a stream output, so display decimation on GUI sinks does not count. has_documentation is
    the YAML `documentation` text only."""
    ports = _ports(cls)
    stream = [p for p in ports if not _is_message(p)]
    params = [p for p in cls.parameters_data if p["id"] not in AUTO_PARAMS]
    enum_ids = {p["id"] for p in params if p.get("dtype") == "enum"}
    type_params = sorted({name for p in stream if _templated(p.get("dtype"))
                          for name in _TEMPLATE_NAME.findall(p["dtype"]) if name in enum_ids})
    has_stream_out = any(not _is_message(p) for p in cls.outputs_data or [])
    rate_params = sorted(p["id"] for p in params if _RATE_PARAM.match(p["id"])) if has_stream_out else []
    category = list(cls.category or [])
    flags = sorted(cls.flags.data)
    yaml_file = _yaml_file(cls, blocks_dir)
    return {
        "id": cls.key,
        "label": cls.label,
        "module": module_of(cls.key, yaml_file, trees),
        "category": category,
        "flags": flags,
        "builtin": str(getattr(cls, "loaded_from", "")) == "(build-in)",
        "deprecated": "deprecated" in flags or "Deprecated" in category,
        "deprecated_flag": "deprecated" in flags,
        "has_documentation": bool((cls.documentation or {}).get("", "").strip()),
        "n_params": len(params),
        "has_type_param": bool(type_params) or "type" in enum_ids,
        "type_params": type_params,
        "n_inputs": len(cls.inputs_data or []),
        "n_outputs": len(cls.outputs_data or []),
        "n_message_ports": sum(1 for p in ports if _is_message(p)),
        "literally_typed_ports": sum(1 for p in stream if p.get("dtype") and not _templated(p.get("dtype"))),
        "templated_ports": sum(1 for p in stream if _templated(p.get("dtype"))),
        "templated_multiplicity": any(_templated(p.get("multiplicity")) for p in ports),
        "has_decim_or_interp": bool(rate_params),
        "rate_params": rate_params,
        "n_asserts": len(cls.asserts or []),
        "has_value": bool(getattr(cls, "value", None)),
        "yaml_file": yaml_file,
        "yaml_sha256": sha256_file(blocks_dir / yaml_file) if yaml_file else None,
    }


def catalog(platform=None, blocks_dir: Path = GNURADIO_BLOCKS) -> list[dict]:
    """One record per loaded block, sorted by id."""
    platform = platform if platform is not None else load_platform(blocks_dir)
    trees = tree_modules(blocks_dir)
    return [block_record(platform.blocks[key], trees, blocks_dir) for key in sorted(platform.blocks)]


# --------------------------------------------------------------------------- eligibility and families


def exclusion_reasons(rec: dict) -> list[str]:
    """All reasons a catalog record is ineligible for block questions, in REASONS order."""
    reasons = []
    no_ports = rec.get("n_inputs", 0) == 0 and rec.get("n_outputs", 0) == 0
    if (rec["id"] in NON_BLOCK_IDS or rec["id"].startswith("variable") or "not_dsp" in rec.get("flags", [])
            or (rec.get("has_value") and no_ports)):
        reasons.append("non-block")
    if rec.get("deprecated"):
        reasons.append("deprecated")
    if rec.get("module") in EXCLUDED_MODULES:
        reasons.append("hardware-or-network-module")
    return reasons


def family_key(name: str) -> str:
    key = _TYPE_WORD.sub("", name.lower())
    if key.endswith("_hier"):
        key = key[: -len("_hier")]
    return _VERSION_DIGIT.sub("", key)


def assign_families(records: Iterable[dict]) -> dict[str, str]:
    """{block id: family}. Ids are linked through the key of their id and of their YAML file stem."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    ids = []
    for rec in records:
        ids.append(rec["id"])
        union("id:" + rec["id"], "key:" + family_key(rec["id"]))
        if rec.get("yaml_file"):
            union("id:" + rec["id"], "key:" + family_key(rec["yaml_file"][: -len(".block.yml")]))
    members: dict[str, list[str]] = defaultdict(list)
    for bid in ids:
        members[find("id:" + bid)].append(bid)
    out = {}
    for group in members.values():
        name = min(family_key(b) for b in group)
        for bid in group:
            out[bid] = name
    return out


# --------------------------------------------------------------------------- outputs


def _reason(rec: dict) -> str | None:
    """Every exclusion reason, comma-separated in REASONS order (None for eligible blocks)."""
    return ", ".join(rec["reasons"]) or None


def build(platform=None, blocks_dir: Path = GNURADIO_BLOCKS) -> dict[str, Any]:
    """The full catalog document (deterministic: no timestamps, sorted keys and ids)."""
    cat = catalog(platform, blocks_dir)
    families = assign_families(cat)
    for rec in cat:
        rec["family"] = families[rec["id"]]
        rec["reasons"] = exclusion_reasons(rec)
        rec["eligible"] = not rec["reasons"]
    blocks = [{"id": rec["id"], "label": rec["label"], "module": rec["module"], "family": rec["family"],
               "eligible": rec["eligible"], "reason": _reason(rec), "yaml_file": rec["yaml_file"],
               "yaml_sha256": rec["yaml_sha256"]} for rec in cat]
    modules: dict[str, dict[str, int]] = {}
    for rec in cat:
        m = modules.setdefault(rec["module"], {"blocks": 0, "eligible": 0, "excluded": 0})
        m["blocks"] += 1
        m["eligible" if rec["eligible"] else "excluded"] += 1
    reasons = Counter(r for rec in cat for r in rec["reasons"][:1])
    n_eligible = sum(rec["eligible"] for rec in cat)
    return {
        "generator": GENERATOR,
        "gnuradio_version": gnuradio_version(),
        "catalog_dir": str(blocks_dir),
        "catalog_hash": catalog_hash(blocks_dir),
        "catalog_hash_rule": "sha256 over sorted *.block.yml of: file name + NUL + sha256 hex + LF",
        "n_block_yml_files": len(list(blocks_dir.glob("*.block.yml"))),
        "rules": {
            "module": "tree file <module>.tree.yml listing the id; else first word of the YAML file "
                      "name; else 'grc' (built-ins without YAML)",
            "eligibility": {"non-block": "ids " + ", ".join(NON_BLOCK_IDS) + "; ids starting with "
                                         "'variable'; not_dsp flag; a value expression and no ports",
                            "deprecated": "flag 'deprecated' or a 'Deprecated' category",
                            "hardware-or-network-module": ", ".join(EXCLUDED_MODULES)},
            "family": "key = lower-case name minus one trailing type-signature word of letters "
                      "bcfisvx, minus a trailing _hier, minus one trailing version digit; blocks "
                      "share a family when ids or YAML file stems share a key",
        },
        "counts": {"blocks": len(cat), "eligible": n_eligible, "excluded": len(cat) - n_eligible,
                   "excluded_by_primary_reason": dict(sorted(reasons.items())),
                   "eligible_families": len({rec["family"] for rec in cat if rec["eligible"]})},
        "modules": dict(sorted(modules.items())),
        "blocks": blocks,
        "_catalog": cat,     # stripped before writing; used by the report
    }


def _code(text: str) -> str:
    return "`" + text + "`"


def report_md(doc: dict) -> str:
    cat, c = doc["_catalog"], doc["counts"]
    lines = [
        "# GNU Radio block catalog",
        "",
        "Generated by `./gb blocks` (`src/grcbench/tier_k/catalog.py`). Do not edit by hand: re-run the command.",
        "",
        f"- GNU Radio {doc['gnuradio_version']}, catalog `{doc['catalog_dir']}`: "
        f"{doc['n_block_yml_files']} `.block.yml` files, {c['blocks']} blocks loaded by GRC "
        "(the extra ones are GRC built-ins).",
        f"- Catalog hash: `{doc['catalog_hash']}`.",
        f"- Eligible: {c['eligible']} blocks in {c['eligible_families']} families; excluded: {c['excluded']}.",
        "- There is no test/train block split: training data may use any block. Tier K's GNU Radio questions "
        "are drawn from a seed, and every evaluation can draw a fresh set (`./gb blockq`, "
        "`reports/tier_k_block_questions.md`).",
        "",
        "## Exclusions (primary reason)",
        "",
        "| Reason | Blocks |",
        "|---|---:|",
    ]
    lines += [f"| {r} | {n} |" for r, n in c["excluded_by_primary_reason"].items()]
    both = sorted(r["id"] for r in cat if len(r["reasons"]) > 1)
    lines += ["", f"Blocks with more than one reason ({len(both)}; the JSON `reason` field lists all): "
              + ", ".join(_code(b) for b in both) + ".", ""]
    dep_cat = sorted(r["id"] for r in cat if r["deprecated"] and not r["deprecated_flag"]
                     and r["module"] not in EXCLUDED_MODULES)
    lines += ["Deprecated by GRC category only (no `deprecated` flag), excluded by our rule: "
              + (", ".join(_code(b) for b in dep_cat) or "none") + ".", ""]
    lines += ["## Per module", "", "| Module | Blocks | Eligible | Excluded |", "|---|---:|---:|---:|"]
    for mod, m in doc["modules"].items():
        lines.append(f"| {mod} | {m['blocks']} | {m['eligible']} | {m['excluded']} |")
    feats = [("has_type_param", "type parameter (port dtype set by an enum)"),
             ("templated_ports", "at least one templated port"),
             ("literally_typed_ports", "at least one literally typed port"),
             ("n_message_ports", "message ports"),
             ("has_decim_or_interp", "decimation/interpolation parameter"),
             ("n_asserts", "assert constraints"),
             ("templated_multiplicity", "port count set by a parameter"),
             ("has_documentation", "YAML documentation text")]
    lines += ["", "## Features of the eligible blocks", "", "| Feature | Blocks |", "|---|---:|"]
    for key, label in feats:
        lines.append(f"| {label} | {sum(1 for r in cat if r['eligible'] and r[key])} |")
    multi: dict[str, list[dict]] = defaultdict(list)
    for r in cat:
        if r["eligible"]:
            multi[r["family"]].append(r)
    multi = {f: rs for f, rs in sorted(multi.items()) if len(rs) > 1}
    lines += ["", f"## Families with more than one eligible block ({len(multi)})", "",
              "| Family | Blocks |", "|---|---|"]
    for fam, rs in multi.items():
        lines.append(f"| {fam} | " + ", ".join(_code(r["id"]) for r in rs) + " |")
    lines += ["", "## Notes", "",
              "- `video_sdl_sink` (needs an SDL display) and `xmlrpc_server`/`xmlrpc_client` (open a "
              "socket; listed in the blocks tree) are not in the excluded module list and stay eligible."]
    return "\n".join(lines) + "\n"


CATALOG_JSON = "reports/block_catalog.json"
CATALOG_MD = "reports/block_catalog.md"


def public_doc(doc: dict) -> dict:
    """The catalog document as written (internal `_` keys removed)."""
    return {k: v for k, v in doc.items() if not k.startswith("_")}


def write_outputs(doc: dict, root: Path = REPO) -> list[Path]:
    """Write the catalog JSON and its report under `root`; returns the paths (JSON first)."""
    written = [root / CATALOG_JSON, root / CATALOG_MD]
    write_json(written[0], public_doc(doc))
    write_text(written[1], report_md(doc))
    return written


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb blocks", description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="rebuild in a temporary folder and fail if any written file differs")
    args = ap.parse_args(argv)
    doc = build()
    if args.check:
        import tempfile
        stale = []
        with tempfile.TemporaryDirectory() as tmp:
            for fresh in write_outputs(doc, Path(tmp)):
                rel = fresh.relative_to(tmp)
                current = REPO / rel
                if not current.exists() or current.read_bytes() != fresh.read_bytes():
                    stale.append(str(rel))
        print("block catalog outputs up to date" if not stale else "differs from a fresh build: "
              + ", ".join(stale))
        return 1 if stale else 0
    for path in write_outputs(doc):
        print(f"wrote {path.relative_to(REPO)}")
    c = doc["counts"]
    print(f"{c['blocks']} blocks; {c['eligible']} eligible in {c['eligible_families']} families; "
          f"{c['excluded']} excluded; catalog_hash {doc['catalog_hash'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
