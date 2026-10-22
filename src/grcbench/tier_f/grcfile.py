"""The .grc file format: rung V1 (parse + schema), answer extraction, and the compact form.

V1 (proposal Table 5): "YAML parses and conforms to the .grc schema". A .grc file is YAML with
top-level `options`, `blocks`, `connections` and `metadata.file_format` (GNU Radio 3.10 uses
file_format 1). Each block has `name`, `id` and `parameters`; each connection is
[src_block, src_port, dst_block, dst_port].

The compact form drops GUI coordinates (`states`) and parameters left at their defaults, so that a
complete flowgraph fits the 8K-token Tier-F answer cap (eval plan §7.1). GRC fills in the defaults
when it loads the file; tests check that a compacted file loads to the same flowgraph.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)
THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(answer: str) -> str:
    """Remove reasoning ("thinking") text before scoring (eval plan §7.1)."""
    return THINK.sub("", answer)


def extract_grc(answer: str) -> str | None:
    """Answer-extraction rule for Tier F (pre-registered).

    1. Strip reasoning text.
    2. Take the first fenced code block whose language tag is yaml, yml or grc, or is untagged,
       and whose content parses as a YAML mapping with a `blocks` key.
    3. Otherwise, if the whole remaining answer parses as such a mapping, use it.
    4. Otherwise there is no answer (the item fails V1).
    """
    text = strip_reasoning(answer)
    for lang, body in FENCE.findall(text):
        if lang.lower() not in {"", "yaml", "yml", "grc"}:
            continue
        if _looks_like_grc(body):
            return body
    if _looks_like_grc(text):
        return text
    return None


def _looks_like_grc(text: str) -> bool:
    try:
        data = yaml.safe_load(text)
    except Exception:
        return False
    return isinstance(data, dict) and "blocks" in data


def v1_parse(text: str) -> tuple[bool, list[str], dict | None]:
    """Rung V1. Returns (passed, errors, parsed data)."""
    errors: list[str] = []
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return False, [f"YAML does not parse: {exc}".splitlines()[0]], None
    if not isinstance(data, dict):
        return False, ["top level is not a mapping"], None
    for key in ("options", "blocks", "connections", "metadata"):
        if key not in data:
            errors.append(f"missing top-level key '{key}'")
    opts = data.get("options")
    if not isinstance(opts, dict) or not isinstance(opts.get("parameters"), dict):
        errors.append("options.parameters must be a mapping")
    blocks = data.get("blocks")
    if not isinstance(blocks, list):
        errors.append("blocks must be a list")
        blocks = []
    for i, blk in enumerate(blocks):
        if not isinstance(blk, dict):
            errors.append(f"blocks[{i}] is not a mapping")
            continue
        for key in ("name", "id"):
            if not isinstance(blk.get(key), str) or not blk.get(key):
                errors.append(f"blocks[{i}] has no '{key}'")
        if "parameters" in blk and not isinstance(blk["parameters"], dict):
            errors.append(f"blocks[{i}].parameters is not a mapping")
        # duplicate names are left to GRC's own validation (V3): a disabled duplicate is legal
    conns = data.get("connections")
    if conns is None:
        conns = []
    if not isinstance(conns, list):
        errors.append("connections must be a list")
        conns = []
    for i, con in enumerate(conns):
        if not (isinstance(con, list) and len(con) == 4):
            errors.append(f"connections[{i}] is not [src, src_port, dst, dst_port]")
    meta = data.get("metadata")
    if not isinstance(meta, dict) or meta.get("file_format") != 1:
        errors.append("metadata.file_format must be 1 (GNU Radio 3.10)")
    return (not errors), errors, data


def dump(data: dict[str, Any]) -> str:
    """Serialise a flowgraph dict as .grc YAML (block style, stable key order)."""
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=100, allow_unicode=True)


def compact(data: dict[str, Any], defaults: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    """Drop `states` and parameters equal to their catalog defaults (defaults: block id -> {param: default})."""
    defaults = defaults or {}
    out = {"options": {"parameters": dict(data.get("options", {}).get("parameters", {}))},
           "blocks": [], "connections": [list(c) for c in data.get("connections", []) or []],
           "metadata": {"file_format": 1}}
    for blk in data.get("blocks", []) or []:
        dflt = defaults.get(blk.get("id"), {})
        params = {k: v for k, v in (blk.get("parameters") or {}).items()
                  if k not in dflt or str(v) != str(dflt[k])}
        new = {"name": blk["name"], "id": blk["id"], "parameters": params}
        state = (blk.get("states") or {}).get("state")
        if state and state != "enabled":
            new["states"] = {"state": state}
        out["blocks"].append(new)
    return out
