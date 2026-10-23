"""JSON schemas for every benchmark item, checked by `./gb audit` and the tests.

Every item carries the fields the evaluation plan needs for release and analysis:
  canary      the GRC-Bench canary string (eval plan §6, §15); never shown to a model under test
  license     licence of the item's source material (eval plan §14: releases respect source licences)
  source      where the item comes from (frozen question, block catalog, GNU Radio example, ...)
  cluster_id  items that are not independent share a cluster (eval plan §9: clustered intervals)
  provenance  who or what drafted the item (eval plan §5.1 role table; Claude-drafted items say so)
  review      team review status (the team reviews, edits and decides on every drafted item)
"""

from __future__ import annotations

from typing import Any

import jsonschema

from .canary import CANARY

REVIEW_STATUSES = ["pending-team-review", "approved", "approved-with-edits", "rejected", "not-required"]
DRAFTERS_NOTE = "Claude (Anthropic)"   # value used in provenance.drafted_by for Claude-drafted text

BASE: dict[str, Any] = {
    "type": "object",
    "required": ["id", "tier", "split", "version", "canary", "license", "source", "cluster_id",
                 "provenance", "review"],
    "properties": {
        "id": {"type": "string", "minLength": 3},
        "tier": {"enum": ["K", "F", "A"]},
        "split": {"enum": ["test", "dev"]},
        "version": {"type": "integer", "minimum": 1},
        "canary": {"const": CANARY},
        "license": {"type": "string", "minLength": 2},
        "source": {
            "type": "object",
            "required": ["kind"],
            "properties": {"kind": {"type": "string"}, "ref": {"type": "string"},
                           "sha256": {"type": "string"}},
        },
        "cluster_id": {"type": "string", "minLength": 1},
        "provenance": {
            "type": "object",
            "required": ["drafted_by", "generator"],
            "properties": {"drafted_by": {"type": "string"}, "generator": {"type": "string"},
                           "generator_version": {"type": "string"}},
        },
        "review": {
            "type": "object",
            "required": ["status"],
            "properties": {"status": {"enum": REVIEW_STATUSES},
                           "reviewers": {"type": "array", "items": {"type": "string"}},
                           "notes": {"type": "string"}},
        },
    },
}

TIER_K = {
    "allOf": [BASE, {
        "type": "object",
        "required": ["question", "task_type", "difficulty", "item_kind"],
        "properties": {
            "tier": {"const": "K"},
            "question": {"type": "string", "minLength": 5},
            "task_type": {"enum": ["factual", "reasoning", "calculation", "design", "diagnosis",
                                   "gnuradio_usage"]},
            "difficulty": {"enum": ["basic", "intermediate", "advanced"]},
            "item_kind": {"enum": ["frozen_question", "bank_question", "block_question"]},
        },
    }],
}

# V5 oracle vocabulary (must equal grcbench.tier_f.oracles.FEATURES and graph_checks.KINDS; a test
# checks this). A typo in a feature or check kind is a schema error at taskio.save time instead of a
# check that silently fails at V5.
TIER_F_FEATURES = ["peak_freq", "band_fraction", "rms", "mean", "max_abs", "min", "max", "corr_ref",
                   "equal_prefix", "max_int_dev", "int_fraction"]
TIER_F_GRAPH_CHECKS = ["block_present", "block_absent", "path", "same_blocks_as_start",
                       "same_connections_as_start", "max_functional_blocks"]
TIER_F_ORACLE = {
    "type": ["object", "null"],
    "properties": {
        "checks": {"type": "array", "items": {
            "type": "object", "required": ["point", "feature"],
            "properties": {"point": {"type": "string"}, "feature": {"enum": TIER_F_FEATURES}}}},
        "graph_checks": {"type": "array", "items": {
            "type": "object", "required": ["kind"],
            "properties": {"kind": {"enum": TIER_F_GRAPH_CHECKS}}}},
    },
}

TIER_F = {
    "allOf": [BASE, {
        "type": "object",
        "required": ["family", "level", "instruction", "reference_grc", "observation"],
        "properties": {
            "tier": {"const": "F"},
            "family": {"enum": ["build", "modify", "repair", "parameter-tune"]},
            "level": {"enum": ["L1", "L2", "L3"]},
            "instruction": {"type": "string", "minLength": 20},
            "reference_grc": {"type": "string"},
            "start_grc": {"type": ["string", "null"]},
            "observation": {"type": "object"},
            "oracle": TIER_F_ORACLE,
        },
    }],
}

TIER_A = {
    "allOf": [BASE, {
        "type": "object",
        "required": ["category", "instruction", "start_state", "acceptance", "scripted_user"],
        "properties": {
            "tier": {"const": "A"},
            "category": {"enum": ["build", "repair", "param", "epy", "modify", "knowledge",
                                  "save_block", "file_ops", "run_verify", "refusal"]},
            "instruction": {"type": "string", "minLength": 20},
            "start_state": {"type": "object"},
            "scripted_user": {"type": "object"},
            "acceptance": {"type": "object"},
        },
    }],
}

BY_TIER = {"K": TIER_K, "F": TIER_F, "A": TIER_A}


def validate(record: dict[str, Any]) -> list[str]:
    """Return a list of human-readable schema errors (empty if the record is valid)."""
    schema = BY_TIER.get(record.get("tier"), BASE)
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
            for e in sorted(validator.iter_errors(record), key=lambda e: list(e.absolute_path))]
