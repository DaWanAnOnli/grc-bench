"""The training team's data release that GRC-Bench checks against: Qoherent cleaned dataset v3.

The training team prepared `project/cleaned_data/qoherent_v3_2026-11-01.zip` (1 Nov 2026): train,
validation and test splits of the Q&A questions and agent traces, in a model-neutral conversational
format. GRC-Bench reads it and never writes it:

  * the Tier-K dev split is drawn from its validation questions (splits/dev.py, dev split v2);
  * `./gb audit` checks that no Tier-K dev or test question, and no canary, is in its training files;
  * `./gb placeholder-rubrics` copies its validation reference answers into placeholder rubrics for a
    pipeline test (tier_k/placeholder_rubrics.py).

The archive is read in place with zipfile (nothing is extracted) and pinned by its SHA-256, the value
in the release's own checksum file. Q&A question ids come from the metadata files, which are aligned
row by row with the data files and name each row's `kind` and `question_id`.
"""

from __future__ import annotations

import functools
import json
import zipfile
from pathlib import Path

from grcbench.common.io import sha256_file
from grcbench.common.paths import PROJECT

RELEASE = PROJECT / "cleaned_data" / "qoherent_v3_2026-11-01.zip"
RELEASE_SHA256 = "c0705e6b8b5f15a9ba845f686214f78e90ff67fc21d22132513a237c0121a50b"
ROOT = "qoherent_v3/"
VALIDATION_METADATA = "metadata/validation.jsonl"
VALIDATION_QUESTIONS = "evaluation/questions_validation.jsonl"   # question, reference_answer, reference_model
TRAINING_METADATA = ("metadata/train.jsonl", "metadata/optional__qna_all_answers_train.jsonl",
                     "metadata/optional__recovery_train.jsonl")
TRAINING_FILES = ("train.jsonl", "optional/qna_all_answers_train.jsonl", "optional/recovery_train.jsonl")


class ReleaseError(SystemExit):
    """The release is missing or is not the pinned archive."""


@functools.lru_cache(maxsize=4)
def _checked(path: Path, sha256: str) -> Path:
    if not path.exists():
        raise ReleaseError(f"training release not found: {path}")
    got = sha256_file(path)
    if got != sha256:
        raise ReleaseError(f"{path.name}: SHA-256 {got} is not the pinned {sha256}")
    return path


def _lines(path: Path, member: str):
    with zipfile.ZipFile(path) as z, z.open(ROOT + member) as f:
        for raw in f:
            if raw.strip():
                yield json.loads(raw)


def _qna_ids(path: Path, member: str) -> list[str]:
    seen: dict[str, None] = {}
    for row in _lines(path, member):
        if row.get("kind") == "qna" and row.get("question_id"):
            seen.setdefault(row["question_id"])
    return list(seen)


def validation_question_ids(path: Path = RELEASE, sha256: str = RELEASE_SHA256) -> list[str]:
    """Q&A question ids of the validation split, in file order."""
    return _qna_ids(_checked(path, sha256), VALIDATION_METADATA)


def validation_reference_answers(path: Path = RELEASE, sha256: str = RELEASE_SHA256) -> dict[str, dict[str, str]]:
    """Question id -> {question, reference_answer, reference_model} of the validation questions."""
    return {r["id"]: {k: r.get(k) for k in ("question", "reference_answer", "reference_model")}
            for r in _lines(_checked(path, sha256), VALIDATION_QUESTIONS)}


def training_question_ids(path: Path = RELEASE, sha256: str = RELEASE_SHA256) -> set[str]:
    """Q&A question ids in any training export (the default file and the optional ones)."""
    p = _checked(path, sha256)
    return {q for member in TRAINING_METADATA for q in _qna_ids(p, member)}


def training_files_containing(text: str, path: Path = RELEASE, sha256: str = RELEASE_SHA256) -> list[str]:
    """The training files (relative to the release root) whose bytes contain `text`."""
    p, needle = _checked(path, sha256), text.encode("utf-8")
    with zipfile.ZipFile(p) as z:
        return [m for m in TRAINING_FILES if needle in z.read(ROOT + m)]


def describe(path: Path = RELEASE) -> dict[str, object]:
    """What a manifest records about the release (no file access)."""
    return {"file": path.relative_to(PROJECT).as_posix(), "sha256": RELEASE_SHA256,
            "validation": ROOT + VALIDATION_METADATA,
            "training": [ROOT + m for m in TRAINING_METADATA]}
