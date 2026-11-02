"""The training team's data release (splits.release): read in place, pinned, question ids by split."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from grcbench.common.canary import CANARY_GUID
from grcbench.splits import release


def _meta(rows: list[tuple[str, str | None]]) -> str:
    return "".join(json.dumps({"id": f"{k}_{q or i}", "kind": k, "question_id": q}) + "\n"
                   for i, (k, q) in enumerate(rows))


def _zip(tmp_path: Path, train_text: str = '{"id": "a"}\n') -> tuple[Path, str]:
    path = tmp_path / "release.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(release.ROOT + "metadata/train.jsonl", _meta([("qna", "q_a"), ("trace", None), ("qna", "q_b")]))
        z.writestr(release.ROOT + "metadata/optional__qna_all_answers_train.jsonl",
                   _meta([("qna", "q_a"), ("qna", "q_c")]))
        z.writestr(release.ROOT + "metadata/optional__recovery_train.jsonl", _meta([("trace", None)]))
        z.writestr(release.ROOT + "metadata/validation.jsonl",
                   _meta([("qna", "q_v2"), ("qna", "q_v1"), ("qna", "q_v2"), ("trace", None)]))
        z.writestr(release.ROOT + "train.jsonl", train_text)
        z.writestr(release.ROOT + "optional/qna_all_answers_train.jsonl", '{"id": "b"}\n')
        z.writestr(release.ROOT + "optional/recovery_train.jsonl", '{"id": "c"}\n')
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_validation_ids_are_qna_only_in_file_order_without_repeats(tmp_path: Path):
    path, sha = _zip(tmp_path)
    assert release.validation_question_ids(path, sha) == ["q_v2", "q_v1"]


def test_training_ids_cover_every_training_export(tmp_path: Path):
    path, sha = _zip(tmp_path)
    assert release.training_question_ids(path, sha) == {"q_a", "q_b", "q_c"}


def test_a_changed_or_missing_archive_is_refused(tmp_path: Path):
    path, _ = _zip(tmp_path)
    with pytest.raises(release.ReleaseError, match="SHA-256"):
        release.validation_question_ids(path, "0" * 64)
    with pytest.raises(release.ReleaseError, match="not found"):
        release.validation_question_ids(tmp_path / "absent.zip", "0" * 64)


def test_training_files_containing_finds_the_canary(tmp_path: Path):
    path, sha = _zip(tmp_path, train_text=json.dumps({"id": "a", "text": f"x {CANARY_GUID} y"}) + "\n")
    assert release.training_files_containing(CANARY_GUID, path, sha) == ["train.jsonl"]
    (tmp_path / "clean").mkdir()
    clean, sha2 = _zip(tmp_path / "clean")
    assert release.training_files_containing(CANARY_GUID, clean, sha2) == []


def test_describe_names_the_pinned_archive():
    d = release.describe()
    assert d["file"] == "cleaned_data/qoherent_v3_2026-11-01.zip"
    assert d["sha256"] == release.RELEASE_SHA256 == (
        "c0705e6b8b5f15a9ba845f686214f78e90ff67fc21d22132513a237c0121a50b")
    assert d["validation"] == "qoherent_v3/metadata/validation.jsonl"


@pytest.mark.skipif(not release.RELEASE.exists(), reason="the training release is not on this machine")
def test_the_real_release_matches_its_published_counts():
    val = release.validation_question_ids()
    assert len(val) == 148
    assert not set(val) & release.training_question_ids()
