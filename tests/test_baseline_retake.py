"""`gb baseline take --root`: only the named roots' manifests are replaced."""

import pytest

from grcbench.common import baseline


@pytest.fixture
def roots(tmp_path, monkeypatch):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "f.txt").write_text("one", encoding="utf-8")
    (tmp_path / "proj").mkdir()
    monkeypatch.setattr(baseline, "ROOTS", {"a": tmp_path / "a", "b": tmp_path / "b"})
    monkeypatch.setattr(baseline, "PROJECT", tmp_path / "proj")
    monkeypatch.setattr(baseline, "ROOT_FILES", [])
    monkeypatch.setattr(baseline, "BASE_DIR", tmp_path / "base")
    baseline.take()
    return tmp_path


def test_retake_replaces_only_the_named_root(roots, capsys):
    for name in ("a", "b"):
        (roots / name / "f.txt").write_text("two", encoding="utf-8")
    assert baseline.main(["take", "--root", "a"]) == 0
    capsys.readouterr()
    assert baseline.check() == 1
    out = capsys.readouterr().out
    assert "a" + " " * 39 + " unchanged" in out
    assert "b" + " " * 39 + " CHANGED (1 files)" in out


def test_retake_refuses_unknown_or_missing_names(roots):
    with pytest.raises(SystemExit, match="unknown root"):
        baseline.main(["take", "--root", "a,nope"])
    with pytest.raises(SystemExit, match="none given"):
        baseline.main(["take", "--root"])
