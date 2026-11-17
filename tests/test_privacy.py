"""Published files name no person's user folder (common/privacy.py).

The example paths are built at run time, so that this file holds no user folder itself.
"""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

import pytest

from grcbench.common import io, paths, privacy
from grcbench.common.paths import REPO

NAME = "al" + "ice"                                 # an example user name, never written out whole
WIN = "C:\\" + "Users\\" + NAME
MNT = "/mnt/c/" + "Users/" + NAME
HOME = "/home/" + NAME


@pytest.mark.parametrize("text", [
    WIN + "\\Documents\\x.md",
    WIN.replace("\\", "\\\\") + "\\\\AppData\\\\Local\\\\t.log",      # as JSON escapes it
    WIN.replace("\\", "/") + "/x",
    MNT + "/Docs/x.py",
    HOME + "/grc-agent/x.py",
    "claude\\C--Users-" + NAME + "-Documents-proj\\scratchpad",
    "/c/" + "Users/" + NAME + "/Documents/x.md",                     # Git Bash
])
def test_a_user_folder_is_found_and_redacted(text):
    assert privacy.private_paths(text)
    out = privacy.redact(text, project_root=PurePosixPath("/mnt/c/Users/someone/proj"),
                         home=PurePosixPath("/home/someone"))
    assert privacy.private_paths(out) == [] and NAME not in out


@pytest.mark.parametrize("text", [
    "/home/user/project/out.dat",                    # the Tier-A sandbox
    "C:\\Users\\someone\\x", "/home/you/grc-agent", "/home/u/proj/beat.dat",
    "<home>\\AppData", "~/grc-bench-cache", "<project>/grc-bench", "claude\\C--Users-<user>-Documents",
])
def test_sandbox_and_example_paths_are_not_private(text):
    assert privacy.private_paths(text) == []


def test_a_claude_scratch_folder_becomes_scratch():
    """A session's scratch folder is a path on this machine even with its user redacted (code review a37)."""
    temp = "\\".join(["App" + "Data", "Local", "Temp", "cl" + "aude"])          # built at run time, as above
    text = "<home>\\" + temp + "\\C--Users-<user>-Documents-proj\\0123-abcd\\scratchpad\\x.log"
    assert privacy.private_paths(text)
    out = privacy.redact(text, project_root=PurePosixPath("/mnt/c/Users/someone/proj"),
                         home=PurePosixPath("/home/someone"))
    assert out == "<scratch>\\scratchpad\\x.log" and privacy.private_paths(out) == []
    escaped = text.replace("\\", "\\\\")                                # as JSON escapes it
    assert privacy.redact(escaped, project_root=PurePosixPath("/mnt/c/Users/someone/proj"),
                          home=PurePosixPath("/home/someone")) == "<scratch>\\\\scratchpad\\\\x.log"


def test_a_sheet_generated_before_any_decision_is_redacted(tmp_path):
    from grcbench.review import fill
    generated = "task_id,instruction\ngbf-test-1,see " + MNT + "/x.grc\n"
    out = fill.with_new_rows("tier_f", generated, reviews_dir=tmp_path / "none", path=tmp_path / "s.csv")
    assert NAME not in out and "<home>/x.grc" in out


def test_the_project_folder_of_another_machine_becomes_project():
    root = PurePosixPath("/mnt/c/Users/someone/Docs/proj")
    for user_folder in (WIN, MNT):
        sep = "\\" if user_folder.startswith("C:") else "/"
        text = sep.join([user_folder, "Docs", "proj", "grc-bench", "items"])
        assert privacy.redact(text, project_root=root, home=PurePosixPath("/home/someone")) == \
            sep.join(["<project>", "grc-bench", "items"])


def test_reports_are_redacted_when_written(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REPORTS", tmp_path / "reports")
    io.write_text(tmp_path / "reports" / "r.md", f"model {HOME}/m.gguf")
    io.write_json(tmp_path / "reports" / "r.json", {"path": f"{HOME}/m.gguf"})
    io.write_jsonl(tmp_path / "reports" / "r.jsonl", [{"path": f"{HOME}/m.gguf"}])
    io.write_text(tmp_path / "work.md", f"model {HOME}/m.gguf")               # not a report: kept
    for name in ("r.md", "r.json", "r.jsonl"):
        text = (tmp_path / "reports" / name).read_text(encoding="utf-8")
        assert NAME not in text and "~/m.gguf" in text
    assert HOME in (tmp_path / "work.md").read_text(encoding="utf-8")


def test_no_tracked_file_names_a_user_folder():
    try:
        files = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True,
                               text=True).stdout.split("\0")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not available")
    found = {}
    for f in filter(None, files):
        p = Path(REPO / f)
        if not p.is_file():
            continue
        data = p.read_bytes()
        if b"\0" in data:
            continue
        if hits := privacy.private_paths(data.decode("utf-8", errors="replace")):
            found[f] = sorted(set(hits))[:3]
    assert found == {}
