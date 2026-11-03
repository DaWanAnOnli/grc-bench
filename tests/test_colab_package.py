"""Tests for `./gb colab-package`: the zip that runs `./gb` on a Colab GPU holds the dev split and never the
test split."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from grcbench.common import colab_package

U2604 = colab_package.COLAB_HANDOFF / "u2604.py"


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    return colab_package.build(tmp_path_factory.mktemp("colab"))


def test_the_package_runs_gb_on_the_dev_split(package):
    names = set(zipfile.ZipFile(package).namelist())
    root = colab_package.NAME + "/"
    for rel in ("gb", "src/grcbench/__main__.py", "items/tier_k/dev/questions.jsonl", "reports/block_catalog.json",
                "docs/preregistration/run_list.yaml", colab_package.RUBRICS, colab_package.NOTEBOOK, "colab/u2604.py",
                "VERSION.txt"):
        assert root + rel in names, rel
    assert any(n.startswith(root + "items/tier_f/dev/") for n in names)
    assert any(n.startswith(root + "items/tier_a/dev/") for n in names)


def test_the_package_can_check_the_sandbox_inside_a_tier_a_episode(package):
    """The notebook's step 7b runs GRC-Bench's own sandbox tests and the golden replays of the fixture tasks
    (shell commands and flowgraph runs inside the episode sandbox) on Colab; MODEL_STEPS=False makes a short
    run of the checks without llama.cpp's build and the model runs."""
    with zipfile.ZipFile(package) as z:
        names = set(z.namelist())
        notebook = z.read(f"{colab_package.NAME}/{colab_package.NOTEBOOK}").decode()
    root = colab_package.NAME + "/"
    for rel in colab_package.SANDBOX_TESTS:
        assert root + rel in names, rel
    assert root + "tests/fixtures/tier_a/tasks/fx-a-modify-tone/golden_calls.json" in names
    assert root + "tests/fixtures/tier_f_task/fx_modify_tone/task.json" in names
    assert "tier-a smoke --no-probe" in notebook and "MODEL_STEPS" in notebook and "7b_episode_sandbox" in notebook


def test_the_package_holds_no_test_item_and_no_bytecode(package):
    names = zipfile.ZipFile(package).namelist()
    assert not [n for n in names if "/test" in n.split(colab_package.NAME, 1)[1] and "/items/" in n]
    assert not [n for n in names if "__pycache__" in n or n.endswith(".pyc")]


def test_the_rubric_file_is_the_placeholder_set_for_the_dev_split(package):
    with zipfile.ZipFile(package) as z:
        rows = [json.loads(line) for line in z.read(f"{colab_package.NAME}/{colab_package.RUBRICS}").splitlines()]
    assert rows and {r["review"]["status"] for r in rows} == {"placeholder"} and {r["split"] for r in rows} == {"dev"}


def test_a_file_quoting_a_test_question_is_refused():
    test_text = colab_package.test_texts()[0]
    with pytest.raises(colab_package.PackageRefused, match="test item"):
        colab_package.check_content({"notes.md": f"see: {test_text}".encode()}, [test_text])


def test_a_test_instruction_split_across_source_literals_is_refused():
    """Builder code writes an instruction as several adjacent string literals, so the whole text never appears
    in one piece in the file; the parser joins them, and the joined string is searched too."""
    text = next(t for t in colab_package.test_texts() if len(t) > 120)
    source = f'INSTRUCTION = ("{text[:70]}"\n               "{text[70:]}")\n'
    with pytest.raises(colab_package.PackageRefused, match="test item"):
        colab_package.check_content({"src/grcbench/x.py": source.encode()}, [text])


def test_the_modules_that_build_test_tasks_are_not_packaged(package):
    names = set(zipfile.ZipFile(package).namelist())
    for module in colab_package.TEST_CONTENT_MODULES:
        assert f"{colab_package.NAME}/{module}" not in names, module
    assert {"src/grcbench/tier_a/first_set.py", "src/grcbench/tier_a/spec_drafts.py",
            "src/grcbench/tier_a/second_set.py"} <= set(colab_package.TEST_CONTENT_MODULES)


def _u2604():
    spec = importlib.util.spec_from_file_location("u2604", U2604)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_u2604_binds_the_machines_proc_and_enters_with_pivot_root():
    """Colab refuses a fresh /proc, and Linux refuses a user namespace (GRC-Bench's sandbox on Colab) to a chrooted
    process: the container binds the machine's /proc and enters its root with pivot_root."""
    script = _u2604().inside_script("/opt/u", [("/content", "/content")])
    assert "mount --rbind /proc /opt/u/proc" in script and "mount --rbind /content /opt/u/content" in script
    assert "chroot" not in script and "--mount-proc" not in script


@pytest.mark.skipif(not shutil.which("bwrap") or not shutil.which("unshare"), reason="bubblewrap or unshare missing")
def test_u2604_runs_a_sandbox_with_a_user_namespace_of_its_own_inside(tmp_path):
    root = tmp_path / "rootfs"
    for d in ("etc", "tmp", "usr"):
        (root / d).mkdir(parents=True)
    for link in ("bin", "sbin", "lib", "lib64"):
        if Path("/" + link).is_symlink():
            (root / link).symlink_to(os.readlink("/" + link))
    shutil.copy("/etc/os-release", root / "etc" / "os-release")
    cmd = ("test -e /proc/self/status && echo proc; "
           "bwrap --ro-bind / / --dev /dev --tmpfs /tmp --unshare-user --unshare-net --unshare-pid -- true && echo nested")
    res = subprocess.run([sys.executable, str(U2604), str(root), "--bind", "/usr", "--", cmd],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0 and res.stdout.split() == ["proc", "nested"], res.stdout + res.stderr


def test_another_notebook_runs_the_same_steps_with_its_own_model_server(package):
    """The team's evaluation notebook (grc-model-eval's colab/eval.ipynb) runs GRC-Bench's steps as one script,
    made from this notebook's own code cells, with the model server it already runs: the tiers are served by
    that server, Tier E measures its model file, and the results go to its run folder."""
    with zipfile.ZipFile(package) as z:
        script = z.read(f"{colab_package.NAME}/{colab_package.STEPS}").decode()
        notebook = json.loads(z.read(f"{colab_package.NAME}/{colab_package.NOTEBOOK}"))
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert cells[0].startswith("#@title Settings")
    for cell in cells[1:]:                           # every code cell but the Settings form, word for word
        assert cell in script
    compile(script, "grcbench_colab_steps.py", "exec")
    for name in ("SERVER_URL", "SERVER_MODEL", "MODEL_FILE", "RESULTS_DIR", "--base-url", "--gguf", "fetch --cuda"):
        assert name in script, name


def test_the_notebook_downloads_the_pinned_cuda_build():
    from grcbench.common.paths import LLAMA_CPP_BUILD
    nb = json.loads((colab_package.COLAB_HANDOFF / "grcbench_colab_test.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "fetch --cuda" in text and f'LLAMA_BUILD = "{LLAMA_CPP_BUILD}"' in text and "{LLAMA_BUILD}-cuda" in text
    assert "b10333" not in text and "CUDA_ARCH" not in text and "REUSE_LLAMA_BUILD" not in text


def test_with_another_notebooks_server_tier_e_runs_only_after_that_server_stops():
    """Tier E measures with no other llama.cpp process running: with SERVER_URL, step 8 runs Tiers K, F and A,
    calls STOP_SERVER, then Tier E alone; without STOP_SERVER it leaves Tier E out (and step 9 with it)."""
    text = (colab_package.COLAB_HANDOFF / "grcbench_colab_steps.py").read_text(encoding="utf-8")
    e2e_step = text[text.index("def e2e():"):text.index('step("8_e2e"')]
    speeds_step = text[text.index("def speeds():"):text.index('step("9_speeds"')]
    assert e2e_step.index("--only {','.join(served)}") < e2e_step.index("STOP_SERVER()") < e2e_step.index("--only E")
    assert "if SERVER_URL and \"E\" in tiers and not STOP_SERVER:" in e2e_step
    assert '"E" not in RESULT["steps"].get("8_e2e", {}).get("tiers_run", [])' in speeds_step
