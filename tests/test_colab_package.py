"""Tests for `./gb colab-package`: the zip that runs `./gb` on a Colab GPU holds the dev split and never the
test split."""

from __future__ import annotations

import json
import zipfile

import pytest

from grcbench.common import colab_package


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
