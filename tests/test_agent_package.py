"""`./gb agent-package` (tier_a/agent_package.py): the teammate's zip holds the runner and nothing of the benchmark."""

from __future__ import annotations

import hashlib
import shutil
import uuid
import zipfile

import pytest

from grcbench.common.paths import ITEMS, WORK
from grcbench.runlog.log import benchmark_item_ids
from grcbench.tier_a import agent_package as pkg


@pytest.fixture
def out():
    path = WORK / "pytest-package" / uuid.uuid4().hex[:10]
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_the_closure_holds_the_runner_and_no_content_module():
    files = pkg.closure()
    for needed in ("grcbench/tier_a/adhoc.py", "grcbench/tier_a/runner.py", "grcbench/tier_a/headless_proxy.py",
                   "grcbench/tier_a/relay.py", "grcbench/tier_a/_launch.py", "grcbench/common/sandbox.py",
                   "grcbench/common/environment.py", "grcbench/__main__.py"):
        assert needed in files
    assert not [f for f in files if any(f.startswith(c) for c in pkg.CONTENT_MODULES)]


def test_the_zip(out):
    target = pkg.build(out)
    with zipfile.ZipFile(target) as zf:
        names = {i.filename: i for i in zf.infolist()}
        data = {n: zf.read(n) for n in names}
    top = pkg.NAME + "/"
    assert all(n.startswith(top) for n in names)
    for rel in (*pkg.STATIC_FILES, "VERSION.txt", "src/grcbench/tier_a/adhoc.py"):
        assert top + rel in names
    assert not [n for n in names if "/items/" in n or "/reports/" in n or n.endswith("first_set.py")]
    for rel in pkg.EXECUTABLE:
        assert (names[top + rel].external_attr >> 16) & 0o111                     # stays executable
    version = data[top + "VERSION.txt"].decode()
    for n, blob in data.items():
        if not n.endswith("VERSION.txt"):
            assert f"{hashlib.sha256(blob).hexdigest()}  {n[len(top):]}" in version
    ids = set(benchmark_item_ids(ITEMS))
    assert ids and not [n for n, blob in data.items() if set(pkg._TOKEN.findall(blob.decode("utf-8", "ignore"))) & ids]


def test_the_guards_refuse_benchmark_content():
    some_id = sorted(benchmark_item_ids(ITEMS))[0]
    with pytest.raises(pkg.PackageError):
        pkg.check_content({"src/grcbench/tier_a/first_set.py": b"x"}, set())
    with pytest.raises(pkg.PackageError):
        pkg.check_content({"TUTORIAL.md": f"see {some_id} for an example".encode()}, {some_id})
    pkg.check_content({"TUTORIAL.md": b"nothing to see"}, {some_id})
