"""Tests for the block catalog and the test/train block partition (plan WS3a; eval plan §4.2, §6)."""

from __future__ import annotations

import csv
import hashlib
import io
import random
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import pytest

from grcbench.common.paths import GNURADIO_BLOCKS, REPO
from grcbench.tier_k import catalog as C

# --------------------------------------------------------------------------- pure functions


def test_family_key_strips_type_word_hier_and_version_digit():
    cases = {
        "blocks_add_xx": "blocks_add",
        "blocks_multiply_const_vxx": "blocks_multiply_const",
        "blocks_multiply_const_xx": "blocks_multiply_const",
        "filterbank_vcvcf": "filterbank",
        "pfb_channelizer_hier_ccf": "pfb_channelizer",
        "analog_agc2_xx": "analog_agc",
        "channels_channel_model2": "channels_channel_model",
        "blocks_nlog10_ff": "blocks_nlog10",          # two digits: not a version number
        "fec_encode_ccsds_27_bb": "fec_encode_ccsds_27",
        "low_pass_filter": "low_pass_filter",
        "blocks_complex_to_mag": "blocks_complex_to_mag",
    }
    assert {k: C.family_key(k) for k in cases} == cases


def test_assign_families_links_ids_and_yaml_stems():
    recs = [
        {"id": "fec_encode_ccsds_27_bb", "yaml_file": "fec_decode_ccsds_27_bb.block.yml"},
        {"id": "fec_decode_ccsds_27_fb", "yaml_file": "fec_decode_ccsds_27_fb.block.yml"},
        {"id": "blocks_multiply_const_xx", "yaml_file": "blocks_multiply_const_xx.block.yml"},
        {"id": "blocks_multiply_const_vxx", "yaml_file": "blocks_multiply_const_vxx.block.yml"},
        {"id": "blocks_add_xx", "yaml_file": "blocks_add_xx.block.yml"},
        {"id": "_dummy", "yaml_file": None},
    ]
    fam = C.assign_families(recs)
    assert fam["fec_encode_ccsds_27_bb"] == fam["fec_decode_ccsds_27_fb"] == "fec_decode_ccsds_27"
    assert fam["blocks_multiply_const_xx"] == fam["blocks_multiply_const_vxx"] == "blocks_multiply_const"
    assert fam["blocks_add_xx"] == "blocks_add" and fam["_dummy"] == "_dummy"
    assert C.assign_families(list(reversed(recs))) == fam


def test_allocate_largest_remainder_and_ties():
    assert C.allocate({"a": 5, "b": 3, "c": 2}, 3) == {"a": 1, "b": 1, "c": 1}
    assert C.allocate({"y": 1, "x": 1}, 1) == {"x": 1, "y": 0}          # tie -> stratum name
    alloc = C.allocate({"m1": 40, "m2": 17, "m3": 1}, 17)
    assert sum(alloc.values()) == 17 and alloc["m1"] == 12
    assert C.round_half_up(Fraction(5, 2)) == 3 and C.round_half_up(Fraction(3, 10) * 402) == 121


def test_exclusion_reasons_order_and_rules():
    base = {"flags": [], "n_inputs": 1, "n_outputs": 1, "has_value": False, "deprecated": False,
            "module": "blocks"}
    assert C.exclusion_reasons(dict(base, id="blocks_add_xx")) == []
    assert C.exclusion_reasons(dict(base, id="variable_qtgui_range")) == ["non-block"]
    assert C.exclusion_reasons(dict(base, id="pad_source")) == ["non-block"]
    assert C.exclusion_reasons(dict(base, id="json_config", has_value=True, n_inputs=0, n_outputs=0)) == ["non-block"]
    assert C.exclusion_reasons(dict(base, id="qtgui_dialgauge", has_value=True)) == []   # has ports
    assert C.exclusion_reasons(dict(base, id="blocks_throttle", deprecated=True)) == ["deprecated"]
    assert C.exclusion_reasons(dict(base, id="soapy_source", deprecated=True, module="soapy")) == [
        "deprecated", "hardware-or-network-module"]
    for module in C.EXCLUDED_MODULES:
        assert C.exclusion_reasons(dict(base, id=f"{module}_x", module=module)) == ["hardware-or-network-module"]


def _synthetic(n_modules=4, seed=7):
    rng = random.Random(seed)
    recs = []
    for m in range(n_modules):
        for f in range(rng.randint(3, 25)):
            for v in range(rng.choice([1, 1, 1, 2, 3])):
                recs.append({"id": f"mod{m}_fam{f}_{'x' * (v + 1)}", "module": f"mod{m}",
                             "family": f"mod{m}_fam{f}", "eligible": rng.random() > 0.1})
    return recs


def test_partition_is_order_independent_and_keeps_families_together():
    recs = _synthetic()
    first = C.partition(recs)
    shuffled = recs[:]
    random.Random(1).shuffle(shuffled)
    assert C.partition(shuffled) == first
    sides = defaultdict(set)
    for r in recs:
        side = first["split"][r["id"]]
        assert (side is None) == (not r["eligible"])
        if side:
            sides[r["family"]].add(side)
    assert all(len(s) == 1 for s in sides.values())
    n_fam = len(first["family_stratum"])
    assert first["n_test_families"] == C.round_half_up(Fraction(3, 10) * n_fam)
    per_module = defaultdict(int)
    for fam, side in {r["family"]: first["split"][r["id"]] for r in recs if r["eligible"]}.items():
        per_module[first["family_stratum"][fam]] += side == "test"
    assert dict(per_module) == {m: k for m, k in first["allocation"].items()}
    assert C.partition(recs, seed="another-seed")["split"] != first["split"]


# --------------------------------------------------------------------------- GNU Radio catalog


def _snapshot(root: Path) -> dict[str, tuple[int, int]] | None:
    if not root.exists():
        return None
    out = {}
    for path in sorted(root.rglob("*")):
        st = path.stat()
        out[str(path)] = (st.st_mtime_ns, st.st_size if path.is_file() else -1)
    st = root.stat()
    out[str(root)] = (st.st_mtime_ns, -1)
    return out


WATCHED = [Path.home() / ".cache" / "gnuradio", Path.home() / ".local" / "state" / "gnuradio"]


@pytest.fixture(scope="module")
def loaded():
    pytest.importorskip("gnuradio")
    before = [_snapshot(p) for p in WATCHED]
    plat = C.load_platform()
    doc = C.build(plat)
    after = [_snapshot(p) for p in WATCHED]
    return {"platform": plat, "doc": doc, "before": before, "after": after,
            "by_id": {r["id"]: r for r in doc["_catalog"]}}


@pytest.mark.gnuradio
def test_loading_never_touches_the_user_gnuradio_cache(loaded):
    assert loaded["before"] == loaded["after"]


@pytest.mark.gnuradio
def test_platform_loads_the_pinned_catalog(loaded):
    assert len(loaded["platform"].blocks) == 585
    assert loaded["doc"]["n_block_yml_files"] == 580
    assert loaded["doc"]["gnuradio_version"] == "3.10.12.0"
    assert len(loaded["doc"]["_catalog"]) == 585


@pytest.mark.gnuradio
def test_catalog_records_of_known_blocks(loaded):
    r = loaded["by_id"]
    add = r["blocks_add_xx"]
    assert add["module"] == "blocks" and add["has_type_param"] and add["type_params"] == ["type"]
    assert add["templated_ports"] == 2 and add["literally_typed_ports"] == 0 and add["templated_multiplicity"]
    assert add["n_asserts"] >= 1 and add["eligible"]
    lpf = r["low_pass_filter"]
    assert lpf["module"] == "filter" and lpf["yaml_file"] == "filter_low_pass_filter.block.yml"
    assert lpf["has_decim_or_interp"] and lpf["rate_params"] == ["decim", "interp"]
    assert r["analog_sig_source_x"]["n_message_ports"] == 1
    assert r["blocks_message_debug"]["n_message_ports"] == 4 and not r["blocks_message_debug"]["has_type_param"]
    assert r["blocks_throttle"]["deprecated_flag"] and r["blocks_throttle"]["reasons"] == ["deprecated"]
    assert r["digital_psk_mod"]["deprecated"] and not r["digital_psk_mod"]["deprecated_flag"]
    assert r["uhd_usrp_source"]["reasons"] == ["hardware-or-network-module"]
    assert r["audio_sink"]["module"] == "audio" and not r["audio_sink"]["eligible"]
    assert r["variable"]["reasons"] == ["non-block"] and r["options"]["builtin"]
    assert r["_dummy"]["yaml_file"] is None and r["_dummy"]["module"] == "grc"
    assert r["xmlrpc_server"]["module"] == "blocks"           # tree file wins over the file-name prefix
    for rec in (add, lpf):
        assert rec["yaml_sha256"] == hashlib.sha256((GNURADIO_BLOCKS / rec["yaml_file"]).read_bytes()).hexdigest()
    flagged = sorted(k for k, v in r.items() if v["deprecated_flag"])
    assert len(flagged) == 10


@pytest.mark.gnuradio
def test_catalog_hash_is_the_ladder_hash():
    h = hashlib.sha256()
    for path in sorted(GNURADIO_BLOCKS.glob("*.block.yml")):
        h.update(path.name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).hexdigest().encode() + b"\n")
    assert C.catalog_hash() == h.hexdigest()
    try:
        from grcbench.tier_f.grc import catalog_hash as ladder_hash
    except ImportError:
        pytest.skip("tier_f.grc not available")
    assert C.catalog_hash() == ladder_hash()


@pytest.mark.gnuradio
def test_partition_rules_hold_on_the_real_catalog(loaded):
    doc = loaded["doc"]
    sides = defaultdict(set)
    for b in doc["blocks"]:
        rec = loaded["by_id"][b["id"]]
        assert b["eligible"] == (b["reason"] is None) == (b["split"] is not None)
        assert b["reason"] == (", ".join(rec["reasons"]) or None)
        if b["split"]:
            assert rec["module"] not in C.EXCLUDED_MODULES and not rec["deprecated"]
            assert not b["id"].startswith("variable") and b["id"] not in C.NON_BLOCK_IDS
            sides[b["family"]].add(b["split"])
    assert all(len(s) == 1 for s in sides.values()), "a block family straddles test and train"
    c = doc["counts"]
    assert c["test"] + c["train"] == c["eligible"] and 0.25 <= c["test"] / c["eligible"] <= 0.35
    assert c["test_families"] == C.round_half_up(Fraction(3, 10) * c["eligible_families"])
    for module, m in doc["modules"].items():
        if m["test_families"] + m["train_families"] >= 4:
            assert m["test"] and m["train"], module
    assert doc["cross_module_families"] == []


@pytest.mark.gnuradio
def test_build_is_deterministic(loaded, tmp_path):
    again = C.build()                                       # fresh platform load
    a = C.write_outputs(loaded["doc"], tmp_path / "a")
    b = C.write_outputs(again, tmp_path / "b")
    assert [p.relative_to(tmp_path / "a") for p in a] == [p.relative_to(tmp_path / "b") for p in b]
    for pa, pb in zip(a, b):
        assert pa.read_bytes() == pb.read_bytes(), pa.name
    shuffled = loaded["doc"]["_catalog"][:]
    random.Random(3).shuffle(shuffled)
    assert C.partition(shuffled)["split"] == C.partition(loaded["doc"]["_catalog"])["split"]
    committed = REPO / C.PARTITION_JSON
    if committed.exists():                                  # the checked-in partition is current
        assert committed.read_bytes() == a[0].read_bytes()


@pytest.mark.gnuradio
def test_handoff_and_review_sheet(loaded):
    doc = loaded["doc"]
    text = C.handoff_md(doc)
    assert "Use train-side blocks only" in text and "Different wording" in text
    assert doc["catalog_hash"] in text and "BENCHMARK DATA" not in text
    rows = list(csv.DictReader(io.StringIO(C.review_sheet(doc))))
    assert len(rows) == 585 and {r["split"] for r in rows} == {"test", "train", "excluded"}
    assert sum(r["split"] == "test" for r in rows) == doc["counts"]["test"]
    train_ids = {b["id"] for b in doc["blocks"] if b["split"] == "train"}
    test_ids = {b["id"] for b in doc["blocks"] if b["split"] == "test"}
    listed = set(re.findall(r"\w+", text.split("## Train-side blocks you may use")[1]))
    assert train_ids <= listed and not (test_ids & listed)
