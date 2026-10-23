"""Tests for the GNU Radio block catalog (plan WS3a): modules, families, eligibility; no block split."""

from __future__ import annotations

import hashlib
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
def test_eligibility_rules_hold_on_the_real_catalog(loaded):
    doc = loaded["doc"]
    for b in doc["blocks"]:
        rec = loaded["by_id"][b["id"]]
        assert "split" not in b                              # there is no test/train block split
        assert b["eligible"] == (b["reason"] is None)
        assert b["reason"] == (", ".join(rec["reasons"]) or None)
        if b["eligible"]:
            assert rec["module"] not in C.EXCLUDED_MODULES and not rec["deprecated"]
            assert not b["id"].startswith("variable") and b["id"] not in C.NON_BLOCK_IDS
    c = doc["counts"]
    assert (c["blocks"], c["eligible"], c["excluded"]) == (585, 415, 170)
    assert c["eligible_families"] == len({b["family"] for b in doc["blocks"] if b["eligible"]})
    assert sum(m["eligible"] for m in doc["modules"].values()) == c["eligible"]


@pytest.mark.gnuradio
def test_build_is_deterministic_and_the_committed_catalog_is_current(loaded, tmp_path):
    again = C.build()                                       # fresh platform load
    a = C.write_outputs(loaded["doc"], tmp_path / "a")
    b = C.write_outputs(again, tmp_path / "b")
    assert [p.relative_to(tmp_path / "a") for p in a] == [p.relative_to(tmp_path / "b") for p in b]
    for pa, pb in zip(a, b):
        assert pa.read_bytes() == pb.read_bytes(), pa.name
    for p, rel in zip(a, (C.CATALOG_JSON, C.CATALOG_MD)):
        assert (REPO / rel).read_bytes() == p.read_bytes(), rel
    assert "no test/train block split" in (REPO / C.CATALOG_MD).read_text(encoding="utf-8")
    assert not (REPO / "splits" / "block_partition.json").exists()
