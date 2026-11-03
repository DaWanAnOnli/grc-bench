"""Tests for the Tier-F generator's model and plan (grcbench.tier_f.generate); no GNU Radio needed."""

from __future__ import annotations

import random
from collections import Counter

from grcbench.tier_f import generate as G


def _sig(kind="complex", tones=((1000, 0.5),), fs=32000.0, **kw):
    return G.Sig(kind, fs, tuple(G.Tone(f, a) for f, a in tones), **kw)


def test_the_plan_reaches_the_preregistered_counts():
    """Pre-registration §2: 150 test tasks (60 L1, 60 L2, 30 L3) and 45 dev tasks, with the 30 built ones."""
    built = {"test": Counter(L1=10, L2=9, L3=1), "dev": Counter(L1=5, L2=5)}
    for split, total in (("test", 150), ("dev", 45)):
        plan = G.plan(split)
        levels = Counter(level for _, level in plan.values())
        assert sum(levels.values()) + sum(built[split].values()) == total
        assert len(set(plan)) == len(plan)
    assert Counter(level for _, level in G.plan("test").values()) + built["test"] == Counter(L1=60, L2=60, L3=30)
    assert min(int(t.split("-")[-1]) for t in G.plan("test")) == 21
    assert min(int(t.split("-")[-1]) for t in G.plan("dev")) == 11


def test_a_filter_keeps_the_tones_of_its_passband():
    p = {"ftype": "low_pass_filter", "cut": 3000, "width": 300, "gap": (1000, 5000), "rate": "samp_rate"}
    out = G.Filter().apply(_sig(tones=((1000, 0.5), (5000, 0.8))), p)
    assert [t.f for t in out.tones] == [1000]


def test_a_resampler_changes_the_rate_seen_by_every_later_stage():
    sig = _sig(fs=48000.0)
    out = G.Resample().apply(sig, {"interp": 2, "decim": 1})
    assert out.fs == 96000 and out.rate == "samp_rate*2/1"
    lo = G.Shift().design(random.Random(1), out)
    assert lo is None or lo["rate"] == "samp_rate*2/1"
    assert G.per("samp_rate") == "/samp_rate" and G.per("samp_rate*2/1") == "/(samp_rate*2/1)"


def test_only_the_quantizer_leaves_an_integer_grid():
    q = G.run_stage(G.Quantize(), _sig("float", ((1000, 0.4),)), {"scale": 10, "back": 10})
    assert q.int_scale == 10.0
    after = G.run_stage(G.Resample(), q, {"interp": 3, "decim": 2})
    assert after is not None and after.int_scale is None


def test_the_rectifier_and_the_magnitude_give_levels():
    rect = G.Rectify().apply(_sig("float", ((1000, 2.0),)), {})
    assert abs(rect.dc - 4 / 3.141592653589793) < 1e-9 and rect.strongest().f == 2000
    mag = G.Magnitude().apply(_sig("complex", ((1500, 0.6),)), {})
    assert mag.level_only and mag.dc == 0.6


def test_a_mutant_counts_as_rejected_only_when_it_clearly_misses():
    sig = _sig(tones=((2000, 1.0),))
    checks, _ = G.oracle_checks(sig, "out", "main")
    assert not G.rejected(sig, checks)                                          # the reference itself
    assert not G.rejected(_sig(tones=((2000, 1.02),)), checks)                  # inside the tolerance
    assert G.rejected(_sig(tones=((2500, 1.0),)), checks)                       # another frequency
    assert G.rejected(_sig(tones=((2000, 1.5),)), checks)                       # 50 % louder
    assert not G.rejected(_sig(tones=((2500, 1.0),)), checks, point="second")   # another output's checks


def test_a_repair_symptom_states_what_differs():
    want = _sig("float", ((2800, 0.8), (4500, 0.6)))
    moved = _sig("float", ((3600, 0.8), (4500, 0.6)))
    assert G._summary(moved, want) == "its output holds components at 3.6 kHz and 4.5 kHz"
    louder = _sig("float", ((2800, 1.6), (4500, 0.6)))
    assert G._summary(louder, want).startswith("its RMS level is")


def test_tasks_of_a_cell_stay_within_the_level_cap():
    """L2 keeps one task per cell; L1 and L3 tasks of a split share a cell up to CELL_CAP[level], with one
    cluster_id per cell."""
    import json
    cells: dict[tuple[str, str, str], int] = Counter()
    for p in sorted(G.TF.glob("*/*/task.json")):
        t = json.loads(p.read_text(encoding="utf-8"))
        cells[(t["split"], t["level"], t["cluster_id"])] += 1
    assert cells
    for (split, level, _), n in cells.items():
        assert n <= G.CELL_CAP.get(level, 1), (split, level, n)
    assert "L2" not in G.CELL_CAP


def test_the_disjointness_text_holds_for_tasks_that_share_a_cell():
    """Tasks of a shared cell must not claim that no other item uses their signature."""
    import json
    for p in sorted(G.TF.glob("*/*/task.json")):
        t = json.loads(p.read_text(encoding="utf-8"))
        if t["source"]["kind"] == "generated_chain":
            assert "used by no other" not in t["disjointness"], t["id"]
            assert "share its cluster_id" in t["disjointness"], t["id"]


def test_the_same_id_and_attempt_give_the_same_random_draws():
    a, b = G.rng_for("gbf-test-21", 3), G.rng_for("gbf-test-21", 3)
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]
    assert G.rng_for("gbf-test-21", 4).random() != G.rng_for("gbf-test-21", 3).random()
