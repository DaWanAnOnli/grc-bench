"""`./gb seeds` — inventory the GNU Radio example flowgraphs and split them into test/dev/train seeds.

Proposal §4.4: flowgraph-task seeds are "the ~80+ GNU Radio examples ... grouped by source
repository for splitting", and benchmark and training tasks come "from disjoint seed files".
Eval plan §4.3: "All flowgraphs from one source file stay in one split."

For each of the 230 example .grc files shipped with GNU Radio 3.10.12 this:
  1. classifies it: headless, GUI-only (convertible), hardware/network (excluded), hierarchical;
  2. runs the ladder V1-V4 on it, with GUI/audio/file sinks converted and every former sink's input
     observed ("auto" observation points), to see whether it can serve as a headless seed;
  3. records size (blocks), rate changes and an L1/L2/L3 estimate (proposal §5.2 levels).
Usable seeds (V1-V4 pass) are split by source file into test / dev / train, stratified by example
directory, ranked by SHA-256 with a fixed seed. Seeds that GRC-Agent or Qoherent's session
collection already used are forced into train. Unusable examples are listed but get no split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from ..common.io import read_json, sha256_file, write_json, write_text
from ..common.paths import GNURADIO_EXAMPLES, HANDOFF, REPORTS, SPLITS
from .ladder import check

SEED = "grc-bench-seeds-v1-2026-10-14"
SHARES = {"test": 0.40, "dev": 0.15, "train": 0.45}
HARDWARE_PREFIXES = ("uhd_", "soapy_", "zeromq_", "network_", "xmlrpc_", "iio_", "video_", "ctrlport_",
                     "osmosdr_", "audio_source", "rfnoc_")
GUI_PREFIXES = ("qtgui_", "variable_qtgui_")
# Seeds used by GRC-Agent's tests or by Qoherent's session collection (reports/reference/training_task_specs.json)
FORCED_TRAIN_STEMS = {"dial_tone", "fm_rx", "broken_unconnected_sink", "resampler_demo", "run_test_null_sink",
                      "demo_qam", "qtgui_vector_sink_example", "5gnalizer", "5G_nebula", "new_dmeo", "empty"}
RATE_KEYS = ("decim", "interp", "decimation", "interpolation", "resamp_ratio", "rate")
# Substrings of block ids that mark an L3 trait (proposal §5.2: tags/PDUs/message ports, OFDM, Embedded
# Python). "message" and "strobe" catch the message-port blocks (blocks_message_strobe[_random],
# blocks_message_debug, blocks_tags_strobe), whose ids do not contain "msg".
L3_HINTS = ("ofdm", "pdu", "tag", "epy_", "msg", "message", "strobe", "packet", "header", "crc", "fec_",
            "dtv_", "trellis")
LICENSE = "GPL-3.0-or-later (example flowgraph shipped with GNU Radio 3.10.12)"


def classify(data: dict) -> tuple[str, list[str]]:
    blocks = [b for b in data.get("blocks", []) or [] if (b.get("states") or {}).get("state", "enabled") != "disabled"]
    ids = [b.get("id", "") for b in blocks]
    hw = sorted({i for i in ids if i.startswith(HARDWARE_PREFIXES)})
    opts = (data.get("options") or {}).get("parameters", {})
    if str(opts.get("generate_options", "")).startswith("hb"):
        return "hierarchical", hw
    if hw:
        return "hardware_or_network", hw
    if any(i.startswith(GUI_PREFIXES) for i in ids):
        return "gui_only", []
    return "headless", []


def features(data: dict) -> dict:
    blocks = [b for b in data.get("blocks", []) or [] if (b.get("states") or {}).get("state", "enabled") != "disabled"]
    functional = [b for b in blocks if not b.get("id", "").startswith(("variable", "qtgui_tab", "import", "parameter"))
                  and b.get("id") not in ("options", "note")]
    ids = [b.get("id", "") for b in functional]
    rate_change = any(any(k in (b.get("parameters") or {}) for k in RATE_KEYS) and str(
        (b.get("parameters") or {}).get(next((k for k in RATE_KEYS if k in (b.get("parameters") or {})), ""), "1")) not in ("1", "1.0")
        for b in functional) or any("resampler" in i or "decim" in i or "interp" in i for i in ids)
    hints = sorted({h for h in L3_HINTS for i in ids if h in i})
    n = len(functional)
    if n > 15 or hints:
        level = "L3"
    elif n <= 5 and not rate_change:
        level = "L1"
    else:
        level = "L2"
    return {"n_blocks": n, "rate_change": bool(rate_change), "l3_hints": hints, "level_estimate": level,
            "block_ids": sorted(set(ids))}


def inspect_one(path: Path) -> dict:
    rel = path.relative_to(GNURADIO_EXAMPLES).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    rec = {"file": rel, "directory": str(Path(rel).parent), "stem": path.stem, "sha256": sha256_file(path),
           "license": LICENSE}
    try:
        data = yaml.safe_load(text)
    except Exception as exc:
        rec.update({"class": "unparseable", "error": str(exc).splitlines()[0]})
        return rec
    cls, hw = classify(data)
    rec.update({"class": cls, "hardware_blocks": hw, **features(data)})
    if cls in ("headless", "gui_only"):
        # seeds only need data to flow at every former sink; low-rate GUI displays get few items
        task = {"observation": {"points": "auto", "run": {"n_items": 4096, "min_items": 64, "timeout_s": 15}}}
        res = check(text, task, max_rung=4, timeout=90)
        v4 = res["rungs"].get("V4", {})
        rec["ladder"] = {"highest_passed": res["highest_passed"], "first_failed": res["first_failed"],
                         "errors": next((res["rungs"][r].get("errors", [])[:3] for r in ("V1", "V2", "V3", "V4")
                                         if res["rungs"].get(r, {}).get("passed") is False), []),
                         "items": v4.get("items"), "elapsed_s": v4.get("elapsed_s"),
                         "finished_on_its_own": v4.get("finished_on_its_own")}
        rec["usable_seed"] = res["highest_passed"] == "V4" and bool(v4.get("items"))
    else:
        rec["usable_seed"] = False
    return rec


def _rank(key: str) -> str:
    return hashlib.sha256(f"{SEED}\0{key}".encode()).hexdigest()


def split(records: list[dict]) -> dict[str, str]:
    """Largest-remainder allocation of usable seeds per directory, SHA-256 ranked within it."""
    assign: dict[str, str] = {}
    usable = [r for r in records if r.get("usable_seed")]
    forced = [r for r in usable if r["stem"] in FORCED_TRAIN_STEMS]
    for r in forced:
        assign[r["file"]] = "train"
    pool = [r for r in usable if r["file"] not in assign]
    by_dir: dict[str, list[dict]] = {}
    for r in pool:
        by_dir.setdefault(r["directory"], []).append(r)
    for d, members in sorted(by_dir.items()):
        members.sort(key=lambda r: _rank(r["file"]))
        n = len(members)
        quotas = {k: v * n for k, v in SHARES.items()}
        counts = {k: int(q) for k, q in quotas.items()}
        left = n - sum(counts.values())
        for k in sorted(SHARES, key=lambda k: (-(quotas[k] - counts[k]), list(SHARES).index(k)))[:left]:
            counts[k] += 1
        i = 0
        for k in ("test", "dev", "train"):
            for r in members[i:i + counts[k]]:
                assign[r["file"]] = k
            i += counts[k]
    return assign


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb seeds")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reuse", action="store_true", help="reuse reports/seed_inventory.json ladder results")
    args = ap.parse_args(argv)
    paths = sorted(GNURADIO_EXAMPLES.rglob("*.grc"))
    inv_path = REPORTS / "seed_inventory.json"
    if args.reuse and inv_path.exists():
        records = read_json(inv_path)["examples"]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            records = list(ex.map(inspect_one, paths))
    assign = split(records)
    for r in records:
        r["split"] = assign.get(r["file"])
        r["forced_train"] = r["stem"] in FORCED_TRAIN_STEMS and r.get("usable_seed", False)
    write_json(inv_path, {"gnuradio_examples": str(GNURADIO_EXAMPLES), "n_examples": len(records),
                          "examples": records})
    seeds = {r["file"]: {"split": r["split"], "level_estimate": r.get("level_estimate"),
                         "n_blocks": r.get("n_blocks"), "sha256": r["sha256"], "license": r["license"]}
             for r in records if r.get("split")}
    write_json(SPLITS / "flowgraph_seed_split.json", {
        "seed": SEED, "shares": SHARES, "unit": "source file", "stratified_by": "example directory",
        "forced_train_stems": sorted(FORCED_TRAIN_STEMS),
        "rule": "usable seeds (ladder V1-V4 pass after headless conversion) split by largest remainder per "
                "directory, ranked by sha256(seed + NUL + file); forced-train stems always train; "
                "unusable examples get no split",
        "counts": {k: sum(1 for v in seeds.values() if v["split"] == k) for k in SHARES},
        "seeds": seeds})
    _report(records)
    _handoff(seeds)
    print(json.dumps({"examples": len(records), "usable": sum(1 for r in records if r.get("usable_seed")),
                      **{k: sum(1 for v in seeds.values() if v["split"] == k) for k in SHARES}}))
    return 0


def _report(records: list[dict]) -> None:
    from collections import Counter
    cls = Counter(r["class"] for r in records)
    lines = ["# Seed inventory: GNU Radio 3.10.12 example flowgraphs", "",
             "Generated by `./gb seeds` (src/grcbench/tier_f/seeds.py). Each headless or GUI-only example was",
             "run through ladder rungs V1-V4 after headless conversion (Qt/audio/file sinks -> null sinks, Qt",
             "widget variables -> plain variables), observing the input of every former sink.", "",
             "## Classes", "", "| Class | Examples |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(cls.items())]
    usable = [r for r in records if r.get("usable_seed")]
    lines += ["", f"Usable seeds (pass V1-V4): **{len(usable)}** of "
              f"{cls.get('headless', 0) + cls.get('gui_only', 0)} headless or GUI-only examples.", "",
              "## Failures among headless / GUI-only examples", "", "| First failed rung | Examples |", "|---|---|"]
    fails = Counter((r.get("ladder") or {}).get("first_failed") for r in records
                    if r["class"] in ("headless", "gui_only") and not r.get("usable_seed"))
    lines += [f"| {k} | {v} |" for k, v in sorted(fails.items(), key=lambda kv: str(kv[0]))]
    lines += ["", "## Usable seeds", "", "| File | Level (est.) | Blocks | Rate change | Split |", "|---|---|---|---|---|"]
    for r in sorted(usable, key=lambda r: r["file"]):
        lines.append(f"| {r['file']} | {r['level_estimate']} | {r['n_blocks']} | {'yes' if r['rate_change'] else 'no'} "
                     f"| {r.get('split')}{' (forced)' if r.get('forced_train') else ''} |")
    lines += ["", "## Not usable (headless or GUI-only examples that failed a rung)", "",
              "| File | First failed | Reason |", "|---|---|---|"]
    for r in sorted((r for r in records if r["class"] in ("headless", "gui_only") and not r.get("usable_seed")),
                    key=lambda r: r["file"]):
        lad = r.get("ladder") or {}
        reason = "; ".join(lad.get("errors", []))[:160].replace("|", "/")
        lines.append(f"| {r['file']} | {lad.get('first_failed')} | {reason} |")
    write_text(REPORTS / "seed_inventory.md", "\n".join(lines))


def _handoff(seeds: dict) -> None:
    train = sorted(f for f, v in seeds.items() if v["split"] == "train")
    held = sorted(f for f, v in seeds.items() if v["split"] in ("test", "dev"))
    text = ["# Flowgraph seed split (for the training team)", "",
            "GRC-Bench Tier-F and Tier-A tasks are built from the GNU Radio example flowgraphs listed as",
            "test or dev below. Training flowgraph tasks and programmatic agent sessions (proposal §4.4) must",
            "come from **other** seed files: the train seeds below, permissively licensed community flowgraphs,",
            "or hand-written chains. Never derive training items from a test or dev seed file, including by",
            "editing or fault-injecting it. Full detail: splits/flowgraph_seed_split.json, reports/seed_inventory.md.",
            "", f"## Held out for GRC-Bench ({len(held)} files): do not use", ""]
    text += [f"- {f}" for f in held]
    text += ["", f"## Train seeds ({len(train)} files)", ""] + [f"- {f}" for f in train]
    write_text(HANDOFF / "training_team" / "flowgraph_seed_split.md", "\n".join(text))
