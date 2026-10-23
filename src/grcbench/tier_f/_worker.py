"""Sandboxed executor for ladder rungs V2-V5 (run as `python -m grcbench.tier_f._worker spec.json`).

Runs inside bubblewrap (see common/sandbox.py) because GNU Radio evaluates parameters as Python,
Embedded-Python blocks run on load, and V5 runs the generated program. One OS process per check.

  V2 load      GRC loads the graph; fails on unknown block ids (GRC silently makes them "dummy"
               blocks), dangling connections, enum values not in the block's options (GRC silently
               coerces them), or a block catalog that differs from the pinned one.
  V3 validate  GRC validation: ports connected, types and vector lengths match, parameters evaluate.
  V4 codegen   GRC generates the Python top block.
  V5 run       Headless bounded run. GUI and audio sinks (and file sinks, to avoid disk I/O) become
               null sinks of the same type; Qt widget variables become plain variables. The harness
               taps each observation point with a capture sink (_tap_block), starts the flowgraph,
               waits until every tap holds n_items, the timeout expires or the flowgraph finishes on
               its own, then stops it. Liveness: every tap must hold at least min_items by then.
               Fails on any exception, on missing liveness, or if the flowgraph does not stop
               cleanly in time.
               Taps never back-pressure the graph: a tap keeps the first n_items items and then keeps
               consuming (and dropping) everything it is offered, so a full tap on one observation
               point cannot stall an upstream block it shares with a slower point (the former Head +
               Vector Sink tap stopped reading once full, which starved the other points).
Writes result.json and obs/<point>.npy into the run directory.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
import threading
import time
import traceback
from pathlib import Path

GUI_SINK_PREFIXES = ("qtgui_",)
SINKS_TO_NULL = {"audio_sink", "blocks_file_sink"}
REMOVE_ALWAYS = {"qtgui_tab_widget", "qtgui_grbackground", "qtgui_graphicitem", "qtgui_graphicoverlay"}


def _normalise(data: dict) -> dict:
    """Fill fields GRC's loader requires but a compact answer may omit (documented leniency)."""
    data = dict(data)
    opts = dict(data.get("options") or {})
    opts["parameters"] = dict(opts.get("parameters") or {})
    # GRC's own default. GRC looks up an unknown block id as a hier block on this path while it loads;
    # in the isolated episode process the parameter otherwise reads None and import_data raises
    # TypeError, so a graph that holds an unknown block id never loads.
    opts["parameters"].setdefault("hier_block_src_path", ".:")
    opts.setdefault("states", {})
    data["options"] = opts
    blocks = []
    for blk in data.get("blocks") or []:
        blk = dict(blk)
        blk.setdefault("parameters", {})
        blk.setdefault("states", {})
        blk["parameters"] = {k: ("" if v is None else v) for k, v in (blk["parameters"] or {}).items()}
        blocks.append(blk)
    data["blocks"] = blocks
    data["connections"] = [[str(x) for x in c] for c in (data.get("connections") or [])]
    data.setdefault("metadata", {"file_format": 1})
    return data


def _v2(platform, data, catalog_ok):
    from grcbench.tier_f.grc import enum_options
    errors, hallucinated = [], []
    if not catalog_ok:
        errors.append("block catalog differs from the pinned GNU Radio 3.10.12 catalog")
    fg = platform.make_flow_graph()
    conn_failed = fg.import_data(data)
    fg.rewrite()
    for blk in fg.blocks:
        if blk.is_dummy_block:
            hallucinated.append(blk.key)
            errors.append(f"unknown block id '{blk.key}' (block '{blk.name}')")
    if conn_failed:
        errors.append("a connection refers to a block or port that does not exist")
    # Enum lint: GRC silently replaces an enum value it does not recognise (e.g. type 'cmplx' becomes
    # 'complex'). Flag exactly those replacements: the value GRC ended up with differs from the file's.
    # Qt GUI blocks are skipped: their enums are cosmetic (colours, orientation) and the harness
    # replaces those blocks before running anyway. Surrounding quotes are not a difference.
    loaded = {b.name: b for b in fg.blocks}
    for blk in data.get("blocks", []):
        gblk = loaded.get(blk.get("name"))
        if gblk is None or gblk.is_dummy_block or str(blk.get("id", "")).startswith(GUI_SINK_PREFIXES + ("variable_qtgui_",)):
            continue
        for pid, allowed in enum_options(blk.get("id", "")).items():
            if pid not in (blk.get("parameters") or {}) or pid not in gblk.params:
                continue
            given, used = str(blk["parameters"][pid]), str(gblk.params[pid].get_value())
            if given.strip("'\"") != used.strip("'\""):
                errors.append(f"block '{blk.get('name')}': parameter '{pid}' = '{given}' is not one of "
                              f"{allowed} (GRC would silently use '{used}')")
    return fg, errors, sorted(set(hallucinated))


def _codegen(platform, fg, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = platform.Generator(fg, str(out_dir))
    gen.write()
    return Path(gen.file_path)


def _null_type(dtype: str) -> str:
    from grcbench.tier_f.grc import STREAM_TYPES
    return STREAM_TYPES.get(dtype, ("byte", "int8", 1))[0]


def _convert_headless(platform, fg, sink_mode: str = "null"):
    """In-place: GUI/audio/file sinks -> null sinks; Qt widget variables -> plain variables.

    sink_mode="file" (used to export task seeds, never for checking) instead turns every input of a
    GUI/audio sink into its own File Sink "<sink>_<i>.dat", so tasks can name a concrete observation point.

    Returns {sink block name: [(src block name, src port key, dtype, vlen), ...]} for every replaced
    sink, so that observation points defined as "whatever feeds sink X" can still be tapped."""
    fed_by: dict[str, list] = {}
    for blk in list(fg.blocks):
        if not blk.enabled or blk.key == "options":
            continue
        key, name = blk.key, blk.name
        if key.startswith("variable_qtgui_"):
            value = None
            for pid in ("value", "default", "initial"):
                if pid in blk.params:
                    value = blk.params[pid].get_value()
                    break
            in_msgs = [c for c in fg.connections if c.source_block is blk or c.sink_block is blk]
            for c in in_msgs:
                fg.remove_element(c)
            fg.remove_element(blk)
            var = fg.new_block("variable")
            var.params["id"].set_value(name)
            var.params["value"].set_value(value if value not in (None, "") else "0")
            continue
        is_gui_sink = key.startswith(GUI_SINK_PREFIXES) and not blk.sources
        if key in REMOVE_ALWAYS or (key.startswith(GUI_SINK_PREFIXES) and not any(
                p.domain == "stream" for p in blk.sinks)):
            for c in [c for c in fg.connections if c.source_block is blk or c.sink_block is blk]:
                fg.remove_element(c)
            fg.remove_element(blk)
            continue
        if sink_mode == "file" and key == "blocks_file_sink":
            continue
        if key in SINKS_TO_NULL or is_gui_sink or (key.startswith(GUI_SINK_PREFIXES) and any(
                p.domain == "stream" for p in blk.sinks)):
            stream_in = [c for c in fg.connections if c.sink_block is blk and c.sink_port.domain == "stream"]
            stream_in.sort(key=lambda c: int(c.sink_port.key) if str(c.sink_port.key).isdigit() else 0)
            feeds = [(c.source_block.name, c.source_port.key, c.source_port.dtype, int(c.source_port.vlen or 1))
                     for c in stream_in]
            fed_by[name] = feeds
            for c in [c for c in fg.connections if c.source_block is blk or c.sink_block is blk]:
                fg.remove_element(c)
            fg.remove_element(blk)
            if not feeds:
                continue
            if sink_mode == "file":
                fg.rewrite()
                for i, (src_name, src_port, dtype_i, vlen_i) in enumerate(feeds):
                    fsink = fg.new_block("blocks_file_sink")
                    fsink.params["id"].set_value(f"{name}_{i}")
                    fsink.params["type"].set_value(_null_type(dtype_i))
                    fsink.params["vlen"].set_value(str(vlen_i))
                    fsink.params["file"].set_value(f"{name}_{i}.dat")
                    fg.rewrite()
                    src_blk = next(b for b in fg.blocks if b.name == src_name)
                    src_p = next(p for p in src_blk.sources if str(p.key) == str(src_port))
                    fg.connect(src_p, next(p for p in fsink.sinks if p.domain == "stream"))
                continue
            dtype, vlen = feeds[0][2], feeds[0][3]
            null = fg.new_block("blocks_null_sink")
            null.params["id"].set_value(name)   # keep the name: snippets/Python blocks may refer to it
            null.params["type"].set_value(_null_type(dtype))
            null.params["vlen"].set_value(str(vlen))
            null.params["num_inputs"].set_value(str(len(feeds)))
            fg.rewrite()
            for i, (src_name, src_port, _, _) in enumerate(feeds):
                src_blk = next(b for b in fg.blocks if b.name == src_name)
                src_p = next(p for p in src_blk.sources if str(p.key) == str(src_port))
                dst_p = next(p for p in null.sinks if str(p.key) == str(i))
                fg.connect(src_p, dst_p)
    opts = fg.options_block
    opts.params["generate_options"].set_value("no_gui")
    if "run_options" in opts.params:
        opts.params["run_options"].set_value("run")
    fg.rewrite()
    fg.validate()
    return fed_by


def _resolve_points(fg, points: list[dict], fed_by: dict) -> list[dict]:
    """Observation points -> concrete (src block name, port, dtype, vlen)."""
    resolved = []
    for pt in points:
        name = pt["name"]
        if "edge" in pt:
            blk = next((b for b in fg.blocks if b.name == pt["edge"]["block"]), None)
            if blk is None:
                resolved.append({"name": name, "error": f"block '{pt['edge']['block']}' not found"})
                continue
            port = next((p for p in blk.sources if str(p.key) == str(pt["edge"].get("port", "0"))), None)
            if port is None:
                resolved.append({"name": name, "error": "port not found"})
                continue
            resolved.append({"name": name, "src": blk.name, "port": str(port.key), "dtype": port.dtype,
                             "vlen": int(port.vlen or 1)})
        elif "sink" in pt:
            want = pt["sink"]
            match = None
            for sink_name, feeds in fed_by.items():
                if sink_name == "__meta__":
                    continue
                if want.get("block_name") and want["block_name"] != sink_name:
                    continue
                meta = fed_by.get("__meta__", {}).get(sink_name, {})
                if want.get("block_id") and meta.get("id") != want["block_id"]:
                    continue
                if any(str(meta.get("params", {}).get(k)) != str(v) for k, v in (want.get("params") or {}).items()):
                    continue
                match = feeds
                break
            if not match:
                resolved.append({"name": name, "error": f"no sink matching {want}"})
                continue
            idx = int(pt.get("input", 0))
            if idx >= len(match):
                resolved.append({"name": name, "error": f"sink has no input {idx}"})
                continue
            src, port, dtype, vlen = match[idx]
            resolved.append({"name": name, "src": src, "port": str(port), "dtype": dtype, "vlen": vlen})
    return resolved


def _sink_meta(fg) -> dict:
    """Block id and evaluated parameters of every sink before conversion (for sink matching)."""
    meta = {}
    for blk in fg.blocks:
        if blk.key in SINKS_TO_NULL or blk.key.startswith(GUI_SINK_PREFIXES):
            params = {}
            for pid, prm in blk.params.items():
                try:
                    val = prm.get_evaluated()
                except Exception:
                    val = prm.get_value()
                params[pid] = Path(val).name if pid == "file" and isinstance(val, str) else val
            meta[blk.name] = {"id": blk.key, "params": params}
    return meta


def _tap_block(np_dtype: str, vlen: int, n_items: int):
    """A sink that stores the first n_items items it receives and consumes (drops) the rest, so it
    never blocks its upstream block. `count` (items stored, at most n_items) is safe to poll from
    another thread; `data()` returns the stored items, shape (count,) or (count, vlen)."""
    import numpy as np
    from gnuradio import gr

    dtype = np.dtype(np_dtype)

    class _Tap(gr.sync_block):
        def __init__(self):
            gr.sync_block.__init__(self, name="grcbench_tap",
                                   in_sig=[(dtype, vlen) if vlen > 1 else dtype], out_sig=None)
            self.n_items = n_items
            self.buf = np.zeros((n_items, vlen) if vlen > 1 else n_items, dtype=dtype)
            self.count = 0

        def work(self, input_items, output_items):
            x = input_items[0]
            k = min(len(x), self.n_items - self.count)
            if k > 0:
                self.buf[self.count:self.count + k] = x[:k]
                self.count += k
            return len(x)

        def data(self):
            return self.buf[:self.count].copy()

    return _Tap()


def _run(module_path: Path, cls_name: str, taps: list[dict], n_items: int, min_items: int,
         timeout_s: float, grace_s: float, obs_dir: Path) -> dict:
    import numpy as np
    from grcbench.tier_f.grc import STREAM_TYPES
    np.random.seed(0)
    random.seed(0)
    spec = importlib.util.spec_from_file_location("grcbench_generated", module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(module_path.parent))
    spec.loader.exec_module(mod)
    tb = getattr(mod, cls_name)()
    sinks = {}
    for tap in taps:
        if "error" in tap:
            continue
        _, np_dtype, _ = STREAM_TYPES.get(tap["dtype"], ("byte", "int8", 1))
        vlen = int(tap["vlen"])
        src_obj = getattr(tb, tap["src"])
        sink = _tap_block(np_dtype, vlen, n_items)
        tb.connect((src_obj, int(tap["port"])), sink)
        sinks[tap["name"]] = sink
    tb.start()
    waiter = threading.Thread(target=tb.wait, daemon=True)
    waiter.start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if sinks and all(s.count >= n_items for s in sinks.values()):
            break
        if not waiter.is_alive():
            break
        time.sleep(0.02)
    elapsed = time.monotonic() - t0
    finished_on_its_own = not waiter.is_alive()
    tb.stop()
    waiter.join(grace_s)
    stopped = not waiter.is_alive()
    counts = {}
    obs_dir.mkdir(parents=True, exist_ok=True)
    for name, sink in sinks.items():
        arr = sink.data()
        np.save(obs_dir / f"{name}.npy", arr)
        counts[name] = int(arr.shape[0])
    live = all(c >= min_items for c in counts.values())
    return {"elapsed_s": round(elapsed, 3), "stopped_cleanly": stopped, "items": counts,
            "finished_on_its_own": finished_on_its_own, "liveness": live}


def main(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    run_dir = Path(spec["run_dir"])
    res = {"rungs": {}, "hallucinated_blocks": []}
    out = run_dir / "result.json"

    def done(code=0):
        out.write_text(json.dumps(res, indent=1, default=str))
        sys.stdout.flush()
        import os
        os._exit(code)

    try:
        import yaml
        from grcbench.tier_f.grc import platform
        data = _normalise(yaml.safe_load(Path(spec["grc_path"]).read_text()))
        plat = platform()
        fg, errors, halluc = _v2(plat, data, spec.get("catalog_ok", True))
        res["hallucinated_blocks"] = halluc
        res["rungs"]["V2"] = {"passed": not errors, "errors": errors}
        if errors or spec.get("max_rung", 6) < 3:
            done()
        fg.validate()
        v3_err = list(dict.fromkeys(fg.get_error_messages()))
        res["rungs"]["V3"] = {"passed": fg.is_valid() and not v3_err, "errors": v3_err[:50]}
        if not res["rungs"]["V3"]["passed"] or spec.get("max_rung", 6) < 4:
            done()
        try:
            gen_path = _codegen(plat, fg, run_dir / "gen")
            res["rungs"]["V4"] = {"passed": gen_path.exists(), "errors": [], "file": str(gen_path)}
        except Exception as exc:
            res["rungs"]["V4"] = {"passed": False, "errors": [f"{type(exc).__name__}: {exc}"]}
            done()
        if spec.get("max_rung", 6) < 5:
            done()
        if fg.get_option("generate_options").startswith("hb"):
            res["rungs"]["V5"] = {"passed": False, "errors": ["hierarchical block: no top block to run"]}
            done()
        meta = _sink_meta(fg)
        fed_by = _convert_headless(plat, fg)
        fed_by["__meta__"] = meta
        conv_err = list(dict.fromkeys(fg.get_error_messages()))
        if not fg.is_valid():
            res["rungs"]["V5"] = {"passed": False, "errors": ["headless conversion produced an invalid graph"] + conv_err[:20]}
            done()
        try:  # a debugging aid only; never affects the verdict
            plain = json.loads(json.dumps(fg.export_data(), default=str))
            (run_dir / "converted.grc").write_text(yaml.safe_dump(plain, sort_keys=False))
        except Exception:
            pass
        conv_path = _codegen(plat, fg, run_dir / "gen_headless")
        points = spec.get("points", [])
        if points == "auto":   # seeds: observe whatever fed every replaced sink
            points = [{"name": f"{n}__in{i}", "sink": {"block_name": n}, "input": i}
                      for n, feeds in fed_by.items() if n != "__meta__" for i in range(len(feeds))]
        res["observation_points"] = points
        taps = _resolve_points(fg, points, fed_by)
        run_cfg = spec.get("run", {})
        info = _run(conv_path, fg.get_option("id"), taps, int(run_cfg.get("n_items", 8192)),
                    int(run_cfg.get("min_items", run_cfg.get("n_items", 8192))), float(run_cfg.get("timeout_s", 10)),
                    float(run_cfg.get("grace_s", 5)), run_dir / "obs")
        tap_errors = [f"observation '{t['name']}': {t['error']}" for t in taps if "error" in t]
        errs = []
        if not info["stopped_cleanly"]:
            errs.append("flowgraph did not stop within the grace period")
        if taps and not info["liveness"]:
            errs.append(f"liveness: an observation point received too few items {info['items']}")
        res["rungs"]["V5"] = {"passed": not errs, "errors": errs, "observation_errors": tap_errors, **info}
        done()
    except SystemExit:
        raise
    except BaseException:
        tb_text = traceback.format_exc()
        last = max((k for k in ("V2", "V3", "V4", "V5") if k in res["rungs"]), default=None)
        nxt = {"None": "V2", "V2": "V3", "V3": "V4", "V4": "V5", "V5": "V5"}[str(last)]
        res["rungs"][nxt] = {"passed": False, "errors": [tb_text.strip().splitlines()[-1]], "traceback": tb_text[-4000:]}
        done(0)


if __name__ == "__main__":
    main(sys.argv[1])
