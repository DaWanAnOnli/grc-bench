"""The Tier-A grader: a task is judged by its end state (eval plan §4.4 "Grading", §5.4).

Eval plan §4.4: "A task is judged by its end state: the final saved flowgraph passes the Tier-F
checks and the task's own assertions ... an empty or missing answer never counts as a success, and
every task's grader is tested against a deliberately wrong end state." `grade(task, episode)` does
exactly that, mechanically, with no prose rubric:

  * the harness itself re-runs the FINAL SAVED graph (the file on disk, never the agent's own run
    or its reading of the graph) through grcbench.tier_f.ladder.check up to the task's required
    rung;
  * then the task's own assertions (acceptance, below), each a rule checked against the end state:
    GNU Radio's own view of the final graph (_grc_view: parameters as GRC resolves and evaluates
    them), the project folder, the run's block library and the final answer;
  * a cap failure (eval plan §4.4: "A run that hits a cap is a failure") fails whatever the end
    state; an infrastructure error makes the episode invalid (re-run, not scored);
  * a missing final graph fails every graph assertion, an empty final answer fails every answer
    assertion, and a task with no assertion at all is refused (AcceptanceError).

task.json "acceptance" (all keys optional, at least one assertion required; checked in this order):

  graph            {"path": "<file in the project folder>"} the graph to grade; default: the
                   episode's final active graph (its saved file, EpisodeResult.final_graph_path)
  ladder           {"rung": "V1".."V5", "observation": {...}, "oracle": {...}}
                   ladder.check(final graph text, max_rung=rung) must pass every rung up to `rung`
                   (observation/oracle in the Tier-F format; V5 needs an oracle; graph_checks such
                   as same_blocks_as_start compare with the task's start graph)
  blocks_present   [{"id": str | "ids": [...], "min": 1, "max": null, "state": "enabled" | "any",
                     "params": {param: constraint}}]
                   at least `min` (at most `max`) blocks of these ids, enabled unless state "any",
                   whose parameters meet every constraint
  blocks_absent    [{"id" | "ids"}]   no enabled block of these ids
  connections      [{"src": {"id" | "ids", "port": "0", "params": {...}},
                     "dst": {...}, "min": 1}]
                   connections between enabled blocks, matched by block ids, port keys and
                   parameter constraints (never by instance names)
  files            [{"path": "<in the project folder>", "regex": [...], "json_keys": ["a", "a.b"],
                     "yaml_keys": [...], "absent": false,
                     "csv_sizes": {"dir": "<folder>", "header": ["file", "bytes"], "names": [...]}}]
                   the file exists (or, with absent, does not); every regex matches (re.search,
                   MULTILINE); the file parses as JSON / YAML holding every dotted key path; with
                   csv_sizes it is a CSV manifest: the header row, then one "name,size" row per file of
                   <folder> it lists, each size equal to that file's size, every name in names listed
  hier_block       {"block_id": str, "label": str?, "inputs": int?, "outputs": int?,
                    "params": [param ids]?}
                   <block_id>.block.yml and <block_id>.py are in the run's block library
                   (<run>/home/.grc_gnuradio, the episode's GRC_HIER_PATH, where save_block writes),
                   the YAML declares that id (label, port counts, parameters as given), and GNU
                   Radio's catalog loads it from that library
  graph_unchanged  true   the final graph equals the start graph as GRC resolves both (same block
                   names, ids, states, parameter values and connections; positions ignored)
  answer           {"facts": [{"name": str, "source": SOURCE, "match": MATCH, "tol"?, "rel_tol"?}]}
                   rule-checked facts in the final answer; each expected value is DERIVED at
                   grading time from the graph or block YAML, never written by hand:
                     SOURCE {"graph": "start" | "final", "block": {"name": n} | {"id": block id},
                             "param": p, "use": "evaluated" (default) | "raw", "regex"?: r}
                            (regex: the value is the first group of re.search(r, value), e.g. one
                            number out of a message strobe's pmt.to_pmt({...}) expression)
                            {"block_yaml": block id, "path": "parameters[id=type].options"}
                            {"file": "<in the project folder>", "dtype": "float32" | "complex64" | ...,
                             "skip"?: n, "n"?: n, "feature": a numeric oracle feature, its options}
                            (a value measured from a file the run wrote, e.g. the rms of recorded audio)
                     MATCH  "number" (a number in the answer equals the value within tol/rel_tol;
                            SI prefixes k M G m u count), "text" (the value as a whole word,
                            case-insensitive), "all_of" / "any_of" (every / some item of a list)

Parameter constraints (blocks_present, connections): a number (numeric equality on GRC's
evaluated value, default tolerance 1e-9 relative), a string (raw text equality, quotes ignored;
numeric if both parse as numbers), or {"value": x, "tol": a, "rel_tol": r} / {"equals": s} /
{"in": [...]} / {"regex": r}.

Result (deterministic for a given end state):
    {"passed": bool, "valid": bool, "reasons": [why it failed], "checks": [{"name", "passed",
     "detail", ...}], "graph": {"path", "sha256", "exists"}, "status": episode status,
     "grader": GRADER_VERSION}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..common.io import read_json, sha256_file
from ..common.paths import GNURADIO_BLOCKS
from . import isolation
from .task import Task, load_task

GRADER_VERSION = "grcbench.tier_a.grader/1"
# oracle features that measure a number (a "file" fact source; grcbench.tier_f.oracles)
NUMERIC_FEATURES = ("peak_freq", "band_fraction", "rms", "mean", "max_abs", "min", "max", "max_int_dev",
                    "int_fraction")
ACCEPTANCE_KEYS = ("graph", "ladder", "blocks_present", "blocks_absent", "connections", "files",
                   "hier_block", "graph_unchanged", "answer")
ASSERTIONS = ACCEPTANCE_KEYS[1:]
GRAPH_ASSERTIONS = ("ladder", "blocks_present", "blocks_absent", "connections", "graph_unchanged")
RUNGS = ("V1", "V2", "V3", "V4", "V5")
IGNORED_KEYS = ("note",)                        # free text a task may carry next to its assertions
MATCHES = ("number", "text", "all_of", "any_of")
REL_TOL_DEFAULT = 1e-9
SI = {"k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9, "m": 1e-3, "u": 1e-6, "µ": 1e-6}


class AcceptanceError(ValueError):
    """A task's acceptance block cannot be graded (unknown keys, no assertion, bad specs)."""


@dataclass
class EndState:
    """What the grader looks at: an episode's end state (from its result) or a synthetic one."""

    project: Path
    status: str = "completed"
    end_reason: str = ""
    caps_hit: list[dict[str, Any]] = field(default_factory=list)
    final_graph_path: str | None = None
    final_text: str = ""
    library_dir: Path | None = None
    run_dir: Path | None = None
    synthetic: bool = False


def end_state(episode: Any) -> EndState:
    """The EndState of an EpisodeResult, its dict, or a run directory (result.json)."""
    if isinstance(episode, EndState):
        return episode
    if isinstance(episode, (str, Path)):
        episode = read_json(Path(episode) / "result.json")
    data = episode.to_dict() if hasattr(episode, "to_dict") else dict(episode)
    run_dir = Path(data["run_dir"])
    lay = isolation.RunLayout(run_dir)
    return EndState(project=lay.project, status=data.get("status", "completed"),
                    end_reason=data.get("end_reason", ""), caps_hit=list(data.get("caps_hit") or []),
                    final_graph_path=data.get("final_graph_path"), final_text=data.get("final_text") or "",
                    library_dir=lay.hier_dir, run_dir=run_dir)


# --------------------------------------------------------------------------------------------
# Acceptance specs
# --------------------------------------------------------------------------------------------

def _ids(spec: dict[str, Any]) -> list[str]:
    if "ids" in spec:
        return [str(i) for i in spec["ids"]]
    if "id" in spec:
        return [str(spec["id"])]
    return []


def acceptance_errors(acceptance: dict[str, Any]) -> list[str]:
    """Problems with an acceptance block (empty if it can be graded)."""
    errs: list[str] = []
    if not isinstance(acceptance, dict):
        return ["acceptance must be an object"]
    unknown = sorted(set(acceptance) - set(ACCEPTANCE_KEYS) - set(IGNORED_KEYS))
    if unknown:
        errs.append(f"unknown acceptance keys {unknown}")
    if not any(k in acceptance for k in ASSERTIONS):
        errs.append("no assertion: an end state cannot be judged (eval plan §4.4)")
    lad = acceptance.get("ladder")
    if lad is not None:
        if lad.get("rung") not in RUNGS:
            errs.append(f"ladder.rung must be one of {list(RUNGS)}")
        elif lad["rung"] == "V5":
            oracle = lad.get("oracle") or {}
            if not oracle.get("checks") and not oracle.get("graph_checks"):
                errs.append("ladder.rung V5 needs an oracle with checks or graph_checks")
            if oracle.get("checks") and not (lad.get("observation") or {}).get("points"):
                errs.append("ladder.rung V5 with signal checks needs observation points")
    for key in ("blocks_present", "blocks_absent"):
        for i, spec in enumerate(acceptance.get(key) or []):
            if not _ids(spec):
                errs.append(f"{key}[{i}] names no block id")
    for i, spec in enumerate(acceptance.get("connections") or []):
        for end in ("src", "dst"):
            if not isinstance(spec.get(end), dict) or not _ids(spec[end]):
                errs.append(f"connections[{i}].{end} names no block id")
    for i, spec in enumerate(acceptance.get("files") or []):
        p = spec.get("path")
        if not p or Path(p).is_absolute() or ".." in Path(p).parts:
            errs.append(f"files[{i}].path must be a relative path inside the project folder")
        for rx in spec.get("regex") or []:
            try:
                re.compile(rx)
            except re.error as exc:
                errs.append(f"files[{i}] regex {rx!r}: {exc}")
        cs = spec.get("csv_sizes")
        if cs is not None:
            d = cs.get("dir")
            if not d or Path(d).is_absolute() or ".." in Path(d).parts:
                errs.append(f"files[{i}].csv_sizes.dir must be a relative folder inside the project folder")
            if not isinstance(cs.get("names"), list):
                errs.append(f"files[{i}].csv_sizes.names must be a list")
    hb = acceptance.get("hier_block")
    if hb is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(hb.get("block_id", ""))):
        errs.append("hier_block.block_id must be a Python identifier")
    if "graph_unchanged" in acceptance and acceptance["graph_unchanged"] is not True:
        errs.append("graph_unchanged must be true when present")
    ans = acceptance.get("answer")
    if ans is not None:
        facts = ans.get("facts") or []
        if not facts:
            errs.append("answer.facts is empty")
        for i, fact in enumerate(facts):
            src = fact.get("source") or {}
            if fact.get("match") not in MATCHES:
                errs.append(f"answer.facts[{i}].match must be one of {list(MATCHES)}")
            if "graph" in src:
                if src["graph"] not in ("start", "final") or not isinstance(src.get("block"), dict) \
                        or not src.get("param"):
                    errs.append(f"answer.facts[{i}].source needs graph start|final, block and param")
            elif "block_yaml" in src:
                if not src.get("path"):
                    errs.append(f"answer.facts[{i}].source.block_yaml needs a path")
            elif "file" in src:
                f = src.get("file")
                if not f or Path(f).is_absolute() or ".." in Path(f).parts:
                    errs.append(f"answer.facts[{i}].source.file must be a relative path inside the project folder")
                if src.get("feature") not in NUMERIC_FEATURES:
                    errs.append(f"answer.facts[{i}].source.feature must be one of {list(NUMERIC_FEATURES)}")
            else:
                errs.append(f"answer.facts[{i}].source must be a graph or block_yaml source")
            if "regex" in src:
                try:
                    groups = re.compile(src["regex"]).groups
                except re.error as exc:
                    groups = f"invalid ({exc})"
                if groups != 1:
                    errs.append(f"answer.facts[{i}].source.regex needs exactly one group: {groups}")
    return errs


def _uses_start_graph(acceptance: dict[str, Any]) -> bool:
    if acceptance.get("graph_unchanged"):
        return True
    return any((f.get("source") or {}).get("graph") == "start"
               for f in (acceptance.get("answer") or {}).get("facts") or [])


# --------------------------------------------------------------------------------------------
# Value matching
# --------------------------------------------------------------------------------------------

def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().strip("'\"").replace("_", ""))
    except (TypeError, ValueError):
        return None


def close(actual: float, expected: float, tol: float | None = None, rel_tol: float | None = None) -> bool:
    """|actual - expected| within `tol` (absolute) or `rel_tol` (relative); by default 1e-9 relative."""
    diff = abs(actual - expected)
    if tol is None and rel_tol is None:
        return diff <= REL_TOL_DEFAULT * max(1.0, abs(expected))
    return (tol is not None and diff <= tol) or (rel_tol is not None and diff <= rel_tol * abs(expected))


def _text(value: Any) -> str:
    return str(value).strip().strip("'\"")


def param_ok(block: dict[str, Any], param: str, constraint: Any) -> tuple[bool, str]:
    """Does `block` (a _grc_view block) meet one parameter constraint (module docstring)?"""
    if param not in block["params"]:
        return False, f"no parameter {param!r}"
    raw = block["params"][param]
    evaluated = block["evaluated"].get(param)
    actual = _num(evaluated) if _num(evaluated) is not None else _num(raw)
    if isinstance(constraint, dict):
        if "value" in constraint:
            want = _num(constraint["value"])
            if want is None or actual is None:
                return False, f"{param}={raw!r} is not numeric"
            ok = close(actual, want, constraint.get("tol"), constraint.get("rel_tol"))
            return ok, f"{param}={raw!r} ({actual:g}) vs {want:g}"
        if "equals" in constraint:
            return _text(raw) == _text(constraint["equals"]), f"{param}={raw!r}"
        if "in" in constraint:
            return _text(raw) in {_text(v) for v in constraint["in"]}, f"{param}={raw!r}"
        if "regex" in constraint:
            return re.search(constraint["regex"], raw) is not None, f"{param}={raw!r}"
        return False, f"unknown constraint {constraint!r}"
    want = _num(constraint)
    if isinstance(constraint, (int, float)) or (want is not None and actual is not None):
        if actual is None or want is None:
            return False, f"{param}={raw!r} is not numeric"
        return close(actual, want), f"{param}={raw!r} ({actual:g}) vs {want:g}"
    ok = _text(raw) == _text(constraint) or (isinstance(evaluated, str) and _text(evaluated) == _text(constraint))
    return ok, f"{param}={raw!r}"


def _block_matches(block: dict[str, Any], spec: dict[str, Any]) -> bool:
    if block["key"] not in _ids(spec):
        return False
    state = spec.get("state", "enabled")
    if state != "any" and block["state"] != state:
        return False
    return all(param_ok(block, p, c)[0] for p, c in (spec.get("params") or {}).items())


_NUMBER = re.compile(r"(?<![\w.])([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)"
                     r"(?:\s?([kKMGmuµ])(?=[A-Za-z/]|(?<=[kKMG])\b))?")


def numbers_in(text: str) -> list[float]:
    """Every number written in `text`, plain and (when an SI prefix follows) scaled: "32 kHz" gives
    32 and 32000, "1,000" gives 1000, "2e3" gives 2000."""
    out: list[float] = []
    for m in _NUMBER.finditer(text or ""):
        try:
            value = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        out.append(value)
        if m.group(2):
            out.append(value * SI[m.group(2)])
    return out


def _word_in(item: Any, text: str) -> bool:
    word = _text(item)
    if not word:
        return False
    return re.search(r"(?<![\w])" + re.escape(word) + r"(?![\w])", text, re.IGNORECASE) is not None


def match_fact(value: Any, answer: str, match: str, tol: float | None = None,
               rel_tol: float | None = None) -> tuple[bool, str]:
    """Does the answer state `value` (module docstring, MATCH)?"""
    if match == "number":
        want = _num(value)
        if want is None:
            return False, f"expected value {value!r} is not a number"
        found = [x for x in numbers_in(answer) if close(x, want, tol, rel_tol)]
        return bool(found), f"expects {want:g}"
    if match == "text":
        return _word_in(value, answer), f"expects {_text(value)!r}"
    items = value if isinstance(value, list) else [value]
    hits = [i for i in items if _word_in(i, answer)]
    if match == "all_of":
        missing = [_text(i) for i in items if i not in hits]
        return not missing and bool(items), f"expects all of {[_text(i) for i in items]}" + (
            f"; missing {missing}" if missing else "")
    return bool(hits), f"expects one of {[_text(i) for i in items]}"


def yaml_path(data: Any, path: str) -> Any:
    """Follow "a.b[id=x].c.0" through YAML data (KeyError if a step does not exist)."""
    for token in re.findall(r"[^.\[\]]+|\[[^\]]*\]", path):
        if token.startswith("["):
            key, _, want = token[1:-1].partition("=")
            if not isinstance(data, list):
                raise KeyError(token)
            hits = [d for d in data if isinstance(d, dict) and str(d.get(key)) == want]
            if len(hits) != 1:
                raise KeyError(token)
            data = hits[0]
        elif isinstance(data, list):
            data = data[int(token)]
        elif isinstance(data, dict):
            if token not in data:
                raise KeyError(token)
            data = data[token]
        else:
            raise KeyError(token)
    return data


def block_yaml(block_id: str, blocks_dir: Path = GNURADIO_BLOCKS) -> dict[str, Any]:
    """The catalog YAML of a block (GNU Radio's own <id>.block.yml)."""
    import yaml
    path = Path(blocks_dir) / f"{block_id}.block.yml"
    if not path.is_file():
        raise KeyError(f"no catalog YAML for {block_id!r}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------------

def _check(name: str, passed: bool, detail: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail, **extra}


def _ladder_check(graph_text: str, spec: dict[str, Any], start_data: dict | None,
                  ladder_check: Callable[..., dict] | None) -> dict[str, Any]:
    from ..tier_f import ladder
    check = ladder_check or ladder.check
    rung = spec["rung"]
    n = RUNGS.index(rung) + 1
    task_like = {"observation": spec.get("observation") or {}, "oracle": spec.get("oracle") or {},
                 "_start_data": start_data}
    res = check(graph_text, task_like, max_rung=n)
    failed = [r for r in RUNGS[:n] if (res.get("rungs") or {}).get(r, {}).get("passed") is not True]
    first = failed[0] if failed else None
    rec = ((res.get("rungs") or {}).get(first) or {}) if first else {}
    errors = list(rec.get("errors", []))
    # an oracle failure has no error text: name the failed signal and graph checks instead
    errors += [f"{c.get('point')} {c.get('feature')} = {c.get('error') or c.get('value')}"
               for c in rec.get("checks", []) if not c.get("passed")]
    errors += [f"graph check {g.get('kind')}: {g.get('detail', '')}".rstrip(": ")
               for g in rec.get("graph_checks", []) if not g.get("passed")]
    detail = f"passes V1-{rung}" if first is None else f"fails at {first}: {'; '.join(map(str, errors))[:500]}"
    return _check("ladder", first is None, detail, required=rung, highest_passed=res.get("highest_passed"),
                  first_failed=first, hallucinated_blocks=res.get("hallucinated_blocks", []))


def _enabled(view: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {n: b for n, b in view["blocks"].items() if b["state"] == "enabled"}


def _blocks_present(view: dict[str, Any], spec: dict[str, Any], i: int) -> dict[str, Any]:
    hits = sorted(n for n, b in view["blocks"].items() if _block_matches(b, spec))
    lo, hi = spec.get("min", 1), spec.get("max")
    ok = len(hits) >= lo and (hi is None or len(hits) <= hi)
    near = []
    if not ok:
        for n, b in sorted(view["blocks"].items()):
            if b["key"] in _ids(spec):
                bad = [d for p, c in (spec.get("params") or {}).items() for good, d in [param_ok(b, p, c)] if not good]
                near.append(f"{n} ({b['state']}): " + ("; ".join(bad) if bad else "state"))
    detail = f"{len(hits)} matching block(s) {hits}" + (f"; candidates: {near}" if near else "")
    return _check(f"blocks_present[{i}]", ok, detail, ids=_ids(spec))


def _blocks_absent(view: dict[str, Any], spec: dict[str, Any], i: int) -> dict[str, Any]:
    hits = sorted(n for n, b in _enabled(view).items() if b["key"] in _ids(spec))
    return _check(f"blocks_absent[{i}]", not hits, f"{len(hits)} forbidden block(s) {hits}", ids=_ids(spec))


def _connections(view: dict[str, Any], spec: dict[str, Any], i: int) -> dict[str, Any]:
    blocks = _enabled(view)
    src, dst = spec["src"], spec["dst"]
    hits = []
    for s, sp, d, dp in view["connections"]:
        if s not in blocks or d not in blocks:
            continue
        if "port" in src and str(src["port"]) != sp:
            continue
        if "port" in dst and str(dst["port"]) != dp:
            continue
        if _block_matches(blocks[s], {**src, "state": "enabled"}) and _block_matches(blocks[d], {**dst, "state": "enabled"}):
            hits.append(f"{s}:{sp}->{d}:{dp}")
    lo = spec.get("min", 1)
    return _check(f"connections[{i}]", len(hits) >= lo, f"{len(hits)} matching connection(s) {hits}",
                  src=_ids(src), dst=_ids(dst))


def _files(project: Path, spec: dict[str, Any], i: int) -> dict[str, Any]:
    import yaml
    rel = spec["path"]
    path = project / rel
    name = f"files[{i}]"
    if not isolation.is_inside(path, project):
        return _check(name, False, f"{rel}: outside the project folder", path=rel)
    if spec.get("absent"):
        return _check(name, not path.exists(), f"{rel} {'exists' if path.exists() else 'is absent'}", path=rel)
    if not path.is_file():
        return _check(name, False, f"{rel} does not exist", path=rel)
    text = path.read_text(encoding="utf-8", errors="replace")
    problems = [f"regex {rx!r} does not match" for rx in spec.get("regex") or []
                if re.search(rx, text, re.MULTILINE) is None]
    for kind, loader in (("json_keys", json.loads), ("yaml_keys", yaml.safe_load)):
        keys = spec.get(kind) or []
        if not keys:
            continue
        try:
            data = loader(text)
        except Exception as exc:                # noqa: BLE001 - the file does not parse
            problems.append(f"does not parse as {kind.split('_')[0].upper()}: {type(exc).__name__}")
            continue
        for key in keys:
            try:
                yaml_path(data, key)
            except (KeyError, IndexError, ValueError, TypeError):
                problems.append(f"missing key {key!r}")
    if spec.get("csv_sizes") is not None:
        problems += _csv_sizes(project, text, spec["csv_sizes"])
    return _check(name, not problems, f"{rel}: " + ("; ".join(problems) if problems else "ok"), path=rel)


def _csv_sizes(project: Path, text: str, cs: dict[str, Any]) -> list[str]:
    """Problems of a CSV manifest of file sizes (files[].csv_sizes): header, sizes, the names listed."""
    import csv
    import io
    rows = [r for r in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in r)]
    header = [h.strip() for h in cs.get("header") or ["file", "bytes"]]
    if not rows or [h.strip() for h in rows[0]] != header:
        return [f"the header line is not {','.join(header)}"]
    folder, problems, listed = project / cs["dir"], [], set()
    for row in rows[1:]:
        if len(row) != 2:
            problems.append(f"row {row!r} does not have two fields")
            continue
        name, size = row[0].strip(), row[1].strip()
        listed.add(name)
        path = folder / name
        if not isolation.is_inside(path, project) or not path.is_file():
            problems.append(f"{cs['dir']}/{name} does not exist")
        elif not size.isdigit() or int(size) != path.stat().st_size:
            problems.append(f"{name}: size {size} in the manifest, {path.stat().st_size} bytes on disk")
    problems += [f"{n} is not listed" for n in cs.get("names") or [] if n not in listed]
    return problems


def _hier_block(library: Path | None, spec: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    import yaml
    bid = spec["block_id"]
    if library is None or not library.is_dir():
        return _check("hier_block", False, "the run has no block library", block_id=bid)
    yml, py = library / f"{bid}.block.yml", library / f"{bid}.py"
    problems = [f"{p.name} is not in the run's block library" for p in (yml, py) if not p.is_file()]
    if yml.is_file():
        try:
            data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            data = {}
            problems.append(f"{yml.name} does not parse: {exc}")
        if data.get("id") != bid:
            problems.append(f"{yml.name} declares id {data.get('id')!r}")
        if "label" in spec and str(data.get("label")) != spec["label"]:
            problems.append(f"label {data.get('label')!r} != {spec['label']!r}")
        for kind in ("inputs", "outputs"):
            if kind in spec and len(data.get(kind) or []) != spec[kind]:
                problems.append(f"{len(data.get(kind) or [])} {kind}, expected {spec[kind]}")
        have = {p.get("id") for p in data.get("parameters") or [] if isinstance(p, dict)}
        missing = [p for p in spec.get("params") or [] if p not in have]
        if missing:
            problems.append(f"parameters {missing} missing")
    if not (catalog.get(bid) or {}).get("in_catalog"):
        problems.append("GNU Radio's catalog does not load it from the run's library")
    return _check("hier_block", not problems, "; ".join(problems) or f"{bid} saved and loadable", block_id=bid)


def graph_diff(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Differences between two _grc_view views (block names, ids, states, parameters, connections)."""
    out: list[str] = []
    for name in sorted(set(a["blocks"]) | set(b["blocks"])):
        x, y = a["blocks"].get(name), b["blocks"].get(name)
        if x is None or y is None:
            out.append(f"block {name!r} only in the {'second' if x is None else 'first'} graph")
            continue
        if x["key"] != y["key"]:
            out.append(f"{name}: id {x['key']} != {y['key']}")
            continue
        if x["state"] != y["state"]:
            out.append(f"{name}: state {x['state']} != {y['state']}")
        for k in sorted(set(x["params"]) | set(y["params"])):
            if x["params"].get(k) != y["params"].get(k):
                out.append(f"{name}.{k}: {x['params'].get(k)!r} != {y['params'].get(k)!r}")
    ca, cb = {tuple(c) for c in a["connections"]}, {tuple(c) for c in b["connections"]}
    out += [f"connection {':'.join(c[:2])}->{':'.join(c[2:])} only in the first graph" for c in sorted(ca - cb)]
    out += [f"connection {':'.join(c[:2])}->{':'.join(c[2:])} only in the second graph" for c in sorted(cb - ca)]
    return out


def _fact_value(fact: dict[str, Any], graphs: dict[str, dict[str, Any] | None], project: Path | None = None) -> Any:
    src = fact["source"]
    if "block_yaml" in src:
        return yaml_path(block_yaml(src["block_yaml"]), src["path"])
    if "file" in src:
        return _measured(src, project)
    view = graphs.get(src["graph"])
    if view is None or not view.get("ok"):
        raise KeyError(f"the {src['graph']} graph is not available")
    sel = src["block"]
    hits = [b for n, b in sorted(view["blocks"].items())
            if ("name" not in sel or n == sel["name"]) and ("id" not in sel or b["key"] == sel["id"])]
    if len(hits) != 1:
        raise KeyError(f"block selector {sel} matches {len(hits)} blocks in the {src['graph']} graph")
    block = hits[0]
    if src["param"] not in block["params"]:
        raise KeyError(f"no parameter {src['param']!r}")
    if src.get("use", "evaluated") == "raw":
        value = block["params"][src["param"]]
    else:
        value = block["evaluated"].get(src["param"])
        value = value if value is not None else block["params"][src["param"]]
    if "regex" in src:                     # one value out of an expression, e.g. a message strobe's message
        m = re.search(src["regex"], str(value))
        if not m:
            raise KeyError(f"regex {src['regex']!r} does not match {value!r}")
        value = m.group(1)
    return value


def _measured(src: dict[str, Any], project: Path | None) -> float:
    """An oracle feature of a file the run wrote ("file" fact source)."""
    import numpy as np
    from ..tier_f import oracles
    path = (project or Path(".")) / src["file"]
    if project is None or not isolation.is_inside(path, project) or not path.is_file():
        raise KeyError(f"{src['file']} does not exist in the project folder")
    x = np.fromfile(path, dtype=np.dtype(src.get("dtype", "float32")))
    skip = int(src.get("skip", 0))
    x = x[skip:skip + int(src["n"])] if "n" in src else x[skip:]
    if not len(x):
        raise ValueError(f"{src['file']} holds no samples after skipping {skip}")
    chk = {k: v for k, v in src.items() if k not in ("file", "dtype", "skip", "n")}
    return float(oracles.feature_value(chk["feature"], x, chk))


def _answer(answer_text: str, spec: dict[str, Any], graphs: dict[str, dict[str, Any] | None],
            project: Path | None = None) -> list[dict[str, Any]]:
    out = []
    empty = not (answer_text or "").strip()
    for i, fact in enumerate(spec.get("facts") or []):
        name = f"answer[{i}]"
        label = fact.get("name") or name
        try:
            value = _fact_value(fact, graphs, project)
        except (KeyError, IndexError, ValueError) as exc:
            out.append(_check(name, False, f"{label}: expected value not derivable: {exc}", fact=label))
            continue
        if empty:
            out.append(_check(name, False, f"{label}: the final answer is empty", fact=label, expected=value))
            continue
        ok, detail = match_fact(value, answer_text, fact["match"], fact.get("tol"), fact.get("rel_tol"))
        out.append(_check(name, ok, f"{label}: {detail}", fact=label, expected=value))
    return out


# --------------------------------------------------------------------------------------------
# grade
# --------------------------------------------------------------------------------------------

def final_graph_file(task: Task, end: EndState) -> Path | None:
    """The graph the task is graded on: acceptance.graph.path in the project folder, else the
    episode's final active graph file. None if there is none."""
    rel = (task.acceptance.get("graph") or {}).get("path")
    if rel:
        return end.project / rel
    return Path(end.final_graph_path) if end.final_graph_path else None


def grade(task: Task | Path | str, episode: Any, *, ladder_check: Callable[..., dict] | None = None
          ) -> dict[str, Any]:
    """Grade one episode's end state against the task's acceptance block (module docstring)."""
    import yaml

    from . import _grc_view
    task = task if isinstance(task, Task) else load_task(task)
    acc = task.acceptance
    errs = acceptance_errors(acc)
    if errs:
        raise AcceptanceError(f"{task.id}: " + "; ".join(errs))
    end = end_state(episode)
    out: dict[str, Any] = {"task_id": task.id, "passed": False, "valid": end.status != "infrastructure_error",
                           "status": end.status, "reasons": [], "checks": [], "graph": {},
                           "grader": GRADER_VERSION, "synthetic": end.synthetic}
    if not out["valid"]:
        out["reasons"].append("episode invalid (infrastructure error): re-run, not scored")
        return out
    if end.status == "cap_failure":
        cap = (end.caps_hit[0] if end.caps_hit else {}).get("cap", "?")
        out["reasons"].append(f"cap failure ({cap}): a run that hits a cap is a failure (eval plan §4.4)")

    graph = final_graph_file(task, end)
    exists = bool(graph is not None and graph.is_file() and isolation.is_inside(graph, end.project))
    out["graph"] = {"path": str(graph) if graph else None, "exists": exists,
                    "sha256": sha256_file(graph) if exists else None}
    start = task.start_dir / task.active_graph if task.active_graph is not None else None
    want_start = _uses_start_graph(acc) and start is not None and start.is_file()
    needs_view = any(k in acc for k in ("blocks_present", "blocks_absent", "connections", "graph_unchanged",
                                        "hier_block", "answer"))
    views: dict[str, Any] = {"graphs": {}, "library_blocks": {}}
    if needs_view:
        paths: dict[str, Any] = {"final": graph if exists else None}
        if want_start:
            paths["start"] = start
        hb = acc.get("hier_block")
        views = _grc_view.views(paths, library_dir=end.library_dir,
                                library_blocks=[hb["block_id"]] if hb else [])
    final_view = views["graphs"].get("final") if exists else None
    graphs = {"final": final_view, "start": views["graphs"].get("start")}
    missing = "the final graph is missing" if not exists else (
        f"the final graph does not load: {final_view.get('error')}" if final_view is not None and not final_view.get("ok") else None)

    checks: list[dict[str, Any]] = []
    if "ladder" in acc:
        if not exists:
            checks.append(_check("ladder", False, "the final graph is missing", required=acc["ladder"]["rung"]))
        else:
            start_data = yaml.safe_load(start.read_text(encoding="utf-8")) if start is not None and start.is_file() else None
            checks.append(_ladder_check(graph.read_text(encoding="utf-8"), acc["ladder"], start_data, ladder_check))
    for key, fn in (("blocks_present", _blocks_present), ("blocks_absent", _blocks_absent),
                    ("connections", _connections)):
        for i, spec in enumerate(acc.get(key) or []):
            checks.append(_check(f"{key}[{i}]", False, missing) if missing else fn(final_view, spec, i))
    for i, spec in enumerate(acc.get("files") or []):
        checks.append(_files(end.project, spec, i))
    if "hier_block" in acc:
        checks.append(_hier_block(end.library_dir, acc["hier_block"], views.get("library_blocks", {})))
    if acc.get("graph_unchanged"):
        sview = graphs.get("start")
        if missing:
            checks.append(_check("graph_unchanged", False, missing))
        elif sview is None or not sview.get("ok"):
            checks.append(_check("graph_unchanged", False, "the task has no loadable start graph"))
        else:
            diff = graph_diff(sview, final_view)
            checks.append(_check("graph_unchanged", not diff, "unchanged" if not diff else "; ".join(diff[:20]),
                                 differences=len(diff)))
    if "answer" in acc:
        checks += _answer(end.final_text, acc["answer"], graphs, end.project)

    out["checks"] = checks
    out["reasons"] += [f"{c['name']}: {c['detail']}" for c in checks if not c["passed"]]
    out["passed"] = bool(checks) and all(c["passed"] for c in checks) and end.status != "cap_failure"
    return out
