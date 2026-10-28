"""`./gb tier-k judge DIR`: the judge runner for a routed Tier-K run (pre-registration §5.1, §8 decision 7).

It reads a run that `./gb tier-k score DIR` has routed (DIR/manifest.json, answers.jsonl, summary.json,
judge_queue.jsonl) and sends every queued answer to the judge (judge_prompt):

  1. Checks before any request. Every queued request must be what judge_prompt.build_messages builds
     for its item, rubric and answer under the current judge version, and the run's summary must carry
     the current rules version; otherwise the run is scored again first (`./gb tier-k score DIR`). A
     test-split run needs allow_test (--allow-test) and calibration (--calibration FILE), a record of a
     passed calibration of this judge prompt ({"passed": true, "prompt_version", "prompt_sha256"}): the judge
     grades no test answer before it passes (§5.1). The file's SHA-256 goes with every request.
  2. Each answer (FAILURE_POLICY): the primary judge with the JUDGE_REQUEST settings and seed + attempt
     index, at most MAX_ATTEMPTS times; a reply that parse_verdict rejects (with the rules' result, so
     the values of numbers the rules decided are not checked) or that is cut at max_tokens is a
     malformed verdict and is asked again. Then the fallback judge once, with the base seed. Then the
     answer scores 0 (judge_failed).
  3. A request that fails (an HTTP error, a timeout, a reply without choices) is sent again, at most
     MAX_TRIES times. If it keeps failing, the answer is request_failed: never a malformed verdict, not
     scored, and sent again when the command runs again (only answers without a final result are
     sent). MAX_FAILED_IN_A_ROW such answers in a row stop the run. Every answer of the run is routed
     again, and the queue must hold exactly those that go to the judge. A score row is written as soon
     as it is decided, with the SHA-256 of its request and the judge and rules versions; a later run of
     the command reuses it only for that same request.

Writes DIR/judge_requests.jsonl (appended; one row per request: judge, model, attempt, try, seed, the
body fields that set the reasoning mode, usage, finish reason, validity and errors, the judge prompt
version and the calibration file's hash), DIR/judge_scores.jsonl (explain_score per judged answer, or
request_failed) and adds to DIR/summary.json: n_judged, n_judge_failed, n_unknown_unit,
n_judge_request_failed, report_counts over every answer of the run (the rules' answers too) and
"judge": "complete" or "incomplete". API keys come from the environment variables named on the
command line; they are sent as a bearer token and never written.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from grcbench.common.io import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from grcbench.tier_k import judge_prompt as jp
from grcbench.tier_k import rules
from grcbench.tier_k import run as krun

MAX_TRIES = krun.MAX_TRIES                # requests per judge attempt before the answer is request_failed
MAX_FAILED_IN_A_ROW = 3                   # request_failed answers in a row: the judge is down, the run stops
TIMEOUT_S = 1800.0                        # per request (a 16,384-token reply)
REQUEST_KEYS = ("temperature", "top_p", "max_tokens", "response_format")   # JUDGE_REQUEST fields sent as is


class JudgeRefused(RuntimeError):
    """The run cannot be judged as it is (nothing was sent)."""


@dataclass(frozen=True)
class Endpoint:
    """One judge: an OpenAI-compatible base URL, the model name, the environment variable that holds
    its API key (None: no key) and the body fields that set its reasoning mode (recorded per request)."""

    base_url: str
    model: str
    api_key_env: str | None = None
    reasoning: dict[str, Any] = field(default_factory=dict)


def _request(ep: Endpoint, messages: list[dict[str, str]], seed: int, timeout: float) -> dict[str, Any]:
    """One chat completion with the JUDGE_REQUEST settings; raises run.ChatError when there is no reply."""
    body = {**ep.reasoning, **{k: jp.JUDGE_REQUEST[k] for k in REQUEST_KEYS}, "seed": seed,
            "model": ep.model, "messages": messages}
    headers = {"Content-Type": "application/json"}
    if ep.api_key_env:
        headers["Authorization"] = f"Bearer {os.environ[ep.api_key_env]}"
    req = urllib.request.Request(ep.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers)
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:      # noqa: S310 - a declared judge API
            reply = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError, ValueError) as exc:
        raise krun.ChatError(f"{type(exc).__name__}: {exc}") from None
    choices = reply.get("choices") if isinstance(reply, dict) else None
    if not choices:
        raise krun.ChatError(f"reply without choices: {json.dumps(reply)[:300]}")
    msg = choices[0].get("message") or {}
    return {"text": msg.get("content") or "", "finish_reason": choices[0].get("finish_reason"),
            "usage": reply.get("usage"), "seconds": round(time.monotonic() - t0, 3)}


def _judge_answer(messages: list[dict[str, str]], rubric: dict[str, Any], rules_result: dict[str, Any] | None,
                  judges: dict[str, Endpoint], log: Callable[..., None], timeout: float,
                  retry_wait: float) -> dict[str, Any]:
    """The verdict for one answer under FAILURE_POLICY: {"judge", "verdict"}, {"judge_failed"} or
    {"request_failed", "error"}."""
    base = jp.JUDGE_REQUEST["seed"]
    plan = [("primary", base + i, i) for i in range(jp.MAX_ATTEMPTS)] + [("fallback", base, 0)]
    for label, seed, attempt in plan:
        for t in range(MAX_TRIES):
            try:
                got = _request(judges[label], messages, seed, timeout)
                break
            except krun.ChatError as exc:
                log(judge=label, attempt=attempt, attempt_try=t, seed=seed, error=str(exc))
                if t + 1 == MAX_TRIES:
                    return {"request_failed": True, "error": str(exc)}
                time.sleep(retry_wait)
        if got["finish_reason"] == "length":
            errors = [f"reply cut at max_tokens ({jp.JUDGE_REQUEST['max_tokens']})"]
        else:
            try:
                verdict = jp.parse_verdict(got["text"], rubric, rules_result=rules_result)
                errors = []
            except jp.VerdictError as exc:
                errors = exc.errors
        log(judge=label, attempt=attempt, attempt_try=t, seed=seed, finish_reason=got["finish_reason"],
            usage=got["usage"], seconds=got["seconds"], valid=not errors, errors=errors[:10])
        if not errors:
            return {"judge": label, "verdict": verdict}
    return {"judge_failed": True}


def _routed(item: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    return krun.route_answer(item, answer["text"], truncated=bool(answer.get("truncated")),
                             reasoning=answer.get("reasoning"))


def judge(out_dir: Path, *, primary: Endpoint, fallback: Endpoint, allow_test: bool = False,
          calibration: Path | None = None, timeout: float = TIMEOUT_S, retry_wait: float = 5.0) -> dict[str, Any]:
    """Judge every queued answer of a routed run (module docstring); returns the updated summary."""
    out_dir = Path(out_dir)
    manifest = read_json(out_dir / "manifest.json")
    summary = read_json(out_dir / "summary.json")
    for ep in (primary, fallback):
        if ep.api_key_env and not os.environ.get(ep.api_key_env):
            raise JudgeRefused(f"the API key variable {ep.api_key_env} is not set")
    if manifest["split"] == "test":
        if not allow_test:
            raise JudgeRefused("judging test-split answers needs allow_test (--allow-test) and a passed calibration")
        if calibration is None:
            raise JudgeRefused(f"judging test-split answers needs a passed calibration of {jp.PROMPT_VERSION} "
                               "(--calibration FILE)")
        record = read_json(calibration)
        if (record.get("passed") is not True or record.get("prompt_version") != jp.PROMPT_VERSION
                or record.get("prompt_sha256") != jp.prompt_sha256()):
            raise JudgeRefused(f"{calibration} records no passed calibration of {jp.PROMPT_VERSION} "
                               f"(prompt SHA-256 {jp.prompt_sha256()[:12]}...)")
    calibration_sha256 = sha256_file(Path(calibration)) if calibration is not None else None
    if summary.get("rules_version") != rules.RULES_VERSION:
        raise JudgeRefused(f"the run was routed with {summary.get('rules_version')}, not {rules.RULES_VERSION}: "
                           "score the run again (./gb tier-k score DIR)")
    items = {i["id"]: i for i in krun.load_split(manifest["split"])}
    answers = read_jsonl(out_dir / "answers.jsonl")
    by_key = {(a["item_id"], a["sample"]): a for a in answers}
    missing = sorted({a["item_id"] for a in answers if a["item_id"] not in items})
    if missing:
        raise JudgeRefused(f"answers for items not in the {manifest['split']} split: {missing[:10]}")
    # every answer routed again: the queue must hold exactly the answers that go to the judge, each with the
    # request the current template builds (an answer whose rubric was approved after scoring is not lost)
    routed_all = {k: _routed(items[a["item_id"]], a) for k, a in by_key.items()
                  if not a.get("error") and a["item_id"] in items}
    to_judge = {k: r for k, r in routed_all.items() if r["route"] == "judge"}
    queue = read_jsonl(out_dir / "judge_queue.jsonl")
    queued = {(q["item_id"], q["sample"]): q for q in queue}
    if (len(queued) != len(queue) or set(queued) != set(to_judge)
            or any(to_judge[k]["judge_messages"] != q["messages"] for k, q in queued.items())):
        raise JudgeRefused(f"the judge queue does not match the run's answers under {rules.RULES_VERSION} and judge "
                           f"template {jp.PROMPT_VERSION}: score the run again (./gb tier-k score DIR)")
    work = [((q["item_id"], q["sample"]), q, krun.rubric_for(items[q["item_id"]]), to_judge[(q["item_id"], q["sample"])])
            for q in queue]

    scores_path, requests_path = out_dir / "judge_scores.jsonl", out_dir / "judge_requests.jsonl"
    scores: dict[tuple, dict[str, Any]] = {}
    for r in read_jsonl(scores_path) if scores_path.exists() else []:
        scores[(r["item_id"], r["sample"])] = r                  # appended as decided: the last row counts
    judges = {"primary": primary, "fallback": fallback}
    failed_in_a_row = 0
    with open(requests_path, "a", encoding="utf-8", newline="\n") as log_fh, \
            open(scores_path, "a", encoding="utf-8", newline="\n") as score_fh:
        for key, q, rubric, routed in work:
            stamp = {"request_sha256": _request_sha256(q["messages"]), "judge_prompt_version": jp.PROMPT_VERSION,
                     "rules_version": rules.RULES_VERSION}
            old = scores.get(key)
            if (old and not old.get("request_failed") and all(old.get(k) == v for k, v in stamp.items())
                    and ("verdict" in old or old.get("judge_failed"))):
                # judged earlier with this same request: the verdict is scored again with the current rubric
                # (its numbers, tolerances and weights are not part of the request). A verdict that no longer
                # fits (a number the rules decided before and leave to the judge now) is asked for again.
                try:
                    explained = jp.explain_score(old.get("verdict"), rubric, rules_result=routed["rules_result"],
                                                 judge_failed=bool(old.get("judge_failed")),
                                                 answer=by_key[key]["text"])
                except jp.VerdictError:                  # the stored verdict does not fit the current rubric
                    explained = None
                if explained is not None:
                    scores[key] = {**old, "score": explained["score"], "reason": explained["reason"],
                                   "explained": explained}
                    continue

            def log(**row: Any) -> None:
                ep = judges[row["judge"]]
                full = {"item_id": key[0], "sample": key[1], "model": ep.model, "reasoning_setting": ep.reasoning,
                        "judge_prompt_version": jp.PROMPT_VERSION, "calibration_sha256": calibration_sha256,
                        "utc": krun.runlog.utc_now(), **row}
                log_fh.write(json.dumps(full, ensure_ascii=False, sort_keys=True) + "\n")
                log_fh.flush()

            got = _judge_answer(q["messages"], rubric, routed["rules_result"], judges, log, timeout, retry_wait)
            if got.get("request_failed"):
                row = {"item_id": key[0], "sample": key[1], **stamp, "judge": None, "judge_failed": False,
                       "request_failed": True, "score": None, "reason": "request_failed", "error": got["error"]}
                failed_in_a_row += 1
            else:
                failed_in_a_row = 0
                explained = jp.explain_score(got.get("verdict"), rubric, rules_result=routed["rules_result"],
                                             judge_failed=bool(got.get("judge_failed")), answer=by_key[key]["text"])
                row = {"item_id": key[0], "sample": key[1], **stamp, "judge": got.get("judge"),
                       "judge_failed": bool(got.get("judge_failed")), "request_failed": False,
                       "verdict": got.get("verdict"), "score": explained["score"], "reason": explained["reason"],
                       "explained": explained}
            scores[key] = row
            score_fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")   # kept if the run stops
            score_fh.flush()
            if failed_in_a_row >= MAX_FAILED_IN_A_ROW:
                break
    current = [key for key, *_ in work]
    tmp = scores_path.with_suffix(".jsonl.tmp")              # the last row per answer, replaced in one step
    write_jsonl(tmp, [scores[key] for key in current if key in scores])
    os.replace(tmp, scores_path)
    return _update_summary(out_dir, summary, items, answers, scores, work, primary, fallback)


def _request_sha256(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _update_summary(out_dir: Path, summary: dict[str, Any], items: dict[str, Any], answers: list[dict[str, Any]],
                    scores: dict[tuple, dict[str, Any]], work: list, primary: Endpoint, fallback: Endpoint) -> dict[str, Any]:
    """The judge counts, and report_counts over every answer: the judged ones and those the rules decided."""
    results = []
    for a in answers:
        if a.get("error"):
            continue                                              # a failed answer request: not scored
        item = items[a["item_id"]]
        routed = _routed(item, a)
        s = scores.get((a["item_id"], a["sample"]))
        if routed["route"] == "judge":
            if s and not s.get("request_failed"):
                results.append(s["explained"])
        elif routed["route"] == "rules" and routed["reason"] != "canary_in_answer":
            rubric = krun.rubric_for(item)
            if rubric is None:
                continue
            if routed["rules_result"] is None and not a.get("truncated"):
                # an empty answer to a rubric without numbers: 0, with nothing for the rules to report
                results.append({"reason": routed["reason"], "score": 0.0, "rule_reasons": [routed["reason"]],
                                "redactions": 0})
            else:
                results.append(jp.explain_score(None, rubric, rules_result=routed["rules_result"],
                                                truncated=bool(a.get("truncated")), answer=a["text"]))
    counts = jp.report_counts(results)
    final = [scores[key] for key, *_ in work if key in scores and not scores[key].get("request_failed")]
    summary.update({
        "judge": "complete" if len(final) == len(work) else "incomplete",
        "judge_prompt_version": jp.PROMPT_VERSION,
        "judge_models": {"primary": primary.model, "fallback": fallback.model},
        "n_judged": len(final), "n_judge_failed": counts["judge_failed"], "n_unknown_unit": counts["unknown_unit"],
        "n_judge_request_failed": sum(1 for key, *_ in work if scores.get(key, {}).get("request_failed")),
        "report_counts": counts})
    write_json(out_dir / "summary.json", summary)
    return summary


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb tier-k judge", description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--base-url", required=True, help="the primary judge's OpenAI-compatible base URL")
    ap.add_argument("--model", default=jp.JUDGE_MODELS["primary"])
    ap.add_argument("--api-key-env", default=None, help="environment variable holding the primary judge's key")
    ap.add_argument("--reasoning", default="{}", help="body fields that set the primary judge's reasoning mode (JSON)")
    ap.add_argument("--fallback-base-url", required=True)
    ap.add_argument("--fallback-model", default=jp.JUDGE_MODELS["fallback"])
    ap.add_argument("--fallback-api-key-env", default=None)
    ap.add_argument("--fallback-reasoning", default="{}")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--calibration", type=Path, default=None, help="record of a passed calibration of this version")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    a = ap.parse_args(argv)
    primary = Endpoint(a.base_url, a.model, a.api_key_env, json.loads(a.reasoning))
    fallback = Endpoint(a.fallback_base_url, a.fallback_model, a.fallback_api_key_env, json.loads(a.fallback_reasoning))
    try:
        summary = judge(a.dir, primary=primary, fallback=fallback, allow_test=a.allow_test,
                        calibration=a.calibration, timeout=a.timeout)
    except JudgeRefused as exc:
        print(f"refused: {exc}")
        return 2
    print(json.dumps({k: summary[k] for k in ("judge", "n_judged", "n_judge_failed", "n_unknown_unit",
                                              "n_judge_request_failed", "report_counts")}, indent=2))
    return 0 if summary["judge"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
