"""`./gb tier-k run | score`: answer a split's Tier-K questions with one model, then score the answers.

    ./gb tier-k run --split dev --base-url URL --model NAME [--config-id ID] [--reason TEXT] [--n N]
                    [--b1] [--sampling JSON] [--limit M] [--out DIR]
    ./gb tier-k score DIR

run   sends each question once per sample to an OpenAI-compatible /chat/completions endpoint with the
      fixed Tier-K prompt (answer_prompt: B0, or B1 with the retriever's documentation block), at most
      answer_prompt.MAX_ANSWER_TOKENS tokens (the request's model, messages and max_tokens cannot be
      overridden by --sampling), and writes DIR/answers.jsonl (item_id, sample, text, reasoning,
      finish_reason, truncated, usage, seconds; error for a request that failed MAX_TRIES times) and
      DIR/manifest.json (split, model, configuration, settings, prompt version and hash, rules version,
      git commit). DIR must not hold a run already. A test-split run needs a configuration of the
      declared run list (runlog.guard) and allow_test (--allow-test); the run log records it before the
      first request.
score routes every answer (route_answer) and writes DIR/scores.jsonl (one row per answer),
      DIR/judge_queue.jsonl (the judge requests of the answers the rules cannot decide),
      DIR/no_reference.jsonl (answers with neither an approved rubric nor a block-question key) and
      DIR/summary.json (rules.rule_only_summary, counts per route and reason, and the per-model counts
      of decision 7). It refuses a run with fewer answer rows than questions x samples. A request that
      failed is not an answer: its row is "failed" (request_failed, no credit) and counted. The judge is
      not called here: the judge runner sends the queue and counts judge_failed
      (judge_prompt.report_counts).

route_answer(item, text, reasoning=None):
  1. a canary in the answer or in the server's separate reasoning text: 0 (canary_in_answer, a
     contamination signal); a truncated answer, or one empty after reasoning removal: 0 (rules P6),
     before the reference decides anything;
  2. the rubric: the item's approved rubric, else for a GNU Radio behaviour question the numbers-only
     key_rubric of its execution-derived key. A calculation key is a proposal until the team confirms
     it in the approved rubric, so it is never used here. Without a rubric: no_reference. An approved
     rubric that the rules or the judge template cannot read is an error, never a silent fallback;
  3. judge_prompt.route on the rule cascade's result: "rules" scores by judge_prompt.explain_score
     (the rules' credit with the DU rule), "judge" queues the judge request.
"""

from __future__ import annotations

import argparse
import http.client
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from grcbench.common.io import read_json, read_jsonl, write_json, write_jsonl
from grcbench.common.paths import ITEMS, WORK
from grcbench.runlog import log as runlog
from grcbench.runlog.log import RunRefused
from grcbench.tier_k import answer_prompt, rules
from grcbench.tier_k import judge_prompt as jp

__all__ = ["RunRefused", "chat", "load_split", "reference_for", "rubric_for", "route_answer", "run", "score"]

ITEMS_K = ITEMS / "tier_k"
SPLIT_FILES = {"dev": ("questions.jsonl",), "test": ("frozen_questions.jsonl", "block_questions.jsonl")}
KEY_FILES = ("block_answers.jsonl",)          # calc_answers.jsonl holds proposals: never a scoring key
RESERVED_KEYS = ("model", "messages", "max_tokens", "max_completion_tokens", "n_predict", "stream", "n")
MAX_TRIES = 3                 # requests per answer before it is recorded as failed
MAX_FAILED_IN_A_ROW = 3       # failed answers in a row: the server is down, the run stops (ChatError)


class ChatError(RuntimeError):
    """The server returned no answer (an HTTP error, a timeout, or a reply without choices)."""


def chat(base_url: str, model: str, messages: list[dict[str, str]], *, max_tokens: int,
         sampling: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    """One OpenAI-compatible chat completion: text, reasoning (a server's separate reasoning field),
    finish_reason, usage, seconds. Raises ChatError when there is no answer."""
    body = json.dumps({**sampling, "model": model, "messages": messages, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:      # noqa: S310 - a local or declared server
            reply = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError, ValueError) as exc:
        raise ChatError(f"{type(exc).__name__}: {exc}") from None
    choices = reply.get("choices") if isinstance(reply, dict) else None
    if not choices:
        raise ChatError(f"reply without choices: {json.dumps(reply)[:300]}")
    msg = choices[0].get("message") or {}
    return {"text": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or msg.get("reasoning"),
            "finish_reason": choices[0].get("finish_reason"), "usage": reply.get("usage"),
            "seconds": round(time.monotonic() - t0, 3)}


def load_split(split: str) -> list[dict[str, Any]]:
    """The split's Tier-K questions in file order, each with "rubric" (its record in rubrics.jsonl, or
    None) and "key" (its block-question answer key, or None)."""
    if split not in SPLIT_FILES:
        raise ValueError(f"split must be one of {sorted(SPLIT_FILES)}")
    folder = ITEMS_K / split
    rubrics = {r["question_id"]: r for r in read_jsonl(folder / "rubrics.jsonl")} if (folder / "rubrics.jsonl").exists() else {}
    keys: dict[str, Any] = {}
    for name in KEY_FILES:
        if (folder / name).exists():
            keys.update({r["question_id"]: r["reference"] for r in read_jsonl(folder / name)})
    items = []
    for name in SPLIT_FILES[split]:
        for row in read_jsonl(folder / name):
            items.append({**row, "rubric": rubrics.get(row["id"]), "key": keys.get(row["id"])})
    return items


def rubric_for(item: dict[str, Any]) -> dict[str, Any] | None:
    """The approved rubric, else a key rubric for a GNU Radio behaviour question, else None. An
    approved rubric the judge template or the rules cannot read raises ValueError (with the item id)."""
    rubric = item.get("rubric")
    if rubric is not None:
        try:
            jp.check_reviewed(rubric)
        except jp.RubricError:
            rubric = None                                   # not approved (yet): not used
        else:
            try:
                jp.numeric_reference(rubric)
            except (jp.RubricError, ValueError) as exc:
                raise ValueError(f"{item.get('id')}: the approved rubric cannot be read: {exc}") from None
            return rubric
    if item.get("item_kind") == "block_question" and item.get("key"):
        return jp.key_rubric(item["key"])
    return None


def reference_for(item: dict[str, Any]) -> Any | None:
    """The rule cascade's reference: the approved rubric's numbers, else a block question's key."""
    rubric = rubric_for(item)
    return jp.numeric_reference(rubric) if rubric is not None else None


def route_answer(item: dict[str, Any], text: str, *, truncated: bool = False,
                 reasoning: str | None = None) -> dict[str, Any]:
    """Where one answer goes and, on the rules route, its score (module docstring)."""
    if jp.contains_canary(text) or jp.contains_canary(reasoning or ""):
        zero = {"method": "numeric", "credit": 0.0, "details": {"reason": "canary_in_answer"}}
        return {"route": "rules", "method": "rules", "credit": 0.0, "reason": "canary_in_answer",
                "rules_result": zero if reference_for(item) is not None else None}
    if truncated or not rules.strip_reasoning(text):
        ref = reference_for(item)
        return {"route": "rules", "method": "rules", "credit": 0.0,
                "reason": "truncated" if truncated else "empty_answer",
                "rules_result": rules.score(text, ref, truncated=truncated) if ref is not None else None}
    rubric = rubric_for(item)
    if rubric is None:
        return {"route": "no_reference", "method": None, "credit": None, "reason": "no_reference"}
    ref = jp.numeric_reference(rubric)
    by_rules = rules.score(text, ref) if ref is not None else None
    reason = (by_rules["details"].get("reason") if by_rules else "open_ended")
    if jp.route(rubric, by_rules) == "rules":
        explained = jp.explain_score(None, rubric, rules_result=by_rules, answer=text)
        return {"route": "rules", "method": by_rules["method"], "credit": explained["score"], "reason": reason,
                "rule_reasons": explained["rule_reasons"], "rules_result": by_rules}
    return {"route": "judge", "method": by_rules["method"] if by_rules else None, "credit": None, "reason": reason,
            "rule_reasons": jp._rule_reasons(by_rules), "rules_result": by_rules,
            "judge_messages": jp.build_messages(item["question"], rubric, text),
            "redactions": jp.redact_self_names(rules.strip_reasoning(text))[1]}


def run(split: str, *, base_url: str, model: str, config_id: str, reason: str, n: int = 1, b1: bool = False,
        sampling: dict[str, Any] | None = None, out_dir: Path, allow_test: bool = False,
        limit: int | None = None, timeout: float = 600.0, retry_wait: float = 5.0,
        log_path: Path = runlog.LOG, run_list: Path = runlog.RUN_LIST) -> Path:
    """Answer the split's questions n times each (module docstring) and return out_dir."""
    if split == "test" and not allow_test:
        raise RunRefused("a test-split Tier-K run needs allow_test (--allow-test) and a declared configuration")
    if n < 1:
        raise ValueError("n must be >= 1")
    sampling = dict(sampling or {})
    reserved = sorted(set(sampling) & set(RESERVED_KEYS))
    if reserved:
        raise ValueError(f"--sampling cannot set {reserved}: the Tier-K request fixes them")
    out_dir = Path(out_dir)
    if (out_dir / "answers.jsonl").exists():
        raise ValueError(f"{out_dir} already holds a run; use a new directory")
    items = load_split(split)[:limit]
    settings = {"tier": "K", "n": n, "b1": b1, "sampling": sampling, "max_tokens": answer_prompt.MAX_ANSWER_TOKENS,
                "answer_prompt": answer_prompt.PROMPT_VERSION, "question_set": "reference"}
    runlog.start_run(split=split, config_id=config_id, model=model, settings=settings,
                     items=[i["id"] for i in items], reason=reason, tier="K", log_path=log_path, run_list=run_list)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "manifest.json", {
        "split": split, "model": model, "config_id": config_id, "settings": settings, "base_url": base_url,
        "answer_prompt_version": answer_prompt.PROMPT_VERSION, "answer_prompt_sha256": answer_prompt.prompt_sha256(),
        "rules_version": rules.RULES_VERSION, "judge_prompt_version": jp.PROMPT_VERSION,
        "git_commit": runlog.git_commit(), "git_dirty": runlog.git_dirty(), "n_items": len(items), "n": n,
        "started": runlog.utc_now()})
    max_tokens = answer_prompt.MAX_ANSWER_TOKENS
    failed_in_a_row = 0
    with open(out_dir / "answers.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            retrieved = None
            if b1:
                from grcbench.tier_k.retriever import b1_context
                retrieved = b1_context(item["question"])
            messages = answer_prompt.build_messages(item["question"], retrieved)
            for sample in range(n):
                row: dict[str, Any] = {"item_id": item["id"], "sample": sample}
                for attempt in range(MAX_TRIES):
                    try:
                        got = chat(base_url, model, messages, max_tokens=max_tokens, sampling=sampling, timeout=timeout)
                    except ChatError as exc:
                        row.update(text="", reasoning=None, finish_reason=None, usage=None, seconds=None,
                                   truncated=False, error=str(exc), attempts=attempt + 1)
                        if attempt + 1 < MAX_TRIES:
                            time.sleep(retry_wait)
                        continue
                    used = (got.get("usage") or {}).get("completion_tokens")
                    row = {"item_id": item["id"], "sample": sample, **got,
                           "truncated": got["finish_reason"] == "length" or (isinstance(used, int) and used >= max_tokens)}
                    break
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                fh.flush()
                failed_in_a_row = failed_in_a_row + 1 if row.get("error") else 0
                if failed_in_a_row >= MAX_FAILED_IN_A_ROW:
                    raise ChatError(f"{failed_in_a_row} answers in a row failed (last: {row['error']}); "
                                    f"the run stops, {out_dir / 'answers.jsonl'} holds what was answered")
    return out_dir


def score(out_dir: Path) -> dict[str, Any]:
    """Route and score every answer of a run (module docstring); returns the summary."""
    out_dir = Path(out_dir)
    manifest = read_json(out_dir / "manifest.json")
    items = {i["id"]: i for i in load_split(manifest["split"])}
    answers = read_jsonl(out_dir / "answers.jsonl")
    n = manifest.get("n", manifest.get("settings", {}).get("n", 1))
    pairs = [(a["item_id"], a["sample"]) for a in answers]
    if len(pairs) != manifest["n_items"] * n or len(set(pairs)) != len(pairs) or {s for _, s in pairs} - set(range(n)):
        raise ValueError(f"{out_dir}: {len(pairs)} answer rows ({len(set(pairs))} distinct), the run has "
                         f"{manifest['n_items']} x {n}; not scored")
    unknown = sorted({a["item_id"] for a in answers} - set(items))
    if unknown:
        raise ValueError(f"{out_dir}: answers for items not in the {manifest['split']} split: {unknown[:10]}")
    scores, queue, no_ref, by_rules = [], [], [], []
    redacted = canary = final_line = failed = 0
    for row in answers:
        item = items[row["item_id"]]
        if row.get("error"):
            failed += 1
            scores.append({"item_id": row["item_id"], "sample": row["sample"], "route": "failed", "method": None,
                           "credit": None, "reason": "request_failed"})
            continue
        r = route_answer(item, row["text"], truncated=bool(row.get("truncated")), reasoning=row.get("reasoning"))
        scores.append({"item_id": row["item_id"], "sample": row["sample"], "route": r["route"],
                       "method": r["method"], "credit": r["credit"], "reason": r["reason"]})
        if r.get("rules_result") is not None:
            by_rules.append(r["rules_result"])
        if r["route"] == "judge":
            queue.append({"item_id": row["item_id"], "sample": row["sample"], "messages": r["judge_messages"]})
            redacted += bool(r["redactions"])           # answers with a self-name replaced
        elif r["route"] == "no_reference":
            no_ref.append({"item_id": row["item_id"], "sample": row["sample"]})
        canary += r["reason"] == "canary_in_answer"
        text = rules.strip_reasoning(row["text"])
        final_line += (r["reason"] not in ("truncated", "canary_in_answer") and bool(text)
                       and rules.final_answer(text).rule.split("+")[0] in ("marker", "boxed"))
    write_jsonl(out_dir / "scores.jsonl", scores)
    write_jsonl(out_dir / "judge_queue.jsonl", queue)
    write_jsonl(out_dir / "no_reference.jsonl", no_ref)
    routes = Counter(s["route"] for s in scores)
    counts = rules.reading_counts(by_rules)
    summary = {"split": manifest["split"], "model": manifest["model"], "config_id": manifest["config_id"],
               "n_answers": len(scores), "n_rules": routes["rules"], "n_judge": routes["judge"],
               "n_no_reference": routes["no_reference"], "n_request_failed": failed,
               "by_reason": dict(sorted(Counter(s["reason"] for s in scores).items())),
               "rule_only": rules.rule_only_summary(by_rules), "n_tie_break": counts["tie_break"],
               "n_wrong_form": counts["wrong_form"], "n_redacted": redacted, "n_canary": canary,
               "n_judge_failed": None, "judge": "not run", "n_final_answer_line": final_line,
               "rules_version": rules.RULES_VERSION}
    write_json(out_dir / "summary.json", summary)
    return summary


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb tier-k", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="answer a split's questions with one model")
    r.add_argument("--split", required=True, choices=sorted(SPLIT_FILES))
    r.add_argument("--base-url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--config-id", default="dev-smoke")
    r.add_argument("--reason", default="dev-split check")
    r.add_argument("--n", type=int, default=1)
    r.add_argument("--b1", action="store_true")
    r.add_argument("--sampling", default="{}", help="the model's recommended sampling settings, as JSON")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--allow-test", action="store_true", help="a declared test-split run (sign-off only)")
    r.add_argument("--out", type=Path, default=None)
    s = sub.add_parser("score", help="route and score a run's answers")
    s.add_argument("dir", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        out = a.out or WORK / "tier_k" / f"{a.config_id}-{runlog.utc_now().replace(':', '')}"
        try:
            run(a.split, base_url=a.base_url, model=a.model, config_id=a.config_id, reason=a.reason, n=a.n,
                b1=a.b1, sampling=json.loads(a.sampling), out_dir=out, allow_test=a.allow_test, limit=a.limit)
        except (RunRefused, ValueError) as exc:
            print(f"refused: {exc}")
            return 2
        print(f"answers: {out / 'answers.jsonl'}")
        return 0
    print(json.dumps(score(a.dir), indent=2))
    return 0
