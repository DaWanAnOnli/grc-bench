"""The scripted user of multi-turn Tier-A tasks (eval plan §4.4; proposal §4.4).

Eval plan §4.4: "Multi-turn tasks use a scripted user: set replies for the questions the agent is
expected to ask, and 'please use your best judgement' otherwise. A scripted user makes every run
repeatable, which a simulated user played by another language model would not."

After each agent turn has finished (all approvals resolved, a final answer reached) the runner hands
the agent's reply text to `ScriptedUser.respond`:
  1. If `max_turns` scripted replies have already been sent, the user stays silent: the episode ends.
  2. The rules are tried in order. A rule fires when its regular expression matches the reply, it
     has uses left (`max_uses`, default 1), and either the reply is asking the user something or
     the rule sets `"require_question": false`. The first rule that fires gives the reply.
  3. Otherwise, if the reply is asking the user something, the reply is DEFAULT_REPLY.
  4. Otherwise the agent has finished without asking: the episode ends.
Everything is deterministic: the same agent reply always gets the same user reply.

"Asking" (`is_asking`) is decided on the reply text with fenced code removed: a line or sentence
that ends with a question mark, or a request for input such as "let me know", "would you like",
"should I", "please confirm". A task may replace the rule with its own `asking_pattern`.

The agent's reply text (`reply_text`) is what Qoherent's collection harness hands its simulator
(runner_contract.last_assistant_text: the joined text parts of the last model response), and, when
that is empty because the agent answered through GRC-Agent's structured output (GrcAgentResponse
via the final_result tool), the response's explanation followed by its actions, which is what GRC-
Agent's sidebar shows.

Task record field `scripted_user` (schema grcbench.common.schemas.TIER_A):
    {"rules": [{"pattern": "<regex>", "reply": "<text>", "max_uses": 1, "require_question": true,
                "flags": "i"}],
     "default_reply": "please use your best judgement",     # optional; this is the default
     "max_turns": 3,                                         # scripted replies after the instruction
     "asking_pattern": "<regex>"}                            # optional
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

DEFAULT_REPLY = "please use your best judgement"
DEFAULT_MAX_TURNS = 3

_FENCED = re.compile(r"```.*?(```|\Z)", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_QUESTION_END = re.compile(r"\?[\"')\]*_]*\s*$", re.MULTILINE)
_SENTENCE_QUESTION = re.compile(r"\?[\"')\]*_]*(\s|$)")
_REQUEST = re.compile(
    r"\b(let me know|would you like|would you prefer|do you want|do you prefer|should i\b|shall i\b|"
    r"can you (confirm|clarify|tell me|specify|provide)|could you (confirm|clarify|tell me|specify|provide)|"
    r"please (confirm|clarify|specify|tell me|provide|choose|select|advise)|which (one|option) "
    r"(do|would) you|i need (you|your) (to )?(confirm|input|answer|decision))", re.IGNORECASE)
_FLAG_BITS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}


def _prose(text: str) -> str:
    return _INLINE_CODE.sub(" ", _FENCED.sub(" ", text or ""))


def is_asking(text: str, pattern: str | None = None) -> bool:
    """Whether the agent's reply asks the user something (see the module docstring)."""
    prose = _prose(text)
    if pattern is not None:
        return re.search(pattern, prose, re.IGNORECASE | re.MULTILINE) is not None
    return bool(_QUESTION_END.search(prose) or _SENTENCE_QUESTION.search(prose) or _REQUEST.search(prose))


def _flags(spec: str) -> int:
    bits = 0
    for ch in spec:
        if ch not in _FLAG_BITS:
            raise ValueError(f"unknown regex flag {ch!r} (allowed: i, m, s)")
        bits |= _FLAG_BITS[ch]
    return bits


@dataclass(frozen=True)
class Rule:
    pattern: str
    reply: str
    max_uses: int = 1
    require_question: bool = True
    flags: str = "i"

    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern, _flags(self.flags))


@dataclass(frozen=True)
class UserTurn:
    """One scripted reply: `source` is "rule:<index>" or "default"."""

    message: str
    source: str


@dataclass
class ScriptedUser:
    rules: list[Rule] = field(default_factory=list)
    default_reply: str = DEFAULT_REPLY
    max_turns: int = DEFAULT_MAX_TURNS
    asking_pattern: str | None = None
    turns_sent: int = 0
    uses: dict[int, int] = field(default_factory=dict)
    exhausted: bool = False          # the agent asked again after max_turns replies

    @classmethod
    def from_spec(cls, spec: dict[str, Any] | None) -> "ScriptedUser":
        spec = dict(spec or {})
        rules = [Rule(pattern=r["pattern"], reply=r["reply"], max_uses=int(r.get("max_uses", 1)),
                      require_question=bool(r.get("require_question", True)), flags=str(r.get("flags", "i")))
                 for r in spec.get("rules", [])]
        user = cls(rules=rules, default_reply=str(spec.get("default_reply", DEFAULT_REPLY)),
                   max_turns=int(spec.get("max_turns", DEFAULT_MAX_TURNS)),
                   asking_pattern=spec.get("asking_pattern"))
        user.check()
        return user

    def check(self) -> None:
        """Raise ValueError on an unusable script (bad regex, empty reply, negative limits)."""
        if self.max_turns < 0:
            raise ValueError("scripted_user.max_turns must be >= 0")
        if not self.default_reply.strip():
            raise ValueError("scripted_user.default_reply must not be empty")
        for i, rule in enumerate(self.rules):
            try:
                rule.compiled()
            except re.error as exc:
                raise ValueError(f"scripted_user.rules[{i}].pattern: {exc}") from exc
            if not rule.reply.strip() or rule.max_uses < 1:
                raise ValueError(f"scripted_user.rules[{i}]: empty reply or max_uses < 1")
        if self.asking_pattern is not None:
            re.compile(self.asking_pattern)

    def replies(self) -> list[str]:
        """Every text this user can send (checked for the canary by the task loader)."""
        return [r.reply for r in self.rules] + [self.default_reply]

    def respond(self, agent_reply: str) -> UserTurn | None:
        """The user's next message after an agent turn, or None when the episode should end."""
        asking = is_asking(agent_reply, self.asking_pattern)
        if self.turns_sent >= self.max_turns:
            self.exhausted = asking
            return None
        for i, rule in enumerate(self.rules):
            if self.uses.get(i, 0) >= rule.max_uses:
                continue
            if rule.require_question and not asking:
                continue
            if rule.compiled().search(agent_reply or ""):
                self.uses[i] = self.uses.get(i, 0) + 1
                self.turns_sent += 1
                return UserTurn(rule.reply, f"rule:{i}")
        if asking:
            self.turns_sent += 1
            return UserTurn(self.default_reply, "default")
        return None


# --------------------------------------------------------------------------------------------
# What the user "sees" of an agent turn
# --------------------------------------------------------------------------------------------

def last_assistant_text(messages: list[Any]) -> str:
    """Qoherent's runner_contract.last_assistant_text, verbatim: the last ModelResponse's joined
    text parts, stripped."""
    from pydantic_ai.messages import ModelResponse, TextPart
    for msg in reversed(messages):
        if isinstance(msg, ModelResponse):
            return "".join(part.content for part in msg.parts if isinstance(part, TextPart)).strip()
    return ""


def structured_text(output: Any) -> str:
    """A GrcAgentResponse (explanation + actions_taken) as the text the user reads."""
    explanation = getattr(output, "explanation", None)
    actions = getattr(output, "actions_taken", None)
    if explanation is None and actions is None:
        return ""
    lines = [str(explanation or "").strip()]
    lines += [f"- {a}" for a in (actions or [])]
    return "\n".join(line for line in lines if line).strip()


def reply_text(messages: list[Any], output: Any) -> str:
    """The agent's reply to the user for one finished turn (module docstring)."""
    text = last_assistant_text(messages)
    if text:
        return text
    if isinstance(output, str):
        return output.strip()
    return structured_text(output)
