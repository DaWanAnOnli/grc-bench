"""`./gb agent` (tier_a/adhoc.py), the model relay (tier_a/relay.py) and the runner options they use.

The episode tests run GRC-Agent's real executor in the sandbox with scripted models (never a model
under test) on a test fixture graph, never on a benchmark item.
"""

from __future__ import annotations

import http.server
import json
import shutil
import threading
import uuid
from pathlib import Path
from typing import ClassVar

import pytest

from grcbench.common.paths import ITEMS, REPO, WORK
from grcbench.runlog.log import RunRefused
from grcbench.tier_a import adhoc, models, relay, runner
from grcbench.tier_a.caps import Caps
from grcbench.tier_a.task import load_task
from grcbench.tier_a.user import ScriptedUser

TONE = REPO / "tests" / "fixtures" / "tier_a" / "tasks" / "fx-a-modify-tone" / "start" / "tone.grc"


@pytest.fixture
def root():
    path = WORK / "pytest-agent" / uuid.uuid4().hex[:10]
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------- requests, users, limits

def test_a_request_becomes_a_train_task_with_its_follow_ups(root):
    task_dir = adhoc.write_task(root / "t", "agent-x-c01", "Add a throttle", graph=TONE,
                                then=["Now save it."], replies=[("which rate", "32 kHz")])
    task = load_task(task_dir)
    assert task.split == "train" and str(task.active_graph) == "tone.grc" and task.category == "modify"
    assert [str(p) for p in task.start_files] == ["tone.grc"]
    user = ScriptedUser.from_spec(task.scripted_user)
    assert user.respond("Which rate should I use?").message == "32 kHz"          # a matching answer
    assert user.respond("Done.").message == "Now save it."                       # `then`, whatever was said
    assert user.respond("Saved. Anything else?") is None and user.waiting         # the caller answers
    waiting = ScriptedUser.from_spec(adhoc._scripted_user([], [], None, None))
    assert waiting.respond("Should I also add a sink?") is None and waiting.waiting
    assert waiting.respond("Done.") is None


def test_benchmark_tasks_keep_their_scripted_user():
    user = ScriptedUser.from_spec({"rules": [], "max_turns": 1})
    assert user.respond("Shall I continue?").message == "please use your best judgement" and not user.waiting


def test_limits_presets_and_overrides():
    assert adhoc.make_caps("benchmark") == ("benchmark", Caps.for_setting("benchmark"))
    setting, caps = adhoc.make_caps("gui", {"requests": None, "episode_seconds": 900})
    assert setting == "grc_agent_defaults" and caps.requests is None and caps.episode_seconds == 900
    with pytest.raises(adhoc.AgentRunError):
        adhoc.make_caps("gui", {"turns": 3})
    with pytest.raises(adhoc.AgentRunError):
        adhoc.make_caps("fast")


def test_benchmark_items_are_refused_in_collection_mode():
    dev = min(p for p in (ITEMS / "tier_a" / "dev").iterdir() if (p / "task.json").is_file())
    with pytest.raises(RunRefused):
        runner.run_episode(dev, models.scripted([{"text": "x"}]), "grc_agent_defaults", mode="collection")


def test_relay_paths_and_upstreams():
    assert relay._local_path("/v1/chat/completions") == "/chat/completions"
    assert relay._local_path("/v1/models?x=1") == "/models?x=1"
    assert relay._local_path("/v2/chat") is None and relay._local_path("http://evil/v1/x") is None
    with pytest.raises(ValueError):
        relay._local_path("/v1/../admin")
    with pytest.raises(relay.RelayError):
        relay.Upstream.parse("http://api.example.com/v1")                   # https only
    assert relay.Upstream.parse("https://openrouter.ai/api/v1").describe() == "https://openrouter.ai/api/v1"
    spec = models.hosted("https://openrouter.ai/api/v1", "some/model", "SOME_KEY_NAME")
    assert spec.describe()["relay"] == {"upstream": "https://openrouter.ai/api/v1", "api_key_env": "SOME_KEY_NAME"}


# ---------------------------------------------------------------- whole sessions (sandbox + GNU Radio)

EDIT = {"name": "change_graph", "args": {"reason": "Set the tone frequency to 2 kHz.",
                                         "update_params": [{"instance_name": "src", "params": {"freq": "2000"}}]}}


def _user_messages(run_dir: Path) -> list[str]:
    row = json.loads((run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0])
    return [m["content"] for m in row["messages"] if m["role"] == "user"]


def _view(run_dir: Path) -> dict:
    return json.loads((run_dir / "graph_view.json").read_text(encoding="utf-8"))


@pytest.mark.gnuradio
@pytest.mark.slow
def test_a_session_on_a_graph_file_continues_with_its_history(root):
    first = adhoc.run("Change the tone to 2 kHz.", graph=TONE, name="s1", root=root,
                      model=models.scripted([{"tool_calls": [EDIT]},
                                             {"text": "I set the tone to 2 kHz. Should I save the flowgraph?"}]))
    one = Path(first["run_dir"])
    assert first["status"] == "completed" and first["end_reason"] == "agent_asked" and first["waiting_for_reply"]
    assert [c["tool"] for c in first["tool_calls"]] == ["change_graph"] and first["tool_calls"][0]["result"]
    assert _view(one)["valid"] and _view(one)["blocks"]["src"]["params"]["freq"] == "2000"
    # GRC-Agent's change_graph writes the edit through to the file; GRC marks the page modified
    assert first["graph"]["file_in_project"] == "tone.grc" and first["graph"]["page_saved_flag"] is False
    assert first["graph"]["file_matches_graph"] is True and first["files"]["changed"] == ["tone.grc"]
    assert not [f for f in first["files"]["added"] if f.startswith(".grc_agent/")]
    assert "Change the tone to 2 kHz." in (one / "transcript.md").read_text(encoding="utf-8")
    assert (one / "graph.grc").is_file()

    second = adhoc.continue_run(one, "Yes, save it.",
                                model=models.scripted([{"tool_calls": [{"name": "save_graph", "args": {}}]},
                                                       {"text": "Saved."}]))
    two = Path(second["run_dir"])
    assert second["parent"] == str(one) and second["end_reason"] == "agent_finished"
    assert _user_messages(two) == ["Change the tone to 2 kHz.", "Yes, save it."]      # the history is kept
    assert second["graph"]["page_saved_flag"] is True and second["graph"]["file_matches_graph"] is True
    assert _view(two)["blocks"]["src"]["params"]["freq"] == "2000"
    assert second["session_totals"]["requests"] == first["usage"]["requests"] + second["usage"]["requests"]
    calls = json.loads((root / "s1" / adhoc.SESSION_FILE).read_text(encoding="utf-8"))["calls"]
    assert [c["call"] for c in calls] == [1, 2] and calls[1]["parent"] == str(one)


@pytest.mark.gnuradio
@pytest.mark.slow
def test_an_untitled_graph_carries_its_unsaved_edits_into_the_next_call(root):
    edit = {"name": "change_graph", "args": {"reason": "Use 48 kHz.", "update_params": [
        {"instance_name": "samp_rate", "params": {"value": "48000"}}]}}
    first = adhoc.run("Set the sample rate to 48 kHz.", name="s2", root=root,
                      model=models.scripted([{"tool_calls": [edit]}, {"text": "Done."}]))
    one = Path(first["run_dir"])
    assert first["graph"]["file_in_project"] is None and first["end_reason"] == "agent_finished"
    assert _view(one)["blocks"]["samp_rate"]["params"]["value"] == "48000"
    second = adhoc.continue_run(one, "Thanks.", model=models.scripted([{"text": "You're welcome."}]))
    two = Path(second["run_dir"])
    assert _view(two)["blocks"]["samp_rate"]["params"]["value"] == "48000"          # only in memory before
    assert second["graph"]["file_in_project"] is None and second["final_reply"] == "You're welcome."


class _Upstream(http.server.BaseHTTPRequestHandler):
    """A fake OpenAI-compatible provider: one streamed completion per request; records the
    request line and the Authorization header."""

    seen: ClassVar[list[dict]] = []

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        _Upstream.seen.append({"path": self.path, "auth": self.headers.get("authorization"),
                               "stream": json.loads(body).get("stream")})
        wire = models.WireScript([{"text": "Hello from upstream."}], "text", "fake-model", Path("/dev/null"))
        events = "".join(f"data: {json.dumps(c)}\n\n" for c in wire.stream_chunks(wire.completion(0)))
        data = (events + "data: [DONE]\n\n").encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


@pytest.mark.gnuradio
@pytest.mark.slow
def test_a_hosted_model_goes_through_the_relay_and_the_key_stays_out(root, monkeypatch):
    key = "sk-grcbench-test-" + uuid.uuid4().hex
    monkeypatch.setenv("GRCBENCH_TEST_KEY", key)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _Upstream.seen = []
    try:
        spec = models.ModelSpec("openai_compatible", model_name="fake-model",
                                relay={"upstream": f"http://127.0.0.1:{server.server_port}/api/v1",
                                       "api_key_env": "GRCBENCH_TEST_KEY", "allow_http_loopback": True})
        summary = adhoc.run("Say hello.", name="hosted", root=root, model=spec)
    finally:
        server.shutdown()
    assert summary["status"] == "completed" and summary["final_reply"] == "Hello from upstream.", summary
    assert _Upstream.seen and all(s["path"] == "/api/v1/chat/completions" and s["auth"] == f"Bearer {key}"
                                  and s["stream"] for s in _Upstream.seen)
    assert summary["model"]["relay"]["api_key_env"] == "GRCBENCH_TEST_KEY"
    leaked = [p for p in root.rglob("*") if p.is_file() and key.encode() in p.read_bytes()]
    assert not leaked
    assert not list(relay.RELAY_ROOT.glob(f"*/{relay.SOCKET_NAME}"))          # the relay cleaned up
