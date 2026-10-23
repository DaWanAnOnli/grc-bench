#!/usr/bin/env bash
# grc-agent-headless setup check (TUTORIAL.md, step 6). Safe to run again at any time.
# Checks every prerequisite and says how to fix what is missing; builds, once, the two things a fresh
# machine lacks (GRC-Agent's knowledge databases and the tokenizer file); ends with ./agent selftest.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRC_AGENT_REPO="${GRC_AGENT_REPO:-$HOME/grc-agent}"
CACHE="${GRC_BENCH_CACHE:-$HOME/grc-bench-cache}"
PIN_AGENT=6c98128
PIN_GR=3.10.12
PY="$GRC_AGENT_REPO/.venv/bin/python"
failed=0

ok()   { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s\n      fix: %s\n' "$1" "$2"; failed=1; }

echo "== 1. location"
case "$HERE" in
  /tmp/*|/mnt/*) fail "the package is in $HERE" "unzip it in your Linux home folder: cd ~ && unzip <zip>" ;;
  *) ok "package folder $HERE" ;;
esac

echo "== 2. system tools"
for cmd in bwrap prlimit git python3; do
  if command -v "$cmd" >/dev/null 2>&1; then ok "$cmd"; else
    fail "$cmd not found" "sudo apt install -y bubblewrap util-linux git python3"; fi
done

echo "== 3. GNU Radio $PIN_GR"
gr=$(/usr/bin/python3 -c "from gnuradio import gr; print(gr.version())" 2>/dev/null || true)
case "$gr" in
  "$PIN_GR"*) ok "GNU Radio $gr" ;;
  "") fail "GNU Radio's Python bindings not found" "sudo apt install -y gnuradio (Ubuntu 26.04 ships $PIN_GR)" ;;
  *) fail "GNU Radio $gr (this package is pinned to $PIN_GR)" "use Ubuntu 26.04, whose gnuradio package is $PIN_GR" ;;
esac

echo "== 4. GRC-Agent at $PIN_AGENT in $GRC_AGENT_REPO"
if [ ! -d "$GRC_AGENT_REPO/.git" ]; then
  fail "no GRC-Agent checkout at $GRC_AGENT_REPO" \
       "git clone https://github.com/qoherent/grc-agent.git $GRC_AGENT_REPO && git -C $GRC_AGENT_REPO checkout $PIN_AGENT"
else
  head=$(git -C "$GRC_AGENT_REPO" rev-parse --short=7 HEAD 2>/dev/null)
  if [ "$head" = "$PIN_AGENT" ]; then ok "GRC-Agent commit $head"; else
    fail "GRC-Agent is at $head" "git -C $GRC_AGENT_REPO checkout $PIN_AGENT"; fi
  if [ -x "$PY" ] && "$PY" -c "import grc_agent, gnuradio, pydantic_ai" 2>/dev/null; then
    ok "GRC-Agent's virtual environment ($PY) imports grc_agent and gnuradio"
  else
    fail "GRC-Agent's virtual environment is missing or incomplete" \
         "cd $GRC_AGENT_REPO && uv venv --system-site-packages --python /usr/bin/python3 && uv sync --extra dev --locked --python .venv/bin/python"
  fi
fi

echo "== 5. the sandbox (bubblewrap)"
if bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --unshare-pid --unshare-net --die-with-parent true 2>/tmp/grc-agent-headless-bwrap.txt; then
  ok "bubblewrap can create a sandbox"
else
  fail "bubblewrap cannot create a sandbox: $(head -1 /tmp/grc-agent-headless-bwrap.txt)" \
       "on native Ubuntu 24.04+: sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0 (TUTORIAL.md, Troubleshooting)"
fi
rm -f /tmp/grc-agent-headless-bwrap.txt

if [ "$failed" -ne 0 ]; then
  echo; echo "setup: fix the FAIL lines above, then run ./setup.sh again"; exit 1
fi

echo "== 6. GRC-Agent's knowledge databases (built once)"
VEC="$GRC_AGENT_REPO/src/grc_agent/vectors"
if [ -f "$VEC/catalog_lexical.db" ] && [ -f "$VEC/docs_lexical.db" ]; then
  ok "catalog_lexical.db and docs_lexical.db present"
else
  echo "      building them (lexical search, a few seconds) ..."
  if (cd "$GRC_AGENT_REPO" && GRC_EMBED_BACKEND=lexical QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 "$PY" -c "
from grc_agent.adapter import rag
for domain in ('catalog', 'docs'):
    path, model = rag.get_db_and_model(domain)
    rag._ensure_db_built(domain, path, model)
" >/dev/null 2>&1) && [ -f "$VEC/catalog_lexical.db" ] && [ -f "$VEC/docs_lexical.db" ]; then
    ok "built catalog_lexical.db and docs_lexical.db"
  else
    fail "could not build the knowledge databases" "start GRC-Agent once (cd $GRC_AGENT_REPO && uv run grc-agent) and ask it a question, then run ./setup.sh again"
  fi
fi

echo "== 7. tokenizer file (downloaded once)"
mkdir -p "$CACHE/tiktoken"
if TIKTOKEN_CACHE_DIR="$CACHE/tiktoken" "$PY" -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" 2>/dev/null; then
  ok "cl100k_base in $CACHE/tiktoken"
else
  fail "could not get the cl100k_base tokenizer file" "check the internet connection, then run ./setup.sh again"
fi

if [ "$failed" -ne 0 ]; then
  echo; echo "setup: fix the FAIL lines above, then run ./setup.sh again"; exit 1
fi

echo "== 8. self-test (scripted models; no model server needed)"
if "$HERE/agent" selftest; then
  echo; echo "setup complete: try  ./agent run \"Build a flowgraph with a 1 kHz Signal Source into a QT GUI Time Sink\""
else
  echo; echo "setup: the self-test failed (lines above); TUTORIAL.md, Troubleshooting"; exit 1
fi
