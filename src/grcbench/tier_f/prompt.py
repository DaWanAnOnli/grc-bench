"""The Tier-F prompt (pre-registered; its SHA-256 goes into the pre-registration).

Every model sees the same system prompt and the same user-message layout. The answer must contain
one complete flowgraph as .grc YAML in a fenced code block; grcfile.extract_grc is the extraction
rule and the answer cap (8K tokens including reasoning, eval plan §7.1) applies to the whole reply.
The GRC-Bench canary string is never part of a prompt.
"""

from __future__ import annotations

import hashlib

PROMPT_VERSION = "tier-f-prompt-v1"

SYSTEM = (
    "You are an expert GNU Radio developer. You write flowgraphs for GNU Radio Companion 3.10.\n"
    "Reply with exactly one complete flowgraph in the .grc YAML format inside a single ```yaml code block.\n"
    "Format rules:\n"
    "- Top-level keys: options, blocks, connections, metadata (with file_format: 1).\n"
    "- options.parameters must set id and generate_options; use generate_options: no_gui.\n"
    "- Each block has name, id (a GNU Radio 3.10 block id) and parameters. You may leave out GUI\n"
    "  coordinates (states) and parameters that keep their default values.\n"
    "- Each connection is [source_block_name, source_port, sink_block_name, sink_port].\n"
    "Use only blocks that exist in GNU Radio 3.10. Keep the explanation short or omit it."
)

USER_TEMPLATE_BUILD = "{instruction}"
USER_TEMPLATE_EDIT = "{instruction}\n\nThe current flowgraph is:\n```yaml\n{start_grc}```"


def messages(instruction: str, start_grc: str | None = None) -> list[dict[str, str]]:
    if start_grc:
        user = USER_TEMPLATE_EDIT.format(instruction=instruction.strip(),
                                         start_grc=start_grc if start_grc.endswith("\n") else start_grc + "\n")
    else:
        user = USER_TEMPLATE_BUILD.format(instruction=instruction.strip())
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def prompt_sha256() -> str:
    blob = "\n\0\n".join([PROMPT_VERSION, SYSTEM, USER_TEMPLATE_BUILD, USER_TEMPLATE_EDIT])
    return hashlib.sha256(blob.encode()).hexdigest()
