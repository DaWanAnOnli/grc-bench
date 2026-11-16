"""Private device paths in published files: found by `private_paths`, replaced by `redact`.

A published file (a report, a decided review record, a filled sheet) names no person's user folder:
- this machine's project folder becomes `<project>` (WSL, Windows, Git Bash and JSON-escaped spellings);
- any Windows user folder (C:\\Users\\<name>, /mnt/c/Users/<name>) becomes `<home>`, and the project folder
  inside another machine's user folder becomes `<project>` too;
- any Linux home folder (/home/<name>) becomes `~`;
- a Claude folder name that spells a user folder (C--Users-<name>-...) keeps `<user>` in its place.

Example and sandbox names stay: /home/user (the Tier-A sandbox), someone, you, u.

    private_paths(text) -> list[str]        the user folders named in text
    redact(obj, project_root=None, home=None) -> obj with every string redacted (dicts and lists walked)
"""

from __future__ import annotations

import re
from pathlib import Path, PurePath
from typing import Any

_KEEP = r"(?!(?:someone|user|you|u)\b)"            # example and sandbox user names
_SEP = r"[\\/]+"                                   # /, \ or a JSON-escaped \\
_DRIVE = r"(?:/mnt/[a-z]|\b[A-Za-z]:|/[a-z])"
_END = r"(?=[\\/\s\"'`]|$)"

_NAME = r"[A-Za-z0-9_]+(?:[.-][A-Za-z0-9_]+)*"     # a user name; a full stop after it ends the sentence
WIN_USER = re.compile(r"(?:\b[A-Za-z]:" + _SEP + r"|/mnt/[a-z]/)Users" + _SEP + _KEEP + _NAME)
LINUX_HOME = re.compile(r"/home/" + _KEEP + r"[a-z_][a-z0-9_-]*")
CLAUDE_FOLDER = re.compile(r"\b([A-Za-z])--Users-" + _KEEP + r"[A-Za-z0-9_]+")


def private_paths(text: str) -> list[str]:
    """Every user folder the text names (empty when the text is clean)."""
    return [m.group(0) for rx in (WIN_USER, LINUX_HOME, CLAUDE_FOLDER) for m in rx.finditer(text)]


def _patterns(root: PurePath, home: PurePath) -> list[tuple[re.Pattern, str]]:
    """This machine's project folder, the rest of its user's Windows folder, and this machine's home folder;
    then any machine's user folders, and the project folder inside them."""
    parts, out = list(root.parts), []
    if "Users" in parts and parts.index("Users") + 1 < len(parts):
        i = parts.index("Users")
        rest = "".join(_SEP + re.escape(x) for x in parts[i + 2:])
        out.append((re.compile(_DRIVE + r"?[\\/]+Users" + _SEP + re.escape(parts[i + 1]) + rest + _END), "<project>"))
        out.append((re.compile(_DRIVE + r"[\\/]+Users[\\/]+" + re.escape(parts[i + 1]) + _END), "<home>"))
    else:
        rest = ""
        out.append((re.compile(re.escape(str(root)) + _END), "<project>"))
    if str(home) not in ("", "/", "."):
        out.append((re.compile(re.escape(str(home)) + _END), "~"))
    out.append((WIN_USER, "<home>"))
    if rest:
        out.append((re.compile(r"<home>" + rest + _END), "<project>"))
    out += [(LINUX_HOME, "~"), (CLAUDE_FOLDER, r"\1--Users-<user>")]
    return out


def redact(obj: Any, project_root: PurePath | None = None, home: PurePath | None = None) -> Any:
    """obj with every private device path in its strings replaced (see the module docstring)."""
    from grcbench.common.paths import REPO
    pats = _patterns(project_root or REPO.parent, home or Path.home())

    def one(x: Any) -> Any:
        if isinstance(x, str):
            for p, sub in pats:
                x = p.sub(sub, x)
            return x
        if isinstance(x, dict):
            return {k: one(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(one(v) for v in x)
        return x
    return one(obj)
