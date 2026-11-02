"""Run a command inside an Ubuntu 26.04 root filesystem, GRC-Bench's pinned setup, on a machine with another Linux.

    python3 u2604.py ROOTFS [--bind SRC[:DST]]... [--env NAME=VALUE]... -- COMMAND

No Docker: the command runs in new Linux user, mount and PID namespaces, the way bubblewrap builds its sandbox.
COMMAND is a bash command line; it runs as root of the root filesystem (uid 0 inside the user namespace).

  * Run by real root (a Colab runtime), the namespace maps uids and gids 0-65535 to the same ids outside, so apt
    and dpkg can set file owners. Run by another user, it maps that one user to root: enough to run programs.
  * Inside: a new /proc, /dev and /sys from outside, a copy of /etc/resolv.conf, and each --bind SRC at DST (default:
    the same path). The network is shared, so a server on 127.0.0.1 is reachable from both sides.
  * Nothing outside ROOTFS and the bound paths is changed: deleting ROOTFS removes the whole setup.

Python 3.12 or later (os.unshare); util-linux's mount and unshare outside.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys

PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def inside_script(root: str, binds: list[tuple[str, str]], env: list[str], command: str) -> str:
    """The bash script that sets up the mounts and enters ROOTFS (run with the new namespaces' capabilities)."""
    q = shlex.quote
    lines = ["set -e", "mount --make-rprivate /",
             f"mount --rbind {q(root)} {q(root)}",          # / inside is a mount point (bubblewrap needs that)
             f"mkdir -p {q(root)}/dev {q(root)}/sys {q(root)}/proc {q(root)}/tmp",
             f"mount --rbind /dev {q(root)}/dev",
             f"mount --rbind /sys {q(root)}/sys || true",
             f"rm -f {q(root)}/etc/resolv.conf && cp -L /etc/resolv.conf {q(root)}/etc/resolv.conf"]
    for src, dst in binds:
        target = q(root + dst)
        lines.append(f"if [ -d {q(src)} ]; then mkdir -p {target}; else mkdir -p $(dirname {target}) && touch {target}; fi")
        lines.append(f"mount --rbind {q(src)} {target}")
    envs = " ".join(q(e) for e in ["HOME=/root", f"PATH={PATH}", "LANG=C.UTF-8", "TERM=xterm", *env])
    lines.append(f"exec unshare --pid --fork --mount-proc={q(root)}/proc chroot {q(root)} /usr/bin/env -i {envs} "
                 f"bash -lc {q(command)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rootfs")
    ap.add_argument("--bind", action="append", default=[], metavar="SRC[:DST]")
    ap.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        ap.error("put the command after --")
    cut = argv.index("--")
    a = ap.parse_args(argv[:cut])
    words = argv[cut + 1:]
    if not words:
        ap.error("no command")
    root = os.path.realpath(a.rootfs)
    if not os.path.isfile(os.path.join(root, "etc", "os-release")):
        ap.error(f"{root} is not a root filesystem")
    binds = [(os.path.realpath(b.split(":", 1)[0]), b.split(":", 1)[1] if ":" in b else os.path.realpath(b))
             for b in a.bind]
    script = inside_script(root, binds, a.env, " ".join(words))
    uid, gid = os.getuid(), os.getgid()
    ready_r, ready_w = os.pipe()
    go_r, go_w = os.pipe()
    pid = os.fork()
    if pid == 0:                                    # the child: new user and mount namespaces, then the script
        os.close(ready_r)
        os.close(go_w)
        os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNS)
        os.write(ready_w, b"1")
        if os.read(go_r, 1) != b"1":                # the parent could not write the id maps
            os._exit(111)
        os.execvp("bash", ["bash", "-c", script])
    os.close(ready_w)
    os.close(go_r)
    os.read(ready_r, 1)
    try:
        if uid == 0:
            maps = ("0 0 65536", "0 0 65536")
        else:
            with open(f"/proc/{pid}/setgroups", "w") as fh:
                fh.write("deny")
            maps = (f"0 {uid} 1", f"0 {gid} 1")
        for name, value in zip(("uid_map", "gid_map"), maps):
            with open(f"/proc/{pid}/{name}", "w") as fh:
                fh.write(value)
        os.write(go_w, b"1")
    except OSError as exc:
        print(f"u2604: cannot map ids for the user namespace: {exc}", file=sys.stderr)
        os.write(go_w, b"0")
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


if __name__ == "__main__":
    sys.exit(main())
