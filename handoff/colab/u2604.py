"""Run a command inside an Ubuntu 26.04 root filesystem, GRC-Bench's pinned setup, on a machine with another Linux.

    python3 u2604.py ROOTFS [--bind SRC[:DST]]... [--env NAME=VALUE]... -- COMMAND

No Docker: the command runs in new Linux user and mount namespaces, entered with pivot_root, the way bubblewrap
builds its sandbox. COMMAND is a bash command line; it runs as root of the root filesystem (uid 0 inside the user
namespace).

  * Run by real root (a Colab runtime), the namespace maps uids and gids 0-65535 to the same ids outside, so apt
    and dpkg can set file owners. Run by another user, it maps that one user to root: enough to run programs.
  * Inside: /proc, /dev and /sys from outside (a Colab runtime refuses to mount a fresh /proc, so there is no PID
    namespace either: the processes started inside are ordinary processes of the machine), a copy of
    /etc/resolv.conf, and each --bind SRC at DST (default: the same path). The network is shared, so a server on
    127.0.0.1 is reachable from both sides.
  * pivot_root, not chroot: Linux refuses new user namespaces to a chrooted process, and GRC-Bench's sandbox makes
    one on Colab (GRC_BENCH_BWRAP_USERNS=1).
  * Nothing outside ROOTFS and the bound paths is changed: deleting ROOTFS removes the whole setup.

Python 3.12 or later (os.unshare); util-linux's mount and pivot_root outside, umount inside.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import shlex
import subprocess
import sys

PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
OLD = ".u2604-old"                                  # where pivot_root puts the machine's root, unmounted at once
SYS_PIVOT_ROOT = {"x86_64": 155, "aarch64": 41}     # no pivot_root program in Ubuntu 26.04's util-linux
MNT_DETACH = 2


def inside_script(root: str, binds: list[tuple[str, str]]) -> str:
    """The bash script that sets up the mounts under ROOTFS (run with the new namespaces' capabilities)."""
    q = shlex.quote
    lines = ["set -e", f"export PATH={PATH}", "mount --make-rprivate /",
             f"mount --rbind {q(root)} {q(root)}",          # pivot_root needs the new root to be a mount point
             f"mkdir -p {q(root)}/dev {q(root)}/sys {q(root)}/proc {q(root)}/tmp {q(root)}/{OLD}",
             f"mount --rbind /dev {q(root)}/dev",
             f"mount --rbind /sys {q(root)}/sys || true",
             f"mount --rbind /proc {q(root)}/proc",
             f"rm -f {q(root)}/etc/resolv.conf && cp -L /etc/resolv.conf {q(root)}/etc/resolv.conf"]
    for src, dst in binds:
        target = q(root + dst)
        lines.append(f"if [ -d {q(src)} ]; then mkdir -p {target}; else mkdir -p $(dirname {target}) && touch {target}; fi")
        lines.append(f"mount --rbind {q(src)} {target}")
    return "\n".join(lines)


def enter(root: str, env: list[str], command: str) -> None:
    """pivot_root into ROOTFS, drop the machine's root, and run COMMAND there with a clean environment."""
    libc = ctypes.CDLL(None, use_errno=True)
    os.chdir(root)
    if libc.syscall(SYS_PIVOT_ROOT[platform.machine()], b".", OLD.encode()) != 0:
        raise OSError(ctypes.get_errno(), f"pivot_root into {root}")
    os.chdir("/")
    if libc.umount2(f"/{OLD}".encode(), MNT_DETACH) != 0:
        raise OSError(ctypes.get_errno(), f"umount /{OLD}")
    try:
        os.rmdir(f"/{OLD}")
    except OSError:
        pass
    envs = ["HOME=/root", f"PATH={PATH}", "LANG=C.UTF-8", "TERM=xterm", *env]
    os.execv("/usr/bin/env", ["env", "-i", *envs, "bash", "-lc", command])


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
    script = inside_script(root, binds)
    uid, gid = os.getuid(), os.getgid()
    ready_r, ready_w = os.pipe()
    go_r, go_w = os.pipe()
    pid = os.fork()
    if pid == 0:                                    # the child: new user and mount namespaces, the mounts, then in
        os.close(ready_r)
        os.close(go_w)
        os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNS)
        os.write(ready_w, b"1")
        if os.read(go_r, 1) != b"1":                # the parent could not write the id maps
            os._exit(111)
        mounts = subprocess.run(["bash", "-c", script])
        if mounts.returncode:
            os._exit(mounts.returncode)
        try:
            enter(root, a.env, " ".join(words))
        except (OSError, KeyError) as exc:
            print(f"u2604: cannot enter {root}: {exc!r}", file=sys.stderr)
            os._exit(112)
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
