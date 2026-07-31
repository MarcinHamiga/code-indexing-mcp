"""Symlink primitives shared by the skill installer and the PATH launcher.

Both callers face the same problem: put a link where something the user owns
may already be, without losing that something and without leaving a half-made
link behind when the platform refuses to create one at all.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def backup_path(target: Path) -> Path:
    """Pick a backup name that does not overwrite a backup from an earlier install."""

    candidate = target.with_name(f"{target.name}.bak")
    counter = 2
    while candidate.is_symlink() or candidate.exists():
        candidate = target.with_name(f"{target.name}.bak.{counter}")
        counter += 1
    return candidate


def link_destination(link: Path) -> Path:
    """Where a symlink points, in a form that compares reliably.

    Raw os.readlink output is not comparable: Windows hands back an extended-length
    "\\\\?\\C:\\..." path that never equals the plain path the link was created from,
    which would make every re-install look like a first install.
    """

    return link.resolve()


def replace_link(
    source: Path,
    target: Path,
    *,
    is_directory: bool,
    stale: bool = False,
) -> bool:
    """Point ``target`` at ``source``, backing up any clashing entry.

    Returns True when a new link was created, False when it already existed.
    ``stale`` marks an existing link this installer left behind, which is
    replaced outright rather than backed up.
    """

    if target.is_symlink() and link_destination(target) == source.resolve():
        return False
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Build the replacement link before disturbing what is already there, so a
    # platform that cannot create symlinks at all fails without having moved
    # something the user owns out from under them.
    staged = target.with_name(f"{target.name}.incoming")
    if staged.is_symlink() or staged.exists():
        remove_path(staged)
    staged.symlink_to(source, target_is_directory=is_directory)
    try:
        if stale:
            target.unlink()
        elif target.is_symlink() or target.exists():
            target.rename(backup_path(target))
        staged.rename(target)
    except OSError:
        staged.unlink(missing_ok=True)
        raise
    return True
