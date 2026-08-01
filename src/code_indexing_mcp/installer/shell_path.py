"""The ``code-indexing-mcp`` launcher and the PATH entry that makes it reachable.

The server executable lives inside the installation's virtual environment, and
that absolute path is what every harness config names. A shell has no way to
find it, so the CLI -- ``configure``, ``index``, ``status``, ``daemon`` -- is
unreachable without help. This module puts a launcher in a bin directory and,
when that directory is not already on PATH, offers to add it to the user's
shell profile.

The launcher is a symlink: the venv console script carries an absolute shebang,
so it runs correctly through a link with no wrapper of any kind. Windows is the
exception -- creating a symlink there needs a privilege the installer cannot
assume -- so it gets a ``.cmd`` shim instead.

No Textual import belongs here; the TUI is one caller of this, not its owner.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import string
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .accelerator import server_executable
from .config_files import InstallerError, write_changed_configuration
from .links import backup_path, is_under, link_destination, replace_link

LAUNCHER_NAME = "code-indexing-mcp"

# The block written into a shell profile. Both markers are matched literally on
# rewrite and removal, which is what keeps a second install from appending a
# second copy and lets the uninstaller take back exactly what it added.
BLOCK_START = "# >>> code-indexing-mcp >>>"
BLOCK_END = "# <<< code-indexing-mcp <<<"


@dataclass(frozen=True)
class LauncherResult:
    """What happened to the launcher, and where it ended up."""

    path: Path
    # "created" | "current" | "replaced" | "skipped" | "failed"
    status: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {"created", "current", "replaced"}


@dataclass(frozen=True)
class PathState:
    """Everything the wizard needs to describe the PATH situation up front."""

    bin_directory: Path
    launcher: Path
    on_path: bool
    # A `code-indexing-mcp` that an earlier PATH entry would find first.
    shadowed_by: Path | None
    # Profiles that would be edited if the user asks for it.
    profiles: tuple[Path, ...]
    # True when every profile already carries the block or the directory.
    profiles_current: bool


def default_bin_directory(
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Where the launcher goes: the XDG user binary directory by convention."""

    home = home or Path.home()
    environment = os.environ if environment is None else environment
    for variable in ("CODE_INDEXING_MCP_BIN_DIR", "XDG_BIN_HOME"):
        configured = environment.get(variable)
        if configured:
            return Path(configured).expanduser()
    return home / ".local" / "bin"


def launcher_path(bin_directory: Path, *, platform_name: str | None = None) -> Path:
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return bin_directory / f"{LAUNCHER_NAME}.cmd"
    return bin_directory / LAUNCHER_NAME


def _path_entries(environment: Mapping[str, str]) -> list[Path]:
    raw = environment.get("PATH", "")
    entries: list[Path] = []
    for part in raw.split(os.pathsep):
        if not part:
            continue
        try:
            entries.append(Path(part).expanduser())
        except (OSError, ValueError):
            # A malformed PATH entry is the user's problem, not a reason to
            # abandon the check; the remaining entries still answer the question.
            continue
    return entries


def _same_directory(left: Path, right: Path) -> bool:
    """Compare two directories without requiring either to exist."""

    try:
        return left.resolve() == right.resolve()
    except (OSError, ValueError):
        # An entry the filesystem refuses to stat at all -- a NUL byte, a path
        # past the length limit -- is still comparable as text.
        return left == right


def is_on_path(
    bin_directory: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> bool:
    environment = os.environ if environment is None else environment
    return any(_same_directory(entry, bin_directory) for entry in _path_entries(environment))


def shadowing_executable(
    bin_directory: Path,
    *,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> Path | None:
    """A ``code-indexing-mcp`` an earlier PATH entry would find before ours.

    Reported rather than removed. Something else owns that file, and silently
    winning the name would be a worse surprise than saying which one runs.
    """

    environment = os.environ if environment is None else environment
    ours = launcher_path(bin_directory, platform_name=platform_name)
    for entry in _path_entries(environment):
        if _same_directory(entry, bin_directory):
            return None
        candidate = shutil.which(LAUNCHER_NAME, path=str(entry))
        if candidate is not None:
            found = Path(candidate)
            if not _same_directory(found.parent, ours.parent):
                return found
    return None


def install_launcher(
    install_directory: Path,
    bin_directory: Path,
    *,
    platform_name: str | None = None,
) -> LauncherResult:
    """Put a ``code-indexing-mcp`` launcher in ``bin_directory``.

    Never raises: a launcher is a convenience, and a failure to create one must
    not undo harness configuration that already succeeded.
    """

    platform_name = platform_name or sys.platform
    target = launcher_path(bin_directory, platform_name=platform_name)
    executable = server_executable(install_directory, platform_name=platform_name)
    if not executable.is_file():
        return LauncherResult(
            target,
            "failed",
            f"no server executable at {executable}",
        )
    try:
        if platform_name.startswith("win"):
            return _install_shim(executable, target)
        return _install_symlink(executable, target)
    except (OSError, UnicodeDecodeError) as exc:
        return LauncherResult(target, "failed", str(exc))


def _is_our_launcher(target: Path, executable: Path) -> bool:
    """True when ``target`` is a link this installer left pointing at a venv of ours."""

    if not target.is_symlink():
        return False
    destination = link_destination(target)
    return destination.name == executable.name and ".venv" in destination.parts


def _install_symlink(executable: Path, target: Path) -> LauncherResult:
    stale = _is_our_launcher(target, executable)
    # Resolve the backup name before the move, not after: once replace_link has
    # renamed the old entry, backup_path returns the *next* free name instead of
    # the one the user's file is now under.
    backup = backup_path(target) if (target.is_symlink() or target.exists()) and not stale else None
    created = replace_link(executable, target, is_directory=False, stale=stale)
    if not created:
        return LauncherResult(target, "current", f"already points at {executable}")
    if backup is not None:
        return LauncherResult(target, "replaced", f"the previous entry was kept as {backup.name}")
    return LauncherResult(target, "created", f"points at {executable}")


_SHIM_TEMPLATE = '@echo off\r\n"{executable}" %*\r\n'


def _install_shim(executable: Path, target: Path) -> LauncherResult:
    """Windows gets a batch shim: symlinks there need a privilege we cannot assume."""

    content = _SHIM_TEMPLATE.format(executable=executable)
    # Read as bytes: the shim's CRLF line endings are written verbatim, and
    # universal-newline decoding would turn every re-install into a rewrite.
    existing = target.read_bytes().decode("utf-8") if target.is_file() else None
    if existing == content:
        return LauncherResult(target, "current", f"already runs {executable}")
    target.parent.mkdir(parents=True, exist_ok=True)
    changed = write_changed_configuration(target, existing, content)
    if not changed:  # pragma: no cover - equality was already handled above
        return LauncherResult(target, "current", f"already runs {executable}")
    status = "replaced" if existing is not None else "created"
    return LauncherResult(target, status, f"runs {executable}")


def _removable_launcher(
    target: Path,
    install_directory: Path | None,
    platform_name: str,
) -> bool:
    """True when ``target`` is evidently a launcher this installation created.

    ``.venv`` in the destination is necessary but nowhere near sufficient: a
    user's own symlink to some unrelated project's virtual environment matches
    it too. When the caller knows which checkout is being uninstalled -- and the
    uninstaller always does -- the destination has to point inside that one.
    """

    if platform_name.startswith("win"):
        if not target.is_file():
            return False
        try:
            content = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return False
        if ".venv" not in content or LAUNCHER_NAME not in content:
            return False
        if install_directory is None:
            return True
        # The shim names the executable by absolute path on its one command line.
        return any(
            is_under(Path(candidate), install_directory)
            for candidate in _shim_command_candidates(content)
        )
    if not target.is_symlink():
        return False
    destination = link_destination(target)
    if ".venv" not in destination.parts:
        return False
    if install_directory is None:
        return True
    return is_under(destination, install_directory)


def _shim_command_candidates(content: str) -> list[str]:
    """The quoted executable paths a Windows shim names, if it names any."""

    candidates: list[str] = []
    for line in content.splitlines():
        opening = line.find('"')
        closing = line.find('"', opening + 1) if opening != -1 else -1
        if closing != -1:
            candidates.append(line[opening + 1 : closing])
    return candidates


def remove_launcher(
    bin_directory: Path,
    install_directory: Path | None = None,
    *,
    platform_name: str | None = None,
) -> Path | None:
    """Remove a launcher this installer created. Returns the path if one went away.

    A file at that name that is *not* one of ours is left strictly alone. Pass
    ``install_directory`` -- the checkout being removed -- to require that the
    launcher actually points into it; without it the check falls back to the
    looser "points into some virtual environment" test.
    """

    platform_name = platform_name or sys.platform
    target = launcher_path(bin_directory, platform_name=platform_name)
    if not _removable_launcher(target, install_directory, platform_name):
        return None
    target.unlink()
    return target


def _fish_profile(home: Path, environment: Mapping[str, str]) -> Path:
    configured = environment.get("XDG_CONFIG_HOME")
    base = Path(configured).expanduser() if configured else home / ".config"
    return base / "fish" / "config.fish"


def shell_profiles(
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> tuple[Path, ...]:
    """The profile files to edit, in the order they would be written.

    ``$SHELL`` names the login shell, which decides the primary file; any other
    profile that already exists is included too, so a user who moves between
    bash and zsh does not lose the entry by switching.
    """

    home = home or Path.home()
    environment = os.environ if environment is None else environment
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        # A profile edit on Windows means the registry or `setx`; the installer
        # reports the manual step instead of reaching for either.
        return ()

    zdotdir = environment.get("ZDOTDIR")
    zshrc = (Path(zdotdir).expanduser() if zdotdir else home) / ".zshrc"
    bashrc = home / ".bashrc"
    # macOS terminals start bash as a login shell, which reads .bash_profile and
    # never .bashrc unless that file sources it.
    bash_profile = home / ".bash_profile"
    fish = _fish_profile(home, environment)
    profile = home / ".profile"

    shell = Path(environment.get("SHELL", "")).name
    primary: list[Path] = []
    if shell == "zsh":
        primary = [zshrc]
    elif shell == "bash":
        primary = [bashrc, bash_profile] if platform_name == "darwin" else [bashrc]
    elif shell == "fish":
        primary = [fish]
    elif shell:
        primary = [profile]

    selected: list[Path] = list(primary)
    for candidate in (zshrc, bashrc, bash_profile, fish, profile):
        if candidate not in selected and candidate.is_file():
            selected.append(candidate)
    if not selected:
        selected = [profile]
    return tuple(selected)


# What a double-quoted word still interprets, per shell. POSIX shells run
# command substitution inside double quotes; fish does not, and escaping a
# backtick there would leave a literal backslash in the path.
_POSIX_SPECIALS = ("\\", '"', "`", "$")
_FISH_SPECIALS = ("\\", '"', "$")


def _escaped(text: str, specials: tuple[str, ...]) -> str:
    """Escape ``text`` for use inside a double-quoted shell word."""

    for character in specials:  # the backslash comes first, or it doubles the rest
        text = text.replace(character, f"\\{character}")
    return text


def _quoted_location(bin_directory: Path, home: Path, specials: tuple[str, ...]) -> str:
    """The directory as the inside of a double-quoted word, ``$HOME`` still live.

    A bin directory is not always a tame path: ``--bin-dir`` takes whatever the
    user types, and a space, a quote, or a ``$(`` reaching a shell profile
    unescaped would either break every future shell or run something.
    """

    try:
        relative = bin_directory.relative_to(home).as_posix()
    except ValueError:
        return _escaped(str(bin_directory), specials)
    return "$HOME/" + _escaped(relative, specials)


def _block(bin_directory: Path, profile: Path, home: Path) -> str:
    """The marked block for one profile, in that profile's own syntax."""

    if profile.name == "config.fish":
        line = f'fish_add_path "{_quoted_location(bin_directory, home, _FISH_SPECIALS)}"'
    else:
        line = f'export PATH="{_quoted_location(bin_directory, home, _POSIX_SPECIALS)}:$PATH"'
    return f"{BLOCK_START}\n{line}\n{BLOCK_END}\n"


def _home_relative(path: Path, home: Path) -> str:
    """Render a path under home as ``$HOME/...`` so the line survives a moved home."""

    try:
        relative = path.relative_to(home)
    except ValueError:
        return str(path)
    return f"$HOME/{relative.as_posix()}"


# Characters that continue a path token. A needle followed by one of these is a
# prefix of some other directory, not the directory itself.
_PATH_CHARACTERS = frozenset(string.ascii_letters + string.digits + "-_./~$")


def _mentions_whole_path(line: str, needle: str) -> bool:
    """True when ``needle`` appears in ``line`` as a complete path token.

    A plain substring test reports ``~/bin`` as present in a line that only
    mentions ``~/bin2``, which would leave the user with no PATH entry and an
    install that says their profile already had one.
    """

    start = line.find(needle)
    while start != -1:
        end = start + len(needle)
        before = start == 0 or line[start - 1] not in _PATH_CHARACTERS
        after = end >= len(line) or line[end] not in _PATH_CHARACTERS
        if before and after:
            return True
        start = line.find(needle, start + 1)
    return False


def profile_mentions_directory(text: str, bin_directory: Path, home: Path) -> bool:
    """True when a profile already puts ``bin_directory`` on PATH somehow.

    Deliberately loose about *how* the user wrote it -- our own block, an
    ``export PATH`` line, a ``$HOME``- or ``~``-relative spelling all count --
    but strict about the path itself, so a neighbouring directory whose name
    starts the same way does not silently suppress the entry.
    """

    if BLOCK_START in text:
        return True
    needles = {str(bin_directory), _home_relative(bin_directory, home)}
    with contextlib.suppress(ValueError):  # not under home; the two above suffice
        needles.add(f"~/{bin_directory.relative_to(home).as_posix()}")
    return any(
        _mentions_whole_path(line, needle) for line in text.splitlines() for needle in needles
    )


def update_profile(
    profile: Path,
    bin_directory: Path,
    *,
    home: Path | None = None,
) -> bool:
    """Append the marked block to one profile. Returns True when it was written."""

    home = home or Path.home()
    try:
        original: str | None = profile.read_text(encoding="utf-8")
    except FileNotFoundError:
        original = None
    except UnicodeDecodeError as exc:
        # A profile in some other encoding cannot be safely appended to:
        # rewriting it would decide an encoding on the user's behalf. This is a
        # failure, not a no-op -- returning False here would be indistinguishable
        # from "already configured", and the run would claim a PATH entry it
        # never wrote.
        raise InstallerError(f"{profile} is not valid UTF-8, so it was left alone") from exc
    # Any other OSError -- a permission wall, a directory where a file belongs --
    # is a real failure and travels up to update_profiles, which reports it.
    if original is not None and profile_mentions_directory(original, bin_directory, home):
        return False
    prefix = "" if not original else ("" if original.endswith("\n") else "\n")
    updated = (original or "") + prefix + _block(bin_directory, profile, home)
    return write_changed_configuration(profile, original, updated)


def update_profiles(
    bin_directory: Path,
    profiles: Sequence[Path],
    *,
    home: Path | None = None,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Append the block to each profile, keeping one file's failure isolated."""

    written: list[Path] = []
    failures: list[tuple[Path, str]] = []
    for profile in profiles:
        try:
            if update_profile(profile, bin_directory, home=home):
                written.append(profile)
        except (OSError, InstallerError) as exc:
            failures.append((profile, str(exc)))
    return written, failures


def remove_path_block(profile: Path) -> bool:
    """Remove every marked block from one profile. True when one was there."""

    try:
        original = profile.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    updated = original
    while True:
        start = updated.find(BLOCK_START)
        if start == -1:
            break
        end = updated.find(BLOCK_END, start)
        if end == -1:
            # A start marker with no end marker means the user edited the block
            # by hand. Removing to end-of-file would take their edit with it.
            break
        end += len(BLOCK_END)
        if updated[end : end + 1] == "\n":
            end += 1
        updated = updated[:start] + updated[end:]
    return write_changed_configuration(profile, original, updated)


def inspect(
    bin_directory: Path,
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> PathState:
    """Describe the PATH situation for a bin directory, for display before writing."""

    home = home or Path.home()
    environment = os.environ if environment is None else environment
    on_path = is_on_path(bin_directory, environment=environment)
    profiles = shell_profiles(home=home, environment=environment, platform_name=platform_name)
    current = True
    for profile in profiles:
        try:
            text = profile.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            current = False
            continue
        if not profile_mentions_directory(text, bin_directory, home):
            current = False
    return PathState(
        bin_directory=bin_directory,
        launcher=launcher_path(bin_directory, platform_name=platform_name),
        on_path=on_path,
        shadowed_by=shadowing_executable(
            bin_directory,
            environment=environment,
            platform_name=platform_name,
        ),
        profiles=profiles,
        profiles_current=current,
    )


def activation_hint(
    profiles: Sequence[Path],
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """The command that makes a freshly written PATH entry live in this shell."""

    environment = os.environ if environment is None else environment
    if any(profile.name == "config.fish" for profile in profiles):
        return "exec fish"
    shell = environment.get("SHELL")
    return f"exec {shell} -l" if shell else "exec $SHELL -l"
