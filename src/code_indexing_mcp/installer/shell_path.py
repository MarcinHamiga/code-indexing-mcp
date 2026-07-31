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

import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .accelerator import server_executable
from .config_files import _write_changed_configuration
from .links import backup_path, link_destination, replace_link

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
    backup = (
        backup_path(target) if (target.is_symlink() or target.exists()) and not stale else None
    )
    created = replace_link(executable, target, is_directory=False, stale=stale)
    if not created:
        return LauncherResult(target, "current", f"already points at {executable}")
    if backup is not None:
        return LauncherResult(target, "replaced", f"the previous entry was kept as {backup.name}")
    return LauncherResult(target, "created", f"points at {executable}")


_SHIM_TEMPLATE = "@echo off\r\n\"{executable}\" %*\r\n"


def _install_shim(executable: Path, target: Path) -> LauncherResult:
    """Windows gets a batch shim: symlinks there need a privilege we cannot assume."""

    content = _SHIM_TEMPLATE.format(executable=executable)
    # Read as bytes: the shim's CRLF line endings are written verbatim, and
    # universal-newline decoding would turn every re-install into a rewrite.
    existing = target.read_bytes().decode("utf-8") if target.is_file() else None
    if existing == content:
        return LauncherResult(target, "current", f"already runs {executable}")
    target.parent.mkdir(parents=True, exist_ok=True)
    changed = _write_changed_configuration(target, existing, content)
    if not changed:  # pragma: no cover - equality was already handled above
        return LauncherResult(target, "current", f"already runs {executable}")
    status = "replaced" if existing is not None else "created"
    return LauncherResult(target, status, f"runs {executable}")


def remove_launcher(
    bin_directory: Path,
    *,
    platform_name: str | None = None,
) -> Path | None:
    """Remove a launcher this installer created. Returns the path if one went away.

    A file at that name that is *not* one of ours is left strictly alone.
    """

    platform_name = platform_name or sys.platform
    target = launcher_path(bin_directory, platform_name=platform_name)
    if platform_name.startswith("win"):
        if not target.is_file():
            return None
        try:
            content = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        if ".venv" not in content or LAUNCHER_NAME not in content:
            return None
    elif not target.is_symlink() or ".venv" not in link_destination(target).parts:
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


def _block(bin_directory: Path, profile: Path, home: Path) -> str:
    """The marked block for one profile, in that profile's own syntax."""

    location = _home_relative(bin_directory, home)
    if profile.name == "config.fish":
        line = f"fish_add_path {location}"
    else:
        line = f'export PATH="{location}:$PATH"'
    return f"{BLOCK_START}\n{line}\n{BLOCK_END}\n"


def _home_relative(path: Path, home: Path) -> str:
    """Render a path under home as ``$HOME/...`` so the line survives a moved home."""

    try:
        relative = path.relative_to(home)
    except ValueError:
        return str(path)
    return f"$HOME/{relative.as_posix()}"


def profile_mentions_directory(text: str, bin_directory: Path, home: Path) -> bool:
    """True when a profile already puts ``bin_directory`` on PATH somehow.

    Deliberately loose: a user who wrote their own ``export PATH`` line for this
    directory should not receive a second one from us.
    """

    if BLOCK_START in text:
        return True
    needles = {str(bin_directory), _home_relative(bin_directory, home)}
    return any(needle in line for line in text.splitlines() for needle in needles)


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
    except UnicodeDecodeError:
        # A profile in some other encoding cannot be safely appended to:
        # rewriting it would decide an encoding on the user's behalf.
        return False
    # Any other OSError -- a permission wall, a directory where a file belongs --
    # is a real failure and travels up to update_profiles, which reports it.
    if original is not None and profile_mentions_directory(original, bin_directory, home):
        return False
    prefix = "" if not original else ("" if original.endswith("\n") else "\n")
    updated = (original or "") + prefix + _block(bin_directory, profile, home)
    return _write_changed_configuration(profile, original, updated)


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
        except OSError as exc:
            failures.append((profile, str(exc)))
    return written, failures


def remove_path_block(profile: Path) -> bool:
    """Remove the marked block from one profile. Returns True when it was there."""

    try:
        original = profile.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    start = original.find(BLOCK_START)
    if start == -1:
        return False
    end = original.find(BLOCK_END, start)
    if end == -1:
        # A start marker with no end marker means the user edited the block by
        # hand. Removing to end-of-file would take their edit with it.
        return False
    end += len(BLOCK_END)
    if original[end : end + 1] == "\n":
        end += 1
    updated = original[:start] + original[end:]
    return _write_changed_configuration(profile, original, updated)


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
