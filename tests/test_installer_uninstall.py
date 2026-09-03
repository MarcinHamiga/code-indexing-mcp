"""Tests for taking an installation back out."""

import json
import sys
from pathlib import Path

import pytest

from code_indexing_mcp.installer import harnesses, shell_path
from code_indexing_mcp.installer.accelerator import server_executable
from code_indexing_mcp.installer.config_files import InstallerError
from code_indexing_mcp.installer.orchestrator import StepEvent
from code_indexing_mcp.installer.uninstall import (
    UninstallPlan,
    describe_plan,
    run_uninstall,
    uninstall_main,
)


def _checkout(tmp_path: Path) -> Path:
    directory = tmp_path / "checkout"
    command = server_executable(directory)
    command.parent.mkdir(parents=True, exist_ok=True)
    command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    command.chmod(0o755)
    # The markers --remove-checkout insists on before deleting anything.
    (directory / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (directory / "src" / "code_indexing_mcp").mkdir(parents=True, exist_ok=True)
    return directory


# --- config removal round-trips ----------------------------------------------


@pytest.mark.parametrize(
    "slug",
    [
        "kimi-code",
        "claude-code",
        "opencode",
        "kilocode",
        "codex",
        "antigravity",
        "antigravity-cli",
    ],
)
def test_configure_then_deconfigure_restores_the_original_file(tmp_path: Path, slug: str) -> None:
    """The strongest statement available: byte-identical, comments included."""

    checkout = _checkout(tmp_path)
    path = harnesses.configuration_path(slug, home=tmp_path, environment={})
    path.parent.mkdir(parents=True, exist_ok=True)
    original = (
        '# a comment the user wrote\n[other]\nkeep = "this"\n'
        if slug == "codex"
        else '{\n  // a comment the user wrote\n  "other": {"keep": "this"}\n}\n'
    )
    path.write_text(original, encoding="utf-8")

    harnesses.configure_harness(
        slug, server_executable(checkout), env={}, home=tmp_path, environment={}
    )
    assert harnesses.read_server_entry(slug, home=tmp_path, environment={}) is not None

    _, changed = harnesses.deconfigure_harness(slug, home=tmp_path, environment={})

    assert changed is True
    assert path.read_text(encoding="utf-8") == original
    assert harnesses.read_server_entry(slug, home=tmp_path, environment={}) is None


def test_codex_removal_keeps_the_users_spacing_between_their_own_tables(
    tmp_path: Path,
) -> None:
    """Our table is not always last; collapsing every newline would eat their blank lines."""

    checkout = _checkout(tmp_path)
    path = harnesses.configuration_path("codex", home=tmp_path, environment={})
    path.parent.mkdir(parents=True, exist_ok=True)
    original = "[first]\na = 1\n\n\n[last]\nz = 26\n"
    path.write_text(original, encoding="utf-8")

    harnesses.configure_harness(
        "codex", server_executable(checkout), env={}, home=tmp_path, environment={}
    )
    harnesses.deconfigure_harness("codex", home=tmp_path, environment={})

    assert path.read_text(encoding="utf-8") == original


def test_jsonc_removal_of_a_member_on_the_final_line(tmp_path: Path) -> None:
    """No trailing newline after the entry: the line still goes with it."""

    checkout = _checkout(tmp_path)
    path = harnesses.configuration_path("kimi-code", home=tmp_path, environment={})
    path.parent.mkdir(parents=True, exist_ok=True)
    original = '{\n  "other": {"keep": "this"},\n  "mcpServers": {}\n}'
    path.write_text(original, encoding="utf-8")

    harnesses.configure_harness(
        "kimi-code", server_executable(checkout), env={}, home=tmp_path, environment={}
    )
    harnesses.deconfigure_harness("kimi-code", home=tmp_path, environment={})

    text = path.read_text(encoding="utf-8")
    assert "code-indexing-mcp" not in text
    assert "\n\n" not in text  # no blank line left where the entry was
    assert json.loads(text)["other"] == {"keep": "this"}


def test_the_first_backup_survives_a_second_write(tmp_path: Path) -> None:
    """`.bak` is the file as the user wrote it, not as our previous run left it."""

    checkout = _checkout(tmp_path)
    path = harnesses.configuration_path("kimi-code", home=tmp_path, environment={})
    path.parent.mkdir(parents=True, exist_ok=True)
    pristine = '{\n  "other": {"keep": "this"}\n}\n'
    path.write_text(pristine, encoding="utf-8")

    harnesses.configure_harness(
        "kimi-code", server_executable(checkout), env={}, home=tmp_path, environment={}
    )
    harnesses.deconfigure_harness("kimi-code", home=tmp_path, environment={})

    assert path.with_name(f"{path.name}.bak").read_text(encoding="utf-8") == pristine
    assert path.with_name(f"{path.name}.bak.prev").exists()


def test_deconfigure_is_idempotent_and_honest_about_it(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    harnesses.configure_harness(
        "kimi-code", server_executable(checkout), env={}, home=tmp_path, environment={}
    )

    assert harnesses.deconfigure_harness("kimi-code", home=tmp_path, environment={})[1] is True
    assert harnesses.deconfigure_harness("kimi-code", home=tmp_path, environment={})[1] is False


def test_deconfigure_of_an_untouched_config_reports_no_change(tmp_path: Path) -> None:
    path = harnesses.configuration_path("kimi-code", home=tmp_path, environment={})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"mcpServers": {"someone-else": {"command": "x"}}}\n', encoding="utf-8")

    _, changed = harnesses.deconfigure_harness("kimi-code", home=tmp_path, environment={})

    assert changed is False
    assert "someone-else" in path.read_text(encoding="utf-8")


def test_deconfigure_keeps_the_neighbours_in_a_multi_server_config(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    path = harnesses.configuration_path("kimi-code", home=tmp_path, environment={})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{\n  "mcpServers": {\n'
        '    "before": {"command": "a"},\n'
        '    "after": {"command": "b"}\n'
        "  }\n}\n",
        encoding="utf-8",
    )
    harnesses.configure_harness(
        "kimi-code", server_executable(checkout), env={}, home=tmp_path, environment={}
    )

    harnesses.deconfigure_harness("kimi-code", home=tmp_path, environment={})

    text = path.read_text(encoding="utf-8")
    assert '"before"' in text and '"after"' in text
    assert "code-indexing-mcp" not in text
    # No dangling or doubled comma survived the cut.
    import json

    assert set(json.loads(text)["mcpServers"]) == {"before", "after"}


def test_deconfigure_removes_a_middle_entry_cleanly(tmp_path: Path) -> None:
    import json

    checkout = _checkout(tmp_path)
    path = harnesses.configuration_path("kimi-code", home=tmp_path, environment={})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{\n  "mcpServers": {\n    "before": {"command": "a"}\n  }\n}\n', "utf-8")
    harnesses.configure_harness(
        "kimi-code", server_executable(checkout), env={}, home=tmp_path, environment={}
    )
    # Add one after ours so the entry being removed is in the middle.
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mcpServers"]["after"] = {"command": "b"}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    harnesses.deconfigure_harness("kimi-code", home=tmp_path, environment={})

    assert set(json.loads(path.read_text(encoding="utf-8"))["mcpServers"]) == {"before", "after"}


def test_deconfigure_rejects_a_config_it_cannot_parse(tmp_path: Path) -> None:
    path = harnesses.configuration_path("kimi-code", home=tmp_path, environment={})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(InstallerError, match="Invalid JSON"):
        harnesses.deconfigure_harness("kimi-code", home=tmp_path, environment={})


# --- skills ------------------------------------------------------------------


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX symlinks")
def test_remove_skills_unlinks_only_the_bundled_ones(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    bundled = checkout / "src" / "code_indexing_mcp" / "skills" / "codebase-exploration"
    bundled.mkdir(parents=True)
    (bundled / "SKILL.md").write_text("---\n", encoding="utf-8")
    harnesses.install_skills(["kimi-code"], checkout, home=tmp_path, environment={})
    skills = harnesses.skill_directory("kimi-code", home=tmp_path, environment={})
    assert skills is not None
    theirs = skills / "their-own-skill"
    theirs.mkdir()

    ((_, message),) = harnesses.remove_skills(["kimi-code"], home=tmp_path, environment={})

    assert "1 unlinked" in message
    assert not (skills / "codebase-exploration").exists()
    assert theirs.is_dir()  # untouched


# --- the pipeline ------------------------------------------------------------


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX symlink launcher")
def test_run_uninstall_takes_back_the_entry_launcher_and_path_block(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    harnesses.configure_harness(
        "kimi-code", server_executable(checkout), env={}, home=tmp_path, environment={}
    )
    bin_directory = tmp_path / "bin"
    shell_path.install_launcher(checkout, bin_directory)
    profile = tmp_path / ".zshrc"
    profile.write_text("alias ll='ls -l'\n", encoding="utf-8")
    shell_path.update_profile(profile, bin_directory, home=tmp_path)
    events: list[StepEvent] = []

    result = run_uninstall(
        UninstallPlan(
            install_directory=checkout,
            harness_slugs=("kimi-code",),
            bin_directory=bin_directory,
        ),
        events.append,
        home=tmp_path,
        environment={"SHELL": "/bin/zsh"},
    )

    assert result.failures == []
    assert harnesses.read_server_entry("kimi-code", home=tmp_path, environment={}) is None
    assert not (bin_directory / "code-indexing-mcp").exists()
    assert result.launcher_removed == bin_directory / "code-indexing-mcp"
    assert profile.read_text(encoding="utf-8") == "alias ll='ls -l'\n"
    assert result.profiles_cleared == (profile,)
    assert {event.step for event in events} == {"harnesses", "skills", "path"}


def test_run_uninstall_keeps_data_directories_unless_asked(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()

    result = run_uninstall(
        UninstallPlan(install_directory=_checkout(tmp_path), remove_launcher=False),
        home=tmp_path,
        environment={"CODE_INDEXING_DATA_DIR": str(data), "SHELL": "/bin/zsh"},
    )

    assert result.directories_removed == ()
    assert data.is_dir()


def test_run_uninstall_purges_data_directories_when_asked(tmp_path: Path) -> None:
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    for directory in (data, cache):
        # "lancedb" is one of the markers that identifies the directory as ours.
        (directory / "lancedb").mkdir(parents=True)

    result = run_uninstall(
        UninstallPlan(
            install_directory=_checkout(tmp_path), remove_launcher=False, remove_data=True
        ),
        home=tmp_path,
        environment={
            "CODE_INDEXING_DATA_DIR": str(data),
            "CODE_INDEXING_CACHE_DIR": str(cache),
            "SHELL": "/bin/zsh",
        },
    )

    assert set(result.directories_removed) == {data, cache}
    assert not data.exists() and not cache.exists()


def test_purge_refuses_a_directory_merely_named_code_indexing_mcp(tmp_path: Path) -> None:
    """The old name-based short-circuit is gone: a name alone is not evidence."""

    data = tmp_path / "somewhere" / "code-indexing-mcp"
    cache = tmp_path / "elsewhere" / "code-indexing-mcp"
    data.mkdir(parents=True)
    cache.mkdir(parents=True)

    result = run_uninstall(
        UninstallPlan(
            install_directory=_checkout(tmp_path), remove_launcher=False, remove_data=True
        ),
        home=tmp_path,
        environment={
            "CODE_INDEXING_DATA_DIR": str(data),
            "CODE_INDEXING_CACHE_DIR": str(cache),
            "SHELL": "/bin/zsh",
        },
    )

    assert result.directories_removed == ()
    assert data.is_dir() and cache.is_dir()
    assert result.failures


def test_purge_accepts_a_directory_bearing_only_the_private_sentinel(tmp_path: Path) -> None:
    """A never-populated directory is still recognised by RuntimePaths.ensure_private's marker."""

    data = tmp_path / "data" / "code-indexing-mcp"
    cache = tmp_path / "cache" / "code-indexing-mcp"
    for directory in (data, cache):
        directory.mkdir(parents=True)
        (directory / ".code-indexing-mcp").write_text("", encoding="utf-8")

    result = run_uninstall(
        UninstallPlan(
            install_directory=_checkout(tmp_path), remove_launcher=False, remove_data=True
        ),
        home=tmp_path,
        environment={
            "CODE_INDEXING_DATA_DIR": str(data),
            "CODE_INDEXING_CACHE_DIR": str(cache),
            "SHELL": "/bin/zsh",
        },
    )

    assert set(result.directories_removed) == {data, cache}
    assert not data.exists() and not cache.exists()


def test_purge_refuses_a_directory_that_holds_nothing_of_ours(tmp_path: Path) -> None:
    """A setting can point anywhere; a confirmation prompt is not a safety net."""

    documents = tmp_path / "Documents"
    (documents / "taxes").mkdir(parents=True)

    result = run_uninstall(
        UninstallPlan(
            install_directory=_checkout(tmp_path), remove_launcher=False, remove_data=True
        ),
        home=tmp_path,
        environment={
            "CODE_INDEXING_DATA_DIR": str(documents),
            "CODE_INDEXING_CACHE_DIR": str(documents),
            "SHELL": "/bin/zsh",
        },
    )

    assert result.directories_removed == ()
    assert (documents / "taxes").is_dir()
    assert result.failures


def test_purge_refuses_the_home_directory(tmp_path: Path) -> None:
    result = run_uninstall(
        UninstallPlan(
            install_directory=_checkout(tmp_path), remove_launcher=False, remove_data=True
        ),
        home=tmp_path,
        environment={
            "CODE_INDEXING_DATA_DIR": str(tmp_path),
            "CODE_INDEXING_CACHE_DIR": str(tmp_path / "cache"),
            "SHELL": "/bin/zsh",
        },
    )

    assert result.directories_removed == ()
    assert tmp_path.is_dir()
    assert any("home directory" in message for _, message in result.failures)


def test_remove_checkout_refuses_a_directory_that_is_not_a_checkout(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "work").mkdir(parents=True)

    result = run_uninstall(
        UninstallPlan(install_directory=elsewhere, remove_launcher=False, remove_checkout=True),
        home=tmp_path,
        environment={"SHELL": "/bin/zsh"},
    )

    assert result.directories_removed == ()
    assert (elsewhere / "work").is_dir()


def test_remove_skills_leaves_another_installations_links_alone(tmp_path: Path) -> None:
    """Two checkouts share a skill directory; uninstalling one keeps the other."""

    ours = _checkout(tmp_path)
    theirs = tmp_path / "other-checkout"
    skills = tmp_path / ".agents" / "skills"
    skills.mkdir(parents=True)
    for checkout, name in ((ours, "mine"), (theirs, "not-mine")):
        source = checkout / "src" / "code_indexing_mcp" / "skills" / name
        source.mkdir(parents=True)
        (skills / name).symlink_to(source, target_is_directory=True)

    harnesses.remove_skills(["kimi-code"], ours, home=tmp_path, environment={})

    assert not (skills / "mine").is_symlink()
    assert (skills / "not-mine").is_symlink()


def test_run_uninstall_isolates_one_harnesss_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _checkout(tmp_path)
    bin_directory = tmp_path / "bin"
    shell_path.install_launcher(checkout, bin_directory)
    monkeypatch.setattr(
        harnesses,
        "deconfigure_harness",
        lambda slug, **kwargs: (_ for _ in ()).throw(InstallerError("unwritable")),
    )

    result = run_uninstall(
        UninstallPlan(
            install_directory=checkout,
            harness_slugs=("kimi-code",),
            bin_directory=bin_directory,
        ),
        home=tmp_path,
        environment={"SHELL": "/bin/zsh"},
    )

    assert result.failures == [("kimi-code", "unwritable")]
    # The launcher still went, which is the point of isolating the failure.
    assert result.launcher_removed is not None


# --- the command -------------------------------------------------------------


def test_uninstall_main_asks_before_doing_anything(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    printed: list[str] = []

    code = uninstall_main(
        install_dir=str(checkout),
        harnesses_selection="",
        input_fn=lambda prompt: "n",
        output=printed.append,
        error_output=printed.append,
    )

    assert code == 130
    assert "Uninstall cancelled." in printed


def test_uninstall_main_runs_when_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _checkout(tmp_path)
    ran: list[object] = []
    monkeypatch.setattr(
        "code_indexing_mcp.installer.uninstall.run_uninstall",
        lambda plan, on_event=lambda event: None: ran.append(plan) or _empty_result(),
    )

    code = uninstall_main(
        install_dir=str(checkout),
        harnesses_selection="kimi-code",
        assume_yes=True,
        output=lambda line: None,
        error_output=lambda line: None,
    )

    assert code == 0
    (plan,) = ran
    assert plan.harness_slugs == ("kimi-code",)  # type: ignore[attr-defined]


def _empty_result():  # type: ignore[no-untyped-def]
    from code_indexing_mcp.installer.uninstall import UninstallResult

    return UninstallResult()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX symlink launcher")
def test_the_uninstall_subcommand_runs_the_real_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end through the console script's own argument parsing."""

    from code_indexing_mcp.cli import main

    checkout = _checkout(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    harnesses.configure_harness(
        "kimi-code", server_executable(checkout), env={}, home=tmp_path, environment={}
    )
    bin_directory = tmp_path / "bin"
    shell_path.install_launcher(checkout, bin_directory)

    code = main(
        [
            "uninstall",
            "--install-dir",
            str(checkout),
            "--harnesses",
            "kimi-code",
            "--bin-dir",
            str(bin_directory),
            "--yes",
        ]
    )

    assert code == 0
    assert not (bin_directory / "code-indexing-mcp").exists()
    assert harnesses.read_server_entry("kimi-code", home=tmp_path, environment={}) is None
    out = capsys.readouterr().out
    assert "Uninstall complete." in out
    assert "--purge" in out  # says what it deliberately did not delete


def test_describe_plan_names_the_destructive_parts_loudly(tmp_path: Path) -> None:
    lines = describe_plan(
        UninstallPlan(
            install_directory=tmp_path / "checkout",
            harness_slugs=("kimi-code",),
            bin_directory=tmp_path / "bin",
            remove_data=True,
            remove_checkout=True,
        ),
        home=tmp_path,
        environment={"CODE_INDEXING_DATA_DIR": str(tmp_path / "data")},
    )

    text = "\n".join(lines)
    assert "DELETE" in text
    assert str(tmp_path / "checkout") in text
    assert "code-indexing-mcp" in text
