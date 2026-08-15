"""Git state probing and slot-key identity for branch-aware index slots."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

from conftest import run_git

from code_indexing_mcp.git_state import (
    GitCommandResult,
    GitProbeOutcome,
    GitTimeout,
    GitUnavailable,
    SelectorKind,
    WorktreeStatus,
    partition_id,
    probe_git_state,
    slot_id,
)


def _commit_all(root: Path, message: str) -> None:
    run_git("add", "-A", cwd=root)
    run_git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", message, cwd=root)


def _repo(tmp_path: Path, name: str, *, initial_branch: str = "main") -> Path:
    root = tmp_path / name
    root.mkdir()
    run_git("init", "-q", "--initial-branch", initial_branch, str(root))
    (root / "main.py").write_text("value = 1\n")
    _commit_all(root, "init")
    return root


def _head_oid(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _scripted(mapping: dict[tuple[str, ...], GitCommandResult | Exception]):
    def runner(command: Sequence[str], cwd: Path) -> GitCommandResult:
        key = tuple(command)
        if key not in mapping:
            raise AssertionError(f"unexpected git call: {command}")
        outcome = mapping[key]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return runner


def test_attached_branch_reports_full_ref_and_oid(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")

    state = probe_git_state(root, include_status=True)

    assert state.probe is GitProbeOutcome.GIT
    assert state.selector_kind is SelectorKind.REF
    assert state.selector_value == "refs/heads/main"
    assert state.head_oid == _head_oid(root)
    assert state.worktree is WorktreeStatus.CLEAN
    assert state.project_prefix == ""
    assert state.toplevel == str(root.resolve())
    assert state.repository_identity == str((root / ".git").resolve())
    assert state.checkout_identity == str((root / ".git").resolve())
    assert state.status_fingerprint is not None


def test_dirty_untracked_and_mixed_worktrees_are_classified(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")

    (root / "main.py").write_text("value = 2\n")
    dirty = probe_git_state(root, include_status=True)
    assert dirty.worktree is WorktreeStatus.TRACKED_DIRTY
    assert dirty.dirty_paths == ("main.py",)
    assert dirty.untracked_paths == ()
    assert dirty.status_fingerprint is not None

    (root / "extra.py").write_text("value = 3\n")
    mixed = probe_git_state(root, include_status=True)
    assert mixed.worktree is WorktreeStatus.MIXED
    assert mixed.dirty_paths == ("main.py",)
    assert mixed.untracked_paths == ("extra.py",)
    assert mixed.status_fingerprint != dirty.status_fingerprint

    (root / "main.py").write_text("value = 1\n")
    untracked = probe_git_state(root, include_status=True)
    assert untracked.worktree is WorktreeStatus.UNTRACKED
    assert untracked.untracked_paths == ("extra.py",)


def test_status_is_only_collected_when_requested(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")
    (root / "main.py").write_text("value = 2\n")

    state = probe_git_state(root)

    assert state.worktree is WorktreeStatus.UNKNOWN
    assert state.dirty_paths == ()
    assert state.status_fingerprint is None


def test_detached_head_selects_by_full_commit_oid(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")
    run_git("checkout", "-q", "--detach", cwd=root)

    state = probe_git_state(root)

    assert state.selector_kind is SelectorKind.COMMIT
    assert state.selector_value == _head_oid(root)
    assert state.head_oid == _head_oid(root)


def test_unborn_branch_keeps_its_symbolic_ref_without_a_head(tmp_path: Path) -> None:
    root = tmp_path / "unborn"
    root.mkdir()
    run_git("init", "-q", "--initial-branch", "feature", str(root))
    (root / "main.py").write_text("value = 1\n")

    state = probe_git_state(root, include_status=True)

    assert state.probe is GitProbeOutcome.GIT
    assert state.selector_kind is SelectorKind.REF
    assert state.selector_value == "refs/heads/feature"
    assert state.head_oid is None
    assert state.worktree is WorktreeStatus.UNTRACKED


def test_subdirectory_root_reports_prefix_and_resolves_relative_directories(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path, "repo")
    (root / "sub").mkdir()
    (root / "sub" / "inner.py").write_text("value = 1\n")

    state = probe_git_state(root / "sub", include_status=True)

    assert state.project_prefix == "sub"
    # A subdirectory reports a relative --git-common-dir ('../.git'); it must
    # resolve against the registered root, matching the registered-root rule
    # the storage worktree warnings already follow.
    assert state.repository_identity == str((root / ".git").resolve())
    assert state.toplevel == str(root.resolve())
    assert state.untracked_paths == ("inner.py",)


def test_non_git_directory_falls_back_to_the_workspace_selector(tmp_path: Path) -> None:
    root = tmp_path / "plain"
    root.mkdir()

    state = probe_git_state(root)

    assert state.probe is GitProbeOutcome.NOT_GIT
    assert state.selector_kind is SelectorKind.WORKSPACE
    assert state.selector_value == str(root.resolve())
    assert state.repository_identity is None
    assert state.checkout_identity is None
    assert state.toplevel == str(root.resolve())


def test_linked_worktree_shares_repository_but_not_checkout_identity(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")
    worktree = tmp_path / "wt"
    run_git("worktree", "add", "-q", str(worktree), cwd=root)

    main_state = probe_git_state(root)
    worktree_state = probe_git_state(worktree)

    assert main_state.repository_identity == worktree_state.repository_identity
    assert main_state.checkout_identity != worktree_state.checkout_identity
    assert worktree_state.checkout_identity is not None
    assert worktree_state.checkout_identity.startswith(
        str((root / ".git").resolve()) + "/worktrees/"
    )
    # `git worktree add <path>` checks out an auto-created branch; both
    # checkouts are attached, each on its own selector.
    assert main_state.selector_kind is SelectorKind.REF
    assert worktree_state.selector_kind is SelectorKind.REF


def test_missing_git_binary_is_reported_as_unavailable(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")

    def runner(command: Sequence[str], cwd: Path) -> GitCommandResult:
        raise GitUnavailable("git is missing")

    state = probe_git_state(root, runner=runner)

    assert state.probe is GitProbeOutcome.UNAVAILABLE
    assert state.selector_kind is SelectorKind.WORKSPACE
    assert state.repository_identity is None


def test_slow_git_is_reported_as_a_timeout(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")

    def runner(command: Sequence[str], cwd: Path) -> GitCommandResult:
        raise GitTimeout("too slow")

    state = probe_git_state(root, runner=runner)

    assert state.probe is GitProbeOutcome.TIMEOUT
    assert state.selector_kind is SelectorKind.WORKSPACE


_IDENTITIES = ("git", "rev-parse", "--git-common-dir", "--git-dir", "--show-toplevel")
_HEAD = ("git", "rev-parse", "HEAD")
_SYMBOLIC_REF = ("git", "symbolic-ref", "-q", "HEAD")
_STATUS = ("git", "status", "--porcelain=v1", "-z", "--untracked-files=all")


def test_malformed_identity_output_is_invalid(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")
    runner = _scripted({_IDENTITIES: GitCommandResult(returncode=0, stdout="one-line-only")})

    state = probe_git_state(root, runner=runner)

    assert state.probe is GitProbeOutcome.INVALID


def test_git_directory_without_a_repository_is_not_git(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")
    runner = _scripted({_IDENTITIES: GitCommandResult(returncode=128, stdout="")})

    state = probe_git_state(root, runner=runner)

    assert state.probe is GitProbeOutcome.NOT_GIT


def test_damaged_head_metadata_is_invalid(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")
    identities = "\n".join([str(root / ".git"), str(root / ".git"), str(root)])
    runner = _scripted(
        {
            _IDENTITIES: GitCommandResult(returncode=0, stdout=identities),
            _HEAD: GitCommandResult(returncode=128, stdout=""),
            _SYMBOLIC_REF: GitCommandResult(returncode=1, stdout=""),
        }
    )

    state = probe_git_state(root, runner=runner)

    assert state.probe is GitProbeOutcome.INVALID


def test_status_failure_leaves_the_probe_usable(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")
    identities = "\n".join([str(root / ".git"), str(root / ".git"), str(root)])
    runner = _scripted(
        {
            _IDENTITIES: GitCommandResult(returncode=0, stdout=identities),
            _HEAD: GitCommandResult(returncode=0, stdout=_head_oid(root)),
            _SYMBOLIC_REF: GitCommandResult(returncode=0, stdout="refs/heads/main"),
            _STATUS: GitTimeout("status timed out"),
        }
    )

    state = probe_git_state(root, include_status=True, runner=runner)

    assert state.probe is GitProbeOutcome.GIT
    assert state.head_oid == _head_oid(root)
    assert state.worktree is WorktreeStatus.UNKNOWN
    assert state.status_fingerprint is None


def test_mutable_head_and_dirty_state_do_not_change_the_slot(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")
    project = "project-a"

    before = probe_git_state(root, include_status=True)
    (root / "main.py").write_text("value = 2\n")
    _commit_all(root, "second")
    after = probe_git_state(root, include_status=True)

    assert after.head_oid != before.head_oid
    assert after.worktree is WorktreeStatus.CLEAN
    assert slot_id(project, before) == slot_id(project, after)


def test_selector_repository_checkout_and_prefix_changes_move_the_slot(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")
    project = "project-a"
    attached = probe_git_state(root)

    run_git("branch", "-m", "main", "renamed", cwd=root)
    renamed = probe_git_state(root)
    assert renamed.selector_value == "refs/heads/renamed"
    assert slot_id(project, attached) != slot_id(project, renamed)

    run_git("checkout", "-q", "--detach", cwd=root)
    detached = probe_git_state(root)
    assert slot_id(project, attached) != slot_id(project, detached)

    worktree = tmp_path / "wt"
    run_git("worktree", "add", "-q", str(worktree), cwd=root)
    worktree_state = probe_git_state(worktree)
    assert slot_id(project, detached) != slot_id(project, worktree_state)

    (root / "sub").mkdir()
    subdirectory = probe_git_state(root / "sub")
    assert slot_id(project, attached) != slot_id(project, subdirectory)

    other = _repo(tmp_path, "other")
    other_state = probe_git_state(other)
    assert other_state.selector_value == attached.selector_value
    assert slot_id(project, attached) != slot_id(project, other_state)


def test_degraded_probes_never_share_a_branch_slot(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")
    project = "project-a"
    attached = probe_git_state(root)

    def runner(command: Sequence[str], cwd: Path) -> GitCommandResult:
        raise GitTimeout("too slow")

    degraded = probe_git_state(root, runner=runner)

    assert slot_id(project, attached) != slot_id(project, degraded)


def test_partition_ids_are_opaque_and_path_safe(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")
    slot = slot_id("project-a", probe_git_state(root))

    partition = partition_id(slot)

    assert partition.startswith("slot-")
    assert len(partition) == len("slot-") + 32
    assert all(character in "0123456789abcdef" for character in partition[5:])
    assert "/" not in partition
    assert partition_id(slot) == partition
    assert partition_id(slot_id("project-b", probe_git_state(root))) != partition
