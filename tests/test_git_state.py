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
    changed_paths_between,
    checkout_key,
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


def test_subdirectory_status_does_not_match_a_similarly_prefixed_path(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path, "repo")
    (root / "sub").mkdir()
    (root / "submarine.py").write_text("value = 2\n")
    state = probe_git_state(root / "sub", include_status=True)

    # Git reports paths relative to the repository root. `submarine.py` is not
    # inside the registered `sub` prefix and must remain invisible to it.
    assert state.project_prefix == "sub"
    assert "submarine.py" not in state.untracked_paths


def test_git_probe_rejects_a_toplevel_outside_the_registered_root(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")
    outside = tmp_path / "outside"
    outside.mkdir()
    identities = "\n".join([str(root / ".git"), str(root / ".git"), str(outside)])
    runner = _scripted({_IDENTITIES: GitCommandResult(returncode=0, stdout=identities)})

    state = probe_git_state(root, runner=runner)

    assert state.probe is GitProbeOutcome.INVALID


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
    # Build the expected prefix with Path, not string concatenation: Windows
    # resolves checkout identities with backslashes while a hand-written
    # "/worktrees/" fails to match its own representation of the same path.
    assert worktree_state.checkout_identity.startswith(str((root / ".git" / "worktrees").resolve()))
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
    run_git("worktree", "add", "-q", "-b", "elsewhere", str(worktree), cwd=root)
    worktree_state = probe_git_state(worktree)
    assert worktree_state.selector_value == "refs/heads/elsewhere"
    assert slot_id(project, detached) != slot_id(project, worktree_state)

    (root / "sub").mkdir()
    subdirectory = probe_git_state(root / "sub")
    assert slot_id(project, detached) != slot_id(project, subdirectory)

    other = _repo(tmp_path, "other")
    other_state = probe_git_state(other)
    assert slot_id(project, detached) != slot_id(project, other_state)


def test_a_selector_in_any_worktree_of_one_repository_shares_the_slot(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")
    project = "project-a"

    run_git("checkout", "-q", "--detach", cwd=root)
    head = _head_oid(root)
    main_state = probe_git_state(root)

    worktree = tmp_path / "wt"
    run_git("worktree", "add", "-q", "--detach", str(worktree), cwd=root)
    worktree_state = probe_git_state(worktree)

    assert main_state.repository_identity == worktree_state.repository_identity
    assert main_state.checkout_identity != worktree_state.checkout_identity
    assert main_state.selector_value == worktree_state.selector_value == head
    assert slot_id(project, main_state) == slot_id(project, worktree_state)


def test_checkout_key_is_per_checkout_and_slot_stays_per_branch(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")
    state = probe_git_state(root)

    worktree = tmp_path / "wt"
    run_git("worktree", "add", "-q", "--detach", str(worktree), cwd=root)
    worktree_state = probe_git_state(worktree)

    assert checkout_key(state) == state.checkout_identity
    assert checkout_key(worktree_state) == worktree_state.checkout_identity
    assert checkout_key(state) != checkout_key(worktree_state)

    non_git = tmp_path / "plain"
    non_git.mkdir()
    degraded = probe_git_state(non_git)
    assert checkout_key(degraded) == degraded.selector_value

    def timeout_runner(command: Sequence[str], cwd: Path) -> GitCommandResult:
        raise GitTimeout("too slow")

    timed_out = probe_git_state(root, runner=timeout_runner)
    assert checkout_key(timed_out) == str(root.resolve())


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


def test_changed_paths_between_lists_tracked_changes(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")
    first = _head_oid(root)
    (root / "main.py").write_text("value = 2\n")
    (root / "added.py").write_text("value = 3\n")
    _commit_all(root, "second")
    second = _head_oid(root)

    assert changed_paths_between(root, first, second) == frozenset({"main.py", "added.py"})
    # Identical OIDs describe no change at all, not an unknown answer.
    assert changed_paths_between(root, second, second) == frozenset()


def test_changed_paths_between_re_roots_onto_a_registered_subdirectory(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")
    (root / "sub").mkdir()
    (root / "sub" / "util.py").write_text("value = 1\n")
    (root / "other.py").write_text("value = 1\n")
    _commit_all(root, "first")
    first = _head_oid(root)
    (root / "sub" / "util.py").write_text("value = 2\n")
    (root / "other.py").write_text("value = 2\n")
    _commit_all(root, "second")
    second = _head_oid(root)

    changed = changed_paths_between(root / "sub", first, second, project_prefix="sub")

    # Git prints repository-root-relative paths; a project registered inside
    # sub/ must see its own subtree re-rooted, and only its own subtree.
    assert changed == frozenset({"util.py"})


def test_changed_paths_between_returns_none_when_the_diff_cannot_run(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()

    assert changed_paths_between(plain, "0" * 40, "1" * 40) is None


def test_changed_paths_between_returns_none_for_unreachable_commits(tmp_path: Path) -> None:
    root = _repo(tmp_path, "repo")

    assert changed_paths_between(root, "0" * 40, "1" * 40) is None
