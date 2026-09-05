# Dead-code report

## Approved design

Add a project-scoped MCP `dead_code_report` tool and `dead-code-report` CLI command.
List exported declarations with no exact references as `review` findings, always
labelled `possibly_dead`. Preserve likely/unresolved reference counts and report
coverage and dynamic limitations. Source files are never edited by the report.

Use `ReferenceService`'s existing classifier, pinned to one structural-table version
and active partition. Load coverage and import/export context once for the batch,
then use each declaration's full classified hit list, not a paginated response.
Use structural declaration rows for full source ranges, including split chunks.
Exclude local export syntax from use counts; imports and re-exports in other files
retain the resolver's existing classification. This is not a reachability graph.

Python includes public module-level declarations and literal `__all__` entries;
restrictive/dynamic `__all__` is not evaluated. Other structural languages use
local export rows with matching declaration identity/ranges. Export forms and
members not captured by the extractor are explicitly outside the report's scope.
Inline export anchors must fall within the selected declaration, even when the
extractor assigns identical qualified names to declarations in different scopes;
detached export specifiers are handled separately.

Verify all coverage hashes, including files without old references, because a
changed file may introduce new uses. Aggregate coverage gaps even when no exports
exist. External callers and dynamic uses always prevent a definitive dead-code
claim. Return one deterministic, unpaginated report for batch use.

## Implementation

- `models.py`: `DeadCodeFinding` and `DeadCodeReport` response models.
- `reference_service.py`: export selection, snapshot-bound classification, counts,
  freshness checks and conservative completeness.
- `application.py`: project resolution, stable-checkout retry, structural backfill,
  partition access and the `ApplicationLike` interface.
- `server.py`, `daemon.py`, `cli.py`: MCP/daemon/CLI adapters. Bump the daemon RPC
  protocol to 5 so older processes are replaced using the existing mismatch flow.
- `tests/test_dead_code.py`: export rules across all eight structural languages,
  exact/uncertain uses, local exports, aliases, re-exports, full reference counts,
  coverage gaps, stale files and missing structural indexes.
- `tests/test_cli.py`, `tests/test_server.py`, `tests/test_daemon.py`: real adapter
  round trips and public tool/command contracts.
- `README.md`: usage, report fields and limitations. Remove the original backlog
  entry while preserving the pre-existing ranking-explanation changes.

## Verification

Write and run failing behavioral tests first, then implement and rerun them.
Run `uv run ruff format .`, `uv run ruff format --check .`, `uv run ruff check .`,
`uv run mypy src`, and `uv run pytest -n auto`.

In the restricted development environment, use `uv --cache-dir
/private/tmp/code-indexing-uv-cache run --no-sync ...` to run installed tools without
fetching unavailable dependency metadata. Daemon and socket tests require execution
outside the filesystem/network sandbox.
