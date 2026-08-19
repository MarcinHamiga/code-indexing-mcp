# Phase 8 notes — installer

Date: 2026-08-19
Status: implementation complete; D5 compiled binaries deferred
Branch: `ts-migration`
Plan: [2026-08-17-typescript-migration.md](2026-08-17-typescript-migration.md) §7

Phase 8 fills `ts/packages/installer` and adds a Bun-provisioning bootstrap.
The Python installer at the repository root remains the shipping product.

## What is in the tree

| TypeScript module | Responsibility | Tests |
|---|---|---|
| `src/config-files.ts` | Comment-preserving JSON/JSONC/TOML merge and removal | `test/config-files.test.ts` |
| `src/env-blocks.ts` | Harness env-block read/merge | `test/env-blocks.test.ts` |
| `src/links.ts` | Symlink replace/backup primitives | harness/shell-path tests |
| `src/settings-spec.ts` | Managed `CODE_INDEXING_*` catalog | CLI tests |
| `src/harnesses.ts` | Client config paths, merge, skills | `test/harnesses.test.ts` |
| `src/shell-path.ts` | Launcher + marked PATH block | `test/shell-path.test.ts` |
| `src/accelerator.ts` | Plan, probe, write `accelerator.json`; no second venv | `test/accelerator.test.ts` |
| `src/orchestrator.ts` | Event-emitting install pipeline | uninstall/wizard tests |
| `src/verify.ts` | Post-install checks as warnings | orchestrator path |
| `src/wizard.ts` | UI-agnostic wizard state | `test/wizard.test.ts` |
| `src/cli.ts` | Non-interactive configure entry | `test/cli.test.ts` |
| `src/uninstall.ts` | Evidence-based teardown | `test/uninstall.test.ts` |
| `src/update.ts` | Fast-forward + `bun install` + finalize | bootstrap tests |
| `src/bootstrap.ts` | Clone/update checkout, `bun install --frozen-lockfile` | `test/bootstrap.test.ts` |
| `src/tui/` | OpenTUI Core wizard; panel copy is UI-agnostic | `test/tui-content.test.ts` |
| `ts/install.sh` | Curl-pipeable bootstrap that runs the TS installer under Bun | — |

## Deliberate differences from the Python installer

- One Bun environment. Accelerator prep is `probeAccelerator` + `writeEnvironment`;
  there is no `.venv-accel` and no `uv sync --extra`.
- The server executable is `install_directory/bin/code-indexing-mcp`, a wrapper
  that execs Bun on `ts/packages/server/src/cli.ts`.
- Lock fingerprint hashes `ts/bun.lock`, not `uv.lock`.
- D5 (`bun build --compile`) is deferred; npm/Bun checkout distribution stays.
- Root `install.sh` / `install.py` are unchanged: Python still ships.

## Remaining

Phase 9 cutover soak. Real-hardware accelerator promotion is still pending from
Phase 7. OpenTUI panel chrome is a transliteration of the Textual wizard, not a
pixel-identical Pilot suite.
