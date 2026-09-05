# TUI Refinements Implementation Plan

> Execute sequentially with focused verification before each commit.

**Goal:** Make Syndex search, navigation, and indexing understandable and usable in small terminals.

**Architecture:** Keep the injected TuiService and Textual worker model. Bind asynchronous completions to a UI context generation. Use responsive classes, selectable detail targets, and local navigation history without adding backend APIs.

**Tech Stack:** Python, Textual 8, Rich, pytest headless pilots.

## 1. Context correctness
- Modify `src/code_indexing_mcp/tui/app.py`: invalidate search/detail work on query or project changes, apply completion checks on the UI thread, load project status asynchronously, clear stale previews.
- Extend `tests/test_tui.py` with empty-result and delayed completion regressions. Verify failures before implementation.

## 2. Responsive search and result presentation
- Modify `app.py` and `app.tcss`: compact project toolbar, full-width query row, explicit match modes, query focus, result hierarchy, compact Results/Details switching and wide split.
- Extend `tests/test_tui.py`: actual input width and pane visibility at 80x24, resize, keyboard search and match forwarding.

## 3. Navigable details
- Modify `app.py` and `service.py`: visible detail tabs, selectable outline/reference/impact destinations, source loading by declaration or bounded local context, preview on highlight, bounded history restoring selection/scroll.
- Test navigation round trips and stale preview suppression; service-test source resolution and path containment.

## 4. Progress and everyday usability
- Modify `app.py`, `app.tcss`, `service.py`: separate progress from errors, show lazy-index preparation, actionable empty states, contextual help, copy location and editor handoff, analysis limitations.
- Update `README.md` with final behavior and shortcuts.
- Test lazy progress, errors, help, copy/editor argument handling, and analysis limitations.

## Verification per implementation commit
Run with the existing repository virtual environment and `PYTHONPATH=src` so imports resolve to this worktree:

```
../../.venv/bin/python -m ruff format .
../../.venv/bin/python -m ruff check .
PYTHONPATH=src ../../.venv/bin/python -m mypy src
PYTHONPATH=src ../../.venv/bin/python -m pytest tests/test_tui.py tests/test_tui_service.py -q
```

After all commits: formatter check, lint, mypy, and full `pytest -n auto`. Inspect headless 80x24 and 120x32 layouts, including long paths and errors. Do not push or merge.
