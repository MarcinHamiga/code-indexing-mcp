"""Validation for the skills bundled under src/code_indexing_mcp/skills/."""

import re
from pathlib import Path

import pytest

SKILLS_DIR = Path(__file__).resolve().parent.parent / "src" / "code_indexing_mcp" / "skills"
EXPECTED_SKILLS = {
    "codebase-exploration",
    "cross-repo-debugging",
    "feature-dev",
    "impact-analysis",
    "indexed-review",
}


def _skill_dirs() -> list[Path]:
    return sorted(path for path in SKILLS_DIR.iterdir() if path.is_dir())


def test_no_skill_still_claims_find_symbol_returns_call_sites() -> None:
    """T6: find_symbol resolves declarations by name only; it has never

    returned call sites. `SERVER_INSTRUCTIONS`, `impact-analysis`, and
    `feature-dev` already say so correctly -- this is the last stale copy.
    """
    for skill_dir in _skill_dirs():
        text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert "definitions and call sites" not in text, (
            f"{skill_dir.name}/SKILL.md still claims find_symbol returns call sites"
        )


def _skill_dirs_param() -> list[Path]:
    return _skill_dirs() if SKILLS_DIR.is_dir() else []


def test_all_expected_skills_are_bundled() -> None:
    assert {path.name for path in _skill_dirs()} == EXPECTED_SKILLS


@pytest.mark.parametrize("skill_dir", _skill_dirs_param(), ids=lambda p: p.name)
def test_skill_has_valid_frontmatter(skill_dir: Path) -> None:
    skill_md = skill_dir / "SKILL.md"
    assert skill_md.is_file(), f"missing {skill_md}"
    text = skill_md.read_text(encoding="utf-8")
    frontmatter = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
    assert frontmatter is not None, f"{skill_md} has no frontmatter block"
    name = re.search(r"^name: (.+)$", frontmatter.group(1), re.MULTILINE)
    description = re.search(r"^description: (.+)$", frontmatter.group(1), re.MULTILINE)
    assert name is not None and name.group(1).strip() == skill_dir.name
    assert description is not None and description.group(1).strip()


@pytest.mark.parametrize("skill_dir", _skill_dirs_param(), ids=lambda p: p.name)
def test_skill_references_only_code_indexing_mcp_tools(skill_dir: Path) -> None:
    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    referenced_servers = set(re.findall(r"mcp__([\w-]+?)__", text))
    assert referenced_servers == {"code-indexing-mcp"}


@pytest.mark.parametrize("skill_name", ["impact-analysis", "feature-dev"])
def test_refactoring_workflows_name_the_structural_analysis_tools(skill_name: str) -> None:
    text = (SKILLS_DIR / skill_name / "SKILL.md").read_text(encoding="utf-8")

    assert "mcp__code-indexing-mcp__find_references" in text
    assert "mcp__code-indexing-mcp__analyze_refactor" in text


@pytest.mark.parametrize("skill_name", ["impact-analysis", "feature-dev"])
def test_refactoring_workflows_state_language_coverage_and_completeness(skill_name: str) -> None:
    """T5: a skill that drives find_references/analyze_refactor must tell the

    agent which languages those tools actually cover, what an unsupported
    language returns, and how to read the completeness contract -- otherwise
    the agent has no way to know a rename check that came back clean was
    silently skipped for, say, a C or Lua file. Pinned to the full
    `STRUCTURAL_LANGUAGES` set so a new language step must sweep the skills
    in the same change.
    """
    text = (SKILLS_DIR / skill_name / "SKILL.md").read_text(encoding="utf-8")

    for language in (
        "C#",
        "Go",
        "Java",
        "JavaScript",
        "Python",
        "Rust",
        "TSX",
        "TypeScript",
    ):
        assert language in text
    assert "UNSUPPORTED_LANGUAGE" in text
    assert "completeness" in text.lower()
