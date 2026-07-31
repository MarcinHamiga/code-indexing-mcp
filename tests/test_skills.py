"""Validation for the skills bundled under src/code_indexing_mcp/skills/."""

import re
from pathlib import Path

import pytest

SKILLS_DIR = Path(__file__).resolve().parent.parent / "src" / "code_indexing_mcp" / "skills"
EXPECTED_SKILLS = {
    "codebase-exploration",
    "feature-dev",
    "impact-analysis",
    "indexed-review",
}


def _skill_dirs() -> list[Path]:
    return sorted(path for path in SKILLS_DIR.iterdir() if path.is_dir())


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
    assert "mcp__code-indexing-mcp__" not in text
    assert "mcp__code-indexing-mcp__" in text
