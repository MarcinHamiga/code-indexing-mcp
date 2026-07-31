"""Tests for the Textual-free wizard state."""

import json
from pathlib import Path

import pytest

from code_indexing_mcp.installer import accelerator
from code_indexing_mcp.installer.wizard import WizardState, load_prefill


def _write_kimi_config(home: Path, env: dict[str, str]) -> None:
    directory = home / ".kimi-code"
    directory.mkdir(parents=True)
    (directory / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "code-indexing-mcp": {
                        "command": "/opt/ci-mcp",
                        "args": ["serve"],
                        "env": env,
                    }
                }
            }
        )
    )


def test_prepared_accelerator_reads_the_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record = tmp_path / "accelerator.json"
    record.write_text(json.dumps({"accelerator": "mlx"}))
    monkeypatch.setattr(accelerator, "accelerator_record_path", lambda directory: record)
    assert accelerator.prepared_accelerator(tmp_path) == "mlx"
    record.unlink()
    assert accelerator.prepared_accelerator(tmp_path) is None


def test_detection_report_mentions_platform_and_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "_nvidia_smi_report", lambda: "555.42, GeForce RTX")
    monkeypatch.setattr(accelerator, "_rocm_report", lambda: None)
    facts = accelerator.detection_report()
    assert any(line.startswith("Platform:") for line in facts)
    assert any("555.42" in line for line in facts)
    assert any(line == "ROCm: not detected" for line in facts)


def test_load_prefill_collects_values_and_configured_harnesses(tmp_path: Path) -> None:
    _write_kimi_config(tmp_path, {"CODE_INDEXING_INDEX_MODE": "eager", "UNRELATED": "keep"})
    prefill = load_prefill(home=tmp_path)
    assert prefill.values == {"CODE_INDEXING_INDEX_MODE": "eager"}
    assert prefill.configured_slugs == ("kimi-code",)
    assert prefill.disagreements == ()


def test_load_prefill_reports_disagreements_in_choice_order(tmp_path: Path) -> None:
    _write_kimi_config(tmp_path, {"CODE_INDEXING_INDEX_MODE": "manual"})
    codex = tmp_path / ".codex"
    codex.mkdir()
    (codex / "config.toml").write_text(
        '[mcp_servers.code-indexing-mcp]\ncommand = "/opt/ci-mcp"\nargs = ["serve"]\n'
        'env = { CODE_INDEXING_INDEX_MODE = "eager" }\n'
    )
    prefill = load_prefill(home=tmp_path)
    # codex precedes kimi-code in HARNESS_CHOICES, so its value wins.
    assert prefill.values == {"CODE_INDEXING_INDEX_MODE": "eager"}
    assert prefill.disagreements == ("CODE_INDEXING_INDEX_MODE",)
    assert prefill.configured_slugs == ("codex", "kimi-code")


def test_env_updates_omit_defaults_and_delete_reset_prefills(tmp_path: Path) -> None:
    _write_kimi_config(
        tmp_path, {"CODE_INDEXING_INDEX_MODE": "eager", "CODE_INDEXING_BROKER": "off"}
    )
    state = WizardState.for_reconfigure(Path("/opt/ci-mcp"), home=tmp_path)
    state.set_field("CODE_INDEXING_INDEX_MODE", "lazy")  # back to default -> delete
    state.set_field("CODE_INDEXING_BROKER", "on")  # non-default -> write
    state.set_field("CODE_INDEXING_EMBED_THREADS", "")  # untouched -> no entry
    assert state.env_updates() == {"CODE_INDEXING_INDEX_MODE": None, "CODE_INDEXING_BROKER": "on"}


def test_install_mode_deletes_reset_prefills_too(tmp_path: Path) -> None:
    _write_kimi_config(tmp_path, {"CODE_INDEXING_INDEX_MODE": "eager"})
    state = WizardState.for_install(Path("/opt/ci-mcp"), home=tmp_path)
    state.set_field("CODE_INDEXING_INDEX_MODE", "lazy")  # back to default -> delete
    assert state.env_updates() == {"CODE_INDEXING_INDEX_MODE": None}


def test_env_updates_leave_settings_the_wizard_never_prefilled_alone(tmp_path: Path) -> None:
    state = WizardState.for_install(Path("/opt/ci-mcp"), home=tmp_path)
    state.set_field("CODE_INDEXING_INDEX_MODE", "lazy")
    assert state.env_updates() == {}


def test_load_prefill_canonicalizes_hand_written_values(tmp_path: Path) -> None:
    _write_kimi_config(
        tmp_path, {"CODE_INDEXING_OFFLINE": "true", "CODE_INDEXING_INDEX_MODE": "EAGER"}
    )
    prefill = load_prefill(home=tmp_path)
    assert prefill.values == {"CODE_INDEXING_OFFLINE": "1", "CODE_INDEXING_INDEX_MODE": "eager"}


def test_load_prefill_ignores_values_the_catalog_cannot_read(tmp_path: Path) -> None:
    _write_kimi_config(
        tmp_path, {"CODE_INDEXING_INDEX_MODE": "whenever", "CODE_INDEXING_BROKER": "off"}
    )
    prefill = load_prefill(home=tmp_path)
    assert prefill.values == {"CODE_INDEXING_BROKER": "off"}


def test_load_prefill_does_not_call_spellings_of_one_value_a_disagreement(tmp_path: Path) -> None:
    _write_kimi_config(tmp_path, {"CODE_INDEXING_OFFLINE": "yes"})
    codex = tmp_path / ".codex"
    codex.mkdir()
    (codex / "config.toml").write_text(
        '[mcp_servers.code-indexing-mcp]\ncommand = "/opt/ci-mcp"\nargs = ["serve"]\n'
        'env = { CODE_INDEXING_OFFLINE = "1" }\n'
    )
    prefill = load_prefill(home=tmp_path)
    assert prefill.values == {"CODE_INDEXING_OFFLINE": "1"}
    assert prefill.disagreements == ()


def test_to_plan_carries_everything(tmp_path: Path) -> None:
    state = WizardState.for_install(Path("/opt/ci-mcp"), home=tmp_path)
    state.accelerator = "mlx"
    state.harness_slugs = ["kimi-code"]
    state.set_field("CODE_INDEXING_OFFLINE", "1")
    plan = state.to_plan()
    assert plan.install_directory == Path("/opt/ci-mcp")
    assert plan.accelerator == "mlx"
    assert plan.harness_slugs == ("kimi-code",)
    assert plan.env_updates == {"CODE_INDEXING_OFFLINE": "1"}


def test_for_reconfigure_keeps_prepared_backend_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(accelerator, "prepared_accelerator", lambda directory: "mlx")
    state = WizardState.for_reconfigure(Path("/opt/ci-mcp"), home=tmp_path)
    assert state.accelerator is None
    assert state.prepared_accelerator == "mlx"
