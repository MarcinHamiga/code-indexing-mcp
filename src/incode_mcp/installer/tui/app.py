"""The Textual installer wizard application.

This module gains the real Textual implementation in a later task; the stub
keeps the final ``InstallerApp`` signature so the module CLI type-checks and
reports a clear error when the wizard is unavailable.
"""

from __future__ import annotations

from ..wizard import WizardState


class InstallerApp:
    def __init__(self, state: WizardState) -> None:
        self.state = state
        self.return_code = 130

    def run(self) -> None:
        raise NotImplementedError("the Textual wizard is not available in this environment")
