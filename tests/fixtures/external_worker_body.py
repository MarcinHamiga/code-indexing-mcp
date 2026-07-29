"""A worker body an external launcher test can run without a model.

Lives under fixtures because it is loaded by an interpreter the test starts, not
by the test process: the launcher hands the child a ``module:function``
reference and the child imports it for itself.
"""

from __future__ import annotations

from typing import Any


def echo(connection: Any, config: Any) -> None:
    """Answer the session protocol with facts drawn from the config it got."""
    try:
        while True:
            command, payload = connection.recv()
            if command == "stop":
                return
            if command == "initialize":
                connection.send(("initialized", (tuple(config.providers), config.dimension)))
                continue
            if command == "memory":
                connection.send(("memory", 1))
                continue
            if command == "identity":
                connection.send(("identity", (config.model_id, config.accelerator)))
                continue
            connection.send(("echo", payload))
    finally:
        connection.close()

