"""Stable application errors exposed through CLI and MCP adapters."""

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    PROJECT_NOT_FOUND = "PROJECT_NOT_FOUND"
    AMBIGUOUS_PROJECT = "AMBIGUOUS_PROJECT"
    PROJECT_ID_CONFLICT = "PROJECT_ID_CONFLICT"
    CHUNK_NOT_FOUND = "CHUNK_NOT_FOUND"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    INDEX_INCOMPATIBLE = "INDEX_INCOMPATIBLE"
    INDEX_BUSY = "INDEX_BUSY"
    INDEX_RESOURCE_LIMIT = "INDEX_RESOURCE_LIMIT"
    INDEX_CANCELLED = "INDEX_CANCELLED"
    EMBEDDING_WORKER_FAILED = "EMBEDDING_WORKER_FAILED"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    DAEMON_UNAVAILABLE = "DAEMON_UNAVAILABLE"
    PROTOCOL_ERROR = "PROTOCOL_ERROR"
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    INVALID_FILTER = "INVALID_FILTER"
    STALE_CURSOR = "STALE_CURSOR"
    AMBIGUOUS_SYMBOL = "AMBIGUOUS_SYMBOL"
    UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
    REFERENCE_INDEX_UNAVAILABLE = "REFERENCE_INDEX_UNAVAILABLE"
    INVALID_REFACTOR = "INVALID_REFACTOR"
    INVALID_CURSOR = "INVALID_CURSOR"
    UNSUPPORTED_RUNTIME = "UNSUPPORTED_RUNTIME"


class CodeIndexingError(RuntimeError):
    """An error with a stable machine-readable code."""

    def __init__(self, code: ErrorCode, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details

    def __str__(self) -> str:
        return f"{self.code}: {super().__str__()}"

    def for_client(self) -> str:
        """Render code, message, and details as one line for an MCP tool error.

        ``__str__`` deliberately omits details: it is embedded in ``IndexIssue``
        messages and in daemon frames that already carry ``details`` as a
        separate field, where appending them would duplicate the payload.
        """
        if not self.details:
            return str(self)
        rendered = "; ".join(f"{key}={value}" for key, value in self.details.items())
        return f"{self} [{rendered}]"
