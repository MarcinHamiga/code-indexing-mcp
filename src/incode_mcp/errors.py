"""Stable application errors exposed through CLI and MCP adapters."""

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    PROJECT_NOT_FOUND = "PROJECT_NOT_FOUND"
    AMBIGUOUS_PROJECT = "AMBIGUOUS_PROJECT"
    PROJECT_ID_CONFLICT = "PROJECT_ID_CONFLICT"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    INDEX_INCOMPATIBLE = "INDEX_INCOMPATIBLE"
    INDEX_BUSY = "INDEX_BUSY"
    INDEX_RESOURCE_LIMIT = "INDEX_RESOURCE_LIMIT"
    INDEX_CANCELLED = "INDEX_CANCELLED"
    EMBEDDING_WORKER_FAILED = "EMBEDDING_WORKER_FAILED"
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    INVALID_FILTER = "INVALID_FILTER"


class IncodeError(RuntimeError):
    """An error with a stable machine-readable code."""

    def __init__(self, code: ErrorCode, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details

    def __str__(self) -> str:
        return f"{self.code}: {super().__str__()}"
