"""API error type rendered as a JSON response."""

from __future__ import annotations


class ApiError(Exception):
    """Exception rendered as a JSON error response.

    Args:
        status: HTTP status code.
        code: Short machine-readable error code (``invalid_request``...).
        message: Human-readable explanation.
    """

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
