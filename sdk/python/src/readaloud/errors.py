from __future__ import annotations

class ReadAloudError(Exception):
    """Base class for all SDK errors."""


class ApiError(ReadAloudError):
    """Unexpected API error (bad request, server error, protocol error)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class AuthError(ApiError):
    """Invalid API key or token (HTTP 401)."""


class QuotaError(ApiError):
    """Free tier / quota exhausted (HTTP 402)."""


class CapacityError(ApiError):
    """Server at capacity (WS close 1013, HTTP 503). Retry shortly."""

    def __init__(self, message: str = "at capacity, retry shortly", status: int | None = None,
                 retry_after: float | None = None):
        super().__init__(message, status)
        self.retry_after = retry_after


class VoiceError(ApiError):
    """Unknown or unauthorized voice."""


def from_message(message: str) -> ApiError:
    """Map a server error message to a typed exception."""
    low = message.lower()
    if "capacity" in low:
        return CapacityError(message)
    if "unknown voice" in low:
        return VoiceError(message)
    return ApiError(message)


def from_status(status: int, message: str, retry_after: float | None = None) -> ApiError:
    if status == 401:
        return AuthError(message, status)
    if status == 402:
        return QuotaError(message, status)
    if status == 503:
        return CapacityError(message, status, retry_after)
    if status == 400 and "voice" in message.lower():
        return VoiceError(message, status)
    return ApiError(message, status)
