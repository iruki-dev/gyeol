"""Explicit success/failure for everything that can fail on bad audio.

Public functions that analyse audio return a :class:`Result` instead of a
silent default: ``result.ok`` tells whether ``value`` may be used and
``reason`` says why not.  ``UNRELIABLE`` means a value exists but its
validity conditions are not met (it must not be coached on).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar

T = TypeVar("T")


class Status(str, Enum):
    OK = "ok"
    UNRELIABLE = "unreliable"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"  # missing optional dependency / weights


@dataclass
class Result(Generic[T]):
    status: Status
    value: T | None = None
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status is Status.OK

    @property
    def usable(self) -> bool:
        """OK or UNRELIABLE-with-value (callers must propagate the flag)."""
        return self.value is not None and self.status in (Status.OK, Status.UNRELIABLE)

    def unwrap(self) -> T:
        if self.value is None or self.status in (Status.FAILED, Status.UNAVAILABLE):
            raise ResultError(f"{self.status.value}: {self.reason}")
        return self.value

    @classmethod
    def success(cls, value: T, warnings: list[str] | None = None) -> "Result[T]":
        return cls(Status.OK, value, "", warnings or [])

    @classmethod
    def unreliable(cls, value: T, reason: str) -> "Result[T]":
        return cls(Status.UNRELIABLE, value, reason)

    @classmethod
    def failure(cls, reason: str) -> "Result[T]":
        return cls(Status.FAILED, None, reason)

    @classmethod
    def unavailable(cls, reason: str) -> "Result[T]":
        return cls(Status.UNAVAILABLE, None, reason)


class ResultError(RuntimeError):
    pass
