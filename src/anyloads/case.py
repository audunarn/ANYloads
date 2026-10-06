"""What a load needs to know about the load case it is evaluated for."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from .errors import LoadError
from .expressions import RESERVED_NAMES

__all__ = ["CaseContext", "check_parameter"]


def check_parameter(name: str, value: float) -> tuple[str, float]:
    """Validate a named number a load equation or code may read."""

    name = str(name)
    if not name.isidentifier():
        raise LoadError(f"parameter name {name!r} is not an identifier")
    if name in RESERVED_NAMES:
        raise LoadError(f"parameter name {name!r} is reserved; choose another")
    if isinstance(value, bool):
        raise LoadError(f"parameter {name!r} is a number, not a flag")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise LoadError(f"parameter {name!r} must be a finite number") from None
    if not math.isfinite(number):
        raise LoadError(f"parameter {name!r} must be a finite number")
    return name, number


@dataclass(frozen=True)
class CaseContext:
    """The identity of a load case, independent of any application's model.

    ``number`` is the 1-based load case number handed to pressure code and
    matched by a number-keyed pressure table; ``time`` is the case time
    (seconds, or whatever the engineer's time axis is) or ``None``; ``name`` is
    the case name; ``parameters`` are named numbers its equations and code may
    read.
    """

    number: int = 0
    name: str = ""
    time: float | None = None
    parameters: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "number", int(self.number))
        if self.time is not None:
            object.__setattr__(self, "time", float(self.time))
        object.__setattr__(
            self,
            "parameters",
            dict(check_parameter(key, value) for key, value in self.parameters.items()),
        )
