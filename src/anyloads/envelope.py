"""When a load acts and how strongly: a window and a factor over time.

A load in a long time history should be *one* load that says when it acts, not a
copy in every load case.  An :class:`Envelope` is that statement.  It multiplies
the load by a factor that is zero outside a window and, inside it, comes from an
equation of the case time or from a piecewise-linear table:

* the **window** is a start and a stop, both inclusive, counted either in load
  case numbers (``by="number"``: acts from case 20 to case 300) or in case time
  (``by="time"``: acts from 2.0 s to 30.0 s).  Either end may be left open.
* the **factor** is 1 inside the window by default; ``expression`` gives it as a
  formula (``0.3 + 0.4 * sin(omega * t - 0.8)``) and ``points`` as
  ``(time, factor)`` pairs interpolated linearly and held at the ends (a ramp
  in, a plateau, a ramp out).

An expression may read the case time ``t``, the case ``number``, the steps since
the window opened ``k``, the seconds since it opened ``tau`` (when known), the
case's named parameters and ``pi``, ``e`` and ``g``, and the same whitelisted
functions as any other load equation.  Nothing is executed beyond that.

Several loads each carry their own envelope and simply add, so a wave load that
runs the whole record, a ramped load from case 20 and a pulse between 12 s and
14 s combine without any of them knowing about the others.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .case import CaseContext
from .errors import LoadError
from .expressions import (
    BUILTIN_CONSTANTS,
    check_expression,
    evaluate_expression,
    expression_names,
)

__all__ = ["ENVELOPE_BASES", "ENVELOPE_NAMES", "Envelope"]

ENVELOPE_BASES: tuple[str, ...] = ("number", "time")
#: Names an envelope expression may read besides parameters and constants.
ENVELOPE_NAMES: tuple[str, ...] = ("t", "number", "k", "tau")


def _number(value: Any, what: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise LoadError(f"{what} is a number, not a flag")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise LoadError(f"{what} must be a number") from None
    if not np.isfinite(result):
        raise LoadError(f"{what} must be finite")
    return result


@dataclass(frozen=True)
class Envelope:
    """A window and a factor for one load; see the module documentation."""

    start: float | None = None
    stop: float | None = None
    by: str = "number"
    expression: str | None = None
    points: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.by not in ENVELOPE_BASES:
            raise LoadError(
                f"an envelope window is by {' or '.join(ENVELOPE_BASES)}, not {self.by!r}"
            )
        start = _number(self.start, "envelope start")
        stop = _number(self.stop, "envelope stop")
        if start is not None and stop is not None and stop < start:
            raise LoadError(
                f"envelope stop ({stop:g}) is before its start ({start:g})"
            )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        if self.expression is not None and self.points is not None:
            raise LoadError("an envelope has an expression or points, not both")
        if self.expression is not None:
            check_expression(self.expression)
        if self.points is not None:
            try:
                pairs = tuple((float(a), float(b)) for a, b in self.points)
            except (TypeError, ValueError):
                raise LoadError("envelope points are (time, factor) pairs") from None
            if len(pairs) < 2:
                raise LoadError("envelope points need at least two (time, factor) pairs")
            if not np.all(np.isfinite(pairs)):
                raise LoadError("envelope points must be finite")
            if any(b[0] <= a[0] for a, b in zip(pairs, pairs[1:])):
                raise LoadError("envelope point times must increase")
            object.__setattr__(self, "points", pairs)

    # ------------------------------------------------------------------
    @property
    def uses_time(self) -> bool:
        """Whether evaluating it needs the case time."""

        if self.by == "time" and (self.start is not None or self.stop is not None):
            return True
        if self.points is not None:
            return True
        return self.expression is not None and bool(
            {"t", "tau"} & expression_names(self.expression)
        )

    def names(self) -> frozenset[str]:
        """Parameter names its expression reads (not variables or constants)."""

        if self.expression is None:
            return frozenset()
        return frozenset(
            name
            for name in expression_names(self.expression)
            if name not in ENVELOPE_NAMES and name not in BUILTIN_CONSTANTS
        )

    def active(self, case: CaseContext) -> bool:
        """Whether ``case`` lies inside the window (both ends inclusive)."""

        if self.start is None and self.stop is None:
            return True
        if self.by == "number":
            value = float(case.number)
        else:
            if case.time is None:
                raise LoadError(
                    "an envelope window by time needs a case with a time; set one, "
                    "or generate the cases as a series"
                )
            value = float(case.time)
        # A hair of slack so 0.3 s written as 0.30000000000000004 still counts.
        slack = 1.0e-9 * max(1.0, abs(value))
        if self.start is not None and value < self.start - slack:
            return False
        if self.stop is not None and value > self.stop + slack:
            return False
        return True

    def factor(self, case: CaseContext, *, start_time: float | None = None) -> float:
        """The multiplier at ``case``: 0 outside the window, else its factor.

        ``start_time`` is the case time at which a window by number opens, when
        the caller knows it; it gives the expression ``tau``.  For a window by
        time it is the window's start.
        """

        if not self.active(case):
            return 0.0
        if self.expression is None and self.points is None:
            return 1.0
        if self.points is not None:
            if case.time is None:
                raise LoadError(
                    "this envelope varies with time but the load case has none; "
                    "set one, or generate the cases as a series"
                )
            times, factors = zip(*self.points)
            return float(np.interp(float(case.time), times, factors))
        time = None if case.time is None else float(case.time)
        opened = self.start if self.by == "time" else start_time
        first = self.start if self.by == "number" and self.start is not None else None
        # The envelope's own variables win over a parameter of the same name.
        variables: Dict[str, float] = {
            **case.parameters,
            "number": float(case.number),
            "k": 0.0 if first is None else float(case.number) - first,
        }
        if time is not None:
            variables["t"] = time
            if opened is not None:
                variables["tau"] = time - float(opened)
            elif self.start is None:
                variables["tau"] = time
        try:
            return float(evaluate_expression(self.expression, variables))
        except LoadError as error:
            hint = (
                "; 'tau' needs the window's start time (a window by time, or a "
                "case series)"
                if "'tau'" in str(error)
                else "; the load case has no time"
                if "'t'" in str(error) and time is None
                else ""
            )
            raise type(error)(f"{error}{hint}") from None

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"by": self.by}
        if self.start is not None:
            entry["start"] = self.start
        if self.stop is not None:
            entry["stop"] = self.stop
        if self.expression is not None:
            entry["expression"] = self.expression
        if self.points is not None:
            entry["points"] = [list(pair) for pair in self.points]
        return entry

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Envelope":
        if not isinstance(data, Mapping):
            raise LoadError("an envelope is an object with start, stop, by, expression, points")
        unknown = set(data) - {"start", "stop", "by", "expression", "points"}
        if unknown:
            raise LoadError(f"unknown envelope field(s) {sorted(unknown)}")
        points = data.get("points")
        return cls(
            start=data.get("start"),
            stop=data.get("stop"),
            by=str(data.get("by", "number")),
            expression=data.get("expression"),
            points=None if points is None else tuple(tuple(pair) for pair in points),
        )

    @classmethod
    def coerce(cls, value: "Envelope | Mapping[str, Any] | None") -> "Envelope | None":
        """An envelope from an envelope, a dict or ``None``."""

        if value is None or isinstance(value, Envelope):
            return value
        return cls.from_dict(value)

