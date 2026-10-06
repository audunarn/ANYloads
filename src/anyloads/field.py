"""A pressure field: one definition, evaluated at points for a load case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np

from .case import CaseContext
from .errors import LoadError
from .expressions import check_expression, evaluate_expression
from .tables import PressureTable, PressureTableError, interpolate_table
from .usercode import check_code, run_pressure_code

__all__ = ["PressureField", "table_selector"]


def _finite(value, what: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise LoadError(f"{what} is a number, not a flag")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise LoadError(f"{what} must be a finite number") from None
    if not np.isfinite(number):
        raise LoadError(f"{what} must be a finite number")
    return number


def table_selector(case: CaseContext, table: PressureTable, table_case=None):
    """What identifies a case in a table: its number, time or name."""

    if table_case is not None:
        return table_case
    return {"number": case.number, "time": case.time, "name": case.name}.get(table.key)


@dataclass(frozen=True)
class PressureField:
    """A pressure as a function of position and load case.

    Exactly one source is given:

    ``value``
        a uniform pressure;
    ``expression``
        a formula of the point ``x, y, z`` (metres), the case time ``t`` and
        the case's named parameters, e.g. ``rho*g*max(0, draft - z)``
        (:mod:`anyloads.expressions`);
    ``code``
        Python defining ``def pressure(case, x, y, z): ...``, run only when
        trusted (:mod:`anyloads.usercode`); ``code_vectorized`` calls it once
        with arrays;
    ``table``
        a :class:`PressureTable` of pressures at points per case;
        ``table_case`` pins which identifier to read instead of deriving it
        from the case.

    The result is a magnitude; ``scale`` multiplies it (1000 for a table in
    kPa). Which way it acts is the consumer's choice (see :mod:`anyloads.fluid`).
    """

    value: float | None = None
    expression: str | None = None
    code: str | None = None
    code_vectorized: bool = False
    table: PressureTable | None = None
    table_case: float | int | str | None = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        given = [
            label
            for label, item in (
                ("value", self.value),
                ("expression", self.expression),
                ("code", self.code),
                ("table", self.table),
            )
            if item is not None
        ]
        if len(given) != 1:
            raise LoadError(
                "a pressure field needs exactly one of a value, an expression, "
                f"code or a table (got {', '.join(given) or 'none'})"
            )
        if self.value is not None:
            object.__setattr__(self, "value", _finite(self.value, "pressure"))
        if self.expression is not None:
            check_expression(self.expression)
        if self.code is not None:
            check_code(self.code)
            object.__setattr__(self, "code_vectorized", bool(self.code_vectorized))
        if self.table is not None and not isinstance(self.table, PressureTable):
            raise LoadError("table must be a PressureTable")
        if self.table_case is not None:
            if self.table is None:
                raise LoadError("a table case only applies to a table")
            if isinstance(self.table_case, (bool, np.bool_)):
                raise LoadError("a table case is a number or a name")
        object.__setattr__(self, "scale", _finite(self.scale, "pressure scale"))

    @property
    def kind(self) -> str:
        if self.value is not None:
            return "uniform"
        if self.expression is not None:
            return "expression"
        return "code" if self.code is not None else "table"

    def evaluate(self, case: CaseContext, points: np.ndarray) -> np.ndarray:
        """Pressure magnitude at each point (``(n, 3)`` metres), scaled.

        Raises :class:`~anyloads.LoadError` (a subclass such as
        :class:`~anyloads.UntrustedCodeError`) with a message naming the cause.
        """

        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if self.kind == "uniform":
            values = np.full(len(points), float(self.value))
        elif self.kind == "expression":
            variables: Dict[str, Any] = {
                "x": points[:, 0],
                "y": points[:, 1],
                "z": points[:, 2],
                **case.parameters,
            }
            if case.time is not None:
                variables["t"] = float(case.time)
            try:
                values = evaluate_expression(self.expression, variables, size=len(points))
            except LoadError as error:
                hint = (
                    "; the case has no time"
                    if case.time is None and "'t'" in str(error)
                    else ""
                )
                raise type(error)(f"{error}{hint}") from None
        elif self.kind == "code":
            values = run_pressure_code(
                self.code,
                case=case.number,
                name=case.name,
                time=case.time,
                parameters=case.parameters,
                centroids=points,
                vectorized=self.code_vectorized,
            )
        else:
            values = interpolate_table(
                self.table, table_selector(case, self.table, self.table_case), points
            )
        return self.scale * values

    def to_dict(self) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"scale": float(self.scale)}
        if self.value is not None:
            entry["value"] = float(self.value)
        elif self.expression is not None:
            entry["expression"] = self.expression
        elif self.code is not None:
            entry["code"] = self.code
            if self.code_vectorized:
                entry["code_vectorized"] = True
        else:
            entry["table"] = self.table.to_dict()
            if self.table_case is not None:
                entry["table_case"] = self.table_case
        return entry

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PressureField":
        """Rebuild a field; code in ``data`` is parsed, never run."""

        return cls(
            value=data.get("value"),
            expression=data.get("expression"),
            code=data.get("code"),
            code_vectorized=bool(data.get("code_vectorized", False)),
            table=None if data.get("table") is None else PressureTable.from_dict(data["table"]),
            table_case=data.get("table_case"),
            scale=data.get("scale", 1.0),
        )
