"""Read a pressure table (CSV / text / Excel) and map it onto points.

A table is rows of ``(case identifier,) x, y, z, pressure``. A project keeps
only the file's path and SHA-256 (:class:`PressureTable`);
this module reads the file, refuses it if it has changed, and interpolates the
pressure of one load case onto element centroids.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .errors import LoadError

__all__ = [
    "PressureTable",
    "PressureTableError",
    "TableData",
    "file_sha256",
    "interpolate_table",
    "read_table",
]


class PressureTableError(LoadError):
    """A pressure table cannot be read or does not cover a load case."""


TABLE_KEYS = (None, "number", "time", "name")
TABLE_METHODS = ("nearest", "idw")


def _finite(value, what):
    if isinstance(value, bool):
        raise PressureTableError(f"{what} is a number, not a flag")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise PressureTableError(f"{what} must be a finite number") from None
    if not math.isfinite(number):
        raise PressureTableError(f"{what} must be a finite number")
    return number


@dataclass(frozen=True)
class PressureTable:
    """A file of pressures at points, per load case.

    Each row is ``(case identifier,) x, y, z, pressure``. CSV/TSV/text and Excel
    (``.xlsx``) are read; the file stays where it is and the record keeps its
    path and SHA-256, so a changed file is refused instead of silently giving
    different loads. Build one with :meth:`from_file`.

    ``key`` says what the case-identifier column holds and how a load case picks
    its rows: ``"number"`` (the load case number), ``"time"`` (the case time;
    with ``interpolate_time`` a time between two rows' times is interpolated
    linearly), ``"name"`` (the case name), or ``None`` for a single map used by
    every case. ``x, y, z`` are converted to metres with ``length_scale``.
    Pressure is mapped to each point at its position by the nearest table point
    (``"nearest"``) or inverse-distance weighting over ``neighbours`` points
    (``"idw"``); ``max_distance`` (metres) refuses a point with no table point
    that close.

    ``columns`` overrides column choice (``{"x": "X [mm]", "pressure": 4}``): a
    header name or a 0-based position for ``key``, ``x``, ``y``, ``z``,
    ``pressure``. Without a header row the order is ``[key,] x, y, z, pressure``.
    """

    path: str
    sha256: str
    key: str | None = None
    sheet: str | None = None
    columns: Mapping[str, "str | int"] = field(default_factory=dict)
    length_scale: float = 1.0
    method: str = "nearest"
    neighbours: int = 4
    max_distance: float | None = None
    interpolate_time: bool = False

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise PressureTableError("a pressure table needs a file path")
        if len(str(self.sha256)) != 64:
            raise PressureTableError("a pressure table needs the SHA-256 of its file")
        if self.key not in TABLE_KEYS:
            raise PressureTableError("table key must be one of number, time, name or None")
        if self.method not in TABLE_METHODS:
            raise PressureTableError(
                "table method must be one of " + ", ".join(TABLE_METHODS)
            )
        scale = _finite(self.length_scale, "table length scale")
        if scale <= 0.0:
            raise PressureTableError("table length scale must be positive")
        object.__setattr__(self, "length_scale", scale)
        if (
            isinstance(self.neighbours, bool)
            or int(self.neighbours) != self.neighbours
            or int(self.neighbours) < 1
        ):
            raise PressureTableError("table neighbours must be a positive integer")
        object.__setattr__(self, "neighbours", int(self.neighbours))
        if self.max_distance is not None:
            distance = _finite(self.max_distance, "table maximum distance")
            if distance <= 0.0:
                raise PressureTableError("table maximum distance must be positive")
            object.__setattr__(self, "max_distance", distance)
        columns = dict(self.columns)
        unknown = sorted(set(columns) - {"key", "x", "y", "z", "pressure"})
        if unknown:
            raise PressureTableError(
                f"table columns {unknown} are not key, x, y, z or pressure"
            )
        object.__setattr__(self, "columns", columns)
        if self.interpolate_time and self.key != "time":
            raise PressureTableError("time interpolation needs a time-keyed table")
        object.__setattr__(self, "interpolate_time", bool(self.interpolate_time))

    @classmethod
    def from_file(
        cls,
        path,
        *,
        key: str | None = None,
        sheet: str | None = None,
        columns: Mapping[str, "str | int"] | None = None,
        length_scale: float = 1.0,
        method: str = "nearest",
        neighbours: int = 4,
        max_distance: float | None = None,
        interpolate_time: bool = False,
    ) -> "PressureTable":
        """Record a file and check right now that it can be read."""

        resolved = str(Path(path).expanduser().resolve())
        table = cls(
            path=resolved,
            sha256=file_sha256(resolved),
            key=key,
            sheet=sheet,
            columns=dict(columns or {}),
            length_scale=length_scale,
            method=method,
            neighbours=neighbours,
            max_distance=max_distance,
            interpolate_time=interpolate_time,
        )
        read_table(table)
        return table

    def keys(self) -> list:
        """The distinct case identifiers in the file, in sorted order."""

        return read_table(self).distinct_keys()

    def to_dict(self) -> Dict[str, Any]:
        """Plain data for a project file; defaults are left out."""

        entry: Dict[str, Any] = {"path": self.path, "sha256": self.sha256}
        if self.key is not None:
            entry["key"] = self.key
        if self.sheet is not None:
            entry["sheet"] = self.sheet
        if self.columns:
            entry["columns"] = dict(sorted(self.columns.items()))
        if self.length_scale != 1.0:
            entry["length_scale"] = float(self.length_scale)
        if self.method != "nearest":
            entry["method"] = self.method
        if self.neighbours != 4:
            entry["neighbours"] = int(self.neighbours)
        if self.max_distance is not None:
            entry["max_distance"] = float(self.max_distance)
        if self.interpolate_time:
            entry["interpolate_time"] = True
        return entry

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PressureTable":
        return cls(
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            key=data.get("key"),
            sheet=data.get("sheet"),
            columns=dict(data.get("columns", {})),
            length_scale=data.get("length_scale", 1.0),
            method=str(data.get("method", "nearest")),
            neighbours=data.get("neighbours", 4),
            max_distance=data.get("max_distance"),
            interpolate_time=bool(data.get("interpolate_time", False)),
        )


_ALIASES: Dict[str, tuple[str, ...]] = {
    "key": (
        "case", "load_case", "loadcase", "case_id", "case_no", "case_number",
        "number", "no", "id", "step", "time", "t", "name", "case_name",
    ),
    "x": ("x", "x_m", "xcoord", "x_coord", "x_coordinate"),
    "y": ("y", "y_m", "ycoord", "y_coord", "y_coordinate"),
    "z": ("z", "z_m", "zcoord", "z_coord", "z_coordinate"),
    "pressure": (
        "pressure", "p", "pres", "press", "pressure_pa", "p_pa", "load",
        "value",
    ),
}


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as error:
        raise PressureTableError(f"cannot read pressure table {path}: {error}") from None
    return digest.hexdigest()


def _normal(text: Any) -> str:
    return "".join(
        character if character.isalnum() else "_"
        for character in str(text).strip().casefold()
    ).strip("_")


def _number(cell: Any) -> float | None:
    if cell is None or isinstance(cell, bool):
        return None
    if isinstance(cell, (int, float)):
        return float(cell)
    text = str(cell).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        try:
            return float(text.replace(",", "."))
        except ValueError:
            return None


def _rows(path: Path, sheet: str | None) -> List[List[Any]]:
    suffix = path.suffix.casefold()
    if suffix in {".xlsx", ".xlsm"}:
        try:
            from openpyxl import load_workbook
        except ImportError:
            raise PressureTableError(
                "reading Excel needs openpyxl (pip install openpyxl); or save "
                "the sheet as CSV"
            ) from None
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            if sheet is None:
                worksheet = workbook.worksheets[0]
            elif sheet in workbook.sheetnames:
                worksheet = workbook[sheet]
            else:
                raise PressureTableError(
                    f"{path.name} has no sheet {sheet!r}; sheets: "
                    f"{', '.join(workbook.sheetnames)}"
                )
            rows = [list(row) for row in worksheet.iter_rows(values_only=True)]
        finally:
            workbook.close()
    elif suffix in {".xls"}:
        raise PressureTableError(
            "the old .xls format is not read; save the sheet as .xlsx or CSV"
        )
    else:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
        lines = [
            line for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not lines:
            raise PressureTableError(f"{path.name} is empty")
        sample = "\n".join(lines[:20])
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t| ")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
        if delimiter == " ":
            rows = [line.split() for line in lines]
        else:
            rows = [
                [cell.strip() for cell in row]
                for row in csv.reader(lines, delimiter=delimiter)
            ]
    rows = [
        row for row in rows
        if any(cell is not None and str(cell).strip() != "" for cell in row)
    ]
    if not rows:
        raise PressureTableError(f"{path.name} has no data rows")
    return rows


@dataclass
class TableData:
    """The numbers in a table: case identifiers, points (metres), pressures."""

    keys: np.ndarray | None  # float for number/time, str for name; None = no key
    points: np.ndarray       # (n, 3) metres
    pressure: np.ndarray     # (n,)
    key_kind: str | None
    _trees: Dict[Any, Any] = field(default_factory=dict, repr=False)

    def distinct_keys(self) -> list:
        if self.keys is None:
            return []
        if self.key_kind == "name":
            return list(dict.fromkeys(self.keys.tolist()))
        return sorted(set(float(item) for item in self.keys))


def _resolve_column(header: Sequence[str] | None, which: str, spec) -> int | None:
    wanted = spec.columns.get(which)
    if wanted is not None:
        if isinstance(wanted, int) and not isinstance(wanted, bool):
            return int(wanted)
        if header is None:
            raise PressureTableError(
                f"column {wanted!r} named for {which}, but the file has no "
                "header row; give a 0-based position"
            )
        target = _normal(wanted)
        for index, name in enumerate(header):
            if _normal(name) == target:
                return index
        raise PressureTableError(
            f"column {wanted!r} (for {which}) not found; columns: "
            f"{', '.join(str(item) for item in header)}"
        )
    if header is None:
        return None
    for alias in _ALIASES[which]:
        for index, name in enumerate(header):
            if _normal(name) == alias:
                return index
    return None


_CACHE: "OrderedDict[tuple, TableData]" = OrderedDict()
_CACHE_SIZE = 4


def read_table(spec) -> TableData:
    """Read and validate the table of a ``PressureTable``; cached per file."""

    path = Path(spec.path).expanduser()
    try:
        status = path.stat()
    except OSError:
        raise PressureTableError(f"pressure table {spec.path} does not exist") from None
    cache_key = (
        str(path), status.st_mtime_ns, status.st_size, spec.sha256,
        spec.sheet, spec.key, tuple(sorted((k, str(v)) for k, v in spec.columns.items())),
        spec.length_scale,
    )
    cached = _CACHE.get(cache_key)
    if cached is not None:
        _CACHE.move_to_end(cache_key)
        return cached
    if file_sha256(path) != spec.sha256:
        raise PressureTableError(
            f"pressure table {path.name} has changed since it was added (its "
            "SHA-256 differs); import it again to accept the new contents"
        )

    rows = _rows(path, spec.sheet)
    first = rows[0]
    numeric_first = [_number(cell) is not None for cell in first]
    has_key = spec.key is not None
    order = (
        ["key", "x", "y", "z", "pressure"] if has_key else ["x", "y", "z", "pressure"]
    )
    # A header row is one whose coordinate and pressure cells are not all
    # numbers; without one the order is [key,] x, y, z, pressure.
    numeric_positions = []
    for which in ("x", "y", "z", "pressure"):
        chosen = spec.columns.get(which)
        if chosen is None:
            numeric_positions.append(order.index(which))
        elif isinstance(chosen, int) and not isinstance(chosen, bool):
            numeric_positions.append(int(chosen))
    headerless = bool(numeric_positions) and all(
        position < len(first) and numeric_first[position]
        for position in numeric_positions
    )
    header: List[str] | None
    if headerless:
        header = None
    else:
        header = [str(cell) if cell is not None else "" for cell in first]
        rows = rows[1:]
        if not rows:
            raise PressureTableError(f"{path.name} has a header but no data")
    columns: Dict[str, int | None] = {}
    for which in ("key", "x", "y", "z", "pressure"):
        columns[which] = _resolve_column(header, which, spec)
    if header is None:
        if len(first) < len(order):
            raise PressureTableError(
                f"{path.name} has {len(first)} columns; without a header it "
                f"needs {len(order)}: {', '.join(order)}"
            )
        for position, which in enumerate(order):
            if columns[which] is None:
                columns[which] = position
    needed = ["x", "y", "z", "pressure"] + (["key"] if has_key else [])
    missing = [which for which in needed if columns[which] is None]
    if missing:
        raise PressureTableError(
            f"{path.name}: cannot find column(s) {', '.join(missing)}; columns: "
            f"{', '.join(header or [])}. Name them with columns={{...}}"
        )

    count = len(rows)
    points = np.empty((count, 3))
    pressure = np.empty(count)
    keys: List[Any] = []
    first_row = 2 if header is not None else 1
    for index, row in enumerate(rows):
        try:
            values = [
                _number(row[columns[which]]) for which in ("x", "y", "z", "pressure")
            ]
        except IndexError:
            raise PressureTableError(
                f"{path.name}, row {index + first_row}: too few columns"
            ) from None
        if any(value is None or not math.isfinite(value) for value in values):
            raise PressureTableError(
                f"{path.name}, row {index + first_row}: x, y, z and pressure "
                "must be finite numbers"
            )
        points[index] = values[:3]
        pressure[index] = values[3]
        if has_key:
            cell = row[columns["key"]] if columns["key"] < len(row) else None
            if spec.key == "name":
                if cell is None or str(cell).strip() == "":
                    raise PressureTableError(
                        f"{path.name}, row {index + first_row}: empty case name"
                    )
                keys.append(str(cell).strip())
            else:
                number = _number(cell)
                if number is None or not math.isfinite(number):
                    raise PressureTableError(
                        f"{path.name}, row {index + first_row}: the case "
                        f"identifier {cell!r} is not a number"
                    )
                keys.append(number)
    data = TableData(
        keys=(None if not has_key else np.array(keys, dtype=object if spec.key == "name" else float)),
        points=points * float(spec.length_scale),
        pressure=pressure,
        key_kind=spec.key,
    )
    _CACHE[cache_key] = data
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return data


def _spatial(
    data: TableData,
    token,
    rows: np.ndarray,
    centroids: np.ndarray,
    spec,
    describe: str,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    points = data.points[rows]
    values = data.pressure[rows]
    tree = data._trees.get(token)
    if tree is None:
        tree = cKDTree(points)
        data._trees[token] = tree
    k = 1 if spec.method == "nearest" else min(int(spec.neighbours), len(points))
    distance, index = tree.query(centroids, k=k)
    distance = np.asarray(distance, dtype=float).reshape(len(centroids), -1)
    index = np.asarray(index, dtype=int).reshape(len(centroids), -1)
    if spec.max_distance is not None:
        far = distance[:, 0] > float(spec.max_distance)
        if np.any(far):
            raise PressureTableError(
                f"{int(far.sum())} wet element(s) have no table point within "
                f"{spec.max_distance:g} m for {describe} (farthest: "
                f"{float(distance[:, 0].max()):g} m)"
            )
    if k == 1:
        return values[index[:, 0]]
    # Inverse-distance weighting; a centroid on a table point takes its value.
    with np.errstate(divide="ignore"):
        weights = 1.0 / np.maximum(distance, 1e-12) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("ij,ij->i", weights, values[index])


def interpolate_table(
    spec, selector, centroids: np.ndarray
) -> np.ndarray:
    """Pressure of the rows picked by ``selector`` at ``centroids`` (n, 3).

    ``selector`` is the case number, time or name according to ``spec.key``
    (ignored when the table has no key column).
    """

    data = read_table(spec)
    centroids = np.asarray(centroids, dtype=float).reshape(-1, 3)
    if spec.key is None:
        rows = np.arange(len(data.pressure))
        return _spatial(data, ("all",), rows, centroids, spec, "the table")

    if selector is None:
        raise PressureTableError(
            f"the table is keyed by {spec.key}, but the load case has no "
            f"{spec.key} to look up"
        )
    if spec.key == "name":
        wanted = str(selector).strip()
        rows = np.flatnonzero(data.keys == wanted)
        if not len(rows):
            raise PressureTableError(
                f"the table has no rows for case {wanted!r}; cases: "
                f"{', '.join(map(str, data.distinct_keys()))}"
            )
        return _spatial(data, ("name", wanted), rows, centroids, spec, f"case {wanted!r}")

    try:
        value = float(selector)
    except (TypeError, ValueError):
        raise PressureTableError(
            f"the table is keyed by {spec.key}, but {selector!r} is not a number"
        ) from None
    tolerance = 1e-9 * max(1.0, abs(value))
    exact = np.flatnonzero(np.abs(data.keys.astype(float) - value) <= tolerance)
    label = f"{spec.key} {value:g}"
    if len(exact):
        return _spatial(data, ("n", float(data.keys[exact[0]])), exact, centroids, spec, label)
    available = data.distinct_keys()
    if spec.interpolate_time:
        if value < available[0] or value > available[-1]:
            raise PressureTableError(
                f"time {value:g} is outside the table's times "
                f"{available[0]:g} to {available[-1]:g}"
            )
        upper = next(item for item in available if item > value)
        lower = max(item for item in available if item < value)
        weight = (value - lower) / (upper - lower)
        low_rows = np.flatnonzero(data.keys.astype(float) == lower)
        high_rows = np.flatnonzero(data.keys.astype(float) == upper)
        low = _spatial(data, ("n", lower), low_rows, centroids, spec, f"time {lower:g}")
        high = _spatial(data, ("n", upper), high_rows, centroids, spec, f"time {upper:g}")
        return (1.0 - weight) * low + weight * high
    shown = ", ".join(f"{item:g}" for item in available[:8])
    raise PressureTableError(
        f"the table has no rows for {label}; {spec.key}s: {shown}"
        + (" ..." if len(available) > 8 else "")
        + (". Set interpolate_time to interpolate between times."
           if spec.key == "time" else "")
    )
