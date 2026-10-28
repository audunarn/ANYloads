"""Immutable transient (time-domain) floating-body load history record.

Unlike the spectral ``FloatingCoupledLoadHistory`` this record carries ACTUAL accepted times of a
fixed-step transient run, never a harmonic encoding.  It holds, for exactly the selected frames:
physical component nodal load arrays (each with explicit per-component shape, unit, frame and
attachment identities), the primary six-DOF acceleration, the physical rigid q / v / a, the wind /
relative-speed / thrust histories, and the body / mesh / mass / wind / curve / hydro content
identities.  Frame selection is EXACT (index subsets of the accepted trajectory, no interpolation of
prepared nonlinear loads) and a selected window never resets the trajectory: the source trajectory
identity and the original frame indices are retained.

Only a record whose status is ``"complete"`` may be used as accepted structural output
(:meth:`require_accepted`); non-success records are diagnostic only.  Artifacts are versioned
``.npz`` files (no pickle) with a SHA-256 content hash; allocation and storage are bounded before
large copies are made.  Nothing here claims numerical qualification.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "TRANSIENT_HISTORY_FORMAT",
    "REQUIRED_IDENTITIES",
    "LoadComponent",
    "TransientFloatingLoadHistory",
    "TransientHistoryError",
    "TransientHistoryLimits",
    "load_transient_history",
]

TRANSIENT_HISTORY_FORMAT = "anyloads-transient-floating-load-history-v1"
REQUIRED_IDENTITIES = ("body", "mesh", "mass", "wind", "curve", "hydro")
STATUS_COMPLETE = "complete"
DEFAULT_MAX_BYTES = 2 * 1024**3


class TransientHistoryError(ValueError):
    """A transient history cannot be admitted, selected, saved or loaded."""


@dataclass(frozen=True)
class TransientHistoryLimits:
    """Storage ceiling (bytes) checked before any large copy or file read; may only tighten the default."""

    max_bytes: int = DEFAULT_MAX_BYTES

    def __post_init__(self) -> None:
        if (isinstance(self.max_bytes, bool) or not isinstance(self.max_bytes, (int, np.integer))
                or not 0 < int(self.max_bytes) <= DEFAULT_MAX_BYTES):
            raise TransientHistoryError(f"max_bytes must be an integer in (0, {DEFAULT_MAX_BYTES}]")


def _real_only(value: Any, name: str, dtype=np.float64) -> np.ndarray:
    """Owned copy with the requested REAL dtype; complex input (even zero imaginary part) is refused before the
    cast, so a frequency-domain phasor can never be stored as a real frame sequence."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TransientHistoryError(f"{name}: not a valid array ({exc})") from exc
    if raw.dtype.kind == "c":
        raise TransientHistoryError(f"{name}: complex-valued input is refused (real time-history data required)")
    try:
        return np.array(raw, dtype=dtype, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise TransientHistoryError(f"{name}: not a valid array ({exc})") from exc


def _owned(value: Any, name: str, *, shape: Optional[tuple] = None, ndim: Optional[int] = None,
           dtype=np.float64, finite: bool = True) -> np.ndarray:
    arr = _real_only(value, name, dtype)
    if shape is not None and arr.shape != shape:
        raise TransientHistoryError(f"{name}: expected shape {shape}, got {arr.shape}")
    if ndim is not None and arr.ndim != ndim:
        raise TransientHistoryError(f"{name}: expected {ndim}-D array, got shape {arr.shape}")
    if finite and arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
        raise TransientHistoryError(f"{name}: non-finite values are not admitted")
    if arr.size == 0:
        arr.flags.writeable = False
        return arr
    return np.frombuffer(arr.tobytes(), dtype=arr.dtype).reshape(arr.shape)


def _check_bytes(arrays: Sequence[Any], limits: TransientHistoryLimits, what: str) -> None:
    total = 0
    for a in arrays:
        total += int(np.asarray(a).nbytes) if not hasattr(a, "nbytes") else int(a.nbytes)
    if total > limits.max_bytes:
        raise TransientHistoryError(f"{what} needs {total} B > storage ceiling {limits.max_bytes} B")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TransientHistoryError(f"unsupported metadata value of type {type(value).__name__}")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class LoadComponent:
    """One physical load component: ``values[frame, item, k]`` for ``item`` identified by ``item_ids``.

    ``kind`` names the content (``nodal_force`` k=3, ``nodal_wrench`` k=6, ``pressure`` k=1, ...);
    the per-component shape, unit, frame and attachment are explicit and validated.
    """

    name: str
    kind: str
    unit: str
    frame: str
    attachment: str
    item_ids: Tuple[str, ...]
    values: np.ndarray

    _K = {"nodal_force": 3, "nodal_wrench": 6, "pressure": 1}

    def __post_init__(self) -> None:
        for fname in ("name", "kind", "unit", "attachment"):
            if not isinstance(getattr(self, fname), str) or not getattr(self, fname):
                raise TransientHistoryError(f"component {fname}: non-empty string required")
        if self.frame not in ("world", "body"):
            raise TransientHistoryError("component frame must be 'world' or 'body'")
        ids = tuple(str(i) for i in self.item_ids)
        if len(set(ids)) != len(ids) or not ids:
            raise TransientHistoryError(f"component {self.name!r}: unique non-empty item_ids required")
        vals = _owned(self.values, f"component {self.name!r} values", ndim=3)
        if vals.shape[1] != len(ids):
            raise TransientHistoryError(f"component {self.name!r}: values shape {vals.shape} vs {len(ids)} items")
        k = self._K.get(self.kind)
        if k is not None and vals.shape[2] != k:
            raise TransientHistoryError(f"component {self.name!r}: kind {self.kind!r} needs {k} values per item")
        object.__setattr__(self, "item_ids", ids)
        object.__setattr__(self, "values", vals)

    def select(self, indices: np.ndarray) -> "LoadComponent":
        return LoadComponent(self.name, self.kind, self.unit, self.frame, self.attachment, self.item_ids,
                             self.values[indices])

    def meta(self) -> dict:
        return dict(name=self.name, kind=self.kind, unit=self.unit, frame=self.frame, attachment=self.attachment,
                    item_ids=list(self.item_ids), shape=list(self.values.shape))


@dataclass(frozen=True)
class TransientFloatingLoadHistory:
    """Actual accepted times with components, motion, wind / thrust and content identities."""

    times: np.ndarray
    frame_indices: np.ndarray
    source_frame_count: int
    source_times_hash: str
    components: Tuple[LoadComponent, ...]
    primary_acceleration: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    wind_velocity: np.ndarray
    relative_axial_speed: np.ndarray
    thrust: np.ndarray
    identities: Mapping[str, str]
    status: str = STATUS_COMPLETE
    status_message: str = ""
    window: Mapping[str, Any] = field(default_factory=dict)
    limits: TransientHistoryLimits = field(default_factory=TransientHistoryLimits)

    def __post_init__(self) -> None:
        if not isinstance(self.limits, TransientHistoryLimits):
            raise TransientHistoryError("limits must be a TransientHistoryLimits")
        # bound the allocation BEFORE any large copy
        raw = [self.times, self.primary_acceleration, self.displacement, self.velocity, self.acceleration,
               self.wind_velocity, self.relative_axial_speed, self.thrust, self.frame_indices]
        raw += [c.values for c in self.components if isinstance(c, LoadComponent)]
        _check_bytes(raw, self.limits, "transient history")
        t = _owned(self.times, "times", ndim=1)
        n = t.size
        if n < 1:
            raise TransientHistoryError("at least one accepted frame is required")
        if n > 1 and np.any(np.diff(t) <= 0.0):
            raise TransientHistoryError("times: strictly increasing actual accepted seconds required")
        raw_idx = np.asarray(self.frame_indices)
        if raw_idx.dtype.kind not in "iu":       # bool, float (even integral) and object indices are refused before int64
            raise TransientHistoryError(
                f"frame_indices: integer dtype required (got {raw_idx.dtype}); bool / fractional indices are refused")
        idx = _owned(raw_idx, "frame_indices", shape=(n,), dtype=np.int64)
        if isinstance(self.source_frame_count, bool) or int(self.source_frame_count) != self.source_frame_count:
            raise TransientHistoryError("source_frame_count must be an integer")
        total = int(self.source_frame_count)
        if idx.min() < 0 or idx.max() >= total or (n > 1 and np.any(np.diff(idx) <= 0)):
            raise TransientHistoryError("frame_indices must be strictly increasing and inside the source trajectory")
        h = self.source_times_hash
        if not (isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h)):
            raise TransientHistoryError("source_times_hash must be a sha256 hex digest of the full accepted times")
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "frame_indices", idx)
        object.__setattr__(self, "source_frame_count", total)
        for name, shape in (("primary_acceleration", (n, 6)), ("displacement", (n, 6)), ("velocity", (n, 6)),
                            ("acceleration", (n, 6)), ("wind_velocity", (n, 3)),
                            ("relative_axial_speed", (n,)), ("thrust", (n,))):
            object.__setattr__(self, name, _owned(getattr(self, name), name, shape=shape))
        comps = tuple(self.components)
        if not comps or not all(isinstance(c, LoadComponent) for c in comps):
            raise TransientHistoryError("at least one LoadComponent is required")
        names = [c.name for c in comps]
        if len(set(names)) != len(names):
            raise TransientHistoryError("component names must be unique")
        for c in comps:
            if c.values.shape[0] != n:
                raise TransientHistoryError(f"component {c.name!r} has {c.values.shape[0]} frames, expected {n}")
        object.__setattr__(self, "components", comps)
        ident = {str(k): v for k, v in dict(self.identities).items()}
        missing = [k for k in REQUIRED_IDENTITIES if not isinstance(ident.get(k), str) or not ident.get(k)]
        if missing:
            raise TransientHistoryError(f"identities missing / not non-empty strings: {missing}")
        object.__setattr__(self, "identities", _freeze(ident))
        if not isinstance(self.status, str) or not self.status:
            raise TransientHistoryError("explicit status string required")
        object.__setattr__(self, "window", _freeze(dict(self.window)))

    # ------------------------------------------------------------------ status
    @property
    def n_frames(self) -> int:
        return int(self.times.size)

    @property
    def accepted_for_structural_output(self) -> bool:
        return self.status == STATUS_COMPLETE

    @property
    def qualification_status(self) -> str:
        return "unqualified"

    def require_accepted(self) -> "TransientFloatingLoadHistory":
        if not self.accepted_for_structural_output:
            raise TransientHistoryError(
                f"status {self.status!r} ({self.status_message}) is not accepted structural output; "
                "diagnostic history only")
        return self

    def component(self, name: str) -> LoadComponent:
        for c in self.components:
            if c.name == name:
                return c
        raise TransientHistoryError(f"unknown component {name!r}")

    # ------------------------------------------------------------------ construction from a solver result
    @classmethod
    def from_solver_result(cls, result: Any, components: Sequence[LoadComponent], identities: Mapping[str, str],
                           primary_acceleration: Any = None, limits: Optional[TransientHistoryLimits] = None
                           ) -> "TransientFloatingLoadHistory":
        """All accepted frames of a ``FloatingTransientResult``-like object (duck-typed, no solver import)."""
        times = _real_only(result.times, "times")
        n = times.size
        acc = result.accelerations if primary_acceleration is None else primary_acceleration
        return cls(
            times=times, frame_indices=np.arange(n, dtype=np.int64), source_frame_count=n,
            source_times_hash=hash_times(times), components=tuple(components),
            primary_acceleration=acc, displacement=result.displacements, velocity=result.velocities,
            acceleration=result.accelerations, wind_velocity=result.hub_wind_world,
            relative_axial_speed=result.relative_axial_speed, thrust=result.thrust, identities=identities,
            status=str(result.status), status_message=str(getattr(result, "message", "")),
            window={"kind": "full", "origin_time": float(times[0])},
            limits=limits or TransientHistoryLimits())

    # ------------------------------------------------------------------ exact selection
    def select_frames(self, positions: Any, *, label: str = "frames") -> "TransientFloatingLoadHistory":
        """Exact subset of THIS record's frames (no interpolation); the trajectory origin is retained."""
        pos = np.asarray(positions)
        if pos.ndim != 1 or pos.size < 1 or pos.dtype.kind not in "iu":
            raise TransientHistoryError("positions: a non-empty 1-D integer array is required")
        if pos.min() < 0 or pos.max() >= self.n_frames or np.any(np.diff(pos) <= 0):
            raise TransientHistoryError("positions must be strictly increasing and inside the record")
        w = dict(self.window)
        w.update(kind=label, selected_from_frames=self.n_frames,
                 origin_time=float(w.get("origin_time", self.times[0])))
        return TransientFloatingLoadHistory(
            times=self.times[pos], frame_indices=self.frame_indices[pos], source_frame_count=self.source_frame_count,
            source_times_hash=self.source_times_hash, components=tuple(c.select(pos) for c in self.components),
            primary_acceleration=self.primary_acceleration[pos], displacement=self.displacement[pos],
            velocity=self.velocity[pos], acceleration=self.acceleration[pos], wind_velocity=self.wind_velocity[pos],
            relative_axial_speed=self.relative_axial_speed[pos], thrust=self.thrust[pos], identities=self.identities,
            status=self.status, status_message=self.status_message, window=w, limits=self.limits)

    def select_window(self, t_start: float, t_stop: float) -> "TransientFloatingLoadHistory":
        """Frames with ``t_start <= t <= t_stop`` (exact stored times); the trajectory is not reset."""
        a, b = float(_real_only(t_start, "t_start")), float(_real_only(t_stop, "t_stop"))
        if not (np.isfinite(a) and np.isfinite(b) and a <= b):
            raise TransientHistoryError("a finite ordered window is required")
        pos = np.flatnonzero((self.times >= a) & (self.times <= b))
        if pos.size == 0:
            raise TransientHistoryError("the window contains no accepted frame")
        out = self.select_frames(pos, label="window")
        w = dict(out.window)
        w.update(t_start=a, t_stop=b)
        return TransientFloatingLoadHistory(**{**_fields(out), "window": w})

    # ------------------------------------------------------------------ integrity
    def _meta(self) -> dict:
        return dict(format=TRANSIENT_HISTORY_FORMAT, source_frame_count=self.source_frame_count,
                    source_times_hash=self.source_times_hash, identities=self.identities, status=self.status,
                    status_message=self.status_message, window=self.window,
                    components=[c.meta() for c in self.components], qualification_status="unqualified")

    def _arrays(self) -> dict:
        out = dict(times=self.times, frame_indices=self.frame_indices, primary_acceleration=self.primary_acceleration,
                   displacement=self.displacement, velocity=self.velocity, acceleration=self.acceleration,
                   wind_velocity=self.wind_velocity, relative_axial_speed=self.relative_axial_speed, thrust=self.thrust)
        for i, c in enumerate(self.components):
            out[f"component_{i}"] = c.values
        return out

    def content_hash(self) -> str:
        h = hashlib.sha256()
        meta = _canonical(self._meta())
        h.update(len(meta).to_bytes(8, "little"))
        h.update(meta)
        for name, arr in self._arrays().items():
            data = np.ascontiguousarray(arr, dtype="<i8" if arr.dtype.kind == "i" else "<f8").tobytes()
            h.update(name.encode())
            h.update(len(data).to_bytes(8, "little"))
            h.update(data)
        return h.hexdigest()

    @property
    def stored_bytes(self) -> int:
        return int(sum(a.nbytes for a in self._arrays().values()))

    def save(self, path: str) -> str:
        """Atomic versioned ``.npz`` (no pickle); refuses above the storage ceiling before writing."""
        if 2 * self.stored_bytes > self.limits.max_bytes:
            raise TransientHistoryError(
                f"artifact needs {self.stored_bytes} B (2x on reload) > ceiling {self.limits.max_bytes} B")
        digest = self.content_hash()
        header = _canonical({"meta": self._meta(), "content_hash": digest})
        payload = dict(self._arrays())
        payload["header"] = np.frombuffer(header, dtype=np.uint8)
        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez(handle, **payload)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return digest


def _fields(h: TransientFloatingLoadHistory) -> dict:
    return dict(times=h.times, frame_indices=h.frame_indices, source_frame_count=h.source_frame_count,
                source_times_hash=h.source_times_hash, components=h.components,
                primary_acceleration=h.primary_acceleration, displacement=h.displacement, velocity=h.velocity,
                acceleration=h.acceleration, wind_velocity=h.wind_velocity,
                relative_axial_speed=h.relative_axial_speed, thrust=h.thrust, identities=h.identities,
                status=h.status, status_message=h.status_message, window=h.window, limits=h.limits)


def hash_times(times: Any) -> str:
    """SHA-256 of the full accepted time vector (identity of the source trajectory)."""
    return hashlib.sha256(np.ascontiguousarray(_real_only(times, "times", "<f8")).tobytes()).hexdigest()


MAX_NPZ_MEMBERS = 4096
MAX_HEADER_BYTES = 1 << 20


def _inspect_npz_artifact(path: str) -> tuple:
    """Zip directory + ``.npy`` headers only; no payload array is materialised.

    Refuses unsafe member names, object dtypes, unsupported header versions, sizes inconsistent with the
    header and an oversize metadata member; parses only the bounded ``header`` member (``read_array``).
    Returns ``({key: (shape, dtype, bytes)}, header_dict)`` for the payload members.
    """
    import io
    import zipfile

    def bad(msg: str) -> TransientHistoryError:
        return TransientHistoryError(msg)

    members: dict = {}
    header_raw = None
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            if not 0 < len(infos) <= MAX_NPZ_MEMBERS:
                raise bad("artifact has no members or too many members")
            for info in infos:
                name = info.filename
                if not name.endswith(".npy") or "/" in name or "\\" in name or len(name) <= 4:
                    raise bad(f"unsafe or non-array member {name!r}")
                key = name[:-4]
                if key in members or (key == "header" and header_raw is not None):
                    raise bad(f"duplicate member {name!r}")
                with zf.open(info) as fh:
                    version = np.lib.format.read_magic(fh)
                    if version == (1, 0):
                        shape, _fortran, dtype = np.lib.format.read_array_header_1_0(fh)
                    elif version == (2, 0):
                        shape, _fortran, dtype = np.lib.format.read_array_header_2_0(fh)
                    else:
                        raise bad(f"member {name!r}: unsupported .npy version {version}")
                if dtype.hasobject or dtype.kind not in "fiu":
                    raise bad(f"member {name!r}: unsafe dtype {dtype}")
                count = 1
                for d in shape:
                    if not isinstance(d, int) or d < 0:
                        raise bad(f"member {name!r}: invalid shape {shape}")
                    count *= int(d)
                payload = count * dtype.itemsize
                if not payload <= info.file_size <= payload + 65536:
                    raise bad(f"member {name!r}: declared size {info.file_size} B inconsistent with its header")
                if key == "header":
                    if info.file_size > MAX_HEADER_BYTES or dtype != np.dtype("u1") or len(shape) != 1:
                        raise bad("metadata header member is oversize or not a uint8 vector")
                    header_raw = zf.read(info)
                members[key] = (tuple(int(d) for d in shape), dtype, int(info.file_size))
            if header_raw is None:
                raise bad("artifact has no metadata header")
    except TransientHistoryError:
        raise
    except (zipfile.BadZipFile, OSError, ValueError, EOFError) as exc:
        raise bad(f"artifact is corrupt or unreadable: {exc}") from exc
    try:
        arr = np.lib.format.read_array(io.BytesIO(header_raw), allow_pickle=False)
        header = json.loads(bytes(arr.tobytes()).decode("utf-8"))
    except Exception as exc:
        raise bad(f"artifact metadata header is malformed: {exc}") from exc
    if not isinstance(header, dict) or not isinstance(header.get("meta"), dict):
        raise bad("metadata header lacks 'meta'")
    del members["header"]
    return members, header


def load_transient_history(path: str, *, expected_hash: Optional[str] = None,
                           limits: Optional[TransientHistoryLimits] = None) -> TransientFloatingLoadHistory:
    """Load and verify (format, structure, hash).  File size is checked before any array is read."""
    lim = limits or TransientHistoryLimits()
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise TransientHistoryError(f"cannot read transient history: {exc}") from exc
    if size > lim.max_bytes:
        raise TransientHistoryError(f"artifact file {size} B exceeds the storage ceiling {lim.max_bytes} B")
    # ---- admission BEFORE any array is materialised (zip directory + .npy headers only)
    members, header = _inspect_npz_artifact(path)
    try:
        meta = header["meta"]
        if meta.get("format") != TRANSIENT_HISTORY_FORMAT:
            raise TransientHistoryError("unknown transient-history format / version")
        comps_meta = meta["components"]
        if not isinstance(comps_meta, list) or not 0 < len(comps_meta) <= 1024:
            raise TransientHistoryError("component metadata missing or too many components")
        if "times" not in members or len(members["times"][0]) != 1:
            raise TransientHistoryError("times member missing or not 1-D")
        n = members["times"][0][0]
        expected = {"times": ((n,), "<f8"), "frame_indices": ((n,), "<i8"),
                    "primary_acceleration": ((n, 6), "<f8"), "displacement": ((n, 6), "<f8"),
                    "velocity": ((n, 6), "<f8"), "acceleration": ((n, 6), "<f8"),
                    "wind_velocity": ((n, 3), "<f8"), "relative_axial_speed": ((n,), "<f8"), "thrust": ((n,), "<f8")}
        for i, c in enumerate(comps_meta):
            shape = tuple(c["shape"])
            if len(shape) != 3 or shape[0] != n or any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in shape):
                raise TransientHistoryError(f"component {i} declares an invalid shape {shape}")
            expected[f"component_{i}"] = (shape, "<f8")
        if set(members) != set(expected):
            raise TransientHistoryError(f"unexpected or missing arrays: {sorted(set(members) ^ set(expected))}")
        total = 0
        for key, (shape, dt) in expected.items():
            got_shape, got_dtype, _size = members[key]
            if got_shape != shape or got_dtype != np.dtype(dt):
                raise TransientHistoryError(f"array {key!r} has shape/dtype {got_shape}/{got_dtype}, expected {shape}/{dt}")
            count = 1
            for d in shape:
                count *= int(d)
            total += count * 8
        if 2 * total > lim.max_bytes:
            raise TransientHistoryError(
                f"expanded arrays {total} B (2x while loading) exceed the storage ceiling {lim.max_bytes} B")
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in expected}
        components = tuple(
            LoadComponent(c["name"], c["kind"], c["unit"], c["frame"], c["attachment"], tuple(c["item_ids"]),
                          arrays[f"component_{i}"]) for i, c in enumerate(comps_meta))
        record = TransientFloatingLoadHistory(
            times=arrays["times"], frame_indices=arrays["frame_indices"], source_frame_count=meta["source_frame_count"],
            source_times_hash=meta["source_times_hash"], components=components,
            primary_acceleration=arrays["primary_acceleration"], displacement=arrays["displacement"],
            velocity=arrays["velocity"], acceleration=arrays["acceleration"], wind_velocity=arrays["wind_velocity"],
            relative_axial_speed=arrays["relative_axial_speed"], thrust=arrays["thrust"],
            identities=meta["identities"], status=meta["status"], status_message=meta["status_message"],
            window=meta["window"], limits=lim)
    except TransientHistoryError:
        raise
    except Exception as exc:
        raise TransientHistoryError(f"transient-history artifact is corrupt or malformed: {exc}") from exc
    if record.content_hash() != header.get("content_hash"):
        raise TransientHistoryError("content hash mismatch: the artifact was modified or corrupted")
    if expected_hash is not None and record.content_hash() != expected_hash:
        raise TransientHistoryError("artifact content differs from the expected (stale) hash")
    return record
