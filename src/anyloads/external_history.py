"""Timestamped external load histories and affine fairlead mooring load recovery.

``ExternalLoadHistory`` carries actual accepted times, declared units/frame/identity and
deterministic piecewise-linear interpolation (no extrapolation).  Loads are reduced to rigid
generalized forces with the same virtual-work map ``A_i = [I, -[r_i]x]`` used to apply them to
the structure.  ``AffineFairleadMooring`` recovers per-fairlead forces and the matching
stiffness/damping/pretension-geometry contributions from a supplied fairlead model; a bare 6x6
resultant stiffness without recoverable attachment forces is not accepted.

Rigid generalized coordinates are q = (translation, small rotation theta) through the mean CG.
Forces are expressed in the first-order BODY-ATTACHED frame (components rotate with the body,
body points r are fixed in it), as in the plan's wind pull-back U_b = Q0^T U - theta x U.  A
world-fixed force F_w acting at body point r then has

    f(theta) = f0 + [f0]x theta,        f0 = Q0^T F_w
    m(theta) = r x f(theta) = r x f0 + [r]x[f0]x theta

(force and moment about the CG).  Second-order rotation terms are omitted.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .wind_turbine import WindError, _freeze_mapping, _plain, skew
from .wind_turbine import _frozen as _wind_frozen
from .wind_turbine import _real as _wind_real


def _real(value: Any, name: str) -> np.ndarray:
    """Float64 REAL data; complex input (even with zero imaginary part) is refused before any cast."""
    try:
        return _wind_real(value, name)
    except WindError as exc:
        raise ExternalHistoryError(str(exc)) from exc

__all__ = [
    "AffineFairleadMooring",
    "ExternalHistoryError",
    "ExternalLoadHistory",
    "FairleadSpec",
    "rigid_attachment_map",
    "world_force_pullback",
]

COMPONENTS = ("FX", "FY", "FZ", "MX", "MY", "MZ")
FRAMES = ("world", "body")
UNITS = {"FX": "N", "FY": "N", "FZ": "N", "MX": "N*m", "MY": "N*m", "MZ": "N*m"}


class ExternalHistoryError(ValueError):
    """An external-load record cannot be admitted or evaluated."""


def _frozen(value: Any, name: str, **kwargs: Any) -> np.ndarray:
    """Immutable finite float copy; failures are reported as ExternalHistoryError."""
    try:
        return _wind_frozen(value, name, **kwargs)
    except WindError as exc:
        raise ExternalHistoryError(str(exc)) from exc


def _state(name: str, value: Any) -> np.ndarray:
    """Finite (6,) rigid state vector; NaN/inf or wrong shape is refused."""
    arr = np.array(_real(value, name), dtype=np.float64, copy=True)
    if arr.shape != (6,) or not np.all(np.isfinite(arr)):
        raise ExternalHistoryError(f"{name}: a finite 6-vector is required, got shape {arr.shape}")
    return arr


def _proper_rotation(Q0: Any) -> np.ndarray:
    """Finite orthogonal 3x3 with determinant +1 (body -> world); anything else is refused."""
    Q = np.array(_real(Q0, "Q0"), dtype=np.float64, copy=True)
    if (Q.shape != (3, 3) or not np.all(np.isfinite(Q)) or np.max(np.abs(Q.T @ Q - np.eye(3))) > 1e-10
            or abs(float(np.linalg.det(Q)) - 1.0) > 1e-10):
        raise ExternalHistoryError("Q0: proper rotation (orthogonal, determinant +1, body->world) required")
    return Q


def rigid_attachment_map(point_body: Any) -> np.ndarray:
    """A (3x6): small displacement of a body point, u = A q, q = (translation, theta)."""
    r = _real(point_body, "point_body")
    if r.shape != (3,) or not np.all(np.isfinite(r)):
        raise ExternalHistoryError("point_body: finite 3-vector required")
    return np.hstack([np.eye(3), -skew(r)])


def world_force_pullback(force_world: Any, Q0: Any, point_body: Any) -> tuple:
    """First-order (wrench0, tangent G) of a world-fixed force at a body point.

    wrench(q) = wrench0 + G @ q about the CG in mean-body axes; ``G`` acts on theta only.
    """
    Q = _proper_rotation(Q0)
    f0 = Q.T @ _real(force_world, "force_world")
    r = _real(point_body, "point_body")
    w0 = np.concatenate([f0, np.cross(r, f0)])
    G = np.zeros((6, 6))
    G[:3, 3:] = skew(f0)
    G[3:, 3:] = skew(r) @ skew(f0)
    return w0, G


@dataclass(frozen=True)
class ExternalLoadHistory:
    """Timestamped load channels.

    ``values[k, c]`` is channel ``c`` at ``times[k]``.  A channel is a force or moment
    component (``FX..MZ``) applied at ``points_body`` (a body point relative to the CG, ignored
    for pure moments) and expressed in ``frame`` ('world' = prescribed world direction,
    'body' = attached to the body).  ``identity`` records body/mesh/mass identity strings.
    """

    times: np.ndarray
    values: np.ndarray
    components: tuple
    frames: tuple
    points_body: np.ndarray
    attachments: tuple
    identity: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = _frozen(self.times, "times", ndim=1)
        if t.size < 2 or np.any(np.diff(t) <= 0.0):
            raise ExternalHistoryError("times: at least two unique strictly increasing finite times required")
        comps = tuple(str(c).upper() for c in self.components)
        n = len(comps)
        if n == 0 or any(c not in COMPONENTS for c in comps):
            raise ExternalHistoryError(f"components must be drawn from {COMPONENTS}")
        frames = tuple(str(f) for f in self.frames)
        if len(frames) != n or any(f not in FRAMES for f in frames):
            raise ExternalHistoryError(f"frames: one of {FRAMES} per channel required")
        atts = tuple(str(a) for a in self.attachments)
        if len(atts) != n or any(not a for a in atts):
            raise ExternalHistoryError("attachments: one non-empty identity per channel required")
        try:
            vals = _frozen(self.values, "values", shape=(t.size, n))
            pts = _frozen(self.points_body, "points_body", shape=(n, 3))
        except WindError as exc:
            raise ExternalHistoryError(str(exc)) from exc
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "values", vals)
        object.__setattr__(self, "points_body", pts)
        object.__setattr__(self, "components", comps)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "attachments", atts)
        object.__setattr__(self, "identity", _freeze_mapping(self.identity))

    @property
    def units(self) -> tuple:
        return tuple(UNITS[c] for c in self.components)

    def covers(self, times: Any) -> bool:
        t = _real(times, "times")
        tol = 1e-12 * max(1.0, abs(float(self.times[-1])))
        return bool(t.min() >= self.times[0] - tol and t.max() <= self.times[-1] + tol)

    def at(self, times: Any) -> np.ndarray:
        """Piecewise-linear channel values, shape (n_t, n_channels); no extrapolation."""
        t = np.atleast_1d(_real(times, "times"))
        if not np.all(np.isfinite(t)):
            raise ExternalHistoryError("requested times must be finite")
        if not self.covers(t):
            raise ExternalHistoryError(
                f"requested times [{t.min()!r}, {t.max()!r}] are not covered by the history "
                f"[{self.times[0]!r}, {self.times[-1]!r}]; no extrapolation"
            )
        tc = np.clip(t, self.times[0], self.times[-1])
        return np.stack([np.interp(tc, self.times, self.values[:, c]) for c in range(self.values.shape[1])], axis=1)

    def resample(self, times: Any, *, allow_decimation: bool = False) -> "ExternalLoadHistory":
        """Resample on ``times``; a coarser step than the source is refused unless declared."""
        t = _frozen(times, "times", ndim=1)
        if t.size >= 2 and not allow_decimation:
            if float(np.max(np.diff(t))) > float(np.min(np.diff(self.times))) * (1.0 + 1e-9):
                raise ExternalHistoryError(
                    "resampling to a coarser step than the source would silently alias; "
                    "pass allow_decimation=True to accept it explicitly"
                )
        return ExternalLoadHistory(t, self.at(t), self.components, self.frames, self.points_body,
                                   self.attachments, self.identity)

    def generalized_force(self, times: Any, Q0: Any) -> np.ndarray:
        """Rigid generalized force (n_t, 6) at the mean pose via virtual work.

        A force ``F`` at point ``r`` contributes ``[F; r x F]``; a pure moment contributes
        ``[0; M]``.  World-frame channels are rotated by ``Q0^T``.
        """
        vals = self.at(times)
        Q = _proper_rotation(Q0)
        out = np.zeros((vals.shape[0], 6))
        for c, comp in enumerate(self.components):
            axis = "XYZ".index(comp[1])
            unit = np.zeros(3)
            unit[axis] = 1.0
            if self.frames[c] == "world":
                unit = Q.T @ unit
            if comp[0] == "F":
                arm = np.cross(self.points_body[c], unit)
                out[:, :3] += vals[:, c:c + 1] * unit
                out[:, 3:] += vals[:, c:c + 1] * arm
            else:
                out[:, 3:] += vals[:, c:c + 1] * unit
        return out

    def load_tangent(self, times: Any, Q0: Any) -> np.ndarray:
        """Geometric tangent G(t) (n_t,6,6) of world-direction channels: wrench = w0 + G q."""
        vals = self.at(times)
        Q = _proper_rotation(Q0)
        G = np.zeros((vals.shape[0], 6, 6))
        for c, comp in enumerate(self.components):
            if self.frames[c] != "world":
                continue
            axis = "XYZ".index(comp[1])
            unit = np.zeros(3)
            unit[axis] = 1.0
            if comp[0] == "F":
                for k in range(vals.shape[0]):
                    _, g = world_force_pullback(vals[k, c] * unit, Q, self.points_body[c])
                    G[k] += g
            else:
                f0 = Q.T @ unit
                for k in range(vals.shape[0]):
                    # a world-fixed pure moment: m(theta) = (I - [theta]x) m0
                    G[k, 3:, 3:] += skew(vals[k, c] * f0)
        return G

    def content_hash(self) -> str:
        h = hashlib.sha256(b"anyloads.external_history.v1")
        for arr in (self.times, self.values, self.points_body):
            h.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
        h.update(repr((self.components, self.frames, self.attachments)).encode())
        # nested identity data is canonicalised (sorted keys at every depth) so equal content hashes equally
        h.update(json.dumps(_plain(self.identity), sort_keys=True, separators=(",", ":"), default=str).encode())
        return h.hexdigest()


@dataclass(frozen=True)
class FairleadSpec:
    """One fairlead: identity, body point and supplied force model.

    Force on the platform at the fairlead (mean-body axes):

        F = F0 - K u - C udot,   u = A q (fairlead displacement),

    ``F0`` is the pretension force in WORLD axes at the declared reference pose (it keeps its
    world direction under rotation; ``K`` must NOT include that geometric term, which is
    added here).  ``stiffness``/``damping`` are 3x3 in mean-body axes and need not be symmetric.
    """

    identity: str
    point_body: np.ndarray
    pretension_world: np.ndarray
    stiffness: np.ndarray
    damping: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity:
            raise ExternalHistoryError("fairlead identity: non-empty string required")
        try:
            object.__setattr__(self, "point_body", _frozen(self.point_body, "point_body", shape=(3,)))
            object.__setattr__(self, "pretension_world", _frozen(self.pretension_world, "pretension_world", shape=(3,)))
            object.__setattr__(self, "stiffness", _frozen(self.stiffness, "stiffness", shape=(3, 3)))
            damping = np.zeros((3, 3)) if self.damping is None else self.damping
            object.__setattr__(self, "damping", _frozen(damping, "damping", shape=(3, 3)))
        except WindError as exc:
            raise ExternalHistoryError(str(exc)) from exc
        # supplied mooring damping must be passive: the symmetric part has no negative eigenvalue
        # (aerodynamic negative damping is a distinct, reported turbine effect and never enters here)
        sym = 0.5 * (self.damping + self.damping.T)
        scale = max(1.0, float(np.max(np.abs(self.damping))))
        if float(np.linalg.eigvalsh(sym)[0]) < -1e-12 * scale:
            raise ExternalHistoryError(
                f"fairlead {self.identity!r}: supplied damping is not passive (symmetric part has a negative "
                "eigenvalue); a mooring may not generate energy")


@dataclass(frozen=True)
class AffineFairleadMooring:
    """Supplied linear mooring with per-fairlead force recovery and exact virtual-work reduction."""

    fairleads: tuple
    Q0: np.ndarray
    validity_half_range: Optional[np.ndarray] = None  # |q_i| bound per DOF, None = not declared

    def __post_init__(self) -> None:
        fl = tuple(self.fairleads)
        if not fl or not all(isinstance(f, FairleadSpec) for f in fl):
            raise ExternalHistoryError("at least one FairleadSpec is required")
        ids = [f.identity for f in fl]
        if len(set(ids)) != len(ids):
            raise ExternalHistoryError("fairlead identities must be unique")
        object.__setattr__(self, "fairleads", fl)
        Q = _frozen(self.Q0, "Q0", shape=(3, 3))
        if np.max(np.abs(Q.T @ Q - np.eye(3))) > 1e-10 or abs(float(np.linalg.det(Q)) - 1.0) > 1e-10:
            raise ExternalHistoryError("Q0: proper rotation (orthogonal, determinant +1, body->world) required")
        object.__setattr__(self, "Q0", Q)
        if self.validity_half_range is not None:
            vr = _frozen(self.validity_half_range, "validity_half_range", shape=(6,))
            if np.any(vr < 0.0):
                raise ExternalHistoryError("validity_half_range: nonnegative finite bounds required")
            object.__setattr__(self, "validity_half_range", vr)

    @property
    def identities(self) -> tuple:
        return tuple(f.identity for f in self.fairleads)

    def _f0_body(self, f: FairleadSpec) -> np.ndarray:
        return self.Q0.T @ f.pretension_world

    def _gate(self, q: Any = None) -> None:
        """Operational use requires a DECLARED validity envelope (its absence is not permission) and, when a pose
        is given, a finite pose inside it.  ``geometric_tangent`` / ``identities`` / ``asymmetry`` are inspection
        helpers and are not gated."""
        if self.validity_half_range is None:
            raise ExternalHistoryError(
                "no validity envelope declared; the supplied mooring model may not be used operationally")
        if q is not None:
            self.check_validity(q)

    def pretension_wrench(self) -> np.ndarray:
        """Equilibrium generalized force of the pretensions at the reference pose (needs a declared envelope)."""
        self._gate()
        w = np.zeros(6)
        for f in self.fairleads:
            w += rigid_attachment_map(f.point_body).T @ self._f0_body(f)
        return w

    def geometric_tangent(self) -> np.ndarray:
        """d(wrench)/dq from the world-fixed pretension direction (acts on theta)."""
        G = np.zeros((6, 6))
        for f in self.fairleads:
            G += world_force_pullback(f.pretension_world, self.Q0, f.point_body)[1]
        return G

    def stiffness_matrix(self) -> np.ndarray:
        """K_total with wrench = pretension - K_total q (includes -geometric_tangent); needs a declared envelope."""
        self._gate()
        K = np.zeros((6, 6))
        for f in self.fairleads:
            A = rigid_attachment_map(f.point_body)
            K += A.T @ f.stiffness @ A
        return K - self.geometric_tangent()

    def damping_matrix(self) -> np.ndarray:
        self._gate()
        C = np.zeros((6, 6))
        for f in self.fairleads:
            A = rigid_attachment_map(f.point_body)
            C += A.T @ f.damping @ A
        return C

    def check_validity(self, q: Any) -> None:
        if self.validity_half_range is None:
            raise ExternalHistoryError("no validity envelope declared; the mooring model may not be used silently")
        q = _state("q", q)
        bad = np.abs(q) > self.validity_half_range
        if np.any(bad):
            raise ExternalHistoryError(
                f"pose component(s) {np.flatnonzero(bad).tolist()} leave the mooring validity envelope"
            )

    def fairlead_forces(self, q: Any, v: Any = None) -> dict:
        """Per-fairlead force recovery (mean-body axes) for pose ``q`` and velocity ``v``.

        Includes the first-order rotation of the world-fixed pretension.  The sum of
        ``A_i^T F_i`` equals :meth:`generalized_force` exactly.
        """
        self._gate(q)
        q = _state("q", q)
        v = np.zeros(6) if v is None else _state("v", v)
        theta = q[3:]
        out = {}
        for f in self.fairleads:
            A = rigid_attachment_map(f.point_body)
            f0 = self._f0_body(f)
            force = f0 + np.cross(f0, theta) - f.stiffness @ (A @ q) - f.damping @ (A @ v)
            out[f.identity] = force
        return out

    def generalized_force(self, q: Any, v: Any = None) -> np.ndarray:
        """Rigid reduction w = sum A_i^T F_i^(recovered at first order) about the CG."""
        forces = self.fairlead_forces(q, v)
        w = np.zeros(6)
        for f in self.fairleads:
            w[:3] += forces[f.identity]
            w[3:] += np.cross(f.point_body, forces[f.identity])
        return w

    def linear_generalized_force(self, q: Any, v: Any = None) -> np.ndarray:
        """w0 - K_total q - C v: the affine form used by the transient solver."""
        self._gate(q)
        q = _state("q", q)
        v = np.zeros(6) if v is None else _state("v", v)
        return self.pretension_wrench() - self.stiffness_matrix() @ q - self.damping_matrix() @ v

    def asymmetry(self) -> dict:
        K = self.stiffness_matrix()
        C = self.damping_matrix()
        return {
            "stiffness_skew_norm": float(np.linalg.norm(0.5 * (K - K.T))),
            "stiffness_sym_norm": float(np.linalg.norm(0.5 * (K + K.T))),
            "damping_skew_norm": float(np.linalg.norm(0.5 * (C - C.T))),
        }
