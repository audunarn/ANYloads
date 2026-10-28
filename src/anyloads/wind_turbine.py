"""Immutable SI wind history, rotor-axial thrust curve and fixed-body turbine applicator.

Headless physical records for the floating-wind transient route.  All values are SI
(seconds, metres, m/s, newtons, newton-metres).  Nothing here adds mass, clips a
curve, extrapolates, normalises a realisation or redraws a random stream.

Kaimal realisation (versioned ``KAIMAL_GENERATOR_VERSION``): one-sided longitudinal
spectrum

    S(f) = 4 sigma^2 L / U / (1 + 6 f L / U)^(5/3),   sigma = TI * U

whose exact antiderivative gives bin-integrated energies

    int_{f1}^{f2} S df = sigma^2 [ (1 + a f1)^(-2/3) - (1 + a f2)^(-2/3) ],  a = 6 L / U.

The record is a random-phase Fourier sum on the fixed frequency set
``f_k = k / period`` (bin ``k`` covers ``[(k-1/2), (k+1/2)] / period``).  The frequency
set, amplitudes and phases depend only on (mean, TI, L, seed, period, max_frequency),
never on the sampling step, so refinement re-samples the same continuous realisation.
Variance outside the represented bins (below the first half-bin and above the last
full bin) is reported, not redistributed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np

__all__ = [
    "KAIMAL_GENERATOR_VERSION",
    "THRUST_ABSCISSA",
    "KaimalRealization",
    "ThrustCurve",
    "ThrustCurveDomainError",
    "TurbineLoadApplicator",
    "WindError",
    "WindHistory",
    "kaimal_bin_energy",
    "generate_kaimal_realization",
    "generate_kaimal_wind_history",
]

KAIMAL_GENERATOR_VERSION = "kaimal-random-phase-pcg64-v1"
THRUST_ABSCISSA = "rotor_axial_speed"


class WindError(ValueError):
    """A wind record or request cannot be admitted; nothing was repaired."""


class ThrustCurveDomainError(ValueError):
    """Rotor-axial speed lies outside the supplied thrust-curve domain."""

    def __init__(self, message: str, *, time: Optional[float], value: float, domain: tuple):
        super().__init__(message)
        self.time = time
        self.value = value
        self.domain = domain


def _real(value: Any, name: str) -> np.ndarray:
    """Float64 view/copy of REAL data.  Complex input (even with zero imaginary part) is refused BEFORE any cast:
    a real time history must never silently discard a phasor component."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise WindError(f"{name}: not a numeric array ({exc})") from exc
    if raw.dtype.kind == "c":
        raise WindError(f"{name}: complex-valued input is refused (real data required; no imaginary part is dropped)")
    try:
        return np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise WindError(f"{name}: not a numeric array ({exc})") from exc


def _frozen(value: Any, name: str, *, ndim: Optional[int] = None, shape: Optional[tuple] = None) -> np.ndarray:
    try:
        out = np.array(_real(value, name), dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise WindError(f"{name}: not a numeric array ({exc})") from exc
    if ndim is not None and out.ndim != ndim:
        raise WindError(f"{name}: expected {ndim} dimension(s), got shape {out.shape}")
    if shape is not None and out.shape != shape:
        raise WindError(f"{name}: expected shape {shape}, got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise WindError(f"{name}: non-finite values are not admitted")
    if out.size == 0:
        out.flags.writeable = False
        return out
    return np.frombuffer(out.tobytes(), dtype=np.float64).reshape(out.shape)


def _freeze_mapping(value: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    def deep(v):
        if isinstance(v, Mapping):
            return MappingProxyType({str(k): deep(x) for k, x in v.items()})
        if isinstance(v, (list, tuple)):
            return tuple(deep(x) for x in v)
        if isinstance(v, np.ndarray):
            return tuple(deep(x) for x in v.tolist())
        return v
    return deep(dict(value or {}))


def _plain(v):
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def _unit(vector: Any, name: str) -> np.ndarray:
    v = np.array(_real(vector, name), dtype=np.float64)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise WindError(f"{name}: expected a finite 3-vector")
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise WindError(f"{name}: zero-length direction")
    return v / n


# --------------------------------------------------------------------------- spectrum

def _kaimal_cdf(f: np.ndarray, sigma: float, mean: float, length: float) -> np.ndarray:
    """int_0^f S df (closed form); tends to sigma^2 as f -> inf."""
    a = 6.0 * length / mean
    return sigma * sigma * (1.0 - (1.0 + a * np.asarray(f, dtype=np.float64)) ** (-2.0 / 3.0))


def kaimal_bin_energy(f_lo: Any, f_hi: Any, *, sigma: float, mean: float, length: float) -> np.ndarray:
    """Exact one-sided Kaimal variance in ``[f_lo, f_hi]`` (m^2/s^2)."""
    lo = np.asarray(f_lo, dtype=np.float64)
    hi = np.asarray(f_hi, dtype=np.float64)
    a = 6.0 * length / mean
    return sigma * sigma * ((1.0 + a * lo) ** (-2.0 / 3.0) - (1.0 + a * hi) ** (-2.0 / 3.0))


@dataclass(frozen=True)
class KaimalRealization:
    """One fixed continuous random-phase realisation (independent of any dt)."""

    mean_speed: float
    turbulence_intensity: float
    integral_scale: float
    direction: np.ndarray
    seed: int
    period: float
    max_frequency: float
    frequencies: np.ndarray
    bin_energy: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    missing_low_variance: float
    missing_high_variance: float

    def __post_init__(self) -> None:
        for name in ("direction", "frequencies", "bin_energy", "amplitudes", "phases"):
            object.__setattr__(self, name, _frozen(getattr(self, name), name))
        # direct construction / dataclasses.replace must not admit an impossible realisation
        scalars = {}
        for name in ("mean_speed", "turbulence_intensity", "integral_scale", "period", "max_frequency",
                     "missing_low_variance", "missing_high_variance"):
            try:
                x = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise WindError(f"{name}: not a number ({exc})") from exc
            if not np.isfinite(x):
                raise WindError(f"{name}: must be finite")
            scalars[name] = x
            object.__setattr__(self, name, x)
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or self.seed < 0:
            raise WindError("seed: a fixed non-negative integer is required")
        object.__setattr__(self, "seed", int(self.seed))
        if scalars["turbulence_intensity"] < 0.0:
            raise WindError("turbulence_intensity: must be >= 0")
        if scalars["period"] <= 0.0 or scalars["max_frequency"] <= 0.0:
            raise WindError("period and max_frequency must be positive")
        if scalars["missing_low_variance"] < 0.0 or scalars["missing_high_variance"] < 0.0:
            raise WindError("missing variance must be nonnegative")
        if self.direction.shape != (3,) or abs(float(np.linalg.norm(self.direction)) - 1.0) > 1e-12:
            raise WindError("direction: a unit 3-vector is required")
        n = self.frequencies.size
        for name in ("frequencies", "bin_energy", "amplitudes", "phases"):
            arr = getattr(self, name)
            if arr.ndim != 1 or arr.size != n:
                raise WindError(f"{name}: must be 1-D with one entry per frequency ({n})")
        sigma = scalars["turbulence_intensity"] * scalars["mean_speed"]
        if scalars["turbulence_intensity"] > 0.0:
            if scalars["mean_speed"] <= 0.0:
                raise WindError("mean_speed: generated nonzero turbulence requires a positive mean speed")
            if scalars["integral_scale"] <= 0.0:
                raise WindError("integral_scale: explicit positive integral scale required")
            if n < 1:
                raise WindError("nonzero turbulence needs at least one spectral component")
        elif n != 0 or scalars["missing_low_variance"] != 0.0 or scalars["missing_high_variance"] != 0.0:
            raise WindError("zero turbulence carries no spectral components or missing variance")
        if n:
            expected = np.arange(1, n + 1, dtype=np.float64) / scalars["period"]
            if not np.allclose(self.frequencies, expected, rtol=1e-12, atol=0.0):
                raise WindError("frequencies must be the fixed set k/period, k = 1..n")
            if self.frequencies[-1] > scalars["max_frequency"] * (1.0 + 1e-12):
                raise WindError("frequencies exceed max_frequency")
            if np.any(self.bin_energy < 0.0) or np.any(self.amplitudes < 0.0):
                raise WindError("bin energies and amplitudes must be nonnegative")
            if not np.allclose(0.5 * self.amplitudes**2, self.bin_energy, rtol=1e-10, atol=0.0):
                raise WindError("amplitudes must equal sqrt(2 * bin_energy)")
            if np.any(self.phases < 0.0) or np.any(self.phases > 2.0 * np.pi):
                raise WindError("phases must lie in [0, 2 pi]")
            total = float(np.sum(self.bin_energy)) + scalars["missing_low_variance"] + scalars["missing_high_variance"]
            if abs(total - sigma * sigma) > 1e-9 * max(sigma * sigma, np.finfo(float).tiny):
                raise WindError("represented + missing variance must equal the target variance sigma^2")

    @property
    def target_variance(self) -> float:
        return (self.turbulence_intensity * self.mean_speed) ** 2

    @property
    def represented_variance(self) -> float:
        return float(np.sum(self.bin_energy))

    @property
    def bin_width(self) -> float:
        return 1.0 / self.period

    def fluctuation(self, times: Any) -> np.ndarray:
        """Longitudinal fluctuation u'(t) (m/s) at arbitrary times."""
        t = _real(times, "times")
        if self.frequencies.size == 0:
            return np.zeros_like(t)
        out = np.zeros_like(t)
        for start in range(0, self.frequencies.size, 256):
            sl = slice(start, start + 256)
            phase = 2.0 * np.pi * np.multiply.outer(t, self.frequencies[sl]) + self.phases[sl]
            out = out + np.cos(phase) @ self.amplitudes[sl]
        return out

    def to_wind_history(self, times: Any) -> "WindHistory":
        t = _frozen(times, "times", ndim=1)
        if t.size < 2 or np.any(np.diff(t) <= 0.0):
            raise WindError("times: at least two strictly increasing samples are required")
        # conservative: the LARGEST gap sets the sampling admission (one small interval cannot hide a long gap)
        dt_max = float(np.max(np.diff(t)))
        fmax = float(self.frequencies[-1]) if self.frequencies.size else 0.0
        if fmax > 0.5 / dt_max * (1.0 + 1e-12):
            raise WindError(
                f"aliasing: highest realised frequency {fmax:.6g} Hz exceeds the Nyquist frequency "
                f"{0.5 / dt_max:.6g} Hz of the largest sampling gap {dt_max:.6g} s; refine dt or lower max_frequency"
            )
        if t[0] < 0.0 or t[-1] > self.period * (1.0 + 1e-12):
            raise WindError(
                f"times [{t[0]!r}, {t[-1]!r}] leave the realisation period [0, {self.period!r}]; "
                "the realisation is never looped"
            )
        speed = self.mean_speed + self.fluctuation(t)
        vel = speed[:, None] * self.direction[None, :]
        return WindHistory(
            times=t,
            velocity=vel,
            reference_mean=self.mean_speed * self.direction,
            provenance=dict(
                source="generated_kaimal",
                generator=KAIMAL_GENERATOR_VERSION,
                seed=int(self.seed),
                mean_speed=self.mean_speed,
                turbulence_intensity=self.turbulence_intensity,
                integral_scale=self.integral_scale,
                period=self.period,
                max_frequency=self.max_frequency,
                n_components=int(self.frequencies.size),
                target_variance=self.target_variance,
                represented_variance=self.represented_variance,
                missing_low_variance=self.missing_low_variance,
                missing_high_variance=self.missing_high_variance,
                lowest_frequency=float(self.frequencies[0]) if self.frequencies.size else None,
            ),
        )


def generate_kaimal_realization(
    *, mean_speed: float, turbulence_intensity: float, integral_scale: float, direction: Any,
    seed: int, period: float, max_frequency: float,
) -> KaimalRealization:
    """Deterministic longitudinal Kaimal realisation on a fixed frequency set.

    ``period`` must cover the whole record (including run-in); it is never looped.
    """
    U = float(mean_speed)
    ti = float(turbulence_intensity)
    L = float(integral_scale)
    T = float(period)
    fmax = float(max_frequency)
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise WindError("seed: a fixed non-negative integer is required")
    if not (np.isfinite(ti) and ti >= 0.0):
        raise WindError("turbulence_intensity: finite and >= 0 required")
    if not (np.isfinite(T) and T > 0.0 and np.isfinite(fmax) and fmax > 0.0):
        raise WindError("period and max_frequency must be finite and positive")
    d = _unit(direction, "direction")
    if ti > 0.0:
        if not (np.isfinite(U) and U > 0.0):
            raise WindError("mean_speed: generated nonzero turbulence requires a positive mean speed")
        if not (np.isfinite(L) and L > 0.0):
            raise WindError("integral_scale: explicit positive integral scale required")
    elif not np.isfinite(U):
        raise WindError("mean_speed must be finite")
    df = 1.0 / T
    sigma = ti * U if ti > 0.0 else 0.0
    if sigma == 0.0:
        empty = np.zeros(0)
        return KaimalRealization(U, ti, L, d, int(seed), T, fmax, empty, empty, empty, empty, 0.0, 0.0)
    n_bins = int(np.floor(fmax / df - 0.5 + 1e-12))
    if n_bins < 1:
        raise WindError("max_frequency is below the first full spectral bin; lengthen period or raise it")
    k = np.arange(1, n_bins + 1, dtype=np.float64)
    f = k * df
    energy = kaimal_bin_energy((k - 0.5) * df, (k + 0.5) * df, sigma=sigma, mean=U, length=L)
    amp = np.sqrt(2.0 * energy)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_bins)
    low = float(kaimal_bin_energy(0.0, 0.5 * df, sigma=sigma, mean=U, length=L))
    high = float(sigma * sigma - _kaimal_cdf((n_bins + 0.5) * df, sigma, U, L))
    return KaimalRealization(U, ti, L, d, int(seed), T, fmax, f, energy, amp, phases, low, high)


def generate_kaimal_wind_history(
    *, mean_speed: float, turbulence_intensity: float, integral_scale: float, direction: Any,
    seed: int, dt: float, duration: float, max_frequency: Optional[float] = None,
) -> "WindHistory":
    """Convenience: realisation over ``[0, duration]`` sampled at fixed ``dt``."""
    dt = float(dt)
    duration = float(duration)
    if not (np.isfinite(dt) and dt > 0.0 and np.isfinite(duration) and duration >= dt):
        raise WindError("dt must be positive and duration >= dt")
    n = int(round(duration / dt))
    if abs(n * dt - duration) > 1e-9 * duration:
        raise WindError("duration must be an integer number of dt")
    real = generate_kaimal_realization(
        mean_speed=mean_speed, turbulence_intensity=turbulence_intensity,
        integral_scale=integral_scale, direction=direction, seed=seed, period=duration,
        max_frequency=0.5 / dt if max_frequency is None else max_frequency,
    )
    return real.to_wind_history(np.arange(n + 1, dtype=np.float64) * dt)


# --------------------------------------------------------------------------- wind history

@dataclass(frozen=True)
class WindHistory:
    """Absolute-time world-frame wind velocity vectors with realisation provenance."""

    times: np.ndarray
    velocity: np.ndarray
    reference_mean: np.ndarray
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = _frozen(self.times, "times", ndim=1)
        if t.size < 2:
            raise WindError("times: at least two samples are required")
        if np.any(np.diff(t) <= 0.0):
            raise WindError("times: strictly increasing finite times are required")
        v = _frozen(self.velocity, "velocity", shape=(t.size, 3))
        m = _frozen(self.reference_mean, "reference_mean", shape=(3,))
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "velocity", v)
        object.__setattr__(self, "reference_mean", m)
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))

    @classmethod
    def from_vectors(cls, times: Any, velocity: Any, *, reference_mean: Any = None,
                     source: str = "imported") -> "WindHistory":
        """Import a 3-vector history; calm and arbitrary-direction records are allowed."""
        vel = _real(velocity, "velocity")
        mean = np.mean(vel, axis=0) if reference_mean is None and vel.ndim == 2 else reference_mean
        return cls(times, velocity, mean, {"source": str(source)})

    @classmethod
    def from_longitudinal(cls, times: Any, speed: Any, direction: Any, *, source: str = "imported") -> "WindHistory":
        s = _real(speed, "speed")
        if s.ndim != 1:
            raise WindError("speed: 1-D array required")
        d = _unit(direction, "direction")
        return cls.from_vectors(times, s[:, None] * d[None, :], source=source)

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0])

    @property
    def n_samples(self) -> int:
        return int(self.times.size)

    @property
    def is_uniform(self) -> bool:
        d = np.diff(self.times)
        return bool(np.allclose(d, d[0], rtol=1e-9, atol=0.0))

    @property
    def nyquist_hz(self) -> float:
        return 0.5 / float(np.max(np.diff(self.times)))

    def sample_variance(self) -> np.ndarray:
        """Sample variance per component about the sample mean."""
        return np.var(self.velocity, axis=0)

    def at(self, time: Any) -> np.ndarray:
        """Piecewise-linear velocity; no extrapolation (refuses outside the record)."""
        t = _real(time, "time")
        if not np.all(np.isfinite(t)):
            raise WindError("query time must be finite")
        tol = 1e-12 * max(1.0, abs(float(self.times[-1])))
        if np.any(t < self.times[0] - tol) or np.any(t > self.times[-1] + tol):
            raise WindError(
                f"time {t!r} outside wind record [{self.times[0]!r}, {self.times[-1]!r}]; "
                "no extrapolation or looping"
            )
        tc = np.clip(t, self.times[0], self.times[-1])
        return np.stack([np.interp(tc, self.times, self.velocity[:, i]) for i in range(3)], axis=-1)

    def covers(self, times: Any) -> bool:
        t = _real(times, "times")
        tol = 1e-12 * max(1.0, abs(float(self.times[-1])))
        return bool(t.min() >= self.times[0] - tol and t.max() <= self.times[-1] + tol)

    def check_band_coverage(self, omega_max: float) -> None:
        """Refuse when the sampling Nyquist is below the required hydro band (rad/s)."""
        if 2.0 * np.pi * self.nyquist_hz < float(omega_max):
            raise WindError(
                f"wind sampling Nyquist {2.0 * np.pi * self.nyquist_hz:.6g} rad/s does not cover "
                f"the required band {float(omega_max):.6g} rad/s"
            )

    def content_hash(self) -> str:
        h = hashlib.sha256()
        h.update(b"anyloads.wind_history.v1")
        for arr in (self.times, self.velocity, self.reference_mean):
            h.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
        h.update(json.dumps(_plain(self.provenance), sort_keys=True, default=str).encode())
        return h.hexdigest()


# --------------------------------------------------------------------------- thrust curve

@dataclass(frozen=True)
class ThrustCurve:
    """Piecewise-linear thrust over rotor-axial relative speed; strict domain, no clipping."""

    speeds: np.ndarray
    thrusts: np.ndarray
    abscissa: str = THRUST_ABSCISSA
    name: str = "thrust_curve"

    def __post_init__(self) -> None:
        if self.abscissa != THRUST_ABSCISSA:
            raise WindError(
                f"abscissa={self.abscissa!r}: only {THRUST_ABSCISSA!r} is admitted; a curve declared "
                "against horizontal hub-height speed is refused (no automatic tilt conversion)"
            )
        s = _frozen(self.speeds, "speeds", ndim=1)
        T = _frozen(self.thrusts, "thrusts", shape=s.shape)
        if s.size < 2:
            raise WindError("at least two curve samples are required")
        if np.any(s < 0.0) or np.any(np.diff(s) <= 0.0):
            raise WindError("speeds must be nonnegative and strictly increasing")
        if np.any(T < 0.0):
            raise WindError("thrusts must be nonnegative")
        object.__setattr__(self, "speeds", s)
        object.__setattr__(self, "thrusts", T)

    @property
    def domain(self) -> tuple:
        return (float(self.speeds[0]), float(self.speeds[-1]))

    def segments(self) -> tuple:
        """Per-segment (v0, v1, T0, T1) arrays, ordered by increasing speed."""
        return self.speeds[:-1], self.speeds[1:], self.thrusts[:-1], self.thrusts[1:]

    def _check(self, v: float, time: Optional[float]) -> None:
        lo, hi = self.domain
        if not (lo <= v <= hi):  # also fails for NaN
            raise ThrustCurveDomainError(
                f"rotor-axial speed {v!r} m/s at time {time!r} s is outside the thrust-curve domain "
                f"[{lo!r}, {hi!r}] m/s; no extrapolation, clipping or shutdown is applied",
                time=time, value=float(v), domain=(lo, hi),
            )

    def thrust(self, speed: float, *, time: Optional[float] = None) -> float:
        v = float(_real(speed, "speed"))
        self._check(v, time)
        return float(np.interp(v, self.speeds, self.thrusts))

    def slope(self, speed: float, *, time: Optional[float] = None) -> float:
        """Right-hand slope (left-hand at the last knot)."""
        v = float(_real(speed, "speed"))
        self._check(v, time)
        i = min(int(np.searchsorted(self.speeds, v, side="right")) - 1, self.speeds.size - 2)
        i = max(i, 0)
        return float((self.thrusts[i + 1] - self.thrusts[i]) / (self.speeds[i + 1] - self.speeds[i]))

    def content_hash(self) -> str:
        h = hashlib.sha256(b"anyloads.thrust_curve.v1")
        for arr in (self.speeds, self.thrusts):
            h.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
        h.update(self.abscissa.encode())
        return h.hexdigest()


# --------------------------------------------------------------------------- applicator

def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


@dataclass(frozen=True)
class TurbineLoadApplicator:
    """Fixed-body hub point and rotor axis; conservative wrench of an axial thrust.

    ``rotor_axis_body`` points in the thrust (downwind) direction, in mean-body axes.
    Construct from yaw/tilt with :meth:`from_yaw_tilt`:
    ``n = (cos(tilt)cos(yaw), cos(tilt)sin(yaw), sin(tilt))`` (positive tilt raises the axis
    toward +z).  ``hub_offset_body`` is measured from the mean CG.  The applicator carries no
    mass; the FE assembly owns all turbine mass.  ``attachment`` is an opaque identity of the
    structural node/interface receiving the load.
    """

    hub_offset_body: np.ndarray
    rotor_axis_body: np.ndarray
    attachment: str = "hub"

    def __post_init__(self) -> None:
        r = _frozen(self.hub_offset_body, "hub_offset_body", shape=(3,))
        n = np.array(_real(self.rotor_axis_body, "rotor_axis_body"), dtype=np.float64)
        if n.shape != (3,) or not np.all(np.isfinite(n)) or abs(float(np.linalg.norm(n)) - 1.0) > 1e-10:
            raise WindError("rotor_axis_body: must already be a finite unit 3-vector within 1e-10; it is never "
                            "normalised silently")
        if not isinstance(self.attachment, str) or not self.attachment:
            raise WindError("attachment: non-empty identity string required")
        object.__setattr__(self, "hub_offset_body", r)
        object.__setattr__(self, "rotor_axis_body", _frozen(n, "rotor_axis_body", shape=(3,)))

    @classmethod
    def from_yaw_tilt(cls, hub_offset_body: Any, *, yaw: float = 0.0, tilt: float = 0.0,
                      attachment: str = "hub") -> "TurbineLoadApplicator":
        cy, sy, ct, st = np.cos(yaw), np.sin(yaw), np.cos(tilt), np.sin(tilt)
        return cls(hub_offset_body, np.array([ct * cy, ct * sy, st]), attachment)

    @property
    def generalized_vector(self) -> np.ndarray:
        """b = [n; r x n]: generalized force per unit thrust (work-conjugate to q)."""
        return np.concatenate([self.rotor_axis_body, np.cross(self.hub_offset_body, self.rotor_axis_body)])

    def relative_axial_speed(self, wind_world: Any, Q0: Any, translation_velocity: Any = (0, 0, 0),
                             rotation: Any = (0, 0, 0), angular_velocity: Any = (0, 0, 0)) -> float:
        """Vrel = (U_b - v_h,b) . n with U_b = Q0^T U - theta x (Q0^T U); v_h,b = v + omega x r."""
        Q = _real(Q0, "Q0")
        U0 = Q.T @ _real(wind_world, "wind_world")
        theta = _real(rotation, "rotation")
        Ub = U0 - np.cross(theta, U0)
        vh = _real(translation_velocity, "translation_velocity") + np.cross(
            _real(angular_velocity, "angular_velocity"), self.hub_offset_body)
        return float((Ub - vh) @ self.rotor_axis_body)

    def hub_force_body(self, thrust: float) -> np.ndarray:
        return float(thrust) * self.rotor_axis_body

    def wrench_about(self, point_body: Any, thrust: float) -> np.ndarray:
        """Force and moment of the thrust (applied at the hub) about ``point_body``."""
        F = self.hub_force_body(thrust)
        arm = self.hub_offset_body - _real(point_body, "point_body")
        return np.concatenate([F, np.cross(arm, F)])

    def generalized_force(self, thrust: float) -> np.ndarray:
        return self.wrench_about(np.zeros(3), thrust)

    def power(self, thrust: float, translation_velocity: Any, angular_velocity: Any = (0, 0, 0)) -> float:
        """Virtual power ``b.T . qdot`` of the generalized force (consistency check)."""
        qd = np.concatenate([_real(translation_velocity, "translation_velocity"),
                             _real(angular_velocity, "angular_velocity")])
        return float(self.generalized_force(thrust) @ qd)

    def content_hash(self) -> str:
        h = hashlib.sha256(b"anyloads.turbine_applicator.v1")
        for arr in (self.hub_offset_body, self.rotor_axis_body):
            h.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
        h.update(self.attachment.encode())
        return h.hexdigest()


def skew(v: Any) -> np.ndarray:
    """Cross-product matrix [v]x (shared with the external-history module)."""
    return _skew(_real(v, "v"))
