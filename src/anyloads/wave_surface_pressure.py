"""Immutable composed wave-following pressure history on an exact target of element corners.

Physics (adopted contract, z relative to the mean waterline, eta the local incident+diffraction surface,
pd the incident+diffraction pressure; all spectral components are summed BEFORE the wet/dry operation):

    z >= eta               -> pressure 0 (dry)
    z <  eta and z <= 0    -> p = -rho g z + pd(x, y, z, t)           (signed, no positivity clamp)
    0 <  z <  eta          -> p = -rho g z + pd(x, y, 0, t)           (constant waterline dynamic extension)

The source is stored ONCE as neutral per-frequency complex unit-amplitude fields at the target corner points
(never a dense time-by-mesh tensor) with the saved wave realization; each frame is evaluated at an absolute time.
This module is neutral: no hydrodynamics or FEM imports.  Clipping is delegated lazily to the ANYgeometry public
contract; consistent partial-region load integration belongs to ANYsolver and is applied by the application.
"""
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
import numpy as np
from .sea_state import WaveRealization

TYPE = 'wave_surface_pressure_v1'
ALGORITHM = 'sum_components_then_signed_linear_wet_clip_v1'
STATE_DRY, STATE_PARTIAL, STATE_WET = 0, 1, 2


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(value):
    return type(value) is str and re.fullmatch('(sha256:)?[0-9a-f]{64}', value) is not None


def _clip():
    try:
        from anygeometry.wave_surface_clipping import clip_triangle_positive
    except ImportError as error:  # pragma: no cover - environment boundary
        raise ValueError('ANYgeometry wave_surface_clipping is required to evaluate wet regions') from error
    return clip_triangle_positive


def _complex_rows(real, imag, rows, columns, name):
    r = np.asarray(real); i = np.asarray(imag)
    if r.dtype.kind not in 'fiu' or i.dtype.kind not in 'fiu' or r.shape != (rows, columns) or i.shape != r.shape \
            or not np.isfinite(r).all() or not np.isfinite(i).all():
        raise ValueError(f'finite frequency-by-corner {name} fields required')
    out = np.empty(r.shape, dtype=complex); out.real = r; out.imag = i
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class WaveSurfaceFrame:
    """Instantaneous per-element corner data at one absolute time."""
    time: float
    element_ids: tuple
    corners: tuple          # per element tuple of corner indices (3 or 4)
    eta: np.ndarray         # (P,) elevation at the corner points
    levels: tuple           # per element array: eta - z at its corners (wet where > 0)
    pressures: tuple        # per element array: continuous pressure continuation at its corners

    def __post_init__(self):
        # Immutable backing also prevents callers from re-enabling NumPy writes.
        def freeze(value, shape):
            a = np.asarray(value, dtype=float)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError('finite frame arrays with matching corner shapes required')
            return np.frombuffer(a.tobytes(), dtype=float).reshape(shape)
        ids = tuple(self.element_ids)
        corners = tuple(tuple(c) for c in self.corners)
        if not math.isfinite(float(self.time)) or len(corners) != len(ids) \
                or len(self.levels) != len(ids) or len(self.pressures) != len(ids):
            raise ValueError('matching finite frame time and element data required')
        eta = np.asarray(self.eta)
        if eta.ndim != 1 or any(len(c) not in (3, 4) or any(
                isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or
                not 0 <= i < len(eta) for i in c) for c in corners):
            raise ValueError('valid triangle/quad corner references required')
        object.__setattr__(self, 'element_ids', ids)
        object.__setattr__(self, 'corners', corners)
        object.__setattr__(self, 'eta', freeze(eta, (len(eta),)))
        for name in ('levels', 'pressures'):
            object.__setattr__(self, name, tuple(freeze(a, (len(c),))
                for a, c in zip(getattr(self, name), corners)))

    def wet_summary(self):
        """(state, wet_fraction, mean_pressure) per element; reference-area weighted, clipping via ANYgeometry.

        ``mean_pressure`` is the FULL-area element average (wet-region integral divided by full element area):
        dry elements are exactly zero.  For non-affine quads this is the reference-space average; the exact
        physical integral is the solver adapter's job.
        """
        clip = _clip()
        n = len(self.element_ids)
        state = np.zeros(n, dtype=np.int8); fraction = np.zeros(n); mean = np.zeros(n)
        for e, (g, p) in enumerate(zip(self.levels, self.pressures)):
            if g.max() <= 0.:
                continue
            if len(g) == 3:
                tris = [(g, p, 1.)]
            else:
                gc = g.mean(); pc = p.mean()
                tris = [(np.array([g[a], g[b], gc]), np.array([p[a], p[b], pc]), .25)
                        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))]
            wet = 0.; integral = 0.
            for gg, pp, weight in tris:
                c = clip(gg)
                if c.state == 'dry':
                    continue
                bary = c.parameters
                for a, b, d in c.fan:
                    area = abs(np.linalg.det(bary[[a, b, d]]))
                    wet += weight * area
                    integral += weight * area * float((bary[[a, b, d]] @ pp).mean())
            if wet <= 0.:
                continue
            fraction[e] = min(1., wet)
            mean[e] = integral
            state[e] = STATE_WET if wet > 1. - 1e-12 else STATE_PARTIAL
        return state, fraction, mean


@dataclass(frozen=True)
class WaveSurfacePressureHistory:
    times: tuple
    record_times: tuple
    omegas: tuple                 # unique realised frequencies, in realization order
    element_ids: tuple
    element_corners: tuple        # per element: 3 or 4 corner-point indices
    corner_z: tuple               # (P,) heights relative to the mean waterline
    eta_real: tuple; eta_imag: tuple     # (F,P) unit-amplitude fields, exp(-i omega t)
    pd0_real: tuple; pd0_imag: tuple
    pd_real: tuple; pd_imag: tuple
    heading: float
    rho: float
    gravity: float
    realization_json: str
    mesh_content_sha256: str
    model_content_sha256: str
    target_sha256: str
    source_id: str
    selected_window: tuple = ()
    provenance: str = ''
    source_payload_json: str = ''     # opaque retained source (e.g. field snapshots) for remapping; part of identity
    content_sha256: str = ''
    _arrays: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        for name in ('times', 'record_times', 'omegas', 'corner_z'):
            raw = tuple(getattr(self, name))
            if any(isinstance(x, (bool, np.bool_, complex, np.complexfloating)) for x in raw):
                raise ValueError('real numeric history values required')
            object.__setattr__(self, name, tuple(float(x) for x in raw))
        ids = tuple(self.element_ids)
        if not ids or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in ids) \
                or len(set(ids)) != len(ids):
            raise ValueError('unique integer target element IDs required')
        object.__setattr__(self, 'element_ids', tuple(int(x) for x in ids))
        corners = tuple(tuple(int(i) for i in c) for c in self.element_corners)
        P = len(self.corner_z)
        if len(corners) != len(ids) or any(len(c) not in (3, 4) or min(c) < 0 or max(c) >= P for c in corners):
            raise ValueError('every target element needs 3 or 4 valid corner points')
        object.__setattr__(self, 'element_corners', corners)
        F = len(self.omegas)
        arrays = {}
        for key in ('eta', 'pd0', 'pd'):
            arrays[key] = _complex_rows(getattr(self, key + '_real'), getattr(self, key + '_imag'), F, P, key)
            for part in ('real', 'imag'):
                object.__setattr__(self, f'{key}_{part}', tuple(tuple(float(x) for x in row)
                    for row in getattr(self, f'{key}_{part}')))
        if (not self.times or not self.record_times or not F or not P or not np.isfinite(self.times).all()
                or not np.isfinite(self.corner_z).all() or min(self.omegas) <= 0 or len(set(self.omegas)) != F
                or any(b <= a for a, b in zip(self.times, self.times[1:]))
                or any(b <= a for a, b in zip(self.record_times, self.record_times[1:]))
                or not set(self.times) <= set(self.record_times)
                or not all(math.isfinite(x) for x in (self.heading, self.rho, self.gravity))
                or not 0. <= self.heading < math.tau or self.rho <= 0 or self.gravity <= 0
                or not all(_sha(x) for x in (self.mesh_content_sha256, self.model_content_sha256, self.target_sha256))
                or re.fullmatch('[0-9a-f]{64}', self.source_id) is None):
            raise ValueError('admitted times, frequencies, heading, physics and exact mesh/model/target/source '
                             'identities required')
        realization = WaveRealization.from_dict(json.loads(self.realization_json))
        object.__setattr__(self, 'realization_json', _json(realization.to_dict()))
        weights = {}
        for c in realization.components:
            if c.amplitude == 0:
                continue
            weights[c.omega] = weights.get(c.omega, 0j) + c.amplitude * complex(math.cos(c.phase), math.sin(c.phase))
        order = tuple(dict.fromkeys(c.omega for c in realization.components if c.amplitude != 0))
        # A true zero-amplitude realization is still water: its (inert) frequencies stay listed with zero weight.
        order = order or tuple(dict.fromkeys(c.omega for c in realization.components))
        if tuple(self.omegas) != order:
            raise ValueError('history frequencies differ from the saved realization components')
        window = tuple(float(x) for x in self.selected_window)
        object.__setattr__(self, 'selected_window', window)
        identity = self._digest(arrays)
        if self.content_sha256 and self.content_sha256 != identity:
            raise ValueError('wave surface pressure history content identity mismatch')
        object.__setattr__(self, 'content_sha256', identity)
        for a in arrays.values():
            a.setflags(write=False)
        object.__setattr__(self, '_arrays', dict(arrays, weights=np.array([weights.get(w, 0j) for w in self.omegas]),
                                                  z=np.array(self.corner_z), omega=np.array(self.omegas)))

    def _meta(self):
        return {k: getattr(self, k) for k in ('times', 'record_times', 'omegas', 'element_ids', 'element_corners',
            'corner_z', 'heading', 'rho', 'gravity', 'realization_json', 'mesh_content_sha256',
            'model_content_sha256', 'target_sha256', 'source_id', 'selected_window', 'provenance',
            'source_payload_json')}

    def record_identity(self):
        """Canonical identity of the FULL record, independent of the selected window.

        Same digest as ``content_sha256`` with ``times = record_times`` and an empty selected window, over the
        unchanged field/source/model/mesh/target/settings data; for an unselected history it equals
        ``content_sha256``.  A selected history can therefore always be bound to its original full record.
        """
        meta = dict(self._meta(), times=self.record_times, selected_window=())
        h = hashlib.sha256(_json(dict(type=TYPE, algorithm=ALGORITHM, **meta)).encode())
        for name in ('eta', 'pd0', 'pd'):
            a = np.ascontiguousarray(self._arrays[name]); h.update(name.encode()); h.update(memoryview(a).cast('B'))
        return h.hexdigest()

    def _digest(self, arrays):
        h = hashlib.sha256(_json(dict(type=TYPE, algorithm=ALGORITHM, **self._meta())).encode())
        for name in ('eta', 'pd0', 'pd'):
            a = np.ascontiguousarray(arrays[name]); h.update(name.encode()); h.update(memoryview(a).cast('B'))
        return h.hexdigest()

    @property
    def mapped_history(self):
        return self

    def to_dict(self):
        return dict(type=TYPE, algorithm=ALGORITHM, **self._meta(),
            **{f'{k}_{p}': getattr(self, f'{k}_{p}') for k in ('eta', 'pd0', 'pd') for p in ('real', 'imag')},
            content_sha256=self.content_sha256)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if data.pop('type', None) != TYPE or data.pop('algorithm', None) != ALGORITHM or not data.get('content_sha256'):
            raise ValueError('unsupported or unverified wave surface pressure history')
        return cls(**data)

    def with_window(self, times, window=()):
        """Same source and absolute-phase record, restricted to a subset of the original sample times."""
        data = self.to_dict()
        for key in ('type', 'algorithm', 'content_sha256'):
            data.pop(key)
        data['times'] = tuple(times); data['selected_window'] = tuple(window)
        return type(self)(**data)

    def _fields_at(self, time):
        a = self._arrays
        phase = a['weights'] * np.exp(-1j * a['omega'] * float(time))
        return tuple(np.real(phase @ a[k]) for k in ('eta', 'pd0', 'pd'))

    def elevation_series(self, times=None):
        """(T, P) realised elevation at the corner points for absolute ``times`` (default: the selected times).

        Vectorised for envelope checks (keel emergence / deck overtopping); components summed coherently.
        """
        t = np.asarray(self.times if times is None else times, dtype=float)
        a = self._arrays
        return np.real(np.exp(-1j * np.outer(t, a['omega'])) * a['weights'] @ a['eta'])

    def frame(self, time):
        if isinstance(time, (bool, np.bool_, complex, np.complexfloating)) or not isinstance(
                time, (int, float, np.integer, np.floating)) or not math.isfinite(float(time)) \
                or not self.times[0] <= float(time) <= self.times[-1]:
            raise ValueError('pressure history time is outside its admitted time range')
        eta, pd0, pd = self._fields_at(time)
        z = self._arrays['z']; rg = self.rho * self.gravity
        # Continuous pressure continuation: dynamic part at the actual depth below the mean level, the constant
        # waterline value above it.  The wet/dry decision uses the level eta - z only.
        pressure = -rg * z + np.where(z <= 0., pd, pd0)
        level = eta - z
        return WaveSurfaceFrame(float(time), self.element_ids, self.element_corners, eta,
            tuple(level[list(c)] for c in self.element_corners), tuple(pressure[list(c)] for c in self.element_corners))

    def evaluate(self, time, element_ids=None):
        """Full-area element-average pressure (dry exactly zero); see ``WaveSurfaceFrame.wet_summary``."""
        mean = self.frame(time).wet_summary()[2]
        if element_ids is None:
            return mean
        position = {e: i for i, e in enumerate(self.element_ids)}
        if any(isinstance(e, (bool, np.bool_)) or not isinstance(e, (int, np.integer)) or int(e) not in position
               for e in element_ids):
            raise ValueError('unknown element in wave surface pressure history evaluation')
        return mean[[position[int(e)] for e in element_ids]]
