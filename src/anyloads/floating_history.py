"""Immutable neutral coupled floating-body load history: pressure + weight + primary inertia (+ optional static part).

Per spectral component k the record keeps COMPLEX unit-reference nodal loads for each physical contribution on one
stationary mesh, exp(-i w t) convention, together with the realization (amplitude, phase, omega).  Components stay
separate so pressure, weight perturbation and primary inertial loads can be inspected independently; ``evaluate``
returns their real sum at an actual time.  The residual-IR correction is NOT stored here (it is a solve product).
"""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
import numpy as np

TYPE = 'floating_coupled_load_history_v1'
CONTRIBUTIONS = ('pressure', 'weight', 'inertia')
BINDING_KEYS = ('source', 'equilibrium', 'geometry', 'mass', 'backend', 'dofs', 'axes', 'units', 'reference',
                'phase', 'realization')


def _freeze(value):
    """Recursively read-only: mappings -> MappingProxyType, sequences -> tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value):
    """Plain JSON-able copy of a frozen structure."""
    if isinstance(value, Mapping):
        return {str(k): _thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(v) for v in value]
    return value


def _rebuild(state):
    """Pickle/transport reconstruction: the constructor re-freezes every array and re-verifies the content identity."""
    return FloatingCoupledLoadHistory(**state)


def _ro(value, dtype, name, shape=None):
    out = np.array(value, dtype=dtype, copy=True)
    if shape is not None and out.shape != shape:
        raise ValueError(f'{name} must have shape {shape}')
    if not np.all(np.isfinite(out)):
        raise ValueError(f'{name} must be finite')
    return np.frombuffer(out.tobytes(), dtype=out.dtype).reshape(out.shape)   # bytes-backed, cannot be made writable


@dataclass(frozen=True)
class FloatingCoupledLoadHistory:
    omega: np.ndarray                 # (nk,) positive
    amplitude: np.ndarray             # (nk,) >= 0
    phase: np.ndarray                 # (nk,) rad
    loads: dict                       # contribution -> (nk, n_nodes, 6) complex, per unit amplitude, global axes
    static_load: np.ndarray           # (n_nodes, 6) real: p0 + dry weight (zero-amplitude realization is static)
    node_ids: tuple
    mesh_content_sha256: str
    binding: dict
    content_sha256: str = ''
    times: tuple = ()                 # selected ABSOLUTE sample times of the project load series (increasing)
    selected_window: tuple = ()       # (start, end) absolute seconds of the selection (empty = not a project load yet)
    payload: bytes = b''              # self-contained prepared record (npz bytes: Hq, pose, source, families, settings, keys)

    def __post_init__(self):
        ts = tuple(float(t) for t in self.times)
        if any(not math.isfinite(t) for t in ts) or any(b <= a for a, b in zip(ts, ts[1:])):
            raise ValueError('floating history times must be finite and strictly increasing')
        object.__setattr__(self, 'times', ts)
        win = tuple(float(t) for t in self.selected_window)
        if win and (len(win) != 2 or win[1] < win[0] or (ts and not (win[0] <= ts[0] and ts[-1] <= win[1]))):
            raise ValueError('selected window must be (start, end) containing the selected times')
        object.__setattr__(self, 'selected_window', win)
        if not isinstance(self.payload, (bytes, bytearray)):
            raise ValueError('payload must be bytes')
        object.__setattr__(self, 'payload', bytes(self.payload))
        w = _ro(self.omega, np.float64, 'omega')
        if w.ndim != 1 or w.size == 0 or np.any(w <= 0):
            raise ValueError('floating history needs positive finite frequencies (no static/zero frequency component)')
        nk = w.size
        object.__setattr__(self, 'omega', w)
        a = _ro(self.amplitude, np.float64, 'amplitude', (nk,))
        if np.any(a < 0):
            raise ValueError('amplitudes must be non-negative')
        object.__setattr__(self, 'amplitude', a)
        object.__setattr__(self, 'phase', _ro(self.phase, np.float64, 'phase', (nk,)))
        ids = tuple(int(x) for x in self.node_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('floating history needs unique node identifiers')
        object.__setattr__(self, 'node_ids', ids)
        n = len(ids)
        if set(self.loads) != set(CONTRIBUTIONS):
            raise ValueError(f'loads must carry exactly {CONTRIBUTIONS}')
        object.__setattr__(self, 'loads', MappingProxyType(
            {c: _ro(self.loads[c], np.complex128, c, (nk, n, 6)) for c in CONTRIBUTIONS}))
        object.__setattr__(self, 'static_load', _ro(self.static_load, np.float64, 'static_load', (n, 6)))
        import re
        if re.fullmatch('(sha256:)?[0-9a-f]{64}', self.mesh_content_sha256) is None:
            raise ValueError('floating history needs an exact mesh identity')
        missing = [k for k in BINDING_KEYS if k not in self.binding or self.binding[k] in (None, '', {})]
        if missing:
            raise ValueError(f'floating history binding lacks {missing}')
        # Deep-immutable binding: JSON-normalised, then read-only mappings and tuples all the way down.
        object.__setattr__(self, 'binding', _freeze(json.loads(json.dumps(_thaw(self.binding), sort_keys=True, allow_nan=False))))
        expected = self._digest()
        if self.content_sha256 and self.content_sha256 != expected:
            raise ValueError('floating history content identity mismatch')
        object.__setattr__(self, 'content_sha256', expected)

    def __reduce__(self):
        state = dict(omega=np.array(self.omega), amplitude=np.array(self.amplitude), phase=np.array(self.phase),
                     loads={c: np.array(a) for c, a in self.loads.items()}, static_load=np.array(self.static_load),
                     node_ids=self.node_ids, mesh_content_sha256=self.mesh_content_sha256, binding=_thaw(self.binding),
                     content_sha256=self.content_sha256, times=self.times, selected_window=self.selected_window,
                     payload=self.payload)
        return (_rebuild, (state,))

    @property
    def array_sha256(self):
        """Identity of the load arrays + binding only (independent of the selected times and the embedded payload)."""
        h = hashlib.sha256()
        h.update(json.dumps({'type': TYPE, 'binding': _thaw(self.binding), 'node_ids': self.node_ids,
                             'mesh': self.mesh_content_sha256}, sort_keys=True, separators=(',', ':')).encode())
        for arr in (self.omega, self.amplitude, self.phase, self.static_load,
                    *(self.loads[c] for c in CONTRIBUTIONS)):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    def _digest(self):
        h = hashlib.sha256(self.array_sha256.encode())
        h.update(json.dumps({'times': self.times, 'window': self.selected_window,
                             'payload': hashlib.sha256(self.payload).hexdigest()}, separators=(',', ':')).encode())
        return h.hexdigest()

    def with_selection(self, times, selected_window):
        """Same immutable arrays/payload, other absolute selected times (selected-window project load)."""
        from dataclasses import replace
        return replace(self, times=tuple(times), selected_window=tuple(selected_window), content_sha256='')

    @classmethod
    def from_payload(cls, payload: bytes, *, times=(), selected_window=(), content_sha256=''):
        """Rebuild (and fully validate) the history from an embedded prepared record (npz bytes)."""
        import io
        with np.load(io.BytesIO(payload), allow_pickle=False) as d:
            meta = json.loads(str(d['meta']))
            obj = cls(d['omega'], d['amplitude'], d['phase'],
                      dict(pressure=d['pressure'], weight=d['weight'], inertia=d['inertia']), d['static'],
                      tuple(meta['node_ids']), meta['mesh_content_sha256'], meta['binding'], '', times, selected_window,
                      bytes(payload))
        if meta.get('array_sha256') != obj.array_sha256:
            raise ValueError('embedded floating record arrays do not match their recorded identity')
        if content_sha256 and content_sha256 != obj.content_sha256:
            raise ValueError('floating history content identity mismatch')
        return obj

    @property
    def is_static(self):
        return bool(np.all(self.amplitude == 0.0))

    def _phasor(self, time):
        t = float(time)
        if not math.isfinite(t):
            raise ValueError('floating history time must be finite')
        return self.amplitude * np.exp(1j * self.phase) * np.exp(-1j * self.omega * t)

    def evaluate(self, time, contributions=CONTRIBUTIONS, include_static=True):
        """Real nodal loads (n_nodes, 6) at ``time``; static weight/p0 part optional."""
        z = self._phasor(time)
        out = np.zeros_like(self.static_load)
        for c in contributions:
            if c not in CONTRIBUTIONS:
                raise ValueError('unknown contribution')
            out = out + np.real(np.tensordot(z, self.loads[c], axes=(0, 0)))
        return out + self.static_load if include_static else out

    def evaluate_component(self, time, contribution):
        return self.evaluate(time, (contribution,), include_static=False)

    def to_dict(self):
        """Persistable record: the embedded self-contained payload (zlib+base64) plus selection and identity."""
        if not self.payload:
            raise ValueError('a project floating load must embed its prepared payload')
        import base64, zlib
        return {'type': TYPE, 'times': list(self.times), 'selected_window': list(self.selected_window),
                'mesh_content_sha256': self.mesh_content_sha256, 'content_sha256': self.content_sha256,
                'payload_b64': base64.b64encode(zlib.compress(self.payload, 6)).decode('ascii')}

    @classmethod
    def from_dict(cls, data):
        import base64, zlib
        if not isinstance(data, dict) or data.get('type') != TYPE or not data.get('content_sha256'):
            raise ValueError('unsupported or unidentified floating history record')
        obj = cls.from_payload(zlib.decompress(base64.b64decode(data['payload_b64'])), times=data['times'],
                               selected_window=data['selected_window'], content_sha256=data['content_sha256'])
        if obj.mesh_content_sha256 != data['mesh_content_sha256']:
            raise ValueError('floating history mesh identity mismatch')
        return obj
