"""Bounded pressure synthesis from a structural, data-only response-bank API.

Working budget includes compiled coefficient arrays, selected-index arrays,
time input copy and all NumPy block temporaries/output. Bank input storage and
previous blocks retained by the caller are outside the evaluator's workspace.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np
from .errors import LoadError
from .sea_state import WaveRealization

VERSION = 'pressure_series_v1'
CONVENTIONS = dict(units='Pa/m', phasor='exp(-i omega t)', normals='outward',
    pressure='positive inward', physics='fixed incident+diffraction')


def _budget(value):
    if type(value) is not int or value <= 0:
        raise LoadError('working memory budget must be a positive integer byte count')
    return value


def _frozen(value):
    a = np.asarray(value, dtype=np.float64)
    return np.frombuffer(a.tobytes(), dtype=np.float64).reshape(a.shape)


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class PressureBlock:
    times: np.ndarray
    panel_ids: tuple
    pressures: np.ndarray
    time_slice: slice
    panel_slice: slice
    bank_id: str = ''


@dataclass(frozen=True)
class PressureSnapshot:
    time: float
    panel_ids: tuple
    pressures: np.ndarray
    bank_id: str = ''

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.pressures, dtype=dtype)


class PressureSeriesCancelled(RuntimeError):
    pass


@dataclass(frozen=True, init=False)
class PressureSeries:
    bank_id: str
    bank_reference: str
    realization_json: str
    phase_origin: tuple
    panel_ids: tuple
    omegas: np.ndarray
    coefficients_real: np.ndarray
    coefficients_imag: np.ndarray
    working_memory_bytes: int
    _bank: object

    def __init__(self, *args, **kwargs):
        raise TypeError('PressureSeries is factory-only; use compile_pressure_series or load')

    @property
    def realization(self):
        return WaveRealization.from_dict(json.loads(self.realization_json))

    @property
    def coefficient_bytes(self):
        return self.omegas.nbytes+self.coefficients_real.nbytes+self.coefficients_imag.nbytes

    def validate(self):
        self._bank.validate()
        if self._bank.bank_id != self.bank_id:
            raise LoadError('compiled pressure series bank identity changed')

    def iter_blocks(self, times, *, panel_ids=None, time_block_size=None,
                    panel_block_size=None, cancel=None):
        """Yield owned outputs in time-major/panel-minor order, cancel between blocks.

        Requested block dimensions are maxima, reduced to meet the budget.
        Each block pressure shape is (len(times), len(panel_ids)). No full
        history is implicitly assembled. Caller-owned output accumulation is
        deliberately outside this working-memory contract.
        """
        self.validate()
        # Conversion from a list creates only one array. Existing arrays are
        # caller-owned, but conservatively charged against workspace as well.
        if not isinstance(times, np.ndarray):
            try:
                if self.coefficient_bytes+8*len(times) > self.working_memory_bytes:
                    raise LoadError('time input exceeds working budget')
                if any(isinstance(x, (bool, np.bool_, complex, np.complexfloating)) for x in times):
                    raise LoadError('real time coordinates required, excluding booleans')
            except TypeError as error:
                raise LoadError('time coordinates must be an array or sequence') from error
        try:
            raw = np.asarray(times)
        except (ValueError, TypeError) as error:
            raise LoadError('finite real time coordinates required') from error
        if raw.ndim != 1 or not raw.size or raw.dtype.kind not in 'fiu':
            raise LoadError('nonempty finite strictly increasing time array required')
        # Refuse before allocating the owned input copy.
        ids = self.panel_ids if panel_ids is None else tuple(panel_ids)
        if (not ids or any(type(x) is not int for x in ids) or len(set(ids)) != len(ids)):
            raise LoadError('unique integer source panel IDs required')
        for value in (time_block_size, panel_block_size):
            if value is not None and (type(value) is not int or value <= 0):
                raise LoadError('positive integer block dimensions required')
        k = len(self.omegas)
        base = self.coefficient_bytes+max(8, raw.itemsize)*raw.size+8*len(ids)
        # Per block: selected coefficients(16*k*p), trig(16*t*k),
        # two matrix products(16*t*p). Arrays are built with out= operations.
        def needed(t, p): return base+16*k*p+16*t*k+17*t*p
        if needed(1, 1) > self.working_memory_bytes:
            raise LoadError('working budget cannot fit one minimal pressure block')
        lookup = {x: i for i, x in enumerate(self.panel_ids)}
        try:
            indices = np.fromiter((lookup[x] for x in ids), dtype=np.int64, count=len(ids))
        except KeyError as error:
            raise LoadError('unknown source panel ID') from error
        # Keep integer/float input dtype; ufunc writes doubles directly, avoiding
        # a second history-sized conversion array.
        tvalues = raw
        # Scalar comparisons avoid allocating a full diff or finite mask.
        if any(not math.isfinite(float(x)) for x in tvalues) or any(
                b <= a for a, b in zip(tvalues[:-1], tvalues[1:])):
            raise LoadError('finite strictly increasing times required')
        p = min(len(ids), panel_block_size or len(ids))
        while p > 1 and needed(1, p) > self.working_memory_bytes: p = max(1, p//2)
        t = min(len(tvalues), time_block_size or 1024,
            (self.working_memory_bytes-base-16*k*p)//(16*k+17*p))
        if t < 1: raise LoadError('working budget cannot fit requested block')
        for first in range(0, len(tvalues), t):
            stop = min(first+t, len(tvalues)); ts = tvalues[first:stop]
            largest_time = max(abs(float(x)) for x in ts)
            if k and largest_time > 1. and float(self.omegas.max()) > np.finfo(float).max/largest_time:
                raise LoadError('time-frequency product exceeds finite synthesis range')
            # Allocate and reuse trigonometry across selected panel blocks.
            angles = np.empty((len(ts), k)); np.multiply(ts[:, None], self.omegas, out=angles)
            sine = np.sin(angles); np.cos(angles, out=angles)
            for begin in range(0, len(ids), p):
                if cancel is not None and cancel():
                    raise PressureSeriesCancelled('pressure synthesis cancelled between blocks')
                end = min(begin+p, len(ids)); ix = indices[begin:end]
                cr = self.coefficients_real[:, ix]; ci = self.coefficients_imag[:, ix]
                output = angles @ cr
                temporary = sine @ ci
                np.add(output, temporary, out=output)
                if not np.isfinite(output).all():
                    raise LoadError('nonfinite synthesized pressure')
                del temporary, cr, ci
                yield PressureBlock(ts, ids[begin:end], output, slice(first, stop),
                    slice(begin, end), self.bank_id)
                # Do not retain the previously yielded output during next allocation.
                del output
            del angles, sine

    def snapshot(self, time, *, panel_ids=None):
        """One snapshot; explicit allocation requires room for the assembled row."""
        ids = self.panel_ids if panel_ids is None else tuple(panel_ids)
        if self.coefficient_bytes+8*len(ids)*2+8+32*len(self.omegas)+17 > self.working_memory_bytes:
            raise LoadError('snapshot output does not fit working budget; use iter_blocks')
        # Reserve the assembled output from the block workspace.
        bounded = _with_budget(self, self.working_memory_bytes-8*len(ids))
        output = np.empty(len(ids))
        for block in bounded.iter_blocks([time], panel_ids=ids):
            output[block.panel_slice] = block.pressures[0]
        return PressureSnapshot(float(time), ids, output, self.bank_id)

    def save(self, path, *, bank_reference=None):
        self.validate()
        reference = self.bank_reference if bank_reference is None else str(bank_reference)
        if not reference.strip():
            raise LoadError('saved pressure series requires an explicit bank reference')
        data = dict(version=VERSION, bank_id=self.bank_id,
            bank_reference=reference,
            realization=json.loads(self.realization_json), phase_origin=self.phase_origin,
            working_memory_bytes=self.working_memory_bytes)
        data['content_hash'] = hashlib.sha256(_canonical_json(data).encode()).hexdigest()
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name+'.')
        try:
            with os.fdopen(fd, 'w', encoding='utf8') as stream:
                stream.write(_canonical_json(data)); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)

    @classmethod
    def load(cls, path, *, bank_loader):
        try:
            data = json.loads(Path(path).read_text(encoding='utf8'))
            digest = data.pop('content_hash')
            if (hashlib.sha256(_canonical_json(data).encode()).hexdigest() != digest or
                    data.pop('version') != VERSION):
                raise LoadError('corrupt/incompatible pressure series')
            expected = data.pop('bank_id'); reference = data.pop('bank_reference')
            if type(reference) is not str or not reference.strip():
                raise LoadError('saved pressure series requires an explicit bank reference')
            bank = bank_loader(reference); bank.validate()
            if bank.bank_id != expected:
                raise LoadError('saved pressure series resolves to a changed bank')
            return compile_pressure_series(bank, WaveRealization.from_dict(data.pop('realization')),
                bank_reference=reference, **data)
        except LoadError: raise
        except Exception as error:
            raise LoadError('corrupt pressure series or bank resolution failure') from error


def _create_series(bank_id, bank_reference, realization_json, phase_origin, panel_ids,
        omegas, coefficients_real, coefficients_imag, working_memory_bytes, bank):
    result = object.__new__(PressureSeries)
    values = dict(bank_id=bank_id, bank_reference=bank_reference,
        realization_json=realization_json, phase_origin=tuple(phase_origin), panel_ids=tuple(panel_ids),
        omegas=_frozen(omegas), coefficients_real=_frozen(coefficients_real),
        coefficients_imag=_frozen(coefficients_imag), working_memory_bytes=working_memory_bytes,
        _bank=bank)
    for name, value in values.items(): object.__setattr__(result, name, value)
    return result


def _with_budget(series, budget):
    # Internal snapshot output reservation: share only already immutable fields.
    from dataclasses import fields
    result = object.__new__(PressureSeries)
    for field in fields(PressureSeries):
        object.__setattr__(result, field.name, budget if field.name == 'working_memory_bytes'
            else getattr(series, field.name))
    return result


def compile_pressure_series(bank, realization, *, working_memory_bytes=256*1024**2,
                            phase_origin=None, bank_reference=''):
    """Compile C=H*a*exp(i phi) once; exact frequencies, propagation heading.

    Banks implement validate(), bank_id, omegas, responses, heading (radians),
    panel_ids and phase_origin; no ANYhydro import is required.
    Zero-amplitude components need no bank response; duplicates sum coherently.
    """
    return _compile_response_series(bank,realization,working_memory_bytes=working_memory_bytes,
        phase_origin=phase_origin,bank_reference=bank_reference,expected_conventions=CONVENTIONS)


def _compile_response_series(bank,realization,*,working_memory_bytes=256*1024**2,
        phase_origin=None,bank_reference='',expected_conventions):
    """Shared internal kernel admission for typed pressure/resultant projections."""
    budget = _budget(working_memory_bytes)
    if type(realization) is not WaveRealization:
        raise LoadError('named materialized WaveRealization required')
    try:
        bank.validate(); identity = bank.bank_id
        if bank.conventions != expected_conventions:
            raise LoadError('unsupported pressure-bank units/phasor/physics conventions')
        origin = (0., 0., 0.) if phase_origin is None else tuple(phase_origin)
        if origin != (0., 0., 0.) or origin != tuple(bank.phase_origin):
            raise LoadError('realization phase origin differs from pressure bank')
        heading = float(bank.heading) % math.tau
        frequencies = tuple(float(w) for w in bank.omegas)
        if len(set(frequencies)) != len(frequencies): raise LoadError('duplicate bank frequencies')
        weights = {}
        for c in realization.components:
            if c.amplitude == 0: continue
            component_heading = math.radians(c.direction_degrees % 360.)
            # The legacy component boundary uses degrees. Round-trip conversion
            # may move a heading by a few ULPs; admit only that representation
            # error, preserving circular equivalence and exact frequencies.
            if abs(math.remainder(component_heading-heading, math.tau)) > 4*math.ulp(math.tau):
                raise LoadError('component propagation heading differs from pressure bank')
            if c.omega not in frequencies:
                raise LoadError('exact component frequency absent from pressure bank')
            weights[c.omega] = weights.get(c.omega, 0j)+c.amplitude*complex(math.cos(c.phase), math.sin(c.phase))
        active = tuple(w for w in frequencies if w in weights)
        n = len(bank.panel_ids); k = len(active)
        # Conservative compilation peak includes complex accumulation, immutable
        # real/imag copies and transient bytes. Input bank storage is caller-owned.
        if 64*k*n+16*n+8*k+48*k+8*n+32 > budget:
            raise LoadError('compilation workspace exceeds working budget')
        coefficients = np.empty((k, n), dtype=np.complex128)
        for row, w in enumerate(active):
            np.multiply(bank.responses[frequencies.index(w)], weights[w], out=coefficients[row])
        if not np.isfinite(coefficients).all(): raise LoadError('nonfinite compiled coefficients')
        result = _create_series(identity, str(bank_reference), _canonical_json(realization.to_dict()),
            origin, tuple(bank.panel_ids), active, coefficients.real,
            coefficients.imag, budget, bank)
        result.validate()
        return result
    except LoadError: raise
    except Exception as error:
        raise LoadError(f'invalid pressure bank: {error}') from error
