"""Compact immutable harmonic pressure history on an exact element mesh."""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np


def pressure_history_from_dict(data):
    """Strict additive dispatcher retaining harmonic v1 records unchanged."""
    if not isinstance(data,dict): raise ValueError('typed pressure history record required')
    if data.get('type')=='harmonic_mapped_pressure_v1':
        return MappedPressureHistory.from_dict(data)
    if data.get('type')=='spectral_mapped_pressure_v1':
        from .spectral_pressure_history import SpectralMappedPressureHistory
        return SpectralMappedPressureHistory.from_dict(data)
    if data.get('type') in ('source_aware_mapped_pressure_v1','source_aware_mapped_pressure_v2','source_aware_mapped_pressure_v3'):
        from .source_pressure_history import SourceAwarePressureHistory
        return SourceAwarePressureHistory.from_dict(data)
    if data.get('type')=='wave_surface_pressure_v1':
        from .wave_surface_pressure import WaveSurfacePressureHistory
        return WaveSurfacePressureHistory.from_dict(data)
    if data.get('type')=='floating_coupled_load_history_v1':
        from .floating_history import FloatingCoupledLoadHistory
        return FloatingCoupledLoadHistory.from_dict(data)
    raise ValueError('unsupported mapped pressure history type')


@dataclass(frozen=True)
class MappedPressureHistory:
    times: tuple[float, ...]
    element_ids: tuple[int, ...]
    real: tuple[float, ...]
    imag: tuple[float, ...]
    omega: float
    mesh_content_sha256: str
    source_id: str
    provenance: str = ''
    content_sha256: str = ''

    def __post_init__(self):
        for key in ('times','real','imag'):
            object.__setattr__(self,key,tuple(float(x) for x in getattr(self,key)))
        raw=tuple(self.element_ids)
        if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) for x in raw):
            raise ValueError('history element identifiers must be integers')
        object.__setattr__(self,'element_ids',tuple(int(x) for x in raw))
        if (not self.times or not self.element_ids or len(set(raw))!=len(raw)
                or len(self.real)!=len(raw) or len(self.imag)!=len(raw)):
            raise ValueError('history needs unique elements, one complex pressure each and times')
        if (not all(math.isfinite(x) for x in (*self.times,*self.real,*self.imag,self.omega))
                or self.omega<=0 or any(b<=a for a,b in zip(self.times,self.times[1:]))):
            raise ValueError('history values must be finite with positive frequency and increasing times')
        import re
        if re.fullmatch('(sha256:)?[0-9a-f]{64}',self.mesh_content_sha256) is None or not self.source_id:
            raise ValueError('history needs exact mesh and source identities')
        expected=self._digest()
        if self.content_sha256 and self.content_sha256!=expected:
            raise ValueError('mapped pressure history content identity mismatch')
        object.__setattr__(self,'content_sha256',expected)

    def _payload(self):
        return {key:getattr(self,key) for key in ('times','element_ids','real','imag','omega',
            'mesh_content_sha256','source_id','provenance')}

    def _digest(self):
        return hashlib.sha256(json.dumps(self._payload(),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    def to_dict(self):
        return {'type':'harmonic_mapped_pressure_v1',**self._payload(),'content_sha256':self.content_sha256}

    @classmethod
    def from_dict(cls,data):
        data=dict(data)
        if data.pop('type',None)!='harmonic_mapped_pressure_v1':
            raise ValueError('unsupported mapped pressure history type')
        if not data.get('content_sha256'):
            raise ValueError('persisted pressure history lacks content identity')
        return cls(**data)

    def evaluate(self,time,element_ids=None):
        if time is None or not math.isfinite(float(time)) or not self.times[0]<=float(time)<=self.times[-1]:
            raise ValueError('pressure history time is outside its admitted time range')
        # Analytic harmonic interpolation uses the source frequency at actual
        # solver case.time; no nearest-time selection or silent extrapolation.
        values=np.real((np.asarray(self.real)+1j*np.asarray(self.imag))*np.exp(-1j*self.omega*float(time)))
        if element_ids is None:
            return values
        if any(isinstance(e,(bool,np.bool_)) or not isinstance(e,(int,np.integer)) for e in element_ids):
            raise ValueError('history evaluation requires integer element identifiers')
        position={e:i for i,e in enumerate(self.element_ids)}
        if any(int(e) not in position for e in element_ids):
            raise ValueError('unknown element in mapped pressure history evaluation')
        return values[[position[int(e)] for e in element_ids]]
