"""Compact self-contained spectral histories, using the shared pressure kernel."""
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
import numpy as np
from .sea_state import WaveRealization
from .pressure_series import compile_pressure_series, CONVENTIONS
from .hydro_pressure import transfer_panel_pressure, pressure_wrench

TYPE = 'spectral_mapped_pressure_v1'
ALGORITHM = 'same_surface_nearest_panel_v1'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _freeze(value, dtype=float):
    a=np.asarray(value,dtype=dtype)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


def _mapped_bank_identity(source_id,mesh_id,ids,omegas,responses,heading):
    h=hashlib.sha256(_json(dict(source_bank_id=source_id,mesh_content_sha256=mesh_id,
        algorithm=ALGORITHM,panel_ids=ids,heading=heading,conventions=CONVENTIONS,
        phase_origin=(0.,0.,0.))).encode())
    for a in (omegas,responses):
        h.update(_json((a.shape,a.dtype.str)).encode());h.update(memoryview(a).cast('B'))
    return h.hexdigest()


@dataclass(frozen=True,init=False)
class PressureMappingPlan:
    source_bank_id: str
    mesh_content_sha256: str
    target_geometry_sha256: str
    element_ids: tuple
    panel_indices: tuple
    surface_ids: tuple
    max_distance: float
    algorithm: str = ALGORITHM
    plan_id: str = ''

    def __init__(self,*args,**kwargs):
        raise TypeError('PressureMappingPlan is factory-only; use build_pressure_mapping_plan')

    def validate(self):
        values={name:getattr(self,name) for name in ('source_bank_id','mesh_content_sha256',
            'target_geometry_sha256','element_ids','panel_indices','surface_ids','max_distance','algorithm')}
        if _digest(values)!=self.plan_id: raise ValueError('pressure mapping plan identity changed')


class _BankSourceView:
    """Explicit structural bank view, never a RegularWavePressure producer."""
    def __init__(self, bank): self.bank=bank
    def __getattr__(self,name): return getattr(self.bank,name)
    @property
    def pressure(self): return self.bank.responses[0]
    @property
    def omega(self): return float(self.bank.omegas[0])
    @property
    def source_id(self): return self.bank.bank_id
    def validate(self): self.bank.validate()


def _target_hash(element_ids,centers,normals,areas,surface_ids):
    return _digest(dict(element_ids=list(element_ids),centers=np.asarray(centers).tolist(),
        normals=np.asarray(normals).tolist(),areas=np.asarray(areas).tolist(),surface_ids=list(surface_ids)))


def build_pressure_mapping_plan(bank,element_ids,centers,normals,areas,surface_ids,*,mesh_content_sha256):
    """Perform spatial search exactly once, using existing admitted transfer checks."""
    bank.validate()
    transfer=transfer_panel_pressure(_BankSourceView(bank),element_ids,centers,normals,areas,surface_ids)
    values=dict(source_bank_id=bank.bank_id,mesh_content_sha256=mesh_content_sha256,
        target_geometry_sha256=_target_hash(element_ids,centers,normals,areas,surface_ids),
        element_ids=transfer.element_ids,panel_indices=tuple(int(x) for x in transfer.panel_indices),
        surface_ids=tuple(surface_ids),max_distance=transfer.max_distance,algorithm=ALGORITHM)
    plan=object.__new__(PressureMappingPlan)
    for name,value in values.items(): object.__setattr__(plan,name,value)
    object.__setattr__(plan,'plan_id',_digest(values));plan.validate()
    return plan


def _error(source,target):
    absolute=float(np.linalg.norm(target-source)); denominator=float(np.linalg.norm(source))
    return dict(absolute=absolute,reference_norm=denominator,
        relative=None if denominator==0. else absolute/denominator,zero_reference=denominator==0.)


def map_pressure_bank(bank,plan,centers,normals,areas,*,surface_ids,mesh_content_sha256):
    bank.validate()
    plan.validate()
    if (plan.source_bank_id!=bank.bank_id or plan.mesh_content_sha256!=mesh_content_sha256 or
            plan.algorithm!=ALGORITHM or plan.target_geometry_sha256!=_target_hash(
                plan.element_ids,centers,normals,areas,surface_ids)):
        raise ValueError('pressure mapping plan source/target identity changed')
    mapped=bank.responses[:,plan.panel_indices]
    evidence=[]
    for omega,source_pressure,target_pressure in zip(bank.omegas,bank.responses,mapped):
        sw=pressure_wrench(bank.centers,bank.normals,bank.areas,source_pressure)
        tw=pressure_wrench(centers,normals,areas,target_pressure)
        evidence.append(dict(omega=float(omega),source_real=sw.real.tolist(),source_imag=sw.imag.tolist(),
            target_real=tw.real.tolist(),target_imag=tw.imag.tolist(),force=_error(sw[:3],tw[:3]),
            moment=_error(sw[3:],tw[3:])))
    return mapped,tuple(evidence)


@dataclass(frozen=True,init=False)
class _MappedBank:
    source_bank_id: str
    mesh_content_sha256: str
    panel_ids: tuple
    omegas: np.ndarray
    responses: np.ndarray
    heading: float
    bank_id: str
    phase_origin: tuple = (0.,0.,0.)
    _validated_identity: str = field(default='',init=False,repr=False)
    def __init__(self,*args,**kwargs):
        raise TypeError('mapped banks are private immutable history projections')
    @property
    def conventions(self): return dict(CONVENTIONS)
    def validate(self):
        # Histories own immutable bytes-backed fields; verify once at ingestion,
        # then retain that proof without reserializing every frequency per time.
        if (self.bank_id!=self._validated_identity or self.omegas.flags.writeable or
                self.responses.flags.writeable):
            raise ValueError('mapped response bank changed after compilation')


@dataclass(frozen=True)
class SpectralMappedPressureHistory:
    times: tuple
    element_ids: tuple
    omegas: tuple
    real: tuple
    imag: tuple
    heading: float
    mesh_content_sha256: str
    source_id: str
    realization_json: str
    provenance: str = ''
    content_sha256: str = ''
    working_memory_bytes: int = 256*1024**2
    _compiled: object = field(default=None,init=False,repr=False,compare=False)

    def __post_init__(self):
        for name in ('times','omegas'):
            raw=tuple(getattr(self,name))
            if any(isinstance(x,(bool,np.bool_,complex,np.complexfloating)) for x in raw):
                raise ValueError('real numeric times/frequencies required')
            object.__setattr__(self,name,tuple(float(x) for x in raw))
        ids=tuple(self.element_ids)
        if not ids or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) for x in ids) or len(set(ids))!=len(ids):
            raise ValueError('unique integer mapped element IDs required')
        object.__setattr__(self,'element_ids',tuple(int(x) for x in ids))
        for name in ('real','imag'):
            raw=np.asarray(getattr(self,name))
            if raw.dtype.kind not in 'fiu' or raw.shape!=(len(self.omegas),len(ids)) or not np.isfinite(raw).all():
                raise ValueError('finite frequency-by-element response fields required')
            object.__setattr__(self,name,tuple(tuple(float(x) for x in row) for row in raw))
        if (not self.times or not self.omegas or not np.isfinite(self.times).all() or
                not np.isfinite(self.omegas).all() or min(self.omegas)<=0 or len(set(self.omegas))!=len(self.omegas) or
                any(b<=a for a,b in zip(self.times,self.times[1:])) or not math.isfinite(self.heading) or
                not 0<=self.heading<math.tau or re.fullmatch('(sha256:)?[0-9a-f]{64}',self.mesh_content_sha256) is None or
                re.fullmatch('[0-9a-f]{64}',self.source_id) is None):
            raise ValueError('admitted time range, heading and exact source/mesh identities required')
        realization=WaveRealization.from_dict(json.loads(self.realization_json))
        object.__setattr__(self,'realization_json',_json(realization.to_dict()))
        identity=_digest(self._payload())
        if self.content_sha256 and self.content_sha256!=identity: raise ValueError('spectral history content identity mismatch')
        object.__setattr__(self,'content_sha256',identity)
        # Validate the exact component inventory now, and cache compilation once.
        bank=self._bank()
        object.__setattr__(self,'_compiled',compile_pressure_series(bank,realization,
            working_memory_bytes=self.working_memory_bytes,bank_reference='self-contained:'+identity))

    def _payload(self):
        return {name:getattr(self,name) for name in ('times','element_ids','omegas','real','imag','heading',
            'mesh_content_sha256','source_id','realization_json','provenance','working_memory_bytes')}

    def _bank(self):
        omegas=_freeze(self.omegas);responses=_freeze(np.asarray(self.real)+1j*np.asarray(self.imag),complex)
        identity=_mapped_bank_identity(self.source_id,self.mesh_content_sha256,self.element_ids,
            omegas,responses,self.heading)
        values=dict(source_bank_id=self.source_id,mesh_content_sha256=self.mesh_content_sha256,
            panel_ids=self.element_ids,omegas=omegas,responses=responses,heading=self.heading,
            bank_id=identity,phase_origin=(0.,0.,0.),_validated_identity=identity)
        bank=object.__new__(_MappedBank)
        for name,value in values.items():object.__setattr__(bank,name,value)
        return bank

    def to_dict(self):
        return dict(type=TYPE,**self._payload(),content_sha256=self.content_sha256)

    @classmethod
    def from_dict(cls,data):
        data=dict(data)
        if data.pop('type',None)!=TYPE or not data.get('content_sha256'):
            raise ValueError('unsupported or unverified spectral pressure history')
        return cls(**data)

    def evaluate(self,time,element_ids=None):
        if (isinstance(time,(bool,np.bool_,complex,np.complexfloating)) or
                not isinstance(time,(int,float,np.integer,np.floating))):
            raise ValueError('pressure history evaluation requires a finite real scalar time')
        try:actual_time=float(time)
        except (TypeError,ValueError,OverflowError) as error:
            raise ValueError('pressure history evaluation requires a finite real scalar time') from error
        if not math.isfinite(actual_time) or not self.times[0]<=actual_time<=self.times[-1]:
            raise ValueError('pressure history time is outside its admitted time range')
        ids=self.element_ids if element_ids is None else tuple(int(x) if isinstance(x,np.integer) else x for x in element_ids)
        return self._compiled.snapshot(actual_time,panel_ids=ids).pressures
