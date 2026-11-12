"""Immutable prescribed pressure-only nodal transfers, independent of solvers."""
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
import numpy as np
from .sea_state import WaveRealization

TYPE='prescribed_pressure_history_v1'


def _json(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _ro(x,dtype=complex):
    a=np.asarray(x,dtype=dtype)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class PrescribedPressureHistory:
    """Complex unit-wave-amplitude loads, shaped frequency x node x six DOFs.

    Source payload and binding are retained portable JSON. Geometry and solver
    owners validate their contents when applying or remapping, not this package.
    Mixed pressure/jump banks retain their per-panel semantics in source payload;
    all sheet contributions have already been applied once with orientation in
    the nodal transfer. Time synthesis must not reinterpret or double them.
    """
    times: tuple
    node_ids: tuple
    omegas: tuple
    responses: np.ndarray
    realization_json: str
    mesh_content_sha256: str
    source_payload_json: str
    binding_json: str
    content_sha256: str=''
    working_memory_bytes: int=256*1024**2
    _weights: np.ndarray=field(default=None,init=False,repr=False,compare=False)

    def __post_init__(self):
        for name in ('times','omegas'):
            raw=tuple(getattr(self,name))
            if any(isinstance(x,(bool,complex,np.bool_,np.complexfloating)) for x in raw):
                raise ValueError('real times and frequencies required')
            object.__setattr__(self,name,tuple(float(x) for x in raw))
        ids=tuple(self.node_ids)
        if not ids or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) for x in ids) or len(set(ids))!=len(ids):
            raise ValueError('unique integer nodal IDs required')
        object.__setattr__(self,'node_ids',tuple(int(x) for x in ids))
        z=np.asarray(self.responses)
        if z.dtype.kind not in 'fiuc' or z.shape!=(len(self.omegas),len(ids),6) or not np.isfinite(z).all():
            raise ValueError('finite frequency x node x six pressure-only transfers required')
        object.__setattr__(self,'responses',_ro(z))
        if (not self.times or not self.omegas or not np.isfinite(self.times).all() or not np.isfinite(self.omegas).all()
            or min(self.omegas)<=0 or any(b<=a for a,b in zip(self.omegas,self.omegas[1:]))
            or any(b<=a for a,b in zip(self.times,self.times[1:]))
            or re.fullmatch('(sha256:)?[a-f0-9]{64}',self.mesh_content_sha256) is None
            or type(self.working_memory_bytes) is not int or self.working_memory_bytes<=0):
            raise ValueError('invalid history times, frequencies, mesh identity or memory budget')
        for name in ('realization_json','source_payload_json','binding_json'):
            value=json.loads(getattr(self,name))
            if not isinstance(value,dict): raise ValueError('object metadata required')
            object.__setattr__(self,name,_json(value))
        realization=WaveRealization.from_dict(json.loads(self.realization_json))
        if len(self.times)>1 and max(np.diff(self.times))>=math.pi/max(c.omega for c in realization.components):
            raise ValueError('sampling interval must be below the highest-frequency Nyquist limit')
        source=json.loads(self.source_payload_json)
        if source.get('composition') not in ('fixed_diffraction','total_with_source_motion') or not source.get('bank_id'):
            raise ValueError('explicit pressure composition and source bank identity required')
        heading=float(source['heading']); weights=np.zeros(len(self.omegas),complex)
        for c in realization.components:
            if c.omega not in self.omegas or abs(math.remainder(math.radians(c.direction_degrees)-heading,math.tau))>4*math.ulp(math.tau):
                raise ValueError('realization does not match native bank frequencies/heading')
            weights[self.omegas.index(c.omega)]+=c.amplitude*np.exp(1j*c.phase)
        object.__setattr__(self,'_weights',_ro(weights))
        # One snapshot and its complex reduction must fit; block creation checks
        # all output plus phase temporaries before allocating.
        if self.working_memory_bytes < len(ids)*6*32+len(self.omegas)*32:
            raise ValueError('pressure history workspace too small for one snapshot')
        digest=hashlib.sha256(_json(self._payload()).encode()).hexdigest()
        if self.content_sha256 and self.content_sha256!=digest: raise ValueError('pressure history identity mismatch')
        object.__setattr__(self,'content_sha256',digest)

    @property
    def source_id(self): return json.loads(self.source_payload_json)['bank_id']
    @property
    def realization(self): return WaveRealization.from_dict(json.loads(self.realization_json))
    @property
    def element_ids(self): return tuple(json.loads(self.binding_json).get('preview_element_ids',()))

    def _payload(self):
        d={n:getattr(self,n) for n in ('times','node_ids','omegas','realization_json','mesh_content_sha256',
            'source_payload_json','binding_json','working_memory_bytes')}
        d.update(responses_real=self.responses.real.tolist(),responses_imag=self.responses.imag.tolist())
        return d

    def to_dict(self): return dict(type=TYPE,**self._payload(),content_sha256=self.content_sha256)

    def validate(self):
        if self.responses.flags.writeable or hashlib.sha256(_json(self._payload()).encode()).hexdigest()!=self.content_sha256:
            raise ValueError('pressure-only history changed after construction')

    @classmethod
    def from_dict(cls,data):
        d=dict(data)
        if d.pop('type',None)!=TYPE or not d.get('content_sha256'): raise ValueError('unsupported pressure-only history')
        real=np.asarray(d.pop('responses_real')); imag=np.asarray(d.pop('responses_imag'))
        if real.shape!=imag.shape or real.dtype.kind not in 'fiu' or imag.dtype.kind not in 'fiu': raise ValueError('invalid complex parts')
        return cls(**d,responses=real+1j*imag)

    def _phase(self,time):
        if isinstance(time,(bool,np.bool_)) or not np.isscalar(time) or not np.isreal(time): raise ValueError('real scalar time required')
        t=float(time)
        if not math.isfinite(t) or not self.times[0]<=t<=self.times[-1]: raise ValueError('time outside pressure history')
        return self._weights*np.exp(-1j*np.asarray(self.omegas)*t)

    def evaluate(self,time):
        return np.einsum('k,knj->nj',self._phase(time),self.responses).real.copy()

    def preview_pressures(self,time):
        binding=json.loads(self.binding_json)
        z=np.asarray(binding.get('preview_pressure_real',[]))+1j*np.asarray(binding.get('preview_pressure_imag',[]))
        if z.shape!=(len(self.omegas),len(self.element_ids)) or not np.isfinite(z).all():
            raise ValueError('pressure preview data unavailable or invalid')
        return np.einsum('k,ke->e',self._phase(time),z).real

    def iter_blocks(self,times,*,block_size=64,cancel=None):
        if type(block_size) is not int or block_size<1: raise ValueError('positive block size required')
        n=len(self.node_ids)*6
        limit=(self.working_memory_bytes-len(self.omegas)*32)//(n*32)
        if limit<1: raise ValueError('workspace cannot hold a pressure block')
        size=min(block_size,int(limit)); batch=[]
        for t in times:
            if cancel is not None and cancel(): raise RuntimeError('pressure evaluation cancelled')
            self._phase(t); batch.append(t)
            if len(batch)==size:
                yield np.stack([self.evaluate(x) for x in batch]); batch=[]
        if batch: yield np.stack([self.evaluate(x) for x in batch])
