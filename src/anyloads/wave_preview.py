"""Source-panel dynamic resultants and original-sample time windows, SI.

Six frequency responses are integrated before synthesis. This module never
allocates a panel-by-time pressure history or imports hydrodynamics/FEM/UI.
"""
from dataclasses import dataclass
import hashlib,json,math
import numpy as np
from .pressure_series import _compile_response_series,_frozen,_budget,CONVENTIONS
from .errors import LoadError

RESULTANT_CONVENTIONS=dict(units=('N/m','N/m','N/m','Nm/m','Nm/m','Nm/m'),
    phasor='exp(-i omega t)',normals='outward',pressure='positive inward',
    physics='fixed incident+diffraction',moment_origin=(0.,0.,0.))
LABELS=('Fx','Fy','Fz','Mx','My','Mz')


def _times(values):
    raw=tuple(values)
    if (len(raw)<2 or any(isinstance(x,(bool,np.bool_,complex,np.complexfloating)) or
            not isinstance(x,(int,float,np.integer,np.floating)) or not math.isfinite(x) for x in raw) or
            any(b<=a for a,b in zip(raw,raw[1:]))):
        raise LoadError('at least two finite strictly increasing original times required')
    return tuple(float(x) for x in raw)


@dataclass(frozen=True)
class SampleWindow:
    record_times: tuple
    first: int
    stop: int
    @property
    def times(self):return self.record_times[self.first:self.stop]
    @property
    def interval(self):return self.record_times[self.first],self.record_times[self.stop-1]


def select_sample_window(times,start=None,end=None):
    """Snap explicitly to nearest original samples (ties earlier), >=2 samples."""
    times=_times(times)
    start=times[0] if start is None else start;end=times[-1] if end is None else end
    if any(isinstance(x,(bool,np.bool_,complex,np.complexfloating)) or
            not isinstance(x,(int,float,np.integer,np.floating)) or not math.isfinite(x) for x in (start,end)):
        raise LoadError('finite real window endpoints required')
    if not times[0]<=start<end<=times[-1]:raise LoadError('choose a nonempty interval within the full record')
    def nearest(value):
        right=int(np.searchsorted(times,value));left=max(0,right-1);right=min(right,len(times)-1)
        return left if value-times[left]<=times[right]-value else right
    first,last=nearest(start),nearest(end)
    if last-first<1:raise LoadError('selected interval must contain at least two original samples')
    return SampleWindow(times,first,last+1)


@dataclass(frozen=True,init=False)
class _ResultantBank:
    source_bank_id: str
    bank_id: str
    omegas: np.ndarray
    responses: np.ndarray
    heading: float
    panel_ids: tuple=(0,1,2,3,4,5)
    phase_origin: tuple=(0.,0.,0.)
    @property
    def conventions(self):return dict(RESULTANT_CONVENTIONS)
    def validate(self):
        if self.omegas.flags.writeable or self.responses.flags.writeable:
            raise LoadError('resultant projection is immutable')


@dataclass(frozen=True)
class SourceResultants:
    times: np.ndarray
    values: np.ndarray  # time, Fx Fy Fz Mx My Mz, N/Nm
    source_bank_id: str
    moment_origin: tuple=(0.,0.,0.)
    labels: tuple=LABELS
    units: tuple=('N','N','N','Nm','Nm','Nm')


def source_resultants(bank,realization,times,*,cancel=None,working_memory_bytes=256*1024**2):
    budget=_budget(working_memory_bytes)
    bank.validate()
    if bank.conventions!=CONVENTIONS:raise LoadError('source preview requires exact SI pressure/phasor/physics conventions')
    times=_times(times)
    # Conservative NumPy workspace admission before projection. Original caller
    # bank/Python container storage and internal BLAS workspace are excluded.
    panels=len(bank.areas);frequencies=len(bank.omegas)
    if max(256*panels+512*frequencies,128*len(times)+1024*frequencies+65536)>budget:
        raise LoadError('resultant projection/output workspace exceeds working memory budget')
    if tuple(bank.phase_origin)!=(0.,0.,0.):raise LoadError('preview requires WORLD0 phase origin')
    if cancel is not None and cancel():raise LoadError('resultant preview cancelled')
    weights=-np.asarray(bank.normals)*np.asarray(bank.areas)[:,None]
    weights=np.concatenate((weights,np.cross(np.asarray(bank.centers),weights)),axis=1)
    fields=np.asarray(bank.responses)@weights
    h=hashlib.sha256(json.dumps(dict(source_bank_id=bank.bank_id,conventions=RESULTANT_CONVENTIONS),
        sort_keys=True,separators=(',',':')).encode())
    h.update(memoryview(np.ascontiguousarray(fields)).cast('B'))
    projection=object.__new__(_ResultantBank)
    for name,value in dict(source_bank_id=bank.bank_id,bank_id=h.hexdigest(),omegas=_frozen(bank.omegas),
            responses=np.frombuffer(fields.tobytes(),dtype=fields.dtype).reshape(fields.shape),heading=bank.heading,
            panel_ids=(0,1,2,3,4,5),phase_origin=(0.,0.,0.)).items():object.__setattr__(projection,name,value)
    del weights,fields
    projection_bytes=projection.omegas.nbytes+projection.responses.nbytes
    series=_compile_response_series(projection,realization,working_memory_bytes=budget-projection_bytes,
        expected_conventions=RESULTANT_CONVENTIONS)
    # Explicit output is only Nt*6, never Nt*Npanels.
    needed=len(times)*6*8
    reserve=2*needed+16*len(times)+projection_bytes
    if reserve+series.coefficient_bytes>=budget:
        raise LoadError('six-resultant output exceeds working memory budget')
    from .pressure_series import _with_budget
    bounded=_with_budget(series,budget-reserve)
    values=np.empty((len(times),6))
    for block in bounded.iter_blocks(times,cancel=cancel):values[block.time_slice,block.panel_slice]=block.pressures
    del block
    bank.validate()
    return SourceResultants(_frozen(times),_frozen(values),bank.bank_id)


def excitation_resultants(bank,realization,times,*,cancel=None,working_memory_bytes=256*1024**2):
    """Synthesize compact six-response data; never access original panel data."""
    budget=_budget(working_memory_bytes);times=_times(times);bank.validate()
    if bank.conventions!=RESULTANT_CONVENTIONS:raise LoadError('exact excitation resultant conventions required')
    if np.shape(bank.responses)!=(len(bank.omegas),6):raise LoadError('six excitation responses per frequency required')
    reserve=112*len(times)+bank.responses.nbytes+bank.omegas.nbytes
    if reserve+1024*len(bank.omegas)+65536>budget:raise LoadError('six-resultant output exceeds preview budget')
    series=_compile_response_series(bank,realization,working_memory_bytes=budget-reserve,expected_conventions=RESULTANT_CONVENTIONS)
    if cancel is not None and cancel():raise LoadError('force preview cancelled')
    values=np.empty((len(times),6))
    for block in series.iter_blocks(times,cancel=cancel):values[block.time_slice,block.panel_slice]=block.pressures
    del block
    return SourceResultants(_frozen(times),_frozen(values),bank.bank_id)


def pixel_extrema(times,values,pixels):
    """Per-pixel min/max envelope, retaining extreme events in bounded paint data."""
    t=np.asarray(times);v=np.asarray(values)
    if t.ndim!=1 or v.shape!=(len(t),) or type(pixels) is not int or pixels<=0:
        raise LoadError('matching one-dimensional trace and positive pixel count required')
    if len(t)<=2*pixels:return t,v,v
    edges=np.searchsorted(t,np.linspace(t[0],t[-1],pixels+1))
    edges[-1]=len(t)
    x=[];lo=[];hi=[]
    for first,stop in zip(edges[:-1],edges[1:]):
        if stop>first:
            x.append((t[first]+t[stop-1])/2);lo.append(v[first:stop].min());hi.append(v[first:stop].max())
    return np.asarray(x),np.asarray(lo),np.asarray(hi)
