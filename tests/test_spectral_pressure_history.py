import json
import numpy as np
import pytest
from anyhydro.pressure_bank import PressureResponseBank
from anyloads.marine import AiryComponent
from anyloads.sea_state import WaveRealization
from anyloads.pressure_history import pressure_history_from_dict,MappedPressureHistory
from anyloads.spectral_pressure_history import (build_pressure_mapping_plan,map_pressure_bank,
    SpectralMappedPressureHistory)


def bank():
    return PressureResponseBank(vertices=[[0,0,0],[1,0,0],[1,1,0],[0,1,0]],faces=[[0,1,2,3]],
        surface_ids=('bottom',),centers=[[.5,.5,0]],normals=[[0,0,1]],areas=[1],
        omegas=[.7,1.1],responses=[[1+2j],[3-1j]])


def history(times=(0.,.13,1.)):
    b=bank();r=WaveRealization((AiryComponent(.2,.7,phase=.3),AiryComponent(.1,1.1,phase=1.)))
    return SpectralMappedPressureHistory(times,(42,),tuple(b.omegas),b.responses.real,b.responses.imag,
        b.heading,'a'*64,b.bank_id,json.dumps(r.to_dict()))


def test_mapping_search_once_and_frequency_wrenches(monkeypatch):
    import anyloads.spectral_pressure_history as module
    b=bank();calls=[];transfer=module.transfer_panel_pressure
    def tracked(*args,**kwargs):calls.append(1);return transfer(*args,**kwargs)
    monkeypatch.setattr(module,'transfer_panel_pressure',tracked)
    plan=build_pressure_mapping_plan(b,(42,),b.centers,b.normals,b.areas,b.surface_ids,mesh_content_sha256='a'*64)
    mapped,errors=map_pressure_bank(b,plan,b.centers,b.normals,b.areas,
        surface_ids=b.surface_ids,mesh_content_sha256='a'*64)
    np.testing.assert_array_equal(mapped,b.responses)
    assert len(calls)==1 and len(errors)==2
    from dataclasses import replace
    with pytest.raises(TypeError):replace(plan,panel_indices=(999,))
    from anyloads.spectral_pressure_history import PressureMappingPlan
    with pytest.raises(TypeError):PressureMappingPlan()
    assert all(e['force']['absolute']==0 and e['moment']['absolute']==0 for e in errors)
    with pytest.raises(ValueError):map_pressure_bank(b,plan,b.centers,b.normals,b.areas,
        surface_ids=b.surface_ids,mesh_content_sha256='b'*64)
    with pytest.raises(ValueError):map_pressure_bank(b,plan,b.centers+.1,b.normals,b.areas,
        surface_ids=b.surface_ids,mesh_content_sha256='a'*64)


def test_shared_kernel_actual_time_persistence_and_compact_long_record():
    h=history();compiled=h._compiled
    for time in (.07,.31):
        r=WaveRealization.from_dict(json.loads(h.realization_json));b=bank()
        expected=sum(np.real(b.responses[i]*c.amplitude*np.exp(1j*(c.phase-c.omega*time)))
            for i,c in enumerate(r.components))
        np.testing.assert_allclose(h.evaluate(time),expected,atol=1e-12)
    assert h._compiled is compiled
    restored=pressure_history_from_dict(h.to_dict())
    assert restored.content_sha256==h.content_sha256
    np.testing.assert_array_equal(restored.evaluate(.31),h.evaluate(.31))
    long=history(tuple(np.linspace(0,10800,108001)))
    assert len(long.real)==2 and len(long.real[0])==1 and len(long.times)==108001
    assert 'pressures' not in long.to_dict()
    for time in (-1,2,None,True,False,np.bool_(True),.13+99j,np.complex128(.13+99j),
            np.complex128(.13+0j),object(),'invalid'):
        with pytest.raises(ValueError):h.evaluate(time)
    with pytest.raises(ValueError):h.evaluate(.1,(99,))
    bad=h.to_dict();bad['real']=[[10],[30]]
    with pytest.raises(ValueError):pressure_history_from_dict(bad)
    bad=h.to_dict();bad['type']='spectral_mapped_pressure_v99'
    with pytest.raises(ValueError):pressure_history_from_dict(bad)


def test_harmonic_dispatch_unchanged_and_zero_reference():
    h=MappedPressureHistory((0.,1.),(42,),(2.,),(1.,),.7,'a'*64,'existing')
    restored=pressure_history_from_dict(h.to_dict())
    np.testing.assert_array_equal(restored.evaluate(.2),h.evaluate(.2))
    from anyloads.spectral_pressure_history import _error
    e=_error(np.zeros(3),np.ones(3))
    assert e['zero_reference'] and e['relative'] is None and e['absolute']>0
