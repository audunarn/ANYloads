from types import SimpleNamespace
import numpy as np
import pytest
from anyloads.wave_preview import source_resultants,select_sample_window,pixel_extrema
from anyloads.marine import AiryComponent
from anyloads.sea_state import WaveRealization
from anyloads.pressure_series import CONVENTIONS


def source():
    return SimpleNamespace(omegas=np.array([.7,1.1]),responses=np.array([[1+2j,3-1j],[2-.5j,1+4j]]),
        centers=np.array([[2.,0,-1],[0,3,-2]]),normals=np.array([[1.,0,0],[0,0,-1]]),areas=np.array([2.,3.]),
        heading=0.,phase_origin=(0.,0.,0.),bank_id='a'*64,conventions=dict(CONVENTIONS),validate=lambda:None)


def realization():return WaveRealization((AiryComponent(.5,.7,phase=.3),AiryComponent(.2,1.1,phase=-.4)))


def test_source_panel_resultants_match_direct_pressure_integrals_absolute_times():
    b=source();r=realization();times=(0.,.13,1.,100.,1e6)
    actual=source_resultants(b,r,times)
    expected=[]
    for time in times:
        pressure=sum(b.responses[i]*c.amplitude*np.exp(1j*(c.phase-c.omega*time)) for i,c in enumerate(r.components)).real
        force=-pressure[:,None]*b.areas[:,None]*b.normals
        expected.append(np.r_[force.sum(axis=0),np.cross(b.centers,force).sum(axis=0)])
    np.testing.assert_allclose(actual.values,expected,rtol=1e-10,atol=1e-9)
    assert actual.moment_origin==(0.,0.,0.) and actual.units==('N','N','N','Nm','Nm','Nm')
    assert not actual.values.flags.writeable
    cancelled=WaveRealization((AiryComponent(1,.7),AiryComponent(1,.7,phase=np.pi)))
    assert np.max(np.abs(source_resultants(b,cancelled,(0.,1.)).values))<1e-12


def test_resultant_units_and_memory_refuse_before_projection():
    b=source()
    for budget in (True,0,64):
        with pytest.raises(ValueError):source_resultants(b,realization(),(0.,1.),working_memory_bytes=budget)
    b.conventions={**CONVENTIONS,'units':'kPa/m'}
    with pytest.raises(ValueError,match='conventions'):source_resultants(b,realization(),(0.,1.))
    b=source()
    with pytest.raises(ValueError,match='cancelled'):source_resultants(b,realization(),(0.,1.),cancel=lambda:True)
    with pytest.raises(ValueError,match='workspace'):
        source_resultants(b,realization(),np.linspace(0,10800,108001),working_memory_bytes=1024**2)


def test_window_snap_is_original_absolute_sample_inventory_and_extrema_retained():
    t=tuple(np.arange(101)/10)
    window=select_sample_window(t,1.94,4.31)
    assert window.interval==(1.9,4.3) and window.times==t[19:44]
    for endpoints in ((1.,1.01),(2.,1.),(-1.,2.),(0.,11.),(True,2.)):
        with pytest.raises(ValueError):select_sample_window(t,*endpoints)
    long=np.linspace(0,10800,108001);trace=np.zeros(len(long));trace[51]=123;trace[76]=-99
    x,lo,hi=pixel_extrema(long,trace,500)
    assert len(x)<=500 and hi.max()==123 and lo.min()==-99
