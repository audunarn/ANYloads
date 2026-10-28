import math
import numpy as np
import pytest
from anyhydro.pressure_bank import PressureResponseBank
from anyloads.marine import AiryComponent
from anyloads.sea_state import WaveRealization, SpectralBins, DirectionalSpectrum
from anyloads.errors import LoadError
from anyloads.pressure_series import compile_pressure_series, PressureSeries, PressureSeriesCancelled


def bank(**changes):
    data=dict(vertices=[[0,0,0],[1,0,0],[1,1,0],[0,1,0]],faces=[[0,1,2,3]]*3,
        surface_ids=('a','b','c'), centers=[[.5,.5,0]]*3,normals=[[0,0,1]]*3,
        areas=[1]*3,omegas=[.7,1.1],responses=[[1+2j,2-1j,3],[2,4+2j,-1j]],panel_ids=(7,3,10))
    data.update(changes); return PressureResponseBank(**data)


def realization():
    return WaveRealization((AiryComponent(2,.7,phase=.4),AiryComponent(.3,1.1,phase=-.8),
                            AiryComponent(.2,.7,phase=2.)))


def oracle(b,r,times):
    output=np.zeros((len(times),len(b.panel_ids)))
    for c in r.components:
        h=b.responses[list(b.omegas).index(c.omega)]
        for i,t in enumerate(times): output[i]+=np.real(h*c.amplitude*np.exp(1j*(c.phase-c.omega*t)))
    return output


def test_direct_sum_irregular_long_times_and_selection():
    b=bank(); r=realization(); s=compile_pressure_series(b,r)
    times=np.array([-2.,0.,.012,3.1,100.,1e6]); actual=np.empty((len(times),2))
    for block in s.iter_blocks(times,panel_ids=(10,7),time_block_size=2,panel_block_size=1):
        assert block.bank_id==b.bank_id
        actual[block.time_slice,block.panel_slice]=block.pressures
    expected=oracle(b,r,times)[:,[2,0]]
    scale=sum(c.amplitude*np.abs(b.responses[list(b.omegas).index(c.omega)]) for c in r.components)
    assert np.all(np.abs(actual-expected)<=1e-10*np.maximum(1,scale[[2,0]]))
    np.testing.assert_allclose(s.snapshot(3.1,panel_ids=(10,7)),actual[3],rtol=1e-12,atol=1e-12)
    assert s.snapshot(3.1,panel_ids=(10,7)).bank_id==b.bank_id


def test_reduction_interference_cancellation():
    b=bank()
    one=WaveRealization((AiryComponent(1,.7),))
    np.testing.assert_allclose(compile_pressure_series(b,one).snapshot(0),b.responses[0].real)
    cancel=WaveRealization((AiryComponent(1,.7),AiryComponent(1,.7,phase=math.pi)))
    s=compile_pressure_series(b,cancel)
    assert np.max(np.abs(s.snapshot(.3).pressures)) < 1e-12
    with pytest.raises(PressureSeriesCancelled): next(s.iter_blocks([0,1],cancel=lambda:True))


def test_refusals_and_memory():
    b=bank(); r=realization()
    for budget in (True,0,-1,3.5,1):
        with pytest.raises(LoadError): compile_pressure_series(b,r,working_memory_bytes=budget)
    for c in (AiryComponent(1,np.nextafter(.7,1)),AiryComponent(1,.7,90)):
        with pytest.raises(LoadError): compile_pressure_series(b,WaveRealization((c,)))
    s=compile_pressure_series(b,r,working_memory_bytes=2000)
    for times in ([0,0],[np.nan],[1,0],[],[True],[1+2j],[False,1.]):
        with pytest.raises(LoadError): list(s.iter_blocks(times))
    with pytest.raises(LoadError): list(s.iter_blocks(np.arange(1000)))
    with pytest.raises(LoadError): s.snapshot(0,panel_ids=(999,))
    with pytest.raises(LoadError): compile_pressure_series(bank(phase_origin=(1,0,0)),r)
    zeros=compile_pressure_series(b,WaveRealization((AiryComponent(0,42,90),)))
    assert not np.any(zeros.snapshot(0).pressures)
    assert s.snapshot(0).panel_ids == b.panel_ids
    huge=compile_pressure_series(bank(omegas=[1e300,1.1]),WaveRealization((AiryComponent(1,1e300),)))
    with pytest.raises(LoadError): huge.snapshot(1e300)


def test_persist_actual_realization_and_changed_bank(tmp_path):
    b=bank(); r=realization(); s=compile_pressure_series(b,r,bank_reference='named-bank')
    path=tmp_path/'series.json'; s.save(path)
    restored=PressureSeries.load(path,bank_loader=lambda ref: b if ref=='named-bank' else None)
    assert restored.realization.to_dict()==r.to_dict()
    np.testing.assert_array_equal(restored.snapshot(.3),s.snapshot(.3))
    with pytest.raises(LoadError): PressureSeries.load(path,bank_loader=lambda ref:bank(rho=1000))
    path.write_text(path.read_text().replace('named-bank','other'))
    with pytest.raises(LoadError): PressureSeries.load(path,bank_loader=lambda ref:b)


def test_mutation_refusal():
    b=bank(); s=compile_pressure_series(b,realization())
    object.__setattr__(b,'heading',.2)
    with pytest.raises(Exception): s.snapshot(0)
    with pytest.raises(Exception): compile_pressure_series(b,realization())


def test_seeded_realization_persistence_does_not_regenerate(tmp_path,monkeypatch):
    b=bank(); r=WaveRealization.from_spectrum(SpectralBins((.7,1.1),(.2,.4),(.1,.2)),seed=4321)
    s=compile_pressure_series(b,r,bank_reference='seeded-bank'); path=tmp_path/'seeded.json'; s.save(path)
    def forbidden(*args,**kwargs): raise AssertionError('must not regenerate seed')
    monkeypatch.setattr(np.random,'Generator',forbidden)
    loaded=PressureSeries.load(path,bank_loader=lambda ref:b)
    assert loaded.realization.to_dict()==r.to_dict()
    np.testing.assert_array_equal(loaded.snapshot(1).pressures,s.snapshot(1).pressures)


def test_minimal_block_budget_accounting():
    from anyloads.pressure_series import _with_budget
    s=compile_pressure_series(bank(),realization())
    # Resident coefficients + one time + one index + selected complex-valued
    # real/imag coefficients + sin/cos + two products + finite-output bool.
    k=len(s.omegas); minimal=s.coefficient_bytes+16+16*k+16*k+17
    exact=_with_budget(s,minimal)
    blocks=list(exact.iter_blocks([0.],panel_ids=(7,)))
    assert blocks[0].pressures.shape==(1,1)
    with pytest.raises(LoadError): list(_with_budget(s,minimal-1).iter_blocks([0.],panel_ids=(7,)))


def test_factory_only_and_immutable_compiled_arrays(tmp_path):
    from dataclasses import replace
    s=compile_pressure_series(bank(),realization())
    with pytest.raises(TypeError): PressureSeries()
    with pytest.raises(TypeError): PressureSeries(bank_id='fake')
    with pytest.raises(TypeError): replace(s,coefficients_real=np.zeros_like(s.coefficients_real))
    for a in (s.omegas,s.coefficients_real,s.coefficients_imag):
        with pytest.raises(ValueError): a.setflags(write=True)
        with pytest.raises(ValueError): a.flat[0]=0
    path=tmp_path/'factory.json'; s.save(path,bank_reference='factory-bank')
    loaded=PressureSeries.load(path,bank_loader=lambda ref:bank())
    np.testing.assert_array_equal(loaded.snapshot(2).pressures,s.snapshot(2).pressures)


@pytest.mark.parametrize('heading',[math.radians(30),.1])
def test_degree_boundary_representation_only(heading):
    b=bank(heading=heading)
    bins=SpectralBins((.7,1.1),(.2,.4),(.1,.2))
    r=WaveRealization.from_spectrum(bins,seed=12,
        directions=DirectionalSpectrum((heading,),(1.,)))
    s=compile_pressure_series(b,r)
    assert s.snapshot(0).panel_ids==b.panel_ids
    wrong=WaveRealization((AiryComponent(1,.7,math.degrees(heading)+1e-10),))
    with pytest.raises(LoadError): compile_pressure_series(b,wrong)


def test_persistence_requires_meaningful_bank_reference(tmp_path):
    s=compile_pressure_series(bank(),realization())
    path=tmp_path/'reference.json'
    for reference in (None,'','   '):
        with pytest.raises(LoadError): s.save(path,bank_reference=reference)
        assert not path.exists()
    s.save(path,bank_reference='bank.npz')
    restored=PressureSeries.load(path,bank_loader=lambda reference:bank())
    assert restored.bank_reference=='bank.npz'
