import json
import math
import numpy as np
import pytest
from anyloads import (JonswapSpectrum, SpectralBins, DirectionalSpectrum,
    WaveRealization, RegularWave, LoadError)


def realization():
    return WaveRealization.from_jonswap(JonswapSpectrum(2.,8.),(.7,1.1,1.6),(.2,.2,.2),
        seed=17,directions=DirectionalSpectrum((0.,math.pi/2),(.25,.75)))


@pytest.mark.parametrize('count',[32,200])
@pytest.mark.parametrize('period',[1.3,6.,17.])
def test_equal_bins_shared_edges_accept_representation_only(count,period):
    wp=math.tau/period;width=wp*2.5/count
    omegas=tuple(wp*.5+(i+.5)*width for i in range(count));widths=(width,)*count
    bins=SpectralBins(omegas,(1.,)*count,widths)
    assert bins.omegas==omegas and bins.bandwidths==widths
    JonswapSpectrum(1.,period).discretize(omegas,widths)
    with pytest.raises(LoadError):SpectralBins((1.,1.1),(1.,1.),(.2,.2))
    with pytest.raises(LoadError):SpectralBins((1e308,),(1.,),(1.7e308,))


def test_density_matches_independent_cyclic_frequency_formula():
    spectrum=JonswapSpectrum(2.,8.)
    w=np.array([.5,.7,.8,1.,2.]); f=w/(2*np.pi); fm=1/8
    sigma=np.where(f<=fm,.07,.09)
    b=np.exp(-.5*((f/fm-1)/sigma)**2)
    sf=.0081*9.81**2/(16*np.pi**4)*f**-5*np.exp(-1.25*(fm/f)**4)*3.3**b
    np.testing.assert_allclose(spectrum.density(w,alpha=.0081),sf/(2*np.pi),rtol=1e-12,atol=0.)
    plain=JonswapSpectrum(2.,8.,gamma=1.)
    np.testing.assert_allclose(plain.density(w,alpha=.0081),
        .0081*9.81**2*w**-5*np.exp(-1.25*((2*np.pi/8)/w)**4),rtol=1e-12,atol=0.)


def test_finite_bin_energy_and_direction_weights():
    r=realization()
    assert r.spectral_bins.variance==pytest.approx(2.**2/16,rel=1e-12)
    assert sum(c.amplitude**2/2 for c in r.components)==pytest.approx(.25,rel=1e-12)
    assert sum(c.amplitude**2/2 for c in r.components[::2])==pytest.approx(.25*.25,rel=1e-12)
    assert r.to_dict()==realization().to_dict()
    other=WaveRealization.from_spectrum(r.spectral_bins,seed=18,directions=r.directions,jonswap=r.jonswap)
    assert other.content_hash!=r.content_hash
    zero=WaveRealization.from_jonswap(JonswapSpectrum(0.,8.),(.7,1.1),(.2,.2),seed=0)
    assert all(c.amplitude==0. for c in zero.components)


def test_roundtrip_materializes_phases_without_rng(monkeypatch):
    r=realization(); data=json.loads(json.dumps(r.to_dict()))
    monkeypatch.setattr(np.random,'PCG64',lambda *a:pytest.fail('deserialization regenerated phases'))
    saved=WaveRealization.from_dict(data)
    assert saved.content_hash==r.content_hash and saved.components==r.components
    p=[[1.,2.,-3.],[2.,1.,-10.]]
    for t in (0.,.3,2.):
        a=r.environment(20.).evaluate(p,t); b=saved.environment(20.).evaluate(p,t)
        for key in ('velocity','acceleration','velocity_gradient','acceleration_gradient'):
            np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    data['components'][0]['phase']+=.1
    assert WaveRealization.from_dict(data).content_hash!=r.content_hash
    assert r.components==saved.components


def test_regular_wave_boundary_and_immutable_records():
    r=RegularWave(.2,1.,-math.pi/2,.3).realize()
    assert r.components[0].direction_degrees==270.
    assert WaveRealization.from_dict(r.to_dict()).content_hash==r.content_hash
    w=[.7,1.1]; bins=SpectralBins(w,[.2,.3],[.2,.2]); w[0]=99
    assert bins.omegas[0]==.7
    with pytest.raises(AttributeError): bins.omegas=(1.,)


def test_surface_elevation_uses_saved_superposition_and_datum():
    r=realization();env=r.environment(20.,still_water_level=.4)
    points=np.array([[1.,2.],[3.,4.]])
    for t in (0.,.3,2.):
        expected=np.full(2,.4)
        for c,k in zip(r.components,env.wave_numbers):
            d=[math.cos(math.radians(c.direction_degrees)),math.sin(math.radians(c.direction_degrees))]
            expected+=c.amplitude*np.cos(k*(points@d)-c.omega*t+c.phase)
        np.testing.assert_allclose(env.surface_elevation(points,t),expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(env.surface_elevation(points,0.),
        WaveRealization.from_dict(r.to_dict()).environment(20.,still_water_level=.4).surface_elevation(points,0.))
    for bad in ([[True,0.]],[[0.,float('nan')]],[[1.,2.,3.]],[[1.+1j,0.]]):
        with pytest.raises(LoadError):env.surface_elevation(bad,0.)


@pytest.mark.parametrize('call',[
    lambda: JonswapSpectrum(True,8.),lambda:JonswapSpectrum(1.,0.),
    lambda:JonswapSpectrum(1.,8.,gamma=.5),lambda:SpectralBins([1.],[True],[.2]),
    lambda:SpectralBins([1.,1.1],[1.,1.],[.2,.2]),
    lambda:SpectralBins([True,2.],[.2,.3],[.2,.2]),
    lambda:DirectionalSpectrum((0.,math.tau),(.5,.5)),
    lambda:DirectionalSpectrum((0.,1e-13),(.5,.5)),
    lambda:DirectionalSpectrum((0.,1.),(.2,.2)),lambda:WaveRealization(None),
    lambda:RegularWave(1.,False),lambda:realization().environment(20.,components=()),
])
def test_named_records_refuse_invalid_data(call):
    with pytest.raises(LoadError): call()


@pytest.mark.parametrize('change',[
    lambda d:d.update(api_version='unknown'),lambda d:d.update(provider='python'),
    lambda d:d['components'][0].pop('phase'),lambda d:d.update(seed=True),
    lambda d:d['components'][0].update(amplitude=99.),
    lambda d:d['spectral_bins']['density_per_radian'].__setitem__(0,99.),
])
def test_serialization_refuses_unknown_or_inconsistent_records(change):
    data=realization().to_dict(); change(data)
    with pytest.raises(LoadError): WaveRealization.from_dict(data)
