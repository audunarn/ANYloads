import json
import math

import numpy as np
import pytest

pytest.importorskip('anygeometry.wave_surface_clipping')

from anyloads.marine import AiryComponent
from anyloads.pressure_history import pressure_history_from_dict
from anyloads.sea_state import WaveRealization
from anyloads.wave_surface_pressure import STATE_DRY, STATE_PARTIAL, STATE_WET, WaveSurfacePressureHistory

RHO, G = 1025., 9.81
SHA = 'a' * 64


def history(components, corner_z, kdecay=1., dynamic_scale=1., times=(0., .5, 1., 1.5, 2.), corners=((0, 1, 2),)):
    """Unit fields: eta=1, pd0=rho g * dynamic_scale, pd(z)=pd0*exp(kdecay*z) at every corner point."""
    realization = WaveRealization(tuple(components))
    omegas = tuple(dict.fromkeys(c.omega for c in realization.components))
    z = np.array(corner_z, dtype=float)
    P = len(z)
    eta = np.ones((len(omegas), P))
    pd0 = np.full((len(omegas), P), RHO * G * dynamic_scale)
    pd = pd0 * np.exp(kdecay * np.minimum(z, 0.))
    zeros = np.zeros_like(eta)
    return WaveSurfacePressureHistory(
        times=times, record_times=times, omegas=omegas, element_ids=tuple(range(1, len(corners) + 1)),
        element_corners=corners, corner_z=tuple(z), eta_real=eta.tolist(), eta_imag=zeros.tolist(),
        pd0_real=pd0.tolist(), pd0_imag=zeros.tolist(), pd_real=pd.tolist(), pd_imag=zeros.tolist(),
        heading=0., rho=RHO, gravity=G, realization_json=json.dumps(realization.to_dict()),
        mesh_content_sha256=SHA, model_content_sha256=SHA, target_sha256=SHA, source_id='b' * 64)


def test_still_water_is_exact_hydrostatics_and_above_surface_dry():
    h = history([AiryComponent(0., 1.)] + [AiryComponent(1e-30, 1.)], [-1., -.5, 0.5])
    frame = h.frame(0.)
    # The pressure continuation at submerged corners is rho g depth (dynamic part is ~0 for ~zero amplitude).
    assert frame.pressures[0][0] == pytest.approx(RHO * G * 1., rel=1e-9)
    assert frame.pressures[0][1] == pytest.approx(RHO * G * .5, rel=1e-9)
    assert frame.levels[0][2] < 0.
    state, fraction, mean = frame.wet_summary()
    assert state[0] == STATE_PARTIAL and 0. < fraction[0] < 1.
    assert h.evaluate(0.)[0] == mean[0]


def test_crest_extends_waterline_pressure_and_trough_clips_with_signed_pressure():
    A = .5
    h = history([AiryComponent(A, 1.)], [-1., 0., .3], dynamic_scale=3., times=(0., math.pi))
    crest = h.frame(0.)       # eta = +A at every point
    rg = RHO * G
    assert crest.levels[0].tolist() == pytest.approx([1.5, .5, .2])
    # 0 < z < eta uses the constant waterline dynamic pressure: -rg z + pd0, pd0 = 3 rg A
    assert crest.pressures[0][2] == pytest.approx(-rg * .3 + 3 * rg * A, rel=1e-12)
    assert crest.pressures[0][1] == pytest.approx(3 * rg * A, rel=1e-12)
    assert crest.pressures[0][0] == pytest.approx(rg * 1. + 3 * rg * A * math.exp(-1.), rel=1e-12)
    trough = h.frame(math.pi)  # eta = -A
    assert trough.levels[0].tolist() == pytest.approx([.5, -.5, -.8], abs=1e-12)
    # signed suction is retained underwater (no positivity clamp): rg*(0.9) - 3 rg A exp(-0.9) < rg*0.9
    wet_depth = RHO * G * 1.0 + (-3 * rg * A * math.exp(-1.))
    assert trough.pressures[0][0] == pytest.approx(wet_depth, rel=1e-12)
    deep = history([AiryComponent(A, 1.)], [-1., -.8, -.9], dynamic_scale=40., times=(0., math.pi)).frame(math.pi)
    assert (deep.levels[0] > 0).all() and deep.pressures[0].max() < 0.    # wet AND net suction retained


def test_components_are_summed_before_wet_dry_operation():
    # eta = .3 + .3 at t = 0 exceeds the corner height .5 although each component alone does not.
    h = history([AiryComponent(.3, 1.), AiryComponent(.3, 2.)], [.5, .5, .5], times=(0., 1.))
    frame = h.frame(0.)
    assert frame.levels[0].tolist() == pytest.approx([.1, .1, .1])
    state, fraction, mean = frame.wet_summary()
    assert state[0] == STATE_WET and fraction[0] == pytest.approx(1.)
    single = history([AiryComponent(.3, 1.)], [.5, .5, .5], times=(0., 1.)).frame(0.)
    assert single.wet_summary()[0][0] == STATE_DRY and single.wet_summary()[2][0] == 0.


def test_saved_phase_is_authoritative_and_window_preserves_absolute_phase():
    components = [AiryComponent(.4, 1.3, 0., .7)]
    h = history(components, [-1., -.5, 0.], times=(0., .5, 1., 1.5, 2.))
    window = h.with_window((1., 1.5), (1., 1.5))
    assert np.array_equal(window.frame(1.5).eta, h.frame(1.5).eta)
    assert h.frame(1.5).eta[0] == pytest.approx(.4 * math.cos(-1.3 * 1.5 + .7), rel=1e-12)
    with pytest.raises(ValueError):
        window.frame(.5)                   # outside the selected window; never extrapolated or restamped
    assert window.content_sha256 != h.content_sha256 and window.source_id == h.source_id


def test_roundtrip_dispatch_and_tamper_refusal():
    h = history([AiryComponent(.4, 1.3, 0., .7)], [-1., -.5, 0.])
    data = json.loads(json.dumps(h.to_dict()))
    again = pressure_history_from_dict(data)
    assert type(again) is WaveSurfacePressureHistory and again.content_sha256 == h.content_sha256
    data['pd_real'][0][0] += 1.
    with pytest.raises(ValueError):
        pressure_history_from_dict(data)
    broken = dict(json.loads(json.dumps(h.to_dict())), omegas=[2.])
    with pytest.raises(ValueError):
        pressure_history_from_dict(broken)


def test_no_dense_time_by_mesh_tensor_is_stored():
    h = history([AiryComponent(.4, 1.3)], [-1., -.5, 0.], times=tuple(np.linspace(0, 5, 1001)))
    assert len(h.eta_real) == 1 and len(h.eta_real[0]) == 3


def test_frame_owns_immutable_array_buffers():
    from anyloads.wave_surface_pressure import WaveSurfaceFrame
    eta = np.array([1., 2., 3.]); level = eta.copy(); pressure = eta.copy()
    frame = WaveSurfaceFrame(0., [1], [[0, 1, 2]], eta, [level], [pressure])
    eta[:] = 0.; level[:] = 0.; pressure[:] = 0.
    for array in (frame.eta, *frame.levels, *frame.pressures):
        assert np.array_equal(array, [1., 2., 3.])
        with pytest.raises(ValueError): array[0] = 9.
        with pytest.raises(ValueError): array.setflags(write=True)
    assert frame.element_ids == (1,) and frame.corners == ((0, 1, 2),)
