"""Independent-equation tests for the wind history, thrust curve and turbine applicator."""
import numpy as np
import pytest

from anyloads.wind_turbine import (
    ThrustCurve,
    ThrustCurveDomainError,
    TurbineLoadApplicator,
    WindError,
    WindHistory,
    generate_kaimal_realization,
    generate_kaimal_wind_history,
    kaimal_bin_energy,
)

U, TI, L = 10.0, 0.15, 340.0
SIGMA = TI * U


def _s(f):
    # the plan's one-sided spectrum, written independently of the module
    return 4.0 * SIGMA**2 * L / U / (1.0 + 6.0 * f * L / U) ** (5.0 / 3.0)


def _realization(seed=7, period=200.0, fmax=1.0):
    return generate_kaimal_realization(
        mean_speed=U, turbulence_intensity=TI, integral_scale=L, direction=(1, 0, 0),
        seed=seed, period=period, max_frequency=fmax,
    )


def test_bin_energy_matches_numerical_integral_of_spectrum():
    f = np.geomspace(1e-4, 0.5, 400001)
    s = _s(f)
    numeric = np.sum(0.5 * (s[1:] + s[:-1]) * np.diff(f))
    exact = float(kaimal_bin_energy(1e-4, 0.5, sigma=SIGMA, mean=U, length=L))
    assert exact == pytest.approx(numeric, rel=1e-6)


def test_total_variance_is_exactly_partitioned_into_represented_and_missing():
    r = _realization()
    total = r.represented_variance + r.missing_low_variance + r.missing_high_variance
    assert total == pytest.approx(SIGMA**2, rel=1e-12)
    assert r.missing_low_variance > 0.0 and r.missing_high_variance > 0.0
    # lost low-frequency variance equals the spectrum integral over [0, df/2]
    df = 1.0 / r.period
    f = np.linspace(0.0, 0.5 * df, 200001)
    numeric = np.sum(0.5 * (_s(f)[1:] + _s(f)[:-1]) * np.diff(f))
    assert r.missing_low_variance == pytest.approx(numeric, rel=1e-6)


def test_sampled_period_has_exact_finite_band_variance_and_mean():
    dt, period = 0.5, 200.0
    r = _realization(period=period, fmax=0.5 / dt)
    t = np.arange(int(period / dt)) * dt  # one full period, endpoint excluded
    u = U + r.fluctuation(t)
    assert np.mean(u) == pytest.approx(U, abs=1e-10)
    assert np.var(u) == pytest.approx(r.represented_variance, rel=1e-10)
    assert r.represented_variance < SIGMA**2  # band-limited: not renormalised


def test_bin_energy_per_component_gives_cosine_amplitude():
    r = _realization()
    assert np.allclose(0.5 * r.amplitudes**2, r.bin_energy, rtol=1e-13)


def test_seed_determinism_and_difference():
    a = generate_kaimal_wind_history(mean_speed=U, turbulence_intensity=TI, integral_scale=L,
                                     direction=(1, 0, 0), seed=3, dt=0.5, duration=50.0)
    b = generate_kaimal_wind_history(mean_speed=U, turbulence_intensity=TI, integral_scale=L,
                                     direction=(1, 0, 0), seed=3, dt=0.5, duration=50.0)
    c = generate_kaimal_wind_history(mean_speed=U, turbulence_intensity=TI, integral_scale=L,
                                     direction=(1, 0, 0), seed=4, dt=0.5, duration=50.0)
    assert np.array_equal(a.velocity, b.velocity)
    assert a.content_hash() == b.content_hash()
    assert not np.allclose(a.velocity, c.velocity)
    assert a.provenance["seed"] == 3 and a.provenance["generator"].startswith("kaimal")


def test_refinement_resamples_the_same_realisation():
    r = _realization(period=64.0, fmax=0.5)  # Nyquist of dt = 1
    coarse = r.to_wind_history(np.arange(0, 65) * 1.0)
    fine = r.to_wind_history(np.arange(0, 129) * 0.5)
    quarter = r.to_wind_history(np.arange(0, 257) * 0.25)
    assert np.allclose(coarse.velocity, fine.velocity[::2], atol=1e-12)
    assert np.allclose(fine.velocity, quarter.velocity[::2], atol=1e-12)


def test_aliasing_and_looping_are_refused():
    r = _realization(period=64.0, fmax=1.0)
    with pytest.raises(WindError, match="aliasing"):
        r.to_wind_history(np.arange(0, 65) * 1.5)
    r2 = _realization(period=64.0, fmax=0.5)
    with pytest.raises(WindError, match="never looped"):
        r2.to_wind_history(np.arange(0, 100) * 1.0)


def test_zero_turbulence_is_steady_and_generated_nonzero_needs_positive_mean():
    h = generate_kaimal_wind_history(mean_speed=8.0, turbulence_intensity=0.0, integral_scale=100.0,
                                     direction=(0, 1, 0), seed=0, dt=1.0, duration=10.0)
    assert np.all(h.velocity[:, 0] == 0.0) and np.all(h.velocity[:, 1] == 8.0)
    with pytest.raises(WindError, match="positive mean"):
        generate_kaimal_realization(mean_speed=0.0, turbulence_intensity=0.1, integral_scale=100.0,
                                    direction=(1, 0, 0), seed=0, period=10.0, max_frequency=0.5)
    with pytest.raises(WindError, match="integral_scale"):
        generate_kaimal_realization(mean_speed=5.0, turbulence_intensity=0.1, integral_scale=0.0,
                                    direction=(1, 0, 0), seed=0, period=10.0, max_frequency=0.5)


def test_vector_import_interpolation_domain_and_immutability():
    t = np.array([0.0, 1.0, 3.0])
    vel = np.array([[1.0, 0, 0], [3.0, 2.0, 0], [3.0, 2.0, -4.0]])
    w = WindHistory.from_vectors(t, vel)
    assert np.allclose(w.at(2.0), [3.0, 2.0, -2.0])  # midpoint of the second segment
    assert np.allclose(w.at(0.5), [2.0, 1.0, 0.0])
    with pytest.raises(WindError, match="extrapolation"):
        w.at(3.0001)
    with pytest.raises(ValueError):
        w.velocity[0, 0] = 5.0
    with pytest.raises(ValueError):
        w.velocity.setflags(write=True)
    vel[0, 0] = 99.0  # caller mutation must not leak in
    assert w.velocity[0, 0] == 1.0
    with pytest.raises(WindError):
        WindHistory.from_vectors([0.0, 1.0, 1.0], vel)
    with pytest.raises(WindError):
        WindHistory.from_vectors([0.0, 1.0, 2.0], np.where(vel > 2, np.nan, vel))
    calm = WindHistory.from_vectors(t, np.zeros((3, 3)))
    assert np.all(calm.at(1.5) == 0.0)


def test_one_small_interval_cannot_hide_a_long_gap():
    r = _realization(period=64.0, fmax=0.5)
    with pytest.raises(WindError, match="aliasing"):
        r.to_wind_history(np.array([0.0, 0.01, 30.0, 64.0]))   # one 10 ms interval, gaps of 29.99 s and 34 s
    assert r.to_wind_history(np.arange(0, 65) * 1.0).n_samples == 65


def test_direct_realization_construction_validates():
    from dataclasses import replace
    r = _realization()
    for change in (dict(integral_scale=-1.0), dict(mean_speed=-1.0), dict(turbulence_intensity=-0.1),
                   dict(period=0.0), dict(seed=-3), dict(direction=np.array([0.0, 0.0, 0.0])),
                   dict(phases=r.phases[:-1]), dict(amplitudes=r.amplitudes * 1.01),
                   dict(missing_high_variance=r.missing_high_variance + 1.0), dict(max_frequency=np.nan)):
        with pytest.raises(WindError):
            replace(r, **change)
    assert replace(r, seed=r.seed).seed == r.seed   # an unchanged copy stays admissible


def test_nonfinite_wind_query_is_refused():
    w = WindHistory.from_vectors([0.0, 1.0], [[1.0, 0, 0], [1.0, 0, 0]])
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(WindError):
            w.at(bad)


def test_complex_input_is_refused_before_any_float_cast():
    import warnings
    z = np.array([[10.0 + 0j, 0.0, 0.0], [10.0 + 0j, 0.0, 0.0]])          # zero imaginary part is still complex
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                       # a ComplexWarning would also fail
        with pytest.raises(WindError, match="complex"):
            WindHistory.from_vectors([0.0, 1.0], z)
        with pytest.raises(WindError, match="complex"):
            WindHistory.from_vectors([0.0, 1.0], np.zeros((2, 3)), reference_mean=np.array([1 + 2j, 0, 0]))
        with pytest.raises(WindError, match="complex"):
            WindHistory.from_longitudinal([0.0, 1.0], np.array([1 + 1j, 2 + 0j]), (1, 0, 0))
        with pytest.raises(WindError, match="complex"):
            WindHistory.from_vectors(np.array([0.0 + 1j, 1.0]), np.zeros((2, 3)))
        with pytest.raises(WindError, match="complex"):
            ThrustCurve(np.array([0.0, 20.0]), np.array([1 + 2j, 3 + 4j]))
        with pytest.raises(WindError, match="complex"):
            ThrustCurve(np.array([0.0, 20.0 + 0j]), np.array([1.0, 3.0]))
        with pytest.raises(WindError, match="complex"):
            CURVE.thrust(5.0 + 1j)
        with pytest.raises(WindError, match="complex"):
            WindHistory.from_vectors([0.0, 1.0], np.zeros((2, 3))).at(0.5 + 0j)
        app = TurbineLoadApplicator([0, 0, 90.0], [1.0, 0, 0])
        with pytest.raises(WindError, match="complex"):
            app.relative_axial_speed([10.0, 0, 0], np.eye(3) * (1 + 0j))
        with pytest.raises(WindError, match="complex"):
            app.relative_axial_speed([10.0 + 1j, 0, 0], np.eye(3))
        with pytest.raises(WindError, match="complex"):
            app.wrench_about([0, 0, 1j], 1.0)
        with pytest.raises(WindError, match="complex"):
            TurbineLoadApplicator([0, 0, 90.0 + 0j], [1.0, 0, 0])
    assert WindHistory.from_vectors([0.0, 1.0], np.ones((2, 3))).n_samples == 2     # real input is untouched


def test_band_coverage_check():
    w = WindHistory.from_longitudinal(np.arange(0, 11) * 1.0, np.ones(11), (1, 0, 0))
    w.check_band_coverage(np.pi * 0.99)  # Nyquist 0.5 Hz = pi rad/s
    with pytest.raises(WindError, match="Nyquist"):
        w.check_band_coverage(4.0)


# ------------------------------------------------------------------ thrust curve

CURVE = ThrustCurve([0.0, 4.0, 8.0, 12.0, 20.0], [0.0, 100.0, 400.0, 300.0, 200.0])


def test_curve_knots_interior_and_negative_slope():
    for v, T in zip(CURVE.speeds, CURVE.thrusts):
        assert CURVE.thrust(v) == T
    assert CURVE.thrust(6.0) == pytest.approx(250.0)       # 100 + 0.5*(400-100)
    assert CURVE.thrust(10.0) == pytest.approx(350.0)      # on the descending segment
    assert CURVE.slope(10.0) == pytest.approx(-25.0)
    assert CURVE.slope(8.0) == pytest.approx(-25.0)        # right-hand slope at a knot
    assert CURVE.slope(20.0) == pytest.approx(-12.5)       # left-hand slope at the last knot
    v0, v1, T0, T1 = CURVE.segments()
    assert len(v0) == 4 and np.all(v1 > v0)


def test_curve_domain_failures_report_time_value_and_domain():
    with pytest.raises(ThrustCurveDomainError) as exc:
        CURVE.thrust(20.0001, time=12.5)
    assert exc.value.time == 12.5 and exc.value.value == 20.0001 and exc.value.domain == (0.0, 20.0)
    with pytest.raises(ThrustCurveDomainError):
        CURVE.thrust(-1e-9)
    with pytest.raises(ThrustCurveDomainError):
        CURVE.thrust(float("nan"))


def test_curve_construction_refusals():
    with pytest.raises(WindError, match="horizontal"):
        ThrustCurve([0.0, 1.0], [0.0, 1.0], abscissa="horizontal_hub_speed")
    with pytest.raises(WindError):
        ThrustCurve([0.0, 1.0, 1.0], [0.0, 1.0, 2.0])
    with pytest.raises(WindError):
        ThrustCurve([-1.0, 1.0], [0.0, 1.0])
    with pytest.raises(WindError):
        ThrustCurve([0.0, 1.0], [0.0, -1.0])
    with pytest.raises(WindError):
        ThrustCurve([0.0], [1.0])


# ------------------------------------------------------------------ applicator

def test_generalized_vector_and_conservative_wrench():
    app = TurbineLoadApplicator.from_yaw_tilt([-5.0, 2.0, 90.0], yaw=0.0, tilt=np.deg2rad(6.0))
    n = np.array([np.cos(np.deg2rad(6.0)), 0.0, np.sin(np.deg2rad(6.0))])
    assert np.allclose(app.rotor_axis_body, n, atol=1e-15)
    r = np.array([-5.0, 2.0, 90.0])
    expected = np.concatenate([n, np.cross(r, n)])
    assert np.allclose(app.generalized_vector, expected, atol=1e-12)
    T = 1.2e6
    w = app.generalized_force(T)
    assert np.allclose(w, T * expected, rtol=1e-14)
    # moving the reference point changes the moment by the transported arm only
    p = np.array([1.0, -3.0, 4.0])
    w2 = app.wrench_about(p, T)
    assert np.allclose(w2[:3], w[:3])
    assert np.allclose(w2[3:], w[3:] - np.cross(p, w[:3]), rtol=1e-12)
    # virtual work: wrench . qdot == F . v_hub for a rigid twist
    v, om = np.array([0.3, -0.1, 0.2]), np.array([0.01, 0.02, -0.03])
    vh = v + np.cross(om, r)
    assert app.power(T, v, om) == pytest.approx(T * n @ vh, rel=1e-12)


def test_surge_sign_and_pitch_hub_velocity():
    app = TurbineLoadApplicator([0.0, 0.0, 100.0], [1.0, 0.0, 0.0])
    Q0 = np.eye(3)
    wind = [12.0, 0.0, 0.0]
    assert app.relative_axial_speed(wind, Q0) == pytest.approx(12.0)
    # platform surging downwind at 2 m/s reduces the relative speed
    assert app.relative_axial_speed(wind, Q0, translation_velocity=[2.0, 0, 0]) == pytest.approx(10.0)
    # pitch rate w_y: omega x r = (w_y*100, 0, 0) for r = (0,0,100) -> Vrel = U - 100 w_y
    assert app.relative_axial_speed(wind, Q0, angular_velocity=[0, 0.02, 0]) == pytest.approx(12.0 - 2.0)
    # roll rate gives hub velocity along y only: no axial effect
    assert app.relative_axial_speed(wind, Q0, angular_velocity=[0.05, 0, 0]) == pytest.approx(12.0)


def test_wind_pullback_with_tilted_axis_and_rotated_mean_frame():
    tilt = np.deg2rad(5.0)
    app = TurbineLoadApplicator.from_yaw_tilt([0, 0, 80.0], tilt=tilt)
    th = 0.02
    wind = np.array([10.0, 0.0, 0.0])
    # theta about +y: U_b = U - theta x U ; y x x = -z -> U_b = (U, 0, +th*U)
    got = app.relative_axial_speed(wind, np.eye(3), rotation=[0, th, 0])
    assert got == pytest.approx(10.0 * np.cos(tilt) + th * 10.0 * np.sin(tilt), rel=1e-14)
    # mean frame yawed 90 deg about z: body x axis points along world +y
    Q0 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert app.relative_axial_speed([0.0, 7.0, 0.0], Q0) == pytest.approx(7.0 * np.cos(tilt), rel=1e-14)
    assert app.relative_axial_speed([7.0, 0.0, 0.0], Q0) == pytest.approx(0.0, abs=1e-14)


def test_applicator_refusals_and_immutability():
    with pytest.raises(WindError):
        TurbineLoadApplicator([0, 0, 0], [0, 0, 0])
    with pytest.raises(WindError):
        TurbineLoadApplicator([0, 0, np.inf], [1, 0, 0])
    with pytest.raises(WindError, match="unit"):
        TurbineLoadApplicator([0, 0, 10.0], [2.0, 0, 0])               # never normalised silently
    with pytest.raises(WindError, match="unit"):
        TurbineLoadApplicator([0, 0, 10.0], [1.0, 1e-4, 0])
    app = TurbineLoadApplicator([0, 0, 10.0], [1.0, 0, 0])
    with pytest.raises(ValueError):
        app.hub_offset_body[2] = 0.0
