"""External load history and affine fairlead mooring recovery against independent equations."""
import numpy as np
import pytest

from anyloads.external_history import (
    AffineFairleadMooring,
    ExternalHistoryError,
    ExternalLoadHistory,
    FairleadSpec,
    rigid_attachment_map,
    world_force_pullback,
)


def _cross_matrix(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0.0]])


def _history(**kw):
    t = np.array([0.0, 2.0, 4.0, 10.0])
    vals = np.array([[0.0, 1.0], [2.0, 1.0], [4.0, -1.0], [10.0, -1.0]])
    args = dict(times=t, values=vals, components=("FX", "MY"), frames=("body", "body"),
                points_body=[[0.0, 0.0, 5.0], [0, 0, 0]], attachments=("deck", "deck"),
                identity={"body": "hull-1", "mesh": "abc", "mass": "m-7"})
    args.update(kw)
    return ExternalLoadHistory(**args)


def test_piecewise_linear_values_at_exact_timestamps_and_between():
    h = _history()
    assert np.array_equal(h.at(h.times), h.values)          # exact at supplied stamps
    assert np.allclose(h.at(3.0), [[3.0, 0.0]])             # midpoint of the third segment
    assert np.allclose(h.at(7.0), [[7.0, -1.0]])
    assert h.units == ("N", "N*m")
    assert h.identity["mesh"] == "abc"


def test_no_extrapolation_and_coverage():
    h = _history()
    assert h.covers([0.0, 10.0]) and not h.covers([0.0, 10.5])
    with pytest.raises(ExternalHistoryError, match="extrapolation"):
        h.at([5.0, 10.001])
    with pytest.raises(ExternalHistoryError):
        h.at(-0.5)


def test_construction_refusals_and_mutation_isolation():
    with pytest.raises(ExternalHistoryError):
        _history(times=[0.0, 1.0, 1.0, 2.0])
    with pytest.raises(ExternalHistoryError):
        _history(times=[0.0, 1.0, np.nan, 2.0])
    with pytest.raises(ExternalHistoryError):
        _history(components=("FX", "QQ"))
    with pytest.raises(ExternalHistoryError):
        _history(frames=("body", "ship"))
    with pytest.raises(ExternalHistoryError):
        _history(attachments=("deck", ""))
    vals = np.zeros((4, 2))
    h = _history(values=vals)
    vals[0, 0] = 5.0
    assert h.values[0, 0] == 0.0
    with pytest.raises(ValueError):
        h.values[0, 0] = 1.0


def test_resampling_refuses_aliasing_decimation_unless_declared():
    h = _history()
    fine = h.resample(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert np.allclose(fine.values[:, 0], [0, 1, 2, 3, 4])
    with pytest.raises(ExternalHistoryError, match="alias"):
        h.resample(np.array([0.0, 5.0, 10.0]))
    assert h.resample(np.array([0.0, 5.0, 10.0]), allow_decimation=True).values.shape == (3, 2)


def test_generalized_force_uses_virtual_work_map():
    h = _history()
    g = h.generalized_force([2.0], np.eye(3))[0]
    # FX = 2 N at r = (0,0,5): moment = r x F = (0,0,5)x(2,0,0) = (0, 10, 0); MY = 1 N m adds to My
    assert np.allclose(g, [2.0, 0, 0, 0, 10.0 + 1.0, 0])
    # equals sum_i A_i^T F_i for the force channel
    A = rigid_attachment_map([0, 0, 5.0])
    assert np.allclose(A.T @ np.array([2.0, 0, 0]), [2.0, 0, 0, 0, 10.0, 0])


def test_world_direction_channels_rotate_with_mean_pose():
    Q0 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # body x -> world +y
    h = _history(frames=("world", "world"))
    g = h.generalized_force([2.0], Q0)[0]
    # world +X unit in body axes is Q0^T e_x = (0,-1,0)
    f_body = 2.0 * (Q0.T @ np.array([1.0, 0, 0]))
    m_body = np.cross([0, 0, 5.0], f_body) + 1.0 * (Q0.T @ np.array([0, 1.0, 0]))
    assert np.allclose(g, np.concatenate([f_body, m_body]))


def test_world_force_pullback_matches_rotated_force_to_first_order():
    rng = np.random.default_rng(1)
    Q0 = np.eye(3)
    F = np.array([3.0, -2.0, 1.0])
    r = np.array([1.0, 2.0, -3.0])
    w0, G = world_force_pullback(F, Q0, r)
    theta = 1e-6 * rng.standard_normal(3)
    # exact finite rotation: body-frame components of a world-fixed force, point fixed in the body
    angle = np.linalg.norm(theta)
    k = theta / angle
    R = np.cos(angle) * np.eye(3) + np.sin(angle) * _cross_matrix(k) + (1 - np.cos(angle)) * np.outer(k, k)
    f_exact = R.T @ F
    w_exact = np.concatenate([f_exact, np.cross(r, f_exact)])
    q = np.concatenate([np.zeros(3), theta])
    assert np.max(np.abs((w0 + G @ q) - w_exact)) < 1e-9


def _mooring():
    k1 = np.diag([2.0e4, 3.0e4, 1.0e4])
    k1[0, 1] = 500.0  # deliberately nonsymmetric 3x3 fairlead stiffness
    k2 = np.diag([1.0e4, 2.0e4, 5.0e3])
    c = np.diag([10.0, 20.0, 5.0])
    fl = (
        FairleadSpec("line-A", [40.0, 0.0, -14.0], [-1.0e6, 0.0, 5.0e5], k1, c),
        FairleadSpec("line-B", [-20.0, 35.0, -14.0], [4.0e5, -7.0e5, 3.0e5], k2),
    )
    return AffineFairleadMooring(fl, np.eye(3), validity_half_range=[5, 5, 5, 0.09, 0.09, 0.09])


def test_mooring_resultant_equals_sum_of_recovered_fairlead_forces():
    m = _mooring()
    rng = np.random.default_rng(3)
    q = np.concatenate([rng.uniform(-1, 1, 3), rng.uniform(-0.05, 0.05, 3)])
    v = rng.uniform(-0.5, 0.5, 6)
    forces = m.fairlead_forces(q, v)
    assert set(forces) == {"line-A", "line-B"}
    w = np.zeros(6)
    for f in m.fairleads:
        w[:3] += forces[f.identity]
        w[3:] += np.cross(f.point_body, forces[f.identity])
    assert np.allclose(w, m.generalized_force(q, v), rtol=1e-13, atol=1e-8)
    # the affine solver form is identical at first order
    assert np.allclose(m.linear_generalized_force(q, v), m.generalized_force(q, v), rtol=1e-12, atol=1e-6)


def test_mooring_stiffness_matches_finite_difference_and_virtual_work():
    m = _mooring()
    K = m.stiffness_matrix()
    base = m.pretension_wrench()
    assert np.allclose(m.generalized_force(np.zeros(6)), base)
    h = 1e-3
    for j in range(6):
        e = np.zeros(6)
        e[j] = h
        col = -(m.generalized_force(e) - m.generalized_force(-e)) / (2 * h)
        assert np.allclose(col, K[:, j], rtol=1e-9, atol=1e-6)
    # without pretension, K = sum A^T K_i A exactly
    fl0 = tuple(FairleadSpec(f.identity, f.point_body, [0, 0, 0], f.stiffness, f.damping) for f in m.fairleads)
    m0 = AffineFairleadMooring(fl0, np.eye(3), validity_half_range=[1] * 6)
    expected = sum(rigid_attachment_map(f.point_body).T @ f.stiffness @ rigid_attachment_map(f.point_body)
                   for f in fl0)
    assert np.allclose(m0.stiffness_matrix(), expected, rtol=1e-14)
    assert m0.asymmetry()["stiffness_skew_norm"] > 0.0  # reported, never symmetrised
    C = m.damping_matrix()
    assert np.allclose(C, rigid_attachment_map(m.fairleads[0].point_body).T @ m.fairleads[0].damping
                       @ rigid_attachment_map(m.fairleads[0].point_body))


def test_mooring_pretension_geometric_term_against_closed_form():
    # one fairlead, world pretension F0, zero stiffness: wrench(theta) = [f0 + f0 x th ; r x (f0 + f0 x th)]
    r = np.array([10.0, -4.0, -6.0])
    F0 = np.array([2.0e5, 1.0e5, -3.0e5])
    m = AffineFairleadMooring((FairleadSpec("L", r, F0, np.zeros((3, 3))),), np.eye(3),
                              validity_half_range=[1] * 6)
    th = np.array([0.01, -0.02, 0.03])
    expected_f = F0 + np.cross(F0, th)
    expected = np.concatenate([expected_f, np.cross(r, expected_f)])
    assert np.allclose(m.generalized_force(np.concatenate([np.zeros(3), th])), expected)
    assert np.allclose(m.linear_generalized_force(np.concatenate([np.zeros(3), th])), expected)


def test_reflection_negative_damping_and_nonfinite_state_are_refused():
    k = np.eye(3)
    ok = FairleadSpec("L", [0, 0, 0], [0, 0, 0], k)
    with pytest.raises(ExternalHistoryError, match="proper rotation"):
        AffineFairleadMooring((ok,), np.diag([1.0, 1.0, -1.0]), validity_half_range=[1] * 6)
    with pytest.raises(ExternalHistoryError, match="passive"):
        FairleadSpec("L", [0, 0, 0], [0, 0, 0], k, -np.eye(3))
    # a nonsymmetric damping with a nonnegative symmetric part is still passive
    FairleadSpec("L", [0, 0, 0], [0, 0, 0], k, [[1.0, 0.5, 0], [-0.5, 1.0, 0], [0, 0, 1.0]])
    with pytest.raises(ExternalHistoryError):
        AffineFairleadMooring((ok,), np.eye(3), validity_half_range=[1, 1, 1, -1, 1, 1])
    m = AffineFairleadMooring((ok,), np.eye(3), validity_half_range=[1] * 6)
    for bad in (np.full(6, np.nan), np.array([0, 0, np.inf, 0, 0, 0]), np.zeros(5)):
        with pytest.raises(ExternalHistoryError):
            m.check_validity(bad)
        with pytest.raises(ExternalHistoryError):
            m.generalized_force(bad)
    with pytest.raises(ExternalHistoryError):
        m.fairlead_forces(np.zeros(6), np.full(6, np.nan))
    with pytest.raises(ExternalHistoryError, match="finite"):
        _history().at(np.nan)


def test_complex_load_history_mooring_and_rotation_inputs_are_refused():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ExternalHistoryError, match="complex"):
            _history(values=np.array([[0.0, 1.0], [2.0, 1.0], [4.0, -1.0], [10.0, -1.0]]) * (1 + 0j))
        with pytest.raises(ExternalHistoryError, match="complex"):
            _history(times=np.array([0.0, 2.0, 4.0, 10.0]) + 0j)
        h = _history()
        with pytest.raises(ExternalHistoryError, match="complex"):
            h.at(np.array([1.0 + 0j]))
        with pytest.raises(ExternalHistoryError, match="complex"):
            h.generalized_force([2.0], np.eye(3) * (1 + 0j))
        with pytest.raises(ExternalHistoryError, match="complex"):
            rigid_attachment_map([0, 0, 1 + 0j])
        with pytest.raises(ExternalHistoryError, match="complex"):
            world_force_pullback([1.0 + 1j, 0, 0], np.eye(3), [0, 0, 1.0])
        m = _mooring()
        with pytest.raises(ExternalHistoryError, match="complex"):
            m.check_validity(np.zeros(6, dtype=complex))
        with pytest.raises(ExternalHistoryError, match="complex"):
            FairleadSpec("L", [0, 0, 0], [0, 0, 0], np.eye(3) * (1 + 0j))


def test_operational_mooring_methods_require_a_declared_envelope_and_pose():
    base = _mooring()
    undeclared = AffineFairleadMooring(base.fairleads, np.eye(3))
    for call in (undeclared.pretension_wrench, undeclared.stiffness_matrix, undeclared.damping_matrix,
                 lambda: undeclared.fairlead_forces(np.zeros(6)), lambda: undeclared.generalized_force(np.zeros(6)),
                 lambda: undeclared.linear_generalized_force(np.zeros(6))):
        with pytest.raises(ExternalHistoryError, match="envelope"):
            call()
    undeclared.geometric_tangent()          # inspection helpers stay available
    undeclared.identities
    outside = np.array([0, 0, 0, 0.5, 0, 0.0])
    for call in (base.fairlead_forces, base.generalized_force, base.linear_generalized_force):
        with pytest.raises(ExternalHistoryError, match="envelope"):
            call(outside)
    base.generalized_force(np.zeros(6))


def test_load_history_identity_hash_is_canonical_and_q0_is_validated():
    a = _history(identity={"x": {"b": 1, "a": [1, 2]}, "y": "z"})
    b = _history(identity={"y": "z", "x": {"a": [1, 2], "b": 1}})
    c = _history(identity={"y": "z", "x": {"a": [1, 3], "b": 1}})
    assert a.content_hash() == b.content_hash() != c.content_hash()
    for bad in (np.diag([1.0, 1.0, -1.0]), np.full((3, 3), np.nan), np.eye(2), 2.0 * np.eye(3)):
        with pytest.raises(ExternalHistoryError, match="Q0"):
            a.generalized_force([2.0], bad)
        with pytest.raises(ExternalHistoryError, match="Q0"):
            a.load_tangent([2.0], bad)
        with pytest.raises(ExternalHistoryError, match="Q0"):
            world_force_pullback([1.0, 0, 0], bad, [0, 0, 1.0])


def test_mooring_validity_envelope_and_identity_refusals():
    m = _mooring()
    m.check_validity(np.zeros(6))
    with pytest.raises(ExternalHistoryError, match="envelope"):
        m.check_validity([0, 0, 0, 0.2, 0, 0])
    undeclared = AffineFairleadMooring(m.fairleads, np.eye(3))
    with pytest.raises(ExternalHistoryError, match="validity"):
        undeclared.check_validity(np.zeros(6))
    with pytest.raises(ExternalHistoryError, match="unique"):
        AffineFairleadMooring((m.fairleads[0], m.fairleads[0]), np.eye(3))
    with pytest.raises(ExternalHistoryError):
        AffineFairleadMooring((), np.eye(3))
    with pytest.raises(ExternalHistoryError):
        FairleadSpec("", [0, 0, 0], [0, 0, 0], np.eye(3))
