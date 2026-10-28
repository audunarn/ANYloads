"""TransientFloatingLoadHistory: actual times, exact selection, integrity, refusal."""
from types import SimpleNamespace

import numpy as np
import pytest

from anyloads.transient_floating_history import (
    LoadComponent,
    TransientFloatingLoadHistory,
    TransientHistoryError,
    TransientHistoryLimits,
    hash_times,
    load_transient_history,
)

IDS = dict(body="hull-1", mesh="mesh-abc", mass="mass-7", wind="wind-9", curve="curve-3", hydro="hydro-5")
N = 12


def _result(status="complete"):
    t = 0.5 * np.arange(N) + 3.0                       # actual seconds: not starting at zero
    q = np.linspace(0, 1, N)[:, None] * np.arange(1, 7)[None]
    return SimpleNamespace(times=t, displacements=q, velocities=2 * q, accelerations=3 * q,
                           hub_wind_world=np.tile([10.0, 0, 0], (N, 1)), relative_axial_speed=10.0 - q[:, 0],
                           thrust=1e5 + q[:, 0], status=status, message="ok" if status == "complete" else "failed")


def _components():
    force = np.arange(N * 4 * 3, dtype=float).reshape(N, 4, 3)
    pres = np.arange(N * 5, dtype=float).reshape(N, 5, 1)
    return (LoadComponent("turbine", "nodal_force", "N", "body", "hub-interface", ("n1", "n2", "n3", "n4"), force),
            LoadComponent("pressure", "pressure", "Pa", "body", "wet-panels", tuple(f"p{i}" for i in range(5)), pres))


def _record(**kw):
    return TransientFloatingLoadHistory.from_solver_result(_result(**kw), _components(), IDS)


def test_carries_actual_times_identities_and_components():
    h = _record()
    r = _result()
    assert np.array_equal(h.times, r.times) and h.times[0] == 3.0
    assert h.source_times_hash == hash_times(r.times)
    assert h.identities["mesh"] == "mesh-abc" and h.qualification_status == "unqualified"
    assert h.component("turbine").values.shape == (N, 4, 3) and h.component("pressure").unit == "Pa"
    assert h.n_frames == N and h.accepted_for_structural_output
    assert h.require_accepted() is h
    assert np.array_equal(h.primary_acceleration, r.accelerations)


def test_arrays_are_immutable_and_detached_from_inputs():
    r = _result()
    comps = _components()
    h = TransientFloatingLoadHistory.from_solver_result(r, comps, IDS)
    r.displacements[0, 0] = 99.0
    r.times[0] = -1.0
    assert h.displacement[0, 0] == 0.0 and h.times[0] == 3.0
    for arr in (h.times, h.displacement, h.component("turbine").values, h.frame_indices):
        with pytest.raises(ValueError):
            arr[0] = 1
        with pytest.raises(ValueError):
            arr.setflags(write=True)
    with pytest.raises(TypeError):
        h.identities["mesh"] = "other"


def test_exact_frame_and_window_selection_never_interpolate_or_reset():
    h = _record()
    w = h.select_window(4.0, 6.0)                       # stored times 4.0, 4.5, ..., 6.0
    assert np.array_equal(w.times, h.times[2:7]) and np.array_equal(w.frame_indices, np.arange(2, 7))
    assert np.array_equal(w.component("pressure").values, h.component("pressure").values[2:7])
    assert w.source_times_hash == h.source_times_hash and w.source_frame_count == N
    assert w.window["kind"] == "window" and w.window["origin_time"] == 3.0
    assert w.window["t_start"] == 4.0 and w.window["t_stop"] == 6.0
    nested = w.select_frames([1, 3])
    assert np.array_equal(nested.frame_indices, [3, 5]) and nested.window["origin_time"] == 3.0
    with pytest.raises(TransientHistoryError):
        h.select_window(100.0, 200.0)
    with pytest.raises(TransientHistoryError):
        h.select_frames([3, 3])
    with pytest.raises(TransientHistoryError):
        h.select_frames([0, N])
    with pytest.raises(TransientHistoryError):
        h.select_frames([0.5])


def test_bool_and_fractional_frame_indices_are_refused_in_construction_and_selection():
    h = _record()
    fields = dict(times=h.times, source_frame_count=h.source_frame_count, source_times_hash=h.source_times_hash,
                  components=h.components, primary_acceleration=h.primary_acceleration, displacement=h.displacement,
                  velocity=h.velocity, acceleration=h.acceleration, wind_velocity=h.wind_velocity,
                  relative_axial_speed=h.relative_axial_speed, thrust=h.thrust, identities=h.identities)
    for bad in (np.ones(N, dtype=bool), np.arange(N) + 0.5, np.arange(N, dtype=float), np.array(list(range(N)), dtype=object)):
        with pytest.raises(TransientHistoryError, match="integer"):
            TransientFloatingLoadHistory(frame_indices=bad, **fields)
    for bad in ([True, False], np.array([True, True]), [0.5, 1.5], np.array([1.0, 2.0])):
        with pytest.raises(TransientHistoryError):
            h.select_frames(bad)


def test_complex_frames_times_and_selection_bounds_are_refused():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TransientHistoryError, match="complex"):
            LoadComponent("f", "nodal_force", "N", "body", "n", ("a",), np.ones((2, 1, 3), complex) * (1.0 + 2j))
        with pytest.raises(TransientHistoryError, match="complex"):
            LoadComponent("f", "nodal_force", "N", "body", "n", ("a",), np.ones((2, 1, 3), complex))   # zero imaginary part
        bad = _result(); bad.displacements = bad.displacements.astype(complex)
        with pytest.raises(TransientHistoryError, match="complex"):
            TransientFloatingLoadHistory.from_solver_result(bad, _components(), IDS)
        bad = _result(); bad.times = bad.times.astype(complex)
        with pytest.raises(TransientHistoryError, match="complex"):
            TransientFloatingLoadHistory.from_solver_result(bad, _components(), IDS)
        with pytest.raises(TransientHistoryError, match="complex"):
            hash_times(np.array([0.0, 1.0 + 0j]))
        with pytest.raises(TransientHistoryError, match="complex"):
            _record().select_window(0.0 + 0j, 5.0)


def test_non_success_is_never_accepted_structural_output():
    h = _record(status="no_root")
    assert not h.accepted_for_structural_output
    with pytest.raises(TransientHistoryError, match="not accepted"):
        h.require_accepted()


def test_construction_refusals():
    r = _result()
    base = dict(components=_components(), identities=IDS)
    with pytest.raises(TransientHistoryError, match="identities"):
        TransientFloatingLoadHistory.from_solver_result(r, base["components"], {k: v for k, v in IDS.items() if k != "curve"})
    bad = _result(); bad.times = bad.times.copy(); bad.times[4] = bad.times[3]
    with pytest.raises(TransientHistoryError, match="strictly increasing"):
        TransientFloatingLoadHistory.from_solver_result(bad, base["components"], IDS)
    bad = _result(); bad.displacements = bad.displacements.copy(); bad.displacements[2, 2] = np.nan
    with pytest.raises(TransientHistoryError, match="non-finite"):
        TransientFloatingLoadHistory.from_solver_result(bad, base["components"], IDS)
    short = (LoadComponent("x", "nodal_force", "N", "body", "a", ("n1",), np.zeros((N - 1, 1, 3))),)
    with pytest.raises(TransientHistoryError, match="frames"):
        TransientFloatingLoadHistory.from_solver_result(r, short, IDS)
    with pytest.raises(TransientHistoryError, match="3 values"):
        LoadComponent("x", "nodal_force", "N", "body", "a", ("n1",), np.zeros((N, 1, 2)))
    with pytest.raises(TransientHistoryError, match="unique"):
        TransientFloatingLoadHistory.from_solver_result(r, _components() + _components()[:1], IDS)


def test_allocation_is_bounded_before_large_copies():
    with pytest.raises(TransientHistoryError, match="ceiling"):
        TransientFloatingLoadHistory.from_solver_result(_result(), _components(), IDS,
                                                        limits=TransientHistoryLimits(max_bytes=1000))
    with pytest.raises(TransientHistoryError):
        TransientHistoryLimits(max_bytes=2 * 1024**3 + 1)        # the default ceiling can only tighten


def test_roundtrip_hash_and_safe_loading(tmp_path):
    h = _record().select_window(4.0, 7.0)
    path = str(tmp_path / "h.npz")
    digest = h.save(path)
    g = load_transient_history(path, expected_hash=digest)
    assert g.content_hash() == digest == h.content_hash()
    assert np.array_equal(g.times, h.times) and g.window == h.window and g.identities == h.identities
    assert np.array_equal(g.component("turbine").values, h.component("turbine").values)
    assert g.component("turbine").item_ids == ("n1", "n2", "n3", "n4")
    # stale input: a different expected hash is refused
    with pytest.raises(TransientHistoryError, match="stale"):
        load_transient_history(path, expected_hash="0" * 64)
    # a non-success status survives the roundtrip and stays non-accepted
    bad = _record(status="multiple_roots")
    p2 = str(tmp_path / "bad.npz")
    bad.save(p2)
    assert not load_transient_history(p2).accepted_for_structural_output


def test_tamper_truncation_and_ceiling_are_detected(tmp_path):
    h = _record()
    path = tmp_path / "h.npz"
    h.save(str(path))
    with np.load(str(path), allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["thrust"] = data["thrust"].copy()
    data["thrust"][3] += 1e-9
    tampered = tmp_path / "t.npz"
    np.savez(str(tampered), **data)
    with pytest.raises(TransientHistoryError, match="hash mismatch"):
        load_transient_history(str(tampered))
    cut = tmp_path / "c.npz"
    cut.write_bytes(path.read_bytes()[:1500])
    with pytest.raises(TransientHistoryError):
        load_transient_history(str(cut))
    with pytest.raises(TransientHistoryError, match="ceiling"):
        load_transient_history(str(path), limits=TransientHistoryLimits(max_bytes=500))
    with pytest.raises(TransientHistoryError, match="ceiling"):
        TransientFloatingLoadHistory.from_solver_result(_result(), _components(), IDS,
                                                        limits=TransientHistoryLimits(max_bytes=500))
    assert load_transient_history(str(path)).content_hash() == h.content_hash()
