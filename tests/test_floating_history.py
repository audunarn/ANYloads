import io
import json

import numpy as np
import pytest

from anyloads.floating_history import BINDING_KEYS, CONTRIBUTIONS, FloatingCoupledLoadHistory
from anyloads.pressure_history import pressure_history_from_dict

SHA = 'a' * 64


def make(amplitude=(2.0, 1.0), **over):
    nk, n = 2, 3
    rng = np.random.default_rng(1)
    loads = {c: rng.normal(size=(nk, n, 6)) + 1j * rng.normal(size=(nk, n, 6)) for c in CONTRIBUTIONS}
    args = dict(omega=[1.0, 2.0], amplitude=amplitude, phase=[0.3, 0.0], loads=loads, static_load=np.ones((n, 6)),
                node_ids=(10, 11, 12), mesh_content_sha256=SHA, binding={k: f'{k}-id' for k in BINDING_KEYS})
    args.update(over)
    return FloatingCoupledLoadHistory(**args), loads


def payload_of(h, **extra):
    """Minimal prepared-record layout understood by ``from_payload`` (ANYfem writes a superset of these members)."""
    buffer = io.BytesIO()
    from anyloads.floating_history import _thaw
    meta = dict(node_ids=list(h.node_ids), mesh_content_sha256=h.mesh_content_sha256, binding=_thaw(h.binding),
                array_sha256=h.array_sha256)
    meta.update(extra)
    np.savez(buffer, meta=json.dumps(meta), omega=h.omega, amplitude=h.amplitude, phase=h.phase,
             pressure=h.loads['pressure'], weight=h.loads['weight'], inertia=h.loads['inertia'], static=h.static_load)
    return buffer.getvalue()


def test_evaluate_is_real_sum_of_components_plus_static_and_immutable():
    h, loads = make()
    t = 0.7
    z = np.array([2.0 * np.exp(0.3j) * np.exp(-0.7j), 1.0 * np.exp(-1.4j)])
    expect = sum(np.real(np.tensordot(z, loads[c], axes=(0, 0))) for c in CONTRIBUTIONS) + 1.0
    np.testing.assert_allclose(h.evaluate(t), expect)
    np.testing.assert_allclose(h.evaluate(t, include_static=False) + h.static_load, h.evaluate(t))
    assert not h.loads['pressure'].flags.writeable and h.content_sha256
    with pytest.raises(ValueError):
        h.evaluate(t, ('bogus',))
    with pytest.raises(ValueError):
        h.loads['pressure'].setflags(write=True)                                 # bytes-backed: cannot be made writable


def test_zero_amplitude_is_static_and_identity_changes_with_content():
    h, _ = make(amplitude=(0.0, 0.0))
    assert h.is_static
    np.testing.assert_allclose(h.evaluate(3.0), h.static_load)
    a, _ = make()
    b, _ = make(phase=[0.4, 0.0])
    assert a.content_sha256 != b.content_sha256


def test_refuses_missing_binding_zero_frequency_and_tampered_identity():
    with pytest.raises(ValueError, match='binding'):
        make(binding={k: 'x' for k in BINDING_KEYS[:-1]})
    with pytest.raises(ValueError):
        make(omega=[0.0, 2.0])
    with pytest.raises(ValueError):
        make(content_sha256='b' * 64)


def test_selected_times_payload_persistence_and_dispatcher_roundtrip():
    base, _ = make()
    payload = payload_of(base)
    h = FloatingCoupledLoadHistory.from_payload(payload, times=(0.0, 0.5, 1.0), selected_window=(0.0, 1.0))
    assert h.times == (0.0, 0.5, 1.0) and h.payload == payload and h.array_sha256 == base.array_sha256
    data = h.to_dict()
    json.dumps(data, allow_nan=False)                                            # JSON-safe project record
    back = pressure_history_from_dict(data)                                      # exact-class dispatcher
    assert type(back) is FloatingCoupledLoadHistory and back.content_sha256 == h.content_sha256
    np.testing.assert_allclose(back.evaluate(0.5), h.evaluate(0.5))
    other = h.with_selection((0.0, 1.0), (0.0, 1.0))
    assert other.content_sha256 != h.content_sha256 and other.array_sha256 == h.array_sha256


def test_persistence_refuses_tampering_and_unselected_or_unembedded_loads():
    base, _ = make()
    h = FloatingCoupledLoadHistory.from_payload(payload_of(base), times=(0.0, 1.0), selected_window=(0.0, 1.0))
    data = h.to_dict()
    with pytest.raises(ValueError):
        pressure_history_from_dict({**data, 'content_sha256': 'c' * 64})
    with pytest.raises(ValueError):
        pressure_history_from_dict({**data, 'times': [0.0, 2.0]})
    with pytest.raises(ValueError):
        base.to_dict()                                                            # no embedded payload: not a project load
    with pytest.raises(ValueError):                                               # recorded identity no longer matches the arrays
        FloatingCoupledLoadHistory.from_payload(payload_of(base, array_sha256='0' * 64))
    with pytest.raises(ValueError):
        FloatingCoupledLoadHistory(**{**{k: getattr(base, k) for k in ('omega', 'amplitude', 'phase', 'loads', 'static_load',
                                                                        'node_ids', 'mesh_content_sha256', 'binding')},
                                      'times': (1.0, 0.5)})
