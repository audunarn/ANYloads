import json
import numpy as np
import pytest
from anyloads.prescribed_pressure import PrescribedPressureHistory
from anyloads.pressure_history import pressure_history_from_dict
from anyloads.sea_state import RegularWave


def history(**overrides):
    z=np.zeros((1,2,6),complex); z[0,0,0]=1+2j; z[0,1,5]=3-4j
    kw=dict(times=(0.,.1,.2),node_ids=(3,5),omegas=(1.,),responses=z,
        realization_json=json.dumps(RegularWave(2.,1.,0.,.3).realize().to_dict()),mesh_content_sha256='a'*64,
        source_payload_json=json.dumps(dict(composition='total_with_source_motion',bank_id='b'*64,heading=0.)),
        binding_json=json.dumps(dict(preview_element_ids=[7],preview_pressure_real=[[1.]],preview_pressure_imag=[[2.]])))
    kw.update(overrides); return PrescribedPressureHistory(**kw)


def test_complex_synthesis_preserves_six_dof_and_preview():
    h=history(); expected=np.real(2*np.exp(.1j)*h.responses[0])
    assert np.allclose(h.evaluate(.2),expected)
    assert h.preview_pressures(.2)[0]==pytest.approx(expected[0,0])
    assert np.array_equal(np.concatenate(list(h.iter_blocks(h.times,block_size=1))),np.stack([h.evaluate(t) for t in h.times]))
    assert not h.responses.flags.writeable


def test_portable_dispatch_identity_and_refusals():
    h=history(); restored=pressure_history_from_dict(h.to_dict())
    assert isinstance(restored,PrescribedPressureHistory)
    assert restored.content_sha256==h.content_sha256
    assert np.array_equal(restored.evaluate(.1),h.evaluate(.1))
    bad=h.to_dict();bad['responses_real'][0][0][0]=99
    with pytest.raises(ValueError,match='identity'):PrescribedPressureHistory.from_dict(bad)
    with pytest.raises(ValueError,match='Nyquist'):history(times=(0.,4.))
    with pytest.raises(ValueError,match='outside'):h.evaluate(1.)
    with pytest.raises(ValueError,match='workspace'):history(working_memory_bytes=4)
    with pytest.raises(RuntimeError,match='cancelled'):list(h.iter_blocks(h.times,cancel=lambda:True))
