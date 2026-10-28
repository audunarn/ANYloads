import numpy as np
import pytest
from types import SimpleNamespace
from anyloads.hydro_pressure import transfer_panel_pressure


def source():
    return SimpleNamespace(vertices=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),
        faces=np.array([[0,1,2,3]]),surface_ids=('bottom',),centers=np.array([[.5,.5,0.]]),
        normals=np.array([[0.,0.,-1.]]),areas=np.array([1.]),pressure=np.array([2.+3.j]),
        source_id='fixture',omega=2.)


def transfer(**kwargs):
    args=dict(element_ids=[1],centers=[[.5,.5,0]],normals=[[0,0,-1]],areas=[1.],surface_ids=['bottom'])
    args.update(kwargs)
    return transfer_panel_pressure(source(),**args)


def test_phasor_sign_amplitude_and_wrench():
    p=transfer()
    assert p.snapshots([0.,np.pi/4],amplitude=.5)[:,0] == pytest.approx([1.,1.5])
    assert p.target_wrench == pytest.approx(p.source_wrench)
    assert p.source_wrench[2] == 2.+3.j


@pytest.mark.parametrize('options',[
    {'centers':[[2.,.5,0]]},{'centers':[[.5,.5,.2]]},
    {'areas':[.5]},{'normals':[[0,0,1]]},{'normals':[[0,0,-2]]},
    {'surface_ids':['deck']},{'element_ids':[1.5]},
])
def test_refuses_domain_sign_and_identity_errors(options):
    with pytest.raises(ValueError):
        transfer(**options)


def test_frequency_is_bound_and_negative_dynamic_pressure_preserved():
    p=transfer()
    assert p.snapshots([np.pi/2])[0,0] == pytest.approx(-1.)
    with pytest.raises(ValueError,match='frequency'):
        p.snapshots([0],omega=3.)


def test_refuses_nonfinite_source_payload():
    s=source()
    s.normals[0,0]=np.nan
    with pytest.raises(ValueError,match='source pressure/geometry'):
        transfer_panel_pressure(s,[1],[[.5,.5,0]],[[0,0,-1]],[1.],['bottom'])
