from types import SimpleNamespace
import math
import numpy as np
import pytest
from anyloads.orcawave_spectrum import native_frequency_realization,regular_realization


def bank(w=(.4,.8,1.2,2.)):
    return SimpleNamespace(omegas=np.array(w),heading=0.,validate=lambda:None)


def test_pm_native_bin_energy_analytic():
    r,rep=native_frequency_realization(bank(),significant_height=4.,peak_period=8.,gamma=1.,seed=31)
    wp=math.tau/8; edges=np.asarray(rep['bin_edges'])
    exact=np.exp(-1.25*(wp/edges[1:])**4)-np.exp(-1.25*(wp/edges[:-1])**4)
    assert np.allclose(rep['bin_energy'],exact,rtol=2e-13,atol=1e-15)
    assert sum(c.amplitude**2/2 for c in r.components)==pytest.approx(exact.sum())
    assert rep['achieved_significant_height']<4 and rep['omitted_variance']>0
    assert not rep['renormalized']


def test_jonswap_replay_zero_and_validation():
    a,ra=native_frequency_realization(bank(),significant_height=2.,peak_period=8.,seed=4)
    b,rb=native_frequency_realization(bank(),significant_height=2.,peak_period=8.,seed=4)
    assert a.to_dict()==b.to_dict() and ra==rb
    assert 0<ra['retained_variance']<.25
    zero,rz=native_frequency_realization(bank(),significant_height=0.,peak_period=8.)
    assert all(c.amplitude==0 for c in zero.components) and rz['retained_variance']==0
    with pytest.raises(ValueError):native_frequency_realization(bank((1.,)),significant_height=1,peak_period=8)
    with pytest.raises(ValueError):regular_realization(bank(),omega=.7,height=1)
    assert regular_realization(bank(),omega=.8,height=2).components[0].amplitude==1
