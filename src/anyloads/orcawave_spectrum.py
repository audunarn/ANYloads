"""Native-frequency waves with continuous-spectrum normalization and truncation.

The existing centered-bin JONSWAP API is deliberately unchanged. Actual native
bins are asymmetric, so their metadata is returned separately from the exact
materialized WaveRealization (declared components).
"""
import math
import warnings
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from .sea_state import WaveRealization, RegularWave, JonswapSpectrum
from .marine import AiryComponent


def regular_realization(bank, *, omega, height, phase=0.):
    bank.validate()
    if omega not in bank.omegas: raise ValueError('exact regular-wave frequency absent from bank')
    return RegularWave(height/2,omega,bank.heading,phase).realize()


def _energy_integral(a,b,gamma):
    # x=(omega_peak/omega)^4 transforms the PM factor into exp(-1.25*x).
    if gamma==1.:
        return (math.exp(-1.25*a)-(0. if math.isinf(b) else math.exp(-1.25*b)))/1.25
    def density(x):
        if x==0.: return 1.
        r=x**(-.25); sigma=.07 if r<=1. else .09
        peak=math.exp(-.5*((r-1)/sigma)**2)
        return math.exp(-1.25*x+math.log(gamma)*peak)
    boundaries=[a]+[v for v in (.25,.5,1.,2.,4.,16.,64.) if a<v<b]+[b]
    total=0.
    with warnings.catch_warnings():
        warnings.simplefilter('error',IntegrationWarning)
        for lo,hi in zip(boundaries,boundaries[1:]):
            value,error=quad(density,lo,hi,epsabs=1e-12,epsrel=1e-11,limit=150)
            if not math.isfinite(value) or error>max(1e-11,abs(value)*1e-10):
                raise ValueError('JONSWAP energy integration did not converge')
            total+=value
    return total


def native_frequency_realization(bank, *, significant_height,peak_period,gamma=3.3,seed=0):
    bank.validate()
    spectrum=JonswapSpectrum(significant_height,peak_period,gamma)
    if type(seed) is not int or seed<0: raise ValueError('nonnegative integer seed required')
    w=np.asarray(bank.omegas,dtype=float)
    if len(w)<2 or np.any(np.diff(w)<=0): raise ValueError('at least two ordered native frequencies required')
    edges=np.r_[w[0],w[:-1]+np.diff(w)/2,w[-1]]
    wp=math.tau/spectrum.peak_period
    total=_energy_integral(0.,math.inf,spectrum.gamma)
    fractions=np.array([_energy_integral((wp/b)**4,(wp/a)**4,spectrum.gamma)/total
                       for a,b in zip(edges,edges[1:])])
    fraction=float(fractions.sum())
    if not 0<=fraction<=1+1e-10: raise ValueError('invalid retained spectral energy')
    m0=(spectrum.significant_height/4)**2
    energy=fractions*m0
    if not np.isfinite(energy).all(): raise ValueError('spectral energy outside floating point range')
    phases=np.random.Generator(np.random.PCG64(seed)).uniform(0,math.tau,len(w))
    realization=WaveRealization(tuple(AiryComponent(float(math.sqrt(2*e)),float(omega),
                              math.degrees(bank.heading),float(phi)) for omega,e,phi in zip(w,energy,phases)))
    report=dict(mode='native_integrated_truncated_v1',seed=seed,significant_height=spectrum.significant_height,
        peak_period=spectrum.peak_period,gamma=spectrum.gamma,bin_edges=edges.tolist(),bin_energy=energy.tolist(),
        total_variance=m0,retained_variance=float(energy.sum()),omitted_variance=max(0.,m0-float(energy.sum())),
        retained_fraction=fraction,omitted_fraction=max(0.,1.-fraction),achieved_significant_height=4*math.sqrt(float(energy.sum())),
        phase_algorithm='numpy.PCG64',renormalized=False)
    return realization,report
