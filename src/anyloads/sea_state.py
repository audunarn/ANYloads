"""Named finite-bin sea states with authoritative materialized amplitudes/phases.

Angular-frequency density is m²/(rad/s). JONSWAP bin normalization is explicit;
it does not qualify truncation, frequency resolution or a continuous spectrum.
Legacy AiryComponent degree headings are converted only at the record boundary.
"""
from dataclasses import dataclass, fields
import hashlib
import json
import math
import numpy as np

from .errors import LoadError
from .marine import AiryComponent, AiryEnvironment, _real

SEA_STATE_VERSION = 'marine_wave_realization_v1'
PHASE_ALGORITHM = 'PCG64_uniform_0_2pi_materialized_v1'


def _values(value, name, *, positive=False, nonnegative=False):
    try:
        if any(isinstance(x,(bool,np.bool_)) for x in np.asarray(value,dtype=object).flat):
            raise ValueError()
        raw = np.asarray(value)
        if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in 'iuf':
            raise ValueError()
        return tuple(_real(x, name, positive=positive, nonnegative=nonnegative) for x in raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise LoadError(f'{name} needs a nonempty finite real vector') from error


def _record(data, kind):
    expected = {f.name for f in fields(kind) if f.init}
    if type(data) is not dict or set(data) != expected:
        raise LoadError(f'{kind.__name__} must match its exact named fields')
    return kind(**data)


def _plain(value):
    if hasattr(value, '__dataclass_fields__'):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value) if f.init}
    if isinstance(value, tuple):
        return [_plain(x) for x in value]
    return value


@dataclass(frozen=True)
class SpectralBins:
    omegas: tuple
    density_per_radian: tuple
    bandwidths: tuple

    def __post_init__(self):
        for key, positive in [('omegas', True), ('bandwidths', True), ('density_per_radian', False)]:
            object.__setattr__(self, key, _values(getattr(self, key), key,
                positive=positive, nonnegative=not positive))
        if not len(self.omegas) == len(self.density_per_radian) == len(self.bandwidths):
            raise LoadError('spectral bin vectors must have equal lengths')
        w, dw = np.asarray(self.omegas), np.asarray(self.bandwidths)
        with np.errstate(over='ignore',invalid='ignore'):
            lower=w-dw/2;upper=w+dw/2
        if (not np.isfinite(lower).all() or not np.isfinite(upper).all() or
                np.any(lower<=0.) or np.any(np.diff(w)<=0.) or any(
                    left>right and left-right>4*max(math.ulp(float(left)),math.ulp(float(right)))
                    for left,right in zip(upper[:-1],lower[1:]))):
            raise LoadError('spectral bins must be positive, ordered and nonoverlapping')
        if not math.isfinite(self.variance):
            raise LoadError('spectral variance exceeds the finite supported range')

    @property
    def variance(self):
        return float(np.dot(self.density_per_radian, self.bandwidths))


@dataclass(frozen=True)
class DirectionalSpectrum:
    """Explicit discrete energy fractions at canonical propagation headings."""
    headings_radians: tuple = (0.,)
    weights: tuple = (1.,)

    def __post_init__(self):
        headings = _values(self.headings_radians, 'headings')
        weights = _values(self.weights, 'direction energy weights', nonnegative=True)
        if (len(headings) != len(weights) or any(not 0. <= x < math.tau for x in headings)
                or not math.isclose(math.fsum(weights), 1., rel_tol=1e-14, abs_tol=1e-14)):
            raise LoadError('directions need canonical [0,2pi) radians and weights summing to one')
        for i, x in enumerate(headings):
            if any(abs(math.remainder(x-y, math.tau)) <= 1e-12 for y in headings[:i]):
                raise LoadError('ambiguous duplicate propagation directions')
        object.__setattr__(self, 'headings_radians', headings)
        object.__setattr__(self, 'weights', weights)


@dataclass(frozen=True)
class JonswapSpectrum:
    significant_height: float
    peak_period: float
    gamma: float = 3.3
    sigma_low: float = .07
    sigma_high: float = .09
    gravity: float = 9.81

    def __post_init__(self):
        for f in fields(self):
            object.__setattr__(self, f.name, _real(getattr(self, f.name), f.name,
                positive=f.name != 'significant_height', nonnegative=f.name == 'significant_height'))
        if self.gamma < 1.:
            raise LoadError('JONSWAP peak enhancement gamma must be at least one')
        _real(math.tau/self.peak_period, 'peak angular frequency', positive=True)

    def density(self, omegas, *, alpha):
        """Sω=alpha*g²*ω^-5 exp[-1.25(ωp/ω)^4] gamma^b."""
        w = np.asarray(_values(omegas, 'omega', positive=True))
        alpha = _real(alpha, 'Phillips alpha', nonnegative=True)
        wp = math.tau/self.peak_period
        try:
            with np.errstate(over='raise', invalid='raise', under='ignore'):
                sigma = np.where(w <= wp, self.sigma_low, self.sigma_high)
                b = np.exp(-.5*((w/wp-1.)/sigma)**2)
                shape = np.exp(2.*math.log(self.gravity)-5.*np.log(w)
                    -1.25*(wp/w)**4 + math.log(self.gamma)*b)
                result = alpha*shape
        except FloatingPointError as error:
            raise LoadError('JONSWAP density exceeds the finite supported range') from error
        if not np.isfinite(result).all():
            raise LoadError('JONSWAP density is nonfinite')
        return result

    def discretize(self, omegas, bandwidths):
        """Normalize these declared finite bins to m0=Hs²/16, no tail assumption."""
        shape = self.density(omegas, alpha=1.)
        bins = SpectralBins(omegas, shape, bandwidths)
        try:
            target = (self.significant_height/4.)**2
        except OverflowError:
            raise LoadError('target variance exceeds the finite supported range') from None
        if (not math.isfinite(target) or self.significant_height > 0. and target == 0.
                or bins.variance <= 0.):
            raise LoadError('JONSWAP bins/target variance are not representable; choose an admitted grid')
        with np.errstate(over='ignore', invalid='ignore'):
            density = shape*(target/bins.variance)
        return SpectralBins(bins.omegas, density, bins.bandwidths)


@dataclass(frozen=True)
class RegularWave:
    amplitude: float
    omega: float
    heading_radians: float = 0.
    phase: float = 0.

    def __post_init__(self):
        for key in ('amplitude', 'omega', 'heading_radians', 'phase'):
            object.__setattr__(self, key, _real(getattr(self, key), key,
                positive=key == 'omega', nonnegative=key == 'amplitude'))

    def realize(self):
        return WaveRealization((AiryComponent(self.amplitude, self.omega,
            math.degrees(self.heading_radians % math.tau), self.phase),))


@dataclass(frozen=True)
class WaveRealization:
    """Phases are authoritative saved data, never regenerated on load."""
    components: tuple
    source_kind: str = 'declared_components'
    seed: int | None = None
    spectral_bins: SpectralBins | None = None
    directions: DirectionalSpectrum | None = None
    jonswap: JonswapSpectrum | None = None

    def __post_init__(self):
        try:
            components = tuple(self.components)
        except TypeError as error:
            raise LoadError('realization components must be a sequence') from error
        if not components or any(type(c) is not AiryComponent for c in components):
            raise LoadError('realization needs exact named Airy components')
        if self.source_kind == 'declared_components':
            if any(x is not None for x in (self.seed, self.spectral_bins, self.directions, self.jonswap)):
                raise LoadError('declared components cannot claim spectral generation metadata')
        elif self.source_kind == 'seeded_spectral_bins':
            if (type(self.seed) is not int or self.seed < 0 or type(self.spectral_bins) is not SpectralBins
                    or type(self.directions) is not DirectionalSpectrum):
                raise LoadError('seeded realization requires named bins/directions and a nonnegative integer seed')
            bins, d = self.spectral_bins, self.directions
            if len(components) != len(bins.omegas)*len(d.weights):
                raise LoadError('realization component inventory does not match frequency/direction bins')
            if self.jonswap is not None:
                if type(self.jonswap) is not JonswapSpectrum:
                    raise LoadError('named JONSWAP parameters required')
                expected = self.jonswap.discretize(bins.omegas, bins.bandwidths)
                if not np.allclose(bins.density_per_radian, expected.density_per_radian, rtol=1e-12, atol=0.):
                    raise LoadError('saved spectral bins disagree with declared JONSWAP parameters')
            for i, (w, s, dw) in enumerate(zip(bins.omegas, bins.density_per_radian, bins.bandwidths)):
                for j, (heading, weight) in enumerate(zip(d.headings_radians, d.weights)):
                    c = components[i*len(d.weights)+j]
                    expected = math.sqrt(2.*s*dw*weight)
                    if (c.omega != w or c.direction_degrees != math.degrees(heading)
                            or not math.isclose(c.amplitude, expected, rel_tol=1e-12, abs_tol=0.)):
                        raise LoadError('saved components disagree with spectral energy/direction bins')
        else:
            raise LoadError('unsupported realization source kind')
        object.__setattr__(self, 'components', components)

    @classmethod
    def from_spectrum(cls, bins, *, seed, directions=None, jonswap=None):
        if type(bins) is not SpectralBins or type(seed) is not int or seed < 0:
            raise LoadError('named bins and a nonnegative integer seed required')
        d = DirectionalSpectrum() if directions is None else directions
        if type(d) is not DirectionalSpectrum:
            raise LoadError('named directional spectrum required')
        phases = np.random.Generator(np.random.PCG64(seed)).uniform(0., math.tau,
            len(bins.omegas)*len(d.weights))
        components = tuple(AiryComponent(math.sqrt(2.*s*dw*weight), w, math.degrees(heading),
            float(phases[i*len(d.weights)+j]))
            for i,(w,s,dw) in enumerate(zip(bins.omegas,bins.density_per_radian,bins.bandwidths))
            for j,(heading,weight) in enumerate(zip(d.headings_radians,d.weights)))
        return cls(components, 'seeded_spectral_bins', seed, bins, d, jonswap)

    @classmethod
    def from_jonswap(cls, spectrum, omegas, bandwidths, *, seed, directions=None):
        if type(spectrum) is not JonswapSpectrum:
            raise LoadError('named JonswapSpectrum required')
        return cls.from_spectrum(spectrum.discretize(omegas, bandwidths), seed=seed,
                                 directions=directions, jonswap=spectrum)

    def environment(self, water_depth, **settings):
        if 'components' in settings:
            raise LoadError('realization components cannot be replaced')
        return AiryEnvironment(water_depth, self.components, **settings)

    def to_dict(self):
        if type(self) is not WaveRealization:
            raise LoadError('serialization cannot erase subclass semantics')
        return dict(api_version=SEA_STATE_VERSION, phase_algorithm=(PHASE_ALGORITHM
            if self.seed is not None else 'declared_materialized'), **_plain(self))

    @classmethod
    def from_dict(cls, data):
        if type(data) is not dict or set(data) != {'api_version','phase_algorithm',
                'components','source_kind','seed','spectral_bins','directions','jonswap'}:
            raise LoadError('wave realization must match the exact named schema')
        if data['api_version'] != SEA_STATE_VERSION or data['phase_algorithm'] != (
                PHASE_ALGORITHM if data['seed'] is not None else 'declared_materialized'):
            raise LoadError('unsupported wave realization or phase algorithm')
        if not isinstance(data['components'], (list,tuple)):
            raise LoadError('wave components must be a sequence')
        return cls(tuple(_record(c, AiryComponent) for c in data['components']),
            data['source_kind'], data['seed'],
            None if data['spectral_bins'] is None else _record(data['spectral_bins'], SpectralBins),
            None if data['directions'] is None else _record(data['directions'], DirectionalSpectrum),
            None if data['jonswap'] is None else _record(data['jonswap'], JonswapSpectrum))

    @property
    def content_hash(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(',',':'),
            allow_nan=False).encode('utf-8')).hexdigest()
