"""Data-only linear marine environment and prescribed circular-member loads.

SI units, WORLD axes with z up. Linear waves occupy bed <= z <= still water
level, without stretching or current-modified dispersion. Fluid acceleration
is fixed-spatial partial_t, not material or member-path acceleration. Inputs
are immutable records; deserialization never imports or executes user code.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from scipy.optimize import brentq
from .errors import LoadError

MARINE_API_VERSION = 'marine_loads_v1'


def _real(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise LoadError(f'{name} must be a finite real number')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise LoadError(f'{name} must be a finite real number') from None
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise LoadError(f'{name} is outside its finite admissible range')
    return value


def _vector(value, name):
    try:
        raw = tuple(value)
    except TypeError:
        raise LoadError(f'{name} needs three finite real components') from None
    if len(raw) != 3:
        raise LoadError(f'{name} needs three finite real components')
    return tuple(_real(v, name) for v in raw)


@dataclass(frozen=True)
class AiryComponent:
    """Elevation a*cos(k*d.xy - omega*t + phase); direction is propagation."""
    amplitude: float
    omega: float
    direction_degrees: float = 0.
    phase: float = 0.

    def __post_init__(self):
        for name in ('amplitude', 'omega', 'direction_degrees', 'phase'):
            object.__setattr__(self, name, _real(getattr(self, name), name,
                positive=name == 'omega', nonnegative=name == 'amplitude'))


@dataclass(frozen=True)
class CurrentProfile:
    """Steady WORLD velocities at increasing depths below still water level.

    Linear interpolation; at an internal knot use the deeper interval's slope.
    Environment admission requires exact coverage of the full water column.
    """
    depths: tuple
    velocities: tuple

    def __post_init__(self):
        depths = tuple(_real(v, 'current depth', nonnegative=True) for v in self.depths)
        velocities = tuple(_vector(v, 'current velocity') for v in self.velocities)
        if (len(depths) < 2 or len(depths) != len(velocities) or depths[0] != 0.
                or any(b <= a for a, b in zip(depths, depths[1:]))):
            raise LoadError('current profile needs increasing depths starting at zero and matching velocities')
        object.__setattr__(self, 'depths', depths)
        object.__setattr__(self, 'velocities', velocities)

    @classmethod
    def uniform(cls, water_depth, velocity=(0., 0., 0.)):
        return cls((0., _real(water_depth, 'water depth', positive=True)), (velocity, velocity))


@dataclass(frozen=True)
class FluidKinematics:
    velocity: np.ndarray
    acceleration: np.ndarray
    velocity_gradient: np.ndarray
    acceleration_gradient: np.ndarray
    wave_velocity: np.ndarray
    current_velocity: np.ndarray


@dataclass(frozen=True)
class AiryEnvironment:
    water_depth: float
    components: tuple = ()
    current: CurrentProfile | None = None
    density: float = 1025.
    gravity: float = 9.81
    still_water_level: float = 0.
    acceleration_convention: str = 'eulerian_local'
    wave_numbers: tuple = field(init=False, repr=False)

    def __post_init__(self):
        for name in ('water_depth', 'density', 'gravity', 'still_water_level'):
            object.__setattr__(self, name, _real(getattr(self, name), name,
                positive=name != 'still_water_level'))
        components = tuple(self.components)
        if any(type(c) is not AiryComponent for c in components):
            raise LoadError('waves require exact named AiryComponent records')
        current = CurrentProfile.uniform(self.water_depth) if self.current is None else self.current
        if type(current) is not CurrentProfile or current.depths[-1] != self.water_depth:
            raise LoadError('current profile must cover exactly the admitted water depth')
        if self.acceleration_convention != 'eulerian_local':
            raise LoadError('only eulerian_local acceleration is admitted')
        ks = []
        for c in components:
            # Dimensionless dispersion y*tanh(y)=omega^2*h/g, with a guaranteed
            # bracket. Stable profiles below avoid overflowing cosh/sinh.
            try:
                target = c.omega**2*self.water_depth/self.gravity
            except OverflowError:
                raise LoadError('wave dispersion exceeds the finite supported range') from None
            if not math.isfinite(target) or target <= 0:
                raise LoadError('wave dispersion exceeds the finite supported range')
            upper = max(1., 2.*target, 2.*math.sqrt(target))
            y = brentq(lambda x: x*math.tanh(x)-target, 0., upper,
                       xtol=5e-324, rtol=1e-14)
            ks.append(y/self.water_depth)
        object.__setattr__(self, 'components', components)
        object.__setattr__(self, 'current', current)
        object.__setattr__(self, 'wave_numbers', tuple(ks))

    @classmethod
    def seeded_spectrum(cls, water_depth, omegas, density_per_radian, bandwidths,
                        *, seed, direction_degrees=0., **settings):
        """Explicit finite-bin realization: a_i=sqrt(2*S_i*delta_omega_i).

        PCG64 phases are immediately materialized into authoritative components;
        persistence carries the actual phases, not a request to regenerate them.
        Caller supplies the spectral model/bin widths; no hidden JONSWAP defaults.
        """
        if type(seed) is not int or seed < 0:
            raise LoadError('wave seed must be a nonnegative integer')
        frequencies = tuple(_real(w, 'spectrum frequency', positive=True) for w in omegas)
        densities = tuple(_real(s, 'spectral density', nonnegative=True) for s in density_per_radian)
        widths = tuple(_real(dw, 'spectral bandwidth', positive=True) for dw in bandwidths)
        if (not frequencies or len(frequencies) != len(densities) or len(widths) != len(frequencies)
                or any(b <= a for a, b in zip(frequencies, frequencies[1:]))):
            raise LoadError('spectrum needs increasing frequencies and matching density/bin widths')
        phases = np.random.Generator(np.random.PCG64(seed)).uniform(0., 2.*math.pi, len(frequencies))
        components = tuple(AiryComponent(math.sqrt(2.*s*dw), w, direction_degrees, phase)
            for w, s, dw, phase in zip(frequencies, densities, widths, phases))
        return cls(water_depth, components, **settings)

    @property
    def wet_levels(self):
        return tuple(self.still_water_level-d for d in self.current.depths)

    def surface_elevation(self, points_xy, time):
        """Prescribed incident elevation in WORLD z, with saved Airy phases.

        Coordinates are finite xy batches. Current is superposed for drag only;
        it does not change this still-water wave dispersion or phase evolution.
        """
        time = _real(time, 'time')
        try:
            raw = np.asarray(points_xy,dtype=object)
            if any(isinstance(x,(bool,np.bool_)) for x in raw.flat):
                raise ValueError()
            points = np.asarray(points_xy)
            if (points.ndim < 1 or points.shape[-1] != 2 or points.dtype.kind not in 'iuf'
                    or not np.isfinite(points).all()):
                raise ValueError()
        except (TypeError,ValueError,OverflowError) as error:
            raise LoadError('surface points need finite real WORLD xy') from error
        eta = np.full(points.shape[:-1],self.still_water_level,dtype=float)
        try:
            with np.errstate(over='raise',invalid='raise'):
                for c,k in zip(self.components,self.wave_numbers):
                    direction=np.array((math.cos(math.radians(c.direction_degrees)),
                                        math.sin(math.radians(c.direction_degrees))))
                    eta += c.amplitude*np.cos(k*(points@direction)-c.omega*time+c.phase)
        except FloatingPointError as error:
            raise LoadError('surface phase/elevation exceeds finite supported range') from error
        if not np.isfinite(eta).all():
            raise LoadError('nonfinite surface elevation')
        return eta

    def evaluate(self, points, time):
        """Batch WORLD u, partial_t u and spatial gradients (rows vector/cols xyz)."""
        time = _real(time, 'time')
        try:
            x = np.asarray(points, dtype=float)
        except (ValueError, TypeError):
            raise LoadError('fluid points need finite WORLD xyz') from None
        if x.ndim < 1 or x.shape[-1] != 3 or not np.isfinite(x).all():
            raise LoadError('fluid points need finite WORLD xyz')
        z = x[..., 2]-self.still_water_level
        if np.any(z > 0) or np.any(z < -self.water_depth):
            raise LoadError('Airy kinematics are restricted to bed through still water level')
        u, acc = np.zeros_like(x), np.zeros_like(x)
        gu, ga = np.zeros(x.shape[:-1]+(3, 3)), np.zeros(x.shape[:-1]+(3, 3))
        for c, k in zip(self.components, self.wave_numbers):
            direction = np.array((math.cos(math.radians(c.direction_degrees)),
                                  math.sin(math.radians(c.direction_degrees)), 0.))
            phase = k*(x@direction)-c.omega*time+c.phase
            cos, sin = np.cos(phase), np.sin(phase)
            denominator = -math.expm1(-2.*k*self.water_depth)
            top, reflected = np.exp(k*z), np.exp(-k*(z+2.*self.water_depth))
            ch, sh = (top+reflected)/denominator, (top-reflected)/denominator
            horizontal = c.amplitude*c.omega*ch
            vertical = c.amplitude*c.omega*sh
            u += (horizontal*cos)[..., None]*direction
            u[..., 2] += vertical*sin
            acc += (c.omega*horizontal*sin)[..., None]*direction
            acc[..., 2] -= c.omega*vertical*cos
            for j in range(3):
                gu[..., :, j] += (-k*direction[j]*horizontal*sin)[..., None]*direction
                gu[..., 2, j] += k*direction[j]*vertical*cos
                ga[..., :, j] += (c.omega*k*direction[j]*horizontal*cos)[..., None]*direction
                ga[..., 2, j] += c.omega*k*direction[j]*vertical*sin
            gu[..., :, 2] += (k*vertical*cos)[..., None]*direction
            gu[..., 2, 2] += k*horizontal*sin
            ga[..., :, 2] += (c.omega*k*vertical*sin)[..., None]*direction
            ga[..., 2, 2] -= c.omega*k*horizontal*cos
        depth = -z
        i = np.clip(np.searchsorted(self.current.depths, depth, side='right')-1,
                    0, len(self.current.depths)-2)
        depths = np.asarray(self.current.depths)
        velocities = np.asarray(self.current.velocities)
        slope = (velocities[i+1]-velocities[i])/(depths[i+1]-depths[i])[..., None]
        current = velocities[i]+(depth-depths[i])[..., None]*slope
        gu[..., :, 2] -= slope
        combined = u+current
        if not all(np.isfinite(v).all() for v in (combined, acc, gu, ga)):
            raise LoadError('nonfinite environment kinematics refused')
        return FluidKinematics(combined, acc, gu, ga, u, current)


@dataclass(frozen=True)
class HydrodynamicOwnership:
    primary_role: str = 'morison_full'
    bem_included: bool = False
    drag_correction: bool = False
    buoyancy_owner: str = 'none'

    def __post_init__(self):
        if self.primary_role not in ('potential_flow', 'morison_full', 'none'):
            raise LoadError('unknown primary hydrodynamic role')
        if type(self.bem_included) is not bool or type(self.drag_correction) is not bool:
            raise LoadError('BEM inclusion and drag correction must be explicit booleans')
        if self.primary_role == 'morison_full' and self.bem_included:
            raise LoadError('duplicate_inertia_bem_plus_morison_full')
        if self.primary_role == 'morison_full' and self.drag_correction:
            raise LoadError('duplicate_drag_full_morison_plus_correction')
        if (self.primary_role == 'potential_flow') != self.bem_included:
            raise LoadError('potential-flow role and component BEM inclusion must agree')
        if self.buoyancy_owner not in ('none', 'potential_flow', 'morison'):
            raise LoadError('unknown buoyancy owner')
        if (self.buoyancy_owner == 'morison' and self.bem_included
                or self.buoyancy_owner == 'potential_flow' and not self.bem_included):
            raise LoadError('duplicate_or_incompatible_buoyancy_owner')


@dataclass(frozen=True)
class MorisonSection:
    diameter: float
    drag_coefficient: float = 1.
    added_mass_coefficient: float = 1.

    def __post_init__(self):
        for name in ('diameter', 'drag_coefficient', 'added_mass_coefficient'):
            object.__setattr__(self, name, _real(getattr(self, name), name,
                positive=name == 'diameter', nonnegative=name != 'diameter'))

    @property
    def area(self):
        return math.pi*self.diameter**2/4.


@dataclass(frozen=True)
class MorisonForce:
    force: np.ndarray
    fluid_velocity_derivative: np.ndarray
    fluid_acceleration_derivative: np.ndarray
    structure_velocity_derivative: np.ndarray
    structure_acceleration_derivative: np.ndarray
    axis_derivative: np.ndarray
    components: dict
    drag_attribution: dict
    relative_drag_power: np.ndarray


def morison_force(section, ownership, density, fluid_velocity, fluid_acceleration,
                  structure_velocity, structure_acceleration, axis, *, gravity=9.81,
                  wave_velocity=None, current_velocity=None):
    """Batch force per current metre and analytic ambient vector partials.

    The axis must be unit; its derivative is for tangent variations on the unit
    sphere. Added inertia appears only as -rho*CA*A*P*a. Axis rotation does not
    introduce variational kinetic-energy forces. Drag attribution convention:
    wave+structure-motion drag, current-alone drag, remaining interaction;
    these explain the ONE combined drag and are never additional loads.
    Buoyancy, if Morison-owned, uses nominal full section per wet centreline metre.
    """
    if type(section) is not MorisonSection or type(ownership) is not HydrodynamicOwnership:
        raise LoadError('Morison needs exact named section and ownership records')
    rho, g = _real(density, 'density', positive=True), _real(gravity, 'gravity', positive=True)
    try:
        arrays = [np.asarray(v, dtype=float) for v in
                  (fluid_velocity, fluid_acceleration, structure_velocity, structure_acceleration, axis)]
        if any(v.ndim < 1 or v.shape[-1] != 3 or not np.isfinite(v).all() for v in arrays):
            raise LoadError('Morison kinematics require finite vector batches ending in xyz')
        u, af, v, a, t = np.broadcast_arrays(*arrays)
    except (TypeError, ValueError) as error:
        raise LoadError(f'invalid Morison vector batches: {error}') from error
    if not np.allclose(np.linalg.norm(t, axis=-1), 1., rtol=0., atol=1e-12):
        raise LoadError('member axis must be unit')
    eye = np.broadcast_to(np.eye(3), t.shape[:-1]+(3, 3))
    outer = lambda a, b: a[..., :, None]*b[..., None, :]
    apply = lambda m, v: np.einsum('...ij,...j->...i', m, v)
    P = eye-outer(t, t)
    full = ownership.primary_role == 'morison_full'
    mf = rho*section.area if full else 0.
    ma = mf*section.added_mass_coefficient
    b = .5*rho*section.drag_coefficient*section.diameter if full or ownership.drag_correction else 0.
    rel = u-v
    w = apply(P, rel)
    speed = np.linalg.norm(w, axis=-1)
    inverse = np.divide(1., speed, out=np.zeros_like(speed), where=speed > 0)
    drag_gradient = b*(speed[..., None, None]*eye+inverse[..., None, None]*outer(w, w))
    drag = b*speed[..., None]*w
    fk, added = mf*apply(P, af), ma*apply(P, af-a)
    buoyancy = np.zeros_like(u)
    if ownership.buoyancy_owner == 'morison':
        buoyancy[..., 2] = rho*g*section.area
    axis_map = lambda vector: -(np.sum(t*vector, axis=-1))[..., None, None]*eye-outer(t, vector)
    axis_gradient = axis_map(mf*af+ma*(af-a))+drag_gradient@axis_map(rel)
    if (wave_velocity is None) != (current_velocity is None):
        raise LoadError('drag attribution requires both wave and current velocities')
    if wave_velocity is None:
        wave, current = u, np.zeros_like(u)
    else:
        try:
            wave, current = (np.broadcast_to(np.asarray(value, dtype=float), u.shape)
                             for value in (wave_velocity, current_velocity))
        except (TypeError, ValueError):
            raise LoadError('wave/current attribution shapes must match fluid velocity') from None
        if (not np.isfinite(wave).all() or not np.isfinite(current).all()
                or not np.allclose(wave+current, u, rtol=1e-12, atol=1e-12)):
            raise LoadError('wave/current attribution must sum to actual fluid velocity')
    drag_of = lambda velocity: b*np.linalg.norm(apply(P, velocity), axis=-1)[..., None]*apply(P, velocity)
    wave_drag, current_drag = drag_of(wave-v), drag_of(current)
    components = dict(froude_krylov=fk, added_mass=added, drag=drag, buoyancy=buoyancy)
    force = fk+added+drag+buoyancy
    if not all(np.isfinite(value).all() for value in (force, axis_gradient, drag_gradient)):
        raise LoadError('nonfinite Morison force/derivative refused')
    du = drag_gradient@P
    return MorisonForce(force, du, (mf+ma)*P, -du, -ma*P, axis_gradient,
        components, dict(wave_and_motion=wave_drag, current_alone=current_drag,
                         interaction=drag-wave_drag-current_drag),
        np.sum(drag*(v-u), axis=-1))


def _named(record, kind):
    if type(record) is not kind:
        raise LoadError('serialization cannot erase named marine record types')
    return {name: getattr(record, name) for name, f in record.__dataclass_fields__.items() if f.init}


def environment_to_dict(environment):
    data = _named(environment, AiryEnvironment)
    data['api_version'] = MARINE_API_VERSION
    data['components'] = [_named(c, AiryComponent) for c in environment.components]
    data['current'] = _named(environment.current, CurrentProfile)
    return data


def _fields(data, kind, *, version=False):
    expected = {name for name, f in kind.__dataclass_fields__.items() if f.init}
    if version:
        expected.add('api_version')
    if (type(data) is not dict or set(data) != expected
            or version and data['api_version'] != MARINE_API_VERSION):
        raise LoadError(f'{kind.__name__} fields/version must exactly match the named schema')
    return {name: data[name] for name in expected if name != 'api_version'}


def environment_from_dict(data):
    values = _fields(data, AiryEnvironment, version=True)
    try:
        values['components'] = tuple(AiryComponent(**_fields(c, AiryComponent)) for c in values['components'])
        values['current'] = CurrentProfile(**_fields(values['current'], CurrentProfile))
        return AiryEnvironment(**values)
    except (TypeError, ValueError, OverflowError) as error:
        raise LoadError(f'invalid marine environment: {error}') from error


def section_to_dict(section):
    return _named(section, MorisonSection)


def section_from_dict(data):
    return MorisonSection(**_fields(data, MorisonSection))


def ownership_to_dict(ownership):
    return _named(ownership, HydrodynamicOwnership)


def ownership_from_dict(data):
    return HydrodynamicOwnership(**_fields(data, HydrodynamicOwnership))
