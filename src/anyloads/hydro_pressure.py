"""Explicit same-surface nearest-panel transfer and regular-wave phasing.

This interpolation is not a conservative projection. Resultant errors are
reported against the original source integral without normalization.
"""
from dataclasses import dataclass
import numpy as np


def pressure_wrench(centers, normals, areas, pressure):
    force = -np.asarray(pressure)[..., None] * np.asarray(areas)[..., None] * np.asarray(normals)
    return np.concatenate((force.sum(axis=-2), np.cross(centers, force).sum(axis=-2)), axis=-1)


@dataclass(frozen=True)
class PressureTransfer:
    source_id: str
    element_ids: tuple[int, ...]
    panel_indices: np.ndarray
    phasors: np.ndarray
    source_wrench: np.ndarray
    target_wrench: np.ndarray
    max_distance: float
    omega: float

    def snapshots(self, times, *, amplitude=.5, omega=None):
        if omega is None:
            omega = self.omega
        if omega != self.omega:
            raise ValueError('snapshot frequency differs from source frequency')
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('finite strictly increasing times required')
        if not np.isfinite(amplitude) or amplitude < 0 or not np.isfinite(omega) or omega <= 0:
            raise ValueError('finite nonnegative amplitude and positive omega required')
        return amplitude * np.real(np.exp(-1j*omega*times[:, None])*self.phasors[None, :])


def transfer_panel_pressure(source, element_ids, centers, normals, areas, surface_ids):
    """Map within exact physical surface identity; reject missing/sign mismatches.

    Target cells must lie wholly within caller-certified waterline-conforming
    planar source surfaces. No crossing-waterline cells are admitted here.
    """
    count=len(source.surface_ids)
    if (np.shape(source.centers)!=(count,3) or np.shape(source.normals)!=(count,3)
            or np.shape(source.areas)!=(count,) or np.shape(source.pressure)!=(count,)):
        raise ValueError('source panel payload shapes differ')
    if (not all(np.isfinite(x).all() for x in (source.vertices,source.centers,source.normals,source.areas,source.pressure))
            or np.any(np.asarray(source.areas)<=0)
            or not np.allclose(np.linalg.norm(source.normals,axis=1),1.,rtol=0.,atol=1e-8)):
        raise ValueError('source pressure/geometry must be finite with positive areas and unit normals')
    validator=getattr(source,'validate',None)
    if validator is not None:
        validator()
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in element_ids):
        raise ValueError('integer element identifiers required')
    ids = tuple(int(x) for x in element_ids)
    c, n, a = (np.asarray(x, dtype=float) for x in (centers, normals, areas))
    if c.shape != (len(ids), 3) or n.shape != c.shape or a.shape != (len(ids),) or len(surface_ids) != len(ids):
        raise ValueError('target shape mismatch')
    if not len(ids) or len(set(ids)) != len(ids) or not all(np.isfinite(x).all() for x in (c, n, a)) or np.any(a <= 0):
        raise ValueError('unique cells and finite positive areas required')
    if not np.allclose(np.linalg.norm(n,axis=1),1.,rtol=0.,atol=1e-8):
        raise ValueError('unit outward target normals required')
    indices = []
    distances = []
    for center, normal, label in zip(c, n, surface_ids):
        candidates = np.flatnonzero(np.asarray(source.surface_ids) == label)
        if not len(candidates):
            raise ValueError(f'no source coverage for surface {label}')
        corners = source.vertices[source.faces[candidates].ravel()]
        if np.any(center < corners.min(axis=0)-1e-8) or np.any(center > corners.max(axis=0)+1e-8):
            raise ValueError(f'target outside source planar rectangle: {label}')
        distance = np.linalg.norm(source.centers[candidates]-center, axis=1)
        panel = int(candidates[np.argmin(distance)])
        if np.dot(normal, source.normals[panel]) < 1.-1e-8:
            raise ValueError('target/source outward normal mismatch')
        indices.append(panel)
        distances.append(float(distance.min()))
    for label in set(source.surface_ids) | set(surface_ids):
        sa = source.areas[np.asarray(source.surface_ids) == label].sum()
        ta = a[np.asarray(surface_ids) == label].sum()
        if not np.isclose(sa, ta, rtol=1e-8, atol=1e-8):
            raise ValueError(f'source/target surface area coverage mismatch: {label}: {sa} vs {ta}')
    ix = np.asarray(indices, dtype=int)
    p = np.asarray(source.pressure)[ix].copy()
    sw = pressure_wrench(source.centers, source.normals, source.areas, source.pressure)
    tw = pressure_wrench(c, n, a, p)
    for array in (ix, p, sw, tw):
        array.setflags(write=False)
    return PressureTransfer(source.source_id, ids, ix, p, sw, tw, max(distances), source.omega)
