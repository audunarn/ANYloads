"""Which way fluid pressure pushes on a surface.

A pressure magnitude from a field (hydrostatic, wave, table) presses *into* the
surface from the side the fluid is on. A solver, though, takes a pressure signed
along the surface normal. The conversion depends only on the fluid side, so it
lives here and every consumer applies it identically.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .errors import LoadError

__all__ = ["FLUID_SIDES", "check_fluid_side", "fluid_sign"]

#: ``positive_normal``: the fluid touches the side the normal points to, so the
#: load acts against the normal. ``negative_normal``: the opposite side, so the
#: load acts along it. ``away_from_point``: the fluid is outside a closed volume
#: containing a given interior point (a hull, a tank); the side is decided per
#: point from where the normal points, so inconsistent orientation cannot flip a
#: load.
FLUID_SIDES: tuple[str, ...] = (
    "positive_normal",
    "negative_normal",
    "away_from_point",
)


def check_fluid_side(
    fluid_side: str, interior_point: Sequence[float] | None
) -> np.ndarray | None:
    """Validate a fluid side; returns the interior point as an array if any."""

    if fluid_side not in FLUID_SIDES:
        raise LoadError("fluid side must be one of " + ", ".join(FLUID_SIDES))
    if fluid_side == "away_from_point":
        if interior_point is None:
            raise LoadError("a surface facing away from a point needs that point")
        try:
            point = np.asarray(interior_point, dtype=float)
        except (TypeError, ValueError):
            raise LoadError("the interior point needs three finite components") from None
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise LoadError("the interior point needs three finite components")
        return point.copy()
    if interior_point is not None:
        raise LoadError(
            "an interior point only applies when the fluid side is away_from_point"
        )
    return None


def fluid_sign(
    fluid_side: str,
    interior_point: Sequence[float] | None,
    centroid: np.ndarray,
    normal: np.ndarray,
    *,
    label: str = "surface",
) -> float:
    """Sign turning a pressure magnitude into a normal-signed pressure.

    Positive pressure acts along ``normal``; fluid on the positive-normal side
    pushes against it, so that case is ``-1``.
    """

    if fluid_side == "positive_normal":
        return -1.0
    if fluid_side == "negative_normal":
        return 1.0
    outward = np.asarray(centroid, dtype=float) - np.asarray(interior_point, dtype=float)
    side = float(np.dot(normal, outward))
    if side == 0.0:
        raise LoadError(
            f"{label}: the interior point lies in the plane of an element, so "
            "the fluid side cannot be decided"
        )
    return -1.0 if side > 0.0 else 1.0
