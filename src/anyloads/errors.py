"""Exceptions raised while defining or evaluating a load."""

from __future__ import annotations

__all__ = ["LoadError"]


class LoadError(ValueError):
    """A load definition is invalid or cannot be evaluated.

    Every error this package raises derives from it, so a consumer can catch
    one type and report the message, which is written for the engineer.
    """
