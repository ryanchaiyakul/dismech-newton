"""Analytical residuals implementing upstream Newton model features."""

from .dahl import Dahl, dahl_advance

__all__ = ["Dahl", "dahl_advance"]
