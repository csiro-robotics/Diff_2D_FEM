"""Implicit backward-Euler PNCG-IPC solver, IPC contact between bodies, the actuation
terms, body-level modes (rigid bodies, rigid-mode coarse space) and the adjoint."""
from .solver import PNCG2DSolver
from .integrator import ImplicitIntegrator
from .adjoint import AdjointPNCG2D
from . import kernels, actuation, affine, ipc_contact, body_contact, adjoint
from ..core import math          # the shared constitutive/barrier helpers live in core/

__all__ = ["PNCG2DSolver", "ImplicitIntegrator", "AdjointPNCG2D", "kernels", "actuation",
           "affine", "ipc_contact", "body_contact", "adjoint", "math"]
