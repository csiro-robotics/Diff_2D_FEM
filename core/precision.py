"""Floating-point precision of the simulation — chosen once, before anything is imported.

    DIFF2DFEM_PRECISION=double   (default)   float64 state and kernels
    DIFF2DFEM_PRECISION=single               float32: ~2x the memory bandwidth, for very large
                                             batches where approximate equilibria are acceptable

WHY DOUBLE IS THE DEFAULT. The implicit solver is an iterative (conjugate-gradient) method. In
single precision the elastic forces are differences of nearly equal numbers computed from
absolute positions, so the gradient carries rounding noise of about 6e-8 * E * |x| per node. For
a soft mode — a slender body bending, a body with nothing holding it — the true residual falls
below that noise long before the solve has converged, and the iteration wanders instead of
converging. Measured on a tip-loaded cantilever, the same algorithm run both ways:

    mesh        single precision, 30 000 iterations     double precision, 2 000 iterations
    8 x 40              -1.5 %                                   0.00 %
    8 x 80             -94 %                                     0.00 %
    16 x 160           -99.4 %                                  +0.04 %

Every kernel takes its types from here (`real`, `vec2`, `vec3`, `mat22`) and types its own
literals, so the whole package switches together. The accumulators the solver reduces per
environment are float64 in both modes.
"""
from __future__ import annotations
import os
import numpy as np
import warp as wp

PRECISION = os.environ.get("DIFF2DFEM_PRECISION", "double").strip().lower()
if PRECISION in ("double", "float64", "64", "d"):
    PRECISION = "double"
    real, vec2, vec3, mat22, np_real = wp.float64, wp.vec2d, wp.vec3d, wp.mat22d, np.float64
elif PRECISION in ("single", "float32", "32", "f"):
    PRECISION = "single"
    real, vec2, vec3, mat22, np_real = wp.float32, wp.vec2f, wp.vec3f, wp.mat22f, np.float32
else:
    raise ValueError(f"DIFF2DFEM_PRECISION must be 'double' or 'single', not {PRECISION!r}")

__all__ = ["PRECISION", "real", "vec2", "vec3", "mat22", "np_real"]
