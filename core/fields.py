"""Strain and stress per element, from node positions (numpy) — for pictures and read-outs.

P1 triangles: F is constant over each element, so every field here is one value (or one
tensor) per triangle, in element order (model.body_elements). Plane strain: the
out-of-plane stretch is 1, so E_zz = 0 while sigma_zz is not.

    F      = Ds Dm^-1                                  (core/math.py conventions)
    E      = (F^T F - I) / 2                           Green-Lagrange strain, E_zz = 0
    sigma  = (mu F F^T + (lam ln J - mu) I) / J        Cauchy stress of the solver's own
    s_zz   = lam ln J / J                              Neo-Hookean, Psi = mu/2 (I2 - 2)
                                                       - mu ln J + lam/2 ln^2 J
    von Mises:  strain sqrt(2/3 e':e'),  stress sqrt(3/2 s':s')   (' : deviatoric, 3 x 3)
    signed:     the in-plane principal value of larger magnitude, with its sign —
                > 0 tension, < 0 compression (along a bent beam: one face +, the other -)

The von Mises strain of a bar stretched uniaxially (free to contract) is about its axial
strain. Plane strain holds the out-of-plane stretch at 1, so sigma_zz carries load even
under a pure in-plane pressure, and the von Mises stress is not 0 there.
"""
from __future__ import annotations
import numpy as np


def deformation_gradient(me, x):
    """F per element, (n_elements, 2, 2), at node positions x ((N, 2))."""
    p = np.asarray(x, np.float64)[np.asarray(me.tri)]
    Ds = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]], -1)           # edge vectors as columns
    return Ds @ np.asarray(me.D_m_inv, np.float64)


def _in_3d(T2, zz):
    T = np.zeros((len(T2), 3, 3))
    T[:, :2, :2] = T2
    T[:, 2, 2] = zz
    return T


def green_lagrange(F):
    """Green-Lagrange strain per element as a 3 x 3 tensor (E_zz = 0)."""
    return _in_3d(0.5 * (F.transpose(0, 2, 1) @ F - np.eye(2)), 0.0)


def cauchy_stress(F, mu, lam):
    """Cauchy stress per element as a 3 x 3 tensor [Pa], for the solver's Neo-Hookean."""
    J = np.linalg.det(F)
    lnJ = np.log(np.maximum(J, 1e-12))
    mu, lam = np.asarray(mu, np.float64), np.asarray(lam, np.float64)
    s2 = (mu[:, None, None] * (F @ F.transpose(0, 2, 1))
          + ((lam * lnJ - mu)[:, None, None]) * np.eye(2)) / J[:, None, None]
    return _in_3d(s2, lam * lnJ / J)


def _deviatoric_norm2(T):
    dev = T - np.trace(T, axis1=1, axis2=2)[:, None, None] / 3.0 * np.eye(3)
    return np.einsum("nij,nij->n", dev, dev)


def von_mises_strain(E):
    return np.sqrt(2.0 / 3.0 * _deviatoric_norm2(E))


def von_mises_stress(S):
    return np.sqrt(1.5 * _deviatoric_norm2(S))


def principal_2d(T):
    """In-plane principal values of each tensor (its xy block): (larger, smaller)."""
    a, b, d = T[:, 0, 0], T[:, 0, 1], T[:, 1, 1]
    m, r = 0.5 * (a + d), np.sqrt(0.25 * (a - d) ** 2 + b * b)
    return m + r, m - r


def signed_principal(T):
    """The in-plane principal value of larger magnitude, with its sign: > 0 where the
    material is mostly pulled (tension), < 0 where it is mostly squeezed (compression)."""
    hi, lo = principal_2d(T)
    return np.where(np.abs(hi) >= np.abs(lo), hi, lo)


def element_field(me, x, field, signed=False):
    """'strain' (Green-Lagrange) or 'stress' (Cauchy, Pa) of every element at positions x,
    with the mesh's own moduli (me.mu, me.lam): its von Mises equivalent (>= 0), or with
    signed=True its signed principal value (tension +, compression -)."""
    F = deformation_gradient(me, x)
    if field == "strain":
        T = green_lagrange(F)
        return signed_principal(T) if signed else von_mises_strain(T)
    if field == "stress":
        T = cauchy_stress(F, me.mu, me.lam)
        return signed_principal(T) if signed else von_mises_stress(T)
    raise ValueError(f"field must be 'strain' or 'stress', not {field!r}")


LABELS = {"strain": "von Mises strain", "stress": "von Mises stress [Pa]",
          ("strain", True): "principal strain (+ tension, - compression)",
          ("stress", True): "principal stress [Pa] (+ tension, - compression)"}


def label(field, signed=False):
    return LABELS[(field, True)] if signed else LABELS[field]
