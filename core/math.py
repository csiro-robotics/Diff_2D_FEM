"""2D PNCG-IPC math core — Warp @wp.func primitives.

One P1 triangle is a 6-DOF element stencil (mat22 / vec6).

CONVENTIONS (these are the silent-bug surface):

  * Ds = mat22 with EDGE VECTORS AS COLUMNS:  Ds = [x1-x0 | x2-x0].
    F = Ds @ B with B = D_m_inv = (rest Ds)^-1, so F = I at rest. core/mesh.py builds
    D_m_inv the same way.
  * Node ordering in vec6 is node-major: [n0.x, n0.y, n1.x, n1.y, n2.x, n2.y].
    Node 0 is the apex whose block is minus the sum of the other two.
  * kappa is a PHYSICAL stiffness and BOTH the elastic and barrier terms are
    scaled by h^2 in the incremental potential.

SECOND DERIVATIVES. In 2D the directional form is short enough to use directly:
for Psi = mu/2(I2 - 2) - mu log J + la/2 log^2 J, with G = F^-T,

    dPsi/dF        = mu F + (la logJ - mu) G
    d2Psi/dF2 [dF] = mu dF + la (G:dF) G - (la logJ - mu) G dF^T G

so p^T H p is one directional derivative and diag(H) is six (one per element
DOF). Exact, no 6x6 assembly, and no invariant decomposition to get wrong.
"""
from __future__ import annotations
import warp as wp
from .precision import real, vec2, vec3, mat22, np_real

vec6 = wp.types.vector(6, real)

LOG_EPS = wp.constant(real(1.0e-6))


# ---------------------------------------------------------------- elastic
@wp.func
def ds_from_nodes(p0: vec2, p1: vec2, p2: vec2) -> mat22:
    """Edge vectors as COLUMNS."""
    return mat22(p1[0] - p0[0], p2[0] - p0[0],
                    p1[1] - p0[1], p2[1] - p0[1])


@wp.func
def dF_from_dir(n: int, c: int, B: mat22) -> mat22:
    """dF produced by a unit perturbation of node `n` component `c`.

    F = Ds B is linear in x, so dF = dDs B where dDs has the basis pattern:
      n==0 -> both edges shrink   -> dDs = [-e_c | -e_c]
      n==1 -> first edge only     -> dDs = [ e_c |   0 ]
      n==2 -> second edge only    -> dDs = [  0  |  e_c]
    """
    a = real(0.0)
    b = real(0.0)
    if n == 0:
        a = -real(1.0)
        b = -real(1.0)
    elif n == 1:
        a = real(1.0)
    else:
        b = real(1.0)
    dDs = mat22(real(0.0), real(0.0), real(0.0), real(0.0))
    if c == 0:
        dDs = mat22(a, b, real(0.0), real(0.0))
    else:
        dDs = mat22(real(0.0), real(0.0), a, b)
    return dDs * B


@wp.func
def dF_from_vec6(p: vec6, B: mat22) -> mat22:
    """dF for an arbitrary element displacement vector (node-major vec6)."""
    dDs = mat22(p[2] - p[0], p[4] - p[0],
                   p[3] - p[1], p[5] - p[1])
    return dDs * B


@wp.func
def nh_dPsidF(F: mat22, mu: real, la: real) -> mat22:
    """First Piola: mu F + (la logJ - mu) F^-T."""
    J = wp.determinant(F)
    logJ = wp.log(wp.max(J, LOG_EPS))
    FinvT = wp.transpose(wp.inverse(F))
    return mu * F + (la * logJ - mu) * FinvT


@wp.func
def nh_d2PsidF2_dir(F: mat22, dF: mat22, mu: real, la: real) -> mat22:
    """Directional second derivative d2Psi/dF2 [dF] (see module docstring)."""
    J = wp.determinant(F)
    logJ = wp.log(wp.max(J, LOG_EPS))
    Finv = wp.inverse(F)
    G = wp.transpose(Finv)
    GdF = wp.ddot(G, dF)                       # G : dF
    GdFtG = G * wp.transpose(dF) * G
    return mu * dF + la * GdF * G - (la * logJ - mu) * GdFtG


@wp.func
def nh_pHp_elem(F: mat22, B: mat22, p: vec6, mu: real, la: real) -> real:
    """p^T (d2Psi/dx2) p for one element, per unit rest area."""
    dF = dF_from_vec6(p, B)
    return wp.ddot(dF, nh_d2PsidF2_dir(F, dF, mu, la))


# ---------------------------------------------------------------- IPC barrier (point vs circle)
@wp.func
def cross2(a: vec2, b: vec2) -> real:
    """Scalar 2D cross product (wp.cross has no vec2 overload). Twice the signed
    area of the triangle spanned by a and b."""
    return a[0] * b[1] - a[1] * b[0]


# ---------------------------------------------------------------- IPC barrier (point vs circle)
@wp.func
def barrier_b(d: real, dhat: real) -> real:
    u = d - dhat
    return -(u * u) * wp.log(d / dhat)


@wp.func
def barrier_bp(d: real, dhat: real) -> real:
    """db/dd."""
    u = d - dhat
    return -real(2.0) * u * wp.log(d / dhat) - (u * u) / d


@wp.func
def barrier_bpp(d: real, dhat: real) -> real:
    """d2b/dd2."""
    u = d - dhat
    return -real(2.0) * wp.log(d / dhat) - real(4.0) * u / d + (u * u) / (d * d)


@wp.func
def pair_mu(mu: real, per_body: int, mu_node: wp.array(dtype=real), i: int, j: int) -> real:
    """Coulomb coefficient of a contact between the bodies of nodes i and j: `mu` (the
    model's body_friction) unless per_body, then the harmonic mean of the two bodies' own,
    2 m_i m_j / (m_i + m_j), which is Taccel's rule — 0 if either is 0."""
    if per_body == 0:
        return mu
    mi = mu_node[i]
    mj = mu_node[j]
    sm = mi + mj
    if sm > real(0.0):
        return real(2.0) * mi * mj / sm
    return real(0.0)
