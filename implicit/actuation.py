"""Actuation energies for the implicit PNCG-IPC solve.

Every actuator is one more term in the incremental potential the solver minimises,

    E(x) = 1/2 (x - xhat)^T M (x - xhat) + h^2 [ Psi(x) + contact(x) + ACTUATION(x; u) ]

driven by a control signal u that the caller sets each step. A term supplies the same
three things every built-in term does, and nothing else in the solver has to know it
exists:

    grad_diag(s)    add dE/dx to s.grad and a non-negative diagonal estimate to s.diagH
    pHp(s)          add p^T (d2E/dx2) p to s.pHp, per environment
    energy(s, buf)  add E to buf, per environment            (line search, diagnostics)

All of it is PER ENVIRONMENT, like the rest of the solver: a chamber or a tendon belongs
to one environment and only ever touches that environment's reductions.

2D CONVENTION. The model is plane strain, per unit depth. A pressure is in Pa as usual;
forces and tendon tensions are per metre of out-of-plane depth [N/m]. For a part of
thickness w, a physical force F corresponds to F / w here.

    ExternalForce   nodal force field (gravity, point loads)         E = -f . x
    Pressure        fluid pressure in an enclosed chamber            E = -p A(x)
    Tendon          cable through a sequence of eyelets              E = T L(x)  or  k/2 (L - L0)+^2
    ActiveStrain    prescribed rest-shape change of a set of elements   Psi(F A^-1)
    Springs         linear springs and pins between nodes / the world   E = k/2 (|d| - L0)^2
"""
from __future__ import annotations
import numpy as np
import warp as wp
from ..core.precision import real, vec2, vec3, mat22, np_real

f64 = wp.float64


# ====================================================================== external force
@wp.kernel
def grad_extforce_kernel(f: wp.array(dtype=vec2), h2: real, grad: wp.array(dtype=vec2)):
    i = wp.tid()
    grad[i] = grad[i] - h2 * f[i]


@wp.kernel
def energy_extforce_kernel(x: wp.array(dtype=vec2), xn: wp.array(dtype=vec2),
                           f: wp.array(dtype=vec2), node2env: wp.array(dtype=wp.int32),
                           h2: real, E: wp.array(dtype=f64)):
    """-h2 f.(x - xn). Measured from the start-of-step position xn: a constant offset within
    a step (which is all the line search compares across), and it keeps the product small."""
    i = wp.tid()
    wp.atomic_add(E, node2env[i], f64(-h2 * wp.dot(f[i], x[i] - xn[i])))


class ExternalForce:
    """A nodal force field f (N, 2) [N/m]: gravity (f = m g), point loads, anything that
    does not depend on the configuration. Linear in x, so it has no curvature."""
    in_adjoint = True         # no curvature -> the adjoint operator is unaffected

    def __init__(self, s):
        self.f = wp.zeros(s.N, dtype=vec2, device=s.dev)

    def set(self, f):
        self.f.assign(np.ascontiguousarray(f, np_real))

    def grad_diag(self, s):
        wp.launch(grad_extforce_kernel, dim=s.N, inputs=[self.f, s.h2, s.grad], device=s.dev)

    def pHp(self, s):
        pass

    def energy(self, s, buf):
        wp.launch(energy_extforce_kernel, dim=s.N,
                  inputs=[s.x, s.xn, self.f, s.node2env, s.h2, buf], device=s.dev)


# ====================================================================== pressure chamber
# A chamber is a closed loop of mesh boundary edges, each stored DIRECTED so that the fluid
# is on its left (counter-clockwise around the cavity). Its area is the shoelace sum
#     A = 1/2 sum_edges cross(x_a, x_b)
# and fluid at pressure p does work p dA, so the potential is -p A. The force on a wall
# node, +p dA/dx, is the familiar "pressure x edge length x outward normal", shared equally
# between the edge's two ends -- and because it is the gradient of a potential that is
# evaluated on the CURRENT geometry, it is a true follower load.
#
# A is bilinear in pairs of DIFFERENT nodes, so its Hessian has no nodal diagonal (nothing
# to add to the Jacobi preconditioner) and is indefinite: p^T H p = -p sum cross(p_a, p_b).
# That term is kept, exactly, so the step length stays consistent with the gradient. It is
# small next to the elastic curvature whenever p << E, which is every pneumatic actuator
# short of a ballooning instability -- and that instability is real, not numerical.
@wp.kernel
def grad_pressure_kernel(x: wp.array(dtype=vec2), ea: wp.array(dtype=wp.int32),
                         eb: wp.array(dtype=wp.int32), ech: wp.array(dtype=wp.int32),
                         p: wp.array(dtype=real), h2: real, grad: wp.array(dtype=vec2)):
    e = wp.tid()
    a = ea[e]; b = eb[e]
    t = x[b] - x[a]
    n = vec2(t[1], -t[0])                     # (length x normal) pointing from fluid into solid
    g = (-real(0.5) * h2 * p[ech[e]]) * n              # dE/dx, E = -h2 p A
    wp.atomic_add(grad, a, g)
    wp.atomic_add(grad, b, g)


@wp.kernel
def pHp_pressure_kernel(pdir: wp.array(dtype=vec2), ea: wp.array(dtype=wp.int32),
                        eb: wp.array(dtype=wp.int32), ech: wp.array(dtype=wp.int32),
                        ch2env: wp.array(dtype=wp.int32), p: wp.array(dtype=real),
                        h2: real, pHp: wp.array(dtype=f64)):
    e = wp.tid()
    c = ech[e]
    pa = pdir[ea[e]]; pb = pdir[eb[e]]
    wp.atomic_add(pHp, ch2env[c], f64(-h2 * p[c] * (pa[0] * pb[1] - pa[1] * pb[0])))


@wp.kernel
def energy_pressure_kernel(x: wp.array(dtype=vec2), ea: wp.array(dtype=wp.int32),
                           eb: wp.array(dtype=wp.int32), ech: wp.array(dtype=wp.int32),
                           ch2env: wp.array(dtype=wp.int32), cref: wp.array(dtype=vec2),
                           p: wp.array(dtype=real), h2: real, E: wp.array(dtype=f64)):
    """-h2 p A, with the shoelace sum taken about a point inside the chamber so the cross
    products are of millimetre-sized vectors rather than of absolute positions."""
    e = wp.tid()
    c = ech[e]
    ra = x[ea[e]] - cref[c]; rb = x[eb[e]] - cref[c]
    wp.atomic_add(E, ch2env[c], f64(-h2 * p[c] * real(0.5) * (ra[0] * rb[1] - ra[1] * rb[0])))


class Pressure:
    in_adjoint = False        # its tangent is not in implicit/adjoint.py yet

    def __init__(self, s, ea, eb, ech, ch2env, cref):
        ai = lambda v: wp.array(np.asarray(v, np.int32), dtype=wp.int32, device=s.dev)
        self.ea, self.eb, self.ech, self.ch2env = ai(ea), ai(eb), ai(ech), ai(ch2env)
        self.cref = wp.array(np.asarray(cref, np_real), dtype=vec2, device=s.dev)
        self.n_edges, self.n_ch = len(ea), len(ch2env)
        self.p = wp.zeros(self.n_ch, dtype=real, device=s.dev)

    def set(self, p):
        self.p.assign(np.ascontiguousarray(p, np_real))

    def grad_diag(self, s):
        wp.launch(grad_pressure_kernel, dim=self.n_edges,
                  inputs=[s.x, self.ea, self.eb, self.ech, self.p, s.h2, s.grad], device=s.dev)

    def pHp(self, s):
        wp.launch(pHp_pressure_kernel, dim=self.n_edges,
                  inputs=[s.p, self.ea, self.eb, self.ech, self.ch2env, self.p, s.h2, s.pHp],
                  device=s.dev)

    def energy(self, s, buf):
        wp.launch(energy_pressure_kernel, dim=self.n_edges,
                  inputs=[s.x, self.ea, self.eb, self.ech, self.ch2env, self.cref, self.p,
                          s.h2, buf], device=s.dev)


# ====================================================================== tendon
# A tendon is an ordered list of eyelets (mesh nodes) the cable threads through without
# friction. Its length is L = sum of the straight segments between consecutive eyelets, and
#     dL/dx_j = d_prev - d_next          (unit vectors of the segments meeting at eyelet j)
# is the direction the cable pulls that eyelet in, per unit tension.
#
#   TENSION-controlled (stiff == 0): a constant tension T, the cable simply pays in or out.
#       E = T L.  Curvature T/len (I - d d^T) per segment: positive semi-definite.
#   LENGTH-controlled (stiff > 0): a motor holds the free length at L0; the cable is an
#       elastic string of stiffness k = EA/L0 that cannot push.
#       E = k/2 (L - L0)+^2.  Tension k (L - L0)+, zero when slack.
#       Curvature = tension x (the same geometric term) + k dL dL^T while taut.
#
# The per-tendon state (L, tension, taut stiffness) is refreshed by grad_diag() and reused by
# pHp(), which the solver always evaluates at the same x within an iteration.
@wp.kernel
def tendon_length_kernel(x: wp.array(dtype=vec2), node: wp.array(dtype=wp.int32),
                         nxt: wp.array(dtype=wp.int32), tid_: wp.array(dtype=wp.int32),
                         L: wp.array(dtype=f64)):
    j = wp.tid()
    k = nxt[j]
    if k >= 0:
        wp.atomic_add(L, tid_[j], f64(wp.length(x[node[k]] - x[node[j]])))


@wp.kernel
def tendon_state_kernel(L: wp.array(dtype=f64), L0: wp.array(dtype=real),
                        stiff: wp.array(dtype=real), Tcmd: wp.array(dtype=real),
                        tau: wp.array(dtype=real), keff: wp.array(dtype=real)):
    k = wp.tid()
    if stiff[k] > real(0.0):
        ext = real(L[k]) - L0[k]
        if ext > real(0.0):
            tau[k] = stiff[k] * ext
            keff[k] = stiff[k]
        else:                                          # slack: a cable cannot push
            tau[k] = real(0.0)
            keff[k] = real(0.0)
    else:
        tau[k] = wp.max(Tcmd[k], real(0.0))
        keff[k] = real(0.0)


@wp.func
def _seg_dir(xa: vec2, xb: vec2) -> vec3:
    """(unit direction a->b, 1/length) packed as a vec3."""
    t = xb - xa
    ln = wp.max(wp.length(t), real(1.0e-12))
    return vec3(t[0] / ln, t[1] / ln, real(1.0) / ln)


@wp.kernel
def grad_diag_tendon_kernel(x: wp.array(dtype=vec2), node: wp.array(dtype=wp.int32),
                            prv: wp.array(dtype=wp.int32), nxt: wp.array(dtype=wp.int32),
                            tid_: wp.array(dtype=wp.int32), tau: wp.array(dtype=real),
                            keff: wp.array(dtype=real), h2: real,
                            grad: wp.array(dtype=vec2), diagH: wp.array(dtype=vec2)):
    j = wp.tid()
    n = node[j]
    k = tid_[j]
    gL = vec2(real(0.0), real(0.0))                              # dL/dx at this eyelet
    geo = vec2(real(0.0), real(0.0))                             # diagonal of sum (I - d d^T)/len
    if prv[j] >= 0:
        sp = _seg_dir(x[node[prv[j]]], x[n])
        gL = gL + vec2(sp[0], sp[1])
        geo = geo + vec2((real(1.0) - sp[0] * sp[0]) * sp[2], (real(1.0) - sp[1] * sp[1]) * sp[2])
    if nxt[j] >= 0:
        sn = _seg_dir(x[n], x[node[nxt[j]]])
        gL = gL - vec2(sn[0], sn[1])
        geo = geo + vec2((real(1.0) - sn[0] * sn[0]) * sn[2], (real(1.0) - sn[1] * sn[1]) * sn[2])
    wp.atomic_add(grad, n, (h2 * tau[k]) * gL)
    wp.atomic_add(diagH, n, vec2(h2 * (tau[k] * geo[0] + keff[k] * gL[0] * gL[0]),
                                    h2 * (tau[k] * geo[1] + keff[k] * gL[1] * gL[1])))


@wp.kernel
def pHp_tendon_kernel(x: wp.array(dtype=vec2), pdir: wp.array(dtype=vec2),
                      node: wp.array(dtype=wp.int32), nxt: wp.array(dtype=wp.int32),
                      tid_: wp.array(dtype=wp.int32), t2env: wp.array(dtype=wp.int32),
                      tau: wp.array(dtype=real), h2: real,
                      dL: wp.array(dtype=f64), pHp: wp.array(dtype=f64)):
    """Per segment: the change of length along p (summed per tendon into dL) and the
    geometric curvature tau |dp_perp|^2 / len."""
    j = wp.tid()
    m = nxt[j]
    if m < 0:
        return
    k = tid_[j]
    s = _seg_dir(x[node[j]], x[node[m]])
    d = vec2(s[0], s[1])
    dp = pdir[node[m]] - pdir[node[j]]
    along = wp.dot(dp, d)
    wp.atomic_add(dL, k, f64(along))
    wp.atomic_add(pHp, t2env[k], f64(h2 * tau[k] * (wp.dot(dp, dp) - along * along) * s[2]))


@wp.kernel
def pHp_tendon_taut_kernel(dL: wp.array(dtype=f64), keff: wp.array(dtype=real),
                           t2env: wp.array(dtype=wp.int32), h2: real,
                           pHp: wp.array(dtype=f64)):
    k = wp.tid()
    wp.atomic_add(pHp, t2env[k], f64(h2 * keff[k]) * dL[k] * dL[k])


@wp.kernel
def energy_tendon_kernel(L: wp.array(dtype=f64), L0: wp.array(dtype=real),
                         stiff: wp.array(dtype=real), Tcmd: wp.array(dtype=real),
                         t2env: wp.array(dtype=wp.int32), h2: real, E: wp.array(dtype=f64)):
    k = wp.tid()
    if stiff[k] > real(0.0):
        ext = L[k] - f64(L0[k])
        if ext > f64(real(0.0)):
            wp.atomic_add(E, t2env[k], f64(real(0.5) * h2 * stiff[k]) * ext * ext)
    else:
        wp.atomic_add(E, t2env[k], f64(h2 * wp.max(Tcmd[k], real(0.0))) * L[k])


class Tendon:
    in_adjoint = False        # its tangent is not in implicit/adjoint.py yet

    def __init__(self, s, node, prv, nxt, tid_, t2env, stiff, L0):
        ai = lambda v: wp.array(np.asarray(v, np.int32), dtype=wp.int32, device=s.dev)
        af = lambda v: wp.array(np.asarray(v, np_real), dtype=real, device=s.dev)
        self.node, self.prv, self.nxt, self.tid = ai(node), ai(prv), ai(nxt), ai(tid_)
        self.t2env = ai(t2env)
        self.n_ent, self.n_t = len(node), len(t2env)
        self.stiff, self.L0 = af(stiff), af(L0)
        self.Tcmd = wp.zeros(self.n_t, dtype=real, device=s.dev)
        self.tau = wp.zeros(self.n_t, dtype=real, device=s.dev)
        self.keff = wp.zeros(self.n_t, dtype=real, device=s.dev)
        self.L = wp.zeros(self.n_t, dtype=f64, device=s.dev)
        self.dL = wp.zeros(self.n_t, dtype=f64, device=s.dev)

    def set(self, tension, length):
        self.Tcmd.assign(np.ascontiguousarray(tension, np_real))
        self.L0.assign(np.ascontiguousarray(length, np_real))

    def _lengths(self, s):
        self.L.zero_()
        wp.launch(tendon_length_kernel, dim=self.n_ent,
                  inputs=[s.x, self.node, self.nxt, self.tid, self.L], device=s.dev)

    def grad_diag(self, s):
        self._lengths(s)
        wp.launch(tendon_state_kernel, dim=self.n_t,
                  inputs=[self.L, self.L0, self.stiff, self.Tcmd, self.tau, self.keff], device=s.dev)
        wp.launch(grad_diag_tendon_kernel, dim=self.n_ent,
                  inputs=[s.x, self.node, self.prv, self.nxt, self.tid, self.tau, self.keff,
                          s.h2, s.grad, s.diagH], device=s.dev)

    def pHp(self, s):
        self.dL.zero_()
        wp.launch(pHp_tendon_kernel, dim=self.n_ent,
                  inputs=[s.x, s.p, self.node, self.nxt, self.tid, self.t2env, self.tau, s.h2,
                          self.dL, s.pHp], device=s.dev)
        wp.launch(pHp_tendon_taut_kernel, dim=self.n_t,
                  inputs=[self.dL, self.keff, self.t2env, s.h2, s.pHp], device=s.dev)

    def energy(self, s, buf):
        self._lengths(s)
        wp.launch(energy_tendon_kernel, dim=self.n_t,
                  inputs=[self.L, self.L0, self.stiff, self.Tcmd, self.t2env, s.h2, buf],
                  device=s.dev)


# ====================================================================== active strain
# An actively deforming material -- shape-memory alloy, dielectric or liquid-crystal
# elastomer, thermal expansion, swelling, a muscle -- is modelled by changing the shape its
# elements WANT to be: the stress-free edge vectors become A Dm instead of Dm, so the
# elastic deformation gradient is F A^-1. Nothing in the elastic kernels changes; they read
# the inverse rest-shape matrix Bm and the rest area, and this rewrites those two arrays for
# the elements of each region:
#       Bm = Bm0 A^-1 ,   area = area0 det(A)
# With A = identity it restores the mesh exactly.
@wp.kernel
def apply_activation_kernel(elem: wp.array(dtype=wp.int32), ereg: wp.array(dtype=wp.int32),
                            Bm0: wp.array(dtype=mat22), area0: wp.array(dtype=real),
                            Ainv: wp.array(dtype=mat22), detA: wp.array(dtype=real),
                            Bm: wp.array(dtype=mat22), area: wp.array(dtype=real)):
    j = wp.tid()
    t = elem[j]
    r = ereg[j]
    Bm[t] = Bm0[t] * Ainv[r]
    area[t] = area0[t] * detA[r]


class ActiveStrain:
    """Not an energy term of its own: it re-shapes the rest configuration the elastic term
    already uses, so it has no grad / pHp / energy contribution."""
    in_adjoint = False        # the adjoint would need the activation at every checkpoint

    def __init__(self, s, elem, ereg, n_regions):
        ai = lambda v: wp.array(np.asarray(v, np.int32), dtype=wp.int32, device=s.dev)
        self.elem, self.ereg = ai(elem), ai(ereg)
        self.n_el, self.n_reg = len(elem), int(n_regions)
        if not hasattr(s, "Bm0"):                       # keep the undeformed rest shape
            s.Bm0 = wp.clone(s.Bm); s.area0 = wp.clone(s.area)
        self.Ainv = wp.zeros(self.n_reg, dtype=mat22, device=s.dev)
        self.detA = wp.zeros(self.n_reg, dtype=real, device=s.dev)

    def set(self, s, A):
        """A: (n_regions, 2, 2) activation deformation of each region's rest shape."""
        A = np.asarray(A, np.float64)
        self.Ainv.assign(np.ascontiguousarray(np.linalg.inv(A), np_real))
        self.detA.assign(np.ascontiguousarray(np.linalg.det(A), np_real))
        wp.launch(apply_activation_kernel, dim=self.n_el,
                  inputs=[self.elem, self.ereg, s.Bm0, s.area0, self.Ainv, self.detA,
                          s.Bm, s.area], device=s.dev)

    def grad_diag(self, s): pass
    def pHp(self, s): pass
    def energy(self, s, buf): pass


# ====================================================================== springs and pins
# A linear spring between two nodes, or between a node and a point fixed in the world:
#
#     E = k/2 (|d| - L0)^2,      d = x_a - x_b
#
# With L0 = 0 it is a PIN: it holds two points together and leaves rotation about them free,
# which is a revolute joint between two rigid bodies, or a pivot to the world. With L0 > 0 it
# is a strut or a linear actuator (command L0). k is per unit depth: [N/m per m] = [Pa].
#
#   grad_a  =  k (|d| - L0) n,                      n = d / |d|
#   H_aa    =  k n n^T + k (1 - L0/|d|) (I - n n^T)
# The second coefficient is negative when the spring is compressed; it is clamped at zero in
# the diagonal and in p^T H p, like every other indefinite term.
@wp.func
def _spring(xa: vec2, xb: vec2, k: real, L0: real) -> vec3:
    """(tension, tangential stiffness >= 0, length)"""
    d = xa - xb
    ln = wp.sqrt(wp.dot(d, d) + real(1.0e-30))
    return vec3(k * (ln - L0), wp.max(k * (real(1.0) - L0 / ln), real(0.0)), ln)


@wp.kernel
def grad_diag_spring_kernel(x: wp.array(dtype=vec2), na: wp.array(dtype=wp.int32),
                            nb: wp.array(dtype=wp.int32), anchor: wp.array(dtype=vec2),
                            k: wp.array(dtype=real), L0: wp.array(dtype=real), h2: real,
                            grad: wp.array(dtype=vec2), diagH: wp.array(dtype=vec2)):
    s = wp.tid()
    a = na[s]
    b = nb[s]
    xb = anchor[s]
    if b >= 0:
        xb = x[b]
    r = _spring(x[a], xb, k[s], L0[s])
    n = (x[a] - xb) / r[2]
    g = h2 * r[0] * n
    hd = h2 * vec2(k[s] * n[0] * n[0] + r[1] * (real(1.0) - n[0] * n[0]),
                   k[s] * n[1] * n[1] + r[1] * (real(1.0) - n[1] * n[1]))
    wp.atomic_add(grad, a, g)
    wp.atomic_add(diagH, a, hd)
    if b >= 0:
        wp.atomic_add(grad, b, -g)
        wp.atomic_add(diagH, b, hd)


@wp.kernel
def pHp_spring_kernel(x: wp.array(dtype=vec2), pdir: wp.array(dtype=vec2),
                      na: wp.array(dtype=wp.int32), nb: wp.array(dtype=wp.int32),
                      anchor: wp.array(dtype=vec2), k: wp.array(dtype=real),
                      L0: wp.array(dtype=real), s2env: wp.array(dtype=wp.int32), h2: real,
                      pHp: wp.array(dtype=f64)):
    s = wp.tid()
    a = na[s]
    b = nb[s]
    xb = anchor[s]
    dp = pdir[a]
    if b >= 0:
        xb = x[b]
        dp = dp - pdir[b]
    r = _spring(x[a], xb, k[s], L0[s])
    n = (x[a] - xb) / r[2]
    along = wp.dot(dp, n)
    wp.atomic_add(pHp, s2env[s], f64(h2 * (k[s] * along * along
                                           + r[1] * (wp.dot(dp, dp) - along * along))))


@wp.kernel
def energy_spring_kernel(x: wp.array(dtype=vec2), na: wp.array(dtype=wp.int32),
                         nb: wp.array(dtype=wp.int32), anchor: wp.array(dtype=vec2),
                         k: wp.array(dtype=real), L0: wp.array(dtype=real),
                         s2env: wp.array(dtype=wp.int32), h2: real, E: wp.array(dtype=f64)):
    s = wp.tid()
    xb = anchor[s]
    if nb[s] >= 0:
        xb = x[nb[s]]
    r = _spring(x[na[s]], xb, k[s], L0[s])
    ext = r[2] - L0[s]
    wp.atomic_add(E, s2env[s], f64(real(0.5) * h2 * k[s] * ext * ext))


class Springs:
    """Linear springs / pins. node_b = -1 attaches the spring to a world point (anchor)."""
    in_adjoint = False

    def __init__(self, s, node_a, node_b, stiffness, s2env):
        ai = lambda v: wp.array(np.asarray(v, np.int32), dtype=wp.int32, device=s.dev)
        self.na, self.nb, self.s2env = ai(node_a), ai(node_b), ai(s2env)
        self.n = len(node_a)
        self.k = wp.array(np.asarray(stiffness, np_real), dtype=real, device=s.dev)
        self.L0 = wp.zeros(self.n, dtype=real, device=s.dev)
        self.anchor = wp.zeros(self.n, dtype=vec2, device=s.dev)

    def set(self, anchor, length):
        self.anchor.assign(np.asarray(anchor, np_real).reshape(-1, 2))
        self.L0.assign(np.asarray(length, np_real))

    def grad_diag(self, s):
        wp.launch(grad_diag_spring_kernel, dim=self.n,
                  inputs=[s.x, self.na, self.nb, self.anchor, self.k, self.L0, s.h2,
                          s.grad, s.diagH], device=s.dev)

    def pHp(self, s):
        wp.launch(pHp_spring_kernel, dim=self.n,
                  inputs=[s.x, s.p, self.na, self.nb, self.anchor, self.k, self.L0, self.s2env,
                          s.h2, s.pHp], device=s.dev)

    def energy(self, s, buf):
        wp.launch(energy_spring_kernel, dim=self.n,
                  inputs=[s.x, self.na, self.nb, self.anchor, self.k, self.L0, self.s2env,
                          s.h2, buf], device=s.dev)
