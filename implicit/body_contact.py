"""Friction for the "node" contact between bodies (model.body_contact = "node").

"node" is the penalty contact this package had before IPC contact (ipc_contact.py), kept
because it is the one contact between bodies the adjoint can differentiate: two boundary
nodes of one environment closer than r repel with f_n = k (r - d) along the line between
them (kernels.py). This adds Coulomb friction on those same pairs, in exactly the form the
collider friction takes (kernels.py, grad_friction_disc_kernel), with the pair's RELATIVE
motion in place of a node's motion past a fixed surface:

    v    = ((x_i - x_i^n) - (x_j - x_j^n)) / dt        relative velocity over the step
    v_t  = v - (v.n) n                                 its part across the pair
    grad_i += h^2 mu f_n v_t / max(|v_t|, epsv)        opposes sliding; grad_j gets minus that
    k_t   = h^2 mu f_n / (dt max(|v_t|, epsv))         tangential stiffness of the smoothed law

k_t goes into the Jacobi diagonal and into p^T H p: below the smoothing speed the law is a
linear dashpot, and without its curvature the iteration overshoots that kink on every step
(the same finding as for the colliders). The potential it derives from, h^2 mu f_n f0(|u_t|),
is added to the energy so a line search sees it.

A pair is visited once from each of its nodes. Each visit adds that node's own gradient and
diagonal, and HALF of the pair's p^T H p and energy.

A surface made of nodes is corrugated — a node riding over a row of repelling nodes climbs
flanks that slope by up to 0.35 — so it resists sliding like friction of that size whatever
mu says. Use ipc_contact.py when friction matters.
"""
from __future__ import annotations
import numpy as np
import warp as wp
from ..core.precision import real, vec2, vec3
from ..core.math import pair_mu

f64 = wp.float64


@wp.kernel
def grad_diag_pair_friction_kernel(grid: wp.uint64, hx3: wp.array(dtype=wp.vec3),
                                   x: wp.array(dtype=vec2), xn: wp.array(dtype=vec2),
                                   boundary: wp.array(dtype=wp.int32),
                                   boundary2env: wp.array(dtype=wp.int32),
                                   boundary_start: wp.array(dtype=wp.int32),
                                   boundary_per_env: wp.array(dtype=wp.int32),
                                   excl_flat: wp.array(dtype=wp.uint8),
                                   excl_offset: wp.array(dtype=wp.int32),
                                   radius: real, stiffness: real, mu: real,
                                   mu_node: wp.array(dtype=real), per_body: int, epsv: real,
                                   inv_dt: real, h2: real, free: wp.array(dtype=vec2),
                                   grad: wp.array(dtype=vec2), diagH: wp.array(dtype=vec2)):
    b = wp.tid()
    node = boundary[b]
    e = boundary2env[b]
    start = boundary_start[e]
    n_be = boundary_per_env[e]
    i = b - start
    eoff = excl_offset[e]
    pa = x[node]
    ua = pa - xn[node]
    g = vec2(real(0.0), real(0.0))
    dgx = real(0.0)
    dgy = real(0.0)
    q = wp.hash_grid_query(grid, hx3[b], wp.float32(radius))
    for jb in q:
        if jb != b and boundary2env[jb] == e:
            jl = jb - start
            if excl_flat[eoff + i * n_be + jl] == wp.uint8(0):
                nj = boundary[jb]
                diff = pa - x[nj]
                dd = wp.sqrt(wp.dot(diff, diff) + real(1.0e-12))
                if dd < radius:
                    nh = diff / dd
                    f_n = stiffness * (radius - dd)
                    v = (ua - (x[nj] - xn[nj])) * inv_dt
                    vt = v - wp.dot(v, nh) * nh
                    vtmag = wp.length(vt)
                    mu_p = pair_mu(mu, per_body, mu_node, node, nj)
                    g = g + mu_p * f_n * vt / wp.max(vtmag, epsv)
                    kt = mu_p * f_n * inv_dt / wp.max(vtmag, epsv)
                    dgx = dgx + kt * nh[1] * nh[1]             # tangent = (-n_y, n_x)
                    dgy = dgy + kt * nh[0] * nh[0]
    fr = free[node]
    wp.atomic_add(grad, node, wp.cw_mul(h2 * g, fr))
    wp.atomic_add(diagH, node, wp.cw_mul(vec2(h2 * dgx, h2 * dgy), fr))


@wp.kernel
def pHp_pair_friction_kernel(grid: wp.uint64, hx3: wp.array(dtype=wp.vec3),
                             x: wp.array(dtype=vec2), xn: wp.array(dtype=vec2),
                             p: wp.array(dtype=vec2), boundary: wp.array(dtype=wp.int32),
                             boundary2env: wp.array(dtype=wp.int32),
                             boundary_start: wp.array(dtype=wp.int32),
                             boundary_per_env: wp.array(dtype=wp.int32),
                             excl_flat: wp.array(dtype=wp.uint8),
                             excl_offset: wp.array(dtype=wp.int32),
                             radius: real, stiffness: real, mu: real,
                             mu_node: wp.array(dtype=real), per_body: int, epsv: real,
                             inv_dt: real, h2: real, pHp: wp.array(dtype=f64)):
    b = wp.tid()
    node = boundary[b]
    e = boundary2env[b]
    start = boundary_start[e]
    n_be = boundary_per_env[e]
    i = b - start
    eoff = excl_offset[e]
    pa = x[node]
    ua = pa - xn[node]
    val = real(0.0)
    q = wp.hash_grid_query(grid, hx3[b], wp.float32(radius))
    for jb in q:
        if jb != b and boundary2env[jb] == e:
            jl = jb - start
            if excl_flat[eoff + i * n_be + jl] == wp.uint8(0):
                nj = boundary[jb]
                diff = pa - x[nj]
                dd = wp.sqrt(wp.dot(diff, diff) + real(1.0e-12))
                if dd < radius:
                    nh = diff / dd
                    f_n = stiffness * (radius - dd)
                    v = (ua - (x[nj] - xn[nj])) * inv_dt
                    vt = v - wp.dot(v, nh) * nh
                    kt = pair_mu(mu, per_body, mu_node, node, nj) * f_n * inv_dt / wp.max(wp.length(vt), epsv)
                    dp = p[node] - p[nj]
                    pt = dp[1] * nh[0] - dp[0] * nh[1]         # dp . tangent
                    val = val + real(0.5) * kt * pt * pt
    wp.atomic_add(pHp, e, f64(h2 * val))


@wp.kernel
def energy_pair_friction_kernel(grid: wp.uint64, hx3: wp.array(dtype=wp.vec3),
                                x: wp.array(dtype=vec2), xn: wp.array(dtype=vec2),
                                boundary: wp.array(dtype=wp.int32),
                                boundary2env: wp.array(dtype=wp.int32),
                                boundary_start: wp.array(dtype=wp.int32),
                                boundary_per_env: wp.array(dtype=wp.int32),
                                excl_flat: wp.array(dtype=wp.uint8),
                                excl_offset: wp.array(dtype=wp.int32),
                                radius: real, stiffness: real, mu: real,
                                mu_node: wp.array(dtype=real), per_body: int, epsv: real,
                                inv_dt: real, h2: real, E: wp.array(dtype=f64)):
    b = wp.tid()
    node = boundary[b]
    e = boundary2env[b]
    start = boundary_start[e]
    n_be = boundary_per_env[e]
    i = b - start
    eoff = excl_offset[e]
    pa = x[node]
    ua = pa - xn[node]
    a = epsv / inv_dt
    val = real(0.0)
    q = wp.hash_grid_query(grid, hx3[b], wp.float32(radius))
    for jb in q:
        if jb != b and boundary2env[jb] == e:
            jl = jb - start
            if excl_flat[eoff + i * n_be + jl] == wp.uint8(0):
                nj = boundary[jb]
                diff = pa - x[nj]
                dd = wp.sqrt(wp.dot(diff, diff) + real(1.0e-12))
                if dd < radius:
                    nh = diff / dd
                    f_n = stiffness * (radius - dd)
                    u = ua - (x[nj] - xn[nj])
                    ut = wp.abs(u[1] * nh[0] - u[0] * nh[1])   # |tangential slip|
                    f0 = ut - real(0.5) * a
                    if ut < a:
                        f0 = ut * ut / (real(2.0) * a)
                    val = val + real(0.5) * pair_mu(mu, per_body, mu_node, node, nj) * f_n * f0
    wp.atomic_add(E, e, f64(h2 * val))


class BodyFriction:
    """Coulomb friction on the node-to-node contact pairs. Uses the solver's own contact
    broadphase, which the solver rebuilds at the current positions just before its terms
    run, so the pairs here are exactly the pairs the penalty acts on."""
    in_adjoint = False
    persistent = True           # not an actuator: survives the integrator's term rebuild

    def __init__(self, s, mu, mu_node=None):
        if s.self_stiffness <= 0.0:
            raise ValueError("friction between bodies needs contact between bodies: leave "
                             "self_collision on")
        self.mu = float(mu)
        self.per_body = 0 if mu_node is None else 1               # see IPCContact
        self.mu_node = wp.array(np.zeros(1) if mu_node is None else np.asarray(mu_node, np.float64),
                                dtype=real, device=s.dev)

    def _args(self, s):
        return [s.sc_grid.id, s.hx3, s.x, s.xn]

    def _tail(self, s):
        return [s.bnd, s.b2e, s.sc_bstart, s.sc_bpe, s.sc_excl, s.sc_excl_off,
                s.self_radius, s.self_stiffness, self.mu, self.mu_node, self.per_body,
                s.epsv, s.inv_dt, s.h2]

    def grad_diag(self, s):
        wp.launch(grad_diag_pair_friction_kernel, dim=s.n_b,
                  inputs=self._args(s) + self._tail(s) + [s.free, s.grad, s.diagH], device=s.dev)

    def pair_force(self, s, grad, diagH, free):
        """Add MINUS the friction force on every node to `grad` (solver.contact_force)."""
        tail = self._tail(s); tail[-1] = 1.0                           # h^2 = 1
        wp.launch(grad_diag_pair_friction_kernel, dim=s.n_b,
                  inputs=self._args(s) + tail + [free, grad, diagH], device=s.dev)

    def pHp(self, s):
        wp.launch(pHp_pair_friction_kernel, dim=s.n_b,
                  inputs=self._args(s) + [s.p] + self._tail(s) + [s.pHp], device=s.dev)

    def energy(self, s, buf):
        wp.launch(energy_pair_friction_kernel, dim=s.n_b,
                  inputs=self._args(s) + self._tail(s) + [buf], device=s.dev)
