"""Adjoint of the implicit PNCG-IPC solve (implicit function theorem): dJ/d(density).

Composes a PNCG2DSolver and never modifies the forward solve. Density enters as a
per-element multiplier mu_e = E(rho_e) mu0, lam_e = E(rho_e) lam0, with
E(rho) = Emin + rho (1 - Emin). The gradient dJ/drho comes from ONE linear solve per
checkpointed step with the tangent Hessian (matrix-free Hessian-vector products), the
contact active set FROZEN at the converged state x*:

    forward to the fixed point R(x*, rho) = 0   (not taped)
    H lam = dJ/dx
    dJ/drho_e = -(1 - Emin) lam . [h^2 f_unit_e(x*)]        (unit-modulus element force)

A trajectory is differentiated by the DAMPED backward-Euler recursion

    H_k lam_k = dJ/dx_k + (1 + drag) M lam_{k+1} - drag M lam_{k+2}      (drag = solver.drag)

In the operator: inertia, elasticity, the barrier of ONE collider (the disc, or any other
shape through its level set) and the "node" contact between bodies. Not in it: friction,
IPC contact between bodies, pressure, tendons, active strain, springs, rigid bodies —
transient_adjoint refuses to run with a term it has no tangent for.
"""
from __future__ import annotations
import numpy as np
import warp as wp
from ..core.precision import real, vec2, vec3, mat22, np_real
from ..core.math import (nh_dPsidF, nh_d2PsidF2_dir, dF_from_vec6, dF_from_dir, vec6,
                         barrier_bp, barrier_bpp)
from ..core.sdf import sdf_eval                             # the colliders' level sets
from . import kernels as _PK                                # forward grad_diag kernels (preconditioner)
from ..core import sdf as _SDF                              # sdf grad_diag kernel

f64 = wp.float64


@wp.func
def _ds(p0: vec2, p1: vec2, p2: vec2) -> mat22:
    return mat22(p1[0] - p0[0], p2[0] - p0[0], p1[1] - p0[1], p2[1] - p0[1])


@wp.kernel
def set_density_kernel(rho: wp.array(dtype=real), mu0: wp.array(dtype=real),
                       lam0: wp.array(dtype=real), Emin: real,
                       mu: wp.array(dtype=real), lam: wp.array(dtype=real)):
    """mu0 / lam0 are the FULL-DENSITY moduli of each element: a scene may hold bodies of
    different materials, or a body with a stiffness map, and density scales each element's
    own modulus."""
    t = wp.tid()
    E = Emin + rho[t] * (real(1.0) - Emin)
    mu[t] = E * mu0[t]
    lam[t] = E * lam0[t]


@wp.kernel
def force_elastic_kernel(x: wp.array(dtype=vec2), tri: wp.array(dtype=wp.vec3i),
                         B: wp.array(dtype=mat22), area: wp.array(dtype=real),
                         mu: wp.array(dtype=real), lam: wp.array(dtype=real),
                         h2: real, R: wp.array(dtype=vec2)):
    """R += h2 * f_elastic (= dE_elastic/dx); assembly identical to grad_diag_elastic force part."""
    t = wp.tid()
    ids = tri[t]; i0 = ids[0]; i1 = ids[1]; i2 = ids[2]
    Bt = B[t]
    HH = area[t] * (nh_dPsidF(_ds(x[i0], x[i1], x[i2]) * Bt, mu[t], lam[t]) * wp.transpose(Bt))
    g1 = vec2(HH[0, 0], HH[1, 0]); g2 = vec2(HH[0, 1], HH[1, 1])
    wp.atomic_add(R, i0, h2 * (-(g1 + g2)))
    wp.atomic_add(R, i1, h2 * g1)
    wp.atomic_add(R, i2, h2 * g2)


@wp.kernel
def hvp_elastic_kernel(x: wp.array(dtype=vec2), v: wp.array(dtype=vec2),
                       tri: wp.array(dtype=wp.vec3i), B: wp.array(dtype=mat22),
                       area: wp.array(dtype=real), mu: wp.array(dtype=real),
                       lam: wp.array(dtype=real), h2: real, y: wp.array(dtype=vec2)):
    """y += h2 * (d/dx f_elastic) v  — vector form; assembly mirrors grad_diag_elastic. UNCLAMPED."""
    t = wp.tid()
    ids = tri[t]; i0 = ids[0]; i1 = ids[1]; i2 = ids[2]
    Bt = B[t]
    F = _ds(x[i0], x[i1], x[i2]) * Bt
    ve = vec6(v[i0][0], v[i0][1], v[i1][0], v[i1][1], v[i2][0], v[i2][1])
    dP = nh_d2PsidF2_dir(F, dF_from_vec6(ve, Bt), mu[t], lam[t])
    HH = area[t] * (dP * wp.transpose(Bt))
    h1 = vec2(HH[0, 0], HH[1, 0]); h2v = vec2(HH[0, 1], HH[1, 1])
    wp.atomic_add(y, i0, h2 * (-(h1 + h2v)))
    wp.atomic_add(y, i1, h2 * h1)
    wp.atomic_add(y, i2, h2 * h2v)


@wp.kernel
def chvp_disc_kernel(x: wp.array(dtype=vec2), v: wp.array(dtype=vec2),
                     bnd: wp.array(dtype=wp.int32), b2e: wp.array(dtype=wp.int32),
                     c: vec2, radius: real, dhat: real, kappa: real, h2: real,
                     y: wp.array(dtype=vec2)):
    """y += h2 * H_disc v (vector form of pHp_disc). Frozen gate d in (0,dhat). UNCLAMPED."""
    b = wp.tid(); n = bnd[b]
    diff = x[n] - c; dc = wp.length(diff); d = dc - radius
    if d <= real(0.0) or d >= dhat:
        return
    nh = diff / dc; pn = wp.dot(v[n], nh); pt = v[n] - pn * nh
    wp.atomic_add(y, n, h2 * kappa * (barrier_bpp(d, dhat) * pn * nh + (barrier_bp(d, dhat) / dc) * pt))


@wp.kernel
def chvp_sdf_kernel(x: wp.array(dtype=vec2), v: wp.array(dtype=vec2), bnd: wp.array(dtype=wp.int32),
                    kind: int, cx: real, cy: real, rx: real, ry: real, rref: real,
                    dhat: real, kappa: real, h2: real, y: wp.array(dtype=vec2)):
    """y += h2 * H_contact v for an ARBITRARY level-set object: kappa*(bpp*(g.v)g + bp*(Hphi v)).
    Reduces to chvp_disc when the object is a circle. Frozen gate d in (0,dhat)."""
    b = wp.tid(); n = bnd[b]; s = sdf_eval(kind, cx, cy, rx, ry, rref, x[n])
    if s.d <= real(0.0) or s.d >= dhat:
        return
    gv = wp.dot(s.g, v[n])
    wp.atomic_add(y, n, h2 * kappa * (barrier_bpp(s.d, dhat) * gv * s.g + barrier_bp(s.d, dhat) * (s.H @ v[n])))


@wp.kernel
def self_bp_hvp_kernel(grid: wp.uint64, hx3: wp.array(dtype=wp.vec3),
                       x: wp.array(dtype=vec2), v: wp.array(dtype=vec2),
                       boundary: wp.array(dtype=wp.int32), boundary2env: wp.array(dtype=wp.int32),
                       boundary_start: wp.array(dtype=wp.int32), boundary_per_env: wp.array(dtype=wp.int32),
                       excl_flat: wp.array(dtype=wp.uint8), excl_offset: wp.array(dtype=wp.int32),
                       radius: real, stiffness: real, h2: real, y: wp.array(dtype=vec2)):
    """y += h2 * H_self v (OWN-ROW: each tid writes only its node; NO 0.5). UNCLAMPED.
    Frozen active set: grid built ONCE at x*. Mirrors grad_diag_self_bp broadphase."""
    b = wp.tid()
    node = boundary[b]; e = boundary2env[b]
    start = boundary_start[e]; n_be = boundary_per_env[e]
    i = b - start; eoff = excl_offset[e]
    pa = x[node]; va = v[node]
    hv = vec2(real(0.0), real(0.0))
    q = wp.hash_grid_query(grid, hx3[b], wp.float32(radius))
    for jb in q:
        if jb != b and boundary2env[jb] == e:
            jl = jb - start
            if excl_flat[eoff + i * n_be + jl] == wp.uint8(0):
                diff = pa - x[boundary[jb]]
                dd = wp.sqrt(wp.dot(diff, diff) + real(1.0e-12))
                if dd < radius:
                    nh = diff / dd
                    rd = radius - dd
                    dv = va - v[boundary[jb]]
                    vn = wp.dot(dv, nh); vt = dv - vn * nh
                    hv = hv + stiffness * vn * nh - (stiffness * rd / dd) * vt
    wp.atomic_add(y, node, h2 * hv)


@wp.kernel
def assemble_Ke_kernel(x: wp.array(dtype=vec2), tri: wp.array(dtype=wp.vec3i),
                       B: wp.array(dtype=mat22), area: wp.array(dtype=real),
                       mu: wp.array(dtype=real), lam: wp.array(dtype=real), Kf: wp.array(dtype=real)):
    """Per-element 6x6 elastic tangent (row-major into Kf[t*36:...]); host assembles the sparse K.
    Direct/MINRES solve of the assembled K is robust to the INDEFINITE Neo-Hookean tangent (CG is not)."""
    t = wp.tid(); ids = tri[t]; Bt = B[t]
    F = _ds(x[ids[0]], x[ids[1]], x[ids[2]]) * Bt; m = mu[t]; l = lam[t]; A = area[t]
    for j in range(6):
        nn = j // 2; c = j - 2 * nn
        dP = nh_d2PsidF2_dir(F, dF_from_dir(nn, c, Bt), m, l)
        HH = A * (dP * wp.transpose(Bt))
        f1 = vec2(HH[0, 0], HH[1, 0]); f2 = vec2(HH[0, 1], HH[1, 1]); f0 = -(f1 + f2)
        Kf[t*36 + 0*6 + j] = f0[0]; Kf[t*36 + 1*6 + j] = f0[1]
        Kf[t*36 + 2*6 + j] = f1[0]; Kf[t*36 + 3*6 + j] = f1[1]
        Kf[t*36 + 4*6 + j] = f2[0]; Kf[t*36 + 5*6 + j] = f2[1]


@wp.kernel
def drho_kernel(x: wp.array(dtype=vec2), lam: wp.array(dtype=vec2),
                tri: wp.array(dtype=wp.vec3i), B: wp.array(dtype=mat22),
                area: wp.array(dtype=real), mu0: wp.array(dtype=real),
                lam0: wp.array(dtype=real), h2: real, Emin: real,
                dJdrho: wp.array(dtype=real)):
    """dJ/drho_e += -(1-Emin) * lam . [h2 * f_unit_e(x)]  (unit-modulus; accumulates over steps)."""
    t = wp.tid()
    ids = tri[t]; i0 = ids[0]; i1 = ids[1]; i2 = ids[2]
    Bt = B[t]
    F = _ds(x[i0], x[i1], x[i2]) * Bt
    HH = area[t] * (nh_dPsidF(F, mu0[t], lam0[t]) * wp.transpose(Bt))   # FULL-density element force
    g1 = vec2(HH[0, 0], HH[1, 0]); g2 = vec2(HH[0, 1], HH[1, 1])
    s = wp.dot(lam[i0], -(g1 + g2)) + wp.dot(lam[i1], g1) + wp.dot(lam[i2], g2)
    wp.atomic_add(dJdrho, t, -(real(1.0) - Emin) * h2 * s)


class AdjointPNCG2D:
    """Wraps a PNCG2DSolver to provide dJ/drho via the implicit adjoint: a host Krylov
    solve whose matrix-vector product runs through the Warp Hessian-vector kernels."""

    def __init__(self, solver, Emin=1e-3):
        self.s = solver; self.dev = solver.dev
        me = solver.me
        self.Emin = float(Emin)
        # full-density moduli, per element (the mesh's own: per-body materials and stiffness
        # maps are already in me.mu / me.lam)
        self.mu0 = wp.array(np.asarray(me.mu, np_real), dtype=real, device=self.dev)
        self.lam0 = wp.array(np.asarray(me.lam, np_real), dtype=real, device=self.dev)
        self.N = solver.N; self.n_tri = solver.n_tri
        self.h2 = float(solver.h2)
        # device scratch
        self.free_np = solver.free.numpy().astype(np.float64)          # (N,2) 1=free 0=bc
        self.mass_np = solver.mass.numpy().astype(np.float64)          # (N,) SNAPSHOT -- see refresh_mass

        self._y = wp.zeros(self.N, dtype=vec2, device=self.dev)
        self._drho = wp.zeros(self.n_tri, dtype=real, device=self.dev)

    # ---- mass ----
    def refresh_mass(self):
        """Re-read the solver's nodal mass into the ``mass_np`` snapshot.

        ``mass_np`` is cached (not a live read) because ``apply_H`` IS the Krylov matvec and
        already pays a device->host sync per iteration — reading ``s.mass`` there would
        double that on the hottest path. So the cache has to be refreshed whenever
        ``solver.mass`` is changed after this adjoint was built (a mass that follows the
        density, say); otherwise ``apply_H`` would keep the old values while
        ``_diag_precond`` reads the live ones, and the operator and its preconditioner
        would silently disagree. ``transient_adjoint`` calls this for you.
        """
        self.mass_np = self.s.mass.numpy().astype(np.float64)
        return self.mass_np

    # ---- density ----
    def set_density(self, rho_e):
        rw = wp.array(np.asarray(rho_e, np_real), dtype=real, device=self.dev)
        wp.launch(set_density_kernel, dim=self.n_tri,
                  inputs=[rw, self.mu0, self.lam0, self.Emin, self.s.mu, self.s.lam], device=self.dev)

    # ---- matrix-free operator  H*v = M v + h2 (K_elastic + K_self) v, restricted to free ----
    def apply_H(self, xstar_w, v_np, use_self=False, use_disc=False, use_sdf=False, include_mass=True):
        vw = wp.array(v_np.astype(np_real), dtype=vec2, device=self.dev)
        self._y.zero_()
        wp.launch(hvp_elastic_kernel, dim=self.n_tri,
                  inputs=[xstar_w, vw, self.s.tri, self.s.Bm, self.s.area, self.s.mu, self.s.lam,
                          self.h2, self._y], device=self.dev)
        if use_disc:                                       # disc-contact H*v; frozen gate at x*
            wp.launch(chvp_disc_kernel, dim=self.s.n_b,
                      inputs=[xstar_w, vw, self.s.bnd, self.s.b2e, self.s.c, self.s.radius,
                              self.s.dhat, self.s.kappa, self.h2, self._y], device=self.dev)
        if use_sdf:                                        # any other collider shape, through its level set
            wp.launch(chvp_sdf_kernel, dim=self.s.n_b,
                      inputs=[xstar_w, vw, self.s.bnd, self.s.object_kind, self.s.obj_cx, self.s.obj_cy,
                              self.s.obj_rx, self.s.obj_ry, self.s.obj_rref, self.s.dhat, self.s.kappa,
                              self.h2, self._y], device=self.dev)
        if use_self and self.s.self_stiffness > 0.0:
            wp.launch(self_bp_hvp_kernel, dim=self.s.n_b,
                      inputs=[self.s.sc_grid.id, self.s.hx3, xstar_w, vw, self.s.bnd, self.s.b2e,
                              self.s.sc_bstart, self.s.sc_bpe, self.s.sc_excl, self.s.sc_excl_off,
                              self.s.self_radius, self.s.self_stiffness, self.h2, self._y],
                      device=self.dev)
        Kv = self._y.numpy().astype(np.float64)
        return (self.mass_np[:, None] * v_np + Kv) if include_mass else Kv   # (N,2)

    def spd_probe(self, xstar_w, use_self=False, include_mass=True, n=16, seed=0):
        """min Rayleigh quotient p^T H p / p^T p over random free directions (SPD sanity)."""
        rng = np.random.default_rng(seed); free = self.free_np; vals = []
        for _ in range(n):
            p = rng.standard_normal((self.N, 2)) * free
            Hp = self.apply_H(xstar_w, p, use_self, include_mass) * free
            vals.append((p * Hp).sum() / (p * p).sum())
        return float(min(vals)), float(max(vals))

    def adjoint_cg(self, xstar_w, b_np, diagH_np, use_self=False, include_mass=True, iters=8000, tol=1e-9):
        free = self.free_np
        precond = free / np.maximum(diagH_np, 1e-30)       # (N,2) Jacobi, 0 on bc
        lam = np.zeros((self.N, 2))
        r = (b_np * free).copy()
        z = precond * r; p = z.copy(); rz = (r * z).sum()
        r0 = np.linalg.norm(r); it = 0; neg = 0
        for it in range(1, iters + 1):
            Ap = self.apply_H(xstar_w, p * free, use_self, include_mass) * free
            pAp = (p * Ap).sum()
            if pAp <= 0:                                    # indefinite -> record
                neg += 1
            a = rz / pAp
            lam += a * p; r -= a * Ap
            if np.linalg.norm(r) <= tol * r0 + 1e-30:
                break
            z = precond * r; rz2 = (r * z).sum(); p = z + (rz2 / rz) * p; rz = rz2
        self._cg_info = dict(iters=it, rel_resid=float(np.linalg.norm(r) / (r0 + 1e-30)), neg_curv=neg)
        return lam

    def drho(self, xstar_w, lam_np):
        self._drho.zero_()
        lamw = wp.array(lam_np.astype(np_real), dtype=vec2, device=self.dev)
        wp.launch(drho_kernel, dim=self.n_tri,
                  inputs=[xstar_w, lamw, self.s.tri, self.s.Bm, self.s.area, self.mu0, self.lam0,
                          self.h2, self.Emin, self._drho], device=self.dev)
        return self._drho.numpy().astype(np.float64)

    # ---- objectives on the elastic force: the reaction at held nodes ----
    def elastic_force(self, x_np):
        """Elastic force on every node at positions x, (N, 2), with the current density —
        minus the gradient of the strain energy. At a held (kinematic) node it is the force
        the body passes to whatever holds it: the reaction, in a quasi-static load."""
        xw = wp.array(np.asarray(x_np).astype(np_real), dtype=vec2, device=self.dev)
        g = wp.zeros(self.N, dtype=vec2, device=self.dev)
        d = wp.zeros(self.N, dtype=vec2, device=self.dev)
        wp.launch(_PK.grad_diag_elastic_kernel, dim=self.n_tri,
                  inputs=[xw, self.s.tri, self.s.Bm, self.s.area, self.s.mu, self.s.lam, 1.0, 1, g, d],
                  device=self.dev)
        return -g.numpy().astype(np.float64)

    def force_objective(self, x_np, nodes, direction):
        """J = direction . (the elastic force summed over `nodes`) at state x — e.g. the
        reaction along `direction` of an output held at those nodes. Returns
        (J, dJ/dx, explicit dJ/drho). J depends on the density directly as well as through
        the positions, so the gradient is

            g = transient_adjoint(xs, seed={k: dJ/dx}, rtol=1e-14) + explicit      (x = xs[k])

        The tight rtol matters: this seed (stiffness times a vector at a few nodes) is rough
        and badly scaled, and MINRES's rtol bounds the PRECONDITIONED residual — at the
        default 1e-9 the true residual was 1.6e-3 and the gradient 4 % off; at 1e-14 it
        matches finite differences to 1e-5 (a held plate corner, six steps)."""
        nodes = np.asarray(nodes, np.int64)
        d = np.asarray(direction, np.float64)
        w = np.zeros((self.N, 2)); w[nodes] = d                 # J = -w . grad(strain energy)
        xw = wp.array(np.asarray(x_np).astype(np_real), dtype=vec2, device=self.dev)
        J = float((self.elastic_force(x_np)[nodes] * d).sum())
        dJdx = -self.apply_H(xw, w, include_mass=False) / self.h2      # -K w (K symmetric)
        return J, dJdx, self.drho(xw, w / self.h2)                     # -(1-Emin) w . f_unit_e

    # ---- Jacobi preconditioner: diag(M + h2 (K_elastic + contact)), via the forward grad_diag kernels ----
    def _diag_precond(self, xw, use_disc=False, use_self=False, use_sdf=False):
        self.s.diagH.zero_(); self.s.grad.zero_()
        wp.launch(_PK.grad_diag_mass_kernel, dim=self.N,
                  inputs=[xw, self.s.xhat, self.s.mass, self.s.free, self.s.grad, self.s.grad_prev, self.s.diagH], device=self.dev)
        wp.launch(_PK.grad_diag_elastic_kernel, dim=self.n_tri,
                  inputs=[xw, self.s.tri, self.s.Bm, self.s.area, self.s.mu, self.s.lam, self.h2, 1, self.s.grad, self.s.diagH], device=self.dev)
        if use_disc:
            wp.launch(_PK.grad_diag_disc_kernel, dim=self.s.n_b,
                      inputs=[xw, self.s.bnd, self.s.c, self.s.radius, self.s.dhat, self.s.kappa, self.h2, 1, self.s.grad, self.s.diagH], device=self.dev)
        if use_sdf:
            wp.launch(_SDF.grad_diag_sdf_kernel, dim=self.s.n_b,
                      inputs=[xw, self.s.bnd, self.s.object_kind, self.s.obj_cx, self.s.obj_cy, self.s.obj_rx,
                              self.s.obj_ry, self.s.obj_rref, self.s.dhat, self.s.kappa, self.h2, 1, self.s.grad, self.s.diagH], device=self.dev)
        if use_self and self.s.self_stiffness > 0.0:
            wp.launch(_PK.grad_diag_self_bp_kernel, dim=self.s.n_b,
                      inputs=[self.s.sc_grid.id, self.s.hx3, xw, self.s.bnd, self.s.b2e, self.s.sc_bstart, self.s.sc_bpe,
                              self.s.sc_excl, self.s.sc_excl_off, self.s.self_radius, self.s.self_stiffness, self.h2, 1, self.s.grad, self.s.diagH], device=self.dev)
        return np.maximum(self.s.diagH.numpy().astype(np.float64).reshape(-1), 1e-30)

    # ---- one adjoint solve H_k lam = rhs on free dofs (MINRES; robust to the indefinite contact tangent) ----
    def adjoint_solve(self, xk_np, rhs, use_disc=False, use_self=False, use_sdf=False, rtol=1e-9, maxiter=5000):
        from scipy.sparse.linalg import minres, LinearOperator
        if use_self and self.s.self_stiffness > 0.0:                   # freeze the self-collision grid at x_k
            self.s.x.assign(xk_np.astype(np_real)); self.s._build_sc_grid()
        xw = wp.array(xk_np.astype(np_real), dtype=vec2, device=self.dev)
        ff = self.free_np.reshape(-1) > 0.5; nf = int(ff.sum())
        dinv = 1.0 / self._diag_precond(xw, use_disc, use_self, use_sdf)[ff]

        def mv(vf):
            full = np.zeros((self.N, 2)); full.reshape(-1)[ff] = vf
            Hv = self.apply_H(xw, full, use_self=use_self, use_disc=use_disc, use_sdf=use_sdf) * self.free_np
            return Hv.reshape(-1)[ff]
        A = LinearOperator((nf, nf), matvec=mv); M = LinearOperator((nf, nf), matvec=lambda vf: dinv * vf)
        bf = np.asarray(rhs).reshape(-1)[ff]
        lf, info = minres(A, bf, M=M, rtol=rtol, maxiter=maxiter)
        # DO NOT discard `info`. scipy returns 0 = converged, >0 = maxiter hit WITHOUT converging,
        # <0 = breakdown -- and MINRES assumes a HERMITIAN operator without checking, so a silent
        # failure here returns plausible-looking garbage. Record it; the caller can inspect
        # .minres_calls / .minres_fail / .minres_resid after a transient_adjoint.
        # NB rtol=1e-9 may be unreachable for the same reason eps=1e-8 was on the forward: apply_H
        # accumulates with atomics, so the operator is not deterministic to 1e-9.
        self.minres_calls = getattr(self, "minres_calls", 0) + 1
        if info != 0:
            self.minres_fail = getattr(self, "minres_fail", 0) + 1
            self.minres_last_info = int(info)
        if getattr(self, "minres_track_resid", False):        # one extra matvec; off by default
            nb = np.linalg.norm(bf)
            r = np.linalg.norm(A.matvec(lf) - bf) / max(nb, 1e-300)
            self.minres_resid = max(getattr(self, "minres_resid", 0.0), float(r))
        lam = np.zeros((self.N, 2)); lam.reshape(-1)[ff] = lf
        return lam

    # ---- turnkey transient adjoint: damped backward-in-time recursion over a checkpointed trajectory ----
    def transient_adjoint(self, xs, seed, drag=None, use_disc=False, use_self=False, use_sdf=False, **minres_kw):
        """xs = [x_0 .. x_T] (checkpointed forward states). seed = {k: dJ/dx_k (N,2)} objective
        injection per step. Returns dJ/drho (n_tri,) accumulated over the trajectory. Coupling
        (1+drag,-drag)*M matches the solver's damped backward-Euler predictor."""
        missing = sorted({type(t).__name__ for t in list(getattr(self.s, "terms", [])) +
                          [getattr(self.s, "_act", {}).get("strain")]
                          if t is not None and not getattr(t, "in_adjoint", True)})
        if missing:
            raise NotImplementedError(
                f"the adjoint operator has no tangent for {', '.join(missing)}: the gradient "
                f"would be of a different system. Differentiate without these actuators."
                + (" For contact between bodies, use ImplicitIntegrator(self_collision=False) "
                   "or model.body_contact = 'node', which the adjoint does have a tangent for."
                   if "IPCContact" in missing else ""))
        T = len(xs) - 1
        dr = float(self.s.drag) if drag is None else float(drag)
        self.refresh_mass()          # solver.mass may have been changed since __init__
        m2 = self.mass_np[:, None]
        lam = {T + 1: np.zeros((self.N, 2)), T + 2: np.zeros((self.N, 2))}
        g = np.zeros(self.n_tri)
        for k in range(T, 0, -1):
            rhs = (1.0 + dr) * m2 * lam[k + 1] - dr * m2 * lam[k + 2]
            if k in seed:
                rhs = rhs + seed[k]
            lam[k] = self.adjoint_solve(xs[k], rhs, use_disc=use_disc, use_self=use_self, use_sdf=use_sdf, **minres_kw)
            xw = wp.array(xs[k].astype(np_real), dtype=vec2, device=self.dev)
            g += self.drho(xw, lam[k])
        return g
