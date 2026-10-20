"""Body-level modes in the PNCG solve: rigid (affine) bodies, and a rigid-mode coarse space.

The solver's own unknowns are node positions and its preconditioner is nodal (Jacobi).
That is the right scale for a node moving against its neighbours and the wrong one for a
body moving as a whole: a rigid translation costs no elastic energy at all, yet Jacobi
prices it at the full elastic stiffness of every node, so the step it proposes is too
short by (c dt / dx)^2 — four to six orders of magnitude. This file adds the missing
body-level directions in two forms.

AffineBodies — RIGID BODIES, as stiff affine bodies (Lan et al. 2022, "Affine Body
    Dynamics"). The nodes of a rigid region are slaved to six numbers

        x_i = t + A (X_i - c),        q = (t_x, A_11, A_12, t_y, A_21, A_22)

    so the region can only translate, rotate and — against a very large stiffness —
    strain uniformly. The map is LINEAR in q, x_i = J_i q, which is what lets it live
    inside a conjugate-gradient iteration: the search direction of a slaved node is J_i
    applied to a 6-vector, and every nodal kernel downstream (curvature, inversion and
    CCD step caps, contact, friction, energy) sees an ordinary nodal direction and needs
    no change. The six unknowns get their own 6 x 6 block of the preconditioner:

        H_q = J^T D J  +  h^2 W d2Psi/dA2

    D is the nodal diagonal — lumped mass (exact: the mass matrix is diagonal), contact,
    friction, and the elastic diagonal of soft elements attached to the region — and the
    second term is the region's own elastic Hessian, which for a uniformly strained region
    is one closed-form 4 x 4. At A = rotation that 4 x 4 has an exact null direction, the
    rotation itself, so rotating costs only inertia and contact. That is the whole point.

RigidModes — a COARSE SPACE for free soft bodies. Every connected piece of soft material
    that nothing holds gets three extra directions — two translations and a rotation
    about its centroid — added to the preconditioner,

        P = D^-1 + B (B^T D' B)^-1 B^T,      D' = D without the elastic diagonal

    (elasticity does not resist a rigid motion, so its diagonal must not either). The
    nodes stay free unknowns; only the scaling of the three rigid directions changes.
    Without it a body resting on the ground, or falling, converges in hundreds of
    iterations; with it, in a handful.

Both are per body and per environment like everything else in the solver.
"""
from __future__ import annotations
import numpy as np
import warp as wp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..core.precision import real, vec2, vec3, mat22, np_real
from ..core.math import vec6, nh_dPsidF, nh_d2PsidF2_dir, LOG_EPS

f64 = wp.float64
mat33 = wp.types.matrix((3, 3), real)
mat66 = wp.types.matrix((6, 6), real)
vec4 = wp.types.vector(4, real)
INV_SQRT2 = wp.constant(real(0.7071067811865476))


# ====================================================================== 6 x 6 block algebra
@wp.func
def reduced_hessian(A: mat22, Mu: real, La: real, h2: real, Hx: mat33, Hy: mat33) -> mat66:
    """J^T D J (two 3 x 3 blocks: a diagonal D does not couple x with y) plus the elastic
    Hessian of a region strained uniformly by A, on the entries (A11, A12, A21, A22) =
    q[1], q[2], q[4], q[5]. Mu and La are the region's moduli summed over its area.

    The one direction the neo-Hookean Hessian can be negative along at small strain is the
    TWIST, T = R J_perp / sqrt(2) with R the rotation of A: it is an exact eigenvector, and
    its eigenvalue mu + (la log J - mu) / J goes below zero under any compression. It is
    removed when negative, which leaves a positive semi-definite block."""
    H = mat66()
    for r in range(3):
        for c in range(3):
            H[r, c] = Hx[r, c]
            H[r + 3, c + 3] = Hy[r, c]
    idx = wp.vec4i(1, 2, 4, 5)
    for k in range(4):
        E = mat22(real(0.0), real(0.0), real(0.0), real(0.0))
        if k == 0:
            E = mat22(real(1.0), real(0.0), real(0.0), real(0.0))
        elif k == 1:
            E = mat22(real(0.0), real(1.0), real(0.0), real(0.0))
        elif k == 2:
            E = mat22(real(0.0), real(0.0), real(1.0), real(0.0))
        else:
            E = mat22(real(0.0), real(0.0), real(0.0), real(1.0))
        col = nh_d2PsidF2_dir(A, E, Mu, La)
        cv = vec4(col[0, 0], col[0, 1], col[1, 0], col[1, 1])
        for r in range(4):
            H[idx[r], idx[k]] = H[idx[r], idx[k]] + h2 * cv[r]
    a = A[0, 0] + A[1, 1]
    b = A[1, 0] - A[0, 1]
    n = wp.sqrt(a * a + b * b)
    if n > real(1.0e-12):
        t = vec4(-b, -a, a, -b) * (INV_SQRT2 / n)
        Tm = mat22(t[0], t[1], t[2], t[3])
        lam_t = wp.ddot(Tm, nh_d2PsidF2_dir(A, Tm, Mu, La))
        if lam_t < real(0.0):
            for r in range(4):
                for c in range(4):
                    H[idx[r], idx[c]] = H[idx[r], idx[c]] - h2 * lam_t * t[r] * t[c]
    return H


@wp.func
def chol6(H: mat66) -> mat66:
    """Lower Cholesky factor of a 6 x 6. If H turns out not to be positive definite (large
    strain can make the elastic block indefinite along directions other than the twist) the
    factor of its diagonal is returned instead — always positive, because of the mass."""
    L = mat66()
    ok = int(1)
    for j in range(6):
        s = H[j, j]
        for k in range(j):
            s = s - L[j, k] * L[j, k]
        if s <= real(1.0e-9) * wp.abs(H[j, j]) or s <= real(0.0):
            ok = 0
        if ok != 0:
            d = wp.sqrt(s)
            L[j, j] = d
            for i in range(j + 1, 6):
                v = H[i, j]
                for k in range(j):
                    v = v - L[i, k] * L[j, k]
                L[i, j] = v / d
    if ok == 0:
        L = mat66()
        for j in range(6):
            L[j, j] = wp.sqrt(wp.max(H[j, j], real(1.0e-30)))
    return L


@wp.func
def chol6_solve(L: mat66, b: vec6) -> vec6:
    y = vec6()
    for i in range(6):
        s = b[i]
        for k in range(i):
            s = s - L[i, k] * y[k]
        y[i] = s / L[i, i]
    x = vec6()
    for ii in range(6):
        i = 5 - ii
        s = y[i]
        for k in range(i + 1, 6):
            s = s - L[k, i] * x[k]
        x[i] = s / L[i, i]
    return x


@wp.func
def _A_of(q: vec6) -> mat22:
    return mat22(q[1], q[2], q[4], q[5])


# ====================================================================== affine bodies
@wp.kernel
def affine_gather_kernel(node: wp.array(dtype=wp.int32), grp: wp.array(dtype=wp.int32),
                         w: wp.array(dtype=vec3), grad: wp.array(dtype=vec2),
                         diagH: wp.array(dtype=vec2), gq: wp.array(dtype=vec6),
                         Hx: wp.array(dtype=mat33), Hy: wp.array(dtype=mat33)):
    """J^T grad and J^T D J of each rigid region, from its nodes. The nodal gradient is
    then ZEROED: a slaved node is not an unknown of the nodal iteration, and with a zero
    gradient (and so a zero previous gradient) it drops out of every nodal reduction."""
    k = wp.tid()
    i = node[k]
    b = grp[k]
    ww = w[k]
    g = grad[i]
    d = diagH[i]
    wp.atomic_add(gq, b, vec6(g[0] * ww[0], g[0] * ww[1], g[0] * ww[2],
                              g[1] * ww[0], g[1] * ww[1], g[1] * ww[2]))
    o = wp.outer(ww, ww)
    wp.atomic_add(Hx, b, d[0] * o)
    wp.atomic_add(Hy, b, d[1] * o)
    grad[i] = vec2(real(0.0), real(0.0))


@wp.kernel
def affine_reduce_kernel(q: wp.array(dtype=vec6), Mu: wp.array(dtype=real),
                         La: wp.array(dtype=real), env: wp.array(dtype=wp.int32), h2: real,
                         Hx: wp.array(dtype=mat33), Hy: wp.array(dtype=mat33),
                         gq: wp.array(dtype=vec6), gq_prev: wp.array(dtype=vec6),
                         pq: wp.array(dtype=vec6), zq: wp.array(dtype=vec6),
                         gPg: wp.array(dtype=f64), g_p: wp.array(dtype=f64),
                         g_Py: wp.array(dtype=f64), y_p: wp.array(dtype=f64),
                         y_Py: wp.array(dtype=f64)):
    """Finish the reduced gradient with the region's own elastic force, apply the 6 x 6
    block preconditioner, and add this body's share of the five Dai-Kou dot products."""
    b = wp.tid()
    e = env[b]
    A = _A_of(q[b])
    P = nh_dPsidF(A, Mu[b], La[b])
    g = gq[b] + h2 * vec6(real(0.0), P[0, 0], P[0, 1], real(0.0), P[1, 0], P[1, 1])
    L = chol6(reduced_hessian(A, Mu[b], La[b], h2, Hx[b], Hy[b]))
    y = g - gq_prev[b]
    z = chol6_solve(L, g)
    zy = chol6_solve(L, y)
    pp = pq[b]
    wp.atomic_add(gPg, e, f64(wp.dot(g, z)))
    wp.atomic_add(g_p, e, f64(wp.dot(g, pp)))
    wp.atomic_add(g_Py, e, f64(wp.dot(g, zy)))
    wp.atomic_add(y_p, e, f64(wp.dot(y, pp)))
    wp.atomic_add(y_Py, e, f64(wp.dot(y, zy)))
    gq[b] = g
    zq[b] = z


@wp.kernel
def affine_direction_kernel(zq: wp.array(dtype=vec6), env: wp.array(dtype=wp.int32),
                            beta: wp.array(dtype=f64), pq: wp.array(dtype=vec6)):
    b = wp.tid()
    pq[b] = -zq[b] + real(beta[env[b]]) * pq[b]


@wp.kernel
def affine_expand_kernel(node: wp.array(dtype=wp.int32), grp: wp.array(dtype=wp.int32),
                         w: wp.array(dtype=vec3), v6: wp.array(dtype=vec6),
                         out: wp.array(dtype=vec2)):
    """out_i = J_i v — used for the search direction (v = p_q) and the positions (v = q)."""
    k = wp.tid()
    ww = w[k]
    v = v6[grp[k]]
    out[node[k]] = vec2(ww[0] * v[0] + ww[1] * v[1] + ww[2] * v[2],
                        ww[0] * v[3] + ww[1] * v[4] + ww[2] * v[5])


@wp.kernel
def affine_pHp_kernel(q: wp.array(dtype=vec6), pq: wp.array(dtype=vec6),
                      Mu: wp.array(dtype=real), La: wp.array(dtype=real),
                      env: wp.array(dtype=wp.int32), h2: real, pHp: wp.array(dtype=f64)):
    b = wp.tid()
    dA = _A_of(pq[b])
    val = wp.ddot(dA, nh_d2PsidF2_dir(_A_of(q[b]), dA, Mu[b], La[b]))
    wp.atomic_add(pHp, env[b], f64(h2 * wp.max(val, real(0.0))))


@wp.kernel
def affine_energy_kernel(q: wp.array(dtype=vec6), Mu: wp.array(dtype=real),
                         La: wp.array(dtype=real), env: wp.array(dtype=wp.int32), h2: real,
                         E: wp.array(dtype=f64)):
    b = wp.tid()
    A = _A_of(q[b])
    logJ = wp.log(wp.max(wp.determinant(A), LOG_EPS))
    psi = real(0.5) * Mu[b] * (wp.ddot(A, A) - real(2.0)) - Mu[b] * logJ \
        + real(0.5) * La[b] * logJ * logJ
    wp.atomic_add(E, env[b], f64(h2 * psi))


@wp.kernel
def affine_advance_kernel(q: wp.array(dtype=vec6), pq: wp.array(dtype=vec6),
                          env: wp.array(dtype=wp.int32), alpha: wp.array(dtype=f64),
                          active: wp.array(dtype=wp.int32)):
    b = wp.tid()
    e = env[b]
    if active[e] == 0:
        return
    q[b] = q[b] + real(alpha[e]) * pq[b]


@wp.kernel
def affine_trial_kernel(q: wp.array(dtype=vec6), q0: wp.array(dtype=vec6),
                        pq: wp.array(dtype=vec6), env: wp.array(dtype=wp.int32),
                        alpha: wp.array(dtype=f64), active: wp.array(dtype=wp.int32)):
    """q = q0 + alpha p_q — the line-search trial, mirroring set_x_from_x0_kernel."""
    b = wp.tid()
    e = env[b]
    if active[e] == 0:
        return
    q[b] = q0[b] + real(alpha[e]) * pq[b]


class AffineBodies:
    """The rigid regions of a scene, as stiff affine bodies in reduced coordinates.

    groups: list of dict(elems=global element ids). A region's nodes are the nodes of its
    elements; its stiffness is whatever moduli the mesh holds for those elements, summed
    over their area, and the solver's own copies are zeroed so the nodal elastic kernels
    leave the region to the closed form here."""
    in_adjoint = False
    persistent = True           # not an actuator: survives the integrator's term rebuild

    def __init__(self, s, groups):
        me = s.me
        tri = np.asarray(me.tri)
        mass = np.asarray(me.mass, np.float64)
        X = np.asarray(me.rest_pos, np.float64)
        area = np.asarray(me.rest_area, np.float64)
        node, grp, w, c0, Mu, La, env = [], [], [], [], [], [], []
        owner = np.full(X.shape[0], -1)
        mu_s, lam_s = s.mu.numpy(), s.lam.numpy()
        free = s.free.numpy()
        for b, g in enumerate(groups):
            el = np.asarray(g["elems"], np.int64)
            nd = np.unique(tri[el].reshape(-1))
            if (owner[nd] >= 0).any():
                raise ValueError("two rigid regions share a node; merge them or leave a gap of "
                                 "at least one soft element between them")
            if (free[nd] == 0).any():
                raise ValueError("a rigid region contains kinematically constrained nodes. A "
                                 "body whose motion is PRESCRIBED needs no rigid region: "
                                 "constrain all its nodes and drive them with "
                                 "model.rigid_motion(...)")
            owner[nd] = b
            c = (mass[nd, None] * X[nd]).sum(0) / mass[nd].sum()
            node += list(nd); grp += [b] * len(nd)
            w.append(np.c_[np.ones(len(nd)), X[nd] - c]); c0.append(c)
            Mu.append(float((area[el] * np.asarray(me.mu, np.float64)[el]).sum()))
            La.append(float((area[el] * np.asarray(me.lam, np.float64)[el]).sum()))
            env.append(int(np.asarray(me.tri2env)[el[0]]))
            mu_s[el] = 0.0; lam_s[el] = 0.0
        s.mu.assign(mu_s); s.lam.assign(lam_s)

        d = s.dev
        ai = lambda v: wp.array(np.asarray(v, np.int32), dtype=wp.int32, device=d)
        ar = lambda v, t: wp.array(np.asarray(v, np_real), dtype=t, device=d)
        self.n, self.n_nodes = len(groups), len(node)
        self.node_np, self.grp_np = np.asarray(node, np.int64), np.asarray(grp, np.int64)
        self.w_np = np.concatenate(w, 0)
        self.first = np.concatenate([[0], np.cumsum([len(a) for a in w])])    # region b: rows first[b]:first[b+1]
        self.node, self.grp, self.w = ai(node), ai(grp), ar(self.w_np, vec3)
        self.Mu, self.La, self.env = ar(Mu, real), ar(La, real), ai(env)
        self.c0 = np.asarray(c0, np.float64)
        z6 = lambda: wp.zeros(self.n, dtype=vec6, device=d)
        self.q, self.q0, self.pq, self.gq, self.gq_prev, self.zq = z6(), z6(), z6(), z6(), z6(), z6()
        self.Hx = wp.zeros(self.n, dtype=mat33, device=d)
        self.Hy = wp.zeros(self.n, dtype=mat33, device=d)
        self.slaved = owner >= 0
        self.sync_from_x(s)

    # ---- state ----
    def sync_from_x(self, s, regions=None):
        """Set q from the current node positions (least squares; exact when they are an
        affine image of the rest shape, which is all a rigid region can be). regions: only
        these (the others keep the q they have)."""
        x = s.x.numpy().astype(np.float64)
        q = np.zeros((self.n, 6)) if regions is None else self.q.numpy().astype(np.float64)
        for b in (range(self.n) if regions is None else regions):
            k = slice(self.first[b], self.first[b + 1])
            W = self.w_np[k]
            sol = np.linalg.lstsq(W, x[self.node_np[k]], rcond=None)[0]      # (3, 2)
            q[b, :3], q[b, 3:] = sol[:, 0], sol[:, 1]
        self.q.assign(q.astype(np_real))
        self.pq.zero_(); self.gq.zero_(); self.gq_prev.zero_()
        self._expand(s, self.q, s.x)

    def state(self):
        """(t, A) per region: t (n, 2) is where the region's rest centroid now is, A (n, 2, 2)
        its rotation-and-strain."""
        q = self.q.numpy().astype(np.float64)
        return q[:, [0, 3]], np.stack([q[:, 1:3], q[:, 4:6]], 1)

    def _expand(self, s, v6, out):
        wp.launch(affine_expand_kernel, dim=self.n_nodes,
                  inputs=[self.node, self.grp, self.w, v6, out], device=s.dev)

    # ---- solver hooks (see PNCG2DSolver._iterate) ----
    def begin_step(self, s): pass
    def after_elastic(self, s): pass

    def gather(self, s):
        wp.copy(self.gq_prev, self.gq)
        self.gq.zero_(); self.Hx.zero_(); self.Hy.zero_()
        wp.launch(affine_gather_kernel, dim=self.n_nodes,
                  inputs=[self.node, self.grp, self.w, s.grad, s.diagH, self.gq, self.Hx, self.Hy],
                  device=s.dev)

    def reduce(self, s):
        wp.launch(affine_reduce_kernel, dim=self.n,
                  inputs=[self.q, self.Mu, self.La, self.env, s.h2, self.Hx, self.Hy, self.gq,
                          self.gq_prev, self.pq, self.zq, s.gPg, s.g_p, s.g_Py, s.y_p, s.y_Py],
                  device=s.dev)

    def direction(self, s):
        wp.launch(affine_direction_kernel, dim=self.n,
                  inputs=[self.zq, self.env, s.beta, self.pq], device=s.dev)
        self._expand(s, self.pq, s.p)

    def advance(self, s):
        wp.launch(affine_advance_kernel, dim=self.n,
                  inputs=[self.q, self.pq, self.env, s.alpha, s.active], device=s.dev)
        self._expand(s, self.q, s.x)

    def save(self, s):
        wp.copy(self.q0, self.q)

    def trial(self, s):
        wp.launch(affine_trial_kernel, dim=self.n,
                  inputs=[self.q, self.q0, self.pq, self.env, s.alpha, s.active], device=s.dev)
        self._expand(s, self.q, s.x)

    # ---- energy-term interface (the region's own elastic energy) ----
    def grad_diag(self, s): pass            # added in reduce(), in reduced coordinates

    def pHp(self, s):
        wp.launch(affine_pHp_kernel, dim=self.n,
                  inputs=[self.q, self.pq, self.Mu, self.La, self.env, s.h2, s.pHp], device=s.dev)

    def energy(self, s, buf):
        wp.launch(affine_energy_kernel, dim=self.n,
                  inputs=[self.q, self.Mu, self.La, self.env, s.h2, buf], device=s.dev)


# ====================================================================== rigid-mode coarse space
@wp.kernel
def centroid_kernel(node: wp.array(dtype=wp.int32), grp: wp.array(dtype=wp.int32),
                    x: wp.array(dtype=vec2), mass: wp.array(dtype=real),
                    acc: wp.array(dtype=vec3)):
    k = wp.tid()
    i = node[k]
    m = mass[i]
    wp.atomic_add(acc, grp[k], vec3(m * x[i][0], m * x[i][1], m))


@wp.func
def _arm(x: vec2, a: vec3) -> vec2:
    """Position relative to a piece's centroid, from its (sum m x, sum m y, sum m)."""
    return x - vec2(a[0] / a[2], a[1] / a[2])


@wp.kernel
def rigid_gather_kernel(node: wp.array(dtype=wp.int32), grp: wp.array(dtype=wp.int32),
                        x: wp.array(dtype=vec2), acc: wp.array(dtype=vec3),
                        grad: wp.array(dtype=vec2),
                        grad_prev: wp.array(dtype=vec2), diagH: wp.array(dtype=vec2),
                        diag_el: wp.array(dtype=vec2), mass: wp.array(dtype=real),
                        gq: wp.array(dtype=vec3), gq_prev: wp.array(dtype=vec3),
                        Hc: wp.array(dtype=mat33)):
    """B^T g, B^T g_prev and B^T D' B for the modes B_i = [(1, 0), (0, 1), (-r_y, r_x)],
    r_i = x_i - centroid AT THE CURRENT ITERATE. That is what makes the third mode an exact
    rigid rotation of the body as it now is: internal forces have no moment about current
    positions, so B^T of the elastic gradient vanishes identically and the mode costs no
    elastic energy. (Frozen at the start of the step it is neither, once the body has
    deformed within the step — the elastic forces then show up as a spurious torque,
    divided by an inertia-only curvature.)

    D' = mass + everything added to the diagonal AFTER the elastic kernel (contact,
    friction, actuation); the elastic diagonal is left out, because a rigid motion is
    exactly what elasticity does not resist."""
    k = wp.tid()
    i = node[k]
    b = grp[k]
    rr = _arm(x[i], acc[b])
    g = grad[i]
    gp = grad_prev[i]
    m = mass[i]
    d = diagH[i] - diag_el[i]
    dx = wp.max(d[0], real(0.0)) + m
    dy = wp.max(d[1], real(0.0)) + m
    wp.atomic_add(gq, b, vec3(g[0], g[1], rr[0] * g[1] - rr[1] * g[0]))
    wp.atomic_add(gq_prev, b, vec3(gp[0], gp[1], rr[0] * gp[1] - rr[1] * gp[0]))
    wp.atomic_add(Hc, b, mat33(dx, real(0.0), -dx * rr[1],
                               real(0.0), dy, dy * rr[0],
                               -dx * rr[1], dy * rr[0], dx * rr[1] * rr[1] + dy * rr[0] * rr[0]))


@wp.kernel
def rigid_reduce_kernel(env: wp.array(dtype=wp.int32), Hc: wp.array(dtype=mat33),
                        gq: wp.array(dtype=vec3), gq_prev: wp.array(dtype=vec3),
                        zq: wp.array(dtype=vec3), gPg: wp.array(dtype=f64),
                        g_Py: wp.array(dtype=f64), y_Py: wp.array(dtype=f64)):
    """The coarse part of P g and P y, and its share of the three dot products that
    involve P. (g.p and y.p are plain nodal sums and are already complete.)"""
    b = wp.tid()
    e = env[b]
    Hi = wp.inverse(Hc[b])
    g = gq[b]
    y = g - gq_prev[b]
    z = Hi * g
    zy = Hi * y
    wp.atomic_add(gPg, e, f64(wp.dot(g, z)))
    wp.atomic_add(g_Py, e, f64(wp.dot(g, zy)))
    wp.atomic_add(y_Py, e, f64(wp.dot(y, zy)))
    zq[b] = z


@wp.kernel
def rigid_direction_kernel(node: wp.array(dtype=wp.int32), grp: wp.array(dtype=wp.int32),
                           x: wp.array(dtype=vec2), acc: wp.array(dtype=vec3),
                           zq: wp.array(dtype=vec3), p: wp.array(dtype=vec2)):
    """p -= B z: the coarse correction, on top of the nodal -D^-1 g + beta p."""
    k = wp.tid()
    b = grp[k]
    z = zq[b]
    i = node[k]
    rr = _arm(x[i], acc[b])
    p[i] = p[i] - vec2(z[0] - z[2] * rr[1], z[1] + z[2] * rr[0])


class RigidModes:
    """Rigid-mode coarse space for every connected piece of soft material that has no
    kinematically constrained node and no rigid region (a rigid region already has its own
    body-level unknowns). Returns None from build() when the scene has no such piece.

    NOT DONE: a free piece that CONTAINS a rigid region is skipped whole, so its soft
    parts follow the region at wave speed instead of at once. To include it, the three
    modes have to be expressed on the region's reduced unknowns as well: a translation z_t
    and rotation z_r about the piece's centroid c act on a region (t, A) as
        dt = z_t + z_r J (t - c),   dA = z_r J A,        J = [[0, -1], [1, 0]]
    their gather needs the region's reduced gradient (force = (gq[0], gq[3]), moment from
    gq and A), and the correction must be ADDED to AffineBodies.pq before it is expanded —
    the dot products assume the direction that is actually taken."""

    @classmethod
    def build(cls, s, skip_nodes=None):
        me = s.me
        N = s.N
        tri = np.asarray(me.tri)
        a = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
        b = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
        ncomp, lab = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(N, N)),
                                          directed=False)
        held = np.zeros(ncomp, bool)
        fr = s.free.numpy()
        held[np.unique(lab[(fr == 0).any(1)])] = True
        if skip_nodes is not None and np.any(skip_nodes):
            held[np.unique(lab[np.asarray(skip_nodes, bool)])] = True
        size = np.bincount(lab, minlength=ncomp)
        keep = np.where(~held & (size >= 3))[0]
        if len(keep) == 0:
            return None
        newid = np.full(ncomp, -1); newid[keep] = np.arange(len(keep))
        nodes = np.where(newid[lab] >= 0)[0]
        return cls(s, nodes, newid[lab[nodes]], len(keep))

    def __init__(self, s, nodes, grp, n):
        d = s.dev
        ai = lambda v: wp.array(np.asarray(v, np.int32), dtype=wp.int32, device=d)
        self.n, self.n_nodes = int(n), len(nodes)
        self.node, self.grp = ai(nodes), ai(grp)
        first = np.zeros(n, np.int64); first[np.asarray(grp)[::-1]] = np.asarray(nodes)[::-1]
        self.env = ai(np.asarray(s.me.node2env)[first])
        self.acc = wp.zeros(self.n, dtype=vec3, device=d)      # (sum m x, sum m y, sum m)
        self.gq = wp.zeros(self.n, dtype=vec3, device=d)
        self.gq_prev = wp.zeros(self.n, dtype=vec3, device=d)
        self.zq = wp.zeros(self.n, dtype=vec3, device=d)
        self.Hc = wp.zeros(self.n, dtype=mat33, device=d)
        self.diag_el = wp.zeros(s.N, dtype=vec2, device=d)

    def begin_step(self, s): pass

    def after_elastic(self, s):
        wp.copy(self.diag_el, s.diagH)

    def gather(self, s):
        self.acc.zero_(); self.gq.zero_(); self.gq_prev.zero_(); self.Hc.zero_()
        wp.launch(centroid_kernel, dim=self.n_nodes,
                  inputs=[self.node, self.grp, s.x, s.mass, self.acc], device=s.dev)
        wp.launch(rigid_gather_kernel, dim=self.n_nodes,
                  inputs=[self.node, self.grp, s.x, self.acc, s.grad, s.grad_prev, s.diagH,
                          self.diag_el, s.mass, self.gq, self.gq_prev, self.Hc], device=s.dev)

    def reduce(self, s):
        wp.launch(rigid_reduce_kernel, dim=self.n,
                  inputs=[self.env, self.Hc, self.gq, self.gq_prev, self.zq,
                          s.gPg, s.g_Py, s.y_Py], device=s.dev)

    def direction(self, s):
        wp.launch(rigid_direction_kernel, dim=self.n_nodes,
                  inputs=[self.node, self.grp, s.x, self.acc, self.zq, s.p], device=s.dev)

    def advance(self, s): pass
    def save(self, s): pass
    def trial(self, s): pass
