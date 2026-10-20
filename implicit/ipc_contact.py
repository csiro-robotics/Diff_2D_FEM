"""IPC contact between bodies (and of a body with itself): model.body_contact = "ipc".

The same incremental-potential contact the colliders get (kernels.py, Li et al. 2020),
between every pair of bodies — soft against soft, soft against rigid, rigid against rigid —
and between distant parts of one body. In 2D the contact primitives are a boundary NODE
and a boundary EDGE:

    P(x) = kappa * sum over (node p, edge ab)  b(d(p, ab))   -   vertex corrections
    d(p, ab) = distance from p to the SEGMENT ab,   b = the log barrier of core/math.py

so the barrier grows without bound as a node approaches a surface and no step may cross
one. Three things make that work.

VERTEX CORRECTIONS. Summed plainly over the edges, a node near a mesh vertex is counted
once by each edge that meets there: sliding along a flat face, the energy would rise at
every vertex, and the surface would grip like a ratchet with no friction at all. So each
vertex v that m edges share is subtracted (m - 1) times,

    P = kappa [ sum_edges b(d(p, ab))  -  sum_vertices (m_v - 1) b(|p - v|) ]

(inclusion-exclusion, as in Li et al. 2023, "Convergent IPC"). On a flat face this is
exactly b(distance to the face); at a convex corner it is b(distance to the corner). The
correction is spread over the edges that meet at v, (m_v - 1) / m_v each, so one pass over
the edges builds everything and the terms cancel exactly, not just to rounding.

STEP LIMIT. Before a step along the search direction is taken, every node-edge pair within
`reach` is advanced by additive continuous collision detection (ACCD, Li et al. 2021): the
largest step for which the pair keeps a tenth of its distance. That alone keeps surfaces
apart. It does not keep the iteration stable, because this solver takes its step from the
curvature where it stands and not from a line search, and the barrier's curvature is
anything but constant: zero at dhat, growing like the depth u = dhat - d just inside, like
1 / d^2 near the surface. A step that takes a pair from the edge of the barrier to the
middle of it meets fifty times the curvature it was planned with, is thrown back out, and
the solve hunts. So inside the barrier a pair may deepen by no more than twice its depth
(plus 2 % of dhat, which is also all it may enter by) nor by more than half its distance:
the curvature then grows by a factor of about three per iteration at most, and a pair
still gets from first touch to the middle of the barrier in five. Pairs further apart than
`reach` cannot be seen, so the step is also capped at 0.45 reach / (largest node move) —
which only binds for jumps of the order of a pixel in one iteration.

KINEMATIC NODES ARE SWEPT, NOT TELEPORTED. The solver's own boundary condition puts the
constrained nodes at their targets at the start of the step. That would carry a prescribed
body straight through a neighbour, so with this contact the constrained nodes are moved
by the same step limit instead: as far as they may go before the first iteration, and —
if something was in the way — a little further before each iteration that follows, while
the free nodes give way. The sweep closes a gap to a quarter of dhat and no further: the
kinematic nodes are not part of the minimisation, so nothing else would stop them from
tightening a contact faster than the free nodes can answer. (It needs no gentler entry
than that: the free nodes it presses on move AWAY, into falling curvature, where a step
planned from the curvature at hand falls short rather than long.) A prescribed body
therefore pushes with at most the barrier's pressure at that gap — 4.3 kappa dhat per
node, about 0.4 E with the default body_kappa — and a prescribed motion that needs more
(a block squeezed against a wall, two prescribed bodies driven into each other) stops
short. stats()["bc_lag"] reports by how much; raise model.body_kappa for a harder drive.
A free body in front of a prescribed one is pushed along at about 0.2 dhat per iteration
(the step the solver takes out of a barrier): three pixels per step at the default dhat
and 120 iterations. Faster than that, the prescribed body falls behind its target.

Friction is Coulomb on the slip between the node and its closest point on the edge, with
the pair's own normal force and the smoothed law used everywhere else in the package
(body_contact.py); it carries the same vertex corrections, so it cancels the same way.

The bodies must start apart: the barrier is defined for d > 0 only.
"""
from __future__ import annotations
import numpy as np
import warp as wp
from ..core.precision import real, vec2
from ..core.math import cross2, barrier_b, barrier_bp, barrier_bpp, pair_mu

f64 = wp.float64


# BROADPHASE. One hash grid for all environments: a boundary node goes in at
# (x, y, (STRIDE env + 1/2) cell), so each environment has a layer of cells to itself, three
# cells from the next — out of reach of any query. The grid hashes cells modulo its size,
# so the layers must also land in different slots: STRIDE is odd (coprime with the
# power-of-two depth) and the depth is at least STRIDE x the number of environments, up
# to 8192. (64 cells between layers in a grid 128 deep puts every other environment in
# the same slot — 1024 environments ran seven times slower per iteration for it.)
STRIDE = 3


@wp.kernel
def ipc_pack_kernel(x: wp.array(dtype=vec2), boundary: wp.array(dtype=wp.int32),
                    boundary2env: wp.array(dtype=wp.int32), cell: wp.float32,
                    hx3: wp.array(dtype=wp.vec3)):
    b = wp.tid()
    p = x[boundary[b]]
    hx3[b] = wp.vec3(wp.float32(p[0]), wp.float32(p[1]),
                     (wp.float32(boundary2env[b] * STRIDE) + wp.float32(0.5)) * cell)


@wp.func
def _seg_dist(p: vec2, a: vec2, b: vec2) -> real:
    """Distance from p to the segment ab."""
    ab = b - a
    t = wp.clamp(wp.dot(p - a, ab) / wp.max(wp.dot(ab, ab), real(1.0e-30)), real(0.0), real(1.0))
    return wp.length(p - a - t * ab)


@wp.func
def _accd(p: vec2, a: vec2, b: vec2, dp: vec2, da: vec2, db: vec2, g: real, t_c: real) -> real:
    """Additive CCD of a node against an edge: the largest t <= t_c such that moving the
    three points by t * (dp, da, db) keeps the node at least the distance g (no more than
    its present distance) from the edge. Every partial sum is itself safe, so running out
    of passes only makes the answer more cautious."""
    m = (dp + da + db) / real(3.0)                    # a common translation closes no gap
    dp = dp - m
    da = da - m
    db = db - m
    lp = wp.length(dp) + wp.max(wp.length(da), wp.length(db))
    if lp <= real(1.0e-30):
        return t_c
    d = _seg_dist(p, a, b)
    t = real(0.0)
    tl = (d - g) / lp
    for k in range(32):
        tn = t + tl
        if tn >= t_c:
            return t_c
        d = _seg_dist(p + tn * dp, a + tn * da, b + tn * db)
        if k > 0 and d < g:                           # (the first advance lands ON the gap:
            return t                                  #  rounding must not turn it into no step)
        t = tn
        tl = real(0.9) * d / lp
    return t


@wp.func
def _pair(mode: int, coef: real, ip: int, ia: int, ib: int, wa: real, wb: real,
          d: real, n: vec2, ddd: real, e: int,
          x: wp.array(dtype=vec2), xn: wp.array(dtype=vec2), pdir: wp.array(dtype=vec2),
          dhat: real, kappa: real, mu: real, epsv: real, inv_dt: real, h2: real,
          free: wp.array(dtype=vec2), grad: wp.array(dtype=vec2), diagH: wp.array(dtype=vec2),
          pHp: wp.array(dtype=f64), E: wp.array(dtype=f64)):
    """One barrier-and-friction term between node ip and the point (wa, wb) of edge ia-ib,
    at distance d along the unit vector n, weighted by coef. ddd is the second derivative
    of d along the search direction (mode 1 only)."""
    bp = barrier_bp(d, dhat)
    f_n = -kappa * bp                                           # normal force, >= 0
    tn = vec2(-n[1], n[0])
    u = (x[ip] - xn[ip]) - wa * (x[ia] - xn[ia]) - wb * (x[ib] - xn[ib])
    ut = wp.dot(u, tn)                                          # slip since the step began
    vt = wp.abs(ut) * inv_dt
    kt = mu * f_n * inv_dt / wp.max(vt, epsv)
    if mode == 0:
        bpp = barrier_bpp(d, dhat)
        g = h2 * (-f_n * n + mu * f_n * (ut * inv_dt / wp.max(vt, epsv)) * tn)
        hd = h2 * vec2(kappa * bpp * n[0] * n[0] + kt * tn[0] * tn[0],
                       kappa * bpp * n[1] * n[1] + kt * tn[1] * tn[1])
        wp.atomic_add(grad, ip, coef * wp.cw_mul(g, free[ip]))
        wp.atomic_add(diagH, ip, coef * hd)
        wp.atomic_add(grad, ia, (-coef * wa) * wp.cw_mul(g, free[ia]))
        wp.atomic_add(diagH, ia, (coef * wa * wa) * hd)
        wp.atomic_add(grad, ib, (-coef * wb) * wp.cw_mul(g, free[ib]))
        wp.atomic_add(diagH, ib, (coef * wb * wb) * hd)
    elif mode == 1:
        bpp = barrier_bpp(d, dhat)
        dq = pdir[ip] - wa * pdir[ia] - wb * pdir[ib]
        dn = wp.dot(dq, n)
        dt_ = wp.dot(dq, tn)
        val = wp.max(kappa * (bpp * dn * dn + bp * ddd), real(0.0))
        wp.atomic_add(pHp, e, f64(coef * h2 * (val + kt * dt_ * dt_)))
    else:
        asl = epsv / inv_dt
        au = wp.abs(ut)
        f0 = au - real(0.5) * asl
        if au < asl:
            f0 = au * au / (real(2.0) * asl)
        wp.atomic_add(E, e, f64(coef * h2 * (kappa * barrier_b(d, dhat) + mu * f_n * f0)))


@wp.kernel
def ipc_pair_kernel(grid: wp.uint64, x: wp.array(dtype=vec2), xn: wp.array(dtype=vec2),
                    pdir: wp.array(dtype=vec2),
                    ea: wp.array(dtype=wp.int32), eb: wp.array(dtype=wp.int32),
                    ela: wp.array(dtype=wp.int32), elb: wp.array(dtype=wp.int32),
                    ca: wp.array(dtype=real), cb: wp.array(dtype=real),
                    e2env: wp.array(dtype=wp.int32),
                    boundary: wp.array(dtype=wp.int32), boundary2env: wp.array(dtype=wp.int32),
                    boundary_start: wp.array(dtype=wp.int32),
                    boundary_per_env: wp.array(dtype=wp.int32),
                    excl_flat: wp.array(dtype=wp.uint8), excl_offset: wp.array(dtype=wp.int32),
                    cell: wp.float32, dhat: real, kappa: real, reach: real, gap: real, floor: real,
                    mu: real, mu_node: wp.array(dtype=real), per_body: int,
                    epsv: real, inv_dt: real, h2: real, mode: int,
                    free: wp.array(dtype=vec2),
                    grad: wp.array(dtype=vec2), diagH: wp.array(dtype=vec2),
                    pHp: wp.array(dtype=f64), E: wp.array(dtype=f64),
                    clr: wp.array(dtype=f64), alpha: wp.array(dtype=f64)):
    """Every node near one edge. mode 0: gradient + diagonal; 1: p^T H p, the smallest
    clearance (into clr) and the step limit (into alpha); 2: energy; 3: the clearance and
    the step limit only, along pdir. One kernel, so the passes cannot disagree about which
    pairs are in contact."""
    k = wp.tid()
    a = ea[k]
    b = eb[k]
    e = e2env[k]
    start = boundary_start[e]
    n_be = boundary_per_env[e]
    eoff = excl_offset[e]
    xa = x[a]
    xb = x[b]
    ab = xb - xa
    L2 = wp.max(wp.dot(ab, ab), real(1.0e-30))
    L = wp.sqrt(L2)
    mid = real(0.5) * (xa + xb)
    dlo = real(1.0e-9) * dhat                                  # never take the log of zero
    onset = real(0.02) * dhat                                  # how deep a pair may land on entry
    t_c = real(alpha[e])
    q = wp.hash_grid_query(grid, wp.vec3(wp.float32(mid[0]), wp.float32(mid[1]),
                                         (wp.float32(e * STRIDE) + wp.float32(0.5)) * cell),
                           wp.float32(real(0.5) * L + reach))
    for jb in q:
        pn = boundary[jb]
        jl = jb - start
        ok = boundary2env[jb] == e and pn != a and pn != b
        if ok:
            ok = excl_flat[eoff + jl * n_be + ela[k]] == wp.uint8(0) and \
                excl_flat[eoff + jl * n_be + elb[k]] == wp.uint8(0)
        if ok:
            xp = x[pn]
            w = xp - xa
            tt = wp.dot(w, ab) / L2
            dvec = w - tt * ab                                 # foot inside the edge
            if tt <= real(0.0):
                dvec = w                                       # closest to a
            elif tt >= real(1.0):
                dvec = xp - xb                                 # closest to b
            d = wp.length(dvec)
            if d < reach:
                if mode == 1 or mode == 3:
                    wp.atomic_min(clr, e, f64(d))
                    # the distance this pair must keep (see STEP LIMIT above)
                    keep = wp.max(gap * d, dhat - onset)
                    if d < dhat:
                        keep = d - wp.min(real(2.0) * (dhat - d) + onset, real(0.5) * d)
                    if mode == 3:                              # a sweep of the kinematic nodes
                        keep = wp.min(d, wp.max(gap * d, floor))
                    wp.atomic_min(alpha, e, f64(_accd(xp, xa, xb, pdir[pn], pdir[a], pdir[b],
                                                      keep, t_c)))
                if mode != 3:
                    mu_p = pair_mu(mu, per_body, mu_node, pn, a)      # this pair's friction
                    if tt > real(0.0) and tt < real(1.0) and d < dhat:
                        d = wp.max(d, dlo)
                        ddd = real(0.0)
                        if mode == 1:                          # d'' of the node-line distance
                            de = pdir[b] - pdir[a]
                            dw = pdir[pn] - pdir[a]
                            c0 = cross2(ab, w)
                            c1 = cross2(de, w) + cross2(ab, dw)
                            c2 = real(2.0) * cross2(de, dw)
                            L1 = wp.dot(ab, de) / L
                            Lq = (wp.dot(de, de) - L1 * L1) / L
                            s0 = c0 / L
                            s1 = (c1 - s0 * L1) / L
                            ddd = (c2 - real(2.0) * s1 * L1 - s0 * Lq) / L
                            if c0 < real(0.0):
                                ddd = -ddd
                        _pair(mode, real(1.0), pn, a, b, real(1.0) - tt, tt, d, dvec / d, ddd, e,
                              x, xn, pdir, dhat, kappa, mu_p, epsv, inv_dt, h2, free, grad, diagH,
                              pHp, E)
                    # the two ends: + 1 where the end is the closest point, - its share of
                    # the vertex correction always
                    coef_a = -ca[k]
                    if tt <= real(0.0):
                        coef_a = coef_a + real(1.0)
                    d_a = wp.length(w)
                    if coef_a != real(0.0) and d_a < dhat:
                        d_a = wp.max(d_a, dlo)
                        n_a = w / d_a
                        ddd = real(0.0)
                        if mode == 1:
                            dw = pdir[pn] - pdir[a]
                            dn = wp.dot(dw, n_a)
                            ddd = (wp.dot(dw, dw) - dn * dn) / d_a
                        _pair(mode, coef_a, pn, a, b, real(1.0), real(0.0), d_a, n_a, ddd, e,
                              x, xn, pdir, dhat, kappa, mu_p, epsv, inv_dt, h2, free, grad, diagH,
                              pHp, E)
                    coef_b = -cb[k]
                    if tt >= real(1.0):
                        coef_b = coef_b + real(1.0)
                    wb_ = xp - xb
                    d_b = wp.length(wb_)
                    if coef_b != real(0.0) and d_b < dhat:
                        d_b = wp.max(d_b, dlo)
                        n_b = wb_ / d_b
                        ddd = real(0.0)
                        if mode == 1:
                            dw = pdir[pn] - pdir[b]
                            dn = wp.dot(dw, n_b)
                            ddd = (wp.dot(dw, dw) - dn * dn) / d_b
                        _pair(mode, coef_b, pn, a, b, real(0.0), real(1.0), d_b, n_b, ddd, e,
                              x, xn, pdir, dhat, kappa, mu_p, epsv, inv_dt, h2, free, grad, diagH,
                              pHp, E)


@wp.kernel
def ipc_far_cap_kernel(p_inf: wp.array(dtype=f64), reach: real, alpha: wp.array(dtype=f64)):
    """Pairs further apart than `reach` are not looked at, so nothing may move far enough
    in one go to close that distance: two points approaching each other cover 2 alpha p_inf."""
    e = wp.tid()
    if p_inf[e] > f64(real(0.0)):
        alpha[e] = wp.min(alpha[e], f64(real(0.45) * reach) / p_inf[e])


@wp.kernel
def bc_motion_kernel(x: wp.array(dtype=vec2), bc_idx: wp.array(dtype=wp.int32),
                     bc_target: wp.array(dtype=vec2), node2env: wp.array(dtype=wp.int32),
                     q: wp.array(dtype=vec2), q_inf: wp.array(dtype=f64)):
    """What is left of each kinematic node's way to its target."""
    b = wp.tid()
    i = bc_idx[b]
    d = bc_target[b] - x[i]
    q[i] = d
    wp.atomic_max(q_inf, node2env[i], f64(wp.length(d)))


@wp.kernel
def bc_advance_kernel(x: wp.array(dtype=vec2), xhat: wp.array(dtype=vec2),
                      bc_idx: wp.array(dtype=wp.int32), bc_target: wp.array(dtype=vec2),
                      node2env: wp.array(dtype=wp.int32), beta: wp.array(dtype=f64)):
    b = wp.tid()
    i = bc_idx[b]
    t = real(beta[node2env[i]])
    if t >= real(1.0):
        x[i] = bc_target[b]
    else:
        x[i] = x[i] + t * (bc_target[b] - x[i])
    xhat[i] = x[i]


@wp.kernel
def bc_wake_kernel(q_inf: wp.array(dtype=f64), active: wp.array(dtype=wp.int32)):
    """An environment whose kinematic nodes have just moved is not converged, whatever the
    last iteration concluded."""
    e = wp.tid()
    if q_inf[e] > f64(real(0.0)):
        active[e] = 1


class IPCContact:
    """IPC barrier contact with friction between bodies — see the notes above.

    dhat, kappa  the barrier's reach and stiffness (model.dhat, model.body_kappa)
    reach        pairs closer than this get the exact step limit        (default 2 pixels)
    gap          fraction of its distance a far pair keeps in one iteration  (0.1)"""
    in_adjoint = False
    persistent = True
    bc_floor = 0.25         # kinematic nodes close a gap to no less than this fraction of dhat

    def __init__(self, s, mu, dhat, kappa, reach, gap=0.1, mu_node=None):
        me = s.me
        ai = lambda v: wp.array(np.asarray(v, np.int32), dtype=wp.int32, device=s.dev)
        ar = lambda v: wp.array(np.asarray(v, np.float64), dtype=real, device=s.dev)
        edges = np.asarray(me.boundary_edges, np.int64)
        env = np.asarray(me.node2env)[edges[:, 0]]
        self.bstart = ai(np.asarray(me.boundary_starts[:-1]))
        self.bpe = ai(np.asarray(me.boundary_per_env))
        self.excl = wp.array(np.asarray(me.excl_flat, np.uint8), dtype=wp.uint8, device=s.dev)
        self.excl_off = ai(np.asarray(me.excl_offset))
        # position of each end among its environment's boundary nodes (rows of the exclusion matrix)
        pos = np.full(s.N, -1, np.int64)
        pos[np.asarray(me.boundary_nodes)] = np.arange(s.n_b) - \
            np.asarray(me.boundary_starts)[np.asarray(me.boundary2env)]
        self.ea, self.eb, self.e2env = ai(edges[:, 0]), ai(edges[:, 1]), ai(env)
        self.ela, self.elb = ai(pos[edges[:, 0]]), ai(pos[edges[:, 1]])
        # each edge's share of the correction at its two ends: (m - 1) / m for a vertex of m edges
        m = np.bincount(edges.reshape(-1), minlength=s.N).astype(np.float64)
        share = (m - 1.0) / np.maximum(m, 1.0)
        self.ca, self.cb = ar(share[edges[:, 0]]), ar(share[edges[:, 1]])
        self.n = len(edges)
        self.mu, self.dhat, self.kappa = float(mu), float(dhat), float(kappa)
        # per-body friction (mu= on add_*): each node's body coefficient; pairs take the
        # harmonic mean of theirs. None: every pair uses the scalar mu, exactly as before.
        self.per_body = 0 if mu_node is None else 1
        self.mu_node = ar(np.zeros(1) if mu_node is None else mu_node)
        self.reach, self.gap = float(reach), float(gap)
        nz = int(min(8192, max(4, 1 << int(np.ceil(np.log2(STRIDE * s.B))))))    # see BROADPHASE
        nxy = int(min(128, max(16, 1 << int(0.5 * np.log2((1 << 21) / nz)))))
        self.grid = wp.HashGrid(nxy, nxy, nz, device=s.dev)
        self.hx3 = wp.zeros(s.n_b, dtype=wp.vec3, device=s.dev)
        self.clr = wp.zeros(s.B, dtype=f64, device=s.dev)        # scratch: clearance of a sweep
        self.q = None                                            # kinematic sweep, built on first use
        self.pending = False                                     # kinematic nodes still short of their targets

    def _grid(self, s):
        wp.launch(ipc_pack_kernel, dim=s.n_b,
                  inputs=[s.x, s.bnd, s.b2e, self.reach, self.hx3], device=s.dev)
        self.grid.build(self.hx3, self.reach)

    def _launch(self, s, mode, pdir, E, clr, alpha, grad=None, diagH=None, free=None, friction=True,
                h2=None):
        wp.launch(ipc_pair_kernel, dim=self.n,
                  inputs=[self.grid.id, s.x, s.xn, pdir, self.ea, self.eb, self.ela, self.elb,
                          self.ca, self.cb, self.e2env, s.bnd, s.b2e, self.bstart, self.bpe,
                          self.excl, self.excl_off, self.reach, self.dhat, self.kappa,
                          self.reach, self.gap, (self.bc_floor * self.dhat if mode == 3 else 0.0),
                          (self.mu if friction else 0.0), self.mu_node,
                          (self.per_body if friction else 0), s.epsv, s.inv_dt,
                          (s.h2 if h2 is None else h2), mode,
                          (s.free if free is None else free), (s.grad if grad is None else grad),
                          (s.diagH if diagH is None else diagH), s.pHp, E, clr, alpha], device=s.dev)

    def force(self, s, grad, diagH, free, friction=True):
        """Add MINUS the contact force on every node to `grad` (solver.contact_force): the
        gradient pass with h^2 = 1, nothing masked, friction optional."""
        self._grid(s)
        self._launch(s, 0, s.p, s.Ener, s.clr_min, s.alpha, grad=grad, diagH=diagH, free=free,
                     friction=friction, h2=1.0)

    # ---- energy-term interface (see PNCG2DSolver._iterate) ----
    def grad_diag(self, s):
        self._grid(s)                                            # x is fixed until the next call
        self._launch(s, 0, s.p, s.Ener, s.clr_min, s.alpha)

    def pHp(self, s):
        # The clearance goes to self.clr, NOT to the solver's clr_min: that one switches on
        # a cap of dhat / 2 on every node's ABSOLUTE move per iteration, which is right
        # next to a fixed collider and wrong between two bodies travelling together — a
        # gripper carrying a block would be held to (iterations x dhat / 2) per step.
        # The step limit here is on each pair's RELATIVE motion and needs no such cap.
        self.clr.fill_(1.0e30)
        self._launch(s, 1, s.p, s.Ener, self.clr, s.alpha)
        wp.launch(ipc_far_cap_kernel, dim=s.B, inputs=[s.p_inf, self.reach, s.alpha], device=s.dev)

    def energy(self, s, buf):
        self._launch(s, 2, s.p, buf, s.clr_min, s.alpha)

    # ---- step limit along an arbitrary direction ----
    def ccd(self, s, pdir, p_inf, alpha):
        """Largest step along pdir (per environment, at most 1) that crosses nothing, into
        `alpha`; the smallest clearance met on the way is left in self.clr."""
        self._grid(s)
        alpha.fill_(1.0)
        self.clr.fill_(1.0e30)
        self._launch(s, 3, pdir, s.Ener, self.clr, alpha)
        wp.launch(ipc_far_cap_kernel, dim=s.B, inputs=[p_inf, self.reach, alpha], device=s.dev)

    def min_distance(self, s):
        """Smallest node-edge distance per environment (1e30 where no pair is within reach)."""
        if self.q is None:
            self._alloc(s)
        self.q_inf.zero_()
        self.ccd(s, self.q0, self.q_inf, self.beta)
        return self.clr.numpy().copy()

    def _alloc(self, s):
        self.q = wp.zeros(s.N, dtype=vec2, device=s.dev)         # non-zero on kinematic nodes only
        self.q0 = wp.zeros(s.N, dtype=vec2, device=s.dev)
        self.q_inf = wp.zeros(s.B, dtype=f64, device=s.dev)
        self.beta = wp.zeros(s.B, dtype=f64, device=s.dev)

    # ---- kinematic nodes ----
    def _sweep(self, s):
        """Move the kinematic nodes towards their targets as far as the step limit allows."""
        self.q_inf.zero_()
        wp.launch(bc_motion_kernel, dim=s.n_bc,
                  inputs=[s.x, s.bc_idx, s.bc_target, s.node2env, self.q, self.q_inf], device=s.dev)
        self.ccd(s, self.q, self.q_inf, self.beta)
        wp.launch(bc_advance_kernel, dim=s.n_bc,
                  inputs=[s.x, s.xhat, s.bc_idx, s.bc_target, s.node2env, self.beta], device=s.dev)

    def advance_bc(self, s):
        """Start of a step: in place of the solver's apply_bc."""
        self.pending = False
        if s.n_bc == 0:
            return
        if self.q is None:
            self._alloc(s)
        self._sweep(s)
        self.pending = not bool((self.beta.numpy() >= 1.0).all())

    def before_iteration(self, s, it):
        """Something was in the way: take the kinematic nodes a little further each iteration,
        now that the free nodes have had one more to move aside."""
        if not self.pending:
            return
        self._sweep(s)
        wp.launch(bc_wake_kernel, dim=s.B, inputs=[self.q_inf, s.active], device=s.dev)
        if it % 8 == 7:                                  # one device->host sync, now and then
            self.pending = not bool((self.beta.numpy() >= 1.0).all())
