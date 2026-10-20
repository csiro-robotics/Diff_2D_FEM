"""Driver for the 2D multi-env PNCG-IPC solver — sequences the kernels in kernels.py.

Takes implicit backward-Euler steps, so dt is set by accuracy and not by a CFL limit.

Per timestep (see kernels.py for why the ordering matters):

    init_step        xn = x ; xhat = x + dt*drag*v      <- BEFORE the BC moves
    apply_bc         x[bc] = target(t+dt) ; xhat[bc] = same   (swept instead, with IPC
                                                               contact between bodies)
    reset_step       per-env dE0 = 0, active = 1
    repeat iter_max:
        grad + diagH   (mass, elastic, collider barriers, friction, every term)
        5 per-env Dai-Kou reductions -> finalize_beta -> update_p
        pHp            (the same contributions) + p_inf + clr_min
        alpha caps     (area inversion root, collider CCD, the terms' own) -> alpha_finalize
        update_x       (converged envs are no-ops)
    commit_v         v = (x - xn)/dt

There are no host syncs inside the inner loop except the convergence check every
conv_check iterations: `active` is a device mask and converged envs simply stop moving.
"""
from __future__ import annotations
import math
import numpy as np
import warp as wp
from ..core.precision import real, vec2, vec3, mat22, np_real

from . import kernels as K
from ..core import sdf as SDF                    # arbitrary-object SDF contact

f64 = wp.float64


@wp.kernel
def reset_env_kernel(x: wp.array(dtype=vec2), v: wp.array(dtype=vec2), xn: wp.array(dtype=vec2),
                     rest: wp.array(dtype=vec2), node2env: wp.array(dtype=wp.int32),
                     pick: wp.array(dtype=wp.int32)):
    """Back to rest, at rest, for the nodes of the environments picked."""
    i = wp.tid()
    if pick[node2env[i]] != 0:
        x[i] = rest[i]
        xn[i] = rest[i]
        v[i] = vec2(real(0.0), real(0.0))


@wp.kernel
def zero_vec2_kernel(a: wp.array(dtype=vec2)):
    a[wp.tid()] = vec2(real(0.0), real(0.0))


@wp.kernel
def set_x_from_x0_kernel(x: wp.array(dtype=vec2), x0: wp.array(dtype=vec2),
                         p: wp.array(dtype=vec2), node2env: wp.array(dtype=wp.int32),
                         alpha: wp.array(dtype=f64), active: wp.array(dtype=wp.int32)):
    """x = x0 + alpha*p  (per env, active only) — the line-search trial position."""
    i = wp.tid()
    e = node2env[i]
    if active[e] == 0:
        return
    x[i] = x0[i] + real(alpha[e]) * p[i]


@wp.kernel
def ls_halve_kernel(Ener: wp.array(dtype=f64), Eprev: wp.array(dtype=f64),
                    alpha: wp.array(dtype=f64), active: wp.array(dtype=wp.int32), tol: f64):
    """Backtracking safe-gate: halve alpha for any env whose trial energy rose above the
    pre-step energy (beyond a small tolerance). Guarantees monotone energy decrease, so
    the step can never diverge (blow up a stiff or dense body)."""
    b = wp.tid()
    if active[b] == 0:
        return
    if Ener[b] > Eprev[b] + tol * (wp.abs(Eprev[b]) + f64(real(1.0))):
        alpha[b] = alpha[b] * f64(real(0.5))


@wp.kernel
def cg_guard_kernel(y_p: wp.array(dtype=f64), gPg: wp.array(dtype=f64),
                    pHp: wp.array(dtype=f64), alpha: wp.array(dtype=f64), guard: f64,
                    hold: wp.array(dtype=f64),
                    beta: wp.array(dtype=f64), gTp: wp.array(dtype=f64)):
    """Restart the conjugate direction where the last step did not see the curvature it
    was planned with. Along the last direction the gradient should have changed by
    y.p = alpha p^T H p; Dai-Kou's beta divides by y.p, and where that came out a small
    fraction of the model's (a contact that opened, friction past its kink, a barrier
    entered) beta is large for no good reason and the iteration creeps along one stale
    direction for dozens of iterations. pHp and alpha are still the last iteration's here.

    Not where hold > 0: there the kinematic nodes have just been moved (ipc_contact.py), so
    the gradient changed for a reason that says nothing about the curvature along p."""
    e = wp.tid()
    if hold[e] <= f64(real(0.0)) and y_p[e] < guard * alpha[e] * pHp[e]:
        beta[e] = f64(real(0.0))
        gTp[e] = -gPg[e]


class PNCG2DSolver:
    def __init__(self, mesh, cfg, driven, colliders=(), device="cuda", dt=1.0e-3,
                 iter_max=150, eps=1e-4, tol_x=0.0, conv_check=0, near_factor=4.0, alpha_max=3.0,
                 area_shrink=0.1, ccd_eta=0.9, self_stiffness=0.0, self_radius=0.0,
                 friction=False, line_search=False, n_backtrack=4, ls_tol=1e-7):
        """mesh       the scene's core.mesh.Mesh (every environment, every body)
        cfg        a SimConfig: damping, the barrier's reach and stiffness, friction
        driven     global indices of the kinematically constrained nodes
        colliders  dicts(kind, cx, cy, rx, ry, rref, r, mu), shared by all environments
        The rest is what ImplicitIntegrator passes on; see its docstring."""
        self.cfg, self.dev, self.dt = cfg, device, float(dt)
        self.iter_max, self.eps = int(iter_max), float(eps)
        # tol_x: IPC-style INCREMENT INF-NORM convergence (metres of node motion per iteration).
        # 0.0 disables it, leaving the energy-decrease test alone. In single precision that
        # test is capped by a sqrt(N)*float32 summation floor (~1.6e-5 at N=18k), so a tighter
        # eps never converges there; alpha*p_inf is an atomic MAX and has no such floor.
        self.tol_x = float(tol_x)
        # conv_check: check every N iterations whether ALL envs are done and break out of _iterate.
        # Without it converging early saves nothing — every one of the iter_max iterations is
        # still launched. 0 disables. Costs one device->host sync per check, so keep N >= 8.
        self.conv_check = int(conv_check)
        self.near_factor, self.alpha_max = float(near_factor), float(alpha_max)
        self.area_shrink, self.ccd_eta = float(area_shrink), float(ccd_eta)
        # the "node" contact between bodies (0 = off): stiffness and radius of the node-node
        # penalty. A radius of half a pixel would be inert — boundary edges are a pixel
        # long, so a vertex slips THROUGH an edge while staying more than half a pixel from
        # both its ends. 1.5 pixels keeps it in range of the ends before it can cross.
        self.self_stiffness = float(self_stiffness)
        self.self_radius = float(self_radius)
        self.friction = bool(friction)
        # energy line-search safe-gate: after the model+CCD step is chosen, verify it
        # decreases the total energy; if not, halve it (n_backtrack times). Guarantees
        # the implicit step cannot diverge on a stiff or dense body. Off by default;
        # ~2-3x the cost when on.
        self.line_search = bool(line_search)
        self.n_bt, self.ls_tol = int(n_backtrack), float(ls_tol)
        # extra energy terms (actuation.py): each adds to grad/diagH, pHp and the energy at
        # the same points the built-in terms do. Empty by default.
        self.terms = []
        # body-level modes (affine.py): rigid regions in reduced coordinates, and the
        # rigid-mode coarse space of free soft bodies. Each hooks into the direction
        # update at the points marked `modes` below. Empty by default.
        self.modes = []
        # IPC contact between bodies (ipc_contact.py), when the scene uses it. It is one of
        # `terms`; it is also named here because it sweeps the kinematic nodes to their
        # targets where the plain boundary condition would teleport them.
        self.ipc = None
        # > 0: restart the conjugate direction when the secant curvature along the last
        # step is below this fraction of the model's (cg_guard_kernel). 0: plain Dai-Kou.
        self.cg_guard = 0.0
        me = mesh
        self.me = me
        self.B = int(me.n_envs)
        self.N = me.rest_pos.shape[0]
        self.n_tri = len(me.rest_area)
        self.n_b = len(me.boundary_nodes)

        a32 = lambda v, d: wp.array(np.asarray(v, np_real), dtype=d, device=device)
        ai = lambda v: wp.array(np.asarray(v, np.int32), dtype=wp.int32, device=device)
        self.x = a32(me.rest_pos, vec2)
        self.v = wp.zeros(self.N, dtype=vec2, device=device)
        self.xn = wp.zeros(self.N, dtype=vec2, device=device)
        self.xhat = wp.zeros(self.N, dtype=vec2, device=device)
        self.grad = wp.zeros(self.N, dtype=vec2, device=device)
        self.grad_prev = wp.zeros(self.N, dtype=vec2, device=device)
        self.p = wp.zeros(self.N, dtype=vec2, device=device)
        self.diagH = wp.zeros(self.N, dtype=vec2, device=device)
        self.mass = a32(me.mass, real)
        self.tri = wp.array(np.asarray(me.tri, np.int32), dtype=wp.vec3i, device=device)
        self.Bm = a32(me.D_m_inv, mat22)
        self.area = a32(me.rest_area, real)
        self.mu = a32(me.mu, real)
        self.lam = a32(me.lam, real)
        self.bnd = ai(me.boundary_nodes)
        self.b2e = ai(me.boundary2env)
        self.node2env = ai(me.node2env)
        self.tri2env = ai(me.tri2env)

        free = np.ones((self.N, 2), np_real)
        bc = np.asarray(driven, np.int64)
        free[bc] = 0.0
        self.free = a32(free, vec2)
        self.bc_idx = ai(bc)
        self.bc_target = wp.zeros(len(bc), dtype=vec2, device=device)
        self.n_bc = len(bc)

        z = lambda: wp.zeros(self.B, dtype=f64, device=device)
        self.gPg, self.g_p, self.g_Py = z(), z(), z()
        self.y_p, self.y_Py = z(), z()
        self.beta, self.gTp, self.pHp = z(), z(), z()
        self.alpha, self.p_inf, self.clr_min = z(), z(), z()
        self.dE, self.dE0 = z(), z()
        self.active = wp.zeros(self.B, dtype=wp.int32, device=device)
        self.iters = wp.zeros(self.B, dtype=wp.int32, device=device)
        self.Fcol = wp.zeros(self.B, dtype=vec2, device=device)
        self.Ener = z()
        self.zeroB = z()                                   # per-env zeros (a read-only default)
        self.Els_prev = z()                                # energy before the step (line-search)
        self.x0buf = wp.zeros(self.N, dtype=vec2, device=device)   # pre-step x (line-search)

        self.dhat = float(cfg.ipc_d_hat)
        self.kappa = float(cfg.obstacle_stiffness)
        self.h2 = self.dt * self.dt
        self.inv_dt = 1.0 / self.dt
        self.drag = math.exp(-self.dt * cfg.damping)
        self.epsv = float(cfg.obstacle_friction_epsv)
        # RIGID COLLIDERS. Each is an analytic shape: kind 0 = disc (exact closed-form kernels),
        # kind >= 1 = a signed-distance level set (core/sdf.py). Every contact kernel is launched
        # once per collider; their gradients, curvatures, step caps and energies simply add.
        # `colliders` is a list of dicts(kind, cx, cy, rx, ry, rref, r, mu).
        self.colliders = [dict(c) for c in colliders]
        self._active_colliders = self.colliders
        self.object_kind = -1                       # the SELECTED collider (see _select)
        self.c = vec2(0.0, 0.0); self.radius = 0.0
        self.obj_cx = self.obj_cy = self.obj_rx = self.obj_ry = self.obj_rref = 0.0
        self.mu_fric = float(cfg.obstacle_friction)
        if self.colliders:
            self._select(self.colliders[0])

        # self-collision structure (boundary layout + mesh-adjacency exclusion mask)
        # + a Warp HashGrid broadphase: each boundary node is packed as vec3 with
        # z = env*shift (shift > radius) so environments live in separate z-slabs and
        # never pair across -> O(n_b) instead of O(n_b^2) per iteration.
        if self.self_stiffness > 0.0:
            self.sc_bstart = ai(np.asarray(me.boundary_starts[:-1], np.int32))
            self.sc_bpe = ai(np.asarray(me.boundary_per_env, np.int32))
            self.sc_excl = wp.array(np.asarray(me.excl_flat, np.uint8), dtype=wp.uint8, device=device)
            self.sc_excl_off = ai(np.asarray(me.excl_offset, np.int32))
            self.hx3 = wp.zeros(self.n_b, dtype=wp.vec3, device=device)
            self.sc_grid = wp.HashGrid(128, 128, 128, device=device)
            self.sc_shift = 8.0 * self.self_radius      # z-slab gap between environments

    def set_dt(self, dt):
        """Change the step size (everything derived from it is recomputed)."""
        self.dt = float(dt)
        self.h2 = self.dt * self.dt
        self.inv_dt = 1.0 / self.dt
        self.drag = math.exp(-self.dt * self.cfg.damping)

    def _select(self, col):
        """Make `col` the collider the contact kernels are launched against."""
        self.object_kind = int(col["kind"])
        self.obj_cx, self.obj_cy = float(col["cx"]), float(col["cy"])
        self.obj_rx, self.obj_ry, self.obj_rref = float(col["rx"]), float(col["ry"]), float(col["rref"])
        self.c = vec2(self.obj_cx, self.obj_cy)
        self.radius = float(col.get("r", col["rx"]))          # the disc fast path's radius
        self.mu_fric = float(col["mu"])

    def _build_sc_grid(self):
        """Pack boundary nodes as z-slab-separated vec3 and (re)build the broadphase
        grid at the current x. Cell size = self_radius (query ring covers one cell)."""
        wp.launch(K.pack_boundary_pos_kernel, dim=self.n_b,
                  inputs=[self.x, self.bnd, self.b2e, self.sc_shift, self.hx3], device=self.dev)
        self.sc_grid.build(self.hx3, self.self_radius)

    # ------------------------------------------------------------------ inner solve
    def _iterate(self):
        d = self.dev
        for it in range(self.iter_max):
            if self.ipc is not None:
                self.ipc.before_iteration(self, it)     # kinematic nodes still on their way
            if self.conv_check and it and it % self.conv_check == 0:
                if int(self.active.numpy().sum()) == 0:
                    break
            wp.launch(K.grad_diag_mass_kernel, dim=self.N,
                      inputs=[self.x, self.xhat, self.mass, self.free,
                              self.grad, self.grad_prev, self.diagH], device=d)
            wp.launch(K.grad_diag_elastic_kernel, dim=self.n_tri,
                      inputs=[self.x, self.tri, self.Bm, self.area, self.mu, self.lam,
                              self.h2, 1, self.grad, self.diagH], device=d)
            for mode in self.modes:
                mode.after_elastic(self)
            for _col in self._active_colliders:
                self._select(_col)
                if self.object_kind == 0:
                    wp.launch(K.grad_diag_disc_kernel, dim=self.n_b,
                              inputs=[self.x, self.bnd, self.c, self.radius, self.dhat,
                                      self.kappa, self.h2, 1, self.grad, self.diagH], device=d)
                elif self.object_kind > 0:
                    wp.launch(SDF.grad_diag_sdf_kernel, dim=self.n_b,
                              inputs=[self.x, self.bnd, self.object_kind, self.obj_cx, self.obj_cy,
                                      self.obj_rx, self.obj_ry, self.obj_rref, self.dhat, self.kappa,
                                      self.h2, 1, self.grad, self.diagH], device=d)
            if self.self_stiffness > 0.0:
                self._build_sc_grid()               # rebuild at current x (used by grad + pHp)
                wp.launch(K.grad_diag_self_bp_kernel, dim=self.n_b,
                          inputs=[self.sc_grid.id, self.hx3, self.x, self.bnd, self.b2e,
                                  self.sc_bstart, self.sc_bpe, self.sc_excl, self.sc_excl_off,
                                  self.self_radius, self.self_stiffness, self.h2, 1,
                                  self.grad, self.diagH], device=d)
            if self.friction:
                for _col in self._active_colliders:
                    self._select(_col)
                    if self.object_kind == 0:
                        wp.launch(K.grad_friction_disc_kernel, dim=self.n_b,
                                  inputs=[self.x, self.xn, self.bnd, self.c, self.radius, self.dhat,
                                          self.kappa, self.mu_fric, self.epsv, self.inv_dt, self.h2,
                                          self.free, self.grad, self.diagH], device=d)
                    elif self.object_kind > 0:
                        wp.launch(SDF.grad_friction_sdf_kernel, dim=self.n_b,
                                  inputs=[self.x, self.xn, self.bnd, self.object_kind, self.obj_cx,
                                          self.obj_cy, self.obj_rx, self.obj_ry, self.obj_rref,
                                          self.dhat, self.kappa, self.mu_fric, self.epsv,
                                          self.inv_dt, self.h2, self.free, self.grad, self.diagH],
                                  device=d)
            for term in self.terms:
                term.grad_diag(self)
            for mode in self.modes:                 # reduce the finished gradient to body level
                mode.gather(self)
            wp.launch(K.mask_grad_kernel, dim=self.N,
                      inputs=[self.grad, self.free], device=d)

            for arr in (self.gPg, self.g_p, self.g_Py, self.y_p, self.y_Py):
                arr.zero_()
            wp.launch(K.dk_reduce_kernel, dim=self.N,
                      inputs=[self.grad, self.grad_prev, self.p, self.diagH, self.free,
                              self.node2env, self.gPg, self.g_p, self.g_Py,
                              self.y_p, self.y_Py], device=d)
            for mode in self.modes:                 # body-level share of the same dot products
                mode.reduce(self)
            wp.launch(K.finalize_beta_kernel, dim=self.B,
                      inputs=[self.gPg, self.g_p, self.g_Py, self.y_p, self.y_Py, it,
                              self.beta, self.gTp], device=d)
            if self.cg_guard > 0.0 and it > 0:
                hold = self.ipc.q_inf if self.ipc is not None and self.ipc.pending else self.zeroB
                wp.launch(cg_guard_kernel, dim=self.B,
                          inputs=[self.y_p, self.gPg, self.pHp, self.alpha, self.cg_guard,
                                  hold, self.beta, self.gTp], device=d)
            wp.launch(K.update_p_kernel, dim=self.N,
                      inputs=[self.grad, self.diagH, self.free, self.node2env,
                              self.beta, self.p], device=d)
            for mode in self.modes:                 # body-level part of the direction
                mode.direction(self)

            wp.launch(K.pHp_reset_kernel, dim=self.B,
                      inputs=[self.pHp, self.alpha, self.p_inf, self.clr_min,
                              self.alpha_max], device=d)
            wp.launch(K.pHp_mass_kernel, dim=self.N,
                      inputs=[self.p, self.mass, self.node2env, self.pHp, self.p_inf],
                      device=d)
            wp.launch(K.pHp_elastic_kernel, dim=self.n_tri,
                      inputs=[self.x, self.p, self.tri, self.Bm, self.area, self.mu,
                              self.lam, self.h2, 1, self.tri2env, self.pHp], device=d)
            for _col in self._active_colliders:
                self._select(_col)
                if self.object_kind == 0:
                    wp.launch(K.pHp_disc_kernel, dim=self.n_b,
                              inputs=[self.x, self.p, self.bnd, self.b2e, self.c, self.radius,
                                      self.dhat, self.kappa, self.h2, 1, self.pHp, self.clr_min],
                              device=d)
                elif self.object_kind > 0:
                    wp.launch(SDF.pHp_sdf_kernel, dim=self.n_b,
                              inputs=[self.x, self.p, self.bnd, self.b2e, self.object_kind, self.obj_cx,
                                      self.obj_cy, self.obj_rx, self.obj_ry, self.obj_rref, self.dhat,
                                      self.kappa, self.h2, 1, self.pHp, self.clr_min], device=d)
            if self.friction:
                # alpha = -gTp/pHp must stay self-consistent: gTp is built from `grad`,
                # which contains friction, so pHp has to contain it too. Dispatched on the
                # same object_kind as the friction gradient above.
                for _col in self._active_colliders:
                    self._select(_col)
                    if self.object_kind == 0:
                        wp.launch(K.pHp_friction_disc_kernel, dim=self.n_b,
                                  inputs=[self.x, self.xn, self.bnd, self.b2e, self.p, self.c,
                                          self.radius, self.dhat, self.kappa, self.mu_fric,
                                          self.epsv, self.inv_dt, self.h2, self.free, self.pHp],
                                  device=d)
                    elif self.object_kind > 0:
                        wp.launch(SDF.pHp_friction_sdf_kernel, dim=self.n_b,
                                  inputs=[self.x, self.xn, self.bnd, self.b2e, self.p,
                                          self.object_kind, self.obj_cx, self.obj_cy, self.obj_rx,
                                          self.obj_ry, self.obj_rref, self.dhat, self.kappa,
                                          self.mu_fric, self.epsv, self.inv_dt, self.h2,
                                          self.free, self.pHp], device=d)
            if self.self_stiffness > 0.0:                 # reuse grid from the grad step (x fixed)
                wp.launch(K.pHp_self_bp_kernel, dim=self.n_b,
                          inputs=[self.sc_grid.id, self.hx3, self.x, self.p, self.bnd, self.b2e,
                                  self.sc_bstart, self.sc_bpe, self.sc_excl, self.sc_excl_off,
                                  self.self_radius, self.self_stiffness, self.h2, 1,
                                  self.pHp], device=d)

            for term in self.terms:
                term.pHp(self)
            wp.launch(K.inversion_alpha_kernel, dim=self.n_tri,
                      inputs=[self.x, self.p, self.tri, self.tri2env,
                              self.area_shrink, self.alpha], device=d)
            for _col in self._active_colliders:
                self._select(_col)
                if self.object_kind == 0:
                    wp.launch(K.disc_ccd_kernel, dim=self.n_b,
                              inputs=[self.x, self.p, self.bnd, self.b2e, self.c, self.radius,
                                      self.ccd_eta, self.alpha], device=d)
                elif self.object_kind > 0:
                    wp.launch(SDF.sdf_ccd_kernel, dim=self.n_b,
                              inputs=[self.x, self.p, self.bnd, self.b2e, self.object_kind, self.obj_cx,
                                      self.obj_cy, self.obj_rx, self.obj_ry, self.obj_rref,
                                      self.ccd_eta, self.alpha], device=d)
            wp.launch(K.alpha_finalize_kernel, dim=self.B,
                      inputs=[self.gTp, self.pHp, self.p_inf, self.clr_min, self.alpha,
                              self.dE, self.dE0, self.active, self.iters,
                              self.dhat, self.near_factor, self.eps, self.tol_x, it], device=d)
            if not self.line_search:
                wp.launch(K.update_x_kernel, dim=self.N,
                          inputs=[self.x, self.p, self.node2env, self.alpha, self.active],
                          device=d)
                for mode in self.modes:
                    mode.advance(self)
            else:
                # ENERGY LINE-SEARCH GATE: x is still the pre-step position here; the grad
                # step just built the self-collision grid at it. Verify the model+CCD step
                # lowers the energy; halve it (per env) up to n_bt times if it doesn't.
                self._energy_into(self.Els_prev, rebuild_grid=False)     # E(x0)
                wp.copy(self.x0buf, self.x)                              # save x0
                for mode in self.modes:
                    mode.save(self)
                for _bt in range(self.n_bt):
                    wp.launch(set_x_from_x0_kernel, dim=self.N,
                              inputs=[self.x, self.x0buf, self.p, self.node2env,
                                      self.alpha, self.active], device=d)
                    for mode in self.modes:
                        mode.trial(self)
                    self._energy_into(self.Ener, rebuild_grid=False)     # E(x0 + alpha*p)
                    wp.launch(ls_halve_kernel, dim=self.B,
                              inputs=[self.Ener, self.Els_prev, self.alpha, self.active,
                                      f64(self.ls_tol)], device=d)
                wp.launch(set_x_from_x0_kernel, dim=self.N,               # commit final alpha
                          inputs=[self.x, self.x0buf, self.p, self.node2env,
                                  self.alpha, self.active], device=d)
                for mode in self.modes:
                    mode.trial(self)

    def set_bc_targets(self, targets_full):
        """Absolute kinematic targets: targets_full is (N,2); the BC nodes' rows
        become the positions the next step_targets() drives them to."""
        t = np.asarray(targets_full, np_real)[self.bc_idx.numpy()]
        wp.copy(self.bc_target, wp.array(t, dtype=vec2, device=self.dev))

    def step_targets(self):
        """One implicit step driving the BC nodes to self.bc_target (set absolutely
        via set_bc_targets)."""
        d = self.dev
        wp.launch(K.init_step_kernel, dim=self.N,
                  inputs=[self.x, self.v, self.xn, self.xhat, self.dt, self.drag], device=d)
        if self.ipc is None:
            wp.launch(K.apply_bc_kernel, dim=self.n_bc,
                      inputs=[self.x, self.xhat, self.bc_idx, self.bc_target], device=d)
        else:
            self.ipc.advance_bc(self)               # swept, so nothing is passed through
        wp.launch(K.reset_step_kernel, dim=self.B,
                  inputs=[self.dE0, self.active, self.iters], device=d)
        self.p.zero_(); self.grad.zero_()
        for mode in self.modes:
            mode.begin_step(self)
        self._iterate()
        wp.launch(K.commit_v_kernel, dim=self.N,
                  inputs=[self.x, self.xn, self.v, 1.0 / self.dt], device=d)

    def _energy_into(self, buf, rebuild_grid=True):
        """Total energy (inertia + elastic + disc barrier + self barrier) per env into
        `buf`, GPU-resident (no host copy). rebuild_grid=False reuses the self-collision
        hash grid already built at the iteration-start x — used for line-search trials,
        where rebuilding every backtrack would be wasteful (the step is small)."""
        buf.zero_(); d = self.dev
        wp.launch(K.energy_kernel, dim=self.N,
                  inputs=[self.x, self.xhat, self.mass, self.node2env, buf], device=d)
        wp.launch(K.energy_elastic_kernel, dim=self.n_tri,
                  inputs=[self.x, self.tri, self.Bm, self.area, self.mu, self.lam,
                          self.h2, self.tri2env, buf], device=d)
        for _col in self._active_colliders:
            self._select(_col)
            if self.object_kind == 0:
                wp.launch(K.energy_disc_kernel, dim=self.n_b,
                          inputs=[self.x, self.bnd, self.b2e, self.c, self.radius, self.dhat,
                                  self.kappa, self.h2, buf], device=d)
            elif self.object_kind > 0:
                wp.launch(SDF.energy_sdf_kernel, dim=self.n_b,
                          inputs=[self.x, self.bnd, self.b2e, self.object_kind, self.obj_cx, self.obj_cy,
                                  self.obj_rx, self.obj_ry, self.obj_rref, self.dhat, self.kappa,
                                  self.h2, buf], device=d)
        if self.self_stiffness > 0.0:
            if rebuild_grid:
                self._build_sc_grid()
            wp.launch(K.energy_self_bp_kernel, dim=self.n_b,
                      inputs=[self.sc_grid.id, self.hx3, self.x, self.bnd, self.b2e,
                              self.sc_bstart, self.sc_bpe, self.sc_excl, self.sc_excl_off,
                              self.self_radius, self.self_stiffness, self.h2, buf], device=d)
        if self.friction:                                  # lagged friction potential (gates friction)
            for _col in self._active_colliders:
                self._select(_col)
                if self.object_kind == 0:
                    wp.launch(K.energy_friction_disc_kernel, dim=self.n_b,
                              inputs=[self.x, self.xn, self.bnd, self.b2e, self.c, self.radius,
                                      self.dhat, self.kappa, self.mu_fric, self.epsv, self.inv_dt,
                                      self.h2, self.free, buf], device=d)
                elif self.object_kind > 0:
                    wp.launch(SDF.energy_friction_sdf_kernel, dim=self.n_b,
                              inputs=[self.x, self.xn, self.bnd, self.b2e, self.object_kind,
                                      self.obj_cx, self.obj_cy, self.obj_rx, self.obj_ry,
                                      self.obj_rref, self.dhat, self.kappa, self.mu_fric,
                                      self.epsv, self.inv_dt, self.h2, self.free, buf], device=d)

        for term in self.terms:
            term.energy(self, buf)

    def energy(self):
        self._energy_into(self.Ener, rebuild_grid=True)
        return self.Ener.numpy().copy()

    def _fill_collider_force(self, include_friction):
        """Barrier reaction on the active colliders into self.Fcol, + friction (Newton's
        3rd law) when asked."""
        wp.launch(zero_vec2_kernel, dim=self.B, inputs=[self.Fcol], device=self.dev)
        for _col in self._active_colliders:
            self._select(_col)
            if self.object_kind == 0:
                wp.launch(K.disc_force_kernel, dim=self.n_b,
                          inputs=[self.x, self.bnd, self.b2e, self.c, self.radius, self.dhat,
                                  self.kappa, self.Fcol], device=self.dev)
            elif self.object_kind > 0:
                wp.launch(SDF.sdf_force_kernel, dim=self.n_b,
                          inputs=[self.x, self.bnd, self.b2e, self.object_kind, self.obj_cx, self.obj_cy,
                                  self.obj_rx, self.obj_ry, self.obj_rref, self.dhat, self.kappa,
                                  self.Fcol], device=self.dev)
        if include_friction and self.friction:
            for _col in self._active_colliders:
                self._select(_col)
                if self.object_kind == 0:
                    wp.launch(K.disc_friction_force_kernel, dim=self.n_b,
                              inputs=[self.x, self.xn, self.bnd, self.b2e, self.c, self.radius,
                                      self.dhat, self.kappa, self.mu_fric, self.epsv, self.inv_dt,
                                      self.Fcol], device=self.dev)
                elif self.object_kind > 0:
                    wp.launch(SDF.sdf_friction_force_kernel, dim=self.n_b,
                              inputs=[self.x, self.xn, self.bnd, self.b2e, self.object_kind,
                                      self.obj_cx, self.obj_cy, self.obj_rx, self.obj_ry,
                                      self.obj_rref, self.dhat, self.kappa, self.mu_fric,
                                      self.epsv, self.inv_dt, self.Fcol], device=self.dev)

    # ------------------------------------------------------------------ state
    def reset_envs(self, pick):
        """Return the environments flagged in `pick` ((B,) of 0/1) to rest, on the device."""
        if getattr(self, "x_rest", None) is None:
            self.x_rest = wp.array(np.asarray(self.me.rest_pos, np_real), dtype=vec2, device=self.dev)
        wp.launch(reset_env_kernel, dim=self.N,
                  inputs=[self.x, self.v, self.xn, self.x_rest, self.node2env,
                          wp.array(np.asarray(pick, np.int32), dtype=wp.int32, device=self.dev)],
                  device=self.dev)

    def set_nodes(self, start, x, v=None):
        """Overwrite the positions (and velocities) of the nodes start .. start + len(x)."""
        xw = wp.array(np.asarray(x, np_real), dtype=vec2, device=self.dev)
        wp.copy(self.x, xw, dest_offset=int(start), count=len(x))
        wp.copy(self.xn, xw, dest_offset=int(start), count=len(x))
        if v is not None:
            wp.copy(self.v, wp.array(np.asarray(v, np_real), dtype=vec2, device=self.dev),
                    dest_offset=int(start), count=len(x))

    # ------------------------------------------------------------------ contact read-outs
    def contact_force(self, friction=True, colliders=True, bodies=True):
        """Force that contact exerts ON each node [N/m], (N, 2): the colliders' barrier and
        the contact between bodies, with friction unless friction=False. Kinematic nodes
        included — that is the reaction a prescribed body feels. Evaluated where the bodies
        are now, with the slip of the last step."""
        d = self.dev
        if getattr(self, "_fc", None) is None:
            self._fc = wp.zeros(self.N, dtype=vec2, device=d)          # minus the force
            self._fc_diag = wp.zeros(self.N, dtype=vec2, device=d)     # the kernels' diagonal: unused
            self._all_free = wp.array(np.ones((self.N, 2), np_real), dtype=vec2, device=d)
        g, dg, free = self._fc, self._fc_diag, self._all_free
        g.zero_()
        if colliders:
            for _col in self.colliders:
                self._select(_col)
                rub = friction and self.friction
                if self.object_kind == 0:
                    wp.launch(K.grad_diag_disc_kernel, dim=self.n_b,
                              inputs=[self.x, self.bnd, self.c, self.radius, self.dhat, self.kappa,
                                      1.0, 1, g, dg], device=d)
                    if rub:
                        wp.launch(K.grad_friction_disc_kernel, dim=self.n_b,
                                  inputs=[self.x, self.xn, self.bnd, self.c, self.radius, self.dhat,
                                          self.kappa, self.mu_fric, self.epsv, self.inv_dt, 1.0,
                                          free, g, dg], device=d)
                elif self.object_kind > 0:
                    shape = [self.object_kind, self.obj_cx, self.obj_cy, self.obj_rx, self.obj_ry,
                             self.obj_rref, self.dhat, self.kappa]
                    wp.launch(SDF.grad_diag_sdf_kernel, dim=self.n_b,
                              inputs=[self.x, self.bnd] + shape + [1.0, 1, g, dg], device=d)
                    if rub:
                        wp.launch(SDF.grad_friction_sdf_kernel, dim=self.n_b,
                                  inputs=[self.x, self.xn, self.bnd] + shape +
                                         [self.mu_fric, self.epsv, self.inv_dt, 1.0, free, g, dg],
                                  device=d)
        if bodies:
            if self.ipc is not None:
                self.ipc.force(self, g, dg, free, friction)
            elif self.self_stiffness > 0.0:                            # the "node" contact
                self._build_sc_grid()
                wp.launch(K.grad_diag_self_bp_kernel, dim=self.n_b,
                          inputs=[self.sc_grid.id, self.hx3, self.x, self.bnd, self.b2e,
                                  self.sc_bstart, self.sc_bpe, self.sc_excl, self.sc_excl_off,
                                  self.self_radius, self.self_stiffness, 1.0, 1, g, dg], device=d)
                if friction:
                    for term in self.terms:
                        if hasattr(term, "pair_force"):
                            term.pair_force(self, g, dg, free)
        return -g.numpy().astype(np.float64)

    def collider_force(self, index=None, include_friction=False):
        """Reaction on the colliders, per environment (B, 2): on collider `index`, or on
        all of them together when index is None."""
        self._active_colliders = self.colliders if index is None else [self.colliders[index]]
        try:
            self._fill_collider_force(include_friction)
            return self.Fcol.numpy().copy()
        finally:
            self._active_colliders = self.colliders
