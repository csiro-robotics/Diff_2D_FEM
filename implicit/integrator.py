"""ImplicitIntegrator — implicit backward-Euler PNCG-IPC stepper.

Advances a Model with `integrator.simulate(model, dt)`.
Unconditionally stable: dt is chosen for accuracy, not by the P-wave CFL limit.
Neo-Hookean elasticity + IPC barrier against the colliders + inversion/CCD step caps +
contact between bodies + friction + every actuator of the Model, minimised together each
step by preconditioned nonlinear conjugate gradients.

HOW MANY ITERATIONS. One CG iteration carries information across one element, and a
backward-Euler step of size dt lets an elastic wave cross C = c_p dt / dx of them — the
step's acoustic CFL number. Measured on slender beams over a 4x range of dt and dx, the
step is converged at 1.5 C iterations and not at 1.0 C; with too few, the structure
responds late and too stiffly. max_iter=None (the default) therefore uses 2 C. The cost
per simulated second is then independent of dt: a larger step needs proportionally more
iterations. Bodies that move as a whole are a different matter — see rigid_modes.

Differentiable: `integrator.adjoint(model, dt)` returns the implicit-function-theorem
adjoint of this solve (see implicit/adjoint.py), giving dJ/d(density) for an objective
on the checkpointed trajectory at the cost of one linear solve per step.
"""
from __future__ import annotations
import warnings
import numpy as np

from .solver import PNCG2DSolver
from ..core.precision import np_real
from .adjoint import AdjointPNCG2D
from . import actuation as ACT
from .affine import AffineBodies, RigidModes
from .body_contact import BodyFriction
from .ipc_contact import IPCContact


class ImplicitIntegrator:
    def __init__(self, max_iter=None, eps=None, self_collision=True, friction=True,
                 near_factor=4.0, alpha_max=3.0, area_shrink=0.1, ccd_eta=0.9,
                 self_radius=None, tol_x=0.0, conv_check=8,
                 line_search=False, n_backtrack=4, ls_tol=1e-7, rigid_modes=True,
                 iters_per_cfl=2.0, max_iter_cap=2000, cg_restart=None):
        """max_iter  iterations per step. None: iters_per_cfl x the step's acoustic CFL
                  number (see the module docstring), between 40 (120 with IPC contact
                  between bodies) and max_iter_cap.
        Convergence, per implicit step (either test stops an environment early):
          eps     energy-decrease ratio: stop once an iteration lowers the energy by less
                  than eps x what the step's first iteration did. The error left in the
                  step's displacement is about sqrt(eps). None: 1e-10 in double precision;
                  1e-4 in single, where the sums that form the ratio have a rounding
                  floor near 1e-5 and anything tighter just runs to max_iter.
          tol_x   increment inf-norm [m]: the largest node move this iteration would make.
                  0 disables it (the default).
          conv_check  every N iterations, stop the loop once ALL environments have
                  converged (one device->host sync per check). 0: always run max_iter.
        rigid_modes  give every free piece of soft material its rigid-body directions in
                  the preconditioner (implicit/affine.py). Without it a body nothing holds
                  — resting on the ground, falling, thrown — needs hundreds of iterations
                  to move as a whole.
        line_search  energy backtracking gate on every step (n_backtrack halvings, ls_tol
                  slack): guarantees monotone energy decrease, so a stiff or dense body
                  cannot diverge. ~2-3x the cost; off by default.
        cg_restart  restart the conjugate direction when the gradient changed along the
                  last step by less than this fraction of what the step's curvature
                  promised (solver.cg_guard_kernel). None: 0.5 where bodies are in IPC
                  contact — friction and an opening or closing contact make the plain
                  Dai-Kou direction wander (a block sliding with friction: 26 of 400 steps
                  unconverged without it, none with it, in two thirds of the iterations)
                  — and 0, plain Dai-Kou, otherwise."""
        from ..core.precision import PRECISION
        self.max_iter = None if max_iter is None else int(max_iter)
        self.iters_per_cfl = float(iters_per_cfl); self.max_iter_cap = int(max_iter_cap)
        self.eps = float((1e-10 if PRECISION == "double" else 1e-4) if eps is None else eps)
        self.tol_x = float(tol_x); self.conv_check = int(conv_check)
        self.line_search = bool(line_search)
        self.n_backtrack = int(n_backtrack); self.ls_tol = float(ls_tol)
        self.self_collision = bool(self_collision); self.friction = bool(friction)
        self.near_factor = float(near_factor); self.alpha_max = float(alpha_max)
        self.area_shrink = float(area_shrink); self.ccd_eta = float(ccd_eta)
        self.self_radius = self_radius     # None -> 1.5*dx (clears pull-stage self-penetration)
        self.rigid_modes = bool(rigid_modes)
        self.cg_restart = cg_restart

    def simulate(self, model, dt, control=None):
        """Advance one implicit step, driving the constrained nodes to model's targets."""
        s = self.attach(model, dt)
        s.set_bc_targets(model._targets)
        s.step_targets()
        model.time += dt

    def cfl_number(self, model, dt):
        """Acoustic CFL number of a step: how many elements a P-wave crosses in dt, the
        largest over the soft bodies. Rigid regions are left out — their stiffness never
        enters the nodal iteration."""
        me = model.me
        mu, lam = np.asarray(me.mu, np.float64), np.asarray(me.lam, np.float64)
        soft = np.ones(len(mu), bool)
        for a in model._affine:
            soft[a["elems"]] = False
        C = 0.0
        for h, b in enumerate(model._bodies):
            el = model._tri_range[h]; el = el[soft[el]]
            if len(el) and not b.get("kinematic"):       # a prescribed body's stiffness plays no part
                cp = np.sqrt((lam[el] + 2.0 * mu[el]).max() / b["rho"])
                C = max(C, cp * dt / model._cfl_length(h))
        return float(C)

    def _iters_for(self, model, dt):
        if self.max_iter is not None:
            return self.max_iter
        C = self.cfl_number(model, dt)
        n = int(np.ceil(self.iters_per_cfl * C))
        if n > self.max_iter_cap and not getattr(self, "_warned_cap", False):
            self._warned_cap = True
            warnings.warn(
                f"this step's acoustic CFL number is {C:.0f}, which calls for about {n} "
                f"iterations; using max_iter_cap = {self.max_iter_cap}. The dynamics will be "
                f"under-resolved (the structure responds late). Use a smaller dt, a softer or "
                f"coarser model, rigid bodies for stiff parts, or raise max_iter_cap.",
                stacklevel=4)
        # A contact that closes or opens within the step is not a wave crossing elements:
        # an impact at half a pixel per step took up to 124 iterations where 2 C was 45, and
        # cut short it lost 4 % of the momentum. The extra ones are only spent when needed.
        lo = 120 if self.self_collision and model.body_contact == "ipc" else 40
        return int(np.clip(n, lo, self.max_iter_cap))

    def stats(self, model):
        """Per environment, for the last step: iterations used and whether the step met the
        convergence tolerance before running out of them."""
        s = model._solver
        out = dict(iters=s.iters.numpy().copy(), converged=(s.active.numpy() == 0),
                   max_iter=s.iter_max)
        if s.ipc is not None and s.n_bc:
            # how far the kinematic nodes are from where they were told to go: 0 unless
            # something they cannot push aside was in the way (see ipc_contact.py)
            lag = np.linalg.norm(s.bc_target.numpy() - s.x.numpy()[s.bc_idx.numpy()], axis=1)
            out["bc_lag"] = float(lag.max())
        return out

    def attach(self, model, dt):
        """Build the solver for this (finalized) model if it does not exist yet, and
        return it. simulate() does this lazily; call it yourself when you need the solver
        before the first step — the adjoint has to set the density field first."""
        if model._solver is None:
            self._build(model, dt)
        s = model._solver
        if abs(s.dt - dt) > 1e-12 * dt:                  # the step size changed
            s.set_dt(dt)
            s.iter_max = self._iters_for(model, dt)
        self._sync_actuation(model)
        return s

    # ---- actuation: mirror the model's actuators onto the solver as energy terms ----
    def _sync_actuation(self, model):
        """Rebuild the solver's actuation terms when an actuator was added, and push the
        command values when one changed. Both are no-ops otherwise, so calling this every
        step costs two integer comparisons."""
        s = model._solver
        if getattr(s, "_act_version", None) != model._act_version:
            keep = [t for t in s.terms if isinstance(t, ACT.ExternalForce)
                    or getattr(t, "persistent", False)]
            s.terms = list(keep)
            force = [t for t in keep if isinstance(t, ACT.ExternalForce)]
            s._act = {"force": force[0] if force else None, "pressure": None,
                      "tendon": None, "strain": None, "spring": None}
            if model._springs:
                S = model._springs
                s._act["spring"] = ACT.Springs(s, [k["a"] for k in S], [k["b"] for k in S],
                                               [k["stiffness"] for k in S], [k["env"] for k in S])
                s.terms.append(s._act["spring"])
            if model._chambers:
                C = model._chambers
                s._act["pressure"] = ACT.Pressure(
                    s, np.concatenate([c["ea"] for c in C]), np.concatenate([c["eb"] for c in C]),
                    np.concatenate([np.full(len(c["ea"]), i) for i, c in enumerate(C)]),
                    [model.body_env(c["body"]) for c in C], [c["ref"] for c in C])
                s.terms.append(s._act["pressure"])
            if model._tendons:
                node, prv, nxt, tid = [], [], [], []
                for i, t in enumerate(model._tendons):
                    o, n = len(node), len(t["nodes"])
                    node += list(t["nodes"]); tid += [i] * n
                    prv += [-1] + list(range(o, o + n - 1))
                    nxt += list(range(o + 1, o + n)) + [-1]
                T = model._tendons
                s._act["tendon"] = ACT.Tendon(s, node, prv, nxt, tid,
                                              [model.body_env(t["body"]) for t in T],
                                              [t["stiffness"] for t in T], [t["length"] for t in T])
                s.terms.append(s._act["tendon"])
            if model._regions:
                R = model._regions
                s._act["strain"] = ACT.ActiveStrain(
                    s, np.concatenate([r["elems"] for r in R]),
                    np.concatenate([np.full(len(r["elems"]), i) for i, r in enumerate(R)]), len(R))
            s._act_version = model._act_version
            s._ctrl_version = None
        if s._ctrl_version != model._ctrl_version:
            if s._act["pressure"] is not None:
                s._act["pressure"].set([c["pressure"] for c in model._chambers])
            if s._act["tendon"] is not None:
                s._act["tendon"].set([t["tension"] for t in model._tendons],
                                     [t["length"] for t in model._tendons])
            if s._act["strain"] is not None:
                s._act["strain"].set(s, np.stack([r["A"] for r in model._regions]))
            if s._act["spring"] is not None:
                s._act["spring"].set([k["anchor"] for k in model._springs],
                                     [k["length"] for k in model._springs])
            g = model._gravity
            if model._fext is not None or g != (0.0, 0.0):
                if s._act["force"] is None:
                    s._act["force"] = ACT.ExternalForce(s); s.terms.insert(0, s._act["force"])
                f = np.zeros((s.N, 2), np_real) if model._fext is None else model._fext.copy()
                f += s.mass.numpy()[:, None] * np.asarray(g, np_real)[None, :]
                s._act["force"].set(f)
                # A net force on a body nothing holds drives a RIGID motion. The Jacobi
                # preconditioner scales that as if it were a stiff deformation, so the first
                # step is tiny; capped at alpha_max it can fall under tol_x and be reported
                # as converged with the body not having moved (measured: free fall, 100 %
                # error at alpha_max = 3, tol_x = 1e-8; 0.1 % with the cap lifted).
                loose = [h for h in range(model.n_bodies)
                         if h not in model._constrained or len(model._constrained[h]) == 0]
                if loose and self.alpha_max < 1e3 and not self.rigid_modes \
                        and not getattr(self, "_warned_loose", False):
                    pushed = [h for h in loose if np.abs(f[model._node_range[h]]).max() > 0]
                    if pushed:
                        self._warned_loose = True
                        warnings.warn(
                            f"a force acts on unanchored body {pushed}, but alpha_max = "
                            f"{self.alpha_max:g} caps each iteration's step: its rigid motion "
                            f"will converge very slowly or not at all. Leave rigid_modes "
                            f"on (the default), or lift alpha_max.", stacklevel=3)
            s._ctrl_version = model._ctrl_version

    def adjoint(self, model, dt, Emin=1e-3):
        """Adjoint of the implicit solve on this model: an AdjointPNCG2D over the solver.

            adj = integ.adjoint(model, dt)
            adj.set_density(rho)                 # per-element density in [0, 1]
            model.reset(); xs = [model.get_x()]
            for t in range(T):
                ...set targets...; integ.simulate(model, dt); xs.append(model.get_x())
            g = adj.transient_adjoint(xs, seed={T: dJ_dx_T}, use_disc=True)   # dJ/d rho

        Density scales each element's stiffness, E(rho) = Emin + rho (1 - Emin). The
        contact / self-contact active set is frozen at each converged state.

        Gravity and external forces are fine (they have no curvature). Pressure, tendons,
        active material and IPC contact between bodies are not in the adjoint operator
        yet, and transient_adjoint refuses to run with them rather than return a gradient
        of a different system. model.to_elements / model.to_pixels map a per-pixel
        density field onto a body's elements and a gradient back."""
        return AdjointPNCG2D(self.attach(model, dt), Emin=Emin)

    # ---- lazy solver construction from the finalized Model ----
    def _build(self, model, dt):
        s = PNCG2DSolver(
            model.me, model.cfg, model.all_constrained(), colliders=model._colliders,
            device=model.device, dt=dt, iter_max=self._iters_for(model, dt), eps=self.eps,
            tol_x=self.tol_x, conv_check=self.conv_check,
            line_search=self.line_search, n_backtrack=self.n_backtrack, ls_tol=self.ls_tol,
            near_factor=self.near_factor, alpha_max=self.alpha_max,
            area_shrink=self.area_shrink, ccd_eta=self.ccd_eta,
            self_stiffness=(model.cfg.contact_stiffness
                            if self.self_collision and model.body_contact == "node" else 0.0),
            self_radius=(self.self_radius if self.self_radius is not None else 1.5 * model.dx),
            friction=self.friction)
        if model._v0 is not None and np.any(model._v0):
            s.v.assign(model._v0.astype(np_real))
        # body-level modes: rigid regions first (they take their nodes out of the nodal
        # iteration), then the rigid-mode coarse space of whatever free soft pieces remain
        s.affine = AffineBodies(s, model._affine) if model._affine else None
        if s.affine is not None:
            s.modes.append(s.affine); s.terms.append(s.affine)
        mu_bb = model.body_friction if self.friction else 0.0
        mu_node = model._body_mu_per_node() if self.friction else None     # per-body mu=
        if mu_node is not None and model.body_friction == 0.0:
            bare = [h for h, b in enumerate(model._bodies) if b.get("mu") is None]
            if bare:
                warnings.warn(
                    f"bodies {bare} have no mu= and model.body_friction is 0, so every contact "
                    f"they make with another body is frictionless (a pair slides with the "
                    f"harmonic mean of its two bodies' mu). Give them mu=, or set "
                    f"model.body_friction.", stacklevel=4)
        if model.body_contact not in ("ipc", "node"):
            raise ValueError(f"model.body_contact must be 'ipc' or 'node', "
                             f"not {model.body_contact!r}")
        if self.self_collision and model.body_contact == "ipc":
            s.ipc = IPCContact(s, mu_bb, dhat=model.dhat, kappa=model.body_kappa,
                               reach=max(2.0 * model.dx, 2.0 * model.dhat), mu_node=mu_node)
            s.terms.append(s.ipc)
            gap = s.ipc.min_distance(s)
            if not gap.min() > 0.0:
                raise ValueError(
                    f"two surfaces touch in environment {int(gap.argmin())} before the first "
                    f"step. IPC contact needs the bodies to start apart: leave a gap between "
                    f"them (anything from model.dhat = {model.dhat:.3g} m up is out of the "
                    f"barrier's reach).")
            if any(b.get("mesh") for b in model._bodies) and gap.min() < model.dhat:
                warnings.warn(
                    f"two boundaries are {gap.min():.3g} m apart at rest in environment "
                    f"{int(gap.argmin())}, inside the barrier's reach (model.dhat = "
                    f"{model.dhat:.3g} m): they push each other apart from the first step. A "
                    f"slit in a mesh narrower than dhat does this; widen it or lower model.dhat.",
                    stacklevel=4)
        elif self.self_collision and (mu_bb > 0.0 or (mu_node is not None and mu_node.max() > 0.0)):
            s.terms.append(BodyFriction(s, mu_bb, mu_node))
        self._warn_mesh_scales(model, s)
        s.cg_guard = float((0.5 if s.ipc is not None else 0.0) if self.cg_restart is None
                           else self.cg_restart)
        if self.rigid_modes:
            rm = RigidModes.build(s, skip_nodes=(s.affine.slaved if s.affine is not None else None))
            if rm is not None:
                s.modes.append(rm)
        model._solver = s

    def _warn_mesh_scales(self, model, s):
        """Contact is sampled at boundary nodes. A mesh body's boundary edges can be far
        longer than its median one, which sets the contact scale; say so where it bites."""
        L = max((b["L_max"] for b in model._bodies if b.get("mesh")), default=0.0)
        if L == 0.0:
            return
        if self.self_collision and model.body_contact == "node" and L >= 2.0 * s.self_radius:
            warnings.warn(
                f"a mesh body has boundary edges up to {L * 1e3:.3g} mm, more than twice the "
                f"'node' contact radius {s.self_radius * 1e3:.3g} mm: a node can slip through such "
                f"an edge between its ends. Use body_contact='ipc', or a finer boundary.",
                stacklevel=4)
        from ..core import objects as OBJ
        feat = []                                  # radius of the sharpest curvature of each collider
        for c in model._colliders:
            if c["kind"] in (OBJ.DISC, OBJ.ELLIPSE):
                feat.append(min(c["rx"], c["ry"]))
            elif c["kind"] in (OBJ.BOX, OBJ.TRIANGLE):
                feat.append(c["rref"])
        if feat and L > min(feat):
            warnings.warn(
                f"a mesh body has boundary edges up to {L * 1e3:.3g} mm, longer than the "
                f"smallest collider feature ({min(feat) * 1e3:.3g} mm). Collider contact acts on "
                f"nodes, so the collider can pass between the ends of such an edge; refine the "
                f"mesh's boundary there.", stacklevel=4)

    # ---- read-outs ----
    def collider_force(self, model, index=None, friction=True):
        """Reaction on the colliders (barrier, + friction unless friction=False), per
        environment — shape (B, 2): on collider `index`, or on all of them together when
        index is None."""
        return model._solver.collider_force(index, include_friction=friction)

    def contact_force(self, model, handle=None, friction=True, colliders=True, bodies=True):
        """Force that contact exerts on each node of a body [N/m] — (n_body_nodes, 2), or
        (N, 2) for every node when handle is None. colliders / bodies choose what is
        counted: the colliders' barrier, the contact with other bodies (and of the body
        with itself, which cancels in any sum over the body). friction=False leaves the
        normal part. Kinematic nodes are included: that is the reaction a prescribed body
        feels. After at least one step."""
        f = model._solver.contact_force(friction=friction, colliders=colliders, bodies=bodies)
        return f if handle is None else f[model.body_nodes(handle)]

    def body_contact_force(self, model, handle, friction=True, colliders=True, bodies=True):
        """Resultant of contact_force over a body: (fx, fy) [N/m]."""
        return self.contact_force(model, handle, friction, colliders, bodies).sum(0)
