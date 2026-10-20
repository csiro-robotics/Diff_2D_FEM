"""Model — the scene container for the 2D differentiable FEM.

Holds soft bodies, fixed rigid colliders (ground, walls, discs and other signed-distance
shapes), contact parameters, kinematic boundary conditions and actuators; an Integrator
advances it via `integrator.simulate(model, dt)`. Contact parameters are exposed directly
as attributes (`model.dhat`, `model.kappa`, `model.epsv`, ...).

    model = Model()
    model.add_ground(0.0, mu=0.6)
    h = model.add_soft_body(mask, physical_size=(0.032, 0.083), center=(0.06, 0.05),
                            E=1.0e6, nu=0.40, density=1100.0)
    model.finalize()
    model.set_kinematic_constraint(h, driven_node_ids)
    integ = ImplicitIntegrator()
    for t in range(T):
        x = model.get_body_x(h); x[driven_node_ids] += step
        model.set_kinematic_target(h, x)
        integ.simulate(model, dt)

BODIES are rasterised from pixel masks into P1 triangles; each has its own grid, size,
position and material. A body can also be given directly as a triangle mesh
(add_soft_mesh and its rigid / kinematic counterparts): its vertices are its nodes and its
triangles its elements, in the order given.

ENVIRONMENTS. A scene is one environment by default: every body added lands in
environment 0 and the bodies collide with one another. `env_id` puts a body in another
environment instead. Environments are simulated together in one batched solve but never
interact — each sees the same colliders and nothing of the others — which is how a
population of designs, or a sweep over a control signal, is run in a single call:

    for i, mask in enumerate(designs):
        model.add_soft_body(mask, ..., env_id=i)
"""
from __future__ import annotations
import hashlib
import warnings
import numpy as np

from .core.config import SimConfig
from .core.precision import np_real, PRECISION
from .core.mesh import build_scene, check_mesh, meshes_overlap
from .core import objects as _objects
from .core import actuators as _act


class Model:
    def __init__(self, device="cuda", config=None):
        """A scene starts neutral — no damping, friction 0.5 — and its contact parameters are
        scaled to it at finalize():

            dhat            1/8 pixel               the barrier's reach
            kappa           0.004 x E x pixel/dhat  the barrier's stiffness
            self_stiffness  0.2 x E                 contact between bodies
            epsv            1 mm/s                  below this, friction is smoothed

        with E the modulus of the softest body. Assign any of them (model.dhat = ...) to
        override. A body resting on a collider floats about 0.9 dhat above it, and the
        barrier's reach is a modelling error that shrinks with it: one cycle of a crawling
        gait covered 1.73 mm at dhat = 1 pixel, 1.19 at 1/4, 1.09 at 1/8, 1.07 at 1/16.
        kappa follows dhat so the body sits at the same fraction of it.

        config: a SimConfig to start from instead of the defaults; whatever it sets is
        taken as given, and only what it leaves at None is scaled to the scene."""
        self.cfg = SimConfig() if config is None else config
        self._colliders = []             # rigid colliders: dict(kind, cx, cy, rx, ry, rref, r, mu)
        self.device = device
        self._bodies = []                # dict(mask, size, center, E, nu, rho, env) per body;
                                         # a mesh body: dict(mesh=True, vertices, triangles, h,
                                         # h_cfl, L_max, center, E, nu, rho, env) — no mask/size
        self._mesh_info = {}             # check_mesh results, by mesh digest (a batch: once)
        self._constrained = {}           # handle -> global node indices under kinematic control
        self.me = None                   # the scene Mesh, built at finalize()
        self._solver = None              # attached by the Integrator on first simulate()
        self._targets = None             # (N,2) absolute kinematic targets (host)
        self._node_range = []            # handle -> global node indices of that body
        self._tri_range = []             # handle -> global element indices of that body
        self._v0 = None                  # (N,2) velocities set before the solver exists
        self._epoch = 0                  # bumped by reset(); integrators re-sync their state
        self.time = 0.0                  # simulated time [s]: advanced by every step, zeroed by reset()
        # ---- actuation (see the "actuation" section below) ----
        self._gravity = (0.0, 0.0)
        self._fext = None                # (N, 2) external nodal force, besides gravity
        self._chambers = []              # dict(body, ea, eb, ref, area0, sign, pressure)
        self._tendons = []               # dict(body, nodes, stiffness, tension, length)
        self._regions = []               # dict(body, elems, fiber, A)
        self._affine = []                # dict(body, elems): rigid regions (affine bodies)
        self._affine_of, self._regions_of = {}, {}   # body -> its rigid / active elements so far
        self._springs = []               # dict(a, b, stiffness, length, anchor)
        # ---- contact between bodies (and of a body with itself); set before the first step
        # "ipc":  the IPC log barrier between boundary nodes and boundary edges — what the
        #         colliders get, between every pair of bodies, soft or rigid
        #         (implicit/ipc_contact.py). Surfaces settle within dhat of each other, no
        #         step can cross one, friction is Coulomb's, and kinematic nodes are swept
        #         to their targets instead of teleported. The bodies must start apart.
        # "node": the penalty contact that came before it: boundary nodes repel each other
        #         within 1.5 pixels. The surface it makes is corrugated (it resists sliding
        #         like a friction coefficient of up to 0.35) and it holds bodies 1.5 pixels
        #         apart. Kept because the adjoint can differentiate it.
        self.body_contact = "ipc"
        self.body_friction = 0.0         # Coulomb coefficient between bodies; 0 = frictionless.
                                         # A body's own mu= (add_*) overrides it for that body.
        # "ipc" mode only: the barrier's stiffness between bodies [Pa]. None: 0.1 x E x
        # pixel / dhat (E of the softest body), which leaves surfaces a third of dhat
        # apart when they are pressed together with 0.3 E. model.kappa, the colliders'
        # stiffness, is 25 times softer: it is set for a body lying on the ground under its
        # own weight, and between bodies squeezed in a grip it would leave a few
        # hundredths of dhat — a far stiffer, slower-converging contact for the same load.
        self.body_kappa = None
        # Stiffness [Pa] a rigid region resists strain with. Single precision cannot carry
        # more: at 1e9 the rounding of the region's elastic force feeds energy into its
        # rotation (measured: a free spinning body ends 6 mm from where it should be).
        self.rigid_young = 1.0e9 if PRECISION == "double" else 1.0e7
        self.rigid_nu = 0.30
        self._act_version = 0            # bumped when an actuator is ADDED  -> terms rebuilt
        self._ctrl_version = 0           # bumped when a control value changes -> values pushed

    # ---- contact/material attribute proxies over the config ----
    def _p(name):
        return property(lambda s: getattr(s.cfg, name),
                        lambda s, v: setattr(s.cfg, name, v))
    dhat = _p("ipc_d_hat")
    kappa = _p("obstacle_stiffness")
    self_stiffness = _p("contact_stiffness")
    mu = _p("obstacle_friction")
    epsv = _p("obstacle_friction_epsv")
    damping = _p("damping")
    del _p

    # ---- scene construction ----
    # ---- rigid colliders -------------------------------------------------------------
    # Fixed analytic shapes the bodies touch: the ground, a wall, an object to grasp, an
    # obstacle to climb over. Any number of them; each add_* returns the collider's index.
    def add_disc(self, center, radius, mu=None):
        """Add a rigid circular collider."""
        return self.add_object("disc", center, radius, mu=mu)

    def add_plane(self, normal, origin, mu=None):
        """Add a rigid wall: the half-space behind the line through `origin` whose `normal`
        points into the free side. (Taccel's add_plane(normal, origin, mu).)"""
        n = np.asarray(normal, np.float64)
        if not np.linalg.norm(n) > 0:
            raise ValueError("the plane normal must be non-zero")
        n = n / np.linalg.norm(n)
        return self._set_collider(_objects.HALFSPACE, origin, float(n[0]), float(n[1]), 0.0, mu)

    def add_ground(self, height=0.0, mu=None):
        """Add horizontal ground at y = height — what a walking or crawling robot stands on."""
        return self.add_plane((0.0, 1.0), (0.0, height), mu=mu)

    def add_object(self, kind, center, rx, ry=None, corner_radius=3.0e-3, mu=None):
        """Add the rigid obstacle as a signed-distance shape (core/sdf.py).

        kind   "disc" | "ellipse" | "box" (a square when rx == ry) | "triangle"
               (equilateral, apex +y), or the matching integer from core.objects.
        rx,ry  half-extents; ry defaults to rx. Disc: rx is the radius. Triangle: rx is the
               circumradius.
        corner_radius  rounding of the box / triangle corners. Keep it > 0: a sharp corner
               has unbounded curvature, which makes the barrier ill-conditioned there.

        A disc keeps the exact closed-form contact kernels; every other shape goes through
        the level-set ones. Contact, friction and CCD all dispatch on the same kind."""
        k = _objects.NAMES[kind] if isinstance(kind, str) else int(kind)
        rx = float(rx); ry = rx if ry is None else float(ry)
        rref = float(corner_radius) if k in (_objects.BOX, _objects.TRIANGLE) else rx
        return self._set_collider(k, center, rx, ry, rref, mu)

    def _set_collider(self, kind, center, rx, ry, rref, mu):
        if self._solver is not None:
            raise RuntimeError("add colliders before the first simulate()")
        c = self.cfg
        mu = float(c.obstacle_friction if mu is None else mu)
        self._colliders.append(dict(kind=int(kind), cx=float(center[0]), cy=float(center[1]),
                                    rx=rx, ry=ry, rref=rref, r=rx, mu=mu))
        return len(self._colliders) - 1

    def set_collider_center(self, index, center):
        """Move a collider. Before the first step this just places it. During a rollout it
        TELEPORTS the shape between steps: the move is not swept, so keep each step's
        displacement well under the current clearance (a node the surface jumps past is left
        inside it), and friction treats the collider as stationary."""
        c = self._colliders[index]
        c["cx"], c["cy"] = float(center[0]), float(center[1])
        if self._solver is not None:
            self._solver.colliders[index].update(cx=c["cx"], cy=c["cy"])

    @property
    def colliders(self):
        """Every collider as a host-side core.objects.Obj — for plotting and shape-aware
        metrics (objects.patch / phi / perimeter / contact_arc)."""
        return [_objects.Obj(kind=c["kind"], cx=c["cx"], cy=c["cy"], rx=c["rx"], ry=c["ry"],
                             rref=c["rref"]) for c in self._colliders]

    @property
    def object(self):
        """The first collider (objects.NONE-kind when the scene has none)."""
        cs = self.colliders
        return cs[0] if cs else _objects.Obj(kind=_objects.NONE)

    def add_soft_body(self, mask, physical_size, center,
                      E=None, nu=None, density=None, env_id=0, mu=None):
        """Add a soft body from a pixel mask. Returns its handle.

        mask           (H, W), non-zero = material, ROW 0 AT THE BOTTOM. Each solid pixel
                       becomes two P1 triangles.
        physical_size  (width, height) of the whole pixel grid [m].
        center         (x, y) of the grid's centre [m].
        E              Young's modulus [Pa]: a scalar, or an (H, W) map laid out like the
                       mask — a stiffer skin, a strain-limiting layer, a soft hinge. A
                       pneumatic finger only bends because one side is stiffer than the other.
        nu, density    Poisson ratio and density [kg/m^3].
        env_id         the environment the body lives in (see the module docstring).
        mu             Coulomb friction of this body against other bodies and itself.
                       A pair of bodies slides with the harmonic mean of their two
                       values, 2 mu_a mu_b / (mu_a + mu_b) — 0 if either is 0 — as in
                       Taccel. None: model.body_friction. Colliders keep their own mu.

        A material property left out is the model's default (model.cfg: 1 MPa, 0.40,
        1000 kg/m^3)."""
        if self.me is not None:
            raise RuntimeError("add bodies before finalize()")
        mask = (np.asarray(mask) > 0.5).astype(np.uint8)
        if mask.ndim != 2 or not mask.any():
            raise ValueError("the mask must be a 2-D array with at least one solid pixel")
        if E is not None and not np.isscalar(E):
            E = np.asarray(E, np.float64)
            if E.shape != mask.shape:
                raise ValueError(f"stiffness map is {E.shape}, the mask is {mask.shape}")
            if (E[mask > 0] <= 0).any():
                raise ValueError("stiffness map must be positive wherever the mask is solid")
        c = self.cfg
        if int(env_id) < 0:
            raise ValueError("env_id must be >= 0")
        mu = self._check_mu(mu)
        self._bodies.append(dict(
            mask=mask,
            size=tuple(float(v) for v in physical_size),
            center=tuple(float(v) for v in center),
            E=(c.young if E is None else (float(E) if np.isscalar(E) else E)),
            nu=float(c.nu if nu is None else nu),
            rho=float(c.rho_mass if density is None else density),
            env=int(env_id), mu=mu))
        return len(self._bodies) - 1

    def add_affine_body(self, mask, physical_size, center, density=None,
                        env_id=0, E=None, nu=None, mu=None):
        """Add a RIGID body from a pixel mask. Returns its handle.

        It is an affine body (Affine Body Dynamics): its nodes move as x = t + A X with one
        translation t and one 2 x 2 matrix A for the whole body, and A is held to a rotation
        by a large stiffness (model.rigid_young unless E is given), so the body translates
        and rotates freely and strains by parts per million. Six unknowns however many
        pixels it has, with their own exact block of the preconditioner — which is why this,
        and not a soft body with a huge modulus, is the way to model a rigid part.

        It collides, slides and takes forces like any other body. For a body whose motion
        is PRESCRIBED, use add_kinematic_body instead."""
        h = self.add_soft_body(mask, physical_size, center,
                               E=(self.rigid_young if E is None else E),
                               nu=(self.rigid_nu if nu is None else nu),
                               density=density, env_id=env_id, mu=mu)
        if not np.isscalar(self._bodies[h]["E"]):
            raise ValueError("a rigid body takes a scalar stiffness")
        self._bodies[h]["affine"] = True
        return h

    def add_kinematic_body(self, mask, physical_size, center, env_id=0, mu=None):
        """Add a body whose motion is PRESCRIBED: a jaw, a pusher, a moving platform, a wall
        that can be put somewhere else. Returns its handle.

        Every one of its nodes is kinematically constrained. Move it with
        set_kinematic_pose(handle, translation, angle) each step — or set_kinematic_target
        for a motion that is not rigid. It touches the other bodies through contact like
        any body; under IPC contact it is swept to where it is told to be, not placed
        there, and stops short of a pose it cannot reach (integrator.stats: bc_lag)."""
        h = self.add_soft_body(mask, physical_size, center, env_id=env_id, mu=mu)
        self._bodies[h]["kinematic"] = True
        return h

    # ---- bodies given as triangle meshes ----
    def add_soft_mesh(self, vertices, triangles, E=None, nu=None, density=None, env_id=0, mu=None):
        """Add a soft body given as a triangle mesh. Returns its handle.

        vertices   (n, 2) node positions in the world [m] — where the body is at rest.
        triangles  (m, 3) vertex indices, 0-based; either winding.
        E          Young's modulus [Pa]: a scalar, or one value per triangle ((m,)).
        nu, density, env_id, mu   as add_soft_body.

        Vertex i is node i of the body and triangle j its element j, in the order given —
        so pins, tendons, constraints and per-element fields address the mesh's own
        numbering. Everything that acts on nodes or elements works as on a pixel body;
        what is laid out on a pixel grid takes per-triangle arrays instead (a stiffness map,
        add_affine_region, add_active_region, to_elements), and a hole in the mesh is a
        pressure chamber. The mesh is checked first (core.mesh.check_mesh) and a defect is
        reported by vertex or triangle. Its resolution is the mesh's own: P1 triangles are
        stiff in bending, so give a slender part several elements through its thickness."""
        if self.me is not None:
            raise RuntimeError("add bodies before finalize()")
        info = self._check_mesh(vertices, triangles)
        m = len(info["triangles"])
        if E is not None and not np.isscalar(E):
            E = np.asarray(E, np.float64)
            if E.shape != (m,):
                raise ValueError(f"stiffness per triangle is {E.shape}; the mesh has {m} triangles")
            if not (np.isfinite(E).all() and (E > 0).all()):
                raise ValueError("stiffness must be positive and finite on every triangle")
        c = self.cfg
        if int(env_id) < 0:
            raise ValueError("env_id must be >= 0")
        mu = self._check_mu(mu)
        V = info["vertices"]
        self._bodies.append(dict(
            mesh=True, vertices=V, triangles=info["triangles"],
            h=info["h"], h_cfl=info["h_cfl"], L_max=info["L_max"],
            center=tuple(float(v) for v in 0.5 * (V.min(0) + V.max(0))),
            E=(c.young if E is None else (float(E) if np.isscalar(E) else E)),
            nu=float(c.nu if nu is None else nu),
            rho=float(c.rho_mass if density is None else density),
            env=int(env_id), mu=mu))
        return len(self._bodies) - 1

    def add_affine_mesh(self, vertices, triangles, density=None, env_id=0, E=None, nu=None, mu=None):
        """Add a RIGID body given as a triangle mesh — add_affine_body for a mesh. Returns
        its handle."""
        h = self.add_soft_mesh(vertices, triangles,
                               E=(self.rigid_young if E is None else E),
                               nu=(self.rigid_nu if nu is None else nu),
                               density=density, env_id=env_id, mu=mu)
        if not np.isscalar(self._bodies[h]["E"]):
            raise ValueError("a rigid body takes a scalar stiffness")
        self._bodies[h]["affine"] = True
        return h

    def add_kinematic_mesh(self, vertices, triangles, env_id=0, mu=None):
        """Add a body with PRESCRIBED motion given as a triangle mesh — add_kinematic_body
        for a mesh. Returns its handle."""
        h = self.add_soft_mesh(vertices, triangles, env_id=env_id, mu=mu)
        self._bodies[h]["kinematic"] = True
        return h

    @staticmethod
    def _check_mu(mu):
        if mu is None:
            return None
        mu = float(mu)
        if not (np.isfinite(mu) and mu >= 0.0):
            raise ValueError(f"mu must be a finite number >= 0, not {mu}")
        return mu

    def _body_mu_per_node(self):
        """Each node's body friction (its body's mu, or model.body_friction) — or None when
        no body has its own mu, so that the scalar body_friction is used as before."""
        if all(b.get("mu") is None for b in self._bodies):
            return None
        out = np.zeros(self.n_nodes, np.float64)
        for h, b in enumerate(self._bodies):
            out[self._node_range[h]] = self.body_friction if b.get("mu") is None else b["mu"]
        return out

    def _check_mesh(self, vertices, triangles):
        V = np.ascontiguousarray(vertices, np.float64)
        T = np.ascontiguousarray(triangles)
        key = (V.shape, T.shape, str(T.dtype),
               hashlib.sha1(V.tobytes() + T.tobytes()).hexdigest())
        if key not in self._mesh_info:
            info = check_mesh(V, T)
            s = np.sqrt(2.0 * info["area"])
            if s.max() > 10.0 * s.min():
                warnings.warn(
                    f"this mesh's elements range from {s.min() * 1e3:.3g} to {s.max() * 1e3:.3g} mm. "
                    f"The solver iterations a step needs follow the SMALLEST element (see "
                    f"ImplicitIntegrator.cfl_number); a few small ones make every step dearer.",
                    stacklevel=4)
            self._mesh_info[key] = info
        return self._mesh_info[key]

    def _is_mesh(self, handle):
        return bool(self._bodies[handle].get("mesh"))

    def _need_pixels(self, handle, what):
        if self._is_mesh(handle):
            raise ValueError(f"body {handle} is a mesh body: {what} needs a pixel body. "
                             f"A mesh body takes per-triangle arrays instead.")

    def set_kinematic_pose(self, handle, translation=(0.0, 0.0), angle=0.0, about=None):
        """Where a kinematic body is to be at the end of the next step: its rest shape
        rotated by `angle` [rad] about `about` (default: the centre of its grid, or of a
        mesh body's bounding box), then translated."""
        self.set_kinematic_target(handle, self.rigid_motion(handle, translation, angle, about))

    # ---- build ----
    def finalize(self):
        """Assemble the scene mesh from all bodies."""
        if not self._bodies:
            raise ValueError("add at least one soft body before finalize()")
        B = self._bodies
        # a mesh body's own nodes must not touch through the barrier at rest: exclude its
        # boundary pairs within this distance through the material (core.mesh._material_excl)
        dx = self.dx
        r_excl = max(2.0 * dx, 16.0 * (self.cfg.ipc_d_hat if self.cfg.ipc_d_hat is not None else dx / 8.0))
        # a stiffness MAP is applied after the build: mesh at unit modulus, then scale each
        # element's mu and lambda by the map (nu is the body's own, so both scale together)
        geom = lambda b: (dict(vertices=b["vertices"], triangles=b["triangles"], excl_radius=r_excl,
                               checked=True) if b.get("mesh") else
                          dict(mask_xy=b["mask"].T.copy(), size=b["size"], center=b["center"]))
        self.me = build_scene([dict(**geom(b),
                                    young=(b["E"] if np.isscalar(b["E"]) else 1.0),
                                    nu=b["nu"], rho=b["rho"], env=b["env"]) for b in B])
        me = self.me
        self._node_range = [np.arange(*me.body_node_range[h]) for h in range(len(B))]
        self._tri_range = [np.arange(*me.body_tri_range[h]) for h in range(len(B))]
        for h, b in enumerate(B):
            if np.isscalar(b["E"]):
                continue
            tris = self._tri_range[h]
            scale = (b["E"] if b.get("mesh") else
                     _act.pixel_field_on_elements(b["E"], me.tri_quad, tris, b["mask"].shape[0]))
            me.mu[tris] = me.mu[tris] * scale
            me.lam[tris] = me.lam[tris] * scale
        self._check_layout()
        self._scale_contact()
        self._targets = me.rest_pos.astype(np_real).copy()
        self._x_host = me.rest_pos.astype(np_real).copy()
        self._v0 = np.zeros_like(self._x_host)
        for h, b in enumerate(B):
            if b.get("affine"):
                self._add_affine(h, self._tri_range[h])
            if b.get("kinematic"):
                self._constrained[h] = self._node_range[h].copy()
        return self

    # ---- rigid regions ----
    def add_affine_region(self, handle, region, E=None, nu=None):
        """Make part of a soft body RIGID — a bone in a soft finger, a stiff link between two
        flexure hinges, a mounting plate. Returns the region's index.

        region  (H, W) boolean map laid out like the mask (row 0 at the bottom). Its solid
                pixels become one rigid piece; the soft material around it stays attached,
                because the two share their interface nodes. For a mesh body: a boolean
                array with one entry per triangle, or the indices of its triangles.
        E, nu   stiffness the piece resists strain with (default model.rigid_young / rigid_nu).

        Call after finalize() and before the first step. Regions must not touch each other:
        leave at least one soft element between two rigid pieces."""
        self._need_mesh()
        if self._solver is not None:
            raise RuntimeError("add rigid regions before the first simulate()")
        elems = self._region_elements(handle, region)
        from .core.mesh import lame
        mu, lam = lame(self.rigid_young if E is None else float(E),
                       self.rigid_nu if nu is None else float(nu))
        self.me.mu[elems] = mu; self.me.lam[elems] = lam
        return self._add_affine(handle, elems)

    def _add_affine(self, handle, elems):
        me = self.me
        mine = self._affine_of.setdefault(handle, [])            # only its own body's can overlap it
        if mine and np.intersect1d(elems, np.concatenate(mine)).size:
            raise ValueError("rigid regions must not overlap")
        mine.append(np.asarray(elems, np.int64))
        nodes = np.unique(np.asarray(me.tri)[elems].reshape(-1))
        # nodes of one rigid piece never move relative to each other: no contact among them
        env = self._bodies[handle]["env"]
        bn = np.asarray(me.boundary_nodes)
        lo, hi = int(me.boundary_starts[env]), int(me.boundary_starts[env + 1])
        loc = np.nonzero(np.isin(bn[lo:hi], nodes))[0]
        ex = me.excluded_pairs_per_env[env]
        ex[np.ix_(loc, loc)] = True
        o = int(me.excl_offset[env])
        me.excl_flat[o:o + ex.size] = ex.reshape(-1).astype(np.uint8)
        self._affine.append(dict(body=handle, elems=np.asarray(elems, np.int64), nodes=nodes))
        return len(self._affine) - 1

    def affine_state(self, index):
        """Where a rigid piece is: dict(position (2,) of its rest centroid now, angle [rad],
        A (2, 2) — rotation times the (tiny) strain)."""
        s = self._solver
        if s is None or getattr(s, "affine", None) is None:
            a = self._affine[index]
            m = np.asarray(self.me.mass)[a["nodes"]]
            c = (m[:, None] * self.me.rest_pos[a["nodes"]]).sum(0) / m.sum()
            return dict(position=c, angle=0.0, A=np.eye(2))
        t, A = s.affine.state()
        return dict(position=t[index], angle=float(np.arctan2(A[index][1, 0] - A[index][0, 1],
                                                              A[index][0, 0] + A[index][1, 1])),
                    A=A[index])

    def _scale_contact(self):
        """Fill in the contact parameters the user left unset, from the scene's own scales."""
        c = self.cfg
        soft = [b for b in self._bodies if not b.get("affine") and not b.get("kinematic")]
        E = min((float(b["E"]) if np.isscalar(b["E"]) else
                 float(b["E"].min()) if b.get("mesh") else float(b["E"][b["mask"] > 0].min())
                 for b in soft), default=1.0e6)
        if c.ipc_d_hat is None: c.ipc_d_hat = self.dx / 8.0
        if c.obstacle_stiffness is None: c.obstacle_stiffness = 4.0e-3 * E * self.dx / c.ipc_d_hat
        if c.contact_stiffness is None: c.contact_stiffness = 0.2 * E
        if self.body_kappa is None: self.body_kappa = 0.1 * E * self.dx / c.ipc_d_hat

    def _check_layout(self):
        """Bodies sharing an environment collide, so they must not start inside one another;
        and their contact has one thickness, so their pixels should be of one size."""
        envs = {}
        for h, b in enumerate(self._bodies):
            envs.setdefault(b["env"], []).append(h)
        memo = {}                                            # a batch repeats the same pair
        for env, hs in envs.items():
            for i, a in enumerate(hs):
                for c in hs[i + 1:]:
                    if self._is_mesh(a) or self._is_mesh(c):
                        key = (self._geom_key(a), self._geom_key(c))
                        if key not in memo:
                            memo[key] = self._overlap_mesh(a, c)
                        hit = memo[key]
                    else:
                        hit = self._overlap(a, c) or self._overlap(c, a)
                    if hit:
                        raise ValueError(
                            f"bodies {a} and {c} overlap in environment {env}. Bodies of one "
                            f"environment collide with each other and must start apart; give "
                            f"the members of a batch their own environment with env_id=i.")
            if any(self._is_mesh(h) for h in hs):
                sz = [self.element_size(h) for h in hs]
                if max(sz) > 1.25 * min(sz):
                    warnings.warn(
                        f"environment {env} mixes element sizes (pixel width, or a mesh body's "
                        f"median boundary edge) from {min(sz) * 1e3:.3g} to {max(sz) * 1e3:.3g} mm. "
                        f"Contact between bodies has ONE thickness, set by the largest, which "
                        f"over-thickens the finer body's own contact.", stacklevel=3)
                continue
            px = [self.pixel_size(h)[0] for h in hs]
            if max(px) > 1.25 * min(px):
                warnings.warn(
                    f"environment {env} mixes pixel sizes from {min(px) * 1e3:.3g} to "
                    f"{max(px) * 1e3:.3g} mm. Contact between bodies has ONE thickness, 1.5 x "
                    f"the largest pixel, which over-thickens the finer body's own contact. "
                    f"Prefer one pixel size per scene.", stacklevel=3)

    def _geom_key(self, h):
        b = self._bodies[h]
        if b.get("mesh"):
            return ("mesh", id(b["vertices"]), id(b["triangles"]))  # one array per distinct mesh
        return ("mask", b["mask"].shape, b["mask"].tobytes(), b["size"], b["center"])

    def _body_geometry(self, h):
        """A body's rest mesh in its own numbering: (nodes, triangles)."""
        me, nodes = self.me, self._node_range[h]
        s, e = int(nodes[0]), int(nodes[-1]) + 1
        return np.asarray(me.rest_pos[s:e], np.float64), np.asarray(me.tri)[self._tri_range[h]] - s

    def _overlap_mesh(self, a, c):
        """Overlap of two bodies at least one of which is a mesh, on their rest meshes."""
        return meshes_overlap(*self._body_geometry(a), *self._body_geometry(c))

    def _overlap(self, a, c):
        """Does the centre of any solid pixel of body a fall in a solid pixel of body c?"""
        A, C = self._bodies[a], self._bodies[c]
        (aw, ah), (ax, ay) = A["size"], A["center"]
        (cw, ch), (cx, cy) = C["size"], C["center"]
        if abs(ax - cx) >= 0.5 * (aw + cw) or abs(ay - cy) >= 0.5 * (ah + ch):
            return False                                    # grids do not even intersect
        H, W = A["mask"].shape
        row, col = np.nonzero(A["mask"])
        px = ax - 0.5 * aw + (col + 0.5) * aw / W
        py = ay - 0.5 * ah + (row + 0.5) * ah / H
        Hc, Wc = C["mask"].shape
        jc = np.floor((px - (cx - 0.5 * cw)) / (cw / Wc)).astype(int)
        jr = np.floor((py - (cy - 0.5 * ch)) / (ch / Hc)).astype(int)
        ok = (jc >= 0) & (jc < Wc) & (jr >= 0) & (jr < Hc)
        return bool(C["mask"][jr[ok], jc[ok]].any())

    # ---- layout queries ----
    @property
    def n_bodies(self): return len(self._bodies)

    @property
    def n_envs(self):
        """Number of environments (independent copies of the scene solved together)."""
        return int(self.me.n_envs) if self.me is not None else \
            (max(b["env"] for b in self._bodies) + 1 if self._bodies else 0)

    @property
    def n_nodes(self): return int(self.me.rest_pos.shape[0])

    @property
    def n_elements(self): return int(self.me.tri.shape[0])

    def body_env(self, handle):
        return self._bodies[handle]["env"]

    def body_nodes(self, handle):
        """Global indices of a body's nodes (rows of get_x())."""
        return self._node_range[handle].copy()

    def body_elements(self, handle):
        """Global indices of a body's elements — rows of a per-element density vector."""
        return self._tri_range[handle].copy()

    def pixel_size(self, handle):
        """(width, height) of one pixel of a body [m]. A mesh body has none: see
        element_size."""
        if self._is_mesh(handle):
            raise ValueError(f"body {handle} is a mesh body: it has no pixels. Its length "
                             f"scale is element_size({handle}).")
        b = self._bodies[handle]; H, W = b["mask"].shape
        return b["size"][0] / W, b["size"][1] / H

    def element_size(self, handle):
        """A body's length scale [m], the one contact is scaled from (model.dx is the
        largest): its pixel width, or a mesh body's median boundary-edge length."""
        b = self._bodies[handle]
        return float(b["h"]) if b.get("mesh") else self.pixel_size(handle)[0]

    def _cfl_length(self, handle):
        """The element size that sets how far a wave travels per element: the smaller
        pixel side, or a mesh body's smallest sqrt(2 x triangle area)."""
        b = self._bodies[handle]
        return float(b["h_cfl"]) if b.get("mesh") else min(self.pixel_size(handle))

    def element_areas(self, handle):
        """Rest area of each of a body's elements [m^2], in body_elements order. On a mesh
        of uneven elements a gradient per element scales with its area: divide by it for a
        density-like sensitivity."""
        return np.asarray(self.me.rest_area, np.float64)[self._tri_range[handle]].copy()

    def element_strain(self, x=None, handle=None, signed=False):
        """Strain of every element (core/fields.py) at positions x (default: now) — or of
        one body's elements, in body_elements order. The von Mises equivalent of the
        Green-Lagrange strain (>= 0); signed=True: the in-plane principal strain of larger
        magnitude, + in tension, - in compression."""
        return self._field("strain", x, handle, signed)

    def element_stress(self, x=None, handle=None, signed=False):
        """Cauchy stress [Pa] of every element at positions x (default: now), with the
        bodies' own moduli — or of one body's, in body_elements order. von Mises (>= 0), or
        with signed=True the in-plane principal stress of larger magnitude, + tension,
        - compression. A density field set on the adjoint is not included."""
        return self._field("stress", x, handle, signed)

    def _field(self, field, x, handle, signed=False):
        from .core.fields import element_field
        v = element_field(self.me, self.get_x() if x is None else x, field, signed)
        return v if handle is None else v[self._tri_range[handle]]

    def rest_x(self, handle):
        """Rest positions of a body's nodes ((n_body_nodes, 2))."""
        return self.me.rest_pos[self._node_range[handle]].copy()

    def to_elements(self, handle, field):
        """Values of an (H, W) pixel field (laid out like the mask) on each of a body's
        elements, in the order of body_elements(handle). A mesh body's fields are per
        triangle already: an (m,) array is checked and returned as it is."""
        b = self._bodies[handle]
        field = np.asarray(field, np.float64)
        if b.get("mesh"):
            m = len(self._tri_range[handle])
            if field.shape != (m,):
                raise ValueError(f"field is {field.shape}; mesh body {handle} has {m} triangles")
            return field.copy()
        if field.shape != b["mask"].shape:
            raise ValueError(f"field is {field.shape}, the mask is {b['mask'].shape}")
        return _act.pixel_field_on_elements(field, self.me.tri_quad, self._tri_range[handle],
                                            b["mask"].shape[0])

    def to_pixels(self, handle, values):
        """Sum per-element values (in body_elements order) back onto the body's (H, W)
        pixel grid — the transpose of to_elements, which is what a gradient needs."""
        self._need_pixels(handle, "to_pixels")
        H, W = self._bodies[handle]["mask"].shape
        q = np.asarray(self.me.tri_quad)[self._tri_range[handle]]
        out = np.zeros((H, W))
        np.add.at(out, (q % H, q // H), np.asarray(values, np.float64))
        return out

    def rigid_motion(self, handle, translation=(0.0, 0.0), angle=0.0, about=None):
        """Rest positions of a body moved as a rigid whole — rotated by `angle` [rad] about
        the point `about` (default: the centre of its grid, or of a mesh body's bounding
        box), then translated. Rows are the body's nodes, ready to be passed (or sliced)
        into set_kinematic_target."""
        x = self.rest_x(handle)
        c = np.asarray(self._bodies[handle]["center"] if about is None else about, np.float64)
        ca, sa = np.cos(angle), np.sin(angle)
        R = np.array([[ca, -sa], [sa, ca]])
        return (x - c) @ R.T + c + np.asarray(translation, np.float64)

    # ---- kinematic boundary conditions ----
    def set_kinematic_constraint(self, handle, node_indices):
        """Mark a body's nodes (body-local indices) as position-controlled."""
        self._constrained[handle] = self._node_range[handle][np.asarray(node_indices, int)]

    def set_kinematic_target(self, handle, targets):
        """Set absolute target positions for a body's nodes ((n_body_nodes, 2))."""
        self._targets[self._node_range[handle]] = np.asarray(targets, np_real)

    def get_body_x(self, handle):
        """Current node positions of a body ((n_body_nodes, 2)), synced from the solver."""
        return self.get_x()[self._node_range[handle]]

    def get_x(self):
        """Current positions of ALL nodes ((N, 2), every body) — the state the adjoint
        checkpoints. Returns a copy."""
        if self._solver is not None:
            self._x_host = self._solver.x.numpy()
        return self._x_host.copy()

    def get_body_v(self, handle):
        """Current node velocities of a body ((n_body_nodes, 2))."""
        v = self._v0 if self._solver is None else self._solver.v.numpy()
        return np.asarray(v)[self._node_range[handle]].copy()

    def set_body_velocity(self, handle, velocity=(0.0, 0.0), omega=0.0, about=None):
        """Give a body a rigid velocity field: `velocity` [m/s] plus a spin `omega` [rad/s]
        about the point `about` (default: its current centroid — of the nodes of a pixel
        body, of the mass of a mesh body, whose nodes need not be evenly spread). An initial
        condition — a thrown object, a spinning wheel; call it again at any time to
        overwrite."""
        nodes = self._node_range[handle]
        x = self.get_x()[nodes].astype(np.float64)
        if about is not None:
            c = np.asarray(about, np.float64)
        elif self._is_mesh(handle):
            m = np.asarray(self.me.mass, np.float64)[nodes]
            c = (m[:, None] * x).sum(0) / m.sum()
        else:
            c = x.mean(0)
        r = x - c
        v = np.asarray(velocity, np.float64)[None, :] + omega * np.stack([-r[:, 1], r[:, 0]], 1)
        self._v0[nodes] = v
        if self._solver is not None:
            full = self._solver.v.numpy(); full[nodes] = v
            self._solver.v.assign(full.astype(np_real))

    def set_body_x(self, handle, x, v=None):
        """Put a body's nodes at x ((n_body_nodes, 2)), and give them the velocities v if
        any — a state to start from, or to return to. Its kinematic targets follow, so a
        constrained node stays where it is put. Nothing checks that the body does not land
        in another one: contact assumes the bodies apart."""
        nodes = self._node_range[handle]
        x = np.asarray(x, np_real).reshape(len(nodes), 2)
        self._x_host[nodes] = x; self._targets[nodes] = x
        if v is not None:
            self._v0[nodes] = np.asarray(v, np_real).reshape(len(nodes), 2)
        s = self._solver
        if s is not None:
            s.set_nodes(nodes[0], x, None if v is None else self._v0[nodes])
            self._sync_rigid([i for i, a in enumerate(self._affine) if a["body"] == handle])

    def set_body_pose(self, handle, translation=(0.0, 0.0), angle=0.0, about=None,
                      velocity=(0.0, 0.0), omega=0.0):
        """Put a body somewhere as a rigid image of its rest shape — rotated by `angle` [rad]
        about `about` (default: the centre of its grid, or of a mesh body's bounding box),
        then translated — moving with `velocity` [m/s] and spinning at `omega` [rad/s] about
        its centroid."""
        self.set_body_x(handle, self.rigid_motion(handle, translation, angle, about))
        self.set_body_velocity(handle, velocity, omega)

    def _sync_rigid(self, regions):
        s = self._solver
        if s is not None and getattr(s, "affine", None) is not None and len(regions):
            s.affine.sync_from_x(s, regions=regions)

    def reset(self, env_ids=None):
        """Return every body to rest at zero velocity, kinematic targets included, so a
        rollout can be repeated on the same scene — a new density field, a finite-
        difference check, the next optimisation iterate. env_ids: only these environments
        (an index or a list); the others are not touched. Actuator commands (pressures,
        tensions, activations) are NOT touched either: set them again if the rollout
        should start from different ones."""
        rest = self.me.rest_pos.astype(np_real)
        s = self._solver
        if env_ids is None:
            self._targets = rest.copy(); self._x_host = rest.copy()
            self._v0 = np.zeros_like(rest)
            self._epoch += 1
            self.time = 0.0
            if s is not None:
                s.x.assign(rest); s.v.zero_(); s.xn.zero_()
                if getattr(s, "affine", None) is not None:
                    s.affine.sync_from_x(s)
            return
        pick = np.zeros(self.n_envs, np.int32); pick[np.atleast_1d(env_ids)] = 1
        nodes = pick[np.asarray(self.me.node2env)] != 0
        self._targets[nodes] = rest[nodes]; self._x_host[nodes] = rest[nodes]
        self._v0[nodes] = 0.0
        self._epoch += 1
        if s is not None:
            s.reset_envs(pick)
            self._sync_rigid([i for i, a in enumerate(self._affine)
                              if pick[self._bodies[a["body"]]["env"]]])

    # ================================================================== actuation
    # Besides prescribing node positions (set_kinematic_target), a body can be driven by
    # fluid pressure in its sealed cavities, by tendons, by actively deforming material and
    # by external forces. Add actuators after finalize(); command them every step.
    #
    # UNITS. Plane strain, per unit depth: pressure in Pa, forces and tendon tensions in
    # N per metre of out-of-plane depth. For a part of thickness w, a physical force F is
    # F / w here.
    def _need_mesh(self):
        if self.me is None:
            raise RuntimeError("call finalize() before adding actuators")

    def _node_of(self, handle):
        return lambda ix, iy: self.me.body_node_index(handle, int(ix), int(iy))

    # ---- gravity / external force ----
    @property
    def gravity(self):
        """Gravitational acceleration (gx, gy) [m/s^2]; (0, 0) by default."""
        return self._gravity

    @gravity.setter
    def gravity(self, g):
        self._gravity = (float(g[0]), float(g[1]))
        self._ctrl_version += 1

    def set_external_force(self, handle, force):
        """Nodal forces on a body, (n_body_nodes, 2) [N/m], held until changed."""
        self._need_mesh()
        if self._fext is None:
            self._fext = np.zeros((self.me.rest_pos.shape[0], 2), np_real)
        self._fext[self._node_range[handle]] = np.asarray(force, np_real)
        self._ctrl_version += 1

    # ---- pneumatic / hydraulic: pressure in sealed cavities ----
    def add_pressure_chambers(self, handle, min_pixels=1):
        """Make every sealed cavity of a body a pressure chamber. Returns their handles,
        ordered bottom-to-top then left-to-right by the cavity's lowest pixel.

        A cavity is a void region of the mask that is enclosed by solid — it does not reach
        the edge of the pixel grid. A channel open to the outside is vented, not a chamber.
        On a mesh body every hole in the mesh is a chamber (min_pixels does not apply),
        ordered by its lowest point, then left to right."""
        self._need_mesh()
        if self._is_mesh(handle):
            holes = [l for l in self._mesh_loops(handle) if l[2] > 0]
            rest = self.me.rest_pos
            low = lambda l: tuple(rest[l[0][np.argmin(rest[l[0], 1])]][::-1])   # (y, x) of its lowest node
            return [self._register_chamber(handle, ea, eb, A, sign=1.0)
                    for ea, eb, A in sorted(holes, key=low)]
        mask = self._bodies[handle]["mask"]
        lab, ids = _act.enclosed_voids(mask)
        ids = [k for k in ids if (lab == k).sum() >= min_pixels]
        ids.sort(key=lambda k: tuple(np.argwhere(lab == k)[0]))
        return [self._add_chamber(handle, lab, k) for k in ids]

    def add_pressure_chamber(self, handle, at):
        """Make the sealed cavity containing the world point `at` a pressure chamber — on a
        mesh body, the hole in the mesh that contains it."""
        self._need_mesh()
        if self._is_mesh(handle):
            rest = np.asarray(self.me.rest_pos, np.float64)
            for ea, eb, A in self._mesh_loops(handle):
                if A > 0 and _act.point_in_loop(at, rest, ea, eb):
                    return self._register_chamber(handle, ea, eb, A, sign=1.0)
            raise ValueError(f"point {tuple(at)} is not inside a hole of this mesh body")
        b = self._bodies[handle]; mask = b["mask"]
        px = _act.pixel_of(at, b["size"], b["center"], mask.shape)
        if px is None or mask[px] > 0:
            raise ValueError(f"point {tuple(at)} is not inside a void of this body")
        lab, ids = _act.enclosed_voids(mask)
        if lab[px] not in ids:
            raise ValueError(f"the void at {tuple(at)} is open to the outside, not a sealed cavity")
        return self._add_chamber(handle, lab, int(lab[px]))

    def add_ambient_pressure(self, handle):
        """Let a pressure act on a body's OUTER surface — its outline, including any channel
        open to the outside. Returns a handle used with set_pressure like a chamber's.
        Positive pressure squeezes the body (immersion, a pressurised enclosure); raising it
        together with the cavities' pressure leaves only their difference to do work, which
        is how a vacuum-driven actuator is set up. On a mesh body: its outline, every
        boundary of the mesh that is not a hole."""
        self._need_mesh()
        if self._is_mesh(handle):
            out = [l for l in self._mesh_loops(handle) if l[2] < 0]
            return self._register_chamber(handle, np.concatenate([l[0] for l in out]),
                                          np.concatenate([l[1] for l in out]),
                                          -sum(l[2] for l in out), sign=-1.0)
        mask = self._bodies[handle]["mask"]
        lab, ids = _act.enclosed_voids(mask)
        ea, eb = _act.exterior_edges(mask, lab, ids, self._node_of(handle))
        n_px = int((mask > 0).sum() + sum((lab == k).sum() for k in ids))
        pw, ph = self.pixel_size(handle)
        return self._register_chamber(handle, ea, eb, n_px * pw * ph, sign=-1.0)

    def _add_chamber(self, handle, lab, label):
        ea, eb = _act.chamber_edges(self._bodies[handle]["mask"], lab, label, self._node_of(handle))
        pw, ph = self.pixel_size(handle)
        return self._register_chamber(handle, ea, eb, int((lab == label).sum()) * pw * ph, sign=1.0)

    def _mesh_loops(self, handle):
        """A mesh body's boundary loops, fluid on the left, in global node ids:
        [(ea, eb, signed area)] — holes > 0, outline < 0. Checked: the outline encloses
        the material and the holes, and nothing else."""
        nodes = self._node_range[handle]; s0 = int(nodes[0])
        tri = np.asarray(self.me.tri)[self._tri_range[handle]] - s0
        rest = np.asarray(self.me.rest_pos, np.float64)[nodes]
        loops = [(ea + s0, eb + s0, A) for ea, eb, A in _act.mesh_boundary_loops(tri, rest)]
        outline = -sum(A for _, _, A in loops if A < 0)
        inside = self.element_areas(handle).sum() + sum(A for _, _, A in loops if A > 0)
        if abs(outline - inside) > 1e-9 * outline:
            raise RuntimeError(f"mesh body {handle}: its outline encloses {outline:.9e} m^2, its "
                               f"material and holes {inside:.9e}")
        return loops

    def _register_chamber(self, handle, ea, eb, expect, sign):
        """sign = +1: the edges run counter-clockwise round a cavity (shoelace area > 0).
        sign = -1: they run clockwise round the body (the fluid is the surroundings).
        expect: the area [m^2] they must enclose — from the pixels, or a mesh hole's own."""
        rest = self.me.rest_pos
        ref = rest[np.concatenate([ea, eb])].mean(0)
        got = sign * _act.loop_area(rest, ea, eb, ref)
        if not expect > 0 or abs(got - expect) > 1e-6 * expect:     # orientation / closure guard
            raise RuntimeError(f"pressure boundary encloses {got:.6e} m^2, expected {expect:.6e}")
        self._chambers.append(dict(body=handle, ea=ea, eb=eb, ref=ref, area0=float(expect),
                                   sign=sign, pressure=0.0))
        self._act_version += 1
        return len(self._chambers) - 1

    def set_pressure(self, chamber, pressure):
        """Gauge pressure of the fluid in a chamber [Pa]. Positive inflates."""
        self._chambers[chamber]["pressure"] = float(pressure)
        self._ctrl_version += 1

    def chamber_area(self, chamber):
        """Current area enclosed by a pressure boundary [m^2] — a cavity's volume per unit
        depth; for an ambient-pressure handle, the area inside the body's outline."""
        c = self._chambers[chamber]
        return c["sign"] * _act.loop_area(self.get_x(), c["ea"], c["eb"], c["ref"])

    # ---- tendons / cables ----
    def add_tendon(self, handle, through=None, nodes=None, stiffness=None):
        """Thread a cable through a body. Returns a handle.

        through    waypoints [(x, y), ...] in the rest configuration; each snaps to the
                   nearest mesh node, which becomes a frictionless eyelet.
        nodes      or give the eyelet nodes directly (body-local indices).
        stiffness  None: the cable is TENSION-controlled — command it with
                   set_tendon_tension. A number k [N/m per metre of stretch, i.e. EA / L0]:
                   it is LENGTH-controlled — a motor holds its free length, set with
                   set_tendon_length, and the cable pulls with k (L - L0) but cannot push.

        Anchor an end by putting its eyelet on a kinematically constrained node."""
        self._need_mesh()
        ids = self._node_range[handle]
        if (through is None) == (nodes is None):
            raise ValueError("give exactly one of `through` (points) or `nodes` (indices)")
        path = ids[np.asarray(nodes, int)] if nodes is not None else \
            _act.snap_path(through, self.me.rest_pos[ids], ids)
        if len(path) < 2:
            raise ValueError("a tendon needs at least two distinct eyelets")
        if stiffness is not None and stiffness <= 0:
            raise ValueError("stiffness must be positive (or None for a tension-controlled cable)")
        self._tendons.append(dict(body=handle, nodes=np.asarray(path, np.int64),
                                  stiffness=0.0 if stiffness is None else float(stiffness),
                                  tension=0.0,
                                  length=_act.path_length(self.me.rest_pos, path)))
        self._act_version += 1
        return len(self._tendons) - 1

    def set_tendon_tension(self, tendon, tension):
        """Tension of a tension-controlled cable [N/m]. Negative values are treated as 0."""
        t = self._tendons[tendon]
        if t["stiffness"] > 0:
            raise ValueError("this tendon is length-controlled; use set_tendon_length")
        t["tension"] = float(tension)
        self._ctrl_version += 1

    def set_tendon_length(self, tendon, length):
        """Free length a motor holds a length-controlled cable at [m]."""
        t = self._tendons[tendon]
        if t["stiffness"] <= 0:
            raise ValueError("this tendon is tension-controlled; use set_tendon_tension")
        t["length"] = float(length)
        self._ctrl_version += 1

    def tendon_length(self, tendon):
        """Current path length of a cable through its eyelets [m]."""
        return _act.path_length(self.get_x(), self._tendons[tendon]["nodes"])

    def tendon_tension(self, tendon):
        """Current tension in a cable [N/m] (zero when a length-controlled cable is slack)."""
        t = self._tendons[tendon]
        if t["stiffness"] > 0:
            return t["stiffness"] * max(self.tendon_length(tendon) - t["length"], 0.0)
        return max(t["tension"], 0.0)

    # ---- springs, pins and joints ----
    def add_spring(self, handle_a, point_a, handle_b=None, point_b=None, stiffness=None,
                   rest_length=None):
        """A linear spring from a point of one body to a point of another — or, with
        handle_b=None, to a point fixed in the world. Returns a handle.

        point_a, point_b  rest-configuration points; each end snaps to the nearest node of
                          its body. For a world spring point_b is the anchor (default: where
                          point_a snapped to).
        stiffness         k [Pa]: force per metre of stretch, per unit depth. Default
                          model.rigid_young, i.e. effectively inextensible.
        rest_length       None: the spring is relaxed as built. Between two bodies it must
                          leave them apart; to the world it may be 0 — a PIN, see add_pin.

        Command it with set_spring_length (a linear actuator) and, for a world spring,
        set_spring_anchor (a moving support)."""
        self._need_mesh()
        ia = self._nearest(handle_a, point_a)
        if handle_b is None:
            ib = -1
            anchor = np.asarray(self.me.rest_pos[ia] if point_b is None else point_b, np.float64)
        else:
            ib = self._nearest(handle_b, point_a if point_b is None else point_b)
            if self.body_env(handle_a) != self.body_env(handle_b):
                raise ValueError("a spring cannot join bodies of different environments")
            if ib == ia:
                raise ValueError("both ends of the spring snapped to the same node")
            anchor = np.asarray(self.me.rest_pos[ib], np.float64)
        L = float(np.linalg.norm(self.me.rest_pos[ia] - anchor)) if rest_length is None \
            else float(rest_length)
        if handle_b is not None and L <= 0.0:
            raise ValueError("a spring between two bodies needs a rest length: at zero it would "
                             "pull two nodes to one place while contact holds the bodies apart")
        k = float(self.rigid_young if stiffness is None else stiffness)
        if k <= 0 or L < 0:
            raise ValueError("spring stiffness must be positive and its rest length non-negative")
        self._springs.append(dict(a=int(ia), b=int(ib), stiffness=k, length=L, anchor=anchor,
                                  env=self.body_env(handle_a)))
        self._act_version += 1
        return len(self._springs) - 1

    def add_pin(self, handle, point, handle_b=None, stiffness=None):
        """Pin a point of a body to the world where it is: a pivot — the body can still
        rotate about it. A zero-length spring of very high stiffness.

        There is no pin between two bodies (handle_b): it would hold two nodes at one place
        while contact holds the bodies apart. Join two rigid parts by a flexure inside one
        body (add_affine_region), or by a spring with a rest length (add_spring)."""
        if handle_b is not None:
            raise NotImplementedError(
                "a pin between two bodies is not supported: it would hold two nodes at one "
                "place while contact holds the bodies apart. Pin to the world (leave handle_b "
                "out), join rigid parts by a flexure inside one body (add_affine_region), or "
                "use add_spring with a rest length.")
        return self.add_spring(handle, point, None, point, stiffness, rest_length=0.0)

    def set_spring_length(self, spring, length):
        """Rest length of a spring [m] — drive it to make a linear actuator."""
        self._springs[spring]["length"] = float(length)
        self._ctrl_version += 1

    def set_spring_anchor(self, spring, point):
        """Move the world end of a world spring / pin."""
        if self._springs[spring]["b"] >= 0:
            raise ValueError("this spring joins two bodies and has no world anchor")
        self._springs[spring]["anchor"] = np.asarray(point, np.float64)
        self._ctrl_version += 1

    def spring_force(self, spring):
        """Tension in a spring [N/m] (negative in compression) and its current length [m]."""
        sp = self._springs[spring]
        x = self.get_x().astype(np.float64)
        ln = float(np.linalg.norm(x[sp["a"]] - (x[sp["b"]] if sp["b"] >= 0 else sp["anchor"])))
        return sp["stiffness"] * (ln - sp["length"]), ln

    def _nearest(self, handle, point):
        ids = self._node_range[handle]
        d = np.linalg.norm(self.me.rest_pos[ids] - np.asarray(point, np.float64)[None, :], axis=1)
        return int(ids[d.argmin()])

    # ---- active material ----
    def add_active_region(self, handle, region, fiber=None):
        """Mark part of a body as actively deforming material. Returns a handle.

        region  (H, W) boolean map laid out like the mask (row 0 at the bottom). For a mesh
                body: a boolean array with one entry per triangle, or the indices of its
                triangles.
        fiber   None for isotropic expansion/contraction (thermal, swelling); an angle in
                radians or a 2-vector for a direction the material shortens or lengthens
                along (shape-memory wire, muscle, a poled dielectric elastomer)."""
        self._need_mesh()
        elems = self._region_elements(handle, region)
        mine = self._regions_of.setdefault(handle, [])
        if mine and np.intersect1d(elems, np.concatenate(mine)).size:
            raise ValueError("active regions must not overlap")
        mine.append(np.asarray(elems, np.int64))
        self._regions.append(dict(body=handle, elems=elems, fiber=fiber, A=np.eye(2)))
        self._act_version += 1
        return len(self._regions) - 1

    def set_activation(self, region, stretch, transverse=1.0):
        """Rest-length ratio of an active region: 1 is passive, 0.9 a 10 % contraction, 1.05
        a 5 % expansion. With a fiber direction, `stretch` applies along the fiber and
        `transverse` across it; without one the change is isotropic."""
        r = self._regions[region]
        if stretch <= 0 or transverse <= 0:
            raise ValueError("stretch ratios must be positive")
        r["A"] = _act.activation_matrix(stretch, transverse, r["fiber"])
        self._ctrl_version += 1

    # ---- helpers ----
    def all_constrained(self):
        if not self._constrained:
            return np.zeros(0, np.int64)
        return np.concatenate([self._constrained[h] for h in sorted(self._constrained)])

    def _region_elements(self, handle, region):
        """Global element ids of part of a body: an (H, W) pixel map for a pixel body; a
        boolean per triangle, or triangle indices, for a mesh body."""
        b = self._bodies[handle]
        if not b.get("mesh"):
            region = np.asarray(region) > 0
            shape = b["mask"].shape
            if region.shape != shape:
                raise ValueError(f"region is {region.shape}, the mask is {shape}")
            elems = _act.region_elements(region, self.me.tri_quad, self._tri_range[handle], shape[0])
            if len(elems) == 0:
                raise ValueError("the region contains no solid pixel of this body")
            return elems
        tris = self._tri_range[handle]; m = len(tris)
        r = np.asarray(region)
        if r.dtype == bool:
            if r.shape != (m,):
                raise ValueError(f"region is a boolean array of shape {r.shape}; mesh body {handle} "
                                 f"has {m} triangles: give ({m},) booleans or triangle indices")
            loc = np.nonzero(r)[0]
        else:
            if r.ndim != 1 or not np.issubdtype(r.dtype, np.integer):
                raise ValueError(f"region of a mesh body: ({m},) booleans or a 1-D array of triangle "
                                 f"indices; got {r.dtype} of shape {r.shape}")
            loc = np.unique(r)
            if len(loc) != len(r) or (len(loc) and (loc[0] < 0 or loc[-1] >= m)):
                raise ValueError(f"triangle indices must be distinct and in 0..{m - 1}")
        if len(loc) == 0:
            raise ValueError("the region contains no triangle of this body")
        return tris[loc]

    @property
    def dx(self):
        """Element size [m] — a pixel's width, a mesh body's median boundary edge — the
        largest over the bodies, which is what sets the thickness of contact between
        bodies."""
        return max(self.element_size(h) for h in range(len(self._bodies)))
