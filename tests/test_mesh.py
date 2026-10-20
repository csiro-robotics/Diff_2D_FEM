"""Bodies given as triangle meshes (add_soft_mesh and its rigid / kinematic counterparts).

  1. A mesh body IS a pixel body when it is given the same triangles: built with
     mask_to_mesh, a scene of a soft body with a stiffness map and a rigid block on the
     ground reproduces the pixel scene bit for bit — the mesh, every contact scale, and the
     trajectory. With cavities and an outer pressure the boundaries are the same edges.
  2. Nothing depends on orientation: a hanging bar turned 30 degrees, with gravity turned
     with it, hangs as the upright one does; a turned block rests on a turned plane with its
     weight on it.
  3. Unstructured meshes: a cantilever of jittered triangles under its own weight against
     the double-precision Newton reference on the same mesh; the adjoint against finite
     differences on a jittered block.
  4. Graded meshes keep their own nodes apart: with element sizes spanning 100:1, the
     hop-count exclusion alone would leave boundary pairs inside the barrier's reach at
     rest; the exclusion through the material does not.
  5. Mixed scenes and batches: a mesh block resting on a pixel slab gives the ground both
     weights; a mesh design in a batch reproduces itself alone, bit for bit.
  6. Regions and pressure on a mesh: active growth and a rigid part addressed by triangle;
     a ring meshed in polar coordinates under internal pressure against Lame, and the
     hydrostatic state.
  7. Mass, and everything that is refused: malformed meshes, overlapping bodies, pixel-only
     calls on a mesh body, a pressure boundary with a pinch point.

    python tests/test_mesh.py
"""
import os, sys, warnings
import numpy as np
import warp as wp
import Diff_2D_FEM as D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_static as REF                                          # noqa: E402

DEV = "cuda" if wp.is_cuda_available() else "cpu"
DOUBLE = D.PRECISION == "double"
E, NU, RHO, G0 = 1.0e6, 0.40, 1100.0, 9.81
EP = E / (1 - NU * NU)
PX = 2.0 ** -10                     # ~0.98 mm: dyadic, so pixel and mesh lengths are exact


def _rot(V, th, about=(0.0, 0.0)):
    c, s = np.cos(th), np.sin(th)
    return (np.asarray(V) - about) @ np.array([[c, -s], [s, c]]).T + about


def _jittered(nx, ny, size, origin=(0.0, 0.0), jitter=0.25, seed=0):
    """A criss-cross mesh of an (nx x ny)-cell rectangle with its interior nodes moved at
    random: triangles of every shape and size, a straight outline."""
    rng = np.random.default_rng(seed)
    xs, ys = np.linspace(0, size[0], nx + 1), np.linspace(0, size[1], ny + 1)
    V = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    inner = ((V[:, 0] > 0) & (V[:, 0] < size[0]) & (V[:, 1] > 0) & (V[:, 1] < size[1]))
    V[inner] += jitter * np.array([size[0] / nx, size[1] / ny]) * rng.uniform(-1, 1, (inner.sum(), 2))
    n = lambda i, j: i * (ny + 1) + j
    T = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = n(i, j), n(i + 1, j), n(i + 1, j + 1), n(i, j + 1)
            T += [(a, b, c), (a, c, d)] if (i + j) % 2 else [(a, b, d), (b, c, d)]
    return V + np.asarray(origin, np.float64), np.array(T)


def _polar_ring(ri, ro, n_t, n_r=None):
    """A ring meshed in polar coordinates. n_r=None: radii in geometric progression with
    square-ish elements, so element size grows from the inner to the outer edge."""
    if n_r is None:
        q = 1.0 + 2.0 * np.pi / n_t
        n_r = int(np.ceil(np.log(ro / ri) / np.log(q)))
        r = ri * (ro / ri) ** np.linspace(0.0, 1.0, n_r + 1)
    else:
        r = np.linspace(ri, ro, n_r + 1)
    th = 2.0 * np.pi * np.arange(n_t) / n_t
    V = np.array([(rk * np.cos(t), rk * np.sin(t)) for rk in r for t in th])
    T = []
    for k in range(n_r):
        for j in range(n_t):
            a, b = k * n_t + j, k * n_t + (j + 1) % n_t
            c, d = a + n_t, b + n_t
            T += [(a, b, d), (a, d, c)] if (j + k) % 2 else [(a, b, c), (b, d, c)]
    return V, np.array(T)


def _integ(**kw):
    return D.ImplicitIntegrator(**{**dict(max_iter=3000, eps=1e-14, tol_x=2e-9, conv_check=8,
                                          self_collision=False, friction=False), **kw})


def _run(model, integ, dt, n):
    for _ in range(n):
        integ.simulate(model, dt)


# ---------------------------------------------------------------- 1. a mesh body is a pixel body
def _slab_and_block(as_mesh):
    m = np.ones((6, 20), np.uint8); m[4:, 8:12] = 0                    # a notch on top
    Emap = np.full(m.shape, 0.8e6); Emap[:2] = 2.0e6                     # a stiff base layer
    size, center = (20 * PX, 6 * PX), (0.0, 3 * PX + PX / 4)
    blk, bsize, bcenter = np.ones((4, 4), np.uint8), (4 * PX, 4 * PX), (-5 * PX, 10 * PX + PX / 4)   # 2 px above
    model = D.Model(device="cpu"); model.body_friction = 0.3
    model.add_ground(height=0.0, mu=0.6)
    if as_mesh:
        V, T, pix = D.mask_to_mesh(m, size, center)
        a = model.add_soft_mesh(V, T, E=Emap[pix[:, 0], pix[:, 1]], nu=NU, density=RHO)
        Vb, Tb, _ = D.mask_to_mesh(blk, bsize, bcenter)
        b = model.add_affine_mesh(Vb, Tb, density=600.0)
    else:
        a = model.add_soft_body(m, physical_size=size, center=center, E=Emap, nu=NU, density=RHO)
        b = model.add_affine_body(blk, physical_size=bsize, center=bcenter, density=600.0)
    model.finalize(); model.gravity = (0.0, -G0)
    return model, a, b


def test_mesh_body_reproduces_the_pixel_body():
    wp.init()
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                  # nothing to warn about here
        _mesh_body_reproduces_the_pixel_body()


def _mesh_body_reproduces_the_pixel_body():
    (mp, ap, bp), (mm, am, bm) = _slab_and_block(False), _slab_and_block(True)
    for k in ("rest_pos", "tri", "D_m_inv", "rest_area", "mu", "lam", "mass", "boundary_nodes",
              "boundary_edges", "excl_flat", "node_starts", "tri_starts", "boundary_starts"):
        assert np.array_equal(getattr(mp.me, k), getattr(mm.me, k)), f"mesh array {k} differs"
    ip, im = D.ImplicitIntegrator(), D.ImplicitIntegrator()
    for attr in ("dx", "dhat", "kappa", "body_kappa", "self_stiffness"):
        assert getattr(mp, attr) == getattr(mm, attr), f"{attr}: {getattr(mp, attr)} vs {getattr(mm, attr)}"
    assert ip.cfl_number(mp, 1e-3) == im.cfl_number(mm, 1e-3)
    for _ in range(40):
        ip.simulate(mp, 1e-3); im.simulate(mm, 1e-3)
    assert np.linalg.norm(ip.collider_force(mp)[0]) > 1.0, "nothing touched the ground"
    assert np.array_equal(mp.get_x(), mm.get_x()), "the trajectories differ"
    assert np.array_equal(ip.collider_force(mp), im.collider_force(mm))
    assert np.array_equal(ip.body_contact_force(mp, bp), im.body_contact_force(mm, bm))


def _cavity_body(as_mesh):
    m = np.ones((20, 14), np.uint8); m[5:15, 4:10] = 0; m[2:4, 8:12] = 0   # two cavities, off-centre
    size, center = (14 * PX, 20 * PX), (0.03, 0.05)
    model = D.Model(device="cpu"); model.add_disc(center=(0.0, -5.0), radius=0.01); model.damping = 2000.0
    if as_mesh:
        V, T, _ = D.mask_to_mesh(m, size, center)
        h = model.add_soft_mesh(V, T, E=E, nu=NU, density=RHO)
    else:
        h = model.add_soft_body(m, physical_size=size, center=center, E=E, nu=NU, density=RHO)
    model.finalize()
    return model, h


def test_mesh_pressure_boundaries_are_the_pixel_ones():
    wp.init()
    models = []
    for as_mesh in (False, True):
        model, h = _cavity_body(as_mesh)
        ch = model.add_pressure_chambers(h) + [model.add_ambient_pressure(h)]
        assert len(ch) == 3
        for c, p in zip(ch, (4e4, 2e4, 1e4)):
            model.set_pressure(c, p)
        models.append((model, ch))
    (mp, cp), (mm, cm) = models
    for i, j in zip(cp, cm):
        ep = {tuple(e) for e in zip(mp._chambers[i]["ea"], mp._chambers[i]["eb"])}
        em = {tuple(e) for e in zip(mm._chambers[j]["ea"], mm._chambers[j]["eb"])}
        assert ep == em, f"chamber {i}: the mesh's boundary is not the pixel one"
        assert abs(mp.chamber_area(i) - mm.chamber_area(j)) <= 1e-12 * mp.chamber_area(i)
    for model, _ in models:
        _run(model, _integ(max_iter=300), 5e-3, 8)
    dx = np.abs(mp.get_x() - mm.get_x()).max()
    assert dx < 1e-12, f"pressure response differs by {dx:.2e} m"
    assert np.abs(mm.get_x() - mm.me.rest_pos).max() > 1e-5, "the pressure did nothing"


# ---------------------------------------------------------------- 2. orientation
def test_a_turned_bar_hangs_as_an_upright_one():
    """Hanging bar under its own weight, upright and turned by 30 degrees with gravity
    turned with it: the same extension, which is rho g L^2 / (2 E')."""
    wp.init()
    L, w, g, th = 0.060, 0.006, 200.0, np.deg2rad(30.0)
    V, T, _ = D.mask_to_mesh(np.ones((60, 6), np.uint8), (w, L), (0.0, 0.0))
    top = np.where(V[:, 1] > V[:, 1].max() - 1e-9)[0]
    bottom = np.where(V[:, 1] < V[:, 1].min() + 1e-9)[0]
    out = []
    for a in (0.0, th):
        model = D.Model(device="cpu"); model.damping = 2000.0
        model.add_disc(center=(0.0, -5.0), radius=0.01)
        h = model.add_soft_mesh(_rot(V, a), T, E=E, nu=NU, density=RHO)
        model.finalize()
        model.set_kinematic_constraint(h, top)
        model.gravity = tuple(_rot([[0.0, -g]], a)[0])
        _run(model, _integ(tol_x=0.0), 2e-2, 12)                       # every step to convergence
        out.append(_rot(model.get_body_x(h).astype(np.float64), -a))
    ext = -(out[0][bottom, 1] - V[bottom, 1]).mean()
    exact = RHO * g * L * L / (2 * EP)
    assert abs(ext - exact) < 0.02 * exact, f"extension {ext * 1e6:.2f} um, exact {exact * 1e6:.2f} um"
    dev = np.abs(out[1] - out[0]).max()
    assert dev < 1e-6 * ext, f"the turned bar deviates by {dev / ext:.2e} of the extension"


def test_a_turned_block_rests_on_a_turned_plane():
    wp.init()
    th = np.deg2rad(30.0)
    n = np.array([-np.sin(th), np.cos(th)])
    V, T, _ = D.mask_to_mesh(np.ones((8, 8), np.uint8), (8 * PX, 8 * PX), (0.0, 4 * PX + PX / 2))
    model = D.Model(device=DEV); model.damping = 60.0
    model.add_plane(normal=n, origin=(0.0, 0.0), mu=0.0)
    h = model.add_soft_mesh(_rot(V, th), T, E=E, nu=NU, density=RHO)
    model.finalize()
    model.gravity = tuple(-G0 * n)
    integ = D.ImplicitIntegrator(self_collision=False)
    _run(model, integ, 2e-3, 300)
    weight = np.asarray(model.me.mass).sum() * G0
    F = integ.collider_force(model, friction=False)[0]
    assert abs(abs(F @ n) / weight - 1.0) < (1e-3 if DOUBLE else 0.02), f"the plane carries {abs(F @ n) / weight:.5f} of the weight"
    gap = (model.get_body_x(h) @ n).min()
    assert 0.0 < gap < model.dhat, f"the block rests {gap / model.dhat:.3f} dhat above the plane"


# ---------------------------------------------------------------- 3. unstructured meshes
def test_jittered_cantilever_against_the_reference():
    """A 5:1 cantilever of jittered triangles bent by its own weight: the PNCG solve
    reaches the double-precision Newton solution of the same mesh."""
    wp.init()
    V, T = _jittered(30, 6, (0.020, 0.004), origin=(0.0, -0.002), seed=1)
    model = D.Model(device="cpu"); model.damping = 2000.0
    model.add_disc(center=(0.0, -5.0), radius=0.01)
    h = model.add_soft_mesh(V, T, E=E, nu=NU, density=RHO)
    model.finalize()
    left = np.where(V[:, 0] < 1e-12)[0]; tip = np.where(V[:, 0] > 0.020 - 1e-12)[0]
    model.set_kinematic_constraint(h, left)
    g = 40.0
    fext = np.asarray(model.me.mass, np.float64)[:, None] * np.array([0.0, -g])
    xr, res = REF.solve_static(model.me, left, fext=fext)
    assert res < 1e-8, "the reference did not converge"
    model.gravity = (0.0, -g)
    _run(model, _integ(max_iter=2000, tol_x=0.0, alpha_max=1e9), 5e-2, 12)
    x = model.get_x().astype(np.float64)
    d, dr = (x[tip, 1] - V[tip, 1]).mean(), (xr[tip, 1] - V[tip, 1]).mean()
    assert dr < -2e-4, "the load should bend the beam measurably"
    assert abs(d / dr - 1.0) < (1e-6 if DOUBLE else 0.015), f"tip deflection {d * 1e6:.4f} um, reference {dr * 1e6:.4f} um"


def test_adjoint_on_a_jittered_mesh():
    """The test_adjoint harness on a body of jittered triangles: dJ/drho per triangle
    against central finite differences, without and with a disc in contact."""
    wp.init()
    T_, DT = 12, 2.0e-4
    for mode in ("elastic", "disc"):
        model = D.Model(device="cpu")
        model.add_disc(center=(0.05, -1.0 if mode == "elastic" else 0.0335), radius=0.006, mu=0.0)
        V, Tr = _jittered(6, 10, (0.012, 0.024), origin=(0.044, 0.040), seed=2)
        h = model.add_soft_mesh(V, Tr, E=E, nu=NU, density=RHO)
        model.finalize()
        x0 = model.get_body_x(h)
        top = np.where(x0[:, 1] > x0[:, 1].max() - 1e-9)[0]
        model.set_kinematic_constraint(h, top)
        integ = D.ImplicitIntegrator(max_iter=3000, eps=1e-14, tol_x=1e-8, conv_check=8,
                                     self_collision=False, friction=False)
        adj = integ.adjoint(model, DT)
        assert adj.n_tri == len(Tr)
        rng = np.random.default_rng(0)
        rho0 = 0.45 + 0.5 * rng.random(adj.n_tri)
        low = np.where(x0[:, 1] < np.median(x0[:, 1]))[0]
        c = np.zeros_like(model.get_x(), dtype=np.float64); c[low, 0] = 1.0 / len(low)

        def rollout(rho):
            adj.set_density(rho); model.reset()
            xs = [model.get_x().astype(np.float64)]
            for t in range(1, T_ + 1):
                x = model.get_body_x(h)
                x[top, 1] = x0[top, 1] - 0.003 * t / T_; x[top, 0] = x0[top, 0] + 0.002 * t / T_
                model.set_kinematic_target(h, x)
                integ.simulate(model, DT)
                xs.append(model.get_x().astype(np.float64))
            return xs
        J = lambda rho: float((c * rollout(rho)[-1]).sum())
        xs = rollout(rho0)
        if mode == "disc":
            assert np.linalg.norm(integ.collider_force(model, friction=False)[0]) > 1e-2, "no contact"
        g = adj.transient_adjoint(xs, seed={T_: c}, use_disc=(mode == "disc"))
        assert np.isfinite(g).all() and np.linalg.norm(g) > 0
        d = g / np.abs(g).max()
        fd = (J(rho0 + 0.05 * d) - J(rho0 - 0.05 * d)) / 0.1
        assert abs(g @ d - fd) < 0.02 * abs(fd), f"{mode}: adjoint {g @ d:.5e}, FD {fd:.5e}"
        for _ in range(2):
            d = rng.standard_normal(adj.n_tri); d /= np.abs(d).max()
            fd = (J(rho0 + 0.05 * d) - J(rho0 - 0.05 * d)) / 0.1
            nerr = abs(g @ d - fd) / (np.linalg.norm(g) * np.linalg.norm(d))
            assert nerr < 0.01, f"{mode}: adjoint {g @ d:.5e}, FD {fd:.5e}, normalised error {nerr:.1e}"


# ---------------------------------------------------------------- 4. graded meshes
def _active_gap(me, excl):
    """Smallest distance between a boundary node and a boundary edge that IPC would let
    act on each other (the node excluded from neither end of the edge)."""
    bn = np.asarray(me.boundary_nodes); pos = {int(n): k for k, n in enumerate(bn)}
    X = np.asarray(me.rest_pos, np.float64)
    best = np.inf
    for a, b in np.asarray(me.boundary_edges):
        ok = ~excl[:, pos[int(a)]] & ~excl[:, pos[int(b)]]
        P = X[bn[ok]]; t = X[b] - X[a]
        s = np.clip(((P - X[a]) @ t) / (t @ t), 0.0, 1.0)
        if len(P):
            best = min(best, np.linalg.norm(P - (X[a] + s[:, None] * t), axis=1).min())
    return best


def test_a_graded_mesh_keeps_its_own_nodes_apart():
    wp.init()
    V, T = _polar_ring(0.0001, 0.010, n_t=32)                            # elements from 0.02 to 2 mm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                                  # the 100:1 size warning
        model = D.Model(device="cpu")
        h = model.add_soft_mesh(V, T, E=E, nu=NU, density=RHO)
    model.finalize()
    hop_only = D.build_scene([dict(vertices=V, triangles=T, young=E, nu=NU, rho=RHO, env=0)])
    nb = len(hop_only.boundary_nodes)
    assert _active_gap(hop_only, hop_only.excluded_pairs_per_env[0].reshape(nb, nb)) < model.dhat, \
        "this mesh should defeat a hop-count exclusion (else the test proves nothing)"
    assert _active_gap(model.me, model.me.excluded_pairs_per_env[0]) > model.dhat
    integ = D.ImplicitIntegrator()                                       # IPC on: self-contact active
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                                  # 0.02 mm elements: CFL above the cap
        integ.simulate(model, 1e-3)
    assert not integ.contact_force(model, h).any(), "the body pushes on itself at rest"
    assert np.abs(model.get_x() - model.me.rest_pos).max() < 1e-12


# ---------------------------------------------------------------- 5. mixed scenes and batches
def test_a_mesh_block_rests_on_a_pixel_slab():
    wp.init()
    model = D.Model(device=DEV); model.damping = 60.0
    model.add_ground(height=0.0, mu=0.6)
    slab = model.add_soft_body(np.ones((4, 16), np.uint8), physical_size=(16 * PX, 4 * PX),
                               center=(0.0, 2 * PX + 0.9 * PX / 8), E=E, nu=NU, density=RHO)
    V, T = _jittered(8, 8, (8 * PX, 8 * PX), origin=(-4 * PX, 4 * PX + 0.6 * PX), seed=3)
    block = model.add_soft_mesh(V, T, E=E, nu=NU, density=RHO)
    model.finalize(); model.gravity = (0.0, -G0)
    integ = D.ImplicitIntegrator()
    _run(model, integ, 2e-3, 300)
    gap = model._solver.ipc.min_distance(model._solver)[0]
    assert 0.0 < gap < model.dhat, f"the block rests {gap / model.dhat:.3f} dhat above the slab"
    assert np.abs(model.get_body_v(block)).max() < 2e-3, "the block has not come to rest"
    weight = np.asarray(model.me.mass).sum() * G0
    Fy = abs(integ.collider_force(model, friction=False)[0][1])
    assert abs(Fy / weight - 1.0) < (1e-3 if DOUBLE else 0.02), f"the ground carries {Fy / weight:.5f} of the weights"


def test_a_mesh_design_in_a_batch_is_itself():
    wp.init()
    V1, T1 = _jittered(6, 12, (0.012, 0.024), origin=(0.044, 0.0375), seed=4)
    V2, T2 = _jittered(6, 12, (0.012, 0.024), origin=(0.044, 0.0375), seed=5)
    def scene(designs):
        model = D.Model(device="cpu")
        model.add_disc(center=(0.05, 0.030), radius=0.006, mu=0.6)
        hs = [model.add_soft_mesh(V, T, E=E, nu=NU, density=RHO, env_id=i) for i, (V, T) in enumerate(designs)]
        model.finalize()
        integ = D.ImplicitIntegrator()
        for t in range(1, 26):
            for hh in hs:
                X = model.rest_x(hh); top = X[:, 1] > X[:, 1].max() - 1e-9
                tg = model.get_body_x(hh); tg[top, 1] = X[top, 1] - 0.004 * t / 25
                model.set_kinematic_constraint(hh, np.where(top)[0]); model.set_kinematic_target(hh, tg)
            integ.simulate(model, 1e-3)
        return model, hs, integ
    alone, (ha,), ia = scene([(V1, T1)])
    batch, hb, ib = scene([(V1, T1), (V2, T2), (V1, T1)])
    assert np.linalg.norm(ia.collider_force(alone)[0]) > 1.0, "no contact developed"
    for env in (0, 2):
        assert np.array_equal(batch.get_body_x(hb[env]), alone.get_body_x(ha)), f"environment {env} differs"
        assert np.array_equal(ib.collider_force(batch)[env], ia.collider_force(alone)[0])
    assert np.abs(batch.get_body_x(hb[1]) - alone.get_body_x(ha)).max() > 1e-6


# ---------------------------------------------------------------- 6. regions and pressure
def test_regions_by_triangle():
    wp.init()
    V, T = _jittered(5, 8, (0.010, 0.016), origin=(-0.005, -0.008), seed=6)
    model = D.Model(device="cpu"); model.damping = 2000.0
    model.add_disc(center=(0.0, -5.0), radius=0.01)
    h = model.add_soft_mesh(V, T, E=E, nu=NU, density=RHO)
    model.finalize()
    model.set_activation(model.add_active_region(h, np.ones(len(T), bool)), 1.08)
    _run(model, _integ(alpha_max=1e9), 2e-2, 10)
    X, x = model.me.rest_pos.astype(np.float64), model.get_x().astype(np.float64)
    assert np.abs((x - x.mean(0)) - 1.08 * (X - X.mean(0))).max() < 1e-4 * 0.016
    assert model._solver.energy()[0] < 1e-8

    model = D.Model(device="cpu"); model.damping = 300.0
    V, T = _jittered(30, 6, (0.020, 0.004), origin=(0.0, -0.002), seed=7)
    h = model.add_soft_mesh(V, T, E=E, nu=NU, density=RHO)
    model.finalize()
    right = np.nonzero(V[T].mean(1)[:, 0] > 0.012)[0]                   # triangle indices
    model.add_affine_region(h, right)
    model.set_kinematic_constraint(h, np.where(V[:, 0] < 1e-12)[0])
    model.gravity = (0.0, -200.0)
    _run(model, D.ImplicitIntegrator(self_collision=False, friction=False), 2e-3, 100)
    A = model.affine_state(0)["A"]
    assert np.abs(A.T @ A - np.eye(2)).max() < 1e-5, "the rigid part strained"
    assert abs(model.affine_state(0)["angle"]) > 1e-3, "the rigid part should have turned"


def test_a_ring_under_internal_pressure():
    """Lame, plane strain, small strain, on a ring meshed in polar coordinates: the
    outline is the circle's own polygon, not a staircase."""
    wp.init()
    ri, ro, p = 0.012, 0.023, 1.0e-3 * E
    V, T = _polar_ring(ri, ro, n_t=96, n_r=10)
    model = D.Model(device="cpu"); model.damping = 2000.0
    model.add_disc(center=(0.0, -5.0), radius=0.01)
    h = model.add_soft_mesh(V, T, E=E, nu=NU, density=RHO)
    model.finalize()
    ch = model.add_pressure_chamber(h, at=(0.0, 0.0))
    A0 = model.chamber_area(ch)
    model.set_pressure(ch, p)
    _run(model, _integ(max_iter=4000, tol_x=0.0), 5e-3, 25)          # every step to convergence
    a = np.sqrt(A0 / np.pi)
    b = np.sqrt((model.element_areas(h).sum() + A0) / np.pi)
    ua = (1 + NU) / E * p * a * a / (b * b - a * a) * ((1 - 2 * NU) * a + b * b / a)
    dA, exact = model.chamber_area(ch) - A0, 2 * np.pi * a * ua
    assert abs(dA - exact) < 0.005 * exact, f"cavity area change {dA:.4e}, Lame {exact:.4e} ({dA / exact - 1:+.2%})"


def test_hydrostatic_state_on_a_mesh():
    """Equal pressure in the hole and outside: a uniform stretch s with sigma(s) = -p."""
    wp.init()
    p = 6.0e4
    mu, lam = E / (2 * (1 + NU)), E * NU / ((1 + NU) * (1 - 2 * NU))
    sig = lambda s: (mu * (s - 1 / s) + 2 * lam * np.log(s) / s) / s
    lo, hi = 0.5, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if sig(mid) < -p else (lo, mid)
    s_exact = 0.5 * (lo + hi)
    V, T = _polar_ring(0.004, 0.010, n_t=48, n_r=5)
    model = D.Model(device="cpu"); model.damping = 2000.0
    model.add_disc(center=(0.0, -5.0), radius=0.01)
    h = model.add_soft_mesh(V + [0.03, 0.05], T, E=E, nu=NU, density=RHO)
    model.finalize()
    for c in model.add_pressure_chambers(h) + [model.add_ambient_pressure(h)]:
        model.set_pressure(c, p)
    _run(model, _integ(max_iter=2000), 5e-3, 30)
    X, x = model.me.rest_pos.astype(np.float64), model.get_x().astype(np.float64)
    Xc = X - X.mean(0)
    s_fem = float(((x - x.mean(0)) * Xc).sum() / (Xc ** 2).sum())
    assert abs(s_fem - s_exact) < 5e-3 * (1 - s_exact), f"stretch {s_fem:.6f}, exact {s_exact:.6f}"
    assert np.abs((x - x.mean(0)) - s_exact * Xc).max() < 1e-4 * 0.02


# ---------------------------------------------------------------- 7. mass, and what is refused
def test_mass_and_numbering():
    V, T = _jittered(7, 3, (0.014, 0.006), seed=8)
    model = D.Model(device="cpu")
    h = model.add_soft_mesh(V, T, E=E, nu=NU, density=RHO)
    model.finalize()
    assert np.array_equal(model.rest_x(h), V), "vertex i must be node i"
    assert np.array_equal(np.asarray(model.me.tri)[model.body_elements(h)], T), "triangle j must be element j"
    m = np.asarray(model.me.mass)[model.body_nodes(h)].sum()
    assert abs(m / (RHO * 0.014 * 0.006) - 1.0) < 1e-12
    assert abs(model.element_areas(h).sum() / (0.014 * 0.006) - 1.0) < 1e-12


def _refused(fn, words):
    try:
        fn()
    except (ValueError, NotImplementedError) as e:
        assert words in str(e), f"refused, but with: {e}"
    else:
        raise AssertionError(f"accepted; expected a refusal mentioning {words!r}")


def test_what_is_refused():
    wp.init()
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]]) * 1e-3
    sqT = np.array([[0, 1, 2], [0, 2, 3]])
    m = lambda V, T: (lambda: D.Model(device="cpu").add_soft_mesh(V, T))
    _refused(m(np.vstack([sq, [[5e-3, 5e-3]]]), sqT), "belong to no triangle")
    _refused(m(sq, sqT + 1), "0-based")
    _refused(m(sq, np.vstack([sqT, sqT[:1]])), "duplicate")
    _refused(m(sq, np.array([[0, 1, 1], [0, 2, 3]])), "repeat a vertex")
    _refused(m(np.array([[0, 0], [1, 0], [2, 0], [0, 1.0]]), np.array([[0, 1, 2], [0, 1, 3]])), "zero area")
    _refused(m(np.array([[0, 0], [1, 0], [0, 1], [0.3, 0.6]]), np.array([[0, 1, 2], [0, 1, 3]])), "folds")
    _refused(m(np.array([[0, 0], [1, 0], [0, 1], [0, -1], [1, 1.0]]),
               np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])), "shared by 3")
    _refused(m(np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 0], [1, -1.0]]),
               np.array([[0, 1, 2], [0, 2, 3], [0, 4, 5], [4, 1, 5]])), "T-junction")
    _refused(m(np.array([[0, 0], [2, 0], [1, 1], [1.5, -0.5], [0.5, 1.5]]),
               np.array([[0, 1, 2], [0, 3, 4]])), "cross")
    _refused(m(sq, sqT[:, :2]), "(m, 3)")
    _refused(m(np.array([[0, 0], [1, np.nan], [0, 1.0]]), np.array([[0, 1, 2]])), "not finite")
    _refused(lambda: D.Model(device="cpu").add_soft_mesh(sq, sqT, E=np.ones(3)), "2 triangles")

    # overlapping bodies, in every combination, refused at finalize
    V, T = _jittered(4, 4, (0.004, 0.004), seed=9)
    def two(second):
        def go():
            model = D.Model(device="cpu"); model.add_soft_mesh(V, T); second(model); model.finalize()
        return go
    _refused(two(lambda md: md.add_soft_mesh(V, T)), "overlap")                          # coincident
    bar, barT = _jittered(10, 1, (0.010, 0.001), origin=(-0.003, 0.0015), seed=10)
    _refused(two(lambda md: md.add_soft_mesh(_rot(bar, 0.7, (0.002, 0.002)), barT)), "overlap")   # crossing
    _refused(two(lambda md: md.add_soft_body(np.ones((4, 4), np.uint8), physical_size=(0.004, 0.004),
                                             center=(0.003, 0.003))), "overlap")       # mesh and pixel
    def aligned():                                                       # same lattice, a quarter pixel deep:
        model = D.Model(device="cpu")                                    # every crossing passes through a vertex
        model.add_soft_mesh(*D.mask_to_mesh(np.ones((6, 20)), (20 * PX, 6 * PX), (0.0, 3 * PX))[:2])
        model.add_soft_mesh(*D.mask_to_mesh(np.ones((4, 4)), (4 * PX, 4 * PX), (-5 * PX, 8 * PX - PX / 4))[:2])
        model.finalize()
    _refused(aligned, "overlap")
    model = D.Model(device="cpu"); model.add_soft_mesh(V, T); model.add_soft_mesh(V + [0.004, 0.0], T)
    model.finalize()                                                     # touching is not overlapping
    _refused(lambda: D.ImplicitIntegrator().simulate(model, 1e-3), "start apart")

    # pixel-only calls on a mesh body
    model = D.Model(device="cpu"); h = model.add_soft_mesh(V, T); model.finalize()
    _refused(lambda: model.pixel_size(h), "no pixels")
    _refused(lambda: model.to_pixels(h, np.zeros(len(T))), "needs a pixel body")
    _refused(lambda: model.me.body_node_index(h, 0, 0), "no pixel grid")
    _refused(lambda: model.add_active_region(h, np.ones((4, 4), bool)), "booleans")
    _refused(lambda: model.add_affine_region(h, np.array([0, 0, 1])), "distinct")
    _refused(lambda: model.add_pressure_chamber(h, at=(0.002, 0.002)), "not inside a hole")
    assert model.add_pressure_chambers(h) == []                          # no hole, no chamber

    # a pressure boundary through a pinch point: two squares touching at a corner
    Vb = np.vstack([sq, sq + 1e-3]); Tb = np.vstack([sqT, sqT + 4])
    Vb = np.delete(Vb, 4, 0); Tb = np.where(Tb == 4, 2, np.where(Tb > 4, Tb - 1, Tb))   # one shared corner
    model = D.Model(device="cpu"); h = model.add_soft_mesh(Vb, Tb); model.finalize()
    _refused(lambda: model.add_ambient_pressure(h), "two boundary loops meet")


if __name__ == "__main__":
    wp.init()
    test_mesh_body_reproduces_the_pixel_body();      print("a mesh body is the pixel body, bit for bit   OK")
    test_mesh_pressure_boundaries_are_the_pixel_ones(); print("pressure boundaries are the pixel ones      OK")
    test_a_turned_bar_hangs_as_an_upright_one();     print("a turned bar hangs as an upright one        OK")
    test_a_turned_block_rests_on_a_turned_plane();   print("a turned block rests on a turned plane      OK")
    test_jittered_cantilever_against_the_reference(); print("jittered cantilever vs the reference        OK")
    test_adjoint_on_a_jittered_mesh();               print("adjoint vs finite differences on a mesh     OK")
    test_a_graded_mesh_keeps_its_own_nodes_apart();  print("a graded mesh keeps its nodes apart         OK")
    test_a_mesh_block_rests_on_a_pixel_slab();       print("a mesh block rests on a pixel slab          OK")
    test_a_mesh_design_in_a_batch_is_itself();       print("a mesh design in a batch is itself          OK")
    test_regions_by_triangle();                      print("active and rigid regions by triangle        OK")
    test_a_ring_under_internal_pressure();           print("a ring under internal pressure (Lame)       OK")
    test_hydrostatic_state_on_a_mesh();              print("hydrostatic state on a mesh                 OK")
    test_mass_and_numbering();                       print("mass and numbering                          OK")
    test_what_is_refused();                          print("what is refused                             OK")
    print("mesh body tests PASSED")
