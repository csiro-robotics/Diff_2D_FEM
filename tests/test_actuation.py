"""Actuation: pressure, tendons, active material, gravity, per-pixel stiffness.

Three layers, each catching what the one before cannot:

  1. DERIVATIVES. For every term, the gradient is the derivative of its energy, the
     curvature is the derivative of its gradient, and the preconditioner diagonal is the
     true second derivative — by finite differences.
  2. CLOSED FORMS. Cases with an exact answer: a hydrostatic state (uniform pressure inside
     and out compresses any body uniformly), a thick-walled ring under internal pressure,
     a bar hanging under its own weight, free growth of an active body.
  3. A DOUBLE-PRECISION REFERENCE (reference_static.py) for the bending cases, where
     continuum theory and the discretisation differ by several percent and only a solution
     of the same discrete problem says how well the solver did.

Everything runs on the CPU device, which is exactly repeatable, and on problems small
enough that the iteration converges; see the README on what it does not resolve.

    python tests/test_actuation.py
"""
import os, sys
import numpy as np
import warp as wp
import Diff_2D_FEM as D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_static as REF                                          # noqa: E402

f64 = wp.float64
E, NU, RHO = 1.0e6, 0.40, 1100.0
EP = E / (1 - NU * NU)                                                  # plane-strain modulus
FAR = (0.0, -5.0)                                                       # obstacle parked out of reach


def _model(mask, size, center=(0.0, 0.0), stiffness=E, damping=2000.0):
    model = D.Model(device="cpu")
    model.add_disc(center=FAR, radius=0.01)
    model.damping = damping
    h = model.add_soft_body(mask, physical_size=size, center=center, E=stiffness, nu=NU, density=RHO)
    model.finalize()
    return model, h


def _integ(max_iter=3000, **kw):
    return D.ImplicitIntegrator(max_iter=max_iter, eps=1e-14, tol_x=2e-9, conv_check=8,
                                self_collision=False, friction=False, **kw)


def _run(model, integ, dt, n):
    for _ in range(n):
        integ.simulate(model, dt)


# ====================================================================== 1. derivatives
def _cavity_block():
    m = np.ones((20, 14), np.uint8); m[5:15, 4:10] = 0
    return _model(m, (0.014, 0.020), center=(0.03, 0.05))


PATH = [(0.024, 0.041), (0.025, 0.046), (0.0245, 0.051), (0.026, 0.056), (0.031, 0.0595)]


def _harness(model, name):
    integ = D.ImplicitIntegrator(max_iter=1, self_collision=False, friction=False)
    s = integ.attach(model, 1e-3)
    term = s._act[name]
    rng = np.random.default_rng(1)
    rest = model.me.rest_pos.astype(np.float64)
    X = rest - rest.mean(0)
    # a sizeable, non-symmetric deformation: nothing is tested at the trivial rest state
    x0 = rest + 0.045 * X[:, ::-1] * [1, -1] + 2e-4 * np.sin(300 * X) + 1.5e-4 * rng.standard_normal(rest.shape)
    xn = x0 - 2e-4 * rng.standard_normal(rest.shape)

    def setx(x):
        s.x.assign(x.astype(D.np_real)); s.xn.assign(xn.astype(D.np_real))

    def energy(x):
        setx(x); buf = wp.zeros(s.B, dtype=f64, device=s.dev); term.energy(s, buf)
        return float(buf.numpy().sum())

    def grad(x):
        setx(x); s.grad.zero_(); s.diagH.zero_(); term.grad_diag(s)
        return s.grad.numpy().astype(np.float64), s.diagH.numpy().astype(np.float64)

    def pHp(x, p):
        grad(x); s.p.assign(p.astype(D.np_real)); s.pHp.zero_(); term.pHp(s)
        return float(s.pHp.numpy().sum())
    return x0, energy, grad, pHp, rng


def _check_derivatives(model, name, curvature=True):
    x0, energy, grad, pHp, rng = _harness(model, name)
    g, dH = grad(x0)
    assert np.abs(g).max() > 0, f"{name}: zero gradient, nothing is being tested"
    assert dH.min() >= 0.0, f"{name}: negative preconditioner diagonal {dH.min()}"
    d = rng.standard_normal(x0.shape); d /= np.abs(d).max()
    fd = lambda f, eps: (f(x0 + eps * d) - f(x0 - eps * d)) / (2 * eps)
    err = min(abs(fd(energy, e) - (g * d).sum()) / abs(fd(energy, e)) for e in (4e-5, 2e-5, 1e-5))
    assert err < 1e-3, f"{name}: gradient differs from d(energy)/dx by {err:.1e}"
    if not curvature:
        return
    ph = pHp(x0, d)
    err = min(abs((fd(lambda x: grad(x)[0], e) * d).sum() - ph) / abs(ph) for e in (4e-5, 2e-5, 1e-5))
    assert err < 1e-3, f"{name}: p.H.p differs from d(gradient)/dx by {err:.1e}"
    g, dH = grad(x0)
    idx = np.argwhere(dH > 1e-3 * dH.max()) if dH.max() > 0 else []
    for i, c in idx[:: max(1, len(idx) // 12)]:
        e = np.zeros_like(x0); e[i, c] = 1.0
        true = (grad(x0 + 2e-5 * e)[0][i, c] - grad(x0 - 2e-5 * e)[0][i, c]) / 4e-5
        assert abs(true - dH[i, c]) < 2e-3 * abs(true), f"{name}: diagonal {dH[i, c]:.4e} vs {true:.4e}"


def test_pressure_derivatives():
    wp.init()
    model, h = _cavity_block()
    model.set_pressure(model.add_pressure_chambers(h)[0], 3.0e4)
    model.set_pressure(model.add_ambient_pressure(h), 1.1e4)
    _check_derivatives(model, "pressure")


def test_tendon_derivatives():
    wp.init()
    model, h = _cavity_block()                                           # tension-controlled
    model.set_tendon_tension(model.add_tendon(h, through=PATH), 40.0)
    _check_derivatives(model, "tendon")
    model, h = _cavity_block()                                           # length-controlled, taut
    td = model.add_tendon(h, through=PATH, stiffness=5.0e4)
    model.set_tendon_length(td, 0.9 * model.tendon_length(td))
    _check_derivatives(model, "tendon")
    model, h = _cavity_block()                                           # length-controlled, SLACK
    td = model.add_tendon(h, through=[PATH[0], PATH[-1]], stiffness=5.0e4)
    model.set_tendon_length(td, 2.0 * model.tendon_length(td))
    x0, energy, grad, pHp, rng = _harness(model, "tendon")
    g, dH = grad(x0)
    assert np.abs(g).max() == 0 and np.abs(dH).max() == 0 and energy(x0) == 0 \
        and pHp(x0, rng.standard_normal(x0.shape)) == 0, "a slack cable must contribute nothing"
    assert model.tendon_tension(td) == 0.0


def test_external_force_derivatives():
    wp.init()
    model, h = _cavity_block()
    model.set_external_force(h, np.random.default_rng(3).standard_normal((model.me.rest_pos.shape[0], 2)) * 5.0)
    model.gravity = (0.3, -9.81)
    _check_derivatives(model, "force", curvature=False)


# ====================================================================== 2. closed forms
def test_hydrostatic_state_is_exact():
    """The same pressure in every cavity and on the outside gives a uniform Cauchy stress -p
    for ANY geometry, hence a uniform stretch s with sigma(s) = -p. That exercises the
    cavity and the exterior orientation, the magnitude, and the follower nature of the load."""
    wp.init()
    p = 6.0e4
    mu, lam = E / (2 * (1 + NU)), E * NU / ((1 + NU) * (1 - 2 * NU))
    sig = lambda s: (mu * (s - 1 / s) + 2 * lam * np.log(s) / s) / s     # Neo-Hookean, F = s I
    lo, hi = 0.5, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if sig(mid) < -p else (lo, mid)
    s_exact = 0.5 * (lo + hi)
    m = np.ones((20, 14), np.uint8); m[5:15, 4:10] = 0; m[2:4, 8:12] = 0  # two cavities, off-centre
    model, h = _model(m, (0.014, 0.020), center=(0.03, 0.05))
    chambers = model.add_pressure_chambers(h)
    assert len(chambers) == 2
    for c in chambers + [model.add_ambient_pressure(h)]:
        model.set_pressure(c, p)
    integ = _integ(max_iter=2000)
    _run(model, integ, 5e-3, 30)
    X = model.me.rest_pos.astype(np.float64); x = model.get_x().astype(np.float64)
    Xc = X - X.mean(0)
    s_fem = float(((x - x.mean(0)) * Xc).sum() / (Xc ** 2).sum())
    assert abs(s_fem - s_exact) < 5e-3 * (1 - s_exact), f"stretch {s_fem:.6f}, exact {s_exact:.6f}"
    dev = np.abs((x - x.mean(0)) - s_exact * Xc).max()
    assert dev < 1e-4 * np.ptp(X[:, 1]), f"deviation from the uniform field {dev * 1e6:.2f} um"
    # with the cavities' sign flipped the state is nowhere near hydrostatic: the test can fail
    model.reset()
    for i, c in enumerate(model._chambers):
        model.set_pressure(i, -p if c["sign"] > 0 else p)
    _run(model, integ, 5e-3, 30)
    x = model.get_x().astype(np.float64)
    assert np.abs((x - x.mean(0)) - s_exact * Xc).max() > 100 * dev


def test_thick_walled_ring():
    """Lame: internal pressure in a ring, plane strain, small strain. The pixel ring's stepped
    boundary carries the same resultant load as the circle it approximates."""
    wp.init()
    n, size, p = 48, 0.048, 1.0e-3 * E
    yy, xx = np.mgrid[0:n, 0:n] + 0.5 - n / 2
    r = np.hypot(xx, yy)
    m = ((r >= n // 4) & (r <= n // 2 - 1)).astype(np.uint8)
    model, h = _model(m, (size, size))
    ch = model.add_pressure_chamber(h, at=(0.0, 0.0))
    A0 = model.chamber_area(ch)
    model.set_pressure(ch, p)
    _run(model, _integ(max_iter=4000), 5e-3, 25)
    a = np.sqrt(A0 / np.pi)                                              # equal-area radii of the pixel ring
    b = np.sqrt((m.sum() * (size / n) ** 2 + A0) / np.pi)
    ua = (1 + NU) / E * p * a * a / (b * b - a * a) * ((1 - 2 * NU) * a + b * b / a)
    dA, exact = model.chamber_area(ch) - A0, 2 * np.pi * a * ua
    assert abs(dA - exact) < 0.03 * exact, f"cavity area change {dA:.4e}, Lame {exact:.4e}"


def test_bar_hanging_under_gravity():
    wp.init()
    L, w, g = 0.060, 0.006, 200.0                                        # exaggerated gravity: ~0.4 % strain
    model, h = _model(np.ones((60, 6), np.uint8), (w, L))
    x0 = model.get_body_x(h)
    model.set_kinematic_constraint(h, np.where(x0[:, 1] > x0[:, 1].max() - 1e-6)[0])
    model.gravity = (0.0, -g)
    _run(model, _integ(), 2e-2, 12)
    bottom = np.where(x0[:, 1] < x0[:, 1].min() + 1e-6)[0]
    ext = -(model.get_body_x(h)[bottom, 1] - x0[bottom, 1]).mean()
    exact = RHO * g * L * L / (2 * EP)
    assert abs(ext - exact) < 0.02 * exact, f"extension {ext * 1e6:.2f} um, exact {exact * 1e6:.2f} um"


def test_active_material_free_growth():
    """An unconstrained body whose every element changes rest shape by A simply takes the
    shape A: no stress. alpha_max is lifted because the body is free (see the README)."""
    wp.init()
    m = np.ones((16, 10), np.uint8); m[5:9, 3:7] = 0
    model, h = _model(m, (0.010, 0.016))
    model.set_activation(model.add_active_region(h, np.ones_like(m)), 1.08)
    _run(model, _integ(alpha_max=1e9), 2e-2, 10)
    X = model.me.rest_pos.astype(np.float64); x = model.get_x().astype(np.float64)
    assert np.abs((x - x.mean(0)) - 1.08 * (X - X.mean(0))).max() < 1e-4 * 0.016
    assert model._solver.energy()[0] < 1e-8                              # stress-free

    model, h = _model(m, (0.010, 0.016))
    th = np.deg2rad(30.0)
    model.set_activation(model.add_active_region(h, np.ones_like(m), fiber=th), 0.85, transverse=1.10)
    _run(model, _integ(alpha_max=1e9), 2e-2, 10)
    x = model.get_x().astype(np.float64)
    F = np.linalg.lstsq(X - X.mean(0), x - x.mean(0), rcond=None)[0].T   # best-fit deformation gradient
    assert np.allclose(np.linalg.svd(F)[1], [1.10, 0.85], atol=2e-4)
    f = np.array([np.cos(th), np.sin(th)])
    assert abs(np.linalg.norm(F @ f) - 0.85) < 2e-4                      # ... and along the right direction


def test_stiffness_map_orientation():
    """A stiff patch given in the mask's own layout (row 0 at the bottom) must land on the
    elements at that place — not mirrored, not transposed."""
    wp.init()
    Emap = np.full((12, 8), 1.0e6); Emap[9:12, 0:3] = 7.0e6              # TOP-LEFT
    model, h = _model(np.ones((12, 8), np.uint8), (0.008, 0.012), stiffness=Emap)
    me = model.me
    cen = me.rest_pos[me.tri].mean(1)
    mu_ref = 1.0e6 / (2 * (1 + NU))
    stiff = np.isclose(me.mu, 7 * mu_ref, rtol=1e-6)
    assert (stiff | np.isclose(me.mu, mu_ref, rtol=1e-6)).all()
    assert stiff.sum() == 18
    assert (stiff == ((cen[:, 0] < -0.001) & (cen[:, 1] > 0.003))).all()


# ====================================================================== 3. vs the reference
def _curvature(X, x, L):
    mid = np.abs(X[:, 1]) < 1e-9
    o = np.argsort(X[mid, 0]); s = (X[mid, 0][o] - X[mid, 0].min()) / L
    c = (s > 0.2) & (s < 0.8)
    px, py = x[mid][o][c, 0], x[mid][o][c, 1]
    px, py = px - px.mean(), py - py.mean()
    sol = np.linalg.lstsq(np.c_[2 * px, 2 * py, np.ones_like(px)], px ** 2 + py ** 2, rcond=None)[0]
    return 1.0 / np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2)


def _cantilever(ny, aspect, hb=0.004):
    model, h = _model(np.ones((ny, ny * aspect), np.uint8), (hb * aspect, hb))
    X = model.me.rest_pos.astype(np.float64)
    left = np.where(X[:, 0] < X[:, 0].min() + 1e-9)[0]
    model.set_kinematic_constraint(h, left)
    return model, h, X, left


def test_tendon_bends_a_cantilever():
    """A cable at height e above the mid-line, through an eyelet at every column, loads the
    beam with the pure moment T e: curvature T e / (E' I)."""
    wp.init()
    hb, aspect, T = 0.004, 5, 60.0
    e = hb / 4.0
    beam = T * e / (EP * hb ** 3 / 12)

    def setup(ny):
        model, h, X, left = _cantilever(ny, aspect, hb)
        td = model.add_tendon(h, through=[(xx, e) for xx in np.unique(np.round(X[:, 0], 9))])
        model.set_tendon_tension(td, T)
        return model, X, left, model._tendons[td]["nodes"]

    # the discretisation: the reference converges on beam theory as the mesh refines
    kr = {}
    for ny in (8, 16):
        model, X, left, nodes = setup(ny)
        xr, res = REF.solve_static(model.me, left, tendon_nodes=nodes, T=T)
        assert res < 1e-8
        kr[ny] = _curvature(X, xr, hb * aspect)
    assert abs(kr[16] / beam - 1) < abs(kr[8] / beam - 1) < 0.15, kr
    # the solver: PNCG reaches the reference on the same mesh
    model, X, left, nodes = setup(8)
    integ = D.ImplicitIntegrator(max_iter=2000, eps=1e-14, tol_x=0.0, alpha_max=1e9,
                                 self_collision=False, friction=False)
    _run(model, integ, 5e-2, 12)
    k = _curvature(X, model.get_x().astype(np.float64), hb * aspect)
    assert abs(k / kr[8] - 1) < 0.015, f"PNCG curvature {k:.4f}, reference {kr[8]:.4f}"


def test_bilayer_bends():
    """Half the thickness contracts by 1 %: a bimetal strip. Compared with the reference
    solution for the same rest-shape change (Timoshenko's 1.5 eps / h is its continuum limit)."""
    wp.init()
    hb, aspect, ny, eps_a = 0.004, 5, 8, 0.01
    model, h, X, left = _cantilever(ny, aspect, hb)
    region = np.zeros((ny, ny * aspect), bool); region[: ny // 2] = True
    r = model.add_active_region(h, region)
    model.set_activation(r, 1.0 - eps_a)
    A = model._regions[r]["A"]; el = model._regions[r]["elems"]
    Bm = np.asarray(model.me.D_m_inv, np.float64).copy(); area = np.asarray(model.me.rest_area, np.float64).copy()
    Bm[el] = Bm[el] @ np.linalg.inv(A); area[el] *= np.linalg.det(A)
    xr, res = REF.solve_static(model.me, left, Bm=Bm, area=area)
    assert res < 1e-8
    kr = _curvature(X, xr, hb * aspect)
    assert abs(kr / (1.5 * eps_a / hb) - 1) < 0.15
    integ = D.ImplicitIntegrator(max_iter=2000, eps=1e-14, tol_x=0.0, alpha_max=1e9,
                                 self_collision=False, friction=False)
    _run(model, integ, 5e-2, 12)
    k = _curvature(X, model.get_x().astype(np.float64), hb * aspect)
    assert abs(k / kr - 1) < 0.015, f"PNCG curvature {k:.4f}, reference {kr:.4f}"


# ====================================================================== guards
def test_adjoint_refuses_actuators_it_cannot_differentiate():
    wp.init()
    model, h = _cavity_block()
    integ = _integ(max_iter=50)
    adj = integ.adjoint(model, 1e-3)
    model.gravity = (0.0, -9.81)                                         # no curvature: allowed
    integ.simulate(model, 1e-3)
    xs = [model.me.rest_pos.astype(np.float64), model.get_x().astype(np.float64)]
    seed = {1: np.ones_like(xs[0])}
    adj.transient_adjoint(xs, seed=seed)
    model.set_pressure(model.add_pressure_chambers(h)[0], 1.0e3)
    integ.simulate(model, 1e-3)
    try:
        adj.transient_adjoint(xs, seed=seed)
    except NotImplementedError:
        return
    raise AssertionError("transient_adjoint ran with a pressure term it has no tangent for")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"{name:52} OK")
    print("actuation tests PASSED")
