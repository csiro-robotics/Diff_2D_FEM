"""Rigid (affine) bodies, the rigid-mode coarse space, and pins.

  1. The 6 x 6 block algebra: Cholesky solve against numpy; the reduced elastic Hessian is
     positive semi-definite under compression, where the raw one is not.
  2. A free rigid body reproduces EXACT backward Euler: free fall in closed form, and a
     spinning, translating body against the rigid recursion (a mass-weighted Procrustes
     projection each step).
  3. A free SOFT body needs the rigid-mode coarse space: with it, free fall is right in 5
     iterations; without it, 20 iterations leave the body almost where it was.
  4. A block on a slope slides, or creeps, at the closed-form speed — rigid and soft.
  5. A rigid region inside a soft body: a cantilever with a rigid outer half bends like the
     float64 Newton reference in which that half is simply 1000x stiffer.
  6. A pin: a rigid bar pinned at one end swings with the period of a physical pendulum.
  7. A rigid block thrown at a soft one: linear momentum is conserved through the impact.

    python tests/test_rigid.py
"""
import os
import sys
import numpy as np
import warp as wp
import Diff_2D_FEM as D
from Diff_2D_FEM import real, mat22, np_real
from Diff_2D_FEM.core.math import vec6
from Diff_2D_FEM.implicit import affine as AF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_static as REF                                             # noqa: E402

DEV = "cuda" if wp.is_cuda_available() else "cpu"
DOUBLE = D.PRECISION == "double"
E, NU, RHO, G0 = 1.0e6, 0.40, 1100.0, 9.81


@wp.kernel
def _probe_solve(H: wp.array(dtype=AF.mat66), b: wp.array(dtype=vec6), x: wp.array(dtype=vec6)):
    i = wp.tid()
    x[i] = AF.chol6_solve(AF.chol6(H[i]), b[i])


@wp.kernel
def _probe_hessian(A: wp.array(dtype=mat22), Mu: real, La: real, H: wp.array(dtype=AF.mat66)):
    i = wp.tid()
    z = AF.mat33()
    H[i] = AF.reduced_hessian(A[i], Mu, La, real(1.0), z, z)


def _block(model, kind, center, shape=(12, 16), size=(0.008, 0.006), env_id=0):
    m = np.ones(shape, np.uint8)
    if kind == "rigid":
        return model.add_affine_body(m, physical_size=size, center=center, density=RHO, env_id=env_id)
    return model.add_soft_body(m, physical_size=size, center=center, E=E, nu=NU, density=RHO, env_id=env_id)


def _com(model, h=None):
    m = np.asarray(model.me.mass, np.float64); x = model.get_x().astype(np.float64)
    if h is not None:
        n = model.body_nodes(h); m, x = m[n], x[n]
    return (m[:, None] * x).sum(0) / m.sum()


def test_block_algebra():
    wp.init()
    rng = np.random.default_rng(0)
    n = 500
    M = rng.normal(size=(n, 6, 6)); H = M @ M.transpose(0, 2, 1) + 0.1 * np.eye(6)
    b = rng.normal(size=(n, 6))
    Hw = wp.array(H.astype(np_real), dtype=AF.mat66, device=DEV)
    bw = wp.array(b.astype(np_real), dtype=vec6, device=DEV)
    xw = wp.zeros(n, dtype=vec6, device=DEV)
    wp.launch(_probe_solve, dim=n, inputs=[Hw, bw, xw], device=DEV)
    ref = np.linalg.solve(H, b[:, :, None])[:, :, 0]
    err = np.abs(xw.numpy() - ref).max() / np.abs(ref).max()
    assert err < (1e-10 if DOUBLE else 2e-2), f"6x6 Cholesky solve off by {err:.1e}"

    # reduced elastic Hessian. At a pure rotation, turning further costs no elastic energy:
    # one zero eigenvalue. Strained by a few per cent it must stay positive semi-definite —
    # the raw Hessian is not, along the twist, whenever the region is compressed.
    th = rng.uniform(0, 2 * np.pi, n); c, s = np.cos(th), np.sin(th)
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], 1)
    S = np.eye(2) + 0.02 * rng.uniform(-1, 1, (n, 2, 2)); S = 0.5 * (S + S.transpose(0, 2, 1))
    tol = 1e-9 if DOUBLE else 1e-4
    for name, A in (("rotation", R), ("strained", R @ S)):
        Aw = wp.array(A.astype(np_real), dtype=mat22, device=DEV)
        Hq = wp.zeros(n, dtype=AF.mat66, device=DEV)
        wp.launch(_probe_hessian, dim=n, inputs=[Aw, 1.0, 2.5, Hq], device=DEV)
        H4 = Hq.numpy().astype(np.float64)[:, [1, 2, 4, 5]][:, :, [1, 2, 4, 5]]
        w = np.linalg.eigvalsh(H4)
        assert np.abs(H4 - H4.transpose(0, 2, 1)).max() < tol * w.max(), f"{name}: not symmetric"
        assert (w[:, 0] > -tol * w[:, -1]).all(), f"{name}: reduced elastic Hessian is indefinite"
        if name == "rotation":
            assert (np.abs(w[:, 0]) < tol * w[:, -1]).all(), "rotating should cost no elastic energy"
            assert (w[:, 1] > 0.1 * w[:, -1]).all(), "straining should"
    assert (np.linalg.det(R @ S) < 1).sum() > n // 4, "the sample should include compressed states"


def _rigid_backward_euler(X, m, v0, omega, h, n):
    """Exact backward Euler of a free rigid body: each step projects the inertial predictor
    onto the rigid motions (translation of the centroid, mass-weighted Procrustes rotation)."""
    c = (m[:, None] * X).sum(0) / m.sum(); Xb = X - c
    x = X.copy(); v = np.asarray(v0)[None] + omega * np.c_[-Xb[:, 1], Xb[:, 0]]
    for _ in range(n):
        xh = x + h * v; ch = (m[:, None] * xh).sum(0) / m.sum()
        S = (m[:, None, None] * (xh - ch)[:, :, None] * Xb[:, None, :]).sum(0)
        th = np.arctan2(S[1, 0] - S[0, 1], S[0, 0] + S[1, 1])
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        xn = Xb @ Rm.T + ch; v = (xn - x) / h; x = xn
    return x


def test_free_rigid_body_is_exact_backward_euler():
    wp.init()
    # free fall
    model = D.Model(device=DEV); model.damping = 0.0
    h = _block(model, "rigid", (0.0, 0.05)); model.finalize(); model.gravity = (0.0, -G0)
    y0 = _com(model)[1]
    integ = D.ImplicitIntegrator(self_collision=False, friction=False)
    for _ in range(100):
        integ.simulate(model, 1e-3)
    err = (y0 - _com(model)[1]) / (G0 * 1e-6 * 100 * 101 / 2) - 1
    assert abs(err) < (1e-9 if DOUBLE else 1e-3), f"free fall off by {err:.1e}"
    assert integ.stats(model)["iters"][0] <= 4, "a free rigid body should converge at once"

    # spin + drift
    model = D.Model(device=DEV); model.damping = 0.0
    h = _block(model, "rigid", (0.0, 0.05)); model.finalize()
    model.set_body_velocity(h, velocity=(0.1, 0.0), omega=20.0)
    ref = _rigid_backward_euler(model.rest_x(h), np.asarray(model.me.mass, np.float64),
                                (0.1, 0.0), 20.0, 1e-3, 200)
    integ = D.ImplicitIntegrator(eps=1e-12 if DOUBLE else 1e-7, self_collision=False, friction=False)
    for _ in range(200):
        integ.simulate(model, 1e-3)
    x = model.get_body_x(h).astype(np.float64)
    st = model.affine_state(0)
    assert st["angle"] % (2 * np.pi) > 3.5, "the body should have turned ~3.85 rad"
    err = np.abs(x - ref).max()
    assert err < (1e-6 if DOUBLE else 1e-3), f"spinning body is {err:.1e} m from exact backward Euler"
    strain = np.abs(st["A"].T @ st["A"] - np.eye(2)).max()
    assert strain < (1e-6 if DOUBLE else 1e-5), f"the rigid body strained by {strain:.1e}"


def test_rigid_modes_make_a_free_soft_body_converge():
    wp.init()
    def fall(its, rigid_modes):
        model = D.Model(device=DEV); model.damping = 0.0
        _block(model, "soft", (0.0, 0.05)); model.finalize(); model.gravity = (0.0, -G0)
        y0 = _com(model)[1]
        integ = D.ImplicitIntegrator(max_iter=its, self_collision=False, friction=False,
                                     rigid_modes=rigid_modes)
        for _ in range(100):
            integ.simulate(model, 1e-3)
        return (y0 - _com(model)[1]) / (G0 * 1e-6 * 100 * 101 / 2) - 1
    with_modes = fall(5, True)
    assert abs(with_modes) < (1e-4 if DOUBLE else 5e-3), f"free fall in 5 iterations off by {with_modes:.1e}"
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        without = fall(20, False)
    assert without < -0.9, f"Jacobi alone was expected to stall; it got to {without:+.1%}"


def test_block_on_a_slope():
    wp.init()
    for kind in ("rigid", "soft"):
        for th_deg, mu in ((30.0, 0.30), (15.0, 0.60)):       # slides ; holds and creeps
            th = np.deg2rad(th_deg); c, dt = 20.0, 1e-3
            model = D.Model(device=DEV); model.add_ground(height=0.0, mu=mu); model.damping = c
            reach = 0.5e-3 / 8                                    # the barrier: 1/8 pixel
            _block(model, kind, (0.0, 0.003 + 0.9 * reach)); model.finalize()
            assert np.isclose(model.dhat, reach)
            integ = D.ImplicitIntegrator(self_collision=False)
            model.gravity = (0.0, -G0 * np.cos(th))                       # settle on the level
            for _ in range(300):
                integ.simulate(model, dt)
            W = np.asarray(model.me.mass).sum() * G0 * np.cos(th)
            N = abs(integ.collider_force(model, friction=False)[0][1])
            assert abs(N / W - 1.0) < 5e-3, f"{kind}: ground carries {N / W:.4f} of the weight"
            model.gravity = (G0 * np.sin(th), -G0 * np.cos(th))           # tilt
            xs = []
            for _ in range(400):
                integ.simulate(model, dt); xs.append(_com(model)[0])
            v = np.diff(np.array(xs)[-60:]).mean() / dt
            ce = (1.0 - np.exp(-dt * c)) / dt                             # the drag's rate
            if np.tan(th) > mu:
                expect = G0 * (np.sin(th) - mu * np.cos(th)) / ce
            else:                         # held by friction: it creeps on the smoothed branch
                expect = G0 * np.sin(th) / (mu * G0 * np.cos(th) / model.epsv + ce)
            assert abs(v / expect - 1.0) < (0.01 if DOUBLE else 0.15), \
                f"{kind} block, {th_deg:.0f} deg, mu {mu}: speed {v:.4e}, expected {expect:.4e}"


def test_rigid_region_in_a_soft_body():
    wp.init()
    model = D.Model(device=DEV); model.damping = 300.0
    h = model.add_soft_body(np.ones((6, 30), np.uint8), physical_size=(0.020, 0.004),
                            center=(0.0, 0.0), E=E, nu=NU, density=RHO)
    model.finalize()
    region = np.zeros((6, 30), bool); region[:, 18:] = True
    model.add_affine_region(h, region)
    X = model.rest_x(h)
    left = np.where(X[:, 0] < X[:, 0].min() + 1e-9)[0]
    tip = np.where(X[:, 0] > X[:, 0].max() - 1e-9)[0]
    model.set_kinematic_constraint(h, left)
    f = np.zeros_like(X); f[tip, 1] = -6.0 / len(tip)                     # 6 N/m down, at the tip
    xr, res = REF.solve_static(model.me, left, fext=f, ramps=10)
    assert res < 1e-4, "the reference did not converge"           # (residual scales with 1e9 Pa)
    model.set_external_force(h, f)
    integ = D.ImplicitIntegrator(self_collision=False, friction=False)
    for _ in range(300):
        integ.simulate(model, 2e-3)
    x = model.get_body_x(h).astype(np.float64)
    d, dr = (x[tip, 1] - X[tip, 1]).mean(), (xr[tip, 1] - X[tip, 1]).mean()
    assert dr < -1e-3, "the load should bend the beam by more than a millimetre"
    assert abs(d / dr - 1.0) < (0.01 if DOUBLE else 0.05), \
        f"tip deflection {d * 1e3:.4f} mm, reference {dr * 1e3:.4f} mm"
    A = model.affine_state(0)["A"]
    assert np.abs(A.T @ A - np.eye(2)).max() < 1e-3, "the rigid half strained"
    assert abs(model.affine_state(0)["angle"]) > 0.05, "the rigid half should have rotated"


def test_pin_makes_a_pendulum():
    wp.init()
    model = D.Model(device=DEV); model.damping = 0.0
    h = model.add_affine_body(np.ones((4, 24), np.uint8), physical_size=(0.012, 0.002),
                              center=(0.006, 0.0), density=RHO)
    model.finalize()
    pin = model.add_pin(h, (0.0, 0.0))
    X = model.rest_x(h); m = np.asarray(model.me.mass, np.float64)[model.body_nodes(h)]
    pivot = X[np.linalg.norm(X, axis=1).argmin()]
    d = np.linalg.norm((m[:, None] * X).sum(0) / m.sum() - pivot)
    Ip = (m * ((X - pivot) ** 2).sum(1)).sum()
    th0 = np.deg2rad(10.0)                               # gravity 10 degrees off the bar's axis
    period = 2 * np.pi * np.sqrt(Ip / (m.sum() * G0 * d)) * (1 + th0 ** 2 / 16)
    model.gravity = (G0 * np.cos(th0), -G0 * np.sin(th0))
    dt = 5e-4
    integ = D.ImplicitIntegrator(self_collision=False, friction=False)
    ang, gap = [], 0.0
    for _ in range(int(1.1 * period / dt)):
        integ.simulate(model, dt)
        ang.append(model.affine_state(0)["angle"] + th0)                 # from the gravity direction
        gap = max(gap, abs(model.spring_force(pin)[1]))
    ang = np.array(ang); t = dt * np.arange(1, len(ang) + 1)
    k = np.where(np.sign(ang[1:]) != np.sign(ang[:-1]))[0]
    assert len(k) >= 2, "the bar did not swing through its equilibrium twice"
    cross = [t[i] - ang[i] * dt / (ang[i + 1] - ang[i]) for i in k[:2]]
    got = 2.0 * (cross[1] - cross[0])
    assert abs(got / period - 1.0) < (0.01 if DOUBLE else 0.05), \
        f"period {got * 1e3:.2f} ms, a physical pendulum has {period * 1e3:.2f} ms"
    assert gap < 1e-6, f"the pin let go by {gap:.1e} m"


def test_momentum_through_an_impact():
    wp.init()
    model = D.Model(device=DEV); model.damping = 0.0
    sq = dict(shape=(8, 8), size=(0.004, 0.004))
    rigid = _block(model, "rigid", (0.0, 0.0), **sq)
    soft = _block(model, "soft", (0.0065, 0.0), **sq)
    model.finalize()
    model.set_body_velocity(rigid, velocity=(0.3, 0.0))
    m = np.asarray(model.me.mass, np.float64)
    def momentum():
        v = np.concatenate([model.get_body_v(rigid), model.get_body_v(soft)]).astype(np.float64)
        return (m[:, None] * v).sum(0)
    p0 = momentum()
    integ = D.ImplicitIntegrator(friction=False)
    for _ in range(80):
        integ.simulate(model, 2.5e-4)
    p1 = momentum()
    vs = (m[model.body_nodes(soft), None] * model.get_body_v(soft)).sum(0) / m[model.body_nodes(soft)].sum()
    assert vs[0] > 0.05, f"the soft block was not pushed (v = {vs[0]:.3f} m/s)"
    err = np.linalg.norm(p1 - p0) / np.linalg.norm(p0)
    assert err < (1e-4 if DOUBLE else 2e-2), f"linear momentum changed by {err:.1e}"
    gap = model.get_body_x(soft)[:, 0].min() - model.get_body_x(rigid)[:, 0].max()
    assert gap > 0.0, "the blocks passed through each other"


if __name__ == "__main__":
    test_block_algebra();                               print("6x6 block algebra                       OK")
    test_free_rigid_body_is_exact_backward_euler();     print("free rigid body == exact backward Euler OK")
    test_rigid_modes_make_a_free_soft_body_converge();  print("rigid modes for free soft bodies        OK")
    test_block_on_a_slope();                            print("block on a slope: slide and creep       OK")
    test_rigid_region_in_a_soft_body();                 print("rigid region in a soft body             OK")
    test_pin_makes_a_pendulum();                        print("pin: physical pendulum period           OK")
    test_momentum_through_an_impact();                  print("momentum through an impact              OK")
    print("rigid-body tests PASSED")
