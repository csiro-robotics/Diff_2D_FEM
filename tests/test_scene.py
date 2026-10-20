"""Scenes with several bodies and several environments.

  1. Bodies keep their own grid, size and material: a body simulated in a scene with a
     different neighbour follows the trajectory it follows alone, and each body's mass is
     its density times its area.
  2. Environments do not interact: a design run alone and the same design in a batch next
     to different ones follow the same trajectory.
  3. Bodies sharing an environment collide: a block dropped on another comes to rest on
     it, and the ground carries the weight of both.
  4. Bodies that overlap in one environment are refused.

    python tests/test_scene.py
"""
import numpy as np
import warp as wp
import Diff_2D_FEM as D

DEV = "cuda" if wp.is_cuda_available() else "cpu"
DOUBLE = D.PRECISION == "double"
G0 = 9.81


def _cantilever(model, E, center, size=(0.020, 0.004), shape=(6, 30), env_id=0, density=1100.0):
    return model.add_soft_body(np.ones(shape, np.uint8), physical_size=size, center=center,
                               E=E, nu=0.40, density=density, env_id=env_id)


def _clamp_left(model, h):
    X = model.rest_x(h)
    model.set_kinematic_constraint(h, np.where(X[:, 0] < X[:, 0].min() + 1e-9)[0])


def _droop(model, steps=60):
    model.gravity = (0.0, -G0); model.damping = 100.0
    integ = D.ImplicitIntegrator(max_iter=120, self_collision=False, friction=False)
    for _ in range(steps):
        integ.simulate(model, 1e-3)


def test_bodies_keep_their_own_geometry_and_material():
    wp.init()
    # alone
    one = D.Model(device=DEV); h1 = _cantilever(one, 4.0e6, (0.0, 0.0)); one.finalize()
    _clamp_left(one, h1); _droop(one)
    # the same body, sharing a scene with a softer one on a different grid and of another density
    two = D.Model(device=DEV)
    hs = _cantilever(two, 1.0e6, (0.0, 0.03), size=(0.016, 0.004), shape=(8, 32), density=2000.0)
    hb = _cantilever(two, 4.0e6, (0.0, 0.0))
    two.finalize()
    m = np.asarray(two.me.mass)
    assert np.isclose(m[two.body_nodes(hs)].sum(), 2000.0 * 0.016 * 0.004, rtol=1e-9)
    assert np.isclose(m[two.body_nodes(hb)].sum(), 1100.0 * 0.020 * 0.004, rtol=1e-9)
    _clamp_left(two, hs); _clamp_left(two, hb); _droop(two)
    xa, xb = one.get_body_x(h1).astype(np.float64), two.get_body_x(hb).astype(np.float64)
    sag = np.abs(xa - one.rest_x(h1)).max()
    assert sag > 1e-5, "the cantilever did not deflect"
    # Not to round-off: the two bodies share an environment, hence one CG step length and
    # one beta, so their iterates differ until both have converged.
    err = np.abs(xa - xb).max() / sag
    assert err < (1e-3 if DOUBLE else 0.5), \
        f"a body's trajectory changed by {err:.1e} of its deflection when a neighbour was added"
    # and the softer, heavier one sags more
    assert np.abs(two.get_body_x(hs) - two.rest_x(hs)).max() > 2.0 * sag


def _press(model, hs, steps=40):
    """Drive the top row of every body 6 mm down onto the disc."""
    x0 = [model.rest_x(h) for h in hs]
    tops = [np.where(x[:, 1] > x[:, 1].max() - 1e-9)[0] for x in x0]
    for h, t in zip(hs, tops):
        model.set_kinematic_constraint(h, t)
    integ = D.ImplicitIntegrator(max_iter=100)
    for k in range(1, steps + 1):
        for h, x, t in zip(hs, x0, tops):
            tgt = x.copy(); tgt[t, 1] -= 0.006 * k / steps
            model.set_kinematic_target(h, tgt)
        integ.simulate(model, 1e-3)
    return integ


def test_environments_do_not_interact():
    """Run on the CPU, where kernels are deterministic, so the comparison can be EXACT: a
    design in a batch must reproduce the design alone bit for bit. (On a GPU the atomic
    sums are unordered and two runs of the SAME scene already differ by ~1e-7 m.)"""
    wp.init()
    rng = np.random.default_rng(3)
    d1 = np.ones((24, 12), np.uint8); d1[6:18, 4:8] = 0                     # a frame
    d2 = (rng.random((24, 12)) > 0.15).astype(np.uint8); d2[-1] = 1; d2[:, 0] = 1
    def scene(designs):
        model = D.Model(device="cpu")
        model.add_disc(center=(0.05, 0.030), radius=0.006, mu=0.6)
        hs = [model.add_soft_body(m, physical_size=(0.012, 0.024), center=(0.05, 0.0495),
                                  E=1.0e6, nu=0.40, density=1100.0, env_id=i)
              for i, m in enumerate(designs)]
        model.finalize()
        return model, hs
    alone, (ha,) = scene([d1]); ia = _press(alone, [ha])
    batch, hb = scene([d1, d2, d1]); ib = _press(batch, hb)
    assert batch.n_envs == 3
    xa = alone.get_body_x(ha)
    Fa, Fb = ia.collider_force(alone)[0], ib.collider_force(batch)
    assert np.linalg.norm(Fa) > 1.0, "no contact developed"
    for env, h in ((0, hb[0]), (2, hb[2])):
        assert np.array_equal(batch.get_body_x(h), xa), \
            f"environment {env}: the design did not reproduce its solo trajectory exactly"
        assert np.array_equal(Fb[env], Fa), f"environment {env}: contact force differs"
    assert np.abs(batch.get_body_x(hb[1]).mean(0) - xa.mean(0)).max() > 1e-5, \
        "the different design moved identically"


def test_bodies_in_one_environment_collide():
    wp.init()
    model = D.Model(device=DEV)
    model.add_ground(height=0.0, mu=0.6); model.damping = 60.0
    block = np.ones((12, 16), np.uint8)
    reach = 0.5e-3 / 8                                   # the barrier's default reach: 1/8 pixel
    lower = model.add_soft_body(block, physical_size=(0.008, 0.006), E=1.0e6, nu=0.40,
                                density=1100.0, center=(0.0, 0.003 + 0.9 * reach))
    upper = model.add_soft_body(block, physical_size=(0.008, 0.006), E=1.0e6, nu=0.40,
                                density=1100.0, center=(0.0, 0.003 + 0.9 * reach + 0.0068))
    model.finalize()
    # contact parameters were left unset, so they were scaled to the scene
    assert np.isclose(model.dhat, reach) and np.isclose(model.kappa, 3.2e4) \
        and np.isclose(model.self_stiffness, 2.0e5)
    model.gravity = (0.0, -G0)
    integ = D.ImplicitIntegrator(max_iter=60)
    for _ in range(500):
        integ.simulate(model, 2e-3)
    xl, xu = model.get_body_x(lower), model.get_body_x(upper)
    gap = xu[:, 1].min() - xl[:, 1].max()
    assert 0.0 < gap < 1.5 * model.dx, f"upper block is {gap * 1e3:.3f} mm above the lower one"
    assert np.abs(model.get_body_v(upper)).max() < 2e-3, "the stack has not come to rest"
    weight = np.asarray(model.me.mass).sum() * G0
    Fy = abs(integ.collider_force(model, friction=False)[0][1])
    assert abs(Fy / weight - 1.0) < 0.02, f"ground carries {Fy / weight:.4f} of the two weights"


def test_overlapping_bodies_are_refused():
    model = D.Model(device=DEV)
    for _ in range(2):
        model.add_soft_body(np.ones((8, 8), np.uint8), physical_size=(0.008, 0.008), center=(0.0, 0.0))
    try:
        model.finalize()
    except ValueError as e:
        assert "env_id" in str(e)
    else:
        raise AssertionError("two bodies in the same place were accepted")
    ok = D.Model(device=DEV)
    for i in range(2):
        ok.add_soft_body(np.ones((8, 8), np.uint8), physical_size=(0.008, 0.008), center=(0.0, 0.0), env_id=i)
    assert ok.finalize().n_envs == 2


if __name__ == "__main__":
    test_overlapping_bodies_are_refused();               print("overlapping bodies refused              OK")
    test_bodies_keep_their_own_geometry_and_material();  print("per-body geometry and material          OK")
    test_environments_do_not_interact();                 print("environments are isolated               OK")
    test_bodies_in_one_environment_collide();            print("bodies in one environment collide       OK")
    print("scene tests PASSED")
