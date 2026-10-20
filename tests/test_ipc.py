"""IPC contact between bodies: model.body_contact = "ipc" (implicit/ipc_contact.py).

  1. Sliding a flat block along a flat slab at a fixed gap costs no energy — the vertex
     corrections at work. (Without them the energy doubles at every mesh vertex.)
  2. The gradient is the derivative of the energy and p^T H p the derivative of the
     gradient, against finite differences.
  3. A block on a held slab, gravity tilted: frictionless it slides at g sin(theta) / drag,
     with friction at the Coulomb speed, and below the friction angle it creeps at the
     smoothed law's speed — rigid block and soft.
  4. Two blocks stacked on the ground: the ground carries both weights, and the upper block
     floats within the barrier's reach of the lower one.
  5. Two soft pads pinch a free rigid block and lift: with friction the block comes up with
     them; without, it stays on the ground.
  6. Nothing passes through anything: a block thrown at another at five pixels per step,
     head on and corner to corner. At half a pixel per step momentum goes through the impact.
  7. A body whose motion is prescribed is swept, not teleported: it pushes a free block
     ahead of itself at a pixel and a half per step, and against a wall it stops when it
     has squeezed the block as hard as its drive allows, reporting how far short of its
     target it is.
  8. Bodies that touch before the first step are refused.

    python tests/test_ipc.py
"""
import numpy as np
import warp as wp
import Diff_2D_FEM as D
from Diff_2D_FEM import vec2, np_real

DEV = "cuda" if wp.is_cuda_available() else "cpu"
DOUBLE = D.PRECISION == "double"
E, NU, RHO, G0 = 1.0e6, 0.40, 1100.0, 9.81
PX = 0.5e-3


def _soft(model, shape, center, **kw):
    return model.add_soft_body(np.ones(shape, np.uint8), physical_size=(shape[1] * PX, shape[0] * PX),
                               center=center, E=E, nu=NU, density=RHO, **kw)


def _rigid(model, shape, center, density=RHO):
    return model.add_affine_body(np.ones(shape, np.uint8), physical_size=(shape[1] * PX, shape[0] * PX),
                                 center=center, density=density)


def _gap(model):
    """Smallest node-to-edge distance between surfaces, in units of dhat."""
    s = model._solver
    return float(s.ipc.min_distance(s).min()) / model.dhat


# ---------------------------------------------------------------- 1, 2: the potential itself
def _slab_and_block(gap_in_dhat, mu=0.0):
    model = D.Model(device="cpu"); model.body_contact = "ipc"; model.body_friction = mu
    _soft(model, (4, 80), (0.0, -2 * PX))
    dhat = PX / 8                                             # the default reach: 1/8 pixel
    block = _soft(model, (8, 24), (0.0, 4 * PX + gap_in_dhat * dhat))
    model.finalize()
    assert np.isclose(model.dhat, dhat) and np.isclose(model.body_kappa, 0.1 * E * PX / dhat)
    s = D.ImplicitIntegrator().attach(model, 1e-3)
    return model, s, block


def _set(s, x):
    xa = wp.array(x.astype(np_real), dtype=vec2, device="cpu")
    s.x.assign(xa); s.xn.assign(xa)


def _energy(s, x):
    _set(s, x); s.ipc._grid(s)
    buf = wp.zeros(s.B, dtype=wp.float64, device="cpu"); s.ipc.energy(s, buf)
    return float(buf.numpy()[0])


def _gradient(s, x):
    _set(s, x); s.grad.zero_(); s.diagH.zero_(); s.ipc.grad_diag(s)
    return s.grad.numpy().astype(np.float64).copy()


def test_flat_sliding_costs_nothing():
    wp.init()
    for gap in (0.9, 0.5, 0.1):
        model, s, block = _slab_and_block(gap)
        X = model.me.rest_pos.astype(np.float64); nodes = model.body_nodes(block)
        Es = []
        for shift in np.linspace(0.0, 2 * PX, 81):            # two edge lengths
            x = X.copy(); x[nodes, 0] += shift; Es.append(_energy(s, x))
        Es = np.array(Es)
        # 50 pairs carry the contact; one slab node passes each end of the block per edge
        # length, and that — one pair in fifty — is all the energy may change by
        spread = (Es.max() - Es.min()) / Es.mean()
        assert spread < (0.03 if DOUBLE else 0.05), \
            f"gap {gap} dhat: the energy changes by {spread:.3f} while sliding along a flat face"


def test_derivatives():
    wp.init()
    model, s, block = _slab_and_block(0.5)
    rng = np.random.default_rng(0)
    X = model.me.rest_pos.astype(np.float64); nodes = model.body_nodes(block)
    x = X.copy(); c = x[nodes].mean(0); th = 0.004            # tilt the block, roughen both
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    x[nodes] = (x[nodes] - c) @ R.T + c + np.array([0.13 * PX, 0.2 * model.dhat])
    x += 0.02 * model.dhat * rng.standard_normal(x.shape)
    g = _gradient(s, x)
    act = np.where(np.abs(g).sum(1) > 0)[0]
    assert len(act) > 30, "the probe state is not in contact"
    h = 1e-9 if DOUBLE else 2e-8
    num = np.zeros_like(g)
    for i in act:
        for k in range(2):
            xp, xm = x.copy(), x.copy(); xp[i, k] += h; xm[i, k] -= h
            num[i, k] = (_energy(s, xp) - _energy(s, xm)) / (2 * h)
    err = np.abs(g - num).max() / np.abs(g).max()
    assert err < (1e-6 if DOUBLE else 5e-2), f"gradient differs from d(energy)/dx by {err:.2e}"

    p = np.zeros_like(x); p[act] = rng.standard_normal((len(act), 2))
    _gradient(s, x)                                           # builds the broadphase at x
    s.p.assign(wp.array(p.astype(np_real), dtype=vec2, device="cpu"))
    s.pHp.zero_(); s.alpha.fill_(3.0); s.p_inf.fill_(1.0)
    s.ipc.pHp(s); curv = float(s.pHp.numpy()[0])
    hh = (1e-7 if DOUBLE else 1e-4) * model.dhat
    fd = float((((_gradient(s, x + hh * p) - _gradient(s, x - hh * p)) / (2 * hh)) * p).sum())
    assert abs(curv / fd - 1.0) < (1e-3 if DOUBLE else 0.1), \
        f"p^T H p = {curv:.6e}, finite differences of the gradient give {fd:.6e}"


# ---------------------------------------------------------------- 3: friction
def test_slides_at_the_coulomb_speed():
    wp.init()
    c, dt = 200.0, 1e-3
    ce = (1.0 - np.exp(-dt * c)) / dt
    for kind, mu, th_deg in (("rigid", 0.0, 40.0), ("rigid", 0.3, 30.0), ("rigid", 0.5, 15.0),
                             ("soft", 0.3, 30.0)):
        th = np.deg2rad(th_deg)
        model = D.Model(device=DEV); model.damping = c
        model.body_contact = "ipc"; model.body_friction = mu
        slab = _soft(model, (6, 60), (0.0, 0.0))
        at = (-0.010, 0.0015 + 0.002 + 0.9 * PX / 8)
        block = _rigid(model, (8, 8), at) if kind == "rigid" else _soft(model, (8, 8), at)
        model.finalize()
        model.set_kinematic_constraint(slab, np.arange(model.body_nodes(slab).size))
        integ = D.ImplicitIntegrator()
        model.gravity = (0.0, -G0 * np.cos(th))
        for _ in range(100):
            integ.simulate(model, dt)
        gap = model.get_body_x(block)[:, 1].min() - model.get_body_x(slab)[:, 1].max()
        assert 0.5 * model.dhat < gap < model.dhat, f"resting gap {gap / model.dhat:.2f} dhat"
        model.gravity = (G0 * np.sin(th), -G0 * np.cos(th))
        xs = []
        for _ in range(300):
            integ.simulate(model, dt); xs.append(model.get_body_x(block)[:, 0].mean())
        v = np.diff(np.array(xs)[-200:]).mean() / dt
        if np.tan(th) > mu:
            expect = G0 * (np.sin(th) - mu * np.cos(th)) / ce
        else:
            expect = G0 * np.sin(th) / (mu * G0 * np.cos(th) / model.epsv + ce)
        tol = 0.01 if kind == "rigid" else 0.02
        assert abs(v / expect - 1.0) < (tol if DOUBLE else 0.25), \
            f"{kind} block, mu {mu}, {th_deg:.0f} deg: {v:.4e} m/s, the closed form gives {expect:.4e}"


# ---------------------------------------------------------------- 4: resting contact
def test_a_stack_carries_its_weight():
    wp.init()
    model = D.Model(device=DEV); model.body_contact = "ipc"
    model.add_ground(height=0.0, mu=0.6); model.damping = 60.0
    reach = PX / 8
    lower = _soft(model, (12, 16), (0.0, 6 * PX + 0.9 * reach))
    upper = _soft(model, (12, 16), (0.0, 18 * PX + 0.9 * reach + 0.5 * PX))
    model.finalize()
    model.gravity = (0.0, -G0)
    integ = D.ImplicitIntegrator()
    for _ in range(300):
        integ.simulate(model, 2e-3)
    gap = model.get_body_x(upper)[:, 1].min() - model.get_body_x(lower)[:, 1].max()
    assert 0.0 < gap < model.dhat, f"the upper block is {gap / model.dhat:.3f} dhat above the lower one"
    assert np.abs(model.get_body_v(upper)).max() < 2e-3, "the stack has not come to rest"
    weight = np.asarray(model.me.mass).sum() * G0
    Fy = abs(integ.collider_force(model, friction=False)[0][1])
    assert abs(Fy / weight - 1.0) < (1e-3 if DOUBLE else 0.02), \
        f"the ground carries {Fy / weight:.5f} of the two weights"


# ---------------------------------------------------------------- 5: a grip
def _pinch_and_lift(mu, lift=4.0e-3):
    model = D.Model(device=DEV); model.body_contact = "ipc"; model.body_friction = mu
    model.add_ground(height=0.0, mu=0.5)
    block = _rigid(model, (16, 16), (0.0, 8 * PX + 0.9 * PX / 8), density=600.0)
    pads = [_soft(model, (16, 8), (sgn * 14 * PX, 8 * PX + 0.8e-3)) for sgn in (-1, +1)]
    model.finalize(); model.gravity = (0.0, -G0)
    for sgn, h in zip((-1, +1), pads):                        # the pads are driven by their backs
        X = model.rest_x(h); back = X[:, 0].min() if sgn < 0 else X[:, 0].max()
        model.set_kinematic_constraint(h, np.where(np.abs(X[:, 0] - back) < 1e-9)[0])
    integ = D.ImplicitIntegrator()
    y0 = model.get_body_x(block)[:, 1].mean()
    n_close, n_lift, n_hold = 100, 200, 50
    for k in range(1, n_close + n_lift + n_hold + 1):
        a = min(k / n_close, 1.0); b = min(max(k - n_close, 0) / n_lift, 1.0)
        for sgn, h in zip((-1, +1), pads):
            X = model.rest_x(h); X[:, 0] -= sgn * 2.25 * PX * a; X[:, 1] += lift * b
            model.set_kinematic_target(h, X)
        integ.simulate(model, 1e-3)
    assert np.isfinite(model.get_x()).all() and _gap(model) > 0.0
    return (model.get_body_x(block)[:, 1].mean() - y0) / lift


def test_a_grip_needs_friction():
    wp.init()
    rose = _pinch_and_lift(0.0)
    assert abs(rose) < 0.05, f"with no friction the block came up {rose:.3f} of the lift"
    rose = _pinch_and_lift(0.8)
    assert rose > 0.95, f"with friction the block came up only {rose:.3f} of the lift"


# ---------------------------------------------------------------- 6: no tunnelling
def _throw(other, px_per_step, diagonal=False, steps=120):
    """A rigid block thrown at another body; returns (model, thrown, target, momentum error)."""
    dt = 2.5e-4; v0 = px_per_step * PX / dt
    model = D.Model(device=DEV); model.body_contact = "ipc"
    if diagonal:                              # A's lower right corner at B's upper left corner
        a = _rigid(model, (8, 8), (-6 * PX, 6 * PX)); at = (4 * PX, -4 * PX)
    else:
        a = _rigid(model, (8, 8), (0.0, 0.0)); at = (13 * PX, 0.8 * PX)
    b = _rigid(model, (8, 8), at) if other == "rigid" else _soft(model, (8, 8), at)
    model.finalize()
    if other == "held":
        model.set_kinematic_constraint(b, np.arange(model.body_nodes(b).size))
    vel = (v0 / np.sqrt(2.0), -v0 / np.sqrt(2.0)) if diagonal else (v0, 0.0)
    model.set_body_velocity(a, velocity=vel)
    integ = D.ImplicitIntegrator(friction=False)
    m = np.asarray(model.me.mass, np.float64)
    mom = lambda: (m[:, None] * np.concatenate([model.get_body_v(a), model.get_body_v(b)])).sum(0)
    p0 = mom(); gap = np.inf; inside = False
    for _ in range(steps):
        integ.simulate(model, dt)
        gap = min(gap, _gap(model))
        if other == "held":                   # B does not move: its box is where it started
            xa, xb = model.get_body_x(a), model.rest_x(b)
            lo, hi = xb.min(0) + 1e-9, xb.max(0) - 1e-9
            inside |= bool(((xa > lo) & (xa < hi)).all(1).any())
    assert np.isfinite(model.get_x()).all(), "the state is not finite"
    return gap, inside, np.linalg.norm(mom() - p0) / np.linalg.norm(p0)


def test_nothing_passes_through():
    wp.init()
    for other, diagonal in (("soft", False), ("rigid", False), ("held", True), ("held", False)):
        gap, inside, _ = _throw(other, 5.0, diagonal)
        assert gap > 0.0 and not inside, \
            f"thrown at a {other} block at 5 pixels per step: gap {gap:.2e} dhat, inside {inside}"
    _, _, dp = _throw("soft", 0.5)
    assert dp < (2e-3 if DOUBLE else 5e-2), f"momentum changed by {dp:.1e} through the impact"


# ---------------------------------------------------------------- 7: prescribed bodies
def _push(px_per_step, wall, steps=30):
    model = D.Model(device=DEV); model.body_contact = "ipc"; model.body_friction = 0.3
    model.damping = 20.0
    pusher = _soft(model, (12, 4), (0.0, 0.0))
    block = _soft(model, (8, 8), (8 * PX, 0.0))                # two pixels ahead of the pusher
    if wall:
        stop = _soft(model, (12, 4), (17 * PX, 0.0))           # a held body, three behind it
    model.finalize()
    model.set_kinematic_constraint(pusher, np.arange(model.body_nodes(pusher).size))
    if wall:
        model.set_kinematic_constraint(stop, np.arange(model.body_nodes(stop).size))
    integ = D.ImplicitIntegrator()
    gap = np.inf
    for k in range(1, steps + 1):
        model.set_kinematic_target(pusher, model.rigid_motion(pusher, (k * px_per_step * PX, 0.0)))
        integ.simulate(model, 1e-3)
        gap = min(gap, _gap(model))
    x = model.get_x().astype(np.float64); X = model.me.rest_pos.astype(np.float64)
    t = np.asarray(model.me.tri)
    area = lambda P: (P[t[:, 1], 0] - P[t[:, 0], 0]) * (P[t[:, 2], 1] - P[t[:, 0], 1]) \
        - (P[t[:, 1], 1] - P[t[:, 0], 1]) * (P[t[:, 2], 0] - P[t[:, 0], 0])
    assert np.isfinite(x).all() and gap > 0.0, f"smallest gap {gap:.2e} dhat"
    return model, pusher, block, integ.stats(model)["bc_lag"], float((area(x) / area(X)).min())


def test_a_prescribed_body_is_swept():
    wp.init()
    model, pusher, block, lag, squash = _push(1.5, wall=False)
    front = model.get_body_x(pusher)[:, 0].max()
    assert lag < 1e-9 * PX, f"the pusher is {lag / PX:.2e} pixels short of its target with nothing in its way"
    assert model.get_body_x(block)[:, 0].min() > front, "the block is not ahead of the pusher"
    assert squash > 0.9, "a block pushed through free space was squashed"
    # against a wall the target (47 pixels on) cannot be reached: the pusher must stop
    model, pusher, block, lag, squash = _push(1.5, wall=True)
    xb = model.get_body_x(block)
    assert lag > 25 * PX, f"the pusher claims to be within {lag / PX:.1f} pixels of an unreachable target"
    assert xb[:, 0].max() < 15 * PX and xb[:, 0].min() > model.get_body_x(pusher)[:, 0].max(), \
        "the block is not between the pusher and the wall"
    assert 0.05 < squash < 0.9, f"smallest element is at {squash:.3f} of its rest area"


# ---------------------------------------------------------------- 8
def test_bodies_must_start_apart():
    model = D.Model(device=DEV); model.body_contact = "ipc"
    _soft(model, (8, 8), (0.0, 0.0)); _soft(model, (8, 8), (8 * PX, 0.0))      # face to face
    model.finalize()
    try:
        D.ImplicitIntegrator().simulate(model, 1e-3)
    except ValueError as e:
        assert "start apart" in str(e)
    else:
        raise AssertionError("two bodies that touch at the start were accepted")


if __name__ == "__main__":
    test_flat_sliding_costs_nothing();        print("flat sliding costs no energy            OK")
    test_derivatives();                       print("gradient and curvature vs differences   OK")
    test_slides_at_the_coulomb_speed();       print("slides at the Coulomb speed             OK")
    test_a_stack_carries_its_weight();        print("a stack carries its weight              OK")
    test_a_grip_needs_friction();             print("a grip needs friction                   OK")
    test_nothing_passes_through();            print("nothing passes through                  OK")
    test_a_prescribed_body_is_swept();        print("a prescribed body is swept              OK")
    test_bodies_must_start_apart();           print("bodies that touch are refused           OK")
    print("IPC contact tests PASSED")
