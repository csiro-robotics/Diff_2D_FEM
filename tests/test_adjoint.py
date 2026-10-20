"""The adjoint against finite differences.

A soft block is pressed and sheared over T implicit steps; a non-uniform per-element density
rho scales the stiffness; the objective is J = c . x_T. The adjoint's dJ/drho is compared with
central finite differences along the gradient itself (tests magnitude and sign) and along
random directions (tests direction), for an unconstrained body, a disc in contact, and a
box in contact (the level-set path).

Three things this test has to get right, or it measures noise instead of the gradient:

  * device="cpu". GPU atomics are not order-deterministic, and five identical GPU rollouts
    spread J by ~1e-6 — the same size as the finite-difference signal. The CPU device is
    exactly repeatable.
  * tol_x = 1e-8, not smaller. Positions near 0.05 m have a float32 spacing of ~4e-9 m, so
    a tighter increment tolerance is unreachable and every step just runs to max_iter.
  * dt = 2e-4. At dt = 1e-3 this block is stiffness-dominated (h^2 K outweighs M ~1700x) and
    float32 CG stalls early, leaving J accurate to only ~5e-7 as a function of rho.

  * the finite-difference step. With a box in contact the active set changes with rho more
    abruptly than for a disc, so a large step measures a secant across that change while the
    adjoint gives the tangent with the active set frozen: along the gradient the mismatch is
    7.4% at h=0.2, 4.3% at 0.1, 3.1% at 0.05, 1.3% at 0.025. For the disc it is already 0.07%
    at h=0.05. Hence the per-case steps below.

FRICTION IS OFF. The adjoint operator contains the elastic, contact and self-contact
tangents but no friction tangent, so with friction on it linearises a different system; see
the README ("Differentiability").

    python tests/test_adjoint.py
"""
import numpy as np
import warp as wp
import Diff_2D_FEM as D

T, DT = 12, 2.0e-4
H_FD = {"elastic": 0.05, "disc": 0.05, "box": 0.025}      # see the docstring


def _gradient_vs_fd(mode, h_fd=None):
    h_fd = H_FD[mode] if h_fd is None else h_fd
    model = D.Model(device="cpu")
    oy = -1.0 if mode == "elastic" else 0.0335            # "elastic": obstacle far away, no contact
    if mode == "box":
        model.add_object("box", center=(0.05, oy), rx=0.006, ry=0.006, corner_radius=0.002, mu=0.0)
    else:
        model.add_disc(center=(0.05, oy), radius=0.006, mu=0.0)
    h = model.add_soft_body(np.ones((10, 6), np.uint8), physical_size=(0.012, 0.024),
                            center=(0.05, 0.052), E=1.0e6, nu=0.40, density=1100.0)
    model.finalize()
    x0 = model.get_body_x(h)
    top = np.where(x0[:, 1] > x0[:, 1].max() - 1e-5)[0]
    model.set_kinematic_constraint(h, top)
    integ = D.ImplicitIntegrator(max_iter=3000, eps=1e-14, tol_x=1e-8, conv_check=8,
                                 self_collision=False, friction=False)
    adj = integ.adjoint(model, DT)
    rng = np.random.default_rng(0)
    rho0 = 0.45 + 0.5 * rng.random(adj.n_tri)
    low = np.where(x0[:, 1] < np.median(x0[:, 1]))[0]
    c = np.zeros_like(model.get_x(), dtype=np.float64); c[low, 0] = 1.0 / len(low)

    def rollout(rho):
        adj.set_density(rho); model.reset()
        xs, contact_steps = [model.get_x().astype(np.float64)], 0
        for t in range(1, T + 1):
            x = model.get_body_x(h)
            x[top, 1] = x0[top, 1] - 0.003 * t / T        # press 3 mm
            x[top, 0] = x0[top, 0] + 0.002 * t / T        # shear 2 mm
            model.set_kinematic_target(h, x)
            integ.simulate(model, DT)
            xs.append(model.get_x().astype(np.float64))
            contact_steps += np.linalg.norm(integ.collider_force(model, friction=False)[0]) > 1e-2
        return xs, contact_steps
    J = lambda rho: float((c * rollout(rho)[0][-1]).sum())

    xs, contact_steps = rollout(rho0)
    assert J(rho0) == float((c * xs[-1]).sum()), "CPU rollout is not repeatable"
    if mode == "elastic":
        assert contact_steps == 0
    else:
        assert contact_steps >= T // 2, f"{mode}: in contact on only {contact_steps}/{T} steps"
    g = adj.transient_adjoint(xs, seed={T: c}, use_disc=(mode == "disc"), use_sdf=(mode == "box"))
    assert getattr(adj, "minres_fail", 0) == 0, "an adjoint linear solve did not converge"
    assert np.isfinite(g).all() and np.linalg.norm(g) > 0

    dirs = [g / np.abs(g).max()]
    for _ in range(2):
        d = rng.standard_normal(adj.n_tri); dirs.append(d / np.abs(d).max())
    out = []
    for i, d in enumerate(dirs):
        fd = (J(rho0 + h_fd * d) - J(rho0 - h_fd * d)) / (2 * h_fd)
        gd = float(g @ d)
        out.append((gd, fd, abs(gd - fd) / (np.linalg.norm(g) * np.linalg.norm(d))))
    return out


def _check(mode, tol_along, tol_any):
    res = _gradient_vs_fd(mode)
    gd, fd, _ = res[0]
    rel = abs(gd - fd) / abs(fd)
    assert rel < tol_along, f"{mode}: along the gradient, adjoint {gd:.5e} vs FD {fd:.5e} ({rel:.1%})"
    for gd, fd, nerr in res:
        # error as a fraction of |g||d|: a relative error would blow up for directions that
        # happen to be nearly orthogonal to g, where g.d is itself tiny
        assert nerr < tol_any, f"{mode}: adjoint {gd:.5e} vs FD {fd:.5e}, normalised error {nerr:.2e}"
    return rel, max(r[2] for r in res)


# (along-gradient relative tolerance, any-direction normalised tolerance). Measured values are
# ~0.4% / 0.1% / 1.3% along the gradient; the tolerances leave headroom without being able to
# pass a wrong gradient (a sign flip is a 200% error, a missing contact term tens of percent).
TOL = {"elastic": (0.02, 0.01), "disc": (0.02, 0.01), "box": (0.03, 0.02)}


def test_adjoint_elastic():
    wp.init(); _check("elastic", *TOL["elastic"])


def test_adjoint_disc_contact():
    wp.init(); _check("disc", *TOL["disc"])


def test_adjoint_box_contact():
    wp.init(); _check("box", *TOL["box"])


if __name__ == "__main__":
    wp.init()
    for mode, (ta, tn) in TOL.items():
        rel, nerr = _check(mode, ta, tn)
        print(f"adjoint vs finite differences, {mode:8}: along-gradient error {rel:.2%}, "
              f"worst normalised error {nerr:.1e}   OK")
    print("adjoint tests PASSED")
