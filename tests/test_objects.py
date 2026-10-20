"""Arbitrary-shape obstacles (signed-distance level sets).

  1. The host-side mirror core.objects.phi must agree with the device level-set
     core.sdf.sdf_eval, or a plot / contact-arc measurement drifts from what the solver
     actually contacted.
  2. |grad phi| == 1 outside the object for the true distance fields (box, triangle, disc).
  3. A soft block pressed onto each shape develops a contact reaction and stays finite.
  4. In the sliding branch the friction reaction is mu times the normal reaction, for
     every shape (the level-set friction kernels carry a |grad phi| factor that must
     collapse to 1).

    python tests/test_objects.py
"""
import numpy as np
import warp as wp
import Diff_2D_FEM as D
from Diff_2D_FEM.core import sdf as SDF
from Diff_2D_FEM.core import objects as OBJ
from Diff_2D_FEM import real, vec2, np_real              # the working precision (see core/precision.py)

DEV = "cuda" if wp.is_cuda_available() else "cpu"
MU = 0.6                                                 # friction of the collider under test
SHAPES = {"disc": dict(rx=0.006), "ellipse": dict(rx=0.008, ry=0.005),
          "box": dict(rx=0.007, ry=0.005, corner_radius=0.002),
          "triangle": dict(rx=0.007, corner_radius=0.002)}


@wp.kernel
def _probe(kind: int, cx: real, cy: real, rx: real, ry: real, rref: real,
           pts: wp.array(dtype=vec2), d: wp.array(dtype=real), g: wp.array(dtype=vec2)):
    i = wp.tid()
    s = SDF.sdf_eval(kind, cx, cy, rx, ry, rref, pts[i])
    d[i] = s.d; g[i] = s.g


def _scene(shape, center):
    model = D.Model(device=DEV)
    model.add_object(shape, center=center, mu=MU, **SHAPES[shape])
    h = model.add_soft_body(np.ones((40, 16), np.uint8), physical_size=(0.016, 0.040),
                            center=(0.05, 0.060), E=1.0e6, nu=0.40, density=1100.0)
    model.finalize()
    x0 = model.get_body_x(h)
    top = np.where(x0[:, 1] > x0[:, 1].max() - 1e-4)[0]
    model.set_kinematic_constraint(h, top)
    return model, h, x0, top


def test_host_mirror_matches_device_levelset():
    wp.init()
    rng = np.random.default_rng(0)
    for shape, kw in SHAPES.items():
        m = D.Model(device=DEV); m.add_object(shape, center=(0.05, 0.03), **kw)
        o = m.object
        P = np.array([o.cx, o.cy]) + rng.uniform(-0.02, 0.02, size=(4000, 2))
        P = P[OBJ.phi(P, o) > 1e-5]                         # outside: where contact lives
        assert len(P) > 1000
        pw = wp.array(P.astype(np_real), dtype=vec2, device=DEV)
        d = wp.zeros(len(P), dtype=real, device=DEV); g = wp.zeros(len(P), dtype=vec2, device=DEV)
        wp.launch(_probe, dim=len(P), inputs=[o.kind, o.cx, o.cy, o.rx, o.ry, o.rref, pw, d, g], device=DEV)
        err = np.abs(d.numpy() - OBJ.phi(P, o)).max()
        assert err < 2e-6, f"{shape}: host phi and device sdf_eval differ by {err:.2e} m"
        if shape != "ellipse":                              # ellipse is an algebraic level set, not a distance
            gn = np.linalg.norm(g.numpy(), axis=1)
            assert np.abs(gn - 1.0).max() < 1e-3, f"{shape}: |grad phi| ranges {gn.min():.4f}..{gn.max():.4f}"


def test_every_shape_develops_contact():
    wp.init()
    for shape in SHAPES:
        model, h, x0, top = _scene(shape, center=(0.05, 0.030))
        integ = D.ImplicitIntegrator(max_iter=100)
        fmax = 0.0
        for t in range(1, 61):
            x = model.get_body_x(h); x[top, 1] = x0[top, 1] - 0.012 * t / 60
            model.set_kinematic_target(h, x)
            integ.simulate(model, dt=1e-3)
            fmax = max(fmax, float(np.linalg.norm(integ.collider_force(model)[0])))
        x = model.get_body_x(h)
        assert np.all(np.isfinite(x)), f"{shape}: non-finite positions"
        assert fmax > 1.0, f"{shape}: expected a contact reaction, got max |F| = {fmax}"
        assert OBJ.phi(x, model.object).min() > 0.0, f"{shape}: a node ended up inside the object"


def test_sliding_friction_is_mu_times_normal():
    wp.init()
    for shape in SHAPES:
        # place the object by actual clearance: nearest node half a barrier-width from its surface
        model, h, x0, top = _scene(shape, center=(0.05, 0.0))
        cfg = model.cfg
        def clearance(oy):
            o = model.object; o.cy = oy
            return OBJ.phi(x0, o).min()
        lo, hi = x0[:, 1].min() - 0.05, x0[:, 1].min()
        assert clearance(lo) > 0.5 * cfg.ipc_d_hat > clearance(hi)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if clearance(mid) > 0.5 * cfg.ipc_d_hat else (lo, mid)
        model.set_collider_center(0, (0.05, lo))
        integ = D.ImplicitIntegrator(max_iter=1)
        s = integ.attach(model, 1e-3)
        x = model.get_x()
        xn = x.copy(); xn[:, 0] -= 3e-6                    # 3 um of tangential slide this step
        s.x.assign(x); s.xn.assign(xn)
        Fn = integ.collider_force(model, friction=False)[0].astype(np.float64)
        Ff = integ.collider_force(model, friction=True)[0].astype(np.float64) - Fn
        assert np.linalg.norm(Fn) > 1e-3, f"{shape}: no normal reaction -> not in contact"
        ratio = np.linalg.norm(Ff) / np.linalg.norm(Fn)
        assert abs(ratio - MU) < 0.03 * MU, \
            f"{shape}: |F_friction|/|F_normal| = {ratio:.4f}, expected mu = {MU}"


if __name__ == "__main__":
    test_host_mirror_matches_device_levelset(); print("host mirror == device level-set   OK")
    test_every_shape_develops_contact();        print("every shape develops contact      OK")
    test_sliding_friction_is_mu_times_normal(); print("sliding friction == mu * normal   OK")
    print("object tests PASSED")
