"""Smoke test: the package imports and a block pressed on a disc develops contact force.

    python -m pytest tests/test_smoke.py        # or: python tests/test_smoke.py
"""
import numpy as np
import warp as wp
import Diff_2D_FEM as D


def test_import_surface():
    for name in ("Model", "SimConfig", "Mesh", "build_scene", "objects", "viz",
                 "ImplicitIntegrator", "AdjointPNCG2D"):
        assert hasattr(D, name)


def test_block_pressed_on_a_disc_makes_contact():
    wp.init()
    model = D.Model()
    model.add_disc(center=(0.05, 0.025), radius=0.006, mu=0.6)
    h = model.add_soft_body(np.ones((40, 16), np.uint8), physical_size=(0.016, 0.040),
                            center=(0.05, 0.060), E=1.0e6, nu=0.40, density=1100.0)
    model.finalize()
    x0 = model.get_body_x(h)
    top = np.where(x0[:, 1] > x0[:, 1].max() - 1e-4)[0]
    model.set_kinematic_constraint(h, top)

    integ = D.ImplicitIntegrator(max_iter=100)
    fmax = 0.0
    for t in range(1, 61):
        x = model.get_body_x(h); x[top, 1] = x0[top, 1] - 0.02 * t / 60
        model.set_kinematic_target(h, x)
        integ.simulate(model, dt=1e-3)
        fmax = max(fmax, float(np.linalg.norm(integ.collider_force(model)[0])))
    # positions stay finite (stable) and a contact force developed once the block met the disc
    assert np.all(np.isfinite(model.get_body_x(h)))
    assert fmax > 1.0, f"expected a contact reaction, got max |F|={fmax}"


if __name__ == "__main__":
    test_import_surface(); test_block_pressed_on_a_disc_makes_contact()
    print("smoke test PASSED")
