"""Strain and stress per element (core/fields.py, Model.element_strain / element_stress) and
their pictures (viz.draw / viz.animate with field=...).

  1. Closed forms: a uniform stretch s gives von Mises strain 2e/3 with e = (s^2 - 1)/2,
     and von Mises stress mu (s^2 - 1)/s^2 — exactly, in every element; a rigid rotation
     and translation gives neither.
  2. The solver's own equilibrium: equal pressure in a cavity and outside puts every
     element in the in-plane state sigma = -p I (any geometry), so the Cauchy stress read
     back from the solved positions must be that. Out of the plane, plane strain sets
     sigma_zz = lam ln J / J, which is not -p — so the von Mises stress is |p + sigma_zz|,
     not 0.
  3. Signed fields (signed=True): the principal value of larger magnitude keeps its sign —
     tension +, compression -, the same however the body is turned; a cantilever sagging
     under its own weight is in tension along its top face and in compression along its
     bottom face at the clamp.
  4. The pictures: draw and animate with a field run and write an animation.

    python tests/test_fields.py
"""
import os, sys, tempfile
import numpy as np
import warp as wp
import Diff_2D_FEM as D
from Diff_2D_FEM.core import fields as FLD

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_static as REF                                          # noqa: E402

E, NU, RHO = 1.0e6, 0.40, 1100.0
MU, LAM = E / (2 * (1 + NU)), E * NU / ((1 + NU) * (1 - 2 * NU))


def _block(mask=None):
    m = np.ones((8, 12), np.uint8) if mask is None else mask
    model = D.Model(device="cpu")
    model.add_disc(center=(0.0, -5.0), radius=0.01)
    h = model.add_soft_body(m, physical_size=(0.012, 0.008), center=(0.03, 0.05), E=E, nu=NU, density=RHO)
    model.finalize()
    return model, h


def test_closed_forms():
    model, h = _block()
    X = model.me.rest_pos.astype(np.float64)
    assert model.element_strain(X).max() < 1e-12 and model.element_stress(X).max() < 1e-6     # at rest (rounding only)
    for s in (0.9, 1.05, 1.3):
        x = s * (X - X.mean(0)) + X.mean(0)
        e = 0.5 * (s * s - 1.0)
        assert np.allclose(model.element_strain(x), 2.0 * abs(e) / 3.0, rtol=1e-12, atol=0)
        assert np.allclose(model.element_stress(x), MU * abs(s * s - 1.0) / (s * s), rtol=1e-9, atol=0)
        S = FLD.cauchy_stress(FLD.deformation_gradient(model.me, x), model.me.mu, model.me.lam)
        assert np.allclose(S[:, 2, 2], LAM * np.log(s * s) / (s * s), rtol=1e-9)       # plane strain: sigma_zz
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    x = X @ R.T + [0.004, -0.01]
    assert model.element_strain(x).max() < 1e-12
    assert model.element_stress(x).max() < 1e-6 * E
    assert np.array_equal(model.element_strain(x, handle=h), model.element_strain(x)[model.body_elements(h)])


def test_stress_of_a_hydrostatic_state():
    wp.init()
    p = 6.0e4
    m = np.ones((20, 14), np.uint8); m[5:15, 4:10] = 0; m[2:4, 8:12] = 0
    model = D.Model(device="cpu"); model.damping = 2000.0
    model.add_disc(center=(0.0, -5.0), radius=0.01)
    h = model.add_soft_body(m, physical_size=(0.014, 0.020), center=(0.03, 0.05), E=E, nu=NU, density=RHO)
    model.finalize()
    for c in model.add_pressure_chambers(h) + [model.add_ambient_pressure(h)]:
        model.set_pressure(c, p)
    integ = D.ImplicitIntegrator(max_iter=2000, eps=1e-14, tol_x=0.0, self_collision=False, friction=False)
    for _ in range(30):
        integ.simulate(model, 5e-3)
    S = FLD.cauchy_stress(FLD.deformation_gradient(model.me, model.get_x()), model.me.mu, model.me.lam)
    for i, j in ((0, 0), (1, 1)):
        assert np.abs(S[:, i, j] / -p - 1.0).max() < 1e-3, \
            f"sigma_{'xy'[i]}{'xy'[j]} {S[:, i, j].min():.1f} to {S[:, i, j].max():.1f} Pa, expected {-p:.0f}"
    assert np.abs(S[:, 0, 1]).max() < 1e-3 * p, "shear stress under pure pressure"
    J = np.linalg.det(FLD.deformation_gradient(model.me, model.get_x()))
    assert np.allclose(S[:, 2, 2], LAM * np.log(J) / J, rtol=1e-9)
    assert np.allclose(model.element_stress(), np.abs(-p - S[:, 2, 2]), rtol=2e-3)


def test_signed_fields():
    model, h = _block()
    X = model.me.rest_pos.astype(np.float64); c = X.mean(0)
    for (s1, s2), e_expect in (((1.10, 0.95), 0.5 * (1.10 ** 2 - 1)),       # tension dominates: +0.105
                               ((1.02, 0.90), 0.5 * (0.90 ** 2 - 1))):      # compression dominates: -0.095
        J = s1 * s2
        sig = [(MU * si * si + LAM * np.log(J) - MU) / J for si in (s1, s2)]
        s_expect = sig[int(np.argmax(np.abs(sig)))]
        for th in (0.0, 0.6, -1.3):                                         # any orientation of the stretch
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            x = (X - c) @ (R @ np.diag([s1, s2]) @ R.T).T + c
            assert np.allclose(model.element_strain(x, signed=True), e_expect, rtol=1e-10, atol=0)
            assert np.allclose(model.element_stress(x, signed=True), s_expect, rtol=1e-9, atol=0)
            assert np.allclose(model.element_strain(x @ R.T, signed=True), e_expect, rtol=1e-10, atol=0)  # + a turn

    # a cantilever clamped at x = 0, sagging under its own weight: top face pulled, bottom squeezed
    model = D.Model(device="cpu")
    h = model.add_soft_body(np.ones((8, 40), np.uint8), physical_size=(0.040, 0.008), center=(0.020, 0.0),
                            E=E, nu=NU, density=RHO)
    model.finalize()
    X = model.rest_x(h)
    left = np.where(X[:, 0] < 1e-9)[0]
    x, res = REF.solve_static(model.me, left, fext=np.asarray(model.me.mass)[:, None] * [0.0, -200.0])
    assert res < 1e-8 and x[:, 1].min() < -1e-4, "the beam should sag"
    cen = X[np.asarray(model.me.tri)].mean(1)
    near = cen[:, 0] < 0.010                                                # the first quarter, where the moment is largest
    top, bottom = near & (cen[:, 1] > 0.003), near & (cen[:, 1] < -0.003)
    for field in ("strain", "stress"):
        v = model._field(field, x, None, True)
        assert (v[top] > 0).all(), f"{field}: the top face should be in tension"
        assert (v[bottom] < 0).all(), f"{field}: the bottom face should be in compression"


def test_pictures():
    try:
        import matplotlib; matplotlib.use("Agg")
        from PIL import Image
    except ImportError:
        print("matplotlib / pillow missing: pictures not tested"); return
    import matplotlib.pyplot as plt
    from Diff_2D_FEM import viz
    model, h = _block()
    X = model.me.rest_pos.astype(np.float64)
    frames = [X + np.c_[0.1 * k * (X[:, 0] - X[:, 0].min()), np.zeros(len(X))] for k in range(3)]   # a growing stretch
    for field in ("strain", "stress"):
        ax = viz.draw(model, x=frames[-1], field=field)
        assert len(ax.figure.axes) == 2, "no colour bar"
        plt.close(ax.figure)
    ax = viz.draw(model, x=frames[-1], field="stress", signed=True)
    lo, hi = ax.figure.axes[1].get_ylim()
    assert np.isclose(lo, -hi), "a signed field's scale should be symmetric about 0"
    plt.close(ax.figure)
    ax = viz.draw(model, values=model.me.mu)                         # the old use is unchanged: no colour bar
    assert len(ax.figure.axes) == 1; plt.close(ax.figure)
    with tempfile.TemporaryDirectory() as d:
        for signed in (False, True):
            path = viz.animate(model, frames, os.path.join(d, f"run{signed}.gif"), field="strain", signed=signed,
                               fps=5, titles=[f"frame {k}" for k in range(3)])
            im = Image.open(path)
            assert im.n_frames == 3 and im.is_animated

        # a batch side by side: two environments, a pinned body and a prescribed one in each
        model = D.Model(device="cpu")
        for e in (0, 1):
            model.add_soft_body(np.ones((8, 12), np.uint8), physical_size=(0.012, 0.008), center=(0.03, 0.05),
                                E=E, nu=NU, density=RHO, env_id=e)
            model.add_kinematic_body(np.ones((2, 12), np.uint8), physical_size=(0.012, 0.002), center=(0.03, 0.04),
                                     env_id=e)
        model.finalize()
        for e in (0, 1):
            model.add_pin(2 * e, (0.024, 0.046))
        X = model.me.rest_pos.astype(np.float64)
        frames = [X + np.c_[0.05 * k * (X[:, 0] - 0.024), np.zeros(len(X))] for k in range(3)]
        path = viz.animate(model, frames, os.path.join(d, "batch.gif"), envs="all", field="stress", signed=True,
                           fps=5, titles=[f"frame {k}" for k in range(3)], panel_titles=["first", "second"],
                           bounds=((0.02, 0.035), (0.045, 0.06)))
        im = Image.open(path)
        assert im.n_frames == 3 and im.is_animated


if __name__ == "__main__":
    test_closed_forms();                  print("strain / stress vs closed forms           OK")
    test_stress_of_a_hydrostatic_state(); print("stress of a solved hydrostatic state      OK")
    test_signed_fields();                 print("signed fields: tension +, compression -   OK")
    test_pictures();                      print("draw / animate with a field               OK")
    print("field tests PASSED")
