"""A hard-magnetic soft beam in a uniform field — actuation by external forces.

A strip of elastomer loaded with magnetised particles carries a magnetisation M that turns
with the material. In a field B its energy is -(F M) . B per unit rest volume (Zhao et al.,
"Mechanics of hard-magnetic soft materials", JMPS 2019): linear in the deformation gradient
F, so the nodal forces it derives are the same in every configuration,

    f_a = A (grad N_a . M) B        for each element of rest area A, each of its nodes a,

and are computed once and handed to the solver with set_external_force. They sum to zero
over an element — what is left is a couple, M x B per unit volume — and the beam bends to
turn its magnetisation into the field.

Three runs, each settled:
  1. a weak field across a beam magnetised along its length: the tip deflection of beam
     theory for a uniformly distributed couple, m L^3 / (3 E' I) with m = M B h;
  2. a strong field: the beam curls until its far end points along B;
  3. the last third magnetised the other way: the root still turns up, the end turns
     back down — an S.

    python examples/magnetic_beam.py            # add --plot to save magnetic_beam.png

Per unit depth; SI units.
"""
import sys
import numpy as np
import warp as wp
import Diff_2D_FEM as D

wp.init()
NX, NY = 120, 8                               # 30 mm x 2 mm, 0.25 mm elements
L, H = 0.030, 0.002
E, NU, RHO = 1.0e6, 0.40, 1800.0
M0 = 1.0e5                                    # magnetisation [A/m]


def beam(magnetisation):
    """A cantilever clamped at its left end. magnetisation(X) -> (n, 2), in A/m, at the
    rest positions X of the element centres."""
    model = D.Model()
    model.damping = 2000.0                    # heavily damped: only the settled shape matters
    h = model.add_soft_body(np.ones((NY, NX), np.uint8), physical_size=(L, H),
                            center=(L / 2, 0.0), E=E, nu=NU, density=RHO)
    model.finalize()
    X = model.rest_x(h)
    model.set_kinematic_constraint(h, np.where(X[:, 0] < 1e-9)[0])

    # f_a = A (grad N_a . M) per unit field, on the three nodes of every element
    tri = np.asarray(model.me.tri)[model.body_elements(h)] - model.body_nodes(h)[0]
    G = np.asarray(model.me.D_m_inv)[model.body_elements(h)]          # rows: grad N_1, grad N_2
    A = np.asarray(model.me.rest_area)[model.body_elements(h)]
    Mv = magnetisation(X[tri].mean(1))
    g1, g2 = (G[:, 0] * Mv).sum(1), (G[:, 1] * Mv).sum(1)
    shape = np.zeros(len(X))                  # sum over elements of A (grad N_a . M)
    np.add.at(shape, tri[:, 0], -A * (g1 + g2))
    np.add.at(shape, tri[:, 1], A * g1)
    np.add.at(shape, tri[:, 2], A * g2)
    return model, h, X, shape


def settle(model, steps=12, dt=5.0e-2):
    integ = D.ImplicitIntegrator(max_iter=2000, eps=1e-14, self_collision=False, friction=False)
    for _ in range(steps):
        integ.simulate(model, dt)


def apply(model, h, shape, B):
    """Switch the field B = (Bx, By) [T] on."""
    model.set_external_force(h, shape[:, None] * np.asarray(B, float)[None, :])


def tip(model, h, X):
    """Where the end face is, and which way the beam points there [deg]."""
    x = model.get_body_x(h)
    end = np.where(X[:, 0] > L - 1e-9)[0]; before = np.where(np.abs(X[:, 0] - (L - 8 * L / NX)) < 1e-9)[0]
    d = x[end].mean(0) - x[before].mean(0)
    return x[end].mean(0), np.degrees(np.arctan2(d[1], d[0]))


along = lambda Xc: np.tile([M0, 0.0], (len(Xc), 1))                   # magnetised along the beam

# ---- 1. a weak field: beam theory -----------------------------------------------------------
model, h, X, shape = beam(along)
B = 0.5e-3
apply(model, h, shape, (0.0, B)); settle(model)
w = tip(model, h, X)[0][1]
theory = (M0 * B * H) * L ** 3 / (3.0 * (E / (1.0 - NU ** 2)) * H ** 3 / 12.0)
print(f"field {B * 1e3:4.1f} mT across the beam: tip rises {w * 1e3:.3f} mm; beam theory {theory * 1e3:.3f} mm "
      f"({w / theory - 1:+.1%} — linear triangles, {NY} through the thickness, are that much too stiff)")

# ---- 2. a strong field: the beam turns into it ----------------------------------------------
shots = []
for B in (10.0e-3, 30.0e-3, 100.0e-3):
    apply(model, h, shape, (0.0, B)); settle(model)
    p, a = tip(model, h, X)
    shots.append((f"{B * 1e3:.0f} mT", model.get_x()))
    print(f"field {B * 1e3:4.1f} mT across the beam: tip at ({p[0] * 1e3:5.2f}, {p[1] * 1e3:5.2f}) mm, "
          f"pointing {a:5.1f} deg from where it started (the field is at 90)")

# ---- 3. the last third magnetised the other way ----------------------------------------------
split = lambda Xc: np.where(Xc[:, :1] < 2 * L / 3, 1.0, -1.0) * np.array([M0, 0.0])
model2, h2, X2, shape2 = beam(split)
apply(model2, h2, shape2, (0.0, 30.0e-3)); settle(model2)
x = model2.get_body_x(h2)
p, a = tip(model2, h2, X2)
top = x[np.argmax(x[:, 1])]
print(f"last third reversed, 30 mT: the beam climbs to {top[1] * 1e3:.2f} mm at x = {top[0] * 1e3:.1f} mm, then "
      f"turns back: tip at ({p[0] * 1e3:.2f}, {p[1] * 1e3:.2f}) mm, pointing {a:+.1f} deg — an S")

if "--plot" in sys.argv:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from Diff_2D_FEM import viz
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for (label, xs), c in zip(shots, ("#9CC3D5", "#4C8DAE", "#1F4E66")):
        viz.draw(model, ax=axes[0], x=xs, color=c)
    axes[0].set_title("one domain: " + ", ".join(s[0] for s in shots))
    viz.draw(model2, ax=axes[1]); axes[1].set_title("last third reversed, 30 mT")
    for ax in axes:
        ax.set_xlim(-0.002, 0.033); ax.set_ylim(-0.008, 0.030); ax.set_aspect("equal")
        ax.annotate("B", xy=(0.030, 0.027), xytext=(0.030, 0.019), ha="center",
                    arrowprops=dict(arrowstyle="->", color="k"))
    fig.tight_layout(); fig.savefig("magnetic_beam.png", dpi=130)
    print("wrote magnetic_beam.png")
