"""A forearm on a pivot, flexed by an artificial muscle — actuation by springs and pins.

The forearm is a rigid body, pinned to the world at the elbow (add_pin: a point that may
turn but not move). The muscle is a spring from a point on the forearm to an anchor on the
upper arm, which does not move (add_spring to the world). Contracting the muscle is
shortening the spring's rest length, set_spring_length — a McKibben muscle, a shape-memory
coil, a motor on a lead screw.

With gravity on, the forearm settles where the muscle's moment about the elbow balances its
weight's; that angle has a closed form in one unknown, and each settled pose is checked
against it.

    python examples/muscle_arm.py            # --viz to watch it run, --plot to save muscle_arm.png

Per unit depth; SI units.
"""
import sys
import numpy as np
import warp as wp
import Diff_2D_FEM as D

wp.init()
PX = 0.5e-3
LEN, THICK = 0.040, 0.004                     # the forearm: 40 mm x 4 mm
INSERT = np.array([0.010, 0.002])             # where the muscle attaches, from the elbow
ANCHOR = np.array([0.0, 0.030])               # its other end, on the upper arm
K = 2.0e3                                     # muscle stiffness [N/m per metre of stretch]
RHO, G0, DT = 1100.0, 9.81, 2.0e-3

model = D.Model()
model.damping = 40.0                          # enough to settle each pose in half a second
arm = model.add_affine_body(np.ones((int(THICK / PX), int(LEN / PX)), np.uint8),
                            physical_size=(LEN, THICK), center=(LEN / 2, 0.0), density=RHO)
model.finalize()
model.gravity = (0.0, -G0)
elbow = model.add_pin(arm, (0.0, 0.0))                                 # a pivot fixed in the world
muscle = model.add_spring(arm, INSERT, None, ANCHOR, stiffness=K)      # relaxed as built
rest = model.spring_force(muscle)[1]

mass = np.asarray(model.me.mass)[model.body_nodes(arm)]
weight = mass.sum() * G0
com = (mass[:, None] * model.rest_x(arm)).sum(0) / mass.sum()


def balance(length):
    """The angle at which the muscle's moment about the elbow equals the weight's."""
    def moment(th):
        c, s = np.cos(th), np.sin(th); R = np.array([[c, -s], [s, c]])
        p = R @ INSERT; d = ANCHOR - p; ln = np.linalg.norm(d)
        f = K * (ln - length) * d / ln                                 # the muscle's pull at p
        g = R @ com
        return p[0] * f[1] - p[1] * f[0] - g[0] * weight
    lo, hi = np.radians(-80.0), np.radians(80.0)
    for _ in range(80):                                                # bisection
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if moment(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


integ = D.ImplicitIntegrator()
view = D.Renderer(model) if "--viz" in sys.argv else None
print(f"forearm {LEN * 1e3:.0f} mm, weight {weight:.2f} N/m; muscle {rest * 1e3:.2f} mm at rest, "
      f"{K:.0f} N/m per metre")
shots = []
for contraction in (0.0, 0.10, 0.20, 0.30):
    length = rest * (1.0 - contraction)
    model.set_spring_length(muscle, length)
    for _ in range(300):
        integ.simulate(model, DT)
        if view: view.draw()
    angle = model.affine_state(0)["angle"]
    tension, now = model.spring_force(muscle)
    gap = model.spring_force(elbow)[1]
    shots.append((contraction, model.get_x()))
    print(f"  muscle shortened {contraction:4.0%}: forearm at {np.degrees(angle):+6.2f} deg "
          f"(statics {np.degrees(balance(length)):+6.2f}), tension {tension:5.2f} N/m, "
          f"muscle {now * 1e3:5.2f} mm long, elbow opens {gap * 1e9:.1f} nm", flush=True)

if "--plot" in sys.argv:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from Diff_2D_FEM import viz
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for (c, xs), col in zip(shots, ("#C9D6DF", "#9CC3D5", "#4C8DAE", "#1F4E66")):
        viz.draw(model, ax=ax, x=xs, color=col, springs=False)
        p = xs[model._springs[muscle]["a"]]
        ax.plot([ANCHOR[0], p[0]], [ANCHOR[1], p[1]], color=col, lw=2.0)
    ax.plot(*ANCHOR, "ks", ms=5); ax.plot(0, 0, "ko", ms=5)
    ax.set_xlim(-0.008, 0.045); ax.set_ylim(-0.012, 0.041); ax.set_aspect("equal")
    ax.set_title("muscle shortened by " + ", ".join(f"{c:.0%}" for c, _ in shots))
    fig.tight_layout(); fig.savefig("muscle_arm.png", dpi=130)
    print("wrote muscle_arm.png")

if view:
    view.hold()                                # the window stays until it is closed
