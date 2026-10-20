"""A parallel gripper with soft fingertips picks up a block — actuation by prescribed motion.

Two soft pads stand either side of a rigid block that rests on the ground. The back face
of each pad is the jaw: its nodes are kinematically constrained, and every step they are
told where to be (set_kinematic_target) — first closing on the block, then rising. Nothing
else is driven. The pads squash against the block through IPC contact, and whether the
block comes up with them is decided by friction alone: with a friction coefficient of 0.8
between pad and block it is carried; with none it stays where it was. The forces are read
back with contact_force: how hard the pads squeeze, and what they do to the block.

    python examples/pinch_and_lift.py            # --viz to watch it run, --plot to save pinch_and_lift.png

Per unit depth; SI units.
"""
import sys
import time
import numpy as np
import warp as wp
import Diff_2D_FEM as D

wp.init()
PX = 0.5e-3                                   # 0.5 mm pixels
BLOCK, PAD_W, PAD_H = 16, 8, 16               # an 8 mm block; pads 4 mm thick, 8 mm tall
GAP = 2 * PX                                  # between pad and block before closing
SQUEEZE, LIFT = 0.25 * PX, 4.0e-3             # how far past touching the jaws close; the lift
DT, N_CLOSE, N_LIFT, N_HOLD = 1.0e-3, 100, 200, 50


def grip(friction):
    model = D.Model()
    model.body_friction = friction                           # between the pads and the block
    model.add_ground(height=0.0, mu=0.5)
    block = model.add_affine_body(np.ones((BLOCK, BLOCK), np.uint8),
                                  physical_size=(BLOCK * PX, BLOCK * PX),
                                  center=(0.0, BLOCK * PX / 2 + 0.9 * PX / 8), density=600.0)
    pads = [model.add_soft_body(np.ones((PAD_H, PAD_W), np.uint8),
                                physical_size=(PAD_W * PX, PAD_H * PX),
                                center=(side * (BLOCK * PX / 2 + GAP + PAD_W * PX / 2), PAD_H * PX / 2 + 0.8e-3),
                                E=1.0e6, nu=0.40, density=1100.0) for side in (-1, +1)]
    model.finalize()
    model.gravity = (0.0, -9.81)
    for side, pad in zip((-1, +1), pads):                    # the jaw: each pad's outer face
        X = model.rest_x(pad)
        back = X[:, 0].min() if side < 0 else X[:, 0].max()
        model.set_kinematic_constraint(pad, np.where(np.abs(X[:, 0] - back) < 1e-9)[0])

    integ = D.ImplicitIntegrator()
    view = D.Renderer(model, title=f"friction {friction}",                 # room for the lift
                      bounds=((-0.012, 0.0), (0.012, 0.014))) if "--viz" in sys.argv else None
    y0 = model.get_body_x(block)[:, 1].mean()
    for k in range(1, N_CLOSE + N_LIFT + N_HOLD + 1):
        close = min(k / N_CLOSE, 1.0)
        rise = min(max(k - N_CLOSE, 0) / N_LIFT, 1.0)
        for side, pad in zip((-1, +1), pads):
            X = model.rest_x(pad)
            X[:, 0] -= side * (GAP + SQUEEZE) * close
            X[:, 1] += LIFT * rise
            model.set_kinematic_target(pad, X)               # only the constrained nodes follow it
        integ.simulate(model, DT)
        if view: view.draw()
    if view: view.close()
    rose = model.get_body_x(block)[:, 1].mean() - y0
    squeeze = -integ.body_contact_force(model, pads[0])[0]    # what the block does to the left pad
    lift = integ.body_contact_force(model, block, colliders=False)[1]   # what the pads do to the block
    weight = np.asarray(model.me.mass)[model.body_nodes(block)].sum() * 9.81
    return model, rose, squeeze, lift / weight


results = []
for friction in (0.8, 0.0):
    t0 = time.time()
    model, rose, squeeze, carried = grip(friction)
    results.append((friction, model))
    print(f"friction {friction:.1f} between pad and block: the block rises {rose * 1e3:+.3f} mm of the jaws' "
          f"{LIFT * 1e3:.0f} mm; each pad squeezes with {squeeze:.0f} N/m, and the pads' vertical "
          f"force on the block is {carried:+.2f} x its weight;  {time.time() - t0:.0f} s", flush=True)

if "--plot" in sys.argv:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from Diff_2D_FEM import viz
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, (friction, model) in zip(axes, results):
        viz.draw(model, ax=ax); ax.set_title(f"friction {friction:.1f}")
        ax.set_xlim(-0.012, 0.012); ax.set_ylim(-0.001, 0.014); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig("pinch_and_lift.png", dpi=130)
    print("wrote pinch_and_lift.png")
