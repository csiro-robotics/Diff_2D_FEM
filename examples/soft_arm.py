"""A tendon-driven soft arm pushes a rigid block across the ground.

The arm hangs from a clamp. Two cables run down it, one each side of the mid-line;
pulling one bends the arm that way (the other goes slack — a cable cannot push). The
block is a free RIGID body: six unknowns, standing on the ground under gravity, held by
friction until the arm reaches it. Arm and block touch through IPC contact with friction:
the arm's flank meets the block's side, tips it, and slides it along.

    python examples/soft_arm.py            # --viz to watch it run, --plot to save soft_arm.png

Per unit depth; SI units.
"""
import sys
import time
import numpy as np
import warp as wp
import Diff_2D_FEM as D

wp.init()
PX = 1.0e-3                                   # 1 mm pixels everywhere
ARM_W, ARM_H = 10, 80                         # the arm: 10 mm x 80 mm
TOP = 0.089                                   # height of the clamp
BLOCK = 16                                    # the block: 16 mm square
DT, T_PULL = 2.0e-3, 220.0                    # cable tension at full pull [N/m]

model = D.Model()
model.add_ground(height=0.0, mu=0.5)
model.damping = 20.0
model.body_friction = 0.5                     # between the arm and the block
arm = model.add_soft_body(np.ones((ARM_H, ARM_W), np.uint8),
                          physical_size=(ARM_W * PX, ARM_H * PX),
                          center=(0.0, TOP - 0.5 * ARM_H * PX), E=1.0e6, nu=0.40, density=1100.0)
block = model.add_affine_body(np.ones((BLOCK, BLOCK), np.uint8),
                              physical_size=(BLOCK * PX, BLOCK * PX),
                              center=(0.018, 0.5 * BLOCK * PX + PX / 8), density=600.0)
model.finalize()
model.gravity = (0.0, -9.81)

X = model.rest_x(arm)
model.set_kinematic_constraint(arm, np.where(X[:, 1] > X[:, 1].max() - 1e-9)[0])   # the clamp
rows = np.unique(np.round(X[:, 1], 9))[::-1]                                      # top to bottom
right = model.add_tendon(arm, through=[(+0.003, y) for y in rows])
left = model.add_tendon(arm, through=[(-0.003, y) for y in rows])


def block_x():
    return model.affine_state(0)["position"][0]


integ = D.ImplicitIntegrator()
view = D.Renderer(model) if "--viz" in sys.argv else None
for _ in range(60):                           # let the block settle on the ground
    integ.simulate(model, DT)
x0 = block_x()
print(f"arm {model.body_nodes(arm).size} nodes + rigid block (6 unknowns); "
      f"{integ.stats(model)['max_iter']} iterations/step at most")

t0 = time.time(); frames = []
for phase, (tr, tl, n) in (("pull right", (T_PULL, 0.0, 450)), ("release", (0.0, 0.0, 250))):
    T0 = (model.tendon_tension(right), model.tendon_tension(left))
    for i in range(1, n + 1):
        s = min(i / (0.7 * n), 1.0); s = 0.5 * (1 - np.cos(np.pi * s))            # eased ramp
        model.set_tendon_tension(right, T0[0] + (tr - T0[0]) * s)
        model.set_tendon_tension(left, T0[1] + (tl - T0[1]) * s)
        integ.simulate(model, DT)
        if view: view.draw()
        if i % 10 == 0:
            frames.append(model.get_x())
    tip = model.get_body_x(arm)[np.argmin(X[:, 1])]
    st = model.affine_state(0)
    print(f"  {phase:10}: arm tip at ({tip[0] * 1e3:+6.1f}, {tip[1] * 1e3:5.1f}) mm; block moved "
          f"{(st['position'][0] - x0) * 1e3:+6.2f} mm, tilted {np.rad2deg(st['angle']):+5.2f} deg",
          flush=True)
    if phase == "pull right":
        pushed = model.get_x()

gap = model.get_body_x(block)[:, 0].min() - model.get_body_x(arm)[:, 0].max()
A = model.affine_state(0)["A"]
print(f"block ends {(block_x() - x0) * 1e3:+.2f} mm from where it started, strained by "
      f"{np.abs(A.T @ A - np.eye(2)).max():.1e};  {time.time() - t0:.0f} s of wall time")

if "--plot" in sys.argv:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from Diff_2D_FEM import viz
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    viz.draw(model, ax=axes[0], x=pushed); axes[0].set_title("cable pulled")
    viz.draw(model, ax=axes[1]); axes[1].set_title("released")
    for ax in axes:
        ax.set_xlim(-0.03, 0.07); ax.set_ylim(-0.005, 0.1); ax.axvline(x0, color="k", lw=0.6, ls=":")
    fig.tight_layout(); fig.savefig("soft_arm.png", dpi=130)
    print("wrote soft_arm.png")

if view:
    view.hold()                                # the window stays until it is closed
