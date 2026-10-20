"""A finger with bones: rigid phalanges on flexure hinges, closed by one cable.

One pixel mask, four materials' worth of behaviour. Three stretches of the finger are
made RIGID (add_affine_region): each becomes six unknowns instead of a few hundred nodes,
and the soft material next to it stays attached because the two share their interface
nodes. Between the bones the mask is notched down to a thin strip on the back of the
finger — a flexure hinge. A cable down the palm side spans the notches; pulling it closes
them, and the finger wraps the disc.

    python examples/bone_finger.py            # --viz to watch it run, --plot to save bone_finger.png

Per unit depth; SI units.
"""
import sys
import time
import numpy as np
import warp as wp
import Diff_2D_FEM as D

wp.init()
PX = 1.0e-3
W, H = 12, 72                                    # 12 mm x 72 mm
BONES = [(8, 24), (30, 46), (52, 68)]            # rows of the three phalanges
HINGES = [(24, 30), (46, 52)]                    # rows of the two flexures
DT, TENSION = 2.0e-3, 110.0                      # cable tension at full pull [N/m]

mask = np.ones((H, W), np.uint8)
for r0, r1 in HINGES:
    mask[r0:r1, 4:] = 0                          # leave a 4 mm strip on the back (-x side)

model = D.Model()
disc = model.add_disc(center=(0.021, 0.046), radius=0.009, mu=0.6)
finger = model.add_soft_body(mask, physical_size=(W * PX, H * PX), center=(0.0, 0.5 * H * PX),
                             E=1.0e6, nu=0.40, density=1100.0)
model.finalize()

bones = []
for r0, r1 in BONES:
    region = np.zeros((H, W), bool); region[r0:r1] = True
    bones.append(model.add_affine_region(finger, region))

X = model.rest_x(finger)
model.set_kinematic_constraint(finger, np.where(X[:, 1] < 1e-9)[0])        # clamp the base
eyelets = [(0.004, y * PX) for r0, r1 in [(0, 8)] + BONES for y in range(r0, r1 + 1)]
cable = model.add_tendon(finger, through=eyelets)
L0 = model.tendon_length(cable)

integ = D.ImplicitIntegrator()
view = D.Renderer(model) if "--viz" in sys.argv else None
print(f"finger: {model.n_nodes} nodes, {int(sum(len(a['nodes']) for a in model._affine))} of them "
      f"carried by 3 bones (18 unknowns)")
t0 = time.time()
for i in range(1, 401):
    s = min(i / 300, 1.0)
    model.set_tendon_tension(cable, TENSION * 0.5 * (1 - np.cos(np.pi * s)))
    integ.simulate(model, DT)
    if view: view.draw()
    if i % 100 == 0:
        ang = [np.rad2deg(model.affine_state(b)["angle"]) for b in bones]
        F = integ.collider_force(model, disc)[0]
        print(f"  cable {model.tendon_tension(cable):5.1f} N/m: bones at "
              f"{ang[0]:+6.1f}, {ang[1]:+6.1f}, {ang[2]:+6.1f} deg; cable in "
              f"{(L0 - model.tendon_length(cable)) * 1e3:5.2f} mm; disc carries "
              f"{np.linalg.norm(F):5.2f} N/m", flush=True)

strain = max(np.abs(model.affine_state(b)["A"].T @ model.affine_state(b)["A"] - np.eye(2)).max()
             for b in bones)
print(f"largest strain in a bone {strain:.1e};  {time.time() - t0:.0f} s of wall time")

if "--plot" in sys.argv:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from Diff_2D_FEM import viz
    fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
    viz.draw(model, ax=axes[0], x=model.me.rest_pos); axes[0].set_title("at rest")
    viz.draw(model, ax=axes[1]); axes[1].set_title(f"cable at {TENSION:.0f} N/m")
    for ax in axes:
        ax.set_xlim(-0.012, 0.045); ax.set_ylim(-0.004, 0.078)
    fig.tight_layout(); fig.savefig("bone_finger.png", dpi=130)
    print("wrote bone_finger.png")

if view:
    view.hold()                                # the window stays until it is closed
