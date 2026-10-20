"""A PneuNet bending actuator — actuation by fluid pressure.

The classic pneumatic soft finger (Mosadegh et al., "Pneumatic Networks for Soft Robotics
that Actuate Rapidly", Adv. Funct. Mater. 2014), seen from the side: a row of hollow
bellows standing on a common base, the base backed by a strain-limiting layer. Under
pressure every bellows swells; its side walls balloon into the slits until they meet their
neighbours' and push them away. The top of the finger gets longer, the strain-limiting
layer does not, and the finger curls towards its base.

Each cavity of the mask is a sealed chamber (add_pressure_chambers); they are all given the
same pressure, as if joined by a channel. The walls that meet are parts of ONE body touching
each other — IPC contact of the finger with itself.

    python examples/pneunet.py            # --viz to watch it run, --plot to save pneunet.png

Per unit depth; SI units.
"""
import sys
import time
import numpy as np
import warp as wp
import Diff_2D_FEM as D

wp.init()
PX = 0.25e-3                                  # 0.25 mm pixels
N = 8                                         # bellows
BELLOW, SLIT = 24, 6                          # a bellows is 6 mm wide; slits of 1.5 mm between them
WALL, ROOF = 4, 4                             # 1 mm side walls and roof
BASE, CAVITY = 8, 24                          # a 2 mm base; cavities 6 mm tall
E, NU, RHO = 0.3e6, 0.40, 1070.0              # a soft silicone
E_LIMIT = 4.0                                 # the strain-limiting layer: the lower half of the base, 4 x stiffer
DT = 2.0e-3

W = N * BELLOW + (N - 1) * SLIT
H = BASE + CAVITY + ROOF
mask = np.zeros((H, W), np.uint8)             # row 0 at the bottom
mask[:BASE] = 1
for k in range(N):
    c = k * (BELLOW + SLIT)
    mask[BASE:, c:c + BELLOW] = 1
    mask[BASE:BASE + CAVITY, c + WALL:c + BELLOW - WALL] = 0
stiffness = np.full((H, W), E)
stiffness[:BASE // 2] = E_LIMIT * E

model = D.Model()
model.damping = 60.0                          # enough to follow the pressure ramp without ringing
finger = model.add_soft_body(mask, physical_size=(W * PX, H * PX), center=(W * PX / 2, H * PX / 2),
                             E=stiffness, nu=NU, density=RHO)
model.finalize()
X = model.rest_x(finger)
model.set_kinematic_constraint(finger, np.where(X[:, 0] < 1e-9)[0])    # clamped at its left end
chambers = model.add_pressure_chambers(finger)
volume0 = sum(model.chamber_area(c) for c in chambers)

tip = np.where(X[:, 0] > W * PX - 1e-9)[0]
sole = np.where(X[:, 1] < 1e-9)[0]                                     # the underside, root to tip
above = X[:, 1] > BASE * PX + 1e-9
faces = []                                                             # the two walls facing each slit
for k in range(N - 1):
    c = (k * (BELLOW + SLIT) + BELLOW) * PX
    faces.append((np.where(above & (np.abs(X[:, 0] - c) < 1e-9))[0],
                  np.where(above & (np.abs(X[:, 0] - c - SLIT * PX) < 1e-9))[0]))


def pose():
    """Where the tip is, how far the underside has turned at its end [deg], and how many of
    the slits have closed (neighbouring walls within two barrier widths of each other)."""
    x = model.get_body_x(finger)
    s = sole[np.argsort(X[sole, 0])]
    d = x[s[-1]] - x[s[-6]]
    closed = sum(np.linalg.norm(x[a][:, None] - x[b][None], axis=2).min() < 2 * model.dhat
                 for a, b in faces)
    return x[tip].mean(0), np.degrees(np.arctan2(d[1], d[0])), closed


integ = D.ImplicitIntegrator()
view = D.Renderer(model, bounds=((-0.010, -0.050), (0.066, 0.016))) if "--viz" in sys.argv else None
print(f"PneuNet: {N} bellows, {model.n_nodes} nodes, {len(chambers)} chambers, "
      f"{W * PX * 1e3:.0f} mm x {H * PX * 1e3:.0f} mm")
t0 = time.time(); shots = []; p_now = 0.0
for p in (6.0e3, 12.0e3, 18.0e3):
    ramp = 60
    for i in range(1, ramp + 31):                                      # ramp over 0.12 s, hold 0.06 s
        q = p_now + (p - p_now) * min(i / ramp, 1.0)
        for c in chambers:
            model.set_pressure(c, q)
        integ.simulate(model, DT)
        if view: view.draw()
    p_now = p
    xy, angle, closed = pose()
    shots.append((p, model.get_x()))
    print(f"  {p * 1e-3:4.0f} kPa: tip at ({xy[0] * 1e3:6.2f}, {xy[1] * 1e3:6.2f}) mm, turned {angle:+7.1f} deg, "
          f"chambers +{sum(model.chamber_area(c) for c in chambers) / volume0 - 1:.0%} in volume, "
          f"{closed} of {N - 1} slits closed", flush=True)
print(f"{time.time() - t0:.0f} s of wall time, {integ.stats(model)['max_iter']} iterations/step at most")

if "--plot" in sys.argv:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from Diff_2D_FEM import viz
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    viz.draw(model, ax=axes[0], x=model.me.rest_pos, values=model.me.mu, cmap="cividis")
    axes[0].set_title("at rest (bright: the strain-limiting layer)")
    for (p, xs), c in zip(shots, ("#9CC3D5", "#4C8DAE", "#1F4E66")):
        viz.draw(model, ax=axes[1], x=xs, color=c)
    axes[1].set_title("at " + ", ".join(f"{p * 1e-3:.0f}" for p, _ in shots) + " kPa")
    for ax in axes:
        ax.set_xlim(-0.012, 0.066); ax.set_ylim(-0.052, 0.016); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig("pneunet.png", dpi=130)
    print("wrote pneunet.png")

if view:
    view.hold()                                # the window stays until it is closed
