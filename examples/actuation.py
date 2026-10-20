"""Actuating a soft body four ways: fluid pressure, a tendon, active material, gravity.

Each case is the same cantilevered strip, fixed at its left end, driven to its settled
shape: heavy damping, long steps, and enough iterations per step to converge them.

Units are per unit depth: pressure in Pa, tension in N per metre of depth.

    python examples/actuation.py
"""
import numpy as np
import warp as wp
import Diff_2D_FEM as D

wp.init()
NY, NX, H, L = 8, 40, 0.004, 0.020                    # 4 mm x 20 mm strip, 0.5 mm elements
E, NU, RHO = 1.0e6, 0.40, 1100.0


def strip(mask=None, stiffness=E):
    model = D.Model()
    model.damping = 2000.0                            # heavily damped: we want the settled shape
    mask = np.ones((NY, NX), np.uint8) if mask is None else mask
    h = model.add_soft_body(mask, physical_size=(L, H), center=(0.0, 0.0),
                            E=stiffness, nu=NU, density=RHO)
    model.finalize()
    x0 = model.get_body_x(h)
    model.set_kinematic_constraint(h, np.where(x0[:, 0] < x0[:, 0].min() + 1e-9)[0])   # clamp the left end
    tip = np.where(x0[:, 0] > x0[:, 0].max() - 1e-9)[0]
    return model, h, x0, tip


def settle(model, steps=12, dt=5.0e-2):
    # 50 ms steps are ~4400 elements of wave travel each: far more than the default budget
    # is meant for, so the iteration count is set by hand
    integ = D.ImplicitIntegrator(max_iter=2000, eps=1e-14, self_collision=False, friction=False)
    for _ in range(steps):
        integ.simulate(model, dt)


def tip_drop(model, h, x0, tip):
    return (model.get_body_x(h)[tip, 1] - x0[tip, 1]).mean() * 1e3    # mm, + is up


# ---- 1. pneumatic: sealed chambers along the top, a stiffer layer along the bottom ----------
mask = np.ones((NY, NX), np.uint8)
for c in range(3, NX - 3, 5):
    mask[4:7, c:c + 3] = 0                            # 1.5 mm x 1.5 mm cavities in the upper half
stiff = np.full((NY, NX), E); stiff[:2] = 8 * E       # rows 0-1 (the BOTTOM): strain-limiting layer
model, h, x0, tip = strip(mask, stiff)
chambers = model.add_pressure_chambers(h)
a0 = sum(model.chamber_area(c) for c in chambers)
for c in chambers:
    model.set_pressure(c, 40.0e3)                     # 40 kPa
settle(model)
print(f"pneumatic  {len(chambers)} chambers at 40 kPa: tip moves {tip_drop(model, h, x0, tip):+.3f} mm, "
      f"chamber volume +{(sum(model.chamber_area(c) for c in chambers) / a0 - 1):.1%}")

# ---- 2. tendon: a cable a quarter of the thickness above the mid-line ----------------------
model, h, x0, tip = strip()
cols = np.unique(np.round(x0[:, 0], 9))
cable = model.add_tendon(h, through=[(x, H / 4) for x in cols])       # an eyelet at every column
rest_len = model.tendon_length(cable)
model.set_tendon_tension(cable, 60.0)                 # 60 N/m  (= 1.2 N on a 20 mm wide finger)
settle(model)
print(f"tendon     60 N/m: tip moves {tip_drop(model, h, x0, tip):+.3f} mm, "
      f"cable pays in {(rest_len - model.tendon_length(cable)) * 1e6:.1f} um")

# the same cable driven by a motor that reels in 100 um
model, h, x0, tip = strip()
cable = model.add_tendon(h, through=[(x, H / 4) for x in cols], stiffness=2.0e5)
model.set_tendon_length(cable, model.tendon_length(cable) - 100e-6)
settle(model)
print(f"tendon     reeled in 100 um: tip moves {tip_drop(model, h, x0, tip):+.3f} mm, "
      f"cable tension {model.tendon_tension(cable):.1f} N/m")

# ---- 3. active material: the lower half contracts by 2 % ------------------------------------
model, h, x0, tip = strip()
lower = np.zeros((NY, NX), bool); lower[: NY // 2] = True
region = model.add_active_region(h, lower)
model.set_activation(region, 0.98)
settle(model)
print(f"active     lower half contracts 2 %: tip moves {tip_drop(model, h, x0, tip):+.3f} mm")

# ---- 4. gravity ------------------------------------------------------------------------------
model, h, x0, tip = strip()
model.gravity = (0.0, -9.81)
settle(model)
print(f"gravity    own weight: tip moves {tip_drop(model, h, x0, tip) * 1e3:+.2f} um")
