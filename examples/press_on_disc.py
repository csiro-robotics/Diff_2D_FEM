"""The smallest example — the Model + Integrator API.

A solid soft block is pressed straight down onto a fixed rigid disc: the top row of
nodes is kinematically driven downward, the disc below catches the block, and the
implicit integrator resolves the IPC contact. Prints the growing contact force.

    python examples/press_on_disc.py            # add --viz to watch it run
"""
import sys
import numpy as np
import warp as wp
import Diff_2D_FEM as D

wp.init()

# --- scene --------------------------------------------------------------------
model = D.Model()
model.add_disc(center=(0.05, 0.025), radius=0.006, mu=0.6)   # disc top at 31 mm

mask = np.ones((40, 16), np.uint8)                        # solid 40x16 block
h = model.add_soft_body(mask, physical_size=(0.016, 0.040), center=(0.05, 0.060),
                        E=1.0e6, nu=0.40, density=1100.0)
model.finalize()

# top row of nodes = the kinematic "actuator"
x0 = model.get_body_x(h)
top = np.where(x0[:, 1] > x0[:, 1].max() - 1e-4)[0]
model.set_kinematic_constraint(h, top)
print(f"nodes={x0.shape[0]}  driven(top row)={len(top)}  block y=[{x0[:,1].min()*1e3:.1f},"
      f"{x0[:,1].max()*1e3:.1f}]mm  disc top={ (0.025+0.006)*1e3:.1f}mm", flush=True)

# --- simulate: push the top row down 20 mm over 80 implicit steps -------------
integ = D.ImplicitIntegrator(max_iter=120, self_collision=True, friction=True)
view = D.Renderer(model) if "--viz" in sys.argv else None      # a live window
dt = 1.0e-3
STROKE, N = 0.020, 80
for t in range(1, N + 1):
    x = model.get_body_x(h)
    x[top, 1] = x0[top, 1] - STROKE * t / N               # absolute downward target
    model.set_kinematic_target(h, x)
    integ.simulate(model, dt)
    if view:
        view.begin_frame(model.time)
        view.render()
        view.end_frame()
    if t % 20 == 0 or t == N:
        F = integ.collider_force(model)[0]                    # (Fx, Fy) reaction on disc
        xc = model.get_body_x(h)
        print(f"  step {t:3d}/{N}  drive={STROKE*t/N*1e3:4.1f}mm  "
              f"block bottom={xc[:,1].min()*1e3:5.1f}mm  disc |F|={np.linalg.norm(F):.3g} N/m",
              flush=True)

print("[done] ran through the Model / ImplicitIntegrator API.")

if view:
    view.hold()                                # the window stays until it is closed
