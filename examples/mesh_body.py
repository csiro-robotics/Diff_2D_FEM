"""Bodies given as triangle meshes — add_soft_mesh, add_affine_mesh, mask_to_mesh.

A soft ring stands on the ground. It is not drawn on a pixel grid: it is meshed here in
polar coordinates, so its outline is the circle's own polygon rather than a staircase. A
rigid block — a pixel design, turned by 15 degrees with mask_to_mesh before it is added —
is dropped onto it and comes to rest there, held by friction. Then the ring's hole, which is a pressure chamber because it is a hole
in the mesh, is inflated: the ring swells and lifts the block.

Any mesher's output goes in the same way (gmsh, triangle, meshio): vertices (n, 2) in
metres and triangles (m, 3); vertex i is node i of the body.

    python examples/mesh_body.py            # --viz to watch it run, --plot to save mesh_body.png

Per unit depth; SI units.
"""
import sys
import time
import numpy as np
import warp as wp
import Diff_2D_FEM as D

wp.init()
RI, RO, N_T, N_R = 6.0e-3, 10.0e-3, 72, 6     # the ring: 6 / 10 mm radii, 72 x 6 cells
E, NU, RHO = 0.3e6, 0.40, 1070.0              # a soft silicone
PX = 0.75e-3                                  # the block: 8 x 8 pixels of 0.75 mm
P_MAX, DT, N_DROP, N_INFLATE, N_HOLD = 30.0e3, 1.0e-3, 150, 200, 50


def ring_mesh(ri, ro, n_t, n_r, center):
    """A ring in polar coordinates: (n_r + 1) circles of n_t nodes, two triangles a cell."""
    r = np.linspace(ri, ro, n_r + 1)
    th = 2.0 * np.pi * np.arange(n_t) / n_t
    V = np.array([(rk * np.cos(t), rk * np.sin(t)) for rk in r for t in th]) + center
    T = []
    for k in range(n_r):
        for j in range(n_t):
            a, b = k * n_t + j, k * n_t + (j + 1) % n_t
            T += [(a, b, b + n_t), (a, b + n_t, a + n_t)] if (j + k) % 2 else [(a, b, a + n_t), (b, b + n_t, a + n_t)]
    return V, np.array(T)


# --- scene ---------------------------------------------------------------------
model = D.Model()
model.add_ground(height=0.0, mu=0.8)
model.body_friction = 0.8                     # between the block and the ring
model.damping = 20.0
V, T = ring_mesh(RI, RO, N_T, N_R, center=(0.0, RO + 0.1e-3))
ring = model.add_soft_mesh(V, T, E=E, nu=NU, density=RHO)

Vb, Tb, _ = D.mask_to_mesh(np.ones((8, 8)), (8 * PX, 8 * PX), (0.0, 0.0))
th = np.deg2rad(15.0)
Vb = Vb @ np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]).T
Vb += (0.0, 2 * RO + 2.0e-3 - Vb[:, 1].min())                # its lowest corner 2 mm above the ring
block = model.add_affine_mesh(Vb, Tb, density=600.0)
model.finalize()
model.gravity = (0.0, -9.81)

hole = model.add_pressure_chambers(ring)[0]                   # the ring's hole is its chamber
A0 = model.chamber_area(hole)
print(f"ring: {len(V)} nodes, {len(T)} triangles, element size {model.element_size(ring) * 1e3:.2f} mm;  "
      f"block: {len(Vb)} nodes, turned 15 deg;  hole {A0 * 1e6:.1f} mm^2", flush=True)

# --- simulate: drop the block, then inflate the ring -----------------------------
integ = D.ImplicitIntegrator()
view = D.Renderer(model, bounds=((-0.016, -0.001), (0.016, 0.036))) if "--viz" in sys.argv else None
frames, t0 = [model.get_x()], time.time()
for k in range(1, N_DROP + N_INFLATE + N_HOLD + 1):
    p = P_MAX * min(max(k - N_DROP, 0) / N_INFLATE, 1.0)
    model.set_pressure(hole, p)
    integ.simulate(model, DT)
    if view:
        view.draw()
    if k % 50 == 0:
        frames.append(model.get_x())
        x = model.get_body_x(ring)
        print(f"  step {k:3d}  pressure {p / 1e3:4.1f} kPa  hole {model.chamber_area(hole) / A0:5.3f} x  "
              f"ring top {x[:, 1].max() * 1e3:5.2f} mm  block bottom {model.get_body_x(block)[:, 1].min() * 1e3:5.2f} mm",
              flush=True)
weight = np.asarray(model.me.mass).sum() * 9.81
print(f"ground carries {abs(integ.collider_force(model, friction=False)[0][1]) / weight:.4f} of the two "
      f"weights; {time.time() - t0:.0f} s")

if view:
    view.hold()
if "--plot" in sys.argv:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from Diff_2D_FEM import viz
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
    for ax, x, title in ((axes[0], frames[0], "as built"), (axes[1], frames[-1], f"{P_MAX / 1e3:.0f} kPa in the hole")):
        viz.draw(model, ax=ax, x=x)
        ax.set_title(title); ax.set_xlim(-0.016, 0.016); ax.set_ylim(-0.001, 0.036); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig("mesh_body.png", dpi=130)
    print("wrote mesh_body.png")
