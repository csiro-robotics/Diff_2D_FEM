"""Obstacles of arbitrary shape — the same squeeze against a disc, an ellipse, a box and a
triangle.

The obstacle is an analytic signed-distance function, not a mesh: contact is the IPC barrier
on phi(x), and friction and CCD dispatch on the same shape. `model.object` is the host-side
description of that shape, so the contact length reported here is measured against the
surface the solver actually used.

    python examples/object_shapes.py            # add --viz to watch it run
"""
import sys
import numpy as np
import warp as wp
import Diff_2D_FEM as D
from Diff_2D_FEM.core import objects

wp.init()
SHAPES = {
    "disc":     dict(rx=0.006),
    "ellipse":  dict(rx=0.008, ry=0.005),
    "box":      dict(rx=0.007, ry=0.005, corner_radius=0.002),    # keep corner_radius > 0
    "triangle": dict(rx=0.007, corner_radius=0.002),              # rx = circumradius, apex up
}
STROKE, N, DT = 0.012, 60, 1.0e-3

print(f"{'shape':9} {'object':32} {'peak |F| N/m':>13} {'contact mm':>11} {'of perimeter':>13}")
for shape, kw in SHAPES.items():
    model = D.Model()
    model.add_object(shape, center=(0.05, 0.030), mu=0.6, **kw)
    h = model.add_soft_body(np.ones((40, 16), np.uint8), physical_size=(0.016, 0.040),
                            center=(0.05, 0.060), E=1.0e6, nu=0.40, density=1100.0)
    model.finalize()
    x0 = model.get_body_x(h)
    top = np.where(x0[:, 1] > x0[:, 1].max() - 1e-4)[0]
    model.set_kinematic_constraint(h, top)

    integ = D.ImplicitIntegrator(max_iter=120)
    view = D.Renderer(model, title=shape) if "--viz" in sys.argv else None
    peak = 0.0
    for t in range(1, N + 1):
        x = model.get_body_x(h)
        x[top, 1] = x0[top, 1] - STROKE * t / N
        model.set_kinematic_target(h, x)
        integ.simulate(model, DT)
        if view: view.draw()
        peak = max(peak, float(np.linalg.norm(integ.collider_force(model)[0])))
    if view: view.close()

    obj = model.object
    arc, _ = objects.contact_arc(model.get_x(), model.me.boundary_nodes, model.me.boundary_edges,
                                 obj, model.dhat)
    print(f"{shape:9} {obj.label():32} {peak:13.1f} {arc*1e3:11.2f} {arc/objects.perimeter(obj):13.1%}")
