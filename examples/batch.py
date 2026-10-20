"""Many designs in one solve: a sweep over the width of a flexure hinge.

Each design goes into its own ENVIRONMENT (env_id). Environments are advanced together by
one batched solve — every kernel launch covers all of them — but they never interact:
each has its own step length, its own convergence test, its own copy of the colliders.
(tests/test_scene.py checks that a design in a batch reproduces the design alone bit for
bit.) This is how a population is evaluated in design optimisation.

    python examples/batch.py            # add --viz to watch all ten at once

Per unit depth; SI units.
"""
import sys
import time
import numpy as np
import warp as wp
import Diff_2D_FEM as D

wp.init()
PX, W, H = 1.0e-3, 12, 48
WIDTHS = list(range(2, 12))                       # hinge widths to try [pixels]
DT, STEPS, TENSION = 2.0e-3, 150, 25.0


def design(width):
    m = np.ones((H, W), np.uint8)
    m[20:26, width:] = 0                          # a notch: `width` pixels of hinge remain
    return m


def build(widths):
    model = D.Model()
    model.damping = 50.0
    handles = [model.add_soft_body(design(w), physical_size=(W * PX, H * PX),
                                   center=(0.0, 0.5 * H * PX), E=1.0e6, nu=0.40,
                                   density=1100.0, env_id=i) for i, w in enumerate(widths)]
    model.finalize()
    cables = []
    for h in handles:
        X = model.rest_x(h)
        model.set_kinematic_constraint(h, np.where(X[:, 1] < 1e-9)[0])
        ys = [y * PX for y in range(0, 21)] + [y * PX for y in range(26, H + 1)]
        cables.append(model.add_tendon(h, through=[(0.004, y) for y in ys]))
    return model, handles, cables


def run(widths):
    model, handles, cables = build(widths)
    integ = D.ImplicitIntegrator()
    view = D.Renderer(model, envs="all") if "--viz" in sys.argv and len(widths) > 1 else None
    t0 = time.time()
    for i in range(1, STEPS + 1):
        for c in cables:
            model.set_tendon_tension(c, TENSION * min(i / (0.6 * STEPS), 1.0))
        integ.simulate(model, DT)
        if view: view.draw()
    wall = time.time() - t0
    if view: view.close()
    tips = []
    for h in handles:
        X, x = model.rest_x(h), model.get_body_x(h)
        top = X[:, 1] > X[:, 1].max() - 1e-9                   # the fingertip's end face
        tips.append(np.degrees(np.arctan(np.polyfit(x[top, 0], x[top, 1], 1)[0])))
    return np.array(tips), wall


run(WIDTHS[:1])                                    # compile the kernels before timing
tilt, t_batch = run(WIDTHS)
t_one = sum(run([w])[1] for w in WIDTHS[:3]) / 3
print(f"{len(WIDTHS)} designs in one batch: {t_batch:.1f} s;  one design alone: {t_one:.1f} s  "
      f"->  {len(WIDTHS) * t_one / t_batch:.1f}x the throughput")
print("hinge width [px]   fingertip tilt [deg]")
for w, a in zip(WIDTHS, tilt):
    print(f"      {w:>2}              {a:+7.2f}   " + "#" * int(round(abs(a))))
