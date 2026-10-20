"""A soft crawler: a strip of active material walks over the ground on a travelling wave.

Nothing holds the body. It lies on the ground under gravity; six segments each have an
upper and a lower layer of active material (fibres along the body). Lengthening the upper
layer and shortening the lower one arches a segment off the ground. Sweeping that arch
from the tail to the head carries the body forward the way a caterpillar moves: the part
in the air is moved while the rest, carrying the weight, holds its ground by friction.

    python examples/crawler.py            # --viz to watch it run, --plot to save crawler.png

Per unit depth; SI units.
"""
import sys
import time
import numpy as np
import warp as wp
import Diff_2D_FEM as D

wp.init()
NX, NY = 72, 6                         # pixels: 54 mm x 4.5 mm, 0.75 mm elements
L, H = 0.054, 0.0045
SEG = 6                                # active segments along the body
AMP = 0.10                             # upper layer +10 %, lower layer -10 %
PERIOD, DT, WAVES = 0.6, 2.0e-3, 3

model = D.Model()
model.add_ground(height=0.0, mu=0.8)
model.damping = 5.0                    # a little air drag [1/s]
body = model.add_soft_body(np.ones((NY, NX), np.uint8), physical_size=(L, H),
                           center=(0.0, H / 2 + L / NX / 8),            # resting: the barrier reaches 1/8 pixel
                           E=0.5e6, nu=0.40, density=1100.0)
model.finalize()
model.gravity = (0.0, -9.81)

upper, lower = [], []
for k in range(SEG):
    cols = slice(k * NX // SEG, (k + 1) * NX // SEG)
    r = np.zeros((NY, NX), bool); r[NY // 2:, cols] = True
    upper.append(model.add_active_region(body, r, fiber=0.0))            # fibres along x
    r = np.zeros((NY, NX), bool); r[:NY // 2, cols] = True
    lower.append(model.add_active_region(body, r, fiber=0.0))


def arch(k, phase):
    """Amplitude of segment k when the wave's crest is at `phase` (in segments)."""
    u = (k + 0.5 - phase) / 1.5
    return AMP * np.cos(0.5 * np.pi * u) ** 2 if abs(u) < 1.0 else 0.0


def centroid():
    m = np.asarray(model.me.mass)
    return (m[:, None] * model.get_x()).sum(0) / m.sum()


integ = D.ImplicitIntegrator()
view = D.Renderer(model) if "--viz" in sys.argv else None
for _ in range(50):                    # settle onto the ground
    integ.simulate(model, DT)
x_start = centroid()[0]
print(f"crawler: {model.n_nodes} nodes, {integ.stats(model)['max_iter']} iterations/step at most, "
      f"weight {np.asarray(model.me.mass).sum() * 9.81:.2f} N/m")

steps = int(round(PERIOD / DT))
t0 = time.time(); snapshot = None; peak = 0.0
for wave in range(WAVES):
    x0 = centroid()[0]
    for i in range(steps):
        phase = -1.5 + (SEG + 3.0) * (i + 1) / steps                    # crest: tail -> head
        for k in range(SEG):
            a = arch(k, phase)
            model.set_activation(upper[k], 1.0 + a)
            model.set_activation(lower[k], 1.0 - a)
        integ.simulate(model, DT)
        if view: view.draw()
        lift = model.get_x()[:, 1].max() - H
        if lift > peak:
            peak, snapshot = lift, model.get_x()
    print(f"  wave {wave + 1}: body advanced {(centroid()[0] - x0) * 1e3:+.2f} mm", flush=True)

total = centroid()[0] - x_start
print(f"{WAVES} waves in {WAVES * PERIOD:.1f} s: {total * 1e3:+.2f} mm "
      f"({total / L:+.1%} of the body length); the arch rose {peak * 1e3:.2f} mm;  "
      f"{time.time() - t0:.0f} s of wall time")

if "--plot" in sys.argv:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from Diff_2D_FEM import viz
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True)
    viz.draw(model, ax=axes[0], x=snapshot); axes[0].set_title("mid-stride")
    viz.draw(model, ax=axes[1]); axes[1].set_title(f"after {WAVES} waves")
    for ax in axes:
        ax.axvline(x_start, color="k", lw=0.6, ls=":"); ax.set_ylim(-0.003, 0.016)
    fig.tight_layout(); fig.savefig("crawler.png", dpi=130)
    print("wrote crawler.png")

if view:
    view.hold()                                # the window stays until it is closed
