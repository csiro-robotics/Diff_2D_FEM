"""Inverse design of a PneuNet: which strain-limiting layer makes the finger curl into a
given shape?

A PneuNet bends because its strain-limiting layer stretches less than the bellows above
it (examples/pneunet.py). Where that layer is stiff the finger curls tightly; where it is
soft, hardly at all — so the stiffness under each bellows is a design variable that sets
the curvature along the finger. Here the target is a HOOK: a finger that stays straight
near its root and curls at its tip, inflated to one pressure. CMA-ES searches the
stiffness of the layer under each of the eight bellows (log-scale, 1 to 10 x the
silicone), and every generation of candidates is ONE batched simulation: each candidate
is its own environment (env_id=i), the same mask with its own stiffness map.

The shape compared is the one 0.1 s into a pressure ramp — the same ramp for the target
and every candidate — not the equilibrium curl, which a soft bending finger reaches only
slowly on this solver (the iteration budget; see the README).

The target is the shape a hidden design makes, so a design that matches it exactly
exists; replace TARGET by any curve of your own (the underside's node positions). The
inverse problem need not have one answer: a different stiffness profile can bend into
nearly the same shape, which is why the found profile is plotted next to the hidden one.

    python examples/inverse_pneunet.py           # about 12 min on an RTX A5000
    python examples/inverse_pneunet.py --quick   # a few generations, about 3 min
    python examples/inverse_pneunet.py --plot    # also save inverse_pneunet.png

Per unit depth; SI units.
"""
import sys
import time
import numpy as np
import warp as wp
import Diff_2D_FEM as D
from _cmaes import CMAES                    # examples/_cmaes.py

wp.init()
QUICK = "--quick" in sys.argv
PX = 0.5e-3                                   # 0.5 mm pixels (pneunet.py uses 0.25 mm)
N = 8                                         # bellows, each with its own layer stiffness
BELLOW, SLIT, WALL, ROOF, BASE, CAVITY = 12, 3, 2, 2, 4, 12
E, NU, RHO = 0.3e6, 0.40, 1070.0              # a soft silicone
PRESSURE, RAMP, HOLD, DT = 12.0e3, 40, 10, 2.0e-3
HIDDEN = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])   # log10 of layer stiffness / E: the hook
LO, HI = 0.0, 1.0                             # the layer: 1 to 10 x the silicone
LAM, GENS = (8, 4) if QUICK else (16, 20)

W = N * BELLOW + (N - 1) * SLIT
H = BASE + CAVITY + ROOF
MASK = np.zeros((H, W), np.uint8)             # row 0 at the bottom
MASK[:BASE] = 1
for k in range(N):
    c = k * (BELLOW + SLIT)
    MASK[BASE:, c:c + BELLOW] = 1
    MASK[BASE:BASE + CAVITY, c + WALL:c + BELLOW - WALL] = 0


def stiffness(x):
    """Per-pixel stiffness map: the lower half of the base under bellows k (and the slit
    after it) is 10^x[k] times the silicone."""
    e = np.full((H, W), E)
    for k in range(N):
        c0, c1 = k * (BELLOW + SLIT), (k + 1) * (BELLOW + SLIT)
        e[:BASE // 2, c0:c1] = E * 10.0 ** np.clip(x[k], LO, HI)
    return e


def inflate(designs):
    """Inflate every design to PRESSURE, all in one batched solve; the underside's node
    positions at the end, (n_designs, n_sole, 2), root to tip."""
    model = D.Model()
    model.damping = 60.0
    fingers = [model.add_soft_body(MASK, physical_size=(W * PX, H * PX), center=(W * PX / 2, H * PX / 2),
                                   E=stiffness(x), nu=NU, density=RHO, env_id=i) for i, x in enumerate(designs)]
    model.finalize()
    X = model.rest_x(fingers[0])
    sole = np.where(X[:, 1] < 1e-9)[0]
    sole = sole[np.argsort(X[sole, 0])]
    chambers = []
    for f in fingers:
        model.set_kinematic_constraint(f, np.where(X[:, 0] < 1e-9)[0])    # clamped at its root
        chambers += model.add_pressure_chambers(f)
    integ = D.ImplicitIntegrator()
    for i in range(1, RAMP + HOLD + 1):
        for c in chambers:
            model.set_pressure(c, PRESSURE * min(i / RAMP, 1.0))
        integ.simulate(model, DT)
    inflate.iters = int(integ.stats(model)['max_iter'])
    return np.stack([model.get_body_x(f)[sole] for f in fingers]), X[sole]


def shape_error(soles, target):
    """RMS distance of the underside's nodes from the target's [m]."""
    return np.sqrt(((soles - target[None]) ** 2).sum(2).mean(1))


t0 = time.time()
targets, sole_rest = inflate([HIDDEN])
TARGET = targets[0]
print(f"inverse PneuNet: {N} bellows, {W * PX * 1e3:.0f} x {H * PX * 1e3:.0f} mm, {PRESSURE * 1e-3:.0f} kPa; "
      f"target = the hidden hook ({time.time() - t0:.0f} s)")
es = CMAES(np.full(N, 0.5 * (LO + HI)), 0.4, LAM, np.random.default_rng(0))
history, best = [], (np.inf, None, None)
for gen in range(GENS):
    tg = time.time()
    xs = np.clip(es.ask(), LO, HI)
    soles, _ = inflate(xs)
    err = shape_error(soles, TARGET)
    es.tell(err)
    k = int(np.argmin(err))
    if err[k] < best[0]:
        best = (err[k], xs[k].copy(), soles[k])
    history.append((err.min(), np.median(err)))
    print(f"  generation {gen + 1:2d}: best {err[k] * 1e3:6.3f} mm, median {np.median(err) * 1e3:6.3f} mm, "
          f"step {es.sigma:.3f}   ({time.time() - tg:.0f} s)", flush=True)
err, x_best, sole_best = best
print(f"best design: RMS shape error {err * 1e3:.3f} mm over a {W * PX * 1e3:.0f} mm finger; layer stiffness "
      f"{np.round(10 ** x_best, 1).tolist()} x E (hidden: {np.round(10 ** HIDDEN, 1).tolist()}); "
      f"{time.time() - t0:.0f} s of wall time")

if "--plot" in sys.argv:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    start, _ = inflate([np.full(N, 0.5 * (LO + HI))])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 1.4, 1]})
    h = np.array(history) * 1e3
    axes[0].semilogy(range(1, len(h) + 1), h[:, 0], "o-", ms=3, label="best of generation")
    axes[0].semilogy(range(1, len(h) + 1), h[:, 1], "s--", ms=3, label="median")
    axes[0].set_xlabel("generation"); axes[0].set_ylabel("RMS shape error [mm]"); axes[0].legend(); axes[0].grid(alpha=0.3)
    ax = axes[1]
    ax.plot(sole_rest[:, 0] * 1e3, sole_rest[:, 1] * 1e3, color="0.7", lw=1.5, label="at rest")
    ax.plot(start[0][:, 0] * 1e3, start[0][:, 1] * 1e3, color="#95a5a6", lw=1.5, ls=":", label="uniform layer (start)")
    ax.plot(TARGET[:, 0] * 1e3, TARGET[:, 1] * 1e3, color="k", lw=4, alpha=0.35, label="target")
    ax.plot(sole_best[:, 0] * 1e3, sole_best[:, 1] * 1e3, color="#c0392b", lw=1.5, label="found")
    ax.set_aspect("equal"); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.legend(fontsize=8)
    ax.set_title(f"the finger's underside at {PRESSURE * 1e-3:.0f} kPa")
    ax = axes[2]
    k = np.arange(1, N + 1)
    ax.bar(k - 0.2, 10 ** HIDDEN, 0.4, color="0.6", label="hidden")
    ax.bar(k + 0.2, 10 ** x_best, 0.4, color="#c0392b", label="found")
    ax.set_yscale("log"); ax.set_xlabel("bellows (root to tip)"); ax.set_ylabel("layer stiffness / E")
    ax.legend(); ax.set_title("strain-limiting layer")
    fig.tight_layout(); fig.savefig("inverse_pneunet.png", dpi=130)
    print("wrote inverse_pneunet.png")
