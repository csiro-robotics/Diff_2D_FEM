"""Topology optimisation of a compliant gripper finger with the adjoint: where should the
material go so that the fingertip presses hardest?

The boundary conditions of the 128 x 64 finger datasets (topology-optimised, fin-ray and
graph fingers): the finger hangs from its top edge — the top-left corner held, the
top-right corner pushed DOWN by the actuator — and its tip, the bottom-left corner, is
held as well. The objective is the reaction that held tip feels along (-1, -1)/sqrt 2,
inwards and down: the force the finger would press onto an object there.

The design is a density per pixel (SIMP, exponent 3, after a small cone filter), 30 % of
the plate, solid patches where the boundary conditions act. One design is optimised from
a uniform start by a projected gradient on the adjoint. The objective depends on the
density directly as well as through the positions, so its gradient is the adjoint's plus
an explicit part — AdjointPNCG2D.force_objective, checked against finite differences.
Friction and contact play no part (the adjoint's scope, see the README), and the push is
slow next to the plate's wave speed, so the response is quasi-static.

Forces are per unit depth (N/m), as everywhere in the package.

    python examples/topopt_gripper.py              # --iters 20 for a short run
    python examples/topopt_gripper.py --plot       # also save topopt_gripper.png

Per unit depth; SI units.
"""
import sys
import time
import numpy as np
import warp as wp
from scipy import ndimage
import Diff_2D_FEM as D

wp.init()
NY, NX, PX = 128, 64, 0.5e-3                # the finger: 64 x 32 mm, 0.5 mm pixels (row 0 = tip end)
E, NU, RHO = 1.0e6, 0.40, 1100.0
T, DT, PUSH = 6, 2.0e-3, 0.5e-3             # push the actuator 0.5 mm down over 12 ms
DIRECTION = np.array([-1.0, -1.0]) / np.sqrt(2.0)   # the output's direction: inwards and down
VOL, P_SIMP, R_FILTER = 0.3, 3.0, 2.0       # material fraction, SIMP exponent, filter radius [pixels]
ITERS = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 60
MOVE = 0.1                                  # largest density change per iteration

# where the boundary conditions act, as in the datasets (BCx/BCy, inputy, outputx/outputy)
SUPPORT_COLS, INPUT_COLS = (0, 5), (59, 64)  # pixel columns along the top row (row 127)
FIXED = np.zeros((NY, NX), bool)             # solid, not designed
FIXED[NY - 3:, SUPPORT_COLS[0]:SUPPORT_COLS[1] + 1] = True
FIXED[NY - 3:, INPUT_COLS[0] - 1:INPUT_COLS[1]] = True
FIXED[:3, :3] = True

_r = int(np.ceil(R_FILTER))
_k = np.maximum(0.0, R_FILTER - np.hypot(*np.mgrid[-_r:_r + 1, -_r:_r + 1]))
_norm = ndimage.convolve(np.ones((NY, NX)), _k, mode="constant")
filt = lambda s: ndimage.convolve(s, _k, mode="constant") / _norm
filt_T = lambda g: ndimage.convolve(g / _norm, _k, mode="constant")
density = lambda s: np.where(FIXED, 1.0, filt(s) ** P_SIMP)

# --- the finger ---------------------------------------------------------------------------
model = D.Model()
model.damping = 50.0                         # quasi-static: damp the plate's free vibration
finger = model.add_soft_body(np.ones((NY, NX), np.uint8), physical_size=(NX * PX, NY * PX),
                             center=(0.0, 0.0), E=E, nu=NU, density=RHO)
model.finalize()
X = model.rest_x(finger)
x0, y0, y1 = X[:, 0].min(), X[:, 1].min(), X[:, 1].max()
col = np.round((X[:, 0] - x0) / PX).astype(int)                     # node column 0..NX
top = np.abs(X[:, 1] - y1) < 1e-9
support = np.where(top & (col >= SUPPORT_COLS[0]) & (col <= SUPPORT_COLS[1]))[0]
actuator = np.where(top & (col >= INPUT_COLS[0]) & (col <= INPUT_COLS[1]))[0]
tip = np.where((col <= 1) & (X[:, 1] <= y0 + PX + 1e-9))[0]          # the corner pixel's four nodes
held = np.concatenate([support, actuator, tip])
model.set_kinematic_constraint(finger, held)
integ = D.ImplicitIntegrator(max_iter=3000, eps=1e-14, tol_x=1e-8, conv_check=8,
                             self_collision=False, friction=False)
adj = integ.adjoint(model, DT)
tip_g = model.body_nodes(finger)[tip]


def push(s):
    """Push the actuator down from rest; the states x_0..x_T and the tip's reaction."""
    adj.set_density(model.to_elements(finger, density(s)))
    model.reset()
    xs = [model.get_x().astype(np.float64)]
    for t in range(1, T + 1):
        x = model.get_body_x(finger)
        x[held] = X[held]; x[actuator, 1] -= PUSH * t / T
        model.set_kinematic_target(finger, x)
        integ.simulate(model, DT)
        xs.append(model.get_x().astype(np.float64))
    return xs


def with_volume(s0, lb, ub):
    """Shift the design so that it uses VOL of the plate outside the fixed patches, within
    the move limits [lb, ub]."""
    lo, hi = -2.0, 2.0
    for _ in range(60):
        c = 0.5 * (lo + hi)
        lo, hi = (c, hi) if np.clip(s0 - c, lb, ub)[~FIXED].mean() > VOL else (lo, c)
    return np.clip(s0 - 0.5 * (lo + hi), lb, ub)


s = np.full((NY, NX), VOL)
history, t0 = [], time.time()
print(f"gripper finger: {NX} x {NY} pixels, {VOL:.0%} material, actuator pushed {PUSH * 1e3:.1f} mm; "
      f"maximising the tip's reaction along (-1, -1)")
for it in range(ITERS):
    xs = push(s)
    J, dJdx, explicit = adj.force_objective(xs[-1], tip_g, DIRECTION)
    g_el = adj.transient_adjoint(xs, seed={T: dJdx}, rtol=1e-14, maxiter=50000) + explicit
    g = filt_T(np.where(FIXED, 0.0, model.to_pixels(finger, g_el) * P_SIMP * filt(s) ** (P_SIMP - 1)))
    history.append((time.time() - t0, J))
    if it % 5 == 0 or it == ITERS - 1:
        print(f"  iteration {it + 1:3d}: tip reaction {J:8.3f} N/m   ({time.time() - t0:.0f} s)", flush=True)
    scale = np.percentile(np.abs(g[~FIXED]), 98) or 1.0           # robust to the corner's peaks
    lb, ub = np.maximum(0.0, s - MOVE), np.minimum(1.0, s + MOVE)
    s = with_volume(s + MOVE * np.clip(g / scale, -1.0, 1.0), lb, ub)   # ascent: more reaction
J_end = adj.force_objective(push(s)[-1], tip_g, DIRECTION)[0]
print(f"final design: tip reaction {J_end:.3f} N/m along (-1, -1), from {history[0][1]:.3f} N/m for the uniform "
      f"plate; {time.time() - t0:.0f} s of wall time")
np.save("topopt_gripper_density.npy", density(s))

if "--plot" in sys.argv:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    fig = plt.figure(figsize=(12, 6.2))
    ax = fig.add_subplot(1, 3, 1)
    h = np.array(history)
    ax.plot(range(1, len(h) + 1), h[:, 1], "o-", ms=3)
    ax.set_xlabel("iteration"); ax.set_ylabel("tip reaction along (-1, -1) [N/m]"); ax.grid(alpha=0.3)
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(density(s), origin="lower", cmap="Greys", vmin=0, vmax=1, extent=(0, NX, 0, NY))
    ax.plot(np.mean(SUPPORT_COLS) + 0.5, NY + 1.5, "s", color="#2E86C1", ms=9, clip_on=False, label="held")
    ax.annotate("", xy=(np.mean(INPUT_COLS) + 0.5, NY - 1), xytext=(np.mean(INPUT_COLS) + 0.5, NY + 8),
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=2), annotation_clip=False)
    ax.annotate("", xy=(-5, -5), xytext=(1.5, 1.5), arrowprops=dict(arrowstyle="->", color="#27AE60", lw=2),
                annotation_clip=False)
    ax.set_title("density: held (blue), pushed (red),\nthe tip's force (green)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(1, 3, 3)
    tri = np.asarray(model.me.tri)[model.body_elements(finger)] - int(model.body_nodes(finger)[0])
    keep = model.to_elements(finger, density(s)) > 0.3
    x_end = model.get_body_x(finger)
    ax.add_collection(PolyCollection(X[tri[keep]] * 1e3, facecolors="0.85", edgecolors="none"))
    ax.add_collection(PolyCollection(x_end[tri[keep]] * 1e3, facecolors="#2E86C1", edgecolors="none", alpha=0.8))
    ax.set_aspect("equal"); ax.autoscale(); ax.set_title(f"pushed (blue) over rest (grey)\ntip reaction {J_end:.2f} N/m", fontsize=10)
    fig.tight_layout(); fig.savefig("topopt_gripper.png", dpi=130)
    print("wrote topopt_gripper.png")
