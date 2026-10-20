"""How many iterations per step does a clamped soft structure need? Hypothesis: a fixed
multiple of the step's acoustic CFL number C = c_p dt / dx (one CG iteration carries
information one element; a backward-Euler step lets it travel C elements) — up to the
count a static solve of the structure needs, which is where it levels off for large steps
(the second table)."""
import sys, time, warnings, numpy as np, warp as wp, scipy.sparse.linalg as spla
wp.config.quiet = True; warnings.simplefilter("ignore")
R = str(__import__("pathlib").Path(__file__).resolve().parents[2]); sys.path.insert(0, R); sys.path.insert(0, R + "/Diff_2D_FEM/tests")
import Diff_2D_FEM as D, reference_static as REF
wp.init()
E, NU, RHO = 1.0e6, 0.40, 1100.0; EP = E / (1 - NU * NU)
CP = np.sqrt(E * (1 - NU) / ((1 + NU) * (1 - 2 * NU)) / RHO)
def curvature(X, x, Lb):
    mid = np.abs(X[:, 1]) < 1e-9; xs = X[mid, 0]; o = np.argsort(xs); s = (xs[o] - xs.min()) / Lb
    c = (s > 0.2) & (s < 0.8); px, py = x[mid][o][c, 0], x[mid][o][c, 1]; px = px - px.mean(); py = py - py.mean()
    A = np.c_[2 * px, 2 * py, np.ones_like(px)]; sol = np.linalg.lstsq(A, px ** 2 + py ** 2, rcond=None)[0]
    return 1.0 / np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2)
def beam(ny, dt, its, T_end=0.4, ramp=0.05):
    asp, hb = 10, 0.004; Lb = hb * asp; e = hb / 4; T = 30.0
    model = D.Model(); model.damping = 200.0
    h = model.add_soft_body(np.ones((ny, ny * asp), np.uint8), physical_size=(Lb, hb), center=(0.0, 0.0), E=E, nu=NU, density=RHO)
    model.finalize(); X = model.me.rest_pos.astype(np.float64)
    model.set_kinematic_constraint(h, np.where(X[:, 0] < X[:, 0].min() + 1e-9)[0])
    td = model.add_tendon(h, through=[(xx, e) for xx in np.unique(np.round(X[:, 0], 9))])
    integ = D.ImplicitIntegrator(max_iter=its, eps=1e-10, self_collision=False, friction=False)
    n = int(round(T_end / dt)); nr = max(int(round(ramp / dt)), 1)
    for t in range(1, n + 1):
        model.set_tendon_tension(td, T * min(t / nr, 1.0)); integ.simulate(model, dt)
    return curvature(X, model.get_x().astype(np.float64), Lb) / (T * e / (EP * hb ** 3 / 12))
print(f"c_p = {CP:.1f} m/s")
print(f"  {'case':28} {'C':>6} {'its':>5} {'its/C':>6} {'curvature / beam theory':>24} {'time':>6}")
ref = {}
for ny, dt in ((8, 1e-3), (8, 2e-3), (16, 1e-3), (8, 4e-3)):
    dx = 0.004 / ny; C = CP * dt / dx
    for f in (0.5, 1.0, 1.5, 2.0, 3.0, 6.0):
        its = int(round(f * C)); t0 = time.time(); k = beam(ny, dt, its)
        print(f"  beam {ny}x{ny*10}, dt {dt*1e3:.0f} ms          {C:>6.0f} {its:>5} {f:>6.1f} {k:>24.4f} {time.time()-t0:>5.0f}s", flush=True)
print("large steps, beam 8x80: the count levels off instead of following C")
print(f"  {'dt':>6} {'C':>6} {'its':>5} {'its/C':>6} {'curvature / beam theory':>24} {'time':>6}")
for dt in (10e-3, 20e-3, 40e-3):
    C = CP * dt / (0.004 / 8)
    for its in (177, 354, 500, 707, 1000):
        t0 = time.time(); k = beam(8, dt, its)
        print(f"  {dt*1e3:4.0f}ms {C:>6.0f} {its:>5} {its/C:>6.2f} {k:>24.4f} {time.time()-t0:>5.0f}s", flush=True)
