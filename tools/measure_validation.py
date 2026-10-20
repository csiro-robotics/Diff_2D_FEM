"""Actual values behind the README's validation table, at the package defaults."""
import sys, warnings, numpy as np, warp as wp
wp.config.quiet = True; warnings.simplefilter("ignore")
R = str(__import__("pathlib").Path(__file__).resolve().parents[2]); sys.path.insert(0, R); sys.path.insert(0, R + "/Diff_2D_FEM/tests")
import Diff_2D_FEM as D, test_rigid as T, reference_static as REF
wp.init()
G0, E, NU, RHO = 9.81, 1.0e6, 0.40, 1100.0
print("precision", D.PRECISION)
# free fall of a soft block, defaults, capped iterations
for rm in (False, True):
    row = []
    for its in (5, 20, 150):
        model = D.Model(); T._block(model, "soft", (0.0, 0.05)); model.finalize(); model.gravity = (0.0, -G0)
        y0 = T._com(model)[1]; integ = D.ImplicitIntegrator(max_iter=its, self_collision=False, friction=False, rigid_modes=rm)
        for _ in range(100): integ.simulate(model, 1e-3)
        row.append((y0 - T._com(model)[1]) / (G0 * 1e-6 * 100 * 101 / 2) - 1)
    print(f"free fall, soft, rigid_modes={rm}: " + "  ".join(f"{r:+.2e}" for r in row))
# slope
for kind in ("rigid", "soft"):
    for th_deg, mu in ((30.0, 0.30), (15.0, 0.60)):
        th = np.deg2rad(th_deg); c, dt = 20.0, 1e-3
        model = D.Model(); model.add_ground(height=0.0, mu=mu); model.damping = c
        T._block(model, kind, (0.0, 0.003 + 0.9 * 0.5e-3 / 8)); model.finalize()
        integ = D.ImplicitIntegrator(self_collision=False); model.gravity = (0.0, -G0 * np.cos(th))
        for _ in range(300): integ.simulate(model, dt)
        W = np.asarray(model.me.mass).sum() * G0 * np.cos(th); N = abs(integ.collider_force(model, friction=False)[0][1])
        model.gravity = (G0 * np.sin(th), -G0 * np.cos(th)); xs = []; its = 0
        for _ in range(400): integ.simulate(model, dt); xs.append(T._com(model)[0]); its += int(integ.stats(model)["iters"][0])
        v = np.diff(np.array(xs)[-60:]).mean() / dt; ce = (1.0 - np.exp(-dt * c)) / dt
        ex = G0 * (np.sin(th) - mu * np.cos(th)) / ce if np.tan(th) > mu else G0 * np.sin(th) / (mu * G0 * np.cos(th) / model.epsv + ce)
        print(f"slope {kind:5} {th_deg:.0f} deg mu {mu}: N/W - 1 = {N/W-1:+.2e}   speed/expected - 1 = {v/ex-1:+.2e}   its/step {its/400:.0f} of {integ.stats(model)['max_iter']}")
# rigid region cantilever
model = D.Model(); model.damping = 300.0
h = model.add_soft_body(np.ones((6, 30), np.uint8), physical_size=(0.020, 0.004), center=(0.0, 0.0), E=E, nu=NU, density=RHO); model.finalize()
region = np.zeros((6, 30), bool); region[:, 18:] = True; model.add_affine_region(h, region)
X = model.rest_x(h); left = np.where(X[:, 0] < X[:, 0].min() + 1e-9)[0]; tip = np.where(X[:, 0] > X[:, 0].max() - 1e-9)[0]
model.set_kinematic_constraint(h, left); f = np.zeros_like(X); f[tip, 1] = -6.0 / len(tip)
xr, res = REF.solve_static(model.me, left, fext=f, ramps=10); model.set_external_force(h, f)
integ = D.ImplicitIntegrator(self_collision=False, friction=False)
for _ in range(300): integ.simulate(model, 2e-3)
x = model.get_body_x(h).astype(np.float64)
print(f"rigid-half cantilever: tip {((x[tip,1]-X[tip,1]).mean())*1e3:.4f} mm, reference {((xr[tip,1]-X[tip,1]).mean())*1e3:.4f} mm, ratio - 1 = {(x[tip,1]-X[tip,1]).mean()/(xr[tip,1]-X[tip,1]).mean()-1:+.2e}")
# pendulum
model = D.Model(); h = model.add_affine_body(np.ones((4, 24), np.uint8), physical_size=(0.012, 0.002), center=(0.006, 0.0), density=RHO); model.finalize()
pin = model.add_pin(h, (0.0, 0.0)); X = model.rest_x(h); m = np.asarray(model.me.mass, np.float64)[model.body_nodes(h)]
pivot = X[np.linalg.norm(X, axis=1).argmin()]; d = np.linalg.norm((m[:, None] * X).sum(0) / m.sum() - pivot); Ip = (m * ((X - pivot) ** 2).sum(1)).sum()
th0 = np.deg2rad(10.0); period = 2 * np.pi * np.sqrt(Ip / (m.sum() * G0 * d)) * (1 + th0 ** 2 / 16)
model.gravity = (G0 * np.cos(th0), -G0 * np.sin(th0)); dt = 5e-4; integ = D.ImplicitIntegrator(self_collision=False, friction=False); ang = []; gap = 0
for _ in range(int(1.1 * period / dt)):
    integ.simulate(model, dt); ang.append(model.affine_state(0)["angle"] + th0); gap = max(gap, abs(model.spring_force(pin)[1]))
ang = np.array(ang); t = dt * np.arange(1, len(ang) + 1); k = np.where(np.sign(ang[1:]) != np.sign(ang[:-1]))[0]
cross = [t[i] - ang[i] * dt / (ang[i + 1] - ang[i]) for i in k[:2]]
print(f"pendulum: period/expected - 1 = {2*(cross[1]-cross[0])/period-1:+.2e}; pin gap {gap:.1e} m")
# impact
model = D.Model(); sq = dict(shape=(8, 8), size=(0.004, 0.004))
rigid = T._block(model, "rigid", (0.0, 0.0), **sq); soft = T._block(model, "soft", (0.0065, 0.0), **sq); model.finalize()
model.set_body_velocity(rigid, velocity=(0.3, 0.0)); m = np.asarray(model.me.mass, np.float64)
mom = lambda: (m[:, None] * np.concatenate([model.get_body_v(rigid), model.get_body_v(soft)]).astype(np.float64)).sum(0)
p0 = mom(); integ = D.ImplicitIntegrator(friction=False)
for _ in range(80): integ.simulate(model, 2.5e-4)
print(f"impact: |dp|/|p| = {np.linalg.norm(mom()-p0)/np.linalg.norm(p0):.2e}; soft block v = {(m[model.body_nodes(soft), None]*model.get_body_v(soft)).sum(0)[0]/m[model.body_nodes(soft)].sum():.3f} m/s")
