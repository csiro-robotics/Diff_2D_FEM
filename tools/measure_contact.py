"""Actual values behind the README's table of contact between bodies, for both modes of
model.body_contact: "ipc" (the default) and "node", the penalty contact before it.

    python tools/measure_contact.py            # about seven minutes on an RTX A5000
"""
import sys, warnings, numpy as np, warp as wp
wp.config.quiet = True; warnings.simplefilter("ignore")
R = str(__import__("pathlib").Path(__file__).resolve().parents[2]); sys.path.insert(0, R)
import Diff_2D_FEM as D
wp.init()
G0, E, NU, RHO, PX = 9.81, 1.0e6, 0.40, 1100.0, 0.5e-3
MODES = ("ipc", "node")
print("precision", D.PRECISION)


def body(model, shape, center, rigid=False, density=RHO):
    kw = dict(physical_size=(shape[1] * PX, shape[0] * PX), center=center, density=density)
    if rigid:
        return model.add_affine_body(np.ones(shape, np.uint8), **kw)
    return model.add_soft_body(np.ones(shape, np.uint8), E=E, nu=NU, **kw)


# ---- a flat block held at a fixed gap over a flat slab and slid along it, no friction
def sliding_energy(mode):
    gap = {"ipc": 0.5 * PX / 8, "node": 1.2 * PX}[mode]
    model = D.Model(device="cpu"); model.body_contact = mode
    body(model, (4, 80), (0.0, -2 * PX)); block = body(model, (8, 24), (0.0, 4 * PX + gap))
    model.finalize(); s = D.ImplicitIntegrator().attach(model, 1e-3)
    X = model.me.rest_pos.astype(np.float64); nodes = model.body_nodes(block); Es = []
    for shift in np.linspace(0.0, 2 * PX, 161):
        x = X.copy(); x[nodes, 0] += shift
        xa = wp.array(x.astype(D.np_real), dtype=D.vec2, device="cpu")
        s.x.assign(xa); s.xn.assign(xa); s.xhat.assign(xa)
        if mode == "node":                      # part of the solver's own energy; a rigid shift
            Es.append(float(s.energy()[0]))     # of the block leaves nothing else in it
        else:
            s.grad.zero_(); s.diagH.zero_(); s.ipc.grad_diag(s)       # builds the broadphase
            buf = wp.zeros(s.B, dtype=wp.float64, device="cpu"); s.ipc.energy(s, buf); Es.append(float(buf.numpy()[0]))
    Es = np.array(Es)
    return (Es.max() - Es.min()) / Es.mean()


# ---- a block on a held slab, gravity tilted
def slope(mode, kind, mu, th_deg, steps=400):
    c, dt = 200.0, 1e-3; ce = (1.0 - np.exp(-dt * c)) / dt; th = np.deg2rad(th_deg)
    lift = {"ipc": 0.9 * PX / 8, "node": 1.5 * PX}[mode]
    model = D.Model(); model.damping = c; model.body_contact = mode; model.body_friction = mu
    slab = body(model, (6, 60), (0.0, 0.0)); block = body(model, (8, 8), (-0.010, 7 * PX + lift), rigid=(kind == "rigid"))
    model.finalize(); model.set_kinematic_constraint(slab, np.arange(model.body_nodes(slab).size))
    integ = D.ImplicitIntegrator(); model.gravity = (0.0, -G0 * np.cos(th))
    for _ in range(150): integ.simulate(model, dt)
    gap = (model.get_body_x(block)[:, 1].min() - model.get_body_x(slab)[:, 1].max()) / model.dx
    model.gravity = (G0 * np.sin(th), -G0 * np.cos(th)); xs = [model.get_body_x(block)[:, 0].mean()]; nc = 0
    for _ in range(steps):
        integ.simulate(model, dt); xs.append(model.get_body_x(block)[:, 0].mean()); nc += int(not integ.stats(model)["converged"][0])
    v = np.diff(np.array(xs))[100:] / dt
    ex = G0 * (np.sin(th) - mu * np.cos(th)) / ce if np.tan(th) > mu else G0 * np.sin(th) / (mu * G0 * np.cos(th) / model.epsv + ce)
    w = np.array([v[i:i + 50].mean() for i in range(0, len(v) - 49, 50)]) / ex
    return v.mean() / ex, w.min(), w.max(), gap, nc


# ---- two soft pads pinch a free rigid block resting on the ground, and lift 4 mm
def pinch(mode, mu, lift=4.0e-3):
    model = D.Model(); model.body_contact = mode; model.body_friction = mu; model.add_ground(0.0, mu=0.5)
    block = body(model, (16, 16), (0.0, 8 * PX + 0.9 * PX / 8), rigid=True, density=600.0)
    pads = [body(model, (16, 8), (sgn * 14 * PX, 8 * PX + 0.8e-3)) for sgn in (-1, +1)]
    model.finalize(); model.gravity = (0.0, -G0)
    for sgn, h in zip((-1, +1), pads):
        X = model.rest_x(h); back = X[:, 0].min() if sgn < 0 else X[:, 0].max()
        model.set_kinematic_constraint(h, np.where(np.abs(X[:, 0] - back) < 1e-9)[0])
    integ = D.ImplicitIntegrator(); y0 = model.get_body_x(block)[:, 1].mean()
    for k in range(1, 351):
        a = min(k / 100, 1.0); b = min(max(k - 100, 0) / 200, 1.0)
        for sgn, h in zip((-1, +1), pads):
            X = model.rest_x(h); X[:, 0] -= sgn * 2.25 * PX * a; X[:, 1] += lift * b; model.set_kinematic_target(h, X)
        integ.simulate(model, 1e-3)
    return (model.get_body_x(block)[:, 1].mean() - y0) / lift


for mode in MODES:
    print(f"\nbody_contact = {mode!r}")
    print(f"  energy while a flat block slides along a flat slab, no friction: changes by {sliding_energy(mode):.3f} of its mean")
    for kind, mu, th in (("rigid", 0.0, 40.0), ("rigid", 0.3, 30.0), ("rigid", 0.5, 40.0), ("rigid", 0.5, 15.0),
                         ("soft", 0.3, 30.0), ("soft", 0.5, 40.0)):
        v, lo, hi, gap, nc = slope(mode, kind, mu, th)
        print(f"  {kind:5} block, mu {mu}, {th:.0f} deg: speed / closed form {v:.4f}  (over 50 steps {lo:.3f} .. {hi:.3f});  "
              f"resting gap {gap:.3f} px;  {nc} of 400 steps not converged", flush=True)
    print(f"  pinch and lift: the block comes up {pinch(mode, 0.0):+.3f} of the lift with no friction, {pinch(mode, 0.8):+.3f} with mu 0.8", flush=True)
