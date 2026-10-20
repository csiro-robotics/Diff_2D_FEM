"""Throughput against the number of environments — the numbers behind the README's table.

Three scenes, each copied into n environments and advanced by one batched solve:

  beam   a clamped cantilever bent by a tendon, a different tension in each environment
         (elasticity only)
  strip  a free strip dropped on the ground (collider contact with friction, the rigid-mode
         coarse space, IPC contact of the strip with itself)
  grip   two soft pads pinch a free rigid block and lift it (IPC contact between bodies with
         friction, a rigid body, kinematic nodes)

    python tools/benchmark_envs.py                       # all three, 1 .. 4096 environments
    python tools/benchmark_envs.py grip 1,64,1024

What is timed is steps of the solve alone, after the kernels are compiled and the scene is in
the state named; "host" is the time spent in Python setting the commands for that step.
"""
import gc, sys, time, warnings, numpy as np, warp as wp
wp.config.quiet = True; warnings.simplefilter("ignore")
R = str(__import__("pathlib").Path(__file__).resolve().parents[2]); sys.path.insert(0, R)
import Diff_2D_FEM as D
wp.init()
E, NU, RHO, PX = 1.0e6, 0.40, 1100.0, 0.5e-3


def beam(n):
    m = D.Model(); m.damping = 50.0
    hs = [m.add_soft_body(np.ones((8, 80), np.uint8), physical_size=(0.040, 0.004), center=(0.0, 0.0),
                          E=E, nu=NU, density=RHO, env_id=i) for i in range(n)]
    m.finalize(); cables = []
    for h in hs:
        X = m.rest_x(h); m.set_kinematic_constraint(h, np.where(X[:, 0] < X[:, 0].min() + 1e-9)[0])
        cables.append(m.add_tendon(h, through=[(x, 0.001) for x in np.unique(np.round(X[:, 0], 9))]))
    def command(step):
        for i, c in enumerate(cables):
            m.set_tendon_tension(c, 10.0 + 20.0 * i / max(n - 1, 1))
    return m, D.ImplicitIntegrator(self_collision=False, friction=False), command, 1e-3, 5, "bent by its tendon"


def strip(n):
    m = D.Model(); m.add_ground(0.0, mu=0.8)
    for i in range(n):
        m.add_soft_body(np.ones((6, 72), np.uint8), physical_size=(0.054, 0.0045), center=(0.0, 0.0030),
                        E=0.5 * E, nu=NU, density=RHO, env_id=i)
    m.finalize(); m.gravity = (0.0, -9.81)
    return m, D.ImplicitIntegrator(), (lambda step: None), 2e-3, 30, "lying on the ground"


def grip(n):
    m = D.Model(); m.body_friction = 0.8; m.add_ground(0.0, mu=0.5); pads = []
    for i in range(n):
        m.add_affine_body(np.ones((16, 16), np.uint8), physical_size=(16 * PX, 16 * PX),
                          center=(0.0, 8 * PX + 0.9 * PX / 8), density=600.0, env_id=i)
        pads.append([m.add_soft_body(np.ones((16, 8), np.uint8), physical_size=(8 * PX, 16 * PX),
                                     center=(s * 14 * PX, 8 * PX + 0.8e-3), E=E, nu=NU, density=RHO, env_id=i)
                     for s in (-1, +1)])
    m.finalize(); m.gravity = (0.0, -9.81)
    rest = [[m.rest_x(p) for p in pair] for pair in pads]
    for pair, xs in zip(pads, rest):
        for s, p, X in zip((-1, +1), pair, xs):
            back = X[:, 0].min() if s < 0 else X[:, 0].max()
            m.set_kinematic_constraint(p, np.where(np.abs(X[:, 0] - back) < 1e-9)[0])
    CLOSE = 40
    def command(step):                                   # close in 40 steps, then lift 20 um a step
        a = min(step / CLOSE, 1.0); up = 20.0e-6 * max(step - CLOSE, 0)
        for pair, xs in zip(pads, rest):
            for s, p, X in zip((-1, +1), pair, xs):
                Y = X.copy(); Y[:, 0] -= s * 2.25 * PX * a; Y[:, 1] += up
                m.set_kinematic_target(p, Y)
    return m, D.ImplicitIntegrator(), command, 1e-3, CLOSE + 5, "block gripped and rising"


def run(name, build, n):
    gc.collect(); gpu = str(wp.get_preferred_device()).startswith("cuda")
    used0 = wp.get_mempool_used_mem_current(wp.get_preferred_device()) if gpu else 0     # this process's own
    t0 = time.perf_counter(); m, integ, command, dt, lead, state = build(n); t_build = time.perf_counter() - t0
    step = 0
    for _ in range(lead):                                # compile, and reach the state to be timed
        step += 1; command(step); integ.simulate(m, dt)
    wp.synchronize()
    steps = 10 if n >= 1024 else 20
    t_host = 0.0; its = []; t0 = time.perf_counter()
    for _ in range(steps):
        step += 1
        c0 = time.perf_counter(); command(step); t_host += time.perf_counter() - c0
        integ.simulate(m, dt); its.append(int(integ.stats(m)["iters"].max()))
    wp.synchronize(); t = (time.perf_counter() - t0) / steps
    st = integ.stats(m)
    mem = (wp.get_mempool_used_mem_current(m.device) - used0) / 2 ** 30 if gpu else float("nan")
    print(f"{name:5} {n:5d} envs {m.n_nodes:8d} nodes | build {t_build:5.1f} s | {t * 1e3:8.1f} ms/step "
          f"({np.mean(its):4.0f} of {st['max_iter']} iterations, {t / np.mean(its) / m.n_nodes * 1e9:5.1f} ns/node/iteration, "
          f"host {t_host / steps * 1e3:5.1f} ms) | {n / t:7.0f} env-steps/s = {n * dt / t:6.2f} x real time | "
          f"{int(st['converged'].sum()):5d} converged | {mem:5.2f} GiB   [{state}]", flush=True)


SCENES = {"beam": beam, "strip": strip, "grip": grip}
if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if a in SCENES] or list(SCENES)
    counts = [int(v) for a in sys.argv[1:] if a not in SCENES for v in a.split(",")] or [1, 16, 64, 256, 1024, 4096]
    print(f"precision {D.PRECISION}; device {wp.get_preferred_device()}")
    for name in which:
        for n in counts:
            run(name, SCENES[name], n)
