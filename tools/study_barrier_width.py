import sys, numpy as np, warp as wp, time, warnings
wp.config.quiet = True; warnings.simplefilter("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import Diff_2D_FEM as D
wp.init()
NX, NY, L, H, SEG, AMP, PERIOD, DT = 72, 6, 0.054, 0.0045, 6, 0.10, 0.6, 2e-3
E = 0.5e6
def crawl(dhat_px, kscale):
    model = D.Model(); model.add_ground(height=0.0, mu=0.8); model.damping = 5.0
    px = L / NX; model.dhat = dhat_px * px; model.kappa = 4e-3 * E * (1.0 / dhat_px if kscale else 1.0)
    body = model.add_soft_body(np.ones((NY, NX), np.uint8), physical_size=(L, H), center=(0.0, H / 2 + 0.9 * model.dhat), E=E, nu=0.40, density=1100.0)
    model.finalize(); model.gravity = (0.0, -9.81)
    up, lo = [], []
    for k in range(SEG):
        cols = slice(k * NX // SEG, (k + 1) * NX // SEG)
        r = np.zeros((NY, NX), bool); r[NY // 2:, cols] = True; up.append(model.add_active_region(body, r, fiber=0.0))
        r = np.zeros((NY, NX), bool); r[:NY // 2, cols] = True; lo.append(model.add_active_region(body, r, fiber=0.0))
    arch = lambda k, ph: (AMP * np.cos(0.5 * np.pi * (k + 0.5 - ph) / 1.5) ** 2 if abs((k + 0.5 - ph) / 1.5) < 1 else 0.0)
    m = np.asarray(model.me.mass); com = lambda: (m[:, None] * model.get_x()).sum(0) / m.sum()
    integ = D.ImplicitIntegrator()
    for _ in range(50): integ.simulate(model, DT)
    gap = model.get_x()[:, 1].min(); x0 = com()[0]; steps = int(round(PERIOD / DT)); t0 = time.time(); its = 0
    for i in range(steps):
        ph = -1.5 + (SEG + 3.0) * (i + 1) / steps
        for k in range(SEG):
            a = arch(k, ph); model.set_activation(up[k], 1 + a); model.set_activation(lo[k], 1 - a)
        integ.simulate(model, DT); its += int(integ.stats(model)["iters"][0])
    return (com()[0] - x0) * 1e3, gap / model.dhat, its / steps, time.time() - t0, model.get_x()[:, 1].min()
print(f"  {'dhat [px]':>9} {'kappa':>12} {'advance/wave mm':>16} {'rest gap/dhat':>14} {'its/step':>9} {'min y um':>9} {'time':>5}")
for kscale in (False, True):
    for d in (1.0, 0.5, 0.25, 0.125, 0.0625):
        adv, g, its, tt, ymin = crawl(d, kscale)
        print(f"  {d:>9} {'4e-3 E px/dhat' if kscale else '4e-3 E':>14} {adv:>16.3f} {g:>14.3f} {its:>9.0f} {ymin*1e6:>9.1f} {tt:>4.0f}s", flush=True)
