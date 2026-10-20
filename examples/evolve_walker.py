"""Evolving a pneumatic walker: a block with two air channels, driven 90 degrees apart, and
a genetic algorithm that decides where the material is.

The body is a 100 x 50 mm block with two upright 15 x 25 mm air channels, one near each
end. The channels are inflated in turn, never under vacuum:

    p_left = P (1 + sin wt) / 2        p_right = P (1 + cos wt) / 2

Their walls (2.5 mm) are fixed material, so the channels are always sealed. Everywhere
else in the block the optimiser decides between material and void: it can grow legs, cut
a waist between the channels, or leave the block as drawn. A design is a smooth field
(low-frequency cosine modes, a CPPN in miniature) cut at zero — a genome of 19 numbers —
so even a random design is a coherent shape, not speckle. Where the field would cut the
walker in two, the cut is lowered just enough to join the channels again, so every genome
is one piece. The fitness is how fast the body's centre of mass moves forwards (+x) on a
high-friction floor, measured after the first cycle.

Every generation is ONE simulation: each design is its own environment (env_id=i), the
population advanced by one batched solve. The search is a plain genetic algorithm —
elitism, Gaussian mutation and uniform crossover on the genome; nothing in it knows about
walking. Generation 0 includes the block as drawn, the baseline to beat.

    python examples/evolve_walker.py                    # --quick for a small run
    python examples/evolve_walker.py --plot --viz       # save evolve_walker.png, then watch the best

Per unit depth; SI units.
"""
import sys
import time
import numpy as np
import warp as wp
from scipy import ndimage
import Diff_2D_FEM as D

wp.init()
QUICK = "--quick" in sys.argv
arg = lambda name, default: type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
PX = 1.25e-3                               # 1.25 mm pixels
L, H = 100e-3, 50e-3                       # the block [m]
CW, CH, END, WALL = 15e-3, 25e-3, 12.5e-3, 2.5e-3   # channels: width, height, distance from the ends; wall
E, NU, RHO, MU = 1.0e6, 0.40, 1100.0, 0.8  # a silicone on a high-friction floor
P_MAX = arg("--pressure", 15.0e3)          # [Pa]
PERIOD, CYCLES, DT, N_SETTLE = 0.4, 3, arg("--dt", 2.0e-3), 50
KX, KY = 6, 3                              # cosine modes of the material field, along x and y
POP, GENS = (12, 3) if QUICK else (48, 12)
ELITE, SIGMA, P_CROSS = POP // 4, 0.35, 0.5
rng = np.random.default_rng(arg("--seed", 0))

NX, NY = int(round(L / PX)), int(round(H / PX))     # 80 x 40 pixels, row 0 at the bottom
px = lambda v: int(round(v / PX))
CHANNELS, RINGS = np.zeros((NY, NX), bool), []
for c0 in (px(END), px(L - END - CW)):              # left, right
    ch = np.zeros((NY, NX), bool)
    ch[px((H - CH) / 2):px((H + CH) / 2), c0:c0 + px(CW)] = True
    w = px(WALL)
    RINGS.append(ndimage.binary_dilation(ch, structure=np.ones((2 * w + 1, 2 * w + 1), bool)) & ~ch)
    CHANNELS |= ch
SKELETON = RINGS[0] | RINGS[1]                      # the channels' walls: always material
CHANNEL_X = L / 2 - END - CW / 2                    # the channels' centres, from the block's centre
_x, _y = (np.arange(NX) + 0.5) / NX, (np.arange(NY) + 0.5) / NY
BASIS = np.stack([np.outer(np.cos(np.pi * ky * _y), np.cos(np.pi * kx * _x))
                  for ky in range(KY) for kx in range(KX)])
N_GENES = len(BASIS) + 1                                             # the modes and a bias
BLOCK = np.concatenate([np.zeros(N_GENES - 1), [10.0]])              # the block as drawn


# --- designs: genome -> material field -> pixels ----------------------------------------
def decode(genome):
    """Material where the field exceeds a cut level, plus the walls, minus the channels; the
    piece that holds the walls. The cut is 0, or lowered just enough (bisection) that both
    channels are in that one piece."""
    field = np.tensordot(genome[:-1], BASIS, 1) + genome[-1]

    def piece(cut):
        lab, _ = ndimage.label((SKELETON | (field > cut)) & ~CHANNELS)
        a, b = lab[RINGS[0]][0], lab[RINGS[1]][0]
        return lab == a, a == b

    m, joined = piece(0.0)
    if not joined:
        lo, hi = field.min() - 1.0, 0.0                  # joined at lo (all material), not at hi
        for _ in range(30):
            c = 0.5 * (lo + hi)
            lo, hi = (c, hi) if piece(c)[1] else (lo, c)
        m, _ = piece(lo)
    return m


def random_genome():
    return np.concatenate([rng.normal(0.0, 1.0, N_GENES - 1), [0.0]])


def child(parents):
    g = parents[rng.integers(len(parents))].copy()
    if rng.random() < P_CROSS:
        o = parents[rng.integers(len(parents))]
        take = rng.random(N_GENES) < 0.5
        g[take] = o[take]
    return g + rng.normal(0.0, SIGMA, N_GENES)


# --- one generation: every design in its own environment, one batched solve --------------
def evaluate(masks, view=False, keep=None):
    model = D.Model()
    model.add_ground(height=0.0, mu=MU)
    model.damping = 5.0                                      # a little air drag [1/s]
    bodies, heights = [], []
    for i, m in enumerate(masks):
        cy = (NY / 2 - np.argwhere(m)[:, 0].min()) * PX + PX / 8   # its lowest pixel just above the ground
        heights.append(cy)
        bodies.append(model.add_soft_body(m.astype(np.uint8), physical_size=(L, H), center=(0.0, cy),
                                          E=E, nu=NU, density=RHO, mu=MU, env_id=i))
    model.finalize()
    model.gravity = (0.0, -9.81)
    chambers = [(model.add_pressure_chamber(h, at=(-CHANNEL_X, cy)), model.add_pressure_chamber(h, at=(CHANNEL_X, cy)))
                for h, cy in zip(bodies, heights)]           # a point inside each channel picks it
    mass = np.asarray(model.me.mass)
    centre = lambda h: (mass[model.body_nodes(h), None] * model.get_body_x(h)).sum(0) / mass[model.body_nodes(h)].sum()

    integ = D.ImplicitIntegrator()
    win = D.Renderer(model, envs="all") if view else None
    for _ in range(N_SETTLE):                                # settle onto the ground
        integ.simulate(model, DT)
    frames, per_cycle = [], int(round(PERIOD / DT))
    for i in range(CYCLES * per_cycle):
        if i == per_cycle:                                   # the first cycle is the start-up
            x0 = np.array([centre(h)[0] for h in bodies])
        t = (i + 1) * DT
        ramp = min(1.0, t / (PERIOD / 4))                    # ease the pressure in over a quarter cycle
        for left, right in chambers:
            model.set_pressure(left, ramp * P_MAX * 0.5 * (1.0 + np.sin(2 * np.pi * t / PERIOD)))
            model.set_pressure(right, ramp * P_MAX * 0.5 * (1.0 + np.cos(2 * np.pi * t / PERIOD)))
        integ.simulate(model, DT)
        if win: win.draw()
        if keep is not None and i >= per_cycle and i % keep == 0:
            frames.append(model.get_x())
    dist = np.array([centre(h)[0] for h in bodies]) - x0
    return dist / ((CYCLES - 1) * PERIOD), model, frames, win   # speed [m/s], + what the plot needs


# --- evolution ---------------------------------------------------------------------------
pop = [BLOCK] + [random_genome() for _ in range(POP - 1)]
history, t0 = [], time.time()
print(f"evolving a pneumatic walker: population {POP}, {GENS} generations, {CYCLES} cycles of {PERIOD} s "
      f"at up to {P_MAX * 1e-3:.0f} kPa, {N_GENES} genes")
for gen in range(GENS):
    tg = time.time()
    speed, _, _, _ = evaluate([decode(g) for g in pop])
    if gen == 0:
        block_speed = speed[0]
        print(f"  the block as drawn: {block_speed * 1e3:6.2f} mm/s")
    order = np.argsort(-speed)
    history.append((speed[order[0]], np.median(speed)))
    print(f"  generation {gen + 1:2d}: best {speed[order[0]] * 1e3:6.2f} mm/s, median {np.median(speed) * 1e3:6.2f} mm/s"
          f"   ({time.time() - tg:.0f} s)", flush=True)
    elite = [pop[k] for k in order[:ELITE]]
    if gen < GENS - 1:
        pop = elite + [child(elite) for _ in range(POP - ELITE)]
best = decode(elite[0])
print(f"best walker: {history[-1][0] * 1e3:.2f} mm/s (the block as drawn: {block_speed * 1e3:.2f} mm/s); "
      f"{time.time() - t0:.0f} s of wall time")
np.save("evolve_walker_best.npy", best)

if "--plot" in sys.argv or "--viz" in sys.argv:
    speed, model, frames, win = evaluate([best], view="--viz" in sys.argv, keep=int(round(PERIOD / DT / 4)))
    if "--plot" in sys.argv:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from Diff_2D_FEM import viz
        fig = plt.figure(figsize=(11, 6.2))
        ax = fig.add_subplot(2, 2, 1)
        h = np.array(history) * 1e3
        ax.plot(range(1, len(h) + 1), h[:, 0], "o-", label="best"); ax.plot(range(1, len(h) + 1), h[:, 1], "s--", label="median")
        ax.axhline(block_speed * 1e3, color="0.5", ls=":", label="block as drawn")
        ax.set_xlabel("generation"); ax.set_ylabel("speed [mm/s]"); ax.legend(); ax.grid(alpha=0.3)
        ax = fig.add_subplot(2, 2, 2)
        show = np.where(CHANNELS, 3, np.where(SKELETON, 2, best.astype(int)))
        ax.imshow(show, origin="lower", cmap=ListedColormap(["white", "#7f8c8d", "#34495e", "#aed6f1"]), vmin=-0.5, vmax=3.5)
        ax.set_title("best design: evolved material grey, fixed walls dark, channels blue", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(2, 1, 2)
        for k, x in enumerate(frames[::2]):
            viz.draw(model, ax=ax, x=x, color=plt.cm.viridis(k / max(len(frames[::2]) - 1, 1)))
        xs_all = np.concatenate([f[:, 0] for f in frames])
        ax.set_xlim(xs_all.min() - 0.005, xs_all.max() + 0.005); ax.set_ylim(-0.003, H + 0.01)
        ax.set_title(f"the best walker after start-up, {CYCLES - 1} cycles ({speed[0] * 1e3:.2f} mm/s), dark to light")
        fig.tight_layout(); fig.savefig("evolve_walker.png", dpi=130)
        print("wrote evolve_walker.png")
    if win:
        win.hold()
