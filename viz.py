"""Drawing a scene with matplotlib — bodies, rigid regions, colliders, tendons, springs.

    import matplotlib.pyplot as plt
    from Diff_2D_FEM import viz
    viz.draw(model); plt.savefig("scene.png")

    frames = []                       # an animation: keep the positions you want to show
    for t in range(T):
        integ.simulate(model, dt)
        if t % 10 == 0: frames.append(model.get_x())
    viz.animate(model, frames, "run.gif")

    viz.draw(model, field="stress")             # coloured by von Mises stress, with a colour bar
    viz.animate(model, frames, "run.gif", field="strain")   # recoloured every frame, one scale
    viz.draw(model, field="strain", signed=True)            # tension red, compression blue
    viz.animate(model, frames, "batch.gif", envs="all", field="stress")   # environments side by side

matplotlib is an optional dependency (pip install Diff_2D_FEM[examples]).
"""
from __future__ import annotations
import numpy as np

from .core import objects as _objects
from .core.fields import label as _field_label

SOFT, RIGID, GROUND, PRESCRIBED = "#4C8DAE", "#3B3F46", "#B9B4A7", "#C8C8C8"
PALETTE = ["#4C8DAE", "#D98E48", "#6BA368", "#B5677C", "#8475B2", "#C2B04A"]
FIELD_CMAP = "coolwarm"     # blue (low / compression) - neutral grey - red (high / tension); no rainbow


def _env_elements(model, env):
    """The elements a field is shown on: every body of the environment but the prescribed
    ones (their motion is given, their strain means nothing)."""
    return np.concatenate([model._tri_range[h] for h in range(model.n_bodies)
                           if model.body_env(h) == env and not model._bodies[h].get("kinematic")])


def _field_clim(values, signed):
    """0 to the largest value; signed: symmetric about 0, so grey is unloaded."""
    a = float(np.max(np.abs(values))) if len(values) else 1.0
    return (-a, a) if signed else (0.0, float(np.max(values)))


def _colourbar(ax, norm, cmap, label):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    cb = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.03, label=label)
    cb.formatter.set_powerlimits((-2, 3))            # 1.2 x 1e5 Pa, not 120000
    cb.update_ticks()
    return cb


def draw(model, ax=None, x=None, env=0, color=None, values=None, cmap=None,
         mesh=False, tendons=True, springs=True, margin=0.1, field=None, clim=None, colorbar=None,
         signed=False, _field_values=None):
    """Draw one environment of a scene. Returns the axes.

    x       node positions to draw, (N, 2); default: the model's current state.
    color   one colour for every soft body; default: one colour per body.
    values  per-ELEMENT numbers (n_elements,) to colour the bodies by instead — a density
            field, a stiffness map (model.me.mu), anything laid out like body_elements().
    field   'strain' or 'stress': colour by the von Mises strain or stress at x
            (model.element_strain / element_stress), blue low to red high, with a colour bar.
    signed  with a field: the signed principal value instead of von Mises — tension red (+),
            compression blue (-), unloaded grey; the scale is symmetric about 0.
    cmap    colour map for values / field (default: viridis for values, coolwarm for field).
    clim    (low, high) of the colour scale; default for a field: 0 to its largest value,
            or -a to a with signed=True.
    colorbar  draw a colour bar (default: with a field, not with values).
    mesh    outline every triangle.
    Springs are green lines, and a pin (a spring of zero length to the world) a green dot."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection, LineCollection
    from matplotlib.colors import Normalize
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), layout="constrained")   # room for a colour bar's label
    x = np.asarray(model.get_x() if x is None else x, np.float64)
    if field is not None:
        if values is not None:
            raise ValueError("give values or field, not both")
        values = model._field(field, x, None, signed) if _field_values is None else _field_values
        if clim is None:
            clim = _field_clim(values[_env_elements(model, env)], signed)
    cmap = cmap or (FIELD_CMAP if field is not None else "viridis")
    norm = Normalize(*clim) if clim is not None else None
    me = model.me
    tri = np.asarray(me.tri)
    lo, hi = np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf])

    rigid = np.zeros(len(tri), bool)
    for a in model._affine:
        rigid[a["elems"]] = True
    for h in range(model.n_bodies):
        if model.body_env(h) != env:
            continue
        el = model._tri_range[h]
        P = x[tri[el]]
        lo, hi = np.minimum(lo, P.reshape(-1, 2).min(0)), np.maximum(hi, P.reshape(-1, 2).max(0))
        # without antialiasing the triangles tile exactly; with it, their seams show as hatching
        edge = dict(edgecolors="k", linewidths=0.15) if mesh else \
            dict(edgecolors="none", linewidths=0.0, antialiaseds=False)
        if field is not None and model._bodies[h].get("kinematic"):
            pc = PolyCollection(P, facecolors=PRESCRIBED, **edge)
        elif values is not None:
            pc = PolyCollection(P, array=np.asarray(values)[el], cmap=cmap, norm=norm, **edge)
        else:
            base = color or PALETTE[h % len(PALETTE)]
            pc = PolyCollection(P, facecolors=[RIGID if r else base for r in rigid[el]], **edge)
        ax.add_collection(pc)

    span = float(max(hi - lo)) if np.isfinite(lo).all() else 1.0
    pad = margin * span
    for o in model.colliders:
        if o.kind == _objects.HALFSPACE:                 # an infinite wall: draw what is in view
            _objects.PLANE_EXTENT, keep = 50.0 * span, _objects.PLANE_EXTENT
            ax.add_patch(_objects.patch(o, facecolor=GROUND, edgecolor="none", zorder=0))
            _objects.PLANE_EXTENT = keep
        else:
            ax.add_patch(_objects.patch(o, facecolor=GROUND, edgecolor="#6F6A5E", zorder=0))
            b = _objects.bounds(o)
            lo, hi = np.minimum(lo, [b[0], b[2]]), np.maximum(hi, [b[1], b[3]])

    if tendons and model._tendons:
        segs = [x[t["nodes"]] for t in model._tendons if model.body_env(t["body"]) == env]
        ax.add_collection(LineCollection(segs, colors="#C0392B", linewidths=1.2, zorder=5))
    mine = [k for k in model._springs if k["env"] == env] if springs else []
    if mine:
        # a pin sits on its node, so it is drawn there — its anchor is where it is NOW, which
        # for a moving support is not where it was in an earlier frame
        segs = [[x[k["a"]], x[k["b"]] if k["b"] >= 0 else k["anchor"]] for k in mine
                if k["b"] >= 0 or k["length"] > 0.0]
        if segs:
            ax.add_collection(LineCollection(segs, colors="#2E8B57", linewidths=1.5, zorder=5))
        pts = np.array([x[k["a"]] for k in mine])
        ax.plot(pts[:, 0], pts[:, 1], "o", ms=4, color="#2E8B57", zorder=6)

    ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    if values is not None and (colorbar if colorbar is not None else field is not None):
        _colourbar(ax, norm or Normalize(float(np.min(values)), float(np.max(values))), cmap,
                   _field_label(field, signed) if field is not None else "")
    return ax


def animate(model, frames, path, env=0, fps=25, dpi=110, field=None, clim=None, cmap=None,
            colorbar=True, titles=None, signed=False, envs=None, panel_titles=None, bounds=None,
            panel_size=4.0, **kw):
    """Write the positions in `frames` (a list of (N, 2) arrays) as an animation; the format
    follows the file extension (.gif needs pillow, .mp4 needs ffmpeg). The view is fixed for
    the whole animation.

    env     the environment to show; envs: several side by side instead — a list of them, or
            "all" — e.g. a batch of designs, sharing one colour scale.
    field   'strain' or 'stress': every frame coloured by its own von Mises strain or stress,
            blue low to red high, on ONE scale for the whole animation, with a colour bar.
    signed  the signed principal value instead: tension red (+), compression blue (-).
    clim    (low, high) of that scale; default 0 to the largest value over all frames (and
            environments shown), or symmetric about 0 with signed=True.
    cmap    colour map (default coolwarm).
    titles  one title per frame (e.g. the time or the load), or None.
    panel_titles  one title per environment shown, or None.
    bounds  ((xmin, ymin), (xmax, ymax)) [m], the view of every panel; default: what covers
            each environment's bodies over all frames.
    Other keywords go to draw()."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import Normalize
    shown = list(range(model.n_envs)) if envs == "all" else ([env] if envs is None else list(envs))
    n = len(shown)
    fig, axes = plt.subplots(1, n, figsize=(panel_size * n + (1.2 if field else 0.0), panel_size + 0.6),
                             layout="constrained", squeeze=False)
    axes = axes[0]
    views = []
    for e in shown:
        if bounds is not None:
            views.append(np.asarray(bounds, np.float64))
            continue
        nodes = np.concatenate([model._node_range[h] for h in range(model.n_bodies) if model.body_env(h) == e])
        allx = np.concatenate([np.asarray(f)[nodes] for f in frames])
        lo, hi = allx.min(0), allx.max(0)
        pad = 0.1 * float(max(hi - lo))
        views.append(np.array([lo - pad, hi + pad]))
    values = None
    if field is not None:
        values = [model._field(field, f, None, signed) for f in frames]
        if clim is None:
            el = np.concatenate([_env_elements(model, e) for e in shown])
            clim = _field_clim(np.concatenate([v[el] for v in values]), signed)
        cmap = cmap or FIELD_CMAP
        if colorbar:
            _colourbar(list(axes), Normalize(*clim), cmap, _field_label(field, signed))

    def frame(i):
        for ax, e, (lo, hi) in zip(axes, shown, views):
            ax.clear()
            if field is not None:
                draw(model, ax=ax, x=frames[i], env=e, field=field, signed=signed, cmap=cmap, clim=clim,
                     colorbar=False, _field_values=values[i], **kw)
            else:
                draw(model, ax=ax, x=frames[i], env=e, **kw)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            if panel_titles is not None:
                ax.set_title(panel_titles[shown.index(e)], fontsize=10)
        if titles is not None:
            (axes[0].set_title if n == 1 and panel_titles is None else fig.suptitle)(titles[i])

    FuncAnimation(fig, frame, frames=len(frames)).save(path, fps=fps, dpi=dpi)
    plt.close(fig)
    return path
