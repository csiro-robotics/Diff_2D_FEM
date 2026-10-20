"""Collider geometry on the host — a numpy mirror of core.sdf.sdf_eval.

The solver never sees a mesh for the object: it is an analytic signed-distance function
phi(x), and contact is the IPC barrier on phi. This module gives the *plotting and metrics*
side the same shapes, so a figure or a contact-arc measurement can never disagree with what
the solver contacted.

Keep the formulas here equivalent to core.sdf.sdf_eval — including the corner-radius clamp —
or the drawn outline and the contact set drift apart. tests/test_objects.py checks this.

    obj = Obj(kind=BOX, cx=0.06, cy=0.083, rx=0.018, ry=0.018, rref=0.003)
    phi(P, obj)          # (n,) signed distance, >0 outside
    patch(obj)           # matplotlib artist
    outline(obj, 512)    # (n,2) points ON the surface
    perimeter(obj)       # arclength, for wrap normalisation
    contact_arc(...)     # shape-aware contact length
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

NONE = -1                                     # no collider in the scene
DISC, ELLIPSE, BOX, TRIANGLE, HALFSPACE = 0, 1, 2, 3, 4
NAMES = {"disc": DISC, "ellipse": ELLIPSE, "box": BOX, "square": BOX, "triangle": TRIANGLE}
PLANE_EXTENT = 1.0                            # how much of an (infinite) plane to draw / sample [m]


def _tri_verts(o):
    """Inner-triangle vertices (CCW, apex +y). The rounded shape is this offset by r."""
    r = min(max(o.rref, 0.0), o.rx * 0.5)
    rin = o.rx - r
    a = np.pi / 2 + np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    return o.c + rin * np.stack([np.cos(a), np.sin(a)], 1), r


@dataclass
class Obj:
    kind: int = DISC
    cx: float = 0.060
    cy: float = 0.080
    rx: float = 0.014          # disc: radius; ellipse/box: x half-extent; plane: normal x
    ry: float = 0.014          # ellipse/box: y half-extent;               plane: normal y
    rref: float = 0.014        # ellipse: level-set scale; box: corner radius

    @property
    def c(self):
        return np.array([self.cx, self.cy])

    def _r_round(self):
        """Corner radius, clamped exactly as sdf_eval does."""
        return min(max(self.rref, 0.0), min(self.rx, self.ry))

    @property
    def normal(self):
        """Unit normal of a plane, pointing into the free side."""
        return np.array([self.rx, self.ry])

    def label(self):
        if self.kind == NONE:
            return "no collider"
        if self.kind == HALFSPACE:
            return f"plane through ({self.cx*1e3:.0f}, {self.cy*1e3:.0f}) mm, normal ({self.rx:.2f}, {self.ry:.2f})"
        if self.kind == DISC:
            return f"disc r={self.rx*1e3:.0f}mm"
        if self.kind == ELLIPSE:
            return f"ellipse {self.rx*2e3:.0f}x{self.ry*2e3:.0f}mm"
        if self.kind == BOX:
            return f"box {self.rx*2e3:.0f}x{self.ry*2e3:.0f}mm (r={self._r_round()*1e3:.0f}mm)"
        return f"triangle R={self.rx*1e3:.0f}mm (r={min(max(self.rref,0.0), self.rx*0.5)*1e3:.0f}mm)"


def phi(P, o: Obj):
    """Signed distance (>0 outside). Mirrors core.sdf.sdf_eval."""
    P = np.atleast_2d(np.asarray(P, float))
    if o.kind == NONE:
        return np.full(len(P), np.inf)
    if o.kind == HALFSPACE:
        return (P - o.c) @ o.normal
    if o.kind == DISC:
        return np.linalg.norm(P - o.c, axis=1) - o.rx
    if o.kind == ELLIPSE:
        u = ((P[:, 0] - o.cx) / o.rx) ** 2 + ((P[:, 1] - o.cy) / o.ry) ** 2
        return o.rref * (np.sqrt(np.maximum(u, 1e-18)) - 1.0)
    if o.kind == BOX:
        r = o._r_round()
        q = np.abs(P - o.c) - np.array([o.rx - r, o.ry - r])
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0) - r
    V, r = _tri_verts(o)                       # TRIANGLE: rounded convex polygon
    d = np.full(len(P), np.inf); ins = np.ones(len(P), bool)
    for i in range(3):
        a, b = V[i], V[(i + 1) % 3]
        ab = b - a; t = ((P - a) @ ab) / max(ab @ ab, 1e-30)
        q = a + np.clip(t, 0, 1)[:, None] * ab
        d = np.minimum(d, np.linalg.norm(P - q, axis=1))
        ins &= (ab[0] * (P[:, 1] - a[1]) - ab[1] * (P[:, 0] - a[0])) >= 0
    return np.where(ins, -d, d) - r


def patch(o: Obj, **kw):
    """Matplotlib artist for the ACTUAL shape."""
    from matplotlib.patches import Circle, Ellipse as MplEllipse, FancyBboxPatch, BoxStyle
    if o.kind == HALFSPACE:                             # the solid side, as a wide slab
        from matplotlib.patches import Polygon
        t = np.array([-o.ry, o.rx]); e = PLANE_EXTENT
        return Polygon([o.c - e * t, o.c + e * t, o.c + e * t - e * o.normal, o.c - e * t - e * o.normal],
                       closed=True, **kw)
    if o.kind == DISC:
        return Circle((o.cx, o.cy), o.rx, **kw)
    if o.kind == ELLIPSE:
        return MplEllipse((o.cx, o.cy), 2 * o.rx, 2 * o.ry, **kw)
    if o.kind == BOX:
        r = o._r_round()
        return FancyBboxPatch((o.cx - (o.rx - r), o.cy - (o.ry - r)),
                              2 * (o.rx - r), 2 * (o.ry - r),
                              boxstyle=BoxStyle("Round", pad=r), **kw)
    from matplotlib.patches import Polygon              # rounded triangle: sampled outline
    return Polygon(outline(o, 240), closed=True, **kw)


def outline(o: Obj, n: int = 512):
    """(n,2) points on the surface — chamfer targets / wrap reference."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if o.kind == HALFSPACE:
        return o.c + np.linspace(-PLANE_EXTENT, PLANE_EXTENT, n)[:, None] * np.array([-o.ry, o.rx])
    if o.kind == DISC:
        return o.c + o.rx * np.stack([np.cos(t), np.sin(t)], 1)
    if o.kind == ELLIPSE:
        return o.c + np.stack([o.rx * np.cos(t), o.ry * np.sin(t)], 1)
    if o.kind == TRIANGLE:
        V, r = _tri_verts(o)
        pts = []
        for i in range(3):
            a_, b_ = V[i], V[(i + 1) % 3]
            e = b_ - a_; nrm = np.array([e[1], -e[0]]); nrm /= np.linalg.norm(nrm)
            m = max(n // 6, 8)
            pts.append(a_ + np.linspace(0, 1, m)[:, None] * e + r * nrm)   # offset edge
            th0 = np.arctan2(nrm[1], nrm[0])                                # corner arc at b_
            e2 = V[(i + 2) % 3] - b_
            n2 = np.array([e2[1], -e2[0]]); n2 /= np.linalg.norm(n2)
            th1 = np.arctan2(n2[1], n2[0])
            if th1 < th0:
                th1 += 2 * np.pi
            aa = np.linspace(th0, th1, max(n // 12, 4))
            pts.append(b_ + r * np.stack([np.cos(aa), np.sin(aa)], 1))
        return np.concatenate(pts)
    r = o._r_round(); a, b = o.rx - r, o.ry - r
    seg, quad = [], np.linspace(0, np.pi / 2, max(n // 8, 4))
    for (sx, sy, base) in ((1, 1, 0.0), (-1, 1, np.pi / 2), (-1, -1, np.pi), (1, -1, 1.5 * np.pi)):
        seg.append(np.stack([sx * a + r * np.cos(quad + base),
                             sy * b + r * np.sin(quad + base)], 1))
    edges = [np.stack([np.full(n // 8, a + r), np.linspace(-b, b, n // 8)], 1),
             np.stack([np.linspace(a, -a, n // 8), np.full(n // 8, b + r)], 1),
             np.stack([np.full(n // 8, -a - r), np.linspace(b, -b, n // 8)], 1),
             np.stack([np.linspace(-a, a, n // 8), np.full(n // 8, -b - r)], 1)]
    pts = np.concatenate([edges[0], seg[0], edges[1], seg[1],
                          edges[2], seg[2], edges[3], seg[3]])
    return o.c + pts


def perimeter(o: Obj):
    """Arclength — the wrap denominator. 2*pi*r only describes a circle."""
    if o.kind in (HALFSPACE, NONE):
        return np.inf
    if o.kind == DISC:
        return 2 * np.pi * o.rx
    if o.kind == ELLIPSE:                       # Ramanujan
        a, b = max(o.rx, o.ry), min(o.rx, o.ry)
        h = ((a - b) / (a + b)) ** 2
        return np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
    if o.kind == TRIANGLE:                    # inner-polygon perimeter + one full corner circle
        V, r = _tri_verts(o)
        return float(sum(np.linalg.norm(V[(i + 1) % 3] - V[i]) for i in range(3)) + 2 * np.pi * r)
    r = o._r_round()
    return 4 * ((o.rx - r) + (o.ry - r)) + 2 * np.pi * r


def contact_arc(pos, boundary_nodes, boundary_edges, o: Obj, d_hat):
    """Length of a body's outline in contact with the shape: a boundary edge counts in full
    when both its ends are within d_hat of the surface (by phi), by half when one is."""
    pos = np.asarray(pos); bn = np.asarray(boundary_nodes)
    d = phi(pos[bn], o)
    act = np.zeros(pos.shape[0], bool)
    act[bn[(d > 0.0) & (d < d_hat)]] = True
    e = np.asarray(boundary_edges)
    if len(e) == 0:
        return 0.0, act
    L = np.linalg.norm(pos[e[:, 1]] - pos[e[:, 0]], axis=1)
    a, b = act[e[:, 0]], act[e[:, 1]]
    return float(np.where(a & b, L, np.where(a | b, 0.5 * L, 0.0)).sum()), act


def bounds(o: Obj):
    """(xlo, xhi, ylo, yhi) of the object, for autoscaling a view."""
    if o.kind in (HALFSPACE, NONE):
        return o.cx, o.cx, o.cy, o.cy                     # contributes a point, not an extent
    if o.kind == DISC:
        return o.cx - o.rx, o.cx + o.rx, o.cy - o.rx, o.cy + o.rx
    if o.kind == TRIANGLE:
        return o.cx - o.rx, o.cx + o.rx, o.cy - o.rx, o.cy + o.rx
    return o.cx - o.rx, o.cx + o.rx, o.cy - o.ry, o.cy + o.ry
