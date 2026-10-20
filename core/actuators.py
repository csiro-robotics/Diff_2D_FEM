"""Geometry behind the actuators — which mesh entities a chamber, a tendon or an active
region is made of. Pure numpy; the energies themselves are in implicit/actuation.py.

All pixel arrays use the layout of the body's own mask: shape (H, W), ROW 0 AT THE BOTTOM
(row index grows with y, column index with x). A stiffness map or a region mask is laid
out exactly like the mask that was passed to Model.add_soft_body.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage

_FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


# ---------------------------------------------------------------------- chambers
def enclosed_voids(mask):
    """Label the sealed cavities of a body: connected void regions that do not reach the
    edge of the pixel grid. Returns (labels (H, W) int, [label ids]).

    Voids connect through shared pixel EDGES only (4-connectivity). Two void pixels that
    touch only at a corner are separated by solid meeting at that corner; fluid cannot pass
    through a point, so they are different chambers."""
    solid = np.asarray(mask) > 0
    lab, n = ndimage.label(~solid, structure=_FOUR)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    return lab, [k for k in range(1, n + 1) if k not in border]


def chamber_edges(mask, labels, label, node_of):
    """Directed boundary edges of one cavity, FLUID ON THE LEFT (counter-clockwise round the
    cavity), as two arrays of global node ids (a, b).

    Built pixel by pixel rather than by walking the boundary: every void pixel contributes
    the edge it shares with each solid 4-neighbour, oriented as that edge is traversed going
    counter-clockwise round the pixel. Walking a loop instead would be ambiguous wherever
    two solid pixels touch at a corner (four boundary edges meet at one node)."""
    solid = np.asarray(mask) > 0
    H, W = solid.shape
    ea, eb = [], []
    for iy, ix in zip(*np.nonzero(labels == label)):
        # (neighbour offset) -> (edge start, edge end) as grid-node (ix, iy) offsets
        for (dx, dy), (a, b) in (((1, 0), ((1, 0), (1, 1))),      # right wall, going up
                                 ((0, 1), ((1, 1), (0, 1))),      # top wall, going left
                                 ((-1, 0), ((0, 1), (0, 0))),     # left wall, going down
                                 ((0, -1), ((0, 0), (1, 0)))):    # bottom wall, going right
            jx, jy = ix + dx, iy + dy
            if 0 <= jx < W and 0 <= jy < H and solid[jy, jx]:
                na, nb = node_of(ix + a[0], iy + a[1]), node_of(ix + b[0], iy + b[1])
                if na < 0 or nb < 0:
                    raise RuntimeError("chamber wall node is not in the mesh")
                ea.append(na); eb.append(nb)
    return np.asarray(ea, np.int64), np.asarray(eb, np.int64)


def exterior_edges(mask, labels, enclosed, node_of):
    """Directed edges of a body's OUTER surface — every pixel edge between solid and the
    surroundings (void that reaches the grid edge, or beyond the grid) — with the
    surrounding fluid on the left, i.e. clockwise round the body. Fed to the same pressure
    term as a cavity, an ambient pressure then squeezes the body instead of inflating it."""
    solid = np.asarray(mask) > 0
    H, W = solid.shape
    enclosed = set(enclosed)
    ea, eb = [], []
    for iy, ix in zip(*np.nonzero(solid)):
        # the solid pixel's own counter-clockwise edge is (a -> b); the fluid is on the other
        # side, so the stored edge runs b -> a
        for (dx, dy), (a, b) in (((1, 0), ((1, 0), (1, 1))), ((0, 1), ((1, 1), (0, 1))),
                                 ((-1, 0), ((0, 1), (0, 0))), ((0, -1), ((0, 0), (1, 0)))):
            jx, jy = ix + dx, iy + dy
            outside = not (0 <= jx < W and 0 <= jy < H)
            if outside or (not solid[jy, jx] and labels[jy, jx] not in enclosed):
                ea.append(node_of(ix + b[0], iy + b[1])); eb.append(node_of(ix + a[0], iy + a[1]))
    return np.asarray(ea, np.int64), np.asarray(eb, np.int64)


def mesh_boundary_loops(tri, x):
    """The boundary of a triangle mesh as closed loops of directed edges, FLUID ON THE LEFT
    — the convention of chamber_edges and exterior_edges — as a list of
    (ea, eb, signed area). A HOLE in the mesh runs counter-clockwise (area > 0) and is a
    cavity; a piece of the OUTLINE runs clockwise (area < 0) and is where an ambient
    pressure acts.

    Each boundary edge is taken as it runs in its triangle made counter-clockwise (the
    material on its left) and reversed. Node ids are those of `tri`. A node where two loops
    meet (two triangles touching only at a corner across the boundary) makes the loops
    ambiguous and is refused."""
    tri = np.asarray(tri, np.int64)
    x = np.asarray(x, np.float64)
    p = x[tri]
    cw = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
          - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])) < 0
    t = tri.copy(); t[cw, 1], t[cw, 2] = tri[cw, 2], tri[cw, 1]     # every triangle counter-clockwise
    d = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])   # material on the left
    _, inv, cnt = np.unique(np.sort(d, 1), axis=0, return_inverse=True, return_counts=True)
    fl = d[cnt[inv.reshape(-1)] == 1][:, ::-1]                       # boundary, fluid on the left
    starts, n_out = np.unique(fl[:, 0], return_counts=True)
    if (n_out > 1).any():
        raise ValueError(f"node {starts[np.argmax(n_out > 1)]} is where two boundary loops meet "
                         f"(two triangles touching only at a corner): the pressure boundary is "
                         f"ambiguous there. Separate them by an element, or join them.")
    nxt = dict(zip(fl[:, 0].tolist(), range(len(fl))))
    seen = np.zeros(len(fl), bool)
    loops = []
    for e0 in range(len(fl)):
        if seen[e0]:
            continue
        ids, e = [], e0
        while not seen[e]:
            seen[e] = True; ids.append(e)
            e = nxt[int(fl[e, 1])]
        ea, eb = fl[ids, 0], fl[ids, 1]
        loops.append((ea, eb, loop_area(x, ea, eb)))
    return loops


def point_in_loop(point, x, ea, eb):
    """Is a point inside a closed loop of edges (a -> b)? Even-odd rule."""
    px, py = float(point[0]), float(point[1])
    a, b = np.asarray(x, np.float64)[ea], np.asarray(x, np.float64)[eb]
    up = (a[:, 1] > py) != (b[:, 1] > py)
    xc = a[up, 0] + (py - a[up, 1]) * (b[up, 0] - a[up, 0]) / (b[up, 1] - a[up, 1])
    return bool((xc > px).sum() % 2)


def loop_area(x, ea, eb, ref=None):
    """Signed area enclosed by directed edges (a -> b): positive for a cavity whose edges
    were built by chamber_edges."""
    x = np.asarray(x, np.float64)
    c = x[ea].mean(0) if ref is None else np.asarray(ref, np.float64)
    ra, rb = x[ea] - c, x[eb] - c
    return float(0.5 * (ra[:, 0] * rb[:, 1] - ra[:, 1] * rb[:, 0]).sum())


def pixel_of(point, size, center, shape):
    """(row, col) of the pixel containing a world point, or None if it is off the grid."""
    H, W = shape
    col = int(np.floor((point[0] - (center[0] - 0.5 * size[0])) / (size[0] / W)))
    row = int(np.floor((point[1] - (center[1] - 0.5 * size[1])) / (size[1] / H)))
    return (row, col) if (0 <= row < H and 0 <= col < W) else None


# ---------------------------------------------------------------------- tendons
def snap_path(points, node_pos, node_ids):
    """Nearest mesh node to each waypoint, with consecutive repeats dropped."""
    points = np.atleast_2d(np.asarray(points, np.float64))
    d = np.linalg.norm(node_pos[None, :, :] - points[:, None, :], axis=2)
    ids = np.asarray(node_ids)[d.argmin(1)]
    keep = np.concatenate([[True], ids[1:] != ids[:-1]])
    return ids[keep]


def path_length(x, nodes):
    x = np.asarray(x, np.float64)[np.asarray(nodes)]
    return float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum())


# ---------------------------------------------------------------------- active regions
def region_elements(region, tri_quad, tri_ids, ny):
    """Global element ids of a body whose pixel lies in `region` ((H, W) bool)."""
    region = np.asarray(region) > 0
    q = np.asarray(tri_quad)[tri_ids]                 # body-local quad index = ix*ny + iy
    return np.asarray(tri_ids)[region[q % ny, q // ny]]


def activation_matrix(stretch, transverse=1.0, fiber=None):
    """Rest-shape change A of an active region. `stretch` scales lengths along the fiber
    direction and `transverse` across it; with fiber=None the stretch is isotropic.
    fiber is an angle in radians or a 2-vector."""
    if fiber is None:
        return np.eye(2) * float(stretch)
    f = np.asarray([np.cos(fiber), np.sin(fiber)] if np.isscalar(fiber) else fiber, np.float64)
    f = f / np.linalg.norm(f)
    g = np.array([-f[1], f[0]])
    return float(stretch) * np.outer(f, f) + float(transverse) * np.outer(g, g)


def pixel_field_on_elements(field, tri_quad, tri_ids, ny):
    """Value of an (H, W) pixel field on each of a body's elements."""
    q = np.asarray(tri_quad)[tri_ids]
    return np.asarray(field, np.float64)[q % ny, q // ny]
