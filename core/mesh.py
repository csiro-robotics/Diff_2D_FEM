"""Triangle meshes from pixel masks or given directly (numpy).

A body is either a PIXEL MASK — each solid pixel of its (nx, ny) grid becomes two P1
triangles; grid corner (ix, iy) is node (ny+1)*ix + iy before the nodes no solid pixel
touches are pruned — or a TRIANGLE MESH, taken as given: vertex i is node i, triangle j is
element j. build_scene lays the bodies of every environment out as one flat mesh for the
Warp kernels, with lumped masses and Lame moduli per element.
"""
from __future__ import annotations
import warnings
from collections import deque
from dataclasses import dataclass
import numpy as np


def lame(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


@dataclass
class Mesh:
    n_envs: int
    rest_pos: np.ndarray         # (sum_N, 2)
    tri: np.ndarray              # (sum_n_tri, 3) global node ids
    D_m_inv: np.ndarray          # (sum_n_tri, 2, 2)
    rest_area: np.ndarray        # (sum_n_tri,)
    mu: np.ndarray               # (sum_n_tri,)
    lam: np.ndarray              # (sum_n_tri,)
    mass: np.ndarray             # (sum_N,)
    node2env: np.ndarray         # (sum_N,)
    tri2env: np.ndarray          # (sum_n_tri,)
    boundary_nodes: np.ndarray   # (sum_n_b,) global
    boundary2env: np.ndarray     # (sum_n_b,)
    boundary_edges: np.ndarray   # (sum_n_e, 2) global node ids
    node_starts: np.ndarray      # (n_envs+1,)
    tri_starts: np.ndarray       # (n_envs+1,)
    boundary_starts: np.ndarray  # (n_envs+1,)
    boundary_per_env: np.ndarray # (n_envs,) n_b per env
    excl_flat: np.ndarray        # concatenated per-env (n_b_e^2,) uint8 masks
    excl_offset: np.ndarray      # (n_envs+1,) offsets into excl_flat
    excluded_pairs_per_env: list # list of (n_b_e, n_b_e) bool
    tri_quad: np.ndarray = None  # (sum_n_tri,) per-BODY-local grid-quad idx of each surviving tri (decode: ix=q//ny, iy=q%ny)
    # ---- per-BODY layout: an environment may hold several bodies ----
    body_env: np.ndarray = None           # (n_bodies,) environment of each body
    body_node_range: np.ndarray = None    # (n_bodies, 2) [start, stop) global node ids
    body_tri_range: np.ndarray = None     # (n_bodies, 2) [start, stop) global element ids
    body_grid: np.ndarray = None          # (n_bodies, 2) pixel grid (nx, ny) of each body
    grid_to_live_node_per_body: list = None

    @property
    def sum_N(self): return self.rest_pos.shape[0]
    @property
    def sum_n_tri(self): return self.tri.shape[0]

    def body_node_index(self, body, ix, iy):
        """Global id of the node at grid corner (ix, iy) of a body, or -1 if no solid
        pixel touches that corner."""
        if self.grid_to_live_node_per_body[body] is None:
            raise ValueError(f"body {body} is a mesh body: it has no pixel grid")
        ny = int(self.body_grid[body][1])
        loc = int(self.grid_to_live_node_per_body[body][(ny + 1) * ix + iy])
        return -1 if loc < 0 else int(self.body_node_range[body][0]) + loc


def _build_one(nx, ny, Lx, Ly, center, mask, k_excl):
    rest_pos, tri, tri_quad, remap = _grid_mesh(nx, ny, Lx, Ly, center, mask)
    out = _finish(rest_pos, tri, k_excl)
    out.update(grid_to_live=remap, tri_quad=tri_quad)
    return out


def _grid_mesh(nx, ny, Lx, Ly, center, mask):
    """Nodes and triangles of a pixel grid's solid pixels (all triangles clockwise), the
    grid quad of each triangle, and the grid-corner -> live-node map (-1: pruned)."""
    n_nodes = (nx + 1) * (ny + 1)
    n_quads = nx * ny
    xs = np.linspace(0.0, Lx, nx + 1) + (center[0] - 0.5 * Lx)
    ys = np.linspace(0.0, Ly, ny + 1) + (center[1] - 0.5 * Ly)
    gx = np.broadcast_to(xs[:, None], (nx + 1, ny + 1))
    gy = np.broadcast_to(ys[None, :], (nx + 1, ny + 1))
    rest_pos = np.stack([gx, gy], -1).reshape(-1, 2).astype(np.float64)

    ix = np.repeat(np.arange(nx), ny)          # quad idx = ix*ny + iy
    iy = np.tile(np.arange(ny), nx)
    bl = (ny + 1) * ix + iy; tl = bl + 1; br = bl + (ny + 1); tr = br + 1
    tri1 = np.stack([bl, tl, tr], 1); tri2 = np.stack([bl, tr, br], 1)
    tri = np.stack([tri1, tri2], 1).reshape(-1, 3)       # (2*n_quads, 3)
    quad_of_tri = np.repeat(np.arange(n_quads), 2)

    if mask is None:
        active = np.ones(n_quads, bool)
    else:
        mask = np.asarray(mask, bool)
        if mask.shape != (nx, ny):
            raise ValueError(f"mask must be ({nx},{ny}); got {mask.shape}")
        active = mask.reshape(-1)
    active_tri = active[quad_of_tri]
    tri = tri[active_tri]
    tri_quad = quad_of_tri[active_tri]           # per-env-local grid-quad idx of each surviving tri

    used = np.zeros(n_nodes, bool); used[tri.reshape(-1)] = True
    used_ids = np.nonzero(used)[0]
    remap = np.full(n_nodes, -1, np.int64)
    remap[used_ids] = np.arange(len(used_ids))
    tri = remap[tri]
    rest_pos = rest_pos[used_ids]
    return rest_pos, tri, tri_quad, remap


def _finish(rest_pos, tri, k_excl, excl_radius=0.0):
    """Everything about a body's mesh that does not depend on where it came from: rest
    shape matrices and areas, the boundary, and the self-contact exclusion — pairs of
    boundary nodes within k_excl hops of each other, and with excl_radius > 0 also those
    closer than that THROUGH THE MATERIAL (see _material_excl)."""
    p = rest_pos[tri]                                     # (n_tri,3,2)
    e1 = p[:, 1] - p[:, 0]; e2 = p[:, 2] - p[:, 0]
    D_m = np.stack([e1, e2], -1)                          # (n_tri,2,2) cols e1,e2
    D_m_inv = np.linalg.inv(D_m)
    rest_area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])

    # boundary edges = edges in exactly one triangle
    ec = {}
    for t in tri:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = (min(int(a), int(b)), max(int(a), int(b)))
            ec[k] = ec.get(k, 0) + 1
    bpairs = [k for k, v in ec.items() if v == 1]
    bset = sorted({n for e in bpairs for n in e})
    perim = np.array(bset, np.int64)

    # excluded self-collision pairs via mesh-graph BFS up to k_excl hops
    n_live = rest_pos.shape[0]
    adj = {i: set() for i in range(n_live)}
    for a, b, c in tri:
        a, b, c = int(a), int(b), int(c)
        adj[a] |= {b, c}; adj[b] |= {a, c}; adj[c] |= {a, b}
    nb = len(bset); idx_of = {n: k for k, n in enumerate(bset)}
    excl = np.eye(nb, dtype=bool) if nb else np.zeros((0, 0), bool)
    for src in bset:
        seen = {src: 0}; q = deque([src])
        while q:
            u = q.popleft(); d = seen[u]
            if d == k_excl:
                continue
            for w in adj[u]:
                if w not in seen:
                    seen[w] = d + 1; q.append(w)
        si = idx_of[src]
        for w in seen:
            if w in idx_of:
                excl[si, idx_of[w]] = True
    if nb and excl_radius > 0.0:
        excl |= _material_excl(rest_pos, tri, perim, excl_radius)
    if nb:
        excl = excl | excl.T
    bedges = np.array(bpairs, np.int64) if bpairs else np.zeros((0, 2), np.int64)
    return dict(rest_pos=rest_pos, tri=tri, D_m_inv=D_m_inv, rest_area=rest_area,
                boundary=perim, bedges=bedges, excl=excl, n_nodes=n_live)


def _material_excl(rest_pos, tri, boundary, radius):
    """Pairs of boundary nodes closer than `radius` through the material — by the shortest
    path along mesh edges at rest. Two such nodes are bonded by the body itself, and the
    barrier must not push them apart; a pair across a slit or a concave corner is further
    apart through the material than in a straight line, and stays in contact.

    A hop count is not enough on a mesh of uneven element sizes: two hops of small
    elements can be well inside dhat, and the barrier would then act within the body at
    rest. On a square grid with radius = 2 pixels this adds nothing to the 2-hop set — only
    single edges are shorter than two pixels."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    n = rest_pos.shape[0]
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e = np.unique(np.sort(e, 1), axis=0)
    w = np.linalg.norm(rest_pos[e[:, 0]] - rest_pos[e[:, 1]], axis=1)
    G = coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    nb = len(boundary)
    out = np.zeros((nb, nb), bool)
    chunk = max(1, int(2e7 // max(n, 1)))
    for s in range(0, nb, chunk):
        d = dijkstra(G, directed=False, indices=boundary[s:s + chunk], limit=radius)
        out[s:s + chunk] = d[:, boundary] < radius
    return out


# ---------------------------------------------------------------------- mesh bodies
def check_mesh(vertices, triangles):
    """Validate a triangle mesh for use as a body, and measure it. Raises ValueError,
    naming the offending vertices or triangles, for anything the simulation cannot take:
    malformed or non-finite arrays, out-of-range indices, a triangle that repeats a vertex,
    duplicate triangles, unused vertices, degenerate (zero-area) triangles, an edge shared
    by more than two triangles, a fold (two triangles on the same side of their common
    edge), coincident vertices or a vertex lying on a boundary edge it is not part of (a
    crack), and boundary segments that cross. Either winding is fine, mixed included.

    Returns dict(vertices (n,2) float64, triangles (m,3) int64, area (m,), h, h_cfl, L_max):
      h      median boundary-edge length — the body's contact length scale (the pixel size
             of a pixel body)
      h_cfl  smallest sqrt(2 x area) over the triangles — the size that sets how many solver
             iterations a step needs (a pixel triangle gives the pixel size)
      L_max  longest boundary edge
    """
    V = np.asarray(vertices, np.float64)
    if V.ndim != 2 or V.shape[1] != 2 or len(V) < 3:
        raise ValueError(f"vertices must be an (n, 2) array with n >= 3; got shape {V.shape}")
    if not np.isfinite(V).all():
        raise ValueError(f"vertices {np.nonzero(~np.isfinite(V).all(1))[0][:10].tolist()} are not finite")
    T = np.asarray(triangles)
    if T.ndim != 2 or T.shape[1] != 3 or len(T) == 0:
        raise ValueError(f"triangles must be an (m, 3) array with m >= 1; got shape {T.shape}")
    if not np.issubdtype(T.dtype, np.integer):
        if not (np.issubdtype(T.dtype, np.floating) and np.all(T == np.round(T))):
            raise ValueError(f"triangles must hold integer vertex indices; got dtype {T.dtype}")
    T = T.astype(np.int64)
    if T.min() < 0 or T.max() >= len(V):
        raise ValueError(f"triangle indices run from {T.min()} to {T.max()}, but there are {len(V)} "
                         f"vertices (indices are 0-based)")
    rep = (T[:, 0] == T[:, 1]) | (T[:, 1] == T[:, 2]) | (T[:, 2] == T[:, 0])
    if rep.any():
        raise ValueError(f"triangles {np.nonzero(rep)[0][:10].tolist()} repeat a vertex")
    _, first, cnt = np.unique(np.sort(T, 1), axis=0, return_index=True, return_counts=True)
    if (cnt > 1).any():
        raise ValueError(f"duplicate triangles (same three vertices), e.g. triangle {first[cnt > 1][0]}")
    used = np.zeros(len(V), bool); used[T.reshape(-1)] = True
    if not used.all():
        raise ValueError(f"vertices {np.nonzero(~used)[0][:10].tolist()} belong to no triangle; "
                         f"remove them (vertex i is node i of the body, so none may be left over)")

    p = V[T]
    e1, e2, e3 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 2] - p[:, 1]
    twoA = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    Lmax2 = np.max([(e1 ** 2).sum(1), (e2 ** 2).sum(1), (e3 ** 2).sum(1)], 0)
    q = twoA / Lmax2                                      # 0.87 equilateral, 0.5 a pixel triangle
    if (q < 1e-8).any():
        raise ValueError(f"triangles {np.nonzero(q < 1e-8)[0][:10].tolist()} have (almost) zero area")
    if (q < 0.05).any():
        warnings.warn(f"{int((q < 0.05).sum())} triangles are slivers (2 x area / longest edge^2 below "
                      f"0.05, worst {q.min():.3g}): they stiffen the solve and limit its step. "
                      f"Remesh with a quality constraint.", stacklevel=3)

    # edges: interior ones are shared by two triangles that must lie on opposite sides
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    opp = np.concatenate([T[:, 2], T[:, 0], T[:, 1]])
    Es = np.sort(E, 1)
    key, inv, cnt = np.unique(Es, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    if (cnt > 2).any():
        a, b = key[np.argmax(cnt > 2)]
        raise ValueError(f"edge ({a}, {b}) is shared by {cnt.max()} triangles; at most two may share one")
    side = lambda a, b, c: ((V[b, 0] - V[a, 0]) * (V[c, 1] - V[a, 1])
                            - (V[b, 1] - V[a, 1]) * (V[c, 0] - V[a, 0]))
    order = np.argsort(inv, kind="stable")
    inner = np.nonzero(cnt == 2)[0]
    starts = np.concatenate([[0], np.cumsum(cnt)])[inner]
    o1, o2 = opp[order[starts]], opp[order[starts + 1]]
    s1, s2 = side(key[inner, 0], key[inner, 1], o1), side(key[inner, 0], key[inner, 1], o2)
    if (s1 * s2 >= 0).any():
        a, b = key[inner[np.argmax(s1 * s2 >= 0)]]
        raise ValueError(f"the mesh folds over at edge ({a}, {b}): its two triangles lie on the same side of it")
    bed = key[cnt == 1]
    if len(bed) == 0:
        raise ValueError("the mesh has no boundary")
    L = np.linalg.norm(V[bed[:, 1]] - V[bed[:, 0]], axis=1)

    from scipy.spatial import cKDTree
    tol = 1e-9 * float(np.linalg.norm(np.ptp(V, 0)))
    pairs = cKDTree(V).query_pairs(tol, output_type="ndarray")
    if len(pairs):
        raise ValueError(f"vertices {pairs[0, 0]} and {pairs[0, 1]} are at the same place; merge them")
    bnodes = np.unique(bed)                                # a boundary vertex on another boundary edge: a crack
    mid = 0.5 * (V[bed[:, 0]] + V[bed[:, 1]])
    for k, cand in enumerate(cKDTree(V[bnodes]).query_ball_point(mid, 0.5 * L + tol)):
        cand = bnodes[np.asarray(cand, np.int64)]
        cand = cand[(cand != bed[k, 0]) & (cand != bed[k, 1])]
        if len(cand) and (_point_segment_distance(V[cand], V[bed[k, 0]], V[bed[k, 1]]) < tol).any():
            raise ValueError(f"vertex {cand[0]} lies on boundary edge ({bed[k, 0]}, {bed[k, 1]}) without being "
                             f"part of it (a crack or a T-junction); merge the two sides")
    i, j = _crossing_segments(V, bed, V, bed)
    if len(i):
        raise ValueError(f"boundary edges ({bed[i[0], 0]}, {bed[i[0], 1]}) and ({bed[j[0], 0]}, {bed[j[0], 1]}) "
                         f"cross: the outline intersects itself")
    return dict(vertices=V, triangles=T, area=0.5 * twoA, h=float(np.median(L)),
                h_cfl=float(np.sqrt(twoA).min()), L_max=float(L.max()))


def _build_tri(vertices, triangles, k_excl, excl_radius):
    """A body given as a triangle mesh: nodes and elements in the order given."""
    V = np.ascontiguousarray(vertices, np.float64)
    T = np.ascontiguousarray(triangles, np.int64)
    out = _finish(V, T, k_excl, excl_radius)
    out.update(grid_to_live=None, tri_quad=np.full(len(T), -1, np.int64))
    return out


def mask_to_mesh(mask, physical_size, center):
    """The triangle mesh add_soft_body makes of a pixel mask — same nodes, same triangles,
    in the same order — as (vertices (n,2), triangles (m,3), pixel (m,2)), where pixel[j] is
    the (row, col) of the mask pixel that triangle j belongs to: a per-pixel field `F` is
    F[pixel[:, 0], pixel[:, 1]] on the triangles. Rotate or bend the vertices before passing
    them to add_soft_mesh to place a pixel design at an angle."""
    m = (np.asarray(mask) > 0.5).T.copy()
    if m.ndim != 2 or not m.any():
        raise ValueError("the mask must be a 2-D array with at least one solid pixel")
    nx, ny = m.shape
    rest_pos, tri, tri_quad, _ = _grid_mesh(nx, ny, float(physical_size[0]), float(physical_size[1]),
                                            (float(center[0]), float(center[1])), m)
    return rest_pos, tri, np.stack([tri_quad % ny, tri_quad // ny], 1)


# ---------------------------------------------------------------------- geometry
def _point_segment_distance(P, a, b):
    t = b - a
    s = np.clip(((P - a) @ t) / max(float(t @ t), 1e-300), 0.0, 1.0)
    return np.linalg.norm(P - (a + s[:, None] * t), axis=1)


def _crossing_segments(Va, Ea, Vb, Eb):
    """Index pairs (i, j) of segments Ea[i] (of Va) and Eb[j] (of Vb) that cross properly —
    each strictly separates the other's two ends. Touching, overlapping collinear pieces and
    shared ends do not count. Segments sharing a vertex index are skipped when Va is Vb."""
    from scipy.spatial import cKDTree
    A0, A1, B0, B1 = Va[Ea[:, 0]], Va[Ea[:, 1]], Vb[Eb[:, 0]], Vb[Eb[:, 1]]
    La, Lb = np.linalg.norm(A1 - A0, axis=1), np.linalg.norm(B1 - B0, axis=1)
    near = cKDTree(0.5 * (B0 + B1)).query_ball_point(0.5 * (A0 + A1), 0.5 * (La + Lb.max()) * (1 + 1e-9))
    i = np.repeat(np.arange(len(Ea)), [len(c) for c in near])
    j = np.concatenate([np.asarray(c, np.int64) for c in near]) if len(i) else np.zeros(0, np.int64)
    if Va is Vb:
        keep = (i < j) & ~((Ea[i, 0] == Eb[j, 0]) | (Ea[i, 0] == Eb[j, 1]) |
                           (Ea[i, 1] == Eb[j, 0]) | (Ea[i, 1] == Eb[j, 1]))
        i, j = i[keep], j[keep]
    cr = lambda o, p, q: (p[:, 0] - o[:, 0]) * (q[:, 1] - o[:, 1]) - (p[:, 1] - o[:, 1]) * (q[:, 0] - o[:, 0])
    a0, a1, b0, b1 = A0[i], A1[i], B0[j], B1[j]
    hit = (cr(a0, a1, b0) * cr(a0, a1, b1) < 0) & (cr(b0, b1, a0) * cr(b0, b1, a1) < 0)
    return i[hit], j[hit]


def _triangle_pairs_overlap(P, Q, eps):
    """Do triangles P[k] and Q[k] ((K, 3, 2) each) share an area? Separating axes: two
    convex shapes overlap in area exactly when their projections on every edge normal of
    either overlap by more than a point — so a shared edge or vertex, or one triangle's
    vertex lying on the other's edge, is touching, not overlap."""
    ok = np.ones(len(P), bool)
    for S in (P, Q):
        for e in range(3):
            d = S[:, (e + 1) % 3] - S[:, e]
            n = np.stack([-d[:, 1], d[:, 0]], 1) / np.linalg.norm(d, axis=1)[:, None]
            pp, qq = np.einsum("kij,kj->ki", P, n), np.einsum("kij,kj->ki", Q, n)
            ok &= np.minimum(pp.max(1), qq.max(1)) - np.maximum(pp.min(1), qq.min(1)) > eps
    return ok


def meshes_overlap(Va, Ta, Vb, Tb):
    """Do two bodies, given as meshes, overlap — share an area
    of any size? Exact for any alignment of the two meshes, by testing every pair of
    triangles near each other; bodies that only touch along a face or at a point do not
    overlap here (contact refuses those at the first step)."""
    from scipy.spatial import cKDTree
    lo_a, hi_a, lo_b, hi_b = Va.min(0), Va.max(0), Vb.min(0), Vb.max(0)
    if (hi_a <= lo_b).any() or (hi_b <= lo_a).any():
        return False
    pa, pb = Va[Ta], Vb[Tb]
    ca, cb = pa.mean(1), pb.mean(1)
    ra = np.linalg.norm(pa - ca[:, None], axis=2).max(1)
    rb = np.linalg.norm(pb - cb[:, None], axis=2).max(1)
    eps = 1e-9 * float(np.linalg.norm(np.maximum(hi_a, hi_b) - np.minimum(lo_a, lo_b)))
    near = cKDTree(cb).query_ball_point(ca, (ra + rb.max()) * (1 + 1e-9))
    i = np.repeat(np.arange(len(Ta)), [len(n) for n in near])
    if not len(i):
        return False
    j = np.concatenate([np.asarray(n, np.int64) for n in near])
    for s in range(0, len(i), 200000):                       # in chunks: memory stays bounded
        if _triangle_pairs_overlap(pa[i[s:s + 200000]], pb[j[s:s + 200000]], eps).any():
            return True
    return False


def build_scene(bodies, k_excl=2):
    """Assemble a scene in which every body has its own grid, size, position and material,
    and an environment may hold several bodies.

    bodies: list of dicts with EITHER a pixel grid
        mask_xy  (nx, ny) bool, [ix, iy] with iy growing upward
        size     (Lx, Ly) physical size of the pixel grid [m]
        center   (cx, cy) of the grid
    OR a triangle mesh
        vertices     (n, 2) node positions [m]
        triangles    (m, 3) node indices
        excl_radius  self-contact exclusion through the material [m] (see _material_excl;
                     0 for the hop count alone). Checked with check_mesh unless the dict
                     says checked=True.
    and
        young, nu, rho
        env      index of the environment the body lives in

    Bodies of the same environment share its contact structures: their boundary nodes are
    contiguous, and the mesh-adjacency exclusion (which stops neighbouring nodes of ONE body
    from repelling each other) is block-diagonal per body — so two different bodies of an
    environment do collide with one another, and bodies of different environments never do.

    Body h keeps the index it was given in; its nodes are body_node_range[h]."""
    nb = len(bodies)
    envs = np.array([int(b["env"]) for b in bodies])
    n_env = int(envs.max()) + 1
    if set(envs.tolist()) != set(range(n_env)):
        raise ValueError(f"environment ids must be 0..{n_env - 1} with none skipped; got {sorted(set(envs.tolist()))}")
    order = np.argsort(envs, kind="stable")                  # layout order: by environment
    per, built = {}, {}                                      # a batch repeats the same body: build it once
    for h in order:
        b = bodies[h]
        if "mask_xy" in b:
            m = np.asarray(b["mask_xy"], bool)
            key = (m.shape, m.tobytes(), tuple(b["size"]), tuple(b["center"]))
            if key not in built:
                built[key] = _build_one(m.shape[0], m.shape[1], b["size"][0], b["size"][1], b["center"], m, k_excl)
        else:
            V = np.ascontiguousarray(b["vertices"], np.float64)
            T = np.ascontiguousarray(b["triangles"], np.int64)
            r = float(b.get("excl_radius", 0.0))
            key = ("mesh", V.shape, V.tobytes(), T.shape, T.tobytes(), r)
            if key not in built:
                if not b.get("checked"):
                    check_mesh(V, T)
                built[key] = _build_tri(V, T, k_excl, r)
        per[h] = built[key]

    node_rng, tri_rng = np.zeros((nb, 2), np.int64), np.zeros((nb, 2), np.int64)
    n0 = t0 = 0
    for h in order:
        node_rng[h] = (n0, n0 + per[h]["n_nodes"]); n0 = node_rng[h][1]
        tri_rng[h] = (t0, t0 + len(per[h]["rest_area"])); t0 = tri_rng[h][1]

    cat = lambda key: np.concatenate([per[h][key] for h in order], 0)
    rest_pos, D_m_inv, rest_area, tri_quad = cat("rest_pos"), cat("D_m_inv"), cat("rest_area"), cat("tri_quad")
    tri = np.concatenate([per[h]["tri"] + node_rng[h][0] for h in order], 0)
    boundary = np.concatenate([per[h]["boundary"] + node_rng[h][0] for h in order], 0)
    be = [per[h]["bedges"] + node_rng[h][0] for h in order if len(per[h]["bedges"])]
    boundary_edges = np.concatenate(be, 0) if be else np.zeros((0, 2), np.int64)
    rep = lambda count: np.concatenate([np.full(count(h), envs[h]) for h in order])
    node2env = rep(lambda h: per[h]["n_nodes"])
    tri2env = rep(lambda h: len(per[h]["rest_area"]))
    boundary2env = rep(lambda h: len(per[h]["boundary"]))

    cz = lambda a: np.concatenate([[0], np.cumsum(a)])
    members = [[] for _ in range(n_env)]
    for h in order:
        members[envs[h]].append(h)
    b_pe = np.array([sum(len(per[h]["boundary"]) for h in mem) for mem in members])
    excls = []
    for mem in members:                                      # block-diagonal exclusion per environment
        ex = np.zeros((b_pe[len(excls)],) * 2, bool); o = 0
        for h in mem:
            k = len(per[h]["boundary"]); ex[o:o + k, o:o + k] = per[h]["excl"]; o += k
        excls.append(ex)
    excl_flat = (np.concatenate([e.reshape(-1).astype(np.uint8) for e in excls])
                 if excls else np.zeros(0, np.uint8))
    excl_offset = cz([e.size for e in excls]).astype(np.int64)

    mu, lam, mass = np.zeros(len(rest_area)), np.zeros(len(rest_area)), np.zeros(rest_pos.shape[0])
    for h in order:
        b = bodies[h]; ts = slice(*tri_rng[h])
        mu[ts], lam[ts] = lame(b["young"], b["nu"])
        np.add.at(mass, tri[ts].reshape(-1), np.repeat(b["rho"] * rest_area[ts] / 3.0, 3))
    mass = np.clip(mass, 1e-30, None)

    grid = np.array([np.asarray(bodies[h]["mask_xy"]).shape if "mask_xy" in bodies[h] else (0, 0)
                     for h in range(nb)])
    return Mesh(n_envs=n_env, rest_pos=rest_pos, tri=tri,
                D_m_inv=D_m_inv, rest_area=rest_area, mu=mu, lam=lam, mass=mass,
                node2env=node2env, tri2env=tri2env, boundary_nodes=boundary,
                boundary2env=boundary2env, boundary_edges=boundary_edges,
                node_starts=cz([sum(per[h]["n_nodes"] for h in mem) for mem in members]),
                tri_starts=cz([sum(len(per[h]["rest_area"]) for h in mem) for mem in members]),
                boundary_starts=cz(b_pe), boundary_per_env=b_pe,
                excl_flat=excl_flat, excl_offset=excl_offset, excluded_pairs_per_env=excls,
                tri_quad=tri_quad, body_env=envs, body_node_range=node_rng, body_tri_range=tri_rng,
                body_grid=grid, grid_to_live_node_per_body=[per[h]["grid_to_live"] for h in range(nb)])
