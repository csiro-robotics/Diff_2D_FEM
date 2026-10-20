"""Arbitrary collider geometry via a signed-distance / level-set function phi(x).
Siblings of the disc kernels (implicit/kernels.py). The general barrier contact reuses
the disc algebra with {d, nhat, 1/dc*(I-nn^T)} -> {phi, grad phi, hess phi}:
    force  = h2*kappa*bp(phi)*g            (g = grad phi)
    H*p    = kappa*(bpp*(g.p)g + bp*(Hphi p));  diag/pHp read off the same block.
DISC (kind=0) is the exact fast path; ELLIPSE (kind=1) is the algebraic level-set
u=((x-c)/r)^2, phi=rref*(sqrt(u)-1), which contains the disc as rx=ry=rref=R.
CCD is a conservative bound s <= eta*phi/(-g.p) — valid for CONVEX phi (phi stays above
its linearization, so the linear TOI lower-bounds the true one)."""
import warp as wp
from .precision import real, vec2, vec3, mat22, np_real
from .math import barrier_b, barrier_bp, barrier_bpp

f64 = wp.float64
DISC = wp.constant(0)
ELLIPSE = wp.constant(1)
BOX = wp.constant(2)          # rounded box; rref = corner radius (square when rx == ry)
TRIANGLE = wp.constant(3)     # rounded equilateral triangle, apex +y; rx = circumradius
HALFSPACE = wp.constant(4)    # a wall / the ground: (cx, cy) a point on it, (rx, ry) its unit normal


@wp.func
def _seg_closest(p: vec2, a: vec2, b: vec2):
    """(distance, closest point, is_endpoint) from p to segment ab. `is_endpoint` tells the
    caller whether the nearest feature is a VERTEX (rounded corner -> curvature) or the
    interior of an EDGE (flat face -> zero curvature)."""
    ab = b - a
    t = wp.dot(p - a, ab) / wp.max(wp.dot(ab, ab), real(1.0e-30))
    tc = wp.min(wp.max(t, real(0.0)), real(1.0))
    q = a + tc * ab
    isv = real(0.0)
    if t <= real(0.0) or t >= real(1.0):
        isv = real(1.0)
    return wp.length(p - q), q, isv


@wp.struct
class SdfHit:
    d: real          # clearance phi(x)
    g: vec2        # grad phi
    H: mat22       # hess phi


@wp.func
def sdf_eval(kind: int, cx: real, cy: real, rx: real, ry: real, rref: real, x: vec2) -> SdfHit:
    h = SdfHit()
    if kind == 0:                                          # exact disc; rx = radius
        diff = x - vec2(cx, cy); dc = wp.length(diff); nh = diff / dc
        h.d = dc - rx; h.g = nh
        h.H = (mat22(real(1.0), real(0.0), real(0.0), real(1.0)) - wp.outer(nh, nh)) * (real(1.0) / dc)
    elif kind == 1:                                        # ellipse level-set (rx=ry -> disc)
        ux = (x[0] - cx) / rx; uy = (x[1] - cy) / ry
        u = ux * ux + uy * uy; su = wp.sqrt(wp.max(u, real(1.0e-18)))
        w = vec2((x[0] - cx) / (rx * rx), (x[1] - cy) / (ry * ry))
        h.d = rref * (su - real(1.0)); h.g = (rref / su) * w
        D = mat22(real(1.0) / (rx * rx), real(0.0), real(0.0), real(1.0) / (ry * ry))
        h.H = (rref / su) * (D - (real(1.0) / u) * wp.outer(w, w))
    elif kind == 4:                                        # HALF-SPACE: phi = n . (x - c), flat
        n = vec2(rx, ry)
        h.d = wp.dot(n, x - vec2(cx, cy)); h.g = n
        h.H = mat22(real(0.0), real(0.0), real(0.0), real(0.0))
    else:                                                  # kind == 2: ROUNDED BOX (square: rx == ry)
        # Exact box SDF, |grad phi| == 1 everywhere outside. Fold to the +,+ quadrant with
        # S = diag(sx,sy) (an isometry), evaluate there, then map back: g = S g_f, H = S H_f S.
        #   q   = |x - c| - (half-extent - r)
        #   phi = |max(q,0)| + min(max(qx,qy), 0) - r
        # Curvature is ZERO on the flat faces and on the interior, and equals the rounding
        # circle's (I - n n^T)/|m| only in the CORNER region where both q components are > 0.
        # r = rref clamped to the half-extents; r -> 0 gives a sharp square, whose corner
        # curvature is genuinely unbounded, so keep r > 0 for a well-conditioned barrier.
        r = wp.min(wp.max(rref, real(0.0)), wp.min(rx, ry))
        sx = real(1.0)
        if x[0] < cx:
            sx = -real(1.0)
        sy = real(1.0)
        if x[1] < cy:
            sy = -real(1.0)
        qx = sx * (x[0] - cx) - (rx - r)                    # = |x-cx| - (rx - r)
        qy = sy * (x[1] - cy) - (ry - r)
        mx = wp.max(qx, real(0.0)); my = wp.max(qy, real(0.0))
        ml = wp.sqrt(mx * mx + my * my)
        h.d = ml + wp.min(wp.max(qx, qy), real(0.0)) - r
        if ml > real(1.0e-12):                                   # face or corner region
            gx = mx / ml; gy = my / ml
            h.g = vec2(sx * gx, sy * gy)
            if qx > real(0.0) and qy > real(0.0):                      # CORNER: rounding-circle curvature
                off = -gx * gy / ml
                h.H = mat22(gy * gy / ml, sx * sy * off, sx * sy * off, gx * gx / ml)
            else:                                          # FACE: flat -> no curvature
                h.H = mat22(real(0.0), real(0.0), real(0.0), real(0.0))
        else:                                              # inside the inner box: phi = max(q) - r
            if qx > qy:
                h.g = vec2(sx, real(0.0))
            else:
                h.g = vec2(real(0.0), sy)
            h.H = mat22(real(0.0), real(0.0), real(0.0), real(0.0))
    if kind == 3:                                          # ROUNDED EQUILATERAL TRIANGLE
        # General rounded-convex-polygon construction (the box above is the 4-gon special
        # case, written out for speed): offset an INNER polygon outward by r.
        #     phi   = sgn * dist(x, inner_poly) - r          (|grad phi| == 1, exact SDF)
        #     grad  = sgn * (x - q)/|x - q|                   q = closest point on inner poly
        #     hess  = (I - g g^T)/|x - q|  if q is a VERTEX  (rounding arc, like a circle)
        #           = 0                    if q is on an EDGE (flat face has no curvature)
        # rx = circumradius of the ROUNDED shape, so the inner triangle uses Rin = rx - r.
        r3 = wp.min(wp.max(rref, real(0.0)), rx * real(0.5))
        rin = rx - r3
        c0 = vec2(cx, cy)
        a0 = real(1.5707963267948966)                            # apex at +y
        v0 = c0 + rin * vec2(wp.cos(a0), wp.sin(a0))
        v1 = c0 + rin * vec2(wp.cos(a0 + real(2.0943951023931953)), wp.sin(a0 + real(2.0943951023931953)))
        v2 = c0 + rin * vec2(wp.cos(a0 + real(4.1887902047863905)), wp.sin(a0 + real(4.1887902047863905)))
        d0, q0, s0 = _seg_closest(x, v0, v1)
        d1, q1, s1 = _seg_closest(x, v1, v2)
        d2, q2, s2 = _seg_closest(x, v2, v0)
        dm = d0; qm = q0; sm = s0
        if d1 < dm:
            dm = d1; qm = q1; sm = s1
        if d2 < dm:
            dm = d2; qm = q2; sm = s2
        # convex inside-test: CCW winding -> all cross products non-negative
        e0 = v1 - v0; e1 = v2 - v1; e2 = v0 - v2
        p0 = x - v0; p1 = x - v1; p2 = x - v2
        k0 = e0[0] * p0[1] - e0[1] * p0[0]
        k1 = e1[0] * p1[1] - e1[1] * p1[0]
        k2 = e2[0] * p2[1] - e2[1] * p2[0]
        sgn = real(1.0)
        if k0 >= real(0.0) and k1 >= real(0.0) and k2 >= real(0.0):
            sgn = -real(1.0)
        dl = wp.max(dm, real(1.0e-30))
        h.d = sgn * dm - r3
        h.g = (sgn / dl) * (x - qm)
        if sm > real(0.5) and sgn > real(0.0):                         # rounded corner, outside
            gg = h.g
            h.H = (mat22(real(1.0), real(0.0), real(0.0), real(1.0)) - wp.outer(gg, gg)) * (real(1.0) / dl)
        else:                                              # flat face, or interior
            h.H = mat22(real(0.0), real(0.0), real(0.0), real(0.0))
    return h


@wp.kernel
def grad_diag_sdf_kernel(x: wp.array(dtype=vec2), bnd: wp.array(dtype=wp.int32),
                         kind: int, cx: real, cy: real, rx: real, ry: real, rref: real,
                         dhat: real, kappa: real, h2: real, clamp: int,
                         grad: wp.array(dtype=vec2), diagH: wp.array(dtype=vec2)):
    b = wp.tid(); n = bnd[b]; s = sdf_eval(kind, cx, cy, rx, ry, rref, x[n])
    if s.d <= real(0.0) or s.d >= dhat:
        return
    bp = barrier_bp(s.d, dhat); bpp = barrier_bpp(s.d, dhat)
    wp.atomic_add(grad, n, h2 * kappa * bp * s.g)
    dgx = kappa * (bpp * s.g[0] * s.g[0] + bp * s.H[0, 0])
    dgy = kappa * (bpp * s.g[1] * s.g[1] + bp * s.H[1, 1])
    if clamp != 0:
        dgx = wp.max(dgx, real(0.0)); dgy = wp.max(dgy, real(0.0))
    wp.atomic_add(diagH, n, vec2(h2 * dgx, h2 * dgy))


@wp.kernel
def pHp_sdf_kernel(x: wp.array(dtype=vec2), p: wp.array(dtype=vec2),
                   bnd: wp.array(dtype=wp.int32), b2e: wp.array(dtype=wp.int32),
                   kind: int, cx: real, cy: real, rx: real, ry: real, rref: real,
                   dhat: real, kappa: real, h2: real, clamp: int,
                   pHp: wp.array(dtype=f64), clr_min: wp.array(dtype=f64)):
    b = wp.tid(); n = bnd[b]; e = b2e[b]; s = sdf_eval(kind, cx, cy, rx, ry, rref, x[n])
    wp.atomic_min(clr_min, e, f64(s.d))
    if s.d <= real(0.0) or s.d >= dhat:
        return
    pp = p[n]; gp = wp.dot(s.g, pp)
    val = kappa * (barrier_bpp(s.d, dhat) * gp * gp + barrier_bp(s.d, dhat) * wp.dot(pp, s.H @ pp))
    if clamp != 0:
        val = wp.max(val, real(0.0))
    wp.atomic_add(pHp, e, f64(h2 * val))


@wp.kernel
def energy_sdf_kernel(x: wp.array(dtype=vec2), bnd: wp.array(dtype=wp.int32), b2e: wp.array(dtype=wp.int32),
                      kind: int, cx: real, cy: real, rx: real, ry: real, rref: real,
                      dhat: real, kappa: real, h2: real, E: wp.array(dtype=f64)):
    b = wp.tid(); s = sdf_eval(kind, cx, cy, rx, ry, rref, x[bnd[b]])
    if s.d <= real(0.0) or s.d >= dhat:
        return
    wp.atomic_add(E, b2e[b], f64(h2 * kappa * barrier_b(s.d, dhat)))


@wp.kernel
def sdf_force_kernel(x: wp.array(dtype=vec2), bnd: wp.array(dtype=wp.int32), b2e: wp.array(dtype=wp.int32),
                    kind: int, cx: real, cy: real, rx: real, ry: real, rref: real,
                    dhat: real, kappa: real, F: wp.array(dtype=vec2)):
    b = wp.tid(); s = sdf_eval(kind, cx, cy, rx, ry, rref, x[bnd[b]])
    if s.d <= real(0.0) or s.d >= dhat:
        return
    wp.atomic_add(F, b2e[b], kappa * barrier_bp(s.d, dhat) * s.g)   # reaction ON object


# ---------------------------------------------------------------- friction (SDF siblings)
# The disc friction kernels in implicit/kernels.py assume |grad phi| == 1 (true for a circle, where
# grad phi = nhat). For a general level-set that is FALSE -- the ellipse's phi is algebraic,
# not a distance -- so the normal force magnitude carries the extra |g| factor:
#     F_barrier = kappa*b'(phi)*g   =>   f_n = kappa*|b'(phi)| * |g| ,   nhat = g/|g|
# Setting |g| = 1 recovers the disc kernels exactly, which is the regression check.
@wp.func
def _sdf_fric_frame(s: SdfHit, xk: vec2, xkm1: vec2, kappa: real, dhat: real,
                    inv_dt: real):
    """(f_n, nhat, tdir, |vt|) at a contacting node. Caller must gate 0 < phi < dhat."""
    gn = wp.length(s.g)
    nh = s.g / wp.max(gn, real(1.0e-30))
    f_n = kappa * wp.abs(barrier_bp(s.d, dhat)) * gn
    vel = (xk - xkm1) * inv_dt
    vt = vel - wp.dot(vel, nh) * nh
    vtm = wp.length(vt)
    return f_n, nh, vt, vtm


@wp.kernel
def grad_friction_sdf_kernel(x: wp.array(dtype=vec2), xn: wp.array(dtype=vec2),
                             bnd: wp.array(dtype=wp.int32),
                             kind: int, cx: real, cy: real, rx: real, ry: real, rref: real,
                             dhat: real, kappa: real, mu: real, epsv: real,
                             inv_dt: real, h2: real, free: wp.array(dtype=vec2),
                             grad: wp.array(dtype=vec2), diagH: wp.array(dtype=vec2)):
    """SDF sibling of kernels.grad_friction_disc_kernel, including the kink curvature into
    diagH (see that kernel for why omitting it makes the tangential DOF bang-bang)."""
    b = wp.tid(); n = bnd[b]; fr = free[n]
    if fr[0] <= real(0.5) and fr[1] <= real(0.5):
        return
    s = sdf_eval(kind, cx, cy, rx, ry, rref, x[n])
    if s.d <= real(0.0) or s.d >= dhat:
        return
    f_n, nh, vt, vtm = _sdf_fric_frame(s, x[n], xn[n], kappa, dhat, inv_dt)
    tdir = vt / wp.max(vtm, epsv)
    wp.atomic_add(grad, n, wp.cw_mul(h2 * mu * f_n * tdir, fr))
    kt = h2 * mu * f_n * inv_dt / wp.max(vtm, epsv)
    th = vec2(-nh[1], nh[0])
    wp.atomic_add(diagH, n, wp.cw_mul(vec2(kt * th[0] * th[0], kt * th[1] * th[1]), fr))


@wp.kernel
def pHp_friction_sdf_kernel(x: wp.array(dtype=vec2), xn: wp.array(dtype=vec2),
                            bnd: wp.array(dtype=wp.int32), b2e: wp.array(dtype=wp.int32),
                            p: wp.array(dtype=vec2),
                            kind: int, cx: real, cy: real, rx: real, ry: real, rref: real,
                            dhat: real, kappa: real, mu: real, epsv: real,
                            inv_dt: real, h2: real, free: wp.array(dtype=vec2),
                            pHp: wp.array(dtype=f64)):
    """Keeps alpha = -gTp/pHp self-consistent when friction is on with an SDF object."""
    b = wp.tid(); n = bnd[b]; fr = free[n]
    if fr[0] <= real(0.5) and fr[1] <= real(0.5):
        return
    s = sdf_eval(kind, cx, cy, rx, ry, rref, x[n])
    if s.d <= real(0.0) or s.d >= dhat:
        return
    f_n, nh, vt, vtm = _sdf_fric_frame(s, x[n], xn[n], kappa, dhat, inv_dt)
    kt = h2 * mu * f_n * inv_dt / wp.max(vtm, epsv)
    th = vec2(-nh[1], nh[0])
    pt = wp.dot(wp.cw_mul(p[n], fr), th)
    wp.atomic_add(pHp, b2e[b], f64(kt * pt * pt))


@wp.kernel
def sdf_friction_force_kernel(x: wp.array(dtype=vec2), xn: wp.array(dtype=vec2),
                              bnd: wp.array(dtype=wp.int32), b2e: wp.array(dtype=wp.int32),
                              kind: int, cx: real, cy: real, rx: real, ry: real, rref: real,
                              dhat: real, kappa: real, mu: real, epsv: real,
                              inv_dt: real, F: wp.array(dtype=vec2)):
    """Friction reaction ON the object (sibling of kernels.disc_friction_force_kernel)."""
    b = wp.tid(); n = bnd[b]
    s = sdf_eval(kind, cx, cy, rx, ry, rref, x[n])
    if s.d <= real(0.0) or s.d >= dhat:
        return
    f_n, nh, vt, vtm = _sdf_fric_frame(s, x[n], xn[n], kappa, dhat, inv_dt)
    tdir = vt / wp.max(vtm, epsv)
    wp.atomic_add(F, b2e[b], -(mu * f_n) * tdir)          # opposes the body's slide


@wp.kernel
def energy_friction_sdf_kernel(x: wp.array(dtype=vec2), xn: wp.array(dtype=vec2),
                               bnd: wp.array(dtype=wp.int32), b2e: wp.array(dtype=wp.int32),
                               kind: int, cx: real, cy: real, rx: real, ry: real, rref: real,
                               dhat: real, kappa: real, mu: real, epsv: real,
                               inv_dt: real, h2: real, free: wp.array(dtype=vec2),
                               E: wp.array(dtype=f64)):
    """Lagged friction dissipation potential (line-search parity with the disc path)."""
    b = wp.tid(); n = bnd[b]; fr = free[n]
    if fr[0] <= real(0.5) and fr[1] <= real(0.5):
        return
    s = sdf_eval(kind, cx, cy, rx, ry, rref, x[n])
    if s.d <= real(0.0) or s.d >= dhat:
        return
    f_n, nh, vt, vtm = _sdf_fric_frame(s, x[n], xn[n], kappa, dhat, inv_dt)
    ut = vtm / inv_dt
    a = epsv / inv_dt
    f0 = ut - a * real(0.5)
    if ut < a:
        f0 = ut * ut / (real(2.0) * a)
    wp.atomic_add(E, b2e[b], f64(h2 * mu * f_n * f0))


@wp.kernel
def sdf_ccd_kernel(x: wp.array(dtype=vec2), p: wp.array(dtype=vec2),
                   bnd: wp.array(dtype=wp.int32), b2e: wp.array(dtype=wp.int32),
                   kind: int, cx: real, cy: real, rx: real, ry: real, rref: real,
                   eta: real, alpha: wp.array(dtype=f64)):
    b = wp.tid(); n = bnd[b]; s = sdf_eval(kind, cx, cy, rx, ry, rref, x[n])
    gp = wp.dot(s.g, p[n])                                 # d(phi)/ds along the search dir
    if gp >= real(0.0) or s.d <= real(0.0):                            # moving away / already inside -> no cap
        return
    wp.atomic_min(alpha, b2e[b], f64(eta * s.d / (-gp)))   # conservative TOI (convex phi)
