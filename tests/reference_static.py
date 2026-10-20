"""Reference solution of the DISCRETE static problem, in double precision.

Static equilibrium of the same P1 mesh under the same Neo-Hookean energy the solver uses,
found by Newton's method with a sparse direct solve. It shares no code with the GPU
kernels, so agreement is a real check; and because it converges to machine precision, it
separates two things a comparison with continuum theory cannot:

    discretisation error   what P1 triangles on this mesh can represent
    solver error           how far the PNCG iteration actually gets towards that

Supports elasticity, a dead load, a tension-controlled tendon, and a changed rest shape
(active material). Small problems only — it assembles and factorises the stiffness matrix.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla

def _F(x, tri, Bm):
    Ds = np.stack([x[tri[:, 1]] - x[tri[:, 0]], x[tri[:, 2]] - x[tri[:, 0]]], -1)       # columns
    return Ds @ Bm

def elastic(x, tri, Bm, area, mu, lam, hess=True):
    F = _F(x, tri, Bm); J = np.linalg.det(F)
    if (J <= 0).any(): return np.inf, None, None
    lj = np.log(J); Fi = np.linalg.inv(F); FiT = Fi.transpose(0, 2, 1)
    E = (area * (0.5 * mu * ((F * F).sum((1, 2)) - 2.0) - mu * lj + 0.5 * lam * lj * lj)).sum()
    P = mu[:, None, None] * (F - FiT) + (lam * lj)[:, None, None] * FiT
    Hm = area[:, None, None] * (P @ Bm.transpose(0, 2, 1))                               # cols: nodes 1, 2
    g = np.zeros_like(x); g1, g2 = Hm[:, :, 0], Hm[:, :, 1]
    np.add.at(g, tri[:, 1], g1); np.add.at(g, tri[:, 2], g2); np.add.at(g, tri[:, 0], -(g1 + g2))
    if not hess: return E, g, None
    n = len(tri); Ke = np.zeros((n, 6, 6))
    for a in range(3):
        for c in range(2):
            dDs = np.zeros((n, 2, 2))
            if a == 0: dDs[:, c, :] = -1.0
            else: dDs[:, c, a - 1] = 1.0
            dF = dDs @ Bm
            dP = (mu[:, None, None] * dF + (mu - lam * lj)[:, None, None] * (FiT @ dF.transpose(0, 2, 1) @ FiT)
                  + (lam * np.einsum('nij,nji->n', Fi, dF))[:, None, None] * FiT)
            dH = area[:, None, None] * (dP @ Bm.transpose(0, 2, 1))
            col = np.zeros((n, 6)); col[:, 2:4] = dH[:, :, 0]; col[:, 4:6] = dH[:, :, 1]; col[:, 0:2] = -(dH[:, :, 0] + dH[:, :, 1])
            Ke[:, :, 2 * a + c] = col
    dof = (2 * tri[:, :, None] + np.arange(2)[None, None, :]).reshape(n, 6)
    K = sp.coo_matrix((Ke.ravel(), (np.repeat(dof, 6, 1).ravel(), np.tile(dof, (1, 6)).ravel())),
                      shape=(x.size, x.size)).tocsr()
    return E, g, K

def tendon(x, nodes, T):
    a, b = nodes[:-1], nodes[1:]; t = x[b] - x[a]; ln = np.linalg.norm(t, axis=1); d = t / ln[:, None]
    g = np.zeros_like(x); np.add.at(g, b, T * d); np.add.at(g, a, -T * d)
    I, Jc, V = [], [], []
    for k in range(len(a)):
        blk = T / ln[k] * (np.eye(2) - np.outer(d[k], d[k]))
        for (p, q, sg) in ((a[k], a[k], 1), (b[k], b[k], 1), (a[k], b[k], -1), (b[k], a[k], -1)):
            for i in range(2):
                for j in range(2):
                    I.append(2 * p + i); Jc.append(2 * q + j); V.append(sg * blk[i, j])
    return T * ln.sum(), g, sp.coo_matrix((V, (I, Jc)), shape=(x.size, x.size)).tocsr()

def solve_static(me, fixed, Bm=None, area=None, tendon_nodes=None, T=0.0, ramps=6, tol=1e-11,
                 fext=None):
    x = me.rest_pos.astype(np.float64).copy(); tri = np.asarray(me.tri)
    Bm = np.asarray(me.D_m_inv, np.float64) if Bm is None else Bm
    area = np.asarray(me.rest_area, np.float64) if area is None else area
    mu, lam = np.asarray(me.mu, np.float64), np.asarray(me.lam, np.float64)
    free = np.ones(x.size, bool); free[2 * np.asarray(fixed)] = False; free[2 * np.asarray(fixed) + 1] = False
    for r in range(1, ramps + 1):
        s = r / ramps
        Bm_r = (1 - s) * np.asarray(me.D_m_inv, np.float64) + s * Bm        # ramp the rest-shape change too
        ar_r = (1 - s) * np.asarray(me.rest_area, np.float64) + s * area
        def total(xx, hess=True):
            E, g, K = elastic(xx, tri, Bm_r, ar_r, mu, lam, hess)
            if not np.isfinite(E): return np.inf, None, None
            if tendon_nodes is not None:
                Et, gt, Kt = tendon(xx, tendon_nodes, T * s); E += Et; g = g + gt
                if hess: K = K + Kt
            if fext is not None:                                  # dead load, (N, 2)
                E -= s * (fext * xx).sum(); g = g - s * fext
            return E, g, K
        for it in range(60):
            E, g, K = total(x); r_ = g.ravel()[free]
            if np.abs(r_).max() < tol * max(1.0, np.abs(mu).max() * 1e-3): break
            dx = np.zeros(x.size); dx[free] = spla.spsolve(K[free][:, free].tocsc(), -r_)
            a = 1.0
            while a > 1e-8 and total(x + a * dx.reshape(-1, 2), hess=False)[0] > E: a *= 0.5
            x = x + a * dx.reshape(-1, 2)
    return x, float(np.abs(g.ravel()[free]).max())
