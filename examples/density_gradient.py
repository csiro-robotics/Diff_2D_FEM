"""Differentiating through the implicit solve — dJ/d(density) by the adjoint.

A soft block is pressed and sheared onto a disc over T implicit steps. Each element carries
a density rho that scales its stiffness, E(rho) = Emin + rho (1 - Emin). The objective is
how far the lower half of the block has moved sideways at the end. One adjoint solve per
step gives the sensitivity of that to every element's density; a finite difference along
the gradient checks it.

Runs on the CPU device on purpose: GPU atomics are not order-deterministic, so two
identical GPU rollouts differ by about as much as a finite-difference perturbation does.
The gradient itself is the same on either device.

Friction is off: the adjoint operator has no friction tangent (see the README).

    python examples/density_gradient.py
"""
import numpy as np
import warp as wp
import Diff_2D_FEM as D

wp.init()
T, DT = 12, 2.0e-4

model = D.Model(device="cpu")
model.add_disc(center=(0.05, 0.0335), radius=0.006, mu=0.0)
h = model.add_soft_body(np.ones((10, 6), np.uint8), physical_size=(0.012, 0.024),
                        center=(0.05, 0.052), E=1.0e6, nu=0.40, density=1100.0)
model.finalize()
x0 = model.get_body_x(h)
top = np.where(x0[:, 1] > x0[:, 1].max() - 1e-5)[0]
model.set_kinematic_constraint(h, top)

# tol_x is the convergence test to use before differentiating; 1e-8 m is about as tight as
# float32 positions allow (their spacing near 0.05 m is ~4e-9 m)
integ = D.ImplicitIntegrator(max_iter=3000, eps=1e-14, tol_x=1e-8, conv_check=8,
                             self_collision=False, friction=False)
adj = integ.adjoint(model, DT)                       # builds the solver; density can now be set

rng = np.random.default_rng(0)
rho = 0.45 + 0.5 * rng.random(adj.n_tri)             # a non-uniform density field

lower = np.where(x0[:, 1] < np.median(x0[:, 1]))[0]
dJ_dxT = np.zeros_like(model.get_x(), dtype=np.float64)
dJ_dxT[lower, 0] = 1.0 / len(lower)                  # J = mean x of the lower half at step T


def rollout(rho):
    """Forward solve from rest; returns the checkpointed states x_0 .. x_T."""
    adj.set_density(rho)
    model.reset()
    xs = [model.get_x().astype(np.float64)]
    for t in range(1, T + 1):
        x = model.get_body_x(h)
        x[top, 1] = x0[top, 1] - 0.003 * t / T       # press 3 mm
        x[top, 0] = x0[top, 0] + 0.002 * t / T       # shear 2 mm
        model.set_kinematic_target(h, x)
        integ.simulate(model, DT)
        xs.append(model.get_x().astype(np.float64))
    return xs


def J(rho):
    return float((dJ_dxT * rollout(rho)[-1]).sum())


xs = rollout(rho)
g = adj.transient_adjoint(xs, seed={T: dJ_dxT}, use_disc=True)      # dJ/d rho, one value per element
print(f"J = {float((dJ_dxT * xs[-1]).sum()) * 1e3:.4f} mm    |dJ/drho| = {np.linalg.norm(g):.4e}    "
      f"({adj.n_tri} elements, {adj.minres_calls} adjoint solves, "
      f"{getattr(adj, 'minres_fail', 0)} not converged)")

d = g / np.abs(g).max()                              # perturb every element along the gradient
step = 0.05
fd = (J(rho + step * d) - J(rho - step * d)) / (2 * step)
print(f"directional derivative along the gradient:  adjoint {g @ d:.6e}   "
      f"finite difference {fd:.6e}   ({abs(g @ d - fd) / abs(fd):.2%} apart)")

k = np.argsort(-np.abs(g))[:5]
print("most influential elements (index: dJ/drho):  " + "  ".join(f"{i}: {g[i]:+.2e}" for i in k))
