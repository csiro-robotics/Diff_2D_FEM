"""The parameter set the solver reads: the material of a body added without one, damping,
and contact.

Every field has a plainer name on the Model — model.dhat, model.kappa, model.self_stiffness,
model.mu, model.epsv, model.damping — and that is where to set it. Pass a SimConfig to
Model(config=...) to start a scene from other values. The contact parameters left at None
are scaled to the scene at Model.finalize(); see Model.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimConfig:
    # ---- material of a body added without one (Neo-Hookean) ----
    young: float = 1.0e6              # Young's modulus [Pa]
    nu: float = 0.40                  # Poisson ratio (P1 triangles lock as it nears 0.5)
    rho_mass: float = 1000.0          # density [kg/m^3]

    # ---- damping ----
    damping: float = 0.0              # velocity drag [1/s]: v is scaled by exp(-dt*damping)

    # ---- contact ----
    ipc_d_hat: Optional[float] = None           # the barrier's reach [m];            None: 1/8 pixel
    obstacle_stiffness: Optional[float] = None  # kappa, the colliders' barrier [Pa];  None: 0.004 E pixel/dhat
    contact_stiffness: Optional[float] = None   # penalty of the "node" contact [Pa];  None: 0.2 E
    obstacle_friction: float = 0.5              # Coulomb mu of a collider added without one
    obstacle_friction_epsv: float = 1.0e-3      # friction is smoothed below this speed [m/s]
