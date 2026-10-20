"""Diff_2D_FEM — a differentiable, GPU-batched 2D simulator for soft robots, on NVIDIA Warp.

Soft bodies (Neo-Hookean P1 triangles rasterised from pixel masks, or given as triangle
meshes), rigid bodies (affine
body dynamics), IPC contact with friction between all of them and against analytic
colliders, and every common way of driving a soft robot — all advanced by one implicit
backward-Euler solve (PNCG-IPC) that batches independent environments.

    import numpy as np, Diff_2D_FEM as D

    model = D.Model()
    model.add_ground(0.0, mu=0.8)
    body = model.add_soft_body(mask, physical_size=(0.054, 0.0045), center=(0.0, 0.003),
                               E=0.5e6, nu=0.40, density=1100.0)
    model.finalize()
    model.gravity = (0.0, -9.81)

    integ = D.ImplicitIntegrator()
    for t in range(T):
        ...command the actuators...
        integ.simulate(model, dt=2e-3)
        x = model.get_body_x(body)

Bodies and colliders

    model.add_soft_body(mask, ..., E=scalar_or_map, env_id=0)
    model.add_affine_body(mask, ...)                     # a free rigid body
    model.add_affine_region(body, pixels)                # a rigid part inside a soft body
    model.add_kinematic_body(mask, ...)                  # a body whose motion is prescribed
    model.add_soft_mesh(vertices, triangles, ...)        # any of these from a triangle mesh
    model.add_affine_mesh(...); model.add_kinematic_mesh(...)
    model.add_ground(0.0); model.add_plane(normal, origin); model.add_disc(center, radius)
    model.add_object("box" | "ellipse" | "triangle", center, rx, ry)

Actuation

    model.set_kinematic_constraint(body, nodes); model.set_kinematic_target(body, x)
    for ch in model.add_pressure_chambers(body): model.set_pressure(ch, 40e3)   # pneumatic
    cable = model.add_tendon(body, through=[...]); model.set_tendon_tension(cable, 60.0)
    region = model.add_active_region(body, pixels); model.set_activation(region, 0.98)
    pin = model.add_pin(body, point); spring = model.add_spring(a, pa, b, pb, stiffness)
    model.gravity = (0.0, -9.81); model.set_external_force(body, f)

Reading back, and starting again

    integ.contact_force(model, body); integ.collider_force(model)   # what contact does
    model.reset(env_ids=[...]); model.set_body_pose(body, translation, angle)

Watching it run

    renderer = D.Renderer(model)               # a live window; examples take --viz
    renderer.begin_frame(model.time); renderer.render(); renderer.end_frame()    # every step

Differentiable

    adj = integ.adjoint(model, dt)             # dJ/d(density) via the implicit function theorem

Layout

    sim_model.py   Model — the scene
    implicit/      integrator.py (ImplicitIntegrator), solver.py + kernels.py (PNCG-IPC),
                   ipc_contact.py (IPC contact between bodies), actuation.py (energy
                   terms), affine.py (rigid bodies, rigid-mode coarse space), adjoint.py
    core/          mesh, SDF colliders, actuator geometry, constitutive math, precision
    render.py      the live window (Warp's OpenGL renderer, as in Taccel)
    viz.py         pictures and animations with matplotlib

Double precision by default; set DIFF2DFEM_PRECISION=single before importing for float32.
"""
from .core.precision import PRECISION, real, vec2, mat22, np_real
from .sim_model import Model
from .core.config import SimConfig
from .core.mesh import Mesh, build_scene, check_mesh, mask_to_mesh
from .core import objects
from .implicit.integrator import ImplicitIntegrator
from .implicit.adjoint import AdjointPNCG2D
from . import viz
from .render import Renderer

__version__ = "0.6.0"
__all__ = ["Model", "SimConfig", "Mesh", "build_scene", "check_mesh", "mask_to_mesh", "objects", "viz", "Renderer",
           "ImplicitIntegrator", "AdjointPNCG2D",
           "PRECISION", "real", "vec2", "mat22", "np_real"]
