"""Shared core: mesh assembly, constitutive and barrier math, signed-distance colliders,
actuator geometry, config, precision."""
from .config import SimConfig
from .mesh import Mesh, build_scene, lame, check_mesh, mask_to_mesh
from . import actuators, fields, math, objects, sdf

__all__ = ["SimConfig", "Mesh", "build_scene", "lame", "check_mesh", "mask_to_mesh",
           "actuators", "fields", "math", "objects", "sdf"]
