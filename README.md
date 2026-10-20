# Diff_2D_FEM

**A differentiable, GPU-batched 2D simulator for soft robots, on [NVIDIA Warp](https://github.com/NVIDIA/warp).**

Soft bodies, rigid bodies, contact with friction, and every common way of driving a soft
robot — pressure, tendons, active material, prescribed motion — advanced together by one
implicit solve. Bodies are drawn as pixel masks, so a design is an image — or given as
triangle meshes, made by any mesher. Thousands of
independent scenes run in a single batched solve, and the solve has an adjoint.

It is a 2D counterpart to simulators such as Taccel: the same Model + Integrator shape,
the same ingredients (FEM soft bodies, affine rigid bodies, IPC contact), in plane strain,
with a preconditioned nonlinear conjugate-gradient solver (PNCG) instead of Newton.

```
PneuNet finger curls under pressure        examples/pneunet.py
tendon-driven arm pushes a rigid block     examples/soft_arm.py
crawler walks on a wave of active material examples/crawler.py
soft-pad gripper picks up a block          examples/pinch_and_lift.py
muscle flexes a forearm on a pivot         examples/muscle_arm.py
magnetic beam turns into the field         examples/magnetic_beam.py
ten designs in one solve                   examples/batch.py
```

## Install

```bash
pip install -e .                 # from this directory
pip install -e ".[examples]"     # + matplotlib, for Diff_2D_FEM.viz and --plot
```

Requires `numpy`, `scipy` and `warp-lang` (developed and tested on 1.11). A CUDA GPU is the
default device; `Model(device="cpu")` runs the same kernels on the CPU. The live window
([Watching it run](#watching-it-run)) also needs `pyglet`.

## Quickstart

A strip of soft material lying on the ground, under gravity:

```python
import numpy as np, warp as wp
import Diff_2D_FEM as D
wp.init()

model = D.Model()
model.add_ground(height=0.0, mu=0.8)
body = model.add_soft_body(np.ones((6, 72), np.uint8),          # (H, W) pixels, row 0 at the bottom
                           physical_size=(0.054, 0.0045),        # width, height [m]
                           center=(0.0, 0.0030),
                           E=0.5e6, nu=0.40, density=1100.0)
model.finalize()
model.gravity = (0.0, -9.81)

integ = D.ImplicitIntegrator()
for step in range(200):
    integ.simulate(model, dt=2e-3)
x = model.get_body_x(body)                                       # (n_nodes, 2)
```

Everything else is added to the `Model` the same way: more bodies, colliders, actuators.

## Conventions

- **2D means plane strain, per unit depth.** Pressure is in Pa. Forces, tendon tensions and
  contact reactions are in N per metre of out-of-plane depth; spring stiffness is in Pa.
  For a part of thickness *w*, a physical force *F* is *F / w* here.
- **A body is a pixel mask**, shape (H, W), non-zero = material, **row 0 at the bottom**.
  Every solid pixel becomes two linear (P1) triangles. Per-pixel maps — stiffness, active
  regions, rigid regions — use the layout of the body's own mask.
- **Or a triangle mesh** — vertices and triangles, taken as given ([Meshes](#meshes)).
  Its per-element data is per triangle, in the mesh's own order.
- **SI units throughout.**
- **Double precision** by default. See [Precision](#precision).

## Bodies

```python
h = model.add_soft_body(mask, physical_size=(w, h), center=(x, y),
                        E=1.0e6, nu=0.40, density=1100.0, mu=0.6)   # mu: friction, see Contact

# a stiffer skin, a strain-limiting layer, a soft hinge: E may be a per-pixel map
h = model.add_soft_body(mask, ..., E=stiffness_map)              # (H, W), like the mask

# a free RIGID body
box = model.add_affine_body(np.ones((16, 16)), physical_size=(0.016, 0.016),
                            center=(0.02, 0.008), density=600.0)

# a rigid part INSIDE a soft body (after finalize): a bone, a link, a mounting plate
bone = model.add_affine_region(h, pixels)                        # (H, W) bool
model.affine_state(bone)                                         # position, angle, A
```

Each body has its own grid, size, position and material. Neo-Hookean elasticity.

A body can also be given as a triangle mesh, made elsewhere — a CAD outline meshed with
gmsh, a part that is not drawn on a grid:

```python
h = model.add_soft_mesh(vertices, triangles, E=1.0e6, nu=0.40, density=1100.0)  # (n, 2) m, (m, 3)
r = model.add_affine_mesh(vertices, triangles, density=600.0)                     # rigid
k = model.add_kinematic_mesh(vertices, triangles)                                 # prescribed
```

See [Meshes](#meshes).

**Rigid bodies are affine bodies** (Affine Body Dynamics, Lan et al. 2022). The nodes of a
rigid body or region move as `x = t + A X` with one translation and one 2×2 matrix for the
whole piece, and `A` is held to a rotation by a large stiffness (`model.rigid_young`,
1 GPa). That is six unknowns however many pixels the piece has, with their own exact block
of the preconditioner. Use them for anything stiff — a soft body with a huge modulus would
need thousands of iterations per step instead. A rigid region stays attached to the soft
material around it because the two share their interface nodes; leave at least one soft
pixel between two rigid regions.

A body whose motion is *prescribed* — a jaw, a pusher, a moving platform — is a kinematic
body:

```python
jaw = model.add_kinematic_body(mask, physical_size=(w, h), center=(x, y))
model.set_kinematic_pose(jaw, translation=(dx, dy), angle=0.0)   # every step: where to be next
```

It touches the others through contact like any body, and is swept to its pose rather than
placed there (see [Contact](#contact)).

## Meshes

A body that is not drawn on a grid — an outline from CAD, a part meshed with gmsh or
`triangle` — is passed as its mesh:

```python
h = model.add_soft_mesh(vertices, triangles, E=1.0e6)   # E: a scalar, or one value per triangle
model.add_pin(h, vertices[57])                          # the mesh's own vertex numbers
```

- **Taken as given.** Vertex *i* is node *i* of the body and triangle *j* its element *j*,
  so pins, tendons, constraints and per-element fields use the mesh's own numbering.
  Vertices are world positions in metres; either winding, mixed included.
- **Checked first** (`D.check_mesh`). A defect is reported by vertex or triangle:
  - malformed or non-finite arrays, out-of-range or repeated indices;
  - duplicate triangles, unused vertices, zero-area triangles;
  - an edge shared by three triangles, a fold;
  - coincident vertices or a crack, an outline that crosses itself.

  Slivers get a warning.
- **Per triangle, not per pixel.** These take an (m,) boolean array or triangle indices
  instead of a pixel map:
  - a stiffness map;
  - `add_affine_region`, `add_active_region`;
  - `to_elements`, which takes (m,).

  A mesh body has no `pixel_size` or `to_pixels`.
- **Holes are chambers.**
  - `add_pressure_chambers(h)` makes every hole in the mesh a chamber.
  - `add_pressure_chamber(h, at=point)` makes the hole containing `point` one.
  - `add_ambient_pressure(h)` acts on the outline.
  - A node where two boundary loops meet makes the boundary ambiguous, and pressure refuses it.
- **Its scales.**
  - **Contact** is scaled from the body's median boundary edge, `model.element_size(h)`. `model.dx` is the largest over the bodies, and plays the part of the pixel size.
  - **Iterations per step** follow its smallest element, `sqrt(2 · area)`. A few tiny elements make every step dearer; a spread beyond 10:1 is warned about.
- **Graded meshes keep their own nodes apart.** A body's self-contact excludes:
  - its nodes within two edges of each other, as on pixels;
  - every pair of boundary nodes closer than `max(2 dx, 16 dhat)` *through the material*.

  On a mesh with 0.02 mm elements next to 2 mm ones, two edges are well inside the barrier's reach. The material path is not: nothing of the body pushes on itself at rest. Across a slit or a concave corner the path through the material is longer than the gap, so contact there stays.
- **A pixel body is a special case.** `D.mask_to_mesh(mask, size, center)` returns the mesh
  `add_soft_body` builds, with the pixel of each triangle. Added with `add_soft_mesh`, it
  reproduces the pixel body bit for bit. Turn its vertices first to place a pixel design
  at an angle.
- **The adjoint is per triangle on a mesh as on pixels**: `rho` has one entry per triangle.
  On uneven elements a gradient scales with the element's area (`model.element_areas(h)`).
- **The resolution is the mesh's own.** P1 triangles are stiff in bending
  ([What has been checked](#what-has-been-checked)), so give a slender part several
  elements through its thickness. There is no remeshing in the package; use gmsh or
  `triangle`.

## Environments and batching

A scene is one environment by default: every body lands in environment 0, and bodies of
one environment collide with each other. `env_id` puts a body in another environment.
Environments are advanced together by one batched solve but never interact — each has its
own step length and convergence test and sees its own copy of the colliders.

```python
for i, mask in enumerate(designs):                                # a population of designs
    model.add_soft_body(mask, ..., env_id=i)
```

One environment can be put back to rest while the others carry on, and a body can be put
anywhere:

```python
model.reset(env_ids=[3, 7])                                       # or reset(): all of them
model.set_body_pose(h, translation=(dx, dy), angle=0.3, velocity=(0.1, 0.0))
model.set_body_x(h, x, v)                                         # node by node
```

A design simulated in a batch reproduces the design simulated alone **bit for bit**
(`tests/test_scene.py`, on the CPU, where kernels are deterministic). Ten designs in one
solve take 1.3× the time of one ([`examples/batch.py`](examples/batch.py)).

**Throughput.** Environment-steps per second against the number of environments, in double
precision on an RTX A5000 that another job was loading to about 40 % — lower bounds, then
([`tools/benchmark_envs.py`](tools/benchmark_envs.py)):

| environments | 1 | 16 | 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|---|---|
| cantilever bent by a tendon — 729 nodes, dt 1 ms, 177 iterations | 27 | 353 | 666 | 909 | 1037 | 1061 |
| strip lying on the ground — 511 nodes, dt 2 ms, 167 iterations | 14 | 187 | 472 | 678 | 751 | 744 |
| two pads lifting a rigid block — 595 nodes, dt 1 ms, 170 iterations | 9 | 97 | 285 | 429 | 531 | 521 |

One environment is bound by kernel launches (0.2 ms per iteration; 0.7 with contact
between bodies), not by arithmetic: sixteen take 1.2–1.4× the time of one. From about 1024
on the card is full, and a step takes nodes × iterations × 7 ns for elasticity alone,
16–19 ns with contact. 4096 environments — two to three million nodes — hold 0.8–1.0 GiB
and are set up in five seconds or less; setting every environment's commands from Python
is 0.4 % of the step.

Summed over the batch that is 0.5–1.5× real time at these steps of 1–2 ms. A larger step
buys more, because the iterations a step needs level off (see [The solver](#the-solver)):
at dt 20 ms with `max_iter=707` the cantilever runs at 261 environment-steps per second,
5.2× real time — and at the default budget, 2.1×.

## Colliders

Fixed analytic shapes — signed-distance functions, not meshes. Any number of them.

```python
model.add_ground(height=0.0, mu=0.8)
model.add_plane(normal=(1.0, 0.0), origin=(-0.05, 0.0))            # a wall
model.add_disc(center=(0.05, 0.03), radius=0.006, mu=0.6)
model.add_object("ellipse",  center=(0.05, 0.03), rx=0.008, ry=0.005)
model.add_object("box",      center=(0.05, 0.03), rx=0.007, ry=0.005, corner_radius=0.002)
model.add_object("triangle", center=(0.05, 0.03), rx=0.007, corner_radius=0.002)

F = integ.collider_force(model, index)      # reaction on one collider, per environment
F = integ.collider_force(model)                 # ... on all of them together
```

Keep `corner_radius > 0`: a sharp corner has unbounded curvature. `model.colliders`
returns the same shapes on the host (`core.objects`: `phi`, `outline`, `perimeter`,
`contact_arc`) for plots and contact-length measurements.

## Contact

**Against colliders:** the IPC log-barrier on the signed distance, with continuous
collision detection (a node cannot pass through a surface), and Coulomb friction lagged
over the step and smoothed below a small sliding speed `epsv`.

**Between bodies, and of a body with itself:** the same barrier, between every boundary
node and every boundary edge — soft against soft, soft against rigid, rigid against rigid
([`implicit/ipc_contact.py`](implicit/ipc_contact.py)) — with Coulomb friction.

- **Friction is per body, as in Taccel.** Every `add_*` call takes `mu=`; two bodies in
  contact slide with the harmonic mean of their two values, `2 mu_a mu_b / (mu_a + mu_b)`
  — Taccel's rule — so a body with `mu=0` is frictionless against everything. A body
  added without one uses `model.body_friction` (0 by default): with friction on only one
  of two touching bodies, the pair is frictionless. Colliders keep their own `mu`, which
  is not combined with the bodies' (Taccel's planes work the same way).

  ```python
  pad = model.add_soft_body(mask, ..., mu=0.8)          # a grippy silicone pad
  box = model.add_affine_body(block, ..., mu=0.4)       # pad on box slides at 2·0.8·0.4/1.2 = 0.53
  ```

- **Surfaces stay apart.** Before each step along the search direction, every node–edge
  pair nearby is advanced by additive continuous collision detection, so no step crosses
  a surface at any speed. Bodies settle within `dhat` of each other — an eighth of a pixel
  by default.
- **Flat faces are flat.** A node near a mesh vertex is within reach of both edges that
  meet there. Summed plainly it is counted twice, and a pressed surface grips like a
  ratchet with no friction at all; so each vertex is subtracted once (inclusion–exclusion),
  and on a flat face the potential is exactly the barrier of the distance to the face.
- **Prescribed bodies are swept.** Kinematic nodes are moved to their targets under the
  same step limit instead of being placed there, so a body driven with `rigid_motion`
  pushes what is in front of it rather than appearing inside it. The drive is
  force-limited: it closes a gap to a quarter of `dhat` and no further — about 0.4 E of
  contact pressure at the default `model.body_kappa` — and stops short of a target it
  cannot reach. `integ.stats(model)["bc_lag"]` says by how much.
- **Bodies must start apart.** Two surfaces that touch before the first step are refused.

What contact does to a body can be read back, node by node or as a resultant:

```python
f = integ.contact_force(model, h)                  # (n_nodes, 2) N/m: colliders and other bodies
F = integ.body_contact_force(model, h)             # its sum over the body
N = integ.body_contact_force(model, h, friction=False, colliders=False)   # normal part, bodies only
```

Kinematic nodes are included, so on a kinematic body this is the force its drive has to
supply. Two stacked blocks give back their weights to five digits, and the forces two
bodies exert on each other cancel to rounding.

Measured — a block on a held slab with gravity tilted, and two soft pads pinching a free
rigid block and lifting it ([`tools/measure_contact.py`](tools/measure_contact.py)):

| | `"ipc"` (default) | `"node"` |
|---|---|---|
| resting gap | 0.12 pixel | 1.5 pixels |
| energy change while a flat block slides along a flat slab, no friction | 2 % | 12 % |
| frictionless slide down 40°, speed vs closed form | 1.0000 | 0.89 |
| sliding with friction, rigid block (mu 0.3 at 30°; mu 0.5 at 40°) | 1.0000; 1.0000 | about 0.2, different every run |
| sliding with friction, soft block (same two cases) | 1.0000; 1.0000 | about 0.2 |
| … steadiness: slowest and fastest 50 steps | 1.000 – 1.000 | 0.05 – 0.6 |
| held by friction and creeping (mu 0.5, 15°) | 1.0003 | does not settle |
| pinched with no friction, block lifted by | −0.3 % | −1 % |
| pinched with mu 0.8, block lifted by | 99.6 % | 96.5 % |

The 2 % in the second row is one node in fifty passing the end of the block — a corner
is a corner. `"node"` (`model.body_contact = "node"`) is the penalty contact this package
had before: it repels nodes from nodes, which makes every surface corrugated — it resists
sliding like friction of 0.35 whatever `body_friction` says — and holds bodies a pixel
and a half apart. It is kept because it is the one contact between bodies that has a
tangent in the adjoint.

What it costs. A contact that closes or opens within a step needs more iterations than a
wave crossing the mesh does, so with IPC contact the iteration budget is at least 120 per
step; they are only spent when needed. An impact at half a pixel per step conserves
momentum to 4e−5 with that budget and lost 4 % with 45. A step that still runs out
(`integ.stats(model)["converged"]`) is never penetrated but is not accurate either — a
block thrown at another at five pixels per step stops where it should and keeps the wrong
speed; take a smaller `dt`. A free body pushed by a prescribed one is followed at about
0.2 `dhat` per iteration, some three pixels per step.

Contact parameters default to values scaled from the scene, so a model in millimetres and
one in metres both work untouched:

| | default | |
|---|---|---|
| `model.dhat` | 1/8 pixel | how far from a surface the barrier reaches |
| `model.kappa` | 0.004 · E · pixel / dhat | barrier stiffness |
| `model.body_kappa` | 0.1 · E · pixel / dhat | barrier stiffness between bodies |
| `model.epsv` | 1 mm/s | friction is smoothed below this sliding speed |
| `model.mu` | 0.5 | friction of colliders added without one |
| `model.damping` | 0 | velocity drag [1/s] |

*E* is the modulus of the softest body. A body resting on a collider floats about
0.9 `dhat` above it, and the barrier's reach is a modelling error that shrinks with it.
Distance covered by the crawler of `examples/crawler.py` in one gait cycle:

| dhat | 1 pixel | 1/2 | 1/4 | **1/8** | 1/16 |
|---|---|---|---|---|---|
| advance per cycle [mm] | 1.73 | 1.33 | 1.19 | **1.09** | 1.07 |

It converges, and a barrier one pixel wide — a common choice — overstates this gait by 60 %.

Below the smoothing speed friction does not hold a body still: it creeps at
`epsv × (tangential load) / (mu × normal load)`. Lower `epsv` if that matters.

## Actuation

Every actuator is one more energy term in the implicit solve, commanded each step. Add
them after `finalize()`.

```python
# prescribed motion
model.set_kinematic_constraint(h, node_indices)
model.set_kinematic_target(h, positions)                  # (n_body_nodes, 2), every step

# pneumatic / hydraulic — every sealed cavity of the mask is a chamber
chambers = model.add_pressure_chambers(h)                 # or add_pressure_chamber(h, at=(x, y))
ambient  = model.add_ambient_pressure(h)                  # pressure on the outer surface
model.set_pressure(chambers[0], 40e3)                     # Pa; + inflates

# tendon — a cable through frictionless eyelets (waypoints snap to the nearest node)
cable = model.add_tendon(h, through=[(x0, y0), (x1, y1), ...])          # tension-controlled
model.set_tendon_tension(cable, 60.0)                                   # N/m
motor = model.add_tendon(h, through=[...], stiffness=2e5)               # length-controlled
model.set_tendon_length(motor, model.tendon_length(motor) - 100e-6)     # reel in 100 um

# active material — SMA, DEA / LCE, thermal expansion, swelling, muscle
region = model.add_active_region(h, pixels, fiber=None)   # fiber = angle [rad] or a vector
model.set_activation(region, 0.98)                        # rest-length ratio: 2 % contraction

# springs, pins and joints
pivot = model.add_pin(h, point)                           # to the world: free to rotate about it
strut = model.add_spring(a, pa, b, pb, stiffness=1e6)     # set_spring_length drives it

# loads
model.gravity = (0.0, -9.81)
model.set_external_force(h, nodal_forces)                 # (n_body_nodes, 2) N/m
model.set_body_velocity(h, velocity=(0.3, 0.0), omega=20.0)   # an initial condition
```

Read-outs: `chamber_area`, `tendon_length`, `tendon_tension`, `spring_force`,
`affine_state`, `get_body_x`, `get_body_v`, `model.time`, `element_strain`,
`element_stress`; `integ.contact_force`, `integ.collider_force`.

- **A chamber is a void enclosed by solid.** A channel that reaches the edge of the pixel
  grid is open to the outside and belongs to the ambient pressure. Voids touching only at
  a corner are separate chambers. On a mesh body, a chamber is a hole in the mesh
  ([Meshes](#meshes)).
- **A pneumatic finger needs an asymmetry** — chambers off-centre, or a stiffer layer on
  one side — or it only swells.
- **Cables cannot push**: slack, they contribute nothing.
- **A pin is a spring of zero length and very high stiffness** to a point of the world: it
  holds the body there and leaves it free to turn. There is no pin between two bodies —
  it would hold two nodes at one place while contact holds the bodies apart. Join rigid
  parts by a flexure inside one body (`add_affine_region`, as in `bone_finger.py`).

## The solver

`ImplicitIntegrator` takes backward-Euler steps. Each step minimises the incremental
potential — inertia, elasticity, barrier, friction, and every actuator — with
preconditioned nonlinear conjugate gradients (PNCG; Dai-Kou direction, a Newton step
length from exact curvature along the direction, capped so that no element inverts and no
node crosses a collider). No linear system is assembled, which is what makes batching many
small scenes cheap. It is unconditionally stable: `dt` is chosen for accuracy.

```python
integ = D.ImplicitIntegrator()            # the defaults below
integ.simulate(model, dt)
integ.stats(model)                        # iterations used, converged?, per environment
```

**How many iterations per step** (`max_iter=None`, the default). One CG iteration carries
information across one element; a backward-Euler step lets an elastic wave cross
`C = c_p · dt / dx` of them, the step's acoustic CFL number. Curvature of a tendon-driven
cantilever as a fraction of its converged value, on four combinations of mesh and step:

| iterations per step | 0.5 C | 1.0 C | 1.5 C | 2 C | 6 C |
|---|---|---|---|---|---|
| 8×80 elements, dt 1 ms (C = 88)    | 59 % | 87 % | 100.0 % | 100.0 % | 100.0 % |
| 8×80 elements, dt 2 ms (C = 177)   | 67 % | 98.6 % | 100.0 % | 100.0 % | 100.0 % |
| 16×160 elements, dt 1 ms (C = 177) | 59 % | 87 % | 100.0 % | 100.0 % | 100.0 % |
| 8×80 elements, dt 4 ms (C = 353)   | 84 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |

So the default is `2 C`, between 40 and `max_iter_cap = 2000` (a warning says when the cap
binds). With too few, the structure responds late and too stiffly — it does not blow up.
`integ.cfl_number(model, dt)` reports *C*.

For steps that resolve the motion, the cost per simulated second therefore does not depend
on `dt`. For larger steps it falls: the count stops following *C* and levels off at what a
static solve of the structure needs. The same 8×80 cantilever
([`tools/study_iterations.py`](tools/study_iterations.py), both tables):

| iterations per step | 177 | 354 | 500 | 707 | 1000 |
|---|---|---|---|---|---|
| dt 10 ms (C = 883)  | 65 % | 97.1 % | 100.0 % | 100.0 % | 100.0 % |
| dt 20 ms (C = 1765) | 55 % | 91.6 % | 99.97 % | 100.0 % | 100.0 % |
| dt 40 ms (C = 3531) | 55 % | 79 % | 96.5 % | 100.0 % | 100.0 % |

About 700 iterations do for any step of this beam, where `2 C` asks for 1766 to 7062 and
the cap grants 2000: at large steps the default spends about three times what is needed,
so set `max_iter` yourself.

**Bodies that move as a whole** are a different matter and are handled separately. A rigid
translation costs no elastic energy, yet a nodal preconditioner prices it at the full
elastic stiffness, so the step it proposes is too short by *C²*. Two remedies are built
in: rigid bodies carry their own six unknowns with an exact 6×6 block, and every free
piece of *soft* material gets its three rigid-body directions added to the preconditioner
(`rigid_modes=True`). Free fall of a soft block over 100 steps:

| error of the fall distance | 5 iterations | 20 | 150 |
|---|---|---|---|
| nodal preconditioner only | −100 % | −99.9 % | −0.3 % |
| + rigid modes (the default) | −3e−9 | −1e−9 | −1e−9 |
| rigid (affine) body | 1e−13 | 1e−13 | 1e−13 |

**Stopping early** (`eps`, default 1e−10 in double precision). A step ends once an
iteration lowers the energy by less than `eps` times what the step's first iteration did;
the error left in the step is about `sqrt(eps)`. A body in free flight converges in 2–10
iterations and a rigid one sliding on the ground in about a dozen; clamped structures
usually run the full budget (given 2000 iterations at dt 20 ms, the beam above stopped at
about 1800 — `eps` guards against wasted work on a body at rest; it is not an economical
stopping test).

### Precision

Double precision is the default, and the reason is specific. Elastic forces are
differences of nearly equal numbers formed from absolute positions, so in single precision
their rounding error is about `6e-8 · E · |x|` per node — larger than the restoring force
of a soft bending mode. The iteration then stalls short of equilibrium, by an amount that
grows with slenderness and resolution. Tip curvature of a cantilever against a
double-precision Newton solve of the same mesh:

| mesh | 8×40 | 8×80 | 16×160 |
|---|---|---|---|
| single precision, 30 000 iterations | −1.5 % | −94 % | −99.4 % |
| double precision, 2 000 iterations | 0.00 % | 0.00 % | +0.04 % |

Per iteration, double costs nothing at one environment (which is bound by kernel
launches, not arithmetic) and 1.9–2.9× at 4096 on an RTX A5000
([`tools/benchmark_envs.py`](tools/benchmark_envs.py)). Single precision is kept for very
large batches where approximate equilibria are acceptable; its default `eps` is 1e−4:

```bash
DIFF2DFEM_PRECISION=single python my_script.py        # must be set before the import
```

## Differentiability

```python
adj = integ.adjoint(model, dt)
adj.set_density(rho)                          # per element, in [0, 1]
model.reset(); xs = [model.get_x()]
for t in range(T):
    ...; integ.simulate(model, dt); xs.append(model.get_x())
g = adj.transient_adjoint(xs, seed={T: dJ_dx_T}, use_disc=True)       # dJ / d rho
```

The adjoint of the implicit solve by the implicit function theorem: one linear solve per
step, gradient of an objective on the trajectory with respect to a per-element density
that scales each element's own modulus, `E(rho) = E · (Emin + rho (1 − Emin))`.
`model.to_elements(h, field)` and `model.to_pixels(h, values)` map between a body's pixel
grid and its elements. Nothing in the adjoint depends on the grid: on a mesh body `rho`
is simply one value per triangle (checked against finite differences on jittered
triangles, `tests/test_mesh.py`). See [`examples/density_gradient.py`](examples/density_gradient.py).

Checked against finite differences of the full rollout: cosine 0.9999 or better for
elasticity and for contact with a disc and a box. **Its scope is narrow**, and it refuses
to run outside it rather than return the gradient of a different system:

- contact is differentiated with the active set frozen at each converged state;
- **friction has no tangent**: with `mu = 0.6` the gradient's cosine against finite
  differences is 0.33. Differentiate frictionless rollouts;
- pressure, tendons, active material, springs, rigid bodies and IPC contact between
  bodies are not in the adjoint operator (gravity, external forces and the `"node"`
  contact are fine): differentiate a scene of several bodies with
  `ImplicitIntegrator(self_collision=False)` or `model.body_contact = "node"`;
- the design variable is density only — not shape, not control signals.

The objective is any function of the node positions — or of the elastic FORCE at some
nodes, which is how a reaction is optimised (the force a held output passes to its
support):

```python
J, dJ_dx, explicit = adj.force_objective(xs[T], nodes, direction)   # J = direction . force on nodes
g = adj.transient_adjoint(xs, seed={T: dJ_dx}, rtol=1e-14) + explicit
```

A force depends on the density directly as well as through the positions, hence the
explicit part. The tight `rtol` matters for such a seed: MINRES's tolerance bounds the
preconditioned residual, and at the default 1e-9 a reaction gradient came out 4 % off;
at 1e-14 it matches finite differences to 1e-5. See
[`examples/topopt_gripper.py`](examples/topopt_gripper.py).

## Watching it run

A window that shows the scene while it is being simulated:

```python
renderer = D.Renderer(model)                 # after model.finalize(): opens the window
for step in range(n):
    ...command the actuators...
    integ.simulate(model, dt)
    renderer.begin_frame(model.time)         # model.time: the simulated time so far [s]
    renderer.render()
    renderer.end_frame()
```

These are the three calls per step of Taccel's examples, and the window is the same one:
Warp's OpenGL renderer. `renderer.draw()` is the three in one. The examples open it with
`--viz`:

```bash
python examples/crawler.py --viz
```

| in the window | |
|---|---|
| drag, arrow keys | move the view |
| scroll, `+` / `-` | zoom |
| `F` | fit the view to the bodies as they are now |
| `R` | back to the first view |
| `X` | outline every triangle |
| `SPACE` | pause — the simulation loop waits inside `end_frame()` |
| `ESC` | close; the loop runs on without a picture |

Bodies are coloured as in the pictures of `viz`: one colour per body, rigid parts dark,
kinematic bodies grey, cables red, springs and pins green. The first view holds the bodies
at rest, the colliders and the ground beneath them; give `bounds=((xmin, ymin), (xmax,
ymax))` when the bodies travel further than that.

**Several environments** are shown side by side — `D.Renderer(model, envs="all")`, or a
list of them. A whole batch is a handful of meshes, so drawing it costs a few milliseconds.

**At its true speed.** One scene is advanced at some ten to thirty steps per second, so with
steps of a millisecond or two the live window is slow motion. To see a run as fast as it
happened, keep the states and replay them:

```python
frames = []
for step in range(n):
    integ.simulate(model, dt); frames.append(model.get_x())
renderer.play(frames, dt)                    # real time; speed=0.25 for a quarter of it
```

`renderer.hold()` at the end of a script keeps the window open until it is closed.
`renderer.pixels()` returns the frame on show as an image, and `headless=True` renders
without a window on the screen.

### Pictures and animations

```python
from Diff_2D_FEM import viz
viz.draw(model)                              # bodies, rigid parts, colliders, cables, springs
viz.draw(model, values=model.me.mu)          # colour by any per-element field
viz.animate(model, frames, "run.gif")        # frames: a list of model.get_x()

viz.draw(model, field="stress")              # von Mises stress, blue (low) to red (high), with a colour bar
viz.animate(model, frames, "run.gif", field="strain",        # every frame recoloured, ONE scale
            titles=[f"t = {t:.3f} s" for t in times])        # clim=(lo, hi), cmap=... to override
viz.draw(model, field="stress", signed=True)  # tension red (+), compression blue (-), unloaded grey
viz.animate(model, frames, "batch.gif", envs="all", field="stress",   # a batch side by side,
            panel_titles=[...], bounds=((x0, y0), (x1, y1)))           # one scale, one view
```

`field="strain"` is the von Mises equivalent of the Green–Lagrange strain and
`field="stress"` the von Mises equivalent of the Cauchy stress [Pa] of the solver's own
Neo-Hookean, one value per triangle (`core/fields.py`); the same numbers come out of
`model.element_strain(x)` and `model.element_stress(x)` (`handle=` for one body's, in
`body_elements` order). `signed=True` (in `draw`, `animate` and both read-outs) gives
instead the in-plane principal value of larger magnitude with its sign — positive where
the material is pulled, negative where it is squeezed; along a bent beam, one face red and
the other blue — on a scale symmetric about 0. Plane strain keeps the out-of-plane stretch
at 1, so `σ_zz` is not zero and enters the von Mises stress. Prescribed bodies are drawn grey. The default
colour map is `coolwarm` (blue – grey – red, no rainbow); `cmap="jet"` gives the classic
FEA look.

## Examples

One classic soft robot for each way of driving one:

| actuation | example | |
|---|---|---|
| fluid pressure | [`pneunet.py`](examples/pneunet.py) | a PneuNet bending actuator: eight bellows on a strain-limiting base swell, press on each other and curl the finger 172° at 18 kPa |
| tendons | [`soft_arm.py`](examples/soft_arm.py) | a two-cable arm pushes a free rigid block, which tips and slides |
| | [`bone_finger.py`](examples/bone_finger.py) | rigid phalanges on flexure hinges, one cable, wraps a disc |
| active material | [`crawler.py`](examples/crawler.py) | a free strip walks over the ground on a travelling wave of contraction |
| prescribed motion | [`pinch_and_lift.py`](examples/pinch_and_lift.py) | a parallel gripper with soft fingertips: the block comes up with friction 0.8 and stays down with none |
| springs and pins | [`muscle_arm.py`](examples/muscle_arm.py) | a rigid forearm on a pivot, flexed by a contracting spring; its angle matches statics to 0.01° |
| external forces | [`magnetic_beam.py`](examples/magnetic_beam.py) | a hard-magnetic beam in a uniform field: beam theory at a weak field, turning into a strong one, an S with two domains |

Design and optimisation — a population per batched solve, or the adjoint:

| method | example | |
|---|---|---|
| genetic algorithm | [`evolve_walker.py`](examples/evolve_walker.py) | a pneumatic walker: a 100 x 50 mm block with two upright 15 x 25 mm air channels, inflated 90° apart (P(1 + sin), P(1 + cos)); the genetic algorithm decides where there is material around their fixed walls — legs, a waist, holes — from a 19-number genome, 48 designs per solve, against the block as drawn |
| CMA-ES | [`inverse_pneunet.py`](examples/inverse_pneunet.py) | inverse design: the strain-limiting layer under each of eight bellows, so that the finger curls into a target hook |
| adjoint | [`topopt_gripper.py`](examples/topopt_gripper.py) | topology optimisation of a gripper finger on the 128 × 64 boundary conditions of the finger datasets (top-left held, top-right pushed down, tip held): the material that makes the tip press hardest along (−1, −1) — a reaction-force objective (`force_objective`) |

The CMA-ES of `inverse_pneunet.py` is [`_cmaes.py`](examples/_cmaes.py), numpy only.

And the rest:

| | |
|---|---|
| [`actuation.py`](examples/actuation.py) | pressure, tendon, active material and gravity on one strip |
| [`batch.py`](examples/batch.py) | ten designs in one solve |
| [`object_shapes.py`](examples/object_shapes.py) | the same squeeze against a disc, an ellipse, a box and a triangle |
| [`mesh_body.py`](examples/mesh_body.py) | bodies given as meshes: a ring meshed in polar coordinates, inflated through its hole, lifts a turned rigid block |
| [`press_on_disc.py`](examples/press_on_disc.py) | the smallest one: a block pressed onto a disc |
| [`density_gradient.py`](examples/density_gradient.py) | the adjoint against finite differences |

Run any of them with `python examples/<name>.py`; the robots in the first table take
`--plot` and save a picture. `--viz` opens the live window on every one that moves — all
but `magnetic_beam.py`, `actuation.py` (settled shapes, a dozen long steps each) and
`density_gradient.py`. They run in seconds to two minutes on an RTX A5000; the three design
examples take 3 to 15 minutes (`--quick` for a short run of the first two).

## What has been checked

`tests/` runs in about ten minutes (`python tests/test_<name>.py`, or pytest).
`tests/reference_static.py` is an independent double-precision Newton solver with a sparse
direct factorisation; it is what separates the error of the elements from the error of
the solver.

| check | result |
|---|---|
| gradient = d(energy)/dx, curvature = d(gradient)/dx, preconditioner diagonal — every actuation term | agree to 1e−4 or better |
| hydrostatic state: equal pressure inside and out → uniform stretch, closed form | within 0.12 % of the strain |
| thick-walled ring under internal pressure (Lamé), solved to convergence | 2.2 % (48 px), 1.4 % (96 px): the staircase outline; the circle's polygon as a mesh, 0.05 % |
| bar hanging under its own weight | 0.74 % |
| free growth of an active body, isotropic and along a fibre | 3e−6 of the body size |
| tendon-driven cantilever vs the Newton reference | 0.1 % |
| free rigid body: fall, and spin with drift, vs exact backward Euler | 1e−13; 3e−8 m after 200 steps |
| block on a slope vs closed form, rigid and soft: sliding speed; creep speed | 0.07 %; 3e−6 |
| ground reaction under a resting block; under two stacked soft blocks | the weight to 1e−13; within 2 % |
| cantilever with a rigid outer half vs the reference with that half 1000× stiffer | 0.01 % |
| rigid bar on a pin: period of a physical pendulum | 5e−7; the pin opens 3e−10 m |
| rigid block thrown at a soft one: linear momentum through the impact | conserved to 1e−6 |
| a design in a batch vs the same design alone | identical bit for bit |
| IPC contact between bodies: energy vs gradient vs curvature, against finite differences | 1e−9; 6e−5 |
| … a block on a body vs closed form, rigid and soft: frictionless slide; slide with friction; creep | 1.0000; 1.0000; 1.0003 |
| … two stacked blocks: ground reaction; gap between them | the two weights to 1e−6; 0.99 dhat |
| … a free block pinched by two soft pads and lifted: with friction; without | comes up 99.6 %; stays down |
| … a block thrown at another at five pixels per step, head on and corner to corner | never passes through |
| … a prescribed body pushing a free block; squeezing it against a held one | no lag; stops at the drive's force limit |
| every collider shape: host level set = device level set; sliding friction = mu × normal | 2e−6 m; 3 % |
| adjoint vs a full finite-difference gradient (elasticity, disc contact, box contact) | cosine ≥ 0.9999 |
| **mesh bodies** (`tests/test_mesh.py`): a mesh made of a pixel body's triangles vs that pixel body — mesh, contact scales, trajectory, contact forces | identical bit for bit |
| … a hanging bar turned 30° with gravity turned with it, vs upright | 1.6e−10 of the extension |
| … a cantilever of jittered triangles under its own weight vs the Newton reference on the same mesh | 1e−10 |
| … adjoint vs finite differences on jittered triangles (elasticity; disc contact), along the gradient | 0.01 %; 0.04 % |
| … a ring meshed in polar coordinates under internal pressure (Lamé, 10 elements across) | 0.05 % |
| … a turned block on a turned plane; a mesh block on a pixel slab: ground reaction / weight | 1.000000; 1.000000 |
| … a graded mesh (elements 100:1) at rest with self-contact on | no force on itself |

P1 triangles are stiff in bending. That is a property of the elements, not of the solver,
and refining the mesh removes it — curvature of a slender cantilever against beam theory:
−8.8 % with 8 elements through the thickness, −3.3 % with 16, −1.8 % with 32.

## Limitations

- **Contact between bodies costs iterations.** It is the IPC barrier, and it holds: nothing
  passes through anything. But the solver takes its steps from local curvature, not from a
  line search, and a barrier's curvature changes fast — so a pair may only deepen so much
  per iteration (see [Contact](#contact)), an impact of several pixels per step does not
  converge within the budget, and a prescribed body pushes a free one at a few pixels per
  step at most. Contact is sampled at the nodes: a corner sliding over a mesh shows as a
  bump of one node's worth of energy. Bodies of very different pixel sizes share one
  `dhat`, an eighth of the largest pixel (of the largest median boundary edge, for mesh
  bodies).
- **Colliders do not move by themselves.** `set_collider_center` teleports one between
  steps; the move is not swept and friction treats the collider as stationary.
- **Bending needs resolution.** With 8 elements through the thickness a beam is 9 % too
  stiff; with 16, 3 %.
- **Stiff soft bodies are expensive**: iterations per step grow with the wave speed. Use
  rigid regions for stiff parts.
- **The adjoint is narrow** — see [Differentiability](#differentiability).
- **No remeshing.** A mesh body is simulated on the triangles it is given; refine it
  elsewhere.
- **Overlap of two pixel bodies is checked at pixel centres**, so two bodies overlapping
  by less than half a pixel are not refused at `finalize()` (between mesh bodies the check
  is exact). Lay pixel bodies out with a gap.
- **Plane strain only.** No plane stress, no axisymmetry.
- **GPU runs are not bitwise reproducible**: sums over nodes are unordered, and two runs of
  one scene differ by about 1e−7 m after a few dozen steps in contact. CPU runs are exact.

## Layout

```
sim_model.py     Model — the scene
implicit/        integrator.py   ImplicitIntegrator
                 solver.py, kernels.py   the PNCG-IPC iteration
                 actuation.py    pressure, tendon, active strain, springs, forces
                 affine.py       rigid bodies; rigid-mode coarse space
                 ipc_contact.py  IPC contact between bodies: barrier, step limit, friction
                 body_contact.py friction for the older "node" contact
                 adjoint.py      adjoint of the implicit solve
core/            mesh.py (masks or given meshes -> the scene mesh; check_mesh, mask_to_mesh),
                 sdf.py + objects.py (colliders), fields.py (strain, stress per element),
                 actuators.py (chamber / tendon / region geometry), math.py, precision.py,
                 config.py
render.py        the live window (Renderer)
viz.py           pictures and animations with matplotlib
tests/           the checks in the table above
tools/           the studies and measurements behind the defaults and the tables
examples/
```

## References

The methods this package puts together:

- **IPC** — Li et al., *Incremental Potential Contact: Intersection- and Inversion-free,
  Large-Deformation Dynamics*, ACM Trans. Graph. 39(4), 2020. The incremental potential,
  the log barrier, lagged smoothed friction.
- **PNCG** — Shen et al., *Preconditioned Nonlinear Conjugate Gradient Method for Real-time
  Interior-point Hyperelasticity*, SIGGRAPH 2024. The matrix-free solve; its direction is
  that of Dai & Kou, *A Nonlinear Conjugate Gradient Algorithm with an Optimal Property and
  an Improved Wolfe Line Search*, SIAM J. Optim. 23(1), 2013.
- **Affine body dynamics** — Lan et al., *Affine Body Dynamics: Fast, Stable and
  Intersection-free Simulation of Stiff Materials*, ACM Trans. Graph. 41(4), 2022. Rigid
  bodies and rigid regions.
- **Additive CCD** — Li et al., *Codimensional Incremental Potential Contact*, ACM Trans.
  Graph. 40(4), 2021. The step limit between bodies.
- **Vertex corrections** — Li et al., *Convergent Incremental Potential Contact*,
  arXiv:2307.15908, 2023.
- **Taccel** — Li et al., *Taccel: Scaling Up Vision-based Tactile Robotics via
  High-performance GPU Simulation*, arXiv:2504.12908, 2025. The Model + Integrator shape.
- **Warp** — Macklin, *Warp: A High-performance Python Framework for GPU Simulation and
  Graphics*, NVIDIA GTC 2022. https://github.com/NVIDIA/warp

## Licence

MIT.
