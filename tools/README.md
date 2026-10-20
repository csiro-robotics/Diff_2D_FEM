# Development tools

Not part of the installed package. Run from anywhere; each finds the package relative to
its own location.

| script | what it is for |
|---|---|
| `measure_validation.py` | prints the actual numbers behind the README's validation table, at the package defaults |
| `measure_contact.py` | prints the numbers behind the README's table of contact between bodies, for `"ipc"` and for `"node"`, the penalty contact before it |
| `study_iterations.py` | iterations per step vs the acoustic CFL number (the basis of `max_iter=None`), and where that stops holding: steps of 10–40 ms |
| `benchmark_envs.py` | throughput against the number of environments, 1 to 4096, for three scenes (elasticity only; a strip on the ground; two pads lifting a rigid block) — the README's throughput table |
| `study_barrier_width.py` | how a crawling gait depends on `dhat` (the basis of the 1/8-pixel default) |
