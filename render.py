"""A live window on the simulation — the scene drawn while it runs.

    renderer = D.Renderer(model)                 # after model.finalize()
    for step in range(n):
        ...command the actuators...
        integ.simulate(model, dt)
        renderer.begin_frame(model.time)
        renderer.render()
        renderer.end_frame()

The same three calls per step as in Taccel's examples, and the same window underneath:
Warp's OpenGL renderer (warp.render.OpenGLRenderer, which needs `pyglet`). The 2D scene is
drawn in the plane z = 0 and looked at straight on, so nothing is foreshortened.

    drag            move the view              F      fit the view to the bodies as they are now
    scroll, + / -   zoom                       R      back to the first view
    arrows          move the view              X      outline every triangle
    SPACE           pause (the loop waits)     I      show / hide frame rates
    ESC             close the window

Several environments are shown side by side:

    renderer = D.Renderer(model, envs="all")     # or envs=[0, 5, 9]

A single scene is advanced at some tens of steps per second, so with steps of a millisecond
or two what the window shows is slow motion. To watch a run at its true speed, keep the
states and replay them:

    frames = []
    for step in range(n):
        integ.simulate(model, dt); frames.append(model.get_x())
    renderer.play(frames, dt)                    # real time; speed=0.25 for a quarter of it

    renderer.hold()                              # at the end of a script: stay open until closed

This is the live counterpart of viz.py, which draws stills and writes animations with
matplotlib; both use the same colours.
"""
from __future__ import annotations
import ctypes
import time as _time
import numpy as np

from .core import objects as _objects
from .viz import PALETTE, RIGID, GROUND

CABLE, SPRING, KINEMATIC = "#C0392B", "#2E8B57", "#7A7F87"
FOV = 30.0                                  # degrees; any value does, the scene is flat


def _rgb(colour):
    c = colour.lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _clip(poly, c, n):
    """The part of a convex polygon on the side n.(p - c) <= 0 of a line (Sutherland-Hodgman)."""
    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        da, db = float(np.dot(n, a - c)), float(np.dot(n, b - c))
        if da <= 0.0:
            out.append(a)
        if (da < 0.0) != (db < 0.0) and da != db:
            out.append(a + (da / (da - db)) * (b - a))
    return np.array(out).reshape(-1, 2)


def _fan(poly):
    """Triangles of a convex polygon, as (points with the centroid first, indices), facing +z."""
    poly = np.asarray(poly, np.float64)
    area2 = np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if area2 < 0.0:
        poly = poly[::-1]
    n = len(poly)
    pts = np.vstack([poly.mean(0), poly])
    k = np.arange(n)
    return pts, np.stack([np.zeros(n, np.int64), 1 + k, 1 + (k + 1) % n], 1)


class Renderer:
    """A window that shows a Model while it is being simulated.

    model       a finalized Model.
    envs        the environments to show: an index, a list of them, or "all". Several are
                laid out in a grid of `columns` columns.
    bounds      ((xmin, ymin), (xmax, ymax)) of what one environment needs room for; default:
                its bodies at rest and its colliders. Give it when the bodies travel.
    width, height   of the window, in pixels.
    headless    no window on the screen; frames can still be read with pixels().
    """

    def __init__(self, model, envs=0, title="Diff_2D_FEM", width=1280, height=720, fps=60,
                 bounds=None, margin=0.15, columns=None, background=(1.0, 1.0, 1.0),
                 wireframe=False, info=True, headless=False):
        if model.me is None:
            raise RuntimeError("call model.finalize() before opening a Renderer")
        try:
            import warp.render as _wr
        except Exception as e:                                   # pragma: no cover
            raise ImportError("the live window needs Warp's OpenGL renderer (warp.render)") from e
        self.model = model
        self.title = title
        me = model.me
        rest = np.asarray(me.rest_pos, np.float64)
        if isinstance(envs, str):
            if envs != "all":
                raise ValueError('envs is an index, a list of indices, or "all"')
            envs = list(range(model.n_envs))
        self.envs = [int(e) for e in np.atleast_1d(envs)]
        if not self.envs or min(self.envs) < 0 or max(self.envs) >= model.n_envs:
            raise ValueError(f"envs must lie in 0..{model.n_envs - 1}")
        shown = [h for h in range(model.n_bodies) if model.body_env(h) in self.envs]
        if not shown:
            raise ValueError("the environments chosen hold no body")

        # ---- the room one environment needs, and where each one goes in the grid
        if bounds is None:
            nodes = np.concatenate([model._node_range[h] for h in shown])
            lo, hi = rest[nodes].min(0), rest[nodes].max(0)
            mid, reach = 0.5 * (lo + hi), 3.0 * float((hi - lo).max())
            for o in model.colliders:
                if o.kind != _objects.HALFSPACE:
                    b = _objects.bounds(o)
                    lo, hi = np.minimum(lo, [b[0], b[2]]), np.maximum(hi, [b[1], b[3]])
                else:                               # a ground or wall nearby: keep its surface in view
                    nrm = np.asarray(o.normal, np.float64)
                    gap = float(np.dot(nrm, mid - np.array([o.cx, o.cy])))
                    if abs(gap) < reach:
                        foot = mid - gap * nrm
                        lo, hi = np.minimum(lo, foot), np.maximum(hi, foot)
            for k in model._springs:                 # where a spring is tied to the world
                if k["b"] < 0 and k["env"] in self.envs:
                    lo, hi = np.minimum(lo, k["anchor"]), np.maximum(hi, k["anchor"])
        else:
            lo, hi = np.asarray(bounds[0], np.float64), np.asarray(bounds[1], np.float64)
        size = np.maximum(hi - lo, 1e-9)
        self._size = float(size.max())
        n = len(self.envs)
        cols = int(columns) if columns else int(np.ceil(np.sqrt(n * size[1] / size[0] * width / height)))
        cols = int(np.clip(cols, 1, n))
        pitch = size + 0.2 * self._size
        tile = {e: np.array([(k % cols) * pitch[0], -(k // cols) * pitch[1]]) for k, e in enumerate(self.envs)}
        self._tile, self._cell = tile, (lo - 0.5 * (pitch - size), hi + 0.5 * (pitch - size))
        offs = np.array(list(tile.values()))
        self._view0 = (lo + offs.min(0), hi + offs.max(0), float(margin))

        # ---- triangles, grouped by colour so that a batch of any size is a handful of meshes
        tri = np.asarray(me.tri)
        P = rest[tri]
        ccw = (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) \
            - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]) > 0
        tri = np.where(ccw[:, None], tri, tri[:, [0, 2, 1]])                 # all facing +z
        rigid = np.zeros(len(tri), bool)
        for a in model._affine:
            rigid[a["elems"]] = True
        rank, seen = [], {}                         # a body's place among those of its environment
        for h in range(model.n_bodies):
            e = model.body_env(h)
            rank.append(seen.get(e, 0)); seen[e] = rank[-1] + 1
        node_off = np.zeros((len(rest), 2))
        groups = {}
        for h in shown:
            node_off[model._node_range[h]] = tile[model.body_env(h)]
            el = model._tri_range[h]
            if model._bodies[h].get("kinematic"):
                groups.setdefault(KINEMATIC, []).append(el)
            else:
                groups.setdefault(PALETTE[rank[h] % len(PALETTE)], []).append(el[~rigid[el]])
                groups.setdefault(RIGID, []).append(el[rigid[el]])
        self._meshes = []
        for colour, els in groups.items():
            el = np.concatenate(els)
            if len(el) == 0:
                continue
            nodes, local = np.unique(tri[el], return_inverse=True)
            self._meshes.append(dict(name=f"bodies {colour}", colour=_rgb(colour), nodes=nodes,
                                     off=node_off[nodes], idx=local.reshape(-1, 3).astype(np.int32),
                                     pts=np.zeros((len(nodes), 3), np.float32)))

        # ---- cables and springs: line segments between nodes (or a node and a point of the world)
        seg = [np.stack([t["nodes"][:-1], t["nodes"][1:]], 1) for t in model._tendons
               if model.body_env(t["body"]) in self.envs]
        self._cables = np.concatenate(seg) if seg else np.zeros((0, 2), np.int64)
        springs = [k for k in model._springs if k["env"] in self.envs]
        self._springs = springs
        self._struts = np.array([k["length"] > 0.0 for k in springs], bool)    # a pin has no length to draw
        self._node_off = node_off
        self._collider_key = None

        # ---- the window
        centre, dist = self._frame_view(*self._view0, width / height)
        self._r = _wr.OpenGLRenderer(
            title=title, scaling=1.0, fps=fps, up_axis="Y", screen_width=int(width),
            screen_height=int(height), near_plane=0.02 * dist, far_plane=50.0 * dist,
            camera_fov=FOV, camera_pos=(centre[0], centre[1], dist), camera_front=(0.0, 0.0, -1.0),
            camera_up=(0.0, 1.0, 0.0), background_color=tuple(background), draw_grid=False,
            draw_sky=False, draw_axis=False, show_info=bool(info), render_wireframe=bool(wireframe),
            vsync=False, headless=bool(headless), enable_backface_culling=False,
            enable_mouse_interaction=False, enable_keyboard_interaction=True)
        self._closed, self._headless, self._last = False, bool(headless), None
        try:                                        # pyglet keeps "the last window was closed" for
            self._r.app.event_loop.has_exit = False     # good; a window opened after that starts afresh
        except Exception:
            pass
        self._home = (np.array(centre), float(dist))
        self._centre, self._dist = np.array(centre, np.float64), float(dist)
        self._time, self._steps, self._clock, self._rate = 0.0, 0, _time.perf_counter(), 0.0
        self._flat_light()
        self._controls(headless)

    # ------------------------------------------------------------------ the three calls
    def begin_frame(self, time=None):
        """Start a frame. `time` is the simulated time it shows [s] (model.time).

        Once the window has been closed, this and the two calls after it do nothing: the
        loop around them runs on to its end without a picture."""
        self._time = float(self.model.time if time is None else time)
        if self.is_running():
            self._r.begin_frame(self._time)

    def render(self, x=None):
        """Draw the scene: the model's current state, or the node positions x ((N, 2), as
        model.get_x() returns them) of an earlier one."""
        if not self.is_running():
            return
        x = np.asarray(self.model.get_x() if x is None else x, np.float64)
        self._last = x
        r = self._r
        self._draw_colliders()
        for m in self._meshes:
            m["pts"][:, :2] = x[m["nodes"]] + m["off"]
            r.render_mesh(m["name"], m["pts"], m["idx"], colors=m["colour"])
        thick = 0.004 * self._size
        if len(self._cables):
            v = np.zeros((len(x), 3), np.float32); v[:, :2] = x + self._node_off
            self._lines("cables", np.stack([v[self._cables[:, 0]], v[self._cables[:, 1]]], 1),
                        _rgb(CABLE), thick)
        if self._springs:
            a = np.array([x[k["a"]] + self._node_off[k["a"]] for k in self._springs])
            b = np.array([(x[k["b"]] if k["b"] >= 0 else k["anchor"]) + self._node_off[k["a"]]
                          for k in self._springs])
            ends = np.zeros((len(a), 2, 3), np.float32); ends[:, 0, :2] = a; ends[:, 1, :2] = b
            if self._struts.any():
                self._lines("springs", ends[self._struts], _rgb(SPRING), thick)
            pts = np.zeros((2 * len(a), 3), np.float32); pts[:len(a), :2] = a; pts[len(a):, :2] = b
            r.render_points("spring ends", pts, 2.0 * thick, colors=_rgb(SPRING))

    def end_frame(self):
        """Show the frame. While the window is paused (SPACE) this call waits, and so does
        the simulation loop around it."""
        if not self.is_running():
            return
        self._steps += 1
        now = _time.perf_counter()
        if now - self._clock > 0.5:
            self._rate, self._steps, self._clock = self._steps / (now - self._clock), 0, now
        if not self._r.paused:
            try:
                self._r.window.set_caption(f"{self.title}    t = {self._time:.3f} s    "
                                           f"{self._rate:.0f} frames/s")
            except Exception:
                pass
        self._r.end_frame()

    def draw(self, time=None, x=None):
        """begin_frame, render and end_frame in one call."""
        self.begin_frame(time); self.render(x); self.end_frame()

    def hold(self):
        """Keep the window open on what it shows now, until it is closed (ESC) — for the end
        of a script, where the window would otherwise go with the process. The view can still
        be moved. Returns at once for a window that is not on the screen."""
        while not self._headless and self._last is not None and self.is_running():
            self.draw(self._time, self._last)
            _time.sleep(1.0 / 60.0)

    # ------------------------------------------------------------------ playing a recording
    def play(self, frames, dt, speed=1.0, loop=False):
        """Replay saved states at the speed they happened: frames is a list of model.get_x()
        taken every dt seconds of simulated time. speed=0.5 plays at half speed. Returns when
        the recording ends or the window is closed."""
        frames = list(frames)
        while self.is_running():
            start = _time.perf_counter()
            for k, x in enumerate(frames):
                wait = start + k * dt / speed - _time.perf_counter()
                if wait > 0.0:
                    _time.sleep(wait)
                before = _time.perf_counter()
                self.draw(k * dt, x)
                if not self.is_running():
                    return
                if _time.perf_counter() - before > 0.25:      # it was paused: do not catch up
                    start = _time.perf_counter() - k * dt / speed
            if not loop:
                return

    # ------------------------------------------------------------------ the window
    def is_running(self):
        """False once the window has been closed."""
        if self._closed:
            return False
        try:
            if self._r.window.context is None:       # closed from the window itself (ESC, the X button)
                self._closed = True
        except Exception:
            self._closed = True
        return not self._closed

    @property
    def paused(self):
        return bool(self._r.paused)

    @paused.setter
    def paused(self, value):
        self._r.paused = bool(value)

    def fit(self, x=None):
        """Fit the view to the bodies as they are now (or at the positions x)."""
        x = np.asarray(self.model.get_x() if x is None else x, np.float64)
        nodes = np.concatenate([m["nodes"] for m in self._meshes])
        p = x[nodes] + self._node_off[nodes]
        self._look(*self._frame_view(p.min(0), p.max(0), self._view0[2], self._aspect()))

    def pixels(self):
        """The frame last shown, as an (height, width, 3) uint8 image."""
        import warp as wp
        r = self._r
        img = wp.zeros((r.screen_height, r.screen_width, 3), dtype=wp.float32, device=r._device)
        r.get_pixels(img, split_up_tiles=False, mode="rgb")
        return (np.clip(img.numpy(), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._r.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ internals
    def _lines(self, name, ends, colour, radius):
        """Line segments, ends (n, 2, 3)."""
        try:
            self._r._render_lines(name, ends, colour, radius)
        except AttributeError:                       # a Warp without that method: the public one
            self._r.render_line_list(name, ends.reshape(-1, 3), np.arange(2 * len(ends)), colour, radius)

    def _aspect(self):
        return self._r.screen_width / max(self._r.screen_height, 1)

    @staticmethod
    def _frame_view(lo, hi, margin, aspect):
        """Centre and camera distance that put the box lo..hi in a window of this aspect."""
        lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        span = np.maximum(hi - lo, 1e-9)
        tall = max(span[1], span[0] / aspect) * (1.0 + 2.0 * margin)
        return 0.5 * (lo + hi), 0.5 * tall / np.tan(np.deg2rad(0.5 * FOV))

    def _look(self, centre, dist):
        self._centre, self._dist = np.array(centre, np.float64), float(dist)
        r = self._r
        r.camera_near_plane, r.camera_far_plane = 0.02 * self._dist, 50.0 * self._dist
        r.update_projection_matrix()
        r.update_view_matrix(cam_pos=(self._centre[0], self._centre[1], self._dist),
                             cam_front=(0.0, 0.0, -1.0), cam_up=(0.0, 1.0, 0.0))

    def _per_pixel(self):
        """Metres of the scene that one pixel of the window covers."""
        return 2.0 * self._dist * np.tan(np.deg2rad(0.5 * FOV)) / max(self._r.screen_height, 1)

    def _flat_light(self):
        """Warp's shader lights a surface with 0.3 ambient + the cosine to the sun. With the
        sun 45.6 degrees off the view axis a surface facing the camera gets 0.3 + 0.7 = 1:
        every body shows exactly the colour it was given, and no highlight reaches the eye."""
        try:
            r = self._r
            gl = type(r).gl
            r._switch_context()
            gl.glUseProgram(r._shape_shader.id)
            loc = gl.glGetUniformLocation(r._shape_shader.id, ctypes.c_char_p(b"sunDirection"))
            s = np.sqrt(0.51 / 2.0)
            gl.glUniform3f(loc, s, s, 0.7)
            r._info_label.color = (60, 60, 60, 255)                      # readable on white
        except Exception:
            pass                                    # a different Warp: the colours come out dimmer

    def _controls(self, headless):
        if headless:
            return
        try:
            import pyglet
            key, r = pyglet.window.key, self._r

            def drag(x, y, dx, dy, buttons, modifiers):
                self._look(self._centre - self._per_pixel() * np.array([dx, dy]), self._dist)
                return pyglet.event.EVENT_HANDLED

            def scroll(x, y, sx, sy):
                # keep the point under the cursor where it is
                at = self._centre + self._per_pixel() * np.array([x - 0.5 * r.screen_width,
                                                                   y - 0.5 * r.screen_height])
                f = 0.85 ** sy
                self._look(at + (self._centre - at) * f, self._dist * f)
                return pyglet.event.EVENT_HANDLED

            def held(keys):
                step = 12.0 * self._per_pixel()
                move = np.array([float(keys[key.RIGHT]) - float(keys[key.LEFT]),
                                 float(keys[key.UP]) - float(keys[key.DOWN])])
                zoom = (float(keys[key.MINUS]) + float(keys[key.NUM_SUBTRACT])
                        - float(keys[key.EQUAL]) - float(keys[key.NUM_ADD]))
                if move.any() or zoom:
                    self._look(self._centre + step * move, self._dist * 1.03 ** zoom)
                return pyglet.event.EVENT_HANDLED                         # no fly-through keys

            def pressed(symbol, modifiers):
                if symbol == key.F:
                    self.fit(); return pyglet.event.EVENT_HANDLED
                if symbol == key.R:
                    self._look(*self._home); return pyglet.event.EVENT_HANDLED
                if symbol in (key.C, key.G, key.T, key.B, key.TAB):       # 3-D options: not here
                    return pyglet.event.EVENT_HANDLED

            r.window.on_mouse_drag, r.window.on_mouse_scroll = drag, scroll
            r.register_input_processor(held)
            r.register_key_press_callback(pressed)
        except Exception:
            pass

    def _draw_colliders(self):
        """The colliders as ONE mesh of fixed layout — 96 points round a shape, six round the
        solid side of a plane — so that moving one (model.set_collider_center) only moves
        points. (Replacing a mesh by one of another size goes through the renderer's removal
        of an instance, which corrupts its instance buffers when other meshes are present.)"""
        cols = self.model.colliders
        key = tuple((o.kind, o.cx, o.cy, o.rx, o.ry, o.rref) for o in cols)
        if key == self._collider_key or not cols:
            return
        self._collider_key = key
        if len(self.envs) == 1:                     # alone, a ground runs out of sight
            far = 1.0e3 * self._size
            lo, hi = self._home[0] - far, self._home[0] + far
        else:                                       # in a grid, it stays in its own cell
            lo, hi = self._cell
        box = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
        pts, idx, n = [], [], 0
        for off in self._tile.values():
            for o in cols:
                if o.kind == _objects.HALFSPACE:
                    poly = _clip(box, np.array([o.cx, o.cy]), np.asarray(o.normal, np.float64))
                    if len(poly) < 3:               # the plane misses the cell: nothing to see
                        poly = np.tile(box[:1], (6, 1))
                    poly = np.vstack([poly, np.tile(poly[-1:], (6 - len(poly), 1))])[:6]
                else:
                    poly = _objects.outline(o, 96)[:96]
                    poly = np.vstack([poly, np.tile(poly[-1:], (96 - len(poly), 1))])
                p, i = _fan(poly)
                pts.append(p + off); idx.append(i + n); n += len(p)
        p2 = np.concatenate(pts)
        p3 = np.zeros((len(p2), 3), np.float32); p3[:, :2] = p2; p3[:, 2] = -1e-3 * self._size
        self._r.render_mesh("colliders", p3, np.concatenate(idx).astype(np.int32), colors=_rgb(GROUND))
