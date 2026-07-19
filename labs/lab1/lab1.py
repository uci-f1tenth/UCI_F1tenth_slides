"""F1TENTH Lab 1 — Wall Following.

A four-part deck on how wall following actually works:

    1. What the car sees      — a real map (Levine Hall) and a real LiDAR fan
    2. Two beams              — deriving alpha, D, and the lookahead D'
    3. PID                    — error -> steering, with live side-by-side demos
    4. The full lap           — the finished controller lapping Levine

Render with:  uv run manimgl labs/lab1/lab1.py Lab1 -w

The map assets (levine.png / levine.yaml) are standard ROS occupancy maps.
Map ingestion and the centerline extraction are recycled from the warporacer
simulator (warporacer/track.py): skeletonize the free space and walk the
skeleton's longest closed loop.
"""

from collections import deque, namedtuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from contourpy import contour_generator
from PIL import Image
from scipy.ndimage import label as cc_label
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize

from manimlib import *

ASSETS = Path(__file__).parent
CAR_ASPECT = 1491 / 2700  # car_topview.png height / width; image faces +x

OCC_THRESH = 230          # occupancy image value at/above which a pixel is free
SMOOTH_WINDOW = 51        # savgol window (in map pixels) for centerline smoothing
BEAM_SEP = 45 * DEGREES   # theta: angular gap between the two beams we use

WALL_COLOR = GREY_A
SCAN_COLOR = RED
HIT_COLOR = "#00FFFF"
A_COLOR = TEAL            # the 45-degree beam
B_COLOR = YELLOW          # the 90-degree beam
THETA_COLOR = GOLD
ALPHA_COLOR = BLUE
D_COLOR = GREEN
DP_COLOR = GREEN_B
L_COLOR = MAROON_B
ERR_COLOR = RED
P_COLOR = YELLOW
I_COLOR = BLUE
D_TERM_COLOR = PURPLE
TRAIL_COLOR = GREEN


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def pad3(points_2d: np.ndarray) -> np.ndarray:
    return np.column_stack([points_2d, np.zeros(len(points_2d))])


def unit(angle: float) -> np.ndarray:
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def cross2(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def raycast(origin, angles, segments: np.ndarray, max_range: float = 20.0):
    """Distance from origin to the nearest wall along each angle.

    Solves ray/segment intersection for every (beam, wall segment) pair at
    once: origin + t*dir = a + s*(b - a), keeping the smallest valid t.
    """
    angles = np.atleast_1d(np.asarray(angles, float))
    p = np.asarray(origin, float)[:2]
    d = np.stack([np.cos(angles), np.sin(angles)], axis=1)[:, None]     # (N,1,2)
    a, v = segments[None, :, 0], segments[None, :, 1] - segments[None, :, 0]
    denom = cross2(d, v)                                                # (N,M)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = cross2(a - p, v) / denom
        s = cross2(a - p, d) / denom
    t = np.where((np.abs(denom) > 1e-12) & (t > 1e-9) & (s >= 0) & (s <= 1), t, np.inf)
    return np.minimum(t.min(axis=1), max_range)


# --- centerline extraction (after warporacer/track.py) ---------------------

ADJ = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _neighbors(skel, p):
    h, w = skel.shape
    return [
        (p[0] + dr, p[1] + dc)
        for dr, dc in ADJ
        if 0 <= p[0] + dr < h and 0 <= p[1] + dc < w and skel[p[0] + dr, p[1] + dc]
    ]


def _n_clusters(nbrs) -> int:
    """How many 8-connected clusters the pixels in nbrs form among themselves."""
    todo, count = set(nbrs), 0
    while todo:
        count += 1
        stack = [todo.pop()]
        while stack:
            u = stack.pop()
            for v in [v for v in todo if abs(u[0] - v[0]) <= 1 and abs(u[1] - v[1]) <= 1]:
                todo.remove(v)
                stack.append(v)
    return count


def _cycle_core(skel):
    """Erode the skeleton down to its cycles.

    Repeatedly delete any pixel whose neighbors form at most one connected
    cluster — tips, spur stems, and the 2x2 junction blobs skeletonize leaves
    behind.  A pixel a cycle threads through keeps two separate neighbor
    clusters (one per side), so closed loops survive untouched.
    """
    skel = skel.copy()
    work = deque(map(tuple, np.argwhere(skel)))
    while work:
        p = work.popleft()
        if not skel[p]:
            continue
        nbrs = _neighbors(skel, p)
        if _n_clusters(nbrs) <= 1:
            skel[p] = False
            work.extend(nbrs)
    return skel


def _walk_cycle(skel) -> np.ndarray:
    """Order a cycle's pixels by walking it; returns (N, 2) of (row, col)."""
    start = tuple(map(int, np.argwhere(skel)[0]))
    loop, prev = [start], None
    while True:
        q = next(q for q in _neighbors(skel, loop[-1]) if q != prev)
        if q == start:
            return np.array(loop)
        prev = loop[-1]
        loop.append(q)


class Track:
    """A ROS occupancy map (yaml + image) as scene-space geometry.

    Walls: threshold the image into free space, trace the free/occupied
    boundary, smooth it, keep the big loops.  Centerline: skeletonize the free
    space and walk its longest closed loop (the warporacer waypoint pipeline).
    Also derives a start pose on the centerline and keeps everything in one
    scene frame: centered, y up, `scale` scene units per meter.
    """

    def __init__(self, yaml_path: Path, scene_width: float = 12.0, scene_height: float = 7.2):
        meta = yaml.safe_load(yaml_path.read_text())
        self.image = np.asarray(Image.open(yaml_path.parent / meta["image"]).convert("L"))
        self.res = float(meta["resolution"])
        self.ox, self.oy = float(meta["origin"][0]), float(meta["origin"][1])
        self.h, self.w = self.image.shape
        free = self.image >= OCC_THRESH

        loops = []
        for line in contour_generator(z=free.astype(float)).lines(0.5):
            if len(line) < 40:  # sensor specks
                continue
            pts = savgol_filter(self.px_to_world(line[:-1, 0], line[:-1, 1]),
                                41, 2, axis=0, mode="wrap")
            loops.append(pts[:: max(1, len(pts) // 300)])
        cl = self.centerline_world(free)

        bounds = np.concatenate(loops)
        center = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
        span = np.ptp(bounds, axis=0)
        self.scale = min(scene_width / span[0], scene_height / span[1])  # units per meter
        self.loops = sorted(
            ((lp - center) * self.scale for lp in loops),
            key=lambda lp: -np.prod(np.ptp(lp, axis=0)),
        )
        self.segments = np.concatenate(
            [np.stack([lp, np.roll(lp, -1, axis=0)], axis=1) for lp in self.loops]
        )

        cl = (cl - center) * self.scale
        if cross2(cl, np.roll(cl, -1, axis=0)).sum() < 0:
            cl = cl[::-1]  # counterclockwise, so the wall we follow is on the left
        tangents = np.diff(cl, axis=0, append=cl[:1])
        angles = np.arctan2(tangents[:, 1], tangents[:, 0])
        k = np.argmin(np.hypot(cl[:, 0], cl[:, 1] - cl[:, 1].max()))  # mid north corridor
        self.start_pose = np.array([cl[k, 0], cl[k, 1], angles[k]])
        self.centerline = cl[:: max(1, len(cl) // 600)]

    def px_to_world(self, col, row) -> np.ndarray:
        return np.column_stack(
            [self.ox + col * self.res, self.oy + (self.h - 1 - row) * self.res]
        )

    def centerline_world(self, free) -> np.ndarray:
        # Keep only the main free region and fill small occupied specks first,
        # so the skeleton's one surviving cycle is the track loop and not a
        # ring around a piece of furniture.
        labels, _ = cc_label(free)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        free = labels == sizes.argmax()
        occ, _ = cc_label(~free)
        free = free | (np.bincount(occ.ravel()) < 1600)[occ]  # fill < 2x2 m
        core = _cycle_core(skeletonize(free))
        if not core.any():
            raise RuntimeError("no skeleton cycle; is the track a closed loop?")
        loop = _walk_cycle(core)
        return savgol_filter(self.px_to_world(loop[:, 1], loop[:, 0]),
                             SMOOTH_WINDOW, 3, axis=0, mode="wrap")

    def walls(self, color=WALL_COLOR, width: float = 2.5) -> VGroup:
        return VGroup(*(
            VMobject().set_points_as_corners(pad3(np.vstack([lp, lp[:1]]))).set_stroke(color, width)
            for lp in self.loops
        ))

    def meters(self, scene_dist: float) -> float:
        return scene_dist / self.scale


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------

Terms = namedtuple("Terms", "u p i d")


@dataclass
class PID:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    limit: float = 2.0
    integral: float = field(default=0.0, init=False)
    prev: float = field(default=None, init=False)

    def __call__(self, error: float, dt: float) -> Terms:
        self.integral += error * dt
        deriv = 0.0 if self.prev is None else (error - self.prev) / dt
        self.prev = error
        p, i, d = self.kp * error, self.ki * self.integral, self.kd * deriv
        return Terms(float(np.clip(p + i + d, -self.limit, self.limit)), p, i, d)


Control = namedtuple("Control", "steer speed alpha dist hit_a hit_b")


class WallFollower:
    """The lab's algorithm, verbatim: two beams -> alpha, D, D' -> PID -> drive.

    Follows the wall on the car's left.  Callable as a drive_car control:
    returns (steer, speed) and stashes the full picture in `self.last`.
    """

    def __init__(self, segments, target, lookahead=0.45,
                 v_fast=1.2, v_slow=0.5, pid: PID = None):
        self.segments = segments
        self.target = target
        self.lookahead = lookahead
        self.v_fast, self.v_slow = v_fast, v_slow
        self.pid = pid or PID(kp=6.0, kd=1.5, limit=5.0)
        self.last = None

    def __call__(self, pose, dt):
        x, y, psi = pose
        phi_b, phi_a = psi + PI / 2, psi + PI / 2 - BEAM_SEP
        b, a = raycast((x, y), [phi_b, phi_a], self.segments)
        alpha = np.arctan2(a * np.cos(BEAM_SEP) - b, a * np.sin(BEAM_SEP))
        dist = b * np.cos(alpha)
        future = dist + self.lookahead * np.sin(alpha)
        steer = self.pid(future - self.target, dt).u
        # the real lab's speed rule: ease off the throttle when steering hard
        speed = self.v_slow + (self.v_fast - self.v_slow) * np.exp(-((steer / 1.2) ** 2))
        origin = np.array([x, y, 0.0])
        self.last = Control(steer, speed, alpha, dist,
                            origin + a * unit(phi_a), origin + b * unit(phi_b))
        return steer, speed


# ---------------------------------------------------------------------------
# Cars
# ---------------------------------------------------------------------------

def make_car(length: float = 0.9) -> ImageMobject:
    car = ImageMobject(str(ASSETS / "car_topview.png"), height=length * CAR_ASPECT)
    car.pose = np.zeros(3)  # x, y, heading
    car.speed = 0.0
    car.base_points = car.get_points().copy()  # origin-centered quad, facing +x
    return car


def sync_car(car) -> ImageMobject:
    """Set the sprite's quad from `car.pose` absolutely, so the car always
    points where it is driving even if an animation touched it mid-play."""
    x, y, psi = car.pose
    c, s = np.cos(psi), np.sin(psi)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    car.set_points(car.base_points @ rot.T + [x, y, 0.0])
    return car


def place_car(car, x, y, psi=0.0) -> ImageMobject:
    car.pose[:] = x, y, psi
    return sync_car(car)


def drive_car(car, control, accel: float = None) -> ImageMobject:
    """Integrate a unicycle whose (steer, speed) come from control(pose, dt).
    With accel set, speed ramps toward the command.  Attach after any intro
    animation on the car, or the sim runs while the sprite is still fading in."""
    state = {"v": 0.0}

    def update(mob, dt):
        if dt <= 0:
            return
        dt = min(dt, 0.1)
        steer, v_cmd = control(mob.pose, dt)
        state["v"] = v_cmd if accel is None else min(state["v"] + accel * dt, v_cmd)
        mob.speed = state["v"]
        mob.pose[2] += steer * dt
        mob.pose[0] += state["v"] * np.cos(mob.pose[2]) * dt
        mob.pose[1] += state["v"] * np.sin(mob.pose[2]) * dt
        sync_car(mob)

    car.add_updater(update)
    return car


# ---------------------------------------------------------------------------
# Mobject helpers
# ---------------------------------------------------------------------------

def make_legend(entries: dict, font_size: int = 30) -> VGroup:
    rows = VGroup(*(
        VGroup(
            Line(ORIGIN, 0.45 * RIGHT).set_stroke(color, 4),
            TexText(label, font_size=font_size).set_color(color),
        ).arrange(RIGHT, buff=0.15)
        for label, color in entries.items()
    ))
    return rows.arrange(DOWN, aligned_edge=LEFT, buff=0.18)


def right_angle_mark(corner, dir1, dir2, size: float = 0.18) -> VMobject:
    d1, d2 = normalize(dir1), normalize(dir2)
    return VMobject().set_points_as_corners(
        [corner + d1 * size, corner + (d1 + d2) * size, corner + d2 * size]
    ).set_stroke(GREY_B, 2)


def brace_label(p1, p2, tex: str, color=WHITE, font_size: int = 40) -> VGroup:
    """Brace spanning p1 -> p2, on the right-hand side of that direction."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    direction = rotate_vector(normalize(p2 - p1), -PI / 2)
    brace = Brace(Line(p1, p2), direction)
    label = Tex(tex, font_size=font_size).set_color(color)
    brace.put_at_tip(label)
    return VGroup(brace, label)


def beam_labels(origin, phi_a, a_len, b_len, size=44,
                off=0.32, theta_r=1.05, frac=0.55, b_side=1) -> VGroup:
    """a / b / theta labels beside their beams: each label sits partway along
    its beam, pushed perpendicular to a clear side (b_side=-1 tucks b into the
    theta wedge when something else occupies its outside), with theta on the
    bisector just past the angle arc."""
    phi_b = phi_a + BEAM_SEP
    return VGroup(
        Tex("b", font_size=size).set_color(B_COLOR)
        .move_to(origin + frac * b_len * unit(phi_b) + off * unit(phi_b + b_side * PI / 2)),
        Tex("a", font_size=size).set_color(A_COLOR)
        .move_to(origin + frac * a_len * unit(phi_a) + off * unit(phi_a - PI / 2)),
        Tex(r"\theta", font_size=round(0.9 * size)).set_color(THETA_COLOR)
        .move_to(origin + theta_r * unit(phi_a + BEAM_SEP / 2)),
    )


def lidar_fan(origin, psi, segments, n=61, fov=270 * DEGREES,
              stroke_width=1.2, opacity=0.7, max_range=20.0):
    """A fan of raycast beams plus their hit points."""
    angles = psi + np.linspace(-fov / 2, fov / 2, n)
    dists = raycast(origin, angles, segments, max_range)
    origin = np.array([*origin[:2], 0.0])
    ends = [origin + d * unit(ang) for d, ang in zip(dists, angles)]
    lines = VGroup(*(
        Line(origin, end).set_stroke(SCAN_COLOR, stroke_width, opacity) for end in ends
    ))
    return lines, np.array(ends)


class StreamChart(VGroup):
    """Axes plus named curves that stream in point-by-point from an updater."""

    def __init__(self, series: dict, x_max=9.0, y_range=(-3, 3, 1),
                 width=11.0, height=4.6, **kwargs):
        self.axes = Axes((0, x_max, 1), y_range, width=width, height=height)
        self.axes.add_coordinate_labels(font_size=18, num_decimal_places=0)
        self.curves = {
            name: VMobject().set_stroke(color, 3) for name, color in series.items()
        }
        self.samples = {name: [] for name in series}
        self.t = 0.0
        self.x_max, self.y_lo, self.y_hi = x_max, y_range[0], y_range[1]
        # cache the affine coords->point map so streaming stays cheap
        o = self.axes.c2p(0, 0)
        self._o, self._ux, self._uy = o, self.axes.c2p(1, 0) - o, self.axes.c2p(0, 1) - o
        super().__init__(self.axes, *self.curves.values(), **kwargs)

    def record(self, dt: float, **values):
        self.t += dt
        if self.t > self.x_max:
            return
        for name, value in values.items():
            self.samples[name].append((self.t, float(np.clip(value, self.y_lo, self.y_hi))))
            if len(self.samples[name]) >= 2:
                ts, vs = np.array(self.samples[name]).T
                points = self._o + ts[:, None] * self._ux + vs[:, None] * self._uy
                self.curves[name].set_points_as_corners(points)


# ---------------------------------------------------------------------------
# The deck
# ---------------------------------------------------------------------------

class Lab1(Scene):
    def construct(self):
        self.track = Track(ASSETS / "levine.yaml")
        self.start_pose = self.track.start_pose
        # hold whatever gap the centerline itself has to the left wall
        x, y, psi = self.start_pose
        self.follow_target = raycast((x, y), psi + PI / 2, self.track.segments)[0]
        self._caption = None

        self.intro()
        self.chapter_sensor()
        self.chapter_geometry()
        self.chapter_pid()
        self.chapter_lap()
        self.outro()

    # -- shared furniture ---------------------------------------------------

    def chapter_card(self, index: int, title: str):
        kicker = TexText(f"Part {index} of 4", font_size=34).set_color(GREY_B)
        head = TexText(title, font_size=64)
        rule = Line(LEFT, RIGHT).set_width(head.get_width() + 0.6).set_stroke(GREY_C, 2)
        card = VGroup(kicker, head, rule).arrange(DOWN, buff=0.35)
        self.play(FadeIn(kicker), Write(head), ShowCreation(rule), run_time=1.4)
        self.wait(1.2)
        self.play(FadeOut(card))

    def swap_caption(self, text: str, fix: bool = False, edge=DOWN):
        tex = TexText(text, font_size=34)
        cap = VGroup(
            BackgroundRectangle(tex, buff=0.12, fill_opacity=0.55), tex
        ).to_edge(edge, buff=0.3)
        if fix:
            cap.fix_in_frame()
        anims = [FadeIn(cap)]
        if self._caption is not None:
            anims.append(FadeOut(self._caption))
        self.play(*anims, run_time=0.8)
        self._caption = cap

    def clear_caption(self):
        if self._caption is not None:
            self.play(FadeOut(self._caption), run_time=0.6)
            self._caption = None

    def goal_line(self, width: float = 2.5, opacity: float = 1.0) -> DashedVMobject:
        cl = self.track.centerline
        return DashedVMobject(
            VMobject().set_points_as_corners(pad3(np.vstack([cl, cl[:1]]))),
            num_dashes=140,
        ).set_stroke(DP_COLOR, width, opacity)

    # -- title + outline ----------------------------------------------------

    def intro(self):
        logo = ImageMobject(str(ASSETS / "F1TenthUCIlogo.png"), height=1.5).shift(2.6 * UP)
        title = TexText("F1TENTH Lab 1", font_size=72).shift(0.7 * UP)
        subtitle = TexText("Wall Following", font_size=96).set_color(BLUE_B).shift(0.6 * DOWN)

        cameo = drive_car(place_car(make_car(0.8), -8.2, -2.9), lambda p, dt: (0.0, 2.7))
        tail = TracingTail(cameo, time_traced=4.0, stroke_color=TRAIL_COLOR, stroke_width=(0, 4))
        self.add(cameo, tail)

        self.play(FadeIn(logo, scale=1.1), Write(title))
        self.play(Write(subtitle))
        self.wait(3.5)
        cameo.clear_updaters()
        tail.clear_updaters()
        self.play(FadeOut(logo), FadeOut(title), FadeOut(subtitle),
                  FadeOut(cameo), FadeOut(tail))

        header = TexText("Today", font_size=64).shift(2.4 * UP)
        items = VGroup(
            TexText("1. What the car sees --- LiDAR"),
            TexText("2. Two beams $\\to$ distance and angle"),
            TexText("3. PID --- from error to steering"),
            TexText("4. A full lap of Levine Hall"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.45).shift(0.5 * DOWN)
        self.play(Write(header))
        self.play(LaggedStartMap(FadeIn, items, shift=0.3 * UP, lag_ratio=0.25))
        self.wait(2)
        self.play(FadeOut(header), FadeOut(items))

    # -- part 1: the sensor -------------------------------------------------

    def chapter_sensor(self):
        self.chapter_card(1, "What the car sees")
        track = self.track

        walls = track.walls()
        self.play(Write(walls, run_time=2.5))
        self.swap_caption("Levine Hall --- the classic wall-following course.")
        self.wait(1.5)

        goal = self.goal_line()
        self.play(ShowCreation(goal, run_time=2))
        self.swap_caption(
            "The goal: lap the hallway, holding the middle of the corridor."
        )
        self.wait(2)

        car = place_car(make_car(0.32), *self.start_pose)
        self.play(FadeIn(car), FadeOut(goal))
        self.clear_caption()  # world-anchored captions must not survive the zoom

        frame = self.camera.frame
        frame.save_state()
        self.play(frame.animate.scale(0.38).move_to(car.get_center() + 0.3 * LEFT), run_time=2)

        fan, hits = lidar_fan(self.start_pose, self.start_pose[2], track.segments)
        dots = GlowDots(hits, color=HIT_COLOR, radius=0.06)
        self.play(
            LaggedStartMap(ShowCreation, fan, lag_ratio=0.02, run_time=2),
            FadeIn(dots, run_time=2),
        )
        self.swap_caption(
            "One 2D LiDAR: $\\sim$1080 range readings across $270^\\circ$, 40 times a second.",
            fix=True,
        )
        self.wait(2)

        psi = self.start_pose[2]
        phi_b, phi_a = psi + PI / 2, psi + PI / 2 - BEAM_SEP
        b_len, a_len = raycast(self.start_pose, [phi_b, phi_a], track.segments)
        origin = np.array([self.start_pose[0], self.start_pose[1], 0.0])
        beam_b = Line(origin, origin + b_len * unit(phi_b)).set_stroke(B_COLOR, 2)
        beam_a = Line(origin, origin + a_len * unit(phi_a)).set_stroke(A_COLOR, 2)
        theta_arc = Arc(phi_a, BEAM_SEP, radius=0.12, arc_center=origin).set_stroke(THETA_COLOR, 1.5)
        labels = beam_labels(origin, phi_a, a_len, b_len,
                             size=12, off=0.08, theta_r=0.17, frac=0.75)
        self.swap_caption(
            "Wall following reads just two of them: $b$ at $90^\\circ$ left,"
            " $a$ at $\\theta = 45^\\circ$ ahead of it.",
            fix=True,
        )
        self.play(
            fan.animate.set_stroke(opacity=0.12), dots.animate.set_opacity(0.25),
            ShowCreation(beam_b), ShowCreation(beam_a),
        )
        self.play(ShowCreation(theta_arc), Write(labels))
        self.wait(2.5)

        self.clear_caption()
        self.play(
            Restore(frame),
            *map(FadeOut, (fan, dots, beam_a, beam_b, theta_arc, labels, car, walls)),
            run_time=1.6,
        )

    # -- part 2: the geometry -----------------------------------------------

    def chapter_geometry(self):
        self.chapter_card(2, "Two beams tell you everything")

        WALL_Y, LOOK = 2.2, 1.8
        C = np.array([-2.2, -1.3, 0.0])
        alpha_tr = ValueTracker(12 * DEGREES)

        def geo(alpha):
            psi = -alpha
            w = WALL_Y - C[1]
            phi_b, phi_a = psi + PI / 2, psi + PI / 2 - BEAM_SEP
            b, a = w / np.sin(phi_b), w / np.sin(phi_a)
            return dict(
                psi=psi, a=a, b=b,
                A=C + a * unit(phi_a), B=C + b * unit(phi_b),
                F=C + a * np.cos(BEAM_SEP) * unit(phi_b),
                G=C + LOOK * unit(psi),
                phi_a=phi_a, phi_b=phi_b,
            )

        def beams(alpha):
            g = geo(alpha)
            return VGroup(
                Line(C, g["B"]).set_stroke(B_COLOR, 3),
                Line(C, g["A"]).set_stroke(A_COLOR, 3),
                Arc(g["phi_a"], BEAM_SEP, radius=0.7, arc_center=C).set_stroke(THETA_COLOR, 2.5),
                beam_labels(C, g["phi_a"], g["a"], g["b"], b_side=-1),  # D line owns b's left
            )

        def dist_now(alpha):
            g = geo(alpha)
            d_foot = np.array([C[0], WALL_Y, 0])
            return VGroup(
                DashedLine(C, d_foot).set_stroke(D_COLOR, 3),
                right_angle_mark(d_foot, DOWN, RIGHT),
                DashedLine(C, C + 2.0 * RIGHT).set_stroke(GREY_C, 1.5),
                Arc(0, g["psi"], radius=1.3, arc_center=C).set_stroke(ALPHA_COLOR, 3),
                Tex("D", font_size=44).set_color(D_COLOR)
                .move_to([C[0] - 0.35, (C[1] + WALL_Y) / 2, 0]),
                Tex(r"\alpha", font_size=40).set_color(ALPHA_COLOR)
                .move_to(C + 1.6 * unit(g["psi"] / 2)),
            )

        def dist_future(alpha):
            g = geo(alpha)
            g_foot = np.array([g["G"][0], WALL_Y, 0])
            return VGroup(
                DashedLine(C, g["G"]).set_stroke(GREY_B, 2),
                DashedLine(g["G"], g_foot).set_stroke(DP_COLOR, 3),
                Tex("D'", font_size=44).set_color(DP_COLOR)
                .move_to([g["G"][0] + 0.4, (g["G"][1] + WALL_Y) / 2, 0]),
            )

        wall = Line([-6.4, WALL_Y, 0], [6.4, WALL_Y, 0]).set_stroke(WALL_COLOR, 5)
        ticks = VGroup(*(
            Line([x, WALL_Y, 0], [x + 0.22, WALL_Y + 0.22, 0]).set_stroke(GREY_C, 2)
            for x in np.arange(-6.3, 6.2, 0.45)
        ))
        g0 = geo(alpha_tr.get_value())
        car = place_car(make_car(1.5), *C[:2], g0["psi"])
        ghost = place_car(make_car(1.5), *g0["G"][:2], g0["psi"]).set_opacity(0.35)
        beams0 = beams(alpha_tr.get_value())

        self.play(ShowCreation(wall), ShowCreation(ticks), FadeIn(car))
        self.swap_caption("Follow the wall on the left.  What can two beams tell us?")
        self.play(*map(ShowCreation, beams0[:3]))
        self.play(Write(beams0[3]))
        self.wait(1.5)

        # drop a perpendicular from a's hit point onto the b beam
        foot = DashedLine(g0["B"], g0["F"]).set_stroke(GREY_B, 2)
        drop = Line(g0["A"], g0["F"]).set_stroke(L_COLOR, 3)
        corner = right_angle_mark(
            g0["F"], normalize(g0["A"] - g0["F"]), normalize(C - g0["F"])
        )
        self.swap_caption(
            "Project $a$'s hit point onto the $b$ beam --- a right triangle appears."
        )
        self.play(ShowCreation(foot), ShowCreation(drop), ShowCreation(corner))

        brace_cf = brace_label(g0["F"], C, r"a\cos\theta", A_COLOR)
        self.play(TransformFromCopy(beams0[1], brace_cf))
        self.wait(1.2)
        brace_bf = brace_label(g0["F"], g0["B"], r"a\cos\theta - b", WHITE)
        self.play(TransformMatchingShapes(brace_cf, brace_bf))
        brace_fa = brace_label(g0["A"], g0["F"], r"a\sin\theta", A_COLOR)
        self.play(TransformFromCopy(beams0[1], brace_fa))
        self.wait(1.5)

        # alpha: at the triangle, and equally between heading and wall
        ang_AF = angle_of_vector(g0["F"] - g0["A"])
        alpha_arc_A = Arc(ang_AF, PI - ang_AF, radius=0.85, arc_center=g0["A"]).set_stroke(ALPHA_COLOR, 3)
        alpha_label_A = Tex(r"\alpha", font_size=40).set_color(ALPHA_COLOR).move_to(
            g0["A"] + 1.15 * unit((ang_AF + PI) / 2)
        )
        self.play(ShowCreation(alpha_arc_A), Write(alpha_label_A))
        self.swap_caption(
            r"That angle $\alpha$ is how far the car's heading is off from the wall."
        )

        # glyph indices: a c o s theta - b  /  a s i n theta
        num = Tex("a", r"\cos\theta", "-", "b", font_size=44)
        num[0].set_color(A_COLOR), num[4].set_color(THETA_COLOR), num[6].set_color(B_COLOR)
        den = Tex("a", r"\sin\theta", font_size=44)
        den[0].set_color(A_COLOR), den[4].set_color(THETA_COLOR)
        bar = Line(LEFT, RIGHT).set_width(num.get_width() + 0.15).set_stroke(WHITE, 2)
        frac = VGroup(num, bar, den).arrange(DOWN, buff=0.13)
        lhs = Tex(r"\alpha", "=", r"\arctan", font_size=44)
        lhs[0].set_color(ALPHA_COLOR)
        alpha_eq = VGroup(lhs, frac).arrange(RIGHT, buff=0.18).to_corner(DL, buff=0.5).shift(0.55 * UP)
        self.play(Write(alpha_eq))
        self.wait(2)

        self.play(*map(FadeOut, (foot, drop, corner, brace_bf, brace_fa,
                                 alpha_arc_A, alpha_label_A)))

        # D and the lookahead distance D'
        now0 = dist_now(alpha_tr.get_value())
        self.play(*map(ShowCreation, now0[:4]))
        self.play(Write(now0[4]), Write(now0[5]))
        d_eq = Tex("D", "=", "b", r"\cos\alpha", font_size=44)  # D = b c o s alpha
        d_eq[0].set_color(D_COLOR), d_eq[2].set_color(B_COLOR), d_eq[6].set_color(ALPHA_COLOR)
        d_eq.next_to(alpha_eq, UP, buff=0.4, aligned_edge=LEFT)
        self.swap_caption("Project $b$ the same way: the true distance to the wall.")
        self.play(Write(d_eq))
        self.wait(1.5)

        self.swap_caption(
            "One more step: steer for where the car \\emph{will} be, not where it is."
        )
        fut0 = dist_future(alpha_tr.get_value())
        self.play(FadeIn(ghost), ShowCreation(fut0[0]), ShowCreation(fut0[1]))
        brace_l = brace_label(C, g0["G"], "L", L_COLOR)
        self.play(GrowFromCenter(brace_l), Write(fut0[2]))
        dp_eq = Tex("D'", "=", "D", "+", "L", r"\sin\alpha", font_size=44)  # D ' = D + L s i n alpha
        dp_eq[0:2].set_color(DP_COLOR), dp_eq[3].set_color(D_COLOR)
        dp_eq[5].set_color(L_COLOR), dp_eq[9].set_color(ALPHA_COLOR)
        dp_eq.next_to(d_eq, UP, buff=0.4, aligned_edge=LEFT)
        self.play(Write(dp_eq))
        self.wait(2)

        # make it live: sweep alpha and watch every quantity respond
        readouts = VGroup(*(
            VGroup(Tex(sym + "=", font_size=38).set_color(color),
                   DecimalNumber(0, font_size=38, num_decimal_places=1,
                                 include_sign=sym == r"\alpha", unit=unit_str).set_color(color))
            .arrange(RIGHT, buff=0.12)
            for sym, color, unit_str in [
                (r"\alpha", ALPHA_COLOR, r"^\circ"), ("a", A_COLOR, None),
                ("b", B_COLOR, None), ("D", D_COLOR, None), ("D'", DP_COLOR, None),
            ]
        )).arrange(RIGHT, buff=0.75).to_edge(UP, buff=0.35)
        values = [
            lambda: np.degrees(alpha_tr.get_value()),
            lambda: geo(alpha_tr.get_value())["a"],
            lambda: geo(alpha_tr.get_value())["b"],
            lambda: geo(alpha_tr.get_value())["b"] * np.cos(alpha_tr.get_value()),
            lambda: geo(alpha_tr.get_value())["b"] * np.cos(alpha_tr.get_value())
            + LOOK * np.sin(alpha_tr.get_value()),
        ]
        for (_, dec), fn in zip(readouts, values):
            f_always(dec.set_value, fn)

        live = always_redraw(lambda: VGroup(
            beams(alpha_tr.get_value()),
            dist_now(alpha_tr.get_value()),
            dist_future(alpha_tr.get_value()),
        ))
        self.remove(*beams0, *now0, *fut0)
        self.add(live)
        car.add_updater(lambda m: place_car(m, *C[:2], geo(alpha_tr.get_value())["psi"]))
        ghost.add_updater(lambda m: place_car(
            m, *geo(alpha_tr.get_value())["G"][:2], geo(alpha_tr.get_value())["psi"]
        ))
        self.play(FadeIn(readouts), FadeOut(brace_l))
        self.swap_caption(
            r"Two beams $\to$ both your spacing ($D'$) and your heading error ($\alpha$)."
        )
        for target in (-20 * DEGREES, 20 * DEGREES, 12 * DEGREES):
            self.play(alpha_tr.animate.set_value(target), run_time=2.2)
        self.wait(1)

        for mob in (car, ghost, live, readouts):
            mob.clear_updaters()
        self.clear_caption()
        self.play(*map(FadeOut, (wall, ticks, car, ghost, live, readouts,
                                 alpha_eq, d_eq, dp_eq)))

    # -- part 3: PID --------------------------------------------------------

    def chapter_pid(self):
        self.chapter_card(3, "From error to steering")

        err_eq = Tex("e", "=", "D'", "-", r"D_{\text{target}}", font_size=64)  # e = D ' - D t a r g e t
        err_eq[0].set_color(ERR_COLOR), err_eq[2:4].set_color(DP_COLOR)
        self.play(Write(err_eq))
        self.swap_caption(
            "Too far from the wall: $e>0$, steer left toward it."
            "  Too close: $e<0$, steer right."
        )
        self.wait(2)
        self.play(err_eq.animate.scale(0.6).to_edge(UP, buff=0.4))

        def box(label):
            text = TexText(label, font_size=32)
            frame = RoundedRectangle(
                corner_radius=0.15,
                width=text.get_width() + 0.55, height=max(text.get_height() + 0.5, 0.8),
            ).set_stroke(GREY_A, 2)
            return VGroup(frame, text.move_to(frame))

        blocks = VGroup(
            box("LiDAR"), box("geometry\\\\$\\alpha,\\ D'$"), box("PID"), box("car")
        ).arrange(RIGHT, buff=1.25)
        arrows = VGroup(*(
            Arrow(a.get_right(), b.get_left(), buff=0.1)
            for a, b in zip(blocks[:-1], blocks[1:])
        ))
        arrow_tags = VGroup(
            Tex("e", font_size=34).set_color(ERR_COLOR).next_to(arrows[1], UP, buff=0.1),
            TexText("steering", font_size=26).next_to(arrows[2], UP, buff=0.1),
        )
        loop_back = Arrow(
            blocks[-1].get_bottom(), blocks[0].get_bottom(), path_arc=-1.1, buff=0.15
        )
        loop_tag = TexText("repeat every scan", font_size=26).set_color(GREY_B)
        loop_tag.next_to(loop_back, DOWN, buff=0.15)
        self.play(LaggedStartMap(FadeIn, blocks, lag_ratio=0.2),
                  LaggedStartMap(GrowArrow, arrows, lag_ratio=0.2))
        self.play(Write(arrow_tags), GrowArrow(loop_back), FadeIn(loop_tag))
        self.wait(2)
        self.play(*map(FadeOut, (blocks, arrows, arrow_tags, loop_back, loop_tag)))

        pid_eq = Tex(
            "u", "=", r"K_p\, e", "+", r"K_i \int e\, dt", "+", r"K_d\, \frac{de}{dt}",
            font_size=60,
        )
        terms = [
            pid_eq.select_part(piece)
            for piece in (r"K_p\, e", r"K_i \int e\, dt", r"K_d\, \frac{de}{dt}")
        ]
        for term, color in zip(terms, (P_COLOR, I_COLOR, D_TERM_COLOR)):
            term.set_color(color)
        notes = VGroup(
            TexText("P: push harder the further off you are", font_size=36).set_color(P_COLOR),
            TexText("I: remember error that refuses to go away", font_size=36).set_color(I_COLOR),
            TexText("D: resist fast swings --- damping", font_size=36).set_color(D_TERM_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(1.6 * DOWN)
        self.play(Write(pid_eq))
        self.wait(1)
        self.play(pid_eq.animate.shift(1.2 * UP))
        for note, term in zip(notes, terms):
            self.play(FadeIn(note, shift=0.25 * UP), Indicate(term, color=note.get_color()))
            self.wait(0.8)
        self.wait(1)
        self.play(FadeOut(pid_eq), FadeOut(notes), FadeOut(err_eq))

        # demo A: P alone oscillates, adding D damps it
        self.line_demo(
            specs=[
                (PID(kp=1.5, limit=1.5), 0.0, RED, "$K_p$ only"),
                (PID(kp=2.0, kd=1.2, limit=1.5), 0.0, GREEN, "$K_p + K_d$"),
            ],
            caption="P alone forever overshoots.  The D term sees the swing coming"
                    " and damps it.",
            stop_x=5.7,
        )

        # demo B: a steady disturbance defeats PD; I erases it
        self.line_demo(
            specs=[
                (PID(kp=2.0, kd=1.2, limit=1.5), -0.7, ORANGE, "$K_p + K_d$"),
                (PID(kp=2.0, kd=1.2, ki=1.0, limit=1.5), -0.7, GREEN, "$K_p + K_i + K_d$"),
            ],
            caption="Now the car pulls right --- a bent wheel, a sloped floor.",
            stop_x=4.5,
            annotate_ss_error=True,
        )

        # demo C: watch the three terms do their jobs
        chart = StreamChart(
            {"error": ERR_COLOR, "P": P_COLOR, "I": I_COLOR, "D": D_TERM_COLOR},
            x_max=9.0, y_range=(-3, 3, 1), width=11.5, height=4.4,
        ).shift(0.9 * UP)
        legend = make_legend(
            {"error": ERR_COLOR, "P term": P_COLOR, "I term": I_COLOR, "D term": D_TERM_COLOR},
            font_size=26,
        ).to_corner(UR, buff=0.4)
        target_y = -2.7
        target = DashedLine([-6.3, target_y, 0], [6.3, target_y, 0]).set_stroke(GREY_B, 2)
        car = place_car(make_car(0.55), -5.7, target_y - 1.2)
        pid = PID(kp=2.0, kd=1.2, ki=1.0, limit=1.5)

        def control(p, dt):
            terms = pid(target_y - p[1], dt)
            chart.record(dt, error=target_y - p[1], P=terms.p, I=terms.i, D=terms.d)
            return terms.u - 0.7, 1.25

        self.play(FadeIn(chart), FadeIn(legend), ShowCreation(target), FadeIn(car))
        self.swap_caption(
            "Same disturbed car: watch I slowly take over the correction.", edge=UP
        )
        drive_car(car, control)
        self.wait_until(lambda: car.pose[0] > 5.5, max_time=9.5)
        car.clear_updaters()
        self.wait(1.5)
        self.clear_caption()
        self.play(*map(FadeOut, (chart, legend, target, car)))

    def line_demo(self, specs, caption, stop_x, annotate_ss_error=False):
        """Cars chasing a dashed line, side by side, one PID each."""
        target_y = 0.6
        target = DashedLine([-6.3, target_y, 0], [6.3, target_y, 0]).set_stroke(GREY_B, 2)
        legend = make_legend(
            {label: color for _, _, color, label in specs}, font_size=30
        ).to_corner(UL, buff=0.4)

        cars = [place_car(make_car(0.55), -5.8, target_y - 1.2) for _ in specs]
        tails = [
            TracingTail(car, time_traced=12, stroke_color=color, stroke_width=(0, 4))
            for car, (_, _, color, _) in zip(cars, specs)
        ]

        self.play(ShowCreation(target), FadeIn(legend), *map(FadeIn, cars))
        self.add(*tails)
        for car, (pid, bias, _, _) in zip(cars, specs):
            drive_car(car, lambda p, dt, pid=pid, bias=bias: (pid(target_y - p[1], dt).u + bias, 1.5))
        self.swap_caption(caption)
        self.wait_until(lambda: cars[0].pose[0] > stop_x, max_time=9)
        for mob in (*cars, *tails):
            mob.clear_updaters()

        extras = []
        if annotate_ss_error:
            x, y = cars[0].pose[:2]
            gap = brace_label([x, target_y, 0], [x, y, 0], r"\text{steady-state error}",
                              ORANGE, font_size=30)
            extras.append(gap)
            self.swap_caption("P and D settle for a standoff --- only I keeps pushing"
                              " until the offset is gone.")
            self.play(GrowFromCenter(gap))
        self.wait(2)
        self.clear_caption()
        self.play(*map(FadeOut, (target, legend, *cars, *tails, *extras)))

    # -- part 4: the lap ----------------------------------------------------

    def chapter_lap(self):
        self.chapter_card(4, "The full lap")

        code = Code(
            "def on_scan(ranges):                  # every LiDAR scan, ~40 Hz\n"
            "    b = ranges[deg(90)]               # left beam\n"
            "    a = ranges[deg(45)]               # 45 deg ahead of it\n"
            "    alpha = atan2(a*cos(TH) - b, a*sin(TH))\n"
            "    D  = b*cos(alpha)                 # distance to wall now\n"
            "    Dp = D + L*sin(alpha)             # ...a moment from now\n"
            "    e  = Dp - D_TARGET\n"
            "    steer = Kp*e + Ki*integral(e) + Kd*derivative(e)\n"
            "    speed = FAST if abs(steer) < 0.3 else SLOW\n"
            "    drive(steer, speed)",
            font_size=28,
        )
        self.play(Write(code))
        self.swap_caption("The whole lab is ten lines.  Here it is on the real map.")
        self.wait(3)
        self.clear_caption()
        self.play(FadeOut(code))

        track = self.track
        walls = track.walls()
        goal = self.goal_line(1.8, 0.6)
        self.play(Write(walls, run_time=2), ShowCreation(goal, run_time=2))

        car = place_car(make_car(0.32), *self.start_pose)
        tail = TracingTail(car, time_traced=26, stroke_color=TRAIL_COLOR, stroke_width=(0, 4))
        follower = WallFollower(track.segments, self.follow_target)

        beam_b = Line(ORIGIN, RIGHT).set_stroke(B_COLOR, 2.5)
        beam_a = Line(ORIGIN, RIGHT).set_stroke(A_COLOR, 2.5)
        hit_b = GlowDot(color=HIT_COLOR, radius=0.12)
        hit_a = GlowDot(color=HIT_COLOR, radius=0.12)
        fan = VGroup(*(Line(ORIGIN, RIGHT).set_stroke(SCAN_COLOR, 1, 0.3) for _ in range(15)))

        d_tr, alpha_tr, steer_tr, speed_tr = (ValueTracker(0) for _ in range(4))
        lap = {"wound": 0.0, "prev": np.arctan2(car.pose[1], car.pose[0])}

        def scenery(mob, dt):
            ctl = follower.last
            origin = np.array([mob.pose[0], mob.pose[1], 0.0])
            beam_b.put_start_and_end_on(origin, ctl.hit_b)
            beam_a.put_start_and_end_on(origin, ctl.hit_a)
            hit_b.move_to(ctl.hit_b), hit_a.move_to(ctl.hit_a)
            angles = mob.pose[2] + np.linspace(-135 * DEGREES, 135 * DEGREES, len(fan))
            for line, ang, dist in zip(fan, angles, raycast(mob.pose, angles, track.segments)):
                line.put_start_and_end_on(origin, origin + dist * unit(ang))
            d_tr.set_value(track.meters(ctl.dist))
            alpha_tr.set_value(np.degrees(ctl.alpha))
            steer_tr.set_value(ctl.steer), speed_tr.set_value(mob.speed)
            ang = np.arctan2(mob.pose[1], mob.pose[0])
            lap["wound"] += (ang - lap["prev"] + PI) % TAU - PI
            lap["prev"] = ang

        hud = self.build_hud(d_tr, alpha_tr, steer_tr, speed_tr)
        self.play(FadeIn(car), FadeIn(hud))
        follower(car.pose, 1e-3)
        scenery(car, 0.0)  # place beams before their first visible frame
        self.add(fan, beam_b, beam_a, hit_b, hit_a, tail)
        self.add(hud)  # keep the dashboard above the beams
        drive_car(car, follower, accel=1.5)
        car.add_updater(scenery)
        self.swap_caption(
            "$\\theta=45^\\circ$, a lookahead, and a PD gain --- exactly the math"
            " you just derived."
        )
        self.wait_until(lambda: lap["wound"] > 0.5 * TAU, max_time=20)
        self.swap_caption(
            "Corners handle themselves: the wall falls away, $D'$ jumps,"
            " the car turns after it."
        )
        self.wait_until(lambda: lap["wound"] > 1.1 * TAU, max_time=25)
        car.clear_updaters()
        tail.clear_updaters()
        self.clear_caption()
        self.play(*map(FadeOut, (fan, beam_b, beam_a, hit_b, hit_a, hud, goal, car)))
        self.wait(0.5)
        self.lap_leftovers = Group(walls, tail)

    def build_hud(self, d_tr, alpha_tr, steer_tr, speed_tr) -> Group:
        panel = Rectangle(width=3.1, height=1.9).set_stroke(GREY_C, 1.5)
        panel.set_fill(BLACK, 0.82).to_corner(UR, buff=0.25)
        left = panel.get_left()[0] + 0.25

        def row(label, color, y):
            return TexText(label, font_size=28).set_color(color).move_to(
                [left + 0.55, y, 0]
            )

        y0 = panel.get_top()[1]
        d_txt = row("$D$", D_COLOR, y0 - 0.35)
        d_val = DecimalNumber(0, font_size=28, num_decimal_places=2, unit=r"\,\text{m}")
        d_val.set_color(D_COLOR).next_to(d_txt, RIGHT, buff=0.25)
        a_txt = row(r"$\alpha$", ALPHA_COLOR, y0 - 0.8)
        a_val = DecimalNumber(0, font_size=28, num_decimal_places=1,
                              include_sign=True, unit=r"^\circ")
        a_val.set_color(ALPHA_COLOR).next_to(a_txt, RIGHT, buff=0.25)
        f_always(d_val.set_value, d_tr.get_value)
        f_always(a_val.set_value, alpha_tr.get_value)

        steer_txt = row("steer", ORANGE, y0 - 1.25)
        speed_txt = row("speed", TEAL, y0 - 1.65)
        bar_x = left + 1.55
        steer_axis = Line([bar_x, y0 - 1.25, 0], [bar_x + 1.2, y0 - 1.25, 0]).set_stroke(GREY_D, 2)
        speed_axis = Line([bar_x, y0 - 1.65, 0], [bar_x + 1.2, y0 - 1.65, 0]).set_stroke(GREY_D, 2)
        steer_bar = always_redraw(lambda: Line(
            [bar_x + 0.6, y0 - 1.25, 0],
            [bar_x + 0.6 - np.clip(steer_tr.get_value() / 5, -1, 1) * 0.58, y0 - 1.25, 0],
        ).set_stroke(ORANGE, 6))
        speed_bar = always_redraw(lambda: Line(
            [bar_x, y0 - 1.65, 0],
            [bar_x + np.clip(speed_tr.get_value() / 1.2, 0, 1) * 1.2 + 1e-3, y0 - 1.65, 0],
        ).set_stroke(TEAL, 6))
        return Group(panel, d_txt, d_val, a_txt, a_val, steer_txt, speed_txt,
                     steer_axis, speed_axis, steer_bar, speed_bar)

    # -- outro --------------------------------------------------------------

    def outro(self):
        the_end = TexText("The End!", font_size=96)
        sub = TexText("Lab 1: make it lap.", font_size=40).set_color(GREY_B)
        logo = ImageMobject(str(ASSETS / "F1TenthUCIlogo.png"), height=1.2)
        footer = Group(sub, logo).arrange(RIGHT, buff=0.6).shift(1.6 * DOWN)
        self.play(
            self.lap_leftovers.animate.set_opacity(0.25),
            Write(the_end),
        )
        self.play(FadeIn(footer))
        self.wait(2)
