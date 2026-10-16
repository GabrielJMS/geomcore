"""Generate the viewer scenes embedded in the guides.

Uses the real kernel + the real viewer client, so every scene is computed,
not drawn: exact intersections, marching loops, projection feet.

Run:  python3 docs/assets/scenes/gen_scenes.py
Needs the released packages: pip install "geomcore>=0.33" "geomcore-viewer>=0.2"
(the viewer only writes the JSON here; no server is started). A scene that
shows an unreleased API needs this repo's geomcore-py build instead.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from geomcore import Frame3D, Point2D, Point3D, Vector3D
from geomcore.curves import Ellipse3D, TrimmedCurve2D, TrimmedCurve3D
from geomcore.surfaces import BSplineSurface, Cone, Cylinder, Plane, Sphere, Torus

import geomcore_viewer as gv

OUT = Path(__file__).parent

# Line-drawing figures: grey wire surfaces, hidden lines dashed, live outline.
LINE_ART = {"wire": True, "hidden": "dashed", "silhouette": True}
SURFACE = "#8d96a6"
A, B, C = "#8e44ad", "#3a5fcd", "#a0612b"
POINT = "#222222"
POLE = "#b03a2e"


def save(view: gv.Viewer, name: str) -> None:
    path = OUT / name
    view.save(path)
    n = len(view.objects)
    print(f"saved {path.name} ({n} objects)")


def exact_scene() -> None:
    """Sphere cap cylinder: two exact latitude rings (plane page: no — intersection)."""
    t0 = time.perf_counter()
    view = gv.Viewer("Sphere ∩ cylinder (exact)", spawn=False, open_browser=False)
    s = Sphere(Point3D.origin(), 3.0)
    y = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    view.add(s, label="S", opacity=0.35, group="exact")
    view.add(y, label="Y", opacity=0.35, v=(-3.5, 3.5), group="exact")
    t0 = time.perf_counter()
    for _ in range(200):
        res = s.intersect(y)
    dt = (time.perf_counter() - t0) / 200
    view.add(res, label="rings", group="exact")
    print(f"exact: {res[0]} in {dt * 1e6:.1f} us/call (200-call mean)")
    save(view, "intersect-exact.json")


def marching_scene() -> None:
    """Wavy B-spline patch through a sphere: marching loops (fitted B-splines)."""
    xs = [i * 0.75 for i in range(-4, 5)]
    grid = [[Point3D(x, y, 1.5 * math.sin(x) * math.cos(y)) for y in xs] for x in xs]
    patch = BSplineSurface.approximate(grid, 3, 3, 6, 6)
    s = Sphere(Point3D.origin(), 3.0)
    from geomcore import intersect_marching

    t0 = time.perf_counter()
    curves = intersect_marching(patch, s)
    dt = time.perf_counter() - t0
    print(f"marching: {len(curves)} loop(s) in {dt * 1e3:.0f} ms")

    view = gv.Viewer("Patch ∩ sphere (marching)", spawn=False, open_browser=False)
    umin = min(xs)
    umax = max(xs)
    view.add(s, label="S", opacity=0.3, group="marching")
    view.add(
        patch,
        label="P",
        opacity=0.55,
        u=(umin, umax),
        v=(umin, umax),
        group="marching",
    )
    for i, c in enumerate(curves, start=1):
        view.add(c, label=f"loop{i}", group="marching")
    save(view, "intersect-marching.json")


def projection_scene() -> None:
    """Ellipse with all four feet of one query point."""
    view = gv.Viewer("Four feet on an ellipse", spawn=False, open_browser=False)
    e = Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 3.0, 2.0)
    q = Point3D(0.5, 0.2, 0)
    feet = e.extrema(q)
    print(
        f"projection: {len(feet)} feet: {[(round(t, 4), round(d, 4)) for t, d in feet]}"
    )
    view.add(e, label="E")
    view.add(q, label="Q")
    view.add([e.eval_point(t) for t, _ in feet], label="foot")
    save(view, "projection-feet.json")


def uv_segment(p: tuple[float, float], q: tuple[float, float]) -> TrimmedCurve2D:
    return TrimmedCurve2D.segment(Point2D(*p), Point2D(*q))


def three_curves_scene(
    name: str, title: str, surface, u: tuple, v: tuple, a_v: float
) -> None:
    """A: a v-iso, B: a u-iso (both exact, from the kernel), C: a straight line in (u, v).

    Positions are fractions of the window, as in the parametric-domain figures;
    A and B are also sent as pcurves so the (u, v) pane mirrors the figure.
    """
    at_u = lambda f: u[0] + f * (u[1] - u[0])  # noqa: E731
    at_v = lambda f: v[0] + f * (v[1] - v[0])  # noqa: E731
    view = gv.Viewer(title, spawn=False, open_browser=False)
    s = view.add(surface, label="S", color=SURFACE, u=u, v=v, **LINE_ART)

    a = (at_u(0.05), at_u(0.75), at_v(a_v))
    b = (at_u(0.16), at_v(0.1), at_v(0.9))
    view.add(
        TrimmedCurve3D(surface.v_iso(a[2]), a[0], a[1]),
        label="A: v = const",
        color=A,
        width=3,
    )
    view.add(
        TrimmedCurve3D(surface.u_iso(b[0]), b[1], b[2]),
        label="B: u = const",
        color=B,
        width=3,
    )
    view.pcurve(
        uv_segment((a[0], a[2]), (a[1], a[2])),
        s,
        label="A",
        color=A,
        width=3,
        group="(u, v)",
    )
    view.pcurve(
        uv_segment((b[0], b[1]), (b[0], b[2])),
        s,
        label="B",
        color=B,
        width=3,
        group="(u, v)",
    )
    c = uv_segment((at_u(0.05), at_v(0.1)), (at_u(0.75), at_v(0.9)))
    view.pcurve(c, s, label="C", color=C, width=3)
    save(view, name)


def facing_frame() -> Frame3D:
    """z up, u = 0 turned so the figures' u ≈ 1 faces the viewer's default camera."""
    return Frame3D(
        Point3D.origin(), Vector3D.z(), Vector3D(math.cos(-1.5), math.sin(-1.5), 0)
    )


def three_curves_scenes() -> None:
    f = facing_frame()
    tau = 2 * math.pi
    three_curves_scene(
        "param-cylinder.json",
        "Three curves on a cylinder",
        Cylinder.from_frame(f, 1.5),
        (0, tau),
        (0, 3),
        0.72,
    )
    three_curves_scene(
        "param-cone.json",
        "Three curves on a cone",
        Cone.from_frame(f, 0.45, 0.5),
        (0, tau),
        (0, 3),
        0.72,
    )
    three_curves_scene(
        "param-sphere.json",
        "Three curves on a sphere",
        Sphere.from_frame(f, 1.5),
        (0, tau),
        (-math.pi / 2, math.pi / 2),
        0.62,
    )
    three_curves_scene(
        "param-torus.json",
        "Three curves on a torus",
        Torus.from_frame(f, 3.0, 1.0),
        (0, tau),
        (0, tau),
        0.08,
    )


def uniform_grid_scene() -> None:
    """A 12 × 6 grid on a sphere; one cell at the equator, the same cell against the pole."""
    view = gv.Viewer(
        "A uniform (u, v) grid on a sphere", spawn=False, open_browser=False
    )
    sphere = Sphere.from_frame(facing_frame(), 2.0)
    s = view.add(sphere, label="S", color=SURFACE, isolines=[12, 6], **LINE_ART)
    du, dv = math.pi / 6, math.pi / 6
    u0, u1 = du / 2, 3 * du / 2  # between two drawn isolines (they sit mid-cell)
    cells = {"equator": (-dv / 2, dv / 2), "pole": (math.pi / 2 - dv / 2, math.pi / 2)}
    for where, (v0, v1) in cells.items():
        view.add(
            sphere,
            label=f"at the {where}",
            color="#aebff2",
            opacity=0.6,
            u=(u0, u1),
            v=(v0, v1),
            group="cells",
        )
        outline = [(u0, v0), (u1, v0), (u1, v1), (u0, v1), (u0, v0)]
        view.pcurve(outline, s, label=f"{where} cell", color=B, width=3, group="cells")
    save(view, "param-grid.json")


def point_resolved_scene(
    title: str, surface, u: float, v: float, u_iso_v: tuple
) -> gv.Viewer:
    """The surface, P(u, v) and the two kernel isos through it."""
    view = gv.Viewer(title, spawn=False, open_browser=False)
    view.add(surface, label="S", color=SURFACE, **LINE_ART)
    view.add(
        TrimmedCurve3D(surface.u_iso(u), *u_iso_v), label="u = const", color=A, width=3
    )
    view.add(surface.v_iso(v), label="v = const", color=B, width=3)
    view.add(surface.eval_point(u, v), label="P(u, v)", color=POINT)
    return view


def elementary_scenes() -> None:
    o, z = Point3D.origin(), Vector3D.z()
    sphere = Sphere(o, 2.0)
    # u = -0.4: the default camera looks from u ≈ -1, so P sits in front, just right of centre.
    view = point_resolved_scene(
        "A sphere with a point resolved", sphere, -0.4, 0.5, (-math.pi / 2, math.pi / 2)
    )
    # At v = ±π/2 the parallel collapses: the kernel's iso is a radius-0 circle, shown as a point.
    view.add(
        [sphere.v_iso(math.pi / 2), sphere.v_iso(-math.pi / 2)],
        label="pole",
        color=POLE,
    )
    save(view, "elem-sphere.json")
    view = point_resolved_scene(
        "A torus with a point resolved",
        Torus(o, z, 3.0, 1.0),
        -0.4,
        0.7,
        (0, 2 * math.pi),
    )
    save(view, "elem-torus.json")


def plane_sphere_scene() -> None:
    """A plane at distance δ from a unit sphere: a ring, a tangent point, nothing."""
    view = gv.Viewer("Plane and sphere", spawn=False, open_browser=False)
    for i, (delta, what) in enumerate(
        [(0.5, "δ = 0.5"), (1.0, "δ = r"), (1.4, "δ = 1.4")]
    ):
        c = Point3D(3.5 * (i - 1), 0, 0)
        sphere = Sphere(c, 1.0)
        plane = Plane(Point3D(c.x, 0, delta), Vector3D.z())
        view.add(
            sphere,
            label=f"S ({what})",
            color=SURFACE,
            isolines=[0, 1],
            group=what,
            **LINE_ART,
        )
        view.add(
            plane,
            label=f"P ({what})",
            color=A,
            opacity=0.25,
            u=(-1.6, 1.6),
            v=(-1.6, 1.6),
            group=what,
        )
        view.add(
            sphere.intersect(plane),
            label=f"S ∩ P ({what})",
            color=B,
            width=3,
            group=what,
        )
    save(view, "plane-sphere.json")


def raised_pole_scene() -> None:
    """A flat bicubic patch with one pole raised: only the cells the pole supports move."""
    u_pole_count, v_pole_count = 7, 6
    poles = [
        [Point3D(i, j, 0) for j in range(v_pole_count)] for i in range(u_pole_count)
    ]
    poles[4][4] = Point3D(4, 4, 2.5)
    surface = BSplineSurface(
        3,
        3,
        poles,
        [0, 0.25, 0.5, 0.75, 1],
        [4, 1, 1, 1, 4],
        [0, 1 / 3, 2 / 3, 1],
        [4, 1, 1, 4],
        False,
        False,
    )
    view = gv.Viewer("A surface and one raised point", spawn=False, open_browser=False)
    view.add(surface, label="S", color=B, isolines=[16, 16], **LINE_ART)
    view.add(
        surface,
        label="the only region that can move",
        color=B,
        opacity=0.25,
        u=(0.25, 1),
        v=(1 / 3, 1),
    )
    net = poles + [
        [poles[i][j] for i in range(u_pole_count)] for j in range(v_pole_count)
    ]
    for k, row in enumerate(net):
        view.polyline(
            row, label=f"net {k}", color="#5f6673", dashed=True, group="control net"
        )
    view.add(poles[4][4], label="P₄,₄ raised", color=POLE)
    save(view, "nurbs-raised-pole.json")


exact_scene()
marching_scene()
projection_scene()
three_curves_scenes()
uniform_grid_scene()
elementary_scenes()
plane_sphere_scene()
raised_pole_scene()
