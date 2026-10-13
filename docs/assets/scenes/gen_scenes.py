"""Generate the viewer scenes embedded in the guides.

Uses the real kernel + the real viewer client, so every scene is computed,
not drawn: exact intersections, marching loops, projection feet.

Run:  python3.11 docs/assets/scenes/gen_scenes.py
Needs geomcore + geomcore_viewer importable (this repo's geomcore-py build).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from geomcore import Point3D, Vector3D
from geomcore.curves import Circle3D, Ellipse3D
from geomcore.surfaces import BSplineSurface, Cylinder, Plane, Sphere

import geomcore_viewer as gv

OUT = Path(__file__).parent


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
    grid = [
        [Point3D(x, y, 1.5 * math.sin(x) * math.cos(y)) for y in xs] for x in xs
    ]
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
    print(f"projection: {len(feet)} feet: {[(round(t, 4), round(d, 4)) for t, d in feet]}")
    view.add(e, label="E")
    view.add(q, label="Q")
    view.add([e.eval_point(t) for t, _ in feet], label="foot")
    save(view, "projection-feet.json")


exact_scene()
marching_scene()
projection_scene()
