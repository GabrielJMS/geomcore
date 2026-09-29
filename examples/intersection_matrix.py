"""Pairwise surface-intersection matrix → geomcore-viewer scene.

Builds a 5x5 grid (Plane, Cylinder, Cone, Sphere, Torus), one cell per
(type, type) pair. Each cell shows both surfaces as meshes; where the
geomcore Python API already computes the pair's intersection, the
intersection curve is overlaid in red.

Cells whose pair has no analytic intersection yet show both surfaces with
a "(pending)" tag — filling those cells *is* the Stream 2 roadmap, so
this script doubles as its acceptance visualization: extend PAIR_BUILDERS
as new ``intersect_*`` APIs land.

Usage:
    python examples/intersection_matrix.py [--out scene.json] [--serve]

    --serve  serve the locally built viewer frontend and open the scene
             in a browser (needs geomcore-viewer/frontend/dist).

Viewing without --serve: open the hosted viewer and drag-drop the saved
``.json`` file onto it, or serve it with ``python -m geomcore_viewer``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from geomcore import Frame3D, Point3D, Vector3D
from geomcore.surfaces import Cone, Cylinder, Plane, Sphere, Torus
from geomcore_viewer import Viewer
from geomcore_viewer.sampling import sample_curve, sample_surface_grid

TYPES = ["Plane", "Cylinder", "Cone", "Sphere", "Torus"]

TAU = 2.0 * math.pi

# (nu, nv, u_range, v_range) per surface type for bounded patch sampling.
WINDOWS = {
    "Plane": (24, 24, (-6.0, 6.0), (-6.0, 6.0)),
    "Cylinder": (48, 16, (0.0, TAU), (-6.0, 6.0)),
    "Cone": (48, 20, (0.0, TAU), (-4.0, 6.0)),
    "Sphere": (48, 24, (0.0, TAU), (-1.55, 1.55)),
    "Torus": (48, 24, (0.0, TAU), (0.0, TAU)),
}

SURFACE_COLORS = ("#4d94ff", "#35c163")
CURVE_COLOR = "#ff5533"

# Layout: cells are CELL_STEP apart; surfaces span about +-6 locally.
CELL_STEP = 18.0


def canonical(name: str):
    """One representative instance per surface type (cell-local coords)."""
    if name == "Plane":
        return Plane(Point3D.origin(), Vector3D.z())
    if name == "Cylinder":
        return Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    if name == "Cone":
        return Cone.from_frame(Frame3D.world(), 0.4, 2.0)
    if name == "Sphere":
        return Sphere(Point3D.origin(), 3.0)
    if name == "Torus":
        return Torus(Point3D.origin(), Vector3D.z(), 4.0, 1.0)
    raise KeyError(name)


def sample_surface(surface, kind: str):
    nu, nv, (u0, u1), (v0, v1) = WINDOWS[kind]
    return sample_surface_grid(surface.eval_point, u0, u1, v0, v1, nu, nv)


def shift(points, offset):
    ox, oy, oz = offset
    return [[p[0] + ox, p[1] + oy, p[2] + oz] for p in points]


def pair_plane_plane():
    s1 = canonical("Plane")
    s2 = Plane(Point3D.origin(), Vector3D.x())
    kind, payload = s1.intersect_plane(s2)
    assert kind == "line", kind
    curve = sample_curve(payload.eval_point, -7.0, 7.0, 128)
    return (s1, "Plane"), (s2, "Plane"), (curve, "line")


def pair_plane_sphere():
    s1 = canonical("Plane")
    s2 = canonical("Sphere")
    kind, payload = s1.intersect_sphere(s2)
    assert kind == "circle", kind
    curve = sample_curve(payload.eval_point, 0.0, TAU, 128)
    return (s1, "Plane"), (s2, "Sphere"), (curve, "circle")


def pair_sphere_plane():
    s1 = canonical("Sphere")
    s2 = canonical("Plane")
    kind, payload = s2.intersect_sphere(s1)
    assert kind == "circle", kind
    curve = sample_curve(payload.eval_point, 0.0, TAU, 128)
    return (s1, "Sphere"), (s2, "Plane"), (curve, "circle")


def pair_sphere_sphere():
    s1 = canonical("Sphere")
    s2 = Sphere(Point3D(3.0, 0.0, 0.0), 2.0)
    kind, payload = s1.intersect_sphere(s2)
    assert kind == "circle", kind
    curve = sample_curve(payload.eval_point, 0.0, TAU, 128)
    return (s1, "Sphere"), (s2, "Sphere"), (curve, "circle")


def pair_sphere_cylinder():
    s1 = canonical("Sphere")
    s2 = canonical("Cylinder")
    kind, payload = s1.intersect_cylinder(s2)
    assert kind == "two_circles", kind
    curves = [sample_curve(c.eval_point, 0.0, TAU, 128) for c in payload]
    return (s1, "Sphere"), (s2, "Cylinder"), (curves, "two_circles")


def pair_cylinder_sphere():
    s1 = canonical("Cylinder")
    s2 = canonical("Sphere")
    kind, payload = s2.intersect_cylinder(s1)
    assert kind == "two_circles", kind
    curves = [sample_curve(c.eval_point, 0.0, TAU, 128) for c in payload]
    return (s1, "Cylinder"), (s2, "Sphere"), (curves, "two_circles")


def pair_sphere_cone():
    s1 = canonical("Sphere")
    s2 = canonical("Cone")
    kind, payload = s1.intersect_cone(s2)
    assert kind == "two_circles", kind
    curves = [sample_curve(c.eval_point, 0.0, TAU, 128) for c in payload]
    return (s1, "Sphere"), (s2, "Cone"), (curves, "two_circles")


def pair_cone_sphere():
    s1 = canonical("Cone")
    s2 = canonical("Sphere")
    kind, payload = s2.intersect_cone(s1)
    assert kind == "two_circles", kind
    curves = [sample_curve(c.eval_point, 0.0, TAU, 128) for c in payload]
    return (s1, "Cone"), (s2, "Sphere"), (curves, "two_circles")


def pair_cylinder_cylinder():
    s1 = canonical("Cylinder")
    s2 = Cylinder(Point3D(3.0, 0.0, 0.0), Vector3D.z(), 2.0)
    kind, payload = s1.intersect_cylinder(s2)
    assert kind == "two_lines", kind
    curves = [sample_curve(line.eval_point, -7.0, 7.0, 128) for line in payload]
    return (s1, "Cylinder"), (s2, "Cylinder"), (curves, "two_lines")


def pair_plane_cylinder():
    s1 = canonical("Plane")
    s2 = canonical("Cylinder")
    kind, payload = s1.intersect_cylinder(s2)
    assert kind == "circle", kind
    curve = sample_curve(payload.eval_point, 0.0, TAU, 128)
    return (s1, "Plane"), (s2, "Cylinder"), (curve, "circle")


def pair_cylinder_plane():
    s1 = canonical("Cylinder")
    s2 = canonical("Plane")
    kind, payload = s2.intersect_cylinder(s1)
    assert kind == "circle", kind
    curve = sample_curve(payload.eval_point, 0.0, TAU, 128)
    return (s1, "Cylinder"), (s2, "Plane"), (curve, "circle")


def pair_plane_cone():
    s1 = canonical("Plane")
    s2 = canonical("Cone")
    kind, payload = s1.intersect_cone(s2)
    assert kind == "circle", kind
    curve = sample_curve(payload.eval_point, 0.0, TAU, 128)
    return (s1, "Plane"), (s2, "Cone"), (curve, "circle")


def pair_cone_plane():
    s1 = canonical("Cone")
    s2 = canonical("Plane")
    kind, payload = s2.intersect_cone(s1)
    assert kind == "circle", kind
    curve = sample_curve(payload.eval_point, 0.0, TAU, 128)
    return (s1, "Cone"), (s2, "Plane"), (curve, "circle")


def pair_cone_cylinder():
    s1 = canonical("Cone")
    s2 = canonical("Cylinder")
    kind, payload = s1.intersect_cylinder(s2)
    assert kind == "circle", kind
    curve = sample_curve(payload.eval_point, 0.0, TAU, 128)
    return (s1, "Cone"), (s2, "Cylinder"), (curve, "circle")


def pair_cylinder_cone():
    s1 = canonical("Cylinder")
    s2 = canonical("Cone")
    kind, payload = s2.intersect_cylinder(s1)
    assert kind == "circle", kind
    curve = sample_curve(payload.eval_point, 0.0, TAU, 128)
    return (s1, "Cylinder"), (s2, "Cone"), (curve, "circle")


def pair_cone_cone():
    s1 = canonical("Cone")
    frame2 = Frame3D(Point3D(0.0, 0.0, 1.0), Vector3D.z(), Vector3D.x())
    s2 = Cone.from_frame(frame2, 0.6, 1.0)
    kind, payload = s1.intersect_cone(s2)
    assert kind == "circle", kind
    curve = sample_curve(payload.eval_point, 0.0, TAU, 128)
    return (s1, "Cone"), (s2, "Cone"), (curve, "circle")


def pair_torus_plane():
    s1 = canonical("Torus")
    s2 = canonical("Plane")
    kind, payload = s1.intersect_plane(s2)
    assert kind == "two_circles", kind
    curves = [sample_curve(c.eval_point, 0.0, TAU, 128) for c in payload]
    return (s1, "Torus"), (s2, "Plane"), (curves, "two_circles")


def pair_plane_torus():
    s1 = canonical("Plane")
    s2 = canonical("Torus")
    kind, payload = s2.intersect_plane(s1)
    assert kind == "two_circles", kind
    curves = [sample_curve(c.eval_point, 0.0, TAU, 128) for c in payload]
    return (s1, "Plane"), (s2, "Torus"), (curves, "two_circles")


def pair_torus_sphere():
    s1 = canonical("Torus")
    s2 = Sphere(Point3D(0.0, 0.0, 1.0), 4.0)
    kind, payload = s1.intersect_sphere(s2)
    assert kind == "two_circles", kind
    curves = [sample_curve(c.eval_point, 0.0, TAU, 128) for c in payload]
    return (s1, "Torus"), (s2, "Sphere"), (curves, "two_circles")


def pair_sphere_torus():
    s1 = Sphere(Point3D(0.0, 0.0, 1.0), 4.0)
    s2 = canonical("Torus")
    kind, payload = s2.intersect_sphere(s1)
    assert kind == "two_circles", kind
    curves = [sample_curve(c.eval_point, 0.0, TAU, 128) for c in payload]
    return (s1, "Sphere"), (s2, "Torus"), (curves, "two_circles")


def pair_torus_cylinder():
    s1 = canonical("Torus")
    s2 = Cylinder(Point3D.origin(), Vector3D.z(), 4.5)
    kind, payload = s1.intersect_cylinder(s2)
    assert kind == "two_circles", kind
    curves = [sample_curve(c.eval_point, 0.0, TAU, 128) for c in payload]
    return (s1, "Torus"), (s2, "Cylinder"), (curves, "two_circles")


def pair_cylinder_torus():
    s1 = Cylinder(Point3D.origin(), Vector3D.z(), 4.5)
    s2 = canonical("Torus")
    kind, payload = s2.intersect_cylinder(s1)
    assert kind == "two_circles", kind
    curves = [sample_curve(c.eval_point, 0.0, TAU, 128) for c in payload]
    return (s1, "Cylinder"), (s2, "Torus"), (curves, "two_circles")


def pair_torus_cone():
    s1 = canonical("Torus")
    s2 = Cone.from_frame(Frame3D.world(), 1.1, 2.0)
    kind, payload = s1.intersect_cone(s2)
    assert kind == "two_circles", kind
    curves = [sample_curve(c.eval_point, 0.0, TAU, 128) for c in payload]
    return (s1, "Torus"), (s2, "Cone"), (curves, "two_circles")


def pair_cone_torus():
    s1 = Cone.from_frame(Frame3D.world(), 1.1, 2.0)
    s2 = canonical("Torus")
    kind, payload = s2.intersect_cone(s1)
    assert kind == "two_circles", kind
    curves = [sample_curve(c.eval_point, 0.0, TAU, 128) for c in payload]
    return (s1, "Cone"), (s2, "Torus"), (curves, "two_circles")


def pair_torus_torus():
    s1 = canonical("Torus")
    s2 = Torus(Point3D(0.0, 0.0, 1.5), Vector3D.z(), 4.0, 1.0)
    kind, payload = s1.intersect_torus(s2)
    assert kind == "two_circles", kind
    curves = [sample_curve(c.eval_point, 0.0, TAU, 128) for c in payload]
    return (s1, "Torus"), (s2, "Torus"), (curves, "two_circles")


# Pair-specific builders returning ((surf1, kind1), (surf2, kind2),
# (curve_points, curve_label) | None). Absent pairs fall back to two
# canonical instances with no curve (pending analytic intersection).
PAIR_BUILDERS = {
    ("Plane", "Plane"): pair_plane_plane,
    ("Plane", "Sphere"): pair_plane_sphere,
    ("Plane", "Cylinder"): pair_plane_cylinder,
    ("Plane", "Cone"): pair_plane_cone,
    ("Plane", "Torus"): pair_plane_torus,
    ("Sphere", "Plane"): pair_sphere_plane,
    ("Sphere", "Sphere"): pair_sphere_sphere,
    ("Sphere", "Cylinder"): pair_sphere_cylinder,
    ("Sphere", "Cone"): pair_sphere_cone,
    ("Sphere", "Torus"): pair_sphere_torus,
    ("Cylinder", "Plane"): pair_cylinder_plane,
    ("Cylinder", "Sphere"): pair_cylinder_sphere,
    ("Cylinder", "Cylinder"): pair_cylinder_cylinder,
    ("Cylinder", "Cone"): pair_cylinder_cone,
    ("Cylinder", "Torus"): pair_cylinder_torus,
    ("Cone", "Plane"): pair_cone_plane,
    ("Cone", "Sphere"): pair_cone_sphere,
    ("Cone", "Cylinder"): pair_cone_cylinder,
    ("Cone", "Cone"): pair_cone_cone,
    ("Cone", "Torus"): pair_cone_torus,
    ("Torus", "Plane"): pair_torus_plane,
    ("Torus", "Sphere"): pair_torus_sphere,
    ("Torus", "Cylinder"): pair_torus_cylinder,
    ("Torus", "Cone"): pair_torus_cone,
    ("Torus", "Torus"): pair_torus_torus,
}


def build_cell(row: str, col: str):
    """Resolve the two surfaces and optional intersection curve for a cell."""
    builder = PAIR_BUILDERS.get((row, col))
    if builder is not None:
        return builder()
    if row == col:
        # Self-intersection is the whole surface: show one instance.
        s = canonical(row)
        return (s, row), None, None
    return (canonical(row), row), (canonical(col), col), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output scene.json path")
    ap.add_argument("--serve", action="store_true", help="serve the local viewer frontend")
    ap.add_argument("--port", type=int, default=8765, help="local server port")
    ap.add_argument("--frontend", default=None, help="geomcore-viewer frontend/dist dir")
    args = ap.parse_args()

    viewer = Viewer()
    with_curve, pending = [], []
    for i, row in enumerate(TYPES):
        for j, col in enumerate(TYPES):
            (s1, k1), second, curve_info = build_cell(row, col)
            ox, oy = j * CELL_STEP, -i * CELL_STEP
            tag = f"cell_{row}_{col}"
            verts, idx = sample_surface(s1, k1)
            viewer.add_surface(
                f"{tag}_s1", shift(verts, (ox, oy, 0.0)), idx,
                color=SURFACE_COLORS[0], name=f"{tag} {k1}#1",
            )
            if second is not None:
                (s2, k2) = second
                verts, idx = sample_surface(s2, k2)
                viewer.add_surface(
                    f"{tag}_s2", shift(verts, (ox, oy, 0.0)), idx,
                    color=SURFACE_COLORS[1], name=f"{tag} {k2}#2",
                )
            if curve_info is not None:
                pts, label = curve_info
                # Single polyline or a list of them (twin sections).
                polylines = pts if isinstance(pts[0][0], list) else [pts]
                for k, poly in enumerate(polylines):
                    suffix = f"_x{k}" if len(polylines) > 1 else "_x"
                    viewer.add_curve3d(
                        f"{tag}{suffix}", shift(poly, (ox, oy, 0.0)),
                        color=CURVE_COLOR, name=f"{tag} {label}",
                    )
                with_curve.append(f"({row},{col})")
            else:
                pending.append(f"({row},{col})")

    out = Path(args.out) if args.out else Path(__file__).with_suffix(".scene.json")
    viewer.save(out)
    print(f"saved {out} ({len(viewer.entities)} entities)")
    print(f"cells with intersection curves ({len(with_curve)}): {' '.join(with_curve)}")
    print(f"cells pending analytic intersection ({len(pending)}): {' '.join(pending)}")

    if args.serve:
        import threading

        frontend = Path(args.frontend) if args.frontend else (
            Path.home() / "Documentos/open_source/geomcore-viewer/frontend/dist"
        )
        server = viewer.serve(frontend, port=args.port)
        print(f"serving at {server.url} — Ctrl-C to stop")
        threading.Event().wait()
    else:
        print("view it: drag the .json onto the geomcore-viewer page, or re-run with --serve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
