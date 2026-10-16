"""Pairwise surface-intersection matrix → geomcore-viewer scene.

Builds a 5x5 grid (Plane, Cylinder, Cone, Sphere, Torus), one cell per
(type, type) pair, in receiver order: the cell (row, col) shows
``row.intersect(col)``. Each cell hands the intersection result straight to the viewer, which draws
both surfaces translucent, the curves in the accent colour and a note on the
method that found them: ``exact`` where a closed form exists, ``numeric``
(marched) where not. The viewer windows each unbounded surface (plane,
cylinder, cone) around its intersection. Extend PAIRS to show other
configurations.

Usage:
    python examples/intersection_matrix.py [--out scene.json] [--serve]

    --serve  also send the scene to the live geomcore-viewer (it finds or
             starts its own local server and opens a tab; needs the
             geomcore-viewer package).

Viewing without --serve: open the hosted viewer and drag-drop the saved
``.json`` file onto it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from geomcore import Frame3D, Point3D, Transform, Vector3D
from geomcore.surfaces import Cone, Cylinder, Plane, Sphere, Torus

import geomcore_viewer as gv

TYPES = ["Plane", "Cylinder", "Cone", "Sphere", "Torus"]

# Layout: cells are CELL_STEP apart; surfaces span about +-6 locally.
CELL_STEP = 18.0


def canonical(name: str):
    """One representative instance per surface type (cell-local coords)."""
    if name == "Plane":
        return Plane(Point3D.origin(), Vector3D.unit_z())
    if name == "Cylinder":
        return Cylinder(Point3D.origin(), Vector3D.unit_z(), 2.0)
    if name == "Cone":
        return Cone.from_frame(Frame3D.world(), 0.4, 2.0)
    if name == "Sphere":
        return Sphere(Point3D.origin(), 3.0)
    if name == "Torus":
        return Torus(Point3D.origin(), Vector3D.unit_z(), 4.0, 1.0)
    raise KeyError(name)


# The surfaces of each cell, in receiver order. Configurations other than
# the canonical pair are chosen so that the surfaces meet.
TILTED_CONE = Cone.from_frame(Frame3D(Point3D(0.0, 0.0, 1.0), Vector3D.unit_z(), Vector3D.unit_x()), 0.6, 1.0)
WIDE_CONE = Cone.from_frame(Frame3D.world(), 1.1, 2.0)
RAISED_SPHERE = Sphere(Point3D(0.0, 0.0, 1.0), 4.0)
WIDE_CYLINDER = Cylinder(Point3D.origin(), Vector3D.unit_z(), 4.5)
PAIRS = {
    ("Plane", "Plane"): (canonical("Plane"), Plane(Point3D.origin(), Vector3D.unit_x())),
    ("Sphere", "Sphere"): (canonical("Sphere"), Sphere(Point3D(3.0, 0.0, 0.0), 2.0)),
    ("Cylinder", "Cylinder"): (
        canonical("Cylinder"),
        Cylinder(Point3D(3.0, 0.0, 0.0), Vector3D.unit_z(), 2.0),
    ),
    # Both cones are double-napped: they meet in two circles.
    ("Cone", "Cone"): (canonical("Cone"), TILTED_CONE),
    ("Torus", "Torus"): (
        canonical("Torus"),
        Torus(Point3D(0.0, 0.0, 1.5), Vector3D.unit_z(), 4.0, 1.0),
    ),
    ("Torus", "Sphere"): (canonical("Torus"), RAISED_SPHERE),
    ("Sphere", "Torus"): (RAISED_SPHERE, canonical("Torus")),
    ("Torus", "Cylinder"): (canonical("Torus"), WIDE_CYLINDER),
    ("Cylinder", "Torus"): (WIDE_CYLINDER, canonical("Torus")),
    # The cone meets the torus in rings.
    ("Torus", "Cone"): (canonical("Torus"), WIDE_CONE),
    ("Cone", "Torus"): (WIDE_CONE, canonical("Torus")),
}


def cell_surfaces(row: str, col: str):
    """The two surfaces of a cell: a listed configuration, or both canonical."""
    return PAIRS.get((row, col), (canonical(row), canonical(col)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output scene.json path")
    ap.add_argument("--serve", action="store_true", help="also send the scene to the live viewer")
    args = ap.parse_args()

    view = gv.Viewer("Surface intersection matrix", spawn=args.serve)
    with_curve, empty = [], []
    for i, row in enumerate(TYPES):
        for j, col in enumerate(TYPES):
            move = Transform.translation(Vector3D(j * CELL_STEP, -i * CELL_STEP, 0.0))
            s1, s2 = (surface.transformed(move) for surface in cell_surfaces(row, col))
            result = s1.intersect(s2)
            view.add(result, label=f"{row}×{col}")
            if result.curves:
                with_curve.append(f"({row},{col})")
            else:
                empty.append(f"({row},{col})")

    out = Path(args.out) if args.out else Path(__file__).with_suffix(".scene.json")
    view.save(out)
    print(f"saved {out} ({len(view.objects)} objects)")
    print(f"cells with intersection curves ({len(with_curve)}): {' '.join(with_curve)}")
    print(f"cells with no curve ({len(empty)}): {' '.join(empty)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
