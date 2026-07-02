# Quickstart

## Install

```sh
pip install geomcore
```

Pre-built wheels cover Linux (x86_64, aarch64), macOS, and Windows for
Python 3.9+ — no compiler needed.

## Your first curve

```python
import math
from geomcore import Point3, Vector3
from geomcore.curves import Circle3D

circle = Circle3D.new(Point3.origin(), Vector3.z(), 2.0)

point = circle.eval_point(math.pi / 4)
print(point.x, point.y, point.z)
# 1.4142135623730951 1.414213562373095 0.0
```

Curves are parametric: a circle maps an angle parameter to a 3D point on the
circle. `Circle3D.new` takes the center, the plane normal, and the radius.

## Bulk evaluation

When you need many points — sampling, plotting, meshing — pass the whole
parameter list at once. One native call evaluates everything in compiled code:

```python
params = [i / 100 * math.tau for i in range(100)]
points = circle.eval_points(params)
print(len(points))   # 100
```

## Errors are Pythonic

Constructors validate their input up front and raise `ValueError` with a
descriptive message, so invalid geometry never exists:

```python
Circle3D.new(Point3.origin(), Vector3.z(), -1.0)
# ValueError: radius is negative
```

## Next steps

- [Geometry basics](tutorials/geometry-basics.md) — points, vectors, frames,
  and transforms.
- [Curves and surfaces](tutorials/curves-and-surfaces.md) — the full
  evaluation API.
- [B-splines and NURBS](tutorials/bsplines.md) — free-form geometry from
  control points.
- [Curve-on-surface parametrization](tutorials/curve-on-surface.md) — the
  feature that sets `geomcore` apart.
