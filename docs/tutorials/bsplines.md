# B-splines and NURBS

Free-form curves and surfaces are defined by control points (*poles*), a knot
vector, and a degree. `geomcore` supports non-rational and rational (NURBS)
forms, clamped and periodic, at any degree.

## Knots and multiplicities

`geomcore` describes knot vectors as **unique knots plus multiplicities**
rather than a flat list with repeats. The flat knot vector
`[0, 0, 0, 0, 1, 2, 3, 3, 3, 3]` becomes knots `[0, 1, 2, 3]` with
multiplicities `[4, 1, 1, 4]`. (Coming from a library that uses flat knot
vectors? The [migration guide](../migrating-from-geomdl.md) has a converter.)

## A cubic curve from control points

```python
from geomcore import Point3
from geomcore.curves import BSplineCurve3D

curve = BSplineCurve3D.new(
    3,                                # degree
    [                                 # poles (control points)
        Point3.new(0.0, 0.0, 0.0),
        Point3.new(1.0, 2.0, 0.0),
        Point3.new(3.0, 2.0, 1.0),
        Point3.new(4.0, 0.0, 1.0),
    ],
    [0.0, 1.0],                       # unique knots
    [4, 4],                           # multiplicities (clamped Bezier form)
    False,                            # not periodic
)

start, end = curve.bounds()           # the valid parameter range
mid = curve.eval_point((start + end) / 2)
print(mid.x, mid.y, mid.z)            # 2.0 1.5 0.5
```

`eval_derivative(t, order)`, `eval_points([...])`, `degree()`,
`is_rational()`, and `is_periodic()` work as you'd expect.

## Exact conics with NURBS

Rational B-splines can represent conics *exactly* — the classic example is a
quarter circle as a single rational quadratic Bézier segment:

```python
import math
from geomcore.curves import BSplineCurve3D
from geomcore import Point3

arc = BSplineCurve3D.new_rational(
    2,
    [Point3.new(1.0, 0.0, 0.0), Point3.new(1.0, 1.0, 0.0), Point3.new(0.0, 1.0, 0.0)],
    [1.0, math.sqrt(2.0) / 2.0, 1.0],   # one weight per pole
    [0.0, 1.0],
    [3, 3],
    False,
)

for t in (0.0, 0.25, 0.5, 0.75, 1.0):
    p = arc.eval_point(t)
    print(round(math.hypot(p.x, p.y), 12))   # 1.0 every time — exact
```

## Plotting

`eval_points` pairs naturally with matplotlib:

```python
import matplotlib.pyplot as plt

params = [i / 199 for i in range(200)]
pts = arc.eval_points(params)
plt.plot([p.x for p in pts], [p.y for p in pts])
plt.axis("equal")
plt.show()
```

## Surfaces

A B-spline surface takes a rectangular pole grid and a knot vector per
direction. A bicubic Bézier patch:

```python
import math
from geomcore import Point3
from geomcore.surfaces import BSplineSurface

poles = [
    [Point3.new(i, j, math.sin(i * j * 0.5)) for j in range(4)]
    for i in range(4)
]
patch = BSplineSurface.new(
    3, 3,                 # u-degree, v-degree
    poles,
    [0.0, 1.0], [4, 4],   # u knots + multiplicities
    [0.0, 1.0], [4, 4],   # v knots + multiplicities
    False, False,         # u/v periodicity
)
print(patch.eval_point(0.5, 0.5).z)   # 0.523424129152024
```

`BSplineSurface.new_rational` adds a weight grid for full NURBS surfaces.

!!! note "Not here yet"
    Knot insertion, splitting, and interpolation/fitting are on the
    [roadmap](https://github.com/GabrielJMS/geomcore/blob/main/ROADMAP.md) —
    today `geomcore` covers construction and evaluation.
