# Curve-on-surface parametrization

This is `geomcore`'s signature feature. When a 3D curve lies on a surface, it
has a second life as a **2D curve in the surface's parameter space** — CAD
kernels call this a *pcurve*, and B-rep modeling is built on them (a face's
boundary edges are stored exactly this way).

`parametrize_on` computes that 2D representation **in closed form** — no
approximation, no sampling — for lines and circles on the five elementary
surfaces.

## A circle on a cylinder

A circle of radius 2 at height 1.5 on a coaxial cylinder of radius 2 becomes
a *horizontal straight line* in `(u, v)` space: sweeping the angle `u` while
`v` stays at the height.

```python
from geomcore import Point3, Vector3
from geomcore.curves import Circle3D
from geomcore.surfaces import Cylinder

cylinder = Cylinder.new(Point3.origin(), Vector3.z(), 2.0)
circle = Circle3D.new(Point3.new(0.0, 0.0, 1.5), Vector3.z(), 2.0)

pcurve = circle.parametrize_on(cylinder)
print(type(pcurve).__name__)              # Line2D
print(pcurve.origin().x, pcurve.origin().y)        # 0.0 1.5
print(pcurve.direction().x, pcurve.direction().y)  # 1.0 0.0
```

## The defining identity

A pcurve `p(t)` is correct when composing it with the surface reproduces the
original curve: `surface.eval_point(*p(t)) == curve.eval_point(t)` for every
`t`. Check it yourself:

```python
for t in (0.0, 1.7, 4.1):
    uv = pcurve.eval_point(t)                     # a Point2 in (u, v)
    on_surface = cylinder.eval_point(uv.x, uv.y)  # lift through the surface
    direct = circle.eval_point(t)                 # evaluate the curve directly
    assert abs(on_surface.x - direct.x) < 1e-12
    assert abs(on_surface.y - direct.y) < 1e-12
    assert abs(on_surface.z - direct.z) < 1e-12
```

This identity — at these very parameters — is how every parametrization in
`geomcore` is tested.

## When there is no closed form

Some pairs have no exact 2D representation — a straight line lying on a
sphere's surface is impossible unless it degenerates, and a line tangent to
it simply doesn't lie on it. `parametrize_on` refuses rather than
approximating:

```python
from geomcore.curves import Line3D
from geomcore.surfaces import Sphere

sphere = Sphere.new(Point3.origin(), 3.0)
line = Line3D.new(Point3.new(3.0, 0.0, 0.0), Vector3.z())

line.parametrize_on(sphere)
# ValueError: no closed-form 2D representation exists for this curve/surface pair
```

A numeric-approximation fallback for such pairs is on the
[roadmap](https://github.com/GabrielJMS/geomcore/blob/main/ROADMAP.md).

## Supported pairs

| curve \ surface | Plane | Cylinder | Cone | Sphere | Torus |
|---|---|---|---|---|---|
| `Line3D` | ✓ | ✓ (rulings) | ✓ (rulings) | — | — |
| `Circle3D` | ✓ | ✓ | ✓ | ✓ | ✓ |

(Lines can only lie on ruled surfaces; spheres and tori contain no straight
lines.) The result is a `Line2D` or `Circle2D`, both of which support the
full curve evaluation API in 2D.
