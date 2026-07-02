# Curves and surfaces

All curves share one API: `eval_point(t)`, `eval_derivative(t, order)`,
`eval_points([...])`, and (for the analytic types) `parameter_of(point)`.
Surfaces mirror it with two parameters `(u, v)`.

## Building curves

Every type offers named constructors for the natural ways to define it:

```python
import math
from geomcore import Point3, Vector3
from geomcore.curves import Circle3D, Ellipse3D, Line3D

# A line through two points (parametrized by arc length).
line = Line3D.from_two_points(Point3.origin(), Point3.new(1.0, 1.0, 0.0))

# A circle from center / normal / radius ...
circle = Circle3D.new(Point3.origin(), Vector3.z(), 2.0)

# ... or through three points.
tri = Circle3D.from_three_points(
    Point3.new(2.0, 0.0, 0.0), Point3.new(0.0, 2.0, 0.0), Point3.new(-2.0, 0.0, 0.0)
)
print(tri.radius(), tri.center().x)   # 2.0 0.0

# An ellipse from center, plane normal, major-axis direction, and both radii.
ellipse = Ellipse3D.new(Point3.origin(), Vector3.z(), Vector3.x(), 3.0, 1.5)
```

Constructors raise `ValueError` for degenerate input (zero-length directions,
negative radii, collinear points, ...), so a constructed object is always
valid.

## Evaluation and derivatives

```python
p = ellipse.eval_point(math.pi / 2)     # point at parameter t
velocity = circle.eval_derivative(0.0, 1)      # first derivative (a Vector3)
acceleration = circle.eval_derivative(0.0, 2)  # second derivative
print(velocity.components())      # (0.0, 2.0, 0.0)
print(acceleration.components())  # (-2.0, 0.0, 0.0)
```

## Parameter inversion

For a point known to lie on an analytic curve, `parameter_of` recovers its
parameter — the inverse of `eval_point`:

```python
t = circle.parameter_of(circle.eval_point(1.25))
print(t)   # 1.25
```

## Surfaces

The five elementary surfaces plus B-spline surfaces follow the same pattern
over a 2D parameter space:

```python
from geomcore.surfaces import Cylinder, Plane, Sphere, Torus

plane = Plane.new(Point3.origin(), Vector3.z())
cylinder = Cylinder.new(Point3.origin(), Vector3.z(), 2.0)
sphere = Sphere.new(Point3.origin(), 3.0)
torus = Torus.new(Point3.origin(), Vector3.z(), 5.0, 1.0)

p = sphere.eval_point(0.7, 0.4)
u, v = sphere.parameters_of(p)    # round-trips: (0.7, 0.4)
```

Partial derivatives take an order per direction — `(1, 0)` is ∂/∂u,
`(0, 1)` is ∂/∂v, `(1, 1)` is the mixed second partial:

```python
du = cylinder.eval_derivative(0.0, 1.0, 1, 0)
dv = cylinder.eval_derivative(0.0, 1.0, 0, 1)
print(du.components())   # (0.0, 2.0, 0.0)
print(dv.components())   # (0.0, 0.0, 1.0)
```

## Bulk evaluation

Both curves and surfaces evaluate whole batches in one native call — this is
the fast path for sampling, plotting, and meshing:

```python
grid = [(i / 10, j / 10) for i in range(10) for j in range(10)]
points = torus.eval_points(grid)
print(len(points))   # 100
```
