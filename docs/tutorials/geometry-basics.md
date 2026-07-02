# Geometry basics

Everything in `geomcore` is built from a few small value types: points,
vectors, axes, and frames. They are immutable — constructors validate once,
and the objects can be shared freely afterwards.

## Points and vectors

Points are locations; vectors are directions/displacements. They are distinct
types on purpose — mixing them up is one of the classic geometry bugs.

```python
from geomcore import Point3, Vector3

p = Point3.new(1.0, 2.0, 3.0)
q = Point3.new(4.0, 6.0, 3.0)
print(p.distance(q))          # 5.0

v = Vector3.new(1.0, 0.0, 0.0)
w = Vector3.new(0.0, 1.0, 0.0)
print(v.dot(w))               # 0.0
print(v.cross(w).components())  # (0.0, 0.0, 1.0)
print(v.magnitude())          # 1.0
```

Common constants are static methods: `Point3.origin()`, `Vector3.x()`,
`Vector3.y()`, `Vector3.z()`, `Vector3.zero()`. The 2D twins `Point2` and
`Vector2` work the same way and appear when you work in a surface's parameter
space.

## Axes and frames

An `Axis3` is a located direction (a rotation axis, a cylinder axis). A
`Frame3` is a full right-handed coordinate system — most curves and surfaces
can be built directly from one.

```python
from geomcore import Axis3, Frame3, Point3, Vector3

axis = Axis3.new(Point3.origin(), Vector3.z())

# A frame from origin + z-direction; x/y are chosen deterministically.
frame = Frame3.from_z(Point3.new(0.0, 0.0, 1.0), Vector3.z())
print(frame.x_direction().components())  # (1.0, 0.0, 0.0)

world = Frame3.world()  # the global frame
```

## Transforms

All rigid motions (plus scaling and mirroring) go through one type:

```python
import math
from geomcore import Axis3, Frame3, Point3, Transform, Vector3

rot = Transform.rotation(Axis3.new(Point3.origin(), Vector3.z()), math.pi / 2)
moved = rot.apply_point(Point3.new(1.0, 0.0, 0.0))
print(moved.x, moved.y)       # ~0.0 1.0

mirror = Transform.mirror_plane(Frame3.world())
print(mirror.apply_point(Point3.new(0.0, 0.0, 2.0)).z)  # -2.0
```

Compose with `then` — the name reads in application order:

```python
shift = Transform.translation(Vector3.new(1.0, 0.0, 0.0))
combined = shift.then(rot)    # translate first, then rotate
print(combined.apply_point(Point3.origin()).y)  # 1.0
```

`apply_vector` transforms directions (ignoring translation), which is the
right behavior for normals and tangents under rigid motion.
