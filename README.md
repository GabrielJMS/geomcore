# geomcore

A pure, standalone geometric kernel for CAD-grade curves and surfaces —
parametric evaluation, rigid transformations, and analytic curve-on-surface
parametrization, built as small, independently useful Rust libraries with a
clean API and idiomatic Python bindings.

`geomcore` is not a modeling/CAD system — no topology, no B-rep, no boolean
operations (yet). It's the layer underneath one: parametric curve/surface
evaluation, rigid transformations, and curve-on-surface parametrization,
usable on their own by anyone who needs robust computational geometry — not
just future CAD-kernel builders.

## Status

Pre-1.0: the API can still change between minor versions (breaking
changes are flagged in the changelog). Implemented:

- **Evaluation** of lines, circles, ellipses, parabolas, hyperbolas and
  B-spline curves (rational and periodic) in 3D, plus 2D lines and
  circles; and of planes, cylinders, cones, spheres, tori and B-spline
  surfaces.
- **Transformations**: translation, rotation, mirroring and scaling via a
  single `Transform` type, applied to any curve or surface with
  `transformed`.
- **Containment, projection and extrema** on every curve and surface:
  closed form on the analytic types, exact subdivision on B-splines.
- **Intersections** of any two curves or surfaces, B-splines and trimmed
  curves included, through one `a.intersect(&b, tol)` call that works in
  either order and on shapes whose type is only known at run time. It
  solves the pair exactly when a closed form exists (plane, quadric and
  torus pairs; lines, circles and conics against each other and against
  the elementary surfaces), with tangency and coincidence classification,
  and runs a numeric search otherwise (curve–curve and curve–surface
  subdivision, marching surface–surface fitted with B-splines). Every
  result says which method ran and whether every hit was found. The
  typed `intersect_exact` returns each pair's own exact result, for hot
  loops over elementary shapes.
- **Curve–curve extrema**: the local distance minima between two curves.
- **Curve-on-surface parametrization**: analytic for lines and circles on
  the five elementary surfaces, numeric everywhere else.
- **B-spline fitting**: interpolation and least-squares approximation.

Not implemented: topology/B-rep and boolean operations.

Evaluation, transformations and curve-on-surface parametrization are
validated against golden fixtures with CAD-kernel-grade tolerances
(1e-7).

See [ROADMAP.md](ROADMAP.md) for what's coming next and
[CHANGELOG.md](CHANGELOG.md) for release history.

## Rust

```rust
use geomcore::{Point3D, Vector3D};
use geomcore::curves::Circle3D;

let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
let point = circle.eval_point(std::f64::consts::PI / 4.0);
```

    cargo add geomcore

## Python

```python
from geomcore import Point3D, Vector3D
from geomcore.curves import Circle3D
import math

circle = Circle3D(Point3D.origin(), Vector3D.unit_z(), 2.0)
point = circle.eval_point(math.pi / 4)
```

    pip install geomcore

## Why

Most of what a CAD kernel needs — parametric evaluation, transformations,
curve-on-surface math — is useful on its own, to anyone doing computational
geometry, not just people building a full modeling system. `geomcore` grows
these as small, independently useful, well-tested libraries, with the goal
of eventually adding up to a complete, open, high-performance CAD kernel —
without forcing that scope on anyone who just needs the math.

## Acknowledgments

Some algorithms in this library were written using modules of Open CASCADE Technology (OCCT) as a reference.

## License

Dual-licensed under MIT or Apache-2.0, at your option.

## Contributing

New curve/surface types can be added by implementing `ParametricCurve3D`/
`ParametricSurface` without touching the core crate's dispatch code — see
the crate docs for details.
