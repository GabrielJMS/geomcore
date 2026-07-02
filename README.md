# geomrust

A pure, standalone geometric kernel for CAD-grade curves and surfaces —
parametric evaluation, rigid transformations, and analytic curve-on-surface
parametrization, built as small, independently useful Rust libraries with a
clean API and idiomatic Python bindings.

`geomrust` is not a modeling/CAD system — no topology, no B-rep, no boolean
operations (yet). It's the layer underneath one: parametric curve/surface
evaluation, rigid transformations, and curve-on-surface parametrization,
usable on their own by anyone who needs robust computational geometry — not
just future CAD-kernel builders.

## Status

Proof-of-concept. Implemented so far:

- Parametric evaluation for lines, circles, ellipses, parabolas,
  hyperbolas, and B-spline curves in 3D (plus 2D lines and circles, the
  shapes curve-on-surface parametrization produces).
- Parametric evaluation for planes, cylinders, cones, spheres, tori, and
  B-spline surfaces.
- Rigid transformations (translation, rotation, mirroring, scaling) via a
  single `Transform` type.
- Analytic curve-on-surface parametrization for lines and circles on the
  five elementary surfaces (plane, cylinder, cone, sphere, torus).

Not implemented yet: curve/surface intersection, extrema computation,
topology/B-rep, and a numeric-approximation fallback for
curve-on-surface parametrization when no closed form exists.

Evaluation results are validated against golden fixtures with
CAD-kernel-grade tolerances (1e-7).

## Rust

```rust
use geomrust::{Point3, Vector3};
use geomrust::curves::Circle3D;

let circle = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
let point = circle.eval_point(std::f64::consts::PI / 4.0);
```

    cargo add geomrust

Not yet published to crates.io — coming with the first release.

## Python

```python
from geomrust import Point3, Vector3
from geomrust.curves import Circle3D
import math

circle = Circle3D.new(Point3.origin(), Vector3.z(), 2.0)
point = circle.eval_point(math.pi / 4)
```

    pip install geomrust-py

Not yet published to PyPI — coming with the first release.

## Why

Most of what a CAD kernel needs — parametric evaluation, transformations,
curve-on-surface math — is useful on its own, to anyone doing computational
geometry, not just people building a full modeling system. `geomrust` grows
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
