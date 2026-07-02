# geomcore

[![crates.io](https://img.shields.io/crates/v/geomcore.svg)](https://crates.io/crates/geomcore)
[![docs.rs](https://img.shields.io/docsrs/geomcore)](https://docs.rs/geomcore)
[![PyPI](https://img.shields.io/pypi/v/geomcore.svg)](https://pypi.org/project/geomcore/)
[![CI](https://github.com/GabrielJMS/geomcore/actions/workflows/ci.yml/badge.svg)](https://github.com/GabrielJMS/geomcore/actions/workflows/ci.yml)

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

See [ROADMAP.md](ROADMAP.md) for what's coming next and
[CHANGELOG.md](CHANGELOG.md) for release history.

## Rust

```rust
use geomcore::{Point3, Vector3};
use geomcore::curves::Circle3D;

let circle = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
let point = circle.eval_point(std::f64::consts::PI / 4.0);
```

    cargo add geomcore

API reference: [docs.rs/geomcore](https://docs.rs/geomcore).

## Python

```python
from geomcore import Point3, Vector3
from geomcore.curves import Circle3D
import math

circle = Circle3D.new(Point3.origin(), Vector3.z(), 2.0)
point = circle.eval_point(math.pi / 4)
```

    pip install geomcore

Pre-built wheels for Linux (x86_64, aarch64), macOS, and Windows,
Python 3.9+ (abi3 — one wheel covers all versions).

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
