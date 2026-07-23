# Roadmap

`geomcore` grows from a parametric-evaluation core toward a full geometric-query
library. This roadmap is ordered by dependency, not by date — items build on the
ones before them. If you'd like to help with any of it, see
[Contributing](#contributing) below.

## Shipped (0.1)

- Parametric evaluation for lines, conics, and B-spline curves (rational and
  periodic), and for planes, cylinders, cones, spheres, tori, and B-spline
  surfaces — in Rust and Python.
- Rigid transformations via a single `Transform` type.
- Analytic curve-on-surface parametrization for lines and circles on the five
  elementary surfaces.
- Golden-fixture validation of every numeric result at 1e-7 tolerance.

## Next: geometric queries

1. **Numerical foundations** — robust real-root solving for low-degree
   polynomials with tolerance-aware root clustering (so tangencies are detected,
   not lost to floating-point noise), and Newton iteration for small systems.
2. **Point projection and extrema** — closest point(s) from a point to any
   curve or surface: closed-form on elementary shapes, Newton-based on
   B-splines.
3. **Analytic surface intersections** — plane–plane, plane–cylinder,
   plane–cone, plane–sphere, sphere–sphere, line/conic vs. quadric, and
   friends — returning first-class curve types together with their
   curve-on-surface parametrizations, with explicit tangency and coincidence
   classification.
4. **Curve–curve intersection and extrema** (2D and 3D) — analytic conic pairs
   first, then subdivision + Newton refinement for B-splines.
5. **Generic curve–surface intersection.**

## Later

- B-spline interpolation and approximation from point data.
- General surface–surface intersection (marching + B-spline approximation of
  the result).
- Numeric fallback for curve-on-surface parametrization when no closed form
  exists.
- Python type stubs for IDE autocompletion, and hosted documentation with
  tutorials.

## Someday

- Topology / B-rep. Explicitly out of scope until the geometry layer above is
  solid — `geomcore` should stay useful to people who only need the math.

## Contributing

Contributions are very welcome, at every stage of this roadmap.

- **Pick an item and open an issue first** so we can agree on scope and API
  shape before you invest time.
- **The quality bar:** every numeric feature lands with golden-fixture tests
  (1e-7 tolerance), rustdoc with doctests, and — where it's user-facing —
  Python bindings and tests.
- **Good first contributions:** new curve/surface types via the
  `ParametricCurve3D` / `ParametricSurface` traits (no changes to core dispatch
  needed), examples, documentation, and benchmarks.
