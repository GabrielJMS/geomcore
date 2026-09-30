# Roadmap

`geomcore` grows from a parametric-evaluation core toward a full geometric-query
library. This roadmap is ordered by dependency, not by date — items build on the
ones before them. If you'd like to help with any of it, see
[Contributing](#contributing) below.

## Shipped

- **0.1** — Parametric evaluation for lines, conics, and B-spline curves
  (rational and periodic), and for planes, cylinders, cones, spheres, tori,
  and B-spline surfaces — in Rust and Python. Rigid transformations via a
  single `Transform` type. Analytic curve-on-surface parametrization for
  lines and circles on the five elementary surfaces. Golden-fixture
  validation of every numeric result at 1e-7 tolerance.
- **0.2** — `2D`/`3D` type renames; `__init__`-only Python constructors.
- **0.3–0.4** — Public `Tolerance` API; point containment on all analytic
  curves and surfaces.
- **0.5** — Closed-form point projection (with batch variants) on lines,
  circles, planes, cylinders, spheres, cones, and tori.
- **0.6–0.10** — Analytic intersections with tangency/coincidence
  classification: plane pairs, plane–quadrics, sphere–sphere, line vs.
  quadrics, symmetric quadric pairs, coaxial torus pairs, cone–cone.
  Non-analytic configurations report `NotAnalytic` (never silent).
- **0.11** — Internal numerical toolbox (real polynomial roots with
  multiplicity, Newton, Gauss-Newton, Brent), zero dependencies.
- **0.12–0.14** — Projection and extrema on conics (quartic/cubic
  stationarity), B-spline curves, and B-spline surfaces; containment
  complete across every curve and surface type.
- **0.15–0.16** — Analytic curve–curve intersections: 2D pairs,
  3D line–line, line–circle, and coplanar circle–circle.
- **0.17** — Generic curve–surface intersection over the
  `ParametricCurve3D` × `ParametricSurface` traits.
- **0.18** — Points-to-B-spline interpolation (chordal/centripetal,
  averaged knots, banded LU).
- **0.19** — Marching surface–surface intersection with B-spline fits.
- **0.20** — Numeric curve-on-surface parametrization fallback, validated
  against every analytic pcurve.

## Next

1. **Generic curve–curve intersection and extrema** — subdivision +
   Newton refinement for B-splines in 2D; closest-point enumeration in
   3D (analytic pairs already ship).
2. **B-spline approximation** (least-squares fit, complementing
   interpolation).
3. **Python type stubs** for IDE autocompletion, and hosted documentation
   with tutorials.

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
