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
- **0.21** — Generic 3D curve–curve intersection and extrema.
- **0.22** — B-spline least-squares approximation.
- **0.23–0.25** — Analytic intersections of circles and conics with the
  quadrics, and of lines and circles with conics. Closed-form measures
  (circumference, areas, sphere volume).
- **0.25** — Hand-written Python type stubs (PEP 561) for IDE support.
- **0.26** — `transformed` on every 3D curve and surface. Exact projection
  and extrema on B-spline curves and surfaces. Curve–surface tangency
  decided by distance. Curve–curve extrema as the local distance minima
  over the curves' domains.
- **0.27** — Double-nappe cones in every intersection. Constructors reject
  NaN, infinite inputs and zero-size conics. Marching intersection finds
  curves between its seed grid lines and traces each loop once.
  Villarceau circles in torus–plane intersection.
- **0.28** — Trimmed curves (segments, arcs, pieces of B-splines) whose
  numeric intersections and extrema only search their range.
- **0.29** — One entry point for exact intersections, `a.intersect(&b)` in
  either order, also on shapes whose type is only known at run time; the
  per-pair methods are deprecated. Hits carry a multiplicity. Trimmed
  curves in Python.
- **0.30** — Fixes: tangent contacts of circles and ellipses with conics
  and quadrics near half a turn, circles meeting a cylinder or cone at
  three points, numeric curve–surface hits at a sphere's pole or a cone's
  apex.
- **0.31** — Exact intersections between any two conics, coplanar or in
  crossing planes, with hit multiplicity; faster `contains` on ellipses,
  parabolas and hyperbolas; faster tangency detection through quartics.
- **0.32** — B-spline editing: knot insertion and removal, degree
  elevation, splitting, extraction and reversal, Bézier pieces, arc
  length, and fitting within a deviation. Elementary curves and surfaces
  as exact B-splines; bounding boxes. Curve discretization with proven
  deviations. B-spline intersections and extrema pruned with Bézier hulls,
  reporting whether their search was complete. Faster evaluation, fitting
  and projection.
- **0.33–0.35** — Isoparametric curves `u_iso` / `v_iso` on every
  surface, in Rust and Python. Point coincidence within the confusion
  tolerance; exact value equality in Python.
- **0.36** — `a.intersect(&b)` answers every pair of curves and
  surfaces, B-splines and trimmed curves included: exact when a closed
  form exists, numeric otherwise, and the result says which method ran
  and whether every hit was found. The typed exact path becomes
  `intersect_exact`. Marching seeds from exact iso-curve intersections,
  so curves far along unbounded surfaces are found, and runs to the
  model-size cube. Projections give the closest point. Python results
  are objects holding their operands, Python exceptions mirror the Rust
  errors, and the bindings cover the Rust API.

## Next

1. **Marching intersection** — trace through points where the two
   surfaces touch, and find isolated loops smaller than the iso-curve
   spacing on surfaces other than B-splines (two B-spline surfaces
   already find them from overlapping patch pairs, #30).
2. **Hosted documentation** — tutorials on the docs site, and a Python
   API reference generated from docstrings.

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
