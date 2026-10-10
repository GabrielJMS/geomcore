# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The Rust crate and
the Python distribution are versioned in lockstep.

## [Unreleased]

## [0.26.0] - 2026-10-11

### Added

- `transformed(transform)` on every 3D curve and surface (`Line3D`,
  `Circle3D`, `Ellipse3D`, `Parabola3D`, `Hyperbola3D`,
  `BSplineCurve3D`, `Plane`, `Cylinder`, `Cone`, `Sphere`, `Torus`,
  `BSplineSurface`). Analytic types map their frame and scale their
  radii by the uniform scale factor (a mirror re-derives a right-handed
  frame, reversing the angular direction) and fail only for a
  collapsing transform; B-splines map every control pole. Python:
  same-named methods (analytic types raise `ValueError` on collapse).
- Second-order derivatives for `BSplineSurface::eval_derivative`
  (`(2,0)`, `(1,1)`, `(0,2)`, rational via the quotient rule); the
  fixture goldens now carry them.

### Changed

- B-spline curve and surface evaluation rewritten around the local
  basis functions (The NURBS Book A2.2/A2.3) over the `p+1` (resp.
  `(p+1)(q+1)`) affected poles with fixed-size stack tables, instead of
  copying all poles per call: 10-30x faster on large control nets,
  results unchanged to 1e-14.
- `gauss_newton_1d`/`gauss_newton_2d` evaluate each trial residual once.
- `BSplineSurface::project_point` and `extrema` are exact. Each
  knot-span patch (from a bounding-box hierarchy over the patch poles,
  built on the first projection and visited by increasing box distance)
  is converted to Bernstein form and halved, closest control point
  first. A piece is dropped once its control points are provably
  farther than the best distance so far, and solved by a box-bounded
  Newton minimisation with exact second derivatives once interval
  bounds on the derivatives of the squared distance show it is convex
  or that its minimum lies on one edge or corner; otherwise it is
  halved again. Minima on crease and boundary edges are found on the
  edge curve itself. `extrema` runs the same search on every patch
  without pruning, so it returns every local minimum, its first entry
  being `project_point`'s result. Against dense brute force on 6x6 to
  80x80 nets the previous version missed the global minimum for 7-15%
  of query points (by up to 106 units); on 5,400 random surfaces
  (rational, periodic, creased) the new one is never farther than a
  reference kernel or brute force, at 9-52 us per projection. `extrema`
  searches every patch, so it costs more on large nets (3.4 ms on
  80x80).
- `BSplineCurve3D::project_point` is exact: a branch and bound over the
  knot spans (bounding boxes of the span poles, visited nearest first)
  in which each visited span is scanned exactly. On a span the
  derivative of the squared distance has the sign of a polynomial
  (degree `2p - 1`, `3p - 1` if rational); its rising roots are
  bracketed in Bernstein form (Descartes' rule of signs with de
  Casteljau halving) and refined by a Newton minimisation bounded to
  the bracket, so every minimum is found however close basins sit, and
  the foot point is exact to rounding. On 9,900 random curves (degree
  1-5, rational, periodic, interior knots of any multiplicity) none is
  farther than a dense brute-force search; the previous sampling-based
  search missed 3.9% of them. Speed is unchanged (1-8 us at degree <= 5).
- `BSplineCurve3D::extrema` returns the local minima of the distance
  (plus the ends of an open curve), found by the same exact scan over
  every span; a knot is a minimum when the distance falls into it and
  rises out of it. Its documentation (and `BSplineSurface::extrema`'s,
  in Python) promised all stationary points, which neither returns.
- The bounded Newton minimisation behind B-spline projection accepts a
  step that shrinks the gradient once the squared distance is flat to
  rounding, so the foot converges to full precision (it stalled near
  `sqrt(eps)` in the parameter).
- `Circle3D` intersection with a conic in a parallel, offset plane
  returns `Empty` (it returned `NotAnalytic`).
- `real_roots` (behind every analytic intersection and conic
  projection) scales the polynomial exactly by powers of two, splits it
  into real linear and quadratic factors refined by Newton, and solves
  roots of very different magnitudes each at its own scale. It is about
  2x slower (150-630 ns) for far fewer failures (below).
- Line intersections with spheres, cylinders, cones, tori and in-plane
  conics form their equation about the line's closest approach to the
  geometry, so the tangency band is `tol.confusion` around the chord's
  midpoint wherever the line's origin lies.
- Tangency in curve-surface intersections is decided by distance: a
  line, parabola or hyperbola whose closest approach to a plane, sphere,
  cylinder, cone or torus is within `tol.confusion` of it reports a
  single tangent hit at the touching point. It was decided by how far
  apart the roots fell in parameter space, which construction rounding
  splits into two hits or none (a line tangent to a torus equator was
  missed in 43 of 300 randomized cases).
- `ConicSurfaceIntersection::Coincident` also covers an ellipse,
  parabola or hyperbola lying entirely on a sphere, cylinder or cone
  (a plane section): these reported one to three isolated hits, or
  none.
- `curve_curve_extrema` returns the local minima of the distance over
  the curves' parameter domains, minima at the ends of bounded curves
  included. Two straight curves are solved in closed form, a full line
  against another curve reduces to that curve's distance to the line,
  and other pairs are minimised over the parameter box by a bounded
  Newton with exact second derivatives, each result checked to be a
  true minimum on the box. Unbounded parameters are seeded from the
  curves' own length scale instead of a fixed `[-10, 10]` window.
  Parallel lines, concentric circles and overlapping curves report the
  distance once, with a representative pair (they returned nothing).
  3-16x faster on the pairs measured.
- Python: `eval_derivative` raises `ValueError` for an unsupported
  order (order 0, or above 2 for B-splines and surfaces) instead of a
  `PanicException`.

### Fixed

- B-spline `extrema`/`project_point` silently dropped seeds whose
  Gauss-Newton refinement did not converge, which could return a far
  local minimum (e.g. 44.4 instead of 2.35); neither search seeds any
  more (see Changed).
- Periodic B-spline derivatives just below the last parameter came from
  the first span: wrapping the parameter into the period could round it
  onto the first knot. A parameter already inside the period is left
  unchanged.
- `Circle3D` intersection with an ellipse, parabola or hyperbola whose
  frame's x-axis differs from the circle's lost every hit: the angles
  were solved in the conic's frame but evaluated in the circle's.
- `Ellipse3D`/`Parabola3D`/`Hyperbola3D::contains` agree with
  `project_point`: a point within tolerance of the curve but where
  parameter inversion lands elsewhere (far out on an arm, say)
  was rejected (38, 285 and 54 wrong answers in 1,800 randomized checks
  per curve type).
- `real_roots`: a small nonzero leading coefficient was treated as zero,
  dropping the large root it implies; small roots next to large ones lost
  their precision; huge coefficients produced NaN and panicked; tiny ones
  underflowed into false double roots. Exact triple and quadruple roots
  now report their full multiplicity. On about 7,200 stress polynomials
  per seed, failures against 80-digit arithmetic drop from about 2,000
  (and about 115 panics) to 2, both errors of the reference itself.
- Removed a leftover debug `eprintln!` from the quartic solver that
  printed to stderr on every call since 0.24.0.
- Periodic B-spline curves and surfaces with a seam multiplicity above
  1 evaluated against the wrong poles: pole 0 now starts at the span
  opened by the last copy of the first knot. Pinned by new fixture
  cases verified in exact arithmetic.
- B-spline derivatives exactly on an interior knot now come from the
  span to its right (they came from the left one, which differs at
  C0/C1 knots); on a periodic curve only exactly the last parameter
  closes the last span, other seam copies wrap to the first.
- `Ellipse3D` projection/`extrema` near the parameter `pi`: the
  `tan(t/2)` quartic degenerates there, so on-curve points could
  project up to 10.6 away on a flat ellipse. The reversed quartic now
  covers that half, and every root is Newton-polished on the
  trigonometric stationarity equation.
- `Hyperbola3D` projection/`extrema` roots are Newton-polished on the
  hyperbolic stationarity equation (on-curve points of a thin
  hyperbola projected 3.4e-6 away).
- `Ellipse3D`/`Parabola3D`/`Hyperbola3D::project_point` panicked on a
  query point with a NaN or infinite coordinate; it now returns a NaN
  parameter and distance (and `extrema` an empty list). A hyperbola
  with both radii zero panicked on every projection.
- `curve_curve_extrema` on trimmed curves reported points stuck at a
  domain end as extrema and could miss the true minimum (5.00637
  instead of 5 for concentric arcs); on random trimmed arc pairs its
  minimum was worse than a 400-sample grid in 17 of 60 general and 32
  of 60 concentric pairs. It also missed minima of lines, parabolas and
  hyperbolas beyond parameter 10, and returned nothing for some pairs
  of skew ellipses.
- `Ellipse3D::intersect_ellipse` reported a tangency on an axis vertex
  `sqrt(eps)` off the axis (4.2e-8 for unit-sized ellipses); it is now
  exact.
- `Parabola3D` intersections with a plane missed roots far out on the
  arms (around `u = 3.4e9`), where checking the hit on the surface
  rounds by more than `tol.confusion`.

## [0.25.0] - 2026-09-30

### Added

- Analytic line/circle-vs-conic intersections: `intersect_ellipse/
  parabola/hyperbola` on `Line3D` (plane-piercing graze/miss, coplanar
  quadratic substitution) and on `Circle3D` (coplanar trigonometric
  solve, circular-ellipse coincidence, `NotAnalytic` otherwise),
  sharing `LineConic3DIntersection`/`CircleConic3DIntersection` with
  both curve parameters per hit. Hits on the unmodeled hyperbola
  branch are dropped. Python: per-type methods returning
  `(kind, payload)` tuples.

## [0.24.0] - 2026-09-30

### Added

- Analytic conic-vs-quadric intersections: `intersect_plane/sphere/
  cylinder/cone` on `Ellipse3D`, `Parabola3D`, and `Hyperbola3D`
  (linear trigonometry, quartics in `tan(t/2)`/`t`/`e^t`, tangency
  multiplicities, whole-curve coincidence, nappe filtering), sharing
  `ConicSurfaceIntersection`. Cross-validated against the generic
  solver. Python: per-type methods returning `(kind, payload)` tuples.

## [0.23.0] - 2026-09-30

### Added

- Analytic circle-vs-quadric intersections: `Circle3D::intersect_plane/
  sphere/cylinder/cone` (linear trigonometry for planes/spheres,
  double-angle quartics otherwise, whole-circle detection by
  three-sample containment, tangency multiplicities). Python returns
  `(kind, payload)` tuples.
- Closed-geometry measures: `Circle3D::circumference/disk_area`,
  `Ellipse3D::area`, `Sphere::area/volume` (exact; ellipse perimeter
  deliberately absent — no closed form).

## [0.22.0] - 2026-09-30

### Added

- B-spline least-squares approximation: `BSplineCurve3D::approximate`
  (uniform interior knots, normal equations by band LU with an explicit
  support guard). Python: `BSplineCurve3D.approximate(points, degree,
  num_poles)`.

## [0.21.0] - 2026-09-30

### Added

- Generic 3D curve–curve extrema and intersection
  (`curve_curve_extrema`, `intersect_curve_curve`, over
  `ParametricCurve3D`): coarse grid seeds refined by Gauss-Newton on
  `|C1(s) - C2(t)|^2`, meetings filtered by tolerance. Python:
  `geomcore.curve_curve_extrema` and `geomcore.intersect_curve_curve`.

## [0.20.0] - 2026-09-30

### Added

- Numeric curve-on-surface parametrization fallback
  (`parametrize_numeric`, generic over both traits): uniform samples
  inverted through surface projection, validated against every analytic
  pcurve. Python: `geomcore.parametrize_numeric(curve, surface, n=64)`.

## [0.19.0] - 2026-09-30

### Added

- Generic marching surface-surface intersection
  (`marching_intersection`, also `Surface::intersect_marching`): grid
  seeds projected both ways, adaptive tangent tracing with alternating
  projections, loop/boundary termination, `BSplineCurve3D` fits.
  Approximate by design (fits, no tangency classification, no singular
  points). Python: `geomcore.intersect_marching(a, b)` returning a list
  of `BSplineCurve3D`.

## [0.18.0] - 2026-09-30

### Added

- Points-to-B-spline interpolation: `BSplineCurve3D::interpolate`
  (chordal/centripetal parametrization, averaged knots, banded LU —
  the tracing-to-curve primitive marching SSI will build on). Python:
  `BSplineCurve3D.interpolate(points, degree)`.

## [0.17.0] - 2026-09-30

### Added

- Generic curve–surface intersection (`intersect_curve_surface`, free
  function over `ParametricCurve3D` × `ParametricSurface`): curve samples
  projected onto the surface seed Newton iteration on `C(t) - S(u,v) =
  0`, reporting ordered transversal hits. `project_point` joined the
  `ParametricSurface` trait so every surface works uniformly. Also
  available as top-level `geomcore.intersect_curve_surface` in Python.

## [0.16.0] - 2026-09-30

### Added

- Analytic curve intersections, batch 2: `Line3D::intersect_circle`
  (transversal graze or in-plane 2D reduction) and
  `Circle3D::intersect_circle` (coplanar pairs via shared-plane 2D
  images, `NotAnalytic` otherwise). Python returns `(kind, payload)`
  tuples.

## [0.15.0] - 2026-09-30

### Added

- Analytic curve intersections, batch 1: `Line2D::intersect_line/
  intersect_circle`, `Circle2D::intersect_circle`, and
  `Line3D::intersect_line` (transversal hits with both parameters,
  tangencies, parallel/coincident/skew classification with distances).
  Python returns `(kind, payload)` tuples.

## [0.14.0] - 2026-09-30

### Added

- Point projection, extrema, and containment on `BSplineSurface`:
  dense grid seeds (boundaries included) refined by 2D Gauss-Newton
  (first derivatives only, clamped into non-periodic directions).
  Python returns `(u, v, distance)` tuples. Surface containment is now
  complete across every surface type.

## [0.13.0] - 2026-09-30

### Added

- Point projection, extrema, and containment on `BSplineCurve3D`:
  dense seeds refined by Gauss-Newton (first derivatives only),
  endpoint candidates on open curves, batch variants. Python returns
  `(parameter, distance)` tuples. Curve containment is now complete
  across every curve type.

## [0.12.0] - 2026-09-30

### Added

- Point projection and extrema on conics: `extrema(point, tol)` (all
  stationary points by ascending distance, via quartic/cubic stationarity
  solved with the `math` module) plus `project_point`/`project_points`
  on `Ellipse3D`, `Parabola3D`, and `Hyperbola3D`. Python returns
  `(parameter, distance)` tuples and lists thereof.

## [0.11.0] - 2026-09-30

### Added

- Internal numerical toolbox (`src/math`, zero dependencies): real roots
  of degree ≤ 4 polynomials (stable quadratic, Cardano/trigonometric
  cubic, Ferrari quartic with biquadratic shortcut, Newton polish,
  tolerance clustering with multiplicity so tangencies survive),
  1D Newton, Gauss-Newton drivers for curve/surface projection, Brent
  minimum, and 2x2 solves. Powers projection and extrema next.

## [0.10.0] - 2026-09-29

### Added

- Analytic intersections: `Cone::intersect_cone` for coaxial cones
  (latitude ring, apex point, coincident, empty, `NotAnalytic`).
- Analytic intersections, torus-symmetric batch:
  `Torus::intersect_plane` (meridian/latitude rings),
  `intersect_sphere/cylinder/cone` (coaxial latitude rings via axial
  quadratics with nappe filtering) and `intersect_torus` (coaxial
  rings, coincident). Non-symmetric pairs report `NotAnalytic`.
  Python returns `(kind, payload)` tuples throughout.

## [0.9.0] - 2026-09-29

### Added

- Analytic intersections: `Cone::intersect_cylinder` for coaxial
  cone–cylinder pairs (single latitude ring, apex point for degenerate
  tubes, `NotAnalytic` otherwise). Python returns `(kind, payload)`
  tuples.

## [0.8.0] - 2026-09-29

### Added

- Analytic intersections, Stream 2 batch 3 (symmetric quadrics):
  `Sphere::intersect_cylinder` and `Sphere::intersect_cone` (latitude
  rings, single/tangent/empty) and `Cylinder::intersect_cylinder`
  (generators, tangent, coincident, empty). Each checks its analytic
  criterion first (axis through center / on axis / parallel axes) and
  reports `NotAnalytic` otherwise — the reserved hook for the future
  numeric surface-surface path, never a silent miss. Python returns
  `(kind, payload)` tuples throughout.

## [0.7.0] - 2026-09-29

### Added

- Analytic intersections, Stream 2 batch 2: `Plane::intersect_cylinder`
  (→ circle / ellipse / two lines / tangent line / empty),
  `Plane::intersect_cone` (→ circle / ellipse / parabola / hyperbola
  branch / two apex lines / tangent generator / apex point / empty, with
  single-nappe filtering), and `Line3D::intersect_plane/sphere/cylinder/
  cone` (→ ordered hits with parameters, tangent, single transversal
  hit, empty, coincident). A shared tolerance-aware quadratic solver
  classifies by root separation, so grazing contact is robust across
  scales. Python returns `(kind, payload)` tuples throughout.

## [0.6.0] - 2026-09-29

### Added

- Analytic intersections, Stream 2 batch 1: `Plane::intersect_plane`
  (→ `Line3D`, direction = n1 × n2, with `Parallel`/`Coincident`
  classification), `Plane::intersect_sphere` and
  `Sphere::intersect_sphere` (→ `Circle3D`, with
  `TangentPoint`/`Empty`/`Coincident` classification). Near-degenerate
  configurations classify by `Tolerance` instead of collapsing to noise.
  Python returns `(kind, payload)` tuples, e.g. `("line", Line3D)`,
  `("tangent_point", Point3D)`, `("empty", None)`.

## [0.5.0] - 2026-09-29

### Added

- Closed-form point projection: `project_point(point, tol)` and batch
  `project_points(points, tol)` on `Line3D`, `Circle3D`, `Line2D`,
  `Circle2D` (returning `CurveProjection { parameter, distance }`) and on
  `Plane`, `Cylinder`, `Cone`, `Sphere`, `Torus` (returning
  `SurfaceProjection { u, v, distance }`). Sub-tolerance distances snap
  to `0.0`, so projection agrees with `contains`. Python returns
  `(parameter, distance)` / `(u, v, distance)` tuples.
  (Conic and B-spline projection arrive with the `math` foundations.)

## [0.4.0] - 2026-09-29

### Added

- Point containment: `contains(point, tol)` on all analytic surfaces
  (`Plane`, `Cylinder`, `Cone`, `Sphere`, `Torus`), verified by
  inverse-parameter re-evaluation against `tol.confusion`.
  Python: `contains(point, tol=None)`.
  (`BSplineSurface` containment arrives with numeric projection.)

## [0.3.0] - 2026-09-29

### Added

- Public `Tolerance` API (`confusion`, `angular`, `parametric`, with
  `Tolerance::DEFAULT` matching the kernel's historical internal values);
  internal tolerances now derive from it. Python: `geomcore.Tolerance`.
- Point containment: `contains(point, tol)` on all analytic curves
  (`Line3D`, `Circle3D`, `Ellipse3D`, `Parabola3D`, `Hyperbola3D`,
  `Line2D`, `Circle2D`), verified by inverse-parameter re-evaluation
  against `tol.confusion`. Python: `contains(point, tol=None)`.
  (`BSplineCurve3D` containment arrives with numeric projection.)

## [0.2.0] - 2026-09-29

### Changed

- **Breaking:** renamed value and placement types for `2D`/`3D` consistency
  with the curve names: `Point2` → `Point2D`, `Point3` → `Point3D`,
  `Vector2` → `Vector2D`, `Vector3` → `Vector3D`, `Axis2` → `Axis2D`,
  `Axis3` → `Axis3D`, `Frame2` → `Frame2D`, `Frame3` → `Frame3D`
  (Rust and Python).
- **Breaking (Python):** geometric objects are now constructed with
  `__init__` only (e.g. `Circle3D(Point3D.origin(), Vector3D.z(), 2.0)`);
  the duplicate `.new()` staticmethods were removed.

## [0.1.0] - 2026-07-02

### Added

- Parametric evaluation (point, first and second derivatives, bulk
  `eval_points`) for lines, circles, ellipses, parabolas, hyperbolas, and
  B-spline curves (rational and periodic) in 3D, plus 2D lines and circles.
- Parametric evaluation for planes, cylinders, cones, spheres, tori, and
  B-spline surfaces.
- Rigid transformations (translation, rotation, mirroring, scaling) via a
  single `Transform` type, applicable to all geometry.
- Analytic curve-on-surface parametrization for lines and circles on the five
  elementary surfaces (plane, cylinder, cone, sphere, torus).
- Python bindings (`pip install geomcore`) with native `geomcore.curves` and
  `geomcore.surfaces` submodules, for Python 3.9+ (abi3).
- Golden-fixture validation of all numeric results at 1e-7 tolerance, and
  criterion throughput benchmarks (`docs/benchmarks.md`).

[Unreleased]: https://github.com/GabrielJMS/geomcore/compare/v0.26.0...HEAD
[0.26.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.26.0
[0.25.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.25.0
[0.24.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.24.0
[0.23.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.23.0
[0.22.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.22.0
[0.21.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.21.0
[0.20.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.20.0
[0.19.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.19.0
[0.18.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.18.0
[0.17.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.17.0
[0.16.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.16.0
[0.15.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.15.0
[0.14.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.14.0
[0.13.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.13.0
[0.12.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.12.0
[0.11.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.11.0
[0.10.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.10.0
[0.9.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.9.0
[0.8.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.8.0
[0.7.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.7.0
[0.6.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.6.0
[0.5.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.5.0
[0.4.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.4.0
[0.3.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.3.0
[0.2.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.2.0
[0.1.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.1.0
