# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The Rust crate and
the Python distribution are versioned in lockstep.

## [Unreleased]

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

[Unreleased]: https://github.com/GabrielJMS/geomcore/compare/v0.24.0...HEAD
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
