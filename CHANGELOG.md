# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The Rust crate and
the Python distribution are versioned in lockstep.

## [Unreleased]

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

[Unreleased]: https://github.com/GabrielJMS/geomcore/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/GabrielJMS/geomcore/releases/tag/v0.1.0
