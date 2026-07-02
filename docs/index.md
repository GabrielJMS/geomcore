# geomcore

A pure, standalone geometric kernel for CAD-grade curves and surfaces —
parametric evaluation, rigid transformations, and analytic curve-on-surface
parametrization, implemented in Rust with idiomatic Python bindings.

`geomcore` is not a modeling/CAD system — no topology, no B-rep, no boolean
operations (yet). It's the layer underneath one, usable on its own by anyone
who needs robust computational geometry.

## What's inside

- **Curves**: lines, circles, ellipses, parabolas, hyperbolas, and B-spline /
  NURBS curves (rational and periodic), all with point evaluation, derivatives
  of any order, parameter inversion, and bulk evaluation.
- **Surfaces**: planes, cylinders, cones, spheres, tori, and B-spline / NURBS
  surfaces, with the same evaluation API over `(u, v)`.
- **Transforms**: translation, rotation, mirroring, and scaling through a
  single `Transform` type.
- **Curve-on-surface parametrization**: closed-form 2D representations of
  lines and circles in the parameter space of the five elementary surfaces —
  the building block CAD kernels call *pcurves*.

Every numeric result is validated against independently generated golden
fixtures at CAD-kernel-grade tolerance (1e-7) — see
[Validation](validation.md).

## Install

=== "Python"

    ```sh
    pip install geomcore
    ```

    Pre-built wheels for Linux (x86_64, aarch64), macOS, and Windows,
    Python 3.9+.

=== "Rust"

    ```sh
    cargo add geomcore
    ```

    API reference on [docs.rs/geomcore](https://docs.rs/geomcore).

## Where to go next

- [Quickstart](quickstart.md) — evaluate your first curve in two minutes.
- [Tutorials](tutorials/geometry-basics.md) — from points and frames to NURBS
  and pcurves.
- [Migrating from NURBS-Python](migrating-from-geomdl.md) — a concept map for
  `geomdl` users.
- [Roadmap](https://github.com/GabrielJMS/geomcore/blob/main/ROADMAP.md) —
  what's coming, and how to contribute.

## License

Dual-licensed under MIT or Apache-2.0, at your option.
