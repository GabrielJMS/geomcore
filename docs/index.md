# geomcore

A pure, standalone geometric kernel for CAD-grade curves and surfaces —
parametric evaluation, rigid transformations, and analytic curve-on-surface
parametrization, built as small, independently useful Rust libraries with a
clean API and idiomatic Python bindings.

- [Elementary curves & surfaces](geometry/elementary.md) — the ten closed forms: formulas, domains, seams.
- [NURBS](nurbs/index.md) — B-spline curves, surfaces, editing and fitting, in Rust and Python.
- [Projection & containment](guides/projection.md), [Intersection](guides/intersection.md), [Discretization](guides/discretization.md), [Curve on surface](guides/parametrization.md) — the query guides.
- [Live viewer](viewer.md) — inspect curves and surfaces in your browser.
- [Python quick start](python.md) — `pip install geomcore`.
- [Roadmap](roadmap.md) — where the kernel is heading.

Rust API reference lives on [docs.rs](https://docs.rs/geomcore) (published
with each release). A Python API reference generated from docstrings is on
the roadmap.
