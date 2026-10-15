# Surface normals are unit, with second-derivative and step-off recovery

`ParametricSurface::normal(u, v, tol)` returns the unit `Su × Sv`, or `None` where every recovery rung fails. Recovery runs direct cross, then second-derivative substitution for the degenerate direction (each pair from `{Su, Suv, Suu} × {Sv, Suv, Svv}`, oriented by domain half), then growing diagonal step-offs — so poles and apexes resolve their limiting normal instead of reporting undefined. Raw partials stay on `eval_d1`; there is no `tangent`. A `normal_at(point)` convenience (strict: `None` off-surface) is intended but deferred — no in-tree caller needs it, and both references evaluate normals strictly at parameters.

## Considered options

- **Raw cross product**: rejected. Magnitude carries parametrization speed, not geometry.
- **Direct cross only (`None` at any singularity)**: rejected. Poles and apexes are ordinary geometry (offsets need a normal there); the D2 rung is ~30 lines on the existing `eval_d2`.
- **Arbitrary-order derivative ladder for stubborn singularities**: not now; the D2 rung plus step-off resolve every analytic singularity, and higher orders need a `DN`-past-2 capability with no other consumer yet.
- **Point variant now**: deferred. Additive later; semantics recorded above for when a caller appears.

## Consequences

`Plane::normal()` (inherent, zero-arg) shadows the trait method on `Plane` — use `ParametricSurface::normal(&plane, …)`. Kept deliberately: it matches the references' accessor style (usd-brep keeps zero-arg `.normal()` on planar curves, never on surfaces) and avoids a breaking rename. The same shadowing keeps the parametric form off `PyPlane` in the Python bindings (pyo3 cannot overload); the other surface classes expose `normal(u, v, tol)`.
