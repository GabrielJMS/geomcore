# Surface normals are unit; point variant deferred

`ParametricSurface::normal(u, v, tol)` returns the unit `Su × Sv`, or `None` where degenerate: a partial at or below `tol.confusion` is numerical noise rather than a direction, otherwise the sine of the angle between the partials is checked against `tol.angular`. Raw partials stay on `eval_d1`; there is no `tangent` method. A `normal_at(point)` convenience (strict: `None` off-surface, mirroring SMLib's refusal to evaluate outside its domain) is intended but deferred — no in-tree caller needs it, and both references evaluate normals strictly at parameters.

## Considered options

- **Raw cross product**: rejected. Magnitude carries parametrization speed, not geometry.
- **`D2`/step-off singular recovery (SMLib-style)**: not now; layerable later without signature change.
- **Point variant now**: deferred. Additive later; semantics recorded above for when a caller appears.

## Consequences

`Plane::normal()` (inherent, zero-arg) shadows the trait method on `Plane` — use `ParametricSurface::normal(&plane, …)`. Kept deliberately: it matches the references' accessor style (usd-brep keeps zero-arg `.normal()` on planar curves, never on surfaces) and avoids a breaking rename. The same shadowing keeps the parametric form off `PyPlane` in the Python bindings (pyo3 cannot overload); the other surface classes expose `normal(u, v, tol)`.
