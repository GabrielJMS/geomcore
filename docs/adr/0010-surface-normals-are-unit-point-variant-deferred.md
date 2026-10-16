# Surface normals are unit, with second-derivative and step-off recovery

`ParametricSurface::normal(u, v, tol)` returns the unit `Su × Sv`, or `None` where every recovery rung fails. Recovery runs direct cross, then second-derivative substitution for the degenerate direction (each pair from `{Su, Suv, Suu} × {Sv, Suv, Svv}`, oriented by domain half), then growing diagonal step-offs — so poles, apexes and collapsed B-spline rows resolve their limiting normal instead of reporting undefined. Raw partials stay on `eval_d1`; there is no `tangent`. A `normal_at(point)` convenience (strict: `None` off-surface) is intended but deferred — no in-tree caller needs it, and normals are evaluated at parameters everywhere in the crate.

## A derivative is zero when it covers no distance across its range

A derivative has the unit length per parameter unit (per parameter unit squared for a second derivative), so comparing its length with `tol.confusion`, a length, mixes units: the same well-shaped surface reports no normal anywhere once its knots span `[0, 1e8]`, and never degenerates on `[0, 1e-8]`. The rule is instead the distance the derivative would cover across the parameter range:

- a derivative of order `(i, j)` is negligible when `|D| × width_u^i × width_v^j <= tol.confusion` (and when it is `NaN`), `width = last - first` of the parameter bounds;
- an unbounded range is measured across the model: `width = tol.confusion / tol.angular`, the model size the angular tolerance encodes (ADR 0006, 1e5 by default). Every unbounded direction of the elementary surfaces is arc length, so this only matters for a user surface, and it is the largest stretch of the direction any part of the model can use. An exact-zero test alone would let noise-level derivatives of an unbounded direction through as directions;
- the step-off rung steps by fractions (1e-7 to 1e-4) of the same widths, not by fixed parameter amounts.

Rescaling a parameter by `k` divides the derivative by `k` and multiplies the width by `k`, so the decision, and with it the normal, is the same under any affine reparametrization. The helpers are `Tolerance::reference_width(bounds)` and `Tolerance::is_negligible_derivative(length, reach)` (crate-private), `reach` being the width raised to the derivative's order; a curve's unit tangent follows the same rule with `|C'| × width`. On a unit-speed parametrization the test reads as before: a sphere's partial `Su` is negligible where the parallel circle through the point is shorter than `tol.confusion`.

## Considered options

- **Raw cross product**: rejected. Magnitude carries parametrization speed, not geometry.
- **Direct cross only (`None` at any singularity)**: rejected. Poles and apexes are ordinary geometry (offsets need a normal there); the second-derivative rung is ~30 lines on the existing `eval_d2`.
- **Derivative length against `tol.confusion`**: rejected for the unit mismatch above.
- **Derivative length relative to the other partial**: rejected. It is not invariant when only one parameter is rescaled, and fails where both partials vanish.
- **Arbitrary-order derivative ladder for stubborn singularities**: not now; the second-derivative rung plus step-off resolve every analytic singularity, and higher orders need derivatives past order 2 with no other consumer yet.
- **Point variant now**: deferred. Additive later; semantics recorded above for when a caller appears.

## Consequences

`Plane::normal()` (inherent, zero-arg) shadows the trait method on `Plane` — use `ParametricSurface::normal(&plane, …)`. Kept deliberately: a plane's constant normal is the natural zero-argument accessor, and renaming it would break callers. The same shadowing keeps the parametric form off `PyPlane` in the Python bindings (pyo3 cannot overload); the other surface classes expose `normal(u, v, tol)`.

At a point where the surface has no tangent plane (the apex of a collapsed row whose rows meet at an angle), the normal returned is the limit approached across the parameter line where the partial vanishes, at the same value of the other parameter, so it depends on that parameter. A sphere pole, by contrast, has one limiting normal shared by all meridians: callers must not treat a recovered apex normal as canonical. Likewise, a derivative covering less than `tol.confusion` across its parameter range never counts, so sub-confusion geometry still reports `None` everywhere.
