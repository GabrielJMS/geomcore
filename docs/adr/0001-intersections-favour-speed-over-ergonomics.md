# Intersections favour speed over ergonomics

Superseded by [ADR 0011](0011-intersections-route-to-the-exact-solution-else-numeric.md): `intersect` now routes to a numeric search when no closed form exists, and every result says which method ran. The typed path described here lives on as `intersect_exact`.

Exact intersections are reached through one generic `Intersect<B>` trait, implemented for both argument orders, whose output is the pair's own result type: dispatch is static, results stay on the stack, and the compiler knows what each pair can return. An untyped path on `Surface` and `Curve3D` serves callers that only know the shape at run time, returning one general result per family (surface–surface, curve–surface, curve–curve) held in fixed-capacity inline storage. Neither path falls back to a numeric intersection on its own: a configuration with no closed form reports "not analytic", and the caller decides whether to march or refine.

## Considered options

- **One general result as the only interface** (fewest types to learn): rejected. Every call would pay for a general result, some would allocate, and the compiler would no longer rule out impossible outcomes, such as a hyperbola from plane–sphere.
- **Automatic numeric fallback** on the untyped path: rejected. An exact intersection takes nanoseconds and marching takes milliseconds. A hidden switch between the two makes run time unpredictable, and the returned curves would silently change from exact to approximate.
- **Keeping the per-pair `intersect_*` methods public** alongside the trait: rejected. That would mean two interfaces for the same behaviour. They are deprecated in 0.29 and become crate-private in the next breaking release.
