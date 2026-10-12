# Throughput baseline

Informational numbers from `benches/eval_throughput.rs` (criterion), recorded so
future optimization work has a concrete starting point. They are not a
correctness gate and are expected to shift as internals evolve.

Machine: Linux 6.17, x86-64 desktop, single thread, `cargo bench` defaults
(2026-07-02).

| workload | input size | time / batch | throughput |
|---|---|---|---|
| `Circle3D::eval_points` | 100 000 params | ~1.13 ms | ~88 M evals/s |
| `BSplineCurve3D::eval_points` (degree 3, 20 poles, clamped) | 100 000 params | ~10.5 ms | ~9.5 M evals/s |
| `BSplineSurface::eval_points` (bicubic, 6×6 poles) | 100 000 (u, v) pairs | ~26.1 ms | ~3.8 M evals/s |

The internals behind `eval_points` are deliberately plain scalar loops in this
proof-of-concept; the array-in/array-out API shape exists so that future bulk
optimizations (SoA layouts, SIMD, parallel evaluation) can land behind the same
signatures without breaking callers.

Reproduce with:

```sh
cargo bench --bench eval_throughput
```

# Intersection entry point

`benches/intersect_dispatch.rs` times batches of 1024 exact intersections of
one pair kind four ways: the typed `Intersect::intersect` call, the
deprecated per-pair method it replaces, the untyped call on
`ElementaryCurve`/`ElementarySurface`, and the untyped call on
`Curve3D`/`Surface`. The inputs vary across the batch, so each one mixes
crossings, tangencies and misses.

Machine: Linux 7.0, x86-64 desktop, single thread, `--warm-up-time 1
--measurement-time 3` (2026-10-12). Times are per pair and vary by 5–10%
between runs.

| pair | typed | per-pair method | elementary enums | `Curve3D`/`Surface` |
|---|---|---|---|---|
| plane–sphere | ~66 ns | ~64 ns | ~79 ns | ~96 ns |
| torus–cone (coaxial) | ~190 ns | ~194 ns | ~212 ns | ~227 ns |
| line–circle | ~175 ns | ~176 ns | ~207 ns | ~223 ns |
| circle–cylinder | ~960 ns | ~946 ns | ~986 ns | ~1.01 µs |
| line–torus | ~1.08 µs | ~1.05 µs | ~1.06 µs | ~1.06 µs |
| ellipse–ellipse (coplanar) | ~1.33 µs | ~1.34 µs | ~1.38–1.46 µs | ~1.38 µs |

The typed call costs the same as the per-pair method, which now just
forwards to it. The untyped calls add a `match` on both shapes and the
conversion to the general result, about 15–30 ns per pair: visible on the
closed-form pairs and lost in the noise once a polynomial is solved.

The general results hold their pieces inline, so they are larger than most
typed results and never allocate (sizes from `std::mem::size_of`):

| type | bytes |
|---|---|
| `SurfaceIntersection` | 624 |
| `CurveCurveIntersection` | 200 |
| `CurveSurfaceIntersection` | 168 |
| `ElementarySurface`, `ElementaryCurve` | 120 |
| `PlaneConeIntersection` (largest typed surface result) | 232 |
| `PlaneSphereIntersection` | 112 |
| `LineCircle3DIntersection` | 72 |
| `TorusConeIntersection`, `LineTorusIntersection`, `ConicSurfaceIntersection` (a `Vec`) | 24 + heap |

Reproduce with:

```sh
cargo bench --bench intersect_dispatch
```
