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

## Points with derivatives

The same bench times the point and its derivatives up to second order on
the curve and the surface above, from separate `eval_point` and
`eval_derivative` calls and from one `eval_d2` call, which evaluates the
basis functions once.

Machine: Linux 7.0, x86-64 desktop, single thread, `--warm-up-time 1
--measurement-time 3` (2026-10-12). Times are per parameter.

| workload | separate calls | `eval_d2` |
|---|---|---|
| `BSplineCurve3D`: point, first and second derivatives | ~172 ns | ~95 ns |
| `BSplineSurface`: point, `Su`, `Sv`, `Suu`, `Suv`, `Svv` | ~856 ns | ~208 ns |

The numeric curve–curve extrema and intersection and the marching
intersection evaluate through `eval_d1` and `eval_d2`, which made them
12–17% faster on B-splines; the numeric curve–surface intersection spends
its time projecting seeds onto the surface and is unchanged.

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

# Conic–quadric intersections

`benches/conic_quadric.rs` times batches of 1024 typed `Intersect::intersect`
calls for each of the 16 pairs of a conic (circle, ellipse, parabola,
hyperbola) and a quadric (plane, sphere, cylinder, cone). The conics are
placed at random near the quadric, so each batch mixes crossings,
tangencies and misses.

Machine: Linux 7.0, x86-64 desktop, single thread, `--warm-up-time 1
--measurement-time 4` (2026-10-12). Times are per pair.

| conic | plane | sphere | cylinder | cone |
|---|---|---|---|---|
| circle | ~96 ns | ~319 ns | ~970 ns | ~1.14 µs |
| ellipse | ~100 ns | ~1.26 µs | ~950 ns | ~1.10 µs |
| parabola | ~68 ns | ~820 ns | ~790 ns | ~1.00 µs |
| hyperbola | ~100 ns | ~920 ns | ~1.00 µs | ~1.07 µs |

The plane pairs, and a circle against a sphere, solve a first-order
equation in closed form. The rest solve a quartic: about 570 ns goes to
factoring it, and the tangency check costs a bounded Newton search on the
derivative per pair of neighbouring roots or complex pair, plus one
surface containment test there. Setting up the coefficients takes under
10 ns.

Reproduce with:

```sh
cargo bench --bench conic_quadric
```

# Discretization

`benches/discretization.rs` discretizes 3000 model-like edges: two thirds
cubic B-splines with 1 to 7 spans, one third quarter-circle arcs (half
rational quadratic B-splines, half trimmed `Circle3D`s). Every edge is a
run-time `TrimmedCurve3D`, and its maximum chord deviation is 1e-2 or 1e-3
times its size, with no angle limit. Three measurements: counting the
points, filling one set of buffers reused across the edges (count, then
fill), and the allocating call.

Machine: Linux 7.0, x86-64 desktop, single thread, `--warm-up-time 1
--measurement-time 3` (2026-10-12). Times are per edge.

| chord deviation | points per edge (mean, p95) | count | fill, reused buffer | allocating |
|---|---|---|---|---|
| 1e-2 × size | 13.3, 24 | ~1.37 µs | ~4.46 µs | ~4.47 µs |
| 1e-3 × size | 36.2, 67 | ~1.52 µs | ~5.65 µs | ~5.68 µs |

This is the baseline for later optimizations, and well above the
prototype's 4–20 ns per point. Timing the pieces of one cubic span
(about 430 ns to count) shows where it goes:

- extracting the span's Bézier control points: ~170 ns, redone by every
  call, twice when filling;
- trying a halving of the span: ~64 ns per split, most of it copying
  control point arrays sized for the highest degree;
- the leaf's turning bound, computed for its reported angle deviation
  even when the angle is not limited: ~65 ns.

Filling costs about three counts: it plans the count again, then visits
the runs to write them. Allocating adds little to filling a reused
buffer.

Reproduce with:

```sh
cargo bench --bench discretization
```

# Knot insertion

`benches/knot_insertion.rs` times `insert_knot` with one knot and
`refine_knots` with 8 and 64 knots spread over clamped curves of 32
poles, and `refine_u_knots` / `refine_v_knots` with 8 knots on a bicubic
6×6 surface. Each call returns a new B-spline, so the times include
allocating and filling it.

Machine: Linux 7.0, x86-64 desktop, single thread, `--warm-up-time 1
--measurement-time 3` (2026-10-12), with other jobs running: times vary
by 10–20% between runs.

| shape | 1 knot | 8 knots | 64 knots |
|---|---|---|---|
| cubic curve, 32 poles | ~345 ns | ~460 ns | ~1.49 µs |
| rational cubic curve, 32 poles | ~375 ns | ~520 ns | ~1.59 µs |
| quadratic curve, 32 poles | ~325 ns | ~455 ns | ~1.36 µs |
| bicubic 6×6 surface, in `u` | | ~1.02 µs | |
| bicubic 6×6 surface, in `v` | | ~0.98 µs | |

A refinement computes the new poles in one pass over the list, each knot
blending `degree` poles while the rest are copied once, so 64 knots cost
about four times one knot rather than 64 times: most of the time for a
few knots goes to allocating the new curve. A surface refines all rows or
columns of its net in the same pass, sharing the blending coefficients.

Reproduce with:

```sh
cargo bench --bench knot_insertion
```

# Splitting and Bézier decomposition

The `split_extract` group of `benches/knot_insertion.rs` times
`to_bezier_pieces` on clamped cubic curves of 32 poles (29 pieces),
`extract` of the middle 30% of their range, and `to_bezier_patches` on a
bicubic 6×6 surface (9 patches).

Machine: Linux 7.0, x86-64 desktop, single thread, `--warm-up-time 1
--measurement-time 3` (2026-10-12), with other jobs running (load
average around 13): times vary by up to 20% between runs.

| call | polynomial | rational |
|---|---|---|
| `to_bezier_pieces`, cubic curve, 32 poles | ~5.7 µs | ~6.5 µs |
| `extract` of the middle, same curves | ~0.60 µs | ~0.43 µs |
| `to_bezier_patches`, bicubic 6×6 surface | ~8.3 µs | |

Every interior knot is raised to multiplicity `degree` in one refinement
pass (about 1 µs for the curve), then each piece is sliced out of the
refined list. Most of the time goes to building the pieces: each curve
owns five or six buffers, and each surface patch about twelve. An
extraction refines only the poles around its two ends.

Reproduce with:

```sh
cargo bench --bench knot_insertion -- split_extract
```
