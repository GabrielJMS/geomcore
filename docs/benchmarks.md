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

# Knot removal

`benches/knot_removal.rs` times two workloads on clamped cubic curves,
polynomial and rational, at a tolerance of 1e-6 relative to the curve's
size: removing 8 knots just inserted into a curve of 32 poles (40 with
them), all removed, and trying every one of the 28 interior knots of a
curve of 32 poles, all needed and kept. Each is timed as one
`remove_knot` call per knot on the result of the previous call, and as
one `remove_knots` call for the whole list. On a bicubic 6×6 surface,
8 knots just inserted are removed with `remove_u_knots` or
`remove_v_knots`.

Machine: Linux 7.0, x86-64 desktop, single thread, `--warm-up-time 1
--measurement-time 3` (2026-10-12), with heavy other jobs running (load
about 24 on 16 threads): on an idle machine the polynomial rows ran about
1.5× faster.

| shape | 8 inserted, one by one | 8 inserted, batch | 28 needed, one by one | 28 needed, batch |
|---|---|---|---|---|
| cubic curve, 32 poles | ~4.3 µs | ~1.58 µs | ~9.1 µs | ~2.33 µs |
| rational cubic curve, 32 poles | ~6.0 µs | ~2.19 µs | ~9.7 µs | ~2.40 µs |
| bicubic 6×6 surface, in `u` | | ~2.47 µs | | |
| bicubic 6×6 surface, in `v` | | ~3.09 µs | | |

A batch works in one copy of the poles, made on the first removal, so
an attempt that keeps its knot costs a few dozen nanoseconds, while one
call per knot pays for a new curve each time. A rational attempt first
rejects on the deviation alone, so a knot it keeps costs about as much as
a polynomial one.

Reproduce with:

```sh
cargo bench --bench knot_removal
```
