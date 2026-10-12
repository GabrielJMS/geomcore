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
12–17% faster on B-splines; the numeric curve–surface intersection spent
its time projecting seeds onto the surface and was unchanged (see
[below](#curvesurface-intersection-on-b-splines) for its B-spline path).

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

# Curve–surface intersection on B-splines

`benches/intersect_curve_surface.rs` times `intersect_curve_surface` for 64
cubic B-spline curves against one bicubic B-spline surface (6×6 poles
`(i, j, sin(0.7 i j))`, knots `[0, 1, 2, 3]` with multiplicities
`[4, 1, 1, 4]`). Each curve rises from `z = -2` to `z = 2` along a wavy
path over the surface, so most cross it once.

Machine: Linux 7.0, x86-64 desktop, single thread, `--warm-up-time 1
--measurement-time 3` (2026-10-12). Times are per curve.

| search | time / curve |
|---|---|
| sampled (before) | ~2.78 ms |
| Bézier piece pairs (after) | ~35 µs |

The sampled search projected 128 curve samples onto the surface and
refined the 24 closest, whatever the shapes. The piece-pair search, about
80 times faster, only looks where the control-point boxes of a curve span
and a surface patch overlap, and drops a pair as soon as their hulls part;
each hit then costs a few halvings and one short Newton run. Tangencies
and overlaps still exhaust its budget and pay for both searches.

Reproduce with:

```sh
cargo bench --bench intersect_curve_surface
```

# Curve–curve intersection and extrema on B-splines

`benches/intersect_curve_curve.rs` times `intersect_curve_curve` on 64
pairs of cubic B-spline curves in one plane, one crossing the square
`[0, 5]^2` from left to right and the other from bottom to top along wavy
paths, so that every pair crosses at least once; and `curve_curve_extrema`
on 64 pairs of skew cubic B-spline curves, wavy in all three directions,
one above the other.

Machine: Linux 7.0, x86-64 desktop, single thread, `--warm-up-time 1
--measurement-time 3` (2026-10-12), with the machine otherwise busy: both
searches ran back to back under the same load, so their ratio holds
better than their absolute times. Times are per pair.

| query | grid (before) | Bézier piece pairs (after) |
|---|---|---|
| `intersect_curve_curve`, crossing pairs | ~157 µs | ~9.1 µs |
| `curve_curve_extrema`, skew pairs | ~167 µs | ~55 µs |

The grid search refined the local minima of a 32×32 sampling of the
distance, whatever the shapes. For an intersection, the piece-pair search,
about 17 times faster, only looks where the control-point boxes of two
spans overlap and drops a pair as soon as their hulls part; each hit then
costs a few halvings and one short Gauss–Newton run. Extrema must still
visit every pair of spans, since every local minimum is reported, so the
gain is about 3 times: a pair is dropped once the distance provably falls
along one curve throughout it or bends like a saddle, and solved once it
is provably convex. Tangencies, overlaps and other valleys of equal
distance exhaust the search's budget and pay for both searches.

Reproduce with:

```sh
cargo bench --bench intersect_curve_curve
```

# Mixed elementary and B-spline intersections

The same two benches time an elementary shape against a B-spline,
converted to its exact rational B-spline and searched by Bézier piece
pairs instead of sampled. `intersect_curve_surface`: the 64 cubic curves
above against a cylinder of radius 2 about the vertical axis through the
footprint's centre, and 64 circles of radius 0.6 to 1.2 standing across
the bicubic surface in vertical planes of random directions.
`intersect_curve_curve`: 64 circles of radius 1 to 2 in the plane of the
crossing pairs against the first curve of each pair. `curve_curve_extrema`:
64 ellipses of radii 2 and 1, tilted at random 1 to 2 above the plane,
their halves `[0, π]`, and the lines along their major axes, whole and
over `[-2, 2]`, against the lower curve of each skew pair.

Machine: Linux 7.0, x86-64 desktop, single thread, `--warm-up-time 1
--measurement-time 3` (2026-10-12). Times are per curve or pair.

| query | sampled (before) | after |
|---|---|---|
| `intersect_curve_surface`, cubic × cylinder | ~84 µs | ~62 µs |
| `intersect_curve_surface`, circle × bicubic | ~4.9 ms | ~84 µs |
| `intersect_curve_curve`, circle × cubic | ~66 µs | ~15 µs |
| `curve_curve_extrema`, segment × cubic | ~58 µs | ~29 µs |
| `curve_curve_extrema`, ellipse × cubic | ~66 µs | unchanged, sampled |
| `curve_curve_extrema`, half ellipse × cubic | ~116 µs | unchanged, sampled |
| `curve_curve_extrema`, line × cubic | ~33 µs | unchanged, sampled |

Every search run here ends complete. A circle against a surface gains
most, about 58 times: the sampled search projected 128 points of the
circle onto the surface, each projection a search of its own. Against a
cylinder the gain is smaller, as the cylinder's quarter-turn patches are
large and rational, and their boxes overlap much of each curve. The
extrema search beats the grid on a segment only: on a whole ellipse it
took ~270 µs (a maximum of the distance along the closed curve for every
point of the cubic, and loose bounds on rational pieces), on a half
ellipse ~145 µs, and on a full line ~42 µs against the exact reduction
to the cubic alone, so those pairs stay sampled.

Reproduce with:

```sh
cargo bench --bench intersect_curve_surface -- "cylinder|circle_bspline_surface" --warm-up-time 1 --measurement-time 3
cargo bench --bench intersect_curve_curve -- "_bspline$" --warm-up-time 1 --measurement-time 3
```
