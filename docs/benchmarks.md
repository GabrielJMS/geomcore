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
one pair kind three ways: the typed call (`IntersectExact::intersect_exact`
since 0.36), the untyped call on `ElementaryCurve`/`ElementarySurface`,
and the untyped call on `Curve3D`/`Surface`. The inputs vary across the
batch, so each one mixes crossings, tangencies and misses.

The table below is history (0.29 to 0.35): it was measured when the typed
call was `Intersect::intersect` and the per-pair methods it replaced still
existed, deprecated. Both are gone in 0.36; the per-pair column is kept
for the record.

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

The typed call cost the same as the per-pair method, which then just
forwarded to it. The untyped calls add a `match` on both shapes and the
conversion to the general result, about 15–30 ns per pair: visible on the
closed-form pairs and lost in the noise once a polynomial is solved.

The general results hold their pieces inline, so they are larger than most
typed results and never allocate (sizes from `std::mem::size_of`):

| type | bytes |
|---|---|
| `ExactSurfaceIntersection` | 624 |
| `ExactCurveCurveIntersection` | 264 (200 before 0.36, whose hits had no optional parameters) |
| `ExactCurveSurfaceIntersection` | 168 |
| `ElementarySurface`, `ElementaryCurve` | 120 |
| `PlaneConeIntersection` (largest typed surface result) | 232 |
| `PlaneSphereIntersection` | 112 |
| `LineCircle3DIntersection` | 72 |
| `TorusConeIntersection`, `LineTorusIntersection`, `ConicSurfaceIntersection` (a `Vec`) | 24 + heap |

Reproduce with:

```sh
cargo bench --bench intersect_dispatch
```

## Routed against exact (0.36)

The same bench times `Intersect::intersect` (routed) against
`IntersectExact::intersect_exact` (typed exact) on batches of 1024 pairs
of one kind, here with realistic crossings only: lines crossing a
cylinder of radius 1.5 about the vertical axis at two points, coplanar
circles of radius 2 and 1.5 with centres 1 to 2 apart, crossing at two
points, and the same circles in the plane (2D).

The tables below up to "Parameters on demand" are history: then, the
routed call reported the parameters on both operands, so it was read
against two baselines:

- **exact**: the typed exact call alone. It returns points (and, for a
  line, its own parameter), not the parameters on both operands.
- **exact + parameters**: the typed exact call, then the parameters the
  routed result reports, computed by the caller (`parameters_of` on the
  cylinder, `parameter_of` on each circle). This is the fair baseline:
  the same answer, with no routing.

Machine: Linux 7.0, x86-64 desktop, single thread (2026-10-15), loaded
(load average 10–15 on 16 cores), so criterion medians moved by about
±10% between runs. Times are criterion medians per pair; the fastest of
40 timings of the batch, taken with the same build, is in brackets.

| pair | build | exact | exact + parameters | routed |
|---|---|---|---|---|
| line–cylinder | before | 82 ns | 142 ns | 201 ns |
| line–cylinder | after | 90 ns (77) | 139 ns (133) | 193 ns (188) |
| circle–circle (coplanar) | before | 36 ns | 164 ns | 185 ns |
| circle–circle (coplanar) | after | 34 ns (33) | 111 ns (107) | 151 ns (147) |
| 2D circle–circle | before | 8.6 ns | 82 ns | 133 ns |
| 2D circle–circle | after | 8.9 ns (8.1) | 81 ns (75) | 114 ns (108) |

"Before" is the router as it was: every pair went through the untyped
`ElementaryCurve` / `ElementarySurface` enums. "After" has two changes:

- **Typed routing.** A pair of concrete types (`Line3D` with `Cylinder`,
  `Circle3D` with `Circle3D`, `Circle2D` with `Circle2D`, and every other
  concrete pair `intersect_exact` solves) now calls its typed solver
  directly; only an enum operand goes through the enum dispatch. A
  trimmed curve routes through the type of its basis curve, so a segment
  or an arc gets the same path. The result is bit for bit the one the
  enums give (a test checks it). Measured alone: routed circle–circle
  185 → 172 ns, 2D 133 → 117 ns, line–cylinder within the noise
  (201 → 196 ns).
- **No normalization in the angle of a point.** The parameter of a point
  on a circle or an ellipse is an `atan2` around the frame's z
  direction; the axis was normalized on every call, though a frame keeps
  unit axes by construction. Dropping that square root and division
  saves about 13 ns per circle parameter, on both the routed call and
  the caller's baseline (circle–circle exact + parameters 164 → 111 ns).
  It changes the last bits of those parameters; every test passes
  unchanged.

The routed call is now 40–55 ns over the fair baseline, against 20–60 ns
before (the baseline itself got faster). Counted under callgrind, the
routed call runs about 370 more instructions per pair than the fair
baseline in 3D and 330 in 2D:

- **Moving the result.** A routed result was 304 bytes for
  curve–surface, 328 for curve–curve and 208 in 2D, since it kept up to
  4 hits and 2 overlaps inline to avoid allocating; the baseline returns
  about 100. Copying it was about 125 instructions per circle–circle pair
  and 60 per 2D pair.
- **Building the result.** Converting the solver's hits into the routed
  hits, clipping them to the parameter ranges, and sorting them: about
  150 instructions per circle–circle pair.
- **The seam in 2D.** A 2D hit just below the seam of a full circle is
  reported at its start, like in 3D (`Circle2D::parameter_in_bounds`):
  about 65 instructions per pair.

The inline hit lists themselves were measured when they replaced heap
lists (one allocation per exact routed call, two for a 2D pair): the
routed time moved from about 192 to 189 ns for line–cylinder, 179 to
167 ns for circle–circle, and not at all in 2D, since the larger result
costs part of the allocation back in copies.

Overlaps come only from coincident curves, and an empty `Vec` does not
allocate, so the overlaps went back to a plain `Vec` while the hits stay
inline. The results shrank to 296, 264 and 144 bytes. Timed the same way
(the median of five interleaved runs, each the fastest of 30 timings) the
routed call barely moved: line–cylinder 184 → 183 ns, circle–circle
147 → 143 ns, 2D circle–circle 108 → 109 ns; callgrind counts 1547 →
1549, 1599 → 1582 and 1181 → 1184 instructions per pair. The gain is
in memory: a caller keeping many results stores up to a third less.

The four `parameter_of` calls of a circle–circle pair were each needed
(two hits on two circles), and each was computed once.

### Parameters on demand

A hit parameter is now an `Option`: the routed result stores a
parameter only when the solver computed it on its way to the point (the
line of a line–cylinder pair, the substituted conic of a conic pair),
and leaves the rest `None`. A caller who needs one calls the operand's
`parameter_of` / `parameters_of` on the hit point. The routed call no
longer recovers parameters nobody asked for, so it is read against the
typed exact call alone: the difference is routing plus building the
result. A trimmed operand still computes its parameter, to clip the hit
to its range, and stores it.

Same machine, loaded (load average 8–10). Times are the fastest of
30 timings of 100 batches, in three interleaved runs per build; the
criterion median after the change is in brackets.

| pair | exact | routed before | routed after |
|---|---|---|---|
| line–cylinder | 73 ns (77) | 184 ns | 127 ns (139) |
| circle–circle (coplanar) | 31 ns (32) | 145 ns | 76 ns (79) |
| 2D circle–circle | 8 ns (7.5) | 111 ns | 35 ns (35) |

Callgrind instructions per pair:

| pair | exact | routed before | routed after |
|---|---|---|---|
| line–cylinder | 714 | 1545 | 1161 |
| circle–circle (coplanar) | 352 | 1567 | 759 |
| 2D circle–circle | 79 | 1188 | 321 |

The `Option` fields make each hit 16 bytes larger (a curve–curve hit 48
→ 64 bytes, a 2D hit 40 → 56, a curve–surface hit 56 → 72), and the
results with inline hits grow with them: 296 → 360 bytes for
curve–surface, 264 → 328 for curve–curve and 144 → 176 in 2D. What is
left over the exact call is mostly moving that result (memcpy is 13–15%
of the routed instructions) and converting the solver's hits into it.

The 2D router then skipped the range test for two unrestricted curves,
as the 3D one already did: 2D circle–circle routed 321 → 247
instructions per pair, 35 → 25 ns. The other two pairs did not move
(1161 and 759 instructions).

### Surface–surface

A plane at heights up to 0.9 across a unit sphere, one circle each. The
typed exact call returns the plane–sphere result alone; the routed call
builds the general result, with its curve in a `Curve3D` in a `Vec` (one
allocation per pair). Criterion medians per pair of three runs, loaded
machine (load average 5–8), batches of 1024:

| pair | exact | routed |
|---|---|---|
| plane–sphere | 77–82 ns | 113–120 ns |

Reproduce with:

```sh
cargo bench --bench intersect_dispatch -- intersect_routed
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

`benches/intersect_curve_surface.rs` times the numeric curve–surface
search for 64 cubic B-spline curves against one bicubic B-spline surface (6×6 poles
`(i, j, sin(0.7 i j))`, knots `[0, 1, 2, 3]` with multiplicities
`[4, 1, 1, 4]`). Each curve rises from `z = -2` to `z = 2` along a wavy
path over the surface, so most cross it once. Since 0.36 the bench calls
the routed `intersect`, which sends these pairs to the numeric search; the
tables below name that search after the free function
`intersect_curve_surface` it was called through before, which still runs
it on any curve and surface.

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

`benches/intersect_curve_curve.rs` times the numeric curve–curve search
(through the routed `intersect` since 0.36, through the free function
`intersect_curve_curve`, which still runs it on any two curves, before) on 64 pairs of cubic B-spline curves in one plane, one crossing the square
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

# Marching intersection

`benches/marching_intersection.rs` times the routed `intersect` on two
bicubic B-spline surfaces (6×6 poles over `[0, 5]^2`, knots `[0, 1, 2,
3]` with multiplicities `[4, 1, 1, 4]`), with heights `sin(0.7 i j)` and
`0.5 cos(1.3 i) cos(1.1 j)` at their poles `(i, j)`. They cross in six
curves: five open across the footprint and one closed loop. The numbers
recorded below were measured before routing, on a direct marching call;
the routed call adds only a cheap closed-form check.

Machine: Linux 7.0, x86-64 desktop, single thread, `--warm-up-time 1
--measurement-time 3` (2026-10-12). Times are per call.

| seeds | time | curves traced |
|---|---|---|
| grid (before) | ~24.2 ms | 4 of 6 |
| grid, traces ending on the bounds | ~24.5 ms | 6 |
| Bézier patch pairs (after) | ~3.4 ms | 6 |

The grid seeds project 625 samples of each surface onto the other and
converge every distance minimum; before traces ended on a parameter
bound, two curves running into the bounds were dropped. The patch-pair
search halves the pairs of patches whose boxes overlap until each is
dropped or small enough to seed from, about 4000 visits of a few hundred
nanoseconds; most leaves lie on a curve traced already and are skipped
by a box test against the traced points, so about half the time goes to
the search and most of the rest to tracing the six curves.

The same bench times elementary pairs:
- `torus_plane_oblique`: a torus of radii 3 and 1 about z against the
  plane through `(0.5, 0, 0.2)` tilted 0.6 radians about y (two loops);
- `sphere_torus`: a sphere of radius 1.2 centred on that torus's centre
  circle (two loops);
- `cylinder_extruded_wall`: a cylinder of radius 1 about z against a cubic
  B-spline wall extruded along z (two open curves);
- `cone_cylinder_unbounded`: a cone of semi-angle 0.3 against a cylinder
  whose axis is parallel to a generator (two curves to infinity).

Fastest of three runs, same settings (2026-10-15, machine loaded). Three
versions:
- the grid over `[-10, 10]`, with traces of at most 2000 steps of at most
  0.1 each way;
- the iso-curve seeding of the first review round, whose steps grew with
  the distance from the seed;
- the current one. Steps follow the curvature with no fixed cap, are
  halved when the midpoint strays from the chord or the step crosses more
  than one knot of a B-spline side, and stop on the model-size cube.
  Bounded pairs add both grids.

| case | grid | first round | current |
|---|---|---|---|
| `bspline_surface_pair` | ~2.61 ms | ~2.60 ms | ~3.11 ms |
| `torus_plane_oblique` | ~478 µs | ~441 µs | ~580 µs |
| `sphere_torus` | ~682 µs | ~524 µs | ~1.02 ms |
| `cylinder_extruded_wall` | ~1.86 ms | ~460 µs | ~415 µs |
| `cone_cylinder_unbounded` | ~5.73 ms (4001 poles per curve) | ~771 µs (407) | ~665 µs (215) |

The midpoint check costs one more Newton solve per step. It makes the
B-spline pair and the torus–plane pair about 20 % slower than the grid.
The sphere–torus pair also traces from both grids, because the torus
meridians alone can pass between small loops. In exchange, a step can no
longer cross a feature whose ends share a tangent, such as a small bump
on a straight wall, which the first round did.

The cone–cylinder curves used to stop after a fixed number of steps.
They now run to the cube 50 000 away, the step doubling along the nearly
straight branches. The torus seeds from 24 meridian circles cut by the
sphere or the plane (exact), and the plane from 48 lines across the
torus's box (each a quartic). The sphere's circles would meet the torus
only through the sampled search, so they are left out. The wall's grid
finds the curves that the cylinder's rulings miss, since the rulings run
along the extrusion.

Measured outside the bench (best of five calls, release), pairs without
a closed form: the cone pairs through the routed `intersect` on
`Surface`, the tori (radii 3 and 1, and 3 and 0.8 tilted and shifted)
through the same routed `intersect` (direct marching call at the time):

| pair | grid | first round | current |
|---|---|---|---|
| cone × cylinder along a generator | ~4.1 ms | ~0.54 ms | ~0.50 ms |
| cone × cone, parallel axes | ~4.4 ms | ~0.44 ms | ~0.40 ms |
| torus × torus (grid seeds in all three) | ~1.04 ms | ~1.06 ms | ~1.39 ms |

The same bench times pairs where the surfaces lie on each other or touch,
and asserts each answer outside the timed loop:
- `sphere_bspline_copy`: a sphere of radius 0.8 against its own rational
  B-spline form (coincident);
- `cylinder_own_patch`: a cylinder of radius 1 against its patch over a
  whole turn and `z` in `[0, 2]` (incomplete, one region covering the
  patch);
- `plane_own_patch`: a plane against its own bilinear patch (likewise);
- `bspline_sphere_itself`: that B-spline sphere against itself
  (coincident);
- `plane_tangent_to_cylinder_patch`: the plane `x = 1` against that
  cylinder patch, which it touches along a ruling (one curve, tangent).

Before the coincidence check, seeding cut every iso-curve of the
elementary side with the other surface. On surfaces lying on each other
each iso-curve lies on the other surface, so its search never separates
and falls back to sampling: about 0.5–2 ms per iso-curve, for 24 to 48 of
them. The plane touching the patch paid the same for its 22 lines
touching the patch. Now a 7 × 7 grid of one surface's points is
projected onto the other first, and the check stops at the first point
off it, which costs crossing pairs one or two projections. Where the
grids of seeds find only touches, the iso-curves are left out.

Fastest of three runs, same settings (2026-10-15, machine loaded). The
0.35 marching call, the routed call before the check, and the current
one:

| case | 0.35 | before | current |
|---|---|---|---|
| `sphere_bspline_copy` | ~6.8 ms (2 spurious curves) | ~787 ms (none, not coincident) | ~306 µs (coincident) |
| `cylinder_own_patch` | ~2.0 ms (none) | ~469 ms (none, not coincident) | ~10 µs (one region) |
| `plane_own_patch` | ~0.32 ms (none) | ~42 ms (none, not coincident) | ~5.7 µs (one region) |
| `bspline_sphere_itself` | ~270 ms (35 spurious curves) | ~1.04 s (25 spurious curves) | ~568 µs (coincident) |
| `plane_tangent_to_cylinder_patch` | ~1.9 ms (none) | ~24 ms (one curve, not tangent) | ~90 µs (one tangent curve) |

The check makes the crossing pairs above at most a few percent slower,
within the noise of the loaded machine.

Reproduce with:

```sh
cargo bench --bench marching_intersection
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
