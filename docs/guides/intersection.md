# Intersection

One entry point, two routes. `a.intersect(&b, tol)` picks the exact
closed form whenever the pair has one and reports `NotAnalytic` when it
doesn't — never a silent miss. The numeric route (curve–curve,
curve–surface, marching surface–surface) covers the rest and reports
whether its search was complete. Adapted from
[Intersection](https://avenirgeometry.com/articles/intersection/).

## What an intersection returns

Every result is one of four things: hits with parameters and a
multiplicity (1 for a crossing, 2+ for a tangency), `Coincident` (the
shapes overlap entirely), `Empty`, or `NotAnalytic` (no closed form —
the caller decides whether to run the numeric path). Hits on circles
and ellipses keep their parameters inside `bounds()` and sorted, and a
tangency is decided by distance, not by root separation:

<figure class="geom-fig">
<img src="../../assets/figures/intersection/01-a-tangency-is-a-distance.svg" alt="Tangency decided by distance">
<figcaption>A tangency is a distance: a closest approach within confusion is one tangent hit, not two hits or none.</figcaption>
</figure>

## Exact: one call, either order

Between two concrete types the call returns the pair's own result
type; between run-time values (`Curve3D`, `Surface`,
`ElementaryCurve`, `ElementarySurface`) it returns the general result
(`CurveCurveIntersection`, `CurveSurfaceIntersection`,
`SurfaceIntersection`). Dispatch is static — the call costs what the
per-pair method costs.

=== "Python"

    ```python
    kind, payload = plane.intersect(sphere)   # ("circle", Circle3D)
    kind, payload = line.intersect(sphere)    # ("hits", [(t, Point3D, multiplicity)])
    ```

=== "Rust"

    ```rust
    use geomcore::{Intersect, Tolerance};

    let tol = Tolerance::DEFAULT;
    let result = plane.intersect(&sphere, tol);
    assert_eq!(sphere.intersect(&plane, tol), result);
    ```

Covered pairs include plane/quadric/torus surfaces against each other
in their symmetric configurations, lines/circles/conics against each
other and against the elementary surfaces, and any two conics —
coplanar (up to four hits) or in crossing planes (met on the shared
line). A sphere cut by a coaxial cylinder, computed live below —
two exact rings in ~0.4 µs on a Linux x86-64 desktop, single thread:

<iframe src="https://geomcore.github.io/geomcore-viewer/?scene=https://geomcore.github.io/geomcore/assets/scenes/intersect-exact.json" width="100%" height="480" style="border: 1px solid #333; border-radius: 8px;" allow="fullscreen" title="Sphere and cylinder with their two exact intersection rings"></iframe>

Two more landmarks, as figures:

<figure class="geom-fig">
<img src="../../assets/figures/intersection/15-plane-and-plane.svg" alt="Plane and plane intersection">
<figcaption>Plane against plane: a line, or Parallel / Coincident classification by tolerance.</figcaption>
</figure>

<figure class="geom-fig">
<img src="../../assets/figures/intersection/16-plane-and-sphere.svg" alt="Plane and sphere intersection">
<figcaption>Plane against sphere: a circle, a tangent point, or empty — classified, not sampled.</figcaption>
</figure>

<iframe src="https://geomcore.github.io/geomcore-viewer/?scene=https://geomcore.github.io/geomcore/assets/scenes/plane-sphere.json" width="100%" height="480" style="border: 1px solid #333; border-radius: 8px;" allow="fullscreen" title="A plane against a sphere at three distances: a circle, a tangent point, nothing"></iframe>

## Numeric: curves

`intersect_curve_curve` and `intersect_curve_surface` work over the
`ParametricCurve3D` × `ParametricSurface` traits, so custom types and
trimmed pieces work too. On B-splines there is no sampling grid:
Bézier pieces subdivide where their control boxes overlap until each
pair holds at most one hit, then Newton solves it — crossings a grid
would miss between samples are found, and tangencies/overlaps fall
back to the sampled search instead of burning the budget.

The search reports whether it finished: the return values read as
their hit slices (`.hits()`) and carry a `Completeness` — `Complete`
when every piece pair was proven apart or isolated, `Incomplete` with
the parameter regions where hits may be missing otherwise.

=== "Python"

    ```python
    from geomcore import intersect_curve_surface, intersect_curve_curve

    hits = intersect_curve_surface(spline_curve, spline_surface)  # [(t, (u, v), Point3D)]
    hits = intersect_curve_curve(curve_a, curve_b)                # [(s, Point3D, t)]
    ```

=== "Rust"

    ```rust
    use geomcore::{intersect_curve_curve, intersect_curve_surface};

    let hits = intersect_curve_surface(&spline, &surface, tol);
    for h in hits.hits() {
        let _ = (h.curve_param, h.point);
    }
    let meetings = intersect_curve_curve(&a, &b, tol);
    ```

## Numeric: marching surfaces

`marching_intersection` (or `Surface::intersect_marching`) seeds from
the two surfaces, traces with curvature-following steps corrected by
Newton on both surfaces, and fits each loop or arc as a
`BSplineCurve3D`. Two B-spline surfaces additionally seed from
overlapping patch pairs, so isolated loops smaller than the seed grid
are found there.

<figure class="geom-fig">
<img src="../../assets/figures/intersection/31-marching-round-a-cylinder-and-a-sphere.svg" alt="Marching round a cylinder and a sphere">
<figcaption>Marching: seed, trace along the tangent, correct back onto both surfaces, fit the trail.</figcaption>
</figure>

<figure class="geom-fig">
<img src="../../assets/figures/intersection/33-the-loop-no-edge-sees.svg" alt="An isolated loop">
<figcaption>The hard case: an isolated loop no seed grid line crosses. Patch-pair seeding finds these on B-spline pairs.</figcaption>
</figure>

=== "Python"

    ```python
    from geomcore import intersect_marching

    loops = intersect_marching(surface_a, surface_b)   # [BSplineCurve3D, ...]
    ```

=== "Rust"

    ```rust
    use geomcore::marching_intersection;

    let loops: Vec<BSplineCurve3D> = marching_intersection(&a, &b, tol);
    ```

Marching is approximate by design — fitted curves, no tangency
classification, no singular points. Where the surfaces touch instead
of crossing, traces stop rather than turn onto the other branch. A
wavy B-spline patch through a sphere, marched live below — one fitted
loop in ~8 ms on the same machine (the loop arrives in the viewer as
a sampled polyline until native B-spline types land,
[viewer#5](https://github.com/geomcore/geomcore-viewer/issues/5);
tunable sampling density is
[viewer#12](https://github.com/geomcore/geomcore-viewer/issues/12)):

<iframe src="https://geomcore.github.io/geomcore-viewer/?scene=https://geomcore.github.io/geomcore/assets/scenes/intersect-marching.json" width="100%" height="480" style="border: 1px solid #333; border-radius: 8px;" allow="fullscreen" title="B-spline patch and sphere with their marching intersection loop"></iframe>

All the scenes on this page were computed by the kernel itself —
see `docs/assets/scenes/gen_scenes.py` — not drawn by hand.
