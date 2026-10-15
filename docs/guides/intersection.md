# Intersection

One entry point, two methods. `a.intersect(&b, tol)` works for any two
curves or surfaces, B-splines and trimmed curves included, in either
order. It solves the pair exactly whenever a closed form exists, and
runs a numeric search (curve–curve, curve–surface, marching
surface–surface) otherwise. The result says which method ran and
whether every hit was found, so a switch from nanoseconds to
milliseconds is never hidden. Adapted from
[Intersection](https://avenirgeometry.com/articles/intersection/).

## What an intersection returns

One result per family: `CurveCurveIntersection` (and
`CurveCurveIntersection2D`), `CurveSurfaceIntersection` and
`SurfaceSurfaceIntersection`. Each one carries:

- the **hits**: the point, the multiplicity (1 for a crossing, 2+ for a
  tangency) and the parameters on both operands. A parameter is stored
  when the solver computed it on its way to the point, and is `None`
  otherwise; `operand.parameter_of(hit.point)` (or `parameters_of` on a
  surface) recovers it in one call;
- the **overlaps**, where a curve lies on the other shape: a parameter
  range on each curve plus `same_direction`, or the curve's ranges on a
  surface;
- for two surfaces, the **curves** (each with a `tangent` flag), the
  **contact points** and whether the surfaces are **coincident**;
- the **method** (`Exact` or `Numeric`) and the **completeness**: an
  exact result is always complete; a numeric one is complete, incomplete
  (with the regions where hits may be missing) or sampled.

Hits on circles and ellipses keep their parameters inside `bounds()`,
and a tangency is decided by distance, not by root separation:

<figure class="geom-fig">
<img src="../../assets/figures/intersection/01-a-tangency-is-a-distance.svg" alt="Tangency decided by distance">
<figcaption>A tangency is a distance: a closest approach within confusion is one tangent hit, not two hits or none.</figcaption>
</figure>

## Exact: one call, either order

=== "Python"

    ```python
    result = plane.intersect(sphere)    # SurfaceSurfaceIntersection
    circle = result.curves[0].curve     # Circle3D
    hits = line.intersect(sphere).hits  # [CurveSurfaceHit], in either order
    ```

=== "Rust"

    ```rust
    use geomcore::{Intersect, IntersectionMethod, Tolerance};

    let tol = Tolerance::DEFAULT;
    let result = plane.intersect(&sphere, tol);
    assert_eq!(result.method(), IntersectionMethod::Exact);
    assert_eq!(sphere.intersect(&plane, tol), result);
    ```

A pair of concrete types calls its exact solver directly, with no
run-time dispatch; only `Curve3D`, `Surface`, `ElementaryCurve` or
`ElementarySurface` operands pay for a `match`. For hot loops over
elementary shapes, `a.intersect_exact(&b, tol)` (the `IntersectExact`
trait) returns the pair's own enum, such as `PlaneSphereIntersection`,
on the stack, recovers no parameter it does not need, and reports
`NotAnalytic` where no closed form exists instead of searching.

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

The routed `a.intersect(&b, tol)` takes the numeric search for pairs
without a closed form (a B-spline, or an ellipse against a torus), and
trimmed pieces are searched over their own range. A curve or surface
type you define yourself, by implementing `ParametricCurve3D` or
`ParametricSurface`, goes through `intersect_curve_curve(a, b, tol)` and
`intersect_curve_surface(curve, surface, tol)`: they always run the
numeric search and return the same result types as the routed call.
On B-splines there is no sampling grid: Bézier pieces subdivide where their control boxes overlap until each
pair holds at most one hit, then Newton solves it — crossings a grid
would miss between samples are found, and tangencies/overlaps fall
back to the sampled search instead of burning the budget.

Where a curve lies on the other operand (a planar B-spline in a plane, a
straight B-spline on a line, two B-splines along one line, a circle and
its own `to_bspline()`), the pieces never separate. The hits and regions
left along the curve are then joined into an **overlap** wherever the
curve stays on the other operand between them (samples within
`tol.confusion`, tangents along it), extended by halving to where the
curve leaves it, within `tol.confusion` in space of the true end. The
hits inside drop out, and the result stays `Incomplete`, with an
`Unseparated` region over the overlap: samples found it, not a proof. A
stretch where the operands only touch (they part as the square of the
distance from the contact) stays a hit; an overlap between curves whose
range on the second curve would cross that curve's seam is not joined,
and keeps its hits and region.

The search reports whether it finished: the result's `method()` is
`Numeric`, and its `completeness()` is `Complete` when every piece pair
was proven apart or isolated, `Incomplete` with the parameter regions
where hits may be missing otherwise, and `Sampled` when the hits come
from samples (a pair without a B-spline, such as an ellipse against a
torus).

=== "Python"

    ```python
    result = spline_curve.intersect(spline_surface)  # CurveSurfaceIntersection
    for hit in result.hits:
        hit.curve_parameter, hit.surface_parameters, hit.point
    complete = result.is_complete()
    meetings = curve_a.intersect(curve_b)            # CurveCurveIntersection
    ```

=== "Rust"

    ```rust
    use geomcore::Intersect;

    let result = spline.intersect(&surface, tol);
    for h in result.hits() {
        let _ = (h.curve_parameter, h.surface_parameters, h.point);
    }
    let complete = result.is_complete();
    let meetings = a.intersect(&b, tol);
    ```

## Numeric: marching surfaces

For a pair without a closed form, the routed `intersect` of two
surfaces seeds from the points where iso-curves of an elementary side
(rulings and circles of a cylinder or cone, circles of a sphere or
torus, lines of a plane) meet the other surface exactly, so an
unbounded surface is searched whole. A B-spline side adds its own
seeds, and two bounded surfaces add a parameter grid. It then traces
with curvature-following steps corrected by Newton on both surfaces,
until the trace closes or leaves the model-size cube, and fits each
loop or arc as a `BSplineCurve3D`. Two B-spline surfaces additionally
seed from overlapping patch pairs, so isolated loops smaller than the
seed spacing are found there. A marched result is `Sampled`: a loop
smaller than the iso-curve spacing can be missed.

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
    # Exact curves (Line3D, Circle3D, ...) where a closed form exists,
    # BSplineCurve3D where marched.
    result = surface_a.intersect(surface_b)
    curves = [piece.curve for piece in result.curves]
    ```

=== "Rust"

    ```rust
    use geomcore::Intersect;

    let result = a.intersect(&b, tol);
    let curves = result.curves();   // BSpline curves where marched
    ```

Before seeding, a surface with bounded parameters is checked for lying
on the other: a 7 × 7 grid of its points, and the centres of the
grid's cells, must lie within confusion of the other surface, with the
normals parallel within the tilt that distance allows between samples
(about twice the confusion over their spacing), so a fitted or slightly
moved copy still counts. Surfaces lying on each
other, such as a sphere and its own `to_bspline()`, come back coincident
(`Sampled`, since samples found it). A patch lying on a larger or
unbounded surface, such as a cylinder and its `to_bspline_patch`, comes
back `Incomplete` with one `Unseparated` region covering the patch, and
no curves. Where the surfaces touch along a curve (a plane tangent to a
cylinder patch along a ruling, a sphere inside a cylinder of its radius)
the curve is traced along the direction in which they keep touching and
reported with `tangent` set; a touch at a point is a contact point.

Marching is approximate by design. Its limits:
- the curves are fitted, to about 1e-4 at the default tolerance;
- no singular points: a touch at a sphere's pole or a cone's apex is
  not reported;
- the coincidence check sees only a whole surface lying on the other:
  two patches that overlap in part are marched; the seeds where they lie
  on each other make the result `Incomplete` with an `Unseparated`
  region around the overlap, but spurious curves may still cross it;
- where crossing curves touch each other, traces stop rather than turn
  onto the other branch;
- a loop that no seeding iso-curve crosses and that leaves no distance
  minimum on either grid is missed: a plane cutting a small circle
  (radius 0.014) off a B-spline sphere of radius 1 near its equator or
  pole comes back `Sampled` with no curves.

A wavy B-spline patch through a sphere, marched live below — one fitted
loop in ~8 ms on the same machine (the loop arrives in the viewer as
a sampled polyline until native B-spline types land,
[viewer#5](https://github.com/geomcore/geomcore-viewer/issues/5);
tunable sampling density is
[viewer#12](https://github.com/geomcore/geomcore-viewer/issues/12)):

<iframe src="https://geomcore.github.io/geomcore-viewer/?scene=https://geomcore.github.io/geomcore/assets/scenes/intersect-marching.json" width="100%" height="480" style="border: 1px solid #333; border-radius: 8px;" allow="fullscreen" title="B-spline patch and sphere with their marching intersection loop"></iframe>

All the scenes on this page were computed by the kernel itself —
see `docs/assets/scenes/gen_scenes.py` — not drawn by hand.
