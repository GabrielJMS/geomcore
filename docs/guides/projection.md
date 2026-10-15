# Projection & containment

Three questions, one distance: *is the point on the shape*
(`contains`), *where is the nearest point* (`project_point`), and
*where are all the local minima* (`extrema`). Every query takes a
`Tolerance` and decides by distance against `tol.confusion`. Adapted
from [Inversion and projection](https://avenirgeometry.com/articles/projection/).

## The foot-point condition

A foot point is where the curve meets its own tangent at a right angle
— the distance is stationary exactly where $(C(u) - P) \cdot C'(u) =
0$. On the ten [elementary](../geometry/elementary.md) geometries this
stationarity equation has a closed form (quartic/cubic on the conics,
an `atan2` on the rest), so projection is exact. On B-splines there is
no closed form: the kernel subdivides Bézier pieces with a bounding-box
hierarchy and solves each convex piece with bounded Newton — every
minimum found, however close two basins sit.

<figure class="geom-fig">
<img src="../../assets/figures/projection/01-four-feet-on-an-ellipse.svg" alt="Four foot points on an ellipse">
<figcaption>Four feet on an ellipse: one query point, four stationary points, one nearest.</figcaption>
</figure>

An ellipse already shows why `extrema` (all minima) and
`project_point` (the nearest) are different calls: a single point can
have four feet, and the nearest is only one of them. Computed live
below — `extrema` on a 3×2 ellipse from a near-centre query, four feet:

<iframe src="https://geomcore.github.io/geomcore-viewer/?scene=https://geomcore.github.io/geomcore/assets/scenes/projection-feet.json" width="100%" height="480" style="border: 1px solid #333; border-radius: 8px;" allow="fullscreen" title="Ellipse with the four projection feet of one query point"></iframe>

<figure class="geom-fig">
<img src="../../assets/figures/projection/02-three-candidates-on-a-b-spline.svg" alt="Candidate feet on a B-spline">
<figcaption>On a B-spline the candidates come from every span — the search must cover all of them to never miss the nearest.</figcaption>
</figure>

## contains

A point *lies on* a curve or surface when it is within `tol.confusion`
of it, tested by re-evaluating the inverse parameter — never by
pointer or topology.

=== "Python"

    ```python
    circle.contains(Point3D(2, 0, 0))          # True, on the radius-2 circle
    sphere.contains(Point3D(0, 0, 2.01))       # False, off the surface
    ```

=== "Rust"

    ```rust
    use geomcore::Tolerance;

    let tol = Tolerance::DEFAULT;
    assert!(circle.contains(Point3D::new(2.0, 0.0, 0.0), tol));
    assert!(!sphere.contains(Point3D::new(0.0, 0.0, 2.01), tol));
    ```

## project_point

Returns the closest point: its parameter(s), the point itself and the
distance; sub-tolerance distances snap to `0.0`, so projection agrees
with `contains`. In Python the result is a `CurveProjection`,
`CurveProjection2D` or `SurfaceProjection` that also holds the `target`
it was computed on and the `query_point`. Batch variants
(`project_points`) share the setup and return a list.

=== "Python"

    ```python
    foot = circle.project_point(Point3D(3, 0, 0))
    foot.parameter, foot.point, foot.distance, foot.target is circle
    sfoot = sphere.project_point(Point3D(0, 0, 5))
    sfoot.u, sfoot.v, sfoot.point, sfoot.distance
    ```

=== "Rust"

    ```rust
    let foot = circle.project_point(Point3D::new(3.0, 0.0, 0.0), tol);
    assert_eq!((foot.parameter, foot.distance), (0.0, 1.0));
    assert_eq!(foot.point, Point3D::new(2.0, 0.0, 0.0));

    let sfoot = sphere.project_point(Point3D::new(0.0, 0.0, 5.0), tol);
    ```

## extrema

All local minima of the distance, nearest first — plus the open ends
of bounded curves. What `project_point` returns first, `extrema`
returns whole.

=== "Python"

    ```python
    minima = ellipse.extrema(Point3D(4, 1, 0))   # [CurveProjection, ...] nearest first
    ```

=== "Rust"

    ```rust
    let minima = ellipse.extrema(Point3D::new(4.0, 1.0, 0.0), tol);
    ```

On seams the parameters are normalised into `bounds()`; a hit within
confusion of the seam reports 0 whichever side it was found from, so
callers can test results against `bounds()` and trimmed ranges without
losing hits.

## Curves against curves

`curve_curve_extrema` reports the local minima of the distance between
two curves over their domains — the primitive behind closest-approach
queries and trimmed-arc handling.

=== "Python"

    ```python
    from geomcore import curve_curve_extrema

    minima = curve_curve_extrema(line, circle).minima
    distances = [m.distance for m in minima]
    ```

=== "Rust"

    ```rust
    use geomcore::curve_curve_extrema;

    let minima = curve_curve_extrema(&line, &circle, tol);
    let closest = minima.minima().iter().map(|m| m.distance);
    ```
