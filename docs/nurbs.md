# NURBS

B-splines — rational ones are NURBS — are the freeform half of the kernel:
`BSplineCurve3D` and `BSplineSurface`, built from a degree, control poles,
a knot vector and optional weights. Everything else (evaluation, editing,
fitting, exact elementary conversions, intersections) hangs off those two
types. The structure below mirrors the
[NURBS-Python basics](https://nurbs-python.readthedocs.io/en/5.x/basics.html);
the difference is that every snippet runs in Rust *and* Python — pick a
language once and the tabs follow you across pages.

The curve below is the running example: the same quadratic Bézier the
geomdl tutorial starts from, as a clamped degree-2 B-spline.

<iframe src="https://gabrieljms.github.io/geomcore-viewer/?scene=https://gabrieljms.github.io/geomcore/assets/scenes/nurbs-curve.json" width="100%" height="480" style="border: 1px solid #333; border-radius: 8px;" allow="fullscreen" title="Quadratic B-spline with control polygon"></iframe>

## Create a curve

Three components: degree, poles, knots with multiplicities. A clamped,
single-span quadratic is a Bézier curve.

=== "Rust"

    ```rust
    use geomcore::{BSplineCurve3D, Point3D};

    let poles = vec![
        Point3D::new(1.0, 0.0, 0.0),
        Point3D::new(1.0, 1.0, 0.0),
        Point3D::new(0.0, 1.0, 0.0),
    ];
    let curve = BSplineCurve3D::new(2, poles, vec![0.0, 1.0], vec![3, 3], false).unwrap();
    assert_eq!(curve.degree(), 2);
    ```

=== "Python"

    ```python
    from geomcore import Point3D
    from geomcore.curves import BSplineCurve3D

    poles = [Point3D(1, 0, 0), Point3D(1, 1, 0), Point3D(0, 1, 0)]
    curve = BSplineCurve3D(2, poles, [0.0, 1.0], [3, 3], False)
    assert curve.degree() == 2
    ```

Rational (NURBS) curves take one weight per pole via `new_rational` /
`BSplineCurve3D.new_rational`; periodic ones pass `periodic=True` with a
cyclic knot vector.

## Evaluate

=== "Rust"

    ```rust
    let p = curve.eval_point(0.5);
    assert_eq!(p, Point3D::new(0.75, 0.75, 0.0));
    let pts = curve.eval_points(&[0.0, 0.25, 0.5, 0.75, 1.0]);
    ```

=== "Python"

    ```python
    p = curve.eval_point(0.5)
    assert (p.x, p.y, p.z) == (0.75, 0.75, 0.0)
    pts = curve.eval_points([0.0, 0.25, 0.5, 0.75, 1.0])
    ```

Derivatives come from `eval_derivative` (order 1–2 on curves); `bounds()`
gives the parameter range.

## Sampling density

geomdl controls plot density with `delta`/`sample_size`. Here sampling
with guarantees is `discretize`: points joined by segments, each carrying
a proven chord-deviation and angle-deviation bound.

=== "Rust"

    ```rust
    use geomcore::{DiscretizationCriteria, Discretize3D};

    let d = curve.discretize(DiscretizationCriteria::new(0.01).unwrap());
    assert_eq!(d.points[0], curve.eval_point(0.0));
    ```

=== "Python"

    ```python
    d = curve.discretize(0.01)
    assert d.points[0] == curve.eval_point(0.0)
    ```

## Edit: knots and degree

Insertion, refinement, elevation and removal keep the shape exactly
(rational poles are edited in homogeneous form). These are Rust-only for
now — Python bindings are tracked in
[geomcore#58](https://github.com/GabrielJMS/geomcore/issues/58).

```rust
let refined = curve.insert_knot(0.5, 1).unwrap();
assert_eq!(refined.eval_point(0.5), curve.eval_point(0.5));

let dense = curve.refine_knots(&[0.25, 0.5, 0.75]).unwrap();

let elevated = curve.elevate_degree(1).unwrap();
assert_eq!(elevated.degree(), 3);

let simplified = elevated.remove_knot(0.5, 1, 1e-9).unwrap();
```

## Split, extract, reverse

`split_at` cuts into two clamped pieces, `extract` keeps a sub-range,
`reversed` mirrors the parameter, `to_bezier_pieces` decomposes into
Bézier spans. Same Python story as above
([geomcore#58](https://github.com/GabrielJMS/geomcore/issues/58)).

```rust
let (left, right) = curve.split_at(0.5).unwrap();
let piece = curve.extract(0.0, 0.5).unwrap();
let rev = curve.reversed();
assert_eq!(rev.eval_point(0.0), curve.eval_point(1.0));
assert_eq!(curve.to_bezier_pieces().len(), 1); // single span
```

Ranges that cross the seam (up to one period) work on periodic curves.
The pieces are what the viewer can *not* yet take directly — trimmed
curves have no scene mapping
([viewer#6](https://github.com/GabrielJMS/geomcore-viewer/issues/6)) —
so sample them to polylines for display.

## Fit points

Interpolation passes through the points; `approximate` fits with a fixed
pole count; `approximate_within` adds knots until every point is within
`max_deviation` (Rust-only, see
[geomcore#58](https://github.com/GabrielJMS/geomcore/issues/58)).

=== "Rust"

    ```rust
    use geomcore::PointParametrization;

    let points = curve.eval_points(&[0.0, 0.25, 0.5, 0.75, 1.0]);
    let interp = BSplineCurve3D::interpolate(&points, 3, PointParametrization::Centripetal).unwrap();
    let approx = BSplineCurve3D::approximate(&points, 2, 3, PointParametrization::Centripetal).unwrap();
    let within = BSplineCurve3D::approximate_within(&points, 2, 1e-6, PointParametrization::Centripetal).unwrap();
    ```

=== "Python"

    ```python
    points = curve.eval_points([0.0, 0.25, 0.5, 0.75, 1.0])
    interp = BSplineCurve3D.interpolate(points, 3)
    approx = BSplineCurve3D.approximate(points, 2, 3)
    ```

## Exact elementary conversions

Circles and ellipses convert to exact periodic rational quadratics;
trimmed lines, arcs and conic pieces become clamped B-splines over the
same range; planes, cylinders, cones, spheres and tori to exact patches.
Rust-only for now
([geomcore#58](https://github.com/GabrielJMS/geomcore/issues/58)).

```rust
use geomcore::{Circle3D, Ellipse3D, Point3D, Vector3D};

let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
let circle_bs = circle.to_bspline();
assert!(circle_bs.is_periodic());

let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.0).unwrap();
let ellipse_bs = ellipse.to_bspline();
```

## Measure and bound

Arc length integrates the speed to a requested error; `bounding_box`
gives the tight pole box (elementary shapes get closed-form boxes).

```rust
use geomcore::Bound3D;

let len = curve.length(1e-9);
let bb = curve.bounding_box().unwrap();
assert_eq!((bb.min(), bb.max()), (Point3D::new(0.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 0.0)));
```

Boxes have no viewer mapping yet
([viewer#8](https://github.com/GabrielJMS/geomcore-viewer/issues/8)).

## Surfaces

Same three components per direction (`_u`/`_v` suffixes, as in geomdl).
A bilinear patch over the unit square:

=== "Rust"

    ```rust
    use geomcore::BSplineSurface;

    let poles = vec![
        vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(1.0, 0.0, 0.0)],
        vec![Point3D::new(0.0, 1.0, 0.0), Point3D::new(1.0, 1.0, 0.0)],
    ];
    let surf = BSplineSurface::new(1, 1, poles, vec![0.0, 1.0], vec![2, 2],
                                   vec![0.0, 1.0], vec![2, 2], false, false).unwrap();
    assert_eq!(surf.eval_point(0.5, 0.5), Point3D::new(0.5, 0.5, 0.0));
    ```

=== "Python"

    ```python
    from geomcore.surfaces import BSplineSurface

    poles = [[Point3D(0, 0, 0), Point3D(1, 0, 0)],
             [Point3D(0, 1, 0), Point3D(1, 1, 0)]]
    surf = BSplineSurface(1, 1, poles, [0.0, 1.0], [2, 2], [0.0, 1.0], [2, 2], False, False)
    assert surf.eval_point(0.5, 0.5) == Point3D(0.5, 0.5, 0)
    ```

Surface fitting (`approximate` over a point grid, `approximate_within`)
and per-direction editing follow the curve API in Rust. In the viewer a
B-spline surface arrives as a 48×48 mesh over an explicit `u=`/`v=`
window — there is no domain getter yet
([viewer#7](https://github.com/GabrielJMS/geomcore-viewer/issues/7)) —
and curves on B-spline surfaces have no pcurve mapping
([viewer#9](https://github.com/GabrielJMS/geomcore-viewer/issues/9)).

## Intersections

B-spline curve–curve, curve–surface and surface–surface (marching)
queries subdivide Bézier pieces with hull pruning instead of a sampling
grid, and report whether the search was `Complete`. Fell out of the
0.32 scope for tabs; see the changelog.

## View it

From a script or notebook, send the real objects — B-splines travel
sampled (curve → 256-segment polyline, surface → 48×48 mesh) until
native protocol types land
([viewer#5](https://github.com/GabrielJMS/geomcore-viewer/issues/5)):

=== "Python"

    ```python
    import geomcore_viewer as gv

    view = gv.Viewer("Quadratic B-spline")
    view.add(curve, label="C")
    view.polyline([Point3D(1, 0, 0), Point3D(1, 1, 0), Point3D(0, 1, 0)],
                  label="control polygon", dashed=True)
    ```

=== "Rust"

    ```rust
    use geomcore_viewer::{Geom, Opts, Viewer};

    let mut view = Viewer::new("Quadratic B-spline")?;
    view.add(Geom::from(&curve), Opts::new().label("C"))?;
    ```
