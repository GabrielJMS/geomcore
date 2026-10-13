# NURBS curves

Create, evaluate and sample `BSplineCurve3D`. Editing (knots, degree,
splitting) and fitting live in [Editing & fitting](editing-fitting.md).

## Create

Three components: degree, poles, knots with multiplicities. A clamped,
single-span quadratic is a Bézier curve — the same one geomdl's
[basics](https://nurbs-python.readthedocs.io/en/5.x/basics.html)
start from.

=== "Python"

    ```python
    from geomcore import Point3D
    from geomcore.curves import BSplineCurve3D

    poles = [Point3D(1, 0, 0), Point3D(1, 1, 0), Point3D(0, 1, 0)]
    curve = BSplineCurve3D(2, poles, [0.0, 1.0], [3, 3], False)
    assert curve.degree() == 2
    ```

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

Rational (NURBS) curves take one weight per pole via
`BSplineCurve3D.new_rational` / `new_rational`; periodic ones pass
`periodic=True` with a cyclic knot vector.

## Evaluate

=== "Python"

    ```python
    p = curve.eval_point(0.5)
    assert (p.x, p.y, p.z) == (0.75, 0.75, 0.0)
    pts = curve.eval_points([0.0, 0.25, 0.5, 0.75, 1.0])
    ```

=== "Rust"

    ```rust
    let p = curve.eval_point(0.5);
    assert_eq!(p, Point3D::new(0.75, 0.75, 0.0));
    let pts = curve.eval_points(&[0.0, 0.25, 0.5, 0.75, 1.0]);
    ```

Derivatives come from `eval_derivative` (order 1–2); `eval_d1`/`eval_d2`
return the point with its derivatives from a single basis pass.
`bounds()` gives the parameter range.

## Sampling density

geomdl controls plot density with `delta`/`sample_size`. Here sampling
with guarantees is `discretize`: points joined by segments, each
carrying a proven chord-deviation and angle-deviation bound — see
[Discretization](../guides/discretization.md).

=== "Python"

    ```python
    d = curve.discretize(0.01)
    assert d.points[0] == curve.eval_point(0.0)
    ```

=== "Rust"

    ```rust
    use geomcore::{DiscretizationCriteria, Discretize3D};

    let d = curve.discretize(DiscretizationCriteria::new(0.01).unwrap());
    assert_eq!(d.points[0], curve.eval_point(0.0));
    ```

## View it

From a script or notebook, send the real objects — B-splines travel
sampled (curve → 256-segment polyline) until native protocol types land
([viewer#5](https://github.com/geomcore/geomcore-viewer/issues/5)):

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
