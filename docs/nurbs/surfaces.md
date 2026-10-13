# NURBS surfaces

`BSplineSurface` carries the same three components per direction
(degree, poles, knots) with `_u`/`_v` suffixes, as in geomdl. A
bilinear patch over the unit square:

=== "Python"

    ```python
    from geomcore import Point3D
    from geomcore.surfaces import BSplineSurface

    poles = [[Point3D(0, 0, 0), Point3D(1, 0, 0)],
             [Point3D(0, 1, 0), Point3D(1, 1, 0)]]
    surf = BSplineSurface(1, 1, poles, [0.0, 1.0], [2, 2], [0.0, 1.0], [2, 2], False, False)
    assert surf.eval_point(0.5, 0.5) == Point3D(0.5, 0.5, 0)
    ```

=== "Rust"

    ```rust
    use geomcore::{BSplineSurface, Point3D};

    let poles = vec![
        vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(1.0, 0.0, 0.0)],
        vec![Point3D::new(0.0, 1.0, 0.0), Point3D::new(1.0, 1.0, 0.0)],
    ];
    let surf = BSplineSurface::new(1, 1, poles, vec![0.0, 1.0], vec![2, 2],
                                   vec![0.0, 1.0], vec![2, 2], false, false).unwrap();
    assert_eq!(surf.eval_point(0.5, 0.5), Point3D::new(0.5, 0.5, 0.0));
    ```

Derivatives come from `eval_derivative` (`du`, `dv` up to second
order); per-direction editing (`insert_u_knot`, `elevate_v_degree`,
`split_u_at`, `extract`, `reversed_u`/`reversed_v`,
`to_bezier_patches`) mirrors the curve API in
[Editing & fitting](editing-fitting.md), and is Rust-only for now
([geomcore#58](https://github.com/geomcore/geomcore/issues/58)).

Surface fitting (`approximate` over a point grid,
`approximate_within`) is covered there too.

## Viewing

In the viewer a B-spline surface arrives as a 48×48 mesh. Python needs
an explicit `u=`/`v=` window because the bindings don't expose the
surface's `u_bounds`/`v_bounds` yet (the Rust client reads them itself;
bindings tracked in
[geomcore#58](https://github.com/geomcore/geomcore/issues/58),
[viewer#7](https://github.com/geomcore/geomcore-viewer/issues/7)) —
and curves on B-spline surfaces have no pcurve mapping
([viewer#9](https://github.com/geomcore/geomcore-viewer/issues/9)).

=== "Python"

    ```python
    import geomcore_viewer as gv

    view = gv.Viewer("Bilinear patch")
    view.add(surf, label="S", u=(0.0, 1.0), v=(0.0, 1.0))
    ```

=== "Rust"

    ```rust
    use geomcore_viewer::{Geom, Opts, Viewer};

    let mut view = Viewer::new("Bilinear patch")?;
    view.add(Geom::from(&surf), Opts::new().label("S"))?;
    ```
