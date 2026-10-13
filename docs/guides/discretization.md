# Discretization

Points on a curve, in parameter order, joined by straight segments —
each point keeping its parameter, each segment keeping its chord
deviation and angle deviation. The deviations are proven upper bounds,
never estimates: a chord-deviation bound is the largest distance from
the curve piece to its chord, an angle-deviation bound the largest
angle between the curve's tangent and the chord.

## Why proven bounds

A uniform parameter grid is not uniform on the curve — on an ellipse
the chords bunch at the major-axis ends and stretch at the sides, and
the error differs by a factor of 2.4. Discretization inverts the
problem: give the allowed deviation, get the points that hold it.

<figure class="geom-fig">
<img src="../../assets/figures/meshing/01-two-arcs-of-the-same-circle.svg" alt="Two arcs of the same circle">
<figcaption>Two arcs of one circle: the same deviation asks for different point counts on different curvatures.</figcaption>
</figure>

Circles use exact deviations; B-splines split per knot span and bound
each span from its Bézier control points (homogeneous form when
rational, reporting π where no bound exists — next to a cusp or
(nearly) coincident poles). Ellipses, trimmed lines/parabolas/
hyperbolas and run-time curves use exact closed-form deviations of
equal parameter steps; a trimmed line is one segment. Closed curves
end on their first point, with at least three segments.

<figure class="geom-fig">
<img src="../../assets/figures/meshing/02-a-curve-of-angular-step-against.svg" alt="Angular step against deviation">
<figcaption>The angle bound comes from the tangent's turning over the step — where the speed reaches zero no bound exists.</figcaption>
</figure>

## Use

Any positive chord deviation is accepted, including below the
confusion tolerance (points cannot lie closer than coordinate rounding,
about 1e-15 of the curve's distance from the origin). `point_count`
sizes a caller-owned buffer first; `discretize_into` fills it without
allocating.

=== "Python"

    ```python
    d = curve.discretize(0.01)                    # chord deviation ≤ 0.01
    d = curve.discretize(0.01, 0.05)              # ... and angle deviation ≤ 0.05 rad
    params, points = d.parameters, d.points
    chord, angle = d.chord_deviations, d.angle_deviations
    ```

=== "Rust"

    ```rust
    use geomcore::{DiscretizationCriteria, Discretize3D, ParametricCurve3D};

    let criteria = DiscretizationCriteria::new(0.01).unwrap();
    let n = curve.point_count(criteria);
    let d = curve.discretize(criteria);
    assert_eq!(d.points[0], curve.eval_point(curve.bounds().0));
    ```

In the plane, `Discretize2D` gives a `Discretization2D` of a
`Circle2D` or a trimmed 2D curve, its chord deviations measured in the
plane. For surface meshing these curves are the edges first — faces
triangulate after their boundary is fixed.
