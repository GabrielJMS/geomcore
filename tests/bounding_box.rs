//! Bounding boxes through the public interface: every box holds the
//! shape, sampled densely, and the box of an elementary shape is reached by
//! the shape on every face.

use geomcore::curves::{ElementaryCurve, ParametricCurve3D};
use geomcore::surfaces::{ElementarySurface, ParametricSurface};
use geomcore::{
    BSplineCurve3D, BSplineSurface, Bound3D, BoundingBox3D, Circle3D, Cone, Curve3D, Cylinder,
    Ellipse3D, Hyperbola3D, Line3D, Parabola3D, Plane, Point3D, Sphere, Surface, Torus,
    TrimmedCurve3D, Vector3D,
};
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, PI, TAU};

// ---------------------------------------------------------------- checker

fn coordinate(p: Point3D, k: usize) -> f64 {
    [p.x, p.y, p.z][k]
}

/// The largest value of `f` over `[a, b]`: dense samples, then a
/// golden-section refinement around the best one.
fn maximum(a: f64, b: f64, f: impl Fn(f64) -> f64) -> f64 {
    const SAMPLES: usize = 96;
    let step = (b - a) / SAMPLES as f64;
    let mut best = (f(a), a);
    for i in 1..=SAMPLES {
        let u = if i == SAMPLES { b } else { a + step * i as f64 };
        let value = f(u);
        if value > best.0 {
            best = (value, u);
        }
    }
    let (mut lo, mut hi) = ((best.1 - step).max(a), (best.1 + step).min(b));
    let ratio = 0.5 * (5.0_f64.sqrt() - 1.0);
    for _ in 0..60 {
        let (x1, x2) = (hi - ratio * (hi - lo), lo + ratio * (hi - lo));
        if f(x1) > f(x2) {
            hi = x2;
        } else {
            lo = x1;
        }
    }
    best.0.max(f(0.5 * (lo + hi)))
}

/// The extremes `(min, max)` of each coordinate of `point` over `[a, b]`.
fn extremes(a: f64, b: f64, point: impl Fn(f64) -> Point3D) -> [(f64, f64); 3] {
    std::array::from_fn(|k| {
        let low = -maximum(a, b, |t| -coordinate(point(t), k));
        let high = maximum(a, b, |t| coordinate(point(t), k));
        (low, high)
    })
}

/// The extremes of each coordinate of a surface over its parameter box.
fn surface_extremes<S: ParametricSurface>(
    surface: &S,
    (u0, u1): (f64, f64),
    (v0, v1): (f64, f64),
) -> [(f64, f64); 3] {
    std::array::from_fn(|k| {
        let value = |sign: f64| {
            maximum(u0, u1, |u| {
                maximum(v0, v1, |v| sign * coordinate(surface.eval_point(u, v), k))
            })
        };
        (-value(-1.0), value(1.0))
    })
}

fn scale(b: BoundingBox3D) -> f64 {
    1.0 + b.min().distance(Point3D::ORIGIN) + b.max().distance(Point3D::ORIGIN)
}

/// Asserts that the box holds every extreme and, if `tight`, that the
/// extremes reach every face.
fn check(b: BoundingBox3D, extremes: [(f64, f64); 3], tight: bool, what: &str) {
    let rounding = 1e-12 * scale(b);
    for (k, (low, high)) in extremes.into_iter().enumerate() {
        let (min, max) = (coordinate(b.min(), k), coordinate(b.max(), k));
        assert!(min <= low + rounding, "{what}: axis {k}, {min} > {low}");
        assert!(high <= max + rounding, "{what}: axis {k}, {high} > {max}");
        if tight {
            let reach = 1e-9 * scale(b);
            assert!(low - min <= reach, "{what}: axis {k}, min {min} vs {low}");
            assert!(max - high <= reach, "{what}: axis {k}, max {max} vs {high}");
        }
    }
}

/// The box of `shape` through the implementation for references.
fn through_reference<T: Bound3D>(shape: &T) -> Option<BoundingBox3D> {
    <&T as Bound3D>::bounding_box(&shape)
}

fn check_curve<C: ParametricCurve3D + Bound3D>(curve: &C, tight: bool, what: &str) {
    let (first, last) = curve.bounds();
    let b = curve.bounding_box().expect("bounded curve");
    check(
        b,
        extremes(first, last, |t| curve.eval_point(t)),
        tight,
        what,
    );
}

/// Tilted directions: none along an axis, one along each.
fn normals() -> Vec<Vector3D> {
    vec![
        Vector3D::new(0.3, -0.5, 0.8),
        Vector3D::new(-1.0, 2.0, 0.25),
        Vector3D::new(0.0, 0.6, -0.8),
        Vector3D::X,
        Vector3D::Y,
        Vector3D::Z,
    ]
}

/// A direction orthogonal to `n`.
fn across(n: Vector3D) -> Vector3D {
    let other = if n.x.abs() < 0.9 {
        Vector3D::X
    } else {
        Vector3D::Y
    };
    n.cross(other)
}

/// Ranges of an angle: arcs on either side of the seam, across it, a
/// whole turn and a sliver.
const ARCS: [(f64, f64); 7] = [
    (0.0, FRAC_PI_2),
    (0.3, 2.9),
    (-1.0, 1.0),
    (5.0, 7.5),
    (1.0, 1.0 + TAU),
    (2.0, 2.001),
    (-4.0, -0.5),
];

// ---------------------------------------------------------------- elementary curves

#[test]
fn test_circles_and_arcs_are_tight() {
    let center = Point3D::new(1.0, -2.0, 3.0);
    for n in normals() {
        let circle = Circle3D::new(center, n, 2.5).unwrap();
        check_curve(&circle, true, "circle");
        for (first, last) in ARCS {
            let arc = TrimmedCurve3D::new(circle, first, last).unwrap();
            check_curve(&arc, true, &format!("arc {first}..{last} about {n:?}"));
        }
    }
}

#[test]
fn test_quarter_arc_box_is_its_extreme_points() {
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    let arc = TrimmedCurve3D::new(circle, 0.25, FRAC_PI_2 + 0.25).unwrap();
    let b = arc.bounding_box().unwrap();
    // The arc passes through the top of the circle but not its sides.
    assert_eq!(b.max().y, 1.0);
    assert_eq!(b.min().x, arc.end_point().x);
    assert_eq!(b.max().x, arc.start_point().x);
    assert_eq!(b.min().y, arc.start_point().y);
}

#[test]
fn test_ellipses_and_arcs_are_tight() {
    let center = Point3D::new(-3.0, 0.5, 2.0);
    for n in normals() {
        let ellipse = Ellipse3D::new(center, n, across(n), 4.0, 1.5).unwrap();
        check_curve(&ellipse, true, "ellipse");
        for (first, last) in ARCS {
            let arc = TrimmedCurve3D::new(ellipse, first, last).unwrap();
            check_curve(&arc, true, &format!("elliptic arc {first}..{last}"));
        }
    }
}

#[test]
fn test_trimmed_parabolas_are_tight() {
    let apex = Point3D::new(0.5, 1.0, -1.0);
    for n in normals() {
        let parabola = Parabola3D::new(apex, n, across(n), 0.75).unwrap();
        for (first, last) in [(-3.0, 2.0), (0.5, 4.0), (-1e-3, 1e-3), (-10.0, -2.0)] {
            let piece = TrimmedCurve3D::new(parabola, first, last).unwrap();
            check_curve(&piece, true, &format!("parabola {first}..{last}"));
        }
    }
}

#[test]
fn test_trimmed_hyperbolas_are_tight() {
    let center = Point3D::new(2.0, 0.0, 1.0);
    for n in normals() {
        let hyperbola = Hyperbola3D::new(center, n, across(n), 2.0, 0.7).unwrap();
        for (first, last) in [(-1.5, 2.0), (0.2, 1.0), (-3.0, -0.1)] {
            let piece = TrimmedCurve3D::new(hyperbola, first, last).unwrap();
            check_curve(&piece, true, &format!("hyperbola {first}..{last}"));
        }
    }
}

#[test]
fn test_segments_are_their_end_points() {
    let (a, b) = (Point3D::new(1.0, 5.0, -2.0), Point3D::new(-3.0, 6.0, 0.0));
    let segment = TrimmedCurve3D::segment(a, b).unwrap();
    assert_eq!(segment.bounding_box(), Some(BoundingBox3D::new(a, b)));
    let line = Line3D::new(a, Vector3D::new(1.0, 1.0, 1.0)).unwrap();
    let piece = TrimmedCurve3D::new(line, -2.0, 3.0).unwrap();
    check_curve(&piece, true, "line piece");
}

#[test]
fn test_unbounded_curves_have_no_box() {
    let n = Vector3D::new(0.3, -0.5, 0.8);
    let curves: [ElementaryCurve; 3] = [
        Line3D::new(Point3D::ORIGIN, n).unwrap().into(),
        Parabola3D::new(Point3D::ORIGIN, n, across(n), 1.0)
            .unwrap()
            .into(),
        Hyperbola3D::new(Point3D::ORIGIN, n, across(n), 2.0, 1.0)
            .unwrap()
            .into(),
    ];
    for curve in curves {
        assert_eq!(curve.bounding_box(), None);
        assert_eq!(Curve3D::from(curve).bounding_box(), None);
    }
}

#[test]
fn test_run_time_curves_agree_with_concrete_ones() {
    let n = Vector3D::new(-1.0, 2.0, 0.25);
    let circle = Circle3D::new(Point3D::new(1.0, 2.0, 3.0), n, 2.0).unwrap();
    let ellipse = Ellipse3D::new(Point3D::ORIGIN, n, across(n), 3.0, 1.0).unwrap();
    let parabola = Parabola3D::new(Point3D::ORIGIN, n, across(n), 0.5).unwrap();
    let hyperbola = Hyperbola3D::new(Point3D::ORIGIN, n, across(n), 1.0, 2.0).unwrap();
    let line = Line3D::new(Point3D::ORIGIN, n).unwrap();
    let curves: [ElementaryCurve; 5] = [
        circle.into(),
        ellipse.into(),
        parabola.into(),
        hyperbola.into(),
        line.into(),
    ];
    for curve in curves {
        let wrapped = Curve3D::from(curve);
        assert_eq!(wrapped.bounding_box(), curve.bounding_box());
        let (first, last) = (0.5, 2.0);
        let piece = TrimmedCurve3D::new(curve, first, last).unwrap();
        let wrapped_piece = TrimmedCurve3D::new(wrapped, first, last).unwrap();
        assert!(piece.bounding_box().is_some());
        assert_eq!(piece.bounding_box(), wrapped_piece.bounding_box());
        assert_eq!(through_reference(&piece), piece.bounding_box());
    }
    assert_eq!(
        TrimmedCurve3D::new(circle, 0.5, 2.0)
            .unwrap()
            .bounding_box(),
        TrimmedCurve3D::new(Curve3D::from(circle), 0.5, 2.0)
            .unwrap()
            .bounding_box()
    );
    assert_eq!(through_reference(&circle), circle.bounding_box());
}

// ---------------------------------------------------------------- B-spline curves

fn cubic() -> BSplineCurve3D {
    let poles = vec![
        Point3D::new(0.0, 0.0, 0.0),
        Point3D::new(1.0, 2.0, -1.0),
        Point3D::new(2.0, -1.0, 1.0),
        Point3D::new(3.0, 3.0, 0.5),
        Point3D::new(4.0, -2.0, 2.0),
        Point3D::new(5.0, 0.5, -0.5),
    ];
    BSplineCurve3D::new(3, poles, vec![0.0, 1.0, 2.5, 3.0], vec![4, 1, 1, 4], false).unwrap()
}

/// A rational quadratic: the unit circle as four 90° arcs.
fn rational_circle() -> BSplineCurve3D {
    let s = FRAC_1_SQRT_2;
    let poles = vec![
        Point3D::new(1.0, 0.0, 0.0),
        Point3D::new(1.0, 1.0, 0.0),
        Point3D::new(0.0, 1.0, 0.0),
        Point3D::new(-1.0, 1.0, 0.0),
        Point3D::new(-1.0, 0.0, 0.0),
        Point3D::new(-1.0, -1.0, 0.0),
        Point3D::new(0.0, -1.0, 0.0),
        Point3D::new(1.0, -1.0, 0.0),
        Point3D::new(1.0, 0.0, 0.0),
    ];
    let weights = vec![1.0, s, 1.0, s, 1.0, s, 1.0, s, 1.0];
    BSplineCurve3D::new_rational(
        2,
        poles,
        weights,
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
        vec![3, 2, 2, 2, 3],
        false,
    )
    .unwrap()
}

fn rational_cubic() -> BSplineCurve3D {
    let curve = cubic();
    let weights = vec![1.0, 4.0, 0.3, 2.0, 0.5, 1.0];
    BSplineCurve3D::new_rational(
        3,
        curve.poles().to_vec(),
        weights,
        curve.knots().to_vec(),
        curve.multiplicities().to_vec(),
        false,
    )
    .unwrap()
}

fn periodic() -> BSplineCurve3D {
    let poles = vec![
        Point3D::new(1.0, 0.0, 0.5),
        Point3D::new(0.0, 2.0, -0.5),
        Point3D::new(-1.5, 0.0, 1.0),
        Point3D::new(0.0, -1.0, 0.0),
        Point3D::new(2.0, -2.0, -1.0),
    ];
    BSplineCurve3D::new(
        3,
        poles,
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        vec![1; 6],
        true,
    )
    .unwrap()
}

fn bsplines() -> Vec<(&'static str, BSplineCurve3D)> {
    vec![
        ("cubic", cubic()),
        ("rational circle", rational_circle()),
        ("rational cubic", rational_cubic()),
        ("periodic", periodic()),
    ]
}

#[test]
fn test_bspline_boxes_hold_the_curve() {
    for (name, curve) in bsplines() {
        check_curve(&curve, false, name);
        let pole_box = BoundingBox3D::from_points(curve.poles().iter().copied()).unwrap();
        assert_eq!(curve.bounding_box(), Some(pole_box), "{name}");
        assert_eq!(Curve3D::from(curve.clone()).bounding_box(), Some(pole_box));
    }
}

#[test]
fn test_trimmed_bspline_boxes_hold_the_piece_and_reach_its_ends() {
    for (name, curve) in bsplines() {
        let (start, end) = curve.bounds();
        let length = end - start;
        let mut ranges = vec![
            (start, end),
            (start + 0.1 * length, start + 0.45 * length),
            (start + 0.37 * length, start + 0.38 * length),
            (start + 0.05 * length, start + 0.95 * length),
        ];
        if curve.is_periodic() {
            ranges.push((end - 0.3 * length, end + 0.2 * length));
            ranges.push((start - 2.0 * length, start - 1.1 * length));
        }
        let whole = curve.bounding_box().unwrap();
        for (first, last) in ranges {
            let what = format!("{name} over {first}..{last}");
            let piece = TrimmedCurve3D::new(&curve, first, last).unwrap();
            check_curve(&piece, false, &what);
            let b = piece.bounding_box().unwrap();
            let rounding = 1e-12 * scale(b);
            assert!(
                whole.enlarged(rounding).encloses(b.min())
                    && whole.enlarged(rounding).encloses(b.max()),
                "{what}: beyond the pole box"
            );
            for end_point in [piece.start_point(), piece.end_point()] {
                assert!(b.enlarged(rounding).encloses(end_point), "{what}");
            }
            let owned = TrimmedCurve3D::new(curve.clone(), first, last).unwrap();
            assert_eq!(owned.bounding_box(), Some(b));
            let wrapped = TrimmedCurve3D::new(Curve3D::from(curve.clone()), first, last).unwrap();
            assert_eq!(wrapped.bounding_box(), Some(b));
        }
    }
}

#[test]
fn test_trimmed_bspline_box_is_tighter_than_the_pole_box() {
    // The rational circle's poles reach the corners of the square around
    // it; a piece's Bézier control points stay near the piece.
    let curve = rational_circle();
    let piece = TrimmedCurve3D::new(&curve, 0.2, 0.8).unwrap();
    let b = piece.bounding_box().unwrap();
    assert!(b.max().y < 1.0 && b.min().x > 0.0, "{b:?}");
}

#[test]
fn test_trimmed_linear_bspline_box_is_exact() {
    let poles = vec![
        Point3D::new(0.0, 0.0, 0.0),
        Point3D::new(2.0, 2.0, 0.0),
        Point3D::new(4.0, 0.0, 1.0),
    ];
    let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0, 2.0], vec![2, 1, 2], false).unwrap();
    let piece = TrimmedCurve3D::new(&curve, 0.5, 1.5).unwrap();
    check_curve(&piece, true, "polyline");
}

// ---------------------------------------------------------------- surfaces

#[test]
fn test_spheres_and_tori_are_tight() {
    let center = Point3D::new(1.0, -1.0, 2.0);
    let sphere = Sphere::new(center, 3.0).unwrap();
    let b = sphere.bounding_box().unwrap();
    check(
        b,
        surface_extremes(&sphere, sphere.u_bounds(), sphere.v_bounds()),
        true,
        "sphere",
    );
    for n in normals() {
        for (major, minor) in [(5.0, 1.5), (1.0, 2.0), (0.0, 1.0), (2.0, 0.0)] {
            let torus = Torus::new(center, n, major, minor).unwrap();
            let b = torus.bounding_box().unwrap();
            let extremes = surface_extremes(&torus, torus.u_bounds(), torus.v_bounds());
            check(
                b,
                extremes,
                true,
                &format!("torus {major}, {minor} about {n:?}"),
            );
        }
    }
}

#[test]
fn test_unbounded_surfaces_have_no_box() {
    let n = Vector3D::new(0.3, -0.5, 0.8);
    let surfaces: [ElementarySurface; 3] = [
        Plane::new(Point3D::ORIGIN, n).unwrap().into(),
        Cylinder::new(Point3D::ORIGIN, n, 1.0).unwrap().into(),
        Cone::new(Point3D::ORIGIN, n, 0.3, 1.0).unwrap().into(),
    ];
    for surface in surfaces {
        assert_eq!(surface.bounding_box(), None);
        assert_eq!(Surface::from(surface).bounding_box(), None);
    }
}

#[test]
fn test_run_time_surfaces_agree_with_concrete_ones() {
    let sphere = Sphere::new(Point3D::new(0.0, 1.0, 0.0), 2.0).unwrap();
    let torus = Torus::new(Point3D::ORIGIN, Vector3D::new(1.0, 1.0, 0.0), 3.0, 1.0).unwrap();
    for surface in [
        ElementarySurface::from(sphere),
        ElementarySurface::from(torus),
    ] {
        assert!(surface.bounding_box().is_some());
        assert_eq!(
            Surface::from(surface).bounding_box(),
            surface.bounding_box()
        );
        assert_eq!(through_reference(&surface), surface.bounding_box());
    }
}

fn bspline_surface(rational: bool) -> BSplineSurface {
    let poles: Vec<Vec<Point3D>> = (0..4)
        .map(|i| {
            (0..3)
                .map(|j| {
                    let (x, y) = (i as f64, j as f64);
                    Point3D::new(x, y, (x * 1.3 + y * 0.7).sin() * 2.0)
                })
                .collect()
        })
        .collect();
    let (u_knots, u_mults) = (vec![0.0, 0.4, 1.0], vec![3, 1, 3]);
    let (v_knots, v_mults) = (vec![0.0, 1.0], vec![3, 3]);
    if rational {
        let weights = vec![
            vec![1.0, 2.0, 1.0],
            vec![0.5, 3.0, 1.0],
            vec![1.0, 0.2, 4.0],
            vec![1.0, 1.0, 1.0],
        ];
        BSplineSurface::new_rational(
            2, 2, poles, weights, u_knots, u_mults, v_knots, v_mults, false, false,
        )
        .unwrap()
    } else {
        BSplineSurface::new(
            2, 2, poles, u_knots, u_mults, v_knots, v_mults, false, false,
        )
        .unwrap()
    }
}

#[test]
fn test_bspline_surface_boxes_hold_the_surface() {
    for rational in [false, true] {
        let surface = bspline_surface(rational);
        let b = surface.bounding_box().unwrap();
        let (u, v) = (surface.u_bounds(), surface.v_bounds());
        check(
            b,
            surface_extremes(&surface, u, v),
            false,
            "B-spline surface",
        );
        let corners = [
            surface.eval_point(u.0, v.0),
            surface.eval_point(u.1, v.1),
            surface.eval_point(u.0, v.1),
        ];
        assert!(corners.iter().all(|&p| b.encloses(p)));
        assert_eq!(Surface::from(surface.clone()).bounding_box(), Some(b));
    }
}

// ---------------------------------------------------------------- culling

#[test]
fn test_boxes_cull_apart_shapes_and_keep_meeting_ones() {
    let sphere = Sphere::new(Point3D::ORIGIN, 1.0).unwrap();
    let circle = Circle3D::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Z, 0.5).unwrap();
    let far = TrimmedCurve3D::new(
        Circle3D::new(Point3D::new(4.0, 0.0, 0.0), Vector3D::Z, 1.0).unwrap(),
        PI - 0.5,
        PI + 0.5,
    )
    .unwrap();
    let s = sphere.bounding_box().unwrap();
    assert!(s.overlaps(circle.bounding_box().unwrap()));
    let far = far.bounding_box().unwrap();
    assert!(!s.overlaps(far));
    // The arc's leftmost point is at x = 3, a distance of 2 from the sphere.
    assert!(!s.enlarged(1.9).overlaps(far));
    assert!(s.enlarged(1.0).overlaps(far.enlarged(1.0)));
}
