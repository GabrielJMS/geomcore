//! The routed intersection: exact when a closed form exists, numeric
//! otherwise, and the result says which method ran.

use geomcore::curves::{Curve3D, ElementaryCurve, TrimmedCurve3D};
use geomcore::surfaces::Surface;
use geomcore::*;
use std::f64::consts::{PI, TAU};

const TOL: Tolerance = Tolerance::DEFAULT;

fn assert_hits_lie_on<C: ParametricCurve3D, S: ParametricSurface>(
    r: &CurveSurfaceIntersection,
    c: &C,
    s: &S,
) {
    for h in r.hits() {
        assert!(
            c.eval_point(h.curve_parameter).distance(h.point) <= 1e-7,
            "curve parameter does not give the point"
        );
        let (u, v) = h.surface_parameters;
        assert!(
            s.eval_point(u, v).distance(h.point) <= 1e-7,
            "surface parameters do not give the point"
        );
    }
}

#[test]
fn test_line_sphere_routes_exact_with_surface_parameters() {
    let line = Line3D::new(Point3D::new(0.0, 0.0, -5.0), Vector3D::Z).unwrap();
    let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    let r = line.intersect(&sphere, TOL);
    assert_eq!(r.method(), IntersectionMethod::Exact);
    assert!(r.is_complete());
    assert_eq!(r.hits().len(), 2); // through both poles
    assert_hits_lie_on(&r, &line, &sphere);
}

#[test]
fn test_reverse_order_reads_the_same() {
    let line = Line3D::new(Point3D::new(-5.0, 0.5, 0.0), Vector3D::X).unwrap();
    let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    assert_eq!(
        line.intersect(&cylinder, TOL),
        cylinder.intersect(&line, TOL)
    );
}

#[test]
fn test_routed_matches_exact_hits() {
    let circle = Circle3D::new(Point3D::new(0.0, 0.0, 0.5), Vector3D::X, 2.0).unwrap();
    let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    let routed = circle.intersect(&sphere, TOL);
    let ExactCurveSurfaceIntersection::Points(exact) =
        ElementaryCurve::from(circle).intersect_exact(&ElementarySurface::from(sphere), TOL)
    else {
        panic!("expected points")
    };
    let routed_parameters: Vec<f64> = routed.hits().iter().map(|h| h.curve_parameter).collect();
    let exact_parameters: Vec<f64> = exact.iter().map(|h| h.parameter).collect();
    assert_eq!(routed_parameters, exact_parameters);
}

#[test]
fn test_line_in_plane_is_an_unbounded_overlap() {
    let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    let r = line.intersect(&plane, TOL);
    assert!(r.hits().is_empty());
    assert_eq!(r.overlaps(), &[(f64::NEG_INFINITY, f64::INFINITY)]);
    assert!(!r.is_empty());
}

#[test]
fn test_circle_on_sphere_is_a_full_period_overlap() {
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    let r = circle.intersect(&sphere, TOL);
    assert_eq!(r.overlaps(), &[(0.0, TAU)]);
}

#[test]
fn test_ellipse_torus_routes_numeric() {
    let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 3.0, 1.0).unwrap();
    let ellipse = Ellipse3D::from_frame(Frame3D::WORLD, 5.0, 1.0).unwrap();
    let r = ellipse.intersect(&torus, TOL);
    assert_eq!(r.method(), IntersectionMethod::Numeric);
    assert!(!r.hits().is_empty());
    assert_hits_lie_on(&r, &ellipse, &torus);
}

#[test]
fn test_bspline_through_enum_routes_numeric() {
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::X, 2.0).unwrap();
    let curve = Curve3D::from(circle.to_bspline());
    let plane = Surface::from(Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap());
    let r = curve.intersect(&plane, TOL);
    assert_eq!(r.method(), IntersectionMethod::Numeric);
    assert_eq!(r.hits().len(), 2);
    assert_hits_lie_on(&r, &curve, &plane);
}

#[test]
fn test_trimmed_arc_across_the_seam_keeps_only_its_hits() {
    // A unit circle in z = 0 meets the plane x = cos(0.3) at ±0.3 (≡ 2π - 0.3).
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    let plane = Plane::new(Point3D::new(0.3f64.cos(), 0.0, 0.0), Vector3D::X).unwrap();
    let arc = TrimmedCurve3D::new(circle, -1.0, 1.0).unwrap();
    let r = arc.intersect(&plane, TOL);
    let mut parameters: Vec<f64> = r.hits().iter().map(|h| h.curve_parameter).collect();
    parameters.sort_by(f64::total_cmp);
    assert_eq!(parameters.len(), 2);
    assert!((parameters[0] + 0.3).abs() < 1e-9 && (parameters[1] - 0.3).abs() < 1e-9);
    let far = TrimmedCurve3D::new(circle, 1.0, PI).unwrap();
    assert!(far.intersect(&plane, TOL).is_empty());
}

#[test]
fn test_tangent_hit_has_multiplicity_two() {
    let line = Line3D::new(Point3D::new(-5.0, 1.0, 0.0), Vector3D::X).unwrap();
    let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    let r = line.intersect(&cylinder, TOL);
    assert_eq!(r.hits().len(), 1);
    assert!(r.hits()[0].multiplicity >= 2);
}

fn assert_curve_hits_lie_on<A: ParametricCurve3D, B: ParametricCurve3D>(
    r: &CurveCurveIntersection,
    a: &A,
    b: &B,
) {
    for h in r.hits() {
        assert!(a.eval_point(h.first_parameter).distance(h.point) <= 1e-7);
        assert!(b.eval_point(h.second_parameter).distance(h.point) <= 1e-7);
    }
}

/// Every overlap range is monotone and lies in its curve's range.
fn assert_overlaps_within(r: &CurveCurveIntersection, first: (f64, f64), second: (f64, f64)) {
    let inside = |(low, high): (f64, f64), (start, end): (f64, f64)| {
        low <= high && low >= start - 1e-12 && high <= end + 1e-12
    };
    for o in r.overlaps() {
        assert!(
            inside(o.first_range, first),
            "{:?} not in {:?}",
            o.first_range,
            first
        );
        assert!(
            inside(o.second_range, second),
            "{:?} not in {:?}",
            o.second_range,
            second
        );
    }
}

fn close(a: (f64, f64), b: (f64, f64)) -> bool {
    (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9
}

#[test]
fn test_circle_line_reads_from_the_receiver() {
    let line = Line3D::new(Point3D::new(-5.0, 0.0, 0.0), Vector3D::X).unwrap();
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    let r = circle.intersect(&line, TOL);
    assert_eq!(r.method(), IntersectionMethod::Exact);
    assert_curve_hits_lie_on(&r, &circle, &line);
    let forward = line.intersect(&circle, TOL);
    assert_eq!(r.hits().len(), forward.hits().len());
    for y in forward.hits() {
        assert!(
            r.hits()
                .iter()
                .any(|x| x.first_parameter == y.second_parameter
                    && x.second_parameter == y.first_parameter)
        );
    }
}

#[test]
fn test_coincident_lines_overlap_everywhere() {
    let a = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    let r = a.intersect(&a, TOL);
    assert_eq!(r.overlaps().len(), 1);
    assert_eq!(
        r.overlaps()[0].first_range,
        (f64::NEG_INFINITY, f64::INFINITY)
    );
}

#[test]
fn test_overlapping_segments_clip_to_both_ranges() {
    let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    let a = TrimmedCurve3D::new(line, 0.0, 2.0).unwrap();
    let b = TrimmedCurve3D::new(line, 1.0, 3.0).unwrap();
    let r = a.intersect(&b, TOL);
    assert_eq!(r.overlaps().len(), 1);
    let o = r.overlaps()[0];
    assert!((o.first_range.0 - 1.0).abs() < 1e-12 && (o.first_range.1 - 2.0).abs() < 1e-12);
    assert!((o.second_range.0 - 1.0).abs() < 1e-12 && (o.second_range.1 - 2.0).abs() < 1e-12);
}

#[test]
fn test_bspline_circle_routes_numeric() {
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    let other = Circle3D::new(Point3D::new(2.0, 0.0, 0.0), Vector3D::Z, 2.0).unwrap();
    let bspline = other.to_bspline();
    let r = circle.intersect(&bspline, TOL);
    assert_eq!(r.method(), IntersectionMethod::Numeric);
    assert_eq!(r.hits().len(), 2);
    assert_curve_hits_lie_on(&r, &circle, &bspline);
}

#[test]
fn test_skew_lines_are_empty_and_complete() {
    let a = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    let b = Line3D::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Y).unwrap();
    let r = a.intersect(&b, TOL);
    assert!(r.is_empty() && r.is_complete());
}

#[test]
fn test_coincident_arcs_overlap_on_both_sides_of_the_seam() {
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    let a = TrimmedCurve3D::new(circle, 0.0, 5.0).unwrap();
    let b = TrimmedCurve3D::new(circle, 4.0, 7.0).unwrap();
    let r = a.intersect(&b, TOL);
    assert_eq!(r.method(), IntersectionMethod::Exact);
    assert_eq!(r.overlaps().len(), 2);
    assert_overlaps_within(&r, (0.0, 5.0), (4.0, 7.0));
    let o = r.overlaps();
    assert!(close(o[0].first_range, (0.0, 7.0 - TAU)));
    assert!(close(o[0].second_range, (TAU, 7.0)));
    assert!(close(o[1].first_range, (4.0, 5.0)));
    assert!(close(o[1].second_range, (4.0, 5.0)));
    // The other order reads the same overlaps from b's side.
    let reverse = b.intersect(&a, TOL);
    assert_eq!(reverse.overlaps().len(), 2);
    assert_overlaps_within(&reverse, (4.0, 7.0), (0.0, 5.0));
}

#[test]
fn test_coincident_lines_with_opposite_directions() {
    let a = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    let b = Line3D::new(Point3D::new(5.0, 0.0, 0.0), -Vector3D::X).unwrap();
    // a covers x in [0, 4], b covers x in [2, 5].
    let first = TrimmedCurve3D::new(a, 0.0, 4.0).unwrap();
    let second = TrimmedCurve3D::new(b, 0.0, 3.0).unwrap();
    let r = first.intersect(&second, TOL);
    assert_eq!(r.overlaps().len(), 1);
    assert!(close(r.overlaps()[0].first_range, (2.0, 4.0)));
    assert!(close(r.overlaps()[0].second_range, (1.0, 3.0)));
    // Unbounded and opposite: both ranges are the whole line.
    let whole = a.intersect(&b, TOL);
    assert_eq!(whole.overlaps().len(), 1);
    let o = whole.overlaps()[0];
    assert_eq!(o.first_range, (f64::NEG_INFINITY, f64::INFINITY));
    assert_eq!(o.second_range, (f64::NEG_INFINITY, f64::INFINITY));
}

#[test]
fn test_coincident_full_circles_with_different_frames() {
    let a = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    let rotated = Frame3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::new(1.0, 1.0, 0.0)).unwrap();
    let flipped = Frame3D::new(Point3D::ORIGIN, -Vector3D::Z, Vector3D::Y).unwrap();
    for frame in [rotated, flipped] {
        let b = Circle3D::from_frame(frame, 2.0).unwrap();
        let r = a.intersect(&b, TOL);
        assert_eq!(r.method(), IntersectionMethod::Exact);
        assert_eq!(r.overlaps().len(), 1, "{frame:?}");
        let o = r.overlaps()[0];
        assert!(close(o.first_range, a.bounds()));
        assert!((o.second_range.1 - o.second_range.0 - TAU).abs() < 1e-9);
        assert_overlaps_within(&r, a.bounds(), b.bounds());
    }
}

#[test]
fn test_trimmed_curves_keep_only_their_hits() {
    let line = Line3D::new(Point3D::new(-5.0, 0.0, 0.0), Vector3D::X).unwrap();
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    // The arc across the seam keeps the hit at angle 0, not the one at PI.
    let arc = TrimmedCurve3D::new(circle, -1.0, 1.0).unwrap();
    let r = line.intersect(&arc, TOL);
    assert_eq!(r.hits().len(), 1);
    assert!((r.hits()[0].first_parameter - 7.0).abs() < 1e-12);
    assert!(r.hits()[0].second_parameter.abs() < 1e-12);
    assert_curve_hits_lie_on(&r, &line, &arc);
    // A segment that stops before the circle meets it once, at PI.
    let segment = TrimmedCurve3D::new(line, 0.0, 5.0).unwrap();
    let r = circle.intersect(&segment, TOL);
    assert_eq!(r.hits().len(), 1);
    assert!((r.hits()[0].first_parameter - PI).abs() < 1e-12);
    assert_curve_hits_lie_on(&r, &circle, &segment);
}

#[test]
fn test_routed_curve_curve_matches_exact_hits() {
    let line = Line3D::new(Point3D::new(-5.0, 0.5, 0.0), Vector3D::X).unwrap();
    let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.0).unwrap();
    let routed = Curve3D::from(line).intersect(&Curve3D::from(ellipse), TOL);
    let exact = ElementaryCurve::from(line).intersect_exact(&ElementaryCurve::from(ellipse), TOL);
    let ExactCurveCurveIntersection::Points(exact) = exact else {
        panic!("expected hits");
    };
    assert_eq!(routed.hits(), &exact[..]);
}

#[test]
fn test_arc_across_the_seam_of_a_full_circle_splits_there() {
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    let arc = TrimmedCurve3D::new(circle, -1.0, 1.0).unwrap();
    let r = arc.intersect(&circle, TOL);
    assert_eq!(r.overlaps().len(), 2);
    assert_overlaps_within(&r, (-1.0, 1.0), circle.bounds());
    let o = r.overlaps();
    assert!(close(o[0].first_range, (-1.0, 0.0)));
    assert!(close(o[0].second_range, (TAU - 1.0, TAU)));
    assert!(close(o[1].first_range, (0.0, 1.0)));
    assert!(close(o[1].second_range, (0.0, 1.0)));
    // Read from the full circle: one overlap per side of the arc's seam.
    let reverse = circle.intersect(&arc, TOL);
    assert_eq!(reverse.overlaps().len(), 2);
    assert_overlaps_within(&reverse, circle.bounds(), (-1.0, 1.0));
}
