//! The routed intersection: exact when a closed form exists, numeric
//! otherwise, and the result says which method ran.

use geomcore::curves::{Curve3D, ElementaryCurve, TrimmedCurve2D, TrimmedCurve3D};
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
        if r.method() == IntersectionMethod::Numeric {
            // The numeric search computes every parameter.
            assert!(h.curve_parameter.is_some() && h.surface_parameters.is_some());
        }
        if let Some(t) = h.curve_parameter {
            assert!(
                c.eval_point(t).distance(h.point) <= 1e-7,
                "curve parameter does not give the point"
            );
        }
        if let Some((u, v)) = h.surface_parameters {
            assert!(
                s.eval_point(u, v).distance(h.point) <= 1e-7,
                "surface parameters do not give the point"
            );
        }
    }
}

#[test]
fn test_line_sphere_routes_exact_and_leaves_surface_parameters_to_the_caller() {
    let line = Line3D::new(Point3D::new(0.0, 0.0, -5.0), Vector3D::Z).unwrap();
    let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    let r = line.intersect(&sphere, TOL);
    assert_eq!(r.method(), IntersectionMethod::Exact);
    assert!(r.is_complete());
    assert_eq!(r.hits().len(), 2); // through both poles
    assert_hits_lie_on(&r, &line, &sphere);
    for h in r.hits() {
        // The line parameter comes out of the solve; the sphere's do not.
        assert!(h.curve_parameter.is_some());
        assert_eq!(h.surface_parameters, None);
        let (u, v) = sphere.parameters_of(h.point);
        assert!(sphere.eval_point(u, v).distance(h.point) <= 1e-7);
    }
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
    let routed_parameters: Vec<Option<f64>> =
        routed.hits().iter().map(|h| h.curve_parameter).collect();
    let exact_parameters: Vec<Option<f64>> = exact.iter().map(|h| Some(h.parameter)).collect();
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
    // The arc's range test needs the curve parameter, so it is stored.
    let mut parameters: Vec<f64> = r.hits().iter().filter_map(|h| h.curve_parameter).collect();
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
        if let Some(t) = h.first_parameter {
            assert!(a.eval_point(t).distance(h.point) <= 1e-7);
        }
        if let Some(t) = h.second_parameter {
            assert!(b.eval_point(t).distance(h.point) <= 1e-7);
        }
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
    assert!(o.same_direction);
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
    assert!(o.iter().all(|piece| piece.same_direction));
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
    assert!(!r.overlaps()[0].same_direction);
    // Unbounded and opposite: both ranges are the whole line.
    let whole = a.intersect(&b, TOL);
    assert_eq!(whole.overlaps().len(), 1);
    let o = whole.overlaps()[0];
    assert_eq!(o.first_range, (f64::NEG_INFINITY, f64::INFINITY));
    assert_eq!(o.second_range, (f64::NEG_INFINITY, f64::INFINITY));
    assert!(!o.same_direction);
}

#[test]
fn test_coincident_full_circles_with_different_frames() {
    let a = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    let rotated = Frame3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::new(1.0, 1.0, 0.0)).unwrap();
    let flipped = Frame3D::new(Point3D::ORIGIN, -Vector3D::Z, Vector3D::Y).unwrap();
    for (frame, same_direction) in [(rotated, true), (flipped, false)] {
        let b = Circle3D::from_frame(frame, 2.0).unwrap();
        let r = a.intersect(&b, TOL);
        assert_eq!(r.method(), IntersectionMethod::Exact);
        assert_eq!(r.overlaps().len(), 1, "{frame:?}");
        let o = r.overlaps()[0];
        assert_eq!(o.same_direction, same_direction, "{frame:?}");
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
    assert!((r.hits()[0].first_parameter.unwrap() - 7.0).abs() < 1e-12);
    // The trimmed arc stores its angle, which agrees with the circle's own.
    let angle = r.hits()[0].second_parameter.unwrap();
    assert!(angle.abs() < 1e-12);
    assert!((angle - circle.parameter_of(r.hits()[0].point)).abs() < 1e-12);
    assert_curve_hits_lie_on(&r, &line, &arc);
    // A segment that stops before the circle meets it once, at PI.
    let segment = TrimmedCurve3D::new(line, 0.0, 5.0).unwrap();
    let r = circle.intersect(&segment, TOL);
    assert_eq!(r.hits().len(), 1);
    // The full circle's angle is not computed; the segment's range test
    // computed the line parameter.
    let hit = r.hits()[0];
    assert_eq!(hit.first_parameter, None);
    assert!((circle.parameter_of(hit.point) - PI).abs() < 1e-12);
    assert!((hit.second_parameter.unwrap() - 3.0).abs() < 1e-12);
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

fn assert_hits_2d_lie_on<A: ParametricCurve2D, B: ParametricCurve2D>(
    r: &CurveCurveIntersection2D,
    first: &A,
    second: &B,
) {
    for h in r.hits() {
        if let Some(t) = h.first_parameter {
            assert!(first.eval_point(t).distance(h.point) <= 1e-7);
        }
        if let Some(t) = h.second_parameter {
            assert!(second.eval_point(t).distance(h.point) <= 1e-7);
        }
    }
}

#[test]
fn test_2d_line_circle_hits() {
    let line = Line2D::new(Point2D::new(-5.0, 0.0), Vector2D::new(1.0, 0.0)).unwrap();
    let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
    let r = circle.intersect(&line, TOL);
    assert_eq!(r.hits().len(), 2);
    assert_hits_2d_lie_on(&r, &circle, &line);
    // The line first reads the same hits from its own side.
    let from_line = line.intersect(&circle, TOL);
    assert_eq!(from_line.hits().len(), 2);
    assert_hits_2d_lie_on(&from_line, &line, &circle);
    assert!((from_line.hits()[0].first_parameter.unwrap() - 3.0).abs() < 1e-9);
}

#[test]
fn test_2d_trimmed_segment_misses() {
    let line = Line2D::new(Point2D::new(-5.0, 0.0), Vector2D::new(1.0, 0.0)).unwrap();
    let segment = TrimmedCurve2D::new(line, 0.0, 1.0).unwrap();
    let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
    assert!(segment.intersect(&circle, TOL).is_empty());
    assert!(circle.intersect(&segment, TOL).is_empty());
}

#[test]
fn test_2d_tangent_circles() {
    let a = Circle2D::new(Point2D::ORIGIN, 1.0).unwrap();
    let b = Circle2D::new(Point2D::new(2.0, 0.0), 1.0).unwrap();
    let r = a.intersect(&b, TOL);
    assert_eq!(r.hits().len(), 1);
    assert_eq!(r.hits()[0].multiplicity, 2);
    assert_hits_2d_lie_on(&r, &a, &b);
}

#[test]
fn test_2d_tangent_line_has_multiplicity_two() {
    let line = Line2D::new(Point2D::new(-5.0, 2.0), Vector2D::X).unwrap();
    let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
    let r = line.intersect(&circle, TOL);
    assert_eq!(r.hits().len(), 1);
    assert_eq!(r.hits()[0].multiplicity, 2);
}

#[test]
fn test_2d_lines_cross_and_parallels_are_empty() {
    let a = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    let b = Line2D::new(Point2D::new(1.0, -1.0), Vector2D::Y).unwrap();
    let r = a.intersect(&b, TOL);
    assert_eq!(r.hits().len(), 1);
    assert!((r.hits()[0].first_parameter.unwrap() - 1.0).abs() < 1e-12);
    assert!((r.hits()[0].second_parameter.unwrap() - 1.0).abs() < 1e-12);
    let parallel = Line2D::new(Point2D::new(0.0, 1.0), Vector2D::X).unwrap();
    let r = a.intersect(&parallel, TOL);
    assert!(r.is_empty() && r.is_complete());
}

#[test]
fn test_2d_routed_hits_equal_exact_hits() {
    let line = Line2D::new(Point2D::new(-5.0, 0.5), Vector2D::X).unwrap();
    let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
    let routed = Curve2D::from(line).intersect(&Curve2D::from(circle), TOL);
    let LineCircle2DIntersection::Points((t1, p1), (t2, p2)) = line.intersect_exact(&circle, TOL)
    else {
        panic!("expected two hits");
    };
    assert_eq!(routed.hits()[0].first_parameter, Some(t1));
    assert_eq!(routed.hits()[0].point, p1);
    assert_eq!(routed.hits()[1].first_parameter, Some(t2));
    assert_eq!(routed.hits()[1].point, p2);
}

#[test]
fn test_2d_trimmed_arc_keeps_only_its_hits() {
    let line = Line2D::new(Point2D::new(-5.0, 0.0), Vector2D::X).unwrap();
    let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
    // The arc across the seam keeps the hit at angle 0, not the one at PI.
    let arc = TrimmedCurve2D::new(circle, -1.0, 1.0).unwrap();
    let r = arc.intersect(&line, TOL);
    assert_eq!(r.hits().len(), 1);
    assert!(r.hits()[0].first_parameter.unwrap().abs() < 1e-12);
    assert!((r.hits()[0].second_parameter.unwrap() - 7.0).abs() < 1e-12);
}

#[test]
fn test_2d_coincident_segments_with_opposite_directions() {
    let a = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    let b = Line2D::new(Point2D::new(5.0, 0.0), -Vector2D::X).unwrap();
    let first = TrimmedCurve2D::new(a, 0.0, 4.0).unwrap();
    let second = TrimmedCurve2D::new(b, 0.0, 3.0).unwrap();
    let r = first.intersect(&second, TOL);
    assert_eq!(r.overlaps().len(), 1);
    assert!(close(r.overlaps()[0].first_range, (2.0, 4.0)));
    assert!(close(r.overlaps()[0].second_range, (1.0, 3.0)));
    assert!(!r.overlaps()[0].same_direction);
    let whole = a.intersect(&a, TOL);
    assert_eq!(whole.overlaps().len(), 1);
    assert_eq!(
        whole.overlaps()[0].first_range,
        (f64::NEG_INFINITY, f64::INFINITY)
    );
    assert!(whole.overlaps()[0].same_direction);
}

#[test]
fn test_2d_coincident_arcs_overlap_on_both_sides_of_the_seam() {
    let circle = Circle2D::new(Point2D::ORIGIN, 1.0).unwrap();
    let a = TrimmedCurve2D::new(circle, 0.0, 5.0).unwrap();
    let b = TrimmedCurve2D::new(circle, 4.0, 7.0).unwrap();
    let r = a.intersect(&b, TOL);
    assert_eq!(r.overlaps().len(), 2);
    let o = r.overlaps();
    assert!(close(o[0].first_range, (0.0, 7.0 - TAU)));
    assert!(close(o[0].second_range, (TAU, 7.0)));
    assert!(close(o[1].first_range, (4.0, 5.0)));
    assert!(close(o[1].second_range, (4.0, 5.0)));
    assert!(o.iter().all(|piece| piece.same_direction));
}

#[test]
fn test_2d_coincident_circles_in_opposite_directions() {
    let a = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
    let frame = Frame2D::new(Point2D::ORIGIN, Vector2D::X, Vector2D::new(0.0, -1.0)).unwrap();
    let b = Circle2D::from_frame(frame, 2.0).unwrap();
    let r = a.intersect(&b, TOL);
    assert_eq!(r.overlaps().len(), 1);
    let o = r.overlaps()[0];
    assert!(!o.same_direction);
    assert!(close(o.first_range, a.bounds()));
    assert!((o.second_range.1 - o.second_range.0 - TAU).abs() < 1e-9);
}

#[test]
fn test_consecutive_segments_meet_at_their_shared_end() {
    let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    let a = TrimmedCurve3D::new(line, 0.0, 1.0).unwrap();
    let b = TrimmedCurve3D::new(line, 1.0, 2.0).unwrap();
    let r = a.intersect(&b, TOL);
    assert_eq!(r.method(), IntersectionMethod::Exact);
    assert!(r.overlaps().is_empty());
    assert_eq!(r.hits().len(), 1);
    let hit = r.hits()[0];
    assert!((hit.first_parameter.unwrap() - 1.0).abs() < 1e-12);
    assert!((hit.second_parameter.unwrap() - 1.0).abs() < 1e-12);
    assert_eq!(hit.multiplicity, 1);
    assert!(hit.point.distance(Point3D::new(1.0, 0.0, 0.0)) < 1e-12);
}

#[test]
fn test_consecutive_arcs_meet_at_their_shared_end() {
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    let a = TrimmedCurve3D::new(circle, 0.0, PI / 2.0).unwrap();
    let b = TrimmedCurve3D::new(circle, PI / 2.0, PI).unwrap();
    let r = a.intersect(&b, TOL);
    assert!(r.overlaps().is_empty());
    assert_eq!(r.hits().len(), 1);
    let hit = r.hits()[0];
    assert!((hit.first_parameter.unwrap() - PI / 2.0).abs() < 1e-12);
    assert!((hit.second_parameter.unwrap() - PI / 2.0).abs() < 1e-12);
    assert!(hit.point.distance(Point3D::new(0.0, 1.0, 0.0)) < 1e-12);
}

#[test]
fn test_consecutive_arcs_meet_at_the_seam() {
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    let a = TrimmedCurve3D::new(circle, 1.5 * PI, TAU).unwrap();
    let b = TrimmedCurve3D::new(circle, 0.0, PI / 2.0).unwrap();
    let r = a.intersect(&b, TOL);
    assert!(r.overlaps().is_empty());
    assert_eq!(r.hits().len(), 1);
    let hit = r.hits()[0];
    assert!((hit.first_parameter.unwrap() - TAU).abs() < 1e-12);
    assert!(hit.second_parameter.unwrap().abs() < 1e-12);
    assert!(hit.point.distance(Point3D::new(1.0, 0.0, 0.0)) < 1e-12);
    // Read from the other side, the seam is the start of the receiver.
    let reverse = b.intersect(&a, TOL);
    assert_eq!(reverse.hits().len(), 1);
    assert!(reverse.hits()[0].first_parameter.unwrap().abs() < 1e-12);
    assert!((reverse.hits()[0].second_parameter.unwrap() - TAU).abs() < 1e-12);
}

#[test]
fn test_arcs_overlapping_on_a_span_report_no_extra_hit() {
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    let a = TrimmedCurve3D::new(circle, 0.0, PI).unwrap();
    let b = TrimmedCurve3D::new(circle, PI / 2.0, 1.5 * PI).unwrap();
    let r = a.intersect(&b, TOL);
    assert!(r.hits().is_empty());
    assert_eq!(r.overlaps().len(), 1);
    assert!(close(r.overlaps()[0].first_range, (PI / 2.0, PI)));
    // An arc against its full circle overlaps it without a hit at the seam.
    let arc = TrimmedCurve3D::new(circle, 0.0, 1.0).unwrap();
    let r = arc.intersect(&circle, TOL);
    assert!(r.hits().is_empty());
    assert_eq!(r.overlaps().len(), 1);
}

#[test]
fn test_2d_consecutive_segments_meet_at_their_shared_end() {
    let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    let a = TrimmedCurve2D::new(line, 0.0, 1.0).unwrap();
    let b = TrimmedCurve2D::new(line, 1.0, 2.0).unwrap();
    let r = a.intersect(&b, TOL);
    assert!(r.overlaps().is_empty());
    assert_eq!(r.hits().len(), 1);
    let hit = r.hits()[0];
    assert!((hit.first_parameter.unwrap() - 1.0).abs() < 1e-12);
    assert!((hit.second_parameter.unwrap() - 1.0).abs() < 1e-12);
    assert_eq!(hit.multiplicity, 1);
    assert!(hit.point.distance(Point2D::new(1.0, 0.0)) < 1e-12);
}

#[test]
fn test_2d_consecutive_arcs_meet_at_their_shared_end() {
    let circle = Circle2D::new(Point2D::ORIGIN, 1.0).unwrap();
    let a = TrimmedCurve2D::new(circle, 0.0, PI / 2.0).unwrap();
    let b = TrimmedCurve2D::new(circle, PI / 2.0, PI).unwrap();
    let r = a.intersect(&b, TOL);
    assert!(r.overlaps().is_empty());
    assert_eq!(r.hits().len(), 1);
    let hit = r.hits()[0];
    assert!((hit.first_parameter.unwrap() - PI / 2.0).abs() < 1e-12);
    assert!((hit.second_parameter.unwrap() - PI / 2.0).abs() < 1e-12);
    assert!(hit.point.distance(Point2D::new(0.0, 1.0)) < 1e-12);
}

#[test]
fn test_2d_arcs_overlapping_on_a_span_report_no_extra_hit() {
    let circle = Circle2D::new(Point2D::ORIGIN, 1.0).unwrap();
    let a = TrimmedCurve2D::new(circle, 0.0, PI).unwrap();
    let b = TrimmedCurve2D::new(circle, PI / 2.0, 1.5 * PI).unwrap();
    let r = a.intersect(&b, TOL);
    assert!(r.hits().is_empty());
    assert_eq!(r.overlaps().len(), 1);
}

#[test]
fn test_hit_just_below_the_seam_leaves_the_angle_to_the_caller() {
    // The line crosses the circle a hair below angle 0. The circle's angle
    // is not computed: the caller reads it from the point, near the seam.
    let near_seam = |angle: f64| angle.min(TAU - angle) < 1e-9;
    let circle = Circle2D::new(Point2D::ORIGIN, 1.0).unwrap();
    let line = Line2D::new(Point2D::new(-5.0, -1e-11), Vector2D::X).unwrap();
    let r = circle.intersect(&line, TOL);
    assert_eq!(r.hits().len(), 2);
    assert!(r.hits().iter().all(|h| h.first_parameter.is_none()));
    let seam_hit = r.hits().iter().find(|h| h.point.x > 0.0).unwrap();
    assert!(near_seam(circle.parameter_of(seam_hit.point)));
    let circle_3d = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    let line_3d = Line3D::new(Point3D::new(-5.0, -1e-11, 0.0), Vector3D::X).unwrap();
    let r_3d = circle_3d.intersect(&line_3d, TOL);
    assert!(r_3d.hits().iter().all(|h| h.first_parameter.is_none()));
    // Two circles meeting a hair below the seam of the first.
    let other = Circle2D::new(Point2D::new(2.0, -2e-11), 1.0).unwrap();
    let r = circle.intersect(&other, TOL);
    assert_eq!(r.hits().len(), 1);
    assert_eq!(r.hits()[0].first_parameter, None);
    assert!(near_seam(circle.parameter_of(r.hits()[0].point)));
}

#[test]
fn test_concrete_pairs_route_like_their_enums_bit_for_bit() {
    let line = Line3D::new(Point3D::new(-5.0, 0.3, 0.2), Vector3D::new(1.0, 0.1, 0.4)).unwrap();
    let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::new(0.1, 0.0, 1.0), 1.5).unwrap();
    let typed = line.intersect(&cylinder, TOL);
    assert_eq!(typed.hits().len(), 2);
    assert_eq!(
        typed,
        Curve3D::from(line).intersect(&Surface::from(cylinder), TOL)
    );
    let segment = TrimmedCurve3D::new(line, 0.0, 5.0).unwrap();
    assert_eq!(
        segment.intersect(&cylinder, TOL),
        TrimmedCurve3D::new(Curve3D::from(line), 0.0, 5.0)
            .unwrap()
            .intersect(&Surface::from(cylinder), TOL)
    );

    let first = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    let second = Circle3D::new(Point3D::new(1.5, 0.0, 0.0), Vector3D::Z, 1.0).unwrap();
    let typed = first.intersect(&second, TOL);
    assert_eq!(typed.hits().len(), 2);
    assert_eq!(
        typed,
        Curve3D::from(first).intersect(&Curve3D::from(second), TOL)
    );

    let first = Circle2D::new(Point2D::new(0.0, 0.0), 2.0).unwrap();
    let second = Circle2D::new(Point2D::new(1.5, 0.5), 1.0).unwrap();
    let typed = first.intersect(&second, TOL);
    assert_eq!(typed.hits().len(), 2);
    assert_eq!(
        typed,
        Curve2D::from(first).intersect(&Curve2D::from(second), TOL)
    );
}

/// A curve type defined outside the crate: a helix of radius 2 about the
/// z axis, rising 1 per turn, over three turns.
struct Helix;

impl ParametricCurve3D for Helix {
    fn eval_point(&self, u: f64) -> Point3D {
        Point3D::new(2.0 * u.cos(), 2.0 * u.sin(), u / TAU)
    }
    fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        match order % 4 {
            1 => Vector3D::new(-2.0 * u.sin(), 2.0 * u.cos(), 1.0 / TAU),
            2 => Vector3D::new(-2.0 * u.cos(), -2.0 * u.sin(), 0.0),
            3 => Vector3D::new(2.0 * u.sin(), -2.0 * u.cos(), 0.0),
            _ => Vector3D::new(2.0 * u.cos(), 2.0 * u.sin(), 0.0),
        }
    }
    fn bounds(&self) -> (f64, f64) {
        (0.0, 3.0 * TAU)
    }
    fn period(&self) -> Option<f64> {
        None
    }
}

#[test]
fn test_caller_defined_curve_intersects_through_the_public_numeric_functions() {
    // A slanted plane through the axis at height 1.5 crosses each turn.
    let plane = Plane::new(Point3D::new(0.0, 0.0, 1.5), Vector3D::new(0.05, 0.0, 1.0)).unwrap();
    let r = intersect_curve_surface(&Helix, &plane, TOL);
    assert_eq!(r.method(), IntersectionMethod::Numeric);
    assert!(r.overlaps().is_empty());
    assert_eq!(r.hits().len(), 1);
    assert_hits_lie_on(&r, &Helix, &plane);

    // A vertical line on the helix's cylinder meets it once per turn.
    let line = Line3D::new(Point3D::new(0.0, 2.0, 0.0), Vector3D::Z).unwrap();
    let r = intersect_curve_curve(&Helix, &line, TOL);
    assert_eq!(r.method(), IntersectionMethod::Numeric);
    let parameters: Vec<f64> = r
        .hits()
        .iter()
        .map(|h| h.first_parameter.unwrap())
        .collect();
    assert_eq!(parameters.len(), 3);
    for (turn, parameter) in parameters.iter().enumerate() {
        assert!((parameter - (PI / 2.0 + turn as f64 * TAU)).abs() < 1e-9);
    }
}

#[test]
fn test_segments_apart_by_less_than_the_confusion_touch() {
    // A unit-speed line: a parameter gap of 5e-9 is 5e-9 in space.
    let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    let a = TrimmedCurve3D::new(line, 0.0, 1.0).unwrap();
    let b = TrimmedCurve3D::new(line, 1.0 + 5e-9, 2.0).unwrap();
    let r = a.intersect(&b, TOL);
    assert!(r.overlaps().is_empty());
    assert_eq!(r.hits().len(), 1);
    assert_eq!(r.hits()[0].multiplicity, 1);
    assert!(r.hits()[0].point.distance(Point3D::new(1.0, 0.0, 0.0)) < 1e-8);
}

#[test]
fn test_segments_apart_by_more_than_the_confusion_do_not_meet() {
    let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    let a = TrimmedCurve3D::new(line, 0.0, 1.0).unwrap();
    let b = TrimmedCurve3D::new(line, 1.0 + 1e-3, 2.0).unwrap();
    let r = a.intersect(&b, TOL);
    assert!(r.is_empty());
    assert!(r.is_complete());
}

#[test]
fn test_2d_consecutive_arcs_meet_at_the_seam() {
    let circle = Circle2D::new(Point2D::ORIGIN, 1.0).unwrap();
    let a = TrimmedCurve2D::new(circle, 1.5 * PI, TAU).unwrap();
    let b = TrimmedCurve2D::new(circle, 0.0, PI / 2.0).unwrap();
    let r = a.intersect(&b, TOL);
    assert!(r.overlaps().is_empty());
    assert_eq!(r.hits().len(), 1);
    let hit = r.hits()[0];
    assert!((hit.first_parameter.unwrap() - TAU).abs() < 1e-12);
    assert!(hit.second_parameter.unwrap().abs() < 1e-12);
    assert!(hit.point.distance(Point2D::new(1.0, 0.0)) < 1e-12);
}

#[test]
fn test_touch_between_coincident_curves_running_opposite_ways() {
    // The reversed circle has the point at angle a at parameter -a.
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    let reversed = Circle3D::from_frame(
        Frame3D::new(Point3D::ORIGIN, -Vector3D::Z, Vector3D::X).unwrap(),
        1.0,
    )
    .unwrap();
    let a = TrimmedCurve3D::new(circle, 0.0, PI / 2.0).unwrap();
    // Angles [PI / 2, PI] on the reversed circle.
    let b = TrimmedCurve3D::new(reversed, PI, 1.5 * PI).unwrap();
    let r = a.intersect(&b, TOL);
    assert!(r.overlaps().is_empty());
    assert_eq!(r.hits().len(), 1);
    let hit = r.hits()[0];
    assert!((hit.first_parameter.unwrap() - PI / 2.0).abs() < 1e-12);
    assert!((hit.second_parameter.unwrap() - 1.5 * PI).abs() < 1e-12);
    assert!(hit.point.distance(Point3D::new(0.0, 1.0, 0.0)) < 1e-12);

    // A segment of the line running the other way: x in [1, 2].
    let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    let backwards = Line3D::new(Point3D::ORIGIN, -Vector3D::X).unwrap();
    let a = TrimmedCurve3D::new(line, 0.0, 1.0).unwrap();
    let b = TrimmedCurve3D::new(backwards, -2.0, -1.0).unwrap();
    let r = a.intersect(&b, TOL);
    assert!(r.overlaps().is_empty());
    assert_eq!(r.hits().len(), 1);
    assert!((r.hits()[0].first_parameter.unwrap() - 1.0).abs() < 1e-12);
    assert!((r.hits()[0].second_parameter.unwrap() + 1.0).abs() < 1e-12);
}

#[test]
fn test_arcs_making_up_the_whole_circle_touch_twice() {
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    let a = TrimmedCurve3D::new(circle, 0.0, PI).unwrap();
    let b = TrimmedCurve3D::new(circle, PI, TAU).unwrap();
    let r = a.intersect(&b, TOL);
    assert!(r.overlaps().is_empty());
    let pairs: Vec<(f64, f64)> = r
        .hits()
        .iter()
        .map(|h| (h.first_parameter.unwrap(), h.second_parameter.unwrap()))
        .collect();
    assert_eq!(pairs.len(), 2);
    assert!(close(pairs[0], (0.0, TAU)));
    assert!(close(pairs[1], (PI, PI)));

    let circle = Circle2D::new(Point2D::ORIGIN, 1.0).unwrap();
    let a = TrimmedCurve2D::new(circle, 0.0, PI).unwrap();
    let b = TrimmedCurve2D::new(circle, PI, TAU).unwrap();
    let r = a.intersect(&b, TOL);
    assert!(r.overlaps().is_empty());
    assert_eq!(r.hits().len(), 2);
}

#[test]
fn test_routed_circle_circle_leaves_both_angles_to_the_caller() {
    let first = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    let second = Circle3D::new(Point3D::new(2.0, 0.0, 0.0), Vector3D::Z, 2.0).unwrap();
    let r = first.intersect(&second, TOL);
    assert_eq!(r.method(), IntersectionMethod::Exact);
    assert_eq!(r.hits().len(), 2);
    for h in r.hits() {
        assert_eq!((h.first_parameter, h.second_parameter), (None, None));
        let (a, b) = (first.parameter_of(h.point), second.parameter_of(h.point));
        assert!(first.eval_point(a).distance(h.point) < 1e-12);
        assert!(second.eval_point(b).distance(h.point) < 1e-12);
    }
}

#[test]
fn test_trimmed_operands_store_their_parameter_within_range() {
    let first = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    let second = Circle3D::new(Point3D::new(2.0, 0.0, 0.0), Vector3D::Z, 2.0).unwrap();
    // An arc across the seam of the first circle holds both hits.
    let arc = TrimmedCurve3D::new(first, -PI / 2.0, PI / 2.0).unwrap();
    let r = arc.intersect(&second, TOL);
    assert_eq!(r.hits().len(), 2);
    for h in r.hits() {
        let angle = h.first_parameter.unwrap();
        assert!((-PI / 2.0..=PI / 2.0).contains(&angle));
        // The stored angle is the circle's own, up to a whole turn.
        let own = first.parameter_of(h.point);
        assert!(((angle - own).rem_euclid(TAU) + 1e-12) % TAU < 1e-9);
        // The untrimmed side is left to the caller.
        assert_eq!(h.second_parameter, None);
    }
    // Every hit has a first parameter, so the hits are sorted by it.
    assert!(r.hits()[0].first_parameter < r.hits()[1].first_parameter);
}

#[test]
fn test_numeric_path_stores_every_parameter() {
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    let other = Circle3D::new(Point3D::new(2.0, 0.0, 0.0), Vector3D::Z, 2.0).unwrap();
    let r = circle.intersect(&other.to_bspline(), TOL);
    assert_eq!(r.method(), IntersectionMethod::Numeric);
    assert_eq!(r.hits().len(), 2);
    assert!(
        r.hits()
            .iter()
            .all(|h| h.first_parameter.is_some() && h.second_parameter.is_some())
    );
    assert_curve_hits_lie_on(&r, &circle, &other.to_bspline());
    let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 3.0, 1.0).unwrap();
    let ellipse = Ellipse3D::from_frame(Frame3D::WORLD, 5.0, 1.0).unwrap();
    let r = ellipse.intersect(&torus, TOL);
    assert!(
        r.hits()
            .iter()
            .all(|h| h.curve_parameter.is_some() && h.surface_parameters.is_some())
    );
}

/// A hit within `tol.confusion` in space of the seam, just outside an arc
/// that starts or ends there, on a unit circle in the xy plane.
struct SeamCase {
    /// The arc's range.
    range: (f64, f64),
    /// The angle of the hit: a hair past the arc's seam end.
    angle: f64,
    /// The direction angle of the line through the hit; its second hit on
    /// the circle falls outside the arc.
    line_angle: f64,
    /// The angle of the centre of a unit circle through the hit; its
    /// second hit falls outside the arc.
    crossing_angle: f64,
    /// The arc parameter the hit keeps: the seam's representative.
    expected: f64,
}

const SEAM_CASES: [SeamCase; 2] = [
    SeamCase {
        range: (0.0, PI),
        angle: -5e-9,
        line_angle: PI / 3.0,
        crossing_angle: -PI / 3.0,
        expected: 0.0,
    },
    SeamCase {
        range: (PI, TAU),
        angle: 5e-9,
        line_angle: 2.0 * PI / 3.0,
        crossing_angle: PI / 3.0,
        expected: TAU,
    },
];

fn assert_seam_parameter(parameter: Option<f64>, case: &SeamCase) {
    let parameter = parameter.expect("a trimmed side stores its parameter");
    assert!(
        (parameter - case.expected).abs() < 1e-12,
        "parameter {parameter}, expected {}",
        case.expected
    );
}

#[test]
fn test_hit_a_hair_past_the_seam_end_of_an_arc_is_kept_in_3d() {
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    for case in &SEAM_CASES {
        let arc = TrimmedCurve3D::new(circle, case.range.0, case.range.1).unwrap();
        let hit = circle.eval_point(case.angle);
        let (sin, cos) = case.line_angle.sin_cos();
        let line = Line3D::new(hit, Vector3D::new(cos, sin, 0.0)).unwrap();
        let (sin, cos) = (case.crossing_angle + case.angle).sin_cos();
        let crossing = Circle3D::new(Point3D::new(cos, sin, 0.0), Vector3D::Z, 1.0).unwrap();
        let (sin, cos) = case.line_angle.sin_cos();
        let plane = Plane::new(hit, Vector3D::new(-sin, cos, 0.0)).unwrap();

        let r = arc.intersect(&line, TOL);
        assert_eq!(r.hits().len(), 1, "arc with line, range {:?}", case.range);
        assert_seam_parameter(r.hits()[0].first_parameter, case);
        let r = line.intersect(&arc, TOL);
        assert_eq!(r.hits().len(), 1, "line with arc, range {:?}", case.range);
        assert_seam_parameter(r.hits()[0].second_parameter, case);
        let r = arc.intersect(&crossing, TOL);
        assert_eq!(r.hits().len(), 1, "arc with circle, range {:?}", case.range);
        assert_seam_parameter(r.hits()[0].first_parameter, case);
        let r = crossing.intersect(&arc, TOL);
        assert_eq!(r.hits().len(), 1, "circle with arc, range {:?}", case.range);
        assert_seam_parameter(r.hits()[0].second_parameter, case);
        let r = arc.intersect(&plane, TOL);
        assert_eq!(r.hits().len(), 1, "arc with plane, range {:?}", case.range);
        assert_seam_parameter(r.hits()[0].curve_parameter, case);
    }
}

#[test]
fn test_hit_a_hair_past_the_seam_end_of_an_arc_is_kept_in_2d() {
    let circle = Circle2D::new(Point2D::ORIGIN, 1.0).unwrap();
    for case in &SEAM_CASES {
        let arc = TrimmedCurve2D::new(circle, case.range.0, case.range.1).unwrap();
        let hit = circle.eval_point(case.angle);
        let (sin, cos) = case.line_angle.sin_cos();
        let line = Line2D::new(hit, Vector2D::new(cos, sin)).unwrap();
        let (sin, cos) = (case.crossing_angle + case.angle).sin_cos();
        let crossing = Circle2D::new(Point2D::new(cos, sin), 1.0).unwrap();

        let r = arc.intersect(&line, TOL);
        assert_eq!(r.hits().len(), 1, "arc with line, range {:?}", case.range);
        assert_seam_parameter(r.hits()[0].first_parameter, case);
        let r = line.intersect(&arc, TOL);
        assert_eq!(r.hits().len(), 1, "line with arc, range {:?}", case.range);
        assert_seam_parameter(r.hits()[0].second_parameter, case);
        let r = arc.intersect(&crossing, TOL);
        assert_eq!(r.hits().len(), 1, "arc with circle, range {:?}", case.range);
        assert_seam_parameter(r.hits()[0].first_parameter, case);
        let r = crossing.intersect(&arc, TOL);
        assert_eq!(r.hits().len(), 1, "circle with arc, range {:?}", case.range);
        assert_seam_parameter(r.hits()[0].second_parameter, case);
    }
}

#[test]
fn test_hit_farther_than_the_confusion_past_the_seam_is_dropped() {
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    let arc = TrimmedCurve3D::new(circle, 0.0, PI).unwrap();
    let hit = circle.eval_point(-1e-6);
    let (sin, cos) = (PI / 3.0).sin_cos();
    let line = Line3D::new(hit, Vector3D::new(cos, sin, 0.0)).unwrap();
    assert!(arc.intersect(&line, TOL).hits().is_empty());
}

/// Two arcs of one circle whose ends meet at the seam, `gap` apart in
/// angle: `[gap, PI/2]` and `[3PI/2, 2PI]`, then the second shifted one
/// turn up against `[PI, 2PI]`.
fn seam_touch_ranges(gap: f64) -> [((f64, f64), (f64, f64)); 2] {
    [
        ((gap, PI / 2.0), (3.0 * PI / 2.0, TAU)),
        ((TAU + gap, TAU + 1.0), (PI, TAU)),
    ]
}

fn assert_touch_parameters(
    first: Option<f64>,
    second: Option<f64>,
    eval: (Point3D, Point3D),
    point: Point3D,
    ranges: ((f64, f64), (f64, f64)),
) {
    let (first, second) = (first.unwrap(), second.unwrap());
    let within = |p: f64, (lo, hi): (f64, f64)| p >= lo - 1e-9 && p <= hi + 1e-9;
    assert!(
        within(first, ranges.0),
        "first {first} outside {:?}",
        ranges.0
    );
    assert!(
        within(second, ranges.1),
        "second {second} outside {:?}",
        ranges.1
    );
    assert!(
        eval.0.distance(point) <= 1e-7,
        "first parameter is not the point"
    );
    assert!(
        eval.1.distance(point) <= 1e-7,
        "second parameter is not the point"
    );
}

#[test]
fn test_arcs_touching_across_the_seam_store_the_near_ends() {
    for radius in [1.0, 0.5, 0.01] {
        for gap_in_space in [0.0, 1e-10, 5e-9, 5e-8, 9e-8] {
            let gap = gap_in_space / radius;
            for (range_a, range_b) in seam_touch_ranges(gap) {
                let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, radius).unwrap();
                let a = TrimmedCurve3D::new(circle, range_a.0, range_a.1).unwrap();
                let b = TrimmedCurve3D::new(circle, range_b.0, range_b.1).unwrap();
                for (first, second, ranges) in
                    [(&a, &b, (range_a, range_b)), (&b, &a, (range_b, range_a))]
                {
                    let r = first.intersect(second, TOL);
                    assert!(r.overlaps().is_empty());
                    assert_eq!(r.hits().len(), 1, "radius {radius}, gap {gap}");
                    let h = r.hits()[0];
                    assert_touch_parameters(
                        h.first_parameter,
                        h.second_parameter,
                        (
                            circle.eval_point(h.first_parameter.unwrap()),
                            circle.eval_point(h.second_parameter.unwrap()),
                        ),
                        h.point,
                        ranges,
                    );
                }
                let circle = Circle2D::new(Point2D::ORIGIN, radius).unwrap();
                let a = TrimmedCurve2D::new(circle, range_a.0, range_a.1).unwrap();
                let b = TrimmedCurve2D::new(circle, range_b.0, range_b.1).unwrap();
                let lift = |p: Point2D| Point3D::new(p.x, p.y, 0.0);
                for (first, second, ranges) in
                    [(&a, &b, (range_a, range_b)), (&b, &a, (range_b, range_a))]
                {
                    let r = first.intersect(second, TOL);
                    assert!(r.overlaps().is_empty());
                    assert_eq!(r.hits().len(), 1, "2D radius {radius}, gap {gap}");
                    let h = r.hits()[0];
                    assert_touch_parameters(
                        h.first_parameter,
                        h.second_parameter,
                        (
                            lift(circle.eval_point(h.first_parameter.unwrap())),
                            lift(circle.eval_point(h.second_parameter.unwrap())),
                        ),
                        lift(h.point),
                        ranges,
                    );
                }
            }
        }
    }
}

#[test]
fn test_reviewer_seam_touch_stores_two_pi_on_the_second_arc() {
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    let a = TrimmedCurve3D::new(circle, 5e-8, PI / 2.0).unwrap();
    let b = TrimmedCurve3D::new(circle, 3.0 * PI / 2.0, TAU).unwrap();
    let r = a.intersect(&b, TOL);
    assert_eq!(r.hits().len(), 1);
    assert!((r.hits()[0].second_parameter.unwrap() - TAU).abs() < 1e-12);
}

/// A caller-defined surface: a plane the crate does not know.
struct Floor(Plane);

impl ParametricSurface for Floor {
    fn eval_point(&self, u: f64, v: f64) -> Point3D {
        self.0.eval_point(u, v)
    }
    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3D {
        self.0.eval_derivative(u, v, du, dv)
    }
    fn u_bounds(&self) -> (f64, f64) {
        self.0.u_bounds()
    }
    fn v_bounds(&self) -> (f64, f64) {
        self.0.v_bounds()
    }
    fn u_period(&self) -> Option<f64> {
        None
    }
    fn v_period(&self) -> Option<f64> {
        None
    }
    fn project_point(&self, point: Point3D, tol: Tolerance) -> SurfaceProjection {
        ParametricSurface::project_point(&self.0, point, tol)
    }
    fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        ParametricSurface::contains(&self.0, point, tol)
    }
}

#[test]
fn test_caller_defined_surface_intersects_through_the_public_numeric_function() {
    let floor = Floor(Plane::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Z).unwrap());
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::X, 2.0).unwrap();
    let r = intersect_curve_surface(&circle, &floor, TOL);
    assert_eq!(r.method(), IntersectionMethod::Numeric);
    assert!(r.overlaps().is_empty());
    assert_eq!(r.hits().len(), 2);
    assert_hits_lie_on(&r, &circle, &floor);
    for h in r.hits() {
        assert!((h.point.z - 1.0).abs() < 1e-9);
    }
}

/// Hits of the arc `[0, PI]` of a circle of `radius` in the xy plane with
/// a line, a plane and (2D) a line, all through the point at angle
/// `PI + excess`, a hair past the arc's end; their other hit on the circle
/// falls outside the arc. Returns the stored arc parameters, `None` for a
/// dropped hit.
fn hits_past_the_end_of_an_arc(radius: f64, excess: f64) -> Vec<Option<f64>> {
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, radius).unwrap();
    let arc = TrimmedCurve3D::new(circle, 0.0, PI).unwrap();
    let hit = circle.eval_point(PI + excess);
    let (sin, cos) = (-PI / 3.0).sin_cos();
    let line = Line3D::new(hit, Vector3D::new(cos, sin, 0.0)).unwrap();
    let plane = Plane::new(hit, Vector3D::new(-sin, cos, 0.0)).unwrap();
    let circle_2d = Circle2D::new(Point2D::ORIGIN, radius).unwrap();
    let arc_2d = TrimmedCurve2D::new(circle_2d, 0.0, PI).unwrap();
    let line_2d = Line2D::new(Point2D::new(hit.x, hit.y), Vector2D::new(cos, sin)).unwrap();
    let only = |parameters: Vec<Option<f64>>| {
        assert!(parameters.len() <= 1, "radius {radius}, excess {excess}");
        parameters.first().copied().flatten()
    };
    vec![
        only(
            arc.intersect(&line, TOL)
                .hits()
                .iter()
                .map(|h| h.first_parameter)
                .collect(),
        ),
        only(
            line.intersect(&arc, TOL)
                .hits()
                .iter()
                .map(|h| h.second_parameter)
                .collect(),
        ),
        only(
            arc.intersect(&plane, TOL)
                .hits()
                .iter()
                .map(|h| h.curve_parameter)
                .collect(),
        ),
        only(
            arc_2d
                .intersect(&line_2d, TOL)
                .hits()
                .iter()
                .map(|h| h.first_parameter)
                .collect(),
        ),
        only(
            line_2d
                .intersect(&arc_2d, TOL)
                .hits()
                .iter()
                .map(|h| h.second_parameter)
                .collect(),
        ),
    ]
}

#[test]
fn test_hit_a_hair_past_a_plain_end_of_an_arc_is_kept_at_the_end() {
    for parameter in hits_past_the_end_of_an_arc(1.0, 5e-9) {
        assert_eq!(parameter, Some(PI));
    }
    for parameter in hits_past_the_end_of_an_arc(1.0, 1e-6) {
        assert_eq!(parameter, None);
    }
}

#[test]
fn test_excess_past_a_range_end_is_measured_in_space() {
    // 5e-8 in space is kept, 2e-7 is not, whatever the radius.
    for radius in [10.0, 0.01] {
        for parameter in hits_past_the_end_of_an_arc(radius, 5e-8 / radius) {
            assert_eq!(parameter, Some(PI), "radius {radius}");
        }
        for parameter in hits_past_the_end_of_an_arc(radius, 2e-7 / radius) {
            assert_eq!(parameter, None, "radius {radius}");
        }
    }
    // A segment: 5e-8 past its end in space (fifty times `tol.parametric`)
    // is kept at the end; 2e-7 is not.
    let segment =
        TrimmedCurve3D::new(Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap(), 0.0, 1.0).unwrap();
    for (excess, expected) in [(5e-8, Some(1.0)), (2e-7, None)] {
        let crossing = Line3D::new(Point3D::new(1.0 + excess, -1.0, 0.0), Vector3D::Y).unwrap();
        let r = segment.intersect(&crossing, TOL);
        assert_eq!(
            r.hits().first().map(|h| h.first_parameter.unwrap()),
            expected
        );
        let r = crossing.intersect(&segment, TOL);
        assert_eq!(
            r.hits().first().map(|h| h.second_parameter.unwrap()),
            expected
        );
        let wall = Plane::new(Point3D::new(1.0 + excess, 0.0, 0.0), Vector3D::X).unwrap();
        let r = segment.intersect(&wall, TOL);
        assert_eq!(
            r.hits().first().map(|h| h.curve_parameter.unwrap()),
            expected
        );
    }
}
