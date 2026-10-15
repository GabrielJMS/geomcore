//! The routed intersection: exact when a closed form exists, numeric
//! otherwise, and the result says which method ran.

use geomcore::curves::{Curve3D, TrimmedCurve3D};
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
