//! The routed surface-surface intersection: exact when a closed form
//! exists, marching otherwise, and the result says which method ran.

use geomcore::curves::Curve3D;
use geomcore::surfaces::Surface;
use geomcore::*;

const TOL: Tolerance = Tolerance::DEFAULT;

/// Every curve, sampled at 50 parameters across its range, lies on both
/// surfaces within `limit`.
fn assert_curves_lie_on<A: ParametricSurface, B: ParametricSurface>(
    result: &SurfaceSurfaceIntersection,
    a: &A,
    b: &B,
    limit: f64,
) {
    for piece in result.curves() {
        let (start, end) = piece.curve.bounds();
        for i in 0..50 {
            let parameter = start + (end - start) * i as f64 / 49.0;
            let point = piece.curve.eval_point(parameter);
            assert!(a.project_point(point, TOL).distance <= limit);
            assert!(b.project_point(point, TOL).distance <= limit);
        }
    }
}

#[test]
fn test_plane_sphere_routes_exact_with_the_same_circle() {
    let plane = Plane::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Z).unwrap();
    let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    let routed = plane.intersect(&sphere, TOL);
    assert_eq!(routed.method(), IntersectionMethod::Exact);
    assert!(routed.is_complete());
    let ExactSurfaceIntersection::Meets { curves, points } =
        ElementarySurface::from(plane).intersect_exact(&ElementarySurface::from(sphere), TOL)
    else {
        panic!("expected a circle");
    };
    assert_eq!(
        (routed.curves().len(), curves.len(), points.len()),
        (1, 1, 0)
    );
    let (circle, tangent) = curves[0];
    assert_eq!(routed.curves()[0].curve, Curve3D::from(circle));
    assert_eq!(routed.curves()[0].tangent, tangent);
}

#[test]
fn test_both_argument_orders_give_the_same_curves() {
    let plane = Plane::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Z).unwrap();
    let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    assert_eq!(
        plane.intersect(&sphere, TOL).curves(),
        sphere.intersect(&plane, TOL).curves()
    );
    let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    assert_eq!(
        cylinder.intersect(&sphere, TOL).curves(),
        sphere.intersect(&cylinder, TOL).curves()
    );
    let (surface, other) = (Surface::from(plane), Surface::from(sphere));
    assert_eq!(
        surface.intersect(&other, TOL).curves(),
        other.intersect(&surface, TOL).curves()
    );
    assert_eq!(
        surface.intersect(&sphere, TOL),
        plane.intersect(&sphere, TOL)
    );
}

#[test]
fn test_sphere_cylinder_off_axis_routes_numeric() {
    let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    let cylinder = Cylinder::new(Point3D::new(0.7, 0.0, 0.0), Vector3D::Z, 1.0).unwrap();
    let result = sphere.intersect(&cylinder, TOL);
    assert_eq!(result.method(), IntersectionMethod::Numeric);
    assert_eq!(result.completeness(), &Completeness::Sampled);
    assert!(!result.curves().is_empty());
    assert!(result.curves().iter().all(|piece| !piece.tangent));
    assert_curves_lie_on(&result, &sphere, &cylinder, 1e-4);
}

#[test]
fn test_bspline_sphere_and_plane_route_numeric() {
    let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap().to_bspline();
    let plane = Plane::new(Point3D::new(0.0, 0.0, 0.5), Vector3D::Z).unwrap();
    let result = sphere.intersect(&plane, TOL);
    assert_eq!(result.method(), IntersectionMethod::Numeric);
    assert_eq!(result.curves().len(), 1);
    assert_curves_lie_on(&result, &sphere, &plane, 1e-4);
    // The enum receiver takes the same road.
    let enum_result = Surface::from(sphere.clone()).intersect(&Surface::from(plane), TOL);
    assert_eq!(enum_result.method(), IntersectionMethod::Numeric);
    assert_eq!(enum_result.curves().len(), 1);
}

#[test]
fn test_coincident_planes_are_coincident() {
    let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    let result = plane.intersect(&plane, TOL);
    assert!(result.is_coincident());
    assert_eq!(result.method(), IntersectionMethod::Exact);
    assert!(result.curves().is_empty() && !result.is_empty());
}

#[test]
fn test_tangent_sphere_gives_a_contact_point_and_parallel_planes_nothing() {
    let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    let touching = Plane::new(Point3D::new(0.0, 0.0, 2.0), Vector3D::Z).unwrap();
    let result = touching.intersect(&sphere, TOL);
    assert_eq!(result.contact_points().len(), 1);
    assert!(result.curves().is_empty());
    let high = Plane::new(Point3D::new(0.0, 0.0, 3.0), Vector3D::Z).unwrap();
    let low = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    assert!(high.intersect(&low, TOL).is_empty());
}

#[test]
fn test_grazing_cylinders_report_a_tangent_line() {
    let a = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    let b = Cylinder::new(Point3D::new(2.0, 0.0, 0.0), Vector3D::Z, 1.0).unwrap();
    let result = a.intersect(&b, TOL);
    assert_eq!(result.curves().len(), 1);
    assert!(result.curves()[0].tangent);
}
