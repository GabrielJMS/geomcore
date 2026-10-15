//! Every projection carries the closest point: the curve or surface evaluated
//! at the returned parameters.

use geomcore::curves::{
    Curve2D, Curve3D, ElementaryCurve, ParametricCurve2D, ParametricCurve3D, PointExtrema3D,
    TrimmedCurve3D,
};
use geomcore::surfaces::{ElementarySurface, ParametricSurface, Surface};
use geomcore::{
    Circle2D, Circle3D, Cone, CurveProjection, Cylinder, Ellipse3D, Frame3D, Hyperbola3D, Line2D,
    Line3D, Parabola3D, Plane, Point2D, Point3D, Sphere, Tolerance, Torus, Vector2D, Vector3D,
};

fn query_points() -> Vec<Point3D> {
    vec![
        Point3D::new(3.0, -1.0, 2.0),
        Point3D::new(0.0, 0.0, 5.0),
        Point3D::new(-2.5, 4.0, -1.0),
        Point3D::new(0.1, 0.2, 0.3),
    ]
}

fn check_curve<C: ParametricCurve3D>(
    name: &str,
    curve: &C,
    project: impl Fn(Point3D) -> CurveProjection,
) {
    for q in query_points() {
        let p = project(q);
        let expected = curve.eval_point(p.parameter);
        assert!(
            p.point.distance(expected) <= 1e-12,
            "{name}: point is not eval(parameter) for {q:?}"
        );
        assert!(
            (p.point.distance(q) - p.distance).abs() <= 1e-7 || p.distance == 0.0,
            "{name}: distance"
        );
    }
}

fn check_surface<S: ParametricSurface>(name: &str, surface: &S) {
    for q in query_points() {
        let p = surface.project_point(q, Tolerance::DEFAULT);
        let expected = surface.eval_point(p.u, p.v);
        assert!(
            p.point.distance(expected) <= 1e-12,
            "{name}: point is not eval(u, v) for {q:?}"
        );
    }
}

#[test]
fn test_every_curve_projection_returns_its_point() {
    let tol = Tolerance::DEFAULT;
    let line = Line3D::new(Point3D::ORIGIN, Vector3D::new(1.0, 1.0, 0.0)).unwrap();
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    let ellipse = Ellipse3D::from_frame(Frame3D::WORLD, 3.0, 1.0).unwrap();
    let parabola = Parabola3D::from_frame(Frame3D::WORLD, 0.5).unwrap();
    let hyperbola = Hyperbola3D::from_frame(Frame3D::WORLD, 1.0, 0.5).unwrap();
    let bspline = circle.to_bspline();
    check_curve("line", &line, |q| line.project_point(q, tol));
    check_curve("circle", &circle, |q| circle.project_point(q, tol));
    check_curve("ellipse", &ellipse, |q| ellipse.project_point(q, tol));
    check_curve("parabola", &parabola, |q| parabola.project_point(q, tol));
    check_curve("hyperbola", &hyperbola, |q| hyperbola.project_point(q, tol));
    check_curve("bspline", &bspline, |q| bspline.project_point(q, tol));
    let general = Curve3D::from(ellipse);
    check_curve("Curve3D", &general, |q| general.project_point(q, tol));
    let bspline_general = Curve3D::from(bspline.clone());
    check_curve("Curve3D bspline", &bspline_general, |q| {
        bspline_general.project_point(q, tol)
    });
    let elementary = ElementaryCurve::Hyperbola(hyperbola);
    check_curve("ElementaryCurve", &elementary, |q| {
        elementary.project_point(q, tol)
    });
    let arc = TrimmedCurve3D::new(circle, 0.5, 2.0).unwrap();
    check_curve("trimmed", &arc, |q| arc.project_point(q, tol));
}

#[test]
fn test_elementary_curve_extrema_and_batch_projection_agree_with_the_basis() {
    let tol = Tolerance::DEFAULT;
    let ellipse = Ellipse3D::from_frame(Frame3D::WORLD, 3.0, 1.0).unwrap();
    let elementary = ElementaryCurve::Ellipse(ellipse);
    let q = Point3D::new(1.0, 2.0, 0.5);
    assert_eq!(
        PointExtrema3D::extrema(&elementary, q, tol),
        ellipse.extrema(q, tol)
    );
    let points = query_points();
    assert_eq!(
        elementary.project_points(&points, tol),
        ellipse.project_points(&points, tol)
    );
    let general = Curve3D::from(ellipse);
    assert_eq!(
        general.project_points(&points, tol),
        ellipse.project_points(&points, tol)
    );
}

#[test]
fn test_curve_2d_projection_returns_its_point() {
    let tol = Tolerance::DEFAULT;
    let curves: Vec<Curve2D> = vec![
        Line2D::new(Point2D::ORIGIN, Vector2D::new(1.0, 1.0))
            .unwrap()
            .into(),
        Circle2D::new(Point2D::new(1.0, 0.0), 2.0).unwrap().into(),
    ];
    for curve in &curves {
        for q in [Point2D::new(3.0, -1.0), Point2D::new(-0.5, 0.25)] {
            let p = curve.project_point(q, tol);
            assert!(p.point.distance(curve.eval_point(p.parameter)) <= 1e-12);
        }
        let batch = curve.project_points(&[Point2D::new(3.0, -1.0)], tol);
        assert_eq!(batch[0], curve.project_point(Point2D::new(3.0, -1.0), tol));
    }
}

#[test]
fn test_every_surface_projection_returns_its_point() {
    let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.5).unwrap();
    let cone = Cone::new(Point3D::ORIGIN, Vector3D::Z, 0.4, 1.0).unwrap();
    let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 3.0, 1.0).unwrap();
    check_surface("plane", &plane);
    check_surface("cylinder", &cylinder);
    check_surface("cone", &cone);
    check_surface("sphere", &sphere);
    check_surface("torus", &torus);
    check_surface("Surface", &Surface::from(torus));
    check_surface("ElementarySurface", &ElementarySurface::Cone(cone));
    check_surface("bspline", &sphere.to_bspline());
}

#[test]
fn test_surface_enum_inherent_batch_projection() {
    let tol = Tolerance::DEFAULT;
    let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    let points = query_points();
    let surface = Surface::from(sphere);
    assert_eq!(
        surface.project_points(&points, tol),
        sphere.project_points(&points, tol)
    );
    let elementary = ElementarySurface::Sphere(sphere);
    assert_eq!(
        elementary.project_points(&points, tol),
        sphere.project_points(&points, tol)
    );
}

#[test]
fn test_contained_point_projects_onto_itself() {
    let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    let q = Point3D::new(0.0, 0.0, 2.0);
    let p = sphere.project_point(q, Tolerance::DEFAULT);
    assert_eq!(p.distance, 0.0);
    assert!(p.point.distance(q) <= 1e-12);
}
