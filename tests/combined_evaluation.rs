//! `eval_d1` and `eval_d2` return what `eval_point` and `eval_derivative`
//! return one at a time, on every curve and surface type and through the
//! run-time enums, trimmed curves and references.

use geomcore::curves::{Curve3D, ParametricCurve3D, TrimmedCurve3D};
use geomcore::surfaces::{ParametricSurface, Surface};
use geomcore::{
    BSplineCurve3D, BSplineSurface, Circle3D, Cone, Cylinder, Ellipse3D, Hyperbola3D, Line3D,
    Parabola3D, Plane, Point3D, Sphere, Torus, Vector3D,
};

const X: Vector3D = Vector3D::X;
const Z: Vector3D = Vector3D::Z;

fn assert_close(a: Vector3D, b: Vector3D, what: &str) {
    let scale = 1.0 + a.magnitude().max(b.magnitude());
    assert!(
        (a - b).magnitude() <= 1e-12 * scale,
        "{what}: {a:?} != {b:?}"
    );
}

fn assert_curve<C: ParametricCurve3D>(curve: &C, params: &[f64]) {
    for &u in params {
        let p = curve.eval_point(u) - Point3D::ORIGIN;
        let (d1, d2) = (curve.eval_derivative(u, 1), curve.eval_derivative(u, 2));
        let (p1, c1) = curve.eval_d1(u);
        assert_close(p1 - Point3D::ORIGIN, p, "eval_d1 point");
        assert_close(c1, d1, "eval_d1 first derivative");
        let (p2, c1, c2) = curve.eval_d2(u);
        assert_close(p2 - Point3D::ORIGIN, p, "eval_d2 point");
        assert_close(c1, d1, "eval_d2 first derivative");
        assert_close(c2, d2, "eval_d2 second derivative");
    }
}

fn assert_surface<S: ParametricSurface>(surface: &S, params: &[(f64, f64)]) {
    for &(u, v) in params {
        let p = surface.eval_point(u, v) - Point3D::ORIGIN;
        let d = |du, dv| surface.eval_derivative(u, v, du, dv);
        let (p1, [su, sv]) = surface.eval_d1(u, v);
        assert_close(p1 - Point3D::ORIGIN, p, "eval_d1 point");
        assert_close(su, d(1, 0), "eval_d1 Su");
        assert_close(sv, d(0, 1), "eval_d1 Sv");
        let (p2, [su, sv], [suu, suv, svv]) = surface.eval_d2(u, v);
        assert_close(p2 - Point3D::ORIGIN, p, "eval_d2 point");
        assert_close(su, d(1, 0), "eval_d2 Su");
        assert_close(sv, d(0, 1), "eval_d2 Sv");
        assert_close(suu, d(2, 0), "eval_d2 Suu");
        assert_close(suv, d(1, 1), "eval_d2 Suv");
        assert_close(svv, d(0, 2), "eval_d2 Svv");
    }
}

fn cubic() -> BSplineCurve3D {
    let poles = (0..7)
        .map(|i| {
            let t = i as f64;
            Point3D::new(t, (t * 0.9).sin() * 2.0, 0.3 * t * t)
        })
        .collect();
    BSplineCurve3D::new(
        3,
        poles,
        vec![0.0, 1.0, 2.0, 2.5, 4.0],
        vec![4, 1, 1, 1, 4],
        false,
    )
    .unwrap()
}

fn rational_periodic() -> BSplineCurve3D {
    let poles = (0..6)
        .map(|i| {
            let a = i as f64 * std::f64::consts::TAU / 6.0;
            Point3D::new(2.0 * a.cos(), a.sin(), 0.2 * (2.0 * a).sin())
        })
        .collect();
    let weights = vec![1.0, 0.7, 1.4, 1.0, 0.8, 1.2];
    let knots = (0..7).map(|i| i as f64).collect();
    BSplineCurve3D::new_rational(2, poles, weights, knots, vec![1; 7], true).unwrap()
}

fn patch(rational: bool) -> BSplineSurface {
    let poles: Vec<Vec<Point3D>> = (0..5)
        .map(|i| {
            (0..4)
                .map(|j| Point3D::new(i as f64, j as f64, ((i * j) as f64 * 0.7).sin()))
                .collect()
        })
        .collect();
    let (uk, um) = (vec![0.0, 1.0, 2.0, 3.0], vec![3u32, 1, 1, 3]);
    let (vk, vm) = (vec![0.0, 1.5, 3.0], vec![3u32, 1, 3]);
    if rational {
        let weights = (0..5)
            .map(|i| {
                (0..4)
                    .map(|j| 1.0 + 0.15 * ((i + 2 * j) % 3) as f64)
                    .collect()
            })
            .collect();
        BSplineSurface::new_rational(2, 2, poles, weights, uk, um, vk, vm, false, false).unwrap()
    } else {
        BSplineSurface::new(2, 2, poles, uk, um, vk, vm, false, false).unwrap()
    }
}

const CURVE_PARAMS: [f64; 6] = [-0.7, 0.0, 0.4, 1.0, 2.2, 3.9];

#[test]
fn test_elementary_curves_combined_evaluation() {
    assert_curve(
        &Line3D::new(Point3D::new(1.0, 2.0, 3.0), X + Z).unwrap(),
        &CURVE_PARAMS,
    );
    assert_curve(
        &Circle3D::new(Point3D::ORIGIN, Z, 2.0).unwrap(),
        &CURVE_PARAMS,
    );
    assert_curve(
        &Ellipse3D::new(Point3D::ORIGIN, Z, X, 3.0, 1.0).unwrap(),
        &CURVE_PARAMS,
    );
    assert_curve(
        &Parabola3D::new(Point3D::ORIGIN, Z, X, 0.5).unwrap(),
        &CURVE_PARAMS,
    );
    assert_curve(
        &Hyperbola3D::new(Point3D::ORIGIN, Z, X, 2.0, 1.0).unwrap(),
        &CURVE_PARAMS,
    );
}

#[test]
fn test_bspline_curves_combined_evaluation() {
    // Knots, interior points and a parameter outside the period.
    assert_curve(&cubic(), &[0.0, 0.3, 1.0, 2.0, 2.4, 2.5, 3.7, 4.0]);
    assert_curve(
        &rational_periodic(),
        &[0.0, 0.5, 1.0, 2.9, 5.99, 6.0, 7.3, -1.2],
    );
}

#[test]
fn test_wrapped_curves_combined_evaluation() {
    let curve = cubic();
    assert_curve(&Curve3D::from(curve.clone()), &[0.2, 2.5, 3.1]);
    assert_curve(
        &Curve3D::from(Circle3D::new(Point3D::ORIGIN, Z, 2.0).unwrap()),
        &[0.2, 3.1],
    );
    assert_curve(
        &TrimmedCurve3D::new(curve.clone(), 0.5, 3.0).unwrap(),
        &[0.5, 1.7, 3.0],
    );
    assert_curve(
        &TrimmedCurve3D::new(&curve, 0.5, 3.0).unwrap(),
        &[0.5, 1.7, 3.0],
    );
    assert_curve(&&curve, &[0.2, 2.5]);
}

const SURFACE_PARAMS: [(f64, f64); 5] =
    [(0.0, 0.0), (0.3, 1.1), (1.0, 2.0), (2.6, 0.4), (3.0, 3.0)];

#[test]
fn test_elementary_surfaces_combined_evaluation() {
    let o = Point3D::new(0.5, -1.0, 2.0);
    let axis = Vector3D::new(0.2, 0.3, 1.0);
    assert_surface(&Plane::new(o, axis).unwrap(), &SURFACE_PARAMS);
    assert_surface(&Cylinder::new(o, axis, 1.5).unwrap(), &SURFACE_PARAMS);
    assert_surface(&Cone::new(o, axis, 0.4, 1.5).unwrap(), &SURFACE_PARAMS);
    assert_surface(
        &Sphere::new(o, 2.0).unwrap(),
        &[(0.3, 0.2), (2.0, -1.1), (5.0, 1.2)],
    );
    assert_surface(&Torus::new(o, axis, 3.0, 1.0).unwrap(), &SURFACE_PARAMS);
}

#[test]
fn test_bspline_surfaces_combined_evaluation() {
    assert_surface(&patch(false), &SURFACE_PARAMS);
    assert_surface(&patch(true), &SURFACE_PARAMS);
    assert_surface(&Surface::from(patch(true)), &SURFACE_PARAMS);
    assert_surface(
        &Surface::from(Sphere::new(Point3D::ORIGIN, 2.0).unwrap()),
        &[(0.3, 0.2)],
    );
}
