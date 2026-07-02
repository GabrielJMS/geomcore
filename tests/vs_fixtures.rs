//! Golden-fixture replay tests: build geomrust values from recorded JSON
//! specs and check parity against recorded expected outputs. Grows with
//! every fixture-replay task.

mod common;

use common::{
    FrameJson, assert_close, assert_point3, assert_vector3, frame3, load, point3, vector3,
};
use geomrust::curves::ParametricCurve2D;
use geomrust::surfaces::{BSplineSurface, ParametricSurface};
use geomrust::{
    Axis3, BSplineCurve3D, Circle3D, Cone, Curve2D, Cylinder, Ellipse3D, Hyperbola3D, Line3D,
    Parabola3D, ParametrizeError, Plane, Sphere, Torus, Transform,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct MirrorAxisSpec {
    origin: [f64; 3],
    dir: [f64; 3],
}

#[derive(Deserialize)]
struct MirrorPlaneSpec {
    frame: FrameJson,
}

#[derive(Deserialize)]
struct TransformCase {
    kind: String,
    spec: serde_json::Value,
    apply_to_points: Vec<[f64; 3]>,
    expected_points: Vec<[f64; 3]>,
    apply_to_vectors: Vec<[f64; 3]>,
    expected_vectors: Vec<[f64; 3]>,
}

#[derive(Deserialize)]
struct TransformsFixture {
    cases: Vec<TransformCase>,
}

fn build_transform(cases: &[TransformCase], index: usize) -> Transform {
    let case = &cases[index];
    match case.kind.as_str() {
        "translation" => {
            #[derive(Deserialize)]
            struct Spec {
                offset: [f64; 3],
            }
            let spec: Spec = serde_json::from_value(case.spec.clone()).unwrap();
            Transform::translation(vector3(spec.offset))
        }
        "rotation" => {
            #[derive(Deserialize)]
            struct Spec {
                axis_origin: [f64; 3],
                axis_dir: [f64; 3],
                angle: f64,
            }
            let spec: Spec = serde_json::from_value(case.spec.clone()).unwrap();
            let axis = Axis3::new(point3(spec.axis_origin), vector3(spec.axis_dir)).unwrap();
            Transform::rotation(axis, spec.angle)
        }
        "scale" => {
            #[derive(Deserialize)]
            struct Spec {
                center: [f64; 3],
                factor: f64,
            }
            let spec: Spec = serde_json::from_value(case.spec.clone()).unwrap();
            Transform::scaling(point3(spec.center), spec.factor)
        }
        "mirror_point" => {
            #[derive(Deserialize)]
            struct Spec {
                point: [f64; 3],
            }
            let spec: Spec = serde_json::from_value(case.spec.clone()).unwrap();
            Transform::mirror_point(point3(spec.point))
        }
        "mirror_axis" => {
            let spec: MirrorAxisSpec = serde_json::from_value(case.spec.clone()).unwrap();
            let axis = Axis3::new(point3(spec.origin), vector3(spec.dir)).unwrap();
            Transform::mirror_axis(axis)
        }
        "mirror_plane" => {
            let spec: MirrorPlaneSpec = serde_json::from_value(case.spec.clone()).unwrap();
            Transform::mirror_plane(frame3(&spec.frame))
        }
        "compose" => {
            #[derive(Deserialize)]
            struct Spec {
                compose: [usize; 2],
            }
            let spec: Spec = serde_json::from_value(case.spec.clone()).unwrap();
            let first = build_transform(cases, spec.compose[0]);
            let second = build_transform(cases, spec.compose[1]);
            first.then(second)
        }
        other => panic!("unknown transform kind: {other}"),
    }
}

#[test]
fn transforms_match_golden_fixture() {
    let fixture: TransformsFixture = serde_json::from_value(load("transforms.json")).unwrap();

    for (i, case) in fixture.cases.iter().enumerate() {
        let transform = build_transform(&fixture.cases, i);

        assert_eq!(
            case.apply_to_points.len(),
            case.expected_points.len(),
            "case {i} ({}): points/expected_points length mismatch",
            case.kind
        );
        for (p, expected) in case.apply_to_points.iter().zip(&case.expected_points) {
            let actual = transform.apply_point(point3(*p));
            assert_point3(actual, *expected);
        }

        assert_eq!(
            case.apply_to_vectors.len(),
            case.expected_vectors.len(),
            "case {i} ({}): vectors/expected_vectors length mismatch",
            case.kind
        );
        for (v, expected) in case.apply_to_vectors.iter().zip(&case.expected_vectors) {
            let actual = transform.apply_vector(vector3(*v));
            assert_vector3(actual, *expected);
        }
    }
}

#[derive(Deserialize)]
struct LineSample {
    u: f64,
    point: [f64; 3],
    d1: [f64; 3],
    d2: [f64; 3],
    d3: [f64; 3],
}

#[derive(Deserialize)]
struct LineParameterOf {
    point: [f64; 3],
    u: f64,
}

#[derive(Deserialize)]
struct LineCase {
    origin: [f64; 3],
    direction: [f64; 3],
    samples: Vec<LineSample>,
    parameter_of: Vec<LineParameterOf>,
}

#[derive(Deserialize)]
struct CircleSample {
    u: f64,
    point: [f64; 3],
    d1: [f64; 3],
    d2: [f64; 3],
    d3: [f64; 3],
}

#[derive(Deserialize)]
struct CircleParameterOf {
    point: [f64; 3],
    u: f64,
}

#[derive(Deserialize)]
struct CircleCase {
    frame: FrameJson,
    radius: f64,
    samples: Vec<CircleSample>,
    parameter_of: Vec<CircleParameterOf>,
}

#[derive(Deserialize)]
struct ThreePointCase {
    p1: [f64; 3],
    p2: [f64; 3],
    p3: [f64; 3],
    frame: FrameJson,
    radius: f64,
}

#[derive(Deserialize)]
struct EllipseSample {
    u: f64,
    point: [f64; 3],
    d1: [f64; 3],
    d2: [f64; 3],
    d3: [f64; 3],
}

#[derive(Deserialize)]
struct EllipseParameterOf {
    point: [f64; 3],
    u: f64,
}

#[derive(Deserialize)]
struct EllipseCase {
    frame: FrameJson,
    major_radius: f64,
    minor_radius: f64,
    samples: Vec<EllipseSample>,
    parameter_of: Vec<EllipseParameterOf>,
}

#[derive(Deserialize)]
struct ParabolaSample {
    u: f64,
    point: [f64; 3],
    d1: [f64; 3],
    d2: [f64; 3],
    d3: [f64; 3],
}

#[derive(Deserialize)]
struct ParabolaParameterOf {
    point: [f64; 3],
    u: f64,
}

#[derive(Deserialize)]
struct ParabolaCase {
    frame: FrameJson,
    focal: f64,
    samples: Vec<ParabolaSample>,
    parameter_of: Vec<ParabolaParameterOf>,
}

#[derive(Deserialize)]
struct HyperbolaSample {
    u: f64,
    point: [f64; 3],
    d1: [f64; 3],
    d2: [f64; 3],
    d3: [f64; 3],
}

#[derive(Deserialize)]
struct HyperbolaParameterOf {
    point: [f64; 3],
    u: f64,
}

#[derive(Deserialize)]
struct HyperbolaCase {
    frame: FrameJson,
    major_radius: f64,
    minor_radius: f64,
    samples: Vec<HyperbolaSample>,
    parameter_of: Vec<HyperbolaParameterOf>,
}

#[derive(Deserialize)]
struct CurvesAnalyticFixture {
    lines: Vec<LineCase>,
    circles: Vec<CircleCase>,
    three_point_cases: Vec<ThreePointCase>,
    ellipses: Vec<EllipseCase>,
    parabolas: Vec<ParabolaCase>,
    hyperbolas: Vec<HyperbolaCase>,
}

#[test]
fn lines_match_golden_fixture() {
    let fixture: CurvesAnalyticFixture =
        serde_json::from_value(load("curves_analytic.json")).unwrap();

    for (i, case) in fixture.lines.iter().enumerate() {
        let line = Line3D::new(point3(case.origin), vector3(case.direction))
            .unwrap_or_else(|e| panic!("case {i}: failed to build Line3D: {e}"));

        for sample in &case.samples {
            let point = line.eval_point(sample.u);
            assert_point3(point, sample.point);

            let d1 = line.eval_derivative(sample.u, 1);
            assert_vector3(d1, sample.d1);

            let d2 = line.eval_derivative(sample.u, 2);
            assert_vector3(d2, sample.d2);

            let d3 = line.eval_derivative(sample.u, 3);
            assert_vector3(d3, sample.d3);
        }

        for inversion in &case.parameter_of {
            let u = line.parameter_of(point3(inversion.point));
            assert_close(u, inversion.u);
        }
    }
}

#[test]
fn circles_match_golden_fixture() {
    let fixture: CurvesAnalyticFixture =
        serde_json::from_value(load("curves_analytic.json")).unwrap();

    for (i, case) in fixture.circles.iter().enumerate() {
        let circle = Circle3D::from_frame(frame3(&case.frame), case.radius)
            .unwrap_or_else(|e| panic!("case {i}: failed to build Circle3D: {e}"));

        for sample in &case.samples {
            let point = circle.eval_point(sample.u);
            assert_point3(point, sample.point);

            let d1 = circle.eval_derivative(sample.u, 1);
            assert_vector3(d1, sample.d1);

            let d2 = circle.eval_derivative(sample.u, 2);
            assert_vector3(d2, sample.d2);

            let d3 = circle.eval_derivative(sample.u, 3);
            assert_vector3(d3, sample.d3);
        }

        for inversion in &case.parameter_of {
            let u = circle.parameter_of(point3(inversion.point));
            assert_close(u, inversion.u);
        }
    }
}

#[test]
fn circle_three_point_cases_match_golden_fixture() {
    let fixture: CurvesAnalyticFixture =
        serde_json::from_value(load("curves_analytic.json")).unwrap();

    for (i, case) in fixture.three_point_cases.iter().enumerate() {
        let circle = Circle3D::from_three_points(point3(case.p1), point3(case.p2), point3(case.p3))
            .unwrap_or_else(|e| panic!("case {i}: failed to build Circle3D: {e}"));

        assert_close(circle.radius(), case.radius);

        let frame = circle.frame();
        assert_point3(frame.origin(), case.frame.origin);
        assert_vector3(frame.x_direction(), case.frame.x_dir);
        assert_vector3(frame.y_direction(), case.frame.y_dir);
        assert_vector3(frame.z_direction(), case.frame.z_dir);
    }
}

#[test]
fn ellipses_match_golden_fixture() {
    let fixture: CurvesAnalyticFixture =
        serde_json::from_value(load("curves_analytic.json")).unwrap();

    for (i, case) in fixture.ellipses.iter().enumerate() {
        let ellipse =
            Ellipse3D::from_frame(frame3(&case.frame), case.major_radius, case.minor_radius)
                .unwrap_or_else(|e| panic!("case {i}: failed to build Ellipse3D: {e}"));

        for sample in &case.samples {
            let point = ellipse.eval_point(sample.u);
            assert_point3(point, sample.point);

            let d1 = ellipse.eval_derivative(sample.u, 1);
            assert_vector3(d1, sample.d1);

            let d2 = ellipse.eval_derivative(sample.u, 2);
            assert_vector3(d2, sample.d2);

            let d3 = ellipse.eval_derivative(sample.u, 3);
            assert_vector3(d3, sample.d3);
        }

        for inversion in &case.parameter_of {
            let u = ellipse.parameter_of(point3(inversion.point));
            assert_close(u, inversion.u);
        }
    }
}

#[test]
fn parabolas_match_golden_fixture() {
    let fixture: CurvesAnalyticFixture =
        serde_json::from_value(load("curves_analytic.json")).unwrap();

    for (i, case) in fixture.parabolas.iter().enumerate() {
        let parabola = Parabola3D::from_frame(frame3(&case.frame), case.focal)
            .unwrap_or_else(|e| panic!("case {i}: failed to build Parabola3D: {e}"));

        for sample in &case.samples {
            let point = parabola.eval_point(sample.u);
            assert_point3(point, sample.point);

            let d1 = parabola.eval_derivative(sample.u, 1);
            assert_vector3(d1, sample.d1);

            let d2 = parabola.eval_derivative(sample.u, 2);
            assert_vector3(d2, sample.d2);

            let d3 = parabola.eval_derivative(sample.u, 3);
            assert_vector3(d3, sample.d3);
        }

        for inversion in &case.parameter_of {
            let u = parabola.parameter_of(point3(inversion.point));
            assert_close(u, inversion.u);
        }
    }
}

#[test]
fn hyperbolas_match_golden_fixture() {
    let fixture: CurvesAnalyticFixture =
        serde_json::from_value(load("curves_analytic.json")).unwrap();

    for (i, case) in fixture.hyperbolas.iter().enumerate() {
        let hyperbola =
            Hyperbola3D::from_frame(frame3(&case.frame), case.major_radius, case.minor_radius)
                .unwrap_or_else(|e| panic!("case {i}: failed to build Hyperbola3D: {e}"));

        for sample in &case.samples {
            let point = hyperbola.eval_point(sample.u);
            assert_point3(point, sample.point);

            let d1 = hyperbola.eval_derivative(sample.u, 1);
            assert_vector3(d1, sample.d1);

            let d2 = hyperbola.eval_derivative(sample.u, 2);
            assert_vector3(d2, sample.d2);

            let d3 = hyperbola.eval_derivative(sample.u, 3);
            assert_vector3(d3, sample.d3);
        }

        for inversion in &case.parameter_of {
            let u = hyperbola.parameter_of(point3(inversion.point));
            assert_close(u, inversion.u);
        }
    }
}

#[derive(Deserialize)]
struct EllipseCenterTwoPointsCase {
    s1: [f64; 3],
    s2: [f64; 3],
    center: [f64; 3],
    frame: FrameJson,
    major_radius: f64,
    minor_radius: f64,
}

#[derive(Deserialize)]
struct HyperbolaCenterTwoPointsCase {
    s1: [f64; 3],
    s2: [f64; 3],
    center: [f64; 3],
    frame: FrameJson,
    major_radius: f64,
    minor_radius: f64,
}

#[derive(Deserialize)]
struct ConstructionFixture {
    ellipses_center_two_points: Vec<EllipseCenterTwoPointsCase>,
    hyperbolas_center_two_points: Vec<HyperbolaCenterTwoPointsCase>,
}

#[test]
fn ellipses_center_two_points_match_golden_fixture() {
    let fixture: ConstructionFixture = serde_json::from_value(load("construction.json")).unwrap();

    for (i, case) in fixture.ellipses_center_two_points.iter().enumerate() {
        let ellipse = Ellipse3D::from_center_and_points(
            point3(case.center),
            point3(case.s1),
            point3(case.s2),
        )
        .unwrap_or_else(|e| panic!("case {i}: failed to build Ellipse3D: {e}"));

        assert_close(ellipse.major_radius(), case.major_radius);
        assert_close(ellipse.minor_radius(), case.minor_radius);

        let frame = ellipse.frame();
        assert_point3(frame.origin(), case.frame.origin);
        assert_vector3(frame.x_direction(), case.frame.x_dir);
        assert_vector3(frame.y_direction(), case.frame.y_dir);
        assert_vector3(frame.z_direction(), case.frame.z_dir);
    }
}

#[test]
fn hyperbolas_center_two_points_match_golden_fixture() {
    let fixture: ConstructionFixture = serde_json::from_value(load("construction.json")).unwrap();

    for (i, case) in fixture.hyperbolas_center_two_points.iter().enumerate() {
        let hyperbola = Hyperbola3D::from_center_and_points(
            point3(case.center),
            point3(case.s1),
            point3(case.s2),
        )
        .unwrap_or_else(|e| panic!("case {i}: failed to build Hyperbola3D: {e}"));

        assert_close(hyperbola.major_radius(), case.major_radius);
        assert_close(hyperbola.minor_radius(), case.minor_radius);

        let frame = hyperbola.frame();
        assert_point3(frame.origin(), case.frame.origin);
        assert_vector3(frame.x_direction(), case.frame.x_dir);
        assert_vector3(frame.y_direction(), case.frame.y_dir);
        assert_vector3(frame.z_direction(), case.frame.z_dir);
    }
}

#[derive(Deserialize)]
struct BSplineSample {
    u: f64,
    point: [f64; 3],
    d1: [f64; 3],
    d2: [f64; 3],
}

#[derive(Deserialize)]
struct BSplineCase {
    name: String,
    degree: usize,
    periodic: bool,
    poles: Vec<[f64; 3]>,
    weights: Option<Vec<f64>>,
    knots: Vec<f64>,
    mults: Vec<u32>,
    samples: Vec<BSplineSample>,
}

#[derive(Deserialize)]
struct BSplineFixture {
    cases: Vec<BSplineCase>,
}

fn build_bspline(case: &BSplineCase) -> BSplineCurve3D {
    let poles: Vec<geomrust::Point3> = case.poles.iter().map(|&p| point3(p)).collect();
    let result = match &case.weights {
        Some(weights) => BSplineCurve3D::new_rational(
            case.degree,
            poles,
            weights.clone(),
            case.knots.clone(),
            case.mults.clone(),
            case.periodic,
        ),
        None => BSplineCurve3D::new(
            case.degree,
            poles,
            case.knots.clone(),
            case.mults.clone(),
            case.periodic,
        ),
    };
    result.unwrap_or_else(|e| panic!("{}: failed to build BSplineCurve3D: {e}", case.name))
}

#[test]
fn bsplines_match_golden_fixture() {
    let fixture: BSplineFixture = serde_json::from_value(load("curves_bspline.json")).unwrap();

    for case in &fixture.cases {
        let curve = build_bspline(case);

        for sample in &case.samples {
            let point = curve.eval_point(sample.u);
            assert_point3(point, sample.point);

            let d1 = curve.eval_derivative(sample.u, 1);
            assert_vector3(d1, sample.d1);

            let d2 = curve.eval_derivative(sample.u, 2);
            assert_vector3(d2, sample.d2);
        }
    }
}

// ---- surfaces_analytic.json: the five analytic surface types ----

#[derive(Deserialize)]
struct SurfaceSample {
    u: f64,
    v: f64,
    point: [f64; 3],
    d1u: [f64; 3],
    d1v: [f64; 3],
    d2u: [f64; 3],
    d2v: [f64; 3],
    d2uv: [f64; 3],
}

#[derive(Deserialize)]
struct SurfaceParametersOf {
    point: [f64; 3],
    u: f64,
    v: f64,
}

#[derive(Deserialize)]
struct PlaneCase {
    frame: FrameJson,
    samples: Vec<SurfaceSample>,
    parameters_of: Vec<SurfaceParametersOf>,
}

#[derive(Deserialize)]
struct CylinderCase {
    frame: FrameJson,
    radius: f64,
    samples: Vec<SurfaceSample>,
    parameters_of: Vec<SurfaceParametersOf>,
}

#[derive(Deserialize)]
struct ConeCase {
    frame: FrameJson,
    ref_radius: f64,
    semi_angle: f64,
    samples: Vec<SurfaceSample>,
    parameters_of: Vec<SurfaceParametersOf>,
}

#[derive(Deserialize)]
struct SphereCase {
    frame: FrameJson,
    radius: f64,
    samples: Vec<SurfaceSample>,
    parameters_of: Vec<SurfaceParametersOf>,
}

#[derive(Deserialize)]
struct TorusCase {
    frame: FrameJson,
    major_radius: f64,
    minor_radius: f64,
    samples: Vec<SurfaceSample>,
    parameters_of: Vec<SurfaceParametersOf>,
}

#[derive(Deserialize)]
struct SurfacesAnalyticFixture {
    planes: Vec<PlaneCase>,
    cylinders: Vec<CylinderCase>,
    cones: Vec<ConeCase>,
    spheres: Vec<SphereCase>,
    tori: Vec<TorusCase>,
}

fn assert_surface_samples_and_parameters<F>(
    samples: &[SurfaceSample],
    parameters_of: &[SurfaceParametersOf],
    eval_point: impl Fn(f64, f64) -> geomrust::Point3,
    eval_derivative: impl Fn(f64, f64, u32, u32) -> geomrust::Vector3,
    parameters: F,
) where
    F: Fn(geomrust::Point3) -> (f64, f64),
{
    for sample in samples {
        let (u, v) = (sample.u, sample.v);
        assert_point3(eval_point(u, v), sample.point);
        assert_vector3(eval_derivative(u, v, 1, 0), sample.d1u);
        assert_vector3(eval_derivative(u, v, 0, 1), sample.d1v);
        assert_vector3(eval_derivative(u, v, 2, 0), sample.d2u);
        assert_vector3(eval_derivative(u, v, 0, 2), sample.d2v);
        assert_vector3(eval_derivative(u, v, 1, 1), sample.d2uv);
    }

    for inversion in parameters_of {
        let (u, v) = parameters(point3(inversion.point));
        assert_close(u, inversion.u);
        assert_close(v, inversion.v);
    }
}

#[test]
fn planes_match_golden_fixture() {
    let fixture: SurfacesAnalyticFixture =
        serde_json::from_value(load("surfaces_analytic.json")).unwrap();

    for case in &fixture.planes {
        let plane = Plane::from_frame(frame3(&case.frame));
        assert_surface_samples_and_parameters(
            &case.samples,
            &case.parameters_of,
            |u, v| plane.eval_point(u, v),
            |u, v, du, dv| plane.eval_derivative(u, v, du, dv),
            |p| plane.parameters_of(p),
        );
    }
}

#[test]
fn cylinders_match_golden_fixture() {
    let fixture: SurfacesAnalyticFixture =
        serde_json::from_value(load("surfaces_analytic.json")).unwrap();

    for (i, case) in fixture.cylinders.iter().enumerate() {
        let cylinder = Cylinder::from_frame(frame3(&case.frame), case.radius)
            .unwrap_or_else(|e| panic!("case {i}: failed to build Cylinder: {e}"));
        assert_surface_samples_and_parameters(
            &case.samples,
            &case.parameters_of,
            |u, v| cylinder.eval_point(u, v),
            |u, v, du, dv| cylinder.eval_derivative(u, v, du, dv),
            |p| cylinder.parameters_of(p),
        );
    }
}

#[test]
fn cones_match_golden_fixture() {
    let fixture: SurfacesAnalyticFixture =
        serde_json::from_value(load("surfaces_analytic.json")).unwrap();

    for (i, case) in fixture.cones.iter().enumerate() {
        let cone = Cone::from_frame(frame3(&case.frame), case.semi_angle, case.ref_radius)
            .unwrap_or_else(|e| panic!("case {i}: failed to build Cone: {e}"));
        assert_surface_samples_and_parameters(
            &case.samples,
            &case.parameters_of,
            |u, v| cone.eval_point(u, v),
            |u, v, du, dv| cone.eval_derivative(u, v, du, dv),
            |p| cone.parameters_of(p),
        );
    }
}

#[test]
fn spheres_match_golden_fixture() {
    let fixture: SurfacesAnalyticFixture =
        serde_json::from_value(load("surfaces_analytic.json")).unwrap();

    for (i, case) in fixture.spheres.iter().enumerate() {
        let sphere = Sphere::from_frame(frame3(&case.frame), case.radius)
            .unwrap_or_else(|e| panic!("case {i}: failed to build Sphere: {e}"));
        assert_surface_samples_and_parameters(
            &case.samples,
            &case.parameters_of,
            |u, v| sphere.eval_point(u, v),
            |u, v, du, dv| sphere.eval_derivative(u, v, du, dv),
            |p| sphere.parameters_of(p),
        );
    }
}

#[test]
fn tori_match_golden_fixture() {
    let fixture: SurfacesAnalyticFixture =
        serde_json::from_value(load("surfaces_analytic.json")).unwrap();

    for (i, case) in fixture.tori.iter().enumerate() {
        let torus = Torus::from_frame(frame3(&case.frame), case.major_radius, case.minor_radius)
            .unwrap_or_else(|e| panic!("case {i}: failed to build Torus: {e}"));
        assert_surface_samples_and_parameters(
            &case.samples,
            &case.parameters_of,
            |u, v| torus.eval_point(u, v),
            |u, v, du, dv| torus.eval_derivative(u, v, du, dv),
            |p| torus.parameters_of(p),
        );
    }
}

// ---- construction.json: plane/cone/cylinder derived-construction goldens ----

#[derive(Deserialize)]
struct PlaneFromThreePointsCase {
    p1: [f64; 3],
    p2: [f64; 3],
    p3: [f64; 3],
    frame: FrameJson,
}

#[derive(Deserialize)]
struct PlaneFromCoefficientsCase {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    frame: FrameJson,
}

#[derive(Deserialize)]
struct ConeTwoPointsRadiiCase {
    p1: [f64; 3],
    p2: [f64; 3],
    r1: f64,
    r2: f64,
    frame: FrameJson,
    ref_radius: f64,
    semi_angle: f64,
}

#[derive(Deserialize)]
struct CircleFrameJson {
    frame: FrameJson,
    radius: f64,
}

#[derive(Deserialize)]
struct CylinderFromCircleCase {
    circle: CircleFrameJson,
    frame: FrameJson,
    radius: f64,
}

#[derive(Deserialize)]
struct SurfaceConstructionFixture {
    planes_from_three_points: Vec<PlaneFromThreePointsCase>,
    planes_from_coefficients: Vec<PlaneFromCoefficientsCase>,
    cones_two_points_radii: Vec<ConeTwoPointsRadiiCase>,
    cylinders_from_circle: Vec<CylinderFromCircleCase>,
}

fn assert_frame_matches(frame: geomrust::Frame3, expected: &FrameJson) {
    assert_point3(frame.origin(), expected.origin);
    assert_vector3(frame.x_direction(), expected.x_dir);
    assert_vector3(frame.y_direction(), expected.y_dir);
    assert_vector3(frame.z_direction(), expected.z_dir);
}

#[test]
fn planes_from_three_points_match_golden_fixture() {
    let fixture: SurfaceConstructionFixture =
        serde_json::from_value(load("construction.json")).unwrap();

    for (i, case) in fixture.planes_from_three_points.iter().enumerate() {
        let plane = Plane::from_three_points(point3(case.p1), point3(case.p2), point3(case.p3))
            .unwrap_or_else(|e| panic!("case {i}: failed to build Plane: {e}"));
        assert_frame_matches(plane.frame(), &case.frame);
    }
}

#[test]
fn planes_from_coefficients_match_golden_fixture() {
    let fixture: SurfaceConstructionFixture =
        serde_json::from_value(load("construction.json")).unwrap();

    for (i, case) in fixture.planes_from_coefficients.iter().enumerate() {
        let plane = Plane::from_coefficients(case.a, case.b, case.c, case.d)
            .unwrap_or_else(|e| panic!("case {i}: failed to build Plane: {e}"));
        assert_frame_matches(plane.frame(), &case.frame);
    }
}

#[test]
fn cones_from_two_points_and_radii_match_golden_fixture() {
    let fixture: SurfaceConstructionFixture =
        serde_json::from_value(load("construction.json")).unwrap();

    for (i, case) in fixture.cones_two_points_radii.iter().enumerate() {
        let cone =
            Cone::from_two_points_and_radii(point3(case.p1), point3(case.p2), case.r1, case.r2)
                .unwrap_or_else(|e| panic!("case {i}: failed to build Cone: {e}"));
        assert_close(cone.ref_radius(), case.ref_radius);
        assert_close(cone.semi_angle(), case.semi_angle);
        assert_frame_matches(cone.frame(), &case.frame);
    }
}

#[test]
fn cylinders_from_circle_match_golden_fixture() {
    let fixture: SurfaceConstructionFixture =
        serde_json::from_value(load("construction.json")).unwrap();

    for case in &fixture.cylinders_from_circle {
        let circle = Circle3D::from_frame(frame3(&case.circle.frame), case.circle.radius).unwrap();
        let cylinder = Cylinder::from_circle(&circle);
        assert_close(cylinder.radius(), case.radius);
        assert_frame_matches(cylinder.frame(), &case.frame);
    }
}

// ---- surfaces_bspline.json: tensor-product B-spline surfaces ----

#[derive(Deserialize)]
struct BSplineSurfaceSample {
    u: f64,
    v: f64,
    point: [f64; 3],
    d1u: [f64; 3],
    d1v: [f64; 3],
}

#[derive(Deserialize)]
struct BSplineSurfaceCase {
    name: String,
    u_degree: usize,
    v_degree: usize,
    u_periodic: bool,
    v_periodic: bool,
    poles: Vec<Vec<[f64; 3]>>,
    weights: Option<Vec<Vec<f64>>>,
    u_knots: Vec<f64>,
    u_mults: Vec<u32>,
    v_knots: Vec<f64>,
    v_mults: Vec<u32>,
    samples: Vec<BSplineSurfaceSample>,
}

#[derive(Deserialize)]
struct BSplineSurfaceFixture {
    cases: Vec<BSplineSurfaceCase>,
}

fn build_bspline_surface(case: &BSplineSurfaceCase) -> BSplineSurface {
    let poles: Vec<Vec<geomrust::Point3>> = case
        .poles
        .iter()
        .map(|row| row.iter().map(|&p| point3(p)).collect())
        .collect();
    let result = match &case.weights {
        Some(weights) => BSplineSurface::new_rational(
            case.u_degree,
            case.v_degree,
            poles,
            weights.clone(),
            case.u_knots.clone(),
            case.u_mults.clone(),
            case.v_knots.clone(),
            case.v_mults.clone(),
            case.u_periodic,
            case.v_periodic,
        ),
        None => BSplineSurface::new(
            case.u_degree,
            case.v_degree,
            poles,
            case.u_knots.clone(),
            case.u_mults.clone(),
            case.v_knots.clone(),
            case.v_mults.clone(),
            case.u_periodic,
            case.v_periodic,
        ),
    };
    result.unwrap_or_else(|e| panic!("{}: failed to build BSplineSurface: {e}", case.name))
}

#[test]
fn bspline_surfaces_match_golden_fixture() {
    let fixture: BSplineSurfaceFixture =
        serde_json::from_value(load("surfaces_bspline.json")).unwrap();

    for case in &fixture.cases {
        let surface = build_bspline_surface(case);

        for sample in &case.samples {
            let (u, v) = (sample.u, sample.v);
            assert_point3(surface.eval_point(u, v), sample.point);
            assert_vector3(surface.eval_derivative(u, v, 1, 0), sample.d1u);
            assert_vector3(surface.eval_derivative(u, v, 0, 1), sample.d1v);
        }
    }
}

// ---- parametrize.json: analytic curve-on-surface parametrization ----

#[derive(Deserialize)]
struct ParametrizeSurface {
    kind: String,
    frame: FrameJson,
    radius: Option<f64>,
    ref_radius: Option<f64>,
    semi_angle: Option<f64>,
    major_radius: Option<f64>,
    minor_radius: Option<f64>,
}

#[derive(Deserialize)]
struct ParametrizeCurve {
    kind: String,
    origin: Option<[f64; 3]>,
    direction: Option<[f64; 3]>,
    frame: Option<FrameJson>,
    radius: Option<f64>,
}

#[derive(Deserialize)]
struct ParametrizeExpect {
    kind: String,
    origin: Option<[f64; 2]>,
    direction: Option<[f64; 2]>,
    center: Option<[f64; 2]>,
    x_dir: Option<[f64; 2]>,
    y_dir: Option<[f64; 2]>,
    radius: Option<f64>,
}

#[derive(Deserialize)]
struct ParametrizeNormalized {
    origin: [f64; 2],
    direction: [f64; 2],
}

#[derive(Deserialize)]
struct ParametrizeCase {
    name: String,
    surface: ParametrizeSurface,
    curve: ParametrizeCurve,
    expect: ParametrizeExpect,
    #[serde(default)]
    in_window: bool,
    #[serde(default)]
    consistency_params: Vec<f64>,
    #[serde(default)]
    consistency_points: Vec<[f64; 3]>,
    normalized: Option<ParametrizeNormalized>,
}

#[derive(Deserialize)]
struct ParametrizeFixture {
    cases: Vec<ParametrizeCase>,
}

enum ParamCurve {
    Line(Line3D),
    Circle(Circle3D),
}

fn build_parametrize_surface(s: &ParametrizeSurface) -> geomrust::Surface {
    let frame = frame3(&s.frame);
    match s.kind.as_str() {
        "plane" => Plane::from_frame(frame).into(),
        "cylinder" => Cylinder::from_frame(frame, s.radius.unwrap())
            .unwrap()
            .into(),
        "cone" => Cone::from_frame(frame, s.semi_angle.unwrap(), s.ref_radius.unwrap())
            .unwrap()
            .into(),
        "sphere" => Sphere::from_frame(frame, s.radius.unwrap()).unwrap().into(),
        "torus" => Torus::from_frame(frame, s.major_radius.unwrap(), s.minor_radius.unwrap())
            .unwrap()
            .into(),
        other => panic!("unknown parametrize surface kind: {other}"),
    }
}

fn build_parametrize_curve(c: &ParametrizeCurve) -> ParamCurve {
    match c.kind.as_str() {
        "line" => ParamCurve::Line(
            Line3D::new(point3(c.origin.unwrap()), vector3(c.direction.unwrap())).unwrap(),
        ),
        "circle" => {
            let frame = frame3(c.frame.as_ref().unwrap());
            ParamCurve::Circle(Circle3D::from_frame(frame, c.radius.unwrap()).unwrap())
        }
        other => panic!("unknown parametrize curve kind: {other}"),
    }
}

fn parametrize_curve(
    curve: &ParamCurve,
    surface: &geomrust::Surface,
) -> Result<Curve2D, ParametrizeError> {
    match curve {
        ParamCurve::Line(l) => l.parametrize_on(surface.clone()),
        ParamCurve::Circle(c) => c.parametrize_on(surface.clone()),
    }
}

#[test]
fn parametrize_matches_golden_fixture() {
    let fixture: ParametrizeFixture = serde_json::from_value(load("parametrize.json")).unwrap();

    for case in &fixture.cases {
        let surface = build_parametrize_surface(&case.surface);
        let curve = build_parametrize_curve(&case.curve);
        let result = parametrize_curve(&curve, &surface);

        // not_analytic cases: must report NotAnalytic and stop.
        if case.expect.kind == "not_analytic" {
            assert_eq!(
                result,
                Err(ParametrizeError::NotAnalytic),
                "case {}: expected NotAnalytic",
                case.name
            );
            continue;
        }

        let pcurve =
            result.unwrap_or_else(|e| panic!("case {}: unexpected error {e:?}", case.name));

        // (a) result kind matches.
        match (case.expect.kind.as_str(), &pcurve) {
            ("line2d", Curve2D::Line(_)) => {}
            ("circle2d", Curve2D::Circle(_)) => {}
            (k, got) => panic!("case {}: expected kind {k}, got {got:?}", case.name),
        }

        // (b) consistency invariant: surface(q(t)) ~= consistency_points[i].
        for (t, expected) in case.consistency_params.iter().zip(&case.consistency_points) {
            let uv = pcurve.eval_point(*t);
            let on_surface = surface.eval_point(uv.x, uv.y);
            assert_point3(on_surface, *expected);
        }

        // (c) q(0) lies inside the canonical parameter window.
        let q0 = pcurve.eval_point(0.0);
        match case.surface.kind.as_str() {
            "cylinder" | "cone" => {
                assert!(
                    (-1e-9..=std::f64::consts::TAU + 1e-9).contains(&q0.x),
                    "case {}: u(0) = {} out of [0, 2pi)",
                    case.name,
                    q0.x
                );
            }
            "sphere" => {
                assert!(
                    (-1e-9..=std::f64::consts::TAU + 1e-9).contains(&q0.x),
                    "case {}: u(0) = {} out of [0, 2pi)",
                    case.name,
                    q0.x
                );
                assert!(
                    (-std::f64::consts::FRAC_PI_2 - 1e-9..=std::f64::consts::FRAC_PI_2 + 1e-9)
                        .contains(&q0.y),
                    "case {}: v(0) = {} out of [-pi/2, pi/2]",
                    case.name,
                    q0.y
                );
            }
            "torus" => {
                assert!(
                    (-1e-9..=std::f64::consts::TAU + 1e-9).contains(&q0.x),
                    "case {}: u(0) = {} out of [0, 2pi)",
                    case.name,
                    q0.x
                );
                assert!(
                    (-1e-9..=std::f64::consts::TAU + 1e-9).contains(&q0.y),
                    "case {}: v(0) = {} out of [0, 2pi)",
                    case.name,
                    q0.y
                );
            }
            _ => {}
        }

        // (d) literal comparison where the raw golden is already canonical.
        // Plane cases have no window, so compare literally always.
        if case.surface.kind == "plane" {
            match &pcurve {
                Curve2D::Line(l) => {
                    let o = case.expect.origin.unwrap();
                    let d = case.expect.direction.unwrap();
                    assert_close(l.origin().x, o[0]);
                    assert_close(l.origin().y, o[1]);
                    assert_close(l.direction().x, d[0]);
                    assert_close(l.direction().y, d[1]);
                }
                Curve2D::Circle(c) => {
                    let center = case.expect.center.unwrap();
                    let x_dir = case.expect.x_dir.unwrap();
                    let y_dir = case.expect.y_dir.unwrap();
                    assert_close(c.center().x, center[0]);
                    assert_close(c.center().y, center[1]);
                    assert_close(c.frame().x_direction().x, x_dir[0]);
                    assert_close(c.frame().x_direction().y, x_dir[1]);
                    assert_close(c.frame().y_direction().x, y_dir[0]);
                    assert_close(c.frame().y_direction().y, y_dir[1]);
                    assert_close(c.radius(), case.expect.radius.unwrap());
                }
                _ => unreachable!(),
            }
        } else if case.expect.kind == "line2d" && case.in_window && case.normalized.is_none() {
            // Periodic-surface line whose raw projection was already in-window:
            // raw == normalized, so compare origin/direction literally.
            if let Curve2D::Line(l) = &pcurve {
                let o = case.expect.origin.unwrap();
                let d = case.expect.direction.unwrap();
                assert_close(l.origin().x, o[0]);
                assert_close(l.origin().y, o[1]);
                assert_close(l.direction().x, d[0]);
                assert_close(l.direction().y, d[1]);
            }
        }

        // (e) sphere meridian cases: compare against the normalized golden.
        if let (Some(norm), Curve2D::Line(l)) = (&case.normalized, &pcurve) {
            assert_close(l.origin().x, norm.origin[0]);
            assert_close(l.origin().y, norm.origin[1]);
            assert_close(l.direction().x, norm.direction[0]);
            assert_close(l.direction().y, norm.direction[1]);
        }
    }
}
