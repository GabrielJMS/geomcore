//! Golden-fixture replay tests: build geomrust values from recorded JSON
//! specs and check parity against recorded expected outputs. Grows with
//! every fixture-replay task.

mod common;

use common::{
    FrameJson, assert_close, assert_point3, assert_vector3, frame3, load, point3, vector3,
};
use geomrust::{Axis3, Circle3D, Ellipse3D, Hyperbola3D, Line3D, Parabola3D, Transform};
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
