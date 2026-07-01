//! Golden-fixture replay tests: build geomrust values from recorded JSON
//! specs and check parity against recorded expected outputs. Grows with
//! every fixture-replay task.

mod common;

use common::{FrameJson, assert_point3, assert_vector3, frame3, load, point3, vector3};
use geomrust::{Axis3, Transform};
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
