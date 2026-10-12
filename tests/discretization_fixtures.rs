//! Golden fixtures for discretization: the counts, parameters and points
//! of representative curves, so a later optimization cannot silently
//! change the output. Unlike the other fixtures, these record geomcore's
//! own output. After an intended change, regenerate them with
//!
//! ```text
//! UPDATE_DISCRETIZATION_FIXTURES=1 cargo test --test discretization_fixtures
//! ```
//!
//! and review the diff of `tests/fixtures/discretization.json`.

mod common;

use common::{assert_close, load};
use geomcore::discretization::{DiscretizationCriteria, Discretize2D, Discretize3D};
use geomcore::{
    BSplineCurve3D, Circle2D, Circle3D, Ellipse3D, Parabola3D, Point2D, Point3D, TrimmedCurve3D,
    Vector3D,
};
use serde_json::{Value, json};

const FIXTURE: &str = "discretization.json";

/// One discretization: its curve, criteria and output points as
/// coordinate lists (three in 3D, two in 2D).
struct Case {
    name: String,
    criteria: DiscretizationCriteria,
    parameters: Vec<f64>,
    points: Vec<Vec<f64>>,
}

fn criteria() -> [DiscretizationCriteria; 2] {
    let chord = DiscretizationCriteria::new(1e-2).unwrap();
    [chord, chord.with_max_angle_deviation(0.05).unwrap()]
}

fn cases_3d(name: &str, curve: &impl Discretize3D) -> Vec<Case> {
    criteria()
        .into_iter()
        .map(|criteria| {
            let d = curve.discretize(criteria);
            Case {
                name: name.to_owned(),
                criteria,
                parameters: d.parameters,
                points: d.points.iter().map(|p| vec![p.x, p.y, p.z]).collect(),
            }
        })
        .collect()
}

fn cases_2d(name: &str, curve: &impl Discretize2D) -> Vec<Case> {
    criteria()
        .into_iter()
        .map(|criteria| {
            let d = curve.discretize(criteria);
            Case {
                name: name.to_owned(),
                criteria,
                parameters: d.parameters,
                points: d.points.iter().map(|p| vec![p.x, p.y]).collect(),
            }
        })
        .collect()
}

/// The discretizations the fixture records, computed now.
fn cases() -> Vec<Case> {
    let normal = Vector3D::new(0.2, -0.3, 1.0);
    let circle = Circle3D::new(Point3D::new(1.0, 2.0, 3.0), normal, 2.0).unwrap();
    let circle_arc = TrimmedCurve3D::new(circle, 0.5, 4.0).unwrap();
    let poles: Vec<Point3D> = (0..7)
        .map(|i| {
            let a = f64::from(i);
            Point3D::new(a, (1.3 * a).sin(), 0.4 * a.cos())
        })
        .collect();
    let knots = vec![0.0, 1.0, 2.5, 3.0, 4.0];
    let multiplicities = vec![4, 1, 1, 1, 4];
    let nonrational = BSplineCurve3D::new(
        3,
        poles.clone(),
        knots.clone(),
        multiplicities.clone(),
        false,
    )
    .unwrap();
    let weights = vec![1.0, 0.5, 2.0, 1.0, 3.0, 0.7, 1.0];
    let rational =
        BSplineCurve3D::new_rational(3, poles, weights, knots, multiplicities, false).unwrap();
    let x_direction = Vector3D::X.cross(normal);
    let ellipse = Ellipse3D::new(Point3D::ORIGIN, normal, x_direction, 3.0, 1.0).unwrap();
    let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 0.5).unwrap();
    let parabola_arc = TrimmedCurve3D::new(parabola, -2.0, 3.0).unwrap();
    let circle_2d = Circle2D::new(Point2D::new(0.5, -1.0), 1.5).unwrap();
    [
        cases_3d("circle_arc", &circle_arc),
        cases_3d("nonrational_bspline", &nonrational),
        cases_3d("rational_bspline", &rational),
        cases_3d("ellipse", &ellipse),
        cases_3d("trimmed_parabola", &parabola_arc),
        cases_2d("circle_2d", &circle_2d),
    ]
    .into_iter()
    .flatten()
    .collect()
}

fn to_json(cases: &[Case]) -> Value {
    let cases: Vec<Value> = cases
        .iter()
        .map(|case| {
            json!({
                "name": case.name,
                "max_chord_deviation": case.criteria.max_chord_deviation(),
                "max_angle_deviation": case.criteria.max_angle_deviation(),
                "count": case.parameters.len(),
                "parameters": case.parameters,
                "points": case.points,
            })
        })
        .collect();
    json!({
        "_note": "geomcore's own discretization output; regenerate with UPDATE_DISCRETIZATION_FIXTURES=1 cargo test --test discretization_fixtures",
        "cases": cases,
    })
}

#[test]
fn test_discretizations_match_the_fixtures() {
    let cases = cases();
    if std::env::var_os("UPDATE_DISCRETIZATION_FIXTURES").is_some() {
        let path = format!("{}/tests/fixtures/{FIXTURE}", env!("CARGO_MANIFEST_DIR"));
        let text = serde_json::to_string_pretty(&to_json(&cases)).unwrap();
        std::fs::write(path, text + "\n").unwrap();
        return;
    }
    let fixture = load(FIXTURE);
    let expected = fixture["cases"].as_array().unwrap();
    assert_eq!(expected.len(), cases.len(), "number of cases");
    for (case, expected) in cases.iter().zip(expected) {
        let name = &case.name;
        assert_eq!(expected["name"], json!(name));
        assert_eq!(
            expected["max_chord_deviation"],
            json!(case.criteria.max_chord_deviation()),
            "{name}"
        );
        assert_eq!(
            expected["max_angle_deviation"],
            json!(case.criteria.max_angle_deviation()),
            "{name}"
        );
        assert_eq!(
            expected["count"],
            json!(case.parameters.len()),
            "{name}: count"
        );
        let parameters = expected["parameters"].as_array().unwrap();
        for (actual, expected) in case.parameters.iter().zip(parameters) {
            assert_close(*actual, expected.as_f64().unwrap());
        }
        let points = expected["points"].as_array().unwrap();
        for (actual, expected) in case.points.iter().zip(points) {
            let expected = expected.as_array().unwrap();
            assert_eq!(actual.len(), expected.len(), "{name}: dimension");
            for (actual, expected) in actual.iter().zip(expected) {
                assert_close(*actual, expected.as_f64().unwrap());
            }
        }
    }
}
