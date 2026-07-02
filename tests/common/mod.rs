//! Shared helpers for golden-fixture replay tests: JSON loading and
//! geomcore-type conversions, reused by every `tests/vs_fixtures.rs`-style
//! integration test.
#![allow(dead_code)] // not every helper is used by every fixture-replay task

use serde::Deserialize;

/// A frame recorded in JSON fixtures: origin plus x/y/z unit directions.
#[derive(Deserialize)]
pub struct FrameJson {
    /// Frame origin.
    pub origin: [f64; 3],
    /// Frame x direction.
    pub x_dir: [f64; 3],
    /// Frame y direction.
    pub y_dir: [f64; 3],
    /// Frame z direction.
    pub z_dir: [f64; 3],
}

/// Converts a `[x, y, z]` array into a [`geomcore::Point3`].
pub fn point3(a: [f64; 3]) -> geomcore::Point3 {
    geomcore::Point3::new(a[0], a[1], a[2])
}

/// Converts a `[x, y, z]` array into a [`geomcore::Vector3`].
pub fn vector3(a: [f64; 3]) -> geomcore::Vector3 {
    geomcore::Vector3::new(a[0], a[1], a[2])
}

/// Converts a [`FrameJson`] into a [`geomcore::Frame3`].
pub fn frame3(f: &FrameJson) -> geomcore::Frame3 {
    geomcore::Frame3::new(point3(f.origin), vector3(f.z_dir), vector3(f.x_dir)).unwrap()
}

/// Loads and parses a fixture file by name from `tests/fixtures/`.
pub fn load(name: &str) -> serde_json::Value {
    serde_json::from_str(
        &std::fs::read_to_string(format!(
            "{}/tests/fixtures/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap(),
    )
    .unwrap()
}

/// Asserts that `actual` is close to `expected` within a relative-or-absolute
/// tolerance of `1e-7 * max(1, |expected|)`.
#[track_caller]
pub fn assert_close(actual: f64, expected: f64) {
    let tol = 1e-7 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tol,
        "actual {actual} != expected {expected}"
    );
}

/// Asserts that `actual` matches the `[x, y, z]` array `expected`,
/// component-wise, within [`assert_close`]'s tolerance.
#[track_caller]
pub fn assert_point3(actual: geomcore::Point3, expected: [f64; 3]) {
    assert_close(actual.x, expected[0]);
    assert_close(actual.y, expected[1]);
    assert_close(actual.z, expected[2]);
}

/// Asserts that `actual` matches the `[x, y, z]` array `expected`,
/// component-wise, within [`assert_close`]'s tolerance.
#[track_caller]
pub fn assert_vector3(actual: geomcore::Vector3, expected: [f64; 3]) {
    assert_close(actual.x, expected[0]);
    assert_close(actual.y, expected[1]);
    assert_close(actual.z, expected[2]);
}
