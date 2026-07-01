//! Evaluation and parameter-inversion formulas for elementary analytic
//! surfaces (plane, cylinder, cone, sphere, torus).
//!
//! Every surface is evaluated relative to a placement [`Frame3`]: a local
//! origin and three orthonormal directions that the parametric formulas are
//! expressed in terms of. Derivatives are keyed by `(du, dv)` with
//! `1 <= du + dv <= 2`; combinations whose value is identically zero for the
//! given surface (e.g. all second derivatives of a plane) return
//! [`Vector3::ZERO`] rather than being treated as an error.

use crate::curve_math::analytic::wrap_to_turn;
use crate::{Frame3, Point3, Vector3};

// ---- plane ----

/// Evaluates the plane in `frame` at `(u, v)`: `origin + u*x_dir + v*y_dir`.
pub(crate) fn plane_d0(frame: &Frame3, u: f64, v: f64) -> Point3 {
    frame.point_at(u, v, 0.0)
}

/// Derivative of the plane at order `(du, dv)`, `1 <= du + dv <= 2`.
///
/// `Su = x_dir`, `Sv = y_dir`; every second derivative (`Suu`, `Svv`, `Suv`)
/// is identically zero since the plane is linear in `u` and `v`.
pub(crate) fn plane_derivative(frame: &Frame3, du: u32, dv: u32) -> Vector3 {
    debug_assert!((1..=2).contains(&(du + dv)));
    match (du, dv) {
        (1, 0) => frame.x_direction(),
        (0, 1) => frame.y_direction(),
        _ => Vector3::ZERO,
    }
}

/// Recovers `(u, v)` of a point on (or near) the plane: its local x/y
/// coordinates in `frame`.
pub(crate) fn plane_parameters(frame: &Frame3, p: Point3) -> (f64, f64) {
    let (x, y, _) = frame.local_coordinates(p);
    (x, y)
}

// ---- cylinder ----

/// Evaluates the cylinder of radius `r` in `frame` at `(u, v)`:
/// `origin + r*cos(u)*x_dir + r*sin(u)*y_dir + v*z_dir`.
pub(crate) fn cylinder_d0(frame: &Frame3, r: f64, u: f64, v: f64) -> Point3 {
    let a1 = r * u.cos();
    let a2 = r * u.sin();
    frame.point_at(a1, a2, v)
}

/// Derivative of the cylinder at order `(du, dv)`, `1 <= du + dv <= 2`.
///
/// `Su = -a2*x_dir + a1*y_dir`, `Sv = z_dir`, `Suu = -a1*x_dir - a2*y_dir`;
/// `Svv` and `Suv` are identically zero (height `v` and angle `u` are
/// independent, and `Sv` is the constant `z_dir`).
pub(crate) fn cylinder_derivative(
    frame: &Frame3,
    r: f64,
    u: f64,
    _v: f64,
    du: u32,
    dv: u32,
) -> Vector3 {
    debug_assert!((1..=2).contains(&(du + dv)));
    let a1 = r * u.cos();
    let a2 = r * u.sin();
    let x = frame.x_direction();
    let y = frame.y_direction();
    match (du, dv) {
        (1, 0) => -a2 * x + a1 * y,
        (0, 1) => frame.z_direction(),
        (2, 0) => -a1 * x - a2 * y,
        _ => Vector3::ZERO,
    }
}

/// Recovers `(u, v)` of a point on (or near) the cylinder: `u = atan2(y, x)`
/// wrapped into `[0, 2*PI)`, `v = z`, in `frame`'s local coordinates.
pub(crate) fn cylinder_parameters(frame: &Frame3, p: Point3) -> (f64, f64) {
    let (x, y, z) = frame.local_coordinates(p);
    (wrap_to_turn(y.atan2(x)), z)
}

// ---- cone ----

/// Evaluates the cone of reference radius `ref_r` and semi-angle
/// `semi_angle` in `frame` at `(u, v)`: `R = ref_r + v*sin(semi_angle)`,
/// `origin + R*cos(u)*x_dir + R*sin(u)*y_dir + v*cos(semi_angle)*z_dir`.
pub(crate) fn cone_d0(frame: &Frame3, ref_r: f64, semi_angle: f64, u: f64, v: f64) -> Point3 {
    let r = ref_r + v * semi_angle.sin();
    let a3 = v * semi_angle.cos();
    frame.point_at(r * u.cos(), r * u.sin(), a3)
}

/// Derivative of the cone at order `(du, dv)`, `1 <= du + dv <= 2`.
///
/// `Su = -R*sin(u)*x_dir + R*cos(u)*y_dir`,
/// `Sv = sin(semi_angle)*(cos(u)*x_dir + sin(u)*y_dir) + cos(semi_angle)*z_dir`,
/// `Suu = -R*cos(u)*x_dir - R*sin(u)*y_dir`,
/// `Suv = sin(semi_angle)*(-sin(u)*x_dir + cos(u)*y_dir)`; `Svv` is
/// identically zero (`R` is linear in `v`, so `Sv` does not depend on `v`).
pub(crate) fn cone_derivative(
    frame: &Frame3,
    ref_r: f64,
    semi_angle: f64,
    u: f64,
    v: f64,
    du: u32,
    dv: u32,
) -> Vector3 {
    debug_assert!((1..=2).contains(&(du + dv)));
    let r = ref_r + v * semi_angle.sin();
    let x = frame.x_direction();
    let y = frame.y_direction();
    let (su, cu) = (u.sin(), u.cos());
    match (du, dv) {
        (1, 0) => -r * su * x + r * cu * y,
        (0, 1) => semi_angle.sin() * (cu * x + su * y) + semi_angle.cos() * frame.z_direction(),
        (2, 0) => -r * cu * x - r * su * y,
        (1, 1) => semi_angle.sin() * (-su * x + cu * y),
        _ => Vector3::ZERO,
    }
}

/// Recovers `(u, v)` of a point on (or near) the cone.
///
/// `u` is `0` exactly on the axis (`x, y` both `~0`); below the apex
/// (`-ref_r > z*tan(semi_angle)`) the point lies on the opposite nappe, so
/// `u` is measured from the branch `atan2(-y, -x)` instead of the usual
/// `atan2(y, x)`; both are wrapped into `[0, 2*PI)`.
/// `v = sin(semi_angle)*(x*cos(u) + y*sin(u) - ref_r) + cos(semi_angle)*z`.
pub(crate) fn cone_parameters(
    frame: &Frame3,
    ref_r: f64,
    semi_angle: f64,
    p: Point3,
) -> (f64, f64) {
    let (x, y, z) = frame.local_coordinates(p);
    let u = if x.abs() < 1e-9 && y.abs() < 1e-9 {
        0.0
    } else if -ref_r > z * semi_angle.tan() {
        wrap_to_turn((-y).atan2(-x))
    } else {
        wrap_to_turn(y.atan2(x))
    };
    let v = semi_angle.sin() * (x * u.cos() + y * u.sin() - ref_r) + semi_angle.cos() * z;
    (u, v)
}

// ---- sphere ----

/// Evaluates the sphere of radius `r` in `frame` at `(u, v)`:
/// `Rcv = r*cos(v)`, `origin + Rcv*cos(u)*x_dir + Rcv*sin(u)*y_dir + r*sin(v)*z_dir`.
pub(crate) fn sphere_d0(frame: &Frame3, r: f64, u: f64, v: f64) -> Point3 {
    let rcv = r * v.cos();
    frame.point_at(rcv * u.cos(), rcv * u.sin(), r * v.sin())
}

/// Derivative of the sphere at order `(du, dv)`, `1 <= du + dv <= 2`.
///
/// `Su = Rcv*(-sin(u)*x_dir + cos(u)*y_dir)`,
/// `Sv = -r*sin(v)*(cos(u)*x_dir + sin(u)*y_dir) + r*cos(v)*z_dir`,
/// `Suu = -Rcv*(cos(u)*x_dir + sin(u)*y_dir)`,
/// `Svv = -r*cos(v)*(cos(u)*x_dir + sin(u)*y_dir) - r*sin(v)*z_dir`,
/// `Suv = -r*sin(v)*(-sin(u)*x_dir + cos(u)*y_dir)`.
pub(crate) fn sphere_derivative(
    frame: &Frame3,
    r: f64,
    u: f64,
    v: f64,
    du: u32,
    dv: u32,
) -> Vector3 {
    debug_assert!((1..=2).contains(&(du + dv)));
    let x = frame.x_direction();
    let y = frame.y_direction();
    let z = frame.z_direction();
    let (su, cu) = (u.sin(), u.cos());
    let (sv, cv) = (v.sin(), v.cos());
    let rcv = r * cv;
    match (du, dv) {
        (1, 0) => rcv * (-su * x + cu * y),
        (0, 1) => -r * sv * (cu * x + su * y) + rcv * z,
        (2, 0) => -rcv * (cu * x + su * y),
        (0, 2) => -rcv * (cu * x + su * y) - r * sv * z,
        (1, 1) => -r * sv * (-su * x + cu * y),
        _ => Vector3::ZERO,
    }
}

/// Recovers `(u, v)` of a point on (or near) the sphere.
///
/// At the poles (`hypot(x, y) < 1e-9`), `u = 0` and `v = +-PI/2` by the sign
/// of `z`; otherwise `u = atan2(y, x)` wrapped into `[0, 2*PI)` and
/// `v = atan(z / hypot(x, y))`.
pub(crate) fn sphere_parameters(frame: &Frame3, _r: f64, p: Point3) -> (f64, f64) {
    let (x, y, z) = frame.local_coordinates(p);
    let l = x.hypot(y);
    if l < 1e-9 {
        (
            0.0,
            if z >= 0.0 {
                std::f64::consts::FRAC_PI_2
            } else {
                -std::f64::consts::FRAC_PI_2
            },
        )
    } else {
        (wrap_to_turn(y.atan2(x)), (z / l).atan())
    }
}

// ---- torus ----

/// Evaluates the torus of major radius `maj` and minor radius `min` in
/// `frame` at `(u, v)`: `R = maj + min*cos(v)`,
/// `origin + R*cos(u)*x_dir + R*sin(u)*y_dir + min*sin(v)*z_dir`.
pub(crate) fn torus_d0(frame: &Frame3, maj: f64, min: f64, u: f64, v: f64) -> Point3 {
    let r = maj + min * v.cos();
    frame.point_at(r * u.cos(), r * u.sin(), min * v.sin())
}

/// Derivative of the torus at order `(du, dv)`, `1 <= du + dv <= 2`.
///
/// `Su = R*(-sin(u)*x_dir + cos(u)*y_dir)`,
/// `Sv = -min*sin(v)*(cos(u)*x_dir + sin(u)*y_dir) + min*cos(v)*z_dir`,
/// `Suu = -R*(cos(u)*x_dir + sin(u)*y_dir)`,
/// `Svv = -min*cos(v)*(cos(u)*x_dir + sin(u)*y_dir) - min*sin(v)*z_dir`,
/// `Suv = min*sin(v)*(sin(u)*x_dir - cos(u)*y_dir)`.
pub(crate) fn torus_derivative(
    frame: &Frame3,
    maj: f64,
    min: f64,
    u: f64,
    v: f64,
    du: u32,
    dv: u32,
) -> Vector3 {
    debug_assert!((1..=2).contains(&(du + dv)));
    let x = frame.x_direction();
    let y = frame.y_direction();
    let z = frame.z_direction();
    let (su, cu) = (u.sin(), u.cos());
    let (sv, cv) = (v.sin(), v.cos());
    let r = maj + min * cv;
    match (du, dv) {
        (1, 0) => r * (-su * x + cu * y),
        (0, 1) => -min * sv * (cu * x + su * y) + min * cv * z,
        (2, 0) => -r * (cu * x + su * y),
        (0, 2) => -min * cv * (cu * x + su * y) - min * sv * z,
        (1, 1) => min * sv * (su * x - cu * y),
        _ => Vector3::ZERO,
    }
}

/// Recovers `(u, v)` of a point on (or near) the torus.
///
/// `u = atan2(y, x)` wrapped into `[0, 2*PI)`. When `maj < min` the tube can
/// swallow the axis, so `atan2(y, x)` alone no longer distinguishes the near
/// and far side of the tube; the branch (`u` vs `u + PI`) whose centerline
/// point `maj*(cos, sin, 0)` sits at distance closest to the true minor
/// radius `min` from `p` (measured in full 3D, not just the `(x, y)`
/// projection — `z` matters once the tube surrounds the axis) is picked
/// instead. `dP = (x - maj*cos(u), y - maj*sin(u), z)` is then the offset
/// from the tube's circular axis at that `u`, expressed in `frame`'s local
/// x/y/z coordinates. `dP` lies entirely in the meridian half-plane spanned
/// by the radial direction `cos(u)*x_dir + sin(u)*y_dir` and `z_dir` (the two
/// are orthonormal), so `v` is simply `atan2(dP . z_dir, dP . radial)`,
/// wrapped into `[0, 2*PI)`.
pub(crate) fn torus_parameters(frame: &Frame3, maj: f64, min: f64, p: Point3) -> (f64, f64) {
    let (x, y, z) = frame.local_coordinates(p);
    // 3D distance from `p` to the tube's centerline circle (radius `maj`,
    // in the frame's xy-plane) at angle `candidate_u`. On the true branch
    // this distance is exactly `min`; on the wrong branch it generally
    // isn't, so comparing the two candidates' *offset from `min`* (not
    // their raw magnitude) picks the correct one.
    let offset_from_minor_radius = |candidate_u: f64| {
        let dx = x - maj * candidate_u.cos();
        let dy = y - maj * candidate_u.sin();
        (dx.hypot(dy).hypot(z) - min).abs()
    };
    let mut u = y.atan2(x);
    if maj < min {
        let alt = u + std::f64::consts::PI;
        if offset_from_minor_radius(alt) < offset_from_minor_radius(u) {
            u = alt;
        }
    }
    let u = wrap_to_turn(u);
    let dp_radial = x * u.cos() + y * u.sin() - maj;
    let v = wrap_to_turn(z.atan2(dp_radial));
    (u, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, TAU};

    fn skewed_frame() -> Frame3 {
        let z = Vector3::Z;
        let x_hint = Vector3::new(1.0, 2.0, 2.0).normalized().unwrap();
        Frame3::new(Point3::new(1.0, -2.0, 0.5), z, x_hint).unwrap()
    }

    fn assert_point3_close(actual: Point3, expected: Point3) {
        assert!(
            (actual.x - expected.x).abs() < 1e-9,
            "x: {actual:?} vs {expected:?}"
        );
        assert!(
            (actual.y - expected.y).abs() < 1e-9,
            "y: {actual:?} vs {expected:?}"
        );
        assert!(
            (actual.z - expected.z).abs() < 1e-9,
            "z: {actual:?} vs {expected:?}"
        );
    }

    fn assert_vector3_close(actual: Vector3, expected: Vector3) {
        assert!(
            (actual.x - expected.x).abs() < 1e-9,
            "x: {actual:?} vs {expected:?}"
        );
        assert!(
            (actual.y - expected.y).abs() < 1e-9,
            "y: {actual:?} vs {expected:?}"
        );
        assert!(
            (actual.z - expected.z).abs() < 1e-9,
            "z: {actual:?} vs {expected:?}"
        );
    }

    // ---- plane ----

    #[test]
    fn test_plane_d0_world() {
        assert_point3_close(
            plane_d0(&Frame3::WORLD, 2.0, 3.0),
            Point3::new(2.0, 3.0, 0.0),
        );
    }

    #[test]
    fn test_plane_derivative_first_orders() {
        assert_vector3_close(plane_derivative(&Frame3::WORLD, 1, 0), Vector3::X);
        assert_vector3_close(plane_derivative(&Frame3::WORLD, 0, 1), Vector3::Y);
    }

    #[test]
    fn test_plane_derivative_second_orders_are_zero() {
        for (du, dv) in [(2, 0), (0, 2), (1, 1)] {
            assert_vector3_close(plane_derivative(&Frame3::WORLD, du, dv), Vector3::ZERO);
        }
    }

    #[test]
    fn test_plane_parameters_round_trip_skewed() {
        let frame = skewed_frame();
        for (u, v) in [(0.3, 2.0), (5.5, -1.2)] {
            let p = plane_d0(&frame, u, v);
            let (ru, rv) = plane_parameters(&frame, p);
            assert!((ru - u).abs() < 1e-9);
            assert!((rv - v).abs() < 1e-9);
        }
    }

    // ---- cylinder ----

    #[test]
    fn test_cylinder_d0_world_hand_computed() {
        assert_point3_close(
            cylinder_d0(&Frame3::WORLD, 2.0, 0.0, 5.0),
            Point3::new(2.0, 0.0, 5.0),
        );
        assert_point3_close(
            cylinder_d0(&Frame3::WORLD, 2.0, PI / 2.0, 1.0),
            Point3::new(0.0, 2.0, 1.0),
        );
    }

    #[test]
    fn test_cylinder_su_at_zero() {
        assert_vector3_close(
            cylinder_derivative(&Frame3::WORLD, 2.0, 0.0, 5.0, 1, 0),
            Vector3::new(0.0, 2.0, 0.0),
        );
    }

    #[test]
    fn test_cylinder_sv_is_z_dir() {
        assert_vector3_close(
            cylinder_derivative(&Frame3::WORLD, 2.0, 0.7, 5.0, 0, 1),
            Vector3::Z,
        );
    }

    #[test]
    fn test_cylinder_svv_and_suv_are_zero() {
        assert_vector3_close(
            cylinder_derivative(&Frame3::WORLD, 2.0, 0.7, 5.0, 0, 2),
            Vector3::ZERO,
        );
        assert_vector3_close(
            cylinder_derivative(&Frame3::WORLD, 2.0, 0.7, 5.0, 1, 1),
            Vector3::ZERO,
        );
    }

    #[test]
    fn test_cylinder_parameters_round_trip_skewed() {
        let frame = skewed_frame();
        for (u, v) in [(0.3, 2.0), (5.5, -1.2), (PI / 2.0, 0.0)] {
            let p = cylinder_d0(&frame, 2.5, u, v);
            let (ru, rv) = cylinder_parameters(&frame, p);
            assert!((ru - u).abs() < 1e-9, "u: {ru} vs {u}");
            assert!((rv - v).abs() < 1e-9, "v: {rv} vs {v}");
        }
    }

    #[test]
    fn test_cylinder_parameters_u_in_zero_to_tau() {
        let frame = skewed_frame();
        for u in [-2.0, -0.1, 6.0] {
            let p = cylinder_d0(&frame, 1.0, u, 0.5);
            let (ru, _) = cylinder_parameters(&frame, p);
            assert!((0.0..TAU).contains(&ru), "got {ru}");
        }
    }

    // ---- cone ----

    #[test]
    fn test_cone_d0_world_hand_computed() {
        assert_point3_close(
            cone_d0(&Frame3::WORLD, 2.0, 0.4, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        );
    }

    #[test]
    fn test_cone_parameters_round_trip_skewed() {
        let frame = skewed_frame();
        for (u, v) in [(0.3, 1.0), (5.5, -0.5), (2.0, 2.0)] {
            let p = cone_d0(&frame, 2.0, 0.4, u, v);
            let (ru, rv) = cone_parameters(&frame, 2.0, 0.4, p);
            assert!((ru - u).abs() < 1e-7, "u: {ru} vs {u}");
            assert!((rv - v).abs() < 1e-7, "v: {rv} vs {v}");
        }
    }

    #[test]
    fn test_cone_parameters_on_axis_gives_u_zero() {
        let frame = Frame3::WORLD;
        // A point on the axis itself (x, y both zero) picks u = 0.
        let p = frame.point_at(0.0, 0.0, 3.0);
        let (u, _) = cone_parameters(&frame, 2.0, 0.4, p);
        assert_eq!(u, 0.0);
    }

    // ---- sphere ----

    #[test]
    fn test_sphere_d0_world_pole() {
        assert_point3_close(
            sphere_d0(&Frame3::WORLD, 3.0, 0.0, PI / 2.0),
            Point3::new(0.0, 0.0, 3.0),
        );
    }

    #[test]
    fn test_sphere_parameters_round_trip_skewed() {
        let frame = skewed_frame();
        for (u, v) in [(0.3, 0.5), (5.5, -0.8), (2.0, 1.0)] {
            let p = sphere_d0(&frame, 3.0, u, v);
            let (ru, rv) = sphere_parameters(&frame, 3.0, p);
            assert!((ru - u).abs() < 1e-9, "u: {ru} vs {u}");
            assert!((rv - v).abs() < 1e-9, "v: {rv} vs {v}");
        }
    }

    #[test]
    fn test_sphere_parameters_pole_north() {
        let frame = skewed_frame();
        let p = sphere_d0(&frame, 3.0, 0.0, PI / 2.0);
        let (u, v) = sphere_parameters(&frame, 3.0, p);
        assert_eq!(u, 0.0);
        assert!((v - PI / 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_sphere_parameters_pole_south() {
        let frame = skewed_frame();
        let p = sphere_d0(&frame, 3.0, 0.0, -PI / 2.0);
        let (u, v) = sphere_parameters(&frame, 3.0, p);
        assert_eq!(u, 0.0);
        assert!((v + PI / 2.0).abs() < 1e-9);
    }

    // ---- torus ----

    #[test]
    fn test_torus_d0_world_hand_computed() {
        assert_point3_close(
            torus_d0(&Frame3::WORLD, 5.0, 1.5, 0.0, 0.0),
            Point3::new(6.5, 0.0, 0.0),
        );
        assert_point3_close(
            torus_d0(&Frame3::WORLD, 5.0, 1.5, PI, PI),
            Point3::new(-3.5, 0.0, 0.0),
        );
    }

    #[test]
    fn test_torus_parameters_round_trip_skewed() {
        let frame = skewed_frame();
        for (u, v) in [(0.3, 1.0), (5.5, 0.5), (2.0, 4.0)] {
            let p = torus_d0(&frame, 5.0, 1.5, u, v);
            let (ru, rv) = torus_parameters(&frame, 5.0, 1.5, p);
            assert!((ru - u).abs() < 1e-9, "u: {ru} vs {u}");
            assert!((rv - v).abs() < 1e-9, "v: {rv} vs {v}");
        }
    }

    #[test]
    fn test_torus_parameters_major_less_than_minor_branch() {
        // maj = 1, min = 2.5: the tube swallows the axis, so the near/far
        // branch pick matters for recovering u correctly.
        let frame = Frame3::WORLD;
        let (u, v) = (0.7, 2.0);
        let p = torus_d0(&frame, 1.0, 2.5, u, v);
        let (ru, rv) = torus_parameters(&frame, 1.0, 2.5, p);
        assert!((ru - u).abs() < 1e-9, "u: {ru} vs {u}");
        assert!((rv - v).abs() < 1e-9, "v: {rv} vs {v}");
    }

    // ---- fixture replay: the correctness gate ----

    #[test]
    fn test_fixture_replay_all_cases() {
        replay_fixture();
    }

    // --- fixture parsing (minimal, no serde derive to keep it explicit) ---

    use serde_json::Value;

    fn replay_fixture() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/surfaces_analytic.json"
        );
        let text = std::fs::read_to_string(path).expect("read fixture");
        let json: Value = serde_json::from_str(&text).expect("parse fixture");

        for case in json["planes"].as_array().unwrap() {
            replay_case(
                case,
                plane_d0,
                |frame, du, dv, _, _| plane_derivative(frame, du, dv),
                plane_parameters,
            );
        }
        for case in json["cylinders"].as_array().unwrap() {
            let r = case["radius"].as_f64().unwrap();
            replay_case(
                case,
                move |frame, u, v| cylinder_d0(frame, r, u, v),
                move |frame, du, dv, u, v| cylinder_derivative(frame, r, u, v, du, dv),
                cylinder_parameters,
            );
        }
        for case in json["cones"].as_array().unwrap() {
            let ref_r = case["ref_radius"].as_f64().unwrap();
            let semi_angle = case["semi_angle"].as_f64().unwrap();
            replay_case(
                case,
                move |frame, u, v| cone_d0(frame, ref_r, semi_angle, u, v),
                move |frame, du, dv, u, v| cone_derivative(frame, ref_r, semi_angle, u, v, du, dv),
                move |frame, p| cone_parameters(frame, ref_r, semi_angle, p),
            );
        }
        for case in json["spheres"].as_array().unwrap() {
            let r = case["radius"].as_f64().unwrap();
            replay_case(
                case,
                move |frame, u, v| sphere_d0(frame, r, u, v),
                move |frame, du, dv, u, v| sphere_derivative(frame, r, u, v, du, dv),
                move |frame, p| sphere_parameters(frame, r, p),
            );
        }
        for case in json["tori"].as_array().unwrap() {
            let maj = case["major_radius"].as_f64().unwrap();
            let min = case["minor_radius"].as_f64().unwrap();
            replay_case(
                case,
                move |frame, u, v| torus_d0(frame, maj, min, u, v),
                move |frame, du, dv, u, v| torus_derivative(frame, maj, min, u, v, du, dv),
                move |frame, p| torus_parameters(frame, maj, min, p),
            );
        }
    }

    fn arr_f64(v: &Value) -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    }

    fn point3_from(v: &Value) -> Point3 {
        let a = arr_f64(v);
        Point3::new(a[0], a[1], a[2])
    }

    fn vector3_from(v: &Value) -> Vector3 {
        let a = arr_f64(v);
        Vector3::new(a[0], a[1], a[2])
    }

    fn frame_from(v: &Value) -> Frame3 {
        let origin = point3_from(&v["origin"]);
        let x_dir = vector3_from(&v["x_dir"]);
        let z_dir = vector3_from(&v["z_dir"]);
        // Fixture frames are already orthonormal right-handed, so
        // reconstructing via `Frame3::new` with the stored x/z directions
        // reproduces the stored y_dir exactly (z x_hint projection is a
        // no-op when x_hint is already perpendicular to z).
        Frame3::new(origin, z_dir, x_dir).expect("fixture frame is well-formed")
    }

    fn assert_vector3_tol(actual: Vector3, expected: Vector3, ctx: &str) {
        for (a, e, axis) in [
            (actual.x, expected.x, "x"),
            (actual.y, expected.y, "y"),
            (actual.z, expected.z, "z"),
        ] {
            let tol = 1e-7 * f64::max(1.0, e.abs());
            assert!(
                (a - e).abs() <= tol,
                "{ctx} {axis}: got {a}, expected {e} (tol {tol})"
            );
        }
    }

    fn replay_case(
        case: &Value,
        d0: impl Fn(&Frame3, f64, f64) -> Point3,
        derivative: impl Fn(&Frame3, u32, u32, f64, f64) -> Vector3,
        parameters: impl Fn(&Frame3, Point3) -> (f64, f64),
    ) {
        let frame = frame_from(&case["frame"]);
        for sample in case["samples"].as_array().unwrap() {
            let u = sample["u"].as_f64().unwrap();
            let v = sample["v"].as_f64().unwrap();
            let ctx = format!("u={u} v={v}");

            let point = d0(&frame, u, v);
            assert_vector3_tol(
                Vector3::new(point.x, point.y, point.z),
                vector3_from(&sample["point"]),
                &format!("{ctx} point"),
            );

            for (key, du, dv) in [
                ("d1u", 1u32, 0u32),
                ("d1v", 0, 1),
                ("d2u", 2, 0),
                ("d2v", 0, 2),
                ("d2uv", 1, 1),
            ] {
                let got = derivative(&frame, du, dv, u, v);
                assert_vector3_tol(got, vector3_from(&sample[key]), &format!("{ctx} {key}"));
            }
        }

        for pair in case["parameters_of"].as_array().unwrap() {
            let p = point3_from(&pair["point"]);
            let expected_u = pair["u"].as_f64().unwrap();
            let expected_v = pair["v"].as_f64().unwrap();
            let (u, v) = parameters(&frame, p);
            let tol_u = 1e-7 * f64::max(1.0, expected_u.abs());
            let tol_v = 1e-7 * f64::max(1.0, expected_v.abs());
            assert!(
                (u - expected_u).abs() <= tol_u,
                "u: got {u}, expected {expected_u}"
            );
            assert!(
                (v - expected_v).abs() <= tol_v,
                "v: got {v}, expected {expected_v}"
            );
        }
    }
}
