//! Evaluation and parameter-inversion formulas for elementary analytic
//! curves (line, circle, ellipse, parabola, hyperbola), in both 3D and 2D.
//!
//! Every curve is evaluated relative to a placement (an [`Axis3`]/[`Axis2`]
//! for the line, a [`Frame3`]/[`Frame2`] for the conics): a local origin and
//! one or more unit directions that the parametric formulas are expressed
//! in terms of.

use std::f64::consts::TAU;

use crate::{Axis2, Axis3, Frame2, Frame3, Point2, Point3, Vector2, Vector3};

/// Evaluates the line through `axis` at parameter `u`: `origin + u * direction`.
pub(crate) fn line_d0(axis: &Axis3, u: f64) -> Point3 {
    axis.origin() + u * axis.direction()
}

/// First derivative of the line: the constant direction vector.
pub(crate) fn line_d1(axis: &Axis3) -> Vector3 {
    axis.direction()
}

/// Evaluates the circle of radius `r` in `frame`'s plane at angle `u`:
/// `origin + r*cos(u)*x_dir + r*sin(u)*y_dir`.
pub(crate) fn circle_d0(frame: &Frame3, r: f64, u: f64) -> Point3 {
    let a1 = r * u.cos();
    let a2 = r * u.sin();
    frame.origin() + a1 * frame.x_direction() + a2 * frame.y_direction()
}

/// Derivative of the given `order` (>= 1) of the circle at parameter `u`.
///
/// The derivatives cycle with period 4 in `order`: d1 leads the position by
/// a quarter turn, d2 is the negated position vector, d3 is the negated d1,
/// and d4 (== d0's linear part) repeats the cycle.
pub(crate) fn circle_dn(frame: &Frame3, r: f64, u: f64, order: u32) -> Vector3 {
    debug_assert!(order >= 1, "derivative order must be >= 1");
    let a1 = r * u.cos();
    let a2 = r * u.sin();
    let x = frame.x_direction();
    let y = frame.y_direction();
    match order % 4 {
        1 => -a2 * x + a1 * y,
        2 => -a1 * x - a2 * y,
        3 => a2 * x - a1 * y,
        _ => a1 * x + a2 * y,
    }
}

/// Evaluates the ellipse with semi-major `maj` (along x) and semi-minor
/// `min` (along y) in `frame`'s plane at angle `u`:
/// `origin + maj*cos(u)*x_dir + min*sin(u)*y_dir`.
pub(crate) fn ellipse_d0(frame: &Frame3, maj: f64, min: f64, u: f64) -> Point3 {
    frame.origin() + maj * u.cos() * frame.x_direction() + min * u.sin() * frame.y_direction()
}

/// Derivative of the given `order` (>= 1) of the ellipse at parameter `u`.
///
/// Cycles with period 4 in `order`, mirroring [`circle_dn`] but with
/// independent major/minor scaling on the x/y components.
pub(crate) fn ellipse_dn(frame: &Frame3, maj: f64, min: f64, u: f64, order: u32) -> Vector3 {
    debug_assert!(order >= 1, "derivative order must be >= 1");
    let x = frame.x_direction();
    let y = frame.y_direction();
    let (s, c) = (u.sin(), u.cos());
    match order % 4 {
        1 => -maj * s * x + min * c * y,
        2 => -maj * c * x - min * s * y,
        3 => maj * s * x - min * c * y,
        _ => maj * c * x + min * s * y,
    }
}

/// Evaluates the parabola with focal distance `focal` in `frame`'s plane at
/// parameter `u`: `origin + (u^2 / (4*focal))*x_dir + u*y_dir`.
pub(crate) fn parabola_d0(frame: &Frame3, focal: f64, u: f64) -> Point3 {
    frame.origin() + (u * u / (4.0 * focal)) * frame.x_direction() + u * frame.y_direction()
}

/// Derivative of the given `order` (>= 1) of the parabola at parameter `u`.
///
/// The parabola is a degree-2 polynomial curve: the first derivative is
/// linear in `u`, the second is the constant `x_dir / (2*focal)`, and every
/// derivative of order above 2 vanishes.
pub(crate) fn parabola_dn(frame: &Frame3, focal: f64, u: f64, order: u32) -> Vector3 {
    debug_assert!(order >= 1, "derivative order must be >= 1");
    match order {
        1 => (u / (2.0 * focal)) * frame.x_direction() + frame.y_direction(),
        2 => frame.x_direction() * (1.0 / (2.0 * focal)),
        _ => Vector3::ZERO,
    }
}

/// Evaluates the hyperbola with semi-major `maj` (along x) and semi-minor
/// `min` (along y) in `frame`'s plane at parameter `u`:
/// `origin + maj*cosh(u)*x_dir + min*sinh(u)*y_dir`.
pub(crate) fn hyperbola_d0(frame: &Frame3, maj: f64, min: f64, u: f64) -> Point3 {
    frame.origin() + maj * u.cosh() * frame.x_direction() + min * u.sinh() * frame.y_direction()
}

/// Derivative of the given `order` (>= 1) of the hyperbola at parameter `u`.
///
/// Unlike the trigonometric conics, hyperbolic derivatives do not cycle: odd
/// orders follow the `sinh`/`cosh` pattern of the first derivative, even
/// orders follow the `cosh`/`sinh` pattern of the second derivative (which
/// then repeats for every higher even/odd order, since d/du of sinh is cosh
/// and d/du of cosh is sinh).
pub(crate) fn hyperbola_dn(frame: &Frame3, maj: f64, min: f64, u: f64, order: u32) -> Vector3 {
    debug_assert!(order >= 1, "derivative order must be >= 1");
    let x = frame.x_direction();
    let y = frame.y_direction();
    let (sh, ch) = (u.sinh(), u.cosh());
    if order % 2 == 1 {
        maj * sh * x + min * ch * y
    } else {
        maj * ch * x + min * sh * y
    }
}

/// Recovers the parameter `u` of a point on (or near) the line: the signed
/// projection `(p - origin) . direction`.
pub(crate) fn line_parameter(axis: &Axis3, p: Point3) -> f64 {
    (p - axis.origin()).dot(axis.direction())
}

/// Recovers the angular parameter of a point on (or near) the circle,
/// wrapped into `[0, 2*PI)`.
///
/// Projects `p - origin` onto the circle's plane implicitly by taking the
/// signed angle from `x_dir` to `p - origin`, referenced against `z_dir`.
pub(crate) fn circle_parameter(frame: &Frame3, p: Point3) -> f64 {
    let op = p - frame.origin();
    wrap_to_turn(frame.x_direction().angle_with_ref(op, frame.z_direction()))
}

/// Recovers the angular parameter of a point on (or near) the ellipse,
/// wrapped into `[0, 2*PI)`.
///
/// The point is expressed in the frame's local x/y coordinates, the y
/// coordinate is rescaled by `maj/min` to undo the ellipse's anisotropic
/// scaling (mapping the ellipse back onto a circle of radius `maj`), and the
/// angle is then measured the same way as [`circle_parameter`].
pub(crate) fn ellipse_parameter(frame: &Frame3, maj: f64, min: f64, p: Point3) -> f64 {
    let op = p - frame.origin();
    let nx = op.dot(frame.x_direction());
    let ny = op.dot(frame.y_direction());
    let om = nx * frame.x_direction() + ny * (maj / min) * frame.y_direction();
    wrap_to_turn(frame.x_direction().angle_with_ref(om, frame.z_direction()))
}

/// Recovers the parameter of a point on (or near) the parabola: the
/// projection `(p - origin) . y_dir`.
pub(crate) fn parabola_parameter(frame: &Frame3, p: Point3) -> f64 {
    (p - frame.origin()).dot(frame.y_direction())
}

/// Recovers the parameter of a point on (or near) the hyperbola:
/// `asinh(((p - origin) . y_dir) / min)`.
pub(crate) fn hyperbola_parameter(frame: &Frame3, min: f64, p: Point3) -> f64 {
    (((p - frame.origin()).dot(frame.y_direction())) / min).asinh()
}

/// Wraps `u` into `[first, last)` given the period `last - first`.
///
/// Implements `max(first, u + period * ceil((first - u) / period))`
/// literally: this pulls `u` up to the smallest representative that is
/// `>= first`. At the upper boundary, `u == last` is itself a multiple of
/// the period away from `first` (`(first - last) / period == -1`, an exact
/// integer), so `ceil` is a no-op there and the formula wraps `last` down to
/// `first` rather than leaving it fixed — i.e. the interval really is
/// half-open `[first, last)`, with `last` folding back to `first`.
pub(crate) fn in_period(u: f64, first: f64, last: f64) -> f64 {
    let period = last - first;
    f64::max(first, u + period * ((first - u) / period).ceil())
}

/// Wraps a signed angle in `(-PI, PI]` into `[0, 2*PI)`, treating an exact
/// `2*PI` result as `0`.
fn wrap_to_turn(angle: f64) -> f64 {
    let wrapped = if angle < 0.0 { angle + TAU } else { angle };
    if wrapped >= TAU { 0.0 } else { wrapped }
}

/// Evaluates the 2D line through `axis` at parameter `u`: `origin + u * direction`.
pub(crate) fn line2d_d0(axis: &Axis2, u: f64) -> Point2 {
    axis.origin() + u * axis.direction()
}

/// Evaluates the 2D circle of radius `r` in `frame` at angle `u`:
/// `origin + r*cos(u)*x_dir + r*sin(u)*y_dir`.
pub(crate) fn circle2d_d0(frame: &Frame2, r: f64, u: f64) -> Point2 {
    let a1 = r * u.cos();
    let a2 = r * u.sin();
    frame.origin() + a1 * frame.x_direction() + a2 * frame.y_direction()
}

/// Derivative of the given `order` (>= 1) of the 2D circle at parameter `u`.
///
/// Cycles with period 4 in `order`, exactly as [`circle_dn`] but with the
/// frame's 2D x/y directions (no z component, and no handedness
/// special-casing: the formula uses `x_dir`/`y_dir` as stored).
pub(crate) fn circle2d_dn(frame: &Frame2, r: f64, u: f64, order: u32) -> Vector2 {
    debug_assert!(order >= 1, "derivative order must be >= 1");
    let a1 = r * u.cos();
    let a2 = r * u.sin();
    let x = frame.x_direction();
    let y = frame.y_direction();
    match order % 4 {
        1 => -a2 * x + a1 * y,
        2 => -a1 * x - a2 * y,
        3 => a2 * x - a1 * y,
        _ => a1 * x + a2 * y,
    }
}

/// Recovers the parameter `u` of a point on (or near) the 2D line: the
/// signed projection `(p - origin) . direction`.
pub(crate) fn line2d_parameter(axis: &Axis2, p: Point2) -> f64 {
    (p - axis.origin()).dot(axis.direction())
}

/// Recovers the angular parameter of a point on (or near) the 2D circle,
/// wrapped into `[0, 2*PI)`.
///
/// Uses `atan2` of the point's local coordinates (`p - origin` dotted with
/// `x_dir` and `y_dir` respectively) since [`Vector2`] has no 3D-style
/// signed-angle-with-reference helper; the frame may be direct or indirect
/// (left-handed) and the formula applies unchanged either way.
pub(crate) fn circle2d_parameter(frame: &Frame2, p: Point2) -> f64 {
    let op = p - frame.origin();
    let x = op.dot(frame.x_direction());
    let y = op.dot(frame.y_direction());
    wrap_to_turn(y.atan2(x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn skewed_frame3() -> Frame3 {
        let z = Vector3::Z;
        let x_hint = Vector3::new(1.0, 2.0, 2.0).normalized().unwrap();
        Frame3::new(Point3::new(1.0, -2.0, 0.5), z, x_hint).unwrap()
    }

    fn skewed_frame2() -> Frame2 {
        let x_dir = Vector2::new(3.0, 4.0).normalized().unwrap();
        Frame2::from_x(Point2::new(1.0, -2.0), x_dir).unwrap()
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

    // ---- line ----

    #[test]
    fn test_line_d0_world() {
        let axis = Axis3::new(Point3::ORIGIN, Vector3::X).unwrap();
        assert_point3_close(line_d0(&axis, 3.0), Point3::new(3.0, 0.0, 0.0));
    }

    #[test]
    fn test_line_d1_is_constant_direction() {
        let axis = Axis3::new(Point3::new(1.0, 2.0, 3.0), Vector3::Y).unwrap();
        assert_eq!(line_d1(&axis), Vector3::Y);
    }

    #[test]
    fn test_line_parameter_round_trip_world() {
        let axis = Axis3::new(Point3::ORIGIN, Vector3::X).unwrap();
        for u in [0.3, 2.0, -5.5] {
            let p = line_d0(&axis, u);
            assert!((line_parameter(&axis, p) - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_line_parameter_round_trip_skewed() {
        let axis = Axis3::new(
            Point3::new(1.0, -2.0, 0.5),
            Vector3::new(1.0, 2.0, 2.0).normalized().unwrap(),
        )
        .unwrap();
        for u in [0.3, 2.0, 5.5] {
            let p = line_d0(&axis, u);
            assert!((line_parameter(&axis, p) - u).abs() < 1e-9);
        }
    }

    // ---- circle ----

    #[test]
    fn test_circle_d0_world_zero() {
        assert_point3_close(
            circle_d0(&Frame3::WORLD, 1.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
        );
    }

    #[test]
    fn test_circle_d0_world_quarter_turn() {
        assert_point3_close(
            circle_d0(&Frame3::WORLD, 1.0, PI / 2.0),
            Point3::new(0.0, 1.0, 0.0),
        );
    }

    #[test]
    fn test_circle_dn_order1_at_zero() {
        assert_vector3_close(
            circle_dn(&Frame3::WORLD, 1.0, 0.0, 1),
            Vector3::new(0.0, 1.0, 0.0),
        );
    }

    #[test]
    fn test_circle_dn_order2_at_zero() {
        assert_vector3_close(
            circle_dn(&Frame3::WORLD, 1.0, 0.0, 2),
            Vector3::new(-1.0, 0.0, 0.0),
        );
    }

    #[test]
    fn test_circle_dn_cycles_with_period_4() {
        let frame = skewed_frame3();
        for order in [1u32, 2, 3, 4] {
            let a = circle_dn(&frame, 1.5, 0.7, order);
            let b = circle_dn(&frame, 1.5, 0.7, order + 4);
            assert_vector3_close(a, b);
        }
    }

    #[test]
    fn test_circle_parameter_round_trip_world() {
        for u in [0.3, 2.0, 5.5] {
            let p = circle_d0(&Frame3::WORLD, 2.5, u);
            let recovered = circle_parameter(&Frame3::WORLD, p);
            assert!((recovered - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_circle_parameter_round_trip_skewed() {
        let frame = skewed_frame3();
        for u in [0.3, 2.0, 5.5] {
            let p = circle_d0(&frame, 2.5, u);
            let recovered = circle_parameter(&frame, p);
            assert!((recovered - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_circle_parameter_is_in_zero_to_tau() {
        let frame = skewed_frame3();
        for u in [-2.0, -0.1, 6.0, 100.0] {
            let p = circle_d0(&frame, 1.0, u);
            let recovered = circle_parameter(&frame, p);
            assert!((0.0..TAU).contains(&recovered), "got {recovered}");
        }
    }

    // ---- ellipse ----

    #[test]
    fn test_ellipse_d0_quarter_turn() {
        assert_point3_close(
            ellipse_d0(&Frame3::WORLD, 2.0, 1.0, PI / 2.0),
            Point3::new(0.0, 1.0, 0.0),
        );
    }

    #[test]
    fn test_ellipse_dn_order1_at_zero() {
        assert_vector3_close(
            ellipse_dn(&Frame3::WORLD, 2.0, 1.0, 0.0, 1),
            Vector3::new(0.0, 1.0, 0.0),
        );
    }

    #[test]
    fn test_ellipse_dn_cycles_with_period_4() {
        let frame = skewed_frame3();
        for order in [1u32, 2, 3, 4] {
            let a = ellipse_dn(&frame, 3.0, 1.2, 0.9, order);
            let b = ellipse_dn(&frame, 3.0, 1.2, 0.9, order + 4);
            assert_vector3_close(a, b);
        }
    }

    #[test]
    fn test_ellipse_parameter_round_trip_world() {
        for u in [0.3, 2.0, 5.5] {
            let p = ellipse_d0(&Frame3::WORLD, 2.0, 1.0, u);
            let recovered = ellipse_parameter(&Frame3::WORLD, 2.0, 1.0, p);
            assert!((recovered - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_ellipse_parameter_round_trip_skewed() {
        let frame = skewed_frame3();
        for u in [0.3, 2.0, 5.5] {
            let p = ellipse_d0(&frame, 3.0, 1.2, u);
            let recovered = ellipse_parameter(&frame, 3.0, 1.2, p);
            assert!((recovered - u).abs() < 1e-9);
        }
    }

    // ---- parabola ----

    #[test]
    fn test_parabola_d0_focal_one() {
        assert_point3_close(
            parabola_d0(&Frame3::WORLD, 1.0, 2.0),
            Point3::new(1.0, 2.0, 0.0),
        );
    }

    #[test]
    fn test_parabola_dn_order3_is_zero() {
        assert_vector3_close(parabola_dn(&Frame3::WORLD, 1.0, 2.0, 3), Vector3::ZERO);
    }

    #[test]
    fn test_parabola_dn_order_above_two_is_zero() {
        for order in [3u32, 4, 10] {
            assert_vector3_close(parabola_dn(&Frame3::WORLD, 1.0, 2.0, order), Vector3::ZERO);
        }
    }

    #[test]
    fn test_parabola_parameter_round_trip_world() {
        for u in [0.3, 2.0, -5.5] {
            let p = parabola_d0(&Frame3::WORLD, 1.5, u);
            let recovered = parabola_parameter(&Frame3::WORLD, p);
            assert!((recovered - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_parabola_parameter_round_trip_skewed() {
        let frame = skewed_frame3();
        for u in [0.3, 2.0, -5.5] {
            let p = parabola_d0(&frame, 1.5, u);
            let recovered = parabola_parameter(&frame, p);
            assert!((recovered - u).abs() < 1e-9);
        }
    }

    // ---- hyperbola ----

    #[test]
    fn test_hyperbola_d0_at_zero() {
        assert_point3_close(
            hyperbola_d0(&Frame3::WORLD, 2.0, 1.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        );
    }

    #[test]
    fn test_hyperbola_dn_odd_order_pattern() {
        let frame = Frame3::WORLD;
        let u: f64 = 0.8;
        let expected = 2.0 * u.sinh() * Vector3::X + 1.0 * u.cosh() * Vector3::Y;
        assert_vector3_close(hyperbola_dn(&frame, 2.0, 1.0, u, 1), expected);
        assert_vector3_close(hyperbola_dn(&frame, 2.0, 1.0, u, 3), expected);
    }

    #[test]
    fn test_hyperbola_dn_even_order_pattern() {
        let frame = Frame3::WORLD;
        let u: f64 = 0.8;
        let expected = 2.0 * u.cosh() * Vector3::X + 1.0 * u.sinh() * Vector3::Y;
        assert_vector3_close(hyperbola_dn(&frame, 2.0, 1.0, u, 2), expected);
        assert_vector3_close(hyperbola_dn(&frame, 2.0, 1.0, u, 4), expected);
    }

    #[test]
    fn test_hyperbola_parameter_round_trip_world() {
        for u in [0.3, 2.0, -1.5] {
            let p = hyperbola_d0(&Frame3::WORLD, 2.0, 1.0, u);
            let recovered = hyperbola_parameter(&Frame3::WORLD, 1.0, p);
            assert!((recovered - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_hyperbola_parameter_round_trip_skewed() {
        let frame = skewed_frame3();
        for u in [0.3, 2.0, -1.5] {
            let p = hyperbola_d0(&frame, 2.0, 1.0, u);
            let recovered = hyperbola_parameter(&frame, 1.0, p);
            assert!((recovered - u).abs() < 1e-9);
        }
    }

    // ---- in_period ----

    #[test]
    fn test_in_period_above_last_wraps_down() {
        assert!((in_period(7.0, 0.0, TAU) - (7.0 - TAU)).abs() < 1e-9);
    }

    #[test]
    fn test_in_period_below_first_wraps_up() {
        assert!((in_period(-0.5, 0.0, TAU) - (TAU - 0.5)).abs() < 1e-9);
    }

    #[test]
    fn test_in_period_at_first_is_identity() {
        assert_eq!(in_period(0.0, 0.0, TAU), 0.0);
    }

    #[test]
    fn test_in_period_at_last_boundary_wraps_to_first() {
        // The interval is half-open [first, last): u == last is exactly one
        // period above first, so ceil((first - u) / period) == -1 exactly
        // and the formula folds it back down to first, not left at last.
        assert_eq!(in_period(TAU, 0.0, TAU), 0.0);
    }

    // ---- 2D twins ----

    #[test]
    fn test_line2d_d0_world() {
        let axis = Axis2::new(Point2::ORIGIN, Vector2::X).unwrap();
        let p = line2d_d0(&axis, 3.0);
        assert!((p.x - 3.0).abs() < 1e-9);
        assert!((p.y - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_line2d_parameter_round_trip_world() {
        let axis = Axis2::new(Point2::ORIGIN, Vector2::X).unwrap();
        for u in [0.3, 2.0, -5.5] {
            let p = line2d_d0(&axis, u);
            assert!((line2d_parameter(&axis, p) - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_line2d_parameter_round_trip_skewed() {
        let axis = Axis2::new(
            Point2::new(1.0, -2.0),
            Vector2::new(3.0, 4.0).normalized().unwrap(),
        )
        .unwrap();
        for u in [0.3, 2.0, 5.5] {
            let p = line2d_d0(&axis, u);
            assert!((line2d_parameter(&axis, p) - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_circle2d_d0_world_quarter_turn() {
        let p = circle2d_d0(&Frame2::WORLD, 1.0, PI / 2.0);
        assert!((p.x - 0.0).abs() < 1e-9);
        assert!((p.y - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_circle2d_dn_cycles_with_period_4() {
        let frame = skewed_frame2();
        for order in [1u32, 2, 3, 4] {
            let a = circle2d_dn(&frame, 1.5, 0.7, order);
            let b = circle2d_dn(&frame, 1.5, 0.7, order + 4);
            assert!((a.x - b.x).abs() < 1e-9);
            assert!((a.y - b.y).abs() < 1e-9);
        }
    }

    #[test]
    fn test_circle2d_parameter_round_trip_world() {
        for u in [0.3, 2.0, 5.5] {
            let p = circle2d_d0(&Frame2::WORLD, 2.5, u);
            let recovered = circle2d_parameter(&Frame2::WORLD, p);
            assert!((recovered - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_circle2d_parameter_round_trip_skewed() {
        let frame = skewed_frame2();
        for u in [0.3, 2.0, 5.5] {
            let p = circle2d_d0(&frame, 2.5, u);
            let recovered = circle2d_parameter(&frame, p);
            assert!((recovered - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_circle2d_parameter_round_trip_indirect_frame() {
        // Left-handed (indirect) frame: y_dir is the clockwise perpendicular.
        let x_dir = Vector2::new(1.0, 0.0);
        let frame = Frame2::new(Point2::new(0.5, -1.0), x_dir, -x_dir.perp()).unwrap();
        assert!(!frame.is_direct());
        for u in [0.3, 2.0, 5.5] {
            let p = circle2d_d0(&frame, 1.7, u);
            let recovered = circle2d_parameter(&frame, p);
            assert!((recovered - u).abs() < 1e-9);
        }
    }
}
