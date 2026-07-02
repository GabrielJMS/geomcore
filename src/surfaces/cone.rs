//! Cones in 3D: parametric evaluation, parameter inversion, and
//! construction from an apex angle or two circular sections, thin wrappers
//! over [`crate::surface_math::analytic`].

use crate::surface_math::analytic;
use crate::tol;
use crate::{Frame3, Point3, Vector3};
use std::fmt;

/// Error returned when a [`Cone`] cannot be constructed from the given
/// inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConeConstructionError {
    /// The requested reference radius is negative.
    NegativeRadius,
    /// The semi-angle is out of the accepted range (see
    /// [`Cone::from_frame`]).
    BadAngle,
    /// The two points used to derive the axis are coincident (or too close
    /// to distinguish).
    NullAxis,
    /// The derived semi-angle is (numerically) zero or a right angle.
    NullAngle,
    /// The axis direction has zero length.
    NullNormal,
}

impl fmt::Display for ConeConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            ConeConstructionError::NegativeRadius => "radius is negative",
            ConeConstructionError::BadAngle => "semi-angle is out of range",
            ConeConstructionError::NullAxis => "the two points are confused",
            ConeConstructionError::NullAngle => "semi-angle is null or a right angle",
            ConeConstructionError::NullNormal => "axis direction has zero length",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ConeConstructionError {}

/// A cone in 3D: a [`Frame3`] (origin plus local x/y/z directions, `z`
/// being the axis), a semi-angle, and a reference radius (the radius of the
/// circular section through the origin), evaluated as
/// `R = ref_radius + v*sin(semi_angle)`,
/// `origin + R*cos(u)*x_dir + R*sin(u)*y_dir + v*cos(semi_angle)*z_dir`.
///
/// # Examples
///
/// ```
/// use geomcore::{Cone, Point3, Vector3};
/// let cone = Cone::new(Point3::ORIGIN, Vector3::Z, 0.4, 2.0).unwrap();
/// assert_eq!(cone.eval_point(0.0, 0.0), Point3::new(2.0, 0.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cone {
    frame: Frame3,
    semi_angle: f64,
    ref_radius: f64,
}

impl Cone {
    /// Creates a cone from a frame, a semi-angle, and a reference radius.
    ///
    /// The semi-angle must satisfy `0 < |semi_angle| < PI/2` (both bounds
    /// excluded with `tol::ANGULAR` tolerance); a negative
    /// semi-angle flips the direction the cone opens in along `z`.
    ///
    /// # Errors
    ///
    /// Returns [`ConeConstructionError::BadAngle`] if `semi_angle` is
    /// outside the accepted range, or
    /// [`ConeConstructionError::NegativeRadius`] if `ref_radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cone, Frame3};
    /// let cone = Cone::from_frame(Frame3::WORLD, 0.4, 2.0).unwrap();
    /// assert_eq!(cone.ref_radius(), 2.0);
    /// ```
    pub fn from_frame(
        frame: Frame3,
        semi_angle: f64,
        ref_radius: f64,
    ) -> Result<Cone, ConeConstructionError> {
        if !is_valid_semi_angle(semi_angle) {
            return Err(ConeConstructionError::BadAngle);
        }
        if ref_radius < 0.0 {
            return Err(ConeConstructionError::NegativeRadius);
        }
        Ok(Cone {
            frame,
            semi_angle,
            ref_radius,
        })
    }

    /// Creates a cone from a center, an axis direction, a semi-angle, and a
    /// reference radius.
    ///
    /// The frame is derived from `axis_direction` via [`Frame3::from_z`].
    ///
    /// # Errors
    ///
    /// Returns [`ConeConstructionError::NullNormal`] if `axis_direction`
    /// cannot be normalized (zero length), or any error from
    /// [`Cone::from_frame`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cone, Point3, Vector3};
    /// let cone = Cone::new(Point3::ORIGIN, Vector3::Z, 0.4, 2.0).unwrap();
    /// assert_eq!(cone.semi_angle(), 0.4);
    /// ```
    pub fn new(
        center: Point3,
        axis_direction: Vector3,
        semi_angle: f64,
        ref_radius: f64,
    ) -> Result<Cone, ConeConstructionError> {
        let frame = Frame3::from_z(center, axis_direction)
            .map_err(|_| ConeConstructionError::NullNormal)?;
        Cone::from_frame(frame, semi_angle, ref_radius)
    }

    /// Creates a cone through two circular sections: `p1` with radius `r1`
    /// and `p2` with radius `r2`.
    ///
    /// The axis runs from `p1` to `p2`; the reference radius is `r1`
    /// (the section at `p1`) and the semi-angle is
    /// `atan((r2 - r1) / dist(p1, p2))` (negative when the cone narrows
    /// from `p1` to `p2`). The frame's origin is `p1`, its z direction is
    /// `normalize(p2 - p1)`, and its x direction is an arbitrary vector
    /// perpendicular to `z`, matching the reference implementation's
    /// convention (see `arbitrary_perpendicular_frame`).
    ///
    /// # Errors
    ///
    /// Returns [`ConeConstructionError::NullAxis`] if `p1` and `p2` are
    /// coincident (or too close to distinguish),
    /// [`ConeConstructionError::NegativeRadius`] if `r1 < 0 || r2 < 0`, or
    /// [`ConeConstructionError::NullAngle`] if the derived semi-angle is
    /// (numerically) zero or a right angle.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cone, Point3};
    /// let cone = Cone::from_two_points_and_radii(
    ///     Point3::ORIGIN,
    ///     Point3::new(0.0, 0.0, 4.0),
    ///     2.0,
    ///     1.0,
    /// )
    /// .unwrap();
    /// assert_eq!(cone.ref_radius(), 2.0);
    /// ```
    pub fn from_two_points_and_radii(
        p1: Point3,
        p2: Point3,
        r1: f64,
        r2: f64,
    ) -> Result<Cone, ConeConstructionError> {
        let delta = p2 - p1;
        let dist = delta.magnitude();
        if dist < tol::CONFUSION {
            return Err(ConeConstructionError::NullAxis);
        }
        if r1 < 0.0 || r2 < 0.0 {
            return Err(ConeConstructionError::NegativeRadius);
        }

        let semi_angle = ((r2 - r1) / dist).atan();
        if !is_valid_semi_angle(semi_angle) {
            return Err(ConeConstructionError::NullAngle);
        }

        let z_dir = delta * (1.0 / dist);
        let frame = arbitrary_perpendicular_frame(p1, z_dir);
        Ok(Cone {
            frame,
            semi_angle,
            ref_radius: r1,
        })
    }

    /// Returns the cone's frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cone, Frame3};
    /// let cone = Cone::from_frame(Frame3::WORLD, 0.4, 2.0).unwrap();
    /// assert_eq!(cone.frame(), Frame3::WORLD);
    /// ```
    pub fn frame(&self) -> Frame3 {
        self.frame
    }

    /// Returns the cone's semi-angle.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cone, Frame3};
    /// let cone = Cone::from_frame(Frame3::WORLD, 0.4, 2.0).unwrap();
    /// assert_eq!(cone.semi_angle(), 0.4);
    /// ```
    pub fn semi_angle(&self) -> f64 {
        self.semi_angle
    }

    /// Returns the cone's reference radius (the radius of the circular
    /// section through the origin, at `v = 0`).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cone, Frame3};
    /// let cone = Cone::from_frame(Frame3::WORLD, 0.4, 2.0).unwrap();
    /// assert_eq!(cone.ref_radius(), 2.0);
    /// ```
    pub fn ref_radius(&self) -> f64 {
        self.ref_radius
    }

    /// Returns the apex of the cone: `origin + (-ref_radius /
    /// tan(semi_angle)) * z_dir`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cone, Point3, Vector3};
    /// let cone = Cone::new(Point3::ORIGIN, Vector3::Z, std::f64::consts::FRAC_PI_4, 1.0).unwrap();
    /// let apex = cone.apex();
    /// assert!((apex.z + 1.0).abs() < 1e-9);
    /// ```
    pub fn apex(&self) -> Point3 {
        let v = -self.ref_radius / self.semi_angle.tan();
        self.frame.point_at(0.0, 0.0, v)
    }

    /// Evaluates the point on the cone at `(u, v)`. See the type-level docs
    /// for the formula.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cone, Point3, Vector3};
    /// let cone = Cone::new(Point3::ORIGIN, Vector3::Z, 0.4, 2.0).unwrap();
    /// assert_eq!(cone.eval_point(0.0, 0.0), Point3::new(2.0, 0.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64, v: f64) -> Point3 {
        analytic::cone_d0(&self.frame, self.ref_radius, self.semi_angle, u, v)
    }

    /// Evaluates the points on the cone at each `(u, v)` in `uvs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cone, Point3, Vector3};
    /// let cone = Cone::new(Point3::ORIGIN, Vector3::Z, 0.4, 2.0).unwrap();
    /// let points = cone.eval_points(&[(0.0, 0.0)]);
    /// assert_eq!(points[0], Point3::new(2.0, 0.0, 0.0));
    /// ```
    pub fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3> {
        uvs.iter().map(|&(u, v)| self.eval_point(u, v)).collect()
    }

    /// Evaluates the derivative of order `(du, dv)` at `(u, v)`. See
    /// `surface_math::analytic::cone_derivative` for the formulas.
    ///
    /// # Panics
    ///
    /// Panics if `du + dv == 0` (use [`Cone::eval_point`] for the position
    /// itself) or if `du + dv > 2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cone, Point3, Vector3};
    /// let cone = Cone::new(Point3::ORIGIN, Vector3::Z, 0.4, 2.0).unwrap();
    /// let d1v = cone.eval_derivative(0.0, 0.0, 0, 1);
    /// assert!(d1v.x > 0.0);
    /// ```
    pub fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3 {
        match du + dv {
            0 => panic!(
                "eval_derivative: du + dv must be >= 1 (use eval_point for the (0, 0) order)"
            ),
            1..=2 => analytic::cone_derivative(
                &self.frame,
                self.ref_radius,
                self.semi_angle,
                u,
                v,
                du,
                dv,
            ),
            _ => panic!(
                "eval_derivative: order du={du}, dv={dv} is not supported (du + dv must be <= 2)"
            ),
        }
    }

    /// Recovers `(u, v)` of a point on (or near) the cone. See
    /// `surface_math::analytic::cone_parameters` for the formula,
    /// including the opposite-nappe handling below the apex.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cone, Point3, Vector3};
    /// let cone = Cone::new(Point3::ORIGIN, Vector3::Z, 0.4, 2.0).unwrap();
    /// let (u, v) = cone.parameters_of(Point3::new(2.0, 0.0, 0.0));
    /// assert_eq!((u, v), (0.0, 0.0));
    /// ```
    pub fn parameters_of(&self, point: Point3) -> (f64, f64) {
        analytic::cone_parameters(&self.frame, self.ref_radius, self.semi_angle, point)
    }
}

/// Checks that `semi_angle` is within the accepted range for a cone:
/// `0 < |semi_angle| < PI/2`, both bounds excluded with
/// [`crate::tol::ANGULAR`] tolerance.
fn is_valid_semi_angle(semi_angle: f64) -> bool {
    let angle = semi_angle.abs();
    angle >= tol::ANGULAR && (std::f64::consts::FRAC_PI_2 - angle) >= tol::ANGULAR
}

/// Builds a frame at `origin` with `z_dir = normalize(axis_direction)` and
/// an arbitrary x direction perpendicular to it, matching the reference
/// implementation's convention for cones derived from two points and radii:
/// zero the smallest-magnitude component of `z_dir` and swap the other two.
/// This is the sign-mirror of the analogous helper in `plane.rs` (used for
/// `Plane::from_coefficients`); the two call paths were empirically found
/// (from `construction.json`'s golden cases) to use different underlying
/// "arbitrary perpendicular direction" conventions upstream.
///
/// `axis_direction` must already be a unit vector.
fn arbitrary_perpendicular_frame(origin: Point3, axis_direction: Vector3) -> Frame3 {
    let z = axis_direction;
    let (ax, ay, az) = (z.x.abs(), z.y.abs(), z.z.abs());
    let x_dir = if ax <= ay && ax <= az {
        Vector3::new(0.0, -z.z, z.y)
    } else if ay <= ax && ay <= az {
        Vector3::new(-z.z, 0.0, z.x)
    } else {
        Vector3::new(-z.y, z.x, 0.0)
    }
    .normalized()
    .expect("z is a unit vector and nonzero on at least two axes, so the swap is nonzero");
    Frame3::new(origin, z, x_dir).expect("x_dir constructed perpendicular to z by design")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector3;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

    // ---- construction: from_frame ----

    #[test]
    fn test_from_frame_ok() {
        let c = Cone::from_frame(Frame3::WORLD, 0.4, 2.0).unwrap();
        assert_eq!(c.semi_angle(), 0.4);
        assert_eq!(c.ref_radius(), 2.0);
    }

    #[test]
    fn test_from_frame_negative_angle_ok() {
        let c = Cone::from_frame(Frame3::WORLD, -0.4, 2.0).unwrap();
        assert_eq!(c.semi_angle(), -0.4);
    }

    #[test]
    fn test_from_frame_negative_radius_errors() {
        assert_eq!(
            Cone::from_frame(Frame3::WORLD, 0.4, -1.0),
            Err(ConeConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_from_frame_zero_angle_errors() {
        assert_eq!(
            Cone::from_frame(Frame3::WORLD, 0.0, 1.0),
            Err(ConeConstructionError::BadAngle)
        );
    }

    #[test]
    fn test_from_frame_right_angle_errors() {
        assert_eq!(
            Cone::from_frame(Frame3::WORLD, FRAC_PI_2, 1.0),
            Err(ConeConstructionError::BadAngle)
        );
        assert_eq!(
            Cone::from_frame(Frame3::WORLD, -FRAC_PI_2, 1.0),
            Err(ConeConstructionError::BadAngle)
        );
    }

    #[test]
    fn test_from_frame_valid_boundary_ok() {
        assert!(Cone::from_frame(Frame3::WORLD, FRAC_PI_4, 1.0).is_ok());
    }

    // ---- construction: new ----

    #[test]
    fn test_new_ok() {
        let c = Cone::new(Point3::ORIGIN, Vector3::Z, 0.4, 2.0).unwrap();
        assert_eq!(c.frame().z_direction(), Vector3::Z);
    }

    #[test]
    fn test_new_null_normal_errors() {
        assert_eq!(
            Cone::new(Point3::ORIGIN, Vector3::ZERO, 0.4, 2.0),
            Err(ConeConstructionError::NullNormal)
        );
    }

    // ---- construction: from_two_points_and_radii ----

    #[test]
    fn test_from_two_points_and_radii_null_axis_errors() {
        let p = Point3::new(1.0, 2.0, 3.0);
        assert_eq!(
            Cone::from_two_points_and_radii(p, p, 1.0, 2.0),
            Err(ConeConstructionError::NullAxis)
        );
    }

    #[test]
    fn test_from_two_points_and_radii_negative_radius_errors() {
        assert_eq!(
            Cone::from_two_points_and_radii(Point3::ORIGIN, Point3::new(0.0, 0.0, 1.0), -1.0, 2.0),
            Err(ConeConstructionError::NegativeRadius)
        );
        assert_eq!(
            Cone::from_two_points_and_radii(Point3::ORIGIN, Point3::new(0.0, 0.0, 1.0), 1.0, -2.0),
            Err(ConeConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_from_two_points_and_radii_null_angle_errors_when_radii_equal() {
        // r1 == r2 gives a zero semi-angle: no valid cone (it's a cylinder).
        assert_eq!(
            Cone::from_two_points_and_radii(Point3::ORIGIN, Point3::new(0.0, 0.0, 1.0), 2.0, 2.0),
            Err(ConeConstructionError::NullAngle)
        );
    }

    #[test]
    fn test_from_two_points_and_radii_matches_golden_case_shrinking() {
        // From tests/fixtures/construction.json: cones_two_points_radii[0].
        let c =
            Cone::from_two_points_and_radii(Point3::ORIGIN, Point3::new(0.0, 0.0, 4.0), 2.0, 1.0)
                .unwrap();
        assert!((c.semi_angle() - (-0.24497866312686414)).abs() < 1e-9);
        assert_eq!(c.ref_radius(), 2.0);
        let frame = c.frame();
        assert_eq!(frame.origin(), Point3::ORIGIN);
        assert_vector3_close(frame.x_direction(), Vector3::new(0.0, -1.0, 0.0));
        assert_vector3_close(frame.y_direction(), Vector3::new(1.0, 0.0, 0.0));
        assert_vector3_close(frame.z_direction(), Vector3::Z);
    }

    #[test]
    fn test_from_two_points_and_radii_matches_golden_case_growing() {
        // From tests/fixtures/construction.json: cones_two_points_radii[1].
        let c = Cone::from_two_points_and_radii(
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(2.0, 3.0, 1.0),
            1.0,
            2.5,
        )
        .unwrap();
        assert!((c.semi_angle() - 0.5908727501454191).abs() < 1e-9);
        assert_eq!(c.ref_radius(), 1.0);
        let frame = c.frame();
        assert_eq!(frame.origin(), Point3::new(1.0, 1.0, 1.0));
        assert_vector3_close(
            frame.x_direction(),
            Vector3::new(-0.894427190999916, 0.447213595499958, 0.0),
        );
        assert_vector3_close(frame.y_direction(), Vector3::new(0.0, 0.0, 1.0));
        assert_vector3_close(
            frame.z_direction(),
            Vector3::new(0.4472135954999579, 0.8944271909999159, 0.0),
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

    // ---- evaluation ----

    #[test]
    fn test_eval_point() {
        let c = Cone::new(Point3::ORIGIN, Vector3::Z, 0.4, 2.0).unwrap();
        assert_eq!(c.eval_point(0.0, 0.0), Point3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn test_eval_points_matches_loop() {
        let c = Cone::new(Point3::ORIGIN, Vector3::Z, 0.4, 2.0).unwrap();
        let uvs = [(0.0, 0.0), (0.5, 1.0)];
        let expected: Vec<Point3> = uvs.iter().map(|&(u, v)| c.eval_point(u, v)).collect();
        assert_eq!(c.eval_points(&uvs), expected);
    }

    #[test]
    fn test_apex() {
        let c = Cone::new(Point3::ORIGIN, Vector3::Z, FRAC_PI_4, 1.0).unwrap();
        let apex = c.apex();
        assert!(apex.x.abs() < 1e-9);
        assert!(apex.y.abs() < 1e-9);
        assert!((apex.z + 1.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "du + dv must be >= 1")]
    fn test_eval_derivative_zero_order_panics() {
        let c = Cone::new(Point3::ORIGIN, Vector3::Z, 0.4, 2.0).unwrap();
        c.eval_derivative(0.0, 0.0, 0, 0);
    }

    #[test]
    #[should_panic(expected = "du + dv must be <= 2")]
    fn test_eval_derivative_order_too_high_panics() {
        let c = Cone::new(Point3::ORIGIN, Vector3::Z, 0.4, 2.0).unwrap();
        c.eval_derivative(0.0, 0.0, 2, 1);
    }

    #[test]
    fn test_parameters_of_round_trip() {
        let c = Cone::new(Point3::ORIGIN, Vector3::Z, 0.4, 2.0).unwrap();
        let (u, v) = c.parameters_of(Point3::new(2.0, 0.0, 0.0));
        assert_eq!((u, v), (0.0, 0.0));
    }

    // ---- ConeConstructionError ----

    #[test]
    fn test_error_display() {
        assert_eq!(
            ConeConstructionError::NegativeRadius.to_string(),
            "radius is negative"
        );
        assert_eq!(
            ConeConstructionError::BadAngle.to_string(),
            "semi-angle is out of range"
        );
        assert_eq!(
            ConeConstructionError::NullAxis.to_string(),
            "the two points are confused"
        );
        assert_eq!(
            ConeConstructionError::NullAngle.to_string(),
            "semi-angle is null or a right angle"
        );
        assert_eq!(
            ConeConstructionError::NullNormal.to_string(),
            "axis direction has zero length"
        );
    }

    #[test]
    fn test_error_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&ConeConstructionError::NegativeRadius);
    }
}
