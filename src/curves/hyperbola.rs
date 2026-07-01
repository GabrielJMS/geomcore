//! Hyperbolas in 3D: parametric evaluation, parameter inversion, and
//! center+two-points construction, a thin wrapper over
//! [`crate::curve_math::analytic`].

use crate::curve_math::analytic;
use crate::tol;
use crate::{Frame3, Point3, Vector3};
use std::fmt;

/// Error returned when a [`Hyperbola3D`] cannot be constructed from the
/// given inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyperbolaConstructionError {
    /// The major or minor radius is negative.
    NegativeRadius,
    /// The normal (or the x direction) has zero length, or the x direction
    /// is parallel to the normal.
    NullNormal,
    /// Two (or more) of the three points given to build the hyperbola are
    /// coincident (or too close to distinguish).
    ConfusedPoints,
    /// The second point lies on the major-axis line (or is otherwise
    /// collinear with the center and the first point), so no minor axis can
    /// be derived.
    CollinearPoints,
}

impl fmt::Display for HyperbolaConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            HyperbolaConstructionError::NegativeRadius => "radius is negative",
            HyperbolaConstructionError::NullNormal => "normal has zero length",
            HyperbolaConstructionError::ConfusedPoints => "the points are confused",
            HyperbolaConstructionError::CollinearPoints => "the points are collinear",
        };
        f.write_str(message)
    }
}

impl std::error::Error for HyperbolaConstructionError {}

/// A hyperbola in 3D: a plane [`Frame3`] (origin at the center, plus local
/// x/y directions defining the plane, the major axis, and the transverse
/// direction), a semi-major radius, and a semi-minor radius, evaluated as
/// `center + major*cosh(u)*x_dir + minor*sinh(u)*y_dir`.
///
/// # Examples
///
/// ```
/// use geomrust::{Hyperbola3D, Point3, Vector3};
/// let hyperbola = Hyperbola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 2.0, 1.0).unwrap();
/// assert_eq!(hyperbola.eval_point(0.0), Point3::new(2.0, 0.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hyperbola3D {
    frame: Frame3,
    major_radius: f64,
    minor_radius: f64,
}

impl Hyperbola3D {
    /// Creates a hyperbola from a center, a normal, an x-axis hint, a
    /// semi-major radius, and a semi-minor radius.
    ///
    /// The plane frame is derived from `normal` and `x_direction` via
    /// [`Frame3::new`].
    ///
    /// # Errors
    ///
    /// Returns [`HyperbolaConstructionError::NullNormal`] if `normal` cannot
    /// be normalized, if `x_direction` cannot be normalized, or if
    /// `x_direction` is parallel to `normal`; or
    /// [`HyperbolaConstructionError::NegativeRadius`] if `major_radius < 0`
    /// or `minor_radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Hyperbola3D, Point3, Vector3};
    /// let hyperbola = Hyperbola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 2.0, 1.0).unwrap();
    /// assert_eq!(hyperbola.major_radius(), 2.0);
    /// assert_eq!(hyperbola.minor_radius(), 1.0);
    /// ```
    pub fn new(
        center: Point3,
        normal: Vector3,
        x_direction: Vector3,
        major_radius: f64,
        minor_radius: f64,
    ) -> Result<Hyperbola3D, HyperbolaConstructionError> {
        let frame = Frame3::new(center, normal, x_direction)
            .map_err(|_| HyperbolaConstructionError::NullNormal)?;
        Hyperbola3D::from_frame(frame, major_radius, minor_radius)
    }

    /// Creates a hyperbola from a plane frame, a semi-major radius, and a
    /// semi-minor radius directly.
    ///
    /// Unlike [`crate::Ellipse3D`], `minor_radius` may exceed `major_radius`
    /// (only negativity is checked).
    ///
    /// # Errors
    ///
    /// Returns [`HyperbolaConstructionError::NegativeRadius`] if
    /// `major_radius < 0` or `minor_radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Hyperbola3D, Frame3};
    /// let hyperbola = Hyperbola3D::from_frame(Frame3::WORLD, 2.0, 1.0).unwrap();
    /// assert_eq!(hyperbola.frame(), Frame3::WORLD);
    /// ```
    pub fn from_frame(
        frame: Frame3,
        major_radius: f64,
        minor_radius: f64,
    ) -> Result<Hyperbola3D, HyperbolaConstructionError> {
        if major_radius < 0.0 || minor_radius < 0.0 {
            return Err(HyperbolaConstructionError::NegativeRadius);
        }
        Ok(Hyperbola3D {
            frame,
            major_radius,
            minor_radius,
        })
    }

    /// Creates a hyperbola from a center, a major-axis end point `s1`, and a
    /// second point `s2` that fixes the minor axis and the plane.
    ///
    /// The major axis direction is `x_axis = normalize(s1 - center)`, with
    /// semi-major radius `major_radius = |s1 - center|`. The semi-minor
    /// radius `minor_radius` is the distance from `s2` to the line
    /// `(center, x_axis)`. The plane normal is
    /// `normalize(x_axis × (s2 - center))`, and the resulting frame is
    /// `Frame3::new(center, normal, x_axis)`.
    ///
    /// # Errors
    ///
    /// Returns [`HyperbolaConstructionError::ConfusedPoints`] if any pair
    /// among `{s1, s2, center}` is within [`crate::tol::CONFUSION`]; or
    /// [`HyperbolaConstructionError::CollinearPoints`] if `s2` lies on the
    /// line `(center, x_axis)`, or `x_axis` is parallel to `s2 - center`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Hyperbola3D, Point3};
    /// let hyperbola = Hyperbola3D::from_center_and_points(
    ///     Point3::ORIGIN,
    ///     Point3::new(2.0, 0.0, 0.0),
    ///     Point3::new(0.0, 1.0, 0.0),
    /// )
    /// .unwrap();
    /// assert_eq!(hyperbola.major_radius(), 2.0);
    /// assert_eq!(hyperbola.minor_radius(), 1.0);
    /// ```
    pub fn from_center_and_points(
        center: Point3,
        s1: Point3,
        s2: Point3,
    ) -> Result<Hyperbola3D, HyperbolaConstructionError> {
        if center.distance(s1) < tol::CONFUSION
            || center.distance(s2) < tol::CONFUSION
            || s1.distance(s2) < tol::CONFUSION
        {
            return Err(HyperbolaConstructionError::ConfusedPoints);
        }

        let v1 = s1 - center;
        let d1 = v1.magnitude();
        let x_axis = v1 * (1.0 / d1);

        let v2 = s2 - center;
        let proj = v2.dot(x_axis);
        let perp = v2 - proj * x_axis;
        let d2 = perp.magnitude();

        if d2 < tol::CONFUSION {
            return Err(HyperbolaConstructionError::CollinearPoints);
        }

        let normal = x_axis.cross(v2);
        if normal.magnitude() < tol::CONFUSION {
            return Err(HyperbolaConstructionError::CollinearPoints);
        }

        let frame = Frame3::new(center, normal, x_axis)
            .map_err(|_| HyperbolaConstructionError::CollinearPoints)?;
        Hyperbola3D::from_frame(frame, d1, d2)
    }

    /// Returns the center of the hyperbola.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Hyperbola3D, Point3, Vector3};
    /// let hyperbola = Hyperbola3D::new(Point3::new(1.0, 2.0, 3.0), Vector3::Z, Vector3::X, 2.0, 1.0).unwrap();
    /// assert_eq!(hyperbola.center(), Point3::new(1.0, 2.0, 3.0));
    /// ```
    pub fn center(&self) -> Point3 {
        self.frame.origin()
    }

    /// Returns the hyperbola's plane frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Hyperbola3D, Frame3};
    /// let hyperbola = Hyperbola3D::from_frame(Frame3::WORLD, 2.0, 1.0).unwrap();
    /// assert_eq!(hyperbola.frame(), Frame3::WORLD);
    /// ```
    pub fn frame(&self) -> Frame3 {
        self.frame
    }

    /// Returns the semi-major radius of the hyperbola.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Hyperbola3D, Frame3};
    /// let hyperbola = Hyperbola3D::from_frame(Frame3::WORLD, 2.0, 1.0).unwrap();
    /// assert_eq!(hyperbola.major_radius(), 2.0);
    /// ```
    pub fn major_radius(&self) -> f64 {
        self.major_radius
    }

    /// Returns the semi-minor radius of the hyperbola.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Hyperbola3D, Frame3};
    /// let hyperbola = Hyperbola3D::from_frame(Frame3::WORLD, 2.0, 1.0).unwrap();
    /// assert_eq!(hyperbola.minor_radius(), 1.0);
    /// ```
    pub fn minor_radius(&self) -> f64 {
        self.minor_radius
    }

    /// Evaluates the point on the hyperbola at parameter `u`:
    /// `center + major*cosh(u)*x_dir + minor*sinh(u)*y_dir`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Hyperbola3D, Point3, Vector3};
    /// let hyperbola = Hyperbola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 2.0, 1.0).unwrap();
    /// assert_eq!(hyperbola.eval_point(0.0), Point3::new(2.0, 0.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64) -> Point3 {
        analytic::hyperbola_d0(&self.frame, self.major_radius, self.minor_radius, u)
    }

    /// Evaluates the points on the hyperbola at each parameter in `us`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Hyperbola3D, Point3, Vector3};
    /// let hyperbola = Hyperbola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 2.0, 1.0).unwrap();
    /// let points = hyperbola.eval_points(&[0.0, 1.0]);
    /// assert_eq!(points[0], Point3::new(2.0, 0.0, 0.0));
    /// ```
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point3> {
        us.iter().map(|&u| self.eval_point(u)).collect()
    }

    /// Evaluates the derivative of the given `order` at parameter `u`.
    ///
    /// Unlike the trigonometric conics, hyperbolic derivatives do not cycle
    /// with period 4 (see [`crate::curve_math::analytic::hyperbola_dn`]):
    /// odd orders follow the `sinh`/`cosh` pattern of the first derivative,
    /// even orders follow the `cosh`/`sinh` pattern of the second.
    ///
    /// # Panics
    ///
    /// Panics if `order == 0`; use [`Hyperbola3D::eval_point`] to evaluate
    /// the position itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Hyperbola3D, Point3, Vector3};
    /// let hyperbola = Hyperbola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 2.0, 1.0).unwrap();
    /// assert_eq!(hyperbola.eval_derivative(0.0, 1), Vector3::new(0.0, 1.0, 0.0));
    /// ```
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector3 {
        match order {
            0 => panic!("eval_derivative: order must be >= 1 (use eval_point for order 0)"),
            _ => {
                analytic::hyperbola_dn(&self.frame, self.major_radius, self.minor_radius, u, order)
            }
        }
    }

    /// Recovers the parameter of a point on (or near) the hyperbola:
    /// `asinh(((point - center) . y_dir) / minor_radius)`. Unbounded (no
    /// wrapping).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Hyperbola3D, Point3, Vector3};
    /// let hyperbola = Hyperbola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 2.0, 1.0).unwrap();
    /// assert!((hyperbola.parameter_of(Point3::new(2.0, 0.0, 0.0)) - 0.0).abs() < 1e-9);
    /// ```
    pub fn parameter_of(&self, point: Point3) -> f64 {
        analytic::hyperbola_parameter(&self.frame, self.minor_radius, point)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Frame3, Hyperbola3D, HyperbolaConstructionError, Point3, Vector3};

    // ---- construction ----

    #[test]
    fn test_hyperbola3d_new_ok() {
        let h = Hyperbola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 2.0, 1.0).unwrap();
        assert_eq!(h.center(), Point3::ORIGIN);
        assert_eq!(h.major_radius(), 2.0);
        assert_eq!(h.minor_radius(), 1.0);
    }

    #[test]
    fn test_hyperbola3d_new_negative_major_radius_errors() {
        assert_eq!(
            Hyperbola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, -2.0, 1.0),
            Err(HyperbolaConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_hyperbola3d_new_negative_minor_radius_errors() {
        assert_eq!(
            Hyperbola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 2.0, -1.0),
            Err(HyperbolaConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_hyperbola3d_new_minor_greater_than_major_allowed() {
        let h = Hyperbola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 1.0, 2.0).unwrap();
        assert_eq!(h.major_radius(), 1.0);
        assert_eq!(h.minor_radius(), 2.0);
    }

    #[test]
    fn test_hyperbola3d_new_null_normal_errors() {
        assert_eq!(
            Hyperbola3D::new(Point3::ORIGIN, Vector3::ZERO, Vector3::X, 2.0, 1.0),
            Err(HyperbolaConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_hyperbola3d_new_parallel_x_direction_errors() {
        assert_eq!(
            Hyperbola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::Z, 2.0, 1.0),
            Err(HyperbolaConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_hyperbola3d_from_frame_ok() {
        let h = Hyperbola3D::from_frame(Frame3::WORLD, 2.0, 1.0).unwrap();
        assert_eq!(h.frame(), Frame3::WORLD);
        assert_eq!(h.major_radius(), 2.0);
        assert_eq!(h.minor_radius(), 1.0);
    }

    #[test]
    fn test_hyperbola3d_from_frame_negative_major_radius_errors() {
        assert_eq!(
            Hyperbola3D::from_frame(Frame3::WORLD, -2.0, 1.0),
            Err(HyperbolaConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_hyperbola3d_from_frame_negative_minor_radius_errors() {
        assert_eq!(
            Hyperbola3D::from_frame(Frame3::WORLD, 2.0, -1.0),
            Err(HyperbolaConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_hyperbola3d_from_center_and_points_ok() {
        let h = Hyperbola3D::from_center_and_points(
            Point3::ORIGIN,
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        )
        .unwrap();
        assert_eq!(h.major_radius(), 2.0);
        assert_eq!(h.minor_radius(), 1.0);
    }

    #[test]
    fn test_hyperbola3d_from_center_and_points_confused_points_errors() {
        assert_eq!(
            Hyperbola3D::from_center_and_points(
                Point3::ORIGIN,
                Point3::ORIGIN,
                Point3::new(0.0, 1.0, 0.0),
            ),
            Err(HyperbolaConstructionError::ConfusedPoints)
        );
    }

    #[test]
    fn test_hyperbola3d_from_center_and_points_collinear_points_errors() {
        assert_eq!(
            Hyperbola3D::from_center_and_points(
                Point3::ORIGIN,
                Point3::new(2.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
            ),
            Err(HyperbolaConstructionError::CollinearPoints)
        );
    }

    // ---- evaluation ----

    #[test]
    fn test_hyperbola3d_eval_point_zero() {
        let h = Hyperbola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 2.0, 1.0).unwrap();
        let p = h.eval_point(0.0);
        assert!((p.x - 2.0).abs() < 1e-9);
        assert!(p.y.abs() < 1e-9);
        assert!(p.z.abs() < 1e-9);
    }

    #[test]
    fn test_hyperbola3d_eval_points_matches_loop() {
        let h = Hyperbola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 2.0, 1.0).unwrap();
        let us = [0.0, 0.5, 1.5];
        let expected: Vec<Point3> = us.iter().map(|&u| h.eval_point(u)).collect();
        assert_eq!(h.eval_points(&us), expected);
    }

    #[test]
    fn test_hyperbola3d_eval_derivative_order1() {
        let h = Hyperbola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 2.0, 1.0).unwrap();
        let d1 = h.eval_derivative(0.0, 1);
        assert!(d1.x.abs() < 1e-9);
        assert!((d1.y - 1.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic]
    fn test_hyperbola3d_eval_derivative_order0_panics() {
        let h = Hyperbola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 2.0, 1.0).unwrap();
        h.eval_derivative(0.0, 0);
    }

    #[test]
    fn test_hyperbola3d_parameter_of_round_trip() {
        let h = Hyperbola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 2.0, 1.0).unwrap();
        for u in [0.3, 2.0, -1.5] {
            let p = h.eval_point(u);
            assert!((h.parameter_of(p) - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_hyperbola3d_parameter_of_is_unbounded() {
        let h = Hyperbola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 2.0, 1.0).unwrap();
        let p = h.eval_point(5.0);
        assert!((h.parameter_of(p) - 5.0).abs() < 1e-6);
    }

    // ---- HyperbolaConstructionError ----

    #[test]
    fn test_hyperbola_construction_error_display() {
        assert_eq!(
            HyperbolaConstructionError::NegativeRadius.to_string(),
            "radius is negative"
        );
        assert_eq!(
            HyperbolaConstructionError::NullNormal.to_string(),
            "normal has zero length"
        );
        assert_eq!(
            HyperbolaConstructionError::ConfusedPoints.to_string(),
            "the points are confused"
        );
        assert_eq!(
            HyperbolaConstructionError::CollinearPoints.to_string(),
            "the points are collinear"
        );
    }

    #[test]
    fn test_hyperbola_construction_error_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&HyperbolaConstructionError::NegativeRadius);
    }
}
