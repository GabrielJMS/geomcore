//! Ellipses in 3D: parametric evaluation, parameter inversion, and
//! center+two-points construction, a thin wrapper over
//! [`crate::curve_math::analytic`].

use crate::curve_math::analytic;
use crate::tol;
use crate::{Frame3, Point3, Vector3};
use std::fmt;

/// Error returned when an [`Ellipse3D`] cannot be constructed from the given
/// inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EllipseConstructionError {
    /// The requested minor radius is negative.
    NegativeRadius,
    /// The major radius is smaller than the minor radius.
    InvertedRadii,
    /// The normal (or the x direction) has zero length, or the x direction
    /// is parallel to the normal.
    NullNormal,
    /// The major-axis point coincides with the center, so no major axis can
    /// be derived.
    NullAxis,
    /// The two points given do not determine a valid major/minor axis pair
    /// (the minor point is farther from, or as close to, the axis line as
    /// the major point, or the two points are collinear with the center).
    InvertedAxis,
}

impl fmt::Display for EllipseConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            EllipseConstructionError::NegativeRadius => "minor radius is negative",
            EllipseConstructionError::InvertedRadii => "major radius is smaller than minor radius",
            EllipseConstructionError::NullNormal => "normal has zero length",
            EllipseConstructionError::NullAxis => "major-axis point coincides with the center",
            EllipseConstructionError::InvertedAxis => {
                "the two points do not determine a valid axis pair"
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for EllipseConstructionError {}

/// An ellipse in 3D: a plane [`Frame3`] (origin plus local x/y directions
/// defining the plane, the major axis, and the angular origin), a semi-major
/// radius, and a semi-minor radius, evaluated as
/// `origin + major*cos(u)*x_dir + minor*sin(u)*y_dir`.
///
/// # Examples
///
/// ```
/// use geomrust::{Ellipse3D, Point3, Vector3};
/// let ellipse = Ellipse3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 3.0, 1.5).unwrap();
/// assert_eq!(ellipse.eval_point(0.0), Point3::new(3.0, 0.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse3D {
    frame: Frame3,
    major_radius: f64,
    minor_radius: f64,
}

impl Ellipse3D {
    /// Creates an ellipse from a center, a normal, an x-axis hint, a
    /// semi-major radius, and a semi-minor radius.
    ///
    /// The plane frame is derived from `normal` and `x_direction` via
    /// [`Frame3::new`].
    ///
    /// # Errors
    ///
    /// Returns [`EllipseConstructionError::NullNormal`] if `normal` cannot
    /// be normalized, if `x_direction` cannot be normalized, or if
    /// `x_direction` is parallel to `normal`;
    /// [`EllipseConstructionError::NegativeRadius`] if `minor_radius < 0`; or
    /// [`EllipseConstructionError::InvertedRadii`] if
    /// `major_radius < minor_radius`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Ellipse3D, Point3, Vector3};
    /// let ellipse = Ellipse3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 3.0, 1.5).unwrap();
    /// assert_eq!(ellipse.major_radius(), 3.0);
    /// assert_eq!(ellipse.minor_radius(), 1.5);
    /// ```
    pub fn new(
        center: Point3,
        normal: Vector3,
        x_direction: Vector3,
        major_radius: f64,
        minor_radius: f64,
    ) -> Result<Ellipse3D, EllipseConstructionError> {
        let frame = Frame3::new(center, normal, x_direction)
            .map_err(|_| EllipseConstructionError::NullNormal)?;
        Ellipse3D::from_frame(frame, major_radius, minor_radius)
    }

    /// Creates an ellipse from a plane frame, a semi-major radius, and a
    /// semi-minor radius directly.
    ///
    /// # Errors
    ///
    /// Returns [`EllipseConstructionError::NegativeRadius`] if
    /// `minor_radius < 0`, or [`EllipseConstructionError::InvertedRadii`] if
    /// `major_radius < minor_radius`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Ellipse3D, Frame3};
    /// let ellipse = Ellipse3D::from_frame(Frame3::WORLD, 3.0, 1.5).unwrap();
    /// assert_eq!(ellipse.frame(), Frame3::WORLD);
    /// ```
    pub fn from_frame(
        frame: Frame3,
        major_radius: f64,
        minor_radius: f64,
    ) -> Result<Ellipse3D, EllipseConstructionError> {
        if minor_radius < 0.0 {
            return Err(EllipseConstructionError::NegativeRadius);
        }
        if major_radius < minor_radius {
            return Err(EllipseConstructionError::InvertedRadii);
        }
        Ok(Ellipse3D {
            frame,
            major_radius,
            minor_radius,
        })
    }

    /// Creates an ellipse from a center, a major-axis end point `s1`, and a
    /// second point `s2` that fixes the minor axis and the plane.
    ///
    /// The major axis direction is `x_axis = normalize(s1 - center)`, with
    /// semi-major radius `d1 = |s1 - center|`. The semi-minor radius `d2` is
    /// the distance from `s2` to the line `(center, x_axis)`. The plane
    /// normal is `normalize(x_axis × (s2 - center))`, and the resulting
    /// frame is `Frame3::new(center, normal, x_axis)`.
    ///
    /// # Errors
    ///
    /// Returns [`EllipseConstructionError::NullAxis`] if `d1` is below
    /// [`crate::tol::CONFUSION`] (`s1` coincides with `center`); or
    /// [`EllipseConstructionError::InvertedAxis`] if `d1 < d2`, if `d2` is
    /// below `1e-7`, or if `x_axis` is parallel to `s2 - center` (the two
    /// points and the center are collinear).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Ellipse3D, Point3};
    /// let ellipse = Ellipse3D::from_center_and_points(
    ///     Point3::ORIGIN,
    ///     Point3::new(3.0, 0.0, 0.0),
    ///     Point3::new(0.0, 1.5, 0.0),
    /// )
    /// .unwrap();
    /// assert_eq!(ellipse.major_radius(), 3.0);
    /// assert_eq!(ellipse.minor_radius(), 1.5);
    /// ```
    pub fn from_center_and_points(
        center: Point3,
        s1: Point3,
        s2: Point3,
    ) -> Result<Ellipse3D, EllipseConstructionError> {
        let v1 = s1 - center;
        let d1 = v1.magnitude();
        if d1 < tol::CONFUSION {
            return Err(EllipseConstructionError::NullAxis);
        }
        let x_axis = v1 * (1.0 / d1);

        let v2 = s2 - center;
        let proj = v2.dot(x_axis);
        let perp = v2 - proj * x_axis;
        let d2 = perp.magnitude();

        if d1 < d2 || d2 < 1e-7 {
            return Err(EllipseConstructionError::InvertedAxis);
        }

        let normal = x_axis.cross(v2);
        if normal.magnitude() < 1e-7 {
            return Err(EllipseConstructionError::InvertedAxis);
        }

        let frame = Frame3::new(center, normal, x_axis)
            .map_err(|_| EllipseConstructionError::InvertedAxis)?;
        Ellipse3D::from_frame(frame, d1, d2)
    }

    /// Returns the center of the ellipse.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Ellipse3D, Point3, Vector3};
    /// let ellipse = Ellipse3D::new(Point3::new(1.0, 2.0, 3.0), Vector3::Z, Vector3::X, 3.0, 1.5).unwrap();
    /// assert_eq!(ellipse.center(), Point3::new(1.0, 2.0, 3.0));
    /// ```
    pub fn center(&self) -> Point3 {
        self.frame.origin()
    }

    /// Returns the ellipse's plane frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Ellipse3D, Frame3};
    /// let ellipse = Ellipse3D::from_frame(Frame3::WORLD, 3.0, 1.5).unwrap();
    /// assert_eq!(ellipse.frame(), Frame3::WORLD);
    /// ```
    pub fn frame(&self) -> Frame3 {
        self.frame
    }

    /// Returns the semi-major radius of the ellipse.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Ellipse3D, Frame3};
    /// let ellipse = Ellipse3D::from_frame(Frame3::WORLD, 3.0, 1.5).unwrap();
    /// assert_eq!(ellipse.major_radius(), 3.0);
    /// ```
    pub fn major_radius(&self) -> f64 {
        self.major_radius
    }

    /// Returns the semi-minor radius of the ellipse.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Ellipse3D, Frame3};
    /// let ellipse = Ellipse3D::from_frame(Frame3::WORLD, 3.0, 1.5).unwrap();
    /// assert_eq!(ellipse.minor_radius(), 1.5);
    /// ```
    pub fn minor_radius(&self) -> f64 {
        self.minor_radius
    }

    /// Evaluates the point on the ellipse at angular parameter `u`:
    /// `center + major*cos(u)*x_dir + minor*sin(u)*y_dir`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Ellipse3D, Point3, Vector3};
    /// let ellipse = Ellipse3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 3.0, 1.5).unwrap();
    /// assert_eq!(ellipse.eval_point(0.0), Point3::new(3.0, 0.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64) -> Point3 {
        analytic::ellipse_d0(&self.frame, self.major_radius, self.minor_radius, u)
    }

    /// Evaluates the points on the ellipse at each parameter in `us`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Ellipse3D, Point3, Vector3};
    /// let ellipse = Ellipse3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 3.0, 1.5).unwrap();
    /// let points = ellipse.eval_points(&[0.0, 1.0]);
    /// assert_eq!(points[0], Point3::new(3.0, 0.0, 0.0));
    /// ```
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point3> {
        us.iter().map(|&u| self.eval_point(u)).collect()
    }

    /// Evaluates the derivative of the given `order` at parameter `u`.
    ///
    /// Derivatives cycle with period 4 in `order` (see
    /// [`crate::curve_math::analytic::ellipse_dn`]).
    ///
    /// # Panics
    ///
    /// Panics if `order == 0`; use [`Ellipse3D::eval_point`] to evaluate the
    /// position itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Ellipse3D, Point3, Vector3};
    /// let ellipse = Ellipse3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 3.0, 1.5).unwrap();
    /// assert_eq!(ellipse.eval_derivative(0.0, 1), Vector3::new(0.0, 1.5, 0.0));
    /// ```
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector3 {
        match order {
            0 => panic!("eval_derivative: order must be >= 1 (use eval_point for order 0)"),
            _ => analytic::ellipse_dn(&self.frame, self.major_radius, self.minor_radius, u, order),
        }
    }

    /// Recovers the angular parameter of a point on (or near) the ellipse,
    /// wrapped into `[0, 2*PI)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Ellipse3D, Point3, Vector3};
    /// let ellipse = Ellipse3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 3.0, 1.5).unwrap();
    /// assert!((ellipse.parameter_of(Point3::new(0.0, 1.5, 0.0)) - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    /// ```
    pub fn parameter_of(&self, point: Point3) -> f64 {
        analytic::ellipse_parameter(&self.frame, self.major_radius, self.minor_radius, point)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Ellipse3D, EllipseConstructionError, Frame3, Point3, Vector3};

    // ---- construction ----

    #[test]
    fn test_ellipse3d_new_ok() {
        let e = Ellipse3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 3.0, 1.5).unwrap();
        assert_eq!(e.center(), Point3::ORIGIN);
        assert_eq!(e.major_radius(), 3.0);
        assert_eq!(e.minor_radius(), 1.5);
    }

    #[test]
    fn test_ellipse3d_new_negative_minor_radius_errors() {
        assert_eq!(
            Ellipse3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 3.0, -1.0),
            Err(EllipseConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_ellipse3d_new_inverted_radii_errors() {
        assert_eq!(
            Ellipse3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 1.0, 2.0),
            Err(EllipseConstructionError::InvertedRadii)
        );
    }

    #[test]
    fn test_ellipse3d_new_null_normal_errors() {
        assert_eq!(
            Ellipse3D::new(Point3::ORIGIN, Vector3::ZERO, Vector3::X, 3.0, 1.5),
            Err(EllipseConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_ellipse3d_new_parallel_x_direction_errors() {
        assert_eq!(
            Ellipse3D::new(Point3::ORIGIN, Vector3::Z, Vector3::Z, 3.0, 1.5),
            Err(EllipseConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_ellipse3d_from_frame_ok() {
        let e = Ellipse3D::from_frame(Frame3::WORLD, 3.0, 1.5).unwrap();
        assert_eq!(e.frame(), Frame3::WORLD);
        assert_eq!(e.major_radius(), 3.0);
        assert_eq!(e.minor_radius(), 1.5);
    }

    #[test]
    fn test_ellipse3d_from_frame_negative_radius_errors() {
        assert_eq!(
            Ellipse3D::from_frame(Frame3::WORLD, 3.0, -1.0),
            Err(EllipseConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_ellipse3d_from_frame_inverted_radii_errors() {
        assert_eq!(
            Ellipse3D::from_frame(Frame3::WORLD, 1.0, 2.0),
            Err(EllipseConstructionError::InvertedRadii)
        );
    }

    #[test]
    fn test_ellipse3d_from_center_and_points_ok() {
        let e = Ellipse3D::from_center_and_points(
            Point3::ORIGIN,
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(0.0, 1.5, 0.0),
        )
        .unwrap();
        assert_eq!(e.major_radius(), 3.0);
        assert_eq!(e.minor_radius(), 1.5);
    }

    #[test]
    fn test_ellipse3d_from_center_and_points_null_axis_errors() {
        assert_eq!(
            Ellipse3D::from_center_and_points(
                Point3::ORIGIN,
                Point3::ORIGIN,
                Point3::new(0.0, 1.5, 0.0),
            ),
            Err(EllipseConstructionError::NullAxis)
        );
    }

    #[test]
    fn test_ellipse3d_from_center_and_points_inverted_axis_errors_when_minor_larger() {
        assert_eq!(
            Ellipse3D::from_center_and_points(
                Point3::ORIGIN,
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 3.0, 0.0),
            ),
            Err(EllipseConstructionError::InvertedAxis)
        );
    }

    #[test]
    fn test_ellipse3d_from_center_and_points_inverted_axis_errors_when_collinear() {
        assert_eq!(
            Ellipse3D::from_center_and_points(
                Point3::ORIGIN,
                Point3::new(3.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
            ),
            Err(EllipseConstructionError::InvertedAxis)
        );
    }

    // ---- evaluation ----

    #[test]
    fn test_ellipse3d_eval_point_zero() {
        let e = Ellipse3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 3.0, 1.5).unwrap();
        let p = e.eval_point(0.0);
        assert!((p.x - 3.0).abs() < 1e-9);
        assert!(p.y.abs() < 1e-9);
        assert!(p.z.abs() < 1e-9);
    }

    #[test]
    fn test_ellipse3d_eval_points_matches_loop() {
        let e = Ellipse3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 3.0, 1.5).unwrap();
        let us = [0.0, 0.5, 1.5];
        let expected: Vec<Point3> = us.iter().map(|&u| e.eval_point(u)).collect();
        assert_eq!(e.eval_points(&us), expected);
    }

    #[test]
    fn test_ellipse3d_eval_derivative_order1() {
        let e = Ellipse3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 3.0, 1.5).unwrap();
        let d1 = e.eval_derivative(0.0, 1);
        assert!(d1.x.abs() < 1e-9);
        assert!((d1.y - 1.5).abs() < 1e-9);
    }

    #[test]
    #[should_panic]
    fn test_ellipse3d_eval_derivative_order0_panics() {
        let e = Ellipse3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 3.0, 1.5).unwrap();
        e.eval_derivative(0.0, 0);
    }

    #[test]
    fn test_ellipse3d_parameter_of_round_trip() {
        let e = Ellipse3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 3.0, 1.5).unwrap();
        for u in [0.3, 2.0, 5.5] {
            let p = e.eval_point(u);
            assert!((e.parameter_of(p) - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_ellipse3d_parameter_of_in_zero_to_tau() {
        let e = Ellipse3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 3.0, 1.5).unwrap();
        for u in [-1.0, -0.1, 7.0] {
            let p = e.eval_point(u);
            let recovered = e.parameter_of(p);
            assert!((0.0..std::f64::consts::TAU).contains(&recovered));
        }
    }

    // ---- EllipseConstructionError ----

    #[test]
    fn test_ellipse_construction_error_display() {
        assert_eq!(
            EllipseConstructionError::NegativeRadius.to_string(),
            "minor radius is negative"
        );
        assert_eq!(
            EllipseConstructionError::InvertedRadii.to_string(),
            "major radius is smaller than minor radius"
        );
        assert_eq!(
            EllipseConstructionError::NullNormal.to_string(),
            "normal has zero length"
        );
        assert_eq!(
            EllipseConstructionError::NullAxis.to_string(),
            "major-axis point coincides with the center"
        );
        assert_eq!(
            EllipseConstructionError::InvertedAxis.to_string(),
            "the two points do not determine a valid axis pair"
        );
    }

    #[test]
    fn test_ellipse_construction_error_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&EllipseConstructionError::NegativeRadius);
    }
}
