//! Infinite lines in 2D and 3D: parametric evaluation and parameter
//! inversion, thin wrappers over [`crate::curve_math::analytic`].

use crate::curve_math::analytic;
use crate::curves::Curve2D;
use crate::curves::parametrize::{self, ParametrizeError};
use crate::surfaces::Surface;
use crate::{Axis2, Axis3, Point2, Point3, Vector2, Vector3};
use std::fmt;

/// Error returned when a [`Line3D`] or [`Line2D`] cannot be constructed from
/// the given inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineConstructionError {
    /// A direction vector could not be normalized (zero length).
    NullDirection,
    /// The two points given to build the line are coincident (or too close
    /// to distinguish), so no direction can be derived from them.
    ConfusedPoints,
}

impl fmt::Display for LineConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            LineConstructionError::NullDirection => "direction has zero length",
            LineConstructionError::ConfusedPoints => "the two points are confused",
        };
        f.write_str(message)
    }
}

impl std::error::Error for LineConstructionError {}

/// An infinite line in 3D: an origin point and a unit direction, evaluated
/// as `origin + u * direction`.
///
/// # Examples
///
/// ```
/// use geomrust::{Line3D, Point3, Vector3};
/// let line = Line3D::new(Point3::ORIGIN, Vector3::X).unwrap();
/// assert_eq!(line.eval_point(3.0), Point3::new(3.0, 0.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line3D {
    axis: Axis3,
}

impl Line3D {
    /// Creates a new line from an origin and a direction.
    ///
    /// The direction is normalized. Returns
    /// [`LineConstructionError::NullDirection`] if `direction` cannot be
    /// normalized (zero length).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Line3D, Point3, Vector3};
    /// let line = Line3D::new(Point3::ORIGIN, Vector3::new(2.0, 0.0, 0.0)).unwrap();
    /// assert_eq!(line.direction(), Vector3::X);
    /// ```
    pub fn new(origin: Point3, direction: Vector3) -> Result<Line3D, LineConstructionError> {
        let axis =
            Axis3::new(origin, direction).map_err(|_| LineConstructionError::NullDirection)?;
        Ok(Line3D { axis })
    }

    /// Creates a line from an axis directly.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Axis3, Line3D, Point3, Vector3};
    /// let axis = Axis3::new(Point3::ORIGIN, Vector3::Y).unwrap();
    /// let line = Line3D::from_axis(axis);
    /// assert_eq!(line.axis(), axis);
    /// ```
    pub fn from_axis(axis: Axis3) -> Line3D {
        Line3D { axis }
    }

    /// Creates a line through two points; the direction points from `p1` to
    /// `p2`.
    ///
    /// The two points must be distinct. Returns
    /// [`LineConstructionError::ConfusedPoints`] if `p2 - p1` cannot be
    /// normalized (the points are coincident or too close to distinguish).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Line3D, Point3};
    /// let line = Line3D::from_two_points(Point3::ORIGIN, Point3::new(2.0, 0.0, 0.0)).unwrap();
    /// assert_eq!(line.origin(), Point3::ORIGIN);
    /// ```
    pub fn from_two_points(p1: Point3, p2: Point3) -> Result<Line3D, LineConstructionError> {
        let direction = (p2 - p1)
            .normalized()
            .ok_or(LineConstructionError::ConfusedPoints)?;
        Ok(Line3D {
            axis: Axis3::new(p1, direction).map_err(|_| LineConstructionError::ConfusedPoints)?,
        })
    }

    /// Returns the origin point of the line.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Line3D, Point3, Vector3};
    /// let line = Line3D::new(Point3::new(1.0, 2.0, 3.0), Vector3::X).unwrap();
    /// assert_eq!(line.origin(), Point3::new(1.0, 2.0, 3.0));
    /// ```
    pub fn origin(&self) -> Point3 {
        self.axis.origin()
    }

    /// Returns the unit direction of the line.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Line3D, Point3, Vector3};
    /// let line = Line3D::new(Point3::ORIGIN, Vector3::new(0.0, 5.0, 0.0)).unwrap();
    /// assert_eq!(line.direction(), Vector3::Y);
    /// ```
    pub fn direction(&self) -> Vector3 {
        self.axis.direction()
    }

    /// Returns the line's underlying axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Axis3, Line3D, Point3, Vector3};
    /// let axis = Axis3::new(Point3::ORIGIN, Vector3::X).unwrap();
    /// let line = Line3D::from_axis(axis);
    /// assert_eq!(line.axis(), axis);
    /// ```
    pub fn axis(&self) -> Axis3 {
        self.axis
    }

    /// Evaluates the point on the line at parameter `u`: `origin + u * direction`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Line3D, Point3, Vector3};
    /// let line = Line3D::new(Point3::ORIGIN, Vector3::X).unwrap();
    /// assert_eq!(line.eval_point(3.0), Point3::new(3.0, 0.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64) -> Point3 {
        analytic::line_d0(&self.axis, u)
    }

    /// Evaluates the points on the line at each parameter in `us`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Line3D, Point3, Vector3};
    /// let line = Line3D::new(Point3::ORIGIN, Vector3::X).unwrap();
    /// let points = line.eval_points(&[0.0, 1.0, 2.0]);
    /// assert_eq!(points, vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0)]);
    /// ```
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point3> {
        us.iter().map(|&u| self.eval_point(u)).collect()
    }

    /// Evaluates the derivative of the given `order` at parameter `u`.
    ///
    /// The line is affine in `u`, so the first derivative is the constant
    /// direction vector and every derivative of order 2 or higher is zero.
    /// `u` does not affect the result (included for API consistency with
    /// curved parametric types).
    ///
    /// # Panics
    ///
    /// Panics if `order == 0`; use [`Line3D::eval_point`] to evaluate the
    /// position itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Line3D, Point3, Vector3};
    /// let line = Line3D::new(Point3::ORIGIN, Vector3::new(0.0, 3.0, 0.0)).unwrap();
    /// assert_eq!(line.eval_derivative(1.5, 1), Vector3::Y);
    /// assert_eq!(line.eval_derivative(1.5, 2), Vector3::ZERO);
    /// ```
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector3 {
        let _ = u;
        match order {
            0 => panic!("eval_derivative: order must be >= 1 (use eval_point for order 0)"),
            1 => analytic::line_d1(&self.axis),
            _ => Vector3::ZERO,
        }
    }

    /// Recovers the parameter `u` of a point on (or near) the line: the
    /// signed projection `(point - origin) . direction`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Line3D, Point3, Vector3};
    /// let line = Line3D::new(Point3::ORIGIN, Vector3::X).unwrap();
    /// assert_eq!(line.parameter_of(Point3::new(2.5, 0.0, 0.0)), 2.5);
    /// ```
    pub fn parameter_of(&self, point: Point3) -> f64 {
        analytic::line_parameter(&self.axis, point)
    }

    /// Computes the exact 2D representation of this line in a surface's
    /// parameter space: a [`Curve2D`] `q(t)` such that
    /// `surface.eval_point(q(t)) == self.eval_point(t)` for the same `t`.
    ///
    /// Only a few line/surface pairs admit a closed-form 2D image: a line on
    /// a plane (projects to a 2D line), a line parallel to a cylinder or cone
    /// axis / a cone generator (a vertical iso-`u` line in `(u, v)`). Every
    /// other pair — including a line on a sphere or torus — returns
    /// [`ParametrizeError::NotAnalytic`].
    ///
    /// The projection math assumes the line lies on the surface. As a
    /// geomrust safeguard, the candidate 2D image is verified against the
    /// surface at a few parameters; if `surface.eval_point(q(t))` disagrees
    /// with `self.eval_point(t)`, [`ParametrizeError::CurveNotOnSurface`] is
    /// returned. For periodic surfaces the result is normalized so `q(0)`
    /// lies in the canonical parameter window.
    ///
    /// # Examples
    ///
    /// A vertical line on a cylinder maps to a vertical line in `(u, v)`:
    ///
    /// ```
    /// use geomrust::curves::{Curve2D, ParametricCurve2D};
    /// use geomrust::{Cylinder, Line3D, Point3, Vector3};
    ///
    /// let cylinder = Cylinder::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
    /// let line = Line3D::new(Point3::new(2.0, 0.0, 0.0), Vector3::Z).unwrap();
    /// let pcurve = line.parametrize_on(&cylinder).unwrap();
    /// // q(0) sits at u = 0 (angle of x-axis), v = 0 (height of the origin).
    /// let q0 = pcurve.eval_point(0.0);
    /// assert!(q0.x.abs() < 1e-9 && q0.y.abs() < 1e-9);
    /// assert!(matches!(pcurve, Curve2D::Line(_)));
    /// ```
    pub fn parametrize_on(&self, surface: impl Into<Surface>) -> Result<Curve2D, ParametrizeError> {
        parametrize::line_on_surface(self, &surface.into())
    }
}

/// An infinite line in 2D: an origin point and a unit direction, evaluated
/// as `origin + u * direction`.
///
/// # Examples
///
/// ```
/// use geomrust::{Line2D, Point2, Vector2};
/// let line = Line2D::new(Point2::ORIGIN, Vector2::X).unwrap();
/// assert_eq!(line.eval_point(3.0), Point2::new(3.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line2D {
    axis: Axis2,
}

impl Line2D {
    /// Creates a new line from an origin and a direction.
    ///
    /// The direction is normalized. Returns
    /// [`LineConstructionError::NullDirection`] if `direction` cannot be
    /// normalized (zero length).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Line2D, Point2, Vector2};
    /// let line = Line2D::new(Point2::ORIGIN, Vector2::new(0.0, 5.0)).unwrap();
    /// assert_eq!(line.direction(), Vector2::Y);
    /// ```
    pub fn new(origin: Point2, direction: Vector2) -> Result<Line2D, LineConstructionError> {
        let axis =
            Axis2::new(origin, direction).map_err(|_| LineConstructionError::NullDirection)?;
        Ok(Line2D { axis })
    }

    /// Creates a line from an axis directly.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Axis2, Line2D, Point2, Vector2};
    /// let axis = Axis2::new(Point2::ORIGIN, Vector2::X).unwrap();
    /// let line = Line2D::from_axis(axis);
    /// assert_eq!(line.axis(), axis);
    /// ```
    pub fn from_axis(axis: Axis2) -> Line2D {
        Line2D { axis }
    }

    /// Creates a line through two points; the direction points from `p1` to
    /// `p2`.
    ///
    /// The two points must be distinct. Returns
    /// [`LineConstructionError::ConfusedPoints`] if `p2 - p1` cannot be
    /// normalized (the points are coincident or too close to distinguish).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Line2D, Point2};
    /// let line = Line2D::from_two_points(Point2::ORIGIN, Point2::new(0.0, 4.0)).unwrap();
    /// assert_eq!(line.origin(), Point2::ORIGIN);
    /// ```
    pub fn from_two_points(p1: Point2, p2: Point2) -> Result<Line2D, LineConstructionError> {
        let direction = (p2 - p1)
            .normalized()
            .ok_or(LineConstructionError::ConfusedPoints)?;
        Ok(Line2D {
            axis: Axis2::new(p1, direction).map_err(|_| LineConstructionError::ConfusedPoints)?,
        })
    }

    /// Returns the origin point of the line.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Line2D, Point2, Vector2};
    /// let line = Line2D::new(Point2::new(1.0, 2.0), Vector2::X).unwrap();
    /// assert_eq!(line.origin(), Point2::new(1.0, 2.0));
    /// ```
    pub fn origin(&self) -> Point2 {
        self.axis.origin()
    }

    /// Returns the unit direction of the line.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Line2D, Point2, Vector2};
    /// let line = Line2D::new(Point2::ORIGIN, Vector2::new(0.0, 5.0)).unwrap();
    /// assert_eq!(line.direction(), Vector2::Y);
    /// ```
    pub fn direction(&self) -> Vector2 {
        self.axis.direction()
    }

    /// Returns the line's underlying axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Axis2, Line2D, Point2, Vector2};
    /// let axis = Axis2::new(Point2::ORIGIN, Vector2::X).unwrap();
    /// let line = Line2D::from_axis(axis);
    /// assert_eq!(line.axis(), axis);
    /// ```
    pub fn axis(&self) -> Axis2 {
        self.axis
    }

    /// Evaluates the point on the line at parameter `u`: `origin + u * direction`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Line2D, Point2, Vector2};
    /// let line = Line2D::new(Point2::ORIGIN, Vector2::X).unwrap();
    /// assert_eq!(line.eval_point(3.0), Point2::new(3.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64) -> Point2 {
        analytic::line2d_d0(&self.axis, u)
    }

    /// Evaluates the points on the line at each parameter in `us`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Line2D, Point2, Vector2};
    /// let line = Line2D::new(Point2::ORIGIN, Vector2::X).unwrap();
    /// let points = line.eval_points(&[0.0, 1.0, 2.0]);
    /// assert_eq!(points, vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0), Point2::new(2.0, 0.0)]);
    /// ```
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point2> {
        us.iter().map(|&u| self.eval_point(u)).collect()
    }

    /// Evaluates the derivative of the given `order` at parameter `u`.
    ///
    /// The line is affine in `u`, so the first derivative is the constant
    /// direction vector and every derivative of order 2 or higher is zero.
    /// `u` does not affect the result (included for API consistency with
    /// curved parametric types).
    ///
    /// # Panics
    ///
    /// Panics if `order == 0`; use [`Line2D::eval_point`] to evaluate the
    /// position itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Line2D, Point2, Vector2};
    /// let line = Line2D::new(Point2::ORIGIN, Vector2::new(0.0, 3.0)).unwrap();
    /// assert_eq!(line.eval_derivative(1.5, 1), Vector2::Y);
    /// assert_eq!(line.eval_derivative(1.5, 2), Vector2::ZERO);
    /// ```
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector2 {
        let _ = u;
        match order {
            0 => panic!("eval_derivative: order must be >= 1 (use eval_point for order 0)"),
            1 => self.axis.direction(),
            _ => Vector2::ZERO,
        }
    }

    /// Recovers the parameter `u` of a point on (or near) the line: the
    /// signed projection `(point - origin) . direction`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Line2D, Point2, Vector2};
    /// let line = Line2D::new(Point2::ORIGIN, Vector2::X).unwrap();
    /// assert_eq!(line.parameter_of(Point2::new(2.5, 0.0)), 2.5);
    /// ```
    pub fn parameter_of(&self, point: Point2) -> f64 {
        analytic::line2d_parameter(&self.axis, point)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Axis2, Axis3, Line2D, Line3D, LineConstructionError, Point2, Point3, Vector2, Vector3,
    };

    // ---- Line3D construction ----

    #[test]
    fn test_line3d_new_normalizes_direction() {
        let line = Line3D::new(Point3::ORIGIN, Vector3::new(2.0, 0.0, 0.0)).unwrap();
        assert_eq!(line.origin(), Point3::ORIGIN);
        assert_eq!(line.direction(), Vector3::X);
    }

    #[test]
    fn test_line3d_new_null_direction_errors() {
        assert_eq!(
            Line3D::new(Point3::ORIGIN, Vector3::ZERO),
            Err(LineConstructionError::NullDirection)
        );
    }

    #[test]
    fn test_line3d_from_axis() {
        let axis = Axis3::new(Point3::new(1.0, 2.0, 3.0), Vector3::Y).unwrap();
        let line = Line3D::from_axis(axis);
        assert_eq!(line.origin(), Point3::new(1.0, 2.0, 3.0));
        assert_eq!(line.direction(), Vector3::Y);
        assert_eq!(line.axis(), axis);
    }

    #[test]
    fn test_line3d_from_two_points() {
        let p1 = Point3::new(0.0, 0.0, 0.0);
        let p2 = Point3::new(2.0, 0.0, 0.0);
        let line = Line3D::from_two_points(p1, p2).unwrap();
        assert_eq!(line.origin(), p1);
        assert_eq!(line.direction(), Vector3::X);
    }

    #[test]
    fn test_line3d_from_two_points_confused_errors() {
        let p = Point3::new(1.0, 2.0, 3.0);
        assert_eq!(
            Line3D::from_two_points(p, p),
            Err(LineConstructionError::ConfusedPoints)
        );
    }

    // ---- Line3D evaluation ----

    #[test]
    fn test_line3d_eval_point() {
        let line = Line3D::new(Point3::ORIGIN, Vector3::X).unwrap();
        assert_eq!(line.eval_point(3.0), Point3::new(3.0, 0.0, 0.0));
    }

    #[test]
    fn test_line3d_eval_points_matches_loop() {
        let line = Line3D::new(Point3::new(1.0, -2.0, 0.5), Vector3::Y).unwrap();
        let us = [0.0, 1.5, -3.0];
        let expected: Vec<Point3> = us.iter().map(|&u| line.eval_point(u)).collect();
        assert_eq!(line.eval_points(&us), expected);
    }

    #[test]
    fn test_line3d_eval_derivative_order1_is_direction() {
        let line = Line3D::new(Point3::ORIGIN, Vector3::new(0.0, 3.0, 0.0)).unwrap();
        assert_eq!(line.eval_derivative(1.5, 1), Vector3::Y);
    }

    #[test]
    fn test_line3d_eval_derivative_order2_is_zero() {
        let line = Line3D::new(Point3::ORIGIN, Vector3::X).unwrap();
        assert_eq!(line.eval_derivative(1.5, 2), Vector3::ZERO);
    }

    #[test]
    fn test_line3d_eval_derivative_order3_is_zero() {
        let line = Line3D::new(Point3::ORIGIN, Vector3::X).unwrap();
        assert_eq!(line.eval_derivative(1.5, 3), Vector3::ZERO);
    }

    #[test]
    #[should_panic]
    fn test_line3d_eval_derivative_order0_panics() {
        let line = Line3D::new(Point3::ORIGIN, Vector3::X).unwrap();
        line.eval_derivative(1.5, 0);
    }

    #[test]
    fn test_line3d_parameter_of_round_trip() {
        let line = Line3D::new(Point3::new(1.0, -2.0, 0.5), Vector3::new(1.0, 2.0, 2.0)).unwrap();
        for u in [0.3, 2.0, -5.5] {
            let p = line.eval_point(u);
            assert!((line.parameter_of(p) - u).abs() < 1e-9);
        }
    }

    // ---- Line2D construction ----

    #[test]
    fn test_line2d_new_normalizes_direction() {
        let line = Line2D::new(Point2::ORIGIN, Vector2::new(0.0, 5.0)).unwrap();
        assert_eq!(line.origin(), Point2::ORIGIN);
        assert_eq!(line.direction(), Vector2::Y);
    }

    #[test]
    fn test_line2d_new_null_direction_errors() {
        assert_eq!(
            Line2D::new(Point2::ORIGIN, Vector2::ZERO),
            Err(LineConstructionError::NullDirection)
        );
    }

    #[test]
    fn test_line2d_from_axis() {
        let axis = Axis2::new(Point2::new(1.0, 2.0), Vector2::X).unwrap();
        let line = Line2D::from_axis(axis);
        assert_eq!(line.origin(), Point2::new(1.0, 2.0));
        assert_eq!(line.direction(), Vector2::X);
        assert_eq!(line.axis(), axis);
    }

    #[test]
    fn test_line2d_from_two_points() {
        let p1 = Point2::new(0.0, 0.0);
        let p2 = Point2::new(0.0, 4.0);
        let line = Line2D::from_two_points(p1, p2).unwrap();
        assert_eq!(line.origin(), p1);
        assert_eq!(line.direction(), Vector2::Y);
    }

    #[test]
    fn test_line2d_from_two_points_confused_errors() {
        let p = Point2::new(1.0, 2.0);
        assert_eq!(
            Line2D::from_two_points(p, p),
            Err(LineConstructionError::ConfusedPoints)
        );
    }

    // ---- Line2D evaluation ----

    #[test]
    fn test_line2d_eval_point() {
        let line = Line2D::new(Point2::ORIGIN, Vector2::X).unwrap();
        assert_eq!(line.eval_point(3.0), Point2::new(3.0, 0.0));
    }

    #[test]
    fn test_line2d_eval_points_matches_loop() {
        let line = Line2D::new(Point2::new(1.0, -2.0), Vector2::Y).unwrap();
        let us = [0.0, 1.5, -3.0];
        let expected: Vec<Point2> = us.iter().map(|&u| line.eval_point(u)).collect();
        assert_eq!(line.eval_points(&us), expected);
    }

    #[test]
    fn test_line2d_eval_derivative_order1_is_direction() {
        let line = Line2D::new(Point2::ORIGIN, Vector2::new(0.0, 3.0)).unwrap();
        assert_eq!(line.eval_derivative(1.5, 1), Vector2::Y);
    }

    #[test]
    fn test_line2d_eval_derivative_order2_is_zero() {
        let line = Line2D::new(Point2::ORIGIN, Vector2::X).unwrap();
        assert_eq!(line.eval_derivative(1.5, 2), Vector2::ZERO);
    }

    #[test]
    #[should_panic]
    fn test_line2d_eval_derivative_order0_panics() {
        let line = Line2D::new(Point2::ORIGIN, Vector2::X).unwrap();
        line.eval_derivative(1.5, 0);
    }

    #[test]
    fn test_line2d_parameter_of_round_trip() {
        let line = Line2D::new(Point2::new(1.0, -2.0), Vector2::new(3.0, 4.0)).unwrap();
        for u in [0.3, 2.0, 5.5] {
            let p = line.eval_point(u);
            assert!((line.parameter_of(p) - u).abs() < 1e-9);
        }
    }

    // ---- LineConstructionError ----

    #[test]
    fn test_line_construction_error_display() {
        assert_eq!(
            LineConstructionError::NullDirection.to_string(),
            "direction has zero length"
        );
        assert_eq!(
            LineConstructionError::ConfusedPoints.to_string(),
            "the two points are confused"
        );
    }

    #[test]
    fn test_line_construction_error_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&LineConstructionError::NullDirection);
    }
}
