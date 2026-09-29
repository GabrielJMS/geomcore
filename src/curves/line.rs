//! Infinite lines in 2D and 3D: parametric evaluation and parameter
//! inversion, thin wrappers over [`crate::curve_math::analytic`].

use crate::curve_math::analytic;
use crate::curves::Curve2D;
use crate::curves::parametrize::{self, ParametrizeError};
use crate::surfaces::Surface;
use crate::{Axis2D, Axis3D, Point2D, Point3D, Tolerance, Vector2D, Vector3D};
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
/// use geomcore::{Line3D, Point3D, Vector3D};
/// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
/// assert_eq!(line.eval_point(3.0), Point3D::new(3.0, 0.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line3D {
    axis: Axis3D,
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
    /// use geomcore::{Line3D, Point3D, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::new(2.0, 0.0, 0.0)).unwrap();
    /// assert_eq!(line.direction(), Vector3D::X);
    /// ```
    pub fn new(origin: Point3D, direction: Vector3D) -> Result<Line3D, LineConstructionError> {
        let axis =
            Axis3D::new(origin, direction).map_err(|_| LineConstructionError::NullDirection)?;
        Ok(Line3D { axis })
    }

    /// Creates a line from an axis directly.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Axis3D, Line3D, Point3D, Vector3D};
    /// let axis = Axis3D::new(Point3D::ORIGIN, Vector3D::Y).unwrap();
    /// let line = Line3D::from_axis(axis);
    /// assert_eq!(line.axis(), axis);
    /// ```
    pub fn from_axis(axis: Axis3D) -> Line3D {
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
    /// use geomcore::{Line3D, Point3D};
    /// let line = Line3D::from_two_points(Point3D::ORIGIN, Point3D::new(2.0, 0.0, 0.0)).unwrap();
    /// assert_eq!(line.origin(), Point3D::ORIGIN);
    /// ```
    pub fn from_two_points(p1: Point3D, p2: Point3D) -> Result<Line3D, LineConstructionError> {
        let direction = (p2 - p1)
            .normalized()
            .ok_or(LineConstructionError::ConfusedPoints)?;
        Ok(Line3D {
            axis: Axis3D::new(p1, direction).map_err(|_| LineConstructionError::ConfusedPoints)?,
        })
    }

    /// Returns the origin point of the line.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D, Vector3D};
    /// let line = Line3D::new(Point3D::new(1.0, 2.0, 3.0), Vector3D::X).unwrap();
    /// assert_eq!(line.origin(), Point3D::new(1.0, 2.0, 3.0));
    /// ```
    pub fn origin(&self) -> Point3D {
        self.axis.origin()
    }

    /// Returns the unit direction of the line.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::new(0.0, 5.0, 0.0)).unwrap();
    /// assert_eq!(line.direction(), Vector3D::Y);
    /// ```
    pub fn direction(&self) -> Vector3D {
        self.axis.direction()
    }

    /// Returns the line's underlying axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Axis3D, Line3D, Point3D, Vector3D};
    /// let axis = Axis3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let line = Line3D::from_axis(axis);
    /// assert_eq!(line.axis(), axis);
    /// ```
    pub fn axis(&self) -> Axis3D {
        self.axis
    }

    /// Evaluates the point on the line at parameter `u`: `origin + u * direction`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// assert_eq!(line.eval_point(3.0), Point3D::new(3.0, 0.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64) -> Point3D {
        analytic::line_d0(&self.axis, u)
    }

    /// Evaluates the points on the line at each parameter in `us`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let points = line.eval_points(&[0.0, 1.0, 2.0]);
    /// assert_eq!(points, vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(1.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)]);
    /// ```
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
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
    /// use geomcore::{Line3D, Point3D, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::new(0.0, 3.0, 0.0)).unwrap();
    /// assert_eq!(line.eval_derivative(1.5, 1), Vector3D::Y);
    /// assert_eq!(line.eval_derivative(1.5, 2), Vector3D::ZERO);
    /// ```
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        let _ = u;
        match order {
            0 => panic!("eval_derivative: order must be >= 1 (use eval_point for order 0)"),
            1 => analytic::line_d1(&self.axis),
            _ => Vector3D::ZERO,
        }
    }

    /// Recovers the parameter `u` of a point on (or near) the line: the
    /// signed projection `(point - origin) . direction`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// assert_eq!(line.parameter_of(Point3D::new(2.5, 0.0, 0.0)), 2.5);
    /// ```
    pub fn parameter_of(&self, point: Point3D) -> f64 {
        analytic::line_parameter(&self.axis, point)
    }

    /// Returns whether `point` lies on the line: the inverse parameter is
    /// recovered with [`Line3D::parameter_of`] and re-evaluated, and the
    /// point counts as contained when the re-evaluated point is within
    /// `tol.confusion` of it.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D, Tolerance, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// assert!(line.contains(Point3D::new(2.5, 0.0, 0.0), tol));
    /// assert!(!line.contains(Point3D::new(2.5, 1.0, 0.0), tol));
    /// ```
    pub fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        let u = self.parameter_of(point);
        self.eval_point(u).distance(point) <= tol.confusion
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
    /// geomcore safeguard, the candidate 2D image is verified against the
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
    /// use geomcore::curves::{Curve2D, ParametricCurve2D};
    /// use geomcore::{Cylinder, Line3D, Point3D, Vector3D};
    ///
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let line = Line3D::new(Point3D::new(2.0, 0.0, 0.0), Vector3D::Z).unwrap();
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
/// use geomcore::{Line2D, Point2D, Vector2D};
/// let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
/// assert_eq!(line.eval_point(3.0), Point2D::new(3.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line2D {
    axis: Axis2D,
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
    /// use geomcore::{Line2D, Point2D, Vector2D};
    /// let line = Line2D::new(Point2D::ORIGIN, Vector2D::new(0.0, 5.0)).unwrap();
    /// assert_eq!(line.direction(), Vector2D::Y);
    /// ```
    pub fn new(origin: Point2D, direction: Vector2D) -> Result<Line2D, LineConstructionError> {
        let axis =
            Axis2D::new(origin, direction).map_err(|_| LineConstructionError::NullDirection)?;
        Ok(Line2D { axis })
    }

    /// Creates a line from an axis directly.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Axis2D, Line2D, Point2D, Vector2D};
    /// let axis = Axis2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// let line = Line2D::from_axis(axis);
    /// assert_eq!(line.axis(), axis);
    /// ```
    pub fn from_axis(axis: Axis2D) -> Line2D {
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
    /// use geomcore::{Line2D, Point2D};
    /// let line = Line2D::from_two_points(Point2D::ORIGIN, Point2D::new(0.0, 4.0)).unwrap();
    /// assert_eq!(line.origin(), Point2D::ORIGIN);
    /// ```
    pub fn from_two_points(p1: Point2D, p2: Point2D) -> Result<Line2D, LineConstructionError> {
        let direction = (p2 - p1)
            .normalized()
            .ok_or(LineConstructionError::ConfusedPoints)?;
        Ok(Line2D {
            axis: Axis2D::new(p1, direction).map_err(|_| LineConstructionError::ConfusedPoints)?,
        })
    }

    /// Returns the origin point of the line.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line2D, Point2D, Vector2D};
    /// let line = Line2D::new(Point2D::new(1.0, 2.0), Vector2D::X).unwrap();
    /// assert_eq!(line.origin(), Point2D::new(1.0, 2.0));
    /// ```
    pub fn origin(&self) -> Point2D {
        self.axis.origin()
    }

    /// Returns the unit direction of the line.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line2D, Point2D, Vector2D};
    /// let line = Line2D::new(Point2D::ORIGIN, Vector2D::new(0.0, 5.0)).unwrap();
    /// assert_eq!(line.direction(), Vector2D::Y);
    /// ```
    pub fn direction(&self) -> Vector2D {
        self.axis.direction()
    }

    /// Returns the line's underlying axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Axis2D, Line2D, Point2D, Vector2D};
    /// let axis = Axis2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// let line = Line2D::from_axis(axis);
    /// assert_eq!(line.axis(), axis);
    /// ```
    pub fn axis(&self) -> Axis2D {
        self.axis
    }

    /// Evaluates the point on the line at parameter `u`: `origin + u * direction`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line2D, Point2D, Vector2D};
    /// let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// assert_eq!(line.eval_point(3.0), Point2D::new(3.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64) -> Point2D {
        analytic::line2d_d0(&self.axis, u)
    }

    /// Evaluates the points on the line at each parameter in `us`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line2D, Point2D, Vector2D};
    /// let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// let points = line.eval_points(&[0.0, 1.0, 2.0]);
    /// assert_eq!(points, vec![Point2D::new(0.0, 0.0), Point2D::new(1.0, 0.0), Point2D::new(2.0, 0.0)]);
    /// ```
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point2D> {
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
    /// use geomcore::{Line2D, Point2D, Vector2D};
    /// let line = Line2D::new(Point2D::ORIGIN, Vector2D::new(0.0, 3.0)).unwrap();
    /// assert_eq!(line.eval_derivative(1.5, 1), Vector2D::Y);
    /// assert_eq!(line.eval_derivative(1.5, 2), Vector2D::ZERO);
    /// ```
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector2D {
        let _ = u;
        match order {
            0 => panic!("eval_derivative: order must be >= 1 (use eval_point for order 0)"),
            1 => self.axis.direction(),
            _ => Vector2D::ZERO,
        }
    }

    /// Recovers the parameter `u` of a point on (or near) the line: the
    /// signed projection `(point - origin) . direction`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line2D, Point2D, Vector2D};
    /// let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// assert_eq!(line.parameter_of(Point2D::new(2.5, 0.0)), 2.5);
    /// ```
    pub fn parameter_of(&self, point: Point2D) -> f64 {
        analytic::line2d_parameter(&self.axis, point)
    }

    /// Returns whether `point` lies on the line: the inverse parameter is
    /// recovered with [`Line2D::parameter_of`] and re-evaluated, and the
    /// point counts as contained when the re-evaluated point is within
    /// `tol.confusion` of it.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line2D, Point2D, Tolerance, Vector2D};
    /// let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// assert!(line.contains(Point2D::new(2.5, 0.0), tol));
    /// assert!(!line.contains(Point2D::new(2.5, 1.0), tol));
    /// ```
    pub fn contains(&self, point: Point2D, tol: Tolerance) -> bool {
        let u = self.parameter_of(point);
        self.eval_point(u).distance(point) <= tol.confusion
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Axis2D, Axis3D, Line2D, Line3D, LineConstructionError, Point2D, Point3D, Tolerance,
        Vector2D, Vector3D,
    };

    // ---- Line3D construction ----

    #[test]
    fn test_line3d_new_normalizes_direction() {
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::new(2.0, 0.0, 0.0)).unwrap();
        assert_eq!(line.origin(), Point3D::ORIGIN);
        assert_eq!(line.direction(), Vector3D::X);
    }

    #[test]
    fn test_line3d_new_null_direction_errors() {
        assert_eq!(
            Line3D::new(Point3D::ORIGIN, Vector3D::ZERO),
            Err(LineConstructionError::NullDirection)
        );
    }

    #[test]
    fn test_line3d_from_axis() {
        let axis = Axis3D::new(Point3D::new(1.0, 2.0, 3.0), Vector3D::Y).unwrap();
        let line = Line3D::from_axis(axis);
        assert_eq!(line.origin(), Point3D::new(1.0, 2.0, 3.0));
        assert_eq!(line.direction(), Vector3D::Y);
        assert_eq!(line.axis(), axis);
    }

    #[test]
    fn test_line3d_from_two_points() {
        let p1 = Point3D::new(0.0, 0.0, 0.0);
        let p2 = Point3D::new(2.0, 0.0, 0.0);
        let line = Line3D::from_two_points(p1, p2).unwrap();
        assert_eq!(line.origin(), p1);
        assert_eq!(line.direction(), Vector3D::X);
    }

    #[test]
    fn test_line3d_from_two_points_confused_errors() {
        let p = Point3D::new(1.0, 2.0, 3.0);
        assert_eq!(
            Line3D::from_two_points(p, p),
            Err(LineConstructionError::ConfusedPoints)
        );
    }

    // ---- Line3D evaluation ----

    #[test]
    fn test_line3d_eval_point() {
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        assert_eq!(line.eval_point(3.0), Point3D::new(3.0, 0.0, 0.0));
    }

    #[test]
    fn test_line3d_eval_points_matches_loop() {
        let line = Line3D::new(Point3D::new(1.0, -2.0, 0.5), Vector3D::Y).unwrap();
        let us = [0.0, 1.5, -3.0];
        let expected: Vec<Point3D> = us.iter().map(|&u| line.eval_point(u)).collect();
        assert_eq!(line.eval_points(&us), expected);
    }

    #[test]
    fn test_line3d_eval_derivative_order1_is_direction() {
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::new(0.0, 3.0, 0.0)).unwrap();
        assert_eq!(line.eval_derivative(1.5, 1), Vector3D::Y);
    }

    #[test]
    fn test_line3d_eval_derivative_order2_is_zero() {
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        assert_eq!(line.eval_derivative(1.5, 2), Vector3D::ZERO);
    }

    #[test]
    fn test_line3d_eval_derivative_order3_is_zero() {
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        assert_eq!(line.eval_derivative(1.5, 3), Vector3D::ZERO);
    }

    #[test]
    #[should_panic]
    fn test_line3d_eval_derivative_order0_panics() {
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        line.eval_derivative(1.5, 0);
    }

    #[test]
    fn test_line3d_parameter_of_round_trip() {
        let line = Line3D::new(Point3D::new(1.0, -2.0, 0.5), Vector3D::new(1.0, 2.0, 2.0)).unwrap();
        for u in [0.3, 2.0, -5.5] {
            let p = line.eval_point(u);
            assert!((line.parameter_of(p) - u).abs() < 1e-9);
        }
    }

    // ---- Line2D construction ----

    #[test]
    fn test_line2d_new_normalizes_direction() {
        let line = Line2D::new(Point2D::ORIGIN, Vector2D::new(0.0, 5.0)).unwrap();
        assert_eq!(line.origin(), Point2D::ORIGIN);
        assert_eq!(line.direction(), Vector2D::Y);
    }

    #[test]
    fn test_line2d_new_null_direction_errors() {
        assert_eq!(
            Line2D::new(Point2D::ORIGIN, Vector2D::ZERO),
            Err(LineConstructionError::NullDirection)
        );
    }

    #[test]
    fn test_line2d_from_axis() {
        let axis = Axis2D::new(Point2D::new(1.0, 2.0), Vector2D::X).unwrap();
        let line = Line2D::from_axis(axis);
        assert_eq!(line.origin(), Point2D::new(1.0, 2.0));
        assert_eq!(line.direction(), Vector2D::X);
        assert_eq!(line.axis(), axis);
    }

    #[test]
    fn test_line2d_from_two_points() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(0.0, 4.0);
        let line = Line2D::from_two_points(p1, p2).unwrap();
        assert_eq!(line.origin(), p1);
        assert_eq!(line.direction(), Vector2D::Y);
    }

    #[test]
    fn test_line2d_from_two_points_confused_errors() {
        let p = Point2D::new(1.0, 2.0);
        assert_eq!(
            Line2D::from_two_points(p, p),
            Err(LineConstructionError::ConfusedPoints)
        );
    }

    // ---- Line2D evaluation ----

    #[test]
    fn test_line2d_eval_point() {
        let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
        assert_eq!(line.eval_point(3.0), Point2D::new(3.0, 0.0));
    }

    #[test]
    fn test_line2d_eval_points_matches_loop() {
        let line = Line2D::new(Point2D::new(1.0, -2.0), Vector2D::Y).unwrap();
        let us = [0.0, 1.5, -3.0];
        let expected: Vec<Point2D> = us.iter().map(|&u| line.eval_point(u)).collect();
        assert_eq!(line.eval_points(&us), expected);
    }

    #[test]
    fn test_line2d_eval_derivative_order1_is_direction() {
        let line = Line2D::new(Point2D::ORIGIN, Vector2D::new(0.0, 3.0)).unwrap();
        assert_eq!(line.eval_derivative(1.5, 1), Vector2D::Y);
    }

    #[test]
    fn test_line2d_eval_derivative_order2_is_zero() {
        let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
        assert_eq!(line.eval_derivative(1.5, 2), Vector2D::ZERO);
    }

    #[test]
    #[should_panic]
    fn test_line2d_eval_derivative_order0_panics() {
        let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
        line.eval_derivative(1.5, 0);
    }

    #[test]
    fn test_line2d_parameter_of_round_trip() {
        let line = Line2D::new(Point2D::new(1.0, -2.0), Vector2D::new(3.0, 4.0)).unwrap();
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

    #[test]
    fn test_line3d_contains() {
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let tol = Tolerance::DEFAULT;
        assert!(line.contains(Point3D::new(2.5, 0.0, 0.0), tol));
        assert!(line.contains(line.eval_point(-3.0), tol));
        assert!(!line.contains(Point3D::new(2.5, 1.0, 0.0), tol));
        // Tolerance boundary: 1e-6 off the line fails at default, passes loose.
        let near = Point3D::new(0.0, 1e-6, 0.0);
        assert!(!line.contains(near, tol));
        assert!(line.contains(
            near,
            Tolerance {
                confusion: 1e-5,
                ..tol
            }
        ));
    }

    #[test]
    fn test_line2d_contains() {
        let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
        let tol = Tolerance::DEFAULT;
        assert!(line.contains(Point2D::new(2.5, 0.0), tol));
        assert!(line.contains(line.eval_point(-3.0), tol));
        assert!(!line.contains(Point2D::new(2.5, 1.0), tol));
    }
}
