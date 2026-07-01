//! Circles in 2D and 3D: parametric evaluation, parameter inversion, and
//! circumcircle construction, thin wrappers over
//! [`crate::curve_math::analytic`].

use crate::curve_math::analytic;
use crate::tol;
use crate::{Axis3, Frame2, Frame3, Point2, Point3, Vector2, Vector3};
use std::fmt;

/// Error returned when a [`Circle3D`] or [`Circle2D`] cannot be constructed
/// from the given inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircleConstructionError {
    /// The requested radius is negative.
    NegativeRadius,
    /// The normal (or main axis direction) has zero length.
    NullNormal,
    /// Two (or more) of the three points given to build the circle are
    /// coincident (or too close to distinguish).
    ConfusedPoints,
    /// The three points given to build the circle are collinear, so no
    /// (finite-radius) circle passes through all three.
    CollinearPoints,
}

impl fmt::Display for CircleConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            CircleConstructionError::NegativeRadius => "radius is negative",
            CircleConstructionError::NullNormal => "normal has zero length",
            CircleConstructionError::ConfusedPoints => "the points are confused",
            CircleConstructionError::CollinearPoints => "the points are collinear",
        };
        f.write_str(message)
    }
}

impl std::error::Error for CircleConstructionError {}

/// A circle in 3D: a plane [`Frame3`] (origin plus local x/y directions
/// defining the plane and the angular origin) and a radius, evaluated as
/// `origin + radius*cos(u)*x_dir + radius*sin(u)*y_dir`.
///
/// # Examples
///
/// ```
/// use geomrust::{Circle3D, Point3, Vector3};
/// let circle = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
/// assert_eq!(circle.eval_point(0.0), Point3::new(2.0, 0.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle3D {
    frame: Frame3,
    radius: f64,
}

impl Circle3D {
    /// Creates a circle from a center, a normal, and a radius.
    ///
    /// The plane frame is derived from `normal` via [`Frame3::from_z`].
    ///
    /// # Errors
    ///
    /// Returns [`CircleConstructionError::NullNormal`] if `normal` cannot be
    /// normalized (zero length), or
    /// [`CircleConstructionError::NegativeRadius`] if `radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle3D, Point3, Vector3};
    /// let circle = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
    /// assert_eq!(circle.radius(), 2.0);
    /// assert_eq!(circle.normal(), Vector3::Z);
    /// ```
    pub fn new(
        center: Point3,
        normal: Vector3,
        radius: f64,
    ) -> Result<Circle3D, CircleConstructionError> {
        let frame =
            Frame3::from_z(center, normal).map_err(|_| CircleConstructionError::NullNormal)?;
        Circle3D::from_frame(frame, radius)
    }

    /// Creates a circle from a main axis (center and normal) and a radius.
    ///
    /// # Errors
    ///
    /// Returns [`CircleConstructionError::NegativeRadius`] if `radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Axis3, Circle3D, Point3, Vector3};
    /// let axis = Axis3::new(Point3::ORIGIN, Vector3::Z).unwrap();
    /// let circle = Circle3D::from_axis(axis, 2.0).unwrap();
    /// assert_eq!(circle.center(), Point3::ORIGIN);
    /// ```
    pub fn from_axis(axis: Axis3, radius: f64) -> Result<Circle3D, CircleConstructionError> {
        let frame = Frame3::from_z(axis.origin(), axis.direction())
            .map_err(|_| CircleConstructionError::NullNormal)?;
        Circle3D::from_frame(frame, radius)
    }

    /// Creates a circle from a plane frame and a radius directly.
    ///
    /// # Errors
    ///
    /// Returns [`CircleConstructionError::NegativeRadius`] if `radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle3D, Frame3};
    /// let circle = Circle3D::from_frame(Frame3::WORLD, 2.0).unwrap();
    /// assert_eq!(circle.frame(), Frame3::WORLD);
    /// ```
    pub fn from_frame(frame: Frame3, radius: f64) -> Result<Circle3D, CircleConstructionError> {
        if radius < 0.0 {
            return Err(CircleConstructionError::NegativeRadius);
        }
        Ok(Circle3D { frame, radius })
    }

    /// Creates the circumcircle through three points.
    ///
    /// Failure checks run before any circumcenter computation: any pairwise
    /// distance below [`crate::tol::CONFUSION`] is reported as
    /// [`CircleConstructionError::ConfusedPoints`] (this also covers all
    /// three points being coincident); otherwise, if the points are
    /// collinear (`|(p2-p1) x (p3-p1)| <= tol::CONFUSION * max(|p2-p1|,
    /// |p3-p1|)`), [`CircleConstructionError::CollinearPoints`] is returned.
    ///
    /// The center is computed via the standard circumcenter closed form:
    /// with `a = p1 - p3`, `b = p2 - p3`, `n = a x b`,
    /// `center = p3 + ((|a|^2 * b - |b|^2 * a) x n) / (2 * |n|^2)`. The
    /// radius is `|center - p1|`. The resulting frame's normal is
    /// `normalize((p2-p1) x (p3-p2))` and its x direction is derived from
    /// the hint `p1 - center` (rejected perpendicular to the normal),
    /// matching the reference implementation's axis convention.
    ///
    /// # Errors
    ///
    /// Returns [`CircleConstructionError::ConfusedPoints`] or
    /// [`CircleConstructionError::CollinearPoints`] as described above.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle3D, Point3};
    /// let circle = Circle3D::from_three_points(
    ///     Point3::new(1.0, 0.0, 0.0),
    ///     Point3::new(0.0, 1.0, 0.0),
    ///     Point3::new(-1.0, 0.0, 0.0),
    /// )
    /// .unwrap();
    /// assert!((circle.radius() - 1.0).abs() < 1e-9);
    /// ```
    pub fn from_three_points(
        p1: Point3,
        p2: Point3,
        p3: Point3,
    ) -> Result<Circle3D, CircleConstructionError> {
        if p1.distance(p2) < tol::CONFUSION
            || p2.distance(p3) < tol::CONFUSION
            || p1.distance(p3) < tol::CONFUSION
        {
            return Err(CircleConstructionError::ConfusedPoints);
        }

        let v12 = p2 - p1;
        let v13 = p3 - p1;
        let collinear_cross = v12.cross(v13).magnitude();
        if collinear_cross <= tol::CONFUSION * v12.magnitude().max(v13.magnitude()) {
            return Err(CircleConstructionError::CollinearPoints);
        }

        let a = p1 - p3;
        let b = p2 - p3;
        let n = a.cross(b);
        let center = p3
            + ((a.square_magnitude() * b - b.square_magnitude() * a).cross(n))
                * (1.0 / (2.0 * n.square_magnitude()));
        let radius = center.distance(p1);

        let normal = (p2 - p1).cross(p3 - p2);
        let frame = Frame3::new(center, normal, p1 - center)
            .map_err(|_| CircleConstructionError::CollinearPoints)?;

        Circle3D::from_frame(frame, radius)
    }

    /// Returns the center of the circle.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle3D, Point3, Vector3};
    /// let circle = Circle3D::new(Point3::new(1.0, 2.0, 3.0), Vector3::Z, 2.0).unwrap();
    /// assert_eq!(circle.center(), Point3::new(1.0, 2.0, 3.0));
    /// ```
    pub fn center(&self) -> Point3 {
        self.frame.origin()
    }

    /// Returns the radius of the circle.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle3D, Point3, Vector3};
    /// let circle = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
    /// assert_eq!(circle.radius(), 2.0);
    /// ```
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Returns the circle's plane frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle3D, Frame3};
    /// let circle = Circle3D::from_frame(Frame3::WORLD, 2.0).unwrap();
    /// assert_eq!(circle.frame(), Frame3::WORLD);
    /// ```
    pub fn frame(&self) -> Frame3 {
        self.frame
    }

    /// Returns the unit normal of the circle's plane.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle3D, Point3, Vector3};
    /// let circle = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
    /// assert_eq!(circle.normal(), Vector3::Z);
    /// ```
    pub fn normal(&self) -> Vector3 {
        self.frame.z_direction()
    }

    /// Evaluates the point on the circle at angular parameter `u`:
    /// `center + radius*cos(u)*x_dir + radius*sin(u)*y_dir`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle3D, Point3, Vector3};
    /// let circle = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
    /// assert_eq!(circle.eval_point(0.0), Point3::new(2.0, 0.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64) -> Point3 {
        analytic::circle_d0(&self.frame, self.radius, u)
    }

    /// Evaluates the points on the circle at each parameter in `us`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle3D, Point3, Vector3};
    /// let circle = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
    /// let points = circle.eval_points(&[0.0, 1.0]);
    /// assert_eq!(points[0], Point3::new(2.0, 0.0, 0.0));
    /// ```
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point3> {
        us.iter().map(|&u| self.eval_point(u)).collect()
    }

    /// Evaluates the derivative of the given `order` at parameter `u`.
    ///
    /// Derivatives cycle with period 4 in `order` (see
    /// [`crate::curve_math::analytic::circle_dn`]).
    ///
    /// # Panics
    ///
    /// Panics if `order == 0`; use [`Circle3D::eval_point`] to evaluate the
    /// position itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle3D, Point3, Vector3};
    /// let circle = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
    /// assert_eq!(circle.eval_derivative(0.0, 1), Vector3::new(0.0, 2.0, 0.0));
    /// ```
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector3 {
        match order {
            0 => panic!("eval_derivative: order must be >= 1 (use eval_point for order 0)"),
            _ => analytic::circle_dn(&self.frame, self.radius, u, order),
        }
    }

    /// Recovers the angular parameter of a point on (or near) the circle,
    /// wrapped into `[0, 2*PI)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle3D, Point3, Vector3};
    /// let circle = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
    /// assert!((circle.parameter_of(Point3::new(0.0, 2.0, 0.0)) - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    /// ```
    pub fn parameter_of(&self, point: Point3) -> f64 {
        analytic::circle_parameter(&self.frame, point)
    }
}

/// A circle in 2D: a [`Frame2`] (origin plus local x/y directions defining
/// the angular origin) and a radius, evaluated as
/// `origin + radius*cos(u)*x_dir + radius*sin(u)*y_dir`.
///
/// # Examples
///
/// ```
/// use geomrust::{Circle2D, Point2};
/// let circle = Circle2D::new(Point2::ORIGIN, 2.0).unwrap();
/// assert_eq!(circle.eval_point(0.0), Point2::new(2.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle2D {
    frame: Frame2,
    radius: f64,
}

impl Circle2D {
    /// Creates a circle from a center and a radius, using a world-aligned
    /// frame (x/y directions matching [`Vector2::X`]/[`Vector2::Y`]) at
    /// `center`.
    ///
    /// # Errors
    ///
    /// Returns [`CircleConstructionError::NegativeRadius`] if `radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle2D, Point2};
    /// let circle = Circle2D::new(Point2::new(1.0, 2.0), 3.0).unwrap();
    /// assert_eq!(circle.center(), Point2::new(1.0, 2.0));
    /// ```
    pub fn new(center: Point2, radius: f64) -> Result<Circle2D, CircleConstructionError> {
        let frame = Frame2::new(center, Vector2::X, Vector2::Y)
            .expect("Vector2::X and Vector2::Y are orthonormal by construction");
        Circle2D::from_frame(frame, radius)
    }

    /// Creates a circle from a frame and a radius directly.
    ///
    /// # Errors
    ///
    /// Returns [`CircleConstructionError::NegativeRadius`] if `radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle2D, Frame2};
    /// let circle = Circle2D::from_frame(Frame2::WORLD, 2.0).unwrap();
    /// assert_eq!(circle.frame(), Frame2::WORLD);
    /// ```
    pub fn from_frame(frame: Frame2, radius: f64) -> Result<Circle2D, CircleConstructionError> {
        if radius < 0.0 {
            return Err(CircleConstructionError::NegativeRadius);
        }
        Ok(Circle2D { frame, radius })
    }

    /// Returns the center of the circle.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle2D, Point2};
    /// let circle = Circle2D::new(Point2::new(1.0, 2.0), 2.0).unwrap();
    /// assert_eq!(circle.center(), Point2::new(1.0, 2.0));
    /// ```
    pub fn center(&self) -> Point2 {
        self.frame.origin()
    }

    /// Returns the radius of the circle.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle2D, Point2};
    /// let circle = Circle2D::new(Point2::ORIGIN, 2.0).unwrap();
    /// assert_eq!(circle.radius(), 2.0);
    /// ```
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Returns the circle's frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle2D, Frame2};
    /// let circle = Circle2D::from_frame(Frame2::WORLD, 2.0).unwrap();
    /// assert_eq!(circle.frame(), Frame2::WORLD);
    /// ```
    pub fn frame(&self) -> Frame2 {
        self.frame
    }

    /// Evaluates the point on the circle at angular parameter `u`:
    /// `center + radius*cos(u)*x_dir + radius*sin(u)*y_dir`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle2D, Point2};
    /// let circle = Circle2D::new(Point2::ORIGIN, 2.0).unwrap();
    /// assert_eq!(circle.eval_point(0.0), Point2::new(2.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64) -> Point2 {
        analytic::circle2d_d0(&self.frame, self.radius, u)
    }

    /// Evaluates the points on the circle at each parameter in `us`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle2D, Point2};
    /// let circle = Circle2D::new(Point2::ORIGIN, 2.0).unwrap();
    /// let points = circle.eval_points(&[0.0, 1.0]);
    /// assert_eq!(points[0], Point2::new(2.0, 0.0));
    /// ```
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point2> {
        us.iter().map(|&u| self.eval_point(u)).collect()
    }

    /// Evaluates the derivative of the given `order` at parameter `u`.
    ///
    /// Derivatives cycle with period 4 in `order` (see
    /// [`crate::curve_math::analytic::circle2d_dn`]).
    ///
    /// # Panics
    ///
    /// Panics if `order == 0`; use [`Circle2D::eval_point`] to evaluate the
    /// position itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle2D, Point2, Vector2};
    /// let circle = Circle2D::new(Point2::ORIGIN, 2.0).unwrap();
    /// assert_eq!(circle.eval_derivative(0.0, 1), Vector2::new(0.0, 2.0));
    /// ```
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector2 {
        match order {
            0 => panic!("eval_derivative: order must be >= 1 (use eval_point for order 0)"),
            _ => analytic::circle2d_dn(&self.frame, self.radius, u, order),
        }
    }

    /// Recovers the angular parameter of a point on (or near) the circle,
    /// wrapped into `[0, 2*PI)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Circle2D, Point2};
    /// let circle = Circle2D::new(Point2::ORIGIN, 2.0).unwrap();
    /// assert!((circle.parameter_of(Point2::new(0.0, 2.0)) - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    /// ```
    pub fn parameter_of(&self, point: Point2) -> f64 {
        analytic::circle2d_parameter(&self.frame, point)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Circle2D, Circle3D, CircleConstructionError, Frame2, Frame3, Point2, Point3, Vector2,
        Vector3,
    };

    // ---- Circle3D construction ----

    #[test]
    fn test_circle3d_new_ok() {
        let c = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
        assert_eq!(c.center(), Point3::ORIGIN);
        assert_eq!(c.radius(), 2.0);
        assert_eq!(c.normal(), Vector3::Z);
    }

    #[test]
    fn test_circle3d_new_negative_radius_errors() {
        assert_eq!(
            Circle3D::new(Point3::ORIGIN, Vector3::Z, -1.0),
            Err(CircleConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_circle3d_new_null_normal_errors() {
        assert_eq!(
            Circle3D::new(Point3::ORIGIN, Vector3::ZERO, 1.0),
            Err(CircleConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_circle3d_from_axis_ok() {
        let axis = crate::Axis3::new(Point3::new(1.0, 2.0, 3.0), Vector3::Y).unwrap();
        let c = Circle3D::from_axis(axis, 3.0).unwrap();
        assert_eq!(c.center(), Point3::new(1.0, 2.0, 3.0));
        assert_eq!(c.normal(), Vector3::Y);
        assert_eq!(c.radius(), 3.0);
    }

    #[test]
    fn test_circle3d_from_axis_negative_radius_errors() {
        let axis = crate::Axis3::new(Point3::ORIGIN, Vector3::Z).unwrap();
        assert_eq!(
            Circle3D::from_axis(axis, -1.0),
            Err(CircleConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_circle3d_from_frame_ok() {
        let frame = Frame3::WORLD;
        let c = Circle3D::from_frame(frame, 5.0).unwrap();
        assert_eq!(c.frame(), frame);
        assert_eq!(c.radius(), 5.0);
    }

    #[test]
    fn test_circle3d_from_frame_negative_radius_errors() {
        assert_eq!(
            Circle3D::from_frame(Frame3::WORLD, -0.1),
            Err(CircleConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_circle3d_from_three_points_ok() {
        let p1 = Point3::new(1.0, 0.0, 0.0);
        let p2 = Point3::new(0.0, 1.0, 0.0);
        let p3 = Point3::new(-1.0, 0.0, 0.0);
        let c = Circle3D::from_three_points(p1, p2, p3).unwrap();
        assert!((c.radius() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_circle3d_from_three_points_two_coincident_errors() {
        let p = Point3::new(1.0, 2.0, 3.0);
        let p2 = Point3::new(4.0, 5.0, 6.0);
        assert_eq!(
            Circle3D::from_three_points(p, p, p2),
            Err(CircleConstructionError::ConfusedPoints)
        );
    }

    #[test]
    fn test_circle3d_from_three_points_all_coincident_errors() {
        let p = Point3::new(1.0, 2.0, 3.0);
        assert_eq!(
            Circle3D::from_three_points(p, p, p),
            Err(CircleConstructionError::ConfusedPoints)
        );
    }

    #[test]
    fn test_circle3d_from_three_points_collinear_errors() {
        let p1 = Point3::new(0.0, 0.0, 0.0);
        let p2 = Point3::new(1.0, 0.0, 0.0);
        let p3 = Point3::new(2.0, 0.0, 0.0);
        assert_eq!(
            Circle3D::from_three_points(p1, p2, p3),
            Err(CircleConstructionError::CollinearPoints)
        );
    }

    // ---- Circle3D evaluation ----

    #[test]
    fn test_circle3d_eval_point_zero() {
        let c = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
        let p = c.eval_point(0.0);
        assert!((p.x - 2.0).abs() < 1e-9);
        assert!(p.y.abs() < 1e-9);
        assert!(p.z.abs() < 1e-9);
    }

    #[test]
    fn test_circle3d_eval_points_matches_loop() {
        let c = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
        let us = [0.0, 0.5, 1.5];
        let expected: Vec<Point3> = us.iter().map(|&u| c.eval_point(u)).collect();
        assert_eq!(c.eval_points(&us), expected);
    }

    #[test]
    fn test_circle3d_eval_derivative_order1() {
        let c = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
        let d1 = c.eval_derivative(0.0, 1);
        assert!(d1.x.abs() < 1e-9);
        assert!((d1.y - 2.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic]
    fn test_circle3d_eval_derivative_order0_panics() {
        let c = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
        c.eval_derivative(0.0, 0);
    }

    #[test]
    fn test_circle3d_parameter_of_round_trip() {
        let c = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
        for u in [0.3, 2.0, 5.5] {
            let p = c.eval_point(u);
            assert!((c.parameter_of(p) - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_circle3d_parameter_of_in_zero_to_tau() {
        let c = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
        for u in [-1.0, -0.1, 7.0] {
            let p = c.eval_point(u);
            let recovered = c.parameter_of(p);
            assert!((0.0..std::f64::consts::TAU).contains(&recovered));
        }
    }

    // ---- CircleConstructionError ----

    #[test]
    fn test_circle_construction_error_display() {
        assert_eq!(
            CircleConstructionError::NegativeRadius.to_string(),
            "radius is negative"
        );
        assert_eq!(
            CircleConstructionError::NullNormal.to_string(),
            "normal has zero length"
        );
        assert_eq!(
            CircleConstructionError::ConfusedPoints.to_string(),
            "the points are confused"
        );
        assert_eq!(
            CircleConstructionError::CollinearPoints.to_string(),
            "the points are collinear"
        );
    }

    #[test]
    fn test_circle_construction_error_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&CircleConstructionError::NegativeRadius);
    }

    // ---- Circle2D construction ----

    #[test]
    fn test_circle2d_new_ok() {
        let c = Circle2D::new(Point2::ORIGIN, 2.0).unwrap();
        assert_eq!(c.center(), Point2::ORIGIN);
        assert_eq!(c.radius(), 2.0);
        assert_eq!(c.frame(), Frame2::WORLD);
    }

    #[test]
    fn test_circle2d_new_negative_radius_errors() {
        assert_eq!(
            Circle2D::new(Point2::ORIGIN, -1.0),
            Err(CircleConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_circle2d_from_frame_ok() {
        let frame = Frame2::WORLD;
        let c = Circle2D::from_frame(frame, 3.0).unwrap();
        assert_eq!(c.frame(), frame);
        assert_eq!(c.radius(), 3.0);
    }

    #[test]
    fn test_circle2d_from_frame_negative_radius_errors() {
        assert_eq!(
            Circle2D::from_frame(Frame2::WORLD, -1.0),
            Err(CircleConstructionError::NegativeRadius)
        );
    }

    // ---- Circle2D evaluation ----

    #[test]
    fn test_circle2d_eval_point_zero() {
        let c = Circle2D::new(Point2::ORIGIN, 2.0).unwrap();
        let p = c.eval_point(0.0);
        assert!((p.x - 2.0).abs() < 1e-9);
        assert!(p.y.abs() < 1e-9);
    }

    #[test]
    fn test_circle2d_eval_points_matches_loop() {
        let c = Circle2D::new(Point2::ORIGIN, 2.0).unwrap();
        let us = [0.0, 0.5, 1.5];
        let expected: Vec<Point2> = us.iter().map(|&u| c.eval_point(u)).collect();
        assert_eq!(c.eval_points(&us), expected);
    }

    #[test]
    fn test_circle2d_eval_derivative_order1() {
        let c = Circle2D::new(Point2::ORIGIN, 2.0).unwrap();
        let d1 = c.eval_derivative(0.0, 1);
        assert!(d1.x.abs() < 1e-9);
        assert!((d1.y - 2.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic]
    fn test_circle2d_eval_derivative_order0_panics() {
        let c = Circle2D::new(Point2::ORIGIN, 2.0).unwrap();
        c.eval_derivative(0.0, 0);
    }

    #[test]
    fn test_circle2d_parameter_of_round_trip() {
        let c = Circle2D::new(Point2::ORIGIN, 2.0).unwrap();
        for u in [0.3, 2.0, 5.5] {
            let p = c.eval_point(u);
            assert!((c.parameter_of(p) - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_circle2d_from_frame_arbitrary() {
        let frame = Frame2::from_x(Point2::new(1.0, -2.0), Vector2::new(3.0, 4.0)).unwrap();
        let c = Circle2D::from_frame(frame, 2.5).unwrap();
        for u in [0.3, 2.0, 5.5] {
            let p = c.eval_point(u);
            assert!((c.parameter_of(p) - u).abs() < 1e-9);
        }
    }
}
