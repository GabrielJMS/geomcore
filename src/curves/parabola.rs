//! Parabolas in 3D: parametric evaluation and parameter inversion, a thin
//! wrapper over [`crate::curve_math::analytic`].

use crate::curve_math::analytic;
use crate::{Frame3, Point3, Vector3};
use std::fmt;

/// Error returned when a [`Parabola3D`] cannot be constructed from the given
/// inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParabolaConstructionError {
    /// The requested focal distance is negative.
    NegativeFocal,
    /// The normal (or the x direction) has zero length, or the x direction
    /// is parallel to the normal.
    NullNormal,
}

impl fmt::Display for ParabolaConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            ParabolaConstructionError::NegativeFocal => "focal distance is negative",
            ParabolaConstructionError::NullNormal => "normal has zero length",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ParabolaConstructionError {}

/// A parabola in 3D: a plane [`Frame3`] (origin at the apex, plus local x/y
/// directions defining the plane and the axis of symmetry) and a focal
/// distance, evaluated as
/// `apex + (u^2 / (4*focal))*x_dir + u*y_dir`.
///
/// # Examples
///
/// ```
/// use geomrust::{Parabola3D, Point3, Vector3};
/// let parabola = Parabola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 1.0).unwrap();
/// assert_eq!(parabola.eval_point(2.0), Point3::new(1.0, 2.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Parabola3D {
    frame: Frame3,
    focal: f64,
}

impl Parabola3D {
    /// Creates a parabola from an apex, a normal, an x-axis hint, and a
    /// focal distance.
    ///
    /// The plane frame is derived from `normal` and `x_direction` via
    /// [`Frame3::new`].
    ///
    /// # Errors
    ///
    /// Returns [`ParabolaConstructionError::NullNormal`] if `normal` cannot
    /// be normalized, if `x_direction` cannot be normalized, or if
    /// `x_direction` is parallel to `normal`; or
    /// [`ParabolaConstructionError::NegativeFocal`] if `focal < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Parabola3D, Point3, Vector3};
    /// let parabola = Parabola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 1.0).unwrap();
    /// assert_eq!(parabola.focal(), 1.0);
    /// ```
    pub fn new(
        apex: Point3,
        normal: Vector3,
        x_direction: Vector3,
        focal: f64,
    ) -> Result<Parabola3D, ParabolaConstructionError> {
        let frame = Frame3::new(apex, normal, x_direction)
            .map_err(|_| ParabolaConstructionError::NullNormal)?;
        Parabola3D::from_frame(frame, focal)
    }

    /// Creates a parabola from a plane frame and a focal distance directly.
    ///
    /// # Errors
    ///
    /// Returns [`ParabolaConstructionError::NegativeFocal`] if `focal < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Parabola3D, Frame3};
    /// let parabola = Parabola3D::from_frame(Frame3::WORLD, 1.7).unwrap();
    /// assert_eq!(parabola.frame(), Frame3::WORLD);
    /// ```
    pub fn from_frame(frame: Frame3, focal: f64) -> Result<Parabola3D, ParabolaConstructionError> {
        if focal < 0.0 {
            return Err(ParabolaConstructionError::NegativeFocal);
        }
        Ok(Parabola3D { frame, focal })
    }

    /// Returns the apex of the parabola.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Parabola3D, Point3, Vector3};
    /// let parabola = Parabola3D::new(Point3::new(1.0, 2.0, 3.0), Vector3::Z, Vector3::X, 1.0).unwrap();
    /// assert_eq!(parabola.apex(), Point3::new(1.0, 2.0, 3.0));
    /// ```
    pub fn apex(&self) -> Point3 {
        self.frame.origin()
    }

    /// Returns the parabola's plane frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Parabola3D, Frame3};
    /// let parabola = Parabola3D::from_frame(Frame3::WORLD, 1.7).unwrap();
    /// assert_eq!(parabola.frame(), Frame3::WORLD);
    /// ```
    pub fn frame(&self) -> Frame3 {
        self.frame
    }

    /// Returns the focal distance of the parabola.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Parabola3D, Frame3};
    /// let parabola = Parabola3D::from_frame(Frame3::WORLD, 1.7).unwrap();
    /// assert_eq!(parabola.focal(), 1.7);
    /// ```
    pub fn focal(&self) -> f64 {
        self.focal
    }

    /// Evaluates the point on the parabola at parameter `u`:
    /// `apex + (u^2 / (4*focal))*x_dir + u*y_dir`.
    ///
    /// A parabola with focal = 0 is degenerate; evaluation returns
    /// non-finite values.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Parabola3D, Point3, Vector3};
    /// let parabola = Parabola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 1.0).unwrap();
    /// assert_eq!(parabola.eval_point(2.0), Point3::new(1.0, 2.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64) -> Point3 {
        analytic::parabola_d0(&self.frame, self.focal, u)
    }

    /// Evaluates the points on the parabola at each parameter in `us`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Parabola3D, Point3, Vector3};
    /// let parabola = Parabola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 1.0).unwrap();
    /// let points = parabola.eval_points(&[0.0, 2.0]);
    /// assert_eq!(points[0], Point3::ORIGIN);
    /// ```
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point3> {
        us.iter().map(|&u| self.eval_point(u)).collect()
    }

    /// Evaluates the derivative of the given `order` at parameter `u`.
    ///
    /// The parabola is a degree-2 polynomial curve: the first derivative is
    /// linear in `u`, the second is a constant, and every derivative of
    /// order above 2 is zero (see
    /// [`crate::curve_math::analytic::parabola_dn`]).
    ///
    /// A parabola with focal = 0 is degenerate; evaluation returns
    /// non-finite values.
    ///
    /// # Panics
    ///
    /// Panics if `order == 0`; use [`Parabola3D::eval_point`] to evaluate
    /// the position itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Parabola3D, Point3, Vector3};
    /// let parabola = Parabola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 1.0).unwrap();
    /// assert_eq!(parabola.eval_derivative(0.0, 1), Vector3::Y);
    /// ```
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector3 {
        match order {
            0 => panic!("eval_derivative: order must be >= 1 (use eval_point for order 0)"),
            _ => analytic::parabola_dn(&self.frame, self.focal, u, order),
        }
    }

    /// Recovers the parameter of a point on (or near) the parabola: the
    /// projection `(point - apex) . y_dir`. Unbounded (no wrapping).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Parabola3D, Point3, Vector3};
    /// let parabola = Parabola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 1.0).unwrap();
    /// assert_eq!(parabola.parameter_of(Point3::new(1.0, 2.0, 0.0)), 2.0);
    /// ```
    pub fn parameter_of(&self, point: Point3) -> f64 {
        analytic::parabola_parameter(&self.frame, point)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Frame3, Parabola3D, ParabolaConstructionError, Point3, Vector3};

    // ---- construction ----

    #[test]
    fn test_parabola3d_new_ok() {
        let p = Parabola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 1.7).unwrap();
        assert_eq!(p.apex(), Point3::ORIGIN);
        assert_eq!(p.focal(), 1.7);
    }

    #[test]
    fn test_parabola3d_new_negative_focal_errors() {
        assert_eq!(
            Parabola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, -1.0),
            Err(ParabolaConstructionError::NegativeFocal)
        );
    }

    #[test]
    fn test_parabola3d_new_zero_focal_allowed() {
        let p = Parabola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 0.0).unwrap();
        assert_eq!(p.focal(), 0.0);
    }

    #[test]
    fn test_parabola3d_new_null_normal_errors() {
        assert_eq!(
            Parabola3D::new(Point3::ORIGIN, Vector3::ZERO, Vector3::X, 1.0),
            Err(ParabolaConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_parabola3d_new_parallel_x_direction_errors() {
        assert_eq!(
            Parabola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::Z, 1.0),
            Err(ParabolaConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_parabola3d_from_frame_ok() {
        let p = Parabola3D::from_frame(Frame3::WORLD, 1.7).unwrap();
        assert_eq!(p.frame(), Frame3::WORLD);
        assert_eq!(p.focal(), 1.7);
    }

    #[test]
    fn test_parabola3d_from_frame_negative_focal_errors() {
        assert_eq!(
            Parabola3D::from_frame(Frame3::WORLD, -0.1),
            Err(ParabolaConstructionError::NegativeFocal)
        );
    }

    // ---- evaluation ----

    #[test]
    fn test_parabola3d_eval_point_zero() {
        let p = Parabola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 1.0).unwrap();
        let pt = p.eval_point(0.0);
        assert!(pt.x.abs() < 1e-9);
        assert!(pt.y.abs() < 1e-9);
        assert!(pt.z.abs() < 1e-9);
    }

    #[test]
    fn test_parabola3d_eval_points_matches_loop() {
        let p = Parabola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 1.7).unwrap();
        let us = [0.0, 0.5, 1.5];
        let expected: Vec<Point3> = us.iter().map(|&u| p.eval_point(u)).collect();
        assert_eq!(p.eval_points(&us), expected);
    }

    #[test]
    fn test_parabola3d_eval_derivative_order1() {
        let p = Parabola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 1.0).unwrap();
        let d1 = p.eval_derivative(0.0, 1);
        assert!(d1.x.abs() < 1e-9);
        assert!((d1.y - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_parabola3d_eval_derivative_order3_is_zero() {
        let p = Parabola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 1.0).unwrap();
        let d3 = p.eval_derivative(0.5, 3);
        assert_eq!(d3, Vector3::ZERO);
    }

    #[test]
    #[should_panic]
    fn test_parabola3d_eval_derivative_order0_panics() {
        let p = Parabola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 1.0).unwrap();
        p.eval_derivative(0.0, 0);
    }

    #[test]
    fn test_parabola3d_parameter_of_round_trip() {
        let p = Parabola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 1.7).unwrap();
        for u in [0.3, 2.0, -5.5] {
            let pt = p.eval_point(u);
            assert!((p.parameter_of(pt) - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_parabola3d_parameter_of_is_unbounded() {
        let p = Parabola3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 1.7).unwrap();
        let pt = p.eval_point(100.0);
        assert!((p.parameter_of(pt) - 100.0).abs() < 1e-6);
    }

    // ---- ParabolaConstructionError ----

    #[test]
    fn test_parabola_construction_error_display() {
        assert_eq!(
            ParabolaConstructionError::NegativeFocal.to_string(),
            "focal distance is negative"
        );
        assert_eq!(
            ParabolaConstructionError::NullNormal.to_string(),
            "normal has zero length"
        );
    }

    #[test]
    fn test_parabola_construction_error_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&ParabolaConstructionError::NegativeFocal);
    }
}
