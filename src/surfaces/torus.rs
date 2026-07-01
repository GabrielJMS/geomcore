//! Tori in 3D: parametric evaluation and parameter inversion, thin wrappers
//! over [`crate::surface_math::analytic`].

use crate::surface_math::analytic;
use crate::{Frame3, Point3, Vector3};
use std::fmt;

/// Error returned when a [`Torus`] cannot be constructed from the given
/// inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorusConstructionError {
    /// Either the major or the minor radius is negative. Unlike some
    /// analytic surfaces, geomrust deliberately rejects a negative major
    /// radius too (rather than accepting it as geometrically meaningless
    /// but harmless): a negative major radius has no consistent
    /// parametric meaning here, so it is treated the same as a negative
    /// minor radius.
    NegativeRadius,
    /// The main axis direction has zero length.
    NullNormal,
}

impl fmt::Display for TorusConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            TorusConstructionError::NegativeRadius => "radius is negative",
            TorusConstructionError::NullNormal => "normal has zero length",
        };
        f.write_str(message)
    }
}

impl std::error::Error for TorusConstructionError {}

/// A torus in 3D: a [`Frame3`] (origin plus local x/y/z directions) plus a
/// major and a minor radius, evaluated as `R = major + minor*cos(v)`,
/// `origin + R*cos(u)*x_dir + R*sin(u)*y_dir + minor*sin(v)*z_dir`.
///
/// # Examples
///
/// ```
/// use geomrust::{Point3, Torus, Vector3};
/// let torus = Torus::new(Point3::ORIGIN, Vector3::Z, 5.0, 1.5).unwrap();
/// assert_eq!(torus.eval_point(0.0, 0.0), Point3::new(6.5, 0.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Torus {
    frame: Frame3,
    major_radius: f64,
    minor_radius: f64,
}

impl Torus {
    /// Creates a torus from a center, a normal, a major radius, and a
    /// minor radius.
    ///
    /// The frame is derived from `normal` via [`Frame3::from_z`].
    ///
    /// # Errors
    ///
    /// Returns [`TorusConstructionError::NullNormal`] if `normal` cannot be
    /// normalized (zero length), or
    /// [`TorusConstructionError::NegativeRadius`] if `major_radius < 0` or
    /// `minor_radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Point3, Torus, Vector3};
    /// let torus = Torus::new(Point3::ORIGIN, Vector3::Z, 5.0, 1.5).unwrap();
    /// assert_eq!(torus.major_radius(), 5.0);
    /// assert_eq!(torus.minor_radius(), 1.5);
    /// ```
    pub fn new(
        center: Point3,
        normal: Vector3,
        major_radius: f64,
        minor_radius: f64,
    ) -> Result<Torus, TorusConstructionError> {
        let frame =
            Frame3::from_z(center, normal).map_err(|_| TorusConstructionError::NullNormal)?;
        Torus::from_frame(frame, major_radius, minor_radius)
    }

    /// Creates a torus from a frame, a major radius, and a minor radius
    /// directly.
    ///
    /// # Errors
    ///
    /// Returns [`TorusConstructionError::NegativeRadius`] if
    /// `major_radius < 0` or `minor_radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame3, Torus};
    /// let torus = Torus::from_frame(Frame3::WORLD, 5.0, 1.5).unwrap();
    /// assert_eq!(torus.frame(), Frame3::WORLD);
    /// ```
    pub fn from_frame(
        frame: Frame3,
        major_radius: f64,
        minor_radius: f64,
    ) -> Result<Torus, TorusConstructionError> {
        if major_radius < 0.0 || minor_radius < 0.0 {
            return Err(TorusConstructionError::NegativeRadius);
        }
        Ok(Torus {
            frame,
            major_radius,
            minor_radius,
        })
    }

    /// Returns the center of the torus.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Point3, Torus, Vector3};
    /// let torus = Torus::new(Point3::new(1.0, 2.0, 3.0), Vector3::Z, 5.0, 1.5).unwrap();
    /// assert_eq!(torus.center(), Point3::new(1.0, 2.0, 3.0));
    /// ```
    pub fn center(&self) -> Point3 {
        self.frame.origin()
    }

    /// Returns the torus's frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame3, Torus};
    /// let torus = Torus::from_frame(Frame3::WORLD, 5.0, 1.5).unwrap();
    /// assert_eq!(torus.frame(), Frame3::WORLD);
    /// ```
    pub fn frame(&self) -> Frame3 {
        self.frame
    }

    /// Returns the torus's major radius (the distance from the center to
    /// the tube's centerline circle).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame3, Torus};
    /// let torus = Torus::from_frame(Frame3::WORLD, 5.0, 1.5).unwrap();
    /// assert_eq!(torus.major_radius(), 5.0);
    /// ```
    pub fn major_radius(&self) -> f64 {
        self.major_radius
    }

    /// Returns the torus's minor radius (the tube's own radius).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame3, Torus};
    /// let torus = Torus::from_frame(Frame3::WORLD, 5.0, 1.5).unwrap();
    /// assert_eq!(torus.minor_radius(), 1.5);
    /// ```
    pub fn minor_radius(&self) -> f64 {
        self.minor_radius
    }

    /// Evaluates the point on the torus at `(u, v)`. See the type-level
    /// docs for the formula.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Point3, Torus, Vector3};
    /// let torus = Torus::new(Point3::ORIGIN, Vector3::Z, 5.0, 1.5).unwrap();
    /// assert_eq!(torus.eval_point(0.0, 0.0), Point3::new(6.5, 0.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64, v: f64) -> Point3 {
        analytic::torus_d0(&self.frame, self.major_radius, self.minor_radius, u, v)
    }

    /// Evaluates the points on the torus at each `(u, v)` in `uvs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Point3, Torus, Vector3};
    /// let torus = Torus::new(Point3::ORIGIN, Vector3::Z, 5.0, 1.5).unwrap();
    /// let points = torus.eval_points(&[(0.0, 0.0)]);
    /// assert_eq!(points[0], Point3::new(6.5, 0.0, 0.0));
    /// ```
    pub fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3> {
        uvs.iter().map(|&(u, v)| self.eval_point(u, v)).collect()
    }

    /// Evaluates the derivative of order `(du, dv)` at `(u, v)`. See
    /// [`crate::surface_math::analytic::torus_derivative`] for the
    /// formulas.
    ///
    /// # Panics
    ///
    /// Panics if `du + dv == 0` (use [`Torus::eval_point`] for the
    /// position itself) or if `du + dv > 2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Point3, Torus, Vector3};
    /// let torus = Torus::new(Point3::ORIGIN, Vector3::Z, 5.0, 1.5).unwrap();
    /// assert_eq!(torus.eval_derivative(0.0, 0.0, 0, 1), Vector3::Z * 1.5);
    /// ```
    pub fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3 {
        match du + dv {
            0 => panic!(
                "eval_derivative: du + dv must be >= 1 (use eval_point for the (0, 0) order)"
            ),
            1..=2 => analytic::torus_derivative(
                &self.frame,
                self.major_radius,
                self.minor_radius,
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

    /// Recovers `(u, v)` of a point on (or near) the torus. See
    /// [`crate::surface_math::analytic::torus_parameters`] for the formula,
    /// including the near/far branch handling when `major_radius <
    /// minor_radius`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Point3, Torus, Vector3};
    /// let torus = Torus::new(Point3::ORIGIN, Vector3::Z, 5.0, 1.5).unwrap();
    /// let (u, v) = torus.parameters_of(Point3::new(6.5, 0.0, 0.0));
    /// assert_eq!((u, v), (0.0, 0.0));
    /// ```
    pub fn parameters_of(&self, point: Point3) -> (f64, f64) {
        analytic::torus_parameters(&self.frame, self.major_radius, self.minor_radius, point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- construction ----

    #[test]
    fn test_new_ok() {
        let t = Torus::new(Point3::ORIGIN, Vector3::Z, 5.0, 1.5).unwrap();
        assert_eq!(t.major_radius(), 5.0);
        assert_eq!(t.minor_radius(), 1.5);
        assert_eq!(t.center(), Point3::ORIGIN);
    }

    #[test]
    fn test_new_null_normal_errors() {
        assert_eq!(
            Torus::new(Point3::ORIGIN, Vector3::ZERO, 5.0, 1.5),
            Err(TorusConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_new_negative_major_radius_errors() {
        assert_eq!(
            Torus::new(Point3::ORIGIN, Vector3::Z, -5.0, 1.5),
            Err(TorusConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_new_negative_minor_radius_errors() {
        assert_eq!(
            Torus::new(Point3::ORIGIN, Vector3::Z, 5.0, -1.5),
            Err(TorusConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_from_frame_ok() {
        let t = Torus::from_frame(Frame3::WORLD, 5.0, 1.5).unwrap();
        assert_eq!(t.frame(), Frame3::WORLD);
    }

    #[test]
    fn test_from_frame_negative_radius_errors() {
        assert_eq!(
            Torus::from_frame(Frame3::WORLD, -5.0, 1.5),
            Err(TorusConstructionError::NegativeRadius)
        );
        assert_eq!(
            Torus::from_frame(Frame3::WORLD, 5.0, -1.5),
            Err(TorusConstructionError::NegativeRadius)
        );
    }

    // ---- evaluation ----

    #[test]
    fn test_eval_point() {
        let t = Torus::new(Point3::ORIGIN, Vector3::Z, 5.0, 1.5).unwrap();
        assert_eq!(t.eval_point(0.0, 0.0), Point3::new(6.5, 0.0, 0.0));
    }

    #[test]
    fn test_eval_points_matches_loop() {
        let t = Torus::new(Point3::ORIGIN, Vector3::Z, 5.0, 1.5).unwrap();
        let uvs = [(0.0, 0.0), (0.5, 1.0)];
        let expected: Vec<Point3> = uvs.iter().map(|&(u, v)| t.eval_point(u, v)).collect();
        assert_eq!(t.eval_points(&uvs), expected);
    }

    #[test]
    #[should_panic(expected = "du + dv must be >= 1")]
    fn test_eval_derivative_zero_order_panics() {
        let t = Torus::new(Point3::ORIGIN, Vector3::Z, 5.0, 1.5).unwrap();
        t.eval_derivative(0.0, 0.0, 0, 0);
    }

    #[test]
    #[should_panic(expected = "du + dv must be <= 2")]
    fn test_eval_derivative_order_too_high_panics() {
        let t = Torus::new(Point3::ORIGIN, Vector3::Z, 5.0, 1.5).unwrap();
        t.eval_derivative(0.0, 0.0, 2, 1);
    }

    #[test]
    fn test_parameters_of_round_trip() {
        let t = Torus::new(Point3::ORIGIN, Vector3::Z, 5.0, 1.5).unwrap();
        let (u, v) = t.parameters_of(Point3::new(6.5, 0.0, 0.0));
        assert_eq!((u, v), (0.0, 0.0));
    }

    // ---- TorusConstructionError ----

    #[test]
    fn test_error_display() {
        assert_eq!(
            TorusConstructionError::NegativeRadius.to_string(),
            "radius is negative"
        );
        assert_eq!(
            TorusConstructionError::NullNormal.to_string(),
            "normal has zero length"
        );
    }

    #[test]
    fn test_error_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&TorusConstructionError::NegativeRadius);
    }
}
