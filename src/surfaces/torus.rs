//! Tori in 3D: parametric evaluation and parameter inversion, thin wrappers
//! over [`crate::surface_math::analytic`].

use crate::surface_math::analytic;
use crate::{Frame3D, Point3D, Tolerance, Vector3D};
use std::fmt;

/// Error returned when a [`Torus`] cannot be constructed from the given
/// inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorusConstructionError {
    /// Either the major or the minor radius is negative. Unlike some
    /// analytic surfaces, geomcore deliberately rejects a negative major
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

/// A torus in 3D: a [`Frame3D`] (origin plus local x/y/z directions) plus a
/// major and a minor radius, evaluated as `R = major + minor*cos(v)`,
/// `origin + R*cos(u)*x_dir + R*sin(u)*y_dir + minor*sin(v)*z_dir`.
///
/// # Examples
///
/// ```
/// use geomcore::{Point3D, Torus, Vector3D};
/// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
/// assert_eq!(torus.eval_point(0.0, 0.0), Point3D::new(6.5, 0.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Torus {
    frame: Frame3D,
    major_radius: f64,
    minor_radius: f64,
}

impl Torus {
    /// Creates a torus from a center, a normal, a major radius, and a
    /// minor radius.
    ///
    /// The frame is derived from `normal` via [`Frame3D::from_z`].
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
    /// use geomcore::{Point3D, Torus, Vector3D};
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
    /// assert_eq!(torus.major_radius(), 5.0);
    /// assert_eq!(torus.minor_radius(), 1.5);
    /// ```
    pub fn new(
        center: Point3D,
        normal: Vector3D,
        major_radius: f64,
        minor_radius: f64,
    ) -> Result<Torus, TorusConstructionError> {
        let frame =
            Frame3D::from_z(center, normal).map_err(|_| TorusConstructionError::NullNormal)?;
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
    /// use geomcore::{Frame3D, Torus};
    /// let torus = Torus::from_frame(Frame3D::WORLD, 5.0, 1.5).unwrap();
    /// assert_eq!(torus.frame(), Frame3D::WORLD);
    /// ```
    pub fn from_frame(
        frame: Frame3D,
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
    /// use geomcore::{Point3D, Torus, Vector3D};
    /// let torus = Torus::new(Point3D::new(1.0, 2.0, 3.0), Vector3D::Z, 5.0, 1.5).unwrap();
    /// assert_eq!(torus.center(), Point3D::new(1.0, 2.0, 3.0));
    /// ```
    pub fn center(&self) -> Point3D {
        self.frame.origin()
    }

    /// Returns the torus's frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Frame3D, Torus};
    /// let torus = Torus::from_frame(Frame3D::WORLD, 5.0, 1.5).unwrap();
    /// assert_eq!(torus.frame(), Frame3D::WORLD);
    /// ```
    pub fn frame(&self) -> Frame3D {
        self.frame
    }

    /// Returns the torus's major radius (the distance from the center to
    /// the tube's centerline circle).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Frame3D, Torus};
    /// let torus = Torus::from_frame(Frame3D::WORLD, 5.0, 1.5).unwrap();
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
    /// use geomcore::{Frame3D, Torus};
    /// let torus = Torus::from_frame(Frame3D::WORLD, 5.0, 1.5).unwrap();
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
    /// use geomcore::{Point3D, Torus, Vector3D};
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
    /// assert_eq!(torus.eval_point(0.0, 0.0), Point3D::new(6.5, 0.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64, v: f64) -> Point3D {
        analytic::torus_d0(&self.frame, self.major_radius, self.minor_radius, u, v)
    }

    /// Evaluates the points on the torus at each `(u, v)` in `uvs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Torus, Vector3D};
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
    /// let points = torus.eval_points(&[(0.0, 0.0)]);
    /// assert_eq!(points[0], Point3D::new(6.5, 0.0, 0.0));
    /// ```
    pub fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3D> {
        uvs.iter().map(|&(u, v)| self.eval_point(u, v)).collect()
    }

    /// Evaluates the derivative of order `(du, dv)` at `(u, v)`. See
    /// `surface_math::analytic::torus_derivative` for the
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
    /// use geomcore::{Point3D, Torus, Vector3D};
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
    /// assert_eq!(torus.eval_derivative(0.0, 0.0, 0, 1), Vector3D::Z * 1.5);
    /// ```
    pub fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3D {
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
    /// `surface_math::analytic::torus_parameters` for the formula,
    /// including the near/far branch handling when `major_radius <
    /// minor_radius`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Torus, Vector3D};
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
    /// let (u, v) = torus.parameters_of(Point3D::new(6.5, 0.0, 0.0));
    /// assert_eq!((u, v), (0.0, 0.0));
    /// ```
    pub fn parameters_of(&self, point: Point3D) -> (f64, f64) {
        analytic::torus_parameters(&self.frame, self.major_radius, self.minor_radius, point)
    }

    /// Returns whether `point` lies on the torus: the inverse parameters
    /// are recovered with [`Torus::parameters_of`] and re-evaluated, and
    /// the point counts as contained when the re-evaluated point is within
    /// `tol.confusion` of it.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Tolerance, Torus, Vector3D};
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// assert!(torus.contains(Point3D::new(5.0, 0.0, 0.0), tol));
    /// assert!(!torus.contains(Point3D::ORIGIN, tol));
    /// ```
    pub fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        let (u, v) = self.parameters_of(point);
        self.eval_point(u, v).distance(point) <= tol.confusion
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- construction ----

    #[test]
    fn test_new_ok() {
        let t = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
        assert_eq!(t.major_radius(), 5.0);
        assert_eq!(t.minor_radius(), 1.5);
        assert_eq!(t.center(), Point3D::ORIGIN);
    }

    #[test]
    fn test_new_null_normal_errors() {
        assert_eq!(
            Torus::new(Point3D::ORIGIN, Vector3D::ZERO, 5.0, 1.5),
            Err(TorusConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_new_negative_major_radius_errors() {
        assert_eq!(
            Torus::new(Point3D::ORIGIN, Vector3D::Z, -5.0, 1.5),
            Err(TorusConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_new_negative_minor_radius_errors() {
        assert_eq!(
            Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, -1.5),
            Err(TorusConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_from_frame_ok() {
        let t = Torus::from_frame(Frame3D::WORLD, 5.0, 1.5).unwrap();
        assert_eq!(t.frame(), Frame3D::WORLD);
    }

    #[test]
    fn test_from_frame_negative_radius_errors() {
        assert_eq!(
            Torus::from_frame(Frame3D::WORLD, -5.0, 1.5),
            Err(TorusConstructionError::NegativeRadius)
        );
        assert_eq!(
            Torus::from_frame(Frame3D::WORLD, 5.0, -1.5),
            Err(TorusConstructionError::NegativeRadius)
        );
    }

    // ---- evaluation ----

    #[test]
    fn test_eval_point() {
        let t = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
        assert_eq!(t.eval_point(0.0, 0.0), Point3D::new(6.5, 0.0, 0.0));
    }

    #[test]
    fn test_eval_points_matches_loop() {
        let t = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
        let uvs = [(0.0, 0.0), (0.5, 1.0)];
        let expected: Vec<Point3D> = uvs.iter().map(|&(u, v)| t.eval_point(u, v)).collect();
        assert_eq!(t.eval_points(&uvs), expected);
    }

    #[test]
    #[should_panic(expected = "du + dv must be >= 1")]
    fn test_eval_derivative_zero_order_panics() {
        let t = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
        t.eval_derivative(0.0, 0.0, 0, 0);
    }

    #[test]
    #[should_panic(expected = "du + dv must be <= 2")]
    fn test_eval_derivative_order_too_high_panics() {
        let t = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
        t.eval_derivative(0.0, 0.0, 2, 1);
    }

    #[test]
    fn test_parameters_of_round_trip() {
        let t = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
        let (u, v) = t.parameters_of(Point3D::new(6.5, 0.0, 0.0));
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

    #[test]
    fn test_torus_contains() {
        let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
        let tol = Tolerance::DEFAULT;
        assert!(torus.contains(Point3D::new(5.0, 0.0, 0.0), tol));
        assert!(torus.contains(torus.eval_point(1.0, 2.0), tol));
        assert!(!torus.contains(Point3D::ORIGIN, tol));
        assert!(!torus.contains(Point3D::new(4.0, 0.0, 0.0), tol));
    }
}
