//! Spheres in 3D: parametric evaluation and parameter inversion, thin
//! wrappers over [`crate::surface_math::analytic`].

use crate::surface_math::analytic;
use crate::{Frame3, Point3, Vector3};
use std::fmt;

/// Error returned when a [`Sphere`] cannot be constructed from the given
/// inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SphereConstructionError {
    /// The requested radius is negative.
    NegativeRadius,
    /// The main axis direction has zero length. Unreachable through
    /// [`Sphere::new`] or [`Sphere::from_frame`] (both build the frame from
    /// already-unit directions); kept for parity with sibling surface
    /// error enums and future direction-taking constructors.
    NullNormal,
}

impl fmt::Display for SphereConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            SphereConstructionError::NegativeRadius => "radius is negative",
            SphereConstructionError::NullNormal => "axis direction has zero length",
        };
        f.write_str(message)
    }
}

impl std::error::Error for SphereConstructionError {}

/// A sphere in 3D: a [`Frame3`] (origin plus local x/y/z directions) and a
/// radius, evaluated as `Rcv = r*cos(v)`,
/// `origin + Rcv*cos(u)*x_dir + Rcv*sin(u)*y_dir + r*sin(v)*z_dir`.
///
/// # Examples
///
/// ```
/// use geomcore::{Sphere, Point3};
/// let sphere = Sphere::new(Point3::ORIGIN, 3.0).unwrap();
/// assert_eq!(sphere.eval_point(0.0, 0.0), Point3::new(3.0, 0.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere {
    frame: Frame3,
    radius: f64,
}

impl Sphere {
    /// Creates a sphere from a center and a radius, using a world-aligned
    /// frame (x/y/z directions matching [`Vector3::X`]/[`Vector3::Y`]/
    /// [`Vector3::Z`]) at `center`.
    ///
    /// # Errors
    ///
    /// Returns [`SphereConstructionError::NegativeRadius`] if
    /// `radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3, Sphere};
    /// let sphere = Sphere::new(Point3::new(1.0, 2.0, 3.0), 2.0).unwrap();
    /// assert_eq!(sphere.center(), Point3::new(1.0, 2.0, 3.0));
    /// ```
    pub fn new(center: Point3, radius: f64) -> Result<Sphere, SphereConstructionError> {
        let frame = Frame3::new(center, Vector3::Z, Vector3::X)
            .expect("Vector3::Z and Vector3::X are orthonormal by construction");
        Sphere::from_frame(frame, radius)
    }

    /// Creates a sphere from a frame and a radius directly.
    ///
    /// # Errors
    ///
    /// Returns [`SphereConstructionError::NegativeRadius`] if
    /// `radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Frame3, Sphere};
    /// let sphere = Sphere::from_frame(Frame3::WORLD, 2.0).unwrap();
    /// assert_eq!(sphere.frame(), Frame3::WORLD);
    /// ```
    pub fn from_frame(frame: Frame3, radius: f64) -> Result<Sphere, SphereConstructionError> {
        if radius < 0.0 {
            return Err(SphereConstructionError::NegativeRadius);
        }
        Ok(Sphere { frame, radius })
    }

    /// Returns the center of the sphere.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3, Sphere};
    /// let sphere = Sphere::new(Point3::new(1.0, 2.0, 3.0), 2.0).unwrap();
    /// assert_eq!(sphere.center(), Point3::new(1.0, 2.0, 3.0));
    /// ```
    pub fn center(&self) -> Point3 {
        self.frame.origin()
    }

    /// Returns the sphere's frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Frame3, Sphere};
    /// let sphere = Sphere::from_frame(Frame3::WORLD, 2.0).unwrap();
    /// assert_eq!(sphere.frame(), Frame3::WORLD);
    /// ```
    pub fn frame(&self) -> Frame3 {
        self.frame
    }

    /// Returns the sphere's radius.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3, Sphere};
    /// let sphere = Sphere::new(Point3::ORIGIN, 2.0).unwrap();
    /// assert_eq!(sphere.radius(), 2.0);
    /// ```
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Evaluates the point on the sphere at `(u, v)`. See the type-level
    /// docs for the formula.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3, Sphere};
    /// let sphere = Sphere::new(Point3::ORIGIN, 3.0).unwrap();
    /// assert_eq!(sphere.eval_point(0.0, 0.0), Point3::new(3.0, 0.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64, v: f64) -> Point3 {
        analytic::sphere_d0(&self.frame, self.radius, u, v)
    }

    /// Evaluates the points on the sphere at each `(u, v)` in `uvs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3, Sphere};
    /// let sphere = Sphere::new(Point3::ORIGIN, 3.0).unwrap();
    /// let points = sphere.eval_points(&[(0.0, 0.0)]);
    /// assert_eq!(points[0], Point3::new(3.0, 0.0, 0.0));
    /// ```
    pub fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3> {
        uvs.iter().map(|&(u, v)| self.eval_point(u, v)).collect()
    }

    /// Evaluates the derivative of order `(du, dv)` at `(u, v)`. See
    /// `surface_math::analytic::sphere_derivative` for the
    /// formulas.
    ///
    /// # Panics
    ///
    /// Panics if `du + dv == 0` (use [`Sphere::eval_point`] for the
    /// position itself) or if `du + dv > 2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3, Sphere, Vector3};
    /// let sphere = Sphere::new(Point3::ORIGIN, 3.0).unwrap();
    /// assert_eq!(sphere.eval_derivative(0.0, 0.0, 0, 1), Vector3::Z * 3.0);
    /// ```
    pub fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3 {
        match du + dv {
            0 => panic!(
                "eval_derivative: du + dv must be >= 1 (use eval_point for the (0, 0) order)"
            ),
            1..=2 => analytic::sphere_derivative(&self.frame, self.radius, u, v, du, dv),
            _ => panic!(
                "eval_derivative: order du={du}, dv={dv} is not supported (du + dv must be <= 2)"
            ),
        }
    }

    /// Recovers `(u, v)` of a point on (or near) the sphere. See
    /// `surface_math::analytic::sphere_parameters` for the
    /// formula, including the pole handling.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3, Sphere};
    /// let sphere = Sphere::new(Point3::ORIGIN, 3.0).unwrap();
    /// let (u, v) = sphere.parameters_of(Point3::new(3.0, 0.0, 0.0));
    /// assert_eq!((u, v), (0.0, 0.0));
    /// ```
    pub fn parameters_of(&self, point: Point3) -> (f64, f64) {
        analytic::sphere_parameters(&self.frame, self.radius, point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    // ---- construction ----

    #[test]
    fn test_new_ok_world_aligned() {
        let s = Sphere::new(Point3::new(1.0, 2.0, 3.0), 2.0).unwrap();
        assert_eq!(s.center(), Point3::new(1.0, 2.0, 3.0));
        assert_eq!(s.frame().x_direction(), Vector3::X);
        assert_eq!(s.frame().y_direction(), Vector3::Y);
        assert_eq!(s.frame().z_direction(), Vector3::Z);
    }

    #[test]
    fn test_new_negative_radius_errors() {
        assert_eq!(
            Sphere::new(Point3::ORIGIN, -1.0),
            Err(SphereConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_from_frame_ok() {
        let s = Sphere::from_frame(Frame3::WORLD, 5.0).unwrap();
        assert_eq!(s.frame(), Frame3::WORLD);
        assert_eq!(s.radius(), 5.0);
    }

    #[test]
    fn test_from_frame_negative_radius_errors() {
        assert_eq!(
            Sphere::from_frame(Frame3::WORLD, -0.1),
            Err(SphereConstructionError::NegativeRadius)
        );
    }

    // ---- evaluation ----

    #[test]
    fn test_eval_point() {
        let s = Sphere::new(Point3::ORIGIN, 3.0).unwrap();
        assert_eq!(s.eval_point(0.0, 0.0), Point3::new(3.0, 0.0, 0.0));
    }

    #[test]
    fn test_eval_point_pole() {
        let s = Sphere::new(Point3::ORIGIN, 3.0).unwrap();
        let p = s.eval_point(0.0, FRAC_PI_2);
        assert!(p.x.abs() < 1e-9);
        assert!(p.y.abs() < 1e-9);
        assert!((p.z - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_eval_points_matches_loop() {
        let s = Sphere::new(Point3::ORIGIN, 3.0).unwrap();
        let uvs = [(0.0, 0.0), (0.5, 1.0)];
        let expected: Vec<Point3> = uvs.iter().map(|&(u, v)| s.eval_point(u, v)).collect();
        assert_eq!(s.eval_points(&uvs), expected);
    }

    #[test]
    #[should_panic(expected = "du + dv must be >= 1")]
    fn test_eval_derivative_zero_order_panics() {
        let s = Sphere::new(Point3::ORIGIN, 3.0).unwrap();
        s.eval_derivative(0.0, 0.0, 0, 0);
    }

    #[test]
    #[should_panic(expected = "du + dv must be <= 2")]
    fn test_eval_derivative_order_too_high_panics() {
        let s = Sphere::new(Point3::ORIGIN, 3.0).unwrap();
        s.eval_derivative(0.0, 0.0, 2, 1);
    }

    #[test]
    fn test_parameters_of_round_trip() {
        let s = Sphere::new(Point3::ORIGIN, 3.0).unwrap();
        let (u, v) = s.parameters_of(Point3::new(3.0, 0.0, 0.0));
        assert_eq!((u, v), (0.0, 0.0));
    }

    // ---- SphereConstructionError ----

    #[test]
    fn test_error_display() {
        assert_eq!(
            SphereConstructionError::NegativeRadius.to_string(),
            "radius is negative"
        );
        assert_eq!(
            SphereConstructionError::NullNormal.to_string(),
            "axis direction has zero length"
        );
    }

    #[test]
    fn test_error_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&SphereConstructionError::NegativeRadius);
    }
}
