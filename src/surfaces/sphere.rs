//! Spheres in 3D: parametric evaluation and parameter inversion, thin
//! wrappers over [`crate::surface_math::analytic`].

use crate::projection::{self, SurfaceProjection};
use crate::surface_math::analytic;
use crate::{Frame3D, Point3D, Tolerance, Vector3D};
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

/// A sphere in 3D: a [`Frame3D`] (origin plus local x/y/z directions) and a
/// radius, evaluated as `Rcv = r*cos(v)`,
/// `origin + Rcv*cos(u)*x_dir + Rcv*sin(u)*y_dir + r*sin(v)*z_dir`.
///
/// # Examples
///
/// ```
/// use geomcore::{Sphere, Point3D};
/// let sphere = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
/// assert_eq!(sphere.eval_point(0.0, 0.0), Point3D::new(3.0, 0.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere {
    frame: Frame3D,
    radius: f64,
}

impl Sphere {
    /// Creates a sphere from a center and a radius, using a world-aligned
    /// frame (x/y/z directions matching [`Vector3D::X`]/[`Vector3D::Y`]/
    /// [`Vector3D::Z`]) at `center`.
    ///
    /// # Errors
    ///
    /// Returns [`SphereConstructionError::NegativeRadius`] if
    /// `radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Sphere};
    /// let sphere = Sphere::new(Point3D::new(1.0, 2.0, 3.0), 2.0).unwrap();
    /// assert_eq!(sphere.center(), Point3D::new(1.0, 2.0, 3.0));
    /// ```
    pub fn new(center: Point3D, radius: f64) -> Result<Sphere, SphereConstructionError> {
        let frame = Frame3D::new(center, Vector3D::Z, Vector3D::X)
            .expect("Vector3D::Z and Vector3D::X are orthonormal by construction");
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
    /// use geomcore::{Frame3D, Sphere};
    /// let sphere = Sphere::from_frame(Frame3D::WORLD, 2.0).unwrap();
    /// assert_eq!(sphere.frame(), Frame3D::WORLD);
    /// ```
    pub fn from_frame(frame: Frame3D, radius: f64) -> Result<Sphere, SphereConstructionError> {
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
    /// use geomcore::{Point3D, Sphere};
    /// let sphere = Sphere::new(Point3D::new(1.0, 2.0, 3.0), 2.0).unwrap();
    /// assert_eq!(sphere.center(), Point3D::new(1.0, 2.0, 3.0));
    /// ```
    pub fn center(&self) -> Point3D {
        self.frame.origin()
    }

    /// Returns the sphere's frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Frame3D, Sphere};
    /// let sphere = Sphere::from_frame(Frame3D::WORLD, 2.0).unwrap();
    /// assert_eq!(sphere.frame(), Frame3D::WORLD);
    /// ```
    pub fn frame(&self) -> Frame3D {
        self.frame
    }

    /// Returns the sphere's radius.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Sphere};
    /// let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
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
    /// use geomcore::{Point3D, Sphere};
    /// let sphere = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
    /// assert_eq!(sphere.eval_point(0.0, 0.0), Point3D::new(3.0, 0.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64, v: f64) -> Point3D {
        analytic::sphere_d0(&self.frame, self.radius, u, v)
    }

    /// Evaluates the points on the sphere at each `(u, v)` in `uvs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Sphere};
    /// let sphere = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
    /// let points = sphere.eval_points(&[(0.0, 0.0)]);
    /// assert_eq!(points[0], Point3D::new(3.0, 0.0, 0.0));
    /// ```
    pub fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3D> {
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
    /// use geomcore::{Point3D, Sphere, Vector3D};
    /// let sphere = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
    /// assert_eq!(sphere.eval_derivative(0.0, 0.0, 0, 1), Vector3D::Z * 3.0);
    /// ```
    pub fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3D {
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
    /// use geomcore::{Point3D, Sphere};
    /// let sphere = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
    /// let (u, v) = sphere.parameters_of(Point3D::new(3.0, 0.0, 0.0));
    /// assert_eq!((u, v), (0.0, 0.0));
    /// ```
    pub fn parameters_of(&self, point: Point3D) -> (f64, f64) {
        analytic::sphere_parameters(&self.frame, self.radius, point)
    }

    /// Returns whether `point` lies on the sphere: the inverse parameters
    /// are recovered with [`Sphere::parameters_of`] and re-evaluated, and
    /// the point counts as contained when the re-evaluated point is within
    /// `tol.confusion` of it.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Sphere, Tolerance};
    /// let sphere = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// assert!(sphere.contains(Point3D::new(3.0, 0.0, 0.0), tol));
    /// assert!(!sphere.contains(Point3D::ORIGIN, tol));
    /// ```
    pub fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        let (u, v) = self.parameters_of(point);
        self.eval_point(u, v).distance(point) <= tol.confusion
    }

    /// Projects `point` onto the sphere, returning the `(u, v)` parameters
    /// of the closest point and its distance. Distances within
    /// `tol.confusion` snap to `0.0`, matching [`Sphere::contains`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Sphere, Tolerance};
    /// let sphere = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
    /// let proj = sphere.project_point(Point3D::new(4.0, 0.0, 0.0), Tolerance::DEFAULT);
    /// assert_eq!((proj.u, proj.v), (0.0, 0.0));
    /// assert_eq!(proj.distance, 1.0);
    /// ```
    pub fn project_point(&self, point: Point3D, tol: Tolerance) -> SurfaceProjection {
        let (u, v) = self.parameters_of(point);
        let distance = self.eval_point(u, v).distance(point);
        SurfaceProjection {
            u,
            v,
            distance: projection::snap_distance(distance, tol.confusion),
        }
    }

    /// Projects each point in `points` onto the sphere.
    ///
    /// Default-style batch wrapper over [`Sphere::project_point`]: one
    /// native call per batch, mirroring [`Sphere::eval_points`].
    pub fn project_points(&self, points: &[Point3D], tol: Tolerance) -> Vec<SurfaceProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    // ---- construction ----

    #[test]
    fn test_new_ok_world_aligned() {
        let s = Sphere::new(Point3D::new(1.0, 2.0, 3.0), 2.0).unwrap();
        assert_eq!(s.center(), Point3D::new(1.0, 2.0, 3.0));
        assert_eq!(s.frame().x_direction(), Vector3D::X);
        assert_eq!(s.frame().y_direction(), Vector3D::Y);
        assert_eq!(s.frame().z_direction(), Vector3D::Z);
    }

    #[test]
    fn test_new_negative_radius_errors() {
        assert_eq!(
            Sphere::new(Point3D::ORIGIN, -1.0),
            Err(SphereConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_from_frame_ok() {
        let s = Sphere::from_frame(Frame3D::WORLD, 5.0).unwrap();
        assert_eq!(s.frame(), Frame3D::WORLD);
        assert_eq!(s.radius(), 5.0);
    }

    #[test]
    fn test_from_frame_negative_radius_errors() {
        assert_eq!(
            Sphere::from_frame(Frame3D::WORLD, -0.1),
            Err(SphereConstructionError::NegativeRadius)
        );
    }

    // ---- evaluation ----

    #[test]
    fn test_eval_point() {
        let s = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
        assert_eq!(s.eval_point(0.0, 0.0), Point3D::new(3.0, 0.0, 0.0));
    }

    #[test]
    fn test_eval_point_pole() {
        let s = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
        let p = s.eval_point(0.0, FRAC_PI_2);
        assert!(p.x.abs() < 1e-9);
        assert!(p.y.abs() < 1e-9);
        assert!((p.z - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_eval_points_matches_loop() {
        let s = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
        let uvs = [(0.0, 0.0), (0.5, 1.0)];
        let expected: Vec<Point3D> = uvs.iter().map(|&(u, v)| s.eval_point(u, v)).collect();
        assert_eq!(s.eval_points(&uvs), expected);
    }

    #[test]
    #[should_panic(expected = "du + dv must be >= 1")]
    fn test_eval_derivative_zero_order_panics() {
        let s = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
        s.eval_derivative(0.0, 0.0, 0, 0);
    }

    #[test]
    #[should_panic(expected = "du + dv must be <= 2")]
    fn test_eval_derivative_order_too_high_panics() {
        let s = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
        s.eval_derivative(0.0, 0.0, 2, 1);
    }

    #[test]
    fn test_parameters_of_round_trip() {
        let s = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
        let (u, v) = s.parameters_of(Point3D::new(3.0, 0.0, 0.0));
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

    #[test]
    fn test_sphere_contains() {
        let sphere = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
        let tol = Tolerance::DEFAULT;
        assert!(sphere.contains(Point3D::new(3.0, 0.0, 0.0), tol));
        assert!(sphere.contains(sphere.eval_point(0.7, 0.4), tol));
        assert!(!sphere.contains(Point3D::ORIGIN, tol));
        assert!(!sphere.contains(Point3D::new(4.0, 0.0, 0.0), tol));
    }

    #[test]
    fn test_sphere_project_point() {
        let sphere = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
        let tol = Tolerance::DEFAULT;
        let proj = sphere.project_point(Point3D::new(4.0, 0.0, 0.0), tol);
        assert_eq!((proj.u, proj.v), (0.0, 0.0));
        assert_eq!(proj.distance, 1.0);
        // Center: every surface point is equidistant; distance is exact.
        let center = sphere.project_point(Point3D::ORIGIN, tol);
        assert_eq!(center.distance, 3.0);
    }
}
