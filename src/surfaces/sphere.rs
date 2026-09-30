//! Spheres in 3D: parametric evaluation and parameter inversion, thin
//! wrappers over [`crate::surface_math::analytic`].

use crate::projection::{self, SurfaceProjection};
use crate::surface_math::analytic;
use crate::{
    Circle3D, Cone, Cylinder, Frame3D, Point3D, SphereConeIntersection, SphereCylinderIntersection,
    SphereSphereIntersection, Tolerance, Vector3D,
};
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

    /// Surface area (`4*PI*radius^2`).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Sphere};
    /// use std::f64::consts::PI;
    /// let sphere = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
    /// assert!((sphere.area() - PI * 36.0).abs() < 1e-9);
    /// ```
    pub fn area(&self) -> f64 {
        4.0 * std::f64::consts::PI * self.radius * self.radius
    }

    /// Enclosed volume (`4/3*PI*radius^3`).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Sphere};
    /// use std::f64::consts::PI;
    /// let sphere = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
    /// assert!((sphere.volume() - 4.0 / 3.0 * PI * 27.0).abs() < 1e-9);
    /// ```
    pub fn volume(&self) -> f64 {
        4.0 / 3.0 * std::f64::consts::PI * self.radius.powi(3)
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

    /// Intersects this sphere with a cylinder.
    ///
    /// The analytic path needs the cylinder axis through the sphere
    /// center: with the axis at distance `dist` (against
    /// `tol.confusion`), a tube radius beyond `radius + tol.confusion`
    /// misses, a tube at the radius grazes one equatorial ring, and a
    /// thinner tube cuts two symmetric latitude rings at axial offsets
    /// `+-sqrt(radius^2 - tube^2)`. Anything else is a space quartic and
    /// reports [`SphereCylinderIntersection::NotAnalytic`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cylinder, Point3D, Sphere, SphereCylinderIntersection, Tolerance, Vector3D};
    /// let sphere = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match sphere.intersect_cylinder(&cylinder, tol) {
    ///     SphereCylinderIntersection::TwoCircles(c1, c2) => {
    ///         assert_eq!(c1.radius(), 2.0);
    ///     }
    ///     _ => panic!("expected two circles"),
    /// }
    /// ```
    pub fn intersect_cylinder(
        &self,
        cylinder: &Cylinder,
        tol: Tolerance,
    ) -> SphereCylinderIntersection {
        let axis = cylinder.axis();
        let a = axis.direction();
        let w = self.center() - axis.origin();
        let axial = w.dot(a);
        let dist = (w - a * axial).magnitude();
        if dist > tol.confusion {
            return SphereCylinderIntersection::NotAnalytic;
        }
        let r = self.radius();
        let tube = cylinder.radius();
        if tube > r + tol.confusion {
            SphereCylinderIntersection::Empty
        } else if tube >= r - tol.confusion {
            let circle = Circle3D::new(self.center(), a, tube)
                .expect("tube radius is non-negative by construction");
            SphereCylinderIntersection::Circle(circle)
        } else {
            let h = (r * r - tube * tube).sqrt();
            let mk = |s: f64| {
                Circle3D::new(self.center() + a * (s * h), a, tube)
                    .expect("tube radius is non-negative by construction")
            };
            SphereCylinderIntersection::TwoCircles(mk(1.0), mk(-1.0))
        }
    }

    /// Intersects this sphere with a cone.
    ///
    /// The analytic path needs the sphere center on the cone axis: axial
    /// offsets `alpha` solve `alpha^2/cos^2(phi) - 2*tc*alpha + tc^2 -
    /// R^2 = 0`, i.e. latitude rings where the cone radius meets the
    /// sphere section. Roots behind the apex are dropped, so the result
    /// is two rings, one ring, a grazing ring, or empty. Anything else is
    /// a space quartic and reports
    /// [`SphereConeIntersection::NotAnalytic`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cone, Frame3D, Point3D, Sphere, SphereConeIntersection, Tolerance};
    /// let sphere = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
    /// let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match sphere.intersect_cone(&cone, tol) {
    ///     SphereConeIntersection::TwoCircles(_, _) => {}
    ///     _ => panic!("expected two circles"),
    /// }
    /// ```
    pub fn intersect_cone(&self, cone: &Cone, tol: Tolerance) -> SphereConeIntersection {
        let a = cone.frame().z_direction();
        let w = self.center() - cone.apex();
        let tc = w.dot(a);
        let dist = (w - a * tc).magnitude();
        if dist > tol.confusion {
            return SphereConeIntersection::NotAnalytic;
        }
        let r = self.radius();
        let cos_phi = cone.semi_angle().cos();
        // disc = R^2 - tc^2*sin^2(phi), in length-squared units.
        let disc = r * r - tc * tc * (1.0 - cos_phi * cos_phi);
        let band = tol.confusion * (r * r + tc * tc).max(1.0);
        if disc < -band {
            return SphereConeIntersection::Empty;
        }
        let mk = |alpha: f64| {
            Circle3D::new(cone.apex() + a * alpha, a, alpha * cone.semi_angle().tan())
                .expect("latitude radius is non-negative by construction")
        };
        if disc <= band {
            let alpha = tc * cos_phi * cos_phi;
            if alpha < -tol.confusion {
                SphereConeIntersection::Empty
            } else {
                SphereConeIntersection::TangentCircle(mk(alpha.max(0.0)))
            }
        } else {
            let root = disc.sqrt() * cos_phi;
            let base = tc * cos_phi * cos_phi;
            let (a1, a2) = (base + root, base - root);
            match (a1 >= -tol.confusion, a2 >= -tol.confusion) {
                (true, true) => SphereConeIntersection::TwoCircles(mk(a1), mk(a2)),
                (true, false) => SphereConeIntersection::Circle(mk(a1)),
                (false, true) => SphereConeIntersection::Circle(mk(a2)),
                (false, false) => SphereConeIntersection::Empty,
            }
        }
    }

    /// Intersects this sphere with another sphere.
    ///
    /// Concentric spheres (center distance within `tol.confusion`) are
    /// [`SphereSphereIntersection::Coincident`] when the radii agree within
    /// tolerance, else [`SphereSphereIntersection::Empty`]. Otherwise the
    /// radical plane cuts the center line at distance
    /// `a = (r1^2 - r2^2 + d^2) / (2d)` from this center; a negative
    /// `r1^2 - a^2` (beyond tolerance) misses
    /// ([`SphereSphereIntersection::Empty`]), a near-zero one grazes in a
    /// single [`SphereSphereIntersection::TangentPoint`], and the rest meet
    /// in a [`Circle3D`] whose axis points from this center to the other.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Sphere, SphereSphereIntersection, Tolerance};
    /// let s1 = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    /// let s2 = Sphere::new(Point3D::new(3.0, 0.0, 0.0), 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match s1.intersect_sphere(&s2, tol) {
    ///     SphereSphereIntersection::Circle(circle) => {
    ///         assert_eq!(circle.center(), Point3D::new(1.5, 0.0, 0.0));
    ///     }
    ///     _ => panic!("expected a circle"),
    /// }
    /// ```
    pub fn intersect_sphere(&self, other: &Sphere, tol: Tolerance) -> SphereSphereIntersection {
        let c1 = self.center();
        let c2 = other.center();
        let r1 = self.radius();
        let r2 = other.radius();
        let axis = c2 - c1;
        let d = axis.magnitude();
        if d <= tol.confusion {
            if (r1 - r2).abs() <= tol.confusion {
                SphereSphereIntersection::Coincident
            } else {
                SphereSphereIntersection::Empty
            }
        } else {
            let dir = axis * (1.0 / d);
            let a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
            let center = c1 + dir * a;
            let disc = r1 * r1 - a * a;
            // Tangent band in length-squared units: |R - |h|| * |R + |h||.
            let band = tol.confusion * 2.0 * (r1 + r2 + d).max(1.0);
            if disc < -band {
                SphereSphereIntersection::Empty
            } else if disc <= band {
                SphereSphereIntersection::TangentPoint(center)
            } else {
                // disc > 0, and dir is unit: construction cannot fail.
                let circle = Circle3D::new(center, dir, disc.sqrt())
                    .expect("section radius is positive by construction");
                SphereSphereIntersection::Circle(circle)
            }
        }
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

    #[test]
    fn test_sphere_sphere_intersection_circle() {
        let tol = Tolerance::DEFAULT;
        let s1 = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        let s2 = Sphere::new(Point3D::new(3.0, 0.0, 0.0), 2.0).unwrap();
        match s1.intersect_sphere(&s2, tol) {
            SphereSphereIntersection::Circle(circle) => {
                assert_eq!(circle.center(), Point3D::new(1.5, 0.0, 0.0));
                assert!((circle.radius() - 1.75f64.sqrt()).abs() < 1e-12);
                assert_eq!(circle.normal(), Vector3D::X);
                // The circle lies on both spheres.
                assert!(s1.contains(circle.eval_point(1.0), tol));
                assert!(s2.contains(circle.eval_point(2.0), tol));
            }
            _ => panic!("expected a circle"),
        }
    }

    #[test]
    fn test_sphere_sphere_intersection_degenerate() {
        let tol = Tolerance::DEFAULT;
        let s1 = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        // External tangency.
        match s1.intersect_sphere(&Sphere::new(Point3D::new(4.0, 0.0, 0.0), 2.0).unwrap(), tol) {
            SphereSphereIntersection::TangentPoint(p) => assert_eq!(p, Point3D::new(2.0, 0.0, 0.0)),
            _ => panic!("expected a tangent point"),
        }
        // Separate.
        match s1.intersect_sphere(&Sphere::new(Point3D::new(5.0, 0.0, 0.0), 2.0).unwrap(), tol) {
            SphereSphereIntersection::Empty => {}
            _ => panic!("expected empty"),
        }
        // One inside the other without contact.
        match s1.intersect_sphere(&Sphere::new(Point3D::ORIGIN, 1.0).unwrap(), tol) {
            SphereSphereIntersection::Empty => {}
            _ => panic!("expected empty"),
        }
        // Coincident.
        match s1.intersect_sphere(&Sphere::new(Point3D::ORIGIN, 2.0).unwrap(), tol) {
            SphereSphereIntersection::Coincident => {}
            _ => panic!("expected coincident"),
        }
    }

    #[test]
    fn test_sphere_cylinder_intersection() {
        let tol = Tolerance::DEFAULT;
        let sphere = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        match sphere.intersect_cylinder(&cylinder, tol) {
            SphereCylinderIntersection::TwoCircles(c1, c2) => {
                assert_eq!(c1.radius(), 2.0);
                assert_eq!(c2.radius(), 2.0);
                assert!((c1.center().z - 5.0f64.sqrt()).abs() < 1e-9);
                assert!((c2.center().z + 5.0f64.sqrt()).abs() < 1e-9);
                for p in c1.eval_points(&[0.0, 2.0]) {
                    assert!(sphere.contains(p, tol));
                    assert!(cylinder.contains(p, tol));
                }
            }
            _ => panic!("expected two circles"),
        }
        // Grazing tube.
        let grazing = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 3.0).unwrap();
        match sphere.intersect_cylinder(&grazing, tol) {
            SphereCylinderIntersection::Circle(c) => {
                assert_eq!(c.center(), Point3D::ORIGIN);
                assert_eq!(c.radius(), 3.0);
            }
            _ => panic!("expected a circle"),
        }
        // Oversize tube misses.
        let fat = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 4.0).unwrap();
        assert_eq!(
            sphere.intersect_cylinder(&fat, tol),
            SphereCylinderIntersection::Empty
        );
        // Offset axis: space quartic, no closed form.
        let off = Cylinder::new(Point3D::new(0.0, 1.0, 0.0), Vector3D::Z, 2.0).unwrap();
        assert_eq!(
            sphere.intersect_cylinder(&off, tol),
            SphereCylinderIntersection::NotAnalytic
        );
    }

    #[test]
    fn test_sphere_cone_intersection() {
        let tol = Tolerance::DEFAULT;
        let sphere = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
        let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
        match sphere.intersect_cone(&cone, tol) {
            SphereConeIntersection::TwoCircles(c1, c2) => {
                // Rings at axial 6.194 (r 2.619) and 1.832 (r 0.775).
                assert!((c1.center().z - 1.4636).abs() < 1e-3);
                assert!((c1.radius() - 2.6188).abs() < 1e-3);
                assert!((c2.center().z + 2.8983).abs() < 1e-3);
                assert!((c2.radius() - 0.7746).abs() < 1e-3);
                for p in c1.eval_points(&[0.0, 2.0]) {
                    assert!(sphere.contains(p, tol));
                    assert!(cone.contains(p, tol));
                }
                for p in c2.eval_points(&[1.0, 3.0]) {
                    assert!(sphere.contains(p, tol));
                    assert!(cone.contains(p, tol));
                }
            }
            _ => panic!("expected two circles"),
        }
        // Small sphere around the origin: no nappe contact.
        let small = Sphere::new(Point3D::ORIGIN, 0.5).unwrap();
        assert_eq!(
            small.intersect_cone(&cone, tol),
            SphereConeIntersection::Empty
        );
        // Behind-apex sphere: roots off the nappe.
        let behind = Sphere::new(Point3D::new(0.0, 0.0, -6.0), 0.5).unwrap();
        assert_eq!(
            behind.intersect_cone(&cone, tol),
            SphereConeIntersection::Empty
        );
        // Off-axis center: space quartic, no closed form.
        let off = Sphere::new(Point3D::new(1.0, 0.0, 0.0), 1.0).unwrap();
        assert_eq!(
            off.intersect_cone(&cone, tol),
            SphereConeIntersection::NotAnalytic
        );
    }

    #[test]
    fn test_sphere_cone_intersection_tangent() {
        let tol = Tolerance::DEFAULT;
        // R = tc*sin(phi): grazing ring at alpha = tc*cos^2(phi).
        let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
        let tc = (Point3D::ORIGIN - cone.apex()).dot(Vector3D::Z);
        let sphere = Sphere::new(Point3D::ORIGIN, tc * 0.4f64.sin()).unwrap();
        match sphere.intersect_cone(&cone, tol) {
            SphereConeIntersection::TangentCircle(c) => {
                assert!(sphere.contains(c.eval_point(0.0), tol));
                assert!(cone.contains(c.eval_point(1.0), tol));
            }
            _ => panic!("expected a tangent circle"),
        }
    }
}
