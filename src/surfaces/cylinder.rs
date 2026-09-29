//! Cylinders in 3D: parametric evaluation, parameter inversion, and
//! construction from an axis or an existing circle, thin wrappers over
//! [`crate::surface_math::analytic`].

use crate::projection::{self, SurfaceProjection};
use crate::surface_math::analytic;
use crate::{
    Axis3D, Circle3D, CylinderCylinderIntersection, Frame3D, Line3D, Point3D, Tolerance, Vector3D,
};
use std::fmt;

/// Error returned when a [`Cylinder`] cannot be constructed from the given
/// inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CylinderConstructionError {
    /// The requested radius is negative.
    NegativeRadius,
    /// The axis direction has zero length.
    NullNormal,
}

impl fmt::Display for CylinderConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            CylinderConstructionError::NegativeRadius => "radius is negative",
            CylinderConstructionError::NullNormal => "axis direction has zero length",
        };
        f.write_str(message)
    }
}

impl std::error::Error for CylinderConstructionError {}

/// A cylinder in 3D: a [`Frame3D`] (origin plus local x/y/z directions, `z`
/// being the axis) and a radius, evaluated as
/// `origin + r*cos(u)*x_dir + r*sin(u)*y_dir + v*z_dir`.
///
/// # Examples
///
/// ```
/// use geomcore::{Cylinder, Point3D, Vector3D};
/// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
/// assert_eq!(cylinder.eval_point(0.0, 5.0), Point3D::new(2.0, 0.0, 5.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cylinder {
    frame: Frame3D,
    radius: f64,
}

impl Cylinder {
    /// Creates a cylinder from a center, an axis direction, and a radius.
    ///
    /// The frame is derived from `axis_direction` via [`Frame3D::from_z`].
    ///
    /// # Errors
    ///
    /// Returns [`CylinderConstructionError::NullNormal`] if
    /// `axis_direction` cannot be normalized (zero length), or
    /// [`CylinderConstructionError::NegativeRadius`] if `radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cylinder, Point3D, Vector3D};
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// assert_eq!(cylinder.radius(), 2.0);
    /// ```
    pub fn new(
        center: Point3D,
        axis_direction: Vector3D,
        radius: f64,
    ) -> Result<Cylinder, CylinderConstructionError> {
        let frame = Frame3D::from_z(center, axis_direction)
            .map_err(|_| CylinderConstructionError::NullNormal)?;
        Cylinder::from_frame(frame, radius)
    }

    /// Creates a cylinder from a frame and a radius directly.
    ///
    /// # Errors
    ///
    /// Returns [`CylinderConstructionError::NegativeRadius`] if
    /// `radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cylinder, Frame3D};
    /// let cylinder = Cylinder::from_frame(Frame3D::WORLD, 2.0).unwrap();
    /// assert_eq!(cylinder.frame(), Frame3D::WORLD);
    /// ```
    pub fn from_frame(frame: Frame3D, radius: f64) -> Result<Cylinder, CylinderConstructionError> {
        if radius < 0.0 {
            return Err(CylinderConstructionError::NegativeRadius);
        }
        Ok(Cylinder { frame, radius })
    }

    /// Creates a cylinder from a main axis (center and direction) and a
    /// radius.
    ///
    /// # Errors
    ///
    /// Returns [`CylinderConstructionError::NegativeRadius`] if
    /// `radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Axis3D, Cylinder, Point3D, Vector3D};
    /// let axis = Axis3D::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let cylinder = Cylinder::from_axis(axis, 2.0).unwrap();
    /// assert_eq!(cylinder.axis(), axis);
    /// ```
    pub fn from_axis(axis: Axis3D, radius: f64) -> Result<Cylinder, CylinderConstructionError> {
        let frame = Frame3D::from_z(axis.origin(), axis.direction())
            .map_err(|_| CylinderConstructionError::NullNormal)?;
        Cylinder::from_frame(frame, radius)
    }

    /// Creates a cylinder from an existing circle: the cylinder's frame and
    /// radius are exactly the circle's. Infallible: a [`Circle3D`] already
    /// carries a valid (non-negative) radius.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle3D, Cylinder, Point3D, Vector3D};
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let cylinder = Cylinder::from_circle(&circle);
    /// assert_eq!(cylinder.radius(), 2.0);
    /// assert_eq!(cylinder.frame(), circle.frame());
    /// ```
    pub fn from_circle(circle: &Circle3D) -> Cylinder {
        Cylinder {
            frame: circle.frame(),
            radius: circle.radius(),
        }
    }

    /// Returns the cylinder's frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cylinder, Frame3D};
    /// let cylinder = Cylinder::from_frame(Frame3D::WORLD, 2.0).unwrap();
    /// assert_eq!(cylinder.frame(), Frame3D::WORLD);
    /// ```
    pub fn frame(&self) -> Frame3D {
        self.frame
    }

    /// Returns the cylinder's radius.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cylinder, Frame3D};
    /// let cylinder = Cylinder::from_frame(Frame3D::WORLD, 2.0).unwrap();
    /// assert_eq!(cylinder.radius(), 2.0);
    /// ```
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Returns the cylinder's main axis (origin and z direction).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cylinder, Point3D, Vector3D};
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// assert_eq!(cylinder.axis().direction(), Vector3D::Z);
    /// ```
    pub fn axis(&self) -> Axis3D {
        self.frame.axis()
    }

    /// Evaluates the point on the cylinder at `(u, v)`:
    /// `origin + r*cos(u)*x_dir + r*sin(u)*y_dir + v*z_dir`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cylinder, Point3D, Vector3D};
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// assert_eq!(cylinder.eval_point(0.0, 5.0), Point3D::new(2.0, 0.0, 5.0));
    /// ```
    pub fn eval_point(&self, u: f64, v: f64) -> Point3D {
        analytic::cylinder_d0(&self.frame, self.radius, u, v)
    }

    /// Evaluates the points on the cylinder at each `(u, v)` in `uvs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cylinder, Point3D, Vector3D};
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let points = cylinder.eval_points(&[(0.0, 0.0)]);
    /// assert_eq!(points[0], Point3D::new(2.0, 0.0, 0.0));
    /// ```
    pub fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3D> {
        uvs.iter().map(|&(u, v)| self.eval_point(u, v)).collect()
    }

    /// Evaluates the derivative of order `(du, dv)` at `(u, v)`.
    ///
    /// `Su = r*(-sin(u)*x_dir + cos(u)*y_dir)`, `Sv = z_dir`,
    /// `Suu = -r*(cos(u)*x_dir + sin(u)*y_dir)`; `Svv` and `Suv` are
    /// identically zero.
    ///
    /// # Panics
    ///
    /// Panics if `du + dv == 0` (use [`Cylinder::eval_point`] for the
    /// position itself) or if `du + dv > 2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cylinder, Point3D, Vector3D};
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// assert_eq!(cylinder.eval_derivative(0.0, 5.0, 0, 1), Vector3D::Z);
    /// ```
    pub fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3D {
        match du + dv {
            0 => panic!(
                "eval_derivative: du + dv must be >= 1 (use eval_point for the (0, 0) order)"
            ),
            1..=2 => analytic::cylinder_derivative(&self.frame, self.radius, u, v, du, dv),
            _ => panic!(
                "eval_derivative: order du={du}, dv={dv} is not supported (du + dv must be <= 2)"
            ),
        }
    }

    /// Recovers `(u, v)` of a point on (or near) the cylinder:
    /// `u = atan2(y, x)` wrapped into `[0, 2*PI)`, `v = z`, in the
    /// cylinder's local coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cylinder, Point3D, Vector3D};
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let (u, v) = cylinder.parameters_of(Point3D::new(2.0, 0.0, 5.0));
    /// assert_eq!((u, v), (0.0, 5.0));
    /// ```
    pub fn parameters_of(&self, point: Point3D) -> (f64, f64) {
        analytic::cylinder_parameters(&self.frame, point)
    }

    /// Returns whether `point` lies on the cylinder: the inverse parameters
    /// are recovered with [`Cylinder::parameters_of`] and re-evaluated, and
    /// the point counts as contained when the re-evaluated point is within
    /// `tol.confusion` of it.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cylinder, Point3D, Tolerance, Vector3D};
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// assert!(cylinder.contains(Point3D::new(2.0, 0.0, 0.0), tol));
    /// assert!(!cylinder.contains(Point3D::ORIGIN, tol));
    /// ```
    pub fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        let (u, v) = self.parameters_of(point);
        self.eval_point(u, v).distance(point) <= tol.confusion
    }

    /// Projects `point` onto the cylinder, returning the `(u, v)`
    /// parameters of the closest point and its distance. Distances within
    /// `tol.confusion` snap to `0.0`, matching [`Cylinder::contains`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cylinder, Point3D, Tolerance, Vector3D};
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let proj = cylinder.project_point(Point3D::new(3.0, 0.0, 5.0), Tolerance::DEFAULT);
    /// assert_eq!((proj.u, proj.v), (0.0, 5.0));
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

    /// Projects each point in `points` onto the cylinder.
    ///
    /// Default-style batch wrapper over [`Cylinder::project_point`]: one
    /// native call per batch, mirroring [`Cylinder::eval_points`].
    pub fn project_points(&self, points: &[Point3D], tol: Tolerance) -> Vec<SurfaceProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }

    /// Intersects this cylinder with another cylinder.
    ///
    /// The analytic path needs parallel axes (against `tol.angular`):
    /// with the axis separation `d`, disjoint (`d` beyond `r1 + r2`),
    /// nested (`d` below `|r1 - r2|`), grazing, and twin-generator cases
    /// follow the planar two-circle logic, positioned by the radical
    /// formula `s = (r1^2 - r2^2 + d^2) / (2d)`. Coaxial cylinders of
    /// equal radius coincide; concentric ones of different radii miss.
    /// Skew or crossing axes give a space quartic and report
    /// [`CylinderCylinderIntersection::NotAnalytic`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cylinder, CylinderCylinderIntersection, Point3D, Tolerance, Vector3D};
    /// let c1 = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let c2 = Cylinder::new(Point3D::new(3.0, 0.0, 0.0), Vector3D::Z, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match c1.intersect_cylinder(&c2, tol) {
    ///     CylinderCylinderIntersection::TwoLines(_, _) => {}
    ///     _ => panic!("expected two lines"),
    /// }
    /// ```
    pub fn intersect_cylinder(
        &self,
        other: &Cylinder,
        tol: Tolerance,
    ) -> CylinderCylinderIntersection {
        let a1 = self.axis().direction();
        if self
            .axis()
            .direction()
            .cross(other.axis().direction())
            .magnitude()
            > tol.angular
        {
            return CylinderCylinderIntersection::NotAnalytic;
        }
        let w = other.axis().origin() - self.axis().origin();
        let m = w - a1 * w.dot(a1);
        let d = m.magnitude();
        let (r1, r2) = (self.radius(), other.radius());
        if d <= tol.confusion {
            if (r1 - r2).abs() <= tol.confusion {
                return CylinderCylinderIntersection::Coincident;
            }
            return CylinderCylinderIntersection::Empty;
        }
        let out = r1 + r2;
        let inn = (r1 - r2).abs();
        if d > out + tol.confusion || d < inn - tol.confusion {
            return CylinderCylinderIntersection::Empty;
        }
        let expect = "generator direction is unit by construction";
        let mhat = m * (1.0 / d);
        if (d - out).abs() <= tol.confusion || (d - inn).abs() <= tol.confusion {
            let p = self.axis().origin() + mhat * ((r1 * r1 - r2 * r2 + d * d) / (2.0 * d));
            CylinderCylinderIntersection::TangentLine(Line3D::new(p, a1).expect(expect))
        } else {
            let s = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
            let h = (r1 * r1 - s * s).max(0.0).sqrt();
            let base = self.axis().origin() + mhat * s;
            let nhat = a1.cross(mhat);
            CylinderCylinderIntersection::TwoLines(
                Line3D::new(base + nhat * h, a1).expect(expect),
                Line3D::new(base - nhat * h, a1).expect(expect),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector3D;
    use std::f64::consts::PI;

    // ---- construction ----

    #[test]
    fn test_new_ok() {
        let c = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        assert_eq!(c.radius(), 2.0);
        assert_eq!(c.axis().direction(), Vector3D::Z);
    }

    #[test]
    fn test_new_negative_radius_errors() {
        assert_eq!(
            Cylinder::new(Point3D::ORIGIN, Vector3D::Z, -1.0),
            Err(CylinderConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_new_null_normal_errors() {
        assert_eq!(
            Cylinder::new(Point3D::ORIGIN, Vector3D::ZERO, 1.0),
            Err(CylinderConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_from_frame_negative_radius_errors() {
        assert_eq!(
            Cylinder::from_frame(Frame3D::WORLD, -0.1),
            Err(CylinderConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_from_axis_ok() {
        let axis = Axis3D::new(Point3D::new(1.0, 2.0, 3.0), Vector3D::Y).unwrap();
        let c = Cylinder::from_axis(axis, 3.0).unwrap();
        assert_eq!(c.axis(), axis);
        assert_eq!(c.radius(), 3.0);
    }

    #[test]
    fn test_from_axis_negative_radius_errors() {
        let axis = Axis3D::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        assert_eq!(
            Cylinder::from_axis(axis, -1.0),
            Err(CylinderConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_from_circle_matches_circle() {
        let circle = Circle3D::new(Point3D::new(1.0, 2.0, 3.0), Vector3D::Y, 2.5).unwrap();
        let cylinder = Cylinder::from_circle(&circle);
        assert_eq!(cylinder.frame(), circle.frame());
        assert_eq!(cylinder.radius(), circle.radius());
    }

    // ---- evaluation ----

    #[test]
    fn test_eval_point() {
        let c = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        assert_eq!(c.eval_point(0.0, 5.0), Point3D::new(2.0, 0.0, 5.0));
    }

    #[test]
    fn test_eval_points_matches_loop() {
        let c = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let uvs = [(0.0, 0.0), (PI / 2.0, 1.0)];
        let expected: Vec<Point3D> = uvs.iter().map(|&(u, v)| c.eval_point(u, v)).collect();
        assert_eq!(c.eval_points(&uvs), expected);
    }

    #[test]
    fn test_eval_derivative_first_orders() {
        let c = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        assert_eq!(c.eval_derivative(0.0, 5.0, 0, 1), Vector3D::Z);
    }

    #[test]
    fn test_eval_derivative_second_orders_zero_for_sv() {
        let c = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        assert_eq!(c.eval_derivative(0.7, 5.0, 0, 2), Vector3D::ZERO);
        assert_eq!(c.eval_derivative(0.7, 5.0, 1, 1), Vector3D::ZERO);
    }

    #[test]
    #[should_panic(expected = "du + dv must be >= 1")]
    fn test_eval_derivative_zero_order_panics() {
        let c = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        c.eval_derivative(0.0, 0.0, 0, 0);
    }

    #[test]
    #[should_panic(expected = "du + dv must be <= 2")]
    fn test_eval_derivative_order_too_high_panics() {
        let c = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        c.eval_derivative(0.0, 0.0, 2, 1);
    }

    #[test]
    fn test_parameters_of_round_trip() {
        let c = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let (u, v) = c.parameters_of(Point3D::new(2.0, 0.0, 5.0));
        assert_eq!((u, v), (0.0, 5.0));
    }

    // ---- CylinderConstructionError ----

    #[test]
    fn test_error_display() {
        assert_eq!(
            CylinderConstructionError::NegativeRadius.to_string(),
            "radius is negative"
        );
        assert_eq!(
            CylinderConstructionError::NullNormal.to_string(),
            "axis direction has zero length"
        );
    }

    #[test]
    fn test_error_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&CylinderConstructionError::NegativeRadius);
    }

    #[test]
    fn test_cylinder_contains() {
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let tol = Tolerance::DEFAULT;
        assert!(cylinder.contains(Point3D::new(2.0, 0.0, 0.0), tol));
        assert!(cylinder.contains(cylinder.eval_point(1.0, 3.0), tol));
        assert!(!cylinder.contains(Point3D::ORIGIN, tol));
        assert!(!cylinder.contains(Point3D::new(3.0, 0.0, 0.0), tol));
    }

    #[test]
    fn test_cylinder_project_point() {
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let tol = Tolerance::DEFAULT;
        let proj = cylinder.project_point(Point3D::new(3.0, 0.0, 5.0), tol);
        assert_eq!((proj.u, proj.v), (0.0, 5.0));
        assert_eq!(proj.distance, 1.0);
        // Axis point: the whole height circle is equidistant; distance is exact.
        let axis = cylinder.project_point(Point3D::new(0.0, 0.0, 5.0), tol);
        assert_eq!(axis.v, 5.0);
        assert_eq!(axis.distance, 2.0);
    }

    #[test]
    fn test_cylinder_cylinder_intersection_two_lines() {
        let tol = Tolerance::DEFAULT;
        let c1 = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let c2 = Cylinder::new(Point3D::new(3.0, 0.0, 0.0), Vector3D::Z, 2.0).unwrap();
        match c1.intersect_cylinder(&c2, tol) {
            CylinderCylinderIntersection::TwoLines(l1, l2) => {
                assert_eq!(l1.direction(), Vector3D::Z);
                assert!((l1.origin().x - 1.5).abs() < 1e-12);
                assert!((l1.origin().y - 1.75f64.sqrt()).abs() < 1e-9);
                assert!((l2.origin().y + 1.75f64.sqrt()).abs() < 1e-9);
                for p in [l1.eval_point(2.0), l2.eval_point(-1.0)] {
                    assert!(c1.contains(p, tol));
                    assert!(c2.contains(p, tol));
                }
            }
            _ => panic!("expected two lines"),
        }
    }

    #[test]
    fn test_cylinder_cylinder_intersection_degenerate() {
        let tol = Tolerance::DEFAULT;
        let c1 = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        // External tangency.
        let tangent = Cylinder::new(Point3D::new(4.0, 0.0, 0.0), Vector3D::Z, 2.0).unwrap();
        match c1.intersect_cylinder(&tangent, tol) {
            CylinderCylinderIntersection::TangentLine(l) => {
                assert_eq!(l.origin(), Point3D::new(2.0, 0.0, 0.0));
            }
            _ => panic!("expected a tangent line"),
        }
        // Separate.
        let far = Cylinder::new(Point3D::new(5.0, 0.0, 0.0), Vector3D::Z, 2.0).unwrap();
        assert_eq!(
            c1.intersect_cylinder(&far, tol),
            CylinderCylinderIntersection::Empty
        );
        // Concentric, different radii.
        let nested = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
        assert_eq!(
            c1.intersect_cylinder(&nested, tol),
            CylinderCylinderIntersection::Empty
        );
        // Coincident.
        assert_eq!(
            c1.intersect_cylinder(&c1, tol),
            CylinderCylinderIntersection::Coincident
        );
        // Crossing axes: space quartic, no closed form.
        let crossed = Cylinder::new(Point3D::ORIGIN, Vector3D::X, 2.0).unwrap();
        assert_eq!(
            c1.intersect_cylinder(&crossed, tol),
            CylinderCylinderIntersection::NotAnalytic
        );
    }
}
