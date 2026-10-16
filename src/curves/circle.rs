//! Circles in 2D and 3D: parametric evaluation, parameter inversion, and
//! circumcircle construction, thin wrappers over
//! [`crate::curve_math::analytic`].

use crate::Intersect;
use crate::curve_math::analytic;
use crate::curves::Curve2D;
use crate::curves::parametrize::{self, ParametrizeError};
use crate::frame::frame_error;
use crate::intersect::{
    CircleSurfaceIntersection, ConicConicIntersection, ConicSurfaceIntersection,
    intersect_conic_quadric,
};
use crate::projection::{self, CurveProjection};
use crate::surfaces::Surface;
use crate::tol;
use crate::{
    Axis3D, CircleCircle2DIntersection, CircleCircle3DIntersection, Cone, Cylinder, Ellipse3D,
    Frame2D, Frame3D, Hyperbola3D, Parabola3D, Plane, Point2D, Point3D, Sphere, Tolerance,
    Transform, Vector2D, Vector3D,
};
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
    /// An input coordinate, radius, or angle is NaN or infinite.
    NonFinite,
}

impl fmt::Display for CircleConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            CircleConstructionError::NegativeRadius => "radius is negative",
            CircleConstructionError::NullNormal => "normal has zero length",
            CircleConstructionError::ConfusedPoints => "the points are confused",
            CircleConstructionError::CollinearPoints => "the points are collinear",
            CircleConstructionError::NonFinite => "an input is NaN or infinite",
        };
        f.write_str(message)
    }
}

impl std::error::Error for CircleConstructionError {}

/// A circle in 3D: a plane [`Frame3D`] (origin plus local x/y directions
/// defining the plane and the angular origin) and a radius, evaluated as
/// `origin + radius*cos(u)*x_dir + radius*sin(u)*y_dir`.
///
/// # Examples
///
/// ```
/// use geomcore::{Circle3D, Point3D, Vector3D};
/// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
/// assert_eq!(circle.eval_point(0.0), Point3D::new(2.0, 0.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle3D {
    frame: Frame3D,
    radius: f64,
}

impl Circle3D {
    /// Creates a circle from a center, a normal, and a radius.
    ///
    /// The plane frame is derived from `normal` via [`Frame3D::from_z`].
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
    /// use geomcore::{Circle3D, Point3D, Vector3D};
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// assert_eq!(circle.radius(), 2.0);
    /// assert_eq!(circle.normal(), Vector3D::Z);
    /// ```
    pub fn new(
        center: Point3D,
        normal: Vector3D,
        radius: f64,
    ) -> Result<Circle3D, CircleConstructionError> {
        let frame = Frame3D::from_z(center, normal).map_err(frame_error(
            CircleConstructionError::NonFinite,
            CircleConstructionError::NullNormal,
        ))?;
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
    /// use geomcore::{Axis3D, Circle3D, Point3D, Vector3D};
    /// let axis = Axis3D::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let circle = Circle3D::from_axis(axis, 2.0).unwrap();
    /// assert_eq!(circle.center(), Point3D::ORIGIN);
    /// ```
    pub fn from_axis(axis: Axis3D, radius: f64) -> Result<Circle3D, CircleConstructionError> {
        let frame = Frame3D::from_z(axis.origin(), axis.direction()).map_err(frame_error(
            CircleConstructionError::NonFinite,
            CircleConstructionError::NullNormal,
        ))?;
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
    /// use geomcore::{Circle3D, Frame3D};
    /// let circle = Circle3D::from_frame(Frame3D::WORLD, 2.0).unwrap();
    /// assert_eq!(circle.frame(), Frame3D::WORLD);
    /// ```
    pub fn from_frame(frame: Frame3D, radius: f64) -> Result<Circle3D, CircleConstructionError> {
        if !(radius.is_finite()) {
            return Err(CircleConstructionError::NonFinite);
        }
        if radius < 0.0 {
            return Err(CircleConstructionError::NegativeRadius);
        }
        Ok(Circle3D { frame, radius })
    }

    /// Creates the circumcircle through three points.
    ///
    /// Failure checks run before any circumcenter computation: any pairwise
    /// distance below `tol::CONFUSION` is reported as
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
    /// use geomcore::{Circle3D, Point3D};
    /// let circle = Circle3D::from_three_points(
    ///     Point3D::new(1.0, 0.0, 0.0),
    ///     Point3D::new(0.0, 1.0, 0.0),
    ///     Point3D::new(-1.0, 0.0, 0.0),
    /// )
    /// .unwrap();
    /// assert!((circle.radius() - 1.0).abs() < 1e-9);
    /// ```
    pub fn from_three_points(
        p1: Point3D,
        p2: Point3D,
        p3: Point3D,
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
        let frame = Frame3D::new(center, normal, p1 - center).map_err(frame_error(
            CircleConstructionError::NonFinite,
            CircleConstructionError::CollinearPoints,
        ))?;

        Circle3D::from_frame(frame, radius)
    }

    /// Returns the center of the circle.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle3D, Point3D, Vector3D};
    /// let circle = Circle3D::new(Point3D::new(1.0, 2.0, 3.0), Vector3D::Z, 2.0).unwrap();
    /// assert_eq!(circle.center(), Point3D::new(1.0, 2.0, 3.0));
    /// ```
    pub fn center(&self) -> Point3D {
        self.frame.origin()
    }

    /// Returns the radius of the circle.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle3D, Point3D, Vector3D};
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
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
    /// use geomcore::{Circle3D, Frame3D};
    /// let circle = Circle3D::from_frame(Frame3D::WORLD, 2.0).unwrap();
    /// assert_eq!(circle.frame(), Frame3D::WORLD);
    /// ```
    pub fn frame(&self) -> Frame3D {
        self.frame
    }

    /// Returns the unit normal of the circle's plane on the side the
    /// circle turns counterclockwise about: `x × y` of its frame. That is
    /// the frame's z direction on a direct frame and its negation on an
    /// indirect one (such as the image of a circle under a mirror).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle3D, Frame3D, Point3D, Transform, Vector3D};
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// assert_eq!(circle.normal(), Vector3D::Z);
    /// // Mirrored across its own plane, every point stays put, so the
    /// // normal is unchanged while the frame's z direction flips.
    /// let mirrored = circle.transformed(Transform::mirror_plane(Frame3D::WORLD)).unwrap();
    /// assert_eq!(mirrored.normal(), Vector3D::Z);
    /// assert_eq!(mirrored.frame().z_direction(), -Vector3D::Z);
    /// ```
    pub fn normal(&self) -> Vector3D {
        if self.frame.is_direct() {
            self.frame.z_direction()
        } else {
            -self.frame.z_direction()
        }
    }

    /// Returns the image of this circle under `transform`: the center and
    /// frame directions map point-wise while the radius scales by the
    /// transform's uniform scale factor.
    ///
    /// Every frame axis maps through the transform, so the image's
    /// parametrization is the transform of this one: under a rotation,
    /// translation or mirror, `transformed(t).eval_point(u)` equals
    /// `t.apply_point(self.eval_point(u))`, and a mirror changes the
    /// frame's handedness. Fails with [`CircleConstructionError::NullNormal`] only
    /// when the transform collapses the frame (a zero scale factor).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle3D, Point3D, Transform, Vector3D};
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let moved = circle.transformed(Transform::translation(Vector3D::new(0.0, 0.0, 5.0))).unwrap();
    /// assert_eq!(moved.center(), Point3D::new(0.0, 0.0, 5.0));
    /// assert_eq!(moved.radius(), 2.0);
    /// ```
    pub fn transformed(&self, transform: Transform) -> Result<Circle3D, CircleConstructionError> {
        let (frame, scale) = transform.apply_frame(self.frame()).map_err(frame_error(
            CircleConstructionError::NonFinite,
            CircleConstructionError::NullNormal,
        ))?;
        Circle3D::from_frame(frame, self.radius() * scale)
    }

    /// Evaluates the point on the circle at angular parameter `u`:
    /// `center + radius*cos(u)*x_dir + radius*sin(u)*y_dir`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle3D, Point3D, Vector3D};
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// assert_eq!(circle.eval_point(0.0), Point3D::new(2.0, 0.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64) -> Point3D {
        analytic::circle_d0(&self.frame, self.radius, u)
    }

    /// Evaluates the points on the circle at each parameter in `us`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle3D, Point3D, Vector3D};
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let points = circle.eval_points(&[0.0, 1.0]);
    /// assert_eq!(points[0], Point3D::new(2.0, 0.0, 0.0));
    /// ```
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        us.iter().map(|&u| self.eval_point(u)).collect()
    }

    /// Evaluates the derivative of the given `order` at parameter `u`.
    ///
    /// Derivatives cycle with period 4 in `order` (see
    /// `curve_math::analytic::circle_dn`).
    ///
    /// # Panics
    ///
    /// Panics if `order == 0`; use [`Circle3D::eval_point`] to evaluate the
    /// position itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle3D, Point3D, Vector3D};
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// assert_eq!(circle.eval_derivative(0.0, 1), Vector3D::new(0.0, 2.0, 0.0));
    /// ```
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        match order {
            0 => panic!("eval_derivative: order must be >= 1 (use eval_point for order 0)"),
            _ => analytic::circle_dn(&self.frame, self.radius, u, order),
        }
    }

    /// The parameter to report for a point found at parameter `u`: `u`
    /// wrapped into [`bounds`](crate::curves::ParametricCurve3D::bounds),
    /// and the start of the bounds when within `tol.confusion` of the seam
    /// along the curve, so a point at the seam has one parameter.
    pub(crate) fn parameter_in_bounds(&self, u: f64, tol: Tolerance) -> f64 {
        let (first, last) = crate::curves::ParametricCurve3D::bounds(self);
        // The speed is the radius everywhere.
        analytic::seam_wrapped(u, first, last - first, self.radius, tol.confusion)
    }

    /// Recovers the angular parameter of a point on (or near) the circle,
    /// wrapped into `[0, 2*PI)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle3D, Point3D, Vector3D};
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// assert!((circle.parameter_of(Point3D::new(0.0, 2.0, 0.0)) - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    /// ```
    pub fn parameter_of(&self, point: Point3D) -> f64 {
        analytic::circle_parameter(&self.frame, point)
    }

    /// Returns whether `point` lies on the circle: the inverse parameter is
    /// recovered with [`Circle3D::parameter_of`] and re-evaluated, and the
    /// point counts as contained when the re-evaluated point is within
    /// `tol.confusion` of it.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle3D, Point3D, Tolerance, Vector3D};
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// assert!(circle.contains(Point3D::new(2.0, 0.0, 0.0), tol));
    /// assert!(!circle.contains(Point3D::new(3.0, 0.0, 0.0), tol));
    /// ```
    pub fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        let u = self.parameter_of(point);
        self.eval_point(u).distance(point) <= tol.confusion
    }

    /// Projects `point` onto the circle, returning the parameter of the
    /// closest point and its distance. Distances within `tol.confusion`
    /// snap to `0.0`, matching [`Circle3D::contains`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle3D, Point3D, Tolerance, Vector3D};
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let proj = circle.project_point(Point3D::new(3.0, 0.0, 0.0), Tolerance::DEFAULT);
    /// assert_eq!(proj.parameter, 0.0);
    /// assert_eq!(proj.distance, 1.0);
    /// ```
    pub fn project_point(&self, point: Point3D, tol: Tolerance) -> CurveProjection {
        let u = self.parameter_of(point);
        let distance = self.eval_point(u).distance(point);
        CurveProjection {
            parameter: self.parameter_in_bounds(u, tol),
            distance: projection::snap_distance(distance, tol.confusion),
        }
    }

    /// Projects each point in `points` onto the circle.
    ///
    /// Default-style batch wrapper over [`Circle3D::project_point`]: one
    /// native call per batch, mirroring [`Circle3D::eval_points`].
    pub fn project_points(&self, points: &[Point3D], tol: Tolerance) -> Vec<CurveProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }

    /// Intersects with a [`Circle3D`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_circle(&self, other: &Circle3D, tol: Tolerance) -> CircleCircle3DIntersection {
        self.intersect(other, tol)
    }

    /// Intersects with an [`Ellipse3D`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_ellipse(&self, ellipse: &Ellipse3D, tol: Tolerance) -> ConicConicIntersection {
        self.intersect(ellipse, tol)
    }

    /// Intersects with a [`Parabola3D`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_parabola(
        &self,
        parabola: &Parabola3D,
        tol: Tolerance,
    ) -> ConicConicIntersection {
        self.intersect(parabola, tol)
    }

    /// Intersects with a [`Hyperbola3D`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_hyperbola(
        &self,
        hyperbola: &Hyperbola3D,
        tol: Tolerance,
    ) -> ConicConicIntersection {
        self.intersect(hyperbola, tol)
    }

    /// Circumference (`2*PI*radius`).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle3D, Point3D, Vector3D};
    /// use std::f64::consts::TAU;
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// assert_eq!(circle.circumference(), TAU * 2.0);
    /// ```
    pub fn circumference(&self) -> f64 {
        std::f64::consts::TAU * self.radius()
    }

    /// Area of the enclosed disk (`PI*radius^2`).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle3D, Point3D, Vector3D};
    /// use std::f64::consts::PI;
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// assert!((circle.disk_area() - PI * 4.0).abs() < 1e-12);
    /// ```
    pub fn disk_area(&self) -> f64 {
        std::f64::consts::PI * self.radius() * self.radius()
    }

    /// Intersects with a [`Plane`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_plane(&self, plane: &Plane, tol: Tolerance) -> CircleSurfaceIntersection {
        self.intersect(plane, tol)
    }

    /// Intersects with a [`Sphere`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_sphere(&self, sphere: &Sphere, tol: Tolerance) -> CircleSurfaceIntersection {
        self.intersect(sphere, tol)
    }

    /// Intersects with a [`Cylinder`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_cylinder(
        &self,
        cylinder: &Cylinder,
        tol: Tolerance,
    ) -> CircleSurfaceIntersection {
        self.intersect(cylinder, tol)
    }

    /// Intersects with a [`Cone`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_cone(&self, cone: &Cone, tol: Tolerance) -> CircleSurfaceIntersection {
        self.intersect(cone, tol)
    }

    /// Computes the exact 2D representation of this circle in a surface's
    /// parameter space: a [`Curve2D`] `q(t)` such that
    /// `surface.eval_point(q(t)) == self.eval_point(t)` for the same `t`.
    ///
    /// The closed-form cases are: a circle in a plane (maps to a 2D circle);
    /// a circle whose axis matches a cylinder or cone axis (a horizontal
    /// iso-`v` line in `(u, v)`); a meridian or parallel of a sphere; and a
    /// poloidal or toroidal circle of a torus (each a straight line in
    /// `(u, v)`). Every other pair returns
    /// [`ParametrizeError::NotAnalytic`].
    ///
    /// The projection math assumes the circle lies on the surface. As a
    /// geomcore safeguard, the candidate 2D image is verified against the
    /// surface at a few parameters; if `surface.eval_point(q(t))` is farther
    /// than `tol.confusion` from `self.eval_point(t)`,
    /// [`ParametrizeError::CurveNotOnSurface`] is returned (this also rejects, for example, a circle whose radius does
    /// not match the cylinder). For periodic surfaces the result is
    /// normalized so `q(0)` lies in the canonical parameter window.
    ///
    /// # Examples
    ///
    /// A cross-section circle of a cylinder maps to a horizontal line in
    /// `(u, v)` at the section height:
    ///
    /// ```
    /// use geomcore::curves::{Curve2D, ParametricCurve2D};
    /// use geomcore::{Circle3D, Cylinder, Point3D, Tolerance, Vector3D};
    ///
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let circle = Circle3D::new(Point3D::new(0.0, 0.0, 3.0), Vector3D::Z, 2.0).unwrap();
    /// let pcurve = circle.parametrize_on(&cylinder, Tolerance::DEFAULT).unwrap();
    /// // q(t) = (t, 3): the angular parameter runs in u, the height v is fixed.
    /// let q = pcurve.eval_point(1.0);
    /// assert!((q.x - 1.0).abs() < 1e-9);
    /// assert!((q.y - 3.0).abs() < 1e-9);
    /// assert!(matches!(pcurve, Curve2D::Line(_)));
    /// ```
    pub fn parametrize_on(
        &self,
        surface: impl Into<Surface>,
        tol: Tolerance,
    ) -> Result<Curve2D, ParametrizeError> {
        parametrize::circle_on_surface(self, &surface.into(), tol)
    }
}

/// A circle in 2D: a [`Frame2D`] (origin plus local x/y directions defining
/// the angular origin) and a radius, evaluated as
/// `origin + radius*cos(u)*x_dir + radius*sin(u)*y_dir`.
///
/// # Examples
///
/// ```
/// use geomcore::{Circle2D, Point2D};
/// let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
/// assert_eq!(circle.eval_point(0.0), Point2D::new(2.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle2D {
    frame: Frame2D,
    radius: f64,
}

impl Circle2D {
    /// Creates a circle from a center and a radius, using a world-aligned
    /// frame (x/y directions matching [`Vector2D::X`]/[`Vector2D::Y`]) at
    /// `center`.
    ///
    /// # Errors
    ///
    /// Returns [`CircleConstructionError::NegativeRadius`] if `radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle2D, Point2D};
    /// let circle = Circle2D::new(Point2D::new(1.0, 2.0), 3.0).unwrap();
    /// assert_eq!(circle.center(), Point2D::new(1.0, 2.0));
    /// ```
    pub fn new(center: Point2D, radius: f64) -> Result<Circle2D, CircleConstructionError> {
        // X and Y are orthonormal, so only a non-finite center can fail.
        let frame = Frame2D::new(center, Vector2D::X, Vector2D::Y)
            .map_err(|_| CircleConstructionError::NonFinite)?;
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
    /// use geomcore::{Circle2D, Frame2D};
    /// let circle = Circle2D::from_frame(Frame2D::WORLD, 2.0).unwrap();
    /// assert_eq!(circle.frame(), Frame2D::WORLD);
    /// ```
    pub fn from_frame(frame: Frame2D, radius: f64) -> Result<Circle2D, CircleConstructionError> {
        if !(radius.is_finite()) {
            return Err(CircleConstructionError::NonFinite);
        }
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
    /// use geomcore::{Circle2D, Point2D};
    /// let circle = Circle2D::new(Point2D::new(1.0, 2.0), 2.0).unwrap();
    /// assert_eq!(circle.center(), Point2D::new(1.0, 2.0));
    /// ```
    pub fn center(&self) -> Point2D {
        self.frame.origin()
    }

    /// Returns the radius of the circle.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle2D, Point2D};
    /// let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
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
    /// use geomcore::{Circle2D, Frame2D};
    /// let circle = Circle2D::from_frame(Frame2D::WORLD, 2.0).unwrap();
    /// assert_eq!(circle.frame(), Frame2D::WORLD);
    /// ```
    pub fn frame(&self) -> Frame2D {
        self.frame
    }

    /// Evaluates the point on the circle at angular parameter `u`:
    /// `center + radius*cos(u)*x_dir + radius*sin(u)*y_dir`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle2D, Point2D};
    /// let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
    /// assert_eq!(circle.eval_point(0.0), Point2D::new(2.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64) -> Point2D {
        analytic::circle2d_d0(&self.frame, self.radius, u)
    }

    /// Evaluates the points on the circle at each parameter in `us`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle2D, Point2D};
    /// let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
    /// let points = circle.eval_points(&[0.0, 1.0]);
    /// assert_eq!(points[0], Point2D::new(2.0, 0.0));
    /// ```
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point2D> {
        us.iter().map(|&u| self.eval_point(u)).collect()
    }

    /// Evaluates the derivative of the given `order` at parameter `u`.
    ///
    /// Derivatives cycle with period 4 in `order` (see
    /// `curve_math::analytic::circle2d_dn`).
    ///
    /// # Panics
    ///
    /// Panics if `order == 0`; use [`Circle2D::eval_point`] to evaluate the
    /// position itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle2D, Point2D, Vector2D};
    /// let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
    /// assert_eq!(circle.eval_derivative(0.0, 1), Vector2D::new(0.0, 2.0));
    /// ```
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector2D {
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
    /// use geomcore::{Circle2D, Point2D};
    /// let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
    /// assert!((circle.parameter_of(Point2D::new(0.0, 2.0)) - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    /// ```
    pub fn parameter_of(&self, point: Point2D) -> f64 {
        analytic::circle2d_parameter(&self.frame, point)
    }

    /// Returns whether `point` lies on the circle: the inverse parameter is
    /// recovered with [`Circle2D::parameter_of`] and re-evaluated, and the
    /// point counts as contained when the re-evaluated point is within
    /// `tol.confusion` of it.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle2D, Point2D, Tolerance};
    /// let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// assert!(circle.contains(Point2D::new(2.0, 0.0), tol));
    /// assert!(!circle.contains(Point2D::new(3.0, 0.0), tol));
    /// ```
    pub fn contains(&self, point: Point2D, tol: Tolerance) -> bool {
        let u = self.parameter_of(point);
        self.eval_point(u).distance(point) <= tol.confusion
    }

    /// Projects `point` onto the circle, returning the parameter of the
    /// closest point and its distance. Distances within `tol.confusion`
    /// snap to `0.0`, matching [`Circle2D::contains`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle2D, Point2D, Tolerance};
    /// let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
    /// let proj = circle.project_point(Point2D::new(3.0, 0.0), Tolerance::DEFAULT);
    /// assert_eq!(proj.parameter, 0.0);
    /// assert_eq!(proj.distance, 1.0);
    /// ```
    pub fn project_point(&self, point: Point2D, tol: Tolerance) -> CurveProjection {
        let u = self.parameter_of(point);
        let distance = self.eval_point(u).distance(point);
        CurveProjection {
            parameter: u,
            distance: projection::snap_distance(distance, tol.confusion),
        }
    }

    /// Projects each point in `points` onto the circle.
    ///
    /// Default-style batch wrapper over [`Circle2D::project_point`]: one
    /// native call per batch, mirroring [`Circle2D::eval_points`].
    pub fn project_points(&self, points: &[Point2D], tol: Tolerance) -> Vec<CurveProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }

    /// Intersects with a [`Circle2D`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_circle(&self, other: &Circle2D, tol: Tolerance) -> CircleCircle2DIntersection {
        self.intersect(other, tol)
    }
}

impl Intersect<Plane> for Circle3D {
    type Output = CircleSurfaceIntersection;

    /// Intersects this circle with a plane.
    ///
    /// In the circle frame the equation is first-order trigonometric
    /// (`C*cos t + D*sin t + E = 0`), solved exactly with tangency
    /// multiplicity; a coincident plane returns the circle itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{Circle3D, CircleSurfaceIntersection, Plane, Point3D, Tolerance, Vector3D};
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match circle.intersect(&plane, tol) {
    ///     CircleSurfaceIntersection::Hits(hits) => assert_eq!(hits.len(), 2),
    ///     _ => panic!("expected hits"),
    /// }
    /// ```
    fn intersect(&self, plane: &Plane, tol: Tolerance) -> CircleSurfaceIntersection {
        match intersect_conic_quadric(self, plane, tol) {
            ConicSurfaceIntersection::Coincident => CircleSurfaceIntersection::Circle(*self),
            ConicSurfaceIntersection::Hits(hits) => CircleSurfaceIntersection::Hits(hits),
            ConicSurfaceIntersection::Empty => CircleSurfaceIntersection::Empty,
        }
    }
}

impl Intersect<Sphere> for Circle3D {
    type Output = CircleSurfaceIntersection;

    /// Intersects this circle with a sphere.
    ///
    /// In circle-frame coordinates the sphere equation collapses to
    /// first-order trigonometric form (the `cos^2 + sin^2` terms fold to
    /// a constant), solved exactly; a fully contained circle returns
    /// itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{Circle3D, CircleSurfaceIntersection, Point3D, Sphere, Tolerance, Vector3D};
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let sphere = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match circle.intersect(&sphere, tol) {
    ///     CircleSurfaceIntersection::Empty => {}
    ///     _ => panic!("expected empty"),
    /// }
    /// ```
    fn intersect(&self, sphere: &Sphere, tol: Tolerance) -> CircleSurfaceIntersection {
        match intersect_conic_quadric(self, sphere, tol) {
            ConicSurfaceIntersection::Coincident => CircleSurfaceIntersection::Circle(*self),
            ConicSurfaceIntersection::Hits(hits) => CircleSurfaceIntersection::Hits(hits),
            ConicSurfaceIntersection::Empty => CircleSurfaceIntersection::Empty,
        }
    }
}

impl Intersect<Cylinder> for Circle3D {
    type Output = CircleSurfaceIntersection;

    /// Intersects this circle with a cylinder.
    ///
    /// The cylinder equation in circle-frame coordinates carries
    /// double-angle terms, solved as a quartic in `tan(t/2)`; a fully
    /// contained circle returns itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{Circle3D, CircleSurfaceIntersection, Cylinder, Point3D, Tolerance, Vector3D};
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// // The equator lies entirely on the cylinder.
    /// match circle.intersect(&cylinder, tol) {
    ///     CircleSurfaceIntersection::Circle(_) => {}
    ///     _ => panic!("expected the whole circle"),
    /// }
    /// ```
    fn intersect(&self, cylinder: &Cylinder, tol: Tolerance) -> CircleSurfaceIntersection {
        match intersect_conic_quadric(self, cylinder, tol) {
            ConicSurfaceIntersection::Coincident => CircleSurfaceIntersection::Circle(*self),
            ConicSurfaceIntersection::Hits(hits) => CircleSurfaceIntersection::Hits(hits),
            ConicSurfaceIntersection::Empty => CircleSurfaceIntersection::Empty,
        }
    }
}

impl Intersect<Cone> for Circle3D {
    type Output = CircleSurfaceIntersection;

    /// Intersects this circle with a cone.
    ///
    /// The half-angle equation in circle-frame coordinates carries
    /// double-angle terms, solved as a quartic in `tan(t/2)`; a fully
    /// contained circle returns itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{Circle3D, CircleSurfaceIntersection, Cone, Frame3D, Point3D, Tolerance, Vector3D};
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// // The reference circle lies entirely on the cone.
    /// match circle.intersect(&cone, tol) {
    ///     CircleSurfaceIntersection::Circle(_) => {}
    ///     _ => panic!("expected the whole circle"),
    /// }
    /// ```
    fn intersect(&self, cone: &Cone, tol: Tolerance) -> CircleSurfaceIntersection {
        match intersect_conic_quadric(self, cone, tol) {
            ConicSurfaceIntersection::Coincident => CircleSurfaceIntersection::Circle(*self),
            ConicSurfaceIntersection::Hits(hits) => CircleSurfaceIntersection::Hits(hits),
            ConicSurfaceIntersection::Empty => CircleSurfaceIntersection::Empty,
        }
    }
}

impl Intersect<Circle3D> for Circle3D {
    type Output = CircleCircle3DIntersection;

    /// Intersects this circle with another 3D circle.
    ///
    /// Only coplanar pairs admit a closed form. They are coplanar when one
    /// circle lies in the other's plane: turning it onto that plane moves
    /// it by at most `tol.confusion` (an angle below
    /// [`tol.angular_for_extent`](Tolerance::angular_for_extent) of its
    /// radius). Both circles then map to 2D images in that plane, the 2D
    /// problem solves, and hits map back through the plane evaluation.
    /// Distinct parallel planes report
    /// [`CircleCircle3DIntersection::Empty`]; crossing planes report
    /// [`CircleCircle3DIntersection::NotAnalytic`], which the
    /// [`ElementaryCurve`](crate::curves::ElementaryCurve) intersection
    /// resolves on the line the planes share.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{Circle3D, CircleCircle3DIntersection, Point3D, Tolerance, Vector3D};
    /// let c1 = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let c2 = Circle3D::new(Point3D::new(3.0, 0.0, 0.0), Vector3D::Z, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match c1.intersect(&c2, tol) {
    ///     CircleCircle3DIntersection::Points(_, _) => {}
    ///     _ => panic!("expected two points"),
    /// }
    /// ```
    fn intersect(&self, other: &Circle3D, tol: Tolerance) -> CircleCircle3DIntersection {
        let host = if lies_in_plane_of(other, self, tol) {
            self
        } else if lies_in_plane_of(self, other, tol) {
            other
        } else if self.normal().cross(other.normal()).magnitude() <= tol.angular {
            // Distinct parallel planes.
            return CircleCircle3DIntersection::Empty;
        } else {
            return CircleCircle3DIntersection::NotAnalytic;
        };
        if self.center().distance(other.center()) <= tol.confusion
            && (self.radius() - other.radius()).abs() <= tol.confusion
        {
            return CircleCircle3DIntersection::Coincident;
        }
        // Both circles in the host's plane coordinates.
        let plane = Plane::from_frame(host.frame());
        let image = |circle: &Circle3D| {
            let w = circle.center() - plane.frame().origin();
            let center = Point2D::new(
                w.dot(plane.frame().x_direction()),
                w.dot(plane.frame().y_direction()),
            );
            Circle2D::new(center, circle.radius()).expect("radius is non-negative by construction")
        };
        match image(self).intersect(&image(other), tol) {
            CircleCircle2DIntersection::Points(p1, p2) => CircleCircle3DIntersection::Points(
                plane.eval_point(p1.x, p1.y),
                plane.eval_point(p2.x, p2.y),
            ),
            CircleCircle2DIntersection::Tangent(p) => {
                CircleCircle3DIntersection::Tangent(plane.eval_point(p.x, p.y))
            }
            CircleCircle2DIntersection::Empty => CircleCircle3DIntersection::Empty,
            CircleCircle2DIntersection::Coincident => CircleCircle3DIntersection::Coincident,
        }
    }
}

/// Whether `circle` lies in the plane of `host`: its centre is on that
/// plane and turning its normal onto the host's moves it by at most the
/// confusion tolerance.
fn lies_in_plane_of(circle: &Circle3D, host: &Circle3D, tol: Tolerance) -> bool {
    circle.normal().cross(host.normal()).magnitude() <= tol.angular_for_extent(circle.radius())
        && Plane::from_frame(host.frame()).contains(circle.center(), tol)
}

impl Intersect<Circle2D> for Circle2D {
    type Output = CircleCircle2DIntersection;

    /// Intersects this circle with another 2D circle.
    ///
    /// Concentric circles (center distance within `tol.confusion`) are
    /// coincident for equal radii, else empty; otherwise the radical line
    /// positions the meeting points by `a = (r1^2 - r2^2 + d^2) / (2d)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{Circle2D, CircleCircle2DIntersection, Point2D, Tolerance};
    /// let c1 = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
    /// let c2 = Circle2D::new(Point2D::new(3.0, 0.0), 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match c1.intersect(&c2, tol) {
    ///     CircleCircle2DIntersection::Points(_, _) => {}
    ///     _ => panic!("expected two points"),
    /// }
    /// ```
    fn intersect(&self, other: &Circle2D, tol: Tolerance) -> CircleCircle2DIntersection {
        let (c1, c2) = (self.center(), other.center());
        let (r1, r2) = (self.radius(), other.radius());
        let w = c2 - c1;
        let d = w.magnitude();
        if d <= tol.confusion {
            if (r1 - r2).abs() <= tol.confusion {
                return CircleCircle2DIntersection::Coincident;
            }
            return CircleCircle2DIntersection::Empty;
        }
        let out = r1 + r2;
        let inn = (r1 - r2).abs();
        if d > out + tol.confusion || d < inn - tol.confusion {
            return CircleCircle2DIntersection::Empty;
        }
        let n = w * (1.0 / d);
        let a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
        let base = c1 + n * a;
        if (d - out).abs() <= tol.confusion || (d - inn).abs() <= tol.confusion {
            CircleCircle2DIntersection::Tangent(base)
        } else {
            let h = (r1 * r1 - a * a).max(0.0).sqrt();
            let perp = n.perp();
            CircleCircle2DIntersection::Points(base + perp * h, base - perp * h)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Intersect;
    use crate::{
        Circle2D, Circle3D, CircleCircle2DIntersection, CircleCircle3DIntersection,
        CircleConstructionError, CircleSurfaceIntersection, ConicConicIntersection, Ellipse3D,
        Frame2D, Frame3D, Hyperbola3D, Parabola3D, Point2D, Point3D, Tolerance, Transform,
        Vector2D, Vector3D,
    };
    use std::f64::consts::PI;

    // ---- Circle3D construction ----

    #[test]
    fn test_circle3d_new_ok() {
        let c = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        assert_eq!(c.center(), Point3D::ORIGIN);
        assert_eq!(c.radius(), 2.0);
        assert_eq!(c.normal(), Vector3D::Z);
    }

    #[test]
    fn test_circle3d_new_negative_radius_errors() {
        assert_eq!(
            Circle3D::new(Point3D::ORIGIN, Vector3D::Z, -1.0),
            Err(CircleConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_circle3d_new_null_normal_errors() {
        assert_eq!(
            Circle3D::new(Point3D::ORIGIN, Vector3D::ZERO, 1.0),
            Err(CircleConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_circle3d_from_axis_ok() {
        let axis = crate::Axis3D::new(Point3D::new(1.0, 2.0, 3.0), Vector3D::Y).unwrap();
        let c = Circle3D::from_axis(axis, 3.0).unwrap();
        assert_eq!(c.center(), Point3D::new(1.0, 2.0, 3.0));
        assert_eq!(c.normal(), Vector3D::Y);
        assert_eq!(c.radius(), 3.0);
    }

    #[test]
    fn test_circle3d_from_axis_negative_radius_errors() {
        let axis = crate::Axis3D::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        assert_eq!(
            Circle3D::from_axis(axis, -1.0),
            Err(CircleConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_circle3d_from_frame_ok() {
        let frame = Frame3D::WORLD;
        let c = Circle3D::from_frame(frame, 5.0).unwrap();
        assert_eq!(c.frame(), frame);
        assert_eq!(c.radius(), 5.0);
    }

    #[test]
    fn test_circle3d_from_frame_negative_radius_errors() {
        assert_eq!(
            Circle3D::from_frame(Frame3D::WORLD, -0.1),
            Err(CircleConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_circle3d_from_three_points_ok() {
        let p1 = Point3D::new(1.0, 0.0, 0.0);
        let p2 = Point3D::new(0.0, 1.0, 0.0);
        let p3 = Point3D::new(-1.0, 0.0, 0.0);
        let c = Circle3D::from_three_points(p1, p2, p3).unwrap();
        assert!((c.radius() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_circle3d_from_three_points_two_coincident_errors() {
        let p = Point3D::new(1.0, 2.0, 3.0);
        let p2 = Point3D::new(4.0, 5.0, 6.0);
        assert_eq!(
            Circle3D::from_three_points(p, p, p2),
            Err(CircleConstructionError::ConfusedPoints)
        );
    }

    #[test]
    fn test_circle3d_from_three_points_all_coincident_errors() {
        let p = Point3D::new(1.0, 2.0, 3.0);
        assert_eq!(
            Circle3D::from_three_points(p, p, p),
            Err(CircleConstructionError::ConfusedPoints)
        );
    }

    #[test]
    fn test_circle3d_from_three_points_collinear_errors() {
        let p1 = Point3D::new(0.0, 0.0, 0.0);
        let p2 = Point3D::new(1.0, 0.0, 0.0);
        let p3 = Point3D::new(2.0, 0.0, 0.0);
        assert_eq!(
            Circle3D::from_three_points(p1, p2, p3),
            Err(CircleConstructionError::CollinearPoints)
        );
    }

    // ---- Circle3D evaluation ----

    #[test]
    fn test_circle3d_eval_point_zero() {
        let c = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let p = c.eval_point(0.0);
        assert!((p.x - 2.0).abs() < 1e-9);
        assert!(p.y.abs() < 1e-9);
        assert!(p.z.abs() < 1e-9);
    }

    #[test]
    fn test_circle3d_eval_points_matches_loop() {
        let c = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let us = [0.0, 0.5, 1.5];
        let expected: Vec<Point3D> = us.iter().map(|&u| c.eval_point(u)).collect();
        assert_eq!(c.eval_points(&us), expected);
    }

    #[test]
    fn test_circle3d_eval_derivative_order1() {
        let c = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let d1 = c.eval_derivative(0.0, 1);
        assert!(d1.x.abs() < 1e-9);
        assert!((d1.y - 2.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic]
    fn test_circle3d_eval_derivative_order0_panics() {
        let c = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        c.eval_derivative(0.0, 0);
    }

    #[test]
    fn test_circle3d_parameter_of_round_trip() {
        let c = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        for u in [0.3, 2.0, 5.5] {
            let p = c.eval_point(u);
            assert!((c.parameter_of(p) - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_circle3d_parameter_of_in_zero_to_tau() {
        let c = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
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
        let c = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
        assert_eq!(c.center(), Point2D::ORIGIN);
        assert_eq!(c.radius(), 2.0);
        assert_eq!(c.frame(), Frame2D::WORLD);
    }

    #[test]
    fn test_circle2d_new_negative_radius_errors() {
        assert_eq!(
            Circle2D::new(Point2D::ORIGIN, -1.0),
            Err(CircleConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_circle2d_from_frame_ok() {
        let frame = Frame2D::WORLD;
        let c = Circle2D::from_frame(frame, 3.0).unwrap();
        assert_eq!(c.frame(), frame);
        assert_eq!(c.radius(), 3.0);
    }

    #[test]
    fn test_circle2d_from_frame_negative_radius_errors() {
        assert_eq!(
            Circle2D::from_frame(Frame2D::WORLD, -1.0),
            Err(CircleConstructionError::NegativeRadius)
        );
    }

    // ---- Circle2D evaluation ----

    #[test]
    fn test_circle2d_eval_point_zero() {
        let c = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
        let p = c.eval_point(0.0);
        assert!((p.x - 2.0).abs() < 1e-9);
        assert!(p.y.abs() < 1e-9);
    }

    #[test]
    fn test_circle2d_eval_points_matches_loop() {
        let c = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
        let us = [0.0, 0.5, 1.5];
        let expected: Vec<Point2D> = us.iter().map(|&u| c.eval_point(u)).collect();
        assert_eq!(c.eval_points(&us), expected);
    }

    #[test]
    fn test_circle2d_eval_derivative_order1() {
        let c = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
        let d1 = c.eval_derivative(0.0, 1);
        assert!(d1.x.abs() < 1e-9);
        assert!((d1.y - 2.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic]
    fn test_circle2d_eval_derivative_order0_panics() {
        let c = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
        c.eval_derivative(0.0, 0);
    }

    #[test]
    fn test_circle2d_parameter_of_round_trip() {
        let c = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
        for u in [0.3, 2.0, 5.5] {
            let p = c.eval_point(u);
            assert!((c.parameter_of(p) - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_circle2d_from_frame_arbitrary() {
        let frame = Frame2D::from_x(Point2D::new(1.0, -2.0), Vector2D::new(3.0, 4.0)).unwrap();
        let c = Circle2D::from_frame(frame, 2.5).unwrap();
        for u in [0.3, 2.0, 5.5] {
            let p = c.eval_point(u);
            assert!((c.parameter_of(p) - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_circle3d_contains() {
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let tol = Tolerance::DEFAULT;
        assert!(circle.contains(Point3D::new(2.0, 0.0, 0.0), tol));
        assert!(circle.contains(circle.eval_point(1.0), tol));
        assert!(!circle.contains(Point3D::ORIGIN, tol));
        assert!(!circle.contains(Point3D::new(3.0, 0.0, 0.0), tol));
    }

    #[test]
    fn test_circle2d_contains() {
        let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
        let tol = Tolerance::DEFAULT;
        assert!(circle.contains(Point2D::new(2.0, 0.0), tol));
        assert!(circle.contains(circle.eval_point(1.0), tol));
        assert!(!circle.contains(Point2D::ORIGIN, tol));
        assert!(!circle.contains(Point2D::new(3.0, 0.0), tol));
    }

    #[test]
    fn test_circle3d_project_point() {
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let tol = Tolerance::DEFAULT;
        let on = circle.project_point(Point3D::new(2.0, 0.0, 0.0), tol);
        assert_eq!(on.parameter, 0.0);
        assert_eq!(on.distance, 0.0);
        let off = circle.project_point(Point3D::new(3.0, 0.0, 0.0), tol);
        assert_eq!(off.parameter, 0.0);
        assert_eq!(off.distance, 1.0);
        // Axis point: every circle point is equidistant; distance is exact.
        let axis = circle.project_point(Point3D::new(0.0, 0.0, 5.0), tol);
        assert!((axis.distance - (4.0 + 25.0f64).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn test_circle3d_project_points_batch() {
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let tol = Tolerance::DEFAULT;
        let points = [Point3D::new(2.0, 0.0, 0.0), Point3D::new(0.0, 3.0, 0.0)];
        let projs = circle.project_points(&points, tol);
        assert_eq!(projs.len(), 2);
        assert_eq!(projs[0].distance, 0.0);
        assert_eq!(projs[1].distance, 1.0);
    }

    #[test]
    fn test_circle2d_project_point() {
        let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
        let tol = Tolerance::DEFAULT;
        let proj = circle.project_point(Point2D::new(3.0, 0.0), tol);
        assert_eq!(proj.parameter, 0.0);
        assert_eq!(proj.distance, 1.0);
    }

    #[test]
    fn test_circle2d_intersect_circle() {
        let tol = Tolerance::DEFAULT;
        let c1 = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
        let c2 = Circle2D::new(Point2D::new(3.0, 0.0), 2.0).unwrap();
        match c1.intersect(&c2, tol) {
            CircleCircle2DIntersection::Points(p1, p2) => {
                assert!((p1.x - 1.5).abs() < 1e-9);
                assert!((p1.y - 1.75f64.sqrt()).abs() < 1e-9);
                assert!((p2.y + 1.75f64.sqrt()).abs() < 1e-9);
                assert!(c1.contains(p1, tol));
                assert!(c2.contains(p2, tol));
            }
            _ => panic!("expected two points"),
        }
        // External tangency.
        let tangent = Circle2D::new(Point2D::new(4.0, 0.0), 2.0).unwrap();
        match c1.intersect(&tangent, tol) {
            CircleCircle2DIntersection::Tangent(p) => {
                assert_eq!(p, Point2D::new(2.0, 0.0));
            }
            _ => panic!("expected a tangent"),
        }
        // Separate and coincident.
        let far = Circle2D::new(Point2D::new(5.0, 0.0), 2.0).unwrap();
        assert_eq!(c1.intersect(&far, tol), CircleCircle2DIntersection::Empty);
        assert_eq!(
            c1.intersect(&c1, tol),
            CircleCircle2DIntersection::Coincident
        );
    }

    #[test]
    fn test_circle3d_intersect_circle() {
        let tol = Tolerance::DEFAULT;
        let c1 = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let c2 = Circle3D::new(Point3D::new(3.0, 0.0, 0.0), Vector3D::Z, 2.0).unwrap();
        match c1.intersect(&c2, tol) {
            CircleCircle3DIntersection::Points(p1, p2) => {
                assert!((p1.x - 1.5).abs() < 1e-9);
                assert!((p1.y - 1.75f64.sqrt()).abs() < 1e-9);
                assert!((p2.y + 1.75f64.sqrt()).abs() < 1e-9);
                assert!(c1.contains(p1, tol));
                assert!(c2.contains(p2, tol));
            }
            _ => panic!("expected two points"),
        }
        // Tilted partner: no closed form.
        let tilted = Circle3D::new(Point3D::ORIGIN, Vector3D::X, 2.0).unwrap();
        assert_eq!(
            c1.intersect(&tilted, tol),
            CircleCircle3DIntersection::NotAnalytic
        );
        // Distinct parallel planes: no meeting.
        let lifted = Circle3D::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Z, 2.0).unwrap();
        assert_eq!(
            c1.intersect(&lifted, tol),
            CircleCircle3DIntersection::Empty
        );
        // Coincident.
        assert_eq!(
            c1.intersect(&c1, tol),
            CircleCircle3DIntersection::Coincident
        );
    }

    #[test]
    fn test_circle3d_intersect_circle_in_plane_within_loose_tolerance() {
        // A partner 1e-4 off the plane is coplanar at confusion 1e-3 (#26).
        let loose = Tolerance {
            confusion: 1e-3,
            ..Tolerance::DEFAULT
        };
        let c1 = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
        let c2 = Circle3D::new(Point3D::new(0.5, 0.0, 1e-4), Vector3D::Z, 1.0).unwrap();
        match c1.intersect(&c2, loose) {
            CircleCircle3DIntersection::Points(p1, p2) => {
                for p in [p1, p2] {
                    assert!((p.x - 0.25).abs() < 1e-12);
                    assert!((p.y.abs() - 0.9375f64.sqrt()).abs() < 1e-12);
                    assert!(c1.contains(p, loose) && c2.contains(p, loose));
                }
            }
            other => panic!("expected two points, got {other:?}"),
        }
    }

    #[test]
    fn test_circle3d_intersect_ellipse() {
        let tol = Tolerance::DEFAULT;
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
        // Four transversal hits; each carries consistent parameters.
        match circle.intersect(&ellipse, tol) {
            ConicConicIntersection::Hits(hits) => {
                assert_eq!(hits.len(), 4);
                for h in &hits {
                    assert!(circle.contains(h.point, tol));
                    assert!(ellipse.contains(h.point, tol));
                    assert_eq!(h.first_param, circle.parameter_of(h.point));
                    assert_eq!(h.second_param, ellipse.parameter_of(h.point));
                }
            }
            _ => panic!("expected four points"),
        }
        // Circular ellipse sharing center, radius, and plane.
        let round = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 2.0).unwrap();
        assert_eq!(
            circle.intersect(&round, tol),
            ConicConicIntersection::Coincident
        );
        // Crossing planes: on the line they share, the circle is at
        // y = ±2 and the ellipse at y = ±1.5.
        let tilted = Ellipse3D::new(Point3D::ORIGIN, Vector3D::X, Vector3D::Z, 3.0, 1.5).unwrap();
        assert_eq!(
            circle.intersect(&tilted, tol),
            ConicConicIntersection::Empty
        );
        // Distant ellipse: nothing.
        let far = Ellipse3D::new(
            Point3D::new(10.0, 0.0, 0.0),
            Vector3D::Z,
            Vector3D::X,
            3.0,
            1.5,
        )
        .unwrap();
        assert_eq!(circle.intersect(&far, tol), ConicConicIntersection::Empty);
        // Parallel but offset plane: nothing.
        let lifted = Ellipse3D::new(
            Point3D::new(0.0, 0.0, 1.0),
            Vector3D::Z,
            Vector3D::X,
            3.0,
            1.5,
        )
        .unwrap();
        assert_eq!(
            circle.intersect(&lifted, tol),
            ConicConicIntersection::Empty
        );
    }

    #[test]
    fn test_circle3d_intersect_ellipse_rotated_frames() {
        // The circle's x axis differs from the ellipse's (it used to be
        // solved in one frame and evaluated in the other, losing every hit).
        let tol = Tolerance::DEFAULT;
        let x = Vector3D::new(0.1f64.cos(), 0.1f64.sin(), 0.0);
        let circle =
            Circle3D::from_frame(Frame3D::new(Point3D::ORIGIN, Vector3D::Z, x).unwrap(), 2.0)
                .unwrap();
        let ellipse = Ellipse3D::new(
            Point3D::new(0.5, 0.0, 0.0),
            Vector3D::Z,
            Vector3D::X,
            3.0,
            1.0,
        )
        .unwrap();
        match circle.intersect(&ellipse, tol) {
            ConicConicIntersection::Hits(hits) => {
                assert_eq!(hits.len(), 4);
                for h in &hits {
                    assert!(circle.contains(h.point, tol));
                    assert!(ellipse.contains(h.point, tol));
                    assert!(circle.eval_point(h.first_param).distance(h.point) < 1e-12);
                }
            }
            other => panic!("expected four points, got {other:?}"),
        }
    }

    #[test]
    fn test_circle3d_intersect_parabola() {
        let tol = Tolerance::DEFAULT;
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
        let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
        match circle.intersect(&parabola, tol) {
            ConicConicIntersection::Hits(hits) => {
                assert_eq!(hits.len(), 2);
                for h in &hits {
                    assert!(circle.contains(h.point, tol));
                    assert!(parabola.contains(h.point, tol));
                }
            }
            _ => panic!("expected two points"),
        }
        // Crossing planes: on the line they share, the circle is at
        // y = ±1 and the parabola at its apex.
        let tilted = Parabola3D::new(Point3D::ORIGIN, Vector3D::X, Vector3D::Z, 1.0).unwrap();
        assert_eq!(
            circle.intersect(&tilted, tol),
            ConicConicIntersection::Empty
        );
    }

    #[test]
    fn test_circle3d_intersect_hyperbola() {
        let tol = Tolerance::DEFAULT;
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 3.0).unwrap();
        let hyperbola =
            Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
        // Mirror-branch roots are dropped: two right-branch hits.
        match circle.intersect(&hyperbola, tol) {
            ConicConicIntersection::Hits(hits) => {
                assert_eq!(hits.len(), 2);
                for h in &hits {
                    assert!(circle.contains(h.point, tol));
                    assert!(hyperbola.contains(h.point, tol));
                    assert!(h.point.x > 0.0);
                }
            }
            _ => panic!("expected two points"),
        }
        // The osculating circle at the right vertex (radius b²/a): a lone
        // four-point contact.
        let grazing = Circle3D::new(Point3D::new(2.5, 0.0, 0.0), Vector3D::Z, 0.5).unwrap();
        match grazing.intersect(&hyperbola, tol) {
            ConicConicIntersection::Hits(hits) => {
                assert_eq!(hits.len(), 1);
                assert_eq!(hits[0].multiplicity, 4);
                assert!(hits[0].point.distance(Point3D::new(2.0, 0.0, 0.0)) < 1e-9);
            }
            other => panic!("expected a tangent, got {other:?}"),
        }
    }

    #[test]
    fn test_circle3d_intersect_plane() {
        use crate::Plane;
        let tol = Tolerance::DEFAULT;
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        // Transversal plane through the x-axis: hits at (±2, 0, 0).
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        match circle.intersect(&plane, tol) {
            CircleSurfaceIntersection::Hits(hits) => {
                assert_eq!(hits.len(), 2);
                let mut pts: Vec<Point3D> = hits.iter().map(|h| h.point).collect();
                pts.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap());
                assert!(pts[0].distance(Point3D::new(0.0, -2.0, 0.0)) < 1e-9);
                assert!(pts[1].distance(Point3D::new(0.0, 2.0, 0.0)) < 1e-9);
                for h in &hits {
                    assert!(circle.contains(h.point, tol));
                    assert!(plane.contains(h.point, tol));
                }
            }
            _ => panic!("expected hits"),
        }
        // Coplanar: the whole circle.
        let flat = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        match circle.intersect(&flat, tol) {
            CircleSurfaceIntersection::Circle(_) => {}
            _ => panic!("expected the whole circle"),
        }
        // Parallel offset: miss.
        let miss = Plane::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Z).unwrap();
        assert_eq!(
            circle.intersect(&miss, tol),
            CircleSurfaceIntersection::Empty
        );
    }

    #[test]
    fn test_circle3d_intersect_sphere_cylinder_cone() {
        use crate::{Cone, Cylinder, Frame3D, Sphere};
        let tol = Tolerance::DEFAULT;
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        // Small off-center sphere: two hits, cross-checked both sides.
        let sphere = Sphere::new(Point3D::new(1.0, 0.0, 0.0), 2.0).unwrap();
        match circle.intersect(&sphere, tol) {
            CircleSurfaceIntersection::Hits(hits) => {
                assert_eq!(hits.len(), 2);
                for h in &hits {
                    assert!(circle.contains(h.point, tol));
                    assert!(sphere.contains(h.point, tol));
                }
            }
            _ => panic!("expected hits"),
        }
        // Coaxial cylinder wider than the circle: miss.
        let fat = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 3.0).unwrap();
        assert_eq!(
            circle.intersect(&fat, tol),
            CircleSurfaceIntersection::Empty
        );
        // Tilted cylinder through the ring: hits on both.
        let tilted = Cylinder::new(
            Point3D::ORIGIN,
            Vector3D::new(0.0, 1.0, 1.0).normalized().unwrap(),
            2.0,
        )
        .unwrap();
        match circle.intersect(&tilted, tol) {
            CircleSurfaceIntersection::Hits(hits) => {
                assert!(!hits.is_empty());
                for h in &hits {
                    assert!(circle.contains(h.point, tol));
                    assert!(tilted.contains(h.point, tol));
                }
            }
            _ => panic!("expected hits"),
        }
        // Reference cone contains the ring by construction.
        let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
        match circle.intersect(&cone, tol) {
            CircleSurfaceIntersection::Circle(_) => {}
            _ => panic!("expected the whole circle"),
        }
    }

    /// The circumcircle of an equilateral triangle inscribed in `ellipse`
    /// with a corner on its major vertex, which the circle puts at its
    /// parameter 0; the other corners fall at 2π/3 and 4π/3. The circle
    /// touches the ellipse at the vertex and crosses it at the other two
    /// corners, and nowhere else.
    fn circle_through_inscribed_triangle(ellipse: &Ellipse3D) -> Circle3D {
        let (a, b) = (ellipse.major_radius(), ellipse.minor_radius());
        // Corners (a, 0) and (x, ±y) with (a - x)² = 3y² on the ellipse:
        // x is the quadratic's root other than a.
        let k = 1.0 / (3.0 * b * b);
        let x = (a * a * k - 1.0) / (1.0 / (a * a) + k) / a;
        let side = 2.0 * (a - x) / 3f64.sqrt();
        let radius = side / 3f64.sqrt();
        let frame = ellipse.frame();
        let center = frame.origin() + frame.x_direction() * (a - radius);
        let frame = Frame3D::new(center, frame.z_direction(), frame.x_direction()).unwrap();
        Circle3D::from_frame(frame, radius).unwrap()
    }

    /// Asserts `circle` meets `surface` at the three triangle corners
    /// only, not along its whole length.
    fn assert_three_corner_hits(
        circle: &Circle3D,
        surface: &impl crate::ParametricSurface,
        result: CircleSurfaceIntersection,
    ) {
        let tol = Tolerance::DEFAULT;
        for t in [0.0, 2.0 * PI / 3.0, 4.0 * PI / 3.0] {
            assert!(surface.contains(circle.eval_point(t), tol));
        }
        assert!(!surface.contains(circle.eval_point(1.0), tol));
        let CircleSurfaceIntersection::Hits(hits) = result else {
            panic!("expected the three corners, got {result:?}");
        };
        assert_eq!(hits.len(), 3, "{hits:?}");
        for h in &hits {
            assert!(surface.contains(h.point, tol));
        }
    }

    #[test]
    fn test_circle3d_on_three_samples_is_not_on_cylinder_or_cone() {
        use crate::{Cone, Cylinder, Plane, PlaneConeIntersection, PlaneCylinderIntersection};
        let tol = Tolerance::DEFAULT;
        let plane = Plane::new(Point3D::new(0.0, 0.0, 0.5), Vector3D::Z).unwrap();
        let tilt = Vector3D::new(1.0, 0.0, 1.0).normalized().unwrap();
        let cylinder = Cylinder::new(Point3D::ORIGIN, tilt, 1.0).unwrap();
        let PlaneCylinderIntersection::Ellipse(section) = plane.intersect(&cylinder, tol) else {
            panic!("expected an ellipse");
        };
        let circle = circle_through_inscribed_triangle(&section);
        assert_three_corner_hits(&circle, &cylinder, circle.intersect(&cylinder, tol));

        let axis = Vector3D::new(0.3, 0.0, 1.0);
        let cone = Cone::new(Point3D::new(0.0, 0.0, -1.0), axis, 0.4, 0.6).unwrap();
        let PlaneConeIntersection::Ellipse(section) = plane.intersect(&cone, tol) else {
            panic!("expected an ellipse");
        };
        let circle = circle_through_inscribed_triangle(&section);
        assert_three_corner_hits(&circle, &cone, circle.intersect(&cone, tol));
    }

    /// The circle of `radius` touching the planar `curve` at parameter `t`,
    /// on the side `(T.y, -T.x)` of the tangent `T` in the XY plane: outside
    /// an ellipse, inside a parabola or hyperbola branch.
    fn circle_touching(curve: &impl crate::ParametricCurve3D, t: f64, radius: f64) -> Circle3D {
        let tangent = curve.eval_derivative(t, 1);
        let side = Vector3D::new(tangent.y, -tangent.x, 0.0)
            .normalized()
            .unwrap();
        Circle3D::new(curve.eval_point(t) + side * radius, Vector3D::Z, radius).unwrap()
    }

    /// Contacts near `s = π` on the circle, where `tan(s/2)` is large.
    #[test]
    fn test_circle3d_conic_tangency_near_s_pi() {
        let tol = Tolerance::DEFAULT;
        let (o, z, x) = (Point3D::ORIGIN, Vector3D::Z, Vector3D::X);
        let ellipse = Ellipse3D::new(o, z, x, 2.0, 1.0).unwrap();
        let parabola = Parabola3D::new(o, z, x, 1.0).unwrap();
        let hyperbola = Hyperbola3D::new(o, z, x, 1.0, 1.0).unwrap();
        for i in -30..=30 {
            let t = f64::from(i) * 0.01;
            // Radii below the conic's smallest radius of curvature, so the
            // circle touches it at one point only.
            let cases = [
                (
                    "ellipse",
                    circle_touching(&ellipse, t, 1.0),
                    ellipse.eval_point(t),
                ),
                (
                    "parabola",
                    circle_touching(&parabola, t, 1.5),
                    parabola.eval_point(t),
                ),
                (
                    "hyperbola",
                    circle_touching(&hyperbola, t, 0.4),
                    hyperbola.eval_point(t),
                ),
            ];
            for (name, circle, contact) in cases {
                let result = match name {
                    "ellipse" => circle.intersect(&ellipse, tol),
                    "parabola" => circle.intersect(&parabola, tol),
                    _ => circle.intersect(&hyperbola, tol),
                };
                let ConicConicIntersection::Hits(hits) = &result else {
                    panic!("{name} at t = {t}: expected one tangent hit, got {result:?}");
                };
                assert!(
                    hits.len() == 1 && hits[0].multiplicity == 2,
                    "{name} at t = {t}: expected one tangent hit, got {result:?}"
                );
                let hit = hits[0];
                assert!(
                    hit.point.distance(contact) < 1e-6,
                    "{name} at t = {t}: {hit:?}"
                );
            }
        }
    }

    #[test]
    fn test_circle3d_cylinder_tangency_near_s_pi() {
        use crate::Cylinder;
        let tol = Tolerance::DEFAULT;
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
        for i in -30..=30 {
            let phi = f64::from(i) * 0.01;
            let center = Point3D::new(1.5 * phi.cos(), 1.5 * phi.sin(), 0.0);
            let circle = Circle3D::new(center, Vector3D::Z, 0.5).unwrap();
            let result = circle.intersect(&cylinder, tol);
            let CircleSurfaceIntersection::Hits(hits) = &result else {
                panic!("at {phi}: expected one tangent hit, got {result:?}");
            };
            assert_eq!(hits.len(), 1, "at {phi}: {hits:?}");
            assert_eq!(hits[0].multiplicity, 2, "at {phi}: {hits:?}");
            let contact = Point3D::new(phi.cos(), phi.sin(), 0.0);
            assert!(hits[0].point.distance(contact) < 1e-6, "at {phi}: {hits:?}");
        }
    }

    /// Exact external tangencies in random poses: every one is a single
    /// tangent hit, wherever the contact falls on the circle.
    #[test]
    fn test_circle3d_ellipse_random_external_tangency() {
        let tol = Tolerance::DEFAULT;
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for case in 0..500 {
            let normal = Vector3D::new(next() - 0.5, next() - 0.5, next() - 0.5);
            let hint = Vector3D::new(next() - 0.5, next() - 0.5, next() - 0.5);
            let frame = Frame3D::new(
                Point3D::new(4.0 * next() - 2.0, 4.0 * next() - 2.0, 4.0 * next() - 2.0),
                normal,
                hint,
            )
            .unwrap();
            let a = 0.5 + 3.0 * next();
            let b = a * (0.2 + 0.8 * next());
            let ellipse = Ellipse3D::from_frame(frame, a, b).unwrap();
            let t = std::f64::consts::TAU * next();
            let radius = 0.1 + 3.0 * next();
            // The outward normal in the ellipse's plane.
            let tangent = ellipse.eval_derivative(t, 1);
            let outward = tangent.cross(frame.z_direction()).normalized().unwrap();
            let contact = ellipse.eval_point(t);
            let circle =
                Circle3D::new(contact + outward * radius, frame.z_direction(), radius).unwrap();
            let result = circle.intersect(&ellipse, tol);
            let ConicConicIntersection::Hits(hits) = &result else {
                panic!("case {case}: expected one tangent hit, got {result:?}");
            };
            assert!(
                hits.len() == 1 && hits[0].multiplicity == 2,
                "case {case}: expected one tangent hit, got {result:?}"
            );
            let hit = hits[0];
            assert!(hit.point.distance(contact) < 1e-6, "case {case}: {hit:?}");
        }
    }

    // ---- Circle3D transformed ----

    #[test]
    fn test_circle3d_transformed_translation_spot_check() {
        let circle = Circle3D::new(Point3D::new(1.0, 0.0, 0.0), Vector3D::Z, 2.0).unwrap();
        let t = Transform::translation(Vector3D::new(0.0, 0.0, 5.0));
        let moved = circle.transformed(t).unwrap();
        assert_eq!(moved.center(), Point3D::new(1.0, 0.0, 5.0));
        assert_eq!(moved.radius(), 2.0);
        assert_eq!(moved.normal(), Vector3D::Z);
        // Hand-computed evaluation at u = 0.
        assert_eq!(moved.eval_point(0.0), Point3D::new(3.0, 0.0, 5.0));
    }

    #[test]
    fn test_circle3d_transformed_scale_and_round_trip() {
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let t = Transform::scaling(Point3D::new(1.0, 0.0, 0.0), 2.0);
        let moved = circle.transformed(t).unwrap();
        assert_eq!(moved.center(), Point3D::new(-1.0, 0.0, 0.0));
        assert_eq!(moved.radius(), 4.0);
        // Transformed points stay on the transformed circle.
        let tol = Tolerance::DEFAULT;
        for u in [0.0, 1.0, 2.5] {
            assert!(moved.contains(t.apply_point(circle.eval_point(u)), tol));
        }
        // Round trip through the inverse scale restores the circle.
        let back = moved
            .transformed(Transform::scaling(Point3D::new(1.0, 0.0, 0.0), 0.5))
            .unwrap();
        assert!(back.center().distance(circle.center()) < 1e-9);
        assert!((back.radius() - circle.radius()).abs() < 1e-9);
    }

    #[test]
    fn test_circle3d_transformed_mirror_commutes_with_evaluation() {
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let m = Transform::mirror_plane(Frame3D::WORLD);
        let moved = circle.transformed(m).unwrap();
        assert_eq!(moved.center(), Point3D::ORIGIN);
        assert_eq!(moved.radius(), 2.0);
        // Mirrored across its own plane, every point stays put: the
        // parametrization and the normal are unchanged, only the frame's
        // z direction flips.
        assert_eq!(moved.normal(), Vector3D::Z);
        assert!(!moved.frame().is_direct());
        let tol = Tolerance::DEFAULT;
        for u in [0.0, 0.7, 2.1] {
            let expected = m.apply_point(circle.eval_point(u));
            assert!(moved.contains(expected, tol));
            assert!(moved.eval_point(u).distance(expected) < 1e-12);
            assert!((moved.parameter_of(expected) - u).abs() < 1e-12);
        }
        // Mirroring twice is the identity.
        assert_eq!(moved.transformed(m).unwrap(), circle);
    }

    #[test]
    fn test_circle3d_transformed_zero_scale_errors() {
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        assert_eq!(
            circle.transformed(Transform::scaling(Point3D::ORIGIN, 0.0)),
            Err(CircleConstructionError::NullNormal)
        );
    }
}
