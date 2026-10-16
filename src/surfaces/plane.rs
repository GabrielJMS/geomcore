//! Planes in 3D: parametric evaluation, parameter inversion, and
//! construction from points or an implicit equation, thin wrappers over
//! [`crate::surface_math::analytic`].

use crate::Intersect;
use crate::frame::frame_error;
use crate::projection::{self, SurfaceProjection};
use crate::surface_math::analytic;
use crate::surfaces::iso::{SurfaceIsoError, check_iso_param, iso_line};
use crate::tol;
use crate::{
    Circle3D, Cone, Cylinder, Ellipse3D, Frame3D, Hyperbola3D, Line3D, Parabola3D,
    PlaneConeIntersection, PlaneCylinderIntersection, PlanePlaneIntersection,
    PlaneSphereIntersection, Point3D, Sphere, Tolerance, Transform, Vector3D,
};
use std::fmt;

/// Error returned when a [`Plane`] cannot be constructed from the given
/// inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneConstructionError {
    /// The normal has zero length.
    NullNormal,
    /// Two (or more) of the points given to build the plane are coincident
    /// (or too close to distinguish).
    ConfusedPoints,
    /// The three points given to build the plane are collinear, so no
    /// normal can be derived from them.
    CollinearPoints,
    /// The equation `ax + by + cz + d = 0` is degenerate: `a`, `b`, and `c`
    /// are all (numerically) zero, so no normal can be derived from it.
    BadEquation,
    /// An input coordinate, radius, or angle is NaN or infinite.
    NonFinite,
}

impl fmt::Display for PlaneConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            PlaneConstructionError::NullNormal => "normal has zero length",
            PlaneConstructionError::ConfusedPoints => "the points are confused",
            PlaneConstructionError::CollinearPoints => "the points are collinear",
            PlaneConstructionError::BadEquation => "the equation is degenerate",
            PlaneConstructionError::NonFinite => "an input is NaN or infinite",
        };
        f.write_str(message)
    }
}

impl std::error::Error for PlaneConstructionError {}

/// A plane in 3D: a [`Frame3D`] (origin plus local x/y directions spanning
/// the plane), evaluated as `origin + u*x_dir + v*y_dir`.
///
/// # Examples
///
/// ```
/// use geomcore::{Plane, Point3D, Vector3D};
/// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
/// assert_eq!(plane.eval_point(2.0, 3.0), Point3D::new(2.0, 3.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    frame: Frame3D,
}

impl Plane {
    /// Creates a plane from a point and a normal.
    ///
    /// The plane frame is derived from `normal` via [`Frame3D::from_z`].
    ///
    /// # Errors
    ///
    /// Returns [`PlaneConstructionError::NullNormal`] if `normal` cannot be
    /// normalized (zero length).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Plane, Point3D, Vector3D};
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// assert_eq!(plane.normal(), Vector3D::Z);
    /// ```
    pub fn new(point: Point3D, normal: Vector3D) -> Result<Plane, PlaneConstructionError> {
        let frame = Frame3D::from_z(point, normal).map_err(frame_error(
            PlaneConstructionError::NonFinite,
            PlaneConstructionError::NullNormal,
        ))?;
        Ok(Plane::from_frame(frame))
    }

    /// Creates a plane from a frame directly. Infallible: any frame spans a
    /// valid plane.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Frame3D, Plane};
    /// let plane = Plane::from_frame(Frame3D::WORLD);
    /// assert_eq!(plane.frame(), Frame3D::WORLD);
    /// ```
    pub fn from_frame(frame: Frame3D) -> Plane {
        Plane { frame }
    }

    /// Creates a plane through three points.
    ///
    /// Fails with [`PlaneConstructionError::ConfusedPoints`] if `p1` and
    /// `p2` are coincident (or too close to distinguish); otherwise, if the
    /// three points are collinear (`|(p2-p1) x (p3-p1)| <= tol::CONFUSION *
    /// max(|p2-p1|, |p3-p1|)`), returns
    /// [`PlaneConstructionError::CollinearPoints`].
    ///
    /// The resulting frame's origin is `p1`, its normal is
    /// `normalize((p2-p1) x (p3-p1))`, and its x direction is
    /// `normalize(p2-p1)`.
    ///
    /// # Errors
    ///
    /// Returns [`PlaneConstructionError::ConfusedPoints`] or
    /// [`PlaneConstructionError::CollinearPoints`] as described above.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Plane, Point3D, Vector3D};
    /// let plane = Plane::from_three_points(
    ///     Point3D::ORIGIN,
    ///     Point3D::new(1.0, 0.0, 0.0),
    ///     Point3D::new(0.0, 1.0, 0.0),
    /// )
    /// .unwrap();
    /// assert_eq!(plane.normal(), Vector3D::Z);
    /// ```
    pub fn from_three_points(
        p1: Point3D,
        p2: Point3D,
        p3: Point3D,
    ) -> Result<Plane, PlaneConstructionError> {
        if p1.distance(p2) < tol::CONFUSION {
            return Err(PlaneConstructionError::ConfusedPoints);
        }

        let v12 = p2 - p1;
        let v13 = p3 - p1;
        let normal = v12.cross(v13);
        if normal.magnitude() <= tol::CONFUSION * v12.magnitude().max(v13.magnitude()) {
            return Err(PlaneConstructionError::CollinearPoints);
        }

        let frame = Frame3D::new(p1, normal, v12).map_err(frame_error(
            PlaneConstructionError::NonFinite,
            PlaneConstructionError::CollinearPoints,
        ))?;
        Ok(Plane::from_frame(frame))
    }

    /// Creates a plane from the implicit equation `ax + by + cz + d = 0`.
    ///
    /// Fails with [`PlaneConstructionError::BadEquation`] if
    /// `a*a + b*b + c*c <= f64::MIN_POSITIVE` (the equation has no
    /// well-defined normal).
    ///
    /// The plane's normal is `normalize((a, b, c))`. Its location is *not*
    /// the point closest to the world origin; it matches the reference
    /// implementation's axis-intercept convention: `(0, 0, -d/c)` if `c` is
    /// the largest-magnitude coefficient, `(0, -d/b, 0)` if `b` is, or
    /// `(-d/a, 0, 0)` if `a` is (picking whichever axis the equation is
    /// solved against avoids dividing by a near-zero coefficient). The x
    /// direction is an arbitrary vector perpendicular to the normal, chosen
    /// so that zeroing the smallest-magnitude component of the normal and
    /// swapping the other two (with a sign matching the golden fixture)
    /// gives a perpendicular vector; y is `normal x x`.
    ///
    /// # Errors
    ///
    /// Returns [`PlaneConstructionError::BadEquation`] as described above.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Plane;
    /// let plane = Plane::from_coefficients(0.0, 0.0, 1.0, -2.0).unwrap();
    /// assert_eq!(plane.frame().origin(), geomcore::Point3D::new(0.0, 0.0, 2.0));
    /// ```
    pub fn from_coefficients(
        a: f64,
        b: f64,
        c: f64,
        d: f64,
    ) -> Result<Plane, PlaneConstructionError> {
        if !(a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite()) {
            return Err(PlaneConstructionError::NonFinite);
        }
        if a * a + b * b + c * c <= f64::MIN_POSITIVE {
            return Err(PlaneConstructionError::BadEquation);
        }

        let (aa, ab, ac) = (a.abs(), b.abs(), c.abs());
        let origin = if ac >= aa && ac >= ab {
            Point3D::new(0.0, 0.0, -d / c)
        } else if ab >= aa && ab >= ac {
            Point3D::new(0.0, -d / b, 0.0)
        } else {
            Point3D::new(-d / a, 0.0, 0.0)
        };
        if !origin.is_finite() {
            // Finite coefficients whose plane lies beyond f64 range.
            return Err(PlaneConstructionError::BadEquation);
        }

        let normal = Vector3D::new(a, b, c);
        let frame = arbitrary_perpendicular_frame(origin, normal);
        Ok(Plane::from_frame(frame))
    }

    /// Returns the plane's frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Frame3D, Plane};
    /// let plane = Plane::from_frame(Frame3D::WORLD);
    /// assert_eq!(plane.frame(), Frame3D::WORLD);
    /// ```
    pub fn frame(&self) -> Frame3D {
        self.frame
    }

    /// Returns the unit normal of the plane: the frame's z direction on
    /// a direct frame, its negation on an indirect one (see
    /// [`Frame3D::is_direct`]), so the normal follows the orientation
    /// and reverses with [`reversed_u`](Self::reversed_u) or
    /// [`reversed_v`](Self::reversed_v).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Plane, Point3D, Vector3D};
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// assert_eq!(plane.normal(), Vector3D::Z);
    /// assert_eq!(plane.reversed_u().normal(), -Vector3D::Z);
    /// ```
    pub fn normal(&self) -> Vector3D {
        if self.frame.is_direct() {
            self.frame.z_direction()
        } else {
            -self.frame.z_direction()
        }
    }

    /// Returns the plane running in the opposite `u` direction: the same
    /// point set with the frame's x direction negated, so the frame turns
    /// indirect (see [`Frame3D::is_direct`]) and the normal reverses.
    /// `reversed_u().eval_point(u, v)` is `eval_point(-u, v)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Plane, Point3D, Vector3D};
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let reversed = plane.reversed_u();
    /// assert!(!reversed.frame().is_direct());
    /// assert_eq!(reversed.eval_point(1.0, 2.0), plane.eval_point(-1.0, 2.0));
    /// assert_eq!(reversed.normal(), -Vector3D::Z);
    /// ```
    pub fn reversed_u(&self) -> Plane {
        Plane {
            frame: self.frame.flipped_x(),
        }
    }

    /// Returns the plane running in the opposite `v` direction: the same
    /// point set with the frame's y direction negated.
    /// `reversed_v().eval_point(u, v)` is `eval_point(u, -v)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Plane, Point3D, Vector3D};
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let reversed = plane.reversed_v();
    /// assert_eq!(reversed.eval_point(1.0, 2.0), plane.eval_point(1.0, -2.0));
    /// ```
    pub fn reversed_v(&self) -> Plane {
        Plane {
            frame: self.frame.flipped_y(),
        }
    }

    /// Maps `u` onto the [`reversed_u`](Self::reversed_u) plane: the
    /// parameter of the same point there.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Plane, Point3D, Vector3D};
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let u = plane.u_reversed_parameter(1.0);
    /// let there = plane.reversed_u().eval_point(u, 0.5);
    /// assert!(there.distance(plane.eval_point(1.0, 0.5)) < 1e-12);
    /// ```
    pub fn u_reversed_parameter(&self, u: f64) -> f64 {
        -u
    }

    /// Maps `v` onto the [`reversed_v`](Self::reversed_v) plane.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Plane, Point3D, Vector3D};
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let v = plane.v_reversed_parameter(0.5);
    /// let there = plane.reversed_v().eval_point(1.0, v);
    /// assert!(there.distance(plane.eval_point(1.0, 0.5)) < 1e-12);
    /// ```
    pub fn v_reversed_parameter(&self, v: f64) -> f64 {
        -v
    }

    /// The `u` isoparametric line at `u`: `u` fixed, running along `v`.
    /// A line through `origin + u * x_direction` in the `y` direction,
    /// so `u_iso(u).eval_point(t)` is `eval_point(u, t)` for every `t`.
    ///
    /// Fails with [`SurfaceIsoError::NonFinite`] for a NaN or infinite
    /// `u`; any finite value is accepted.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Plane, Point3D, Vector3D};
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let iso = plane.u_iso(2.0).unwrap();
    /// assert_eq!(iso.eval_point(3.0), plane.eval_point(2.0, 3.0));
    /// ```
    pub fn u_iso(&self, u: f64) -> Result<Line3D, SurfaceIsoError> {
        let u = check_iso_param(u, None)?;
        iso_line(self.eval_point(u, 0.0), self.frame.y_direction())
    }

    /// The `v` isoparametric line at `v`: `v` fixed, running along `u`.
    /// A line through `origin + v * y_direction` in the `x` direction,
    /// so `v_iso(v).eval_point(t)` is `eval_point(t, v)` for every `t`.
    ///
    /// Fails with [`SurfaceIsoError::NonFinite`] for a NaN or infinite
    /// `v`; any finite value is accepted.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Plane, Point3D, Vector3D};
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let iso = plane.v_iso(2.0).unwrap();
    /// assert_eq!(iso.eval_point(3.0), plane.eval_point(3.0, 2.0));
    /// ```
    pub fn v_iso(&self, v: f64) -> Result<Line3D, SurfaceIsoError> {
        let v = check_iso_param(v, None)?;
        iso_line(self.eval_point(0.0, v), self.frame.x_direction())
    }

    /// Returns the image of this plane under `transform`: the frame origin
    /// and directions map point-wise.
    ///
    /// Every frame axis maps through the transform, so the image's
    /// parametrization is the transform of this one: under a rotation,
    /// translation or mirror, `transformed(t).eval_point(u, v)` equals
    /// `t.apply_point(self.eval_point(u, v))`. A mirror changes the
    /// frame's handedness, so the image's normal is the opposite of
    /// the mirrored normal (as for any map that reverses orientation);
    /// call `reversed_u` or `reversed_v` on the image to keep the
    /// mirrored side. Fails with
    /// [`PlaneConstructionError::NullNormal`] only when the transform
    /// collapses the frame (a zero scale factor). `u` and `v` are
    /// lengths, so under a scale `s` the image of the point at `(u, v)`
    /// sits at `(s * u, s * v)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Plane, Point3D, Transform, Vector3D};
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let moved = plane.transformed(Transform::translation(Vector3D::new(0.0, 0.0, 2.0))).unwrap();
    /// assert_eq!(moved.eval_point(1.0, 1.0), Point3D::new(1.0, 1.0, 2.0));
    /// ```
    pub fn transformed(&self, transform: Transform) -> Result<Plane, PlaneConstructionError> {
        let (frame, _) = transform.apply_frame(self.frame()).map_err(frame_error(
            PlaneConstructionError::NonFinite,
            PlaneConstructionError::NullNormal,
        ))?;
        Ok(Plane::from_frame(frame))
    }

    /// Evaluates the point on the plane at `(u, v)`:
    /// `origin + u*x_dir + v*y_dir`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Plane, Point3D, Vector3D};
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// assert_eq!(plane.eval_point(2.0, 3.0), Point3D::new(2.0, 3.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64, v: f64) -> Point3D {
        analytic::plane_d0(&self.frame, u, v)
    }

    /// Evaluates the points on the plane at each `(u, v)` in `uvs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Plane, Point3D, Vector3D};
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let points = plane.eval_points(&[(1.0, 0.0), (0.0, 1.0)]);
    /// assert_eq!(points[0], Point3D::new(1.0, 0.0, 0.0));
    /// ```
    pub fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3D> {
        uvs.iter().map(|&(u, v)| self.eval_point(u, v)).collect()
    }

    /// Evaluates the derivative of order `(du, dv)` at `(u, v)`.
    ///
    /// `Su = x_dir`, `Sv = y_dir`; every second derivative is identically
    /// zero (the plane is linear in `u` and `v`).
    ///
    /// # Panics
    ///
    /// Panics if `du + dv == 0` (use [`Plane::eval_point`] for the position
    /// itself) or if `du + dv > 2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Plane, Point3D, Vector3D};
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// assert_eq!(plane.eval_derivative(0.0, 0.0, 1, 0), Vector3D::X);
    /// ```
    pub fn eval_derivative(&self, _u: f64, _v: f64, du: u32, dv: u32) -> Vector3D {
        match du + dv {
            0 => panic!(
                "eval_derivative: du + dv must be >= 1 (use eval_point for the (0, 0) order)"
            ),
            1..=2 => analytic::plane_derivative(&self.frame, du, dv),
            _ => panic!(
                "eval_derivative: order du={du}, dv={dv} is not supported (du + dv must be <= 2)"
            ),
        }
    }

    /// Recovers `(u, v)` of a point on (or near) the plane: its local x/y
    /// coordinates in the plane's frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Plane, Point3D, Vector3D};
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// assert_eq!(plane.parameters_of(Point3D::new(2.0, 3.0, 0.0)), (2.0, 3.0));
    /// ```
    pub fn parameters_of(&self, point: Point3D) -> (f64, f64) {
        analytic::plane_parameters(&self.frame, point)
    }

    /// Returns whether `point` lies on the plane: the inverse parameters
    /// are recovered with [`Plane::parameters_of`] and re-evaluated, and
    /// the point counts as contained when the re-evaluated point is within
    /// `tol.confusion` of it.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Plane, Point3D, Tolerance, Vector3D};
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// assert!(plane.contains(Point3D::ORIGIN, tol));
    /// assert!(plane.contains(Point3D::new(1.0, 2.0, 0.0), tol));
    /// assert!(!plane.contains(Point3D::new(0.0, 0.0, 1.0), tol));
    /// ```
    pub fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        let (u, v) = self.parameters_of(point);
        self.eval_point(u, v).distance(point) <= tol.confusion
    }

    /// Projects `point` onto the plane, returning the `(u, v)` parameters
    /// of the closest point and its distance. Distances within
    /// `tol.confusion` snap to `0.0`, matching [`Plane::contains`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Plane, Point3D, Tolerance, Vector3D};
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let proj = plane.project_point(Point3D::new(1.0, 2.0, 3.0), Tolerance::DEFAULT);
    /// assert_eq!((proj.u, proj.v), (1.0, 2.0));
    /// assert_eq!(proj.distance, 3.0);
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

    /// Projects each point in `points` onto the plane.
    ///
    /// Default-style batch wrapper over [`Plane::project_point`]: one
    /// native call per batch, mirroring [`Plane::eval_points`].
    pub fn project_points(&self, points: &[Point3D], tol: Tolerance) -> Vec<SurfaceProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }

    /// Intersects with a [`Plane`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_plane(&self, other: &Plane, tol: Tolerance) -> PlanePlaneIntersection {
        self.intersect(other, tol)
    }

    /// Intersects with a [`Sphere`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_sphere(&self, sphere: &Sphere, tol: Tolerance) -> PlaneSphereIntersection {
        self.intersect(sphere, tol)
    }

    /// Intersects with a [`Cylinder`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_cylinder(
        &self,
        cylinder: &Cylinder,
        tol: Tolerance,
    ) -> PlaneCylinderIntersection {
        self.intersect(cylinder, tol)
    }

    /// Intersects with a [`Cone`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_cone(&self, cone: &Cone, tol: Tolerance) -> PlaneConeIntersection {
        self.intersect(cone, tol)
    }
}

/// Builds a frame at `origin` with `z_dir = normalize(normal)` and an
/// arbitrary x direction perpendicular to it, matching the reference
/// implementation's convention for planes derived from an implicit
/// equation: zero the smallest-magnitude component of the normal and swap
/// the other two (with a sign that reproduces the golden fixture).
///
/// `normal` must be non-zero (checked by the caller).
fn arbitrary_perpendicular_frame(origin: Point3D, normal: Vector3D) -> Frame3D {
    let z = normal.normalized().expect("normal is non-zero (checked)");
    let (ax, ay, az) = (z.x.abs(), z.y.abs(), z.z.abs());
    let x_dir = if ax <= ay && ax <= az {
        Vector3D::new(0.0, z.z, -z.y)
    } else if ay <= ax && ay <= az {
        Vector3D::new(z.z, 0.0, -z.x)
    } else {
        Vector3D::new(z.y, -z.x, 0.0)
    }
    .normalized()
    .expect("z has unit length and is nonzero on at least two axes, so the swap is nonzero");
    Frame3D::new(origin, z, x_dir).expect("x_dir constructed perpendicular to z by design")
}

impl Intersect<Plane> for Plane {
    type Output = PlanePlaneIntersection;

    /// Intersects this plane with another plane.
    ///
    /// Non-parallel planes meet in a [`Line3D`]: the direction is the
    /// normalized cross product of the unit normals, and a point on the
    /// line is found by solving the two plane equations in the coordinate
    /// plane of the direction's largest component (whose 2x2 determinant
    /// is that component, hence nonzero). Parallel planes report
    /// [`PlanePlaneIntersection::Parallel`], or
    /// [`PlanePlaneIntersection::Coincident`] when an origin point of one
    /// lies on the other within `tol`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{Plane, PlanePlaneIntersection, Point3D, Tolerance, Vector3D};
    /// let xy = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let zy = Plane::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match xy.intersect(&zy, tol) {
    ///     PlanePlaneIntersection::Line(line) => {
    ///         assert_eq!(line.origin(), Point3D::ORIGIN);
    ///         assert_eq!(line.direction(), Vector3D::Y);
    ///     }
    ///     _ => panic!("expected a line"),
    /// }
    /// ```
    fn intersect(&self, other: &Plane, tol: Tolerance) -> PlanePlaneIntersection {
        let n1 = self.normal();
        let n2 = other.normal();
        let d = n1.cross(n2);
        let sin = d.magnitude();
        if sin <= tol.angular {
            if other.contains(self.frame.origin(), tol) {
                PlanePlaneIntersection::Coincident
            } else {
                PlanePlaneIntersection::Parallel
            }
        } else {
            let dir = d * (1.0 / sin);
            let o1 = self.frame.origin();
            let o2 = other.frame.origin();
            let e1 = n1.x * o1.x + n1.y * o1.y + n1.z * o1.z;
            let e2 = n2.x * o2.x + n2.y * o2.y + n2.z * o2.z;
            let a = [n1.x, n1.y, n1.z];
            let b = [n2.x, n2.y, n2.z];
            let ax = [d.x.abs(), d.y.abs(), d.z.abs()];
            let k = if ax[0] >= ax[1] && ax[0] >= ax[2] {
                0
            } else if ax[1] >= ax[2] {
                1
            } else {
                2
            };
            let (i, j) = match k {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            // det == +-d[k], nonzero by the parallel guard above.
            let det = a[i] * b[j] - a[j] * b[i];
            let xi = (e1 * b[j] - e2 * a[j]) / det;
            let xj = (a[i] * e2 - b[i] * e1) / det;
            let mut p = [0.0, 0.0, 0.0];
            p[i] = xi;
            p[j] = xj;
            let line = Line3D::new(Point3D::new(p[0], p[1], p[2]), dir)
                .expect("intersection direction is nonzero by construction");
            PlanePlaneIntersection::Line(line)
        }
    }
}

impl Intersect<Sphere> for Plane {
    type Output = PlaneSphereIntersection;

    /// Intersects this plane with a sphere.
    ///
    /// The signed distance `h` from the sphere center to the plane decides:
    /// `|h|` beyond `radius + tol.confusion` misses
    /// ([`PlaneSphereIntersection::Empty`]); within `tol.confusion` of the
    /// radius it grazes in the foot point
    /// ([`PlaneSphereIntersection::TangentPoint`]); otherwise the section
    /// is a [`Circle3D`] centered at the foot point with radius
    /// `sqrt(radius^2 - h^2)` in the plane normal direction.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{Plane, PlaneSphereIntersection, Point3D, Sphere, Tolerance, Vector3D};
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match plane.intersect(&sphere, tol) {
    ///     PlaneSphereIntersection::Circle(circle) => {
    ///         assert_eq!(circle.center(), Point3D::ORIGIN);
    ///         assert_eq!(circle.radius(), 2.0);
    ///     }
    ///     _ => panic!("expected a circle"),
    /// }
    /// ```
    fn intersect(&self, sphere: &Sphere, tol: Tolerance) -> PlaneSphereIntersection {
        let n = self.normal();
        let h = n.dot(sphere.center() - self.frame.origin());
        let h_abs = h.abs();
        let r = sphere.radius();
        if h_abs > r + tol.confusion {
            PlaneSphereIntersection::Empty
        } else if h_abs >= r - tol.confusion {
            PlaneSphereIntersection::TangentPoint(sphere.center() - n * h)
        } else {
            let center = sphere.center() - n * h;
            // r - h_abs > tol.confusion > 0, so the radius is positive and
            // the normal is unit: construction cannot fail.
            let circle = Circle3D::new(center, n, (r * r - h * h).sqrt())
                .expect("section radius is positive by construction");
            PlaneSphereIntersection::Circle(circle)
        }
    }
}

impl Intersect<Cylinder> for Plane {
    type Output = PlaneCylinderIntersection;

    /// Intersects this plane with a cylinder.
    ///
    /// With `s = n.a` (plane normal against cylinder axis) and
    /// `sin = |n x a|`: a perpendicular plane (`sin` within
    /// `tol.angular`) cuts a [`Circle3D`]; a plane parallel to the axis
    /// (`|s|` within tolerance) misses, grazes one generator, or cuts two
    /// generators, by the axis distance against the radius; otherwise the
    /// section is an [`Ellipse3D`] with minor radius `r` and major radius
    /// `r/|s|` around the axis piercing point.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{Cylinder, Plane, PlaneCylinderIntersection, Point3D, Tolerance, Vector3D};
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match plane.intersect(&cylinder, tol) {
    ///     PlaneCylinderIntersection::Circle(circle) => {
    ///         assert_eq!(circle.center(), Point3D::ORIGIN);
    ///         assert_eq!(circle.radius(), 2.0);
    ///     }
    ///     _ => panic!("expected a circle"),
    /// }
    /// ```
    fn intersect(&self, cylinder: &Cylinder, tol: Tolerance) -> PlaneCylinderIntersection {
        let n = self.normal();
        let a = cylinder.axis().direction();
        let r = cylinder.radius();
        let s = n.dot(a);
        let cross = n.cross(a);
        let sin = cross.magnitude();
        let o = self.frame.origin();
        let c = cylinder.axis().origin();
        if sin <= tol.angular {
            // Perpendicular: circle around the axis piercing point.
            // s is +-1 here, so the division is exact.
            let center = c + a * ((o - c).dot(n) / s);
            let circle = Circle3D::new(center, n, r)
                .expect("cylinder radius is non-negative by construction");
            PlaneCylinderIntersection::Circle(circle)
        } else if s.abs() <= tol.angular {
            // Parallel to the axis: generators at foot +- w*sqrt(r^2-h^2).
            let h = (c - o).dot(n);
            let h_abs = h.abs();
            if h_abs > r + tol.confusion {
                PlaneCylinderIntersection::Empty
            } else {
                let foot = c - n * h;
                let expect = "generator direction is unit by construction";
                if h_abs >= r - tol.confusion {
                    PlaneCylinderIntersection::TangentLine(Line3D::new(foot, a).expect(expect))
                } else {
                    let w = cross * (1.0 / sin);
                    let off = (r * r - h * h).sqrt();
                    PlaneCylinderIntersection::TwoLines(
                        Line3D::new(foot + w * off, a).expect(expect),
                        Line3D::new(foot - w * off, a).expect(expect),
                    )
                }
            }
        } else {
            // Oblique: ellipse with minor r and major r/|s|.
            let h = (c - o).dot(n);
            let foot = c - n * h;
            let m = (a - n * s) * (1.0 / sin);
            let lambda_c = -h * sin / s;
            let center = foot + m * lambda_c;
            let ellipse = Ellipse3D::new(center, n, m, r / s.abs(), r)
                .expect("ellipse radii ordered by construction");
            PlaneCylinderIntersection::Ellipse(ellipse)
        }
    }
}

impl Intersect<Cone> for Plane {
    type Output = PlaneConeIntersection;

    /// Intersects this plane with a cone.
    ///
    /// The cone is double: both nappes, on either side of the apex,
    /// belong to it. With the apex distance `sigma`, the axis/normal
    /// cosine `s`, and the critical angle `PI/2 - |semi_angle|`: a plane
    /// through the apex (`|sigma|` within tolerance) meets just the apex,
    /// one generator, or two generators; a perpendicular plane cuts a
    /// circle; otherwise the section is an ellipse, a parabola (plane
    /// parallel to a generator), or a hyperbola with one branch on each
    /// nappe, built in closed form in the plane's `(M, Y)` basis around
    /// the apex foot point.
    ///
    /// Whether the plane is parallel to a generator (a tangent line through
    /// the apex, a parabola off it) is decided on the angle between the
    /// plane and the nearest generator, within `tol.angular`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{Cone, Frame3D, Plane, PlaneConeIntersection, Point3D, Tolerance, Vector3D};
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match plane.intersect(&cone, tol) {
    ///     PlaneConeIntersection::Circle(circle) => {
    ///         assert_eq!(circle.center(), Point3D::ORIGIN);
    ///         assert!((circle.radius() - 2.0).abs() < 1e-12);
    ///     }
    ///     _ => panic!("expected a circle"),
    /// }
    /// ```
    fn intersect(&self, cone: &Cone, tol: Tolerance) -> PlaneConeIntersection {
        let n = self.normal();
        let a = cone.frame().z_direction();
        // The double cone about `a` with half-angle |semi_angle|.
        let phi = cone.semi_angle().abs();
        let apex = cone.apex();
        let o = self.frame.origin();
        let sigma = (o - apex).dot(n);
        let s = n.dot(a);
        let sin_gamma = n.cross(a).magnitude();
        // The angle between the normal and the axis (up to orientation),
        // and the one at which the plane is parallel to a generator. The
        // sections are told apart on their difference, which is linear in
        // the tilt of the plane.
        let gamma = sin_gamma.atan2(s.abs());
        let gamma_p = std::f64::consts::FRAC_PI_2 - phi;
        if sigma.abs() <= tol.confusion {
            // Through the apex: generators satisfy
            // cos(theta) = -cos(phi)*s / (sin(phi)*sin_gamma).
            if gamma < gamma_p - tol.angular {
                return PlaneConeIntersection::ApexPoint(apex);
            }
            let e1 = (n - a * s) * (1.0 / sin_gamma);
            let e2 = a.cross(e1);
            let expect = "generator direction is unit by construction";
            if gamma <= gamma_p + tol.angular {
                // The generator in the plane, at theta = 0 or PI.
                let g = a * phi.cos() - e1 * (phi.sin() * s.signum());
                PlaneConeIntersection::TangentLine(Line3D::new(apex, g).expect(expect))
            } else {
                let raw = -phi.cos() * s / (phi.sin() * sin_gamma);
                let theta = raw.clamp(-1.0, 1.0).acos();
                let (sin_t, cos_t) = (theta.sin(), theta.cos());
                let g = |w: f64| a * phi.cos() + (e1 * cos_t + e2 * (w * sin_t)) * phi.sin();
                PlaneConeIntersection::TwoLines(
                    Line3D::new(apex, g(1.0)).expect(expect),
                    Line3D::new(apex, g(-1.0)).expect(expect),
                )
            }
        } else {
            if sin_gamma <= tol.angular {
                // Perpendicular: circle at axial distance alpha0 (s = +-1).
                let alpha0 = sigma / s;
                let center = apex + a * alpha0;
                let circle = Circle3D::new(center, n, alpha0.abs() * phi.tan())
                    .expect("section radius is non-negative by construction");
                PlaneConeIntersection::Circle(circle)
            } else {
                // In-plane basis: M holds the axis component, Y = N x M.
                let m = (a - n * s) * (1.0 / sin_gamma);
                let q = apex + n * sigma;
                let tan_phi = phi.tan();
                let a2 = s * s - tan_phi * tan_phi * sin_gamma * sin_gamma;
                let b2 = -2.0 * sigma * s * sin_gamma / phi.cos().powi(2);
                let c0 = sigma * sigma * (sin_gamma * sin_gamma - tan_phi * tan_phi * s * s);
                if gamma < gamma_p - tol.angular {
                    // Ellipse: center at lambda_c, semi-axes from K > 0.
                    let lambda_c = -b2 / (2.0 * a2);
                    let k = (b2 * b2 / (4.0 * a2) - c0).max(0.0);
                    let center = q + m * lambda_c;
                    let semi_l = (k / a2).sqrt();
                    let semi_w = k.sqrt();
                    let (major, minor, x) = if semi_l >= semi_w {
                        (semi_l, semi_w, m)
                    } else {
                        (semi_w, semi_l, n.cross(m))
                    };
                    let ellipse = Ellipse3D::new(center, n, x, major, minor)
                        .expect("ellipse frame valid by construction");
                    PlaneConeIntersection::Ellipse(ellipse)
                } else if (gamma - gamma_p).abs() <= tol.angular {
                    // Parabola: vertex at lambda_v, opening along -sign(B2)*M.
                    // B2 cannot vanish here (sigma, s, sin_gamma all nonzero).
                    let lambda_v = -c0 / b2;
                    let vertex = q + m * lambda_v;
                    let x = if b2 >= 0.0 { m * -1.0 } else { m };
                    let parabola = Parabola3D::new(vertex, n, x, b2.abs() / 4.0)
                        .expect("parabola frame valid by construction");
                    PlaneConeIntersection::Parabola(parabola)
                } else {
                    // Hyperbola: center at lambda_c; the branch opening along
                    // +M lies on the nappe toward +a (alpha grows along +M
                    // since sin_gamma > 0), the one along -M on the other.
                    let lambda_c = -b2 / (2.0 * a2);
                    let k = (b2 * b2 / (4.0 * a2) - c0).min(0.0);
                    let center = q + m * lambda_c;
                    let (major, minor) = ((k / a2).max(0.0).sqrt(), (-k).sqrt());
                    let branch = |x: Vector3D| {
                        Hyperbola3D::new(center, n, x, major, minor)
                            .expect("hyperbola radii positive by construction")
                    };
                    PlaneConeIntersection::Hyperbola(branch(m), branch(m * -1.0))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector3D;

    // ---- construction ----

    #[test]
    fn test_new_ok() {
        let p = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        assert_eq!(p.normal(), Vector3D::Z);
        assert_eq!(p.frame().origin(), Point3D::ORIGIN);
    }

    #[test]
    fn test_new_null_normal_errors() {
        assert_eq!(
            Plane::new(Point3D::ORIGIN, Vector3D::ZERO),
            Err(PlaneConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_from_frame_is_infallible_and_roundtrips() {
        let p = Plane::from_frame(Frame3D::WORLD);
        assert_eq!(p.frame(), Frame3D::WORLD);
    }

    #[test]
    fn test_from_three_points_ok() {
        let p = Plane::from_three_points(
            Point3D::ORIGIN,
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
        )
        .unwrap();
        assert_eq!(p.normal(), Vector3D::Z);
        assert_eq!(p.frame().origin(), Point3D::ORIGIN);
    }

    #[test]
    fn test_from_three_points_confused_errors() {
        let p1 = Point3D::new(1.0, 2.0, 3.0);
        assert_eq!(
            Plane::from_three_points(p1, p1, Point3D::new(4.0, 5.0, 6.0)),
            Err(PlaneConstructionError::ConfusedPoints)
        );
    }

    #[test]
    fn test_from_three_points_collinear_errors() {
        let p1 = Point3D::new(0.0, 0.0, 0.0);
        let p2 = Point3D::new(1.0, 0.0, 0.0);
        let p3 = Point3D::new(2.0, 0.0, 0.0);
        assert_eq!(
            Plane::from_three_points(p1, p2, p3),
            Err(PlaneConstructionError::CollinearPoints)
        );
    }

    #[test]
    fn test_from_coefficients_ok_axis_aligned() {
        let p = Plane::from_coefficients(0.0, 0.0, 1.0, -2.0).unwrap();
        assert_eq!(p.frame().origin(), Point3D::new(0.0, 0.0, 2.0));
        assert_eq!(p.normal(), Vector3D::Z);
    }

    #[test]
    fn test_from_coefficients_bad_equation_errors() {
        assert_eq!(
            Plane::from_coefficients(0.0, 0.0, 0.0, 5.0),
            Err(PlaneConstructionError::BadEquation)
        );
    }

    #[test]
    fn test_from_coefficients_matches_golden_case() {
        // From tests/fixtures/construction.json: planes_from_coefficients[0].
        let p = Plane::from_coefficients(1.0, 2.0, 3.0, 4.0).unwrap();
        let frame = p.frame();
        assert!((frame.origin().x - 0.0).abs() < 1e-9);
        assert!((frame.origin().y - 0.0).abs() < 1e-9);
        assert!((frame.origin().z - (-1.3333333333333333)).abs() < 1e-9);
        let x = frame.x_direction();
        assert!((x.x - 1.3877787807814454e-17).abs() < 1e-9);
        assert!((x.y - 0.8320502943378437).abs() < 1e-9);
        assert!((x.z - (-0.5547001962252293)).abs() < 1e-9);
        let y = frame.y_direction();
        assert!((y.x - (-0.9636241116594315)).abs() < 1e-9);
        assert!((y.y - 0.14824986333222026).abs() < 1e-9);
        assert!((y.z - 0.22237479499833032).abs() < 1e-9);
    }

    // ---- evaluation ----

    #[test]
    fn test_eval_point() {
        let p = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        assert_eq!(p.eval_point(2.0, 3.0), Point3D::new(2.0, 3.0, 0.0));
    }

    #[test]
    fn test_eval_points_matches_loop() {
        let p = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let uvs = [(0.0, 0.0), (1.0, 2.0), (-1.0, 3.0)];
        let expected: Vec<Point3D> = uvs.iter().map(|&(u, v)| p.eval_point(u, v)).collect();
        assert_eq!(p.eval_points(&uvs), expected);
    }

    #[test]
    fn test_eval_derivative_first_orders() {
        let p = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        assert_eq!(p.eval_derivative(0.0, 0.0, 1, 0), Vector3D::X);
        assert_eq!(p.eval_derivative(0.0, 0.0, 0, 1), Vector3D::Y);
    }

    #[test]
    fn test_eval_derivative_second_orders_are_zero() {
        let p = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        assert_eq!(p.eval_derivative(0.0, 0.0, 2, 0), Vector3D::ZERO);
        assert_eq!(p.eval_derivative(0.0, 0.0, 0, 2), Vector3D::ZERO);
        assert_eq!(p.eval_derivative(0.0, 0.0, 1, 1), Vector3D::ZERO);
    }

    #[test]
    #[should_panic(expected = "du + dv must be >= 1")]
    fn test_eval_derivative_zero_order_panics() {
        let p = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        p.eval_derivative(0.0, 0.0, 0, 0);
    }

    #[test]
    #[should_panic(expected = "du + dv must be <= 2")]
    fn test_eval_derivative_order_too_high_panics() {
        let p = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        p.eval_derivative(0.0, 0.0, 2, 1);
    }

    #[test]
    fn test_parameters_of_round_trip() {
        let p = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        assert_eq!(p.parameters_of(Point3D::new(2.0, 3.0, 0.0)), (2.0, 3.0));
    }

    // ---- PlaneConstructionError ----

    #[test]
    fn test_error_display() {
        assert_eq!(
            PlaneConstructionError::NullNormal.to_string(),
            "normal has zero length"
        );
        assert_eq!(
            PlaneConstructionError::ConfusedPoints.to_string(),
            "the points are confused"
        );
        assert_eq!(
            PlaneConstructionError::CollinearPoints.to_string(),
            "the points are collinear"
        );
        assert_eq!(
            PlaneConstructionError::BadEquation.to_string(),
            "the equation is degenerate"
        );
    }

    #[test]
    fn test_error_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&PlaneConstructionError::NullNormal);
    }

    #[test]
    fn test_plane_contains() {
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let tol = Tolerance::DEFAULT;
        assert!(plane.contains(Point3D::ORIGIN, tol));
        assert!(plane.contains(Point3D::new(1.0, 2.0, 0.0), tol));
        assert!(plane.contains(plane.eval_point(1.0, -2.0), tol));
        assert!(!plane.contains(Point3D::new(0.0, 0.0, 1.0), tol));
        // Tolerance boundary: 1e-6 off the plane fails at default, passes loose.
        let near = Point3D::new(0.0, 0.0, 1e-6);
        assert!(!plane.contains(near, tol));
        assert!(plane.contains(
            near,
            Tolerance {
                confusion: 1e-5,
                ..tol
            }
        ));
    }

    #[test]
    fn test_plane_project_point() {
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let tol = Tolerance::DEFAULT;
        let proj = plane.project_point(Point3D::new(1.0, 2.0, 3.0), tol);
        assert_eq!((proj.u, proj.v), (1.0, 2.0));
        assert_eq!(proj.distance, 3.0);
        let on = plane.project_point(Point3D::new(1.0, 2.0, 0.0), tol);
        assert_eq!(on.distance, 0.0);
    }

    #[test]
    fn test_plane_project_points_batch() {
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let tol = Tolerance::DEFAULT;
        let points = [
            Point3D::ORIGIN,
            Point3D::new(0.0, 0.0, 2.0),
            Point3D::new(1.0, 1.0, -3.0),
        ];
        let projs = plane.project_points(&points, tol);
        assert_eq!(projs.len(), 3);
        assert_eq!(projs[0].distance, 0.0);
        assert_eq!(projs[1].distance, 2.0);
        assert_eq!((projs[2].u, projs[2].v), (1.0, 1.0));
        assert_eq!(projs[2].distance, 3.0);
    }

    #[test]
    fn test_plane_plane_intersection_line() {
        let xy = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let zy = Plane::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let tol = Tolerance::DEFAULT;
        match xy.intersect(&zy, tol) {
            PlanePlaneIntersection::Line(line) => {
                assert_eq!(line.origin(), Point3D::ORIGIN);
                assert_eq!(line.direction(), Vector3D::Y);
                // The line lies on both planes.
                assert!(xy.contains(line.eval_point(2.0), tol));
                assert!(zy.contains(line.eval_point(-1.0), tol));
            }
            _ => panic!("expected a line"),
        }
    }

    #[test]
    fn test_plane_plane_intersection_parallel_coincident() {
        let tol = Tolerance::DEFAULT;
        let p1 = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let p2 = Plane::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Z).unwrap();
        assert_eq!(p1.intersect(&p2, tol), PlanePlaneIntersection::Parallel);
        // Flipped normal is still the same plane.
        let p3 = Plane::new(Point3D::ORIGIN, Vector3D::Z * -1.0).unwrap();
        assert_eq!(p1.intersect(&p3, tol), PlanePlaneIntersection::Coincident);
        assert_eq!(p1.intersect(&p1, tol), PlanePlaneIntersection::Coincident);
    }

    #[test]
    fn test_plane_sphere_intersection() {
        let tol = Tolerance::DEFAULT;
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        // Diametral section: great circle.
        match plane.intersect(&Sphere::new(Point3D::ORIGIN, 2.0).unwrap(), tol) {
            PlaneSphereIntersection::Circle(circle) => {
                assert_eq!(circle.center(), Point3D::ORIGIN);
                assert_eq!(circle.radius(), 2.0);
                assert_eq!(circle.normal(), Vector3D::Z);
            }
            _ => panic!("expected a circle"),
        }
        // Offset section.
        match plane.intersect(&Sphere::new(Point3D::new(0.0, 0.0, 1.0), 2.0).unwrap(), tol) {
            PlaneSphereIntersection::Circle(circle) => {
                assert_eq!(circle.center(), Point3D::ORIGIN);
                assert!((circle.radius() - 3.0f64.sqrt()).abs() < 1e-12);
            }
            _ => panic!("expected a circle"),
        }
        // Grazing contact.
        match plane.intersect(&Sphere::new(Point3D::new(0.0, 0.0, 2.0), 2.0).unwrap(), tol) {
            PlaneSphereIntersection::TangentPoint(p) => assert_eq!(p, Point3D::ORIGIN),
            _ => panic!("expected a tangent point"),
        }
        // Clear miss.
        match plane.intersect(&Sphere::new(Point3D::new(0.0, 0.0, 5.0), 2.0).unwrap(), tol) {
            PlaneSphereIntersection::Empty => {}
            _ => panic!("expected empty"),
        }
    }

    #[test]
    fn test_plane_cylinder_intersection_ellipse() {
        let tol = Tolerance::DEFAULT;
        let tilt = 0.3f64;
        let plane =
            Plane::new(Point3D::ORIGIN, Vector3D::new(0.0, tilt.sin(), tilt.cos())).unwrap();
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        match plane.intersect(&cylinder, tol) {
            PlaneCylinderIntersection::Ellipse(ellipse) => {
                assert_eq!(ellipse.center(), Point3D::ORIGIN);
                assert_eq!(ellipse.minor_radius(), 2.0);
                assert!((ellipse.major_radius() - 2.0 / tilt.cos()).abs() < 1e-9);
                for p in ellipse.eval_points(&[0.0, 1.0, 2.0, 4.0]) {
                    assert!(plane.contains(p, tol));
                    assert!(cylinder.contains(p, tol));
                }
            }
            _ => panic!("expected an ellipse"),
        }
    }

    #[test]
    fn test_plane_cylinder_intersection_parallel() {
        let tol = Tolerance::DEFAULT;
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        // Two generators at x = +-sqrt(3), y = 1.
        let plane = Plane::new(Point3D::new(0.0, 1.0, 0.0), Vector3D::Y).unwrap();
        match plane.intersect(&cylinder, tol) {
            PlaneCylinderIntersection::TwoLines(l1, l2) => {
                assert_eq!(l1.direction(), Vector3D::Z);
                assert_eq!(l2.direction(), Vector3D::Z);
                assert!((l1.origin().x - 3.0f64.sqrt()).abs() < 1e-9);
                assert!((l2.origin().x + 3.0f64.sqrt()).abs() < 1e-9);
            }
            _ => panic!("expected two lines"),
        }
        // Grazing generator.
        let tangent = Plane::new(Point3D::new(0.0, 2.0, 0.0), Vector3D::Y).unwrap();
        match tangent.intersect(&cylinder, tol) {
            PlaneCylinderIntersection::TangentLine(l) => {
                assert_eq!(l.origin(), Point3D::new(0.0, 2.0, 0.0));
            }
            _ => panic!("expected a tangent line"),
        }
        // Clear miss.
        let miss = Plane::new(Point3D::new(0.0, 3.0, 0.0), Vector3D::Y).unwrap();
        assert_eq!(
            miss.intersect(&cylinder, tol),
            PlaneCylinderIntersection::Empty
        );
    }

    #[test]
    fn test_plane_cone_intersection_ellipse() {
        let tol = Tolerance::DEFAULT;
        let tilt = 0.2f64;
        let plane =
            Plane::new(Point3D::ORIGIN, Vector3D::new(0.0, tilt.sin(), tilt.cos())).unwrap();
        let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
        match plane.intersect(&cone, tol) {
            PlaneConeIntersection::Ellipse(ellipse) => {
                assert!(ellipse.major_radius() > ellipse.minor_radius());
                for p in ellipse.eval_points(&[0.0, 1.0, 2.0, 4.0]) {
                    assert!(plane.contains(p, tol));
                    assert!(cone.contains(p, tol));
                }
            }
            _ => panic!("expected an ellipse"),
        }
    }

    #[test]
    fn test_plane_cone_intersection_parabola() {
        let tol = Tolerance::DEFAULT;
        let tilt = std::f64::consts::FRAC_PI_2 - 0.4;
        let plane =
            Plane::new(Point3D::ORIGIN, Vector3D::new(0.0, tilt.sin(), tilt.cos())).unwrap();
        let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
        match plane.intersect(&cone, tol) {
            PlaneConeIntersection::Parabola(parabola) => {
                for p in parabola.eval_points(&[-2.0, 0.0, 2.0]) {
                    assert!(plane.contains(p, tol));
                    assert!(cone.contains(p, tol));
                }
            }
            _ => panic!("expected a parabola"),
        }
    }

    #[test]
    fn test_plane_cone_intersection_hyperbola() {
        let tol = Tolerance::DEFAULT;
        let plane = Plane::new(Point3D::new(1.0, 0.0, 0.0), Vector3D::X).unwrap();
        let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
        match plane.intersect(&cone, tol) {
            PlaneConeIntersection::Hyperbola(upper, lower) => {
                let apex_height = cone.apex().z;
                for (branch, side) in [(upper, 1.0), (lower, -1.0)] {
                    for p in branch.eval_points(&[-2.0, 0.0, 2.0]) {
                        assert!(plane.contains(p, tol));
                        assert!(cone.contains(p, tol));
                        // One branch on each nappe.
                        assert!(side * (p.z - apex_height) > 0.0);
                    }
                }
            }
            _ => panic!("expected a hyperbola"),
        }
    }

    #[test]
    fn test_plane_sections_lie_on_their_surface() {
        use crate::{ConicSurfaceIntersection, Ellipse3D, Sphere};
        let tol = Tolerance::DEFAULT;
        let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.5).unwrap();
        let oblique = Plane::new(
            Point3D::new(0.0, 0.0, 1.0),
            Vector3D::new(0.0, 0.2f64.sin(), 0.2f64.cos()),
        )
        .unwrap();
        let PlaneCylinderIntersection::Ellipse(ellipse) = oblique.intersect(&cylinder, tol) else {
            panic!("expected an ellipse");
        };
        assert_eq!(
            ellipse.intersect(&cylinder, tol),
            ConicSurfaceIntersection::Coincident
        );
        let PlaneConeIntersection::Ellipse(ellipse) = oblique.intersect(&cone, tol) else {
            panic!("expected an ellipse");
        };
        assert_eq!(
            ellipse.intersect(&cone, tol),
            ConicSurfaceIntersection::Coincident
        );
        let tilt = std::f64::consts::FRAC_PI_2 - 0.4;
        let parallel = Plane::new(
            Point3D::new(0.0, 0.5, 0.0),
            Vector3D::new(0.0, tilt.sin(), tilt.cos()),
        )
        .unwrap();
        let PlaneConeIntersection::Parabola(parabola) = parallel.intersect(&cone, tol) else {
            panic!("expected a parabola");
        };
        assert_eq!(
            parabola.intersect(&cone, tol),
            ConicSurfaceIntersection::Coincident
        );
        let steep = Plane::new(Point3D::new(1.0, 0.0, 0.0), Vector3D::X).unwrap();
        let PlaneConeIntersection::Hyperbola(upper, lower) = steep.intersect(&cone, tol) else {
            panic!("expected a hyperbola");
        };
        for branch in [upper, lower] {
            assert_eq!(
                branch.intersect(&cone, tol),
                ConicSurfaceIntersection::Coincident
            );
        }
        // A circular ellipse on a sphere, and one slightly larger that only
        // crosses it.
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        let center = Point3D::new(0.0, 0.0, 1.0);
        let r = 3.0f64.sqrt();
        let on = Ellipse3D::new(center, Vector3D::Z, Vector3D::X, r, r).unwrap();
        assert_eq!(
            on.intersect(&sphere, tol),
            ConicSurfaceIntersection::Coincident
        );
        let off = Ellipse3D::new(center, Vector3D::Z, Vector3D::X, r + 1e-3, r).unwrap();
        assert!(matches!(
            off.intersect(&sphere, tol),
            ConicSurfaceIntersection::Hits(_)
        ));
    }

    #[test]
    fn test_plane_cone_intersection_degenerate() {
        let tol = Tolerance::DEFAULT;
        let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
        let apex = cone.apex();
        // Perpendicular plane through the apex: just the tip.
        let perp = Plane::new(apex, Vector3D::Z).unwrap();
        assert_eq!(
            perp.intersect(&cone, tol),
            PlaneConeIntersection::ApexPoint(apex)
        );
        // Steep plane through the apex: two generators.
        let steep = Plane::new(apex, Vector3D::new(1.0, 0.0, 0.1).normalized().unwrap()).unwrap();
        match steep.intersect(&cone, tol) {
            PlaneConeIntersection::TwoLines(l1, l2) => {
                assert_eq!(l1.origin(), apex);
                assert_eq!(l2.origin(), apex);
                assert!(cone.contains(l1.eval_point(2.0), tol));
                assert!(cone.contains(l2.eval_point(2.0), tol));
            }
            _ => panic!("expected two lines"),
        }
    }

    /// A plane through the origin tilted `tilt` from the z axis, about y.
    fn tilted(origin: Point3D, tilt: f64) -> Plane {
        Plane::new(origin, Vector3D::new(tilt.sin(), 0.0, tilt.cos())).unwrap()
    }

    #[test]
    fn test_plane_cone_tangency_band_through_the_apex_is_linear_in_tilt() {
        // Issue #8: the band used to be tested on the cosine of the
        // generator angle, quadratic in the tilt, so it was a third of the
        // angular tolerance wide.
        let cone = Cone::new(Point3D::ORIGIN, Vector3D::Z, 30f64.to_radians(), 0.0).unwrap();
        let tangent = 60f64.to_radians();
        let kind = |d: f64, angular: f64| {
            let tol = Tolerance::new(1e-7, angular, 1e-9);
            match tilted(Point3D::ORIGIN, tangent + d).intersect(&cone, tol) {
                PlaneConeIntersection::ApexPoint(_) => "apex",
                PlaneConeIntersection::TangentLine(_) => "tangent",
                PlaneConeIntersection::TwoLines(..) => "two",
                other => panic!("unexpected {other:?}"),
            }
        };
        for angular in [1e-12, 1e-9] {
            assert_eq!(kind(-2.0 * angular, angular), "apex");
            assert_eq!(kind(-0.5 * angular, angular), "tangent");
            assert_eq!(kind(0.0, angular), "tangent");
            assert_eq!(kind(0.5 * angular, angular), "tangent");
            assert_eq!(kind(2.0 * angular, angular), "two");
        }
        // The issue's reproduction: 1e-10 rad short of tangency is beyond
        // the default band but within a 1e-9 one.
        assert_eq!(kind(-1e-10, 1e-12), "apex");
        assert_eq!(kind(-1e-10, 1e-9), "tangent");
    }

    #[test]
    fn test_plane_cone_tangent_line_is_the_generator_in_the_plane() {
        let tol = Tolerance::DEFAULT;
        for semi_angle in [0.4, -0.4] {
            let cone = Cone::from_frame(Frame3D::WORLD, semi_angle, 2.0).unwrap();
            let apex = cone.apex();
            for tilt in [
                std::f64::consts::FRAC_PI_2 - 0.4,
                std::f64::consts::FRAC_PI_2 + 0.4,
            ] {
                let plane = tilted(apex, tilt);
                let PlaneConeIntersection::TangentLine(line) = plane.intersect(&cone, tol) else {
                    panic!("expected a tangent line at tilt {tilt}");
                };
                for t in [-5.0, 5.0] {
                    assert!(plane.contains(line.eval_point(t), tol));
                    assert!(cone.contains(line.eval_point(t), tol));
                }
            }
        }
    }

    #[test]
    fn test_plane_cone_parabola_band_off_the_apex_is_linear_in_tilt() {
        let cone = Cone::new(Point3D::ORIGIN, Vector3D::Z, 30f64.to_radians(), 0.0).unwrap();
        let tangent = 60f64.to_radians();
        let tol = Tolerance::DEFAULT;
        let at = |d: f64| tilted(Point3D::new(0.0, 0.0, 1.0), tangent + d).intersect(&cone, tol);
        assert!(matches!(at(-2e-12), PlaneConeIntersection::Ellipse(_)));
        assert!(matches!(at(-0.5e-12), PlaneConeIntersection::Parabola(_)));
        assert!(matches!(at(0.5e-12), PlaneConeIntersection::Parabola(_)));
        assert!(matches!(at(2e-12), PlaneConeIntersection::Hyperbola(..)));
    }

    #[test]
    fn test_plane_cone_sections_beyond_the_apex() {
        // The cone is double: sections past the apex lie on the opposite
        // nappe, for either sign of the semi-angle.
        let tol = Tolerance::DEFAULT;
        for semi_angle in [0.4, -0.4] {
            let cone = Cone::from_frame(Frame3D::WORLD, semi_angle, 2.0).unwrap();
            let beyond = 2.0 * (cone.apex().z);
            let on_both =
                |p: Point3D, plane: &Plane| plane.contains(p, tol) && cone.contains(p, tol);
            let flat = Plane::new(Point3D::new(0.0, 0.0, beyond), Vector3D::Z).unwrap();
            match flat.intersect(&cone, tol) {
                PlaneConeIntersection::Circle(circle) => {
                    assert!((circle.radius() - 2.0).abs() < 1e-9);
                    assert!(
                        circle
                            .eval_points(&[0.0, 2.0])
                            .into_iter()
                            .all(|p| on_both(p, &flat))
                    );
                }
                _ => panic!("expected a circle"),
            }
            let tilted = Plane::new(
                Point3D::new(0.0, 0.0, beyond),
                Vector3D::new(0.0, 0.2f64.sin(), 0.2f64.cos()),
            )
            .unwrap();
            match tilted.intersect(&cone, tol) {
                PlaneConeIntersection::Ellipse(ellipse) => {
                    let points = ellipse.eval_points(&[0.0, 1.0, 2.0, 4.0]);
                    assert!(points.into_iter().all(|p| on_both(p, &tilted)));
                }
                _ => panic!("expected an ellipse"),
            }
            let along = std::f64::consts::FRAC_PI_2 - 0.4;
            let parallel = Plane::new(
                Point3D::new(0.0, 0.0, beyond),
                Vector3D::new(0.0, along.sin(), along.cos()),
            )
            .unwrap();
            match parallel.intersect(&cone, tol) {
                PlaneConeIntersection::Parabola(parabola) => {
                    let points = parabola.eval_points(&[-2.0, 0.0, 2.0]);
                    assert!(points.into_iter().all(|p| on_both(p, &parallel)));
                }
                _ => panic!("expected a parabola"),
            }
            // Lines through the apex lie on the cone on both sides of it.
            let apex_plane = Plane::new(cone.apex(), Vector3D::X).unwrap();
            match apex_plane.intersect(&cone, tol) {
                PlaneConeIntersection::TwoLines(l1, l2) => {
                    for t in [-3.0, 3.0] {
                        assert!(cone.contains(l1.eval_point(t), tol));
                        assert!(cone.contains(l2.eval_point(t), tol));
                    }
                }
                _ => panic!("expected two lines"),
            }
        }
    }

    // ---- transformed ----

    #[test]
    fn test_plane_transformed_translation_spot_check() {
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let t = Transform::translation(Vector3D::new(1.0, -1.0, 2.0));
        let moved = plane.transformed(t).unwrap();
        assert_eq!(moved.normal(), Vector3D::Z);
        // Hand-computed evaluation, shifted.
        assert_eq!(moved.eval_point(2.0, 3.0), Point3D::new(3.0, 2.0, 2.0));
    }

    #[test]
    fn test_plane_transformed_round_trip_and_mirror() {
        use crate::Axis3D;
        use std::f64::consts::PI;
        let plane = Plane::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Z).unwrap();
        let axis = Axis3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let t = Transform::rotation(axis, PI / 2.0);
        let moved = plane.transformed(t).unwrap();
        assert!((moved.normal() - -Vector3D::Y).magnitude() < 1e-9);
        let tol = Tolerance::DEFAULT;
        for (u, v) in [(0.0, 0.0), (1.0, -2.0), (-3.0, 0.5)] {
            assert!(moved.contains(t.apply_point(plane.eval_point(u, v)), tol));
        }
        let back = moved
            .transformed(Transform::rotation(axis, -PI / 2.0))
            .unwrap();
        assert!(back.frame().origin().distance(plane.frame().origin()) < 1e-9);
        assert!((back.normal() - plane.normal()).magnitude() < 1e-9);
        // Mirror across the xy-plane preserves the point set.
        let m = Transform::mirror_plane(Frame3D::WORLD);
        let mirrored = plane.transformed(m).unwrap();
        assert!(mirrored.contains(m.apply_point(Point3D::new(4.0, -2.0, 1.0)), tol));
        assert_eq!(mirrored.transformed(m).unwrap(), plane);
    }

    #[test]
    fn test_reversed_planes_share_point_set_with_flipped_normal() {
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        for (u, v) in [(0.0, 0.0), (1.0, -2.0), (-3.0, 0.5)] {
            let ru = plane.reversed_u();
            assert!(!ru.frame().is_direct());
            assert_eq!(ru.eval_point(u, v), plane.eval_point(-u, v));
            assert_eq!(ru.u_reversed_parameter(plane.u_reversed_parameter(u)), u);
            let rv = plane.reversed_v();
            assert!(!rv.frame().is_direct());
            assert_eq!(rv.eval_point(u, v), plane.eval_point(u, -v));
            assert_eq!(rv.v_reversed_parameter(plane.v_reversed_parameter(v)), v);
        }
        assert_eq!(plane.reversed_u().normal(), -Vector3D::Z);
        assert_eq!(plane.reversed_u().reversed_u(), plane);
        assert_eq!(plane.reversed_v().reversed_v(), plane);
    }
}
