//! Tori in 3D: parametric evaluation and parameter inversion, thin wrappers
//! over [`crate::surface_math::analytic`].

use crate::Intersect;
use crate::frame::frame_error;
use crate::math::{QuadraticSolution, solve_quadratic};
use crate::projection::{self, SurfaceProjection};
use crate::surface_math::analytic;
use crate::surfaces::iso::{SurfaceIsoError, check_iso_param, iso_circle};
use crate::{
    Circle3D, Cone, Cylinder, Frame3D, LatitudeRing, Plane, Point3D, Sphere, Tolerance,
    TorusConeIntersection, TorusCylinderIntersection, TorusPlaneIntersection,
    TorusSphereIntersection, TorusTorusIntersection, Transform, Vector3D,
};
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
    /// An input coordinate, radius, or angle is NaN or infinite.
    NonFinite,
}

impl fmt::Display for TorusConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            TorusConstructionError::NegativeRadius => "radius is negative",
            TorusConstructionError::NullNormal => "normal has zero length",
            TorusConstructionError::NonFinite => "an input is NaN or infinite",
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
        let frame = Frame3D::from_z(center, normal).map_err(frame_error(
            TorusConstructionError::NonFinite,
            TorusConstructionError::NullNormal,
        ))?;
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
        if !(major_radius.is_finite() && minor_radius.is_finite()) {
            return Err(TorusConstructionError::NonFinite);
        }
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

    /// Returns the torus running in the opposite angular direction: the
    /// same point set with the frame's y direction negated, so the frame
    /// turns indirect and the normal reverses.
    /// `reversed_u().eval_point(u, v)` is `eval_point(-u, v)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Torus, Vector3D};
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
    /// let reversed = torus.reversed_u();
    /// assert!(!reversed.frame().is_direct());
    /// assert_eq!(reversed.eval_point(1.0, 0.5), torus.eval_point(-1.0, 0.5));
    /// ```
    pub fn reversed_u(&self) -> Torus {
        Torus {
            frame: self.frame.flipped_y(),
            major_radius: self.major_radius,
            minor_radius: self.minor_radius,
        }
    }

    /// Returns the torus running in the opposite tube direction: the same
    /// point set with the frame's z direction negated.
    /// `reversed_v().eval_point(u, v)` is `eval_point(u, -v)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Torus, Vector3D};
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
    /// let reversed = torus.reversed_v();
    /// assert_eq!(reversed.eval_point(1.0, 0.5), torus.eval_point(1.0, -0.5));
    /// ```
    pub fn reversed_v(&self) -> Torus {
        Torus {
            frame: self.frame.flipped_z(),
            major_radius: self.major_radius,
            minor_radius: self.minor_radius,
        }
    }

    /// Maps `u` onto the [`reversed_u`](Self::reversed_u) torus: the
    /// parameter of the same point there (`period − u`).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Torus, Vector3D};
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
    /// let u = torus.u_reversed_parameter(1.0);
    /// let there = torus.reversed_u().eval_point(u, 0.5);
    /// assert!(there.distance(torus.eval_point(1.0, 0.5)) < 1e-12);
    /// ```
    pub fn u_reversed_parameter(&self, u: f64) -> f64 {
        use std::f64::consts::TAU;
        TAU - u
    }

    /// Maps `v` onto the [`reversed_v`](Self::reversed_v) torus: the
    /// parameter of the same point there (`period − v`).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Torus, Vector3D};
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
    /// let v = torus.v_reversed_parameter(0.5);
    /// let there = torus.reversed_v().eval_point(1.0, v);
    /// assert!(there.distance(torus.eval_point(1.0, 0.5)) < 1e-12);
    /// ```
    pub fn v_reversed_parameter(&self, v: f64) -> f64 {
        use std::f64::consts::TAU;
        TAU - v
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

    /// The `u` isoparametric circle at `u`: the meridian, a section of the
    /// tube of the minor radius centred `major_radius` from the center in
    /// the half-plane at angle `u`, so `u_iso(u).eval_point(t)` is
    /// `eval_point(u, t)` for every `t`.
    ///
    /// Fails with [`SurfaceIsoError::NonFinite`] for a NaN or infinite
    /// `u`; any finite value is accepted.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Torus, Vector3D};
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
    /// let iso = torus.u_iso(0.0).unwrap();
    /// assert_eq!(iso.center(), Point3D::new(5.0, 0.0, 0.0));
    /// assert_eq!(iso.radius(), 1.5);
    /// ```
    pub fn u_iso(&self, u: f64) -> Result<Circle3D, SurfaceIsoError> {
        let u = check_iso_param(u, None)?;
        let frame = self.frame;
        let radial = frame.x_direction() * u.cos() + frame.y_direction() * u.sin();
        iso_circle(
            frame.origin() + radial * self.major_radius,
            radial.cross(frame.z_direction()),
            radial,
            self.minor_radius,
        )
    }

    /// The `v` isoparametric circle at `v`: the parallel of radius
    /// `|major + minor * cos(v)|` at height `minor * sin(v)` along the
    /// axis, so `v_iso(v).eval_point(t)` is `eval_point(t, v)` for every
    /// `t`. Where a horn or spindle torus pinches on its axis it is the
    /// circle of radius zero.
    ///
    /// Fails with [`SurfaceIsoError::NonFinite`] for a NaN or infinite
    /// `v`; any finite value is accepted.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Torus, Vector3D};
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
    /// let iso = torus.v_iso(0.0).unwrap();
    /// assert_eq!(iso.center(), Point3D::ORIGIN);
    /// assert_eq!(iso.radius(), 6.5);
    /// ```
    pub fn v_iso(&self, v: f64) -> Result<Circle3D, SurfaceIsoError> {
        let v = check_iso_param(v, None)?;
        let frame = self.frame;
        iso_circle(
            frame.point_at(0.0, 0.0, self.minor_radius * v.sin()),
            frame.x_direction().cross(frame.y_direction()),
            frame.x_direction(),
            self.major_radius + self.minor_radius * v.cos(),
        )
    }

    /// Returns the image of this torus under `transform`: the center and
    /// frame directions map point-wise while both radii scale by the
    /// transform's uniform scale factor.
    ///
    /// Every frame axis maps through the transform, so the image's
    /// parametrization is the transform of this one: under a rotation,
    /// translation or mirror, `transformed(t).eval_point(u, v)` equals
    /// `t.apply_point(self.eval_point(u, v))`. A mirror changes the
    /// frame's handedness, so the image's normal is the opposite of
    /// the mirrored normal (as for any map that reverses orientation);
    /// call `reversed_u` or `reversed_v` on the image to keep the
    /// mirrored side. Fails with [`TorusConstructionError::NullNormal`] only
    /// when the transform collapses the frame (a zero scale factor).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Torus, Transform, Vector3D};
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
    /// let moved = torus.transformed(Transform::scaling(Point3D::ORIGIN, 2.0)).unwrap();
    /// assert_eq!(moved.major_radius(), 10.0);
    /// assert_eq!(moved.minor_radius(), 3.0);
    /// ```
    pub fn transformed(&self, transform: Transform) -> Result<Torus, TorusConstructionError> {
        let (frame, scale) = transform.apply_frame(self.frame()).map_err(frame_error(
            TorusConstructionError::NonFinite,
            TorusConstructionError::NullNormal,
        ))?;
        Torus::from_frame(
            frame,
            self.major_radius() * scale,
            self.minor_radius() * scale,
        )
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

    /// Projects `point` onto the torus, returning the `(u, v)` parameters
    /// of the closest point and its distance. Distances within
    /// `tol.confusion` snap to `0.0`, matching [`Torus::contains`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Tolerance, Torus, Vector3D};
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
    /// let proj = torus.project_point(Point3D::new(6.0, 0.0, 0.0), Tolerance::DEFAULT);
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

    /// Projects each point in `points` onto the torus.
    ///
    /// Default-style batch wrapper over [`Torus::project_point`]: one
    /// native call per batch, mirroring [`Torus::eval_points`].
    pub fn project_points(&self, points: &[Point3D], tol: Tolerance) -> Vec<SurfaceProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }

    /// Shared degenerate-torus guard: a tube or major radius within
    /// tolerance of zero is a curve, not a surface for intersection.
    fn analytic_ok(&self, tol: Tolerance) -> bool {
        self.major_radius() > tol.confusion && self.minor_radius() > tol.confusion
    }

    /// Intersects with a [`Plane`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_plane(&self, plane: &Plane, tol: Tolerance) -> TorusPlaneIntersection {
        self.intersect(plane, tol)
    }

    /// Intersects with a [`Sphere`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_sphere(&self, sphere: &Sphere, tol: Tolerance) -> TorusSphereIntersection {
        self.intersect(sphere, tol)
    }

    /// Intersects with a [`Cylinder`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_cylinder(
        &self,
        cylinder: &Cylinder,
        tol: Tolerance,
    ) -> TorusCylinderIntersection {
        self.intersect(cylinder, tol)
    }

    /// Intersects with a [`Cone`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_cone(&self, cone: &Cone, tol: Tolerance) -> TorusConeIntersection {
        self.intersect(cone, tol)
    }

    /// Intersects with a [`Torus`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_torus(&self, other: &Torus, tol: Tolerance) -> TorusTorusIntersection {
        self.intersect(other, tol)
    }
}

impl Intersect<Plane> for Torus {
    type Output = TorusPlaneIntersection;

    /// Intersects this torus with a plane.
    ///
    /// A plane containing the axis cuts two meridian circles; a plane
    /// perpendicular to the axis cuts latitude rings (two, one grazing,
    /// or one spindle ring); a plane through the center tangent to the
    /// tube twice (tilted `asin(r / R)` from the equatorial plane) cuts
    /// two Villarceau circles of radius `R`. Anything else is a quartic
    /// spiric section and reports [`TorusPlaneIntersection::NotAnalytic`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{Point3D, Tolerance, Torus, TorusPlaneIntersection, Vector3D};
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
    /// let plane = geomcore::Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match torus.intersect(&plane, tol) {
    ///     TorusPlaneIntersection::TwoCircles(_, _) => {}
    ///     _ => panic!("expected two circles"),
    /// }
    /// ```
    fn intersect(&self, plane: &Plane, tol: Tolerance) -> TorusPlaneIntersection {
        let (major, minor) = (self.major_radius(), self.minor_radius());
        if !self.analytic_ok(tol) {
            return TorusPlaneIntersection::NotAnalytic;
        }
        let n = plane.normal();
        let a = self.frame().z_direction();
        let o = self.center();
        let s = n.dot(a);
        let sin = n.cross(a).magnitude();
        // Turning the plane about the centre moves it across the torus by
        // at most the angle times the torus's reach.
        let angular = tol.angular_for_extent(major + minor);
        if s.abs() <= angular {
            // Axis-containing plane: the axis must lie in it.
            if ((o - plane.frame().origin()).dot(n)).abs() > tol.confusion {
                return TorusPlaneIntersection::NotAnalytic;
            }
            // sin is near 1 here (s near 0), so the division is safe.
            let m = n.cross(a) * (1.0 / sin);
            let mk = |sgn: f64| {
                Circle3D::new(o + m * (sgn * major), n, minor)
                    .expect("tube radius is positive by construction")
            };
            TorusPlaneIntersection::TwoCircles(mk(1.0), mk(-1.0))
        } else if sin <= angular {
            let h = (o - plane.frame().origin()).dot(n);
            let h_abs = h.abs();
            if h_abs > minor + tol.confusion {
                TorusPlaneIntersection::Empty
            } else if h_abs >= minor - tol.confusion {
                let center = o - n * h;
                TorusPlaneIntersection::TangentCircle(
                    Circle3D::new(center, n, major)
                        .expect("major radius is positive by construction"),
                )
            } else {
                let s_rad = (minor * minor - h * h).sqrt();
                let center = o - n * h;
                let mk = |rad: f64| Circle3D::new(center, n, rad).expect("kept radii non-negative");
                if major - s_rad >= -tol.confusion {
                    TorusPlaneIntersection::TwoCircles(
                        mk(major + s_rad),
                        mk((major - s_rad).max(0.0)),
                    )
                } else {
                    TorusPlaneIntersection::Circle(mk(major + s_rad))
                }
            }
        } else {
            // Bitangent plane through the center, tilted asin(r / R) from
            // the equatorial plane: two Villarceau circles of radius R,
            // centered r to either side along the line where the two
            // planes meet.
            let through_center = ((o - plane.frame().origin()).dot(n)).abs() <= tol.confusion;
            if through_center && (major * sin - minor).abs() <= tol.confusion {
                let m = n.cross(a) * (1.0 / sin);
                let mk = |sgn: f64| {
                    Circle3D::new(o + m * (sgn * minor), n, major)
                        .expect("major radius is positive by construction")
                };
                TorusPlaneIntersection::TwoCircles(mk(1.0), mk(-1.0))
            } else {
                TorusPlaneIntersection::NotAnalytic
            }
        }
    }
}

impl Intersect<Sphere> for Torus {
    type Output = TorusSphereIntersection;

    /// Intersects this torus with a sphere.
    ///
    /// The analytic path needs the sphere center on the torus axis:
    /// eliminating the radial coordinate gives
    /// `2*R*rho = K + 2*tc*z`, a quadratic in `z` whose roots yield
    /// latitude rings. Anything else reports
    /// [`TorusSphereIntersection::NotAnalytic`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{Point3D, Sphere, Tolerance, Torus, TorusSphereIntersection, Vector3D};
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
    /// let sphere = Sphere::new(Point3D::ORIGIN, 5.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match torus.intersect(&sphere, tol) {
    ///     TorusSphereIntersection::TangentCircle(_) => {}
    ///     _ => panic!("expected a tangent circle"),
    /// }
    /// ```
    fn intersect(&self, sphere: &Sphere, tol: Tolerance) -> TorusSphereIntersection {
        let (major, _) = (self.major_radius(), self.minor_radius());
        if !self.analytic_ok(tol) {
            return TorusSphereIntersection::NotAnalytic;
        }
        let a = self.frame().z_direction();
        let o = self.center();
        let w = sphere.center() - o;
        let tc = w.dot(a);
        if (w - a * tc).magnitude() > tol.confusion {
            return TorusSphereIntersection::NotAnalytic;
        }
        let rs = sphere.radius();
        let k = major * major - self.minor_radius() * self.minor_radius() - tc * tc + rs * rs;
        // Substitute rho(z) into the sphere equation.
        let aq = (tc / major) * (tc / major) + 1.0;
        let bq = 2.0 * (k * tc / (2.0 * major * major) - tc);
        let cq = (k / (2.0 * major)) * (k / (2.0 * major)) + tc * tc - rs * rs;
        let ring = |z: f64| {
            let rho = (k + 2.0 * tc * z) / (2.0 * major);
            if rho >= -tol.confusion {
                Some(Circle3D::new(o + a * z, a, rho.max(0.0)).expect("kept radii non-negative"))
            } else {
                None
            }
        };
        match solve_quadratic(aq, bq, cq, tol) {
            QuadraticSolution::Two(z1, z2) => match (ring(z1), ring(z2)) {
                (Some(c1), Some(c2)) => TorusSphereIntersection::TwoCircles(c1, c2),
                (Some(c), None) | (None, Some(c)) => TorusSphereIntersection::Circle(c),
                (None, None) => TorusSphereIntersection::Empty,
            },
            QuadraticSolution::One(z) => match ring(z) {
                Some(c) => TorusSphereIntersection::TangentCircle(c),
                None => TorusSphereIntersection::Empty,
            },
            QuadraticSolution::Empty => TorusSphereIntersection::Empty,
            QuadraticSolution::Linear(_) | QuadraticSolution::Degenerate => {
                unreachable!("axial quadratic coefficient exceeds 1")
            }
        }
    }
}

impl Intersect<Cylinder> for Torus {
    type Output = TorusCylinderIntersection;

    /// Intersects this torus with a cylinder.
    ///
    /// The analytic path needs coaxial axes: the tube radius against the
    /// torus tube decides miss, graze, or a symmetric ring pair.
    /// Anything else reports
    /// [`TorusCylinderIntersection::NotAnalytic`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{Cylinder, Point3D, Tolerance, Torus, TorusCylinderIntersection, Vector3D};
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 4.5).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match torus.intersect(&cylinder, tol) {
    ///     TorusCylinderIntersection::TwoCircles(_, _) => {}
    ///     _ => panic!("expected two circles"),
    /// }
    /// ```
    fn intersect(&self, cylinder: &Cylinder, tol: Tolerance) -> TorusCylinderIntersection {
        let (major, minor) = (self.major_radius(), self.minor_radius());
        if !self.analytic_ok(tol) {
            return TorusCylinderIntersection::NotAnalytic;
        }
        let a = self.frame().z_direction();
        let o = self.center();
        let ac = cylinder.axis().direction();
        // Coaxial when the cylinder's axis passes through the centre and
        // turning it onto the torus's moves it by at most the confusion
        // tolerance across the torus.
        if a.cross(ac).magnitude() > tol.angular_for_extent(major + minor) {
            return TorusCylinderIntersection::NotAnalytic;
        }
        let w = cylinder.axis().origin() - o;
        if (w - ac * w.dot(ac)).magnitude() > tol.confusion {
            return TorusCylinderIntersection::NotAnalytic;
        }
        let rc = cylinder.radius();
        let gap = (rc - major).abs();
        if gap > minor + tol.confusion {
            TorusCylinderIntersection::Empty
        } else if gap >= minor - tol.confusion {
            TorusCylinderIntersection::TangentCircle(
                Circle3D::new(o, a, rc).expect("tube radius is non-negative by construction"),
            )
        } else {
            let h = (minor * minor - gap * gap).max(0.0).sqrt();
            let mk = |s: f64| {
                Circle3D::new(o + a * (s * h), a, rc)
                    .expect("tube radius is non-negative by construction")
            };
            TorusCylinderIntersection::TwoCircles(mk(1.0), mk(-1.0))
        }
    }
}

impl Intersect<Cone> for Torus {
    type Output = TorusConeIntersection;

    /// Intersects this torus with a cone.
    ///
    /// The analytic path needs coaxial axes. On each nappe the cone
    /// radius is linear in the torus-axial coordinate `z`, `rho = m*(z -
    /// za)` with `m = ±|tan(phi)|`, so `(rho - R)^2 + z^2 = r^2` is a
    /// quadratic per nappe; its roots on that nappe's side of the apex
    /// are rings (a double root a grazing one), up to four in all.
    /// Anything else reports [`TorusConeIntersection::NotAnalytic`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{Cone, Frame3D, Point3D, Tolerance, Torus, TorusConeIntersection, Vector3D};
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
    /// let cone = Cone::from_frame(Frame3D::WORLD, 1.1, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match torus.intersect(&cone, tol) {
    ///     TorusConeIntersection::Rings(rings) => assert_eq!(rings.len(), 2),
    ///     _ => panic!("expected rings"),
    /// }
    /// ```
    fn intersect(&self, cone: &Cone, tol: Tolerance) -> TorusConeIntersection {
        let (major, minor) = (self.major_radius(), self.minor_radius());
        if !self.analytic_ok(tol) {
            return TorusConeIntersection::NotAnalytic;
        }
        let a = self.frame().z_direction();
        let o = self.center();
        let ac = cone.frame().z_direction();
        let w = cone.apex() - o;
        // Coaxial when the apex lies on the axis and turning the cone's
        // axis about it moves the cone by at most the confusion tolerance
        // across the torus.
        let reach = w.magnitude() + major + minor;
        if a.cross(ac).magnitude() > tol.angular_for_extent(reach) {
            return TorusConeIntersection::NotAnalytic;
        }
        if (w - a * w.dot(a)).magnitude() > tol.confusion {
            return TorusConeIntersection::NotAnalytic;
        }
        let za = w.dot(a);
        let k = cone.semi_angle().tan().abs();
        let mut rings: Vec<LatitudeRing> = Vec::new();
        for m in [k, -k] {
            // (m*(z - za) - R)^2 + z^2 = r^2, kept where m*(z - za) >= 0.
            let aq = m * m + 1.0;
            let bq = -2.0 * m * (m * za + major);
            let cq = (m * za + major) * (m * za + major) - minor * minor;
            let mut push = |z: f64, tangent: bool| {
                let rho = m * (z - za);
                if rho >= -tol.confusion {
                    let circle =
                        Circle3D::new(o + a * z, a, rho.max(0.0)).expect("kept radii non-negative");
                    // A ring on the apex is found from both nappes.
                    if !rings.iter().any(|r| r.circle == circle) {
                        rings.push(LatitudeRing { circle, tangent });
                    }
                }
            };
            match solve_quadratic(aq, bq, cq, tol) {
                QuadraticSolution::Two(z1, z2) => {
                    push(z1, false);
                    push(z2, false);
                }
                QuadraticSolution::One(z) => push(z, true),
                QuadraticSolution::Empty => {}
                QuadraticSolution::Linear(_) | QuadraticSolution::Degenerate => {
                    unreachable!("axial quadratic coefficient exceeds 1")
                }
            }
        }
        if rings.is_empty() {
            return TorusConeIntersection::Empty;
        }
        rings.sort_by(|p, q| {
            let height = |r: &LatitudeRing| (r.circle.center() - o).dot(a);
            height(p).total_cmp(&height(q))
        });
        TorusConeIntersection::Rings(rings)
    }
}

impl Intersect<Torus> for Torus {
    type Output = TorusTorusIntersection;

    /// Intersects this torus with another torus.
    ///
    /// The analytic path needs collinear axes. Stacked equal-major tori
    /// reduce to one height; otherwise elimination gives a quadratic in
    /// the axial coordinate. Anything else (up to degree 16) reports
    /// [`TorusTorusIntersection::NotAnalytic`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{Point3D, Tolerance, Torus, TorusTorusIntersection, Vector3D};
    /// let t1 = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
    /// let t2 = Torus::new(Point3D::new(0.0, 0.0, 1.5), Vector3D::Z, 4.0, 1.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match t1.intersect(&t2, tol) {
    ///     TorusTorusIntersection::TwoCircles(_, _) => {}
    ///     _ => panic!("expected two circles"),
    /// }
    /// ```
    fn intersect(&self, other: &Torus, tol: Tolerance) -> TorusTorusIntersection {
        let (r1, s1) = (self.major_radius(), self.minor_radius());
        let (r2, s2) = (other.major_radius(), other.minor_radius());
        if !self.analytic_ok(tol) || !other.analytic_ok(tol) {
            return TorusTorusIntersection::NotAnalytic;
        }
        let a = self.frame().z_direction();
        let o1 = self.center();
        // Coaxial when turning the other torus about its centre onto this
        // one's axis moves it by at most the confusion tolerance.
        if a.cross(other.frame().z_direction()).magnitude() > tol.angular_for_extent(r2 + s2) {
            return TorusTorusIntersection::NotAnalytic;
        }
        let w = other.center() - o1;
        if (w - a * w.dot(a)).magnitude() > tol.confusion {
            return TorusTorusIntersection::NotAnalytic;
        }
        let z2 = w.dot(a);
        if w.magnitude() <= tol.confusion
            && (r1 - r2).abs() <= tol.confusion
            && (s1 - s2).abs() <= tol.confusion
        {
            return TorusTorusIntersection::Coincident;
        }
        let ring = |z: f64, rho: f64| {
            if rho >= -tol.confusion {
                Some(Circle3D::new(o1 + a * z, a, rho.max(0.0)).expect("kept radii non-negative"))
            } else {
                None
            }
        };
        let dr = r2 - r1;
        if dr.abs() <= tol.confusion {
            // Same center circle: stacked (z2 != 0 here) or nested-empty.
            if z2.abs() <= tol.confusion {
                return TorusTorusIntersection::Empty;
            }
            let l1 = (s1 * s1 - s2 * s2) + z2 * z2;
            let z_star = l1 / (2.0 * z2);
            let s_rad = s1 * s1 - z_star * z_star;
            if s_rad < -tol.confusion * (s1 * s1 + z2 * z2 + 1.0) {
                TorusTorusIntersection::Empty
            } else if s_rad.abs() <= tol.confusion * (s1 * s1 + z2 * z2 + 1.0) {
                TorusTorusIntersection::TangentCircle(
                    Circle3D::new(o1 + a * z_star, a, r1)
                        .expect("major radius is positive by construction"),
                )
            } else {
                let root = s_rad.sqrt();
                match (ring(z_star, r1 + root), ring(z_star, r1 - root)) {
                    (Some(c1), Some(c2)) => TorusTorusIntersection::TwoCircles(c1, c2),
                    (Some(c), None) | (None, Some(c)) => TorusTorusIntersection::Circle(c),
                    (None, None) => TorusTorusIntersection::Empty,
                }
            }
        } else {
            // General: rho linear in z, then a quadratic in z.
            let l1 = (s1 * s1 - s2 * s2) - (r1 * r1 - r2 * r2) + z2 * z2;
            let aq = (z2 / dr) * (z2 / dr) + 1.0;
            let bq = 2.0 * ((l1 / (2.0 * dr) - r1) * (-z2 / dr));
            let cq = (l1 / (2.0 * dr) - r1) * (l1 / (2.0 * dr) - r1) - s1 * s1;
            let rho_of = |z: f64| (l1 - 2.0 * z * z2) / (2.0 * dr);
            match solve_quadratic(aq, bq, cq, tol) {
                QuadraticSolution::Two(z1, z2) => {
                    match (ring(z1, rho_of(z1)), ring(z2, rho_of(z2))) {
                        (Some(c1), Some(c2)) => TorusTorusIntersection::TwoCircles(c1, c2),
                        (Some(c), None) | (None, Some(c)) => TorusTorusIntersection::Circle(c),
                        (None, None) => TorusTorusIntersection::Empty,
                    }
                }
                QuadraticSolution::One(z) => match ring(z, rho_of(z)) {
                    Some(c) => TorusTorusIntersection::TangentCircle(c),
                    None => TorusTorusIntersection::Empty,
                },
                QuadraticSolution::Empty => TorusTorusIntersection::Empty,
                QuadraticSolution::Linear(_) | QuadraticSolution::Degenerate => {
                    unreachable!("axial quadratic coefficient exceeds 1")
                }
            }
        }
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

    #[test]
    fn test_torus_project_point() {
        let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
        let tol = Tolerance::DEFAULT;
        let proj = torus.project_point(Point3D::new(6.0, 0.0, 0.0), tol);
        assert_eq!((proj.u, proj.v), (0.0, 0.0));
        assert_eq!(proj.distance, 1.0);
        let on = torus.project_point(torus.eval_point(1.0, 2.0), tol);
        assert_eq!(on.distance, 0.0);
    }

    #[test]
    fn test_torus_plane_intersection() {
        let tol = Tolerance::DEFAULT;
        let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
        // Perpendicular: latitude rings of radii 5 and 3.
        let flat = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        match torus.intersect(&flat, tol) {
            TorusPlaneIntersection::TwoCircles(c1, c2) => {
                let mut radii = [c1.radius(), c2.radius()];
                radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
                assert_eq!(radii, [3.0, 5.0]);
                for p in c1.eval_points(&[0.0, 2.0]) {
                    assert!(torus.contains(p, tol));
                    assert!(flat.contains(p, tol));
                }
            }
            _ => panic!("expected two circles"),
        }
        // Axis-containing: two meridian rings.
        let meridional = Plane::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        match torus.intersect(&meridional, tol) {
            TorusPlaneIntersection::TwoCircles(c1, c2) => {
                assert_eq!(c1.radius(), 1.0);
                assert_eq!(c2.radius(), 1.0);
                let mut xs = [c1.center(), c2.center()];
                xs.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap());
                assert_eq!(xs[0], Point3D::new(0.0, -4.0, 0.0));
                assert_eq!(xs[1], Point3D::new(0.0, 4.0, 0.0));
            }
            _ => panic!("expected two circles"),
        }
        // Grazing the tube top.
        let grazing = Plane::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Z).unwrap();
        match torus.intersect(&grazing, tol) {
            TorusPlaneIntersection::TangentCircle(c) => {
                assert_eq!(c.center(), Point3D::new(0.0, 0.0, 1.0));
                assert_eq!(c.radius(), 4.0);
            }
            _ => panic!("expected a tangent circle"),
        }
        // Clear miss above the tube.
        let miss = Plane::new(Point3D::new(0.0, 0.0, 2.0), Vector3D::Z).unwrap();
        assert_eq!(torus.intersect(&miss, tol), TorusPlaneIntersection::Empty);
        // Tilted: quartic spiric section, no closed form.
        let tilted = Plane::new(Point3D::ORIGIN, Vector3D::new(0.0, 1.0, 1.0)).unwrap();
        assert_eq!(
            torus.intersect(&tilted, tol),
            TorusPlaneIntersection::NotAnalytic
        );
    }

    /// Torus off the origin with a tilted axis, and the plane through its
    /// center whose normal makes `asin(r / R) + extra` with the axis.
    fn tilted_torus_and_plane(extra: f64) -> (Torus, Plane, Vector3D) {
        let axis = Vector3D::new(1.0, 1.0, 2.0).normalized().unwrap();
        let torus = Torus::new(Point3D::new(1.0, -2.0, 0.5), axis, 3.0, 1.0).unwrap();
        let side = axis.cross(Vector3D::X).normalized().unwrap();
        let tilt = (1.0f64 / 3.0).asin() + extra;
        let normal = axis * tilt.cos() + side * tilt.sin();
        let plane = Plane::new(torus.center(), normal).unwrap();
        (torus, plane, side)
    }

    #[test]
    fn test_torus_plane_villarceau_circles() {
        let tol = Tolerance::DEFAULT;
        let (torus, plane, side) = tilted_torus_and_plane(0.0);
        match torus.intersect(&plane, tol) {
            TorusPlaneIntersection::TwoCircles(c1, c2) => {
                for c in [c1, c2] {
                    assert!((c.radius() - 3.0).abs() < 1e-12);
                    assert!((c.center().distance(torus.center()) - 1.0).abs() < 1e-12);
                    let ts: Vec<f64> = (0..12).map(|k| k as f64 * 0.5).collect();
                    for p in c.eval_points(&ts) {
                        assert!(torus.contains(p, tol), "{p:?}");
                        assert!(plane.contains(p, tol), "{p:?}");
                    }
                }
                // Centers on either side of the center, along the line
                // where the plane meets the equatorial plane.
                let line = plane.normal().cross(torus.frame().z_direction());
                let offsets = [c1.center() - torus.center(), c2.center() - torus.center()];
                assert!((offsets[0] + offsets[1]).magnitude() < 1e-12);
                assert!(offsets[0].cross(line).magnitude() < 1e-12);
                assert!(offsets[0].dot(side).abs() < 1e-12);
            }
            other => panic!("expected two Villarceau circles, got {other:?}"),
        }
        // Tilted the other way.
        let (torus, plane, _) = tilted_torus_and_plane(-2.0 * (1.0f64 / 3.0).asin());
        assert!(matches!(
            torus.intersect(&plane, tol),
            TorusPlaneIntersection::TwoCircles(..)
        ));
    }

    #[test]
    fn test_torus_plane_near_villarceau_is_not_analytic() {
        let tol = Tolerance::DEFAULT;
        let (torus, plane, _) = tilted_torus_and_plane(1e-4);
        assert_eq!(
            torus.intersect(&plane, tol),
            TorusPlaneIntersection::NotAnalytic
        );
        let (torus, plane, _) = tilted_torus_and_plane(0.0);
        let shifted = Plane::new(torus.center() + plane.normal() * 1e-4, plane.normal()).unwrap();
        assert_eq!(
            torus.intersect(&shifted, tol),
            TorusPlaneIntersection::NotAnalytic
        );
    }

    #[test]
    fn test_torus_sphere_intersection() {
        let tol = Tolerance::DEFAULT;
        let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
        let sphere = Sphere::new(Point3D::new(0.0, 0.0, 1.0), 4.0).unwrap();
        match torus.intersect(&sphere, tol) {
            TorusSphereIntersection::TwoCircles(c1, c2) => {
                for p in c1.eval_points(&[0.0, 2.0]) {
                    assert!(torus.contains(p, tol));
                    assert!(sphere.contains(p, tol));
                }
                for p in c2.eval_points(&[1.0, 3.0]) {
                    assert!(torus.contains(p, tol));
                    assert!(sphere.contains(p, tol));
                }
            }
            _ => panic!("expected two circles"),
        }
        // Enclosing sphere grazing the outer equator.
        let big = Sphere::new(Point3D::ORIGIN, 5.0).unwrap();
        match torus.intersect(&big, tol) {
            TorusSphereIntersection::TangentCircle(c) => {
                assert_eq!(c.center(), Point3D::ORIGIN);
                assert_eq!(c.radius(), 5.0);
            }
            _ => panic!("expected a tangent circle"),
        }
        // Small sphere in the hole.
        let small = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        assert_eq!(torus.intersect(&small, tol), TorusSphereIntersection::Empty);
        // Off-axis center: no closed form.
        let off = Sphere::new(Point3D::new(1.0, 0.0, 0.0), 4.0).unwrap();
        assert_eq!(
            torus.intersect(&off, tol),
            TorusSphereIntersection::NotAnalytic
        );
    }

    #[test]
    fn test_torus_cylinder_intersection() {
        let tol = Tolerance::DEFAULT;
        let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 4.5).unwrap();
        match torus.intersect(&cylinder, tol) {
            TorusCylinderIntersection::TwoCircles(c1, c2) => {
                assert_eq!(c1.radius(), 4.5);
                assert!((c1.center().z - 0.8660254037844386).abs() < 1e-9);
                assert!((c2.center().z + 0.8660254037844386).abs() < 1e-9);
                for p in c1.eval_points(&[0.0, 2.0]) {
                    assert!(torus.contains(p, tol));
                    assert!(cylinder.contains(p, tol));
                }
            }
            _ => panic!("expected two circles"),
        }
        // Grazing the outer equator.
        let grazing = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 5.0).unwrap();
        match torus.intersect(&grazing, tol) {
            TorusCylinderIntersection::TangentCircle(c) => {
                assert_eq!(c.center(), Point3D::ORIGIN);
                assert_eq!(c.radius(), 5.0);
            }
            _ => panic!("expected a tangent circle"),
        }
        // Oversize tube misses.
        let fat = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 6.0).unwrap();
        assert_eq!(torus.intersect(&fat, tol), TorusCylinderIntersection::Empty);
        // Offset axis: no closed form.
        let off = Cylinder::new(Point3D::new(1.0, 0.0, 0.0), Vector3D::Z, 4.5).unwrap();
        assert_eq!(
            torus.intersect(&off, tol),
            TorusCylinderIntersection::NotAnalytic
        );
    }

    #[test]
    fn test_torus_cone_intersection() {
        let tol = Tolerance::DEFAULT;
        let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
        let cone = Cone::from_frame(Frame3D::WORLD, 1.1, 2.0).unwrap();
        let check = |rings: &[LatitudeRing], cone: &Cone| {
            for ring in rings {
                for p in ring.circle.eval_points(&[0.0, 1.0, 3.0]) {
                    assert!(torus.contains(p, tol));
                    assert!(cone.contains(p, tol));
                }
            }
            let heights: Vec<f64> = rings.iter().map(|r| r.circle.center().z).collect();
            assert!(heights.windows(2).all(|w| w[0] < w[1]), "{heights:?}");
        };
        match torus.intersect(&cone, tol) {
            TorusConeIntersection::Rings(rings) => {
                assert_eq!(rings.len(), 2);
                assert!(rings.iter().all(|r| !r.tangent));
                check(&rings, &cone);
            }
            _ => panic!("expected rings"),
        }
        // Apex at the centre, generators through the tube: two rings on
        // each nappe.
        let flat = Cone::from_frame(Frame3D::WORLD, 1.4, 0.0).unwrap();
        match torus.intersect(&flat, tol) {
            TorusConeIntersection::Rings(rings) => {
                assert_eq!(rings.len(), 4);
                check(&rings, &flat);
                assert!(rings[0].circle.center().z < 0.0 && rings[3].circle.center().z > 0.0);
            }
            _ => panic!("expected four rings"),
        }
        // Generators tangent to the tube: one grazing ring per nappe.
        let grazing = Cone::from_frame(Frame3D::WORLD, 15f64.sqrt().atan(), 0.0).unwrap();
        match torus.intersect(&grazing, tol) {
            TorusConeIntersection::Rings(rings) => {
                assert_eq!(rings.len(), 2);
                assert!(rings.iter().all(|r| r.tangent));
                check(&rings, &grazing);
            }
            _ => panic!("expected two grazing rings"),
        }
        // Tilted cone axis: no closed form.
        let tilted = Cone::new(Point3D::ORIGIN, Vector3D::X, 1.1, 2.0).unwrap();
        assert_eq!(
            torus.intersect(&tilted, tol),
            TorusConeIntersection::NotAnalytic
        );
    }

    #[test]
    fn test_torus_torus_intersection() {
        let tol = Tolerance::DEFAULT;
        let t1 = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
        let t2 = Torus::new(Point3D::new(0.0, 0.0, 1.5), Vector3D::Z, 4.0, 1.0).unwrap();
        match t1.intersect(&t2, tol) {
            TorusTorusIntersection::TwoCircles(c1, c2) => {
                for p in c1.eval_points(&[0.0, 2.0]) {
                    assert!(t1.contains(p, tol));
                    assert!(t2.contains(p, tol));
                }
                for p in c2.eval_points(&[1.0, 3.0]) {
                    assert!(t1.contains(p, tol));
                    assert!(t2.contains(p, tol));
                }
            }
            _ => panic!("expected two circles"),
        }
        // Coincident.
        assert_eq!(t1.intersect(&t1, tol), TorusTorusIntersection::Coincident);
        // Crossed axes: up to degree 16, no closed form.
        let crossed = Torus::new(Point3D::ORIGIN, Vector3D::X, 4.0, 1.0).unwrap();
        assert_eq!(
            t1.intersect(&crossed, tol),
            TorusTorusIntersection::NotAnalytic
        );
    }

    // ---- transformed ----

    #[test]
    fn test_torus_transformed_translation_spot_check() {
        let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
        let t = Transform::translation(Vector3D::new(0.0, 0.0, 2.0));
        let moved = torus.transformed(t).unwrap();
        assert_eq!(moved.center(), Point3D::new(0.0, 0.0, 2.0));
        assert_eq!(moved.major_radius(), 5.0);
        assert_eq!(moved.minor_radius(), 1.5);
        // Hand-computed evaluation at (0, 0), shifted.
        assert_eq!(moved.eval_point(0.0, 0.0), Point3D::new(6.5, 0.0, 2.0));
    }

    #[test]
    fn test_torus_transformed_round_trip_and_mirror() {
        use crate::Axis3D;
        use std::f64::consts::PI;
        let torus = Torus::new(Point3D::new(2.0, 0.0, 0.0), Vector3D::Z, 5.0, 1.5).unwrap();
        let axis = Axis3D::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let t = Transform::rotation(axis, PI / 2.0).then(Transform::scaling(Point3D::ORIGIN, 2.0));
        let moved = torus.transformed(t).unwrap();
        assert_eq!(moved.major_radius(), 10.0);
        assert_eq!(moved.minor_radius(), 3.0);
        assert!(moved.center().distance(Point3D::new(0.0, 4.0, 0.0)) < 1e-9);
        let tol = Tolerance::DEFAULT;
        for (u, v) in [(0.0, 0.0), (1.0, 0.3), (4.0, -0.5)] {
            assert!(moved.contains(t.apply_point(torus.eval_point(u, v)), tol));
        }
        let inv =
            Transform::scaling(Point3D::ORIGIN, 0.5).then(Transform::rotation(axis, -PI / 2.0));
        let back = moved.transformed(inv).unwrap();
        assert!((back.major_radius() - torus.major_radius()).abs() < 1e-9);
        assert!((back.minor_radius() - torus.minor_radius()).abs() < 1e-9);
        assert!(back.center().distance(torus.center()) < 1e-9);
        // Mirror across the xy-plane preserves the point set.
        let m = Transform::mirror_plane(Frame3D::WORLD);
        let mirrored = torus.transformed(m).unwrap();
        for (u, v) in [(0.0, 0.0), (1.0, 0.3)] {
            assert!(mirrored.contains(m.apply_point(torus.eval_point(u, v)), tol));
        }
        assert_eq!(mirrored.transformed(m).unwrap(), torus);
    }

    #[test]
    fn test_reversed_tori_share_point_set() {
        let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
        for (u, v) in [(0.0, 0.0), (1.0, 0.3), (5.0, 2.0)] {
            let ru = torus.reversed_u();
            assert!(!ru.frame().is_direct());
            assert!((ru.eval_point(u, v) - torus.eval_point(-u, v)).magnitude() < 1e-11);
            assert!((ru.u_reversed_parameter(torus.u_reversed_parameter(u)) - u).abs() < 1e-12);
            let rv = torus.reversed_v();
            assert!(!rv.frame().is_direct());
            assert!((rv.eval_point(u, v) - torus.eval_point(u, -v)).magnitude() < 1e-11);
            assert!((rv.v_reversed_parameter(torus.v_reversed_parameter(v)) - v).abs() < 1e-12);
        }
        assert_eq!(torus.reversed_u().reversed_u(), torus);
        assert_eq!(torus.reversed_v().reversed_v(), torus);
    }
}
