//! Ellipses in 3D: parametric evaluation, parameter inversion, and
//! center+two-points construction, a thin wrapper over
//! [`crate::curve_math::analytic`].

use crate::curve_math::analytic;
use crate::curves::{Curve2D, ParametrizeError};
use crate::intersect::{
    ConicSurfaceIntersection, solve_trig, solve_trig_linear, verify_conic_hits,
};
use crate::math::real_roots;
use crate::projection::{self, CurveProjection};
use crate::surfaces::Surface;
use crate::tol;
use crate::{Cone, Cylinder, Frame3D, Plane, Point3D, Sphere, Tolerance, Vector3D};
use std::fmt;

/// Error returned when an [`Ellipse3D`] cannot be constructed from the given
/// inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EllipseConstructionError {
    /// The requested minor radius is negative.
    NegativeRadius,
    /// The major radius is smaller than the minor radius.
    InvertedRadii,
    /// The normal (or the x direction) has zero length, or the x direction
    /// is parallel to the normal.
    NullNormal,
    /// The major-axis point coincides with the center, so no major axis can
    /// be derived.
    NullAxis,
    /// The two points given do not determine a valid major/minor axis pair
    /// (the minor point is farther from, or as close to, the axis line as
    /// the major point, or the two points are collinear with the center).
    InvertedAxis,
}

impl fmt::Display for EllipseConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            EllipseConstructionError::NegativeRadius => "minor radius is negative",
            EllipseConstructionError::InvertedRadii => "major radius is smaller than minor radius",
            EllipseConstructionError::NullNormal => "normal has zero length",
            EllipseConstructionError::NullAxis => "major-axis point coincides with the center",
            EllipseConstructionError::InvertedAxis => {
                "the two points do not determine a valid axis pair"
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for EllipseConstructionError {}

/// An ellipse in 3D: a plane [`Frame3D`] (origin plus local x/y directions
/// defining the plane, the major axis, and the angular origin), a semi-major
/// radius, and a semi-minor radius, evaluated as
/// `origin + major*cos(u)*x_dir + minor*sin(u)*y_dir`.
///
/// # Examples
///
/// ```
/// use geomcore::{Ellipse3D, Point3D, Vector3D};
/// let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
/// assert_eq!(ellipse.eval_point(0.0), Point3D::new(3.0, 0.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse3D {
    frame: Frame3D,
    major_radius: f64,
    minor_radius: f64,
}

impl Ellipse3D {
    /// Creates an ellipse from a center, a normal, an x-axis hint, a
    /// semi-major radius, and a semi-minor radius.
    ///
    /// The plane frame is derived from `normal` and `x_direction` via
    /// [`Frame3D::new`].
    ///
    /// # Errors
    ///
    /// Returns [`EllipseConstructionError::NullNormal`] if `normal` cannot
    /// be normalized, if `x_direction` cannot be normalized, or if
    /// `x_direction` is parallel to `normal`;
    /// [`EllipseConstructionError::NegativeRadius`] if `minor_radius < 0`; or
    /// [`EllipseConstructionError::InvertedRadii`] if
    /// `major_radius < minor_radius`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Point3D, Vector3D};
    /// let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
    /// assert_eq!(ellipse.major_radius(), 3.0);
    /// assert_eq!(ellipse.minor_radius(), 1.5);
    /// ```
    pub fn new(
        center: Point3D,
        normal: Vector3D,
        x_direction: Vector3D,
        major_radius: f64,
        minor_radius: f64,
    ) -> Result<Ellipse3D, EllipseConstructionError> {
        let frame = Frame3D::new(center, normal, x_direction)
            .map_err(|_| EllipseConstructionError::NullNormal)?;
        Ellipse3D::from_frame(frame, major_radius, minor_radius)
    }

    /// Creates an ellipse from a plane frame, a semi-major radius, and a
    /// semi-minor radius directly.
    ///
    /// # Errors
    ///
    /// Returns [`EllipseConstructionError::NegativeRadius`] if
    /// `minor_radius < 0`, or [`EllipseConstructionError::InvertedRadii`] if
    /// `major_radius < minor_radius`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Frame3D};
    /// let ellipse = Ellipse3D::from_frame(Frame3D::WORLD, 3.0, 1.5).unwrap();
    /// assert_eq!(ellipse.frame(), Frame3D::WORLD);
    /// ```
    pub fn from_frame(
        frame: Frame3D,
        major_radius: f64,
        minor_radius: f64,
    ) -> Result<Ellipse3D, EllipseConstructionError> {
        if minor_radius < 0.0 {
            return Err(EllipseConstructionError::NegativeRadius);
        }
        if major_radius < minor_radius {
            return Err(EllipseConstructionError::InvertedRadii);
        }
        Ok(Ellipse3D {
            frame,
            major_radius,
            minor_radius,
        })
    }

    /// Creates an ellipse from a center, a major-axis end point `s1`, and a
    /// second point `s2` that fixes the minor axis and the plane.
    ///
    /// The major axis direction is `x_axis = normalize(s1 - center)`, with
    /// semi-major radius `d1 = |s1 - center|`. The semi-minor radius `d2` is
    /// the distance from `s2` to the line `(center, x_axis)`. The plane
    /// normal is `normalize(x_axis × (s2 - center))`, and the resulting
    /// frame is `Frame3D::new(center, normal, x_axis)`.
    ///
    /// # Errors
    ///
    /// Returns [`EllipseConstructionError::NullAxis`] if `d1` is below
    /// `tol::CONFUSION` (`s1` coincides with `center`); or
    /// [`EllipseConstructionError::InvertedAxis`] if `d1 < d2`, if `d2` is
    /// below `1e-7`, or if `x_axis` is parallel to `s2 - center` (the two
    /// points and the center are collinear).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Point3D};
    /// let ellipse = Ellipse3D::from_center_and_points(
    ///     Point3D::ORIGIN,
    ///     Point3D::new(3.0, 0.0, 0.0),
    ///     Point3D::new(0.0, 1.5, 0.0),
    /// )
    /// .unwrap();
    /// assert_eq!(ellipse.major_radius(), 3.0);
    /// assert_eq!(ellipse.minor_radius(), 1.5);
    /// ```
    pub fn from_center_and_points(
        center: Point3D,
        s1: Point3D,
        s2: Point3D,
    ) -> Result<Ellipse3D, EllipseConstructionError> {
        let v1 = s1 - center;
        let d1 = v1.magnitude();
        if d1 < tol::CONFUSION {
            return Err(EllipseConstructionError::NullAxis);
        }
        let x_axis = v1 * (1.0 / d1);

        let v2 = s2 - center;
        let proj = v2.dot(x_axis);
        let perp = v2 - proj * x_axis;
        let d2 = perp.magnitude();

        if d1 < d2 || d2 < 1e-7 {
            return Err(EllipseConstructionError::InvertedAxis);
        }

        let normal = x_axis.cross(v2);
        if normal.magnitude() < 1e-7 {
            return Err(EllipseConstructionError::InvertedAxis);
        }

        let frame = Frame3D::new(center, normal, x_axis)
            .map_err(|_| EllipseConstructionError::InvertedAxis)?;
        Ellipse3D::from_frame(frame, d1, d2)
    }

    /// Returns the center of the ellipse.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Point3D, Vector3D};
    /// let ellipse = Ellipse3D::new(Point3D::new(1.0, 2.0, 3.0), Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
    /// assert_eq!(ellipse.center(), Point3D::new(1.0, 2.0, 3.0));
    /// ```
    pub fn center(&self) -> Point3D {
        self.frame.origin()
    }

    /// Returns the ellipse's plane frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Frame3D};
    /// let ellipse = Ellipse3D::from_frame(Frame3D::WORLD, 3.0, 1.5).unwrap();
    /// assert_eq!(ellipse.frame(), Frame3D::WORLD);
    /// ```
    pub fn frame(&self) -> Frame3D {
        self.frame
    }

    /// Returns the semi-major radius of the ellipse.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Frame3D};
    /// let ellipse = Ellipse3D::from_frame(Frame3D::WORLD, 3.0, 1.5).unwrap();
    /// assert_eq!(ellipse.major_radius(), 3.0);
    /// ```
    pub fn major_radius(&self) -> f64 {
        self.major_radius
    }

    /// Returns the semi-minor radius of the ellipse.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Frame3D};
    /// let ellipse = Ellipse3D::from_frame(Frame3D::WORLD, 3.0, 1.5).unwrap();
    /// assert_eq!(ellipse.minor_radius(), 1.5);
    /// ```
    pub fn minor_radius(&self) -> f64 {
        self.minor_radius
    }

    /// Area enclosed by the ellipse (`PI*major*minor`).
    ///
    /// (The perimeter has no closed form, so this kernel does not provide
    /// one — not even an approximation disguised as a property.)
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Point3D, Vector3D};
    /// use std::f64::consts::PI;
    /// let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
    /// assert!((ellipse.area() - PI * 4.5).abs() < 1e-12);
    /// ```
    pub fn area(&self) -> f64 {
        std::f64::consts::PI * self.major_radius * self.minor_radius
    }

    /// Evaluates the point on the ellipse at angular parameter `u`:
    /// `center + major*cos(u)*x_dir + minor*sin(u)*y_dir`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Point3D, Vector3D};
    /// let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
    /// assert_eq!(ellipse.eval_point(0.0), Point3D::new(3.0, 0.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64) -> Point3D {
        analytic::ellipse_d0(&self.frame, self.major_radius, self.minor_radius, u)
    }

    /// Evaluates the points on the ellipse at each parameter in `us`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Point3D, Vector3D};
    /// let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
    /// let points = ellipse.eval_points(&[0.0, 1.0]);
    /// assert_eq!(points[0], Point3D::new(3.0, 0.0, 0.0));
    /// ```
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        us.iter().map(|&u| self.eval_point(u)).collect()
    }

    /// Evaluates the derivative of the given `order` at parameter `u`.
    ///
    /// Derivatives cycle with period 4 in `order` (see
    /// `curve_math::analytic::ellipse_dn`).
    ///
    /// # Panics
    ///
    /// Panics if `order == 0`; use [`Ellipse3D::eval_point`] to evaluate the
    /// position itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Point3D, Vector3D};
    /// let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
    /// assert_eq!(ellipse.eval_derivative(0.0, 1), Vector3D::new(0.0, 1.5, 0.0));
    /// ```
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        match order {
            0 => panic!("eval_derivative: order must be >= 1 (use eval_point for order 0)"),
            _ => analytic::ellipse_dn(&self.frame, self.major_radius, self.minor_radius, u, order),
        }
    }

    /// Recovers the angular parameter of a point on (or near) the ellipse,
    /// wrapped into `[0, 2*PI)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Point3D, Vector3D};
    /// let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
    /// assert!((ellipse.parameter_of(Point3D::new(0.0, 1.5, 0.0)) - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    /// ```
    pub fn parameter_of(&self, point: Point3D) -> f64 {
        analytic::ellipse_parameter(&self.frame, self.major_radius, self.minor_radius, point)
    }

    /// Returns whether `point` lies on the ellipse: the inverse parameter
    /// is recovered with [`Ellipse3D::parameter_of`] and re-evaluated, and
    /// the point counts as contained when the re-evaluated point is within
    /// `tol.confusion` of it.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Point3D, Tolerance, Vector3D};
    /// let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// assert!(ellipse.contains(Point3D::new(3.0, 0.0, 0.0), tol));
    /// assert!(!ellipse.contains(Point3D::new(4.0, 0.0, 0.0), tol));
    /// ```
    pub fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        let u = self.parameter_of(point);
        self.eval_point(u).distance(point) <= tol.confusion
    }

    /// All stationary points of the distance from `point` to the ellipse,
    /// ordered by ascending distance.
    ///
    /// The stationarity condition `(E(t) - P).E'(t) = 0` becomes a quartic
    /// in `u = tan(t/2)` (solved by `real_roots`); `t = PI`, which the
    /// substitution misses, is always added as a candidate. The first
    /// entry is the global closest point (see [`Ellipse3D::project_point`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Point3D, Tolerance, Vector3D};
    /// let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
    /// let extrema = ellipse.extrema(Point3D::new(4.0, 0.0, 0.0), Tolerance::DEFAULT);
    /// assert!(!extrema.is_empty());
    /// assert_eq!(extrema[0].distance, 1.0);
    /// ```
    pub fn extrema(&self, point: Point3D, tol: Tolerance) -> Vec<CurveProjection> {
        let rel = point - self.center();
        let px = rel.dot(self.frame.x_direction());
        let py = rel.dot(self.frame.y_direction());
        let (a, b) = (self.major_radius, self.minor_radius);
        let c2 = a * a - b * b;
        let coeffs = [
            -b * py,
            2.0 * (a * px - c2),
            0.0,
            2.0 * (a * px + c2),
            b * py,
        ];
        let mut ts: Vec<f64> = real_roots(&coeffs, tol.confusion)
            .into_iter()
            .map(|r| 2.0 * r.value.atan())
            .collect();
        ts.push(std::f64::consts::PI);
        ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut params: Vec<f64> = Vec::with_capacity(ts.len());
        for t in ts {
            let fresh = match params.last() {
                Some(&last) => (t - last).abs() > tol.parametric * (1.0 + t.abs()),
                None => true,
            };
            if fresh {
                params.push(t);
            }
        }
        let mut out: Vec<CurveProjection> = params
            .into_iter()
            .map(|t| {
                let distance = self.eval_point(t).distance(point);
                CurveProjection {
                    parameter: t,
                    distance: projection::snap_distance(distance, tol.confusion),
                }
            })
            .collect();
        out.sort_by(|x, y| x.distance.partial_cmp(&y.distance).unwrap());
        out
    }

    /// Projects `point` onto the ellipse: the nearest of
    /// [`Ellipse3D::extrema`]. Distances within `tol.confusion` snap to
    /// `0.0`, matching [`Ellipse3D::contains`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Point3D, Tolerance, Vector3D};
    /// let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
    /// let proj = ellipse.project_point(Point3D::new(4.0, 0.0, 0.0), Tolerance::DEFAULT);
    /// assert_eq!(proj.parameter, 0.0);
    /// assert_eq!(proj.distance, 1.0);
    /// ```
    pub fn project_point(&self, point: Point3D, tol: Tolerance) -> CurveProjection {
        self.extrema(point, tol)
            .into_iter()
            .next()
            .expect("a compact smooth curve attains its minimum distance")
    }

    /// Projects each point in `points` onto the ellipse.
    ///
    /// Default-style batch wrapper over [`Ellipse3D::project_point`]: one
    /// native call per batch, mirroring [`Ellipse3D::eval_points`].
    pub fn project_points(&self, points: &[Point3D], tol: Tolerance) -> Vec<CurveProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }

    /// Intersects this ellipse with a plane.
    ///
    /// First-order trigonometric in frame coordinates, solved exactly;
    /// a coplanar-coincident plane holds the whole ellipse.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{ConicSurfaceIntersection, Ellipse3D, Plane, Point3D, Tolerance, Vector3D};
    /// let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match ellipse.intersect_plane(&plane, tol) {
    ///     ConicSurfaceIntersection::Hits(hits) => assert_eq!(hits.len(), 2),
    ///     _ => panic!("expected hits"),
    /// }
    /// ```
    pub fn intersect_plane(&self, plane: &Plane, tol: Tolerance) -> ConicSurfaceIntersection {
        let n = plane.normal();
        if self.frame().z_direction().cross(n).magnitude() <= tol.angular
            && plane.contains(self.center(), tol)
        {
            return ConicSurfaceIntersection::Coincident;
        }
        let x = self.frame().x_direction();
        let y = self.frame().y_direction();
        let (a, b) = (self.major_radius, self.minor_radius);
        let c = a * x.dot(n);
        let d = b * y.dot(n);
        let e = (self.center() - plane.frame().origin()).dot(n);
        let scale = (self.center() - plane.frame().origin()).magnitude() + a + b;
        let hits = verify_conic_hits(self, plane, solve_trig_linear(c, d, e, scale, tol), tol);
        if hits.is_empty() {
            ConicSurfaceIntersection::Empty
        } else {
            ConicSurfaceIntersection::Hits(hits)
        }
    }

    /// Intersects this ellipse with a sphere.
    ///
    /// Double-angle trigonometric form solved as a quartic; hits verified
    /// on the sphere.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{ConicSurfaceIntersection, Ellipse3D, Point3D, Sphere, Tolerance, Vector3D};
    /// let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
    /// let sphere = Sphere::new(Point3D::ORIGIN, 3.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match ellipse.intersect_sphere(&sphere, tol) {
    ///     // Both major vertices graze the sphere (the antipodal one
    ///     // arrives via the PI candidate, reported multiplicity 1).
    ///     ConicSurfaceIntersection::Hits(hits) => {
    ///         assert_eq!(hits.len(), 2);
    ///     }
    ///     _ => panic!("expected hits"),
    /// }
    /// ```
    pub fn intersect_sphere(&self, sphere: &Sphere, tol: Tolerance) -> ConicSurfaceIntersection {
        let (a, b) = (self.major_radius, self.minor_radius);
        let w = self.center() - sphere.center();
        let x = self.frame().x_direction();
        let y = self.frame().y_direction();
        let r = sphere.radius();
        let a2 = (a * a - b * b) / 2.0;
        let b2 = 0.0;
        let c1 = 2.0 * a * x.dot(w);
        let d1 = 2.0 * b * y.dot(w);
        let e = w.dot(w) + (a * a + b * b) / 2.0 - r * r;
        let hits = verify_conic_hits(self, sphere, solve_trig(a2, b2, c1, d1, e, tol), tol);
        if hits.is_empty() {
            ConicSurfaceIntersection::Empty
        } else {
            ConicSurfaceIntersection::Hits(hits)
        }
    }

    /// Intersects this ellipse with a cylinder; hits verified on the
    /// cylinder. See [`Ellipse3D::intersect_plane`] for the calling shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{ConicSurfaceIntersection, Cylinder, Ellipse3D, Point3D, Tolerance, Vector3D};
    /// let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::X, 1.5).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match ellipse.intersect_cylinder(&cylinder, tol) {
    ///     ConicSurfaceIntersection::Hits(hits) => assert_eq!(hits.len(), 2),
    ///     _ => panic!("expected hits"),
    /// }
    /// ```
    pub fn intersect_cylinder(
        &self,
        cylinder: &Cylinder,
        tol: Tolerance,
    ) -> ConicSurfaceIntersection {
        let (a, b) = (self.major_radius, self.minor_radius);
        let ax = cylinder.axis().direction();
        let w = self.center() - cylinder.axis().origin();
        let x = self.frame().x_direction();
        let y = self.frame().y_direction();
        let r = cylinder.radius();
        let (xa, ya, wa) = (x.dot(ax), y.dot(ax), w.dot(ax));
        let (xw, yw) = (x.dot(w), y.dot(w));
        let ww = w.dot(w);
        let a2 = (a * a - b * b) / 2.0 - (a * a * xa * xa - b * b * ya * ya) / 2.0;
        let b2 = -a * b * xa * ya;
        let c1 = 2.0 * a * xw - 2.0 * wa * a * xa;
        let d1 = 2.0 * b * yw - 2.0 * wa * b * ya;
        let e = ww + (a * a + b * b) / 2.0
            - r * r
            - wa * wa
            - (a * a * xa * xa + b * b * ya * ya) / 2.0;
        let hits = verify_conic_hits(self, cylinder, solve_trig(a2, b2, c1, d1, e, tol), tol);
        if hits.is_empty() {
            ConicSurfaceIntersection::Empty
        } else {
            ConicSurfaceIntersection::Hits(hits)
        }
    }

    /// Intersects this ellipse with a cone; hits verified on the cone
    /// (nappe included). See [`Ellipse3D::intersect_plane`] for the
    /// calling shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cone, ConicSurfaceIntersection, Ellipse3D, Frame3D, Point3D, Tolerance, Vector3D};
    /// let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
    /// let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match ellipse.intersect_cone(&cone, tol) {
    ///     ConicSurfaceIntersection::Hits(hits) => assert!(!hits.is_empty()),
    ///     _ => panic!("expected hits"),
    /// }
    /// ```
    pub fn intersect_cone(&self, cone: &Cone, tol: Tolerance) -> ConicSurfaceIntersection {
        let (a, b) = (self.major_radius, self.minor_radius);
        let ax = cone.frame().z_direction();
        let cos_phi = cone.semi_angle().cos();
        let c2 = cos_phi * cos_phi;
        let w = self.center() - cone.apex();
        let x = self.frame().x_direction();
        let y = self.frame().y_direction();
        let (xa, ya, wa) = (x.dot(ax), y.dot(ax), w.dot(ax));
        let (xw, yw) = (x.dot(w), y.dot(w));
        let ww = w.dot(w);
        let a2 = (a * a * xa * xa - b * b * ya * ya) / 2.0 - c2 * (a * a - b * b) / 2.0;
        let b2 = a * b * xa * ya;
        let c1 = 2.0 * wa * a * xa - c2 * 2.0 * a * xw;
        let d1 = 2.0 * wa * b * ya - c2 * 2.0 * b * yw;
        let e =
            wa * wa + (a * a * xa * xa + b * b * ya * ya) / 2.0 - c2 * (ww + (a * a + b * b) / 2.0);
        let hits = verify_conic_hits(self, cone, solve_trig(a2, b2, c1, d1, e, tol), tol);
        if hits.is_empty() {
            ConicSurfaceIntersection::Empty
        } else {
            ConicSurfaceIntersection::Hits(hits)
        }
    }

    /// Computes the exact 2D representation of this ellipse in a surface's
    /// parameter space.
    ///
    /// No ellipse/surface pair has a closed-form 2D image in this release, so
    /// this always returns [`ParametrizeError::NotAnalytic`]. The `surface`
    /// argument is accepted for signature parity with the analytic
    /// [`crate::Line3D::parametrize_on`] and
    /// [`crate::Circle3D::parametrize_on`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::ParametrizeError;
    /// use geomcore::{Ellipse3D, Plane, Point3D, Vector3D};
    ///
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
    /// assert_eq!(ellipse.parametrize_on(&plane), Err(ParametrizeError::NotAnalytic));
    /// ```
    pub fn parametrize_on(&self, surface: impl Into<Surface>) -> Result<Curve2D, ParametrizeError> {
        let _ = surface.into();
        Err(ParametrizeError::NotAnalytic)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ConicSurfaceIntersection, Ellipse3D, EllipseConstructionError, Frame3D, Point3D, Sphere,
        Tolerance, Vector3D, intersect_curve_surface,
    };

    // ---- construction ----

    #[test]
    fn test_ellipse3d_new_ok() {
        let e = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
        assert_eq!(e.center(), Point3D::ORIGIN);
        assert_eq!(e.major_radius(), 3.0);
        assert_eq!(e.minor_radius(), 1.5);
    }

    #[test]
    fn test_ellipse3d_new_negative_minor_radius_errors() {
        assert_eq!(
            Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, -1.0),
            Err(EllipseConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_ellipse3d_new_inverted_radii_errors() {
        assert_eq!(
            Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0, 2.0),
            Err(EllipseConstructionError::InvertedRadii)
        );
    }

    #[test]
    fn test_ellipse3d_new_null_normal_errors() {
        assert_eq!(
            Ellipse3D::new(Point3D::ORIGIN, Vector3D::ZERO, Vector3D::X, 3.0, 1.5),
            Err(EllipseConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_ellipse3d_new_parallel_x_direction_errors() {
        assert_eq!(
            Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::Z, 3.0, 1.5),
            Err(EllipseConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_ellipse3d_from_frame_ok() {
        let e = Ellipse3D::from_frame(Frame3D::WORLD, 3.0, 1.5).unwrap();
        assert_eq!(e.frame(), Frame3D::WORLD);
        assert_eq!(e.major_radius(), 3.0);
        assert_eq!(e.minor_radius(), 1.5);
    }

    #[test]
    fn test_ellipse3d_from_frame_negative_radius_errors() {
        assert_eq!(
            Ellipse3D::from_frame(Frame3D::WORLD, 3.0, -1.0),
            Err(EllipseConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_ellipse3d_from_frame_inverted_radii_errors() {
        assert_eq!(
            Ellipse3D::from_frame(Frame3D::WORLD, 1.0, 2.0),
            Err(EllipseConstructionError::InvertedRadii)
        );
    }

    #[test]
    fn test_ellipse3d_from_center_and_points_ok() {
        let e = Ellipse3D::from_center_and_points(
            Point3D::ORIGIN,
            Point3D::new(3.0, 0.0, 0.0),
            Point3D::new(0.0, 1.5, 0.0),
        )
        .unwrap();
        assert_eq!(e.major_radius(), 3.0);
        assert_eq!(e.minor_radius(), 1.5);
    }

    #[test]
    fn test_ellipse3d_from_center_and_points_null_axis_errors() {
        assert_eq!(
            Ellipse3D::from_center_and_points(
                Point3D::ORIGIN,
                Point3D::ORIGIN,
                Point3D::new(0.0, 1.5, 0.0),
            ),
            Err(EllipseConstructionError::NullAxis)
        );
    }

    #[test]
    fn test_ellipse3d_from_center_and_points_inverted_axis_errors_when_minor_larger() {
        assert_eq!(
            Ellipse3D::from_center_and_points(
                Point3D::ORIGIN,
                Point3D::new(1.0, 0.0, 0.0),
                Point3D::new(0.0, 3.0, 0.0),
            ),
            Err(EllipseConstructionError::InvertedAxis)
        );
    }

    #[test]
    fn test_ellipse3d_from_center_and_points_inverted_axis_errors_when_collinear() {
        assert_eq!(
            Ellipse3D::from_center_and_points(
                Point3D::ORIGIN,
                Point3D::new(3.0, 0.0, 0.0),
                Point3D::new(1.0, 0.0, 0.0),
            ),
            Err(EllipseConstructionError::InvertedAxis)
        );
    }

    // ---- evaluation ----

    #[test]
    fn test_ellipse3d_eval_point_zero() {
        let e = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
        let p = e.eval_point(0.0);
        assert!((p.x - 3.0).abs() < 1e-9);
        assert!(p.y.abs() < 1e-9);
        assert!(p.z.abs() < 1e-9);
    }

    #[test]
    fn test_ellipse3d_eval_points_matches_loop() {
        let e = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
        let us = [0.0, 0.5, 1.5];
        let expected: Vec<Point3D> = us.iter().map(|&u| e.eval_point(u)).collect();
        assert_eq!(e.eval_points(&us), expected);
    }

    #[test]
    fn test_ellipse3d_eval_derivative_order1() {
        let e = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
        let d1 = e.eval_derivative(0.0, 1);
        assert!(d1.x.abs() < 1e-9);
        assert!((d1.y - 1.5).abs() < 1e-9);
    }

    #[test]
    #[should_panic]
    fn test_ellipse3d_eval_derivative_order0_panics() {
        let e = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
        e.eval_derivative(0.0, 0);
    }

    #[test]
    fn test_ellipse3d_parameter_of_round_trip() {
        let e = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
        for u in [0.3, 2.0, 5.5] {
            let p = e.eval_point(u);
            assert!((e.parameter_of(p) - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_ellipse3d_parameter_of_in_zero_to_tau() {
        let e = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
        for u in [-1.0, -0.1, 7.0] {
            let p = e.eval_point(u);
            let recovered = e.parameter_of(p);
            assert!((0.0..std::f64::consts::TAU).contains(&recovered));
        }
    }

    // ---- EllipseConstructionError ----

    #[test]
    fn test_ellipse_construction_error_display() {
        assert_eq!(
            EllipseConstructionError::NegativeRadius.to_string(),
            "minor radius is negative"
        );
        assert_eq!(
            EllipseConstructionError::InvertedRadii.to_string(),
            "major radius is smaller than minor radius"
        );
        assert_eq!(
            EllipseConstructionError::NullNormal.to_string(),
            "normal has zero length"
        );
        assert_eq!(
            EllipseConstructionError::NullAxis.to_string(),
            "major-axis point coincides with the center"
        );
        assert_eq!(
            EllipseConstructionError::InvertedAxis.to_string(),
            "the two points do not determine a valid axis pair"
        );
    }

    #[test]
    fn test_ellipse_construction_error_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&EllipseConstructionError::NegativeRadius);
    }

    #[test]
    fn test_ellipse3d_contains() {
        let e = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
        let tol = Tolerance::DEFAULT;
        assert!(e.contains(Point3D::new(3.0, 0.0, 0.0), tol));
        assert!(e.contains(Point3D::new(0.0, 1.5, 0.0), tol));
        assert!(e.contains(e.eval_point(1.0), tol));
        assert!(!e.contains(Point3D::ORIGIN, tol));
        assert!(!e.contains(Point3D::new(4.0, 0.0, 0.0), tol));
    }

    #[test]
    fn test_ellipse3d_extrema() {
        let e = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
        let tol = Tolerance::DEFAULT;
        // Outside on the major axis: nearest (3,0,0), farthest (-3,0,0).
        let ext = e.extrema(Point3D::new(4.0, 0.0, 0.0), tol);
        assert!(ext.len() >= 2);
        assert_eq!(ext[0].distance, 1.0);
        assert!((ext[0].parameter - 0.0).abs() < 1e-9);
        assert!((ext.last().unwrap().distance - 7.0).abs() < 1e-9);
        // Center: all four axis endpoints equidistant in pairs.
        let center_ext = e.extrema(Point3D::ORIGIN, tol);
        assert_eq!(center_ext.len(), 4);
        assert!((center_ext[0].distance - 1.5).abs() < 1e-9);
        assert!((center_ext[3].distance - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_ellipse3d_project_point() {
        let e = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
        let tol = Tolerance::DEFAULT;
        let proj = e.project_point(Point3D::new(0.0, 3.0, 0.0), tol);
        assert!((proj.distance - 1.5).abs() < 1e-9);
        assert!((proj.parameter - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
        // Batch agrees with scalar.
        let batch = e.project_points(&[Point3D::new(4.0, 0.0, 0.0)], tol);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].distance, 1.0);
    }

    #[test]
    fn test_ellipse3d_intersect_sphere_crosscheck() {
        let tol = Tolerance::DEFAULT;
        let e = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
        // Tilted sphere cutting the ring twice per side.
        let sphere = Sphere::new(Point3D::new(0.0, 0.0, 1.0), 2.5).unwrap();
        match e.intersect_sphere(&sphere, tol) {
            ConicSurfaceIntersection::Hits(hits) => {
                assert!(!hits.is_empty());
                for h in &hits {
                    assert!(e.contains(h.point, tol));
                    assert!(sphere.contains(h.point, tol));
                }
                // Generic solver agrees on the count.
                let generic = intersect_curve_surface(&e, &sphere, tol);
                assert_eq!(generic.len(), hits.len());
            }
            _ => panic!("expected hits"),
        }
    }
}
