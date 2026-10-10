//! Ellipses in 3D: parametric evaluation, parameter inversion, and
//! center+two-points construction, a thin wrapper over
//! [`crate::curve_math::analytic`].

use crate::curve_math::analytic;
use crate::curves::{Curve2D, ParametrizeError};
use crate::intersect::{
    ConicSurfaceIntersection, EllipseEllipse3DIntersection, EllipseEllipseHit, conic_lies_on,
    solve_trig, solve_trig_linear, verify_conic_hits,
};
use crate::math::real_roots;
use crate::projection::{self, CurveProjection};
use crate::surfaces::Surface;
use crate::tol;
use crate::{Cone, Cylinder, Frame3D, Plane, Point3D, Sphere, Tolerance, Transform, Vector3D};
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

/// Five equally spaced parameters, enough to decide whether a whole
/// ellipse lies on a quadric (see `conic_lies_on`).
const ELLIPSE_SAMPLES: [f64; 5] = [
    0.0,
    0.4 * std::f64::consts::PI,
    0.8 * std::f64::consts::PI,
    1.2 * std::f64::consts::PI,
    1.6 * std::f64::consts::PI,
];

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

    /// Returns the image of this ellipse under `transform`: the center and
    /// frame directions map point-wise while both radii scale by the
    /// transform's uniform scale factor.
    ///
    /// A mirror flips handedness, so the image frame is re-derived
    /// right-handed: the point set is preserved but the angular direction
    /// reverses. Fails with [`EllipseConstructionError::NullNormal`] only
    /// when the transform collapses the frame (a zero scale factor).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Point3D, Transform, Vector3D};
    /// let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
    /// let moved = ellipse.transformed(Transform::scaling(Point3D::ORIGIN, 2.0)).unwrap();
    /// assert_eq!(moved.center(), Point3D::ORIGIN);
    /// assert_eq!(moved.major_radius(), 6.0);
    /// assert_eq!(moved.minor_radius(), 3.0);
    /// ```
    pub fn transformed(&self, transform: Transform) -> Result<Ellipse3D, EllipseConstructionError> {
        let (frame, scale) = transform
            .apply_frame(self.frame())
            .map_err(|_| EllipseConstructionError::NullNormal)?;
        Ellipse3D::from_frame(
            frame,
            self.major_radius() * scale,
            self.minor_radius() * scale,
        )
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

    /// Returns whether `point` lies on the ellipse: within `tol.confusion`
    /// of it, exactly when [`Ellipse3D::project_point`] reports distance `0.0`.
    /// The inverse parameter of [`Ellipse3D::parameter_of`] (exact on the
    /// curve, cheap) accepts most contained points; the projection settles
    /// the rest, since off the curve that parameter is not the closest.
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
            || self.project_point(point, tol).distance <= tol.confusion
    }

    /// All stationary points of the distance from `point` to the ellipse,
    /// ordered by ascending distance.
    ///
    /// The stationarity condition `(E(t) - P).E'(t) = 0` becomes a quartic
    /// in `s = tan(t/2)` (solved by `real_roots`). That substitution is
    /// ill-conditioned for roots near `t = PI` (its leading coefficient
    /// vanishes there), so the reversed quartic in `r = tan((t - PI)/2)` is
    /// solved too, and every candidate is polished by Newton iteration on
    /// the trigonometric condition itself. The first entry is the global
    /// closest point (see [`Ellipse3D::project_point`]).
    ///
    /// A query point with a non-finite coordinate has no extrema.
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
        if ![point.x, point.y, point.z].iter().all(|c| c.is_finite()) {
            return Vec::new();
        }
        let rel = point - self.center();
        let px = rel.dot(self.frame.x_direction());
        let py = rel.dot(self.frame.y_direction());
        let (a, b) = (self.major_radius, self.minor_radius);
        let c2 = a * a - b * b;
        // The stationarity condition g(t) = (E(t) - P).E'(t) in frame
        // coordinates, and its derivative.
        let g = |t: f64| {
            let (sn, cs) = t.sin_cos();
            -c2 * sn * cs + a * px * sn - b * py * cs
        };
        let dg = |t: f64| {
            let (sn, cs) = t.sin_cos();
            -c2 * (cs * cs - sn * sn) + a * px * cs + b * py * sn
        };
        let coeffs = [
            -b * py,
            2.0 * (a * px - c2),
            0.0,
            2.0 * (a * px + c2),
            b * py,
        ];
        // s = -1/r maps the quartic to its reversal with alternating signs.
        let reversed = [coeffs[4], -coeffs[3], coeffs[2], -coeffs[1], coeffs[0]];
        let mut ts: Vec<f64> = real_roots(&coeffs, tol.confusion)
            .into_iter()
            .map(|r| 2.0 * r.value.atan())
            .chain(
                real_roots(&reversed, tol.confusion)
                    .into_iter()
                    .map(|r| std::f64::consts::PI + 2.0 * r.value.atan()),
            )
            .chain([0.0, std::f64::consts::PI])
            .map(|mut t| {
                for _ in 0..8 {
                    let slope = dg(t);
                    if slope == 0.0 {
                        break;
                    }
                    let next = t - g(t) / slope;
                    let residual = g(next).abs();
                    if residual.is_nan() || residual >= g(t).abs() {
                        break;
                    }
                    t = next;
                }
                // Into (-PI, PI], the substitution's range.
                let w = t.rem_euclid(std::f64::consts::TAU);
                if w > std::f64::consts::PI {
                    w - std::f64::consts::TAU
                } else {
                    w
                }
            })
            .collect();
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
        out.sort_by(|x, y| x.distance.total_cmp(&y.distance));
        out
    }

    /// Projects `point` onto the ellipse: the nearest of
    /// [`Ellipse3D::extrema`]. Distances within `tol.confusion` snap to
    /// `0.0`, matching [`Ellipse3D::contains`].
    ///
    /// A query point with a non-finite coordinate projects to a NaN
    /// parameter and distance.
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
            .unwrap_or(CurveProjection::UNDEFINED)
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
        if conic_lies_on(self, sphere, ELLIPSE_SAMPLES, tol) {
            return ConicSurfaceIntersection::Coincident;
        }
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
        if conic_lies_on(self, cylinder, ELLIPSE_SAMPLES, tol) {
            return ConicSurfaceIntersection::Coincident;
        }
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

    /// Intersects this ellipse with a cone (both nappes); hits verified on
    /// the cone. See [`Ellipse3D::intersect_plane`] for the
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
        if conic_lies_on(self, cone, ELLIPSE_SAMPLES, tol) {
            return ConicSurfaceIntersection::Coincident;
        }
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

    /// Intersects this ellipse with another 3D ellipse.
    ///
    /// Only coplanar pairs admit a closed form (parallel normals within
    /// `tol.angular` plus the other center in this plane within
    /// `tol.confusion`). Distinct parallel planes report
    /// [`EllipseEllipse3DIntersection::Empty`]; crossing planes report
    /// [`EllipseEllipse3DIntersection::NotAnalytic`]. Whole-curve
    /// coincidence (same center, plane, and radii within tolerance)
    /// reports [`EllipseEllipse3DIntersection::Coincident`].
    ///
    /// In this frame the other ellipse reads
    /// `qa*x^2 + qb*y^2 + qc*x*y + qd*x + qe*y + qf = 0`, a rotated and
    /// shifted copy of `(x'/a2)^2 + (y'/b2)^2 = 1`. Writing it as
    /// `qb*y^2 + beta(x)*y + (qa*x^2 + qd*x + qf) = 0` with
    /// `beta(x) = qc*x + qe` and substituting this ellipse's
    /// `y^2 = delta(x) = b1^2*(1 - x^2/a1^2)` gives
    /// `beta(x)*y + eta(x) = 0` with `eta(x) = qb*delta(x) + qa*x^2 +
    /// qd*x + qf`; squaring yields the quartic
    /// `beta(x)^2*delta(x) - eta(x)^2 = 0` in `x`, solved by
    /// `real_roots`. Each root recovers `y = -eta/beta` (or, when `beta`
    /// vanishes within tolerance, `y = ±sqrt(delta)`), maps back to
    /// `center + x*x_dir + y*y_dir`, and verifies with `contains` on both
    /// ellipses, so squaring artifacts are dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, EllipseEllipse3DIntersection, Point3D, Tolerance, Vector3D};
    /// let e1 = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 2.0).unwrap();
    /// let e2 = Ellipse3D::new(Point3D::new(4.0, 0.0, 0.0), Vector3D::Z, Vector3D::X, 3.0, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match e1.intersect_ellipse(&e2, tol) {
    ///     EllipseEllipse3DIntersection::Points(hits) => assert_eq!(hits.len(), 2),
    ///     _ => panic!("expected two points"),
    /// }
    /// ```
    pub fn intersect_ellipse(
        &self,
        other: &Ellipse3D,
        tol: Tolerance,
    ) -> EllipseEllipse3DIntersection {
        let n1 = self.frame().z_direction();
        let n2 = other.frame().z_direction();
        if n1.cross(n2).magnitude() > tol.angular {
            return EllipseEllipse3DIntersection::NotAnalytic;
        }
        if !Plane::from_frame(self.frame()).contains(other.center(), tol) {
            // Distinct parallel planes.
            return EllipseEllipse3DIntersection::Empty;
        }
        if self.center().distance(other.center()) <= tol.confusion
            && (self.major_radius() - other.major_radius()).abs() <= tol.confusion
            && (self.minor_radius() - other.minor_radius()).abs() <= tol.confusion
            && ((self.major_radius() - self.minor_radius()).abs() <= tol.confusion
                || self
                    .frame()
                    .x_direction()
                    .cross(other.frame().x_direction())
                    .magnitude()
                    <= tol.angular)
        {
            return EllipseEllipse3DIntersection::Coincident;
        }
        let x1 = self.frame().x_direction();
        let y1 = self.frame().y_direction();
        let x2 = other.frame().x_direction();
        let y2 = other.frame().y_direction();
        let (a1, b1) = (self.major_radius(), self.minor_radius());
        let (a2, b2) = (other.major_radius(), other.minor_radius());
        let rel = other.center() - self.center();
        let (cx, cy) = (rel.dot(x1), rel.dot(y1));
        let (p, q) = (x2.dot(x1), x2.dot(y1));
        let (r, s) = (y2.dot(x1), y2.dot(y1));
        let (ia, ib) = (1.0 / (a2 * a2), 1.0 / (b2 * b2));
        let qa = p * p * ia + r * r * ib;
        let qb = q * q * ia + s * s * ib;
        let qc = 2.0 * (p * q * ia + r * s * ib);
        let qd = -(2.0 * qa * cx + qc * cy);
        let qe = -(2.0 * qb * cy + qc * cx);
        let qf = qa * cx * cx + qb * cy * cy + qc * cx * cy - 1.0;
        let d0 = b1 * b1;
        let d2 = -(b1 * b1) / (a1 * a1);
        let (n0, n1) = (qb * d0 + qf, qd);
        let n2 = qb * d2 + qa;
        // beta(x)^2 * (d0 + d2*x^2) - (n0 + n1*x + n2*x^2)^2, ascending.
        let quartic = [
            qe * qe * d0 - n0 * n0,
            2.0 * qc * qe * d0 - 2.0 * n0 * n1,
            qc * qc * d0 + qe * qe * d2 - (2.0 * n0 * n2 + n1 * n1),
            2.0 * qc * qe * d2 - 2.0 * n1 * n2,
            qc * qc * d2 - n2 * n2,
        ];
        let beta_scale = qc.abs() * a1 + qe.abs();
        let eta_scale = n0.abs() + n1.abs() * a1 + n2.abs() * a1 * a1;
        let mut hits: Vec<EllipseEllipseHit> = Vec::new();
        let push = |x: f64, y: f64, hits: &mut Vec<EllipseEllipseHit>| {
            let point = self.center() + x * x1 + y * y1;
            if self.contains(point, tol) && other.contains(point, tol) {
                let hit = EllipseEllipseHit {
                    first_param: self.parameter_of(point),
                    point,
                    second_param: other.parameter_of(point),
                };
                if !hits
                    .iter()
                    .any(|h: &EllipseEllipseHit| h.point.distance(point) <= tol.confusion)
                {
                    hits.push(hit);
                }
            }
        };
        for root in real_roots(&quartic, tol.confusion) {
            let x = root.value;
            let beta = qc * x + qe;
            if beta.abs() > tol.confusion * (1.0 + beta_scale) {
                let eta = n0 + n1 * x + n2 * x * x;
                push(x, -eta / beta, &mut hits);
            } else {
                let eta = n0 + n1 * x + n2 * x * x;
                if eta.abs() > tol.confusion * (1.0 + eta_scale) {
                    continue;
                }
                let delta = d0 + d2 * x * x;
                if delta < -tol.confusion * (1.0 + d0) {
                    continue;
                }
                let y = delta.max(0.0).sqrt();
                // The pair `±y` merging into one point is a tangency on
                // this ellipse's axis: `delta` vanishes there to first
                // order, so its square root amplifies the rounding in `x`
                // (`y ~ sqrt(eps)`). The symmetric pair's midpoint is exact.
                if 2.0 * y <= tol.confusion {
                    push(x, 0.0, &mut hits);
                } else {
                    push(x, y, &mut hits);
                    push(x, -y, &mut hits);
                }
            }
        }
        hits.sort_by(|a, b| a.first_param.partial_cmp(&b.first_param).unwrap());
        match hits.len() {
            0 => EllipseEllipse3DIntersection::Empty,
            1 => EllipseEllipse3DIntersection::Tangent(hits[0]),
            _ => EllipseEllipse3DIntersection::Points(hits),
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
        Axis3D, ConicSurfaceIntersection, Ellipse3D, EllipseConstructionError,
        EllipseEllipse3DIntersection, Frame3D, Point3D, Sphere, Tolerance, Transform, Vector3D,
        intersect_curve_curve, intersect_curve_surface,
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
    fn test_ellipse3d_project_point_near_pi() {
        // The tan(t/2) quartic degenerates as t -> pi; on-curve points there
        // used to project up to 10.65 away (a flat ellipse) or 1e-6 away.
        let tol = Tolerance::DEFAULT;
        for (a, b, u) in [
            (5.880108059940415, 0.6432456550284315, -6.283117687273833),
            (3.0, 1.0, std::f64::consts::PI + 1e-6),
            (3.0, 1.0, std::f64::consts::PI),
        ] {
            let e = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, a, b).unwrap();
            let proj = e.project_point(e.eval_point(u), tol);
            assert!(proj.distance < 1e-12, "({a}, {b}, {u}): {}", proj.distance);
        }
    }

    #[test]
    fn test_ellipse3d_intersect_ellipse() {
        let tol = Tolerance::DEFAULT;
        let e1 = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
        // Offset circle cutting the ring four times, with consistent parameters.
        let e2 = Ellipse3D::new(
            Point3D::new(0.5, 0.0, 0.0),
            Vector3D::Z,
            Vector3D::X,
            2.0,
            2.0,
        )
        .unwrap();
        match e1.intersect_ellipse(&e2, tol) {
            EllipseEllipse3DIntersection::Points(hits) => {
                assert_eq!(hits.len(), 4);
                for h in &hits {
                    assert!(e1.contains(h.point, tol));
                    assert!(e2.contains(h.point, tol));
                    assert_eq!(h.first_param, e1.parameter_of(h.point));
                    assert_eq!(h.second_param, e2.parameter_of(h.point));
                }
                let mut ordered: Vec<f64> = hits.iter().map(|h| h.first_param).collect();
                ordered.sort_by(|a, b| a.partial_cmp(b).unwrap());
                assert_eq!(
                    hits.iter().map(|h| h.first_param).collect::<Vec<_>>(),
                    ordered
                );
            }
            _ => panic!("expected four points"),
        }
        // Grazing circle at the major vertex: a lone tangent.
        let grazing = Ellipse3D::new(
            Point3D::new(5.0, 0.0, 0.0),
            Vector3D::Z,
            Vector3D::X,
            2.0,
            2.0,
        )
        .unwrap();
        match e1.intersect_ellipse(&grazing, tol) {
            EllipseEllipse3DIntersection::Tangent(hit) => {
                assert!(hit.point.distance(Point3D::new(3.0, 0.0, 0.0)) < 1e-12);
                assert_eq!(hit.first_param, e1.parameter_of(hit.point));
                assert_eq!(hit.second_param, grazing.parameter_of(hit.point));
            }
            _ => panic!("expected a tangent"),
        }
        // Distant ellipse: nothing.
        let far = Ellipse3D::new(
            Point3D::new(10.0, 0.0, 0.0),
            Vector3D::Z,
            Vector3D::X,
            3.0,
            1.5,
        )
        .unwrap();
        assert_eq!(
            e1.intersect_ellipse(&far, tol),
            EllipseEllipse3DIntersection::Empty
        );
        // Same center, plane, and radii: coincident.
        assert_eq!(
            e1.intersect_ellipse(&e1, tol),
            EllipseEllipse3DIntersection::Coincident
        );
        // Tilted partner: no closed form.
        let tilted = Ellipse3D::new(Point3D::ORIGIN, Vector3D::X, Vector3D::Z, 3.0, 1.5).unwrap();
        assert_eq!(
            e1.intersect_ellipse(&tilted, tol),
            EllipseEllipse3DIntersection::NotAnalytic
        );
        // Distinct parallel planes: no meeting.
        let lifted = Ellipse3D::new(
            Point3D::new(0.0, 0.0, 1.0),
            Vector3D::Z,
            Vector3D::X,
            3.0,
            1.5,
        )
        .unwrap();
        assert_eq!(
            e1.intersect_ellipse(&lifted, tol),
            EllipseEllipse3DIntersection::Empty
        );
    }

    #[test]
    fn test_ellipse3d_intersect_ellipse_matches_generic() {
        // Deterministic pseudo-random coplanar pairs (LCG) cross-checked
        // against the generic curve-curve solver: same count, same points.
        let tol = Tolerance::DEFAULT;
        let mut state: u64 = 0x0fed_cba9_8765_4321;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f64 / 2147483648.0
        };
        for _ in 0..25 {
            let major1 = 1.0 + 2.0 * next();
            let e1 = Ellipse3D::new(
                Point3D::new(4.0 * next() - 2.0, 4.0 * next() - 2.0, 0.0),
                Vector3D::Z,
                Vector3D::X,
                major1,
                (0.5 + next()).min(major1),
            )
            .unwrap();
            // Second frame rotated in-plane by a random angle.
            let angle = next() * std::f64::consts::TAU;
            let (c, s) = (angle.cos(), angle.sin());
            let frame = Frame3D::new(
                Point3D::new(4.0 * next() - 2.0, 4.0 * next() - 2.0, 0.0),
                Vector3D::Z,
                Vector3D::new(c, s, 0.0),
            )
            .unwrap();
            let major = 1.0 + 2.0 * next();
            let e2 = Ellipse3D::from_frame(frame, major, (0.5 + next()).min(major)).unwrap();
            let analytic = e1.intersect_ellipse(&e2, tol);
            let generic = intersect_curve_curve(&e1, &e2, tol);
            match analytic {
                EllipseEllipse3DIntersection::Points(hits) => {
                    assert_eq!(
                        hits.len(),
                        generic.len(),
                        "count mismatch for {e1:?} {e2:?}"
                    );
                    for (h, g) in hits.iter().zip(generic.iter()) {
                        assert!(
                            h.point.distance(g.point) < 1e-6,
                            "point mismatch for {e1:?} {e2:?}"
                        );
                    }
                }
                EllipseEllipse3DIntersection::Tangent(hit) => {
                    assert_eq!(generic.len(), 1, "count mismatch for {e1:?} {e2:?}");
                    assert!(hit.point.distance(generic[0].point) < 1e-6);
                }
                EllipseEllipse3DIntersection::Empty => {
                    assert!(generic.is_empty(), "generic found hits for {e1:?} {e2:?}")
                }
                _ => panic!("random coplanar pair must be analytic"),
            }
        }
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

    // ---- Ellipse3D transformed ----

    #[test]
    fn test_ellipse3d_transformed_rotation_spot_check() {
        use std::f64::consts::PI;
        let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
        let axis = Axis3D::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let moved = ellipse
            .transformed(Transform::rotation(axis, PI / 2.0))
            .unwrap();
        assert!(moved.center().distance(Point3D::ORIGIN) < 1e-9);
        assert_eq!(moved.major_radius(), 3.0);
        assert_eq!(moved.minor_radius(), 1.5);
        // Major axis (was X) now lies along Y: u = 0 evaluates on Y.
        assert!(moved.eval_point(0.0).distance(Point3D::new(0.0, 3.0, 0.0)) < 1e-9);
    }

    #[test]
    fn test_ellipse3d_transformed_round_trip_and_mirror() {
        let ellipse = Ellipse3D::new(
            Point3D::new(1.0, 2.0, 0.0),
            Vector3D::Z,
            Vector3D::X,
            3.0,
            1.5,
        )
        .unwrap();
        let t = Transform::translation(Vector3D::new(-1.0, 0.5, 2.0))
            .then(Transform::scaling(Point3D::ORIGIN, 0.5));
        let moved = ellipse.transformed(t).unwrap();
        assert!(moved.center().distance(Point3D::new(0.0, 1.25, 1.0)) < 1e-9);
        assert_eq!(moved.major_radius(), 1.5);
        assert_eq!(moved.minor_radius(), 0.75);
        let tol = Tolerance::DEFAULT;
        for u in [0.0, 1.0, 4.0] {
            assert!(moved.contains(t.apply_point(ellipse.eval_point(u)), tol));
        }
        let inv = Transform::scaling(Point3D::ORIGIN, 2.0)
            .then(Transform::translation(Vector3D::new(1.0, -0.5, -2.0)));
        let back = moved.transformed(inv).unwrap();
        for u in [0.0, 1.0, 4.0] {
            assert!(back.eval_point(u).distance(ellipse.eval_point(u)) < 1e-9);
        }
        // Mirror across the xy-plane preserves the point set.
        let m = Transform::mirror_plane(Frame3D::WORLD);
        let mirrored = ellipse.transformed(m).unwrap();
        for u in [0.0, 1.0, 4.0] {
            assert!(mirrored.contains(m.apply_point(ellipse.eval_point(u)), tol));
        }
        assert_eq!(mirrored.transformed(m).unwrap(), ellipse);
    }
}

#[cfg(test)]
mod projection_input_tests {
    use super::*;

    #[test]
    fn test_ellipse3d_project_non_finite_point_is_nan() {
        let curve = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 2.0).unwrap();
        let tol = Tolerance::DEFAULT;
        for p in [
            Point3D::new(f64::NAN, 0.0, 0.0),
            Point3D::new(f64::INFINITY, 0.0, 0.0),
        ] {
            assert!(curve.extrema(p, tol).is_empty());
            let proj = curve.project_point(p, tol);
            assert!(proj.parameter.is_nan() && proj.distance.is_nan());
        }
    }
}
