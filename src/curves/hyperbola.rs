//! Hyperbolas in 3D: parametric evaluation, parameter inversion, and
//! center+two-points construction, a thin wrapper over
//! [`crate::curve_math::analytic`].

use crate::curve_math::analytic;
use crate::curves::{Curve2D, ParametrizeError};
use crate::intersect::{
    ConicSurfaceIntersection, QuadraticSolution, solve_quadratic, verify_conic_hits,
};
use crate::math::real_roots;
use crate::projection::{self, CurveProjection};
use crate::surfaces::Surface;
use crate::tol;
use crate::{Cone, Cylinder, Frame3D, Plane, Point3D, Sphere, Tolerance, Vector3D};
use std::fmt;

/// Error returned when a [`Hyperbola3D`] cannot be constructed from the
/// given inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyperbolaConstructionError {
    /// The major or minor radius is negative.
    NegativeRadius,
    /// The normal (or the x direction) has zero length, or the x direction
    /// is parallel to the normal.
    NullNormal,
    /// Two (or more) of the three points given to build the hyperbola are
    /// coincident (or too close to distinguish).
    ConfusedPoints,
    /// The second point lies on the major-axis line (or is otherwise
    /// collinear with the center and the first point), so no minor axis can
    /// be derived.
    CollinearPoints,
}

impl fmt::Display for HyperbolaConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            HyperbolaConstructionError::NegativeRadius => "radius is negative",
            HyperbolaConstructionError::NullNormal => "normal has zero length",
            HyperbolaConstructionError::ConfusedPoints => "the points are confused",
            HyperbolaConstructionError::CollinearPoints => "the points are collinear",
        };
        f.write_str(message)
    }
}

impl std::error::Error for HyperbolaConstructionError {}

/// A hyperbola in 3D: a plane [`Frame3D`] (origin at the center, plus local
/// x/y directions defining the plane, the major axis, and the transverse
/// direction), a semi-major radius, and a semi-minor radius, evaluated as
/// `center + major*cosh(u)*x_dir + minor*sinh(u)*y_dir`.
///
/// # Examples
///
/// ```
/// use geomcore::{Hyperbola3D, Point3D, Vector3D};
/// let hyperbola = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
/// assert_eq!(hyperbola.eval_point(0.0), Point3D::new(2.0, 0.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hyperbola3D {
    frame: Frame3D,
    major_radius: f64,
    minor_radius: f64,
}

impl Hyperbola3D {
    /// Creates a hyperbola from a center, a normal, an x-axis hint, a
    /// semi-major radius, and a semi-minor radius.
    ///
    /// The plane frame is derived from `normal` and `x_direction` via
    /// [`Frame3D::new`].
    ///
    /// # Errors
    ///
    /// Returns [`HyperbolaConstructionError::NullNormal`] if `normal` cannot
    /// be normalized, if `x_direction` cannot be normalized, or if
    /// `x_direction` is parallel to `normal`; or
    /// [`HyperbolaConstructionError::NegativeRadius`] if `major_radius < 0`
    /// or `minor_radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Hyperbola3D, Point3D, Vector3D};
    /// let hyperbola = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
    /// assert_eq!(hyperbola.major_radius(), 2.0);
    /// assert_eq!(hyperbola.minor_radius(), 1.0);
    /// ```
    pub fn new(
        center: Point3D,
        normal: Vector3D,
        x_direction: Vector3D,
        major_radius: f64,
        minor_radius: f64,
    ) -> Result<Hyperbola3D, HyperbolaConstructionError> {
        let frame = Frame3D::new(center, normal, x_direction)
            .map_err(|_| HyperbolaConstructionError::NullNormal)?;
        Hyperbola3D::from_frame(frame, major_radius, minor_radius)
    }

    /// Creates a hyperbola from a plane frame, a semi-major radius, and a
    /// semi-minor radius directly.
    ///
    /// Unlike [`crate::Ellipse3D`], `minor_radius` may exceed `major_radius`
    /// (only negativity is checked).
    ///
    /// # Errors
    ///
    /// Returns [`HyperbolaConstructionError::NegativeRadius`] if
    /// `major_radius < 0` or `minor_radius < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Hyperbola3D, Frame3D};
    /// let hyperbola = Hyperbola3D::from_frame(Frame3D::WORLD, 2.0, 1.0).unwrap();
    /// assert_eq!(hyperbola.frame(), Frame3D::WORLD);
    /// ```
    pub fn from_frame(
        frame: Frame3D,
        major_radius: f64,
        minor_radius: f64,
    ) -> Result<Hyperbola3D, HyperbolaConstructionError> {
        if major_radius < 0.0 || minor_radius < 0.0 {
            return Err(HyperbolaConstructionError::NegativeRadius);
        }
        Ok(Hyperbola3D {
            frame,
            major_radius,
            minor_radius,
        })
    }

    /// Creates a hyperbola from a center, a major-axis end point `s1`, and a
    /// second point `s2` that fixes the minor axis and the plane.
    ///
    /// The major axis direction is `x_axis = normalize(s1 - center)`, with
    /// semi-major radius `major_radius = |s1 - center|`. The semi-minor
    /// radius `minor_radius` is the distance from `s2` to the line
    /// `(center, x_axis)`. The plane normal is
    /// `normalize(x_axis × (s2 - center))`, and the resulting frame is
    /// `Frame3D::new(center, normal, x_axis)`.
    ///
    /// # Errors
    ///
    /// Returns [`HyperbolaConstructionError::ConfusedPoints`] if any pair
    /// among `{s1, s2, center}` is within `tol::CONFUSION`; or
    /// [`HyperbolaConstructionError::CollinearPoints`] if `s2` lies on the
    /// line `(center, x_axis)`, or `x_axis` is parallel to `s2 - center`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Hyperbola3D, Point3D};
    /// let hyperbola = Hyperbola3D::from_center_and_points(
    ///     Point3D::ORIGIN,
    ///     Point3D::new(2.0, 0.0, 0.0),
    ///     Point3D::new(0.0, 1.0, 0.0),
    /// )
    /// .unwrap();
    /// assert_eq!(hyperbola.major_radius(), 2.0);
    /// assert_eq!(hyperbola.minor_radius(), 1.0);
    /// ```
    pub fn from_center_and_points(
        center: Point3D,
        s1: Point3D,
        s2: Point3D,
    ) -> Result<Hyperbola3D, HyperbolaConstructionError> {
        if center.distance(s1) < tol::CONFUSION
            || center.distance(s2) < tol::CONFUSION
            || s1.distance(s2) < tol::CONFUSION
        {
            return Err(HyperbolaConstructionError::ConfusedPoints);
        }

        let v1 = s1 - center;
        let d1 = v1.magnitude();
        let x_axis = v1 * (1.0 / d1);

        let v2 = s2 - center;
        let proj = v2.dot(x_axis);
        let perp = v2 - proj * x_axis;
        let d2 = perp.magnitude();

        if d2 < tol::CONFUSION {
            return Err(HyperbolaConstructionError::CollinearPoints);
        }

        let normal = x_axis.cross(v2);
        if normal.magnitude() < tol::CONFUSION {
            return Err(HyperbolaConstructionError::CollinearPoints);
        }

        let frame = Frame3D::new(center, normal, x_axis)
            .map_err(|_| HyperbolaConstructionError::CollinearPoints)?;
        Hyperbola3D::from_frame(frame, d1, d2)
    }

    /// Returns the center of the hyperbola.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Hyperbola3D, Point3D, Vector3D};
    /// let hyperbola = Hyperbola3D::new(Point3D::new(1.0, 2.0, 3.0), Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
    /// assert_eq!(hyperbola.center(), Point3D::new(1.0, 2.0, 3.0));
    /// ```
    pub fn center(&self) -> Point3D {
        self.frame.origin()
    }

    /// Returns the hyperbola's plane frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Hyperbola3D, Frame3D};
    /// let hyperbola = Hyperbola3D::from_frame(Frame3D::WORLD, 2.0, 1.0).unwrap();
    /// assert_eq!(hyperbola.frame(), Frame3D::WORLD);
    /// ```
    pub fn frame(&self) -> Frame3D {
        self.frame
    }

    /// Returns the semi-major radius of the hyperbola.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Hyperbola3D, Frame3D};
    /// let hyperbola = Hyperbola3D::from_frame(Frame3D::WORLD, 2.0, 1.0).unwrap();
    /// assert_eq!(hyperbola.major_radius(), 2.0);
    /// ```
    pub fn major_radius(&self) -> f64 {
        self.major_radius
    }

    /// Returns the semi-minor radius of the hyperbola.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Hyperbola3D, Frame3D};
    /// let hyperbola = Hyperbola3D::from_frame(Frame3D::WORLD, 2.0, 1.0).unwrap();
    /// assert_eq!(hyperbola.minor_radius(), 1.0);
    /// ```
    pub fn minor_radius(&self) -> f64 {
        self.minor_radius
    }

    /// Evaluates the point on the hyperbola at parameter `u`:
    /// `center + major*cosh(u)*x_dir + minor*sinh(u)*y_dir`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Hyperbola3D, Point3D, Vector3D};
    /// let hyperbola = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
    /// assert_eq!(hyperbola.eval_point(0.0), Point3D::new(2.0, 0.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64) -> Point3D {
        analytic::hyperbola_d0(&self.frame, self.major_radius, self.minor_radius, u)
    }

    /// Evaluates the points on the hyperbola at each parameter in `us`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Hyperbola3D, Point3D, Vector3D};
    /// let hyperbola = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
    /// let points = hyperbola.eval_points(&[0.0, 1.0]);
    /// assert_eq!(points[0], Point3D::new(2.0, 0.0, 0.0));
    /// ```
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        us.iter().map(|&u| self.eval_point(u)).collect()
    }

    /// Evaluates the derivative of the given `order` at parameter `u`.
    ///
    /// Unlike the trigonometric conics, hyperbolic derivatives do not cycle
    /// with period 4 (see `curve_math::analytic::hyperbola_dn`):
    /// odd orders follow the `sinh`/`cosh` pattern of the first derivative,
    /// even orders follow the `cosh`/`sinh` pattern of the second.
    ///
    /// # Panics
    ///
    /// Panics if `order == 0`; use [`Hyperbola3D::eval_point`] to evaluate
    /// the position itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Hyperbola3D, Point3D, Vector3D};
    /// let hyperbola = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
    /// assert_eq!(hyperbola.eval_derivative(0.0, 1), Vector3D::new(0.0, 1.0, 0.0));
    /// ```
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        match order {
            0 => panic!("eval_derivative: order must be >= 1 (use eval_point for order 0)"),
            _ => {
                analytic::hyperbola_dn(&self.frame, self.major_radius, self.minor_radius, u, order)
            }
        }
    }

    /// Recovers the parameter of a point on (or near) the hyperbola:
    /// `asinh(((point - center) . y_dir) / minor_radius)`. Unbounded (no
    /// wrapping).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Hyperbola3D, Point3D, Vector3D};
    /// let hyperbola = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
    /// assert!((hyperbola.parameter_of(Point3D::new(2.0, 0.0, 0.0)) - 0.0).abs() < 1e-9);
    /// ```
    pub fn parameter_of(&self, point: Point3D) -> f64 {
        analytic::hyperbola_parameter(&self.frame, self.minor_radius, point)
    }

    /// Returns whether `point` lies on the hyperbola: the inverse
    /// parameter is recovered with [`Hyperbola3D::parameter_of`] and
    /// re-evaluated, and the point counts as contained when the
    /// re-evaluated point is within `tol.confusion` of it.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Hyperbola3D, Point3D, Tolerance, Vector3D};
    /// let hyperbola = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// assert!(hyperbola.contains(Point3D::new(2.0, 0.0, 0.0), tol));
    /// assert!(!hyperbola.contains(Point3D::new(2.0, 0.0, 5.0), tol));
    /// ```
    pub fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        let u = self.parameter_of(point);
        self.eval_point(u).distance(point) <= tol.confusion
    }

    /// All stationary points of the distance from `point` to the (single)
    /// hyperbola branch, ordered by ascending distance.
    ///
    /// The stationarity condition becomes a quartic in `u = e^t` (solved
    /// by `real_roots`; only positive roots give real parameters). The
    /// first entry is the global closest point (see
    /// [`Hyperbola3D::project_point`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Hyperbola3D, Point3D, Tolerance, Vector3D};
    /// let hyperbola = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
    /// let extrema = hyperbola.extrema(Point3D::new(2.0, 0.0, 5.0), Tolerance::DEFAULT);
    /// assert!(!extrema.is_empty());
    /// ```
    pub fn extrema(&self, point: Point3D, tol: Tolerance) -> Vec<CurveProjection> {
        let rel = point - self.center();
        let px = rel.dot(self.frame.x_direction());
        let py = rel.dot(self.frame.y_direction());
        let (a, b) = (self.major_radius, self.minor_radius);
        let s2 = a * a + b * b;
        let coeffs = [
            -s2,
            2.0 * (a * px - b * py),
            0.0,
            -2.0 * (a * px + b * py),
            s2,
        ];
        let mut out: Vec<CurveProjection> = real_roots(&coeffs, tol.confusion)
            .into_iter()
            .filter(|r| r.value > 0.0)
            .map(|r| {
                let t = r.value.ln();
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

    /// Projects `point` onto the hyperbola: the nearest of
    /// [`Hyperbola3D::extrema`]. Distances within `tol.confusion` snap to
    /// `0.0`, matching [`Hyperbola3D::contains`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Hyperbola3D, Point3D, Tolerance, Vector3D};
    /// let hyperbola = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
    /// let proj = hyperbola.project_point(Point3D::new(3.0, 0.0, 0.0), Tolerance::DEFAULT);
    /// assert!((proj.parameter.abs() - 0.62236).abs() < 1e-4);
    /// assert!((proj.distance - 0.8f64.sqrt()).abs() < 1e-9);
    /// ```
    pub fn project_point(&self, point: Point3D, tol: Tolerance) -> CurveProjection {
        self.extrema(point, tol)
            .into_iter()
            .next()
            .expect("distance to a hyperbola branch attains its minimum")
    }

    /// Projects each point in `points` onto the hyperbola.
    ///
    /// Default-style batch wrapper over [`Hyperbola3D::project_point`]:
    /// one native call per batch, mirroring [`Hyperbola3D::eval_points`].
    pub fn project_points(&self, points: &[Point3D], tol: Tolerance) -> Vec<CurveProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }

    /// Plane coincident check: parallel normals and center in the plane.
    fn plane_coincident(&self, plane: &Plane, tol: Tolerance) -> bool {
        let n = plane.normal();
        self.frame().z_direction().cross(n).magnitude() <= tol.angular
            && plane.contains(self.center(), tol)
    }

    /// Positive `u = e^t` roots as parameters (filtering `u <= 0`).
    fn params_from_u(&self, roots: Vec<crate::math::RealRoot>) -> Vec<(f64, u32)> {
        roots
            .into_iter()
            .filter(|r| r.value > 0.0)
            .map(|r| (r.value.ln(), r.multiplicity))
            .collect()
    }

    /// Intersects this hyperbola with a plane: quadratic in `u = e^t`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{ConicSurfaceIntersection, Hyperbola3D, Plane, Point3D, Tolerance, Vector3D};
    /// let hyperbola = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
    /// let plane = Plane::new(Point3D::new(2.0, 0.0, 0.0), Vector3D::X).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match hyperbola.intersect_plane(&plane, tol) {
    ///     ConicSurfaceIntersection::Hits(hits) => assert_eq!(hits.len(), 1),
    ///     _ => panic!("expected hits"),
    /// }
    /// ```
    pub fn intersect_plane(&self, plane: &Plane, tol: Tolerance) -> ConicSurfaceIntersection {
        if self.plane_coincident(plane, tol) {
            return ConicSurfaceIntersection::Coincident;
        }
        let n = plane.normal();
        let x = self.frame().x_direction();
        let y = self.frame().y_direction();
        let (a, b) = (self.major_radius, self.minor_radius);
        let e = (self.center() - plane.frame().origin()).dot(n);
        // [(a*xn + b*yn)u^2 + 2e*u + (a*xn - b*yn)] / 2u = 0.
        let aq = a * x.dot(n) + b * y.dot(n);
        let bq = 2.0 * e;
        let cq = a * x.dot(n) - b * y.dot(n);
        let mut candidates = Vec::new();
        match solve_quadratic(aq, bq, cq, tol) {
            QuadraticSolution::Two(u1, u2) => {
                for u in [u1, u2] {
                    if u > 0.0 {
                        candidates.push((u.ln(), 1));
                    }
                }
            }
            QuadraticSolution::One(u) => {
                if u > 0.0 {
                    candidates.push((u.ln(), 2));
                }
            }
            QuadraticSolution::Linear(u) => {
                if u > 0.0 {
                    candidates.push((u.ln(), 1));
                }
            }
            QuadraticSolution::Empty | QuadraticSolution::Degenerate => {}
        }
        let hits = verify_conic_hits(self, plane, candidates, tol);
        if hits.is_empty() {
            ConicSurfaceIntersection::Empty
        } else {
            ConicSurfaceIntersection::Hits(hits)
        }
    }

    /// Intersects this hyperbola with a sphere; hits verified on the
    /// sphere. See [`Hyperbola3D::intersect_plane`] for the calling shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{ConicSurfaceIntersection, Hyperbola3D, Point3D, Sphere, Tolerance, Vector3D};
    /// let hyperbola = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
    /// let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match hyperbola.intersect_sphere(&sphere, tol) {
    ///     ConicSurfaceIntersection::Hits(hits) => assert!(!hits.is_empty()),
    ///     _ => panic!("expected hits"),
    /// }
    /// ```
    pub fn intersect_sphere(&self, sphere: &Sphere, tol: Tolerance) -> ConicSurfaceIntersection {
        let (a, b) = (self.major_radius, self.minor_radius);
        let w = self.center() - sphere.center();
        let x = self.frame().x_direction();
        let y = self.frame().y_direction();
        let r = sphere.radius();
        // |E - S|^2 - R^2, x4u^2 cleared: full quartic (see derivation in
        // intersect_plane docs pattern above).
        let (xw, yw) = (x.dot(w), y.dot(w));
        let ww = w.dot(w);
        let g4 = a * a + b * b;
        let g3 = 4.0 * a * xw + 4.0 * b * yw;
        let g2 = 2.0 * (a * a - b * b) + 4.0 * (ww - r * r);
        let g1 = 4.0 * a * xw - 4.0 * b * yw;
        let g0 = a * a + b * b;
        let roots = real_roots(&[g0, g1, g2, g3, g4], tol.confusion);
        let hits = verify_conic_hits(self, sphere, self.params_from_u(roots), tol);
        if hits.is_empty() {
            ConicSurfaceIntersection::Empty
        } else {
            ConicSurfaceIntersection::Hits(hits)
        }
    }

    /// Intersects this hyperbola with a cylinder; hits verified on the
    /// cylinder (single nappe is automatic here). See
    /// [`Hyperbola3D::intersect_plane`] for the calling shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{ConicSurfaceIntersection, Cylinder, Hyperbola3D, Point3D, Tolerance, Vector3D};
    /// let hyperbola = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match hyperbola.intersect_cylinder(&cylinder, tol) {
    ///     ConicSurfaceIntersection::Hits(hits) => assert!(!hits.is_empty()),
    ///     _ => panic!("expected hits"),
    /// }
    /// ```
    pub fn intersect_cylinder(
        &self,
        cylinder: &Cylinder,
        tol: Tolerance,
    ) -> ConicSurfaceIntersection {
        let (ra, rb) = (self.major_radius, self.minor_radius);
        let ax = cylinder.axis().direction();
        let w = self.center() - cylinder.axis().origin();
        let x = self.frame().x_direction();
        let y = self.frame().y_direction();
        let r = cylinder.radius();
        let (xa, ya, wa) = (x.dot(ax), y.dot(ax), w.dot(ax));
        let (xw, yw) = (x.dot(w), y.dot(w));
        let ww = w.dot(w);
        let p = ra * xa + rb * ya;
        let q = ra * xa - rb * ya;
        // G(u) = F(u) - H(u): F from |E - C0|^2, H = (P*u^2 + 2*wa*u + Q)^2.
        let f4 = ra * ra + rb * rb;
        let f3 = 4.0 * ra * xw + 4.0 * rb * yw;
        let f2 = 4.0 * ww + 2.0 * ra * ra - 2.0 * rb * rb - 4.0 * r * r;
        let f1 = 4.0 * ra * xw - 4.0 * rb * yw;
        let f0 = ra * ra + rb * rb;
        let h4 = p * p;
        let h3 = 4.0 * wa * p;
        let h2 = 4.0 * wa * wa + 2.0 * p * q;
        let h1 = 4.0 * wa * q;
        let h0 = q * q;
        let g = [f0 - h0, f1 - h1, f2 - h2, f3 - h3, f4 - h4];
        let roots = real_roots(&g, tol.confusion);
        let hits = verify_conic_hits(self, cylinder, self.params_from_u(roots), tol);
        if hits.is_empty() {
            ConicSurfaceIntersection::Empty
        } else {
            ConicSurfaceIntersection::Hits(hits)
        }
    }

    /// Intersects this hyperbola with a cone; hits verified on the nappe.
    /// See [`Hyperbola3D::intersect_plane`] for the calling shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cone, ConicSurfaceIntersection, Frame3D, Hyperbola3D, Point3D, Tolerance, Vector3D};
    /// let hyperbola = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
    /// let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match hyperbola.intersect_cone(&cone, tol) {
    ///     ConicSurfaceIntersection::Hits(hits) => assert!(!hits.is_empty()),
    ///     _ => panic!("expected hits"),
    /// }
    /// ```
    pub fn intersect_cone(&self, cone: &Cone, tol: Tolerance) -> ConicSurfaceIntersection {
        let (ra, rb) = (self.major_radius, self.minor_radius);
        let ac = cone.frame().z_direction();
        let cos_phi = cone.semi_angle().cos();
        let c2 = cos_phi * cos_phi;
        let v = self.center() - cone.apex();
        let x = self.frame().x_direction();
        let y = self.frame().y_direction();
        let (xa, ya, va) = (x.dot(ac), y.dot(ac), v.dot(ac));
        let (xv, yv) = (x.dot(v), y.dot(v));
        let vv = v.dot(v);
        let p2 = ra * xa + rb * ya;
        let q2 = ra * xa - rb * ya;
        // G(u) = (P2*u^2 + 2*va*u + Q2)^2 - c^2 * F(u).
        let f4 = ra * ra + rb * rb;
        let f3 = 4.0 * ra * xv + 4.0 * rb * yv;
        let f2 = 4.0 * vv + 2.0 * ra * ra - 2.0 * rb * rb;
        let f1 = 4.0 * ra * xv - 4.0 * rb * yv;
        let f0 = ra * ra + rb * rb;
        let h4 = p2 * p2;
        let h3 = 4.0 * va * p2;
        let h2 = 4.0 * va * va + 2.0 * p2 * q2;
        let h1 = 4.0 * va * q2;
        let h0 = q2 * q2;
        let g = [
            h4 - c2 * f4,
            h3 - c2 * f3,
            h2 - c2 * f2,
            h1 - c2 * f1,
            h0 - c2 * f0,
        ];
        // Ascending order for the solver.
        let q = [g[4], g[3], g[2], g[1], g[0]];
        let roots = real_roots(&q, tol.confusion);
        let hits = verify_conic_hits(self, cone, self.params_from_u(roots), tol);
        if hits.is_empty() {
            ConicSurfaceIntersection::Empty
        } else {
            ConicSurfaceIntersection::Hits(hits)
        }
    }

    /// Computes the exact 2D representation of this hyperbola in a surface's
    /// parameter space.
    ///
    /// No hyperbola/surface pair has a closed-form 2D image in this release,
    /// so this always returns [`ParametrizeError::NotAnalytic`]. The `surface`
    /// argument is accepted for signature parity with the analytic
    /// [`crate::Line3D::parametrize_on`] and
    /// [`crate::Circle3D::parametrize_on`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::ParametrizeError;
    /// use geomcore::{Hyperbola3D, Plane, Point3D, Vector3D};
    ///
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let hyperbola = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
    /// assert_eq!(hyperbola.parametrize_on(&plane), Err(ParametrizeError::NotAnalytic));
    /// ```
    pub fn parametrize_on(&self, surface: impl Into<Surface>) -> Result<Curve2D, ParametrizeError> {
        let _ = surface.into();
        Err(ParametrizeError::NotAnalytic)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Cone, ConicSurfaceIntersection, Frame3D, Hyperbola3D, HyperbolaConstructionError, Point3D,
        Tolerance, Vector3D, intersect_curve_surface,
    };

    // ---- construction ----

    #[test]
    fn test_hyperbola3d_new_ok() {
        let h = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
        assert_eq!(h.center(), Point3D::ORIGIN);
        assert_eq!(h.major_radius(), 2.0);
        assert_eq!(h.minor_radius(), 1.0);
    }

    #[test]
    fn test_hyperbola3d_new_negative_major_radius_errors() {
        assert_eq!(
            Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, -2.0, 1.0),
            Err(HyperbolaConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_hyperbola3d_new_negative_minor_radius_errors() {
        assert_eq!(
            Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, -1.0),
            Err(HyperbolaConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_hyperbola3d_new_minor_greater_than_major_allowed() {
        let h = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0, 2.0).unwrap();
        assert_eq!(h.major_radius(), 1.0);
        assert_eq!(h.minor_radius(), 2.0);
    }

    #[test]
    fn test_hyperbola3d_new_null_normal_errors() {
        assert_eq!(
            Hyperbola3D::new(Point3D::ORIGIN, Vector3D::ZERO, Vector3D::X, 2.0, 1.0),
            Err(HyperbolaConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_hyperbola3d_new_parallel_x_direction_errors() {
        assert_eq!(
            Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::Z, 2.0, 1.0),
            Err(HyperbolaConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_hyperbola3d_from_frame_ok() {
        let h = Hyperbola3D::from_frame(Frame3D::WORLD, 2.0, 1.0).unwrap();
        assert_eq!(h.frame(), Frame3D::WORLD);
        assert_eq!(h.major_radius(), 2.0);
        assert_eq!(h.minor_radius(), 1.0);
    }

    #[test]
    fn test_hyperbola3d_from_frame_negative_major_radius_errors() {
        assert_eq!(
            Hyperbola3D::from_frame(Frame3D::WORLD, -2.0, 1.0),
            Err(HyperbolaConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_hyperbola3d_from_frame_negative_minor_radius_errors() {
        assert_eq!(
            Hyperbola3D::from_frame(Frame3D::WORLD, 2.0, -1.0),
            Err(HyperbolaConstructionError::NegativeRadius)
        );
    }

    #[test]
    fn test_hyperbola3d_from_center_and_points_ok() {
        let h = Hyperbola3D::from_center_and_points(
            Point3D::ORIGIN,
            Point3D::new(2.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
        )
        .unwrap();
        assert_eq!(h.major_radius(), 2.0);
        assert_eq!(h.minor_radius(), 1.0);
    }

    #[test]
    fn test_hyperbola3d_from_center_and_points_confused_points_errors() {
        assert_eq!(
            Hyperbola3D::from_center_and_points(
                Point3D::ORIGIN,
                Point3D::ORIGIN,
                Point3D::new(0.0, 1.0, 0.0),
            ),
            Err(HyperbolaConstructionError::ConfusedPoints)
        );
    }

    #[test]
    fn test_hyperbola3d_from_center_and_points_collinear_points_errors() {
        assert_eq!(
            Hyperbola3D::from_center_and_points(
                Point3D::ORIGIN,
                Point3D::new(2.0, 0.0, 0.0),
                Point3D::new(1.0, 0.0, 0.0),
            ),
            Err(HyperbolaConstructionError::CollinearPoints)
        );
    }

    // ---- evaluation ----

    #[test]
    fn test_hyperbola3d_eval_point_zero() {
        let h = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
        let p = h.eval_point(0.0);
        assert!((p.x - 2.0).abs() < 1e-9);
        assert!(p.y.abs() < 1e-9);
        assert!(p.z.abs() < 1e-9);
    }

    #[test]
    fn test_hyperbola3d_eval_points_matches_loop() {
        let h = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
        let us = [0.0, 0.5, 1.5];
        let expected: Vec<Point3D> = us.iter().map(|&u| h.eval_point(u)).collect();
        assert_eq!(h.eval_points(&us), expected);
    }

    #[test]
    fn test_hyperbola3d_eval_derivative_order1() {
        let h = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
        let d1 = h.eval_derivative(0.0, 1);
        assert!(d1.x.abs() < 1e-9);
        assert!((d1.y - 1.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic]
    fn test_hyperbola3d_eval_derivative_order0_panics() {
        let h = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
        h.eval_derivative(0.0, 0);
    }

    #[test]
    fn test_hyperbola3d_parameter_of_round_trip() {
        let h = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
        for u in [0.3, 2.0, -1.5] {
            let p = h.eval_point(u);
            assert!((h.parameter_of(p) - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_hyperbola3d_parameter_of_is_unbounded() {
        let h = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
        let p = h.eval_point(5.0);
        assert!((h.parameter_of(p) - 5.0).abs() < 1e-6);
    }

    // ---- HyperbolaConstructionError ----

    #[test]
    fn test_hyperbola_construction_error_display() {
        assert_eq!(
            HyperbolaConstructionError::NegativeRadius.to_string(),
            "radius is negative"
        );
        assert_eq!(
            HyperbolaConstructionError::NullNormal.to_string(),
            "normal has zero length"
        );
        assert_eq!(
            HyperbolaConstructionError::ConfusedPoints.to_string(),
            "the points are confused"
        );
        assert_eq!(
            HyperbolaConstructionError::CollinearPoints.to_string(),
            "the points are collinear"
        );
    }

    #[test]
    fn test_hyperbola_construction_error_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&HyperbolaConstructionError::NegativeRadius);
    }

    #[test]
    fn test_hyperbola3d_contains() {
        let h = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
        let tol = Tolerance::DEFAULT;
        assert!(h.contains(Point3D::new(2.0, 0.0, 0.0), tol));
        assert!(h.contains(h.eval_point(1.0), tol));
        assert!(!h.contains(Point3D::ORIGIN, tol));
        assert!(!h.contains(Point3D::new(2.0, 0.0, 5.0), tol));
    }

    #[test]
    fn test_hyperbola3d_project_point() {
        let h = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
        let tol = Tolerance::DEFAULT;
        // On the transverse axis right of the vertex: the true minima are
        // a symmetric pair off the vertex, closer than the vertex itself.
        let proj = h.project_point(Point3D::new(3.0, 0.0, 0.0), tol);
        assert!((proj.parameter.abs() - 0.62236).abs() < 1e-4);
        assert!((proj.distance - 0.8f64.sqrt()).abs() < 1e-9);
        // Off-curve: verify against brute-force sampling (which can only
        // overestimate the true minimum).
        let query = Point3D::new(2.0, 0.0, 5.0);
        let off = h.project_point(query, tol);
        let mut best = f64::INFINITY;
        for i in -1000..=1000 {
            let d = h.eval_point(i as f64 * 0.01).distance(query);
            best = best.min(d);
        }
        assert!(off.distance <= best);
        assert!(best - off.distance < 1e-3);
    }

    #[test]
    fn test_hyperbola3d_intersect_cone_crosscheck() {
        let tol = Tolerance::DEFAULT;
        let h = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
        let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
        match h.intersect_cone(&cone, tol) {
            ConicSurfaceIntersection::Hits(hits) => {
                assert!(!hits.is_empty());
                for h in &hits {
                    assert!(cone.contains(h.point, tol));
                }
                let generic = intersect_curve_surface(&h, &cone, tol);
                assert_eq!(generic.len(), hits.len());
            }
            _ => panic!("expected hits"),
        }
    }
}

#[cfg(test)]
mod dbg_hyp2 {
    use super::*;
    use crate::math::real_roots;
    #[test]
    fn dbg_roots() {
        let h = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
        let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
        let tol = Tolerance::DEFAULT;
        // Recompute the quartic exactly as intersect_cone does.
        let (ra, rb) = (h.major_radius, h.minor_radius);
        let ac = cone.frame().z_direction();
        let cos_phi = cone.semi_angle().cos();
        let c2 = cos_phi * cos_phi;
        let v = h.center() - cone.apex();
        let x = h.frame().x_direction();
        let y = h.frame().y_direction();
        let (xa, ya, va) = (x.dot(ac), y.dot(ac), v.dot(ac));
        let (xv, yv) = (x.dot(v), y.dot(v));
        let vv = v.dot(v);
        let p2 = ra * xa + rb * ya;
        let q2 = ra * xa - rb * ya;
        let f4 = ra * ra + rb * rb;
        let f3 = 4.0 * ra * xv + 4.0 * rb * yv;
        let f2 = 4.0 * vv + 2.0 * ra * ra - 2.0 * rb * rb;
        let f1 = 4.0 * ra * xv - 4.0 * rb * yv;
        let f0 = ra * ra + rb * rb;
        let h4 = p2 * p2;
        let h3 = 4.0 * va * p2;
        let h2 = 4.0 * va * va + 2.0 * p2 * q2;
        let h1 = 4.0 * va * q2;
        let h0 = q2 * q2;
        let g = [
            h4 - c2 * f4,
            h3 - c2 * f3,
            h2 - c2 * f2,
            h1 - c2 * f1,
            h0 - c2 * f0,
        ];
        eprintln!("g = {g:?}");
        // NOTE: intersect_cone passes [g4..g0] scrambled — see below.
        let roots = real_roots(&[g[4], g[3], g[2], g[1], g[0]], tol.confusion);
        eprintln!("roots(scrambled) = {roots:?}");
        let roots2 = real_roots(&[g[0], g[1], g[2], g[3], g[4]], tol.confusion);
        eprintln!("roots(asc) = {roots2:?}");
    }
}
