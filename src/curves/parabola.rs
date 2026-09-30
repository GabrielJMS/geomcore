//! Parabolas in 3D: parametric evaluation and parameter inversion, a thin
//! wrapper over [`crate::curve_math::analytic`].

use crate::curve_math::analytic;
use crate::curves::{Curve2D, ParametrizeError};
use crate::intersect::{
    ConicSurfaceIntersection, QuadraticSolution, solve_quadratic, verify_conic_hits,
};
use crate::math::real_roots;
use crate::projection::{self, CurveProjection};
use crate::surfaces::Surface;
use crate::{Cone, Cylinder, Frame3D, Plane, Point3D, Sphere, Tolerance, Vector3D};
use std::fmt;

/// Error returned when a [`Parabola3D`] cannot be constructed from the given
/// inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParabolaConstructionError {
    /// The requested focal distance is negative.
    NegativeFocal,
    /// The normal (or the x direction) has zero length, or the x direction
    /// is parallel to the normal.
    NullNormal,
}

impl fmt::Display for ParabolaConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            ParabolaConstructionError::NegativeFocal => "focal distance is negative",
            ParabolaConstructionError::NullNormal => "normal has zero length",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ParabolaConstructionError {}

/// A parabola in 3D: a plane [`Frame3D`] (origin at the apex, plus local x/y
/// directions defining the plane and the axis of symmetry) and a focal
/// distance, evaluated as
/// `apex + (u^2 / (4*focal))*x_dir + u*y_dir`.
///
/// # Examples
///
/// ```
/// use geomcore::{Parabola3D, Point3D, Vector3D};
/// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
/// assert_eq!(parabola.eval_point(2.0), Point3D::new(1.0, 2.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Parabola3D {
    frame: Frame3D,
    focal: f64,
}

impl Parabola3D {
    /// Creates a parabola from an apex, a normal, an x-axis hint, and a
    /// focal distance.
    ///
    /// The plane frame is derived from `normal` and `x_direction` via
    /// [`Frame3D::new`].
    ///
    /// # Errors
    ///
    /// Returns [`ParabolaConstructionError::NullNormal`] if `normal` cannot
    /// be normalized, if `x_direction` cannot be normalized, or if
    /// `x_direction` is parallel to `normal`; or
    /// [`ParabolaConstructionError::NegativeFocal`] if `focal < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Parabola3D, Point3D, Vector3D};
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// assert_eq!(parabola.focal(), 1.0);
    /// ```
    pub fn new(
        apex: Point3D,
        normal: Vector3D,
        x_direction: Vector3D,
        focal: f64,
    ) -> Result<Parabola3D, ParabolaConstructionError> {
        let frame = Frame3D::new(apex, normal, x_direction)
            .map_err(|_| ParabolaConstructionError::NullNormal)?;
        Parabola3D::from_frame(frame, focal)
    }

    /// Creates a parabola from a plane frame and a focal distance directly.
    ///
    /// # Errors
    ///
    /// Returns [`ParabolaConstructionError::NegativeFocal`] if `focal < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Parabola3D, Frame3D};
    /// let parabola = Parabola3D::from_frame(Frame3D::WORLD, 1.7).unwrap();
    /// assert_eq!(parabola.frame(), Frame3D::WORLD);
    /// ```
    pub fn from_frame(frame: Frame3D, focal: f64) -> Result<Parabola3D, ParabolaConstructionError> {
        if focal < 0.0 {
            return Err(ParabolaConstructionError::NegativeFocal);
        }
        Ok(Parabola3D { frame, focal })
    }

    /// Returns the apex of the parabola.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Parabola3D, Point3D, Vector3D};
    /// let parabola = Parabola3D::new(Point3D::new(1.0, 2.0, 3.0), Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// assert_eq!(parabola.apex(), Point3D::new(1.0, 2.0, 3.0));
    /// ```
    pub fn apex(&self) -> Point3D {
        self.frame.origin()
    }

    /// Returns the parabola's plane frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Parabola3D, Frame3D};
    /// let parabola = Parabola3D::from_frame(Frame3D::WORLD, 1.7).unwrap();
    /// assert_eq!(parabola.frame(), Frame3D::WORLD);
    /// ```
    pub fn frame(&self) -> Frame3D {
        self.frame
    }

    /// Returns the focal distance of the parabola.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Parabola3D, Frame3D};
    /// let parabola = Parabola3D::from_frame(Frame3D::WORLD, 1.7).unwrap();
    /// assert_eq!(parabola.focal(), 1.7);
    /// ```
    pub fn focal(&self) -> f64 {
        self.focal
    }

    /// Evaluates the point on the parabola at parameter `u`:
    /// `apex + (u^2 / (4*focal))*x_dir + u*y_dir`.
    ///
    /// A parabola with focal = 0 is degenerate; evaluation returns
    /// non-finite values.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Parabola3D, Point3D, Vector3D};
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// assert_eq!(parabola.eval_point(2.0), Point3D::new(1.0, 2.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64) -> Point3D {
        analytic::parabola_d0(&self.frame, self.focal, u)
    }

    /// Evaluates the points on the parabola at each parameter in `us`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Parabola3D, Point3D, Vector3D};
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// let points = parabola.eval_points(&[0.0, 2.0]);
    /// assert_eq!(points[0], Point3D::ORIGIN);
    /// ```
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        us.iter().map(|&u| self.eval_point(u)).collect()
    }

    /// Evaluates the derivative of the given `order` at parameter `u`.
    ///
    /// The parabola is a degree-2 polynomial curve: the first derivative is
    /// linear in `u`, the second is a constant, and every derivative of
    /// order above 2 is zero (see
    /// `curve_math::analytic::parabola_dn`).
    ///
    /// A parabola with focal = 0 is degenerate; evaluation returns
    /// non-finite values.
    ///
    /// # Panics
    ///
    /// Panics if `order == 0`; use [`Parabola3D::eval_point`] to evaluate
    /// the position itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Parabola3D, Point3D, Vector3D};
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// assert_eq!(parabola.eval_derivative(0.0, 1), Vector3D::Y);
    /// ```
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        match order {
            0 => panic!("eval_derivative: order must be >= 1 (use eval_point for order 0)"),
            _ => analytic::parabola_dn(&self.frame, self.focal, u, order),
        }
    }

    /// Recovers the parameter of a point on (or near) the parabola: the
    /// projection `(point - apex) . y_dir`. Unbounded (no wrapping).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Parabola3D, Point3D, Vector3D};
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// assert_eq!(parabola.parameter_of(Point3D::new(1.0, 2.0, 0.0)), 2.0);
    /// ```
    pub fn parameter_of(&self, point: Point3D) -> f64 {
        analytic::parabola_parameter(&self.frame, point)
    }

    /// Returns whether `point` lies on the parabola: the inverse
    /// parameter is recovered with [`Parabola3D::parameter_of`] and
    /// re-evaluated, and the point counts as contained when the
    /// re-evaluated point is within `tol.confusion` of it.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Parabola3D, Point3D, Tolerance, Vector3D};
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// assert!(parabola.contains(parabola.eval_point(2.0), tol));
    /// assert!(!parabola.contains(Point3D::new(0.0, 2.0, 5.0), tol));
    /// ```
    pub fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        let u = self.parameter_of(point);
        self.eval_point(u).distance(point) <= tol.confusion
    }

    /// All stationary points of the distance from `point` to the
    /// parabola, ordered by ascending distance.
    ///
    /// The stationarity condition is the cubic
    /// `t^3 + (8f^2 - 4f*px)t - 8f^2*py = 0` in the frame coordinates
    /// (solved by `real_roots`). The first entry is the global closest
    /// point (see [`Parabola3D::project_point`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Parabola3D, Point3D, Tolerance, Vector3D};
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// let extrema = parabola.extrema(Point3D::new(0.0, 2.0, 5.0), Tolerance::DEFAULT);
    /// assert!(!extrema.is_empty());
    /// ```
    pub fn extrema(&self, point: Point3D, tol: Tolerance) -> Vec<CurveProjection> {
        let rel = point - self.apex();
        let px = rel.dot(self.frame.x_direction());
        let py = rel.dot(self.frame.y_direction());
        let f = self.focal();
        let coeffs = [-8.0 * f * f * py, 8.0 * f * f - 4.0 * f * px, 0.0, 1.0];
        let mut out: Vec<CurveProjection> = real_roots(&coeffs, tol.confusion)
            .into_iter()
            .map(|r| {
                let distance = self.eval_point(r.value).distance(point);
                CurveProjection {
                    parameter: r.value,
                    distance: projection::snap_distance(distance, tol.confusion),
                }
            })
            .collect();
        out.sort_by(|x, y| x.distance.partial_cmp(&y.distance).unwrap());
        out
    }

    /// Projects `point` onto the parabola: the nearest of
    /// [`Parabola3D::extrema`]. Distances within `tol.confusion` snap to
    /// `0.0`, matching [`Parabola3D::contains`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Parabola3D, Point3D, Tolerance, Vector3D};
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// let proj = parabola.project_point(Point3D::new(1.0, 2.0, 0.0), Tolerance::DEFAULT);
    /// assert!((proj.parameter - 2.0).abs() < 1e-9);
    /// ```
    pub fn project_point(&self, point: Point3D, tol: Tolerance) -> CurveProjection {
        self.extrema(point, tol)
            .into_iter()
            .next()
            .expect("distance to a parabola attains its minimum")
    }

    /// Projects each point in `points` onto the parabola.
    ///
    /// Default-style batch wrapper over [`Parabola3D::project_point`]:
    /// one native call per batch, mirroring [`Parabola3D::eval_points`].
    pub fn project_points(&self, points: &[Point3D], tol: Tolerance) -> Vec<CurveProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }

    /// Plane coincident check: parallel normals and apex in the plane.
    fn plane_coincident(&self, plane: &Plane, tol: Tolerance) -> bool {
        let n = plane.normal();
        self.frame().z_direction().cross(n).magnitude() <= tol.angular
            && plane.contains(self.apex(), tol)
    }

    /// Intersects this parabola with a plane.
    ///
    /// Quadratic in the curve parameter, solved exactly; a
    /// coplanar-coincident plane holds the whole parabola.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{ConicSurfaceIntersection, Parabola3D, Plane, Point3D, Tolerance, Vector3D};
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match parabola.intersect_plane(&plane, tol) {
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
        let f = self.focal();
        // c*t^2 + d*t + e = 0 with E(t) = apex + (t^2/4f)X + tY.
        let c = x.dot(n) / (4.0 * f);
        let d = y.dot(n);
        let e = (self.apex() - plane.frame().origin()).dot(n);
        let roots = solve_quadratic(c, d, e, tol);
        let candidates: Vec<(f64, u32)> = match roots {
            QuadraticSolution::Two(t1, t2) => vec![(t1, 1), (t2, 1)],
            QuadraticSolution::One(t) => vec![(t, 2)],
            QuadraticSolution::Linear(t) => vec![(t, 1)],
            QuadraticSolution::Empty | QuadraticSolution::Degenerate => Vec::new(),
        };
        let hits = verify_conic_hits(self, plane, candidates, tol);
        if hits.is_empty() {
            ConicSurfaceIntersection::Empty
        } else {
            ConicSurfaceIntersection::Hits(hits)
        }
    }

    /// Intersects this parabola with a sphere; hits verified on the
    /// sphere. See [`Parabola3D::intersect_plane`] for the calling shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{ConicSurfaceIntersection, Parabola3D, Point3D, Sphere, Tolerance, Vector3D};
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match parabola.intersect_sphere(&sphere, tol) {
    ///     ConicSurfaceIntersection::Hits(hits) => assert_eq!(hits.len(), 2),
    ///     _ => panic!("expected hits"),
    /// }
    /// ```
    pub fn intersect_sphere(&self, sphere: &Sphere, tol: Tolerance) -> ConicSurfaceIntersection {
        let x = self.frame().x_direction();
        let y = self.frame().y_direction();
        let f = self.focal();
        let w = self.apex() - sphere.center();
        let r = sphere.radius();
        let q = [
            w.dot(w) - r * r,
            2.0 * y.dot(w),
            1.0 + x.dot(w) / (2.0 * f),
            0.0,
            1.0 / (16.0 * f * f),
        ];
        let candidates: Vec<(f64, u32)> = crate::math::real_roots(&q, tol.confusion)
            .into_iter()
            .map(|root| (root.value, root.multiplicity))
            .collect();
        let hits = verify_conic_hits(self, sphere, candidates, tol);
        if hits.is_empty() {
            ConicSurfaceIntersection::Empty
        } else {
            ConicSurfaceIntersection::Hits(hits)
        }
    }

    /// Intersects this parabola with a cylinder; hits verified on the
    /// cylinder (single nappe is automatic here). See
    /// [`Parabola3D::intersect_plane`] for the calling shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{ConicSurfaceIntersection, Cylinder, Parabola3D, Point3D, Tolerance, Vector3D};
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match parabola.intersect_cylinder(&cylinder, tol) {
    ///     ConicSurfaceIntersection::Hits(hits) => assert_eq!(hits.len(), 2),
    ///     _ => panic!("expected hits"),
    /// }
    /// ```
    pub fn intersect_cylinder(
        &self,
        cylinder: &Cylinder,
        tol: Tolerance,
    ) -> ConicSurfaceIntersection {
        let x = self.frame().x_direction();
        let y = self.frame().y_direction();
        let f = self.focal();
        let a = cylinder.axis().direction();
        let w = self.apex() - cylinder.axis().origin();
        let r = cylinder.radius();
        let (xa, ya, wa) = (x.dot(a), y.dot(a), w.dot(a));
        let (xw, yw) = (x.dot(w), y.dot(w));
        let ww = w.dot(w);
        let alpha = xa / (4.0 * f);
        let beta = ya;
        // |E - C0|^2 - axial^2 - rho^2, E - C0 = W + (t^2/4f)X + tY.
        let q = [
            ww - wa * wa - r * r,
            2.0 * yw - 2.0 * wa * beta,
            1.0 + xw / (2.0 * f) - (beta * beta + 2.0 * wa * alpha),
            -2.0 * alpha * beta,
            1.0 / (16.0 * f * f) - alpha * alpha,
        ];
        let candidates: Vec<(f64, u32)> = crate::math::real_roots(&q, tol.confusion)
            .into_iter()
            .map(|root| (root.value, root.multiplicity))
            .collect();
        let hits = verify_conic_hits(self, cylinder, candidates, tol);
        if hits.is_empty() {
            ConicSurfaceIntersection::Empty
        } else {
            ConicSurfaceIntersection::Hits(hits)
        }
    }

    /// Intersects this parabola with a cone; hits verified on the nappe.
    /// See [`Parabola3D::intersect_plane`] for the calling shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cone, ConicSurfaceIntersection, Frame3D, Parabola3D, Point3D, Tolerance, Vector3D};
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match parabola.intersect_cone(&cone, tol) {
    ///     ConicSurfaceIntersection::Hits(hits) => assert!(!hits.is_empty()),
    ///     _ => panic!("expected hits"),
    /// }
    /// ```
    pub fn intersect_cone(&self, cone: &Cone, tol: Tolerance) -> ConicSurfaceIntersection {
        let x = self.frame().x_direction();
        let y = self.frame().y_direction();
        let f = self.focal();
        let a = cone.frame().z_direction();
        let cos_phi = cone.semi_angle().cos();
        let c2 = cos_phi * cos_phi;
        let w = self.apex() - cone.apex();
        let (xa, ya, wa) = (x.dot(a), y.dot(a), w.dot(a));
        let (xw, yw) = (x.dot(w), y.dot(w));
        let ww = w.dot(w);
        let alpha = xa / (4.0 * f);
        let beta = ya;
        // m(t)^2 - c^2*|E - A|^2 with m = va + alpha*t^2 + beta*t.
        let va = wa;
        let q = [
            va * va - c2 * ww,
            2.0 * va * beta - c2 * 2.0 * yw,
            (2.0 * va * alpha + beta * beta) - c2 * (1.0 + xw / (2.0 * f)),
            2.0 * alpha * beta,
            alpha * alpha - c2 / (16.0 * f * f),
        ];
        let candidates: Vec<(f64, u32)> = crate::math::real_roots(&q, tol.confusion)
            .into_iter()
            .map(|root| (root.value, root.multiplicity))
            .collect();
        let hits = verify_conic_hits(self, cone, candidates, tol);
        if hits.is_empty() {
            ConicSurfaceIntersection::Empty
        } else {
            ConicSurfaceIntersection::Hits(hits)
        }
    }

    /// Computes the exact 2D representation of this parabola in a surface's
    /// parameter space.
    ///
    /// No parabola/surface pair has a closed-form 2D image in this release, so
    /// this always returns [`ParametrizeError::NotAnalytic`]. The `surface`
    /// argument is accepted for signature parity with the analytic
    /// [`crate::Line3D::parametrize_on`] and
    /// [`crate::Circle3D::parametrize_on`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::ParametrizeError;
    /// use geomcore::{Parabola3D, Plane, Point3D, Vector3D};
    ///
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// assert_eq!(parabola.parametrize_on(&plane), Err(ParametrizeError::NotAnalytic));
    /// ```
    pub fn parametrize_on(&self, surface: impl Into<Surface>) -> Result<Curve2D, ParametrizeError> {
        let _ = surface.into();
        Err(ParametrizeError::NotAnalytic)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ConicSurfaceIntersection, Cylinder, Frame3D, Parabola3D, ParabolaConstructionError, Plane,
        Point3D, Tolerance, Vector3D, intersect_curve_surface,
    };

    // ---- construction ----

    #[test]
    fn test_parabola3d_new_ok() {
        let p = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.7).unwrap();
        assert_eq!(p.apex(), Point3D::ORIGIN);
        assert_eq!(p.focal(), 1.7);
    }

    #[test]
    fn test_parabola3d_new_negative_focal_errors() {
        assert_eq!(
            Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, -1.0),
            Err(ParabolaConstructionError::NegativeFocal)
        );
    }

    #[test]
    fn test_parabola3d_new_zero_focal_allowed() {
        let p = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 0.0).unwrap();
        assert_eq!(p.focal(), 0.0);
    }

    #[test]
    fn test_parabola3d_new_null_normal_errors() {
        assert_eq!(
            Parabola3D::new(Point3D::ORIGIN, Vector3D::ZERO, Vector3D::X, 1.0),
            Err(ParabolaConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_parabola3d_new_parallel_x_direction_errors() {
        assert_eq!(
            Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::Z, 1.0),
            Err(ParabolaConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_parabola3d_from_frame_ok() {
        let p = Parabola3D::from_frame(Frame3D::WORLD, 1.7).unwrap();
        assert_eq!(p.frame(), Frame3D::WORLD);
        assert_eq!(p.focal(), 1.7);
    }

    #[test]
    fn test_parabola3d_from_frame_negative_focal_errors() {
        assert_eq!(
            Parabola3D::from_frame(Frame3D::WORLD, -0.1),
            Err(ParabolaConstructionError::NegativeFocal)
        );
    }

    // ---- evaluation ----

    #[test]
    fn test_parabola3d_eval_point_zero() {
        let p = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
        let pt = p.eval_point(0.0);
        assert!(pt.x.abs() < 1e-9);
        assert!(pt.y.abs() < 1e-9);
        assert!(pt.z.abs() < 1e-9);
    }

    #[test]
    fn test_parabola3d_eval_points_matches_loop() {
        let p = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.7).unwrap();
        let us = [0.0, 0.5, 1.5];
        let expected: Vec<Point3D> = us.iter().map(|&u| p.eval_point(u)).collect();
        assert_eq!(p.eval_points(&us), expected);
    }

    #[test]
    fn test_parabola3d_eval_derivative_order1() {
        let p = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
        let d1 = p.eval_derivative(0.0, 1);
        assert!(d1.x.abs() < 1e-9);
        assert!((d1.y - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_parabola3d_eval_derivative_order3_is_zero() {
        let p = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
        let d3 = p.eval_derivative(0.5, 3);
        assert_eq!(d3, Vector3D::ZERO);
    }

    #[test]
    #[should_panic]
    fn test_parabola3d_eval_derivative_order0_panics() {
        let p = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
        p.eval_derivative(0.0, 0);
    }

    #[test]
    fn test_parabola3d_parameter_of_round_trip() {
        let p = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.7).unwrap();
        for u in [0.3, 2.0, -5.5] {
            let pt = p.eval_point(u);
            assert!((p.parameter_of(pt) - u).abs() < 1e-9);
        }
    }

    #[test]
    fn test_parabola3d_parameter_of_is_unbounded() {
        let p = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.7).unwrap();
        let pt = p.eval_point(100.0);
        assert!((p.parameter_of(pt) - 100.0).abs() < 1e-6);
    }

    // ---- ParabolaConstructionError ----

    #[test]
    fn test_parabola_construction_error_display() {
        assert_eq!(
            ParabolaConstructionError::NegativeFocal.to_string(),
            "focal distance is negative"
        );
        assert_eq!(
            ParabolaConstructionError::NullNormal.to_string(),
            "normal has zero length"
        );
    }

    #[test]
    fn test_parabola_construction_error_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&ParabolaConstructionError::NegativeFocal);
    }

    #[test]
    fn test_parabola3d_contains() {
        let p = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
        let tol = Tolerance::DEFAULT;
        assert!(p.contains(p.eval_point(2.0), tol));
        assert!(p.contains(p.eval_point(-1.5), tol));
        assert!(!p.contains(Point3D::new(0.0, 2.0, 5.0), tol));
    }

    #[test]
    fn test_parabola3d_project_point() {
        let p = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
        let tol = Tolerance::DEFAULT;
        // (1,2,0) is on the curve (t = 2).
        let on = p.project_point(Point3D::new(1.0, 2.0, 0.0), tol);
        assert!((on.parameter - 2.0).abs() < 1e-9);
        assert_eq!(on.distance, 0.0);
        // Off-curve: verify against brute-force sampling (which can only
        // overestimate the true minimum).
        let query = Point3D::new(0.0, 2.0, 5.0);
        let proj = p.project_point(query, tol);
        let mut best = f64::INFINITY;
        for i in -1000..=1000 {
            let d = p.eval_point(i as f64 * 0.01).distance(query);
            best = best.min(d);
        }
        assert!(proj.distance <= best);
        assert!(best - proj.distance < 1e-3);
        assert!(p.contains(p.eval_point(proj.parameter), tol));
    }

    #[test]
    fn test_parabola3d_intersect_plane_coincident() {
        let tol = Tolerance::DEFAULT;
        let p = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        assert_eq!(
            p.intersect_plane(&plane, tol),
            ConicSurfaceIntersection::Coincident
        );
        // Parallel offset: miss.
        let miss = Plane::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Z).unwrap();
        assert_eq!(
            p.intersect_plane(&miss, tol),
            ConicSurfaceIntersection::Empty
        );
    }

    #[test]
    fn test_parabola3d_intersect_cylinder_crosscheck() {
        let tol = Tolerance::DEFAULT;
        let p = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
        match p.intersect_cylinder(&cylinder, tol) {
            ConicSurfaceIntersection::Hits(hits) => {
                assert!(!hits.is_empty());
                for h in &hits {
                    assert!(p.contains(h.point, tol));
                    assert!(cylinder.contains(h.point, tol));
                }
                let generic = intersect_curve_surface(&p, &cylinder, tol);
                assert_eq!(generic.len(), hits.len());
            }
            _ => panic!("expected hits"),
        }
    }
}
