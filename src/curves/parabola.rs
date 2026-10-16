//! Parabolas in 3D: parametric evaluation and parameter inversion, a thin
//! wrapper over [`crate::curve_math::analytic`].

use crate::Intersect;
use crate::curve_math::analytic;
use crate::curves::{Curve2D, ParametrizeError};
use crate::frame::frame_error;
use crate::intersect::{ConicSurfaceIntersection, intersect_conic_quadric};
use crate::math::real_roots;
use crate::projection::{self, CurveProjection};
use crate::surfaces::Surface;
use crate::{Cone, Cylinder, Frame3D, Plane, Point3D, Sphere, Tolerance, Transform, Vector3D};
use std::fmt;

/// Error returned when a [`Parabola3D`] cannot be constructed from the given
/// inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParabolaConstructionError {
    /// The requested focal distance is zero or negative (a zero focal
    /// distance collapses the parabola onto its axis).
    NonPositiveFocal,
    /// The normal (or the x direction) has zero length, or the x direction
    /// is parallel to the normal.
    NullNormal,
    /// An input coordinate, radius, or angle is NaN or infinite.
    NonFinite,
}

impl fmt::Display for ParabolaConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            ParabolaConstructionError::NonPositiveFocal => "focal distance is not positive",
            ParabolaConstructionError::NullNormal => "normal has zero length",
            ParabolaConstructionError::NonFinite => "an input is NaN or infinite",
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
    /// `x_direction` is parallel to `normal`;
    /// [`ParabolaConstructionError::NonPositiveFocal`] if `focal <= 0`; or
    /// [`ParabolaConstructionError::NonFinite`] if any input is NaN or
    /// infinite.
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
        let frame = Frame3D::new(apex, normal, x_direction).map_err(frame_error(
            ParabolaConstructionError::NonFinite,
            ParabolaConstructionError::NullNormal,
        ))?;
        Parabola3D::from_frame(frame, focal)
    }

    /// Creates a parabola from a plane frame and a focal distance directly.
    ///
    /// # Errors
    ///
    /// Returns [`ParabolaConstructionError::NonPositiveFocal`] if `focal <= 0`,
    /// or [`ParabolaConstructionError::NonFinite`] if it is NaN or infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Parabola3D, Frame3D};
    /// let parabola = Parabola3D::from_frame(Frame3D::WORLD, 1.7).unwrap();
    /// assert_eq!(parabola.frame(), Frame3D::WORLD);
    /// ```
    pub fn from_frame(frame: Frame3D, focal: f64) -> Result<Parabola3D, ParabolaConstructionError> {
        if !(focal.is_finite()) {
            return Err(ParabolaConstructionError::NonFinite);
        }
        if focal <= 0.0 {
            return Err(ParabolaConstructionError::NonPositiveFocal);
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

    /// Returns the image of this parabola under `transform`: the apex and
    /// frame directions map point-wise while the focal distance scales by
    /// the transform's uniform scale factor.
    ///
    /// Every frame axis maps through the transform, so the image's
    /// parametrization is the transform of this one: under a rotation,
    /// translation or mirror, `transformed(t).eval_point(u)` equals
    /// `t.apply_point(self.eval_point(u))`, and a mirror changes the
    /// frame's handedness. Fails with [`ParabolaConstructionError::NullNormal`]
    /// only when the transform collapses the frame (a zero scale factor).
    /// `u` is a length along the y direction, so under a scale `s` the
    /// image of the point at `u` sits at `s * u`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Parabola3D, Point3D, Transform, Vector3D};
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// let moved = parabola.transformed(Transform::translation(Vector3D::new(1.0, 0.0, 0.0))).unwrap();
    /// assert_eq!(moved.apex(), Point3D::new(1.0, 0.0, 0.0));
    /// assert_eq!(moved.focal(), 1.0);
    /// ```
    pub fn transformed(
        &self,
        transform: Transform,
    ) -> Result<Parabola3D, ParabolaConstructionError> {
        let (frame, scale) = transform.apply_frame(self.frame()).map_err(frame_error(
            ParabolaConstructionError::NonFinite,
            ParabolaConstructionError::NullNormal,
        ))?;
        Parabola3D::from_frame(frame, self.focal() * scale)
    }

    /// Evaluates the point on the parabola at parameter `u`:
    /// `apex + (u^2 / (4*focal))*x_dir + u*y_dir`.
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

    /// Returns whether `point` lies on the parabola: within `tol.confusion`
    /// of it, exactly when [`Parabola3D::project_point`] reports distance `0.0`.
    /// Its implicit equation decides most points at once: off the plane,
    /// clearly far from the curve or clearly on it, with rigorous bounds
    /// for a quadratic equation. Of the points left, within a few
    /// tolerances of the curve, the inverse parameter of
    /// [`Parabola3D::parameter_of`] (exact on the curve, cheap) accepts most; the
    /// projection settles the rest.
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
        let f = self.focal;
        // `y² = 4fx`: on it `|∇f| ≥ 4f`.
        let near =
            analytic::conic_proximity(&self.frame, point, tol.confusion, 2.0, 4.0 * f, |x, y| {
                (y * y - 4.0 * f * x, -4.0 * f, 2.0 * y)
            });
        if let Some(near) = near {
            return near;
        }
        let u = self.parameter_of(point);
        self.eval_point(u).distance(point) <= tol.confusion
            || self.project_point(point, tol).distance <= tol.confusion
    }

    /// All stationary points of the distance from `point` to the
    /// parabola, ordered by ascending distance.
    ///
    /// The stationarity condition is the cubic
    /// `t^3 + (8f^2 - 4f*px)t - 8f^2*py = 0` in the frame coordinates
    /// (solved by `real_roots`). The first entry is the global closest
    /// point (see [`Parabola3D::project_point`]).
    ///
    /// A query point with a non-finite coordinate has no extrema.
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
        if ![point.x, point.y, point.z].iter().all(|c| c.is_finite()) {
            return Vec::new();
        }
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
        out.sort_by(|x, y| x.distance.total_cmp(&y.distance));
        out
    }

    /// Projects `point` onto the parabola: the nearest of
    /// [`Parabola3D::extrema`]. Distances within `tol.confusion` snap to
    /// `0.0`, matching [`Parabola3D::contains`].
    ///
    /// A query point with a non-finite coordinate projects to a NaN
    /// parameter and distance.
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
            .unwrap_or(CurveProjection::UNDEFINED)
    }

    /// Projects each point in `points` onto the parabola.
    ///
    /// Default-style batch wrapper over [`Parabola3D::project_point`]:
    /// one native call per batch, mirroring [`Parabola3D::eval_points`].
    pub fn project_points(&self, points: &[Point3D], tol: Tolerance) -> Vec<CurveProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }

    /// Intersects with a [`Plane`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_plane(&self, plane: &Plane, tol: Tolerance) -> ConicSurfaceIntersection {
        self.intersect(plane, tol)
    }

    /// Intersects with a [`Sphere`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_sphere(&self, sphere: &Sphere, tol: Tolerance) -> ConicSurfaceIntersection {
        self.intersect(sphere, tol)
    }

    /// Intersects with a [`Cylinder`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_cylinder(
        &self,
        cylinder: &Cylinder,
        tol: Tolerance,
    ) -> ConicSurfaceIntersection {
        self.intersect(cylinder, tol)
    }

    /// Intersects with a [`Cone`]; see its [`Intersect`] implementation.
    #[deprecated(since = "0.29.0", note = "use `Intersect::intersect`")]
    pub fn intersect_cone(&self, cone: &Cone, tol: Tolerance) -> ConicSurfaceIntersection {
        self.intersect(cone, tol)
    }

    /// Computes the exact 2D representation of this parabola in a surface's
    /// parameter space.
    ///
    /// No parabola/surface pair has a closed-form 2D image in this release, so
    /// this always returns [`ParametrizeError::NotAnalytic`]. The `surface`
    /// and `tol` arguments are accepted for signature parity with the analytic
    /// [`crate::Line3D::parametrize_on`] and
    /// [`crate::Circle3D::parametrize_on`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::ParametrizeError;
    /// use geomcore::{Parabola3D, Plane, Point3D, Tolerance, Vector3D};
    ///
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// assert_eq!(parabola.parametrize_on(&plane, Tolerance::DEFAULT), Err(ParametrizeError::NotAnalytic));
    /// ```
    pub fn parametrize_on(
        &self,
        surface: impl Into<Surface>,
        _tol: Tolerance,
    ) -> Result<Curve2D, ParametrizeError> {
        let _ = surface.into();
        Err(ParametrizeError::NotAnalytic)
    }
}

impl Intersect<Plane> for Parabola3D {
    type Output = ConicSurfaceIntersection;

    /// Intersects this parabola with a plane.
    ///
    /// Quadratic in the curve parameter, solved exactly; a
    /// coplanar-coincident plane holds the whole parabola.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{ConicSurfaceIntersection, Parabola3D, Plane, Point3D, Tolerance, Vector3D};
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match parabola.intersect(&plane, tol) {
    ///     ConicSurfaceIntersection::Hits(hits) => assert_eq!(hits.len(), 1),
    ///     _ => panic!("expected hits"),
    /// }
    /// ```
    fn intersect(&self, plane: &Plane, tol: Tolerance) -> ConicSurfaceIntersection {
        intersect_conic_quadric(self, plane, tol)
    }
}

impl Intersect<Sphere> for Parabola3D {
    type Output = ConicSurfaceIntersection;

    /// Intersects this parabola with a sphere; hits verified on the
    /// sphere. See [`Parabola3D::intersect_plane`] for the calling shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{ConicSurfaceIntersection, Parabola3D, Point3D, Sphere, Tolerance, Vector3D};
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match parabola.intersect(&sphere, tol) {
    ///     ConicSurfaceIntersection::Hits(hits) => assert_eq!(hits.len(), 2),
    ///     _ => panic!("expected hits"),
    /// }
    /// ```
    fn intersect(&self, sphere: &Sphere, tol: Tolerance) -> ConicSurfaceIntersection {
        intersect_conic_quadric(self, sphere, tol)
    }
}

impl Intersect<Cylinder> for Parabola3D {
    type Output = ConicSurfaceIntersection;

    /// Intersects this parabola with a cylinder; hits verified on the
    /// cylinder. See
    /// [`Parabola3D::intersect_plane`] for the calling shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{ConicSurfaceIntersection, Cylinder, Parabola3D, Point3D, Tolerance, Vector3D};
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match parabola.intersect(&cylinder, tol) {
    ///     ConicSurfaceIntersection::Hits(hits) => assert_eq!(hits.len(), 2),
    ///     _ => panic!("expected hits"),
    /// }
    /// ```
    fn intersect(&self, cylinder: &Cylinder, tol: Tolerance) -> ConicSurfaceIntersection {
        intersect_conic_quadric(self, cylinder, tol)
    }
}

impl Intersect<Cone> for Parabola3D {
    type Output = ConicSurfaceIntersection;

    /// Intersects this parabola with a cone; hits verified on
    /// either nappe.
    /// See [`Parabola3D::intersect_plane`] for the calling shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Intersect;
    /// use geomcore::{Cone, ConicSurfaceIntersection, Frame3D, Parabola3D, Point3D, Tolerance, Vector3D};
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match parabola.intersect(&cone, tol) {
    ///     ConicSurfaceIntersection::Hits(hits) => assert!(!hits.is_empty()),
    ///     _ => panic!("expected hits"),
    /// }
    /// ```
    fn intersect(&self, cone: &Cone, tol: Tolerance) -> ConicSurfaceIntersection {
        intersect_conic_quadric(self, cone, tol)
    }
}

#[cfg(test)]
mod tests {
    use crate::Intersect;
    use crate::{
        Axis3D, ConicSurfaceIntersection, Cylinder, Frame3D, Parabola3D, ParabolaConstructionError,
        Plane, Point3D, Sphere, Tolerance, Transform, Vector3D, intersect_curve_surface,
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
            Err(ParabolaConstructionError::NonPositiveFocal)
        );
    }

    #[test]
    fn test_parabola3d_new_zero_focal_errors() {
        assert_eq!(
            Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 0.0),
            Err(ParabolaConstructionError::NonPositiveFocal)
        );
    }

    #[test]
    fn test_parabola3d_new_non_finite_errors() {
        for focal in [f64::NAN, f64::INFINITY] {
            assert_eq!(
                Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, focal),
                Err(ParabolaConstructionError::NonFinite)
            );
        }
        assert_eq!(
            Parabola3D::new(
                Point3D::new(f64::NAN, 0.0, 0.0),
                Vector3D::Z,
                Vector3D::X,
                1.0
            ),
            Err(ParabolaConstructionError::NonFinite)
        );
        assert_eq!(
            Parabola3D::new(
                Point3D::ORIGIN,
                Vector3D::Z,
                Vector3D::new(f64::INFINITY, 0.0, 0.0),
                1.0
            ),
            Err(ParabolaConstructionError::NonFinite)
        );
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
            Err(ParabolaConstructionError::NonPositiveFocal)
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
            ParabolaConstructionError::NonPositiveFocal.to_string(),
            "focal distance is not positive"
        );
        assert_eq!(
            ParabolaConstructionError::NullNormal.to_string(),
            "normal has zero length"
        );
    }

    #[test]
    fn test_parabola_construction_error_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&ParabolaConstructionError::NonPositiveFocal);
    }

    #[test]
    fn test_parabola3d_contains() {
        let p = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
        let tol = Tolerance::DEFAULT;
        assert!(p.contains(p.eval_point(2.0), tol));
        assert!(p.contains(p.eval_point(-1.5), tol));
        assert!(!p.contains(Point3D::new(0.0, 2.0, 5.0), tol));
        // Within tolerance along the normal far out on the arm: the
        // inverse parameter misses, but the closest point is 3e-8 away, so
        // `contains` must agree with `project_point`.
        let normal = p
            .eval_derivative(49.0, 1)
            .cross(Vector3D::Z)
            .normalized()
            .unwrap();
        let near = p.eval_point(49.0) + normal * 3e-8;
        assert_eq!(p.project_point(near, tol).distance, 0.0);
        assert!(p.contains(near, tol));
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
    fn test_parabola3d_apex_tangent_to_sphere() {
        // The apex sits on the sphere and the parabola opens into it: the
        // apex is a tangency (double root), which input rounding must not
        // split into nothing; the second case also crosses the sphere twice.
        let tol = Tolerance::DEFAULT;
        let cases = [
            (
                [-1.731231704419697, -0.7340958362561167, 4.433406474277298],
                [-0.9529764985851513, 0.1260554568656934, 0.2755826825814103],
                [0.3021731849933078, 0.4641631581607747, 0.8326127124162926],
                2.05398299873137,
                1.114835151487306,
                1,
            ),
            (
                [-3.686261961698878, 2.2461263633249473, 0.025839678441132108],
                [
                    -0.28025499704197765,
                    -0.9386923410260545,
                    -0.20078303098625241,
                ],
                [
                    -0.04967289504831476,
                    0.22306643447753607,
                    -0.9735368351053791,
                ],
                3.172852261703441,
                1.4836846217149213,
                3,
            ),
        ];
        for (o, z, x, r, focal, count) in cases {
            let centre = Point3D::new(o[0], o[1], o[2]);
            let normal = Vector3D::new(z[0], z[1], z[2]);
            let xdir = Vector3D::new(x[0], x[1], x[2]);
            let sphere =
                Sphere::from_frame(Frame3D::new(centre, normal, xdir).unwrap(), r).unwrap();
            let apex = centre - xdir * r;
            let parabola =
                Parabola3D::from_frame(Frame3D::new(apex, normal, xdir).unwrap(), focal).unwrap();
            match parabola.intersect(&sphere, tol) {
                ConicSurfaceIntersection::Hits(hits) => {
                    assert_eq!(hits.len(), count, "{hits:?}");
                    let touch = hits
                        .iter()
                        .find(|h| h.multiplicity == 2)
                        .unwrap_or_else(|| panic!("no tangent hit in {hits:?}"));
                    assert!(touch.parameter.abs() < 1e-6);
                    assert!(touch.point.distance(apex) < 1e-9);
                }
                other => panic!("expected hits, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_parabola3d_plane_far_root() {
        // The plane is nearly parallel to the parabola's axis: the second
        // crossing is about 3.4e9 from the origin, where the point's own
        // rounding exceeds the tolerance; it must still be reported.
        let tol = Tolerance::DEFAULT;
        let parabola = Parabola3D::from_frame(
            Frame3D::new(
                Point3D::new(-1.846934674952041, -3.3584204401499695, -2.27622906393242),
                Vector3D::new(
                    -0.06113599293430543,
                    0.9818338297888833,
                    -0.17962383207701216,
                ),
                Vector3D::new(
                    0.008163240668319863,
                    -0.1794625991638955,
                    -0.983730927135632,
                ),
            )
            .unwrap(),
            1.6472512486455413,
        )
        .unwrap();
        let plane = Plane::from_frame(
            Frame3D::new(
                Point3D::new(-0.7209761248595257, -0.9220441826499641, 1.8828489881612174),
                Vector3D::new(
                    -0.5111262109292593,
                    -0.8462850182163955,
                    0.15017211606544442,
                ),
                Vector3D::new(0.14562539970239027, 0.08692546117230952, 0.9855136768008349),
            )
            .unwrap(),
        );
        match parabola.intersect(&plane, tol) {
            ConicSurfaceIntersection::Hits(hits) => {
                assert_eq!(hits.len(), 2, "{hits:?}");
                assert!((hits[0].parameter + 3.576729279665239).abs() < 1e-9);
                assert!((hits[1].parameter / 148960.13911146074 - 1.0).abs() < 1e-9);
                let far = Point3D::new(27341838.29244079, -604367152.2404089, -3312810312.8581567);
                assert!(hits[1].point.distance(far) < 1e-6 * 3.4e9);
            }
            other => panic!("expected hits, got {other:?}"),
        }
    }

    #[test]
    fn test_parabola3d_intersect_plane_coincident() {
        let tol = Tolerance::DEFAULT;
        let p = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        assert_eq!(
            p.intersect(&plane, tol),
            ConicSurfaceIntersection::Coincident
        );
        // Parallel offset: miss.
        let miss = Plane::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Z).unwrap();
        assert_eq!(p.intersect(&miss, tol), ConicSurfaceIntersection::Empty);
    }

    #[test]
    fn test_parabola3d_intersect_cylinder_crosscheck() {
        let tol = Tolerance::DEFAULT;
        let p = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
        match p.intersect(&cylinder, tol) {
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

    // ---- Parabola3D transformed ----

    #[test]
    fn test_parabola3d_transformed_translation_spot_check() {
        let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
        let t = Transform::translation(Vector3D::new(1.0, -2.0, 3.0));
        let moved = parabola.transformed(t).unwrap();
        assert_eq!(moved.apex(), Point3D::new(1.0, -2.0, 3.0));
        assert_eq!(moved.focal(), 1.0);
        // Hand-computed: apex + (u^2 / 4) * X + u * Y, shifted.
        assert_eq!(moved.eval_point(2.0), Point3D::new(2.0, 0.0, 3.0));
    }

    #[test]
    fn test_parabola3d_transformed_round_trip_and_mirror() {
        use std::f64::consts::PI;
        let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0).unwrap();
        let axis = Axis3D::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let t = Transform::rotation(axis, PI / 2.0).then(Transform::scaling(Point3D::ORIGIN, 3.0));
        let moved = parabola.transformed(t).unwrap();
        assert_eq!(moved.focal(), 6.0);
        let tol = Tolerance::DEFAULT;
        for u in [0.0, 1.0, -2.5] {
            assert!(moved.contains(t.apply_point(parabola.eval_point(u)), tol));
        }
        let inv = Transform::scaling(Point3D::ORIGIN, 1.0 / 3.0)
            .then(Transform::rotation(axis, -PI / 2.0));
        let back = moved.transformed(inv).unwrap();
        for u in [0.0, 1.0, -2.5] {
            assert!(back.eval_point(u).distance(parabola.eval_point(u)) < 1e-9);
        }
        // Mirror across the xy-plane preserves the point set.
        let m = Transform::mirror_plane(Frame3D::WORLD);
        let mirrored = parabola.transformed(m).unwrap();
        for u in [0.0, 1.0, -2.5] {
            assert!(mirrored.contains(m.apply_point(parabola.eval_point(u)), tol));
        }
        assert_eq!(mirrored.transformed(m).unwrap(), parabola);
    }
}

#[cfg(test)]
mod projection_input_tests {
    use super::*;

    #[test]
    fn test_parabola3d_project_non_finite_point_is_nan() {
        let curve = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
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
