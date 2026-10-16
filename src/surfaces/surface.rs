//! [`ParametricSurface`] trait and the [`Surface`] enum adaptor unifying the
//! concrete analytic surface types.

use crate::curves::{BSplineCurve3D, Curve3D};
use crate::surfaces::{BSplineSurface, Cone, Cylinder, Plane, Sphere, SurfaceIsoError, Torus};
use crate::{Point3D, SurfaceProjection, Tolerance, Vector3D};
use std::f64::consts::{FRAC_PI_2, TAU};

/// Principal, mean and Gaussian curvatures at a surface point, with the
/// principal directions. See
/// [`ParametricSurface::curvatures`](ParametricSurface::curvatures).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceCurvatures {
    /// Smaller principal curvature.
    pub min_curvature: f64,
    /// Larger principal curvature.
    pub max_curvature: f64,
    /// Unit tangent direction of [`min_curvature`](Self::min_curvature).
    pub min_direction: Vector3D,
    /// Unit tangent direction of [`max_curvature`](Self::max_curvature),
    /// always perpendicular to [`min_direction`](Self::min_direction).
    pub max_direction: Vector3D,
    /// Mean curvature, `(min_curvature + max_curvature) / 2`.
    pub mean: f64,
    /// Gaussian curvature, `min_curvature * max_curvature`.
    pub gaussian: f64,
}

/// Common interface for parametric surfaces. Implement this to add a new
/// surface type that can be used generically (see [`Surface`] for a
/// heterogeneous-collection adaptor built on top of it).
///
/// Concrete surface types (e.g. [`Plane`], [`Sphere`]) already provide
/// inherent methods with the same names; their trait impls simply delegate
/// to those inherent methods, so calling code can use either form.
///
/// # Examples
///
/// The default [`eval_points`](ParametricSurface::eval_points) maps
/// [`eval_point`](ParametricSurface::eval_point) over each parameter pair:
///
/// ```
/// use geomcore::surfaces::ParametricSurface;
/// use geomcore::{Plane, Point3D, Vector3D};
///
/// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
/// let uvs = [(0.0, 0.0), (1.0, 2.0)];
/// let expected: Vec<Point3D> = uvs.iter().map(|&(u, v)| plane.eval_point(u, v)).collect();
/// assert_eq!(ParametricSurface::eval_points(&plane, &uvs), expected);
/// ```
pub trait ParametricSurface {
    /// Evaluates the point on the surface at parameters `(u, v)`.
    fn eval_point(&self, u: f64, v: f64) -> Point3D;

    /// Evaluates the derivative of order `(du, dv)` at parameters `(u, v)`.
    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3D;

    /// Evaluates the point and the first derivatives `[Su, Sv]` at
    /// parameters `(u, v)`.
    ///
    /// Default implementation: [`eval_point`](Self::eval_point) and
    /// [`eval_derivative`](Self::eval_derivative). B-spline surfaces
    /// override it to evaluate their basis functions once.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::surfaces::ParametricSurface;
    /// use geomcore::{Plane, Point3D, Vector3D};
    ///
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let (point, [su, sv]) = plane.eval_d1(1.0, 2.0);
    /// assert_eq!(point, Point3D::new(1.0, 2.0, 0.0));
    /// assert_eq!(su.cross(sv), Vector3D::Z);
    /// ```
    fn eval_d1(&self, u: f64, v: f64) -> (Point3D, [Vector3D; 2]) {
        (
            self.eval_point(u, v),
            [
                self.eval_derivative(u, v, 1, 0),
                self.eval_derivative(u, v, 0, 1),
            ],
        )
    }

    /// Evaluates the point, the first derivatives `[Su, Sv]` and the second
    /// derivatives `[Suu, Suv, Svv]` at parameters `(u, v)`.
    ///
    /// Default implementation: [`eval_point`](Self::eval_point) and
    /// [`eval_derivative`](Self::eval_derivative). B-spline surfaces
    /// override it to evaluate their basis functions once.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::surfaces::ParametricSurface;
    /// use geomcore::{Point3D, Sphere};
    ///
    /// let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    /// let (point, _, [suu, _, svv]) = sphere.eval_d2(0.0, 0.0);
    /// assert_eq!(point, Point3D::new(2.0, 0.0, 0.0));
    /// assert!((suu + (point - Point3D::ORIGIN)).magnitude() < 1e-15);
    /// assert!((svv + (point - Point3D::ORIGIN)).magnitude() < 1e-15);
    /// ```
    fn eval_d2(&self, u: f64, v: f64) -> (Point3D, [Vector3D; 2], [Vector3D; 3]) {
        (
            self.eval_point(u, v),
            [
                self.eval_derivative(u, v, 1, 0),
                self.eval_derivative(u, v, 0, 1),
            ],
            [
                self.eval_derivative(u, v, 2, 0),
                self.eval_derivative(u, v, 1, 1),
                self.eval_derivative(u, v, 0, 2),
            ],
        )
    }

    /// Returns the surface's `u` parameter bounds as `(first, last)`.
    /// Unbounded surfaces use `(f64::NEG_INFINITY, f64::INFINITY)`.
    fn u_bounds(&self) -> (f64, f64);

    /// Returns the surface's `v` parameter bounds as `(first, last)`.
    /// Unbounded surfaces use `(f64::NEG_INFINITY, f64::INFINITY)`.
    fn v_bounds(&self) -> (f64, f64);

    /// Returns the surface's period along `u`, or `None` if the surface is
    /// not periodic in `u`.
    fn u_period(&self) -> Option<f64>;

    /// Returns the surface's period along `v`, or `None` if the surface is
    /// not periodic in `v`.
    fn v_period(&self) -> Option<f64>;

    /// Principal, mean and Gaussian curvatures at parameters `(u, v)`,
    /// or `None` where the unit normal degenerates (vanishing or
    /// parallel partials, as at a sphere's poles or a cone's apex) or a
    /// derivative is not finite.
    ///
    /// Curvatures are measured along the unit normal `Su × Sv`: negative
    /// bends away from it, so an outward-normal sphere has `-1/radius`.
    /// The principal directions are always a unit, mutually perpendicular
    /// tangent pair. At an umbilical point (all directions principal, as
    /// on a sphere or a plane) the minimum direction lies along `Su`;
    /// near one, where the two curvatures agree to rounding, the pair is
    /// arbitrary but still perpendicular. The sign of each direction is
    /// not specified.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::surfaces::ParametricSurface;
    /// use geomcore::{Cylinder, Point3D, Tolerance, Vector3D};
    ///
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let c = ParametricSurface::curvatures(&cylinder, 0.0, 3.0, Tolerance::DEFAULT).unwrap();
    /// assert_eq!((c.min_curvature, c.max_curvature), (-0.5, 0.0));
    /// assert_eq!((c.mean, c.gaussian), (-0.25, 0.0));
    /// assert_eq!(c.min_direction, Vector3D::Y);
    /// assert_eq!(c.max_direction, Vector3D::Z);
    /// ```
    fn curvatures(&self, u: f64, v: f64, tol: Tolerance) -> Option<SurfaceCurvatures> {
        let (_, [su, sv], [suu, suv, svv]) = self.eval_d2(u, v);
        let (su_length, sv_length) = (su.magnitude(), sv.magnitude());
        let cross = su.cross(sv);
        let cross_length = cross.magnitude();
        // Same degeneracy test as the unit normal; the negated form also
        // rejects NaN derivatives.
        let regular = su_length > tol.confusion
            && sv_length > tol.confusion
            && cross_length > tol.angular * su_length * sv_length;
        if !regular {
            return None;
        }
        let normal = cross * (1.0 / cross_length);
        // Orthonormal tangent frame: `first` along Su, `second = normal ×
        // first = (|Su|² Sv − (Su·Sv) Su) / (|Su| |Su × Sv|)`. Each frame
        // vector is a parameter step `(du, dv)` mapped through the partials.
        let first = su * (1.0 / su_length);
        let second = normal.cross(first);
        let first_du = 1.0 / su_length;
        let second_du = -su.dot(sv) / (su_length * cross_length);
        let second_dv = su_length / cross_length;
        // Second fundamental form coefficients, then the shape operator in
        // the orthonormal frame: the symmetric matrix [[a, b], [b, c]].
        let (form_uu, form_uv, form_vv) = (suu.dot(normal), suv.dot(normal), svv.dot(normal));
        let a = form_uu * first_du * first_du;
        let b = first_du * (form_uu * second_du + form_uv * second_dv);
        let c = form_uu * second_du * second_du
            + 2.0 * form_uv * second_du * second_dv
            + form_vv * second_dv * second_dv;
        // Eigenvalues `mean ± half_gap`, free of the `H² − K` cancellation
        // that leaks a spurious gap of order √ε at umbilical points.
        let mean = 0.5 * (a + c);
        let half_gap = (0.5 * (a - c)).hypot(b);
        let (min_curvature, max_curvature) = (mean - half_gap, mean + half_gap);
        // Eigenvectors in frame coordinates, perpendicular by construction.
        // `offset = |a − c| / 2 + half_gap` sums non-negative terms, so it
        // carries no cancellation and vanishes only at an exact umbilic.
        let offset = 0.5 * (a - c).abs() + half_gap;
        let ((min_x, min_y), (max_x, max_y)) = if offset == 0.0 {
            ((1.0, 0.0), (0.0, 1.0))
        } else if a >= c {
            ((-b, offset), (offset, b))
        } else {
            ((offset, -b), (b, offset))
        };
        let scale = 1.0 / min_x.hypot(min_y);
        Some(SurfaceCurvatures {
            min_curvature,
            max_curvature,
            min_direction: (first * min_x + second * min_y) * scale,
            max_direction: (first * max_x + second * max_y) * scale,
            mean,
            gaussian: a * c - b * b,
        })
    }

    /// Evaluates the points on the surface at each `(u, v)` in `uvs`.
    ///
    /// Default implementation: maps [`eval_point`](Self::eval_point) over
    /// `uvs`.
    fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3D> {
        uvs.iter().map(|&(u, v)| self.eval_point(u, v)).collect()
    }

    /// Projects `point` onto the surface, returning the `(u, v)` parameters
    /// of the closest point and its distance.
    fn project_point(&self, point: Point3D, tol: Tolerance) -> SurfaceProjection;

    /// Returns whether `point` lies on the surface.
    fn contains(&self, point: Point3D, tol: Tolerance) -> bool;

    /// The surface as a B-spline, or `None` for any other surface.
    ///
    /// Numeric algorithms such as
    /// [`intersect_curve_surface`](crate::intersect_curve_surface) use it
    /// to work on the B-spline's polynomial patches. Default
    /// implementation: `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::surfaces::{BSplineSurface, ParametricSurface, Surface};
    /// use geomcore::{Plane, Point3D, Vector3D};
    ///
    /// let poles = vec![
    ///     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
    ///     vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 0.0)],
    /// ];
    /// let (knots, mults) = (vec![0.0, 1.0], vec![2, 2]);
    /// let patch =
    ///     BSplineSurface::new(1, 1, poles, knots.clone(), mults.clone(), knots, mults, false, false)
    ///         .unwrap();
    /// let surface = Surface::from(&patch);
    /// assert_eq!(surface.as_bspline(), Some(&patch));
    ///
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// assert_eq!(ParametricSurface::as_bspline(&plane), None);
    /// ```
    fn as_bspline(&self) -> Option<&BSplineSurface> {
        None
    }

    /// The surface as an elementary surface, or `None` for any other
    /// surface.
    ///
    /// Numeric algorithms such as
    /// [`intersect_curve_surface`](crate::intersect_curve_surface) use it
    /// to convert the surface to an exact B-spline when the curve is one.
    /// Default implementation: `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::surfaces::{ElementarySurface, ParametricSurface};
    /// use geomcore::{Point3D, Sphere};
    ///
    /// let sphere = Sphere::new(Point3D::ORIGIN, 1.0).unwrap();
    /// assert_eq!(
    ///     ParametricSurface::as_elementary(&sphere),
    ///     Some(ElementarySurface::Sphere(sphere))
    /// );
    /// ```
    fn as_elementary(&self) -> Option<ElementarySurface> {
        None
    }
}

impl ParametricSurface for Plane {
    fn eval_point(&self, u: f64, v: f64) -> Point3D {
        Plane::eval_point(self, u, v)
    }

    fn project_point(&self, point: Point3D, tol: Tolerance) -> SurfaceProjection {
        Plane::project_point(self, point, tol)
    }

    fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        Plane::contains(self, point, tol)
    }

    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3D {
        Plane::eval_derivative(self, u, v, du, dv)
    }

    fn u_bounds(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn v_bounds(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn u_period(&self) -> Option<f64> {
        None
    }

    fn v_period(&self) -> Option<f64> {
        None
    }

    fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3D> {
        Plane::eval_points(self, uvs)
    }

    fn as_elementary(&self) -> Option<ElementarySurface> {
        Some(ElementarySurface::Plane(*self))
    }
}

impl ParametricSurface for Cylinder {
    fn eval_point(&self, u: f64, v: f64) -> Point3D {
        Cylinder::eval_point(self, u, v)
    }

    fn project_point(&self, point: Point3D, tol: Tolerance) -> SurfaceProjection {
        Cylinder::project_point(self, point, tol)
    }

    fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        Cylinder::contains(self, point, tol)
    }

    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3D {
        Cylinder::eval_derivative(self, u, v, du, dv)
    }

    fn u_bounds(&self) -> (f64, f64) {
        (0.0, TAU)
    }

    fn v_bounds(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn u_period(&self) -> Option<f64> {
        Some(TAU)
    }

    fn v_period(&self) -> Option<f64> {
        None
    }

    fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3D> {
        Cylinder::eval_points(self, uvs)
    }

    fn as_elementary(&self) -> Option<ElementarySurface> {
        Some(ElementarySurface::Cylinder(*self))
    }
}

impl ParametricSurface for Cone {
    fn eval_point(&self, u: f64, v: f64) -> Point3D {
        Cone::eval_point(self, u, v)
    }

    fn project_point(&self, point: Point3D, tol: Tolerance) -> SurfaceProjection {
        Cone::project_point(self, point, tol)
    }

    fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        Cone::contains(self, point, tol)
    }

    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3D {
        Cone::eval_derivative(self, u, v, du, dv)
    }

    fn u_bounds(&self) -> (f64, f64) {
        (0.0, TAU)
    }

    fn v_bounds(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn u_period(&self) -> Option<f64> {
        Some(TAU)
    }

    fn v_period(&self) -> Option<f64> {
        None
    }

    fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3D> {
        Cone::eval_points(self, uvs)
    }

    fn as_elementary(&self) -> Option<ElementarySurface> {
        Some(ElementarySurface::Cone(*self))
    }
}

impl ParametricSurface for Sphere {
    fn eval_point(&self, u: f64, v: f64) -> Point3D {
        Sphere::eval_point(self, u, v)
    }

    fn project_point(&self, point: Point3D, tol: Tolerance) -> SurfaceProjection {
        Sphere::project_point(self, point, tol)
    }

    fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        Sphere::contains(self, point, tol)
    }

    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3D {
        Sphere::eval_derivative(self, u, v, du, dv)
    }

    fn u_bounds(&self) -> (f64, f64) {
        (0.0, TAU)
    }

    fn v_bounds(&self) -> (f64, f64) {
        (-FRAC_PI_2, FRAC_PI_2)
    }

    fn u_period(&self) -> Option<f64> {
        Some(TAU)
    }

    fn v_period(&self) -> Option<f64> {
        None
    }

    fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3D> {
        Sphere::eval_points(self, uvs)
    }

    fn as_elementary(&self) -> Option<ElementarySurface> {
        Some(ElementarySurface::Sphere(*self))
    }
}

impl ParametricSurface for Torus {
    fn eval_point(&self, u: f64, v: f64) -> Point3D {
        Torus::eval_point(self, u, v)
    }

    fn project_point(&self, point: Point3D, tol: Tolerance) -> SurfaceProjection {
        Torus::project_point(self, point, tol)
    }

    fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        Torus::contains(self, point, tol)
    }

    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3D {
        Torus::eval_derivative(self, u, v, du, dv)
    }

    fn u_bounds(&self) -> (f64, f64) {
        (0.0, TAU)
    }

    fn v_bounds(&self) -> (f64, f64) {
        (0.0, TAU)
    }

    fn u_period(&self) -> Option<f64> {
        Some(TAU)
    }

    fn v_period(&self) -> Option<f64> {
        Some(TAU)
    }

    fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3D> {
        Torus::eval_points(self, uvs)
    }

    fn as_elementary(&self) -> Option<ElementarySurface> {
        Some(ElementarySurface::Torus(*self))
    }
}

/// Enum adaptor unifying the concrete analytic surface types behind a
/// single value type, so callers can build a heterogeneous collection of
/// surfaces and evaluate them uniformly through [`ParametricSurface`].
///
/// Marked `#[non_exhaustive]`: future tasks add further variants (e.g. a
/// B-spline surface) without that being a breaking change.
///
/// # Examples
///
/// ```
/// use geomcore::surfaces::{ParametricSurface, Surface};
/// use geomcore::{Plane, Point3D, Sphere, Vector3D};
///
/// let surfaces: Vec<Surface> = vec![
///     Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap().into(),
///     Sphere::new(Point3D::ORIGIN, 2.0).unwrap().into(),
/// ];
/// let points: Vec<Point3D> = surfaces.iter().map(|s| s.eval_point(0.0, 0.0)).collect();
/// assert_eq!(points[0], Point3D::ORIGIN);
/// assert_eq!(points[1], Point3D::new(2.0, 0.0, 0.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Surface {
    /// A [`Plane`] surface.
    Plane(Plane),
    /// A [`Cylinder`] surface.
    Cylinder(Cylinder),
    /// A [`Cone`] surface.
    Cone(Cone),
    /// A [`Sphere`] surface.
    Sphere(Sphere),
    /// A [`Torus`] surface.
    Torus(Torus),
    /// A [`BSplineSurface`] (tensor-product NURBS) surface.
    BSpline(BSplineSurface),
}

impl Surface {
    /// Marching intersection with another surface: traces the meeting
    /// curves numerically and fits [`BSplineCurve3D`] approximants.
    ///
    /// See [`marching_intersection`](crate::marching_intersection) for the
    /// algorithm, scope, and approximation guarantees.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Plane, Point3D, Surface, Tolerance, Vector3D};
    /// let xy = Surface::from(Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap());
    /// let zy = Surface::from(Plane::new(Point3D::ORIGIN, Vector3D::X).unwrap());
    /// let curves = xy.intersect_marching(&zy, Tolerance::DEFAULT);
    /// assert_eq!(curves.len(), 1);
    /// ```
    pub fn intersect_marching(&self, other: &Surface, tol: Tolerance) -> Vec<BSplineCurve3D> {
        crate::marching_intersection(self, other, tol)
    }

    /// The `u` isoparametric curve at `u`: `u` fixed, running along `v`,
    /// so `u_iso(u).eval_point(t)` is `eval_point(u, t)` for every `t`.
    /// A line on a plane, cylinder or cone, a circle on a sphere or torus,
    /// a B-spline curve on a B-spline surface; see each type's `u_iso`.
    ///
    /// # Errors
    ///
    /// Returns [`SurfaceIsoError::NonFinite`] for a NaN or infinite `u`,
    /// and [`SurfaceIsoError::OutOfRange`] for a `u` outside the bounds of
    /// a non-periodic B-spline direction.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Curve3D, Cylinder, Point3D, Surface, Vector3D};
    /// let cylinder = Surface::from(Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap());
    /// assert!(matches!(cylinder.u_iso(0.0), Ok(Curve3D::Line(_))));
    /// assert!(matches!(cylinder.v_iso(0.0), Ok(Curve3D::Circle(_))));
    /// ```
    pub fn u_iso(&self, u: f64) -> Result<Curve3D, SurfaceIsoError> {
        Ok(match self {
            Surface::Plane(s) => Curve3D::Line(s.u_iso(u)?),
            Surface::Cylinder(s) => Curve3D::Line(s.u_iso(u)?),
            Surface::Cone(s) => Curve3D::Line(s.u_iso(u)?),
            Surface::Sphere(s) => Curve3D::Circle(s.u_iso(u)?),
            Surface::Torus(s) => Curve3D::Circle(s.u_iso(u)?),
            Surface::BSpline(s) => Curve3D::BSpline(s.u_iso(u)?),
        })
    }

    /// The `v` isoparametric curve at `v`: `v` fixed, running along `u`,
    /// so `v_iso(v).eval_point(t)` is `eval_point(t, v)` for every `t`.
    /// A line on a plane, a circle on a cylinder, cone, sphere or torus, a
    /// B-spline curve on a B-spline surface; see each type's `v_iso`.
    ///
    /// # Errors
    ///
    /// As [`Surface::u_iso`], for `v`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Curve3D, Plane, Point3D, Surface, Vector3D};
    /// let plane = Surface::from(Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap());
    /// assert!(matches!(plane.v_iso(1.0), Ok(Curve3D::Line(_))));
    /// ```
    pub fn v_iso(&self, v: f64) -> Result<Curve3D, SurfaceIsoError> {
        Ok(match self {
            Surface::Plane(s) => Curve3D::Line(s.v_iso(v)?),
            Surface::Cylinder(s) => Curve3D::Circle(s.v_iso(v)?),
            Surface::Cone(s) => Curve3D::Circle(s.v_iso(v)?),
            Surface::Sphere(s) => Curve3D::Circle(s.v_iso(v)?),
            Surface::Torus(s) => Curve3D::Circle(s.v_iso(v)?),
            Surface::BSpline(s) => Curve3D::BSpline(s.v_iso(v)?),
        })
    }
}

impl ParametricSurface for Surface {
    fn eval_point(&self, u: f64, v: f64) -> Point3D {
        match self {
            Surface::Plane(s) => s.eval_point(u, v),
            Surface::Cylinder(s) => s.eval_point(u, v),
            Surface::Cone(s) => s.eval_point(u, v),
            Surface::Sphere(s) => s.eval_point(u, v),
            Surface::Torus(s) => s.eval_point(u, v),
            Surface::BSpline(s) => s.eval_point(u, v),
        }
    }

    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3D {
        match self {
            Surface::Plane(s) => s.eval_derivative(u, v, du, dv),
            Surface::Cylinder(s) => s.eval_derivative(u, v, du, dv),
            Surface::Cone(s) => s.eval_derivative(u, v, du, dv),
            Surface::Sphere(s) => s.eval_derivative(u, v, du, dv),
            Surface::Torus(s) => s.eval_derivative(u, v, du, dv),
            Surface::BSpline(s) => s.eval_derivative(u, v, du, dv),
        }
    }

    fn eval_d1(&self, u: f64, v: f64) -> (Point3D, [Vector3D; 2]) {
        match self {
            Surface::Plane(s) => ParametricSurface::eval_d1(s, u, v),
            Surface::Cylinder(s) => ParametricSurface::eval_d1(s, u, v),
            Surface::Cone(s) => ParametricSurface::eval_d1(s, u, v),
            Surface::Sphere(s) => ParametricSurface::eval_d1(s, u, v),
            Surface::Torus(s) => ParametricSurface::eval_d1(s, u, v),
            Surface::BSpline(s) => s.eval_d1(u, v),
        }
    }

    fn eval_d2(&self, u: f64, v: f64) -> (Point3D, [Vector3D; 2], [Vector3D; 3]) {
        match self {
            Surface::Plane(s) => ParametricSurface::eval_d2(s, u, v),
            Surface::Cylinder(s) => ParametricSurface::eval_d2(s, u, v),
            Surface::Cone(s) => ParametricSurface::eval_d2(s, u, v),
            Surface::Sphere(s) => ParametricSurface::eval_d2(s, u, v),
            Surface::Torus(s) => ParametricSurface::eval_d2(s, u, v),
            Surface::BSpline(s) => s.eval_d2(u, v),
        }
    }

    fn u_bounds(&self) -> (f64, f64) {
        match self {
            Surface::Plane(s) => s.u_bounds(),
            Surface::Cylinder(s) => s.u_bounds(),
            Surface::Cone(s) => s.u_bounds(),
            Surface::Sphere(s) => s.u_bounds(),
            Surface::Torus(s) => s.u_bounds(),
            Surface::BSpline(s) => s.u_bounds(),
        }
    }

    fn v_bounds(&self) -> (f64, f64) {
        match self {
            Surface::Plane(s) => s.v_bounds(),
            Surface::Cylinder(s) => s.v_bounds(),
            Surface::Cone(s) => s.v_bounds(),
            Surface::Sphere(s) => s.v_bounds(),
            Surface::Torus(s) => s.v_bounds(),
            Surface::BSpline(s) => s.v_bounds(),
        }
    }

    fn u_period(&self) -> Option<f64> {
        match self {
            Surface::Plane(s) => s.u_period(),
            Surface::Cylinder(s) => s.u_period(),
            Surface::Cone(s) => s.u_period(),
            Surface::Sphere(s) => s.u_period(),
            Surface::Torus(s) => s.u_period(),
            Surface::BSpline(s) => s.u_period(),
        }
    }

    fn v_period(&self) -> Option<f64> {
        match self {
            Surface::Plane(s) => s.v_period(),
            Surface::Cylinder(s) => s.v_period(),
            Surface::Cone(s) => s.v_period(),
            Surface::Sphere(s) => s.v_period(),
            Surface::Torus(s) => s.v_period(),
            Surface::BSpline(s) => s.v_period(),
        }
    }

    fn project_point(&self, point: Point3D, tol: Tolerance) -> SurfaceProjection {
        match self {
            Surface::Plane(s) => s.project_point(point, tol),
            Surface::Cylinder(s) => s.project_point(point, tol),
            Surface::Cone(s) => s.project_point(point, tol),
            Surface::Sphere(s) => s.project_point(point, tol),
            Surface::Torus(s) => s.project_point(point, tol),
            Surface::BSpline(s) => s.project_point(point, tol),
        }
    }

    fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        match self {
            Surface::Plane(s) => s.contains(point, tol),
            Surface::Cylinder(s) => s.contains(point, tol),
            Surface::Cone(s) => s.contains(point, tol),
            Surface::Sphere(s) => s.contains(point, tol),
            Surface::Torus(s) => s.contains(point, tol),
            Surface::BSpline(s) => s.contains(point, tol),
        }
    }

    fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3D> {
        match self {
            Surface::Plane(s) => s.eval_points(uvs),
            Surface::Cylinder(s) => s.eval_points(uvs),
            Surface::Cone(s) => s.eval_points(uvs),
            Surface::Sphere(s) => s.eval_points(uvs),
            Surface::Torus(s) => s.eval_points(uvs),
            Surface::BSpline(s) => s.eval_points(uvs),
        }
    }

    fn curvatures(&self, u: f64, v: f64, tol: Tolerance) -> Option<SurfaceCurvatures> {
        match self {
            Surface::Plane(s) => ParametricSurface::curvatures(s, u, v, tol),
            Surface::Cylinder(s) => ParametricSurface::curvatures(s, u, v, tol),
            Surface::Cone(s) => ParametricSurface::curvatures(s, u, v, tol),
            Surface::Sphere(s) => ParametricSurface::curvatures(s, u, v, tol),
            Surface::Torus(s) => ParametricSurface::curvatures(s, u, v, tol),
            Surface::BSpline(s) => ParametricSurface::curvatures(s, u, v, tol),
        }
    }

    fn as_bspline(&self) -> Option<&BSplineSurface> {
        match self {
            Surface::BSpline(s) => Some(s),
            _ => None,
        }
    }

    fn as_elementary(&self) -> Option<ElementarySurface> {
        Surface::as_elementary(self)
    }
}

impl From<Plane> for Surface {
    /// Wraps a [`Plane`] as a [`Surface::Plane`].
    fn from(surface: Plane) -> Surface {
        Surface::Plane(surface)
    }
}

impl From<&Plane> for Surface {
    /// Copies a [`Plane`] reference into a [`Surface::Plane`].
    fn from(surface: &Plane) -> Surface {
        Surface::Plane(*surface)
    }
}

impl From<Cylinder> for Surface {
    /// Wraps a [`Cylinder`] as a [`Surface::Cylinder`].
    fn from(surface: Cylinder) -> Surface {
        Surface::Cylinder(surface)
    }
}

impl From<&Cylinder> for Surface {
    /// Copies a [`Cylinder`] reference into a [`Surface::Cylinder`].
    fn from(surface: &Cylinder) -> Surface {
        Surface::Cylinder(*surface)
    }
}

impl From<Cone> for Surface {
    /// Wraps a [`Cone`] as a [`Surface::Cone`].
    fn from(surface: Cone) -> Surface {
        Surface::Cone(surface)
    }
}

impl From<&Cone> for Surface {
    /// Copies a [`Cone`] reference into a [`Surface::Cone`].
    fn from(surface: &Cone) -> Surface {
        Surface::Cone(*surface)
    }
}

impl From<Sphere> for Surface {
    /// Wraps a [`Sphere`] as a [`Surface::Sphere`].
    fn from(surface: Sphere) -> Surface {
        Surface::Sphere(surface)
    }
}

impl From<&Sphere> for Surface {
    /// Copies a [`Sphere`] reference into a [`Surface::Sphere`].
    fn from(surface: &Sphere) -> Surface {
        Surface::Sphere(*surface)
    }
}

impl From<Torus> for Surface {
    /// Wraps a [`Torus`] as a [`Surface::Torus`].
    fn from(surface: Torus) -> Surface {
        Surface::Torus(surface)
    }
}

impl From<&Torus> for Surface {
    /// Copies a [`Torus`] reference into a [`Surface::Torus`].
    fn from(surface: &Torus) -> Surface {
        Surface::Torus(*surface)
    }
}

impl From<BSplineSurface> for Surface {
    /// Wraps a [`BSplineSurface`] as a [`Surface::BSpline`].
    fn from(surface: BSplineSurface) -> Surface {
        Surface::BSpline(surface)
    }
}

impl From<&BSplineSurface> for Surface {
    /// Clones a [`BSplineSurface`] reference into a [`Surface::BSpline`].
    ///
    /// Unlike the analytic surfaces (which are `Copy`), a [`BSplineSurface`]
    /// owns its pole and knot buffers, so this performs a deep clone.
    fn from(surface: &BSplineSurface) -> Surface {
        Surface::BSpline(surface.clone())
    }
}

/// An elementary surface: a plane, cylinder, cone, sphere or torus, the
/// surfaces with a closed-form definition.
///
/// Unlike [`Surface`] it is `Copy` and owns no heap memory.
///
/// # Examples
///
/// ```
/// use geomcore::surfaces::{ElementarySurface, Surface};
/// use geomcore::{Plane, Point3D, Vector3D};
///
/// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
/// let surface = Surface::from(plane);
/// assert_eq!(surface.as_elementary(), Some(ElementarySurface::Plane(plane)));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElementarySurface {
    /// A [`Plane`] surface.
    Plane(Plane),
    /// A [`Cylinder`] surface.
    Cylinder(Cylinder),
    /// A [`Cone`] surface.
    Cone(Cone),
    /// A [`Sphere`] surface.
    Sphere(Sphere),
    /// A [`Torus`] surface.
    Torus(Torus),
}

impl Surface {
    /// The surface as an [`ElementarySurface`], or `None` for a B-spline.
    pub fn as_elementary(&self) -> Option<ElementarySurface> {
        match self {
            Surface::Plane(s) => Some(ElementarySurface::Plane(*s)),
            Surface::Cylinder(s) => Some(ElementarySurface::Cylinder(*s)),
            Surface::Cone(s) => Some(ElementarySurface::Cone(*s)),
            Surface::Sphere(s) => Some(ElementarySurface::Sphere(*s)),
            Surface::Torus(s) => Some(ElementarySurface::Torus(*s)),
            Surface::BSpline(_) => None,
        }
    }
}

impl ParametricSurface for ElementarySurface {
    fn eval_point(&self, u: f64, v: f64) -> Point3D {
        match self {
            ElementarySurface::Plane(s) => s.eval_point(u, v),
            ElementarySurface::Cylinder(s) => s.eval_point(u, v),
            ElementarySurface::Cone(s) => s.eval_point(u, v),
            ElementarySurface::Sphere(s) => s.eval_point(u, v),
            ElementarySurface::Torus(s) => s.eval_point(u, v),
        }
    }

    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3D {
        match self {
            ElementarySurface::Plane(s) => s.eval_derivative(u, v, du, dv),
            ElementarySurface::Cylinder(s) => s.eval_derivative(u, v, du, dv),
            ElementarySurface::Cone(s) => s.eval_derivative(u, v, du, dv),
            ElementarySurface::Sphere(s) => s.eval_derivative(u, v, du, dv),
            ElementarySurface::Torus(s) => s.eval_derivative(u, v, du, dv),
        }
    }

    fn u_bounds(&self) -> (f64, f64) {
        match self {
            ElementarySurface::Plane(s) => s.u_bounds(),
            ElementarySurface::Cylinder(s) => s.u_bounds(),
            ElementarySurface::Cone(s) => s.u_bounds(),
            ElementarySurface::Sphere(s) => s.u_bounds(),
            ElementarySurface::Torus(s) => s.u_bounds(),
        }
    }

    fn v_bounds(&self) -> (f64, f64) {
        match self {
            ElementarySurface::Plane(s) => s.v_bounds(),
            ElementarySurface::Cylinder(s) => s.v_bounds(),
            ElementarySurface::Cone(s) => s.v_bounds(),
            ElementarySurface::Sphere(s) => s.v_bounds(),
            ElementarySurface::Torus(s) => s.v_bounds(),
        }
    }

    fn u_period(&self) -> Option<f64> {
        match self {
            ElementarySurface::Plane(s) => s.u_period(),
            ElementarySurface::Cylinder(s) => s.u_period(),
            ElementarySurface::Cone(s) => s.u_period(),
            ElementarySurface::Sphere(s) => s.u_period(),
            ElementarySurface::Torus(s) => s.u_period(),
        }
    }

    fn v_period(&self) -> Option<f64> {
        match self {
            ElementarySurface::Plane(s) => s.v_period(),
            ElementarySurface::Cylinder(s) => s.v_period(),
            ElementarySurface::Cone(s) => s.v_period(),
            ElementarySurface::Sphere(s) => s.v_period(),
            ElementarySurface::Torus(s) => s.v_period(),
        }
    }

    fn project_point(&self, point: Point3D, tol: Tolerance) -> SurfaceProjection {
        match self {
            ElementarySurface::Plane(s) => s.project_point(point, tol),
            ElementarySurface::Cylinder(s) => s.project_point(point, tol),
            ElementarySurface::Cone(s) => s.project_point(point, tol),
            ElementarySurface::Sphere(s) => s.project_point(point, tol),
            ElementarySurface::Torus(s) => s.project_point(point, tol),
        }
    }

    fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        match self {
            ElementarySurface::Plane(s) => s.contains(point, tol),
            ElementarySurface::Cylinder(s) => s.contains(point, tol),
            ElementarySurface::Cone(s) => s.contains(point, tol),
            ElementarySurface::Sphere(s) => s.contains(point, tol),
            ElementarySurface::Torus(s) => s.contains(point, tol),
        }
    }

    fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3D> {
        match self {
            ElementarySurface::Plane(s) => s.eval_points(uvs),
            ElementarySurface::Cylinder(s) => s.eval_points(uvs),
            ElementarySurface::Cone(s) => s.eval_points(uvs),
            ElementarySurface::Sphere(s) => s.eval_points(uvs),
            ElementarySurface::Torus(s) => s.eval_points(uvs),
        }
    }

    fn as_elementary(&self) -> Option<ElementarySurface> {
        Some(*self)
    }
}

impl From<ElementarySurface> for Surface {
    /// Wraps an [`ElementarySurface`] as the matching [`Surface`] variant.
    fn from(surface: ElementarySurface) -> Surface {
        match surface {
            ElementarySurface::Plane(s) => Surface::Plane(s),
            ElementarySurface::Cylinder(s) => Surface::Cylinder(s),
            ElementarySurface::Cone(s) => Surface::Cone(s),
            ElementarySurface::Sphere(s) => Surface::Sphere(s),
            ElementarySurface::Torus(s) => Surface::Torus(s),
        }
    }
}

impl From<Plane> for ElementarySurface {
    /// Wraps a [`Plane`] as an [`ElementarySurface::Plane`].
    fn from(surface: Plane) -> ElementarySurface {
        ElementarySurface::Plane(surface)
    }
}

impl From<Cylinder> for ElementarySurface {
    /// Wraps a [`Cylinder`] as an [`ElementarySurface::Cylinder`].
    fn from(surface: Cylinder) -> ElementarySurface {
        ElementarySurface::Cylinder(surface)
    }
}

impl From<Cone> for ElementarySurface {
    /// Wraps a [`Cone`] as an [`ElementarySurface::Cone`].
    fn from(surface: Cone) -> ElementarySurface {
        ElementarySurface::Cone(surface)
    }
}

impl From<Sphere> for ElementarySurface {
    /// Wraps a [`Sphere`] as an [`ElementarySurface::Sphere`].
    fn from(surface: Sphere) -> ElementarySurface {
        ElementarySurface::Sphere(surface)
    }
}

impl From<Torus> for ElementarySurface {
    /// Wraps a [`Torus`] as an [`ElementarySurface::Torus`].
    fn from(surface: Torus) -> ElementarySurface {
        ElementarySurface::Torus(surface)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point3D;

    fn sample<S: ParametricSurface>(s: &S) -> Point3D {
        s.eval_point(0.3, 0.4)
    }

    fn plane() -> Plane {
        Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap()
    }

    fn cylinder() -> Cylinder {
        Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap()
    }

    fn cone() -> Cone {
        Cone::new(Point3D::ORIGIN, Vector3D::Z, 0.4, 2.0).unwrap()
    }

    fn sphere() -> Sphere {
        Sphere::new(Point3D::ORIGIN, 3.0).unwrap()
    }

    fn torus() -> Torus {
        Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap()
    }

    fn bspline() -> BSplineSurface {
        // Bilinear degree-1x1 patch over the unit square (clamped both ways).
        let poles = vec![
            vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
            vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 1.0)],
        ];
        BSplineSurface::new(
            1,
            1,
            poles,
            vec![0.0, 1.0],
            vec![2, 2],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap()
    }

    // ---- generic fn agrees with inherent calls ----

    #[test]
    fn test_sample_plane_agrees_with_inherent() {
        let p = plane();
        assert_eq!(sample(&p), p.eval_point(0.3, 0.4));
    }

    #[test]
    fn test_sample_cylinder_agrees_with_inherent() {
        let c = cylinder();
        assert_eq!(sample(&c), c.eval_point(0.3, 0.4));
    }

    #[test]
    fn test_sample_cone_agrees_with_inherent() {
        let c = cone();
        assert_eq!(sample(&c), c.eval_point(0.3, 0.4));
    }

    #[test]
    fn test_sample_sphere_agrees_with_inherent() {
        let s = sphere();
        assert_eq!(sample(&s), s.eval_point(0.3, 0.4));
    }

    #[test]
    fn test_sample_torus_agrees_with_inherent() {
        let t = torus();
        assert_eq!(sample(&t), t.eval_point(0.3, 0.4));
    }

    #[test]
    fn test_sample_bspline_agrees_with_inherent() {
        let b = bspline();
        assert_eq!(sample(&b), b.eval_point(0.3, 0.4));
    }

    #[test]
    fn test_sample_surface_enum_agrees_with_inherent() {
        let p = plane();
        let wrapped: Surface = p.into();
        assert_eq!(sample(&wrapped), p.eval_point(0.3, 0.4));
    }

    // ---- heterogeneous Vec<Surface> eval ----

    #[test]
    fn test_heterogeneous_vec_eval() {
        let p = plane();
        let c = cylinder();
        let cn = cone();
        let s = sphere();
        let t = torus();
        let surfaces: Vec<Surface> = vec![p.into(), c.into(), cn.into(), s.into(), t.into()];
        let points: Vec<Point3D> = surfaces
            .iter()
            .map(|surf| surf.eval_point(0.2, 0.1))
            .collect();
        assert_eq!(points[0], p.eval_point(0.2, 0.1));
        assert_eq!(points[1], c.eval_point(0.2, 0.1));
        assert_eq!(points[2], cn.eval_point(0.2, 0.1));
        assert_eq!(points[3], s.eval_point(0.2, 0.1));
        assert_eq!(points[4], t.eval_point(0.2, 0.1));
    }

    // ---- From round-trips: owned and by-reference (copying) ----

    #[test]
    fn test_from_plane_round_trips() {
        let p = plane();
        match Surface::from(p) {
            Surface::Plane(inner) => assert_eq!(inner, p),
            _ => panic!("expected Surface::Plane"),
        }
    }

    #[test]
    fn test_from_plane_ref_round_trips() {
        let p = plane();
        match Surface::from(&p) {
            Surface::Plane(inner) => assert_eq!(inner, p),
            _ => panic!("expected Surface::Plane"),
        }
    }

    #[test]
    fn test_from_cylinder_round_trips() {
        let c = cylinder();
        match Surface::from(c) {
            Surface::Cylinder(inner) => assert_eq!(inner, c),
            _ => panic!("expected Surface::Cylinder"),
        }
    }

    #[test]
    fn test_from_cylinder_ref_round_trips() {
        let c = cylinder();
        match Surface::from(&c) {
            Surface::Cylinder(inner) => assert_eq!(inner, c),
            _ => panic!("expected Surface::Cylinder"),
        }
    }

    #[test]
    fn test_from_cone_round_trips() {
        let c = cone();
        match Surface::from(c) {
            Surface::Cone(inner) => assert_eq!(inner, c),
            _ => panic!("expected Surface::Cone"),
        }
    }

    #[test]
    fn test_from_cone_ref_round_trips() {
        let c = cone();
        match Surface::from(&c) {
            Surface::Cone(inner) => assert_eq!(inner, c),
            _ => panic!("expected Surface::Cone"),
        }
    }

    #[test]
    fn test_from_sphere_round_trips() {
        let s = sphere();
        match Surface::from(s) {
            Surface::Sphere(inner) => assert_eq!(inner, s),
            _ => panic!("expected Surface::Sphere"),
        }
    }

    #[test]
    fn test_from_sphere_ref_round_trips() {
        let s = sphere();
        match Surface::from(&s) {
            Surface::Sphere(inner) => assert_eq!(inner, s),
            _ => panic!("expected Surface::Sphere"),
        }
    }

    #[test]
    fn test_from_torus_round_trips() {
        let t = torus();
        match Surface::from(t) {
            Surface::Torus(inner) => assert_eq!(inner, t),
            _ => panic!("expected Surface::Torus"),
        }
    }

    #[test]
    fn test_from_torus_ref_round_trips() {
        let t = torus();
        match Surface::from(&t) {
            Surface::Torus(inner) => assert_eq!(inner, t),
            _ => panic!("expected Surface::Torus"),
        }
    }

    #[test]
    fn test_from_bspline_round_trips() {
        let b = bspline();
        match Surface::from(b.clone()) {
            Surface::BSpline(inner) => assert_eq!(inner, b),
            _ => panic!("expected Surface::BSpline"),
        }
    }

    #[test]
    fn test_from_bspline_ref_clones() {
        let b = bspline();
        match Surface::from(&b) {
            Surface::BSpline(inner) => assert_eq!(inner, b),
            _ => panic!("expected Surface::BSpline"),
        }
    }

    #[test]
    #[allow(clippy::needless_borrows_for_generic_args)]
    fn test_into_from_reference_supports_parametrize_on_style_call() {
        // Mirrors the Task 17 usage pattern: a function taking
        // `impl Into<Surface>` called with a `&Cylinder` (Cylinder is Copy,
        // so clippy would rather we drop the borrow; the point of this test
        // is specifically that the borrowed form is accepted).
        fn accepts(surface: impl Into<Surface>) -> Surface {
            surface.into()
        }
        let c = cylinder();
        match accepts(&c) {
            Surface::Cylinder(inner) => assert_eq!(inner, c),
            _ => panic!("expected Surface::Cylinder"),
        }
    }

    // ---- bounds/periods table, per type (inherent trait impl + enum dispatch) ----

    #[test]
    fn test_plane_bounds_and_periods() {
        let p = plane();
        assert_eq!(
            ParametricSurface::u_bounds(&p),
            (f64::NEG_INFINITY, f64::INFINITY)
        );
        assert_eq!(
            ParametricSurface::v_bounds(&p),
            (f64::NEG_INFINITY, f64::INFINITY)
        );
        assert_eq!(ParametricSurface::u_period(&p), None);
        assert_eq!(ParametricSurface::v_period(&p), None);

        let wrapped: Surface = p.into();
        assert_eq!(wrapped.u_bounds(), (f64::NEG_INFINITY, f64::INFINITY));
        assert_eq!(wrapped.v_bounds(), (f64::NEG_INFINITY, f64::INFINITY));
        assert_eq!(wrapped.u_period(), None);
        assert_eq!(wrapped.v_period(), None);
    }

    #[test]
    fn test_cylinder_bounds_and_periods() {
        let c = cylinder();
        assert_eq!(ParametricSurface::u_bounds(&c), (0.0, TAU));
        assert_eq!(
            ParametricSurface::v_bounds(&c),
            (f64::NEG_INFINITY, f64::INFINITY)
        );
        assert_eq!(ParametricSurface::u_period(&c), Some(TAU));
        assert_eq!(ParametricSurface::v_period(&c), None);

        let wrapped: Surface = c.into();
        assert_eq!(wrapped.u_bounds(), (0.0, TAU));
        assert_eq!(wrapped.v_bounds(), (f64::NEG_INFINITY, f64::INFINITY));
        assert_eq!(wrapped.u_period(), Some(TAU));
        assert_eq!(wrapped.v_period(), None);
    }

    #[test]
    fn test_cone_bounds_and_periods() {
        let c = cone();
        assert_eq!(ParametricSurface::u_bounds(&c), (0.0, TAU));
        assert_eq!(
            ParametricSurface::v_bounds(&c),
            (f64::NEG_INFINITY, f64::INFINITY)
        );
        assert_eq!(ParametricSurface::u_period(&c), Some(TAU));
        assert_eq!(ParametricSurface::v_period(&c), None);

        let wrapped: Surface = c.into();
        assert_eq!(wrapped.u_bounds(), (0.0, TAU));
        assert_eq!(wrapped.v_bounds(), (f64::NEG_INFINITY, f64::INFINITY));
        assert_eq!(wrapped.u_period(), Some(TAU));
        assert_eq!(wrapped.v_period(), None);
    }

    #[test]
    fn test_sphere_bounds_and_periods() {
        let s = sphere();
        assert_eq!(ParametricSurface::u_bounds(&s), (0.0, TAU));
        assert_eq!(ParametricSurface::v_bounds(&s), (-FRAC_PI_2, FRAC_PI_2));
        assert_eq!(ParametricSurface::u_period(&s), Some(TAU));
        assert_eq!(ParametricSurface::v_period(&s), None);

        let wrapped: Surface = s.into();
        assert_eq!(wrapped.u_bounds(), (0.0, TAU));
        assert_eq!(wrapped.v_bounds(), (-FRAC_PI_2, FRAC_PI_2));
        assert_eq!(wrapped.u_period(), Some(TAU));
        assert_eq!(wrapped.v_period(), None);
    }

    #[test]
    fn test_torus_bounds_and_periods() {
        let t = torus();
        assert_eq!(ParametricSurface::u_bounds(&t), (0.0, TAU));
        assert_eq!(ParametricSurface::v_bounds(&t), (0.0, TAU));
        assert_eq!(ParametricSurface::u_period(&t), Some(TAU));
        assert_eq!(ParametricSurface::v_period(&t), Some(TAU));

        let wrapped: Surface = t.into();
        assert_eq!(wrapped.u_bounds(), (0.0, TAU));
        assert_eq!(wrapped.v_bounds(), (0.0, TAU));
        assert_eq!(wrapped.u_period(), Some(TAU));
        assert_eq!(wrapped.v_period(), Some(TAU));
    }

    #[test]
    fn test_bspline_bounds_and_periods() {
        let b = bspline();
        assert_eq!(ParametricSurface::u_bounds(&b), (0.0, 1.0));
        assert_eq!(ParametricSurface::v_bounds(&b), (0.0, 1.0));
        assert_eq!(ParametricSurface::u_period(&b), None);
        assert_eq!(ParametricSurface::v_period(&b), None);

        let wrapped: Surface = b.into();
        assert_eq!(wrapped.u_bounds(), (0.0, 1.0));
        assert_eq!(wrapped.v_bounds(), (0.0, 1.0));
        assert_eq!(wrapped.u_period(), None);
        assert_eq!(wrapped.v_period(), None);
    }

    #[test]
    fn test_bspline_enum_dispatch_agrees_with_inherent() {
        let b = bspline();
        let wrapped: Surface = b.clone().into();
        assert_eq!(wrapped.eval_point(0.3, 0.4), b.eval_point(0.3, 0.4));
        assert_eq!(
            wrapped.eval_derivative(0.3, 0.4, 1, 0),
            b.eval_derivative(0.3, 0.4, 1, 0)
        );
        assert_eq!(
            wrapped.eval_derivative(0.3, 0.4, 0, 1),
            b.eval_derivative(0.3, 0.4, 0, 1)
        );
    }

    // ---- default eval_points == mapped eval_point ----

    #[test]
    fn test_default_eval_points_matches_mapped_eval_point_for_each_type() {
        let uvs = [(0.0, 0.0), (0.5, 0.2), (1.0, -0.3), (-2.0, 1.1)];

        let p = plane();
        let expected: Vec<Point3D> = uvs.iter().map(|&(u, v)| p.eval_point(u, v)).collect();
        assert_eq!(ParametricSurface::eval_points(&p, &uvs), expected);

        let c = cylinder();
        let expected: Vec<Point3D> = uvs.iter().map(|&(u, v)| c.eval_point(u, v)).collect();
        assert_eq!(ParametricSurface::eval_points(&c, &uvs), expected);

        let cn = cone();
        let expected: Vec<Point3D> = uvs.iter().map(|&(u, v)| cn.eval_point(u, v)).collect();
        assert_eq!(ParametricSurface::eval_points(&cn, &uvs), expected);

        let s = sphere();
        let expected: Vec<Point3D> = uvs.iter().map(|&(u, v)| s.eval_point(u, v)).collect();
        assert_eq!(ParametricSurface::eval_points(&s, &uvs), expected);

        let t = torus();
        let expected: Vec<Point3D> = uvs.iter().map(|&(u, v)| t.eval_point(u, v)).collect();
        assert_eq!(ParametricSurface::eval_points(&t, &uvs), expected);
    }

    #[test]
    fn test_surface_enum_eval_points_matches_mapped_eval_point() {
        let uvs = [(0.0, 0.0), (0.5, 0.2), (1.0, -0.3), (-2.0, 1.1)];
        let surfaces: Vec<Surface> = vec![
            plane().into(),
            cylinder().into(),
            cone().into(),
            sphere().into(),
            torus().into(),
        ];
        for surface in &surfaces {
            let expected: Vec<Point3D> =
                uvs.iter().map(|&(u, v)| surface.eval_point(u, v)).collect();
            assert_eq!(surface.eval_points(&uvs), expected);
        }
    }

    #[test]
    fn test_eval_derivative_delegates_to_inherent() {
        let p = plane();
        assert_eq!(
            ParametricSurface::eval_derivative(&p, 0.3, 0.4, 1, 0),
            p.eval_derivative(0.3, 0.4, 1, 0)
        );

        let s = sphere();
        assert_eq!(
            ParametricSurface::eval_derivative(&s, 0.3, 0.4, 1, 0),
            s.eval_derivative(0.3, 0.4, 1, 0)
        );

        let wrapped: Surface = torus().into();
        assert_eq!(
            wrapped.eval_derivative(0.3, 0.4, 0, 1),
            torus().eval_derivative(0.3, 0.4, 0, 1)
        );
    }

    // ---- elementary surfaces ----

    #[test]
    fn test_elementary_surface_agrees_with_surface() {
        let tol = Tolerance::DEFAULT;
        let surfaces: [ElementarySurface; 5] = [
            plane().into(),
            cylinder().into(),
            cone().into(),
            sphere().into(),
            torus().into(),
        ];
        let query = Point3D::new(0.7, -1.2, 2.5);
        for surface in surfaces {
            let wrapped = Surface::from(surface);
            assert_eq!(wrapped.as_elementary(), Some(surface));
            assert_eq!(surface.eval_point(0.3, 0.4), wrapped.eval_point(0.3, 0.4));
            assert_eq!(
                surface.eval_derivative(0.3, 0.4, 1, 1),
                wrapped.eval_derivative(0.3, 0.4, 1, 1)
            );
            assert_eq!(surface.u_bounds(), wrapped.u_bounds());
            assert_eq!(surface.v_bounds(), wrapped.v_bounds());
            assert_eq!(surface.u_period(), wrapped.u_period());
            assert_eq!(surface.v_period(), wrapped.v_period());
            assert_eq!(
                surface.project_point(query, tol),
                wrapped.project_point(query, tol)
            );
            assert_eq!(surface.contains(query, tol), wrapped.contains(query, tol));
        }
    }

    #[test]
    fn test_bspline_surface_is_not_elementary() {
        assert_eq!(Surface::from(bspline()).as_elementary(), None);
    }

    // ---- curvatures: principal, mean, Gaussian ----

    fn assert_frame(c: &SurfaceCurvatures, normal: Vector3D) {
        // Unit, orthonormal, tangent directions; k1 + k2 = 2H, k1*k2 = K.
        for direction in [c.min_direction, c.max_direction] {
            assert!((direction.magnitude() - 1.0).abs() < 1e-12);
            assert!(direction.dot(normal).abs() < 1e-12);
        }
        assert!(c.min_direction.dot(c.max_direction).abs() < 1e-12);
        assert!((c.min_curvature + c.max_curvature - 2.0 * c.mean).abs() < 1e-12);
        assert!((c.min_curvature * c.max_curvature - c.gaussian).abs() < 1e-12);
        assert!(c.min_curvature <= c.max_curvature);
    }

    #[test]
    fn test_curvatures_plane_are_zero() {
        let tol = Tolerance::DEFAULT;
        let c = ParametricSurface::curvatures(&plane(), 1.0, 2.0, tol).unwrap();
        assert_eq!(
            c,
            SurfaceCurvatures {
                min_curvature: 0.0,
                max_curvature: 0.0,
                min_direction: Vector3D::X,
                max_direction: Vector3D::Y,
                mean: 0.0,
                gaussian: 0.0,
            }
        );
    }

    #[test]
    fn test_curvatures_cylinder_hoop_and_ruling() {
        let tol = Tolerance::DEFAULT;
        let c = ParametricSurface::curvatures(&cylinder(), 0.0, 3.0, tol).unwrap();
        assert!((c.min_curvature + 0.5).abs() < 1e-12);
        assert_eq!(c.max_curvature, 0.0);
        assert_eq!((c.mean, c.gaussian), (-0.25, 0.0));
        assert!((c.min_direction.dot(Vector3D::Y).abs() - 1.0).abs() < 1e-12);
        assert!((c.max_direction.dot(Vector3D::Z).abs() - 1.0).abs() < 1e-12);
        assert_frame(&c, Vector3D::X);
    }

    #[test]
    fn test_curvatures_sphere_umbilic() {
        let tol = Tolerance::DEFAULT;
        let c = ParametricSurface::curvatures(&sphere(), 0.0, 0.0, tol).unwrap();
        assert!((c.min_curvature + 1.0 / 3.0).abs() < 1e-15);
        assert!((c.max_curvature + 1.0 / 3.0).abs() < 1e-15);
        assert!((c.mean + 1.0 / 3.0).abs() < 1e-15);
        assert!((c.gaussian - 1.0 / 9.0).abs() < 1e-15);
        assert_frame(&c, Vector3D::X);
    }

    #[test]
    fn test_curvatures_torus_outer_equator() {
        let tol = Tolerance::DEFAULT;
        let c = ParametricSurface::curvatures(&torus(), 0.0, 0.0, tol).unwrap();
        assert!((c.min_curvature + 1.0 / 1.5).abs() < 1e-12);
        assert!((c.max_curvature + 1.0 / 6.5).abs() < 1e-12);
        assert!((c.gaussian - 1.0 / (6.5 * 1.5)).abs() < 1e-12);
        assert!((c.min_direction.dot(Vector3D::Z).abs() - 1.0).abs() < 1e-12);
        assert!((c.max_direction.dot(Vector3D::Y).abs() - 1.0).abs() < 1e-12);
        assert_frame(&c, Vector3D::X);
    }

    #[test]
    fn test_curvatures_torus_inner_equator_saddle() {
        use std::f64::consts::PI;

        // Inner equator (u=0, v=pi): tube k=-1/r, ring k=+1/(R-r) (the
        // ring bends toward the normal there, flipping its sign).
        let tol = Tolerance::DEFAULT;
        let c = ParametricSurface::curvatures(&torus(), 0.0, PI, tol).unwrap();
        assert!((c.min_curvature + 2.0 / 3.0).abs() < 1e-12);
        assert!((c.max_curvature - 2.0 / 7.0).abs() < 1e-12);
        assert!((c.mean + 4.0 / 21.0).abs() < 1e-12);
        assert!((c.gaussian + 4.0 / 21.0).abs() < 1e-12);
        assert!((c.min_direction.dot(Vector3D::Z).abs() - 1.0).abs() < 1e-12);
        assert!((c.max_direction.dot(Vector3D::Y).abs() - 1.0).abs() < 1e-12);
        assert_frame(&c, -Vector3D::X);
    }

    #[test]
    fn test_curvatures_cone_developable() {
        let tol = Tolerance::DEFAULT;
        let c = ParametricSurface::curvatures(&cone(), 0.0, 0.0, tol).unwrap();
        assert!((c.min_curvature + 0.4f64.cos() / 2.0).abs() < 1e-9);
        assert!(c.max_curvature.abs() < 1e-9);
        assert!(c.gaussian.abs() < 1e-9);
        assert_frame(&c, Vector3D::new(0.4f64.cos(), 0.0, -0.4f64.sin()));
    }

    #[test]
    fn test_curvatures_saddle_is_negative_gaussian() {
        // Bilinear patch S(u, v) = (u, v, u*v): K = -1/(1+u²+v²)².
        let tol = Tolerance::DEFAULT;
        let c = ParametricSurface::curvatures(&bspline(), 0.5, 0.5, tol).unwrap();
        assert!((c.gaussian + 1.0 / 2.25).abs() < 1e-9);
        assert!((c.mean + 0.25 / 1.5f64.powf(1.5)).abs() < 1e-9);
        let n = Vector3D::new(-0.5, -0.5, 1.0);
        assert_frame(&c, n * (1.0 / n.magnitude()));
    }

    #[test]
    fn test_curvatures_none_at_singular_points() {
        use std::f64::consts::FRAC_PI_2;

        let tol = Tolerance::DEFAULT;
        assert_eq!(
            ParametricSurface::curvatures(&sphere(), 0.0, FRAC_PI_2, tol),
            None
        );
        let v_apex = -2.0 / 0.4f64.sin();
        assert_eq!(
            ParametricSurface::curvatures(&cone(), 0.0, v_apex, tol),
            None
        );
    }

    /// Normal curvature `II(d, d) / I(d, d)` along a tangent `direction`,
    /// solved back to its parameter step, as an independent check that a
    /// reported principal direction carries its principal curvature.
    fn normal_curvature_along<S: ParametricSurface>(
        surface: &S,
        u: f64,
        v: f64,
        direction: Vector3D,
    ) -> f64 {
        let (_, [su, sv], [suu, suv, svv]) = surface.eval_d2(u, v);
        let normal = su.cross(sv) * (1.0 / su.cross(sv).magnitude());
        let (e, f, g) = (su.dot(su), su.dot(sv), sv.dot(sv));
        let (along_u, along_v) = (direction.dot(su), direction.dot(sv));
        let determinant = e * g - f * f;
        let du = (g * along_u - f * along_v) / determinant;
        let dv = (e * along_v - f * along_u) / determinant;
        let second = suu * (du * du) + suv * (2.0 * du * dv) + svv * (dv * dv);
        let tangent = su * du + sv * dv;
        second.dot(normal) / tangent.dot(tangent)
    }

    #[test]
    fn test_curvatures_saddle_directions_carry_principal_curvatures() {
        // Skewed parametrization (Su·Sv ≠ 0): each direction's normal
        // curvature matches its reported principal curvature.
        let tol = Tolerance::DEFAULT;
        let surface = bspline();
        for (u, v) in [(0.5, 0.5), (0.1, 0.8), (0.9, 0.3)] {
            let c = ParametricSurface::curvatures(&surface, u, v, tol).unwrap();
            let along_min = normal_curvature_along(&surface, u, v, c.min_direction);
            let along_max = normal_curvature_along(&surface, u, v, c.max_direction);
            assert!((along_min - c.min_curvature).abs() < 1e-14);
            assert!((along_max - c.max_curvature).abs() < 1e-14);
            let squared = 1.0 + u * u + v * v;
            assert!((c.gaussian + 1.0 / (squared * squared)).abs() < 1e-14);
        }
    }

    #[test]
    fn test_curvatures_sphere_everywhere_umbilic_with_orthonormal_directions() {
        // Rounding splits the two curvatures by ~1e-16 away from the
        // equator; the directions must stay a perpendicular unit pair and
        // the curvatures must not pick up a spurious √ε gap.
        let tol = Tolerance::DEFAULT;
        let sphere = Sphere::new(Point3D::new(1.0, 2.0, 3.0), 3.0).unwrap();
        for i in 0..40 {
            let (u, v) = (0.157 * f64::from(i), -1.5 + 0.075 * f64::from(i));
            let c = ParametricSurface::curvatures(&sphere, u, v, tol).unwrap();
            assert!((c.min_curvature + 1.0 / 3.0).abs() < 1e-14, "{u}, {v}");
            assert!((c.max_curvature + 1.0 / 3.0).abs() < 1e-14, "{u}, {v}");
            let (_, [su, sv]) = sphere.eval_d1(u, v);
            assert_frame(&c, su.cross(sv) * (1.0 / su.cross(sv).magnitude()));
        }
    }

    #[test]
    fn test_curvatures_rational_cylinder_patch() {
        // Quarter circle of radius 2 swept along z: hoop curvature -1/2
        // along the unit normal, zero along the ruling, at any weight.
        let weight = std::f64::consts::FRAC_1_SQRT_2;
        let poles = vec![
            vec![Point3D::new(2.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 1.0)],
            vec![Point3D::new(2.0, 2.0, 0.0), Point3D::new(2.0, 2.0, 1.0)],
            vec![Point3D::new(0.0, 2.0, 0.0), Point3D::new(0.0, 2.0, 1.0)],
        ];
        let weights = vec![vec![1.0, 1.0], vec![weight, weight], vec![1.0, 1.0]];
        let surface = BSplineSurface::new_rational(
            2,
            1,
            poles,
            weights,
            vec![0.0, 1.0],
            vec![3, 3],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap();
        let tol = Tolerance::DEFAULT;
        for u in [0.0, 0.3, 0.5, 1.0] {
            let c = ParametricSurface::curvatures(&surface, u, 0.5, tol).unwrap();
            assert!((c.min_curvature + 0.5).abs() < 1e-14);
            assert!(c.max_curvature.abs() < 1e-14);
            assert!((c.max_direction.dot(Vector3D::Z).abs() - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn test_curvatures_none_for_non_finite_parameters() {
        let tol = Tolerance::DEFAULT;
        assert_eq!(
            ParametricSurface::curvatures(&torus(), f64::NAN, 0.0, tol),
            None
        );
        assert_eq!(
            ParametricSurface::curvatures(&cylinder(), f64::INFINITY, 0.0, tol),
            None
        );
    }

    #[test]
    fn test_surface_enum_curvatures_dispatch_agrees_with_concrete() {
        let tol = Tolerance::DEFAULT;
        let surfaces: Vec<Surface> = vec![
            plane().into(),
            cylinder().into(),
            cone().into(),
            sphere().into(),
            torus().into(),
            bspline().into(),
        ];
        for surface in &surfaces {
            let expected = match surface {
                Surface::Plane(s) => ParametricSurface::curvatures(s, 0.3, 0.2, tol),
                Surface::Cylinder(s) => ParametricSurface::curvatures(s, 0.3, 0.2, tol),
                Surface::Cone(s) => ParametricSurface::curvatures(s, 0.3, 0.2, tol),
                Surface::Sphere(s) => ParametricSurface::curvatures(s, 0.3, 0.2, tol),
                Surface::Torus(s) => ParametricSurface::curvatures(s, 0.3, 0.2, tol),
                Surface::BSpline(s) => ParametricSurface::curvatures(s, 0.3, 0.2, tol),
            };
            assert_eq!(surface.curvatures(0.3, 0.2, tol), expected);
        }
    }
}
