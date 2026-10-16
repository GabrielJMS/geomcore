//! [`ParametricSurface`] trait and the [`Surface`] enum adaptor unifying the
//! concrete analytic surface types.

use crate::curves::Curve3D;
use crate::surfaces::{BSplineSurface, Cone, Cylinder, Plane, Sphere, SurfaceIsoError, Torus};
use crate::{Point3D, SurfaceProjection, Tolerance, Vector3D};
use std::f64::consts::{FRAC_PI_2, TAU};

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
    /// The numeric search behind
    /// [`Intersect::intersect`](crate::Intersect::intersect) uses it
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
    /// The numeric search behind
    /// [`Intersect::intersect`](crate::Intersect::intersect) uses it
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
    /// Projects `point` onto the surface: the closest parameters, the closest
    /// point and the distance. Distances within `tol.confusion` snap to
    /// `0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::surfaces::Surface;
    /// use geomcore::{Plane, Point3D, Tolerance, Vector3D};
    ///
    /// let surface = Surface::from(Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap());
    /// let projection = surface.project_point(Point3D::new(1.0, 2.0, 3.0), Tolerance::DEFAULT);
    /// assert_eq!(projection.point, Point3D::new(1.0, 2.0, 0.0));
    /// assert_eq!(projection.distance, 3.0);
    /// ```
    pub fn project_point(&self, point: Point3D, tol: Tolerance) -> SurfaceProjection {
        ParametricSurface::project_point(self, point, tol)
    }

    /// Projects each point in `points` onto the surface.
    pub fn project_points(&self, points: &[Point3D], tol: Tolerance) -> Vec<SurfaceProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }

    /// The surface parameters `(u, v)` of the point of the surface closest
    /// to `point`.
    ///
    /// An elementary surface uses its closed form (see
    /// [`ElementarySurface::parameters_of`]). A B-spline projects the point
    /// with [`Tolerance::DEFAULT`] and returns the projection's parameters;
    /// call [`project_point`](Surface::project_point) to choose the
    /// tolerance.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::surfaces::Surface;
    /// use geomcore::{Plane, Point3D, Vector3D};
    ///
    /// let surface = Surface::from(Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap());
    /// let (u, v) = surface.parameters_of(Point3D::new(1.0, 2.0, 0.0));
    /// let back = geomcore::ParametricSurface::eval_point(&surface, u, v);
    /// assert!(back.distance(Point3D::new(1.0, 2.0, 0.0)) < 1e-12);
    /// ```
    pub fn parameters_of(&self, point: Point3D) -> (f64, f64) {
        match self {
            Surface::Plane(s) => s.parameters_of(point),
            Surface::Cylinder(s) => s.parameters_of(point),
            Surface::Cone(s) => s.parameters_of(point),
            Surface::Sphere(s) => s.parameters_of(point),
            Surface::Torus(s) => s.parameters_of(point),
            Surface::BSpline(s) => {
                let projection = s.project_point(point, Tolerance::DEFAULT);
                (projection.u, projection.v)
            }
        }
    }

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

impl ElementarySurface {
    /// The surface parameters `(u, v)` of the point of the surface closest
    /// to `point`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::surfaces::ElementarySurface;
    /// use geomcore::{Plane, Point3D, Vector3D};
    ///
    /// let surface = ElementarySurface::from(Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap());
    /// let (u, v) = surface.parameters_of(Point3D::new(1.0, 2.0, 0.0));
    /// let back = geomcore::ParametricSurface::eval_point(&surface, u, v);
    /// assert!(back.distance(Point3D::new(1.0, 2.0, 0.0)) < 1e-12);
    /// ```
    pub fn parameters_of(&self, point: Point3D) -> (f64, f64) {
        match self {
            ElementarySurface::Plane(s) => s.parameters_of(point),
            ElementarySurface::Cylinder(s) => s.parameters_of(point),
            ElementarySurface::Cone(s) => s.parameters_of(point),
            ElementarySurface::Sphere(s) => s.parameters_of(point),
            ElementarySurface::Torus(s) => s.parameters_of(point),
        }
    }

    /// Projects `point` onto the surface: the closest parameters, the closest
    /// point and the distance. Distances within `tol.confusion` snap to
    /// `0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::surfaces::ElementarySurface;
    /// use geomcore::{Plane, Point3D, Tolerance, Vector3D};
    ///
    /// let surface = ElementarySurface::from(Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap());
    /// let projection = surface.project_point(Point3D::new(1.0, 2.0, 3.0), Tolerance::DEFAULT);
    /// assert_eq!(projection.point, Point3D::new(1.0, 2.0, 0.0));
    /// assert_eq!(projection.distance, 3.0);
    /// ```
    pub fn project_point(&self, point: Point3D, tol: Tolerance) -> SurfaceProjection {
        ParametricSurface::project_point(self, point, tol)
    }

    /// Projects each point in `points` onto the surface.
    pub fn project_points(&self, points: &[Point3D], tol: Tolerance) -> Vec<SurfaceProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
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
}
