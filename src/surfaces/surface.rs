//! [`ParametricSurface`] trait and the [`Surface`] enum adaptor unifying the
//! concrete analytic surface types.

use crate::surfaces::{BSplineSurface, Cone, Cylinder, Plane, Sphere, Torus};
use crate::{Point3, Vector3};
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
/// use geomcore::{Plane, Point3, Vector3};
///
/// let plane = Plane::new(Point3::ORIGIN, Vector3::Z).unwrap();
/// let uvs = [(0.0, 0.0), (1.0, 2.0)];
/// let expected: Vec<Point3> = uvs.iter().map(|&(u, v)| plane.eval_point(u, v)).collect();
/// assert_eq!(ParametricSurface::eval_points(&plane, &uvs), expected);
/// ```
pub trait ParametricSurface {
    /// Evaluates the point on the surface at parameters `(u, v)`.
    fn eval_point(&self, u: f64, v: f64) -> Point3;

    /// Evaluates the derivative of order `(du, dv)` at parameters `(u, v)`.
    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3;

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
    fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3> {
        uvs.iter().map(|&(u, v)| self.eval_point(u, v)).collect()
    }
}

impl ParametricSurface for Plane {
    fn eval_point(&self, u: f64, v: f64) -> Point3 {
        Plane::eval_point(self, u, v)
    }

    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3 {
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

    fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3> {
        Plane::eval_points(self, uvs)
    }
}

impl ParametricSurface for Cylinder {
    fn eval_point(&self, u: f64, v: f64) -> Point3 {
        Cylinder::eval_point(self, u, v)
    }

    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3 {
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

    fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3> {
        Cylinder::eval_points(self, uvs)
    }
}

impl ParametricSurface for Cone {
    fn eval_point(&self, u: f64, v: f64) -> Point3 {
        Cone::eval_point(self, u, v)
    }

    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3 {
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

    fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3> {
        Cone::eval_points(self, uvs)
    }
}

impl ParametricSurface for Sphere {
    fn eval_point(&self, u: f64, v: f64) -> Point3 {
        Sphere::eval_point(self, u, v)
    }

    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3 {
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

    fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3> {
        Sphere::eval_points(self, uvs)
    }
}

impl ParametricSurface for Torus {
    fn eval_point(&self, u: f64, v: f64) -> Point3 {
        Torus::eval_point(self, u, v)
    }

    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3 {
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

    fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3> {
        Torus::eval_points(self, uvs)
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
/// use geomcore::{Plane, Point3, Sphere, Vector3};
///
/// let surfaces: Vec<Surface> = vec![
///     Plane::new(Point3::ORIGIN, Vector3::Z).unwrap().into(),
///     Sphere::new(Point3::ORIGIN, 2.0).unwrap().into(),
/// ];
/// let points: Vec<Point3> = surfaces.iter().map(|s| s.eval_point(0.0, 0.0)).collect();
/// assert_eq!(points[0], Point3::ORIGIN);
/// assert_eq!(points[1], Point3::new(2.0, 0.0, 0.0));
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

impl ParametricSurface for Surface {
    fn eval_point(&self, u: f64, v: f64) -> Point3 {
        match self {
            Surface::Plane(s) => s.eval_point(u, v),
            Surface::Cylinder(s) => s.eval_point(u, v),
            Surface::Cone(s) => s.eval_point(u, v),
            Surface::Sphere(s) => s.eval_point(u, v),
            Surface::Torus(s) => s.eval_point(u, v),
            Surface::BSpline(s) => s.eval_point(u, v),
        }
    }

    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3 {
        match self {
            Surface::Plane(s) => s.eval_derivative(u, v, du, dv),
            Surface::Cylinder(s) => s.eval_derivative(u, v, du, dv),
            Surface::Cone(s) => s.eval_derivative(u, v, du, dv),
            Surface::Sphere(s) => s.eval_derivative(u, v, du, dv),
            Surface::Torus(s) => s.eval_derivative(u, v, du, dv),
            Surface::BSpline(s) => s.eval_derivative(u, v, du, dv),
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

    fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3> {
        match self {
            Surface::Plane(s) => s.eval_points(uvs),
            Surface::Cylinder(s) => s.eval_points(uvs),
            Surface::Cone(s) => s.eval_points(uvs),
            Surface::Sphere(s) => s.eval_points(uvs),
            Surface::Torus(s) => s.eval_points(uvs),
            Surface::BSpline(s) => s.eval_points(uvs),
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point3;

    fn sample<S: ParametricSurface>(s: &S) -> Point3 {
        s.eval_point(0.3, 0.4)
    }

    fn plane() -> Plane {
        Plane::new(Point3::ORIGIN, Vector3::Z).unwrap()
    }

    fn cylinder() -> Cylinder {
        Cylinder::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap()
    }

    fn cone() -> Cone {
        Cone::new(Point3::ORIGIN, Vector3::Z, 0.4, 2.0).unwrap()
    }

    fn sphere() -> Sphere {
        Sphere::new(Point3::ORIGIN, 3.0).unwrap()
    }

    fn torus() -> Torus {
        Torus::new(Point3::ORIGIN, Vector3::Z, 5.0, 1.5).unwrap()
    }

    fn bspline() -> BSplineSurface {
        // Bilinear degree-1x1 patch over the unit square (clamped both ways).
        let poles = vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
            vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)],
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
        let points: Vec<Point3> = surfaces
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
        let expected: Vec<Point3> = uvs.iter().map(|&(u, v)| p.eval_point(u, v)).collect();
        assert_eq!(ParametricSurface::eval_points(&p, &uvs), expected);

        let c = cylinder();
        let expected: Vec<Point3> = uvs.iter().map(|&(u, v)| c.eval_point(u, v)).collect();
        assert_eq!(ParametricSurface::eval_points(&c, &uvs), expected);

        let cn = cone();
        let expected: Vec<Point3> = uvs.iter().map(|&(u, v)| cn.eval_point(u, v)).collect();
        assert_eq!(ParametricSurface::eval_points(&cn, &uvs), expected);

        let s = sphere();
        let expected: Vec<Point3> = uvs.iter().map(|&(u, v)| s.eval_point(u, v)).collect();
        assert_eq!(ParametricSurface::eval_points(&s, &uvs), expected);

        let t = torus();
        let expected: Vec<Point3> = uvs.iter().map(|&(u, v)| t.eval_point(u, v)).collect();
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
            let expected: Vec<Point3> =
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
}
