//! The single intersection entry point.
//!
//! [`Intersect`] is implemented once per exact pair, in both argument
//! orders, with the pair's own result type as its output. The general
//! results of the untyped path hold their pieces in an [`InlineList`], so
//! no intersection result allocates.

use crate::curves::ElementaryCurve;
use crate::surfaces::{ElementarySurface, Surface};
use crate::{
    Cone, ConeConeIntersection, ConeCylinderIntersection, Cylinder, CylinderCylinderIntersection,
    Plane, PlaneConeIntersection, PlaneCylinderIntersection, PlanePlaneIntersection,
    PlaneSphereIntersection, Point3D, Sphere, SphereConeIntersection, SphereCylinderIntersection,
    SphereSphereIntersection, Tolerance, Torus, TorusConeIntersection, TorusCylinderIntersection,
    TorusPlaneIntersection, TorusSphereIntersection, TorusTorusIntersection,
};
use std::ops::{Deref, DerefMut};

/// Exact intersection of two shapes.
///
/// Each implementation returns the result type of its pair, so the
/// compiler knows which outcomes a pair can produce, and dispatch is
/// static. A configuration with no closed form reports "not analytic":
/// no implementation falls back to a numeric intersection on its own.
pub trait Intersect<Rhs: ?Sized = Self> {
    /// The result of intersecting `Self` with `Rhs`.
    type Output;

    /// Intersects `self` with `other`, deciding tangency and coincidence
    /// with `tol`.
    fn intersect(&self, other: &Rhs, tol: Tolerance) -> Self::Output;
}

/// Implements the reversed argument order of a pair whose result does
/// not depend on the order, by swapping the arguments.
macro_rules! reversed {
    ($($a:ty => $b:ty),* $(,)?) => {$(
        impl Intersect<$b> for $a {
            type Output = <$b as Intersect<$a>>::Output;

            /// The same result as in the other argument order.
            fn intersect(&self, other: &$b, tol: Tolerance) -> Self::Output {
                other.intersect(self, tol)
            }
        }
    )*};
}

reversed!(
    Sphere => Plane,
    Cylinder => Plane,
    Cone => Plane,
    Plane => Torus,
    Cylinder => Sphere,
    Cone => Sphere,
    Sphere => Torus,
    Cylinder => Cone,
    Cylinder => Torus,
    Cone => Torus,
);

/// The intersection of two surfaces whatever their types: the general
/// result of the untyped path.
///
/// The typed results convert into it with `From`.
///
/// # Examples
///
/// ```
/// use geomcore::surfaces::Surface;
/// use geomcore::{Intersect, Plane, Point3D, Sphere, SurfaceIntersection, Tolerance, Vector3D};
/// let plane = Surface::from(Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap());
/// let sphere = Surface::from(Sphere::new(Point3D::ORIGIN, 2.0).unwrap());
/// match plane.intersect(&sphere, Tolerance::DEFAULT) {
///     SurfaceIntersection::Meets { curves, points } => {
///         assert_eq!((curves.len(), points.len()), (1, 0));
///         let (_circle, tangent) = curves[0];
///         assert!(!tangent);
///     }
///     _ => panic!("expected a circle"),
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[expect(
    clippy::large_enum_variant,
    reason = "results are held inline instead of boxed (docs/adr/0001)"
)]
pub enum SurfaceIntersection {
    /// The surfaces do not meet.
    Empty,
    /// The surfaces coincide.
    Coincident,
    /// No closed form for this configuration, or one of the surfaces is a
    /// B-spline. The caller decides whether to run a numeric
    /// intersection.
    NotAnalytic,
    /// The surfaces meet in curves, contact points, or both.
    Meets {
        /// The intersection curves, each marked `true` where the surfaces
        /// graze along it instead of crossing.
        curves: InlineList<(ElementaryCurve, bool), 4>,
        /// The contact points: isolated tangencies or singular points such
        /// as a cone apex.
        points: InlineList<Point3D, 4>,
    },
}

impl SurfaceIntersection {
    /// Curves, each with its tangency flag.
    fn curves<C: Into<ElementaryCurve>, const K: usize>(
        curves: [(C, bool); K],
    ) -> SurfaceIntersection {
        let mut list = InlineList::new();
        for (curve, tangent) in curves {
            list.push((curve.into(), tangent));
        }
        SurfaceIntersection::Meets {
            curves: list,
            points: InlineList::new(),
        }
    }

    /// A single contact point.
    fn point(point: Point3D) -> SurfaceIntersection {
        let mut points = InlineList::new();
        points.push(point);
        SurfaceIntersection::Meets {
            curves: InlineList::new(),
            points,
        }
    }
}

impl From<PlanePlaneIntersection> for SurfaceIntersection {
    fn from(result: PlanePlaneIntersection) -> SurfaceIntersection {
        match result {
            PlanePlaneIntersection::Line(l) => SurfaceIntersection::curves([(l, false)]),
            PlanePlaneIntersection::Parallel => SurfaceIntersection::Empty,
            PlanePlaneIntersection::Coincident => SurfaceIntersection::Coincident,
        }
    }
}

impl From<PlaneSphereIntersection> for SurfaceIntersection {
    fn from(result: PlaneSphereIntersection) -> SurfaceIntersection {
        match result {
            PlaneSphereIntersection::Circle(c) => SurfaceIntersection::curves([(c, false)]),
            PlaneSphereIntersection::TangentPoint(p) => SurfaceIntersection::point(p),
            PlaneSphereIntersection::Empty => SurfaceIntersection::Empty,
        }
    }
}

impl From<SphereSphereIntersection> for SurfaceIntersection {
    fn from(result: SphereSphereIntersection) -> SurfaceIntersection {
        match result {
            SphereSphereIntersection::Circle(c) => SurfaceIntersection::curves([(c, false)]),
            SphereSphereIntersection::TangentPoint(p) => SurfaceIntersection::point(p),
            SphereSphereIntersection::Empty => SurfaceIntersection::Empty,
            SphereSphereIntersection::Coincident => SurfaceIntersection::Coincident,
        }
    }
}

impl From<PlaneCylinderIntersection> for SurfaceIntersection {
    fn from(result: PlaneCylinderIntersection) -> SurfaceIntersection {
        match result {
            PlaneCylinderIntersection::Circle(c) => SurfaceIntersection::curves([(c, false)]),
            PlaneCylinderIntersection::Ellipse(e) => SurfaceIntersection::curves([(e, false)]),
            PlaneCylinderIntersection::TwoLines(l1, l2) => {
                SurfaceIntersection::curves([(l1, false), (l2, false)])
            }
            PlaneCylinderIntersection::TangentLine(l) => SurfaceIntersection::curves([(l, true)]),
            PlaneCylinderIntersection::Empty => SurfaceIntersection::Empty,
        }
    }
}

impl From<PlaneConeIntersection> for SurfaceIntersection {
    fn from(result: PlaneConeIntersection) -> SurfaceIntersection {
        match result {
            PlaneConeIntersection::Circle(c) => SurfaceIntersection::curves([(c, false)]),
            PlaneConeIntersection::Ellipse(e) => SurfaceIntersection::curves([(e, false)]),
            PlaneConeIntersection::Parabola(p) => SurfaceIntersection::curves([(p, false)]),
            PlaneConeIntersection::Hyperbola(h1, h2) => {
                SurfaceIntersection::curves([(h1, false), (h2, false)])
            }
            PlaneConeIntersection::TwoLines(l1, l2) => {
                SurfaceIntersection::curves([(l1, false), (l2, false)])
            }
            PlaneConeIntersection::TangentLine(l) => SurfaceIntersection::curves([(l, true)]),
            PlaneConeIntersection::ApexPoint(p) => SurfaceIntersection::point(p),
        }
    }
}

impl From<SphereCylinderIntersection> for SurfaceIntersection {
    fn from(result: SphereCylinderIntersection) -> SurfaceIntersection {
        match result {
            // A single ring is the equator of a tube as wide as the sphere.
            SphereCylinderIntersection::Circle(c) => SurfaceIntersection::curves([(c, true)]),
            SphereCylinderIntersection::TwoCircles(c1, c2) => {
                SurfaceIntersection::curves([(c1, false), (c2, false)])
            }
            SphereCylinderIntersection::Empty => SurfaceIntersection::Empty,
            SphereCylinderIntersection::NotAnalytic => SurfaceIntersection::NotAnalytic,
        }
    }
}

impl From<SphereConeIntersection> for SurfaceIntersection {
    fn from(result: SphereConeIntersection) -> SurfaceIntersection {
        match result {
            SphereConeIntersection::TwoCircles(c1, c2) => {
                SurfaceIntersection::curves([(c1, false), (c2, false)])
            }
            SphereConeIntersection::TangentCircle(c) => SurfaceIntersection::curves([(c, true)]),
            SphereConeIntersection::Empty => SurfaceIntersection::Empty,
            SphereConeIntersection::NotAnalytic => SurfaceIntersection::NotAnalytic,
        }
    }
}

impl From<CylinderCylinderIntersection> for SurfaceIntersection {
    fn from(result: CylinderCylinderIntersection) -> SurfaceIntersection {
        match result {
            CylinderCylinderIntersection::TwoLines(l1, l2) => {
                SurfaceIntersection::curves([(l1, false), (l2, false)])
            }
            CylinderCylinderIntersection::TangentLine(l) => {
                SurfaceIntersection::curves([(l, true)])
            }
            CylinderCylinderIntersection::Empty => SurfaceIntersection::Empty,
            CylinderCylinderIntersection::Coincident => SurfaceIntersection::Coincident,
            CylinderCylinderIntersection::NotAnalytic => SurfaceIntersection::NotAnalytic,
        }
    }
}

impl From<ConeCylinderIntersection> for SurfaceIntersection {
    fn from(result: ConeCylinderIntersection) -> SurfaceIntersection {
        match result {
            ConeCylinderIntersection::TwoCircles(c1, c2) => {
                SurfaceIntersection::curves([(c1, false), (c2, false)])
            }
            ConeCylinderIntersection::ApexPoint(p) => SurfaceIntersection::point(p),
            ConeCylinderIntersection::NotAnalytic => SurfaceIntersection::NotAnalytic,
        }
    }
}

impl From<TorusPlaneIntersection> for SurfaceIntersection {
    fn from(result: TorusPlaneIntersection) -> SurfaceIntersection {
        match result {
            TorusPlaneIntersection::TwoCircles(c1, c2) => {
                SurfaceIntersection::curves([(c1, false), (c2, false)])
            }
            TorusPlaneIntersection::TangentCircle(c) => SurfaceIntersection::curves([(c, true)]),
            TorusPlaneIntersection::Circle(c) => SurfaceIntersection::curves([(c, false)]),
            TorusPlaneIntersection::Empty => SurfaceIntersection::Empty,
            TorusPlaneIntersection::NotAnalytic => SurfaceIntersection::NotAnalytic,
        }
    }
}

impl From<TorusSphereIntersection> for SurfaceIntersection {
    fn from(result: TorusSphereIntersection) -> SurfaceIntersection {
        match result {
            TorusSphereIntersection::Circle(c) => SurfaceIntersection::curves([(c, false)]),
            TorusSphereIntersection::TwoCircles(c1, c2) => {
                SurfaceIntersection::curves([(c1, false), (c2, false)])
            }
            TorusSphereIntersection::TangentCircle(c) => SurfaceIntersection::curves([(c, true)]),
            TorusSphereIntersection::Empty => SurfaceIntersection::Empty,
            TorusSphereIntersection::NotAnalytic => SurfaceIntersection::NotAnalytic,
        }
    }
}

impl From<TorusCylinderIntersection> for SurfaceIntersection {
    fn from(result: TorusCylinderIntersection) -> SurfaceIntersection {
        match result {
            TorusCylinderIntersection::TwoCircles(c1, c2) => {
                SurfaceIntersection::curves([(c1, false), (c2, false)])
            }
            TorusCylinderIntersection::TangentCircle(c) => SurfaceIntersection::curves([(c, true)]),
            TorusCylinderIntersection::Empty => SurfaceIntersection::Empty,
            TorusCylinderIntersection::NotAnalytic => SurfaceIntersection::NotAnalytic,
        }
    }
}

impl From<TorusConeIntersection> for SurfaceIntersection {
    fn from(result: TorusConeIntersection) -> SurfaceIntersection {
        match result {
            TorusConeIntersection::Rings(rings) => {
                let mut curves = InlineList::new();
                for ring in rings {
                    curves.push((ring.circle.into(), ring.tangent));
                }
                SurfaceIntersection::Meets {
                    curves,
                    points: InlineList::new(),
                }
            }
            TorusConeIntersection::Empty => SurfaceIntersection::Empty,
            TorusConeIntersection::NotAnalytic => SurfaceIntersection::NotAnalytic,
        }
    }
}

impl From<TorusTorusIntersection> for SurfaceIntersection {
    fn from(result: TorusTorusIntersection) -> SurfaceIntersection {
        match result {
            TorusTorusIntersection::Circle(c) => SurfaceIntersection::curves([(c, false)]),
            TorusTorusIntersection::TwoCircles(c1, c2) => {
                SurfaceIntersection::curves([(c1, false), (c2, false)])
            }
            TorusTorusIntersection::TangentCircle(c) => SurfaceIntersection::curves([(c, true)]),
            TorusTorusIntersection::Empty => SurfaceIntersection::Empty,
            TorusTorusIntersection::Coincident => SurfaceIntersection::Coincident,
            TorusTorusIntersection::NotAnalytic => SurfaceIntersection::NotAnalytic,
        }
    }
}

impl From<ConeConeIntersection> for SurfaceIntersection {
    fn from(result: ConeConeIntersection) -> SurfaceIntersection {
        match result {
            ConeConeIntersection::Circle(c) => SurfaceIntersection::curves([(c, false)]),
            ConeConeIntersection::TwoCircles(c1, c2) => {
                SurfaceIntersection::curves([(c1, false), (c2, false)])
            }
            ConeConeIntersection::ApexPoint(p) => SurfaceIntersection::point(p),
            ConeConeIntersection::Coincident => SurfaceIntersection::Coincident,
            ConeConeIntersection::NotAnalytic => SurfaceIntersection::NotAnalytic,
        }
    }
}

impl Intersect for ElementarySurface {
    type Output = SurfaceIntersection;

    /// Dispatches to the exact intersection of the pair.
    fn intersect(&self, other: &ElementarySurface, tol: Tolerance) -> SurfaceIntersection {
        use ElementarySurface as S;
        match (self, other) {
            (S::Plane(a), S::Plane(b)) => a.intersect(b, tol).into(),
            (S::Plane(a), S::Cylinder(b)) => a.intersect(b, tol).into(),
            (S::Plane(a), S::Cone(b)) => a.intersect(b, tol).into(),
            (S::Plane(a), S::Sphere(b)) => a.intersect(b, tol).into(),
            (S::Plane(a), S::Torus(b)) => a.intersect(b, tol).into(),
            (S::Cylinder(a), S::Plane(b)) => a.intersect(b, tol).into(),
            (S::Cylinder(a), S::Cylinder(b)) => a.intersect(b, tol).into(),
            (S::Cylinder(a), S::Cone(b)) => a.intersect(b, tol).into(),
            (S::Cylinder(a), S::Sphere(b)) => a.intersect(b, tol).into(),
            (S::Cylinder(a), S::Torus(b)) => a.intersect(b, tol).into(),
            (S::Cone(a), S::Plane(b)) => a.intersect(b, tol).into(),
            (S::Cone(a), S::Cylinder(b)) => a.intersect(b, tol).into(),
            (S::Cone(a), S::Cone(b)) => a.intersect(b, tol).into(),
            (S::Cone(a), S::Sphere(b)) => a.intersect(b, tol).into(),
            (S::Cone(a), S::Torus(b)) => a.intersect(b, tol).into(),
            (S::Sphere(a), S::Plane(b)) => a.intersect(b, tol).into(),
            (S::Sphere(a), S::Cylinder(b)) => a.intersect(b, tol).into(),
            (S::Sphere(a), S::Cone(b)) => a.intersect(b, tol).into(),
            (S::Sphere(a), S::Sphere(b)) => a.intersect(b, tol).into(),
            (S::Sphere(a), S::Torus(b)) => a.intersect(b, tol).into(),
            (S::Torus(a), S::Plane(b)) => a.intersect(b, tol).into(),
            (S::Torus(a), S::Cylinder(b)) => a.intersect(b, tol).into(),
            (S::Torus(a), S::Cone(b)) => a.intersect(b, tol).into(),
            (S::Torus(a), S::Sphere(b)) => a.intersect(b, tol).into(),
            (S::Torus(a), S::Torus(b)) => a.intersect(b, tol).into(),
        }
    }
}

impl Intersect for Surface {
    type Output = SurfaceIntersection;

    /// Dispatches to the exact intersection of the pair, or reports
    /// [`SurfaceIntersection::NotAnalytic`] when either is a B-spline.
    fn intersect(&self, other: &Surface, tol: Tolerance) -> SurfaceIntersection {
        match (self.as_elementary(), other.as_elementary()) {
            (Some(a), Some(b)) => a.intersect(&b, tol),
            _ => SurfaceIntersection::NotAnalytic,
        }
    }
}

/// Up to `N` items stored inline, read as a slice.
///
/// Holds the pieces of an intersection result without allocating.
/// Its capacity is a proven bound on the pair it serves, so it never
/// fills up.
#[derive(Clone, Copy)]
pub struct InlineList<T: Copy, const N: usize>(Slots<T, N>);

/// The storage of an [`InlineList`]. The first item pushed fills every
/// slot, so the unused slots always hold a valid value.
#[derive(Clone, Copy)]
enum Slots<T: Copy, const N: usize> {
    Empty,
    Filled { items: [T; N], len: u8 },
}

impl<T: Copy, const N: usize> InlineList<T, N> {
    /// An empty list.
    pub const fn new() -> InlineList<T, N> {
        InlineList(Slots::Empty)
    }

    /// Appends `item`.
    ///
    /// # Panics
    ///
    /// Panics if the list already holds `N` items: the capacity is the
    /// most pieces an intersection can produce, so this is a bug.
    pub(crate) fn push(&mut self, item: T) {
        match &mut self.0 {
            Slots::Empty => {
                assert!(N > 0, "InlineList has no capacity");
                self.0 = Slots::Filled {
                    items: [item; N],
                    len: 1,
                };
            }
            Slots::Filled { items, len } => {
                let at = *len as usize;
                assert!(at < N, "InlineList capacity {N} exceeded");
                items[at] = item;
                *len += 1;
            }
        }
    }
}

impl<T: Copy, const N: usize> Default for InlineList<T, N> {
    fn default() -> InlineList<T, N> {
        InlineList::new()
    }
}

impl<T: Copy, const N: usize> Deref for InlineList<T, N> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match &self.0 {
            Slots::Empty => &[],
            Slots::Filled { items, len } => &items[..*len as usize],
        }
    }
}

impl<T: Copy, const N: usize> DerefMut for InlineList<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        match &mut self.0 {
            Slots::Empty => &mut [],
            Slots::Filled { items, len } => &mut items[..*len as usize],
        }
    }
}

impl<T: Copy + PartialEq, const N: usize> PartialEq for InlineList<T, N> {
    fn eq(&self, other: &InlineList<T, N>) -> bool {
        **self == **other
    }
}

impl<T: Copy + std::fmt::Debug, const N: usize> std::fmt::Debug for InlineList<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, T: Copy, const N: usize> IntoIterator for &'a InlineList<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::curves::{Curve3D, ParametricCurve3D, PointExtrema3D};
    use crate::surfaces::{ElementarySurface, ParametricSurface};
    use crate::{
        Cone, Cylinder, Frame3D, Plane, Point3D, Sphere, Torus, TorusConeIntersection, Vector3D,
        marching_intersection,
    };

    const TOL: Tolerance = Tolerance::DEFAULT;

    /// Surfaces in the configurations the exact intersections handle:
    /// coaxial about z, through or across the axis, and a few that are not.
    fn surfaces() -> Vec<ElementarySurface> {
        let z = Vector3D::Z;
        vec![
            Plane::new(Point3D::new(0.0, 0.0, 0.5), z).unwrap().into(),
            // Tangent to the first torus along its top ring, and to the
            // first sphere at its pole.
            Plane::new(Point3D::new(0.0, 0.0, 1.0), z).unwrap().into(),
            Plane::new(Point3D::new(0.0, 0.0, 2.0), z).unwrap().into(),
            Plane::new(Point3D::ORIGIN, Vector3D::Y).unwrap().into(),
            Plane::new(Point3D::new(0.5, 0.0, 0.0), Vector3D::X)
                .unwrap()
                .into(),
            Plane::new(Point3D::new(0.0, 0.0, 0.2), Vector3D::new(0.0, 0.3, 1.0))
                .unwrap()
                .into(),
            Sphere::new(Point3D::ORIGIN, 2.0).unwrap().into(),
            Sphere::new(Point3D::ORIGIN, 4.5).unwrap().into(),
            Sphere::new(Point3D::new(0.0, 0.0, 1.0), 1.5)
                .unwrap()
                .into(),
            Cylinder::new(Point3D::ORIGIN, z, 1.0).unwrap().into(),
            Cylinder::new(Point3D::ORIGIN, z, 4.5).unwrap().into(),
            Cylinder::new(Point3D::new(1.0, 0.0, 0.0), z, 1.0)
                .unwrap()
                .into(),
            Cone::from_frame(Frame3D::WORLD, 0.4, 0.0).unwrap().into(),
            Cone::from_frame(Frame3D::WORLD, 1.4, 0.0).unwrap().into(),
            Cone::new(Point3D::new(0.0, 0.0, 1.0), z, 0.4, 0.5)
                .unwrap()
                .into(),
            Torus::new(Point3D::ORIGIN, z, 4.0, 1.0).unwrap().into(),
            Torus::new(Point3D::new(0.0, 0.0, 0.5), z, 3.5, 1.0)
                .unwrap()
                .into(),
        ]
    }

    /// Points along a curve, over at most `[-3, 3]` of its parameter.
    fn samples(curve: &ElementaryCurve) -> Vec<Point3D> {
        let (lo, hi) = curve.bounds();
        let (lo, hi) = (lo.max(-3.0), hi.min(3.0));
        (0..=8)
            .map(|i| curve.eval_point(lo + (hi - lo) * f64::from(i) / 8.0))
            .collect()
    }

    fn distance_to(curve: &ElementaryCurve, point: Point3D) -> f64 {
        Curve3D::from(*curve)
            .extrema(point, TOL)
            .iter()
            .map(|e| e.distance)
            .fold(f64::INFINITY, f64::min)
    }

    fn on_some_curve(curves: &[(ElementaryCurve, bool)], point: Point3D, within: f64) -> bool {
        curves.iter().any(|(c, _)| distance_to(c, point) <= within)
    }

    /// Every curve and contact point of the result lies on both surfaces.
    fn assert_on_both(result: &SurfaceIntersection, a: &ElementarySurface, b: &ElementarySurface) {
        if let SurfaceIntersection::Meets { curves, points } = result {
            assert!(!curves.is_empty() || !points.is_empty());
            let on_both = |p: Point3D| a.contains(p, TOL) && b.contains(p, TOL);
            for (curve, _) in curves {
                for p in samples(curve) {
                    assert!(on_both(p), "{p:?} of {curve:?} is off {a:?} or {b:?}");
                }
            }
            for &p in points {
                assert!(on_both(p), "contact {p:?} is off {a:?} or {b:?}");
            }
        }
    }

    /// The two results have the same outcome and the same geometry.
    fn assert_same(ab: &SurfaceIntersection, ba: &SurfaceIntersection) {
        match (ab, ba) {
            (
                SurfaceIntersection::Meets {
                    curves: c1,
                    points: p1,
                },
                SurfaceIntersection::Meets {
                    curves: c2,
                    points: p2,
                },
            ) => {
                assert_eq!(
                    (c1.len(), p1.len()),
                    (c2.len(), p2.len()),
                    "{ab:?} vs {ba:?}"
                );
                for (curve, tangent) in c1 {
                    for p in samples(curve) {
                        assert!(on_some_curve(c2, p, TOL.confusion), "{ab:?} vs {ba:?}");
                    }
                    let twin = c2
                        .iter()
                        .find(|(c, _)| distance_to(c, curve.eval_point(0.1)) <= TOL.confusion);
                    assert_eq!(twin.map(|(_, t)| *t), Some(*tangent), "{ab:?} vs {ba:?}");
                }
                for &p in p1 {
                    assert!(p2.iter().any(|&q| (q - p).magnitude() <= TOL.confusion));
                }
            }
            _ => assert_eq!(ab, ba),
        }
    }

    #[test]
    fn test_surface_intersection_is_symmetric_and_on_both() {
        let all = surfaces();
        for a in &all {
            for b in &all {
                let ab = a.intersect(b, TOL);
                assert_same(&ab, &b.intersect(a, TOL));
                assert_on_both(&ab, a, b);
            }
        }
    }

    #[test]
    fn test_surface_intersection_of_a_surface_with_itself_is_coincident() {
        for s in surfaces() {
            assert_eq!(
                s.intersect(&s, TOL),
                SurfaceIntersection::Coincident,
                "{s:?}"
            );
        }
    }

    #[test]
    fn test_surface_intersection_matches_marching() {
        // Marching traces bounded loops: compare on the pairs involving a
        // sphere or a torus, where every exact curve is closed.
        let all = surfaces();
        let closed = |s: &ElementarySurface| {
            matches!(
                s,
                ElementarySurface::Sphere(_) | ElementarySurface::Torus(_)
            )
        };
        let mut compared = 0;
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                if !(closed(a) || closed(b)) {
                    continue;
                }
                let SurfaceIntersection::Meets { curves, .. } = a.intersect(b, TOL) else {
                    continue;
                };
                if curves.is_empty() || curves.iter().any(|(_, tangent)| *tangent) {
                    continue;
                }
                let traced = marching_intersection(a, b, TOL);
                assert!(
                    !traced.is_empty(),
                    "marching found nothing for {a:?} and {b:?}"
                );
                for curve in &traced {
                    let (lo, hi) = ParametricCurve3D::bounds(curve);
                    for k in 0..=16 {
                        let p = curve.eval_point(lo + (hi - lo) * f64::from(k) / 16.0);
                        assert!(
                            on_some_curve(&curves, p, 1e-4),
                            "{p:?} off the exact curves of {a:?} and {b:?}"
                        );
                    }
                }
                compared += 1;
            }
        }
        assert!(compared >= 10, "only {compared} pairs compared");
    }

    #[test]
    fn test_surface_intersection_matches_typed_results() {
        let plane = Plane::new(Point3D::new(0.0, 0.0, 0.5), Vector3D::Z).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
        assert_eq!(
            ElementarySurface::from(plane).intersect(&sphere.into(), TOL),
            SurfaceIntersection::from(plane.intersect(&sphere, TOL))
        );
        assert_eq!(
            ElementarySurface::from(sphere).intersect(&torus.into(), TOL),
            SurfaceIntersection::from(sphere.intersect(&torus, TOL))
        );
    }

    #[test]
    #[allow(deprecated)]
    fn test_deprecated_surface_methods_match_the_trait() {
        let plane = Plane::new(Point3D::new(0.0, 0.0, 0.5), Vector3D::Z).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
        let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 0.0).unwrap();
        let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
        assert_eq!(
            plane.intersect_plane(&plane, TOL),
            plane.intersect(&plane, TOL)
        );
        assert_eq!(
            plane.intersect_sphere(&sphere, TOL),
            plane.intersect(&sphere, TOL)
        );
        assert_eq!(
            plane.intersect_cylinder(&cylinder, TOL),
            plane.intersect(&cylinder, TOL)
        );
        assert_eq!(
            plane.intersect_cone(&cone, TOL),
            plane.intersect(&cone, TOL)
        );
        assert_eq!(
            sphere.intersect_sphere(&sphere, TOL),
            sphere.intersect(&sphere, TOL)
        );
        assert_eq!(
            sphere.intersect_cylinder(&cylinder, TOL),
            sphere.intersect(&cylinder, TOL)
        );
        assert_eq!(
            sphere.intersect_cone(&cone, TOL),
            sphere.intersect(&cone, TOL)
        );
        assert_eq!(
            cylinder.intersect_cylinder(&cylinder, TOL),
            cylinder.intersect(&cylinder, TOL)
        );
        assert_eq!(cone.intersect_cone(&cone, TOL), cone.intersect(&cone, TOL));
        assert_eq!(
            cone.intersect_cylinder(&cylinder, TOL),
            cone.intersect(&cylinder, TOL)
        );
        assert_eq!(
            torus.intersect_plane(&plane, TOL),
            torus.intersect(&plane, TOL)
        );
        assert_eq!(
            torus.intersect_sphere(&sphere, TOL),
            torus.intersect(&sphere, TOL)
        );
        assert_eq!(
            torus.intersect_cylinder(&cylinder, TOL),
            torus.intersect(&cylinder, TOL)
        );
        assert_eq!(
            torus.intersect_cone(&cone, TOL),
            torus.intersect(&cone, TOL)
        );
        assert_eq!(
            torus.intersect_torus(&torus, TOL),
            torus.intersect(&torus, TOL)
        );
    }

    #[test]
    fn test_surface_intersection_holds_four_rings() {
        // Each nappe of a wide cone crosses the tube twice.
        let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
        let cone = Cone::from_frame(Frame3D::WORLD, 1.4, 0.0).unwrap();
        let TorusConeIntersection::Rings(rings) = torus.intersect(&cone, TOL) else {
            panic!("expected rings");
        };
        assert_eq!(rings.len(), 4);
        let SurfaceIntersection::Meets { curves, points } =
            SurfaceIntersection::from(torus.intersect(&cone, TOL))
        else {
            panic!("expected curves");
        };
        assert_eq!((curves.len(), points.len()), (4, 0));
    }

    #[test]
    fn test_surface_intersection_of_a_bspline_is_not_analytic() {
        let plane = Surface::from(Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap());
        let poles = vec![
            vec![Point3D::new(0.0, 0.0, -1.0), Point3D::new(0.0, 1.0, -1.0)],
            vec![Point3D::new(1.0, 0.0, 1.0), Point3D::new(1.0, 1.0, 1.0)],
        ];
        let patch = Surface::from(
            crate::BSplineSurface::new(
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
            .unwrap(),
        );
        assert_eq!(
            plane.intersect(&patch, TOL),
            SurfaceIntersection::NotAnalytic
        );
        assert_eq!(
            patch.intersect(&plane, TOL),
            SurfaceIntersection::NotAnalytic
        );
        assert_eq!(
            plane.intersect(&plane, TOL),
            SurfaceIntersection::Coincident
        );
    }

    #[test]
    fn test_inline_list_empty() {
        let list = InlineList::<f64, 4>::new();
        assert!(list.is_empty());
        assert_eq!(&*list, &[] as &[f64]);
    }

    #[test]
    fn test_inline_list_push_reads_as_slice() {
        let mut list = InlineList::<f64, 4>::new();
        list.push(3.0);
        list.push(1.0);
        list.push(2.0);
        assert_eq!(&*list, &[3.0, 1.0, 2.0]);
        list.sort_by(f64::total_cmp);
        assert_eq!(&*list, &[1.0, 2.0, 3.0]);
        assert_eq!((&list).into_iter().count(), 3);
    }

    #[test]
    fn test_inline_list_fills_to_capacity() {
        let mut list = InlineList::<u8, 4>::new();
        for i in 0..4 {
            list.push(i);
        }
        assert_eq!(&*list, &[0, 1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "capacity 4 exceeded")]
    fn test_inline_list_overflow_panics() {
        let mut list = InlineList::<u8, 4>::new();
        for i in 0..5 {
            list.push(i);
        }
    }

    #[test]
    fn test_inline_list_equality_ignores_spare_slots() {
        // The spare slots hold copies of the first item pushed, which
        // differ between these two lists.
        let mut a = InlineList::<u8, 4>::new();
        a.push(1);
        a.push(2);
        let mut b = InlineList::<u8, 4>::new();
        b.push(2);
        b.push(1);
        b.sort();
        assert_eq!(a, b);
        assert_ne!(a, InlineList::new());
        assert_eq!(format!("{a:?}"), "[1, 2]");
    }

    #[test]
    fn test_inline_list_adds_no_more_than_a_word() {
        // The empty case costs at most one tag word over the items.
        assert!(std::mem::size_of::<InlineList<[f64; 3], 4>>() <= 4 * 24 + 8);
    }
}
