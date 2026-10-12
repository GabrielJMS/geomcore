//! The single intersection entry point.
//!
//! [`Intersect`] is implemented once per exact pair, in both argument
//! orders, with the pair's own result type as its output. The general
//! results of the untyped path hold their pieces in an [`InlineList`], so
//! no intersection result allocates.

use crate::curves::{Curve3D, ElementaryCurve, ParametricCurve3D};
use crate::intersect::intersect_conic_conic;
use crate::surfaces::{ElementarySurface, Surface};
use crate::{
    Circle2D, Circle3D, CircleCircle3DIntersection, CircleSurfaceIntersection,
    ConicConicIntersection, ConicSurfaceHit, ConicSurfaceIntersection, CurveCurveHit, Ellipse3D,
    Hyperbola3D, Line2D, Line3D, LineCircle3DIntersection, LineConic3DIntersection, LineConicHit,
    LineLine3DIntersection, LinePlaneIntersection, LineQuadricIntersection, LineTorusIntersection,
    Parabola3D,
};
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

reversed!(
    Plane => Line3D,
    Sphere => Line3D,
    Cylinder => Line3D,
    Cone => Line3D,
    Torus => Line3D,
    Plane => Circle3D,
    Sphere => Circle3D,
    Cylinder => Circle3D,
    Cone => Circle3D,
    Plane => Ellipse3D,
    Sphere => Ellipse3D,
    Cylinder => Ellipse3D,
    Cone => Ellipse3D,
    Plane => Parabola3D,
    Sphere => Parabola3D,
    Cylinder => Parabola3D,
    Cone => Parabola3D,
    Plane => Hyperbola3D,
    Sphere => Hyperbola3D,
    Cylinder => Hyperbola3D,
    Cone => Hyperbola3D,
);

/// The intersection of a curve and a surface whatever their types: the
/// general result of the untyped path.
///
/// The typed results convert into it with `From`.
///
/// # Examples
///
/// ```
/// use geomcore::curves::Curve3D;
/// use geomcore::surfaces::Surface;
/// use geomcore::{CurveSurfaceIntersection, Intersect, Line3D, Point3D, Sphere, Tolerance, Vector3D};
/// let line = Curve3D::from(Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap());
/// let sphere = Surface::from(Sphere::new(Point3D::ORIGIN, 2.0).unwrap());
/// let CurveSurfaceIntersection::Points(hits) = line.intersect(&sphere, Tolerance::DEFAULT) else {
///     panic!("expected hits");
/// };
/// let parameters: Vec<f64> = hits.iter().map(|hit| hit.parameter).collect();
/// assert_eq!(parameters, [-2.0, 2.0]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CurveSurfaceIntersection {
    /// The curve misses the surface.
    Empty,
    /// The whole curve lies on the surface.
    Coincident,
    /// No closed form for this pair, or one of the shapes is a B-spline.
    /// The caller decides whether to run a numeric intersection.
    NotAnalytic,
    /// The hits, ordered by curve parameter; a multiplicity of 2 or more
    /// marks a tangency.
    Points(InlineList<ConicSurfaceHit, 4>),
}

impl CurveSurfaceIntersection {
    /// The hits ordered by curve parameter, or `Empty` when there are none.
    fn points(hits: impl IntoIterator<Item = ConicSurfaceHit>) -> CurveSurfaceIntersection {
        let mut list = InlineList::new();
        for hit in hits {
            list.push(hit);
        }
        if list.is_empty() {
            return CurveSurfaceIntersection::Empty;
        }
        list.sort_by(|a, b| a.parameter.total_cmp(&b.parameter));
        CurveSurfaceIntersection::Points(list)
    }
}

/// A hit at `parameter` and `point` with the given multiplicity.
fn hit(parameter: f64, point: Point3D, multiplicity: u32) -> ConicSurfaceHit {
    ConicSurfaceHit {
        parameter,
        point,
        multiplicity,
    }
}

impl From<LinePlaneIntersection> for CurveSurfaceIntersection {
    fn from(result: LinePlaneIntersection) -> CurveSurfaceIntersection {
        match result {
            LinePlaneIntersection::Point(t, p) => CurveSurfaceIntersection::points([hit(t, p, 1)]),
            LinePlaneIntersection::Parallel => CurveSurfaceIntersection::Empty,
            LinePlaneIntersection::Coincident => CurveSurfaceIntersection::Coincident,
        }
    }
}

impl From<LineQuadricIntersection> for CurveSurfaceIntersection {
    fn from(result: LineQuadricIntersection) -> CurveSurfaceIntersection {
        match result {
            LineQuadricIntersection::TwoPoints((t1, p1), (t2, p2)) => {
                CurveSurfaceIntersection::points([hit(t1, p1, 1), hit(t2, p2, 1)])
            }
            LineQuadricIntersection::OnePoint(t, p) => {
                CurveSurfaceIntersection::points([hit(t, p, 1)])
            }
            LineQuadricIntersection::Tangent(t, p) => {
                CurveSurfaceIntersection::points([hit(t, p, 2)])
            }
            LineQuadricIntersection::Empty => CurveSurfaceIntersection::Empty,
            LineQuadricIntersection::Coincident => CurveSurfaceIntersection::Coincident,
        }
    }
}

impl From<LineTorusIntersection> for CurveSurfaceIntersection {
    fn from(result: LineTorusIntersection) -> CurveSurfaceIntersection {
        match result {
            LineTorusIntersection::Hits(hits) => CurveSurfaceIntersection::points(
                hits.into_iter()
                    .map(|h| hit(h.line_param, h.point, h.multiplicity)),
            ),
            LineTorusIntersection::Empty => CurveSurfaceIntersection::Empty,
        }
    }
}

impl From<CircleSurfaceIntersection> for CurveSurfaceIntersection {
    fn from(result: CircleSurfaceIntersection) -> CurveSurfaceIntersection {
        match result {
            CircleSurfaceIntersection::Circle(_) => CurveSurfaceIntersection::Coincident,
            CircleSurfaceIntersection::Hits(hits) => CurveSurfaceIntersection::points(hits),
            CircleSurfaceIntersection::Empty => CurveSurfaceIntersection::Empty,
        }
    }
}

impl From<ConicSurfaceIntersection> for CurveSurfaceIntersection {
    fn from(result: ConicSurfaceIntersection) -> CurveSurfaceIntersection {
        match result {
            ConicSurfaceIntersection::Hits(hits) => CurveSurfaceIntersection::points(hits),
            ConicSurfaceIntersection::Coincident => CurveSurfaceIntersection::Coincident,
            ConicSurfaceIntersection::Empty => CurveSurfaceIntersection::Empty,
        }
    }
}

impl Intersect<ElementarySurface> for ElementaryCurve {
    type Output = CurveSurfaceIntersection;

    /// Dispatches to the exact intersection of the pair, or reports
    /// [`CurveSurfaceIntersection::NotAnalytic`] for a circle or conic
    /// against a torus, which has no closed form.
    fn intersect(&self, other: &ElementarySurface, tol: Tolerance) -> CurveSurfaceIntersection {
        use ElementaryCurve as C;
        use ElementarySurface as S;
        match (self, other) {
            (C::Line(c), S::Torus(s)) => c.intersect(s, tol).into(),
            (_, S::Torus(_)) => CurveSurfaceIntersection::NotAnalytic,
            (C::Line(c), S::Plane(s)) => c.intersect(s, tol).into(),
            (C::Line(c), S::Cylinder(s)) => c.intersect(s, tol).into(),
            (C::Line(c), S::Cone(s)) => c.intersect(s, tol).into(),
            (C::Line(c), S::Sphere(s)) => c.intersect(s, tol).into(),
            (C::Circle(c), S::Plane(s)) => c.intersect(s, tol).into(),
            (C::Circle(c), S::Cylinder(s)) => c.intersect(s, tol).into(),
            (C::Circle(c), S::Cone(s)) => c.intersect(s, tol).into(),
            (C::Circle(c), S::Sphere(s)) => c.intersect(s, tol).into(),
            (C::Ellipse(c), S::Plane(s)) => c.intersect(s, tol).into(),
            (C::Ellipse(c), S::Cylinder(s)) => c.intersect(s, tol).into(),
            (C::Ellipse(c), S::Cone(s)) => c.intersect(s, tol).into(),
            (C::Ellipse(c), S::Sphere(s)) => c.intersect(s, tol).into(),
            (C::Parabola(c), S::Plane(s)) => c.intersect(s, tol).into(),
            (C::Parabola(c), S::Cylinder(s)) => c.intersect(s, tol).into(),
            (C::Parabola(c), S::Cone(s)) => c.intersect(s, tol).into(),
            (C::Parabola(c), S::Sphere(s)) => c.intersect(s, tol).into(),
            (C::Hyperbola(c), S::Plane(s)) => c.intersect(s, tol).into(),
            (C::Hyperbola(c), S::Cylinder(s)) => c.intersect(s, tol).into(),
            (C::Hyperbola(c), S::Cone(s)) => c.intersect(s, tol).into(),
            (C::Hyperbola(c), S::Sphere(s)) => c.intersect(s, tol).into(),
        }
    }
}

impl Intersect<ElementaryCurve> for ElementarySurface {
    type Output = CurveSurfaceIntersection;

    /// The same result as in the other argument order.
    fn intersect(&self, other: &ElementaryCurve, tol: Tolerance) -> CurveSurfaceIntersection {
        other.intersect(self, tol)
    }
}

impl Intersect<Surface> for Curve3D {
    type Output = CurveSurfaceIntersection;

    /// Dispatches to the exact intersection of the pair, or reports
    /// [`CurveSurfaceIntersection::NotAnalytic`] when either is a
    /// B-spline.
    fn intersect(&self, other: &Surface, tol: Tolerance) -> CurveSurfaceIntersection {
        match (self.as_elementary(), other.as_elementary()) {
            (Some(c), Some(s)) => c.intersect(&s, tol),
            _ => CurveSurfaceIntersection::NotAnalytic,
        }
    }
}

impl Intersect<Curve3D> for Surface {
    type Output = CurveSurfaceIntersection;

    /// The same result as in the other argument order.
    fn intersect(&self, other: &Curve3D, tol: Tolerance) -> CurveSurfaceIntersection {
        other.intersect(self, tol)
    }
}

reversed!(
    Circle3D => Line3D,
    Ellipse3D => Line3D,
    Parabola3D => Line3D,
    Hyperbola3D => Line3D,
    Circle2D => Line2D,
);

/// The intersection of two curves whatever their types: the general
/// result of the untyped path.
///
/// # Examples
///
/// ```
/// use geomcore::curves::Curve3D;
/// use geomcore::{Circle3D, CurveCurveIntersection, Intersect, Line3D, Point3D, Tolerance, Vector3D};
/// let line = Curve3D::from(Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap());
/// let circle = Curve3D::from(Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap());
/// let CurveCurveIntersection::Points(hits) = line.intersect(&circle, Tolerance::DEFAULT) else {
///     panic!("expected hits");
/// };
/// let on_line: Vec<f64> = hits.iter().map(|hit| hit.first_param).collect();
/// assert_eq!(on_line, [-2.0, 2.0]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CurveCurveIntersection {
    /// The curves do not meet.
    Empty,
    /// The curves coincide.
    Coincident,
    /// No closed form for this pair or configuration, or one of the curves
    /// is a B-spline. The caller decides whether to run a numeric
    /// intersection.
    NotAnalytic,
    /// The hits, ordered by parameter on the first curve; a multiplicity
    /// of 2 or more marks a tangency.
    Points(InlineList<CurveCurveHit, 4>),
}

impl CurveCurveIntersection {
    /// The hits ordered by first parameter, or `Empty` when there are none.
    fn points(hits: impl IntoIterator<Item = CurveCurveHit>) -> CurveCurveIntersection {
        let mut list = InlineList::new();
        for hit in hits {
            list.push(hit);
        }
        if list.is_empty() {
            return CurveCurveIntersection::Empty;
        }
        list.sort_by(|a, b| a.first_param.total_cmp(&b.first_param));
        CurveCurveIntersection::Points(list)
    }

    /// The same intersection seen from the second curve.
    fn swapped(self) -> CurveCurveIntersection {
        match self {
            CurveCurveIntersection::Points(hits) => {
                CurveCurveIntersection::points(hits.iter().map(|h| CurveCurveHit {
                    first_param: h.second_param,
                    second_param: h.first_param,
                    ..*h
                }))
            }
            other => other,
        }
    }
}

/// A hit at `first` on the first curve and `second` on the second.
fn pair(first: f64, point: Point3D, second: f64, multiplicity: u32) -> CurveCurveHit {
    CurveCurveHit {
        first_param: first,
        point,
        second_param: second,
        multiplicity,
    }
}

fn line_line(result: LineLine3DIntersection) -> CurveCurveIntersection {
    match result {
        LineLine3DIntersection::Point(s, p, t) => {
            CurveCurveIntersection::points([pair(s, p, t, 1)])
        }
        LineLine3DIntersection::Parallel(_) | LineLine3DIntersection::Skew { .. } => {
            CurveCurveIntersection::Empty
        }
        LineLine3DIntersection::Coincident => CurveCurveIntersection::Coincident,
    }
}

/// The multiplicity of a lone line hit: the line-circle and line-conic
/// results report one hit as tangent even where the line pierces the
/// curve's plane or crosses a single branch, so tangency is decided by the
/// directions. A near-tangent pair of roots merges at the closest
/// approach, where the directions are parallel.
fn lone_hit_multiplicity(
    line: &Line3D,
    curve: &impl ParametricCurve3D,
    parameter: f64,
    tol: Tolerance,
) -> u32 {
    let a = line.direction();
    let b = curve.eval_derivative(parameter, 1);
    if a.cross(b).magnitude() <= tol.angular * b.magnitude() {
        2
    } else {
        1
    }
}

fn line_circle(
    result: LineCircle3DIntersection,
    line: &Line3D,
    circle: &Circle3D,
    tol: Tolerance,
) -> CurveCurveIntersection {
    let on_circle = |t: f64, p: Point3D| {
        pair(
            t,
            p,
            circle.parameter_in_bounds(circle.parameter_of(p), tol),
            1,
        )
    };
    match result {
        LineCircle3DIntersection::Points((t1, p1), (t2, p2)) => {
            CurveCurveIntersection::points([on_circle(t1, p1), on_circle(t2, p2)])
        }
        LineCircle3DIntersection::Tangent(t, p) => {
            let hit = on_circle(t, p);
            let multiplicity = lone_hit_multiplicity(line, circle, hit.second_param, tol);
            CurveCurveIntersection::points([CurveCurveHit {
                multiplicity,
                ..hit
            }])
        }
        LineCircle3DIntersection::Empty => CurveCurveIntersection::Empty,
    }
}

fn line_conic(
    result: LineConic3DIntersection,
    line: &Line3D,
    conic: &impl ParametricCurve3D,
    tol: Tolerance,
) -> CurveCurveIntersection {
    let from = |h: LineConicHit, m: u32| pair(h.line_param, h.point, h.conic_param, m);
    match result {
        LineConic3DIntersection::Points(h1, h2) => {
            CurveCurveIntersection::points([from(h1, 1), from(h2, 1)])
        }
        LineConic3DIntersection::Tangent(h) => {
            let multiplicity = lone_hit_multiplicity(line, conic, h.conic_param, tol);
            CurveCurveIntersection::points([from(h, multiplicity)])
        }
        LineConic3DIntersection::Empty => CurveCurveIntersection::Empty,
    }
}

fn circle_circle(
    result: CircleCircle3DIntersection,
    a: &Circle3D,
    b: &Circle3D,
    tol: Tolerance,
) -> CurveCurveIntersection {
    let on = |c: &Circle3D, p: Point3D| c.parameter_in_bounds(c.parameter_of(p), tol);
    let at = |p: Point3D, m: u32| pair(on(a, p), p, on(b, p), m);
    match result {
        CircleCircle3DIntersection::Points(p, q) => {
            CurveCurveIntersection::points([at(p, 1), at(q, 1)])
        }
        CircleCircle3DIntersection::Tangent(p) => CurveCurveIntersection::points([at(p, 2)]),
        CircleCircle3DIntersection::Empty => CurveCurveIntersection::Empty,
        CircleCircle3DIntersection::Coincident => CurveCurveIntersection::Coincident,
        // Crossing planes: every hit lies on the line the planes share.
        CircleCircle3DIntersection::NotAnalytic => conic_conic(intersect_conic_conic(a, b, tol)),
    }
}

fn conic_conic(result: ConicConicIntersection) -> CurveCurveIntersection {
    match result {
        ConicConicIntersection::Hits(hits) => CurveCurveIntersection::points(hits),
        ConicConicIntersection::Coincident => CurveCurveIntersection::Coincident,
        ConicConicIntersection::Empty => CurveCurveIntersection::Empty,
    }
}

impl Intersect for ElementaryCurve {
    type Output = CurveCurveIntersection;

    /// Dispatches to the exact intersection of the pair; every pair of
    /// lines, circles and conics has one.
    fn intersect(&self, other: &ElementaryCurve, tol: Tolerance) -> CurveCurveIntersection {
        use ElementaryCurve as C;
        match (self, other) {
            (C::Line(a), C::Line(b)) => line_line(a.intersect(b, tol)),
            (C::Line(a), C::Circle(b)) => line_circle(a.intersect(b, tol), a, b, tol),
            (C::Line(a), C::Ellipse(b)) => line_conic(a.intersect(b, tol), a, b, tol),
            (C::Line(a), C::Parabola(b)) => line_conic(a.intersect(b, tol), a, b, tol),
            (C::Line(a), C::Hyperbola(b)) => line_conic(a.intersect(b, tol), a, b, tol),
            (C::Circle(a), C::Circle(b)) => circle_circle(a.intersect(b, tol), a, b, tol),
            (C::Circle(a), C::Ellipse(b)) => conic_conic(a.intersect(b, tol)),
            (C::Circle(a), C::Parabola(b)) => conic_conic(a.intersect(b, tol)),
            (C::Circle(a), C::Hyperbola(b)) => conic_conic(a.intersect(b, tol)),
            (C::Ellipse(a), C::Circle(b)) => conic_conic(a.intersect(b, tol)),
            (C::Ellipse(a), C::Ellipse(b)) => conic_conic(a.intersect(b, tol)),
            (C::Ellipse(a), C::Parabola(b)) => conic_conic(a.intersect(b, tol)),
            (C::Ellipse(a), C::Hyperbola(b)) => conic_conic(a.intersect(b, tol)),
            (C::Parabola(a), C::Circle(b)) => conic_conic(a.intersect(b, tol)),
            (C::Parabola(a), C::Ellipse(b)) => conic_conic(a.intersect(b, tol)),
            (C::Parabola(a), C::Parabola(b)) => conic_conic(a.intersect(b, tol)),
            (C::Parabola(a), C::Hyperbola(b)) => conic_conic(a.intersect(b, tol)),
            (C::Hyperbola(a), C::Circle(b)) => conic_conic(a.intersect(b, tol)),
            (C::Hyperbola(a), C::Ellipse(b)) => conic_conic(a.intersect(b, tol)),
            (C::Hyperbola(a), C::Parabola(b)) => conic_conic(a.intersect(b, tol)),
            (C::Hyperbola(a), C::Hyperbola(b)) => conic_conic(a.intersect(b, tol)),
            (_, C::Line(_)) => other.intersect(self, tol).swapped(),
        }
    }
}

impl Intersect for Curve3D {
    type Output = CurveCurveIntersection;

    /// Dispatches to the exact intersection of the pair, or reports
    /// [`CurveCurveIntersection::NotAnalytic`] when either is a B-spline.
    fn intersect(&self, other: &Curve3D, tol: Tolerance) -> CurveCurveIntersection {
        match (self.as_elementary(), other.as_elementary()) {
            (Some(a), Some(b)) => a.intersect(&b, tol),
            _ => CurveCurveIntersection::NotAnalytic,
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
    use crate::{Circle2D, Line2D, Point2D, Vector2D};
    use crate::{
        Circle3D, Cone, Cylinder, Ellipse3D, Frame3D, Hyperbola3D, Line3D, Parabola3D, Plane,
        Point3D, Sphere, Torus, TorusConeIntersection, Vector3D, intersect_curve_curve,
        intersect_curve_surface, marching_intersection,
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
            ElementarySurface::from(plane).intersect(&ElementarySurface::from(sphere), TOL),
            SurfaceIntersection::from(plane.intersect(&sphere, TOL))
        );
        assert_eq!(
            ElementarySurface::from(sphere).intersect(&ElementarySurface::from(torus), TOL),
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

    /// Curves crossing, grazing, lying on and missing the fixture surfaces.
    fn curves() -> Vec<ElementaryCurve> {
        let (x, y, z) = (Vector3D::X, Vector3D::Y, Vector3D::Z);
        vec![
            Line3D::new(Point3D::new(-5.0, 0.0, 0.3), x).unwrap().into(),
            Line3D::new(Point3D::new(-5.0, 0.0, 0.0), x).unwrap().into(),
            Line3D::new(Point3D::new(-5.0, 2.0, 0.0), x).unwrap().into(),
            Line3D::new(Point3D::ORIGIN, z).unwrap().into(),
            Line3D::new(Point3D::new(1.0, 0.0, 0.0), z).unwrap().into(),
            Line3D::new(Point3D::new(-3.0, 0.5, -1.0), Vector3D::new(1.0, 0.2, 0.4))
                .unwrap()
                .into(),
            Circle3D::new(Point3D::ORIGIN, z, 1.0).unwrap().into(),
            Circle3D::new(Point3D::new(1.5, 0.0, 0.0), z, 1.0)
                .unwrap()
                .into(),
            Circle3D::new(Point3D::ORIGIN, y, 2.0).unwrap().into(),
            Circle3D::new(Point3D::new(0.5, 0.0, 0.3), x, 1.5)
                .unwrap()
                .into(),
            Ellipse3D::new(Point3D::ORIGIN, z, x, 3.0, 1.5)
                .unwrap()
                .into(),
            Ellipse3D::new(Point3D::new(0.0, 0.0, 0.5), y, x, 3.0, 1.5)
                .unwrap()
                .into(),
            Parabola3D::new(Point3D::ORIGIN, z, x, 1.0).unwrap().into(),
            Parabola3D::new(Point3D::new(0.0, 0.0, -1.0), y, z, 0.5)
                .unwrap()
                .into(),
            Hyperbola3D::new(Point3D::ORIGIN, z, x, 2.0, 1.0)
                .unwrap()
                .into(),
            Hyperbola3D::new(Point3D::new(0.0, 0.0, 0.2), y, x, 1.0, 1.0)
                .unwrap()
                .into(),
        ]
    }

    /// Hits are ordered, on the curve at their parameter, on the surface,
    /// and the whole curve lies on the surface when coincident.
    fn assert_curve_surface_valid(
        result: &CurveSurfaceIntersection,
        curve: &ElementaryCurve,
        surface: &ElementarySurface,
    ) {
        match result {
            CurveSurfaceIntersection::Points(hits) => {
                assert!(!hits.is_empty());
                for pair in hits.windows(2) {
                    assert!(pair[0].parameter <= pair[1].parameter, "{hits:?}");
                }
                for hit in hits {
                    let scale = 1.0 + hit.point.distance(Point3D::ORIGIN);
                    let on_curve = curve.eval_point(hit.parameter);
                    assert!(
                        (on_curve - hit.point).magnitude() <= 1e-9 * scale,
                        "{hit:?} of {curve:?}"
                    );
                    assert!(surface.contains(hit.point, TOL), "{hit:?} off {surface:?}");
                    assert!(hit.multiplicity >= 1);
                }
            }
            CurveSurfaceIntersection::Coincident => {
                for p in samples(curve) {
                    assert!(
                        surface.contains(p, TOL),
                        "{p:?} of {curve:?} off {surface:?}"
                    );
                }
            }
            CurveSurfaceIntersection::Empty | CurveSurfaceIntersection::NotAnalytic => {}
        }
    }

    #[test]
    fn test_curve_surface_intersection_is_symmetric_and_valid() {
        for curve in &curves() {
            for surface in &surfaces() {
                let result = curve.intersect(surface, TOL);
                assert_eq!(result, surface.intersect(curve, TOL));
                assert_curve_surface_valid(&result, curve, surface);
            }
        }
    }

    #[test]
    fn test_curve_surface_intersection_matches_numeric() {
        // The numeric solve seeds an unbounded curve over [-10, 10].
        let window = |c: &ElementaryCurve, t: f64| c.period().is_some() || t.abs() <= 10.0;
        let mut compared = 0;
        for curve in &curves() {
            for surface in &surfaces() {
                let hits = match curve.intersect(surface, TOL) {
                    CurveSurfaceIntersection::Points(hits) => hits,
                    CurveSurfaceIntersection::Empty => InlineList::new(),
                    _ => continue,
                };
                let numeric = intersect_curve_surface(curve, surface, TOL);
                let near = |p: Point3D, q: Point3D| {
                    (p - q).magnitude() <= 1e-6 * (1.0 + p.distance(Point3D::ORIGIN))
                };
                for n in &numeric {
                    assert!(
                        hits.iter().any(|h| near(h.point, n.point)),
                        "numeric {n:?} not exact for {curve:?} and {surface:?}: {hits:?}"
                    );
                }
                for h in hits
                    .iter()
                    .filter(|h| h.multiplicity == 1 && window(curve, h.parameter))
                {
                    assert!(
                        numeric.iter().any(|n| near(h.point, n.point)),
                        "exact {h:?} missed numerically for {curve:?} and {surface:?}"
                    );
                }
                compared += 1;
            }
        }
        assert!(compared >= 100, "only {compared} pairs compared");
    }

    #[test]
    fn test_curve_surface_intersection_holds_four_hits() {
        let line: ElementaryCurve = Line3D::new(Point3D::new(-6.0, 0.0, 0.3), Vector3D::X)
            .unwrap()
            .into();
        let torus: ElementarySurface = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0)
            .unwrap()
            .into();
        let CurveSurfaceIntersection::Points(hits) = line.intersect(&torus, TOL) else {
            panic!("expected hits");
        };
        assert_eq!(hits.len(), 4);
    }

    #[test]
    fn test_curve_surface_intersection_without_closed_form_is_not_analytic() {
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Y, 4.0).unwrap();
        let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
        assert_eq!(
            ElementaryCurve::from(circle).intersect(&ElementarySurface::from(torus), TOL),
            CurveSurfaceIntersection::NotAnalytic
        );
        let bspline = Curve3D::from(
            crate::BSplineCurve3D::new(
                1,
                vec![Point3D::new(0.0, 0.0, -1.0), Point3D::new(0.0, 0.0, 1.0)],
                vec![0.0, 1.0],
                vec![2, 2],
                false,
            )
            .unwrap(),
        );
        let plane = Surface::from(Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap());
        assert_eq!(
            bspline.intersect(&plane, TOL),
            CurveSurfaceIntersection::NotAnalytic
        );
        assert_eq!(
            plane.intersect(&bspline, TOL),
            CurveSurfaceIntersection::NotAnalytic
        );
        let line = Curve3D::from(Line3D::new(Point3D::ORIGIN, Vector3D::Z).unwrap());
        let CurveSurfaceIntersection::Points(hits) = line.intersect(&plane, TOL) else {
            panic!("expected a hit");
        };
        assert_eq!((hits.len(), hits[0].point), (1, Point3D::ORIGIN));
    }

    #[test]
    #[allow(deprecated)]
    fn test_deprecated_curve_surface_methods_match_the_trait() {
        let plane = Plane::new(Point3D::new(0.0, 0.0, 0.5), Vector3D::Z).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
        let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 0.0).unwrap();
        let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
        let line = Line3D::new(Point3D::new(-5.0, 0.0, 0.3), Vector3D::X).unwrap();
        assert_eq!(
            line.intersect_plane(&plane, TOL),
            line.intersect(&plane, TOL)
        );
        assert_eq!(
            line.intersect_sphere(&sphere, TOL),
            line.intersect(&sphere, TOL)
        );
        assert_eq!(
            line.intersect_cylinder(&cylinder, TOL),
            line.intersect(&cylinder, TOL)
        );
        assert_eq!(line.intersect_cone(&cone, TOL), line.intersect(&cone, TOL));
        assert_eq!(
            line.intersect_torus(&torus, TOL),
            line.intersect(&torus, TOL)
        );
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Y, 2.0).unwrap();
        assert_eq!(
            circle.intersect_plane(&plane, TOL),
            circle.intersect(&plane, TOL)
        );
        assert_eq!(
            circle.intersect_sphere(&sphere, TOL),
            circle.intersect(&sphere, TOL)
        );
        assert_eq!(
            circle.intersect_cylinder(&cylinder, TOL),
            circle.intersect(&cylinder, TOL)
        );
        assert_eq!(
            circle.intersect_cone(&cone, TOL),
            circle.intersect(&cone, TOL)
        );
        let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Y, Vector3D::X, 3.0, 1.5).unwrap();
        assert_eq!(
            ellipse.intersect_plane(&plane, TOL),
            ellipse.intersect(&plane, TOL)
        );
        assert_eq!(
            ellipse.intersect_sphere(&sphere, TOL),
            ellipse.intersect(&sphere, TOL)
        );
        assert_eq!(
            ellipse.intersect_cylinder(&cylinder, TOL),
            ellipse.intersect(&cylinder, TOL)
        );
        assert_eq!(
            ellipse.intersect_cone(&cone, TOL),
            ellipse.intersect(&cone, TOL)
        );
        let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Y, Vector3D::X, 1.0).unwrap();
        assert_eq!(
            parabola.intersect_plane(&plane, TOL),
            parabola.intersect(&plane, TOL)
        );
        assert_eq!(
            parabola.intersect_sphere(&sphere, TOL),
            parabola.intersect(&sphere, TOL)
        );
        assert_eq!(
            parabola.intersect_cylinder(&cylinder, TOL),
            parabola.intersect(&cylinder, TOL)
        );
        assert_eq!(
            parabola.intersect_cone(&cone, TOL),
            parabola.intersect(&cone, TOL)
        );
        let hyperbola =
            Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Y, Vector3D::X, 1.0, 1.0).unwrap();
        assert_eq!(
            hyperbola.intersect_plane(&plane, TOL),
            hyperbola.intersect(&plane, TOL)
        );
        assert_eq!(
            hyperbola.intersect_sphere(&sphere, TOL),
            hyperbola.intersect(&sphere, TOL)
        );
        assert_eq!(
            hyperbola.intersect_cylinder(&cylinder, TOL),
            hyperbola.intersect(&cylinder, TOL)
        );
        assert_eq!(
            hyperbola.intersect_cone(&cone, TOL),
            hyperbola.intersect(&cone, TOL)
        );
    }

    fn near(p: Point3D, q: Point3D) -> bool {
        p.distance(q) <= TOL.confusion * (1.0 + p.distance(Point3D::ORIGIN))
    }

    /// Hits are ordered, on both curves at their parameters, and the whole
    /// first curve lies on the second when coincident.
    fn assert_curve_curve_valid(
        result: &CurveCurveIntersection,
        a: &ElementaryCurve,
        b: &ElementaryCurve,
    ) {
        match result {
            CurveCurveIntersection::Points(hits) => {
                assert!(!hits.is_empty());
                for pair in hits.windows(2) {
                    assert!(pair[0].first_param <= pair[1].first_param, "{hits:?}");
                }
                for hit in hits {
                    assert!(
                        near(a.eval_point(hit.first_param), hit.point),
                        "{hit:?} off {a:?}"
                    );
                    assert!(
                        near(b.eval_point(hit.second_param), hit.point),
                        "{hit:?} off {b:?}"
                    );
                    assert!(hit.multiplicity >= 1);
                }
            }
            CurveCurveIntersection::Coincident => {
                for p in samples(a) {
                    assert!(
                        distance_to(b, p) <= TOL.confusion,
                        "{p:?} of {a:?} off {b:?}"
                    );
                }
            }
            CurveCurveIntersection::Empty | CurveCurveIntersection::NotAnalytic => {}
        }
    }

    #[test]
    fn test_curve_curve_intersection_is_symmetric_and_valid() {
        let mut ellipses = curves();
        ellipses.push(
            Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::Y, 3.0, 1.5)
                .unwrap()
                .into(),
        );
        for a in &ellipses {
            for b in &ellipses {
                let ab = a.intersect(b, TOL);
                let ba = b.intersect(a, TOL);
                assert_curve_curve_valid(&ab, a, b);
                match (&ab, &ba) {
                    (CurveCurveIntersection::Points(h1), CurveCurveIntersection::Points(h2)) => {
                        assert_eq!(h1.len(), h2.len(), "{ab:?} vs {ba:?}");
                        for h in h1 {
                            assert!(
                                h2.iter()
                                    .any(|g| near(g.point, h.point)
                                        && g.multiplicity == h.multiplicity),
                                "{ab:?} vs {ba:?}"
                            );
                        }
                    }
                    _ => assert_eq!(ab, ba),
                }
            }
        }
    }

    #[test]
    fn test_curve_curve_intersection_matches_numeric() {
        // Far along an unbounded curve the numeric seeds thin out.
        let window = |c: &ElementaryCurve, t: f64| c.period().is_some() || t.abs() <= 5.0;
        let mut compared = 0;
        for a in &curves() {
            for b in &curves() {
                let hits = match a.intersect(b, TOL) {
                    CurveCurveIntersection::Points(hits) => hits,
                    CurveCurveIntersection::Empty => InlineList::new(),
                    _ => continue,
                };
                let numeric = intersect_curve_curve(a, b, TOL);
                let close = |p: Point3D, q: Point3D| {
                    p.distance(q) <= 1e-6 * (1.0 + p.distance(Point3D::ORIGIN))
                };
                for n in &numeric {
                    assert!(
                        hits.iter().any(|h| close(h.point, n.point)),
                        "numeric {n:?} not exact for {a:?} and {b:?}: {hits:?}"
                    );
                }
                for h in hits.iter().filter(|h| {
                    h.multiplicity == 1 && window(a, h.first_param) && window(b, h.second_param)
                }) {
                    assert!(
                        numeric.iter().any(|n| close(h.point, n.point)),
                        "exact {h:?} missed numerically for {a:?} and {b:?}"
                    );
                }
                compared += 1;
            }
        }
        assert!(compared >= 100, "only {compared} pairs compared");
    }

    #[test]
    fn test_numeric_curve_curve_hits_mark_tangency() {
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let crossing = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let grazing = Line3D::new(Point3D::new(0.0, 2.0, 0.0), Vector3D::X).unwrap();
        let multiplicities = |hits: crate::CurveCurveHits| -> Vec<u32> {
            hits.iter().map(|h| h.multiplicity).collect()
        };
        assert_eq!(
            multiplicities(intersect_curve_curve(&crossing, &circle, TOL)),
            [1, 1]
        );
        assert_eq!(
            multiplicities(intersect_curve_curve(&grazing, &circle, TOL)),
            [2]
        );
    }

    #[test]
    fn test_curve_curve_multiplicity_marks_only_tangency() {
        let multiplicities = |a: ElementaryCurve, b: ElementaryCurve| -> Vec<u32> {
            match a.intersect(&b, TOL) {
                CurveCurveIntersection::Points(hits) => {
                    hits.iter().map(|h| h.multiplicity).collect()
                }
                other => panic!("expected hits, got {other:?}"),
            }
        };
        let circle: ElementaryCurve = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0)
            .unwrap()
            .into();
        let parabola: ElementaryCurve =
            Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0)
                .unwrap()
                .into();
        let line = |origin: Point3D, direction: Vector3D| -> ElementaryCurve {
            Line3D::new(origin, direction).unwrap().into()
        };
        // Through the circle's plane at a point of the circle: a crossing.
        let piercing = line(Point3D::new(2.0, 0.0, -1.0), Vector3D::Z);
        assert_eq!(multiplicities(piercing, circle), [1]);
        assert_eq!(multiplicities(circle, piercing), [1]);
        // In the plane, touching the circle, or within tolerance of it.
        assert_eq!(
            multiplicities(line(Point3D::new(0.0, 2.0, 0.0), Vector3D::X), circle),
            [2]
        );
        assert_eq!(
            multiplicities(
                line(Point3D::new(0.0, 2.0 + 1e-9, 0.0), Vector3D::X),
                circle
            ),
            [2]
        );
        // Parallel to the parabola's axis: one crossing. Along the tangent
        // at the vertex: a tangency.
        assert_eq!(
            multiplicities(line(Point3D::new(0.0, 1.0, 0.0), Vector3D::X), parabola),
            [1]
        );
        assert_eq!(
            multiplicities(line(Point3D::ORIGIN, Vector3D::Y), parabola),
            [2]
        );
        // Coplanar circles touching once.
        let touching: ElementaryCurve =
            Circle3D::new(Point3D::new(3.0, 0.0, 0.0), Vector3D::Z, 1.0)
                .unwrap()
                .into();
        assert_eq!(multiplicities(circle, touching), [2]);
    }

    #[test]
    fn test_curve_curve_intersection_holds_four_hits() {
        let a = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
        let b = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::Y, 3.0, 1.5).unwrap();
        let CurveCurveIntersection::Points(hits) =
            ElementaryCurve::from(a).intersect(&ElementaryCurve::from(b), TOL)
        else {
            panic!("expected hits");
        };
        assert_eq!(hits.len(), 4);
    }

    #[test]
    fn test_curve_curve_intersection_is_not_analytic_only_for_bsplines() {
        let parabola: ElementaryCurve =
            Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0)
                .unwrap()
                .into();
        let hyperbola: ElementaryCurve =
            Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0)
                .unwrap()
                .into();
        let CurveCurveIntersection::Points(hits) = parabola.intersect(&hyperbola, TOL) else {
            panic!("expected the parabola to cross the hyperbola");
        };
        assert_eq!(hits.len(), 2);
        // Circles in crossing planes meet on the line the planes share.
        let flat: ElementaryCurve = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0)
            .unwrap()
            .into();
        // Centered above the x axis, radius √5: crosses it at x = ±2.
        let upright: ElementaryCurve =
            Circle3D::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Y, 5f64.sqrt())
                .unwrap()
                .into();
        let CurveCurveIntersection::Points(hits) = flat.intersect(&upright, TOL) else {
            panic!("expected hits");
        };
        assert_eq!(hits.len(), 2);
        for (h, x) in hits.iter().zip([2.0, -2.0]) {
            assert!(h.point.distance(Point3D::new(x, 0.0, 0.0)) < 1e-12);
            assert_eq!(h.multiplicity, 1);
        }
        let bspline = Curve3D::from(
            crate::BSplineCurve3D::new(
                1,
                vec![Point3D::new(0.0, -1.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
                vec![0.0, 1.0],
                vec![2, 2],
                false,
            )
            .unwrap(),
        );
        let line = Curve3D::from(Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap());
        assert_eq!(
            bspline.intersect(&line, TOL),
            CurveCurveIntersection::NotAnalytic
        );
        assert_eq!(
            line.intersect(&bspline, TOL),
            CurveCurveIntersection::NotAnalytic
        );
        assert_eq!(
            line.intersect(&line, TOL),
            CurveCurveIntersection::Coincident
        );
    }

    #[test]
    #[allow(deprecated)]
    fn test_deprecated_curve_curve_methods_match_the_trait() {
        let line = Line3D::new(Point3D::new(-5.0, 0.0, 0.0), Vector3D::X).unwrap();
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::Y, 3.0, 1.5).unwrap();
        let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
        let hyperbola =
            Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0, 1.0).unwrap();
        assert_eq!(line.intersect_line(&line, TOL), line.intersect(&line, TOL));
        assert_eq!(
            line.intersect_circle(&circle, TOL),
            line.intersect(&circle, TOL)
        );
        assert_eq!(
            line.intersect_ellipse(&ellipse, TOL),
            line.intersect(&ellipse, TOL)
        );
        assert_eq!(
            line.intersect_parabola(&parabola, TOL),
            line.intersect(&parabola, TOL)
        );
        assert_eq!(
            line.intersect_hyperbola(&hyperbola, TOL),
            line.intersect(&hyperbola, TOL)
        );
        assert_eq!(
            circle.intersect_circle(&circle, TOL),
            circle.intersect(&circle, TOL)
        );
        assert_eq!(
            circle.intersect_ellipse(&ellipse, TOL),
            circle.intersect(&ellipse, TOL)
        );
        assert_eq!(
            circle.intersect_parabola(&parabola, TOL),
            circle.intersect(&parabola, TOL)
        );
        assert_eq!(
            circle.intersect_hyperbola(&hyperbola, TOL),
            circle.intersect(&hyperbola, TOL)
        );
        assert_eq!(
            ellipse.intersect_ellipse(&ellipse, TOL),
            ellipse.intersect(&ellipse, TOL)
        );
        let line2 = Line2D::new(Point2D::new(-5.0, 0.0), Vector2D::new(1.0, 0.0)).unwrap();
        let circle2 = Circle2D::new(Point2D::new(0.0, 0.0), 2.0).unwrap();
        assert_eq!(
            line2.intersect_line(&line2, TOL),
            line2.intersect(&line2, TOL)
        );
        assert_eq!(
            line2.intersect_circle(&circle2, TOL),
            line2.intersect(&circle2, TOL)
        );
        assert_eq!(
            circle2.intersect_circle(&circle2, TOL),
            circle2.intersect(&circle2, TOL)
        );
        assert_eq!(
            circle2.intersect(&line2, TOL),
            line2.intersect(&circle2, TOL)
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
    /// Every parameter reported on a circle or ellipse lies in its
    /// `bounds()`, hits come sorted by it, and a hit or projection at the
    /// seam reports its start: through every surface and curve pair, the
    /// projection and the extrema, in random poses.
    #[test]
    fn test_circle_and_ellipse_parameters_lie_in_bounds() {
        let mut state: u64 = 0x5eed_0f15_c0de_cafe;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f64 / 2147483648.0
        };
        let unit = |next: &mut dyn FnMut() -> f64| loop {
            let v = Vector3D::new(2.0 * next() - 1.0, 2.0 * next() - 1.0, 2.0 * next() - 1.0);
            if let Some(v) = v.normalized().filter(|_| v.magnitude() > 0.2) {
                return v;
            }
        };
        let in_bounds = |curve: &ElementaryCurve, t: f64| {
            let (lo, hi) = curve.bounds();
            assert!(lo <= t && t < hi, "{t} outside [{lo}, {hi}) on {curve:?}");
        };
        let mut hits = 0;
        for _ in 0..150 {
            let point = |next: &mut dyn FnMut() -> f64| {
                Point3D::new(2.0 * next() - 1.0, 2.0 * next() - 1.0, 2.0 * next() - 1.0)
            };
            let frame = Frame3D::new(point(&mut next), unit(&mut next), unit(&mut next));
            let Ok(frame) = frame else { continue };
            let (a, b) = (0.6 + next(), 0.3 + 0.3 * next());
            let curves: [ElementaryCurve; 2] = [
                Circle3D::from_frame(frame, a).unwrap().into(),
                Ellipse3D::from_frame(frame, a, b).unwrap().into(),
            ];
            let surfaces: [ElementarySurface; 4] = [
                Plane::new(point(&mut next), unit(&mut next))
                    .unwrap()
                    .into(),
                Sphere::new(point(&mut next), 0.5 + next()).unwrap().into(),
                Cylinder::new(point(&mut next), unit(&mut next), 0.2 + next())
                    .unwrap()
                    .into(),
                Cone::new(point(&mut next), unit(&mut next), 0.2 + next(), 0.5)
                    .unwrap()
                    .into(),
            ];
            let others: [ElementaryCurve; 4] = [
                Line3D::new(point(&mut next), unit(&mut next))
                    .unwrap()
                    .into(),
                // In the curves' plane, so coplanar pairs are exercised.
                Line3D::new(
                    frame.origin(),
                    frame.x_direction() + frame.y_direction() * next(),
                )
                .unwrap()
                .into(),
                Circle3D::from_frame(
                    Frame3D::new(
                        frame.origin() + frame.x_direction() * next(),
                        frame.z_direction(),
                        unit(&mut next),
                    )
                    .unwrap(),
                    0.4 + next(),
                )
                .unwrap()
                .into(),
                Ellipse3D::new(point(&mut next), unit(&mut next), unit(&mut next), 1.0, 0.5)
                    .unwrap()
                    .into(),
            ];
            for curve in &curves {
                for surface in &surfaces {
                    if let CurveSurfaceIntersection::Points(list) = curve.intersect(surface, TOL) {
                        for h in list.iter() {
                            in_bounds(curve, h.parameter);
                            hits += 1;
                        }
                        assert!(list.windows(2).all(|w| w[0].parameter <= w[1].parameter));
                    }
                }
                for other in &others {
                    for (first, second, swapped) in [(curve, other, false), (other, curve, true)] {
                        if let CurveCurveIntersection::Points(list) = first.intersect(second, TOL) {
                            for h in list.iter() {
                                let t = if swapped {
                                    h.second_param
                                } else {
                                    h.first_param
                                };
                                in_bounds(curve, t);
                                hits += 1;
                            }
                        }
                    }
                }
                let query = Point3D::ORIGIN + (point(&mut next) - Point3D::ORIGIN) * 3.0;
                let (projected, extrema) = match curve {
                    ElementaryCurve::Circle(c) => {
                        (c.project_point(query, TOL), c.extrema(query, TOL))
                    }
                    ElementaryCurve::Ellipse(e) => {
                        (e.project_point(query, TOL), e.extrema(query, TOL))
                    }
                    _ => unreachable!(),
                };
                in_bounds(curve, projected.parameter);
                for e in extrema {
                    in_bounds(curve, e.parameter);
                }
            }
        }
        assert!(hits > 500, "only {hits} hits exercised");

        // At the seam: the plane x = 1 grazes the unit circle there, and
        // the plane y = 0 crosses it there and half a turn on.
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
        let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0, 0.5).unwrap();
        for curve in [
            ElementaryCurve::from(circle),
            ElementaryCurve::from(ellipse),
        ] {
            let params = |surface: ElementarySurface| match curve.intersect(&surface, TOL) {
                CurveSurfaceIntersection::Points(list) => {
                    list.iter().map(|h| h.parameter).collect::<Vec<_>>()
                }
                other => panic!("expected hits, got {other:?}"),
            };
            let grazing = Plane::new(Point3D::new(1.0, 0.0, 0.0), Vector3D::X).unwrap();
            assert_eq!(params(grazing.into()), [0.0]);
            let across = Plane::new(Point3D::ORIGIN, Vector3D::Y).unwrap();
            let got = params(across.into());
            assert_eq!(got[0], 0.0);
            assert!((got[1] - std::f64::consts::PI).abs() < 1e-12);
            // A point a hair below the seam projects onto it.
            let below = Point3D::new(1.5, -1e-9, 0.0);
            let projected = match curve {
                ElementaryCurve::Circle(c) => c.project_point(below, TOL),
                ElementaryCurve::Ellipse(e) => e.project_point(below, TOL),
                _ => unreachable!(),
            };
            assert_eq!(projected.parameter, 0.0);
        }
    }
}
