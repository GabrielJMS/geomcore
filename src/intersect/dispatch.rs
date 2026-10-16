//! The exact intersection entry point.
//!
//! [`IntersectExact`] is the zero-cost typed path, implemented once per exact pair, in both argument
//! orders, with the pair's own result type as its output. The general
//! results of the untyped path hold their hits in an [`InlineList`], so
//! a result made of hits does not allocate. The results that hold curves
//! or a variable number of hits do: [`LineTorusIntersection`],
//! [`TorusConeIntersection::Rings`], [`CircleSurfaceIntersection`] and
//! [`ConicConicIntersection`] hold a `Vec`.

use crate::curves::{Curve3D, ElementaryCurve};
use crate::intersect::results::sort_by_known_parameter;
use crate::intersect::{Conic, intersect_conic_conic};
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

/// Exact intersection of two shapes: the typed, stack-only path.
///
/// Each implementation returns the result type of its pair, so the
/// compiler knows which outcomes a pair can produce, and dispatch is
/// static. A configuration with no closed form reports "not analytic";
/// this trait never runs a numeric intersection. To get an answer for
/// every pair, with the method that produced it, use [`Intersect`](crate::intersect::Intersect).
pub trait IntersectExact<Rhs: ?Sized = Self> {
    /// The result of intersecting `Self` with `Rhs`.
    type Output;

    /// Intersects `self` with `other`, deciding tangency and coincidence
    /// with `tol`.
    fn intersect_exact(&self, other: &Rhs, tol: Tolerance) -> Self::Output;
}

/// Implements the reversed argument order of a pair whose result does
/// not depend on the order, by swapping the arguments.
macro_rules! reversed {
    ($($a:ty => $b:ty),* $(,)?) => {$(
        impl IntersectExact<$b> for $a {
            type Output = <$b as IntersectExact<$a>>::Output;

            /// The same result as in the other argument order.
            fn intersect_exact(&self, other: &$b, tol: Tolerance) -> Self::Output {
                other.intersect_exact(self, tol)
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
/// use geomcore::{IntersectExact, Plane, Point3D, Sphere, ExactSurfaceIntersection, Tolerance, Vector3D};
/// let plane = Surface::from(Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap());
/// let sphere = Surface::from(Sphere::new(Point3D::ORIGIN, 2.0).unwrap());
/// match plane.intersect_exact(&sphere, Tolerance::DEFAULT) {
///     ExactSurfaceIntersection::Meets { curves, points } => {
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
pub enum ExactSurfaceIntersection {
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

impl ExactSurfaceIntersection {
    /// Curves, each with its tangency flag.
    fn curves<C: Into<ElementaryCurve>, const K: usize>(
        curves: [(C, bool); K],
    ) -> ExactSurfaceIntersection {
        let mut list = InlineList::new();
        for (curve, tangent) in curves {
            list.push((curve.into(), tangent));
        }
        ExactSurfaceIntersection::Meets {
            curves: list,
            points: InlineList::new(),
        }
    }

    /// A single contact point.
    fn point(point: Point3D) -> ExactSurfaceIntersection {
        let mut points = InlineList::new();
        points.push(point);
        ExactSurfaceIntersection::Meets {
            curves: InlineList::new(),
            points,
        }
    }
}

impl From<PlanePlaneIntersection> for ExactSurfaceIntersection {
    fn from(result: PlanePlaneIntersection) -> ExactSurfaceIntersection {
        match result {
            PlanePlaneIntersection::Line(l) => ExactSurfaceIntersection::curves([(l, false)]),
            PlanePlaneIntersection::Parallel => ExactSurfaceIntersection::Empty,
            PlanePlaneIntersection::Coincident => ExactSurfaceIntersection::Coincident,
        }
    }
}

impl From<PlaneSphereIntersection> for ExactSurfaceIntersection {
    fn from(result: PlaneSphereIntersection) -> ExactSurfaceIntersection {
        match result {
            PlaneSphereIntersection::Circle(c) => ExactSurfaceIntersection::curves([(c, false)]),
            PlaneSphereIntersection::TangentPoint(p) => ExactSurfaceIntersection::point(p),
            PlaneSphereIntersection::Empty => ExactSurfaceIntersection::Empty,
        }
    }
}

impl From<SphereSphereIntersection> for ExactSurfaceIntersection {
    fn from(result: SphereSphereIntersection) -> ExactSurfaceIntersection {
        match result {
            SphereSphereIntersection::Circle(c) => ExactSurfaceIntersection::curves([(c, false)]),
            SphereSphereIntersection::TangentPoint(p) => ExactSurfaceIntersection::point(p),
            SphereSphereIntersection::Empty => ExactSurfaceIntersection::Empty,
            SphereSphereIntersection::Coincident => ExactSurfaceIntersection::Coincident,
        }
    }
}

impl From<PlaneCylinderIntersection> for ExactSurfaceIntersection {
    fn from(result: PlaneCylinderIntersection) -> ExactSurfaceIntersection {
        match result {
            PlaneCylinderIntersection::Circle(c) => ExactSurfaceIntersection::curves([(c, false)]),
            PlaneCylinderIntersection::Ellipse(e) => ExactSurfaceIntersection::curves([(e, false)]),
            PlaneCylinderIntersection::TwoLines(l1, l2) => {
                ExactSurfaceIntersection::curves([(l1, false), (l2, false)])
            }
            PlaneCylinderIntersection::TangentLine(l) => {
                ExactSurfaceIntersection::curves([(l, true)])
            }
            PlaneCylinderIntersection::Empty => ExactSurfaceIntersection::Empty,
        }
    }
}

impl From<PlaneConeIntersection> for ExactSurfaceIntersection {
    fn from(result: PlaneConeIntersection) -> ExactSurfaceIntersection {
        match result {
            PlaneConeIntersection::Circle(c) => ExactSurfaceIntersection::curves([(c, false)]),
            PlaneConeIntersection::Ellipse(e) => ExactSurfaceIntersection::curves([(e, false)]),
            PlaneConeIntersection::Parabola(p) => ExactSurfaceIntersection::curves([(p, false)]),
            PlaneConeIntersection::Hyperbola(h1, h2) => {
                ExactSurfaceIntersection::curves([(h1, false), (h2, false)])
            }
            PlaneConeIntersection::TwoLines(l1, l2) => {
                ExactSurfaceIntersection::curves([(l1, false), (l2, false)])
            }
            PlaneConeIntersection::TangentLine(l) => ExactSurfaceIntersection::curves([(l, true)]),
            PlaneConeIntersection::ApexPoint(p) => ExactSurfaceIntersection::point(p),
        }
    }
}

impl From<SphereCylinderIntersection> for ExactSurfaceIntersection {
    fn from(result: SphereCylinderIntersection) -> ExactSurfaceIntersection {
        match result {
            // A single ring is the equator of a tube as wide as the sphere.
            SphereCylinderIntersection::Circle(c) => ExactSurfaceIntersection::curves([(c, true)]),
            SphereCylinderIntersection::TwoCircles(c1, c2) => {
                ExactSurfaceIntersection::curves([(c1, false), (c2, false)])
            }
            SphereCylinderIntersection::Empty => ExactSurfaceIntersection::Empty,
            SphereCylinderIntersection::NotAnalytic => ExactSurfaceIntersection::NotAnalytic,
        }
    }
}

impl From<SphereConeIntersection> for ExactSurfaceIntersection {
    fn from(result: SphereConeIntersection) -> ExactSurfaceIntersection {
        match result {
            SphereConeIntersection::TwoCircles(c1, c2) => {
                ExactSurfaceIntersection::curves([(c1, false), (c2, false)])
            }
            SphereConeIntersection::TangentCircle(c) => {
                ExactSurfaceIntersection::curves([(c, true)])
            }
            SphereConeIntersection::Empty => ExactSurfaceIntersection::Empty,
            SphereConeIntersection::NotAnalytic => ExactSurfaceIntersection::NotAnalytic,
        }
    }
}

impl From<CylinderCylinderIntersection> for ExactSurfaceIntersection {
    fn from(result: CylinderCylinderIntersection) -> ExactSurfaceIntersection {
        match result {
            CylinderCylinderIntersection::TwoLines(l1, l2) => {
                ExactSurfaceIntersection::curves([(l1, false), (l2, false)])
            }
            CylinderCylinderIntersection::TangentLine(l) => {
                ExactSurfaceIntersection::curves([(l, true)])
            }
            CylinderCylinderIntersection::Empty => ExactSurfaceIntersection::Empty,
            CylinderCylinderIntersection::Coincident => ExactSurfaceIntersection::Coincident,
            CylinderCylinderIntersection::NotAnalytic => ExactSurfaceIntersection::NotAnalytic,
        }
    }
}

impl From<ConeCylinderIntersection> for ExactSurfaceIntersection {
    fn from(result: ConeCylinderIntersection) -> ExactSurfaceIntersection {
        match result {
            ConeCylinderIntersection::TwoCircles(c1, c2) => {
                ExactSurfaceIntersection::curves([(c1, false), (c2, false)])
            }
            ConeCylinderIntersection::ApexPoint(p) => ExactSurfaceIntersection::point(p),
            ConeCylinderIntersection::NotAnalytic => ExactSurfaceIntersection::NotAnalytic,
        }
    }
}

impl From<TorusPlaneIntersection> for ExactSurfaceIntersection {
    fn from(result: TorusPlaneIntersection) -> ExactSurfaceIntersection {
        match result {
            TorusPlaneIntersection::TwoCircles(c1, c2) => {
                ExactSurfaceIntersection::curves([(c1, false), (c2, false)])
            }
            TorusPlaneIntersection::TangentCircle(c) => {
                ExactSurfaceIntersection::curves([(c, true)])
            }
            TorusPlaneIntersection::Circle(c) => ExactSurfaceIntersection::curves([(c, false)]),
            TorusPlaneIntersection::Empty => ExactSurfaceIntersection::Empty,
            TorusPlaneIntersection::NotAnalytic => ExactSurfaceIntersection::NotAnalytic,
        }
    }
}

impl From<TorusSphereIntersection> for ExactSurfaceIntersection {
    fn from(result: TorusSphereIntersection) -> ExactSurfaceIntersection {
        match result {
            TorusSphereIntersection::Circle(c) => ExactSurfaceIntersection::curves([(c, false)]),
            TorusSphereIntersection::TwoCircles(c1, c2) => {
                ExactSurfaceIntersection::curves([(c1, false), (c2, false)])
            }
            TorusSphereIntersection::TangentCircle(c) => {
                ExactSurfaceIntersection::curves([(c, true)])
            }
            TorusSphereIntersection::Empty => ExactSurfaceIntersection::Empty,
            TorusSphereIntersection::NotAnalytic => ExactSurfaceIntersection::NotAnalytic,
        }
    }
}

impl From<TorusCylinderIntersection> for ExactSurfaceIntersection {
    fn from(result: TorusCylinderIntersection) -> ExactSurfaceIntersection {
        match result {
            TorusCylinderIntersection::TwoCircles(c1, c2) => {
                ExactSurfaceIntersection::curves([(c1, false), (c2, false)])
            }
            TorusCylinderIntersection::TangentCircle(c) => {
                ExactSurfaceIntersection::curves([(c, true)])
            }
            TorusCylinderIntersection::Empty => ExactSurfaceIntersection::Empty,
            TorusCylinderIntersection::NotAnalytic => ExactSurfaceIntersection::NotAnalytic,
        }
    }
}

impl From<TorusConeIntersection> for ExactSurfaceIntersection {
    fn from(result: TorusConeIntersection) -> ExactSurfaceIntersection {
        match result {
            TorusConeIntersection::Rings(rings) => {
                let mut curves = InlineList::new();
                for ring in rings {
                    curves.push((ring.circle.into(), ring.tangent));
                }
                ExactSurfaceIntersection::Meets {
                    curves,
                    points: InlineList::new(),
                }
            }
            TorusConeIntersection::Empty => ExactSurfaceIntersection::Empty,
            TorusConeIntersection::NotAnalytic => ExactSurfaceIntersection::NotAnalytic,
        }
    }
}

impl From<TorusTorusIntersection> for ExactSurfaceIntersection {
    fn from(result: TorusTorusIntersection) -> ExactSurfaceIntersection {
        match result {
            TorusTorusIntersection::Circle(c) => ExactSurfaceIntersection::curves([(c, false)]),
            TorusTorusIntersection::TwoCircles(c1, c2) => {
                ExactSurfaceIntersection::curves([(c1, false), (c2, false)])
            }
            TorusTorusIntersection::TangentCircle(c) => {
                ExactSurfaceIntersection::curves([(c, true)])
            }
            TorusTorusIntersection::Empty => ExactSurfaceIntersection::Empty,
            TorusTorusIntersection::Coincident => ExactSurfaceIntersection::Coincident,
            TorusTorusIntersection::NotAnalytic => ExactSurfaceIntersection::NotAnalytic,
        }
    }
}

impl From<ConeConeIntersection> for ExactSurfaceIntersection {
    fn from(result: ConeConeIntersection) -> ExactSurfaceIntersection {
        match result {
            ConeConeIntersection::Circle(c) => ExactSurfaceIntersection::curves([(c, false)]),
            ConeConeIntersection::TwoCircles(c1, c2) => {
                ExactSurfaceIntersection::curves([(c1, false), (c2, false)])
            }
            ConeConeIntersection::ApexPoint(p) => ExactSurfaceIntersection::point(p),
            ConeConeIntersection::Coincident => ExactSurfaceIntersection::Coincident,
            ConeConeIntersection::NotAnalytic => ExactSurfaceIntersection::NotAnalytic,
        }
    }
}

impl IntersectExact for ElementarySurface {
    type Output = ExactSurfaceIntersection;

    /// Dispatches to the exact intersection of the pair.
    fn intersect_exact(
        &self,
        other: &ElementarySurface,
        tol: Tolerance,
    ) -> ExactSurfaceIntersection {
        use ElementarySurface as S;
        match (self, other) {
            (S::Plane(a), S::Plane(b)) => a.intersect_exact(b, tol).into(),
            (S::Plane(a), S::Cylinder(b)) => a.intersect_exact(b, tol).into(),
            (S::Plane(a), S::Cone(b)) => a.intersect_exact(b, tol).into(),
            (S::Plane(a), S::Sphere(b)) => a.intersect_exact(b, tol).into(),
            (S::Plane(a), S::Torus(b)) => a.intersect_exact(b, tol).into(),
            (S::Cylinder(a), S::Plane(b)) => a.intersect_exact(b, tol).into(),
            (S::Cylinder(a), S::Cylinder(b)) => a.intersect_exact(b, tol).into(),
            (S::Cylinder(a), S::Cone(b)) => a.intersect_exact(b, tol).into(),
            (S::Cylinder(a), S::Sphere(b)) => a.intersect_exact(b, tol).into(),
            (S::Cylinder(a), S::Torus(b)) => a.intersect_exact(b, tol).into(),
            (S::Cone(a), S::Plane(b)) => a.intersect_exact(b, tol).into(),
            (S::Cone(a), S::Cylinder(b)) => a.intersect_exact(b, tol).into(),
            (S::Cone(a), S::Cone(b)) => a.intersect_exact(b, tol).into(),
            (S::Cone(a), S::Sphere(b)) => a.intersect_exact(b, tol).into(),
            (S::Cone(a), S::Torus(b)) => a.intersect_exact(b, tol).into(),
            (S::Sphere(a), S::Plane(b)) => a.intersect_exact(b, tol).into(),
            (S::Sphere(a), S::Cylinder(b)) => a.intersect_exact(b, tol).into(),
            (S::Sphere(a), S::Cone(b)) => a.intersect_exact(b, tol).into(),
            (S::Sphere(a), S::Sphere(b)) => a.intersect_exact(b, tol).into(),
            (S::Sphere(a), S::Torus(b)) => a.intersect_exact(b, tol).into(),
            (S::Torus(a), S::Plane(b)) => a.intersect_exact(b, tol).into(),
            (S::Torus(a), S::Cylinder(b)) => a.intersect_exact(b, tol).into(),
            (S::Torus(a), S::Cone(b)) => a.intersect_exact(b, tol).into(),
            (S::Torus(a), S::Sphere(b)) => a.intersect_exact(b, tol).into(),
            (S::Torus(a), S::Torus(b)) => a.intersect_exact(b, tol).into(),
        }
    }
}

impl IntersectExact for Surface {
    type Output = ExactSurfaceIntersection;

    /// Dispatches to the exact intersection of the pair, or reports
    /// [`ExactSurfaceIntersection::NotAnalytic`] when either is a B-spline.
    fn intersect_exact(&self, other: &Surface, tol: Tolerance) -> ExactSurfaceIntersection {
        match (self.as_elementary(), other.as_elementary()) {
            (Some(a), Some(b)) => a.intersect_exact(&b, tol),
            _ => ExactSurfaceIntersection::NotAnalytic,
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
/// use geomcore::{ExactCurveSurfaceIntersection, IntersectExact, Line3D, Point3D, Sphere, Tolerance, Vector3D};
/// let line = Curve3D::from(Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap());
/// let sphere = Surface::from(Sphere::new(Point3D::ORIGIN, 2.0).unwrap());
/// let ExactCurveSurfaceIntersection::Points(hits) = line.intersect_exact(&sphere, Tolerance::DEFAULT) else {
///     panic!("expected hits");
/// };
/// let parameters: Vec<f64> = hits.iter().map(|hit| hit.parameter).collect();
/// assert_eq!(parameters, [-2.0, 2.0]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExactCurveSurfaceIntersection {
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

impl ExactCurveSurfaceIntersection {
    /// The hits ordered by curve parameter, or `Empty` when there are none.
    fn points(hits: impl IntoIterator<Item = ConicSurfaceHit>) -> ExactCurveSurfaceIntersection {
        let mut list = InlineList::new();
        for hit in hits {
            list.push(hit);
        }
        if list.is_empty() {
            return ExactCurveSurfaceIntersection::Empty;
        }
        list.sort_by(|a, b| a.parameter.total_cmp(&b.parameter));
        ExactCurveSurfaceIntersection::Points(list)
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

impl From<LinePlaneIntersection> for ExactCurveSurfaceIntersection {
    fn from(result: LinePlaneIntersection) -> ExactCurveSurfaceIntersection {
        match result {
            LinePlaneIntersection::Point(t, p) => {
                ExactCurveSurfaceIntersection::points([hit(t, p, 1)])
            }
            LinePlaneIntersection::Parallel => ExactCurveSurfaceIntersection::Empty,
            LinePlaneIntersection::Coincident => ExactCurveSurfaceIntersection::Coincident,
        }
    }
}

impl From<LineQuadricIntersection> for ExactCurveSurfaceIntersection {
    fn from(result: LineQuadricIntersection) -> ExactCurveSurfaceIntersection {
        match result {
            LineQuadricIntersection::TwoPoints((t1, p1), (t2, p2)) => {
                ExactCurveSurfaceIntersection::points([hit(t1, p1, 1), hit(t2, p2, 1)])
            }
            LineQuadricIntersection::OnePoint(t, p) => {
                ExactCurveSurfaceIntersection::points([hit(t, p, 1)])
            }
            LineQuadricIntersection::Tangent(t, p) => {
                ExactCurveSurfaceIntersection::points([hit(t, p, 2)])
            }
            LineQuadricIntersection::Empty => ExactCurveSurfaceIntersection::Empty,
            LineQuadricIntersection::Coincident => ExactCurveSurfaceIntersection::Coincident,
        }
    }
}

impl From<LineTorusIntersection> for ExactCurveSurfaceIntersection {
    fn from(result: LineTorusIntersection) -> ExactCurveSurfaceIntersection {
        match result {
            LineTorusIntersection::Hits(hits) => ExactCurveSurfaceIntersection::points(
                hits.into_iter()
                    .map(|h| hit(h.line_parameter, h.point, h.multiplicity)),
            ),
            LineTorusIntersection::Empty => ExactCurveSurfaceIntersection::Empty,
        }
    }
}

impl From<CircleSurfaceIntersection> for ExactCurveSurfaceIntersection {
    fn from(result: CircleSurfaceIntersection) -> ExactCurveSurfaceIntersection {
        match result {
            CircleSurfaceIntersection::Circle(_) => ExactCurveSurfaceIntersection::Coincident,
            CircleSurfaceIntersection::Hits(hits) => ExactCurveSurfaceIntersection::points(hits),
            CircleSurfaceIntersection::Empty => ExactCurveSurfaceIntersection::Empty,
        }
    }
}

impl From<ConicSurfaceIntersection> for ExactCurveSurfaceIntersection {
    fn from(result: ConicSurfaceIntersection) -> ExactCurveSurfaceIntersection {
        match result {
            ConicSurfaceIntersection::Hits(hits) => ExactCurveSurfaceIntersection::points(hits),
            ConicSurfaceIntersection::Coincident => ExactCurveSurfaceIntersection::Coincident,
            ConicSurfaceIntersection::Empty => ExactCurveSurfaceIntersection::Empty,
        }
    }
}

/// The exact intersection of a curve with a surface of type `S` in the
/// general form, solved by the pair's typed solver.
///
/// The router calls it on the concrete types of a pair, so a pair known at
/// compile time skips the untyped enums and their dispatch. The untyped
/// [`IntersectExact`] goes through the same implementations, so both paths
/// return the same result.
pub(crate) trait ExactCurveSurface<S> {
    /// Intersects `self` with `surface`, deciding tangency and coincidence
    /// with `tol`.
    fn exact_curve_surface(&self, surface: &S, tol: Tolerance) -> ExactCurveSurfaceIntersection;
}

/// Implements [`ExactCurveSurface`] for each pair with a closed form
/// (`solved`) through its typed solver, and for each pair without one
/// (`not_analytic`) as [`ExactCurveSurfaceIntersection::NotAnalytic`].
macro_rules! exact_curve_surface {
    (solved: $($curve:ty => [$($surface:ty),*]),*) => {
        $($(
            impl ExactCurveSurface<$surface> for $curve {
                #[inline]
                fn exact_curve_surface(
                    &self,
                    surface: &$surface,
                    tol: Tolerance,
                ) -> ExactCurveSurfaceIntersection {
                    self.intersect_exact(surface, tol).into()
                }
            }
        )*)*
    };
    (not_analytic: $($other:ty => $torus:ty),*) => {
        $(
            impl ExactCurveSurface<$torus> for $other {
                /// A circle or conic against a torus has no closed form.
                #[inline]
                fn exact_curve_surface(
                    &self,
                    _surface: &$torus,
                    _tol: Tolerance,
                ) -> ExactCurveSurfaceIntersection {
                    ExactCurveSurfaceIntersection::NotAnalytic
                }
            }
        )*
    };
}

exact_curve_surface!(
    solved:
    Line3D => [Plane, Cylinder, Cone, Sphere, Torus],
    Circle3D => [Plane, Cylinder, Cone, Sphere],
    Ellipse3D => [Plane, Cylinder, Cone, Sphere],
    Parabola3D => [Plane, Cylinder, Cone, Sphere],
    Hyperbola3D => [Plane, Cylinder, Cone, Sphere]
);

exact_curve_surface!(
    not_analytic:
    Circle3D => Torus,
    Ellipse3D => Torus,
    Parabola3D => Torus,
    Hyperbola3D => Torus
);

/// Implements [`ExactCurveSurface`] between each concrete curve and the
/// untyped [`ElementarySurface`], and between [`ElementaryCurve`] and each
/// concrete surface, by dispatching on the untyped operand.
macro_rules! exact_curve_surface_untyped {
    (curves: $($curve:ty),*; surfaces: $($surface:ty),*) => {
        $(
            impl ExactCurveSurface<ElementarySurface> for $curve {
                #[inline]
                fn exact_curve_surface(
                    &self,
                    surface: &ElementarySurface,
                    tol: Tolerance,
                ) -> ExactCurveSurfaceIntersection {
                    match surface {
                        ElementarySurface::Plane(s) => self.exact_curve_surface(s, tol),
                        ElementarySurface::Cylinder(s) => self.exact_curve_surface(s, tol),
                        ElementarySurface::Cone(s) => self.exact_curve_surface(s, tol),
                        ElementarySurface::Sphere(s) => self.exact_curve_surface(s, tol),
                        ElementarySurface::Torus(s) => self.exact_curve_surface(s, tol),
                    }
                }
            }
        )*
        $(
            impl ExactCurveSurface<$surface> for ElementaryCurve {
                #[inline]
                fn exact_curve_surface(
                    &self,
                    surface: &$surface,
                    tol: Tolerance,
                ) -> ExactCurveSurfaceIntersection {
                    match self {
                        ElementaryCurve::Line(c) => c.exact_curve_surface(surface, tol),
                        ElementaryCurve::Circle(c) => c.exact_curve_surface(surface, tol),
                        ElementaryCurve::Ellipse(c) => c.exact_curve_surface(surface, tol),
                        ElementaryCurve::Parabola(c) => c.exact_curve_surface(surface, tol),
                        ElementaryCurve::Hyperbola(c) => c.exact_curve_surface(surface, tol),
                    }
                }
            }
        )*
    };
}

exact_curve_surface_untyped!(
    curves: Line3D, Circle3D, Ellipse3D, Parabola3D, Hyperbola3D;
    surfaces: Plane, Cylinder, Cone, Sphere, Torus, ElementarySurface
);

impl IntersectExact<ElementarySurface> for ElementaryCurve {
    type Output = ExactCurveSurfaceIntersection;

    /// Dispatches to the exact intersection of the pair, or reports
    /// [`ExactCurveSurfaceIntersection::NotAnalytic`] for a circle or conic
    /// against a torus, which has no closed form.
    fn intersect_exact(
        &self,
        other: &ElementarySurface,
        tol: Tolerance,
    ) -> ExactCurveSurfaceIntersection {
        self.exact_curve_surface(other, tol)
    }
}

impl IntersectExact<ElementaryCurve> for ElementarySurface {
    type Output = ExactCurveSurfaceIntersection;

    /// The same result as in the other argument order.
    fn intersect_exact(
        &self,
        other: &ElementaryCurve,
        tol: Tolerance,
    ) -> ExactCurveSurfaceIntersection {
        other.intersect_exact(self, tol)
    }
}

impl IntersectExact<Surface> for Curve3D {
    type Output = ExactCurveSurfaceIntersection;

    /// Dispatches to the exact intersection of the pair, or reports
    /// [`ExactCurveSurfaceIntersection::NotAnalytic`] when either is a
    /// B-spline.
    fn intersect_exact(&self, other: &Surface, tol: Tolerance) -> ExactCurveSurfaceIntersection {
        match (self.as_elementary(), other.as_elementary()) {
            (Some(c), Some(s)) => c.intersect_exact(&s, tol),
            _ => ExactCurveSurfaceIntersection::NotAnalytic,
        }
    }
}

impl IntersectExact<Curve3D> for Surface {
    type Output = ExactCurveSurfaceIntersection;

    /// The same result as in the other argument order.
    fn intersect_exact(&self, other: &Curve3D, tol: Tolerance) -> ExactCurveSurfaceIntersection {
        other.intersect_exact(self, tol)
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
/// use geomcore::{Circle3D, ExactCurveCurveIntersection, IntersectExact, Line3D, Point3D, Tolerance, Vector3D};
/// let line = Curve3D::from(Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap());
/// let circle = Curve3D::from(Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap());
/// let ExactCurveCurveIntersection::Points(hits) = line.intersect_exact(&circle, Tolerance::DEFAULT) else {
///     panic!("expected hits");
/// };
/// let on_line: Vec<Option<f64>> = hits.iter().map(|hit| hit.first_parameter).collect();
/// assert_eq!(on_line, [Some(-2.0), Some(2.0)]);
/// // The circle's angles are not computed.
/// assert_eq!(hits[0].second_parameter, None);
/// ```
// The hits are stored inline so an exact result never allocates; boxing
// them would allocate on every call.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExactCurveCurveIntersection {
    /// The curves do not meet.
    Empty,
    /// The curves coincide.
    Coincident,
    /// No closed form for this pair or configuration, or one of the curves
    /// is a B-spline. The caller decides whether to run a numeric
    /// intersection.
    NotAnalytic,
    /// The hits, ordered by parameter on the first curve when every hit
    /// stores one, in the solver's order otherwise; a multiplicity of 2 or
    /// more marks a tangency. A parameter the solver did not compute is
    /// `None`.
    Points(InlineList<CurveCurveHit, 4>),
}

impl ExactCurveCurveIntersection {
    /// The hits ordered by first parameter when every hit has one (in the
    /// solver's order otherwise), or `Empty` when there are none.
    fn points(hits: impl IntoIterator<Item = CurveCurveHit>) -> ExactCurveCurveIntersection {
        let mut list = InlineList::new();
        for hit in hits {
            list.push(hit);
        }
        if list.is_empty() {
            return ExactCurveCurveIntersection::Empty;
        }
        sort_by_known_parameter(&mut list, |hit| hit.first_parameter);
        ExactCurveCurveIntersection::Points(list)
    }

    /// The same intersection seen from the second curve.
    fn swapped(self) -> ExactCurveCurveIntersection {
        match self {
            ExactCurveCurveIntersection::Points(hits) => {
                ExactCurveCurveIntersection::points(hits.iter().map(|h| CurveCurveHit {
                    first_parameter: h.second_parameter,
                    second_parameter: h.first_parameter,
                    ..*h
                }))
            }
            other => other,
        }
    }
}

/// A hit at `first` on the first curve and `second` on the second, each
/// `None` when the solver did not compute it.
fn pair(
    first: Option<f64>,
    point: Point3D,
    second: Option<f64>,
    multiplicity: u32,
) -> CurveCurveHit {
    CurveCurveHit {
        first_parameter: first,
        point,
        second_parameter: second,
        multiplicity,
    }
}

fn line_line(result: LineLine3DIntersection) -> ExactCurveCurveIntersection {
    match result {
        LineLine3DIntersection::Point(s, p, t) => {
            ExactCurveCurveIntersection::points([pair(Some(s), p, Some(t), 1)])
        }
        LineLine3DIntersection::Parallel(_) | LineLine3DIntersection::Skew { .. } => {
            ExactCurveCurveIntersection::Empty
        }
        LineLine3DIntersection::Coincident => ExactCurveCurveIntersection::Coincident,
    }
}

/// The multiplicity of a lone line hit: the line-circle and line-conic
/// results report one hit as tangent even where the line pierces the
/// curve's plane or crosses a single branch, so tangency is decided by the
/// directions, parallel across the conic's extent. A near-tangent pair of
/// roots merges at the closest approach, where the directions are
/// parallel.
fn lone_hit_multiplicity(line: &Line3D, conic: &impl Conic, parameter: f64, tol: Tolerance) -> u32 {
    let a = line.direction();
    let b = conic.eval_derivative(parameter, 1);
    let angular = tol.angular_for_extent(conic.shape().extent());
    if a.cross(b).magnitude() <= angular * b.magnitude() {
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
) -> ExactCurveCurveIntersection {
    match result {
        LineCircle3DIntersection::Points((t1, p1), (t2, p2)) => {
            ExactCurveCurveIntersection::points([
                pair(Some(t1), p1, None, 1),
                pair(Some(t2), p2, None, 1),
            ])
        }
        // The tangency test needs the circle's parameter, so it is kept.
        LineCircle3DIntersection::Tangent(t, p) => {
            let angle = circle.parameter_in_bounds(circle.parameter_of(p), tol);
            let multiplicity = lone_hit_multiplicity(line, circle, angle, tol);
            ExactCurveCurveIntersection::points([pair(Some(t), p, Some(angle), multiplicity)])
        }
        LineCircle3DIntersection::Empty => ExactCurveCurveIntersection::Empty,
    }
}

fn line_conic(
    result: LineConic3DIntersection,
    line: &Line3D,
    conic: &impl Conic,
    tol: Tolerance,
) -> ExactCurveCurveIntersection {
    let from =
        |h: LineConicHit, m: u32| pair(Some(h.line_parameter), h.point, Some(h.conic_parameter), m);
    match result {
        LineConic3DIntersection::Points(h1, h2) => {
            ExactCurveCurveIntersection::points([from(h1, 1), from(h2, 1)])
        }
        LineConic3DIntersection::Tangent(h) => {
            let multiplicity = lone_hit_multiplicity(line, conic, h.conic_parameter, tol);
            ExactCurveCurveIntersection::points([from(h, multiplicity)])
        }
        LineConic3DIntersection::Empty => ExactCurveCurveIntersection::Empty,
    }
}

fn circle_circle(
    result: CircleCircle3DIntersection,
    a: &Circle3D,
    b: &Circle3D,
    tol: Tolerance,
) -> ExactCurveCurveIntersection {
    // The solver finds the points only: the angles are left to the caller.
    let at = |p: Point3D, m: u32| pair(None, p, None, m);
    match result {
        CircleCircle3DIntersection::Points(p, q) => {
            ExactCurveCurveIntersection::points([at(p, 1), at(q, 1)])
        }
        CircleCircle3DIntersection::Tangent(p) => ExactCurveCurveIntersection::points([at(p, 2)]),
        CircleCircle3DIntersection::Empty => ExactCurveCurveIntersection::Empty,
        CircleCircle3DIntersection::Coincident => ExactCurveCurveIntersection::Coincident,
        // Crossing planes: every hit lies on the line the planes share.
        CircleCircle3DIntersection::NotAnalytic => conic_conic(intersect_conic_conic(a, b, tol)),
    }
}

fn conic_conic(result: ConicConicIntersection) -> ExactCurveCurveIntersection {
    match result {
        ConicConicIntersection::Hits(hits) => ExactCurveCurveIntersection::points(hits),
        ConicConicIntersection::Coincident => ExactCurveCurveIntersection::Coincident,
        ConicConicIntersection::Empty => ExactCurveCurveIntersection::Empty,
    }
}

/// The exact intersection of a curve with a curve of type `B` in the
/// general form, solved by the pair's typed solver. A hit stores the
/// parameters the solver computed and leaves the others `None`: no
/// parameter is recovered from a point only to be reported.
///
/// The router calls it on the concrete types of a pair, so a pair known at
/// compile time skips the untyped enums and their dispatch. The untyped
/// [`IntersectExact`] goes through the same implementations, so both paths
/// return the same result.
pub(crate) trait ExactCurveCurve<B> {
    /// Intersects `self` with `other`, deciding tangency and coincidence
    /// with `tol`.
    fn exact_curve_curve(&self, other: &B, tol: Tolerance) -> ExactCurveCurveIntersection;
}

/// Implements [`ExactCurveCurve`] for each listed pair with the given body,
/// which reads the pair as `(a, b)`.
macro_rules! exact_curve_curve {
    ($(($first:ty, $second:ty) => |$a:ident, $b:ident, $tol:ident| $body:expr;)*) => {$(
        impl ExactCurveCurve<$second> for $first {
            #[inline]
            fn exact_curve_curve(
                &self,
                other: &$second,
                $tol: Tolerance,
            ) -> ExactCurveCurveIntersection {
                let ($a, $b) = (self, other);
                $body
            }
        }
    )*};
}

exact_curve_curve!(
    (Line3D, Line3D) => |a, b, tol| line_line(a.intersect_exact(b, tol));
    (Line3D, Circle3D) => |a, b, tol| line_circle(a.intersect_exact(b, tol), a, b, tol);
    (Line3D, Ellipse3D) => |a, b, tol| line_conic(a.intersect_exact(b, tol), a, b, tol);
    (Line3D, Parabola3D) => |a, b, tol| line_conic(a.intersect_exact(b, tol), a, b, tol);
    (Line3D, Hyperbola3D) => |a, b, tol| line_conic(a.intersect_exact(b, tol), a, b, tol);
    (Circle3D, Line3D) => |a, b, tol| b.exact_curve_curve(a, tol).swapped();
    (Ellipse3D, Line3D) => |a, b, tol| b.exact_curve_curve(a, tol).swapped();
    (Parabola3D, Line3D) => |a, b, tol| b.exact_curve_curve(a, tol).swapped();
    (Hyperbola3D, Line3D) => |a, b, tol| b.exact_curve_curve(a, tol).swapped();
    (Circle3D, Circle3D) => |a, b, tol| circle_circle(a.intersect_exact(b, tol), a, b, tol);
    (Circle3D, Ellipse3D) => |a, b, tol| conic_conic(a.intersect_exact(b, tol));
    (Circle3D, Parabola3D) => |a, b, tol| conic_conic(a.intersect_exact(b, tol));
    (Circle3D, Hyperbola3D) => |a, b, tol| conic_conic(a.intersect_exact(b, tol));
    (Ellipse3D, Circle3D) => |a, b, tol| conic_conic(a.intersect_exact(b, tol));
    (Ellipse3D, Ellipse3D) => |a, b, tol| conic_conic(a.intersect_exact(b, tol));
    (Ellipse3D, Parabola3D) => |a, b, tol| conic_conic(a.intersect_exact(b, tol));
    (Ellipse3D, Hyperbola3D) => |a, b, tol| conic_conic(a.intersect_exact(b, tol));
    (Parabola3D, Circle3D) => |a, b, tol| conic_conic(a.intersect_exact(b, tol));
    (Parabola3D, Ellipse3D) => |a, b, tol| conic_conic(a.intersect_exact(b, tol));
    (Parabola3D, Parabola3D) => |a, b, tol| conic_conic(a.intersect_exact(b, tol));
    (Parabola3D, Hyperbola3D) => |a, b, tol| conic_conic(a.intersect_exact(b, tol));
    (Hyperbola3D, Circle3D) => |a, b, tol| conic_conic(a.intersect_exact(b, tol));
    (Hyperbola3D, Ellipse3D) => |a, b, tol| conic_conic(a.intersect_exact(b, tol));
    (Hyperbola3D, Parabola3D) => |a, b, tol| conic_conic(a.intersect_exact(b, tol));
    (Hyperbola3D, Hyperbola3D) => |a, b, tol| conic_conic(a.intersect_exact(b, tol));
);

/// Implements [`ExactCurveCurve`] between each concrete curve and the
/// untyped [`ElementaryCurve`], in both orders, by dispatching on the
/// untyped operand.
macro_rules! exact_curve_curve_untyped {
    ($($curve:ty),*) => {$(
        impl ExactCurveCurve<ElementaryCurve> for $curve {
            #[inline]
            fn exact_curve_curve(
                &self,
                other: &ElementaryCurve,
                tol: Tolerance,
            ) -> ExactCurveCurveIntersection {
                match other {
                    ElementaryCurve::Line(b) => self.exact_curve_curve(b, tol),
                    ElementaryCurve::Circle(b) => self.exact_curve_curve(b, tol),
                    ElementaryCurve::Ellipse(b) => self.exact_curve_curve(b, tol),
                    ElementaryCurve::Parabola(b) => self.exact_curve_curve(b, tol),
                    ElementaryCurve::Hyperbola(b) => self.exact_curve_curve(b, tol),
                }
            }
        }

        impl ExactCurveCurve<$curve> for ElementaryCurve {
            #[inline]
            fn exact_curve_curve(
                &self,
                other: &$curve,
                tol: Tolerance,
            ) -> ExactCurveCurveIntersection {
                match self {
                    ElementaryCurve::Line(a) => a.exact_curve_curve(other, tol),
                    ElementaryCurve::Circle(a) => a.exact_curve_curve(other, tol),
                    ElementaryCurve::Ellipse(a) => a.exact_curve_curve(other, tol),
                    ElementaryCurve::Parabola(a) => a.exact_curve_curve(other, tol),
                    ElementaryCurve::Hyperbola(a) => a.exact_curve_curve(other, tol),
                }
            }
        }
    )*};
}

exact_curve_curve_untyped!(Line3D, Circle3D, Ellipse3D, Parabola3D, Hyperbola3D);

impl ExactCurveCurve<ElementaryCurve> for ElementaryCurve {
    #[inline]
    fn exact_curve_curve(
        &self,
        other: &ElementaryCurve,
        tol: Tolerance,
    ) -> ExactCurveCurveIntersection {
        match self {
            ElementaryCurve::Line(a) => a.exact_curve_curve(other, tol),
            ElementaryCurve::Circle(a) => a.exact_curve_curve(other, tol),
            ElementaryCurve::Ellipse(a) => a.exact_curve_curve(other, tol),
            ElementaryCurve::Parabola(a) => a.exact_curve_curve(other, tol),
            ElementaryCurve::Hyperbola(a) => a.exact_curve_curve(other, tol),
        }
    }
}

impl IntersectExact for ElementaryCurve {
    type Output = ExactCurveCurveIntersection;

    /// Dispatches to the exact intersection of the pair; every pair of
    /// lines, circles and conics has one.
    fn intersect_exact(
        &self,
        other: &ElementaryCurve,
        tol: Tolerance,
    ) -> ExactCurveCurveIntersection {
        self.exact_curve_curve(other, tol)
    }
}

impl IntersectExact for Curve3D {
    type Output = ExactCurveCurveIntersection;

    /// Dispatches to the exact intersection of the pair, or reports
    /// [`ExactCurveCurveIntersection::NotAnalytic`] when either is a B-spline.
    fn intersect_exact(&self, other: &Curve3D, tol: Tolerance) -> ExactCurveCurveIntersection {
        match (self.as_elementary(), other.as_elementary()) {
            (Some(a), Some(b)) => a.intersect_exact(&b, tol),
            _ => ExactCurveCurveIntersection::NotAnalytic,
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
    use crate::intersect::marching::marched_curves;
    use crate::surfaces::{ElementarySurface, ParametricSurface};

    use crate::{
        Circle3D, Cone, Cylinder, Ellipse3D, Frame3D, Hyperbola3D, Line3D, Parabola3D, Plane,
        Point3D, Sphere, Torus, TorusConeIntersection, Vector3D, search_curve_curve,
        search_curve_surface,
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
    fn assert_on_both(
        result: &ExactSurfaceIntersection,
        a: &ElementarySurface,
        b: &ElementarySurface,
    ) {
        if let ExactSurfaceIntersection::Meets { curves, points } = result {
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
    fn assert_same(ab: &ExactSurfaceIntersection, ba: &ExactSurfaceIntersection) {
        match (ab, ba) {
            (
                ExactSurfaceIntersection::Meets {
                    curves: c1,
                    points: p1,
                },
                ExactSurfaceIntersection::Meets {
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
                let ab = a.intersect_exact(b, TOL);
                assert_same(&ab, &b.intersect_exact(a, TOL));
                assert_on_both(&ab, a, b);
            }
        }
    }

    #[test]
    fn test_surface_intersection_of_a_surface_with_itself_is_coincident() {
        for s in surfaces() {
            assert_eq!(
                s.intersect_exact(&s, TOL),
                ExactSurfaceIntersection::Coincident,
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
                let ExactSurfaceIntersection::Meets { curves, .. } = a.intersect_exact(b, TOL)
                else {
                    continue;
                };
                if curves.is_empty() || curves.iter().any(|(_, tangent)| *tangent) {
                    continue;
                }
                let traced = marched_curves(a, b, TOL);
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
            ElementarySurface::from(plane).intersect_exact(&ElementarySurface::from(sphere), TOL),
            ExactSurfaceIntersection::from(plane.intersect_exact(&sphere, TOL))
        );
        assert_eq!(
            ElementarySurface::from(sphere).intersect_exact(&ElementarySurface::from(torus), TOL),
            ExactSurfaceIntersection::from(sphere.intersect_exact(&torus, TOL))
        );
    }

    #[test]
    fn test_surface_intersection_holds_four_rings() {
        // Each nappe of a wide cone crosses the tube twice.
        let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
        let cone = Cone::from_frame(Frame3D::WORLD, 1.4, 0.0).unwrap();
        let TorusConeIntersection::Rings(rings) = torus.intersect_exact(&cone, TOL) else {
            panic!("expected rings");
        };
        assert_eq!(rings.len(), 4);
        let ExactSurfaceIntersection::Meets { curves, points } =
            ExactSurfaceIntersection::from(torus.intersect_exact(&cone, TOL))
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
            plane.intersect_exact(&patch, TOL),
            ExactSurfaceIntersection::NotAnalytic
        );
        assert_eq!(
            patch.intersect_exact(&plane, TOL),
            ExactSurfaceIntersection::NotAnalytic
        );
        assert_eq!(
            plane.intersect_exact(&plane, TOL),
            ExactSurfaceIntersection::Coincident
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
        result: &ExactCurveSurfaceIntersection,
        curve: &ElementaryCurve,
        surface: &ElementarySurface,
    ) {
        match result {
            ExactCurveSurfaceIntersection::Points(hits) => {
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
            ExactCurveSurfaceIntersection::Coincident => {
                for p in samples(curve) {
                    assert!(
                        surface.contains(p, TOL),
                        "{p:?} of {curve:?} off {surface:?}"
                    );
                }
            }
            ExactCurveSurfaceIntersection::Empty | ExactCurveSurfaceIntersection::NotAnalytic => {}
        }
    }

    #[test]
    fn test_curve_surface_intersection_is_symmetric_and_valid() {
        for curve in &curves() {
            for surface in &surfaces() {
                let result = curve.intersect_exact(surface, TOL);
                assert_eq!(result, surface.intersect_exact(curve, TOL));
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
                let hits = match curve.intersect_exact(surface, TOL) {
                    ExactCurveSurfaceIntersection::Points(hits) => hits,
                    ExactCurveSurfaceIntersection::Empty => InlineList::new(),
                    _ => continue,
                };
                let numeric = search_curve_surface(curve, surface, TOL);
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
        let ExactCurveSurfaceIntersection::Points(hits) = line.intersect_exact(&torus, TOL) else {
            panic!("expected hits");
        };
        assert_eq!(hits.len(), 4);
    }

    #[test]
    fn test_curve_surface_intersection_without_closed_form_is_not_analytic() {
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Y, 4.0).unwrap();
        let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
        assert_eq!(
            ElementaryCurve::from(circle).intersect_exact(&ElementarySurface::from(torus), TOL),
            ExactCurveSurfaceIntersection::NotAnalytic
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
            bspline.intersect_exact(&plane, TOL),
            ExactCurveSurfaceIntersection::NotAnalytic
        );
        assert_eq!(
            plane.intersect_exact(&bspline, TOL),
            ExactCurveSurfaceIntersection::NotAnalytic
        );
        let line = Curve3D::from(Line3D::new(Point3D::ORIGIN, Vector3D::Z).unwrap());
        let ExactCurveSurfaceIntersection::Points(hits) = line.intersect_exact(&plane, TOL) else {
            panic!("expected a hit");
        };
        assert_eq!((hits.len(), hits[0].point), (1, Point3D::ORIGIN));
    }

    fn near(p: Point3D, q: Point3D) -> bool {
        p.distance(q) <= TOL.confusion * (1.0 + p.distance(Point3D::ORIGIN))
    }

    /// Hits are ordered, on both curves at their parameters, and the whole
    /// first curve lies on the second when coincident.
    fn assert_curve_curve_valid(
        result: &ExactCurveCurveIntersection,
        a: &ElementaryCurve,
        b: &ElementaryCurve,
    ) {
        match result {
            ExactCurveCurveIntersection::Points(hits) => {
                assert!(!hits.is_empty());
                // Sorted by first parameter when every hit stores one.
                if hits.iter().all(|h| h.first_parameter.is_some()) {
                    for pair in hits.windows(2) {
                        assert!(
                            pair[0].first_parameter <= pair[1].first_parameter,
                            "{hits:?}"
                        );
                    }
                }
                for hit in hits {
                    if let Some(t) = hit.first_parameter {
                        assert!(near(a.eval_point(t), hit.point), "{hit:?} off {a:?}");
                    }
                    if let Some(t) = hit.second_parameter {
                        assert!(near(b.eval_point(t), hit.point), "{hit:?} off {b:?}");
                    }
                    assert!(near(a.eval_point(a.parameter_of(hit.point)), hit.point));
                    assert!(near(b.eval_point(b.parameter_of(hit.point)), hit.point));
                    assert!(hit.multiplicity >= 1);
                }
            }
            ExactCurveCurveIntersection::Coincident => {
                for p in samples(a) {
                    assert!(
                        distance_to(b, p) <= TOL.confusion,
                        "{p:?} of {a:?} off {b:?}"
                    );
                }
            }
            ExactCurveCurveIntersection::Empty | ExactCurveCurveIntersection::NotAnalytic => {}
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
                let ab = a.intersect_exact(b, TOL);
                let ba = b.intersect_exact(a, TOL);
                assert_curve_curve_valid(&ab, a, b);
                match (&ab, &ba) {
                    (
                        ExactCurveCurveIntersection::Points(h1),
                        ExactCurveCurveIntersection::Points(h2),
                    ) => {
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
                let hits = match a.intersect_exact(b, TOL) {
                    ExactCurveCurveIntersection::Points(hits) => hits,
                    ExactCurveCurveIntersection::Empty => InlineList::new(),
                    _ => continue,
                };
                let numeric = search_curve_curve(a, b, TOL);
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
                    h.multiplicity == 1
                        && window(
                            a,
                            h.first_parameter.unwrap_or_else(|| a.parameter_of(h.point)),
                        )
                        && window(
                            b,
                            h.second_parameter
                                .unwrap_or_else(|| b.parameter_of(h.point)),
                        )
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
        let multiplicities =
            |hits: crate::intersect::curve_curve_hull::CurveCurveHits| -> Vec<u32> {
                hits.iter().map(|h| h.multiplicity).collect()
            };
        assert_eq!(
            multiplicities(search_curve_curve(&crossing, &circle, TOL)),
            [1, 1]
        );
        assert_eq!(
            multiplicities(search_curve_curve(&grazing, &circle, TOL)),
            [2]
        );
    }

    #[test]
    fn test_curve_curve_multiplicity_marks_only_tangency() {
        let multiplicities = |a: ElementaryCurve, b: ElementaryCurve| -> Vec<u32> {
            match a.intersect_exact(&b, TOL) {
                ExactCurveCurveIntersection::Points(hits) => {
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
        let ExactCurveCurveIntersection::Points(hits) =
            ElementaryCurve::from(a).intersect_exact(&ElementaryCurve::from(b), TOL)
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
        let ExactCurveCurveIntersection::Points(hits) = parabola.intersect_exact(&hyperbola, TOL)
        else {
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
        let ExactCurveCurveIntersection::Points(hits) = flat.intersect_exact(&upright, TOL) else {
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
            bspline.intersect_exact(&line, TOL),
            ExactCurveCurveIntersection::NotAnalytic
        );
        assert_eq!(
            line.intersect_exact(&bspline, TOL),
            ExactCurveCurveIntersection::NotAnalytic
        );
        assert_eq!(
            line.intersect_exact(&line, TOL),
            ExactCurveCurveIntersection::Coincident
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
                    if let ExactCurveSurfaceIntersection::Points(list) =
                        curve.intersect_exact(surface, TOL)
                    {
                        for h in list.iter() {
                            in_bounds(curve, h.parameter);
                            hits += 1;
                        }
                        assert!(list.windows(2).all(|w| w[0].parameter <= w[1].parameter));
                    }
                }
                for other in &others {
                    for (first, second, swapped) in [(curve, other, false), (other, curve, true)] {
                        if let ExactCurveCurveIntersection::Points(list) =
                            first.intersect_exact(second, TOL)
                        {
                            for h in list.iter() {
                                let stored = if swapped {
                                    h.second_parameter
                                } else {
                                    h.first_parameter
                                };
                                if let Some(t) = stored {
                                    in_bounds(curve, t);
                                }
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
            let params = |surface: ElementarySurface| match curve.intersect_exact(&surface, TOL) {
                ExactCurveSurfaceIntersection::Points(list) => {
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
