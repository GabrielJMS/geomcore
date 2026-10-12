//! Analytic intersection result types.
//!
//! [`Intersect::intersect`] on two analytic curves or surfaces returns one
//! of these enums: the intersecting geometry on success, or an explicit
//! classification when the pair is tangent, disjoint, parallel, or
//! coincident. Tangency and coincidence are decided with
//! [`crate::Tolerance`], so near-degenerate configurations classify
//! robustly instead of collapsing to noise.
//!
//! When the types are only known at run time, the same call on
//! [`crate::surfaces::Surface`], [`crate::curves::Curve3D`] or their
//! elementary enums returns a general result instead:
//! [`SurfaceIntersection`], [`CurveSurfaceIntersection`] or
//! [`CurveCurveIntersection`].

use crate::curves::{BSplineCurve3D, InterpParametrization, ParametricCurve3D};
use crate::math::{
    brent_minimum, newton_3d, newton_min_1d, newton_min_2d, solve_2x2, solve_3x3, solve_4x4,
};
use crate::surfaces::ParametricSurface;
use crate::{
    Circle3D, Ellipse3D, Hyperbola3D, Line3D, Parabola3D, Plane, Point2D, Point3D, Tolerance,
    Vector3D,
};

mod conic;
mod curve_curve;
mod curve_surface;
mod dispatch;
mod pieces;
pub(crate) use conic::{intersect_conic_conic, intersect_conic_quadric};
pub use dispatch::{
    CurveCurveIntersection, CurveSurfaceIntersection, InlineList, Intersect, SurfaceIntersection,
};

/// Result of intersecting two planes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlanePlaneIntersection {
    /// The planes meet in a line.
    Line(Line3D),
    /// The planes are parallel but distinct.
    Parallel,
    /// The planes coincide.
    Coincident,
}

/// Result of intersecting a plane with a sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlaneSphereIntersection {
    /// The plane cuts the sphere in a circle.
    Circle(Circle3D),
    /// The plane grazes the sphere in a single point.
    TangentPoint(Point3D),
    /// The plane misses the sphere.
    Empty,
}

/// Result of intersecting two spheres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SphereSphereIntersection {
    /// The spheres meet in a circle.
    Circle(Circle3D),
    /// The spheres touch in a single point.
    TangentPoint(Point3D),
    /// The spheres are separate, one inside the other without contact, or
    /// concentric with different radii.
    Empty,
    /// The spheres coincide (same center and radius within tolerance).
    Coincident,
}

/// Result of intersecting a plane with a cylinder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlaneCylinderIntersection {
    /// The plane is perpendicular to the axis: a circle.
    Circle(Circle3D),
    /// The general oblique section: an ellipse.
    Ellipse(Ellipse3D),
    /// The plane is parallel to the axis and cuts two generators.
    TwoLines(Line3D, Line3D),
    /// The plane is parallel to the axis and grazes one generator.
    TangentLine(Line3D),
    /// The plane is parallel to the axis and misses the cylinder.
    Empty,
}

/// Result of intersecting a plane with a (double) cone.
///
/// A plane missing the apex always meets the cone: in a conic on one
/// nappe, or a two-branch hyperbola across both.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlaneConeIntersection {
    /// The plane is perpendicular to the axis: a circle.
    Circle(Circle3D),
    /// The plane meets one nappe in an ellipse.
    Ellipse(Ellipse3D),
    /// The plane is parallel to a generator: a parabola.
    Parabola(Parabola3D),
    /// The plane cuts both nappes: the two branches of a hyperbola, the
    /// second the first turned half a turn about its centre.
    Hyperbola(Hyperbola3D, Hyperbola3D),
    /// The plane passes through the apex and contains two generators.
    TwoLines(Line3D, Line3D),
    /// The plane passes through the apex along a single generator.
    TangentLine(Line3D),
    /// The plane passes through the apex without meeting a generator.
    ApexPoint(Point3D),
}

/// Result of intersecting a line with a plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinePlaneIntersection {
    /// Transversal hit: the line parameter and the point.
    Point(f64, Point3D),
    /// The line is parallel to the plane but offset from it.
    Parallel,
    /// The line lies in the plane.
    Coincident,
}

/// Result of intersecting a line with a sphere, cylinder, or cone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineQuadricIntersection {
    /// Two transversal hits: `(parameter, point)` pairs, ordered.
    TwoPoints((f64, Point3D), (f64, Point3D)),
    /// A single transversal hit: the linear case (line parallel to a
    /// cylinder axis or cone generator).
    OnePoint(f64, Point3D),
    /// Grazing contact: the line parameter and the point.
    Tangent(f64, Point3D),
    /// No intersection.
    Empty,
    /// The line lies entirely on the surface (a cylinder or cone
    /// generator; a line in a plane reports through
    /// [`LinePlaneIntersection::Coincident`] instead).
    Coincident,
}

/// Result of intersecting a 3D line with a torus: up to four hits.
///
/// Substituting the unit-speed parametrization into the torus implicit
/// equation gives a quartic with leading coefficient exactly 1 (so, unlike
/// the quadratic solver, there is no linear/degenerate arm); roots cluster
/// within `tol.confusion * (1 + |t|)`, `t` measured from the line's
/// closest approach to the torus centre, and verify with
/// [`crate::Torus::contains`].
#[derive(Debug, Clone, PartialEq)]
pub enum LineTorusIntersection {
    /// Isolated hits with multiplicity, ordered by line parameter.
    Hits(Vec<LineTorusHit>),
    /// No intersection.
    Empty,
}

/// One hit of an analytic line-vs-torus solve: the line parameter, the
/// hit point, the torus `(u, v)` parameters, and the root multiplicity
/// (2+ marks grazing tangency).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineTorusHit {
    /// Parameter on the line.
    pub line_param: f64,
    /// The hit point.
    pub point: Point3D,
    /// Parameters on the torus.
    pub surface_params: (f64, f64),
    /// Root multiplicity (≥ 2 signals tangency).
    pub multiplicity: u32,
}

/// Result of intersecting a sphere with a cylinder.
///
/// The analytic path requires the cylinder axis through the sphere
/// center (within `tol.confusion`); otherwise the section is a space
/// quartic and reports [`SphereCylinderIntersection::NotAnalytic`],
/// reserved for the numeric surface-surface path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SphereCylinderIntersection {
    /// Latitude circle(s): one tangent ring or a symmetric pair.
    Circle(Circle3D),
    /// Two latitude rings symmetric about the equatorial plane.
    TwoCircles(Circle3D, Circle3D),
    /// The cylinder misses the sphere.
    Empty,
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting a sphere with a cone.
///
/// The analytic path requires the sphere center on the cone axis
/// (within `tol.confusion`); otherwise the section is a space quartic
/// and reports [`SphereConeIntersection::NotAnalytic`], reserved for the
/// numeric surface-surface path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SphereConeIntersection {
    /// Two latitude circles (on one nappe or one on each).
    TwoCircles(Circle3D, Circle3D),
    /// Grazing contact ring.
    TangentCircle(Circle3D),
    /// The sphere misses the cone.
    Empty,
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting two cylinders.
///
/// The analytic path requires parallel axes (within `tol.angular`);
/// skew or crossing axes give a space quartic and report
/// [`CylinderCylinderIntersection::NotAnalytic`], reserved for the
/// numeric surface-surface path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CylinderCylinderIntersection {
    /// Two shared generators.
    TwoLines(Line3D, Line3D),
    /// One shared grazing generator.
    TangentLine(Line3D),
    /// Separate, nested without contact, or concentric with different radii.
    Empty,
    /// Same axis and radius within tolerance.
    Coincident,
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting a cone with a cylinder.
///
/// The analytic path needs coaxial axes (within tolerance): the tube of
/// radius `r` meets each nappe at axial distance `r/|tan(phi)|` from the
/// apex, two latitude circles (or the apex for a degenerate tube).
/// Anything else is a space quartic and reports
/// [`ConeCylinderIntersection::NotAnalytic`], reserved for the numeric
/// surface-surface path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConeCylinderIntersection {
    /// One latitude circle on each nappe, the first along the cone axis.
    TwoCircles(Circle3D, Circle3D),
    /// Degenerate tube (its axis) meeting the cone at the apex.
    ApexPoint(Point3D),
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting a torus with a plane.
///
/// The analytic path needs the plane through the axis (two meridian
/// circles), perpendicular to it (latitude circles), or through the
/// center and tangent to the tube twice (two Villarceau circles).
/// Anything else is a quartic spiric section and reports
/// [`TorusPlaneIntersection::NotAnalytic`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TorusPlaneIntersection {
    /// Two circles: meridian (axis-containing plane), latitude
    /// (perpendicular plane) or Villarceau (bitangent plane).
    TwoCircles(Circle3D, Circle3D),
    /// Grazing ring (perpendicular plane at tube top/bottom).
    TangentCircle(Circle3D),
    /// Single latitude ring (spindle case).
    Circle(Circle3D),
    /// The plane misses the torus.
    Empty,
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting a torus with a sphere.
///
/// The analytic path needs the sphere center on the torus axis, reducing
/// to a quadratic in the axial coordinate. Anything else is high-degree
/// and reports [`TorusSphereIntersection::NotAnalytic`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TorusSphereIntersection {
    /// Latitude ring.
    Circle(Circle3D),
    /// Two latitude rings.
    TwoCircles(Circle3D, Circle3D),
    /// Grazing ring.
    TangentCircle(Circle3D),
    /// No ring.
    Empty,
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting a torus with a cylinder.
///
/// The analytic path needs coaxial axes. Anything else reports
/// [`TorusCylinderIntersection::NotAnalytic`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TorusCylinderIntersection {
    /// Two latitude rings symmetric about the equatorial plane.
    TwoCircles(Circle3D, Circle3D),
    /// Grazing ring in the equatorial plane.
    TangentCircle(Circle3D),
    /// The tube misses the torus.
    Empty,
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// One latitude ring of a coaxial intersection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatitudeRing {
    /// The ring.
    pub circle: Circle3D,
    /// Whether the surfaces graze along it instead of crossing.
    pub tangent: bool,
}

/// Result of intersecting a torus with a cone.
///
/// The analytic path needs coaxial axes, reducing to a quadratic in the
/// axial coordinate per nappe. Anything else reports
/// [`TorusConeIntersection::NotAnalytic`].
#[derive(Debug, Clone, PartialEq)]
pub enum TorusConeIntersection {
    /// Latitude rings, up to two per nappe, ordered along the torus axis.
    Rings(Vec<LatitudeRing>),
    /// The cone misses the torus.
    Empty,
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting two tori.
///
/// The analytic path needs collinear axes, reducing to quadratics in the
/// axial coordinate. Anything else (up to degree 16) reports
/// [`TorusTorusIntersection::NotAnalytic`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TorusTorusIntersection {
    /// Latitude ring.
    Circle(Circle3D),
    /// Two latitude rings.
    TwoCircles(Circle3D, Circle3D),
    /// Grazing ring.
    TangentCircle(Circle3D),
    /// No rings.
    Empty,
    /// Same center and radii within tolerance.
    Coincident,
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting two cones.
///
/// The analytic path needs coaxial axes: equal latitude-ring radii give
/// up to two axial positions (or apex/coincident degenerates). Anything
/// else is a space quartic and reports
/// [`ConeConeIntersection::NotAnalytic`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConeConeIntersection {
    /// One latitude ring (equal semi-angles, distinct apexes).
    Circle(Circle3D),
    /// Two latitude rings, ordered along the first cone's axis.
    TwoCircles(Circle3D, Circle3D),
    /// The rings collapse to the shared apex.
    ApexPoint(Point3D),
    /// Same apex, axis, and semi-angle within tolerance.
    Coincident,
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting two 2D lines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineLine2DIntersection {
    /// Transversal hit: line parameter on self, point, line parameter other.
    Point(f64, Point2D, f64),
    /// Parallel distinct lines.
    Parallel,
    /// Coincident lines.
    Coincident,
}

/// Result of intersecting a 2D line with a 2D circle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineCircle2DIntersection {
    /// Two hits: `(line parameter, point)` pairs, ordered.
    Points((f64, Point2D), (f64, Point2D)),
    /// Grazing contact: line parameter and point.
    Tangent(f64, Point2D),
    /// No intersection.
    Empty,
}

/// Result of intersecting two 2D circles.
///
/// Parameters are intentionally omitted (no natural primary curve);
/// recover them with [`crate::Circle2D::parameter_of`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircleCircle2DIntersection {
    /// Two meeting points.
    Points(Point2D, Point2D),
    /// Grazing contact point.
    Tangent(Point2D),
    /// Separate, nested, or concentric with different radii.
    Empty,
    /// Same center and radius within tolerance.
    Coincident,
}

/// Result of intersecting two 3D lines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineLine3DIntersection {
    /// Transversal hit: parameter on self, point, parameter on other.
    Point(f64, Point3D, f64),
    /// Parallel distinct lines, with their distance.
    Parallel(f64),
    /// Coincident lines.
    Coincident,
    /// Skew lines: parameters and distance of the closest pair.
    Skew {
        /// Parameter on the first line.
        s: f64,
        /// Parameter on the second line.
        t: f64,
        /// Distance between the closest points.
        distance: f64,
    },
}

/// Result of intersecting a 3D line with a 3D circle.
///
/// The line meets the circle's plane in at most one point (transversal
/// graze or miss); a line lying in the plane reduces to the 2D problem.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineCircle3DIntersection {
    /// Two hits: `(line parameter, point)` pairs, ordered.
    Points((f64, Point3D), (f64, Point3D)),
    /// Grazing contact: line parameter and point.
    Tangent(f64, Point3D),
    /// No intersection.
    Empty,
}

/// Result of intersecting two 3D circles.
///
/// Only coplanar pairs admit a closed form (via 2D images in the shared
/// plane). Distinct parallel planes are [`CircleCircle3DIntersection::Empty`];
/// crossing planes report [`CircleCircle3DIntersection::NotAnalytic`].
/// Parameters are omitted (no natural primary curve); recover them with
/// [`Circle3D::parameter_of`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircleCircle3DIntersection {
    /// Two meeting points.
    Points(Point3D, Point3D),
    /// Grazing contact point.
    Tangent(Point3D),
    /// No intersection.
    Empty,
    /// Same center, radius, and plane within tolerance.
    Coincident,
    /// No closed form: the circle planes cross.
    NotAnalytic,
}

/// One hit of an analytic line-vs-conic solve: parameter on the line,
/// the hit point, and parameter on the conic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineConicHit {
    /// Parameter on the line.
    pub line_param: f64,
    /// The hit point.
    pub point: Point3D,
    /// Parameter on the conic.
    pub conic_param: f64,
}

/// Result of intersecting a 3D line with an ellipse, parabola, or
/// hyperbola.
///
/// A transversal line meets the conic's plane once (graze or miss); a
/// line lying in the plane substitutes into the conic's implicit equation
/// (quadratic in the line parameter).
#[derive(Debug, Clone, PartialEq)]
pub enum LineConic3DIntersection {
    /// Two hits, ordered by line parameter.
    Points(LineConicHit, LineConicHit),
    /// Grazing contact (double or linear root).
    Tangent(LineConicHit),
    /// No intersection.
    Empty,
}

/// Implicit quadratic `A*x^2 + B*y^2 + C*x*y + D*x + E*y + F = 0` of a
/// conic in its own frame coordinates: the substitution target for
/// coplanar line/circle solves.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ImplicitConic {
    a2: f64,
    b2: f64,
    c2: f64,
    d1: f64,
    e1: f64,
    f0: f64,
}

impl ImplicitConic {
    /// `(x/a)^2 + (y/b)^2 = 1`.
    pub(crate) fn ellipse(a: f64, b: f64) -> ImplicitConic {
        ImplicitConic {
            a2: 1.0 / (a * a),
            b2: 1.0 / (b * b),
            c2: 0.0,
            d1: 0.0,
            e1: 0.0,
            f0: -1.0,
        }
    }

    /// `y^2 = 4*f*x`.
    pub(crate) fn parabola(focal: f64) -> ImplicitConic {
        ImplicitConic {
            a2: 0.0,
            b2: 1.0,
            c2: 0.0,
            d1: -4.0 * focal,
            e1: 0.0,
            f0: 0.0,
        }
    }

    /// `(x/a)^2 - (y/b)^2 = 1`.
    pub(crate) fn hyperbola(a: f64, b: f64) -> ImplicitConic {
        ImplicitConic {
            a2: 1.0 / (a * a),
            b2: -1.0 / (b * b),
            c2: 0.0,
            d1: 0.0,
            e1: 0.0,
            f0: -1.0,
        }
    }

    /// Coefficients `(qa, qb, qc)` of `qa*t^2 + qb*t + qc = 0` for the
    /// line `(ox + t*dx, oy + t*dy)` substituted into the implicit
    /// equation.
    pub(crate) fn line_substitute(&self, ox: f64, oy: f64, dx: f64, dy: f64) -> (f64, f64, f64) {
        let qa = self.a2 * dx * dx + self.b2 * dy * dy + self.c2 * dx * dy;
        let qb = 2.0 * self.a2 * ox * dx
            + 2.0 * self.b2 * oy * dy
            + self.c2 * (ox * dy + oy * dx)
            + self.d1 * dx
            + self.e1 * dy;
        let qc = self.a2 * ox * ox
            + self.b2 * oy * oy
            + self.c2 * ox * oy
            + self.d1 * ox
            + self.e1 * oy
            + self.f0;
        (qa, qb, qc)
    }
}

/// Frame placement plus implicit equation of a conic for the coplanar
/// line/circle solves: the section plane, the frame axes and reference
/// point (center or apex) the implicit coordinates are measured in, and
/// the [`ImplicitConic`] itself.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlacedConic {
    /// The conic's plane.
    pub plane: Plane,
    /// The frame x direction.
    pub x: Vector3D,
    /// The frame y direction.
    pub y: Vector3D,
    /// The frame origin (center or apex).
    pub origin: Point3D,
    /// The implicit equation in frame coordinates.
    pub implicit: ImplicitConic,
}

/// A hit of a curve on a surface: the curve parameter, the point, and the
/// root multiplicity (2+ marks grazing tangency).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConicSurfaceHit {
    /// Parameter on the curve, within its bounds: in `[0, 2π)` on a
    /// circle or ellipse, where a hit at the seam reports `0`.
    pub parameter: f64,
    /// The hit point.
    pub point: Point3D,
    /// Root multiplicity (≥ 2 signals tangency).
    pub multiplicity: u32,
}

/// Result of intersecting a 3D circle with a plane, sphere, cylinder, or
/// cone: either the whole circle lies on the surface, isolated hits
/// report, or nothing does.
#[derive(Debug, Clone, PartialEq)]
pub enum CircleSurfaceIntersection {
    /// The whole circle lies on the surface (sample-verified).
    Circle(Circle3D),
    /// Isolated hits with multiplicity, ordered by circle parameter.
    Hits(Vec<ConicSurfaceHit>),
    /// No intersection.
    Empty,
}

/// Result of intersecting an ellipse, parabola, or hyperbola with a
/// plane, sphere, cylinder, or cone: isolated hits, whole-curve
/// coincidence, or nothing.
#[derive(Debug, Clone, PartialEq)]
pub enum ConicSurfaceIntersection {
    /// Isolated hits with multiplicity (≥ 2 marks tangency), ordered by
    /// curve parameter.
    Hits(Vec<ConicSurfaceHit>),
    /// The whole curve lies on the surface: in the plane, or a plane
    /// section of the sphere, cylinder, or cone.
    Coincident,
    /// No intersection.
    Empty,
}

/// Result of intersecting two conics: a circle, ellipse, parabola or
/// hyperbola with another (two circles have
/// [`CircleCircle3DIntersection`]).
///
/// Coplanar conics meet at up to four points: one conic's
/// parametrization substitutes into the other's implicit equation. Conics
/// in crossing planes meet at up to two, on the line where the planes
/// cross. A hyperbola is one branch: hits on its mirror branch are not
/// reported.
///
/// # Examples
///
/// ```
/// use geomcore::{ConicConicIntersection, Ellipse3D, Frame3D, Hyperbola3D, Intersect, Tolerance};
/// let ellipse = Ellipse3D::from_frame(Frame3D::WORLD, 3.0, 2.0).unwrap();
/// let hyperbola = Hyperbola3D::from_frame(Frame3D::WORLD, 1.0, 0.5).unwrap();
/// match ellipse.intersect(&hyperbola, Tolerance::DEFAULT) {
///     // The branch x > 0 meets the ellipse at (2.4739, ±1.1314).
///     ConicConicIntersection::Hits(hits) => assert_eq!(hits.len(), 2),
///     other => panic!("expected two hits, got {other:?}"),
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum ConicConicIntersection {
    /// The hits, ordered by parameter on the first conic; a multiplicity
    /// of 2 or more marks a tangency.
    Hits(Vec<CurveCurveHit>),
    /// The two conics are the same curve.
    Coincident,
    /// The conics do not meet.
    Empty,
}

/// One local minimum of the distance between two 3D curves (see
/// [`curve_curve_extrema`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveCurveExtremum {
    /// Parameter on the first curve.
    pub first_param: f64,
    /// Parameter on the second curve.
    pub second_param: f64,
    /// Distance between the two points.
    pub distance: f64,
}

/// One meeting of two 3D curves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveCurveHit {
    /// Parameter on the first curve, within its bounds (on a circle or
    /// ellipse, a hit at the seam reports the start of the bounds).
    pub first_param: f64,
    /// The meeting point (on the first curve; within tolerance of both).
    pub point: Point3D,
    /// Parameter on the second curve, within its bounds.
    pub second_param: f64,
    /// 1 where the curves cross, 2 or more where they are tangent.
    pub multiplicity: u32,
}

/// Grid points per finite parameter domain in the 2D extrema search.
const SEED_GRID: usize = 32;
/// Samples per finite parameter domain in the 1D (line against curve)
/// extrema search.
const SEED_GRID_1D: usize = 256;
/// Most seeds taken from one sampled grid (or edge) of the extrema
/// search.
const MAX_SEEDS: usize = 64;
/// Unbounded parameters are seeded until the curve has moved this many
/// configuration scales from its reference point.
const SEED_REACH: f64 = 64.0;

/// Parameter domain of one curve in [`curve_curve_extrema`].
#[derive(Debug, Clone, Copy)]
enum ParamDomain {
    /// A full period from `start`: unconstrained, wrapped into
    /// `[start, start + period)` on output.
    Periodic { start: f64, period: f64 },
    /// `[lo, hi]`; either end may be infinite.
    Interval(f64, f64),
}

impl ParamDomain {
    /// The domain of `curve`; a periodic curve bounded to less than a
    /// period is an arc, so it is an interval.
    fn of<C: ParametricCurve3D>(curve: &C) -> ParamDomain {
        let (lo, hi) = curve.bounds();
        match curve.period() {
            Some(period) if lo.is_finite() && hi - lo >= period * (1.0 - 1e-12) => {
                ParamDomain::Periodic { start: lo, period }
            }
            _ => ParamDomain::Interval(lo, hi),
        }
    }

    /// The box constraint of the minimiser (`None`: unconstrained).
    fn bounds(self) -> Option<(f64, f64)> {
        match self {
            ParamDomain::Interval(lo, hi) if lo.is_finite() || hi.is_finite() => Some((lo, hi)),
            _ => None,
        }
    }

    fn is_periodic(self) -> bool {
        matches!(self, ParamDomain::Periodic { .. })
    }

    /// Whether the domain is the whole real line.
    fn is_unbounded(self) -> bool {
        matches!(self, ParamDomain::Interval(lo, hi) if lo.is_infinite() && hi.is_infinite())
    }

    /// Whether the domain has finite extent (a period counts).
    fn is_finite(self) -> bool {
        match self {
            ParamDomain::Periodic { .. } => true,
            ParamDomain::Interval(lo, hi) => lo.is_finite() && hi.is_finite(),
        }
    }

    /// The finite ends of an interval (none for a period).
    fn finite_ends(self) -> impl Iterator<Item = f64> {
        let ends = match self {
            ParamDomain::Periodic { .. } => [f64::NAN; 2],
            ParamDomain::Interval(lo, hi) => [lo, hi],
        };
        ends.into_iter().filter(|u| u.is_finite())
    }

    /// The reference parameter: the start of the domain, else `0` (or
    /// the upper end when that is below `0`).
    fn reference(self) -> f64 {
        match self {
            ParamDomain::Periodic { start, .. } => start,
            ParamDomain::Interval(lo, _) if lo.is_finite() => lo,
            ParamDomain::Interval(_, hi) => hi.min(0.0),
        }
    }

    /// `u` brought into the domain: wrapped into the period, or clamped
    /// and snapped onto an end it is within the minimiser's resolution of
    /// (where a step would be too small to move it).
    fn wrap(self, u: f64) -> f64 {
        let near = |end: f64| end.is_finite() && (u - end).abs() <= 2e-14 * (1.0 + end.abs());
        match self {
            ParamDomain::Periodic { start, period } => start + (u - start).rem_euclid(period),
            ParamDomain::Interval(lo, _) if near(lo) => lo,
            ParamDomain::Interval(_, hi) if near(hi) => hi,
            ParamDomain::Interval(lo, hi) => u.clamp(lo, hi),
        }
    }

    /// A finite domain as the parameter range of a piece-pair search.
    fn range(self) -> curve_surface::Range {
        match self {
            ParamDomain::Periodic { start, period } => curve_surface::Range {
                lo: start,
                hi: start + period,
                periodic: true,
            },
            ParamDomain::Interval(lo, hi) => curve_surface::Range {
                lo,
                hi,
                periodic: false,
            },
        }
    }

    /// The signed step from `u` to `v`, the shorter way round a period.
    fn step(self, u: f64, v: f64) -> f64 {
        match self {
            ParamDomain::Periodic { period, .. } => {
                let d = (v - u).rem_euclid(period);
                if d > 0.5 * period { d - period } else { d }
            }
            ParamDomain::Interval(..) => v - u,
        }
    }

    /// Seed parameters, ascending. A finite domain gets `count` uniform
    /// samples. An unbounded direction gets a geometric progression away
    /// from the reference parameter, starting at a step that moves the
    /// curve `scale / 16` and growing by a quarter per sample until the
    /// curve is [`SEED_REACH`] scales from its reference point: dense
    /// where the curves interact, sparse far out, at any absolute size.
    fn seeds<C: ParametricCurve3D>(self, curve: &C, scale: f64, count: usize) -> Vec<f64> {
        match self {
            ParamDomain::Periodic { start, period } => (0..count)
                .map(|i| start + period * i as f64 / count as f64)
                .collect(),
            // The ends exactly, so a minimum held there is seeded on it.
            ParamDomain::Interval(lo, hi) if self.is_finite() => (0..count)
                .map(|i| match i {
                    0 => lo,
                    _ if i == count - 1 => hi,
                    _ => lo + (hi - lo) * i as f64 / (count - 1) as f64,
                })
                .collect(),
            ParamDomain::Interval(lo, hi) => {
                let u0 = self.reference();
                let (p0, d1) = curve.eval_d1(u0);
                let speed = d1.magnitude();
                let first = scale / (16.0 * speed);
                let first = if first.is_finite() && first > 0.0 {
                    first
                } else {
                    1.0
                };
                let mut seeds: Vec<f64> = self.finite_ends().collect();
                seeds.push(u0);
                for side in [-1.0, 1.0] {
                    let mut step = first;
                    for _ in 0..96 {
                        let u = u0 + side * step;
                        if u < lo || u > hi {
                            break;
                        }
                        seeds.push(u);
                        if curve.eval_point(u).distance(p0) >= SEED_REACH * scale {
                            break;
                        }
                        step *= 1.25;
                    }
                }
                seeds.sort_by(f64::total_cmp);
                seeds.dedup();
                seeds
            }
        }
    }
}

/// A straight curve with constant speed: `origin + (u - u0) * dir`.
#[derive(Debug, Clone, Copy)]
struct Straight {
    u0: f64,
    origin: Point3D,
    dir: Vector3D,
}

impl Straight {
    /// `curve` as a straight curve, if it is one over its whole domain
    /// (lines, straight segments): checked against the affine extension
    /// of its reference point and tangent at a spread of parameters.
    fn of<C: ParametricCurve3D>(curve: &C, dom: ParamDomain) -> Option<Straight> {
        const SPREAD: [f64; 6] = [0.137, 0.382, 0.5, 0.731, 0.919, 1.0];
        const REACH: [f64; 4] = [0.7, 3.1, 17.3, 101.9];
        let probes: Vec<f64> = match dom {
            ParamDomain::Periodic { .. } => return None,
            ParamDomain::Interval(lo, hi) if dom.is_finite() => {
                SPREAD.iter().map(|f| lo + (hi - lo) * f).collect()
            }
            ParamDomain::Interval(lo, _) if lo.is_finite() => {
                REACH.iter().map(|r| lo + r).collect()
            }
            ParamDomain::Interval(_, hi) if hi.is_finite() => {
                REACH.iter().map(|r| hi - r).collect()
            }
            ParamDomain::Interval(..) => REACH.iter().flat_map(|&r| [-r, r]).collect(),
        };
        let u0 = dom.reference();
        let (origin, dir) = curve.eval_d1(u0);
        let line = Straight { u0, origin, dir };
        if line.dir.magnitude() <= 0.0 {
            return None;
        }
        let size = |p: Point3D| (p - Point3D::ORIGIN).magnitude();
        probes
            .into_iter()
            .all(|u| {
                let (p, q) = (curve.eval_point(u), line.point(u));
                p.distance(q) <= 1e-12 * (size(p) + size(line.origin) + p.distance(line.origin))
            })
            .then_some(line)
    }

    fn point(&self, u: f64) -> Point3D {
        self.origin + self.dir * (u - self.u0)
    }

    /// Parameter of the foot of `p` on the (unbounded) line.
    fn foot(&self, p: Point3D) -> f64 {
        self.u0 + (p - self.origin).dot(self.dir) / self.dir.square_magnitude()
    }
}

/// The extremum record of the pair `(s, t)`.
fn curve_pair_extremum<A, B>(a: &A, b: &B, s: f64, t: f64) -> CurveCurveExtremum
where
    A: ParametricCurve3D,
    B: ParametricCurve3D,
{
    CurveCurveExtremum {
        first_param: s,
        second_param: t,
        distance: a.eval_point(s).distance(b.eval_point(t)),
    }
}

/// Generic closest-point enumeration between two 3D curves: every local
/// minimum of the distance `|C1(s) - C2(t)|` over the two parameter
/// domains, ordered by distance.
///
/// Bounded (trimmed) domains are honoured exactly: a minimum on a domain
/// end or a corner of the parameter box is reported, and only true local
/// minima of the distance restricted to the box are. The squared
/// distance is minimised over the box by Newton steps with exact second
/// derivatives (parameters held at a bound by an outward gradient stay
/// there), and each result must pass the box optimality check: every
/// parameter stationary (tangential residual within `tol.confusion`)
/// with a positive semidefinite Hessian, or held at a bound.
///
/// Two straight curves (lines, segments) are solved in closed form (one
/// minimum). A full line against any curve reduces exactly to the minima
/// of that curve's distance to the line, along the curve alone. Otherwise
/// the minimiser starts from the discrete local minima of a parameter
/// grid, and of dense samples along each finite edge of the box. An
/// unbounded parameter is seeded geometrically from its reference
/// parameter, out to many times the configuration's own length scale
/// (the gap between the curves plus their sizes), and the minimiser may
/// leave the seeded range.
///
/// Configurations with a continuous family of equal minima (parallel
/// lines, concentric circles, overlapping curves) report the distance
/// once, with a representative pair: the first curve at its reference
/// parameter (`0` on a line or circle, the start of a bounded domain)
/// with its closest point on the second, when that pair belongs to the
/// family, else a member found by the search.
///
/// Two B-spline curves, or trimmed curves over B-splines (see
/// [`ParametricCurve3D::as_bspline`]), are seeded without a grid: every
/// pair of their Bézier pieces is halved until the derivatives of the
/// squared distance provably keep a sign over it (no minimum inside), or
/// its Hessian is provably indefinite, or provably positive definite, so
/// that it holds at most one minimum, which Newton's method reaches; the
/// box's edges are searched the same way along one curve. A valley of
/// equal distance or a tangency never isolates: it exhausts the search's
/// budget, and the grid above is used instead.
///
/// # Examples
///
/// ```
/// use geomcore::{Line3D, Point3D, Tolerance, Vector3D, curve_curve_extrema};
/// let l1 = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
/// let l2 = Line3D::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Y).unwrap();
/// let ext = curve_curve_extrema(&l1, &l2, Tolerance::DEFAULT);
/// assert_eq!(ext.len(), 1);
/// assert_eq!(ext[0].distance, 1.0);
///
/// // Parallel lines: one representative pair at the distance.
/// let l3 = Line3D::new(Point3D::new(0.0, 2.0, 0.0), Vector3D::X).unwrap();
/// let ext = curve_curve_extrema(&l1, &l3, Tolerance::DEFAULT);
/// assert_eq!(ext.len(), 1);
/// assert_eq!((ext[0].first_param, ext[0].distance), (0.0, 2.0));
/// ```
pub fn curve_curve_extrema<A, B>(a: &A, b: &B, tol: Tolerance) -> Vec<CurveCurveExtremum>
where
    A: ParametricCurve3D,
    B: ParametricCurve3D,
{
    curve_curve_minima(a, b, tol, true)
}

/// [`curve_curve_extrema`], through the piece-pair search on B-splines if
/// `pieces`, else always through the grid.
fn curve_curve_minima<A, B>(a: &A, b: &B, tol: Tolerance, pieces: bool) -> Vec<CurveCurveExtremum>
where
    A: ParametricCurve3D,
    B: ParametricCurve3D,
{
    let (da, db) = (ParamDomain::of(a), ParamDomain::of(b));
    let mut out = match (Straight::of(a, da), Straight::of(b, db)) {
        (Some(la), Some(lb)) => vec![straight_pair_minimum(a, la, da, b, lb, db, tol)],
        (Some(la), None) if da.is_unbounded() => line_curve_minima(a, la, b, db, tol),
        (None, Some(lb)) if db.is_unbounded() => line_curve_minima(b, lb, a, da, tol)
            .into_iter()
            .map(|e| CurveCurveExtremum {
                first_param: e.second_param,
                second_param: e.first_param,
                distance: e.distance,
            })
            .collect(),
        _ => match (a.as_bspline(), b.as_bspline()) {
            (Some(sa), Some(sb)) if pieces => {
                curve_curve::bspline_minima(sa, da.range(), sb, db.range(), tol)
                    .map(|seeds| refine_minima(a, da, b, db, seeds, tol))
            }
            _ => None,
        }
        .unwrap_or_else(|| box_minima(a, da, b, db, tol)),
    };
    out.sort_by(|x, y| x.distance.total_cmp(&y.distance));
    out
}

/// The minimum distance between two straight curves. The squared
/// distance is a convex quadratic, so its minimum over the parameter box
/// is the unconstrained (skew) solution when that lies inside, and
/// otherwise lies on an edge of the box, where it is the clamped foot of
/// the edge's end point. A parallel pair has a valley of minima; its
/// representative is the first edge candidate (the first curve's start)
/// or, for two full lines, the first curve at `0` and its foot.
fn straight_pair_minimum<A, B>(
    a: &A,
    la: Straight,
    da: ParamDomain,
    b: &B,
    lb: Straight,
    db: ParamDomain,
    tol: Tolerance,
) -> CurveCurveExtremum
where
    A: ParametricCurve3D,
    B: ParametricCurve3D,
{
    let (d1, d2) = (la.dir, lb.dir);
    let w = la.origin - lb.origin;
    let (aa, ab, bb) = (d1.dot(d1), d1.dot(d2), d2.dot(d2));
    let denom = aa * bb - ab * ab;
    if denom > tol.angular * aa * bb {
        let (e1, e2) = (d1.dot(w), d2.dot(w));
        let s = la.u0 + (ab * e2 - bb * e1) / denom;
        let t = lb.u0 + (aa * e2 - ab * e1) / denom;
        if da.wrap(s) == s && db.wrap(t) == t {
            return curve_pair_extremum(a, b, s, t);
        }
    }
    let edges = da
        .finite_ends()
        .map(|s| (s, db.wrap(lb.foot(la.point(s)))))
        .chain(db.finite_ends().map(|t| (da.wrap(la.foot(lb.point(t))), t)));
    let mut best: Option<CurveCurveExtremum> = None;
    for (s, t) in edges {
        let e = curve_pair_extremum(a, b, s, t);
        // Strict: on a parallel valley the first edge stays.
        if best.is_none_or(|m| e.distance < m.distance) {
            best = Some(e);
        }
    }
    best.unwrap_or_else(|| {
        let s = da.reference();
        curve_pair_extremum(a, b, s, lb.foot(la.point(s)))
    })
}

/// Local minima of the distance between a full line and a curve, as
/// `(line parameter, curve parameter)` extrema. The line's closest point
/// to each curve point is its foot, so the minima are exactly those of
/// the curve's distance to the line along the curve parameter alone:
/// seeded at the discrete minima of dense samples and refined by bounded
/// Newton with exact second derivatives.
fn line_curve_minima<L, C>(
    line_curve: &L,
    line: Straight,
    curve: &C,
    dom: ParamDomain,
    tol: Tolerance,
) -> Vec<CurveCurveExtremum>
where
    L: ParametricCurve3D,
    C: ParametricCurve3D,
{
    let unit = line.dir * (1.0 / line.dir.magnitude());
    let perp = |v: Vector3D| v - unit * v.dot(unit);
    // Half the squared distance to the line, its first two derivatives
    // along the curve, and the curve's speed.
    let local = |t: f64| {
        let (p, c1, c2) = curve.eval_d2(t);
        let r = perp(p - line.origin);
        let c1p = perp(c1);
        (
            (0.5 * r.dot(r), r.dot(c1), c1p.dot(c1p) + r.dot(c2)),
            c1.magnitude(),
        )
    };
    let extremum =
        |t: f64| curve_pair_extremum(line_curve, curve, line.foot(curve.eval_point(t)), t);
    let gap = perp(curve.eval_point(dom.reference()) - line.origin).magnitude();
    let scale = configuration_scale(gap, 0.0, curve_size(curve, dom));
    let seeds = dom.seeds(curve, scale, SEED_GRID_1D);
    let values: Vec<f64> = seeds.iter().map(|&t| local(t).0.0).collect();
    let bounds = dom.bounds();
    let mut found = Vec::new();
    for (k, _) in grid_minima(&values, [seeds.len(), 1], [dom.is_periodic(), false]) {
        let t = dom.wrap(newton_min_1d(|t| local(t).0, seeds[k], bounds, 1e-14, 100));
        let ((_, g, h), speed) = local(t);
        let optimal = is_box_minimum(
            [t, 0.0],
            [g, 0.0],
            [[h, 0.0], [0.0, 0.0]],
            [speed, 0.0],
            [bounds, None],
            tol,
        );
        if t.is_finite() && optimal {
            found.push(extremum(t));
        }
    }
    let distance = |t: f64| {
        line_curve
            .eval_point(line.foot(curve.eval_point(t)))
            .distance(curve.eval_point(t))
    };
    let same_valley = |p: &CurveCurveExtremum, q: &CurveCurveExtremum| {
        let dt = dom.step(p.second_param, q.second_param);
        flat_between(
            p.distance,
            q.distance,
            |f| distance(p.second_param + f * dt),
            tol,
        )
    };
    merge_valleys(found, same_valley, |_| extremum(dom.reference()), tol)
}

/// Local minima of the distance between two curves over their parameter
/// box: Newton minimisation of the squared distance from the discrete
/// local minima of a parameter grid and of dense samples along each
/// finite edge, kept when box-optimal.
fn box_minima<A, B>(
    a: &A,
    da: ParamDomain,
    b: &B,
    db: ParamDomain,
    tol: Tolerance,
) -> Vec<CurveCurveExtremum>
where
    A: ParametricCurve3D,
    B: ParametricCurve3D,
{
    let gap = a
        .eval_point(da.reference())
        .distance(b.eval_point(db.reference()));
    let scale = configuration_scale(gap, curve_size(a, da), curve_size(b, db));
    let (ss, ts) = (da.seeds(a, scale, SEED_GRID), db.seeds(b, scale, SEED_GRID));
    let pb: Vec<Point3D> = ts.iter().map(|&t| b.eval_point(t)).collect();
    let values: Vec<f64> = ss
        .iter()
        .flat_map(|&s| {
            let p = a.eval_point(s);
            pb.iter().map(move |q| p.distance(*q))
        })
        .collect();
    let shape = [ss.len(), ts.len()];
    let mut seeds: Vec<[f64; 2]> =
        grid_minima(&values, shape, [da.is_periodic(), db.is_periodic()])
            .into_iter()
            .map(|(i, j)| [ss[i], ts[j]])
            .collect();
    // A minimum on an edge of the box can have a basin narrower than the
    // grid: seed each finite edge at the discrete minima of dense samples.
    let edge = |p: Point3D, curve: &dyn Fn(f64) -> Point3D, us: &[f64], dom: ParamDomain| {
        let values: Vec<f64> = us.iter().map(|&u| p.distance(curve(u))).collect();
        grid_minima(&values, [us.len(), 1], [dom.is_periodic(), false])
            .into_iter()
            .map(|(k, _)| us[k])
            .collect::<Vec<f64>>()
    };
    let (dense_s, dense_t) = (
        da.seeds(a, scale, SEED_GRID_1D),
        db.seeds(b, scale, SEED_GRID_1D),
    );
    for s in da.finite_ends() {
        let on_b = edge(a.eval_point(s), &|t| b.eval_point(t), &dense_t, db);
        seeds.extend(on_b.into_iter().map(|t| [s, t]));
    }
    for t in db.finite_ends() {
        let on_a = edge(b.eval_point(t), &|s| a.eval_point(s), &dense_s, da);
        seeds.extend(on_a.into_iter().map(|s| [s, t]));
    }
    refine_minima(a, da, b, db, seeds, tol)
}

/// The local minima reached by Newton minimisation of the squared
/// distance over the box from `seeds`, kept when box-optimal, one per
/// valley of equal distance.
fn refine_minima<A, B>(
    a: &A,
    da: ParamDomain,
    b: &B,
    db: ParamDomain,
    seeds: Vec<[f64; 2]>,
    tol: Tolerance,
) -> Vec<CurveCurveExtremum>
where
    A: ParametricCurve3D,
    B: ParametricCurve3D,
{
    // Half the squared distance with its gradient and Hessian, and the
    // two curves' speeds.
    let local = |s: f64, t: f64| {
        let (pa, a1, a2) = a.eval_d2(s);
        let (pb, b1, b2) = b.eval_d2(t);
        let r = pa - pb;
        let cross = -a1.dot(b1);
        (
            (
                0.5 * r.dot(r),
                [r.dot(a1), -r.dot(b1)],
                [
                    [a1.dot(a1) + r.dot(a2), cross],
                    [cross, b1.dot(b1) - r.dot(b2)],
                ],
            ),
            [a1.magnitude(), b1.magnitude()],
        )
    };
    let bounds = [da.bounds(), db.bounds()];
    let mut found = Vec::new();
    for seed in seeds {
        let [s, t] = newton_min_2d(|s, t| local(s, t).0, seed, bounds, 1e-14, 100);
        let (s, t) = (da.wrap(s), db.wrap(t));
        let ((_, g, h), speed) = local(s, t);
        if s.is_finite() && t.is_finite() && is_box_minimum([s, t], g, h, speed, bounds, tol) {
            found.push(curve_pair_extremum(a, b, s, t));
        }
    }
    let same_valley = |p: &CurveCurveExtremum, q: &CurveCurveExtremum| {
        let ds = da.step(p.first_param, q.first_param);
        let dt = db.step(p.second_param, q.second_param);
        flat_between(
            p.distance,
            q.distance,
            |f| {
                a.eval_point(p.first_param + f * ds)
                    .distance(b.eval_point(p.second_param + f * dt))
            },
            tol,
        )
    };
    // A valley's representative: the first curve at its reference
    // parameter and its closest point on the second.
    let representative = |e: &CurveCurveExtremum| {
        let s = da.reference();
        let p = a.eval_point(s);
        let objective = |t: f64| {
            let (q, b1, b2) = b.eval_d2(t);
            let r = p - q;
            (0.5 * r.dot(r), -r.dot(b1), b1.dot(b1) - r.dot(b2))
        };
        let t = newton_min_1d(objective, e.second_param, db.bounds(), 1e-14, 100);
        curve_pair_extremum(a, b, s, db.wrap(t))
    };
    merge_valleys(found, same_valley, representative, tol)
}

/// Length scale of a configuration: the gap between the curves'
/// reference points plus both curves' sizes (`1` if that degenerates).
fn configuration_scale(gap: f64, size_a: f64, size_b: f64) -> f64 {
    let scale = gap + size_a + size_b;
    if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    }
}

/// Size of a curve for [`configuration_scale`]: the reach of a finite
/// domain's grid points from its reference point, or for an unbounded
/// domain the radius of curvature at the reference (`0` when straight).
fn curve_size<C: ParametricCurve3D>(curve: &C, dom: ParamDomain) -> f64 {
    let u0 = dom.reference();
    if dom.is_finite() {
        let p0 = curve.eval_point(u0);
        dom.seeds(curve, 1.0, SEED_GRID)
            .into_iter()
            .map(|u| curve.eval_point(u).distance(p0))
            .fold(0.0, f64::max)
    } else {
        let (_, d1, d2) = curve.eval_d2(u0);
        let bend = d1.cross(d2).magnitude();
        if bend > 0.0 {
            d1.magnitude().powi(3) / bend
        } else {
            0.0
        }
    }
}

/// Nodes of a row-major `shape[0] x shape[1]` grid of samples no higher
/// than any of their 8 neighbours (wrapping across a periodic
/// direction), lowest first, at most [`MAX_SEEDS`].
fn grid_minima(values: &[f64], shape: [usize; 2], wrap: [bool; 2]) -> Vec<(usize, usize)> {
    let [n, m] = shape;
    let neighbour = |i: usize, di: isize, len: usize, wrap: bool| {
        let k = i as isize + di;
        if (0..len as isize).contains(&k) {
            Some(k as usize)
        } else if wrap && len > 1 {
            Some(k.rem_euclid(len as isize) as usize)
        } else {
            None
        }
    };
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..m {
            let v = values[i * m + j];
            let lowest = v.is_finite()
                && (-1..=1).all(|di| {
                    (-1..=1).all(|dj| {
                        match (neighbour(i, di, n, wrap[0]), neighbour(j, dj, m, wrap[1])) {
                            (Some(x), Some(y)) => {
                                values[x * m + y] >= v || values[x * m + y].is_nan()
                            }
                            _ => true,
                        }
                    })
                });
            if lowest {
                out.push((i, j));
            }
        }
    }
    out.sort_by(|p, q| values[p.0 * m + p.1].total_cmp(&values[q.0 * m + q.1]));
    out.truncate(MAX_SEEDS);
    out
}

/// Box optimality of a minimiser's result `x` for half the squared
/// distance: every parameter is either stationary (tangential residual
/// `|g_i| / speed_i` within `tol.confusion`) or held at a bound by an
/// outward gradient, and the Hessian over the stationary parameters is
/// positive semidefinite (rejecting saddles a seed happened to start on).
fn is_box_minimum(
    x: [f64; 2],
    g: [f64; 2],
    h: [[f64; 2]; 2],
    speed: [f64; 2],
    bounds: [Option<(f64, f64)>; 2],
    tol: Tolerance,
) -> bool {
    let mut stationary = [false; 2];
    for i in 0..2 {
        let held = match bounds[i] {
            Some((lo, hi)) => (x[i] <= lo && g[i] > 0.0) || (x[i] >= hi && g[i] < 0.0),
            None => false,
        };
        stationary[i] = g[i].abs() <= tol.confusion * speed[i];
        if !stationary[i] && !held {
            return false;
        }
    }
    let slack = 1e-6 * (h[0][0].abs() + h[1][1].abs() + h[0][1].abs() + h[1][0].abs());
    match stationary {
        [true, true] => {
            let (p, q, r) = (h[0][0], 0.5 * (h[0][1] + h[1][0]), h[1][1]);
            0.5 * (p + r) - (0.25 * (p - r) * (p - r) + q * q).sqrt() >= -slack
        }
        [true, false] => h[0][0] >= -slack,
        [false, true] => h[1][1] >= -slack,
        [false, false] => true,
    }
}

/// Whether two minima at distances `d0` and `d1` lie on one valley: the
/// distances agree and so does the distance along the straight parameter
/// path between them (`along(f)`, `f` in `[0, 1]`), within
/// `tol.confusion`. Copies of one minimum pass trivially.
fn flat_between(d0: f64, d1: f64, along: impl Fn(f64) -> f64, tol: Tolerance) -> bool {
    (d0 - d1).abs() <= tol.confusion
        && (1..8).all(|k| (along(k as f64 / 8.0) - d0).abs() <= tol.confusion)
}

/// Collapses minima on one valley of equal distance (and copies of one
/// minimum) into a single entry, lowest distance first. A genuine
/// valley, met by seeds at distinct parameters, is reported at its
/// `representative` when that pair belongs to it.
fn merge_valleys(
    mut found: Vec<CurveCurveExtremum>,
    same_valley: impl Fn(&CurveCurveExtremum, &CurveCurveExtremum) -> bool,
    representative: impl Fn(&CurveCurveExtremum) -> CurveCurveExtremum,
    tol: Tolerance,
) -> Vec<CurveCurveExtremum> {
    let apart = |u: f64, v: f64| (u - v).abs() > tol.parametric * (1.0 + u.abs());
    found.sort_by(|x, y| x.distance.total_cmp(&y.distance));
    let mut kept: Vec<(CurveCurveExtremum, bool)> = Vec::new();
    for e in found {
        match kept.iter_mut().find(|(k, _)| same_valley(k, &e)) {
            Some((k, valley)) => {
                *valley |=
                    apart(k.first_param, e.first_param) || apart(k.second_param, e.second_param);
            }
            None => kept.push((e, false)),
        }
    }
    kept.into_iter()
        .map(|(e, valley)| {
            if valley {
                let r = representative(&e);
                if same_valley(&e, &r) {
                    return r;
                }
            }
            e
        })
        .collect()
}

/// Generic 3D curve-curve intersection: [`curve_curve_extrema`] filtered
/// to meetings within `tol.confusion`, ordered by first-curve parameter.
/// A hit where the tangents are parallel within `tol.angular` has
/// multiplicity 2.
/// Overlapping curves report a single representative point; use the
/// analytic methods to detect coincidence.
///
/// Two B-spline curves, or trimmed curves over B-splines (see
/// [`ParametricCurve3D::as_bspline`]), are searched without extrema: the
/// pairs of their Bézier pieces whose control points' boxes overlap are
/// halved until each provably holds at most one hit, which Gauss-Newton
/// steps from the pair's centre then reach. A pair that never separates
/// (a tangency, an overlap) exhausts the search's budget, and the extrema
/// above are used instead.
///
/// # Examples
///
/// ```
/// use geomcore::{Circle3D, Line3D, Point3D, Tolerance, Vector3D, intersect_curve_curve};
/// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
/// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
/// let hits = intersect_curve_curve(&line, &circle, Tolerance::DEFAULT);
/// assert_eq!(hits.len(), 2);
/// ```
pub fn intersect_curve_curve<A, B>(a: &A, b: &B, tol: Tolerance) -> Vec<CurveCurveHit>
where
    A: ParametricCurve3D,
    B: ParametricCurve3D,
{
    if let (Some(sa), Some(sb)) = (a.as_bspline(), b.as_bspline()) {
        let (ra, rb) = (ParamDomain::of(a).range(), ParamDomain::of(b).range());
        if let Some(meetings) = curve_curve::bspline_curve_curve(sa, ra, sb, rb, tol) {
            return hits_at_minima(a, b, meetings, tol);
        }
    }
    hits_at_minima(a, b, curve_curve_extrema(a, b, tol), tol)
}

/// [`intersect_curve_curve`] through the grid-seeded extrema alone.
#[cfg(test)]
fn sampled_curve_curve<A, B>(a: &A, b: &B, tol: Tolerance) -> Vec<CurveCurveHit>
where
    A: ParametricCurve3D,
    B: ParametricCurve3D,
{
    hits_at_minima(a, b, curve_curve_minima(a, b, tol, false), tol)
}

/// The hits among the distance minima `extrema` (closest first): those
/// within `tol.confusion`, merged within `sqrt(tol.confusion)`, ordered
/// by first-curve parameter.
fn hits_at_minima<A, B>(
    a: &A,
    b: &B,
    extrema: Vec<CurveCurveExtremum>,
    tol: Tolerance,
) -> Vec<CurveCurveHit>
where
    A: ParametricCurve3D,
    B: ParametricCurve3D,
{
    let hits: Vec<CurveCurveHit> = extrema
        .into_iter()
        .filter(|e| e.distance <= tol.confusion)
        .map(|e| {
            let ta = a.eval_derivative(e.first_param, 1);
            let tb = b.eval_derivative(e.second_param, 1);
            let sin = ta.cross(tb).magnitude() / (ta.magnitude() * tb.magnitude());
            CurveCurveHit {
                first_param: e.first_param,
                point: a.eval_point(e.first_param),
                second_param: e.second_param,
                multiplicity: if sin <= tol.angular { 2 } else { 1 },
            }
        })
        .collect();
    // Contact-level dedupe (see intersect_curve_surface).
    let merge_gap = tol.confusion.sqrt();
    let mut merged: Vec<CurveCurveHit> = Vec::new();
    for h in hits {
        if merged
            .iter()
            .all(|m: &CurveCurveHit| m.point.distance(h.point) > merge_gap)
        {
            merged.push(h);
        }
    }
    merged.sort_by(|x, y| x.first_param.partial_cmp(&y.first_param).unwrap());
    merged
}

/// One transversal meeting of a curve and a surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveSurfaceHit {
    /// Parameter on the curve.
    pub curve_param: f64,
    /// Parameters on the surface.
    pub surface_params: (f64, f64),
    /// The meeting point (on the curve; within tolerance of the surface).
    pub point: Point3D,
}

/// Generic curve-surface intersection for any parametric curve and surface.
///
/// Curve samples projected onto the surface seed Newton iteration on
/// `C(t) - S(u, v) = 0` (first derivatives only); converged roots with a
/// residual within `tol.confusion` report, ordered by curve parameter.
/// Where Newton fails from a sample that is a local minimum of the
/// distance to the surface, the distance from `C(t)` to the surface is
/// minimized over `t` alone between the neighbouring samples. That finds
/// hits at singular points of the surface parametrization (a sphere
/// pole, a cone apex, a collapsed B-spline edge), where the Newton system
/// is singular; their `surface_params` are the projection's.
/// Unbounded curves seed `[-10, 10]`. Overlapping pairs yield dense hit
/// sets along the overlap — use the analytic `intersect_*` methods when
/// classification (tangent, coincident) matters.
///
/// A B-spline curve, or a trimmed curve over one (see
/// [`ParametricCurve3D::as_bspline`]), against a B-spline surface is
/// searched without sampling: the curve's knot spans over its range and
/// the surface's knot-span patches are Bézier pieces, and the pairs of
/// pieces whose control points' boxes overlap are halved until each
/// provably holds at most one hit, which Newton iteration from the pair's
/// centre then reaches. Parameters are reported within the ranges (a
/// periodic one within its period). A pair that never separates (a
/// tangency, an overlap, a hit at a singular point) exhausts the search's
/// budget, and the samples above are used instead.
///
/// # Examples
///
/// ```
/// use geomcore::{Line3D, Point3D, Sphere, Tolerance, Vector3D, intersect_curve_surface};
/// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
/// let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
/// let hits = intersect_curve_surface(&line, &sphere, Tolerance::DEFAULT);
/// assert_eq!(hits.len(), 2);
/// assert!((hits[0].curve_param + 2.0).abs() < 1e-9);
/// assert!((hits[1].curve_param - 2.0).abs() < 1e-9);
/// ```
pub fn intersect_curve_surface<C, S>(curve: &C, surface: &S, tol: Tolerance) -> Vec<CurveSurfaceHit>
where
    C: ParametricCurve3D,
    S: ParametricSurface,
{
    if let (Some(bspline), Some(patches)) = (curve.as_bspline(), surface.as_bspline()) {
        let periodic = curve.period().is_some();
        let hits =
            curve_surface::bspline_curve_surface(bspline, curve.bounds(), periodic, patches, tol);
        if let Some(hits) = hits {
            return hits;
        }
    }
    sampled_curve_surface(curve, surface, tol)
}

/// The sampled search of [`intersect_curve_surface`].
fn sampled_curve_surface<C, S>(curve: &C, surface: &S, tol: Tolerance) -> Vec<CurveSurfaceHit>
where
    C: ParametricCurve3D,
    S: ParametricSurface,
{
    let (t0, t1) = curve.bounds();
    // Unbounded curves seed a documented default window.
    const WINDOW: f64 = 10.0;
    let (lo, hi) = (
        if t0.is_finite() { t0 } else { -WINDOW },
        if t1.is_finite() { t1 } else { WINDOW },
    );
    const SEEDS: usize = 128;
    const KEEP: usize = 24;
    let step = (hi - lo) / (SEEDS - 1) as f64;
    let samples: Vec<(f64, f64)> = (0..SEEDS)
        .map(|i| {
            let t = lo + step * i as f64;
            (t, surface.project_point(curve.eval_point(t), tol).distance)
        })
        .collect();
    let local_min = |i: usize| {
        (i == 0 || samples[i].1 <= samples[i - 1].1)
            && (i + 1 == SEEDS || samples[i].1 <= samples[i + 1].1)
    };
    let mut order: Vec<usize> = (0..SEEDS).collect();
    order.sort_by(|&a, &b| samples[a].1.total_cmp(&samples[b].1));
    let mut hits: Vec<CurveSurfaceHit> = Vec::new();
    // Contact-level dedupe (sqrt scale): near-tangent seeds reach distinct
    // points of one grazing contact; transversal hits are far apart by
    // comparison. Extrema (stationary points) keep tight dedupe elsewhere
    // — distinct minima are real data.
    let merge_gap = tol.confusion.sqrt();
    let push = |hits: &mut Vec<CurveSurfaceHit>, hit: CurveSurfaceHit| {
        if !hits
            .iter()
            .any(|h| h.point.distance(hit.point) <= merge_gap)
        {
            hits.push(hit);
        }
    };
    for i in order.into_iter().take(KEEP) {
        let t = samples[i].0;
        let proj = surface.project_point(curve.eval_point(t), tol);
        let f = |x: [f64; 3]| {
            let d = curve.eval_point(x[0]) - surface.eval_point(x[1], x[2]);
            [d.x, d.y, d.z]
        };
        let jac = |x: [f64; 3]| {
            let (_, ct) = curve.eval_d1(x[0]);
            let (_, [su, sv]) = surface.eval_d1(x[1], x[2]);
            [
                [ct.x, -su.x, -sv.x],
                [ct.y, -su.y, -sv.y],
                [ct.z, -su.z, -sv.z],
            ]
        };
        if let Some([rt, ru, rv]) = newton_3d(f, jac, [t, proj.u, proj.v], tol.confusion, 50) {
            let p = curve.eval_point(rt);
            if p.distance(surface.eval_point(ru, rv)) <= tol.confusion {
                push(
                    &mut hits,
                    CurveSurfaceHit {
                        curve_param: rt,
                        surface_params: (ru, rv),
                        point: p,
                    },
                );
                continue;
            }
        }
        if local_min(i) {
            // The distance to the projected point, not the projection's
            // own, which snaps to zero within `tol.confusion` and would
            // leave the minimum anywhere in that band.
            let distance = |t: f64| {
                let p = curve.eval_point(t);
                let proj = surface.project_point(p, tol);
                p.distance(surface.eval_point(proj.u, proj.v))
            };
            let (a, b) = ((t - step).max(lo), (t + step).min(hi));
            let rt = brent_minimum(distance, a, b, 1e-13);
            let p = curve.eval_point(rt);
            let proj = surface.project_point(p, tol);
            if proj.distance <= tol.confusion {
                push(
                    &mut hits,
                    CurveSurfaceHit {
                        curve_param: rt,
                        surface_params: (proj.u, proj.v),
                        point: p,
                    },
                );
            }
        }
    }
    hits.sort_by(|a, b| a.curve_param.partial_cmp(&b.curve_param).unwrap());
    hits
}

/// Marching surface-surface intersection for any two parametric surfaces.
///
/// Seeds come from both parameter grids: the samples where the distance
/// to the other surface is a local minimum are converged onto the
/// intersection by Newton's method on `S1(u1, v1) = S2(u2, v2)`, so curves
/// running between grid lines are found too. Each seed traces both ways
/// along the `N1 x N2` tangent; every step is corrected by Newton's method
/// on the same system plus the plane through the predicted point normal to
/// the tangent, and the step length follows the curvature (a chord
/// deviation of about `1000 * confusion`, at most 0.1). A trace ends when
/// it closes a loop, reaches a parameter bound, or stalls. Traces fit
/// [`BSplineCurve3D`] interpolants (degree 3, closed loops by repeated end
/// points).
///
/// The fits are APPROXIMATIONS (interpolation error, typically ~1e-4 at
/// default tolerance — assert residuals, not containment): exact
/// classification stays with the analytic `intersect_*` methods. Likewise
/// no tangency classification (grazing pairs may trace partially or not
/// at all), no marching through singular points (poles, apexes), and a
/// fixed seed grid (a loop that leaves no distance minimum on either grid
/// is missed) — v1 heuristics, documented so callers can judge fit.
///
/// # Examples
///
/// ```
/// use geomcore::{Plane, Point3D, Tolerance, Vector3D, marching_intersection};
/// let xy = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
/// let zy = Plane::new(Point3D::ORIGIN, Vector3D::X).unwrap();
/// let curves = marching_intersection(&xy, &zy, Tolerance::DEFAULT);
/// assert_eq!(curves.len(), 1);
/// ```
pub fn marching_intersection<A, B>(a: &A, b: &B, tol: Tolerance) -> Vec<BSplineCurve3D>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let mut seeds = grid_seeds(a, b, tol);
    seeds.extend(
        grid_seeds(b, a, tol)
            .into_iter()
            .map(|x| [x[2], x[3], x[0], x[1]]),
    );
    let mut curves: Vec<BSplineCurve3D> = Vec::new();
    // Seeds on an already traced curve are suppressed by projection onto
    // its fit (the fit error exceeds confusion, so tight containment
    // cannot decide duplication). Near-tangent distinct curves (closer
    // than this) are out of scope.
    let suppress = 1000.0 * tol.confusion;
    for seed in seeds {
        let p = a.eval_point(seed[0], seed[1]);
        if curves
            .iter()
            .any(|c| c.project_point(p, tol).distance <= suppress)
        {
            continue;
        }
        if let Some(points) = trace(a, b, seed, tol) {
            let degree = 3.min(points.len() - 1);
            if let Ok(fit) =
                BSplineCurve3D::interpolate(&points, degree, InterpParametrization::Centripetal)
            {
                curves.push(fit);
            }
        }
    }
    curves
}

/// Parameters `[s_u, s_v, other_u, other_v]` of intersection points
/// converged from the samples of a grid on `s` where the distance to
/// `other` is a local minimum. Unbounded directions sample a documented
/// default window.
fn grid_seeds<S, T>(s: &S, other: &T, tol: Tolerance) -> Vec<[f64; 4]>
where
    S: ParametricSurface,
    T: ParametricSurface,
{
    const N: usize = 24;
    let (u0, u1) = seed_window(s.u_bounds());
    let (v0, v1) = seed_window(s.v_bounds());
    let mut samples: Vec<([f64; 4], f64)> = Vec::with_capacity((N + 1) * (N + 1));
    for j in 0..=N {
        for i in 0..=N {
            let (u, v) = (
                u0 + (u1 - u0) * i as f64 / N as f64,
                v0 + (v1 - v0) * j as f64 / N as f64,
            );
            let q = other.project_point(s.eval_point(u, v), tol);
            samples.push(([u, v, q.u, q.v], q.distance));
        }
    }
    let distance = |i: usize, j: usize| samples[j * (N + 1) + i].1;
    let mut seeds = Vec::new();
    for j in 0..=N {
        for i in 0..=N {
            let d = distance(i, j);
            let neighbours = [
                (i > 0).then(|| distance(i - 1, j)),
                (i < N).then(|| distance(i + 1, j)),
                (j > 0).then(|| distance(i, j - 1)),
                (j < N).then(|| distance(i, j + 1)),
            ];
            if neighbours.into_iter().flatten().all(|e| d <= e)
                && let Some(x) = meet(s, other, samples[j * (N + 1) + i].0, None, tol)
            {
                seeds.push(x);
            }
        }
    }
    seeds
}

/// Trace the intersection curve through `seed` both ways. Returns its
/// points in order (a closed loop ends on its first point), or `None`
/// when fewer than four were traced.
fn trace<A, B>(a: &A, b: &B, seed: [f64; 4], tol: Tolerance) -> Option<Vec<Point3D>>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    // Chord deviation allowed per step, and the longest step.
    let chord_deviation = 1000.0 * tol.confusion;
    const H_MAX: f64 = 0.1;
    // Turn always allowed in one step. Beyond it, a step may turn by at
    // most `CURVATURE_GROWTH` times the previous step's curvature times
    // its chord: on a smooth curve turn/chord tends to the curvature as
    // the step is halved, so the step is eventually accepted, while at a
    // crossing of two branches (where the surfaces touch) the turn onto
    // the other branch stays finite, and the trace ends there.
    const FREE_TURN: f64 = 0.05;
    const CURVATURE_GROWTH: f64 = 4.0;
    let start = a.eval_point(seed[0], seed[1]);
    let t0 = tangent(a, b, seed, tol)?;
    let mut halves: [Vec<Point3D>; 2] = [Vec::new(), Vec::new()];
    let mut closed = false;
    for (half, dir) in halves.iter_mut().zip([1.0, -1.0]) {
        if closed {
            break;
        }
        let (mut x, mut p, mut t) = (seed, start, t0 * dir);
        let mut h = (8.0 * chord_deviation).sqrt().min(H_MAX);
        // Curvature of the last accepted step (none before the first).
        let mut last_curvature = 0.0;
        for _ in 0..2000 {
            // Predict along the tangent, correct onto both surfaces in the
            // plane normal to it; halve the step until the result stays on
            // this branch with a small chord deviation.
            let mut step = None;
            while h >= tol.confusion {
                let target = p + t * h;
                if let Some(y) = meet(a, b, predict(a, b, x, t * h), Some((target, t)), tol)
                    && let Some(ty) = tangent(a, b, y, tol).map(|ty| ty * dir)
                {
                    let q = a.eval_point(y[0], y[1]);
                    let chord = q.distance(p);
                    let turn = t.cross(ty).magnitude().atan2(t.dot(ty));
                    let smooth = turn <= FREE_TURN.max(CURVATURE_GROWTH * last_curvature * chord);
                    if ty.dot(t) > 0.0
                        && chord <= 2.0 * h
                        && chord * turn <= 32.0 * chord_deviation
                        && smooth
                    {
                        step = Some((y, q, ty, turn / chord.max(f64::MIN_POSITIVE)));
                        break;
                    }
                }
                h *= 0.5;
            }
            let Some((y, q, ty, curvature)) = step else {
                break;
            };
            // Back at the seed: close the loop on it.
            let along = (start - p).dot(q - p) / (q - p).dot(q - p).max(f64::MIN_POSITIVE);
            if half.len() >= 3
                && (0.0..=1.0).contains(&along)
                && (p + (q - p) * along).distance(start)
                    <= 4.0 * chord_deviation + 10.0 * tol.confusion
            {
                if p.distance(start) < 0.1 * h {
                    half.pop();
                }
                half.push(start);
                closed = true;
                break;
            }
            half.push(q);
            (x, p, t) = (y, q, ty);
            last_curvature = curvature;
            h = (2.0 * h)
                .min(H_MAX)
                .min((8.0 * chord_deviation / curvature.max(f64::MIN_POSITIVE)).sqrt());
        }
    }
    let [forward, backward] = halves;
    let mut points: Vec<Point3D> = backward.into_iter().rev().collect();
    points.push(start);
    points.extend(forward);
    (points.len() >= 4).then_some(points)
}

/// Unit tangent `N1 x N2` of the intersection at parameters `x`, or `None`
/// where the surfaces are tangent or a normal degenerates.
fn tangent<A, B>(a: &A, b: &B, x: [f64; 4], tol: Tolerance) -> Option<Vector3D>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let (_, [au, av]) = a.eval_d1(x[0], x[1]);
    let (_, [bu, bv]) = b.eval_d1(x[2], x[3]);
    let (na, nb) = (au.cross(av), bu.cross(bv));
    let t = na.cross(nb);
    let scale = na.magnitude() * nb.magnitude();
    if t.magnitude() <= tol.angular * scale.max(f64::MIN_POSITIVE) {
        None
    } else {
        Some(t * (1.0 / t.magnitude()))
    }
}

/// Parameters `x` moved to first order by the displacement `d` on both
/// surfaces (unchanged on a surface whose tangent plane degenerates).
fn predict<A, B>(a: &A, b: &B, x: [f64; 4], d: Vector3D) -> [f64; 4]
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let shift = |s: &dyn ParametricSurface, u: f64, v: f64| {
        let (_, [su, sv]) = s.eval_d1(u, v);
        let m = [[su.dot(su), su.dot(sv)], [su.dot(sv), sv.dot(sv)]];
        solve_2x2(m, [su.dot(d), sv.dot(d)], 1e-12).unwrap_or([0.0, 0.0])
    };
    let [du1, dv1] = shift(a, x[0], x[1]);
    let [du2, dv2] = shift(b, x[2], x[3]);
    [x[0] + du1, x[1] + dv1, x[2] + du2, x[3] + dv2]
}

/// Newton's method for a point on both surfaces from parameters
/// `x = [u1, v1, u2, v2]`: solves `S1(u1, v1) = S2(u2, v2)`, together with
/// `(S1 - origin) . normal = 0` when `plane` is given; without it the
/// minimum-norm step heads for the nearest intersection point. Steps are
/// halved until the residual drops, and non-periodic parameters stay in
/// their bounds. `None` when it does not converge to `confusion`.
fn meet<A, B>(
    a: &A,
    b: &B,
    x: [f64; 4],
    plane: Option<(Point3D, Vector3D)>,
    tol: Tolerance,
) -> Option<[f64; 4]>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let clamp = |s: &dyn ParametricSurface, u: f64, v: f64| {
        let (u0, u1) = s.u_bounds();
        let (v0, v1) = s.v_bounds();
        (
            if s.u_period().is_none() {
                u.clamp(u0, u1)
            } else {
                u
            },
            if s.v_period().is_none() {
                v.clamp(v0, v1)
            } else {
                v
            },
        )
    };
    let bounded = |x: [f64; 4]| {
        let (u1, v1) = clamp(a, x[0], x[1]);
        let (u2, v2) = clamp(b, x[2], x[3]);
        [u1, v1, u2, v2]
    };
    let residual = |x: [f64; 4]| {
        let pa = a.eval_point(x[0], x[1]);
        let f = pa - b.eval_point(x[2], x[3]);
        let r = plane.map_or(0.0, |(origin, normal)| (pa - origin).dot(normal));
        (f, r, f.dot(f) + r * r)
    };
    let mut x = bounded(x);
    let (mut f, mut r, mut merit) = residual(x);
    for _ in 0..32 {
        if f.magnitude() <= 0.1 * tol.confusion && r.abs() <= 0.1 * tol.confusion {
            return Some(x);
        }
        let (_, [au, av]) = a.eval_d1(x[0], x[1]);
        let (_, [bu, bv]) = b.eval_d1(x[2], x[3]);
        let cols = [au, av, -bu, -bv];
        let delta = match plane {
            Some((_, normal)) => {
                let m = [
                    cols.map(|c| c.x),
                    cols.map(|c| c.y),
                    cols.map(|c| c.z),
                    [cols[0].dot(normal), cols[1].dot(normal), 0.0, 0.0],
                ];
                solve_4x4(m, [-f.x, -f.y, -f.z, -r], 1e-12)?
            }
            None => {
                // Minimum-norm step `J^T y` with `J J^T y = -f`.
                let rows = [cols.map(|c| c.x), cols.map(|c| c.y), cols.map(|c| c.z)];
                let dot = |p: [f64; 4], q: [f64; 4]| (0..4).map(|k| p[k] * q[k]).sum::<f64>();
                let m = rows.map(|p| rows.map(|q| dot(p, q)));
                let y = solve_3x3(m, [-f.x, -f.y, -f.z], 1e-12)?;
                cols.map(|c| c.dot(Vector3D::new(y[0], y[1], y[2])))
            }
        };
        let mut scale = 1.0;
        let mut improved = false;
        for _ in 0..12 {
            let trial = bounded([0, 1, 2, 3].map(|k| x[k] + scale * delta[k]));
            let (tf, tr, tm) = residual(trial);
            if tm < merit {
                (x, f, r, merit) = (trial, tf, tr, tm);
                improved = true;
                break;
            }
            scale *= 0.5;
        }
        if !improved {
            break;
        }
    }
    (f.magnitude() <= tol.confusion && r.abs() <= tol.confusion).then_some(x)
}

/// Clamp a parameter bound pair to a finite seed window.
fn seed_window((lo, hi): (f64, f64)) -> (f64, f64) {
    const WINDOW: f64 = 10.0;
    (
        if lo.is_finite() { lo } else { -WINDOW },
        if hi.is_finite() { hi } else { WINDOW },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tolerance;
    use crate::{Line3D, LineCircle3DIntersection, Point3D, Sphere, Vector3D};

    #[test]
    fn test_generic_line_sphere_matches_analytic() {
        let tol = Tolerance::DEFAULT;
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        let hits = intersect_curve_surface(&line, &sphere, tol);
        assert_eq!(hits.len(), 2);
        assert!((hits[0].curve_param + 2.0).abs() < 1e-9);
        assert!((hits[1].curve_param - 2.0).abs() < 1e-9);
        // Cross-check against the analytic solver.
        match line.intersect(&sphere, tol) {
            LineQuadricIntersection::TwoPoints((t1, _), (t2, _)) => {
                assert!((hits[0].curve_param - t1).abs() < 1e-9);
                assert!((hits[1].curve_param - t2).abs() < 1e-9);
            }
            _ => panic!("analytic expected two points"),
        }
    }

    #[test]
    fn test_generic_line_sphere_miss() {
        let tol = Tolerance::DEFAULT;
        let line = Line3D::new(Point3D::new(0.0, 0.0, 3.0), Vector3D::X).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        assert!(intersect_curve_surface(&line, &sphere, tol).is_empty());
    }

    /// Hits where the surface parametrization is singular (`Su × Sv = 0`):
    /// the Newton system on `(t, u, v)` is singular there.
    #[test]
    fn test_generic_curve_surface_hits_at_singular_points() {
        use crate::{BSplineCurve3D, Cone};
        let tol = Tolerance::DEFAULT;
        let on_both = |hit: &CurveSurfaceHit, surface: &dyn ParametricSurface| {
            surface.contains(hit.point, tol)
                && surface
                    .eval_point(hit.surface_params.0, hit.surface_params.1)
                    .distance(hit.point)
                    <= 1e-6
        };

        // A line along the sphere's axis crosses at both poles.
        let sphere = Sphere::new(Point3D::ORIGIN, 1.0).unwrap();
        let line = Line3D::new(Point3D::new(0.0, 0.0, -3.0), Vector3D::Z).unwrap();
        let hits = intersect_curve_surface(&line, &sphere, tol);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert!((hits[0].curve_param - 2.0).abs() < 1e-9, "{hits:?}");
        assert!((hits[1].curve_param - 4.0).abs() < 1e-9, "{hits:?}");
        assert!(hits.iter().all(|h| on_both(h, &sphere)));

        // A line through a cone's apex, square to the axis, meets it there
        // only.
        let cone = Cone::new(Point3D::new(0.0, 0.0, -1.0), Vector3D::Z, 0.5, 0.5).unwrap();
        let apex = cone.apex();
        let line = Line3D::new(apex - Vector3D::X * 3.0, Vector3D::X).unwrap();
        let hits = intersect_curve_surface(&line, &cone, tol);
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert!(hits[0].point.distance(apex) < 1e-6, "{hits:?}");
        assert!(on_both(&hits[0], &cone));

        // A B-spline curve, which has no exact solver, crossing the sphere
        // at its north pole (t = 0.5) and once more on the way out.
        let poles = vec![
            Point3D::new(-1.0, 0.0, 3.0),
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 1.0),
        ];
        let curve = BSplineCurve3D::new(2, poles, vec![0.0, 1.0], vec![3, 3], false).unwrap();
        assert_eq!(curve.eval_point(0.5), Point3D::new(0.0, 0.0, 1.0));
        let hits = intersect_curve_surface(&curve, &sphere, tol);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert!((hits[0].curve_param - 0.5).abs() < 1e-9, "{hits:?}");
        assert!(hits.iter().all(|h| on_both(h, &sphere)));
    }

    #[test]
    fn test_generic_segment_plane_hit() {
        use crate::{BSplineCurve3D, Plane};
        let tol = Tolerance::DEFAULT;
        let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
        let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
        let plane = Plane::new(Point3D::new(1.0, 0.0, 0.0), Vector3D::X).unwrap();
        let hits = intersect_curve_surface(&curve, &plane, tol);
        assert_eq!(hits.len(), 1);
        assert!((hits[0].curve_param - 0.5).abs() < 1e-9);
        assert_eq!(hits[0].point, Point3D::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_marching_plane_plane_line() {
        use crate::Plane;
        let tol = Tolerance::DEFAULT;
        let xy = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let zy = Plane::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let curves = marching_intersection(&xy, &zy, tol);
        assert_eq!(curves.len(), 1);
        // The traced line runs along Y through the origin.
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let p = curves[0].eval_point(t);
            assert!(p.x.abs() < 1e-6, "{p:?}");
            assert!(p.z.abs() < 1e-6, "{p:?}");
        }
    }

    #[test]
    fn test_marching_plane_sphere_circle() {
        use crate::{Plane, Sphere};
        let tol = Tolerance::DEFAULT;
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        let curves = marching_intersection(&plane, &sphere, tol);
        assert_eq!(curves.len(), 1);
        // Approximate fit: residuals within marching scale (not confusion).
        for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
            let p = curves[0].eval_point(t);
            assert!(plane.project_point(p, tol).distance < 1e-3, "{p:?}");
            assert!(sphere.project_point(p, tol).distance < 1e-3, "{p:?}");
        }
    }

    #[test]
    fn test_marching_loop_is_traced_once() {
        use crate::{Plane, Sphere};
        let tol = Tolerance::DEFAULT;
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        let curves = marching_intersection(&plane, &sphere, tol);
        let curve = assert_one_curve_on_both(&curves, &plane, &sphere);
        let (t0, t1) = curve.bounds();
        let n = 2000;
        let length: f64 = (0..n)
            .map(|k| {
                let at = |k: usize| curve.eval_point(t0 + (t1 - t0) * k as f64 / n as f64);
                at(k).distance(at(k + 1))
            })
            .sum();
        let circumference = 4.0 * std::f64::consts::PI;
        assert!(
            (length - circumference).abs() < 1e-3 * circumference,
            "{length}"
        );
    }

    /// Asserts one traced curve whose samples lie on both surfaces.
    fn assert_one_curve_on_both<A, B>(curves: &[BSplineCurve3D], a: &A, b: &B) -> BSplineCurve3D
    where
        A: ParametricSurface,
        B: ParametricSurface,
    {
        let tol = Tolerance::DEFAULT;
        assert_eq!(curves.len(), 1);
        let curve = curves[0].clone();
        let (t0, t1) = curve.bounds();
        for k in 0..=40 {
            let p = curve.eval_point(t0 + (t1 - t0) * k as f64 / 40.0);
            assert!(a.project_point(p, tol).distance < 1e-3, "{p:?}");
            assert!(b.project_point(p, tol).distance < 1e-3, "{p:?}");
        }
        curve
    }

    #[test]
    fn test_marching_plane_sphere_off_grid_heights() {
        use crate::{Plane, Sphere};
        let tol = Tolerance::DEFAULT;
        let sphere = Sphere::new(Point3D::ORIGIN, 1.0).unwrap();
        // No grid row of either surface lands on these circles.
        for h in [0.3, 0.8, -0.45, 0.123] {
            let plane = Plane::new(Point3D::new(0.0, 0.0, h), Vector3D::Z).unwrap();
            let curves = marching_intersection(&plane, &sphere, tol);
            let curve = assert_one_curve_on_both(&curves, &plane, &sphere);
            let r = (1.0 - h * h).sqrt();
            for t in [0.0, 0.3, 0.6, 0.9] {
                let p = curve.eval_point(t);
                assert!((p.x.hypot(p.y) - r).abs() < 1e-3, "h = {h}: {p:?}");
            }
        }
    }

    #[test]
    fn test_marching_cylinder_offset_sphere() {
        use crate::{Cylinder, Sphere};
        let tol = Tolerance::DEFAULT;
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
        let sphere = Sphere::new(Point3D::new(0.8, 0.0, 0.0), 1.78).unwrap();
        let curves = marching_intersection(&cylinder, &sphere, tol);
        assert_one_curve_on_both(&curves, &cylinder, &sphere);
    }

    #[test]
    fn test_marching_shallow_angle_spheres() {
        use crate::Sphere;
        let tol = Tolerance::DEFAULT;
        // Unit spheres whose centers are `d` apart meet at about `d`
        // radians, on the circle z = d/2.
        let a = Sphere::new(Point3D::ORIGIN, 1.0).unwrap();
        for d in [0.05, 0.0175] {
            let b = Sphere::new(Point3D::new(0.0, 0.0, d), 1.0).unwrap();
            let curves = marching_intersection(&a, &b, tol);
            let curve = assert_one_curve_on_both(&curves, &a, &b);
            let r = (1.0 - d * d / 4.0).sqrt();
            for t in [0.0, 0.3, 0.6, 0.9] {
                let p = curve.eval_point(t);
                assert!((p.z - d / 2.0).abs() < 1e-3, "d = {d}: {p:?}");
                assert!((p.x.hypot(p.y) - r).abs() < 1e-3, "d = {d}: {p:?}");
            }
        }
    }

    #[test]
    fn test_marching_small_loop() {
        // A loop much smaller than the default step: the step follows the
        // curvature.
        use crate::{Plane, Sphere};
        let tol = Tolerance::DEFAULT;
        let plane = Plane::new(Point3D::new(0.0, 0.0, 0.003), Vector3D::Z).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 0.01).unwrap();
        let curves = marching_intersection(&plane, &sphere, tol);
        let curve = assert_one_curve_on_both(&curves, &plane, &sphere);
        let r = (1e-4f64 - 9e-6).sqrt();
        for t in [0.0, 0.3, 0.6, 0.9] {
            let p = curve.eval_point(t);
            assert!((p.x.hypot(p.y) - r).abs() < 1e-4, "{p:?}");
        }
    }

    #[test]
    fn test_marching_open_trace_is_not_closed() {
        use crate::Plane;
        let tol = Tolerance::DEFAULT;
        let xy = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let zy = Plane::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let curves = marching_intersection(&xy, &zy, tol);
        assert_eq!(curves.len(), 1);
        let (t0, t1) = curves[0].bounds();
        let gap = curves[0].eval_point(t0).distance(curves[0].eval_point(t1));
        assert!(gap > 10.0, "open line fitted as a loop: end gap {gap}");
    }

    /// Polyline length of a traced curve.
    fn traced_length(curve: &BSplineCurve3D) -> f64 {
        let (t0, t1) = curve.bounds();
        let n = 4000;
        let at = |k: usize| curve.eval_point(t0 + (t1 - t0) * k as f64 / n as f64);
        (0..n).map(|k| at(k).distance(at(k + 1))).sum()
    }

    /// The torus of radii 3 and 1 about z, and the plane through its
    /// center tilted `asin(1/3) + d` about y.
    fn villarceau_pair(d: f64) -> (crate::Torus, crate::Plane) {
        let torus = crate::Torus::new(Point3D::ORIGIN, Vector3D::Z, 3.0, 1.0).unwrap();
        let a = (1.0f64 / 3.0).asin() + d;
        let plane =
            crate::Plane::new(Point3D::ORIGIN, Vector3D::new(-a.sin(), 0.0, a.cos())).unwrap();
        (torus, plane)
    }

    #[test]
    fn test_marching_stops_at_a_self_crossing() {
        // Issue #7: at the Villarceau tilt the section is two circles of
        // radius 3 crossing where the plane touches the tube. A trace used
        // to turn onto the other circle there and cover an arc twice.
        let tol = Tolerance::DEFAULT;
        let (torus, plane) = villarceau_pair(0.0);
        let curves = marching_intersection(&torus, &plane, tol);
        assert!(!curves.is_empty());
        for curve in &curves {
            let (t0, t1) = curve.bounds();
            for k in 0..=40 {
                let p = curve.eval_point(t0 + (t1 - t0) * k as f64 / 40.0);
                assert!(torus.project_point(p, tol).distance < 1e-3, "{p:?}");
                assert!(plane.project_point(p, tol).distance < 1e-3, "{p:?}");
            }
        }
        let total: f64 = curves.iter().map(traced_length).sum();
        let expected = 4.0 * std::f64::consts::PI * 3.0;
        assert!(
            (total - expected).abs() < 1e-3 * expected,
            "total length {total}, expected {expected}: {:?}",
            curves.iter().map(traced_length).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_marching_near_the_villarceau_tilt_gives_two_loops() {
        let tol = Tolerance::DEFAULT;
        for d in [-1e-3, 1e-3] {
            let (torus, plane) = villarceau_pair(d);
            let curves = marching_intersection(&torus, &plane, tol);
            assert_eq!(curves.len(), 2, "d = {d}");
            for curve in &curves {
                let (t0, t1) = curve.bounds();
                let gap = curve.eval_point(t0).distance(curve.eval_point(t1));
                assert!(gap < 1e-6, "d = {d}: open curve, end gap {gap}");
            }
        }
    }

    #[test]
    fn test_marching_miss_is_empty() {
        use crate::{Plane, Sphere};
        let tol = Tolerance::DEFAULT;
        let plane = Plane::new(Point3D::new(0.0, 0.0, 5.0), Vector3D::Z).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        assert!(marching_intersection(&plane, &sphere, tol).is_empty());
    }

    #[test]
    fn test_curve_curve_extrema_skew_matches_analytic() {
        use crate::{Line3D, LineLine3DIntersection};
        let tol = Tolerance::DEFAULT;
        let l1 = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let l2 = Line3D::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Y).unwrap();
        let ext = curve_curve_extrema(&l1, &l2, tol);
        assert_eq!(ext.len(), 1);
        assert!((ext[0].distance - 1.0).abs() < 1e-9);
        match l1.intersect(&l2, tol) {
            LineLine3DIntersection::Skew { s, t, distance } => {
                assert!((ext[0].first_param - s).abs() < 1e-9);
                assert!((ext[0].second_param - t).abs() < 1e-9);
                assert!((ext[0].distance - distance).abs() < 1e-12);
            }
            _ => panic!("analytic expected skew"),
        }
    }

    fn segment(from: Point3D, to: Point3D) -> crate::TrimmedCurve3D<Line3D> {
        crate::TrimmedCurve3D::segment(from, to).unwrap()
    }

    fn arc(
        center: Point3D,
        radius: f64,
        lo: f64,
        hi: f64,
    ) -> crate::TrimmedCurve3D<crate::Circle3D> {
        let circle = crate::Circle3D::new(center, Vector3D::Z, radius).unwrap();
        crate::TrimmedCurve3D::new(circle, lo, hi).unwrap()
    }

    /// Smallest distance over an `n x n` parameter grid of two bounded
    /// curves.
    fn grid_min<A: ParametricCurve3D, B: ParametricCurve3D>(a: &A, b: &B, n: usize) -> f64 {
        let ((a0, a1), (b0, b1)) = (a.bounds(), b.bounds());
        let at = |lo: f64, hi: f64, i: usize| lo + (hi - lo) * i as f64 / (n - 1) as f64;
        let pb: Vec<Point3D> = (0..n).map(|j| b.eval_point(at(b0, b1, j))).collect();
        (0..n)
            .flat_map(|i| {
                let p = a.eval_point(at(a0, a1, i));
                pb.iter().map(move |q| p.distance(*q))
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// Every reported extremum is a local minimum on the parameter box:
    /// no nearby feasible pair is closer.
    fn assert_local_minima<A: ParametricCurve3D, B: ParametricCurve3D>(
        a: &A,
        b: &B,
        ext: &[CurveCurveExtremum],
    ) {
        let ((a0, a1), (b0, b1)) = (a.bounds(), b.bounds());
        for e in ext {
            for (ds, dt) in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)] {
                for sign in [-1.0, 1.0] {
                    let h = 1e-4 * sign;
                    let s = (e.first_param + h * ds).clamp(a0, a1);
                    let t = (e.second_param + h * dt).clamp(b0, b1);
                    let d = a.eval_point(s).distance(b.eval_point(t));
                    assert!(
                        d >= e.distance - 1e-12,
                        "{e:?} is not a minimum: {d} at ({s}, {t})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_curve_curve_extrema_concentric_arcs_corner() {
        // r5 on [0, 2] against r10 on [2, 4]: the arcs only line up at
        // the shared end angle 2, a corner of the parameter box.
        let tol = Tolerance::DEFAULT;
        let a = arc(Point3D::ORIGIN, 5.0, 0.0, 2.0);
        let b = arc(Point3D::ORIGIN, 10.0, 2.0, 4.0);
        let ext = curve_curve_extrema(&a, &b, tol);
        assert!((ext[0].distance - 5.0).abs() < 1e-12, "{ext:?}");
        assert_eq!((ext[0].first_param, ext[0].second_param), (2.0, 2.0));
        // The far corner (0, 4) is a genuine local minimum of the box
        // too; nothing else is reported.
        assert_eq!(ext.len(), 2, "{ext:?}");
        assert_eq!((ext[1].first_param, ext[1].second_param), (0.0, 4.0));
        assert_local_minima(&a, &b, &ext);
    }

    #[test]
    fn test_curve_curve_extrema_concentric_arcs_same_range() {
        // Same angular range: a valley of minima, reported once.
        let tol = Tolerance::DEFAULT;
        let a = arc(Point3D::ORIGIN, 5.0, 0.0, 2.0);
        let b = arc(Point3D::ORIGIN, 10.0, 0.0, 2.0);
        let ext = curve_curve_extrema(&a, &b, tol);
        assert_eq!(ext.len(), 1, "{ext:?}");
        assert!((ext[0].distance - 5.0).abs() < 1e-12);
        assert_eq!(ext[0].first_param, 0.0);
        assert!(ext[0].second_param.abs() < 1e-9, "{ext:?}");
    }

    #[test]
    fn test_curve_curve_extrema_concentric_circles() {
        let tol = Tolerance::DEFAULT;
        let a = crate::Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 100.0).unwrap();
        let b = crate::Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 50.0).unwrap();
        let ext = curve_curve_extrema(&a, &b, tol);
        assert_eq!(ext.len(), 1, "{ext:?}");
        assert!((ext[0].distance - 50.0).abs() < 1e-12);
        assert_eq!(ext[0].first_param, 0.0);
        assert!(ext[0].second_param.abs() < 1e-9, "{ext:?}");
    }

    #[test]
    fn test_curve_curve_extrema_parallel_segments() {
        // Offset (0, 4, 2): distance sqrt(20) over the overlap.
        let tol = Tolerance::DEFAULT;
        let (p0, p1) = (Point3D::ORIGIN, Point3D::new(10.0, 0.0, 0.0));
        let (q0, q1) = (Point3D::new(2.0, 4.0, 2.0), Point3D::new(8.0, 4.0, 2.0));
        let ext = curve_curve_extrema(&segment(p0, p1), &segment(q0, q1), tol);
        assert_eq!(ext.len(), 1, "{ext:?}");
        assert!((ext[0].distance - 4.47213595499958).abs() < 1e-12);
        assert!((ext[0].first_param - 2.0).abs() < 1e-12);
        assert_eq!(ext[0].second_param, 0.0);
        // The same pair as degree-1 B-splines.
        let spline = |p: Point3D, q: Point3D| {
            BSplineCurve3D::new(1, vec![p, q], vec![0.0, 1.0], vec![2, 2], false).unwrap()
        };
        let ext = curve_curve_extrema(&spline(p0, p1), &spline(q0, q1), tol);
        assert_eq!(ext.len(), 1, "{ext:?}");
        assert!((ext[0].distance - 20f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_curve_curve_extrema_parallel_lines() {
        use crate::LineLine3DIntersection;
        let tol = Tolerance::DEFAULT;
        let a = Line3D::new(Point3D::new(3.0, 0.0, 0.0), Vector3D::X).unwrap();
        let b = Line3D::new(Point3D::new(0.0, 9.0, 5.0), Vector3D::new(-2.0, 0.0, 0.0)).unwrap();
        let ext = curve_curve_extrema(&a, &b, tol);
        assert_eq!(ext.len(), 1, "{ext:?}");
        assert!((ext[0].distance - 106f64.sqrt()).abs() < 1e-12);
        assert_eq!(ext[0].first_param, 0.0);
        // The representative's foot: (3, 9, 5) on the second line.
        let foot = b.eval_point(ext[0].second_param);
        assert!(foot.distance(Point3D::new(3.0, 9.0, 5.0)) < 1e-12);
        match a.intersect(&b, tol) {
            LineLine3DIntersection::Parallel(d) => assert!((ext[0].distance - d).abs() < 1e-12),
            _ => panic!("analytic expected parallel"),
        }
    }

    #[test]
    fn test_curve_curve_extrema_opposite_lines() {
        let tol = Tolerance::DEFAULT;
        let a = Line3D::new(Point3D::new(1.0, 0.0, 0.0), Vector3D::new(0.0, 0.0, -1.0)).unwrap();
        let b = Line3D::new(Point3D::new(5.0, 0.0, 0.0), Vector3D::Z).unwrap();
        let ext = curve_curve_extrema(&a, &b, tol);
        assert_eq!(ext.len(), 1, "{ext:?}");
        assert_eq!(ext[0].distance, 4.0);
        assert_eq!((ext[0].first_param, ext[0].second_param), (0.0, 0.0));
    }

    #[test]
    fn test_curve_curve_extrema_segment_end_to_interior() {
        // The first segment's end (s = 1) is closest to the second's
        // middle (t = 0.5), at distance 1.5 * sqrt(2).
        let tol = Tolerance::DEFAULT;
        let a = segment(Point3D::ORIGIN, Point3D::new(1.0, 0.0, 0.0));
        let b = segment(Point3D::new(2.5, -0.5, 1.5), Point3D::new(2.5, 0.5, 1.5));
        let ext = curve_curve_extrema(&a, &b, tol);
        assert_eq!(ext.len(), 1, "{ext:?}");
        assert_eq!(ext[0].first_param, 1.0);
        assert!((ext[0].second_param - 0.5).abs() < 1e-12);
        assert!((ext[0].distance - 1.5 * 2f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_curve_curve_extrema_line_far_minima() {
        // An ellipse around the X axis at x = 40, far outside any fixed
        // window: two minima (distance 3) at line parameter 40.
        let tol = Tolerance::DEFAULT;
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let ellipse = crate::Ellipse3D::new(
            Point3D::new(40.0, 0.0, 0.0),
            Vector3D::X,
            Vector3D::Y,
            5.0,
            3.0,
        )
        .unwrap();
        let ext = curve_curve_extrema(&line, &ellipse, tol);
        assert_eq!(ext.len(), 2, "{ext:?}");
        let quarter = std::f64::consts::FRAC_PI_2;
        let mut params: Vec<f64> = ext.iter().map(|e| e.second_param).collect();
        params.sort_by(f64::total_cmp);
        assert!((params[0] - quarter).abs() < 1e-7, "{ext:?}");
        assert!((params[1] - 3.0 * quarter).abs() < 1e-7, "{ext:?}");
        for e in &ext {
            assert!((e.distance - 3.0).abs() < 1e-12, "{e:?}");
            assert!((e.first_param - 40.0).abs() < 1e-9, "{e:?}");
        }
        // Swapped order swaps the parameters.
        let swapped = curve_curve_extrema(&ellipse, &line, tol);
        assert_eq!(swapped.len(), 2);
        assert!((swapped[0].second_param - 40.0).abs() < 1e-9);
    }

    #[test]
    fn test_curve_curve_extrema_parabola_far_circle() {
        // A small circle beside a parabola far from its apex (u = 60):
        // the minimum, distance 4, is found without a fixed window.
        use crate::{Circle3D, Parabola3D};
        let tol = Tolerance::DEFAULT;
        let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
        let u = 60.0;
        let normal = parabola
            .eval_derivative(u, 1)
            .cross(Vector3D::Z)
            .normalized()
            .unwrap();
        let center = parabola.eval_point(u) + normal * 5.0;
        let circle = Circle3D::new(center, Vector3D::Z, 1.0).unwrap();
        let ext = curve_curve_extrema(&parabola, &circle, tol);
        assert!((ext[0].distance - 4.0).abs() < 1e-9, "{ext:?}");
        assert!((ext[0].first_param - u).abs() < 1e-7, "{ext:?}");
    }

    #[test]
    fn test_curve_curve_extrema_random_arcs_beat_grid() {
        // Random trimmed arc pairs, general and concentric: the reported
        // minimum is never worse than a dense grid, and every reported
        // pair is a genuine local minimum on the parameter box.
        let tol = Tolerance::DEFAULT;
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut rand = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for case in 0..120 {
            let ca = Point3D::new(10.0 * rand() - 5.0, 10.0 * rand() - 5.0, 0.0);
            let cb = if case % 2 == 1 {
                ca
            } else {
                Point3D::new(10.0 * rand() - 5.0, 10.0 * rand() - 5.0, 0.0)
            };
            let (ra, rb) = (1.0 + 9.0 * rand(), 1.0 + 9.0 * rand());
            let (a0, b0) = (6.0 * rand(), 6.0 * rand());
            let a = arc(ca, ra, a0, a0 + 0.2 + 3.0 * rand());
            let b = arc(cb, rb, b0, b0 + 0.2 + 3.0 * rand());
            let ext = curve_curve_extrema(&a, &b, tol);
            assert!(!ext.is_empty(), "case {case}");
            let grid = grid_min(&a, &b, 400);
            assert!(
                ext[0].distance <= grid + 1e-12,
                "case {case}: {ext:?} > grid {grid}, arcs {:?} {:?} {:?} {:?} {:?} {:?}",
                ca,
                ra,
                a.bounds(),
                cb,
                rb,
                b.bounds()
            );
            assert_local_minima(&a, &b, &ext);
        }
    }

    #[test]
    fn test_intersect_curve_curve_line_circle() {
        use crate::{Circle3D, Line3D};
        let tol = Tolerance::DEFAULT;
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let hits = intersect_curve_curve(&line, &circle, tol);
        assert_eq!(hits.len(), 2);
        assert!((hits[0].first_param + 2.0).abs() < 1e-9);
        assert!((hits[1].first_param - 2.0).abs() < 1e-9);
        // Cross-check against the analytic solver.
        match line.intersect(&circle, tol) {
            LineCircle3DIntersection::Points((t1, _), (t2, _)) => {
                assert!((hits[0].first_param - t1).abs() < 1e-9);
                assert!((hits[1].first_param - t2).abs() < 1e-9);
            }
            _ => panic!("analytic expected two points"),
        }
    }
}
