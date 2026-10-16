//! Intersections of curves and surfaces, and their results.
//!
//! [`Intersect::intersect`] is the entry point: it solves a pair exactly
//! when a closed form exists and runs a numeric search otherwise. Its
//! family results ([`CurveCurveIntersection`],
//! [`CurveCurveIntersection2D`], [`CurveSurfaceIntersection`]) carry the
//! hits, the overlaps, the method that ran ([`IntersectionMethod`]) and
//! whether every hit was found ([`Completeness`]); for two surfaces,
//! [`SurfaceSurfaceIntersection`] carries the curves, the contact points
//! and whether the surfaces coincide. A hit stores the
//! parameters the solver computed on its way to the point and leaves the
//! others `None`, for the caller to recover with the operand's
//! `parameter_of` or `parameters_of` when it needs them.
//!
//! ```
//! use geomcore::{Circle3D, Completeness, Intersect, IntersectionMethod, Line3D, Point3D};
//! use geomcore::{Tolerance, Vector3D};
//!
//! let line = Line3D::new(Point3D::new(-5.0, 0.0, 0.0), Vector3D::X).unwrap();
//! let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
//! let result = line.intersect(&circle, Tolerance::DEFAULT);
//! assert_eq!(result.method(), IntersectionMethod::Exact);
//! assert_eq!(result.completeness(), &Completeness::Complete);
//! assert_eq!(result.hits().len(), 2);
//! ```
//!
//! [`IntersectExact::intersect_exact`] is the typed fast path. On two
//! elementary curves or surfaces it returns one of the pair's own enums:
//! the intersecting geometry on success, or an explicit classification
//! when the pair is tangent, disjoint, parallel, or coincident. Tangency
//! and coincidence are decided with [`crate::Tolerance`], so
//! near-degenerate configurations classify robustly instead of collapsing
//! to noise. It recovers no parameters it does not need and never falls
//! back to a numeric search.
//!
//! When the types are only known at run time, the same call on
//! [`crate::surfaces::Surface`], [`crate::curves::Curve3D`] or their
//! elementary enums returns a general result instead:
//! [`ExactSurfaceIntersection`], [`ExactCurveSurfaceIntersection`] or
//! [`ExactCurveCurveIntersection`].

use crate::{
    Circle3D, Ellipse3D, Hyperbola3D, Line3D, Parabola3D, Plane, Point2D, Point3D, Vector3D,
};

mod completeness;
mod conic;
mod curve_curve;
mod curve_curve_hull;
mod curve_surface;
mod curve_surface_hull;
mod dispatch;
mod inline_or_heap_list;
mod marching;
mod mixed;
mod overlap_runs;
mod pieces;
mod results;
mod routing;
pub use completeness::{Completeness, IncompleteReason};
mod surface_surface;
pub(crate) use conic::{Conic, intersect_conic_conic, intersect_conic_quadric};
pub use curve_curve::curve_curve_extrema;
pub use curve_curve::intersect_curve_curve;
#[cfg(test)]
pub(crate) use curve_curve::search_curve_curve;
pub use curve_curve_hull::{CurveCurveExtrema, CurveCurveRegion};
pub use curve_surface::intersect_curve_surface;
#[cfg(test)]
pub(crate) use curve_surface::search_curve_surface;
pub use curve_surface_hull::CurveSurfaceRegion;
pub use dispatch::{
    ExactCurveCurveIntersection, ExactCurveSurfaceIntersection, ExactSurfaceIntersection,
    InlineList, IntersectExact,
};
pub(crate) use results::curve_surface_multiplicity;
pub use results::{
    CurveCurveIntersection, CurveCurveIntersection2D, CurveCurveOverlap, CurveSurfaceIntersection,
    IntersectionCurve, IntersectionMethod, SurfaceSurfaceIntersection, SurfaceSurfaceRegion,
};
pub use routing::Intersect;

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

/// One hit of an exact line-vs-torus solve: the line parameter, the
/// hit point, the torus `(u, v)` parameters, and the root multiplicity
/// (2+ marks grazing tangency).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineTorusHit {
    /// Parameter on the line.
    pub line_parameter: f64,
    /// The hit point.
    pub point: Point3D,
    /// Parameters on the torus.
    pub surface_parameters: (f64, f64),
    /// Root multiplicity (≥ 2 signals tangency).
    pub multiplicity: u32,
}

/// Result of intersecting a sphere with a cylinder.
///
/// The closed form requires the cylinder axis through the sphere
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
    /// No closed form for this configuration.
    NotAnalytic,
}

/// Result of intersecting a sphere with a cone.
///
/// The closed form requires the sphere center on the cone axis
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
    /// No closed form for this configuration.
    NotAnalytic,
}

/// Result of intersecting two cylinders.
///
/// The closed form requires parallel axes (within `tol.angular`);
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
    /// No closed form for this configuration.
    NotAnalytic,
}

/// Result of intersecting a cone with a cylinder.
///
/// The closed form needs coaxial axes (within tolerance): the tube of
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
    /// No closed form for this configuration.
    NotAnalytic,
}

/// Result of intersecting a torus with a plane.
///
/// The closed form needs the plane through the axis (two meridian
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
    /// No closed form for this configuration.
    NotAnalytic,
}

/// Result of intersecting a torus with a sphere.
///
/// The closed form needs the sphere center on the torus axis, reducing
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
    /// No closed form for this configuration.
    NotAnalytic,
}

/// Result of intersecting a torus with a cylinder.
///
/// The closed form needs coaxial axes. Anything else reports
/// [`TorusCylinderIntersection::NotAnalytic`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TorusCylinderIntersection {
    /// Two latitude rings symmetric about the equatorial plane.
    TwoCircles(Circle3D, Circle3D),
    /// Grazing ring in the equatorial plane.
    TangentCircle(Circle3D),
    /// The tube misses the torus.
    Empty,
    /// No closed form for this configuration.
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
/// The closed form needs coaxial axes, reducing to a quadratic in the
/// axial coordinate per nappe. Anything else reports
/// [`TorusConeIntersection::NotAnalytic`].
#[derive(Debug, Clone, PartialEq)]
pub enum TorusConeIntersection {
    /// Latitude rings, up to two per nappe, ordered along the torus axis.
    Rings(Vec<LatitudeRing>),
    /// The cone misses the torus.
    Empty,
    /// No closed form for this configuration.
    NotAnalytic,
}

/// Result of intersecting two tori.
///
/// The closed form needs collinear axes, reducing to quadratics in the
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
    /// No closed form for this configuration.
    NotAnalytic,
}

/// Result of intersecting two cones.
///
/// The closed form needs coaxial axes: equal latitude-ring radii give
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
    /// No closed form for this configuration.
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
/// This typed solver handles coplanar pairs (via 2D images in the shared
/// plane). Distinct parallel planes are [`CircleCircle3DIntersection::Empty`];
/// crossing planes report [`CircleCircle3DIntersection::NotAnalytic`]: they
/// meet in up to two points, one of which may be a lone crossing, which the
/// variants here cannot hold. The routed
/// [`Intersect::intersect`] solves them exactly
/// as conics, on the line the planes share.
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
    /// The circle planes cross: no closed form in this typed solver (the
    /// routed `intersect` solves the pair exactly as conics).
    NotAnalytic,
}

/// One hit of an exact line-vs-conic solve: parameter on the line,
/// the hit point, and parameter on the conic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineConicHit {
    /// Parameter on the line.
    pub line_parameter: f64,
    /// The hit point.
    pub point: Point3D,
    /// Parameter on the conic.
    pub conic_parameter: f64,
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
    /// How far the conic reaches from its origin: infinite for a parabola
    /// or a hyperbola.
    pub extent: f64,
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
/// use geomcore::{ConicConicIntersection, Ellipse3D, Frame3D, Hyperbola3D, IntersectExact, Tolerance};
/// let ellipse = Ellipse3D::from_frame(Frame3D::WORLD, 3.0, 2.0).unwrap();
/// let hyperbola = Hyperbola3D::from_frame(Frame3D::WORLD, 1.0, 0.5).unwrap();
/// match ellipse.intersect_exact(&hyperbola, Tolerance::DEFAULT) {
///     // The branch x > 0 meets the ellipse at (2.4739, ±1.1314).
///     ConicConicIntersection::Hits(hits) => assert_eq!(hits.len(), 2),
///     other => panic!("expected two hits, got {other:?}"),
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum ConicConicIntersection {
    /// The hits, ordered by parameter on the first conic when every hit
    /// stores one; a multiplicity of 2 or more marks a tangency. Only the
    /// parameter of the conic that was solved for is stored.
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
    pub first_parameter: f64,
    /// Parameter on the second curve.
    pub second_parameter: f64,
    /// Distance between the two points.
    pub distance: f64,
}

/// One meeting of two 3D curves.
///
/// A parameter is stored only when the solver computed it on its way to
/// the point; the others are `None`, and the caller recovers one with the
/// curve's `parameter_of(hit.point)` when it needs it.
///
/// # Examples
///
/// ```
/// use geomcore::{Circle3D, Intersect, Point3D, Tolerance, Vector3D};
///
/// // Two coplanar circles: the solver finds the points, not the angles.
/// let first = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
/// let second = Circle3D::new(Point3D::new(2.0, 0.0, 0.0), Vector3D::Z, 2.0).unwrap();
/// let result = first.intersect(&second, Tolerance::DEFAULT);
/// let hit = result.hits()[0];
/// assert_eq!(hit.first_parameter, None);
/// let angle = first.parameter_of(hit.point);
/// assert!(first.eval_point(angle).distance(hit.point) < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveCurveHit {
    /// Parameter on the first curve, in its bounds or the trimmed curve's
    /// range: the hit lies within `tol.confusion` in space of the range,
    /// and a hit just beyond an end (the seam included) reports that end.
    /// `None` when the solver did not compute it; recover it with the
    /// curve's `parameter_of(hit.point)`.
    pub first_parameter: Option<f64>,
    /// Parameter on the second curve, in its bounds or the trimmed curve's
    /// range, within `tol.confusion` in space of the range (a hit just
    /// beyond an end reports that end). `None` when the solver did not
    /// compute it; recover it with the curve's `parameter_of(hit.point)`.
    pub second_parameter: Option<f64>,
    /// The meeting point, within tolerance of both curves.
    pub point: Point3D,
    /// 1 where the curves cross, 2 or more where they are tangent.
    pub multiplicity: u32,
}

/// One meeting of two 2D curves.
///
/// A parameter is stored only when the solver computed it on its way to
/// the point (a line's, or a trimmed curve's for its range test); the
/// others are `None`, and the caller recovers one with the curve's
/// `parameter_of(hit.point)` when it needs it.
///
/// # Examples
///
/// ```
/// use geomcore::{CurveCurveHit2D, Point2D};
/// let hit = CurveCurveHit2D {
///     first_parameter: Some(1.0),
///     second_parameter: None,
///     point: Point2D::new(1.0, 0.0),
///     multiplicity: 1,
/// };
/// assert_eq!(hit.multiplicity, 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveCurveHit2D {
    /// Parameter on the first curve, in its bounds or the trimmed curve's
    /// range, within `tol.confusion` in space of the range (a hit just
    /// beyond an end reports that end). `None` when the solver did not
    /// compute it; recover it with the curve's `parameter_of(hit.point)`.
    pub first_parameter: Option<f64>,
    /// Parameter on the second curve, in its bounds or the trimmed curve's
    /// range, within `tol.confusion` in space of the range (a hit just
    /// beyond an end reports that end). `None` when the solver did not
    /// compute it; recover it with the curve's `parameter_of(hit.point)`.
    pub second_parameter: Option<f64>,
    /// The meeting point, within tolerance of both curves.
    pub point: Point2D,
    /// 1 where the curves cross, 2 or more where they are tangent.
    pub multiplicity: u32,
}

/// One meeting of a curve and a surface.
///
/// The numeric search stores every parameter. An exact solver stores the
/// curve parameter, which it solves for, and leaves the surface
/// parameters `None`, for the caller to recover with the surface's
/// `parameters_of(hit.point)` when it needs them.
///
/// # Examples
///
/// ```
/// use geomcore::{CurveSurfaceHit, Point3D};
/// let hit = CurveSurfaceHit {
///     curve_parameter: Some(2.0),
///     surface_parameters: None,
///     point: Point3D::new(2.0, 0.0, 0.0),
///     multiplicity: 1,
/// };
/// assert_eq!(hit.curve_parameter, Some(2.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveSurfaceHit {
    /// Parameter on the curve, in its bounds or the trimmed curve's range,
    /// within `tol.confusion` in space of the range (a hit just beyond an
    /// end reports that end). `None` when the solver did not compute it;
    /// recover it with the curve's `parameter_of(hit.point)`.
    pub curve_parameter: Option<f64>,
    /// Parameters on the surface. `None` when the solver did not compute
    /// them; recover them with the surface's `parameters_of(hit.point)`.
    pub surface_parameters: Option<(f64, f64)>,
    /// The meeting point (on the curve; within tolerance of the surface).
    pub point: Point3D,
    /// 1 where the curve crosses the surface, 2 or more where the curve is
    /// tangent to it.
    pub multiplicity: u32,
}
