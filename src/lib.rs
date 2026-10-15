//! A pure, standalone geometric kernel for CAD-grade curves and surfaces.
//!
//! `geomcore` provides parametric evaluation of elementary curves and
//! surfaces (lines, circles, ellipses, parabolas, hyperbolas, B-splines;
//! planes, cylinders, cones, spheres, tori, B-spline surfaces), rigid
//! transformations, and analytic curve-on-surface parametrization: computing
//! the exact 2D representation of a 3D curve in a surface's parameter space
//! wherever a closed form exists.
//!
//! # Namespace layout
//!
//! - The crate root re-exports the placement and value types shared by
//!   everything else: [`Point3D`]/[`Point2D`], [`Vector3D`]/[`Vector2D`],
//!   [`Axis3D`]/[`Axis2D`], [`Frame3D`]/[`Frame2D`], and [`Transform`].
//! - [`curves`] holds the 3D and 2D curve types ([`curves::Line3D`],
//!   [`curves::Circle3D`], [`curves::Ellipse3D`], [`curves::Parabola3D`],
//!   [`curves::Hyperbola3D`], [`curves::BSplineCurve3D`],
//!   [`curves::Line2D`], [`curves::Circle2D`]), the [`curves::Curve3D`]/
//!   [`curves::Curve2D`] enum adaptors, the
//!   [`curves::TrimmedCurve3D`]/[`curves::TrimmedCurve2D`] pieces of a
//!   curve over a parameter range (segments, arcs), and the
//!   [`curves::ParametricCurve3D`]/[`curves::ParametricCurve2D`] traits.
//! - [`surfaces`] holds the five elementary surfaces
//!   ([`surfaces::Plane`], [`surfaces::Cylinder`], [`surfaces::Cone`],
//!   [`surfaces::Sphere`], [`surfaces::Torus`]) plus
//!   [`surfaces::BSplineSurface`], the [`surfaces::Surface`] enum adaptor,
//!   and the [`surfaces::ParametricSurface`] trait.
//! - [`bounding_box`] holds [`BoundingBox3D`] and the [`Bound3D`] trait,
//!   through which curves and surfaces report the axis-aligned box
//!   bounding them.
//!
//! # Quick start
//!
//! ```
//! use geomcore::{Cylinder, Point3D, Tolerance, Vector3D};
//! use geomcore::curves::Circle3D;
//!
//! // Evaluate a circle at a single parameter, and over a batch of parameters.
//! let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
//! let point = circle.eval_point(std::f64::consts::PI / 4.0);
//! let points = circle.eval_points(&[0.0, 1.0, 2.0]);
//! assert_eq!(points[0], circle.eval_point(0.0));
//!
//! // Curve-on-surface parametrization: this circle is coaxial with the
//! // cylinder, so it has an exact 2D image, a horizontal line in (u, v).
//! let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
//! let pcurve = circle.parametrize_on(&cylinder, Tolerance::DEFAULT).unwrap();
//! # let _ = point;
//! ```
//!
//! # Intersections
//!
//! [`Intersect`] is the one entry point for exact intersections. Between
//! two concrete types it returns the pair's own result, in either order:
//!
//! ```
//! use geomcore::{Intersect, PlaneSphereIntersection, Point3D, Vector3D};
//! use geomcore::surfaces::{Plane, Sphere};
//!
//! let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
//! let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
//! let tol = geomcore::Tolerance::default();
//! let PlaneSphereIntersection::Circle(circle) = plane.intersect(&sphere, tol) else {
//!     panic!("expected a circle");
//! };
//! assert_eq!(sphere.intersect(&plane, tol), PlaneSphereIntersection::Circle(circle));
//! ```
//!
//! Between [`surfaces::Surface`] or [`curves::Curve3D`] values it returns
//! a general result such as [`SurfaceIntersection`], which reports
//! `NotAnalytic` for a B-spline or a pair without a closed form; the
//! numeric functions ([`marching_intersection`],
//! [`intersect_curve_surface`], [`intersect_curve_curve`]) cover those.
//!
//! # Extending the kernel
//!
//! New curve or surface types are added by implementing
//! [`curves::ParametricCurve3D`] (or [`curves::ParametricCurve2D`]) and
//! [`surfaces::ParametricSurface`] respectively, without touching the
//! existing dispatch code in [`curves::Curve3D`]/[`surfaces::Surface`].
#![warn(missing_docs)]

pub(crate) mod tol {
    /// Angular tolerance for parallelism/orthogonality checks (radians).
    pub const ANGULAR: f64 = crate::Tolerance::DEFAULT.angular;
    /// Distance below which two points are considered coincident.
    pub const CONFUSION: f64 = crate::Tolerance::DEFAULT.confusion;
    /// Parametric-space tolerance.
    pub const P_CONFUSION: f64 = crate::Tolerance::DEFAULT.parametric;
}

/// Axis-aligned bounding boxes of curves and surfaces.
pub mod bounding_box;
/// Internal analytic curve evaluation math.
pub(crate) mod curve_math;
/// Public curve types: lines, conics, and B-spline curves.
pub mod curves;
/// Discretization of curves into points joined by straight segments, with
/// proven chord and angle deviations.
pub mod discretization;
/// Axis and frame placement types.
pub mod frame;
/// Exact intersection result types.
pub mod intersect;
/// Internal numerical toolbox (polynomial roots, Newton, Brent).
pub(crate) mod math;
/// Points in 2D and 3D space.
pub mod point;
/// Point-projection result types.
pub mod projection;
/// Internal analytic surface evaluation math.
pub(crate) mod surface_math;
/// Public surface types: planes, cylinders, cones, spheres, tori, and
/// B-spline surfaces.
pub mod surfaces;
/// Distance and angle tolerances shared by all geometric queries.
pub mod tolerance;
/// Rigid and affine transformations.
pub mod transform;
/// Vectors in 2D and 3D space.
pub mod vector;

pub use bounding_box::{Bound3D, BoundingBox3D};
pub use curves::{
    BSplineConstructionError, BSplineConversionError, BSplineCurve3D, BSplineSplitError, Circle2D,
    Circle3D, CircleConstructionError, Curve2D, Curve3D, DegreeElevationError, ElementaryCurve,
    Ellipse3D, EllipseConstructionError, Hyperbola3D, HyperbolaConstructionError,
    KnotInsertionError, KnotRemovalError, Line2D, Line3D, LineConstructionError, Parabola3D,
    ParabolaConstructionError, ParametricCurve2D, ParametricCurve3D, ParametrizeError,
    PointExtrema2D, PointExtrema3D, PointParametrization, TrimmedCurve2D, TrimmedCurve3D,
    TrimmedCurveConstructionError, parametrize_numeric,
};
pub use discretization::{
    Discretization2D, Discretization3D, DiscretizationCriteria, DiscretizationCriteriaError,
    Discretize2D, Discretize3D,
};
pub use frame::{Axis2D, Axis3D, Frame2D, Frame3D, FrameConstructionError};
pub use intersect::intersect_curve_surface;
pub use intersect::marching_intersection;
pub use intersect::{
    CircleCircle2DIntersection, CircleCircle3DIntersection, CircleSurfaceIntersection,
    ConeConeIntersection, ConeCylinderIntersection, ConicConicIntersection, ConicSurfaceHit,
    ConicSurfaceIntersection, CurveCurveExtremum, CurveCurveHit, CurveSurfaceHit,
    CylinderCylinderIntersection, LatitudeRing, LineCircle2DIntersection, LineCircle3DIntersection,
    LineConic3DIntersection, LineConicHit, LineLine2DIntersection, LineLine3DIntersection,
    LinePlaneIntersection, LineQuadricIntersection, LineTorusHit, LineTorusIntersection,
    PlaneConeIntersection, PlaneCylinderIntersection, PlanePlaneIntersection,
    PlaneSphereIntersection, SphereConeIntersection, SphereCylinderIntersection,
    SphereSphereIntersection, TorusConeIntersection, TorusCylinderIntersection,
    TorusPlaneIntersection, TorusSphereIntersection, TorusTorusIntersection,
};
pub use intersect::{
    Completeness, CurveCurveExtrema, CurveCurveHits, CurveCurveRegion, CurveSurfaceHits,
    CurveSurfaceRegion, IncompleteReason,
};
pub use intersect::{
    CurveCurveIntersection, CurveSurfaceIntersection, InlineList, Intersect, SurfaceIntersection,
};
pub use intersect::{curve_curve_extrema, intersect_curve_curve};
pub use point::{Point2D, Point3D};
pub use projection::{CurveProjection, SurfaceProjection};
pub use surfaces::{
    BSplineSurface, Cone, ConeConstructionError, Cylinder, CylinderConstructionError,
    ElementarySurface, ParametricSurface, Plane, PlaneConstructionError, Sphere,
    SphereConstructionError, Surface, SurfaceCurvatures, SurfaceIsoError, Torus,
    TorusConstructionError,
};
pub use tolerance::Tolerance;
pub use transform::Transform;
pub use vector::{Vector2D, Vector3D};
