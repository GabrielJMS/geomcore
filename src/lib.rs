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
//!   everything else: [`Point3`]/[`Point2`], [`Vector3`]/[`Vector2`],
//!   [`Axis3`]/[`Axis2`], [`Frame3`]/[`Frame2`], and [`Transform`].
//! - [`curves`] holds the 3D and 2D curve types ([`curves::Line3D`],
//!   [`curves::Circle3D`], [`curves::Ellipse3D`], [`curves::Parabola3D`],
//!   [`curves::Hyperbola3D`], [`curves::BSplineCurve3D`],
//!   [`curves::Line2D`], [`curves::Circle2D`]), the [`curves::Curve3D`]/
//!   [`curves::Curve2D`] enum adaptors, and the
//!   [`curves::ParametricCurve3D`]/[`curves::ParametricCurve2D`] traits.
//! - [`surfaces`] holds the five elementary analytic surfaces
//!   ([`surfaces::Plane`], [`surfaces::Cylinder`], [`surfaces::Cone`],
//!   [`surfaces::Sphere`], [`surfaces::Torus`]) plus
//!   [`surfaces::BSplineSurface`], the [`surfaces::Surface`] enum adaptor,
//!   and the [`surfaces::ParametricSurface`] trait.
//!
//! # Quick start
//!
//! ```
//! use geomcore::{Cylinder, Point3, Vector3};
//! use geomcore::curves::Circle3D;
//!
//! // Evaluate a circle at a single parameter, and over a batch of parameters.
//! let circle = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
//! let point = circle.eval_point(std::f64::consts::PI / 4.0);
//! let points = circle.eval_points(&[0.0, 1.0, 2.0]);
//! assert_eq!(points[0], circle.eval_point(0.0));
//!
//! // Curve-on-surface parametrization: this circle is coaxial with the
//! // cylinder, so it has an exact 2D image, a horizontal line in (u, v).
//! let cylinder = Cylinder::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
//! let pcurve = circle.parametrize_on(&cylinder).unwrap();
//! # let _ = point;
//! ```
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
    pub const ANGULAR: f64 = 1e-12;
    /// Distance below which two points are considered coincident.
    pub const CONFUSION: f64 = 1e-7;
    /// Parametric-space tolerance.
    pub const P_CONFUSION: f64 = 1e-9;
}

/// Internal analytic curve evaluation math.
pub(crate) mod curve_math;
/// Public curve types: lines, conics, and B-spline curves.
pub mod curves;
/// Axis and frame placement types.
pub mod frame;
/// Points in 2D and 3D space.
pub mod point;
/// Internal analytic surface evaluation math.
pub(crate) mod surface_math;
/// Public surface types: planes, cylinders, cones, spheres, tori, and
/// B-spline surfaces.
pub mod surfaces;
/// Rigid and affine transformations.
pub mod transform;
/// Vectors in 2D and 3D space.
pub mod vector;

pub use curves::{
    BSplineConstructionError, BSplineCurve3D, Circle2D, Circle3D, CircleConstructionError, Curve2D,
    Curve3D, Ellipse3D, EllipseConstructionError, Hyperbola3D, HyperbolaConstructionError, Line2D,
    Line3D, LineConstructionError, Parabola3D, ParabolaConstructionError, ParametricCurve2D,
    ParametricCurve3D, ParametrizeError,
};
pub use frame::{Axis2, Axis3, Frame2, Frame3, FrameConstructionError};
pub use point::{Point2, Point3};
pub use surfaces::{
    BSplineSurface, Cone, ConeConstructionError, Cylinder, CylinderConstructionError,
    ParametricSurface, Plane, PlaneConstructionError, Sphere, SphereConstructionError, Surface,
    Torus, TorusConstructionError,
};
pub use transform::Transform;
pub use vector::{Vector2, Vector3};
