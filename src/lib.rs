//! A pure, standalone geometric kernel for CAD-grade curves and surfaces.
//!
//! `geomrust` provides parametric evaluation of elementary curves and
//! surfaces (lines, circles, ellipses, parabolas, hyperbolas, B-splines;
//! planes, cylinders, cones, spheres, tori, B-spline surfaces), rigid
//! transformations, and analytic curve-on-surface parametrization.
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
/// Public curve types: lines, and (later) circles, ellipses, and more.
pub mod curves;
/// Axis and frame placement types.
pub mod frame;
/// Points in 2D and 3D space.
pub mod point;
/// Internal analytic surface evaluation math.
pub(crate) mod surface_math;
/// Public surface types: planes, cylinders, cones, spheres, and tori.
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
