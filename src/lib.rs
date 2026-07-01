//! A pure, standalone geometric kernel for CAD-grade curves and surfaces.
//!
//! `geomrust` provides parametric evaluation of elementary curves and
//! surfaces (lines, circles, ellipses, parabolas, hyperbolas, B-splines;
//! planes, cylinders, cones, spheres, tori, B-spline surfaces), rigid
//! transformations, and analytic curve-on-surface parametrization.
#![warn(missing_docs)]

#[allow(dead_code)] // used by upcoming modules
pub(crate) mod tol {
    /// Angular tolerance for parallelism/orthogonality checks (radians).
    pub const ANGULAR: f64 = 1e-12;
    /// Distance below which two points are considered coincident.
    pub const CONFUSION: f64 = 1e-7;
    /// Parametric-space tolerance.
    pub const P_CONFUSION: f64 = 1e-9;
}

/// Points in 2D and 3D space.
pub mod point;
/// Vectors in 2D and 3D space.
pub mod vector;

pub use point::{Point2, Point3};
pub use vector::{Vector2, Vector3};
