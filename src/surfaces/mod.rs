//! Public surface types: the five elementary analytic surfaces (plane,
//! cylinder, cone, sphere, torus) plus B-spline surfaces, built on top of
//! the crate's placement type ([`crate::Frame3D`]).

mod bspline;
mod cone;
mod conversion;
mod cylinder;
mod fitting;
mod iso;
mod plane;
mod sphere;
mod surface;
mod torus;

pub use bspline::BSplineSurface;
pub use cone::{Cone, ConeConstructionError};
pub use cylinder::{Cylinder, CylinderConstructionError};
pub use iso::SurfaceIsoError;
pub use plane::{Plane, PlaneConstructionError};
pub use sphere::{Sphere, SphereConstructionError};
pub use surface::{ElementarySurface, ParametricSurface, Surface, SurfaceCurvatures};
pub use torus::{Torus, TorusConstructionError};
