//! Public surface types: the five elementary analytic surfaces (plane,
//! cylinder, cone, sphere, torus) built on top of the crate's placement
//! type ([`crate::Frame3`]).

mod cone;
mod cylinder;
mod plane;
mod sphere;
mod torus;

pub use cone::{Cone, ConeConstructionError};
pub use cylinder::{Cylinder, CylinderConstructionError};
pub use plane::{Plane, PlaneConstructionError};
pub use sphere::{Sphere, SphereConstructionError};
pub use torus::{Torus, TorusConstructionError};
