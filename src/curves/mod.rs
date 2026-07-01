//! Public curve types: parametric curves in 2D and 3D built on top of the
//! crate's placement types ([`crate::Axis3`]/[`crate::Axis2`],
//! [`crate::Frame3`]/[`crate::Frame2`]).

mod line;
pub use line::{Line2D, Line3D, LineConstructionError};
