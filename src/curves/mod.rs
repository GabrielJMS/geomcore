//! Public curve types: parametric curves in 2D and 3D built on top of the
//! crate's placement types ([`crate::Axis3`]/[`crate::Axis2`],
//! [`crate::Frame3`]/[`crate::Frame2`]).

mod circle;
mod ellipse;
mod hyperbola;
mod line;
mod parabola;
pub use circle::{Circle2D, Circle3D, CircleConstructionError};
pub use ellipse::{Ellipse3D, EllipseConstructionError};
pub use hyperbola::{Hyperbola3D, HyperbolaConstructionError};
pub use line::{Line2D, Line3D, LineConstructionError};
pub use parabola::{Parabola3D, ParabolaConstructionError};
