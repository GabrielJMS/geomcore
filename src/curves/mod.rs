//! Public curve types: parametric curves in 2D and 3D built on top of the
//! crate's placement types ([`crate::Axis3D`]/[`crate::Axis2D`],
//! [`crate::Frame3D`]/[`crate::Frame2D`]).

mod bspline;
mod circle;
mod conversion;
mod curve;
mod curve2d;
mod ellipse;
pub(crate) mod fitting;
mod hyperbola;
mod line;
mod parabola;
mod parametrize;
mod trimmed;
pub use bspline::{
    BSplineConstructionError, BSplineCurve3D, BSplineSplitError, DegreeElevationError,
    KnotInsertionError, KnotRemovalError,
};
pub use circle::{Circle2D, Circle3D, CircleConstructionError};
pub use conversion::BSplineConversionError;
pub use curve::{Curve3D, ElementaryCurve, ParametricCurve3D, PointExtrema3D, Side};
pub use curve2d::{Curve2D, ParametricCurve2D, PointExtrema2D};
pub use ellipse::{Ellipse3D, EllipseConstructionError};
pub use fitting::PointParametrization;
pub use hyperbola::{Hyperbola3D, HyperbolaConstructionError};
pub use line::{Line2D, Line3D, LineConstructionError};
pub use parabola::{Parabola3D, ParabolaConstructionError};
pub use parametrize::{ParametrizeError, parametrize_numeric};
pub use trimmed::{TrimmedCurve2D, TrimmedCurve3D, TrimmedCurveConstructionError};
