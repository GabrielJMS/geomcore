//! Point-projection result types.
//!
//! [`CurveProjection`], [`CurveProjection2D`] and [`SurfaceProjection`] are
//! returned by the `project_point` / `project_points` methods on curves and
//! surfaces: the closest parameter(s), the closest point itself (the curve or
//! surface evaluated there) and the distance from the query point.
//! Distances at or below the query tolerance are snapped to exactly `0.0`,
//! so `contains(point, tol)` agrees with
//! `project_point(point, tol).distance == 0.0`.

use crate::{Point2D, Point3D};

/// Closest point of a query point on a curve: the parameter, the point and
/// the distance.
///
/// # Examples
///
/// ```
/// use geomcore::{Line3D, Point3D, Tolerance, Vector3D};
///
/// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
/// let projection = line.project_point(Point3D::new(2.0, 1.0, 0.0), Tolerance::DEFAULT);
/// assert_eq!(projection.point, Point3D::new(2.0, 0.0, 0.0));
/// assert_eq!(projection.distance, 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveProjection {
    /// Parameter of the closest point.
    pub parameter: f64,
    /// The closest point: the curve evaluated at `parameter`. For a query
    /// point within tolerance of the curve this is still the point on the
    /// curve, not the query point.
    pub point: Point3D,
    /// Distance from the query point to the curve (`0.0` when the query
    /// point is within tolerance of the curve).
    pub distance: f64,
}

impl CurveProjection {
    /// The projection of a query point with a non-finite coordinate:
    /// parameter, point and distance are all NaN.
    pub(crate) const UNDEFINED: CurveProjection = CurveProjection {
        parameter: f64::NAN,
        point: Point3D::new(f64::NAN, f64::NAN, f64::NAN),
        distance: f64::NAN,
    };
}

/// Closest point of a query point on a 2D curve: the parameter, the point and
/// the distance. The planar counterpart of [`CurveProjection`].
///
/// # Examples
///
/// ```
/// use geomcore::{Line2D, Point2D, Tolerance, Vector2D};
///
/// let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
/// let projection = line.project_point(Point2D::new(2.0, 1.0), Tolerance::DEFAULT);
/// assert_eq!(projection.point, Point2D::new(2.0, 0.0));
/// assert_eq!(projection.distance, 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveProjection2D {
    /// Parameter of the closest point.
    pub parameter: f64,
    /// The closest point: the curve evaluated at `parameter`.
    pub point: Point2D,
    /// Distance from the query point to the curve (`0.0` when the query
    /// point is within tolerance of the curve).
    pub distance: f64,
}

impl CurveProjection2D {
    /// The projection of a query point with a non-finite coordinate:
    /// parameter, point and distance are all NaN.
    pub(crate) const UNDEFINED: CurveProjection2D = CurveProjection2D {
        parameter: f64::NAN,
        point: Point2D::new(f64::NAN, f64::NAN),
        distance: f64::NAN,
    };
}

/// Closest point of a query point on a surface: the parameters, the point and
/// the distance.
///
/// # Examples
///
/// ```
/// use geomcore::{Plane, Point3D, Tolerance, Vector3D};
///
/// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
/// let projection = plane.project_point(Point3D::new(1.0, 2.0, 3.0), Tolerance::DEFAULT);
/// assert_eq!(projection.point, Point3D::new(1.0, 2.0, 0.0));
/// assert_eq!(projection.distance, 3.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceProjection {
    /// `u` parameter of the closest point.
    pub u: f64,
    /// `v` parameter of the closest point.
    pub v: f64,
    /// The closest point: the surface evaluated at `(u, v)`. For a query
    /// point within tolerance of the surface this is still the point on the
    /// surface, not the query point.
    pub point: Point3D,
    /// Distance from the query point to the surface (`0.0` when the query
    /// point is within tolerance of the surface).
    pub distance: f64,
}

/// Snap a projection distance to `0.0` when it is within `confusion`.
///
/// This keeps projection coherent with [`crate::Tolerance`]-based
/// containment: a contained point projects to distance exactly zero.
pub(crate) fn snap_distance(distance: f64, confusion: f64) -> f64 {
    if distance <= confusion { 0.0 } else { distance }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snap_distance() {
        assert_eq!(snap_distance(1e-9, 1e-7), 0.0);
        assert_eq!(snap_distance(1e-7, 1e-7), 0.0);
        assert_eq!(snap_distance(1e-6, 1e-7), 1e-6);
    }
}
