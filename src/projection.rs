//! Point-projection result types.
//!
//! [`CurveProjection`] and [`SurfaceProjection`] are returned by the
//! `project_point` / `project_points` methods on curves and surfaces: the
//! closest parameter(s) together with the distance from the query point.
//! Distances at or below the query tolerance are snapped to exactly `0.0`,
//! so `contains(point, tol)` agrees with
//! `project_point(point, tol).distance == 0.0`.

/// Closest point of a query point on a curve: the parameter and the distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveProjection {
    /// Parameter of the closest point.
    pub parameter: f64,
    /// Distance from the query point to the curve (`0.0` when the query
    /// point is within tolerance of the curve).
    pub distance: f64,
}

/// Closest point of a query point on a surface: the parameters and the distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceProjection {
    /// `u` parameter of the closest point.
    pub u: f64,
    /// `v` parameter of the closest point.
    pub v: f64,
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
