//! The results of the routed intersection: one self-describing struct per
//! family of pairs.
//!
//! Each result records its hits and pieces, how they were found
//! ([`IntersectionMethod`]) and whether every hit was found
//! ([`Completeness`]). The fields are private and read through accessors,
//! so the storage can change without breaking callers.

// The crate-private builders are consumed by the routing that follows.
#![allow(dead_code)]

use super::{
    Completeness, CurveCurveHit, CurveCurveHit2D, CurveCurveRegion, CurveSurfaceHit,
    CurveSurfaceRegion,
};
use crate::{Curve3D, Point3D, Tolerance, Vector3D};

/// How an intersection was computed.
///
/// # Examples
///
/// ```
/// use geomcore::IntersectionMethod;
/// assert_ne!(IntersectionMethod::Exact, IntersectionMethod::Numeric);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntersectionMethod {
    /// From a closed form: the hits are exact and complete.
    Exact,
    /// From a numeric search (sampling, subdivision or marching).
    Numeric,
}

/// Two parameter ranges over which two curves coincide.
///
/// A whole-curve coincidence is one overlap over both curves' bounds,
/// which may be infinite.
///
/// # Examples
///
/// ```
/// use geomcore::CurveCurveOverlap;
/// let overlap = CurveCurveOverlap { first_range: (0.0, 1.0), second_range: (3.0, 4.0) };
/// assert_eq!(overlap.second_range.1 - overlap.second_range.0, 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveCurveOverlap {
    /// The parameter range on the first curve.
    pub first_range: (f64, f64),
    /// The parameter range on the second curve.
    pub second_range: (f64, f64),
}

/// One curve along which two surfaces meet.
///
/// # Examples
///
/// ```
/// use geomcore::{Curve3D, IntersectionCurve, Line3D, Point3D, Vector3D};
/// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
/// let piece = IntersectionCurve { curve: Curve3D::Line(line), tangent: false };
/// assert!(!piece.tangent);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct IntersectionCurve {
    /// The curve of contact.
    pub curve: Curve3D,
    /// `true` where the surfaces graze along the curve. A numeric curve
    /// is always `false`, because marching does not trace grazing curves.
    pub tangent: bool,
}

/// Where a numeric surface-surface search may have missed contact.
///
/// # Examples
///
/// ```
/// use geomcore::{Point3D, SurfaceSurfaceRegion};
/// let region = SurfaceSurfaceRegion {
///     first_u_range: (0.0, 1.0),
///     first_v_range: (0.0, 1.0),
///     second_u_range: (0.0, 1.0),
///     second_v_range: (0.0, 1.0),
///     min_corner: Point3D::ORIGIN,
///     max_corner: Point3D::new(1.0, 1.0, 1.0),
/// };
/// assert_eq!(region.first_u_range, (0.0, 1.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceSurfaceRegion {
    /// The first surface's `u` parameter interval.
    pub first_u_range: (f64, f64),
    /// The first surface's `v` parameter interval.
    pub first_v_range: (f64, f64),
    /// The second surface's `u` parameter interval.
    pub second_u_range: (f64, f64),
    /// The second surface's `v` parameter interval.
    pub second_v_range: (f64, f64),
    /// The corner of the box with the smallest coordinates.
    pub min_corner: Point3D,
    /// The corner of the box with the largest coordinates.
    pub max_corner: Point3D,
}

fn sorted_by_first(mut hits: Vec<CurveCurveHit>) -> Vec<CurveCurveHit> {
    hits.sort_by(|a, b| a.first_parameter.total_cmp(&b.first_parameter));
    hits
}

fn swapped_overlaps(overlaps: Vec<CurveCurveOverlap>) -> Vec<CurveCurveOverlap> {
    overlaps
        .into_iter()
        .map(|o| CurveCurveOverlap {
            first_range: o.second_range,
            second_range: o.first_range,
        })
        .collect()
}

/// The multiplicity of a curve-surface hit: 2 when the curve's `tangent`
/// is perpendicular to the surface `normal` within the angular tolerance
/// for the hit's extent (the curve grazes the surface), 1 otherwise. A
/// zero `tangent` or `normal` (a singular point) counts as 1.
pub(crate) fn curve_surface_multiplicity(
    tangent: Vector3D,
    normal: Vector3D,
    point: Point3D,
    tol: Tolerance,
) -> u32 {
    let scale = tangent.magnitude() * normal.magnitude();
    if scale == 0.0 || !scale.is_finite() {
        return 1;
    }
    let extent = point.distance(Point3D::ORIGIN) + 1.0;
    if (tangent.dot(normal) / scale).abs() <= tol.angular_for_extent(extent) {
        2
    } else {
        1
    }
}

/// The intersection of two 3D curves.
///
/// Read it through [`hits`](Self::hits), [`overlaps`](Self::overlaps),
/// [`method`](Self::method), [`completeness`](Self::completeness),
/// [`is_complete`](Self::is_complete) and [`is_empty`](Self::is_empty).
/// `Intersect::intersect` returns it; an exact result is always
/// complete, so an empty one proves the curves do not meet.
#[derive(Debug, Clone, PartialEq)]
pub struct CurveCurveIntersection {
    hits: Vec<CurveCurveHit>,
    overlaps: Vec<CurveCurveOverlap>,
    method: IntersectionMethod,
    completeness: Completeness<CurveCurveRegion>,
}

impl CurveCurveIntersection {
    /// An exact, complete result; the hits are sorted by first parameter.
    pub(crate) fn exact(hits: Vec<CurveCurveHit>, overlaps: Vec<CurveCurveOverlap>) -> Self {
        CurveCurveIntersection {
            hits: sorted_by_first(hits),
            overlaps,
            method: IntersectionMethod::Exact,
            completeness: Completeness::Complete,
        }
    }

    /// A numeric result with the status of its search.
    pub(crate) fn numeric(
        hits: Vec<CurveCurveHit>,
        completeness: Completeness<CurveCurveRegion>,
    ) -> Self {
        CurveCurveIntersection {
            hits: sorted_by_first(hits),
            overlaps: Vec::new(),
            method: IntersectionMethod::Numeric,
            completeness,
        }
    }

    /// The same result read from the other curve's side: `first_*` and
    /// `second_*` exchanged, the hits re-sorted by the new first parameter.
    pub(crate) fn swapped(self) -> Self {
        let hits = self
            .hits
            .into_iter()
            .map(|h| CurveCurveHit {
                first_parameter: h.second_parameter,
                second_parameter: h.first_parameter,
                ..h
            })
            .collect();
        CurveCurveIntersection {
            hits: sorted_by_first(hits),
            overlaps: swapped_overlaps(self.overlaps),
            ..self
        }
    }

    /// The isolated hits, sorted by first parameter.
    pub fn hits(&self) -> &[CurveCurveHit] {
        &self.hits
    }

    /// The ranges over which the curves coincide.
    pub fn overlaps(&self) -> &[CurveCurveOverlap] {
        &self.overlaps
    }

    /// Whether the result came from a closed form or a numeric search.
    pub fn method(&self) -> IntersectionMethod {
        self.method
    }

    /// Whether every hit was found, and where hits may be missing if not.
    pub fn completeness(&self) -> &Completeness<CurveCurveRegion> {
        &self.completeness
    }

    /// Whether every hit was found.
    pub fn is_complete(&self) -> bool {
        self.completeness.is_complete()
    }

    /// Whether no hit and no overlap was found. With an incomplete result
    /// this means "found nothing", not "nothing exists".
    pub fn is_empty(&self) -> bool {
        self.hits.is_empty() && self.overlaps.is_empty()
    }
}

/// The intersection of two 2D curves. Always exact and complete.
///
/// Read it through [`hits`](Self::hits), [`overlaps`](Self::overlaps),
/// [`method`](Self::method) (always [`IntersectionMethod::Exact`]),
/// [`completeness`](Self::completeness) (always complete),
/// [`is_complete`](Self::is_complete) and [`is_empty`](Self::is_empty).
/// `Intersect::intersect` returns it.
#[derive(Debug, Clone, PartialEq)]
pub struct CurveCurveIntersection2D {
    hits: Vec<CurveCurveHit2D>,
    overlaps: Vec<CurveCurveOverlap>,
    completeness: Completeness<CurveCurveRegion>,
}

impl CurveCurveIntersection2D {
    /// An exact, complete result; the hits are sorted by first parameter.
    pub(crate) fn exact(mut hits: Vec<CurveCurveHit2D>, overlaps: Vec<CurveCurveOverlap>) -> Self {
        hits.sort_by(|a, b| a.first_parameter.total_cmp(&b.first_parameter));
        CurveCurveIntersection2D {
            hits,
            overlaps,
            completeness: Completeness::Complete,
        }
    }

    /// The same result read from the other curve's side.
    pub(crate) fn swapped(self) -> Self {
        let hits = self
            .hits
            .into_iter()
            .map(|h| CurveCurveHit2D {
                first_parameter: h.second_parameter,
                second_parameter: h.first_parameter,
                ..h
            })
            .collect();
        CurveCurveIntersection2D::exact(hits, swapped_overlaps(self.overlaps))
    }

    /// The isolated hits, sorted by first parameter.
    pub fn hits(&self) -> &[CurveCurveHit2D] {
        &self.hits
    }

    /// The ranges over which the curves coincide.
    pub fn overlaps(&self) -> &[CurveCurveOverlap] {
        &self.overlaps
    }

    /// Always [`IntersectionMethod::Exact`].
    pub fn method(&self) -> IntersectionMethod {
        IntersectionMethod::Exact
    }

    /// Always [`Completeness::Complete`].
    pub fn completeness(&self) -> &Completeness<CurveCurveRegion> {
        &self.completeness
    }

    /// Always `true`.
    pub fn is_complete(&self) -> bool {
        self.completeness.is_complete()
    }

    /// Whether no hit and no overlap was found.
    pub fn is_empty(&self) -> bool {
        self.hits.is_empty() && self.overlaps.is_empty()
    }
}

/// The intersection of a curve and a surface.
///
/// Read it through [`hits`](Self::hits), [`overlaps`](Self::overlaps) (the
/// curve's parameter ranges that lie on the surface),
/// [`method`](Self::method), [`completeness`](Self::completeness),
/// [`is_complete`](Self::is_complete) and [`is_empty`](Self::is_empty).
/// `Intersect::intersect` returns it.
#[derive(Debug, Clone, PartialEq)]
pub struct CurveSurfaceIntersection {
    hits: Vec<CurveSurfaceHit>,
    overlaps: Vec<(f64, f64)>,
    method: IntersectionMethod,
    completeness: Completeness<CurveSurfaceRegion>,
}

impl CurveSurfaceIntersection {
    /// An exact, complete result; the hits are sorted by curve parameter.
    pub(crate) fn exact(hits: Vec<CurveSurfaceHit>, overlaps: Vec<(f64, f64)>) -> Self {
        CurveSurfaceIntersection {
            hits: sorted_by_curve_parameter(hits),
            overlaps,
            method: IntersectionMethod::Exact,
            completeness: Completeness::Complete,
        }
    }

    /// A numeric result with the status of its search.
    pub(crate) fn numeric(
        hits: Vec<CurveSurfaceHit>,
        completeness: Completeness<CurveSurfaceRegion>,
    ) -> Self {
        CurveSurfaceIntersection {
            hits: sorted_by_curve_parameter(hits),
            overlaps: Vec::new(),
            method: IntersectionMethod::Numeric,
            completeness,
        }
    }

    /// The isolated hits, sorted by curve parameter.
    pub fn hits(&self) -> &[CurveSurfaceHit] {
        &self.hits
    }

    /// The curve's parameter ranges that lie on the surface.
    pub fn overlaps(&self) -> &[(f64, f64)] {
        &self.overlaps
    }

    /// Whether the result came from a closed form or a numeric search.
    pub fn method(&self) -> IntersectionMethod {
        self.method
    }

    /// Whether every hit was found, and where hits may be missing if not.
    pub fn completeness(&self) -> &Completeness<CurveSurfaceRegion> {
        &self.completeness
    }

    /// Whether every hit was found.
    pub fn is_complete(&self) -> bool {
        self.completeness.is_complete()
    }

    /// Whether no hit and no overlap was found. With an incomplete result
    /// this means "found nothing", not "nothing exists".
    pub fn is_empty(&self) -> bool {
        self.hits.is_empty() && self.overlaps.is_empty()
    }
}

fn sorted_by_curve_parameter(mut hits: Vec<CurveSurfaceHit>) -> Vec<CurveSurfaceHit> {
    hits.sort_by(|a, b| a.curve_parameter.total_cmp(&b.curve_parameter));
    hits
}

/// The intersection of two surfaces.
///
/// Read it through [`curves`](Self::curves),
/// [`contact_points`](Self::contact_points),
/// [`is_coincident`](Self::is_coincident), [`method`](Self::method),
/// [`completeness`](Self::completeness),
/// [`is_complete`](Self::is_complete) and [`is_empty`](Self::is_empty).
/// `Intersect::intersect` returns it.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceSurfaceIntersection {
    curves: Vec<IntersectionCurve>,
    contact_points: Vec<Point3D>,
    coincident: bool,
    method: IntersectionMethod,
    completeness: Completeness<SurfaceSurfaceRegion>,
}

impl SurfaceSurfaceIntersection {
    /// An exact, complete result.
    pub(crate) fn exact(
        curves: Vec<IntersectionCurve>,
        contact_points: Vec<Point3D>,
        coincident: bool,
    ) -> Self {
        SurfaceSurfaceIntersection {
            curves,
            contact_points,
            coincident,
            method: IntersectionMethod::Exact,
            completeness: Completeness::Complete,
        }
    }

    /// A numeric result with the status of its search.
    pub(crate) fn numeric(
        curves: Vec<IntersectionCurve>,
        completeness: Completeness<SurfaceSurfaceRegion>,
    ) -> Self {
        SurfaceSurfaceIntersection {
            curves,
            contact_points: Vec::new(),
            coincident: false,
            method: IntersectionMethod::Numeric,
            completeness,
        }
    }

    /// The curves along which the surfaces meet.
    pub fn curves(&self) -> &[IntersectionCurve] {
        &self.curves
    }

    /// The isolated points where the surfaces touch.
    pub fn contact_points(&self) -> &[Point3D] {
        &self.contact_points
    }

    /// Whether the surfaces are the same surface.
    pub fn is_coincident(&self) -> bool {
        self.coincident
    }

    /// Whether the result came from a closed form or a numeric search.
    pub fn method(&self) -> IntersectionMethod {
        self.method
    }

    /// Whether every piece was found, and where some may be missing if not.
    pub fn completeness(&self) -> &Completeness<SurfaceSurfaceRegion> {
        &self.completeness
    }

    /// Whether every piece was found.
    pub fn is_complete(&self) -> bool {
        self.completeness.is_complete()
    }

    /// Whether there is no curve, no contact point and no coincidence.
    /// With an incomplete result this means "found nothing", not "nothing
    /// exists".
    pub fn is_empty(&self) -> bool {
        self.curves.is_empty() && self.contact_points.is_empty() && !self.coincident
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Completeness, Point3D};

    fn hit(first: f64, second: f64) -> CurveCurveHit {
        CurveCurveHit {
            first_parameter: first,
            second_parameter: second,
            point: Point3D::ORIGIN,
            multiplicity: 1,
        }
    }

    #[test]
    fn test_exact_curve_curve_is_complete_and_sorted() {
        let r = CurveCurveIntersection::exact(vec![hit(2.0, 0.0), hit(1.0, 5.0)], vec![]);
        assert_eq!(r.method(), IntersectionMethod::Exact);
        assert!(r.is_complete());
        assert_eq!(r.hits()[0].first_parameter, 1.0);
        assert!(!r.is_empty());
    }

    #[test]
    fn test_swapped_exchanges_parameters_and_resorts() {
        let r = CurveCurveIntersection::exact(
            vec![hit(1.0, 5.0), hit(2.0, 0.0)],
            vec![CurveCurveOverlap {
                first_range: (0.0, 1.0),
                second_range: (3.0, 4.0),
            }],
        )
        .swapped();
        assert_eq!(
            (r.hits()[0].first_parameter, r.hits()[0].second_parameter),
            (0.0, 2.0)
        );
        assert_eq!(r.overlaps()[0].first_range, (3.0, 4.0));
    }

    #[test]
    fn test_empty_incomplete_is_empty_but_not_complete() {
        let r = CurveCurveIntersection::numeric(vec![], Completeness::Sampled);
        assert!(r.is_empty());
        assert!(!r.is_complete());
        assert_eq!(r.method(), IntersectionMethod::Numeric);
    }

    #[test]
    fn test_coincident_surfaces_are_not_empty() {
        let r = SurfaceSurfaceIntersection::exact(vec![], vec![], true);
        assert!(r.is_coincident());
        assert!(!r.is_empty());
    }

    #[test]
    fn test_unbounded_overlap_is_not_empty() {
        let r = CurveSurfaceIntersection::exact(vec![], vec![(f64::NEG_INFINITY, f64::INFINITY)]);
        assert!(!r.is_empty());
        assert_eq!(r.overlaps().len(), 1);
    }

    #[test]
    fn test_multiplicity_is_two_when_tangent_is_perpendicular_to_normal() {
        let tol = Tolerance::DEFAULT;
        let p = Point3D::ORIGIN;
        assert_eq!(
            curve_surface_multiplicity(Vector3D::X, Vector3D::Z, p, tol),
            2
        );
        assert_eq!(
            curve_surface_multiplicity(Vector3D::Z, Vector3D::Z, p, tol),
            1
        );
        assert_eq!(
            curve_surface_multiplicity(Vector3D::ZERO, Vector3D::Z, p, tol),
            1
        );
    }

    #[test]
    fn test_two_dimensional_result_is_exact_and_complete() {
        let r = CurveCurveIntersection2D::exact(vec![], vec![]);
        assert_eq!(r.method(), IntersectionMethod::Exact);
        assert!(r.is_complete() && r.is_empty());
    }
}
