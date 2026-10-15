//! The results of the routed intersection: one self-describing struct per
//! family of pairs.
//!
//! Each result records its hits and pieces, how they were found
//! ([`IntersectionMethod`]) and whether every hit was found
//! ([`Completeness`]). The fields are private and read through accessors,
//! so the storage can change without breaking callers.

use super::inline_or_heap_list::InlineOrHeapList;
use super::{
    Completeness, CurveCurveHit, CurveCurveHit2D, CurveCurveRegion, CurveSurfaceHit,
    CurveSurfaceRegion,
};
use crate::{Curve3D, Point3D, Tolerance, Vector3D};

/// The hits of a 3D curve-curve result. The inline capacity is the most
/// hits an exact pair yields (two conics meet at most 4 times), so an
/// exact result never allocates; a numeric result keeps its `Vec`.
pub(crate) type CurveCurveHitList = InlineOrHeapList<CurveCurveHit, 4>;

/// The hits of a 2D curve-curve result: lines and circles meet at most
/// twice.
pub(crate) type CurveCurveHitList2D = InlineOrHeapList<CurveCurveHit2D, 2>;

/// The overlaps of a curve-curve result: a coincidence restricted to two
/// parameter ranges gives one piece, or two when it crosses a seam; a rare
/// third piece moves the list to the heap.
pub(crate) type CurveCurveOverlapList = InlineOrHeapList<CurveCurveOverlap, 2>;

/// The hits of a curve-surface result. The inline capacity is the most
/// hits an exact pair yields (a line or conic meets a quadric or torus at
/// most 4 times).
pub(crate) type CurveSurfaceHitList = InlineOrHeapList<CurveSurfaceHit, 4>;

/// The overlaps of a curve-surface result: an exact coincidence is one
/// parameter range.
pub(crate) type CurveSurfaceOverlapList = InlineOrHeapList<(f64, f64), 1>;

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
/// let overlap = CurveCurveOverlap {
///     first_range: (0.0, 1.0),
///     second_range: (3.0, 4.0),
///     same_direction: true,
/// };
/// assert_eq!(overlap.second_range.1 - overlap.second_range.0, 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveCurveOverlap {
    /// The parameter range on the first curve.
    pub first_range: (f64, f64),
    /// The parameter range on the second curve.
    pub second_range: (f64, f64),
    /// Whether both curves run the same way along the overlap. With
    /// opposite directions, `first_range.0` pairs with `second_range.1`.
    pub same_direction: bool,
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

#[inline]
fn sorted_by_first(mut hits: CurveCurveHitList) -> CurveCurveHitList {
    hits.sort_by(|a, b| a.first_parameter.total_cmp(&b.first_parameter));
    hits
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
/// [`Intersect::intersect`](crate::Intersect::intersect) returns it,
/// reading the receiver's parameters first; an exact result is always
/// complete, so an empty one proves the curves do not meet.
///
/// # Examples
///
/// ```
/// use geomcore::{Circle3D, Intersect, IntersectionMethod, Line3D, Point3D, Tolerance, Vector3D};
///
/// // A line through a circle of radius 2 crosses it twice.
/// let line = Line3D::new(Point3D::new(-5.0, 0.0, 0.0), Vector3D::X).unwrap();
/// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
/// let result = line.intersect(&circle, Tolerance::DEFAULT);
/// assert_eq!(result.method(), IntersectionMethod::Exact);
/// assert!(result.is_complete());
/// let on_line: Vec<f64> = result.hits().iter().map(|h| h.first_parameter).collect();
/// assert_eq!(on_line, [3.0, 7.0]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CurveCurveIntersection {
    hits: CurveCurveHitList,
    overlaps: CurveCurveOverlapList,
    method: IntersectionMethod,
    completeness: Completeness<CurveCurveRegion>,
}

impl CurveCurveIntersection {
    /// An exact, complete result; the hits are sorted by first parameter.
    ///
    /// It takes a whole list: the router hands over the exact solver's
    /// own list unchanged when neither curve is trimmed, which measured
    /// faster than pushing the hits one by one.
    pub(crate) fn exact(hits: CurveCurveHitList, overlaps: CurveCurveOverlapList) -> Self {
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
            hits: sorted_by_first(hits.into()),
            overlaps: CurveCurveOverlapList::new(),
            method: IntersectionMethod::Numeric,
            completeness,
        }
    }

    /// The isolated hits, sorted by first parameter.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::TrimmedCurve3D;
    /// use geomcore::{Circle3D, Intersect, Line3D, Point3D, Tolerance, Vector3D};
    ///
    /// // An arc across the seam at angle 0 keeps only the hit at angle 0.
    /// let line = Line3D::new(Point3D::new(-5.0, 0.0, 0.0), Vector3D::X).unwrap();
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let arc = TrimmedCurve3D::new(circle, -1.0, 1.0).unwrap();
    /// let result = arc.intersect(&line, Tolerance::DEFAULT);
    /// let hit = result.hits()[0];
    /// assert_eq!(result.hits().len(), 1);
    /// assert_eq!(hit.second_parameter, 7.0);
    /// assert!(hit.point.distance(Point3D::new(2.0, 0.0, 0.0)) < 1e-12);
    /// ```
    pub fn hits(&self) -> &[CurveCurveHit] {
        &self.hits
    }

    /// The ranges over which the curves coincide, sorted by the start of
    /// the first range. Each range is increasing and lies in its curve's
    /// own range; an overlap crossing a closed curve's seam comes in two
    /// pieces. `same_direction` says whether the curves run the same way
    /// along the overlap; with opposite directions, `first_range.0` pairs
    /// with `second_range.1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::TrimmedCurve3D;
    /// use geomcore::{CurveCurveOverlap, Intersect, Line3D, Point3D, Tolerance, Vector3D};
    ///
    /// // Two segments of one line share x in [1, 2].
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let first = TrimmedCurve3D::new(line, 0.0, 2.0).unwrap();
    /// let second = TrimmedCurve3D::new(line, 1.0, 3.0).unwrap();
    /// let result = first.intersect(&second, Tolerance::DEFAULT);
    /// assert!(result.hits().is_empty());
    /// assert_eq!(
    ///     result.overlaps(),
    ///     &[CurveCurveOverlap {
    ///         first_range: (1.0, 2.0),
    ///         second_range: (1.0, 2.0),
    ///         same_direction: true,
    ///     }]
    /// );
    /// ```
    pub fn overlaps(&self) -> &[CurveCurveOverlap] {
        &self.overlaps
    }

    /// Whether the result came from a closed form or a numeric search.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle3D, Intersect, IntersectionMethod, Point3D, Tolerance, Vector3D};
    ///
    /// // A B-spline has no closed form: the numeric search runs.
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let other = Circle3D::new(Point3D::new(2.0, 0.0, 0.0), Vector3D::Z, 2.0).unwrap();
    /// let result = circle.intersect(&other.to_bspline(), Tolerance::DEFAULT);
    /// assert_eq!(result.method(), IntersectionMethod::Numeric);
    /// assert_eq!(result.hits().len(), 2);
    /// ```
    pub fn method(&self) -> IntersectionMethod {
        self.method
    }

    /// Whether every hit was found, and where hits may be missing if not.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle3D, Completeness, Intersect, Line3D, Point3D, Tolerance, Vector3D};
    ///
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    /// let result = circle.intersect(&line, Tolerance::DEFAULT);
    /// assert_eq!(result.completeness(), &Completeness::Complete);
    /// ```
    pub fn completeness(&self) -> &Completeness<CurveCurveRegion> {
        &self.completeness
    }

    /// Whether every hit was found.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Frame3D, Intersect, Line3D, Point3D, Tolerance, Vector3D};
    ///
    /// let ellipse = Ellipse3D::from_frame(Frame3D::WORLD, 3.0, 1.0).unwrap();
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::Y).unwrap();
    /// assert!(ellipse.intersect(&line, Tolerance::DEFAULT).is_complete());
    /// ```
    pub fn is_complete(&self) -> bool {
        self.completeness.is_complete()
    }

    /// Whether no hit and no overlap was found. With an incomplete result
    /// this means "found nothing", not "nothing exists".
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Intersect, Line3D, Point3D, Tolerance, Vector3D};
    ///
    /// // An exact empty result proves the skew lines never meet.
    /// let first = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let second = Line3D::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Y).unwrap();
    /// let result = first.intersect(&second, Tolerance::DEFAULT);
    /// assert!(result.is_empty() && result.is_complete());
    /// ```
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
///
/// # Examples
///
/// ```
/// use geomcore::{Circle2D, Intersect, IntersectionMethod, Line2D, Point2D, Tolerance, Vector2D};
///
/// // The x axis crosses a circle of radius 2 at x = -2 and x = 2.
/// let line = Line2D::new(Point2D::new(-5.0, 0.0), Vector2D::X).unwrap();
/// let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
/// let result = circle.intersect(&line, Tolerance::DEFAULT);
/// // Hits come sorted by the circle's angle: x = 2 (angle 0) first, then x = -2.
/// let along_line: Vec<f64> = result.hits().iter().map(|hit| hit.second_parameter).collect();
/// assert!((along_line[0] - 7.0).abs() < 1e-9 && (along_line[1] - 3.0).abs() < 1e-9);
/// assert_eq!(result.method(), IntersectionMethod::Exact);
/// assert!(result.is_complete());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CurveCurveIntersection2D {
    hits: CurveCurveHitList2D,
    overlaps: CurveCurveOverlapList,
    completeness: Completeness<CurveCurveRegion>,
}

impl CurveCurveIntersection2D {
    /// An exact, complete result; the hits are sorted by first parameter.
    ///
    /// The hits are pushed into the result itself: the result is about
    /// 200 to 300 bytes, and collecting the hits first, then moving the
    /// list in, measured slower.
    #[inline]
    pub(crate) fn exact(
        hits: impl IntoIterator<Item = CurveCurveHit2D>,
        overlaps: CurveCurveOverlapList,
    ) -> Self {
        let mut result = CurveCurveIntersection2D {
            hits: CurveCurveHitList2D::new(),
            overlaps,
            completeness: Completeness::Complete,
        };
        for hit in hits {
            result.hits.push(hit);
        }
        result
            .hits
            .sort_by(|a, b| a.first_parameter.total_cmp(&b.first_parameter));
        result
    }

    /// The isolated hits, sorted by first parameter. A hit outside a
    /// trimmed curve's range is dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::TrimmedCurve2D;
    /// use geomcore::{Circle2D, Intersect, Line2D, Point2D, Tolerance, Vector2D};
    ///
    /// // The segment covers x in [-5, -1], so only the near side is met.
    /// let line = Line2D::new(Point2D::new(-5.0, 0.0), Vector2D::X).unwrap();
    /// let segment = TrimmedCurve2D::new(line, 0.0, 4.0).unwrap();
    /// let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
    /// let result = segment.intersect(&circle, Tolerance::DEFAULT);
    /// assert_eq!(result.hits().len(), 1);
    /// assert!((result.hits()[0].first_parameter - 3.0).abs() < 1e-9);
    /// ```
    pub fn hits(&self) -> &[CurveCurveHit2D] {
        &self.hits
    }

    /// The ranges over which the curves coincide, with the same rules as
    /// [`CurveCurveIntersection::overlaps`]: sorted by the start of the
    /// first range, each range increasing and inside its curve's own range,
    /// two pieces when an overlap crosses a closed curve's seam, and
    /// `same_direction` telling whether the curves run the same way.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::TrimmedCurve2D;
    /// use geomcore::{CurveCurveOverlap, Intersect, Line2D, Point2D, Tolerance, Vector2D};
    ///
    /// // Two segments of one line share x in [1, 2]; the second runs backwards.
    /// let forward = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// let backward = Line2D::new(Point2D::new(3.0, 0.0), -Vector2D::X).unwrap();
    /// let first = TrimmedCurve2D::new(forward, 0.0, 2.0).unwrap();
    /// let second = TrimmedCurve2D::new(backward, 0.0, 2.0).unwrap();
    /// let result = first.intersect(&second, Tolerance::DEFAULT);
    /// let overlap = result.overlaps()[0];
    /// assert_eq!(overlap.first_range, (1.0, 2.0));
    /// assert_eq!(overlap.second_range, (1.0, 2.0));
    /// assert!(!overlap.same_direction);
    /// ```
    pub fn overlaps(&self) -> &[CurveCurveOverlap] {
        &self.overlaps
    }

    /// Always [`IntersectionMethod::Exact`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Intersect, IntersectionMethod, Line2D, Point2D, Tolerance, Vector2D};
    /// let a = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// let b = Line2D::new(Point2D::ORIGIN, Vector2D::Y).unwrap();
    /// assert_eq!(a.intersect(&b, Tolerance::DEFAULT).method(), IntersectionMethod::Exact);
    /// ```
    pub fn method(&self) -> IntersectionMethod {
        IntersectionMethod::Exact
    }

    /// Always [`Completeness::Complete`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Completeness, Intersect, Line2D, Point2D, Tolerance, Vector2D};
    /// let a = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// let result = a.intersect(&a, Tolerance::DEFAULT);
    /// assert_eq!(result.completeness(), &Completeness::Complete);
    /// ```
    pub fn completeness(&self) -> &Completeness<CurveCurveRegion> {
        &self.completeness
    }

    /// Always `true`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle2D, Intersect, Point2D, Tolerance};
    /// let a = Circle2D::new(Point2D::ORIGIN, 1.0).unwrap();
    /// let b = Circle2D::new(Point2D::new(5.0, 0.0), 1.0).unwrap();
    /// assert!(a.intersect(&b, Tolerance::DEFAULT).is_complete());
    /// ```
    pub fn is_complete(&self) -> bool {
        self.completeness.is_complete()
    }

    /// Whether no hit and no overlap was found.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Intersect, Line2D, Point2D, Tolerance, Vector2D};
    /// let a = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// let b = Line2D::new(Point2D::new(0.0, 1.0), Vector2D::X).unwrap();
    /// assert!(a.intersect(&b, Tolerance::DEFAULT).is_empty());
    /// ```
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
/// [`Intersect::intersect`](crate::Intersect::intersect) returns it, with
/// the curve and the surface in either order.
///
/// # Examples
///
/// ```
/// use geomcore::{Cylinder, Intersect, IntersectionMethod, Line3D, Point3D, Tolerance, Vector3D};
///
/// // A line across a unit cylinder enters and leaves it.
/// let line = Line3D::new(Point3D::new(-5.0, 0.0, 0.0), Vector3D::X).unwrap();
/// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
/// let result = line.intersect(&cylinder, Tolerance::DEFAULT);
/// assert_eq!(result.method(), IntersectionMethod::Exact);
/// assert!(result.is_complete());
/// let parameters: Vec<f64> = result.hits().iter().map(|h| h.curve_parameter).collect();
/// assert_eq!(parameters, [4.0, 6.0]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CurveSurfaceIntersection {
    hits: CurveSurfaceHitList,
    overlaps: CurveSurfaceOverlapList,
    method: IntersectionMethod,
    completeness: Completeness<CurveSurfaceRegion>,
}

impl CurveSurfaceIntersection {
    /// An exact, complete result; the hits are sorted by curve parameter.
    ///
    /// Like [`CurveCurveIntersection2D::exact`], it pushes the hits into
    /// the result itself.
    #[inline]
    pub(crate) fn exact(
        hits: impl IntoIterator<Item = CurveSurfaceHit>,
        overlaps: CurveSurfaceOverlapList,
    ) -> Self {
        let mut result = CurveSurfaceIntersection {
            hits: CurveSurfaceHitList::new(),
            overlaps,
            method: IntersectionMethod::Exact,
            completeness: Completeness::Complete,
        };
        for hit in hits {
            result.hits.push(hit);
        }
        result
            .hits
            .sort_by(|a, b| a.curve_parameter.total_cmp(&b.curve_parameter));
        result
    }

    /// A numeric result with the status of its search.
    pub(crate) fn numeric(
        hits: Vec<CurveSurfaceHit>,
        completeness: Completeness<CurveSurfaceRegion>,
    ) -> Self {
        CurveSurfaceIntersection {
            hits: sorted_by_curve_parameter(hits.into()),
            overlaps: CurveSurfaceOverlapList::new(),
            method: IntersectionMethod::Numeric,
            completeness,
        }
    }

    /// The isolated hits, sorted by curve parameter.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Intersect, Line3D, Plane, Point3D, Tolerance, Vector3D};
    ///
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let plane = Plane::new(Point3D::new(0.0, 0.0, 2.0), Vector3D::Z).unwrap();
    /// let result = line.intersect(&plane, Tolerance::DEFAULT);
    /// let hit = result.hits()[0];
    /// assert_eq!(hit.curve_parameter, 2.0);
    /// assert_eq!(hit.point, Point3D::new(0.0, 0.0, 2.0));
    /// assert_eq!(hit.multiplicity, 1);
    /// ```
    pub fn hits(&self) -> &[CurveSurfaceHit] {
        &self.hits
    }

    /// The curve's parameter ranges that lie on the surface.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Intersect, Line3D, Plane, Point3D, Tolerance, Vector3D};
    ///
    /// // A line lying in the plane: the whole, unbounded line is an overlap.
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let result = line.intersect(&plane, Tolerance::DEFAULT);
    /// assert!(result.hits().is_empty());
    /// assert_eq!(result.overlaps(), &[(f64::NEG_INFINITY, f64::INFINITY)]);
    /// ```
    pub fn overlaps(&self) -> &[(f64, f64)] {
        &self.overlaps
    }

    /// Whether the result came from a closed form or a numeric search.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Frame3D, Intersect, IntersectionMethod, Point3D, Tolerance, Torus, Vector3D};
    ///
    /// // An ellipse and a torus have no closed form: the numeric search runs.
    /// let ellipse = Ellipse3D::from_frame(Frame3D::WORLD, 5.0, 1.0).unwrap();
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 3.0, 1.0).unwrap();
    /// let result = ellipse.intersect(&torus, Tolerance::DEFAULT);
    /// assert_eq!(result.method(), IntersectionMethod::Numeric);
    /// ```
    pub fn method(&self) -> IntersectionMethod {
        self.method
    }

    /// Whether every hit was found, and where hits may be missing if not.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Completeness, Intersect, Line3D, Point3D, Sphere, Tolerance, Vector3D};
    ///
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let sphere = Sphere::new(Point3D::ORIGIN, 1.0).unwrap();
    /// let result = line.intersect(&sphere, Tolerance::DEFAULT);
    /// assert_eq!(result.completeness(), &Completeness::Complete);
    /// ```
    pub fn completeness(&self) -> &Completeness<CurveSurfaceRegion> {
        &self.completeness
    }

    /// Whether every hit was found.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Intersect, Line3D, Point3D, Sphere, Tolerance, Vector3D};
    ///
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let sphere = Sphere::new(Point3D::ORIGIN, 1.0).unwrap();
    /// assert!(sphere.intersect(&line, Tolerance::DEFAULT).is_complete());
    /// ```
    pub fn is_complete(&self) -> bool {
        self.completeness.is_complete()
    }

    /// Whether no hit and no overlap was found. With an incomplete result
    /// this means "found nothing", not "nothing exists".
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Intersect, Line3D, Point3D, Sphere, Tolerance, Vector3D};
    ///
    /// // An exact empty result proves the line misses the sphere.
    /// let line = Line3D::new(Point3D::new(0.0, 5.0, 0.0), Vector3D::X).unwrap();
    /// let sphere = Sphere::new(Point3D::ORIGIN, 1.0).unwrap();
    /// let result = line.intersect(&sphere, Tolerance::DEFAULT);
    /// assert!(result.is_empty() && result.is_complete());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.hits.is_empty() && self.overlaps.is_empty()
    }
}

#[inline]
fn sorted_by_curve_parameter(mut hits: CurveSurfaceHitList) -> CurveSurfaceHitList {
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
    #[allow(dead_code)] // Consumed by the routing of Task 9.
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
    #[allow(dead_code)] // Consumed by the routing of Task 9.
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
        let r = CurveCurveIntersection::exact(
            [hit(2.0, 0.0), hit(1.0, 5.0)].into_iter().collect(),
            CurveCurveOverlapList::new(),
        );
        assert_eq!(r.method(), IntersectionMethod::Exact);
        assert!(r.is_complete());
        assert_eq!(r.hits()[0].first_parameter, 1.0);
        assert!(!r.is_empty());
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
        let r = CurveSurfaceIntersection::exact(
            [],
            [(f64::NEG_INFINITY, f64::INFINITY)].into_iter().collect(),
        );
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
        let r = CurveCurveIntersection2D::exact([], CurveCurveOverlapList::new());
        assert_eq!(r.method(), IntersectionMethod::Exact);
        assert!(r.is_complete() && r.is_empty());
    }
}
