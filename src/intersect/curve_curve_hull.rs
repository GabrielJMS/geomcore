//! The results of the numeric curve–curve intersection and extrema
//! ([`CurveCurveHits`], [`CurveCurveExtrema`]), and the intersection and
//! distance minima of two B-spline curves by subdivision of their Bézier
//! pieces (see [`bspline_curve_curve`] and [`bspline_minima`]).

use super::completeness::{Region, box_union, meet, of_unsettled, span};
use super::curve_surface_hull::Range;
use super::pieces::{
    CurvePieces, Pool, apart_along, bounding_box, box_size, cross, curves_meet_at_most_once, dot,
    euclidean, extent, norm, sub, tangent_sign,
};
use super::results::sort_by_known_parameter;
use super::{Completeness, CurveCurveExtremum, CurveCurveHit};
use crate::curve_math::bspline::LARGE;
use crate::curves::BSplineCurve3D;
use crate::math::boxtree::{Aabb, BoxTree, boxes_apart};
use crate::math::{newton_min_1d, newton_min_2d, solve_2x2};
use crate::surface_math::bspline::{Interval, derivative_bounds};
use crate::{Point3D, Tolerance};
use std::ops::Deref;

/// The hits of a numeric curve–curve intersection, ordered by parameter
/// on the first curve, and whether they are all of them (see
/// [`search_curve_curve`](crate::search_curve_curve)).
///
/// It reads as the slice of its hits.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CurveCurveHits {
    hits: Vec<CurveCurveHit>,
    completeness: Completeness<CurveCurveRegion>,
}

impl CurveCurveHits {
    /// `hits`, ordered by first-curve parameter, with their completeness.
    pub(super) fn new(
        hits: Vec<CurveCurveHit>,
        completeness: Completeness<CurveCurveRegion>,
    ) -> Self {
        CurveCurveHits { hits, completeness }
    }

    /// Hits found from samples.
    pub(super) fn sampled(hits: Vec<CurveCurveHit>) -> Self {
        Self::new(hits, Completeness::Sampled)
    }

    /// Adds the `sampled` hits farther than `sqrt(tol.confusion)` from
    /// every hit already found, keeping the order by first-curve
    /// parameter.
    pub(super) fn add_sampled(&mut self, sampled: Vec<CurveCurveHit>, tol: Tolerance) {
        let merge_gap = tol.confusion.sqrt();
        for hit in sampled {
            if self
                .hits
                .iter()
                .all(|h| h.point.distance(hit.point) > merge_gap)
            {
                self.hits.push(hit);
            }
        }
        sort_by_known_parameter(&mut self.hits, |hit| hit.first_parameter);
    }

    /// The hits, ordered by first-curve parameter, without the
    /// completeness.
    pub fn into_hits(self) -> Vec<CurveCurveHit> {
        self.hits
    }

    /// Whether the hits are all of them, and if not, why and where hits
    /// may be missing.
    pub fn completeness(&self) -> &Completeness<CurveCurveRegion> {
        &self.completeness
    }

    /// Whether the hits are all of them ([`Completeness::Complete`]).
    pub fn is_complete(&self) -> bool {
        self.completeness.is_complete()
    }
}

impl Deref for CurveCurveHits {
    type Target = [CurveCurveHit];

    fn deref(&self) -> &[CurveCurveHit] {
        &self.hits
    }
}

impl IntoIterator for CurveCurveHits {
    type Item = CurveCurveHit;
    type IntoIter = std::vec::IntoIter<CurveCurveHit>;

    fn into_iter(self) -> std::vec::IntoIter<CurveCurveHit> {
        self.hits.into_iter()
    }
}

impl<'a> IntoIterator for &'a CurveCurveHits {
    type Item = &'a CurveCurveHit;
    type IntoIter = std::slice::Iter<'a, CurveCurveHit>;

    fn into_iter(self) -> std::slice::Iter<'a, CurveCurveHit> {
        self.hits.iter()
    }
}

/// The local minima of the distance between two curves, closest first,
/// and whether they are all of them (see
/// [`curve_curve_extrema`](crate::curve_curve_extrema)).
///
/// It reads as the slice of its minima.
///
/// # Examples
///
/// ```
/// use geomcore::{BSplineCurve3D, Point3D, Tolerance, curve_curve_extrema};
/// // An arch peaking at (1, 1, 0), under a rail across it at height 1.
/// let arch = vec![
///     Point3D::new(0.0, 0.0, 0.0),
///     Point3D::new(1.0, 2.0, 0.0),
///     Point3D::new(2.0, 0.0, 0.0),
/// ];
/// let arch = BSplineCurve3D::new(2, arch, vec![0.0, 1.0], vec![3, 3], false).unwrap();
/// let rail = vec![Point3D::new(1.0, 0.0, 1.0), Point3D::new(1.0, 2.0, 1.0)];
/// let rail = BSplineCurve3D::new(1, rail, vec![0.0, 1.0], vec![2, 2], false).unwrap();
/// let minima = curve_curve_extrema(&arch, &rail, Tolerance::DEFAULT);
/// // Both are B-splines: the search proves there is no other minimum.
/// assert!(minima.is_complete());
/// assert_eq!(minima.len(), 1);
/// assert!((minima[0].distance - 1.0).abs() < 1e-12);
/// for minimum in &minima {
///     assert!((minimum.first_parameter - 0.5).abs() < 1e-9);
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CurveCurveExtrema {
    minima: Vec<CurveCurveExtremum>,
    completeness: Completeness<CurveCurveRegion>,
}

impl CurveCurveExtrema {
    /// `minima`, put in order of distance, with their completeness.
    pub(super) fn new(
        mut minima: Vec<CurveCurveExtremum>,
        completeness: Completeness<CurveCurveRegion>,
    ) -> Self {
        minima.sort_by(|x, y| x.distance.total_cmp(&y.distance));
        CurveCurveExtrema {
            minima,
            completeness,
        }
    }

    /// Minima found from samples.
    pub(super) fn sampled(minima: Vec<CurveCurveExtremum>) -> Self {
        Self::new(minima, Completeness::Sampled)
    }

    /// The minima, closest first.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D, Tolerance, Vector3D, curve_curve_extrema};
    /// let l1 = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let l2 = Line3D::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Y).unwrap();
    /// let minima = curve_curve_extrema(&l1, &l2, Tolerance::DEFAULT);
    /// assert_eq!(minima.minima()[0].distance, 1.0);
    /// ```
    pub fn minima(&self) -> &[CurveCurveExtremum] {
        &self.minima
    }

    /// The minima, closest first, without the completeness.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D, Tolerance, Vector3D, curve_curve_extrema};
    /// let l1 = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let l2 = Line3D::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Y).unwrap();
    /// let minima: Vec<_> = curve_curve_extrema(&l1, &l2, Tolerance::DEFAULT).into_minima();
    /// assert_eq!(minima.len(), 1);
    /// ```
    pub fn into_minima(self) -> Vec<CurveCurveExtremum> {
        self.minima
    }

    /// Whether the minima are all of them, and if not, why and where
    /// minima may be missing.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Completeness, Line3D, Point3D, Tolerance, Vector3D, curve_curve_extrema};
    /// let l1 = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let l2 = Line3D::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Y).unwrap();
    /// let minima = curve_curve_extrema(&l1, &l2, Tolerance::DEFAULT);
    /// assert_eq!(minima.completeness(), &Completeness::Sampled);
    /// ```
    pub fn completeness(&self) -> &Completeness<CurveCurveRegion> {
        &self.completeness
    }

    /// Whether the minima are all of them ([`Completeness::Complete`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D, Tolerance, Vector3D, curve_curve_extrema};
    /// let l1 = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let l2 = Line3D::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Y).unwrap();
    /// // Only a search with a B-spline on one side proves its minima complete.
    /// assert!(!curve_curve_extrema(&l1, &l2, Tolerance::DEFAULT).is_complete());
    /// ```
    pub fn is_complete(&self) -> bool {
        self.completeness.is_complete()
    }
}

impl Deref for CurveCurveExtrema {
    type Target = [CurveCurveExtremum];

    fn deref(&self) -> &[CurveCurveExtremum] {
        &self.minima
    }
}

impl IntoIterator for CurveCurveExtrema {
    type Item = CurveCurveExtremum;
    type IntoIter = std::vec::IntoIter<CurveCurveExtremum>;

    fn into_iter(self) -> std::vec::IntoIter<CurveCurveExtremum> {
        self.minima.into_iter()
    }
}

impl<'a> IntoIterator for &'a CurveCurveExtrema {
    type Item = &'a CurveCurveExtremum;
    type IntoIter = std::slice::Iter<'a, CurveCurveExtremum>;

    fn into_iter(self) -> std::slice::Iter<'a, CurveCurveExtremum> {
        self.minima.iter()
    }
}

/// Where an incomplete curve–curve intersection may have missed hits, or
/// incomplete curve–curve extrema minima: a parameter interval on each
/// curve, with a box in space holding both curves over their intervals.
/// A missed hit or minimum has its first parameter in `first_range`, its
/// second in `second_range`, and its points in the box.
///
/// # Examples
///
/// ```
/// use geomcore::{BSplineCurve3D, Completeness, Intersect, Point3D, Tolerance};
/// // A parabola touching the segment at (2, 0, 0), at s = 0.5.
/// let poles = vec![
///     Point3D::new(0.0, 1.0, 0.0),
///     Point3D::new(2.0, -1.0, 0.0),
///     Point3D::new(4.0, 1.0, 0.0),
/// ];
/// let parabola = BSplineCurve3D::new(2, poles, vec![0.0, 1.0], vec![3, 3], false).unwrap();
/// let ends = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(4.0, 0.0, 0.0)];
/// let segment = BSplineCurve3D::new(1, ends, vec![0.0, 1.0], vec![2, 2], false).unwrap();
/// let result = parabola.intersect(&segment, Tolerance::DEFAULT);
/// let Completeness::Incomplete { regions, .. } = result.completeness() else {
///     panic!("a tangent contact never separates");
/// };
/// assert_eq!(regions.len(), 1);
/// let (s0, s1) = regions[0].first_range;
/// assert!(s0 <= 0.5 && 0.5 <= s1 && s1 - s0 < 1e-2);
/// assert!(regions[0].min_corner.x <= 2.0 && 2.0 <= regions[0].max_corner.x);
/// // The sampled search, run as well, finds the contact.
/// assert_eq!(result.hits().len(), 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveCurveRegion {
    /// The first curve's parameter interval.
    pub first_range: (f64, f64),
    /// The second curve's parameter interval.
    pub second_range: (f64, f64),
    /// The corner of the box with the smallest coordinates.
    pub min_corner: Point3D,
    /// The corner of the box with the largest coordinates.
    pub max_corner: Point3D,
}

impl Region for CurveCurveRegion {
    fn touches(&self, other: &CurveCurveRegion) -> bool {
        meet(self.first_range, other.first_range) && meet(self.second_range, other.second_range)
    }

    fn union(&self, other: &CurveCurveRegion) -> CurveCurveRegion {
        let (min_corner, max_corner) = box_union(
            (self.min_corner, self.max_corner),
            (other.min_corner, other.max_corner),
        );
        CurveCurveRegion {
            first_range: span(self.first_range, other.first_range),
            second_range: span(self.second_range, other.second_range),
            min_corner,
            max_corner,
        }
    }

    fn start(&self) -> f64 {
        self.first_range.0
    }
}

/// Halvings allowed per initial pair of pieces, pooled over all of them,
/// before a search gives up.
const SPLITS_PER_PAIR: usize = 256;
/// Halvings from a pair of pieces down to one subpiece pair, beyond which
/// the pair counts as never separating.
const MAX_DEPTH: u32 = 64;
/// Gauss-Newton steps from the centre of a subpiece pair.
const NEWTON_STEPS: usize = 32;

/// The meetings of `a` over the range `ra` with `b` over `rb`, as extrema
/// at distances within `tol.confusion`, closest first, complete unless
/// some pair of pieces was left unsettled.
///
/// The curves' knot spans over their ranges are Bézier pieces; the pairs
/// whose control-point boxes come within `tol.confusion` of each other
/// are found through a box hierarchy over the second curve's pieces.
/// Each pair is then tested, and its larger piece halved, until it is
/// settled:
///
/// - dropped once the control points' boxes, or their projections on one
///   of three axes (normal to both chords, and normal to each chord
///   within the plane of both), are `tol.confusion` apart;
/// - once the pieces provably meet at most once (see
///   [`curves_meet_at_most_once`]), solved from the pair's centre by
///   Gauss-Newton steps minimising `|A(s) - B(t)|^2` on the B-splines
///   themselves (3D curves need not meet exactly). A hit within the pair
///   settles it, and so does one within `sqrt(tol.confusion)` of the
///   whole first piece, as the pair's hit would merge into it; any other
///   hit is still kept.
///
/// A tangency or an overlap never separates into such pairs. A pair is
/// left as never separating once both its pieces lie within
/// `tol.confusion` of one line (see [`along_one_line`]), or after
/// [`MAX_DEPTH`] halvings; the search goes on with the other pairs. If
/// the budget of halvings runs out first, the pairs still pending are
/// left unsettled too. Either way, the result is incomplete, with the
/// unsettled pairs as its regions.
pub(super) fn bspline_curve_curve(
    a: &BSplineCurve3D,
    ra: Range,
    b: &BSplineCurve3D,
    rb: Range,
    tol: Tolerance,
) -> (Vec<CurveCurveExtremum>, Completeness<CurveCurveRegion>) {
    search_meetings(a, ra, b, rb, tol, SPLITS_PER_PAIR)
}

/// [`bspline_curve_curve`] with `splits_per_pair` halvings allowed per
/// initial pair of pieces.
fn search_meetings(
    a: &BSplineCurve3D,
    ra: Range,
    b: &BSplineCurve3D,
    rb: Range,
    tol: Tolerance,
    splits_per_pair: usize,
) -> (Vec<CurveCurveExtremum>, Completeness<CurveCurveRegion>) {
    let (pieces_a, pieces_b) = (
        CurvePieces::new(a, (ra.lo, ra.hi)),
        CurvePieces::new(b, (rb.lo, rb.hi)),
    );
    let mut pairs = Pairs::new(
        [pieces_a.width, pieces_b.width],
        pieces_a.len() + pieces_b.len(),
        splits_per_pair,
    );
    let mut points = Vec::new();
    let boxes_b: Vec<Aabb> = (0..pieces_b.len())
        .map(|j| {
            euclidean(pieces_b.piece(j), &mut points);
            bounding_box(&points)
        })
        .collect();
    let tree = BoxTree::new(&boxes_b, 1);
    for i in 0..pieces_a.len() {
        euclidean(pieces_a.piece(i), &mut points);
        tree.overlapping(&bounding_box(&points), tol.confusion, |j| {
            pairs.push(
                [pieces_a.piece(i), pieces_b.piece(j)],
                [pieces_a.ranges[i], pieces_b.ranges[j]],
            );
        });
    }
    let gap = tol.confusion;
    let mut found = Vec::new();
    let unsettled = pairs.search(|pairs, node| {
        let [pa, pb] = &pairs.points;
        if boxes_apart(&bounding_box(pa), &bounding_box(pb), gap) || separated(pa, pb, gap) {
            return Visit::Settled;
        }
        if curves_meet_at_most_once(pa, pb)
            && let Some((hit, settles)) = meeting(a, ra, b, rb, pa, node.ranges, tol)
        {
            found.push(hit);
            if settles {
                return Visit::Settled;
            }
        }
        match along_one_line(pa, pb, gap) {
            Some(axis) if apart_along(axis, pa, pb, gap) => Visit::Settled,
            Some(_) => Visit::Unseparated,
            None => Visit::Split,
        }
    });
    found.sort_by(|p, q| p.distance.total_cmp(&q.distance));
    (found, unsettled.completeness())
}

/// The meeting of the subpieces over `ranges` of a pair that meets at
/// most once, the first one with the Euclidean control points `pa`,
/// solved from the pair's centre (see [`closest_pair`]): the hit, if one
/// lies within `tol.confusion` of both curves, and whether it settles the
/// pair, lying within it or within `sqrt(tol.confusion)` of the whole
/// first piece.
fn meeting(
    a: &BSplineCurve3D,
    ra: Range,
    b: &BSplineCurve3D,
    rb: Range,
    pa: &[[f64; 3]],
    ranges: [(f64, f64); 2],
    tol: Tolerance,
) -> Option<(CurveCurveExtremum, bool)> {
    let [(s0, s1), (t0, t1)] = ranges;
    let x = closest_pair(a, b, [0.5 * (s0 + s1), 0.5 * (t0 + t1)])?;
    let (s, t) = (ra.admit(x[0])?, rb.admit(x[1])?);
    let point = a.eval_point(s);
    let distance = point.distance(b.eval_point(t));
    if distance > tol.confusion {
        return None;
    }
    let within = |x: f64, (lo, hi): (f64, f64), range: Range| {
        lo - range.slack() <= x && x <= hi + range.slack()
    };
    let inside = within(x[0], (s0, s1), ra) && within(x[1], (t0, t1), rb);
    let merge_gap = tol.confusion.sqrt();
    let settles = inside
        || pa
            .iter()
            .all(|p| norm(sub(*p, [point.x, point.y, point.z])) <= merge_gap);
    let hit = CurveCurveExtremum {
        first_parameter: s,
        second_parameter: t,
        distance,
    };
    Some((hit, settles))
}

/// Whether the hulls of the curve pieces with the Euclidean control
/// points `a` and `b` are more than `gap` apart along the normal to both
/// chords or the normal to either chord in their plane: the axes that
/// part skew pieces, and pieces side by side in a plane.
fn separated(a: &[[f64; 3]], b: &[[f64; 3]], gap: f64) -> bool {
    let chord = |p: &[[f64; 3]]| sub(p[p.len() - 1], p[0]);
    let (ca, cb) = (chord(a), chord(b));
    let mut normal = cross(ca, cb);
    if normal == [0.0; 3] {
        normal = cross(ca, sub(b[0], a[0]));
    }
    [normal, cross(ca, normal), cross(cb, normal)]
        .into_iter()
        .any(|axis| apart_along(axis, a, b, gap))
}

/// The chord of the longer of the curve pieces with the Euclidean control
/// points `a` and `b`, if the line along it holds every control point of
/// both within `gap`: no halving of the pair can then be told apart
/// across that line, only along it.
fn along_one_line(a: &[[f64; 3]], b: &[[f64; 3]], gap: f64) -> Option<[f64; 3]> {
    let chord = |p: &[[f64; 3]]| sub(p[p.len() - 1], p[0]);
    let (ca, cb) = (chord(a), chord(b));
    let (origin, axis) = if dot(ca, ca) >= dot(cb, cb) {
        (a[0], ca)
    } else {
        (b[0], cb)
    };
    let length = norm(axis);
    if !(length > 0.0 && length.is_finite()) {
        return None;
    }
    let gap = gap * length;
    a.iter()
        .chain(b)
        .all(|p| norm(cross(sub(*p, origin), axis)) <= gap)
        .then_some(axis)
}

/// Gauss-Newton minimisation of `|A(s) - B(t)|^2` from `x`, each step
/// halved until the distance decreases: the closest pair it reaches, or
/// `None` when the tangents are parallel or the steps do not settle.
fn closest_pair(a: &BSplineCurve3D, b: &BSplineCurve3D, mut x: [f64; 2]) -> Option<[f64; 2]> {
    let eval = |x: [f64; 2]| {
        let (p, dp) = a.eval_d1(x[0]);
        let (q, dq) = b.eval_d1(x[1]);
        (p - q, dp, dq)
    };
    let (mut f, mut da, mut db) = eval(x);
    for _ in 0..NEWTON_STEPS {
        if f.square_magnitude() == 0.0 {
            return Some(x);
        }
        let cross = -da.dot(db);
        let normal = [[da.dot(da), cross], [cross, db.dot(db)]];
        let step = solve_2x2(normal, [-da.dot(f), db.dot(f)], 1e-14)?;
        let mut scale = 1.0;
        loop {
            let trial = [x[0] + scale * step[0], x[1] + scale * step[1]];
            let (ft, dat, dbt) = eval(trial);
            if ft.square_magnitude() < f.square_magnitude() {
                (x, f, da, db) = (trial, ft, dat, dbt);
                break;
            }
            scale *= 0.5;
            if scale < 1e-3 {
                return Some(x);
            }
        }
        let moved = (scale * step[0]).abs() + (scale * step[1]).abs();
        if moved <= 1e-14 * (2.0 + x[0].abs() + x[1].abs()) {
            return Some(x);
        }
    }
    None
}

/// Seeds for the local minima of the distance between `a` over the range
/// `ra` and `b` over `rb`: parameter pairs at or next to every minimum,
/// for the caller to refine and check on the whole parameter box,
/// complete unless some pair of pieces was left unsettled (where minima
/// may then lack a seed).
///
/// A minimum inside the box is a stationary point of `f = |A(s) -
/// B(t)|^2 / 2` with a positive semidefinite Hessian. Every pair of the
/// curves' Bézier pieces is tested, and its larger piece halved, until it
/// is settled:
///
/// - dropped when `f_s` or `f_t` keeps one sign over it (see
///   [`tangent_sign`]), or its Hessian, enclosed by interval arithmetic
///   (see [`derivative_bounds`]), is nowhere positive semidefinite (a
///   saddle or a maximum);
/// - once its Hessian is positive definite throughout, `f` is strictly
///   convex on it, so it holds at most one stationary point, reached by
///   Newton's method bounded to the pair.
///
/// A minimum on an end of a (non-periodic) range is a minimum of the
/// distance from that end's point to the other curve, found the same way
/// along the other curve alone; the corners of the box are seeds too.
/// Valleys of equal distance (overlaps, concentric arcs) and tangencies
/// never become convex. A pair whose pieces lie within `tol.confusion`
/// of one line (a contact, see [`along_one_line`]) is left as never
/// separating, as is one after [`MAX_DEPTH`] halvings; a valley at a
/// distance usually exhausts the budget first.
pub(super) fn bspline_minima(
    a: &BSplineCurve3D,
    ra: Range,
    b: &BSplineCurve3D,
    rb: Range,
    tol: Tolerance,
) -> (Vec<[f64; 2]>, Completeness<CurveCurveRegion>) {
    search_minima(a, ra, b, rb, tol, SPLITS_PER_PAIR)
}

/// [`bspline_minima`] with `splits_per_pair` halvings allowed per initial
/// pair of pieces of each search.
fn search_minima(
    a: &BSplineCurve3D,
    ra: Range,
    b: &BSplineCurve3D,
    rb: Range,
    tol: Tolerance,
    splits_per_pair: usize,
) -> (Vec<[f64; 2]>, Completeness<CurveCurveRegion>) {
    let pieces = [
        CurvePieces::new(a, (ra.lo, ra.hi)),
        CurvePieces::new(b, (rb.lo, rb.hi)),
    ];
    let ends = |r: Range| if r.periodic { vec![] } else { vec![r.lo, r.hi] };
    let (ends_a, ends_b) = (ends(ra), ends(rb));
    let mut seeds: Vec<[f64; 2]> = ends_a
        .iter()
        .flat_map(|&s| ends_b.iter().map(move |&t| [s, t]))
        .collect();
    let mut pairs = Pairs::new(
        [pieces[0].width, pieces[1].width],
        pieces[0].len() * pieces[1].len(),
        splits_per_pair,
    );
    for i in 0..pieces[0].len() {
        for j in 0..pieces[1].len() {
            pairs.push(
                [pieces[0].piece(i), pieces[1].piece(j)],
                [pieces[0].ranges[i], pieces[1].ranges[j]],
            );
        }
    }
    let mut unsettled = pairs.search(|pairs, node| {
        let [pa, pb] = &pairs.points;
        if tangent_sign(pa, pb).is_some() || tangent_sign(pb, pa).is_some() {
            return Visit::Settled;
        }
        let (va, da) = curve_bounds(pairs.net(0, node), pa);
        let (vb, db) = curve_bounds(pairs.net(1, node), pb);
        let [mut h11, mut h12, mut h22] = [Interval::ZERO; 3];
        for c in 0..3 {
            let r = va[c].sub(vb[c]);
            h11 = h11.add(da[0][c].square()).add(r.mul(da[1][c]));
            h22 = h22.add(db[0][c].square()).sub(r.mul(db[1][c]));
            h12 = h12.add(da[0][c].mul(db[0][c]));
        }
        let off = h12.square();
        if h11.hi < 0.0 || h22.hi < 0.0 || h11.mul(h22).hi < off.lo {
            return Visit::Settled;
        }
        if !(h11.lo > 0.0 && h22.lo > 0.0 && h11.lo * h22.lo > off.hi) {
            return match along_one_line(pa, pb, tol.confusion) {
                Some(_) => Visit::Unseparated,
                None => Visit::Split,
            };
        }
        let (net_a, net_b) = (pairs.net(0, node), pairs.net(1, node));
        let local = |u: f64, v: f64| {
            let [pa, a1, a2] = bezier_d2(net_a, u);
            let [pb, b1, b2] = bezier_d2(net_b, v);
            let r = sub(pa, pb);
            let cross = -dot(a1, b1);
            (
                (
                    0.5 * dot(r, r),
                    [dot(r, a1), -dot(r, b1)],
                    [
                        [dot(a1, a1) + dot(r, a2), cross],
                        [cross, dot(b1, b1) - dot(r, b2)],
                    ],
                ),
                [norm(a1), norm(b1)],
            )
        };
        let unit = Some((0.0, 1.0));
        let x = newton_min_2d(|u, v| local(u, v).0, [0.5; 2], [unit; 2], 1e-14, 100);
        let ((_, g, _), speed) = local(x[0], x[1]);
        let stationary =
            (0..2).all(|i| (0.0 < x[i] && x[i] < 1.0) || g[i].abs() <= tol.confusion * speed[i]);
        if stationary {
            let [(s0, s1), (t0, t1)] = node.ranges;
            seeds.push([s0 + x[0] * (s1 - s0), t0 + x[1] * (t1 - t0)]);
        }
        Visit::Settled
    });
    // The point at an end, and the direction in which the other curve's
    // point must lie for the distance to rise into the range there.
    let end = |curve: &BSplineCurve3D, u: f64, r: Range| {
        let (p, d) = curve.eval_d1(u);
        let hold = if u == r.lo { d } else { -d };
        ([p.x, p.y, p.z], [hold.x, hold.y, hold.z])
    };
    for &s in &ends_a {
        let (point, hold) = end(a, s, ra);
        let (ts, left) = minima_from_point(point, hold, &pieces[1], tol, splits_per_pair);
        seeds.extend(ts.into_iter().map(|t| [s, t]));
        unsettled.absorb(left, |r| CurveCurveRegion {
            first_range: (s, s),
            ..r
        });
    }
    for &t in &ends_b {
        let (point, hold) = end(b, t, rb);
        let (ss, left) = minima_from_point(point, hold, &pieces[0], tol, splits_per_pair);
        seeds.extend(ss.into_iter().map(|s| [s, t]));
        unsettled.absorb(left, |r| CurveCurveRegion {
            first_range: r.second_range,
            second_range: (t, t),
            ..r
        });
    }
    // A seed on an end the distance falls away from is no minimum there.
    seeds.retain(|x| {
        let ((pa, a1), (pb, b1)) = (a.eval_d1(x[0]), b.eval_d1(x[1]));
        let r = pa - pb;
        let (g, speed) = ([r.dot(a1), -r.dot(b1)], [a1.magnitude(), b1.magnitude()]);
        [ra, rb].into_iter().enumerate().all(|(i, r)| {
            let held =
                !r.periodic && ((x[i] <= r.lo && g[i] > 0.0) || (x[i] >= r.hi && g[i] < 0.0));
            held || g[i].abs() <= tol.confusion * speed[i]
        })
    });
    (seeds, unsettled.completeness())
}

/// The parameters of a curve, cut into `pieces`, at the stationary points
/// of the distance from `point` with a positive definite second
/// derivative (the minima inside the pieces), found as in
/// [`bspline_minima`] along the curve alone, where `(point - C) . hold`
/// is not below `-tol.confusion |hold|` (the pieces beyond are dropped
/// at once), with the pairs of the point and a piece left unsettled.
fn minima_from_point(
    point: [f64; 3],
    hold: [f64; 3],
    pieces: &CurvePieces,
    tol: Tolerance,
    splits_per_pair: usize,
) -> (Vec<f64>, Unsettled) {
    let reach = dot(point, hold) + tol.confusion * norm(hold);
    let mut pairs = Pairs::new([1, pieces.width], pieces.len(), splits_per_pair);
    let fixed = [point[0], point[1], point[2], 1.0];
    for j in 0..pieces.len() {
        pairs.push([&fixed, pieces.piece(j)], [(0.0, 0.0), pieces.ranges[j]]);
    }
    let mut found = Vec::new();
    let unsettled = pairs.search(|pairs, node| {
        let [fixed, pc] = &pairs.points;
        if extent(pc, hold).0 > reach || tangent_sign(pc, fixed).is_some() {
            return Visit::Settled;
        }
        let (value, d) = curve_bounds(pairs.net(1, node), pc);
        let mut second = Interval::ZERO;
        for c in 0..3 {
            let at = Interval {
                lo: point[c],
                hi: point[c],
            };
            second = second
                .add(d[0][c].square())
                .add(value[c].sub(at).mul(d[1][c]));
        }
        if second.hi < 0.0 {
            return Visit::Settled;
        }
        if second.lo <= 0.0 {
            return Visit::Split;
        }
        let net = pairs.net(1, node);
        let local = |u: f64| {
            let [c, d1, d2] = bezier_d2(net, u);
            let r = sub(c, point);
            (
                (0.5 * dot(r, r), dot(r, d1), dot(d1, d1) + dot(r, d2)),
                norm(d1),
            )
        };
        let u = newton_min_1d(|u| local(u).0, 0.5, Some((0.0, 1.0)), 1e-14, 100);
        let ((_, g, _), speed) = local(u);
        if (0.0 < u && u < 1.0) || g.abs() <= tol.confusion * speed {
            let (lo, hi) = node.ranges[1];
            found.push(lo + u * (hi - lo));
        }
        Visit::Settled
    });
    (found, unsettled)
}

/// The point and its first two derivatives at `u` in `[0, 1]` of the
/// Bézier piece with the homogeneous control points `net`, in its own
/// parameter: de Casteljau's algorithm down to the three points whose
/// differences give the homogeneous derivatives, then the quotient rule.
fn bezier_d2(net: &[f64], u: f64) -> [[f64; 3]; 3] {
    let n = net.len() / 4;
    let p = (n - 1) as f64;
    let mut q = [[0.0; 4]; LARGE];
    for (k, x) in net.as_chunks::<4>().0.iter().enumerate() {
        q[k] = *x;
    }
    let lerp = |a: [f64; 4], b: [f64; 4]| std::array::from_fn(|c| a[c] + u * (b[c] - a[c]));
    let kept = n.min(3);
    for r in 0..n - kept {
        for i in 0..n - 1 - r {
            q[i] = lerp(q[i], q[i + 1]);
        }
    }
    let (h, h1, h2): ([f64; 4], [f64; 4], [f64; 4]) = if kept == 3 {
        let (r0, r1) = (lerp(q[0], q[1]), lerp(q[1], q[2]));
        (
            lerp(r0, r1),
            std::array::from_fn(|c| p * (r1[c] - r0[c])),
            std::array::from_fn(|c| p * (p - 1.0) * (q[2][c] - 2.0 * q[1][c] + q[0][c])),
        )
    } else {
        (
            lerp(q[0], q[1]),
            std::array::from_fn(|c| q[1][c] - q[0][c]),
            [0.0; 4],
        )
    };
    let w = h[3];
    let point: [f64; 3] = std::array::from_fn(|c| h[c] / w);
    let first: [f64; 3] = std::array::from_fn(|c| (h1[c] - h1[3] * point[c]) / w);
    let second = std::array::from_fn(|c| (h2[c] - 2.0 * h1[3] * first[c] - h2[3] * point[c]) / w);
    [point, first, second]
}

/// Enclosures, over the Bézier piece with the homogeneous control points
/// `net` and the Euclidean ones `points`, in its own parameter over `[0,
/// 1]`, of each coordinate of its point and of its first and second
/// derivatives.
///
/// A polynomial piece's derivatives lie in the hulls of its hodographs'
/// control points, scaled differences of its own; a rational one's are
/// those of [`derivative_bounds`].
fn curve_bounds(net: &[f64], points: &[[f64; 3]]) -> ([Interval; 3], [[Interval; 3]; 2]) {
    if net.as_chunks::<4>().0.iter().any(|x| x[3] != 1.0) {
        let (value, d) = derivative_bounds(net, points, points.len(), 1, 4, true);
        return (value, [d[0], d[2]]);
    }
    let n = points.len();
    let p = (n - 1) as f64;
    let empty = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    let mut value = [empty; 3];
    let mut first = [empty; 3];
    let mut second = [empty; 3];
    let widen = |i: &mut Interval, x: f64| {
        i.lo = i.lo.min(x);
        i.hi = i.hi.max(x);
    };
    for k in 0..n {
        for c in 0..3 {
            widen(&mut value[c], points[k][c]);
            if k + 1 < n {
                widen(&mut first[c], p * (points[k + 1][c] - points[k][c]));
            }
            if k + 2 < n {
                let bend = points[k + 2][c] - 2.0 * points[k + 1][c] + points[k][c];
                widen(&mut second[c], p * (p - 1.0) * bend);
            }
        }
    }
    let settle = |i: Interval| if i.lo > i.hi { Interval::ZERO } else { i };
    (value, [first.map(settle), second.map(settle)])
}

/// A pair of subpieces awaiting its visit: one per curve, in slot
/// `slots[k]` of pool `k` over the parameters `ranges[k]`, and the
/// halvings from the pair it came from.
struct Node {
    slots: [usize; 2],
    ranges: [(f64, f64); 2],
    depth: u32,
}

/// What a visit makes of a pair of subpieces.
enum Visit {
    /// It holds nothing more to find, or what it holds was found.
    Settled,
    /// Its larger piece is to be halved.
    Split,
    /// It never separates: it is left as a region.
    Unseparated,
}

/// The pairs a search left unsettled, as regions, and whether its budget
/// ran out.
struct Unsettled {
    regions: Vec<CurveCurveRegion>,
    exhausted: bool,
}

impl Unsettled {
    /// Adds the pairs `other` left, each region `placed` in the curves'
    /// parameters.
    fn absorb(&mut self, other: Unsettled, placed: impl Fn(CurveCurveRegion) -> CurveCurveRegion) {
        self.regions.extend(other.regions.into_iter().map(placed));
        self.exhausted |= other.exhausted;
    }

    fn completeness(self) -> Completeness<CurveCurveRegion> {
        of_unsettled(self.regions, self.exhausted)
    }
}

/// The pending pairs of a search over the pieces of two curves (or of a
/// point, a piece of one control point, and a curve).
struct Pairs {
    shapes: [(usize, usize, usize); 2],
    pools: [Pool; 2],
    pending: Vec<Node>,
    /// The Euclidean control points of the pair being visited.
    points: [Vec<[f64; 3]>; 2],
    /// The pairs left as never separating.
    unseparated: Vec<CurveCurveRegion>,
    splits_per_pair: usize,
}

impl Pairs {
    /// No pairs yet, of pieces of `widths` homogeneous control points,
    /// with room for `count` pairs and some halvings of them, and
    /// `splits_per_pair` halvings allowed per pair queued.
    fn new(widths: [usize; 2], count: usize, splits_per_pair: usize) -> Self {
        let room = 4 * count + 16;
        Pairs {
            shapes: widths.map(|w| (w, 1, 4)),
            pools: widths.map(|w| {
                let mut pool = Pool::new(4 * w);
                pool.reserve(room);
                pool
            }),
            pending: Vec::with_capacity(room),
            points: widths.map(Vec::with_capacity),
            unseparated: Vec::new(),
            splits_per_pair,
        }
    }

    /// Queues the pair of pieces with the homogeneous control points
    /// `nets` over `ranges`.
    fn push(&mut self, nets: [&[f64]; 2], ranges: [(f64, f64); 2]) {
        let slots = [0, 1].map(|k| {
            let slot = self.pools[k].take();
            self.pools[k].get_mut(slot).copy_from_slice(nets[k]);
            slot
        });
        self.pending.push(Node {
            slots,
            ranges,
            depth: 0,
        });
    }

    /// The homogeneous control points of `node`'s piece of curve `k`.
    fn net(&self, k: usize, node: &Node) -> &[f64] {
        self.pools[k].get(node.slots[k])
    }

    /// Visits every pending pair, with its control points loaded into
    /// [`Pairs::points`]: `visit` settles it, leaves it as never
    /// separating (so does reaching [`MAX_DEPTH`]), or has the larger of
    /// its pieces halved and both halves queued, until the budget of
    /// halvings runs out. Returns the pairs left unsettled.
    fn search(mut self, mut visit: impl FnMut(&Self, &Node) -> Visit) -> Unsettled {
        let mut budget = self.splits_per_pair * self.pending.len();
        let mut exhausted = false;
        while let Some(node) = self.pending.pop() {
            self.load(&node);
            match visit(&self, &node) {
                Visit::Settled => {}
                Visit::Split if node.depth < MAX_DEPTH => {
                    if budget == 0 {
                        self.pending.push(node);
                        exhausted = true;
                        break;
                    }
                    budget -= 1;
                    let sizes = self.points.each_ref().map(|p| box_size(&bounding_box(p)));
                    self.split(node, usize::from(sizes[1] >= sizes[0]));
                    continue;
                }
                Visit::Split | Visit::Unseparated => {
                    let region = self.region(&node);
                    self.unseparated.push(region);
                }
            }
            for k in 0..2 {
                self.pools[k].release(node.slots[k]);
            }
        }
        let mut regions = std::mem::take(&mut self.unseparated);
        for node in std::mem::take(&mut self.pending) {
            self.load(&node);
            regions.push(self.region(&node));
        }
        Unsettled { regions, exhausted }
    }

    /// Loads the Euclidean control points of `node` into
    /// [`Pairs::points`].
    fn load(&mut self, node: &Node) {
        for k in 0..2 {
            euclidean(self.pools[k].get(node.slots[k]), &mut self.points[k]);
        }
    }

    /// The region of `node`, loaded into [`Pairs::points`].
    fn region(&self, node: &Node) -> CurveCurveRegion {
        let corners = |p: &[[f64; 3]]| {
            let b = bounding_box(p);
            (
                Point3D::new(b[0], b[1], b[2]),
                Point3D::new(b[3], b[4], b[5]),
            )
        };
        let (min_corner, max_corner) =
            box_union(corners(&self.points[0]), corners(&self.points[1]));
        CurveCurveRegion {
            first_range: node.ranges[0],
            second_range: node.ranges[1],
            min_corner,
            max_corner,
        }
    }

    /// Halves the piece of curve `k` of `node` and queues both halves,
    /// each with a copy of the other piece.
    fn split(&mut self, node: Node, k: usize) {
        let other = 1 - k;
        let upper = self.pools[k].halve(node.slots[k], self.shapes[k], true);
        let copy = self.pools[other].take();
        self.pools[other].copy(node.slots[other], copy);
        let (lo, hi) = node.ranges[k];
        let mid = 0.5 * (lo + hi);
        let depth = node.depth + 1;
        let mut slots = node.slots;
        let mut ranges = node.ranges;
        slots[k] = upper;
        slots[other] = copy;
        ranges[k] = (mid, hi);
        self.pending.push(Node {
            slots,
            ranges,
            depth,
        });
        ranges[k] = (lo, mid);
        self.pending.push(Node {
            ranges,
            depth,
            ..node
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point3D;
    use crate::curves::{Curve3D, ParametricCurve3D, TrimmedCurve3D};
    use crate::intersect::curve_curve::{
        ParamDomain, curve_curve_minima, hits_at_minima, refine_minima, sampled_curve_curve,
    };
    use crate::intersect::{IncompleteReason, curve_curve_extrema, search_curve_curve};

    const TOL: Tolerance = Tolerance::DEFAULT;

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn wave(&mut self) -> f64 {
            2.0 * self.next() - 1.0
        }
    }

    /// A cubic through six poles from `from(0)` to `from(1)`, offset at
    /// random by up to `amplitude`; rational with random weights if
    /// `rational`.
    fn cubic(
        rng: &mut Lcg,
        from: impl Fn(f64) -> [f64; 3],
        amplitude: [f64; 3],
        rational: bool,
    ) -> BSplineCurve3D {
        let poles = (0..6)
            .map(|k| {
                let [x, y, z] = from(k as f64 / 5.0);
                Point3D::new(
                    x + amplitude[0] * rng.wave(),
                    y + amplitude[1] * rng.wave(),
                    z + amplitude[2] * rng.wave(),
                )
            })
            .collect();
        let (knots, mults) = (vec![0.0, 1.0, 2.0, 3.0], vec![4, 1, 1, 4]);
        if rational {
            let weights = (0..6).map(|_| 0.4 + 1.6 * rng.next()).collect();
            BSplineCurve3D::new_rational(3, poles, weights, knots, mults, false).unwrap()
        } else {
            BSplineCurve3D::new(3, poles, knots, mults, false).unwrap()
        }
    }

    /// Two cubics in the plane `z = x / 2`, across and up the square `[0,
    /// 5]^2` with waves of `wave`, so that they cross.
    fn crossing(rng: &mut Lcg, rational: bool, wave: f64) -> (BSplineCurve3D, BSplineCurve3D) {
        let tilt = |[x, y, _]: [f64; 3]| [x, y, 0.5 * x];
        let (y0, y1, x0, x1) = (
            5.0 * rng.next(),
            5.0 * rng.next(),
            5.0 * rng.next(),
            5.0 * rng.next(),
        );
        let across = cubic(
            rng,
            |s| [-0.5 + 6.0 * s, y0 + (y1 - y0) * s, 0.0],
            [0.0, wave, 0.0],
            rational,
        );
        let up = cubic(
            rng,
            |s| [x0 + (x1 - x0) * s, -0.5 + 6.0 * s, 0.0],
            [wave, 0.0, 0.0],
            rational,
        );
        let tilted = |c: &BSplineCurve3D| {
            let poles = c.poles().iter().map(|p| {
                let [x, y, z] = tilt([p.x, p.y, p.z]);
                Point3D::new(x, y, z)
            });
            let (knots, mults) = (c.knots().to_vec(), c.multiplicities().to_vec());
            match c.weights() {
                Some(w) => BSplineCurve3D::new_rational(
                    3,
                    poles.collect(),
                    w.to_vec(),
                    knots,
                    mults,
                    false,
                ),
                None => BSplineCurve3D::new(3, poles.collect(), knots, mults, false),
            }
            .unwrap()
        };
        (tilted(&across), tilted(&up))
    }

    /// Two wavy cubics, one about `z = 0` and one about `z = 2`.
    fn skew(rng: &mut Lcg, rational: bool) -> (BSplineCurve3D, BSplineCurve3D) {
        let (y0, y1, x0, x1) = (
            5.0 * rng.next(),
            5.0 * rng.next(),
            5.0 * rng.next(),
            5.0 * rng.next(),
        );
        let below = cubic(
            rng,
            |s| [6.0 * s, y0 + (y1 - y0) * s, 0.0],
            [0.5, 1.0, 0.5],
            rational,
        );
        let above = cubic(
            rng,
            |s| [x0 + (x1 - x0) * s, 6.0 * s, 2.0],
            [1.0, 0.5, 0.5],
            rational,
        );
        (below, above)
    }

    /// The hits of the piece-pair search, which must be complete.
    fn search<A: ParametricCurve3D, B: ParametricCurve3D>(a: &A, b: &B) -> Vec<CurveCurveHit> {
        let (ra, rb) = (ParamDomain::of(a).range(), ParamDomain::of(b).range());
        let (sa, sb) = (a.as_bspline().unwrap(), b.as_bspline().unwrap());
        let (meetings, completeness) = bspline_curve_curve(sa, ra, sb, rb, TOL);
        assert!(completeness.is_complete(), "{completeness:?}");
        hits_at_minima(a, b, meetings, TOL)
    }

    /// The minima through the piece-pair seeds, which must be complete.
    fn minima<A: ParametricCurve3D, B: ParametricCurve3D>(a: &A, b: &B) -> Vec<CurveCurveExtremum> {
        let (da, db) = (ParamDomain::of(a), ParamDomain::of(b));
        let (sa, sb) = (a.as_bspline().unwrap(), b.as_bspline().unwrap());
        let (seeds, completeness) = bspline_minima(sa, da.range(), sb, db.range(), TOL);
        assert!(completeness.is_complete(), "{completeness:?}");
        let mut out = refine_minima(a, da, b, db, seeds, TOL);
        out.sort_by(|x, y| x.distance.total_cmp(&y.distance));
        out
    }

    /// Asserts the search finds the sampled search's hits, with the same
    /// parameters and multiplicities, that every hit lies on both curves
    /// within their ranges (the search may find crossings the samples
    /// miss), and returns them.
    #[track_caller]
    fn assert_hits_match<A: ParametricCurve3D, B: ParametricCurve3D>(
        a: &A,
        b: &B,
    ) -> Vec<CurveCurveHit> {
        let found = search(a, b);
        let ((a0, a1), (b0, b1)) = (a.bounds(), b.bounds());
        for hit in &found {
            assert!((a0..=a1).contains(&hit.first_parameter.unwrap()), "{hit:?}");
            assert!(
                (b0..=b1).contains(&hit.second_parameter.unwrap()),
                "{hit:?}"
            );
            assert!(
                hit.point
                    .distance(a.eval_point(hit.first_parameter.unwrap()))
                    < 1e-12
            );
            assert!(
                hit.point
                    .distance(b.eval_point(hit.second_parameter.unwrap()))
                    <= TOL.confusion
            );
        }
        for pair in found.windows(2) {
            assert!(pair[0].first_parameter < pair[1].first_parameter);
        }
        let sampled = sampled_curve_curve(a, b, TOL);
        for want in &sampled {
            assert!(
                found.iter().any(|hit| hit.point.distance(want.point) < 1e-9
                    && (hit.first_parameter.unwrap() - want.first_parameter.unwrap()).abs() < 1e-9
                    && (hit.second_parameter.unwrap() - want.second_parameter.unwrap()).abs()
                        < 1e-9
                    && hit.multiplicity == want.multiplicity),
                "{want:?} missing from {found:#?}"
            );
        }
        let hits = search_curve_curve(a, b, TOL);
        assert!(hits.is_complete(), "{:?}", hits.completeness());
        assert_eq!(hits.into_hits(), found);
        found
    }

    /// Asserts the search finds the grid's minima, that every minimum it
    /// reports is a local minimum on the parameter box (it may find
    /// minima the grid misses), and returns them.
    #[track_caller]
    fn assert_minima_match<A: ParametricCurve3D, B: ParametricCurve3D>(
        a: &A,
        b: &B,
    ) -> Vec<CurveCurveExtremum> {
        let found = minima(a, b);
        let grid = curve_curve_minima(a, b, TOL, false);
        for want in &grid {
            assert!(
                found
                    .iter()
                    .any(|e| (e.first_parameter - want.first_parameter).abs() < 1e-7
                        && (e.second_parameter - want.second_parameter).abs() < 1e-7
                        && (e.distance - want.distance).abs() < 1e-12),
                "{want:?} missing from {found:#?}"
            );
        }
        let (da, db) = (ParamDomain::of(a), ParamDomain::of(b));
        let ((a0, a1), (b0, b1)) = (a.bounds(), b.bounds());
        for e in &found {
            for (ds, dt) in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)] {
                for h in [-1e-4, 1e-4] {
                    let (s, t) = (e.first_parameter + h * ds, e.second_parameter + h * dt);
                    let s = if da.is_periodic() { s } else { s.clamp(a0, a1) };
                    let t = if db.is_periodic() { t } else { t.clamp(b0, b1) };
                    let d = a.eval_point(s).distance(b.eval_point(t));
                    assert!(
                        d >= e.distance - 1e-12,
                        "{e:?} is no minimum: {d} at ({s}, {t})"
                    );
                }
            }
        }
        let extrema = curve_curve_extrema(a, b, TOL);
        assert!(extrema.is_complete(), "{:?}", extrema.completeness());
        assert_eq!(extrema.into_minima(), found);
        found
    }

    #[test]
    fn test_crossing_polynomial_curves_match_sampled() {
        let mut rng = Lcg(42);
        let mut count = 0;
        for _ in 0..32 {
            let (a, b) = crossing(&mut rng, false, 1.0);
            count += assert_hits_match(&a, &b).len();
            assert_minima_match(&a, &b);
        }
        assert!(count >= 32, "{count}");
    }

    #[test]
    fn test_crossing_rational_curves_match_sampled() {
        let mut rng = Lcg(7);
        let mut count = 0;
        for _ in 0..24 {
            let (a, b) = crossing(&mut rng, true, 1.0);
            count += assert_hits_match(&a, &b).len();
            assert_minima_match(&a, &b);
        }
        assert!(count >= 24, "{count}");
    }

    #[test]
    fn test_skew_curves_minima_match_grid() {
        let mut rng = Lcg(3);
        for k in 0..48 {
            let (a, b) = skew(&mut rng, k % 3 == 2);
            assert!(assert_hits_match(&a, &b).is_empty());
            assert!(!assert_minima_match(&a, &b).is_empty());
        }
    }

    #[test]
    fn test_curves_crossing_many_times_match_sampled() {
        let mut rng = Lcg(11);
        let mut most = 0;
        for k in 0..24 {
            let (a, b) = crossing(&mut rng, k % 2 == 1, 3.0);
            most = most.max(assert_hits_match(&a, &b).len());
            assert_minima_match(&a, &b);
        }
        assert!(most >= 3, "{most}");
    }

    /// A closed cubic loop of radius `radius` about `centre`, its poles
    /// alternately `lift` above and below.
    fn ring(centre: [f64; 3], radius: f64, lift: f64) -> BSplineCurve3D {
        let poles = (0..8)
            .map(|k| {
                let angle = k as f64 * std::f64::consts::TAU / 8.0;
                let z = if k % 2 == 0 { lift } else { -lift };
                Point3D::new(
                    centre[0] + radius * angle.cos(),
                    centre[1] + radius * angle.sin(),
                    centre[2] + z,
                )
            })
            .collect();
        let knots = (0..=8).map(f64::from).collect();
        BSplineCurve3D::new(3, poles, knots, vec![1; 9], true).unwrap()
    }

    #[test]
    fn test_periodic_curves_match_sampled() {
        let loop_a = ring([2.5, 2.5, 0.0], 2.0, 0.0);
        let loop_b = ring([3.5, 2.0, 0.0], 1.5, 0.0);
        let hits = assert_hits_match(&loop_a, &loop_b);
        assert_eq!(hits.len(), 2);
        assert!(
            hits.iter()
                .all(|h| (0.0..8.0).contains(&h.first_parameter.unwrap()))
        );
        assert!(
            hits.iter()
                .all(|h| (0.0..8.0).contains(&h.second_parameter.unwrap()))
        );
        assert_minima_match(&loop_a, &loop_b);
        let mut rng = Lcg(5);
        for _ in 0..8 {
            let (across, _) = crossing(&mut rng, false, 1.0);
            let flat = |c: &BSplineCurve3D| {
                let poles = c
                    .poles()
                    .iter()
                    .map(|p| Point3D::new(p.x, p.y, 0.0))
                    .collect();
                BSplineCurve3D::new(
                    3,
                    poles,
                    c.knots().to_vec(),
                    c.multiplicities().to_vec(),
                    false,
                )
                .unwrap()
            };
            let across = flat(&across);
            assert_hits_match(&loop_a, &across);
            assert_hits_match(&across, &loop_a);
            assert_minima_match(&across, &loop_a);
        }
        // A wobbling loop under a skew cubic: distance minima only.
        let wobbly = ring([2.5, 2.5, 0.0], 2.0, 0.5);
        let (_, above) = skew(&mut rng, false);
        assert!(assert_hits_match(&wobbly, &above).is_empty());
        assert!(!assert_minima_match(&wobbly, &above).is_empty());
        assert_minima_match(&above, &wobbly);
    }

    #[test]
    fn test_trimmed_curves_search_only_their_range() {
        let ring = ring([2.5, 2.5, 0.0], 2.0, 0.0);
        let wavy = BSplineCurve3D::new(
            3,
            (0..8)
                .map(|k| {
                    Point3D::new(
                        -0.5 + 0.85 * k as f64,
                        if k % 2 == 0 { -4.0 } else { 9.0 },
                        0.0,
                    )
                })
                .collect(),
            (0..=5).map(f64::from).collect(),
            vec![4, 1, 1, 1, 1, 4],
            false,
        )
        .unwrap();
        let all = assert_hits_match(&ring, &wavy);
        assert!(all.len() >= 6, "{all:#?}");
        // Across the seam of the ring, and inside the wavy curve.
        let wrapped = |t: f64| if t < 6.5 { t + 8.0 } else { t };
        let expected: Vec<f64> = all
            .iter()
            .map(|h| wrapped(h.first_parameter.unwrap()))
            .filter(|&t| t <= 9.5)
            .collect();
        assert!(expected.iter().any(|&t| t < 8.0) && expected.iter().any(|&t| t > 8.0));
        let borrowed = TrimmedCurve3D::new(&ring, 6.5, 9.5).unwrap();
        let owned = TrimmedCurve3D::new(ring.clone(), 6.5, 9.5).unwrap();
        for hits in [
            assert_hits_match(&borrowed, &wavy),
            assert_hits_match(&owned, &wavy),
        ] {
            assert_eq!(hits.len(), expected.len());
            for hit in &hits {
                assert!(
                    expected
                        .iter()
                        .any(|&t| (t - hit.first_parameter.unwrap()).abs() < 1e-9)
                );
            }
        }
        let (first, last) = (
            all[1]
                .second_parameter
                .unwrap()
                .min(all[2].second_parameter.unwrap()),
            4.2,
        );
        let middle = TrimmedCurve3D::new(&wavy, first, last).unwrap();
        let hits = assert_hits_match(&ring, &middle);
        let inside = all
            .iter()
            .filter(|h| (first - 1e-9..=last + 1e-9).contains(&h.second_parameter.unwrap()))
            .count();
        assert_eq!(hits.len(), inside);
        assert_minima_match(&borrowed, &wavy);
        assert_minima_match(&owned, &middle);
        let mut rng = Lcg(9);
        let (below, above) = skew(&mut rng, true);
        let piece = TrimmedCurve3D::new(&below, 0.4, 2.3).unwrap();
        assert_minima_match(&piece, &TrimmedCurve3D::new(above, 1.2, 2.9).unwrap());
    }

    fn quadratic(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> BSplineCurve3D {
        let poles = [p0, p1, p2].map(|[x, y, z]| Point3D::new(x, y, z)).to_vec();
        BSplineCurve3D::new(2, poles, vec![0.0, 1.0], vec![3, 3], false).unwrap()
    }

    fn segment(from: [f64; 3], to: [f64; 3]) -> BSplineCurve3D {
        let poles = [from, to].map(|[x, y, z]| Point3D::new(x, y, z)).to_vec();
        BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap()
    }

    /// Searches without falling back and checks the hits lie on both.
    fn assert_search_hits(a: &BSplineCurve3D, b: &BSplineCurve3D) -> Vec<CurveCurveHit> {
        let hits = search(a, b);
        for hit in &hits {
            assert!((a.bounds().0..=a.bounds().1).contains(&hit.first_parameter.unwrap()));
            assert!((b.bounds().0..=b.bounds().1).contains(&hit.second_parameter.unwrap()));
            assert!(
                hit.point
                    .distance(b.eval_point(hit.second_parameter.unwrap()))
                    <= TOL.confusion
            );
        }
        assert_eq!(search_curve_curve(a, b, TOL).into_hits(), hits);
        hits
    }

    #[test]
    fn test_close_hits_near_a_tangency_are_separated() {
        let line = segment([0.0, 0.0, 0.0], [4.0, 0.0, 0.0]);
        for depth in [1e-3, 1e-6] {
            // `y = (x - 2)^2 / 4 - depth` on `[0, 4]`.
            let dip = quadratic(
                [0.0, 1.0 - depth, 0.0],
                [2.0, -1.0 - depth, 0.0],
                [4.0, 1.0 - depth, 0.0],
            );
            let hits = assert_search_hits(&dip, &line);
            assert_eq!(hits.len(), 2, "{hits:#?}");
            let half = (4.0 * depth).sqrt();
            assert!((hits[0].point.x - (2.0 - half)).abs() < 1e-9, "{hits:#?}");
            assert!((hits[1].point.x - (2.0 + half)).abs() < 1e-9, "{hits:#?}");
            assert!(hits.iter().all(|h| h.multiplicity == 1));
        }
    }

    /// The reason and regions of an incomplete result.
    #[track_caller]
    fn incomplete(
        completeness: &Completeness<CurveCurveRegion>,
    ) -> (IncompleteReason, &[CurveCurveRegion]) {
        match completeness {
            Completeness::Incomplete { reason, regions } => (*reason, regions),
            other => panic!("expected an incomplete search, got {other:?}"),
        }
    }

    /// Whether `region` holds the parameters `(s, t)` and the points of
    /// `a` at `s` and `b` at `t`.
    fn holds<A: ParametricCurve3D, B: ParametricCurve3D>(
        region: &CurveCurveRegion,
        (a, s): (&A, f64),
        (b, t): (&B, f64),
    ) -> bool {
        let within = |x: f64, (lo, hi): (f64, f64)| lo - 1e-9 <= x && x <= hi + 1e-9;
        let in_box = |p: Point3D| {
            let (lo, hi) = (region.min_corner, region.max_corner);
            within(p.x, (lo.x, hi.x)) && within(p.y, (lo.y, hi.y)) && within(p.z, (lo.z, hi.z))
        };
        within(s, region.first_range)
            && within(t, region.second_range)
            && in_box(a.eval_point(s))
            && in_box(b.eval_point(t))
    }

    /// Asserts every hit lies on both curves within their ranges, and that
    /// the sampled search's hits are among them.
    #[track_caller]
    fn assert_sampled_hits_kept<A: ParametricCurve3D, B: ParametricCurve3D>(
        hits: &CurveCurveHits,
        a: &A,
        b: &B,
    ) {
        let ((a0, a1), (b0, b1)) = (a.bounds(), b.bounds());
        for hit in hits {
            assert!((a0..=a1).contains(&hit.first_parameter.unwrap()), "{hit:?}");
            assert!(
                (b0..=b1).contains(&hit.second_parameter.unwrap()),
                "{hit:?}"
            );
            assert!(
                hit.point
                    .distance(b.eval_point(hit.second_parameter.unwrap()))
                    <= TOL.confusion
            );
        }
        for pair in hits.windows(2) {
            assert!(pair[0].first_parameter <= pair[1].first_parameter);
        }
        for want in sampled_curve_curve(a, b, TOL) {
            let near = |h: &CurveCurveHit| h.point.distance(want.point) <= TOL.confusion.sqrt();
            assert!(hits.iter().any(near), "{want:?} missing from {hits:?}");
        }
    }

    #[test]
    fn test_tangent_contact_is_unseparated_around_it() {
        let line = segment([0.0, 0.0, 0.0], [4.0, 0.0, 0.0]);
        let touching = quadratic([0.0, 1.0, 0.0], [2.0, -1.0, 0.0], [4.0, 1.0, 0.0]);
        let contact = Point3D::new(2.0, 0.0, 0.0);
        assert_eq!(touching.eval_point(0.5), contact);
        for (a, b, at) in [
            (&touching, &line, (0.5, 0.5)),
            (&line, &touching, (0.5, 0.5)),
        ] {
            let hits = search_curve_curve(a, b, TOL);
            let (reason, regions) = incomplete(hits.completeness());
            assert_eq!(reason, IncompleteReason::Unseparated);
            assert_eq!(regions.len(), 1, "{regions:#?}");
            assert!(holds(&regions[0], (a, at.0), (b, at.1)), "{regions:#?}");
            let (s0, s1) = regions[0].first_range;
            let (t0, t1) = regions[0].second_range;
            assert!(s1 - s0 < 1e-2 && t1 - t0 < 1e-2, "{regions:#?}");
            assert_eq!(hits.len(), 1, "{hits:#?}");
            assert!(hits[0].point.distance(contact) < TOL.confusion.sqrt());
            assert_sampled_hits_kept(&hits, a, b);
            // The extrema never settle around the contact either.
            let extrema = curve_curve_extrema(a, b, TOL);
            let (_, regions) = incomplete(extrema.completeness());
            assert!(regions.iter().any(|r| holds(r, (a, at.0), (b, at.1))));
            assert!(extrema[0].distance < TOL.confusion, "{extrema:#?}");
        }
    }

    #[test]
    fn test_straight_overlap_is_unseparated_along_it() {
        // A polyline along a diagonal, and a segment over part of it.
        let along = |x: f64| [x, 0.5 * x, -0.25 * x];
        let poles = (0..5)
            .map(|k| {
                let [x, y, z] = along(k as f64);
                Point3D::new(x, y, z)
            })
            .collect();
        let knots = (0..=4).map(f64::from).collect();
        let polyline = BSplineCurve3D::new(1, poles, knots, vec![2, 1, 1, 1, 2], false).unwrap();
        let part = segment(along(1.5), along(3.5));
        for swapped in [false, true] {
            let hits = if swapped {
                search_curve_curve(&part, &polyline, TOL)
            } else {
                search_curve_curve(&polyline, &part, TOL)
            };
            let (reason, regions) = incomplete(hits.completeness());
            assert_eq!(reason, IncompleteReason::Unseparated);
            assert_eq!(regions.len(), 1, "{regions:#?}");
            for k in 0..=16 {
                let x = 1.5 + 2.0 * k as f64 / 16.0;
                let (s, t) = (x, (x - 1.5) / 2.0);
                let held = if swapped {
                    holds(&regions[0], (&part, t), (&polyline, s))
                } else {
                    holds(&regions[0], (&polyline, s), (&part, t))
                };
                assert!(held, "{x}: {regions:#?}");
            }
            assert!(!hits.is_empty());
            if swapped {
                assert_sampled_hits_kept(&hits, &part, &polyline);
            } else {
                assert_sampled_hits_kept(&hits, &polyline, &part);
            }
        }
    }

    #[test]
    fn test_curved_overlap_is_incomplete_along_it() {
        let mut rng = Lcg(1);
        let (curve, _) = crossing(&mut rng, false, 1.0);
        let part = TrimmedCurve3D::new(&curve, 0.5, 2.0).unwrap();
        let hits = search_curve_curve(&curve, &part, TOL);
        let (_, regions) = incomplete(hits.completeness());
        for k in 0..=16 {
            let s = 0.5 + 1.5 * k as f64 / 16.0;
            assert!(
                regions.iter().any(|r| holds(r, (&curve, s), (&part, s))),
                "{s}: {regions:#?}"
            );
        }
        assert!(!hits.is_empty());
        assert_sampled_hits_kept(&hits, &curve, &part);
        let extrema = curve_curve_extrema(&curve, &part, TOL);
        let (_, regions) = incomplete(extrema.completeness());
        for k in 0..=16 {
            let s = 0.5 + 1.5 * k as f64 / 16.0;
            assert!(regions.iter().any(|r| holds(r, (&curve, s), (&part, s))));
        }
        assert!(extrema[0].distance < TOL.confusion, "{extrema:#?}");
    }

    /// The regions of a search without budget: none if it settled every
    /// pair on its first visit.
    #[track_caller]
    fn exhausted(completeness: &Completeness<CurveCurveRegion>) -> &[CurveCurveRegion] {
        if completeness.is_complete() {
            return &[];
        }
        let (reason, regions) = incomplete(completeness);
        assert_eq!(reason, IncompleteReason::BudgetExhausted);
        regions
    }

    #[test]
    fn test_exhausted_budget_leaves_the_pending_pairs() {
        let mut rng = Lcg(17);
        let mut count = 0;
        for k in 0..8 {
            let (a, b) = crossing(&mut rng, k % 2 == 1, 3.0);
            let (da, db) = (ParamDomain::of(&a), ParamDomain::of(&b));
            let (ra, rb) = (da.range(), db.range());
            let all = search(&a, &b);
            let (meetings, completeness) = search_meetings(&a, ra, &b, rb, TOL, 0);
            let found = hits_at_minima(&a, &b, meetings, TOL);
            let regions = exhausted(&completeness);
            count += usize::from(!regions.is_empty());
            for hit in &all {
                let held = regions.iter().any(|r| {
                    holds(
                        r,
                        (&a, hit.first_parameter.unwrap()),
                        (&b, hit.second_parameter.unwrap()),
                    )
                });
                let near = |h: &CurveCurveHit| h.point.distance(hit.point) < 1e-9;
                assert!(held || found.iter().any(near), "{hit:?}: {regions:#?}");
            }
            for pair in regions.windows(2) {
                assert!(!pair[0].touches(&pair[1]));
            }
            // The same for the distance minima.
            let all = minima(&a, &b);
            let (seeds, completeness) = search_minima(&a, ra, &b, rb, TOL, 0);
            let found = refine_minima(&a, da, &b, db, seeds, TOL);
            let regions = exhausted(&completeness);
            count += usize::from(!regions.is_empty());
            for e in &all {
                let (s, t) = (e.first_parameter, e.second_parameter);
                let held = regions.iter().any(|r| holds(r, (&a, s), (&b, t)));
                let near = |f: &CurveCurveExtremum| {
                    (f.first_parameter - s).abs() < 1e-7 && (f.second_parameter - t).abs() < 1e-7
                };
                assert!(held || found.iter().any(near), "{e:?}: {regions:#?}");
            }
        }
        assert!(count >= 12, "{count}");
    }

    #[test]
    fn test_crossing_and_skew_bsplines_are_complete() {
        let mut rng = Lcg(23);
        for k in 0..8 {
            let (a, b) = crossing(&mut rng, k % 2 == 1, 1.0);
            let hits = search_curve_curve(&a, &b, TOL);
            assert!(hits.is_complete(), "{:?}", hits.completeness());
            assert!(!hits.is_empty());
            let (below, above) = skew(&mut rng, k % 2 == 0);
            let extrema = curve_curve_extrema(&below, &above, TOL);
            assert_eq!(extrema.completeness(), &Completeness::Complete);
            assert!(!extrema.is_empty());
            assert!(search_curve_curve(&below, &above, TOL).is_complete());
        }
    }

    #[test]
    fn test_elementary_pairs_are_sampled_and_mixed_pairs_are_not() {
        let circle = crate::Circle3D::new(Point3D::ORIGIN, crate::Vector3D::Z, 2.0).unwrap();
        let line = crate::Line3D::new(Point3D::ORIGIN, crate::Vector3D::X).unwrap();
        let hits = search_curve_curve(&line, &circle, TOL);
        assert_eq!(hits.completeness(), &Completeness::Sampled);
        assert_eq!(hits.len(), 2);
        assert_eq!(
            curve_curve_extrema(&line, &circle, TOL).completeness(),
            &Completeness::Sampled
        );
        let chord = segment([-3.0, 0.5, 0.0], [3.0, 0.5, 0.0]);
        let hits = search_curve_curve(&chord, &circle, TOL);
        assert_eq!(hits.completeness(), &Completeness::Complete);
        assert_eq!(hits.len(), 2);
        let extrema = curve_curve_extrema(&circle, &chord, TOL);
        assert_eq!(extrema.completeness(), &Completeness::Sampled);
        assert!(!extrema.is_empty());
        let line = crate::Line3D::new(Point3D::new(0.0, 0.5, 1.0), crate::Vector3D::X).unwrap();
        let extrema = curve_curve_extrema(&line, &chord, TOL);
        assert_eq!(extrema.completeness(), &Completeness::Sampled);
        let bend = quadratic([-1.0, 0.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 0.0]);
        let extrema = curve_curve_extrema(&line, &bend, TOL);
        assert_eq!(extrema.completeness(), &Completeness::Sampled);
        let piece = crate::curves::TrimmedCurve3D::new(line, -2.0, 2.0).unwrap();
        let extrema = curve_curve_extrema(&piece, &bend, TOL);
        assert_eq!(extrema.completeness(), &Completeness::Complete);
    }

    #[test]
    fn test_hits_at_curve_ends_and_knots() {
        let bend = quadratic([0.0, 0.0, 0.0], [1.0, 2.0, 0.5], [2.0, 0.0, 1.0]);
        let end = bend.eval_point(1.0);
        let middle = bend.eval_point(0.3);
        let cases = [
            // End to end, end to start, end on the interior.
            (bend.eval_point(1.0), [3.0, 1.0, 0.0], end),
            (
                Point3D::new(-1.0, 1.0, 0.5),
                [0.0, 0.0, 0.0],
                Point3D::ORIGIN,
            ),
            (
                Point3D::new(middle.x, middle.y - 1.0, middle.z),
                [middle.x, middle.y + 1.0, middle.z],
                middle,
            ),
        ]
        .map(|(from, to, at): (Point3D, [f64; 3], Point3D)| {
            (segment([from.x, from.y, from.z], to), at)
        });
        for (line, at) in &cases {
            for hits in [
                assert_search_hits(&bend, line),
                assert_search_hits(line, &bend),
            ] {
                assert_eq!(hits.len(), 1, "{hits:#?}");
                assert!(hits[0].point.distance(*at) < 1e-12, "{hits:#?}");
            }
        }
        // Through a knot of each curve.
        let knotted = |p: [[f64; 3]; 3]| {
            let poles = p.map(|[x, y, z]| Point3D::new(x, y, z)).to_vec();
            BSplineCurve3D::new(1, poles, vec![0.0, 1.0, 2.0], vec![2, 1, 2], false).unwrap()
        };
        let a = knotted([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.5, 0.0]]);
        let b = knotted([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 3.0, 0.0]]);
        let hits = assert_search_hits(&a, &b);
        assert_eq!(hits.len(), 1, "{hits:#?}");
        assert_eq!(
            (hits[0].first_parameter, hits[0].second_parameter),
            (Some(1.0), Some(1.0))
        );
    }

    #[test]
    fn test_enums_and_borrows_take_the_search() {
        let mut rng = Lcg(21);
        let (a, b) = crossing(&mut rng, true, 2.0);
        let (ea, eb) = (Curve3D::BSpline(a.clone()), Curve3D::BSpline(b.clone()));
        let hits = assert_hits_match(&ea, &eb);
        assert_eq!(hits, search_curve_curve(&a, &b, TOL).into_hits());
        assert_eq!(hits, search_curve_curve(&&a, &b, TOL).into_hits());
        assert_eq!(
            curve_curve_extrema(&ea, &eb, TOL).into_minima(),
            assert_minima_match(&a, &b)
        );
    }

    #[test]
    fn test_concentric_arcs_are_incomplete_along_the_valley() {
        // Exact rational quarter circles of radii 2 and 3: a valley.
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let arc = |r: f64| {
            let poles = vec![
                Point3D::new(r, 0.0, 0.0),
                Point3D::new(r, r, 0.0),
                Point3D::new(0.0, r, 0.0),
            ];
            BSplineCurve3D::new_rational(
                2,
                poles,
                vec![1.0, w, 1.0],
                vec![0.0, 1.0],
                vec![3, 3],
                false,
            )
            .unwrap()
        };
        let (a, b) = (arc(2.0), arc(3.0));
        let ext = curve_curve_extrema(&a, &b, TOL);
        let (_, regions) = incomplete(ext.completeness());
        // Both arcs share their parametrization: the valley is `s = t`.
        for k in 0..=16 {
            let s = k as f64 / 16.0;
            assert!(
                regions.iter().any(|r| holds(r, (&a, s), (&b, s))),
                "{s}: {regions:#?}"
            );
        }
        assert_eq!(
            ext.minima(),
            curve_curve_minima(&a, &b, TOL, false).minima()
        );
        assert_eq!(ext.len(), 1, "{ext:#?}");
        assert!((ext[0].distance - 1.0).abs() < 1e-12);
        // Apart, the arcs never meet: no hit, and none missing.
        assert!(search_curve_curve(&a, &b, TOL).is_empty());
        assert!(search_curve_curve(&a, &b, TOL).is_complete());
    }

    #[test]
    fn test_minima_on_ends_and_corners_match_grid() {
        // A bow beside a segment, the bow's ends closest: two corners.
        let bow = quadratic([0.0, 0.0, 0.0], [1.0, -1.0, 0.3], [2.0, 0.0, 0.0]);
        let rail = segment([-0.5, 1.0, 0.0], [2.5, 1.2, 0.0]);
        let ext = assert_minima_match(&bow, &rail);
        assert!(
            ext.iter()
                .any(|e| e.first_parameter == 0.0 || e.first_parameter == 1.0),
            "{ext:#?}"
        );
        // An end of one closest to the interior of the other.
        let stub = segment([1.0, 2.0, 0.5], [1.2, 4.0, 0.5]);
        let ext = assert_minima_match(&bow, &stub);
        assert!(ext.iter().any(|e| e.second_parameter == 0.0), "{ext:#?}");
        // A closed loop: no ends of its own.
        let loop_a = ring([0.0, 0.0, 0.0], 2.0, 0.3);
        let ext = assert_minima_match(&stub, &loop_a);
        assert!(ext.iter().any(|e| e.first_parameter == 0.0), "{ext:#?}");
        assert!(ext.iter().all(|e| (0.0..8.0).contains(&e.second_parameter)));
    }
}
