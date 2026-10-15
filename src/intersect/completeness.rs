//! Whether a numeric intersection found every hit (see [`Completeness`]).

use crate::Point3D;

/// Whether the hits of a numeric intersection are all of its hits.
///
/// A search that subdivides B-splines can prove it: every pair of pieces
/// was either shown apart by its bounding boxes or shrunk until it held
/// at most one hit, which was then solved for. A sampled search proves
/// nothing. `R` is the kind of region an incomplete search reports, such
/// as [`CurveSurfaceRegion`](crate::CurveSurfaceRegion) or
/// [`CurveCurveRegion`](crate::CurveCurveRegion). The local minima of the
/// distance between two curves report the same status, a minimum
/// standing for a hit.
///
/// # Examples
///
/// ```
/// use geomcore::{Completeness, Ellipse3D, Frame3D, Intersect, Point3D, Tolerance, Torus, Vector3D};
/// let ellipse = Ellipse3D::from_frame(Frame3D::WORLD, 5.0, 1.0).unwrap();
/// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 3.0, 1.0).unwrap();
/// let result = ellipse.intersect(&torus, Tolerance::DEFAULT);
/// // An ellipse and a torus have no closed form, and neither is a
/// // B-spline: the hits come from samples.
/// assert_eq!(result.completeness(), &Completeness::Sampled);
/// assert!(!result.is_complete());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Completeness<R> {
    /// Every hit was found: no hit lies outside those reported.
    Complete,
    /// The search did not settle everywhere: hits may be missing, but
    /// only within `regions`.
    Incomplete {
        /// Why the search did not settle.
        reason: IncompleteReason,
        /// Where hits may be missing, disjoint after merging the ones
        /// that touch.
        regions: Vec<R>,
    },
    /// The hits were found from samples, so some may be missing anywhere.
    Sampled,
}

impl<R> Completeness<R> {
    /// Whether every hit was found.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Completeness, CurveSurfaceRegion};
    /// assert!(Completeness::<CurveSurfaceRegion>::Complete.is_complete());
    /// assert!(!Completeness::<CurveSurfaceRegion>::Sampled.is_complete());
    /// ```
    pub fn is_complete(&self) -> bool {
        matches!(self, Completeness::Complete)
    }
}

/// Complete when a search left no pair of pieces unsettled, else
/// incomplete with the merged `regions` of the pairs it left: out of
/// budget if `exhausted`, else never separating.
pub(super) fn of_unsettled<R: Region>(regions: Vec<R>, exhausted: bool) -> Completeness<R> {
    if regions.is_empty() {
        return Completeness::Complete;
    }
    Completeness::Incomplete {
        reason: if exhausted {
            IncompleteReason::BudgetExhausted
        } else {
            IncompleteReason::Unseparated
        },
        regions: merged(regions),
    }
}

/// The completeness of the hits of two searches taken together: complete
/// when both are, else incomplete with the regions of both, out of budget
/// if either ran out; sampled if either is.
pub(super) fn both<R: Region>(a: Completeness<R>, b: Completeness<R>) -> Completeness<R> {
    use Completeness::{Complete, Incomplete, Sampled};
    match (a, b) {
        (Sampled, _) | (_, Sampled) => Sampled,
        (Complete, other) | (other, Complete) => other,
        (
            Incomplete {
                reason: ra,
                mut regions,
            },
            Incomplete {
                reason: rb,
                regions: more,
            },
        ) => {
            regions.extend(more);
            let exhausted = [ra, rb].contains(&IncompleteReason::BudgetExhausted);
            of_unsettled(regions, exhausted)
        }
    }
}

/// A region of an incomplete search: parameter ranges on both shapes and
/// a box in space.
pub(super) trait Region: Sized {
    /// Whether the two regions' parameter ranges meet.
    fn touches(&self, other: &Self) -> bool;
    /// The smallest region holding both.
    fn union(&self, other: &Self) -> Self;
    /// The start of the first shape's parameter range.
    fn start(&self) -> f64;
}

/// Whether two closed intervals meet.
pub(super) fn meet((a0, a1): (f64, f64), (b0, b1): (f64, f64)) -> bool {
    a0 <= b1 && b0 <= a1
}

/// The smallest interval holding both.
pub(super) fn span((a0, a1): (f64, f64), (b0, b1): (f64, f64)) -> (f64, f64) {
    (a0.min(b0), a1.max(b1))
}

/// The smallest box, as its two corners, holding both.
pub(super) fn box_union(
    (p, r): (Point3D, Point3D),
    (q, s): (Point3D, Point3D),
) -> (Point3D, Point3D) {
    (
        Point3D::new(p.x.min(q.x), p.y.min(q.y), p.z.min(q.z)),
        Point3D::new(r.x.max(s.x), r.y.max(s.y), r.z.max(s.z)),
    )
}

/// `regions` with every group of touching ones merged into one, ordered
/// by the start of their first parameter ranges.
pub(super) fn merged<R: Region>(mut regions: Vec<R>) -> Vec<R> {
    let mut out: Vec<R> = Vec::new();
    while let Some(mut region) = regions.pop() {
        loop {
            let before = out.len();
            out.retain(|r| {
                let touches = region.touches(r);
                if touches {
                    region = region.union(r);
                }
                !touches
            });
            if out.len() == before {
                break;
            }
        }
        out.push(region);
    }
    out.sort_by(|a, b| a.start().total_cmp(&b.start()));
    out
}

/// Why a numeric intersection is [`Completeness::Incomplete`].
///
/// # Examples
///
/// ```
/// use geomcore::{
///     BSplineCurve3D, BSplineSurface, Completeness, IncompleteReason, Intersect, Point3D,
///     Tolerance,
/// };
/// let poles = vec![
///     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 2.0, 0.0)],
///     vec![Point3D::new(2.0, 0.0, 0.0), Point3D::new(2.0, 2.0, 0.0)],
/// ];
/// let (knots, mults) = (vec![0.0, 1.0], vec![2, 2]);
/// let square = BSplineSurface::new(
///     1, 1, poles, knots.clone(), mults.clone(), knots.clone(), mults.clone(), false, false,
/// )
/// .unwrap();
/// // A segment lying on the square never separates from it.
/// let ends = vec![Point3D::new(0.5, 0.5, 0.0), Point3D::new(1.5, 1.0, 0.0)];
/// let segment = BSplineCurve3D::new(1, ends, knots, mults, false).unwrap();
/// let result = segment.intersect(&square, Tolerance::DEFAULT);
/// match result.completeness() {
///     Completeness::Incomplete { reason, regions } => {
///         assert_eq!(*reason, IncompleteReason::Unseparated);
///         assert_eq!(regions[0].curve_range, (0.0, 1.0));
///     }
///     other => panic!("expected an incomplete search, got {other:?}"),
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncompleteReason {
    /// The search ran out of its budget of halvings. Its regions hold the
    /// pairs of pieces it had not settled yet, as well as any that never
    /// separated. A curve lying on a curved surface, two curves
    /// overlapping along a curved stretch, or a valley of equal distance
    /// between two curves (concentric arcs) often ends this way: covering
    /// it with pieces small enough to be told apart from a crossing, or
    /// from an isolated minimum, takes more halvings than the budget
    /// allows.
    BudgetExhausted,
    /// Some pairs of pieces never separated, however small they were
    /// halved: a tangent contact, an overlap (a curve lying on the
    /// surface or along the other curve), or a hit at a singular point.
    /// Its regions hold them.
    Unseparated,
}
