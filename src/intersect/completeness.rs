//! Whether a numeric intersection found every hit (see [`Completeness`]).

/// Whether the hits of a numeric intersection are all of its hits.
///
/// A search that subdivides B-splines can prove it: every pair of pieces
/// was either shown apart by its bounding boxes or shrunk until it held
/// at most one hit, which was then solved for. A sampled search proves
/// nothing. `R` is the kind of region an incomplete search reports, such
/// as [`CurveSurfaceRegion`](crate::CurveSurfaceRegion).
///
/// # Examples
///
/// ```
/// use geomcore::{Completeness, Line3D, Point3D, Sphere, Tolerance, Vector3D, intersect_curve_surface};
/// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
/// let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
/// let hits = intersect_curve_surface(&line, &sphere, Tolerance::DEFAULT);
/// // Neither shape is a B-spline: the hits come from samples.
/// assert_eq!(hits.completeness(), &Completeness::Sampled);
/// assert!(!hits.is_complete());
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

/// Why a numeric intersection is [`Completeness::Incomplete`].
///
/// # Examples
///
/// ```
/// use geomcore::{
///     BSplineCurve3D, BSplineSurface, Completeness, IncompleteReason, Point3D, Tolerance,
///     intersect_curve_surface,
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
/// let hits = intersect_curve_surface(&segment, &square, Tolerance::DEFAULT);
/// match hits.completeness() {
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
    /// separated. A curve lying on a curved surface often ends this way:
    /// covering the overlap with pieces small enough to be told apart
    /// from a crossing takes more halvings than the budget allows.
    BudgetExhausted,
    /// Some pairs of pieces never separated, however small they were
    /// halved: a tangent contact, an overlap (the curve lying on the
    /// surface), or a hit at a singular point. Its regions hold them.
    Unseparated,
}
