//! The routed intersection entry point.
//!
//! [`Intersect`] gives the full answer for every supported pair: the exact
//! intersection when a closed form exists, the numeric one otherwise, and
//! the result says which method ran. [`IntersectExact`] is the typed exact
//! path, and never falls back to numeric work.
//!
//! This module also holds the helpers the routed implementations share:
//! [`CurveOperand`], which tells the router whether a curve has an
//! elementary basis and which parameter range restricts it, and the
//! parameter-range clipping that turns an exact intersection of the basis
//! into the intersection of the trimmed curve.

use crate::Tolerance;
use crate::curves::{
    BSplineCurve3D, Circle2D, Circle3D, Curve2D, Curve3D, ElementaryCurve, Ellipse3D, Hyperbola3D,
    Line2D, Line3D, Parabola3D, ParametricCurve2D, ParametricCurve3D, TrimmedCurve2D,
    TrimmedCurve3D,
};
use crate::intersect::dispatch::{ExactCurveCurve, ExactCurveSurface};
use crate::intersect::results::{
    CurveCurveHitList, CurveCurveOverlapList, CurveSurfaceOverlapList,
};
use crate::intersect::{
    CircleCircle2DIntersection, CurveCurveHit, CurveCurveHit2D, CurveCurveIntersection,
    CurveCurveIntersection2D, CurveCurveOverlap, CurveSurfaceHit, CurveSurfaceIntersection,
    ExactCurveCurveIntersection, ExactCurveSurfaceIntersection, InlineList, IntersectExact,
    LineCircle2DIntersection, LineLine2DIntersection, intersect_curve_curve,
    intersect_curve_surface,
};
use crate::surfaces::{
    BSplineSurface, Cone, Cylinder, ElementarySurface, ParametricSurface, Plane, Sphere, Surface,
    Torus,
};
use crate::{Point2D, Point3D};

/// Routed intersection: exact when a closed form exists, numeric otherwise.
///
/// Unlike [`IntersectExact`], the typed exact path that never runs numeric
/// work, `intersect` works for every supported pair, and its result says which
/// method ran (`method()`) and whether it is complete.
///
/// The operands may come in either order; a curve-surface result always
/// reads the curve parameter first.
///
/// # Examples
///
/// ```
/// use geomcore::{Intersect, IntersectionMethod, Line3D, Point3D, Sphere, Tolerance, Vector3D};
///
/// // A line along z through a sphere of radius 2 crosses it at both poles.
/// let line = Line3D::new(Point3D::new(0.0, 0.0, -5.0), Vector3D::Z).unwrap();
/// let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
/// let result = line.intersect(&sphere, Tolerance::DEFAULT);
/// assert_eq!(result.method(), IntersectionMethod::Exact);
/// assert!(result.is_complete());
/// let parameters: Vec<f64> = result.hits().iter().map(|h| h.curve_parameter).collect();
/// assert_eq!(parameters, [3.0, 7.0]);
/// assert_eq!(sphere.intersect(&line, Tolerance::DEFAULT), result);
/// ```
pub trait Intersect<Rhs: ?Sized = Self> {
    /// The result of intersecting `Self` with `Rhs`.
    type Output;

    /// Intersects `self` with `other`, deciding tangency and coincidence
    /// with `tol`.
    fn intersect(&self, other: &Rhs, tol: Tolerance) -> Self::Output;
}

/// An elementary curve the router solves exactly against every surface
/// basis and every curve basis, through the pair's typed solver.
///
/// The concrete curves are their own basis, so a pair whose types are known
/// at compile time is solved without converting to the untyped
/// [`ElementaryCurve`] or dispatching at run time; [`ElementaryCurve`] is
/// the basis of the curves whose kind is only known at run time.
pub(crate) trait CurveBasis:
    ParametricCurve3D
    + Copy
    + Into<ElementaryCurve>
    + ExactCurveSurface<Plane>
    + ExactCurveSurface<Cylinder>
    + ExactCurveSurface<Cone>
    + ExactCurveSurface<Sphere>
    + ExactCurveSurface<Torus>
    + ExactCurveSurface<ElementarySurface>
    + ExactCurveCurve<Line3D>
    + ExactCurveCurve<Circle3D>
    + ExactCurveCurve<Ellipse3D>
    + ExactCurveCurve<Parabola3D>
    + ExactCurveCurve<Hyperbola3D>
    + ExactCurveCurve<ElementaryCurve>
{
}

impl CurveBasis for Line3D {}
impl CurveBasis for Circle3D {}
impl CurveBasis for Ellipse3D {}
impl CurveBasis for Parabola3D {}
impl CurveBasis for Hyperbola3D {}
impl CurveBasis for ElementaryCurve {}

/// An elementary surface as the router sees it: the surface parameters of
/// a hit's point.
pub(crate) trait SurfaceBasis: Copy {
    /// The `(u, v)` parameters of `point`, a point on the surface.
    fn surface_parameters_of(&self, point: Point3D) -> (f64, f64);
}

macro_rules! surface_basis {
    ($($surface:ty),*) => {$(
        impl SurfaceBasis for $surface {
            #[inline]
            fn surface_parameters_of(&self, point: Point3D) -> (f64, f64) {
                self.parameters_of(point)
            }
        }
    )*};
}

surface_basis!(Plane, Cylinder, Cone, Sphere, Torus, ElementarySurface);

/// How a curve enters the router: its elementary basis (when it has one)
/// and the parameter range it is restricted to (`None` = the basis's own
/// bounds).
pub(crate) trait CurveOperand: ParametricCurve3D {
    /// The basis type: the curve's own type when it is a concrete
    /// elementary curve, so the pair is solved by its typed solver.
    type Basis: CurveBasis;

    /// The elementary basis and the restricting parameter range, or `None`
    /// when the curve has no elementary basis.
    fn elementary_basis(&self) -> Option<Operand<Self::Basis>>;
}

macro_rules! elementary_operand {
    ($($curve:ty),*) => {$(
        impl CurveOperand for $curve {
            type Basis = $curve;

            #[inline]
            fn elementary_basis(&self) -> Option<Operand<$curve>> {
                Some((*self, None))
            }
        }
    )*};
}

elementary_operand!(
    Line3D,
    Circle3D,
    Ellipse3D,
    Parabola3D,
    Hyperbola3D,
    ElementaryCurve
);

impl CurveOperand for BSplineCurve3D {
    type Basis = ElementaryCurve;

    fn elementary_basis(&self) -> Option<Operand<ElementaryCurve>> {
        None
    }
}

impl CurveOperand for Curve3D {
    type Basis = ElementaryCurve;

    #[inline]
    fn elementary_basis(&self) -> Option<Operand<ElementaryCurve>> {
        self.as_elementary().map(|basis| (basis, None))
    }
}

impl<C: CurveOperand> CurveOperand for TrimmedCurve3D<C> {
    type Basis = C::Basis;

    /// A trimmed curve of a trimmed curve keeps the inner basis and the
    /// outer range: the outer range lies within the inner one.
    #[inline]
    fn elementary_basis(&self) -> Option<Operand<C::Basis>> {
        self.basis()
            .elementary_basis()
            .map(|(basis, _)| (basis, Some((self.first(), self.last()))))
    }
}

/// How a surface enters the router: its elementary basis, when it has one.
pub(crate) trait SurfaceOperand: ParametricSurface {
    /// The basis type: the surface's own type when it is a concrete
    /// elementary surface, so the pair is solved by its typed solver.
    type Basis: SurfaceBasis;

    /// The elementary basis, or `None` when the surface has none.
    fn surface_basis(&self) -> Option<Self::Basis>;
}

macro_rules! elementary_surface_operand {
    ($($surface:ty),*) => {$(
        impl SurfaceOperand for $surface {
            type Basis = $surface;

            #[inline]
            fn surface_basis(&self) -> Option<$surface> {
                Some(*self)
            }
        }
    )*};
}

elementary_surface_operand!(Plane, Cylinder, Cone, Sphere, Torus, ElementarySurface);

impl SurfaceOperand for BSplineSurface {
    type Basis = ElementarySurface;

    fn surface_basis(&self) -> Option<ElementarySurface> {
        None
    }
}

impl SurfaceOperand for Surface {
    type Basis = ElementarySurface;

    #[inline]
    fn surface_basis(&self) -> Option<ElementarySurface> {
        self.as_elementary()
    }
}

/// Whether `parameter` lies in `range` within `tol.parametric`, and its
/// representative in the range.
///
/// On a periodic curve the parameter is shifted by whole periods into
/// `[range.0, range.0 + period)`. A parameter within `tol.parametric`
/// below `range.0` is accepted as it is, not shifted.
pub(crate) fn in_range(
    parameter: f64,
    range: (f64, f64),
    period: Option<f64>,
    tol: Tolerance,
) -> Option<f64> {
    let Some(period) = period else {
        let inside = parameter >= range.0 - tol.parametric && parameter <= range.1 + tol.parametric;
        return inside.then_some(parameter);
    };
    let shifted = parameter - period * ((parameter - range.0) / period).floor();
    if shifted <= range.1 + tol.parametric {
        return Some(shifted);
    }
    let just_below = shifted - period;
    (just_below >= range.0 - tol.parametric).then_some(just_below)
}

/// The representative of `parameter` in `range`, when the curve is
/// restricted to one (see [`in_range`]); the parameter itself when it is not.
fn representative_in(
    parameter: f64,
    range: Option<(f64, f64)>,
    period: Option<f64>,
    tol: Tolerance,
) -> Option<f64> {
    match range {
        Some(range) => in_range(parameter, range, period, tol),
        None => Some(parameter),
    }
}

/// Clips an overlap `(a, b)` to `range`, shifting by periods like
/// [`in_range`].
///
/// Returns the pieces of the overlap that lie in the range, ordered by
/// start, each expressed in the range's representation: none, one, or two
/// when the overlap meets the range on both sides of the seam of a periodic
/// curve. An overlap covering the whole period returns the range as one
/// piece. The three slots hold the three shifted copies the search tries,
/// so the list never fills up.
///
/// A piece no longer than `tol.parametric` is kept as a single point (see
/// [`is_single_point`]): the overlap only touches the range there. Such a
/// point is dropped when it is, up to a period, the end of another piece,
/// so a touch is never reported twice.
pub(crate) fn clip_range(
    overlap: (f64, f64),
    range: (f64, f64),
    period: Option<f64>,
    tol: Tolerance,
) -> InlineList<(f64, f64), 3> {
    let clip = |(low, high): (f64, f64)| {
        let clipped = (low.max(range.0), high.min(range.1));
        (clipped.1 - clipped.0 >= -tol.parametric).then_some(clipped)
    };
    let mut pieces = InlineList::new();
    let Some(period) = period else {
        if let Some(piece) = clip(overlap) {
            pieces.push(piece);
        }
        return pieces;
    };
    if overlap.1 - overlap.0 >= period - tol.parametric {
        pieces.push(range);
        return pieces;
    }
    let first_copy = ((range.0 - overlap.1) / period).floor();
    let mut copies = InlineList::<(f64, f64), 3>::new();
    for step in 0..3 {
        let shift = (first_copy + f64::from(step)) * period;
        if let Some(piece) = clip((overlap.0 + shift, overlap.1 + shift)) {
            copies.push(piece);
        }
    }
    let same_point = |a: f64, b: f64| {
        let gap = (a - b).rem_euclid(period);
        gap.min(period - gap) <= tol.parametric
    };
    for (index, &piece) in copies.iter().enumerate() {
        let repeated = is_single_point(piece, tol)
            && copies.iter().enumerate().any(|(other_index, &other)| {
                let earlier_point = is_single_point(other, tol) && other_index < index;
                let longer = !is_single_point(other, tol);
                (earlier_point || longer)
                    && (same_point(piece.0, other.0) || same_point(piece.0, other.1))
            });
        if !repeated {
            pieces.push(piece);
        }
    }
    pieces.sort_by(|a, b| a.0.total_cmp(&b.0));
    pieces
}

/// Whether a piece clipped by [`clip_range`] is no longer than
/// `tol.parametric`: a single point where the overlap touches the range.
pub(crate) fn is_single_point(piece: (f64, f64), tol: Tolerance) -> bool {
    piece.1 - piece.0 <= tol.parametric
}

/// Intersects a curve with a surface: the exact path when the curve has an
/// elementary basis, the surface is elementary and the pair has a closed
/// form; the numeric search otherwise.
#[inline]
pub(crate) fn route_curve_surface<C: CurveOperand, S: SurfaceOperand>(
    curve: &C,
    surface: &S,
    tol: Tolerance,
) -> CurveSurfaceIntersection
where
    C::Basis: ExactCurveSurface<S::Basis>,
{
    if let (Some((basis, range)), Some(elementary)) =
        (curve.elementary_basis(), surface.surface_basis())
    {
        let exact = basis.exact_curve_surface(&elementary, tol);
        if !matches!(exact, ExactCurveSurfaceIntersection::NotAnalytic) {
            return from_exact_curve_surface(exact, &basis, range, &elementary, tol);
        }
    }
    // A trimmed curve is searched over its own range only: no clipping.
    let hits = intersect_curve_surface(curve, surface, tol);
    let completeness = hits.completeness().clone();
    CurveSurfaceIntersection::numeric(hits.into_hits(), completeness)
}

/// Turns the exact intersection of the curve's elementary `basis` with the
/// surface into the routed result: hits outside `range` dropped, the kept
/// ones carrying their representative in `range` and the surface
/// parameters of their point.
#[inline]
fn from_exact_curve_surface(
    exact: ExactCurveSurfaceIntersection,
    basis: &impl ParametricCurve3D,
    range: Option<(f64, f64)>,
    elementary: &impl SurfaceBasis,
    tol: Tolerance,
) -> CurveSurfaceIntersection {
    match exact {
        ExactCurveSurfaceIntersection::Points(list) => {
            let hits = list.iter().filter_map(|hit| {
                let curve_parameter = match range {
                    Some(range) => in_range(hit.parameter, range, basis.period(), tol)?,
                    None => hit.parameter,
                };
                Some(CurveSurfaceHit {
                    curve_parameter,
                    surface_parameters: elementary.surface_parameters_of(hit.point),
                    point: hit.point,
                    multiplicity: hit.multiplicity,
                })
            });
            CurveSurfaceIntersection::exact(hits, CurveSurfaceOverlapList::new())
        }
        // The whole curve lies on the surface: a trimmed curve over its
        // range, the basis over its bounds (infinite for a line).
        ExactCurveSurfaceIntersection::Coincident => {
            let overlap = range.unwrap_or(basis.bounds());
            CurveSurfaceIntersection::exact([], [overlap].into_iter().collect())
        }
        ExactCurveSurfaceIntersection::Empty => {
            CurveSurfaceIntersection::exact([], CurveSurfaceOverlapList::new())
        }
        ExactCurveSurfaceIntersection::NotAnalytic => {
            unreachable!("the router sends a pair with no closed form to the numeric search")
        }
    }
}

/// Implements [`Intersect`] for every curve of the first list with every
/// surface of the second, in both orders. The surface-first order returns
/// the same result as the curve-first one.
macro_rules! curve_surface {
    (@pairs $curve:ty; [$($surface:ty),*]) => {$(
        impl Intersect<$surface> for $curve {
            type Output = CurveSurfaceIntersection;

            #[inline]
            fn intersect(&self, other: &$surface, tol: Tolerance) -> CurveSurfaceIntersection {
                route_curve_surface(self, other, tol)
            }
        }

        impl Intersect<$curve> for $surface {
            type Output = CurveSurfaceIntersection;

            #[inline]
            fn intersect(&self, other: &$curve, tol: Tolerance) -> CurveSurfaceIntersection {
                route_curve_surface(other, self, tol)
            }
        }
    )*};
    ($($curve:ty),*; $surfaces:tt) => {$(
        curve_surface!(@pairs $curve; $surfaces);
    )*};
}

curve_surface!(
    Line3D, Circle3D, Ellipse3D, Parabola3D, Hyperbola3D, BSplineCurve3D, Curve3D, ElementaryCurve;
    [Plane, Cylinder, Cone, Sphere, Torus, BSplineSurface, Surface, ElementarySurface]
);

/// Implements [`Intersect`] between a trimmed curve over any curve operand
/// and each surface, in both orders.
macro_rules! trimmed_curve_surface {
    ($($surface:ty),*) => {$(
        impl<X: CurveOperand> Intersect<$surface> for TrimmedCurve3D<X> {
            type Output = CurveSurfaceIntersection;

            #[inline]
            fn intersect(&self, other: &$surface, tol: Tolerance) -> CurveSurfaceIntersection {
                route_curve_surface(self, other, tol)
            }
        }

        impl<X: CurveOperand> Intersect<TrimmedCurve3D<X>> for $surface {
            type Output = CurveSurfaceIntersection;

            #[inline]
            fn intersect(&self, other: &TrimmedCurve3D<X>, tol: Tolerance) -> CurveSurfaceIntersection {
                route_curve_surface(other, self, tol)
            }
        }
    )*};
}

trimmed_curve_surface!(
    Plane,
    Cylinder,
    Cone,
    Sphere,
    Torus,
    BSplineSurface,
    Surface,
    ElementarySurface
);

/// Intersects two curves: the exact path when both have an elementary
/// basis, the numeric search otherwise.
#[inline]
pub(crate) fn route_curve_curve<A: CurveOperand, B: CurveOperand>(
    a: &A,
    b: &B,
    tol: Tolerance,
) -> CurveCurveIntersection
where
    A::Basis: ExactCurveCurve<B::Basis>,
{
    if let (Some(first), Some(second)) = (a.elementary_basis(), b.elementary_basis()) {
        let exact = first.0.exact_curve_curve(&second.0, tol);
        debug_assert!(
            !matches!(exact, ExactCurveCurveIntersection::NotAnalytic),
            "every pair of elementary curves has a closed form"
        );
        if !matches!(exact, ExactCurveCurveIntersection::NotAnalytic) {
            return from_exact_curve_curve(exact, first, second, tol);
        }
    }
    // A trimmed curve is searched over its own range only: no clipping.
    let hits = intersect_curve_curve(a, b, tol);
    let completeness = hits.completeness().clone();
    CurveCurveIntersection::numeric(hits.into_hits(), completeness)
}

/// An elementary basis and the parameter range restricting it.
type Operand<C> = (C, Option<(f64, f64)>);

/// Turns the exact intersection of two elementary bases into the routed
/// result: hits outside either range dropped, the kept ones carrying their
/// representative in each range, and a coincidence clipped to both ranges.
#[inline]
fn from_exact_curve_curve<A: CurveBasis, B: CurveBasis>(
    exact: ExactCurveCurveIntersection,
    (first, first_range): Operand<A>,
    (second, second_range): Operand<B>,
    tol: Tolerance,
) -> CurveCurveIntersection {
    match exact {
        // Neither basis is restricted: the solver's list is kept as it is,
        // already ordered by first parameter.
        ExactCurveCurveIntersection::Points(list)
            if first_range.is_none() && second_range.is_none() =>
        {
            CurveCurveIntersection::exact(
                CurveCurveHitList::Inline(list),
                CurveCurveOverlapList::new(),
            )
        }
        ExactCurveCurveIntersection::Points(list) => {
            let hits = list.iter().filter_map(|hit| {
                Some(CurveCurveHit {
                    first_parameter: representative_in(
                        hit.first_parameter,
                        first_range,
                        first.period(),
                        tol,
                    )?,
                    second_parameter: representative_in(
                        hit.second_parameter,
                        second_range,
                        second.period(),
                        tol,
                    )?,
                    ..*hit
                })
            });
            CurveCurveIntersection::exact(hits.collect(), CurveCurveOverlapList::new())
        }
        // Rare: the coincidence is worked out on the untyped curves, so the
        // overlap mapping exists once.
        ExactCurveCurveIntersection::Coincident => {
            let (first, second): (ElementaryCurve, ElementaryCurve) = (first.into(), second.into());
            let first_range = first_range.unwrap_or(first.bounds());
            let second_range = second_range.unwrap_or(second.bounds());
            let shared = coincident_overlaps((&first, first_range), (&second, second_range), tol);
            let hits = shared
                .touches
                .iter()
                .map(|&(first_parameter, second_parameter)| CurveCurveHit {
                    first_parameter,
                    second_parameter,
                    point: first.eval_point(first_parameter),
                    multiplicity: 1,
                });
            CurveCurveIntersection::exact(hits.collect(), shared.overlaps)
        }
        ExactCurveCurveIntersection::Empty => {
            CurveCurveIntersection::exact(CurveCurveHitList::new(), CurveCurveOverlapList::new())
        }
        ExactCurveCurveIntersection::NotAnalytic => {
            unreachable!("the router sends a pair with no closed form to the numeric search")
        }
    }
}

/// The overlaps of two coincident curves restricted to `first_range` and
/// `second_range`.
///
/// Each overlap holds a monotone range in each curve's own range: the
/// second range is mapped onto the first curve and clipped there (two
/// pieces when it crosses the first curve's seam), then each piece is
/// mapped back onto the second curve and split where it crosses the
/// second curve's seam.
///
/// A piece no longer than `tol.parametric`, where the ranges only touch
/// (two consecutive pieces of one line or circle), is not an overlap: it
/// comes back in [`CoincidentPieces::touches`] as a parameter pair, each
/// parameter the point's representative in its range.
fn coincident_overlaps<C: CoincidentCurve>(
    (first, first_range): (&C, (f64, f64)),
    (second, second_range): (&C, (f64, f64)),
    tol: Tolerance,
) -> CoincidentPieces {
    let same_direction = runs_along(second, second_range, first);
    let on_first = mapped_range(second, second_range, first, same_direction, tol);
    let mut overlaps = CurveCurveOverlapList::new();
    let mut touches = InlineList::new();
    for &piece in clip_range(on_first, first_range, first.cycle_length(), tol).iter() {
        if is_single_point(piece, tol) {
            let first_parameter = (0.5 * (piece.0 + piece.1)).clamp(first_range.0, first_range.1);
            let found = second.parameter_of_image(first, first_parameter);
            let second_parameter = in_range(found, second_range, second.cycle_length(), tol)
                .unwrap_or(found)
                .clamp(second_range.0, second_range.1);
            touches.push((first_parameter, second_parameter));
            continue;
        }
        let on_second = mapped_range(first, piece, second, same_direction, tol);
        let pieces_on_second = clip_range(on_second, second_range, second.cycle_length(), tol);
        if let [single] = pieces_on_second[..] {
            overlaps.push(CurveCurveOverlap {
                first_range: piece,
                second_range: single,
                same_direction,
            });
            continue;
        }
        // The piece crosses the second curve's seam: read the first curve's
        // part of each side back from its end points.
        for &part in pieces_on_second.iter() {
            let end = |parameter: f64| {
                let found = first.parameter_of_image(second, parameter);
                in_range(found, piece, first.cycle_length(), tol)
                    .unwrap_or(found)
                    .clamp(piece.0, piece.1)
            };
            let (start, finish) = (end(part.0), end(part.1));
            overlaps.push(CurveCurveOverlap {
                first_range: (start.min(finish), start.max(finish)),
                second_range: part,
                same_direction,
            });
        }
    }
    overlaps.sort_by(|p, q| p.first_range.0.total_cmp(&q.first_range.0));
    CoincidentPieces { overlaps, touches }
}

/// What two coincident curves restricted to parameter ranges share.
struct CoincidentPieces {
    /// The overlaps, ordered by start on the first curve.
    overlaps: CurveCurveOverlapList,
    /// The `(first, second)` parameters of the points where the ranges
    /// only touch, ordered by first parameter. At most one per clipped
    /// piece, so three slots never fill up.
    touches: InlineList<(f64, f64), 3>,
}

/// What the overlap mapping needs from a curve, in 2D or 3D, to relate it
/// to a coincident curve of the same kind.
pub(crate) trait CoincidentCurve {
    /// The period, or `None` when the curve is not periodic.
    fn cycle_length(&self) -> Option<f64>;

    /// The parameter, on this curve, of the point `other` has at
    /// `parameter`; `other` coincides with this curve.
    fn parameter_of_image(&self, other: &Self, parameter: f64) -> f64;

    /// Whether this curve at `parameter` and `other` at the same point
    /// run the same way.
    fn runs_with(&self, parameter: f64, other: &Self) -> bool;
}

impl CoincidentCurve for ElementaryCurve {
    fn cycle_length(&self) -> Option<f64> {
        self.period()
    }

    fn parameter_of_image(&self, other: &Self, parameter: f64) -> f64 {
        self.parameter_of(other.eval_point(parameter))
    }

    fn runs_with(&self, parameter: f64, other: &Self) -> bool {
        let image = other.parameter_of_image(self, parameter);
        self.eval_derivative(parameter, 1)
            .dot(other.eval_derivative(image, 1))
            >= 0.0
    }
}

impl CoincidentCurve for Curve2D {
    fn cycle_length(&self) -> Option<f64> {
        self.period()
    }

    fn parameter_of_image(&self, other: &Self, parameter: f64) -> f64 {
        let point = other.eval_point(parameter);
        match self {
            Curve2D::Line(line) => line.parameter_of(point),
            Curve2D::Circle(circle) => circle.parameter_of(point),
        }
    }

    fn runs_with(&self, parameter: f64, other: &Self) -> bool {
        let image = other.parameter_of_image(self, parameter);
        self.eval_derivative(parameter, 1)
            .dot(other.eval_derivative(image, 1))
            >= 0.0
    }
}

/// Whether the coincident curves `from` and `to` run the same way: the
/// sign of the tangents' dot product at a sample of `range`.
fn runs_along<C: CoincidentCurve>(from: &C, range: (f64, f64), to: &C) -> bool {
    let sample = match (range.0.is_finite(), range.1.is_finite()) {
        (true, true) => 0.5 * (range.0 + range.1),
        (true, false) => range.0,
        (false, true) => range.1,
        (false, false) => 0.0,
    };
    from.runs_with(sample, to)
}

/// The image of `range` of curve `from` on the coincident curve `to`, as a
/// monotone interval that may lie outside `to`'s own representation.
/// `along` says whether the curves run the same way.
///
/// An infinite end maps to the infinity its direction gives on `to`. On a
/// periodic curve the image starts at the image of the range's start in
/// the direction of travel, and a full period maps to a full period.
fn mapped_range<C: CoincidentCurve>(
    from: &C,
    range: (f64, f64),
    to: &C,
    along: bool,
    tol: Tolerance,
) -> (f64, f64) {
    let image = |parameter: f64| to.parameter_of_image(from, parameter);
    let Some(period) = to.cycle_length() else {
        let end = |parameter: f64| {
            if parameter.is_finite() {
                image(parameter)
            } else if along {
                parameter
            } else {
                -parameter
            }
        };
        let (start, finish) = (end(range.0), end(range.1));
        return (start.min(finish), start.max(finish));
    };
    let full_period = from.cycle_length().unwrap_or(period);
    if range.1 - range.0 >= full_period - tol.parametric {
        let start = image(range.0);
        return (start, start + period);
    }
    let (start, finish) = if along {
        (image(range.0), image(range.1))
    } else {
        (image(range.1), image(range.0))
    };
    (start, start + (finish - start).rem_euclid(period))
}

/// Implements [`Intersect`] for every curve of the list with every curve of
/// the list, the receiver read first.
macro_rules! curve_curve {
    (@row $first:ty; [$($second:ty),*]) => {$(
        impl Intersect<$second> for $first {
            type Output = CurveCurveIntersection;

            #[inline]
            fn intersect(&self, other: &$second, tol: Tolerance) -> CurveCurveIntersection {
                route_curve_curve(self, other, tol)
            }
        }
    )*};
    ([$($first:ty),*]) => {
        curve_curve!(@all [$($first),*]; [$($first),*]);
    };
    (@all [$($first:ty),*]; $seconds:tt) => {$(
        curve_curve!(@row $first; $seconds);
    )*};
}

curve_curve!([
    Line3D,
    Circle3D,
    Ellipse3D,
    Parabola3D,
    Hyperbola3D,
    BSplineCurve3D,
    Curve3D,
    ElementaryCurve
]);

/// Implements [`Intersect`] between a trimmed curve over any curve operand
/// and each curve of the list, in both orders.
macro_rules! trimmed_curve_curve {
    ($($curve:ty),*) => {$(
        impl<X: CurveOperand> Intersect<$curve> for TrimmedCurve3D<X> {
            type Output = CurveCurveIntersection;

            #[inline]
            fn intersect(&self, other: &$curve, tol: Tolerance) -> CurveCurveIntersection {
                route_curve_curve(self, other, tol)
            }
        }

        impl<X: CurveOperand> Intersect<TrimmedCurve3D<X>> for $curve
        where
            <$curve as CurveOperand>::Basis: ExactCurveCurve<X::Basis>,
        {
            type Output = CurveCurveIntersection;

            #[inline]
            fn intersect(&self, other: &TrimmedCurve3D<X>, tol: Tolerance) -> CurveCurveIntersection {
                route_curve_curve(self, other, tol)
            }
        }
    )*};
}

trimmed_curve_curve!(
    Line3D,
    Circle3D,
    Ellipse3D,
    Parabola3D,
    Hyperbola3D,
    BSplineCurve3D,
    Curve3D,
    ElementaryCurve
);

impl<X: CurveOperand, Y: CurveOperand> Intersect<TrimmedCurve3D<Y>> for TrimmedCurve3D<X>
where
    X::Basis: ExactCurveCurve<Y::Basis>,
{
    type Output = CurveCurveIntersection;

    #[inline]
    fn intersect(&self, other: &TrimmedCurve3D<Y>, tol: Tolerance) -> CurveCurveIntersection {
        route_curve_curve(self, other, tol)
    }
}

/// The exact intersection of two 2D bases before the parameter ranges are
/// applied: at most two hits, or a coincidence.
pub(crate) enum ExactCurveCurve2DIntersection {
    /// The hits; `None` marks an empty slot.
    Hits([Option<CurveCurveHit2D>; 2]),
    /// The curves coincide.
    Coincident,
}

/// The exact intersection of a 2D curve with a 2D curve of type `B`, solved
/// by the pair's typed solver, the parameters on both curves recovered once.
pub(crate) trait ExactCurveCurve2D<B> {
    /// Intersects `self` with `other`, deciding tangency and coincidence
    /// with `tol`.
    fn exact_curve_curve_2d(&self, other: &B, tol: Tolerance) -> ExactCurveCurve2DIntersection;
}

/// A hit read from the first curve, with the given parameters.
fn hit_2d(
    first_parameter: f64,
    second_parameter: f64,
    point: Point2D,
    multiplicity: u32,
) -> Option<CurveCurveHit2D> {
    Some(CurveCurveHit2D {
        first_parameter,
        second_parameter,
        point,
        multiplicity,
    })
}

/// The parameter of `point` on `circle`, snapped to the start of the bounds
/// at the seam.
#[inline]
fn on_circle(circle: &Circle2D, point: Point2D, tol: Tolerance) -> f64 {
    circle.parameter_in_bounds(circle.parameter_of(point), tol)
}

impl ExactCurveCurve2D<Line2D> for Line2D {
    #[inline]
    fn exact_curve_curve_2d(
        &self,
        other: &Line2D,
        tol: Tolerance,
    ) -> ExactCurveCurve2DIntersection {
        match self.intersect_exact(other, tol) {
            LineLine2DIntersection::Point(t, point, s) => {
                ExactCurveCurve2DIntersection::Hits([hit_2d(t, s, point, 1), None])
            }
            LineLine2DIntersection::Parallel => ExactCurveCurve2DIntersection::Hits([None, None]),
            LineLine2DIntersection::Coincident => ExactCurveCurve2DIntersection::Coincident,
        }
    }
}

impl ExactCurveCurve2D<Circle2D> for Line2D {
    #[inline]
    fn exact_curve_curve_2d(
        &self,
        circle: &Circle2D,
        tol: Tolerance,
    ) -> ExactCurveCurve2DIntersection {
        let at =
            |t, point, multiplicity| hit_2d(t, on_circle(circle, point, tol), point, multiplicity);
        ExactCurveCurve2DIntersection::Hits(match self.intersect_exact(circle, tol) {
            LineCircle2DIntersection::Points((t, p), (u, q)) => [at(t, p, 1), at(u, q, 1)],
            LineCircle2DIntersection::Tangent(t, p) => [at(t, p, 2), None],
            LineCircle2DIntersection::Empty => [None, None],
        })
    }
}

impl ExactCurveCurve2D<Line2D> for Circle2D {
    #[inline]
    fn exact_curve_curve_2d(&self, line: &Line2D, tol: Tolerance) -> ExactCurveCurve2DIntersection {
        let at =
            |t, point, multiplicity| hit_2d(on_circle(self, point, tol), t, point, multiplicity);
        ExactCurveCurve2DIntersection::Hits(match line.intersect_exact(self, tol) {
            LineCircle2DIntersection::Points((t, p), (u, q)) => [at(t, p, 1), at(u, q, 1)],
            LineCircle2DIntersection::Tangent(t, p) => [at(t, p, 2), None],
            LineCircle2DIntersection::Empty => [None, None],
        })
    }
}

impl ExactCurveCurve2D<Circle2D> for Circle2D {
    #[inline]
    fn exact_curve_curve_2d(
        &self,
        other: &Circle2D,
        tol: Tolerance,
    ) -> ExactCurveCurve2DIntersection {
        let at = |point, multiplicity| {
            hit_2d(
                on_circle(self, point, tol),
                on_circle(other, point, tol),
                point,
                multiplicity,
            )
        };
        ExactCurveCurve2DIntersection::Hits(match self.intersect_exact(other, tol) {
            CircleCircle2DIntersection::Points(p, q) => [at(p, 1), at(q, 1)],
            CircleCircle2DIntersection::Tangent(p) => [at(p, 2), None],
            CircleCircle2DIntersection::Empty => [None, None],
            CircleCircle2DIntersection::Coincident => {
                return ExactCurveCurve2DIntersection::Coincident;
            }
        })
    }
}

/// Implements [`ExactCurveCurve2D`] between each concrete 2D curve and the
/// untyped [`Curve2D`], in both orders, by dispatching on the untyped
/// operand.
macro_rules! exact_curve_curve_2d_untyped {
    ($($curve:ty),*) => {$(
        impl ExactCurveCurve2D<Curve2D> for $curve {
            #[inline]
            fn exact_curve_curve_2d(
                &self,
                other: &Curve2D,
                tol: Tolerance,
            ) -> ExactCurveCurve2DIntersection {
                match other {
                    Curve2D::Line(b) => self.exact_curve_curve_2d(b, tol),
                    Curve2D::Circle(b) => self.exact_curve_curve_2d(b, tol),
                }
            }
        }

        impl ExactCurveCurve2D<$curve> for Curve2D {
            #[inline]
            fn exact_curve_curve_2d(
                &self,
                other: &$curve,
                tol: Tolerance,
            ) -> ExactCurveCurve2DIntersection {
                match self {
                    Curve2D::Line(a) => a.exact_curve_curve_2d(other, tol),
                    Curve2D::Circle(a) => a.exact_curve_curve_2d(other, tol),
                }
            }
        }
    )*};
}

exact_curve_curve_2d_untyped!(Line2D, Circle2D);

impl ExactCurveCurve2D<Curve2D> for Curve2D {
    #[inline]
    fn exact_curve_curve_2d(
        &self,
        other: &Curve2D,
        tol: Tolerance,
    ) -> ExactCurveCurve2DIntersection {
        match self {
            Curve2D::Line(a) => a.exact_curve_curve_2d(other, tol),
            Curve2D::Circle(a) => a.exact_curve_curve_2d(other, tol),
        }
    }
}

/// A 2D curve the router solves exactly against every 2D basis: the
/// concrete curves are their own basis, [`Curve2D`] the basis of a curve
/// whose kind is only known at run time.
pub(crate) trait CurveBasis2D:
    ParametricCurve2D
    + Copy
    + Into<Curve2D>
    + ExactCurveCurve2D<Line2D>
    + ExactCurveCurve2D<Circle2D>
    + ExactCurveCurve2D<Curve2D>
{
}

impl CurveBasis2D for Line2D {}
impl CurveBasis2D for Circle2D {}
impl CurveBasis2D for Curve2D {}

/// How a 2D curve enters the router: its basis and the parameter range it
/// is restricted to (`None` = the basis's own bounds).
pub(crate) trait CurveOperand2D: ParametricCurve2D {
    /// The basis type: the curve's own type when it is concrete, so the
    /// pair is solved by its typed solver.
    type Basis: CurveBasis2D;

    /// The basis and the restricting parameter range.
    fn basis_and_range(&self) -> (Self::Basis, Option<(f64, f64)>);
}

macro_rules! basis_operand_2d {
    ($($curve:ty),*) => {$(
        impl CurveOperand2D for $curve {
            type Basis = $curve;

            #[inline]
            fn basis_and_range(&self) -> ($curve, Option<(f64, f64)>) {
                (*self, None)
            }
        }
    )*};
}

basis_operand_2d!(Line2D, Circle2D, Curve2D);

impl<C: CurveOperand2D> CurveOperand2D for TrimmedCurve2D<C> {
    type Basis = C::Basis;

    /// A trimmed curve of a trimmed curve keeps the inner basis and the
    /// outer range: the outer range lies within the inner one.
    #[inline]
    fn basis_and_range(&self) -> (C::Basis, Option<(f64, f64)>) {
        let (basis, _) = self.basis().basis_and_range();
        (basis, Some((self.first(), self.last())))
    }
}

/// Intersects two 2D curves. Every pair has a closed form, so the result
/// is exact and complete.
#[inline]
pub(crate) fn route_curve_curve_2d<A: CurveOperand2D, B: CurveOperand2D>(
    a: &A,
    b: &B,
    tol: Tolerance,
) -> CurveCurveIntersection2D
where
    A::Basis: ExactCurveCurve2D<B::Basis>,
{
    let (first, first_range) = a.basis_and_range();
    let (second, second_range) = b.basis_and_range();
    match first.exact_curve_curve_2d(&second, tol) {
        ExactCurveCurve2DIntersection::Coincident => {
            coincidence_2d(first.into(), first_range, second.into(), second_range, tol)
        }
        ExactCurveCurve2DIntersection::Hits(hits) => {
            let kept = hits.into_iter().flatten().filter_map(|hit| {
                Some(CurveCurveHit2D {
                    first_parameter: representative_in(
                        hit.first_parameter,
                        first_range,
                        first.period(),
                        tol,
                    )?,
                    second_parameter: representative_in(
                        hit.second_parameter,
                        second_range,
                        second.period(),
                        tol,
                    )?,
                    ..hit
                })
            });
            CurveCurveIntersection2D::exact(kept, CurveCurveOverlapList::new())
        }
    }
}

/// The result for two coincident 2D bases restricted to their ranges.
fn coincidence_2d(
    first: Curve2D,
    first_range: Option<(f64, f64)>,
    second: Curve2D,
    second_range: Option<(f64, f64)>,
    tol: Tolerance,
) -> CurveCurveIntersection2D {
    let first_range = first_range.unwrap_or(first.bounds());
    let second_range = second_range.unwrap_or(second.bounds());
    let shared = coincident_overlaps((&first, first_range), (&second, second_range), tol);
    let hits = shared
        .touches
        .iter()
        .map(|&(first_parameter, second_parameter)| CurveCurveHit2D {
            first_parameter,
            second_parameter,
            point: first.eval_point(first_parameter),
            multiplicity: 1,
        });
    CurveCurveIntersection2D::exact(hits, shared.overlaps)
}

/// Implements [`Intersect`] for every 2D curve of the list with every one
/// of the list, the receiver read first.
macro_rules! curve_curve_2d {
    (@row $first:ty; [$($second:ty),*]) => {$(
        impl Intersect<$second> for $first {
            type Output = CurveCurveIntersection2D;

            #[inline]
            fn intersect(&self, other: &$second, tol: Tolerance) -> CurveCurveIntersection2D {
                route_curve_curve_2d(self, other, tol)
            }
        }
    )*};
    ([$($first:ty),*]) => {
        curve_curve_2d!(@all [$($first),*]; [$($first),*]);
    };
    (@all [$($first:ty),*]; $seconds:tt) => {$(
        curve_curve_2d!(@row $first; $seconds);
    )*};
}

curve_curve_2d!([Line2D, Circle2D, Curve2D]);

/// Implements [`Intersect`] between a trimmed 2D curve over any operand and
/// each 2D curve of the list, in both orders.
macro_rules! trimmed_curve_curve_2d {
    ($($curve:ty),*) => {$(
        impl<X: CurveOperand2D> Intersect<$curve> for TrimmedCurve2D<X> {
            type Output = CurveCurveIntersection2D;

            #[inline]
            fn intersect(&self, other: &$curve, tol: Tolerance) -> CurveCurveIntersection2D {
                route_curve_curve_2d(self, other, tol)
            }
        }

        impl<X: CurveOperand2D> Intersect<TrimmedCurve2D<X>> for $curve
        where
            <$curve as CurveOperand2D>::Basis: ExactCurveCurve2D<X::Basis>,
        {
            type Output = CurveCurveIntersection2D;

            #[inline]
            fn intersect(&self, other: &TrimmedCurve2D<X>, tol: Tolerance) -> CurveCurveIntersection2D {
                route_curve_curve_2d(self, other, tol)
            }
        }
    )*};
}

trimmed_curve_curve_2d!(Line2D, Circle2D, Curve2D);

impl<X: CurveOperand2D, Y: CurveOperand2D> Intersect<TrimmedCurve2D<Y>> for TrimmedCurve2D<X>
where
    X::Basis: ExactCurveCurve2D<Y::Basis>,
{
    type Output = CurveCurveIntersection2D;

    #[inline]
    fn intersect(&self, other: &TrimmedCurve2D<Y>, tol: Tolerance) -> CurveCurveIntersection2D {
        route_curve_curve_2d(self, other, tol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;
    const TOL: Tolerance = Tolerance::DEFAULT;

    #[test]
    fn test_in_range_plain() {
        assert_eq!(in_range(0.5, (0.0, 1.0), None, TOL), Some(0.5));
        assert_eq!(in_range(1.5, (0.0, 1.0), None, TOL), None);
        assert_eq!(
            in_range(1.0 + 1e-12, (0.0, 1.0), None, TOL),
            Some(1.0 + 1e-12)
        );
        assert_eq!(in_range(-1e-12, (0.0, 1.0), None, TOL), Some(-1e-12));
    }

    #[test]
    fn test_in_range_wraps_across_the_seam() {
        let shifted = in_range(6.0, (-1.0, 1.0), Some(TAU), TOL).unwrap();
        assert!((shifted - (6.0 - TAU)).abs() < 1e-15);
        assert_eq!(in_range(3.0, (-1.0, 1.0), Some(TAU), TOL), None);
    }

    #[test]
    fn test_in_range_accepts_just_below_the_start_on_a_periodic_curve() {
        let found = in_range(-1.0 - 1e-12, (-1.0, 1.0), Some(TAU), TOL).unwrap();
        assert!((found - (-1.0 - 1e-12)).abs() < 1e-12);
        assert_eq!(in_range(-1.0 - 1e-3, (-1.0, 1.0), Some(TAU), TOL), None);
    }

    #[test]
    fn test_clip_range() {
        assert_eq!(
            *clip_range((f64::NEG_INFINITY, f64::INFINITY), (2.0, 3.0), None, TOL),
            [(2.0, 3.0)]
        );
        assert_eq!(
            *clip_range((0.0, TAU), (-1.0, 1.0), Some(TAU), TOL),
            [(-1.0, 1.0)]
        );
        assert!(clip_range((0.0, 1.0), (2.0, 3.0), None, TOL).is_empty());
    }

    #[test]
    fn test_clip_range_shifts_a_partial_overlap_across_the_seam() {
        let pieces = clip_range((6.0, 6.0 + 0.5), (-1.0, 1.0), Some(TAU), TOL);
        assert_eq!(pieces.len(), 1);
        let (low, high) = pieces[0];
        assert!((low - (6.0 - TAU)).abs() < 1e-15);
        assert!((high - (6.5 - TAU)).abs() < 1e-15);
        assert!(clip_range((3.0, 4.0), (-1.0, 1.0), Some(TAU), TOL).is_empty());
    }

    #[test]
    fn test_clip_range_returns_both_pieces_across_the_seam() {
        // Arcs [0, 5] and [4, 7] of one circle share [4, 5] and [0, 7 - TAU].
        let pieces = clip_range((4.0, 7.0), (0.0, 5.0), Some(TAU), TOL);
        assert_eq!(pieces.len(), 2);
        assert!((pieces[0].0 - 0.0).abs() < 1e-15 && (pieces[0].1 - (7.0 - TAU)).abs() < 1e-15);
        assert_eq!(pieces[1], (4.0, 5.0));
    }

    #[test]
    fn test_clip_range_keeps_a_touch_as_a_single_point() {
        assert_eq!(*clip_range((1.0, 2.0), (0.0, 1.0), None, TOL), [(1.0, 1.0)]);
        let seam = clip_range((0.0, 1.0), (5.0, TAU), Some(TAU), TOL);
        assert_eq!(*seam, [(TAU, TAU)]);
        assert!(is_single_point(seam[0], TOL));
    }

    #[test]
    fn test_clip_range_drops_a_touch_repeating_the_end_of_a_piece() {
        // The copy shifted by a period touches the full range at TAU, the
        // same point as the start of the real piece.
        assert_eq!(
            *clip_range((0.0, 1.0), (0.0, TAU), Some(TAU), TOL),
            [(0.0, 1.0)]
        );
    }

    #[test]
    fn test_trimmed_operand_reports_basis_and_range() {
        let circle = crate::Circle3D::new(crate::Point3D::ORIGIN, crate::Vector3D::Z, 1.0).unwrap();
        let arc = crate::curves::TrimmedCurve3D::new(circle, -1.0, 1.0).unwrap();
        let (basis, range) = arc.elementary_basis().unwrap();
        // A trimmed circle routes through the typed circle solver.
        assert_eq!(basis, circle);
        assert_eq!(range, Some((-1.0, 1.0)));
    }

    #[test]
    fn test_trimmed_of_trimmed_keeps_the_outer_range() {
        let line = crate::Line3D::new(crate::Point3D::ORIGIN, crate::Vector3D::X).unwrap();
        let outer =
            TrimmedCurve3D::new(TrimmedCurve3D::new(line, 0.0, 10.0).unwrap(), 2.0, 3.0).unwrap();
        assert_eq!(outer.elementary_basis().unwrap().1, Some((2.0, 3.0)));
        let spline_free: Curve3D = line.into();
        assert_eq!(spline_free.elementary_basis().unwrap().1, None);
    }
}
