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

use crate::Point2D;
use crate::Tolerance;
use crate::curves::{
    BSplineCurve3D, Circle2D, Circle3D, Curve2D, Curve3D, ElementaryCurve, Ellipse3D, Hyperbola3D,
    Line2D, Line3D, Parabola3D, ParametricCurve2D, ParametricCurve3D, TrimmedCurve2D,
    TrimmedCurve3D,
};
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

/// How a curve enters the router: its elementary basis (when it has one)
/// and the parameter range it is restricted to (`None` = the basis's own
/// bounds).
pub(crate) trait CurveOperand: ParametricCurve3D {
    /// The elementary basis and the restricting parameter range, or `None`
    /// when the curve has no elementary basis.
    fn elementary_basis(&self) -> Option<(ElementaryCurve, Option<(f64, f64)>)>;
}

macro_rules! elementary_operand {
    ($($curve:ty),*) => {$(
        impl CurveOperand for $curve {
            fn elementary_basis(&self) -> Option<(ElementaryCurve, Option<(f64, f64)>)> {
                Some(((*self).into(), None))
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
    fn elementary_basis(&self) -> Option<(ElementaryCurve, Option<(f64, f64)>)> {
        None
    }
}

impl CurveOperand for Curve3D {
    fn elementary_basis(&self) -> Option<(ElementaryCurve, Option<(f64, f64)>)> {
        self.as_elementary().map(|basis| (basis, None))
    }
}

impl<C: CurveOperand> CurveOperand for TrimmedCurve3D<C> {
    /// A trimmed curve of a trimmed curve keeps the inner basis and the
    /// outer range: the outer range lies within the inner one.
    fn elementary_basis(&self) -> Option<(ElementaryCurve, Option<(f64, f64)>)> {
        self.basis()
            .elementary_basis()
            .map(|(basis, _)| (basis, Some((self.first(), self.last()))))
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
/// Returns the pieces of the overlap that lie in the range beyond
/// `tol.parametric`, ordered by start, each expressed in the range's
/// representation: none, one, or two when the overlap meets the range on
/// both sides of the seam of a periodic curve. An overlap covering the
/// whole period returns the range as one piece. The three slots hold the
/// three shifted copies the search tries, so the list never fills up.
pub(crate) fn clip_range(
    overlap: (f64, f64),
    range: (f64, f64),
    period: Option<f64>,
    tol: Tolerance,
) -> InlineList<(f64, f64), 3> {
    let clip = |(low, high): (f64, f64)| {
        let clipped = (low.max(range.0), high.min(range.1));
        (clipped.1 - clipped.0 > tol.parametric).then_some(clipped)
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
    for step in 0..3 {
        let shift = (first_copy + f64::from(step)) * period;
        if let Some(piece) = clip((overlap.0 + shift, overlap.1 + shift)) {
            pieces.push(piece);
        }
    }
    pieces.sort_by(|a, b| a.0.total_cmp(&b.0));
    pieces
}

/// Intersects a curve with a surface: the exact path when the curve has an
/// elementary basis, the surface is elementary and the pair has a closed
/// form; the numeric search otherwise.
#[inline]
pub(crate) fn route_curve_surface<C: CurveOperand, S: ParametricSurface>(
    curve: &C,
    surface: &S,
    tol: Tolerance,
) -> CurveSurfaceIntersection {
    if let (Some((basis, range)), Some(elementary)) =
        (curve.elementary_basis(), surface.as_elementary())
    {
        let exact = basis.intersect_exact(&elementary, tol);
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
    basis: &ElementaryCurve,
    range: Option<(f64, f64)>,
    elementary: &ElementarySurface,
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
                    surface_parameters: elementary.parameters_of(hit.point),
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
) -> CurveCurveIntersection {
    if let (Some(first), Some(second)) = (a.elementary_basis(), b.elementary_basis()) {
        let exact = first.0.intersect_exact(&second.0, tol);
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
type Operand = (ElementaryCurve, Option<(f64, f64)>);

/// Turns the exact intersection of two elementary bases into the routed
/// result: hits outside either range dropped, the kept ones carrying their
/// representative in each range, and a coincidence clipped to both ranges.
#[inline]
fn from_exact_curve_curve(
    exact: ExactCurveCurveIntersection,
    (first, first_range): Operand,
    (second, second_range): Operand,
    tol: Tolerance,
) -> CurveCurveIntersection {
    let representative = |parameter, curve: &ElementaryCurve, range| {
        representative_in(parameter, range, curve.period(), tol)
    };
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
                    first_parameter: representative(hit.first_parameter, &first, first_range)?,
                    second_parameter: representative(hit.second_parameter, &second, second_range)?,
                    ..*hit
                })
            });
            CurveCurveIntersection::exact(hits.collect(), CurveCurveOverlapList::new())
        }
        ExactCurveCurveIntersection::Coincident => {
            let first_range = first_range.unwrap_or(first.bounds());
            let second_range = second_range.unwrap_or(second.bounds());
            let overlaps = coincident_overlaps((&first, first_range), (&second, second_range), tol);
            CurveCurveIntersection::exact(CurveCurveHitList::new(), overlaps)
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
fn coincident_overlaps<C: CoincidentCurve>(
    (first, first_range): (&C, (f64, f64)),
    (second, second_range): (&C, (f64, f64)),
    tol: Tolerance,
) -> CurveCurveOverlapList {
    let same_direction = runs_along(second, second_range, first);
    let on_first = mapped_range(second, second_range, first, same_direction, tol);
    let mut overlaps = CurveCurveOverlapList::new();
    for &piece in clip_range(on_first, first_range, first.cycle_length(), tol).iter() {
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
    overlaps
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

        impl<X: CurveOperand> Intersect<TrimmedCurve3D<X>> for $curve {
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

impl<X: CurveOperand, Y: CurveOperand> Intersect<TrimmedCurve3D<Y>> for TrimmedCurve3D<X> {
    type Output = CurveCurveIntersection;

    #[inline]
    fn intersect(&self, other: &TrimmedCurve3D<Y>, tol: Tolerance) -> CurveCurveIntersection {
        route_curve_curve(self, other, tol)
    }
}

/// How a 2D curve enters the router: its basis and the parameter range it
/// is restricted to (`None` = the basis's own bounds).
pub(crate) trait CurveOperand2D: ParametricCurve2D {
    /// The basis and the restricting parameter range.
    fn basis_and_range(&self) -> (Curve2D, Option<(f64, f64)>);
}

macro_rules! basis_operand_2d {
    ($($curve:ty),*) => {$(
        impl CurveOperand2D for $curve {
            fn basis_and_range(&self) -> (Curve2D, Option<(f64, f64)>) {
                ((*self).into(), None)
            }
        }
    )*};
}

basis_operand_2d!(Line2D, Circle2D, Curve2D);

impl<C: CurveOperand2D> CurveOperand2D for TrimmedCurve2D<C> {
    /// A trimmed curve of a trimmed curve keeps the inner basis and the
    /// outer range: the outer range lies within the inner one.
    fn basis_and_range(&self) -> (Curve2D, Option<(f64, f64)>) {
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
) -> CurveCurveIntersection2D {
    let (first, first_range) = a.basis_and_range();
    let (second, second_range) = b.basis_and_range();
    let representative = |parameter, curve: &Curve2D, range| {
        representative_in(parameter, range, curve.cycle_length(), tol)
    };
    let hit = |first_parameter, second_parameter, point: Point2D, multiplicity| {
        Some(CurveCurveHit2D {
            first_parameter: representative(first_parameter, &first, first_range)?,
            second_parameter: representative(second_parameter, &second, second_range)?,
            point,
            multiplicity,
        })
    };
    // At most two hits; `None` marks an empty slot or a hit outside a range.
    let hits: [Option<CurveCurveHit2D>; 2] = match (&first, &second) {
        (Curve2D::Line(line), Curve2D::Line(other)) => match line.intersect_exact(other, tol) {
            LineLine2DIntersection::Point(t, point, s) => [hit(t, s, point, 1), None],
            LineLine2DIntersection::Parallel => [None, None],
            LineLine2DIntersection::Coincident => {
                return coincidence_2d(first, first_range, second, second_range, tol);
            }
        },
        (Curve2D::Line(line), Curve2D::Circle(circle)) => {
            let on_circle = |point| circle.parameter_of(point);
            match line.intersect_exact(circle, tol) {
                LineCircle2DIntersection::Points((t, p), (u, q)) => {
                    [hit(t, on_circle(p), p, 1), hit(u, on_circle(q), q, 1)]
                }
                LineCircle2DIntersection::Tangent(t, p) => [hit(t, on_circle(p), p, 2), None],
                LineCircle2DIntersection::Empty => [None, None],
            }
        }
        (Curve2D::Circle(circle), Curve2D::Line(line)) => {
            let on_circle = |point| circle.parameter_of(point);
            match line.intersect_exact(circle, tol) {
                LineCircle2DIntersection::Points((t, p), (u, q)) => {
                    [hit(on_circle(p), t, p, 1), hit(on_circle(q), u, q, 1)]
                }
                LineCircle2DIntersection::Tangent(t, p) => [hit(on_circle(p), t, p, 2), None],
                LineCircle2DIntersection::Empty => [None, None],
            }
        }
        (Curve2D::Circle(circle), Curve2D::Circle(other)) => {
            let both = |point: Point2D, multiplicity| {
                hit(
                    circle.parameter_of(point),
                    other.parameter_of(point),
                    point,
                    multiplicity,
                )
            };
            match circle.intersect_exact(other, tol) {
                CircleCircle2DIntersection::Points(p, q) => [both(p, 1), both(q, 1)],
                CircleCircle2DIntersection::Tangent(p) => [both(p, 2), None],
                CircleCircle2DIntersection::Empty => [None, None],
                CircleCircle2DIntersection::Coincident => {
                    return coincidence_2d(first, first_range, second, second_range, tol);
                }
            }
        }
    };
    CurveCurveIntersection2D::exact(hits.into_iter().flatten(), CurveCurveOverlapList::new())
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
    let overlaps = coincident_overlaps((&first, first_range), (&second, second_range), tol);
    CurveCurveIntersection2D::exact([], overlaps)
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

        impl<X: CurveOperand2D> Intersect<TrimmedCurve2D<X>> for $curve {
            type Output = CurveCurveIntersection2D;

            #[inline]
            fn intersect(&self, other: &TrimmedCurve2D<X>, tol: Tolerance) -> CurveCurveIntersection2D {
                route_curve_curve_2d(self, other, tol)
            }
        }
    )*};
}

trimmed_curve_curve_2d!(Line2D, Circle2D, Curve2D);

impl<X: CurveOperand2D, Y: CurveOperand2D> Intersect<TrimmedCurve2D<Y>> for TrimmedCurve2D<X> {
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
    fn test_trimmed_operand_reports_basis_and_range() {
        let circle = crate::Circle3D::new(crate::Point3D::ORIGIN, crate::Vector3D::Z, 1.0).unwrap();
        let arc = crate::curves::TrimmedCurve3D::new(circle, -1.0, 1.0).unwrap();
        let (basis, range) = arc.elementary_basis().unwrap();
        assert!(matches!(basis, ElementaryCurve::Circle(_)));
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
