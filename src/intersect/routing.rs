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
    BSplineCurve3D, Circle3D, Curve3D, ElementaryCurve, Ellipse3D, Hyperbola3D, Line3D, Parabola3D,
    ParametricCurve3D, TrimmedCurve3D,
};
use crate::intersect::{
    CurveSurfaceHit, CurveSurfaceIntersection, ExactCurveSurfaceIntersection, IntersectExact,
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

/// Clips an overlap `(a, b)` to `range`, shifting by periods like
/// [`in_range`].
///
/// Returns the pieces of the overlap that lie in the range beyond
/// `tol.parametric`, ordered by start, each expressed in the range's
/// representation: none, one, or two when the overlap meets the range on
/// both sides of the seam of a periodic curve. An overlap covering the
/// whole period returns the range as one piece.
// Consumed by the curve-curve routing (Task 6), where a coincidence of two
// curves is clipped to each curve's range.
#[allow(dead_code)]
pub(crate) fn clip_range(
    overlap: (f64, f64),
    range: (f64, f64),
    period: Option<f64>,
    tol: Tolerance,
) -> Vec<(f64, f64)> {
    let clip = |(low, high): (f64, f64)| {
        let clipped = (low.max(range.0), high.min(range.1));
        (clipped.1 - clipped.0 > tol.parametric).then_some(clipped)
    };
    let Some(period) = period else {
        return clip(overlap).into_iter().collect();
    };
    if overlap.1 - overlap.0 >= period - tol.parametric {
        return vec![range];
    }
    let first_copy = ((range.0 - overlap.1) / period).floor();
    let mut pieces: Vec<(f64, f64)> = (0..3)
        .filter_map(|step| {
            let shift = (first_copy + f64::from(step)) * period;
            clip((overlap.0 + shift, overlap.1 + shift))
        })
        .collect();
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
fn from_exact_curve_surface(
    exact: ExactCurveSurfaceIntersection,
    basis: &ElementaryCurve,
    range: Option<(f64, f64)>,
    elementary: &ElementarySurface,
    tol: Tolerance,
) -> CurveSurfaceIntersection {
    match exact {
        ExactCurveSurfaceIntersection::Points(list) => {
            let hits = list
                .iter()
                .filter_map(|hit| {
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
                })
                .collect();
            CurveSurfaceIntersection::exact(hits, Vec::new())
        }
        // The whole curve lies on the surface: a trimmed curve over its
        // range, the basis over its bounds (infinite for a line).
        ExactCurveSurfaceIntersection::Coincident => {
            CurveSurfaceIntersection::exact(Vec::new(), vec![range.unwrap_or(basis.bounds())])
        }
        ExactCurveSurfaceIntersection::Empty => {
            CurveSurfaceIntersection::exact(Vec::new(), Vec::new())
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
            clip_range((f64::NEG_INFINITY, f64::INFINITY), (2.0, 3.0), None, TOL),
            vec![(2.0, 3.0)]
        );
        assert_eq!(
            clip_range((0.0, TAU), (-1.0, 1.0), Some(TAU), TOL),
            vec![(-1.0, 1.0)]
        );
        assert_eq!(clip_range((0.0, 1.0), (2.0, 3.0), None, TOL), vec![]);
    }

    #[test]
    fn test_clip_range_shifts_a_partial_overlap_across_the_seam() {
        let pieces = clip_range((6.0, 6.0 + 0.5), (-1.0, 1.0), Some(TAU), TOL);
        assert_eq!(pieces.len(), 1);
        let (low, high) = pieces[0];
        assert!((low - (6.0 - TAU)).abs() < 1e-15);
        assert!((high - (6.5 - TAU)).abs() < 1e-15);
        assert_eq!(clip_range((3.0, 4.0), (-1.0, 1.0), Some(TAU), TOL), vec![]);
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
