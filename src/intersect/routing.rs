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

// The helpers are consumed by the pair implementations that follow.
#![allow(dead_code)]

use crate::Tolerance;
use crate::curves::{
    BSplineCurve3D, Circle3D, Curve3D, ElementaryCurve, Ellipse3D, Hyperbola3D, Line3D, Parabola3D,
    ParametricCurve3D, TrimmedCurve3D,
};
#[cfg(doc)]
use crate::intersect::IntersectExact;

/// Routed intersection: exact when a closed form exists, numeric otherwise.
///
/// Unlike [`IntersectExact`], the typed exact path that never runs numeric
/// work, `intersect` works for every supported pair, and its result says which
/// method ran (`method()`) and whether it is complete.
///
/// ```ignore
/// // Becomes a real doctest once the pair implementations land.
/// use geomcore::{Intersect, Line3D, Plane, Point3D, Tolerance, Vector3D};
///
/// let line = Line3D::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
/// let plane = Plane::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Z).unwrap();
/// let result = line.intersect(&plane, Tolerance::DEFAULT);
/// assert!(result.is_complete());
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
/// Returns `None` when they do not intersect beyond `tol.parametric`. On a
/// periodic curve an overlap covering the whole period returns `range`;
/// when the overlap meets the range in two pieces across the seam, the
/// longer one is returned.
pub(crate) fn clip_range(
    overlap: (f64, f64),
    range: (f64, f64),
    period: Option<f64>,
    tol: Tolerance,
) -> Option<(f64, f64)> {
    let clip = |(low, high): (f64, f64)| {
        let clipped = (low.max(range.0), high.min(range.1));
        (clipped.1 - clipped.0 > tol.parametric).then_some(clipped)
    };
    let Some(period) = period else {
        return clip(overlap);
    };
    if overlap.1 - overlap.0 >= period - tol.parametric {
        return Some(range);
    }
    let first_copy = ((range.0 - overlap.1) / period).floor();
    (0..3)
        .filter_map(|step| {
            let shift = (first_copy + f64::from(step)) * period;
            clip((overlap.0 + shift, overlap.1 + shift))
        })
        .max_by(|a, b| (a.1 - a.0).total_cmp(&(b.1 - b.0)))
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
            clip_range((f64::NEG_INFINITY, f64::INFINITY), (2.0, 3.0), None, TOL),
            Some((2.0, 3.0))
        );
        assert_eq!(
            clip_range((0.0, TAU), (-1.0, 1.0), Some(TAU), TOL),
            Some((-1.0, 1.0))
        );
        assert_eq!(clip_range((0.0, 1.0), (2.0, 3.0), None, TOL), None);
    }

    #[test]
    fn test_clip_range_shifts_a_partial_overlap_across_the_seam() {
        let (low, high) = clip_range((6.0, 6.0 + 0.5), (-1.0, 1.0), Some(TAU), TOL).unwrap();
        assert!((low - (6.0 - TAU)).abs() < 1e-15);
        assert!((high - (6.5 - TAU)).abs() < 1e-15);
        assert_eq!(clip_range((3.0, 4.0), (-1.0, 1.0), Some(TAU), TOL), None);
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
