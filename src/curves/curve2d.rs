//! [`ParametricCurve2D`] trait and the [`Curve2D`] enum adaptor unifying the
//! concrete 2D curve types.

use crate::curves::{Circle2D, Line2D};
use crate::{Point2, Vector2};
use std::f64::consts::TAU;

/// Common interface for 2D parametric curves. Implement this to add a new
/// curve type that can be used generically (see [`Curve2D`] for a
/// heterogeneous-collection adaptor built on top of it).
///
/// Mirrors [`crate::curves::ParametricCurve3D`] with [`Point2`]/[`Vector2`]
/// in place of `Point3`/`Vector3`. Concrete curve types (e.g. [`Line2D`],
/// [`Circle2D`]) already provide inherent methods with the same names; their
/// trait impls simply delegate to those inherent methods, so calling code
/// can use either form.
///
/// # Examples
///
/// The default [`eval_points`](ParametricCurve2D::eval_points) maps
/// [`eval_point`](ParametricCurve2D::eval_point) over each parameter:
///
/// ```
/// use geomcore::curves::ParametricCurve2D;
/// use geomcore::{Line2D, Point2, Vector2};
///
/// let line = Line2D::new(Point2::ORIGIN, Vector2::X).unwrap();
/// let us = [0.0, 1.0, 2.0];
/// let expected: Vec<Point2> = us.iter().map(|&u| line.eval_point(u)).collect();
/// assert_eq!(ParametricCurve2D::eval_points(&line, &us), expected);
/// ```
///
/// The default [`is_periodic`](ParametricCurve2D::is_periodic) reports
/// whether [`period`](ParametricCurve2D::period) is `Some`:
///
/// ```
/// use geomcore::curves::ParametricCurve2D;
/// use geomcore::{Circle2D, Line2D, Point2};
///
/// let line = Line2D::new(Point2::ORIGIN, geomcore::Vector2::X).unwrap();
/// assert!(!ParametricCurve2D::is_periodic(&line));
///
/// let circle = Circle2D::new(Point2::ORIGIN, 1.0).unwrap();
/// assert!(ParametricCurve2D::is_periodic(&circle));
/// ```
pub trait ParametricCurve2D {
    /// Evaluates the point on the curve at parameter `u`.
    fn eval_point(&self, u: f64) -> Point2;

    /// Evaluates the derivative of the given `order` at parameter `u`.
    fn eval_derivative(&self, u: f64, order: u32) -> Vector2;

    /// Returns the curve's parameter bounds as `(first, last)`. Unbounded
    /// curves use `(f64::NEG_INFINITY, f64::INFINITY)`.
    fn bounds(&self) -> (f64, f64);

    /// Returns the curve's period, or `None` if the curve is not periodic.
    fn period(&self) -> Option<f64>;

    /// Evaluates the points on the curve at each parameter in `us`.
    ///
    /// Default implementation: maps [`eval_point`](Self::eval_point) over
    /// `us`.
    fn eval_points(&self, us: &[f64]) -> Vec<Point2> {
        us.iter().map(|&u| self.eval_point(u)).collect()
    }

    /// Returns whether the curve is periodic, i.e. whether
    /// [`period`](Self::period) is `Some`.
    fn is_periodic(&self) -> bool {
        self.period().is_some()
    }
}

impl ParametricCurve2D for Line2D {
    fn eval_point(&self, u: f64) -> Point2 {
        Line2D::eval_point(self, u)
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector2 {
        Line2D::eval_derivative(self, u, order)
    }

    fn bounds(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn period(&self) -> Option<f64> {
        None
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point2> {
        Line2D::eval_points(self, us)
    }
}

impl ParametricCurve2D for Circle2D {
    fn eval_point(&self, u: f64) -> Point2 {
        Circle2D::eval_point(self, u)
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector2 {
        Circle2D::eval_derivative(self, u, order)
    }

    fn bounds(&self) -> (f64, f64) {
        (0.0, TAU)
    }

    fn period(&self) -> Option<f64> {
        Some(TAU)
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point2> {
        Circle2D::eval_points(self, us)
    }
}

/// Enum adaptor unifying the concrete 2D curve types behind a single value
/// type, so callers can build a heterogeneous collection of curves and
/// evaluate them uniformly through [`ParametricCurve2D`].
///
/// Marked `#[non_exhaustive]`: future tasks add further variants without
/// that being a breaking change.
///
/// # Examples
///
/// ```
/// use geomcore::curves::{Curve2D, ParametricCurve2D};
/// use geomcore::{Circle2D, Line2D, Point2, Vector2};
///
/// let curves: Vec<Curve2D> = vec![
///     Line2D::new(Point2::ORIGIN, Vector2::X).unwrap().into(),
///     Circle2D::new(Point2::ORIGIN, 1.0).unwrap().into(),
/// ];
/// let points: Vec<Point2> = curves.iter().map(|c| c.eval_point(0.0)).collect();
/// assert_eq!(points[0], Point2::ORIGIN);
/// assert_eq!(points[1], Point2::new(1.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Curve2D {
    /// A [`Line2D`] curve.
    Line(Line2D),
    /// A [`Circle2D`] curve.
    Circle(Circle2D),
}

impl ParametricCurve2D for Curve2D {
    fn eval_point(&self, u: f64) -> Point2 {
        match self {
            Curve2D::Line(c) => c.eval_point(u),
            Curve2D::Circle(c) => c.eval_point(u),
        }
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector2 {
        match self {
            Curve2D::Line(c) => c.eval_derivative(u, order),
            Curve2D::Circle(c) => c.eval_derivative(u, order),
        }
    }

    fn bounds(&self) -> (f64, f64) {
        match self {
            Curve2D::Line(c) => c.bounds(),
            Curve2D::Circle(c) => c.bounds(),
        }
    }

    fn period(&self) -> Option<f64> {
        match self {
            Curve2D::Line(c) => c.period(),
            Curve2D::Circle(c) => c.period(),
        }
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point2> {
        match self {
            Curve2D::Line(c) => c.eval_points(us),
            Curve2D::Circle(c) => c.eval_points(us),
        }
    }
}

impl From<Line2D> for Curve2D {
    /// Wraps a [`Line2D`] as a [`Curve2D::Line`].
    fn from(curve: Line2D) -> Curve2D {
        Curve2D::Line(curve)
    }
}

impl From<Circle2D> for Curve2D {
    /// Wraps a [`Circle2D`] as a [`Curve2D::Circle`].
    fn from(curve: Circle2D) -> Curve2D {
        Curve2D::Circle(curve)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample<C: ParametricCurve2D>(c: &C) -> Point2 {
        c.eval_point(0.5)
    }

    fn line() -> Line2D {
        Line2D::new(Point2::ORIGIN, Vector2::X).unwrap()
    }

    fn circle() -> Circle2D {
        Circle2D::new(Point2::ORIGIN, 2.0).unwrap()
    }

    // ---- generic fn agrees with inherent calls ----

    #[test]
    fn test_sample_line_agrees_with_inherent() {
        let l = line();
        assert_eq!(sample(&l), l.eval_point(0.5));
    }

    #[test]
    fn test_sample_circle_agrees_with_inherent() {
        let c = circle();
        assert_eq!(sample(&c), c.eval_point(0.5));
    }

    #[test]
    fn test_sample_curve2d_enum_agrees_with_inherent() {
        let l = line();
        let wrapped: Curve2D = l.into();
        assert_eq!(sample(&wrapped), l.eval_point(0.5));
    }

    // ---- heterogeneous Vec<Curve2D> eval ----

    #[test]
    fn test_heterogeneous_vec_eval() {
        let l = line();
        let c = circle();
        let curves: Vec<Curve2D> = vec![l.into(), c.into()];
        let points: Vec<Point2> = curves.iter().map(|curve| curve.eval_point(0.25)).collect();
        assert_eq!(points[0], l.eval_point(0.25));
        assert_eq!(points[1], c.eval_point(0.25));
    }

    // ---- From round-trips ----

    #[test]
    fn test_from_line_round_trips() {
        let l = line();
        match Curve2D::from(l) {
            Curve2D::Line(inner) => assert_eq!(inner, l),
            _ => panic!("expected Curve2D::Line"),
        }
    }

    #[test]
    fn test_from_circle_round_trips() {
        let c = circle();
        match Curve2D::from(c) {
            Curve2D::Circle(inner) => assert_eq!(inner, c),
            _ => panic!("expected Curve2D::Circle"),
        }
    }

    // ---- bounds/period table, per type (inherent trait impl + enum dispatch) ----

    #[test]
    fn test_line_bounds_and_period() {
        let l = line();
        assert_eq!(
            ParametricCurve2D::bounds(&l),
            (f64::NEG_INFINITY, f64::INFINITY)
        );
        assert_eq!(ParametricCurve2D::period(&l), None);
        assert!(!ParametricCurve2D::is_periodic(&l));

        let wrapped: Curve2D = l.into();
        assert_eq!(wrapped.bounds(), (f64::NEG_INFINITY, f64::INFINITY));
        assert_eq!(wrapped.period(), None);
        assert!(!wrapped.is_periodic());
    }

    #[test]
    fn test_circle_bounds_and_period() {
        let c = circle();
        assert_eq!(ParametricCurve2D::bounds(&c), (0.0, TAU));
        assert_eq!(ParametricCurve2D::period(&c), Some(TAU));
        assert!(ParametricCurve2D::is_periodic(&c));

        let wrapped: Curve2D = c.into();
        assert_eq!(wrapped.bounds(), (0.0, TAU));
        assert_eq!(wrapped.period(), Some(TAU));
        assert!(wrapped.is_periodic());
    }

    // ---- default eval_points == mapped eval_point ----

    #[test]
    fn test_default_eval_points_matches_mapped_eval_point_for_each_type() {
        let us = [0.0, 0.5, 1.0, -2.0];

        let l = line();
        let expected: Vec<Point2> = us.iter().map(|&u| l.eval_point(u)).collect();
        assert_eq!(ParametricCurve2D::eval_points(&l, &us), expected);

        let c = circle();
        let expected: Vec<Point2> = us.iter().map(|&u| c.eval_point(u)).collect();
        assert_eq!(ParametricCurve2D::eval_points(&c, &us), expected);
    }

    #[test]
    fn test_curve2d_enum_eval_points_matches_mapped_eval_point() {
        let us = [0.0, 0.5, 1.0, -2.0];
        let curves: Vec<Curve2D> = vec![line().into(), circle().into()];
        for curve in &curves {
            let expected: Vec<Point2> = us.iter().map(|&u| curve.eval_point(u)).collect();
            assert_eq!(curve.eval_points(&us), expected);
        }
    }

    #[test]
    fn test_eval_derivative_delegates_to_inherent() {
        let l = line();
        assert_eq!(
            ParametricCurve2D::eval_derivative(&l, 0.5, 1),
            l.eval_derivative(0.5, 1)
        );

        let c = circle();
        assert_eq!(
            ParametricCurve2D::eval_derivative(&c, 0.5, 1),
            c.eval_derivative(0.5, 1)
        );
    }
}
