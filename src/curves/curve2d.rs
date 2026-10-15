//! [`ParametricCurve2D`] trait and the [`Curve2D`] enum adaptor unifying the
//! concrete 2D curve types.

use crate::curves::{Circle2D, Line2D};
use crate::projection::CurveProjection2D;
use crate::{Point2D, Tolerance, Vector2D};
use std::f64::consts::TAU;

/// Common interface for 2D parametric curves. Implement this to add a new
/// curve type that can be used generically (see [`Curve2D`] for a
/// heterogeneous-collection adaptor built on top of it).
///
/// Mirrors [`crate::curves::ParametricCurve3D`] with [`Point2D`]/[`Vector2D`]
/// in place of `Point3D`/`Vector3D`. Concrete curve types (e.g. [`Line2D`],
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
/// use geomcore::{Line2D, Point2D, Vector2D};
///
/// let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
/// let us = [0.0, 1.0, 2.0];
/// let expected: Vec<Point2D> = us.iter().map(|&u| line.eval_point(u)).collect();
/// assert_eq!(ParametricCurve2D::eval_points(&line, &us), expected);
/// ```
///
/// The default [`is_periodic`](ParametricCurve2D::is_periodic) reports
/// whether [`period`](ParametricCurve2D::period) is `Some`:
///
/// ```
/// use geomcore::curves::ParametricCurve2D;
/// use geomcore::{Circle2D, Line2D, Point2D};
///
/// let line = Line2D::new(Point2D::ORIGIN, geomcore::Vector2D::X).unwrap();
/// assert!(!ParametricCurve2D::is_periodic(&line));
///
/// let circle = Circle2D::new(Point2D::ORIGIN, 1.0).unwrap();
/// assert!(ParametricCurve2D::is_periodic(&circle));
/// ```
pub trait ParametricCurve2D {
    /// Evaluates the point on the curve at parameter `u`.
    fn eval_point(&self, u: f64) -> Point2D;

    /// Evaluates the derivative of the given `order` at parameter `u`.
    fn eval_derivative(&self, u: f64, order: u32) -> Vector2D;

    /// Evaluates the point and the first derivative at parameter `u`.
    ///
    /// Default implementation: [`eval_point`](Self::eval_point) and
    /// [`eval_derivative`](Self::eval_derivative) of order 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::ParametricCurve2D;
    /// use geomcore::{Circle2D, Point2D, Vector2D};
    ///
    /// let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
    /// let (point, tangent) = ParametricCurve2D::eval_d1(&circle, 0.0);
    /// assert_eq!(point, Point2D::new(2.0, 0.0));
    /// assert!((tangent - Vector2D::new(0.0, 2.0)).magnitude() < 1e-15);
    /// ```
    fn eval_d1(&self, u: f64) -> (Point2D, Vector2D) {
        (self.eval_point(u), self.eval_derivative(u, 1))
    }

    /// Evaluates the point and the first two derivatives at parameter `u`.
    ///
    /// Default implementation: [`eval_point`](Self::eval_point) and
    /// [`eval_derivative`](Self::eval_derivative) of orders 1 and 2.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::ParametricCurve2D;
    /// use geomcore::{Circle2D, Point2D, Vector2D};
    ///
    /// let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
    /// let (point, _, acceleration) = ParametricCurve2D::eval_d2(&circle, 0.0);
    /// assert_eq!(point, Point2D::new(2.0, 0.0));
    /// assert!((acceleration - Vector2D::new(-2.0, 0.0)).magnitude() < 1e-15);
    /// ```
    fn eval_d2(&self, u: f64) -> (Point2D, Vector2D, Vector2D) {
        (
            self.eval_point(u),
            self.eval_derivative(u, 1),
            self.eval_derivative(u, 2),
        )
    }

    /// Returns the curve's parameter bounds as `(first, last)`. Unbounded
    /// curves use `(f64::NEG_INFINITY, f64::INFINITY)`.
    fn bounds(&self) -> (f64, f64);

    /// Returns the curve's period, or `None` if the curve is not periodic.
    fn period(&self) -> Option<f64>;

    /// Evaluates the points on the curve at each parameter in `us`.
    ///
    /// Default implementation: maps [`eval_point`](Self::eval_point) over
    /// `us`.
    fn eval_points(&self, us: &[f64]) -> Vec<Point2D> {
        us.iter().map(|&u| self.eval_point(u)).collect()
    }

    /// Returns whether the curve is periodic, i.e. whether
    /// [`period`](Self::period) is `Some`.
    fn is_periodic(&self) -> bool {
        self.period().is_some()
    }
}

/// Distance extrema from a point to a 2D curve: what a
/// [`TrimmedCurve2D`](crate::curves::TrimmedCurve2D) needs from its basis
/// to project onto a piece of it. Mirrors
/// [`crate::curves::PointExtrema3D`].
///
/// # Examples
///
/// ```
/// use geomcore::curves::PointExtrema2D;
/// use geomcore::{Circle2D, Point2D, Tolerance};
///
/// let circle = Circle2D::new(Point2D::ORIGIN, 1.0).unwrap();
/// let extrema = circle.extrema(Point2D::new(3.0, 0.0), Tolerance::DEFAULT);
/// assert_eq!(extrema[0].parameter, 0.0);
/// assert_eq!(extrema[0].distance, 2.0);
/// ```
pub trait PointExtrema2D: ParametricCurve2D {
    /// Parameters on the curve's own domain that include every local
    /// minimum of the distance from `point` (other stationary points may
    /// be listed too; a valley of equal minima is represented by one of
    /// its points), ordered by ascending distance, the global closest
    /// point first. Distances within `tol.confusion` snap to `0.0`.
    ///
    /// A query point with a non-finite coordinate has no extrema.
    fn extrema(&self, point: Point2D, tol: Tolerance) -> Vec<CurveProjection2D>;
}

/// A reference to a curve is the curve, so a
/// [`TrimmedCurve2D`](crate::curves::TrimmedCurve2D) can borrow its
/// basis instead of owning it.
impl<C: ParametricCurve2D + ?Sized> ParametricCurve2D for &C {
    fn eval_point(&self, u: f64) -> Point2D {
        (**self).eval_point(u)
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector2D {
        (**self).eval_derivative(u, order)
    }

    fn eval_d1(&self, u: f64) -> (Point2D, Vector2D) {
        (**self).eval_d1(u)
    }

    fn eval_d2(&self, u: f64) -> (Point2D, Vector2D, Vector2D) {
        (**self).eval_d2(u)
    }

    fn bounds(&self) -> (f64, f64) {
        (**self).bounds()
    }

    fn period(&self) -> Option<f64> {
        (**self).period()
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point2D> {
        (**self).eval_points(us)
    }
}

impl<C: PointExtrema2D + ?Sized> PointExtrema2D for &C {
    fn extrema(&self, point: Point2D, tol: Tolerance) -> Vec<CurveProjection2D> {
        (**self).extrema(point, tol)
    }
}

/// The single local minimum of a line, or of a circle (whose distance
/// from a point falls monotonically to it from the farthest point).
fn only_minimum(point: Point2D, project: impl Fn() -> CurveProjection2D) -> Vec<CurveProjection2D> {
    if point.is_finite() {
        vec![project()]
    } else {
        Vec::new()
    }
}

impl PointExtrema2D for Line2D {
    fn extrema(&self, point: Point2D, tol: Tolerance) -> Vec<CurveProjection2D> {
        only_minimum(point, || self.project_point(point, tol))
    }
}

impl PointExtrema2D for Circle2D {
    fn extrema(&self, point: Point2D, tol: Tolerance) -> Vec<CurveProjection2D> {
        only_minimum(point, || self.project_point(point, tol))
    }
}

impl ParametricCurve2D for Line2D {
    fn eval_point(&self, u: f64) -> Point2D {
        Line2D::eval_point(self, u)
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector2D {
        Line2D::eval_derivative(self, u, order)
    }

    fn bounds(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn period(&self) -> Option<f64> {
        None
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point2D> {
        Line2D::eval_points(self, us)
    }
}

impl ParametricCurve2D for Circle2D {
    fn eval_point(&self, u: f64) -> Point2D {
        Circle2D::eval_point(self, u)
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector2D {
        Circle2D::eval_derivative(self, u, order)
    }

    fn bounds(&self) -> (f64, f64) {
        (0.0, TAU)
    }

    fn period(&self) -> Option<f64> {
        Some(TAU)
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point2D> {
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
/// use geomcore::{Circle2D, Line2D, Point2D, Vector2D};
///
/// let curves: Vec<Curve2D> = vec![
///     Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap().into(),
///     Circle2D::new(Point2D::ORIGIN, 1.0).unwrap().into(),
/// ];
/// let points: Vec<Point2D> = curves.iter().map(|c| c.eval_point(0.0)).collect();
/// assert_eq!(points[0], Point2D::ORIGIN);
/// assert_eq!(points[1], Point2D::new(1.0, 0.0));
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
    fn eval_point(&self, u: f64) -> Point2D {
        match self {
            Curve2D::Line(c) => c.eval_point(u),
            Curve2D::Circle(c) => c.eval_point(u),
        }
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector2D {
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

    fn eval_points(&self, us: &[f64]) -> Vec<Point2D> {
        match self {
            Curve2D::Line(c) => c.eval_points(us),
            Curve2D::Circle(c) => c.eval_points(us),
        }
    }
}

impl PointExtrema2D for Curve2D {
    fn extrema(&self, point: Point2D, tol: Tolerance) -> Vec<CurveProjection2D> {
        match self {
            Curve2D::Line(c) => PointExtrema2D::extrema(c, point, tol),
            Curve2D::Circle(c) => PointExtrema2D::extrema(c, point, tol),
        }
    }
}

impl Curve2D {
    /// Projects `point` onto the curve: the closest parameter, the closest
    /// point and the distance. Distances within `tol.confusion` snap to
    /// `0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::Curve2D;
    /// use geomcore::{Line2D, Point2D, Tolerance, Vector2D};
    ///
    /// let curve = Curve2D::from(Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap());
    /// let projection = curve.project_point(Point2D::new(2.0, 1.0), Tolerance::DEFAULT);
    /// assert_eq!(projection.point, Point2D::new(2.0, 0.0));
    /// assert_eq!(projection.distance, 1.0);
    /// ```
    pub fn project_point(&self, point: Point2D, tol: Tolerance) -> CurveProjection2D {
        match self {
            Curve2D::Line(c) => c.project_point(point, tol),
            Curve2D::Circle(c) => c.project_point(point, tol),
        }
    }

    /// Projects each point in `points` onto the curve.
    pub fn project_points(&self, points: &[Point2D], tol: Tolerance) -> Vec<CurveProjection2D> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }

    /// The parameter of the point of the curve closest to `point`, by the
    /// curve's closed form.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::Curve2D;
    /// use geomcore::{Circle2D, Point2D};
    ///
    /// let curve = Curve2D::from(Circle2D::new(Point2D::ORIGIN, 2.0).unwrap());
    /// let parameter = curve.parameter_of(Point2D::new(0.0, 2.0));
    /// assert!((parameter - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    /// ```
    pub fn parameter_of(&self, point: Point2D) -> f64 {
        match self {
            Curve2D::Line(c) => c.parameter_of(point),
            Curve2D::Circle(c) => c.parameter_of(point),
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

    fn sample<C: ParametricCurve2D>(c: &C) -> Point2D {
        c.eval_point(0.5)
    }

    fn line() -> Line2D {
        Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap()
    }

    fn circle() -> Circle2D {
        Circle2D::new(Point2D::ORIGIN, 2.0).unwrap()
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
        let points: Vec<Point2D> = curves.iter().map(|curve| curve.eval_point(0.25)).collect();
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
        let expected: Vec<Point2D> = us.iter().map(|&u| l.eval_point(u)).collect();
        assert_eq!(ParametricCurve2D::eval_points(&l, &us), expected);

        let c = circle();
        let expected: Vec<Point2D> = us.iter().map(|&u| c.eval_point(u)).collect();
        assert_eq!(ParametricCurve2D::eval_points(&c, &us), expected);
    }

    #[test]
    fn test_curve2d_enum_eval_points_matches_mapped_eval_point() {
        let us = [0.0, 0.5, 1.0, -2.0];
        let curves: Vec<Curve2D> = vec![line().into(), circle().into()];
        for curve in &curves {
            let expected: Vec<Point2D> = us.iter().map(|&u| curve.eval_point(u)).collect();
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
