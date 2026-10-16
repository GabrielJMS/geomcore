//! [`ParametricCurve3D`] trait and the [`Curve3D`] enum adaptor unifying the
//! concrete 3D curve types.

use crate::curves::{BSplineCurve3D, Circle3D, Ellipse3D, Hyperbola3D, Line3D, Parabola3D};
use crate::projection::CurveProjection;
use crate::{Point3D, Tolerance, Vector3D};
use std::f64::consts::TAU;

/// Which side of a parameter to evaluate from, for one-sided limits at
/// knots and seams.
///
/// `Below` approaches from the span ending at the parameter, `Above`
/// from the span starting there (the side plain evaluation takes,
/// except at the upper bound). Away from knots and seams both sides
/// agree. Shared by curve and surface sided-evaluation queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Limit from parameters below (the span ending here).
    Below,
    /// Limit from parameters above (the span starting here).
    Above,
}

/// Common interface for 3D parametric curves. Implement this to add a new
/// curve type that can be used generically (see [`Curve3D`] for a
/// heterogeneous-collection adaptor built on top of it).
///
/// Concrete curve types (e.g. [`Line3D`], [`Circle3D`]) already provide
/// inherent methods with the same names; their trait impls simply delegate
/// to those inherent methods, so calling code can use either form.
///
/// # Examples
///
/// The default [`eval_points`](ParametricCurve3D::eval_points) maps
/// [`eval_point`](ParametricCurve3D::eval_point) over each parameter:
///
/// ```
/// use geomcore::curves::ParametricCurve3D;
/// use geomcore::{Line3D, Point3D, Vector3D};
///
/// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
/// let us = [0.0, 1.0, 2.0];
/// let expected: Vec<Point3D> = us.iter().map(|&u| line.eval_point(u)).collect();
/// assert_eq!(ParametricCurve3D::eval_points(&line, &us), expected);
/// ```
///
/// The default [`is_periodic`](ParametricCurve3D::is_periodic) reports
/// whether [`period`](ParametricCurve3D::period) is `Some`:
///
/// ```
/// use geomcore::curves::ParametricCurve3D;
/// use geomcore::{Circle3D, Line3D, Point3D, Vector3D};
///
/// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
/// assert!(!ParametricCurve3D::is_periodic(&line));
///
/// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
/// assert!(ParametricCurve3D::is_periodic(&circle));
/// ```
pub trait ParametricCurve3D {
    /// Evaluates the point on the curve at parameter `u`.
    fn eval_point(&self, u: f64) -> Point3D;

    /// Evaluates the derivative of the given `order` at parameter `u`.
    fn eval_derivative(&self, u: f64, order: u32) -> Vector3D;

    /// Evaluates the point and the first derivative at parameter `u`.
    ///
    /// Default implementation: [`eval_point`](Self::eval_point) and
    /// [`eval_derivative`](Self::eval_derivative). B-spline curves override
    /// it to evaluate their basis functions once.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::ParametricCurve3D;
    /// use geomcore::{Circle3D, Point3D, Vector3D};
    ///
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let (point, tangent) = circle.eval_d1(0.0);
    /// assert_eq!(point, Point3D::new(2.0, 0.0, 0.0));
    /// assert!((tangent - Vector3D::new(0.0, 2.0, 0.0)).magnitude() < 1e-15);
    /// ```
    fn eval_d1(&self, u: f64) -> (Point3D, Vector3D) {
        (self.eval_point(u), self.eval_derivative(u, 1))
    }

    /// Evaluates the point and the first and second derivatives at
    /// parameter `u`.
    ///
    /// Default implementation: [`eval_point`](Self::eval_point) and
    /// [`eval_derivative`](Self::eval_derivative). B-spline curves override
    /// it to evaluate their basis functions once.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::ParametricCurve3D;
    /// use geomcore::{BSplineCurve3D, Point3D, Vector3D};
    ///
    /// let poles = vec![
    ///     Point3D::new(0.0, 0.0, 0.0),
    ///     Point3D::new(1.0, 1.0, 0.0),
    ///     Point3D::new(2.0, 0.0, 0.0),
    /// ];
    /// let curve = BSplineCurve3D::new(2, poles, vec![0.0, 1.0], vec![3, 3], false).unwrap();
    /// let (point, d1, d2) = curve.eval_d2(0.5);
    /// assert_eq!(point, Point3D::new(1.0, 0.5, 0.0));
    /// assert_eq!(d1, Vector3D::new(2.0, 0.0, 0.0));
    /// assert_eq!(d2, Vector3D::new(0.0, -4.0, 0.0));
    /// ```
    fn eval_d2(&self, u: f64) -> (Point3D, Vector3D, Vector3D) {
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
    fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        us.iter().map(|&u| self.eval_point(u)).collect()
    }

    /// Returns whether the curve is periodic, i.e. whether
    /// [`period`](Self::period) is `Some`.
    fn is_periodic(&self) -> bool {
        self.period().is_some()
    }

    /// The B-spline this curve evaluates: the curve itself, or the basis
    /// of a trimmed B-spline, of which the curve is the piece over
    /// [`bounds`](Self::bounds). `None` for any other curve.
    ///
    /// Numeric algorithms such as
    /// [`intersect_curve_surface`](crate::intersect_curve_surface) use it
    /// to work on the B-spline's polynomial pieces. Default
    /// implementation: `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::{ParametricCurve3D, TrimmedCurve3D};
    /// use geomcore::{BSplineCurve3D, Line3D, Point3D, Vector3D};
    ///
    /// let poles = vec![Point3D::ORIGIN, Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// let piece = TrimmedCurve3D::new(&curve, 0.25, 0.5).unwrap();
    /// assert_eq!(piece.as_bspline(), Some(&curve));
    ///
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// assert_eq!(ParametricCurve3D::as_bspline(&line), None);
    /// ```
    fn as_bspline(&self) -> Option<&BSplineCurve3D> {
        None
    }

    /// The elementary curve this curve evaluates: the curve itself, or the
    /// basis of a trimmed elementary curve, of which the curve is the
    /// piece over [`bounds`](Self::bounds). `None` for any other curve.
    ///
    /// Numeric algorithms such as
    /// [`intersect_curve_surface`](crate::intersect_curve_surface) use it
    /// to convert the curve to an exact B-spline when the other shape is
    /// one. Default implementation: `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::{ElementaryCurve, ParametricCurve3D, TrimmedCurve3D};
    /// use geomcore::{BSplineCurve3D, Circle3D, Point3D, Vector3D};
    ///
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    /// let arc = TrimmedCurve3D::new(&circle, 0.0, 1.0).unwrap();
    /// assert_eq!(arc.as_elementary(), Some(ElementaryCurve::Circle(circle)));
    ///
    /// let poles = vec![Point3D::ORIGIN, Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert_eq!(ParametricCurve3D::as_elementary(&curve), None);
    /// ```
    fn as_elementary(&self) -> Option<ElementaryCurve> {
        None
    }
}

/// Distance extrema from a point to a 3D curve: what a
/// [`TrimmedCurve3D`](crate::curves::TrimmedCurve3D) needs from its basis
/// to project onto a piece of it.
///
/// # Examples
///
/// ```
/// use geomcore::curves::PointExtrema3D;
/// use geomcore::{Line3D, Point3D, Tolerance, Vector3D};
///
/// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
/// let extrema = line.extrema(Point3D::new(2.0, 1.0, 0.0), Tolerance::DEFAULT);
/// assert_eq!(extrema[0].parameter, 2.0);
/// assert_eq!(extrema[0].distance, 1.0);
/// ```
pub trait PointExtrema3D: ParametricCurve3D {
    /// Parameters on the curve's own domain that include every local
    /// minimum of the distance from `point` (other stationary points may
    /// be listed too; a valley of equal minima is represented by one of
    /// its points), ordered by ascending distance, the global closest
    /// point first. Distances within `tol.confusion` snap to `0.0`.
    ///
    /// A query point with a non-finite coordinate has no extrema.
    fn extrema(&self, point: Point3D, tol: Tolerance) -> Vec<CurveProjection>;
}

/// A reference to a curve is the curve, so a
/// [`TrimmedCurve3D`](crate::curves::TrimmedCurve3D) can borrow its
/// basis instead of owning it.
impl<C: ParametricCurve3D + ?Sized> ParametricCurve3D for &C {
    fn eval_point(&self, u: f64) -> Point3D {
        (**self).eval_point(u)
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        (**self).eval_derivative(u, order)
    }

    fn eval_d1(&self, u: f64) -> (Point3D, Vector3D) {
        (**self).eval_d1(u)
    }

    fn eval_d2(&self, u: f64) -> (Point3D, Vector3D, Vector3D) {
        (**self).eval_d2(u)
    }

    fn bounds(&self) -> (f64, f64) {
        (**self).bounds()
    }

    fn period(&self) -> Option<f64> {
        (**self).period()
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        (**self).eval_points(us)
    }

    fn as_bspline(&self) -> Option<&BSplineCurve3D> {
        (**self).as_bspline()
    }

    fn as_elementary(&self) -> Option<ElementaryCurve> {
        (**self).as_elementary()
    }
}

impl<C: PointExtrema3D + ?Sized> PointExtrema3D for &C {
    fn extrema(&self, point: Point3D, tol: Tolerance) -> Vec<CurveProjection> {
        (**self).extrema(point, tol)
    }
}

/// The single local minimum of a line, or of a circle (whose distance
/// from a point falls monotonically to it from the farthest point).
fn only_minimum(point: Point3D, project: impl Fn() -> CurveProjection) -> Vec<CurveProjection> {
    if point.is_finite() {
        vec![project()]
    } else {
        Vec::new()
    }
}

impl PointExtrema3D for Line3D {
    fn extrema(&self, point: Point3D, tol: Tolerance) -> Vec<CurveProjection> {
        only_minimum(point, || self.project_point(point, tol))
    }
}

impl PointExtrema3D for Circle3D {
    fn extrema(&self, point: Point3D, tol: Tolerance) -> Vec<CurveProjection> {
        only_minimum(point, || self.project_point(point, tol))
    }
}

impl PointExtrema3D for Ellipse3D {
    fn extrema(&self, point: Point3D, tol: Tolerance) -> Vec<CurveProjection> {
        Ellipse3D::extrema(self, point, tol)
    }
}

impl PointExtrema3D for Parabola3D {
    fn extrema(&self, point: Point3D, tol: Tolerance) -> Vec<CurveProjection> {
        Parabola3D::extrema(self, point, tol)
    }
}

impl PointExtrema3D for Hyperbola3D {
    fn extrema(&self, point: Point3D, tol: Tolerance) -> Vec<CurveProjection> {
        Hyperbola3D::extrema(self, point, tol)
    }
}

impl PointExtrema3D for BSplineCurve3D {
    fn extrema(&self, point: Point3D, tol: Tolerance) -> Vec<CurveProjection> {
        if point.is_finite() {
            BSplineCurve3D::extrema(self, point, tol)
        } else {
            Vec::new()
        }
    }
}

impl ParametricCurve3D for Line3D {
    fn eval_point(&self, u: f64) -> Point3D {
        Line3D::eval_point(self, u)
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        Line3D::eval_derivative(self, u, order)
    }

    fn bounds(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn period(&self) -> Option<f64> {
        None
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        Line3D::eval_points(self, us)
    }

    fn as_elementary(&self) -> Option<ElementaryCurve> {
        Some(ElementaryCurve::Line(*self))
    }
}

impl ParametricCurve3D for Circle3D {
    fn eval_point(&self, u: f64) -> Point3D {
        Circle3D::eval_point(self, u)
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        Circle3D::eval_derivative(self, u, order)
    }

    fn bounds(&self) -> (f64, f64) {
        (0.0, TAU)
    }

    fn period(&self) -> Option<f64> {
        Some(TAU)
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        Circle3D::eval_points(self, us)
    }

    fn as_elementary(&self) -> Option<ElementaryCurve> {
        Some(ElementaryCurve::Circle(*self))
    }
}

impl ParametricCurve3D for Ellipse3D {
    fn eval_point(&self, u: f64) -> Point3D {
        Ellipse3D::eval_point(self, u)
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        Ellipse3D::eval_derivative(self, u, order)
    }

    fn bounds(&self) -> (f64, f64) {
        (0.0, TAU)
    }

    fn period(&self) -> Option<f64> {
        Some(TAU)
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        Ellipse3D::eval_points(self, us)
    }

    fn as_elementary(&self) -> Option<ElementaryCurve> {
        Some(ElementaryCurve::Ellipse(*self))
    }
}

impl ParametricCurve3D for Parabola3D {
    fn eval_point(&self, u: f64) -> Point3D {
        Parabola3D::eval_point(self, u)
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        Parabola3D::eval_derivative(self, u, order)
    }

    fn bounds(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn period(&self) -> Option<f64> {
        None
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        Parabola3D::eval_points(self, us)
    }

    fn as_elementary(&self) -> Option<ElementaryCurve> {
        Some(ElementaryCurve::Parabola(*self))
    }
}

impl ParametricCurve3D for Hyperbola3D {
    fn eval_point(&self, u: f64) -> Point3D {
        Hyperbola3D::eval_point(self, u)
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        Hyperbola3D::eval_derivative(self, u, order)
    }

    fn bounds(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn period(&self) -> Option<f64> {
        None
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        Hyperbola3D::eval_points(self, us)
    }

    fn as_elementary(&self) -> Option<ElementaryCurve> {
        Some(ElementaryCurve::Hyperbola(*self))
    }
}

impl ParametricCurve3D for BSplineCurve3D {
    fn eval_point(&self, u: f64) -> Point3D {
        BSplineCurve3D::eval_point(self, u)
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        BSplineCurve3D::eval_derivative(self, u, order)
    }

    fn eval_d1(&self, u: f64) -> (Point3D, Vector3D) {
        BSplineCurve3D::eval_d1(self, u)
    }

    fn eval_d2(&self, u: f64) -> (Point3D, Vector3D, Vector3D) {
        BSplineCurve3D::eval_d2(self, u)
    }

    fn bounds(&self) -> (f64, f64) {
        BSplineCurve3D::bounds(self)
    }

    fn period(&self) -> Option<f64> {
        if self.is_periodic() {
            let (first, last) = BSplineCurve3D::bounds(self);
            Some(last - first)
        } else {
            None
        }
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        BSplineCurve3D::eval_points(self, us)
    }

    fn as_bspline(&self) -> Option<&BSplineCurve3D> {
        Some(self)
    }
}

/// Enum adaptor unifying the concrete 3D curve types behind a single value
/// type, so callers can build a heterogeneous collection of curves and
/// evaluate them uniformly through [`ParametricCurve3D`].
///
/// Marked `#[non_exhaustive]`: future tasks add further variants (e.g. a
/// B-spline curve) without that being a breaking change.
///
/// # Examples
///
/// ```
/// use geomcore::curves::{Curve3D, ParametricCurve3D};
/// use geomcore::{Circle3D, Line3D, Point3D, Vector3D};
///
/// let curves: Vec<Curve3D> = vec![
///     Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap().into(),
///     Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap().into(),
/// ];
/// let points: Vec<Point3D> = curves.iter().map(|c| c.eval_point(0.0)).collect();
/// assert_eq!(points[0], Point3D::ORIGIN);
/// assert_eq!(points[1], Point3D::new(1.0, 0.0, 0.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Curve3D {
    /// A [`Line3D`] curve.
    Line(Line3D),
    /// A [`Circle3D`] curve.
    Circle(Circle3D),
    /// An [`Ellipse3D`] curve.
    Ellipse(Ellipse3D),
    /// A [`Parabola3D`] curve.
    Parabola(Parabola3D),
    /// A [`Hyperbola3D`] curve.
    Hyperbola(Hyperbola3D),
    /// A [`BSplineCurve3D`] curve.
    BSpline(BSplineCurve3D),
}

impl ParametricCurve3D for Curve3D {
    fn eval_point(&self, u: f64) -> Point3D {
        match self {
            Curve3D::Line(c) => c.eval_point(u),
            Curve3D::Circle(c) => c.eval_point(u),
            Curve3D::Ellipse(c) => c.eval_point(u),
            Curve3D::Parabola(c) => c.eval_point(u),
            Curve3D::Hyperbola(c) => c.eval_point(u),
            Curve3D::BSpline(c) => c.eval_point(u),
        }
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        match self {
            Curve3D::Line(c) => c.eval_derivative(u, order),
            Curve3D::Circle(c) => c.eval_derivative(u, order),
            Curve3D::Ellipse(c) => c.eval_derivative(u, order),
            Curve3D::Parabola(c) => c.eval_derivative(u, order),
            Curve3D::Hyperbola(c) => c.eval_derivative(u, order),
            Curve3D::BSpline(c) => c.eval_derivative(u, order),
        }
    }

    fn eval_d1(&self, u: f64) -> (Point3D, Vector3D) {
        match self {
            Curve3D::Line(c) => ParametricCurve3D::eval_d1(c, u),
            Curve3D::Circle(c) => ParametricCurve3D::eval_d1(c, u),
            Curve3D::Ellipse(c) => ParametricCurve3D::eval_d1(c, u),
            Curve3D::Parabola(c) => ParametricCurve3D::eval_d1(c, u),
            Curve3D::Hyperbola(c) => ParametricCurve3D::eval_d1(c, u),
            Curve3D::BSpline(c) => c.eval_d1(u),
        }
    }

    fn eval_d2(&self, u: f64) -> (Point3D, Vector3D, Vector3D) {
        match self {
            Curve3D::Line(c) => ParametricCurve3D::eval_d2(c, u),
            Curve3D::Circle(c) => ParametricCurve3D::eval_d2(c, u),
            Curve3D::Ellipse(c) => ParametricCurve3D::eval_d2(c, u),
            Curve3D::Parabola(c) => ParametricCurve3D::eval_d2(c, u),
            Curve3D::Hyperbola(c) => ParametricCurve3D::eval_d2(c, u),
            Curve3D::BSpline(c) => c.eval_d2(u),
        }
    }

    fn bounds(&self) -> (f64, f64) {
        match self {
            Curve3D::Line(c) => c.bounds(),
            Curve3D::Circle(c) => c.bounds(),
            Curve3D::Ellipse(c) => c.bounds(),
            Curve3D::Parabola(c) => c.bounds(),
            Curve3D::Hyperbola(c) => c.bounds(),
            Curve3D::BSpline(c) => ParametricCurve3D::bounds(c),
        }
    }

    fn period(&self) -> Option<f64> {
        match self {
            Curve3D::Line(c) => c.period(),
            Curve3D::Circle(c) => c.period(),
            Curve3D::Ellipse(c) => c.period(),
            Curve3D::Parabola(c) => c.period(),
            Curve3D::Hyperbola(c) => c.period(),
            Curve3D::BSpline(c) => ParametricCurve3D::period(c),
        }
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        match self {
            Curve3D::Line(c) => c.eval_points(us),
            Curve3D::Circle(c) => c.eval_points(us),
            Curve3D::Ellipse(c) => c.eval_points(us),
            Curve3D::Parabola(c) => c.eval_points(us),
            Curve3D::Hyperbola(c) => c.eval_points(us),
            Curve3D::BSpline(c) => c.eval_points(us),
        }
    }

    fn as_bspline(&self) -> Option<&BSplineCurve3D> {
        match self {
            Curve3D::BSpline(c) => Some(c),
            _ => None,
        }
    }

    fn as_elementary(&self) -> Option<ElementaryCurve> {
        Curve3D::as_elementary(self)
    }
}

impl PointExtrema3D for Curve3D {
    fn extrema(&self, point: Point3D, tol: Tolerance) -> Vec<CurveProjection> {
        match self {
            Curve3D::Line(c) => PointExtrema3D::extrema(c, point, tol),
            Curve3D::Circle(c) => PointExtrema3D::extrema(c, point, tol),
            Curve3D::Ellipse(c) => PointExtrema3D::extrema(c, point, tol),
            Curve3D::Parabola(c) => PointExtrema3D::extrema(c, point, tol),
            Curve3D::Hyperbola(c) => PointExtrema3D::extrema(c, point, tol),
            Curve3D::BSpline(c) => PointExtrema3D::extrema(c, point, tol),
        }
    }
}

impl From<Line3D> for Curve3D {
    /// Wraps a [`Line3D`] as a [`Curve3D::Line`].
    fn from(curve: Line3D) -> Curve3D {
        Curve3D::Line(curve)
    }
}

impl From<Circle3D> for Curve3D {
    /// Wraps a [`Circle3D`] as a [`Curve3D::Circle`].
    fn from(curve: Circle3D) -> Curve3D {
        Curve3D::Circle(curve)
    }
}

impl From<Ellipse3D> for Curve3D {
    /// Wraps an [`Ellipse3D`] as a [`Curve3D::Ellipse`].
    fn from(curve: Ellipse3D) -> Curve3D {
        Curve3D::Ellipse(curve)
    }
}

impl From<Parabola3D> for Curve3D {
    /// Wraps a [`Parabola3D`] as a [`Curve3D::Parabola`].
    fn from(curve: Parabola3D) -> Curve3D {
        Curve3D::Parabola(curve)
    }
}

impl From<Hyperbola3D> for Curve3D {
    /// Wraps a [`Hyperbola3D`] as a [`Curve3D::Hyperbola`].
    fn from(curve: Hyperbola3D) -> Curve3D {
        Curve3D::Hyperbola(curve)
    }
}

impl From<BSplineCurve3D> for Curve3D {
    /// Wraps a [`BSplineCurve3D`] as a [`Curve3D::BSpline`].
    fn from(curve: BSplineCurve3D) -> Curve3D {
        Curve3D::BSpline(curve)
    }
}

/// An elementary curve: a line, circle, ellipse, parabola or hyperbola,
/// the curves with a closed-form definition.
///
/// Unlike [`Curve3D`] it is `Copy` and owns no heap memory, so it can be
/// stored inline in intersection results.
///
/// # Examples
///
/// ```
/// use geomcore::curves::{Curve3D, ElementaryCurve};
/// use geomcore::{Line3D, Point3D, Vector3D};
///
/// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
/// let curve = Curve3D::from(line);
/// assert_eq!(curve.as_elementary(), Some(ElementaryCurve::Line(line)));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElementaryCurve {
    /// A [`Line3D`] curve.
    Line(Line3D),
    /// A [`Circle3D`] curve.
    Circle(Circle3D),
    /// An [`Ellipse3D`] curve.
    Ellipse(Ellipse3D),
    /// A [`Parabola3D`] curve.
    Parabola(Parabola3D),
    /// A [`Hyperbola3D`] curve.
    Hyperbola(Hyperbola3D),
}

impl Curve3D {
    /// The curve as an [`ElementaryCurve`], or `None` for a B-spline.
    pub fn as_elementary(&self) -> Option<ElementaryCurve> {
        match self {
            Curve3D::Line(c) => Some(ElementaryCurve::Line(*c)),
            Curve3D::Circle(c) => Some(ElementaryCurve::Circle(*c)),
            Curve3D::Ellipse(c) => Some(ElementaryCurve::Ellipse(*c)),
            Curve3D::Parabola(c) => Some(ElementaryCurve::Parabola(*c)),
            Curve3D::Hyperbola(c) => Some(ElementaryCurve::Hyperbola(*c)),
            Curve3D::BSpline(_) => None,
        }
    }
}

impl ParametricCurve3D for ElementaryCurve {
    fn eval_point(&self, u: f64) -> Point3D {
        match self {
            ElementaryCurve::Line(c) => c.eval_point(u),
            ElementaryCurve::Circle(c) => c.eval_point(u),
            ElementaryCurve::Ellipse(c) => c.eval_point(u),
            ElementaryCurve::Parabola(c) => c.eval_point(u),
            ElementaryCurve::Hyperbola(c) => c.eval_point(u),
        }
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        match self {
            ElementaryCurve::Line(c) => c.eval_derivative(u, order),
            ElementaryCurve::Circle(c) => c.eval_derivative(u, order),
            ElementaryCurve::Ellipse(c) => c.eval_derivative(u, order),
            ElementaryCurve::Parabola(c) => c.eval_derivative(u, order),
            ElementaryCurve::Hyperbola(c) => c.eval_derivative(u, order),
        }
    }

    fn bounds(&self) -> (f64, f64) {
        match self {
            ElementaryCurve::Line(c) => c.bounds(),
            ElementaryCurve::Circle(c) => c.bounds(),
            ElementaryCurve::Ellipse(c) => c.bounds(),
            ElementaryCurve::Parabola(c) => c.bounds(),
            ElementaryCurve::Hyperbola(c) => c.bounds(),
        }
    }

    fn period(&self) -> Option<f64> {
        match self {
            ElementaryCurve::Line(c) => c.period(),
            ElementaryCurve::Circle(c) => c.period(),
            ElementaryCurve::Ellipse(c) => c.period(),
            ElementaryCurve::Parabola(c) => c.period(),
            ElementaryCurve::Hyperbola(c) => c.period(),
        }
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        match self {
            ElementaryCurve::Line(c) => c.eval_points(us),
            ElementaryCurve::Circle(c) => c.eval_points(us),
            ElementaryCurve::Ellipse(c) => c.eval_points(us),
            ElementaryCurve::Parabola(c) => c.eval_points(us),
            ElementaryCurve::Hyperbola(c) => c.eval_points(us),
        }
    }

    fn as_elementary(&self) -> Option<ElementaryCurve> {
        Some(*self)
    }
}

impl From<ElementaryCurve> for Curve3D {
    /// Wraps an [`ElementaryCurve`] as the matching [`Curve3D`] variant.
    fn from(curve: ElementaryCurve) -> Curve3D {
        match curve {
            ElementaryCurve::Line(c) => Curve3D::Line(c),
            ElementaryCurve::Circle(c) => Curve3D::Circle(c),
            ElementaryCurve::Ellipse(c) => Curve3D::Ellipse(c),
            ElementaryCurve::Parabola(c) => Curve3D::Parabola(c),
            ElementaryCurve::Hyperbola(c) => Curve3D::Hyperbola(c),
        }
    }
}

impl From<Line3D> for ElementaryCurve {
    /// Wraps a [`Line3D`] as an [`ElementaryCurve::Line`].
    fn from(curve: Line3D) -> ElementaryCurve {
        ElementaryCurve::Line(curve)
    }
}

impl From<Circle3D> for ElementaryCurve {
    /// Wraps a [`Circle3D`] as an [`ElementaryCurve::Circle`].
    fn from(curve: Circle3D) -> ElementaryCurve {
        ElementaryCurve::Circle(curve)
    }
}

impl From<Ellipse3D> for ElementaryCurve {
    /// Wraps an [`Ellipse3D`] as an [`ElementaryCurve::Ellipse`].
    fn from(curve: Ellipse3D) -> ElementaryCurve {
        ElementaryCurve::Ellipse(curve)
    }
}

impl From<Parabola3D> for ElementaryCurve {
    /// Wraps a [`Parabola3D`] as an [`ElementaryCurve::Parabola`].
    fn from(curve: Parabola3D) -> ElementaryCurve {
        ElementaryCurve::Parabola(curve)
    }
}

impl From<Hyperbola3D> for ElementaryCurve {
    /// Wraps a [`Hyperbola3D`] as an [`ElementaryCurve::Hyperbola`].
    fn from(curve: Hyperbola3D) -> ElementaryCurve {
        ElementaryCurve::Hyperbola(curve)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample<C: ParametricCurve3D>(c: &C) -> Point3D {
        c.eval_point(0.5)
    }

    fn line() -> Line3D {
        Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap()
    }

    fn circle() -> Circle3D {
        Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap()
    }

    fn ellipse() -> Ellipse3D {
        Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap()
    }

    fn parabola() -> Parabola3D {
        Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap()
    }

    fn hyperbola() -> Hyperbola3D {
        Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap()
    }

    fn bspline() -> BSplineCurve3D {
        // Clamped degree-1 curve: straight segment (0,0,0) -> (2,0,0).
        let poles = vec![Point3D::ORIGIN, Point3D::new(2.0, 0.0, 0.0)];
        BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap()
    }

    fn bspline_periodic() -> BSplineCurve3D {
        // Periodic degree-1 triangle, knots [0,1,2,3] period 3.
        let poles = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
        ];
        BSplineCurve3D::new(1, poles, vec![0.0, 1.0, 2.0, 3.0], vec![1, 1, 1, 1], true).unwrap()
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
    fn test_sample_ellipse_agrees_with_inherent() {
        let e = ellipse();
        assert_eq!(sample(&e), e.eval_point(0.5));
    }

    #[test]
    fn test_sample_parabola_agrees_with_inherent() {
        let p = parabola();
        assert_eq!(sample(&p), p.eval_point(0.5));
    }

    #[test]
    fn test_sample_hyperbola_agrees_with_inherent() {
        let h = hyperbola();
        assert_eq!(sample(&h), h.eval_point(0.5));
    }

    #[test]
    fn test_sample_bspline_agrees_with_inherent() {
        let b = bspline();
        assert_eq!(sample(&b), b.eval_point(0.5));
    }

    #[test]
    fn test_sample_curve3d_enum_agrees_with_inherent() {
        let l = line();
        let wrapped: Curve3D = l.into();
        assert_eq!(sample(&wrapped), l.eval_point(0.5));
    }

    // ---- heterogeneous Vec<Curve3D> eval ----

    #[test]
    fn test_heterogeneous_vec_eval() {
        let l = line();
        let c = circle();
        let e = ellipse();
        let p = parabola();
        let h = hyperbola();
        let b = bspline();
        let curves: Vec<Curve3D> = vec![
            l.into(),
            c.into(),
            e.into(),
            p.into(),
            h.into(),
            b.clone().into(),
        ];
        let points: Vec<Point3D> = curves.iter().map(|curve| curve.eval_point(0.25)).collect();
        assert_eq!(points[0], l.eval_point(0.25));
        assert_eq!(points[1], c.eval_point(0.25));
        assert_eq!(points[2], e.eval_point(0.25));
        assert_eq!(points[3], p.eval_point(0.25));
        assert_eq!(points[4], h.eval_point(0.25));
        assert_eq!(points[5], b.eval_point(0.25));
    }

    // ---- From round-trips ----

    #[test]
    fn test_from_line_round_trips() {
        let l = line();
        match Curve3D::from(l) {
            Curve3D::Line(inner) => assert_eq!(inner, l),
            _ => panic!("expected Curve3D::Line"),
        }
    }

    #[test]
    fn test_from_circle_round_trips() {
        let c = circle();
        match Curve3D::from(c) {
            Curve3D::Circle(inner) => assert_eq!(inner, c),
            _ => panic!("expected Curve3D::Circle"),
        }
    }

    #[test]
    fn test_from_ellipse_round_trips() {
        let e = ellipse();
        match Curve3D::from(e) {
            Curve3D::Ellipse(inner) => assert_eq!(inner, e),
            _ => panic!("expected Curve3D::Ellipse"),
        }
    }

    #[test]
    fn test_from_parabola_round_trips() {
        let p = parabola();
        match Curve3D::from(p) {
            Curve3D::Parabola(inner) => assert_eq!(inner, p),
            _ => panic!("expected Curve3D::Parabola"),
        }
    }

    #[test]
    fn test_from_hyperbola_round_trips() {
        let h = hyperbola();
        match Curve3D::from(h) {
            Curve3D::Hyperbola(inner) => assert_eq!(inner, h),
            _ => panic!("expected Curve3D::Hyperbola"),
        }
    }

    #[test]
    fn test_from_bspline_round_trips() {
        let b = bspline();
        match Curve3D::from(b.clone()) {
            Curve3D::BSpline(inner) => assert_eq!(inner, b),
            _ => panic!("expected Curve3D::BSpline"),
        }
    }

    // ---- bounds/period table, per type (inherent trait impl + enum dispatch) ----

    #[test]
    fn test_line_bounds_and_period() {
        let l = line();
        assert_eq!(
            ParametricCurve3D::bounds(&l),
            (f64::NEG_INFINITY, f64::INFINITY)
        );
        assert_eq!(ParametricCurve3D::period(&l), None);
        assert!(!ParametricCurve3D::is_periodic(&l));

        let wrapped: Curve3D = l.into();
        assert_eq!(wrapped.bounds(), (f64::NEG_INFINITY, f64::INFINITY));
        assert_eq!(wrapped.period(), None);
        assert!(!wrapped.is_periodic());
    }

    #[test]
    fn test_circle_bounds_and_period() {
        let c = circle();
        assert_eq!(ParametricCurve3D::bounds(&c), (0.0, TAU));
        assert_eq!(ParametricCurve3D::period(&c), Some(TAU));
        assert!(ParametricCurve3D::is_periodic(&c));

        let wrapped: Curve3D = c.into();
        assert_eq!(wrapped.bounds(), (0.0, TAU));
        assert_eq!(wrapped.period(), Some(TAU));
        assert!(wrapped.is_periodic());
    }

    #[test]
    fn test_ellipse_bounds_and_period() {
        let e = ellipse();
        assert_eq!(ParametricCurve3D::bounds(&e), (0.0, TAU));
        assert_eq!(ParametricCurve3D::period(&e), Some(TAU));
        assert!(ParametricCurve3D::is_periodic(&e));

        let wrapped: Curve3D = e.into();
        assert_eq!(wrapped.bounds(), (0.0, TAU));
        assert_eq!(wrapped.period(), Some(TAU));
        assert!(wrapped.is_periodic());
    }

    #[test]
    fn test_parabola_bounds_and_period() {
        let p = parabola();
        assert_eq!(
            ParametricCurve3D::bounds(&p),
            (f64::NEG_INFINITY, f64::INFINITY)
        );
        assert_eq!(ParametricCurve3D::period(&p), None);
        assert!(!ParametricCurve3D::is_periodic(&p));

        let wrapped: Curve3D = p.into();
        assert_eq!(wrapped.bounds(), (f64::NEG_INFINITY, f64::INFINITY));
        assert_eq!(wrapped.period(), None);
        assert!(!wrapped.is_periodic());
    }

    #[test]
    fn test_hyperbola_bounds_and_period() {
        let h = hyperbola();
        assert_eq!(
            ParametricCurve3D::bounds(&h),
            (f64::NEG_INFINITY, f64::INFINITY)
        );
        assert_eq!(ParametricCurve3D::period(&h), None);
        assert!(!ParametricCurve3D::is_periodic(&h));

        let wrapped: Curve3D = h.into();
        assert_eq!(wrapped.bounds(), (f64::NEG_INFINITY, f64::INFINITY));
        assert_eq!(wrapped.period(), None);
        assert!(!wrapped.is_periodic());
    }

    #[test]
    fn test_bspline_clamped_bounds_and_period() {
        let b = bspline();
        assert_eq!(ParametricCurve3D::bounds(&b), (0.0, 1.0));
        assert_eq!(ParametricCurve3D::period(&b), None);
        assert!(!ParametricCurve3D::is_periodic(&b));

        let wrapped: Curve3D = b.into();
        assert_eq!(wrapped.bounds(), (0.0, 1.0));
        assert_eq!(wrapped.period(), None);
        assert!(!wrapped.is_periodic());
    }

    #[test]
    fn test_bspline_periodic_bounds_and_period() {
        let b = bspline_periodic();
        assert_eq!(ParametricCurve3D::bounds(&b), (0.0, 3.0));
        assert_eq!(ParametricCurve3D::period(&b), Some(3.0));
        assert!(ParametricCurve3D::is_periodic(&b));

        let wrapped: Curve3D = b.into();
        assert_eq!(wrapped.bounds(), (0.0, 3.0));
        assert_eq!(wrapped.period(), Some(3.0));
        assert!(wrapped.is_periodic());
    }

    // ---- default eval_points == mapped eval_point ----

    #[test]
    fn test_default_eval_points_matches_mapped_eval_point_for_each_type() {
        let us = [0.0, 0.5, 1.0, -2.0];

        let l = line();
        let expected: Vec<Point3D> = us.iter().map(|&u| l.eval_point(u)).collect();
        assert_eq!(ParametricCurve3D::eval_points(&l, &us), expected);

        let c = circle();
        let expected: Vec<Point3D> = us.iter().map(|&u| c.eval_point(u)).collect();
        assert_eq!(ParametricCurve3D::eval_points(&c, &us), expected);

        let e = ellipse();
        let expected: Vec<Point3D> = us.iter().map(|&u| e.eval_point(u)).collect();
        assert_eq!(ParametricCurve3D::eval_points(&e, &us), expected);

        let p = parabola();
        let expected: Vec<Point3D> = us.iter().map(|&u| p.eval_point(u)).collect();
        assert_eq!(ParametricCurve3D::eval_points(&p, &us), expected);

        let h = hyperbola();
        let expected: Vec<Point3D> = us.iter().map(|&u| h.eval_point(u)).collect();
        assert_eq!(ParametricCurve3D::eval_points(&h, &us), expected);

        let b = bspline();
        let expected: Vec<Point3D> = us.iter().map(|&u| b.eval_point(u)).collect();
        assert_eq!(ParametricCurve3D::eval_points(&b, &us), expected);
    }

    #[test]
    fn test_curve3d_enum_eval_points_matches_mapped_eval_point() {
        let us = [0.0, 0.5, 1.0, -2.0];
        let curves: Vec<Curve3D> = vec![
            line().into(),
            circle().into(),
            ellipse().into(),
            parabola().into(),
            hyperbola().into(),
            bspline().into(),
        ];
        for curve in &curves {
            let expected: Vec<Point3D> = us.iter().map(|&u| curve.eval_point(u)).collect();
            assert_eq!(curve.eval_points(&us), expected);
        }
    }

    #[test]
    fn test_eval_derivative_delegates_to_inherent() {
        let l = line();
        assert_eq!(
            ParametricCurve3D::eval_derivative(&l, 0.5, 1),
            l.eval_derivative(0.5, 1)
        );

        let c = circle();
        assert_eq!(
            ParametricCurve3D::eval_derivative(&c, 0.5, 1),
            c.eval_derivative(0.5, 1)
        );

        let b = bspline();
        assert_eq!(
            ParametricCurve3D::eval_derivative(&b, 0.5, 1),
            b.eval_derivative(0.5, 1)
        );
    }

    // ---- elementary curves ----

    #[test]
    fn test_elementary_curve_agrees_with_curve3d() {
        let curves: [ElementaryCurve; 5] = [
            line().into(),
            circle().into(),
            ellipse().into(),
            parabola().into(),
            hyperbola().into(),
        ];
        for curve in curves {
            let wrapped = Curve3D::from(curve);
            assert_eq!(wrapped.as_elementary(), Some(curve));
            assert_eq!(curve.eval_point(0.5), wrapped.eval_point(0.5));
            assert_eq!(
                curve.eval_derivative(0.5, 2),
                wrapped.eval_derivative(0.5, 2)
            );
            assert_eq!(curve.bounds(), wrapped.bounds());
            assert_eq!(curve.period(), wrapped.period());
            assert_eq!(
                curve.eval_points(&[0.0, 0.5]),
                wrapped.eval_points(&[0.0, 0.5])
            );
        }
    }

    #[test]
    fn test_bspline_curve_is_not_elementary() {
        assert_eq!(Curve3D::from(bspline()).as_elementary(), None);
    }
}
