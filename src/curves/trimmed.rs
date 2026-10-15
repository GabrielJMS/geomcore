//! Curves restricted to a parameter interval of a basis curve: a segment of
//! a line, an arc of a circle or conic, a piece of a B-spline.
//!
//! A trimmed curve is geometry only: a basis curve and `[first, last]`. It
//! carries no vertices, orientation or tolerance; those belong to the
//! topology (edges) of a kernel built on top of geomcore.

use crate::curves::{
    BSplineCurve3D, Curve2D, Curve3D, ElementaryCurve, Line2D, Line3D, LineConstructionError,
    ParametricCurve2D, ParametricCurve3D, PointExtrema2D, PointExtrema3D,
};
use crate::projection::CurveProjection;
use crate::{Point2D, Point3D, Tolerance, Vector2D, Vector3D};
use std::fmt;

/// Error returned when a [`TrimmedCurve3D`] or [`TrimmedCurve2D`] cannot be
/// constructed from the given range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimmedCurveConstructionError {
    /// `first` or `last` is NaN or infinite.
    NonFinite,
    /// `first` is not strictly less than `last`.
    EmptyRange,
    /// The range leaves the bounds of a non-periodic basis.
    OutsideBasis,
    /// The range is longer than the period of a periodic basis.
    ExceedsPeriod,
}

impl fmt::Display for TrimmedCurveConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            TrimmedCurveConstructionError::NonFinite => "a range end is NaN or infinite",
            TrimmedCurveConstructionError::EmptyRange => "first is not less than last",
            TrimmedCurveConstructionError::OutsideBasis => "the range leaves the basis bounds",
            TrimmedCurveConstructionError::ExceedsPeriod => {
                "the range is longer than the basis period"
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for TrimmedCurveConstructionError {}

/// Validates `[first, last]` against a basis with the given bounds and
/// period. A periodic basis accepts any start, so a range may cross its
/// seam, but no more than one period.
fn check_range(
    first: f64,
    last: f64,
    bounds: (f64, f64),
    period: Option<f64>,
) -> Result<(), TrimmedCurveConstructionError> {
    if !first.is_finite() || !last.is_finite() {
        return Err(TrimmedCurveConstructionError::NonFinite);
    }
    if first >= last {
        return Err(TrimmedCurveConstructionError::EmptyRange);
    }
    match period {
        Some(period) if last - first > period * (1.0 + 1e-12) => {
            Err(TrimmedCurveConstructionError::ExceedsPeriod)
        }
        Some(_) => Ok(()),
        None if first < bounds.0 || last > bounds.1 => {
            Err(TrimmedCurveConstructionError::OutsideBasis)
        }
        None => Ok(()),
    }
}

/// The extrema of a trimmed curve from those of its basis: the basis
/// extrema inside `[first, last]` (shifted by whole periods into it), then
/// the two ends, ordered by ascending distance. On a tie the interior
/// extremum wins, and an end at the parameter of a kept extremum is
/// dropped.
fn restrict(
    extrema: Vec<CurveProjection>,
    (first, last): (f64, f64),
    period: Option<f64>,
    end_distance: impl Fn(f64) -> f64,
    tol: Tolerance,
) -> Vec<CurveProjection> {
    let mut out: Vec<CurveProjection> = extrema
        .into_iter()
        .filter_map(|e| {
            let t = match period {
                Some(period) => first + (e.parameter - first).rem_euclid(period),
                None => e.parameter,
            };
            (first <= t && t <= last).then_some(CurveProjection {
                parameter: t,
                distance: e.distance,
            })
        })
        .collect();
    for t in [first, last] {
        let same =
            |e: &CurveProjection| (e.parameter - t).abs() <= tol.parametric * (1.0 + t.abs());
        if !out.iter().any(same) {
            out.push(CurveProjection {
                parameter: t,
                distance: end_distance(t),
            });
        }
    }
    out.sort_by(|x, y| x.distance.total_cmp(&y.distance));
    out
}

/// A 3D curve restricted to the parameter range `[first, last]` of a basis
/// curve: a line segment, a circular or elliptic arc, a piece of a
/// B-spline.
///
/// The parametrisation is the basis's own, so `eval_point(u)` is the basis
/// point at `u` and [`bounds`](ParametricCurve3D::bounds) is
/// `(first, last)`. A trimmed curve is never periodic, even when it spans
/// a whole period of its basis. Evaluation outside the range is not
/// checked and returns the basis point.
///
/// The basis type defaults to [`Curve3D`], so `TrimmedCurve3D` alone holds
/// a piece of any curve; a concrete basis such as
/// `TrimmedCurve3D<Line3D>` keeps static dispatch.
///
/// # Examples
///
/// ```
/// use geomcore::curves::{ParametricCurve3D, TrimmedCurve3D};
/// use geomcore::{Circle3D, Point3D, Tolerance, Vector3D};
/// use std::f64::consts::PI;
///
/// // The upper half of the unit circle.
/// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
/// let arc = TrimmedCurve3D::new(circle, 0.0, PI).unwrap();
/// assert_eq!(arc.bounds(), (0.0, PI));
/// assert!(!arc.is_periodic());
///
/// // A point below the arc projects onto its nearer end.
/// let proj = arc.project_point(Point3D::new(0.5, -2.0, 0.0), Tolerance::DEFAULT);
/// assert_eq!(proj.parameter, 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrimmedCurve3D<C = Curve3D> {
    basis: C,
    first: f64,
    last: f64,
}

impl<C: ParametricCurve3D> TrimmedCurve3D<C> {
    /// Restricts `basis` to the parameter range `[first, last]`.
    ///
    /// The ends must be finite with `first < last`. On a non-periodic basis
    /// the range must lie within its [`bounds`](ParametricCurve3D::bounds)
    /// ([`TrimmedCurveConstructionError::OutsideBasis`] otherwise). On a
    /// periodic basis any start is accepted, so an arc may cross the seam,
    /// but the range spans at most one period
    /// ([`TrimmedCurveConstructionError::ExceedsPeriod`] otherwise).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::{TrimmedCurve3D, TrimmedCurveConstructionError};
    /// use geomcore::{Circle3D, Point3D, Vector3D};
    ///
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    /// // An arc across the seam at angle 0.
    /// assert!(TrimmedCurve3D::new(circle, -1.0, 1.0).is_ok());
    /// assert_eq!(
    ///     TrimmedCurve3D::new(circle, 0.0, 7.0).err(),
    ///     Some(TrimmedCurveConstructionError::ExceedsPeriod)
    /// );
    /// ```
    pub fn new(basis: C, first: f64, last: f64) -> Result<Self, TrimmedCurveConstructionError> {
        check_range(first, last, basis.bounds(), basis.period())?;
        Ok(TrimmedCurve3D { basis, first, last })
    }

    /// Returns the basis curve.
    pub fn basis(&self) -> &C {
        &self.basis
    }

    /// Consumes the trimmed curve and returns its basis.
    pub fn into_basis(self) -> C {
        self.basis
    }

    /// Returns the start of the parameter range.
    pub fn first(&self) -> f64 {
        self.first
    }

    /// Returns the end of the parameter range.
    pub fn last(&self) -> f64 {
        self.last
    }

    /// Returns the point at the start of the range.
    pub fn start_point(&self) -> Point3D {
        self.basis.eval_point(self.first)
    }

    /// Returns the point at the end of the range.
    pub fn end_point(&self) -> Point3D {
        self.basis.eval_point(self.last)
    }

    /// Returns whether the piece's ends meet: a piece spanning a full
    /// period of a periodic basis, or whose start and end points coincide
    /// within `tol.confusion`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::TrimmedCurve3D;
    /// use geomcore::{Circle3D, Point3D, Tolerance, Vector3D};
    ///
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    /// let whole = TrimmedCurve3D::new(circle, 0.0, std::f64::consts::TAU).unwrap();
    /// assert!(whole.is_closed(Tolerance::DEFAULT));
    /// let arc = TrimmedCurve3D::new(circle, 0.0, 1.0).unwrap();
    /// assert!(!arc.is_closed(Tolerance::DEFAULT));
    /// ```
    pub fn is_closed(&self, tol: Tolerance) -> bool {
        if let Some(period) = self.basis.period()
            && self.last - self.first >= period - tol.parametric
        {
            return true;
        }
        self.start_point().distance(self.end_point()) <= tol.confusion
    }

    /// Evaluates the basis point at parameter `u`.
    pub fn eval_point(&self, u: f64) -> Point3D {
        self.basis.eval_point(u)
    }

    /// Evaluates the basis points at each parameter in `us`.
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        self.basis.eval_points(us)
    }

    /// Evaluates the basis derivative of the given `order` at parameter `u`.
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        self.basis.eval_derivative(u, order)
    }
}

impl<C: ParametricCurve3D + Into<Curve3D>> TrimmedCurve3D<C> {
    /// Moves the basis into a [`Curve3D`], keeping the range, so pieces of
    /// different curve types fit in one collection.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::TrimmedCurve3D;
    /// use geomcore::{Circle3D, Point3D, Vector3D};
    ///
    /// let segment = TrimmedCurve3D::segment(Point3D::ORIGIN, Point3D::new(1.0, 0.0, 0.0)).unwrap();
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    /// let arc = TrimmedCurve3D::new(circle, 0.0, 1.0).unwrap();
    /// let pieces: Vec<TrimmedCurve3D> = vec![segment.into_curve3d(), arc.into_curve3d()];
    /// assert_eq!(pieces.len(), 2);
    /// ```
    pub fn into_curve3d(self) -> TrimmedCurve3D<Curve3D> {
        TrimmedCurve3D {
            basis: self.basis.into(),
            first: self.first,
            last: self.last,
        }
    }
}

impl<C: PointExtrema3D> TrimmedCurve3D<C> {
    /// Projects `point` onto the trimmed curve: the closest of the basis
    /// extrema inside the range and the two ends. Distances within
    /// `tol.confusion` snap to `0.0`, matching
    /// [`TrimmedCurve3D::contains`].
    ///
    /// A query point with a non-finite coordinate projects to a NaN
    /// parameter and distance.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::TrimmedCurve3D;
    /// use geomcore::{Point3D, Tolerance};
    ///
    /// let segment = TrimmedCurve3D::segment(Point3D::ORIGIN, Point3D::new(2.0, 0.0, 0.0)).unwrap();
    /// let proj = segment.project_point(Point3D::new(5.0, 1.0, 0.0), Tolerance::DEFAULT);
    /// assert_eq!(proj.parameter, 2.0);
    /// assert_eq!(proj.distance, 10.0_f64.sqrt());
    /// ```
    pub fn project_point(&self, point: Point3D, tol: Tolerance) -> CurveProjection {
        PointExtrema3D::extrema(self, point, tol)
            .into_iter()
            .next()
            .unwrap_or(CurveProjection::UNDEFINED)
    }

    /// Projects each point in `points` onto the trimmed curve.
    pub fn project_points(&self, points: &[Point3D], tol: Tolerance) -> Vec<CurveProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }

    /// Returns whether `point` lies on the trimmed curve within
    /// `tol.confusion`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::TrimmedCurve3D;
    /// use geomcore::{Point3D, Tolerance};
    ///
    /// let segment = TrimmedCurve3D::segment(Point3D::ORIGIN, Point3D::new(2.0, 0.0, 0.0)).unwrap();
    /// assert!(segment.contains(Point3D::new(1.0, 0.0, 0.0), Tolerance::DEFAULT));
    /// assert!(!segment.contains(Point3D::new(3.0, 0.0, 0.0), Tolerance::DEFAULT));
    /// ```
    pub fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        self.project_point(point, tol).distance <= tol.confusion
    }
}

impl TrimmedCurve3D<Line3D> {
    /// The segment from `start` to `end`: the line through them, trimmed
    /// to `[0, |end - start|]`, so the parameter is the arc length from
    /// `start`.
    ///
    /// Returns [`LineConstructionError::ConfusedPoints`] when the points
    /// coincide and [`LineConstructionError::NonFinite`] for a non-finite
    /// coordinate.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::TrimmedCurve3D;
    /// use geomcore::Point3D;
    ///
    /// let end = Point3D::new(3.0, 4.0, 0.0);
    /// let segment = TrimmedCurve3D::segment(Point3D::ORIGIN, end).unwrap();
    /// assert_eq!(segment.last(), 5.0);
    /// assert_eq!(segment.end_point(), end);
    /// ```
    pub fn segment(start: Point3D, end: Point3D) -> Result<Self, LineConstructionError> {
        if !start.is_finite() || !end.is_finite() {
            return Err(LineConstructionError::NonFinite);
        }
        let line = Line3D::from_two_points(start, end)?;
        Ok(TrimmedCurve3D {
            basis: line,
            first: 0.0,
            last: start.distance(end),
        })
    }
}

impl<C: ParametricCurve3D> ParametricCurve3D for TrimmedCurve3D<C> {
    fn eval_point(&self, u: f64) -> Point3D {
        self.basis.eval_point(u)
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        self.basis.eval_derivative(u, order)
    }

    fn eval_d1(&self, u: f64) -> (Point3D, Vector3D) {
        self.basis.eval_d1(u)
    }

    fn eval_d2(&self, u: f64) -> (Point3D, Vector3D, Vector3D) {
        self.basis.eval_d2(u)
    }

    fn bounds(&self) -> (f64, f64) {
        (self.first, self.last)
    }

    fn period(&self) -> Option<f64> {
        None
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        self.basis.eval_points(us)
    }

    fn is_closed(&self, tol: Tolerance) -> bool {
        TrimmedCurve3D::is_closed(self, tol)
    }

    fn as_bspline(&self) -> Option<&BSplineCurve3D> {
        self.basis.as_bspline()
    }

    fn as_elementary(&self) -> Option<ElementaryCurve> {
        self.basis.as_elementary()
    }
}

impl<C: PointExtrema3D> PointExtrema3D for TrimmedCurve3D<C> {
    fn extrema(&self, point: Point3D, tol: Tolerance) -> Vec<CurveProjection> {
        if !point.is_finite() {
            return Vec::new();
        }
        let end_distance = |t: f64| {
            crate::projection::snap_distance(
                self.basis.eval_point(t).distance(point),
                tol.confusion,
            )
        };
        restrict(
            self.basis.extrema(point, tol),
            (self.first, self.last),
            self.basis.period(),
            end_distance,
            tol,
        )
    }
}

/// A 2D curve restricted to the parameter range `[first, last]` of a basis
/// curve: a line segment or a circular arc, typically a trimmed pcurve in
/// a surface's parameter space. Mirrors [`TrimmedCurve3D`].
///
/// # Examples
///
/// ```
/// use geomcore::curves::{ParametricCurve2D, TrimmedCurve2D};
/// use geomcore::{Point2D, Tolerance};
///
/// let segment = TrimmedCurve2D::segment(Point2D::ORIGIN, Point2D::new(0.0, 2.0)).unwrap();
/// assert_eq!(segment.bounds(), (0.0, 2.0));
/// let proj = segment.project_point(Point2D::new(1.0, -1.0), Tolerance::DEFAULT);
/// assert_eq!(proj.parameter, 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrimmedCurve2D<C = Curve2D> {
    basis: C,
    first: f64,
    last: f64,
}

impl<C: ParametricCurve2D> TrimmedCurve2D<C> {
    /// Restricts `basis` to the parameter range `[first, last]`, under the
    /// rules of [`TrimmedCurve3D::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::{TrimmedCurve2D, TrimmedCurveConstructionError};
    /// use geomcore::{Line2D, Point2D, Vector2D};
    ///
    /// let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// assert!(TrimmedCurve2D::new(line, -1.0, 1.0).is_ok());
    /// assert_eq!(
    ///     TrimmedCurve2D::new(line, 1.0, 1.0).err(),
    ///     Some(TrimmedCurveConstructionError::EmptyRange)
    /// );
    /// ```
    pub fn new(basis: C, first: f64, last: f64) -> Result<Self, TrimmedCurveConstructionError> {
        check_range(first, last, basis.bounds(), basis.period())?;
        Ok(TrimmedCurve2D { basis, first, last })
    }

    /// Returns the basis curve.
    pub fn basis(&self) -> &C {
        &self.basis
    }

    /// Consumes the trimmed curve and returns its basis.
    pub fn into_basis(self) -> C {
        self.basis
    }

    /// Returns the start of the parameter range.
    pub fn first(&self) -> f64 {
        self.first
    }

    /// Returns the end of the parameter range.
    pub fn last(&self) -> f64 {
        self.last
    }

    /// Returns the point at the start of the range.
    pub fn start_point(&self) -> Point2D {
        self.basis.eval_point(self.first)
    }

    /// Returns the point at the end of the range.
    pub fn end_point(&self) -> Point2D {
        self.basis.eval_point(self.last)
    }

    /// Evaluates the basis point at parameter `u`.
    pub fn eval_point(&self, u: f64) -> Point2D {
        self.basis.eval_point(u)
    }

    /// Evaluates the basis points at each parameter in `us`.
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point2D> {
        self.basis.eval_points(us)
    }

    /// Evaluates the basis derivative of the given `order` at parameter `u`.
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector2D {
        self.basis.eval_derivative(u, order)
    }
}

impl<C: ParametricCurve2D + Into<Curve2D>> TrimmedCurve2D<C> {
    /// Moves the basis into a [`Curve2D`], keeping the range, so pieces of
    /// different curve types fit in one collection.
    pub fn into_curve2d(self) -> TrimmedCurve2D<Curve2D> {
        TrimmedCurve2D {
            basis: self.basis.into(),
            first: self.first,
            last: self.last,
        }
    }
}

impl<C: PointExtrema2D> TrimmedCurve2D<C> {
    /// Projects `point` onto the trimmed curve: the closest of the basis
    /// extrema inside the range and the two ends. Distances within
    /// `tol.confusion` snap to `0.0`, matching
    /// [`TrimmedCurve2D::contains`].
    ///
    /// A query point with a non-finite coordinate projects to a NaN
    /// parameter and distance.
    pub fn project_point(&self, point: Point2D, tol: Tolerance) -> CurveProjection {
        PointExtrema2D::extrema(self, point, tol)
            .into_iter()
            .next()
            .unwrap_or(CurveProjection::UNDEFINED)
    }

    /// Projects each point in `points` onto the trimmed curve.
    pub fn project_points(&self, points: &[Point2D], tol: Tolerance) -> Vec<CurveProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }

    /// Returns whether `point` lies on the trimmed curve within
    /// `tol.confusion`.
    pub fn contains(&self, point: Point2D, tol: Tolerance) -> bool {
        self.project_point(point, tol).distance <= tol.confusion
    }
}

impl TrimmedCurve2D<Line2D> {
    /// The segment from `start` to `end`: the line through them, trimmed
    /// to `[0, |end - start|]`, so the parameter is the arc length from
    /// `start`.
    ///
    /// Returns [`LineConstructionError::ConfusedPoints`] when the points
    /// coincide and [`LineConstructionError::NonFinite`] for a non-finite
    /// coordinate.
    pub fn segment(start: Point2D, end: Point2D) -> Result<Self, LineConstructionError> {
        if !start.is_finite() || !end.is_finite() {
            return Err(LineConstructionError::NonFinite);
        }
        let line = Line2D::from_two_points(start, end)?;
        Ok(TrimmedCurve2D {
            basis: line,
            first: 0.0,
            last: start.distance(end),
        })
    }
}

impl<C: ParametricCurve2D> ParametricCurve2D for TrimmedCurve2D<C> {
    fn eval_point(&self, u: f64) -> Point2D {
        self.basis.eval_point(u)
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector2D {
        self.basis.eval_derivative(u, order)
    }

    fn bounds(&self) -> (f64, f64) {
        (self.first, self.last)
    }

    fn period(&self) -> Option<f64> {
        None
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point2D> {
        self.basis.eval_points(us)
    }
}

impl<C: PointExtrema2D> PointExtrema2D for TrimmedCurve2D<C> {
    fn extrema(&self, point: Point2D, tol: Tolerance) -> Vec<CurveProjection> {
        if !point.is_finite() {
            return Vec::new();
        }
        let end_distance = |t: f64| {
            crate::projection::snap_distance(
                self.basis.eval_point(t).distance(point),
                tol.confusion,
            )
        };
        restrict(
            self.basis.extrema(point, tol),
            (self.first, self.last),
            self.basis.period(),
            end_distance,
            tol,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intersect::{curve_curve_extrema, intersect_curve_curve, intersect_curve_surface};
    use crate::{Circle2D, Circle3D, Ellipse3D, Sphere};
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    const TOL: Tolerance = Tolerance::DEFAULT;

    fn circle() -> Circle3D {
        Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    // ---- construction ----

    #[test]
    fn test_new_rejects_non_finite_ends() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let nf = Some(TrimmedCurveConstructionError::NonFinite);
            assert_eq!(TrimmedCurve3D::new(circle(), bad, 1.0).err(), nf);
            assert_eq!(TrimmedCurve3D::new(circle(), 0.0, bad).err(), nf);
        }
    }

    #[test]
    fn test_new_rejects_empty_and_reversed_ranges() {
        let empty = Some(TrimmedCurveConstructionError::EmptyRange);
        assert_eq!(TrimmedCurve3D::new(circle(), 1.0, 1.0).err(), empty);
        assert_eq!(TrimmedCurve3D::new(circle(), 2.0, 1.0).err(), empty);
    }

    #[test]
    fn test_new_periodic_basis_accepts_any_start_up_to_one_period() {
        assert!(TrimmedCurve3D::new(circle(), -1.0, 1.0).is_ok());
        assert!(TrimmedCurve3D::new(circle(), 10.0, 10.0 + TAU).is_ok());
        assert_eq!(
            TrimmedCurve3D::new(circle(), 0.0, TAU + 1e-6).err(),
            Some(TrimmedCurveConstructionError::ExceedsPeriod)
        );
    }

    #[test]
    fn test_new_open_basis_stays_within_bounds() {
        let poles = vec![Point3D::ORIGIN, Point3D::new(2.0, 0.0, 0.0)];
        let b = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
        assert!(TrimmedCurve3D::new(b.clone(), 0.0, 1.0).is_ok());
        assert_eq!(
            TrimmedCurve3D::new(b, -0.1, 0.5).err(),
            Some(TrimmedCurveConstructionError::OutsideBasis)
        );
    }

    #[test]
    fn test_segment_rejects_confused_and_non_finite_points() {
        let p = Point3D::new(1.0, 2.0, 3.0);
        assert_eq!(
            TrimmedCurve3D::segment(p, p).err(),
            Some(LineConstructionError::ConfusedPoints)
        );
        assert_eq!(
            TrimmedCurve3D::segment(p, Point3D::new(f64::NAN, 0.0, 0.0)).err(),
            Some(LineConstructionError::NonFinite)
        );
        assert_eq!(
            TrimmedCurve2D::segment(Point2D::ORIGIN, Point2D::ORIGIN).err(),
            Some(LineConstructionError::ConfusedPoints)
        );
    }

    // ---- evaluation ----

    #[test]
    fn test_segment_is_parametrised_by_arc_length() {
        let (a, b) = (Point3D::new(1.0, 1.0, 1.0), Point3D::new(1.0, 4.0, 5.0));
        let s = TrimmedCurve3D::segment(a, b).unwrap();
        assert_eq!(s.bounds(), (0.0, 5.0));
        assert_eq!(s.start_point(), a);
        assert!(s.end_point().distance(b) < 1e-12);
        assert!(s.eval_point(2.5).distance(Point3D::new(1.0, 2.5, 3.0)) < 1e-12);
    }

    #[test]
    fn test_trimmed_is_never_periodic() {
        let full = TrimmedCurve3D::new(circle(), 0.0, TAU).unwrap();
        assert_eq!(full.period(), None);
        assert_eq!(full.bounds(), (0.0, TAU));
    }

    #[test]
    fn test_evaluation_delegates_to_basis() {
        let arc = TrimmedCurve3D::new(circle(), 0.5, 2.0).unwrap();
        assert_eq!(arc.eval_point(1.0), circle().eval_point(1.0));
        assert_eq!(
            arc.eval_derivative(1.0, 2),
            circle().eval_derivative(1.0, 2)
        );
        assert_eq!(
            arc.eval_points(&[0.5, 2.0]),
            vec![arc.start_point(), arc.end_point()]
        );
    }

    // ---- projection ----

    #[test]
    fn test_segment_projection_clamps_to_ends() {
        let s = TrimmedCurve3D::segment(Point3D::ORIGIN, Point3D::new(2.0, 0.0, 0.0)).unwrap();
        let inside = s.project_point(Point3D::new(1.5, 1.0, 0.0), TOL);
        assert_eq!((inside.parameter, inside.distance), (1.5, 1.0));
        let before = s.project_point(Point3D::new(-3.0, 4.0, 0.0), TOL);
        assert_eq!((before.parameter, before.distance), (0.0, 5.0));
        let after = s.project_point(Point3D::new(5.0, 0.0, 4.0), TOL);
        assert_eq!((after.parameter, after.distance), (2.0, 5.0));
    }

    #[test]
    fn test_arc_across_seam_projects_inside_and_onto_ends() {
        // Angles [-PI/2, PI/2]: the right half of the unit circle.
        let arc = TrimmedCurve3D::new(circle(), 1.5 * PI, 2.5 * PI).unwrap();
        // Basis parameter 0 is shifted by a period into the range.
        let right = arc.project_point(Point3D::new(3.0, 0.0, 0.0), TOL);
        assert!(close(right.parameter, TAU));
        assert!(close(right.distance, 2.0));
        // Left of the arc: the nearer end.
        let left = arc.project_point(Point3D::new(-3.0, 0.1, 0.0), TOL);
        assert!(close(left.parameter, 2.5 * PI));
    }

    #[test]
    fn test_projection_finds_an_interior_local_minimum() {
        // The global closest point of (0, 0.5) on this ellipse is at PI/2,
        // outside the lower half kept here; the lower half's closest point
        // is the interior local minimum at 3PI/2, nearer than either end.
        let e = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5).unwrap();
        let lower = TrimmedCurve3D::new(e, PI, TAU).unwrap();
        let proj = lower.project_point(Point3D::new(0.0, 0.5, 0.0), TOL);
        assert!(close(proj.parameter, 1.5 * PI));
        assert!(close(proj.distance, 2.0));
    }

    #[test]
    fn test_contains_respects_the_range() {
        let arc = TrimmedCurve3D::new(circle(), 0.0, FRAC_PI_2).unwrap();
        assert!(arc.contains(Point3D::new(0.0, 1.0, 0.0), TOL));
        assert!(!arc.contains(Point3D::new(0.0, -1.0, 0.0), TOL));
    }

    #[test]
    fn test_non_finite_query_is_undefined() {
        let s = TrimmedCurve3D::segment(Point3D::ORIGIN, Point3D::new(1.0, 0.0, 0.0)).unwrap();
        let proj = s.project_point(Point3D::new(f64::NAN, 0.0, 0.0), TOL);
        assert!(proj.parameter.is_nan() && proj.distance.is_nan());
    }

    #[test]
    fn test_extrema_keep_ends_and_sort_by_distance() {
        let s = TrimmedCurve3D::segment(Point3D::ORIGIN, Point3D::new(4.0, 0.0, 0.0)).unwrap();
        let ext = PointExtrema3D::extrema(&s, Point3D::new(1.0, 1.0, 0.0), TOL);
        let params: Vec<f64> = ext.iter().map(|e| e.parameter).collect();
        assert_eq!(params, vec![1.0, 0.0, 4.0]);
    }

    #[test]
    fn test_curve3d_basis_projects_like_the_concrete_basis() {
        let arc = TrimmedCurve3D::new(circle(), 1.0, 2.0).unwrap();
        let erased = arc.into_curve3d();
        let q = Point3D::new(-2.0, -0.5, 1.0);
        assert_eq!(arc.project_point(q, TOL), erased.project_point(q, TOL));
    }

    #[test]
    fn test_2d_arc_projection_and_containment() {
        let c = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
        let arc = TrimmedCurve2D::new(c, 0.0, FRAC_PI_2).unwrap();
        let proj = arc.project_point(Point2D::new(1.0, 1.0), TOL);
        assert!(close(proj.parameter, PI / 4.0));
        let end = arc.project_point(Point2D::new(-1.0, -5.0), TOL);
        assert_eq!(end.parameter, 0.0);
        assert!(arc.contains(Point2D::new(0.0, 2.0), TOL));
        assert!(!arc.contains(Point2D::new(-2.0, 0.0), TOL));
        let erased = arc.into_curve2d();
        assert_eq!(erased.bounds(), (0.0, FRAC_PI_2));
    }

    // ---- generic algorithms see the range ----

    #[test]
    fn test_intersect_curve_curve_ignores_hits_outside_the_ranges() {
        let s = TrimmedCurve3D::segment(Point3D::new(-2.0, 0.5, 0.0), Point3D::new(2.0, 0.5, 0.0))
            .unwrap();
        let full = intersect_curve_curve(&s, &circle(), TOL);
        assert_eq!(full.len(), 2);
        let upper_right = TrimmedCurve3D::new(circle(), 0.0, FRAC_PI_2).unwrap();
        let hits = intersect_curve_curve(&s, &upper_right, TOL);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].point.x > 0.0);
    }

    #[test]
    fn test_intersect_curve_surface_ignores_hits_outside_the_range() {
        let sphere = Sphere::new(Point3D::ORIGIN, 1.0).unwrap();
        let through =
            TrimmedCurve3D::segment(Point3D::new(-2.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0))
                .unwrap();
        assert_eq!(intersect_curve_surface(&through, &sphere, TOL).len(), 2);
        let half = TrimmedCurve3D::segment(Point3D::ORIGIN, Point3D::new(2.0, 0.0, 0.0)).unwrap();
        let hits = intersect_curve_surface(&half, &sphere, TOL);
        assert_eq!(hits.len(), 1);
        assert!(close(hits[0].curve_param, 1.0));
    }

    #[test]
    fn test_curve_curve_extrema_between_segments() {
        let a = TrimmedCurve3D::segment(Point3D::ORIGIN, Point3D::new(1.0, 0.0, 0.0)).unwrap();
        let b = TrimmedCurve3D::segment(Point3D::new(3.0, -1.0, 1.0), Point3D::new(3.0, 1.0, 1.0))
            .unwrap();
        let ext = curve_curve_extrema(&a, &b, TOL);
        assert!(close(ext[0].distance, 5.0_f64.sqrt()));
        assert!(close(ext[0].first_param, 1.0));
        assert!(close(ext[0].second_param, 1.0));
    }

    // ---- borrowed basis ----

    fn wavy() -> BSplineCurve3D {
        let poles = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 2.0, 0.5),
            Point3D::new(2.0, -1.0, 0.0),
            Point3D::new(3.0, 1.5, -0.5),
            Point3D::new(4.0, 0.0, 0.0),
        ];
        BSplineCurve3D::new(3, poles, vec![0.0, 0.5, 1.0], vec![4, 1, 4], false).unwrap()
    }

    #[test]
    fn test_borrowed_basis_matches_owned() {
        let curve = wavy();
        let owned = TrimmedCurve3D::new(curve.clone(), 0.1, 0.8).unwrap();
        let borrowed = TrimmedCurve3D::new(&curve, 0.1, 0.8).unwrap();
        let point = Point3D::new(2.0, 1.0, 1.0);
        assert_eq!(borrowed.bounds(), owned.bounds());
        assert_eq!(borrowed.extrema(point, TOL), owned.extrema(point, TOL));
        assert_eq!(
            borrowed.project_point(point, TOL),
            owned.project_point(point, TOL)
        );

        let line =
            TrimmedCurve3D::segment(Point3D::new(2.0, -3.0, 0.0), Point3D::new(2.0, 3.0, 0.0))
                .unwrap();
        assert_eq!(
            intersect_curve_curve(&borrowed, &line, TOL),
            intersect_curve_curve(&owned, &line, TOL)
        );
        assert_eq!(
            curve_curve_extrema(&line, &borrowed, TOL),
            curve_curve_extrema(&line, &owned, TOL)
        );
        let sphere = Sphere::new(Point3D::new(2.0, 0.0, 0.0), 1.0).unwrap();
        assert_eq!(
            intersect_curve_surface(&borrowed, &sphere, TOL),
            intersect_curve_surface(&owned, &sphere, TOL)
        );

        let circle = Circle2D::new(Point2D::new(1.0, 1.0), 2.0).unwrap();
        let owned = TrimmedCurve2D::new(circle, 1.0, 4.0).unwrap();
        let borrowed = TrimmedCurve2D::new(&circle, 1.0, 4.0).unwrap();
        let point = Point2D::new(-3.0, 0.5);
        assert_eq!(borrowed.extrema(point, TOL), owned.extrema(point, TOL));
        assert_eq!(
            borrowed.project_point(point, TOL),
            owned.project_point(point, TOL)
        );
    }
}
