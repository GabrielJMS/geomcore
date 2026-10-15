//! B-spline (NURBS) curves in 3D: de Boor evaluation with derivatives,
//! rational (weighted) poles, and periodic curves, a thin wrapper
//! over [`crate::curve_math::bspline`].
//!
//! The poles are stored once, with a separate weight per pole if rational;
//! the knot vector is expanded into its flat knot sequence at construction,
//! so evaluation never re-derives it.

use std::fmt;

use crate::curve_math::bspline as math;
use crate::curve_math::editing::{self, Elevation, Insertion, Lanes};
use crate::curves::{Curve2D, ParametrizeError};
use crate::math::boxtree::{BoxTree, Lazy};
use crate::math::{newton_min_1d, quadrature};
use crate::projection::CurveProjection;
use crate::surfaces::Surface;
use crate::{Point3D, Tolerance, Transform, Vector3D};

/// Error returned when a [`BSplineCurve3D`] cannot be constructed from the
/// given degree, poles, knots, multiplicities, and (for rational curves)
/// weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BSplineConstructionError {
    /// The degree is `< 1` or `> 25`.
    InvalidDegree,
    /// Fewer than two distinct knot values were given.
    TooFewKnots,
    /// The knot values are not strictly increasing.
    KnotsNotIncreasing,
    /// A multiplicity is too large: an interior multiplicity exceeds the
    /// degree, or (non-periodic) an end multiplicity exceeds `degree + 1`, or
    /// (periodic) any multiplicity exceeds the degree.
    MultiplicityTooLarge,
    /// The curve is periodic but the first and last multiplicities differ.
    PeriodicEndMultiplicityMismatch,
    /// The pole count is inconsistent with the knots and multiplicities:
    /// non-periodic requires `n_poles == sum(mults) - degree - 1`; periodic
    /// requires `n_poles == sum(mults) - mults[last]`.
    PoleCountMismatch,
    /// The number of weights does not equal the number of poles.
    WeightCountMismatch,
    /// A weight is zero or negative.
    NonPositiveWeight,
    /// Interpolation needs at least two points.
    TooFewPoints,
    /// Interpolation degree exceeds points minus one.
    DegreeTooHigh,
    /// Two consecutive interpolation points coincide.
    ConfusedPoints,
    /// More poles requested than data points.
    TooManyPoles,
}

impl fmt::Display for BSplineConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            BSplineConstructionError::InvalidDegree => "degree must be between 1 and 25",
            BSplineConstructionError::TooFewKnots => "fewer than two distinct knots",
            BSplineConstructionError::KnotsNotIncreasing => "knots are not strictly increasing",
            BSplineConstructionError::MultiplicityTooLarge => "a knot multiplicity is too large",
            BSplineConstructionError::PeriodicEndMultiplicityMismatch => {
                "periodic end multiplicities differ"
            }
            BSplineConstructionError::PoleCountMismatch => {
                "pole count is inconsistent with knots and multiplicities"
            }
            BSplineConstructionError::WeightCountMismatch => {
                "weight count does not match pole count"
            }
            BSplineConstructionError::NonPositiveWeight => "a weight is zero or negative",
            BSplineConstructionError::TooFewPoints => "interpolation needs at least two points",
            BSplineConstructionError::DegreeTooHigh => {
                "interpolation degree exceeds pole count minus one"
            }
            BSplineConstructionError::ConfusedPoints => {
                "two consecutive interpolation points coincide"
            }
            BSplineConstructionError::TooManyPoles => "more poles requested than data points",
        };
        f.write_str(message)
    }
}

impl std::error::Error for BSplineConstructionError {}

/// Error returned when knots cannot be inserted into a [`BSplineCurve3D`]
/// or a direction of a [`crate::BSplineSurface`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnotInsertionError {
    /// A knot to insert is NaN or infinite.
    NonFiniteKnot,
    /// A knot to insert lies outside the parameter bounds.
    KnotOutOfRange,
    /// The knots to insert are not in non-decreasing order.
    KnotsNotSorted,
    /// A multiplicity would exceed its maximum: the degree, or
    /// `degree + 1` at the end knots of a non-periodic B-spline.
    MultiplicityTooLarge,
}

impl fmt::Display for KnotInsertionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            KnotInsertionError::NonFiniteKnot => "a knot to insert is not finite",
            KnotInsertionError::KnotOutOfRange => {
                "a knot to insert is outside the parameter bounds"
            }
            KnotInsertionError::KnotsNotSorted => "the knots to insert are not sorted",
            KnotInsertionError::MultiplicityTooLarge => {
                "a knot multiplicity would exceed its maximum"
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for KnotInsertionError {}

/// Error returned when the degree of a [`BSplineCurve3D`] or of a direction
/// of a [`crate::BSplineSurface`] cannot be raised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegreeElevationError {
    /// The raised degree would exceed 25, the largest supported.
    DegreeTooHigh,
}

impl fmt::Display for DegreeElevationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DegreeElevationError::DegreeTooHigh => f.write_str("raised degree would exceed 25"),
        }
    }
}

impl std::error::Error for DegreeElevationError {}

/// Error returned when knots cannot be removed from a [`BSplineCurve3D`]
/// or a direction of a [`crate::BSplineSurface`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnotRemovalError {
    /// The largest deviation allowed is negative, NaN or infinite.
    InvalidDeviation,
    /// A knot to remove is NaN or infinite.
    NonFiniteKnot,
    /// A knot to remove lies outside the parameter bounds, or on one of
    /// them for a non-periodic B-spline, whose end knots stay.
    KnotOutOfRange,
    /// A knot to remove is not one of the knots.
    KnotNotFound,
}

impl fmt::Display for KnotRemovalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            KnotRemovalError::InvalidDeviation => {
                "the largest deviation allowed is not a finite non-negative distance"
            }
            KnotRemovalError::NonFiniteKnot => "a knot to remove is not finite",
            KnotRemovalError::KnotOutOfRange => {
                "a knot to remove is not strictly inside the parameter bounds"
            }
            KnotRemovalError::KnotNotFound => "a knot to remove is not a knot",
        };
        f.write_str(message)
    }
}

impl std::error::Error for KnotRemovalError {}

/// Error returned when a [`BSplineCurve3D`], or a direction of a
/// [`crate::BSplineSurface`], cannot be split at a parameter or have a
/// piece extracted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BSplineSplitError {
    /// A parameter is NaN or infinite.
    NonFinite,
    /// A parameter lies outside the bounds: a split parameter on any
    /// B-spline, a range end on a non-periodic one.
    OutOfRange,
    /// The range is empty (`first >= last`), or the split parameter is a
    /// bound, which would leave an empty piece.
    EmptyRange,
    /// The range is longer than the period of a periodic B-spline.
    ExceedsPeriod,
}

impl fmt::Display for BSplineSplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            BSplineSplitError::NonFinite => "a parameter is NaN or infinite",
            BSplineSplitError::OutOfRange => "a parameter is outside the bounds",
            BSplineSplitError::EmptyRange => "a piece would have an empty range",
            BSplineSplitError::ExceedsPeriod => "the range is longer than the period",
        };
        f.write_str(message)
    }
}

impl std::error::Error for BSplineSplitError {}

/// A B-spline (optionally rational, optionally periodic) curve in 3D,
/// evaluated by de Boor's algorithm.
///
/// Poles define the control polygon; `knots`/`multiplicities` describe the
/// (non-flattened) knot vector, expanded internally into the flat knot
/// sequence used for evaluation. A rational curve additionally carries a
/// `weight` per pole and is evaluated in homogeneous coordinates, then
/// projected back to Euclidean space via the quotient rule.
///
/// # Examples
///
/// A clamped cubic through six poles:
///
/// ```
/// use geomcore::{BSplineCurve3D, Point3D};
///
/// let poles = vec![
///     Point3D::new(0.0, 0.0, 0.0),
///     Point3D::new(1.0, 2.0, 0.0),
///     Point3D::new(2.0, 2.0, 1.0),
///     Point3D::new(3.0, 0.0, 1.0),
///     Point3D::new(4.0, 1.0, 0.0),
///     Point3D::new(5.0, 0.0, 0.0),
/// ];
/// let knots = vec![0.0, 1.0, 2.0, 3.0];
/// let mults = vec![4, 1, 1, 4];
/// let curve = BSplineCurve3D::new(3, poles, knots, mults, false).unwrap();
///
/// assert_eq!(curve.eval_point(0.0), Point3D::new(0.0, 0.0, 0.0));
/// assert_eq!(curve.bounds(), (0.0, 3.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BSplineCurve3D {
    knots: math::KnotVector,
    poles: Vec<Point3D>,
    /// One weight per pole if rational.
    weights: Option<Vec<f64>>,
    /// Bounding-box hierarchy of the knot spans, built on the first
    /// projection.
    spans: Lazy<SpanTree>,
}

/// The knot-span boundaries of a curve and the box hierarchy over its spans.
#[derive(Clone)]
struct SpanTree {
    breaks: Vec<f64>,
    tree: BoxTree,
}

impl BSplineCurve3D {
    /// Creates a non-rational (polynomial) B-spline curve.
    ///
    /// # Errors
    ///
    /// Returns a [`BSplineConstructionError`] if `degree`, `knots`,
    /// `multiplicities`, and `poles.len()` are not mutually consistent (see
    /// the error variants for the exact conditions checked).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D};
    ///
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert_eq!(curve.eval_point(0.5), Point3D::new(1.0, 0.0, 0.0));
    /// ```
    pub fn new(
        degree: usize,
        poles: Vec<Point3D>,
        knots: Vec<f64>,
        multiplicities: Vec<u32>,
        periodic: bool,
    ) -> Result<BSplineCurve3D, BSplineConstructionError> {
        Self::build(degree, poles, None, knots, multiplicities, periodic)
    }

    /// Creates a rational (NURBS) B-spline curve with one weight per pole.
    ///
    /// # Errors
    ///
    /// Returns [`BSplineConstructionError::WeightCountMismatch`] if
    /// `weights.len() != poles.len()`, or
    /// [`BSplineConstructionError::NonPositiveWeight`] if any weight is `<=
    /// 0`. Otherwise validates degree/knots/multiplicities/poles exactly as
    /// [`BSplineCurve3D::new`] does.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D};
    ///
    /// // Rational quadratic quarter circle.
    /// let w = std::f64::consts::FRAC_1_SQRT_2;
    /// let poles = vec![
    ///     Point3D::new(1.0, 0.0, 0.0),
    ///     Point3D::new(1.0, 1.0, 0.0),
    ///     Point3D::new(0.0, 1.0, 0.0),
    /// ];
    /// let curve = BSplineCurve3D::new_rational(
    ///     2,
    ///     poles,
    ///     vec![1.0, w, 1.0],
    ///     vec![0.0, 1.0],
    ///     vec![3, 3],
    ///     false,
    /// )
    /// .unwrap();
    /// let p = curve.eval_point(1.0);
    /// assert!((p.x - 0.0).abs() < 1e-9 && (p.y - 1.0).abs() < 1e-9);
    /// ```
    pub fn new_rational(
        degree: usize,
        poles: Vec<Point3D>,
        weights: Vec<f64>,
        knots: Vec<f64>,
        multiplicities: Vec<u32>,
        periodic: bool,
    ) -> Result<BSplineCurve3D, BSplineConstructionError> {
        Self::build(
            degree,
            poles,
            Some(weights),
            knots,
            multiplicities,
            periodic,
        )
    }

    fn build(
        degree: usize,
        poles: Vec<Point3D>,
        weights: Option<Vec<f64>>,
        knots: Vec<f64>,
        multiplicities: Vec<u32>,
        periodic: bool,
    ) -> Result<BSplineCurve3D, BSplineConstructionError> {
        if let Some(ws) = &weights {
            if ws.len() != poles.len() {
                return Err(BSplineConstructionError::WeightCountMismatch);
            }
            if ws.iter().any(|&w| w <= 0.0) {
                return Err(BSplineConstructionError::NonPositiveWeight);
            }
        }
        let knots = math::KnotVector::new(degree, knots, multiplicities, periodic, poles.len())?;
        Ok(BSplineCurve3D {
            knots,
            poles,
            weights,
            spans: Lazy::default(),
        })
    }

    /// A curve from parts already known to be consistent (as
    /// [`validate_direction`](math::validate_direction) and the weight
    /// checks would find them), `flat` being the flat knots of `knots` and
    /// `multiplicities`: fitting and editing results skip revalidation.
    pub(crate) fn from_validated_parts(
        degree: usize,
        poles: Vec<Point3D>,
        weights: Option<Vec<f64>>,
        knots: Vec<f64>,
        multiplicities: Vec<u32>,
        flat: Vec<f64>,
        periodic: bool,
    ) -> BSplineCurve3D {
        let knots = math::KnotVector::from_validated(degree, knots, multiplicities, flat, periodic);
        debug_assert_eq!(knots.pole_count(), poles.len());
        BSplineCurve3D {
            knots,
            poles,
            weights,
            spans: Lazy::default(),
        }
    }

    /// Returns the curve's degree.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D};
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert_eq!(curve.degree(), 1);
    /// ```
    pub fn degree(&self) -> usize {
        self.knots.degree()
    }

    /// Returns whether the curve is periodic.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D};
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert!(!curve.is_periodic());
    /// ```
    pub fn is_periodic(&self) -> bool {
        self.knots.is_periodic()
    }

    /// Returns whether the curve's ends meet: a periodic curve, or one
    /// whose first and last evaluated points coincide within
    /// `tol.confusion`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D, Tolerance};
    /// let poles = vec![
    ///     Point3D::new(0.0, 0.0, 0.0),
    ///     Point3D::new(1.0, 0.0, 0.0),
    ///     Point3D::new(0.0, 0.0, 0.0),
    /// ];
    /// let closed = BSplineCurve3D::new(1, poles, vec![0.0, 1.0, 2.0], vec![2, 1, 2], false).unwrap();
    /// assert!(closed.is_closed(Tolerance::DEFAULT));
    /// ```
    pub fn is_closed(&self, tol: Tolerance) -> bool {
        if self.is_periodic() {
            return true;
        }
        let (lo, hi) = self.bounds();
        self.eval_point(lo).distance(self.eval_point(hi)) <= tol.confusion
    }

    /// Returns whether the curve is rational (has per-pole weights).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D};
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert!(!curve.is_rational());
    /// ```
    pub fn is_rational(&self) -> bool {
        self.weights.is_some()
    }

    /// Returns the control poles.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D};
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles.clone(), vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert_eq!(curve.poles(), poles.as_slice());
    /// ```
    pub fn poles(&self) -> &[Point3D] {
        &self.poles
    }

    /// Returns the per-pole weights, or `None` if the curve is not rational.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D};
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert_eq!(curve.weights(), None);
    /// ```
    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    /// Returns the (non-flattened) knot values.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D};
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert_eq!(curve.knots(), &[0.0, 1.0]);
    /// ```
    pub fn knots(&self) -> &[f64] {
        self.knots.knots()
    }

    /// Returns the knot multiplicities.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D};
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert_eq!(curve.multiplicities(), &[2, 2]);
    /// ```
    pub fn multiplicities(&self) -> &[u32] {
        self.knots.multiplicities()
    }

    /// Returns the image of this curve under `transform`: every control
    /// pole maps point-wise while degree, knots, multiplicities,
    /// periodicity, and weights are unchanged.
    ///
    /// B-spline evaluation is affinely invariant, so mapping the poles is
    /// exact for every rigid, mirrored, or uniformly scaled transform.
    /// Infallible: pole images always form a valid net over the unchanged
    /// knot vector, even for a collapsing (zero-scale) transform.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D, Transform, Vector3D};
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// let moved = curve.transformed(Transform::translation(Vector3D::new(0.0, 5.0, 0.0)));
    /// assert_eq!(moved.eval_point(0.5), Point3D::new(1.0, 5.0, 0.0));
    /// ```
    pub fn transformed(&self, transform: Transform) -> BSplineCurve3D {
        let poles: Vec<Point3D> = self
            .poles
            .iter()
            .map(|&p| transform.apply_point(p))
            .collect();
        BSplineCurve3D {
            knots: self.knots.clone(),
            poles,
            weights: self.weights.clone(),
            spans: Lazy::default(),
        }
    }

    /// Returns the same curve with the knot `u` inserted `times` times:
    /// its multiplicity rises by `times` (or it becomes a new knot of
    /// multiplicity `times`) and the curve gains `times` poles, while every
    /// point and derivative stays the same.
    ///
    /// A rational curve is refined in homogeneous coordinates. On a
    /// periodic curve both bounds name the seam knot, whose first and last
    /// multiplicities rise together; the result stays periodic. Inserting
    /// zero times returns the curve unchanged. A value equal to an existing
    /// knot raises its multiplicity; any other value, however close,
    /// becomes a new knot.
    ///
    /// # Errors
    ///
    /// [`KnotInsertionError::NonFiniteKnot`] if `u` is NaN or infinite,
    /// [`KnotInsertionError::KnotOutOfRange`] if it lies outside
    /// [`BSplineCurve3D::bounds`], and
    /// [`KnotInsertionError::MultiplicityTooLarge`] if its multiplicity
    /// would exceed the degree (`degree + 1` at the end knots of a
    /// non-periodic curve, so a clamped end takes no more).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, KnotInsertionError, Point3D};
    ///
    /// let poles = vec![
    ///     Point3D::new(0.0, 0.0, 0.0),
    ///     Point3D::new(1.0, 2.0, 0.0),
    ///     Point3D::new(3.0, 2.0, 1.0),
    ///     Point3D::new(4.0, 0.0, 1.0),
    /// ];
    /// let curve = BSplineCurve3D::new(3, poles, vec![0.0, 1.0], vec![4, 4], false).unwrap();
    /// let refined = curve.insert_knot(0.25, 2).unwrap();
    /// assert_eq!(refined.knots(), &[0.0, 0.25, 1.0]);
    /// assert_eq!(refined.multiplicities(), &[4, 2, 4]);
    /// assert_eq!(refined.poles().len(), 6);
    /// for u in [0.0, 0.1, 0.25, 0.6, 1.0] {
    ///     assert!(refined.eval_point(u).distance(curve.eval_point(u)) < 1e-12);
    /// }
    /// assert_eq!(
    ///     curve.insert_knot(0.25, 4),
    ///     Err(KnotInsertionError::MultiplicityTooLarge)
    /// );
    /// ```
    pub fn insert_knot(&self, u: f64, times: u32) -> Result<BSplineCurve3D, KnotInsertionError> {
        let copies = [u; math::MAX_DEGREE + 1];
        // More copies than any multiplicity allows fail the same way.
        self.refine_knots(&copies[..(times as usize).min(copies.len())])
    }

    /// Returns the same curve with all of `knots` inserted in one pass:
    /// each value inserted once, a repeated value as many times as it
    /// appears. The curve gains one pole per value while every point and
    /// derivative stays the same; see [`BSplineCurve3D::insert_knot`] for
    /// rational and periodic curves.
    ///
    /// The poles are computed once for the whole list rather than knot by
    /// knot: each value blends `degree` poles, and the other poles are
    /// copied once.
    ///
    /// # Errors
    ///
    /// [`KnotInsertionError::NonFiniteKnot`] if a value is NaN or infinite,
    /// [`KnotInsertionError::KnotsNotSorted`] if `knots` is not in
    /// non-decreasing order, [`KnotInsertionError::KnotOutOfRange`] if a
    /// value lies outside [`BSplineCurve3D::bounds`], and
    /// [`KnotInsertionError::MultiplicityTooLarge`] as for
    /// [`BSplineCurve3D::insert_knot`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, KnotInsertionError, Point3D};
    ///
    /// let poles = vec![
    ///     Point3D::new(0.0, 0.0, 0.0),
    ///     Point3D::new(1.0, 2.0, 0.0),
    ///     Point3D::new(3.0, 2.0, 1.0),
    ///     Point3D::new(4.0, 0.0, 1.0),
    /// ];
    /// let curve = BSplineCurve3D::new(3, poles, vec![0.0, 1.0], vec![4, 4], false).unwrap();
    /// let refined = curve.refine_knots(&[0.25, 0.5, 0.5, 0.75]).unwrap();
    /// assert_eq!(refined.knots(), &[0.0, 0.25, 0.5, 0.75, 1.0]);
    /// assert_eq!(refined.multiplicities(), &[4, 1, 2, 1, 4]);
    /// for u in [0.0, 0.3, 0.5, 0.9, 1.0] {
    ///     assert!(refined.eval_point(u).distance(curve.eval_point(u)) < 1e-12);
    /// }
    /// assert_eq!(
    ///     curve.refine_knots(&[0.5, 0.25]),
    ///     Err(KnotInsertionError::KnotsNotSorted)
    /// );
    /// ```
    pub fn refine_knots(&self, knots: &[f64]) -> Result<BSplineCurve3D, KnotInsertionError> {
        if knots.is_empty() {
            return Ok(self.clone());
        }
        let (direction, poles, weights) = editing::edit(
            &Insertion(knots),
            &self.knots,
            &self.poles,
            self.weights.as_deref(),
            Lanes::CURVE,
        )?;
        Ok(BSplineCurve3D {
            knots: direction,
            poles,
            weights,
            spans: Lazy::default(),
        })
    }

    /// Returns the same curve with its degree raised by `amount`: it
    /// evaluates to the same points and derivatives everywhere, up to
    /// rounding.
    ///
    /// Every knot keeps its value and its multiplicity rises by `amount`,
    /// so the curve keeps its continuity at each knot; the pole count
    /// grows by `amount` per knot span. A rational curve is raised in
    /// homogeneous coordinates and stays rational, a periodic one stays
    /// periodic with the same bounds. An unclamped non-periodic curve (end
    /// multiplicities below `degree + 1`) comes back clamped at its bounds,
    /// with multiplicity `degree + amount + 1` there, and the knots outside
    /// its bounds dropped. An `amount` of 0 returns an identical copy.
    ///
    /// One sweep over the knot spans raises the degree by any amount at
    /// once (The NURBS Book, A5.9): each span is brought to Bézier form,
    /// raised by the Bernstein elevation formula, and the knot it shares
    /// with the previous span removed back down to its new multiplicity.
    ///
    /// # Errors
    ///
    /// [`DegreeElevationError::DegreeTooHigh`] if the raised degree would
    /// exceed 25.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D};
    ///
    /// // Rational quadratic quarter circle, raised to a cubic.
    /// let w = std::f64::consts::FRAC_1_SQRT_2;
    /// let poles = vec![
    ///     Point3D::new(1.0, 0.0, 0.0),
    ///     Point3D::new(1.0, 1.0, 0.0),
    ///     Point3D::new(0.0, 1.0, 0.0),
    /// ];
    /// let arc = BSplineCurve3D::new_rational(2, poles, vec![1.0, w, 1.0], vec![0.0, 1.0], vec![3, 3], false)
    ///     .unwrap();
    /// let cubic = arc.elevate_degree(1).unwrap();
    /// assert_eq!(cubic.degree(), 3);
    /// assert_eq!(cubic.multiplicities(), &[4, 4]);
    /// assert_eq!(cubic.poles().len(), 4);
    /// for i in 0..=10 {
    ///     let u = i as f64 / 10.0;
    ///     assert!(cubic.eval_point(u).distance(arc.eval_point(u)) < 1e-12);
    /// }
    /// ```
    pub fn elevate_degree(&self, amount: usize) -> Result<BSplineCurve3D, DegreeElevationError> {
        let (direction, poles, weights) = editing::edit(
            &Elevation(amount),
            &self.knots,
            &self.poles,
            self.weights.as_deref(),
            Lanes::CURVE,
        )?;
        Ok(BSplineCurve3D {
            knots: direction,
            poles,
            weights,
            spans: Lazy::default(),
        })
    }

    /// Returns the curve with the knot `u` removed up to `times` times,
    /// and how many times it was: each removal takes one copy of the knot
    /// and one pole away, as long as the curve stays within
    /// `max_deviation` of this one at every parameter. The first removal
    /// that would move it further stops, keeping the knot's remaining
    /// copies, so a knot that cannot be removed is kept and reported
    /// removed zero times.
    ///
    /// The deviation is bounded, not sampled: every point of the result
    /// lies within `max_deviation` of the point of this curve at the same
    /// parameter. Removing a knot just inserted gives back the poles of
    /// the curve before the insertion, up to rounding. `times` above the
    /// multiplicity removes every copy it can, and removing zero times
    /// returns the curve unchanged. A rational curve is solved in
    /// homogeneous coordinates, its bound carried over to the points, and
    /// keeps a knot whose removal would make a weight non-positive. On a
    /// periodic curve the seam keeps one copy, so the bounds stay; a knot
    /// whose neighbourhood does not fit in one period (fewer than
    /// `degree + 2` minus its multiplicity poles) is kept.
    ///
    /// # Errors
    ///
    /// [`KnotRemovalError::InvalidDeviation`] if `max_deviation` is
    /// negative, NaN or infinite, [`KnotRemovalError::NonFiniteKnot`] if `u`
    /// is NaN or infinite, [`KnotRemovalError::KnotOutOfRange`] if it lies
    /// outside [`BSplineCurve3D::bounds`] or, on a non-periodic curve, on
    /// one of them, and [`KnotRemovalError::KnotNotFound`] if it is not one
    /// of [`BSplineCurve3D::knots`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, KnotRemovalError, Point3D};
    ///
    /// // A polyline whose second vertex lies on the segment around it.
    /// let poles = vec![
    ///     Point3D::new(0.0, 0.0, 0.0),
    ///     Point3D::new(1.0, 0.0, 0.0),
    ///     Point3D::new(2.0, 0.0, 0.0),
    ///     Point3D::new(3.0, 1.0, 0.0),
    /// ];
    /// let knots = vec![0.0, 1.0, 2.0, 3.0];
    /// let curve = BSplineCurve3D::new(1, poles, knots, vec![2, 1, 1, 2], false).unwrap();
    /// let (simpler, removed) = curve.remove_knot(1.0, 1, 1e-9).unwrap();
    /// assert_eq!(removed, 1);
    /// assert_eq!(simpler.knots(), &[0.0, 2.0, 3.0]);
    /// assert_eq!(simpler.poles().len(), 3);
    /// // The corner at the third vertex stays.
    /// let (same, removed) = curve.remove_knot(2.0, 1, 1e-9).unwrap();
    /// assert_eq!(removed, 0);
    /// assert_eq!(same, curve);
    /// assert_eq!(
    ///     curve.remove_knot(1.5, 1, 1e-9),
    ///     Err(KnotRemovalError::KnotNotFound)
    /// );
    /// ```
    pub fn remove_knot(
        &self,
        u: f64,
        times: u32,
        max_deviation: f64,
    ) -> Result<(BSplineCurve3D, u32), KnotRemovalError> {
        let copies = [u; math::MAX_DEGREE + 1];
        let copies = &copies[..(times as usize).min(copies.len())];
        if copies.is_empty() {
            editing::check_removal(&self.knots, &[u], max_deviation)?;
        }
        let (curve, removed) = self.remove_knots(copies, max_deviation)?;
        Ok((curve, removed as u32))
    }

    /// Returns the curve with each of `knots` removed once, in order, a
    /// value repeated as many times as it appears, and how many copies
    /// were removed in all: every removal keeps the curve within
    /// `max_deviation` of this one at every parameter, however many are
    /// made, and a knot whose removal would move it further is kept.
    ///
    /// The whole list is removed in one buffer, much faster than removing
    /// knot by knot; see [`BSplineCurve3D::remove_knot`] for the deviation
    /// bound and for rational and periodic curves. Which knots go depends
    /// on the order: each removal uses part of the deviation allowed
    /// around the knot. Copies beyond a knot's multiplicity are skipped.
    ///
    /// # Errors
    ///
    /// As [`BSplineCurve3D::remove_knot`], for any value of `knots`; no
    /// knot is removed then.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D};
    ///
    /// let poles = vec![
    ///     Point3D::new(0.0, 0.0, 0.0),
    ///     Point3D::new(1.0, 2.0, 0.0),
    ///     Point3D::new(3.0, 2.0, 1.0),
    ///     Point3D::new(4.0, 0.0, 1.0),
    /// ];
    /// let curve = BSplineCurve3D::new(3, poles, vec![0.0, 1.0], vec![4, 4], false).unwrap();
    /// let refined = curve.refine_knots(&[0.25, 0.5, 0.5, 0.75]).unwrap();
    /// let (simpler, removed) = refined.remove_knots(&[0.25, 0.5, 0.5, 0.75], 1e-9).unwrap();
    /// assert_eq!(removed, 4);
    /// assert_eq!(simpler.knots(), curve.knots());
    /// for (p, q) in simpler.poles().iter().zip(curve.poles()) {
    ///     assert!(p.distance(*q) < 1e-12);
    /// }
    /// ```
    pub fn remove_knots(
        &self,
        knots: &[f64],
        max_deviation: f64,
    ) -> Result<(BSplineCurve3D, usize), KnotRemovalError> {
        editing::check_removal(&self.knots, knots, max_deviation)?;
        let removal = editing::remove_knots(
            &self.knots,
            &self.poles,
            self.weights.as_deref(),
            Lanes::CURVE,
            knots,
            max_deviation,
        );
        Ok(match removal {
            Some(removed) => (
                BSplineCurve3D {
                    knots: removed.knots,
                    poles: removed.points,
                    weights: removed.weights,
                    spans: Lazy::default(),
                },
                removed.count,
            ),
            None => (self.clone(), 0),
        })
    }

    /// Returns the polynomial (or rational) piece of each knot span within
    /// the [`BSplineCurve3D::bounds`], in order, as a Bézier B-spline: one
    /// span with both end knots of multiplicity `degree + 1`, its poles
    /// the span's Bézier control points.
    ///
    /// The pieces are found span after span (The NURBS Book, A5.6), each
    /// sharing its first pole with the previous piece's last. A periodic
    /// curve is decomposed from its lower bound.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D};
    ///
    /// let poles: Vec<Point3D> =
    ///     (0..6).map(|i| Point3D::new(i as f64, (i % 2) as f64, 0.0)).collect();
    /// let knots = vec![0.0, 1.0, 2.0, 3.0];
    /// let curve = BSplineCurve3D::new(3, poles, knots, vec![4, 1, 1, 4], false).unwrap();
    /// let pieces = curve.to_bezier_pieces();
    /// assert_eq!(pieces.len(), 3);
    /// assert_eq!(pieces[1].bounds(), (1.0, 2.0));
    /// assert_eq!(pieces[1].multiplicities(), &[4, 4]);
    /// assert!(pieces[1].eval_point(1.5).distance(curve.eval_point(1.5)) < 1e-12);
    /// ```
    pub fn to_bezier_pieces(&self) -> Vec<BSplineCurve3D> {
        let p = self.degree();
        let mut pieces = Vec::with_capacity(self.knots.knots().len() - 1);
        editing::curve_pieces(
            &self.knots,
            &self.poles,
            self.weights.as_deref(),
            |x, y, poles, weights| {
                pieces.push(BSplineCurve3D {
                    knots: math::KnotVector::bezier(p, x, y),
                    poles,
                    weights,
                    spans: Lazy::default(),
                });
            },
        );
        pieces
    }

    /// Returns the piece of the curve over `[first, last]` as a clamped,
    /// non-periodic B-spline of the same degree, whose bounds are exactly
    /// `(first, last)` and which evaluates as this curve there, points and
    /// derivatives.
    ///
    /// Both ends are raised to multiplicity `degree` in one refinement of
    /// the knot spans between them, an end already a knot of that
    /// multiplicity being left alone, and only the poles of the piece are
    /// kept, those no insertion touched bit for bit. A rational curve gives
    /// a rational piece. On a periodic curve, as for a
    /// [`crate::TrimmedCurve3D`], any `first` is accepted and the piece may
    /// cross the seam, `last` lying past the upper bound, but spans at most
    /// one period. Extracting the whole range of a non-periodic curve
    /// clamps it, and returns a clamped one unchanged.
    ///
    /// # Errors
    ///
    /// [`BSplineSplitError::NonFinite`] if an end is NaN or infinite,
    /// [`BSplineSplitError::EmptyRange`] if `first >= last`,
    /// [`BSplineSplitError::OutOfRange`] if the range leaves the
    /// [`BSplineCurve3D::bounds`] of a non-periodic curve, and
    /// [`BSplineSplitError::ExceedsPeriod`] if it is longer than the
    /// period of a periodic one.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, BSplineSplitError, Point3D};
    ///
    /// let poles = vec![
    ///     Point3D::new(0.0, 0.0, 0.0),
    ///     Point3D::new(1.0, 2.0, 0.0),
    ///     Point3D::new(2.0, 2.0, 1.0),
    ///     Point3D::new(3.0, 0.0, 1.0),
    ///     Point3D::new(4.0, 1.0, 0.0),
    /// ];
    /// let curve =
    ///     BSplineCurve3D::new(3, poles, vec![0.0, 1.0, 2.0], vec![4, 1, 4], false).unwrap();
    /// let piece = curve.extract(0.5, 1.5).unwrap();
    /// assert_eq!(piece.bounds(), (0.5, 1.5));
    /// assert_eq!(piece.multiplicities(), &[4, 1, 4]);
    /// for u in [0.5, 0.8, 1.2, 1.5] {
    ///     assert!(piece.eval_point(u).distance(curve.eval_point(u)) < 1e-12);
    /// }
    /// assert_eq!(curve.extract(1.5, 0.5), Err(BSplineSplitError::EmptyRange));
    /// ```
    pub fn extract(&self, first: f64, last: f64) -> Result<BSplineCurve3D, BSplineSplitError> {
        editing::check_piece(&self.knots, first, last)?;
        if editing::is_clamped_to(&self.knots, first, last) {
            return Ok(self.clone());
        }
        let [piece] = self.cut(&[first, last]);
        Ok(piece)
    }

    /// Splits the curve at `u` into the pieces over `[first, u]` and
    /// `[u, last]` of its [`BSplineCurve3D::bounds`]: clamped, non-periodic
    /// B-splines of the same degree, evaluating as this curve over their
    /// ranges and meeting at the point at `u`. A periodic curve is opened
    /// at its seam; to open it elsewhere, use
    /// [`BSplineCurve3D::extract`] over one period.
    ///
    /// `u` is raised to multiplicity `degree` by one refinement, which
    /// also clamps an unclamped end, and the poles are shared out between
    /// the two pieces.
    ///
    /// # Errors
    ///
    /// [`BSplineSplitError::NonFinite`] if `u` is NaN or infinite,
    /// [`BSplineSplitError::OutOfRange`] if it lies outside the bounds,
    /// and [`BSplineSplitError::EmptyRange`] if it is one of them.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, BSplineSplitError, Point3D};
    ///
    /// let poles = vec![
    ///     Point3D::new(0.0, 0.0, 0.0),
    ///     Point3D::new(1.0, 2.0, 0.0),
    ///     Point3D::new(3.0, 2.0, 1.0),
    ///     Point3D::new(4.0, 0.0, 1.0),
    /// ];
    /// let curve = BSplineCurve3D::new(3, poles, vec![0.0, 1.0], vec![4, 4], false).unwrap();
    /// let (left, right) = curve.split_at(0.25).unwrap();
    /// assert_eq!((left.bounds(), right.bounds()), ((0.0, 0.25), (0.25, 1.0)));
    /// assert_eq!(left.poles()[3], right.poles()[0]);
    /// assert!(left.eval_point(0.1).distance(curve.eval_point(0.1)) < 1e-12);
    /// assert!(right.eval_point(0.7).distance(curve.eval_point(0.7)) < 1e-12);
    /// assert_eq!(curve.split_at(1.0), Err(BSplineSplitError::EmptyRange));
    /// ```
    pub fn split_at(&self, u: f64) -> Result<(BSplineCurve3D, BSplineCurve3D), BSplineSplitError> {
        editing::check_split(&self.knots, u)?;
        let (first, last) = self.bounds();
        let [left, right] = self.cut(&[first, u, last]);
        Ok((left, right))
    }

    /// Returns the curve traversed the other way: its point at `u` is this
    /// curve's point at `first + last - u` over the same
    /// [`BSplineCurve3D::bounds`], so its derivatives of odd order change
    /// sign.
    ///
    /// The poles (and weights) are listed in reverse order and the knots
    /// mirrored, the bounds exactly; a periodic curve stays periodic, its
    /// poles also rotated to keep the seam. Reversing twice gives the
    /// curve back, the interior knots within rounding of the mirroring.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D};
    ///
    /// let poles = vec![
    ///     Point3D::new(0.0, 0.0, 0.0),
    ///     Point3D::new(1.0, 2.0, 0.0),
    ///     Point3D::new(3.0, 2.0, 1.0),
    ///     Point3D::new(4.0, 0.0, 1.0),
    /// ];
    /// let curve = BSplineCurve3D::new(3, poles, vec![0.0, 1.0], vec![4, 4], false).unwrap();
    /// let reversed = curve.reversed();
    /// assert_eq!(reversed.eval_point(0.0), curve.eval_point(1.0));
    /// let (d, e) = (curve.eval_derivative(0.3, 1), reversed.eval_derivative(0.7, 1));
    /// assert!((d + e).magnitude() < 1e-12);
    /// assert_eq!(reversed.reversed(), curve);
    /// ```
    pub fn reversed(&self) -> BSplineCurve3D {
        let old: Vec<usize> = (0..self.poles.len())
            .map(|i| editing::reversed_slot(&self.knots, i))
            .collect();
        BSplineCurve3D {
            knots: editing::reversed_knots(&self.knots),
            poles: old.iter().map(|&i| self.poles[i]).collect(),
            weights: self
                .weights
                .as_ref()
                .map(|w| old.iter().map(|&i| w[i]).collect()),
            spans: Lazy::default(),
        }
    }

    /// The curve's pieces between the increasing `cuts`, checked against
    /// its bounds.
    fn cut<const N: usize>(&self, cuts: &[f64]) -> [BSplineCurve3D; N] {
        let pieces = editing::cut(
            &self.knots,
            &self.poles,
            self.weights.as_deref(),
            Lanes::CURVE,
            cuts,
        );
        let pieces = pieces.into_iter().map(|piece| BSplineCurve3D {
            knots: piece.knots,
            poles: piece.points,
            weights: piece.weights,
            spans: Lazy::default(),
        });
        let pieces: Vec<BSplineCurve3D> = pieces.collect();
        pieces
            .try_into()
            .unwrap_or_else(|_| unreachable!("a piece between each two cuts"))
    }

    /// Returns the curve's parameter bounds `(first, last)`.
    ///
    /// For a clamped (non-periodic) curve this is the active span
    /// `(flat[degree], flat[len - 1 - degree])`. For a periodic curve this
    /// is `(knots[0], knots[last])`, the full period.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D};
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert_eq!(curve.bounds(), (0.0, 1.0));
    /// ```
    pub fn bounds(&self) -> (f64, f64) {
        self.knots.bounds()
    }

    /// Evaluates the point on the curve at parameter `u`.
    ///
    /// For periodic curves, `u` outside [`BSplineCurve3D::bounds`] is wrapped
    /// into the period before evaluation.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D};
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert_eq!(curve.eval_point(0.5), Point3D::new(1.0, 0.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64) -> Point3D {
        math::point(&self.knots, self.pole_view(), u)
    }

    /// Evaluates the points on the curve at each parameter in `us`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D};
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// let points = curve.eval_points(&[0.0, 0.5, 1.0]);
    /// assert_eq!(points[1], Point3D::new(1.0, 0.0, 0.0));
    /// ```
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        us.iter().map(|&u| self.eval_point(u)).collect()
    }

    /// Evaluates the derivative of the given `order` at parameter `u`.
    ///
    /// Only first and second derivatives are supported.
    ///
    /// # Panics
    ///
    /// Panics if `order == 0` (use [`BSplineCurve3D::eval_point`] for the
    /// position itself) or if `order > 2`, with a message stating that only
    /// first and second derivatives are supported.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D, Vector3D};
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert_eq!(curve.eval_derivative(0.5, 1), Vector3D::new(2.0, 0.0, 0.0));
    /// ```
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        match order {
            0 => panic!("eval_derivative: order must be >= 1 (use eval_point for order 0)"),
            1 => math::derivatives::<1>(&self.knots, self.pole_view(), u)[1],
            2 => math::derivatives::<2>(&self.knots, self.pole_view(), u)[2],
            _ => panic!(
                "eval_derivative: order {order} is not supported (only first and second derivatives are supported)"
            ),
        }
    }

    /// Evaluates the point and the first derivative at parameter `u`, from
    /// one evaluation of the basis functions.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D, Vector3D};
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// let (point, d1) = curve.eval_d1(0.5);
    /// assert_eq!(point, Point3D::new(1.0, 0.0, 0.0));
    /// assert_eq!(d1, Vector3D::new(2.0, 0.0, 0.0));
    /// ```
    pub fn eval_d1(&self, u: f64) -> (Point3D, Vector3D) {
        let [c, d1, _] = math::derivatives::<1>(&self.knots, self.pole_view(), u);
        (Point3D::ORIGIN + c, d1)
    }

    /// Evaluates the point and the first and second derivatives at
    /// parameter `u`, from one evaluation of the basis functions.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D, Vector3D};
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
    pub fn eval_d2(&self, u: f64) -> (Point3D, Vector3D, Vector3D) {
        let [c, d1, d2] = math::derivatives::<2>(&self.knots, self.pole_view(), u);
        (Point3D::ORIGIN + c, d1, d2)
    }

    /// The length of the curve over its bounds, within `max_error` as far
    /// as the quadrature's own error estimate tells: the speed `|C'|` is
    /// integrated over each knot span by adaptive Gauss–Kronrod quadrature
    /// on the span's Bernstein form, each span getting a share of
    /// `max_error` in proportion to its width. A `max_error` below what
    /// rounding allows gets as close as rounding does.
    ///
    /// The length over `[first, last]` is that of
    /// [`BSplineCurve3D::extract`]`(first, last)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D};
    ///
    /// // The parabola y = x² over [0, 1].
    /// let poles = vec![
    ///     Point3D::new(0.0, 0.0, 0.0),
    ///     Point3D::new(0.5, 0.0, 0.0),
    ///     Point3D::new(1.0, 1.0, 0.0),
    /// ];
    /// let curve = BSplineCurve3D::new(2, poles, vec![0.0, 1.0], vec![3, 3], false).unwrap();
    /// let exact = 0.5 * 5f64.sqrt() + 0.25 * (2.0 + 5f64.sqrt()).ln();
    /// assert!((curve.length(1e-12) - exact).abs() <= 1e-12);
    /// ```
    pub fn length(&self, max_error: f64) -> f64 {
        let (first, last) = self.bounds();
        let breaks = self.knots.breaks();
        let degree = self.degree();
        let mut net = [[0.0; 4]; math::LARGE];
        breaks
            .windows(2)
            .map(|span| {
                let (a, b) = (span[0], span[1]);
                math::span_bezier(&self.knots, self.pole_view(), 0.5 * (a + b), &mut net);
                let net = &net[..=degree];
                let share = max_error * (b - a) / (last - first);
                quadrature::integrate(&|t| bezier_speed(net, t), 0.0, 1.0, share)
            })
            .sum()
    }

    /// Computes the exact 2D representation of this B-spline curve in a
    /// surface's parameter space.
    ///
    /// No B-spline-curve/surface pair has a closed-form 2D image in this
    /// release, so this always returns [`ParametrizeError::NotAnalytic`]. The
    /// `surface` and `tol` arguments are accepted for signature parity with the analytic
    /// [`crate::Line3D::parametrize_on`] and
    /// [`crate::Circle3D::parametrize_on`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::ParametrizeError;
    /// use geomcore::{BSplineCurve3D, Plane, Point3D, Tolerance, Vector3D};
    ///
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert_eq!(curve.parametrize_on(&plane, Tolerance::DEFAULT), Err(ParametrizeError::NotAnalytic));
    /// ```
    pub fn parametrize_on(
        &self,
        surface: impl Into<Surface>,
        _tol: Tolerance,
    ) -> Result<Curve2D, ParametrizeError> {
        let _ = surface.into();
        Err(ParametrizeError::NotAnalytic)
    }

    /// Returns whether `point` lies on the curve: the curve is projected
    /// with [`BSplineCurve3D::project_point`] and the point counts as
    /// contained when the projected distance is within `tol.confusion`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D, Tolerance};
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// assert!(curve.contains(Point3D::new(1.0, 0.0, 0.0), tol));
    /// assert!(!curve.contains(Point3D::new(1.0, 1.0, 0.0), tol));
    /// ```
    pub fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        self.project_point(point, tol).distance <= tol.confusion
    }

    /// The local minima of the distance from `point` to the curve, ordered
    /// by ascending distance, plus the interval ends of an open curve.
    ///
    /// Every knot span is scanned exactly (see
    /// [`BSplineCurve3D::project_point`]): its interior minima are
    /// bracketed and refined by a Newton minimisation bounded to the
    /// bracket, and a knot is a minimum when the distance falls into it and
    /// rises out of it (a crease). Minima at the same point merge. The
    /// first entry is the global closest point.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D, Tolerance};
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// let extrema = curve.extrema(Point3D::new(1.0, 1.0, 0.0), Tolerance::DEFAULT);
    /// assert!(!extrema.is_empty());
    /// assert_eq!(extrema[0].distance, 1.0);
    /// ```
    pub fn extrema(&self, point: Point3D, tol: Tolerance) -> Vec<CurveProjection> {
        use crate::projection::snap_distance;
        let (first, last) = self.bounds();
        let breaks = &self.span_tree().breaks;
        let n = breaks.len() - 1;
        let mut params = vec![self.project_point(point, tol).parameter];
        let ends: Vec<SpanEnds> = (0..n)
            .map(|k| self.scan_span(point, breaks[k], breaks[k + 1], &mut params))
            .collect();
        for k in 1..n {
            if ends[k - 1].end_falling && ends[k].start_rising {
                params.push(breaks[k]);
            }
        }
        if !self.is_periodic() {
            params.push(first);
            params.push(last);
        } else if ends[n - 1].end_falling && ends[0].start_rising {
            params.push(first);
        }
        let mut out: Vec<CurveProjection> = Vec::new();
        let mut kept: Vec<Point3D> = Vec::new();
        for t in params {
            let p = self.eval_point(t);
            if kept.iter().any(|q| q.distance(p) <= tol.confusion) {
                continue;
            }
            kept.push(p);
            out.push(CurveProjection {
                parameter: t,
                distance: snap_distance(p.distance(point), tol.confusion),
            });
        }
        out.sort_by(|x, y| x.distance.total_cmp(&y.distance));
        out
    }

    /// Projects `point` onto the curve: the global minimum of the distance,
    /// the first entry of [`BSplineCurve3D::extrema`]. Distances within
    /// `tol.confusion` snap to `0.0`, matching [`BSplineCurve3D::contains`].
    ///
    /// Branch and bound over the knot spans: each span lies in the bounding
    /// box of its poles (convex hull property), so spans are visited by
    /// increasing box distance (through a box hierarchy built once per
    /// curve) until no remaining box can hold a closer point. A visited
    /// span is scanned exactly: on it the curve is a polynomial (or
    /// rational) piece, so the derivative of the squared distance has the
    /// sign of a polynomial whose roots are bracketed in Bernstein form.
    /// Every interior minimum is found however close basins sit, then
    /// refined to full precision by a Newton minimisation bounded to its
    /// bracket.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D, Tolerance};
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// let proj = curve.project_point(Point3D::new(1.0, 1.0, 0.0), Tolerance::DEFAULT);
    /// assert!((proj.parameter - 0.5).abs() < 1e-9);
    /// assert_eq!(proj.distance, 1.0);
    /// ```
    pub fn project_point(&self, point: Point3D, tol: Tolerance) -> CurveProjection {
        use crate::projection::snap_distance;
        let spans = self.span_tree();
        let breaks = &spans.breaks;
        // (distance, parameter) of the closest point so far.
        let mut best = (f64::INFINITY, breaks[0]);
        let mut minima = Vec::new();
        spans.tree.nearest_first(point, |k| {
            let (a, b) = (breaks[k], breaks[k + 1]);
            minima.clear();
            let ends = self.scan_span(point, a, b, &mut minima);
            let interior = minima
                .iter()
                .map(|&t| (self.eval_point(t).distance(point), t));
            for (d, t) in [(ends.distance[0], a), (ends.distance[1], b)]
                .into_iter()
                .chain(interior)
            {
                if d < best.0 {
                    best = (d, t);
                }
            }
            best.0
        });
        let (first, last) = self.bounds();
        let t = if self.is_periodic() && best.1 == last {
            first
        } else {
            best.1
        };
        CurveProjection {
            parameter: t,
            distance: snap_distance(self.eval_point(t).distance(point), tol.confusion),
        }
    }

    /// Projects each point in `points` onto the curve.
    ///
    /// Default-style batch wrapper over [`BSplineCurve3D::project_point`]:
    /// one native call per batch, mirroring [`BSplineCurve3D::eval_points`].
    pub fn project_points(&self, points: &[Point3D], tol: Tolerance) -> Vec<CurveProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }

    /// The homogeneous Bernstein control points `(x w, y w, z w, w)`
    /// (`w = 1` if nonrational) of the polynomial piece on the span
    /// containing `u` (pass a parameter inside the span), over the span's
    /// own interval, into `points[..=degree]`.
    pub(crate) fn span_bezier_points(&self, u: f64, points: &mut [[f64; 4]]) {
        math::span_bezier(&self.knots, self.pole_view(), u, points);
    }

    /// The poles as read by evaluation.
    fn pole_view(&self) -> math::Poles<'_> {
        math::Poles {
            points: &self.poles,
            weights: self.weights.as_deref(),
        }
    }

    /// The poles weighting the knot spans from the one containing `u0` to
    /// the one containing `u1`, `u0 <= u1` at most a period apart.
    pub(crate) fn run_poles(&self, u0: f64, u1: f64) -> impl Iterator<Item = Point3D> + '_ {
        let n = self.poles.len();
        let (first, last) = (
            self.knots.span_first_pole(u0),
            self.knots.span_first_pole(u1),
        );
        // Periodic pole indices wrap, so the run may end below its start.
        let count = ((last + n - first) % n + self.degree() + 1).min(n);
        (first..first + count).map(move |i| self.poles[i % n])
    }

    /// Calls `visit(span, piece)` in order for every knot span
    /// `span = (start, end)` meeting `[first, last]`, `piece` being the
    /// part of `[first, last]` inside it. A periodic curve repeats its
    /// spans every period, so a range across the seam takes them from
    /// consecutive periods.
    pub(crate) fn for_each_span_in(
        &self,
        first: f64,
        last: f64,
        mut visit: impl FnMut((f64, f64), (f64, f64)),
    ) {
        let (start, end) = self.bounds();
        let interior = self
            .knots()
            .iter()
            .copied()
            .filter(|&k| k > start && k < end);
        let (shifts, period) = match self.knots.period() {
            Some(period) => {
                let shift = |u: f64| ((u - start) / period).floor() as i64;
                (shift(first)..=shift(last), period)
            }
            None => (0..=0, 0.0),
        };
        let mut span_start = start + *shifts.start() as f64 * period;
        for shift in shifts {
            let offset = shift as f64 * period;
            for knot in interior.clone().chain([end]) {
                let span_end = knot + offset;
                let (a, b) = (span_start.max(first), span_end.min(last));
                if a < b {
                    visit((span_start, span_end), (a, b));
                }
                span_start = span_end;
            }
        }
    }

    /// The knot-span box hierarchy, built on first use.
    fn span_tree(&self) -> &SpanTree {
        self.spans.get_or_init(|| {
            let breaks = self.knots.breaks();
            let boxes = math::span_boxes(&self.knots, self.pole_view(), &breaks);
            SpanTree {
                tree: BoxTree::new(&boxes, 1),
                breaks,
            }
        })
    }

    /// Scans the knot span `[a, b]` for minima of the distance to `point`:
    /// pushes every interior local minimum (refined) to `minima` and
    /// returns how the distance behaves at the span ends.
    ///
    /// With the span's Bernstein control points, the homogeneous offset
    /// `A = P_w - point * w` and the weight `w` are polynomials of the
    /// degree `p`, and `d/dt |A / w|^2` has the sign of
    /// `G = (A' . A) w - |A|^2 w'` (`A' . A` if nonrational), of degree
    /// `3p - 1` (`2p - 1`). Interior minima are the roots where `G` rises
    /// through zero.
    fn scan_span(&self, point: Point3D, a: f64, b: f64, minima: &mut Vec<f64>) -> SpanEnds {
        // `G` has up to `3 * degree` coefficients.
        const SMALL: usize = 3 * math::SMALL;
        if self.degree() < math::SMALL {
            self.scan_span_sized::<SMALL>(point, a, b, minima)
        } else {
            self.scan_span_sized::<{ crate::math::bernstein::CAPACITY }>(point, a, b, minima)
        }
    }

    fn scan_span_sized<const N: usize>(
        &self,
        point: Point3D,
        a: f64,
        b: f64,
        minima: &mut Vec<f64>,
    ) -> SpanEnds {
        use crate::math::bernstein::{add_product, rising_roots, scale, unscale};
        let p = self.degree();
        let rational = self.is_rational();
        let mut points = [[0.0; 4]; math::LARGE];
        self.span_bezier_points(0.5 * (a + b), &mut points);
        let mut bezier = [[0.0; N]; 3];
        let mut weight = [1.0; N];
        for (k, x) in points.iter().enumerate().take(p + 1) {
            for c in 0..3 {
                bezier[c][k] = x[c];
            }
            weight[k] = x[3];
        }
        let q = [point.x, point.y, point.z];
        // Offsets `A` per coordinate and their differences (`A'` up to the
        // factor `p`, as `w'`), scaled; `magnitude` bounds their rounding.
        let mut offset = [[0.0; N]; 3];
        let mut slope = [[0.0; N]; 3];
        let mut magnitude: f64 = 0.0;
        for c in 0..3 {
            for k in 0..=p {
                offset[c][k] = bezier[c][k] - q[c] * weight[k];
                magnitude = magnitude.max(bezier[c][k].abs() + q[c].abs() * weight[k]);
            }
            for k in 0..p {
                slope[c][k] = offset[c][k + 1] - offset[c][k];
            }
            scale(&mut offset[c][..=p]);
            scale(&mut slope[c][..p]);
        }
        // `A' . A` (degree `2p - 1`) and, if rational, `|A|^2` (`2p`).
        let mut g = [0.0; N];
        let degree_g = if rational { 3 * p - 1 } else { 2 * p - 1 };
        if rational {
            let mut dot = [0.0; N];
            let mut square = [0.0; N];
            for c in 0..3 {
                add_product(&slope[c][..p], &offset[c][..=p], 1.0, &mut dot);
                add_product(&offset[c][..=p], &offset[c][..=p], 1.0, &mut square);
            }
            let mut weight_slope = [0.0; N];
            for k in 0..p {
                weight_slope[k] = weight[k + 1] - weight[k];
            }
            let mut scaled_weight = weight;
            scale(&mut scaled_weight[..=p]);
            scale(&mut weight_slope[..p]);
            add_product(&dot[..2 * p], &scaled_weight[..=p], 1.0, &mut g);
            add_product(&square[..=2 * p], &weight_slope[..p], -1.0, &mut g);
        } else {
            for c in 0..3 {
                add_product(&slope[c][..p], &offset[c][..=p], 1.0, &mut g);
            }
        }
        let g = &mut g[..=degree_g];
        unscale(g);
        let max_weight = weight[..=p].iter().fold(0.0f64, |m, &w| m.max(w));
        let tiny = 1e3 * f64::EPSILON * magnitude * magnitude * max_weight;
        let h = b - a;
        rising_roots::<N>(g, tiny, |s0, s1, s| {
            let bounds = (a + s0 * h, if s1 == 1.0 { b } else { a + s1 * h });
            minima.push(self.refine_projection(point, a + s * h, Some(bounds)));
        });
        // The curve passes through the end control points (whose scaled
        // offsets are unscaled: `C(p, 0) = C(p, p) = 1`).
        let end_distance = |k: usize| {
            (0..3)
                .map(|c| (offset[c][k] / weight[k]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        SpanEnds {
            distance: [end_distance(0), end_distance(p)],
            start_rising: g[0] >= -tiny,
            end_falling: g[degree_g] <= tiny,
        }
    }

    /// Bounded Newton minimisation of `|C(t) - point|^2` from `t` with
    /// exact second derivatives, within `bounds` (`None`: unbounded, for a
    /// smooth periodic curve); a periodic result is wrapped into
    /// [`BSplineCurve3D::bounds`].
    fn refine_projection(&self, point: Point3D, t: f64, bounds: Option<(f64, f64)>) -> f64 {
        let objective = |t: f64| {
            let [c, d1, d2] =
                math::derivatives::<2>(&self.knots, self.pole_view(), math::inward(t, bounds));
            let r = c - (point - Point3D::ORIGIN);
            (0.5 * r.dot(r), d1.dot(r), d1.dot(d1) + d2.dot(r))
        };
        let t = newton_min_1d(objective, t, bounds, 1e-14, 100);
        if self.is_periodic() {
            let (first, last) = self.bounds();
            first + (t - first).rem_euclid(last - first)
        } else {
            t
        }
    }
}

/// How the distance to a point behaves at the ends of one knot span.
struct SpanEnds {
    /// Distances at the span start and end.
    distance: [f64; 2],
    /// Whether the distance does not fall moving into the span from its
    /// start.
    start_rising: bool,
    /// Whether the distance does not rise moving to the span end.
    end_falling: bool,
}

/// The speed `|C'(t)|` at `t` of the Bézier piece with the homogeneous
/// control points `net`, over `[0, 1]`: de Casteljau down to the last two
/// points, whose difference gives the derivative.
fn bezier_speed(net: &[[f64; 4]], t: f64) -> f64 {
    let degree = net.len() - 1;
    let mut points = [[0.0; 4]; math::LARGE];
    points[..=degree].copy_from_slice(net);
    for level in (1..degree).rev() {
        for i in 0..=level {
            let next = points[i + 1];
            for (x, y) in points[i].iter_mut().zip(next) {
                *x += t * (y - *x);
            }
        }
    }
    let ([a, b], s) = ([points[0], points[1]], degree as f64);
    let h: [f64; 4] = std::array::from_fn(|c| a[c] + t * (b[c] - a[c]));
    let d: [f64; 4] = std::array::from_fn(|c| s * (b[c] - a[c]));
    // The quotient rule on `C = H / w`.
    let w = h[3];
    let tangent = Vector3D::new(
        d[0] * w - h[0] * d[3],
        d[1] * w - h[1] * d[3],
        d[2] * w - h[2] * d[3],
    );
    tangent.magnitude() / (w * w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Axis3D, Frame3D, PointParametrization};

    // ---- construction / accessors ----

    #[test]
    fn test_new_ok_accessors() {
        let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
        let curve =
            BSplineCurve3D::new(1, poles.clone(), vec![0.0, 1.0], vec![2, 2], false).unwrap();
        assert_eq!(curve.degree(), 1);
        assert!(!curve.is_periodic());
        assert!(!curve.is_rational());
        assert_eq!(curve.poles(), poles.as_slice());
        assert_eq!(curve.weights(), None);
        assert_eq!(curve.knots(), &[0.0, 1.0]);
        assert_eq!(curve.multiplicities(), &[2, 2]);
        assert_eq!(curve.bounds(), (0.0, 1.0));
    }

    #[test]
    fn test_new_rational_ok_accessors() {
        let poles = vec![
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(1.0, 1.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
        ];
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let curve = BSplineCurve3D::new_rational(
            2,
            poles,
            vec![1.0, w, 1.0],
            vec![0.0, 1.0],
            vec![3, 3],
            false,
        )
        .unwrap();
        assert!(curve.is_rational());
        assert_eq!(curve.weights(), Some([1.0, w, 1.0].as_slice()));
    }

    #[test]
    fn test_new_invalid_degree_errors() {
        let poles = vec![Point3D::ORIGIN];
        let err = BSplineCurve3D::new(0, poles, vec![0.0, 1.0], vec![1, 1], false).unwrap_err();
        assert_eq!(err, BSplineConstructionError::InvalidDegree);
    }

    #[test]
    fn test_new_pole_count_mismatch_errors() {
        let poles = vec![Point3D::ORIGIN; 3];
        let err = BSplineCurve3D::new(3, poles, vec![0.0, 1.0, 2.0, 3.0], vec![4, 1, 1, 4], false)
            .unwrap_err();
        assert_eq!(err, BSplineConstructionError::PoleCountMismatch);
    }

    #[test]
    fn test_new_rational_weight_count_mismatch_errors() {
        let poles = vec![Point3D::ORIGIN, Point3D::new(1.0, 0.0, 0.0)];
        let err =
            BSplineCurve3D::new_rational(1, poles, vec![1.0], vec![0.0, 1.0], vec![2, 2], false)
                .unwrap_err();
        assert_eq!(err, BSplineConstructionError::WeightCountMismatch);
    }

    #[test]
    fn test_new_rational_non_positive_weight_errors() {
        let poles = vec![Point3D::ORIGIN, Point3D::new(1.0, 0.0, 0.0)];
        let err = BSplineCurve3D::new_rational(
            1,
            poles,
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
        )
        .unwrap_err();
        assert_eq!(err, BSplineConstructionError::NonPositiveWeight);

        let poles = vec![Point3D::ORIGIN, Point3D::new(1.0, 0.0, 0.0)];
        let err = BSplineCurve3D::new_rational(
            1,
            poles,
            vec![1.0, -2.0],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
        )
        .unwrap_err();
        assert_eq!(err, BSplineConstructionError::NonPositiveWeight);
    }

    #[test]
    fn test_weight_checks_run_before_direction_validation() {
        // Bad weight count AND bad degree: weight check should win, matching
        // the brief's "type-level validation: validate_direction + weights
        // checks" ordering (weights first).
        let poles = vec![Point3D::ORIGIN];
        let err = BSplineCurve3D::new_rational(0, poles, vec![], vec![0.0, 1.0], vec![1, 1], false)
            .unwrap_err();
        assert_eq!(err, BSplineConstructionError::WeightCountMismatch);
    }

    #[test]
    fn test_error_display_all_variants() {
        let variants = [
            BSplineConstructionError::InvalidDegree,
            BSplineConstructionError::TooFewKnots,
            BSplineConstructionError::KnotsNotIncreasing,
            BSplineConstructionError::MultiplicityTooLarge,
            BSplineConstructionError::PeriodicEndMultiplicityMismatch,
            BSplineConstructionError::PoleCountMismatch,
            BSplineConstructionError::WeightCountMismatch,
            BSplineConstructionError::NonPositiveWeight,
        ];
        for v in variants {
            assert!(!v.to_string().is_empty());
        }
    }

    #[test]
    fn test_error_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&BSplineConstructionError::InvalidDegree);
    }

    // ---- eval_point / eval_points / eval_derivative ----

    #[test]
    fn test_eval_point_degree1_line_midpoint() {
        let poles = vec![Point3D::ORIGIN, Point3D::new(2.0, 0.0, 0.0)];
        let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
        assert_eq!(curve.eval_point(0.5), Point3D::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_eval_points_matches_mapped_eval_point() {
        let poles = vec![Point3D::ORIGIN, Point3D::new(2.0, 0.0, 0.0)];
        let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
        let us = [0.0, 0.3, 0.5, 1.0];
        let expected: Vec<Point3D> = us.iter().map(|&u| curve.eval_point(u)).collect();
        assert_eq!(curve.eval_points(&us), expected);
    }

    #[test]
    fn test_eval_derivative_order1_constant_tangent() {
        let poles = vec![Point3D::ORIGIN, Point3D::new(2.0, 0.0, 0.0)];
        let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
        assert_eq!(curve.eval_derivative(0.5, 1), Vector3D::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn test_eval_derivative_order2_of_line_is_zero() {
        let poles = vec![Point3D::ORIGIN, Point3D::new(2.0, 0.0, 0.0)];
        let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
        assert_eq!(curve.eval_derivative(0.5, 2), Vector3D::ZERO);
    }

    #[test]
    #[should_panic(expected = "order must be >= 1")]
    fn test_eval_derivative_order0_panics() {
        let poles = vec![Point3D::ORIGIN, Point3D::new(2.0, 0.0, 0.0)];
        let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
        curve.eval_derivative(0.5, 0);
    }

    #[test]
    #[should_panic(expected = "only first and second derivatives are supported")]
    fn test_eval_derivative_order3_panics_with_clear_message() {
        let poles = vec![Point3D::ORIGIN, Point3D::new(2.0, 0.0, 0.0)];
        let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
        curve.eval_derivative(0.5, 3);
    }

    // ---- periodic seam + wrap ----

    #[test]
    fn test_periodic_curve_is_closed() {
        use crate::curves::ParametricCurve3D;

        let curve = periodic_ring_curve();
        assert!(curve.is_periodic());
        assert!(curve.is_closed(Tolerance::DEFAULT));
        assert!(ParametricCurve3D::is_closed(&curve, Tolerance::DEFAULT));
    }

    #[test]
    fn test_periodic_bounds_are_full_knot_span() {
        let curve = periodic_ring_curve();
        assert_eq!(curve.bounds(), (0.0, 6.0));
    }

    #[test]
    fn test_periodic_seam_point_matches_across_wrap() {
        let curve = periodic_ring_curve();
        let (first, last) = curve.bounds();
        let p_first = curve.eval_point(first);
        let p_last = curve.eval_point(last);
        assert!((p_first.x - p_last.x).abs() < 1e-9);
        assert!((p_first.y - p_last.y).abs() < 1e-9);
        assert!((p_first.z - p_last.z).abs() < 1e-9);
    }

    #[test]
    fn test_periodic_out_of_window_u_matches_golden_samples() {
        let curve = periodic_ring_curve();
        // Golden samples from tests/fixtures/curves_bspline.json
        // (periodic_cubic_ring case) at u = -0.5 and u = 7.3.
        let p = curve.eval_point(-0.5);
        assert!((p.x - 1.4375000000000002).abs() < 1e-7);
        assert!((p.y - 0.829941011960087).abs() < 1e-7);
        assert!((p.z - -6.938893903907228e-17).abs() < 1e-7);

        let p = curve.eval_point(7.3);
        assert!((p.x - -1.2338333333333324).abs() < 1e-7);
        assert!((p.y - 1.1134199941321936).abs() < 1e-7);
        assert!((p.z - 0.056799999999999996).abs() < 1e-7);
    }

    fn periodic_ring_curve() -> BSplineCurve3D {
        let poles = vec![
            Point3D::new(2.0, 0.0, 0.3),
            Point3D::new(1.0000000000000002, 1.7320508075688772, -0.3),
            Point3D::new(-0.9999999999999996, 1.7320508075688774, 0.3),
            Point3D::new(-2.0, 2.4492935982947064e-16, -0.3),
            Point3D::new(-1.0000000000000009, -1.7320508075688767, 0.3),
            Point3D::new(1.0000000000000002, -1.7320508075688772, -0.3),
        ];
        let knots = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mults = vec![1; 7];
        BSplineCurve3D::new(3, poles, knots, mults, true).unwrap()
    }

    fn degree_one_segment() -> BSplineCurve3D {
        let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
        BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap()
    }

    #[test]
    fn test_bspline_project_point_segment() {
        let curve = degree_one_segment();
        let tol = Tolerance::DEFAULT;
        // Interior foot.
        let proj = curve.project_point(Point3D::new(1.0, 1.0, 0.0), tol);
        assert!((proj.parameter - 0.5).abs() < 1e-9);
        assert!((proj.distance - 1.0).abs() < 1e-9);
        // Beyond the end: the endpoint wins.
        let end = curve.project_point(Point3D::new(5.0, 0.0, 0.0), tol);
        assert!((end.parameter - 1.0).abs() < 1e-9);
        assert!((end.distance - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_bspline_contains_segment() {
        let curve = degree_one_segment();
        let tol = Tolerance::DEFAULT;
        assert!(curve.contains(Point3D::new(1.0, 0.0, 0.0), tol));
        assert!(curve.contains(curve.eval_point(0.25), tol));
        assert!(!curve.contains(Point3D::new(1.0, 1.0, 0.0), tol));
        assert!(!curve.contains(Point3D::new(3.0, 0.0, 0.0), tol));
    }

    #[test]
    fn test_bspline_project_point_periodic_ring() {
        let curve = periodic_ring_curve();
        let tol = Tolerance::DEFAULT;
        // On-curve point projects to (near-)zero distance.
        let on = curve.eval_point(2.5);
        assert_eq!(curve.project_point(on, tol).distance, 0.0);
        assert!(curve.contains(on, tol));
        // Off-curve: beat against dense brute-force sampling.
        let query = Point3D::new(0.5, 1.0, 2.0);
        let proj = curve.project_point(query, tol);
        let (first, last) = curve.bounds();
        let mut best = f64::INFINITY;
        for i in 0..=2000 {
            let d = curve
                .eval_point(first + (last - first) * i as f64 / 2000.0)
                .distance(query);
            best = best.min(d);
        }
        assert!(proj.distance <= best);
        assert!(best - proj.distance < 1e-3);
    }

    #[test]
    #[allow(clippy::approx_constant)] // random pole data, not an approximation of PI
    fn test_bspline_project_point_close_basins_in_one_span() {
        // On the span [6.12, 7.5] the distance to points of the curve near
        // t = 6.55 has a second, shallow basin 0.15 away; a sampling of the
        // span lands in the wrong one.
        let poles: Vec<Point3D> = [
            (-2.25, 3.33, -1.92),
            (1.71, 1.64, 0.97),
            (1.34, 1.35, 2.74),
            (4.44, 4.84, -0.38),
            (-1.01, -3.14, 0.75),
            (3.09, 2.88, -2.34),
            (-2.89, -2.3, 1.67),
            (-2.52, -0.25, -2.4),
            (-0.46, -4.8, 2.63),
            (-1.1, -3.33, 3.21),
            (1.66, 0.37, -3.9),
            (1.89, -4.08, 4.82),
            (2.57, 0.13, 4.41),
            (1.98, 0.86, -3.55),
            (1.38, -3.48, -2.22),
            (1.78, -0.64, -4.31),
            (3.03, 2.35, 4.74),
            (-2.61, -1.25, -2.56),
            (-1.6, 4.56, -3.09),
            (4.73, -3.44, -0.01),
            (-4.69, 3.97, -3.48),
            (-3.26, 3.84, 0.16),
        ]
        .iter()
        .map(|&(x, y, z)| Point3D::new(x, y, z))
        .collect();
        let weights = vec![
            2.85, 3.49, 0.48, 4.48, 2.42, 0.46, 4.03, 4.34, 4.38, 2.57, 4.8, 0.85, 1.9, 1.18, 4.07,
            2.17, 1.43, 1.28, 3.19, 4.94, 1.84, 1.29,
        ];
        let knots = vec![2.07, 2.56, 3.15, 4.75, 5.48, 6.12, 7.5, 8.3, 9.67];
        let curve = BSplineCurve3D::new_rational(
            5,
            poles,
            weights,
            knots,
            vec![6, 1, 1, 1, 4, 3, 2, 4, 6],
            false,
        )
        .unwrap();
        for i in 0..=200 {
            let t = 6.12 + 1.38 * i as f64 / 200.0;
            let q = curve.eval_point(t);
            let proj = curve.project_point(q, Tolerance::DEFAULT);
            assert_eq!(proj.distance, 0.0, "t = {t}: {proj:?}");
            assert_eq!(curve.extrema(q, Tolerance::DEFAULT)[0].distance, 0.0);
        }
    }

    #[test]
    fn test_bspline_project_point_exact_foot() {
        // Cubic Bezier with x = 3u, y = 3u(1 - u): the foot of
        // (1.526, -0.65) is exactly u = 0.6 (C'(0.6) = (3, -0.6) is normal
        // to the residual (0.274, 1.37)). The old first-order refinement
        // stopped at the seed sample u = 0.59375.
        let poles = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 1.0, 0.0),
            Point3D::new(2.0, 1.0, 0.0),
            Point3D::new(3.0, 0.0, 0.0),
        ];
        let curve = BSplineCurve3D::new(3, poles, vec![0.0, 1.0], vec![4, 4], false).unwrap();
        let proj = curve.project_point(Point3D::new(1.526, -0.65, 0.0), Tolerance::DEFAULT);
        assert!((proj.parameter - 0.6).abs() < 1e-12);
        assert!((proj.distance - 1.951976f64.sqrt()).abs() < 1e-12);
        // Quadratic on an offset knot range: the minimum is a stationary
        // point (u = 1.756876, d = 5.349035), not a raw sample.
        let poles = vec![
            Point3D::new(2.0, 0.0, 0.0),
            Point3D::new(0.66, 3.437, 1.22),
            Point3D::new(-4.644, 1.852, 2.439),
        ];
        let curve = BSplineCurve3D::new(2, poles, vec![1.0242, 2.3891], vec![3, 3], false).unwrap();
        let query = Point3D::new(1.436, 5.4066, 5.122);
        let proj = curve.project_point(query, Tolerance::DEFAULT);
        assert!((proj.parameter - 1.756876).abs() < 1e-6);
        assert!((proj.distance - 5.349035).abs() < 1e-6);
        let r = curve.eval_point(proj.parameter) - query;
        let d1 = curve.eval_derivative(proj.parameter, 1);
        assert!(
            r.dot(d1).abs() < 1e-12 * r.magnitude() * d1.magnitude(),
            "{}",
            r.dot(d1)
        );
    }

    #[test]
    fn test_interpolate_line_points() {
        let tol = Tolerance::DEFAULT;
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(3.0, 0.0, 0.0),
        ];
        for param in [
            PointParametrization::Chordal,
            PointParametrization::Centripetal,
        ] {
            let curve = BSplineCurve3D::interpolate(&points, 2, param).unwrap();
            for p in &points {
                assert!(curve.contains(*p, tol), "{param:?}");
            }
            // Collinear data interpolates the straight segment.
            assert!((curve.eval_point(0.5).y).abs() < 1e-9);
            assert!((curve.eval_point(0.5).z).abs() < 1e-9);
        }
    }

    #[test]
    fn test_interpolate_circle_points() {
        let tol = Tolerance::DEFAULT;
        let points: Vec<Point3D> = (0..9)
            .map(|i| {
                let u = i as f64 / 8.0 * std::f64::consts::TAU;
                Point3D::new(2.0 * u.cos(), 2.0 * u.sin(), 0.0)
            })
            .collect();
        let curve =
            BSplineCurve3D::interpolate(&points, 3, PointParametrization::Centripetal).unwrap();
        for p in &points {
            assert!(curve.contains(*p, tol));
        }
        // Near-circular: radius stays close to 2.
        for i in 0..=40 {
            let p = curve.eval_point(i as f64 / 40.0);
            let r = (p.x * p.x + p.y * p.y).sqrt();
            assert!((r - 2.0).abs() < 0.05, "{p:?}");
        }
    }

    #[test]
    fn test_interpolate_errors() {
        let one = vec![Point3D::ORIGIN];
        assert_eq!(
            BSplineCurve3D::interpolate(&one, 1, PointParametrization::Chordal),
            Err(BSplineConstructionError::TooFewPoints)
        );
        let three = vec![
            Point3D::ORIGIN,
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(1.0, 1.0, 0.0),
        ];
        assert_eq!(
            BSplineCurve3D::interpolate(&three, 0, PointParametrization::Chordal),
            Err(BSplineConstructionError::InvalidDegree)
        );
        assert_eq!(
            BSplineCurve3D::interpolate(&three, 3, PointParametrization::Chordal),
            Err(BSplineConstructionError::DegreeTooHigh)
        );
        let dup = vec![
            Point3D::ORIGIN,
            Point3D::ORIGIN,
            Point3D::new(1.0, 0.0, 0.0),
        ];
        assert_eq!(
            BSplineCurve3D::interpolate(&dup, 1, PointParametrization::Chordal),
            Err(BSplineConstructionError::ConfusedPoints)
        );
    }

    #[test]
    fn test_approximate_circle_fewer_poles() {
        let points: Vec<Point3D> = (0..12)
            .map(|i| {
                let u = i as f64 / 12.0 * std::f64::consts::TAU;
                Point3D::new(2.0 * u.cos(), 2.0 * u.sin(), 0.0)
            })
            .collect();
        let curve =
            BSplineCurve3D::approximate(&points, 3, 7, PointParametrization::Centripetal).unwrap();
        assert_eq!(curve.degree(), 3);
        // Near-circular fit with fewer poles than points.
        for i in 0..=60 {
            let p = curve.eval_point(i as f64 / 60.0);
            let r = (p.x * p.x + p.y * p.y).sqrt();
            assert!((r - 2.0).abs() < 0.05, "{p:?}");
        }
    }

    #[test]
    fn test_approximate_line_data_exact() {
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(2.0, 0.0, 0.0),
            Point3D::new(3.0, 0.0, 0.0),
        ];
        let curve =
            BSplineCurve3D::approximate(&points, 1, 2, PointParametrization::Chordal).unwrap();
        for i in 0..=20 {
            let p = curve.eval_point(i as f64 / 20.0);
            assert!(p.y.abs() < 1e-9 && p.z.abs() < 1e-9, "{p:?}");
            assert!((0.0..=3.0).contains(&p.x), "{p:?}");
        }
    }

    #[test]
    fn test_approximate_errors() {
        let one = vec![Point3D::ORIGIN];
        assert_eq!(
            BSplineCurve3D::approximate(&one, 1, 2, PointParametrization::Chordal),
            Err(BSplineConstructionError::TooFewPoints)
        );
        let four = vec![
            Point3D::ORIGIN,
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(1.0, 1.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
        ];
        assert_eq!(
            BSplineCurve3D::approximate(&four, 1, 5, PointParametrization::Chordal),
            Err(BSplineConstructionError::TooManyPoles)
        );
        assert_eq!(
            BSplineCurve3D::approximate(&four, 4, 3, PointParametrization::Chordal),
            Err(BSplineConstructionError::DegreeTooHigh)
        );
    }

    // ---- transformed ----

    fn transformed_test_curve() -> BSplineCurve3D {
        let poles = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 2.0, 0.0),
            Point3D::new(3.0, 2.0, 1.0),
            Point3D::new(4.0, 0.0, 1.0),
        ];
        BSplineCurve3D::new(3, poles, vec![0.0, 1.0], vec![4, 4], false).unwrap()
    }

    #[test]
    fn test_bspline_curve3d_transformed_translation_spot_check() {
        let curve = transformed_test_curve();
        let t = Transform::translation(Vector3D::new(1.0, 1.0, 1.0));
        let moved = curve.transformed(t);
        // Poles map point-wise; the knot structure is unchanged.
        assert_eq!(moved.poles().len(), curve.poles().len());
        for (a, b) in moved.poles().iter().zip(curve.poles()) {
            assert_eq!(*a, t.apply_point(*b));
        }
        assert_eq!(moved.knots(), curve.knots());
        assert_eq!(moved.bounds(), curve.bounds());
        // Hand-computed: translation shifts every evaluation.
        let p = moved.eval_point(0.5);
        let q = curve.eval_point(0.5);
        assert_eq!(p, Point3D::new(q.x + 1.0, q.y + 1.0, q.z + 1.0));
    }

    #[test]
    fn test_bspline_curve3d_transformed_round_trip_and_mirror() {
        use std::f64::consts::PI;
        let curve = transformed_test_curve();
        let axis = Axis3D::new(Point3D::new(0.0, 1.0, 0.0), Vector3D::Z).unwrap();
        let t = Transform::rotation(axis, PI / 3.0).then(Transform::scaling(Point3D::ORIGIN, 2.0));
        let moved = curve.transformed(t);
        let tol = Tolerance::DEFAULT;
        for i in 0..=10 {
            let u = i as f64 / 10.0;
            assert!(moved.contains(t.apply_point(curve.eval_point(u)), tol));
        }
        let inv =
            Transform::scaling(Point3D::ORIGIN, 0.5).then(Transform::rotation(axis, -PI / 3.0));
        let back = moved.transformed(inv);
        for i in 0..=10 {
            let u = i as f64 / 10.0;
            assert!(back.eval_point(u).distance(curve.eval_point(u)) < 1e-9);
        }
        // Mirror preserves the point set (as a fresh pole net).
        let m = Transform::mirror_plane(Frame3D::WORLD);
        let mirrored = curve.transformed(m);
        for i in 0..=10 {
            let u = i as f64 / 10.0;
            assert!(mirrored.contains(m.apply_point(curve.eval_point(u)), tol));
        }
        assert_eq!(mirrored.transformed(m), curve);
    }

    // ---- knot insertion ----

    /// Nonuniform clamped cubic with an interior double knot, rational if
    /// `rational`.
    fn clamped_curve(rational: bool) -> BSplineCurve3D {
        let poles: Vec<Point3D> = (0..8)
            .map(|i| Point3D::new(i as f64, (i * i % 5) as f64, (i % 3) as f64 - 1.0))
            .collect();
        let knots = vec![0.0, 0.7, 1.0, 2.5, 4.0];
        let mults = vec![4, 1, 2, 1, 4];
        if rational {
            let weights = vec![1.0, 0.4, 2.5, 1.2, 0.8, 3.0, 0.6, 1.5];
            BSplineCurve3D::new_rational(3, poles, weights, knots, mults, false).unwrap()
        } else {
            BSplineCurve3D::new(3, poles, knots, mults, false).unwrap()
        }
    }

    /// Unclamped non-periodic quadratic: bounds `(2, 7)`, both bound knots
    /// interior to the knot vector.
    fn unclamped_curve() -> BSplineCurve3D {
        let poles: Vec<Point3D> = (0..7)
            .map(|i| Point3D::new(i as f64, ((i * 3) % 4) as f64, 0.5 * i as f64))
            .collect();
        let knots: Vec<f64> = (0..10).map(f64::from).collect();
        BSplineCurve3D::new(2, poles, knots, vec![1; 10], false).unwrap()
    }

    /// Periodic cubic with a double seam knot, rational if `rational`.
    fn periodic_seam_curve(rational: bool) -> BSplineCurve3D {
        let poles = periodic_ring_curve().poles().to_vec();
        let knots = vec![0.0, 1.5, 3.0, 4.0, 5.0, 6.0];
        let mults = vec![2, 1, 1, 1, 1, 2];
        if rational {
            let weights = vec![1.0, 2.0, 0.5, 1.5, 0.8, 1.2];
            BSplineCurve3D::new_rational(3, poles, weights, knots, mults, true).unwrap()
        } else {
            BSplineCurve3D::new(3, poles, knots, mults, true).unwrap()
        }
    }

    /// Periodic quadratic with three poles: a period shorter than the
    /// supports of the new poles.
    fn short_periodic_curve() -> BSplineCurve3D {
        let poles = vec![
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(-0.5, 0.9, 0.2),
            Point3D::new(-0.5, -0.9, -0.2),
        ];
        BSplineCurve3D::new(2, poles, vec![0.0, 0.5, 2.0, 3.0], vec![1; 4], true).unwrap()
    }

    /// `refined` evaluates as `curve`, point and first two derivatives,
    /// within 1e-12 relative to the curve's size, at a spread of
    /// parameters, at every knot of both, and, on a periodic curve, past
    /// the bounds.
    fn assert_same_curve(curve: &BSplineCurve3D, refined: &BSplineCurve3D) {
        assert_eq!(refined.bounds(), curve.bounds());
        assert_eq!(refined.is_periodic(), curve.is_periodic());
        assert_eq!(refined.is_rational(), curve.is_rational());
        let rebuilt = match refined.weights() {
            Some(w) => BSplineCurve3D::new_rational(
                refined.degree(),
                refined.poles().to_vec(),
                w.to_vec(),
                refined.knots().to_vec(),
                refined.multiplicities().to_vec(),
                refined.is_periodic(),
            ),
            None => BSplineCurve3D::new(
                refined.degree(),
                refined.poles().to_vec(),
                refined.knots().to_vec(),
                refined.multiplicities().to_vec(),
                refined.is_periodic(),
            ),
        };
        assert!(rebuilt.is_ok(), "invalid refinement: {rebuilt:?}");
        let size = curve
            .poles()
            .iter()
            .map(|p| p.x.abs().max(p.y.abs()).max(p.z.abs()))
            .fold(1.0, f64::max);
        let (first, last) = curve.bounds();
        let mut params: Vec<f64> = (0..=40)
            .map(|i| first + (last - first) * i as f64 / 40.0)
            .collect();
        params.extend(curve.knots().iter().chain(refined.knots()));
        if curve.is_periodic() {
            let period = last - first;
            params.extend([first - 0.3 * period, last + 0.45 * period]);
        }
        for u in params {
            let (p, d1, d2) = curve.eval_d2(u);
            let (q, e1, e2) = refined.eval_d2(u);
            let pairs = [
                (p - Point3D::ORIGIN, q - Point3D::ORIGIN),
                (d1, e1),
                (d2, e2),
            ];
            for (k, (a, b)) in pairs.into_iter().enumerate() {
                let tol = 1e-12 * size.max(a.magnitude());
                assert!(
                    (a - b).magnitude() <= tol,
                    "u = {u}, order {k}: {a:?} vs {b:?}"
                );
            }
        }
    }

    fn all_insertion_curves() -> Vec<BSplineCurve3D> {
        vec![
            clamped_curve(false),
            clamped_curve(true),
            unclamped_curve(),
            periodic_ring_curve(),
            periodic_seam_curve(false),
            periodic_seam_curve(true),
            short_periodic_curve(),
        ]
    }

    #[test]
    fn test_insert_knot_keeps_shape_everywhere() {
        for curve in all_insertion_curves() {
            let (first, last) = curve.bounds();
            let p = curve.degree() as u32;
            let mut values: Vec<(f64, u32)> = vec![
                (first + 0.37 * (last - first), 1),
                (first + 0.61 * (last - first), p),
            ];
            // Raise interior knots to the degree.
            for (&k, &m) in curve.knots().iter().zip(curve.multiplicities()) {
                if k > first && k < last && m < p {
                    values.push((k, p - m));
                }
            }
            for (u, times) in values {
                let refined = curve.insert_knot(u, times).unwrap();
                assert_eq!(refined.poles().len(), curve.poles().len() + times as usize);
                let index = refined.knots().iter().position(|&k| k == u).unwrap();
                let before = curve
                    .knots()
                    .iter()
                    .position(|&k| k == u)
                    .map_or(0, |i| curve.multiplicities()[i]);
                assert_eq!(refined.multiplicities()[index], before + times);
                assert_same_curve(&curve, &refined);
            }
        }
    }

    #[test]
    fn test_insert_knot_keeps_untouched_rational_poles_exactly() {
        // Inserting 1.8 blends poles 3 to 5 only.
        let curve = clamped_curve(true);
        let refined = curve.insert_knot(1.8, 1).unwrap();
        let (poles, weights) = (curve.poles(), curve.weights().unwrap());
        let (new_poles, new_weights) = (refined.poles(), refined.weights().unwrap());
        assert_eq!(new_poles[..4], poles[..4]);
        assert_eq!(new_weights[..4], weights[..4]);
        assert_eq!(new_poles[7..], poles[6..]);
        assert_eq!(new_weights[7..], weights[6..]);
    }

    /// The knots within the bounds, ends included.
    fn breaks_within_bounds(curve: &BSplineCurve3D) -> Vec<f64> {
        let (first, last) = curve.bounds();
        let mut breaks = vec![first];
        breaks.extend(curve.knots().iter().filter(|&&k| k > first && k < last));
        breaks.push(last);
        breaks
    }

    #[test]
    fn test_bezier_pieces_cover_every_span_and_evaluate_as_the_curve() {
        for curve in all_insertion_curves() {
            let p = curve.degree();
            let pieces = curve.to_bezier_pieces();
            let breaks = breaks_within_bounds(&curve);
            assert_eq!(pieces.len(), breaks.len() - 1);
            let size = curve
                .poles()
                .iter()
                .map(|q| q.x.abs().max(q.y.abs()).max(q.z.abs()))
                .fold(1.0, f64::max);
            for (k, piece) in pieces.iter().enumerate() {
                assert_eq!(piece.bounds(), (breaks[k], breaks[k + 1]));
                assert_eq!(piece.multiplicities(), &[p as u32 + 1; 2]);
                assert_eq!(piece.poles().len(), p + 1);
                assert_eq!(piece.is_rational(), curve.is_rational());
                assert!(!piece.is_periodic());
                let (a, b) = piece.bounds();
                for i in 0..=8 {
                    let u = a + (b - a) * i as f64 / 8.0;
                    let tol = 1e-12 * size;
                    assert!(piece.eval_point(u).distance(curve.eval_point(u)) <= tol);
                    if (1..8).contains(&i) {
                        let (_, d1, d2) = curve.eval_d2(u);
                        let (_, e1, e2) = piece.eval_d2(u);
                        assert!((d1 - e1).magnitude() <= tol.max(1e-12 * d1.magnitude()));
                        assert!((d2 - e2).magnitude() <= tol.max(1e-12 * d2.magnitude()));
                    }
                }
            }
            // Neighbors share the pole at their common knot.
            for pair in pieces.windows(2) {
                assert_eq!(pair[0].poles()[p], pair[1].poles()[0]);
                if let (Some(w0), Some(w1)) = (pair[0].weights(), pair[1].weights()) {
                    assert_eq!(w0[p], w1[0]);
                }
            }
        }
    }

    #[test]
    fn test_insert_knot_at_bounds() {
        // Periodic: either bound raises the seam at both ends.
        for curve in [periodic_ring_curve(), periodic_seam_curve(true)] {
            let (first, last) = curve.bounds();
            let seam = curve.multiplicities()[0];
            for u in [first, last] {
                let refined = curve.insert_knot(u, 1).unwrap();
                let mults = refined.multiplicities();
                assert_eq!((mults[0], mults[mults.len() - 1]), (seam + 1, seam + 1));
                assert_eq!(refined.poles().len(), curve.poles().len() + 1);
                assert_same_curve(&curve, &refined);
            }
            let full = curve.insert_knot(first, 3 - seam).unwrap();
            assert_same_curve(&curve, &full);
            assert_eq!(
                curve.insert_knot(last, 4 - seam),
                Err(KnotInsertionError::MultiplicityTooLarge)
            );
        }
        // Unclamped: the bounds are interior knots of the knot vector.
        let curve = unclamped_curve();
        for u in [2.0, 7.0] {
            let refined = curve.insert_knot(u, 1).unwrap();
            assert_same_curve(&curve, &refined);
            assert_eq!(
                curve.insert_knot(u, 2),
                Err(KnotInsertionError::MultiplicityTooLarge)
            );
        }
        // Clamped: the end knots are already at `degree + 1`.
        let curve = clamped_curve(false);
        for u in [0.0, 4.0] {
            assert_eq!(
                curve.insert_knot(u, 1),
                Err(KnotInsertionError::MultiplicityTooLarge)
            );
        }
    }

    #[test]
    fn test_refine_knots_keeps_shape_and_matches_single_insertions() {
        for curve in all_insertion_curves() {
            let (first, last) = curve.bounds();
            let at = |t: f64| first + t * (last - first);
            // New knots (one repeated), an existing interior knot, and on a
            // periodic curve the seam, at both bounds when the degree
            // allows.
            let mut knots = vec![at(0.05), at(0.3), at(0.3), at(0.52), at(0.9)];
            if let Some(&k) = curve.knots().iter().find(|&&k| k > first && k < last) {
                knots.push(k);
            }
            if curve.is_periodic() {
                knots.push(first);
                if curve.multiplicities()[0] + 2 <= curve.degree() as u32 {
                    knots.push(last);
                }
            }
            knots.sort_by(f64::total_cmp);
            let refined = curve.refine_knots(&knots).unwrap();
            assert_eq!(refined.poles().len(), curve.poles().len() + knots.len());
            assert_same_curve(&curve, &refined);
            let mut one_by_one = curve.clone();
            for &u in &knots {
                one_by_one = one_by_one.insert_knot(u, 1).unwrap();
            }
            assert_eq!(one_by_one.knots(), refined.knots());
            assert_eq!(one_by_one.multiplicities(), refined.multiplicities());
            for (a, b) in one_by_one.poles().iter().zip(refined.poles()) {
                assert!(a.distance(*b) < 1e-12, "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn test_refine_knots_many_per_span() {
        for curve in all_insertion_curves() {
            let (first, last) = curve.bounds();
            let knots: Vec<f64> = (1..64)
                .map(|i| first + (last - first) * i as f64 / 64.0)
                .collect();
            let refined = curve.refine_knots(&knots).unwrap();
            assert_same_curve(&curve, &refined);
            // Refining again keeps refining the same shape.
            let twice = refined
                .refine_knots(&[first + 0.01 * (last - first)])
                .unwrap();
            assert_same_curve(&curve, &twice);
        }
    }

    #[test]
    fn test_length_of_a_segment_and_a_circle() {
        let poles = vec![Point3D::new(1.0, 2.0, 3.0), Point3D::new(4.0, 6.0, 3.0)];
        let segment = BSplineCurve3D::new(1, poles, vec![0.0, 2.0], vec![2, 2], false).unwrap();
        assert!((segment.length(1e-12) - 5.0).abs() <= 1e-14);
        let circle = crate::Circle3D::new(Point3D::new(1.0, -2.0, 0.5), Vector3D::Z, 3.0)
            .unwrap()
            .to_bspline();
        let exact = 2.0 * std::f64::consts::PI * 3.0;
        for max_error in [1e-6, 1e-9, 1e-12] {
            let length = circle.length(max_error);
            assert!((length - exact).abs() <= max_error, "{length} vs {exact}");
        }
    }

    #[test]
    fn test_length_adds_up_over_pieces() {
        for curve in all_insertion_curves() {
            let (first, last) = curve.bounds();
            let u = first + 0.37 * (last - first);
            let (low, high) = curve.split_at(u).unwrap();
            let whole = curve.length(1e-12);
            let pieces = low.length(1e-12) + high.length(1e-12);
            assert!((whole - pieces).abs() <= 1e-11, "{whole} vs {pieces}");
            // A fine polyline along the curve is a little shorter.
            let points = curve.eval_points(
                &(0..=4000)
                    .map(|i| first + (last - first) * i as f64 / 4000.0)
                    .collect::<Vec<_>>(),
            );
            let chords: f64 = points.windows(2).map(|w| w[0].distance(w[1])).sum();
            assert!(
                chords <= whole && whole - chords <= 1e-5 * whole,
                "{chords} vs {whole}"
            );
        }
    }

    #[test]
    fn test_refine_knots_beyond_the_stack() {
        // Enough knots that the rational poles refine on the heap; the
        // spans so short that only points and tangents are compared, the
        // second derivatives losing digits to their scale.
        let curve = clamped_curve(true);
        let (first, last) = curve.bounds();
        let knots: Vec<f64> = (1..200)
            .map(|i| first + (last - first) * i as f64 / 200.0)
            .collect();
        let refined = curve.refine_knots(&knots).unwrap();
        assert_eq!(refined.poles().len(), curve.poles().len() + knots.len());
        for i in 0..=100 {
            let u = first + (last - first) * i as f64 / 100.0;
            let (p, d1) = curve.eval_d1(u);
            let (q, e1) = refined.eval_d1(u);
            assert!(p.distance(q) <= 1e-12 * 8.0, "u = {u}: {p:?} vs {q:?}");
            assert!(
                (d1 - e1).magnitude() <= 1e-11 * d1.magnitude().max(8.0),
                "u = {u}: {d1:?} vs {e1:?}"
            );
        }
    }

    #[test]
    fn test_refine_knots_random_curves() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut checked = 0;
        for case in 0..400 {
            let degree = 1 + case % 5;
            let periodic = case % 2 == 0;
            let rational = case % 3 == 0;
            let n_knots = 2 + (next() * 6.0) as usize;
            let mut knots = vec![next() * 4.0 - 2.0];
            for _ in 1..n_knots {
                knots.push(knots[knots.len() - 1] + 0.3 + 0.7 * next());
            }
            let mut mults: Vec<u32> = (0..n_knots)
                .map(|_| 1 + (next() * degree as f64) as u32)
                .collect();
            let last = n_knots - 1;
            if periodic {
                mults[last] = mults[0];
            } else {
                mults[0] = degree as u32 + 1;
                mults[last] = degree as u32 + 1;
            }
            let sum: u32 = mults.iter().sum();
            let n_poles = if periodic {
                (sum - mults[last]) as usize
            } else {
                sum as usize - degree - 1
            };
            if n_poles < degree + 1 {
                continue;
            }
            let poles: Vec<Point3D> = (0..n_poles)
                .map(|_| Point3D::new(4.0 * next() - 2.0, 4.0 * next() - 2.0, 4.0 * next() - 2.0))
                .collect();
            let curve = if rational {
                let weights = (0..n_poles).map(|_| 0.3 + 2.0 * next()).collect();
                BSplineCurve3D::new_rational(degree, poles, weights, knots, mults, periodic)
            } else {
                BSplineCurve3D::new(degree, poles, knots, mults, periodic)
            }
            .unwrap();
            // Existing knots and quarter points of spans, kept within the
            // multiplicity limits.
            let (first, last) = curve.bounds();
            let mut candidates = vec![];
            for w in curve.knots().windows(2) {
                for t in [0.0, 0.25, 0.5, 0.75] {
                    let u = w[0] + t * (w[1] - w[0]);
                    if u >= first && u <= last {
                        candidates.push(u);
                    }
                }
            }
            candidates.push(last);
            let mut insert = vec![];
            for &u in &candidates {
                for _ in 0..(next() * 3.0) as usize {
                    insert.push(u);
                }
            }
            let mut refined = curve.refine_knots(&insert);
            while refined == Err(KnotInsertionError::MultiplicityTooLarge) {
                insert.remove((next() * insert.len() as f64) as usize);
                refined = curve.refine_knots(&insert);
            }
            let refined = refined.unwrap();
            assert_eq!(refined.poles().len(), curve.poles().len() + insert.len());
            assert_same_curve(&curve, &refined);
            checked += 1;
        }
        assert!(checked > 250, "{checked}");
    }

    #[test]
    fn test_insert_zero_times_or_no_knots_is_unchanged() {
        let curve = clamped_curve(true);
        assert_eq!(curve.insert_knot(1.3, 0).unwrap(), curve);
        assert_eq!(curve.refine_knots(&[]).unwrap(), curve);
    }

    #[test]
    fn test_knot_insertion_errors() {
        for curve in [clamped_curve(false), periodic_ring_curve()] {
            let (first, last) = curve.bounds();
            for u in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert_eq!(
                    curve.insert_knot(u, 1),
                    Err(KnotInsertionError::NonFiniteKnot)
                );
                assert_eq!(
                    curve.refine_knots(&[first, u]),
                    Err(KnotInsertionError::NonFiniteKnot)
                );
            }
            for u in [first - 1e-9, last + 1e-9] {
                assert_eq!(
                    curve.insert_knot(u, 1),
                    Err(KnotInsertionError::KnotOutOfRange)
                );
            }
            let mid = 0.5 * (first + last);
            assert_eq!(
                curve.refine_knots(&[mid, mid - 0.1]),
                Err(KnotInsertionError::KnotsNotSorted)
            );
            assert_eq!(
                curve.insert_knot(mid, 4),
                Err(KnotInsertionError::MultiplicityTooLarge)
            );
            assert_eq!(
                curve.insert_knot(mid, u32::MAX),
                Err(KnotInsertionError::MultiplicityTooLarge)
            );
            assert_eq!(
                curve.refine_knots(&[mid, mid, mid, mid]),
                Err(KnotInsertionError::MultiplicityTooLarge)
            );
        }
        // The double knot of the clamped curve takes one more.
        let curve = clamped_curve(false);
        assert!(curve.insert_knot(1.0, 1).is_ok());
        assert_eq!(
            curve.insert_knot(1.0, 2),
            Err(KnotInsertionError::MultiplicityTooLarge)
        );
    }

    #[test]
    fn test_knot_insertion_error_display() {
        for e in [
            KnotInsertionError::NonFiniteKnot,
            KnotInsertionError::KnotOutOfRange,
            KnotInsertionError::KnotsNotSorted,
            KnotInsertionError::MultiplicityTooLarge,
        ] {
            assert!(!e.to_string().is_empty());
        }
    }

    // ---- knot removal ----

    /// Largest coordinate of the poles, at least one.
    fn pole_size(curve: &BSplineCurve3D) -> f64 {
        curve
            .poles()
            .iter()
            .map(|p| p.x.abs().max(p.y.abs()).max(p.z.abs()))
            .fold(1.0, f64::max)
    }

    /// `simpler` has the knots, multiplicities and periodicity of
    /// `curve`, and its poles and weights within 1e-12 relative.
    fn assert_same_poles(curve: &BSplineCurve3D, simpler: &BSplineCurve3D) {
        assert_eq!(simpler.knots(), curve.knots());
        assert_eq!(simpler.multiplicities(), curve.multiplicities());
        assert_eq!(simpler.is_periodic(), curve.is_periodic());
        let size = pole_size(curve);
        for (p, q) in simpler.poles().iter().zip(curve.poles()) {
            assert!(p.distance(*q) <= 1e-12 * size, "{p:?} vs {q:?}");
        }
        match (simpler.weights(), curve.weights()) {
            (Some(a), Some(b)) => {
                for (x, y) in a.iter().zip(b) {
                    assert!((x - y).abs() <= 1e-12 * y, "{x} vs {y}");
                }
            }
            (a, b) => assert_eq!(a.is_some(), b.is_some()),
        }
    }

    /// Largest distance between the points of `a` and `b` at the same
    /// parameter, over 2001 parameters and every knot of both; asserts
    /// `b` is a valid curve with the bounds of `a`.
    fn sampled_deviation(a: &BSplineCurve3D, b: &BSplineCurve3D) -> f64 {
        let rebuilt = match b.weights() {
            Some(w) => BSplineCurve3D::new_rational(
                b.degree(),
                b.poles().to_vec(),
                w.to_vec(),
                b.knots().to_vec(),
                b.multiplicities().to_vec(),
                b.is_periodic(),
            ),
            None => BSplineCurve3D::new(
                b.degree(),
                b.poles().to_vec(),
                b.knots().to_vec(),
                b.multiplicities().to_vec(),
                b.is_periodic(),
            ),
        };
        assert!(rebuilt.is_ok(), "invalid removal: {rebuilt:?}");
        assert_eq!(a.bounds(), b.bounds());
        let (first, last) = a.bounds();
        (0..=2000)
            .map(|i| first + (last - first) * i as f64 / 2000.0)
            .chain(a.knots().iter().chain(b.knots()).copied())
            .filter(|u| (first..=last).contains(u))
            .map(|u| a.eval_point(u).distance(b.eval_point(u)))
            .fold(0.0, f64::max)
    }

    /// The knots `remove_knot` accepts: those strictly inside the bounds
    /// of a non-periodic curve, every knot of a periodic one but its last.
    fn removable_knots(curve: &BSplineCurve3D) -> Vec<(f64, u32)> {
        let (first, last) = curve.bounds();
        curve
            .knots()
            .iter()
            .zip(curve.multiplicities())
            .filter(|&(&k, _)| {
                if curve.is_periodic() {
                    k < last
                } else {
                    first < k && k < last
                }
            })
            .map(|(&k, &m)| (k, m))
            .collect()
    }

    /// Uniform numbers in `[0, 1)` from an xorshift generator.
    fn uniform(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// A random curve of degree `1 + case % 5`, periodic for even `case`,
    /// rational for `case` a multiple of 3, with random knots and
    /// multiplicities; `None` when they leave too few poles.
    fn random_curve(case: usize, next: &mut impl FnMut() -> f64) -> Option<BSplineCurve3D> {
        let degree = 1 + case % 5;
        let periodic = case.is_multiple_of(2);
        let rational = case.is_multiple_of(3);
        let n_knots = 2 + (next() * 6.0) as usize;
        let mut knots = vec![next() * 4.0 - 2.0];
        for _ in 1..n_knots {
            knots.push(knots[knots.len() - 1] + 0.3 + 0.7 * next());
        }
        let mut mults: Vec<u32> = (0..n_knots)
            .map(|_| 1 + (next() * degree as f64) as u32)
            .collect();
        let last = n_knots - 1;
        if periodic {
            mults[last] = mults[0];
        } else {
            mults[0] = degree as u32 + 1;
            mults[last] = degree as u32 + 1;
        }
        let sum: u32 = mults.iter().sum();
        let n_poles = if periodic {
            (sum - mults[last]) as usize
        } else {
            sum as usize - degree - 1
        };
        if n_poles < degree + 1 {
            return None;
        }
        let poles: Vec<Point3D> = (0..n_poles)
            .map(|_| Point3D::new(4.0 * next() - 2.0, 4.0 * next() - 2.0, 4.0 * next() - 2.0))
            .collect();
        let curve = if rational {
            let weights = (0..n_poles).map(|_| 0.3 + 2.0 * next()).collect();
            BSplineCurve3D::new_rational(degree, poles, weights, knots, mults, periodic)
        } else {
            BSplineCurve3D::new(degree, poles, knots, mults, periodic)
        };
        Some(curve.unwrap())
    }

    /// Random curves of degrees 1 to 5, clamped or periodic, polynomial or
    /// rational, with random multiplicities.
    fn random_curves(count: usize, seed: u64) -> Vec<BSplineCurve3D> {
        let mut next = uniform(seed);
        (0..count)
            .filter_map(|case| random_curve(case, &mut next))
            .collect()
    }

    #[test]
    fn test_remove_inserted_knot_gives_back_the_curve() {
        for curve in all_insertion_curves() {
            let (first, last) = curve.bounds();
            let tol = 1e-9 * pole_size(&curve);
            let mut cases = vec![
                (first + 0.37 * (last - first), 1),
                (first + 0.81 * (last - first), curve.degree() as u32),
            ];
            for (k, m) in removable_knots(&curve) {
                let room = curve.degree() as u32 - m;
                if room > 0 && (k != first || curve.is_periodic()) {
                    cases.push((k, room));
                }
            }
            for (u, times) in cases {
                let refined = curve.insert_knot(u, times).unwrap();
                let (simpler, removed) = refined.remove_knot(u, times, tol).unwrap();
                assert_eq!(removed, times, "u = {u}");
                assert_same_poles(&curve, &simpler);
                // Either bound names the seam.
                if curve.is_periodic() && u == first {
                    let (simpler, removed) = refined.remove_knot(last, times, tol).unwrap();
                    assert_eq!(removed, times);
                    assert_same_poles(&curve, &simpler);
                }
            }
        }
    }

    #[test]
    fn test_remove_knot_polynomial_round_trip_is_nearly_exact() {
        let curve = clamped_curve(false);
        let refined = curve.insert_knot(1.6, 2).unwrap();
        let (simpler, removed) = refined.remove_knot(1.6, 2, 1e-13).unwrap();
        assert_eq!(removed, 2);
        for (p, q) in simpler.poles().iter().zip(curve.poles()) {
            assert!(p.distance(*q) < 1e-14, "{p:?} vs {q:?}");
        }
    }

    #[test]
    fn test_remove_knot_keeps_untouched_rational_poles_exactly() {
        // Removing 1.8 again solves poles 4 and 5 only.
        let curve = clamped_curve(true);
        let refined = curve.insert_knot(1.8, 1).unwrap();
        let (simpler, removed) = refined.remove_knot(1.8, 1, 1e-9).unwrap();
        assert_eq!(removed, 1);
        let (poles, weights) = (curve.poles(), curve.weights().unwrap());
        let (new_poles, new_weights) = (simpler.poles(), simpler.weights().unwrap());
        assert_eq!(new_poles[..4], poles[..4]);
        assert_eq!(new_weights[..4], weights[..4]);
        assert_eq!(new_poles[6..], poles[6..]);
        assert_eq!(new_weights[6..], weights[6..]);
    }

    #[test]
    fn test_remove_knots_undoes_a_refinement() {
        for curve in all_insertion_curves() {
            let (first, last) = curve.bounds();
            let at = |t: f64| first + t * (last - first);
            let knots = vec![at(0.05), at(0.3), at(0.3), at(0.52), at(0.9)];
            let refined = curve.refine_knots(&knots).unwrap();
            let tol = 1e-9 * pole_size(&curve);
            let (simpler, removed) = refined.remove_knots(&knots, tol).unwrap();
            assert_eq!(removed, knots.len());
            assert_same_poles(&curve, &simpler);
            let reversed: Vec<f64> = knots.iter().rev().copied().collect();
            let (simpler, removed) = refined.remove_knots(&reversed, tol).unwrap();
            assert_eq!(removed, knots.len());
            assert_same_poles(&curve, &simpler);
        }
    }

    #[test]
    fn test_remove_knot_keeps_knots_the_shape_needs() {
        // Not the unclamped curve: three of its poles are collinear.
        let curves = all_insertion_curves()
            .into_iter()
            .filter(|c| c.is_periodic() || c.multiplicities()[0] > 1);
        for curve in curves {
            let tol = 1e-6 * pole_size(&curve);
            for (k, m) in removable_knots(&curve) {
                let (same, removed) = curve.remove_knot(k, m, tol).unwrap();
                assert_eq!(removed, 0, "u = {k}");
                assert_eq!(same, curve);
            }
            let all: Vec<f64> = removable_knots(&curve).iter().map(|&(k, _)| k).collect();
            let (same, removed) = curve.remove_knots(&all, tol).unwrap();
            assert_eq!(removed, 0);
            assert_eq!(same, curve);
        }
    }

    #[test]
    fn test_remove_knot_partially() {
        // The interior knot 0.7 is needed; two more copies of it are not.
        let curve = clamped_curve(true);
        let refined = curve.insert_knot(0.7, 2).unwrap();
        assert_eq!(refined.multiplicities()[1], 3);
        let tol = 1e-9 * pole_size(&curve);
        let (simpler, removed) = refined.remove_knot(0.7, 3, tol).unwrap();
        assert_eq!(removed, 2);
        assert_same_poles(&curve, &simpler);
        let (simpler, removed) = refined.remove_knot(0.7, u32::MAX, tol).unwrap();
        assert_eq!(removed, 2);
        assert_same_poles(&curve, &simpler);
        let (simpler, removed) = refined.remove_knot(0.7, 1, tol).unwrap();
        assert_eq!(removed, 1);
        assert_eq!(simpler.multiplicities()[1], 2);
        assert!(sampled_deviation(&curve, &simpler) <= tol);
    }

    /// `curve` with its poles replaced by `poles`.
    fn with_poles(curve: &BSplineCurve3D, poles: Vec<Point3D>) -> BSplineCurve3D {
        let (knots, mults) = (curve.knots().to_vec(), curve.multiplicities().to_vec());
        match curve.weights() {
            Some(w) => BSplineCurve3D::new_rational(
                curve.degree(),
                poles,
                w.to_vec(),
                knots,
                mults,
                curve.is_periodic(),
            ),
            None => BSplineCurve3D::new(curve.degree(), poles, knots, mults, curve.is_periodic()),
        }
        .unwrap()
    }

    #[test]
    fn test_removal_stays_within_max_deviation() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut kept, mut removed_all) = (0, 0);
        for (index, curve) in random_curves(120, 0x9e37_79b9_7f4a_7c15).iter().enumerate() {
            // Knots inserted into a random curve, then every pole moved by
            // up to `noise`: the knots come out at deviations around it.
            let (first, last) = curve.bounds();
            let insert: Vec<f64> = (0..5)
                .map(|i| first + (last - first) * (i as f64 + 0.1 + 0.8 * next()) / 5.0)
                .collect();
            let refined = curve.refine_knots(&insert).unwrap();
            let noise = 1e-3 * pole_size(&refined);
            let poles = refined
                .poles()
                .iter()
                .map(|&p| p + Vector3D::new(next() - 0.5, next() - 0.5, next() - 0.5) * noise)
                .collect();
            let noisy = with_poles(&refined, poles);
            let mut all = vec![];
            for (k, m) in removable_knots(&noisy) {
                all.extend(std::iter::repeat_n(k, m as usize));
            }
            for factor in [0.3, 1.0, 3.0, 30.0] {
                let tol = factor * noise;
                let (simpler, removed) = noisy.remove_knots(&all, tol).unwrap();
                let deviation = sampled_deviation(&noisy, &simpler);
                assert!(deviation <= tol, "case {index}: {deviation} > {tol}");
                assert_eq!(simpler.poles().len(), noisy.poles().len() - removed);
                removed_all += removed;
                kept += all.len() - removed;
            }
            for (k, m) in removable_knots(&noisy) {
                let (simpler, removed) = noisy.remove_knot(k, m, 3.0 * noise).unwrap();
                assert!(removed <= m);
                let deviation = sampled_deviation(&noisy, &simpler);
                assert!(deviation <= 3.0 * noise, "case {index}, u = {k}");
            }
        }
        assert!(removed_all > 300 && kept > 300, "{removed_all} {kept}");
    }

    #[test]
    fn test_removal_of_perturbed_insertions_meets_the_bound_and_stops_past_it() {
        // Poles moved off an inserted knot by `shift`: removal deviates by
        // about `shift`, within a tolerance well above it, not one well
        // below.
        for curve in all_insertion_curves() {
            let (first, last) = curve.bounds();
            let u = first + 0.43 * (last - first);
            let refined = curve.insert_knot(u, 1).unwrap();
            let size = pole_size(&curve);
            let shift = 1e-4 * size;
            let mut poles = refined.poles().to_vec();
            let i = poles.len() / 2;
            poles[i] = poles[i] + Vector3D::new(shift, -shift, 0.5 * shift);
            let perturbed = with_poles(&refined, poles);
            let (_, removed) = perturbed.remove_knot(u, 1, 1e-3 * shift).unwrap();
            assert_eq!(removed, 0);
            let tol = 100.0 * shift;
            let (simpler, removed) = perturbed.remove_knot(u, 1, tol).unwrap();
            assert_eq!(removed, 1);
            assert!(sampled_deviation(&perturbed, &simpler) <= tol);
        }
    }

    #[test]
    fn test_remove_collinear_polyline_vertices() {
        let poles: Vec<Point3D> = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            .iter()
            .map(|&x| Point3D::new(x, if x == 3.0 { 1.0 } else { 0.0 }, 0.0))
            .collect();
        let knots: Vec<f64> = (0..6).map(f64::from).collect();
        let curve = BSplineCurve3D::new(1, poles, knots, vec![2, 1, 1, 1, 1, 2], false).unwrap();
        let (simpler, removed) = curve.remove_knots(&[1.0, 2.0, 3.0, 4.0], 1e-12).unwrap();
        assert_eq!(removed, 1);
        assert_eq!(simpler.knots(), &[0.0, 2.0, 3.0, 4.0, 5.0]);
        // A loose tolerance cuts the corner, within the bound.
        let (cut, removed) = curve.remove_knots(&[1.0, 2.0, 3.0, 4.0], 0.6).unwrap();
        assert!(removed >= 2);
        assert!(sampled_deviation(&curve, &cut) <= 0.6);
    }

    #[test]
    fn test_remove_every_knot_within_a_loose_tolerance_leaves_a_bezier() {
        let curve = clamped_curve(false);
        let all = [0.7, 1.0, 1.0, 2.5];
        let tol = 1e3 * pole_size(&curve);
        let (bezier, removed) = curve.remove_knots(&all, tol).unwrap();
        assert_eq!(removed, 4);
        assert_eq!(bezier.knots(), &[0.0, 4.0]);
        assert_eq!(bezier.poles().len(), 4);
        assert!(sampled_deviation(&curve, &bezier) <= tol);
    }

    #[test]
    fn test_remove_knot_rational_far_from_origin() {
        // The bound is measured about the poles, not the origin: a curve
        // moved far away removes the same knots.
        let offset = Vector3D::new(3e4, -2e4, 1e4);
        let moved = clamped_curve(true).transformed(Transform::translation(offset));
        let refined = moved.insert_knot(1.6, 2).unwrap();
        let (simpler, removed) = refined.remove_knot(1.6, 2, 1e-8).unwrap();
        assert_eq!(removed, 2);
        assert_eq!(simpler.knots(), moved.knots());
        assert!(sampled_deviation(&moved, &simpler) <= 1e-8);
    }

    #[test]
    fn test_remove_seam_knot() {
        let curve = periodic_seam_curve(true);
        let (first, last) = curve.bounds();
        // The double seam keeps one copy at most removed; the shape needs
        // both here.
        let (same, removed) = curve.remove_knot(last, 2, 1e-6).unwrap();
        assert_eq!(removed, 0);
        assert_eq!(same, curve);
        // A single seam always stays, however loose the tolerance.
        let ring = periodic_ring_curve();
        let (same, removed) = ring.remove_knot(first, 1, 1e6).unwrap();
        assert_eq!(removed, 0);
        assert_eq!(same, ring);
        // A loose tolerance takes the second seam copy away.
        let curve = periodic_seam_curve(false);
        let (simpler, removed) = curve.remove_knot(first, 2, 1e6).unwrap();
        assert_eq!(removed, 1);
        assert_eq!(simpler.multiplicities()[0], 1);
        assert_eq!(simpler.bounds(), curve.bounds());
        assert!(sampled_deviation(&curve, &simpler) <= 1e6);
    }

    #[test]
    fn test_remove_from_short_periodic_curve() {
        // Three poles of a quadratic: a removal needs two poles per period.
        let curve = short_periodic_curve();
        let (_, removed) = curve.remove_knot(0.5, 1, 1e6).unwrap();
        assert_eq!(removed, 1);
        let refined = curve.insert_knot(1.2, 1).unwrap();
        let (simpler, removed) = refined.remove_knot(1.2, 1, 1e-9).unwrap();
        assert_eq!(removed, 1);
        assert_same_poles(&curve, &simpler);
    }

    #[test]
    fn test_remove_zero_times_or_no_knots_is_unchanged() {
        let curve = clamped_curve(true);
        assert_eq!(curve.remove_knot(0.7, 0, 1.0).unwrap(), (curve.clone(), 0));
        assert_eq!(curve.remove_knots(&[], 1.0).unwrap(), (curve.clone(), 0));
    }

    #[test]
    fn test_knot_removal_errors() {
        for curve in [
            clamped_curve(false),
            periodic_ring_curve(),
            unclamped_curve(),
        ] {
            let (first, last) = curve.bounds();
            let knot = removable_knots(&curve)[1].0;
            for tol in [-1e-9, f64::NAN, f64::INFINITY] {
                assert_eq!(
                    curve.remove_knot(knot, 1, tol),
                    Err(KnotRemovalError::InvalidDeviation)
                );
                assert_eq!(
                    curve.remove_knots(&[], tol),
                    Err(KnotRemovalError::InvalidDeviation)
                );
            }
            for u in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert_eq!(
                    curve.remove_knot(u, 1, 1e-3),
                    Err(KnotRemovalError::NonFiniteKnot)
                );
                assert_eq!(
                    curve.remove_knots(&[knot, u], 1e-3),
                    Err(KnotRemovalError::NonFiniteKnot)
                );
            }
            for u in [first - 1.0, last + 1.0] {
                assert_eq!(
                    curve.remove_knot(u, 1, 1e-3),
                    Err(KnotRemovalError::KnotOutOfRange)
                );
            }
            let between =
                0.5 * (knot + curve.knots()[curve.knots().partition_point(|&k| k <= knot)]);
            assert_eq!(
                curve.remove_knot(between, 1, 1e-3),
                Err(KnotRemovalError::KnotNotFound)
            );
            assert_eq!(
                curve.remove_knot(between, 0, 1e-3),
                Err(KnotRemovalError::KnotNotFound)
            );
            assert_eq!(
                curve.remove_knots(&[knot, between], 1e-3),
                Err(KnotRemovalError::KnotNotFound)
            );
        }
        // A non-periodic curve keeps its end knots, clamped or not.
        for (curve, ends) in [
            (clamped_curve(true), [0.0, 4.0]),
            (unclamped_curve(), [2.0, 7.0]),
        ] {
            for u in ends {
                assert_eq!(
                    curve.remove_knot(u, 1, 1e6),
                    Err(KnotRemovalError::KnotOutOfRange)
                );
            }
        }
        // Knots of an unclamped curve outside its bounds.
        assert_eq!(
            unclamped_curve().remove_knot(1.0, 1, 1e6),
            Err(KnotRemovalError::KnotOutOfRange)
        );
    }

    #[test]
    fn test_knot_removal_error_display() {
        for e in [
            KnotRemovalError::InvalidDeviation,
            KnotRemovalError::NonFiniteKnot,
            KnotRemovalError::KnotOutOfRange,
            KnotRemovalError::KnotNotFound,
        ] {
            assert!(!e.to_string().is_empty());
        }
    }

    #[test]
    fn test_bspline_curve3d_transformed_rational_keeps_weights() {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let poles = vec![
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(1.0, 1.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
        ];
        let curve = BSplineCurve3D::new_rational(
            2,
            poles,
            vec![1.0, w, 1.0],
            vec![0.0, 1.0],
            vec![3, 3],
            false,
        )
        .unwrap();
        let t = Transform::translation(Vector3D::new(5.0, 0.0, 0.0));
        let moved = curve.transformed(t);
        assert_eq!(moved.weights(), curve.weights());
        assert!(moved.is_rational());
        let p = moved.eval_point(1.0);
        assert!((p.x - 5.0).abs() < 1e-9 && (p.y - 1.0).abs() < 1e-9);
    }

    /// `a` and `b` agree within 1e-12 relative to `size` and to `a`.
    fn assert_close(a: Vector3D, b: Vector3D, size: f64, context: &str) {
        let tol = 1e-12 * size.max(a.magnitude());
        assert!((a - b).magnitude() <= tol, "{context}: {a:?} vs {b:?}");
    }

    /// The shortest nonzero knot span of `curve`.
    fn shortest_span(curve: &BSplineCurve3D) -> f64 {
        curve
            .knots()
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(f64::INFINITY, f64::min)
    }

    /// `piece` is a valid clamped non-periodic curve over exactly
    /// `(first, last)` evaluating as `curve` there: points and first two
    /// derivatives on a dense grid, points at its knots and ends. A piece
    /// with a span shorter than any of `curve` computes its derivatives of
    /// order `k` from poles that much closer together, their rounding
    /// growing as the ratio to the power `k`.
    fn assert_piece(curve: &BSplineCurve3D, piece: &BSplineCurve3D, (first, last): (f64, f64)) {
        let p = curve.degree();
        assert_eq!(piece.bounds(), (first, last));
        assert_eq!(piece.degree(), p);
        assert!(!piece.is_periodic());
        assert_eq!(piece.is_rational(), curve.is_rational());
        let mults = piece.multiplicities();
        assert_eq!(
            (mults[0], mults[mults.len() - 1]),
            (p as u32 + 1, p as u32 + 1)
        );
        let rebuilt = match piece.weights() {
            Some(w) => BSplineCurve3D::new_rational(
                p,
                piece.poles().to_vec(),
                w.to_vec(),
                piece.knots().to_vec(),
                mults.to_vec(),
                false,
            ),
            None => BSplineCurve3D::new(
                p,
                piece.poles().to_vec(),
                piece.knots().to_vec(),
                mults.to_vec(),
                false,
            ),
        };
        assert!(rebuilt.is_ok(), "invalid piece: {rebuilt:?}");
        let size = pole_size(curve);
        let stretch = (shortest_span(curve) / shortest_span(piece)).max(1.0);
        for i in 0..48 {
            let u = first + (last - first) * (i as f64 + 0.5) / 48.0;
            let (p, d1, d2) = curve.eval_d2(u);
            let (q, e1, e2) = piece.eval_d2(u);
            assert_close(
                p - Point3D::ORIGIN,
                q - Point3D::ORIGIN,
                size,
                &format!("u = {u}"),
            );
            assert_close(d1, e1, size * stretch, &format!("d1 at {u}"));
            assert_close(d2, e2, size * stretch * stretch, &format!("d2 at {u}"));
        }
        for &u in piece.knots() {
            let (p, q) = (curve.eval_point(u), piece.eval_point(u));
            assert_close(
                p - Point3D::ORIGIN,
                q - Point3D::ORIGIN,
                size,
                &format!("knot {u}"),
            );
        }
    }

    #[test]
    fn test_extract_evaluates_as_the_curve() {
        for curve in all_insertion_curves() {
            let (first, last) = curve.bounds();
            let at = |t: f64| first + t * (last - first);
            let interior: Vec<f64> = curve
                .knots()
                .iter()
                .copied()
                .filter(|&k| k > first && k < last)
                .collect();
            let mut ranges = vec![(at(0.13), at(0.71)), (first, last), (at(0.4), at(0.41))];
            if let (Some(&k0), Some(&k1)) = (interior.first(), interior.last()) {
                ranges.extend([(k0, at(0.9)), (at(0.05), k1), (first, k0), (k1, last)]);
            }
            if curve.is_periodic() {
                // Across the seam, wholly in the next or previous period,
                // and whole periods opened anywhere.
                ranges.extend([
                    (at(0.6), at(1.3)),
                    (at(-0.4), at(0.2)),
                    (at(1.1), at(1.5)),
                    (at(-1.7), at(-1.2)),
                    (at(0.37), at(0.37) + (last - first)),
                    (last, 2.0 * last - first),
                    (at(-1.0), first),
                ]);
                if let Some(&k) = interior.first() {
                    ranges.push((k, k + (last - first)));
                }
            }
            for (a, b) in ranges {
                let piece = curve.extract(a, b).unwrap();
                assert_piece(&curve, &piece, (a, b));
            }
            // A period on, the seam is the same knot, not a sliver span off.
            if curve.is_periodic() {
                let next = curve.extract(last, 2.0 * last - first).unwrap();
                let this = curve.extract(first, last).unwrap();
                assert_eq!(next.multiplicities(), this.multiplicities());
            }
        }
    }

    #[test]
    fn test_extract_at_knots_of_full_multiplicity_copies_poles() {
        // Cubic with a triple knot at 2: the curve is two independent
        // pieces, and extracting either inserts nothing.
        let poles: Vec<Point3D> = (0..7)
            .map(|i| Point3D::new(i as f64, ((i * 5) % 3) as f64, 0.5 * i as f64))
            .collect();
        let weights = vec![1.0, 2.0, 0.5, 1.5, 0.7, 1.1, 3.0];
        let knots = vec![0.0, 2.0, 3.0];
        let curve =
            BSplineCurve3D::new_rational(3, poles, weights, knots, vec![4, 3, 4], false).unwrap();
        let left = curve.extract(0.0, 2.0).unwrap();
        assert_eq!(left.poles(), &curve.poles()[..4]);
        assert_eq!(left.weights(), Some(&curve.weights().unwrap()[..4]));
        let right = curve.extract(2.0, 3.0).unwrap();
        assert_eq!(right.poles(), &curve.poles()[3..]);
        assert_piece(&curve, &left, (0.0, 2.0));
        assert_piece(&curve, &right, (2.0, 3.0));
        // The whole of a clamped curve is the curve.
        assert_eq!(curve.extract(0.0, 3.0).unwrap(), curve);
        let (l, r) = curve.split_at(2.0).unwrap();
        assert_eq!((l, r), (left, right));
    }

    #[test]
    fn test_extract_clamps_an_unclamped_curve() {
        let curve = unclamped_curve();
        let piece = curve.extract(2.0, 7.0).unwrap();
        assert_eq!(piece.knots()[0], 2.0);
        assert_piece(&curve, &piece, (2.0, 7.0));
    }

    #[test]
    fn test_split_at_gives_two_continuous_pieces() {
        for curve in all_insertion_curves() {
            let (first, last) = curve.bounds();
            let mut cuts = vec![first + 0.37 * (last - first), first + 0.01 * (last - first)];
            cuts.extend(curve.knots().iter().filter(|&&k| k > first && k < last));
            for u in cuts {
                let (left, right) = curve.split_at(u).unwrap();
                assert_piece(&curve, &left, (first, u));
                assert_piece(&curve, &right, (u, last));
                // `u` raised to the degree, its pole shared.
                assert_eq!(left.poles().last(), right.poles().first());
                let present = curve
                    .knots()
                    .iter()
                    .position(|&k| k == u)
                    .map_or(0, |i| curve.multiplicities()[i] as usize);
                let whole = curve.extract(first, last).unwrap().poles().len();
                assert_eq!(
                    left.poles().len() + right.poles().len(),
                    whole + curve.degree() - present + 1
                );
                let size = pole_size(&curve);
                let (p, d) = left.eval_d1(u);
                let (q, e) = right.eval_d1(u);
                assert_close(
                    p - Point3D::ORIGIN,
                    q - Point3D::ORIGIN,
                    size,
                    "split point",
                );
                if !curve.knots().contains(&u) {
                    assert_close(d, e, size, "split tangent");
                }
            }
        }
    }

    #[test]
    fn test_split_extract_and_bezier_pieces_of_random_curves() {
        let mut next = uniform(0x2545_f491_4f6c_dd1d);
        let mut checked = 0;
        for case in 0..300 {
            let Some(curve) = random_curve(case, &mut next) else {
                continue;
            };
            let (first, last) = curve.bounds();
            let period = last - first;
            let (s, t) = (next(), next());
            let (a, b) = if curve.is_periodic() {
                // Anywhere within two periods, up to one period long.
                let a = first + (2.0 * s - 0.5) * period;
                (a, a + t * period)
            } else {
                (first + s.min(t) * period, first + s.max(t) * period)
            };
            if a < b {
                assert_piece(&curve, &curve.extract(a, b).unwrap(), (a, b));
            }
            let u = first + (0.05 + 0.9 * next()) * period;
            let (left, right) = curve.split_at(u).unwrap();
            assert_piece(&curve, &left, (first, u));
            assert_piece(&curve, &right, (u, last));
            for piece in curve.to_bezier_pieces() {
                assert_eq!(piece.poles().len(), curve.degree() + 1);
                assert_piece(&curve, &piece, piece.bounds());
            }
            checked += 1;
        }
        assert!(checked > 200, "{checked}");
    }

    #[test]
    fn test_split_and_extract_errors() {
        for curve in [clamped_curve(false), periodic_seam_curve(true)] {
            let (first, last) = curve.bounds();
            let mid = 0.5 * (first + last);
            for u in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert_eq!(curve.split_at(u), Err(BSplineSplitError::NonFinite));
                assert_eq!(curve.extract(u, mid), Err(BSplineSplitError::NonFinite));
                assert_eq!(curve.extract(mid, u), Err(BSplineSplitError::NonFinite));
            }
            for u in [first - 1e-9, last + 1e-9] {
                assert_eq!(curve.split_at(u), Err(BSplineSplitError::OutOfRange));
            }
            for u in [first, last] {
                assert_eq!(curve.split_at(u), Err(BSplineSplitError::EmptyRange));
            }
            assert_eq!(curve.extract(mid, mid), Err(BSplineSplitError::EmptyRange));
            assert_eq!(
                curve.extract(mid, first),
                Err(BSplineSplitError::EmptyRange)
            );
        }
        let curve = clamped_curve(true);
        assert_eq!(curve.extract(-0.1, 1.0), Err(BSplineSplitError::OutOfRange));
        assert_eq!(
            curve.extract(1.0, 4.0 + 1e-9),
            Err(BSplineSplitError::OutOfRange)
        );
        let curve = periodic_seam_curve(false);
        assert!(curve.extract(-5.0, 1.0).is_ok());
        assert_eq!(
            curve.extract(-5.0, 1.0 + 1e-6),
            Err(BSplineSplitError::ExceedsPeriod)
        );
        for e in [
            BSplineSplitError::NonFinite,
            BSplineSplitError::OutOfRange,
            BSplineSplitError::EmptyRange,
            BSplineSplitError::ExceedsPeriod,
        ] {
            assert!(!e.to_string().is_empty());
        }
    }

    /// `reversed` evaluates as `curve` at the mirrored parameter, odd
    /// derivatives with the opposite sign, and reverses back to `curve`.
    fn assert_reversed(curve: &BSplineCurve3D, reversed: &BSplineCurve3D) {
        let (first, last) = curve.bounds();
        assert_eq!(reversed.bounds(), (first, last));
        assert_eq!(reversed.is_periodic(), curve.is_periodic());
        assert_eq!(reversed.is_rational(), curve.is_rational());
        let size = pole_size(curve);
        let mut params: Vec<f64> = (0..=48)
            .map(|i| first + (last - first) * i as f64 / 48.0)
            .collect();
        if curve.is_periodic() {
            params.extend([first - 0.3 * (last - first), last + 0.45 * (last - first)]);
        }
        for u in params {
            let mirrored = match u {
                u if u == first => last,
                u if u == last => first,
                u => first + last - u,
            };
            let (p, d1, d2) = curve.eval_d2(mirrored);
            let (q, e1, e2) = reversed.eval_d2(u);
            assert_close(
                p - Point3D::ORIGIN,
                q - Point3D::ORIGIN,
                size,
                &format!("u = {u}"),
            );
            // At a knot the two sides of a jump in a derivative trade
            // places.
            if !curve.knots().contains(&mirrored) {
                assert_close(d1, -e1, size, &format!("d1 at {u}"));
                assert_close(d2, e2, size, &format!("d2 at {u}"));
            }
        }
        let back = reversed.reversed();
        assert_eq!(back.poles(), curve.poles());
        assert_eq!(back.weights(), curve.weights());
        assert_eq!(back.multiplicities(), curve.multiplicities());
        let scale = first.abs().max(last.abs());
        for (k, l) in back.knots().iter().zip(curve.knots()) {
            assert!((k - l).abs() <= f64::EPSILON * scale, "{k} vs {l}");
        }
        assert_eq!(back.bounds(), curve.bounds());
    }

    #[test]
    fn test_reversed_mirrors_the_parameter() {
        for curve in all_insertion_curves() {
            let reversed = curve.reversed();
            assert_reversed(&curve, &reversed);
        }
        let mut next = uniform(0x0bad_5eed_1234_5678);
        for case in 0..200 {
            if let Some(curve) = random_curve(case, &mut next) {
                assert_reversed(&curve, &curve.reversed());
            }
        }
        // Dyadic knots mirror exactly: reversing twice gives the curve back.
        let poles: Vec<Point3D> = (0..6)
            .map(|i| Point3D::new(i as f64, (i * i % 4) as f64, 0.0))
            .collect();
        let weights = vec![1.0, 0.5, 2.0, 1.25, 0.75, 1.0];
        let knots = vec![0.0, 1.0, 2.5, 4.0];
        let curve = BSplineCurve3D::new_rational(3, poles, weights, knots, vec![4, 1, 1, 4], false)
            .unwrap()
            .extract(0.5, 3.75)
            .unwrap();
        assert_eq!(curve.reversed().reversed(), curve);
        assert_eq!(
            periodic_ring_curve().reversed().reversed(),
            periodic_ring_curve()
        );
    }

    #[test]
    fn test_extract_keeps_untouched_rational_poles_exactly() {
        // Cubic over knots 0..7: cutting at 1.5 keeps the poles from 4 on,
        // cutting at 5.5 those up to 5.
        let poles: Vec<Point3D> = (0..10)
            .map(|i| {
                let x = i as f64;
                Point3D::new(x.sin(), (1.7 * x).cos() / 3.0, 0.7 * x.sqrt())
            })
            .collect();
        let weights: Vec<f64> = (0..10).map(|i| 1.0 / (1.3 + 0.37 * i as f64)).collect();
        let knots: Vec<f64> = (0..8).map(f64::from).collect();
        let mut mults = vec![1; 8];
        (mults[0], mults[7]) = (4, 4);
        let curve = BSplineCurve3D::new_rational(3, poles, weights, knots, mults, false).unwrap();
        let piece = curve.extract(1.5, 5.5).unwrap();
        assert_piece(&curve, &piece, (1.5, 5.5));
        assert_eq!(piece.poles().len(), 8);
        assert_eq!(&piece.poles()[3..5], &curve.poles()[4..6]);
        assert_eq!(
            &piece.weights().unwrap()[3..5],
            &curve.weights().unwrap()[4..6]
        );
        // Across the seam of a periodic curve, the kept poles wrap.
        let curve = periodic_seam_curve(true);
        let (first, last) = curve.bounds();
        let period = last - first;
        let piece = curve
            .extract(last - 0.3 * period, last + 0.4 * period)
            .unwrap();
        assert_piece(&curve, &piece, (last - 0.3 * period, last + 0.4 * period));
        let weights = curve.weights().unwrap();
        let kept = piece
            .poles()
            .iter()
            .zip(piece.weights().unwrap())
            .filter(|&(p, w)| {
                curve
                    .poles()
                    .iter()
                    .zip(weights)
                    .any(|(q, v)| (p, w) == (q, v))
            })
            .count();
        assert!(kept > 0, "no pole kept exactly");
    }
}
