//! Discretization: points on a bounded curve, in order of parameter,
//! joined by straight segments, each segment carrying a proven chord
//! deviation and angle deviation (see `CONTEXT.md`).
//!
//! The work is done in two steps, so a caller controls memory and threads:
//! [`Discretize3D::point_count`] counts the points for the given criteria,
//! and [`Discretize3D::discretize_into`] writes exactly that many into
//! slices the caller owns, without allocating. [`Discretize3D::discretize`]
//! wraps both and allocates the result.
//!
//! Ranges are given by trimmed curves: an infinite line, parabola or
//! hyperbola has no discretization until it is trimmed.
//!
//! ```compile_fail
//! use geomcore::discretization::{DiscretizationCriteria, Discretize3D};
//! use geomcore::{Line3D, Point3D, Vector3D};
//!
//! let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
//! line.discretize(DiscretizationCriteria::new(0.1).unwrap());
//! ```

mod bspline;
mod elementary;

#[cfg(doc)]
use crate::Tolerance;
use crate::curves::{ParametricCurve2D, ParametricCurve3D, TrimmedCurve2D, TrimmedCurve3D};
use crate::{BSplineCurve3D, Circle2D, Circle3D, Ellipse3D, Point2D, Point3D};
use std::fmt;

/// Error returned when a [`DiscretizationCriteria`] value is not a
/// positive finite number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscretizationCriteriaError {
    /// The maximum chord deviation is not positive and finite.
    InvalidChordDeviation,
    /// The maximum angle deviation is not positive and finite.
    InvalidAngleDeviation,
}

impl fmt::Display for DiscretizationCriteriaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DiscretizationCriteriaError::InvalidChordDeviation => {
                "the maximum chord deviation must be positive and finite"
            }
            DiscretizationCriteriaError::InvalidAngleDeviation => {
                "the maximum angle deviation must be positive and finite"
            }
        })
    }
}

impl std::error::Error for DiscretizationCriteriaError {}

/// What a discretization must meet: a maximum chord deviation and,
/// optionally, a maximum angle deviation (radians).
///
/// # Examples
///
/// ```
/// use geomcore::discretization::DiscretizationCriteria;
///
/// let criteria = DiscretizationCriteria::new(1e-3)?.with_max_angle_deviation(0.1)?;
/// assert_eq!(criteria.max_chord_deviation(), 1e-3);
/// assert_eq!(criteria.max_angle_deviation(), Some(0.1));
/// # Ok::<(), geomcore::discretization::DiscretizationCriteriaError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiscretizationCriteria {
    max_chord_deviation: f64,
    max_angle_deviation: Option<f64>,
}

impl DiscretizationCriteria {
    /// Criteria asking for a chord deviation of at most
    /// `max_chord_deviation`, with no limit on the angle deviation.
    ///
    /// Any positive chord deviation is accepted, but the points cannot lie
    /// closer to the curve than their coordinates' rounding: about 1e-15
    /// times the curve's distance from the origin, 1e-13 near it and 1e-10
    /// at the edge of a [`Tolerance::MODEL_SIZE`] model. A finer request
    /// only adds points, about three times as many for each tenfold;
    /// [`point_count`](Discretize3D::point_count) gives the count before
    /// anything is allocated.
    pub fn new(max_chord_deviation: f64) -> Result<Self, DiscretizationCriteriaError> {
        if !(max_chord_deviation.is_finite() && max_chord_deviation > 0.0) {
            return Err(DiscretizationCriteriaError::InvalidChordDeviation);
        }
        Ok(DiscretizationCriteria {
            max_chord_deviation,
            max_angle_deviation: None,
        })
    }

    /// The same criteria, also asking for an angle deviation of at most
    /// `max_angle_deviation` radians. A limit of π or more asks for
    /// nothing: no angle exceeds π.
    pub fn with_max_angle_deviation(
        self,
        max_angle_deviation: f64,
    ) -> Result<Self, DiscretizationCriteriaError> {
        if !(max_angle_deviation.is_finite() && max_angle_deviation > 0.0) {
            return Err(DiscretizationCriteriaError::InvalidAngleDeviation);
        }
        Ok(DiscretizationCriteria {
            max_angle_deviation: Some(max_angle_deviation),
            ..self
        })
    }

    /// The largest chord deviation a segment may have.
    pub fn max_chord_deviation(&self) -> f64 {
        self.max_chord_deviation
    }

    /// The largest angle deviation a segment may have, if limited.
    pub fn max_angle_deviation(&self) -> Option<f64> {
        self.max_angle_deviation
    }

    /// The angle limit that constrains anything: below π.
    fn angle_limit(&self) -> Option<f64> {
        self.max_angle_deviation
            .filter(|&a| a < std::f64::consts::PI)
    }
}

/// A discretized 3D curve: points on the curve in order of parameter,
/// joined by straight segments.
///
/// `parameters[i]` is the parameter of `points[i]`; segment `i` joins
/// `points[i]` and `points[i + 1]` and carries proven upper bounds on its
/// chord deviation (the largest distance from the curve to the segment)
/// and angle deviation (the largest angle, in radians, between the
/// curve's tangent and the segment), so the deviation lists have one
/// entry fewer than the points. An angle deviation of π means no bound
/// was found there: the curve's speed may reach or nearly reach zero, or,
/// when no angle limit was asked, the segment's piece of B-spline turns
/// too much to be bounded without the halving only that limit pays for.
///
/// The bounds are proven in exact arithmetic from the curve's definition;
/// the rounding of evaluating a point, a few units in the last place of
/// its coordinates, is not included.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct Discretization3D {
    /// The parameter of each point, increasing.
    pub parameters: Vec<f64>,
    /// The points on the curve.
    pub points: Vec<Point3D>,
    /// The chord deviation of each segment.
    pub chord_deviations: Vec<f64>,
    /// The angle deviation of each segment, in radians.
    pub angle_deviations: Vec<f64>,
}

/// Discretization of a bounded 3D curve over its whole
/// [`bounds`](ParametricCurve3D::bounds).
///
/// Both end points are the curve's points at the range ends; a closed
/// curve (a full circle, a periodic B-spline, a trimmed curve spanning a
/// whole period) gives a last point identical to the first and at least
/// three segments. The same curve and criteria always give the same
/// points.
///
/// Elementary curves are split into equal parameter steps, the fewest the
/// closed-form deviations allow. B-splines are split per knot span, on the
/// span's Bézier control points: a piece is halved while that saves
/// points, then split into the fewest equal steps its bounds allow.
/// The tangent's turning is bounded from the derivative's control points;
/// with an angle limit, a piece where the speed may reach or nearly reach
/// zero (a cusp, coincident or nearly coincident poles) is halved a few
/// times, and what remains reports an angle deviation of π while still
/// meeting the chord deviation. Without an angle limit, pieces are halved
/// for the chord deviation only, and one turning too much reports π. A
/// rational piece's tangent is bounded the same way, in homogeneous form.
///
/// # Examples
///
/// ```
/// use geomcore::discretization::{DiscretizationCriteria, Discretize3D};
/// use geomcore::{Circle3D, Point3D, Vector3D};
///
/// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
/// let criteria = DiscretizationCriteria::new(1e-3).unwrap();
///
/// // Count, then fill buffers the caller owns...
/// let n = circle.point_count(criteria);
/// let (mut parameters, mut points) = (vec![0.0; n], vec![Point3D::ORIGIN; n]);
/// let (mut chords, mut angles) = (vec![0.0; n - 1], vec![0.0; n - 1]);
/// circle.discretize_into(criteria, &mut parameters, &mut points, &mut chords, &mut angles);
/// assert!(chords.iter().all(|&d| d <= 1e-3));
///
/// // ...or let geomcore allocate the result.
/// let d = circle.discretize(criteria);
/// assert_eq!(d.points, points);
/// assert_eq!(d.points[0], d.points[n - 1]);
/// ```
pub trait Discretize3D {
    /// The number of points the discretization has for `criteria`.
    ///
    /// A chord deviation far below the curve's size asks for as many points
    /// as it takes; the count saturates at `usize::MAX`, which no buffer
    /// can hold.
    fn point_count(&self, criteria: DiscretizationCriteria) -> usize;

    /// Writes the discretization into caller-owned slices, without
    /// allocating: [`point_count`](Self::point_count) parameters and
    /// points, and one chord deviation and angle deviation fewer.
    ///
    /// # Panics
    ///
    /// Panics if a slice has the wrong length.
    fn discretize_into(
        &self,
        criteria: DiscretizationCriteria,
        parameters: &mut [f64],
        points: &mut [Point3D],
        chord_deviations: &mut [f64],
        angle_deviations: &mut [f64],
    );

    /// Counts, allocates and fills the discretization.
    fn discretize(&self, criteria: DiscretizationCriteria) -> Discretization3D {
        let n = self.point_count(criteria);
        let mut d = Discretization3D {
            parameters: vec![0.0; n],
            points: vec![Point3D::ORIGIN; n],
            chord_deviations: vec![0.0; n - 1],
            angle_deviations: vec![0.0; n - 1],
        };
        self.discretize_into(
            criteria,
            &mut d.parameters,
            &mut d.points,
            &mut d.chord_deviations,
            &mut d.angle_deviations,
        );
        d
    }
}

/// A discretized 2D curve, such as a curve in a surface's parameter
/// plane: the 2D counterpart of [`Discretization3D`], its chord
/// deviations being distances in the plane.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct Discretization2D {
    /// The parameter of each point, increasing.
    pub parameters: Vec<f64>,
    /// The points on the curve.
    pub points: Vec<Point2D>,
    /// The chord deviation of each segment.
    pub chord_deviations: Vec<f64>,
    /// The angle deviation of each segment, in radians.
    pub angle_deviations: Vec<f64>,
}

/// Discretization of a bounded 2D curve over its whole
/// [`bounds`](ParametricCurve2D::bounds): the 2D counterpart of
/// [`Discretize3D`], with the same rules.
///
/// # Examples
///
/// ```
/// use geomcore::discretization::{DiscretizationCriteria, Discretize2D};
/// use geomcore::{Circle2D, Line2D, Point2D, TrimmedCurve2D, Vector2D};
///
/// let criteria = DiscretizationCriteria::new(1e-3).unwrap();
/// let circle = Circle2D::new(Point2D::ORIGIN, 1.0).unwrap();
/// let d = circle.discretize(criteria);
/// assert_eq!(d.points[0], d.points[d.points.len() - 1]);
///
/// // A segment is its own chord.
/// let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
/// let segment = TrimmedCurve2D::new(line, 0.0, 2.0).unwrap();
/// assert_eq!(segment.point_count(criteria), 2);
/// ```
pub trait Discretize2D {
    /// The number of points the discretization has for `criteria`.
    ///
    /// A chord deviation far below the curve's size asks for as many points
    /// as it takes; the count saturates at `usize::MAX`, which no buffer
    /// can hold.
    fn point_count(&self, criteria: DiscretizationCriteria) -> usize;

    /// Writes the discretization into caller-owned slices, without
    /// allocating: [`point_count`](Self::point_count) parameters and
    /// points, and one chord deviation and angle deviation fewer.
    ///
    /// # Panics
    ///
    /// Panics if a slice has the wrong length.
    fn discretize_into(
        &self,
        criteria: DiscretizationCriteria,
        parameters: &mut [f64],
        points: &mut [Point2D],
        chord_deviations: &mut [f64],
        angle_deviations: &mut [f64],
    );

    /// Counts, allocates and fills the discretization.
    fn discretize(&self, criteria: DiscretizationCriteria) -> Discretization2D {
        let n = self.point_count(criteria);
        let mut d = Discretization2D {
            parameters: vec![0.0; n],
            points: vec![Point2D::ORIGIN; n],
            chord_deviations: vec![0.0; n - 1],
            angle_deviations: vec![0.0; n - 1],
        };
        self.discretize_into(
            criteria,
            &mut d.parameters,
            &mut d.points,
            &mut d.chord_deviations,
            &mut d.angle_deviations,
        );
        d
    }
}

pub(crate) mod sealed {
    use super::DiscretizationCriteria;

    /// Equal parameter steps over `[first, last]`, each segment with the
    /// same deviations.
    #[derive(Debug, Clone, Copy)]
    pub struct Run {
        pub(crate) first: f64,
        pub(crate) last: f64,
        pub(crate) segments: usize,
        pub(crate) chord_deviation: f64,
        pub(crate) angle_deviation: f64,
    }

    /// Discretization of a range of a curve: what a trimmed curve needs
    /// from its basis. Implemented by every curve, bounded or not, in 3D
    /// and in 2D.
    pub trait DiscretizeRange {
        /// The curve's points: `Point3D` or `Point2D`.
        type Point: Copy;

        /// Calls `visit` with the runs covering `[first, last]` in order,
        /// each with a function evaluating the point at a fraction of
        /// the run. Every run has at least `min_segments` segments.
        fn for_each_run<F>(
            &self,
            first: f64,
            last: f64,
            criteria: &DiscretizationCriteria,
            min_segments: usize,
            visit: F,
        ) where
            F: FnMut(Run, &dyn Fn(f64) -> Self::Point);
    }
}
use sealed::{DiscretizeRange, Run};

/// The point count of `[first, last]`, and the fewest segments per run
/// that give a closed result at least three segments.
fn plan<C: DiscretizeRange + ?Sized>(
    curve: &C,
    (first, last): (f64, f64),
    closed: bool,
    criteria: &DiscretizationCriteria,
) -> (usize, usize) {
    let count = |min_segments| {
        let (mut runs, mut segments) = (0usize, 0usize);
        curve.for_each_run(first, last, criteria, min_segments, |run, _| {
            runs += 1;
            segments = segments.saturating_add(run.segments);
        });
        (runs, segments)
    };
    let (runs, segments) = count(1);
    if closed && segments < 3 {
        let min_segments = 3usize.div_ceil(runs);
        (count(min_segments).1 + 1, min_segments)
    } else {
        (segments.saturating_add(1), 1)
    }
}

/// The caller's slices a discretization is written into.
struct Slices<'a, P> {
    parameters: &'a mut [f64],
    points: &'a mut [P],
    chord_deviations: &'a mut [f64],
    angle_deviations: &'a mut [f64],
}

impl<P> Slices<'_, P> {
    /// Panics unless the slices hold `n` points.
    fn check_lengths(&self, n: usize) {
        assert_eq!(self.parameters.len(), n, "parameters: wrong length");
        assert_eq!(self.points.len(), n, "points: wrong length");
        let segments = n - 1;
        let chords = self.chord_deviations.len();
        assert_eq!(chords, segments, "chord deviations: wrong length");
        let angles = self.angle_deviations.len();
        assert_eq!(angles, segments, "angle deviations: wrong length");
    }
}

/// Writes the discretization of `[first, last]` into `out`, the end points
/// from `point_at_parameter`: the runs may leave out a rounding error at
/// either end.
fn fill<C: DiscretizeRange + ?Sized>(
    curve: &C,
    (first, last): (f64, f64),
    closed: bool,
    criteria: &DiscretizationCriteria,
    point_at_parameter: impl Fn(f64) -> C::Point,
    out: Slices<'_, C::Point>,
) {
    let (n, min_segments) = plan(curve, (first, last), closed, criteria);
    out.check_lengths(n);
    let mut i = 0;
    curve.for_each_run(first, last, criteria, min_segments, |run, point_at| {
        let step = 1.0 / run.segments as f64;
        for j in 0..run.segments {
            let s = j as f64 * step;
            out.parameters[i] = run.first + (run.last - run.first) * s;
            out.points[i] = point_at(s);
            out.chord_deviations[i] = run.chord_deviation;
            out.angle_deviations[i] = run.angle_deviation;
            i += 1;
        }
    });
    out.parameters[0] = first;
    out.parameters[n - 1] = last;
    out.points[0] = point_at_parameter(first);
    out.points[n - 1] = if closed {
        out.points[0]
    } else {
        point_at_parameter(last)
    };
}

/// Parameters, points, chord deviations and angle deviations.
type Buffers<P> = (Vec<f64>, Vec<P>, Vec<f64>, Vec<f64>);

/// The discretization of `[first, last]` in a single pass, into buffers
/// that grow as the runs come: parameters, points, chord deviations and
/// angle deviations, as [`fill`] writes them.
fn collect<C: DiscretizeRange + ?Sized>(
    curve: &C,
    (first, last): (f64, f64),
    closed: bool,
    criteria: &DiscretizationCriteria,
    point_at_parameter: impl Fn(f64) -> C::Point,
) -> Buffers<C::Point> {
    let gather = |min_segments, out: &mut Buffers<C::Point>| {
        let (parameters, points, chord_deviations, angle_deviations) = out;
        parameters.clear();
        points.clear();
        chord_deviations.clear();
        angle_deviations.clear();
        let mut runs = 0usize;
        curve.for_each_run(first, last, criteria, min_segments, |run, point_at| {
            runs += 1;
            let step = 1.0 / run.segments as f64;
            for j in 0..run.segments {
                let s = j as f64 * step;
                parameters.push(run.first + (run.last - run.first) * s);
                points.push(point_at(s));
            }
            chord_deviations.extend(std::iter::repeat_n(run.chord_deviation, run.segments));
            angle_deviations.extend(std::iter::repeat_n(run.angle_deviation, run.segments));
        });
        runs
    };
    let mut out = Default::default();
    let runs = gather(1, &mut out);
    // The same rule as `plan`.
    if closed && out.2.len() < 3 {
        gather(3usize.div_ceil(runs), &mut out);
    }
    let (mut parameters, mut points, chord_deviations, angle_deviations) = out;
    let start = point_at_parameter(first);
    parameters.push(last);
    points.push(start);
    let n = points.len();
    parameters[0] = first;
    points[0] = start;
    parameters[n - 1] = last;
    if !closed {
        points[n - 1] = point_at_parameter(last);
    }
    (parameters, points, chord_deviations, angle_deviations)
}

/// The smallest count `k >= 1` with `fits(k)`, `fits` being monotone,
/// searched from the estimate `guess` (which may be off by rounding, or
/// by far): `usize::MAX` if no count fits.
fn smallest_count(guess: f64, fits: impl Fn(usize) -> bool) -> usize {
    // `as` saturates: NaN gives 0, infinity `usize::MAX`.
    let start = (guess.ceil() as usize).max(1);
    // Gallop away from the guess until `low` does not fit (or is 0) and
    // `high` fits, then bisect.
    let (mut low, mut high) = (0, start);
    let mut step = 1usize;
    if fits(start) {
        while high > 1 {
            let below = high.saturating_sub(step).max(1);
            if !fits(below) {
                low = below;
                break;
            }
            high = below;
            step = step.saturating_mul(2);
        }
    } else {
        low = start;
        loop {
            if low == usize::MAX {
                return usize::MAX;
            }
            let above = low.saturating_add(step);
            if fits(above) {
                high = above;
                break;
            }
            low = above;
            step = step.saturating_mul(2);
        }
    }
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if fits(middle) {
            high = middle;
        } else {
            low = middle;
        }
    }
    high
}

/// Implements a discretization trait (`Discretize3D` or `Discretize2D`,
/// writing points of type `$point`) over the whole bounds of bounded
/// curves.
macro_rules! discretize_bounded {
    ($discretize:ident, $discretization:ident, $point:ty: $($curve:ty),*) => {$(
        impl $discretize for $curve {
            fn discretize(&self, criteria: DiscretizationCriteria) -> $discretization {
                let (parameters, points, chord_deviations, angle_deviations) = collect(
                    self,
                    self.bounds(),
                    self.is_periodic(),
                    &criteria,
                    |u| self.eval_point(u),
                );
                $discretization {
                    parameters,
                    points,
                    chord_deviations,
                    angle_deviations,
                }
            }

            fn point_count(&self, criteria: DiscretizationCriteria) -> usize {
                plan(self, self.bounds(), self.is_periodic(), &criteria).0
            }

            fn discretize_into(
                &self,
                criteria: DiscretizationCriteria,
                parameters: &mut [f64],
                points: &mut [$point],
                chord_deviations: &mut [f64],
                angle_deviations: &mut [f64],
            ) {
                fill(
                    self,
                    self.bounds(),
                    self.is_periodic(),
                    &criteria,
                    |u| self.eval_point(u),
                    Slices {
                        parameters,
                        points,
                        chord_deviations,
                        angle_deviations,
                    },
                );
            }
        }
    )*};
}

discretize_bounded!(Discretize3D, Discretization3D, Point3D: Circle3D, Ellipse3D, BSplineCurve3D);
discretize_bounded!(Discretize2D, Discretization2D, Point2D: Circle2D);

/// Implements a discretization trait for a trimmed curve type over its
/// range: closed when the range spans a whole period of its basis.
macro_rules! discretize_trimmed {
    ($discretize:ident, $discretization:ident, $point:ty, $trimmed:ident, $parametric:ident) => {
        impl<C: DiscretizeRange<Point = $point> + $parametric> $discretize for $trimmed<C> {
            fn discretize(&self, criteria: DiscretizationCriteria) -> $discretization {
                let (parameters, points, chord_deviations, angle_deviations) = collect(
                    self.basis(),
                    self.bounds(),
                    spans_period(self.bounds(), self.basis().period()),
                    &criteria,
                    |u| self.basis().eval_point(u),
                );
                $discretization {
                    parameters,
                    points,
                    chord_deviations,
                    angle_deviations,
                }
            }

            fn point_count(&self, criteria: DiscretizationCriteria) -> usize {
                let closed = spans_period(self.bounds(), self.basis().period());
                plan(self.basis(), self.bounds(), closed, &criteria).0
            }

            fn discretize_into(
                &self,
                criteria: DiscretizationCriteria,
                parameters: &mut [f64],
                points: &mut [$point],
                chord_deviations: &mut [f64],
                angle_deviations: &mut [f64],
            ) {
                fill(
                    self.basis(),
                    self.bounds(),
                    spans_period(self.bounds(), self.basis().period()),
                    &criteria,
                    |u| self.basis().eval_point(u),
                    Slices {
                        parameters,
                        points,
                        chord_deviations,
                        angle_deviations,
                    },
                );
            }
        }
    };
}

/// Whether `[first, last]` spans a whole `period`.
fn spans_period((first, last): (f64, f64), period: Option<f64>) -> bool {
    period.is_some_and(|period| last - first >= period * (1.0 - 1e-12))
}

discretize_trimmed!(
    Discretize3D,
    Discretization3D,
    Point3D,
    TrimmedCurve3D,
    ParametricCurve3D
);
discretize_trimmed!(
    Discretize2D,
    Discretization2D,
    Point2D,
    TrimmedCurve2D,
    ParametricCurve2D
);

impl<C: DiscretizeRange + ?Sized> DiscretizeRange for &C {
    type Point = C::Point;

    fn for_each_run<F>(
        &self,
        first: f64,
        last: f64,
        criteria: &DiscretizationCriteria,
        min_segments: usize,
        visit: F,
    ) where
        F: FnMut(Run, &dyn Fn(f64) -> Self::Point),
    {
        (**self).for_each_run(first, last, criteria, min_segments, visit);
    }
}
