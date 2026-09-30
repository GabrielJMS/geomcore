//! B-spline (NURBS) curves in 3D: de Boor evaluation with derivatives,
//! rational (weighted) poles, and periodic (closed) curves, a thin wrapper
//! over [`crate::curve_math::bspline`].
//!
//! Construction packs the poles (and, for rational curves, the weights) into
//! a cached homogeneous flat buffer once, together with the flattened knot
//! sequence, so evaluation never re-derives them.

use std::fmt;

use crate::curve_math::bspline as math;
use crate::curves::{Curve2D, ParametrizeError};
use crate::math::gauss_newton_1d;
use crate::projection::CurveProjection;
use crate::surfaces::Surface;
use crate::{Point3D, Tolerance, Vector3D};

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
                "interpolation degree exceeds points minus one"
            }
            BSplineConstructionError::ConfusedPoints => {
                "two consecutive interpolation points coincide"
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for BSplineConstructionError {}

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
    degree: usize,
    periodic: bool,
    poles: Vec<Point3D>,
    weights: Option<Vec<f64>>,
    knots: Vec<f64>,
    mults: Vec<u32>,
    /// Cached flat (expanded) knot sequence: knot `i` repeated `mults[i]`
    /// times.
    flat: Vec<f64>,
    /// Cached flat pole buffer: `dim` coordinates per pole, where
    /// `dim = 4` (homogeneous `(x*w, y*w, z*w, w)`) if rational, else `3`.
    flat_poles: Vec<f64>,
}

/// Point parametrization for [`BSplineCurve3D::interpolate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpParametrization {
    /// Cumulative chord lengths (simplest; overshoots at sharp turns).
    Chordal,
    /// Square roots of chord lengths (tamer at sharp turns; default choice).
    Centripetal,
}

/// All `n_poles` B-spline basis values at `t` over the full knot vector
/// (Cox-de Boor, `0/0 = 0` at repeated knots).
fn basis_values(full: &[f64], degree: usize, t: f64) -> Vec<f64> {
    let n_poles = full.len() - degree - 1;
    let mut out = vec![0.0; n_poles];
    let (lo, hi) = (full[degree], full[n_poles]);
    let tt = t.clamp(lo, hi);
    let mut span = n_poles - 1;
    if tt < hi {
        span = degree;
        while full[span + 1] <= tt {
            span += 1;
        }
    }
    let mut n = vec![0.0; degree + 1];
    let mut left = vec![0.0; degree + 1];
    let mut right = vec![0.0; degree + 1];
    n[0] = 1.0;
    for k in 1..=degree {
        left[k] = tt - full[span + 1 - k];
        right[k] = full[span + k] - tt;
        let mut saved = 0.0;
        for r in 0..k {
            let denom = right[r + 1] + left[k - r];
            let temp = if denom == 0.0 { 0.0 } else { n[r] / denom };
            n[r] = saved + right[r + 1] * temp;
            saved = left[k - r] * temp;
        }
        n[k] = saved;
    }
    for (r, &v) in n.iter().enumerate() {
        out[span - degree + r] = v;
    }
    out
}

/// In-place band LU without pivoting (bandwidth `p` each side), solving the
/// three coordinate columns. Safe here: averaged interpolation knots give a
/// totally positive system.
fn banded_solve(band: &mut [Vec<f64>], p: usize, rhs: &mut [Vec<f64>]) {
    let m = band.len();
    for k in 0..m {
        for i in (k + 1)..(k + p + 1).min(m) {
            let factor = band[i][p + k - i] / band[k][p];
            band[i][p + k - i] = factor;
            for j in (k + 1)..(k + p + 1).min(m) {
                band[i][p + j - i] -= factor * band[k][p + j - k];
            }
        }
    }
    for col in rhs.iter_mut() {
        for i in 0..m {
            let (done, rest) = col.split_at_mut(i);
            let elem = &mut rest[0];
            for j in i.saturating_sub(p)..i {
                *elem -= band[i][p + j - i] * done[j];
            }
        }
        for i in (0..m).rev() {
            let (left, right) = col.split_at_mut(i + 1);
            let elem = &mut left[i];
            for (k, r) in right.iter().enumerate().take(p) {
                *elem -= band[i][p + 1 + k] * r;
            }
            *elem /= band[i][p];
        }
    }
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
        math::validate_direction(degree, poles.len(), &knots, &multiplicities, periodic)?;

        let flat = math::flat_knots(&knots, &multiplicities);
        let flat_poles = pack_flat_poles(&poles, weights.as_deref());

        Ok(BSplineCurve3D {
            degree,
            periodic,
            poles,
            weights,
            knots,
            mults: multiplicities,
            flat,
            flat_poles,
        })
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
        self.degree
    }

    /// Interpolates a clamped non-rational B-spline through `points`.
    ///
    /// Parameters come from chord lengths (`Chordal`) or their square
    /// roots (`Centripetal`, better behaved at sharp turns), normalized
    /// to `[0, 1]`; interior knots are parameter averages, and the
    /// interior poles solve the banded interpolation system by LU without
    /// pivoting (safe: averaged knots satisfy Schoenberg-Whitney, so the
    /// matrix is totally positive). End poles equal the end points.
    ///
    /// # Errors
    ///
    /// Returns [`BSplineConstructionError::TooFewPoints`] for fewer than
    /// two points, [`BSplineConstructionError::InvalidDegree`] outside
    /// `1..=25`, [`BSplineConstructionError::DegreeTooHigh`] when
    /// `degree` exceeds points minus one, or
    /// [`BSplineConstructionError::ConfusedPoints`] for coincident
    /// consecutive points.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, InterpParametrization, Point3D, Tolerance};
    /// let points = vec![
    ///     Point3D::new(0.0, 0.0, 0.0),
    ///     Point3D::new(1.0, 1.0, 0.0),
    ///     Point3D::new(2.0, 0.0, 0.0),
    /// ];
    /// let curve =
    ///     BSplineCurve3D::interpolate(&points, 2, InterpParametrization::Centripetal).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// for p in &points {
    ///     assert!(curve.contains(*p, tol));
    /// }
    /// ```
    pub fn interpolate(
        points: &[Point3D],
        degree: usize,
        param: InterpParametrization,
    ) -> Result<BSplineCurve3D, BSplineConstructionError> {
        let n = points.len();
        if n < 2 {
            return Err(BSplineConstructionError::TooFewPoints);
        }
        if !(1..=25).contains(&degree) {
            return Err(BSplineConstructionError::InvalidDegree);
        }
        if degree > n - 1 {
            return Err(BSplineConstructionError::DegreeTooHigh);
        }
        // Parameters from chord lengths.
        let exponent = match param {
            InterpParametrization::Chordal => 1.0,
            InterpParametrization::Centripetal => 0.5,
        };
        let mut params = vec![0.0; n];
        for i in 1..n {
            let d = points[i].distance(points[i - 1]);
            if d <= crate::tol::CONFUSION {
                return Err(BSplineConstructionError::ConfusedPoints);
            }
            params[i] = params[i - 1] + d.powf(exponent);
        }
        let total = params[n - 1];
        for t in &mut params {
            *t /= total;
        }
        // Averaged interior knots, clamped ends; compress to distinct + mults.
        let p = degree;
        let mut full = vec![0.0; p + 1];
        for j in 1..=(n - p - 1) {
            let mut acc = 0.0;
            for k in 0..p {
                acc += params[j + k];
            }
            full.push(acc / p as f64);
        }
        full.extend(std::iter::repeat_n(1.0, p + 1));
        let mut knots = vec![full[0]];
        let mut mults = vec![1u32];
        for &u in &full[1..] {
            if u > *knots.last().expect("knots non-empty") {
                knots.push(u);
                mults.push(1);
            } else {
                let last = mults.len() - 1;
                mults[last] += 1;
            }
        }
        // Interior system: A[j][i] = N_i(t_j), ends fixed to end points.
        let m = n - 2;
        let mut poles = vec![Point3D::ORIGIN; n];
        poles[0] = points[0];
        poles[n - 1] = points[n - 1];
        if m > 0 {
            let mut band = vec![vec![0.0; 2 * p + 1]; m];
            // One right-hand side per coordinate.
            let mut rhs = vec![vec![0.0; m]; 3];
            for (row, j) in (1..n - 1).enumerate() {
                let basis = basis_values(&full, p, params[j]);
                for (i, &b) in basis.iter().enumerate().take(n - 1).skip(1) {
                    if b != 0.0 && i + p >= j && i <= j + p {
                        band[row][p + i - j] = b;
                    }
                }
                for k in 0..3 {
                    let coord = |pt: Point3D| [pt.x, pt.y, pt.z][k];
                    rhs[k][row] = coord(points[j])
                        - basis[0] * coord(points[0])
                        - basis[n - 1] * coord(points[n - 1]);
                }
            }
            banded_solve(&mut band, p, &mut rhs);
            for row in 0..m {
                poles[row + 1] = Point3D::new(rhs[0][row], rhs[1][row], rhs[2][row]);
            }
        }
        Self::new(degree, poles, knots, mults, false)
    }

    /// Returns whether the curve is periodic (closed).
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
        self.periodic
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
        &self.knots
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
        &self.mults
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
        let sum_mults: u32 = self.mults.iter().sum();
        math::param_range(&self.flat, self.degree, self.periodic, sum_mults as usize)
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
        let euc = self.eval_euclidean(u, 0);
        Point3D::new(euc[0], euc[1], euc[2])
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
            1 | 2 => {
                let euc = self.eval_euclidean(u, order as usize);
                let base = order as usize * 3;
                Vector3D::new(euc[base], euc[base + 1], euc[base + 2])
            }
            _ => panic!(
                "eval_derivative: order {order} is not supported (only first and second derivatives are supported)"
            ),
        }
    }

    /// Computes the exact 2D representation of this B-spline curve in a
    /// surface's parameter space.
    ///
    /// No B-spline-curve/surface pair has a closed-form 2D image in this
    /// release, so this always returns [`ParametrizeError::NotAnalytic`]. The
    /// `surface` argument is accepted for signature parity with the analytic
    /// [`crate::Line3D::parametrize_on`] and
    /// [`crate::Circle3D::parametrize_on`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::ParametrizeError;
    /// use geomcore::{BSplineCurve3D, Plane, Point3D, Vector3D};
    ///
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert_eq!(curve.parametrize_on(&plane), Err(ParametrizeError::NotAnalytic));
    /// ```
    pub fn parametrize_on(&self, surface: impl Into<Surface>) -> Result<Curve2D, ParametrizeError> {
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

    /// All stationary points of the distance from `point` to the curve,
    /// ordered by ascending distance.
    ///
    /// Dense uniform seeds over [`BSplineCurve3D::bounds`] are refined
    /// with Gauss-Newton on `(C(t) - P).C'(t) = 0` (first derivatives
    /// only); distinct seeds converging to the same point merge, and the
    /// interval ends join as candidates on open curves. The first entry
    /// is the global closest point (see [`BSplineCurve3D::project_point`]).
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
        let clamp = |t: f64| {
            if self.is_periodic() {
                t
            } else {
                t.clamp(first, last)
            }
        };
        // Dense seeds; keep only well-separated ones for refinement.
        const SEEDS: usize = 64;
        const KEEP: usize = 8;
        let mut samples: Vec<(f64, f64)> = (0..SEEDS)
            .map(|i| {
                let u = first + (last - first) * i as f64 / (SEEDS - 1) as f64;
                (u, self.eval_point(u).distance(point))
            })
            .collect();
        samples.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut params: Vec<f64> = Vec::new();
        for (u, _) in samples.into_iter().take(KEEP) {
            let p = self.eval_point(u);
            if params
                .iter()
                .any(|&t| self.eval_point(t).distance(p) <= tol.confusion)
            {
                continue;
            }
            let residual = |t: f64| {
                let q = self.eval_point(clamp(t)) - point;
                [q.x, q.y, q.z]
            };
            let jacobian = |t: f64| {
                let v = self.eval_derivative(clamp(t), 1);
                [v.x, v.y, v.z]
            };
            if let Some(t) = gauss_newton_1d(residual, jacobian, u, tol.confusion, 50) {
                params.push(clamp(t));
            }
        }
        // Interval ends are candidates on open curves; the best raw sample
        // guarantees a non-empty answer when nothing converges.
        if !self.is_periodic() {
            params.push(first);
            params.push(last);
        }
        if params.is_empty() {
            params.push(first);
        }
        let mut out: Vec<CurveProjection> = Vec::new();
        for t in params {
            let p = self.eval_point(t);
            if out.iter().any(|e: &CurveProjection| {
                self.eval_point(e.parameter).distance(p) <= tol.confusion
            }) {
                continue;
            }
            let distance = p.distance(point);
            out.push(CurveProjection {
                parameter: t,
                distance: snap_distance(distance, tol.confusion),
            });
        }
        out.sort_by(|x, y| x.distance.partial_cmp(&y.distance).unwrap());
        out
    }

    /// Projects `point` onto the curve: the nearest of
    /// [`BSplineCurve3D::extrema`]. Distances within `tol.confusion` snap
    /// to `0.0`, matching [`BSplineCurve3D::contains`].
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
        self.extrema(point, tol)
            .into_iter()
            .next()
            .expect("seeding guarantees a non-empty candidate list")
    }

    /// Projects each point in `points` onto the curve.
    ///
    /// Default-style batch wrapper over [`BSplineCurve3D::project_point`]:
    /// one native call per batch, mirroring [`BSplineCurve3D::eval_points`].
    pub fn project_points(&self, points: &[Point3D], tol: Tolerance) -> Vec<CurveProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }

    /// Evaluates the value and derivatives up to `n` (`n <= 2`) at `u`,
    /// returning Euclidean coordinates as `[f, f', f'']` flattened (9
    /// values, trailing ones zero if `n < 2`).
    fn eval_euclidean(&self, u: f64, n: usize) -> [f64; 9] {
        let dim = if self.is_rational() { 4 } else { 3 };
        let mut raw = vec![0.0f64; 3 * dim];
        math::eval_dn(
            u,
            self.degree,
            self.periodic,
            &self.flat,
            &self.flat_poles,
            dim,
            n,
            &mut raw,
        );
        let mut euc = [0.0f64; 9];
        if self.is_rational() {
            math::rational_derivatives(&raw, n, &mut euc);
        } else {
            euc.copy_from_slice(&raw);
        }
        euc
    }
}

/// Packs poles (and, if rational, weights) into a flat coordinate buffer:
/// `(x, y, z)` per pole if `weights` is `None`, else the homogeneous
/// `(x*w, y*w, z*w, w)`.
fn pack_flat_poles(poles: &[Point3D], weights: Option<&[f64]>) -> Vec<f64> {
    match weights {
        None => poles.iter().flat_map(|p| [p.x, p.y, p.z]).collect(),
        Some(ws) => poles
            .iter()
            .zip(ws)
            .flat_map(|(p, &w)| [p.x * w, p.y * w, p.z * w, w])
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_interpolate_line_points() {
        let tol = Tolerance::DEFAULT;
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(3.0, 0.0, 0.0),
        ];
        for param in [
            InterpParametrization::Chordal,
            InterpParametrization::Centripetal,
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
            BSplineCurve3D::interpolate(&points, 3, InterpParametrization::Centripetal).unwrap();
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
            BSplineCurve3D::interpolate(&one, 1, InterpParametrization::Chordal),
            Err(BSplineConstructionError::TooFewPoints)
        );
        let three = vec![
            Point3D::ORIGIN,
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(1.0, 1.0, 0.0),
        ];
        assert_eq!(
            BSplineCurve3D::interpolate(&three, 0, InterpParametrization::Chordal),
            Err(BSplineConstructionError::InvalidDegree)
        );
        assert_eq!(
            BSplineCurve3D::interpolate(&three, 3, InterpParametrization::Chordal),
            Err(BSplineConstructionError::DegreeTooHigh)
        );
        let dup = vec![
            Point3D::ORIGIN,
            Point3D::ORIGIN,
            Point3D::new(1.0, 0.0, 0.0),
        ];
        assert_eq!(
            BSplineCurve3D::interpolate(&dup, 1, InterpParametrization::Chordal),
            Err(BSplineConstructionError::ConfusedPoints)
        );
    }
}
