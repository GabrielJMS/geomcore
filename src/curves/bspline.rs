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
use crate::math::boxtree::{BoxTree, Lazy};
use crate::math::newton_min_1d;
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

/// Point parametrization for [`BSplineCurve3D::interpolate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpParametrization {
    /// Cumulative chord lengths (simplest; overshoots at sharp turns).
    Chordal,
    /// Square roots of chord lengths (tamer at sharp turns; default choice).
    Centripetal,
}

/// Normalized chord-like parameters for `points` (`exponent` 1.0 chordal,
/// 0.5 centripetal), or `ConfusedPoints` for coincident consecutive points.
fn chord_params(points: &[Point3D], exponent: f64) -> Result<Vec<f64>, BSplineConstructionError> {
    let n = points.len();
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
    Ok(params)
}

/// Full clamped knot vector for `m_poles` poles of `degree` from `params`:
/// parameter averages interior, ends clamped. Shared by interpolation
/// (`m_poles == params.len()`) and least-squares approximation.
fn averaged_full_knots(params: &[f64], degree: usize, m_poles: usize) -> Vec<f64> {
    let p = degree;
    let mut full = vec![0.0; p + 1];
    for j in 1..=(m_poles - p - 1) {
        let mut acc = 0.0;
        for k in 0..p {
            acc += params[j + k];
        }
        full.push(acc / p as f64);
    }
    full.extend(std::iter::repeat_n(1.0, p + 1));
    full
}

/// Compress a full knot vector into distinct values + multiplicities.
fn compress_knots(full: &[f64]) -> (Vec<f64>, Vec<u32>) {
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
    (knots, mults)
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
            spans: Lazy::default(),
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
        let exponent = match param {
            InterpParametrization::Chordal => 1.0,
            InterpParametrization::Centripetal => 0.5,
        };
        let params = chord_params(points, exponent)?;
        let p = degree;
        let full = averaged_full_knots(&params, p, n);
        let (knots, mults) = compress_knots(&full);
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

    /// Least-squares B-spline fit through `points` with `num_poles` poles.
    ///
    /// Same chord-like parametrization as
    /// [`BSplineCurve3D::interpolate`], but uniform interior knots and an
    /// overdetermined system: the normal equations `A'A x = A'b` solve by
    /// band LU (bandwidth `2*degree`). The fit approximates rather than
    /// hits the data; directions with no data in any basis support fail
    /// explicitly instead of producing NaNs.
    ///
    /// # Errors
    ///
    /// [`BSplineConstructionError::TooFewPoints`] for fewer than two
    /// points, [`BSplineConstructionError::InvalidDegree`] outside
    /// `1..=25`, [`BSplineConstructionError::DegreeTooHigh`] when
    /// `degree` exceeds poles minus one,
    /// [`BSplineConstructionError::TooManyPoles`] for more poles than
    /// points, [`BSplineConstructionError::ConfusedPoints`] for
    /// coincident consecutive points, or
    /// [`BSplineConstructionError::PoleCountMismatch`] when the data
    /// leaves a basis function unsupported.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, InterpParametrization, Point3D, Tolerance};
    /// let points: Vec<Point3D> = (0..9)
    ///     .map(|i| {
    ///         let u = i as f64 / 8.0 * std::f64::consts::TAU;
    ///         Point3D::new(2.0 * u.cos(), 2.0 * u.sin(), 0.0)
    ///     })
    ///     .collect();
    /// let curve =
    ///     BSplineCurve3D::approximate(&points, 3, 6, InterpParametrization::Centripetal)
    ///         .unwrap();
    /// assert_eq!(curve.degree(), 3);
    /// // Near-circular fit: radius stays within a few percent of 2.
    /// for i in 0..=40 {
    ///     let p = curve.eval_point(i as f64 / 40.0);
    ///     let r = (p.x * p.x + p.y * p.y).sqrt();
    ///     assert!((r - 2.0).abs() < 0.1, "{p:?}");
    /// }
    /// ```
    pub fn approximate(
        points: &[Point3D],
        degree: usize,
        num_poles: usize,
        param: InterpParametrization,
    ) -> Result<BSplineCurve3D, BSplineConstructionError> {
        let n = points.len();
        if n < 2 {
            return Err(BSplineConstructionError::TooFewPoints);
        }
        if !(1..=25).contains(&degree) {
            return Err(BSplineConstructionError::InvalidDegree);
        }
        if degree > num_poles.saturating_sub(1) {
            return Err(BSplineConstructionError::DegreeTooHigh);
        }
        if num_poles > n {
            return Err(BSplineConstructionError::TooManyPoles);
        }
        let exponent = match param {
            InterpParametrization::Chordal => 1.0,
            InterpParametrization::Centripetal => 0.5,
        };
        let params = chord_params(points, exponent)?;
        let p = degree;
        let m = num_poles;
        // Uniform interior knots (data-independent spreading); the
        // column-norm guard below rejects data too sparse to support them.
        let mut full = vec![0.0; p + 1];
        for j in 1..=(m - p - 1) {
            full.push(j as f64 / (m - p) as f64);
        }
        full.extend(std::iter::repeat_n(1.0, p + 1));
        let (knots, mults) = compress_knots(&full);
        // Normal equations over all data rows (bandwidth 2p).
        let bw = 2 * p;
        let mut band = vec![vec![0.0; 2 * bw + 1]; m];
        let mut rhs = vec![vec![0.0; m]; 3];
        for (j, &t) in params.iter().enumerate() {
            let basis = basis_values(&full, p, t);
            let coords = [points[j].x, points[j].y, points[j].z];
            for (i1, &b1) in basis.iter().enumerate() {
                if b1 == 0.0 {
                    continue;
                }
                for (i2, &b2) in basis.iter().enumerate() {
                    if b2 != 0.0 && i1 + bw >= i2 && i2 + bw >= i1 {
                        band[i1][bw + i2 - i1] += b1 * b2;
                    }
                }
                for k in 0..3 {
                    rhs[k][i1] += b1 * coords[k];
                }
            }
        }
        let max_diag = band.iter().map(|row| row[bw].abs()).fold(0.0f64, f64::max);
        if max_diag <= 0.0 || band.iter().any(|row| row[bw] <= 1e-12 * max_diag) {
            return Err(BSplineConstructionError::PoleCountMismatch);
        }
        banded_solve(&mut band, bw, &mut rhs);
        if rhs.iter().flatten().any(|v| !v.is_finite()) {
            return Err(BSplineConstructionError::PoleCountMismatch);
        }
        let poles: Vec<Point3D> = (0..m)
            .map(|i| Point3D::new(rhs[0][i], rhs[1][i], rhs[2][i]))
            .collect();
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
        let flat_poles = pack_flat_poles(&poles, self.weights.as_deref());
        BSplineCurve3D {
            degree: self.degree,
            periodic: self.periodic,
            poles,
            weights: self.weights.clone(),
            knots: self.knots.clone(),
            mults: self.mults.clone(),
            flat: self.flat.clone(),
            flat_poles,
            spans: Lazy::default(),
        }
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
        if !self.periodic {
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
        let t = if self.periodic && best.1 == last {
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
        let mut coordinates = [[0.0; math::LARGE]; 3];
        let mut weights = [1.0; math::LARGE];
        let dim = if self.is_rational() { 4 } else { 3 };
        {
            let [x, y, z] = &mut coordinates;
            let mut out: [&mut [f64]; 4] = [x, y, z, &mut weights];
            math::span_bezier(
                u,
                self.degree,
                self.periodic,
                &self.flat,
                &self.flat_poles,
                &mut out[..dim],
            );
        }
        for (k, point) in points.iter_mut().enumerate().take(self.degree + 1) {
            *point = [
                coordinates[0][k],
                coordinates[1][k],
                coordinates[2][k],
                weights[k],
            ];
        }
    }

    /// The knot-span box hierarchy, built on first use.
    fn span_tree(&self) -> &SpanTree {
        self.spans.get_or_init(|| {
            let (first, last) = self.bounds();
            let breaks = math::span_breaks(&self.knots, first, last);
            let dim = if self.is_rational() { 4 } else { 3 };
            let boxes = math::span_boxes(
                &breaks,
                self.degree,
                self.periodic,
                &self.flat,
                &self.flat_poles,
                dim,
            );
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
        if self.degree < math::SMALL {
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
        let p = self.degree;
        let rational = self.is_rational();
        let dim = if rational { 4 } else { 3 };
        let mut bezier = [[0.0; N]; 3];
        let mut weight = [1.0; N];
        {
            let [x, y, z] = &mut bezier;
            let mut out: [&mut [f64]; 4] = [x, y, z, &mut weight];
            math::span_bezier(
                0.5 * (a + b),
                p,
                self.periodic,
                &self.flat,
                &self.flat_poles,
                &mut out[..dim],
            );
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
            let e = self.eval_euclidean(math::inward(t, bounds), 2);
            let r = [e[0] - point.x, e[1] - point.y, e[2] - point.z];
            let dot = |a: &[f64], b: &[f64]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            (
                0.5 * dot(&r, &r),
                dot(&e[3..6], &r),
                dot(&e[3..6], &e[3..6]) + dot(&e[6..9], &r),
            )
        };
        let t = newton_min_1d(objective, t, bounds, 1e-14, 100);
        if self.periodic {
            let (first, last) = self.bounds();
            first + (t - first).rem_euclid(last - first)
        } else {
            t
        }
    }

    /// Evaluates the value and derivatives up to `n` (`n <= 2`) at `u`,
    /// returning Euclidean coordinates as `[f, f', f'']` flattened (9
    /// values, trailing ones zero if `n < 2`).
    fn eval_euclidean(&self, u: f64, n: usize) -> [f64; 9] {
        let dim = if self.is_rational() { 4 } else { 3 };
        let mut raw = [0.0f64; 3 * 4];
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
            euc.copy_from_slice(&raw[..9]);
        }
        euc
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
    use crate::{Axis3D, Frame3D};

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

    #[test]
    fn test_approximate_circle_fewer_poles() {
        let points: Vec<Point3D> = (0..12)
            .map(|i| {
                let u = i as f64 / 12.0 * std::f64::consts::TAU;
                Point3D::new(2.0 * u.cos(), 2.0 * u.sin(), 0.0)
            })
            .collect();
        let curve =
            BSplineCurve3D::approximate(&points, 3, 7, InterpParametrization::Centripetal).unwrap();
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
            BSplineCurve3D::approximate(&points, 1, 2, InterpParametrization::Chordal).unwrap();
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
            BSplineCurve3D::approximate(&one, 1, 2, InterpParametrization::Chordal),
            Err(BSplineConstructionError::TooFewPoints)
        );
        let four = vec![
            Point3D::ORIGIN,
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(1.0, 1.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
        ];
        assert_eq!(
            BSplineCurve3D::approximate(&four, 1, 5, InterpParametrization::Chordal),
            Err(BSplineConstructionError::TooManyPoles)
        );
        assert_eq!(
            BSplineCurve3D::approximate(&four, 4, 3, InterpParametrization::Chordal),
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
}
