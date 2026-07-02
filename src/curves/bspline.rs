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
use crate::surfaces::Surface;
use crate::{Point3, Vector3};

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
/// use geomcore::{BSplineCurve3D, Point3};
///
/// let poles = vec![
///     Point3::new(0.0, 0.0, 0.0),
///     Point3::new(1.0, 2.0, 0.0),
///     Point3::new(2.0, 2.0, 1.0),
///     Point3::new(3.0, 0.0, 1.0),
///     Point3::new(4.0, 1.0, 0.0),
///     Point3::new(5.0, 0.0, 0.0),
/// ];
/// let knots = vec![0.0, 1.0, 2.0, 3.0];
/// let mults = vec![4, 1, 1, 4];
/// let curve = BSplineCurve3D::new(3, poles, knots, mults, false).unwrap();
///
/// assert_eq!(curve.eval_point(0.0), Point3::new(0.0, 0.0, 0.0));
/// assert_eq!(curve.bounds(), (0.0, 3.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BSplineCurve3D {
    degree: usize,
    periodic: bool,
    poles: Vec<Point3>,
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
    /// use geomcore::{BSplineCurve3D, Point3};
    ///
    /// let poles = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert_eq!(curve.eval_point(0.5), Point3::new(1.0, 0.0, 0.0));
    /// ```
    pub fn new(
        degree: usize,
        poles: Vec<Point3>,
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
    /// use geomcore::{BSplineCurve3D, Point3};
    ///
    /// // Rational quadratic quarter circle.
    /// let w = std::f64::consts::FRAC_1_SQRT_2;
    /// let poles = vec![
    ///     Point3::new(1.0, 0.0, 0.0),
    ///     Point3::new(1.0, 1.0, 0.0),
    ///     Point3::new(0.0, 1.0, 0.0),
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
        poles: Vec<Point3>,
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
        poles: Vec<Point3>,
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
    /// use geomcore::{BSplineCurve3D, Point3};
    /// let poles = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert_eq!(curve.degree(), 1);
    /// ```
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Returns whether the curve is periodic (closed).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3};
    /// let poles = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0)];
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
    /// use geomcore::{BSplineCurve3D, Point3};
    /// let poles = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0)];
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
    /// use geomcore::{BSplineCurve3D, Point3};
    /// let poles = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles.clone(), vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert_eq!(curve.poles(), poles.as_slice());
    /// ```
    pub fn poles(&self) -> &[Point3] {
        &self.poles
    }

    /// Returns the per-pole weights, or `None` if the curve is not rational.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3};
    /// let poles = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0)];
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
    /// use geomcore::{BSplineCurve3D, Point3};
    /// let poles = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0)];
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
    /// use geomcore::{BSplineCurve3D, Point3};
    /// let poles = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0)];
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
    /// use geomcore::{BSplineCurve3D, Point3};
    /// let poles = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0)];
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
    /// use geomcore::{BSplineCurve3D, Point3};
    /// let poles = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert_eq!(curve.eval_point(0.5), Point3::new(1.0, 0.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64) -> Point3 {
        let euc = self.eval_euclidean(u, 0);
        Point3::new(euc[0], euc[1], euc[2])
    }

    /// Evaluates the points on the curve at each parameter in `us`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3};
    /// let poles = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// let points = curve.eval_points(&[0.0, 0.5, 1.0]);
    /// assert_eq!(points[1], Point3::new(1.0, 0.0, 0.0));
    /// ```
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point3> {
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
    /// use geomcore::{BSplineCurve3D, Point3, Vector3};
    /// let poles = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert_eq!(curve.eval_derivative(0.5, 1), Vector3::new(2.0, 0.0, 0.0));
    /// ```
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector3 {
        match order {
            0 => panic!("eval_derivative: order must be >= 1 (use eval_point for order 0)"),
            1 | 2 => {
                let euc = self.eval_euclidean(u, order as usize);
                let base = order as usize * 3;
                Vector3::new(euc[base], euc[base + 1], euc[base + 2])
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
    /// use geomcore::{BSplineCurve3D, Plane, Point3, Vector3};
    ///
    /// let plane = Plane::new(Point3::ORIGIN, Vector3::Z).unwrap();
    /// let poles = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0)];
    /// let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    /// assert_eq!(curve.parametrize_on(&plane), Err(ParametrizeError::NotAnalytic));
    /// ```
    pub fn parametrize_on(&self, surface: impl Into<Surface>) -> Result<Curve2D, ParametrizeError> {
        let _ = surface.into();
        Err(ParametrizeError::NotAnalytic)
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
fn pack_flat_poles(poles: &[Point3], weights: Option<&[f64]>) -> Vec<f64> {
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
        let poles = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0)];
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
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
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
        let poles = vec![Point3::ORIGIN];
        let err = BSplineCurve3D::new(0, poles, vec![0.0, 1.0], vec![1, 1], false).unwrap_err();
        assert_eq!(err, BSplineConstructionError::InvalidDegree);
    }

    #[test]
    fn test_new_pole_count_mismatch_errors() {
        let poles = vec![Point3::ORIGIN; 3];
        let err = BSplineCurve3D::new(3, poles, vec![0.0, 1.0, 2.0, 3.0], vec![4, 1, 1, 4], false)
            .unwrap_err();
        assert_eq!(err, BSplineConstructionError::PoleCountMismatch);
    }

    #[test]
    fn test_new_rational_weight_count_mismatch_errors() {
        let poles = vec![Point3::ORIGIN, Point3::new(1.0, 0.0, 0.0)];
        let err =
            BSplineCurve3D::new_rational(1, poles, vec![1.0], vec![0.0, 1.0], vec![2, 2], false)
                .unwrap_err();
        assert_eq!(err, BSplineConstructionError::WeightCountMismatch);
    }

    #[test]
    fn test_new_rational_non_positive_weight_errors() {
        let poles = vec![Point3::ORIGIN, Point3::new(1.0, 0.0, 0.0)];
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

        let poles = vec![Point3::ORIGIN, Point3::new(1.0, 0.0, 0.0)];
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
        let poles = vec![Point3::ORIGIN];
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
        let poles = vec![Point3::ORIGIN, Point3::new(2.0, 0.0, 0.0)];
        let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
        assert_eq!(curve.eval_point(0.5), Point3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_eval_points_matches_mapped_eval_point() {
        let poles = vec![Point3::ORIGIN, Point3::new(2.0, 0.0, 0.0)];
        let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
        let us = [0.0, 0.3, 0.5, 1.0];
        let expected: Vec<Point3> = us.iter().map(|&u| curve.eval_point(u)).collect();
        assert_eq!(curve.eval_points(&us), expected);
    }

    #[test]
    fn test_eval_derivative_order1_constant_tangent() {
        let poles = vec![Point3::ORIGIN, Point3::new(2.0, 0.0, 0.0)];
        let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
        assert_eq!(curve.eval_derivative(0.5, 1), Vector3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn test_eval_derivative_order2_of_line_is_zero() {
        let poles = vec![Point3::ORIGIN, Point3::new(2.0, 0.0, 0.0)];
        let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
        assert_eq!(curve.eval_derivative(0.5, 2), Vector3::ZERO);
    }

    #[test]
    #[should_panic(expected = "order must be >= 1")]
    fn test_eval_derivative_order0_panics() {
        let poles = vec![Point3::ORIGIN, Point3::new(2.0, 0.0, 0.0)];
        let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
        curve.eval_derivative(0.5, 0);
    }

    #[test]
    #[should_panic(expected = "only first and second derivatives are supported")]
    fn test_eval_derivative_order3_panics_with_clear_message() {
        let poles = vec![Point3::ORIGIN, Point3::new(2.0, 0.0, 0.0)];
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
            Point3::new(2.0, 0.0, 0.3),
            Point3::new(1.0000000000000002, 1.7320508075688772, -0.3),
            Point3::new(-0.9999999999999996, 1.7320508075688774, 0.3),
            Point3::new(-2.0, 2.4492935982947064e-16, -0.3),
            Point3::new(-1.0000000000000009, -1.7320508075688767, 0.3),
            Point3::new(1.0000000000000002, -1.7320508075688772, -0.3),
        ];
        let knots = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mults = vec![1; 7];
        BSplineCurve3D::new(3, poles, knots, mults, true).unwrap()
    }
}
