//! Internal B-spline curve evaluation: de Boor recursion with derivatives,
//! rational (NURBS) support, and periodic (closed) curves.
//!
//! Poles are handled as flat `&[f64]` buffers with a `dim` coordinate stride
//! (`dim = 3` for ordinary curves, `dim = 4` for the homogeneous coordinates
//! `(x*w, y*w, z*w, w)` of a rational curve). Values and derivatives are
//! written into caller-provided output slices laid out the same way.
//!
//! The evaluation strategy is the canonical de Boor triangular scheme on a
//! *flat* (expanded) knot sequence. Derivatives are obtained by forming the
//! poles of the derivative curve via divided differences and recursing at one
//! lower degree; rational derivatives apply the quotient rule to homogeneous
//! derivatives. Periodic curves are evaluated by tiling one period's worth of
//! flat knots into three-plus period-shifted copies and evaluating in the
//! middle copy, wrapping pole indices modulo the pole count.

use super::analytic::in_period;

/// Maximum supported B-spline degree.
const MAX_DEGREE: usize = 25;

/// Error type re-exported here for use by [`validate_direction`]; the
/// canonical definition (and its docs) live on [`crate::curves`], next to
/// [`crate::curves::BSplineCurve3D`], the type that actually constructs and
/// reports these errors.
pub(crate) use crate::curves::BSplineConstructionError;

/// Builds the flattened knot sequence: knot `i` repeated `mults[i]` times.
///
/// This is plain repetition with no padding; e.g. `flat_knots(&[0, 1, 2],
/// &[3, 1, 3]) == [0, 0, 0, 1, 2, 2, 2]`.
pub(crate) fn flat_knots(knots: &[f64], mults: &[u32]) -> Vec<f64> {
    let mut out = Vec::with_capacity(mults.iter().map(|&m| m as usize).sum());
    for (&k, &m) in knots.iter().zip(mults.iter()) {
        for _ in 0..m {
            out.push(k);
        }
    }
    out
}

/// The active parameter range `(first, last)` for evaluation.
///
/// Non-periodic: the clamped span `[flat[degree], flat[len - 1 - degree]]`.
/// Periodic: `[flat[0], flat[len - 1]]` on the *plain* flat knots, whose width
/// is the period; here `flat` is expected to be [`flat_knots`] of the full
/// (unexpanded) knots/mults, and `sum_mults == flat.len()`.
pub(crate) fn param_range(
    flat: &[f64],
    degree: usize,
    periodic: bool,
    sum_mults: usize,
) -> (f64, f64) {
    debug_assert_eq!(flat.len(), sum_mults);
    if periodic {
        (flat[0], flat[flat.len() - 1])
    } else {
        (flat[degree], flat[flat.len() - 1 - degree])
    }
}

/// Locates the span index `s` with `flat[s] <= u < flat[s+1]` in the active
/// region, returning `(span, u_clamped_or_wrapped)`.
///
/// For non-periodic curves `u` is clamped into the active range; for periodic
/// curves `u` is wrapped via [`in_period`]. Ties at a knot resolve to the left
/// span except at the range start, and the span is nudged so it has nonzero
/// width and a full window of `degree + 1` poles exists.
pub(crate) fn locate_span(flat: &[f64], degree: usize, periodic: bool, u: f64) -> (usize, f64) {
    let lo = degree;
    let hi = flat.len() - 1 - degree;
    let first = flat[lo];
    let last = flat[hi];
    let u = if periodic {
        in_period(u, first, last)
    } else {
        u.clamp(first, last)
    };

    // Binary search for the rightmost lo <= s < hi with flat[s] <= u.
    let mut span = lo;
    let (mut left, mut right) = (lo, hi);
    while left < right {
        let mid = left + (right - left) / 2;
        if flat[mid] <= u {
            span = mid;
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    // Resolve knot ties to the left span, but never below the range start.
    while span > lo && flat[span] == flat[span - 1] {
        span -= 1;
    }
    // Guarantee a nonzero-width span (flat[span] < flat[span+1]).
    while span < hi - 1 && flat[span] >= flat[span + 1] {
        span += 1;
    }
    (span, u)
}

/// de Boor triangular recursion at `span` on flat knots `flat`, writing the
/// point into `out[..dim]`.
///
/// `poles` is a flat buffer with `dim` coordinates per pole; the window
/// `poles[(span - degree)..=span]` is used (all indices must be in range —
/// the caller guarantees this for non-periodic curves, and pre-tiles the pole
/// buffer for periodic ones).
fn deboor(
    u: f64,
    degree: usize,
    flat: &[f64],
    poles: &[f64],
    dim: usize,
    span: usize,
    out: &mut [f64],
) {
    // Local pole copies d_0..d_degree.
    let mut d = vec![0.0f64; (degree + 1) * dim];
    for i in 0..=degree {
        let idx = span - degree + i;
        d[i * dim..(i + 1) * dim].copy_from_slice(&poles[idx * dim..(idx + 1) * dim]);
    }
    for r in 1..=degree {
        for i in (r..=degree).rev() {
            let j = span - degree + i;
            let alpha = (u - flat[j]) / (flat[j + degree - r + 1] - flat[j]);
            for c in 0..dim {
                let lo = d[(i - 1) * dim + c];
                let hi = d[i * dim + c];
                d[i * dim + c] = (1.0 - alpha) * lo + alpha * hi;
            }
        }
    }
    out[..dim].copy_from_slice(&d[degree * dim..degree * dim + dim]);
}

/// Poles of the derivative curve via divided differences:
/// `Q_i = degree * (P_{i+1} - P_i) / (flat[i + degree + 1] - flat[i + 1])`.
///
/// Returns the new pole buffer (one fewer pole) and the new flat-knot array
/// (first and last knots dropped), i.e. the degree-`(degree - 1)` curve.
fn derivative_curve(
    degree: usize,
    flat: &[f64],
    poles: &[f64],
    dim: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = poles.len() / dim;
    let mut q = vec![0.0f64; (n - 1) * dim];
    for i in 0..n - 1 {
        let denom = flat[i + degree + 1] - flat[i + 1];
        if denom != 0.0 {
            let scale = degree as f64 / denom;
            for c in 0..dim {
                q[i * dim + c] = scale * (poles[(i + 1) * dim + c] - poles[i * dim + c]);
            }
        }
        // denom == 0 leaves Q_i at zero (repeated knots contribute nothing).
    }
    let new_flat = flat[1..flat.len() - 1].to_vec();
    (q, new_flat)
}

/// Evaluates the curve value into `out[..dim]` (see module docs for layout).
///
/// Only exercised directly by this module's own tests today: the public
/// [`crate::curves::BSplineCurve3D`] always calls [`eval_dn`] with `n`
/// derived from the requested derivative order (0 for `eval_point`), rather
/// than special-casing `n == 0`. Kept `pub(crate)` for API symmetry with
/// [`eval_dn`] and as the natural entry point for position-only evaluation.
#[allow(dead_code)]
pub(crate) fn eval_d0(
    u: f64,
    degree: usize,
    periodic: bool,
    flat: &[f64],
    poles: &[f64],
    dim: usize,
    out: &mut [f64],
) {
    eval_dn(u, degree, periodic, flat, poles, dim, 0, out);
}

/// Evaluates the value and first `n` derivatives, writing `f, f', ..., f^n`
/// consecutively into `out`, each occupying `dim` coordinates
/// (so `out` must hold `(n + 1) * dim` values).
///
/// `flat` is the plain [`flat_knots`] of the full knots/mults. Non-periodic
/// evaluation uses it directly; periodic evaluation ignores it and rebuilds a
/// tiled flat-knot sequence internally, so callers pass the same `flat` in
/// both cases. For periodic curves the caller must also pass the *plain* flat
/// knots — the period is derived from its endpoints.
#[allow(clippy::too_many_arguments)] // dim-generic flat-buffer evaluation signature
pub(crate) fn eval_dn(
    u: f64,
    degree: usize,
    periodic: bool,
    flat: &[f64],
    poles: &[f64],
    dim: usize,
    n: usize,
    out: &mut [f64],
) {
    if periodic {
        eval_dn_periodic(u, degree, flat, poles, dim, n, out);
    } else {
        eval_dn_nonperiodic(u, degree, flat, poles, dim, n, out);
    }
}

/// Non-periodic value + derivatives on the plain flat knots.
fn eval_dn_nonperiodic(
    u: f64,
    degree: usize,
    flat: &[f64],
    poles: &[f64],
    dim: usize,
    n: usize,
    out: &mut [f64],
) {
    let (span, uu) = locate_span(flat, degree, false, u);

    let mut cur_poles = poles.to_vec();
    let mut cur_flat = flat.to_vec();
    let mut cur_degree = degree;
    for k in 0..=n {
        let slot = &mut out[k * dim..(k + 1) * dim];
        if k > degree {
            // Derivatives beyond the degree of a polynomial curve vanish.
            slot.iter_mut().for_each(|c| *c = 0.0);
            continue;
        }
        // Dropping k front knots shifts the span index left by k.
        let sp = span - k;
        deboor(uu, cur_degree, &cur_flat, &cur_poles, dim, sp, slot);
        if cur_degree >= 1 && k < n {
            let (q, nf) = derivative_curve(cur_degree, &cur_flat, &cur_poles, dim);
            cur_poles = q;
            cur_flat = nf;
            cur_degree -= 1;
        }
    }
}

/// Periodic value + derivatives.
///
/// Builds one period's flat-knot *block* (the flat knots of all but the last
/// knot, so the seam knot is not duplicated), tiles it into four period-shifted
/// copies (offsets `-P, 0, +P, +2P`), and evaluates in the middle copy. Poles
/// are tiled to align with the extended knots; the alignment offset is
/// `-degree` so the first middle-copy window starts at pole 0's window.
fn eval_dn_periodic(
    u: f64,
    degree: usize,
    flat: &[f64],
    poles: &[f64],
    dim: usize,
    n: usize,
    out: &mut [f64],
) {
    let n_poles = poles.len() / dim;
    let first = flat[0];
    let last = flat[flat.len() - 1];
    let period = last - first;
    let uw = in_period(u, first, last);

    // Period block: flat knots of all knots except the last (the seam).
    // `flat` is a plain repetition, so the block is `flat` with its trailing
    // run of the last knot value removed (so the seam is not duplicated when
    // period-shifted copies are concatenated).
    let mut block_len = flat.len();
    while block_len > 0 && flat[block_len - 1] == last {
        block_len -= 1;
    }
    let block = &flat[..block_len];

    // Extended flat knots: four period-shifted copies.
    let mut ext_flat = Vec::with_capacity(block_len * 4);
    for shift in [-period, 0.0, period, 2.0 * period] {
        ext_flat.extend(block.iter().map(|&k| k + shift));
    }
    let mid_off = block_len; // start of the middle copy

    // Locate span within the middle copy.
    let mut span = mid_off + block_len - 1;
    for s in mid_off..mid_off + block_len {
        if ext_flat[s] <= uw && uw < ext_flat[s + 1] {
            span = s;
            break;
        }
    }

    // Extended pole buffer aligned with ext_flat, wrapping modulo n_poles.
    let n_ext = ext_flat.len() - degree - 1;
    let mut ext_poles = vec![0.0f64; n_ext * dim];
    for k in 0..n_ext {
        let logical = k as isize - mid_off as isize - degree as isize;
        let pidx = logical.rem_euclid(n_poles as isize) as usize;
        ext_poles[k * dim..(k + 1) * dim].copy_from_slice(&poles[pidx * dim..(pidx + 1) * dim]);
    }

    let mut cur_poles = ext_poles;
    let mut cur_flat = ext_flat;
    let mut cur_degree = degree;
    for k in 0..=n {
        let slot = &mut out[k * dim..(k + 1) * dim];
        if k > degree {
            slot.iter_mut().for_each(|c| *c = 0.0);
            continue;
        }
        let sp = span - k;
        deboor(uw, cur_degree, &cur_flat, &cur_poles, dim, sp, slot);
        if cur_degree >= 1 && k < n {
            let (q, nf) = derivative_curve(cur_degree, &cur_flat, &cur_poles, dim);
            cur_poles = q;
            cur_flat = nf;
            cur_degree -= 1;
        }
    }
}

/// Applies the rational quotient rule to homogeneous value+derivatives.
///
/// `homog` holds `H, H', ..., H^n` each as `dim = 4` coordinates
/// `(x, y, z, w)`; `out` receives the euclidean `f, f', ..., f^n` each as
/// `dim = 3` coordinates: `f = H/w`, `f' = (H' - f·w')/w`,
/// `f'' = (H'' - 2·f'·w' - f·w'')/w`. Only `n <= 2` is supported.
pub(crate) fn rational_derivatives(homog: &[f64], n: usize, out: &mut [f64]) {
    debug_assert!(
        n <= 2,
        "rational derivatives only implemented up to order 2"
    );
    let w0 = homog[3];
    // f = H / w
    for c in 0..3 {
        out[c] = homog[c] / w0;
    }
    if n >= 1 {
        let w1 = homog[4 + 3];
        for c in 0..3 {
            out[3 + c] = (homog[4 + c] - out[c] * w1) / w0;
        }
    }
    if n >= 2 {
        let w1 = homog[4 + 3];
        let w2 = homog[8 + 3];
        for c in 0..3 {
            out[6 + c] = (homog[8 + c] - 2.0 * out[3 + c] * w1 - out[c] * w2) / w0;
        }
    }
}

/// Validates a B-spline direction (shared by curves and surface directions).
///
/// Checks degree bounds, distinct knot count, strict monotonicity,
/// multiplicity bounds, periodic end-multiplicity match, and the pole-count
/// relation. Does *not* validate weights (see [`BSplineConstructionError`]).
pub(crate) fn validate_direction(
    degree: usize,
    n_poles: usize,
    knots: &[f64],
    mults: &[u32],
    periodic: bool,
) -> Result<(), BSplineConstructionError> {
    if !(1..=MAX_DEGREE).contains(&degree) {
        return Err(BSplineConstructionError::InvalidDegree);
    }
    if knots.len() < 2 || mults.len() < 2 || knots.len() != mults.len() {
        return Err(BSplineConstructionError::TooFewKnots);
    }
    for pair in knots.windows(2) {
        if pair[1] <= pair[0] {
            return Err(BSplineConstructionError::KnotsNotIncreasing);
        }
    }

    let last = mults.len() - 1;
    let deg = degree as u32;
    if periodic {
        if mults[0] != mults[last] {
            return Err(BSplineConstructionError::PeriodicEndMultiplicityMismatch);
        }
        if mults.iter().any(|&m| m > deg) {
            return Err(BSplineConstructionError::MultiplicityTooLarge);
        }
    } else {
        if mults[0] > deg + 1 || mults[last] > deg + 1 {
            return Err(BSplineConstructionError::MultiplicityTooLarge);
        }
        if mults[1..last].iter().any(|&m| m > deg) {
            return Err(BSplineConstructionError::MultiplicityTooLarge);
        }
    }

    let sum_mults: u32 = mults.iter().sum();
    let expected = if periodic {
        sum_mults as i64 - mults[last] as i64
    } else {
        sum_mults as i64 - degree as i64 - 1
    };
    if expected < 0 || n_poles as i64 != expected {
        return Err(BSplineConstructionError::PoleCountMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- flat_knots ----

    #[test]
    fn test_flat_knots_repeats() {
        assert_eq!(
            flat_knots(&[0.0, 1.0, 2.0], &[3, 1, 3]),
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]
        );
    }

    #[test]
    fn test_flat_knots_all_single() {
        assert_eq!(
            flat_knots(&[0.0, 1.0, 2.0], &[1, 1, 1]),
            vec![0.0, 1.0, 2.0]
        );
    }

    // ---- degree-1 line ----

    #[test]
    fn test_degree1_line_midpoint_and_derivative() {
        // Two poles (0,0,0) and (2,0,0); knots [0,1] mults [2,2] => a straight
        // segment. At u = 0.5 the point is the midpoint (1,0,0); the first
        // derivative is the constant tangent (2,0,0).
        let poles = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let flat = flat_knots(&[0.0, 1.0], &[2, 2]);
        let mut out = [0.0f64; 6];
        eval_dn(0.5, 1, false, &flat, &poles, 3, 1, &mut out);
        assert_close(&out[0..3], &[1.0, 0.0, 0.0]);
        assert_close(&out[3..6], &[2.0, 0.0, 0.0]);
    }

    // ---- degree-2 de Casteljau by hand ----

    #[test]
    fn test_degree2_bezier_by_hand() {
        // Poles (0,0,0),(1,2,0),(2,0,0); knots [0,1] mults [3,3] => a Bezier
        // parabola. de Casteljau at t = 0.5:
        //   lerp(P0,P1)=(0.5,1,0), lerp(P1,P2)=(1.5,1,0),
        //   lerp of those = (1,1,0).
        // Derivative curve poles: 2*(P1-P0)=(2,4,0), 2*(P2-P1)=(2,-4,0);
        //   at t=0.5 the tangent is their midpoint (2,0,0).
        let poles = [0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 0.0];
        let flat = flat_knots(&[0.0, 1.0], &[3, 3]);
        let mut out = [0.0f64; 9];
        eval_dn(0.5, 2, false, &flat, &poles, 3, 2, &mut out);
        assert_close(&out[0..3], &[1.0, 1.0, 0.0]);
        assert_close(&out[3..6], &[2.0, 0.0, 0.0]);
    }

    // ---- rational quarter circle ----

    #[test]
    fn test_rational_quarter_circle_on_unit_circle() {
        // Standard rational quadratic quarter circle: poles (1,0),(1,1),(0,1),
        // weights [1, sqrt(2)/2, 1], degree 2, knots [0,1] mults [3,3].
        let w = std::f64::consts::FRAC_1_SQRT_2; // sqrt(2)/2
        // Homogeneous poles (x*w, y*w, z*w, w).
        let homog_poles = [
            1.0,
            0.0,
            0.0,
            1.0, // P0 weight 1
            1.0 * w,
            1.0 * w,
            0.0,
            w, // P1 weight sqrt2/2
            0.0,
            1.0,
            0.0,
            1.0, // P2 weight 1
        ];
        let flat = flat_knots(&[0.0, 1.0], &[3, 3]);
        for t in [0.0, 0.25, 0.5, 1.0] {
            let mut h = [0.0f64; 4];
            eval_d0(t, 2, false, &flat, &homog_poles, 4, &mut h);
            let mut euc = [0.0f64; 3];
            rational_derivatives(&h, 0, &mut euc);
            let r = (euc[0] * euc[0] + euc[1] * euc[1]).sqrt();
            assert!((r - 1.0).abs() < 1e-12, "t={t} radius={r}");
        }
        // Endpoint t = 1 lands exactly on (0,1,0).
        let mut h = [0.0f64; 4];
        eval_d0(1.0, 2, false, &flat, &homog_poles, 4, &mut h);
        let mut euc = [0.0f64; 3];
        rational_derivatives(&h, 0, &mut euc);
        assert_close(&euc, &[0.0, 1.0, 0.0]);
    }

    // ---- periodic wrap smoke ----

    #[test]
    fn test_periodic_value_repeats_over_period() {
        // Fixture case (c) poles (hexagon ring), degree 3, knots [0..6]
        // mults all 1, period 6. Evaluating at u and u + period must give the
        // same point (and derivatives).
        let poles = periodic_ring_poles();
        let flat = flat_knots(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 1, 1, 1, 1, 1]);
        for u in [0.3, 1.7, 4.9] {
            let mut a = [0.0f64; 9];
            let mut b = [0.0f64; 9];
            eval_dn(u, 3, true, &flat, &poles, 3, 2, &mut a);
            eval_dn(u + 6.0, 3, true, &flat, &poles, 3, 2, &mut b);
            assert_close(&a, &b);
        }
    }

    fn periodic_ring_poles() -> Vec<f64> {
        vec![
            2.0,
            0.0,
            0.3, //
            1.0000000000000002,
            1.7320508075688772,
            -0.3, //
            -0.9999999999999996,
            1.7320508075688774,
            0.3, //
            -2.0,
            2.4492935982947064e-16,
            -0.3, //
            -1.0000000000000009,
            -1.7320508075688767,
            0.3, //
            1.0000000000000002,
            -1.7320508075688772,
            -0.3, //
        ]
    }

    // ---- validation: every error variant ----

    #[test]
    fn test_validation_ok_nonperiodic() {
        // clamped cubic, 6 poles, knots [0,1,2,3] mults [4,1,1,4].
        assert_eq!(
            validate_direction(3, 6, &[0.0, 1.0, 2.0, 3.0], &[4, 1, 1, 4], false),
            Ok(())
        );
    }

    #[test]
    fn test_validation_ok_periodic() {
        assert_eq!(
            validate_direction(3, 6, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1; 7], true),
            Ok(())
        );
    }

    #[test]
    fn test_validation_invalid_degree_low() {
        assert_eq!(
            validate_direction(0, 1, &[0.0, 1.0], &[1, 1], false),
            Err(BSplineConstructionError::InvalidDegree)
        );
    }

    #[test]
    fn test_validation_invalid_degree_high() {
        assert_eq!(
            validate_direction(26, 1, &[0.0, 1.0], &[1, 1], false),
            Err(BSplineConstructionError::InvalidDegree)
        );
    }

    #[test]
    fn test_validation_too_few_knots() {
        assert_eq!(
            validate_direction(1, 1, &[0.0], &[2], false),
            Err(BSplineConstructionError::TooFewKnots)
        );
    }

    #[test]
    fn test_validation_knots_not_increasing() {
        assert_eq!(
            validate_direction(1, 2, &[1.0, 1.0], &[2, 2], false),
            Err(BSplineConstructionError::KnotsNotIncreasing)
        );
    }

    #[test]
    fn test_validation_multiplicity_too_large_interior() {
        // Interior mult 3 > degree 2.
        assert_eq!(
            validate_direction(2, 3, &[0.0, 1.0, 2.0], &[3, 3, 3], false),
            Err(BSplineConstructionError::MultiplicityTooLarge)
        );
    }

    #[test]
    fn test_validation_multiplicity_too_large_end() {
        // End mult 5 > degree + 1 = 4 for degree 3.
        assert_eq!(
            validate_direction(3, 3, &[0.0, 1.0], &[5, 3], false),
            Err(BSplineConstructionError::MultiplicityTooLarge)
        );
    }

    #[test]
    fn test_validation_periodic_mult_too_large() {
        // Periodic: any mult > degree is invalid.
        assert_eq!(
            validate_direction(2, 4, &[0.0, 1.0, 2.0, 3.0], &[3, 1, 1, 3], true),
            Err(BSplineConstructionError::MultiplicityTooLarge)
        );
    }

    #[test]
    fn test_validation_periodic_end_mismatch() {
        assert_eq!(
            validate_direction(3, 6, &[0.0, 1.0, 2.0, 3.0], &[2, 1, 1, 3], true),
            Err(BSplineConstructionError::PeriodicEndMultiplicityMismatch)
        );
    }

    #[test]
    fn test_validation_pole_count_mismatch_nonperiodic() {
        // sum(mults) - degree - 1 = 10 - 3 - 1 = 6; give 5.
        assert_eq!(
            validate_direction(3, 5, &[0.0, 1.0, 2.0, 3.0], &[4, 1, 1, 4], false),
            Err(BSplineConstructionError::PoleCountMismatch)
        );
    }

    #[test]
    fn test_validation_pole_count_mismatch_periodic() {
        // sum(mults) - mults[last] = 7 - 1 = 6; give 7.
        assert_eq!(
            validate_direction(3, 7, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1; 7], true),
            Err(BSplineConstructionError::PoleCountMismatch)
        );
    }

    // The weight variants are not raised by validate_direction; assert they
    // exist and format, documenting their intended use by the constructing
    // type.
    #[test]
    fn test_weight_error_variants_display() {
        assert!(
            !BSplineConstructionError::WeightCountMismatch
                .to_string()
                .is_empty()
        );
        assert!(
            !BSplineConstructionError::NonPositiveWeight
                .to_string()
                .is_empty()
        );
    }

    // ---- fixture replay: the correctness gate ----

    #[test]
    fn test_fixture_replay_all_cases() {
        replay_fixture();
    }

    // ---- helpers ----

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let tol = 1e-9 * f64::max(1.0, e.abs());
            assert!((a - e).abs() <= tol, "index {i}: {a} vs {e}");
        }
    }

    // --- fixture parsing (minimal, no serde derive to keep it explicit) ---

    use serde_json::Value;

    fn replay_fixture() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/curves_bspline.json"
        );
        let text = std::fs::read_to_string(path).expect("read fixture");
        let json: Value = serde_json::from_str(&text).expect("parse fixture");
        let cases = json["cases"].as_array().expect("cases array");
        for case in cases {
            replay_case(case);
        }
    }

    fn arr_f64(v: &Value) -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    }

    fn replay_case(case: &Value) {
        let name = case["name"].as_str().unwrap();
        let degree = case["degree"].as_u64().unwrap() as usize;
        let periodic = case["periodic"].as_bool().unwrap();
        let knots = arr_f64(&case["knots"]);
        let mults: Vec<u32> = case["mults"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let pole_rows: Vec<Vec<f64>> = case["poles"]
            .as_array()
            .unwrap()
            .iter()
            .map(arr_f64)
            .collect();
        let weights: Option<Vec<f64>> = case
            .get("weights")
            .and_then(|w| if w.is_null() { None } else { Some(arr_f64(w)) });

        let flat = flat_knots(&knots, &mults);
        let rational = weights.is_some();
        let dim = if rational { 4 } else { 3 };

        // Build the flat pole buffer (homogeneous if rational).
        let mut poles = Vec::with_capacity(pole_rows.len() * dim);
        for (i, p) in pole_rows.iter().enumerate() {
            if let Some(ws) = &weights {
                let w = ws[i];
                poles.extend_from_slice(&[p[0] * w, p[1] * w, p[2] * w, w]);
            } else {
                poles.extend_from_slice(&p[0..3]);
            }
        }

        for sample in case["samples"].as_array().unwrap() {
            let u = sample["u"].as_f64().unwrap();
            let mut raw = vec![0.0f64; 3 * dim];
            eval_dn(u, degree, periodic, &flat, &poles, dim, 2, &mut raw);
            let euc = if rational {
                let mut e = [0.0f64; 9];
                rational_derivatives(&raw, 2, &mut e);
                e.to_vec()
            } else {
                raw
            };
            for (key, idx) in [("point", 0usize), ("d1", 1), ("d2", 2)] {
                let expected = arr_f64(&sample[key]);
                let got = &euc[idx * 3..idx * 3 + 3];
                for (c, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
                    let tol = 1e-7 * f64::max(1.0, e.abs());
                    assert!(
                        (g - e).abs() <= tol,
                        "{name} u={u} {key}[{c}]: got {g}, expected {e} (tol {tol})"
                    );
                }
            }
        }
    }
}
