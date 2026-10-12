//! Internal B-spline curve evaluation: basis functions with derivatives,
//! rational (NURBS) support, and periodic (closed) curves.
//!
//! Poles are handled as flat `&[f64]` buffers with a `dim` coordinate stride
//! (`dim = 3` for ordinary curves, `dim = 4` for the homogeneous coordinates
//! `(x*w, y*w, z*w, w)` of a rational curve). Values and derivatives are
//! written into caller-provided output slices laid out the same way.
//!
//! Evaluation computes the `degree + 1` non-zero basis functions and their
//! derivatives at the parameter (The NURBS Book, A2.3) and sums the poles
//! they weight, so its cost depends on the degree only, never on the pole
//! count, and nothing is allocated. Rational derivatives apply the quotient
//! rule to homogeneous derivatives. Periodic curves see one period's flat
//! knots tiled endlessly, each copy shifted by the period, with pole indices
//! wrapping modulo the pole count; tiled knots are computed on demand.

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
/// curves `u` is wrapped via [`in_period`]. A `u` exactly on an interior
/// knot belongs to the span on its right, and the range end to the last nonzero-width
/// span; this only matters for derivatives across a knot where they are
/// discontinuous.
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
    // The rightmost such `s` already opens a nonzero-width span, except at
    // the range end (`u == last`), which steps back to the last real span.
    while span > lo && flat[span] >= flat[span + 1] {
        span -= 1;
    }
    (span, u)
}

/// Highest derivative order the evaluator computes.
pub(crate) const MAX_DERIVATIVE: usize = 2;

/// Basis-table width for degrees below 8: covers every curve and surface
/// seen in practice, and zeroing an 8-wide table is far cheaper than a
/// `MAX_DEGREE + 1`-wide one.
pub(crate) const SMALL: usize = 8;

/// Basis-table width for every supported degree.
pub(crate) const LARGE: usize = MAX_DEGREE + 1;

/// The `degree + 1` basis functions that are non-zero at a parameter, with
/// their derivatives, and the poles they weight.
pub(crate) struct Window<const N: usize> {
    /// `ders[k][j]`: `k`-th derivative of the `j`-th non-zero basis function.
    pub(crate) ders: [[f64; N]; MAX_DERIVATIVE + 1],
    /// Index of the pole weighted by the first non-zero basis function.
    first: usize,
    /// Pole count, by which indices wrap in a periodic direction.
    n_poles: usize,
    /// Whether pole indices wrap.
    periodic: bool,
}

impl<const N: usize> Window<N> {
    /// An empty window, filled by [`window`].
    pub(crate) fn new() -> Self {
        Window {
            ders: [[0.0; N]; MAX_DERIVATIVE + 1],
            first: 0,
            n_poles: 0,
            periodic: false,
        }
    }

    /// Index of the pole weighted by the `j`-th non-zero basis function,
    /// wrapped modulo the pole count in a periodic direction.
    #[inline]
    pub(crate) fn pole(&self, j: usize) -> usize {
        let index = self.first + j;
        if self.periodic && index >= self.n_poles {
            index % self.n_poles
        } else {
            index
        }
    }
}

/// Locates `u` and evaluates into `w` the non-zero basis functions there,
/// with derivatives up to `order`, touching only the `2 * degree` knots
/// around the span.
///
/// Non-periodic directions use [`locate_span`] on the plain flat knots.
/// Periodic directions behave as if one period's flat knots (all but the
/// trailing seam run) were tiled endlessly, each copy shifted by the period,
/// with poles wrapping modulo `n_poles`; the tiled knot at any index is
/// computed on demand, so nothing is allocated.
pub(crate) fn window<const N: usize>(
    u: f64,
    degree: usize,
    periodic: bool,
    flat: &[f64],
    n_poles: usize,
    order: usize,
    w: &mut Window<N>,
) {
    let mut left = [0.0f64; N];
    let mut right = [0.0f64; N];
    w.n_poles = n_poles;
    w.periodic = periodic;
    w.first = span_knots(u, degree, periodic, flat, n_poles, &mut left, &mut right);
    match order {
        0 => basis_values(degree, &left, &right, &mut w.ders[0]),
        1 => basis_first(degree, &left, &right, &mut w.ders),
        _ => basis_derivatives(degree, &left, &right, order, &mut w.ders),
    }
}

/// Locates `u` and writes the knots around its span relative to it,
/// `left[j] = u - U[span + 1 - j]` and `right[j] = U[span + j] - u` for `j`
/// in `1..=degree`; returns the index of the span's first pole (see
/// [`window`] for the periodic tiling).
fn span_knots<const N: usize>(
    u: f64,
    degree: usize,
    periodic: bool,
    flat: &[f64],
    n_poles: usize,
    left: &mut [f64; N],
    right: &mut [f64; N],
) -> usize {
    debug_assert!(degree < N, "basis table too narrow for degree {degree}");
    if periodic {
        let first = flat[0];
        let last = flat[flat.len() - 1];
        let period = last - first;
        // Exactly `last` closes the last span instead of wrapping
        // to the first (they differ only in derivatives across the seam).
        let uw = if u == last {
            last
        } else {
            in_period(u, first, last)
        };
        let mut block_len = flat.len();
        while block_len > 0 && flat[block_len - 1] == last {
            block_len -= 1;
        }
        let block = &flat[..block_len];
        // Rightmost knot of the period block at or below `uw`; the next tiled
        // knot is then strictly above it, so the span has nonzero width.
        let local = block
            .partition_point(|&k| k <= uw)
            .saturating_sub(1)
            .min(block_len - 1);
        let len = block_len as isize;
        let knot = |offset: isize| {
            let k = local as isize + offset;
            block[k.rem_euclid(len) as usize] + period * k.div_euclid(len) as f64
        };
        for j in 1..=degree {
            left[j] = uw - knot(1 - j as isize);
            right[j] = knot(j as isize) - uw;
        }
        // Periodic convention: pole 0 starts the span opening at the
        // *last* copy of the first knot, so
        // a seam multiplicity `m` shifts the pole index back by `m - 1`.
        let seam_mult = block.iter().take_while(|&&k| k == first).count();
        (local + n_poles + 1 - seam_mult) % n_poles
    } else {
        let (span, uu) = locate_span(flat, degree, false, u);
        for j in 1..=degree {
            left[j] = uu - flat[span + 1 - j];
            right[j] = flat[span + j] - uu;
        }
        span - degree
    }
}

/// Index of the first of the `degree + 1` poles weighting the span
/// containing `u` (indices wrap modulo `n_poles` in a periodic direction;
/// see [`window`]).
pub(crate) fn span_first_pole(
    u: f64,
    degree: usize,
    periodic: bool,
    flat: &[f64],
    n_poles: usize,
) -> usize {
    let (mut left, mut right) = ([0.0; LARGE], [0.0; LARGE]);
    span_knots(u, degree, periodic, flat, n_poles, &mut left, &mut right)
}

/// Non-zero basis functions alone (The NURBS Book, A2.2): the value-only
/// fast path of [`basis_derivatives`], without its triangular table.
fn basis_values<const N: usize>(
    degree: usize,
    left: &[f64; N],
    right: &[f64; N],
    out: &mut [f64; N],
) {
    out[0] = 1.0;
    for j in 1..=degree {
        let mut saved = 0.0;
        for r in 0..j {
            let temp = out[r] / (right[r + 1] + left[j - r]);
            out[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        out[j] = saved;
    }
}

/// Non-zero basis functions and their first derivatives: A2.2, whose last
/// step divides each degree-`(p - 1)` function by exactly the knot span its
/// derivative needs, `N'_j = p * (temp_{j-1} - temp_j)`.
fn basis_first<const N: usize>(
    degree: usize,
    left: &[f64; N],
    right: &[f64; N],
    ders: &mut [[f64; N]; MAX_DERIVATIVE + 1],
) {
    let p = degree;
    let [values, firsts, _] = ders;
    values[0] = 1.0;
    for j in 1..p {
        let mut saved = 0.0;
        for r in 0..j {
            let temp = values[r] / (right[r + 1] + left[j - r]);
            values[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        values[j] = saved;
    }
    let mut saved = 0.0;
    let mut previous = 0.0;
    for r in 0..p {
        let temp = values[r] / (right[r + 1] + left[p - r]);
        values[r] = saved + right[r + 1] * temp;
        saved = left[p - r] * temp;
        firsts[r] = p as f64 * (previous - temp);
        previous = temp;
    }
    values[p] = saved;
    firsts[p] = p as f64 * previous;
}

/// Non-zero basis functions and their derivatives up to `order` (The NURBS
/// Book, A2.3), from `left[j] = u - U[span + 1 - j]` and
/// `right[j] = U[span + j] - u` for `j` in `1..=degree`.
///
/// Every divisor is a difference of two knots on either side of a
/// nonzero-width span, so none is zero. Derivatives above the degree vanish
/// and are left at zero.
fn basis_derivatives<const N: usize>(
    degree: usize,
    left: &[f64; N],
    right: &[f64; N],
    order: usize,
    ders: &mut [[f64; N]; MAX_DERIVATIVE + 1],
) {
    let p = degree;
    // ndu: basis functions in the upper triangle, knot differences in the
    // lower one.
    let mut ndu = [[0.0f64; N]; N];
    ndu[0][0] = 1.0;
    for j in 1..=p {
        let mut saved = 0.0;
        for r in 0..j {
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    for j in 0..=p {
        ders[0][j] = ndu[j][p];
    }
    let order = order.min(p).min(MAX_DERIVATIVE);
    if order == 0 {
        return;
    }
    let mut a = [[0.0f64; MAX_DERIVATIVE + 1]; 2];
    for r in 0..=p {
        let (mut s1, mut s2) = (0, 1);
        a[0][0] = 1.0;
        for k in 1..=order {
            let mut d = 0.0;
            let pk = p - k;
            if r >= k {
                let rk = r - k;
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
                d = a[s2][0] * ndu[rk][pk];
            }
            let j1 = if r + 1 >= k { 1 } else { k - r };
            let j2 = if r <= pk + 1 { k - 1 } else { p - r };
            for j in j1..=j2 {
                // r + j >= k holds for every j in j1..=j2.
                let rkj = r + j - k;
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rkj];
                d += a[s2][j] * ndu[rkj][pk];
            }
            if r <= pk {
                a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
                d += a[s2][k] * ndu[r][pk];
            }
            ders[k][r] = d;
            std::mem::swap(&mut s1, &mut s2);
        }
    }
    let mut factor = p as f64;
    for (k, row) in ders.iter_mut().enumerate().take(order + 1).skip(1) {
        for value in row.iter_mut().take(p + 1) {
            *value *= factor;
        }
        factor *= (p - k) as f64;
    }
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

/// Evaluates the value and first `n` derivatives (`n <=`
/// [`MAX_DERIVATIVE`]), writing `f, f', ..., f^n` consecutively into `out`,
/// each occupying `dim` coordinates (so `out` must hold `(n + 1) * dim`
/// values).
///
/// `flat` is the plain [`flat_knots`] of the full knots/mults, for periodic
/// curves too: the period is derived from its endpoints.
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
    if degree < SMALL {
        eval_dn_sized::<SMALL>(u, degree, periodic, flat, poles, dim, n, out);
    } else {
        eval_dn_sized::<LARGE>(u, degree, periodic, flat, poles, dim, n, out);
    }
}

#[allow(clippy::too_many_arguments)] // see `eval_dn`
fn eval_dn_sized<const N: usize>(
    u: f64,
    degree: usize,
    periodic: bool,
    flat: &[f64],
    poles: &[f64],
    dim: usize,
    n: usize,
    out: &mut [f64],
) {
    debug_assert!(n <= MAX_DERIVATIVE, "derivative order {n} not supported");
    let mut w = Window::<N>::new();
    window(u, degree, periodic, flat, poles.len() / dim, n, &mut w);
    let out = &mut out[..(n + 1) * dim];
    // Fixed strides let the coordinate loop unroll.
    match dim {
        3 => accumulate::<N, 3>(&w, degree, poles, out),
        4 => accumulate::<N, 4>(&w, degree, poles, out),
        _ => accumulate_dyn(&w, degree, poles, dim, out),
    }
}

/// `out[k] = Σ_j ders[k][j] * poles[j]` with a compile-time stride `D`.
fn accumulate<const N: usize, const D: usize>(
    w: &Window<N>,
    degree: usize,
    poles: &[f64],
    out: &mut [f64],
) {
    for (k, slot) in out.as_chunks_mut::<D>().0.iter_mut().enumerate() {
        let mut sum = [0.0f64; D];
        for j in 0..=degree {
            let weight = w.ders[k][j];
            let base = w.pole(j) * D;
            let pole: &[f64; D] = poles[base..base + D].try_into().unwrap();
            for c in 0..D {
                sum[c] += weight * pole[c];
            }
        }
        *slot = sum;
    }
}

/// [`accumulate`] for any stride.
fn accumulate_dyn<const N: usize>(
    w: &Window<N>,
    degree: usize,
    poles: &[f64],
    dim: usize,
    out: &mut [f64],
) {
    out.fill(0.0);
    for (k, slot) in out.chunks_exact_mut(dim).enumerate() {
        for j in 0..=degree {
            let weight = w.ders[k][j];
            let index = w.pole(j);
            let pole = &poles[index * dim..(index + 1) * dim];
            for (o, &c) in slot.iter_mut().zip(pole) {
                *o += weight * c;
            }
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

/// `lo`, the knots strictly inside `(lo, hi)`, and `hi`: the boundaries
/// of the knot spans (polynomial pieces) of one direction.
pub(crate) fn span_breaks(knots: &[f64], lo: f64, hi: f64) -> Vec<f64> {
    let mut breaks = vec![lo];
    breaks.extend(knots.iter().copied().filter(|&k| k > lo && k < hi));
    breaks.push(hi);
    breaks
}

/// Whether the direction fails to be C1 at each of `breaks` (from
/// [`span_breaks`]): an end of a non-periodic direction, a crease (a knot
/// of multiplicity `>= degree`), or a periodic seam that is a crease. A
/// minimum of a smooth function there need not be a critical point.
pub(crate) fn crease_breaks(
    knots: &[f64],
    mults: &[u32],
    degree: usize,
    periodic: bool,
    breaks: &[f64],
) -> Vec<bool> {
    let last = breaks.len() - 1;
    breaks
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            if i == 0 || i == last {
                !periodic || mults[0] as usize >= degree
            } else {
                knots
                    .iter()
                    .position(|&k| k == x)
                    .is_some_and(|k| mults[k] as usize >= degree)
            }
        })
        .collect()
}

/// The [`bezier_matrix`] of every span between consecutive `breaks` (from
/// [`span_breaks`]): the first pole index of each span, and the matrices
/// row-major, `(degree + 1)^2` values per span.
pub(crate) fn span_bezier_matrices(
    breaks: &[f64],
    degree: usize,
    periodic: bool,
    flat: &[f64],
    n_poles: usize,
) -> (Vec<usize>, Vec<f64>) {
    fn fill<const N: usize>(
        breaks: &[f64],
        degree: usize,
        periodic: bool,
        flat: &[f64],
        n_poles: usize,
    ) -> (Vec<usize>, Vec<f64>) {
        let n = degree + 1;
        let mut firsts = Vec::with_capacity(breaks.len() - 1);
        let mut matrices = Vec::with_capacity((breaks.len() - 1) * n * n);
        let mut matrix = [[0.0f64; N]; N];
        for w in breaks.windows(2) {
            let mid = 0.5 * (w[0] + w[1]);
            firsts.push(bezier_matrix(
                mid,
                degree,
                periodic,
                flat,
                n_poles,
                &mut matrix,
            ));
            for row in &matrix[..n] {
                matrices.extend_from_slice(&row[..n]);
            }
        }
        (firsts, matrices)
    }
    if degree < SMALL {
        fill::<SMALL>(breaks, degree, periodic, flat, n_poles)
    } else {
        fill::<LARGE>(breaks, degree, periodic, flat, n_poles)
    }
}

/// The parameter intervals of one direction over `[lo, hi]` on which it
/// is at least C1, so that a local refinement bounded to one of them
/// minimises a smooth function: the range cut at every knot of
/// multiplicity `>= degree` (a C0 crease). `None` for a periodic
/// direction smooth everywhere, where refinement may run unbounded. On a
/// periodic direction with a smooth seam, the interval across the seam
/// runs from the last crease past `hi` to the first crease plus the
/// period.
pub(crate) fn smooth_regions(
    knots: &[f64],
    mults: &[u32],
    degree: usize,
    periodic: bool,
    lo: f64,
    hi: f64,
) -> Option<Vec<(f64, f64)>> {
    let creases: Vec<f64> = knots
        .iter()
        .zip(mults)
        .filter(|&(&k, &m)| k > lo && k < hi && m as usize >= degree)
        .map(|(&k, _)| k)
        .collect();
    let smooth_seam = periodic && (mults[0] as usize) < degree;
    if smooth_seam {
        if creases.is_empty() {
            return None;
        }
        let mut regions: Vec<(f64, f64)> = creases.windows(2).map(|w| (w[0], w[1])).collect();
        regions.push((creases[creases.len() - 1], creases[0] + (hi - lo)));
        return Some(regions);
    }
    let mut cuts = vec![lo];
    cuts.extend(creases);
    cuts.push(hi);
    Some(cuts.windows(2).map(|w| (w[0], w[1])).collect())
}

/// The bounds for refining from `t` (within `[lo, lo + period]`): each
/// region of [`smooth_regions`] containing `t` (two at a crease), with
/// `t` shifted by the period into those crossing a periodic seam; a
/// single unbounded entry when `regions` is `None`.
pub(crate) fn regions_at(
    regions: &Option<Vec<(f64, f64)>>,
    t: f64,
    period: Option<f64>,
) -> Vec<(Option<(f64, f64)>, f64)> {
    let Some(regions) = regions else {
        return vec![(None, t)];
    };
    let mut out = Vec::new();
    for &(a, b) in regions {
        let shifted = std::iter::once(t).chain(period.map(|p| t + p));
        if let Some(x) = shifted.into_iter().find(|&x| a <= x && x <= b) {
            out.push((Some((a, b)), x));
        }
    }
    out
}

/// The parameter at which to evaluate derivatives for a refinement
/// bounded to `bounds`: `x` itself, except exactly at the upper bound,
/// where the next float down selects the span on the left (evaluation
/// takes the span on the right of a knot, which beyond a crease is not
/// the region being refined).
pub(crate) fn inward(x: f64, bounds: Option<(f64, f64)>) -> f64 {
    match bounds {
        Some((_, hi)) if x >= hi => x.next_down(),
        _ => x,
    }
}

/// Euclidean bounding box `[min.., max..]` of the `degree + 1` poles of
/// every span between consecutive `breaks` (each span lies in it, by the
/// convex hull property; weights are positive). `flat_poles` holds `dim`
/// coordinates per pole (`4`: homogeneous).
pub(crate) fn span_boxes(
    breaks: &[f64],
    degree: usize,
    periodic: bool,
    flat: &[f64],
    flat_poles: &[f64],
    dim: usize,
) -> Vec<[f64; 6]> {
    if degree < SMALL {
        boxes::<SMALL>(breaks, degree, periodic, flat, flat_poles, dim)
    } else {
        boxes::<LARGE>(breaks, degree, periodic, flat, flat_poles, dim)
    }
}

fn boxes<const N: usize>(
    breaks: &[f64],
    degree: usize,
    periodic: bool,
    flat: &[f64],
    flat_poles: &[f64],
    dim: usize,
) -> Vec<[f64; 6]> {
    let n_poles = flat_poles.len() / dim;
    let mut w = Window::<N>::new();
    breaks
        .windows(2)
        .map(|span| {
            // The window at the midpoint selects exactly the span's poles.
            window(
                0.5 * (span[0] + span[1]),
                degree,
                periodic,
                flat,
                n_poles,
                0,
                &mut w,
            );
            let mut b = [
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ];
            for j in 0..=degree {
                let base = w.pole(j) * dim;
                let inv = if dim == 4 {
                    1.0 / flat_poles[base + 3]
                } else {
                    1.0
                };
                for c in 0..3 {
                    let x = flat_poles[base + c] * inv;
                    b[c] = b[c].min(x);
                    b[c + 3] = b[c + 3].max(x);
                }
            }
            b
        })
        .collect()
}

/// Bernstein control points, over its own parameter interval, of the
/// polynomial piece on the span containing `u` (pass a parameter inside
/// the span): coordinate `c` of point `k` goes to `out[c][k]`, for the
/// `dim = out.len() <= 4` coordinates of `flat_poles` (homogeneous when
/// `dim == 4`).
///
/// The points are the span's poles mapped by [`bezier_matrix`].
pub(crate) fn span_bezier(
    u: f64,
    degree: usize,
    periodic: bool,
    flat: &[f64],
    flat_poles: &[f64],
    out: &mut [&mut [f64]],
) {
    if degree < SMALL {
        bezier::<SMALL>(u, degree, periodic, flat, flat_poles, out);
    } else {
        bezier::<LARGE>(u, degree, periodic, flat, flat_poles, out);
    }
}

fn bezier<const N: usize>(
    u: f64,
    degree: usize,
    periodic: bool,
    flat: &[f64],
    flat_poles: &[f64],
    out: &mut [&mut [f64]],
) {
    let dim = out.len();
    debug_assert!(dim <= 4);
    let n_poles = flat_poles.len() / dim;
    let mut matrix = [[0.0f64; N]; N];
    let first = bezier_matrix(u, degree, periodic, flat, n_poles, &mut matrix);
    for (c, o) in out.iter_mut().enumerate() {
        for (k, row) in matrix.iter().enumerate().take(degree + 1) {
            o[k] = (0..=degree)
                .map(|j| row[j] * flat_poles[((first + j) % n_poles) * dim + c])
                .sum();
        }
    }
}

/// The matrix taking the `degree + 1` poles of the span containing `u`
/// (pass a parameter inside the span) to its Bernstein control points
/// over its own interval: point `k` is `sum_j matrix[k][j] * P[first +
/// j]`, the pole index wrapping modulo `n_poles` in a periodic direction.
/// Returns `first`. Rows are convex weights, so the map is stable.
///
/// Point `k` is the blossom of the piece at `k` copies of the span end and
/// `degree - k` of its start, by de Boor's recurrence with the argument
/// varying by level, carried out on the poles' coefficients.
#[allow(clippy::needless_range_loop)] // index arithmetic on the triangular recurrence
pub(crate) fn bezier_matrix<const N: usize>(
    u: f64,
    degree: usize,
    periodic: bool,
    flat: &[f64],
    n_poles: usize,
    matrix: &mut [[f64; N]; N],
) -> usize {
    let p = degree;
    let mut left = [0.0f64; N];
    let mut right = [0.0f64; N];
    let first = span_knots(u, p, periodic, flat, n_poles, &mut left, &mut right);
    // The span's `2p` surrounding knots relative to `u`, the span itself
    // being `[knot(p - 1), knot(p)]`.
    let knot = |m: usize| {
        if m < p {
            -left[p - m]
        } else {
            right[m + 1 - p]
        }
    };
    let (a, b) = (knot(p - 1), knot(p));
    // Level `r`, index `j` blends between `knot(j - 1)` and `knot(p + j - r)`.
    let mut inverse = [[0.0f64; N]; N];
    for r in 1..=p {
        for j in r..=p {
            inverse[r][j] = 1.0 / (knot(p + j - r) - knot(j - 1));
        }
    }
    // `d[j][i]`: coefficient of pole `i` in the level's point `j`, nonzero
    // for `i` in `j - r..=j` at level `r`.
    let mut d = [[0.0f64; N]; N];
    for k in 0..=p {
        for (j, dj) in d.iter_mut().enumerate().take(p + 1) {
            dj[..=p].fill(0.0);
            dj[j] = 1.0;
        }
        for r in 1..=p {
            let x = if r <= k { b } else { a };
            for j in (r..=p).rev() {
                let alpha = (x - knot(j - 1)) * inverse[r][j];
                for i in j - r..=j {
                    d[j][i] = (1.0 - alpha) * d[j - 1][i] + alpha * d[j][i];
                }
            }
        }
        matrix[k][..=p].copy_from_slice(&d[p][..=p]);
    }
    first
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

    // ---- span_bezier ----

    /// The Bernstein form of every span matches the evaluator inside it.
    fn assert_bezier_matches(
        degree: usize,
        periodic: bool,
        knots: &[f64],
        mults: &[u32],
        poles: &[f64],
        dim: usize,
    ) {
        let flat = flat_knots(knots, mults);
        let binomial =
            |n: usize, k: usize| (0..k).fold(1.0, |c, i| c * (n - i) as f64 / (i + 1) as f64);
        for span in knots.windows(2) {
            let (a, b) = (span[0], span[1]);
            let mut coords = vec![vec![0.0; degree + 1]; dim];
            let mut out: Vec<&mut [f64]> = coords.iter_mut().map(|c| c.as_mut_slice()).collect();
            span_bezier(0.5 * (a + b), degree, periodic, &flat, poles, &mut out);
            for s in [0.0, 0.2, 0.5, 0.9, 1.0] {
                let mut expected = vec![0.0; dim];
                eval_dn(
                    a + s * (b - a),
                    degree,
                    periodic,
                    &flat,
                    poles,
                    dim,
                    0,
                    &mut expected,
                );
                for c in 0..dim {
                    let got: f64 = (0..=degree)
                        .map(|k| {
                            binomial(degree, k)
                                * s.powi(k as i32)
                                * (1.0 - s).powi((degree - k) as i32)
                                * coords[c][k]
                        })
                        .sum();
                    assert!(
                        (got - expected[c]).abs() < 1e-12,
                        "span [{a}, {b}] s {s}: {got} vs {}",
                        expected[c]
                    );
                }
            }
        }
    }

    #[test]
    fn test_span_bezier_nonuniform_with_interior_multiplicity() {
        let poles: Vec<f64> = (0..8)
            .flat_map(|i| [i as f64, (i * i % 5) as f64, (i % 3) as f64 - 1.0])
            .collect();
        assert_bezier_matches(
            3,
            false,
            &[0.0, 0.7, 1.0, 2.5, 4.0],
            &[4, 1, 2, 1, 4],
            &poles,
            3,
        );
    }

    #[test]
    fn test_span_bezier_rational_homogeneous() {
        let weights = [1.0, 0.4, 2.5, 1.2, 0.8, 3.0];
        let poles: Vec<f64> = (0..6)
            .flat_map(|i| {
                let w = weights[i];
                [i as f64 * w, ((i * 7) % 4) as f64 * w, -(i as f64) * w, w]
            })
            .collect();
        assert_bezier_matches(
            2,
            false,
            &[-1.0, 0.5, 1.5, 3.0, 3.5],
            &[3, 1, 1, 1, 3],
            &poles,
            4,
        );
    }

    #[test]
    fn test_span_bezier_periodic_with_seam_multiplicity() {
        let poles = periodic_ring_poles();
        assert_bezier_matches(
            3,
            true,
            &[0.0, 1.0, 2.0, 3.5, 4.0, 5.0, 6.0],
            &[1, 1, 1, 1, 1, 1, 1],
            &poles,
            3,
        );
        assert_bezier_matches(
            3,
            true,
            &[0.0, 1.5, 3.0, 4.0, 5.0, 6.0],
            &[2, 1, 1, 1, 1, 2],
            &poles,
            3,
        );
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
