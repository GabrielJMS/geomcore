//! Degree elevation of one B-spline direction in a single sweep over its
//! knot spans (after The NURBS Book, A5.9). Each span is brought to Bézier
//! form by knot insertion and raised by the Bernstein elevation formula,
//! then the knot it shares with the previous span is removed back down to
//! its old multiplicity plus the increment, which is exactly the continuity
//! the raised curve keeps there.
//!
//! Poles are flat buffers with a `dim` stride, homogeneous for rational
//! B-splines (so their elevation is exact too); a surface raises one
//! direction by treating each row of its net as one long pole.
//!
//! Every step is an affine combination of poles, derived from blossoms: the
//! pole over the knots `u_1, .., u_p` is the blossom `F(u_1, .., u_p)` of
//! the B-spline, symmetric and affine in each argument.

use super::bspline::{LARGE, flat_knots};

/// One direction raised in degree.
pub(crate) struct Elevated {
    pub(crate) knots: Vec<f64>,
    pub(crate) mults: Vec<u32>,
    pub(crate) flat: Vec<f64>,
    /// Poles with the input's stride.
    pub(crate) poles: Vec<f64>,
}

/// Raises a direction of `degree` by `by >= 1`, its poles `dim` values
/// each, `flat` being [`flat_knots`] of `knots` and `mults`.
///
/// A periodic direction stays periodic, every multiplicity raised by `by`.
/// A non-periodic one comes back clamped at the ends of its parameter
/// range, with multiplicity `degree + by + 1` there and raised by `by` at
/// the knots inside; knots outside the range (only an unclamped direction
/// has some) are dropped.
#[allow(clippy::too_many_arguments)] // one direction's full knot data
pub(crate) fn elevate(
    degree: usize,
    by: usize,
    periodic: bool,
    knots: &[f64],
    mults: &[u32],
    flat: &[f64],
    poles: &[f64],
    dim: usize,
) -> Elevated {
    debug_assert!(by >= 1);
    if periodic {
        elevate_periodic(degree, by, knots, mults, flat, poles, dim)
    } else {
        sweep(degree, by, flat, poles, dim)
    }
}

/// [`elevate`] on a periodic direction: the sweep runs over one period
/// unrolled into a non-periodic direction, and one period of the raised
/// poles is read back.
///
/// The unrolled direction starts a few knots before the seam, so that the
/// seam is an interior knot of the sweep, removed like any other: the
/// raised poles next to it are then those of the periodic direction.
fn elevate_periodic(
    degree: usize,
    by: usize,
    knots: &[f64],
    mults: &[u32],
    flat: &[f64],
    poles: &[f64],
    dim: usize,
) -> Elevated {
    let (p, n) = (degree, poles.len() / dim);
    let spans = knots.len() - 1;
    let (first, last) = (knots[0], knots[spans]);
    let period = last - first;
    let seam = mults[0] as usize;
    // Flat knot `j` of the periodic direction tiled endlessly (one period
    // is `flat` without its trailing seam run), as the evaluator sees it.
    let tiled = |j: isize| {
        let (k, shift) = (j.rem_euclid(n as isize) as usize, j.div_euclid(n as isize));
        match shift {
            0 => flat[k],
            1 if flat[k] == first => last,
            _ => flat[k] + period * shift as f64,
        }
    };
    // The first raised pole spans `p - seam` knots before the seam, which
    // must be knots of the sweep: step back over whole knots until their
    // raised multiplicities cover them. `start` is the flat index of the
    // last copy of the sweep's first knot, and `offset` the raised pole
    // that becomes pole 0 (the first of the span opening at the last copy
    // of the seam).
    let mut start = seam as isize - 1;
    let mut offset = 0;
    let mut missing = (p - seam) as isize;
    let (mut k, mut shift) = (0, 0);
    while missing > 0 {
        if k == 0 {
            k = spans;
            shift -= 1;
        }
        k -= 1;
        let raised = mults[k] as usize + by;
        missing -= raised as isize;
        offset += raised;
        let last_copy = mults[..=k].iter().sum::<u32>() as isize - 1;
        start = last_copy + shift * n as isize;
        if missing <= 0 {
            offset += seam + by - raised;
        }
    }
    // Pole `i` of the unrolled direction is periodic pole `start + 1 -
    // seam + i`, over the tiled knots from `start - p + i + 1`.
    let count = (n as isize + p as isize - start) as usize;
    let unrolled_flat: Vec<f64> = (0..count + p + 1)
        .map(|i| tiled(start - p as isize + i as isize))
        .collect();
    let mut unrolled = Vec::with_capacity(count * dim);
    for i in 0..count {
        let k = (start + 1 - seam as isize + i as isize).rem_euclid(n as isize) as usize;
        unrolled.extend_from_slice(&poles[k * dim..(k + 1) * dim]);
    }
    let swept = sweep(p, by, &unrolled_flat, &unrolled, dim);
    let pole_count = n + by * spans;
    let mults: Vec<u32> = mults.iter().map(|&m| m + by as u32).collect();
    Elevated {
        flat: flat_knots(knots, &mults),
        knots: knots.to_vec(),
        mults,
        poles: swept.poles[offset * dim..(offset + pole_count) * dim].to_vec(),
    }
}

/// [`elevate`] on a non-periodic direction, over its range `[u[p], u[n]]`.
fn sweep(degree: usize, by: usize, u: &[f64], poles: &[f64], dim: usize) -> Elevated {
    // Constant strides let the coordinate loops unroll.
    match dim {
        3 => sweep_with(degree, by, u, poles, 3),
        4 => sweep_with(degree, by, u, poles, 4),
        _ => sweep_with(degree, by, u, poles, dim),
    }
}

#[inline(always)]
fn sweep_with(degree: usize, by: usize, u: &[f64], poles: &[f64], dim: usize) -> Elevated {
    let (p, q) = (degree, degree + by);
    let n = poles.len() / dim;
    let (lo, hi) = (u[p], u[n]);
    let mut s = p;
    while u[s + 1] <= lo {
        s += 1;
    }
    let inside = &u[s + 1..n];
    let interior = inside.iter().filter(|&&x| x < hi).count();
    let breaks = (s + 1..n).filter(|&k| u[k] < hi && u[k] > u[k - 1]).count();
    let pole_count = q + 1 + interior + by * breaks;
    let mut out = Elevated {
        knots: Vec::with_capacity(breaks + 2),
        mults: Vec::with_capacity(breaks + 2),
        flat: Vec::with_capacity(pole_count + q + 1),
        poles: vec![0.0; pole_count * dim],
    };
    // Poles written so far.
    let mut written = 0;

    let w = p + 1;
    let mut scratch = vec![0.0; (q + 1) * w + 4 * w * dim];
    let (coefficients, rest) = scratch.split_at_mut((q + 1) * w);
    elevation_matrix(p, by, coefficients);
    let (piece, rest) = rest.split_at_mut(w * dim);
    let (next, rest) = rest.split_at_mut(w * dim);
    let (left, right) = rest.split_at_mut(w * dim);

    // `piece[i]` is the blossom at `x^(p - i)` and the `i` knots after
    // the span start `x`: the span's poles once `x` has `p` copies.
    piece.copy_from_slice(&poles[(s - p) * dim..(s + 1) * dim]);
    clamp_start(p, &u[s + 1 - p..s + 1 + p], piece, next, dim);
    let mut alphas = [0.0; LARGE];
    let mut start_mult = 0;
    loop {
        let (x, y) = (u[s], u[s + 1]);
        let mut mult = 1;
        while mult < p && u[s + 1 + mult] == y {
            mult += 1;
        }
        // Insert `y` until it has `p` copies: the piece becomes Bézier on
        // `[x, y]`, and the last pole after each insertion is one of the
        // next span's.
        let inserts = p - mult;
        for (k, alpha) in alphas.iter_mut().enumerate().take(p).skip(mult) {
            *alpha = (y - x) / (u[s + k + 1] - x);
        }
        for j in 1..=inserts {
            for k in (mult + j..=p).rev() {
                let a = alphas[k - j];
                for c in 0..dim {
                    piece[k * dim + c] =
                        a * piece[k * dim + c] + (1.0 - a) * piece[(k - 1) * dim + c];
                }
            }
            next[(inserts - j) * dim..(inserts - j + 1) * dim]
                .copy_from_slice(&piece[p * dim..(p + 1) * dim]);
        }
        if written == 0 {
            raise(
                &mut out.poles[..(q + 1) * dim],
                piece,
                coefficients,
                p,
                by,
                0,
                dim,
            );
            written = q + 1;
            push_knot(&mut out, x, q + 1);
        } else {
            // The piece's first point is the previous piece's last.
            let joint = written - 1;
            let removals = p - start_mult;
            join(
                &mut out.poles,
                &out.flat,
                joint,
                removals,
                (x, y),
                (piece, coefficients, p, by),
                dim,
                left,
                right,
            );
            written = joint + q - removals + 1;
            push_knot(&mut out, x, q - removals);
        }
        if y == hi {
            debug_assert_eq!(written, pole_count);
            push_knot(&mut out, y, q + 1);
            return out;
        }
        let s_next = s + mult;
        piece[..inserts * dim].copy_from_slice(&next[..inserts * dim]);
        piece[inserts * dim..].copy_from_slice(&poles[s * dim..(s_next + 1) * dim]);
        start_mult = mult;
        s = s_next;
    }
}

/// Brings the first span's `p + 1` poles to the form [`sweep`] expects,
/// `x` (`knots[p - 1]`, the span start) repeated `p` times: `knots` are
/// the `2p` knots around the span. `spare` holds as many values as
/// `piece`.
///
/// Inserting `x` turns the pole at `knots[i..i + p]` into one at `x`
/// and `knots[i + 1..i + p]`; after `r` rounds the last pole is the
/// blossom at `x^r` and `knots[p..2p - r]`.
#[inline(always)]
fn clamp_start(p: usize, knots: &[f64], piece: &mut [f64], spare: &mut [f64], dim: usize) {
    let x = knots[p - 1];
    if knots[0] == x {
        return;
    }
    spare[p * dim..(p + 1) * dim].copy_from_slice(&piece[p * dim..(p + 1) * dim]);
    for r in 1..=p {
        for i in (r..=p).rev() {
            let a = (x - knots[i - 1]) / (knots[i + p - r] - knots[i - 1]);
            for c in 0..dim {
                piece[i * dim + c] = (1.0 - a) * piece[(i - 1) * dim + c] + a * piece[i * dim + c];
            }
        }
        spare[(p - r) * dim..(p - r + 1) * dim].copy_from_slice(&piece[p * dim..(p + 1) * dim]);
    }
    piece.copy_from_slice(spare);
}

/// The `(p + by + 1) x (p + 1)` matrix, row-major, raising Bernstein
/// coefficients of degree `p` to degree `p + by`: point `i` is
/// `sum_j C(p, j) C(by, i - j) / C(p + by, i) * b_j`. Entries outside
/// `i - by <= j <= i` are left untouched.
fn elevation_matrix(p: usize, by: usize, out: &mut [f64]) {
    let q = p + by;
    // Rows `p`, `by` and `q` of Pascal's triangle, exact in `f64`.
    let mut row = [0.0; LARGE];
    let (mut of_p, mut of_by) = ([0.0; LARGE], [0.0; LARGE]);
    row[0] = 1.0;
    for n in 0..=q {
        if n == p {
            of_p = row;
        }
        if n == by {
            of_by = row;
        }
        if n < q {
            for k in (1..=n + 1).rev() {
                row[k] += row[k - 1];
            }
        }
    }
    for i in 0..=q {
        for j in i.saturating_sub(by)..=i.min(p) {
            out[i * (p + 1) + j] = of_p[j] * of_by[i - j] / row[i];
        }
    }
}

/// Writes the raised points `from..` of the Bézier `piece` of degree `p`
/// into `out`, as many as it holds.
#[inline(always)]
fn raise(
    out: &mut [f64],
    piece: &[f64],
    coefficients: &[f64],
    p: usize,
    by: usize,
    from: usize,
    dim: usize,
) {
    for (point, i) in out.chunks_exact_mut(dim).zip(from..) {
        let (lo, hi) = (i.saturating_sub(by), i.min(p));
        let weight = coefficients[i * (p + 1) + lo];
        for (x, &y) in point.iter_mut().zip(&piece[lo * dim..(lo + 1) * dim]) {
            *x = weight * y;
        }
        for j in lo + 1..=hi {
            let weight = coefficients[i * (p + 1) + j];
            for (x, &y) in point.iter_mut().zip(&piece[j * dim..(j + 1) * dim]) {
                *x += weight * y;
            }
        }
    }
}

fn push_knot(out: &mut Elevated, x: f64, mult: usize) {
    out.knots.push(x);
    out.mults.push(mult as u32);
    out.flat.extend(std::iter::repeat_n(x, mult));
}

/// Appends the Bézier piece `(piece, coefficients, p, by)` on `[b, c]`,
/// raised to degree `q = p + by`, to the poles swept so far, whose last
/// one, `poles[joint]`, is the piece's first point, and removes `count`
/// of the `q` copies of `b` between them. `flat` holds the knots swept so
/// far, the last being `flat[joint]`.
///
/// With `l_k = flat[joint + 1 - k]`, the pole `joint - i` is the blossom
/// `G(i, 0)` and raised point `k` is `G(0, k)`, where `G(i, k)` is the
/// blossom at `l_i, .., l_1`, `q - i - k` copies of `b` and `k` of `c`.
/// Writing one `b` as the affine combination of `l_(i + 1)` and `c` gives
/// `G(i, k) = a_i G(i, k + 1) + (1 - a_i) G(i + 1, k)`, with `a_i = (b -
/// l_(i + 1)) / (c - l_(i + 1))`. The new poles `G(i, count - i)`, for
/// `0 < i < count`, replace the poles `joint - i`; they are solved for
/// from the left side (`i` above `count / 2`) or from the right, each in
/// as few steps as possible, and averaged in the middle. Raised points
/// `count..` follow from `joint` on, unchanged.
#[allow(clippy::too_many_arguments)] // two scratch buffers of the sweep
#[inline(always)]
fn join(
    poles: &mut [f64],
    flat: &[f64],
    joint: usize,
    count: usize,
    (b, c): (f64, f64),
    (piece, coefficients, p, by): (&[f64], &[f64], usize, usize),
    dim: usize,
    left: &mut [f64],
    right: &mut [f64],
) {
    let (r, q) = (count, p + by);
    if r >= 2 {
        let alpha = |i: usize| {
            let l = flat[joint - i];
            (b - l) / (c - l)
        };
        // `left[i - lower]` holds `G(i, k)` for `i >= lower`,
        // `right[k - base]` holds `G(i, k)` for `k >= base`.
        let (lower, upper) = (r - r / 2, r / 2);
        let base = r - upper;
        for i in lower..=r {
            left[(i - lower) * dim..(i - lower + 1) * dim]
                .copy_from_slice(&poles[(joint - i) * dim..(joint - i + 1) * dim]);
        }
        for level in 1..=r - lower {
            for i in lower..=r - level {
                let a = alpha(i);
                let (row, above) = left[(i - lower) * dim..].split_at_mut(dim);
                for (x, &y) in row.iter_mut().zip(&above[..dim]) {
                    *x = (*x - (1.0 - a) * y) / a;
                }
            }
        }
        raise(
            &mut right[..(r - base + 1) * dim],
            piece,
            coefficients,
            p,
            by,
            base,
            dim,
        );
        for level in 1..=upper {
            let a = alpha(level - 1);
            for k in base..=r - level {
                let (row, above) = right[(k - base) * dim..].split_at_mut(dim);
                for (x, &y) in row.iter_mut().zip(&above[..dim]) {
                    *x = (*x - a * y) / (1.0 - a);
                }
            }
        }
        for i in 1..r {
            let target = &mut poles[(joint - i) * dim..(joint - i + 1) * dim];
            let from_left = &left[(i.max(lower) - lower) * dim..][..dim];
            let from_right = &right[((r - i).max(base) - base) * dim..][..dim];
            for c in 0..dim {
                target[c] = if i < lower {
                    from_right[c]
                } else if i > upper {
                    from_left[c]
                } else {
                    0.5 * (from_left[c] + from_right[c])
                };
            }
        }
    }
    // Without a removal the joint stays as swept.
    let from = r.max(1);
    let start = joint + from - r;
    raise(
        &mut poles[start * dim..(start + q + 1 - from) * dim],
        piece,
        coefficients,
        p,
        by,
        from,
        dim,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve_math::bspline::eval_dn;

    #[test]
    fn test_elevation_matrix_rows_are_convex_weights() {
        for (p, by) in [(1, 1), (2, 3), (3, 1), (5, 4), (12, 13)] {
            let w = p + 1;
            let mut m = vec![0.0; (p + by + 1) * w];
            elevation_matrix(p, by, &mut m);
            for row in m.chunks_exact(w) {
                let sum: f64 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-14, "p {p} by {by}: {row:?}");
                assert!(row.iter().all(|&x| x >= 0.0));
            }
        }
    }

    /// Evaluates `poles` (stride `dim`) over the knots at `u`.
    fn eval(p: usize, periodic: bool, flat: &[f64], poles: &[f64], dim: usize, u: f64) -> Vec<f64> {
        let mut out = vec![0.0; 3 * dim];
        eval_dn(u, p, periodic, flat, poles, dim, 2, &mut out);
        out
    }

    #[test]
    fn test_wide_poles_elevate_like_each_coordinate() {
        // A stride of 7 takes the generic path; each coordinate is a
        // separate function of the parameter.
        let (p, dim) = (3, 7);
        let knots = [0.0, 0.4, 1.1, 1.5, 2.6, 3.0];
        let mults = [4, 2, 1, 3, 1, 4];
        let flat = flat_knots(&knots, &mults);
        let n = flat.len() - p - 1;
        let poles: Vec<f64> = (0..n * dim)
            .map(|i| ((i * 37 % 11) as f64 - 5.0) * 0.3)
            .collect();
        for by in [1, 2, 4] {
            let raised = elevate(p, by, false, &knots, &mults, &flat, &poles, dim);
            let expected: Vec<u32> = mults.iter().map(|&m| m + by as u32).collect();
            assert_eq!(raised.mults, expected);
            assert_eq!(raised.knots, knots);
            assert_eq!(raised.poles.len() / dim + p + by + 1, raised.flat.len());
            for i in 0..=60 {
                let u = 3.0 * i as f64 / 60.0;
                let a = eval(p, false, &flat, &poles, dim, u);
                let b = eval(p + by, false, &raised.flat, &raised.poles, dim, u);
                for (x, y) in a.iter().zip(&b) {
                    assert!(
                        (x - y).abs() < 1e-12 * (1.0 + x.abs()),
                        "by {by} u {u}: {x} vs {y}"
                    );
                }
            }
        }
    }
}
