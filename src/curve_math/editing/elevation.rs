//! Degree elevation of one direction in a single sweep over its knot spans
//! (after The NURBS Book, A5.9). Each span is brought to Bézier form by
//! knot insertion and raised by the Bernstein elevation formula, then the
//! knot it shares with the previous span is removed back down to its old
//! multiplicity plus the increment, the continuity the raised direction
//! keeps there.
//!
//! Every step is an affine combination of poles, derived from blossoms: the
//! pole over the knots `u_1, .., u_p` is the blossom `F(u_1, .., u_p)` of
//! the B-spline, symmetric and affine in each argument.

use super::bezier::walk;
use super::{Edit, Edited, Kept, Lanes, One, Pole, Source, Width};
use crate::curve_math::bspline::{KnotVector, LARGE, MAX_DEGREE, flat_knots};
use crate::curves::DegreeElevationError;

/// Raising the degree by the amount: see [`elevate`].
pub(crate) struct Elevation(pub(crate) usize);

impl Edit for Elevation {
    type Error = DegreeElevationError;

    fn apply<S: Source>(
        &self,
        direction: &KnotVector,
        poles: S,
        lanes: Lanes,
    ) -> Result<Edited<S::Pole>, DegreeElevationError> {
        elevate(direction, poles, lanes, self.0)
    }
}

/// `knots` and its poles with the degree raised by `by`.
///
/// A periodic direction stays periodic, every multiplicity raised by `by`.
/// A non-periodic one comes back clamped at the ends of its parameter
/// range, with multiplicity `degree + by + 1` there and raised by `by` at
/// the knots inside; knots outside the range (only an unclamped direction
/// has some) are dropped.
fn elevate<S: Source>(
    knots: &KnotVector,
    poles: S,
    lanes: Lanes,
    by: usize,
) -> Result<Edited<S::Pole>, DegreeElevationError> {
    let p = knots.degree();
    if by > MAX_DEGREE - p {
        return Err(DegreeElevationError::DegreeTooHigh);
    }
    if by == 0 {
        let kept = Kept {
            leading: knots.pole_count(),
            trailing: 0,
        };
        let poles = (0..poles.len()).map(|i| poles.get(i)).collect();
        return Ok(Edited {
            knots: knots.clone(),
            poles,
            kept,
        });
    }
    let plan = if knots.is_periodic() {
        Plan::periodic(knots, by)
    } else {
        Plan::clamped(knots.flat(), p, by)
    };
    let mut coefficients = vec![0.0; (p + by + 1) * (p + 1)];
    elevation_matrix(p, by, &mut coefficients);
    let new_n = plan.knots.pole_count();
    let mut out = vec![S::Pole::default(); new_n * lanes.count];
    if lanes.consecutive {
        let n = poles.len() / lanes.count;
        for (lane, lane_out) in out.chunks_exact_mut(new_n).enumerate() {
            let lane_poles = poles.slice(lane * n..(lane + 1) * n);
            plan.run(knots, &coefficients, lane_poles, One, lane_out);
        }
    } else {
        plan.run(knots, &coefficients, poles, lanes.count, &mut out);
    }
    Ok(Edited {
        knots: plan.knots,
        poles: out,
        kept: plan.kept,
    })
}

/// What a direction's elevation does, the same for every lane: the raised
/// knots, and the sweep that makes the raised poles.
struct Plan {
    by: usize,
    /// The raised direction.
    knots: KnotVector,
    kept: Kept,
    /// For a periodic direction, the unrolled direction swept instead.
    unrolled: Option<Unrolled>,
}

/// A periodic direction's period unrolled into a non-periodic one: pole
/// `i` is periodic pole `first + i` (wrapped), over the flat knots `flat`;
/// raised pole `offset` of `raised`, its raised direction, is the periodic
/// direction's raised pole 0.
struct Unrolled {
    flat: Vec<f64>,
    first: isize,
    count: usize,
    raised: KnotVector,
    offset: usize,
}

impl Plan {
    /// The plan for the non-periodic direction of degree `p` over `u`,
    /// clamped at its range `[u[p], u[n]]`.
    fn clamped(u: &[f64], p: usize, by: usize) -> Plan {
        let q = p + by;
        let n = u.len() - p - 1;
        let (lo, hi) = (u[p], u[n]);
        // At most every pole but the last `p` starts a new knot.
        let mut knots = Vec::with_capacity(n - p + 1);
        let mut multiplicities = Vec::with_capacity(n - p + 1);
        knots.push(lo);
        multiplicities.push(q as u32 + 1);
        for &x in &u[p + 1..n] {
            if x > lo && x < hi {
                if knots.last() == Some(&x) {
                    *multiplicities.last_mut().unwrap() += 1;
                } else {
                    knots.push(x);
                    multiplicities.push(by as u32 + 1);
                }
            }
        }
        knots.push(hi);
        multiplicities.push(q as u32 + 1);
        let flat = flat_knots(&knots, &multiplicities);
        // The first and last raised points of a clamped end are its poles.
        let kept = Kept {
            leading: usize::from(u[p + 1] > lo && u[1] == lo),
            trailing: usize::from(u[n - 1] < hi && u[n + p - 1] == hi),
        };
        Plan {
            by,
            knots: KnotVector::from_validated(q, knots, multiplicities, flat, false),
            kept,
            unrolled: None,
        }
    }

    /// The plan for a periodic direction: the sweep runs over one period
    /// unrolled into a non-periodic direction, and one period of the
    /// raised poles is read back.
    ///
    /// The unrolled direction starts a few knots before the seam, so that
    /// the seam is an interior knot of the sweep, removed like any other:
    /// the raised poles next to it are then those of the periodic
    /// direction.
    fn periodic(direction: &KnotVector, by: usize) -> Plan {
        let (p, n) = (direction.degree(), direction.pole_count());
        let (knots, mults, flat) = (
            direction.knots(),
            direction.multiplicities(),
            direction.flat(),
        );
        let spans = knots.len() - 1;
        let (first, last) = (knots[0], knots[spans]);
        let period = last - first;
        let seam = mults[0] as usize;
        // Flat knot `j` of the periodic direction tiled endlessly (one
        // period is `flat` without its trailing seam run), as the
        // evaluator sees it.
        let tiled = |j: isize| {
            let (k, shift) = (j.rem_euclid(n as isize) as usize, j.div_euclid(n as isize));
            match shift {
                0 => flat[k],
                1 if flat[k] == first => last,
                _ => flat[k] + period * shift as f64,
            }
        };
        // The first raised pole spans `p - seam` raised knots before the
        // seam, which must be knots of the sweep: step back over whole
        // knots until their raised copies cover them. `start` is the flat
        // index of the last copy of the sweep's first knot, and `offset`
        // the raised pole that becomes pole 0 (the first of the span
        // opening at the last copy of the seam).
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
                // The sweep's first knot is clamped, not raised: in its
                // place, the seam's raised copies. In this order, as an
                // earlier knot may have more copies than the seam.
                offset = offset + seam + by - raised;
            }
        }
        // Pole `i` of the unrolled direction is periodic pole `start + 1 -
        // seam + i`, over the tiled knots from `start - p + i + 1`.
        let count = (n as isize + p as isize - start) as usize;
        let unrolled_flat: Vec<f64> = (0..count + p + 1)
            .map(|i| tiled(start - p as isize + i as isize))
            .collect();
        let swept = Plan::clamped(&unrolled_flat, p, by).knots;
        let raised: Vec<u32> = mults.iter().map(|&m| m + by as u32).collect();
        let raised_flat = flat_knots(knots, &raised);
        Plan {
            by,
            knots: KnotVector::from_validated(p + by, knots.to_vec(), raised, raised_flat, true),
            kept: Kept::default(),
            unrolled: Some(Unrolled {
                flat: unrolled_flat,
                first: start + 1 - seam as isize,
                count,
                raised: swept,
                offset,
            }),
        }
    }

    /// Writes the raised poles of `poles`, held in slots of `width` poles,
    /// to `out`.
    fn run<S: Source>(
        &self,
        direction: &KnotVector,
        coefficients: &[f64],
        poles: S,
        width: impl Width,
        out: &mut [S::Pole],
    ) {
        let (p, by) = (direction.degree(), self.by);
        match &self.unrolled {
            None => sweep(
                p,
                by,
                direction.flat(),
                self.knots.flat(),
                coefficients,
                poles,
                width,
                out,
            ),
            Some(unrolled) => {
                let w = width.get();
                let n = direction.pole_count() as isize;
                let mut unrolled_poles = vec![S::Pole::default(); unrolled.count * w];
                for (i, slot) in unrolled_poles.chunks_exact_mut(w).enumerate() {
                    let k = (unrolled.first + i as isize).rem_euclid(n) as usize;
                    poles.copy_to(k * w, slot);
                }
                let mut swept = vec![S::Pole::default(); unrolled.raised.pole_count() * w];
                sweep(
                    p,
                    by,
                    &unrolled.flat,
                    unrolled.raised.flat(),
                    coefficients,
                    unrolled_poles.as_slice(),
                    width,
                    &mut swept,
                );
                let start = unrolled.offset * w;
                out.copy_from_slice(&swept[start..start + out.len()]);
            }
        }
    }
}

/// Raises the non-periodic direction of degree `p` over `u`, its poles in
/// slots of `width`, into `out`, `new` being the raised flat knots: each
/// Bézier piece raised, and joined to the ones before.
#[allow(clippy::too_many_arguments)] // the parts of A5.9
fn sweep<S: Source>(
    p: usize,
    by: usize,
    u: &[f64],
    new: &[f64],
    coefficients: &[f64],
    poles: S,
    width: impl Width,
    out: &mut [S::Pole],
) {
    let w = width.get();
    let q = p + by;
    let mut scratch = vec![S::Pole::default(); 2 * (p + 1) * w];
    let (left, right) = scratch.split_at_mut((p + 1) * w);
    // Poles written so far.
    let mut written = 0;
    walk(p, u, poles, width, |k, x, y, x_copies, piece| {
        if k == 0 {
            raise(
                &mut out[..(q + 1) * w],
                piece,
                coefficients,
                p,
                by,
                0,
                width,
            );
            written = q + 1;
        } else {
            // The piece's first point is the previous piece's last.
            let joint = written - 1;
            let removals = p - x_copies;
            let piece = (piece, coefficients, p, by);
            join(out, new, joint, removals, (x, y), piece, width, left, right);
            written = joint + q - removals + 1;
        }
    });
    debug_assert_eq!(written * w, out.len());
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

/// Writes the raised slots `from..` of the Bézier `piece` of degree `p`
/// into `out`, as many as it holds.
fn raise<P: Pole>(
    out: &mut [P],
    piece: &[P],
    coefficients: &[f64],
    p: usize,
    by: usize,
    from: usize,
    width: impl Width,
) {
    let w = width.get();
    for slot in 0..out.len() / w {
        let i = from + slot;
        let (lo, hi) = (i.saturating_sub(by), i.min(p));
        let row = &coefficients[i * (p + 1)..(i + 1) * (p + 1)];
        for c in 0..w {
            let mut pole = piece[lo * w + c].scale(row[lo]);
            for j in lo + 1..=hi {
                pole = pole.combine(1.0, piece[j * w + c], row[j]);
            }
            out[slot * w + c] = pole;
        }
    }
}

/// Appends the Bézier piece `(piece, coefficients, p, by)` on `[b, c]`,
/// raised to degree `q = p + by`, to the slots swept so far, whose last
/// one, `poles[joint]`, is the piece's first point, and removes `count` of
/// the `q` copies of `b` between them. `flat` holds the raised knots, the
/// last swept so far being `flat[joint]`.
///
/// With `l_k = flat[joint + 1 - k]`, the slot `joint - i` is the blossom
/// `G(i, 0)` and raised point `k` is `G(0, k)`, where `G(i, k)` is the
/// blossom at `l_i, .., l_1`, `q - i - k` copies of `b` and `k` of `c`.
/// Writing one `b` as the affine combination of `l_(i + 1)` and `c` gives
/// `G(i, k) = a_i G(i, k + 1) + (1 - a_i) G(i + 1, k)`, with `a_i = (b -
/// l_(i + 1)) / (c - l_(i + 1))`. The new slots `G(i, count - i)`, for `0
/// < i < count`, replace the slots `joint - i`; they are solved for from
/// the left side (`i` above `count / 2`) or from the right, each in as few
/// steps as possible, and averaged in the middle. Raised points `count..`
/// follow from `joint` on, unchanged.
#[allow(clippy::too_many_arguments)] // two scratch buffers of the sweep
fn join<P: Pole>(
    poles: &mut [P],
    flat: &[f64],
    joint: usize,
    count: usize,
    (b, c): (f64, f64),
    (piece, coefficients, p, by): (&[P], &[f64], usize, usize),
    width: impl Width,
    left: &mut [P],
    right: &mut [P],
) {
    let w = width.get();
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
            for c in 0..w {
                left[(i - lower) * w + c] = poles[(joint - i) * w + c];
            }
        }
        for level in 1..=r - lower {
            for i in lower..=r - level {
                let a = alpha(i);
                let (t, b) = (1.0 / a, -(1.0 - a) / a);
                let slot = (i - lower) * w;
                for c in 0..w {
                    left[slot + c] = left[slot + c].combine(t, left[slot + w + c], b);
                }
            }
        }
        raise(
            &mut right[..(r - base + 1) * w],
            piece,
            coefficients,
            p,
            by,
            base,
            width,
        );
        for level in 1..=upper {
            let a = alpha(level - 1);
            let (t, b) = (1.0 / (1.0 - a), -a / (1.0 - a));
            for k in base..=r - level {
                let slot = (k - base) * w;
                for c in 0..w {
                    right[slot + c] = right[slot + c].combine(t, right[slot + w + c], b);
                }
            }
        }
        for i in 1..r {
            let target = (joint - i) * w;
            let from_left = (i.max(lower) - lower) * w;
            let from_right = ((r - i).max(base) - base) * w;
            for c in 0..w {
                let (l, r) = (left[from_left + c], right[from_right + c]);
                poles[target + c] = if i < lower {
                    r
                } else if i > upper {
                    l
                } else {
                    l.combine(0.5, r, 0.5)
                };
            }
        }
    }
    // Without a removal the joint stays as swept.
    let from = r.max(1);
    let start = joint + from - r;
    raise(
        &mut poles[start * w..(start + q + 1 - from) * w],
        piece,
        coefficients,
        p,
        by,
        from,
        width,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
