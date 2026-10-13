//! The Bézier pieces of one direction (after The NURBS Book, A5.6): each
//! knot span's poles once its end knots have `degree` copies, found span
//! after span by inserting the next knot until it has them.

use std::borrow::Cow;

use super::{Lanes, One, Pole, Source, Width};
use crate::curve_math::bspline::{KnotVector, LARGE};

/// The Bézier pieces of `knots` and its poles over its range, a periodic
/// direction's from its lower bound: the knots at which pieces meet (see
/// [`KnotVector::breaks`]) and the pieces in order, piece `k` holding `degree +
/// 1` poles of every lane, the lanes of a piece laid out as those of
/// `poles`.
pub(crate) fn bezier_pieces<S: Source>(
    knots: &KnotVector,
    poles: S,
    lanes: Lanes,
) -> (Vec<f64>, Vec<S::Pole>) {
    let p = knots.degree();
    let breaks = knots.breaks();
    let mut out = vec![S::Pole::default(); (breaks.len() - 1) * (p + 1) * lanes.count];
    if lanes.consecutive {
        let lane_n = poles.len() / lanes.count;
        let piece_len = (p + 1) * lanes.count;
        let mut walker = Walker::new(knots, 1);
        for lane in 0..lanes.count {
            let lane_poles = poles.slice(lane * lane_n..(lane + 1) * lane_n);
            walker.for_each_piece(lane_poles, One, |k, _, _, piece| {
                let at = k * piece_len + lane * (p + 1);
                out[at..at + p + 1].copy_from_slice(piece);
            });
        }
    } else {
        let w = lanes.count;
        let piece_len = (p + 1) * w;
        Walker::new(knots, w).for_each_piece(poles, w, |k, _, _, piece| {
            out[k * piece_len..(k + 1) * piece_len].copy_from_slice(piece);
        });
    }
    (breaks, out)
}

/// Walks the Bézier pieces of one direction, again and again over poles
/// in slots of the same width, with the same scratch space.
pub(crate) struct Walker<'a, P> {
    degree: usize,
    /// The flat knots walked: a periodic direction's unrolled.
    flat: Cow<'a, [f64]>,
    /// The periodic direction's pole count, if periodic.
    period: Option<usize>,
    scratch: Vec<P>,
    /// A periodic direction's poles, unrolled.
    unrolled: Vec<P>,
}

impl<'a, P: Pole> Walker<'a, P> {
    /// A walker over `knots` for poles in slots of `width`.
    pub(crate) fn new(knots: &'a KnotVector, width: usize) -> Walker<'a, P> {
        let p = knots.degree();
        let periodic = knots.is_periodic();
        Walker {
            degree: p,
            flat: match periodic {
                true => Cow::Owned(unroll(knots)),
                false => Cow::Borrowed(knots.flat()),
            },
            period: periodic.then(|| knots.pole_count()),
            scratch: vec![P::default(); 2 * (p + 1) * width],
            unrolled: Vec::new(),
        }
    }

    /// Visits the Bézier pieces of `poles` (in slots of `width`, as given
    /// to [`Walker::new`]): `visit(k, x, y, piece)` for piece `k` over the
    /// span `[x, y]`, its `degree + 1` slots.
    pub(crate) fn for_each_piece<S: Source<Pole = P>>(
        &mut self,
        poles: S,
        width: impl Width,
        mut visit: impl FnMut(usize, f64, f64, &[P]),
    ) {
        let (p, flat, scratch) = (self.degree, &self.flat[..], &mut self.scratch[..]);
        match self.period {
            None => walk(p, flat, poles, width, scratch, |k, x, y, _, piece| {
                visit(k, x, y, piece)
            }),
            Some(n) => {
                // The unrolled direction's pole `i` is the periodic pole `i`
                // wrapped.
                let w = width.get();
                self.unrolled.resize((flat.len() - p - 1) * w, P::default());
                for (i, slot) in self.unrolled.chunks_exact_mut(w).enumerate() {
                    poles.copy_to((i % n) * w, slot);
                }
                let unrolled = &self.unrolled[..];
                walk(p, flat, unrolled, width, scratch, |k, x, y, _, piece| {
                    visit(k, x, y, piece)
                });
            }
        }
    }
}

/// The flat knots of a periodic direction's period unrolled into a
/// non-periodic direction over the same range, starting at the last copy
/// of the seam: pole `i` of the unrolled direction is periodic pole `i`
/// wrapped.
fn unroll(knots: &KnotVector) -> Vec<f64> {
    let (p, n, flat) = (knots.degree(), knots.pole_count(), knots.flat());
    let (first, last) = knots.bounds();
    let period = last - first;
    let seam = knots.multiplicities()[0] as isize;
    // Flat knot `j` of the period tiled endlessly, as the evaluator sees
    // it; one period is `flat` without its trailing seam run.
    let tiled = |j: isize| {
        let (k, shift) = (j.rem_euclid(n as isize) as usize, j.div_euclid(n as isize));
        match shift {
            0 => flat[k],
            1 if flat[k] == first => last,
            _ => flat[k] + period * shift as f64,
        }
    };
    let start = seam - 1 - p as isize;
    (0..n + 2 * p + 1 - seam as usize + 1)
        .map(|i| tiled(start + i as isize))
        .collect()
}

/// Visits the Bézier pieces of the non-periodic direction of degree `p`
/// over the flat knots `u`, its poles in slots of `width`, over its range
/// `[u[p], u[n]]`: `visit(k, x, y, x_copies, piece)` for piece `k` over
/// the span `[x, y]`, its `p + 1` slots, told how many copies of `x` the
/// direction has (none for the first piece's). `scratch` holds `2 (p + 1)`
/// slots.
pub(super) fn walk<S: Source>(
    p: usize,
    u: &[f64],
    poles: S,
    width: impl Width,
    scratch: &mut [S::Pole],
    mut visit: impl FnMut(usize, f64, f64, usize, &[S::Pole]),
) {
    let w = width.get();
    let n = poles.len() / w;
    let (lo, hi) = (u[p], u[n]);
    let mut s = p;
    while u[s + 1] <= lo {
        s += 1;
    }
    let (piece, next) = scratch[..2 * (p + 1) * w].split_at_mut((p + 1) * w);
    // `piece[i]` is the blossom at `x^(p - i)` and the `i` knots after the
    // span start `x`: the span's poles once `x` has `p` copies.
    poles.copy_to((s - p) * w, piece);
    clamp_start(p, &u[s + 1 - p..s + 1 + p], piece, next, width);
    let mut alphas = [0.0; LARGE];
    let mut x_copies = 0;
    for k in 0.. {
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
                for c in 0..w {
                    piece[k * w + c] = piece[k * w + c].combine(a, piece[(k - 1) * w + c], 1.0 - a);
                }
            }
            for c in 0..w {
                next[(inserts - j) * w + c] = piece[p * w + c];
            }
        }
        visit(k, x, y, x_copies, piece);
        if y == hi {
            return;
        }
        let s_next = s + mult;
        piece[..inserts * w].copy_from_slice(&next[..inserts * w]);
        poles.copy_to(s * w, &mut piece[inserts * w..]);
        x_copies = mult;
        s = s_next;
    }
}

/// Brings the first span's `p + 1` slots to the form [`walk`] expects,
/// `x` (`knots[p - 1]`, the span start) repeated `p` times: `knots` are
/// the `2p` knots around the span. `spare` holds as many slots as
/// `piece`.
///
/// Inserting `x` turns the pole at `knots[i..i + p]` into one at `x` and
/// `knots[i + 1..i + p]`; after `r` rounds the last pole is the blossom at
/// `x^r` and `knots[p..2p - r]`.
fn clamp_start<P: Pole>(
    p: usize,
    knots: &[f64],
    piece: &mut [P],
    spare: &mut [P],
    width: impl Width,
) {
    let w = width.get();
    let x = knots[p - 1];
    if knots[0] == x {
        return;
    }
    for c in 0..w {
        spare[p * w + c] = piece[p * w + c];
    }
    for r in 1..=p {
        for i in (r..=p).rev() {
            let a = (x - knots[i - 1]) / (knots[i + p - r] - knots[i - 1]);
            for c in 0..w {
                piece[i * w + c] = piece[(i - 1) * w + c].combine(1.0 - a, piece[i * w + c], a);
            }
        }
        for c in 0..w {
            spare[(p - r) * w + c] = piece[p * w + c];
        }
    }
    piece.copy_from_slice(&spare[..piece.len()]);
}
