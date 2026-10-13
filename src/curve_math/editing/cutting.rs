//! Cutting one direction into clamped pieces at increasing parameters,
//! and reversing it.

use std::borrow::Cow;
use std::ops::Range;

use super::{Lanes, Pole, Source, Weighted, refine};
use crate::Point3D;
use crate::curve_math::bspline::KnotVector;
use crate::curves::BSplineSplitError;

/// Checks the range `[first, last]` of a piece of `direction`. A periodic
/// direction accepts any start, so a piece may cross the seam, but no more
/// than one period, with the same slack as [`crate::TrimmedCurve3D::new`].
pub(crate) fn check_piece(
    direction: &KnotVector,
    first: f64,
    last: f64,
) -> Result<(), BSplineSplitError> {
    if !first.is_finite() || !last.is_finite() {
        return Err(BSplineSplitError::NonFinite);
    }
    if first >= last {
        return Err(BSplineSplitError::EmptyRange);
    }
    let (lower, upper) = direction.bounds();
    if direction.is_periodic() {
        if last - first > (upper - lower) * (1.0 + 1e-12) {
            return Err(BSplineSplitError::ExceedsPeriod);
        }
    } else if first < lower || last > upper {
        return Err(BSplineSplitError::OutOfRange);
    }
    Ok(())
}

/// Checks a parameter `u` at which to split `direction`: strictly within
/// its bounds, so neither piece is empty.
pub(crate) fn check_split(direction: &KnotVector, u: f64) -> Result<(), BSplineSplitError> {
    let (first, last) = direction.bounds();
    if !u.is_finite() {
        return Err(BSplineSplitError::NonFinite);
    }
    if u < first || u > last {
        return Err(BSplineSplitError::OutOfRange);
    }
    if u == first || u == last {
        return Err(BSplineSplitError::EmptyRange);
    }
    Ok(())
}

/// Whether `direction` is clamped, both end knots of multiplicity
/// `degree + 1`, with bounds exactly `(first, last)`: its piece over that
/// range is the direction itself.
pub(crate) fn is_clamped_to(direction: &KnotVector, first: f64, last: f64) -> bool {
    let (p, flat) = (direction.degree(), direction.flat());
    let end = flat.len() - 1;
    !direction.is_periodic()
        && (flat[0], flat[p]) == (first, first)
        && (flat[end - p], flat[end]) == (last, last)
}

/// A piece of a cut direction: its clamped knots, and its poles laid out
/// as the poles cut, with their weights if rational.
pub(crate) struct Piece {
    pub(crate) knots: KnotVector,
    pub(crate) points: Vec<Point3D>,
    pub(crate) weights: Option<Vec<f64>>,
}

/// Cuts a direction into clamped pieces at the increasing `cuts` (two or
/// more, checked by [`check_piece`] or [`check_split`]), piece `j` running
/// from cut `j` to cut `j + 1` with those exact bounds; a periodic
/// direction's pieces may cross the seam.
///
/// Every cut is raised to multiplicity `degree` in one refinement of the
/// knot spans from the first cut to the last (see [`Stretch`]), a cut
/// already of that multiplicity being left alone, and two pieces share
/// the pole at their common cut. A rational piece keeps the poles no
/// insertion touched bit for bit.
pub(crate) fn cut(
    direction: &KnotVector,
    points: &[Point3D],
    weights: Option<&[f64]>,
    lanes: Lanes,
    cuts: &[f64],
) -> Vec<Piece> {
    let pieces = 0..cuts.len() - 1;
    match weights {
        None => {
            let stretch = Stretch::new(direction, points, lanes, cuts);
            pieces
                .map(|j| Piece {
                    knots: stretch.knots(j),
                    points: stretch.poles(j),
                    weights: None,
                })
                .collect()
        }
        Some(weights) => {
            let stretch = Stretch::new(direction, Weighted { points, weights }, lanes, cuts);
            pieces
                .map(|j| stretch.rational_piece(j, points, weights))
                .collect()
        }
    }
}

/// A direction cut at increasing parameters: the stretch of its knot
/// spans from the first cut to the last, copied out with the poles
/// weighting them (a periodic direction's from its tiled knots and wrapped
/// poles, so a stretch may cross the seam) and refined until every cut is
/// a knot of multiplicity `degree` or more.
struct Stretch<P> {
    degree: usize,
    /// The refined stretch's flat knots.
    flat: Vec<f64>,
    /// The refined stretch's poles: `rows` rows of `count` slots of
    /// `width` poles, as [`super::refine`] lays them out.
    poles: Vec<P>,
    rows: usize,
    width: usize,
    count: usize,
    cuts: Vec<CutKnot>,
    /// The direction's slot that the unrefined stretch's slot `t` is,
    /// `first_slot + t`, wrapped on a periodic direction of `pole_count`
    /// slots.
    first_slot: isize,
    pole_count: usize,
}

/// A cut of a [`Stretch`].
struct CutKnot {
    /// The parameter cut at, a piece's exact bound.
    value: f64,
    /// The indices of its first and last copies in the refined knots.
    first_copy: usize,
    last_copy: usize,
    /// The unrefined stretch's knots at or below it, and its copies there.
    at_or_below: usize,
    copies: usize,
}

impl<P: Pole> Stretch<P> {
    /// The stretch of `direction` from the first of `cuts` to the last,
    /// its `poles` laid out by `lanes`, refined.
    fn new<S: Source<Pole = P>>(
        direction: &KnotVector,
        poles: S,
        lanes: Lanes,
        cuts: &[f64],
    ) -> Stretch<P> {
        let p = direction.degree();
        let n = direction.pole_count();
        let (rows, width) = if lanes.consecutive {
            (lanes.count, 1)
        } else {
            (1, lanes.count)
        };
        let (old, first_slot) = stretch_knots(direction, cuts[0], cuts[cuts.len() - 1]);
        let old_count = old.len() - p - 1;
        let mut stretch = Vec::with_capacity(rows * old_count * width);
        for row in 0..rows {
            let slot = |t: usize| (row * n + t) * width;
            if direction.is_periodic() {
                for t in 0..old_count {
                    let q = (first_slot + t as isize).rem_euclid(n as isize) as usize;
                    poles.extend_to(slot(q)..slot(q + 1), &mut stretch);
                }
            } else {
                let q = first_slot as usize;
                poles.extend_to(slot(q)..slot(q + old_count), &mut stretch);
            }
        }
        let at = snapped(p, &old, cuts);
        let mut insert = Vec::with_capacity(at.len() * p);
        let mut found = Vec::with_capacity(at.len());
        for &c in &at {
            let at_or_below = old.partition_point(|&k| k <= c);
            let copies = at_or_below - old[..at_or_below].partition_point(|&k| k < c);
            insert.extend(std::iter::repeat_n(c, p.saturating_sub(copies)));
            found.push((at_or_below, copies));
        }
        let (flat, poles) = if insert.is_empty() {
            (old.into_owned(), stretch)
        } else {
            let mut new = Vec::with_capacity(old.len() + insert.len());
            let (mut i, mut j) = (0, 0);
            while i < old.len() || j < insert.len() {
                if j == insert.len() || (i < old.len() && old[i] <= insert[j]) {
                    new.push(old[i]);
                    i += 1;
                } else {
                    new.push(insert[j]);
                    j += 1;
                }
            }
            // The stretch's lanes are laid out as the direction's.
            let (refined, _) = refine(p, &old, &new, &insert, stretch.as_slice(), lanes, false);
            (new, refined)
        };
        let cuts = cuts
            .iter()
            .zip(&at)
            .zip(found)
            .map(|((&value, &c), (at_or_below, copies))| {
                let first_copy = flat.partition_point(|&k| k < c);
                let last_copy = flat.partition_point(|&k| k <= c) - 1;
                CutKnot {
                    value,
                    first_copy,
                    last_copy,
                    at_or_below,
                    copies,
                }
            })
            .collect();
        Stretch {
            degree: p,
            count: flat.len() - p - 1,
            flat,
            poles,
            rows,
            width,
            cuts,
            first_slot,
            pole_count: n,
        }
    }

    /// The clamped knots of piece `j`: its two cuts, of multiplicity
    /// `degree + 1`, around the knots between them.
    fn knots(&self, j: usize) -> KnotVector {
        let p = self.degree;
        let (a, b) = (&self.cuts[j], &self.cuts[j + 1]);
        let interior = &self.flat[a.last_copy + 1..b.first_copy];
        let mut flat = Vec::with_capacity(interior.len() + 2 * p + 2);
        flat.extend(std::iter::repeat_n(a.value, p + 1));
        flat.extend_from_slice(interior);
        flat.extend(std::iter::repeat_n(b.value, p + 1));
        KnotVector::from_flat(p, flat, false)
    }

    /// The refined stretch's slots holding piece `j`'s poles: from the
    /// last copy of cut `j` to the first of cut `j + 1`.
    fn slots(&self, j: usize) -> Range<usize> {
        self.cuts[j].last_copy - self.degree..self.cuts[j + 1].first_copy
    }

    /// The poles of piece `j`, laid out as the poles cut.
    fn poles(&self, j: usize) -> Vec<P> {
        let slots = self.slots(j);
        let mut out = Vec::with_capacity(self.rows * slots.len() * self.width);
        for row in 0..self.rows {
            let at = |slot: usize| (row * self.count + slot) * self.width;
            out.extend_from_slice(&self.poles[at(slots.start)..at(slots.end)]);
        }
        out
    }

    /// The slots of piece `j` no insertion touched, its `leading` first
    /// slots being new: the direction's slot each of them is.
    ///
    /// Raising a cut with `s` copies in span `k` to `degree` copies keeps
    /// every pole up to `k - degree` and from `k - s` (The NURBS Book,
    /// A5.1), so a piece keeps the poles from the first kept right of its
    /// first cut to the last kept left of its second; if those cross, the
    /// insertions at both cuts touch every pole of the piece.
    fn kept(&self, j: usize) -> (usize, impl Iterator<Item = usize> + use<P>) {
        let p = self.degree as isize;
        let (a, b) = (&self.cuts[j], &self.cuts[j + 1]);
        let raised = |cut: &CutKnot| cut.copies.min(self.degree) as isize;
        let first = a.at_or_below as isize - 1 - raised(a);
        let last = (b.at_or_below - b.copies) as isize - 1 - p + raised(b);
        let (start, n) = (self.first_slot, self.pole_count as isize);
        let slots = (first..=last).map(move |t| (start + t).rem_euclid(n) as usize);
        ((p - raised(a)) as usize, slots)
    }
}

impl Stretch<[f64; 4]> {
    /// Piece `j` of a rational direction: the poles no insertion touched
    /// copied from the direction's `points` and `weights`, the others
    /// projected from homogeneous form.
    fn rational_piece(&self, j: usize, points: &[Point3D], weights: &[f64]) -> Piece {
        let slots = self.slots(j);
        let (leading, kept) = self.kept(j);
        let kept: Vec<usize> = kept.collect();
        let size = self.rows * slots.len() * self.width;
        let (mut new_points, mut new_weights) =
            (Vec::with_capacity(size), Vec::with_capacity(size));
        for row in 0..self.rows {
            for (i, slot) in slots.clone().enumerate() {
                let kept_slot = i.checked_sub(leading).and_then(|i| kept.get(i));
                for c in 0..self.width {
                    match kept_slot {
                        Some(&q) => {
                            let at = (row * self.pole_count + q) * self.width + c;
                            new_points.push(points[at]);
                            new_weights.push(weights[at]);
                        }
                        None => {
                            let [x, y, z, w] =
                                self.poles[(row * self.count + slot) * self.width + c];
                            let inverse = 1.0 / w;
                            new_points.push(Point3D::new(x * inverse, y * inverse, z * inverse));
                            new_weights.push(w);
                        }
                    }
                }
            }
        }
        Piece {
            knots: self.knots(j),
            points: new_points,
            weights: Some(new_weights),
        }
    }
}

/// The flat knots of the stretch of `direction` from `a` to `b`, from
/// `degree` knots before the span of `a` to `degree + 1` after the span
/// ending at `b`, and the direction's slot weighting its first function:
/// a periodic direction's tiled, so the stretch may cross the seam.
fn stretch_knots(direction: &KnotVector, a: f64, b: f64) -> (Cow<'_, [f64]>, isize) {
    let (p, flat, n) = (direction.degree(), direction.flat(), direction.pole_count());
    if !direction.is_periodic() {
        let ia = flat[..n].partition_point(|&k| k <= a) - 1;
        let ib = flat[..=n].partition_point(|&k| k < b) - 1;
        return (Cow::Borrowed(&flat[ia - p..=ib + p + 1]), (ia - p) as isize);
    }
    let seam = flat.len() - n;
    let (block, first) = (&flat[..n], flat[0]);
    let period = flat[flat.len() - 1] - first;
    let len = n as isize;
    let knot = |t: isize| match flat.get(t as usize) {
        Some(&k) if t >= 0 => k,
        _ => block[t.rem_euclid(len) as usize] + period * t.div_euclid(len) as f64,
    };
    // The last tiled index whose knot is at or below (`strict`: below) `x`,
    // from a guess corrected for rounding.
    let last_index = |x: f64, strict: bool| {
        let before = |k: f64| if strict { k < x } else { k <= x };
        let tile = ((x - first) / period).floor();
        let mut t = tile as isize * len
            + block.partition_point(|&k| before(k + period * tile)) as isize
            - 1;
        while before(knot(t + 1)) {
            t += 1;
        }
        while !before(knot(t)) {
            t -= 1;
        }
        t
    };
    let (ia, ib) = (last_index(a, false), last_index(b, true));
    let p = p as isize;
    let knots = (ia - p..=ib + p + 1).map(knot).collect();
    // Pole `q` weights the function starting at tiled knot
    // `q + seam - degree - 1`.
    (Cow::Owned(knots), ia - p - (seam as isize - p - 1))
}

/// The knots the increasing `cuts` raise in the stretch knots `old`: a
/// cut within rounding of a knot of the stretch's range, such as a seam
/// one period on computed by the caller, is that knot rather than a sliver
/// span away, unless snapping would make two cuts meet.
fn snapped(degree: usize, old: &[f64], cuts: &[f64]) -> Vec<f64> {
    let near = 4.0 * f64::EPSILON * old[0].abs().max(old[old.len() - 1].abs());
    let range = old[degree]..=old[old.len() - 1 - degree];
    let at: Vec<f64> = cuts
        .iter()
        .map(|&c| {
            let i = old.partition_point(|&k| k < c);
            let snap =
                |j: usize| Some(old[j]).filter(|&k| (k - c).abs() <= near && range.contains(&k));
            snap(i).or_else(|| snap(i.checked_sub(1)?)).unwrap_or(c)
        })
        .collect();
    if at.windows(2).any(|w| w[0] >= w[1]) {
        cuts.to_vec()
    } else {
        at
    }
}

/// `direction` traversed the other way over the same bounds: knot `k`
/// becomes `first + last - k`, the bounds trading places exactly.
pub(crate) fn reversed_knots(direction: &KnotVector) -> KnotVector {
    let (first, last) = direction.bounds();
    let mirror = |k: f64| {
        if k == first {
            last
        } else if k == last {
            first
        } else {
            first + last - k
        }
    };
    let flat = direction.flat().iter().rev().map(|&k| mirror(k)).collect();
    KnotVector::from_flat(direction.degree(), flat, direction.is_periodic())
}

/// The slot of `direction` that slot `i` of the reversed direction takes:
/// the mirror image, which on a periodic direction is also rotated so
/// that slot `i` keeps weighting the function starting at tiled knot `i +
/// seam - degree - 1`, `seam` being the seam multiplicity.
pub(crate) fn reversed_slot(direction: &KnotVector, i: usize) -> usize {
    let n = direction.pole_count();
    if direction.is_periodic() {
        let seam = (direction.flat().len() - n) as isize;
        (direction.degree() as isize - seam - i as isize).rem_euclid(n as isize) as usize
    } else {
        n - 1 - i
    }
}
