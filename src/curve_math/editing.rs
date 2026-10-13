//! Editing one direction of a B-spline: knot insertion, degree elevation
//! and Bézier pieces of the poles that direction weights, a curve's or
//! every row of a surface's.
//!
//! The algorithms are generic over the [`Pole`] they combine: points for a
//! nonrational shape, homogeneous points for a rational one. [`Lanes`]
//! says where the poles of each independent row sit, so one pass serves a
//! curve and both directions of a surface, every row with the same
//! coefficients.

mod bezier;
mod elevation;

use bezier::bezier_pieces;
pub(crate) use elevation::Elevation;

use super::bspline::KnotVector;
use crate::Point3D;
use crate::curves::KnotInsertionError;

/// A pole an editing algorithm combines linearly: a point of a
/// nonrational shape, or a homogeneous point `(w x, w y, w z, w)` of a
/// rational one.
pub(crate) trait Pole: Copy + Default {
    /// `a self + b other`.
    fn combine(self, a: f64, other: Self, b: f64) -> Self;

    /// `a self`.
    fn scale(self, a: f64) -> Self;
}

impl Pole for Point3D {
    #[inline]
    fn combine(self, a: f64, other: Self, b: f64) -> Self {
        Point3D::new(
            a * self.x + b * other.x,
            a * self.y + b * other.y,
            a * self.z + b * other.z,
        )
    }

    #[inline]
    fn scale(self, a: f64) -> Self {
        Point3D::new(a * self.x, a * self.y, a * self.z)
    }
}

impl Pole for [f64; 4] {
    #[inline]
    fn combine(self, a: f64, other: Self, b: f64) -> Self {
        std::array::from_fn(|c| a * self[c] + b * other[c])
    }

    #[inline]
    fn scale(self, a: f64) -> Self {
        self.map(|c| a * c)
    }
}

/// The poles an edit reads, by index: a slice of them, or a rational
/// shape's points and weights read as homogeneous points, so that they
/// need no converted copy.
pub(crate) trait Source: Copy {
    type Pole: Pole;

    fn len(self) -> usize;

    fn get(self, i: usize) -> Self::Pole;

    /// The poles `range`.
    fn slice(self, range: std::ops::Range<usize>) -> Self;

    /// Copies the poles from `start` into `out`.
    #[inline]
    fn copy_to(self, start: usize, out: &mut [Self::Pole]) {
        for (k, out) in out.iter_mut().enumerate() {
            *out = self.get(start + k);
        }
    }
}

impl<P: Pole> Source for &[P] {
    type Pole = P;

    #[inline]
    fn len(self) -> usize {
        <[P]>::len(self)
    }

    #[inline]
    fn get(self, i: usize) -> P {
        self[i]
    }

    #[inline]
    fn slice(self, range: std::ops::Range<usize>) -> Self {
        &self[range]
    }

    /// A few poles, as most runs between inserted knots are, are cheaper
    /// copied one by one.
    #[inline]
    fn copy_to(self, start: usize, out: &mut [P]) {
        let poles = &self[start..start + out.len()];
        if out.len() <= 4 {
            for (out, &pole) in out.iter_mut().zip(poles) {
                *out = pole;
            }
        } else {
            out.copy_from_slice(poles);
        }
    }
}

/// A rational shape's points and weights, read as homogeneous points
/// `(w x, w y, w z, w)`.
#[derive(Clone, Copy)]
struct Weighted<'a> {
    points: &'a [Point3D],
    weights: &'a [f64],
}

impl Source for Weighted<'_> {
    type Pole = [f64; 4];

    #[inline]
    fn len(self) -> usize {
        self.points.len()
    }

    #[inline]
    fn get(self, i: usize) -> [f64; 4] {
        let (p, w) = (self.points[i], self.weights[i]);
        [w * p.x, w * p.y, w * p.z, w]
    }

    #[inline]
    fn slice(self, range: std::ops::Range<usize>) -> Self {
        Weighted {
            points: &self.points[range.clone()],
            weights: &self.weights[range],
        }
    }
}

/// The points and weights of homogeneous poles.
fn from_homogeneous(
    edited: &Edited<[f64; 4]>,
    points: &[Point3D],
    weights: &[f64],
    lanes: Lanes,
) -> (Vec<Point3D>, Vec<f64>) {
    // Consecutive lanes are rows of their own; interleaved ones one row of
    // slots, a slot holding every lane's pole.
    let (rows, width) = if lanes.consecutive {
        (lanes.count, 1)
    } else {
        (1, lanes.count)
    };
    let (old_row, new_row) = (points.len() / rows, edited.poles.len() / rows);
    let (leading, trailing) = (edited.kept.leading * width, edited.kept.trailing * width);
    let mut new_points = Vec::with_capacity(edited.poles.len());
    let mut new_weights = Vec::with_capacity(edited.poles.len());
    for row in 0..rows {
        let (old, new) = (
            row * old_row..(row + 1) * old_row,
            row * new_row..(row + 1) * new_row,
        );
        let (old_end, new_end) = (old.end - trailing, new.end - trailing);
        new_points.extend_from_slice(&points[old.start..old.start + leading]);
        new_weights.extend_from_slice(&weights[old.start..old.start + leading]);
        let changed = &edited.poles[new.start + leading..new_end];
        new_points.extend(changed.iter().map(|&[x, y, z, w]| {
            let inverse = 1.0 / w;
            Point3D::new(x * inverse, y * inverse, z * inverse)
        }));
        new_weights.extend(changed.iter().map(|h| h[3]));
        new_points.extend_from_slice(&points[old_end..old.end]);
        new_weights.extend_from_slice(&weights[old_end..old.end]);
    }
    (new_points, new_weights)
}

/// An edited direction and its poles, laid out as the poles edited.
pub(crate) struct Edited<P> {
    pub(crate) knots: KnotVector,
    pub(crate) poles: Vec<P>,
    /// The poles at either end of each lane the edit left as they were.
    pub(crate) kept: Kept,
}

/// How many poles at the start and at the end of each lane an edit left
/// as they were: copied back as they are, a rational shape's kept poles
/// cost no division and keep every bit.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Kept {
    pub(crate) leading: usize,
    pub(crate) trailing: usize,
}

/// An edit of one direction, run on whichever poles the shape has.
pub(crate) trait Edit {
    type Error;

    /// The edited direction and its poles, laid out as `poles`.
    fn apply<S: Source>(
        &self,
        direction: &KnotVector,
        poles: S,
        lanes: Lanes,
    ) -> Result<Edited<S::Pole>, Self::Error>;
}

/// The edited direction, poles and weights: `edit` run on the points, or
/// on homogeneous points if there are `weights`.
#[allow(clippy::type_complexity)] // the parts a B-spline stores
pub(crate) fn edit<E: Edit>(
    edit: &E,
    direction: &KnotVector,
    points: &[Point3D],
    weights: Option<&[f64]>,
    lanes: Lanes,
) -> Result<(KnotVector, Vec<Point3D>, Option<Vec<f64>>), E::Error> {
    match weights {
        None => {
            let edited = edit.apply(direction, points, lanes)?;
            Ok((edited.knots, edited.poles, None))
        }
        Some(weights) => {
            let edited = edit.apply(direction, Weighted { points, weights }, lanes)?;
            let (points, weights) = from_homogeneous(&edited, points, weights, lanes);
            Ok((edited.knots, points, Some(weights)))
        }
    }
}

/// The Bézier pieces of a curve's direction (see [`bezier_pieces`]): the
/// knots at which they meet, their points, `degree + 1` per piece, and
/// their weights if there are `weights`.
#[allow(clippy::type_complexity)] // the parts of the pieces
pub(crate) fn curve_pieces(
    direction: &KnotVector,
    points: &[Point3D],
    weights: Option<&[f64]>,
) -> (Vec<f64>, Vec<Point3D>, Option<Vec<f64>>) {
    match weights {
        None => {
            let (breaks, points) = bezier_pieces(direction, points, Lanes::CURVE);
            (breaks, points, None)
        }
        Some(weights) => {
            let (breaks, poles) =
                bezier_pieces(direction, Weighted { points, weights }, Lanes::CURVE);
            let (points, weights) = cartesian(&poles);
            (breaks, points, Some(weights))
        }
    }
}

/// The Bézier patches of a surface over `u` and `v`, its rows of poles
/// along `v`: the knots at which they meet in each direction, and their
/// points and weights (if there are `weights`), patch `(i, j)` (the `i`-th
/// `u` span, `j`-th `v` span) the `(i, j)`-th block of `(u degree + 1)
/// (v degree + 1)` poles, row by row.
#[allow(clippy::type_complexity)] // the parts of the patches
pub(crate) fn surface_patches(
    u: &KnotVector,
    v: &KnotVector,
    points: &[Point3D],
    weights: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>, Vec<Point3D>, Option<Vec<f64>>) {
    match weights {
        None => {
            let (u_breaks, v_breaks, points) = patches(u, v, points);
            (u_breaks, v_breaks, points, None)
        }
        Some(weights) => {
            let (u_breaks, v_breaks, poles) = patches(u, v, Weighted { points, weights });
            let (points, weights) = cartesian(&poles);
            (u_breaks, v_breaks, points, Some(weights))
        }
    }
}

/// [`surface_patches`] of `poles`: the strips of `u` pieces, then each
/// strip's `v` pieces.
fn patches<S: Source>(
    u: &KnotVector,
    v: &KnotVector,
    poles: S,
) -> (Vec<f64>, Vec<f64>, Vec<S::Pole>) {
    let (u_breaks, strips) = bezier_pieces(u, poles, Lanes::columns(v.pole_count()));
    let strip_len = (u.degree() + 1) * v.pole_count();
    let strip_count = u_breaks.len() - 1;
    let mut v_breaks = Vec::new();
    let mut out = Vec::new();
    for strip in strips.chunks_exact(strip_len) {
        let (breaks, patches) = bezier_pieces(v, strip, Lanes::rows(u.degree() + 1));
        if out.is_empty() {
            out.reserve_exact(strip_count * patches.len());
        }
        out.extend_from_slice(&patches);
        v_breaks = breaks;
    }
    (u_breaks, v_breaks, out)
}

/// The points and weights of homogeneous poles.
fn cartesian(poles: &[[f64; 4]]) -> (Vec<Point3D>, Vec<f64>) {
    let points = poles
        .iter()
        .map(|&[x, y, z, w]| {
            let inverse = 1.0 / w;
            Point3D::new(x * inverse, y * inverse, z * inverse)
        })
        .collect();
    (points, poles.iter().map(|h| h[3]).collect())
}

/// Inserting the non-decreasing knots: see [`insert_knots`].
pub(crate) struct Insertion<'a>(pub(crate) &'a [f64]);

impl Edit for Insertion<'_> {
    type Error = KnotInsertionError;

    fn apply<S: Source>(
        &self,
        direction: &KnotVector,
        poles: S,
        lanes: Lanes,
    ) -> Result<Edited<S::Pole>, KnotInsertionError> {
        insert_knots(direction, poles, lanes, self.0)
    }
}

/// Where the poles of `count` independent rows sit in a flat buffer, each
/// row holding a direction's poles: one row for a curve; for a surface
/// with poles stored row by row (`u` major), one lane per column in the
/// `u` direction, one per row in the `v` direction.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Lanes {
    pub(crate) count: usize,
    /// Whether a lane's poles are consecutive (lane `l` of `n` poles at
    /// `l n..(l + 1) n`), or interleaved (pole `i` of every lane at
    /// `i count..(i + 1) count`).
    consecutive: bool,
}

impl Lanes {
    /// A curve's poles.
    pub(crate) const CURVE: Lanes = Lanes {
        count: 1,
        consecutive: true,
    };

    /// The `u` direction of a surface whose rows hold `v_count` poles.
    pub(crate) fn columns(v_count: usize) -> Lanes {
        Lanes {
            count: v_count,
            consecutive: false,
        }
    }

    /// The `v` direction of a surface with `u_count` rows.
    pub(crate) fn rows(u_count: usize) -> Lanes {
        Lanes {
            count: u_count,
            consecutive: true,
        }
    }

    /// Index of pole `i` of lane `lane`, each lane holding `poles` poles.
    #[inline]
    fn index(self, lane: usize, i: usize, poles: usize) -> usize {
        if self.consecutive {
            lane * poles + i
        } else {
            i * self.count + lane
        }
    }
}

/// Inserts the non-decreasing `insert` knots into a direction in one
/// pass, leaving the shape unchanged: the new direction, and its poles
/// laid out as `poles`. A value equal to an existing knot raises its
/// multiplicity, any other value becomes a new knot. On a periodic
/// direction both parameter bounds name the seam knot.
fn insert_knots<S: Source>(
    knots: &KnotVector,
    poles: S,
    lanes: Lanes,
    insert: &[f64],
) -> Result<Edited<S::Pole>, KnotInsertionError> {
    let (first, last) = knots.bounds();
    if insert.iter().any(|x| !x.is_finite()) {
        return Err(KnotInsertionError::NonFiniteKnot);
    }
    if insert.windows(2).any(|w| w[1] < w[0]) {
        return Err(KnotInsertionError::KnotsNotSorted);
    }
    if insert
        .first()
        .zip(insert.last())
        .is_some_and(|(&lo, &hi)| lo < first || hi > last)
    {
        return Err(KnotInsertionError::KnotOutOfRange);
    }
    if insert.is_empty() {
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
    let (degree, periodic) = (knots.degree(), knots.is_periodic());
    let seam = |x: f64| periodic && (x == first || x == last);
    let seam_count = insert.iter().filter(|&&x| seam(x)).count();
    let interior: &[f64] = if seam_count == 0 {
        insert
    } else {
        // Sorted, so the seam values are the ones at either end.
        let leading = insert.iter().take_while(|&&x| x == first).count();
        &insert[leading..leading + insert.len() - seam_count]
    };
    let old_flat = knots.flat();
    let mut new_flat = Vec::with_capacity(old_flat.len() + insert.len());
    if seam_count > 0 {
        new_flat.extend(std::iter::repeat_n(first, seam_count));
    }
    let mut copied = 0;
    for &x in interior {
        let mut at = copied;
        while at < old_flat.len() && old_flat[at] <= x {
            at += 1;
        }
        append(&mut new_flat, &old_flat[copied..at]);
        new_flat.push(x);
        copied = at;
    }
    append(&mut new_flat, &old_flat[copied..]);
    if seam_count > 0 {
        new_flat.extend(std::iter::repeat_n(last, seam_count));
    }
    let (old_knots, old_mults) = (knots.knots(), knots.multiplicities());
    let capacity = old_knots.len() + interior.len();
    let (mut new_knots, mut new_mults) =
        (Vec::with_capacity(capacity), Vec::with_capacity(capacity));
    let mut copied = 0;
    for run in interior.chunk_by(|a, b| a == b) {
        let (x, count) = (run[0], run.len() as u32);
        let mut at = copied;
        while old_knots[at] < x {
            at += 1;
        }
        append(&mut new_knots, &old_knots[copied..at]);
        append(&mut new_mults, &old_mults[copied..at]);
        new_knots.push(x);
        if old_knots[at] == x {
            new_mults.push(old_mults[at] + count);
            at += 1;
        } else {
            new_mults.push(count);
        }
        copied = at;
    }
    append(&mut new_knots, &old_knots[copied..]);
    append(&mut new_mults, &old_mults[copied..]);
    if seam_count > 0 {
        let end = new_mults.len() - 1;
        new_mults[0] += seam_count as u32;
        new_mults[end] += seam_count as u32;
    }
    // A clamped end may have one more than the degree.
    let end = new_mults.len() - 1;
    let too_large = new_mults.iter().enumerate().any(|(i, &m)| {
        let clamped_end = !periodic && (i == 0 || i == end);
        m > degree as u32 + u32::from(clamped_end)
    });
    if too_large {
        return Err(KnotInsertionError::MultiplicityTooLarge);
    }
    let (poles, kept) = if periodic {
        // Seam insertions first, at the seam's first copy.
        let mut tiled_insert = vec![first; seam_count];
        tiled_insert.extend_from_slice(interior);
        let poles = refine_periodic(
            degree,
            knots.flat(),
            &new_flat,
            knots.multiplicities()[0] as usize,
            &tiled_insert,
            poles,
            lanes,
        );
        (poles, Kept::default())
    } else {
        let mut out = vec![S::Pole::default(); poles.len() + insert.len() * lanes.count];
        refine(
            degree,
            knots.flat(),
            &new_flat,
            insert,
            poles,
            lanes,
            &mut out,
        );
        let (first_span, last_span) = spans(degree, knots.flat(), insert);
        let kept = Kept {
            leading: first_span - degree + 1,
            trailing: knots.pole_count() - last_span,
        };
        (out, kept)
    };
    let knots = KnotVector::from_validated(degree, new_knots, new_mults, new_flat, periodic);
    Ok(Edited { knots, poles, kept })
}

/// Appends `values` to `out`: a few values, as most runs between inserted
/// knots are, are cheaper pushed one by one than copied.
#[inline]
fn append<T: Copy>(out: &mut Vec<T>, values: &[T]) {
    if values.len() <= 4 {
        for &value in values {
            out.push(value);
        }
    } else {
        out.extend_from_slice(values);
    }
}

/// The poles after inserting the knots `x` into the flat knots `old` in
/// one pass (The NURBS Book, A5.4), `new` being `old` and `x` merged, for
/// `x` non-decreasing within `[old[degree], old[old.len() - 1 - degree]]`.
///
/// `poles` and `out` hold the lanes' poles as `lanes` lays them out, the
/// old and the new count per lane. Interleaved lanes are refined together,
/// consecutive ones one by one: see [`refine_slots`].
fn refine<S: Source>(
    degree: usize,
    old: &[f64],
    new: &[f64],
    x: &[f64],
    poles: S,
    lanes: Lanes,
    out: &mut [S::Pole],
) {
    if lanes.consecutive {
        let (n, new_n) = (poles.len() / lanes.count, out.len() / lanes.count);
        for (lane, lane_out) in out.chunks_exact_mut(new_n).enumerate() {
            let lane_poles = poles.slice(lane * n..(lane + 1) * n);
            refine_slots(degree, old, new, x, lane_poles, One, lane_out);
        }
    } else {
        refine_slots(degree, old, new, x, poles, lanes.count, out);
    }
}

/// The spans of the first and the last of the nonempty `x` in the flat
/// knots `old`: the last `span` in `degree..` below the last pole's with
/// `old[span] <= u`.
fn spans(degree: usize, old: &[f64], x: &[f64]) -> (usize, usize) {
    let n = old.len() - degree - 1;
    let span = |u: f64| degree + old[degree..n].partition_point(|&k| k <= u) - 1;
    (span(x[0]), span(x[x.len() - 1]))
}

/// The number of poles in a slot, a constant one for a single lane.
trait Width: Copy {
    fn get(self) -> usize;
}

impl Width for usize {
    #[inline]
    fn get(self) -> usize {
        self
    }
}

/// A single lane's width.
#[derive(Clone, Copy)]
struct One;

impl Width for One {
    #[inline]
    fn get(self) -> usize {
        1
    }
}

/// [`refine`] on `poles` held in slots of `width` poles, slot `i` holding
/// pole `i` of every lane. The slots away from the inserted knots are
/// copied; each inserted knot then blends `degree` slots, right to left,
/// every lane with the same coefficients.
fn refine_slots<S: Source>(
    degree: usize,
    old: &[f64],
    new: &[f64],
    x: &[f64],
    poles: S,
    width: impl Width,
    out: &mut [S::Pole],
) {
    let width = width.get();
    let p = degree;
    let n = old.len() - p - 1;
    let s = x.len();
    debug_assert_eq!(new.len(), old.len() + s);
    debug_assert_eq!(poles.len(), width * n);
    debug_assert_eq!(out.len(), width * (n + s));
    if s == 0 {
        poles.copy_to(0, out);
        return;
    }
    let (a, last) = spans(p, old, x);
    let b = last + 1;
    poles.copy_to(0, &mut out[..(a - p + 1) * width]);
    poles.copy_to((b - 1) * width, &mut out[(b - 1 + s) * width..]);
    let mut i = b + p - 1;
    let mut k = b + p + s - 1;
    for &xj in x.iter().rev() {
        // The poles right of `xj` move along by the knots inserted since.
        let mut moved = i;
        while xj <= old[moved] && moved > a {
            moved -= 1;
        }
        let count = i - moved;
        poles.copy_to(
            (moved - p) * width,
            &mut out[(k - p - count) * width..(k - p) * width],
        );
        (i, k) = (moved, k - count);
        // Slots `k - p - 1..=k`: the first takes its neighbour, then each of
        // the next `p` blends with the one after it.
        let slots = &mut out[(k - p - 1) * width..(k + 1) * width];
        let (head, tail) = slots.split_at_mut(width);
        head.copy_from_slice(&tail[..width]);
        let knots = new[k + 1..k + p + 1].iter().zip(&old[i - p + 1..i + 1]);
        for (index, (&new_knot, &old_knot)) in knots.enumerate() {
            let pair = &mut slots[(index + 1) * width..(index + 3) * width];
            let (left, right) = pair.split_at_mut(width);
            let alpha = new_knot - xj;
            if alpha == 0.0 {
                left.copy_from_slice(right);
            } else {
                let alpha = alpha / (new_knot - old_knot);
                for (left, &right) in left.iter_mut().zip(&*right) {
                    *left = left.combine(alpha, right, 1.0 - alpha);
                }
            }
        }
        k -= 1;
    }
}

/// [`refine`] on a periodic direction, whose `n` poles weight the basis
/// functions of one period's flat knots tiled endlessly: pole `i` weights
/// the function starting at tiled knot `i + seam - degree - 1`, `seam`
/// being the seam multiplicity.
///
/// The direction is unwrapped into a non-periodic one over a window of
/// tiled knots, the poles repeated as they wrap, whose parameter range
/// runs from the old knot at or below the first new pole's interior knots
/// to the seam one period on. Every periodic copy of `x` in that range is
/// inserted by [`refine`], and the new period's poles are read off the
/// refined window: a new pole depends only on its interior knots, which
/// the refined window holds with every knot in between. `x` lists the
/// knots to insert within `[first, last)`, the seam ones as `first`.
fn refine_periodic<S: Source>(
    degree: usize,
    flat: &[f64],
    new_flat: &[f64],
    seam: usize,
    x: &[f64],
    poles: S,
    lanes: Lanes,
) -> Vec<S::Pole> {
    let (first, period) = (flat[0], flat[flat.len() - 1] - flat[0]);
    let new_seam = seam + x.iter().take_while(|&&k| k == first).count();
    let (n, new_n) = (flat.len() - seam, new_flat.len() - new_seam);
    let (block, new_block) = (&flat[..n], &new_flat[..new_n]);
    let tiled = |block: &[f64], k: isize| {
        let len = block.len() as isize;
        block[k.rem_euclid(len) as usize] + period * k.div_euclid(len) as f64
    };
    let p = degree as isize;
    // Tiled index of the knot starting the first new pole's support.
    let new_start = new_seam as isize - p - 1;
    // The window's range starts at the last copy of the old knot at or
    // below the first interior knot of that support.
    let (index, tile) = (
        (new_start + 1).rem_euclid(new_n as isize) as usize,
        (new_start + 1).div_euclid(new_n as isize),
    );
    let old_index = block.partition_point(|&k| k <= new_block[index]) - 1;
    let anchor = old_index as isize + tile * n as isize;
    let anchor_knot = tiled(block, anchor);
    let window_start = anchor - p;
    let window_end = (n + seam) as isize - 1;
    let old_knots: Vec<f64> = (window_start..=window_end + p)
        .map(|k| tiled(block, k))
        .collect();
    let mut inserted = Vec::with_capacity(x.len() * (1 - tile) as usize + new_seam - seam);
    for t in tile..=0 {
        inserted.extend(
            x.iter()
                .map(|&k| k + period * t as f64)
                .filter(|&k| k >= anchor_knot),
        );
    }
    inserted.extend(std::iter::repeat_n(
        tiled(block, n as isize),
        new_seam - seam,
    ));
    let mut new_knots = Vec::with_capacity(old_knots.len() + inserted.len());
    let (mut i, mut j) = (0, 0);
    while i < old_knots.len() || j < inserted.len() {
        if j == inserted.len() || (i < old_knots.len() && old_knots[i] <= inserted[j]) {
            new_knots.push(old_knots[i]);
            i += 1;
        } else {
            new_knots.push(inserted[j]);
            j += 1;
        }
    }
    let count = old_knots.len() - p as usize - 1;
    let offset = seam as isize - p - 1;
    let window = Lanes::rows(lanes.count);
    let mut window_poles = vec![S::Pole::default(); lanes.count * count];
    for l in 0..lanes.count {
        for w in 0..count {
            let pole = (window_start + w as isize - offset).rem_euclid(n as isize) as usize;
            window_poles[window.index(l, w, count)] = poles.get(lanes.index(l, pole, n));
        }
    }
    let new_count = count + inserted.len();
    let mut refined = vec![S::Pole::default(); lanes.count * new_count];
    refine(
        degree,
        &old_knots,
        &new_knots,
        &inserted,
        window_poles.as_slice(),
        window,
        &mut refined,
    );
    // Window pole `w` weights the new tiled function `w + shift`, aligned
    // on the last copy of the anchor knot.
    let anchor_last = (new_block.partition_point(|&k| k <= block[old_index]) - 1) as isize
        + tile * new_n as isize;
    let shift = anchor_last - (new_knots.partition_point(|&k| k <= anchor_knot) as isize - 1);
    let start = (new_start - shift) as usize;
    debug_assert!(start + new_n <= new_count);
    let mut out = vec![S::Pole::default(); lanes.count * new_n];
    for l in 0..lanes.count {
        for w in 0..new_n {
            out[lanes.index(l, w, new_n)] = refined[window.index(l, start + w, new_count)];
        }
    }
    out
}
