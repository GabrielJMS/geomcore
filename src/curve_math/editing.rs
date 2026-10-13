//! Editing one direction of a B-spline: knot insertion and removal,
//! degree elevation, cutting into pieces, Bézier pieces included, and
//! reversal of the poles that direction weights, a curve's or every row
//! of a surface's.
//!
//! The algorithms are generic over the [`Pole`] they combine: points for a
//! nonrational shape, homogeneous points for a rational one. [`Lanes`]
//! says where the poles of each independent row sit, so one pass serves a
//! curve and both directions of a surface, every row with the same
//! coefficients.

mod bezier;
mod cutting;
mod elevation;
mod removal;

use bezier::{Walker, bezier_pieces};
pub(crate) use cutting::{
    check_piece, check_split, cut, is_clamped_to, reversed_knots, reversed_slot,
};
pub(crate) use elevation::Elevation;
pub(crate) use removal::{check_removal, remove_knots};

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

    /// The points of `poles` and their weights, if rational.
    fn parts(poles: &[Self]) -> (Vec<Point3D>, Option<Vec<f64>>);
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

    fn parts(poles: &[Self]) -> (Vec<Point3D>, Option<Vec<f64>>) {
        (poles.to_vec(), None)
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

    fn parts(poles: &[Self]) -> (Vec<Point3D>, Option<Vec<f64>>) {
        let points = poles
            .iter()
            .map(|&[x, y, z, w]| {
                let inverse = 1.0 / w;
                Point3D::new(x * inverse, y * inverse, z * inverse)
            })
            .collect();
        (points, Some(poles.iter().map(|h| h[3]).collect()))
    }
}

/// The poles an edit reads, by index: a slice of them, or a rational
/// shape's points and weights read as homogeneous points, so that they
/// need no converted copy.
pub(crate) trait Source: Copy {
    type Pole: Pole;

    /// Whether reading a pole converts it, so that the poles an edit keeps
    /// are better copied from the shape than through the edit.
    const CONVERTS: bool;

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

    /// Appends the poles `range` to `out`.
    #[inline]
    fn extend_to(self, range: std::ops::Range<usize>, out: &mut Vec<Self::Pole>) {
        out.extend(range.map(|i| self.get(i)));
    }
}

impl<P: Pole> Source for &[P] {
    type Pole = P;

    const CONVERTS: bool = false;

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

    #[inline]
    fn extend_to(self, range: std::ops::Range<usize>, out: &mut Vec<P>) {
        out.extend_from_slice(&self[range]);
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

    const CONVERTS: bool = true;

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

    #[inline]
    fn copy_to(self, start: usize, out: &mut [[f64; 4]]) {
        let range = start..start + out.len();
        let poles = self.points[range.clone()].iter().zip(&self.weights[range]);
        for (out, (p, &w)) in out.iter_mut().zip(poles) {
            *out = [w * p.x, w * p.y, w * p.z, w];
        }
    }

    #[inline]
    fn extend_to(self, range: std::ops::Range<usize>, out: &mut Vec<[f64; 4]>) {
        let poles = self.points[range.clone()].iter().zip(&self.weights[range]);
        out.extend(poles.map(|(p, &w)| [w * p.x, w * p.y, w * p.z, w]));
    }
}

/// The points and weights of homogeneous poles.
fn from_homogeneous(
    edited: &Edited<[f64; 4]>,
    points: &[Point3D],
    weights: &[f64],
    lanes: Lanes,
) -> (Vec<Point3D>, Vec<f64>) {
    let (rows, width) = lanes.rows_and_width();
    let (old_row, new_row) = (points.len() / rows, edited.knots.pole_count() * width);
    let (leading, trailing) = (edited.kept.leading * width, edited.kept.trailing * width);
    let changed_count = new_row - leading - trailing;
    let stride = edited.poles.len() / rows;
    let mut new_points = Vec::with_capacity(new_row * rows);
    let mut new_weights = Vec::with_capacity(new_row * rows);
    for row in 0..rows {
        let old = row * old_row..(row + 1) * old_row;
        let old_end = old.end - trailing;
        new_points.extend_from_slice(&points[old.start..old.start + leading]);
        new_weights.extend_from_slice(&weights[old.start..old.start + leading]);
        let first = row * stride + leading - edited.start * width;
        let changed = &edited.poles[first..first + changed_count];
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
    /// The pole (or slot, for interleaved lanes) each lane of `poles`
    /// starts at, those before it being kept ones left out: every lane as
    /// long, holding at least all its changed poles.
    pub(crate) start: usize,
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

    /// The edited direction, points and weights of a rational shape:
    /// [`Edit::apply`] on homogeneous points.
    #[allow(clippy::type_complexity)] // the parts a B-spline stores
    fn apply_rational(
        &self,
        direction: &KnotVector,
        points: &[Point3D],
        weights: &[f64],
        lanes: Lanes,
    ) -> Result<(KnotVector, Vec<Point3D>, Vec<f64>), Self::Error> {
        let edited = self.apply(direction, Weighted { points, weights }, lanes)?;
        let (points, weights) = from_homogeneous(&edited, points, weights, lanes);
        Ok((edited.knots, points, weights))
    }
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
            let (knots, points, weights) =
                edit.apply_rational(direction, points, weights, lanes)?;
            Ok((knots, points, Some(weights)))
        }
    }
}

/// Visits the Bézier pieces of a curve's direction over its range, a
/// periodic direction's from its lower bound: `visit(x, y, points,
/// weights)` for the piece over `[x, y]`, its `degree + 1` points and their
/// weights if there are `weights`.
pub(crate) fn curve_pieces(
    direction: &KnotVector,
    points: &[Point3D],
    weights: Option<&[f64]>,
    mut visit: impl FnMut(f64, f64, Vec<Point3D>, Option<Vec<f64>>),
) {
    fn run<S: Source>(
        direction: &KnotVector,
        poles: S,
        visit: &mut impl FnMut(f64, f64, Vec<Point3D>, Option<Vec<f64>>),
    ) {
        Walker::new(direction, 1).for_each_piece(poles, One, |_, x, y, piece| {
            let (points, weights) = S::Pole::parts(piece);
            visit(x, y, points, weights);
        });
    }
    match weights {
        None => run(direction, points, &mut visit),
        Some(weights) => run(direction, Weighted { points, weights }, &mut visit),
    }
}

/// Visits the Bézier patches of a surface over `u` and `v`, its rows of
/// poles along `v`: `visit(i, (u0, u1), (v0, v1), points, weights)` for the
/// patch over `[u0, u1] x [v0, v1]`, the `i`-th along `u`, its `(u degree
/// + 1) (v degree + 1)` points row by row and their weights if there are
/// `weights`; patches come along `u` for each span along `v`.
pub(crate) fn surface_patches(
    u: &KnotVector,
    v: &KnotVector,
    points: &[Point3D],
    weights: Option<&[f64]>,
    mut visit: impl FnMut(usize, (f64, f64), (f64, f64), Vec<Point3D>, Option<Vec<f64>>),
) {
    #[allow(clippy::type_complexity)] // the visitor of `surface_patches`
    fn run<S: Source>(
        u: &KnotVector,
        v: &KnotVector,
        poles: S,
        visit: &mut impl FnMut(usize, (f64, f64), (f64, f64), Vec<Point3D>, Option<Vec<f64>>),
    ) {
        // The strips of `v` pieces: strip `j` holds `v degree + 1` poles of
        // every row, a grid of rows along `v` whose `u` pieces are patches.
        let (v_breaks, strips) = bezier_pieces(v, poles, Lanes::rows(u.pole_count()));
        let width = v.degree() + 1;
        let mut walker = Walker::new(u, width);
        for (strip, span) in strips
            .chunks_exact(u.pole_count() * width)
            .zip(v_breaks.windows(2))
        {
            walker.for_each_piece(strip, width, |i, x, y, patch| {
                let (points, weights) = S::Pole::parts(patch);
                visit(i, (x, y), (span[0], span[1]), points, weights);
            });
        }
    }
    match weights {
        None => run(u, v, points, &mut visit),
        Some(weights) => run(u, v, Weighted { points, weights }, &mut visit),
    }
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

    fn apply_rational(
        &self,
        direction: &KnotVector,
        points: &[Point3D],
        weights: &[f64],
        lanes: Lanes,
    ) -> Result<(KnotVector, Vec<Point3D>, Vec<f64>), KnotInsertionError> {
        if direction.is_periodic() || self.0.is_empty() {
            let edited = insert_knots(direction, Weighted { points, weights }, lanes, self.0)?;
            let (points, weights) = from_homogeneous(&edited, points, weights, lanes);
            return Ok((edited.knots, points, weights));
        }
        insert_rational(direction, points, weights, lanes, self.0)
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

    /// The rows of slots the lanes make and the poles in a slot:
    /// consecutive lanes are rows of their own, interleaved ones one row
    /// of slots, a slot holding every lane's pole.
    fn rows_and_width(self) -> (usize, usize) {
        if self.consecutive {
            (self.count, 1)
        } else {
            (1, self.count)
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
    check_insertion(knots, insert)?;
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
            start: 0,
        });
    }
    let (degree, periodic) = (knots.degree(), knots.is_periodic());
    let Merged {
        flat: new_flat,
        seam_count,
        interior,
    } = merged_knots(knots, insert)?;
    let interior = &insert[interior];
    let (poles, kept, start) = if periodic {
        // Seam insertions first, at the seam's first copy.
        let mut tiled_insert = vec![knots.bounds().0; seam_count];
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
        (poles, Kept::default(), 0)
    } else {
        // Converted poles are refined only from the first changed one on.
        let trimmed = S::CONVERTS;
        let (out, start) = refine(
            degree,
            knots.flat(),
            &new_flat,
            insert,
            poles,
            lanes,
            trimmed,
        );
        let (first_span, last_span) = spans(degree, knots.flat(), insert);
        let kept = Kept {
            leading: first_span - degree + 1,
            trailing: knots.pole_count() - last_span,
        };
        (out, kept, start)
    };
    let knots = KnotVector::from_flat(degree, new_flat, periodic);
    Ok(Edited {
        knots,
        poles,
        kept,
        start,
    })
}

/// Checks the knots to insert into `knots`: finite, non-decreasing and
/// within its bounds.
fn check_insertion(knots: &KnotVector, insert: &[f64]) -> Result<(), KnotInsertionError> {
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
    Ok(())
}

/// New flat knots: see [`merged_knots`].
struct Merged {
    flat: Vec<f64>,
    /// How many of the inserted knots are seam copies, on a periodic
    /// direction.
    seam_count: usize,
    /// The indices of the other inserted knots.
    interior: std::ops::Range<usize>,
}

/// The flat knots of `knots` with `insert` (checked by
/// [`check_insertion`]) merged in, an inserted knot after the old copies of its value,
/// checking that no multiplicity grows too large: at most the degree, one
/// more at a clamped end.
fn merged_knots(knots: &KnotVector, insert: &[f64]) -> Result<Merged, KnotInsertionError> {
    let (first, last) = knots.bounds();
    let (degree, periodic) = (knots.degree(), knots.is_periodic());
    let seam = |x: f64| periodic && (x == first || x == last);
    let seam_count = insert.iter().filter(|&&x| seam(x)).count();
    let interior = if seam_count == 0 {
        0..insert.len()
    } else {
        // Sorted, so the seam values are the ones at either end.
        let leading = insert.iter().take_while(|&&x| x == first).count();
        leading..leading + insert.len() - seam_count
    };
    // A clamped end may have one more copy than the degree, a periodic
    // seam no more than it.
    if seam_count > 0 && knots.multiplicities()[0] as usize + seam_count > degree {
        return Err(KnotInsertionError::MultiplicityTooLarge);
    }
    let old_flat = knots.flat();
    let mut new_flat = Vec::with_capacity(old_flat.len() + insert.len());
    if seam_count > 0 {
        new_flat.extend(std::iter::repeat_n(first, seam_count));
    }
    let ends = (old_flat[0], old_flat[old_flat.len() - 1]);
    let mut copied = 0;
    for &x in &insert[interior.clone()] {
        let at = copied + old_flat[copied..].iter().take_while(|&&k| k <= x).count();
        append(&mut new_flat, &old_flat[copied..at]);
        new_flat.push(x);
        copied = at;
        // The copies of `x` so far all end the new knots.
        let copies = new_flat.iter().rev().take_while(|&&k| k == x).count();
        let clamped_end = !periodic && (x == ends.0 || x == ends.1);
        if copies > degree + usize::from(clamped_end) {
            return Err(KnotInsertionError::MultiplicityTooLarge);
        }
    }
    append(&mut new_flat, &old_flat[copied..]);
    if seam_count > 0 {
        new_flat.extend(std::iter::repeat_n(last, seam_count));
    }
    Ok(Merged {
        flat: new_flat,
        seam_count,
        interior,
    })
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
/// `x` non-decreasing within `[old[degree], old[old.len() - 1 - degree]]`:
/// the new poles laid out as `lanes` lays out `poles`, each lane's from
/// the slot returned, all of them or if `trimmed` only those of
/// [`refined_slots`].
///
/// Each lane's poles are its kept ones copied, then its changed ones
/// refined in place: interleaved lanes together, consecutive ones one by
/// one (see [`refine_slots`]).
fn refine<S: Source>(
    degree: usize,
    old: &[f64],
    new: &[f64],
    x: &[f64],
    poles: S,
    lanes: Lanes,
    trimmed: bool,
) -> (Vec<S::Pole>, usize) {
    let (p, s) = (degree, x.len());
    let n = old.len() - p - 1;
    let (rows, width) = lanes.rows_and_width();
    if s == 0 {
        let mut out = Vec::with_capacity(poles.len());
        poles.extend_to(0..poles.len(), &mut out);
        return (out, 0);
    }
    let (a, last) = spans(p, old, x);
    let b = last + 1;
    let slots = refined_slots(p, n, s, (a, last), trimmed);
    let mut out = Vec::with_capacity(slots.len() * lanes.count);
    for row in 0..rows {
        let row_start = row * n * width;
        // The slots up to `a - p` and from `b - 1 + s` keep old poles.
        poles.extend_to(
            row_start + slots.start * width..row_start + (a - p + 1) * width,
            &mut out,
        );
        out.resize(out.len() + (b + s - a + p - 2) * width, S::Pole::default());
        poles.extend_to(
            row_start + (b - 1) * width..row_start + (slots.end - s) * width,
            &mut out,
        );
    }
    let row_len = slots.len() * width;
    for (row, row_out) in out.chunks_exact_mut(row_len).enumerate() {
        let row_poles = poles.slice(row * n * width..(row + 1) * n * width);
        if lanes.consecutive {
            refine_slots(p, old, new, x, row_poles, One, slots.start, row_out);
        } else {
            refine_slots(p, old, new, x, row_poles, width, slots.start, row_out);
        }
    }
    (out, slots.start)
}

/// The new slots [`refine`] writes, of `n + s` after inserting `s` knots
/// whose first and last are in the old knot spans `spans`: all, or if
/// `trimmed` those the refinement reaches, from the kept one before the
/// first changed one (written over with itself) to `degree + 1` after the
/// last changed one (read).
fn refined_slots(
    degree: usize,
    n: usize,
    s: usize,
    (first_span, last_span): (usize, usize),
    trimmed: bool,
) -> std::ops::Range<usize> {
    if trimmed {
        first_span - degree..(last_span + 1 + degree + s).min(n + s)
    } else {
        0..n + s
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

/// [`insert_knots`] of the nonempty `insert` into a non-periodic rational
/// direction: the new direction, points and weights.
///
/// The slots [`refine`] writes are refined in homogeneous form, on the
/// stack if few enough as for most insertions, and the changed ones
/// divided back into the new arrays, the kept ones copied as they are.
#[allow(clippy::type_complexity)] // the parts a B-spline stores
fn insert_rational(
    knots: &KnotVector,
    points: &[Point3D],
    weights: &[f64],
    lanes: Lanes,
    insert: &[f64],
) -> Result<(KnotVector, Vec<Point3D>, Vec<f64>), KnotInsertionError> {
    /// The most homogeneous poles refined on the stack: few enough to
    /// zero quickly.
    const STACK: usize = 64;
    check_insertion(knots, insert)?;
    let new_flat = merged_knots(knots, insert)?.flat;
    let (p, old, s) = (knots.degree(), knots.flat(), insert.len());
    let n = knots.pole_count();
    let (rows, width) = lanes.rows_and_width();
    let (a, last) = spans(p, old, insert);
    let b = last + 1;
    let slots = refined_slots(p, n, s, (a, last), true);
    // Slots up to `a - p` and from `b - 1 + s` are kept.
    let (leading, trailing_start) = (a - p + 1, b - 1 + s);
    let source = Weighted { points, weights };
    let window_row = slots.len() * width;
    let size = rows * (n + s) * width;
    let (mut new_points, mut new_weights) = (Vec::with_capacity(size), Vec::with_capacity(size));
    let mut stack = [[0.0; 4]; STACK];
    let mut heap = Vec::new();
    let window = if rows * window_row <= STACK {
        &mut stack[..rows * window_row]
    } else {
        heap.resize(rows * window_row, [0.0; 4]);
        &mut heap[..]
    };
    for (row, out) in window.chunks_exact_mut(window_row).enumerate() {
        // The window's kept slots, for the refinement to read.
        let at = |i: usize| (i - slots.start) * width;
        let old_slot = |i: usize| (row * n + i) * width;
        source.copy_to(old_slot(slots.start), &mut out[..at(leading)]);
        source.copy_to(old_slot(b - 1), &mut out[at(trailing_start)..]);
        let row_poles = source.slice(old_slot(0)..old_slot(n));
        if lanes.consecutive {
            refine_slots(p, old, &new_flat, insert, row_poles, One, slots.start, out);
        } else {
            refine_slots(
                p,
                old,
                &new_flat,
                insert,
                row_poles,
                width,
                slots.start,
                out,
            );
        }
        new_points.extend_from_slice(&points[old_slot(0)..old_slot(leading)]);
        new_weights.extend_from_slice(&weights[old_slot(0)..old_slot(leading)]);
        let changed = &out[at(leading)..at(trailing_start)];
        new_points.extend(changed.iter().map(|&[x, y, z, w]| {
            let inverse = 1.0 / w;
            Point3D::new(x * inverse, y * inverse, z * inverse)
        }));
        new_weights.extend(changed.iter().map(|h| h[3]));
        new_points.extend_from_slice(&points[old_slot(b - 1)..old_slot(n)]);
        new_weights.extend_from_slice(&weights[old_slot(b - 1)..old_slot(n)]);
    }
    let knots = KnotVector::from_flat(p, new_flat, false);
    Ok((knots, new_points, new_weights))
}

/// [`refine`] of the changed slots of `out`, the new slots from `start`
/// on, the kept ones already copied: `poles` are held in slots of `width`
/// poles, slot `i` holding pole `i` of every lane. Each inserted knot
/// blends `degree` slots, right to left, every lane with the same
/// coefficients.
#[allow(clippy::too_many_arguments)] // the parts of A5.4
fn refine_slots<S: Source>(
    degree: usize,
    old: &[f64],
    new: &[f64],
    x: &[f64],
    poles: S,
    width: impl Width,
    start: usize,
    out: &mut [S::Pole],
) {
    let width = width.get();
    let p = degree;
    let s = x.len();
    debug_assert_eq!(new.len(), old.len() + s);
    debug_assert_eq!(poles.len(), width * (old.len() - p - 1));
    let (a, last) = spans(p, old, x);
    let b = last + 1;
    // Slot `i` of the new poles, from the first written.
    let at = |i: usize| (i - start) * width;
    let mut i = b + p - 1;
    let mut k = b + p + s - 1;
    for &xj in x.iter().rev() {
        // The poles right of `xj` move along by the knots inserted since.
        let count = old[a + 1..=i]
            .iter()
            .rev()
            .take_while(|&&k| xj <= k)
            .count();
        let moved = i - count;
        poles.copy_to((moved - p) * width, &mut out[at(k - p - count)..at(k - p)]);
        (i, k) = (moved, k - count);
        // Slots `k - p - 1..=k`: the first takes its neighbour, then each of
        // the next `p` blends with the one after it.
        let slots = &mut out[at(k - p - 1)..at(k + 1)];
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
    let (refined, _) = refine(
        degree,
        &old_knots,
        &new_knots,
        &inserted,
        window_poles.as_slice(),
        window,
        false,
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
