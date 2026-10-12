//! B-splines: split, then a priori, on each knot span's Bézier control
//! points.
//!
//! A piece of degree `n` over its own interval, split into `k` equal
//! steps, deviates from each chord by at most `n (n − 1) M / (8 k²)`, `M`
//! the largest second difference of its control points (the curve's
//! second derivative is at most `n (n − 1) M`, and a curve whose ends lie
//! on a segment strays from it by at most `h² / 8` times its second
//! derivative over a step `h`).
//!
//! A rational piece works on the homogeneous control points `(w P, w)`.
//! With `c` the centroid of the points `P_i`, `R` their largest distance
//! to `c`, and `x_i = w_i (P_i − c)`, it deviates by at most
//! `n (n − 1) (max |ΔΔx_i| + R max |ΔΔw_i|) / (8 k² min w_i)`: the
//! homogeneous chord projects onto the 3D chord, and the curve and its
//! chords stay within `R` of `c`.
//!
//! A piece is halved (de Casteljau at its middle) while its halves need
//! clearly fewer steps than it does, so a span whose second derivative
//! varies gets fewer points.

use super::sealed::{DiscretizeRange, Run};
use super::{DiscretizationCriteria, smallest_count};
use crate::math::bernstein;
use crate::{BSplineCurve3D, Point3D};
use std::f64::consts::PI;

/// Control points of the highest supported degree, 25.
const CAPACITY: usize = 26;

/// How many times a span may be halved.
const MAX_DEPTH: u32 = 30;

/// How many times a span may be halved looking for a bound on the angle
/// deviation: a piece next to a cusp never gets one.
const MAX_UNBOUNDED_DEPTH: u32 = 8;

/// A lower bound on the speed below this fraction of the largest
/// derivative control point counts as none: the poles nearly coincide, and
/// bounding the turning there would take as many points as the gap is
/// small.
const MIN_SPEED_RATIO: f64 = 1e-6;

/// A piece is halved only if its halves need fewer than this fraction of
/// its steps.
const SPLIT_GAIN: f64 = 0.85;

/// Coefficients of the polynomial giving a rational piece's tangent
/// direction, of degree `2 × 25 − 1`.
const DIRECTION_CAPACITY: usize = 2 * (CAPACITY - 1);

/// A homogeneous control point `(x w, y w, z w, w)`.
type Homogeneous = [f64; 4];

/// A polynomial piece over the parameter interval `[first, last]`, in
/// Bernstein form over `[0, 1]`.
#[derive(Clone, Copy)]
struct Piece {
    first: f64,
    last: f64,
    degree: usize,
    rational: bool,
    points: [Homogeneous; CAPACITY],
}

fn lerp(a: Homogeneous, b: Homogeneous, s: f64) -> Homogeneous {
    std::array::from_fn(|c| a[c] + (b[c] - a[c]) * s)
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// The second difference of the first three coordinates of `p[i..i + 3]`.
fn second_difference(p: &[Homogeneous], i: usize) -> [f64; 3] {
    std::array::from_fn(|c| p[i + 2][c] - 2.0 * p[i + 1][c] + p[i][c])
}

impl Piece {
    fn control(&self) -> &[Homogeneous] {
        &self.points[..=self.degree]
    }

    /// The two pieces either side of the fraction `s`, by de Casteljau.
    fn split(&self, s: f64) -> (Piece, Piece) {
        let n = self.degree;
        let mut t = self.points;
        let (mut left, mut right) = (*self, *self);
        for r in 1..=n {
            for i in 0..=n - r {
                t[i] = lerp(t[i], t[i + 1], s);
            }
            left.points[r] = t[0];
            right.points[n - r] = t[n - r];
        }
        let middle = self.first + (self.last - self.first) * s;
        left.last = middle;
        right.first = middle;
        (left, right)
    }

    /// The piece restricted to `[first, last]` inside its interval.
    fn restricted(&self, first: f64, last: f64) -> Piece {
        let length = self.last - self.first;
        let mut piece = *self;
        if first > self.first {
            piece = piece.split((first - self.first) / length).1;
        }
        if last < self.last {
            piece = piece
                .split((last - piece.first) / (self.last - piece.first))
                .0;
        }
        piece.first = first;
        piece.last = last;
        piece
    }

    /// The point at the fraction `s` of the piece, by de Casteljau.
    fn point_at(&self, s: f64) -> Point3D {
        let n = self.degree;
        let mut t = self.points;
        for r in 1..=n {
            for i in 0..=n - r {
                t[i] = lerp(t[i], t[i + 1], s);
            }
        }
        let [x, y, z, w] = t[0];
        Point3D::new(x / w, y / w, z / w)
    }

    /// `K` such that `k` equal steps deviate by at most `K / k²`.
    fn chord_constant(&self) -> f64 {
        let n = self.degree;
        if n < 2 {
            return 0.0;
        }
        let p = self.control();
        let factor = (n * (n - 1)) as f64 / 8.0;
        let largest = |f: &dyn Fn(usize) -> f64| (0..n - 1).map(f).fold(0.0, f64::max);
        if !self.rational {
            return factor * largest(&|i| norm(second_difference(p, i)));
        }
        let projected = |h: &Homogeneous| [h[0] / h[3], h[1] / h[3], h[2] / h[3]];
        let mut centroid = [0.0; 3];
        for h in p {
            let q = projected(h);
            for c in 0..3 {
                centroid[c] += q[c] / (n + 1) as f64;
            }
        }
        let radius = p
            .iter()
            .map(|h| {
                let q = projected(h);
                norm(std::array::from_fn(|c| q[c] - centroid[c]))
            })
            .fold(0.0, f64::max);
        let mut offsets = [[0.0; 4]; CAPACITY];
        for (x, h) in offsets.iter_mut().zip(p) {
            *x = [
                h[0] - centroid[0] * h[3],
                h[1] - centroid[1] * h[3],
                h[2] - centroid[2] * h[3],
                h[3],
            ];
        }
        let offset = largest(&|i| norm(second_difference(&offsets, i)));
        let weight = largest(&|i| (p[i + 2][3] - 2.0 * p[i + 1][3] + p[i][3]).abs());
        let smallest_weight = p.iter().map(|h| h[3]).fold(f64::INFINITY, f64::min);
        factor * (offset + radius * weight) / smallest_weight
    }

    /// `T` such that the tangent turns by at most `T / k` over each of
    /// `k` equal steps, `None` where no lower bound on the speed exists.
    ///
    /// The tangent of a nonrational piece has the direction of the
    /// differences of its control points. The tangent of a rational piece
    /// `X / W` has the direction of `X' W − X W'` (its derivative times
    /// `W²`, `W` being positive), a polynomial of degree `2n − 1` whose
    /// Bernstein coefficients come from the products of the homogeneous
    /// control points' differences and weights.
    fn turning(&self) -> Option<f64> {
        let p = self.control();
        let n = self.degree;
        let mut directions = [[0.0; 3]; DIRECTION_CAPACITY];
        if !self.rational {
            for (i, d) in directions.iter_mut().enumerate().take(n) {
                *d = std::array::from_fn(|c| p[i + 1][c] - p[i][c]);
            }
            return turning(&directions[..n]);
        }
        // Scaled Bernstein coefficients multiply by convolution.
        let mut weights = [0.0; CAPACITY];
        let mut weight_differences = [0.0; CAPACITY];
        for i in 0..=n {
            weights[i] = p[i][3];
        }
        for i in 0..n {
            weight_differences[i] = p[i + 1][3] - p[i][3];
        }
        bernstein::scale(&mut weights[..=n]);
        bernstein::scale(&mut weight_differences[..n]);
        for c in 0..3 {
            let mut coordinates = [0.0; CAPACITY];
            let mut differences = [0.0; CAPACITY];
            for i in 0..=n {
                coordinates[i] = p[i][c];
            }
            for i in 0..n {
                differences[i] = p[i + 1][c] - p[i][c];
            }
            bernstein::scale(&mut coordinates[..=n]);
            bernstein::scale(&mut differences[..n]);
            let mut product = [0.0; DIRECTION_CAPACITY];
            let product = &mut product[..2 * n];
            bernstein::add_product(&differences[..n], &weights[..=n], 1.0, product);
            bernstein::add_product(&coordinates[..=n], &weight_differences[..n], -1.0, product);
            bernstein::unscale(product);
            for (d, &value) in directions.iter_mut().zip(product.iter()) {
                d[c] = value;
            }
        }
        turning(&directions[..2 * n])
    }

    /// The fewest equal steps meeting the criteria, `None` if the angle
    /// deviation is limited but has no bound on this piece.
    fn segments(&self, criteria: &DiscretizationCriteria) -> Option<usize> {
        let chord = chord_segments(self.chord_constant(), criteria.max_chord_deviation);
        let Some(limit) = criteria.angle_limit() else {
            return Some(chord);
        };
        let turning = self.turning()?;
        let angle = smallest_count(turning / limit, |k| angle_deviation(turning, k) <= limit);
        Some(chord.max(angle))
    }

    /// Visits the leaves of the halving of this piece, each as a run.
    fn visit_leaves<F>(
        &self,
        criteria: &DiscretizationCriteria,
        depth: u32,
        min_segments: usize,
        visit: &mut F,
    ) where
        F: FnMut(Run, &dyn Fn(f64) -> Point3D),
    {
        let segments = self.segments(criteria);
        let may_split = match segments {
            Some(k) => depth < MAX_DEPTH && k > 2,
            None => depth < MAX_UNBOUNDED_DEPTH,
        };
        if may_split {
            let (left, right) = self.split(0.5);
            let halves = left.segments(criteria).zip(right.segments(criteria));
            let worth = match (segments, halves) {
                (None, _) => true,
                (Some(k), Some((l, r))) => (l as f64 + r as f64) < SPLIT_GAIN * k as f64,
                (Some(_), None) => false,
            };
            if worth {
                left.visit_leaves(criteria, depth + 1, min_segments, visit);
                right.visit_leaves(criteria, depth + 1, min_segments, visit);
                return;
            }
        }
        let chord_constant = self.chord_constant();
        let segments = segments
            .unwrap_or_else(|| chord_segments(chord_constant, criteria.max_chord_deviation))
            .max(min_segments);
        let run = Run {
            first: self.first,
            last: self.last,
            segments,
            chord_deviation: chord_deviation(chord_constant, segments),
            angle_deviation: self.turning().map_or(PI, |t| angle_deviation(t, segments)),
        };
        visit(run, &|s| self.point_at(s));
    }
}

/// The turning of a curve over `[0, 1]` whose tangent has the direction
/// of the polynomial with Bernstein coefficients `d`, of degree `m`: the
/// tangent's direction turns at most `|D'| / |D|` per unit of parameter,
/// `|D'|` is at most `m max |Δd_j|`, and `|D|` at least the smallest
/// component of the `d_j` along their mean direction, if positive and not
/// negligible (see [`MIN_SPEED_RATIO`]).
fn turning(d: &[[f64; 3]]) -> Option<f64> {
    let m = d.len().checked_sub(1)?;
    // The mean of the coefficients' directions, not of the coefficients:
    // a few large ones would pull it away from the others.
    let sum: [f64; 3] = std::array::from_fn(|c| d.iter().map(|v| v[c] / norm(*v)).sum());
    let length = norm(sum);
    if length.is_nan() || length <= 0.0 {
        return None;
    }
    let slowest = d
        .iter()
        .map(|v| (v[0] * sum[0] + v[1] * sum[1] + v[2] * sum[2]) / length)
        .fold(f64::INFINITY, f64::min);
    let fastest = d.iter().map(|v| norm(*v)).fold(0.0, f64::max);
    if slowest.is_nan() || slowest <= MIN_SPEED_RATIO * fastest {
        return None;
    }
    let change = (0..m)
        .map(|j| norm(std::array::from_fn(|c| d[j + 1][c] - d[j][c])))
        .fold(0.0, f64::max);
    Some(m as f64 * change / slowest)
}

/// The angle deviation of each of `segments` equal steps of a piece whose
/// tangent turns by at most `turning`. The directions of the tangent over
/// a step lie within half the step's turning `Θ` of its middle direction,
/// and so does the chord (their average) while `Θ < π`, so the tangent
/// and the chord differ by at most `Θ`; no angle exceeds π.
fn angle_deviation(turning: f64, segments: usize) -> f64 {
    (turning / segments as f64).min(PI)
}

fn chord_deviation(constant: f64, segments: usize) -> f64 {
    constant / (segments as f64 * segments as f64)
}

/// The fewest steps `k` with `constant / k² <= max`.
fn chord_segments(constant: f64, max: f64) -> usize {
    smallest_count((constant / max).sqrt(), |k| {
        chord_deviation(constant, k) <= max
    })
}

impl DiscretizeRange for BSplineCurve3D {
    type Point = Point3D;

    fn for_each_run<F>(
        &self,
        first: f64,
        last: f64,
        criteria: &DiscretizationCriteria,
        min_segments: usize,
        mut visit: F,
    ) where
        F: FnMut(Run, &dyn Fn(f64) -> Point3D),
    {
        let mut piece = Piece {
            first: 0.0,
            last: 0.0,
            degree: self.degree(),
            rational: self.is_rational(),
            points: [[0.0; 4]; CAPACITY],
        };
        // A span overlapping an end of the range by a rounding error, as
        // when a range overshoots a knot, would make a segment of no
        // length: the end's neighbouring span covers it instead.
        let negligible = 1e-12 * (last - first).max(first.abs()).max(last.abs());
        self.for_each_span_in(first, last, |(span_start, span_end), (a, b)| {
            let at_one_end = (a == first) != (b == last);
            if at_one_end && b - a <= negligible {
                return;
            }
            self.span_bezier_points(0.5 * (span_start + span_end), &mut piece.points);
            piece.first = span_start;
            piece.last = span_end;
            piece
                .restricted(a, b)
                .visit_leaves(criteria, 0, min_segments, &mut visit);
        });
    }
}
