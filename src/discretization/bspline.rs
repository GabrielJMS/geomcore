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
use crate::{BSplineCurve3D, Point3D};
use std::f64::consts::{FRAC_PI_2, PI};

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
    /// control points' differences and weights (see [`Context`]).
    fn turning(&self, context: &Context) -> Option<f64> {
        let p = self.control();
        let n = self.degree;
        let mut directions = [[0.0; 3]; DIRECTION_CAPACITY];
        if !self.rational {
            for (i, d) in directions.iter_mut().enumerate().take(n) {
                *d = std::array::from_fn(|c| p[i + 1][c] - p[i][c]);
            }
            return turning(&directions[..n], context);
        }
        let products = &context.direction_products;
        for i in 0..n {
            let difference: Homogeneous = std::array::from_fn(|c| p[i + 1][c] - p[i][c]);
            for (j, q) in p.iter().enumerate() {
                let f = products.factor(i, j);
                let d = &mut directions[i + j];
                for c in 0..3 {
                    d[c] += f * (difference[c] * q[3] - q[c] * difference[3]);
                }
            }
        }
        turning(&directions[..2 * n], context)
    }

    /// The piece's bounds for the criteria.
    fn bounds(&self, context: &Context) -> Bounds {
        let chord_constant = self.chord_constant();
        let chord = chord_segments(chord_constant, context.criteria.max_chord_deviation);
        let Some(limit) = context.criteria.angle_limit() else {
            return Bounds {
                chord_constant,
                turning: None,
                segments: Some(chord),
            };
        };
        let turning = self.turning(context);
        let segments = turning.map(|turning| {
            let angle = smallest_count(turning / limit, |k| angle_deviation(turning, k) <= limit);
            chord.max(angle)
        });
        Bounds {
            chord_constant,
            turning,
            segments,
        }
    }

    /// Visits the leaves of the halving of this piece, of `bounds`, each
    /// as a run.
    fn visit_leaves<F>(
        &self,
        bounds: Bounds,
        context: &Context,
        depth: u32,
        min_segments: usize,
        visit: &mut F,
    ) where
        F: FnMut(Run, &dyn Fn(f64) -> Point3D),
    {
        let may_split = match bounds.segments {
            Some(k) => depth < MAX_DEPTH && k > 2,
            None => depth < MAX_UNBOUNDED_DEPTH,
        };
        if may_split {
            let (left, right) = self.split(0.5);
            let (left_bounds, right_bounds) = (left.bounds(context), right.bounds(context));
            let halves = left_bounds.segments.zip(right_bounds.segments);
            let worth = match (bounds.segments, halves) {
                (None, _) => true,
                (Some(k), Some((l, r))) => (l as f64 + r as f64) < SPLIT_GAIN * k as f64,
                (Some(_), None) => false,
            };
            if worth {
                left.visit_leaves(left_bounds, context, depth + 1, min_segments, visit);
                right.visit_leaves(right_bounds, context, depth + 1, min_segments, visit);
                return;
            }
        }
        let criteria = context.criteria;
        let segments = bounds
            .segments
            .unwrap_or_else(|| chord_segments(bounds.chord_constant, criteria.max_chord_deviation))
            .max(min_segments);
        let turning = match criteria.angle_limit() {
            Some(_) => bounds.turning,
            None => self.turning(context),
        };
        let run = Run {
            first: self.first,
            last: self.last,
            segments,
            chord_deviation: chord_deviation(bounds.chord_constant, segments),
            angle_deviation: turning.map_or(PI, |t| angle_deviation(t, segments)),
        };
        visit(run, &|s| self.point_at(s));
    }
}

/// A piece's bounds for the criteria.
#[derive(Clone, Copy)]
struct Bounds {
    /// See [`Piece::chord_constant`].
    chord_constant: f64,
    /// See [`Piece::turning`]; computed only if the angle deviation is
    /// limited.
    turning: Option<f64>,
    /// The fewest equal steps meeting the criteria, `None` if the angle
    /// deviation is limited but has no bound.
    segments: Option<usize>,
}

/// Entries of a row of binomial coefficients, up to `C(98, k)`: the product
/// of two tangent directions of a rational piece of degree 25, each of
/// degree `2 × 25 − 1`.
const ROW: usize = 2 * DIRECTION_CAPACITY - 1;

/// The products of Bernstein polynomials of degrees `p` and `q`:
/// `B_i^p B_j^q = C(p, i) C(q, j) / C(p + q, i + j) B_{i+j}^{p+q}`.
struct Products {
    first: [f64; ROW],
    second: [f64; ROW],
    inverse_product: [f64; ROW],
}

impl Products {
    fn new(p: usize, q: usize) -> Self {
        let binomials = |m: usize| {
            let mut row = [0.0; ROW];
            row[0] = 1.0;
            for k in 0..m {
                row[k + 1] = row[k] * (m - k) as f64 / (k + 1) as f64;
            }
            row
        };
        let mut inverse_product = binomials(p + q);
        for x in &mut inverse_product[..=p + q] {
            *x = 1.0 / *x;
        }
        Self {
            first: binomials(p),
            second: binomials(q),
            inverse_product,
        }
    }

    /// The factor of `B_i^p B_j^q` in `B_{i+j}^{p+q}`.
    fn factor(&self, i: usize, j: usize) -> f64 {
        self.first[i] * self.second[j] * self.inverse_product[i + j]
    }
}

/// What the pieces of one curve share: the criteria, and the products of
/// Bernstein polynomials their tangents need.
struct Context<'a> {
    criteria: &'a DiscretizationCriteria,
    /// For a rational curve of degree `n`, degree `n − 1` by degree `n`:
    /// the tangent's direction `X' W − X W'`.
    direction_products: Products,
    /// For a tangent's direction `D` of degree `m`, degree `m` by degree
    /// `m − 1`: `D × D'`.
    cross_products: Products,
    /// Degree `m` by degree `m`: `D · D`.
    dot_products: Products,
}

impl<'a> Context<'a> {
    fn new(criteria: &'a DiscretizationCriteria, degree: usize, rational: bool) -> Self {
        let n = degree.max(1);
        let m = if rational { 2 * n - 1 } else { n - 1 };
        Self {
            criteria,
            direction_products: Products::new(n - 1, if rational { n } else { 0 }),
            cross_products: Products::new(m, m.saturating_sub(1)),
            dot_products: Products::new(m, m),
        }
    }
}

/// The turning of a curve over `[0, 1]` whose tangent has the direction
/// of the polynomial `D` with Bernstein coefficients `d`, of degree `m`,
/// the direction turning at the rate `|D × D'| / |D|²`.
///
/// Two bounds of it are taken, the smaller kept. First, it is at most
/// `|D'| / |D|`, `|D'|` at most `m max |Δd_j|`, and `|D|` at least the
/// smallest component of the `d_j` along their mean direction, if
/// positive and not negligible (see [`MIN_SPEED_RATIO`]; otherwise there
/// is no bound). Second, `D × D'` and `D · D` are polynomials of degree
/// `2m − 1` and `2m`, bounded by their Bernstein coefficients: the
/// largest norm of the first over the smallest of the second, if
/// positive. The second leaves out the change of `D` along itself, which
/// does not turn the tangent.
fn turning(d: &[[f64; 3]], context: &Context) -> Option<f64> {
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
    let mut change = 0.0f64;
    let mut cross = [[0.0; 3]; ROW];
    let mut dot = [0.0; ROW];
    for j in 0..m {
        let difference: [f64; 3] = std::array::from_fn(|c| d[j + 1][c] - d[j][c]);
        change = change.max(norm(difference));
        for (i, v) in d.iter().enumerate() {
            let f = m as f64 * context.cross_products.factor(i, j);
            let product = &mut cross[i + j];
            product[0] += f * (v[1] * difference[2] - v[2] * difference[1]);
            product[1] += f * (v[2] * difference[0] - v[0] * difference[2]);
            product[2] += f * (v[0] * difference[1] - v[1] * difference[0]);
        }
    }
    // Symmetric: each pair once, off the diagonal twice.
    for (i, v) in d.iter().enumerate() {
        for (j, w) in d.iter().enumerate().skip(i) {
            let f = if i == j { 1.0 } else { 2.0 } * context.dot_products.factor(i, j);
            dot[i + j] += f * (v[0] * w[0] + v[1] * w[1] + v[2] * w[2]);
        }
    }
    let by_speed = m as f64 * change / slowest;
    let slowest_squared = dot[..=2 * m].iter().copied().fold(f64::INFINITY, f64::min);
    if slowest_squared > 0.0 {
        let squared = |v: &[f64; 3]| v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        let fastest_turn = cross[..2 * m]
            .iter()
            .map(squared)
            .fold(0.0, f64::max)
            .sqrt();
        Some(by_speed.min(fastest_turn / slowest_squared))
    } else {
        Some(by_speed)
    }
}

/// The angle deviation of each of `segments` equal steps of a piece whose
/// tangent turns by at most `turning`, `Θ` over a step.
///
/// While `Θ <= π / 2`, it is at most `Θ / 2`. The chord's direction is the
/// mean of the unit tangents `T(s)` over the step (weighted by speed); at
/// angles `α(s) <= Θ` from the tangent `T(t)`, the chord makes an angle
/// with `T(t)` whose tangent is at most `mean sin α / mean cos α`, which
/// grows with each `α(s)`. The angles are largest, `α(s) = Θ |s − t| / h`
/// over a step of length `h`, for a circular arc seen from an end, where
/// the chord is at `Θ / 2`. Beyond, the tangent and chord differ by at
/// most `Θ`, the chord lying among the tangents; no angle exceeds π.
fn angle_deviation(turning: f64, segments: usize) -> f64 {
    let step = turning / segments as f64;
    if step <= FRAC_PI_2 {
        0.5 * step
    } else {
        step.min(PI)
    }
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
        let context = Context::new(criteria, self.degree(), self.is_rational());
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
            let piece = piece.restricted(a, b);
            let bounds = piece.bounds(&context);
            piece.visit_leaves(bounds, &context, 0, min_segments, &mut visit);
        });
    }
}
