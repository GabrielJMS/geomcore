//! Bézier pieces of B-splines, and the tests on pairs of them shared by the
//! subdivision intersectors: control-point boxes, separating axes, and a
//! cone test proving that a curve piece meets a surface piece at most once.
//!
//! Every piece lies in the convex hull of its Euclidean control points
//! (weights are positive), so a box or slab around those points holds the
//! piece, and halving a piece by de Casteljau shrinks the hull onto it.

use crate::curves::BSplineCurve3D;
use crate::math::boxtree::Aabb;
use crate::surface_math::bspline::{self as surface_math, Interval};

/// The Bézier pieces of a B-spline curve over a parameter range: one per
/// knot span the range meets, cut to the range.
pub(crate) struct CurvePieces {
    /// Control points per piece: the degree plus one.
    pub(crate) width: usize,
    /// Homogeneous control points `(x w, y w, z w, w)` (`w = 1` on a
    /// polynomial curve), `width` per piece, flattened.
    pub(crate) points: Vec<f64>,
    /// The parameter interval of each piece.
    pub(crate) ranges: Vec<(f64, f64)>,
}

impl CurvePieces {
    /// The pieces of `curve` over `[lo, hi]`, which lies within its bounds
    /// unless the curve is periodic; a periodic range may cross the seam.
    pub(crate) fn new(curve: &BSplineCurve3D, (lo, hi): (f64, f64)) -> Self {
        let width = curve.degree() + 1;
        let knots = curve.knots();
        let period = curve
            .is_periodic()
            .then(|| knots[knots.len() - 1] - knots[0]);
        let mut breaks = vec![lo];
        match period {
            None => breaks.extend(knots.iter().copied().filter(|&k| k > lo && k < hi)),
            Some(period) => {
                let mut copy = ((lo - knots[0]) / period).floor();
                while knots[0] + copy * period < hi {
                    let shift = copy * period;
                    breaks.extend(
                        knots[..knots.len() - 1]
                            .iter()
                            .map(|&k| k + shift)
                            .filter(|&k| k > lo && k < hi),
                    );
                    copy += 1.0;
                }
            }
        }
        breaks.push(hi);
        let n = breaks.len() - 1;
        let mut points = Vec::with_capacity(n * width * 4);
        let mut ranges = Vec::with_capacity(n);
        let mut span = vec![[0.0; 4]; width];
        for w in breaks.windows(2) {
            let (t0, t1) = (w[0], w[1]);
            let mid = 0.5 * (t0 + t1);
            curve.span_bezier_points(mid, &mut span);
            let (a, b) = span_around(knots, period, mid);
            restrict(&mut span, (t0 - a) / (b - a), (t1 - a) / (b - a));
            points.extend(span.iter().flatten());
            ranges.push((t0, t1));
        }
        CurvePieces {
            width,
            points,
            ranges,
        }
    }

    /// The number of pieces.
    pub(crate) fn len(&self) -> usize {
        self.ranges.len()
    }

    /// The homogeneous control points of piece `i`, flattened.
    pub(crate) fn piece(&self, i: usize) -> &[f64] {
        let size = 4 * self.width;
        &self.points[i * size..(i + 1) * size]
    }
}

/// The knot span `[a, b]` containing `t`, the knots repeated every
/// `period` on a periodic curve.
fn span_around(knots: &[f64], period: Option<f64>, t: f64) -> (f64, f64) {
    let shift = period.map_or(0.0, |p| ((t - knots[0]) / p).floor() * p);
    let j = knots
        .partition_point(|&k| k + shift <= t)
        .clamp(1, knots.len() - 1);
    (knots[j - 1] + shift, knots[j] + shift)
}

/// Cuts the Bézier control points `points`, over `[0, 1]`, to the
/// sub-interval `[f0, f1]` by two de Casteljau subdivisions.
fn restrict(points: &mut [[f64; 4]], f0: f64, f1: f64) {
    let p = points.len() - 1;
    let lerp = |a: [f64; 4], b: [f64; 4], f: f64| std::array::from_fn(|c| a[c] + f * (b[c] - a[c]));
    if f0 > 0.0 {
        // Point `i` ends as the right part's point `i`.
        for r in 1..=p {
            for i in 0..=p - r {
                points[i] = lerp(points[i], points[i + 1], f0);
            }
        }
    }
    let g = if f0 > 0.0 { (f1 - f0) / (1.0 - f0) } else { f1 };
    if g < 1.0 {
        // Point `i` ends as the left part's point `i`.
        for r in 1..=p {
            for i in (r..=p).rev() {
                points[i] = lerp(points[i - 1], points[i], g);
            }
        }
    }
}

/// Storage for pieces of one size, in slots reused once released.
pub(crate) struct Pool {
    size: usize,
    data: Vec<f64>,
    free: Vec<usize>,
    /// A copy of the piece being halved.
    scratch: Vec<f64>,
}

impl Pool {
    /// An empty pool of slots of `size` values.
    pub(crate) fn new(size: usize) -> Self {
        Pool {
            size,
            data: Vec::new(),
            free: Vec::new(),
            scratch: vec![0.0; size],
        }
    }

    /// A free slot, its values unspecified.
    pub(crate) fn take(&mut self) -> usize {
        self.free.pop().unwrap_or_else(|| {
            let slot = self.data.len();
            self.data.resize(slot + self.size, 0.0);
            slot
        })
    }

    /// Makes room for `slots` more slots without reallocating.
    pub(crate) fn reserve(&mut self, slots: usize) {
        self.data.reserve(slots * self.size);
    }

    /// Releases `slot` for reuse.
    pub(crate) fn release(&mut self, slot: usize) {
        self.free.push(slot);
    }

    pub(crate) fn get(&self, slot: usize) -> &[f64] {
        &self.data[slot..slot + self.size]
    }

    pub(crate) fn get_mut(&mut self, slot: usize) -> &mut [f64] {
        &mut self.data[slot..slot + self.size]
    }

    /// Copies the values of slot `from` into slot `to`.
    pub(crate) fn copy(&mut self, from: usize, to: usize) {
        self.data.copy_within(from..from + self.size, to);
    }

    /// Halves the Bernstein net in `slot`, row-major `[rows][cols][h]`,
    /// at the midpoint along the rows if `along_rows` (see
    /// [`surface_math::halve_net`]): `slot` receives the lower half and
    /// the returned slot the upper half. A curve is a net of one column.
    pub(crate) fn halve(
        &mut self,
        slot: usize,
        (rows, cols, h): (usize, usize, usize),
        along_rows: bool,
    ) -> usize {
        let other = self.take();
        let size = self.size;
        self.scratch.copy_from_slice(&self.data[slot..slot + size]);
        let (lower, upper) = if slot < other {
            let (a, b) = self.data.split_at_mut(other);
            (&mut a[slot..slot + size], &mut b[..size])
        } else {
            let (a, b) = self.data.split_at_mut(slot);
            (&mut b[..size], &mut a[other..other + size])
        };
        surface_math::halve_net(&self.scratch, rows, cols, h, along_rows, lower, upper);
        other
    }
}

/// The Euclidean points of homogeneous `(x w, y w, z w, w)` points.
pub(crate) fn euclidean(homogeneous: &[f64], out: &mut Vec<[f64; 3]>) {
    surface_math::net_points(homogeneous, 4, out);
}

/// The axis-aligned box of `points`.
pub(crate) fn bounding_box(points: &[[f64; 3]]) -> Aabb {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for x in points {
        for c in 0..3 {
            b[c] = b[c].min(x[c]);
            b[c + 3] = b[c + 3].max(x[c]);
        }
    }
    b
}

/// The length of the box's diagonal.
pub(crate) fn box_size(b: &Aabb) -> f64 {
    ((b[3] - b[0]).powi(2) + (b[4] - b[1]).powi(2) + (b[5] - b[2]).powi(2)).sqrt()
}

/// Whether the projections of `a` and `b` onto `axis` are more than `gap`
/// (times the axis length) apart: then so are their convex hulls.
pub(crate) fn apart_along(axis: [f64; 3], a: &[[f64; 3]], b: &[[f64; 3]], gap: f64) -> bool {
    let ((a0, a1), (b0, b1)) = (extent(a, axis), extent(b, axis));
    let gap = gap * dot(axis, axis).sqrt();
    a0 > b1 + gap || b0 > a1 + gap
}

/// The range of the projections of `points` onto `axis`.
pub(crate) fn extent(points: &[[f64; 3]], axis: [f64; 3]) -> (f64, f64) {
    points
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
            let y = dot(*x, axis);
            (lo.min(y), hi.max(y))
        })
}

/// Whether two curve pieces, with the Euclidean control points `a` and
/// `b`, meet at most once.
///
/// Two meetings `A(s1) = B(t1)` and `A(s2) = B(t2)` would give `(s2 - s1)
/// c = (t2 - t1) d` with `c` and `d` integrals of tangents along the
/// pieces, so in their tangent cones: some tangent of one would be
/// parallel to one of the other. The cones (see [`tangent_cone`]) rule
/// that out when the angle between their axes, as lines, exceeds the sum
/// of their half-angles.
pub(crate) fn curves_meet_at_most_once(a: &[[f64; 3]], b: &[[f64; 3]]) -> bool {
    let (Some((axis_a, half_a)), Some((axis_b, half_b))) = (tangent_cone(a), tangent_cone(b))
    else {
        return false;
    };
    let between = norm(cross(axis_a, axis_b)).atan2(dot(axis_a, axis_b).abs());
    half_a + half_b < between
}

/// The cone holding the tangents of the curve piece with the Euclidean
/// control points `points`, positive combinations of their consecutive
/// differences (see [`meets_at_most_once`]): its axis, the chord, and its
/// half-angle, below a right angle; `None` if the chord vanishes or a
/// difference turns away from it.
fn tangent_cone(points: &[[f64; 3]]) -> Option<([f64; 3], f64)> {
    let axis = unit(sub(points[points.len() - 1], points[0]))?;
    let mut half = 0.0f64;
    for d in points.windows(2).filter_map(|w| unit(sub(w[1], w[0]))) {
        let cos = dot(d, axis);
        if cos <= 0.0 {
            return None;
        }
        half = half.max(norm(cross(d, axis)).atan2(cos));
    }
    Some((axis, half))
}

/// The sign of `(x - y) . d` over every `x` in the convex hull of
/// `curve`, `y` in that of `other`, and `d` tangent to the curve piece
/// with the Euclidean control points `curve`: `Some(1.0)` or
/// `Some(-1.0)` when the hulls are apart along each consecutive
/// difference of `curve` (tangents are positive combinations of them, see
/// [`meets_at_most_once`]), always in the same order; else `None`, the
/// product possibly vanishing.
///
/// With `x` on the curve and `y` on another shape, the product is the
/// derivative along the curve of half their squared distance.
pub(crate) fn tangent_sign(curve: &[[f64; 3]], other: &[[f64; 3]]) -> Option<f64> {
    let mut sign = None;
    for w in curve.windows(2) {
        let d = sub(w[1], w[0]);
        if d == [0.0; 3] {
            continue;
        }
        let ((a0, a1), (b0, b1)) = (extent(curve, d), extent(other, d));
        let side = if a0 > b1 {
            1.0
        } else if a1 < b0 {
            -1.0
        } else {
            return None;
        };
        if sign.is_some_and(|s| s != side) {
            return None;
        }
        sign = Some(side);
    }
    sign
}

/// The normal of the plane fitted to the corners of a net of Euclidean
/// points, row-major `[rows][cols]`: the cross product of its mean
/// directions along the rows and along the columns.
pub(crate) fn net_normal(points: &[[f64; 3]], rows: usize, cols: usize) -> [f64; 3] {
    let [along_s, along_t] = net_directions(points, rows, cols);
    cross(along_s, along_t)
}

/// The mean directions of a net along its rows (`s`) and its columns
/// (`t`), from its corners.
fn net_directions(points: &[[f64; 3]], rows: usize, cols: usize) -> [[f64; 3]; 2] {
    let corner = |i: usize, j: usize| points[i * cols + j];
    let (c00, c01) = (corner(0, 0), corner(0, cols - 1));
    let (c10, c11) = (corner(rows - 1, 0), corner(rows - 1, cols - 1));
    [
        add(sub(c10, c00), sub(c11, c01)),
        add(sub(c01, c00), sub(c11, c10)),
    ]
}

/// Whether a curve piece and a surface piece meet at most once (in
/// space: the surface may reach one point from several parameters where
/// its parametrisation is singular).
///
/// The curve's tangents lie in a cone `K_c` and the surface's partial
/// derivatives in cones `K_s` and `K_t`. Two meetings `C(t1) = S(s1, t1')`
/// and `C(t2) = S(s2, t2')` would give `(t2 - t1) c = (s2 - s1) a + (t2' -
/// t1') b` with `c`, `a`, `b` integrals of derivatives along straight
/// parameter paths, so in those cones. That cannot happen when every
/// triple of directions from the three cones is linearly independent.
/// With the cones' axes (unit) and their spreads (bounds on the distance
/// from the axis to any unit direction in them), the determinant of a
/// triple differs from that of the axes by at most the sum of the
/// spreads, which therefore must stay below it.
///
/// The curve piece has the Euclidean control points `curve`; its
/// derivative is a positive combination of their differences `P_j - P_i`,
/// `j > i` (of consecutive ones if it is polynomial), so of consecutive
/// ones either way, the others being sums of them. The surface piece has the Bernstein
/// net `net` (`h` values per point, homogeneous when `h == 4`) and the
/// Euclidean control points `points`, row-major `[rows][cols]`; its
/// derivatives lie in the cones of the differences along its rows and
/// along its columns if it is polynomial, else in the boxes that
/// [`surface_math::derivative_bounds`] encloses them in.
pub(crate) fn meets_at_most_once(
    curve: &[[f64; 3]],
    net: &[f64],
    points: &[[f64; 3]],
    (rows, cols, h): (usize, usize, usize),
) -> bool {
    let p = curve.len() - 1;
    let Some(c) = unit(sub(curve[p], curve[0])) else {
        return false;
    };
    let [along_s, along_t] = net_directions(points, rows, cols);
    let (Some(a), Some(b)) = (unit(along_s), unit(along_t)) else {
        return false;
    };
    let det = dot(c, cross(a, b)).abs();
    let spread = |axis: [f64; 3], directions: &mut dyn Iterator<Item = [f64; 3]>| {
        directions
            .filter_map(unit)
            .map(|d| norm(sub(d, axis)))
            .fold(0.0f64, f64::max)
    };
    let c_spread = spread(c, &mut (0..p).map(|i| sub(curve[i + 1], curve[i])));
    if c_spread >= det {
        return false;
    }
    let (s_spread, t_spread) = if h == 4 {
        let (_, d) = surface_math::derivative_bounds(net, points, rows, cols, h, false);
        (box_spread(a, d[0]), box_spread(b, d[1]))
    } else {
        let at = |i: usize, j: usize| points[i * cols + j];
        (
            spread(
                a,
                &mut (0..rows - 1)
                    .flat_map(|i| (0..cols).map(move |j| sub(at(i + 1, j), at(i, j)))),
            ),
            spread(
                b,
                &mut (0..rows)
                    .flat_map(|i| (0..cols - 1).map(move |j| sub(at(i, j + 1), at(i, j)))),
            ),
        )
    };
    c_spread + s_spread + t_spread < det
}

/// A cone of directions: its axis (unit) and its spread, a bound on the
/// distance from the axis to any unit direction in it.
pub(crate) type Cone = ([f64; 3], f64);

/// The cones of the partial derivatives `[S_s, S_t]` of the surface piece
/// with the Bernstein net `net` and the Euclidean control points
/// `points`, as in [`meets_at_most_once`]; `None` when a direction of
/// the net degenerates, or a difference of control points vanishes (a
/// derivative may then vanish too).
pub(crate) fn derivative_cones(
    net: &[f64],
    points: &[[f64; 3]],
    (rows, cols, h): (usize, usize, usize),
) -> Option<[Cone; 2]> {
    let [along_s, along_t] = net_directions(points, rows, cols);
    let (a, b) = (unit(along_s)?, unit(along_t)?);
    if h == 4 {
        let (_, d) = surface_math::derivative_bounds(net, points, rows, cols, h, false);
        return Some([(a, box_spread(a, d[0])), (b, box_spread(b, d[1]))]);
    }
    let at = |i: usize, j: usize| points[i * cols + j];
    let s_spread = spread_of_all(
        a,
        (0..rows - 1).flat_map(|i| (0..cols).map(move |j| sub(at(i + 1, j), at(i, j)))),
    )?;
    let t_spread = spread_of_all(
        b,
        (0..rows).flat_map(|i| (0..cols - 1).map(move |j| sub(at(i, j + 1), at(i, j)))),
    )?;
    Some([(a, s_spread), (b, t_spread)])
}

/// The largest distance from `axis` (unit) to the unit `directions`, or
/// `None` when one of them is zero.
fn spread_of_all(axis: [f64; 3], mut directions: impl Iterator<Item = [f64; 3]>) -> Option<f64> {
    directions.try_fold(0.0f64, |s, d| Some(s.max(norm(sub(unit(d)?, axis)))))
}

/// Whether no tangent plane of one surface piece is parallel to a tangent
/// plane of the other, from the cones of their partial derivatives
/// ([`derivative_cones`]).
///
/// Parallel tangent planes `span(S_s, S_t)` and `span(R_s, R_t)` put
/// `R_s` and `R_t` in the first, so `det(S_s, S_t, R_s) = 0`; the cones
/// rule that out when the determinant of their axes exceeds the sum of
/// their spreads, as in [`meets_at_most_once`]. Then the pieces' normals
/// are never parallel, and their intersection has no closed loop within
/// both (Sederberg and Meyers, "Loop detection in surface patch
/// intersections", 1988: the discs a closed loop bounds on the two
/// surfaces hold points with parallel normals).
pub(crate) fn normals_never_parallel(s: [Cone; 2], r: [Cone; 2]) -> bool {
    let independent =
        |[(a, sa), (b, sb)]: [Cone; 2], (c, sc): Cone| dot(c, cross(a, b)).abs() > sa + sb + sc;
    r.iter().any(|&c| independent(s, c)) || s.iter().any(|&c| independent(r, c))
}

/// Whether a net of Euclidean points, row-major `[rows][cols]`, stretches
/// along its rows (`s`) at least as much as along its columns: the mean
/// length of its polylines in each direction, compared.
pub(crate) fn stretches_along_rows(points: &[[f64; 3]], rows: usize, cols: usize) -> bool {
    let at = |i: usize, j: usize| points[i * cols + j];
    let along_rows: f64 = (0..rows - 1)
        .flat_map(|i| (0..cols).map(move |j| norm(sub(at(i + 1, j), at(i, j)))))
        .sum();
    let along_cols: f64 = (0..rows)
        .flat_map(|i| (0..cols - 1).map(move |j| norm(sub(at(i, j + 1), at(i, j)))))
        .sum();
    along_rows * rows as f64 >= along_cols * cols as f64
}

/// The thickness of `points` along `axis`: the width of their projections
/// on it, over its length (infinite on a zero axis).
pub(crate) fn thickness(points: &[[f64; 3]], axis: [f64; 3]) -> f64 {
    let Some(n) = unit(axis) else {
        return f64::INFINITY;
    };
    let (lo, hi) = points
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
            let y = dot(*x, n);
            (lo.min(y), hi.max(y))
        });
    hi - lo
}

/// The spread about `axis` (unit) of the directions in the box `b`: the
/// largest distance from `axis` to a unit corner direction, the angle to
/// the axis being quasi-convex, or infinite when the box holds the
/// origin.
fn box_spread(axis: [f64; 3], b: [Interval; 3]) -> f64 {
    if b.iter().all(|x| x.lo <= 0.0 && x.hi >= 0.0) {
        return f64::INFINITY;
    }
    let mut spread = 0.0f64;
    for corner in 0..8 {
        let x = std::array::from_fn(|c| {
            if corner >> c & 1 == 0 {
                b[c].lo
            } else {
                b[c].hi
            }
        });
        match unit(x) {
            Some(d) if dot(d, axis) > 0.0 => spread = spread.max(norm(sub(d, axis))),
            _ => return f64::INFINITY,
        }
    }
    spread
}

pub(crate) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn unit(a: [f64; 3]) -> Option<[f64; 3]> {
    let n = norm(a);
    (n > 0.0 && n.is_finite()).then(|| [a[0] / n, a[1] / n, a[2] / n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point3D;
    use crate::math::boxtree::boxes_apart;

    /// The point at `f` in `[0, 1]` of the homogeneous Bézier `piece`.
    fn bezier_point(piece: &[f64], f: f64) -> Point3D {
        let mut points: Vec<[f64; 4]> = piece.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
        for r in 1..points.len() {
            for i in 0..points.len() - r {
                points[i] =
                    std::array::from_fn(|c| points[i][c] + f * (points[i + 1][c] - points[i][c]));
            }
        }
        let [x, y, z, w] = points[0];
        Point3D::new(x / w, y / w, z / w)
    }

    #[track_caller]
    fn assert_pieces_follow(curve: &BSplineCurve3D, range: (f64, f64), breaks: &[f64]) {
        let pieces = CurvePieces::new(curve, range);
        assert_eq!(pieces.width, curve.degree() + 1);
        let ends: Vec<f64> = std::iter::once(range.0)
            .chain(pieces.ranges.iter().map(|r| r.1))
            .collect();
        assert_eq!(ends, breaks);
        for i in 0..pieces.len() {
            let (t0, t1) = pieces.ranges[i];
            for f in [0.0, 0.3, 0.5, 1.0] {
                let expected = curve.eval_point(t0 + f * (t1 - t0));
                assert!(bezier_point(pieces.piece(i), f).distance(expected) < 1e-12);
            }
        }
    }

    fn knotted(weights: Option<Vec<f64>>) -> BSplineCurve3D {
        let poles: Vec<Point3D> = (0..6)
            .map(|k| Point3D::new(k as f64, (k * k % 5) as f64, (k % 3) as f64))
            .collect();
        let (knots, mults) = (vec![0.0, 1.0, 1.5, 3.0], vec![4, 1, 1, 4]);
        match weights {
            None => BSplineCurve3D::new(3, poles, knots, mults, false),
            Some(w) => BSplineCurve3D::new_rational(3, poles, w, knots, mults, false),
        }
        .unwrap()
    }

    #[test]
    fn test_curve_pieces_follow_the_curve() {
        let weights = vec![1.0, 0.5, 2.0, 1.5, 0.7, 1.0];
        for curve in [knotted(None), knotted(Some(weights))] {
            assert_pieces_follow(&curve, (0.0, 3.0), &[0.0, 1.0, 1.5, 3.0]);
            assert_pieces_follow(&curve, (0.2, 1.2), &[0.2, 1.0, 1.2]);
            assert_pieces_follow(&curve, (1.1, 1.4), &[1.1, 1.4]);
            assert_pieces_follow(&curve, (1.0, 1.5), &[1.0, 1.5]);
        }
    }

    #[test]
    fn test_curve_pieces_cross_a_periodic_seam() {
        let poles = (0..6)
            .map(|k| {
                let angle = k as f64 * std::f64::consts::TAU / 6.0;
                Point3D::new(2.0 * angle.cos(), 2.0 * angle.sin(), (k % 2) as f64)
            })
            .collect();
        let knots = (0..=6).map(f64::from).collect();
        let ring = BSplineCurve3D::new(3, poles, knots, vec![1; 7], true).unwrap();
        assert_pieces_follow(&ring, (0.0, 6.0), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_pieces_follow(&ring, (4.5, 7.25), &[4.5, 5.0, 6.0, 7.0, 7.25]);
        assert_pieces_follow(&ring, (-1.5, 0.5), &[-1.5, -1.0, 0.0, 0.5]);
    }

    #[test]
    fn test_pool_halves_into_two_slots() {
        let mut pool = Pool::new(4 * 3);
        let slot = pool.take();
        // A quadratic from (0, 0, 0) to (4, 0, 0) through control (2, 2, 0).
        pool.get_mut(slot)
            .copy_from_slice(&[0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 0.0, 1.0, 4.0, 0.0, 0.0, 1.0]);
        let upper = pool.halve(slot, (3, 1, 4), true);
        assert_ne!(upper, slot);
        assert_eq!(
            pool.get(slot),
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0]
        );
        assert_eq!(
            pool.get(upper),
            &[2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0, 0.0, 0.0, 1.0]
        );
        pool.release(slot);
        assert_eq!(pool.take(), slot);
    }

    const SQUARE: [[f64; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ];

    #[test]
    fn test_separating_axis_along_the_patch_normal() {
        // The plane `z = x + y`, and points 0.5 and 0.6 above it.
        let tilted = SQUARE.map(|[x, y, _]| [x, y, x + y]);
        let normal = net_normal(&tilted, 2, 2);
        let above = [[0.9, 0.1, 1.5], [0.1, 0.9, 1.6]];
        let apart = |gap: f64| boxes_apart(&bounding_box(&above), &bounding_box(&tilted), gap);
        assert!(!apart(0.0));
        assert!(apart_along(normal, &above, &tilted, 0.25));
        assert!(!apart_along(normal, &above, &tilted, 0.3));
    }

    fn flatten(points: &[[f64; 3]]) -> Vec<f64> {
        points.iter().flatten().copied().collect()
    }

    fn cones(points: &[[f64; 3]]) -> Option<[Cone; 2]> {
        derivative_cones(&flatten(points), points, (2, 2, 3))
    }

    #[test]
    fn test_normals_never_parallel() {
        let flat = cones(&SQUARE).unwrap();
        let tilted = cones(&SQUARE.map(|[x, y, _]| [x, y, x])).unwrap();
        let lifted = cones(&SQUARE.map(|[x, y, _]| [x, y, 1.0])).unwrap();
        // Crossing planes never have parallel normals; parallel ones do.
        assert!(normals_never_parallel(flat, tilted));
        assert!(normals_never_parallel(tilted, flat));
        assert!(!normals_never_parallel(flat, lifted));
        // A slightly warped square: never parallel to the tilted plane, but
        // parallel at one point to a shallow one.
        let warped = cones(&[
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.1],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, -0.1],
        ])
        .unwrap();
        let shallow = cones(&SQUARE.map(|[x, y, _]| [x, y, -0.05 * x])).unwrap();
        assert!(normals_never_parallel(warped, tilted));
        assert!(!normals_never_parallel(warped, shallow));
        // The rational square's cones come from interval bounds.
        let rational: Vec<f64> = SQUARE
            .iter()
            .zip([1.0, 1.2, 0.9, 1.1])
            .flat_map(|([x, y, z], w)| [x * w, y * w, z * w, w])
            .collect();
        let weighted = derivative_cones(&rational, &SQUARE, (2, 2, 4)).unwrap();
        assert!(normals_never_parallel(weighted, tilted));
        assert!(!normals_never_parallel(weighted, lifted));
    }

    #[test]
    fn test_derivative_cones_reject_coincident_points() {
        let collapsed = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        assert!(cones(&collapsed).is_none());
        let line = SQUARE.map(|[x, _, _]| [x, 0.0, 0.0]);
        assert!(cones(&line).is_none());
    }

    #[test]
    fn test_stretch_and_thickness() {
        let long_in_s = SQUARE.map(|[x, y, z]| [4.0 * x, y, z]);
        assert!(stretches_along_rows(&long_in_s, 2, 2));
        let long_in_t = SQUARE.map(|[x, y, z]| [x, 4.0 * y, z]);
        assert!(!stretches_along_rows(&long_in_t, 2, 2));
        let tilted = SQUARE.map(|[x, y, _]| [x, y, x + y]);
        assert!(thickness(&tilted, net_normal(&tilted, 2, 2)) < 1e-15);
        assert_eq!(thickness(&tilted, [0.0, 0.0, 3.0]), 2.0);
        assert_eq!(thickness(&tilted, [0.0; 3]), f64::INFINITY);
    }

    #[test]
    fn test_meets_at_most_once() {
        let polynomial: Vec<f64> = SQUARE.iter().flatten().copied().collect();
        // The same square, rational: weighted corners, still flat.
        let rational: Vec<f64> = SQUARE
            .iter()
            .zip([1.0, 1.2, 0.9, 1.1])
            .flat_map(|([x, y, z], w)| [x * w, y * w, z * w, w])
            .collect();
        for (net, h) in [(polynomial, 3), (rational, 4)] {
            let once = |curve: &[[f64; 3]]| meets_at_most_once(curve, &net, &SQUARE, (2, 2, h));
            // Steep through the square: once, even curving a little.
            assert!(once(&[[0.5, 0.5, -1.0], [0.6, 0.4, 0.0], [0.5, 0.5, 1.0]]));
            // Bending back, or lying in its plane: maybe more.
            assert!(!once(&[
                [0.5, 0.5, -1.0],
                [0.5, 0.5, 1.0],
                [0.6, 0.5, -1.0]
            ]));
            assert!(!once(&[[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]]));
        }
        // A line meets a plane once, however shallow; a polynomial net's
        // differences say so, the rational one's interval bounds need not.
        let net: Vec<f64> = SQUARE.iter().flatten().copied().collect();
        assert!(meets_at_most_once(
            &[[0.0, 0.5, -0.01], [1.0, 0.5, 0.01]],
            &net,
            &SQUARE,
            (2, 2, 3)
        ));
    }
}
