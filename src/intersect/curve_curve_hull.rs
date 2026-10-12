//! Intersection and distance minima of two B-spline curves by subdivision
//! of their Bézier pieces (see [`bspline_curve_curve`] and
//! [`bspline_minima`]).

use super::CurveCurveExtremum;
use super::curve_surface_hull::Range;
use super::pieces::{
    CurvePieces, Pool, apart_along, bounding_box, box_size, cross, curves_meet_at_most_once, dot,
    euclidean, extent, norm, sub, tangent_sign,
};
use crate::Tolerance;
use crate::curve_math::bspline::LARGE;
use crate::curves::BSplineCurve3D;
use crate::math::boxtree::{Aabb, BoxTree, boxes_apart};
use crate::math::{newton_min_1d, newton_min_2d, solve_2x2};
use crate::surface_math::bspline::{Interval, derivative_bounds};

/// Halvings allowed per initial pair of pieces, pooled over all of them,
/// before a search gives up.
const SPLITS_PER_PAIR: usize = 256;
/// Halvings from a pair of pieces down to one subpiece pair.
const MAX_DEPTH: u32 = 64;
/// Gauss-Newton steps from the centre of a subpiece pair.
const NEWTON_STEPS: usize = 32;

/// The meetings of `a` over the range `ra` with `b` over `rb`, as extrema
/// at distances within `tol.confusion`, closest first; `None` when the
/// search cannot isolate them within its budget.
///
/// The curves' knot spans over their ranges are Bézier pieces; the pairs
/// whose control-point boxes come within `tol.confusion` of each other
/// are found through a box hierarchy over the second curve's pieces.
/// Each pair is then tested, and its larger piece halved, until it is
/// settled:
///
/// - dropped once the control points' boxes, or their projections on one
///   of three axes (normal to both chords, and normal to each chord
///   within the plane of both), are `tol.confusion` apart;
/// - once the pieces provably meet at most once (see
///   [`curves_meet_at_most_once`]), solved from the pair's centre by
///   Gauss-Newton steps minimising `|A(s) - B(t)|^2` on the B-splines
///   themselves (3D curves need not meet exactly). A hit within the pair
///   settles it, and so does one within `sqrt(tol.confusion)` of the
///   whole first piece, as the pair's hit would merge into it; any other
///   hit is still kept.
///
/// A tangency or an overlap never separates into such pairs: it exhausts
/// the budget or the depth limit.
pub(super) fn bspline_curve_curve(
    a: &BSplineCurve3D,
    ra: Range,
    b: &BSplineCurve3D,
    rb: Range,
    tol: Tolerance,
) -> Option<Vec<CurveCurveExtremum>> {
    let (pieces_a, pieces_b) = (
        CurvePieces::new(a, (ra.lo, ra.hi)),
        CurvePieces::new(b, (rb.lo, rb.hi)),
    );
    let mut pairs = Pairs::new(
        [pieces_a.width, pieces_b.width],
        pieces_a.len() + pieces_b.len(),
    );
    let mut points = Vec::new();
    let boxes_b: Vec<Aabb> = (0..pieces_b.len())
        .map(|j| {
            euclidean(pieces_b.piece(j), &mut points);
            bounding_box(&points)
        })
        .collect();
    let tree = BoxTree::new(&boxes_b, 1);
    for i in 0..pieces_a.len() {
        euclidean(pieces_a.piece(i), &mut points);
        tree.overlapping(&bounding_box(&points), tol.confusion, |j| {
            pairs.push(
                [pieces_a.piece(i), pieces_b.piece(j)],
                [pieces_a.ranges[i], pieces_b.ranges[j]],
            );
        });
    }
    let gap = tol.confusion;
    let merge_gap = tol.confusion.sqrt();
    let mut found = Vec::new();
    pairs
        .search(|pairs, node| {
            let [pa, pb] = &pairs.points;
            if boxes_apart(&bounding_box(pa), &bounding_box(pb), gap) || separated(pa, pb, gap) {
                return true;
            }
            if !curves_meet_at_most_once(pa, pb) {
                return false;
            }
            let [(s0, s1), (t0, t1)] = node.ranges;
            let Some(x) = closest_pair(a, b, [0.5 * (s0 + s1), 0.5 * (t0 + t1)]) else {
                return false;
            };
            let (Some(s), Some(t)) = (ra.admit(x[0]), rb.admit(x[1])) else {
                return false;
            };
            let point = a.eval_point(s);
            let distance = point.distance(b.eval_point(t));
            if distance > tol.confusion {
                return false;
            }
            found.push(CurveCurveExtremum {
                first_param: s,
                second_param: t,
                distance,
            });
            let within = |x: f64, (lo, hi): (f64, f64), range: Range| {
                lo - range.slack() <= x && x <= hi + range.slack()
            };
            let inside = within(x[0], (s0, s1), ra) && within(x[1], (t0, t1), rb);
            inside
                || pa
                    .iter()
                    .all(|p| norm(sub(*p, [point.x, point.y, point.z])) <= merge_gap)
        })
        .ok()?;
    found.sort_by(|p, q| p.distance.total_cmp(&q.distance));
    Some(found)
}

/// Whether the hulls of the curve pieces with the Euclidean control
/// points `a` and `b` are more than `gap` apart along the normal to both
/// chords or the normal to either chord in their plane: the axes that
/// part skew pieces, and pieces side by side in a plane.
fn separated(a: &[[f64; 3]], b: &[[f64; 3]], gap: f64) -> bool {
    let chord = |p: &[[f64; 3]]| sub(p[p.len() - 1], p[0]);
    let (ca, cb) = (chord(a), chord(b));
    let mut normal = cross(ca, cb);
    if normal == [0.0; 3] {
        normal = cross(ca, sub(b[0], a[0]));
    }
    [normal, cross(ca, normal), cross(cb, normal)]
        .into_iter()
        .any(|axis| apart_along(axis, a, b, gap))
}

/// Gauss-Newton minimisation of `|A(s) - B(t)|^2` from `x`, each step
/// halved until the distance decreases: the closest pair it reaches, or
/// `None` when the tangents are parallel or the steps do not settle.
fn closest_pair(a: &BSplineCurve3D, b: &BSplineCurve3D, mut x: [f64; 2]) -> Option<[f64; 2]> {
    let eval = |x: [f64; 2]| {
        let (p, dp) = a.eval_d1(x[0]);
        let (q, dq) = b.eval_d1(x[1]);
        (p - q, dp, dq)
    };
    let (mut f, mut da, mut db) = eval(x);
    for _ in 0..NEWTON_STEPS {
        if f.square_magnitude() == 0.0 {
            return Some(x);
        }
        let cross = -da.dot(db);
        let normal = [[da.dot(da), cross], [cross, db.dot(db)]];
        let step = solve_2x2(normal, [-da.dot(f), db.dot(f)], 1e-14)?;
        let mut scale = 1.0;
        loop {
            let trial = [x[0] + scale * step[0], x[1] + scale * step[1]];
            let (ft, dat, dbt) = eval(trial);
            if ft.square_magnitude() < f.square_magnitude() {
                (x, f, da, db) = (trial, ft, dat, dbt);
                break;
            }
            scale *= 0.5;
            if scale < 1e-3 {
                return Some(x);
            }
        }
        let moved = (scale * step[0]).abs() + (scale * step[1]).abs();
        if moved <= 1e-14 * (2.0 + x[0].abs() + x[1].abs()) {
            return Some(x);
        }
    }
    None
}

/// Seeds for the local minima of the distance between `a` over the range
/// `ra` and `b` over `rb`: parameter pairs at or next to every minimum,
/// for the caller to refine and check on the whole parameter box; `None`
/// when the search cannot isolate them within its budget.
///
/// A minimum inside the box is a stationary point of `f = |A(s) -
/// B(t)|^2 / 2` with a positive semidefinite Hessian. Every pair of the
/// curves' Bézier pieces is tested, and its larger piece halved, until it
/// is settled:
///
/// - dropped when `f_s` or `f_t` keeps one sign over it (see
///   [`tangent_sign`]), or its Hessian, enclosed by interval arithmetic
///   (see [`derivative_bounds`]), is nowhere positive semidefinite (a
///   saddle or a maximum);
/// - once its Hessian is positive definite throughout, `f` is strictly
///   convex on it, so it holds at most one stationary point, reached by
///   Newton's method bounded to the pair.
///
/// A minimum on an end of a (non-periodic) range is a minimum of the
/// distance from that end's point to the other curve, found the same way
/// along the other curve alone; the corners of the box are seeds too.
/// Valleys of equal distance (overlaps, concentric arcs) and tangencies
/// never become convex: they exhaust the budget or the depth limit.
pub(super) fn bspline_minima(
    a: &BSplineCurve3D,
    ra: Range,
    b: &BSplineCurve3D,
    rb: Range,
    tol: Tolerance,
) -> Option<Vec<[f64; 2]>> {
    let pieces = [
        CurvePieces::new(a, (ra.lo, ra.hi)),
        CurvePieces::new(b, (rb.lo, rb.hi)),
    ];
    let ends = |r: Range| if r.periodic { vec![] } else { vec![r.lo, r.hi] };
    let (ends_a, ends_b) = (ends(ra), ends(rb));
    let mut seeds: Vec<[f64; 2]> = ends_a
        .iter()
        .flat_map(|&s| ends_b.iter().map(move |&t| [s, t]))
        .collect();
    let mut pairs = Pairs::new(
        [pieces[0].width, pieces[1].width],
        pieces[0].len() * pieces[1].len(),
    );
    for i in 0..pieces[0].len() {
        for j in 0..pieces[1].len() {
            pairs.push(
                [pieces[0].piece(i), pieces[1].piece(j)],
                [pieces[0].ranges[i], pieces[1].ranges[j]],
            );
        }
    }
    pairs
        .search(|pairs, node| {
            let [pa, pb] = &pairs.points;
            if tangent_sign(pa, pb).is_some() || tangent_sign(pb, pa).is_some() {
                return true;
            }
            let (va, da) = curve_bounds(pairs.net(0, node), pa);
            let (vb, db) = curve_bounds(pairs.net(1, node), pb);
            let [mut h11, mut h12, mut h22] = [Interval::ZERO; 3];
            for c in 0..3 {
                let r = va[c].sub(vb[c]);
                h11 = h11.add(da[0][c].square()).add(r.mul(da[1][c]));
                h22 = h22.add(db[0][c].square()).sub(r.mul(db[1][c]));
                h12 = h12.add(da[0][c].mul(db[0][c]));
            }
            let off = h12.square();
            if h11.hi < 0.0 || h22.hi < 0.0 || h11.mul(h22).hi < off.lo {
                return true;
            }
            if !(h11.lo > 0.0 && h22.lo > 0.0 && h11.lo * h22.lo > off.hi) {
                return false;
            }
            let (net_a, net_b) = (pairs.net(0, node), pairs.net(1, node));
            let local = |u: f64, v: f64| {
                let [pa, a1, a2] = bezier_d2(net_a, u);
                let [pb, b1, b2] = bezier_d2(net_b, v);
                let r = sub(pa, pb);
                let cross = -dot(a1, b1);
                (
                    (
                        0.5 * dot(r, r),
                        [dot(r, a1), -dot(r, b1)],
                        [
                            [dot(a1, a1) + dot(r, a2), cross],
                            [cross, dot(b1, b1) - dot(r, b2)],
                        ],
                    ),
                    [norm(a1), norm(b1)],
                )
            };
            let unit = Some((0.0, 1.0));
            let x = newton_min_2d(|u, v| local(u, v).0, [0.5; 2], [unit; 2], 1e-14, 100);
            let ((_, g, _), speed) = local(x[0], x[1]);
            let stationary = (0..2)
                .all(|i| (0.0 < x[i] && x[i] < 1.0) || g[i].abs() <= tol.confusion * speed[i]);
            if stationary {
                let [(s0, s1), (t0, t1)] = node.ranges;
                seeds.push([s0 + x[0] * (s1 - s0), t0 + x[1] * (t1 - t0)]);
            }
            true
        })
        .ok()?;
    // The point at an end, and the direction in which the other curve's
    // point must lie for the distance to rise into the range there.
    let end = |curve: &BSplineCurve3D, u: f64, r: Range| {
        let (p, d) = curve.eval_d1(u);
        let hold = if u == r.lo { d } else { -d };
        ([p.x, p.y, p.z], [hold.x, hold.y, hold.z])
    };
    for &s in &ends_a {
        let (point, hold) = end(a, s, ra);
        let ts = minima_from_point(point, hold, &pieces[1], tol)?;
        seeds.extend(ts.into_iter().map(|t| [s, t]));
    }
    for &t in &ends_b {
        let (point, hold) = end(b, t, rb);
        let ss = minima_from_point(point, hold, &pieces[0], tol)?;
        seeds.extend(ss.into_iter().map(|s| [s, t]));
    }
    // A seed on an end the distance falls away from is no minimum there.
    seeds.retain(|x| {
        let ((pa, a1), (pb, b1)) = (a.eval_d1(x[0]), b.eval_d1(x[1]));
        let r = pa - pb;
        let (g, speed) = ([r.dot(a1), -r.dot(b1)], [a1.magnitude(), b1.magnitude()]);
        [ra, rb].into_iter().enumerate().all(|(i, r)| {
            let held =
                !r.periodic && ((x[i] <= r.lo && g[i] > 0.0) || (x[i] >= r.hi && g[i] < 0.0));
            held || g[i].abs() <= tol.confusion * speed[i]
        })
    });
    Some(seeds)
}

/// The parameters of a curve, cut into `pieces`, at the stationary points
/// of the distance from `point` with a positive definite second
/// derivative (the minima inside the pieces), found as in
/// [`bspline_minima`] along the curve alone, where `(point - C) . hold`
/// is not below `-tol.confusion |hold|` (the pieces beyond are dropped
/// at once); `None` when the search cannot isolate them within its
/// budget.
fn minima_from_point(
    point: [f64; 3],
    hold: [f64; 3],
    pieces: &CurvePieces,
    tol: Tolerance,
) -> Option<Vec<f64>> {
    let reach = dot(point, hold) + tol.confusion * norm(hold);
    let mut pairs = Pairs::new([1, pieces.width], pieces.len());
    let fixed = [point[0], point[1], point[2], 1.0];
    for j in 0..pieces.len() {
        pairs.push([&fixed, pieces.piece(j)], [(0.0, 0.0), pieces.ranges[j]]);
    }
    let mut found = Vec::new();
    pairs
        .search(|pairs, node| {
            let [fixed, pc] = &pairs.points;
            if extent(pc, hold).0 > reach || tangent_sign(pc, fixed).is_some() {
                return true;
            }
            let (value, d) = curve_bounds(pairs.net(1, node), pc);
            let mut second = Interval::ZERO;
            for c in 0..3 {
                let at = Interval {
                    lo: point[c],
                    hi: point[c],
                };
                second = second
                    .add(d[0][c].square())
                    .add(value[c].sub(at).mul(d[1][c]));
            }
            if second.hi < 0.0 {
                return true;
            }
            if second.lo <= 0.0 {
                return false;
            }
            let net = pairs.net(1, node);
            let local = |u: f64| {
                let [c, d1, d2] = bezier_d2(net, u);
                let r = sub(c, point);
                (
                    (0.5 * dot(r, r), dot(r, d1), dot(d1, d1) + dot(r, d2)),
                    norm(d1),
                )
            };
            let u = newton_min_1d(|u| local(u).0, 0.5, Some((0.0, 1.0)), 1e-14, 100);
            let ((_, g, _), speed) = local(u);
            if (0.0 < u && u < 1.0) || g.abs() <= tol.confusion * speed {
                let (lo, hi) = node.ranges[1];
                found.push(lo + u * (hi - lo));
            }
            true
        })
        .ok()?;
    Some(found)
}

/// The point and its first two derivatives at `u` in `[0, 1]` of the
/// Bézier piece with the homogeneous control points `net`, in its own
/// parameter: de Casteljau's algorithm down to the three points whose
/// differences give the homogeneous derivatives, then the quotient rule.
fn bezier_d2(net: &[f64], u: f64) -> [[f64; 3]; 3] {
    let n = net.len() / 4;
    let p = (n - 1) as f64;
    let mut q = [[0.0; 4]; LARGE];
    for (k, x) in net.chunks_exact(4).enumerate() {
        q[k] = [x[0], x[1], x[2], x[3]];
    }
    let lerp = |a: [f64; 4], b: [f64; 4]| std::array::from_fn(|c| a[c] + u * (b[c] - a[c]));
    let kept = n.min(3);
    for r in 0..n - kept {
        for i in 0..n - 1 - r {
            q[i] = lerp(q[i], q[i + 1]);
        }
    }
    let (h, h1, h2): ([f64; 4], [f64; 4], [f64; 4]) = if kept == 3 {
        let (r0, r1) = (lerp(q[0], q[1]), lerp(q[1], q[2]));
        (
            lerp(r0, r1),
            std::array::from_fn(|c| p * (r1[c] - r0[c])),
            std::array::from_fn(|c| p * (p - 1.0) * (q[2][c] - 2.0 * q[1][c] + q[0][c])),
        )
    } else {
        (
            lerp(q[0], q[1]),
            std::array::from_fn(|c| q[1][c] - q[0][c]),
            [0.0; 4],
        )
    };
    let w = h[3];
    let point: [f64; 3] = std::array::from_fn(|c| h[c] / w);
    let first: [f64; 3] = std::array::from_fn(|c| (h1[c] - h1[3] * point[c]) / w);
    let second = std::array::from_fn(|c| (h2[c] - 2.0 * h1[3] * first[c] - h2[3] * point[c]) / w);
    [point, first, second]
}

/// Enclosures, over the Bézier piece with the homogeneous control points
/// `net` and the Euclidean ones `points`, in its own parameter over `[0,
/// 1]`, of each coordinate of its point and of its first and second
/// derivatives.
///
/// A polynomial piece's derivatives lie in the hulls of its hodographs'
/// control points, scaled differences of its own; a rational one's are
/// those of [`derivative_bounds`].
fn curve_bounds(net: &[f64], points: &[[f64; 3]]) -> ([Interval; 3], [[Interval; 3]; 2]) {
    if net.chunks_exact(4).any(|x| x[3] != 1.0) {
        let (value, d) = derivative_bounds(net, points, points.len(), 1, 4, true);
        return (value, [d[0], d[2]]);
    }
    let n = points.len();
    let p = (n - 1) as f64;
    let empty = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    let mut value = [empty; 3];
    let mut first = [empty; 3];
    let mut second = [empty; 3];
    let widen = |i: &mut Interval, x: f64| {
        i.lo = i.lo.min(x);
        i.hi = i.hi.max(x);
    };
    for k in 0..n {
        for c in 0..3 {
            widen(&mut value[c], points[k][c]);
            if k + 1 < n {
                widen(&mut first[c], p * (points[k + 1][c] - points[k][c]));
            }
            if k + 2 < n {
                let bend = points[k + 2][c] - 2.0 * points[k + 1][c] + points[k][c];
                widen(&mut second[c], p * (p - 1.0) * bend);
            }
        }
    }
    let settle = |i: Interval| if i.lo > i.hi { Interval::ZERO } else { i };
    (value, [first.map(settle), second.map(settle)])
}

/// A pair of subpieces awaiting its visit: one per curve, in slot
/// `slots[k]` of pool `k` over the parameters `ranges[k]`, and the
/// halvings from the pair it came from.
struct Node {
    slots: [usize; 2],
    ranges: [(f64, f64); 2],
    depth: u32,
}

/// The search ran out of budget or depth.
struct Exhausted;

/// The pending pairs of a search over the pieces of two curves (or of a
/// point, a piece of one control point, and a curve).
struct Pairs {
    shapes: [(usize, usize, usize); 2],
    pools: [Pool; 2],
    pending: Vec<Node>,
    /// The Euclidean control points of the pair being visited.
    points: [Vec<[f64; 3]>; 2],
    budget: usize,
}

impl Pairs {
    /// No pairs yet, of pieces of `widths` homogeneous control points,
    /// with room for `count` pairs and some halvings of them.
    fn new(widths: [usize; 2], count: usize) -> Self {
        let room = 4 * count + 16;
        Pairs {
            shapes: widths.map(|w| (w, 1, 4)),
            pools: widths.map(|w| {
                let mut pool = Pool::new(4 * w);
                pool.reserve(room);
                pool
            }),
            pending: Vec::with_capacity(room),
            points: widths.map(Vec::with_capacity),
            budget: 0,
        }
    }

    /// Queues the pair of pieces with the homogeneous control points
    /// `nets` over `ranges`.
    fn push(&mut self, nets: [&[f64]; 2], ranges: [(f64, f64); 2]) {
        let slots = [0, 1].map(|k| {
            let slot = self.pools[k].take();
            self.pools[k].get_mut(slot).copy_from_slice(nets[k]);
            slot
        });
        self.pending.push(Node {
            slots,
            ranges,
            depth: 0,
        });
    }

    /// The homogeneous control points of `node`'s piece of curve `k`.
    fn net(&self, k: usize, node: &Node) -> &[f64] {
        self.pools[k].get(node.slots[k])
    }

    /// Visits every pending pair, with its control points loaded into
    /// [`Pairs::points`]: `visit` settles it (returns `true`) or the
    /// larger of its pieces is halved and both halves queued.
    fn search(&mut self, mut visit: impl FnMut(&Self, &Node) -> bool) -> Result<(), Exhausted> {
        self.budget = SPLITS_PER_PAIR * self.pending.len();
        while let Some(node) = self.pending.pop() {
            for k in 0..2 {
                euclidean(self.pools[k].get(node.slots[k]), &mut self.points[k]);
            }
            if visit(self, &node) {
                for k in 0..2 {
                    self.pools[k].release(node.slots[k]);
                }
                continue;
            }
            if node.depth >= MAX_DEPTH || self.budget == 0 {
                return Err(Exhausted);
            }
            self.budget -= 1;
            let sizes = self.points.each_ref().map(|p| box_size(&bounding_box(p)));
            self.split(node, usize::from(sizes[1] >= sizes[0]));
        }
        Ok(())
    }

    /// Halves the piece of curve `k` of `node` and queues both halves,
    /// each with a copy of the other piece.
    fn split(&mut self, node: Node, k: usize) {
        let other = 1 - k;
        let upper = self.pools[k].halve(node.slots[k], self.shapes[k], true);
        let copy = self.pools[other].take();
        self.pools[other].copy(node.slots[other], copy);
        let (lo, hi) = node.ranges[k];
        let mid = 0.5 * (lo + hi);
        let depth = node.depth + 1;
        let mut slots = node.slots;
        let mut ranges = node.ranges;
        slots[k] = upper;
        slots[other] = copy;
        ranges[k] = (mid, hi);
        self.pending.push(Node {
            slots,
            ranges,
            depth,
        });
        ranges[k] = (lo, mid);
        self.pending.push(Node {
            ranges,
            depth,
            ..node
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point3D;
    use crate::curves::{Curve3D, ParametricCurve3D, TrimmedCurve3D};
    use crate::intersect::curve_curve::{
        ParamDomain, curve_curve_minima, hits_at_minima, refine_minima, sampled_curve_curve,
    };
    use crate::intersect::{CurveCurveHit, curve_curve_extrema, intersect_curve_curve};

    const TOL: Tolerance = Tolerance::DEFAULT;

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn wave(&mut self) -> f64 {
            2.0 * self.next() - 1.0
        }
    }

    /// A cubic through six poles from `from(0)` to `from(1)`, offset at
    /// random by up to `amplitude`; rational with random weights if
    /// `rational`.
    fn cubic(
        rng: &mut Lcg,
        from: impl Fn(f64) -> [f64; 3],
        amplitude: [f64; 3],
        rational: bool,
    ) -> BSplineCurve3D {
        let poles = (0..6)
            .map(|k| {
                let [x, y, z] = from(k as f64 / 5.0);
                Point3D::new(
                    x + amplitude[0] * rng.wave(),
                    y + amplitude[1] * rng.wave(),
                    z + amplitude[2] * rng.wave(),
                )
            })
            .collect();
        let (knots, mults) = (vec![0.0, 1.0, 2.0, 3.0], vec![4, 1, 1, 4]);
        if rational {
            let weights = (0..6).map(|_| 0.4 + 1.6 * rng.next()).collect();
            BSplineCurve3D::new_rational(3, poles, weights, knots, mults, false).unwrap()
        } else {
            BSplineCurve3D::new(3, poles, knots, mults, false).unwrap()
        }
    }

    /// Two cubics in the plane `z = x / 2`, across and up the square `[0,
    /// 5]^2` with waves of `wave`, so that they cross.
    fn crossing(rng: &mut Lcg, rational: bool, wave: f64) -> (BSplineCurve3D, BSplineCurve3D) {
        let tilt = |[x, y, _]: [f64; 3]| [x, y, 0.5 * x];
        let (y0, y1, x0, x1) = (
            5.0 * rng.next(),
            5.0 * rng.next(),
            5.0 * rng.next(),
            5.0 * rng.next(),
        );
        let across = cubic(
            rng,
            |s| [-0.5 + 6.0 * s, y0 + (y1 - y0) * s, 0.0],
            [0.0, wave, 0.0],
            rational,
        );
        let up = cubic(
            rng,
            |s| [x0 + (x1 - x0) * s, -0.5 + 6.0 * s, 0.0],
            [wave, 0.0, 0.0],
            rational,
        );
        let tilted = |c: &BSplineCurve3D| {
            let poles = c.poles().iter().map(|p| {
                let [x, y, z] = tilt([p.x, p.y, p.z]);
                Point3D::new(x, y, z)
            });
            let (knots, mults) = (c.knots().to_vec(), c.multiplicities().to_vec());
            match c.weights() {
                Some(w) => BSplineCurve3D::new_rational(
                    3,
                    poles.collect(),
                    w.to_vec(),
                    knots,
                    mults,
                    false,
                ),
                None => BSplineCurve3D::new(3, poles.collect(), knots, mults, false),
            }
            .unwrap()
        };
        (tilted(&across), tilted(&up))
    }

    /// Two wavy cubics, one about `z = 0` and one about `z = 2`.
    fn skew(rng: &mut Lcg, rational: bool) -> (BSplineCurve3D, BSplineCurve3D) {
        let (y0, y1, x0, x1) = (
            5.0 * rng.next(),
            5.0 * rng.next(),
            5.0 * rng.next(),
            5.0 * rng.next(),
        );
        let below = cubic(
            rng,
            |s| [6.0 * s, y0 + (y1 - y0) * s, 0.0],
            [0.5, 1.0, 0.5],
            rational,
        );
        let above = cubic(
            rng,
            |s| [x0 + (x1 - x0) * s, 6.0 * s, 2.0],
            [1.0, 0.5, 0.5],
            rational,
        );
        (below, above)
    }

    /// The hits of the piece-pair search, which must not fall back.
    fn search<A: ParametricCurve3D, B: ParametricCurve3D>(a: &A, b: &B) -> Vec<CurveCurveHit> {
        let (ra, rb) = (ParamDomain::of(a).range(), ParamDomain::of(b).range());
        let (sa, sb) = (a.as_bspline().unwrap(), b.as_bspline().unwrap());
        let meetings =
            bspline_curve_curve(sa, ra, sb, rb, TOL).expect("the search isolates every hit");
        hits_at_minima(a, b, meetings, TOL)
    }

    /// The minima through the piece-pair seeds, which must not fall back.
    fn minima<A: ParametricCurve3D, B: ParametricCurve3D>(a: &A, b: &B) -> Vec<CurveCurveExtremum> {
        let (da, db) = (ParamDomain::of(a), ParamDomain::of(b));
        let (sa, sb) = (a.as_bspline().unwrap(), b.as_bspline().unwrap());
        let seeds = bspline_minima(sa, da.range(), sb, db.range(), TOL)
            .expect("the search isolates every minimum");
        let mut out = refine_minima(a, da, b, db, seeds, TOL);
        out.sort_by(|x, y| x.distance.total_cmp(&y.distance));
        out
    }

    /// Asserts the search finds the sampled search's hits, with the same
    /// parameters and multiplicities, that every hit lies on both curves
    /// within their ranges (the search may find crossings the samples
    /// miss), and returns them.
    #[track_caller]
    fn assert_hits_match<A: ParametricCurve3D, B: ParametricCurve3D>(
        a: &A,
        b: &B,
    ) -> Vec<CurveCurveHit> {
        let found = search(a, b);
        let ((a0, a1), (b0, b1)) = (a.bounds(), b.bounds());
        for hit in &found {
            assert!((a0..=a1).contains(&hit.first_param), "{hit:?}");
            assert!((b0..=b1).contains(&hit.second_param), "{hit:?}");
            assert!(hit.point.distance(a.eval_point(hit.first_param)) < 1e-12);
            assert!(hit.point.distance(b.eval_point(hit.second_param)) <= TOL.confusion);
        }
        for pair in found.windows(2) {
            assert!(pair[0].first_param < pair[1].first_param);
        }
        let sampled = sampled_curve_curve(a, b, TOL);
        for want in &sampled {
            assert!(
                found.iter().any(|hit| hit.point.distance(want.point) < 1e-9
                    && (hit.first_param - want.first_param).abs() < 1e-9
                    && (hit.second_param - want.second_param).abs() < 1e-9
                    && hit.multiplicity == want.multiplicity),
                "{want:?} missing from {found:#?}"
            );
        }
        assert_eq!(intersect_curve_curve(a, b, TOL), found);
        found
    }

    /// Asserts the search finds the grid's minima, that every minimum it
    /// reports is a local minimum on the parameter box (it may find
    /// minima the grid misses), and returns them.
    #[track_caller]
    fn assert_minima_match<A: ParametricCurve3D, B: ParametricCurve3D>(
        a: &A,
        b: &B,
    ) -> Vec<CurveCurveExtremum> {
        let found = minima(a, b);
        let grid = curve_curve_minima(a, b, TOL, false);
        for want in &grid {
            assert!(
                found
                    .iter()
                    .any(|e| (e.first_param - want.first_param).abs() < 1e-7
                        && (e.second_param - want.second_param).abs() < 1e-7
                        && (e.distance - want.distance).abs() < 1e-12),
                "{want:?} missing from {found:#?}"
            );
        }
        let (da, db) = (ParamDomain::of(a), ParamDomain::of(b));
        let ((a0, a1), (b0, b1)) = (a.bounds(), b.bounds());
        for e in &found {
            for (ds, dt) in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)] {
                for h in [-1e-4, 1e-4] {
                    let (s, t) = (e.first_param + h * ds, e.second_param + h * dt);
                    let s = if da.is_periodic() { s } else { s.clamp(a0, a1) };
                    let t = if db.is_periodic() { t } else { t.clamp(b0, b1) };
                    let d = a.eval_point(s).distance(b.eval_point(t));
                    assert!(
                        d >= e.distance - 1e-12,
                        "{e:?} is no minimum: {d} at ({s}, {t})"
                    );
                }
            }
        }
        assert_eq!(curve_curve_extrema(a, b, TOL), found);
        found
    }

    #[test]
    fn test_crossing_polynomial_curves_match_sampled() {
        let mut rng = Lcg(42);
        let mut count = 0;
        for _ in 0..32 {
            let (a, b) = crossing(&mut rng, false, 1.0);
            count += assert_hits_match(&a, &b).len();
            assert_minima_match(&a, &b);
        }
        assert!(count >= 32, "{count}");
    }

    #[test]
    fn test_crossing_rational_curves_match_sampled() {
        let mut rng = Lcg(7);
        let mut count = 0;
        for _ in 0..24 {
            let (a, b) = crossing(&mut rng, true, 1.0);
            count += assert_hits_match(&a, &b).len();
            assert_minima_match(&a, &b);
        }
        assert!(count >= 24, "{count}");
    }

    #[test]
    fn test_skew_curves_minima_match_grid() {
        let mut rng = Lcg(3);
        for k in 0..48 {
            let (a, b) = skew(&mut rng, k % 3 == 2);
            assert!(assert_hits_match(&a, &b).is_empty());
            assert!(!assert_minima_match(&a, &b).is_empty());
        }
    }

    #[test]
    fn test_curves_crossing_many_times_match_sampled() {
        let mut rng = Lcg(11);
        let mut most = 0;
        for k in 0..24 {
            let (a, b) = crossing(&mut rng, k % 2 == 1, 3.0);
            most = most.max(assert_hits_match(&a, &b).len());
            assert_minima_match(&a, &b);
        }
        assert!(most >= 3, "{most}");
    }

    /// A closed cubic loop of radius `radius` about `centre`, its poles
    /// alternately `lift` above and below.
    fn ring(centre: [f64; 3], radius: f64, lift: f64) -> BSplineCurve3D {
        let poles = (0..8)
            .map(|k| {
                let angle = k as f64 * std::f64::consts::TAU / 8.0;
                let z = if k % 2 == 0 { lift } else { -lift };
                Point3D::new(
                    centre[0] + radius * angle.cos(),
                    centre[1] + radius * angle.sin(),
                    centre[2] + z,
                )
            })
            .collect();
        let knots = (0..=8).map(f64::from).collect();
        BSplineCurve3D::new(3, poles, knots, vec![1; 9], true).unwrap()
    }

    #[test]
    fn test_periodic_curves_match_sampled() {
        let loop_a = ring([2.5, 2.5, 0.0], 2.0, 0.0);
        let loop_b = ring([3.5, 2.0, 0.0], 1.5, 0.0);
        let hits = assert_hits_match(&loop_a, &loop_b);
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|h| (0.0..8.0).contains(&h.first_param)));
        assert!(hits.iter().all(|h| (0.0..8.0).contains(&h.second_param)));
        assert_minima_match(&loop_a, &loop_b);
        let mut rng = Lcg(5);
        for _ in 0..8 {
            let (across, _) = crossing(&mut rng, false, 1.0);
            let flat = |c: &BSplineCurve3D| {
                let poles = c
                    .poles()
                    .iter()
                    .map(|p| Point3D::new(p.x, p.y, 0.0))
                    .collect();
                BSplineCurve3D::new(
                    3,
                    poles,
                    c.knots().to_vec(),
                    c.multiplicities().to_vec(),
                    false,
                )
                .unwrap()
            };
            let across = flat(&across);
            assert_hits_match(&loop_a, &across);
            assert_hits_match(&across, &loop_a);
            assert_minima_match(&across, &loop_a);
        }
        // A wobbling loop under a skew cubic: distance minima only.
        let wobbly = ring([2.5, 2.5, 0.0], 2.0, 0.5);
        let (_, above) = skew(&mut rng, false);
        assert!(assert_hits_match(&wobbly, &above).is_empty());
        assert!(!assert_minima_match(&wobbly, &above).is_empty());
        assert_minima_match(&above, &wobbly);
    }

    #[test]
    fn test_trimmed_curves_search_only_their_range() {
        let ring = ring([2.5, 2.5, 0.0], 2.0, 0.0);
        let wavy = BSplineCurve3D::new(
            3,
            (0..8)
                .map(|k| {
                    Point3D::new(
                        -0.5 + 0.85 * k as f64,
                        if k % 2 == 0 { -4.0 } else { 9.0 },
                        0.0,
                    )
                })
                .collect(),
            (0..=5).map(f64::from).collect(),
            vec![4, 1, 1, 1, 1, 4],
            false,
        )
        .unwrap();
        let all = assert_hits_match(&ring, &wavy);
        assert!(all.len() >= 6, "{all:#?}");
        // Across the seam of the ring, and inside the wavy curve.
        let wrapped = |t: f64| if t < 6.5 { t + 8.0 } else { t };
        let expected: Vec<f64> = all
            .iter()
            .map(|h| wrapped(h.first_param))
            .filter(|&t| t <= 9.5)
            .collect();
        assert!(expected.iter().any(|&t| t < 8.0) && expected.iter().any(|&t| t > 8.0));
        let borrowed = TrimmedCurve3D::new(&ring, 6.5, 9.5).unwrap();
        let owned = TrimmedCurve3D::new(ring.clone(), 6.5, 9.5).unwrap();
        for hits in [
            assert_hits_match(&borrowed, &wavy),
            assert_hits_match(&owned, &wavy),
        ] {
            assert_eq!(hits.len(), expected.len());
            for hit in &hits {
                assert!(expected.iter().any(|&t| (t - hit.first_param).abs() < 1e-9));
            }
        }
        let (first, last) = (all[1].second_param.min(all[2].second_param), 4.2);
        let middle = TrimmedCurve3D::new(&wavy, first, last).unwrap();
        let hits = assert_hits_match(&ring, &middle);
        let inside = all
            .iter()
            .filter(|h| (first - 1e-9..=last + 1e-9).contains(&h.second_param))
            .count();
        assert_eq!(hits.len(), inside);
        assert_minima_match(&borrowed, &wavy);
        assert_minima_match(&owned, &middle);
        let mut rng = Lcg(9);
        let (below, above) = skew(&mut rng, true);
        let piece = TrimmedCurve3D::new(&below, 0.4, 2.3).unwrap();
        assert_minima_match(&piece, &TrimmedCurve3D::new(above, 1.2, 2.9).unwrap());
    }

    fn quadratic(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> BSplineCurve3D {
        let poles = [p0, p1, p2].map(|[x, y, z]| Point3D::new(x, y, z)).to_vec();
        BSplineCurve3D::new(2, poles, vec![0.0, 1.0], vec![3, 3], false).unwrap()
    }

    fn segment(from: [f64; 3], to: [f64; 3]) -> BSplineCurve3D {
        let poles = [from, to].map(|[x, y, z]| Point3D::new(x, y, z)).to_vec();
        BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap()
    }

    /// Searches without falling back and checks the hits lie on both.
    fn assert_search_hits(a: &BSplineCurve3D, b: &BSplineCurve3D) -> Vec<CurveCurveHit> {
        let hits = search(a, b);
        for hit in &hits {
            assert!((a.bounds().0..=a.bounds().1).contains(&hit.first_param));
            assert!((b.bounds().0..=b.bounds().1).contains(&hit.second_param));
            assert!(hit.point.distance(b.eval_point(hit.second_param)) <= TOL.confusion);
        }
        assert_eq!(intersect_curve_curve(a, b, TOL), hits);
        hits
    }

    #[test]
    fn test_close_hits_near_a_tangency_are_separated() {
        let line = segment([0.0, 0.0, 0.0], [4.0, 0.0, 0.0]);
        for depth in [1e-3, 1e-6] {
            // `y = (x - 2)^2 / 4 - depth` on `[0, 4]`.
            let dip = quadratic(
                [0.0, 1.0 - depth, 0.0],
                [2.0, -1.0 - depth, 0.0],
                [4.0, 1.0 - depth, 0.0],
            );
            let hits = assert_search_hits(&dip, &line);
            assert_eq!(hits.len(), 2, "{hits:#?}");
            let half = (4.0 * depth).sqrt();
            assert!((hits[0].point.x - (2.0 - half)).abs() < 1e-9, "{hits:#?}");
            assert!((hits[1].point.x - (2.0 + half)).abs() < 1e-9, "{hits:#?}");
            assert!(hits.iter().all(|h| h.multiplicity == 1));
        }
    }

    #[test]
    fn test_tangency_and_overlap_fall_back() {
        let line = segment([0.0, 0.0, 0.0], [4.0, 0.0, 0.0]);
        let touching = quadratic([0.0, 1.0, 0.0], [2.0, -1.0, 0.0], [4.0, 1.0, 0.0]);
        let r = |c: &BSplineCurve3D| ParamDomain::of(c).range();
        assert!(bspline_curve_curve(&touching, r(&touching), &line, r(&line), TOL).is_none());
        let hits = intersect_curve_curve(&touching, &line, TOL);
        assert_eq!(hits, sampled_curve_curve(&touching, &line, TOL));
        assert_eq!(hits.len(), 1, "{hits:#?}");
        assert!(hits[0].point.distance(Point3D::new(2.0, 0.0, 0.0)) < 1e-3);
        let mut rng = Lcg(1);
        let (curve, _) = crossing(&mut rng, false, 1.0);
        let part = TrimmedCurve3D::new(&curve, 0.5, 2.0).unwrap();
        let ranges = (r(&curve), ParamDomain::of(&part).range());
        assert!(bspline_curve_curve(&curve, ranges.0, &curve, ranges.1, TOL).is_none());
        assert!(bspline_minima(&curve, ranges.0, &curve, ranges.1, TOL).is_none());
        let hits = intersect_curve_curve(&curve, &part, TOL);
        assert!(!hits.is_empty());
        assert_eq!(hits, sampled_curve_curve(&curve, &part, TOL));
        assert_eq!(
            curve_curve_extrema(&curve, &part, TOL),
            curve_curve_minima(&curve, &part, TOL, false)
        );
    }

    #[test]
    fn test_hits_at_curve_ends_and_knots() {
        let bend = quadratic([0.0, 0.0, 0.0], [1.0, 2.0, 0.5], [2.0, 0.0, 1.0]);
        let end = bend.eval_point(1.0);
        let middle = bend.eval_point(0.3);
        let cases = [
            // End to end, end to start, end on the interior.
            (bend.eval_point(1.0), [3.0, 1.0, 0.0], end),
            (
                Point3D::new(-1.0, 1.0, 0.5),
                [0.0, 0.0, 0.0],
                Point3D::ORIGIN,
            ),
            (
                Point3D::new(middle.x, middle.y - 1.0, middle.z),
                [middle.x, middle.y + 1.0, middle.z],
                middle,
            ),
        ]
        .map(|(from, to, at): (Point3D, [f64; 3], Point3D)| {
            (segment([from.x, from.y, from.z], to), at)
        });
        for (line, at) in &cases {
            for hits in [
                assert_search_hits(&bend, line),
                assert_search_hits(line, &bend),
            ] {
                assert_eq!(hits.len(), 1, "{hits:#?}");
                assert!(hits[0].point.distance(*at) < 1e-12, "{hits:#?}");
            }
        }
        // Through a knot of each curve.
        let knotted = |p: [[f64; 3]; 3]| {
            let poles = p.map(|[x, y, z]| Point3D::new(x, y, z)).to_vec();
            BSplineCurve3D::new(1, poles, vec![0.0, 1.0, 2.0], vec![2, 1, 2], false).unwrap()
        };
        let a = knotted([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.5, 0.0]]);
        let b = knotted([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 3.0, 0.0]]);
        let hits = assert_search_hits(&a, &b);
        assert_eq!(hits.len(), 1, "{hits:#?}");
        assert_eq!((hits[0].first_param, hits[0].second_param), (1.0, 1.0));
    }

    #[test]
    fn test_enums_and_borrows_take_the_search() {
        let mut rng = Lcg(21);
        let (a, b) = crossing(&mut rng, true, 2.0);
        let (ea, eb) = (Curve3D::BSpline(a.clone()), Curve3D::BSpline(b.clone()));
        let hits = assert_hits_match(&ea, &eb);
        assert_eq!(hits, intersect_curve_curve(&a, &b, TOL));
        assert_eq!(hits, intersect_curve_curve(&&a, &b, TOL));
        assert_eq!(
            curve_curve_extrema(&ea, &eb, TOL),
            assert_minima_match(&a, &b)
        );
    }

    #[test]
    fn test_concentric_arcs_fall_back() {
        // Exact rational quarter circles of radii 2 and 3: a valley.
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let arc = |r: f64| {
            let poles = vec![
                Point3D::new(r, 0.0, 0.0),
                Point3D::new(r, r, 0.0),
                Point3D::new(0.0, r, 0.0),
            ];
            BSplineCurve3D::new_rational(
                2,
                poles,
                vec![1.0, w, 1.0],
                vec![0.0, 1.0],
                vec![3, 3],
                false,
            )
            .unwrap()
        };
        let (a, b) = (arc(2.0), arc(3.0));
        let r = |c: &BSplineCurve3D| ParamDomain::of(c).range();
        assert!(bspline_minima(&a, r(&a), &b, r(&b), TOL).is_none());
        let ext = curve_curve_extrema(&a, &b, TOL);
        assert_eq!(ext, curve_curve_minima(&a, &b, TOL, false));
        assert_eq!(ext.len(), 1, "{ext:#?}");
        assert!((ext[0].distance - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_minima_on_ends_and_corners_match_grid() {
        // A bow beside a segment, the bow's ends closest: two corners.
        let bow = quadratic([0.0, 0.0, 0.0], [1.0, -1.0, 0.3], [2.0, 0.0, 0.0]);
        let rail = segment([-0.5, 1.0, 0.0], [2.5, 1.2, 0.0]);
        let ext = assert_minima_match(&bow, &rail);
        assert!(
            ext.iter()
                .any(|e| e.first_param == 0.0 || e.first_param == 1.0),
            "{ext:#?}"
        );
        // An end of one closest to the interior of the other.
        let stub = segment([1.0, 2.0, 0.5], [1.2, 4.0, 0.5]);
        let ext = assert_minima_match(&bow, &stub);
        assert!(ext.iter().any(|e| e.second_param == 0.0), "{ext:#?}");
        // A closed loop: no ends of its own.
        let loop_a = ring([0.0, 0.0, 0.0], 2.0, 0.3);
        let ext = assert_minima_match(&stub, &loop_a);
        assert!(ext.iter().any(|e| e.first_param == 0.0), "{ext:#?}");
        assert!(ext.iter().all(|e| (0.0..8.0).contains(&e.second_param)));
    }
}
