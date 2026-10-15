//! The result of a numeric curve–surface intersection
//! ([`CurveSurfaceHits`]), and the intersection of a B-spline curve with a
//! B-spline surface by subdivision of their Bézier pieces (see
//! [`bspline_curve_surface`]).

use super::completeness::{Region, both, box_union, meet, of_unsettled, span};
use super::pieces::{
    CurvePieces, Pool, apart_along, bounding_box, box_size, dot, euclidean, meets_at_most_once,
    net_normal, norm, sub,
};
use super::results::sort_by_known_parameter;
use super::{Completeness, CurveSurfaceHit, curve_surface_multiplicity};
use crate::curves::BSplineCurve3D;
use crate::math::boxtree::{Aabb, boxes_apart};
use crate::math::solve_3x3;
use crate::surface_math::bspline::net_points;
use crate::surfaces::{BSplineSurface, ParametricSurface};
use crate::{Point3D, Tolerance};
use std::ops::Deref;

/// The hits of a numeric curve–surface intersection, ordered by curve
/// parameter, and whether they are all of them (see
/// [`search_curve_surface`](crate::search_curve_surface)).
///
/// It reads as the slice of its hits.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CurveSurfaceHits {
    pub(super) hits: Vec<CurveSurfaceHit>,
    pub(super) completeness: Completeness<CurveSurfaceRegion>,
}

impl CurveSurfaceHits {
    /// Hits found from samples.
    pub(super) fn sampled(hits: Vec<CurveSurfaceHit>) -> Self {
        CurveSurfaceHits {
            hits,
            completeness: Completeness::Sampled,
        }
    }

    /// Hits proven to be all of them.
    pub(super) fn complete(hits: Vec<CurveSurfaceHit>) -> Self {
        CurveSurfaceHits {
            hits,
            completeness: Completeness::Complete,
        }
    }

    /// The hits of two searches of one curve against two parts of a
    /// surface: those of `other` farther than `sqrt(tol.confusion)` from
    /// every hit of `self` join them, and so do the regions of both.
    pub(super) fn join(self, other: CurveSurfaceHits, tol: Tolerance) -> Self {
        let mut joined = CurveSurfaceHits {
            hits: self.hits,
            completeness: both(self.completeness, other.completeness),
        };
        joined.add_sampled(other.hits, tol);
        joined
    }

    /// Adds the `sampled` hits farther than `sqrt(tol.confusion)` from
    /// every hit already found, keeping the order by curve parameter.
    pub(super) fn add_sampled(&mut self, sampled: Vec<CurveSurfaceHit>, tol: Tolerance) {
        let merge_gap = tol.confusion.sqrt();
        for hit in sampled {
            if self
                .hits
                .iter()
                .all(|h| h.point.distance(hit.point) > merge_gap)
            {
                self.hits.push(hit);
            }
        }
        sort_by_known_parameter(&mut self.hits, |hit| hit.curve_parameter);
    }

    /// The hits, ordered by curve parameter, without the completeness.
    pub fn into_hits(self) -> Vec<CurveSurfaceHit> {
        self.hits
    }

    /// Whether the hits are all of them, and if not, why and where hits
    /// may be missing.
    pub fn completeness(&self) -> &Completeness<CurveSurfaceRegion> {
        &self.completeness
    }

    /// Whether the hits are all of them ([`Completeness::Complete`]).
    pub fn is_complete(&self) -> bool {
        self.completeness.is_complete()
    }
}

impl Deref for CurveSurfaceHits {
    type Target = [CurveSurfaceHit];

    fn deref(&self) -> &[CurveSurfaceHit] {
        &self.hits
    }
}

impl IntoIterator for CurveSurfaceHits {
    type Item = CurveSurfaceHit;
    type IntoIter = std::vec::IntoIter<CurveSurfaceHit>;

    fn into_iter(self) -> std::vec::IntoIter<CurveSurfaceHit> {
        self.hits.into_iter()
    }
}

impl<'a> IntoIterator for &'a CurveSurfaceHits {
    type Item = &'a CurveSurfaceHit;
    type IntoIter = std::slice::Iter<'a, CurveSurfaceHit>;

    fn into_iter(self) -> std::slice::Iter<'a, CurveSurfaceHit> {
        self.hits.iter()
    }
}

/// Where an incomplete curve–surface intersection may have missed hits:
/// a curve parameter interval against a surface parameter rectangle,
/// with a box in space holding the curve over its interval. A missed hit
/// has its curve parameter in `curve_range`, its surface parameters in
/// `u_range` and `v_range`, and its point in the box.
///
/// # Examples
///
/// ```
/// use geomcore::{
///     BSplineCurve3D, BSplineSurface, Completeness, Intersect, Point3D, Tolerance,
/// };
/// let poles = vec![
///     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 2.0, 0.0)],
///     vec![Point3D::new(2.0, 0.0, 0.0), Point3D::new(2.0, 2.0, 0.0)],
/// ];
/// let (knots, mults) = (vec![0.0, 1.0], vec![2, 2]);
/// let square = BSplineSurface::new(
///     1, 1, poles, knots.clone(), mults.clone(), knots.clone(), mults, false, false,
/// )
/// .unwrap();
/// // A parabola touching the square at (1, 1, 0), at t = 0.5.
/// let poles = vec![
///     Point3D::new(0.5, 1.0, 1.0),
///     Point3D::new(1.0, 1.0, -1.0),
///     Point3D::new(1.5, 1.0, 1.0),
/// ];
/// let parabola = BSplineCurve3D::new(2, poles, knots, vec![3, 3], false).unwrap();
/// let result = parabola.intersect(&square, Tolerance::DEFAULT);
/// let Completeness::Incomplete { regions, .. } = result.completeness() else {
///     panic!("a tangent contact never separates");
/// };
/// assert_eq!(regions.len(), 1);
/// let (t0, t1) = regions[0].curve_range;
/// assert!(t0 <= 0.5 && 0.5 <= t1 && t1 - t0 < 1e-2);
/// assert!(regions[0].min_corner.x <= 1.0 && 1.0 <= regions[0].max_corner.x);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveSurfaceRegion {
    /// The curve parameter interval.
    pub curve_range: (f64, f64),
    /// The surface's `u` parameter interval.
    pub u_range: (f64, f64),
    /// The surface's `v` parameter interval.
    pub v_range: (f64, f64),
    /// The corner of the box with the smallest coordinates.
    pub min_corner: Point3D,
    /// The corner of the box with the largest coordinates.
    pub max_corner: Point3D,
}

impl Region for CurveSurfaceRegion {
    fn touches(&self, other: &CurveSurfaceRegion) -> bool {
        meet(self.curve_range, other.curve_range)
            && meet(self.u_range, other.u_range)
            && meet(self.v_range, other.v_range)
    }

    fn union(&self, other: &CurveSurfaceRegion) -> CurveSurfaceRegion {
        let (min_corner, max_corner) = box_union(
            (self.min_corner, self.max_corner),
            (other.min_corner, other.max_corner),
        );
        CurveSurfaceRegion {
            curve_range: span(self.curve_range, other.curve_range),
            u_range: span(self.u_range, other.u_range),
            v_range: span(self.v_range, other.v_range),
            min_corner,
            max_corner,
        }
    }

    fn start(&self) -> f64 {
        self.curve_range.0
    }
}

/// Halvings allowed per pair of pieces whose boxes overlap, pooled over
/// all of them, before the search gives up.
const SPLITS_PER_PAIR: usize = 1024;
/// Halvings from a pair of pieces down to one subpiece pair, beyond which
/// the pair counts as never separating.
const MAX_DEPTH: u32 = 64;
/// Newton steps from the centre of a subpiece pair.
const NEWTON_STEPS: usize = 32;
/// Parameters within this fraction of a domain's width outside it (or
/// outside a subpiece pair) count as inside.
const SLACK: f64 = 1e-10;

/// The hits of `curve` over `range` (the curve's own bounds if `periodic`,
/// a trimmed range otherwise) with `surface`, complete unless some pair
/// of pieces was left unsettled.
///
/// The curve's knot spans and the surface's knot-span patches are Bézier
/// pieces; the pairs whose control-point boxes come within
/// `tol.confusion` of each other are found through the surface's patch
/// box hierarchy. Each pair is then tested, and halved (the larger piece,
/// a patch along the direction it stretches most) until it is settled:
///
/// - dropped once the control points' boxes, or their projections on the
///   normal of the plane fitted to the patch, are `tol.confusion` apart;
/// - once the pieces provably meet at most once (see
///   [`meets_at_most_once`]), solved by Newton's method on `C(t) = S(u,
///   v)` from the pair's centre, against the B-splines themselves. A hit
///   within the pair settles it, and so does one within the merging
///   distance below of the whole curve piece, as the pair's hit would
///   merge into it; any other hit is still kept. (Near a shallow
///   crossing, the pieces beside it stay within `tol.confusion` of each
///   other without meeting.)
///
/// A tangency or an overlap never separates into such pairs, and a hit at
/// a singular point of the surface defeats Newton's method. A pair is
/// left as never separating once its curve piece is no larger than its
/// patch and both lie within `tol.confusion` of a plane across which
/// they cannot be told apart (see [`flat_against`]), or after
/// [`MAX_DEPTH`] halvings; the search goes on with the other pairs. If
/// the budget of halvings runs out first, the pairs still pending are
/// left unsettled too. Either way, the result is incomplete, with the
/// unsettled pairs as its regions.
///
/// Hits within `sqrt(tol.confusion)` of each other merge, as in the
/// sampled search, keeping the one closest to the surface.
pub(super) fn bspline_curve_surface(
    curve: &BSplineCurve3D,
    range: (f64, f64),
    periodic: bool,
    surface: &BSplineSurface,
    tol: Tolerance,
) -> CurveSurfaceHits {
    search_pieces(curve, range, periodic, surface, tol, SPLITS_PER_PAIR)
}

/// [`bspline_curve_surface`] with `splits_per_pair` halvings allowed per
/// initial pair of pieces.
fn search_pieces(
    curve: &BSplineCurve3D,
    range: (f64, f64),
    periodic: bool,
    surface: &BSplineSurface,
    tol: Tolerance,
    splits_per_pair: usize,
) -> CurveSurfaceHits {
    let pieces = CurvePieces::new(curve, range);
    let mut search = Search::new(curve, &pieces, range, periodic, surface, tol);
    let tree = surface.patch_boxes();
    let (rows, cols, h) = search.patches_shape;
    let mut scratch = vec![0.0; rows * cols * h];
    let mut points = Vec::new();
    for i in 0..pieces.len() {
        euclidean(pieces.piece(i), &mut points);
        tree.overlapping(&bounding_box(&points), tol.confusion, |k| {
            let patch = search.patches.take();
            let rect = surface.patch_net(k, &mut scratch, search.patches.get_mut(patch));
            let piece = search.curves.take();
            search
                .curves
                .get_mut(piece)
                .copy_from_slice(pieces.piece(i));
            search.pending.push(Node {
                curve: piece,
                t: pieces.ranges[i],
                patch,
                rect,
                depth: 0,
            });
        });
    }
    search.budget = splits_per_pair * search.pending.len();
    let mut exhausted = false;
    while let Some(node) = search.pending.pop() {
        if search.visit(node).is_err() {
            exhausted = true;
            break;
        }
    }
    search.finish(exhausted)
}

/// Whether the points of a patch lie within `gap` of a plane across
/// `normal`, and those of a curve piece within `gap` of that slab: no
/// halving of the pair can then be told apart along the normal.
fn flat_against(normal: [f64; 3], curve: &[[f64; 3]], patch: &[[f64; 3]], gap: f64) -> bool {
    let length = norm(normal);
    if !(length > 0.0 && length.is_finite()) {
        return false;
    }
    let extent = |points: &[[f64; 3]]| {
        points
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
                let y = dot(*x, normal);
                (lo.min(y), hi.max(y))
            })
    };
    let ((c0, c1), (p0, p1)) = (extent(curve), extent(patch));
    let gap = gap * length;
    p1 - p0 <= gap && c0 >= p0 - gap && c1 <= p1 + gap
}

/// A parameter interval of the curve or one direction of the surface.
#[derive(Clone, Copy)]
pub(super) struct Range {
    pub(super) lo: f64,
    pub(super) hi: f64,
    pub(super) periodic: bool,
}

impl Range {
    pub(super) fn slack(self) -> f64 {
        SLACK * (self.hi - self.lo)
    }

    /// `x` wrapped into a periodic range (its end reading as its start),
    /// or clamped into the range from within its slack; `None` beyond.
    pub(super) fn admit(self, x: f64) -> Option<f64> {
        let (lo, hi, slack) = (self.lo, self.hi, self.slack());
        if self.periodic {
            let wrapped = lo + (x - lo).rem_euclid(hi - lo);
            Some(if hi - wrapped <= slack { lo } else { wrapped })
        } else if x < lo - slack || x > hi + slack {
            None
        } else {
            Some(x.clamp(lo, hi))
        }
    }
}

/// A pair of subpieces awaiting its visit: the curve's control points in
/// slot `curve` over `t`, the surface's net in slot `patch` over `rect`
/// (`[u0, u1, v0, v1]`), and the halvings from the pair it came from.
struct Node {
    curve: usize,
    t: (f64, f64),
    patch: usize,
    rect: [f64; 4],
    depth: u32,
}

/// The search ran out of budget.
struct Exhausted;

struct Search<'a> {
    curve: &'a BSplineCurve3D,
    surface: &'a BSplineSurface,
    tol: Tolerance,
    ranges: [Range; 3],
    curve_shape: (usize, usize, usize),
    patches_shape: (usize, usize, usize),
    curves: Pool,
    patches: Pool,
    pending: Vec<Node>,
    curve_points: Vec<[f64; 3]>,
    patch_points: Vec<[f64; 3]>,
    /// Hits found, with their distance to the surface.
    found: Vec<(CurveSurfaceHit, f64)>,
    /// The pairs left as never separating.
    unseparated: Vec<CurveSurfaceRegion>,
    budget: usize,
}

impl<'a> Search<'a> {
    fn new(
        curve: &'a BSplineCurve3D,
        pieces: &CurvePieces,
        (lo, hi): (f64, f64),
        periodic: bool,
        surface: &'a BSplineSurface,
        tol: Tolerance,
    ) -> Self {
        let (u0, u1) = surface.u_bounds();
        let (v0, v1) = surface.v_bounds();
        let patches_shape = surface.patch_net_shape();
        let (rows, cols, h) = patches_shape;
        Search {
            curve,
            surface,
            tol,
            ranges: [
                Range { lo, hi, periodic },
                Range {
                    lo: u0,
                    hi: u1,
                    periodic: surface.u_period().is_some(),
                },
                Range {
                    lo: v0,
                    hi: v1,
                    periodic: surface.v_period().is_some(),
                },
            ],
            curve_shape: (pieces.width, 1, 4),
            patches_shape,
            curves: Pool::new(4 * pieces.width),
            patches: Pool::new(rows * cols * h),
            pending: Vec::new(),
            curve_points: Vec::with_capacity(pieces.width),
            patch_points: Vec::with_capacity(rows * cols),
            found: Vec::new(),
            unseparated: Vec::new(),
            budget: 0,
        }
    }

    /// Settles `node`, leaves it as never separating, or halves it into
    /// two pending pairs; out of budget, puts it back.
    fn visit(&mut self, node: Node) -> Result<(), Exhausted> {
        let (rows, cols, h) = self.patches_shape;
        euclidean(self.curves.get(node.curve), &mut self.curve_points);
        net_points(self.patches.get(node.patch), h, &mut self.patch_points);
        let (curve_box, patch_box) = (
            bounding_box(&self.curve_points),
            bounding_box(&self.patch_points),
        );
        let gap = self.tol.confusion;
        let normal = net_normal(&self.patch_points, rows, cols);
        if boxes_apart(&curve_box, &patch_box, gap)
            || apart_along(normal, &self.curve_points, &self.patch_points, gap)
        {
            self.release(&node);
            return Ok(());
        }
        let once = meets_at_most_once(
            &self.curve_points,
            self.patches.get(node.patch),
            &self.patch_points,
            self.patches_shape,
        );
        if once {
            let r = node.rect;
            let centre = [
                0.5 * (node.t.0 + node.t.1),
                0.5 * (r[0] + r[1]),
                0.5 * (r[2] + r[3]),
            ];
            let within = |x: f64, (a, b): (f64, f64), range: Range| {
                a - range.slack() <= x && x <= b + range.slack()
            };
            if let Some(x) = self.newton(centre)
                && let Some(hit) = self.record(x)
            {
                let merges = || {
                    let gap = self.merge_gap();
                    self.curve_points
                        .iter()
                        .all(|p| norm(sub(*p, [hit.x, hit.y, hit.z])) <= gap)
                };
                let inside = within(x[0], node.t, self.ranges[0])
                    && within(x[1], (r[0], r[1]), self.ranges[1])
                    && within(x[2], (r[2], r[3]), self.ranges[2]);
                if inside || merges() {
                    self.release(&node);
                    return Ok(());
                }
            }
        }
        let curve_larger = box_size(&curve_box) >= box_size(&patch_box);
        let stuck =
            !curve_larger && flat_against(normal, &self.curve_points, &self.patch_points, gap);
        if stuck || node.depth >= MAX_DEPTH {
            let region = self.region(&node, &curve_box);
            self.unseparated.push(region);
            self.release(&node);
            return Ok(());
        }
        if self.budget == 0 {
            self.pending.push(node);
            return Err(Exhausted);
        }
        self.budget -= 1;
        self.split(node, curve_larger);
        Ok(())
    }

    /// The region of `node`, whose curve piece's control points have the
    /// box `curve_box`.
    fn region(&self, node: &Node, curve_box: &Aabb) -> CurveSurfaceRegion {
        let r = node.rect;
        CurveSurfaceRegion {
            curve_range: node.t,
            u_range: (r[0], r[1]),
            v_range: (r[2], r[3]),
            min_corner: Point3D::new(curve_box[0], curve_box[1], curve_box[2]),
            max_corner: Point3D::new(curve_box[3], curve_box[4], curve_box[5]),
        }
    }

    /// The hits found, complete unless pairs were left as never
    /// separating or, if `exhausted`, still pending.
    fn finish(mut self, exhausted: bool) -> CurveSurfaceHits {
        let mut regions = std::mem::take(&mut self.unseparated);
        for node in std::mem::take(&mut self.pending) {
            euclidean(self.curves.get(node.curve), &mut self.curve_points);
            regions.push(self.region(&node, &bounding_box(&self.curve_points)));
        }
        CurveSurfaceHits {
            hits: self.hits(),
            completeness: of_unsettled(regions, exhausted),
        }
    }

    /// Halves the curve piece of `node` if `curve`, else its patch along
    /// the direction its control net is longest, and queues both halves.
    fn split(&mut self, node: Node, curve: bool) {
        let depth = node.depth + 1;
        if curve {
            let upper = self.curves.halve(node.curve, self.curve_shape, true);
            let patch = self.copy_patch(node.patch);
            let mid = 0.5 * (node.t.0 + node.t.1);
            self.pending.push(Node {
                t: (mid, node.t.1),
                curve: upper,
                patch,
                depth,
                ..node
            });
            self.pending.push(Node {
                t: (node.t.0, mid),
                depth,
                ..node
            });
            return;
        }
        // The mean length of the net's polylines along each direction.
        let (rows, cols, _) = self.patches_shape;
        let at = |i: usize, j: usize| self.patch_points[i * cols + j];
        let length = |a: [f64; 3], b: [f64; 3]| norm(sub(b, a));
        let along_rows: f64 = (0..rows - 1)
            .flat_map(|i| (0..cols).map(move |j| length(at(i, j), at(i + 1, j))))
            .sum();
        let along_cols: f64 = (0..rows)
            .flat_map(|i| (0..cols - 1).map(move |j| length(at(i, j), at(i, j + 1))))
            .sum();
        let along_u = along_rows * rows as f64 >= along_cols * cols as f64;
        let upper = self.patches.halve(node.patch, self.patches_shape, along_u);
        let curve = self.copy_curve(node.curve);
        let r = node.rect;
        let (lower_rect, upper_rect) = if along_u {
            let mid = 0.5 * (r[0] + r[1]);
            ([r[0], mid, r[2], r[3]], [mid, r[1], r[2], r[3]])
        } else {
            let mid = 0.5 * (r[2] + r[3]);
            ([r[0], r[1], r[2], mid], [r[0], r[1], mid, r[3]])
        };
        self.pending.push(Node {
            curve,
            patch: upper,
            rect: upper_rect,
            depth,
            ..node
        });
        self.pending.push(Node {
            rect: lower_rect,
            depth,
            ..node
        });
    }

    fn copy_patch(&mut self, slot: usize) -> usize {
        let copy = self.patches.take();
        self.patches.copy(slot, copy);
        copy
    }

    fn copy_curve(&mut self, slot: usize) -> usize {
        let copy = self.curves.take();
        self.curves.copy(slot, copy);
        copy
    }

    fn release(&mut self, node: &Node) {
        self.curves.release(node.curve);
        self.patches.release(node.patch);
    }

    /// Newton's method on `C(t) - S(u, v) = 0` from `x`, each step halved
    /// until the residual decreases: the root, or the point where the
    /// residual stops decreasing (rounding), or `None` when the system is
    /// singular or the steps do not settle.
    fn newton(&self, mut x: [f64; 3]) -> Option<[f64; 3]> {
        let eval = |x: [f64; 3]| {
            let (c, ct) = self.curve.eval_d1(x[0]);
            let (s, [su, sv]) = self.surface.eval_d1(x[1], x[2]);
            (c - s, [ct, su, sv])
        };
        let (mut f, mut d) = eval(x);
        for _ in 0..NEWTON_STEPS {
            if f.square_magnitude() == 0.0 {
                return Some(x);
            }
            let [ct, su, sv] = d;
            let jacobian = [
                [ct.x, -su.x, -sv.x],
                [ct.y, -su.y, -sv.y],
                [ct.z, -su.z, -sv.z],
            ];
            let step = solve_3x3(jacobian, [-f.x, -f.y, -f.z], 1e-14)?;
            let mut scale = 1.0;
            loop {
                let trial = std::array::from_fn(|i| x[i] + scale * step[i]);
                let (ft, dt) = eval(trial);
                if ft.square_magnitude() < f.square_magnitude() {
                    (x, f, d) = (trial, ft, dt);
                    break;
                }
                scale *= 0.5;
                if scale < 1e-3 {
                    return Some(x);
                }
            }
            let moved: f64 = step.iter().map(|s| (scale * s).abs()).sum();
            if moved <= 1e-14 * (3.0 + x.iter().map(|v| v.abs()).sum::<f64>()) {
                return Some(x);
            }
        }
        None
    }

    /// Keeps the hit at `x` if its parameters lie in their ranges and its
    /// point within `tol.confusion` of the surface, and returns its point.
    fn record(&mut self, x: [f64; 3]) -> Option<Point3D> {
        let [t, u, v] = [0, 1, 2].map(|i| self.ranges[i].admit(x[i]));
        let (t, u, v) = (t?, u?, v?);
        let point = self.curve.eval_point(t);
        let distance = point.distance(self.surface.eval_point(u, v));
        if distance > self.tol.confusion {
            return None;
        }
        let hit = CurveSurfaceHit {
            curve_parameter: Some(t),
            surface_parameters: Some((u, v)),
            point,
            multiplicity: {
                let (_, tangent) = self.curve.eval_d1(t);
                let (_, [su, sv]) = self.surface.eval_d1(u, v);
                curve_surface_multiplicity(tangent, su.cross(sv), point, self.tol)
            },
        };
        self.found.push((hit, distance));
        Some(point)
    }

    /// The distance within which hits merge.
    fn merge_gap(&self) -> f64 {
        self.tol.confusion.sqrt()
    }

    /// The hits found, merged within `sqrt(tol.confusion)` (the closest to
    /// the surface kept), ordered by curve parameter.
    fn hits(mut self) -> Vec<CurveSurfaceHit> {
        self.found.sort_by(|a, b| a.1.total_cmp(&b.1));
        let merge_gap = self.merge_gap();
        let mut hits: Vec<CurveSurfaceHit> = Vec::new();
        for (hit, _) in self.found {
            if hits.iter().all(|h| h.point.distance(hit.point) > merge_gap) {
                hits.push(hit);
            }
        }
        sort_by_known_parameter(&mut hits, |hit| hit.curve_parameter);
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point3D;
    use crate::curves::{Curve3D, ParametricCurve3D, TrimmedCurve3D};
    use crate::intersect::curve_surface::sampled_curve_surface;
    use crate::intersect::{IncompleteReason, search_curve_surface};
    use crate::surfaces::Surface;

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
    }

    /// A bicubic 6×6 net over `[0, 5]^2` with heights `sin(0.7 i j)`, or
    /// flat; rational with uneven weights if `rational`.
    fn surface(height: impl Fn(usize, usize) -> f64, rational: bool) -> BSplineSurface {
        let poles = (0..6)
            .map(|i| {
                (0..6)
                    .map(|j| Point3D::new(i as f64, j as f64, height(i, j)))
                    .collect()
            })
            .collect();
        let (knots, mults) = (vec![0.0, 1.0, 2.0, 3.0], vec![4, 1, 1, 4]);
        if !rational {
            return BSplineSurface::new(
                3,
                3,
                poles,
                knots.clone(),
                mults.clone(),
                knots,
                mults,
                false,
                false,
            )
            .unwrap();
        }
        let weights = (0..6)
            .map(|i| {
                (0..6)
                    .map(|j| 0.5 + 0.3 * ((7 * i + 3 * j) % 5) as f64)
                    .collect()
            })
            .collect();
        BSplineSurface::new_rational(
            3,
            3,
            poles,
            weights,
            knots.clone(),
            mults.clone(),
            knots,
            mults,
            false,
            false,
        )
        .unwrap()
    }

    fn wavy(rational: bool) -> BSplineSurface {
        surface(|i, j| ((i * j) as f64 * 0.7).sin(), rational)
    }

    fn flat() -> BSplineSurface {
        surface(|_, _| 0.0, false)
    }

    /// A cubic rising from `z = -2` to `z = 2` along a wavy path between
    /// two random points over the surfaces' footprint.
    fn rising(rng: &mut Lcg, rational: bool) -> BSplineCurve3D {
        let (x0, y0) = (0.3 + rng.next() * 4.4, 0.3 + rng.next() * 4.4);
        let (x1, y1) = (0.3 + rng.next() * 4.4, 0.3 + rng.next() * 4.4);
        let poles = (0..6)
            .map(|k| {
                let s = k as f64 / 5.0;
                Point3D::new(
                    x0 + (x1 - x0) * s + 0.3 * (7.0 * s).sin(),
                    y0 + (y1 - y0) * s,
                    -2.0 + 4.0 * s,
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

    /// The hits of the piece-pair search, which must be complete.
    fn search<C: ParametricCurve3D>(curve: &C, surface: &BSplineSurface) -> Vec<CurveSurfaceHit> {
        let bspline = curve.as_bspline().unwrap();
        let periodic = curve.period().is_some();
        let found = bspline_curve_surface(bspline, curve.bounds(), periodic, surface, TOL);
        assert!(found.is_complete(), "{:?}", found.completeness());
        found.into_hits()
    }

    /// Asserts every hit lies on both shapes, with parameters in range.
    #[track_caller]
    fn assert_on_both<C: ParametricCurve3D>(
        hits: &[CurveSurfaceHit],
        curve: &C,
        surface: &BSplineSurface,
    ) {
        let (lo, hi) = curve.bounds();
        let ((u0, u1), (v0, v1)) = (surface.u_bounds(), surface.v_bounds());
        for hit in hits {
            let (u, v) = hit.surface_parameters.unwrap();
            assert!((lo..=hi).contains(&hit.curve_parameter.unwrap()), "{hit:?}");
            assert!((u0..=u1).contains(&u) && (v0..=v1).contains(&v), "{hit:?}");
            assert!(
                hit.point
                    .distance(curve.eval_point(hit.curve_parameter.unwrap()))
                    < 1e-12
            );
            assert!(hit.point.distance(surface.eval_point(u, v)) <= TOL.confusion);
        }
        for pair in hits.windows(2) {
            assert!(pair[0].curve_parameter < pair[1].curve_parameter);
            assert!(pair[0].point.distance(pair[1].point) > TOL.confusion.sqrt());
        }
    }

    /// Asserts the search finds the sampled search's hits within the
    /// curve's range, and returns them.
    #[track_caller]
    fn assert_matches_sampled<C: ParametricCurve3D>(
        curve: &C,
        surface: &BSplineSurface,
    ) -> Vec<CurveSurfaceHit> {
        let found = search(curve, surface);
        assert_on_both(&found, curve, surface);
        let (lo, hi) = curve.bounds();
        let periodic = curve.period().is_some();
        let sampled: Vec<_> = sampled_curve_surface(curve, surface, TOL)
            .into_iter()
            .filter(|h| periodic || (lo - 1e-9..=hi + 1e-9).contains(&h.curve_parameter.unwrap()))
            .collect();
        assert_eq!(found.len(), sampled.len(), "{found:#?}\n{sampled:#?}");
        for hit in &sampled {
            assert!(
                found.iter().any(|f| f.point.distance(hit.point) < 1e-6),
                "{hit:?} missing from {found:?}"
            );
        }
        found
    }

    #[test]
    fn test_polynomial_curves_match_sampled() {
        let surface = wavy(false);
        let mut rng = Lcg(42);
        let mut count = 0;
        for _ in 0..32 {
            count += assert_matches_sampled(&rising(&mut rng, false), &surface).len();
        }
        assert!(count >= 32);
    }

    #[test]
    fn test_rational_curves_and_surface_match_sampled() {
        let surface = wavy(true);
        let mut rng = Lcg(7);
        let mut count = 0;
        for _ in 0..24 {
            count += assert_matches_sampled(&rising(&mut rng, true), &surface).len();
        }
        assert!(count >= 24);
    }

    /// An exact quarter cylinder of radius 2 about the z axis, over `z` in
    /// `[0, 1]`.
    fn quarter_cylinder() -> BSplineSurface {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let arc = [(2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        let poles = arc
            .iter()
            .map(|&(x, y)| vec![Point3D::new(x, y, 0.0), Point3D::new(x, y, 1.0)])
            .collect();
        BSplineSurface::new_rational(
            2,
            1,
            poles,
            vec![vec![1.0, 1.0], vec![w, w], vec![1.0, 1.0]],
            vec![0.0, 1.0],
            vec![3, 3],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap()
    }

    #[test]
    fn test_rational_cylinder_is_hit_at_its_radius() {
        let cylinder = quarter_cylinder();
        for k in 0..=8 {
            let angle = k as f64 * std::f64::consts::FRAC_PI_2 / 8.0;
            let (c, s) = (angle.cos(), angle.sin());
            let poles = vec![
                Point3D::new(0.0, 0.0, 0.3),
                Point3D::new(4.0 * c, 4.0 * s, 0.7),
            ];
            let ray = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
            let hits = assert_matches_sampled(&ray, &cylinder);
            assert_eq!(hits.len(), 1);
            assert!((hits[0].curve_parameter.unwrap() - 0.5).abs() < 1e-12);
            assert!((hits[0].point.z - 0.5).abs() < 1e-12);
        }
    }

    /// A closed cubic loop around `(2.5, 2.5)` swinging above and below
    /// the wavy surface.
    fn periodic_loop() -> BSplineCurve3D {
        let poles = (0..8)
            .map(|k| {
                let angle = k as f64 * std::f64::consts::TAU / 8.0;
                let z = if k % 2 == 0 { 1.5 } else { -1.5 };
                Point3D::new(2.5 + 1.8 * angle.cos(), 2.5 + 1.8 * angle.sin(), z)
            })
            .collect();
        let knots = (0..=8).map(f64::from).collect();
        BSplineCurve3D::new(3, poles, knots, vec![1; 9], true).unwrap()
    }

    #[test]
    fn test_periodic_curve_matches_sampled() {
        let surface = wavy(false);
        let ring = periodic_loop();
        let hits = assert_matches_sampled(&ring, &surface);
        assert!(hits.len() >= 8);
        assert!(hits.iter().all(|h| h.curve_parameter.unwrap() < 8.0));
    }

    #[test]
    fn test_trimmed_periodic_curve_across_the_seam() {
        let surface = wavy(false);
        let ring = periodic_loop();
        let all = search(&ring, &surface);
        let arc = TrimmedCurve3D::new(&ring, 6.5, 9.5).unwrap();
        let hits = assert_matches_sampled(&arc, &surface);
        let wrapped = |t: f64| if t < 6.5 { t + 8.0 } else { t };
        let expected: Vec<f64> = all
            .iter()
            .map(|h| wrapped(h.curve_parameter.unwrap()))
            .filter(|&t| t <= 9.5)
            .collect();
        assert!(expected.iter().any(|&t| t < 8.0) && expected.iter().any(|&t| t > 8.0));
        assert_eq!(hits.len(), expected.len());
        for hit in &hits {
            assert!(
                expected
                    .iter()
                    .any(|&t| (t - hit.curve_parameter.unwrap()).abs() < 1e-9)
            );
        }
    }

    /// A periodic quadratic tube about the z axis, radius about 2, over
    /// `z` in `[0, 3]`.
    fn tube() -> BSplineSurface {
        let poles = (0..6)
            .map(|k| {
                let angle = k as f64 * std::f64::consts::TAU / 6.0;
                let (c, s) = (angle.cos(), angle.sin());
                vec![
                    Point3D::new(2.0 * c, 2.0 * s, 0.0),
                    Point3D::new(2.2 * c, 2.2 * s, 3.0),
                ]
            })
            .collect();
        BSplineSurface::new(
            2,
            1,
            poles,
            (0..=6).map(f64::from).collect(),
            vec![1; 7],
            vec![0.0, 1.0],
            vec![2, 2],
            true,
            false,
        )
        .unwrap()
    }

    #[test]
    fn test_periodic_surface_matches_sampled() {
        let tube = tube();
        for k in 0..24 {
            let angle = k as f64 * std::f64::consts::TAU / 24.0;
            let (c, s) = (angle.cos(), angle.sin());
            let poles = vec![
                Point3D::new(0.0, 0.0, 0.5),
                Point3D::new(4.0 * c, 4.0 * s, 2.5),
            ];
            let ray = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
            let hits = assert_matches_sampled(&ray, &tube);
            assert_eq!(hits.len(), 1);
            let (u, _) = hits[0].surface_parameters.unwrap();
            assert!((0.0..6.0).contains(&u));
        }
        // Through the seam, slanting either way across it.
        let seam = tube.eval_point(0.0, 0.5);
        for slant in [0.3, -0.3] {
            let line = segment(
                Point3D::new(0.3 * seam.x, 0.3 * seam.y + slant, 0.5),
                Point3D::new(1.7 * seam.x, 1.7 * seam.y - slant, 2.5),
            );
            let hits = assert_matches_sampled(&line, &tube);
            assert_eq!(hits.len(), 1, "{hits:#?}");
            assert!(hits[0].point.distance(seam) < 1e-12);
            let (u, v) = hits[0].surface_parameters.unwrap();
            assert!(u.min(6.0 - u) < 1e-9 && (v - 0.5).abs() < 1e-9, "{u} {v}");
        }
    }

    /// A cubic crossing the wavy surface back and forth.
    fn zigzag() -> BSplineCurve3D {
        let poles = (0..10)
            .map(|k| {
                let z = if k % 2 == 0 { 1.5 } else { -1.5 };
                Point3D::new(0.3 + 0.48 * k as f64, 1.0 + 0.3 * k as f64, z)
            })
            .collect();
        let knots = (0..=7).map(f64::from).collect();
        BSplineCurve3D::new(3, poles, knots, vec![4, 1, 1, 1, 1, 1, 1, 4], false).unwrap()
    }

    #[test]
    fn test_trimmed_curves_search_only_their_range() {
        let surface = wavy(false);
        let curve = zigzag();
        let all = assert_matches_sampled(&curve, &surface);
        assert!(all.len() >= 6);
        for (first, last) in [
            (0.7, 2.9),
            (1.0, 5.0),
            (all[2].curve_parameter.unwrap(), 6.0),
        ] {
            let expected: Vec<_> = all
                .iter()
                .filter(|h| (first - 1e-9..=last + 1e-9).contains(&h.curve_parameter.unwrap()))
                .collect();
            let borrowed = TrimmedCurve3D::new(&curve, first, last).unwrap();
            let owned = TrimmedCurve3D::new(curve.clone(), first, last).unwrap();
            for hits in [
                assert_matches_sampled(&borrowed, &surface),
                assert_matches_sampled(&owned, &surface),
                search_curve_surface(&owned, &Surface::from(&surface), TOL).into_hits(),
            ] {
                assert_eq!(hits.len(), expected.len());
                for (hit, want) in hits.iter().zip(&expected) {
                    assert!(
                        (hit.curve_parameter.unwrap() - want.curve_parameter.unwrap()).abs() < 1e-9
                    );
                }
            }
        }
        let enums = search_curve_surface(
            &Curve3D::BSpline(curve.clone()),
            &Surface::from(&surface),
            TOL,
        );
        assert_eq!(enums, search_curve_surface(&curve, &surface, TOL));
    }

    fn segment(from: Point3D, to: Point3D) -> BSplineCurve3D {
        BSplineCurve3D::new(1, vec![from, to], vec![0.0, 1.0], vec![2, 2], false).unwrap()
    }

    #[test]
    fn test_hits_at_curve_ends_and_surface_edges() {
        let plane = flat();
        let cases = [
            // Ending on the surface, starting on it, and through its edges
            // and corners.
            (Point3D::new(2.3, 1.7, -1.0), Point3D::new(2.3, 1.7, 0.0)),
            (Point3D::new(2.3, 1.7, 0.0), Point3D::new(2.9, 1.1, 1.0)),
            (Point3D::new(0.0, 2.2, -1.0), Point3D::new(0.0, 2.2, 1.0)),
            (Point3D::new(3.1, 5.0, 1.0), Point3D::new(3.1, 5.0, -1.0)),
            (Point3D::new(0.0, 0.0, -1.0), Point3D::new(0.0, 0.0, 1.0)),
            (Point3D::new(5.0, 5.0, 1.0), Point3D::new(5.0, 5.0, -1.0)),
        ];
        for (from, to) in cases {
            let curve = segment(from, to);
            let hits = search(&curve, &plane);
            assert_on_both(&hits, &curve, &plane);
            assert_eq!(hits.len(), 1, "{from:?} {to:?}");
            let expected = if from.z == 0.0 {
                from
            } else {
                Point3D::new(from.x, from.y, 0.0)
            };
            assert!(hits[0].point.distance(expected) < 1e-12);
        }
    }

    #[test]
    fn test_hit_on_knots_of_both_is_reported_once() {
        let surface = wavy(false);
        let corner = surface.eval_point(1.0, 2.0);
        let poles = vec![
            Point3D::new(corner.x - 0.3, corner.y - 0.2, corner.z - 1.0),
            corner,
            Point3D::new(corner.x + 0.5, corner.y + 0.1, corner.z + 1.0),
        ];
        let curve =
            BSplineCurve3D::new(1, poles, vec![0.0, 1.0, 2.0], vec![2, 1, 2], false).unwrap();
        let hits = assert_matches_sampled(&curve, &surface);
        assert_eq!(hits.len(), 1);
        assert!((hits[0].curve_parameter.unwrap() - 1.0).abs() < 1e-12);
        let (u, v) = hits[0].surface_parameters.unwrap();
        assert!((u - 1.0).abs() < 1e-12 && (v - 2.0).abs() < 1e-12);
    }

    /// A parabola over the flat surface dipping `depth` below it (meeting
    /// it at `t = (1 ± sqrt(depth / (1 + depth))) / 2`).
    fn dip(depth: f64) -> BSplineCurve3D {
        let poles = vec![
            Point3D::new(1.0, 2.5, 1.0),
            Point3D::new(2.5, 2.5, -1.0 - 2.0 * depth),
            Point3D::new(4.0, 2.5, 1.0),
        ];
        BSplineCurve3D::new(2, poles, vec![0.0, 1.0], vec![3, 3], false).unwrap()
    }

    #[test]
    fn test_close_hits_near_a_tangency_are_separated() {
        let plane = flat();
        for depth in [1e-3, 1e-6] {
            let curve = dip(depth);
            let hits = search(&curve, &plane);
            assert_on_both(&hits, &curve, &plane);
            let half = 0.5 * (depth / (1.0 + depth)).sqrt();
            assert_eq!(hits.len(), 2, "{hits:?}");
            assert!((hits[0].curve_parameter.unwrap() - (0.5 - half)).abs() < 1e-9);
            assert!((hits[1].curve_parameter.unwrap() - (0.5 + half)).abs() < 1e-9);
        }
    }

    /// The reason and regions of an incomplete result.
    #[track_caller]
    fn incomplete(hits: &CurveSurfaceHits) -> (IncompleteReason, &[CurveSurfaceRegion]) {
        match hits.completeness() {
            Completeness::Incomplete { reason, regions } => (*reason, regions),
            other => panic!("expected an incomplete search, got {other:?}"),
        }
    }

    /// Whether `region` holds the curve parameter `t`, the surface
    /// parameters `(u, v)` and the point `p`.
    fn holds(region: &CurveSurfaceRegion, t: f64, (u, v): (f64, f64), p: Point3D) -> bool {
        let within = |x: f64, (lo, hi): (f64, f64)| lo - 1e-9 <= x && x <= hi + 1e-9;
        let (lo, hi) = (region.min_corner, region.max_corner);
        within(t, region.curve_range)
            && within(u, region.u_range)
            && within(v, region.v_range)
            && within(p.x, (lo.x, hi.x))
            && within(p.y, (lo.y, hi.y))
            && within(p.z, (lo.z, hi.z))
    }

    /// Asserts the hits of `search_curve_surface` include the sampled
    /// search's, every one lying on both shapes.
    #[track_caller]
    fn assert_sampled_hits_kept(
        hits: &CurveSurfaceHits,
        curve: &BSplineCurve3D,
        plane: &BSplineSurface,
    ) {
        for hit in hits {
            assert!(
                hit.point
                    .distance(curve.eval_point(hit.curve_parameter.unwrap()))
                    < 1e-12
            );
            assert!(plane.project_point(hit.point, TOL).distance <= TOL.confusion);
        }
        for hit in sampled_curve_surface(curve, plane, TOL) {
            let near = |h: &CurveSurfaceHit| h.point.distance(hit.point) <= TOL.confusion.sqrt();
            assert!(hits.iter().any(near), "{hit:?} missing from {hits:?}");
        }
    }

    #[test]
    fn test_curve_lying_on_the_surface_is_unseparated_along_it() {
        let plane = flat();
        let lying = segment(Point3D::new(1.0, 1.0, 0.0), Point3D::new(4.0, 3.0, 0.0));
        let found = bspline_curve_surface(&lying, lying.bounds(), false, &plane, TOL);
        let (reason, regions) = incomplete(&found);
        assert_eq!(reason, IncompleteReason::Unseparated);
        // The patches it crosses merge into one region over the whole curve.
        assert_eq!(regions.len(), 1, "{regions:#?}");
        assert_eq!(regions[0].curve_range, (0.0, 1.0));
        for k in 0..=16 {
            let t = k as f64 / 16.0;
            let p = lying.eval_point(t);
            let foot = plane.project_point(p, TOL);
            assert!(
                holds(&regions[0], t, (foot.u, foot.v), p),
                "{t}: {regions:?}"
            );
        }
        let hits = search_curve_surface(&lying, &plane, TOL);
        assert_eq!(hits.completeness(), found.completeness());
        assert!(!hits.is_empty());
        assert_sampled_hits_kept(&hits, &lying, &plane);
    }

    #[test]
    fn test_tangent_contact_is_unseparated_around_it() {
        let plane = flat();
        let tangent = dip(0.0);
        let contact = Point3D::new(2.5, 2.5, 0.0);
        assert_eq!(tangent.eval_point(0.5), contact);
        let foot = plane.project_point(contact, TOL);
        let found = bspline_curve_surface(&tangent, tangent.bounds(), false, &plane, TOL);
        let (reason, regions) = incomplete(&found);
        assert_eq!(reason, IncompleteReason::Unseparated);
        assert!(
            regions
                .iter()
                .any(|r| holds(r, 0.5, (foot.u, foot.v), contact))
        );
        for region in regions {
            let (t0, t1) = region.curve_range;
            assert!(t0 <= 0.5 + 1e-3 && t1 >= 0.5 - 1e-3, "{region:?}");
            assert!(t1 - t0 < 1e-2, "{region:?}");
        }
        let hits = search_curve_surface(&tangent, &plane, TOL);
        assert_eq!(hits.completeness(), found.completeness());
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert!(hits[0].point.distance(contact) < TOL.confusion.sqrt());
        assert_sampled_hits_kept(&hits, &tangent, &plane);
    }

    #[test]
    fn test_contact_with_a_curved_surface_is_incomplete_around_it() {
        let cylinder = quarter_cylinder();
        for u in [0.2, 0.5, 0.77] {
            let foot = (u, 0.4);
            let on = cylinder.eval_point(u, 0.4);
            // Tangent to the cylinder at `on`, and lying on it.
            let side = crate::Vector3D::new(on.y, -on.x, 0.0) * 0.125;
            let across = segment(on + side, on - side);
            let found = bspline_curve_surface(&across, across.bounds(), false, &cylinder, TOL);
            let (reason, regions) = incomplete(&found);
            assert_eq!(reason, IncompleteReason::Unseparated);
            assert!(
                regions.iter().any(|r| holds(r, 0.5, foot, on)),
                "{regions:#?}"
            );
            let lying = segment(
                cylinder.eval_point(foot.0, 0.1),
                cylinder.eval_point(foot.0, 0.9),
            );
            let found = bspline_curve_surface(&lying, lying.bounds(), false, &cylinder, TOL);
            let (_, regions) = incomplete(&found);
            for k in 0..=8 {
                let t = k as f64 / 8.0;
                let v = 0.1 + 0.8 * t;
                let p = lying.eval_point(t);
                assert!(regions.iter().any(|r| holds(r, t, (foot.0, v), p)), "{t}");
            }
        }
    }

    #[test]
    fn test_exhausted_budget_leaves_the_pending_pairs() {
        let surface = wavy(false);
        let mut rng = Lcg(3);
        for _ in 0..8 {
            let curve = rising(&mut rng, false);
            let all = search(&curve, &surface);
            let found = search_pieces(&curve, curve.bounds(), false, &surface, TOL, 0);
            let (reason, regions) = incomplete(&found);
            assert_eq!(reason, IncompleteReason::BudgetExhausted);
            assert_on_both(&found, &curve, &surface);
            // Every hit is either found or within a region.
            for hit in &all {
                let held = regions.iter().any(|r| {
                    let curve_parameter = hit.curve_parameter.unwrap();
                    let surface_parameters = hit.surface_parameters.unwrap();
                    holds(r, curve_parameter, surface_parameters, hit.point)
                });
                let near = |h: &CurveSurfaceHit| h.point.distance(hit.point) < 1e-9;
                assert!(held || found.iter().any(near), "{hit:?}: {regions:#?}");
            }
            for pair in regions.windows(2) {
                assert!(!pair[0].touches(&pair[1]));
            }
        }
    }

    #[test]
    fn test_transversal_bsplines_are_complete() {
        let surface = wavy(true);
        let mut rng = Lcg(11);
        for _ in 0..8 {
            let curve = rising(&mut rng, true);
            let hits = search_curve_surface(&curve, &surface, TOL);
            assert!(hits.is_complete(), "{:?}", hits.completeness());
            assert!(!hits.is_empty());
        }
    }

    #[test]
    fn test_mixed_shapes_are_settled_and_elementary_pairs_sampled() {
        let surface = wavy(false);
        let cylinder = crate::Cylinder::new(Point3D::ORIGIN, crate::Vector3D::Z, 2.0).unwrap();
        let ray = segment(Point3D::new(0.0, 0.0, 0.5), Point3D::new(4.0, 0.0, 0.5));
        let hits = search_curve_surface(&ray, &cylinder, TOL);
        assert_eq!(hits.completeness(), &Completeness::Complete);
        assert_eq!(hits.len(), 1);
        let line = crate::Line3D::new(Point3D::new(2.0, 2.0, -1.0), crate::Vector3D::Z).unwrap();
        let hits = search_curve_surface(&line, &surface, TOL);
        assert_eq!(hits.completeness(), &Completeness::Complete);
        assert_eq!(hits.len(), 1);
        let axis = crate::Line3D::new(Point3D::new(1.0, 0.0, 0.0), crate::Vector3D::X).unwrap();
        let hits = search_curve_surface(&axis, &cylinder, TOL);
        assert_eq!(hits.completeness(), &Completeness::Sampled);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_misses_have_no_hits() {
        let surface = wavy(true);
        let above = segment(Point3D::new(0.5, 0.5, 2.0), Point3D::new(4.5, 4.0, 2.5));
        let beside = segment(Point3D::new(6.0, 0.0, -1.0), Point3D::new(6.0, 5.0, 1.0));
        for curve in [above, beside] {
            assert!(assert_matches_sampled(&curve, &surface).is_empty());
        }
    }
}
