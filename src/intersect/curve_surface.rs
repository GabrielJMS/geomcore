//! Intersection of a B-spline curve with a B-spline surface by subdivision
//! of their Bézier pieces (see [`bspline_curve_surface`]).

use super::CurveSurfaceHit;
use super::pieces::{
    CurvePieces, Pool, apart_along, bounding_box, box_size, euclidean, meets_at_most_once,
    net_normal, norm, sub,
};
use crate::curves::BSplineCurve3D;
use crate::math::boxtree::boxes_apart;
use crate::math::solve_3x3;
use crate::surface_math::bspline::net_points;
use crate::surfaces::{BSplineSurface, ParametricSurface};
use crate::{Point3D, Tolerance};

/// Halvings allowed per pair of pieces whose boxes overlap, pooled over
/// all of them, before the search gives up.
const SPLITS_PER_PAIR: usize = 256;
/// Halvings from a pair of pieces down to one subpiece pair.
const MAX_DEPTH: u32 = 64;
/// Newton steps from the centre of a subpiece pair.
const NEWTON_STEPS: usize = 32;
/// Parameters within this fraction of a domain's width outside it (or
/// outside a subpiece pair) count as inside.
const SLACK: f64 = 1e-10;

/// The hits of `curve` over `range` (the curve's own bounds if `periodic`,
/// a trimmed range otherwise) with `surface`, or `None` when the search
/// cannot isolate them within its budget.
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
/// a singular point of the surface defeats Newton's method: both exhaust
/// the budget or the depth limit, and the caller falls back to sampling.
/// Hits within `sqrt(tol.confusion)` of each other merge, as in the
/// sampled search, keeping the one closest to the surface.
pub(super) fn bspline_curve_surface(
    curve: &BSplineCurve3D,
    range: (f64, f64),
    periodic: bool,
    surface: &BSplineSurface,
    tol: Tolerance,
) -> Option<Vec<CurveSurfaceHit>> {
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
    search.budget = SPLITS_PER_PAIR * search.pending.len();
    while let Some(node) = search.pending.pop() {
        search.visit(node).ok()?;
    }
    Some(search.hits())
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

/// The search ran out of budget or depth.
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
            budget: 0,
        }
    }

    /// Settles `node` or halves it into two pending pairs.
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
        if node.depth >= MAX_DEPTH || self.budget == 0 {
            return Err(Exhausted);
        }
        self.budget -= 1;
        self.split(node, box_size(&curve_box) >= box_size(&patch_box));
        Ok(())
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
            curve_param: t,
            surface_params: (u, v),
            point,
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
        hits.sort_by(|a, b| a.curve_param.total_cmp(&b.curve_param));
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point3D;
    use crate::curves::{Curve3D, ParametricCurve3D, TrimmedCurve3D};
    use crate::intersect::{intersect_curve_surface, sampled_curve_surface};
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

    /// The hits of the piece-pair search, which must not fall back.
    fn search<C: ParametricCurve3D>(curve: &C, surface: &BSplineSurface) -> Vec<CurveSurfaceHit> {
        let bspline = curve.as_bspline().unwrap();
        let periodic = curve.period().is_some();
        bspline_curve_surface(bspline, curve.bounds(), periodic, surface, TOL)
            .expect("the search isolates every hit")
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
            let (u, v) = hit.surface_params;
            assert!((lo..=hi).contains(&hit.curve_param), "{hit:?}");
            assert!((u0..=u1).contains(&u) && (v0..=v1).contains(&v), "{hit:?}");
            assert!(hit.point.distance(curve.eval_point(hit.curve_param)) < 1e-12);
            assert!(hit.point.distance(surface.eval_point(u, v)) <= TOL.confusion);
        }
        for pair in hits.windows(2) {
            assert!(pair[0].curve_param < pair[1].curve_param);
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
            .filter(|h| periodic || (lo - 1e-9..=hi + 1e-9).contains(&h.curve_param))
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

    #[test]
    fn test_rational_cylinder_is_hit_at_its_radius() {
        // An exact quarter cylinder of radius 2 about the z axis.
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let arc = [(2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        let poles = arc
            .iter()
            .map(|&(x, y)| vec![Point3D::new(x, y, 0.0), Point3D::new(x, y, 1.0)])
            .collect();
        let cylinder = BSplineSurface::new_rational(
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
        .unwrap();
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
            assert!((hits[0].curve_param - 0.5).abs() < 1e-12);
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
        assert!(hits.iter().all(|h| h.curve_param < 8.0));
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
            .map(|h| wrapped(h.curve_param))
            .filter(|&t| t <= 9.5)
            .collect();
        assert!(expected.iter().any(|&t| t < 8.0) && expected.iter().any(|&t| t > 8.0));
        assert_eq!(hits.len(), expected.len());
        for hit in &hits {
            assert!(expected.iter().any(|&t| (t - hit.curve_param).abs() < 1e-9));
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
            let (u, _) = hits[0].surface_params;
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
            let (u, v) = hits[0].surface_params;
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
        for (first, last) in [(0.7, 2.9), (1.0, 5.0), (all[2].curve_param, 6.0)] {
            let expected: Vec<_> = all
                .iter()
                .filter(|h| (first - 1e-9..=last + 1e-9).contains(&h.curve_param))
                .collect();
            let borrowed = TrimmedCurve3D::new(&curve, first, last).unwrap();
            let owned = TrimmedCurve3D::new(curve.clone(), first, last).unwrap();
            for hits in [
                assert_matches_sampled(&borrowed, &surface),
                assert_matches_sampled(&owned, &surface),
                intersect_curve_surface(&owned, &Surface::from(&surface), TOL),
            ] {
                assert_eq!(hits.len(), expected.len());
                for (hit, want) in hits.iter().zip(&expected) {
                    assert!((hit.curve_param - want.curve_param).abs() < 1e-9);
                }
            }
        }
        let enums = intersect_curve_surface(
            &Curve3D::BSpline(curve.clone()),
            &Surface::from(&surface),
            TOL,
        );
        assert_eq!(enums, intersect_curve_surface(&curve, &surface, TOL));
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
        assert!((hits[0].curve_param - 1.0).abs() < 1e-12);
        let (u, v) = hits[0].surface_params;
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
            assert!((hits[0].curve_param - (0.5 - half)).abs() < 1e-9);
            assert!((hits[1].curve_param - (0.5 + half)).abs() < 1e-9);
        }
    }

    #[test]
    fn test_tangency_and_overlap_fall_back_to_sampling() {
        let plane = flat();
        let tangent = dip(0.0);
        let lying = segment(Point3D::new(1.0, 1.0, 0.0), Point3D::new(4.0, 3.0, 0.0));
        for curve in [tangent, lying] {
            let periodic = curve.period().is_some();
            assert!(bspline_curve_surface(&curve, curve.bounds(), periodic, &plane, TOL).is_none());
            let hits = intersect_curve_surface(&curve, &plane, TOL);
            assert!(!hits.is_empty());
            assert_eq!(hits, sampled_curve_surface(&curve, &plane, TOL));
        }
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
