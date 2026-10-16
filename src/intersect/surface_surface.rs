//! The pairs of Bézier patches of two B-spline surfaces that may meet,
//! halved until each is small enough to seed the marching intersection
//! (see [`patch_pairs`]).

use super::pieces::{
    Pool, apart_along, bounding_box, box_size, derivative_cones, net_normal,
    normals_never_parallel, stretches_along_rows, thickness,
};
use crate::Tolerance;
use crate::math::boxtree::{Aabb, boxes_apart};
use crate::surface_math::bspline::net_points;
use crate::surfaces::BSplineSurface;

/// Halvings allowed per pair of patches whose boxes overlap, pooled over
/// all of them, before the remaining pairs are taken as they are.
const SPLITS_PER_PAIR: usize = 256;
/// Halvings from a pair of patches down to a leaf.
const MAX_DEPTH: u32 = 64;
/// A subpatch is flat when its control points lie within this fraction
/// of its box's diagonal from a plane.
const FLATNESS: f64 = 1.0 / 16.0;
/// Pairs whose normals may be parallel are halved until both boxes are
/// this many times `tol.confusion` across.
const SMALLEST: f64 = 1e4;

/// A pair of subpatches that may meet.
pub(super) struct Leaf {
    /// The parameter rectangles `[u0, u1, v0, v1]` of the subpatches.
    pub(super) rects: [[f64; 4]; 2],
    /// The overlap of their control points' boxes, grown by
    /// `tol.confusion`.
    pub(super) overlap: Aabb,
    /// Whether their normals are never parallel, so that every curve of
    /// their intersection leaves them (see [`normals_never_parallel`]).
    pub(super) open: bool,
}

/// The leaves of the search, and whether it finished within its budget.
pub(super) struct PatchPairs {
    pub(super) leaves: Vec<Leaf>,
    pub(super) complete: bool,
}

/// The pairs of subpatches of `a` and `b` that may meet.
///
/// The pairs of knot-span patches whose control-point boxes come within
/// `tol.confusion` of each other are found through the patch box
/// hierarchies. Each pair is then tested, and the larger subpatch halved
/// along the direction its net stretches most, until the pair is settled:
///
/// - dropped once the control points' boxes, or their projections on the
///   normal of the plane fitted to either subpatch, are `tol.confusion`
///   apart;
/// - kept as an open leaf once the normals of the subpatches are provably
///   never parallel and both are flat: no closed loop hides inside, so a
///   seed converged from the pair finds the curves crossing it;
/// - kept as a leaf once both are `SMALLEST` times `tol.confusion`
///   across while their normals may still be parallel (a tangency, or a
///   loop too small to trace).
///
/// Once the budget of halvings is spent, or a pair has been halved
/// `MAX_DEPTH` times, the pairs left are kept as they are, and the
/// search reports itself incomplete.
pub(super) fn patch_pairs(a: &BSplineSurface, b: &BSplineSurface, tol: Tolerance) -> PatchPairs {
    let shapes = [a.patch_net_shape(), b.patch_net_shape()];
    let sizes = shapes.map(|(rows, cols, h)| rows * cols * h);
    let mut search = Search {
        tol,
        shapes,
        pools: sizes.map(Pool::new),
        points: [Vec::new(), Vec::new()],
        pending: Vec::new(),
        leaves: Vec::new(),
        budget: 0,
        complete: true,
    };
    let mut scratch = vec![0.0; sizes[0].max(sizes[1])];
    let mut net = vec![0.0; sizes[0]];
    let mut points = Vec::new();
    let tree = b.patch_boxes();
    a.patch_boxes()
        .overlapping(tree.bounds(), tol.confusion, |i| {
            let rect_a = a.patch_net(i, &mut scratch, &mut net);
            net_points(&net, shapes[0].2, &mut points);
            tree.overlapping(&bounding_box(&points), tol.confusion, |j| {
                let slot_a = search.pools[0].take();
                search.pools[0].get_mut(slot_a).copy_from_slice(&net);
                let slot_b = search.pools[1].take();
                let rect_b = b.patch_net(j, &mut scratch, search.pools[1].get_mut(slot_b));
                search.pending.push(Node {
                    slots: [slot_a, slot_b],
                    rects: [rect_a, rect_b],
                    depth: 0,
                });
            });
        });
    search.budget = SPLITS_PER_PAIR * search.pending.len();
    while let Some(node) = search.pending.pop() {
        search.visit(node);
    }
    PatchPairs {
        leaves: search.leaves,
        complete: search.complete,
    }
}

/// A pair of subpatches awaiting its visit: their nets in `slots` of the
/// two pools, over `rects`, and the halvings from the pair it came from.
struct Node {
    slots: [usize; 2],
    rects: [[f64; 4]; 2],
    depth: u32,
}

struct Search {
    tol: Tolerance,
    shapes: [(usize, usize, usize); 2],
    pools: [Pool; 2],
    points: [Vec<[f64; 3]>; 2],
    pending: Vec<Node>,
    leaves: Vec<Leaf>,
    budget: usize,
    complete: bool,
}

impl Search {
    /// Settles `node` or halves it into two pending pairs.
    fn visit(&mut self, node: Node) {
        let gap = self.tol.confusion;
        for i in 0..2 {
            let h = self.shapes[i].2;
            net_points(self.pools[i].get(node.slots[i]), h, &mut self.points[i]);
        }
        let [pa, pb] = &self.points;
        let boxes = [bounding_box(pa), bounding_box(pb)];
        let normals = [0, 1].map(|i| {
            let (rows, cols, _) = self.shapes[i];
            net_normal(&self.points[i], rows, cols)
        });
        if boxes_apart(&boxes[0], &boxes[1], gap)
            || normals.iter().any(|&n| apart_along(n, pa, pb, gap))
        {
            self.release(&node);
            return;
        }
        let sizes = boxes.map(|b| box_size(&b));
        let flat = (0..2).all(|i| thickness(&self.points[i], normals[i]) <= FLATNESS * sizes[i]);
        let small = sizes.iter().all(|&s| s <= SMALLEST * gap);
        let open = (flat || small) && {
            let [s, r] = [0, 1].map(|i| {
                derivative_cones(
                    self.pools[i].get(node.slots[i]),
                    &self.points[i],
                    self.shapes[i],
                )
            });
            matches!((s, r), (Some(s), Some(r)) if normals_never_parallel(s, r))
        };
        let settled = (open && flat) || small;
        if settled || node.depth >= MAX_DEPTH || self.budget == 0 {
            self.complete &= settled;
            let overlap = std::array::from_fn(|c| {
                if c < 3 {
                    boxes[0][c].max(boxes[1][c]) - gap
                } else {
                    boxes[0][c].min(boxes[1][c]) + gap
                }
            });
            self.leaves.push(Leaf {
                rects: node.rects,
                overlap,
                open,
            });
            self.release(&node);
            return;
        }
        self.budget -= 1;
        self.split(node, usize::from(sizes[1] > sizes[0]));
    }

    /// Halves subpatch `i` of `node` along the direction its net stretches
    /// most, and queues both halves.
    fn split(&mut self, node: Node, i: usize) {
        let (rows, cols, _) = self.shapes[i];
        let along_u = stretches_along_rows(&self.points[i], rows, cols);
        let upper = self.pools[i].halve(node.slots[i], self.shapes[i], along_u);
        let other = 1 - i;
        let copy = self.pools[other].take();
        self.pools[other].copy(node.slots[other], copy);
        let r = node.rects[i];
        let (lower_rect, upper_rect) = if along_u {
            let mid = 0.5 * (r[0] + r[1]);
            ([r[0], mid, r[2], r[3]], [mid, r[1], r[2], r[3]])
        } else {
            let mid = 0.5 * (r[2] + r[3]);
            ([r[0], r[1], r[2], mid], [r[0], r[1], mid, r[3]])
        };
        let depth = node.depth + 1;
        let mut slots = node.slots;
        let mut rects = node.rects;
        (slots[i], slots[other], rects[i]) = (upper, copy, upper_rect);
        self.pending.push(Node {
            slots,
            rects,
            depth,
        });
        let mut rects = node.rects;
        rects[i] = lower_rect;
        self.pending.push(Node {
            slots: node.slots,
            rects,
            depth,
        });
    }

    fn release(&mut self, node: &Node) {
        self.pools[0].release(node.slots[0]);
        self.pools[1].release(node.slots[1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point3D;
    use crate::curves::BSplineCurve3D;
    use crate::intersect::marching::grid_marching;
    use crate::intersect::marching::marched_curves;

    const TOL: Tolerance = Tolerance::DEFAULT;

    /// A bicubic B-spline over `[0, 5]^2`, clamped, with the interior
    /// knots `knots` in both directions and its poles above the Greville
    /// abscissae, so that `S(u, v) = (u, v, z)` when polynomial: heights
    /// `height(x, y)`, weights `weight(i, j)` if given.
    fn graph(
        knots: &[f64],
        height: impl Fn(f64, f64) -> f64,
        weight: Option<&dyn Fn(usize, usize) -> f64>,
    ) -> BSplineSurface {
        let flat: Vec<f64> = [0.0; 4]
            .iter()
            .chain(knots)
            .chain(&[5.0; 4])
            .copied()
            .collect();
        let n = knots.len() + 4;
        let greville: Vec<f64> = (0..n)
            .map(|i| flat[i + 1..i + 4].iter().sum::<f64>() / 3.0)
            .collect();
        let poles = greville
            .iter()
            .map(|&x| {
                greville
                    .iter()
                    .map(|&y| Point3D::new(x, y, height(x, y)))
                    .collect()
            })
            .collect();
        let breaks: Vec<f64> = [0.0].iter().chain(knots).chain(&[5.0]).copied().collect();
        let mut mults = vec![1; breaks.len()];
        (mults[0], mults[breaks.len() - 1]) = (4, 4);
        match weight {
            None => BSplineSurface::new(
                3,
                3,
                poles,
                breaks.clone(),
                mults.clone(),
                breaks,
                mults,
                false,
                false,
            ),
            Some(w) => BSplineSurface::new_rational(
                3,
                3,
                poles,
                (0..n).map(|i| (0..n).map(|j| w(i, j)).collect()).collect(),
                breaks.clone(),
                mults.clone(),
                breaks,
                mults,
                false,
                false,
            ),
        }
        .unwrap()
    }

    const THIRDS: [f64; 2] = [5.0 / 3.0, 10.0 / 3.0];

    fn wavy() -> BSplineSurface {
        graph(&THIRDS, |x, y| (0.25 * x * y).sin(), None)
    }

    /// Surfaces crossing [`wavy`] in several curves, open and closed.
    fn crossing_wavy() -> Vec<BSplineSurface> {
        vec![
            graph(
                &THIRDS,
                |x, y| 0.5 * (0.8 * x).cos() * (0.7 * y).cos(),
                None,
            ),
            graph(&THIRDS, |_, _| 0.3, None),
            graph(&THIRDS, |x, y| 0.1 * (x - y), None),
            graph(&THIRDS, |x, y| 0.6 * x.sin() * (0.6 * y).cos() - 0.1, None),
        ]
    }

    fn samples(curve: &BSplineCurve3D, n: usize) -> impl Iterator<Item = Point3D> + '_ {
        let (t0, t1) = curve.bounds();
        (0..=n).map(move |k| curve.eval_point(t0 + (t1 - t0) * k as f64 / n as f64))
    }

    fn is_closed(curve: &BSplineCurve3D) -> bool {
        let (t0, t1) = curve.bounds();
        curve.eval_point(t0).distance(curve.eval_point(t1)) < 1e-9
    }

    fn distance_to(curves: &[BSplineCurve3D], p: Point3D) -> f64 {
        curves
            .iter()
            .map(|c| c.project_point(p, TOL).distance)
            .fold(f64::INFINITY, f64::min)
    }

    /// Asserts every curve lies on both surfaces, and that no curve runs
    /// along another (each loop is traced once).
    #[track_caller]
    fn assert_on_both_once(curves: &[BSplineCurve3D], a: &BSplineSurface, b: &BSplineSurface) {
        for (i, curve) in curves.iter().enumerate() {
            for p in samples(curve, 40) {
                assert!(a.project_point(p, TOL).distance < 1e-3, "{p:?}");
                assert!(b.project_point(p, TOL).distance < 1e-3, "{p:?}");
            }
            for (j, other) in curves.iter().enumerate() {
                let near = samples(curve, 20)
                    .filter(|&p| other.project_point(p, TOL).distance < 1e-3)
                    .count();
                assert!(i == j || near <= 2, "curves {i} and {j} overlap");
            }
        }
    }

    /// Asserts that `curves` pass near every sign change of the height
    /// difference of `a` and `b` (graphs over the same parameters) along
    /// the lines of a fine grid.
    #[track_caller]
    fn assert_every_crossing(curves: &[BSplineCurve3D], a: &BSplineSurface, b: &BSplineSurface) {
        let m = 120;
        let at = |k: usize| 5.0 * k as f64 / m as f64;
        let d = |u: f64, v: f64| a.eval_point(u, v).z - b.eval_point(u, v).z;
        let mut crossings = 0;
        for i in 0..m {
            for j in 0..=m {
                for (u0, v0, u1, v1) in [
                    (at(i), at(j), at(i + 1), at(j)),
                    (at(j), at(i), at(j), at(i + 1)),
                ] {
                    let (d0, d1) = (d(u0, v0), d(u1, v1));
                    if d0 * d1 < 0.0 {
                        crossings += 1;
                        let f = d0 / (d0 - d1);
                        let p = a.eval_point(u0 + f * (u1 - u0), v0 + f * (v1 - v0));
                        assert!(distance_to(curves, p) < 0.01, "{p:?} missed");
                    }
                }
            }
        }
        assert!(crossings > 0);
    }

    #[test]
    fn test_bspline_pairs_trace_every_curve_once() {
        let a = wavy();
        for b in crossing_wavy() {
            let curves = marched_curves(&a, &b, TOL);
            assert!(patch_pairs(&a, &b, TOL).complete);
            assert_on_both_once(&curves, &a, &b);
            assert_every_crossing(&curves, &a, &b);
            let grid = grid_marching(&a, &b, TOL);
            assert_eq!(curves.len(), grid.len());
            for curve in &grid {
                assert!(samples(curve, 20).all(|p| distance_to(&curves, p) < 1e-3));
            }
        }
    }

    #[test]
    fn test_small_loop_through_a_nearly_flat_surface() {
        // A bump of width 0.1 about (2.6, 2.6), within one cell of the
        // seed grid (lines spaced 5 / 24), pokes 0.04 through a gently
        // tilted sheet, which also crosses the bump's flat surroundings
        // along x = 2.1.
        let knots = [1.0, 2.0, 2.4, 2.55, 2.575, 2.6, 2.625, 2.65, 2.8, 3.5, 4.25];
        let at_bump = |x: f64, y: f64| (x - 2.6).abs() < 1e-9 && (y - 2.6).abs() < 1e-9;
        let bump = graph(&knots, |x, y| if at_bump(x, y) { 0.2 } else { 0.0 }, None);
        let sheet = graph(&THIRDS, |x, y| 0.1 * (x - 2.1) + 0.005 * y.sin(), None);
        let spacing = 5.0 / 24.0;
        for (a, b) in [(&bump, &sheet), (&sheet, &bump)] {
            let curves = marched_curves(a, b, TOL);
            assert_on_both_once(&curves, a, b);
            assert_eq!(curves.len(), 2, "{curves:?}");
            let loops: Vec<_> = curves.iter().filter(|c| is_closed(c)).collect();
            assert_eq!(loops.len(), 1);
            let points: Vec<Point3D> = samples(loops[0], 40).collect();
            let diameter = points
                .iter()
                .flat_map(|p| points.iter().map(move |q| p.distance(*q)))
                .fold(0.0, f64::max);
            assert!(diameter < 0.5 * spacing, "{diameter}");
            assert!(
                points
                    .iter()
                    .all(|p| p.distance(Point3D::new(2.6, 2.6, p.z)) < 0.05)
            );
            // The grid seeds trace the line but miss the loop.
            let grid = grid_marching(a, b, TOL);
            assert_eq!(grid.len(), 1);
            assert!(!is_closed(&grid[0]));
            assert!(points.iter().all(|&p| distance_to(&grid, p) > 0.4));
        }
    }

    #[test]
    fn test_open_curves_end_on_the_domain_bounds() {
        // A sheet crossing a flat surface along x = 2.5 - 0.5 sin(0.4 y),
        // from the edge y = 0 to the edge y = 5.
        let flat = graph(&THIRDS, |_, _| 0.0, None);
        let sheet = graph(
            &THIRDS,
            |x, y| 0.4 * (x - 2.5) + 0.2 * (0.4 * y).sin(),
            None,
        );
        for curves in [
            marched_curves(&sheet, &flat, TOL),
            grid_marching(&sheet, &flat, TOL),
        ] {
            assert_on_both_once(&curves, &sheet, &flat);
            assert_eq!(curves.len(), 1);
            let (t0, t1) = curves[0].bounds();
            let mut ends = [curves[0].eval_point(t0).y, curves[0].eval_point(t1).y];
            ends.sort_by(f64::total_cmp);
            assert!(
                ends[0].abs() < 1e-7 && (ends[1] - 5.0).abs() < 1e-7,
                "{ends:?}"
            );
        }
    }

    #[test]
    fn test_rational_pairs_match_the_grid_seeds() {
        let weight = |i: usize, j: usize| 0.6 + 0.2 * ((3 * i + 5 * j) % 4) as f64;
        let a = graph(&THIRDS, |x, y| (0.25 * x * y).sin(), Some(&weight));
        for b in crossing_wavy().into_iter().take(2) {
            let curves = marched_curves(&a, &b, TOL);
            assert_on_both_once(&curves, &a, &b);
            let grid = grid_marching(&a, &b, TOL);
            assert!(!grid.is_empty());
            assert_eq!(curves.len(), grid.len());
            for curve in &grid {
                assert!(samples(curve, 20).all(|p| distance_to(&curves, p) < 1e-3));
            }
        }
    }

    /// A periodic quadratic tube about the z axis, radius about 2, over
    /// `z` in `[-1, 3]`.
    fn tube() -> BSplineSurface {
        let poles = (0..6)
            .map(|k| {
                let angle = k as f64 * std::f64::consts::TAU / 6.0;
                let (c, s) = (angle.cos(), angle.sin());
                vec![
                    Point3D::new(2.0 * c, 2.0 * s, -1.0),
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
    fn test_periodic_surface_gives_one_loop_across_its_seam() {
        let tube = tube();
        // A tilted sheet over `[-2.5, 2.5]^2`, through the tube.
        let shift = crate::Transform::translation(crate::Vector3D::new(-2.5, -2.5, 0.0));
        let sheet = graph(&THIRDS, |x, y| 0.2 * x + 0.1 * (y - 2.5), None).transformed(shift);
        let curves = marched_curves(&tube, &sheet, TOL);
        assert_on_both_once(&curves, &tube, &sheet);
        assert_eq!(curves.len(), 1);
        assert!(is_closed(&curves[0]));
        let grid = grid_marching(&tube, &sheet, TOL);
        assert_eq!(grid.len(), 1);
        assert!(samples(&grid[0], 20).all(|p| distance_to(&curves, p) < 1e-3));
    }

    #[test]
    fn test_overlapping_surfaces_exhaust_the_search() {
        let a = wavy();
        assert!(!patch_pairs(&a, &a, TOL).complete);
        // The grid seeds take over; the surfaces are tangent everywhere,
        // so nothing traces.
        assert!(marched_curves(&a, &a, TOL).is_empty());
    }
}
