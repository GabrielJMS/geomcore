//! Bounding-box hierarchy for nearest-first search over the polynomial
//! pieces (knot spans, knot-span patches) of a B-spline, and the lazy cache
//! that holds it.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;
use std::sync::OnceLock;

use crate::Point3D;

/// Axis-aligned box `[min_x, min_y, min_z, max_x, max_y, max_z]`.
pub(crate) type Aabb = [f64; 6];

/// Distance from `p` to the box `b` (zero inside).
pub(crate) fn box_distance(p: Point3D, b: &Aabb) -> f64 {
    let gap = |x: f64, lo: f64, hi: f64| (lo - x).max(x - hi).max(0.0);
    let (dx, dy, dz) = (
        gap(p.x, b[0], b[3]),
        gap(p.y, b[1], b[4]),
        gap(p.z, b[2], b[5]),
    );
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Binary hierarchy over a `rows x cols` grid of leaf boxes (leaf `k` at
/// row `k / cols`, column `k % cols`), built by halving the longer index
/// range, each node's box the union of its children's. The root is
/// `nodes[0]`.
#[derive(Clone)]
pub(crate) struct BoxTree {
    nodes: Vec<Node>,
}

#[derive(Clone)]
struct Node {
    bbox: Aabb,
    kind: Kind,
}

#[derive(Clone, Copy)]
enum Kind {
    Leaf(usize),
    Split(usize, usize),
}

impl BoxTree {
    /// The hierarchy over `leaves` (row-major, `cols` per row; non-empty).
    pub(crate) fn new(leaves: &[Aabb], cols: usize) -> Self {
        debug_assert!(!leaves.is_empty() && leaves.len().is_multiple_of(cols));
        let mut nodes = Vec::with_capacity(2 * leaves.len());
        build(
            leaves,
            cols,
            (0, leaves.len() / cols),
            (0, cols),
            &mut nodes,
        );
        BoxTree { nodes }
    }

    /// The box around every leaf.
    pub(crate) fn bounds(&self) -> &Aabb {
        &self.nodes[0].bbox
    }

    /// Calls `visit` on leaves by increasing box distance from `point`
    /// while that distance is below the best distance so far; `visit`
    /// returns the best distance after processing its leaf. Since every
    /// box contains its piece of the geometry, the leaves skipped cannot
    /// hold anything closer.
    pub(crate) fn nearest_first(&self, point: Point3D, mut visit: impl FnMut(usize) -> f64) {
        let mut best = f64::INFINITY;
        let mut heap = BinaryHeap::from([Pending {
            lower: box_distance(point, &self.nodes[0].bbox),
            node: 0,
        }]);
        while let Some(Pending { lower, node }) = heap.pop() {
            if lower >= best {
                break;
            }
            match self.nodes[node].kind {
                Kind::Split(left, right) => {
                    for child in [left, right] {
                        heap.push(Pending {
                            lower: box_distance(point, &self.nodes[child].bbox),
                            node: child,
                        });
                    }
                }
                Kind::Leaf(k) => best = visit(k),
            }
        }
    }

    /// Calls `visit` on every leaf whose box comes within `gap` of the box
    /// `b` (along each axis), in tree order. Since every box contains its
    /// piece of the geometry, the leaves skipped hold nothing within `gap`
    /// of what `b` contains.
    pub(crate) fn overlapping(&self, b: &Aabb, gap: f64, mut visit: impl FnMut(usize)) {
        let mut stack = vec![0];
        while let Some(node) = stack.pop() {
            if boxes_apart(&self.nodes[node].bbox, b, gap) {
                continue;
            }
            match self.nodes[node].kind {
                Kind::Split(left, right) => stack.extend([right, left]),
                Kind::Leaf(k) => visit(k),
            }
        }
    }
}

/// Whether the boxes `a` and `b` are more than `gap` apart along some axis.
pub(crate) fn boxes_apart(a: &Aabb, b: &Aabb, gap: f64) -> bool {
    (0..3).any(|c| a[c] > b[c + 3] + gap || b[c] > a[c + 3] + gap)
}

/// Appends the subtree over the leaf index ranges `rows` x `cols_range` to
/// `nodes`, returning its root index.
fn build(
    leaves: &[Aabb],
    cols: usize,
    rows: (usize, usize),
    cols_range: (usize, usize),
    nodes: &mut Vec<Node>,
) -> usize {
    let at = nodes.len();
    if rows.1 - rows.0 == 1 && cols_range.1 - cols_range.0 == 1 {
        let k = rows.0 * cols + cols_range.0;
        nodes.push(Node {
            bbox: leaves[k],
            kind: Kind::Leaf(k),
        });
        return at;
    }
    nodes.push(Node {
        bbox: [0.0; 6],
        kind: Kind::Split(0, 0),
    });
    let (first, second) = if rows.1 - rows.0 >= cols_range.1 - cols_range.0 {
        let mid = (rows.0 + rows.1) / 2;
        (((rows.0, mid), cols_range), ((mid, rows.1), cols_range))
    } else {
        let mid = (cols_range.0 + cols_range.1) / 2;
        ((rows, (cols_range.0, mid)), (rows, (mid, cols_range.1)))
    };
    let left = build(leaves, cols, first.0, first.1, nodes);
    let right = build(leaves, cols, second.0, second.1, nodes);
    let (a, b) = (nodes[left].bbox, nodes[right].bbox);
    nodes[at] = Node {
        bbox: std::array::from_fn(|i| {
            if i < 3 {
                a[i].min(b[i])
            } else {
                a[i].max(b[i])
            }
        }),
        kind: Kind::Split(left, right),
    };
    at
}

/// A node awaiting its visit, ordered so that a [`BinaryHeap`] pops the
/// smallest box distance first.
struct Pending {
    lower: f64,
    node: usize,
}

impl PartialEq for Pending {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Pending {}

impl PartialOrd for Pending {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Pending {
    fn cmp(&self, other: &Self) -> Ordering {
        other.lower.total_cmp(&self.lower)
    }
}

/// Data derived from a geometry's fields, computed on first use. It takes
/// no part in equality and prints opaquely, so the owning type can keep
/// deriving `PartialEq` and `Debug`.
#[derive(Clone)]
pub(crate) struct Lazy<T>(OnceLock<Box<T>>);

impl<T> Lazy<T> {
    pub(crate) fn get_or_init(&self, init: impl FnOnce() -> T) -> &T {
        self.0.get_or_init(|| Box::new(init()))
    }
}

impl<T> Default for Lazy<T> {
    fn default() -> Self {
        Lazy(OnceLock::new())
    }
}

impl<T> PartialEq for Lazy<T> {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl<T> fmt::Debug for Lazy<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Lazy")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nearest_first_visits_by_box_distance_and_prunes() {
        // A 1 x 4 row of unit boxes along x.
        let leaves: Vec<Aabb> = (0..4)
            .map(|i| {
                let x = 2.0 * i as f64;
                [x, 0.0, 0.0, x + 1.0, 1.0, 1.0]
            })
            .collect();
        let tree = BoxTree::new(&leaves, 4);
        let mut seen = Vec::new();
        // The query sits right of the last box; each visited leaf reports
        // its own box distance as the best, so only the nearest is visited.
        tree.nearest_first(Point3D::new(10.0, 0.5, 0.5), |k| {
            seen.push(k);
            box_distance(Point3D::new(10.0, 0.5, 0.5), &leaves[k])
        });
        assert_eq!(seen, vec![3]);
        // Without pruning every leaf comes, nearest first.
        let mut all = Vec::new();
        tree.nearest_first(Point3D::new(10.0, 0.5, 0.5), |k| {
            all.push(k);
            f64::INFINITY
        });
        assert_eq!(all, vec![3, 2, 1, 0]);
    }

    #[test]
    fn test_overlapping_visits_only_boxes_within_the_gap() {
        // A 2 x 3 grid of unit boxes, a step of 2 apart.
        let leaves: Vec<Aabb> = (0..6)
            .map(|k| {
                let (x, y) = (2.0 * (k / 3) as f64, 2.0 * (k % 3) as f64);
                [x, y, 0.0, x + 1.0, y + 1.0, 1.0]
            })
            .collect();
        let tree = BoxTree::new(&leaves, 3);
        let query = [1.2, 1.2, 0.5, 1.8, 2.5, 0.6];
        let mut seen = Vec::new();
        tree.overlapping(&query, 0.0, |k| seen.push(k));
        assert!(seen.is_empty());
        tree.overlapping(&query, 0.25, |k| seen.push(k));
        seen.sort();
        assert_eq!(seen, vec![0, 1, 3, 4]);
        let mut all = Vec::new();
        tree.overlapping(tree.bounds(), 0.0, |k| all.push(k));
        all.sort();
        assert_eq!(all, (0..6).collect::<Vec<_>>());
    }

    #[test]
    fn test_box_distance() {
        let b = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        assert_eq!(box_distance(Point3D::new(0.5, 0.5, 0.5), &b), 0.0);
        assert_eq!(box_distance(Point3D::new(4.0, 5.0, 0.5), &b), 5.0);
    }
}
