//! Knot removal within a deviation bound (after The NURBS Book, A5.8):
//! each removal inverts the insertion of one copy of a knot, solving the
//! new poles from both ends of the run it affects, and the equation left
//! over measures how far the shape moves.

use super::{Edited, Kept, Lanes, Pole, Source, Weighted, from_homogeneous};
use crate::Point3D;
use crate::curve_math::bspline::{KnotVector, MAX_DEGREE};
use crate::curves::KnotRemovalError;

/// A direction with knots removed, its poles and weights laid out as the
/// input's, and how many copies were removed.
pub(crate) struct Removed {
    pub(crate) knots: KnotVector,
    pub(crate) points: Vec<Point3D>,
    pub(crate) weights: Option<Vec<f64>>,
    pub(crate) count: usize,
}

/// Checks the arguments of [`remove_knots`]: a finite non-negative
/// `max_deviation`, and knots that are finite, strictly inside the bounds
/// of a non-periodic direction (either bound naming the seam of a periodic
/// one) and equal to one of its knots.
pub(crate) fn check_removal(
    direction: &KnotVector,
    remove: &[f64],
    max_deviation: f64,
) -> Result<(), KnotRemovalError> {
    if !(max_deviation.is_finite() && max_deviation >= 0.0) {
        return Err(KnotRemovalError::InvalidDeviation);
    }
    if remove.iter().any(|x| !x.is_finite()) {
        return Err(KnotRemovalError::NonFiniteKnot);
    }
    let (first, last) = direction.bounds();
    let knots = direction.knots();
    for &x in remove {
        let inside = if direction.is_periodic() {
            first <= x && x <= last
        } else {
            first < x && x < last
        };
        if !inside {
            return Err(KnotRemovalError::KnotOutOfRange);
        }
        let i = knots.partition_point(|&k| k < x);
        if i == knots.len() || knots[i] != x {
            return Err(KnotRemovalError::KnotNotFound);
        }
    }
    Ok(())
}

/// Removes the knots `remove` from a direction one copy at a time, in
/// order, a value repeated as many times as it appears (copies beyond its
/// multiplicity are skipped), keeping every knot whose removal would let
/// the shape deviate from the original by more than `max_deviation` at
/// some parameter; `None` if no copy was removed.
///
/// The new poles of a removal are solved from both ends of the affected
/// run, and the one equation (or the one pole computed twice) left over
/// measures the deviation, an upper bound because the basis functions are
/// non-negative and sum to one. A pole computed twice takes the blend of
/// its two values that minimizes that bound. A rational direction is
/// solved in homogeneous coordinates and its bound carried over to the
/// points (see [`Deviation::bound`]). Each bound adds to the bounds of the
/// knot intervals the deviation reaches, so the deviation from the
/// original stays within `max_deviation` however many knots are removed.
///
/// A knot is removed only if every lane stays within `max_deviation`. On a
/// periodic direction the seam keeps at least one copy, so the bounds
/// stay. The arguments must pass [`check_removal`].
pub(crate) fn remove_knots(
    direction: &KnotVector,
    points: &[Point3D],
    weights: Option<&[f64]>,
    lanes: Lanes,
    remove: &[f64],
    max_deviation: f64,
) -> Option<Removed> {
    match weights {
        None => {
            let removal = run(direction, points.to_vec(), lanes, remove, max_deviation)?;
            Some(Removed {
                knots: removal.knots,
                points: removal.poles,
                weights: None,
                count: removal.count,
            })
        }
        Some(weights) => {
            let source = Weighted { points, weights };
            let mut poles = Vec::with_capacity(points.len());
            source.extend_to(0..points.len(), &mut poles);
            let removal = run(direction, poles, lanes, remove, max_deviation)?;
            let edited = Edited {
                knots: removal.knots,
                poles: removal.poles,
                kept: removal.kept,
                start: 0,
            };
            let (points, weights) = from_homogeneous(&edited, points, weights, lanes);
            Some(Removed {
                knots: edited.knots,
                points,
                weights: Some(weights),
                count: removal.count,
            })
        }
    }
}

/// A pole whose removal deviation can be bounded.
trait Deviation: Pole {
    /// A bound on how far the points of the shape move when each lane's
    /// pole moves by its `diffs` times `coefficient` (a basis function's
    /// maximum), or `None` if it exceeds `max` or is not finite: `centre`
    /// is a pole near the change, and `reach` every pole of every lane
    /// whose function the change overlaps.
    fn bound(
        diffs: &[Self],
        coefficient: f64,
        centre: Self,
        reach: impl Iterator<Item = Self>,
        max: f64,
    ) -> Option<f64>;

    /// Whether the pole can weight a shape: a positive weight.
    fn is_valid(self) -> bool;
}

impl Deviation for Point3D {
    #[inline]
    fn bound(
        diffs: &[Self],
        coefficient: f64,
        _: Self,
        _: impl Iterator<Item = Self>,
        _: f64,
    ) -> Option<f64> {
        let squared = diffs
            .iter()
            .map(|d| d.x * d.x + d.y * d.y + d.z * d.z)
            .fold(0.0, f64::max);
        squared.is_finite().then(|| coefficient * squared.sqrt())
    }

    #[inline]
    fn is_valid(self) -> bool {
        true
    }
}

impl Deviation for [f64; 4] {
    /// Measured about a point `c`, a homogeneous deviation `e` of the shape
    /// `(w (S - c), w)` moves its points by at most
    /// `e sqrt(1 + |S - c|^2) / (w - e)`, unbounded if `e` reaches `w`.
    /// Where the deviation is non-zero, `S` blends the `reach` poles and `w`
    /// their weights: `|S - c|` is at most the largest distance from `c` to
    /// a pole, and `w` at least the smallest weight. On a surface the
    /// deviation blends those of its lanes, so every lane is measured about
    /// the same `c`, the point of `centre`. The bound neither grows with the
    /// distance to the origin nor depends on poles far away.
    fn bound(
        diffs: &[Self],
        coefficient: f64,
        centre: Self,
        reach: impl Iterator<Item = Self>,
        max: f64,
    ) -> Option<f64> {
        let weight = centre[3];
        let c = [centre[0] / weight, centre[1] / weight, centre[2] / weight];
        let length_squared = |[x, y, z, w]: [f64; 4]| {
            let (dx, dy, dz) = (x - c[0] * w, y - c[1] * w, z - c[2] * w);
            dx * dx + dy * dy + dz * dz
        };
        let squared = diffs
            .iter()
            .map(|&d| length_squared(d) + d[3] * d[3])
            .fold(0.0, f64::max);
        let e = coefficient * squared.sqrt();
        // The weight of `centre` is at least the smallest, so this bounds the
        // bound from below.
        if !(e < weight && e <= max * (weight - e)) {
            return None;
        }
        let (mut min_weight, mut radius_squared) = (weight, 0.0_f64);
        for pole in reach {
            let w = pole[3];
            min_weight = min_weight.min(w);
            radius_squared = radius_squared.max(length_squared(pole) / (w * w));
        }
        let bound = if e < min_weight {
            e * (1.0 + radius_squared).sqrt() / (min_weight - e)
        } else {
            f64::INFINITY
        };
        (bound <= max).then_some(bound)
    }

    #[inline]
    fn is_valid(self) -> bool {
        self[3] > 0.0
    }
}

/// What [`run`] returns: the new direction and poles, the poles at either
/// end of each lane left as they were, and the copies removed.
struct Run<P> {
    knots: KnotVector,
    poles: Vec<P>,
    kept: Kept,
    count: usize,
}

/// [`remove_knots`] on the poles, converted if rational.
fn run<P: Deviation>(
    direction: &KnotVector,
    poles: Vec<P>,
    lanes: Lanes,
    remove: &[f64],
    max: f64,
) -> Option<Run<P>> {
    let (rows, width) = if lanes.consecutive {
        (lanes.count, 1)
    } else {
        (1, lanes.count)
    };
    let (first, last) = direction.bounds();
    let flat = direction.flat().to_vec();
    let mut removal = Removal {
        degree: direction.degree(),
        periodic: direction.is_periodic(),
        rows,
        width,
        first,
        last,
        seam: flat.iter().take_while(|&&k| k == flat[0]).count(),
        n: direction.pole_count(),
        flat,
        poles,
        bounds: Vec::new(),
        scratch: vec![P::default(); rows * (direction.degree() + 1) * width],
        diffs: vec![P::default(); lanes.count],
        spare: Vec::new(),
        changed: None,
    };
    let mut count = 0;
    let mut failed = f64::NAN;
    for (i, &u) in remove.iter().enumerate() {
        // A failed knot fails again until something else changes.
        if u == failed {
            continue;
        }
        match removal.solve(u, max) {
            Some(solved) => {
                removal.commit(&solved, i + 1 < remove.len());
                count += 1;
                failed = f64::NAN;
            }
            None => failed = u,
        }
    }
    // Some removal changed some slots.
    let changed = removal.changed?;
    let kept = Kept {
        leading: changed.start,
        trailing: removal.n - changed.end,
    };
    let knots = KnotVector::from_flat(removal.degree, removal.flat, removal.periodic);
    Some(Run {
        knots,
        poles: removal.poles,
        kept,
        count,
    })
}

/// The state of [`run`]: the direction as removed so far. Its poles sit in
/// `rows` rows of `n` slots, each slot holding `width` poles, one of each
/// lane.
struct Removal<P> {
    degree: usize,
    periodic: bool,
    rows: usize,
    width: usize,
    first: f64,
    last: f64,
    /// Seam multiplicity of a periodic direction.
    seam: usize,
    /// Slots per row: poles per lane.
    n: usize,
    flat: Vec<f64>,
    poles: Vec<P>,
    /// Bound on the deviation from the original over each knot interval:
    /// interval `k` runs from flat knot `k` to `k + 1` (tiled knots on a
    /// periodic direction, one period of them). Empty, meaning zero, until
    /// a removal is followed by another attempt.
    bounds: Vec<f64>,
    /// The run of new poles a solved removal writes: per row, `degree - s
    /// + 2` slots, the kept one before the new ones and the one after.
    scratch: Vec<P>,
    /// Each lane's deviation.
    diffs: Vec<P>,
    /// Spare pole buffer for rebuilding a periodic direction.
    spare: Vec<P>,
    /// The slots changed so far, the others being the input's poles; `None`
    /// until a removal.
    changed: Option<std::ops::Range<usize>>,
}

/// A removal [`Removal::solve`] found within bounds, ready to commit.
struct Solved {
    /// Last copy of the knot, its first copy and its multiplicity, in
    /// (tiled) flat knot indices.
    r: isize,
    c: usize,
    s: usize,
    /// Point deviation bound.
    bound: f64,
    /// First knot interval the deviation reaches, and how many it reaches.
    start: isize,
    count: usize,
}

impl<P: Deviation> Removal<P> {
    /// Flat knot `k`, tiled endlessly with the period on a periodic
    /// direction.
    #[inline]
    fn knot(&self, k: isize) -> f64 {
        if self.periodic {
            let n = self.n as isize;
            self.flat[k.rem_euclid(n) as usize] + (self.last - self.first) * k.div_euclid(n) as f64
        } else {
            self.flat[k as usize]
        }
    }

    /// Slot of the pole weighting the basis function that starts at (tiled)
    /// flat knot `f`.
    #[inline]
    fn slot(&self, f: isize) -> usize {
        if self.periodic {
            let shift = self.degree as isize + 1 - self.seam as isize;
            (f + shift).rem_euclid(self.n as isize) as usize
        } else {
            f as usize
        }
    }

    /// The poles of row `row` weighting the function starting at flat knot
    /// `f`, one per lane.
    #[inline]
    fn poles_at(&self, row: usize, f: isize) -> &[P] {
        &self.poles[(row * self.n + self.slot(f)) * self.width..][..self.width]
    }

    /// Solves one removal of the knot `u` into the scratch run, or `None` if
    /// `u` has no copy left to remove, or its removal would move the shape
    /// further than `max` from the original or make a weight non-positive.
    ///
    /// The removal of the copy `r` of multiplicity `s` replaces the poles
    /// of the functions `r - degree ..= r - s` by `degree - s` new ones,
    /// written to the scratch run per row after the unchanged slot before
    /// them and followed by the unchanged one after them.
    fn solve(&mut self, u: f64, max: f64) -> Option<Solved> {
        let (p, n, width) = (self.degree, self.n, self.width);
        // The seam, as its copies at the start of the period, keeps one copy.
        let (u, c, s, kept) = if self.periodic && (u == self.first || u == self.last) {
            (self.first, 0, self.seam, 1)
        } else {
            let c = self.flat.partition_point(|&k| k < u);
            let s = self.flat[c..].iter().take_while(|&&k| k == u).count();
            (u, c, s, 0)
        };
        // Removing one copy of a periodic knot removes it from every
        // period, independently only if each period has room for the
        // affected poles and the unchanged one after them.
        if s <= kept || (self.periodic && n < p - s + 2) {
            return None;
        }
        let w = p - s;
        let r = (c + s - 1) as isize;
        let (a, b) = (r - p as isize, r - s as isize);
        let mut alpha = [0.0; MAX_DEGREE + 1];
        for (k, i) in (a..=b).enumerate() {
            let left = self.knot(i);
            alpha[k] = (u - left) / (self.knot(i + p as isize + 1) - left);
        }
        // Slot `j` of a row's run is the new pole of the function `a - 1 + j`.
        let run_slots = w + 2;
        let steps = w / 2;
        let lo = steps;
        let odd = w % 2 == 1;
        // The deviation blends the basis functions of the equations left
        // over: one, or two when a pole is solved twice.
        let start = a + lo as isize;
        let count = p + 1 + usize::from(odd);
        let (from, to) = (start - p as isize, start + count as isize - 1);
        let reach = if self.periodic {
            from..=to
        } else {
            from.max(0)..=to.min(n as isize - 1)
        };
        let (al, ah) = (alpha[lo], alpha[lo + 1]);
        let lambda = al / (al + 1.0 - ah);
        let coefficient = if odd {
            al * (1.0 - ah) / (al + 1.0 - ah)
        } else {
            1.0
        };
        let (left_coefficient, right_coefficient) = (1.0 / al, 1.0 / (1.0 - ah));
        let mut scratch = std::mem::take(&mut self.scratch);
        let mut diffs = std::mem::take(&mut self.diffs);
        for row in 0..self.rows {
            let run = &mut scratch[row * run_slots * width..][..run_slots * width];
            run[..width].copy_from_slice(self.poles_at(row, a - 1));
            run[(w + 1) * width..].copy_from_slice(self.poles_at(row, b + 1));
            for k in 0..steps {
                let (i, j) = (k, w - k);
                let (x, from) = (alpha[i], self.poles_at(row, a + i as isize));
                let (y, inverse) = (1.0 - x, 1.0 / x);
                let (prev, out) = run[i * width..(i + 2) * width].split_at_mut(width);
                for ((out, &prev), &from) in out.iter_mut().zip(&*prev).zip(from) {
                    *out = from.combine(inverse, prev, -y * inverse);
                }
                let (x, from) = (alpha[j], self.poles_at(row, a + j as isize));
                let inverse = 1.0 / (1.0 - x);
                let (out, next) = run[j * width..(j + 2) * width].split_at_mut(width);
                for ((out, &next), &from) in out.iter_mut().zip(&*next).zip(from) {
                    *out = from.combine(inverse, next, -x * inverse);
                }
            }
            let (low, high) = (self.poles_at(row, start), self.poles_at(row, start + 1));
            let row_diffs = &mut diffs[row * width..][..width];
            for lane in 0..width {
                let prev = run[lo * width + lane];
                row_diffs[lane] = if odd {
                    let next = run[(lo + 2) * width + lane];
                    let left =
                        low[lane].combine(left_coefficient, prev, -(1.0 - al) * left_coefficient);
                    let right =
                        high[lane].combine(right_coefficient, next, -ah * right_coefficient);
                    run[(lo + 1) * width + lane] = left.combine(lambda, right, 1.0 - lambda);
                    left.combine(1.0, right, -1.0)
                } else {
                    let mid = run[(lo + 1) * width + lane];
                    low[lane].combine(1.0, mid.combine(al, prev, 1.0 - al), -1.0)
                };
            }
            if !run[width..(w + 1) * width]
                .iter()
                .all(|&pole| pole.is_valid())
            {
                self.scratch = scratch;
                self.diffs = diffs;
                return None;
            }
        }
        let this = &*self;
        let reach_poles = reach.flat_map(|f| {
            (0..this.rows).flat_map(move |row| this.poles_at(row, f).iter().copied())
        });
        let bound = P::bound(
            &diffs,
            coefficient,
            this.poles_at(0, start)[0],
            reach_poles,
            max,
        );
        self.scratch = scratch;
        self.diffs = diffs;
        let bound = bound?;
        let fits = if self.bounds.is_empty() {
            bound <= max
        } else {
            self.intervals(start, count)
                .all(|k| self.bounds[k] + bound <= max)
        };
        fits.then_some(Solved {
            r,
            c,
            s,
            bound,
            start,
            count,
        })
    }

    /// The `count` knot intervals from `start` on, wrapped on a periodic
    /// direction.
    fn intervals(&self, start: isize, count: usize) -> impl Iterator<Item = usize> + use<P> {
        let (modulus, start, count) = if self.periodic {
            let n = self.n;
            (n, start.rem_euclid(n as isize) as usize, count.min(n))
        } else {
            (usize::MAX, start as usize, count)
        };
        (start..start + count).map(move |k| k % modulus)
    }

    /// Applies a removal [`Removal::solve`] wrote to the scratch run; `more`
    /// if another attempt follows, which needs the deviation bounds.
    fn commit(&mut self, solved: &Solved, more: bool) {
        let Solved {
            r,
            c,
            s,
            bound,
            start,
            count,
        } = *solved;
        let (p, width, n) = (self.degree, self.width, self.n);
        let w = p - s;
        if more && self.bounds.is_empty() {
            let intervals = if self.periodic {
                n
            } else {
                self.flat.len() - 1
            };
            self.bounds = vec![0.0; intervals];
        }
        if !self.bounds.is_empty() {
            for k in self.intervals(start, count) {
                self.bounds[k] += bound;
            }
            let before = if self.periodic {
                (c + n - 1) % n
            } else {
                c - 1
            };
            self.bounds[before] = self.bounds[before].max(self.bounds[c]);
            self.bounds.remove(c);
        }
        if self.periodic {
            self.rebuild_periodic(r, c, s);
            self.changed = Some(0..n - 1);
        } else {
            let a = (r - p as isize) as usize;
            let run_slots = w + 2;
            for row in 0..self.rows {
                let (from, to) = (row * n * width, row * (n - 1) * width);
                if from != to {
                    self.poles.copy_within(from..from + a * width, to);
                }
                self.poles[to + a * width..][..w * width]
                    .copy_from_slice(&self.scratch[(row * run_slots + 1) * width..][..w * width]);
                self.poles.copy_within(
                    from + (a + w + 1) * width..from + n * width,
                    to + (a + w) * width,
                );
            }
            self.poles.truncate(self.rows * (n - 1) * width);
            self.changed = Some(match self.changed.take() {
                None => a..a + w,
                Some(changed) => {
                    // The slots after the old run move down by one.
                    let end = if changed.end > a {
                        changed.end - 1
                    } else {
                        changed.end
                    };
                    changed.start.min(a)..end.max(a + w)
                }
            });
        }
        if self.periodic && c == 0 {
            self.flat.pop();
            self.flat.remove(0);
            self.seam -= 1;
        } else {
            self.flat.remove(c);
        }
        self.n -= 1;
    }

    /// Rebuilds the poles of a periodic direction after a removal: in one
    /// period of new functions, from the unchanged one before the new ones,
    /// a new function before the first old function `r - s + 1` keeps its
    /// index and the others lose one.
    fn rebuild_periodic(&mut self, r: isize, c: usize, s: usize) {
        let (p, width, n) = (self.degree as isize, self.width, self.n);
        let (new_n, new_seam) = (n - 1, self.seam - usize::from(c == 0));
        let run_slots = p as usize - s + 2;
        let (a, b) = (r - p, r - s as isize);
        let mut next = std::mem::take(&mut self.spare);
        next.clear();
        next.resize(self.rows * new_n * width, P::default());
        let base = a - 1;
        for slot in 0..new_n {
            let f = base
                + (slot as isize + new_seam as isize - p - 1 - base).rem_euclid(new_n as isize);
            for row in 0..self.rows {
                let to = &mut next[(row * new_n + slot) * width..][..width];
                if a <= f && f < b {
                    let at = (row * run_slots + (f - base) as usize) * width;
                    to.copy_from_slice(&self.scratch[at..][..width]);
                } else {
                    to.copy_from_slice(self.poles_at(row, if f < a { f } else { f + 1 }));
                }
            }
        }
        self.spare = std::mem::replace(&mut self.poles, next);
    }
}
