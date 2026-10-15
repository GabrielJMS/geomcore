//! Overlaps found by the numeric search: the runs along which a curve
//! lies on the other operand, joined from the hits and regions the
//! search left where it could not separate the operands.
//!
//! Where a curve lies on the other operand, the Bézier-hull search never
//! separates its pieces: it stops with an `Unseparated` region and keeps
//! the hits it found along the way. Before the result is built, the hits
//! and regions are taken in order along the (first) curve, and two
//! consecutive ones are joined when the curve stays on the other operand
//! between them: samples of the stretch lie within `tol.confusion` of it,
//! and the curve's tangent there runs along it. A joined run is then
//! extended to where the curve leaves the other operand, located by
//! halving, and reported as an overlap: the hits inside it are dropped,
//! and the regions inside it give way to one region over it. The result
//! stays `Incomplete`, since samples, not a proof, found the overlap.

use super::completeness::merged;
use super::{
    Completeness, CurveCurveHit, CurveCurveOverlap, CurveCurveRegion, CurveSurfaceHit,
    CurveSurfaceRegion,
};
use crate::curves::ParametricCurve3D;
use crate::surfaces::ParametricSurface;
use crate::{BoundingBox3D, Point3D, Tolerance, Vector3D};

/// Intervals each stretch is cut into to check that the curve lies on the
/// other operand along it.
const RUN_SAMPLES: usize = 8;

/// Halvings locating where a run ends; past 60 the interval is below the
/// spacing of doubles.
const END_HALVINGS: usize = 60;

/// Doublings of the step searching for the end of a run along an
/// unbounded curve.
const UNBOUNDED_DOUBLINGS: usize = 64;

/// Newton iterations following a point of the first curve onto the
/// second.
const FOOT_ITERATIONS: usize = 30;

/// Joins the hits and regions of a numeric curve-surface search into the
/// curve's runs lying on the surface (see the module documentation).
/// A result without an incomplete region has no run and comes back as
/// it is.
pub(super) fn curve_surface_overlaps<C, S>(
    curve: &C,
    surface: &S,
    hits: Vec<CurveSurfaceHit>,
    completeness: Completeness<CurveSurfaceRegion>,
    tol: Tolerance,
) -> (
    Vec<CurveSurfaceHit>,
    Vec<(f64, f64)>,
    Completeness<CurveSurfaceRegion>,
)
where
    C: ParametricCurve3D,
    S: ParametricSurface,
{
    let Completeness::Incomplete { reason, regions } = completeness else {
        return (hits, Vec::new(), completeness);
    };
    let mut on = |t: f64, spacing: f64| {
        let (p, d) = curve.eval_d1(t);
        let foot = surface.project_point(p, tol);
        if foot.distance > tol.confusion {
            return false;
        }
        let (_, [su, sv]) = surface.eval_d1(foot.u, foot.v);
        let normal = su.cross(sv);
        // A degenerate normal (a pole) leaves the distance to decide.
        let across = d.dot(normal).abs() / normal.magnitude().max(f64::MIN_POSITIVE);
        normal.magnitude() <= f64::EPSILON * su.magnitude() * sv.magnitude()
            || across * spacing <= 2.0 * tol.confusion
    };
    let anchors = hits
        .iter()
        .filter_map(|hit| hit.curve_parameter.map(|t| (t, t)))
        .chain(regions.iter().map(|region| region.curve_range))
        .collect();
    let runs = lying_runs(anchors, curve.bounds(), &mut on);
    let runs: Vec<(f64, f64)> = runs
        .into_iter()
        .filter(|&run| {
            longer_than_confusion(curve, run, tol)
                && settled(run, tol, |t| {
                    // The projection's own distance snaps to zero within
                    // confusion: measure to its point.
                    let p = curve.eval_point(t);
                    Some(p.distance(surface.project_point(p, tol).point))
                })
        })
        .collect();
    if runs.is_empty() {
        return (
            hits,
            Vec::new(),
            Completeness::Incomplete { reason, regions },
        );
    }
    let inside = |t: f64| runs.iter().any(|&run| covers(curve, run, t, tol));
    let hits = hits
        .into_iter()
        .filter(|hit| !hit.curve_parameter.is_some_and(inside))
        .collect();
    let mut regions = regions;
    for &run in &runs {
        let points = run_points(curve, run);
        let feet: Vec<(f64, f64)> = points
            .iter()
            .map(|&p| {
                let foot = surface.project_point(p, tol);
                (foot.u, foot.v)
            })
            .collect();
        let (min_corner, max_corner) = corners(&points);
        regions.push(CurveSurfaceRegion {
            curve_range: run,
            u_range: span(feet.iter().map(|f| f.0)),
            v_range: span(feet.iter().map(|f| f.1)),
            min_corner,
            max_corner,
        });
    }
    let regions = merged(regions);
    (hits, runs, Completeness::Incomplete { reason, regions })
}

/// Joins the hits and regions of a numeric curve-curve search into the
/// runs along which the first curve lies on the second (see the module
/// documentation). A run whose second range would cross the second
/// curve's seam, or along which the second curve's parameter does not
/// move one way, is left as it was found. A result without an
/// incomplete region has no run and comes back as it is.
pub(super) fn curve_curve_overlaps<A, B>(
    first: &A,
    second: &B,
    hits: Vec<CurveCurveHit>,
    completeness: Completeness<CurveCurveRegion>,
    tol: Tolerance,
) -> (
    Vec<CurveCurveHit>,
    Vec<CurveCurveOverlap>,
    Completeness<CurveCurveRegion>,
)
where
    A: ParametricCurve3D,
    B: ParametricCurve3D,
{
    let Completeness::Incomplete { reason, regions } = completeness else {
        return (hits, Vec::new(), completeness);
    };
    let mut follower = Follower {
        first,
        second,
        known: hits
            .iter()
            .filter_map(|hit| hit.first_parameter.zip(hit.second_parameter))
            .collect(),
        tol,
    };
    for region in &regions {
        let t = region.first_range.0;
        let (low, high) = region.second_range;
        let candidates = [low, 0.5 * (low + high), high];
        if let Some(s) = candidates
            .into_iter()
            .filter_map(|s| follower.newton(t, s))
            .min_by(|x, y| x.1.total_cmp(&y.1))
            .filter(|foot| foot.1 <= tol.confusion)
            .map(|foot| foot.0)
        {
            follower.known.push((t, s));
        }
    }
    let anchors = hits
        .iter()
        .filter_map(|hit| hit.first_parameter.map(|t| (t, t)))
        .chain(regions.iter().map(|region| region.first_range))
        .collect();
    let runs = lying_runs(anchors, first.bounds(), &mut |t, spacing| {
        follower.lies(t, spacing)
    });
    let mut overlaps = Vec::new();
    for run in runs {
        if !(longer_than_confusion(first, run, tol) && settled(run, tol, |t| follower.distance(t)))
        {
            continue;
        }
        if let Some(overlap) = follower.overlap(run) {
            overlaps.push(overlap);
        }
    }
    if overlaps.is_empty() {
        return (hits, overlaps, Completeness::Incomplete { reason, regions });
    }
    let inside = |t: f64| {
        overlaps
            .iter()
            .any(|overlap| covers(first, overlap.first_range, t, tol))
    };
    let hits = hits
        .into_iter()
        .filter(|hit| !hit.first_parameter.is_some_and(inside))
        .collect();
    let mut regions = regions;
    for overlap in &overlaps {
        let (min_corner, max_corner) = corners(&run_points(first, overlap.first_range));
        regions.push(CurveCurveRegion {
            first_range: overlap.first_range,
            second_range: overlap.second_range,
            min_corner,
            max_corner,
        });
    }
    let regions = merged(regions);
    (hits, overlaps, Completeness::Incomplete { reason, regions })
}

/// Follows points of the first curve onto the second: the second curve's
/// parameter nearest each, by Newton iteration from the known pair of
/// parameters closest along the first curve.
struct Follower<'a, A, B> {
    first: &'a A,
    second: &'a B,
    /// Pairs of parameters `(t, s)` where the curves meet.
    known: Vec<(f64, f64)>,
    tol: Tolerance,
}

impl<A: ParametricCurve3D, B: ParametricCurve3D> Follower<'_, A, B> {
    /// The second curve's parameter of the foot of the first curve's point
    /// at `t`, with its distance, by Newton iteration from `s`; `None`
    /// where the iteration breaks down.
    fn newton(&self, t: f64, s: f64) -> Option<(f64, f64)> {
        let p = self.first.eval_point(t);
        let (low, high) = self.second.bounds();
        let periodic = self.second.period().is_some();
        let mut s = s;
        for _ in 0..FOOT_ITERATIONS {
            let (q, d1, d2) = self.second.eval_d2(s);
            let offset = q - p;
            let slope = d1.dot(d1) + offset.dot(d2);
            if slope <= 0.0 || !slope.is_finite() {
                break;
            }
            let step = offset.dot(d1) / slope;
            let next = if periodic {
                s - step
            } else {
                (s - step).clamp(low, high)
            };
            let settled = (next - s).abs() <= self.tol.parametric;
            s = next;
            if settled {
                break;
            }
        }
        let distance = self.second.eval_point(s).distance(p);
        distance.is_finite().then_some((s, distance))
    }

    /// The second curve's parameter of the foot of the first curve's point
    /// at `t`, with its distance, by Newton iteration from the known pair
    /// nearest along the first curve, extrapolated to `t`.
    fn nearest_foot(&self, t: f64) -> Option<(f64, f64)> {
        let &(near_t, near_s) = self
            .known
            .iter()
            .min_by(|x, y| (x.0 - t).abs().total_cmp(&(y.0 - t).abs()))?;
        let d_first = self.first.eval_derivative(near_t, 1);
        let d_second = self.second.eval_derivative(near_s, 1);
        let ratio = d_first.dot(d_second) / d_second.dot(d_second).max(f64::MIN_POSITIVE);
        self.newton(t, near_s + (t - near_t) * ratio)
    }

    /// The distance from the first curve's point at `t` to the second
    /// curve, near the known pairs.
    fn distance(&self, t: f64) -> Option<f64> {
        self.nearest_foot(t).map(|foot| foot.1)
    }

    /// The second curve's parameter of the first curve's point at `t`,
    /// if that point lies on the second curve.
    fn foot(&mut self, t: f64) -> Option<f64> {
        let (s, distance) = self.nearest_foot(t)?;
        (distance <= self.tol.confusion).then(|| {
            self.known.push((t, s));
            s
        })
    }

    /// Whether the first curve lies on the second at `t`: its point within
    /// confusion of the second curve, and its tangent along the second's
    /// to within confusion over `spacing` of parameter.
    fn lies(&mut self, t: f64, spacing: f64) -> bool {
        let Some(s) = self.foot(t) else {
            return false;
        };
        let d_first = self.first.eval_derivative(t, 1);
        let d_second = self.second.eval_derivative(s, 1);
        let across = d_first.cross(unit(d_second)).magnitude();
        across * spacing <= 2.0 * self.tol.confusion
    }

    /// The overlap over the first curve's `run`: the second curve's
    /// parameters followed along it. `None` where they do not move one
    /// way, or cross the second curve's seam.
    fn overlap(&mut self, (low, high): (f64, f64)) -> Option<CurveCurveOverlap> {
        if !(low.is_finite() && high.is_finite()) {
            // An unbounded overlap lies on an unbounded second curve: its
            // parameters follow the ends' only in the limit.
            return None;
        }
        let steps = (0..=RUN_SAMPLES)
            .map(|k| self.foot(low + (high - low) * k as f64 / RUN_SAMPLES as f64))
            .collect::<Option<Vec<f64>>>()?;
        let same_direction = steps[RUN_SAMPLES] > steps[0];
        let monotone = steps
            .windows(2)
            .all(|pair| (pair[1] > pair[0]) == same_direction && pair[1] != pair[0]);
        if !monotone {
            return None;
        }
        let (mut start, mut end) = (
            steps[0].min(steps[RUN_SAMPLES]),
            steps[0].max(steps[RUN_SAMPLES]),
        );
        let (second_low, second_high) = self.second.bounds();
        if let Some(period) = self.second.period() {
            let shift = ((start - second_low) / period).floor() * period;
            (start, end) = (start - shift, end - shift);
        }
        let slack = self.tol.parametric;
        if start < second_low - slack || end > second_high + slack {
            return None;
        }
        Some(CurveCurveOverlap {
            first_range: (low, high),
            second_range: (start.max(second_low), end.min(second_high)),
            same_direction,
        })
    }
}

/// The runs along which the curve lies on the other operand, by `on(t,
/// spacing)`, found from `anchors`: the curve parameters of the hits as
/// `(t, t)` and the curve ranges of the regions. Each region is probed
/// at `RUN_SAMPLES + 1` points, each hit at its own; two consecutive
/// probes on the other operand join when the curve lies on it at
/// `RUN_SAMPLES - 1` points between them. A chain of two probes or more is
/// extended at each end to where the curve leaves the other operand,
/// found between the chain's end and the next probe (or the curve's
/// `bounds`).
fn lying_runs(
    anchors: Vec<(f64, f64)>,
    bounds: (f64, f64),
    on: &mut impl FnMut(f64, f64) -> bool,
) -> Vec<(f64, f64)> {
    let mut probes: Vec<f64> = anchors
        .into_iter()
        .flat_map(|(low, high)| {
            let count = if high > low { RUN_SAMPLES + 1 } else { 1 };
            (0..count).map(move |k| low + (high - low) * k as f64 / RUN_SAMPLES as f64)
        })
        .filter(|t| t.is_finite())
        .collect();
    probes.sort_by(f64::total_cmp);
    probes.dedup();
    let count = probes.len();
    let gap = |i: usize, j: usize| (probes[j] - probes[i]).abs();
    let lying: Vec<bool> = (0..count)
        .map(|i| {
            let before = if i > 0 { gap(i - 1, i) } else { f64::INFINITY };
            let after = if i + 1 < count {
                gap(i, i + 1)
            } else {
                f64::INFINITY
            };
            let spacing = before.min(after) / RUN_SAMPLES as f64;
            on(probes[i], if spacing.is_finite() { spacing } else { 0.0 })
        })
        .collect();
    let mut joins = |i: usize| {
        let (low, high) = (probes[i], probes[i + 1]);
        let spacing = (high - low) / RUN_SAMPLES as f64;
        (1..RUN_SAMPLES).all(|k| on(low + spacing * k as f64, spacing))
    };
    let mut chains = Vec::new();
    let mut i = 0;
    while i < count {
        let mut j = i;
        while lying[i] && j + 1 < count && lying[j + 1] && joins(j) {
            j += 1;
        }
        if j > i {
            chains.push((i, j));
        }
        i = j + 1;
    }
    chains
        .into_iter()
        .map(|(i, j)| {
            let (low, high) = (probes[i], probes[j]);
            let below = if i > 0 { probes[i - 1] } else { bounds.0 };
            let above = if j + 1 < count {
                probes[j + 1]
            } else {
                bounds.1
            };
            let length = high - low;
            (
                run_end(low, below, length, on),
                run_end(high, above, length, on),
            )
        })
        .collect()
}

/// Where the run ending at `end` (on the other operand) stops, moving
/// toward `limit`: the last point on the other operand, found by halving
/// between the last sample on it and the first off it. Toward an
/// infinite limit the step starts at the run's `length` and doubles.
fn run_end(end: f64, limit: f64, length: f64, on: &mut impl FnMut(f64, f64) -> bool) -> f64 {
    if limit == end {
        return end;
    }
    let mut last_on = end;
    let mut first_off = None;
    if limit.is_finite() {
        let spacing = (limit - end) / RUN_SAMPLES as f64;
        for k in 1..=RUN_SAMPLES {
            let t = if k == RUN_SAMPLES {
                limit
            } else {
                end + spacing * k as f64
            };
            if !on(t, spacing.abs()) {
                first_off = Some(t);
                break;
            }
            last_on = t;
        }
    } else {
        let direction = limit.signum();
        let mut step = length;
        for _ in 0..UNBOUNDED_DOUBLINGS {
            let t = last_on + direction * step;
            if !on(t, step) {
                first_off = Some(t);
                break;
            }
            last_on = t;
            step *= 2.0;
        }
    }
    let Some(mut off) = first_off else {
        return limit;
    };
    for _ in 0..END_HALVINGS {
        let middle = 0.5 * (last_on + off);
        if middle == last_on || middle == off {
            break;
        }
        if on(middle, (off - last_on).abs()) {
            last_on = middle;
        } else {
            off = middle;
        }
    }
    last_on
}

/// Whether the curve stays well within confusion of the other operand
/// inside `run`, by `distance(t)`: within a quarter of `tol.confusion` at
/// the `RUN_SAMPLES - 1` points between its ends.
///
/// This tells an overlap from a tangent contact. Two curves (or a curve
/// and a surface) touching with curvatures differing by `k` part by `k
/// h^2 / 2` at `h` from the contact, so they stay within confusion over a
/// short run, whose ends are where they part by confusion. Three eighths
/// of the run from its middle, they part by `(3/4)^2 = 0.56` of
/// confusion, more than a quarter. Along an overlap, they do not part.
fn settled(run: (f64, f64), tol: Tolerance, mut distance: impl FnMut(f64) -> Option<f64>) -> bool {
    let (low, high) = run;
    if !(low.is_finite() && high.is_finite()) {
        return true;
    }
    (1..RUN_SAMPLES).all(|k| {
        let t = low + (high - low) * k as f64 / RUN_SAMPLES as f64;
        distance(t).is_some_and(|d| d <= 0.25 * tol.confusion)
    })
}

/// Whether the curve over `run` is longer in space than `tol.confusion`
/// (summed over `RUN_SAMPLES` chords; an unbounded run is).
fn longer_than_confusion<C: ParametricCurve3D>(curve: &C, run: (f64, f64), tol: Tolerance) -> bool {
    if !(run.0.is_finite() && run.1.is_finite()) {
        return true;
    }
    let points = run_points(curve, run);
    let length: f64 = points
        .windows(2)
        .map(|pair| pair[0].distance(pair[1]))
        .sum();
    length > tol.confusion
}

/// Whether the curve parameter `t` lies within `run`, or within
/// `tol.confusion` in space of its ends.
fn covers<C: ParametricCurve3D>(
    curve: &C,
    (low, high): (f64, f64),
    t: f64,
    tol: Tolerance,
) -> bool {
    let near = |end: f64| {
        let speed = if end.is_finite() {
            curve.eval_derivative(end, 1).magnitude()
        } else {
            0.0
        };
        (t - end).abs() * speed <= tol.confusion
    };
    (low..=high).contains(&t) || near(low) || near(high)
}

/// `RUN_SAMPLES + 1` points of the curve over a finite `run`; over an
/// unbounded one, the points at its finite ends.
fn run_points<C: ParametricCurve3D>(curve: &C, (low, high): (f64, f64)) -> Vec<Point3D> {
    if !(low.is_finite() && high.is_finite()) {
        return [low, high]
            .into_iter()
            .filter(|t| t.is_finite())
            .map(|t| curve.eval_point(t))
            .collect();
    }
    (0..=RUN_SAMPLES)
        .map(|k| curve.eval_point(low + (high - low) * k as f64 / RUN_SAMPLES as f64))
        .collect()
}

/// The corners of the box around `points`, or of an unbounded box when
/// there are none (an overlap unbounded at both ends).
fn corners(points: &[Point3D]) -> (Point3D, Point3D) {
    match BoundingBox3D::from_points(points.iter().copied()) {
        Some(bounds) => (bounds.min(), bounds.max()),
        None => (
            Point3D::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
            Point3D::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
        ),
    }
}

/// The least and greatest of `values`.
fn span(values: impl Iterator<Item = f64>) -> (f64, f64) {
    values.fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), x| {
        (low.min(x), high.max(x))
    })
}

/// `v` scaled to unit length, or `v` itself when it vanishes.
fn unit(v: Vector3D) -> Vector3D {
    let length = v.magnitude();
    if length > 0.0 { v * (1.0 / length) } else { v }
}

#[cfg(test)]
mod tests {
    use crate::{
        BSplineCurve3D, Circle3D, Completeness, Cylinder, IncompleteReason, Intersect, Line3D,
        Plane, Point3D, Tolerance, Vector3D,
    };

    const TOL: Tolerance = Tolerance::DEFAULT;

    fn segment(from: Point3D, to: Point3D, degree: usize) -> BSplineCurve3D {
        let poles = if degree == 1 {
            vec![from, to]
        } else {
            vec![
                from,
                Point3D::new(
                    0.5 * (from.x + to.x),
                    0.5 * (from.y + to.y),
                    0.5 * (from.z + to.z),
                ),
                to,
            ]
        };
        BSplineCurve3D::new(
            degree,
            poles,
            vec![0.0, 1.0],
            vec![degree as u32 + 1, degree as u32 + 1],
            false,
        )
        .unwrap()
    }

    fn assert_range(actual: (f64, f64), expected: (f64, f64)) {
        let close = (actual.0 - expected.0).abs() < 1e-6 && (actual.1 - expected.1).abs() < 1e-6;
        assert!(close, "expected {expected:?}, got {actual:?}");
    }

    /// Asserts the result is incomplete, `Unseparated`, with a region
    /// whose curve range holds `range`.
    fn assert_unseparated_over<R: std::fmt::Debug>(
        completeness: &Completeness<R>,
        range: (f64, f64),
        curve_range: impl Fn(&R) -> (f64, f64),
    ) {
        let Completeness::Incomplete { reason, regions } = completeness else {
            panic!("expected an incomplete result, got {completeness:?}");
        };
        assert_eq!(*reason, IncompleteReason::Unseparated);
        assert!(regions.iter().any(|region| {
            let (low, high) = curve_range(region);
            low <= range.0 + 1e-6 && high >= range.1 - 1e-6
        }));
    }

    #[test]
    fn test_planar_bspline_lying_in_a_plane_is_one_overlap() {
        let p = Point3D::new;
        let poles = vec![
            p(0.0, 0.0, 0.0),
            p(1.0, 2.0, 0.0),
            p(3.0, -1.0, 0.0),
            p(4.0, 1.0, 0.0),
        ];
        let curve =
            BSplineCurve3D::new(2, poles, vec![0.0, 0.5, 1.0], vec![3, 1, 3], false).unwrap();
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let result = curve.intersect(&plane, TOL);
        assert!(result.hits().is_empty(), "{:?}", result.hits());
        assert_eq!(result.overlaps(), &[(0.0, 1.0)]);
        assert_unseparated_over(result.completeness(), (0.0, 1.0), |r| r.curve_range);
        assert_eq!(plane.intersect(&curve, TOL).overlaps(), &[(0.0, 1.0)]);
    }

    #[test]
    fn test_curve_leaving_a_plane_overlaps_only_where_it_lies() {
        // In the plane z = 0 for t in [0, 0.5], rising off it after.
        let p = Point3D::new;
        let poles = vec![
            p(0.0, 0.0, 0.0),
            p(1.0, 1.0, 0.0),
            p(2.0, 0.0, 0.0),
            p(3.0, 1.0, 1.0),
        ];
        let curve =
            BSplineCurve3D::new(1, poles, vec![0.0, 1.0, 2.0, 3.0], vec![2, 1, 1, 2], false)
                .unwrap();
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let result = curve.intersect(&plane, TOL);
        assert_eq!(result.overlaps().len(), 1);
        assert_range(result.overlaps()[0], (0.0, 2.0));
        assert!(result.hits().is_empty(), "{:?}", result.hits());
    }

    #[test]
    fn test_straight_bspline_on_a_line_overlaps_in_either_order() {
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let curve = segment(Point3D::new(2.0, 0.0, 0.0), Point3D::new(4.0, 0.0, 0.0), 1);
        let result = curve.intersect(&line, TOL);
        assert!(result.hits().is_empty());
        let [overlap] = result.overlaps() else {
            panic!("expected one overlap, got {:?}", result.overlaps());
        };
        assert_range(overlap.first_range, (0.0, 1.0));
        assert_range(overlap.second_range, (2.0, 4.0));
        assert!(overlap.same_direction);
        assert_unseparated_over(result.completeness(), (0.0, 1.0), |r| r.first_range);

        let result = line.intersect(&curve, TOL);
        assert!(result.hits().is_empty());
        let [overlap] = result.overlaps() else {
            panic!("expected one overlap, got {:?}", result.overlaps());
        };
        assert_range(overlap.first_range, (2.0, 4.0));
        assert_range(overlap.second_range, (0.0, 1.0));
        assert!(overlap.same_direction);
        assert_unseparated_over(result.completeness(), (2.0, 4.0), |r| r.first_range);
    }

    #[test]
    fn test_collinear_bsplines_overlap_where_they_share_the_line() {
        let p = Point3D::new;
        let first = segment(p(0.0, 0.0, 0.0), p(2.0, 2.0, 2.0), 2);
        let second = segment(p(1.0, 1.0, 1.0), p(3.0, 3.0, 3.0), 1);
        let result = first.intersect(&second, TOL);
        assert!(result.hits().is_empty());
        let [overlap] = result.overlaps() else {
            panic!("expected one overlap, got {:?}", result.overlaps());
        };
        assert_range(overlap.first_range, (0.5, 1.0));
        assert_range(overlap.second_range, (0.0, 0.5));
        assert!(overlap.same_direction);

        let reversed = segment(p(3.0, 3.0, 3.0), p(1.0, 1.0, 1.0), 1);
        let result = first.intersect(&reversed, TOL);
        let [overlap] = result.overlaps() else {
            panic!("expected one overlap, got {:?}", result.overlaps());
        };
        assert_range(overlap.first_range, (0.5, 1.0));
        assert_range(overlap.second_range, (0.5, 1.0));
        assert!(!overlap.same_direction);
    }

    #[test]
    fn test_circle_and_its_bspline_form_overlap_whole() {
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let copy = circle.to_bspline();
        let result = circle.intersect(&copy, TOL);
        assert!(result.hits().is_empty());
        let [overlap] = result.overlaps() else {
            panic!("expected one overlap, got {:?}", result.overlaps());
        };
        assert_range(overlap.first_range, (0.0, std::f64::consts::TAU));
        assert!(overlap.same_direction);
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let result = copy.intersect(&cylinder, TOL);
        assert!(result.hits().is_empty());
        assert_eq!(result.overlaps().len(), 1);
    }

    #[test]
    fn test_tangent_contact_is_a_hit_not_an_overlap() {
        let p = Point3D::new;
        let poles = vec![p(0.0, 0.0, 1.0), p(2.0, 0.0, -1.0), p(4.0, 0.0, 1.0)];
        let parabola = BSplineCurve3D::new(2, poles, vec![0.0, 1.0], vec![3, 3], false).unwrap();
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let result = parabola.intersect(&plane, TOL);
        assert!(result.overlaps().is_empty(), "{result:?}");
        assert_eq!(result.hits().len(), 1);
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let result = parabola.intersect(&line, TOL);
        assert!(result.overlaps().is_empty());
        assert_eq!(result.hits().len(), 1);
    }
}
