//! Elementary curves: equal parameter steps, their count from closed-form
//! deviations.
//!
//! The chord of a conic step `[m − h/2, m + h/2]` is parallel to the
//! tangent at `m`, so the farthest curve point from the chord's line is at
//! `m`, and the largest angle between the tangent and the chord is at an
//! end of the step. That distance is the chord deviation as long as every
//! point of the step projects onto the chord, which holds when that angle
//! is below a right angle; steps turning more report an infinite chord
//! deviation, so they never fit.

use super::sealed::{DiscretizeRange3D, Run};
use super::{DiscretizationCriteria, smallest_count};
use crate::Point3D;
use crate::curves::ParametricCurve3D;
use crate::{Circle3D, Curve3D, Ellipse3D, Hyperbola3D, Line3D, Parabola3D};
use std::f64::consts::{FRAC_PI_2, TAU};

/// One run of equal steps over `[first, last]`: the fewest steps for
/// which `deviations(k)`, the chord and angle deviation of each of `k`
/// steps (both falling as `k` grows), meet the criteria. `guess`
/// estimates that count.
fn equal_steps<C: ParametricCurve3D + ?Sized, F>(
    curve: &C,
    (first, last): (f64, f64),
    criteria: &DiscretizationCriteria,
    min_segments: usize,
    guess: f64,
    deviations: impl Fn(usize) -> (f64, f64),
    mut visit: F,
) where
    F: FnMut(Run, &dyn Fn(f64) -> Point3D),
{
    let fits = |k| {
        let (chord, angle) = deviations(k);
        chord <= criteria.max_chord_deviation
            && criteria.angle_limit().is_none_or(|limit| angle <= limit)
    };
    let segments = smallest_count(guess, fits).max(min_segments);
    let (chord_deviation, angle_deviation) = deviations(segments);
    let run = Run {
        first,
        last,
        segments,
        chord_deviation,
        angle_deviation,
    };
    visit(run, &|s| curve.eval_point(first + (last - first) * s));
}

impl DiscretizeRange3D for Circle3D {
    /// A segment of arc angle `θ` deviates from the circle by exactly
    /// `r (1 − cos θ/2) = 2 r sin²(θ/4)` and `θ/2`.
    fn for_each_run<F>(
        &self,
        first: f64,
        last: f64,
        criteria: &DiscretizationCriteria,
        min_segments: usize,
        visit: F,
    ) where
        F: FnMut(Run, &dyn Fn(f64) -> Point3D),
    {
        let r = self.radius();
        let sweep = last - first;
        let deviations = |k: usize| {
            let theta = sweep / k as f64;
            (2.0 * r * (0.25 * theta).sin().powi(2), 0.5 * theta)
        };
        // θ ≤ 4 asin(sqrt(ε / 2r)) for the chord, θ ≤ 2α for the angle.
        let ratio = (criteria.max_chord_deviation / (2.0 * r)).min(1.0);
        let mut guess = sweep / (4.0 * ratio.sqrt().asin());
        if let Some(limit) = criteria.angle_limit() {
            guess = guess.max(sweep / (2.0 * limit));
        }
        equal_steps(
            self,
            (first, last),
            criteria,
            min_segments,
            guess,
            deviations,
            visit,
        );
    }
}

/// The chord deviation of a conic step whose farthest point lies
/// `distance` from the chord's line, and whose tangent turns at most
/// `angle` from the chord.
fn conic_chord_deviation(distance: f64, angle: f64) -> f64 {
    if angle < FRAC_PI_2 {
        distance
    } else {
        f64::INFINITY
    }
}

/// The largest `cos x` over `[low, high]`.
fn max_cos(low: f64, high: f64) -> f64 {
    if (low / TAU).ceil() * TAU <= high {
        1.0
    } else {
        low.cos().max(high.cos())
    }
}

/// The value in `[low, high]` nearest zero; `high` when rounding leaves
/// `low` above it.
fn nearest_zero(low: f64, high: f64) -> f64 {
    0.0_f64.max(low).min(high)
}

/// The count of equal steps over `sweep` for a chord deviation of
/// `h²/8 · chord_rate` and an angle deviation of `h/2 · turning_rate`,
/// both rates being their largest over the range, the angle staying below
/// a right angle.
fn conic_guess(
    sweep: f64,
    criteria: &DiscretizationCriteria,
    chord_rate: f64,
    turning_rate: f64,
) -> f64 {
    let guess = sweep / (8.0 * criteria.max_chord_deviation / chord_rate).sqrt();
    let limit = criteria
        .angle_limit()
        .map_or(FRAC_PI_2, |limit| limit.min(FRAC_PI_2));
    guess.max(sweep * turning_rate / (2.0 * limit))
}

impl DiscretizeRange3D for Ellipse3D {
    /// The image of a circle under `diag(a, b)`: a step of `h` around `m`
    /// lies `2 sin²(h/4) / sqrt(cos²m/a² + sin²m/b²)` from its chord, the
    /// largest over a range near the ends of the major axis.
    fn for_each_run<F>(
        &self,
        first: f64,
        last: f64,
        criteria: &DiscretizationCriteria,
        min_segments: usize,
        visit: F,
    ) where
        F: FnMut(Run, &dyn Fn(f64) -> Point3D),
    {
        let (a, b) = (self.major_radius(), self.minor_radius());
        let sweep = last - first;
        // `1 / sqrt(cos²m/a² + sin²m/b²)` over the midpoints `m`, from the
        // largest `cos 2m` over them.
        let distance_factor = |cos_double: f64| {
            let (inverse_a2, inverse_b2) = (1.0 / (a * a), 1.0 / (b * b));
            let sum = inverse_a2 + inverse_b2;
            1.0 / (0.5 * (sum + (inverse_a2 - inverse_b2) * cos_double)).sqrt()
        };
        let deviations = |k: usize| {
            let h = sweep / k as f64;
            // The tangents at an end `t` and the middle `m` of a step have
            // cross product `ab sin(h/2)` and dot product
            // `((a² + b²) cos(h/2) + (b² − a²) cos(t + m)) / 2`.
            let cos_sum = max_cos(2.0 * first + 0.5 * h, 2.0 * last - 0.5 * h);
            let dot = 0.5 * ((a * a + b * b) * (0.5 * h).cos() + (b * b - a * a) * cos_sum);
            let angle = (a * b * (0.5 * h).sin()).atan2(dot);
            let cos_double = max_cos(2.0 * first + h, 2.0 * last - h);
            let distance = 2.0 * (0.25 * h).sin().powi(2) * distance_factor(cos_double);
            (conic_chord_deviation(distance, angle), angle)
        };
        let cos_double = max_cos(2.0 * first, 2.0 * last);
        let cos2 = 0.5 * (1.0 + cos_double);
        let turning_rate = a * b / (a * a * (1.0 - cos2) + b * b * cos2);
        let guess = conic_guess(sweep, criteria, distance_factor(cos_double), turning_rate);
        equal_steps(
            self,
            (first, last),
            criteria,
            min_segments,
            guess,
            deviations,
            visit,
        );
    }
}

impl DiscretizeRange3D for Hyperbola3D {
    /// The image of `(cosh u, sinh u)` under `diag(a, b)`: a step of `h`
    /// around `m` lies `2 sinh²(h/4) / sqrt(cosh²m/a² + sinh²m/b²)` from
    /// its chord, the largest over a range nearest the vertex.
    fn for_each_run<F>(
        &self,
        first: f64,
        last: f64,
        criteria: &DiscretizationCriteria,
        min_segments: usize,
        visit: F,
    ) where
        F: FnMut(Run, &dyn Fn(f64) -> Point3D),
    {
        let (a, b) = (self.major_radius(), self.minor_radius());
        let sweep = last - first;
        let distance_factor =
            |m: f64| 1.0 / ((m.cosh() / a).powi(2) + (m.sinh() / b).powi(2)).sqrt();
        let deviations = |k: usize| {
            let h = sweep / k as f64;
            // The tangents at an end `t` and the middle `m` of a step have
            // cross product `ab sinh(h/2)` and dot product
            // `((b² − a²) cosh(h/2) + (a² + b²) cosh(t + m)) / 2`.
            let sum = nearest_zero(2.0 * first + 0.5 * h, 2.0 * last - 0.5 * h);
            let dot = 0.5 * ((b * b - a * a) * (0.5 * h).cosh() + (a * a + b * b) * sum.cosh());
            let angle = (a * b * (0.5 * h).sinh()).atan2(dot);
            let middle = nearest_zero(first + 0.5 * h, last - 0.5 * h);
            let distance = 2.0 * (0.25 * h).sinh().powi(2) * distance_factor(middle);
            (conic_chord_deviation(distance, angle), angle)
        };
        let nearest = nearest_zero(first, last);
        let turning_rate = a * b / ((a * nearest.sinh()).powi(2) + (b * nearest.cosh()).powi(2));
        let guess = conic_guess(sweep, criteria, distance_factor(nearest), turning_rate);
        equal_steps(
            self,
            (first, last),
            criteria,
            min_segments,
            guess,
            deviations,
            visit,
        );
    }
}

impl DiscretizeRange3D for Parabola3D {
    /// `u²/4f` along the axis, `u` across: a step of `h` around `m` lies
    /// `h²/16f / sqrt(1 + m²/4f²)` from its chord, the largest over a
    /// range nearest the vertex.
    fn for_each_run<F>(
        &self,
        first: f64,
        last: f64,
        criteria: &DiscretizationCriteria,
        min_segments: usize,
        visit: F,
    ) where
        F: FnMut(Run, &dyn Fn(f64) -> Point3D),
    {
        let focal = self.focal();
        let sweep = last - first;
        let distance_factor = |m: f64| 1.0 / (1.0 + (m / (2.0 * focal)).powi(2)).sqrt();
        let deviations = |k: usize| {
            let h = sweep / k as f64;
            // The tangents at an end `t` and the middle `m` of a step have
            // cross product `h/4f` and dot product `1 + tm/4f²`, where
            // `4tm = (t + m)² − h²/4`.
            let sum = nearest_zero(2.0 * first + 0.5 * h, 2.0 * last - 0.5 * h);
            let dot = 1.0 + (sum * sum - 0.25 * h * h) / (16.0 * focal * focal);
            let angle = (h / (4.0 * focal)).atan2(dot);
            let middle = nearest_zero(first + 0.5 * h, last - 0.5 * h);
            let distance = h * h / (16.0 * focal) * distance_factor(middle);
            (conic_chord_deviation(distance, angle), angle)
        };
        let nearest = nearest_zero(first, last);
        let chord_rate = distance_factor(nearest) / (2.0 * focal);
        let turning_rate = distance_factor(nearest).powi(2) / (2.0 * focal);
        let guess = conic_guess(sweep, criteria, chord_rate, turning_rate);
        equal_steps(
            self,
            (first, last),
            criteria,
            min_segments,
            guess,
            deviations,
            visit,
        );
    }
}

impl DiscretizeRange3D for Line3D {
    /// A line is its own chord.
    fn for_each_run<F>(
        &self,
        first: f64,
        last: f64,
        criteria: &DiscretizationCriteria,
        min_segments: usize,
        visit: F,
    ) where
        F: FnMut(Run, &dyn Fn(f64) -> Point3D),
    {
        equal_steps(
            self,
            (first, last),
            criteria,
            min_segments,
            1.0,
            |_| (0.0, 0.0),
            visit,
        );
    }
}

impl DiscretizeRange3D for Curve3D {
    fn for_each_run<F>(
        &self,
        first: f64,
        last: f64,
        criteria: &DiscretizationCriteria,
        min_segments: usize,
        visit: F,
    ) where
        F: FnMut(Run, &dyn Fn(f64) -> Point3D),
    {
        match self {
            Curve3D::Line(c) => c.for_each_run(first, last, criteria, min_segments, visit),
            Curve3D::Circle(c) => c.for_each_run(first, last, criteria, min_segments, visit),
            Curve3D::Ellipse(c) => c.for_each_run(first, last, criteria, min_segments, visit),
            Curve3D::Parabola(c) => c.for_each_run(first, last, criteria, min_segments, visit),
            Curve3D::Hyperbola(c) => c.for_each_run(first, last, criteria, min_segments, visit),
            Curve3D::BSpline(c) => c.for_each_run(first, last, criteria, min_segments, visit),
        }
    }
}
