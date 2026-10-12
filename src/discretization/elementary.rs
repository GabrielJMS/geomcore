//! Elementary curves: equal parameter steps, their count from closed-form
//! deviations.

use super::sealed::{DiscretizeRange3D, Run};
use super::{DiscretizationCriteria, smallest_count};
use crate::Circle3D;
use crate::Point3D;
use crate::curves::ParametricCurve3D;

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
