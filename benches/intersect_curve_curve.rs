//! Cost of the numeric curve–curve intersection and extrema on B-splines.
//!
//! Times `intersect_curve_curve` on 64 pairs of cubic B-spline curves lying
//! in one plane, one crossing the square `[0, 5]^2` from left to right and
//! the other from bottom to top along wavy paths, so that every pair
//! crosses at least once; and `curve_curve_extrema` on 64 pairs of skew
//! cubic B-spline curves, wavy in all three directions, one above the
//! other.
//!
//! Informational only, like `eval_throughput`; not a correctness gate.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use geomcore::{BSplineCurve3D, Point3D, Tolerance, curve_curve_extrema, intersect_curve_curve};

const PAIRS: usize = 64;

/// A deterministic stream of numbers in `[0, 1)`.
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

/// A cubic through six poles, from `from(s)` at `s = 0` to `s = 1`, with
/// random offsets `wave` scaled by `amplitude` per pole.
fn cubic(rng: &mut Lcg, from: impl Fn(f64) -> [f64; 3], amplitude: [f64; 3]) -> BSplineCurve3D {
    let poles = (0..6)
        .map(|k| {
            let [x, y, z] = from(k as f64 / 5.0);
            let mut wave = || 2.0 * rng.next() - 1.0;
            Point3D::new(
                x + amplitude[0] * wave(),
                y + amplitude[1] * wave(),
                z + amplitude[2] * wave(),
            )
        })
        .collect();
    BSplineCurve3D::new(3, poles, vec![0.0, 1.0, 2.0, 3.0], vec![4, 1, 1, 4], false).unwrap()
}

fn crossing_pairs() -> Vec<(BSplineCurve3D, BSplineCurve3D)> {
    let mut rng = Lcg(42);
    (0..PAIRS)
        .map(|_| {
            let (y0, y1) = (5.0 * rng.next(), 5.0 * rng.next());
            let (x0, x1) = (5.0 * rng.next(), 5.0 * rng.next());
            let across = cubic(
                &mut rng,
                |s| [-0.5 + 6.0 * s, y0 + (y1 - y0) * s, 0.0],
                [0.0, 1.0, 0.0],
            );
            let up = cubic(
                &mut rng,
                |s| [x0 + (x1 - x0) * s, -0.5 + 6.0 * s, 0.0],
                [1.0, 0.0, 0.0],
            );
            (across, up)
        })
        .collect()
}

fn skew_pairs() -> Vec<(BSplineCurve3D, BSplineCurve3D)> {
    let mut rng = Lcg(7);
    (0..PAIRS)
        .map(|_| {
            let (y0, y1) = (5.0 * rng.next(), 5.0 * rng.next());
            let (x0, x1) = (5.0 * rng.next(), 5.0 * rng.next());
            let below = cubic(
                &mut rng,
                |s| [6.0 * s, y0 + (y1 - y0) * s, 0.0],
                [0.5, 1.0, 0.5],
            );
            let above = cubic(
                &mut rng,
                |s| [x0 + (x1 - x0) * s, 6.0 * s, 2.0],
                [1.0, 0.5, 0.5],
            );
            (below, above)
        })
        .collect()
}

fn bench_curve_curve(c: &mut Criterion) {
    let tol = Tolerance::default();
    let crossing = crossing_pairs();
    let skew = skew_pairs();
    let mut group = c.benchmark_group("curve_curve");
    group.throughput(Throughput::Elements(PAIRS as u64));
    group.bench_function("intersect_crossing_bsplines", |b| {
        b.iter(|| {
            for (p, q) in black_box(&crossing) {
                black_box(intersect_curve_curve(p, q, tol));
            }
        })
    });
    group.bench_function("extrema_skew_bsplines", |b| {
        b.iter(|| {
            for (p, q) in black_box(&skew) {
                black_box(curve_curve_extrema(p, q, tol));
            }
        })
    });
    group.finish();
}

criterion_group!(benches, bench_curve_curve);
criterion_main!(benches);
