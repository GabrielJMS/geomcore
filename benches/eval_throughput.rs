//! Bulk-evaluation throughput baseline.
//!
//! Informational only — these numbers exist so future optimization work has
//! something concrete to improve on; they are not a correctness gate.

use criterion::{Criterion, criterion_group, criterion_main};
use std::f64::consts::TAU;
use std::hint::black_box;

use geomrust::curves::{BSplineCurve3D, Circle3D};
use geomrust::surfaces::BSplineSurface;
use geomrust::{Point3, Vector3};

const N: usize = 100_000;

fn circle_params() -> Vec<f64> {
    (0..N).map(|i| i as f64 / N as f64 * TAU).collect()
}

fn representative_curve() -> BSplineCurve3D {
    // Degree-3 clamped curve with 20 poles on a gentle helix.
    let poles: Vec<Point3> = (0..20)
        .map(|i| {
            let t = i as f64 / 19.0 * TAU;
            Point3::new(t.cos() * 3.0, t.sin() * 3.0, 0.4 * i as f64)
        })
        .collect();
    let knots: Vec<f64> = (0..=17).map(|i| i as f64).collect();
    let mut mults = vec![1u32; 18];
    mults[0] = 4;
    mults[17] = 4;
    BSplineCurve3D::new(3, poles, knots, mults, false).unwrap()
}

fn representative_surface() -> BSplineSurface {
    // Bicubic clamped 6x6 patch.
    let poles: Vec<Vec<Point3>> = (0..6)
        .map(|i| {
            (0..6)
                .map(|j| Point3::new(i as f64, j as f64, ((i * j) as f64 * 0.7).sin()))
                .collect()
        })
        .collect();
    let knots = vec![0.0, 1.0, 2.0, 3.0];
    let mults = vec![4u32, 1, 1, 4];
    BSplineSurface::new(
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
    .unwrap()
}

fn bench_eval_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("eval_throughput");
    group.sample_size(20);

    let circle = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
    let params = circle_params();
    group.bench_function("circle_eval_points_100k", |b| {
        b.iter(|| black_box(circle.eval_points(black_box(&params))))
    });

    let curve = representative_curve();
    let (first, last) = curve.bounds();
    let curve_params: Vec<f64> = (0..N)
        .map(|i| first + (last - first) * i as f64 / (N - 1) as f64)
        .collect();
    group.bench_function("bspline_curve_eval_points_100k", |b| {
        b.iter(|| black_box(curve.eval_points(black_box(&curve_params))))
    });

    let surface = representative_surface();
    let uvs: Vec<(f64, f64)> = (0..N)
        .map(|i| {
            let t = i as f64 / (N - 1) as f64;
            (t * 3.0, (1.0 - t) * 3.0)
        })
        .collect();
    group.bench_function("bspline_surface_eval_points_100k", |b| {
        b.iter(|| black_box(surface.eval_points(black_box(&uvs))))
    });

    group.finish();
}

criterion_group!(benches, bench_eval_throughput);
criterion_main!(benches);
