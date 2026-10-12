//! Knot removal on B-spline curves and surfaces.
//!
//! Informational only, like the other benches. Two workloads on cubic
//! curves, polynomial and rational, each timed as one call per knot on the
//! result of the previous call and as one call for the whole list:
//! removing 8 knots just inserted into a curve of 32 poles (40 poles with
//! them), and trying to remove every interior knot of a curve of 32 poles,
//! all needed, at a tolerance of 1e-6 relative to its size. On a bicubic
//! 6x6 surface, 8 knots just inserted are removed in either direction.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use geomcore::surfaces::BSplineSurface;
use geomcore::{BSplineCurve3D, Point3D};

/// Clamped cubic of 32 poles on a helix with uniform knots, rational with
/// varying weights if `rational`.
fn curve(rational: bool) -> BSplineCurve3D {
    let poles: Vec<Point3D> = (0..32)
        .map(|i| {
            let t = i as f64 * 0.4;
            Point3D::new(3.0 * t.cos(), 3.0 * t.sin(), 0.3 * i as f64)
        })
        .collect();
    let knots: Vec<f64> = (0..=29).map(f64::from).collect();
    let mut mults = vec![1; 30];
    mults[0] = 4;
    mults[29] = 4;
    if rational {
        let weights = (0..32).map(|i| 1.0 + 0.5 * ((i % 3) as f64)).collect();
        BSplineCurve3D::new_rational(3, poles, weights, knots, mults, false).unwrap()
    } else {
        BSplineCurve3D::new(3, poles, knots, mults, false).unwrap()
    }
}

fn surface() -> BSplineSurface {
    let poles: Vec<Vec<Point3D>> = (0..6)
        .map(|i| {
            (0..6)
                .map(|j| Point3D::new(i as f64, j as f64, ((i * j) as f64 * 0.7).sin()))
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

/// `count` knots spread over `(first, last)`, none on an existing knot.
fn spread(count: usize, (first, last): (f64, f64)) -> Vec<f64> {
    (0..count)
        .map(|i| first + (last - first) * (i as f64 + 0.37) / count as f64)
        .collect()
}

/// Removes `knots` one call per knot, each on the result of the previous.
fn one_by_one(curve: &BSplineCurve3D, knots: &[f64], tol: f64) -> (BSplineCurve3D, u32) {
    let mut current = curve.clone();
    let mut removed = 0;
    for &u in knots {
        let (next, count) = current.remove_knot(u, 1, tol).unwrap();
        current = next;
        removed += count;
    }
    (current, removed)
}

fn bench_knot_removal(c: &mut Criterion) {
    let mut group = c.benchmark_group("knot_removal");
    for (name, rational) in [("polynomial", false), ("rational", true)] {
        let curve = curve(rational);
        let tol = 1e-6 * 9.3;
        let inserted = spread(8, curve.bounds());
        let refined = curve.refine_knots(&inserted).unwrap();
        assert_eq!(one_by_one(&refined, &inserted, tol).1, 8);
        group.bench_function(format!("cubic_{name}_inserted_8_one_by_one"), |b| {
            b.iter(|| one_by_one(black_box(&refined), black_box(&inserted), tol))
        });
        group.bench_function(format!("cubic_{name}_inserted_8_batch"), |b| {
            b.iter(|| {
                black_box(&refined)
                    .remove_knots(black_box(&inserted), tol)
                    .unwrap()
            })
        });
        let interior = &curve.knots()[1..curve.knots().len() - 1];
        assert_eq!(one_by_one(&curve, interior, tol).1, 0);
        group.bench_function(format!("cubic_{name}_needed_28_one_by_one"), |b| {
            b.iter(|| one_by_one(black_box(&curve), black_box(interior), tol))
        });
        group.bench_function(format!("cubic_{name}_needed_28_batch"), |b| {
            b.iter(|| {
                black_box(&curve)
                    .remove_knots(black_box(interior), tol)
                    .unwrap()
            })
        });
    }
    let surface = surface();
    let knots = spread(8, (0.0, 3.0));
    let by_u = surface.refine_u_knots(&knots).unwrap();
    let by_v = surface.refine_v_knots(&knots).unwrap();
    group.bench_function("surface_6x6_bicubic_inserted_u_8_batch", |b| {
        b.iter(|| {
            black_box(&by_u)
                .remove_u_knots(black_box(&knots), 1e-6)
                .unwrap()
        })
    });
    group.bench_function("surface_6x6_bicubic_inserted_v_8_batch", |b| {
        b.iter(|| {
            black_box(&by_v)
                .remove_v_knots(black_box(&knots), 1e-6)
                .unwrap()
        })
    });
    group.finish();
}

criterion_group!(benches, bench_knot_removal);
criterion_main!(benches);
