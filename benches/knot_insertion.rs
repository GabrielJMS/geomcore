//! Knot insertion and refinement on B-spline curves and surfaces.
//!
//! Informational only, like the other benches: one knot, then 8 and 64
//! knots in one refinement, into clamped curves of 32 poles, and 8 knots
//! into a bicubic 6x6 surface in either direction.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use geomcore::surfaces::BSplineSurface;
use geomcore::{BSplineCurve3D, Point3D};

/// Clamped curve of 32 poles on a helix with uniform knots, rational with
/// varying weights if `rational`.
fn curve(degree: usize, rational: bool) -> BSplineCurve3D {
    let poles: Vec<Point3D> = (0..32)
        .map(|i| {
            let t = i as f64 * 0.4;
            Point3D::new(3.0 * t.cos(), 3.0 * t.sin(), 0.3 * i as f64)
        })
        .collect();
    let spans = 32 - degree;
    let knots: Vec<f64> = (0..=spans).map(|i| i as f64).collect();
    let mut mults = vec![1; spans + 1];
    mults[0] = degree as u32 + 1;
    mults[spans] = degree as u32 + 1;
    if rational {
        let weights = (0..32).map(|i| 1.0 + 0.5 * ((i % 3) as f64)).collect();
        BSplineCurve3D::new_rational(degree, poles, weights, knots, mults, false).unwrap()
    } else {
        BSplineCurve3D::new(degree, poles, knots, mults, false).unwrap()
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

fn bench_knot_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("knot_insertion");
    for (name, degree, rational) in [
        ("degree3_polynomial", 3, false),
        ("degree3_rational", 3, true),
        ("degree2_polynomial", 2, false),
    ] {
        let curve = curve(degree, rational);
        let (first, last) = curve.bounds();
        let u = first + 0.37 * (last - first);
        group.bench_function(format!("curve_{name}_insert_1"), |b| {
            b.iter(|| black_box(&curve).insert_knot(black_box(u), 1).unwrap())
        });
        for count in [8, 64] {
            let knots = spread(count, curve.bounds());
            group.bench_function(format!("curve_{name}_refine_{count}"), |b| {
                b.iter(|| black_box(&curve).refine_knots(black_box(&knots)).unwrap())
            });
        }
    }
    let surface = surface();
    let knots = spread(8, (0.0, 3.0));
    group.bench_function("surface_6x6_bicubic_refine_u_8", |b| {
        b.iter(|| {
            black_box(&surface)
                .refine_u_knots(black_box(&knots))
                .unwrap()
        })
    });
    group.bench_function("surface_6x6_bicubic_refine_v_8", |b| {
        b.iter(|| {
            black_box(&surface)
                .refine_v_knots(black_box(&knots))
                .unwrap()
        })
    });
    group.finish();
}

criterion_group!(benches, bench_knot_insertion);
criterion_main!(benches);
