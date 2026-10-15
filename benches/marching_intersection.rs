//! Cost of the marching intersection.
//!
//! Times the routed `intersect` on two bicubic 6×6 B-spline surfaces over
//! `[0, 5]^2`, with heights `sin(0.7 i j)` and `0.5 cos(1.3 i) cos(1.1 j)`
//! at their poles `(i, j)`. They cross in six curves: five open across the
//! footprint and one closed loop. Also times elementary pairs:
//! - a torus of radii 3 and 1 about z against the plane through
//!   `(0.5, 0, 0.2)` tilted 0.6 radians about y (two closed curves);
//! - a sphere of radius 1.2 centred on that torus's centre circle
//!   (two closed curves);
//! - a cylinder of radius 1 about z against a cubic wall extruded along z
//!   (two open curves);
//! - a cone of semi-angle 0.3 against a cylinder whose axis is parallel to
//!   one of its generators (two curves running to the model-size cube).
//!
//! And pairs where the surfaces lie on each other or touch, which the
//! coincidence check settles before marching or which trace a touching
//! curve:
//! - a sphere against its own B-spline form (coincident);
//! - a cylinder against its B-spline patch over a whole turn and
//!   `z` in `[0, 2]` (incomplete, the region covering the patch);
//! - a plane against its own bilinear patch (likewise);
//! - that B-spline sphere against itself (coincident);
//! - the plane `x = 1` against that cylinder patch, which it touches along
//!   a ruling (one curve, flagged tangent).
//!
//! Every pair has no closed form, so `intersect` runs the numeric search;
//! the bench asserts it, and asserts each degenerate pair's answer outside
//! the timed loop. Informational only, like `eval_throughput`; not a
//! correctness gate.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use geomcore::surfaces::BSplineSurface;
use geomcore::{
    Completeness, Cone, Cylinder, IncompleteReason, Intersect, IntersectionMethod, Plane, Point3D,
    Sphere, SurfaceSurfaceIntersection, Tolerance, Torus, Vector3D,
};

fn surface(height: impl Fn(f64, f64) -> f64) -> BSplineSurface {
    let poles = (0..6)
        .map(|i| {
            (0..6)
                .map(|j| {
                    let (x, y) = (i as f64, j as f64);
                    Point3D::new(x, y, height(x, y))
                })
                .collect()
        })
        .collect();
    let (knots, mults) = (vec![0.0, 1.0, 2.0, 3.0], vec![4, 1, 1, 4]);
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

/// A cubic wall waving between x = 0.1 and x = 0.4 over y in
/// `[-2.5, 2.5]`, extruded along z over `[-3, 3]`.
fn extruded_wall() -> BSplineSurface {
    let poles = (0..6)
        .map(|i| {
            let y = -2.5 + i as f64;
            let x = if i % 2 == 0 { 0.4 } else { 0.1 };
            vec![Point3D::new(x, y, -3.0), Point3D::new(x, y, 3.0)]
        })
        .collect();
    BSplineSurface::new(
        3,
        1,
        poles,
        vec![0.0, 1.0, 2.0, 3.0],
        vec![4, 1, 1, 4],
        vec![0.0, 1.0],
        vec![2, 2],
        false,
        false,
    )
    .unwrap()
}

fn assert_marched<A, B>(first: &A, second: &B, tol: Tolerance) -> SurfaceSurfaceIntersection
where
    A: Intersect<B, Output = SurfaceSurfaceIntersection>,
{
    let result = first.intersect(second, tol);
    assert_eq!(
        result.method(),
        IntersectionMethod::Numeric,
        "a closed form now exists: the bench no longer measures marching"
    );
    result
}

/// Asserts that the result reports one surface lying on the other: no
/// curve, not coincident, and an unseparated region.
fn assert_lies_on(result: &SurfaceSurfaceIntersection) {
    assert!(result.curves().is_empty() && !result.is_coincident());
    assert!(matches!(
        result.completeness(),
        Completeness::Incomplete { reason: IncompleteReason::Unseparated, regions } if regions.len() == 1
    ));
}

fn bench_marching_intersection(c: &mut Criterion) {
    let wavy = surface(|x, y| (0.7 * x * y).sin());
    let bumpy = surface(|x, y| 0.5 * (1.3 * x).cos() * (1.1 * y).cos());
    let tol = Tolerance::default();
    let mut group = c.benchmark_group("marching_intersection");
    assert_marched(&wavy, &bumpy, tol);
    group.bench_function("bspline_surface_pair", |b| {
        b.iter(|| black_box(black_box(&wavy).intersect(&bumpy, tol)))
    });
    let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 3.0, 1.0).unwrap();
    let tilt: f64 = 0.6;
    let plane = Plane::new(
        Point3D::new(0.5, 0.0, 0.2),
        Vector3D::new(-tilt.sin(), 0.0, tilt.cos()),
    )
    .unwrap();
    assert_marched(&torus, &plane, tol);
    group.bench_function("torus_plane_oblique", |b| {
        b.iter(|| black_box(black_box(&torus).intersect(&plane, tol)))
    });
    let sphere = Sphere::new(Point3D::new(3.0, 0.0, 0.0), 1.2).unwrap();
    assert_marched(&sphere, &torus, tol);
    group.bench_function("sphere_torus", |b| {
        b.iter(|| black_box(black_box(&sphere).intersect(&torus, tol)))
    });
    let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    let wall = extruded_wall();
    assert_marched(&cylinder, &wall, tol);
    group.bench_function("cylinder_extruded_wall", |b| {
        b.iter(|| black_box(black_box(&cylinder).intersect(&wall, tol)))
    });
    let cone = Cone::new(Point3D::ORIGIN, Vector3D::Z, 0.3, 1.0).unwrap();
    let generator = Vector3D::new(0.3f64.sin(), 0.0, 0.3f64.cos());
    let along_generator = Cylinder::new(Point3D::new(1.0, 0.0, 0.0), generator, 0.5).unwrap();
    assert_marched(&cone, &along_generator, tol);
    group.bench_function("cone_cylinder_unbounded", |b| {
        b.iter(|| black_box(black_box(&cone).intersect(&along_generator, tol)))
    });
    bench_degenerate_pairs(&mut group, tol);
    group.finish();
}

fn bench_degenerate_pairs(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tol: Tolerance,
) {
    let sphere = Sphere::new(Point3D::new(0.3, 0.2, 0.5), 0.8).unwrap();
    let sphere_copy = sphere.to_bspline();
    assert!(assert_marched(&sphere, &sphere_copy, tol).is_coincident());
    group.bench_function("sphere_bspline_copy", |b| {
        b.iter(|| black_box(black_box(&sphere).intersect(&sphere_copy, tol)))
    });
    let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    let cylinder_patch = cylinder
        .to_bspline_patch((0.0, std::f64::consts::TAU), (0.0, 2.0))
        .unwrap();
    assert_lies_on(&assert_marched(&cylinder, &cylinder_patch, tol));
    group.bench_function("cylinder_own_patch", |b| {
        b.iter(|| black_box(black_box(&cylinder).intersect(&cylinder_patch, tol)))
    });
    let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    let plane_patch = plane.to_bspline_patch((-1.0, 2.0), (0.0, 3.0)).unwrap();
    assert_lies_on(&assert_marched(&plane, &plane_patch, tol));
    group.bench_function("plane_own_patch", |b| {
        b.iter(|| black_box(black_box(&plane).intersect(&plane_patch, tol)))
    });
    assert!(assert_marched(&sphere_copy, &sphere_copy, tol).is_coincident());
    group.bench_function("bspline_sphere_itself", |b| {
        b.iter(|| black_box(black_box(&sphere_copy).intersect(&sphere_copy, tol)))
    });
    let touching = Plane::new(Point3D::new(1.0, 0.0, 0.0), Vector3D::X).unwrap();
    let result = assert_marched(&touching, &cylinder_patch, tol);
    assert!(result.curves().len() == 1 && result.curves()[0].tangent);
    group.bench_function("plane_tangent_to_cylinder_patch", |b| {
        b.iter(|| black_box(black_box(&touching).intersect(&cylinder_patch, tol)))
    });
}

criterion_group!(benches, bench_marching_intersection);
criterion_main!(benches);
