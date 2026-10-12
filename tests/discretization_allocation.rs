//! Counting and filling a discretization allocate nothing: a caller can
//! reuse its buffers across curves and threads (ADR 0004). A test binary
//! of its own, as it replaces the global allocator.

use geomcore::discretization::{DiscretizationCriteria, Discretize3D};
use geomcore::{BSplineCurve3D, Circle3D, Curve3D, Ellipse3D, Point3D, TrimmedCurve3D, Vector3D};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAllocator;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn allocations_during(f: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.with(Cell::get);
    f();
    ALLOCATIONS.with(Cell::get) - before
}

/// Counts then fills into buffers sized once, checking neither step
/// allocates.
fn assert_fills_without_allocating<C: Discretize3D>(curve: &C, criteria: DiscretizationCriteria) {
    let mut n = 0;
    assert_eq!(allocations_during(|| n = curve.point_count(criteria)), 0);
    let (mut parameters, mut points) = (vec![0.0; n], vec![Point3D::ORIGIN; n]);
    let (mut chords, mut angles) = (vec![0.0; n - 1], vec![0.0; n - 1]);
    let filling = allocations_during(|| {
        curve.discretize_into(
            criteria,
            &mut parameters,
            &mut points,
            &mut chords,
            &mut angles,
        )
    });
    assert_eq!(filling, 0);
}

#[test]
fn test_count_and_fill_do_not_allocate() {
    let poles = (0..6)
        .map(|i| {
            let a = f64::from(i);
            Point3D::new(a, a.sin(), 0.5 * a.cos())
        })
        .collect();
    let weights = vec![1.0, 0.5, 2.0, 1.0, 3.0, 1.0];
    let spline = BSplineCurve3D::new_rational(
        3,
        poles,
        weights,
        vec![0.0, 1.0, 2.0, 3.0],
        vec![4, 1, 1, 4],
        false,
    )
    .unwrap();
    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.0).unwrap();
    let run_time = TrimmedCurve3D::new(Curve3D::from(spline.clone()), 0.5, 2.5).unwrap();
    let chord = DiscretizationCriteria::new(1e-4).unwrap();
    let both = chord.with_max_angle_deviation(0.05).unwrap();
    for criteria in [chord, both] {
        assert_fills_without_allocating(&spline, criteria);
        assert_fills_without_allocating(&TrimmedCurve3D::new(&spline, 0.5, 2.5).unwrap(), criteria);
        assert_fills_without_allocating(&circle, criteria);
        assert_fills_without_allocating(&TrimmedCurve3D::new(circle, 1.0, 3.0).unwrap(), criteria);
        assert_fills_without_allocating(&ellipse, criteria);
        assert_fills_without_allocating(&run_time, criteria);
    }
}
