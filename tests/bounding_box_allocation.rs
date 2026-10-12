//! Bounding boxes allocate nothing, so culling many pairs costs no heap
//! traffic. A test binary of its own, as it replaces the global allocator.

use geomcore::{
    BSplineCurve3D, BSplineSurface, Bound3D, Circle3D, Curve3D, Ellipse3D, Hyperbola3D, Parabola3D,
    Point3D, Sphere, Surface, Torus, TrimmedCurve3D, Vector3D,
};
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

fn assert_boxes_without_allocating(shape: &impl Bound3D) {
    let before = ALLOCATIONS.with(Cell::get);
    let b = shape.bounding_box();
    assert_eq!(ALLOCATIONS.with(Cell::get) - before, 0);
    assert!(b.is_some());
}

#[test]
fn test_boxes_do_not_allocate() {
    let n = Vector3D::new(0.3, -0.5, 0.8);
    let x = n.cross(Vector3D::X);
    let circle = Circle3D::new(Point3D::ORIGIN, n, 2.0).unwrap();
    let ellipse = Ellipse3D::new(Point3D::ORIGIN, n, x, 3.0, 1.0).unwrap();
    let parabola = Parabola3D::new(Point3D::ORIGIN, n, x, 0.5).unwrap();
    let hyperbola = Hyperbola3D::new(Point3D::ORIGIN, n, x, 1.0, 2.0).unwrap();
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
    let net = vec![
        vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 1.0)],
        vec![Point3D::new(1.0, 0.0, 2.0), Point3D::new(1.0, 1.0, 0.0)],
    ];
    let surface = BSplineSurface::new(
        1,
        1,
        net,
        vec![0.0, 1.0],
        vec![2, 2],
        vec![0.0, 1.0],
        vec![2, 2],
        false,
        false,
    )
    .unwrap();

    assert_boxes_without_allocating(&circle);
    assert_boxes_without_allocating(&ellipse);
    assert_boxes_without_allocating(&TrimmedCurve3D::new(circle, 5.0, 7.0).unwrap());
    assert_boxes_without_allocating(&TrimmedCurve3D::new(ellipse, 1.0, 3.0).unwrap());
    assert_boxes_without_allocating(&TrimmedCurve3D::new(parabola, -2.0, 3.0).unwrap());
    assert_boxes_without_allocating(&TrimmedCurve3D::new(hyperbola, -1.0, 2.0).unwrap());
    assert_boxes_without_allocating(
        &TrimmedCurve3D::segment(Point3D::ORIGIN, Point3D::ORIGIN + n).unwrap(),
    );
    assert_boxes_without_allocating(&spline);
    assert_boxes_without_allocating(&TrimmedCurve3D::new(&spline, 0.5, 2.5).unwrap());
    let run_time = TrimmedCurve3D::new(Curve3D::from(spline.clone()), 0.5, 2.5).unwrap();
    assert_boxes_without_allocating(&run_time);
    assert_boxes_without_allocating(&Sphere::new(Point3D::ORIGIN, 1.0).unwrap());
    assert_boxes_without_allocating(&Torus::new(Point3D::ORIGIN, n, 3.0, 1.0).unwrap());
    assert_boxes_without_allocating(&surface);
    assert_boxes_without_allocating(&Surface::from(surface.clone()));
}
