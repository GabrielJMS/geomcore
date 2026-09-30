use geomcore::{Cone, Frame3D, Hyperbola3D, Point3D, Tolerance, Vector3D};
fn main() {
    let h = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
    let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
    let tol = Tolerance::DEFAULT;
    println!(
        "apex={:?} axis={:?}",
        cone.apex(),
        cone.frame().z_direction()
    );
    let v = h.eval_point(0.0);
    println!("vertex={v:?} on-cone={}", cone.contains(v, tol));
    println!("result={:?}", h.intersect_cone(&cone, tol));
}
