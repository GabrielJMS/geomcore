use geomcore::{Cone, Frame3D, Hyperbola3D, Point3D, Tolerance, Vector3D};
fn main() {
    let h = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
    let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
    let tol = Tolerance::DEFAULT;
    match h.intersect_cone(&cone, tol) {
        geomcore::ConicSurfaceIntersection::Hits(hits) => {
            for h in &hits {
                println!(
                    "t={:.6} mult={} pt={:?} cone-resid={:.3e}",
                    h.parameter,
                    h.multiplicity,
                    h.point,
                    cone.parameters_of(h.point).0
                );
                let (u, v) = cone.parameters_of(h.point);
                let back = cone.eval_point(u, v);
                println!("   back={back:?} resid={:.3e}", back.distance(h.point));
            }
        }
        other => println!("{other:?}"),
    }
}
