//! Boxes of surfaces: closed form for spheres and tori, control points for
//! B-splines; planes, cylinders and cones are unbounded.

use super::{Bound3D, BoundingBox3D, coordinates, point};
use crate::Point3D;
use crate::surfaces::{
    BSplineSurface, Cone, Cylinder, ElementarySurface, Plane, Sphere, Surface, Torus,
};

/// The box `center ± half` on each axis.
fn centered(center: Point3D, half: [f64; 3]) -> BoundingBox3D {
    let c = coordinates(center);
    BoundingBox3D {
        min: point(std::array::from_fn(|k| c[k] - half[k])),
        max: point(std::array::from_fn(|k| c[k] + half[k])),
    }
}

impl Bound3D for Plane {
    fn bounding_box(&self) -> Option<BoundingBox3D> {
        None
    }
}

impl Bound3D for Cylinder {
    fn bounding_box(&self) -> Option<BoundingBox3D> {
        None
    }
}

impl Bound3D for Cone {
    fn bounding_box(&self) -> Option<BoundingBox3D> {
        None
    }
}

impl Bound3D for Sphere {
    fn bounding_box(&self) -> Option<BoundingBox3D> {
        Some(centered(self.center(), [self.radius(); 3]))
    }
}

impl Bound3D for Torus {
    /// Along a unit vector `e`, the torus reaches `R h + r` from its
    /// center, `R` and `r` its major and minor radii and `h` the length of
    /// the projection of `e` onto the torus's plane: the center circle
    /// reaches `R h`, and the tube adds `r` around it.
    fn bounding_box(&self) -> Option<BoundingBox3D> {
        let frame = self.frame();
        let (x, y) = (frame.x_direction(), frame.y_direction());
        let (x, y) = ([x.x, x.y, x.z], [y.x, y.y, y.z]);
        let (major, minor) = (self.major_radius(), self.minor_radius());
        Some(centered(
            self.center(),
            std::array::from_fn(|k| major * x[k].hypot(y[k]) + minor),
        ))
    }
}

impl Bound3D for BSplineSurface {
    fn bounding_box(&self) -> Option<BoundingBox3D> {
        BoundingBox3D::from_points(self.poles().iter().copied())
    }
}

impl Bound3D for Surface {
    fn bounding_box(&self) -> Option<BoundingBox3D> {
        match self {
            Surface::Plane(s) => s.bounding_box(),
            Surface::Cylinder(s) => s.bounding_box(),
            Surface::Cone(s) => s.bounding_box(),
            Surface::Sphere(s) => s.bounding_box(),
            Surface::Torus(s) => s.bounding_box(),
            Surface::BSpline(s) => s.bounding_box(),
        }
    }
}

impl Bound3D for ElementarySurface {
    fn bounding_box(&self) -> Option<BoundingBox3D> {
        match self {
            ElementarySurface::Plane(s) => s.bounding_box(),
            ElementarySurface::Cylinder(s) => s.bounding_box(),
            ElementarySurface::Cone(s) => s.bounding_box(),
            ElementarySurface::Sphere(s) => s.bounding_box(),
            ElementarySurface::Torus(s) => s.bounding_box(),
        }
    }
}
