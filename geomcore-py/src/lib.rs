//! Python bindings for the `geomcore` geometric kernel.
//!
//! The extension module mirrors the Rust crate's namespaces: value types live
//! at the package root (`geomcore`), curves under `geomcore.curves`, and
//! surfaces under `geomcore.surfaces`. Fallible constructors raise
//! `ValueError` carrying the underlying error description.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use geomcore::curves::{
    BSplineCurve3D, Circle2D, Circle3D, Curve2D, Ellipse3D, Hyperbola3D, Line2D, Line3D, Parabola3D,
};
use geomcore::surfaces::{BSplineSurface, Cone, Cylinder, Plane, Sphere, Surface, Torus};
use geomcore::{
    Axis3D, ConeConeIntersection, ConeCylinderIntersection, CylinderCylinderIntersection, Frame3D,
    LinePlaneIntersection, LineQuadricIntersection, PlaneConeIntersection,
    PlaneCylinderIntersection, PlanePlaneIntersection, PlaneSphereIntersection, Point2D, Point3D,
    SphereConeIntersection, SphereCylinderIntersection, SphereSphereIntersection, Tolerance,
    TorusConeIntersection, TorusCylinderIntersection, TorusPlaneIntersection,
    TorusSphereIntersection, TorusTorusIntersection, Transform, Vector2D, Vector3D,
};

fn val_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Root value types
// ---------------------------------------------------------------------------

/// A point in 3D space with `x`, `y` and `z` coordinates.
#[pyclass(name = "Point3D", module = "geomcore")]
#[derive(Clone)]
struct PyPoint3D(Point3D);

#[pymethods]
impl PyPoint3D {
    #[new]
    /// Build a point from its three coordinates.
    fn py_new(x: f64, y: f64, z: f64) -> Self {
        PyPoint3D(Point3D::new(x, y, z))
    }

    /// The origin (0, 0, 0).
    #[staticmethod]
    fn origin() -> Self {
        PyPoint3D(Point3D::ORIGIN)
    }

    #[getter]
    fn x(&self) -> f64 {
        self.0.x
    }

    #[getter]
    fn y(&self) -> f64 {
        self.0.y
    }

    #[getter]
    fn z(&self) -> f64 {
        self.0.z
    }

    /// Euclidean distance to another point.
    fn distance(&self, other: &PyPoint3D) -> f64 {
        self.0.distance(other.0)
    }

    fn __repr__(&self) -> String {
        format!("Point3D({}, {}, {})", self.0.x, self.0.y, self.0.z)
    }
}

/// A vector in 3D space.
///
/// Components are read with `components()`; the unit vectors along the world
/// axes are available as the static methods `x()`, `y()` and `z()`.
#[pyclass(name = "Vector3D", module = "geomcore")]
#[derive(Clone)]
struct PyVector3D(Vector3D);

#[pymethods]
impl PyVector3D {
    #[new]
    /// Build a vector from its three components.
    fn py_new(x: f64, y: f64, z: f64) -> Self {
        PyVector3D(Vector3D::new(x, y, z))
    }

    /// The zero vector.
    #[staticmethod]
    fn zero() -> Self {
        PyVector3D(Vector3D::ZERO)
    }

    /// The unit vector along the world x axis.
    #[staticmethod]
    fn x() -> Self {
        PyVector3D(Vector3D::X)
    }

    /// The unit vector along the world y axis.
    #[staticmethod]
    fn y() -> Self {
        PyVector3D(Vector3D::Y)
    }

    /// The unit vector along the world z axis.
    #[staticmethod]
    fn z() -> Self {
        PyVector3D(Vector3D::Z)
    }

    /// The `(x, y, z)` components as a tuple.
    fn components(&self) -> (f64, f64, f64) {
        (self.0.x, self.0.y, self.0.z)
    }

    /// Dot product with another vector.
    fn dot(&self, other: &PyVector3D) -> f64 {
        self.0.dot(other.0)
    }

    /// Cross product with another vector.
    fn cross(&self, other: &PyVector3D) -> PyVector3D {
        PyVector3D(self.0.cross(other.0))
    }

    /// Euclidean length of the vector.
    fn magnitude(&self) -> f64 {
        self.0.magnitude()
    }

    fn __repr__(&self) -> String {
        format!("Vector3D({}, {}, {})", self.0.x, self.0.y, self.0.z)
    }
}

/// A point in 2D space (used by curves living in a surface's (u, v) domain).
#[pyclass(name = "Point2D", module = "geomcore")]
#[derive(Clone)]
struct PyPoint2D(Point2D);

#[pymethods]
impl PyPoint2D {
    #[new]
    /// Build a point from its two coordinates.
    fn py_new(x: f64, y: f64) -> Self {
        PyPoint2D(Point2D::new(x, y))
    }

    /// The origin (0, 0).
    #[staticmethod]
    fn origin() -> Self {
        PyPoint2D(Point2D::ORIGIN)
    }

    #[getter]
    fn x(&self) -> f64 {
        self.0.x
    }

    #[getter]
    fn y(&self) -> f64 {
        self.0.y
    }

    /// Euclidean distance to another point.
    fn distance(&self, other: &PyPoint2D) -> f64 {
        self.0.distance(other.0)
    }

    fn __repr__(&self) -> String {
        format!("Point2D({}, {})", self.0.x, self.0.y)
    }
}

/// A vector in 2D space.
#[pyclass(name = "Vector2D", module = "geomcore")]
#[derive(Clone)]
struct PyVector2D(Vector2D);

#[pymethods]
impl PyVector2D {
    #[new]
    /// Build a vector from its two components.
    fn py_new(x: f64, y: f64) -> Self {
        PyVector2D(Vector2D::new(x, y))
    }

    #[getter]
    fn x(&self) -> f64 {
        self.0.x
    }

    #[getter]
    fn y(&self) -> f64 {
        self.0.y
    }

    /// Dot product with another vector.
    fn dot(&self, other: &PyVector2D) -> f64 {
        self.0.dot(other.0)
    }

    /// Euclidean length of the vector.
    fn magnitude(&self) -> f64 {
        self.0.magnitude()
    }

    fn __repr__(&self) -> String {
        format!("Vector2D({}, {})", self.0.x, self.0.y)
    }
}

/// An axis: a point plus a unit direction.
#[pyclass(name = "Axis3D", module = "geomcore")]
#[derive(Clone)]
struct PyAxis3D(Axis3D);

#[pymethods]
impl PyAxis3D {
    #[new]
    /// Build an axis from an origin and a direction (normalized internally).
    fn py_new(origin: PyPoint3D, direction: PyVector3D) -> PyResult<Self> {
        Ok(PyAxis3D(
            Axis3D::new(origin.0, direction.0).map_err(val_err)?,
        ))
    }

    /// The axis origin.
    fn origin(&self) -> PyPoint3D {
        PyPoint3D(self.0.origin())
    }

    /// The unit direction of the axis.
    fn direction(&self) -> PyVector3D {
        PyVector3D(self.0.direction())
    }
}

/// A right-handed orthonormal placement frame (origin + x/y/z directions).
#[pyclass(name = "Frame3D", module = "geomcore")]
#[derive(Clone)]
struct PyFrame3D(Frame3D);

#[pymethods]
impl PyFrame3D {
    #[new]
    /// Build a frame from an origin, main (z) direction, and an x hint.
    fn py_new(origin: PyPoint3D, z_direction: PyVector3D, x_hint: PyVector3D) -> PyResult<Self> {
        Ok(PyFrame3D(
            Frame3D::new(origin.0, z_direction.0, x_hint.0).map_err(val_err)?,
        ))
    }

    /// Build a frame from an origin and main (z) direction, deriving an
    /// arbitrary perpendicular x direction.
    #[staticmethod]
    fn from_z(origin: PyPoint3D, z_direction: PyVector3D) -> PyResult<Self> {
        Ok(PyFrame3D(
            Frame3D::from_z(origin.0, z_direction.0).map_err(val_err)?,
        ))
    }

    /// The world frame at the origin.
    #[staticmethod]
    fn world() -> Self {
        PyFrame3D(Frame3D::WORLD)
    }

    /// The frame origin.
    fn origin(&self) -> PyPoint3D {
        PyPoint3D(self.0.origin())
    }

    /// The unit x direction.
    fn x_direction(&self) -> PyVector3D {
        PyVector3D(self.0.x_direction())
    }

    /// The unit y direction.
    fn y_direction(&self) -> PyVector3D {
        PyVector3D(self.0.y_direction())
    }

    /// The unit z direction.
    fn z_direction(&self) -> PyVector3D {
        PyVector3D(self.0.z_direction())
    }
}

/// Distance and angle tolerances shared by all geometric queries
/// (`contains`, and later projection and intersection).
///
/// `Tolerance()` with no arguments equals `Tolerance.default()`.
#[pyclass(name = "Tolerance", module = "geomcore")]
#[derive(Clone)]
struct PyTolerance(Tolerance);

#[pymethods]
impl PyTolerance {
    #[new]
    #[pyo3(signature = (confusion = 1e-7, angular = 1e-12, parametric = 1e-9))]
    /// Build tolerances from the three scales explicitly.
    fn py_new(confusion: f64, angular: f64, parametric: f64) -> Self {
        PyTolerance(Tolerance::new(confusion, angular, parametric))
    }

    /// The default tolerances (confusion 1e-7, angular 1e-12, parametric 1e-9).
    #[staticmethod]
    fn default() -> Self {
        PyTolerance(Tolerance::DEFAULT)
    }

    /// Distance below which two points are considered coincident.
    #[getter]
    fn confusion(&self) -> f64 {
        self.0.confusion
    }

    /// Angular tolerance for parallelism/orthogonality checks (radians).
    #[getter]
    fn angular(&self) -> f64 {
        self.0.angular
    }

    /// Parametric-space tolerance.
    #[getter]
    fn parametric(&self) -> f64 {
        self.0.parametric
    }

    fn __repr__(&self) -> String {
        format!(
            "Tolerance(confusion={}, angular={}, parametric={})",
            self.0.confusion, self.0.angular, self.0.parametric
        )
    }
}

/// A rigid (or mirrored/scaled) affine transformation.
#[pyclass(name = "Transform", module = "geomcore")]
#[derive(Clone)]
struct PyTransform(Transform);

#[pymethods]
impl PyTransform {
    /// The identity transformation.
    #[staticmethod]
    fn identity() -> Self {
        PyTransform(Transform::IDENTITY)
    }

    /// Translation by an offset vector.
    #[staticmethod]
    fn translation(offset: PyVector3D) -> Self {
        PyTransform(Transform::translation(offset.0))
    }

    /// Rotation by `angle` radians about `axis` (right-hand rule).
    #[staticmethod]
    fn rotation(axis: PyAxis3D, angle: f64) -> Self {
        PyTransform(Transform::rotation(axis.0, angle))
    }

    /// Uniform scaling about `center`.
    #[staticmethod]
    fn scaling(center: PyPoint3D, factor: f64) -> Self {
        PyTransform(Transform::scaling(center.0, factor))
    }

    /// Point reflection through `center`.
    #[staticmethod]
    fn mirror_point(center: PyPoint3D) -> Self {
        PyTransform(Transform::mirror_point(center.0))
    }

    /// Reflection about an axis (rotation by pi around the line).
    #[staticmethod]
    fn mirror_axis(axis: PyAxis3D) -> Self {
        PyTransform(Transform::mirror_axis(axis.0))
    }

    /// Reflection across the plane through the frame origin with normal
    /// equal to the frame's z direction.
    #[staticmethod]
    fn mirror_plane(frame: PyFrame3D) -> Self {
        PyTransform(Transform::mirror_plane(frame.0))
    }

    /// Compose transformations: `self` is applied first, then `next`.
    fn then(&self, next: &PyTransform) -> PyTransform {
        PyTransform(self.0.then(next.0))
    }

    /// Apply the transformation to a point.
    fn apply_point(&self, p: PyPoint3D) -> PyPoint3D {
        PyPoint3D(self.0.apply_point(p.0))
    }

    /// Apply the transformation to a vector (ignores the translation part).
    fn apply_vector(&self, v: PyVector3D) -> PyVector3D {
        PyVector3D(self.0.apply_vector(v.0))
    }
}

// ---------------------------------------------------------------------------
// Surfaces
// ---------------------------------------------------------------------------

/// An infinite plane, parametrized by in-plane coordinates (u, v).
#[pyclass(name = "Plane", module = "geomcore.surfaces")]
#[derive(Clone)]
struct PyPlane(Plane);

#[pymethods]
impl PyPlane {
    #[new]
    /// Build a plane from a point and a normal direction.
    fn py_new(point: PyPoint3D, normal: PyVector3D) -> PyResult<Self> {
        Ok(PyPlane(Plane::new(point.0, normal.0).map_err(val_err)?))
    }

    /// Build a plane from a placement frame (normal = frame z direction).
    #[staticmethod]
    fn from_frame(frame: PyFrame3D) -> Self {
        PyPlane(Plane::from_frame(frame.0))
    }

    /// Build a plane through three non-collinear points.
    #[staticmethod]
    fn from_three_points(p1: PyPoint3D, p2: PyPoint3D, p3: PyPoint3D) -> PyResult<Self> {
        Ok(PyPlane(
            Plane::from_three_points(p1.0, p2.0, p3.0).map_err(val_err)?,
        ))
    }

    /// Build a plane from the equation `a*x + b*y + c*z + d = 0`.
    #[staticmethod]
    fn from_coefficients(a: f64, b: f64, c: f64, d: f64) -> PyResult<Self> {
        Ok(PyPlane(
            Plane::from_coefficients(a, b, c, d).map_err(val_err)?,
        ))
    }

    /// The unit normal of the plane.
    fn normal(&self) -> PyVector3D {
        PyVector3D(self.0.normal())
    }

    /// Evaluate the point at surface parameters (u, v).
    fn eval_point(&self, u: f64, v: f64) -> PyPoint3D {
        PyPoint3D(self.0.eval_point(u, v))
    }

    /// Evaluate many (u, v) pairs at once.
    fn eval_points(&self, uvs: Vec<(f64, f64)>) -> Vec<PyPoint3D> {
        self.0
            .eval_points(&uvs)
            .into_iter()
            .map(PyPoint3D)
            .collect()
    }

    /// Evaluate the partial derivative of order (du, dv), with 1 <= du+dv <= 2.
    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> PyVector3D {
        PyVector3D(self.0.eval_derivative(u, v, du, dv))
    }

    /// Recover the (u, v) parameters of a point lying on the plane.
    fn parameters_of(&self, point: PyPoint3D) -> (f64, f64) {
        self.0.parameters_of(point.0)
    }

    /// Returns whether `point` lies on the surface within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0
            .contains(point.0, tol.map(|t| t.0).unwrap_or_default())
    }

    /// Projects `point` onto the surface, returning `(u, v, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64, f64) {
        let proj = self
            .0
            .project_point(point.0, tol.map(|t| t.0).unwrap_or_default());
        (proj.u, proj.v, proj.distance)
    }

    /// Projects each point in `points` onto the surface, returning a
    /// `(u, v, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(
        &self,
        points: Vec<PyPoint3D>,
        tol: Option<PyTolerance>,
    ) -> Vec<(f64, f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tol.map(|t| t.0).unwrap_or_default();
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.u, p.v, p.distance))
            .collect()
    }

    /// Intersects this plane with another plane.
    ///
    /// Returns `("line", Line3D)`, `("parallel", None)` or
    /// `("coincident", None)`.
    #[pyo3(signature = (other, tol = None))]
    fn intersect_plane(
        &self,
        py: Python<'_>,
        other: &PyPlane,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        match self.0.intersect_plane(&other.0, tol) {
            PlanePlaneIntersection::Line(l) => Ok((
                "line".to_string(),
                PyLine3D(l).into_pyobject(py)?.into_any().unbind(),
            )),
            PlanePlaneIntersection::Parallel => Ok(("parallel".to_string(), py.None())),
            PlanePlaneIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
        }
    }

    /// Intersects this plane with a sphere.
    ///
    /// Returns `("circle", Circle3D)`, `("tangent_point", Point3D)` or
    /// `("empty", None)`.
    #[pyo3(signature = (sphere, tol = None))]
    fn intersect_sphere(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        match self.0.intersect_sphere(&sphere.0, tol) {
            PlaneSphereIntersection::Circle(c) => Ok((
                "circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            PlaneSphereIntersection::TangentPoint(p) => Ok((
                "tangent_point".to_string(),
                PyPoint3D(p).into_pyobject(py)?.into_any().unbind(),
            )),
            PlaneSphereIntersection::Empty => Ok(("empty".to_string(), py.None())),
        }
    }

    /// Intersects this plane with a cylinder.
    ///
    /// Returns `("circle", Circle3D)`, `("ellipse", Ellipse3D)`,
    /// `("two_lines", (Line3D, Line3D))`, `("tangent_line", Line3D)` or
    /// `("empty", None)`.
    #[pyo3(signature = (cylinder, tol = None))]
    fn intersect_cylinder(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        match self.0.intersect_cylinder(&cylinder.0, tol) {
            PlaneCylinderIntersection::Circle(c) => Ok((
                "circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            PlaneCylinderIntersection::Ellipse(e) => Ok((
                "ellipse".to_string(),
                PyEllipse3D(e).into_pyobject(py)?.into_any().unbind(),
            )),
            PlaneCylinderIntersection::TwoLines(l1, l2) => Ok((
                "two_lines".to_string(),
                (
                    PyLine3D(l1).into_pyobject(py)?.into_any().unbind(),
                    PyLine3D(l2).into_pyobject(py)?.into_any().unbind(),
                )
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            )),
            PlaneCylinderIntersection::TangentLine(l) => Ok((
                "tangent_line".to_string(),
                PyLine3D(l).into_pyobject(py)?.into_any().unbind(),
            )),
            PlaneCylinderIntersection::Empty => Ok(("empty".to_string(), py.None())),
        }
    }

    /// Intersects this plane with a cone.
    ///
    /// Returns `("circle", Circle3D)`, `("ellipse", Ellipse3D)`,
    /// `("parabola", Parabola3D)`, `("hyperbola", Hyperbola3D)`,
    /// `("two_lines", (Line3D, Line3D))`, `("tangent_line", Line3D)`,
    /// `("apex_point", Point3D)` or `("empty", None)`.
    #[pyo3(signature = (cone, tol = None))]
    fn intersect_cone(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        match self.0.intersect_cone(&cone.0, tol) {
            PlaneConeIntersection::Circle(c) => Ok((
                "circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            PlaneConeIntersection::Ellipse(e) => Ok((
                "ellipse".to_string(),
                PyEllipse3D(e).into_pyobject(py)?.into_any().unbind(),
            )),
            PlaneConeIntersection::Parabola(p) => Ok((
                "parabola".to_string(),
                PyParabola3D(p).into_pyobject(py)?.into_any().unbind(),
            )),
            PlaneConeIntersection::Hyperbola(h) => Ok((
                "hyperbola".to_string(),
                PyHyperbola3D(h).into_pyobject(py)?.into_any().unbind(),
            )),
            PlaneConeIntersection::TwoLines(l1, l2) => Ok((
                "two_lines".to_string(),
                (
                    PyLine3D(l1).into_pyobject(py)?.into_any().unbind(),
                    PyLine3D(l2).into_pyobject(py)?.into_any().unbind(),
                )
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            )),
            PlaneConeIntersection::TangentLine(l) => Ok((
                "tangent_line".to_string(),
                PyLine3D(l).into_pyobject(py)?.into_any().unbind(),
            )),
            PlaneConeIntersection::ApexPoint(p) => Ok((
                "apex_point".to_string(),
                PyPoint3D(p).into_pyobject(py)?.into_any().unbind(),
            )),
            PlaneConeIntersection::Empty => Ok(("empty".to_string(), py.None())),
        }
    }
}

/// An infinite circular cylinder; u is the angle around the axis, v the
/// height along it.
#[pyclass(name = "Cylinder", module = "geomcore.surfaces")]
#[derive(Clone)]
struct PyCylinder(Cylinder);

#[pymethods]
impl PyCylinder {
    #[new]
    /// Build a cylinder from a point on its axis, the axis direction and a radius.
    fn py_new(center: PyPoint3D, axis_direction: PyVector3D, radius: f64) -> PyResult<Self> {
        Ok(PyCylinder(
            Cylinder::new(center.0, axis_direction.0, radius).map_err(val_err)?,
        ))
    }

    /// Build a cylinder from a placement frame (axis = frame z direction).
    #[staticmethod]
    fn from_frame(frame: PyFrame3D, radius: f64) -> PyResult<Self> {
        Ok(PyCylinder(
            Cylinder::from_frame(frame.0, radius).map_err(val_err)?,
        ))
    }

    /// Build a cylinder from an axis and a radius.
    #[staticmethod]
    fn from_axis(axis: PyAxis3D, radius: f64) -> PyResult<Self> {
        Ok(PyCylinder(
            Cylinder::from_axis(axis.0, radius).map_err(val_err)?,
        ))
    }

    /// Build the cylinder containing a circle (axis through the circle
    /// center, along its normal).
    #[staticmethod]
    fn from_circle(circle: &PyCircle3D) -> Self {
        PyCylinder(Cylinder::from_circle(&circle.0))
    }

    /// The cylinder radius.
    fn radius(&self) -> f64 {
        self.0.radius()
    }

    /// Evaluate the point at surface parameters (u, v).
    fn eval_point(&self, u: f64, v: f64) -> PyPoint3D {
        PyPoint3D(self.0.eval_point(u, v))
    }

    /// Evaluate many (u, v) pairs at once.
    fn eval_points(&self, uvs: Vec<(f64, f64)>) -> Vec<PyPoint3D> {
        self.0
            .eval_points(&uvs)
            .into_iter()
            .map(PyPoint3D)
            .collect()
    }

    /// Evaluate the partial derivative of order (du, dv), with 1 <= du+dv <= 2.
    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> PyVector3D {
        PyVector3D(self.0.eval_derivative(u, v, du, dv))
    }

    /// Recover the (u, v) parameters of a point lying on the cylinder.
    fn parameters_of(&self, point: PyPoint3D) -> (f64, f64) {
        self.0.parameters_of(point.0)
    }

    /// Returns whether `point` lies on the surface within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0
            .contains(point.0, tol.map(|t| t.0).unwrap_or_default())
    }

    /// Projects `point` onto the surface, returning `(u, v, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64, f64) {
        let proj = self
            .0
            .project_point(point.0, tol.map(|t| t.0).unwrap_or_default());
        (proj.u, proj.v, proj.distance)
    }

    /// Projects each point in `points` onto the surface, returning a
    /// `(u, v, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(
        &self,
        points: Vec<PyPoint3D>,
        tol: Option<PyTolerance>,
    ) -> Vec<(f64, f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tol.map(|t| t.0).unwrap_or_default();
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.u, p.v, p.distance))
            .collect()
    }

    /// Intersects this cylinder with another cylinder.
    ///
    /// The analytic path needs parallel axes. Returns
    /// `("two_lines", (Line3D, Line3D))`, `("tangent_line", Line3D)`,
    /// `("empty", None)`, `("coincident", None)` or
    /// `("not_analytic", None)` (no closed form; numeric path pending).
    #[pyo3(signature = (other, tol = None))]
    fn intersect_cylinder(
        &self,
        py: Python<'_>,
        other: &PyCylinder,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        match self.0.intersect_cylinder(&other.0, tol) {
            CylinderCylinderIntersection::TwoLines(l1, l2) => Ok((
                "two_lines".to_string(),
                (
                    PyLine3D(l1).into_pyobject(py)?.into_any().unbind(),
                    PyLine3D(l2).into_pyobject(py)?.into_any().unbind(),
                )
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            )),
            CylinderCylinderIntersection::TangentLine(l) => Ok((
                "tangent_line".to_string(),
                PyLine3D(l).into_pyobject(py)?.into_any().unbind(),
            )),
            CylinderCylinderIntersection::Empty => Ok(("empty".to_string(), py.None())),
            CylinderCylinderIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
            CylinderCylinderIntersection::NotAnalytic => {
                Ok(("not_analytic".to_string(), py.None()))
            }
        }
    }
}

/// An infinite cone; u is the angle around the axis, v the distance along a
/// generator from the reference circle.
#[pyclass(name = "Cone", module = "geomcore.surfaces")]
#[derive(Clone)]
struct PyCone(Cone);

#[pymethods]
impl PyCone {
    #[new]
    /// Build a cone from a reference-circle center, axis direction, half-angle and reference radius.
    fn py_new(
        center: PyPoint3D,
        axis_direction: PyVector3D,
        semi_angle: f64,
        ref_radius: f64,
    ) -> PyResult<Self> {
        Ok(PyCone(
            Cone::new(center.0, axis_direction.0, semi_angle, ref_radius).map_err(val_err)?,
        ))
    }

    /// Build a cone from a placement frame, half-angle and reference radius.
    #[staticmethod]
    fn from_frame(frame: PyFrame3D, semi_angle: f64, ref_radius: f64) -> PyResult<Self> {
        Ok(PyCone(
            Cone::from_frame(frame.0, semi_angle, ref_radius).map_err(val_err)?,
        ))
    }

    /// Build a cone through two circular sections given by axis points and radii.
    #[staticmethod]
    fn from_two_points_and_radii(p1: PyPoint3D, p2: PyPoint3D, r1: f64, r2: f64) -> PyResult<Self> {
        Ok(PyCone(
            Cone::from_two_points_and_radii(p1.0, p2.0, r1, r2).map_err(val_err)?,
        ))
    }

    /// The cone apex.
    fn apex(&self) -> PyPoint3D {
        PyPoint3D(self.0.apex())
    }

    /// The half-angle in radians.
    fn semi_angle(&self) -> f64 {
        self.0.semi_angle()
    }

    /// The radius of the reference section at v = 0.
    fn ref_radius(&self) -> f64 {
        self.0.ref_radius()
    }

    /// Evaluate the point at surface parameters (u, v).
    fn eval_point(&self, u: f64, v: f64) -> PyPoint3D {
        PyPoint3D(self.0.eval_point(u, v))
    }

    /// Evaluate many (u, v) pairs at once.
    fn eval_points(&self, uvs: Vec<(f64, f64)>) -> Vec<PyPoint3D> {
        self.0
            .eval_points(&uvs)
            .into_iter()
            .map(PyPoint3D)
            .collect()
    }

    /// Evaluate the partial derivative of order (du, dv), with 1 <= du+dv <= 2.
    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> PyVector3D {
        PyVector3D(self.0.eval_derivative(u, v, du, dv))
    }

    /// Recover the (u, v) parameters of a point lying on the cone.
    fn parameters_of(&self, point: PyPoint3D) -> (f64, f64) {
        self.0.parameters_of(point.0)
    }

    /// Returns whether `point` lies on the surface within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0
            .contains(point.0, tol.map(|t| t.0).unwrap_or_default())
    }

    /// Projects `point` onto the surface, returning `(u, v, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64, f64) {
        let proj = self
            .0
            .project_point(point.0, tol.map(|t| t.0).unwrap_or_default());
        (proj.u, proj.v, proj.distance)
    }

    /// Projects each point in `points` onto the surface, returning a
    /// `(u, v, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(
        &self,
        points: Vec<PyPoint3D>,
        tol: Option<PyTolerance>,
    ) -> Vec<(f64, f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tol.map(|t| t.0).unwrap_or_default();
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.u, p.v, p.distance))
            .collect()
    }

    /// Intersects this cone with a cylinder.
    ///
    /// The analytic path needs coaxial axes. Returns `("circle", Circle3D)`,
    /// `("apex_point", Point3D)` or `("not_analytic", None)` (no closed
    /// form; numeric path pending).
    #[pyo3(signature = (cylinder, tol = None))]
    fn intersect_cylinder(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        match self.0.intersect_cylinder(&cylinder.0, tol) {
            ConeCylinderIntersection::Circle(c) => Ok((
                "circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            ConeCylinderIntersection::ApexPoint(p) => Ok((
                "apex_point".to_string(),
                PyPoint3D(p).into_pyobject(py)?.into_any().unbind(),
            )),
            ConeCylinderIntersection::NotAnalytic => Ok(("not_analytic".to_string(), py.None())),
        }
    }

    /// Intersects this cone with another cone.
    ///
    /// The analytic path needs coaxial axes. Returns `("circle", Circle3D)`,
    /// `("apex_point", Point3D)`, `("empty", None)`,
    /// `("coincident", None)` or `("not_analytic", None)`.
    #[pyo3(signature = (other, tol = None))]
    fn intersect_cone(
        &self,
        py: Python<'_>,
        other: &PyCone,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        match self.0.intersect_cone(&other.0, tol) {
            ConeConeIntersection::Circle(c) => Ok((
                "circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            ConeConeIntersection::ApexPoint(p) => Ok((
                "apex_point".to_string(),
                PyPoint3D(p).into_pyobject(py)?.into_any().unbind(),
            )),
            ConeConeIntersection::Empty => Ok(("empty".to_string(), py.None())),
            ConeConeIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
            ConeConeIntersection::NotAnalytic => Ok(("not_analytic".to_string(), py.None())),
        }
    }
}

/// A sphere; u is the longitude in [0, 2*pi), v the latitude in [-pi/2, pi/2].
#[pyclass(name = "Sphere", module = "geomcore.surfaces")]
#[derive(Clone)]
struct PySphere(Sphere);

#[pymethods]
impl PySphere {
    #[new]
    /// Build a sphere from its center and radius (world-aligned frame).
    fn py_new(center: PyPoint3D, radius: f64) -> PyResult<Self> {
        Ok(PySphere(Sphere::new(center.0, radius).map_err(val_err)?))
    }

    /// Build a sphere from a placement frame and radius.
    #[staticmethod]
    fn from_frame(frame: PyFrame3D, radius: f64) -> PyResult<Self> {
        Ok(PySphere(
            Sphere::from_frame(frame.0, radius).map_err(val_err)?,
        ))
    }

    /// The sphere center.
    fn center(&self) -> PyPoint3D {
        PyPoint3D(self.0.center())
    }

    /// The sphere radius.
    fn radius(&self) -> f64 {
        self.0.radius()
    }

    /// Evaluate the point at surface parameters (u, v).
    fn eval_point(&self, u: f64, v: f64) -> PyPoint3D {
        PyPoint3D(self.0.eval_point(u, v))
    }

    /// Evaluate many (u, v) pairs at once.
    fn eval_points(&self, uvs: Vec<(f64, f64)>) -> Vec<PyPoint3D> {
        self.0
            .eval_points(&uvs)
            .into_iter()
            .map(PyPoint3D)
            .collect()
    }

    /// Evaluate the partial derivative of order (du, dv), with 1 <= du+dv <= 2.
    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> PyVector3D {
        PyVector3D(self.0.eval_derivative(u, v, du, dv))
    }

    /// Recover the (u, v) parameters of a point lying on the sphere.
    fn parameters_of(&self, point: PyPoint3D) -> (f64, f64) {
        self.0.parameters_of(point.0)
    }

    /// Returns whether `point` lies on the surface within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0
            .contains(point.0, tol.map(|t| t.0).unwrap_or_default())
    }

    /// Projects `point` onto the surface, returning `(u, v, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64, f64) {
        let proj = self
            .0
            .project_point(point.0, tol.map(|t| t.0).unwrap_or_default());
        (proj.u, proj.v, proj.distance)
    }

    /// Projects each point in `points` onto the surface, returning a
    /// `(u, v, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(
        &self,
        points: Vec<PyPoint3D>,
        tol: Option<PyTolerance>,
    ) -> Vec<(f64, f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tol.map(|t| t.0).unwrap_or_default();
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.u, p.v, p.distance))
            .collect()
    }

    /// Intersects this sphere with another sphere.
    ///
    /// Returns `("circle", Circle3D)`, `("tangent_point", Point3D)`,
    /// `("empty", None)` or `("coincident", None)`.
    #[pyo3(signature = (other, tol = None))]
    fn intersect_sphere(
        &self,
        py: Python<'_>,
        other: &PySphere,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        match self.0.intersect_sphere(&other.0, tol) {
            SphereSphereIntersection::Circle(c) => Ok((
                "circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            SphereSphereIntersection::TangentPoint(p) => Ok((
                "tangent_point".to_string(),
                PyPoint3D(p).into_pyobject(py)?.into_any().unbind(),
            )),
            SphereSphereIntersection::Empty => Ok(("empty".to_string(), py.None())),
            SphereSphereIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
        }
    }

    /// Intersects this sphere with a cylinder.
    ///
    /// The analytic path needs the cylinder axis through the sphere
    /// center. Returns `("circle", Circle3D)`,
    /// `("two_circles", (Circle3D, Circle3D))`, `("empty", None)` or
    /// `("not_analytic", None)` (no closed form; numeric path pending).
    #[pyo3(signature = (cylinder, tol = None))]
    fn intersect_cylinder(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        match self.0.intersect_cylinder(&cylinder.0, tol) {
            SphereCylinderIntersection::Circle(c) => Ok((
                "circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            SphereCylinderIntersection::TwoCircles(c1, c2) => Ok((
                "two_circles".to_string(),
                (
                    PyCircle3D(c1).into_pyobject(py)?.into_any().unbind(),
                    PyCircle3D(c2).into_pyobject(py)?.into_any().unbind(),
                )
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            )),
            SphereCylinderIntersection::Empty => Ok(("empty".to_string(), py.None())),
            SphereCylinderIntersection::NotAnalytic => Ok(("not_analytic".to_string(), py.None())),
        }
    }

    /// Intersects this sphere with a cone.
    ///
    /// The analytic path needs the sphere center on the cone axis.
    /// Returns `("circle", Circle3D)`,
    /// `("two_circles", (Circle3D, Circle3D))`,
    /// `("tangent_circle", Circle3D)`, `("empty", None)` or
    /// `("not_analytic", None)` (no closed form; numeric path pending).
    #[pyo3(signature = (cone, tol = None))]
    fn intersect_cone(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        match self.0.intersect_cone(&cone.0, tol) {
            SphereConeIntersection::Circle(c) => Ok((
                "circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            SphereConeIntersection::TwoCircles(c1, c2) => Ok((
                "two_circles".to_string(),
                (
                    PyCircle3D(c1).into_pyobject(py)?.into_any().unbind(),
                    PyCircle3D(c2).into_pyobject(py)?.into_any().unbind(),
                )
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            )),
            SphereConeIntersection::TangentCircle(c) => Ok((
                "tangent_circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            SphereConeIntersection::Empty => Ok(("empty".to_string(), py.None())),
            SphereConeIntersection::NotAnalytic => Ok(("not_analytic".to_string(), py.None())),
        }
    }
}

/// A torus; u is the angle around the main axis, v the angle around the tube.
#[pyclass(name = "Torus", module = "geomcore.surfaces")]
#[derive(Clone)]
struct PyTorus(Torus);

#[pymethods]
impl PyTorus {
    #[new]
    /// Build a torus from its center, main-axis direction and the two radii.
    fn py_new(
        center: PyPoint3D,
        normal: PyVector3D,
        major_radius: f64,
        minor_radius: f64,
    ) -> PyResult<Self> {
        Ok(PyTorus(
            Torus::new(center.0, normal.0, major_radius, minor_radius).map_err(val_err)?,
        ))
    }

    /// Build a torus from a placement frame and the two radii.
    #[staticmethod]
    fn from_frame(frame: PyFrame3D, major_radius: f64, minor_radius: f64) -> PyResult<Self> {
        Ok(PyTorus(
            Torus::from_frame(frame.0, major_radius, minor_radius).map_err(val_err)?,
        ))
    }

    /// The distance from the center to the tube center line.
    fn major_radius(&self) -> f64 {
        self.0.major_radius()
    }

    /// The tube radius.
    fn minor_radius(&self) -> f64 {
        self.0.minor_radius()
    }

    /// Evaluate the point at surface parameters (u, v).
    fn eval_point(&self, u: f64, v: f64) -> PyPoint3D {
        PyPoint3D(self.0.eval_point(u, v))
    }

    /// Evaluate many (u, v) pairs at once.
    fn eval_points(&self, uvs: Vec<(f64, f64)>) -> Vec<PyPoint3D> {
        self.0
            .eval_points(&uvs)
            .into_iter()
            .map(PyPoint3D)
            .collect()
    }

    /// Evaluate the partial derivative of order (du, dv), with 1 <= du+dv <= 2.
    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> PyVector3D {
        PyVector3D(self.0.eval_derivative(u, v, du, dv))
    }

    /// Recover the (u, v) parameters of a point lying on the torus.
    fn parameters_of(&self, point: PyPoint3D) -> (f64, f64) {
        self.0.parameters_of(point.0)
    }

    /// Returns whether `point` lies on the surface within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0
            .contains(point.0, tol.map(|t| t.0).unwrap_or_default())
    }

    /// Projects `point` onto the surface, returning `(u, v, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64, f64) {
        let proj = self
            .0
            .project_point(point.0, tol.map(|t| t.0).unwrap_or_default());
        (proj.u, proj.v, proj.distance)
    }

    /// Projects each point in `points` onto the surface, returning a
    /// `(u, v, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(
        &self,
        points: Vec<PyPoint3D>,
        tol: Option<PyTolerance>,
    ) -> Vec<(f64, f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tol.map(|t| t.0).unwrap_or_default();
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.u, p.v, p.distance))
            .collect()
    }

    /// Intersects this torus with a plane.
    ///
    /// Returns `("two_circles", (Circle3D, Circle3D))`,
    /// `("tangent_circle", Circle3D)`, `("circle", Circle3D)`,
    /// `("empty", None)` or `("not_analytic", None)`.
    #[pyo3(signature = (plane, tol = None))]
    fn intersect_plane(
        &self,
        py: Python<'_>,
        plane: &PyPlane,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        match self.0.intersect_plane(&plane.0, tol) {
            TorusPlaneIntersection::TwoCircles(c1, c2) => {
                Ok(("two_circles".to_string(), circles_to_py(py, c1, c2)?))
            }
            TorusPlaneIntersection::TangentCircle(c) => Ok((
                "tangent_circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            TorusPlaneIntersection::Circle(c) => Ok((
                "circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            TorusPlaneIntersection::Empty => Ok(("empty".to_string(), py.None())),
            TorusPlaneIntersection::NotAnalytic => Ok(("not_analytic".to_string(), py.None())),
        }
    }

    /// Intersects this torus with a sphere.
    ///
    /// The analytic path needs the sphere center on the torus axis.
    /// Returns `("circle", Circle3D)`,
    /// `("two_circles", (Circle3D, Circle3D))`,
    /// `("tangent_circle", Circle3D)`, `("empty", None)` or
    /// `("not_analytic", None)`.
    #[pyo3(signature = (sphere, tol = None))]
    fn intersect_sphere(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        match self.0.intersect_sphere(&sphere.0, tol) {
            TorusSphereIntersection::Circle(c) => Ok((
                "circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            TorusSphereIntersection::TwoCircles(c1, c2) => {
                Ok(("two_circles".to_string(), circles_to_py(py, c1, c2)?))
            }
            TorusSphereIntersection::TangentCircle(c) => Ok((
                "tangent_circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            TorusSphereIntersection::Empty => Ok(("empty".to_string(), py.None())),
            TorusSphereIntersection::NotAnalytic => Ok(("not_analytic".to_string(), py.None())),
        }
    }

    /// Intersects this torus with a cylinder.
    ///
    /// The analytic path needs coaxial axes. Returns
    /// `("two_circles", (Circle3D, Circle3D))`,
    /// `("tangent_circle", Circle3D)`, `("empty", None)` or
    /// `("not_analytic", None)`.
    #[pyo3(signature = (cylinder, tol = None))]
    fn intersect_cylinder(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        match self.0.intersect_cylinder(&cylinder.0, tol) {
            TorusCylinderIntersection::TwoCircles(c1, c2) => {
                Ok(("two_circles".to_string(), circles_to_py(py, c1, c2)?))
            }
            TorusCylinderIntersection::TangentCircle(c) => Ok((
                "tangent_circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            TorusCylinderIntersection::Empty => Ok(("empty".to_string(), py.None())),
            TorusCylinderIntersection::NotAnalytic => Ok(("not_analytic".to_string(), py.None())),
        }
    }

    /// Intersects this torus with a cone.
    ///
    /// The analytic path needs coaxial axes. Returns `("circle", Circle3D)`,
    /// `("two_circles", (Circle3D, Circle3D))`,
    /// `("tangent_circle", Circle3D)`, `("empty", None)` or
    /// `("not_analytic", None)`.
    #[pyo3(signature = (cone, tol = None))]
    fn intersect_cone(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        match self.0.intersect_cone(&cone.0, tol) {
            TorusConeIntersection::Circle(c) => Ok((
                "circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            TorusConeIntersection::TwoCircles(c1, c2) => {
                Ok(("two_circles".to_string(), circles_to_py(py, c1, c2)?))
            }
            TorusConeIntersection::TangentCircle(c) => Ok((
                "tangent_circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            TorusConeIntersection::Empty => Ok(("empty".to_string(), py.None())),
            TorusConeIntersection::NotAnalytic => Ok(("not_analytic".to_string(), py.None())),
        }
    }

    /// Intersects this torus with another torus.
    ///
    /// The analytic path needs collinear axes. Returns `("circle", Circle3D)`,
    /// `("two_circles", (Circle3D, Circle3D))`,
    /// `("tangent_circle", Circle3D)`, `("empty", None)`,
    /// `("coincident", None)` or `("not_analytic", None)`.
    #[pyo3(signature = (other, tol = None))]
    fn intersect_torus(
        &self,
        py: Python<'_>,
        other: &PyTorus,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        match self.0.intersect_torus(&other.0, tol) {
            TorusTorusIntersection::Circle(c) => Ok((
                "circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            TorusTorusIntersection::TwoCircles(c1, c2) => {
                Ok(("two_circles".to_string(), circles_to_py(py, c1, c2)?))
            }
            TorusTorusIntersection::TangentCircle(c) => Ok((
                "tangent_circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            TorusTorusIntersection::Empty => Ok(("empty".to_string(), py.None())),
            TorusTorusIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
            TorusTorusIntersection::NotAnalytic => Ok(("not_analytic".to_string(), py.None())),
        }
    }
}

/// A tensor-product B-spline (optionally rational, optionally periodic) surface.
#[pyclass(name = "BSplineSurface", module = "geomcore.surfaces")]
#[derive(Clone)]
struct PyBSplineSurface(BSplineSurface);

#[pymethods]
impl PyBSplineSurface {
    #[new]
    #[allow(clippy::too_many_arguments)]
    /// Build a non-rational B-spline surface from a rectangular pole grid.
    fn py_new(
        u_degree: usize,
        v_degree: usize,
        poles: Vec<Vec<PyPoint3D>>,
        u_knots: Vec<f64>,
        u_multiplicities: Vec<u32>,
        v_knots: Vec<f64>,
        v_multiplicities: Vec<u32>,
        u_periodic: bool,
        v_periodic: bool,
    ) -> PyResult<Self> {
        let poles = poles
            .into_iter()
            .map(|row| row.into_iter().map(|p| p.0).collect())
            .collect();
        Ok(PyBSplineSurface(
            BSplineSurface::new(
                u_degree,
                v_degree,
                poles,
                u_knots,
                u_multiplicities,
                v_knots,
                v_multiplicities,
                u_periodic,
                v_periodic,
            )
            .map_err(val_err)?,
        ))
    }

    /// Build a rational B-spline (NURBS) surface with a weight per pole.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn new_rational(
        u_degree: usize,
        v_degree: usize,
        poles: Vec<Vec<PyPoint3D>>,
        weights: Vec<Vec<f64>>,
        u_knots: Vec<f64>,
        u_multiplicities: Vec<u32>,
        v_knots: Vec<f64>,
        v_multiplicities: Vec<u32>,
        u_periodic: bool,
        v_periodic: bool,
    ) -> PyResult<Self> {
        let poles = poles
            .into_iter()
            .map(|row| row.into_iter().map(|p| p.0).collect())
            .collect();
        Ok(PyBSplineSurface(
            BSplineSurface::new_rational(
                u_degree,
                v_degree,
                poles,
                weights,
                u_knots,
                u_multiplicities,
                v_knots,
                v_multiplicities,
                u_periodic,
                v_periodic,
            )
            .map_err(val_err)?,
        ))
    }

    /// Polynomial degree in the u direction.
    fn u_degree(&self) -> usize {
        self.0.u_degree()
    }

    /// Polynomial degree in the v direction.
    fn v_degree(&self) -> usize {
        self.0.v_degree()
    }

    /// Whether the surface is periodic in u.
    fn is_u_periodic(&self) -> bool {
        self.0.is_u_periodic()
    }

    /// Whether the surface is periodic in v.
    fn is_v_periodic(&self) -> bool {
        self.0.is_v_periodic()
    }

    /// Whether the surface carries rational weights.
    fn is_rational(&self) -> bool {
        self.0.is_rational()
    }

    /// Evaluate the point at surface parameters (u, v).
    fn eval_point(&self, u: f64, v: f64) -> PyPoint3D {
        PyPoint3D(self.0.eval_point(u, v))
    }

    /// Evaluate many (u, v) pairs at once.
    fn eval_points(&self, uvs: Vec<(f64, f64)>) -> Vec<PyPoint3D> {
        self.0
            .eval_points(&uvs)
            .into_iter()
            .map(PyPoint3D)
            .collect()
    }

    /// Evaluate the first partial derivative: (du, dv) must be (1, 0) or (0, 1).
    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> PyVector3D {
        PyVector3D(self.0.eval_derivative(u, v, du, dv))
    }
}

fn extract_surface(obj: &Bound<'_, PyAny>) -> PyResult<Surface> {
    if let Ok(s) = obj.extract::<PyRef<'_, PyPlane>>() {
        return Ok(Surface::from(&s.0));
    }
    if let Ok(s) = obj.extract::<PyRef<'_, PyCylinder>>() {
        return Ok(Surface::from(&s.0));
    }
    if let Ok(s) = obj.extract::<PyRef<'_, PyCone>>() {
        return Ok(Surface::from(&s.0));
    }
    if let Ok(s) = obj.extract::<PyRef<'_, PySphere>>() {
        return Ok(Surface::from(&s.0));
    }
    if let Ok(s) = obj.extract::<PyRef<'_, PyTorus>>() {
        return Ok(Surface::from(&s.0));
    }
    if let Ok(s) = obj.extract::<PyRef<'_, PyBSplineSurface>>() {
        return Ok(Surface::from(&s.0));
    }
    Err(PyTypeError::new_err(
        "expected a geomcore surface (Plane, Cylinder, Cone, Sphere, Torus or BSplineSurface)",
    ))
}

fn curve2d_to_py(py: Python<'_>, curve: Curve2D) -> PyResult<Py<PyAny>> {
    match curve {
        Curve2D::Line(l) => Ok(PyLine2D(l).into_pyobject(py)?.into_any().unbind()),
        Curve2D::Circle(c) => Ok(PyCircle2D(c).into_pyobject(py)?.into_any().unbind()),
        _ => Err(PyValueError::new_err("unsupported 2d curve variant")),
    }
}

fn circles_to_py(py: Python<'_>, c1: Circle3D, c2: Circle3D) -> PyResult<Py<PyAny>> {
    (
        PyCircle3D(c1).into_pyobject(py)?.into_any().unbind(),
        PyCircle3D(c2).into_pyobject(py)?.into_any().unbind(),
    )
        .into_pyobject(py)
        .map(|o| o.into_any().unbind())
}

fn quadric_hit_to_py(
    py: Python<'_>,
    hit: LineQuadricIntersection,
) -> PyResult<(String, Py<PyAny>)> {
    fn point(py: Python<'_>, t: f64, p: Point3D) -> PyResult<Py<PyAny>> {
        (t, PyPoint3D(p).into_pyobject(py)?.into_any().unbind())
            .into_pyobject(py)
            .map(|o| o.into_any().unbind())
    }
    match hit {
        LineQuadricIntersection::TwoPoints((t1, p1), (t2, p2)) => Ok((
            "two_points".to_string(),
            (point(py, t1, p1)?, point(py, t2, p2)?)
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        )),
        LineQuadricIntersection::OnePoint(t, p) => Ok(("one_point".to_string(), point(py, t, p)?)),
        LineQuadricIntersection::Tangent(t, p) => Ok(("tangent".to_string(), point(py, t, p)?)),
        LineQuadricIntersection::Empty => Ok(("empty".to_string(), py.None())),
        LineQuadricIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
    }
}

// ---------------------------------------------------------------------------
// Curves
// ---------------------------------------------------------------------------

/// An infinite straight line in 3D, parametrized by arc length from its origin.
#[pyclass(name = "Line3D", module = "geomcore.curves")]
#[derive(Clone)]
struct PyLine3D(Line3D);

#[pymethods]
impl PyLine3D {
    #[new]
    /// Build a line from an origin and a direction (normalized internally).
    fn py_new(origin: PyPoint3D, direction: PyVector3D) -> PyResult<Self> {
        Ok(PyLine3D(
            Line3D::new(origin.0, direction.0).map_err(val_err)?,
        ))
    }

    /// Build a line from an axis.
    #[staticmethod]
    fn from_axis(axis: PyAxis3D) -> Self {
        PyLine3D(Line3D::from_axis(axis.0))
    }

    /// Build a line through two distinct points.
    #[staticmethod]
    fn from_two_points(p1: PyPoint3D, p2: PyPoint3D) -> PyResult<Self> {
        Ok(PyLine3D(
            Line3D::from_two_points(p1.0, p2.0).map_err(val_err)?,
        ))
    }

    /// The line origin (point at parameter 0).
    fn origin(&self) -> PyPoint3D {
        PyPoint3D(self.0.origin())
    }

    /// The unit direction of the line.
    fn direction(&self) -> PyVector3D {
        PyVector3D(self.0.direction())
    }

    /// Evaluate the point at parameter `u`.
    fn eval_point(&self, u: f64) -> PyPoint3D {
        PyPoint3D(self.0.eval_point(u))
    }

    /// Evaluate many parameters at once.
    fn eval_points(&self, us: Vec<f64>) -> Vec<PyPoint3D> {
        self.0.eval_points(&us).into_iter().map(PyPoint3D).collect()
    }

    /// Evaluate the derivative of the given order (>= 1) at `u`.
    fn eval_derivative(&self, u: f64, order: u32) -> PyVector3D {
        PyVector3D(self.0.eval_derivative(u, order))
    }

    /// Recover the parameter of a point lying on the line.
    fn parameter_of(&self, point: PyPoint3D) -> f64 {
        self.0.parameter_of(point.0)
    }

    /// Returns whether `point` lies on the curve within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0
            .contains(point.0, tol.map(|t| t.0).unwrap_or_default())
    }

    /// Projects `point` onto the curve, returning `(parameter, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64) {
        let proj = self
            .0
            .project_point(point.0, tol.map(|t| t.0).unwrap_or_default());
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the curve, returning a
    /// `(parameter, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(&self, points: Vec<PyPoint3D>, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tol.map(|t| t.0).unwrap_or_default();
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Intersects this line with a plane.
    ///
    /// Returns `("point", (t, Point3D))`, `("parallel", None)` or
    /// `("coincident", None)`.
    #[pyo3(signature = (plane, tol = None))]
    fn intersect_plane(
        &self,
        py: Python<'_>,
        plane: &PyPlane,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        match self.0.intersect_plane(&plane.0, tol) {
            LinePlaneIntersection::Point(t, p) => Ok((
                "point".to_string(),
                (t, PyPoint3D(p).into_pyobject(py)?.into_any().unbind())
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            )),
            LinePlaneIntersection::Parallel => Ok(("parallel".to_string(), py.None())),
            LinePlaneIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
        }
    }

    /// Intersects this line with a sphere.
    ///
    /// Returns `("two_points", ((t1, Point3D), (t2, Point3D)))`,
    /// `("one_point", (t, Point3D))`, `("tangent", (t, Point3D))`,
    /// `("empty", None)` or `("coincident", None)` (the latter never
    /// triggers for spheres, which contain no line).
    #[pyo3(signature = (sphere, tol = None))]
    fn intersect_sphere(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        quadric_hit_to_py(py, self.0.intersect_sphere(&sphere.0, tol))
    }

    /// Intersects this line with a cylinder.
    ///
    /// Same return shape as [`PyLine3D::intersect_sphere`]; a generator
    /// reports `("coincident", None)`.
    #[pyo3(signature = (cylinder, tol = None))]
    fn intersect_cylinder(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        quadric_hit_to_py(py, self.0.intersect_cylinder(&cylinder.0, tol))
    }

    /// Intersects this line with a cone.
    ///
    /// Same return shape as [`PyLine3D::intersect_sphere`]; roots behind
    /// the apex are filtered, so a single surviving hit reports
    /// `("one_point", (t, Point3D))`, and a generator reports
    /// `("coincident", None)`.
    #[pyo3(signature = (cone, tol = None))]
    fn intersect_cone(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        quadric_hit_to_py(py, self.0.intersect_cone(&cone.0, tol))
    }

    /// Compute this line's 2D representation in a surface's (u, v) space.
    ///
    /// Raises `ValueError` if no closed-form representation exists for the
    /// pair, or if the line does not lie on the surface.
    fn parametrize_on(&self, py: Python<'_>, surface: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let s = extract_surface(surface)?;
        curve2d_to_py(py, self.0.parametrize_on(s).map_err(val_err)?)
    }
}

/// A circle in 3D, parametrized by the angle from its frame's x direction.
#[pyclass(name = "Circle3D", module = "geomcore.curves")]
#[derive(Clone)]
struct PyCircle3D(Circle3D);

#[pymethods]
impl PyCircle3D {
    #[new]
    /// Build a circle from its center, plane normal and radius.
    fn py_new(center: PyPoint3D, normal: PyVector3D, radius: f64) -> PyResult<Self> {
        Ok(PyCircle3D(
            Circle3D::new(center.0, normal.0, radius).map_err(val_err)?,
        ))
    }

    /// Build a circle from an axis (center + normal) and a radius.
    #[staticmethod]
    fn from_axis(axis: PyAxis3D, radius: f64) -> PyResult<Self> {
        Ok(PyCircle3D(
            Circle3D::from_axis(axis.0, radius).map_err(val_err)?,
        ))
    }

    /// Build a circle from a placement frame and a radius.
    #[staticmethod]
    fn from_frame(frame: PyFrame3D, radius: f64) -> PyResult<Self> {
        Ok(PyCircle3D(
            Circle3D::from_frame(frame.0, radius).map_err(val_err)?,
        ))
    }

    /// Build the circle through three non-collinear points; the curve starts
    /// at the first point.
    #[staticmethod]
    fn from_three_points(p1: PyPoint3D, p2: PyPoint3D, p3: PyPoint3D) -> PyResult<Self> {
        Ok(PyCircle3D(
            Circle3D::from_three_points(p1.0, p2.0, p3.0).map_err(val_err)?,
        ))
    }

    /// The circle center.
    fn center(&self) -> PyPoint3D {
        PyPoint3D(self.0.center())
    }

    /// The circle radius.
    fn radius(&self) -> f64 {
        self.0.radius()
    }

    /// The unit normal of the circle's plane.
    fn normal(&self) -> PyVector3D {
        PyVector3D(self.0.normal())
    }

    /// Evaluate the point at angle `u` (radians).
    fn eval_point(&self, u: f64) -> PyPoint3D {
        PyPoint3D(self.0.eval_point(u))
    }

    /// Evaluate many parameters at once.
    fn eval_points(&self, us: Vec<f64>) -> Vec<PyPoint3D> {
        self.0.eval_points(&us).into_iter().map(PyPoint3D).collect()
    }

    /// Evaluate the derivative of the given order (>= 1) at `u`.
    fn eval_derivative(&self, u: f64, order: u32) -> PyVector3D {
        PyVector3D(self.0.eval_derivative(u, order))
    }

    /// Recover the parameter in [0, 2*pi) of a point lying on the circle.
    fn parameter_of(&self, point: PyPoint3D) -> f64 {
        self.0.parameter_of(point.0)
    }

    /// Returns whether `point` lies on the curve within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0
            .contains(point.0, tol.map(|t| t.0).unwrap_or_default())
    }

    /// Projects `point` onto the curve, returning `(parameter, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64) {
        let proj = self
            .0
            .project_point(point.0, tol.map(|t| t.0).unwrap_or_default());
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the curve, returning a
    /// `(parameter, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(&self, points: Vec<PyPoint3D>, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tol.map(|t| t.0).unwrap_or_default();
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Compute this circle's 2D representation in a surface's (u, v) space.
    ///
    /// Raises `ValueError` if no closed-form representation exists for the
    /// pair, or if the circle does not lie on the surface.
    fn parametrize_on(&self, py: Python<'_>, surface: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let s = extract_surface(surface)?;
        curve2d_to_py(py, self.0.parametrize_on(s).map_err(val_err)?)
    }
}

/// An ellipse in 3D.
#[pyclass(name = "Ellipse3D", module = "geomcore.curves")]
#[derive(Clone)]
struct PyEllipse3D(Ellipse3D);

#[pymethods]
impl PyEllipse3D {
    #[new]
    /// Build an ellipse from center, normal, major-axis direction and radii.
    fn py_new(
        center: PyPoint3D,
        normal: PyVector3D,
        x_direction: PyVector3D,
        major_radius: f64,
        minor_radius: f64,
    ) -> PyResult<Self> {
        Ok(PyEllipse3D(
            Ellipse3D::new(
                center.0,
                normal.0,
                x_direction.0,
                major_radius,
                minor_radius,
            )
            .map_err(val_err)?,
        ))
    }

    /// Build an ellipse from a placement frame and the two radii.
    #[staticmethod]
    fn from_frame(frame: PyFrame3D, major_radius: f64, minor_radius: f64) -> PyResult<Self> {
        Ok(PyEllipse3D(
            Ellipse3D::from_frame(frame.0, major_radius, minor_radius).map_err(val_err)?,
        ))
    }

    /// Build an ellipse from its center, the major-axis end point and a
    /// second point on the curve.
    #[staticmethod]
    fn from_center_and_points(center: PyPoint3D, s1: PyPoint3D, s2: PyPoint3D) -> PyResult<Self> {
        Ok(PyEllipse3D(
            Ellipse3D::from_center_and_points(center.0, s1.0, s2.0).map_err(val_err)?,
        ))
    }

    /// The ellipse center.
    fn center(&self) -> PyPoint3D {
        PyPoint3D(self.0.center())
    }

    /// The semi-major radius.
    fn major_radius(&self) -> f64 {
        self.0.major_radius()
    }

    /// The semi-minor radius.
    fn minor_radius(&self) -> f64 {
        self.0.minor_radius()
    }

    /// Evaluate the point at angle `u` (radians).
    fn eval_point(&self, u: f64) -> PyPoint3D {
        PyPoint3D(self.0.eval_point(u))
    }

    /// Evaluate many parameters at once.
    fn eval_points(&self, us: Vec<f64>) -> Vec<PyPoint3D> {
        self.0.eval_points(&us).into_iter().map(PyPoint3D).collect()
    }

    /// Evaluate the derivative of the given order (>= 1) at `u`.
    fn eval_derivative(&self, u: f64, order: u32) -> PyVector3D {
        PyVector3D(self.0.eval_derivative(u, order))
    }

    /// Recover the parameter in [0, 2*pi) of a point lying on the ellipse.
    fn parameter_of(&self, point: PyPoint3D) -> f64 {
        self.0.parameter_of(point.0)
    }

    /// Returns whether `point` lies on the curve within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0
            .contains(point.0, tol.map(|t| t.0).unwrap_or_default())
    }

    /// All stationary points of the distance from `point` to the ellipse,
    /// as `(parameter, distance)` tuples ordered by ascending distance.
    #[pyo3(signature = (point, tol = None))]
    fn extrema(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        self.0
            .extrema(point.0, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Projects `point` onto the ellipse, returning `(parameter, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64) {
        let proj = self
            .0
            .project_point(point.0, tol.map(|t| t.0).unwrap_or_default());
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the ellipse, returning a
    /// `(parameter, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(&self, points: Vec<PyPoint3D>, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tol.map(|t| t.0).unwrap_or_default();
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Not available for ellipses in this release; always raises `ValueError`.
    fn parametrize_on(&self, py: Python<'_>, surface: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let s = extract_surface(surface)?;
        curve2d_to_py(py, self.0.parametrize_on(s).map_err(val_err)?)
    }
}

/// A parabola in 3D, parametrized so the apex is at parameter 0.
#[pyclass(name = "Parabola3D", module = "geomcore.curves")]
#[derive(Clone)]
struct PyParabola3D(Parabola3D);

#[pymethods]
impl PyParabola3D {
    #[new]
    /// Build a parabola from apex, plane normal, axis direction and focal distance.
    fn py_new(
        apex: PyPoint3D,
        normal: PyVector3D,
        x_direction: PyVector3D,
        focal: f64,
    ) -> PyResult<Self> {
        Ok(PyParabola3D(
            Parabola3D::new(apex.0, normal.0, x_direction.0, focal).map_err(val_err)?,
        ))
    }

    /// Build a parabola from a placement frame and focal distance.
    #[staticmethod]
    fn from_frame(frame: PyFrame3D, focal: f64) -> PyResult<Self> {
        Ok(PyParabola3D(
            Parabola3D::from_frame(frame.0, focal).map_err(val_err)?,
        ))
    }

    /// The focal distance.
    fn focal(&self) -> f64 {
        self.0.focal()
    }

    /// Evaluate the point at parameter `u`.
    fn eval_point(&self, u: f64) -> PyPoint3D {
        PyPoint3D(self.0.eval_point(u))
    }

    /// Evaluate many parameters at once.
    fn eval_points(&self, us: Vec<f64>) -> Vec<PyPoint3D> {
        self.0.eval_points(&us).into_iter().map(PyPoint3D).collect()
    }

    /// Evaluate the derivative of the given order (>= 1) at `u`.
    fn eval_derivative(&self, u: f64, order: u32) -> PyVector3D {
        PyVector3D(self.0.eval_derivative(u, order))
    }

    /// Recover the parameter of a point lying on the parabola.
    fn parameter_of(&self, point: PyPoint3D) -> f64 {
        self.0.parameter_of(point.0)
    }

    /// Returns whether `point` lies on the curve within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0
            .contains(point.0, tol.map(|t| t.0).unwrap_or_default())
    }

    /// All stationary points of the distance from `point` to the parabola,
    /// as `(parameter, distance)` tuples ordered by ascending distance.
    #[pyo3(signature = (point, tol = None))]
    fn extrema(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        self.0
            .extrema(point.0, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Projects `point` onto the parabola, returning `(parameter, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64) {
        let proj = self
            .0
            .project_point(point.0, tol.map(|t| t.0).unwrap_or_default());
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the parabola, returning a
    /// `(parameter, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(&self, points: Vec<PyPoint3D>, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tol.map(|t| t.0).unwrap_or_default();
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Not available for parabolas in this release; always raises `ValueError`.
    fn parametrize_on(&self, py: Python<'_>, surface: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let s = extract_surface(surface)?;
        curve2d_to_py(py, self.0.parametrize_on(s).map_err(val_err)?)
    }
}

/// One branch of a hyperbola in 3D.
#[pyclass(name = "Hyperbola3D", module = "geomcore.curves")]
#[derive(Clone)]
struct PyHyperbola3D(Hyperbola3D);

#[pymethods]
impl PyHyperbola3D {
    #[new]
    /// Build a hyperbola from center, normal, major-axis direction and radii.
    fn py_new(
        center: PyPoint3D,
        normal: PyVector3D,
        x_direction: PyVector3D,
        major_radius: f64,
        minor_radius: f64,
    ) -> PyResult<Self> {
        Ok(PyHyperbola3D(
            Hyperbola3D::new(
                center.0,
                normal.0,
                x_direction.0,
                major_radius,
                minor_radius,
            )
            .map_err(val_err)?,
        ))
    }

    /// Build a hyperbola from a placement frame and the two radii.
    #[staticmethod]
    fn from_frame(frame: PyFrame3D, major_radius: f64, minor_radius: f64) -> PyResult<Self> {
        Ok(PyHyperbola3D(
            Hyperbola3D::from_frame(frame.0, major_radius, minor_radius).map_err(val_err)?,
        ))
    }

    /// Build a hyperbola from its center, the vertex point and a second
    /// point on the curve.
    #[staticmethod]
    fn from_center_and_points(center: PyPoint3D, s1: PyPoint3D, s2: PyPoint3D) -> PyResult<Self> {
        Ok(PyHyperbola3D(
            Hyperbola3D::from_center_and_points(center.0, s1.0, s2.0).map_err(val_err)?,
        ))
    }

    /// The hyperbola center.
    fn center(&self) -> PyPoint3D {
        PyPoint3D(self.0.center())
    }

    /// The semi-major (transverse) radius.
    fn major_radius(&self) -> f64 {
        self.0.major_radius()
    }

    /// The semi-minor (conjugate) radius.
    fn minor_radius(&self) -> f64 {
        self.0.minor_radius()
    }

    /// Evaluate the point at parameter `u`.
    fn eval_point(&self, u: f64) -> PyPoint3D {
        PyPoint3D(self.0.eval_point(u))
    }

    /// Evaluate many parameters at once.
    fn eval_points(&self, us: Vec<f64>) -> Vec<PyPoint3D> {
        self.0.eval_points(&us).into_iter().map(PyPoint3D).collect()
    }

    /// Evaluate the derivative of the given order (>= 1) at `u`.
    fn eval_derivative(&self, u: f64, order: u32) -> PyVector3D {
        PyVector3D(self.0.eval_derivative(u, order))
    }

    /// Recover the parameter of a point lying on the hyperbola.
    fn parameter_of(&self, point: PyPoint3D) -> f64 {
        self.0.parameter_of(point.0)
    }

    /// Returns whether `point` lies on the curve within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0
            .contains(point.0, tol.map(|t| t.0).unwrap_or_default())
    }

    /// All stationary points of the distance from `point` to the hyperbola,
    /// as `(parameter, distance)` tuples ordered by ascending distance.
    #[pyo3(signature = (point, tol = None))]
    fn extrema(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        self.0
            .extrema(point.0, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Projects `point` onto the hyperbola, returning `(parameter, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64) {
        let proj = self
            .0
            .project_point(point.0, tol.map(|t| t.0).unwrap_or_default());
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the hyperbola, returning a
    /// `(parameter, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(&self, points: Vec<PyPoint3D>, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tol.map(|t| t.0).unwrap_or_default();
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Not available for hyperbolas in this release; always raises `ValueError`.
    fn parametrize_on(&self, py: Python<'_>, surface: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let s = extract_surface(surface)?;
        curve2d_to_py(py, self.0.parametrize_on(s).map_err(val_err)?)
    }
}

/// A B-spline (optionally rational, optionally periodic) curve in 3D.
#[pyclass(name = "BSplineCurve3D", module = "geomcore.curves")]
#[derive(Clone)]
struct PyBSplineCurve3D(BSplineCurve3D);

#[pymethods]
impl PyBSplineCurve3D {
    #[new]
    /// Build a non-rational B-spline curve.
    fn py_new(
        degree: usize,
        poles: Vec<PyPoint3D>,
        knots: Vec<f64>,
        multiplicities: Vec<u32>,
        periodic: bool,
    ) -> PyResult<Self> {
        let poles = poles.into_iter().map(|p| p.0).collect();
        Ok(PyBSplineCurve3D(
            BSplineCurve3D::new(degree, poles, knots, multiplicities, periodic).map_err(val_err)?,
        ))
    }

    /// Build a rational B-spline (NURBS) curve with a weight per pole.
    #[staticmethod]
    fn new_rational(
        degree: usize,
        poles: Vec<PyPoint3D>,
        weights: Vec<f64>,
        knots: Vec<f64>,
        multiplicities: Vec<u32>,
        periodic: bool,
    ) -> PyResult<Self> {
        let poles = poles.into_iter().map(|p| p.0).collect();
        Ok(PyBSplineCurve3D(
            BSplineCurve3D::new_rational(degree, poles, weights, knots, multiplicities, periodic)
                .map_err(val_err)?,
        ))
    }

    /// The polynomial degree.
    fn degree(&self) -> usize {
        self.0.degree()
    }

    /// Whether the curve is periodic.
    fn is_periodic(&self) -> bool {
        self.0.is_periodic()
    }

    /// Whether the curve carries rational weights.
    fn is_rational(&self) -> bool {
        self.0.is_rational()
    }

    /// The (first, last) parameter bounds.
    fn bounds(&self) -> (f64, f64) {
        self.0.bounds()
    }

    /// Evaluate the point at parameter `u`.
    fn eval_point(&self, u: f64) -> PyPoint3D {
        PyPoint3D(self.0.eval_point(u))
    }

    /// Evaluate many parameters at once.
    fn eval_points(&self, us: Vec<f64>) -> Vec<PyPoint3D> {
        self.0.eval_points(&us).into_iter().map(PyPoint3D).collect()
    }

    /// Evaluate the derivative at `u`; orders 1 and 2 are supported.
    fn eval_derivative(&self, u: f64, order: u32) -> PyVector3D {
        PyVector3D(self.0.eval_derivative(u, order))
    }

    /// Not available for B-spline curves in this release; always raises `ValueError`.
    fn parametrize_on(&self, py: Python<'_>, surface: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let s = extract_surface(surface)?;
        curve2d_to_py(py, self.0.parametrize_on(s).map_err(val_err)?)
    }
}

/// A straight line in a surface's 2D (u, v) parameter space.
#[pyclass(name = "Line2D", module = "geomcore.curves")]
#[derive(Clone)]
struct PyLine2D(Line2D);

#[pymethods]
impl PyLine2D {
    #[new]
    /// Build a 2D line from an origin and a direction (normalized internally).
    fn py_new(origin: PyPoint2D, direction: PyVector2D) -> PyResult<Self> {
        Ok(PyLine2D(
            Line2D::new(origin.0, direction.0).map_err(val_err)?,
        ))
    }

    /// The line origin (point at parameter 0).
    fn origin(&self) -> PyPoint2D {
        PyPoint2D(self.0.origin())
    }

    /// The unit direction of the line.
    fn direction(&self) -> PyVector2D {
        PyVector2D(self.0.direction())
    }

    /// Evaluate the point at parameter `u`.
    fn eval_point(&self, u: f64) -> PyPoint2D {
        PyPoint2D(self.0.eval_point(u))
    }

    /// Evaluate many parameters at once.
    fn eval_points(&self, us: Vec<f64>) -> Vec<PyPoint2D> {
        self.0.eval_points(&us).into_iter().map(PyPoint2D).collect()
    }

    /// Evaluate the derivative of the given order (>= 1) at `u`.
    fn eval_derivative(&self, u: f64, order: u32) -> PyVector2D {
        PyVector2D(self.0.eval_derivative(u, order))
    }

    /// Recover the parameter of a point lying on the line.
    fn parameter_of(&self, point: PyPoint2D) -> f64 {
        self.0.parameter_of(point.0)
    }

    /// Returns whether `point` lies on the curve within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint2D, tol: Option<PyTolerance>) -> bool {
        self.0
            .contains(point.0, tol.map(|t| t.0).unwrap_or_default())
    }

    /// Projects `point` onto the curve, returning `(parameter, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint2D, tol: Option<PyTolerance>) -> (f64, f64) {
        let proj = self
            .0
            .project_point(point.0, tol.map(|t| t.0).unwrap_or_default());
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the curve, returning a
    /// `(parameter, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(&self, points: Vec<PyPoint2D>, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tol.map(|t| t.0).unwrap_or_default();
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    fn __repr__(&self) -> String {
        let o = self.0.origin();
        let d = self.0.direction();
        format!(
            "Line2D(origin=({}, {}), direction=({}, {}))",
            o.x, o.y, d.x, d.y
        )
    }
}

/// A circle in a surface's 2D (u, v) parameter space.
#[pyclass(name = "Circle2D", module = "geomcore.curves")]
#[derive(Clone)]
struct PyCircle2D(Circle2D);

#[pymethods]
impl PyCircle2D {
    #[new]
    /// Build a 2D circle from its center and radius.
    fn py_new(center: PyPoint2D, radius: f64) -> PyResult<Self> {
        Ok(PyCircle2D(
            Circle2D::new(center.0, radius).map_err(val_err)?,
        ))
    }

    /// The circle center.
    fn center(&self) -> PyPoint2D {
        PyPoint2D(self.0.center())
    }

    /// The circle radius.
    fn radius(&self) -> f64 {
        self.0.radius()
    }

    /// Evaluate the point at angle `u` (radians).
    fn eval_point(&self, u: f64) -> PyPoint2D {
        PyPoint2D(self.0.eval_point(u))
    }

    /// Evaluate many parameters at once.
    fn eval_points(&self, us: Vec<f64>) -> Vec<PyPoint2D> {
        self.0.eval_points(&us).into_iter().map(PyPoint2D).collect()
    }

    /// Evaluate the derivative of the given order (>= 1) at `u`.
    fn eval_derivative(&self, u: f64, order: u32) -> PyVector2D {
        PyVector2D(self.0.eval_derivative(u, order))
    }

    /// Recover the parameter in [0, 2*pi) of a point lying on the circle.
    fn parameter_of(&self, point: PyPoint2D) -> f64 {
        self.0.parameter_of(point.0)
    }

    /// Returns whether `point` lies on the curve within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint2D, tol: Option<PyTolerance>) -> bool {
        self.0
            .contains(point.0, tol.map(|t| t.0).unwrap_or_default())
    }

    /// Projects `point` onto the curve, returning `(parameter, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint2D, tol: Option<PyTolerance>) -> (f64, f64) {
        let proj = self
            .0
            .project_point(point.0, tol.map(|t| t.0).unwrap_or_default());
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the curve, returning a
    /// `(parameter, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(&self, points: Vec<PyPoint2D>, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tol.map(|t| t.0).unwrap_or_default();
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    fn __repr__(&self) -> String {
        let c = self.0.center();
        format!(
            "Circle2D(center=({}, {}), radius={})",
            c.x,
            c.y,
            self.0.radius()
        )
    }
}

// ---------------------------------------------------------------------------
// Module wiring
// ---------------------------------------------------------------------------

#[pymodule(name = "geomcore")]
fn geomcore_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    m.add_class::<PyPoint3D>()?;
    m.add_class::<PyPoint2D>()?;
    m.add_class::<PyVector3D>()?;
    m.add_class::<PyVector2D>()?;
    m.add_class::<PyAxis3D>()?;
    m.add_class::<PyFrame3D>()?;
    m.add_class::<PyTolerance>()?;
    m.add_class::<PyTransform>()?;

    let curves = PyModule::new(py, "curves")?;
    curves.add_class::<PyLine3D>()?;
    curves.add_class::<PyCircle3D>()?;
    curves.add_class::<PyEllipse3D>()?;
    curves.add_class::<PyParabola3D>()?;
    curves.add_class::<PyHyperbola3D>()?;
    curves.add_class::<PyBSplineCurve3D>()?;
    curves.add_class::<PyLine2D>()?;
    curves.add_class::<PyCircle2D>()?;
    m.add_submodule(&curves)?;

    let surfaces = PyModule::new(py, "surfaces")?;
    surfaces.add_class::<PyPlane>()?;
    surfaces.add_class::<PyCylinder>()?;
    surfaces.add_class::<PyCone>()?;
    surfaces.add_class::<PySphere>()?;
    surfaces.add_class::<PyTorus>()?;
    surfaces.add_class::<PyBSplineSurface>()?;
    m.add_submodule(&surfaces)?;

    // Register the submodules in sys.modules so that
    // `from geomcore.curves import Circle3D` works.
    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("geomcore.curves", &curves)?;
    sys_modules.set_item("geomcore.surfaces", &surfaces)?;

    Ok(())
}
