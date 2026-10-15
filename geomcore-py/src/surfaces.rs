use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use geomcore::IntersectExact;

use geomcore::Bound3D;
use geomcore::surfaces::{
    BSplineSurface, Cone, Cylinder, ParametricSurface, Plane, Sphere, Surface, Torus,
};
use geomcore::{
    ConeConeIntersection, ConeCylinderIntersection, CylinderCylinderIntersection,
    PlaneConeIntersection, PlaneCylinderIntersection, PlanePlaneIntersection,
    PlaneSphereIntersection, SphereConeIntersection, SphereCylinderIntersection,
    SphereSphereIntersection, Tolerance, TorusConeIntersection, TorusCylinderIntersection,
    TorusPlaneIntersection, TorusSphereIntersection, TorusTorusIntersection,
};

use crate::curves::*;
use crate::numeric::*;
use crate::values::*;
use crate::{check_surface_order, parse_parametrization, tolerance, val_err, warn_deprecated};

// ---------------------------------------------------------------------------
// Surfaces
// ---------------------------------------------------------------------------

/// An infinite plane, parametrized by in-plane coordinates (u, v).
#[pyclass(name = "Plane", module = "geomcore.surfaces", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyPlane(pub(crate) Plane);

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

    /// Image of this plane under `transform` (raises ValueError for a
    /// collapsing transform).
    #[pyo3(signature = (transform,))]
    fn transformed(&self, transform: &PyTransform) -> PyResult<Self> {
        Ok(PyPlane(self.0.transformed(transform.0).map_err(val_err)?))
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
    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> PyResult<PyVector3D> {
        check_surface_order(du, dv)?;
        Ok(PyVector3D(self.0.eval_derivative(u, v, du, dv)))
    }

    /// Recover the (u, v) parameters of a point lying on the plane.
    fn parameters_of(&self, point: PyPoint3D) -> (f64, f64) {
        self.0.parameters_of(point.0)
    }

    /// The (first, last) `u` parameter bounds (unbounded here).
    fn u_bounds(&self) -> (f64, f64) {
        ParametricSurface::u_bounds(&self.0)
    }

    /// The (first, last) `v` parameter bounds (unbounded here).
    fn v_bounds(&self) -> (f64, f64) {
        ParametricSurface::v_bounds(&self.0)
    }

    /// The `u` isoparametric line through `(u, 0)` along the plane's `v` axis, parametrized by `v` like the surface:
    /// `u_iso(u).eval_point(t)` is `eval_point(u, t)` (raises `ValueError` for
    /// a non-finite parameter).
    fn u_iso(&self, u: f64) -> PyResult<PyLine3D> {
        Ok(PyLine3D(self.0.u_iso(u).map_err(val_err)?))
    }

    /// The `v` isoparametric line through `(0, v)` along the plane's `u` axis, parametrized by `u` like the surface:
    /// `v_iso(v).eval_point(t)` is `eval_point(t, v)` (raises `ValueError` for
    /// a non-finite parameter).
    fn v_iso(&self, v: f64) -> PyResult<PyLine3D> {
        Ok(PyLine3D(self.0.v_iso(v).map_err(val_err)?))
    }

    /// The axis-aligned box around the shape (`None` when unbounded).
    fn bounding_box(&self) -> Option<PyBoundingBox3D> {
        self.0.bounding_box().map(PyBoundingBox3D)
    }

    /// The patch `u_range × v_range` as an exact bilinear B-spline
    /// surface (raises `ValueError` for a bad range).
    fn to_bspline_patch(
        &self,
        u_range: (f64, f64),
        v_range: (f64, f64),
    ) -> PyResult<PyBSplineSurface> {
        Ok(PyBSplineSurface(
            self.0.to_bspline_patch(u_range, v_range).map_err(val_err)?,
        ))
    }

    /// Returns whether `point` lies on the surface within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// Projects `point` onto the surface, returning `(u, v, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64, f64) {
        let proj = self.0.project_point(point.0, tolerance(tol));
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
        let tol = tolerance(tol);
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.u, p.v, p.distance))
            .collect()
    }

    /// Intersects this plane with another shape, using the exact solver
    /// for the type of `other`. Returns a `(kind, payload)` tuple:
    ///
    /// - `Plane`: Returns `("line", Line3D)`, `("parallel", None)` or
    ///   `("coincident", None)`.
    /// - `Cylinder`: Returns `("circle", Circle3D)`, `("ellipse", Ellipse3D)`,
    ///   `("two_lines", (Line3D, Line3D))`, `("tangent_line", Line3D)` or
    ///   `("empty", None)`.
    /// - `Cone`: Returns `("circle", Circle3D)`, `("ellipse", Ellipse3D)`,
    ///   `("parabola", Parabola3D)`,
    ///   `("hyperbola", (Hyperbola3D, Hyperbola3D))` (one branch per nappe),
    ///   `("two_lines", (Line3D, Line3D))`, `("tangent_line", Line3D)` or
    ///   `("apex_point", Point3D)`.
    /// - `Sphere`: Returns `("circle", Circle3D)`, `("tangent_point", Point3D)`
    ///   or `("empty", None)`.
    /// - `Torus`: Returns `("two_circles", (Circle3D, Circle3D))`,
    ///   `("tangent_circle", Circle3D)`, `("circle", Circle3D)`,
    ///   `("empty", None)` or `("not_analytic", None)`.
    /// - `Line3D`: Returns `("point", (t, Point3D))`, `("parallel", None)` or
    ///   `("coincident", None)`.
    /// - `Circle3D`: Returns `("circle", Circle3D)`,
    ///   `("hits", [(t, Point3D, multiplicity)])` or `("empty", None)`.
    /// - `Ellipse3D`, `Parabola3D`, `Hyperbola3D`: Returns
    ///   `("hits", [(t, Point3D, multiplicity)])`, `("coincident", None)` or
    ///   `("empty", None)`.
    ///
    /// The payload is the same whichever shape is the receiver, so parameters
    /// keep their roles: in `circle.intersect(line)` each `t` is still a line
    /// parameter. Raises `TypeError` for any other type.
    #[pyo3(signature = (other, tol = None))]
    fn intersect(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tolerance(tol);
        dispatch_intersect!(other, "Plane", "Plane, Cylinder, Cone, Sphere, Torus, Line3D, Circle3D, Ellipse3D, Parabola3D or Hyperbola3D", {
            PyPlane => |o| self.plane_result(py, &o, tol),
            PyCylinder => |o| self.cylinder_result(py, &o, tol),
            PyCone => |o| self.cone_result(py, &o, tol),
            PySphere => |o| self.sphere_result(py, &o, tol),
            PyTorus => |o| o.plane_result(py, self, tol),
            PyLine3D => |o| o.plane_result(py, self, tol),
            PyCircle3D => |o| o.plane_result(py, self, tol),
            PyEllipse3D => |o| o.plane_result(py, self, tol),
            PyParabola3D => |o| o.plane_result(py, self, tol),
            PyHyperbola3D => |o| o.plane_result(py, self, tol),
        })
    }

    /// Intersects this plane with another plane.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (other, tol = None))]
    fn intersect_plane(
        &self,
        py: Python<'_>,
        other: &PyPlane,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Plane.intersect_plane")?;
        self.plane_result(py, other, tolerance(tol))
    }

    /// Intersects this plane with a sphere.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (sphere, tol = None))]
    fn intersect_sphere(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Plane.intersect_sphere")?;
        self.sphere_result(py, sphere, tolerance(tol))
    }

    /// Intersects this plane with a cylinder.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (cylinder, tol = None))]
    fn intersect_cylinder(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Plane.intersect_cylinder")?;
        self.cylinder_result(py, cylinder, tolerance(tol))
    }

    /// Intersects this plane with a cone (both nappes).
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (cone, tol = None))]
    fn intersect_cone(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Plane.intersect_cone")?;
        self.cone_result(py, cone, tolerance(tol))
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyPlane {
    pub(crate) fn plane_result(
        &self,
        py: Python<'_>,
        other: &PyPlane,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&other.0, tol) {
            PlanePlaneIntersection::Line(l) => Ok((
                "line".to_string(),
                PyLine3D(l).into_pyobject(py)?.into_any().unbind(),
            )),
            PlanePlaneIntersection::Parallel => Ok(("parallel".to_string(), py.None())),
            PlanePlaneIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
        }
    }

    pub(crate) fn sphere_result(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&sphere.0, tol) {
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

    pub(crate) fn cylinder_result(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&cylinder.0, tol) {
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

    pub(crate) fn cone_result(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&cone.0, tol) {
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
            PlaneConeIntersection::Hyperbola(h1, h2) => Ok((
                "hyperbola".to_string(),
                (
                    PyHyperbola3D(h1).into_pyobject(py)?.into_any().unbind(),
                    PyHyperbola3D(h2).into_pyobject(py)?.into_any().unbind(),
                )
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
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
        }
    }
}

/// An infinite circular cylinder; u is the angle around the axis, v the
/// height along it.
#[pyclass(name = "Cylinder", module = "geomcore.surfaces", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyCylinder(pub(crate) Cylinder);

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

    /// Image of this cylinder under `transform` (raises ValueError for a
    /// collapsing transform).
    #[pyo3(signature = (transform,))]
    fn transformed(&self, transform: &PyTransform) -> PyResult<Self> {
        Ok(PyCylinder(
            self.0.transformed(transform.0).map_err(val_err)?,
        ))
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
    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> PyResult<PyVector3D> {
        check_surface_order(du, dv)?;
        Ok(PyVector3D(self.0.eval_derivative(u, v, du, dv)))
    }

    /// Recover the (u, v) parameters of a point lying on the cylinder.
    fn parameters_of(&self, point: PyPoint3D) -> (f64, f64) {
        self.0.parameters_of(point.0)
    }

    /// The (first, last) `u` parameter bounds (a whole turn).
    fn u_bounds(&self) -> (f64, f64) {
        ParametricSurface::u_bounds(&self.0)
    }

    /// The (first, last) `v` parameter bounds (unbounded here).
    fn v_bounds(&self) -> (f64, f64) {
        ParametricSurface::v_bounds(&self.0)
    }

    /// The `u` isoparametric generator line at angle `u`, parametrized by `v` like the surface:
    /// `u_iso(u).eval_point(t)` is `eval_point(u, t)` (raises `ValueError` for
    /// a non-finite parameter).
    fn u_iso(&self, u: f64) -> PyResult<PyLine3D> {
        Ok(PyLine3D(self.0.u_iso(u).map_err(val_err)?))
    }

    /// The `v` isoparametric section circle at height `v`, parametrized by `u` like the surface:
    /// `v_iso(v).eval_point(t)` is `eval_point(t, v)` (raises `ValueError` for
    /// a non-finite parameter).
    fn v_iso(&self, v: f64) -> PyResult<PyCircle3D> {
        Ok(PyCircle3D(self.0.v_iso(v).map_err(val_err)?))
    }

    /// The axis-aligned box around the shape (`None` when unbounded).
    fn bounding_box(&self) -> Option<PyBoundingBox3D> {
        self.0.bounding_box().map(PyBoundingBox3D)
    }

    /// The patch `u_range × v_range` as an exact rational B-spline
    /// surface (raises `ValueError` for a bad range).
    fn to_bspline_patch(
        &self,
        u_range: (f64, f64),
        v_range: (f64, f64),
    ) -> PyResult<PyBSplineSurface> {
        Ok(PyBSplineSurface(
            self.0.to_bspline_patch(u_range, v_range).map_err(val_err)?,
        ))
    }

    /// Returns whether `point` lies on the surface within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// Projects `point` onto the surface, returning `(u, v, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64, f64) {
        let proj = self.0.project_point(point.0, tolerance(tol));
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
        let tol = tolerance(tol);
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.u, p.v, p.distance))
            .collect()
    }

    /// Intersects this cylinder with another shape, using the exact solver
    /// for the type of `other`. Returns a `(kind, payload)` tuple:
    ///
    /// - `Plane`: Returns `("circle", Circle3D)`, `("ellipse", Ellipse3D)`,
    ///   `("two_lines", (Line3D, Line3D))`, `("tangent_line", Line3D)` or
    ///   `("empty", None)`.
    /// - `Cylinder`: The analytic path needs parallel axes. Returns
    ///   `("two_lines", (Line3D, Line3D))`, `("tangent_line", Line3D)`,
    ///   `("empty", None)`, `("coincident", None)` or `("not_analytic", None)`
    ///   (no closed form; numeric path pending).
    /// - `Cone`: The analytic path needs coaxial axes. Returns
    ///   `("two_circles", (Circle3D, Circle3D))` (one ring per nappe),
    ///   `("apex_point", Point3D)` or `("not_analytic", None)` (no closed form;
    ///   numeric path pending).
    /// - `Sphere`: The analytic path needs the cylinder axis through the sphere
    ///   center. Returns `("circle", Circle3D)`,
    ///   `("two_circles", (Circle3D, Circle3D))`, `("empty", None)` or
    ///   `("not_analytic", None)` (no closed form; numeric path pending).
    /// - `Torus`: The analytic path needs coaxial axes. Returns
    ///   `("two_circles", (Circle3D, Circle3D))`,
    ///   `("tangent_circle", Circle3D)`, `("empty", None)` or
    ///   `("not_analytic", None)`.
    /// - `Line3D`: Returns `("two_points", ((t1, Point3D), (t2, Point3D)))`,
    ///   `("one_point", (t, Point3D))`, `("tangent", (t, Point3D))`,
    ///   `("empty", None)` or `("coincident", None)` when the line is a
    ///   generator.
    /// - `Circle3D`: Returns `("circle", Circle3D)`,
    ///   `("hits", [(t, Point3D, multiplicity)])` or `("empty", None)`.
    /// - `Ellipse3D`, `Parabola3D`, `Hyperbola3D`: Returns
    ///   `("hits", [(t, Point3D, multiplicity)])`, `("coincident", None)` or
    ///   `("empty", None)`.
    ///
    /// The payload is the same whichever shape is the receiver, so parameters
    /// keep their roles: in `circle.intersect(line)` each `t` is still a line
    /// parameter. Raises `TypeError` for any other type.
    #[pyo3(signature = (other, tol = None))]
    fn intersect(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tolerance(tol);
        dispatch_intersect!(other, "Cylinder", "Plane, Cylinder, Cone, Sphere, Torus, Line3D, Circle3D, Ellipse3D, Parabola3D or Hyperbola3D", {
            PyPlane => |o| o.cylinder_result(py, self, tol),
            PyCylinder => |o| self.cylinder_result(py, &o, tol),
            PyCone => |o| o.cylinder_result(py, self, tol),
            PySphere => |o| o.cylinder_result(py, self, tol),
            PyTorus => |o| o.cylinder_result(py, self, tol),
            PyLine3D => |o| o.cylinder_result(py, self, tol),
            PyCircle3D => |o| o.cylinder_result(py, self, tol),
            PyEllipse3D => |o| o.cylinder_result(py, self, tol),
            PyParabola3D => |o| o.cylinder_result(py, self, tol),
            PyHyperbola3D => |o| o.cylinder_result(py, self, tol),
        })
    }

    /// Intersects this cylinder with another cylinder.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (other, tol = None))]
    fn intersect_cylinder(
        &self,
        py: Python<'_>,
        other: &PyCylinder,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Cylinder.intersect_cylinder")?;
        self.cylinder_result(py, other, tolerance(tol))
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyCylinder {
    pub(crate) fn cylinder_result(
        &self,
        py: Python<'_>,
        other: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&other.0, tol) {
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
#[pyclass(name = "Cone", module = "geomcore.surfaces", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyCone(pub(crate) Cone);

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

    /// Image of this cone under `transform` (raises ValueError for a
    /// collapsing transform).
    #[pyo3(signature = (transform,))]
    fn transformed(&self, transform: &PyTransform) -> PyResult<Self> {
        Ok(PyCone(self.0.transformed(transform.0).map_err(val_err)?))
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
    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> PyResult<PyVector3D> {
        check_surface_order(du, dv)?;
        Ok(PyVector3D(self.0.eval_derivative(u, v, du, dv)))
    }

    /// Recover the (u, v) parameters of a point lying on the cone.
    fn parameters_of(&self, point: PyPoint3D) -> (f64, f64) {
        self.0.parameters_of(point.0)
    }

    /// The (first, last) `u` parameter bounds (a whole turn).
    fn u_bounds(&self) -> (f64, f64) {
        ParametricSurface::u_bounds(&self.0)
    }

    /// The (first, last) `v` parameter bounds (unbounded here).
    fn v_bounds(&self) -> (f64, f64) {
        ParametricSurface::v_bounds(&self.0)
    }

    /// The `u` isoparametric generator line at angle `u`, parametrized by `v` like the surface:
    /// `u_iso(u).eval_point(t)` is `eval_point(u, t)` (raises `ValueError` for
    /// a non-finite parameter).
    fn u_iso(&self, u: f64) -> PyResult<PyLine3D> {
        Ok(PyLine3D(self.0.u_iso(u).map_err(val_err)?))
    }

    /// The `v` isoparametric section circle at `v` (radius zero at the apex), parametrized by `u` like the surface:
    /// `v_iso(v).eval_point(t)` is `eval_point(t, v)` (raises `ValueError` for
    /// a non-finite parameter).
    fn v_iso(&self, v: f64) -> PyResult<PyCircle3D> {
        Ok(PyCircle3D(self.0.v_iso(v).map_err(val_err)?))
    }

    /// The axis-aligned box around the shape (`None` when unbounded).
    fn bounding_box(&self) -> Option<PyBoundingBox3D> {
        self.0.bounding_box().map(PyBoundingBox3D)
    }

    /// The patch `u_range × v_range` as an exact rational B-spline
    /// surface (raises `ValueError` for a bad range).
    fn to_bspline_patch(
        &self,
        u_range: (f64, f64),
        v_range: (f64, f64),
    ) -> PyResult<PyBSplineSurface> {
        Ok(PyBSplineSurface(
            self.0.to_bspline_patch(u_range, v_range).map_err(val_err)?,
        ))
    }

    /// Returns whether `point` lies on the surface within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// Projects `point` onto the surface, returning `(u, v, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64, f64) {
        let proj = self.0.project_point(point.0, tolerance(tol));
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
        let tol = tolerance(tol);
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.u, p.v, p.distance))
            .collect()
    }

    /// Intersects this cone with another shape, using the exact solver
    /// for the type of `other`. Returns a `(kind, payload)` tuple:
    ///
    /// - `Plane`: Returns `("circle", Circle3D)`, `("ellipse", Ellipse3D)`,
    ///   `("parabola", Parabola3D)`,
    ///   `("hyperbola", (Hyperbola3D, Hyperbola3D))` (one branch per nappe),
    ///   `("two_lines", (Line3D, Line3D))`, `("tangent_line", Line3D)` or
    ///   `("apex_point", Point3D)`.
    /// - `Cylinder`: The analytic path needs coaxial axes. Returns
    ///   `("two_circles", (Circle3D, Circle3D))` (one ring per nappe),
    ///   `("apex_point", Point3D)` or `("not_analytic", None)` (no closed form;
    ///   numeric path pending).
    /// - `Cone`: The analytic path needs coaxial axes. Returns
    ///   `("circle", Circle3D)`, `("two_circles", (Circle3D, Circle3D))`,
    ///   `("apex_point", Point3D)`, `("coincident", None)` or
    ///   `("not_analytic", None)`.
    /// - `Sphere`: The analytic path needs the sphere center on the cone axis.
    ///   Returns `("two_circles", (Circle3D, Circle3D))`,
    ///   `("tangent_circle", Circle3D)`, `("empty", None)` or
    ///   `("not_analytic", None)` (no closed form; numeric path pending).
    /// - `Torus`: The analytic path needs coaxial axes. Returns
    ///   `("rings", [(Circle3D, tangent), ...])` with up to four rings sorted
    ///   along the torus axis, `("empty", None)` or `("not_analytic", None)`.
    /// - `Line3D`: Returns `("two_points", ((t1, Point3D), (t2, Point3D)))`,
    ///   `("one_point", (t, Point3D))`, `("tangent", (t, Point3D))`,
    ///   `("empty", None)` or `("coincident", None)` when the line is a
    ///   generator. Roots behind the apex are filtered, so a single surviving
    ///   hit reports `("one_point", (t, Point3D))`.
    /// - `Circle3D`: Returns `("circle", Circle3D)`,
    ///   `("hits", [(t, Point3D, multiplicity)])` or `("empty", None)`.
    /// - `Ellipse3D`, `Parabola3D`, `Hyperbola3D`: Returns
    ///   `("hits", [(t, Point3D, multiplicity)])`, `("coincident", None)` or
    ///   `("empty", None)`.
    ///
    /// The payload is the same whichever shape is the receiver, so parameters
    /// keep their roles: in `circle.intersect(line)` each `t` is still a line
    /// parameter. Raises `TypeError` for any other type.
    #[pyo3(signature = (other, tol = None))]
    fn intersect(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tolerance(tol);
        dispatch_intersect!(other, "Cone", "Plane, Cylinder, Cone, Sphere, Torus, Line3D, Circle3D, Ellipse3D, Parabola3D or Hyperbola3D", {
            PyPlane => |o| o.cone_result(py, self, tol),
            PyCylinder => |o| self.cylinder_result(py, &o, tol),
            PyCone => |o| self.cone_result(py, &o, tol),
            PySphere => |o| o.cone_result(py, self, tol),
            PyTorus => |o| o.cone_result(py, self, tol),
            PyLine3D => |o| o.cone_result(py, self, tol),
            PyCircle3D => |o| o.cone_result(py, self, tol),
            PyEllipse3D => |o| o.cone_result(py, self, tol),
            PyParabola3D => |o| o.cone_result(py, self, tol),
            PyHyperbola3D => |o| o.cone_result(py, self, tol),
        })
    }

    /// Intersects this cone with a cylinder.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (cylinder, tol = None))]
    fn intersect_cylinder(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Cone.intersect_cylinder")?;
        self.cylinder_result(py, cylinder, tolerance(tol))
    }

    /// Intersects this cone with another cone.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (other, tol = None))]
    fn intersect_cone(
        &self,
        py: Python<'_>,
        other: &PyCone,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Cone.intersect_cone")?;
        self.cone_result(py, other, tolerance(tol))
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyCone {
    pub(crate) fn cylinder_result(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&cylinder.0, tol) {
            ConeCylinderIntersection::TwoCircles(c1, c2) => {
                Ok(("two_circles".to_string(), circles_to_py(py, c1, c2)?))
            }
            ConeCylinderIntersection::ApexPoint(p) => Ok((
                "apex_point".to_string(),
                PyPoint3D(p).into_pyobject(py)?.into_any().unbind(),
            )),
            ConeCylinderIntersection::NotAnalytic => Ok(("not_analytic".to_string(), py.None())),
        }
    }

    pub(crate) fn cone_result(
        &self,
        py: Python<'_>,
        other: &PyCone,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&other.0, tol) {
            ConeConeIntersection::Circle(c) => Ok((
                "circle".to_string(),
                PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
            )),
            ConeConeIntersection::TwoCircles(c1, c2) => {
                Ok(("two_circles".to_string(), circles_to_py(py, c1, c2)?))
            }
            ConeConeIntersection::ApexPoint(p) => Ok((
                "apex_point".to_string(),
                PyPoint3D(p).into_pyobject(py)?.into_any().unbind(),
            )),
            ConeConeIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
            ConeConeIntersection::NotAnalytic => Ok(("not_analytic".to_string(), py.None())),
        }
    }
}

/// A sphere; u is the longitude in [0, 2*pi), v the latitude in [-pi/2, pi/2].
#[pyclass(name = "Sphere", module = "geomcore.surfaces", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PySphere(pub(crate) Sphere);

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

    /// The surface area (`4*PI*radius^2`).
    fn area(&self) -> f64 {
        self.0.area()
    }

    /// The enclosed volume (`4/3*PI*radius^3`).
    fn volume(&self) -> f64 {
        self.0.volume()
    }

    /// Image of this sphere under `transform` (raises ValueError for a
    /// collapsing transform).
    #[pyo3(signature = (transform,))]
    fn transformed(&self, transform: &PyTransform) -> PyResult<Self> {
        Ok(PySphere(self.0.transformed(transform.0).map_err(val_err)?))
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
    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> PyResult<PyVector3D> {
        check_surface_order(du, dv)?;
        Ok(PyVector3D(self.0.eval_derivative(u, v, du, dv)))
    }

    /// Recover the (u, v) parameters of a point lying on the sphere.
    fn parameters_of(&self, point: PyPoint3D) -> (f64, f64) {
        self.0.parameters_of(point.0)
    }

    /// The (first, last) `u` parameter bounds (a whole turn).
    fn u_bounds(&self) -> (f64, f64) {
        ParametricSurface::u_bounds(&self.0)
    }

    /// The (first, last) `v` parameter bounds.
    fn v_bounds(&self) -> (f64, f64) {
        ParametricSurface::v_bounds(&self.0)
    }

    /// The `u` isoparametric meridian circle at longitude `u`, parametrized by `v` like the surface:
    /// `u_iso(u).eval_point(t)` is `eval_point(u, t)` (raises `ValueError` for
    /// a non-finite parameter).
    fn u_iso(&self, u: f64) -> PyResult<PyCircle3D> {
        Ok(PyCircle3D(self.0.u_iso(u).map_err(val_err)?))
    }

    /// The `v` isoparametric parallel circle at latitude `v` (radius zero at a pole), parametrized by `u` like the surface:
    /// `v_iso(v).eval_point(t)` is `eval_point(t, v)` (raises `ValueError` for
    /// a non-finite parameter).
    fn v_iso(&self, v: f64) -> PyResult<PyCircle3D> {
        Ok(PyCircle3D(self.0.v_iso(v).map_err(val_err)?))
    }

    /// The axis-aligned box around the shape (`None` when unbounded).
    fn bounding_box(&self) -> Option<PyBoundingBox3D> {
        self.0.bounding_box().map(PyBoundingBox3D)
    }

    /// The whole sphere as an exact rational B-spline surface.
    fn to_bspline(&self) -> PyBSplineSurface {
        PyBSplineSurface(self.0.to_bspline())
    }

    /// The patch `u_range × v_range` as an exact rational B-spline
    /// surface (raises `ValueError` for a bad range).
    fn to_bspline_patch(
        &self,
        u_range: (f64, f64),
        v_range: (f64, f64),
    ) -> PyResult<PyBSplineSurface> {
        Ok(PyBSplineSurface(
            self.0.to_bspline_patch(u_range, v_range).map_err(val_err)?,
        ))
    }

    /// Returns whether `point` lies on the surface within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// Projects `point` onto the surface, returning `(u, v, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64, f64) {
        let proj = self.0.project_point(point.0, tolerance(tol));
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
        let tol = tolerance(tol);
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.u, p.v, p.distance))
            .collect()
    }

    /// Intersects this sphere with another shape, using the exact solver
    /// for the type of `other`. Returns a `(kind, payload)` tuple:
    ///
    /// - `Plane`: Returns `("circle", Circle3D)`, `("tangent_point", Point3D)`
    ///   or `("empty", None)`.
    /// - `Cylinder`: The analytic path needs the cylinder axis through the
    ///   sphere center. Returns `("circle", Circle3D)`,
    ///   `("two_circles", (Circle3D, Circle3D))`, `("empty", None)` or
    ///   `("not_analytic", None)` (no closed form; numeric path pending).
    /// - `Cone`: The analytic path needs the sphere center on the cone axis.
    ///   Returns `("two_circles", (Circle3D, Circle3D))`,
    ///   `("tangent_circle", Circle3D)`, `("empty", None)` or
    ///   `("not_analytic", None)` (no closed form; numeric path pending).
    /// - `Sphere`: Returns `("circle", Circle3D)`,
    ///   `("tangent_point", Point3D)`, `("empty", None)` or
    ///   `("coincident", None)`.
    /// - `Torus`: The analytic path needs the sphere center on the torus axis.
    ///   Returns `("circle", Circle3D)`,
    ///   `("two_circles", (Circle3D, Circle3D))`,
    ///   `("tangent_circle", Circle3D)`, `("empty", None)` or
    ///   `("not_analytic", None)`.
    /// - `Line3D`: Returns `("two_points", ((t1, Point3D), (t2, Point3D)))`,
    ///   `("one_point", (t, Point3D))`, `("tangent", (t, Point3D))`,
    ///   `("empty", None)` or `("coincident", None)` (the latter never triggers
    ///   for spheres, which contain no line).
    /// - `Circle3D`: Returns `("circle", Circle3D)`,
    ///   `("hits", [(t, Point3D, multiplicity)])` or `("empty", None)`.
    /// - `Ellipse3D`, `Parabola3D`, `Hyperbola3D`: Returns
    ///   `("hits", [(t, Point3D, multiplicity)])`, `("coincident", None)` or
    ///   `("empty", None)`.
    ///
    /// The payload is the same whichever shape is the receiver, so parameters
    /// keep their roles: in `circle.intersect(line)` each `t` is still a line
    /// parameter. Raises `TypeError` for any other type.
    #[pyo3(signature = (other, tol = None))]
    fn intersect(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tolerance(tol);
        dispatch_intersect!(other, "Sphere", "Plane, Cylinder, Cone, Sphere, Torus, Line3D, Circle3D, Ellipse3D, Parabola3D or Hyperbola3D", {
            PyPlane => |o| o.sphere_result(py, self, tol),
            PyCylinder => |o| self.cylinder_result(py, &o, tol),
            PyCone => |o| self.cone_result(py, &o, tol),
            PySphere => |o| self.sphere_result(py, &o, tol),
            PyTorus => |o| o.sphere_result(py, self, tol),
            PyLine3D => |o| o.sphere_result(py, self, tol),
            PyCircle3D => |o| o.sphere_result(py, self, tol),
            PyEllipse3D => |o| o.sphere_result(py, self, tol),
            PyParabola3D => |o| o.sphere_result(py, self, tol),
            PyHyperbola3D => |o| o.sphere_result(py, self, tol),
        })
    }

    /// Intersects this sphere with another sphere.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (other, tol = None))]
    fn intersect_sphere(
        &self,
        py: Python<'_>,
        other: &PySphere,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Sphere.intersect_sphere")?;
        self.sphere_result(py, other, tolerance(tol))
    }

    /// Intersects this sphere with a cylinder.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (cylinder, tol = None))]
    fn intersect_cylinder(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Sphere.intersect_cylinder")?;
        self.cylinder_result(py, cylinder, tolerance(tol))
    }

    /// Intersects this sphere with a cone.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (cone, tol = None))]
    fn intersect_cone(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Sphere.intersect_cone")?;
        self.cone_result(py, cone, tolerance(tol))
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PySphere {
    pub(crate) fn sphere_result(
        &self,
        py: Python<'_>,
        other: &PySphere,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&other.0, tol) {
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

    pub(crate) fn cylinder_result(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&cylinder.0, tol) {
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

    pub(crate) fn cone_result(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&cone.0, tol) {
            SphereConeIntersection::TwoCircles(c1, c2) => {
                Ok(("two_circles".to_string(), circles_to_py(py, c1, c2)?))
            }
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
#[pyclass(name = "Torus", module = "geomcore.surfaces", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyTorus(pub(crate) Torus);

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

    /// Image of this torus under `transform` (raises ValueError for a
    /// collapsing transform).
    #[pyo3(signature = (transform,))]
    fn transformed(&self, transform: &PyTransform) -> PyResult<Self> {
        Ok(PyTorus(self.0.transformed(transform.0).map_err(val_err)?))
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
    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> PyResult<PyVector3D> {
        check_surface_order(du, dv)?;
        Ok(PyVector3D(self.0.eval_derivative(u, v, du, dv)))
    }

    /// Recover the (u, v) parameters of a point lying on the torus.
    fn parameters_of(&self, point: PyPoint3D) -> (f64, f64) {
        self.0.parameters_of(point.0)
    }

    /// The (first, last) `u` parameter bounds (a whole turn).
    fn u_bounds(&self) -> (f64, f64) {
        ParametricSurface::u_bounds(&self.0)
    }

    /// The (first, last) `v` parameter bounds (a whole turn).
    fn v_bounds(&self) -> (f64, f64) {
        ParametricSurface::v_bounds(&self.0)
    }

    /// The `u` isoparametric meridian circle of the tube at angle `u`, parametrized by `v` like the surface:
    /// `u_iso(u).eval_point(t)` is `eval_point(u, t)` (raises `ValueError` for
    /// a non-finite parameter).
    fn u_iso(&self, u: f64) -> PyResult<PyCircle3D> {
        Ok(PyCircle3D(self.0.u_iso(u).map_err(val_err)?))
    }

    /// The `v` isoparametric parallel circle at tube angle `v` (radius zero at a pinch), parametrized by `u` like the surface:
    /// `v_iso(v).eval_point(t)` is `eval_point(t, v)` (raises `ValueError` for
    /// a non-finite parameter).
    fn v_iso(&self, v: f64) -> PyResult<PyCircle3D> {
        Ok(PyCircle3D(self.0.v_iso(v).map_err(val_err)?))
    }

    /// The axis-aligned box around the shape (`None` when unbounded).
    fn bounding_box(&self) -> Option<PyBoundingBox3D> {
        self.0.bounding_box().map(PyBoundingBox3D)
    }

    /// The whole torus as an exact rational B-spline surface.
    fn to_bspline(&self) -> PyBSplineSurface {
        PyBSplineSurface(self.0.to_bspline())
    }

    /// The patch `u_range × v_range` as an exact rational B-spline
    /// surface (raises `ValueError` for a bad range).
    fn to_bspline_patch(
        &self,
        u_range: (f64, f64),
        v_range: (f64, f64),
    ) -> PyResult<PyBSplineSurface> {
        Ok(PyBSplineSurface(
            self.0.to_bspline_patch(u_range, v_range).map_err(val_err)?,
        ))
    }

    /// Returns whether `point` lies on the surface within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// Projects `point` onto the surface, returning `(u, v, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64, f64) {
        let proj = self.0.project_point(point.0, tolerance(tol));
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
        let tol = tolerance(tol);
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.u, p.v, p.distance))
            .collect()
    }

    /// Intersects this torus with another shape, using the exact solver
    /// for the type of `other`. Returns a `(kind, payload)` tuple:
    ///
    /// - `Plane`: Returns `("two_circles", (Circle3D, Circle3D))`,
    ///   `("tangent_circle", Circle3D)`, `("circle", Circle3D)`,
    ///   `("empty", None)` or `("not_analytic", None)`.
    /// - `Cylinder`: The analytic path needs coaxial axes. Returns
    ///   `("two_circles", (Circle3D, Circle3D))`,
    ///   `("tangent_circle", Circle3D)`, `("empty", None)` or
    ///   `("not_analytic", None)`.
    /// - `Cone`: The analytic path needs coaxial axes. Returns
    ///   `("rings", [(Circle3D, tangent), ...])` with up to four rings sorted
    ///   along the torus axis, `("empty", None)` or `("not_analytic", None)`.
    /// - `Sphere`: The analytic path needs the sphere center on the torus axis.
    ///   Returns `("circle", Circle3D)`,
    ///   `("two_circles", (Circle3D, Circle3D))`,
    ///   `("tangent_circle", Circle3D)`, `("empty", None)` or
    ///   `("not_analytic", None)`.
    /// - `Torus`: The analytic path needs collinear axes. Returns
    ///   `("circle", Circle3D)`, `("two_circles", (Circle3D, Circle3D))`,
    ///   `("tangent_circle", Circle3D)`, `("empty", None)`,
    ///   `("coincident", None)` or `("not_analytic", None)`.
    /// - `Line3D`: Returns `("hits", [(t, Point3D, (u, v), multiplicity)])` (up
    ///   to four hits ordered by line parameter) or `("empty", None)`.
    ///
    /// The payload is the same whichever shape is the receiver, so parameters
    /// keep their roles: in `circle.intersect(line)` each `t` is still a line
    /// parameter. Raises `TypeError` for any other type.
    #[pyo3(signature = (other, tol = None))]
    fn intersect(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tolerance(tol);
        dispatch_intersect!(other, "Torus", "Plane, Cylinder, Cone, Sphere, Torus or Line3D", {
            PyPlane => |o| self.plane_result(py, &o, tol),
            PyCylinder => |o| self.cylinder_result(py, &o, tol),
            PyCone => |o| self.cone_result(py, &o, tol),
            PySphere => |o| self.sphere_result(py, &o, tol),
            PyTorus => |o| self.torus_result(py, &o, tol),
            PyLine3D => |o| o.torus_result(py, self, tol),
        })
    }

    /// Intersects this torus with a plane.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (plane, tol = None))]
    fn intersect_plane(
        &self,
        py: Python<'_>,
        plane: &PyPlane,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Torus.intersect_plane")?;
        self.plane_result(py, plane, tolerance(tol))
    }

    /// Intersects this torus with a sphere.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (sphere, tol = None))]
    fn intersect_sphere(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Torus.intersect_sphere")?;
        self.sphere_result(py, sphere, tolerance(tol))
    }

    /// Intersects this torus with a cylinder.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (cylinder, tol = None))]
    fn intersect_cylinder(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Torus.intersect_cylinder")?;
        self.cylinder_result(py, cylinder, tolerance(tol))
    }

    /// Intersects this torus with a cone (both nappes).
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (cone, tol = None))]
    fn intersect_cone(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Torus.intersect_cone")?;
        self.cone_result(py, cone, tolerance(tol))
    }

    /// Intersects this torus with another torus.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (other, tol = None))]
    fn intersect_torus(
        &self,
        py: Python<'_>,
        other: &PyTorus,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Torus.intersect_torus")?;
        self.torus_result(py, other, tolerance(tol))
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyTorus {
    pub(crate) fn plane_result(
        &self,
        py: Python<'_>,
        plane: &PyPlane,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&plane.0, tol) {
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

    pub(crate) fn sphere_result(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&sphere.0, tol) {
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

    pub(crate) fn cylinder_result(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&cylinder.0, tol) {
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

    pub(crate) fn cone_result(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&cone.0, tol) {
            TorusConeIntersection::Rings(rings) => Ok((
                "rings".to_string(),
                rings
                    .into_iter()
                    .map(|r| (PyCircle3D(r.circle), r.tangent))
                    .collect::<Vec<_>>()
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            )),
            TorusConeIntersection::Empty => Ok(("empty".to_string(), py.None())),
            TorusConeIntersection::NotAnalytic => Ok(("not_analytic".to_string(), py.None())),
        }
    }

    pub(crate) fn torus_result(
        &self,
        py: Python<'_>,
        other: &PyTorus,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&other.0, tol) {
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
#[pyclass(name = "BSplineSurface", module = "geomcore.surfaces", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyBSplineSurface(pub(crate) BSplineSurface);

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

    /// The number of poles in the `u` direction (rows of the net).
    fn u_pole_count(&self) -> usize {
        self.0.u_pole_count()
    }

    /// The number of poles in the `v` direction (columns of the net).
    fn v_pole_count(&self) -> usize {
        self.0.v_pole_count()
    }

    /// The control poles as `poles[i_u][i_v]`.
    fn poles(&self) -> Vec<Vec<PyPoint3D>> {
        let nv = self.0.v_pole_count();
        self.0
            .poles()
            .chunks(nv)
            .map(|row| row.iter().map(|p| PyPoint3D(*p)).collect())
            .collect()
    }

    /// The per-pole weights laid out as `poles`, or `None` when the
    /// surface is not rational.
    fn weights(&self) -> Option<Vec<Vec<f64>>> {
        let nv = self.0.v_pole_count();
        self.0
            .weights()
            .map(|flat| flat.chunks(nv).map(|row| row.to_vec()).collect::<Vec<_>>())
    }

    /// The distinct knots in the `u` direction.
    fn u_knots(&self) -> Vec<f64> {
        self.0.u_knots().to_vec()
    }

    /// The multiplicity of each `u` knot.
    fn u_multiplicities(&self) -> Vec<u32> {
        self.0.u_multiplicities().to_vec()
    }

    /// The distinct knots in the `v` direction.
    fn v_knots(&self) -> Vec<f64> {
        self.0.v_knots().to_vec()
    }

    /// The multiplicity of each `v` knot.
    fn v_multiplicities(&self) -> Vec<u32> {
        self.0.v_multiplicities().to_vec()
    }

    /// The (first, last) `u` parameter bounds.
    fn u_bounds(&self) -> (f64, f64) {
        ParametricSurface::u_bounds(&self.0)
    }

    /// The (first, last) `v` parameter bounds.
    fn v_bounds(&self) -> (f64, f64) {
        ParametricSurface::v_bounds(&self.0)
    }

    /// The `u` isoparametric B-spline curve at `u`, blended from the pole net, parametrized by `v` like the surface:
    /// `u_iso(u).eval_point(t)` is `eval_point(u, t)` (raises `ValueError` for
    /// a non-finite parameter or one outside a non-periodic direction (a periodic one wraps)).
    fn u_iso(&self, u: f64) -> PyResult<PyBSplineCurve3D> {
        Ok(PyBSplineCurve3D(self.0.u_iso(u).map_err(val_err)?))
    }

    /// The `v` isoparametric B-spline curve at `v`, blended from the pole net, parametrized by `u` like the surface:
    /// `v_iso(v).eval_point(t)` is `eval_point(t, v)` (raises `ValueError` for
    /// a non-finite parameter or one outside a non-periodic direction (a periodic one wraps)).
    fn v_iso(&self, v: f64) -> PyResult<PyBSplineCurve3D> {
        Ok(PyBSplineCurve3D(self.0.v_iso(v).map_err(val_err)?))
    }

    /// The axis-aligned box around the shape (`None` when unbounded).
    fn bounding_box(&self) -> Option<PyBoundingBox3D> {
        self.0.bounding_box().map(PyBoundingBox3D)
    }

    /// Least-squares B-spline fit through the grid `points[i][j]`.
    ///
    /// `parametrization` is `"centripetal"` (default) or `"chordal"`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (points, u_degree, v_degree, u_pole_count, v_pole_count, parametrization = "centripetal"))]
    fn approximate(
        points: Vec<Vec<PyPoint3D>>,
        u_degree: usize,
        v_degree: usize,
        u_pole_count: usize,
        v_pole_count: usize,
        parametrization: &str,
    ) -> PyResult<Self> {
        let param = parse_parametrization(parametrization)?;
        let points = points
            .into_iter()
            .map(|row| row.into_iter().map(|p| p.0).collect())
            .collect::<Vec<Vec<_>>>();
        Ok(PyBSplineSurface(
            BSplineSurface::approximate(
                &points,
                u_degree,
                v_degree,
                u_pole_count,
                v_pole_count,
                param,
            )
            .map_err(val_err)?,
        ))
    }

    /// Least-squares B-spline fit through the grid `points[i][j]` with as
    /// few poles as keep every point within `max_deviation` of the
    /// surface.
    ///
    /// `parametrization` is `"centripetal"` (default) or `"chordal"`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (points, u_degree, v_degree, max_deviation, parametrization = "centripetal"))]
    fn approximate_within(
        points: Vec<Vec<PyPoint3D>>,
        u_degree: usize,
        v_degree: usize,
        max_deviation: f64,
        parametrization: &str,
    ) -> PyResult<Self> {
        let param = parse_parametrization(parametrization)?;
        let points = points
            .into_iter()
            .map(|row| row.into_iter().map(|p| p.0).collect())
            .collect::<Vec<Vec<_>>>();
        Ok(PyBSplineSurface(
            BSplineSurface::approximate_within(&points, u_degree, v_degree, max_deviation, param)
                .map_err(val_err)?,
        ))
    }

    /// The surface with the knot `u` inserted `times` times in the `u`
    /// direction (same shape, more poles).
    fn insert_u_knot(&self, u: f64, times: u32) -> PyResult<Self> {
        Ok(PyBSplineSurface(
            self.0.insert_u_knot(u, times).map_err(val_err)?,
        ))
    }

    /// The surface with the knot `v` inserted `times` times in the `v`
    /// direction (same shape, more poles).
    fn insert_v_knot(&self, v: f64, times: u32) -> PyResult<Self> {
        Ok(PyBSplineSurface(
            self.0.insert_v_knot(v, times).map_err(val_err)?,
        ))
    }

    /// The surface with each of `knots` inserted once in the `u`
    /// direction (same shape, more poles).
    fn refine_u_knots(&self, knots: Vec<f64>) -> PyResult<Self> {
        Ok(PyBSplineSurface(
            self.0.refine_u_knots(&knots).map_err(val_err)?,
        ))
    }

    /// The surface with each of `knots` inserted once in the `v`
    /// direction (same shape, more poles).
    fn refine_v_knots(&self, knots: Vec<f64>) -> PyResult<Self> {
        Ok(PyBSplineSurface(
            self.0.refine_v_knots(&knots).map_err(val_err)?,
        ))
    }

    /// The surface of `u` degree raised by `amount` (same shape).
    fn elevate_u_degree(&self, amount: usize) -> PyResult<Self> {
        Ok(PyBSplineSurface(
            self.0.elevate_u_degree(amount).map_err(val_err)?,
        ))
    }

    /// The surface of `v` degree raised by `amount` (same shape).
    fn elevate_v_degree(&self, amount: usize) -> PyResult<Self> {
        Ok(PyBSplineSurface(
            self.0.elevate_v_degree(amount).map_err(val_err)?,
        ))
    }

    /// The surface with the knot `u` removed up to `times` times in the
    /// `u` direction, and how many copies were removed: a removal is kept
    /// only while every point stays within `max_deviation` of this one.
    fn remove_u_knot(&self, u: f64, times: u32, max_deviation: f64) -> PyResult<(Self, u32)> {
        let (surface, removed) = self
            .0
            .remove_u_knot(u, times, max_deviation)
            .map_err(val_err)?;
        Ok((PyBSplineSurface(surface), removed))
    }

    /// The surface with the knot `v` removed up to `times` times in the
    /// `v` direction, and how many copies were removed: a removal is kept
    /// only while every point stays within `max_deviation` of this one.
    fn remove_v_knot(&self, v: f64, times: u32, max_deviation: f64) -> PyResult<(Self, u32)> {
        let (surface, removed) = self
            .0
            .remove_v_knot(v, times, max_deviation)
            .map_err(val_err)?;
        Ok((PyBSplineSurface(surface), removed))
    }

    /// The surface with each of `knots` removed once in the `u`
    /// direction, and how many copies were removed in all.
    fn remove_u_knots(&self, knots: Vec<f64>, max_deviation: f64) -> PyResult<(Self, usize)> {
        let (surface, removed) = self
            .0
            .remove_u_knots(&knots, max_deviation)
            .map_err(val_err)?;
        Ok((PyBSplineSurface(surface), removed))
    }

    /// The surface with each of `knots` removed once in the `v`
    /// direction, and how many copies were removed in all.
    fn remove_v_knots(&self, knots: Vec<f64>, max_deviation: f64) -> PyResult<(Self, usize)> {
        let (surface, removed) = self
            .0
            .remove_v_knots(&knots, max_deviation)
            .map_err(val_err)?;
        Ok((PyBSplineSurface(surface), removed))
    }

    /// The polynomial patch of each pair of knot spans as a Bézier
    /// B-spline surface: `patches[i][j]` covers the `i`-th `u` span and
    /// the `j`-th `v` span.
    fn to_bezier_patches(&self) -> Vec<Vec<Self>> {
        self.0
            .to_bezier_patches()
            .into_iter()
            .map(|row| row.into_iter().map(PyBSplineSurface).collect())
            .collect()
    }

    /// The piece over `u_range × v_range` as a clamped, non-periodic
    /// B-spline (a range may cross the seam of a periodic direction, up
    /// to one period).
    fn extract(&self, u_range: (f64, f64), v_range: (f64, f64)) -> PyResult<Self> {
        Ok(PyBSplineSurface(
            self.0.extract(u_range, v_range).map_err(val_err)?,
        ))
    }

    /// The piece over `[first, last]` in `u`, the `v` direction
    /// unchanged.
    fn extract_u(&self, first: f64, last: f64) -> PyResult<Self> {
        Ok(PyBSplineSurface(
            self.0.extract_u(first, last).map_err(val_err)?,
        ))
    }

    /// The piece over `[first, last]` in `v`, the `u` direction
    /// unchanged.
    fn extract_v(&self, first: f64, last: f64) -> PyResult<Self> {
        Ok(PyBSplineSurface(
            self.0.extract_v(first, last).map_err(val_err)?,
        ))
    }

    /// The two halves split at `u`, each a clamped, non-periodic B-spline.
    fn split_u_at(&self, u: f64) -> PyResult<(Self, Self)> {
        let (a, b) = self.0.split_u_at(u).map_err(val_err)?;
        Ok((PyBSplineSurface(a), PyBSplineSurface(b)))
    }

    /// The two halves split at `v`, each a clamped, non-periodic B-spline.
    fn split_v_at(&self, v: f64) -> PyResult<(Self, Self)> {
        let (a, b) = self.0.split_v_at(v).map_err(val_err)?;
        Ok((PyBSplineSurface(a), PyBSplineSurface(b)))
    }

    /// The surface with the `u` parameter mirrored, keeping a periodic
    /// `u` direction periodic.
    fn reversed_u(&self) -> Self {
        PyBSplineSurface(self.0.reversed_u())
    }

    /// The surface with the `v` parameter mirrored, keeping a periodic
    /// `v` direction periodic.
    fn reversed_v(&self) -> Self {
        PyBSplineSurface(self.0.reversed_v())
    }

    /// Image of this surface under `transform` (maps every control pole).
    #[pyo3(signature = (transform,))]
    fn transformed(&self, transform: &PyTransform) -> Self {
        PyBSplineSurface(self.0.transformed(transform.0))
    }

    /// Returns whether `point` lies on the surface within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// The local minima of the distance from `point` to the surface,
    /// as `(u, v, distance)` tuples ordered by ascending distance.
    #[pyo3(signature = (point, tol = None))]
    fn extrema(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> Vec<(f64, f64, f64)> {
        let tol = tolerance(tol);
        self.0
            .extrema(point.0, tol)
            .iter()
            .map(|p| (p.u, p.v, p.distance))
            .collect()
    }

    /// Projects `point` onto the surface, returning `(u, v, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64, f64) {
        let proj = self.0.project_point(point.0, tolerance(tol));
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
        let tol = tolerance(tol);
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.u, p.v, p.distance))
            .collect()
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
    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> PyResult<PyVector3D> {
        check_surface_order(du, dv)?;
        Ok(PyVector3D(self.0.eval_derivative(u, v, du, dv)))
    }
}

pub(crate) fn extract_surface(obj: &Bound<'_, PyAny>) -> PyResult<Surface> {
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
