use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use geomcore::Bound3D;
use geomcore::surfaces::{
    BSplineSurface, Cone, Cylinder, ParametricSurface, Plane, Sphere, Surface, Torus,
};

use crate::curves::*;
use crate::errors::python_error;
use crate::results::*;
use crate::values::*;
use crate::{check_surface_order, parse_parametrization, tolerance};

// ---------------------------------------------------------------------------
// Surfaces
// ---------------------------------------------------------------------------

/// An infinite plane, parametrized by in-plane coordinates (u, v).
#[pyclass(name = "Plane", module = "geomcore.surfaces", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyPlane(pub(crate) Plane);

#[pymethods]
impl PyPlane {
    /// The point and the first partial derivatives `(du, dv)` at `(u, v)`.
    fn eval_d1(&self, u: f64, v: f64) -> (PyPoint3D, (PyVector3D, PyVector3D)) {
        let (point, [du, dv]) = ParametricSurface::eval_d1(&self.0, u, v);
        (PyPoint3D(point), (PyVector3D(du), PyVector3D(dv)))
    }

    /// The point, the first partial derivatives `(du, dv)` and the second
    /// partial derivatives `(du du, du dv, dv dv)` at `(u, v)`.
    fn eval_d2(
        &self,
        u: f64,
        v: f64,
    ) -> (
        PyPoint3D,
        (PyVector3D, PyVector3D),
        (PyVector3D, PyVector3D, PyVector3D),
    ) {
        let (point, [du, dv], [duu, duv, dvv]) = ParametricSurface::eval_d2(&self.0, u, v);
        (
            PyPoint3D(point),
            (PyVector3D(du), PyVector3D(dv)),
            (PyVector3D(duu), PyVector3D(duv), PyVector3D(dvv)),
        )
    }

    /// The period of the u parameter, or `None` when the surface is not
    /// periodic in u.
    fn u_period(&self) -> Option<f64> {
        ParametricSurface::u_period(&self.0)
    }

    /// The period of the v parameter, or `None` when the surface is not
    /// periodic in v.
    fn v_period(&self) -> Option<f64> {
        ParametricSurface::v_period(&self.0)
    }

    /// The placement frame of the plane.
    fn frame(&self) -> PyFrame3D {
        PyFrame3D(self.0.frame())
    }

    fn __repr__(&self) -> String {
        format!(
            "Plane(origin={}, normal={})",
            crate::values::coordinates(self.0.frame().origin()),
            crate::values::coordinates(self.0.normal())
        )
    }

    #[new]
    /// Build a plane from a point and a normal direction.
    fn py_new(point: PyPoint3D, normal: PyVector3D) -> PyResult<Self> {
        Ok(PyPlane(
            Plane::new(point.0, normal.0).map_err(python_error)?,
        ))
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
            Plane::from_three_points(p1.0, p2.0, p3.0).map_err(python_error)?,
        ))
    }

    /// Build a plane from the equation `a*x + b*y + c*z + d = 0`.
    #[staticmethod]
    fn from_coefficients(a: f64, b: f64, c: f64, d: f64) -> PyResult<Self> {
        Ok(PyPlane(
            Plane::from_coefficients(a, b, c, d).map_err(python_error)?,
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
        Ok(PyPlane(
            self.0.transformed(transform.0).map_err(python_error)?,
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
        Ok(PyLine3D(self.0.u_iso(u).map_err(python_error)?))
    }

    /// The `v` isoparametric line through `(0, v)` along the plane's `u` axis, parametrized by `u` like the surface:
    /// `v_iso(v).eval_point(t)` is `eval_point(t, v)` (raises `ValueError` for
    /// a non-finite parameter).
    fn v_iso(&self, v: f64) -> PyResult<PyLine3D> {
        Ok(PyLine3D(self.0.v_iso(v).map_err(python_error)?))
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
            self.0
                .to_bspline_patch(u_range, v_range)
                .map_err(python_error)?,
        ))
    }

    /// Returns whether `point` lies on the surface within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// Projects `point` onto the surface: the closest point, as a
    /// `SurfaceProjection`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(
        slf: &Bound<'_, Self>,
        point: PyPoint3D,
        tol: Option<PyTolerance>,
    ) -> PySurfaceProjection {
        let projection = slf.borrow().0.project_point(point.0, tolerance(tol));
        PySurfaceProjection::new(projection, slf, point.0)
    }

    /// Projects each point in `points` onto the surface, returning a
    /// `SurfaceProjection` per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(
        slf: &Bound<'_, Self>,
        points: Vec<PyPoint3D>,
        tol: Option<PyTolerance>,
    ) -> Vec<PySurfaceProjection> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tolerance(tol);
        let projections = slf.borrow().0.project_points(&points, tol);
        projections
            .into_iter()
            .zip(points)
            .map(|(projection, point)| PySurfaceProjection::new(projection, slf, point))
            .collect()
    }

    /// Intersects with any curve or surface, in either order.
    ///
    /// Returns a `CurveCurveIntersection`, `CurveCurveIntersection2D`,
    /// `CurveSurfaceIntersection` or `SurfaceSurfaceIntersection`, by pair
    /// family, holding this object as `first` and `other` as `second`. A
    /// pair with a closed form is solved exactly (`method` is `EXACT`),
    /// any other pair by a numeric search (`NUMERIC`), whose `completeness`
    /// tells whether it proved that nothing was missed. Raises `TypeError`
    /// for a 2D curve against a 3D curve or a surface, and for an object
    /// that is not a geomcore curve or surface.
    #[pyo3(signature = (other, tol = None))]
    fn intersect(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<Py<PyAny>> {
        crate::intersect::intersect(slf.as_any(), other, tolerance(tol))
    }
}

/// An infinite circular cylinder; u is the angle around the axis, v the
/// height along it.
#[pyclass(name = "Cylinder", module = "geomcore.surfaces", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyCylinder(pub(crate) Cylinder);

#[pymethods]
impl PyCylinder {
    /// The point and the first partial derivatives `(du, dv)` at `(u, v)`.
    fn eval_d1(&self, u: f64, v: f64) -> (PyPoint3D, (PyVector3D, PyVector3D)) {
        let (point, [du, dv]) = ParametricSurface::eval_d1(&self.0, u, v);
        (PyPoint3D(point), (PyVector3D(du), PyVector3D(dv)))
    }

    /// The point, the first partial derivatives `(du, dv)` and the second
    /// partial derivatives `(du du, du dv, dv dv)` at `(u, v)`.
    fn eval_d2(
        &self,
        u: f64,
        v: f64,
    ) -> (
        PyPoint3D,
        (PyVector3D, PyVector3D),
        (PyVector3D, PyVector3D, PyVector3D),
    ) {
        let (point, [du, dv], [duu, duv, dvv]) = ParametricSurface::eval_d2(&self.0, u, v);
        (
            PyPoint3D(point),
            (PyVector3D(du), PyVector3D(dv)),
            (PyVector3D(duu), PyVector3D(duv), PyVector3D(dvv)),
        )
    }

    /// The period of the u parameter, or `None` when the surface is not
    /// periodic in u.
    fn u_period(&self) -> Option<f64> {
        ParametricSurface::u_period(&self.0)
    }

    /// The period of the v parameter, or `None` when the surface is not
    /// periodic in v.
    fn v_period(&self) -> Option<f64> {
        ParametricSurface::v_period(&self.0)
    }

    /// The axis of the cylinder: its origin and unit direction.
    fn axis(&self) -> PyAxis3D {
        PyAxis3D(self.0.axis())
    }

    /// The placement frame of the cylinder.
    fn frame(&self) -> PyFrame3D {
        PyFrame3D(self.0.frame())
    }

    fn __repr__(&self) -> String {
        format!(
            "Cylinder(origin={}, axis={}, radius={})",
            crate::values::coordinates(self.0.frame().origin()),
            crate::values::coordinates(self.0.frame().z_direction()),
            float_repr(self.0.radius())
        )
    }

    #[new]
    /// Build a cylinder from a point on its axis, the axis direction and a radius.
    fn py_new(center: PyPoint3D, axis_direction: PyVector3D, radius: f64) -> PyResult<Self> {
        Ok(PyCylinder(
            Cylinder::new(center.0, axis_direction.0, radius).map_err(python_error)?,
        ))
    }

    /// Build a cylinder from a placement frame (axis = frame z direction).
    #[staticmethod]
    fn from_frame(frame: PyFrame3D, radius: f64) -> PyResult<Self> {
        Ok(PyCylinder(
            Cylinder::from_frame(frame.0, radius).map_err(python_error)?,
        ))
    }

    /// Build a cylinder from an axis and a radius.
    #[staticmethod]
    fn from_axis(axis: PyAxis3D, radius: f64) -> PyResult<Self> {
        Ok(PyCylinder(
            Cylinder::from_axis(axis.0, radius).map_err(python_error)?,
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
            self.0.transformed(transform.0).map_err(python_error)?,
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
        Ok(PyLine3D(self.0.u_iso(u).map_err(python_error)?))
    }

    /// The `v` isoparametric section circle at height `v`, parametrized by `u` like the surface:
    /// `v_iso(v).eval_point(t)` is `eval_point(t, v)` (raises `ValueError` for
    /// a non-finite parameter).
    fn v_iso(&self, v: f64) -> PyResult<PyCircle3D> {
        Ok(PyCircle3D(self.0.v_iso(v).map_err(python_error)?))
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
            self.0
                .to_bspline_patch(u_range, v_range)
                .map_err(python_error)?,
        ))
    }

    /// Returns whether `point` lies on the surface within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// Projects `point` onto the surface: the closest point, as a
    /// `SurfaceProjection`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(
        slf: &Bound<'_, Self>,
        point: PyPoint3D,
        tol: Option<PyTolerance>,
    ) -> PySurfaceProjection {
        let projection = slf.borrow().0.project_point(point.0, tolerance(tol));
        PySurfaceProjection::new(projection, slf, point.0)
    }

    /// Projects each point in `points` onto the surface, returning a
    /// `SurfaceProjection` per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(
        slf: &Bound<'_, Self>,
        points: Vec<PyPoint3D>,
        tol: Option<PyTolerance>,
    ) -> Vec<PySurfaceProjection> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tolerance(tol);
        let projections = slf.borrow().0.project_points(&points, tol);
        projections
            .into_iter()
            .zip(points)
            .map(|(projection, point)| PySurfaceProjection::new(projection, slf, point))
            .collect()
    }

    /// Intersects with any curve or surface, in either order.
    ///
    /// Returns a `CurveCurveIntersection`, `CurveCurveIntersection2D`,
    /// `CurveSurfaceIntersection` or `SurfaceSurfaceIntersection`, by pair
    /// family, holding this object as `first` and `other` as `second`. A
    /// pair with a closed form is solved exactly (`method` is `EXACT`),
    /// any other pair by a numeric search (`NUMERIC`), whose `completeness`
    /// tells whether it proved that nothing was missed. Raises `TypeError`
    /// for a 2D curve against a 3D curve or a surface, and for an object
    /// that is not a geomcore curve or surface.
    #[pyo3(signature = (other, tol = None))]
    fn intersect(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<Py<PyAny>> {
        crate::intersect::intersect(slf.as_any(), other, tolerance(tol))
    }
}

/// An infinite cone; u is the angle around the axis, v the distance along a
/// generator from the reference circle.
#[pyclass(name = "Cone", module = "geomcore.surfaces", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyCone(pub(crate) Cone);

#[pymethods]
impl PyCone {
    /// The point and the first partial derivatives `(du, dv)` at `(u, v)`.
    fn eval_d1(&self, u: f64, v: f64) -> (PyPoint3D, (PyVector3D, PyVector3D)) {
        let (point, [du, dv]) = ParametricSurface::eval_d1(&self.0, u, v);
        (PyPoint3D(point), (PyVector3D(du), PyVector3D(dv)))
    }

    /// The point, the first partial derivatives `(du, dv)` and the second
    /// partial derivatives `(du du, du dv, dv dv)` at `(u, v)`.
    fn eval_d2(
        &self,
        u: f64,
        v: f64,
    ) -> (
        PyPoint3D,
        (PyVector3D, PyVector3D),
        (PyVector3D, PyVector3D, PyVector3D),
    ) {
        let (point, [du, dv], [duu, duv, dvv]) = ParametricSurface::eval_d2(&self.0, u, v);
        (
            PyPoint3D(point),
            (PyVector3D(du), PyVector3D(dv)),
            (PyVector3D(duu), PyVector3D(duv), PyVector3D(dvv)),
        )
    }

    /// The period of the u parameter, or `None` when the surface is not
    /// periodic in u.
    fn u_period(&self) -> Option<f64> {
        ParametricSurface::u_period(&self.0)
    }

    /// The period of the v parameter, or `None` when the surface is not
    /// periodic in v.
    fn v_period(&self) -> Option<f64> {
        ParametricSurface::v_period(&self.0)
    }

    /// The placement frame of the cone.
    fn frame(&self) -> PyFrame3D {
        PyFrame3D(self.0.frame())
    }

    fn __repr__(&self) -> String {
        format!(
            "Cone(origin={}, axis={}, reference_radius={}, semi_angle={})",
            crate::values::coordinates(self.0.frame().origin()),
            crate::values::coordinates(self.0.frame().z_direction()),
            float_repr(self.0.ref_radius()),
            float_repr(self.0.semi_angle())
        )
    }

    #[new]
    /// Build a cone from a reference-circle center, axis direction, half-angle and reference radius.
    fn py_new(
        center: PyPoint3D,
        axis_direction: PyVector3D,
        semi_angle: f64,
        ref_radius: f64,
    ) -> PyResult<Self> {
        Ok(PyCone(
            Cone::new(center.0, axis_direction.0, semi_angle, ref_radius).map_err(python_error)?,
        ))
    }

    /// Build a cone from a placement frame, half-angle and reference radius.
    #[staticmethod]
    fn from_frame(frame: PyFrame3D, semi_angle: f64, ref_radius: f64) -> PyResult<Self> {
        Ok(PyCone(
            Cone::from_frame(frame.0, semi_angle, ref_radius).map_err(python_error)?,
        ))
    }

    /// Build a cone through two circular sections given by axis points and radii.
    #[staticmethod]
    fn from_two_points_and_radii(p1: PyPoint3D, p2: PyPoint3D, r1: f64, r2: f64) -> PyResult<Self> {
        Ok(PyCone(
            Cone::from_two_points_and_radii(p1.0, p2.0, r1, r2).map_err(python_error)?,
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
        Ok(PyCone(
            self.0.transformed(transform.0).map_err(python_error)?,
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
        Ok(PyLine3D(self.0.u_iso(u).map_err(python_error)?))
    }

    /// The `v` isoparametric section circle at `v` (radius zero at the apex), parametrized by `u` like the surface:
    /// `v_iso(v).eval_point(t)` is `eval_point(t, v)` (raises `ValueError` for
    /// a non-finite parameter).
    fn v_iso(&self, v: f64) -> PyResult<PyCircle3D> {
        Ok(PyCircle3D(self.0.v_iso(v).map_err(python_error)?))
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
            self.0
                .to_bspline_patch(u_range, v_range)
                .map_err(python_error)?,
        ))
    }

    /// Returns whether `point` lies on the surface within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// Projects `point` onto the surface: the closest point, as a
    /// `SurfaceProjection`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(
        slf: &Bound<'_, Self>,
        point: PyPoint3D,
        tol: Option<PyTolerance>,
    ) -> PySurfaceProjection {
        let projection = slf.borrow().0.project_point(point.0, tolerance(tol));
        PySurfaceProjection::new(projection, slf, point.0)
    }

    /// Projects each point in `points` onto the surface, returning a
    /// `SurfaceProjection` per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(
        slf: &Bound<'_, Self>,
        points: Vec<PyPoint3D>,
        tol: Option<PyTolerance>,
    ) -> Vec<PySurfaceProjection> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tolerance(tol);
        let projections = slf.borrow().0.project_points(&points, tol);
        projections
            .into_iter()
            .zip(points)
            .map(|(projection, point)| PySurfaceProjection::new(projection, slf, point))
            .collect()
    }

    /// Intersects with any curve or surface, in either order.
    ///
    /// Returns a `CurveCurveIntersection`, `CurveCurveIntersection2D`,
    /// `CurveSurfaceIntersection` or `SurfaceSurfaceIntersection`, by pair
    /// family, holding this object as `first` and `other` as `second`. A
    /// pair with a closed form is solved exactly (`method` is `EXACT`),
    /// any other pair by a numeric search (`NUMERIC`), whose `completeness`
    /// tells whether it proved that nothing was missed. Raises `TypeError`
    /// for a 2D curve against a 3D curve or a surface, and for an object
    /// that is not a geomcore curve or surface.
    #[pyo3(signature = (other, tol = None))]
    fn intersect(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<Py<PyAny>> {
        crate::intersect::intersect(slf.as_any(), other, tolerance(tol))
    }
}

/// A sphere; u is the longitude in [0, 2*pi), v the latitude in [-pi/2, pi/2].
#[pyclass(name = "Sphere", module = "geomcore.surfaces", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PySphere(pub(crate) Sphere);

#[pymethods]
impl PySphere {
    /// The point and the first partial derivatives `(du, dv)` at `(u, v)`.
    fn eval_d1(&self, u: f64, v: f64) -> (PyPoint3D, (PyVector3D, PyVector3D)) {
        let (point, [du, dv]) = ParametricSurface::eval_d1(&self.0, u, v);
        (PyPoint3D(point), (PyVector3D(du), PyVector3D(dv)))
    }

    /// The point, the first partial derivatives `(du, dv)` and the second
    /// partial derivatives `(du du, du dv, dv dv)` at `(u, v)`.
    fn eval_d2(
        &self,
        u: f64,
        v: f64,
    ) -> (
        PyPoint3D,
        (PyVector3D, PyVector3D),
        (PyVector3D, PyVector3D, PyVector3D),
    ) {
        let (point, [du, dv], [duu, duv, dvv]) = ParametricSurface::eval_d2(&self.0, u, v);
        (
            PyPoint3D(point),
            (PyVector3D(du), PyVector3D(dv)),
            (PyVector3D(duu), PyVector3D(duv), PyVector3D(dvv)),
        )
    }

    /// The period of the u parameter, or `None` when the surface is not
    /// periodic in u.
    fn u_period(&self) -> Option<f64> {
        ParametricSurface::u_period(&self.0)
    }

    /// The period of the v parameter, or `None` when the surface is not
    /// periodic in v.
    fn v_period(&self) -> Option<f64> {
        ParametricSurface::v_period(&self.0)
    }

    /// The placement frame of the sphere.
    fn frame(&self) -> PyFrame3D {
        PyFrame3D(self.0.frame())
    }

    fn __repr__(&self) -> String {
        format!(
            "Sphere(center={}, radius={})",
            crate::values::coordinates(self.0.center()),
            float_repr(self.0.radius())
        )
    }

    #[new]
    /// Build a sphere from its center and radius (world-aligned frame).
    fn py_new(center: PyPoint3D, radius: f64) -> PyResult<Self> {
        Ok(PySphere(
            Sphere::new(center.0, radius).map_err(python_error)?,
        ))
    }

    /// Build a sphere from a placement frame and radius.
    #[staticmethod]
    fn from_frame(frame: PyFrame3D, radius: f64) -> PyResult<Self> {
        Ok(PySphere(
            Sphere::from_frame(frame.0, radius).map_err(python_error)?,
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
        Ok(PySphere(
            self.0.transformed(transform.0).map_err(python_error)?,
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
        Ok(PyCircle3D(self.0.u_iso(u).map_err(python_error)?))
    }

    /// The `v` isoparametric parallel circle at latitude `v` (radius zero at a pole), parametrized by `u` like the surface:
    /// `v_iso(v).eval_point(t)` is `eval_point(t, v)` (raises `ValueError` for
    /// a non-finite parameter).
    fn v_iso(&self, v: f64) -> PyResult<PyCircle3D> {
        Ok(PyCircle3D(self.0.v_iso(v).map_err(python_error)?))
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
            self.0
                .to_bspline_patch(u_range, v_range)
                .map_err(python_error)?,
        ))
    }

    /// Returns whether `point` lies on the surface within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// Projects `point` onto the surface: the closest point, as a
    /// `SurfaceProjection`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(
        slf: &Bound<'_, Self>,
        point: PyPoint3D,
        tol: Option<PyTolerance>,
    ) -> PySurfaceProjection {
        let projection = slf.borrow().0.project_point(point.0, tolerance(tol));
        PySurfaceProjection::new(projection, slf, point.0)
    }

    /// Projects each point in `points` onto the surface, returning a
    /// `SurfaceProjection` per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(
        slf: &Bound<'_, Self>,
        points: Vec<PyPoint3D>,
        tol: Option<PyTolerance>,
    ) -> Vec<PySurfaceProjection> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tolerance(tol);
        let projections = slf.borrow().0.project_points(&points, tol);
        projections
            .into_iter()
            .zip(points)
            .map(|(projection, point)| PySurfaceProjection::new(projection, slf, point))
            .collect()
    }

    /// Intersects with any curve or surface, in either order.
    ///
    /// Returns a `CurveCurveIntersection`, `CurveCurveIntersection2D`,
    /// `CurveSurfaceIntersection` or `SurfaceSurfaceIntersection`, by pair
    /// family, holding this object as `first` and `other` as `second`. A
    /// pair with a closed form is solved exactly (`method` is `EXACT`),
    /// any other pair by a numeric search (`NUMERIC`), whose `completeness`
    /// tells whether it proved that nothing was missed. Raises `TypeError`
    /// for a 2D curve against a 3D curve or a surface, and for an object
    /// that is not a geomcore curve or surface.
    #[pyo3(signature = (other, tol = None))]
    fn intersect(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<Py<PyAny>> {
        crate::intersect::intersect(slf.as_any(), other, tolerance(tol))
    }
}

/// A torus; u is the angle around the main axis, v the angle around the tube.
#[pyclass(name = "Torus", module = "geomcore.surfaces", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyTorus(pub(crate) Torus);

#[pymethods]
impl PyTorus {
    /// The point and the first partial derivatives `(du, dv)` at `(u, v)`.
    fn eval_d1(&self, u: f64, v: f64) -> (PyPoint3D, (PyVector3D, PyVector3D)) {
        let (point, [du, dv]) = ParametricSurface::eval_d1(&self.0, u, v);
        (PyPoint3D(point), (PyVector3D(du), PyVector3D(dv)))
    }

    /// The point, the first partial derivatives `(du, dv)` and the second
    /// partial derivatives `(du du, du dv, dv dv)` at `(u, v)`.
    fn eval_d2(
        &self,
        u: f64,
        v: f64,
    ) -> (
        PyPoint3D,
        (PyVector3D, PyVector3D),
        (PyVector3D, PyVector3D, PyVector3D),
    ) {
        let (point, [du, dv], [duu, duv, dvv]) = ParametricSurface::eval_d2(&self.0, u, v);
        (
            PyPoint3D(point),
            (PyVector3D(du), PyVector3D(dv)),
            (PyVector3D(duu), PyVector3D(duv), PyVector3D(dvv)),
        )
    }

    /// The period of the u parameter, or `None` when the surface is not
    /// periodic in u.
    fn u_period(&self) -> Option<f64> {
        ParametricSurface::u_period(&self.0)
    }

    /// The period of the v parameter, or `None` when the surface is not
    /// periodic in v.
    fn v_period(&self) -> Option<f64> {
        ParametricSurface::v_period(&self.0)
    }

    /// The center of the torus.
    fn center(&self) -> PyPoint3D {
        PyPoint3D(self.0.center())
    }

    /// The placement frame of the torus.
    fn frame(&self) -> PyFrame3D {
        PyFrame3D(self.0.frame())
    }

    fn __repr__(&self) -> String {
        format!(
            "Torus(center={}, axis={}, major_radius={}, minor_radius={})",
            crate::values::coordinates(self.0.center()),
            crate::values::coordinates(self.0.frame().z_direction()),
            float_repr(self.0.major_radius()),
            float_repr(self.0.minor_radius())
        )
    }

    #[new]
    /// Build a torus from its center, main-axis direction and the two radii.
    fn py_new(
        center: PyPoint3D,
        normal: PyVector3D,
        major_radius: f64,
        minor_radius: f64,
    ) -> PyResult<Self> {
        Ok(PyTorus(
            Torus::new(center.0, normal.0, major_radius, minor_radius).map_err(python_error)?,
        ))
    }

    /// Build a torus from a placement frame and the two radii.
    #[staticmethod]
    fn from_frame(frame: PyFrame3D, major_radius: f64, minor_radius: f64) -> PyResult<Self> {
        Ok(PyTorus(
            Torus::from_frame(frame.0, major_radius, minor_radius).map_err(python_error)?,
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
        Ok(PyTorus(
            self.0.transformed(transform.0).map_err(python_error)?,
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
        Ok(PyCircle3D(self.0.u_iso(u).map_err(python_error)?))
    }

    /// The `v` isoparametric parallel circle at tube angle `v` (radius zero at a pinch), parametrized by `u` like the surface:
    /// `v_iso(v).eval_point(t)` is `eval_point(t, v)` (raises `ValueError` for
    /// a non-finite parameter).
    fn v_iso(&self, v: f64) -> PyResult<PyCircle3D> {
        Ok(PyCircle3D(self.0.v_iso(v).map_err(python_error)?))
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
            self.0
                .to_bspline_patch(u_range, v_range)
                .map_err(python_error)?,
        ))
    }

    /// Returns whether `point` lies on the surface within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// Projects `point` onto the surface: the closest point, as a
    /// `SurfaceProjection`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(
        slf: &Bound<'_, Self>,
        point: PyPoint3D,
        tol: Option<PyTolerance>,
    ) -> PySurfaceProjection {
        let projection = slf.borrow().0.project_point(point.0, tolerance(tol));
        PySurfaceProjection::new(projection, slf, point.0)
    }

    /// Projects each point in `points` onto the surface, returning a
    /// `SurfaceProjection` per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(
        slf: &Bound<'_, Self>,
        points: Vec<PyPoint3D>,
        tol: Option<PyTolerance>,
    ) -> Vec<PySurfaceProjection> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tolerance(tol);
        let projections = slf.borrow().0.project_points(&points, tol);
        projections
            .into_iter()
            .zip(points)
            .map(|(projection, point)| PySurfaceProjection::new(projection, slf, point))
            .collect()
    }

    /// Intersects with any curve or surface, in either order.
    ///
    /// Returns a `CurveCurveIntersection`, `CurveCurveIntersection2D`,
    /// `CurveSurfaceIntersection` or `SurfaceSurfaceIntersection`, by pair
    /// family, holding this object as `first` and `other` as `second`. A
    /// pair with a closed form is solved exactly (`method` is `EXACT`),
    /// any other pair by a numeric search (`NUMERIC`), whose `completeness`
    /// tells whether it proved that nothing was missed. Raises `TypeError`
    /// for a 2D curve against a 3D curve or a surface, and for an object
    /// that is not a geomcore curve or surface.
    #[pyo3(signature = (other, tol = None))]
    fn intersect(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<Py<PyAny>> {
        crate::intersect::intersect(slf.as_any(), other, tolerance(tol))
    }
}

/// A tensor-product B-spline (optionally rational, optionally periodic) surface.
#[pyclass(name = "BSplineSurface", module = "geomcore.surfaces", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyBSplineSurface(pub(crate) BSplineSurface);

#[pymethods]
impl PyBSplineSurface {
    /// The point and the first partial derivatives `(du, dv)` at `(u, v)`.
    fn eval_d1(&self, u: f64, v: f64) -> (PyPoint3D, (PyVector3D, PyVector3D)) {
        let (point, [du, dv]) = ParametricSurface::eval_d1(&self.0, u, v);
        (PyPoint3D(point), (PyVector3D(du), PyVector3D(dv)))
    }

    /// The point, the first partial derivatives `(du, dv)` and the second
    /// partial derivatives `(du du, du dv, dv dv)` at `(u, v)`.
    fn eval_d2(
        &self,
        u: f64,
        v: f64,
    ) -> (
        PyPoint3D,
        (PyVector3D, PyVector3D),
        (PyVector3D, PyVector3D, PyVector3D),
    ) {
        let (point, [du, dv], [duu, duv, dvv]) = ParametricSurface::eval_d2(&self.0, u, v);
        (
            PyPoint3D(point),
            (PyVector3D(du), PyVector3D(dv)),
            (PyVector3D(duu), PyVector3D(duv), PyVector3D(dvv)),
        )
    }

    /// The period of the u parameter, or `None` when the surface is not
    /// periodic in u.
    fn u_period(&self) -> Option<f64> {
        ParametricSurface::u_period(&self.0)
    }

    /// The period of the v parameter, or `None` when the surface is not
    /// periodic in v.
    fn v_period(&self) -> Option<f64> {
        ParametricSurface::v_period(&self.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "BSplineSurface(u_degree={}, v_degree={}, pole_count=({}, {}), rational={})",
            self.0.u_degree(),
            self.0.v_degree(),
            self.0.u_pole_count(),
            self.0.v_pole_count(),
            crate::values::py_bool(self.0.is_rational())
        )
    }

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
            .map_err(python_error)?,
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
            .map_err(python_error)?,
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
        Ok(PyBSplineCurve3D(self.0.u_iso(u).map_err(python_error)?))
    }

    /// The `v` isoparametric B-spline curve at `v`, blended from the pole net, parametrized by `u` like the surface:
    /// `v_iso(v).eval_point(t)` is `eval_point(t, v)` (raises `ValueError` for
    /// a non-finite parameter or one outside a non-periodic direction (a periodic one wraps)).
    fn v_iso(&self, v: f64) -> PyResult<PyBSplineCurve3D> {
        Ok(PyBSplineCurve3D(self.0.v_iso(v).map_err(python_error)?))
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
            .map_err(python_error)?,
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
                .map_err(python_error)?,
        ))
    }

    /// The surface with the knot `u` inserted `times` times in the `u`
    /// direction (same shape, more poles).
    fn insert_u_knot(&self, u: f64, times: u32) -> PyResult<Self> {
        Ok(PyBSplineSurface(
            self.0.insert_u_knot(u, times).map_err(python_error)?,
        ))
    }

    /// The surface with the knot `v` inserted `times` times in the `v`
    /// direction (same shape, more poles).
    fn insert_v_knot(&self, v: f64, times: u32) -> PyResult<Self> {
        Ok(PyBSplineSurface(
            self.0.insert_v_knot(v, times).map_err(python_error)?,
        ))
    }

    /// The surface with each of `knots` inserted once in the `u`
    /// direction (same shape, more poles).
    fn refine_u_knots(&self, knots: Vec<f64>) -> PyResult<Self> {
        Ok(PyBSplineSurface(
            self.0.refine_u_knots(&knots).map_err(python_error)?,
        ))
    }

    /// The surface with each of `knots` inserted once in the `v`
    /// direction (same shape, more poles).
    fn refine_v_knots(&self, knots: Vec<f64>) -> PyResult<Self> {
        Ok(PyBSplineSurface(
            self.0.refine_v_knots(&knots).map_err(python_error)?,
        ))
    }

    /// The surface of `u` degree raised by `amount` (same shape).
    fn elevate_u_degree(&self, amount: usize) -> PyResult<Self> {
        Ok(PyBSplineSurface(
            self.0.elevate_u_degree(amount).map_err(python_error)?,
        ))
    }

    /// The surface of `v` degree raised by `amount` (same shape).
    fn elevate_v_degree(&self, amount: usize) -> PyResult<Self> {
        Ok(PyBSplineSurface(
            self.0.elevate_v_degree(amount).map_err(python_error)?,
        ))
    }

    /// The surface with the knot `u` removed up to `times` times in the
    /// `u` direction, and how many copies were removed: a removal is kept
    /// only while every point stays within `max_deviation` of this one.
    fn remove_u_knot(&self, u: f64, times: u32, max_deviation: f64) -> PyResult<(Self, u32)> {
        let (surface, removed) = self
            .0
            .remove_u_knot(u, times, max_deviation)
            .map_err(python_error)?;
        Ok((PyBSplineSurface(surface), removed))
    }

    /// The surface with the knot `v` removed up to `times` times in the
    /// `v` direction, and how many copies were removed: a removal is kept
    /// only while every point stays within `max_deviation` of this one.
    fn remove_v_knot(&self, v: f64, times: u32, max_deviation: f64) -> PyResult<(Self, u32)> {
        let (surface, removed) = self
            .0
            .remove_v_knot(v, times, max_deviation)
            .map_err(python_error)?;
        Ok((PyBSplineSurface(surface), removed))
    }

    /// The surface with each of `knots` removed once in the `u`
    /// direction, and how many copies were removed in all.
    fn remove_u_knots(&self, knots: Vec<f64>, max_deviation: f64) -> PyResult<(Self, usize)> {
        let (surface, removed) = self
            .0
            .remove_u_knots(&knots, max_deviation)
            .map_err(python_error)?;
        Ok((PyBSplineSurface(surface), removed))
    }

    /// The surface with each of `knots` removed once in the `v`
    /// direction, and how many copies were removed in all.
    fn remove_v_knots(&self, knots: Vec<f64>, max_deviation: f64) -> PyResult<(Self, usize)> {
        let (surface, removed) = self
            .0
            .remove_v_knots(&knots, max_deviation)
            .map_err(python_error)?;
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
            self.0.extract(u_range, v_range).map_err(python_error)?,
        ))
    }

    /// The piece over `[first, last]` in `u`, the `v` direction
    /// unchanged.
    fn extract_u(&self, first: f64, last: f64) -> PyResult<Self> {
        Ok(PyBSplineSurface(
            self.0.extract_u(first, last).map_err(python_error)?,
        ))
    }

    /// The piece over `[first, last]` in `v`, the `u` direction
    /// unchanged.
    fn extract_v(&self, first: f64, last: f64) -> PyResult<Self> {
        Ok(PyBSplineSurface(
            self.0.extract_v(first, last).map_err(python_error)?,
        ))
    }

    /// The two halves split at `u`, each a clamped, non-periodic B-spline.
    fn split_u_at(&self, u: f64) -> PyResult<(Self, Self)> {
        let (a, b) = self.0.split_u_at(u).map_err(python_error)?;
        Ok((PyBSplineSurface(a), PyBSplineSurface(b)))
    }

    /// The two halves split at `v`, each a clamped, non-periodic B-spline.
    fn split_v_at(&self, v: f64) -> PyResult<(Self, Self)> {
        let (a, b) = self.0.split_v_at(v).map_err(python_error)?;
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

    /// The local minima of the distance from `point` to the surface, as
    /// `SurfaceProjection` objects ordered by ascending distance.
    #[pyo3(signature = (point, tol = None))]
    fn extrema(
        slf: &Bound<'_, Self>,
        point: PyPoint3D,
        tol: Option<PyTolerance>,
    ) -> Vec<PySurfaceProjection> {
        let tol = tolerance(tol);
        let projections = slf.borrow().0.extrema(point.0, tol);
        projections
            .into_iter()
            .map(|projection| PySurfaceProjection::new(projection, slf, point.0))
            .collect()
    }

    /// Projects `point` onto the surface: the closest point, as a
    /// `SurfaceProjection`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(
        slf: &Bound<'_, Self>,
        point: PyPoint3D,
        tol: Option<PyTolerance>,
    ) -> PySurfaceProjection {
        let projection = slf.borrow().0.project_point(point.0, tolerance(tol));
        PySurfaceProjection::new(projection, slf, point.0)
    }

    /// Projects each point in `points` onto the surface, returning a
    /// `SurfaceProjection` per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(
        slf: &Bound<'_, Self>,
        points: Vec<PyPoint3D>,
        tol: Option<PyTolerance>,
    ) -> Vec<PySurfaceProjection> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tolerance(tol);
        let projections = slf.borrow().0.project_points(&points, tol);
        projections
            .into_iter()
            .zip(points)
            .map(|(projection, point)| PySurfaceProjection::new(projection, slf, point))
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

    /// Intersects with any curve or surface, in either order.
    ///
    /// Returns a `CurveCurveIntersection`, `CurveCurveIntersection2D`,
    /// `CurveSurfaceIntersection` or `SurfaceSurfaceIntersection`, by pair
    /// family, holding this object as `first` and `other` as `second`. A
    /// pair with a closed form is solved exactly (`method` is `EXACT`),
    /// any other pair by a numeric search (`NUMERIC`), whose `completeness`
    /// tells whether it proved that nothing was missed. Raises `TypeError`
    /// for a 2D curve against a 3D curve or a surface, and for an object
    /// that is not a geomcore curve or surface.
    #[pyo3(signature = (other, tol = None))]
    fn intersect(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<Py<PyAny>> {
        crate::intersect::intersect(slf.as_any(), other, tolerance(tol))
    }
}

pub(crate) fn extract_surface(obj: &Bound<'_, PyAny>) -> PyResult<Surface> {
    surface_of(obj).ok_or_else(|| {
        PyTypeError::new_err(
            "expected a geomcore surface (Plane, Cylinder, Cone, Sphere, Torus or BSplineSurface)",
        )
    })
}

/// The surface `obj` holds, if it is one of the surface classes. Each
/// class is checked with a type test, so a miss raises nothing.
pub(crate) fn surface_of(obj: &Bound<'_, PyAny>) -> Option<Surface> {
    if let Ok(s) = obj.downcast::<PyPlane>() {
        return Some(Surface::from(&s.borrow().0));
    }
    if let Ok(s) = obj.downcast::<PyCylinder>() {
        return Some(Surface::from(&s.borrow().0));
    }
    if let Ok(s) = obj.downcast::<PyCone>() {
        return Some(Surface::from(&s.borrow().0));
    }
    if let Ok(s) = obj.downcast::<PySphere>() {
        return Some(Surface::from(&s.borrow().0));
    }
    if let Ok(s) = obj.downcast::<PyTorus>() {
        return Some(Surface::from(&s.borrow().0));
    }
    if let Ok(s) = obj.downcast::<PyBSplineSurface>() {
        return Some(Surface::from(&s.borrow().0));
    }
    None
}
