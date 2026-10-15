use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use geomcore::Bound3D;
use geomcore::IntersectExact;
use geomcore::curves::{
    BSplineCurve3D, Circle2D, Circle3D, Curve3D, Ellipse3D, Hyperbola3D, Line2D, Line3D,
    Parabola3D, ParametricCurve3D, PointExtrema2D, PointExtrema3D, TrimmedCurve2D, TrimmedCurve3D,
};
use geomcore::{
    CircleCircle2DIntersection, CircleCircle3DIntersection, LineCircle2DIntersection,
    LineCircle3DIntersection, LineLine2DIntersection, LineLine3DIntersection,
    LinePlaneIntersection, Point3D, Tolerance,
};

use crate::numeric::*;
use crate::surfaces::*;
use crate::values::*;
use crate::{check_curve_order, parse_parametrization, tolerance, val_err, warn_deprecated};

// ---------------------------------------------------------------------------
// Curves
// ---------------------------------------------------------------------------

/// An infinite straight line in 3D, parametrized by arc length from its origin.
#[pyclass(name = "Line3D", module = "geomcore.curves", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyLine3D(pub(crate) Line3D);

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

    /// Image of this line under `transform` (raises ValueError for a
    /// collapsing transform).
    #[pyo3(signature = (transform,))]
    fn transformed(&self, transform: &PyTransform) -> PyResult<Self> {
        Ok(PyLine3D(self.0.transformed(transform.0).map_err(val_err)?))
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
    fn eval_derivative(&self, u: f64, order: u32) -> PyResult<PyVector3D> {
        check_curve_order(order, u32::MAX)?;
        Ok(PyVector3D(self.0.eval_derivative(u, order)))
    }

    /// Recover the parameter of a point lying on the line.
    fn parameter_of(&self, point: PyPoint3D) -> f64 {
        self.0.parameter_of(point.0)
    }

    /// The (first, last) parameter bounds (unbounded here).
    fn bounds(&self) -> (f64, f64) {
        ParametricCurve3D::bounds(&self.0)
    }

    /// The axis-aligned box around the shape (`None` when unbounded).
    fn bounding_box(&self) -> Option<PyBoundingBox3D> {
        self.0.bounding_box().map(PyBoundingBox3D)
    }

    /// Returns whether `point` lies on the curve within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// Projects `point` onto the curve, returning `(parameter, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64) {
        let proj = self.0.project_point(point.0, tolerance(tol));
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the curve, returning a
    /// `(parameter, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(&self, points: Vec<PyPoint3D>, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tolerance(tol);
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Intersects this line with another shape, using the exact solver
    /// for the type of `other`. Returns a `(kind, payload)` tuple:
    ///
    /// - `Plane`: Returns `("point", (t, Point3D))`, `("parallel", None)` or
    ///   `("coincident", None)`.
    /// - `Cylinder`: Returns `("two_points", ((t1, Point3D), (t2, Point3D)))`,
    ///   `("one_point", (t, Point3D))`, `("tangent", (t, Point3D))`,
    ///   `("empty", None)` or `("coincident", None)` when the line is a
    ///   generator.
    /// - `Cone`: Returns `("two_points", ((t1, Point3D), (t2, Point3D)))`,
    ///   `("one_point", (t, Point3D))`, `("tangent", (t, Point3D))`,
    ///   `("empty", None)` or `("coincident", None)` when the line is a
    ///   generator. Roots behind the apex are filtered, so a single surviving
    ///   hit reports `("one_point", (t, Point3D))`.
    /// - `Sphere`: Returns `("two_points", ((t1, Point3D), (t2, Point3D)))`,
    ///   `("one_point", (t, Point3D))`, `("tangent", (t, Point3D))`,
    ///   `("empty", None)` or `("coincident", None)` (the latter never triggers
    ///   for spheres, which contain no line).
    /// - `Torus`: Returns `("hits", [(t, Point3D, (u, v), multiplicity)])` (up
    ///   to four hits ordered by line parameter) or `("empty", None)`.
    /// - `Line3D`: Returns `("point", (s, Point3D, t))`,
    ///   `("parallel", distance)`, `("coincident", None)` or
    ///   `("skew", (s, t, distance))`.
    /// - `Circle3D`: Returns `("points", ((t1, Point3D), (t2, Point3D)))`,
    ///   `("tangent", (t, Point3D))` or `("empty", None)`.
    /// - `Ellipse3D`, `Parabola3D`, `Hyperbola3D`: Returns
    ///   `("points", ((t1, Point3D, u1), (t2, Point3D, u2)))`,
    ///   `("tangent", (t, Point3D, u))` or `("empty", None)`.
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
        dispatch_intersect!(other, "Line3D", "Plane, Cylinder, Cone, Sphere, Torus, Line3D, Circle3D, Ellipse3D, Parabola3D or Hyperbola3D", {
            PyPlane => |o| self.plane_result(py, &o, tol),
            PyCylinder => |o| self.cylinder_result(py, &o, tol),
            PyCone => |o| self.cone_result(py, &o, tol),
            PySphere => |o| self.sphere_result(py, &o, tol),
            PyTorus => |o| self.torus_result(py, &o, tol),
            PyLine3D => |o| self.line_result(py, &o, tol),
            PyCircle3D => |o| self.circle_result(py, &o, tol),
            PyEllipse3D => |o| self.ellipse_result(py, &o, tol),
            PyParabola3D => |o| self.parabola_result(py, &o, tol),
            PyHyperbola3D => |o| self.hyperbola_result(py, &o, tol),
        })
    }

    /// Intersects this line with a plane.
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
        warn_deprecated(py, "Line3D.intersect_plane")?;
        self.plane_result(py, plane, tolerance(tol))
    }

    /// Intersects this line with a sphere.
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
        warn_deprecated(py, "Line3D.intersect_sphere")?;
        self.sphere_result(py, sphere, tolerance(tol))
    }

    /// Intersects this line with a cylinder.
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
        warn_deprecated(py, "Line3D.intersect_cylinder")?;
        self.cylinder_result(py, cylinder, tolerance(tol))
    }

    /// Intersects this line with a cone.
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
        warn_deprecated(py, "Line3D.intersect_cone")?;
        self.cone_result(py, cone, tolerance(tol))
    }

    /// Intersects this line with another 3D line.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (other, tol = None))]
    fn intersect_line(
        &self,
        py: Python<'_>,
        other: &PyLine3D,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Line3D.intersect_line")?;
        self.line_result(py, other, tolerance(tol))
    }

    /// Intersects this line with a 3D circle.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (circle, tol = None))]
    fn intersect_circle(
        &self,
        py: Python<'_>,
        circle: &PyCircle3D,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Line3D.intersect_circle")?;
        self.circle_result(py, circle, tolerance(tol))
    }

    /// Intersects this line with an ellipse.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (ellipse, tol = None))]
    fn intersect_ellipse(
        &self,
        py: Python<'_>,
        ellipse: &PyEllipse3D,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Line3D.intersect_ellipse")?;
        self.ellipse_result(py, ellipse, tolerance(tol))
    }

    /// Intersects this line with a parabola.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (parabola, tol = None))]
    fn intersect_parabola(
        &self,
        py: Python<'_>,
        parabola: &PyParabola3D,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Line3D.intersect_parabola")?;
        self.parabola_result(py, parabola, tolerance(tol))
    }

    /// Intersects this line with a hyperbola.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (hyperbola, tol = None))]
    fn intersect_hyperbola(
        &self,
        py: Python<'_>,
        hyperbola: &PyHyperbola3D,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Line3D.intersect_hyperbola")?;
        self.hyperbola_result(py, hyperbola, tolerance(tol))
    }

    /// Intersects this line with a torus.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (torus, tol = None))]
    fn intersect_torus(
        &self,
        py: Python<'_>,
        torus: &PyTorus,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Line3D.intersect_torus")?;
        self.torus_result(py, torus, tolerance(tol))
    }

    /// Compute this line's 2D representation in a surface's (u, v) space.
    ///
    /// Raises `ValueError` if no closed-form representation exists for the
    /// pair, or if the line does not lie on the surface within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (surface, tol = None))]
    fn parametrize_on(
        &self,
        py: Python<'_>,
        surface: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<Py<PyAny>> {
        let s = extract_surface(surface)?;
        let tol = tolerance(tol);
        curve2d_to_py(py, self.0.parametrize_on(s, tol).map_err(val_err)?)
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyLine3D {
    pub(crate) fn plane_result(
        &self,
        py: Python<'_>,
        plane: &PyPlane,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&plane.0, tol) {
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

    pub(crate) fn sphere_result(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        quadric_hit_to_py(py, self.0.intersect_exact(&sphere.0, tol))
    }

    pub(crate) fn cylinder_result(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        quadric_hit_to_py(py, self.0.intersect_exact(&cylinder.0, tol))
    }

    pub(crate) fn cone_result(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        quadric_hit_to_py(py, self.0.intersect_exact(&cone.0, tol))
    }

    pub(crate) fn line_result(
        &self,
        py: Python<'_>,
        other: &PyLine3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&other.0, tol) {
            LineLine3DIntersection::Point(s, p, t) => Ok((
                "point".to_string(),
                (s, PyPoint3D(p).into_pyobject(py)?.into_any().unbind(), t)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            )),
            LineLine3DIntersection::Parallel(d) => Ok((
                "parallel".to_string(),
                d.into_pyobject(py)?.into_any().unbind(),
            )),
            LineLine3DIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
            LineLine3DIntersection::Skew { s, t, distance } => Ok((
                "skew".to_string(),
                (s, t, distance).into_pyobject(py)?.into_any().unbind(),
            )),
        }
    }

    pub(crate) fn circle_result(
        &self,
        py: Python<'_>,
        circle: &PyCircle3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        fn hit(py: Python<'_>, t: f64, p: Point3D) -> PyResult<Py<PyAny>> {
            (t, PyPoint3D(p).into_pyobject(py)?.into_any().unbind())
                .into_pyobject(py)
                .map(|o| o.into_any().unbind())
        }
        match self.0.intersect_exact(&circle.0, tol) {
            LineCircle3DIntersection::Points((t1, p1), (t2, p2)) => Ok((
                "points".to_string(),
                (hit(py, t1, p1)?, hit(py, t2, p2)?)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            )),
            LineCircle3DIntersection::Tangent(t, p) => Ok(("tangent".to_string(), hit(py, t, p)?)),
            LineCircle3DIntersection::Empty => Ok(("empty".to_string(), py.None())),
        }
    }

    pub(crate) fn ellipse_result(
        &self,
        py: Python<'_>,
        ellipse: &PyEllipse3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        line_conic_to_py(py, self.0.intersect_exact(&ellipse.0, tol))
    }

    pub(crate) fn parabola_result(
        &self,
        py: Python<'_>,
        parabola: &PyParabola3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        line_conic_to_py(py, self.0.intersect_exact(&parabola.0, tol))
    }

    pub(crate) fn hyperbola_result(
        &self,
        py: Python<'_>,
        hyperbola: &PyHyperbola3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        line_conic_to_py(py, self.0.intersect_exact(&hyperbola.0, tol))
    }

    pub(crate) fn torus_result(
        &self,
        py: Python<'_>,
        torus: &PyTorus,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        line_torus_to_py(py, self.0.intersect_exact(&torus.0, tol))
    }
}

/// A circle in 3D, parametrized by the angle from its frame's x direction.
#[pyclass(name = "Circle3D", module = "geomcore.curves", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyCircle3D(pub(crate) Circle3D);

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

    /// The circumference (`2*PI*radius`).
    fn circumference(&self) -> f64 {
        self.0.circumference()
    }

    /// The enclosed disk area (`PI*radius^2`).
    fn disk_area(&self) -> f64 {
        self.0.disk_area()
    }

    /// The (first, last) parameter bounds (`[0, 2π)`).
    fn bounds(&self) -> (f64, f64) {
        ParametricCurve3D::bounds(&self.0)
    }

    /// The axis-aligned box around the shape (`None` when unbounded).
    fn bounding_box(&self) -> Option<PyBoundingBox3D> {
        self.0.bounding_box().map(PyBoundingBox3D)
    }

    /// The circle as an exact periodic rational quadratic B-spline over
    /// `[0, 2π]`.
    fn to_bspline(&self) -> PyBSplineCurve3D {
        PyBSplineCurve3D(self.0.to_bspline())
    }

    /// The unit normal of the circle's plane.
    fn normal(&self) -> PyVector3D {
        PyVector3D(self.0.normal())
    }

    /// Image of this circle under `transform` (raises ValueError for a
    /// collapsing transform).
    #[pyo3(signature = (transform,))]
    fn transformed(&self, transform: &PyTransform) -> PyResult<Self> {
        Ok(PyCircle3D(
            self.0.transformed(transform.0).map_err(val_err)?,
        ))
    }

    /// Evaluate the point at angle `u` (radians).
    fn eval_point(&self, u: f64) -> PyPoint3D {
        PyPoint3D(self.0.eval_point(u))
    }

    /// Evaluate many parameters at once.
    fn eval_points(&self, us: Vec<f64>) -> Vec<PyPoint3D> {
        self.0.eval_points(&us).into_iter().map(PyPoint3D).collect()
    }

    /// Discretize the circle into points joined by straight segments,
    /// each segment's chord deviation at most `max_chord_deviation` and,
    /// if given, its angle deviation at most `max_angle_deviation`
    /// (radians).
    #[pyo3(signature = (max_chord_deviation, max_angle_deviation = None))]
    fn discretize(
        &self,
        max_chord_deviation: f64,
        max_angle_deviation: Option<f64>,
    ) -> PyResult<PyDiscretization3D> {
        discretize_3d(&self.0, max_chord_deviation, max_angle_deviation)
    }

    /// Evaluate the derivative of the given order (>= 1) at `u`.
    fn eval_derivative(&self, u: f64, order: u32) -> PyResult<PyVector3D> {
        check_curve_order(order, u32::MAX)?;
        Ok(PyVector3D(self.0.eval_derivative(u, order)))
    }

    /// Recover the parameter in [0, 2*pi) of a point lying on the circle.
    fn parameter_of(&self, point: PyPoint3D) -> f64 {
        self.0.parameter_of(point.0)
    }

    /// Returns whether `point` lies on the curve within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// Projects `point` onto the curve, returning `(parameter, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64) {
        let proj = self.0.project_point(point.0, tolerance(tol));
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the curve, returning a
    /// `(parameter, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(&self, points: Vec<PyPoint3D>, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tolerance(tol);
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Compute this circle's 2D representation in a surface's (u, v) space.
    ///
    /// Raises `ValueError` if no closed-form representation exists for the
    /// pair, or if the circle does not lie on the surface within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (surface, tol = None))]
    fn parametrize_on(
        &self,
        py: Python<'_>,
        surface: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<Py<PyAny>> {
        let s = extract_surface(surface)?;
        let tol = tolerance(tol);
        curve2d_to_py(py, self.0.parametrize_on(s, tol).map_err(val_err)?)
    }

    /// Intersects this circle with another shape, using the exact solver
    /// for the type of `other`. Returns a `(kind, payload)` tuple:
    ///
    /// - `Plane`, `Cylinder`, `Cone`, `Sphere`: Returns `("circle", Circle3D)`,
    ///   `("hits", [(t, Point3D, multiplicity)])` or `("empty", None)`.
    /// - `Line3D`: Returns `("points", ((t1, Point3D), (t2, Point3D)))`,
    ///   `("tangent", (t, Point3D))` or `("empty", None)`.
    /// - `Circle3D`: Only coplanar pairs admit a closed form. Returns
    ///   `("points", (Point3D, Point3D))`, `("tangent", Point3D)`,
    ///   `("empty", None)`, `("coincident", None)` or `("not_analytic", None)`.
    /// - `Ellipse3D`, `Parabola3D`, `Hyperbola3D`: Returns
    ///   `("hits", [(s, Point3D, t, multiplicity)])`, `("coincident", None)`
    ///   or `("empty", None)`, with `s` on this circle and `t` on `other`
    ///   (multiplicity ≥ 2 marks tangency). A hyperbola is one branch.
    ///   Most pairs compute only one of `s` and `t`: the other is `None`;
    ///   recover it with that conic's `parameter_of(point)`.
    ///
    /// Apart from conic pairs, the payload is the same whichever shape is the
    /// receiver, so parameters keep their roles: in `circle.intersect(line)`
    /// each `t` is still a line parameter. Raises `TypeError` for any other
    /// type.
    #[pyo3(signature = (other, tol = None))]
    fn intersect(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tolerance(tol);
        dispatch_intersect!(other, "Circle3D", "Plane, Cylinder, Cone, Sphere, Line3D, Circle3D, Ellipse3D, Parabola3D or Hyperbola3D", {
            PyPlane => |o| self.plane_result(py, &o, tol),
            PyCylinder => |o| self.cylinder_result(py, &o, tol),
            PyCone => |o| self.cone_result(py, &o, tol),
            PySphere => |o| self.sphere_result(py, &o, tol),
            PyLine3D => |o| o.circle_result(py, self, tol),
            PyCircle3D => |o| self.circle_result(py, &o, tol),
            PyEllipse3D => |o| self.ellipse_result(py, &o, tol),
            PyParabola3D => |o| self.parabola_result(py, &o, tol),
            PyHyperbola3D => |o| self.hyperbola_result(py, &o, tol),
        })
    }

    /// Intersects this circle with another 3D circle.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (other, tol = None))]
    fn intersect_circle(
        &self,
        py: Python<'_>,
        other: &PyCircle3D,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Circle3D.intersect_circle")?;
        self.circle_result(py, other, tolerance(tol))
    }

    /// Intersects this circle with an ellipse.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (ellipse, tol = None))]
    fn intersect_ellipse(
        &self,
        py: Python<'_>,
        ellipse: &PyEllipse3D,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Circle3D.intersect_ellipse")?;
        self.ellipse_result(py, ellipse, tolerance(tol))
    }

    /// Intersects this circle with a parabola.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (parabola, tol = None))]
    fn intersect_parabola(
        &self,
        py: Python<'_>,
        parabola: &PyParabola3D,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Circle3D.intersect_parabola")?;
        self.parabola_result(py, parabola, tolerance(tol))
    }

    /// Intersects this circle with a hyperbola.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (hyperbola, tol = None))]
    fn intersect_hyperbola(
        &self,
        py: Python<'_>,
        hyperbola: &PyHyperbola3D,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Circle3D.intersect_hyperbola")?;
        self.hyperbola_result(py, hyperbola, tolerance(tol))
    }

    /// Intersects this circle with a plane.
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
        warn_deprecated(py, "Circle3D.intersect_plane")?;
        self.plane_result(py, plane, tolerance(tol))
    }

    /// Intersects this circle with a sphere.
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
        warn_deprecated(py, "Circle3D.intersect_sphere")?;
        self.sphere_result(py, sphere, tolerance(tol))
    }

    /// Intersects this circle with a cylinder.
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
        warn_deprecated(py, "Circle3D.intersect_cylinder")?;
        self.cylinder_result(py, cylinder, tolerance(tol))
    }

    /// Intersects this circle with a cone.
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
        warn_deprecated(py, "Circle3D.intersect_cone")?;
        self.cone_result(py, cone, tolerance(tol))
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyCircle3D {
    pub(crate) fn circle_result(
        &self,
        py: Python<'_>,
        other: &PyCircle3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&other.0, tol) {
            CircleCircle3DIntersection::Points(p1, p2) => Ok((
                "points".to_string(),
                (
                    PyPoint3D(p1).into_pyobject(py)?.into_any().unbind(),
                    PyPoint3D(p2).into_pyobject(py)?.into_any().unbind(),
                )
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            )),
            CircleCircle3DIntersection::Tangent(p) => Ok((
                "tangent".to_string(),
                PyPoint3D(p).into_pyobject(py)?.into_any().unbind(),
            )),
            CircleCircle3DIntersection::Empty => Ok(("empty".to_string(), py.None())),
            CircleCircle3DIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
            CircleCircle3DIntersection::NotAnalytic => Ok(("not_analytic".to_string(), py.None())),
        }
    }

    pub(crate) fn ellipse_result(
        &self,
        py: Python<'_>,
        ellipse: &PyEllipse3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_conic_to_py(py, self.0.intersect_exact(&ellipse.0, tol))
    }

    pub(crate) fn parabola_result(
        &self,
        py: Python<'_>,
        parabola: &PyParabola3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_conic_to_py(py, self.0.intersect_exact(&parabola.0, tol))
    }

    pub(crate) fn hyperbola_result(
        &self,
        py: Python<'_>,
        hyperbola: &PyHyperbola3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_conic_to_py(py, self.0.intersect_exact(&hyperbola.0, tol))
    }

    pub(crate) fn plane_result(
        &self,
        py: Python<'_>,
        plane: &PyPlane,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        circle_surface_to_py(py, self.0.intersect_exact(&plane.0, tol))
    }

    pub(crate) fn sphere_result(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        circle_surface_to_py(py, self.0.intersect_exact(&sphere.0, tol))
    }

    pub(crate) fn cylinder_result(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        circle_surface_to_py(py, self.0.intersect_exact(&cylinder.0, tol))
    }

    pub(crate) fn cone_result(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        circle_surface_to_py(py, self.0.intersect_exact(&cone.0, tol))
    }
}

/// An ellipse in 3D.
#[pyclass(name = "Ellipse3D", module = "geomcore.curves", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyEllipse3D(pub(crate) Ellipse3D);

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

    /// The (first, last) parameter bounds (`[0, 2π)`).
    fn bounds(&self) -> (f64, f64) {
        ParametricCurve3D::bounds(&self.0)
    }

    /// The axis-aligned box around the shape (`None` when unbounded).
    fn bounding_box(&self) -> Option<PyBoundingBox3D> {
        self.0.bounding_box().map(PyBoundingBox3D)
    }

    /// The ellipse as an exact periodic rational quadratic B-spline over
    /// `[0, 2π]`.
    fn to_bspline(&self) -> PyBSplineCurve3D {
        PyBSplineCurve3D(self.0.to_bspline())
    }

    /// The semi-major radius.
    fn major_radius(&self) -> f64 {
        self.0.major_radius()
    }

    /// The semi-minor radius.
    fn minor_radius(&self) -> f64 {
        self.0.minor_radius()
    }

    /// The enclosed area (`PI*major*minor`).
    fn area(&self) -> f64 {
        self.0.area()
    }

    /// Image of this ellipse under `transform` (raises ValueError for a
    /// collapsing transform).
    #[pyo3(signature = (transform,))]
    fn transformed(&self, transform: &PyTransform) -> PyResult<Self> {
        Ok(PyEllipse3D(
            self.0.transformed(transform.0).map_err(val_err)?,
        ))
    }

    /// Evaluate the point at angle `u` (radians).
    fn eval_point(&self, u: f64) -> PyPoint3D {
        PyPoint3D(self.0.eval_point(u))
    }

    /// Evaluate many parameters at once.
    fn eval_points(&self, us: Vec<f64>) -> Vec<PyPoint3D> {
        self.0.eval_points(&us).into_iter().map(PyPoint3D).collect()
    }

    /// Discretize the ellipse into points joined by straight segments,
    /// each segment's chord deviation at most `max_chord_deviation` and,
    /// if given, its angle deviation at most `max_angle_deviation`
    /// (radians).
    #[pyo3(signature = (max_chord_deviation, max_angle_deviation = None))]
    fn discretize(
        &self,
        max_chord_deviation: f64,
        max_angle_deviation: Option<f64>,
    ) -> PyResult<PyDiscretization3D> {
        discretize_3d(&self.0, max_chord_deviation, max_angle_deviation)
    }

    /// Evaluate the derivative of the given order (>= 1) at `u`.
    fn eval_derivative(&self, u: f64, order: u32) -> PyResult<PyVector3D> {
        check_curve_order(order, u32::MAX)?;
        Ok(PyVector3D(self.0.eval_derivative(u, order)))
    }

    /// Recover the parameter in [0, 2*pi) of a point lying on the ellipse.
    fn parameter_of(&self, point: PyPoint3D) -> f64 {
        self.0.parameter_of(point.0)
    }

    /// Returns whether `point` lies on the curve within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// All stationary points of the distance from `point` to the ellipse,
    /// as `(parameter, distance)` tuples ordered by ascending distance.
    #[pyo3(signature = (point, tol = None))]
    fn extrema(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let tol = tolerance(tol);
        self.0
            .extrema(point.0, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Projects `point` onto the ellipse, returning `(parameter, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64) {
        let proj = self.0.project_point(point.0, tolerance(tol));
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the ellipse, returning a
    /// `(parameter, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(&self, points: Vec<PyPoint3D>, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tolerance(tol);
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Intersects this ellipse with another shape, using the exact solver
    /// for the type of `other`. Returns a `(kind, payload)` tuple:
    ///
    /// - `Plane`, `Cylinder`, `Cone`, `Sphere`: Returns
    ///   `("hits", [(t, Point3D, multiplicity)])`, `("coincident", None)` or
    ///   `("empty", None)`.
    /// - `Line3D`: Returns
    ///   `("points", ((t1, Point3D, u1), (t2, Point3D, u2)))`,
    ///   `("tangent", (t, Point3D, u))` or `("empty", None)`.
    /// - `Circle3D`, `Ellipse3D`, `Parabola3D`, `Hyperbola3D`: Returns
    ///   `("hits", [(s, Point3D, t, multiplicity)])`, `("coincident", None)`
    ///   or `("empty", None)`, with `s` on this ellipse and `t` on `other`
    ///   (multiplicity ≥ 2 marks tangency). A hyperbola is one branch.
    ///   Most pairs compute only one of `s` and `t`: the other is `None`;
    ///   recover it with that conic's `parameter_of(point)`.
    ///
    /// Apart from conic pairs, the payload is the same whichever shape is the
    /// receiver, so parameters keep their roles: in `circle.intersect(line)`
    /// each `t` is still a line parameter. Raises `TypeError` for any other
    /// type.
    #[pyo3(signature = (other, tol = None))]
    fn intersect(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tolerance(tol);
        dispatch_intersect!(other, "Ellipse3D", "Plane, Cylinder, Cone, Sphere, Line3D, Circle3D, Ellipse3D, Parabola3D or Hyperbola3D", {
            PyPlane => |o| self.plane_result(py, &o, tol),
            PyCylinder => |o| self.cylinder_result(py, &o, tol),
            PyCone => |o| self.cone_result(py, &o, tol),
            PySphere => |o| self.sphere_result(py, &o, tol),
            PyLine3D => |o| o.ellipse_result(py, self, tol),
            PyCircle3D => |o| conic_conic_to_py(py, self.0.intersect_exact(&o.0, tol)),
            PyEllipse3D => |o| conic_conic_to_py(py, self.0.intersect_exact(&o.0, tol)),
            PyParabola3D => |o| conic_conic_to_py(py, self.0.intersect_exact(&o.0, tol)),
            PyHyperbola3D => |o| conic_conic_to_py(py, self.0.intersect_exact(&o.0, tol)),
        })
    }

    /// Intersects this ellipse with a plane.
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
        warn_deprecated(py, "Ellipse3D.intersect_plane")?;
        self.plane_result(py, plane, tolerance(tol))
    }

    /// Intersects this ellipse with a sphere.
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
        warn_deprecated(py, "Ellipse3D.intersect_sphere")?;
        self.sphere_result(py, sphere, tolerance(tol))
    }

    /// Intersects this ellipse with a cylinder.
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
        warn_deprecated(py, "Ellipse3D.intersect_cylinder")?;
        self.cylinder_result(py, cylinder, tolerance(tol))
    }

    /// Intersects this ellipse with a cone.
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
        warn_deprecated(py, "Ellipse3D.intersect_cone")?;
        self.cone_result(py, cone, tolerance(tol))
    }

    /// Intersects this ellipse with another 3D ellipse.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (other, tol = None))]
    fn intersect_ellipse(
        &self,
        py: Python<'_>,
        other: &PyEllipse3D,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Ellipse3D.intersect_ellipse")?;
        self.ellipse_result(py, other, tolerance(tol))
    }

    /// Not available for ellipses in this release; always raises `ValueError`.
    #[pyo3(signature = (surface, tol = None))]
    fn parametrize_on(
        &self,
        py: Python<'_>,
        surface: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<Py<PyAny>> {
        let s = extract_surface(surface)?;
        let tol = tolerance(tol);
        curve2d_to_py(py, self.0.parametrize_on(s, tol).map_err(val_err)?)
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyEllipse3D {
    pub(crate) fn plane_result(
        &self,
        py: Python<'_>,
        plane: &PyPlane,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect_exact(&plane.0, tol))
    }

    pub(crate) fn sphere_result(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect_exact(&sphere.0, tol))
    }

    pub(crate) fn cylinder_result(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect_exact(&cylinder.0, tol))
    }

    pub(crate) fn cone_result(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect_exact(&cone.0, tol))
    }

    pub(crate) fn ellipse_result(
        &self,
        py: Python<'_>,
        other: &PyEllipse3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_conic_to_py(py, self.0.intersect_exact(&other.0, tol))
    }
}

/// A parabola in 3D, parametrized so the apex is at parameter 0.
#[pyclass(name = "Parabola3D", module = "geomcore.curves", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyParabola3D(pub(crate) Parabola3D);

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

    /// The (first, last) parameter bounds (unbounded here).
    fn bounds(&self) -> (f64, f64) {
        ParametricCurve3D::bounds(&self.0)
    }

    /// The axis-aligned box around the shape (`None` when unbounded).
    fn bounding_box(&self) -> Option<PyBoundingBox3D> {
        self.0.bounding_box().map(PyBoundingBox3D)
    }

    /// Image of this parabola under `transform` (raises ValueError for a
    /// collapsing transform).
    #[pyo3(signature = (transform,))]
    fn transformed(&self, transform: &PyTransform) -> PyResult<Self> {
        Ok(PyParabola3D(
            self.0.transformed(transform.0).map_err(val_err)?,
        ))
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
    fn eval_derivative(&self, u: f64, order: u32) -> PyResult<PyVector3D> {
        check_curve_order(order, u32::MAX)?;
        Ok(PyVector3D(self.0.eval_derivative(u, order)))
    }

    /// Recover the parameter of a point lying on the parabola.
    fn parameter_of(&self, point: PyPoint3D) -> f64 {
        self.0.parameter_of(point.0)
    }

    /// Returns whether `point` lies on the curve within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// All stationary points of the distance from `point` to the parabola,
    /// as `(parameter, distance)` tuples ordered by ascending distance.
    #[pyo3(signature = (point, tol = None))]
    fn extrema(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let tol = tolerance(tol);
        self.0
            .extrema(point.0, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Projects `point` onto the parabola, returning `(parameter, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64) {
        let proj = self.0.project_point(point.0, tolerance(tol));
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the parabola, returning a
    /// `(parameter, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(&self, points: Vec<PyPoint3D>, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tolerance(tol);
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Intersects this parabola with another shape, using the exact solver
    /// for the type of `other`. Returns a `(kind, payload)` tuple:
    ///
    /// - `Plane`, `Cylinder`, `Cone`, `Sphere`: Returns
    ///   `("hits", [(t, Point3D, multiplicity)])`, `("coincident", None)` or
    ///   `("empty", None)`.
    /// - `Line3D`: Returns
    ///   `("points", ((t1, Point3D, u1), (t2, Point3D, u2)))`,
    ///   `("tangent", (t, Point3D, u))` or `("empty", None)`.
    /// - `Circle3D`, `Ellipse3D`, `Parabola3D`, `Hyperbola3D`: Returns
    ///   `("hits", [(s, Point3D, t, multiplicity)])`, `("coincident", None)`
    ///   or `("empty", None)`, with `s` on this parabola and `t` on `other`
    ///   (multiplicity ≥ 2 marks tangency). A hyperbola is one branch.
    ///   Most pairs compute only one of `s` and `t`: the other is `None`;
    ///   recover it with that conic's `parameter_of(point)`.
    ///
    /// Apart from conic pairs, the payload is the same whichever shape is the
    /// receiver, so parameters keep their roles: in `circle.intersect(line)`
    /// each `t` is still a line parameter. Raises `TypeError` for any other
    /// type.
    #[pyo3(signature = (other, tol = None))]
    fn intersect(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tolerance(tol);
        dispatch_intersect!(other, "Parabola3D", "Plane, Cylinder, Cone, Sphere, Line3D, Circle3D, Ellipse3D, Parabola3D or Hyperbola3D", {
            PyPlane => |o| self.plane_result(py, &o, tol),
            PyCylinder => |o| self.cylinder_result(py, &o, tol),
            PyCone => |o| self.cone_result(py, &o, tol),
            PySphere => |o| self.sphere_result(py, &o, tol),
            PyLine3D => |o| o.parabola_result(py, self, tol),
            PyCircle3D => |o| conic_conic_to_py(py, self.0.intersect_exact(&o.0, tol)),
            PyEllipse3D => |o| conic_conic_to_py(py, self.0.intersect_exact(&o.0, tol)),
            PyParabola3D => |o| conic_conic_to_py(py, self.0.intersect_exact(&o.0, tol)),
            PyHyperbola3D => |o| conic_conic_to_py(py, self.0.intersect_exact(&o.0, tol)),
        })
    }

    /// Intersects this parabola with a plane.
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
        warn_deprecated(py, "Parabola3D.intersect_plane")?;
        self.plane_result(py, plane, tolerance(tol))
    }

    /// Intersects this parabola with a sphere.
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
        warn_deprecated(py, "Parabola3D.intersect_sphere")?;
        self.sphere_result(py, sphere, tolerance(tol))
    }

    /// Intersects this parabola with a cylinder.
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
        warn_deprecated(py, "Parabola3D.intersect_cylinder")?;
        self.cylinder_result(py, cylinder, tolerance(tol))
    }

    /// Intersects this parabola with a cone.
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
        warn_deprecated(py, "Parabola3D.intersect_cone")?;
        self.cone_result(py, cone, tolerance(tol))
    }

    /// Not available for parabolas in this release; always raises `ValueError`.
    #[pyo3(signature = (surface, tol = None))]
    fn parametrize_on(
        &self,
        py: Python<'_>,
        surface: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<Py<PyAny>> {
        let s = extract_surface(surface)?;
        let tol = tolerance(tol);
        curve2d_to_py(py, self.0.parametrize_on(s, tol).map_err(val_err)?)
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyParabola3D {
    pub(crate) fn plane_result(
        &self,
        py: Python<'_>,
        plane: &PyPlane,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect_exact(&plane.0, tol))
    }

    pub(crate) fn sphere_result(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect_exact(&sphere.0, tol))
    }

    pub(crate) fn cylinder_result(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect_exact(&cylinder.0, tol))
    }

    pub(crate) fn cone_result(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect_exact(&cone.0, tol))
    }
}

/// One branch of a hyperbola in 3D.
#[pyclass(name = "Hyperbola3D", module = "geomcore.curves", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyHyperbola3D(pub(crate) Hyperbola3D);

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

    /// The (first, last) parameter bounds (unbounded here).
    fn bounds(&self) -> (f64, f64) {
        ParametricCurve3D::bounds(&self.0)
    }

    /// The axis-aligned box around the shape (`None` when unbounded).
    fn bounding_box(&self) -> Option<PyBoundingBox3D> {
        self.0.bounding_box().map(PyBoundingBox3D)
    }

    /// The semi-major (transverse) radius.
    fn major_radius(&self) -> f64 {
        self.0.major_radius()
    }

    /// The semi-minor (conjugate) radius.
    fn minor_radius(&self) -> f64 {
        self.0.minor_radius()
    }

    /// Image of this hyperbola under `transform` (raises ValueError for a
    /// collapsing transform).
    #[pyo3(signature = (transform,))]
    fn transformed(&self, transform: &PyTransform) -> PyResult<Self> {
        Ok(PyHyperbola3D(
            self.0.transformed(transform.0).map_err(val_err)?,
        ))
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
    fn eval_derivative(&self, u: f64, order: u32) -> PyResult<PyVector3D> {
        check_curve_order(order, u32::MAX)?;
        Ok(PyVector3D(self.0.eval_derivative(u, order)))
    }

    /// Recover the parameter of a point lying on the hyperbola.
    fn parameter_of(&self, point: PyPoint3D) -> f64 {
        self.0.parameter_of(point.0)
    }

    /// Returns whether `point` lies on the curve within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// All stationary points of the distance from `point` to the hyperbola,
    /// as `(parameter, distance)` tuples ordered by ascending distance.
    #[pyo3(signature = (point, tol = None))]
    fn extrema(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let tol = tolerance(tol);
        self.0
            .extrema(point.0, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Projects `point` onto the hyperbola, returning `(parameter, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64) {
        let proj = self.0.project_point(point.0, tolerance(tol));
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the hyperbola, returning a
    /// `(parameter, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(&self, points: Vec<PyPoint3D>, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tolerance(tol);
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Intersects this hyperbola with another shape, using the exact solver
    /// for the type of `other`. Returns a `(kind, payload)` tuple:
    ///
    /// - `Plane`, `Cylinder`, `Cone`, `Sphere`: Returns
    ///   `("hits", [(t, Point3D, multiplicity)])`, `("coincident", None)` or
    ///   `("empty", None)`.
    /// - `Line3D`: Returns
    ///   `("points", ((t1, Point3D, u1), (t2, Point3D, u2)))`,
    ///   `("tangent", (t, Point3D, u))` or `("empty", None)`.
    /// - `Circle3D`, `Ellipse3D`, `Parabola3D`, `Hyperbola3D`: Returns
    ///   `("hits", [(s, Point3D, t, multiplicity)])`, `("coincident", None)`
    ///   or `("empty", None)`, with `s` on this hyperbola and `t` on `other`
    ///   (multiplicity ≥ 2 marks tangency). A hyperbola is one branch.
    ///   Most pairs compute only one of `s` and `t`: the other is `None`;
    ///   recover it with that conic's `parameter_of(point)`.
    ///
    /// Apart from conic pairs, the payload is the same whichever shape is the
    /// receiver, so parameters keep their roles: in `circle.intersect(line)`
    /// each `t` is still a line parameter. Raises `TypeError` for any other
    /// type.
    #[pyo3(signature = (other, tol = None))]
    fn intersect(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let tol = tolerance(tol);
        dispatch_intersect!(other, "Hyperbola3D", "Plane, Cylinder, Cone, Sphere, Line3D, Circle3D, Ellipse3D, Parabola3D or Hyperbola3D", {
            PyPlane => |o| self.plane_result(py, &o, tol),
            PyCylinder => |o| self.cylinder_result(py, &o, tol),
            PyCone => |o| self.cone_result(py, &o, tol),
            PySphere => |o| self.sphere_result(py, &o, tol),
            PyLine3D => |o| o.hyperbola_result(py, self, tol),
            PyCircle3D => |o| conic_conic_to_py(py, self.0.intersect_exact(&o.0, tol)),
            PyEllipse3D => |o| conic_conic_to_py(py, self.0.intersect_exact(&o.0, tol)),
            PyParabola3D => |o| conic_conic_to_py(py, self.0.intersect_exact(&o.0, tol)),
            PyHyperbola3D => |o| conic_conic_to_py(py, self.0.intersect_exact(&o.0, tol)),
        })
    }

    /// Intersects this hyperbola with a plane.
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
        warn_deprecated(py, "Hyperbola3D.intersect_plane")?;
        self.plane_result(py, plane, tolerance(tol))
    }

    /// Intersects this hyperbola with a sphere.
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
        warn_deprecated(py, "Hyperbola3D.intersect_sphere")?;
        self.sphere_result(py, sphere, tolerance(tol))
    }

    /// Intersects this hyperbola with a cylinder.
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
        warn_deprecated(py, "Hyperbola3D.intersect_cylinder")?;
        self.cylinder_result(py, cylinder, tolerance(tol))
    }

    /// Intersects this hyperbola with a cone.
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
        warn_deprecated(py, "Hyperbola3D.intersect_cone")?;
        self.cone_result(py, cone, tolerance(tol))
    }

    /// Not available for hyperbolas in this release; always raises `ValueError`.
    #[pyo3(signature = (surface, tol = None))]
    fn parametrize_on(
        &self,
        py: Python<'_>,
        surface: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<Py<PyAny>> {
        let s = extract_surface(surface)?;
        let tol = tolerance(tol);
        curve2d_to_py(py, self.0.parametrize_on(s, tol).map_err(val_err)?)
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyHyperbola3D {
    pub(crate) fn plane_result(
        &self,
        py: Python<'_>,
        plane: &PyPlane,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect_exact(&plane.0, tol))
    }

    pub(crate) fn sphere_result(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect_exact(&sphere.0, tol))
    }

    pub(crate) fn cylinder_result(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect_exact(&cylinder.0, tol))
    }

    pub(crate) fn cone_result(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect_exact(&cone.0, tol))
    }
}

/// A B-spline (optionally rational, optionally periodic) curve in 3D.
#[pyclass(name = "BSplineCurve3D", module = "geomcore.curves", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyBSplineCurve3D(pub(crate) BSplineCurve3D);

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

    /// Interpolate a clamped B-spline through `points`.
    ///
    /// `parametrization` is `"centripetal"` (default, tamer at sharp
    /// turns) or `"chordal"`.
    #[staticmethod]
    #[pyo3(signature = (points, degree, parametrization = "centripetal"))]
    fn interpolate(points: Vec<PyPoint3D>, degree: usize, parametrization: &str) -> PyResult<Self> {
        let param = parse_parametrization(parametrization)?;
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        Ok(PyBSplineCurve3D(
            BSplineCurve3D::interpolate(&points, degree, param).map_err(val_err)?,
        ))
    }

    /// Least-squares B-spline fit through `points` with `num_poles` poles.
    ///
    /// `parametrization` is `"centripetal"` (default) or `"chordal"`.
    #[staticmethod]
    #[pyo3(signature = (points, degree, num_poles, parametrization = "centripetal"))]
    fn approximate(
        points: Vec<PyPoint3D>,
        degree: usize,
        num_poles: usize,
        parametrization: &str,
    ) -> PyResult<Self> {
        let param = parse_parametrization(parametrization)?;
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        Ok(PyBSplineCurve3D(
            BSplineCurve3D::approximate(&points, degree, num_poles, param).map_err(val_err)?,
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

    /// The control poles.
    fn poles(&self) -> Vec<PyPoint3D> {
        self.0.poles().iter().map(|p| PyPoint3D(*p)).collect()
    }

    /// The per-pole weights, or `None` when the curve is not rational.
    fn weights(&self) -> Option<Vec<f64>> {
        self.0.weights().map(|w| w.to_vec())
    }

    /// The distinct knots.
    fn knots(&self) -> Vec<f64> {
        self.0.knots().to_vec()
    }

    /// The multiplicity of each knot.
    fn multiplicities(&self) -> Vec<u32> {
        self.0.multiplicities().to_vec()
    }

    /// The axis-aligned box around the shape (`None` when unbounded).
    fn bounding_box(&self) -> Option<PyBoundingBox3D> {
        self.0.bounding_box().map(PyBoundingBox3D)
    }

    /// Least-squares B-spline fit through `points` with as few poles as
    /// keep every point within `max_deviation` of the curve at its
    /// parameter.
    ///
    /// `parametrization` is `"centripetal"` (default) or `"chordal"`.
    #[staticmethod]
    #[pyo3(signature = (points, degree, max_deviation, parametrization = "centripetal"))]
    fn approximate_within(
        points: Vec<PyPoint3D>,
        degree: usize,
        max_deviation: f64,
        parametrization: &str,
    ) -> PyResult<Self> {
        let param = parse_parametrization(parametrization)?;
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        Ok(PyBSplineCurve3D(
            BSplineCurve3D::approximate_within(&points, degree, max_deviation, param)
                .map_err(val_err)?,
        ))
    }

    /// The curve with the knot `u` inserted `times` times (same shape,
    /// more poles).
    fn insert_knot(&self, u: f64, times: u32) -> PyResult<Self> {
        Ok(PyBSplineCurve3D(
            self.0.insert_knot(u, times).map_err(val_err)?,
        ))
    }

    /// The curve with each of `knots` inserted once (same shape, more
    /// poles).
    fn refine_knots(&self, knots: Vec<f64>) -> PyResult<Self> {
        Ok(PyBSplineCurve3D(
            self.0.refine_knots(&knots).map_err(val_err)?,
        ))
    }

    /// The curve of degree raised by `amount` (same shape).
    fn elevate_degree(&self, amount: usize) -> PyResult<Self> {
        Ok(PyBSplineCurve3D(
            self.0.elevate_degree(amount).map_err(val_err)?,
        ))
    }

    /// The curve with the knot `u` removed up to `times` times, and how
    /// many copies were removed: a removal is kept only while every
    /// point stays within `max_deviation` of this one.
    fn remove_knot(&self, u: f64, times: u32, max_deviation: f64) -> PyResult<(Self, u32)> {
        let (curve, removed) = self
            .0
            .remove_knot(u, times, max_deviation)
            .map_err(val_err)?;
        Ok((PyBSplineCurve3D(curve), removed))
    }

    /// The curve with each of `knots` removed once, in order, and how
    /// many copies were removed in all.
    fn remove_knots(&self, knots: Vec<f64>, max_deviation: f64) -> PyResult<(Self, usize)> {
        let (curve, removed) = self
            .0
            .remove_knots(&knots, max_deviation)
            .map_err(val_err)?;
        Ok((PyBSplineCurve3D(curve), removed))
    }

    /// The polynomial piece of each knot span as a Bézier B-spline, in
    /// order.
    fn to_bezier_pieces(&self) -> Vec<Self> {
        self.0
            .to_bezier_pieces()
            .into_iter()
            .map(PyBSplineCurve3D)
            .collect()
    }

    /// The piece over `[first, last]` as a clamped, non-periodic B-spline
    /// (the range may cross the seam of a periodic curve, up to one
    /// period).
    fn extract(&self, first: f64, last: f64) -> PyResult<Self> {
        Ok(PyBSplineCurve3D(
            self.0.extract(first, last).map_err(val_err)?,
        ))
    }

    /// The two halves split at `u`, each a clamped, non-periodic B-spline.
    fn split_at(&self, u: f64) -> PyResult<(Self, Self)> {
        let (a, b) = self.0.split_at(u).map_err(val_err)?;
        Ok((PyBSplineCurve3D(a), PyBSplineCurve3D(b)))
    }

    /// The curve with its parameter mirrored, keeping a periodic curve
    /// periodic.
    fn reversed(&self) -> Self {
        PyBSplineCurve3D(self.0.reversed())
    }

    /// The arc length, integrating the speed to within `max_error`.
    fn length(&self, max_error: f64) -> f64 {
        self.0.length(max_error)
    }

    /// The (first, last) parameter bounds.
    fn bounds(&self) -> (f64, f64) {
        self.0.bounds()
    }

    /// Returns whether `point` lies on the curve within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// The local minima of the distance from `point` to the curve (plus
    /// the interval ends of an open curve), as `(parameter, distance)`
    /// tuples ordered by ascending distance.
    #[pyo3(signature = (point, tol = None))]
    fn extrema(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let tol = tolerance(tol);
        self.0
            .extrema(point.0, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Projects `point` onto the curve, returning `(parameter, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64) {
        let proj = self.0.project_point(point.0, tolerance(tol));
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the curve, returning a
    /// `(parameter, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(&self, points: Vec<PyPoint3D>, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tolerance(tol);
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Image of this curve under `transform` (maps every control pole).
    #[pyo3(signature = (transform,))]
    fn transformed(&self, transform: &PyTransform) -> Self {
        PyBSplineCurve3D(self.0.transformed(transform.0))
    }

    /// Evaluate the point at parameter `u`.
    fn eval_point(&self, u: f64) -> PyPoint3D {
        PyPoint3D(self.0.eval_point(u))
    }

    /// Evaluate many parameters at once.
    fn eval_points(&self, us: Vec<f64>) -> Vec<PyPoint3D> {
        self.0.eval_points(&us).into_iter().map(PyPoint3D).collect()
    }

    /// Discretize the curve into points joined by straight segments,
    /// each segment's chord deviation at most `max_chord_deviation` and,
    /// if given, its angle deviation at most `max_angle_deviation`
    /// (radians).
    #[pyo3(signature = (max_chord_deviation, max_angle_deviation = None))]
    fn discretize(
        &self,
        max_chord_deviation: f64,
        max_angle_deviation: Option<f64>,
    ) -> PyResult<PyDiscretization3D> {
        discretize_3d(&self.0, max_chord_deviation, max_angle_deviation)
    }

    /// Evaluate the derivative at `u`; orders 1 and 2 are supported.
    fn eval_derivative(&self, u: f64, order: u32) -> PyResult<PyVector3D> {
        check_curve_order(order, 2)?;
        Ok(PyVector3D(self.0.eval_derivative(u, order)))
    }

    /// Not available for B-spline curves in this release; always raises `ValueError`.
    #[pyo3(signature = (surface, tol = None))]
    fn parametrize_on(
        &self,
        py: Python<'_>,
        surface: &Bound<'_, PyAny>,
        tol: Option<PyTolerance>,
    ) -> PyResult<Py<PyAny>> {
        let s = extract_surface(surface)?;
        let tol = tolerance(tol);
        curve2d_to_py(py, self.0.parametrize_on(s, tol).map_err(val_err)?)
    }
}

/// A straight line in a surface's 2D (u, v) parameter space.
#[pyclass(name = "Line2D", module = "geomcore.curves", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyLine2D(pub(crate) Line2D);

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
    fn eval_derivative(&self, u: f64, order: u32) -> PyResult<PyVector2D> {
        check_curve_order(order, u32::MAX)?;
        Ok(PyVector2D(self.0.eval_derivative(u, order)))
    }

    /// Recover the parameter of a point lying on the line.
    fn parameter_of(&self, point: PyPoint2D) -> f64 {
        self.0.parameter_of(point.0)
    }

    /// Returns whether `point` lies on the curve within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint2D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// Projects `point` onto the curve, returning `(parameter, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint2D, tol: Option<PyTolerance>) -> (f64, f64) {
        let proj = self.0.project_point(point.0, tolerance(tol));
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the curve, returning a
    /// `(parameter, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(&self, points: Vec<PyPoint2D>, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tolerance(tol);
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Intersects this line with another shape, using the exact solver
    /// for the type of `other`. Returns a `(kind, payload)` tuple:
    ///
    /// - `Line2D`: Returns `("point", (s, Point2D, t))`, `("parallel", None)`
    ///   or `("coincident", None)`.
    /// - `Circle2D`: Returns `("points", ((t1, Point2D), (t2, Point2D)))`,
    ///   `("tangent", (t, Point2D))` or `("empty", None)`.
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
        dispatch_intersect!(other, "Line2D", "Line2D or Circle2D", {
            PyLine2D => |o| self.line_result(py, &o, tol),
            PyCircle2D => |o| self.circle_result(py, &o, tol),
        })
    }

    /// Intersects this line with another 2D line.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (other, tol = None))]
    fn intersect_line(
        &self,
        py: Python<'_>,
        other: &PyLine2D,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Line2D.intersect_line")?;
        self.line_result(py, other, tolerance(tol))
    }

    /// Intersects this line with a 2D circle.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (circle, tol = None))]
    fn intersect_circle(
        &self,
        py: Python<'_>,
        circle: &PyCircle2D,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Line2D.intersect_circle")?;
        self.circle_result(py, circle, tolerance(tol))
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

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyLine2D {
    pub(crate) fn line_result(
        &self,
        py: Python<'_>,
        other: &PyLine2D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&other.0, tol) {
            LineLine2DIntersection::Point(s, p, t) => Ok((
                "point".to_string(),
                (s, PyPoint2D(p).into_pyobject(py)?.into_any().unbind(), t)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            )),
            LineLine2DIntersection::Parallel => Ok(("parallel".to_string(), py.None())),
            LineLine2DIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
        }
    }

    pub(crate) fn circle_result(
        &self,
        py: Python<'_>,
        circle: &PyCircle2D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&circle.0, tol) {
            LineCircle2DIntersection::Points((t1, p1), (t2, p2)) => Ok((
                "points".to_string(),
                (point2(py, t1, p1)?, point2(py, t2, p2)?)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            )),
            LineCircle2DIntersection::Tangent(t, p) => {
                Ok(("tangent".to_string(), point2(py, t, p)?))
            }
            LineCircle2DIntersection::Empty => Ok(("empty".to_string(), py.None())),
        }
    }
}

/// A circle in a surface's 2D (u, v) parameter space.
#[pyclass(name = "Circle2D", module = "geomcore.curves", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyCircle2D(pub(crate) Circle2D);

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

    /// Discretize the circle into points joined by straight segments,
    /// each segment's chord deviation at most `max_chord_deviation` and,
    /// if given, its angle deviation at most `max_angle_deviation`
    /// (radians).
    #[pyo3(signature = (max_chord_deviation, max_angle_deviation = None))]
    fn discretize(
        &self,
        max_chord_deviation: f64,
        max_angle_deviation: Option<f64>,
    ) -> PyResult<PyDiscretization2D> {
        discretize_2d(&self.0, max_chord_deviation, max_angle_deviation)
    }

    /// Evaluate the derivative of the given order (>= 1) at `u`.
    fn eval_derivative(&self, u: f64, order: u32) -> PyResult<PyVector2D> {
        check_curve_order(order, u32::MAX)?;
        Ok(PyVector2D(self.0.eval_derivative(u, order)))
    }

    /// Recover the parameter in [0, 2*pi) of a point lying on the circle.
    fn parameter_of(&self, point: PyPoint2D) -> f64 {
        self.0.parameter_of(point.0)
    }

    /// Returns whether `point` lies on the curve within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint2D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// Projects `point` onto the curve, returning `(parameter, distance)`.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint2D, tol: Option<PyTolerance>) -> (f64, f64) {
        let proj = self.0.project_point(point.0, tolerance(tol));
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the curve, returning a
    /// `(parameter, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(&self, points: Vec<PyPoint2D>, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tolerance(tol);
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Intersects this circle with another shape, using the exact solver
    /// for the type of `other`. Returns a `(kind, payload)` tuple:
    ///
    /// - `Line2D`: Returns `("points", ((t1, Point2D), (t2, Point2D)))`,
    ///   `("tangent", (t, Point2D))` or `("empty", None)`.
    /// - `Circle2D`: Returns `("points", (Point2D, Point2D))`,
    ///   `("tangent", Point2D)`, `("empty", None)` or `("coincident", None)`.
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
        dispatch_intersect!(other, "Circle2D", "Line2D or Circle2D", {
            PyLine2D => |o| o.circle_result(py, self, tol),
            PyCircle2D => |o| self.circle_result(py, &o, tol),
        })
    }

    /// Intersects this circle with another 2D circle.
    ///
    /// Deprecated since 0.29.0: use `intersect`, which returns the same
    /// result.
    #[pyo3(signature = (other, tol = None))]
    fn intersect_circle(
        &self,
        py: Python<'_>,
        other: &PyCircle2D,
        tol: Option<PyTolerance>,
    ) -> PyResult<(String, Py<PyAny>)> {
        warn_deprecated(py, "Circle2D.intersect_circle")?;
        self.circle_result(py, other, tolerance(tol))
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

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyCircle2D {
    pub(crate) fn circle_result(
        &self,
        py: Python<'_>,
        other: &PyCircle2D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect_exact(&other.0, tol) {
            CircleCircle2DIntersection::Points(p1, p2) => Ok((
                "points".to_string(),
                (
                    PyPoint2D(p1).into_pyobject(py)?.into_any().unbind(),
                    PyPoint2D(p2).into_pyobject(py)?.into_any().unbind(),
                )
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            )),
            CircleCircle2DIntersection::Tangent(p) => Ok((
                "tangent".to_string(),
                PyPoint2D(p).into_pyobject(py)?.into_any().unbind(),
            )),
            CircleCircle2DIntersection::Empty => Ok(("empty".to_string(), py.None())),
            CircleCircle2DIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
        }
    }
}

/// The highest derivative order a 3D basis curve evaluates.
pub(crate) fn max_curve_order(curve: &Curve3D) -> u32 {
    match curve {
        Curve3D::BSpline(_) => 2,
        _ => u32::MAX,
    }
}

/// A 3D curve restricted to the parameter range `[first, last]` of a
/// basis curve: a line segment, an arc of a circle or conic, or a piece of
/// a B-spline. Geometry only, with no vertices or orientation.
///
/// The basis keeps its parametrisation and `bounds()` is
/// `(first, last)`. On a periodic basis (circle, ellipse, periodic
/// B-spline) the range may cross the seam, up to one period. Pass it to
/// `curve_curve_extrema`, `intersect_curve_curve`,
/// `intersect_curve_surface` or `parametrize_numeric` to search only the
/// range.
#[pyclass(name = "TrimmedCurve3D", module = "geomcore.curves", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyTrimmedCurve3D(pub(crate) TrimmedCurve3D);

#[pymethods]
impl PyTrimmedCurve3D {
    #[new]
    /// Restrict `basis` (any 3D curve) to the parameter range
    /// `[first, last]`. Raises `ValueError` for non-finite ends, an empty
    /// range, a range outside a non-periodic basis's bounds, or one longer
    /// than a periodic basis's period.
    fn py_new(basis: &Bound<'_, PyAny>, first: f64, last: f64) -> PyResult<Self> {
        let basis = extract_curve3d(basis)?;
        Ok(PyTrimmedCurve3D(
            TrimmedCurve3D::new(basis, first, last).map_err(val_err)?,
        ))
    }

    /// The segment from `start` to `end`, parametrised by arc length from
    /// `start` over `[0, |end - start|]`.
    #[staticmethod]
    fn segment(start: PyPoint3D, end: PyPoint3D) -> PyResult<Self> {
        let segment = TrimmedCurve3D::segment(start.0, end.0).map_err(val_err)?;
        Ok(PyTrimmedCurve3D(segment.into_curve3d()))
    }

    /// The basis curve.
    fn basis(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        curve3d_to_py(py, self.0.basis().clone())
    }

    /// The start of the parameter range.
    fn first(&self) -> f64 {
        self.0.first()
    }

    /// The end of the parameter range.
    fn last(&self) -> f64 {
        self.0.last()
    }

    /// The (first, last) parameter bounds.
    fn bounds(&self) -> (f64, f64) {
        ParametricCurve3D::bounds(&self.0)
    }

    /// The axis-aligned box around the shape (`None` when unbounded).
    fn bounding_box(&self) -> Option<PyBoundingBox3D> {
        self.0.bounding_box().map(PyBoundingBox3D)
    }

    /// The trimmed piece as an exact B-spline over the same range: a
    /// clamped B-spline for an elementary basis, the extracted piece for
    /// a B-spline one (raises `ValueError` when the poles would overflow
    /// `f64`).
    fn to_bspline(&self) -> PyResult<PyBSplineCurve3D> {
        let (first, last) = (self.0.first(), self.0.last());
        let basis = self.0.basis().clone();
        if let Some(b) = basis.as_bspline() {
            return Ok(PyBSplineCurve3D(b.extract(first, last).map_err(val_err)?));
        }
        let elementary = basis
            .as_elementary()
            .expect("every non-B-spline curve is elementary");
        Ok(PyBSplineCurve3D(
            TrimmedCurve3D::new(elementary, first, last)
                .map_err(val_err)?
                .to_bspline()
                .map_err(val_err)?,
        ))
    }

    /// The point at the start of the range.
    fn start_point(&self) -> PyPoint3D {
        PyPoint3D(self.0.start_point())
    }

    /// The point at the end of the range.
    fn end_point(&self) -> PyPoint3D {
        PyPoint3D(self.0.end_point())
    }

    /// Evaluate the basis point at parameter `u` (not checked against the
    /// range).
    fn eval_point(&self, u: f64) -> PyPoint3D {
        PyPoint3D(self.0.eval_point(u))
    }

    /// Evaluate many parameters at once.
    fn eval_points(&self, us: Vec<f64>) -> Vec<PyPoint3D> {
        self.0.eval_points(&us).into_iter().map(PyPoint3D).collect()
    }

    /// Discretize the trimmed curve into points joined by straight segments,
    /// each segment's chord deviation at most `max_chord_deviation` and,
    /// if given, its angle deviation at most `max_angle_deviation`
    /// (radians).
    #[pyo3(signature = (max_chord_deviation, max_angle_deviation = None))]
    fn discretize(
        &self,
        max_chord_deviation: f64,
        max_angle_deviation: Option<f64>,
    ) -> PyResult<PyDiscretization3D> {
        discretize_3d(&self.0, max_chord_deviation, max_angle_deviation)
    }

    /// Evaluate the derivative of the given order (>= 1) at `u`, within
    /// the orders the basis supports.
    fn eval_derivative(&self, u: f64, order: u32) -> PyResult<PyVector3D> {
        check_curve_order(order, max_curve_order(self.0.basis()))?;
        Ok(PyVector3D(self.0.eval_derivative(u, order)))
    }

    /// Returns whether `point` lies on the trimmed curve within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// The basis's distance extrema from `point` inside the range, plus
    /// the two ends, as `(parameter, distance)` tuples ordered by
    /// ascending distance.
    #[pyo3(signature = (point, tol = None))]
    fn extrema(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let tol = tolerance(tol);
        PointExtrema3D::extrema(&self.0, point.0, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Projects `point` onto the trimmed curve, returning
    /// `(parameter, distance)`: the closest basis extremum inside the
    /// range, or the nearer end.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> (f64, f64) {
        let proj = self.0.project_point(point.0, tolerance(tol));
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the trimmed curve, returning a
    /// `(parameter, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(&self, points: Vec<PyPoint3D>, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tolerance(tol);
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    fn __repr__(&self) -> String {
        let (s, e) = (self.0.start_point(), self.0.end_point());
        format!(
            "TrimmedCurve3D(first={}, last={}, start=({}, {}, {}), end=({}, {}, {}))",
            self.0.first(),
            self.0.last(),
            s.x,
            s.y,
            s.z,
            e.x,
            e.y,
            e.z
        )
    }
}

/// A 2D curve restricted to the parameter range `[first, last]` of a
/// basis curve: a line segment or a circular arc, typically a trimmed
/// pcurve in a surface's (u, v) parameter space. Mirrors `TrimmedCurve3D`.
#[pyclass(name = "TrimmedCurve2D", module = "geomcore.curves", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyTrimmedCurve2D(pub(crate) TrimmedCurve2D);

#[pymethods]
impl PyTrimmedCurve2D {
    #[new]
    /// Restrict `basis` (a Line2D or Circle2D) to the parameter range
    /// `[first, last]`, under the rules of `TrimmedCurve3D`.
    fn py_new(basis: &Bound<'_, PyAny>, first: f64, last: f64) -> PyResult<Self> {
        let basis = extract_curve2d(basis)?;
        Ok(PyTrimmedCurve2D(
            TrimmedCurve2D::new(basis, first, last).map_err(val_err)?,
        ))
    }

    /// The segment from `start` to `end`, parametrised by arc length from
    /// `start` over `[0, |end - start|]`.
    #[staticmethod]
    fn segment(start: PyPoint2D, end: PyPoint2D) -> PyResult<Self> {
        let segment = TrimmedCurve2D::segment(start.0, end.0).map_err(val_err)?;
        Ok(PyTrimmedCurve2D(segment.into_curve2d()))
    }

    /// The basis curve.
    fn basis(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        curve2d_to_py(py, *self.0.basis())
    }

    /// The start of the parameter range.
    fn first(&self) -> f64 {
        self.0.first()
    }

    /// The end of the parameter range.
    fn last(&self) -> f64 {
        self.0.last()
    }

    /// The (first, last) parameter bounds.
    fn bounds(&self) -> (f64, f64) {
        (self.0.first(), self.0.last())
    }

    /// The point at the start of the range.
    fn start_point(&self) -> PyPoint2D {
        PyPoint2D(self.0.start_point())
    }

    /// The point at the end of the range.
    fn end_point(&self) -> PyPoint2D {
        PyPoint2D(self.0.end_point())
    }

    /// Evaluate the basis point at parameter `u` (not checked against the
    /// range).
    fn eval_point(&self, u: f64) -> PyPoint2D {
        PyPoint2D(self.0.eval_point(u))
    }

    /// Evaluate many parameters at once.
    fn eval_points(&self, us: Vec<f64>) -> Vec<PyPoint2D> {
        self.0.eval_points(&us).into_iter().map(PyPoint2D).collect()
    }

    /// Discretize the trimmed curve into points joined by straight segments,
    /// each segment's chord deviation at most `max_chord_deviation` and,
    /// if given, its angle deviation at most `max_angle_deviation`
    /// (radians).
    #[pyo3(signature = (max_chord_deviation, max_angle_deviation = None))]
    fn discretize(
        &self,
        max_chord_deviation: f64,
        max_angle_deviation: Option<f64>,
    ) -> PyResult<PyDiscretization2D> {
        discretize_2d(&self.0, max_chord_deviation, max_angle_deviation)
    }

    /// Evaluate the derivative of the given order (>= 1) at `u`.
    fn eval_derivative(&self, u: f64, order: u32) -> PyResult<PyVector2D> {
        check_curve_order(order, u32::MAX)?;
        Ok(PyVector2D(self.0.eval_derivative(u, order)))
    }

    /// Returns whether `point` lies on the trimmed curve within `tol`
    /// (`Tolerance.default()` when omitted).
    #[pyo3(signature = (point, tol = None))]
    fn contains(&self, point: PyPoint2D, tol: Option<PyTolerance>) -> bool {
        self.0.contains(point.0, tolerance(tol))
    }

    /// The basis's distance extrema from `point` inside the range, plus
    /// the two ends, as `(parameter, distance)` tuples ordered by
    /// ascending distance.
    #[pyo3(signature = (point, tol = None))]
    fn extrema(&self, point: PyPoint2D, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let tol = tolerance(tol);
        PointExtrema2D::extrema(&self.0, point.0, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    /// Projects `point` onto the trimmed curve, returning
    /// `(parameter, distance)`: the closest basis extremum inside the
    /// range, or the nearer end.
    #[pyo3(signature = (point, tol = None))]
    fn project_point(&self, point: PyPoint2D, tol: Option<PyTolerance>) -> (f64, f64) {
        let proj = self.0.project_point(point.0, tolerance(tol));
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the trimmed curve, returning a
    /// `(parameter, distance)` tuple per point.
    #[pyo3(signature = (points, tol = None))]
    fn project_points(&self, points: Vec<PyPoint2D>, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let points = points.into_iter().map(|p| p.0).collect::<Vec<_>>();
        let tol = tolerance(tol);
        self.0
            .project_points(&points, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
    }

    fn __repr__(&self) -> String {
        let (s, e) = (self.0.start_point(), self.0.end_point());
        format!(
            "TrimmedCurve2D(first={}, last={}, start=({}, {}), end=({}, {}))",
            self.0.first(),
            self.0.last(),
            s.x,
            s.y,
            e.x,
            e.y
        )
    }
}
