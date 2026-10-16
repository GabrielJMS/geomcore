//! Python bindings for the `geomcore` geometric kernel.
//!
//! The extension module mirrors the Rust crate's namespaces: value types live
//! at the package root (`geomcore`), curves under `geomcore.curves`, and
//! surfaces under `geomcore.surfaces`. Fallible constructors raise
//! `ValueError` carrying the underlying error description.

use std::ffi::CString;

use pyo3::exceptions::{PyDeprecationWarning, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use geomcore::Intersect;
use geomcore::curves::{
    BSplineCurve3D, Circle2D, Circle3D, Curve2D, Curve3D, Ellipse3D, Hyperbola3D, Line2D, Line3D,
    Parabola3D, ParametricCurve3D, PointExtrema2D, PointExtrema3D, PointParametrization,
    TrimmedCurve2D, TrimmedCurve3D,
};
use geomcore::intersect_curve_surface;
use geomcore::marching_intersection;
use geomcore::parametrize_numeric;
use geomcore::surfaces::{
    BSplineSurface, Cone, Cylinder, ParametricSurface, Plane, Sphere, Surface, Torus,
};
use geomcore::{
    Axis3D, CircleCircle2DIntersection, CircleCircle3DIntersection, CircleSurfaceIntersection,
    ConeConeIntersection, ConeCylinderIntersection, ConicConicIntersection,
    ConicSurfaceIntersection, CylinderCylinderIntersection, Frame3D, LineCircle2DIntersection,
    LineCircle3DIntersection, LineConic3DIntersection, LineConicHit, LineLine2DIntersection,
    LineLine3DIntersection, LinePlaneIntersection, LineQuadricIntersection, LineTorusHit,
    LineTorusIntersection, PlaneConeIntersection, PlaneCylinderIntersection,
    PlanePlaneIntersection, PlaneSphereIntersection, Point2D, Point3D, SphereConeIntersection,
    SphereCylinderIntersection, SphereSphereIntersection, Tolerance, TorusConeIntersection,
    TorusCylinderIntersection, TorusPlaneIntersection, TorusSphereIntersection,
    TorusTorusIntersection, Transform, Vector2D, Vector3D,
};
use geomcore::{
    Bound3D, BoundingBox3D, Discretization2D, Discretization3D, DiscretizationCriteria,
    Discretize2D, Discretize3D, curve_curve_extrema, intersect_curve_curve,
};

fn val_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Checks a curve derivative order lies in `1..=max`.
fn check_curve_order(order: u32, max: u32) -> PyResult<()> {
    if (1..=max).contains(&order) {
        Ok(())
    } else if order == 0 {
        Err(PyValueError::new_err(
            "order must be >= 1 (use eval_point for order 0)",
        ))
    } else {
        Err(PyValueError::new_err(format!(
            "order {order} is not supported (at most {max})"
        )))
    }
}

/// Checks a surface derivative order satisfies `1 <= du + dv <= 2`.
fn check_surface_order(du: u32, dv: u32) -> PyResult<()> {
    match du.checked_add(dv) {
        Some(1 | 2) => Ok(()),
        Some(0) => Err(PyValueError::new_err(
            "du + dv must be >= 1 (use eval_point for the (0, 0) order)",
        )),
        _ => Err(PyValueError::new_err(format!(
            "order du={du}, dv={dv} is not supported (du + dv must be <= 2)"
        ))),
    }
}

fn parse_parametrization(s: &str) -> PyResult<PointParametrization> {
    match s {
        "chordal" => Ok(PointParametrization::Chordal),
        "centripetal" => Ok(PointParametrization::Centripetal),
        _ => Err(PyValueError::new_err(
            "parametrization must be 'chordal' or 'centripetal'",
        )),
    }
}

/// Warns that a per-pair `intersect_*` method is deprecated in favour of
/// `intersect`.
fn warn_deprecated(py: Python<'_>, method: &str) -> PyResult<()> {
    let message = CString::new(format!(
        "{method} is deprecated since 0.29.0; use intersect"
    ))?;
    PyErr::warn(py, &py.get_type::<PyDeprecationWarning>(), &message, 1)
}

/// Body of an `intersect` method: runs the first arm whose class `other`
/// is an instance of, and raises `TypeError` when none matches.
macro_rules! dispatch_intersect {
    ($other:ident, $receiver:literal, $supported:literal, {
        $($class:ty => |$o:ident| $call:expr),+ $(,)?
    }) => {{
        $(
            if let Ok($o) = $other.extract::<PyRef<'_, $class>>() {
                return $call;
            }
        )+
        Err(PyTypeError::new_err(format!(
            concat!($receiver, ".intersect supports ", $supported, ", not {}"),
            $other.get_type().name()?
        )))
    }};
}

// ---------------------------------------------------------------------------
// Root value types
// ---------------------------------------------------------------------------

/// A point in 3D space with `x`, `y` and `z` coordinates.
#[pyclass(name = "Point3D", module = "geomcore", eq)]
#[derive(Clone, PartialEq)]
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

    /// Returns whether `other` is within `tol.confusion` of this point
    /// (`Tolerance.default()` when omitted). `==` compares coordinates
    /// exactly instead.
    #[pyo3(signature = (other, tol = None))]
    fn coincides(&self, other: &PyPoint3D, tol: Option<PyTolerance>) -> bool {
        self.0
            .coincides(other.0, tol.map(|t| t.0).unwrap_or_default())
    }

    fn __repr__(&self) -> String {
        format!("Point3D({}, {}, {})", self.0.x, self.0.y, self.0.z)
    }
}

/// A vector in 3D space.
///
/// Components are read with `components()`; the unit vectors along the world
/// axes are available as the static methods `x()`, `y()` and `z()`.
#[pyclass(name = "Vector3D", module = "geomcore", eq)]
#[derive(Clone, PartialEq)]
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
#[pyclass(name = "Point2D", module = "geomcore", eq)]
#[derive(Clone, PartialEq)]
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

    /// Returns whether `other` is within `tol.confusion` of this point
    /// (`Tolerance.default()` when omitted). `==` compares coordinates
    /// exactly instead.
    #[pyo3(signature = (other, tol = None))]
    fn coincides(&self, other: &PyPoint2D, tol: Option<PyTolerance>) -> bool {
        self.0
            .coincides(other.0, tol.map(|t| t.0).unwrap_or_default())
    }

    fn __repr__(&self) -> String {
        format!("Point2D({}, {})", self.0.x, self.0.y)
    }
}

/// A vector in 2D space.
#[pyclass(name = "Vector2D", module = "geomcore", eq)]
#[derive(Clone, PartialEq)]
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
#[pyclass(name = "Axis3D", module = "geomcore", eq)]
#[derive(Clone, PartialEq)]
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
#[pyclass(name = "Frame3D", module = "geomcore", eq)]
#[derive(Clone, PartialEq)]
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
#[pyclass(name = "Tolerance", module = "geomcore", eq)]
#[derive(Clone, PartialEq)]
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

    /// The side of the cube, centred on the origin, that models are
    /// assumed to fit in (model units); the default angular tolerance is
    /// the default confusion over it.
    #[classattr]
    #[allow(non_snake_case)]
    fn MODEL_SIZE() -> f64 {
        Tolerance::MODEL_SIZE
    }

    /// The angle (radians) below which two directions are parallel for
    /// geometry reaching `extent` from the pivot of the turn: turning by it
    /// moves that geometry by at most `confusion`. Never below `angular`.
    fn angular_for_extent(&self, extent: f64) -> f64 {
        self.0.angular_for_extent(extent)
    }

    fn __repr__(&self) -> String {
        format!(
            "Tolerance(confusion={}, angular={}, parametric={})",
            self.0.confusion, self.0.angular, self.0.parametric
        )
    }
}

/// A rigid (or mirrored/scaled) affine transformation.
#[pyclass(name = "Transform", module = "geomcore", eq)]
#[derive(Clone, PartialEq)]
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

/// The axis-aligned box around a curve or surface, holding the shape but
/// possibly loose for B-splines (their control points' box); `None`
/// wherever `bounding_box` reports no box for an unbounded shape.
#[pyclass(name = "BoundingBox3D", module = "geomcore", eq)]
#[derive(Clone, Copy, PartialEq)]
struct PyBoundingBox3D(BoundingBox3D);

#[pymethods]
impl PyBoundingBox3D {
    #[new]
    /// The box with opposite corners `a` and `b`, in any order.
    fn py_new(a: PyPoint3D, b: PyPoint3D) -> Self {
        PyBoundingBox3D(BoundingBox3D::new(a.0, b.0))
    }

    /// The smallest box holding every point of `points` (`ValueError`
    /// when there are none).
    #[staticmethod]
    fn from_points(points: Vec<PyPoint3D>) -> PyResult<Self> {
        let points = points.into_iter().map(|p| p.0);
        BoundingBox3D::from_points(points)
            .map(PyBoundingBox3D)
            .ok_or_else(|| PyValueError::new_err("from_points needs at least one point"))
    }

    /// The corner with the smallest coordinates.
    fn min(&self) -> PyPoint3D {
        PyPoint3D(self.0.min())
    }

    /// The corner with the largest coordinates.
    fn max(&self) -> PyPoint3D {
        PyPoint3D(self.0.max())
    }

    /// The middle of the box.
    fn center(&self) -> PyPoint3D {
        PyPoint3D(self.0.center())
    }

    /// The vector from the `min` corner to the `max` corner.
    fn diagonal(&self) -> PyVector3D {
        PyVector3D(self.0.diagonal())
    }

    /// The smallest box holding both boxes.
    fn union(&self, other: &PyBoundingBox3D) -> Self {
        PyBoundingBox3D(self.0.union(other.0))
    }

    /// The box grown by `distance` on every side (negative shrinks it).
    fn enlarged(&self, distance: f64) -> Self {
        PyBoundingBox3D(self.0.enlarged(distance))
    }

    /// Whether the box holds `point` on every axis.
    fn encloses(&self, point: PyPoint3D) -> bool {
        self.0.encloses(point.0)
    }

    /// Whether the boxes touch or overlap on every axis.
    fn overlaps(&self, other: &PyBoundingBox3D) -> bool {
        self.0.overlaps(other.0)
    }

    fn __repr__(&self) -> String {
        let (lo, hi) = (self.0.min(), self.0.max());
        format!(
            "BoundingBox3D(min=({}, {}, {}), max=({}, {}, {}))",
            lo.x, lo.y, lo.z, hi.x, hi.y, hi.z
        )
    }
}

// ---------------------------------------------------------------------------
// Surfaces
// ---------------------------------------------------------------------------

/// An infinite plane, parametrized by in-plane coordinates (u, v).
#[pyclass(name = "Plane", module = "geomcore.surfaces", eq)]
#[derive(Clone, PartialEq)]
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
        let tol = tol.map(|t| t.0).unwrap_or_default();
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
        self.plane_result(py, other, tol.map(|t| t.0).unwrap_or_default())
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
        self.sphere_result(py, sphere, tol.map(|t| t.0).unwrap_or_default())
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
        self.cylinder_result(py, cylinder, tol.map(|t| t.0).unwrap_or_default())
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
        self.cone_result(py, cone, tol.map(|t| t.0).unwrap_or_default())
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyPlane {
    fn plane_result(
        &self,
        py: Python<'_>,
        other: &PyPlane,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&other.0, tol) {
            PlanePlaneIntersection::Line(l) => Ok((
                "line".to_string(),
                PyLine3D(l).into_pyobject(py)?.into_any().unbind(),
            )),
            PlanePlaneIntersection::Parallel => Ok(("parallel".to_string(), py.None())),
            PlanePlaneIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
        }
    }

    fn sphere_result(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&sphere.0, tol) {
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

    fn cylinder_result(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&cylinder.0, tol) {
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

    fn cone_result(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&cone.0, tol) {
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
        let tol = tol.map(|t| t.0).unwrap_or_default();
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
        self.cylinder_result(py, other, tol.map(|t| t.0).unwrap_or_default())
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyCylinder {
    fn cylinder_result(
        &self,
        py: Python<'_>,
        other: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&other.0, tol) {
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
        let tol = tol.map(|t| t.0).unwrap_or_default();
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
        self.cylinder_result(py, cylinder, tol.map(|t| t.0).unwrap_or_default())
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
        self.cone_result(py, other, tol.map(|t| t.0).unwrap_or_default())
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyCone {
    fn cylinder_result(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&cylinder.0, tol) {
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

    fn cone_result(
        &self,
        py: Python<'_>,
        other: &PyCone,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&other.0, tol) {
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
        let tol = tol.map(|t| t.0).unwrap_or_default();
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
        self.sphere_result(py, other, tol.map(|t| t.0).unwrap_or_default())
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
        self.cylinder_result(py, cylinder, tol.map(|t| t.0).unwrap_or_default())
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
        self.cone_result(py, cone, tol.map(|t| t.0).unwrap_or_default())
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PySphere {
    fn sphere_result(
        &self,
        py: Python<'_>,
        other: &PySphere,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&other.0, tol) {
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

    fn cylinder_result(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&cylinder.0, tol) {
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

    fn cone_result(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&cone.0, tol) {
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
        let tol = tol.map(|t| t.0).unwrap_or_default();
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
        self.plane_result(py, plane, tol.map(|t| t.0).unwrap_or_default())
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
        self.sphere_result(py, sphere, tol.map(|t| t.0).unwrap_or_default())
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
        self.cylinder_result(py, cylinder, tol.map(|t| t.0).unwrap_or_default())
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
        self.cone_result(py, cone, tol.map(|t| t.0).unwrap_or_default())
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
        self.torus_result(py, other, tol.map(|t| t.0).unwrap_or_default())
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyTorus {
    fn plane_result(
        &self,
        py: Python<'_>,
        plane: &PyPlane,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&plane.0, tol) {
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

    fn sphere_result(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&sphere.0, tol) {
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

    fn cylinder_result(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&cylinder.0, tol) {
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

    fn cone_result(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&cone.0, tol) {
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

    fn torus_result(
        &self,
        py: Python<'_>,
        other: &PyTorus,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&other.0, tol) {
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
        self.0
            .contains(point.0, tol.map(|t| t.0).unwrap_or_default())
    }

    /// The local minima of the distance from `point` to the surface,
    /// as `(u, v, distance)` tuples ordered by ascending distance.
    #[pyo3(signature = (point, tol = None))]
    fn extrema(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> Vec<(f64, f64, f64)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        self.0
            .extrema(point.0, tol)
            .iter()
            .map(|p| (p.u, p.v, p.distance))
            .collect()
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

fn extract_curve3d(obj: &Bound<'_, PyAny>) -> PyResult<Curve3D> {
    if let Ok(c) = obj.extract::<PyRef<'_, PyLine3D>>() {
        return Ok(Curve3D::from(c.0));
    }
    if let Ok(c) = obj.extract::<PyRef<'_, PyCircle3D>>() {
        return Ok(Curve3D::from(c.0));
    }
    if let Ok(c) = obj.extract::<PyRef<'_, PyEllipse3D>>() {
        return Ok(Curve3D::from(c.0));
    }
    if let Ok(c) = obj.extract::<PyRef<'_, PyParabola3D>>() {
        return Ok(Curve3D::from(c.0));
    }
    if let Ok(c) = obj.extract::<PyRef<'_, PyHyperbola3D>>() {
        return Ok(Curve3D::from(c.0));
    }
    if let Ok(c) = obj.extract::<PyRef<'_, PyBSplineCurve3D>>() {
        // BSplineCurve3D owns heap data (not Copy): clone is required here.
        #[allow(clippy::clone_on_copy)]
        return Ok(Curve3D::from(c.0.clone()));
    }
    Err(PyTypeError::new_err(
        "expected a geomcore 3D curve (Line3D, Circle3D, Ellipse3D, Parabola3D, Hyperbola3D or BSplineCurve3D)",
    ))
}

/// A 3D curve passed to a generic function: a whole curve, or a piece of
/// one, whose range bounds the search.
enum CurveArg {
    Whole(Curve3D),
    Trimmed(TrimmedCurve3D),
}

impl ParametricCurve3D for CurveArg {
    fn eval_point(&self, u: f64) -> Point3D {
        match self {
            CurveArg::Whole(c) => c.eval_point(u),
            CurveArg::Trimmed(c) => c.eval_point(u),
        }
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        match self {
            CurveArg::Whole(c) => c.eval_derivative(u, order),
            CurveArg::Trimmed(c) => c.eval_derivative(u, order),
        }
    }

    fn bounds(&self) -> (f64, f64) {
        match self {
            CurveArg::Whole(c) => c.bounds(),
            CurveArg::Trimmed(c) => c.bounds(),
        }
    }

    fn period(&self) -> Option<f64> {
        match self {
            CurveArg::Whole(c) => c.period(),
            CurveArg::Trimmed(c) => c.period(),
        }
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        match self {
            CurveArg::Whole(c) => c.eval_points(us),
            CurveArg::Trimmed(c) => c.eval_points(us),
        }
    }
}

fn extract_curve_arg(obj: &Bound<'_, PyAny>) -> PyResult<CurveArg> {
    if let Ok(c) = obj.extract::<PyRef<'_, PyTrimmedCurve3D>>() {
        return Ok(CurveArg::Trimmed(c.0.clone()));
    }
    extract_curve3d(obj).map(CurveArg::Whole).map_err(|_| {
        PyTypeError::new_err(
            "expected a geomcore 3D curve (Line3D, Circle3D, Ellipse3D, Parabola3D, Hyperbola3D, BSplineCurve3D or TrimmedCurve3D)",
        )
    })
}

fn curve3d_to_py(py: Python<'_>, curve: Curve3D) -> PyResult<Py<PyAny>> {
    match curve {
        Curve3D::Line(c) => Ok(PyLine3D(c).into_pyobject(py)?.into_any().unbind()),
        Curve3D::Circle(c) => Ok(PyCircle3D(c).into_pyobject(py)?.into_any().unbind()),
        Curve3D::Ellipse(c) => Ok(PyEllipse3D(c).into_pyobject(py)?.into_any().unbind()),
        Curve3D::Parabola(c) => Ok(PyParabola3D(c).into_pyobject(py)?.into_any().unbind()),
        Curve3D::Hyperbola(c) => Ok(PyHyperbola3D(c).into_pyobject(py)?.into_any().unbind()),
        Curve3D::BSpline(c) => Ok(PyBSplineCurve3D(c).into_pyobject(py)?.into_any().unbind()),
        _ => Err(PyValueError::new_err("unsupported 3d curve variant")),
    }
}

fn extract_curve2d(obj: &Bound<'_, PyAny>) -> PyResult<Curve2D> {
    if let Ok(c) = obj.extract::<PyRef<'_, PyLine2D>>() {
        return Ok(Curve2D::from(c.0));
    }
    if let Ok(c) = obj.extract::<PyRef<'_, PyCircle2D>>() {
        return Ok(Curve2D::from(c.0));
    }
    Err(PyTypeError::new_err(
        "expected a geomcore 2D curve (Line2D or Circle2D)",
    ))
}

/// A generic intersection hit: `(curve parameter, (u, v), point)`.
type CurveHitPy = (f64, (f64, f64), Py<PyAny>);

/// Closest-point enumeration between any two 3D curves.
///
/// Every local minimum of the distance over the curves' parameter
/// domains, including minima at the ends of bounded curves, as
/// `(first_param, second_param, distance)` tuples ordered by distance.
/// Parallel lines, concentric circles and other continuous families of
/// equal minima report the distance once, with a representative pair.
/// Use the analytic methods when classification matters.
#[pyfunction]
#[pyo3(name = "curve_curve_extrema", signature = (curve_a, curve_b, tol = None))]
fn py_curve_curve_extrema(
    curve_a: &Bound<'_, PyAny>,
    curve_b: &Bound<'_, PyAny>,
    tol: Option<PyTolerance>,
) -> PyResult<Vec<(f64, f64, f64)>> {
    let tol = tol.map(|t| t.0).unwrap_or_default();
    let a = extract_curve_arg(curve_a)?;
    let b = extract_curve_arg(curve_b)?;
    Ok(curve_curve_extrema(&a, &b, tol)
        .iter()
        .map(|e| (e.first_param, e.second_param, e.distance))
        .collect())
}

/// Generic 3D curve-curve intersection: extrema within tolerance.
///
/// Returns `(first_param, Point3D, second_param)` tuples ordered by
/// first-curve parameter.
#[pyfunction]
#[pyo3(name = "intersect_curve_curve", signature = (curve_a, curve_b, tol = None))]
fn py_intersect_curve_curve(
    py: Python<'_>,
    curve_a: &Bound<'_, PyAny>,
    curve_b: &Bound<'_, PyAny>,
    tol: Option<PyTolerance>,
) -> PyResult<Vec<(f64, Py<PyAny>, f64)>> {
    let tol = tol.map(|t| t.0).unwrap_or_default();
    let a = extract_curve_arg(curve_a)?;
    let b = extract_curve_arg(curve_b)?;
    let mut out = Vec::new();
    for h in intersect_curve_curve(&a, &b, tol) {
        out.push((
            h.first_param,
            PyPoint3D(h.point).into_pyobject(py)?.into_any().unbind(),
            h.second_param,
        ));
    }
    Ok(out)
}
/// Generic curve-surface intersection for any curve and surface.
///
/// Samples the curve, projects onto the surface for seeds, and refines
/// with Newton on `C(t) - S(u, v) = 0`. Returns a
/// `(curve_param, (u, v), Point3D)` tuple per transversal hit, ordered
/// by curve parameter. Overlapping pairs yield dense hit sets; use the
/// analytic `intersect_*` methods when classification matters.
#[pyfunction]
#[pyo3(name = "intersect_curve_surface", signature = (curve, surface, tol = None))]
fn py_intersect_curve_surface(
    py: Python<'_>,
    curve: &Bound<'_, PyAny>,
    surface: &Bound<'_, PyAny>,
    tol: Option<PyTolerance>,
) -> PyResult<Vec<CurveHitPy>> {
    let tol = tol.map(|t| t.0).unwrap_or_default();
    let c = extract_curve_arg(curve)?;
    let s = extract_surface(surface)?;
    let mut out = Vec::new();
    for h in intersect_curve_surface(&c, &s, tol) {
        out.push((
            h.curve_param,
            h.surface_params,
            PyPoint3D(h.point).into_pyobject(py)?.into_any().unbind(),
        ));
    }
    Ok(out)
}

/// Numeric curve-on-surface parametrization for pairs without closed form.
///
/// Samples the curve uniformly (`n` points) and inverts each sample onto
/// the surface. Returns `(u, v)` tuples; surface evaluation at them
/// recovers the samples where the curve lies on the surface.
#[pyfunction]
#[pyo3(name = "parametrize_numeric", signature = (curve, surface, n = 64, tol = None))]
fn py_parametrize_numeric(
    curve: &Bound<'_, PyAny>,
    surface: &Bound<'_, PyAny>,
    n: usize,
    tol: Option<PyTolerance>,
) -> PyResult<Vec<(f64, f64)>> {
    let tol = tol.map(|t| t.0).unwrap_or_default();
    let c = extract_curve_arg(curve)?;
    let s = extract_surface(surface)?;
    Ok(parametrize_numeric(&c, &s, n, tol)
        .iter()
        .map(|p| (p.x, p.y))
        .collect())
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

fn conic_surface_to_py(
    py: Python<'_>,
    hit: ConicSurfaceIntersection,
) -> PyResult<(String, Py<PyAny>)> {
    match hit {
        ConicSurfaceIntersection::Hits(hits) => {
            let mut out = Vec::with_capacity(hits.len());
            for h in hits {
                out.push((
                    h.parameter,
                    PyPoint3D(h.point).into_pyobject(py)?.into_any().unbind(),
                    h.multiplicity,
                ));
            }
            Ok((
                "hits".to_string(),
                out.into_pyobject(py)?.into_any().unbind(),
            ))
        }
        ConicSurfaceIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
        ConicSurfaceIntersection::Empty => Ok(("empty".to_string(), py.None())),
    }
}

fn line_conic_hit(py: Python<'_>, h: &LineConicHit) -> PyResult<Py<PyAny>> {
    (
        h.line_param,
        PyPoint3D(h.point).into_pyobject(py)?.into_any().unbind(),
        h.conic_param,
    )
        .into_pyobject(py)
        .map(|o| o.into_any().unbind())
}

fn line_conic_to_py(py: Python<'_>, hit: LineConic3DIntersection) -> PyResult<(String, Py<PyAny>)> {
    match hit {
        LineConic3DIntersection::Points(h1, h2) => Ok((
            "points".to_string(),
            (line_conic_hit(py, &h1)?, line_conic_hit(py, &h2)?)
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        )),
        LineConic3DIntersection::Tangent(h) => Ok(("tangent".to_string(), line_conic_hit(py, &h)?)),
        LineConic3DIntersection::Empty => Ok(("empty".to_string(), py.None())),
    }
}

fn conic_conic_to_py(py: Python<'_>, hit: ConicConicIntersection) -> PyResult<(String, Py<PyAny>)> {
    match hit {
        ConicConicIntersection::Hits(hits) => {
            let mut out = Vec::with_capacity(hits.len());
            for h in &hits {
                out.push(
                    (
                        h.first_param,
                        PyPoint3D(h.point).into_pyobject(py)?.into_any().unbind(),
                        h.second_param,
                        h.multiplicity,
                    )
                        .into_pyobject(py)?
                        .into_any()
                        .unbind(),
                );
            }
            Ok((
                "hits".to_string(),
                out.into_pyobject(py)?.into_any().unbind(),
            ))
        }
        ConicConicIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
        ConicConicIntersection::Empty => Ok(("empty".to_string(), py.None())),
    }
}

fn line_torus_hit(py: Python<'_>, h: &LineTorusHit) -> PyResult<Py<PyAny>> {
    (
        h.line_param,
        PyPoint3D(h.point).into_pyobject(py)?.into_any().unbind(),
        h.surface_params,
        h.multiplicity,
    )
        .into_pyobject(py)
        .map(|o| o.into_any().unbind())
}

fn line_torus_to_py(py: Python<'_>, hit: LineTorusIntersection) -> PyResult<(String, Py<PyAny>)> {
    match hit {
        LineTorusIntersection::Hits(hits) => {
            let mut out = Vec::with_capacity(hits.len());
            for h in &hits {
                out.push(line_torus_hit(py, h)?);
            }
            Ok((
                "hits".to_string(),
                out.into_pyobject(py)?.into_any().unbind(),
            ))
        }
        LineTorusIntersection::Empty => Ok(("empty".to_string(), py.None())),
    }
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

fn point2(py: Python<'_>, t: f64, p: Point2D) -> PyResult<Py<PyAny>> {
    (t, PyPoint2D(p).into_pyobject(py)?.into_any().unbind())
        .into_pyobject(py)
        .map(|o| o.into_any().unbind())
}

fn circle_surface_to_py(
    py: Python<'_>,
    hit: CircleSurfaceIntersection,
) -> PyResult<(String, Py<PyAny>)> {
    match hit {
        CircleSurfaceIntersection::Circle(c) => Ok((
            "circle".to_string(),
            PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
        )),
        CircleSurfaceIntersection::Hits(hits) => {
            let mut out = Vec::with_capacity(hits.len());
            for h in hits {
                out.push((
                    h.parameter,
                    PyPoint3D(h.point).into_pyobject(py)?.into_any().unbind(),
                    h.multiplicity,
                ));
            }
            Ok((
                "hits".to_string(),
                out.into_pyobject(py)?.into_any().unbind(),
            ))
        }
        CircleSurfaceIntersection::Empty => Ok(("empty".to_string(), py.None())),
    }
}

/// Marching surface-surface intersection of any two surfaces.
///
/// Traces the meeting curves numerically and fits B-spline approximants.
/// Returns one `BSplineCurve3D` per traced loop or arc (empty when the
/// surfaces miss).
#[pyfunction]
#[pyo3(name = "intersect_marching", signature = (surface_a, surface_b, tol = None))]
fn py_intersect_marching(
    py: Python<'_>,
    surface_a: &Bound<'_, PyAny>,
    surface_b: &Bound<'_, PyAny>,
    tol: Option<PyTolerance>,
) -> PyResult<Vec<Py<PyAny>>> {
    let tol = tol.map(|t| t.0).unwrap_or_default();
    let a = extract_surface(surface_a)?;
    let b = extract_surface(surface_b)?;
    let mut out = Vec::new();
    for c in marching_intersection(&a, &b, tol) {
        out.push(PyBSplineCurve3D(c).into_pyobject(py)?.into_any().unbind());
    }
    Ok(out)
}

/// A discretized 3D curve: points on the curve in order of parameter,
/// joined by straight segments. `parameters[i]` is the parameter of
/// `points[i]`; segment `i` joins `points[i]` and `points[i + 1]` and
/// carries a proven upper bound on its chord deviation (the largest
/// distance from the curve to the segment) and angle deviation (the
/// largest angle between the curve's tangent and the segment, π where no
/// bound exists).
#[pyclass(name = "Discretization3D", module = "geomcore", frozen, eq)]
#[derive(PartialEq)]
struct PyDiscretization3D(Discretization3D);

#[pymethods]
impl PyDiscretization3D {
    /// The parameter of each point, increasing.
    #[getter]
    fn parameters(&self) -> Vec<f64> {
        self.0.parameters.clone()
    }

    /// The points on the curve.
    #[getter]
    fn points(&self) -> Vec<PyPoint3D> {
        self.0.points.iter().copied().map(PyPoint3D).collect()
    }

    /// The chord deviation of each segment.
    #[getter]
    fn chord_deviations(&self) -> Vec<f64> {
        self.0.chord_deviations.clone()
    }

    /// The angle deviation of each segment, in radians.
    #[getter]
    fn angle_deviations(&self) -> Vec<f64> {
        self.0.angle_deviations.clone()
    }

    fn __repr__(&self) -> String {
        format!("Discretization3D({} points)", self.0.points.len())
    }
}

/// A discretized 2D curve, such as a curve in a surface's parameter
/// plane: the 2D counterpart of `Discretization3D`, its chord deviations
/// being distances in the plane.
#[pyclass(name = "Discretization2D", module = "geomcore", frozen, eq)]
#[derive(PartialEq)]
struct PyDiscretization2D(Discretization2D);

#[pymethods]
impl PyDiscretization2D {
    /// The parameter of each point, increasing.
    #[getter]
    fn parameters(&self) -> Vec<f64> {
        self.0.parameters.clone()
    }

    /// The points on the curve.
    #[getter]
    fn points(&self) -> Vec<PyPoint2D> {
        self.0.points.iter().copied().map(PyPoint2D).collect()
    }

    /// The chord deviation of each segment.
    #[getter]
    fn chord_deviations(&self) -> Vec<f64> {
        self.0.chord_deviations.clone()
    }

    /// The angle deviation of each segment, in radians.
    #[getter]
    fn angle_deviations(&self) -> Vec<f64> {
        self.0.angle_deviations.clone()
    }

    fn __repr__(&self) -> String {
        format!("Discretization2D({} points)", self.0.points.len())
    }
}

/// The criteria from the keyword arguments of `discretize`.
fn discretization_criteria(
    max_chord_deviation: f64,
    max_angle_deviation: Option<f64>,
) -> PyResult<DiscretizationCriteria> {
    let criteria = DiscretizationCriteria::new(max_chord_deviation).map_err(val_err)?;
    match max_angle_deviation {
        Some(angle) => criteria.with_max_angle_deviation(angle).map_err(val_err),
        None => Ok(criteria),
    }
}

fn discretize_3d(
    curve: &impl Discretize3D,
    max_chord_deviation: f64,
    max_angle_deviation: Option<f64>,
) -> PyResult<PyDiscretization3D> {
    let criteria = discretization_criteria(max_chord_deviation, max_angle_deviation)?;
    Ok(PyDiscretization3D(curve.discretize(criteria)))
}

fn discretize_2d(
    curve: &impl Discretize2D,
    max_chord_deviation: f64,
    max_angle_deviation: Option<f64>,
) -> PyResult<PyDiscretization2D> {
    let criteria = discretization_criteria(max_chord_deviation, max_angle_deviation)?;
    Ok(PyDiscretization2D(curve.discretize(criteria)))
}

// ---------------------------------------------------------------------------
// Curves
// ---------------------------------------------------------------------------

/// An infinite straight line in 3D, parametrized by arc length from its origin.
#[pyclass(name = "Line3D", module = "geomcore.curves", eq)]
#[derive(Clone, PartialEq)]
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
        let tol = tol.map(|t| t.0).unwrap_or_default();
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
        self.plane_result(py, plane, tol.map(|t| t.0).unwrap_or_default())
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
        self.sphere_result(py, sphere, tol.map(|t| t.0).unwrap_or_default())
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
        self.cylinder_result(py, cylinder, tol.map(|t| t.0).unwrap_or_default())
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
        self.cone_result(py, cone, tol.map(|t| t.0).unwrap_or_default())
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
        self.line_result(py, other, tol.map(|t| t.0).unwrap_or_default())
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
        self.circle_result(py, circle, tol.map(|t| t.0).unwrap_or_default())
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
        self.ellipse_result(py, ellipse, tol.map(|t| t.0).unwrap_or_default())
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
        self.parabola_result(py, parabola, tol.map(|t| t.0).unwrap_or_default())
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
        self.hyperbola_result(py, hyperbola, tol.map(|t| t.0).unwrap_or_default())
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
        self.torus_result(py, torus, tol.map(|t| t.0).unwrap_or_default())
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
        let tol = tol.map(|t| t.0).unwrap_or_default();
        curve2d_to_py(py, self.0.parametrize_on(s, tol).map_err(val_err)?)
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyLine3D {
    fn plane_result(
        &self,
        py: Python<'_>,
        plane: &PyPlane,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&plane.0, tol) {
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

    fn sphere_result(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        quadric_hit_to_py(py, self.0.intersect(&sphere.0, tol))
    }

    fn cylinder_result(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        quadric_hit_to_py(py, self.0.intersect(&cylinder.0, tol))
    }

    fn cone_result(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        quadric_hit_to_py(py, self.0.intersect(&cone.0, tol))
    }

    fn line_result(
        &self,
        py: Python<'_>,
        other: &PyLine3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&other.0, tol) {
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

    fn circle_result(
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
        match self.0.intersect(&circle.0, tol) {
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

    fn ellipse_result(
        &self,
        py: Python<'_>,
        ellipse: &PyEllipse3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        line_conic_to_py(py, self.0.intersect(&ellipse.0, tol))
    }

    fn parabola_result(
        &self,
        py: Python<'_>,
        parabola: &PyParabola3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        line_conic_to_py(py, self.0.intersect(&parabola.0, tol))
    }

    fn hyperbola_result(
        &self,
        py: Python<'_>,
        hyperbola: &PyHyperbola3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        line_conic_to_py(py, self.0.intersect(&hyperbola.0, tol))
    }

    fn torus_result(
        &self,
        py: Python<'_>,
        torus: &PyTorus,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        line_torus_to_py(py, self.0.intersect(&torus.0, tol))
    }
}

/// A circle in 3D, parametrized by the angle from its frame's x direction.
#[pyclass(name = "Circle3D", module = "geomcore.curves", eq)]
#[derive(Clone, PartialEq)]
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
        let tol = tol.map(|t| t.0).unwrap_or_default();
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
        let tol = tol.map(|t| t.0).unwrap_or_default();
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
        self.circle_result(py, other, tol.map(|t| t.0).unwrap_or_default())
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
        self.ellipse_result(py, ellipse, tol.map(|t| t.0).unwrap_or_default())
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
        self.parabola_result(py, parabola, tol.map(|t| t.0).unwrap_or_default())
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
        self.hyperbola_result(py, hyperbola, tol.map(|t| t.0).unwrap_or_default())
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
        self.plane_result(py, plane, tol.map(|t| t.0).unwrap_or_default())
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
        self.sphere_result(py, sphere, tol.map(|t| t.0).unwrap_or_default())
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
        self.cylinder_result(py, cylinder, tol.map(|t| t.0).unwrap_or_default())
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
        self.cone_result(py, cone, tol.map(|t| t.0).unwrap_or_default())
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyCircle3D {
    fn circle_result(
        &self,
        py: Python<'_>,
        other: &PyCircle3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&other.0, tol) {
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

    fn ellipse_result(
        &self,
        py: Python<'_>,
        ellipse: &PyEllipse3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_conic_to_py(py, self.0.intersect(&ellipse.0, tol))
    }

    fn parabola_result(
        &self,
        py: Python<'_>,
        parabola: &PyParabola3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_conic_to_py(py, self.0.intersect(&parabola.0, tol))
    }

    fn hyperbola_result(
        &self,
        py: Python<'_>,
        hyperbola: &PyHyperbola3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_conic_to_py(py, self.0.intersect(&hyperbola.0, tol))
    }

    fn plane_result(
        &self,
        py: Python<'_>,
        plane: &PyPlane,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        circle_surface_to_py(py, self.0.intersect(&plane.0, tol))
    }

    fn sphere_result(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        circle_surface_to_py(py, self.0.intersect(&sphere.0, tol))
    }

    fn cylinder_result(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        circle_surface_to_py(py, self.0.intersect(&cylinder.0, tol))
    }

    fn cone_result(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        circle_surface_to_py(py, self.0.intersect(&cone.0, tol))
    }
}

/// An ellipse in 3D.
#[pyclass(name = "Ellipse3D", module = "geomcore.curves", eq)]
#[derive(Clone, PartialEq)]
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
        let tol = tol.map(|t| t.0).unwrap_or_default();
        dispatch_intersect!(other, "Ellipse3D", "Plane, Cylinder, Cone, Sphere, Line3D, Circle3D, Ellipse3D, Parabola3D or Hyperbola3D", {
            PyPlane => |o| self.plane_result(py, &o, tol),
            PyCylinder => |o| self.cylinder_result(py, &o, tol),
            PyCone => |o| self.cone_result(py, &o, tol),
            PySphere => |o| self.sphere_result(py, &o, tol),
            PyLine3D => |o| o.ellipse_result(py, self, tol),
            PyCircle3D => |o| conic_conic_to_py(py, self.0.intersect(&o.0, tol)),
            PyEllipse3D => |o| conic_conic_to_py(py, self.0.intersect(&o.0, tol)),
            PyParabola3D => |o| conic_conic_to_py(py, self.0.intersect(&o.0, tol)),
            PyHyperbola3D => |o| conic_conic_to_py(py, self.0.intersect(&o.0, tol)),
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
        self.plane_result(py, plane, tol.map(|t| t.0).unwrap_or_default())
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
        self.sphere_result(py, sphere, tol.map(|t| t.0).unwrap_or_default())
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
        self.cylinder_result(py, cylinder, tol.map(|t| t.0).unwrap_or_default())
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
        self.cone_result(py, cone, tol.map(|t| t.0).unwrap_or_default())
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
        self.ellipse_result(py, other, tol.map(|t| t.0).unwrap_or_default())
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
        let tol = tol.map(|t| t.0).unwrap_or_default();
        curve2d_to_py(py, self.0.parametrize_on(s, tol).map_err(val_err)?)
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyEllipse3D {
    fn plane_result(
        &self,
        py: Python<'_>,
        plane: &PyPlane,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect(&plane.0, tol))
    }

    fn sphere_result(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect(&sphere.0, tol))
    }

    fn cylinder_result(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect(&cylinder.0, tol))
    }

    fn cone_result(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect(&cone.0, tol))
    }

    fn ellipse_result(
        &self,
        py: Python<'_>,
        other: &PyEllipse3D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_conic_to_py(py, self.0.intersect(&other.0, tol))
    }
}

/// A parabola in 3D, parametrized so the apex is at parameter 0.
#[pyclass(name = "Parabola3D", module = "geomcore.curves", eq)]
#[derive(Clone, PartialEq)]
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
        let tol = tol.map(|t| t.0).unwrap_or_default();
        dispatch_intersect!(other, "Parabola3D", "Plane, Cylinder, Cone, Sphere, Line3D, Circle3D, Ellipse3D, Parabola3D or Hyperbola3D", {
            PyPlane => |o| self.plane_result(py, &o, tol),
            PyCylinder => |o| self.cylinder_result(py, &o, tol),
            PyCone => |o| self.cone_result(py, &o, tol),
            PySphere => |o| self.sphere_result(py, &o, tol),
            PyLine3D => |o| o.parabola_result(py, self, tol),
            PyCircle3D => |o| conic_conic_to_py(py, self.0.intersect(&o.0, tol)),
            PyEllipse3D => |o| conic_conic_to_py(py, self.0.intersect(&o.0, tol)),
            PyParabola3D => |o| conic_conic_to_py(py, self.0.intersect(&o.0, tol)),
            PyHyperbola3D => |o| conic_conic_to_py(py, self.0.intersect(&o.0, tol)),
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
        self.plane_result(py, plane, tol.map(|t| t.0).unwrap_or_default())
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
        self.sphere_result(py, sphere, tol.map(|t| t.0).unwrap_or_default())
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
        self.cylinder_result(py, cylinder, tol.map(|t| t.0).unwrap_or_default())
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
        self.cone_result(py, cone, tol.map(|t| t.0).unwrap_or_default())
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
        let tol = tol.map(|t| t.0).unwrap_or_default();
        curve2d_to_py(py, self.0.parametrize_on(s, tol).map_err(val_err)?)
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyParabola3D {
    fn plane_result(
        &self,
        py: Python<'_>,
        plane: &PyPlane,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect(&plane.0, tol))
    }

    fn sphere_result(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect(&sphere.0, tol))
    }

    fn cylinder_result(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect(&cylinder.0, tol))
    }

    fn cone_result(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect(&cone.0, tol))
    }
}

/// One branch of a hyperbola in 3D.
#[pyclass(name = "Hyperbola3D", module = "geomcore.curves", eq)]
#[derive(Clone, PartialEq)]
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
        let tol = tol.map(|t| t.0).unwrap_or_default();
        dispatch_intersect!(other, "Hyperbola3D", "Plane, Cylinder, Cone, Sphere, Line3D, Circle3D, Ellipse3D, Parabola3D or Hyperbola3D", {
            PyPlane => |o| self.plane_result(py, &o, tol),
            PyCylinder => |o| self.cylinder_result(py, &o, tol),
            PyCone => |o| self.cone_result(py, &o, tol),
            PySphere => |o| self.sphere_result(py, &o, tol),
            PyLine3D => |o| o.hyperbola_result(py, self, tol),
            PyCircle3D => |o| conic_conic_to_py(py, self.0.intersect(&o.0, tol)),
            PyEllipse3D => |o| conic_conic_to_py(py, self.0.intersect(&o.0, tol)),
            PyParabola3D => |o| conic_conic_to_py(py, self.0.intersect(&o.0, tol)),
            PyHyperbola3D => |o| conic_conic_to_py(py, self.0.intersect(&o.0, tol)),
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
        self.plane_result(py, plane, tol.map(|t| t.0).unwrap_or_default())
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
        self.sphere_result(py, sphere, tol.map(|t| t.0).unwrap_or_default())
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
        self.cylinder_result(py, cylinder, tol.map(|t| t.0).unwrap_or_default())
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
        self.cone_result(py, cone, tol.map(|t| t.0).unwrap_or_default())
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
        let tol = tol.map(|t| t.0).unwrap_or_default();
        curve2d_to_py(py, self.0.parametrize_on(s, tol).map_err(val_err)?)
    }
}

// Exact intersections behind `intersect` and the deprecated per-pair methods.
impl PyHyperbola3D {
    fn plane_result(
        &self,
        py: Python<'_>,
        plane: &PyPlane,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect(&plane.0, tol))
    }

    fn sphere_result(
        &self,
        py: Python<'_>,
        sphere: &PySphere,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect(&sphere.0, tol))
    }

    fn cylinder_result(
        &self,
        py: Python<'_>,
        cylinder: &PyCylinder,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect(&cylinder.0, tol))
    }

    fn cone_result(
        &self,
        py: Python<'_>,
        cone: &PyCone,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        conic_surface_to_py(py, self.0.intersect(&cone.0, tol))
    }
}

/// A B-spline (optionally rational, optionally periodic) curve in 3D.
#[pyclass(name = "BSplineCurve3D", module = "geomcore.curves", eq)]
#[derive(Clone, PartialEq)]
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
        self.0
            .contains(point.0, tol.map(|t| t.0).unwrap_or_default())
    }

    /// The local minima of the distance from `point` to the curve (plus
    /// the interval ends of an open curve), as `(parameter, distance)`
    /// tuples ordered by ascending distance.
    #[pyo3(signature = (point, tol = None))]
    fn extrema(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
        self.0
            .extrema(point.0, tol)
            .iter()
            .map(|p| (p.parameter, p.distance))
            .collect()
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
        let tol = tol.map(|t| t.0).unwrap_or_default();
        curve2d_to_py(py, self.0.parametrize_on(s, tol).map_err(val_err)?)
    }
}

/// A straight line in a surface's 2D (u, v) parameter space.
#[pyclass(name = "Line2D", module = "geomcore.curves", eq)]
#[derive(Clone, PartialEq)]
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
        let tol = tol.map(|t| t.0).unwrap_or_default();
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
        self.line_result(py, other, tol.map(|t| t.0).unwrap_or_default())
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
        self.circle_result(py, circle, tol.map(|t| t.0).unwrap_or_default())
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
    fn line_result(
        &self,
        py: Python<'_>,
        other: &PyLine2D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&other.0, tol) {
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

    fn circle_result(
        &self,
        py: Python<'_>,
        circle: &PyCircle2D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&circle.0, tol) {
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
        let tol = tol.map(|t| t.0).unwrap_or_default();
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
        self.circle_result(py, other, tol.map(|t| t.0).unwrap_or_default())
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
    fn circle_result(
        &self,
        py: Python<'_>,
        other: &PyCircle2D,
        tol: Tolerance,
    ) -> PyResult<(String, Py<PyAny>)> {
        match self.0.intersect(&other.0, tol) {
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
fn max_curve_order(curve: &Curve3D) -> u32 {
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
struct PyTrimmedCurve3D(TrimmedCurve3D);

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
        self.0
            .contains(point.0, tol.map(|t| t.0).unwrap_or_default())
    }

    /// The basis's distance extrema from `point` inside the range, plus
    /// the two ends, as `(parameter, distance)` tuples ordered by
    /// ascending distance.
    #[pyo3(signature = (point, tol = None))]
    fn extrema(&self, point: PyPoint3D, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
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
        let proj = self
            .0
            .project_point(point.0, tol.map(|t| t.0).unwrap_or_default());
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the trimmed curve, returning a
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
struct PyTrimmedCurve2D(TrimmedCurve2D);

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
        self.0
            .contains(point.0, tol.map(|t| t.0).unwrap_or_default())
    }

    /// The basis's distance extrema from `point` inside the range, plus
    /// the two ends, as `(parameter, distance)` tuples ordered by
    /// ascending distance.
    #[pyo3(signature = (point, tol = None))]
    fn extrema(&self, point: PyPoint2D, tol: Option<PyTolerance>) -> Vec<(f64, f64)> {
        let tol = tol.map(|t| t.0).unwrap_or_default();
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
        let proj = self
            .0
            .project_point(point.0, tol.map(|t| t.0).unwrap_or_default());
        (proj.parameter, proj.distance)
    }

    /// Projects each point in `points` onto the trimmed curve, returning a
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
    m.add_class::<PyBoundingBox3D>()?;
    m.add_class::<PyDiscretization2D>()?;
    m.add_class::<PyDiscretization3D>()?;

    let curves = PyModule::new(py, "curves")?;
    curves.add_class::<PyLine3D>()?;
    curves.add_class::<PyCircle3D>()?;
    curves.add_class::<PyEllipse3D>()?;
    curves.add_class::<PyParabola3D>()?;
    curves.add_class::<PyHyperbola3D>()?;
    curves.add_class::<PyBSplineCurve3D>()?;
    curves.add_class::<PyLine2D>()?;
    curves.add_class::<PyCircle2D>()?;
    curves.add_class::<PyTrimmedCurve3D>()?;
    curves.add_class::<PyTrimmedCurve2D>()?;
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

    m.add_function(wrap_pyfunction!(py_intersect_curve_surface, m)?)?;
    m.add_function(wrap_pyfunction!(py_intersect_marching, m)?)?;
    m.add_function(wrap_pyfunction!(py_parametrize_numeric, m)?)?;
    m.add_function(wrap_pyfunction!(py_curve_curve_extrema, m)?)?;
    m.add_function(wrap_pyfunction!(py_intersect_curve_curve, m)?)?;

    Ok(())
}
