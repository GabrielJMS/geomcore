use pyo3::prelude::*;

use geomcore::{
    Axis2D, Axis3D, Frame2D, Frame3D, Point2D, Point3D, Tolerance, Transform, Vector2D, Vector3D,
};
use geomcore::{
    BoundingBox3D, Discretization2D, Discretization3D, DiscretizationCriteria, Discretize2D,
    Discretize3D,
};

use crate::errors::{invalid_argument, python_error};
use crate::tolerance;

// ---------------------------------------------------------------------------
// Root value types
// ---------------------------------------------------------------------------

/// What `Point3D.__sub__` accepts: a point (giving the vector between the
/// two) or a vector (giving the moved point).
#[derive(FromPyObject)]
enum PointOrVector3D {
    Point(PyPoint3D),
    Vector(PyVector3D),
}

/// What `Point2D.__sub__` accepts: a point or a vector.
#[derive(FromPyObject)]
enum PointOrVector2D {
    Point(PyPoint2D),
    Vector(PyVector2D),
}

/// A point in 3D space with `x`, `y` and `z` coordinates.
#[pyclass(name = "Point3D", module = "geomcore", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyPoint3D(pub(crate) Point3D);

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
        self.0.coincides(other.0, tolerance(tol))
    }

    /// Whether every coordinate is finite.
    fn is_finite(&self) -> bool {
        self.0.is_finite()
    }

    fn __add__(&self, offset: PyVector3D) -> PyPoint3D {
        PyPoint3D(self.0 + offset.0)
    }

    fn __sub__(&self, other: PointOrVector3D, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match other {
            PointOrVector3D::Point(point) => Ok(PyVector3D(self.0 - point.0)
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            PointOrVector3D::Vector(vector) => Ok(PyPoint3D(self.0 - vector.0)
                .into_pyobject(py)?
                .into_any()
                .unbind()),
        }
    }

    fn __repr__(&self) -> String {
        format!("Point3D{}", coordinates(self.0))
    }
}

/// A vector in 3D space.
///
/// Components are read with the properties `x`, `y` and `z` or with
/// `components()`; the unit vectors along the world axes are the static
/// methods `unit_x()`, `unit_y()` and `unit_z()`. Vectors add, subtract,
/// negate and scale by a number with the usual operators.
#[pyclass(name = "Vector3D", module = "geomcore", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyVector3D(pub(crate) Vector3D);

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
    fn unit_x() -> Self {
        PyVector3D(Vector3D::X)
    }

    /// The unit vector along the world y axis.
    #[staticmethod]
    fn unit_y() -> Self {
        PyVector3D(Vector3D::Y)
    }

    /// The unit vector along the world z axis.
    #[staticmethod]
    fn unit_z() -> Self {
        PyVector3D(Vector3D::Z)
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

    /// The squared length of the vector.
    fn square_magnitude(&self) -> f64 {
        self.0.square_magnitude()
    }

    /// Whether every component is finite.
    fn is_finite(&self) -> bool {
        self.0.is_finite()
    }

    /// The unit vector with the same direction, or `None` for a vector too
    /// short to normalize.
    fn normalized(&self) -> Option<PyVector3D> {
        self.0.normalized().map(PyVector3D)
    }

    /// The signed angle from this vector to `other` around `reference`
    /// (right-hand rule), in (-pi, pi]. A zero `reference` gives the unsigned
    /// angle in [0, pi].
    fn angle_with_reference(&self, other: &PyVector3D, reference: &PyVector3D) -> f64 {
        self.0.angle_with_reference(other.0, reference.0)
    }

    fn __add__(&self, other: &PyVector3D) -> PyVector3D {
        PyVector3D(self.0 + other.0)
    }

    fn __sub__(&self, other: &PyVector3D) -> PyVector3D {
        PyVector3D(self.0 - other.0)
    }

    fn __neg__(&self) -> PyVector3D {
        PyVector3D(-self.0)
    }

    fn __mul__(&self, scalar: f64) -> PyVector3D {
        PyVector3D(self.0 * scalar)
    }

    fn __rmul__(&self, scalar: f64) -> PyVector3D {
        PyVector3D(scalar * self.0)
    }

    fn __repr__(&self) -> String {
        format!("Vector3D{}", coordinates(self.0))
    }
}

/// A point in 2D space (used by curves living in a surface's (u, v) domain).
#[pyclass(name = "Point2D", module = "geomcore", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyPoint2D(pub(crate) Point2D);

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
        self.0.coincides(other.0, tolerance(tol))
    }

    /// Whether every coordinate is finite.
    fn is_finite(&self) -> bool {
        self.0.is_finite()
    }

    fn __add__(&self, offset: PyVector2D) -> PyPoint2D {
        PyPoint2D(self.0 + offset.0)
    }

    fn __sub__(&self, other: PointOrVector2D, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match other {
            PointOrVector2D::Point(point) => Ok(PyVector2D(self.0 - point.0)
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            PointOrVector2D::Vector(vector) => Ok(PyPoint2D(self.0 - vector.0)
                .into_pyobject(py)?
                .into_any()
                .unbind()),
        }
    }

    fn __repr__(&self) -> String {
        format!("Point2D{}", coordinates_2d(self.0))
    }
}

/// A vector in 2D space.
#[pyclass(name = "Vector2D", module = "geomcore", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyVector2D(pub(crate) Vector2D);

#[pymethods]
impl PyVector2D {
    #[new]
    /// Build a vector from its two components.
    fn py_new(x: f64, y: f64) -> Self {
        PyVector2D(Vector2D::new(x, y))
    }

    /// The zero vector.
    #[staticmethod]
    fn zero() -> Self {
        PyVector2D(Vector2D::ZERO)
    }

    /// The unit vector along the x axis.
    #[staticmethod]
    fn unit_x() -> Self {
        PyVector2D(Vector2D::X)
    }

    /// The unit vector along the y axis.
    #[staticmethod]
    fn unit_y() -> Self {
        PyVector2D(Vector2D::Y)
    }

    #[getter]
    fn x(&self) -> f64 {
        self.0.x
    }

    #[getter]
    fn y(&self) -> f64 {
        self.0.y
    }

    /// The `(x, y)` components as a tuple.
    fn components(&self) -> (f64, f64) {
        (self.0.x, self.0.y)
    }

    /// Dot product with another vector.
    fn dot(&self, other: &PyVector2D) -> f64 {
        self.0.dot(other.0)
    }

    /// The z component of the 3D cross product: positive when `other` lies
    /// counter-clockwise from this vector.
    fn cross(&self, other: &PyVector2D) -> f64 {
        self.0.cross(other.0)
    }

    /// This vector turned a quarter turn counter-clockwise.
    fn perp(&self) -> PyVector2D {
        PyVector2D(self.0.perp())
    }

    /// Euclidean length of the vector.
    fn magnitude(&self) -> f64 {
        self.0.magnitude()
    }

    /// The squared length of the vector.
    fn square_magnitude(&self) -> f64 {
        self.0.square_magnitude()
    }

    /// Whether every component is finite.
    fn is_finite(&self) -> bool {
        self.0.is_finite()
    }

    /// The unit vector with the same direction, or `None` for a vector too
    /// short to normalize.
    fn normalized(&self) -> Option<PyVector2D> {
        self.0.normalized().map(PyVector2D)
    }

    fn __add__(&self, other: &PyVector2D) -> PyVector2D {
        PyVector2D(self.0 + other.0)
    }

    fn __sub__(&self, other: &PyVector2D) -> PyVector2D {
        PyVector2D(self.0 - other.0)
    }

    fn __neg__(&self) -> PyVector2D {
        PyVector2D(-self.0)
    }

    fn __mul__(&self, scalar: f64) -> PyVector2D {
        PyVector2D(self.0 * scalar)
    }

    fn __rmul__(&self, scalar: f64) -> PyVector2D {
        PyVector2D(scalar * self.0)
    }

    fn __repr__(&self) -> String {
        format!("Vector2D{}", coordinates_2d(self.0))
    }
}

/// An axis: a point plus a unit direction.
#[pyclass(name = "Axis3D", module = "geomcore", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyAxis3D(pub(crate) Axis3D);

#[pymethods]
impl PyAxis3D {
    #[new]
    /// Build an axis from an origin and a direction (normalized internally).
    fn py_new(origin: PyPoint3D, direction: PyVector3D) -> PyResult<Self> {
        Ok(PyAxis3D(
            Axis3D::new(origin.0, direction.0).map_err(python_error)?,
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

/// An axis in the plane: a point plus a unit direction.
#[pyclass(name = "Axis2D", module = "geomcore", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyAxis2D(pub(crate) Axis2D);

#[pymethods]
impl PyAxis2D {
    #[new]
    /// Build an axis from an origin and a direction (normalized internally).
    fn py_new(origin: PyPoint2D, direction: PyVector2D) -> PyResult<Self> {
        Ok(PyAxis2D(
            Axis2D::new(origin.0, direction.0).map_err(python_error)?,
        ))
    }

    /// The axis origin.
    fn origin(&self) -> PyPoint2D {
        PyPoint2D(self.0.origin())
    }

    /// The unit direction of the axis.
    fn direction(&self) -> PyVector2D {
        PyVector2D(self.0.direction())
    }

    fn __repr__(&self) -> String {
        format!(
            "Axis2D(origin={}, direction={})",
            coordinates_2d(self.0.origin()),
            coordinates_2d(self.0.direction())
        )
    }
}

/// An orthonormal placement frame in the plane (origin + x/y directions).
/// The frame is direct when y is x turned a quarter turn counter-clockwise.
#[pyclass(name = "Frame2D", module = "geomcore", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyFrame2D(pub(crate) Frame2D);

#[pymethods]
impl PyFrame2D {
    #[new]
    /// Build a frame from an origin and the x and y directions (each
    /// normalized internally; they must be orthogonal).
    fn py_new(
        origin: PyPoint2D,
        x_direction: PyVector2D,
        y_direction: PyVector2D,
    ) -> PyResult<Self> {
        Ok(PyFrame2D(
            Frame2D::new(origin.0, x_direction.0, y_direction.0).map_err(python_error)?,
        ))
    }

    /// Build a direct frame from an origin and an x direction.
    #[staticmethod]
    fn from_x(origin: PyPoint2D, x_direction: PyVector2D) -> PyResult<Self> {
        Ok(PyFrame2D(
            Frame2D::from_x(origin.0, x_direction.0).map_err(python_error)?,
        ))
    }

    /// The world frame at the origin.
    #[staticmethod]
    fn world() -> Self {
        PyFrame2D(Frame2D::WORLD)
    }

    /// The frame origin.
    fn origin(&self) -> PyPoint2D {
        PyPoint2D(self.0.origin())
    }

    /// The unit x direction.
    fn x_direction(&self) -> PyVector2D {
        PyVector2D(self.0.x_direction())
    }

    /// The unit y direction.
    fn y_direction(&self) -> PyVector2D {
        PyVector2D(self.0.y_direction())
    }

    /// Whether y is x turned a quarter turn counter-clockwise.
    fn is_direct(&self) -> bool {
        self.0.is_direct()
    }

    fn __repr__(&self) -> String {
        format!(
            "Frame2D(origin={}, x_direction={}, y_direction={})",
            coordinates_2d(self.0.origin()),
            coordinates_2d(self.0.x_direction()),
            coordinates_2d(self.0.y_direction())
        )
    }
}

/// A right-handed orthonormal placement frame (origin + x/y/z directions).
#[pyclass(name = "Frame3D", module = "geomcore", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyFrame3D(pub(crate) Frame3D);

#[pymethods]
impl PyFrame3D {
    #[new]
    /// Build a frame from an origin, main (z) direction, and an x hint.
    fn py_new(origin: PyPoint3D, z_direction: PyVector3D, x_hint: PyVector3D) -> PyResult<Self> {
        Ok(PyFrame3D(
            Frame3D::new(origin.0, z_direction.0, x_hint.0).map_err(python_error)?,
        ))
    }

    /// Build a frame from an origin and main (z) direction, deriving an
    /// arbitrary perpendicular x direction.
    #[staticmethod]
    fn from_z(origin: PyPoint3D, z_direction: PyVector3D) -> PyResult<Self> {
        Ok(PyFrame3D(
            Frame3D::from_z(origin.0, z_direction.0).map_err(python_error)?,
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

    /// The axis through the origin along the z direction.
    fn axis(&self) -> PyAxis3D {
        PyAxis3D(self.0.axis())
    }

    /// The coordinates `(u, v, w)` of `point` along the x, y and z
    /// directions, measured from the origin.
    fn local_coordinates(&self, point: PyPoint3D) -> (f64, f64, f64) {
        self.0.local_coordinates(point.0)
    }

    /// The point at `u` along x, `v` along y and `w` along z from the origin.
    fn point_at(&self, u: f64, v: f64, w: f64) -> PyPoint3D {
        PyPoint3D(self.0.point_at(u, v, w))
    }
}

/// Distance and angle tolerances shared by all geometric queries
/// (`contains`, and later projection and intersection).
///
/// `Tolerance()` with no arguments equals `Tolerance.default()`.
#[pyclass(name = "Tolerance", module = "geomcore", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyTolerance(pub(crate) Tolerance);

#[pymethods]
impl PyTolerance {
    #[new]
    #[pyo3(signature = (
        confusion = Tolerance::DEFAULT.confusion,
        angular = Tolerance::DEFAULT.angular,
        parametric = Tolerance::DEFAULT.parametric,
    ))]
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
            python_float(self.0.confusion),
            python_float(self.0.angular),
            python_float(self.0.parametric)
        )
    }
}

/// A rigid (or mirrored/scaled) affine transformation.
#[pyclass(name = "Transform", module = "geomcore", eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyTransform(pub(crate) Transform);

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
pub(crate) struct PyBoundingBox3D(pub(crate) BoundingBox3D);

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
            .ok_or_else(|| invalid_argument("from_points needs at least one point"))
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
            "BoundingBox3D(min={}, max={})",
            coordinates(lo),
            coordinates(hi)
        )
    }
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
pub(crate) struct PyDiscretization3D(pub(crate) Discretization3D);

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
pub(crate) struct PyDiscretization2D(pub(crate) Discretization2D);

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
pub(crate) fn discretization_criteria(
    max_chord_deviation: f64,
    max_angle_deviation: Option<f64>,
) -> PyResult<DiscretizationCriteria> {
    let criteria = DiscretizationCriteria::new(max_chord_deviation).map_err(python_error)?;
    match max_angle_deviation {
        Some(angle) => criteria
            .with_max_angle_deviation(angle)
            .map_err(python_error),
        None => Ok(criteria),
    }
}

pub(crate) fn discretize_3d(
    curve: &impl Discretize3D,
    max_chord_deviation: f64,
    max_angle_deviation: Option<f64>,
) -> PyResult<PyDiscretization3D> {
    let criteria = discretization_criteria(max_chord_deviation, max_angle_deviation)?;
    Ok(PyDiscretization3D(curve.discretize(criteria)))
}

pub(crate) fn discretize_2d(
    curve: &impl Discretize2D,
    max_chord_deviation: f64,
    max_angle_deviation: Option<f64>,
) -> PyResult<PyDiscretization2D> {
    let criteria = discretization_criteria(max_chord_deviation, max_angle_deviation)?;
    Ok(PyDiscretization2D(curve.discretize(criteria)))
}

pub(crate) fn point_count_3d(
    curve: &impl Discretize3D,
    max_chord_deviation: f64,
    max_angle_deviation: Option<f64>,
) -> PyResult<usize> {
    let criteria = discretization_criteria(max_chord_deviation, max_angle_deviation)?;
    Ok(curve.point_count(criteria))
}

pub(crate) fn point_count_2d(
    curve: &impl Discretize2D,
    max_chord_deviation: f64,
    max_angle_deviation: Option<f64>,
) -> PyResult<usize> {
    let criteria = discretization_criteria(max_chord_deviation, max_angle_deviation)?;
    Ok(curve.point_count(criteria))
}

/// Something with three coordinates, as `(x, y, z)` in a `__repr__`.
pub(crate) trait Coordinates {
    fn xyz(&self) -> (f64, f64, f64);
}

impl Coordinates for Point3D {
    fn xyz(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }
}

impl Coordinates for Vector3D {
    fn xyz(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }
}

/// Something with two coordinates, as `(x, y)` in a `__repr__`.
pub(crate) trait Coordinates2D {
    fn xy(&self) -> (f64, f64);
}

impl Coordinates2D for Point2D {
    fn xy(&self) -> (f64, f64) {
        (self.x, self.y)
    }
}

impl Coordinates2D for Vector2D {
    fn xy(&self) -> (f64, f64) {
        (self.x, self.y)
    }
}

/// `(x, y)` as written in a `__repr__`.
pub(crate) fn coordinates_2d(value: impl Coordinates2D) -> String {
    let (x, y) = value.xy();
    format!("({}, {})", float_repr(x), float_repr(y))
}

/// `(x, y, z)` as written in a `__repr__`.
pub(crate) fn coordinates(value: impl Coordinates) -> String {
    let (x, y, z) = value.xyz();
    format!("({}, {}, {})", float_repr(x), float_repr(y), float_repr(z))
}

/// Coordinates below this magnitude print as `0.0` in a `__repr__`: they are
/// rounding noise of a computation that should have given zero (such as
/// `cos(pi / 2)`), not geometry.
const REPR_NOISE: f64 = 1e-15;

/// A float as a `__repr__` writes it: as Python's `repr` (shortest text that
/// reads back to the same number), with magnitudes below `1e-15` as `0.0`.
/// The snap is for readability, so a repr is not an exact round trip of
/// coordinates that small.
pub(crate) fn float_repr(value: f64) -> String {
    if value.abs() < REPR_NOISE {
        return "0.0".to_string();
    }
    python_float(value)
}

/// A float as Python's `repr` writes it. Use it where a tiny value is
/// meaningful, such as a tolerance.
pub(crate) fn python_float(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let sign = if value.is_sign_negative() { "-" } else { "" };
    // `{:e}` gives the shortest round-trip digits: `d.ddde<exponent>`.
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("scientific notation has an exponent");
    let exponent: i32 = exponent.parse().expect("the exponent is an integer");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if !(-4..16).contains(&exponent) {
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        return format!("{sign}{mantissa}e{exponent_sign}{:02}", exponent.abs());
    }
    if exponent >= 0 {
        let integer_length = exponent as usize + 1;
        let padded = format!("{digits:0<integer_length$}");
        let (integer, fraction) = padded.split_at(integer_length);
        let fraction = if fraction.is_empty() { "0" } else { fraction };
        format!("{sign}{integer}.{fraction}")
    } else {
        let zeros = "0".repeat((-exponent - 1) as usize);
        format!("{sign}0.{zeros}{digits}")
    }
}

/// A Rust `bool` as Python writes it.
pub(crate) fn py_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}
