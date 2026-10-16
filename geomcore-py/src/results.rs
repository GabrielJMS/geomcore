//! Intersection, projection and extrema results, mirroring the Rust result types.
//!
//! Every class is frozen. Its fields are read-only properties named as in
//! Rust, compared by value. The intersection results and the curve–curve
//! extrema also hold the operands they were computed from, as `first` and
//! `second`, and the projections hold the object they project onto, as
//! `target`: the exact objects passed, shared rather than copied.

use pyo3::prelude::*;
use pyo3::types::PyList;

use geomcore::{
    Completeness, CurveCurveExtrema, CurveCurveExtremum, CurveCurveHit, CurveCurveHit2D,
    CurveCurveIntersection, CurveCurveIntersection2D, CurveCurveOverlap, CurveCurveRegion,
    CurveProjection, CurveProjection2D, CurveSurfaceHit, CurveSurfaceIntersection,
    CurveSurfaceRegion, IncompleteReason, IntersectionCurve, IntersectionMethod, Point2D, Point3D,
    SurfaceProjection, SurfaceSurfaceIntersection, SurfaceSurfaceRegion,
};

use crate::numeric::curve3d_to_py;
use crate::values::{PyPoint2D, PyPoint3D, coordinates, coordinates_2d, float_repr};

// ---------------------------------------------------------------------------
// repr helpers
// ---------------------------------------------------------------------------

fn point3_repr(p: Point3D) -> String {
    format!("Point3D{}", coordinates(p))
}

fn point2_repr(p: Point2D) -> String {
    format!("Point2D{}", coordinates_2d(p))
}

fn optional_repr(value: Option<f64>) -> String {
    value.map_or_else(|| "None".to_string(), float_repr)
}

fn range_repr((start, end): (f64, f64)) -> String {
    format!("({}, {})", float_repr(start), float_repr(end))
}

fn bool_repr(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

fn list_repr<T>(items: &[T], repr: impl Fn(&T) -> String) -> String {
    let parts: Vec<String> = items.iter().map(repr).collect();
    format!("[{}]", parts.join(", "))
}

/// The value equality of two operands, as Python's `==` sees it.
fn same_operand(py: Python<'_>, a: &Py<PyAny>, b: &Py<PyAny>) -> PyResult<bool> {
    if a.is(b) {
        return Ok(true);
    }
    a.bind(py).eq(b.bind(py))
}

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

/// How an intersection was computed.
///
/// `EXACT` is a closed form, always complete. `NUMERIC` is a search, whose
/// `completeness` tells whether it proved that nothing was missed.
#[pyclass(
    name = "IntersectionMethod",
    module = "geomcore",
    eq,
    eq_int,
    frozen,
    hash
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyIntersectionMethod {
    #[pyo3(name = "EXACT")]
    Exact,
    #[pyo3(name = "NUMERIC")]
    Numeric,
}

impl From<IntersectionMethod> for PyIntersectionMethod {
    fn from(method: IntersectionMethod) -> Self {
        match method {
            IntersectionMethod::Exact => PyIntersectionMethod::Exact,
            IntersectionMethod::Numeric => PyIntersectionMethod::Numeric,
        }
    }
}

#[pymethods]
impl PyIntersectionMethod {
    /// The member's name, as written after the dot.
    #[getter]
    fn name(&self) -> &'static str {
        self.repr().split_once('.').map_or("", |(_, name)| name)
    }
}

impl PyIntersectionMethod {
    fn repr(self) -> &'static str {
        match self {
            PyIntersectionMethod::Exact => "IntersectionMethod.EXACT",
            PyIntersectionMethod::Numeric => "IntersectionMethod.NUMERIC",
        }
    }
}

/// Whether a numeric search proved that it found everything.
///
/// - `COMPLETE`: nothing was missed (always the case for an exact result).
/// - `INCOMPLETE`: the search may have missed something inside its
///   `regions`, and found everything outside them.
/// - `SAMPLED`: the result comes from samples; nothing is proved.
#[pyclass(
    name = "CompletenessStatus",
    module = "geomcore",
    eq,
    eq_int,
    frozen,
    hash
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyCompletenessStatus {
    #[pyo3(name = "COMPLETE")]
    Complete,
    #[pyo3(name = "INCOMPLETE")]
    Incomplete,
    #[pyo3(name = "SAMPLED")]
    Sampled,
}

#[pymethods]
impl PyCompletenessStatus {
    /// The member's name, as written after the dot.
    #[getter]
    fn name(&self) -> &'static str {
        self.repr().split_once('.').map_or("", |(_, name)| name)
    }
}

impl PyCompletenessStatus {
    fn repr(self) -> &'static str {
        match self {
            PyCompletenessStatus::Complete => "CompletenessStatus.COMPLETE",
            PyCompletenessStatus::Incomplete => "CompletenessStatus.INCOMPLETE",
            PyCompletenessStatus::Sampled => "CompletenessStatus.SAMPLED",
        }
    }
}

/// Why an incomplete search stopped.
///
/// - `BUDGET_EXHAUSTED`: it ran out of subdivisions.
/// - `UNSEPARATED`: pieces of the operands stayed within tolerance of each
///   other down to the smallest piece, as at a tangent contact.
#[pyclass(
    name = "IncompleteReason",
    module = "geomcore",
    eq,
    eq_int,
    frozen,
    hash
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyIncompleteReason {
    #[pyo3(name = "BUDGET_EXHAUSTED")]
    BudgetExhausted,
    #[pyo3(name = "UNSEPARATED")]
    Unseparated,
}

impl From<IncompleteReason> for PyIncompleteReason {
    fn from(reason: IncompleteReason) -> Self {
        match reason {
            IncompleteReason::BudgetExhausted => PyIncompleteReason::BudgetExhausted,
            IncompleteReason::Unseparated => PyIncompleteReason::Unseparated,
        }
    }
}

#[pymethods]
impl PyIncompleteReason {
    /// The member's name, as written after the dot.
    #[getter]
    fn name(&self) -> &'static str {
        self.repr().split_once('.').map_or("", |(_, name)| name)
    }
}

impl PyIncompleteReason {
    fn repr(self) -> &'static str {
        match self {
            PyIncompleteReason::BudgetExhausted => "IncompleteReason.BUDGET_EXHAUSTED",
            PyIncompleteReason::Unseparated => "IncompleteReason.UNSEPARATED",
        }
    }
}

// ---------------------------------------------------------------------------
// Regions
// ---------------------------------------------------------------------------

/// Where an incomplete curve–curve search may have missed something: a
/// parameter interval on each curve, and a box in space holding both
/// curves over their intervals.
#[pyclass(name = "CurveCurveRegion", module = "geomcore", eq, frozen)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyCurveCurveRegion(pub(crate) CurveCurveRegion);

#[pymethods]
impl PyCurveCurveRegion {
    /// The first curve's parameter interval.
    #[getter]
    fn first_range(&self) -> (f64, f64) {
        self.0.first_range
    }

    /// The second curve's parameter interval.
    #[getter]
    fn second_range(&self) -> (f64, f64) {
        self.0.second_range
    }

    /// The corner of the box with the smallest coordinates.
    #[getter]
    fn min_corner(&self) -> PyPoint3D {
        PyPoint3D(self.0.min_corner)
    }

    /// The corner of the box with the largest coordinates.
    #[getter]
    fn max_corner(&self) -> PyPoint3D {
        PyPoint3D(self.0.max_corner)
    }

    fn __repr__(&self) -> String {
        format!(
            "CurveCurveRegion(first_range={}, second_range={}, min_corner={}, max_corner={})",
            range_repr(self.0.first_range),
            range_repr(self.0.second_range),
            point3_repr(self.0.min_corner),
            point3_repr(self.0.max_corner),
        )
    }
}

/// Where an incomplete curve–surface search may have missed something: a
/// parameter interval on the curve, a `(u, v)` rectangle on the surface,
/// and a box in space holding both over them.
#[pyclass(name = "CurveSurfaceRegion", module = "geomcore", eq, frozen)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyCurveSurfaceRegion(pub(crate) CurveSurfaceRegion);

#[pymethods]
impl PyCurveSurfaceRegion {
    /// The curve parameter interval.
    #[getter]
    fn curve_range(&self) -> (f64, f64) {
        self.0.curve_range
    }

    /// The surface's `u` parameter interval.
    #[getter]
    fn u_range(&self) -> (f64, f64) {
        self.0.u_range
    }

    /// The surface's `v` parameter interval.
    #[getter]
    fn v_range(&self) -> (f64, f64) {
        self.0.v_range
    }

    /// The corner of the box with the smallest coordinates.
    #[getter]
    fn min_corner(&self) -> PyPoint3D {
        PyPoint3D(self.0.min_corner)
    }

    /// The corner of the box with the largest coordinates.
    #[getter]
    fn max_corner(&self) -> PyPoint3D {
        PyPoint3D(self.0.max_corner)
    }

    fn __repr__(&self) -> String {
        format!(
            "CurveSurfaceRegion(curve_range={}, u_range={}, v_range={}, min_corner={}, max_corner={})",
            range_repr(self.0.curve_range),
            range_repr(self.0.u_range),
            range_repr(self.0.v_range),
            point3_repr(self.0.min_corner),
            point3_repr(self.0.max_corner),
        )
    }
}

/// Where an incomplete surface–surface search may have missed something:
/// a `(u, v)` rectangle on each surface, and a box in space holding both
/// over them.
#[pyclass(name = "SurfaceSurfaceRegion", module = "geomcore", eq, frozen)]
#[derive(Clone, PartialEq)]
pub(crate) struct PySurfaceSurfaceRegion(pub(crate) SurfaceSurfaceRegion);

#[pymethods]
impl PySurfaceSurfaceRegion {
    /// The first surface's `u` parameter interval.
    #[getter]
    fn first_u_range(&self) -> (f64, f64) {
        self.0.first_u_range
    }

    /// The first surface's `v` parameter interval.
    #[getter]
    fn first_v_range(&self) -> (f64, f64) {
        self.0.first_v_range
    }

    /// The second surface's `u` parameter interval.
    #[getter]
    fn second_u_range(&self) -> (f64, f64) {
        self.0.second_u_range
    }

    /// The second surface's `v` parameter interval.
    #[getter]
    fn second_v_range(&self) -> (f64, f64) {
        self.0.second_v_range
    }

    /// The corner of the box with the smallest coordinates.
    #[getter]
    fn min_corner(&self) -> PyPoint3D {
        PyPoint3D(self.0.min_corner)
    }

    /// The corner of the box with the largest coordinates.
    #[getter]
    fn max_corner(&self) -> PyPoint3D {
        PyPoint3D(self.0.max_corner)
    }

    fn __repr__(&self) -> String {
        format!(
            "SurfaceSurfaceRegion(first_u_range={}, first_v_range={}, second_u_range={}, second_v_range={}, min_corner={}, max_corner={})",
            range_repr(self.0.first_u_range),
            range_repr(self.0.first_v_range),
            range_repr(self.0.second_u_range),
            range_repr(self.0.second_v_range),
            point3_repr(self.0.min_corner),
            point3_repr(self.0.max_corner),
        )
    }
}

// ---------------------------------------------------------------------------
// Completeness
// ---------------------------------------------------------------------------

/// The regions of an incomplete result, of the kind its pair family uses.
#[derive(Clone, PartialEq)]
enum Regions {
    CurveCurve(Vec<CurveCurveRegion>),
    CurveSurface(Vec<CurveSurfaceRegion>),
    SurfaceSurface(Vec<SurfaceSurfaceRegion>),
}

/// Whether a result is proved complete and, when it is not, why and where
/// something may have been missed.
///
/// `regions` is empty unless `status` is `INCOMPLETE`; `reason` is `None`
/// unless it is.
#[pyclass(name = "Completeness", module = "geomcore", eq, frozen)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyCompleteness {
    status: PyCompletenessStatus,
    reason: Option<PyIncompleteReason>,
    regions: Regions,
}

impl PyCompleteness {
    fn new<R: Clone>(completeness: &Completeness<R>, wrap: fn(Vec<R>) -> Regions) -> Self {
        let (status, reason, regions) = match completeness {
            Completeness::Complete => (PyCompletenessStatus::Complete, None, Vec::new()),
            Completeness::Incomplete { reason, regions } => (
                PyCompletenessStatus::Incomplete,
                Some((*reason).into()),
                regions.clone(),
            ),
            Completeness::Sampled => (PyCompletenessStatus::Sampled, None, Vec::new()),
        };
        PyCompleteness {
            status,
            reason,
            regions: wrap(regions),
        }
    }

    pub(crate) fn curve_curve(completeness: &Completeness<CurveCurveRegion>) -> Self {
        Self::new(completeness, Regions::CurveCurve)
    }

    pub(crate) fn curve_surface(completeness: &Completeness<CurveSurfaceRegion>) -> Self {
        Self::new(completeness, Regions::CurveSurface)
    }

    pub(crate) fn surface_surface(completeness: &Completeness<SurfaceSurfaceRegion>) -> Self {
        Self::new(completeness, Regions::SurfaceSurface)
    }
}

#[pymethods]
impl PyCompleteness {
    /// `COMPLETE`, `INCOMPLETE` or `SAMPLED`.
    #[getter]
    fn status(&self) -> PyCompletenessStatus {
        self.status
    }

    /// Why the search stopped short, when `status` is `INCOMPLETE`;
    /// `None` otherwise.
    #[getter]
    fn reason(&self) -> Option<PyIncompleteReason> {
        self.reason
    }

    /// Where something may have been missed, when `status` is
    /// `INCOMPLETE`: `CurveCurveRegion`, `CurveSurfaceRegion` or
    /// `SurfaceSurfaceRegion` objects, by pair family. Empty otherwise.
    #[getter]
    fn regions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        match &self.regions {
            Regions::CurveCurve(regions) => {
                PyList::new(py, regions.iter().map(|r| PyCurveCurveRegion(*r)))
            }
            Regions::CurveSurface(regions) => {
                PyList::new(py, regions.iter().map(|r| PyCurveSurfaceRegion(*r)))
            }
            Regions::SurfaceSurface(regions) => {
                PyList::new(py, regions.iter().map(|r| PySurfaceSurfaceRegion(*r)))
            }
        }
    }

    /// Whether the result is proved to hold everything.
    fn is_complete(&self) -> bool {
        self.status == PyCompletenessStatus::Complete
    }

    fn __repr__(&self) -> String {
        if self.status != PyCompletenessStatus::Incomplete {
            return format!("Completeness(status={})", self.status.repr());
        }
        let reason = self.reason.map_or("None", PyIncompleteReason::repr);
        let regions = match &self.regions {
            Regions::CurveCurve(r) => list_repr(r, |r| PyCurveCurveRegion(*r).__repr__()),
            Regions::CurveSurface(r) => list_repr(r, |r| PyCurveSurfaceRegion(*r).__repr__()),
            Regions::SurfaceSurface(r) => list_repr(r, |r| PySurfaceSurfaceRegion(*r).__repr__()),
        };
        format!(
            "Completeness(status={}, reason={reason}, regions={regions})",
            self.status.repr()
        )
    }
}

// ---------------------------------------------------------------------------
// Hits and pieces
// ---------------------------------------------------------------------------

/// Where two 3D curves meet.
///
/// A parameter is `None` when the solver did not compute it. Recover it
/// from `point` with the curve's `parameter_of` when it has one, or with
/// its `project_point` otherwise.
#[pyclass(name = "CurveCurveHit", module = "geomcore", eq, frozen)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyCurveCurveHit(pub(crate) CurveCurveHit);

#[pymethods]
impl PyCurveCurveHit {
    /// The parameter on the first curve, or `None` when not computed.
    #[getter]
    fn first_parameter(&self) -> Option<f64> {
        self.0.first_parameter
    }

    /// The parameter on the second curve, or `None` when not computed.
    #[getter]
    fn second_parameter(&self) -> Option<f64> {
        self.0.second_parameter
    }

    /// The meeting point.
    #[getter]
    fn point(&self) -> PyPoint3D {
        PyPoint3D(self.0.point)
    }

    /// 1 for a crossing, 2 or more where the curves are tangent.
    #[getter]
    fn multiplicity(&self) -> u32 {
        self.0.multiplicity
    }

    fn __repr__(&self) -> String {
        format!(
            "CurveCurveHit(first_parameter={}, second_parameter={}, point={}, multiplicity={})",
            optional_repr(self.0.first_parameter),
            optional_repr(self.0.second_parameter),
            point3_repr(self.0.point),
            self.0.multiplicity,
        )
    }
}

/// Where two 2D curves meet.
///
/// A parameter is `None` when the solver did not compute it. Recover it
/// from `point` with the curve's `parameter_of` when it has one, or with
/// its `project_point` otherwise.
#[pyclass(name = "CurveCurveHit2D", module = "geomcore", eq, frozen)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyCurveCurveHit2D(pub(crate) CurveCurveHit2D);

#[pymethods]
impl PyCurveCurveHit2D {
    /// The parameter on the first curve, or `None` when not computed.
    #[getter]
    fn first_parameter(&self) -> Option<f64> {
        self.0.first_parameter
    }

    /// The parameter on the second curve, or `None` when not computed.
    #[getter]
    fn second_parameter(&self) -> Option<f64> {
        self.0.second_parameter
    }

    /// The meeting point.
    #[getter]
    fn point(&self) -> PyPoint2D {
        PyPoint2D(self.0.point)
    }

    /// 1 for a crossing, 2 or more where the curves are tangent.
    #[getter]
    fn multiplicity(&self) -> u32 {
        self.0.multiplicity
    }

    fn __repr__(&self) -> String {
        format!(
            "CurveCurveHit2D(first_parameter={}, second_parameter={}, point={}, multiplicity={})",
            optional_repr(self.0.first_parameter),
            optional_repr(self.0.second_parameter),
            point2_repr(self.0.point),
            self.0.multiplicity,
        )
    }
}

/// Where a curve meets a surface.
///
/// A parameter is `None` when the solver did not compute it. Recover the
/// curve parameter with the curve's `parameter_of` and the surface
/// parameters with the surface's `parameters_of` when they have them, or
/// with their `project_point` otherwise.
#[pyclass(name = "CurveSurfaceHit", module = "geomcore", eq, frozen)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyCurveSurfaceHit(pub(crate) CurveSurfaceHit);

#[pymethods]
impl PyCurveSurfaceHit {
    /// The parameter on the curve, or `None` when not computed.
    #[getter]
    fn curve_parameter(&self) -> Option<f64> {
        self.0.curve_parameter
    }

    /// The `(u, v)` parameters on the surface, or `None` when not computed.
    #[getter]
    fn surface_parameters(&self) -> Option<(f64, f64)> {
        self.0.surface_parameters
    }

    /// The meeting point.
    #[getter]
    fn point(&self) -> PyPoint3D {
        PyPoint3D(self.0.point)
    }

    /// 1 for a crossing, 2 or more where the curve is tangent to the
    /// surface.
    #[getter]
    fn multiplicity(&self) -> u32 {
        self.0.multiplicity
    }

    fn __repr__(&self) -> String {
        let surface_parameters = self
            .0
            .surface_parameters
            .map_or_else(|| "None".to_string(), range_repr);
        format!(
            "CurveSurfaceHit(curve_parameter={}, surface_parameters={surface_parameters}, point={}, multiplicity={})",
            optional_repr(self.0.curve_parameter),
            point3_repr(self.0.point),
            self.0.multiplicity,
        )
    }
}

/// A range where two curves coincide: a parameter interval on each.
///
/// A whole-curve coincidence is one overlap over both curves' bounds,
/// which may be infinite.
#[pyclass(name = "CurveCurveOverlap", module = "geomcore", eq, frozen)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyCurveCurveOverlap(pub(crate) CurveCurveOverlap);

#[pymethods]
impl PyCurveCurveOverlap {
    /// The first curve's parameter interval.
    #[getter]
    fn first_range(&self) -> (f64, f64) {
        self.0.first_range
    }

    /// The second curve's parameter interval.
    #[getter]
    fn second_range(&self) -> (f64, f64) {
        self.0.second_range
    }

    /// Whether both curves run the same way along the overlap.
    #[getter]
    fn same_direction(&self) -> bool {
        self.0.same_direction
    }

    fn __repr__(&self) -> String {
        format!(
            "CurveCurveOverlap(first_range={}, second_range={}, same_direction={})",
            range_repr(self.0.first_range),
            range_repr(self.0.second_range),
            bool_repr(self.0.same_direction),
        )
    }
}

/// A curve where two surfaces meet.
#[pyclass(name = "IntersectionCurve", module = "geomcore", eq, frozen)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyIntersectionCurve(pub(crate) IntersectionCurve);

#[pymethods]
impl PyIntersectionCurve {
    /// The curve: a line, circle, ellipse, parabola or hyperbola from a
    /// closed form, a `BSplineCurve3D` traced by marching otherwise.
    #[getter]
    fn curve(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        curve3d_to_py(py, self.0.curve.clone())
    }

    /// Whether the surfaces graze along the curve. Always `False` for a
    /// marched curve.
    #[getter]
    fn tangent(&self) -> bool {
        self.0.tangent
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let curve = self.curve(py)?.bind(py).repr()?.to_string();
        Ok(format!(
            "IntersectionCurve(curve={curve}, tangent={})",
            bool_repr(self.0.tangent)
        ))
    }
}

// ---------------------------------------------------------------------------
// Intersection results
// ---------------------------------------------------------------------------

/// The common text of a result's `repr`.
fn result_repr(
    py: Python<'_>,
    class: &str,
    first: &Py<PyAny>,
    second: &Py<PyAny>,
    method: IntersectionMethod,
    completeness: &PyCompleteness,
    content: &str,
) -> PyResult<String> {
    Ok(format!(
        "{class}(first={}, second={}, method={}, completeness={}, {content})",
        first.bind(py).get_type().name()?,
        second.bind(py).get_type().name()?,
        PyIntersectionMethod::from(method).repr(),
        completeness.status.repr(),
    ))
}

/// The intersection of two 3D curves: the points where they meet and the
/// ranges where they coincide.
///
/// `first` and `second` are the operands, in the order of the call.
#[pyclass(name = "CurveCurveIntersection", module = "geomcore", frozen)]
pub(crate) struct PyCurveCurveIntersection {
    pub(crate) result: CurveCurveIntersection,
    pub(crate) first: Py<PyAny>,
    pub(crate) second: Py<PyAny>,
}

#[pymethods]
impl PyCurveCurveIntersection {
    /// The points where the curves meet, by first-curve parameter when
    /// every hit has one.
    #[getter]
    fn hits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.result.hits().iter().map(|h| PyCurveCurveHit(*h)))
    }

    /// The ranges where the curves coincide.
    #[getter]
    fn overlaps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            self.result
                .overlaps()
                .iter()
                .map(|o| PyCurveCurveOverlap(*o)),
        )
    }

    /// `EXACT` or `NUMERIC`.
    #[getter]
    fn method(&self) -> PyIntersectionMethod {
        self.result.method().into()
    }

    /// Whether the result is proved to hold every hit.
    #[getter]
    fn completeness(&self) -> PyCompleteness {
        PyCompleteness::curve_curve(self.result.completeness())
    }

    /// The first operand.
    #[getter]
    fn first(&self, py: Python<'_>) -> Py<PyAny> {
        self.first.clone_ref(py)
    }

    /// The second operand.
    #[getter]
    fn second(&self, py: Python<'_>) -> Py<PyAny> {
        self.second.clone_ref(py)
    }

    /// Whether the result is proved to hold every hit.
    fn is_complete(&self) -> bool {
        self.result.is_complete()
    }

    /// Whether there is no hit and no overlap. An empty result that is not
    /// complete means "found nothing, may have missed something".
    fn is_empty(&self) -> bool {
        self.result.is_empty()
    }

    /// False when empty, so `if a.intersect(b):` reads as "they meet".
    fn __bool__(&self) -> bool {
        !self.result.is_empty()
    }

    fn __eq__(&self, py: Python<'_>, other: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(self.result == other.result
            && same_operand(py, &self.first, &other.first)?
            && same_operand(py, &self.second, &other.second)?)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let content = format!(
            "hits={}, overlaps={}",
            list_repr(self.result.hits(), |h| PyCurveCurveHit(*h).__repr__()),
            list_repr(self.result.overlaps(), |o| PyCurveCurveOverlap(*o)
                .__repr__()),
        );
        result_repr(
            py,
            "CurveCurveIntersection",
            &self.first,
            &self.second,
            self.result.method(),
            &self.completeness(),
            &content,
        )
    }
}

/// The intersection of two 2D curves: the points where they meet and the
/// ranges where they coincide. Always exact.
///
/// `first` and `second` are the operands, in the order of the call.
#[pyclass(name = "CurveCurveIntersection2D", module = "geomcore", frozen)]
pub(crate) struct PyCurveCurveIntersection2D {
    pub(crate) result: CurveCurveIntersection2D,
    pub(crate) first: Py<PyAny>,
    pub(crate) second: Py<PyAny>,
}

#[pymethods]
impl PyCurveCurveIntersection2D {
    /// The points where the curves meet, by first-curve parameter when
    /// every hit has one.
    #[getter]
    fn hits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.result.hits().iter().map(|h| PyCurveCurveHit2D(*h)))
    }

    /// The ranges where the curves coincide.
    #[getter]
    fn overlaps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            self.result
                .overlaps()
                .iter()
                .map(|o| PyCurveCurveOverlap(*o)),
        )
    }

    /// Always `EXACT`.
    #[getter]
    fn method(&self) -> PyIntersectionMethod {
        self.result.method().into()
    }

    /// Always complete.
    #[getter]
    fn completeness(&self) -> PyCompleteness {
        PyCompleteness::curve_curve(self.result.completeness())
    }

    /// The first operand.
    #[getter]
    fn first(&self, py: Python<'_>) -> Py<PyAny> {
        self.first.clone_ref(py)
    }

    /// The second operand.
    #[getter]
    fn second(&self, py: Python<'_>) -> Py<PyAny> {
        self.second.clone_ref(py)
    }

    /// Whether the result is proved to hold every hit.
    fn is_complete(&self) -> bool {
        self.result.is_complete()
    }

    /// Whether there is no hit and no overlap.
    fn is_empty(&self) -> bool {
        self.result.is_empty()
    }

    /// False when empty, so `if a.intersect(b):` reads as "they meet".
    fn __bool__(&self) -> bool {
        !self.result.is_empty()
    }

    fn __eq__(&self, py: Python<'_>, other: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(self.result == other.result
            && same_operand(py, &self.first, &other.first)?
            && same_operand(py, &self.second, &other.second)?)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let content = format!(
            "hits={}, overlaps={}",
            list_repr(self.result.hits(), |h| PyCurveCurveHit2D(*h).__repr__()),
            list_repr(self.result.overlaps(), |o| PyCurveCurveOverlap(*o)
                .__repr__()),
        );
        result_repr(
            py,
            "CurveCurveIntersection2D",
            &self.first,
            &self.second,
            self.result.method(),
            &self.completeness(),
            &content,
        )
    }
}

/// The intersection of a curve and a surface: the points where they meet
/// and the curve's parameter ranges that lie on the surface.
///
/// `first` and `second` are the operands, in the order of the call;
/// whichever order that was, the hit parameters keep their roles, and
/// `curve` and `surface` name the operands by role.
#[pyclass(name = "CurveSurfaceIntersection", module = "geomcore", frozen)]
pub(crate) struct PyCurveSurfaceIntersection {
    pub(crate) result: CurveSurfaceIntersection,
    pub(crate) first: Py<PyAny>,
    pub(crate) second: Py<PyAny>,
}

impl PyCurveSurfaceIntersection {
    /// Whether the call was made on the surface.
    fn surface_first(&self, py: Python<'_>) -> bool {
        crate::surfaces::surface_of(self.first.bind(py)).is_some()
    }
}

#[pymethods]
impl PyCurveSurfaceIntersection {
    /// The points where the curve meets the surface, by curve parameter
    /// when every hit has one.
    #[getter]
    fn hits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.result.hits().iter().map(|h| PyCurveSurfaceHit(*h)))
    }

    /// The curve's `(start, end)` parameter ranges that lie on the
    /// surface. "The circle lies on the sphere" is one range over
    /// `(0, 2π)`.
    #[getter]
    fn overlaps(&self) -> Vec<(f64, f64)> {
        self.result.overlaps().to_vec()
    }

    /// `EXACT` or `NUMERIC`.
    #[getter]
    fn method(&self) -> PyIntersectionMethod {
        self.result.method().into()
    }

    /// Whether the result is proved to hold every hit.
    #[getter]
    fn completeness(&self) -> PyCompleteness {
        PyCompleteness::curve_surface(self.result.completeness())
    }

    /// The first operand.
    #[getter]
    fn first(&self, py: Python<'_>) -> Py<PyAny> {
        self.first.clone_ref(py)
    }

    /// The second operand.
    #[getter]
    fn second(&self, py: Python<'_>) -> Py<PyAny> {
        self.second.clone_ref(py)
    }

    /// The curve operand, whichever of `first` and `second` it is.
    #[getter]
    fn curve(&self, py: Python<'_>) -> Py<PyAny> {
        if self.surface_first(py) {
            self.second.clone_ref(py)
        } else {
            self.first.clone_ref(py)
        }
    }

    /// The surface operand, whichever of `first` and `second` it is.
    #[getter]
    fn surface(&self, py: Python<'_>) -> Py<PyAny> {
        if self.surface_first(py) {
            self.first.clone_ref(py)
        } else {
            self.second.clone_ref(py)
        }
    }

    /// Whether the result is proved to hold every hit.
    fn is_complete(&self) -> bool {
        self.result.is_complete()
    }

    /// Whether there is no hit and no overlap. An empty result that is not
    /// complete means "found nothing, may have missed something".
    fn is_empty(&self) -> bool {
        self.result.is_empty()
    }

    /// False when empty, so `if a.intersect(b):` reads as "they meet".
    fn __bool__(&self) -> bool {
        !self.result.is_empty()
    }

    fn __eq__(&self, py: Python<'_>, other: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(self.result == other.result
            && same_operand(py, &self.first, &other.first)?
            && same_operand(py, &self.second, &other.second)?)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let content = format!(
            "hits={}, overlaps={}",
            list_repr(self.result.hits(), |h| PyCurveSurfaceHit(*h).__repr__()),
            list_repr(self.result.overlaps(), |o| range_repr(*o)),
        );
        result_repr(
            py,
            "CurveSurfaceIntersection",
            &self.first,
            &self.second,
            self.result.method(),
            &self.completeness(),
            &content,
        )
    }
}

/// The intersection of two surfaces: the curves where they meet, the
/// isolated points where they touch, and whether they coincide.
///
/// `first` and `second` are the operands, in the order of the call.
#[pyclass(name = "SurfaceSurfaceIntersection", module = "geomcore", frozen)]
pub(crate) struct PySurfaceSurfaceIntersection {
    pub(crate) result: SurfaceSurfaceIntersection,
    pub(crate) first: Py<PyAny>,
    pub(crate) second: Py<PyAny>,
}

#[pymethods]
impl PySurfaceSurfaceIntersection {
    /// The curves where the surfaces meet.
    #[getter]
    fn curves<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            self.result
                .curves()
                .iter()
                .map(|c| PyIntersectionCurve(c.clone())),
        )
    }

    /// The isolated points where the surfaces touch.
    #[getter]
    fn contact_points(&self) -> Vec<PyPoint3D> {
        self.result
            .contact_points()
            .iter()
            .map(|p| PyPoint3D(*p))
            .collect()
    }

    /// `EXACT` or `NUMERIC`.
    #[getter]
    fn method(&self) -> PyIntersectionMethod {
        self.result.method().into()
    }

    /// Whether the result is proved to hold every curve. Marching reports
    /// `SAMPLED`.
    #[getter]
    fn completeness(&self) -> PyCompleteness {
        PyCompleteness::surface_surface(self.result.completeness())
    }

    /// The first operand.
    #[getter]
    fn first(&self, py: Python<'_>) -> Py<PyAny> {
        self.first.clone_ref(py)
    }

    /// The second operand.
    #[getter]
    fn second(&self, py: Python<'_>) -> Py<PyAny> {
        self.second.clone_ref(py)
    }

    /// Whether the surfaces are the same surface.
    fn is_coincident(&self) -> bool {
        self.result.is_coincident()
    }

    /// Whether the result is proved to hold every curve.
    fn is_complete(&self) -> bool {
        self.result.is_complete()
    }

    /// Whether there is no curve, no contact point and no coincidence.
    fn is_empty(&self) -> bool {
        self.result.is_empty()
    }

    /// False when empty, so `if a.intersect(b):` reads as "they meet".
    fn __bool__(&self) -> bool {
        !self.result.is_empty()
    }

    fn __eq__(&self, py: Python<'_>, other: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(self.result == other.result
            && same_operand(py, &self.first, &other.first)?
            && same_operand(py, &self.second, &other.second)?)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let mut curves = Vec::with_capacity(self.result.curves().len());
        for piece in self.result.curves() {
            curves.push(PyIntersectionCurve(piece.clone()).__repr__(py)?);
        }
        let content = format!(
            "curves=[{}], contact_points={}, coincident={}",
            curves.join(", "),
            list_repr(self.result.contact_points(), |p| point3_repr(*p)),
            bool_repr(self.result.is_coincident()),
        );
        result_repr(
            py,
            "SurfaceSurfaceIntersection",
            &self.first,
            &self.second,
            self.result.method(),
            &self.completeness(),
            &content,
        )
    }
}

// ---------------------------------------------------------------------------
// Projections
// ---------------------------------------------------------------------------

/// Where a 3D curve is closest to a point.
///
/// `target` is the curve that was projected onto, as the exact object the
/// method was called on, and `query_point` is the point that was projected.
#[pyclass(name = "CurveProjection", module = "geomcore", frozen)]
pub(crate) struct PyCurveProjection {
    pub(crate) result: CurveProjection,
    pub(crate) target: Py<PyAny>,
    pub(crate) query_point: Point3D,
}

impl PyCurveProjection {
    pub(crate) fn new(
        result: CurveProjection,
        target: &Bound<'_, impl pyo3::PyClass>,
        query_point: Point3D,
    ) -> Self {
        PyCurveProjection {
            result,
            target: target.clone().into_any().unbind(),
            query_point,
        }
    }
}

#[pymethods]
impl PyCurveProjection {
    /// The parameter of the closest point.
    #[getter]
    fn parameter(&self) -> f64 {
        self.result.parameter
    }

    /// The closest point: the curve evaluated at `parameter`.
    #[getter]
    fn point(&self) -> PyPoint3D {
        PyPoint3D(self.result.point)
    }

    /// The distance from `query_point` to the curve.
    #[getter]
    fn distance(&self) -> f64 {
        self.result.distance
    }

    /// The curve that was projected onto.
    #[getter]
    fn target(&self, py: Python<'_>) -> Py<PyAny> {
        self.target.clone_ref(py)
    }

    /// The point that was projected.
    #[getter]
    fn query_point(&self) -> PyPoint3D {
        PyPoint3D(self.query_point)
    }

    fn __eq__(&self, py: Python<'_>, other: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(self.result == other.result
            && self.query_point == other.query_point
            && same_operand(py, &self.target, &other.target)?)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "CurveProjection(target={}, query_point={}, parameter={}, point={}, distance={})",
            self.target.bind(py).get_type().name()?,
            point3_repr(self.query_point),
            float_repr(self.result.parameter),
            point3_repr(self.result.point),
            float_repr(self.result.distance),
        ))
    }
}

/// Where a 2D curve is closest to a point.
///
/// `target` is the curve that was projected onto, as the exact object the
/// method was called on, and `query_point` is the point that was projected.
#[pyclass(name = "CurveProjection2D", module = "geomcore", frozen)]
pub(crate) struct PyCurveProjection2D {
    pub(crate) result: CurveProjection2D,
    pub(crate) target: Py<PyAny>,
    pub(crate) query_point: Point2D,
}

impl PyCurveProjection2D {
    pub(crate) fn new(
        result: CurveProjection2D,
        target: &Bound<'_, impl pyo3::PyClass>,
        query_point: Point2D,
    ) -> Self {
        PyCurveProjection2D {
            result,
            target: target.clone().into_any().unbind(),
            query_point,
        }
    }
}

#[pymethods]
impl PyCurveProjection2D {
    /// The parameter of the closest point.
    #[getter]
    fn parameter(&self) -> f64 {
        self.result.parameter
    }

    /// The closest point: the curve evaluated at `parameter`.
    #[getter]
    fn point(&self) -> PyPoint2D {
        PyPoint2D(self.result.point)
    }

    /// The distance from `query_point` to the curve.
    #[getter]
    fn distance(&self) -> f64 {
        self.result.distance
    }

    /// The curve that was projected onto.
    #[getter]
    fn target(&self, py: Python<'_>) -> Py<PyAny> {
        self.target.clone_ref(py)
    }

    /// The point that was projected.
    #[getter]
    fn query_point(&self) -> PyPoint2D {
        PyPoint2D(self.query_point)
    }

    fn __eq__(&self, py: Python<'_>, other: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(self.result == other.result
            && self.query_point == other.query_point
            && same_operand(py, &self.target, &other.target)?)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "CurveProjection2D(target={}, query_point={}, parameter={}, point={}, distance={})",
            self.target.bind(py).get_type().name()?,
            point2_repr(self.query_point),
            float_repr(self.result.parameter),
            point2_repr(self.result.point),
            float_repr(self.result.distance),
        ))
    }
}

/// Where a surface is closest to a point.
///
/// `target` is the surface that was projected onto, as the exact object the
/// method was called on, and `query_point` is the point that was projected.
#[pyclass(name = "SurfaceProjection", module = "geomcore", frozen)]
pub(crate) struct PySurfaceProjection {
    pub(crate) result: SurfaceProjection,
    pub(crate) target: Py<PyAny>,
    pub(crate) query_point: Point3D,
}

impl PySurfaceProjection {
    pub(crate) fn new(
        result: SurfaceProjection,
        target: &Bound<'_, impl pyo3::PyClass>,
        query_point: Point3D,
    ) -> Self {
        PySurfaceProjection {
            result,
            target: target.clone().into_any().unbind(),
            query_point,
        }
    }
}

#[pymethods]
impl PySurfaceProjection {
    /// The `u` parameter of the closest point.
    #[getter]
    fn u(&self) -> f64 {
        self.result.u
    }

    /// The `v` parameter of the closest point.
    #[getter]
    fn v(&self) -> f64 {
        self.result.v
    }

    /// The closest point: the surface evaluated at `(u, v)`.
    #[getter]
    fn point(&self) -> PyPoint3D {
        PyPoint3D(self.result.point)
    }

    /// The distance from `query_point` to the surface.
    #[getter]
    fn distance(&self) -> f64 {
        self.result.distance
    }

    /// The surface that was projected onto.
    #[getter]
    fn target(&self, py: Python<'_>) -> Py<PyAny> {
        self.target.clone_ref(py)
    }

    /// The point that was projected.
    #[getter]
    fn query_point(&self) -> PyPoint3D {
        PyPoint3D(self.query_point)
    }

    fn __eq__(&self, py: Python<'_>, other: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(self.result == other.result
            && self.query_point == other.query_point
            && same_operand(py, &self.target, &other.target)?)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "SurfaceProjection(target={}, query_point={}, u={}, v={}, point={}, distance={})",
            self.target.bind(py).get_type().name()?,
            point3_repr(self.query_point),
            float_repr(self.result.u),
            float_repr(self.result.v),
            point3_repr(self.result.point),
            float_repr(self.result.distance),
        ))
    }
}

// ---------------------------------------------------------------------------
// Curve–curve extrema
// ---------------------------------------------------------------------------

/// A local minimum of the distance between two curves.
#[pyclass(name = "CurveCurveExtremum", module = "geomcore", eq, frozen)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyCurveCurveExtremum(pub(crate) CurveCurveExtremum);

#[pymethods]
impl PyCurveCurveExtremum {
    /// The parameter on the first curve.
    #[getter]
    fn first_parameter(&self) -> f64 {
        self.0.first_parameter
    }

    /// The parameter on the second curve.
    #[getter]
    fn second_parameter(&self) -> f64 {
        self.0.second_parameter
    }

    /// The distance between the curves' points at the two parameters.
    #[getter]
    fn distance(&self) -> f64 {
        self.0.distance
    }

    fn __repr__(&self) -> String {
        format!(
            "CurveCurveExtremum(first_parameter={}, second_parameter={}, distance={})",
            float_repr(self.0.first_parameter),
            float_repr(self.0.second_parameter),
            float_repr(self.0.distance),
        )
    }
}

/// The local minima of the distance between two curves, closest first,
/// with whether the search proved there is no other.
///
/// `first` and `second` are the curves, in the order of the call.
#[pyclass(name = "CurveCurveExtrema", module = "geomcore", frozen)]
pub(crate) struct PyCurveCurveExtrema {
    pub(crate) result: CurveCurveExtrema,
    pub(crate) first: Py<PyAny>,
    pub(crate) second: Py<PyAny>,
}

#[pymethods]
impl PyCurveCurveExtrema {
    /// The number of minima; an extrema result is falsy when there is none.
    fn __len__(&self) -> usize {
        self.result.minima().len()
    }

    /// The minima, closest first.
    #[getter]
    fn minima<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            self.result
                .minima()
                .iter()
                .map(|m| PyCurveCurveExtremum(*m)),
        )
    }

    /// Whether the search proved there is no other minimum.
    #[getter]
    fn completeness(&self) -> PyCompleteness {
        PyCompleteness::curve_curve(self.result.completeness())
    }

    /// The first curve.
    #[getter]
    fn first(&self, py: Python<'_>) -> Py<PyAny> {
        self.first.clone_ref(py)
    }

    /// The second curve.
    #[getter]
    fn second(&self, py: Python<'_>) -> Py<PyAny> {
        self.second.clone_ref(py)
    }

    /// Whether the search proved there is no other minimum.
    fn is_complete(&self) -> bool {
        self.result.is_complete()
    }

    fn __eq__(&self, py: Python<'_>, other: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(self.result == other.result
            && same_operand(py, &self.first, &other.first)?
            && same_operand(py, &self.second, &other.second)?)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "CurveCurveExtrema(first={}, second={}, completeness={}, minima={})",
            self.first.bind(py).get_type().name()?,
            self.second.bind(py).get_type().name()?,
            self.completeness().status.repr(),
            list_repr(self.result.minima(), |m| PyCurveCurveExtremum(*m)
                .__repr__()),
        ))
    }
}
