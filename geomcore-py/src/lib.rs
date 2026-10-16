//! Python bindings for the `geomcore` geometric kernel.
//!
//! The extension module mirrors the Rust crate's namespaces: value types live
//! at the package root (`geomcore`), curves under `geomcore.curves`, and
//! surfaces under `geomcore.surfaces`. Fallible operations raise
//! the `GeomcoreError` subclass matching the Rust error type (see `errors`);
//! `GeomcoreError` is a `ValueError`.

mod errors;

use errors::invalid_argument;

use pyo3::prelude::*;
use pyo3::types::PyModule;

use geomcore::Tolerance;
use geomcore::curves::PointParametrization;

/// The tolerance given, or the default.
pub(crate) fn tolerance(tolerance: Option<PyTolerance>) -> Tolerance {
    tolerance.map_or(Tolerance::DEFAULT, |given| given.0)
}

/// Checks a curve derivative order lies in `1..=max`.
pub(crate) fn check_curve_order(order: u32, max: u32) -> PyResult<()> {
    if (1..=max).contains(&order) {
        Ok(())
    } else if order == 0 {
        Err(invalid_argument(
            "order must be >= 1 (use eval_point for order 0)",
        ))
    } else {
        Err(invalid_argument(format!(
            "order {order} is not supported (at most {max})"
        )))
    }
}

/// Checks a surface derivative order satisfies `1 <= du + dv <= 2`.
pub(crate) fn check_surface_order(du: u32, dv: u32) -> PyResult<()> {
    match du.checked_add(dv) {
        Some(1 | 2) => Ok(()),
        Some(0) => Err(invalid_argument(
            "du + dv must be >= 1 (use eval_point for the (0, 0) order)",
        )),
        _ => Err(invalid_argument(format!(
            "order du={du}, dv={dv} is not supported (du + dv must be <= 2)"
        ))),
    }
}

pub(crate) fn parse_parametrization(s: &str) -> PyResult<PointParametrization> {
    match s {
        "chordal" => Ok(PointParametrization::Chordal),
        "centripetal" => Ok(PointParametrization::Centripetal),
        _ => Err(invalid_argument(
            "parametrization must be 'chordal' or 'centripetal'",
        )),
    }
}

mod curves;
mod intersect;
mod numeric;
mod results;
mod surfaces;
mod values;

use curves::{
    PyBSplineCurve3D, PyCircle2D, PyCircle3D, PyEllipse3D, PyHyperbola3D, PyLine2D, PyLine3D,
    PyParabola3D, PyTrimmedCurve2D, PyTrimmedCurve3D,
};
use numeric::{py_curve_curve_extrema, py_parametrize_numeric};
use results::{
    PyCompleteness, PyCompletenessStatus, PyCurveCurveExtrema, PyCurveCurveExtremum,
    PyCurveCurveHit, PyCurveCurveHit2D, PyCurveCurveIntersection, PyCurveCurveIntersection2D,
    PyCurveCurveOverlap, PyCurveCurveRegion, PyCurveProjection, PyCurveProjection2D,
    PyCurveSurfaceHit, PyCurveSurfaceIntersection, PyCurveSurfaceRegion, PyIncompleteReason,
    PyIntersectionCurve, PyIntersectionMethod, PySurfaceProjection, PySurfaceSurfaceIntersection,
    PySurfaceSurfaceRegion,
};
use surfaces::{PyBSplineSurface, PyCone, PyCylinder, PyPlane, PySphere, PyTorus};
use values::{
    PyAxis2D, PyAxis3D, PyBoundingBox3D, PyDiscretization2D, PyDiscretization3D, PyFrame2D,
    PyFrame3D, PyPoint2D, PyPoint3D, PyTolerance, PyTransform, PyVector2D, PyVector3D,
};

// ---------------------------------------------------------------------------
// Module wiring
// ---------------------------------------------------------------------------

#[pymodule(name = "geomcore")]
fn geomcore_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    errors::register(m)?;
    m.add_class::<PyPoint3D>()?;
    m.add_class::<PyPoint2D>()?;
    m.add_class::<PyVector3D>()?;
    m.add_class::<PyVector2D>()?;
    m.add_class::<PyAxis2D>()?;
    m.add_class::<PyAxis3D>()?;
    m.add_class::<PyFrame2D>()?;
    m.add_class::<PyFrame3D>()?;
    m.add_class::<PyTolerance>()?;
    m.add_class::<PyTransform>()?;
    m.add_class::<PyBoundingBox3D>()?;
    m.add_class::<PyDiscretization2D>()?;
    m.add_class::<PyDiscretization3D>()?;

    m.add_class::<PyIntersectionMethod>()?;
    m.add_class::<PyCompletenessStatus>()?;
    m.add_class::<PyIncompleteReason>()?;
    m.add_class::<PyCompleteness>()?;
    m.add_class::<PyCurveCurveRegion>()?;
    m.add_class::<PyCurveSurfaceRegion>()?;
    m.add_class::<PySurfaceSurfaceRegion>()?;
    m.add_class::<PyCurveCurveHit>()?;
    m.add_class::<PyCurveCurveHit2D>()?;
    m.add_class::<PyCurveSurfaceHit>()?;
    m.add_class::<PyCurveCurveOverlap>()?;
    m.add_class::<PyIntersectionCurve>()?;
    m.add_class::<PyCurveCurveIntersection>()?;
    m.add_class::<PyCurveCurveIntersection2D>()?;
    m.add_class::<PyCurveSurfaceIntersection>()?;
    m.add_class::<PySurfaceSurfaceIntersection>()?;
    m.add_class::<PyCurveProjection>()?;
    m.add_class::<PyCurveProjection2D>()?;
    m.add_class::<PySurfaceProjection>()?;
    m.add_class::<PyCurveCurveExtremum>()?;
    m.add_class::<PyCurveCurveExtrema>()?;

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

    m.add_function(wrap_pyfunction!(py_parametrize_numeric, m)?)?;
    m.add_function(wrap_pyfunction!(py_curve_curve_extrema, m)?)?;

    Ok(())
}
