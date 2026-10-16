//! Python exception hierarchy mirroring the Rust error types.
//!
//! Every exception derives from `GeomcoreError`, which itself derives from
//! `ValueError`, so existing `except ValueError` handlers keep working while
//! callers can catch the precise failure.

use pyo3::PyErr;
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

create_exception!(
    geomcore,
    GeomcoreError,
    PyValueError,
    "Base class of every error raised by geomcore. Subclasses `ValueError`."
);
create_exception!(
    geomcore,
    ConstructionError,
    GeomcoreError,
    "A geometric object could not be built from the given arguments (for example a negative radius or a zero-length direction)."
);
create_exception!(
    geomcore,
    FrameConstructionError,
    ConstructionError,
    "A frame could not be built, for example from parallel or zero directions."
);
create_exception!(
    geomcore,
    TrimmedCurveConstructionError,
    ConstructionError,
    "A trimmed curve could not be built, for example from an invalid parameter range."
);
create_exception!(
    geomcore,
    KnotInsertionError,
    GeomcoreError,
    "A knot could not be inserted into a B-spline."
);
create_exception!(
    geomcore,
    KnotRemovalError,
    GeomcoreError,
    "A knot could not be removed from a B-spline."
);
create_exception!(
    geomcore,
    DegreeElevationError,
    GeomcoreError,
    "The degree of a B-spline could not be elevated."
);
create_exception!(
    geomcore,
    BSplineSplitError,
    GeomcoreError,
    "A B-spline could not be split at the given parameter."
);
create_exception!(
    geomcore,
    BSplineConversionError,
    GeomcoreError,
    "A curve or surface patch could not be converted to a B-spline."
);
create_exception!(
    geomcore,
    ParametrizeError,
    GeomcoreError,
    "A curve could not be re-parametrized as requested."
);
create_exception!(
    geomcore,
    SurfaceIsoError,
    GeomcoreError,
    "An iso-parametric curve of a surface could not be extracted."
);
create_exception!(
    geomcore,
    DiscretizationCriteriaError,
    GeomcoreError,
    "Discretization criteria were invalid, for example a non-positive deviation."
);

/// Registers every exception class on the extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("GeomcoreError", py.get_type::<GeomcoreError>())?;
    m.add("ConstructionError", py.get_type::<ConstructionError>())?;
    m.add(
        "FrameConstructionError",
        py.get_type::<FrameConstructionError>(),
    )?;
    m.add(
        "TrimmedCurveConstructionError",
        py.get_type::<TrimmedCurveConstructionError>(),
    )?;
    m.add("KnotInsertionError", py.get_type::<KnotInsertionError>())?;
    m.add("KnotRemovalError", py.get_type::<KnotRemovalError>())?;
    m.add(
        "DegreeElevationError",
        py.get_type::<DegreeElevationError>(),
    )?;
    m.add("BSplineSplitError", py.get_type::<BSplineSplitError>())?;
    m.add(
        "BSplineConversionError",
        py.get_type::<BSplineConversionError>(),
    )?;
    m.add("ParametrizeError", py.get_type::<ParametrizeError>())?;
    m.add("SurfaceIsoError", py.get_type::<SurfaceIsoError>())?;
    m.add(
        "DiscretizationCriteriaError",
        py.get_type::<DiscretizationCriteriaError>(),
    )?;
    Ok(())
}

/// A Rust error type with a dedicated Python exception class.
pub(crate) trait IntoPythonError: std::fmt::Display {
    fn into_python_error(self) -> PyErr;
}

/// Converts a Rust error into its Python exception, for `map_err`.
pub(crate) fn python_error<E: IntoPythonError>(error: E) -> PyErr {
    error.into_python_error()
}

/// A `ValueError` for an invalid argument that has no Rust error type.
pub(crate) fn invalid_argument(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

macro_rules! map_error {
    ($exception:ident: $($rust_error:ty),+ $(,)?) => {
        $(
            impl IntoPythonError for $rust_error {
                fn into_python_error(self) -> PyErr {
                    $exception::new_err(self.to_string())
                }
            }
        )+
    };
}

map_error!(ConstructionError:
    geomcore::BSplineConstructionError,
    geomcore::CircleConstructionError,
    geomcore::EllipseConstructionError,
    geomcore::HyperbolaConstructionError,
    geomcore::LineConstructionError,
    geomcore::ParabolaConstructionError,
    geomcore::surfaces::ConeConstructionError,
    geomcore::surfaces::CylinderConstructionError,
    geomcore::surfaces::PlaneConstructionError,
    geomcore::surfaces::SphereConstructionError,
    geomcore::surfaces::TorusConstructionError,
);
map_error!(FrameConstructionError: geomcore::FrameConstructionError);
map_error!(TrimmedCurveConstructionError: geomcore::TrimmedCurveConstructionError);
map_error!(KnotInsertionError: geomcore::KnotInsertionError);
map_error!(KnotRemovalError: geomcore::KnotRemovalError);
map_error!(DegreeElevationError: geomcore::DegreeElevationError);
map_error!(BSplineSplitError: geomcore::BSplineSplitError);
map_error!(BSplineConversionError: geomcore::BSplineConversionError);
map_error!(ParametrizeError: geomcore::ParametrizeError);
map_error!(SurfaceIsoError: geomcore::surfaces::SurfaceIsoError);
map_error!(DiscretizationCriteriaError: geomcore::DiscretizationCriteriaError);
