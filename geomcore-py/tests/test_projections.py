"""Projection objects: `project_point`, `project_points` and `extrema`."""

import math

import pytest

from geomcore import (
    CurveProjection,
    CurveProjection2D,
    Point2D,
    Point3D,
    SurfaceProjection,
    Vector2D,
    Vector3D,
)
from geomcore.curves import (
    BSplineCurve3D,
    Circle2D,
    Circle3D,
    Line2D,
    TrimmedCurve2D,
    TrimmedCurve3D,
)
from geomcore.surfaces import BSplineSurface, Plane, Sphere


def _assert_points_close(a, b):
    assert a.x == pytest.approx(b.x)
    assert a.y == pytest.approx(b.y)
    if hasattr(a, "z"):
        assert a.z == pytest.approx(b.z)


def test_curve_projection():
    circle = Circle3D(Point3D.origin(), Vector3D.unit_z(), 2.0)
    query = Point3D(3.0, 0.0, 1.0)
    projection = circle.project_point(query)
    assert isinstance(projection, CurveProjection)
    _assert_points_close(projection.point, circle.eval_point(projection.parameter))
    assert projection.distance == pytest.approx(2.0**0.5)
    assert projection.target is circle
    assert projection.query_point == query


def test_curve_projection_2d():
    line = Line2D(Point2D.origin(), Vector2D(1.0, 0.0))
    query = Point2D(2.0, 1.0)
    projection = line.project_point(query)
    assert isinstance(projection, CurveProjection2D)
    assert projection.parameter == pytest.approx(2.0)
    _assert_points_close(projection.point, line.eval_point(projection.parameter))
    assert projection.distance == pytest.approx(1.0)
    assert projection.target is line
    assert projection.query_point == query


def test_circle_projection_2d():
    circle = Circle2D(Point2D.origin(), 2.0)
    query = Point2D(0.0, 3.0)
    projection = circle.project_point(query)
    assert isinstance(projection, CurveProjection2D)
    assert projection.parameter == pytest.approx(math.pi / 2)
    _assert_points_close(projection.point, Point2D(0.0, 2.0))
    assert projection.distance == pytest.approx(1.0)
    assert projection.target is circle
    assert projection.query_point == query


def test_project_points_on_2d_curves():
    queries = [Point2D(3.0, 0.0), Point2D(0.0, -1.0)]
    line = Line2D(Point2D.origin(), Vector2D(1.0, 0.0))
    circle = Circle2D(Point2D.origin(), 2.0)
    arc = TrimmedCurve2D(circle, 0.0, math.pi / 2)
    for curve in [line, circle, arc]:
        projections = curve.project_points(queries)
        assert [type(p) for p in projections] == [CurveProjection2D] * 2
        for projection, query in zip(projections, queries):
            assert projection == curve.project_point(query)
            assert projection.target is curve
            assert projection.query_point == query
            _assert_points_close(projection.point, curve.eval_point(projection.parameter))
    # The arc ends at angle 0: the point below the circle projects there.
    assert arc.project_point(Point2D(0.0, -1.0)).parameter == 0.0


def test_extrema_on_2d_curves():
    circle = Circle2D(Point2D.origin(), 2.0)
    extrema = circle.extrema(Point2D(1.0, 0.0))
    assert [type(e) for e in extrema] == [CurveProjection2D] * len(extrema)
    assert extrema[0].distance == pytest.approx(1.0)
    assert all(e.target is circle for e in extrema)


def test_surface_projection():
    sphere = Sphere(Point3D.origin(), 2.0)
    query = Point3D(0.0, 0.0, 5.0)
    projection = sphere.project_point(query)
    assert isinstance(projection, SurfaceProjection)
    _assert_points_close(projection.point, sphere.eval_point(projection.u, projection.v))
    assert projection.distance == pytest.approx(3.0)
    assert projection.target is sphere
    assert projection.query_point == query


def test_bspline_curve_projection():
    curve = BSplineCurve3D(
        1, [Point3D(0.0, 0.0, 0.0), Point3D(2.0, 0.0, 0.0)], [0.0, 1.0], [2, 2], False
    )
    projection = curve.project_point(Point3D(1.0, 1.0, 0.0))
    _assert_points_close(projection.point, curve.eval_point(projection.parameter))
    assert projection.target is curve
    extrema = curve.extrema(Point3D(1.0, 1.0, 0.0))
    assert all(isinstance(e, CurveProjection) for e in extrema)
    assert all(e.target is curve for e in extrema)


def test_bspline_surface_projection():
    surface = BSplineSurface(
        1,
        1,
        [
            [Point3D(0.0, 0.0, 0.0), Point3D(0.0, 2.0, 0.0)],
            [Point3D(2.0, 0.0, 0.0), Point3D(2.0, 2.0, 0.0)],
        ],
        [0.0, 1.0],
        [2, 2],
        [0.0, 1.0],
        [2, 2],
        False,
        False,
    )
    projection = surface.project_point(Point3D(1.0, 1.0, 1.0))
    _assert_points_close(projection.point, surface.eval_point(projection.u, projection.v))
    assert projection.target is surface
    extrema = surface.extrema(Point3D(1.0, 1.0, 1.0))
    assert all(isinstance(e, SurfaceProjection) and e.target is surface for e in extrema)


def test_trimmed_curve_extrema_hold_the_trimmed_curve():
    circle = Circle3D(Point3D.origin(), Vector3D.unit_z(), 1.0)
    arc = TrimmedCurve3D(circle, 0.0, 1.0)
    extrema = arc.extrema(Point3D(3.0, 0.0, 0.0))
    assert extrema
    assert all(e.target is arc for e in extrema)


def test_project_points_returns_a_list_of_projection_objects():
    plane = Plane(Point3D.origin(), Vector3D.unit_z())
    queries = [Point3D(1.0, 2.0, 3.0), Point3D(0.0, 0.0, 2.0)]
    projections = plane.project_points(queries)
    assert isinstance(projections, list)
    assert [type(p) for p in projections] == [SurfaceProjection] * 2
    for projection, query in zip(projections, queries):
        assert projection.target is plane
        assert projection.query_point == query
        assert projection == plane.project_point(query)

    circle = Circle3D(Point3D.origin(), Vector3D.unit_z(), 2.0)
    assert [type(p) for p in circle.project_points(queries)] == [CurveProjection] * 2


def test_equality_and_repr():
    circle = Circle3D(Point3D.origin(), Vector3D.unit_z(), 2.0)
    query = Point3D(3.0, 0.0, 0.0)
    projection = circle.project_point(query)
    assert projection == circle.project_point(query)
    assert projection != circle.project_point(Point3D(0.0, 4.0, 0.0))
    assert repr(projection) == (
        "CurveProjection(target=Circle3D, query_point=Point3D(3.0, 0.0, 0.0), "
        "parameter=0.0, point=Point3D(2.0, 0.0, 0.0), distance=1.0)"
    )
    assert repr(projection) == repr(circle.project_point(query))
