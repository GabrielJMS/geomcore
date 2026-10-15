"""Tests for the binding gaps closed against the Rust API."""

import math

import pytest

from geomcore import (
    Axis2D,
    Axis3D,
    Frame2D,
    Frame3D,
    Point2D,
    Point3D,
    Vector2D,
    Vector3D,
)
from geomcore.curves import (
    Circle2D,
    Circle3D,
    Ellipse3D,
    Hyperbola3D,
    Line2D,
    Line3D,
    Parabola3D,
)
from geomcore.surfaces import Cone, Cylinder, Plane, Sphere, Torus

Z = Vector3D(0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Group 1: Axis2D, Frame2D, frame(), axis(), center(), apex()
# ---------------------------------------------------------------------------


def test_axis_2d():
    axis = Axis2D(Point2D(1.0, 2.0), Vector2D(0.0, 3.0))
    assert axis.origin() == Point2D(1.0, 2.0)
    assert axis.direction() == Vector2D(0.0, 1.0)
    assert repr(axis).startswith("Axis2D(origin=")
    with pytest.raises(ValueError):
        Axis2D(Point2D(0.0, 0.0), Vector2D(0.0, 0.0))


def test_frame_2d():
    world = Frame2D.world()
    assert world.origin() == Point2D.origin()
    assert world.x_direction() == Vector2D(1.0, 0.0)
    assert world.y_direction() == Vector2D(0.0, 1.0)
    assert world.is_direct()
    frame = Frame2D(Point2D(1.0, 1.0), Vector2D(2.0, 0.0), Vector2D(0.0, -1.0))
    assert not frame.is_direct()
    assert frame.x_direction() == Vector2D(1.0, 0.0)
    assert frame.y_direction() == Vector2D(0.0, -1.0)
    from_x = Frame2D.from_x(Point2D(0.0, 0.0), Vector2D(0.0, 5.0))
    assert from_x.y_direction() == Vector2D(-1.0, 0.0)
    with pytest.raises(ValueError):
        Frame2D(Point2D(0.0, 0.0), Vector2D(1.0, 0.0), Vector2D(1.0, 0.0))
    assert repr(world).startswith("Frame2D(origin=")


def test_frame_on_curves():
    center = Point3D(1.0, 2.0, 3.0)
    expected = Frame3D.from_z(center, Z)
    circle = Circle3D(center, Z, 2.0)
    assert circle.frame().origin() == center
    assert circle.frame().z_direction() == Z
    assert circle.frame() == Circle3D.from_frame(circle.frame(), 2.0).frame()
    ellipse = Ellipse3D.from_frame(expected, 3.0, 2.0)
    assert ellipse.frame() == expected
    parabola = Parabola3D.from_frame(expected, 1.5)
    assert parabola.frame() == expected
    hyperbola = Hyperbola3D.from_frame(expected, 3.0, 2.0)
    assert hyperbola.frame() == expected
    assert Circle2D(Point2D(1.0, 1.0), 2.0).frame() == Frame2D(
        Point2D(1.0, 1.0), Vector2D(1.0, 0.0), Vector2D(0.0, 1.0)
    )


def test_frame_on_surfaces():
    frame = Frame3D.from_z(Point3D(1.0, 2.0, 3.0), Z)
    assert Plane.from_frame(frame).frame() == frame
    assert Cylinder.from_frame(frame, 2.0).frame() == frame
    assert Cone.from_frame(frame, 0.5, 2.0).frame() == frame
    assert Sphere.from_frame(frame, 2.0).frame() == frame
    assert Torus.from_frame(frame, 3.0, 1.0).frame() == frame


def test_axis_on_lines_and_cylinder():
    axis = Axis3D(Point3D(1.0, 0.0, 0.0), Z)
    assert Line3D.from_axis(axis).axis() == axis
    axis_2d = Axis2D(Point2D(1.0, 0.0), Vector2D(0.0, 1.0))
    assert Line2D.from_axis(axis_2d).axis() == axis_2d
    assert Line2D(Point2D(1.0, 0.0), Vector2D(0.0, 2.0)).axis() == axis_2d
    cylinder = Cylinder.from_frame(Frame3D.from_z(Point3D(1.0, 0.0, 0.0), Z), 1.0)
    assert cylinder.axis() == axis


def test_center_on_torus_and_apex_on_parabola():
    frame = Frame3D.from_z(Point3D(1.0, 2.0, 3.0), Z)
    assert Torus.from_frame(frame, 3.0, 1.0).center() == Point3D(1.0, 2.0, 3.0)
    assert Parabola3D.from_frame(frame, 1.5).apex() == Point3D(1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# Group 2: period, eval_d1/eval_d2, 2D bounds, from_two_points, extrema
# ---------------------------------------------------------------------------

from geomcore.curves import BSplineCurve3D, TrimmedCurve2D, TrimmedCurve3D
from geomcore.surfaces import BSplineSurface


def curves_3d():
    frame = Frame3D.from_z(Point3D(0.0, 0.0, 0.0), Z)
    return [
        Line3D(Point3D(0.0, 0.0, 0.0), Vector3D(1.0, 0.0, 0.0)),
        Circle3D(Point3D(0.0, 0.0, 0.0), Z, 2.0),
        Ellipse3D.from_frame(frame, 3.0, 2.0),
        Parabola3D.from_frame(frame, 1.5),
        Hyperbola3D.from_frame(frame, 3.0, 2.0),
        BSplineCurve3D(
            2,
            [Point3D(0.0, 0.0, 0.0), Point3D(1.0, 1.0, 0.0), Point3D(2.0, 0.0, 1.0)],
            [0.0, 1.0],
            [3, 3],
            False,
        ),
        TrimmedCurve3D(Circle3D(Point3D(0.0, 0.0, 0.0), Z, 2.0), 0.0, 1.0),
    ]


def curves_2d():
    return [
        Line2D(Point2D(0.0, 0.0), Vector2D(1.0, 0.0)),
        Circle2D(Point2D(0.0, 0.0), 2.0),
        TrimmedCurve2D(Circle2D(Point2D(0.0, 0.0), 2.0), 0.0, 1.0),
    ]


def test_period_and_is_periodic_on_curves():
    line, circle, ellipse, parabola, hyperbola, spline, trimmed = curves_3d()
    assert line.period() is None and not line.is_periodic()
    assert circle.period() == pytest.approx(math.tau) and circle.is_periodic()
    assert ellipse.period() == pytest.approx(math.tau)
    assert parabola.period() is None and hyperbola.period() is None
    assert spline.period() is None
    assert trimmed.period() is None and not trimmed.is_periodic()
    line_2d, circle_2d, trimmed_2d = curves_2d()
    assert line_2d.period() is None and not line_2d.is_periodic()
    assert circle_2d.period() == pytest.approx(math.tau) and circle_2d.is_periodic()
    assert trimmed_2d.period() is None


def test_u_and_v_period_on_surfaces():
    origin = Point3D(0.0, 0.0, 0.0)
    assert Plane(origin, Z).u_period() is None
    assert Plane(origin, Z).v_period() is None
    assert Cylinder(origin, Z, 1.0).u_period() == pytest.approx(math.tau)
    assert Cylinder(origin, Z, 1.0).v_period() is None
    assert Sphere(origin, 1.0).u_period() == pytest.approx(math.tau)
    assert Sphere(origin, 1.0).v_period() is None
    assert Torus(origin, Z, 3.0, 1.0).v_period() == pytest.approx(math.tau)
    assert Cone(origin, Z, 0.5, 2.0).u_period() == pytest.approx(math.tau)


@pytest.mark.parametrize("curve", curves_3d())
def test_eval_d1_d2_on_3d_curves(curve):
    u = 0.5
    point, first = curve.eval_d1(u)
    assert point == curve.eval_point(u)
    assert first == curve.eval_derivative(u, 1)
    point, first, second = curve.eval_d2(u)
    assert point == curve.eval_point(u)
    assert first == curve.eval_derivative(u, 1)
    assert second == curve.eval_derivative(u, 2)


@pytest.mark.parametrize("curve", curves_2d())
def test_eval_d1_d2_on_2d_curves(curve):
    u = 0.5
    point, first = curve.eval_d1(u)
    assert point == curve.eval_point(u)
    assert first == curve.eval_derivative(u, 1)
    point, first, second = curve.eval_d2(u)
    assert second == curve.eval_derivative(u, 2)


def test_eval_d1_d2_on_surfaces():
    origin = Point3D(0.0, 0.0, 0.0)
    surfaces = [
        Plane(origin, Z),
        Cylinder(origin, Z, 1.0),
        Cone(origin, Z, 0.5, 2.0),
        Sphere(origin, 2.0),
        Torus(origin, Z, 3.0, 1.0),
    ]
    for surface in surfaces:
        u, v = 0.4, 0.3
        point, (du, dv) = surface.eval_d1(u, v)
        assert point == surface.eval_point(u, v)
        assert du == surface.eval_derivative(u, v, 1, 0)
        assert dv == surface.eval_derivative(u, v, 0, 1)
        point, (du, dv), (duu, duv, dvv) = surface.eval_d2(u, v)
        assert duu == surface.eval_derivative(u, v, 2, 0)
        assert duv == surface.eval_derivative(u, v, 1, 1)
        assert dvv == surface.eval_derivative(u, v, 0, 2)


def test_bounds_on_2d_curves():
    line, circle, _ = curves_2d()
    assert line.bounds() == (-math.inf, math.inf)
    assert circle.bounds() == pytest.approx((0.0, math.tau))


def test_line_2d_from_two_points():
    line = Line2D.from_two_points(Point2D(1.0, 1.0), Point2D(1.0, 5.0))
    assert line.origin() == Point2D(1.0, 1.0)
    assert line.direction() == Vector2D(0.0, 1.0)
    with pytest.raises(ValueError):
        Line2D.from_two_points(Point2D(1.0, 1.0), Point2D(1.0, 1.0))


def test_extrema_on_lines_and_circles():
    line = Line3D(Point3D(0.0, 0.0, 0.0), Vector3D(1.0, 0.0, 0.0))
    query = Point3D(2.0, 3.0, 0.0)
    (minimum,) = line.extrema(query)
    assert minimum.point == Point3D(2.0, 0.0, 0.0)
    assert minimum.distance == pytest.approx(3.0)
    assert minimum.target is line

    circle = Circle3D(Point3D(0.0, 0.0, 0.0), Z, 2.0)
    results = circle.extrema(Point3D(3.0, 0.0, 0.0))
    assert results[0].distance == pytest.approx(1.0)
    assert results[0].query_point == Point3D(3.0, 0.0, 0.0)

    line_2d = Line2D(Point2D(0.0, 0.0), Vector2D(1.0, 0.0))
    (minimum_2d,) = line_2d.extrema(Point2D(2.0, 3.0))
    assert minimum_2d.point == Point2D(2.0, 0.0)
    assert minimum_2d.target is line_2d

    circle_2d = Circle2D(Point2D(0.0, 0.0), 2.0)
    assert circle_2d.extrema(Point2D(3.0, 0.0))[0].distance == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Group 3: to_bspline, point_count
# ---------------------------------------------------------------------------


def test_to_bspline_on_elementary_curves_and_surfaces():
    frame = Frame3D.from_z(Point3D(0.0, 0.0, 0.0), Z)
    circle = Circle3D(Point3D(0.0, 0.0, 0.0), Z, 2.0)
    ellipse = Ellipse3D.from_frame(frame, 3.0, 2.0)
    for curve in (circle, ellipse):
        spline = curve.to_bspline()
        assert isinstance(spline, BSplineCurve3D)
        first, last = spline.bounds()
        middle = 0.5 * (first + last)
        assert spline.eval_point(middle).distance(
            curve.eval_point(curve.parameter_of(spline.eval_point(middle)))
        ) == pytest.approx(0.0, abs=1e-9)
    # Unbounded curves convert over a range, through a trimmed curve.
    for curve in (
        Line3D(Point3D(0.0, 0.0, 0.0), Vector3D(1.0, 0.0, 0.0)),
        Parabola3D.from_frame(frame, 1.5),
        Hyperbola3D.from_frame(frame, 3.0, 2.0),
    ):
        spline = TrimmedCurve3D(curve, 0.0, 1.0).to_bspline()
        assert spline.eval_point(0.5).distance(curve.eval_point(0.5)) < 1e-9
    assert isinstance(Sphere(Point3D(0.0, 0.0, 0.0), 1.0).to_bspline(), BSplineSurface)
    assert isinstance(Torus.from_frame(frame, 3.0, 1.0).to_bspline(), BSplineSurface)
    for surface in (
        Plane.from_frame(frame),
        Cylinder.from_frame(frame, 1.0),
        Cone.from_frame(frame, 0.5, 1.0),
    ):
        patch = surface.to_bspline_patch((0.0, 1.0), (0.0, 1.0))
        assert isinstance(patch, BSplineSurface)


def test_point_count_matches_discretize():
    circle = Circle3D(Point3D(0.0, 0.0, 0.0), Z, 2.0)
    assert circle.point_count(0.01) == len(circle.discretize(0.01).points)
    assert circle.point_count(0.01, 0.1) == len(circle.discretize(0.01, 0.1).points)
    ellipse = Ellipse3D.from_frame(Frame3D.from_z(Point3D(0.0, 0.0, 0.0), Z), 3.0, 2.0)
    assert ellipse.point_count(0.01) == len(ellipse.discretize(0.01).points)
    spline = curves_3d()[5]
    assert spline.point_count(0.01) == len(spline.discretize(0.01).points)
    trimmed = TrimmedCurve3D(circle, 0.0, 1.0)
    assert trimmed.point_count(0.01) == len(trimmed.discretize(0.01).points)
    circle_2d = Circle2D(Point2D(0.0, 0.0), 2.0)
    assert circle_2d.point_count(0.01) == len(circle_2d.discretize(0.01).points)
    trimmed_2d = TrimmedCurve2D(circle_2d, 0.0, 1.0)
    assert trimmed_2d.point_count(0.01) == len(trimmed_2d.discretize(0.01).points)
    with pytest.raises(ValueError):
        circle.point_count(0.0)
