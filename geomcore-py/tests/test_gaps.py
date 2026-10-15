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
