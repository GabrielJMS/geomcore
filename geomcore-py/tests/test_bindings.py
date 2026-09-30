"""Behavioral tests for the geomcore Python bindings."""

import math

import pytest

import geomcore
from geomcore import Axis3D, Frame3D, Point2D, Point3D, Tolerance, Transform, Vector2D, Vector3D
from geomcore.curves import BSplineCurve3D, Circle3D, Line2D
from geomcore.surfaces import Cylinder, Sphere


def test_readme_snippet():
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    point = circle.eval_point(math.pi / 4)
    assert point.x == pytest.approx(2.0 * math.cos(math.pi / 4))
    assert point.y == pytest.approx(2.0 * math.sin(math.pi / 4))
    assert point.z == pytest.approx(0.0)


def test_submodule_import_forms():
    # Both attribute access and from-import must work.
    assert geomcore.curves.Circle3D is Circle3D
    from geomcore.curves import Circle3D as C2

    assert C2 is Circle3D


def test_eval_points_bulk():
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    params = [i / 100 * math.tau for i in range(100)]
    points = circle.eval_points(params)
    assert len(points) == 100
    for u, p in zip(params, points):
        assert p.x == pytest.approx(2.0 * math.cos(u))
        assert p.y == pytest.approx(2.0 * math.sin(u))


def test_constructor_error_raises_value_error():
    with pytest.raises(ValueError) as excinfo:
        Circle3D(Point3D.origin(), Vector3D.z(), -1.0)
    assert str(excinfo.value)


def test_init_constructs():
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    assert circle.radius() == 2.0
    # `new` is no longer a constructor: only __init__ builds objects.
    assert not hasattr(Circle3D, "new")


def test_parametrize_on_cylinder():
    # Coaxial circle on a cylinder of the same radius: the pcurve is the
    # horizontal line v = 0 in (u, v) space.
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    pcurve = circle.parametrize_on(cylinder)
    assert isinstance(pcurve, Line2D)
    origin = pcurve.origin()
    direction = pcurve.direction()
    assert origin.x == pytest.approx(0.0)
    assert origin.y == pytest.approx(0.0)
    assert direction.x == pytest.approx(1.0)
    assert direction.y == pytest.approx(0.0)


def test_parametrize_on_not_analytic_raises():
    sphere = Sphere(Point3D.origin(), 2.0)
    line = geomcore.curves.Line3D(Point3D(2.0, 0.0, 0.0), Vector3D.z())
    with pytest.raises(ValueError):
        line.parametrize_on(sphere)


def test_bspline_curve_degree_one_line():
    curve = BSplineCurve3D(
        1,
        [Point3D(0.0, 0.0, 0.0), Point3D(2.0, 0.0, 0.0)],
        [0.0, 1.0],
        [2, 2],
        False,
    )
    mid = curve.eval_point(0.5)
    assert mid.x == pytest.approx(1.0)
    assert mid.y == pytest.approx(0.0)
    assert curve.degree() == 1
    assert not curve.is_periodic()


def test_transform_rotation():
    axis = Axis3D(Point3D.origin(), Vector3D.z())
    rot = Transform.rotation(axis, math.pi / 2)
    p = rot.apply_point(Point3D(1.0, 0.0, 0.0))
    assert p.x == pytest.approx(0.0)
    assert p.y == pytest.approx(1.0)


def test_transform_composition():
    t = Transform.translation(Vector3D(1.0, 0.0, 0.0))
    rot = Transform.rotation(Axis3D(Point3D.origin(), Vector3D.z()), math.pi / 2)
    composed = t.then(rot)
    p = composed.apply_point(Point3D.origin())
    assert p.x == pytest.approx(0.0)
    assert p.y == pytest.approx(1.0)


def test_sphere_parameters_round_trip():
    sphere = Sphere(Point3D.origin(), 3.0)
    p = sphere.eval_point(0.7, 0.4)
    u, v = sphere.parameters_of(p)
    assert u == pytest.approx(0.7)
    assert v == pytest.approx(0.4)


def test_frame3_accessors():
    frame = Frame3D(Point3D.origin(), Vector3D.z(), Vector3D.x())
    assert frame.z_direction().components() == pytest.approx((0.0, 0.0, 1.0))
    assert frame.x_direction().components() == pytest.approx((1.0, 0.0, 0.0))


def test_docstrings_present():
    assert Circle3D.__doc__
    assert Circle3D.eval_point.__doc__


def test_tolerance_defaults():
    tol = Tolerance.default()
    assert tol.confusion == 1e-7
    assert tol.angular == 1e-12
    assert tol.parametric == 1e-9
    assert Tolerance().confusion == tol.confusion
    custom = Tolerance(confusion=1e-3)
    assert custom.confusion == 1e-3
    assert custom.angular == tol.angular


def test_contains_curve():
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    assert circle.contains(Point3D(2.0, 0.0, 0.0))
    assert circle.contains(circle.eval_point(1.0))
    assert not circle.contains(Point3D(3.0, 0.0, 0.0))
    assert not circle.contains(Point3D.origin())
    # Explicit tolerance widens the acceptance band.
    near = Point3D(2.0 + 1e-6, 0.0, 0.0)
    assert not circle.contains(near)
    assert circle.contains(near, Tolerance(confusion=1e-5))


def test_contains_surface():
    from geomcore.surfaces import Plane, Torus

    plane = Plane(Point3D.origin(), Vector3D.z())
    assert plane.contains(Point3D.origin())
    assert plane.contains(Point3D(1.0, 2.0, 0.0))
    assert not plane.contains(Point3D(0.0, 0.0, 1.0))

    sphere = Sphere(Point3D.origin(), 3.0)
    assert sphere.contains(Point3D(3.0, 0.0, 0.0))
    assert not sphere.contains(Point3D.origin())

    torus = Torus(Point3D.origin(), Vector3D.z(), 4.0, 1.0)
    assert torus.contains(Point3D(5.0, 0.0, 0.0))
    assert not torus.contains(Point3D.origin())


def test_project_point_curve():
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    u, dist = circle.project_point(Point3D(3.0, 0.0, 0.0))
    assert u == pytest.approx(0.0)
    assert dist == pytest.approx(1.0)
    u, dist = circle.project_point(Point3D(2.0, 0.0, 0.0))
    assert dist == 0.0
    # Batch variant agrees with scalar calls.
    batch = circle.project_points([Point3D(3.0, 0.0, 0.0), Point3D(2.0, 0.0, 0.0)])
    assert batch[0] == pytest.approx((0.0, 1.0))
    assert batch[1][1] == 0.0


def test_project_point_surface():
    from geomcore.surfaces import Plane

    plane = Plane(Point3D.origin(), Vector3D.z())
    u, v, dist = plane.project_point(Point3D(1.0, 2.0, 3.0))
    assert (u, v) == pytest.approx((1.0, 2.0))
    assert dist == pytest.approx(3.0)
    batch = plane.project_points([Point3D.origin(), Point3D(0.0, 0.0, 2.0)])
    assert batch[0][2] == 0.0
    assert batch[1][2] == pytest.approx(2.0)


def test_intersect_plane_plane():
    from geomcore.surfaces import Plane

    xy = Plane(Point3D.origin(), Vector3D.z())
    zy = Plane(Point3D.origin(), Vector3D.x())
    kind, line = xy.intersect_plane(zy)
    assert kind == "line"
    assert line.origin().x == pytest.approx(0.0)
    # Direction follows the ordered pair (n1 x n2 = Z x X = Y).
    assert line.direction().components() == pytest.approx((0.0, 1.0, 0.0))

    kind, payload = xy.intersect_plane(Plane(Point3D(0.0, 0.0, 1.0), Vector3D.z()))
    assert (kind, payload) == ("parallel", None)

    kind, payload = xy.intersect_plane(xy)
    assert (kind, payload) == ("coincident", None)


def test_intersect_plane_sphere():
    from geomcore.surfaces import Plane

    plane = Plane(Point3D.origin(), Vector3D.z())
    kind, circle = plane.intersect_sphere(Sphere(Point3D.origin(), 2.0))
    assert kind == "circle"
    assert circle.radius() == pytest.approx(2.0)

    kind, point = plane.intersect_sphere(Sphere(Point3D(0.0, 0.0, 2.0), 2.0))
    assert kind == "tangent_point"
    assert (point.x, point.y, point.z) == pytest.approx((0.0, 0.0, 0.0))

    assert plane.intersect_sphere(Sphere(Point3D(0.0, 0.0, 5.0), 2.0)) == ("empty", None)


def test_intersect_sphere_sphere():
    s1 = Sphere(Point3D.origin(), 2.0)
    kind, circle = s1.intersect_sphere(Sphere(Point3D(3.0, 0.0, 0.0), 2.0))
    assert kind == "circle"
    assert circle.center().x == pytest.approx(1.5)

    kind, point = s1.intersect_sphere(Sphere(Point3D(4.0, 0.0, 0.0), 2.0))
    assert kind == "tangent_point"
    assert point.x == pytest.approx(2.0)

    assert s1.intersect_sphere(Sphere(Point3D(5.0, 0.0, 0.0), 2.0)) == ("empty", None)
    assert s1.intersect_sphere(s1)[0] == "coincident"


def test_intersect_plane_cylinder():
    from geomcore.surfaces import Cylinder, Plane

    plane = Plane(Point3D.origin(), Vector3D.z())
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    kind, circle = plane.intersect_cylinder(cylinder)
    assert kind == "circle"
    assert circle.radius() == pytest.approx(2.0)

    side = Plane(Point3D(0.0, 1.0, 0.0), Vector3D.y())
    kind, lines = side.intersect_cylinder(cylinder)
    assert kind == "two_lines"
    assert lines[0].direction().components() == pytest.approx((0.0, 0.0, 1.0))

    assert side.intersect_cylinder(Cylinder(Point3D.origin(), Vector3D.z(), 0.5)) == (
        "empty",
        None,
    )
    miss = Plane(Point3D(0.0, 3.0, 0.0), Vector3D.y())
    assert miss.intersect_cylinder(cylinder) == ("empty", None)


def test_intersect_plane_cone():
    from geomcore import Frame3D
    from geomcore.surfaces import Cone, Plane

    plane = Plane(Point3D.origin(), Vector3D.z())
    cone = Cone.from_frame(Frame3D.world(), 0.4, 2.0)
    kind, circle = plane.intersect_cone(cone)
    assert kind == "circle"
    assert circle.radius() == pytest.approx(2.0)

    below = Plane(Point3D(0.0, 0.0, -6.0), Vector3D.z())
    assert below.intersect_cone(cone) == ("empty", None)

    apex_plane = Plane(cone.apex(), Vector3D.z())
    kind, payload = apex_plane.intersect_cone(cone)
    assert kind == "apex_point"


def test_intersect_line_surface():
    from geomcore.curves import Line3D
    from geomcore.surfaces import Cylinder, Plane

    line = Line3D(Point3D.origin(), Vector3D.x())
    kind, ((t1, p1), (t2, p2)) = line.intersect_sphere(Sphere(Point3D.origin(), 2.0))
    assert kind == "two_points"
    assert (t1, t2) == pytest.approx((-2.0, 2.0))
    assert (p1.x, p2.x) == pytest.approx((-2.0, 2.0))

    kind, (t, p) = Line3D(Point3D.origin(), Vector3D.z()).intersect_plane(
        Plane(Point3D.origin(), Vector3D.z())
    )
    assert kind == "point"
    assert t == pytest.approx(0.0)
    assert line.intersect_plane(Plane(Point3D.origin(), Vector3D.z())) == (
        "coincident",
        None,
    )

    gen = Line3D(Point3D(2.0, 0.0, 0.0), Vector3D.z())
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    assert gen.intersect_cylinder(cylinder) == ("coincident", None)


def test_intersect_symmetric_quadrics():
    import math

    from geomcore import Frame3D
    from geomcore.surfaces import Cone, Cylinder

    sphere = Sphere(Point3D.origin(), 3.0)
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)

    kind, (c1, c2) = sphere.intersect_cylinder(cylinder)
    assert kind == "two_circles"
    assert c1.radius() == pytest.approx(2.0)
    assert c1.center().z == pytest.approx(math.sqrt(5.0))

    kind, _ = sphere.intersect_cylinder(Cylinder(Point3D.origin(), Vector3D.z(), 3.0))
    assert kind == "circle"
    off = Cylinder(Point3D(0.0, 1.0, 0.0), Vector3D.z(), 2.0)
    assert sphere.intersect_cylinder(off) == ("not_analytic", None)

    cone = Cone.from_frame(Frame3D.world(), 0.4, 2.0)
    kind, (r1, r2) = sphere.intersect_cone(cone)
    assert kind == "two_circles"
    assert r1.radius() == pytest.approx(2.6188, abs=1e-3)
    assert r2.radius() == pytest.approx(0.7746, abs=1e-3)
    # Small cone: one root falls behind the apex, leaving a single ring.
    small = Cone.from_frame(Frame3D.world(), 0.4, 0.5)
    kind, ring = sphere.intersect_cone(small)
    assert kind == "circle"
    assert ring.radius() == pytest.approx(1.5786, abs=1e-3)

    c1 = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    c2 = Cylinder(Point3D(3.0, 0.0, 0.0), Vector3D.z(), 2.0)
    kind, (l1, l2) = c1.intersect_cylinder(c2)
    assert kind == "two_lines"
    assert l1.origin().x == pytest.approx(1.5)
    assert c1.intersect_cylinder(c1)[0] == "coincident"
    crossed = Cylinder(Point3D.origin(), Vector3D.x(), 2.0)
    assert c1.intersect_cylinder(crossed) == ("not_analytic", None)


def test_intersect_cone_cylinder():
    from geomcore import Frame3D
    from geomcore.surfaces import Cone, Cylinder

    cone = Cone.from_frame(Frame3D.world(), 0.4, 2.0)
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    kind, circle = cone.intersect_cylinder(cylinder)
    assert kind == "circle"
    assert circle.radius() == pytest.approx(2.0)
    assert circle.center().distance(Point3D.origin()) < 1e-9

    off = Cylinder(Point3D(1.0, 0.0, 0.0), Vector3D.z(), 2.0)
    assert cone.intersect_cylinder(off) == ("not_analytic", None)


def test_intersect_cone_cone():
    from geomcore import Frame3D
    from geomcore.surfaces import Cone

    c1 = Cone.from_frame(Frame3D.world(), 0.4, 2.0)
    frame2 = Frame3D(Point3D(0.0, 0.0, 1.0), Vector3D.z(), Vector3D.x())
    c2 = Cone.from_frame(frame2, 0.6, 1.0)
    kind, circle = c1.intersect_cone(c2)
    assert kind == "circle"
    assert circle.radius() == pytest.approx(4.7245, abs=1e-3)

    assert c1.intersect_cone(c1)[0] == "coincident"
    tilted = Cone(Point3D.origin(), Vector3D.x(), 0.4, 2.0)
    assert c1.intersect_cone(tilted) == ("not_analytic", None)


def test_intersect_torus_symmetric():
    from geomcore.surfaces import Cone, Cylinder, Plane, Torus

    torus = Torus(Point3D.origin(), Vector3D.z(), 4.0, 1.0)

    kind, (c1, c2) = torus.intersect_plane(Plane(Point3D.origin(), Vector3D.z()))
    assert kind == "two_circles"
    assert sorted([c1.radius(), c2.radius()]) == pytest.approx([3.0, 5.0])

    kind, _ = torus.intersect_plane(Plane(Point3D.origin(), Vector3D.x()))
    assert kind == "two_circles"

    kind, c = torus.intersect_sphere(Sphere(Point3D.origin(), 5.0))
    assert kind == "tangent_circle"
    assert c.radius() == pytest.approx(5.0)

    kind, _ = torus.intersect_cylinder(Cylinder(Point3D.origin(), Vector3D.z(), 4.5))
    assert kind == "two_circles"

    kind, _ = torus.intersect_cone(Cone.from_frame(Frame3D.world(), 1.1, 2.0))
    assert kind == "two_circles"

    other = Torus(Point3D(0.0, 0.0, 1.5), Vector3D.z(), 4.0, 1.0)
    kind, _ = torus.intersect_torus(other)
    assert kind == "two_circles"
    assert torus.intersect_torus(torus)[0] == "coincident"
    tilted = Torus(Point3D.origin(), Vector3D.x(), 4.0, 1.0)
    assert torus.intersect_torus(tilted) == ("not_analytic", None)


def test_interpolate_curve():
    curve = BSplineCurve3D.interpolate(
        [Point3D(0.0, 0.0, 0.0), Point3D(1.0, 1.0, 0.0), Point3D(2.0, 0.0, 0.0)], 2
    )
    assert curve.contains(Point3D(0.0, 0.0, 0.0))
    assert curve.contains(Point3D(2.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        BSplineCurve3D.interpolate([Point3D.origin()], 1)


def test_intersect_curve_surface():
    from geomcore import intersect_curve_surface
    from geomcore.curves import Line3D

    line = Line3D(Point3D.origin(), Vector3D.x())
    hits = intersect_curve_surface(line, Sphere(Point3D.origin(), 2.0))
    assert len(hits) == 2
    assert hits[0][0] == pytest.approx(-2.0)
    assert hits[1][0] == pytest.approx(2.0)

    assert intersect_curve_surface(line, Sphere(Point3D(0.0, 0.0, 5.0), 2.0)) == []


def test_intersect_marching():
    from geomcore import intersect_marching
    from geomcore.surfaces import Plane

    xy = Plane(Point3D.origin(), Vector3D.z())
    zy = Plane(Point3D.origin(), Vector3D.x())
    curves = intersect_marching(xy, zy)
    assert len(curves) == 1

    sphere = Sphere(Point3D.origin(), 2.0)
    loops = intersect_marching(xy, sphere)
    assert len(loops) == 1

    miss = Plane(Point3D(0.0, 0.0, 5.0), Vector3D.z())
    assert intersect_marching(miss, sphere) == []


def test_curve_curve_generic():
    from geomcore import curve_curve_extrema, intersect_curve_curve
    from geomcore.curves import Circle3D, Line3D

    l1 = Line3D(Point3D.origin(), Vector3D.x())
    l2 = Line3D(Point3D(0.0, 0.0, 1.0), Vector3D.y())
    ext = curve_curve_extrema(l1, l2)
    assert len(ext) == 1
    assert ext[0][2] == pytest.approx(1.0)

    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    hits = intersect_curve_curve(l1, circle)
    assert len(hits) == 2
    assert hits[0][0] == pytest.approx(-2.0)
    assert hits[1][0] == pytest.approx(2.0)


def test_parametrize_numeric():
    import math

    from geomcore import parametrize_numeric

    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    pcurve = parametrize_numeric(circle, cylinder, 9)
    assert len(pcurve) == 9
    for u, v in pcurve:
        assert v == pytest.approx(0.0)
    # Round-trip through the surface recovers the curve.
    for i, (u, v) in enumerate(pcurve):
        p = cylinder.eval_point(u, v)
        q = circle.eval_point(i / 8 * math.tau)
        assert p.x == pytest.approx(q.x)
        assert p.y == pytest.approx(q.y)


def test_intersect_curves_2d_and_lines():
    from geomcore.curves import Circle2D, Line2D, Line3D

    l1 = Line2D(Point2D(0.0, 0.0), Vector2D(1.0, 0.0))
    l2 = Line2D(Point2D(0.0, 0.0), Vector2D(0.0, 1.0))
    kind, (s, p, t) = l1.intersect_line(l2)
    assert kind == "point"
    assert (s, t) == pytest.approx((0.0, 0.0))

    circle = Circle2D(Point2D(0.0, 0.0), 2.0)
    kind, ((t1, _), (t2, _)) = l1.intersect_circle(circle)
    assert kind == "points"
    assert (t1, t2) == pytest.approx((-2.0, 2.0))

    kind, (p1, p2) = circle.intersect_circle(Circle2D(Point2D(3.0, 0.0), 2.0))
    assert kind == "points"
    assert p1.x == pytest.approx(1.5)

    a = Line3D(Point3D.origin(), Vector3D.x())
    b = Line3D(Point3D.origin(), Vector3D.y())
    kind, (_, p, _) = a.intersect_line(b)
    assert kind == "point"
    assert (p.x, p.y, p.z) == pytest.approx((0.0, 0.0, 0.0))
    kind, payload = a.intersect_line(Line3D(Point3D(0.0, 0.0, 1.0), Vector3D.y()))
    assert kind == "skew"
    assert payload[2] == pytest.approx(1.0)


def test_intersect_curves_3d():
    from geomcore.curves import Circle3D, Line3D

    line = Line3D(Point3D.origin(), Vector3D.x())
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    kind, ((t1, _), (t2, _)) = line.intersect_circle(circle)
    assert kind == "points"
    assert (t1, t2) == pytest.approx((-2.0, 2.0))

    c1 = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    c2 = Circle3D(Point3D(3.0, 0.0, 0.0), Vector3D.z(), 2.0)
    kind, (p1, p2) = c1.intersect_circle(c2)
    assert kind == "points"
    assert p1.x == pytest.approx(1.5)
    tilted = Circle3D(Point3D.origin(), Vector3D.x(), 2.0)
    assert c1.intersect_circle(tilted) == ("not_analytic", None)
    assert c1.intersect_circle(c1)[0] == "coincident"


def _bilinear_patch():
    from geomcore.surfaces import BSplineSurface

    return BSplineSurface(
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


def test_bspline_surface_projection():
    surface = _bilinear_patch()
    u, v, dist = surface.project_point(Point3D(1.0, 1.0, 1.0))
    assert (u, v) == pytest.approx((0.5, 0.5))
    assert dist == pytest.approx(1.0)
    assert surface.contains(Point3D(1.0, 1.0, 0.0))
    assert not surface.contains(Point3D(1.0, 1.0, 1.0))
    ext = surface.extrema(Point3D(1.0, 1.0, 1.0))
    assert len(ext) >= 1
    assert ext[0][2] == pytest.approx(1.0)


def test_bspline_curve_projection():
    curve = BSplineCurve3D(
        1,
        [Point3D(0.0, 0.0, 0.0), Point3D(2.0, 0.0, 0.0)],
        [0.0, 1.0],
        [2, 2],
        False,
    )
    u, dist = curve.project_point(Point3D(1.0, 1.0, 0.0))
    assert u == pytest.approx(0.5)
    assert dist == pytest.approx(1.0)
    assert curve.contains(Point3D(1.0, 0.0, 0.0))
    assert not curve.contains(Point3D(1.0, 1.0, 0.0))
    # Beyond the end: the endpoint wins.
    u, dist = curve.project_point(Point3D(5.0, 0.0, 0.0))
    assert u == pytest.approx(1.0)
    assert dist == pytest.approx(3.0)
    ext = curve.extrema(Point3D(1.0, 1.0, 0.0))
    assert len(ext) >= 1
    assert ext[0][1] == pytest.approx(1.0)


def test_conic_extrema_and_projection():
    from geomcore.curves import Ellipse3D, Hyperbola3D, Parabola3D

    ellipse = Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 3.0, 1.5)
    ext = ellipse.extrema(Point3D(4.0, 0.0, 0.0))
    assert len(ext) >= 2
    assert ext[0][1] == pytest.approx(1.0)
    u, dist = ellipse.project_point(Point3D(0.0, 3.0, 0.0))
    assert dist == pytest.approx(1.5)

    parabola = Parabola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 1.0)
    u, dist = parabola.project_point(Point3D(1.0, 2.0, 0.0))
    assert u == pytest.approx(2.0)
    assert dist == 0.0

    hyperbola = Hyperbola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 2.0, 1.0)
    u, dist = hyperbola.project_point(Point3D(3.0, 0.0, 0.0))
    assert dist == pytest.approx(math.sqrt(0.8))
