import math

from pytest import approx

import gui.core.coordinates as coordinates
from gui.core.point import Point2, Point3
from gui.core.trackball import Trackball
from gui.core.transform import Transform

def test_cartesian_zero():
    cartesian = coordinates.spherical_to_cartesian(math.pi / 2, math.pi / 2)
    _assert_vectors_eq(cartesian, Point3(0, 0, 1))

def _assert_vectors_eq(v1, v2):
    assert v1.x == approx(v2.x)
    assert v1.y == approx(v2.y)
    assert v1.z == approx(v2.z)

def test_transform__identity():
    z_axis = Point3(0., 0., 6.8)
    up = Point3(0, 1, 0)

    transform = Transform.to_world(z_axis, up)

    z_axis_new = transform.transform_direction(Point3(0, 0, 1))

    _assert_vectors_eq(z_axis.normalized(), z_axis_new)

def test_trackball__identity():
    origin_original = Point3(0., 1., 6.8)

    trackball = Trackball(
        origin_original,
        Point3(0., 1., 0.)
    )

    origin_new = trackball.drag(Point2(0, 0))
    _assert_vectors_eq(origin_original, origin_new)
