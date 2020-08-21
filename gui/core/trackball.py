import math

import gui.core.coordinates as coordinates
from gui.core.point import Point2, Point3
from gui.core.transform import Transform

class Trackball:
    def __init__(self, origin, target):
        self.anchor = Point2(0, 0)

        self.reset(origin, target)

    def reset(self, origin, target):
        self.origin = origin
        self.target = target
        self.radius = (self.target - self.origin).length()

    def set_anchor(self, point):
        self.anchor = point

    def drag(self, point):
        velocity = 0.1
        delta = (point - self.anchor) * velocity

        radians = delta / 180. * math.pi

        cartesian = coordinates.spherical_to_cartesian(
            math.pi / 2 - radians.x,
            math.pi / 2 + radians.y
        )
        transform = Transform.to_world((self.origin - self.target), Point3(0, 1, 0))

        new_origin_direction = transform.transform_direction(cartesian)

        return self.target + new_origin_direction * self.radius
