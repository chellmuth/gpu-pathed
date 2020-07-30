import numpy as np

from gui.core.point import Point3

class Transform:
    def __init__(self, data):
        self.data = np.array(data)

    @classmethod
    def to_tangent(cls, z_axis, up):
        return cls.to_world(z_axis, up).transposed()

    @classmethod
    def to_world(cls, z_axis, up):
        x_axis = Point3.cross(z_axis, up).normalized()
        y_axis = Point3.cross(z_axis, x_axis).normalized()
        z_axis = z_axis.normalized()

        return cls([
            [ x_axis.x, y_axis.x, z_axis.x, 0. ],
            [ x_axis.y, y_axis.y, z_axis.y, 0. ],
            [ x_axis.z, y_axis.z, z_axis.z, 0. ],
            [ 0., 0., 0., 1. ],
        ])

    def transposed(self):
        return Transform(self.data.T)

    def transform_direction(self, direction):
        x = np.array([direction.x, direction.y, direction.z, 0.])

        result = self.data.dot(x)
        return Point3(*self.data.dot(x)[:3])
