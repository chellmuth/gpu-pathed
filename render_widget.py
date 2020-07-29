import math

import OpenGL.GL as gl
from PyQt5.QtWidgets import QOpenGLWidget


class RenderWidget(QOpenGLWidget):
    def __init__(self, pt, parent=None):
        super().__init__(parent)

        self.pt = pt
        self.model = self.pt.getSceneModel()

        self.setFixedSize(pt.getWidth(), pt.getHeight())

        origin = Point3.from_vec3(self.model.getCameraOrigin())
        target = Point3.from_vec3(self.model.getCameraTarget())
        self.trackball = Trackball(origin, target)
        self.tracking = False
        self.setMouseTracking(False)

    def update(self):
        super().update()

        if self.tracking: return

        origin = Point3.from_vec3(self.model.getCameraOrigin())
        target = Point3.from_vec3(self.model.getCameraTarget())
        self.trackball.reset(origin, target)

    def mouseReleaseEvent(self, event):
        self.tracking = False

    def mousePressEvent(self, event):
        self.pt.hitTest(event.x(), self.height() - event.y() - 1)

        self.tracking = True
        q_position = event.localPos()
        self.trackball.set_anchor(Point2(q_position.x(), q_position.y()))

    def mouseMoveEvent(self, event):
        q_position = event.localPos()
        origin = self.trackball.drag(Point2(q_position.x(), q_position.y()))
        self.model.setCameraOrigin(origin.x, origin.y, origin.z)
        self.update()

    def wheelEvent(self, event):
        ticks = event.angleDelta().y() / 120
        self.pt.getSceneModel().zoomCamera(ticks)

    def initializeGL(self):
        super().initializeGL()

        width = self.width()
        height = self.height()

        gl.glClearColor(1., 0.5, 1., 1.)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        self.pbo1, self.pbo2 = gl.glGenBuffers(2)

        for pbo in [self.pbo1, self.pbo2]:
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo)
            gl.glBufferData(
                gl.GL_PIXEL_UNPACK_BUFFER,
                4 * width * height,
                None,
                gl.GL_DYNAMIC_DRAW
            )
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

        self.state = self.pt.init(self.pbo1, self.pbo2)

    def paintGL(self):
        if self.state.isRendering:
            self.state = self.pt.pollRender()
        else:
            self.state = self.pt.renderAsync()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.state.pbo)

        gl.glDrawPixels(self.width(), self.height(), gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)

        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)


class Point3:
    @classmethod
    def from_vec3(cls, vec3):
        return cls(
            vec3.x(),
            vec3.y(),
            vec3.z()
        )

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def normalized(self):
        length = self.length()

        return Point3(
            self.x / length,
            self.y / length,
            self.z / length
        )

    def __add__(self, other):
        return Point3(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )

    def __sub__(self, other):
        return Point3(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )

    def __mul__(self, scalar):
        return Point3(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar
        )

    def length(self):
        return math.sqrt(
            self.x ** 2 + self.y ** 2 + self.z ** 2
        )

    def __repr__(self):
        return f"Point3: ({self.x}, {self.y} {self.z})"

class Point2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Point2(
            self.x - other.x,
            self.y - other.y
        )

    def __truediv__(self, scalar):
        return Point2(
            self.x / scalar,
            self.y / scalar
        )

    def __mul__(self, scalar):
        return Point2(
            self.x * scalar,
            self.y * scalar
        )

    def __repr__(self):
        return f"Point2: ({self.x}, {self.y})"

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
        delta = point - self.anchor

        radians = delta / 180. * math.pi

        cartesian = spherical_to_cartesian(math.pi / 2 - radians.x, math.pi / 2 + radians.y)
        transform = Transform.to_world((self.origin - self.target), Point3(0, 1, 0))

        new_origin_direction = transform.transform_direction(cartesian)

        return self.target + new_origin_direction * self.radius

def cross(v1, v2):
    return Point3(
        v1.y * v2.z - v1.z * v2.y,
        -(v1.x * v2.z - v1.z * v2.x),
        v1.x * v2.y - v1.y * v2.x
    )

import numpy as np

class Transform:
    def __init__(self, data):
        self.data = np.array(data)

    @classmethod
    def to_tangent(cls, z_axis, up):
        return cls.to_world(z_axis, up).transposed()

    @classmethod
    def to_world(cls, z_axis, up):
        x_axis = cross(z_axis, up).normalized()
        y_axis = cross(z_axis, x_axis).normalized()
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


# y is up!
def cartesian_to_spherical(cartesian):
    phi = math.atan2(cartesian.z, cartesian.x)

    if phi < 0.:
        phi += 2. * math.pi
    if phi == 2. * math.pi:
        phi = 0.

    theta = math.acos(cartesian.y)

    return phi, theta

def spherical_to_cartesian(phi, theta):
    y = math.cos(theta)

    x = math.sin(theta) * math.cos(phi)
    z = math.sin(theta) * math.sin(phi)

    return Point3(x, y, z)


from pytest import approx

def test_cartesian_zero():
    cartesian = spherical_to_cartesian(math.pi / 2, math.pi / 2)
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
