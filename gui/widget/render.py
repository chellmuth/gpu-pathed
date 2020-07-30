import math

import OpenGL.GL as gl
from PyQt5.QtWidgets import QOpenGLWidget

from gui.core.point import Point2, Point3
from gui.core.trackball import Trackball

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
