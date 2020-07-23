import sys

import OpenGL.GL as gl
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QGroupBox, QHBoxLayout, QLabel, QOpenGLWidget, QSlider, QVBoxLayout, QWidget

import path_tracer
from camera_widget import CameraWidget
from material_widget import MaterialWidget

class App(QWidget):
    def __init__(self, pt, parent=None):
        super().__init__(parent)

        self.setWindowTitle("CUDA / PyQT Demo")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(100)

        self.pt = pt
        self.model = self.pt.getSceneModel()

        layout = QHBoxLayout()

        self.gl = RenderWidget(self.pt, self.handleColorChange, self)
        layout.addWidget(self.gl)
        layout.setAlignment(self.gl, Qt.AlignTop)

        self.sidebar = Sidebar(self.model, self)
        layout.addWidget(self.sidebar)

        self.setLayout(layout)

    def handleColorChange(self):
        self.sidebar.update()

    def update(self):
        self.gl.update()
        self.sidebar.update()


class RenderWidget(QOpenGLWidget):
    def __init__(self, pt, handleColorChange, parent=None):
        super().__init__(parent)

        self.pt = pt
        self.handleColorChange = handleColorChange
        self.setFixedSize(pt.getWidth(), pt.getHeight())

    def mousePressEvent(self, event):
        self.pt.hitTest(event.x(), self.height() - event.y() - 1)
        self.handleColorChange

    def wheelEvent(self, event):
        print("Wheel:", event.angleDelta())

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


class Sidebar(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)

        self.model = model

        # Materials group
        self.materialGroup = MaterialWidget(self.model, self)

        # Camera group
        self.cameraGroup = CameraWidget(self.model, self)

        # Settings group
        self.settingsGroup = QGroupBox("Settings", self)
        settingsLayout = QVBoxLayout()

        self.lightSlider = LightSlider(self.model)
        settingsLayout.addWidget(self.lightSlider)

        self.settingsGroup.setLayout(settingsLayout)

        # Info group
        self.infoGroup = QGroupBox("Info", self)
        infoLayout = QVBoxLayout()

        self.spp = SppLabel(model, self)
        infoLayout.addWidget(self.spp)

        self.infoGroup.setLayout(infoLayout)

        # Sidebar layout
        layout = QVBoxLayout()
        layout.addWidget(self.materialGroup)
        layout.addWidget(self.cameraGroup)
        layout.addWidget(self.settingsGroup)
        layout.addWidget(self.infoGroup)
        layout.addStretch()
        self.setLayout(layout)

    def update(self):
        self.materialGroup.update()
        self.spp.update()

class LightSlider(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)

        self.model = model

        layout = QHBoxLayout()

        self.text = QLabel("Light Position: ", self)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(-100)
        self.slider.setMaximum(100)
        self.slider.setValue(int(self.model.getLightPosition() * 100))

        self.slider.valueChanged.connect(self.handleChanged)

        layout.addWidget(self.text)
        layout.addWidget(self.slider)
        layout.addStretch()

        self.setLayout(layout)

    def handleChanged(self, value):
        self.model.setLightPosition(value / 100.)

class SppLabel(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)

        self.model = model

        layout = QHBoxLayout()
        self.sppLabel = QLabel(self._sppLabelText())
        layout.addWidget(self.sppLabel)
        layout.addStretch()

        self.setLayout(layout)

    def update(self):
        self.sppLabel.setText(self._sppLabelText())

    def _sppLabelText(self):
        return f"spp: {self.model.getSpp()}"


def run():
    app = QApplication(sys.argv)

    width, height = 640, 360
    pt = path_tracer.RenderSession(width, height)

    widget = App(pt)
    widget.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
