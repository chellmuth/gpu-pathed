import sys

import OpenGL.GL as gl
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication, QColorDialog, QGroupBox, QHBoxLayout, QLabel, QOpenGLWidget, QPushButton, QSlider, QVBoxLayout, QWidget

import path_tracer

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

        self.gl = RenderWidget(self.pt, self)
        layout.addWidget(self.gl)

        self.sidebar = Sidebar(self.model, self)
        layout.addWidget(self.sidebar)

        self.setLayout(layout)

    def update(self):
        self.gl.update()
        self.sidebar.update()


class RenderWidget(QOpenGLWidget):
    def __init__(self, pt, parent=None):
        super().__init__(parent)

        self.pt = pt
        self.setFixedSize(640, 360)

    def initializeGL(self):
        super().initializeGL()

        width = self.width()
        height = self.height()

        gl.glClearColor(1., 0.5, 1., 1.)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        self.pbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        gl.glBufferData(
            gl.GL_PIXEL_UNPACK_BUFFER,
            4 * width * height,
            None,
            gl.GL_DYNAMIC_DRAW
        )
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

        self.pt.init(self.pbo, width, height)

    def paintGL(self):
        super().paintGL()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        self.pt.render()

        gl.glDrawPixels(self.width(), self.height(), gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)


class Sidebar(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)

        self.model = model

        # Settings group
        self.settingsGroup = QGroupBox("Settings", self)
        settingsLayout = QVBoxLayout()

        self.albedoButton = AlbedoButton(self.model)
        settingsLayout.addWidget(self.albedoButton)

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
        layout.addWidget(self.settingsGroup)
        layout.addWidget(self.infoGroup)
        layout.addStretch()
        self.setLayout(layout)

    def update(self):
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

class AlbedoButton(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)

        self.model = model

        layout = QHBoxLayout()
        self.text = QLabel("Albedo: ", self)
        self.button = QPushButton("", self)
        self.button.setFixedSize(20, 20)

        layout.addWidget(self.text)
        layout.addWidget(self.button)
        layout.addStretch()

        self.setLayout(layout)

        color = unwrapQcolor(self.model.getColor())
        self.setColor(color)

        self.button.clicked.connect(self.handlePush)

    def setColor(self, color):
        palette = self.button.palette()
        palette.setColor(QPalette.Button, color)
        self.button.setAutoFillBackground(True)
        self.button.setPalette(palette)

    def handlePush(self):
        color = QColorDialog.getColor(parent=self)

        self.model.setColor(color.red() / 255., color.green() / 255., color.blue() / 255.)
        self.setColor(unwrapQcolor(self.model.getColor()))

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


def unwrapQcolor(vec3):
    return QColor(
        255 * vec3.r(),
        255 * vec3.g(),
        255 * vec3.b(),
    )


def run():
    app = QApplication(sys.argv)

    pt = path_tracer.PathTracer()

    widget = App(pt)
    widget.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
