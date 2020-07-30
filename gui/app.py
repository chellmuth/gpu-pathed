import sys

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget
)

import path_tracer
from gui.camera_widget import CameraWidget
from gui.material_widget import MaterialWidget
from gui.render_widget import RenderWidget
from gui.settings_widget import SettingsWidget

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
        layout.setAlignment(self.gl, Qt.AlignTop)

        self.sidebar = Sidebar(self.model, self)
        layout.addWidget(self.sidebar)

        self.setLayout(layout)

    def update(self):
        self.gl.update()
        self.sidebar.update()


class Sidebar(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)

        self.model = model

        # Materials group
        self.materialGroup = MaterialWidget(self.model, self)

        # Camera group
        self.cameraGroup = CameraWidget(self.model, self)

        # Settings group
        self.settingsGroup = SettingsWidget(self.model, self)

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
        self.cameraGroup.update()
        self.spp.update()

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

