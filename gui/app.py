import sys

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QWidget
)

import path_tracer
from gui.widget.render import RenderWidget
from gui.widget.sidebar import SidebarWidget

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

        self.sidebar = SidebarWidget(self.model, self)
        layout.addWidget(self.sidebar)

        self.setLayout(layout)

    def update(self):
        self.gl.update()
        self.sidebar.update()


def run():
    app = QApplication(sys.argv)

    width, height = 400, 400
    pt = path_tracer.RenderSession(width, height)

    widget = App(pt)
    widget.show()

    sys.exit(app.exec_())

