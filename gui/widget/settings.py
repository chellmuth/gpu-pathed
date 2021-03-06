from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget
)

from path_tracer import RendererType

class SettingsWidget(QGroupBox):
    def __init__(self, model, parent=None):
        super().__init__("Settings", parent)

        self.model = model

        layout = QVBoxLayout()

        self.renderer = RendererWidget(
            self.model.getRendererType,
            self.model.setRendererType,
            self
        )
        layout.addWidget(self.renderer)

        self.nee = NextEventEstimationWidget(
            self.model.getNextEventEstimation,
            self.model.setNextEventEstimation,
            self
        )
        layout.addWidget(self.nee)

        self.maxDepth = MaxDepthWidget(
            self.model.getMaxDepth,
            self.model.setMaxDepth,
            self
        )
        layout.addWidget(self.maxDepth)

        self.setLayout(layout)
        self.update()

    def update(self):
        self.renderer.update()


class RendererWidget(QWidget):
    def __init__(self, getter, setter, parent=None):
        super().__init__(parent)

        self.getter = getter
        self.setter = setter

        layout = QHBoxLayout()

        self.rendererLabel = QLabel("Renderer:")
        layout.addWidget(self.rendererLabel)

        self.rendererButton = QPushButton("Renderer")
        menu = QMenu(self)
        menu.addAction("CUDA", lambda: self.handleStateChanged(RendererType.CUDA))
        menu.addAction("OptiX", lambda: self.handleStateChanged(RendererType.Optix))
        menu.addAction("Normals", lambda: self.handleStateChanged(RendererType.Normals))
        self.rendererButton.setMenu(menu)

        layout.addWidget(self.rendererButton)

        self.setLayout(layout)

    def update(self):
        renderer_type = self.getter()
        if renderer_type == RendererType.CUDA:
            self.rendererButton.setText("CUDA")
        elif renderer_type == RendererType.Optix:
            self.rendererButton.setText("OptiX")
        elif renderer_type == RendererType.Normals:
            self.rendererButton.setText("Normals")
        else:
            print("Error, unsupported renderer type")
            exit(1)

    def handleStateChanged(self, renderer_type):
        self.setter(renderer_type)
        self.update()


class MaxDepthWidget(QWidget):
    def __init__(self, getter, setter, parent=None):
        super().__init__(parent)

        self.getter = getter
        self.setter = setter

        layout = QHBoxLayout()

        self.text = QLabel("Max Depth: ", self)
        layout.addWidget(self.text)

        self.maxDepth = QSpinBox(self)
        self.maxDepth.setMinimum(0)
        self.maxDepth.setValue(self.getter())
        self.maxDepth.valueChanged.connect(self.handleValueChanged)
        layout.addWidget(self.maxDepth)

        self.setLayout(layout)

        self.update()

    def handleValueChanged(self):
        self.setter(self.maxDepth.value())
        self.update()

    def update(self):
        maxDepth = self.getter()
        self.maxDepth.setValue(maxDepth)


class NextEventEstimationWidget(QWidget):
    def __init__(self, getter, setter, parent=None):
        super().__init__(parent)

        self.getter = getter
        self.setter = setter

        layout = QHBoxLayout()

        self.text = QLabel("Next Event Estimation: ", self)
        layout.addWidget(self.text)

        self.nee = QCheckBox(self)
        self.nee.setChecked(self.getter())
        self.nee.stateChanged.connect(self.handleStateChanged)
        layout.addWidget(self.nee)

        self.setLayout(layout)

    def handleStateChanged(self, state):
        # 0 == Qt.Unchecked
        # 2 == Qt.Checked

        self.setter(bool(state))
        self.update()

    def update(self):
        state = self.getter()
        if state == self.nee.isChecked():
            return

        self.nee.setChecked(state)


