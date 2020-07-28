from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
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

        self.maxDepth = MaxDepthWidget(
            self.model.getMaxDepth,
            self.model.setMaxDepth,
            self
        )
        layout.addWidget(self.maxDepth)

        self.lightSlider = LightSlider(self.model, self)
        layout.addWidget(self.lightSlider)

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
        menu.addAction("Optix", lambda: self.handleStateChanged(RendererType.Optix))
        self.rendererButton.setMenu(menu)

        layout.addWidget(self.rendererButton)

        self.setLayout(layout)

    def update(self):
        renderer_type = self.getter()
        if renderer_type == RendererType.CUDA:
            self.rendererButton.setText("CUDA")
        elif renderer_type == RendererType.Optix:
            self.rendererButton.setText("Optix")
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
