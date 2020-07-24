from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QMenu, QPushButton, QSlider, QVBoxLayout, QWidget

from path_tracer import RendererType

class SettingsWidget(QGroupBox):
    def __init__(self, model, parent=None):
        super().__init__("Settings", parent)

        self.model = model

        layout = QVBoxLayout()

        self.lightSlider = LightSlider(self.model)
        layout.addWidget(self.lightSlider)

        self.rendererButton = QPushButton("Renderer")
        menu = QMenu(self)
        menu.addAction("CUDA", lambda: self.handleStateChanged(RendererType.CUDA))
        menu.addAction("Optix", lambda: self.handleStateChanged(RendererType.Optix))
        self.rendererButton.setMenu(menu)

        layout.addWidget(self.rendererButton)

        self.setLayout(layout)
        self.update()

    def update(self):
        renderer_type = self.model.getRendererType()
        if renderer_type == RendererType.CUDA:
            self.rendererButton.setText("CUDA")
        elif renderer_type == RendererType.Optix:
            self.rendererButton.setText("Optix")
        else:
            print("Error, unsupported renderer type")
            exit(1)

    def handleStateChanged(self, renderer_type):
        self.model.setRendererType(renderer_type)
        self.update()

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
