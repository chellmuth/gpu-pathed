from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget
)

from gui.widget.color import ColorButton

class LambertianWidget(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)

        self.model = model

        layout = QVBoxLayout()

        self.albedoButton = ColorButton(
            "Albedo",
            self.model.getColor,
            self.model.setColor,
            self
        )
        layout.addWidget(self.albedoButton)

        self.emitWidget = EmitWidget(self.model, self)
        layout.addWidget(self.emitWidget)

        layout.addStretch()
        self.setLayout(layout)

    def update(self):
        self.albedoButton.update()
        self.emitWidget.update()


class EmitWidget(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()

        self.text = QLabel("Emit: ", self)
        layout.addWidget(self.text)

        self.sliders = EmitSliders(model.getEmit, model.setEmit, self)
        layout.addWidget(self.sliders)

        self.setLayout(layout)

    def update(self):
        self.sliders.update()

class EmitSliders(QWidget):
    def __init__(self, getter, setter, parent=None):
        super().__init__(parent)

        self.getter = getter
        self.setter = setter

        layout = QVBoxLayout()

        self.sliderR = self.buildEmitSlider(self.handleRMoved)
        self.sliderG = self.buildEmitSlider(self.handleGMoved)
        self.sliderB = self.buildEmitSlider(self.handleBMoved)

        layout.addWidget(self.sliderR)
        layout.addWidget(self.sliderG)
        layout.addWidget(self.sliderB)
        layout.addStretch()

        self.setLayout(layout)

    def buildEmitSlider(self, handleMoved):
        slider = QSlider(Qt.Horizontal, self)
        slider.setMinimum(0)
        slider.setMaximum(25)
        slider.sliderMoved.connect(handleMoved)
        return slider

    def handleRMoved(self, value):
        self.setter(
            value,
            self.sliderG.value(),
            self.sliderB.value()
        )

    def handleGMoved(self, value):
        self.setter(
            self.sliderR.value(),
            value,
            self.sliderB.value()
        )

    def handleBMoved(self, value):
        self.setter(
            self.sliderR.value(),
            self.sliderG.value(),
            value,
        )

    def update(self):
        emit = self.getter()

        self.sliderR.setValue(emit.r())
        self.sliderG.setValue(emit.g())
        self.sliderB.setValue(emit.b())
