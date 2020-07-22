from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from color_button import ColorButton

class MaterialWidget(QGroupBox):
    def __init__(self, model, parent=None):
        super().__init__("Material", parent)

        self.model = model

        layout = QVBoxLayout()

        self.materialIDLabel = QLabel(self._materialIDText(), self)
        layout.addWidget(self.materialIDLabel)

        self.albedoButton = ColorButton(
            "Albedo",
            self.model.getColor,
            self.model.setColor,
            self
        )
        layout.addWidget(self.albedoButton)

        self.emitButton = ColorButton(
            "Emit",
            self.model.getEmit,
            self.model.setEmit,
            self
        )
        layout.addWidget(self.emitButton)

        self.emitWidget = EmitWidget(self.model, self)
        layout.addWidget(self.emitWidget)

        layout.addStretch()
        self.setLayout(layout)

        self.update()

    def _materialIDText(self):
        return f"Material ID: {self.model.getMaterialIndex()}"

    def update(self):
        self.materialIDLabel.setText(self._materialIDText())
        self.albedoButton.update()
        self.emitButton.update()
        self.emitWidget.update()

        if self.model.getMaterialIndex() == -1:
            self.materialIDLabel.hide()
            self.albedoButton.hide()
            self.emitButton.hide()
            self.emitWidget.hide()
        else:
            self.materialIDLabel.show()
            self.albedoButton.show()
            self.emitButton.show()
            self.emitWidget.show()

class EmitWidget(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()

        self.text = QLabel("Emit: ", self)
        layout.addWidget(self.text)

        self.sliders = EmitSliders(model.getEmit, model.setEmit, self)
        layout.addWidget(self.sliders)

        layout.addStretch()
        self.setLayout(layout)

    def update(self):
        self.sliders.update()

class EmitSliders(QWidget):
    def __init__(self, getter, setter, parent=None):
        super().__init__(parent)

        self.getter = getter
        self.setter = setter

        layout = QVBoxLayout()

        self.sliderR = self.buildEmitSlider()
        self.sliderG = self.buildEmitSlider()
        self.sliderB = self.buildEmitSlider()

        layout.addWidget(self.sliderR)
        layout.addWidget(self.sliderG)
        layout.addWidget(self.sliderB)
        layout.addStretch()

        self.setLayout(layout)

    def buildEmitSlider(self):
        slider = QSlider(Qt.Horizontal, self)
        slider.setMinimum(0)
        slider.setMaximum(50)
        slider.valueChanged.connect(self.handleChanged)
        return slider

    def handleChanged(self, value):
        self.setter(
            self.sliderR.value() / 2.,
            self.sliderG.value() / 2.,
            self.sliderB.value() / 2.
        )

    def update(self):
        emit = self.getter()

        self.sliderR.setValue(emit.r() * 2.)
        self.sliderG.setValue(emit.g() * 2.)
        self.sliderB.setValue(emit.b() * 2.)
