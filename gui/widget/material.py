from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget
)

from gui.widget.color import ColorButton
from path_tracer import MaterialType

class MaterialWidget(QGroupBox):
    def __init__(self, model, parent=None):
        super().__init__("Material", parent)

        self.model = model

        layout = QVBoxLayout()

        self.materialIDLabel = QLabel(self._materialIDText(), self)
        layout.addWidget(self.materialIDLabel)

        self.typeButton = MaterialTypeWidget(
            self.model.getMaterialType,
            self.model.setMaterialType,
            self
        )
        layout.addWidget(self.typeButton)

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

        self.update()

    def _materialIDText(self):
        return f"Material ID: {self.model.getMaterialID()}"

    def update(self):
        if self.model.getMaterialID() == -1:
            self.materialIDLabel.hide()
            self.typeButton.hide()
            self.albedoButton.hide()
            self.emitWidget.hide()
        else:
            self.materialIDLabel.show()
            self.typeButton.show()
            self.albedoButton.show()
            self.emitWidget.show()

            self.materialIDLabel.setText(self._materialIDText())
            self.typeButton.update()
            self.albedoButton.update()
            self.emitWidget.update()

class MaterialTypeWidget(QWidget):
    def __init__(self, getter, setter, parent=None):
        super().__init__(parent)

        self.getter = getter
        self.setter = setter

        layout = QHBoxLayout()

        self.materialLabel = QLabel("Type:")
        layout.addWidget(self.materialLabel)

        self.materialButton = QPushButton("")
        menu = QMenu(self)
        menu.addAction("Lambertian", lambda: self.handleStateChanged(MaterialType.Lambertian))
        menu.addAction("Mirror", lambda: self.handleStateChanged(MaterialType.Mirror))
        self.materialButton.setMenu(menu)

        layout.addWidget(self.materialButton)
        layout.addStretch()

        self.setLayout(layout)

    def update(self):
        material_type = self.getter()
        if material_type == MaterialType.Lambertian:
            self.materialButton.setText("Lambertian")
        elif material_type == MaterialType.Mirror:
            self.materialButton.setText("Mirror")
        else:
            print(material_type)
            print("Error, unsupported material type")
            exit(1)

    def handleStateChanged(self, material_type):
        self.setter(material_type)
        self.update()

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
