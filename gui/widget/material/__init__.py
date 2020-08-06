from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget
)

from gui.widget.material.lambertian import LambertianWidget
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

        self.materialWidget = LambertianWidget(self.model, self)
        layout.addWidget(self.materialWidget)

        layout.addStretch()
        self.setLayout(layout)

        self.update()

    def _materialIDText(self):
        return f"Material ID: {self.model.getMaterialID()}"

    def update(self):
        if self.model.getMaterialID() == -1:
            self.materialIDLabel.hide()
            self.typeButton.hide()
            self.materialWidget.hide()
        else:
            self.materialIDLabel.show()
            self.typeButton.show()
            self.materialWidget.show()

            self.materialIDLabel.setText(self._materialIDText())
            self.typeButton.update()
            self.materialWidget.update()

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
            print("Error, unsupported material type")
            exit(1)

    def handleStateChanged(self, material_type):
        self.setter(material_type)
        self.update()
