from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from path_tracer import MaterialType

class GlassWidget(QWidget):
    MaterialType = MaterialType.Glass

    def __init__(self, model, parent=None):
        super().__init__(parent)

        self.model = model

        layout = QVBoxLayout()

        self.ior = IORWidget(
            self.model.getIOR,
            self.model.setIOR,
            self
        )
        layout.addWidget(self.ior)

        layout.addStretch()
        self.setLayout(layout)

    def update(self):
        self.ior.update()


class IORWidget(QWidget):
    def __init__(self, getter, setter, parent=None):
        super().__init__(parent)

        self.getter = getter
        self.setter = setter

        layout = QHBoxLayout()

        self.text = QLabel("IOR: ", self)
        layout.addWidget(self.text)

        self.ior = QDoubleSpinBox(self)
        self.ior.setMinimum(0.01)
        self.ior.setDecimals(2)
        self.ior.setSingleStep(0.1)
        self.ior.setValue(self.getter())
        self.ior.valueChanged.connect(self.handleValueChanged)
        layout.addWidget(self.ior)

        layout.addStretch()
        self.setLayout(layout)

        self.update()

    def handleValueChanged(self):
        self.setter(self.ior.value())
        self.update()

    def update(self):
        ior = self.getter()
        self.ior.setValue(ior)
