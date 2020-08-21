from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from path_tracer import MaterialType

class MicrofacetWidget(QWidget):
    MaterialType = MaterialType.Microfacet

    def __init__(self, model, parent=None):
        super().__init__(parent)

        self.model = model

        layout = QVBoxLayout()

        self.alpha = AlphaWidget(
            self.model.getAlpha,
            self.model.setAlpha,
            self
        )
        layout.addWidget(self.alpha)

        layout.addStretch()
        self.setLayout(layout)

    def update(self):
        self.alpha.update()


class AlphaWidget(QWidget):
    def __init__(self, getter, setter, parent=None):
        super().__init__(parent)

        self.getter = getter
        self.setter = setter

        layout = QHBoxLayout()

        self.text = QLabel("Alpha: ", self)
        layout.addWidget(self.text)

        self.alpha = QDoubleSpinBox(self)
        self.alpha.setMinimum(0.01)
        self.alpha.setDecimals(2)
        self.alpha.setSingleStep(0.1)
        self.alpha.setValue(self.getter())
        self.alpha.valueChanged.connect(self.handleValueChanged)
        layout.addWidget(self.alpha)

        layout.addStretch()
        self.setLayout(layout)

        self.update()

    def handleValueChanged(self):
        self.setter(self.alpha.value())
        self.update()

    def update(self):
        alpha = self.getter()
        self.alpha.setValue(alpha)
