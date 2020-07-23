from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

class CameraWidget(QGroupBox):
    def __init__(self, model, parent=None):
        super().__init__("Camera", parent)

        self.model = model

        layout = QVBoxLayout()

        origin = VectorWidget("Origin", model.getCameraOrigin, model.setCameraOrigin)
        layout.addWidget(origin)

        target = VectorWidget("Target", model.getCameraTarget, model.setCameraTarget)
        layout.addWidget(target)

        up = VectorWidget("Up", model.getCameraUp, model.setCameraUp)
        layout.addWidget(up)

        self.setLayout(layout)


class VectorWidget(QWidget):
    def __init__(self, name, getter, setter, parent=None):
        super().__init__(parent)

        self.getter = getter
        self.setter = setter

        layout = QHBoxLayout()

        label = QLabel(f"{name}:", self)
        layout.addWidget(label)

        vector = self.getter()

        self.x = QLineEdit()
        self.x.setFixedWidth(40)
        self.x.setText(str(vector.x()))
        self.x.editingFinished.connect(self.handleFinished)
        layout.addWidget(self.x)

        self.y = QLineEdit()
        self.y.setFixedWidth(40)
        self.y.setText(str(vector.y()))
        self.y.editingFinished.connect(self.handleFinished)
        layout.addWidget(self.y)

        self.z = QLineEdit()
        self.z.setFixedWidth(40)
        self.z.setText(str(vector.z()))
        self.z.editingFinished.connect(self.handleFinished)
        layout.addWidget(self.z)

        self.setLayout(layout)

    def handleFinished(self):
        self.setter(
            float(self.x.text()),
            float(self.y.text()),
            float(self.z.text())
        )
