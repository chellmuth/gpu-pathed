from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

class CameraWidget(QGroupBox):
    def __init__(self, model, parent=None):
        super().__init__("Camera", parent)

        self.model = model

        layout = QVBoxLayout()

        self.origin = VectorWidget("Origin", model.getCameraOrigin, model.setCameraOrigin)
        layout.addWidget(self.origin)

        self.target = VectorWidget("Target", model.getCameraTarget, model.setCameraTarget)
        layout.addWidget(self.target)

        self.up = VectorWidget("Up", model.getCameraUp, model.setCameraUp)
        layout.addWidget(self.up)

        self.setLayout(layout)

    def update(self):
        self.origin.update()
        self.target.update()
        self.up.update()


class VectorWidget(QWidget):
    def __init__(self, name, getter, setter, parent=None):
        super().__init__(parent)

        self.getter = getter
        self.setter = setter

        layout = QHBoxLayout()

        label = QLabel(f"{name}:", self)
        layout.addWidget(label)

        self.x = QLineEdit()
        self.x.setFixedWidth(40)
        self.x.editingFinished.connect(self.handleFinished)
        layout.addWidget(self.x)

        self.y = QLineEdit()
        self.y.setFixedWidth(40)
        self.y.editingFinished.connect(self.handleFinished)
        layout.addWidget(self.y)

        self.z = QLineEdit()
        self.z.setFixedWidth(40)
        self.z.editingFinished.connect(self.handleFinished)
        layout.addWidget(self.z)

        self.setLayout(layout)

        self.update()

    def handleFinished(self):
        modified = [
            self.x.isModified(),
            self.y.isModified(),
            self.z.isModified(),
        ]

        if not any(modified): return

        self.setter(
            strToFloat(self.x.text()),
            strToFloat(self.y.text()),
            strToFloat(self.z.text())
        )

        self.update()

    def update(self):
        vector = self.getter()

        if floatToStr(vector.x()) != self.x.text() and not self.x.isModified():
            self.x.setText(floatToStr(vector.x()))

        if floatToStr(vector.y()) != self.y.text() and not self.y.isModified():
            self.y.setText(floatToStr(vector.y()))

        if floatToStr(vector.z()) != self.z.text() and not self.z.isModified():
            self.z.setText(floatToStr(vector.z()))

def strToFloat(s):
    return round(float(s), 2)

def floatToStr(x):
    return str(round(x, 2))
