from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

class CameraWidget(QGroupBox):
    def __init__(self, model, parent=None):
        super().__init__("Camera", parent)

        self.model = model

        layout = QVBoxLayout()

        origin = VectorWidget("Origin", model, model.getCameraOrigin)
        layout.addWidget(origin)

        target = VectorWidget("Target", model, model.getCameraTarget)
        layout.addWidget(target)

        up = VectorWidget("Up", model, model.getCameraUp)
        layout.addWidget(up)

        self.setLayout(layout)


class VectorWidget(QWidget):
    def __init__(self, name, model, getter, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()

        label = QLabel(f"{name}:", self)
        layout.addWidget(label)

        vector = getter()

        x = QLineEdit()
        x.setFixedWidth(40)
        x.setText(str(vector.x()))
        layout.addWidget(x)

        y = QLineEdit()
        y.setFixedWidth(40)
        y.setText(str(vector.y()))
        layout.addWidget(y)

        z = QLineEdit()
        z.setFixedWidth(40)
        z.setText(str(vector.z()))
        layout.addWidget(z)

        self.setLayout(layout)
