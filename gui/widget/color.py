from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QColorDialog, QHBoxLayout, QLabel, QPushButton, QWidget

class ColorButton(QWidget):
    def __init__(self, name, getter, setter, parent=None):
        super().__init__(parent)

        self.getter = getter
        self.setter = setter

        layout = QHBoxLayout()
        self.text = QLabel(f"{name}:", self)
        self.button = QPushButton("", self)
        self.button.setFixedSize(20, 20)

        layout.addWidget(self.text)
        layout.addWidget(self.button)
        layout.addStretch()

        self.setLayout(layout)

        self.button.clicked.connect(self.handlePush)

        self.update()

    def update(self):
        self.setColor(unwrapQcolor(self.getter()))

    def setColor(self, color):
        palette = self.button.palette()
        palette.setColor(QPalette.Button, color)
        self.button.setAutoFillBackground(True)
        self.button.setPalette(palette)

    def handlePush(self):
        color = QColorDialog.getColor(parent=self)

        self.setter(color.red() / 255., color.green() / 255., color.blue() / 255.)
        self.update()

def unwrapQcolor(vec3):
    return QColor(
        255 * vec3.r(),
        255 * vec3.g(),
        255 * vec3.b(),
    )
