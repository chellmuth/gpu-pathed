from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QColorDialog, QHBoxLayout, QLabel, QPushButton, QWidget

class ColorButton(QWidget):
    def __init__(self, name, model, parent=None):
        super().__init__(parent)

        self.model = model

        layout = QHBoxLayout()
        self.text = QLabel(f"{name}:", self)
        self.button = QPushButton("", self)
        self.button.setFixedSize(20, 20)

        layout.addWidget(self.text)
        layout.addWidget(self.button)
        layout.addStretch()

        self.setLayout(layout)

        color = unwrapQcolor(self.model.getColor())
        self.setColor(color)

        self.button.clicked.connect(self.handlePush)

    def update(self):
        self.setColor(unwrapQcolor(self.model.getColor()))

    def setColor(self, color):
        palette = self.button.palette()
        palette.setColor(QPalette.Button, color)
        self.button.setAutoFillBackground(True)
        self.button.setPalette(palette)

    def handlePush(self):
        color = QColorDialog.getColor(parent=self)

        self.model.setColor(color.red() / 255., color.green() / 255., color.blue() / 255.)
        self.update()

def unwrapQcolor(vec3):
    return QColor(
        255 * vec3.r(),
        255 * vec3.g(),
        255 * vec3.b(),
    )
