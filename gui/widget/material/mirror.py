from PyQt5.QtWidgets import QWidget, QVBoxLayout

from path_tracer import MaterialType

class MirrorWidget(QWidget):
    MaterialType = MaterialType.Mirror

    def __init__(self, model, parent=None):
        super().__init__(parent)

        self.model = model

        layout = QVBoxLayout()
        layout.addStretch()
        self.setLayout(layout)

    def update(self):
        pass
