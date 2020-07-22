from PyQt5.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget

from color_button import ColorButton

class MaterialWidget(QGroupBox):
    def __init__(self, model, parent=None):
        super().__init__("Material", parent)

        self.model = model

        layout = QVBoxLayout()

        self.albedoButton = ColorButton("Albedo", self.model, self)
        layout.addWidget(self.albedoButton)

        layout.addStretch()
        self.setLayout(layout)

        self.update()

    def update(self):
        self.albedoButton.update()

        if self.model.getMaterialIndex() == -1:
            self.albedoButton.hide()
        else:
            self.albedoButton.show()
