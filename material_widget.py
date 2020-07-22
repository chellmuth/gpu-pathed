from PyQt5.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget

from color_button import ColorButton

class MaterialWidget(QGroupBox):
    def __init__(self, model, parent=None):
        super().__init__("Material", parent)

        self.model = model

        layout = QVBoxLayout()

        self.materialIDLabel = QLabel(self._materialIDText(), self)
        layout.addWidget(self.materialIDLabel)

        self.albedoButton = ColorButton("Albedo", self.model, self)
        layout.addWidget(self.albedoButton)

        layout.addStretch()
        self.setLayout(layout)

        self.update()

    def _materialIDText(self):
        return f"Material ID: {self.model.getMaterialIndex()}"

    def update(self):
        self.materialIDLabel.setText(self._materialIDText())
        self.albedoButton.update()

        if self.model.getMaterialIndex() == -1:
            self.albedoButton.hide()
            self.materialIDLabel.hide()
        else:
            self.albedoButton.show()
            self.materialIDLabel.show()
