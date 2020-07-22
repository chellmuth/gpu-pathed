from PyQt5.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget

from color_button import ColorButton

class MaterialWidget(QGroupBox):
    def __init__(self, model, parent=None):
        super().__init__("Material", parent)

        self.model = model

        layout = QVBoxLayout()

        self.materialIDLabel = QLabel(self._materialIDText(), self)
        layout.addWidget(self.materialIDLabel)

        self.albedoButton = ColorButton(
            "Albedo",
            self.model.getColor,
            self.model.setColor,
            self
        )
        layout.addWidget(self.albedoButton)

        self.emitButton = ColorButton(
            "Emit",
            self.model.getEmit,
            self.model.setEmit,
            self
        )
        layout.addWidget(self.emitButton)

        layout.addStretch()
        self.setLayout(layout)

        self.update()

    def _materialIDText(self):
        return f"Material ID: {self.model.getMaterialIndex()}"

    def update(self):
        self.materialIDLabel.setText(self._materialIDText())
        self.albedoButton.update()
        self.emitButton.update()

        if self.model.getMaterialIndex() == -1:
            self.materialIDLabel.hide()
            self.albedoButton.hide()
            self.emitButton.hide()
        else:
            self.materialIDLabel.show()
            self.albedoButton.show()
            self.emitButton.show()
