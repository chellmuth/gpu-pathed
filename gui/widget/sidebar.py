from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget
)

from gui.widget.camera import CameraWidget
from gui.widget.material import MaterialWidget
from gui.widget.settings import SettingsWidget

class SidebarWidget(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)

        self.model = model

        # Materials group
        self.materialGroup = MaterialWidget(self.model, self)

        # Camera group
        self.cameraGroup = CameraWidget(self.model, self)

        # Settings group
        self.settingsGroup = SettingsWidget(self.model, self)

        # Info group
        self.infoGroup = QGroupBox("Info", self)
        infoLayout = QVBoxLayout()

        self.spp = SppLabel(model, self)
        infoLayout.addWidget(self.spp)

        self.infoGroup.setLayout(infoLayout)

        # Sidebar layout
        layout = QVBoxLayout()
        layout.addWidget(self.materialGroup)
        layout.addWidget(self.cameraGroup)
        layout.addWidget(self.settingsGroup)
        layout.addWidget(self.infoGroup)
        layout.addStretch()
        self.setLayout(layout)

    def update(self):
        self.materialGroup.update()
        self.cameraGroup.update()
        self.spp.update()

class SppLabel(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)

        self.model = model

        layout = QHBoxLayout()
        self.sppLabel = QLabel(self._sppLabelText())
        layout.addWidget(self.sppLabel)
        layout.addStretch()

        self.setLayout(layout)

    def update(self):
        self.sppLabel.setText(self._sppLabelText())

    def _sppLabelText(self):
        return f"spp: {self.model.getSpp()}"
