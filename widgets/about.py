from PySide2.QtCore import Qt, Signal, Slot
from PySide2.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QVBoxLayout,
                               QLabel, QListWidget, QPushButton)

class About(QDialog):
    sig_path_changed = Signal(str)

    def __init__(self, parent, pixmap):
        super(About, self).__init__(parent)

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("About AI-a Analysis System")
        self.resize(400,300)

        self.top_toolbar_widgets = self._setup_top_toolbar()
        self.bbox = QDialogButtonBox(QDialogButtonBox.Ok)

        logo = QLabel()
        logo.setPixmap(pixmap)
        logo.setScaledContents(False)
        
        description = QLabel(
            'AI-a Noise Recognition Analysis System\n'
            'Version: beta 0.0.1\n'
            'Release date: 2023 01 06\n'
            'Copyright © 2022 AI-Acoustic Technology Co.,Ltd\n'
            'E-mail: sales@ai-a.com.tw\n'
        )
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignCenter)
        top_layout = QHBoxLayout()
        self._add_widgets_to_layout(self.top_toolbar_widgets, top_layout)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.bbox)

        layout = QVBoxLayout()
        layout.addWidget(logo)
        layout.addWidget(description)
        # layout.addLayout(top_layout)
        layout.addLayout(bottom_layout)
        self.setLayout(layout)

        self.bbox.accepted.connect(self.accept)
        self.bbox.rejected.connect(self.reject)

    def _setup_top_toolbar(self):
        '''create top buttons'''
        self.movetop_button = QPushButton("Move to top")      
        self.movetop_button.clicked.connect(self.reject)  
        self.moveup_button = QPushButton('Move up')

        return  [self.movetop_button,self.moveup_button]

    def _add_widgets_to_layout(self, widgets, layout):
        '''add widget to layout'''
        layout.setAlignment(Qt.AlignLeft)
        for widget in widgets:
            if widget is None:
                layout.addStretch(1)
            else:
                layout.addWidget(widget)        