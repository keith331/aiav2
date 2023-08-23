from PySide2.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox
from PySide2.QtGui import QMovie
from PySide2.QtCore import Qt, Signal, Slot, QTimer
from PySide2 import QtCore
import logging
import os.path as osp

class InProgress(QDialog):
    
    def __init__(self, parent, fpath, thread=None):
        super(InProgress, self).__init__(parent, Qt.WindowTitleHint)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Operation in processing")
        self.resize(200,100)
        # self.setWindowModality(Qt.WindowModal) #子視窗需先關閉能可以關主視窗
        self.bbox = QDialogButtonBox(QDialogButtonBox.Cancel)

        if not osp.exists(fpath):
            logging.warning('File lost')
        movie = QMovie(fpath)

        label = QLabel()
        label.setMovie(movie)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.bbox, alignment=QtCore.Qt.AlignCenter)

        # self.bbox.accepted.connect(self.accept)
        self.bbox.rejected.connect(self.stop)

        self.setLayout(layout)
        movie.start()

        self.workingthread = thread 
        # self.workingthread.start()

        self.timer = QTimer(self)
        self.timer.setInterval(3000)
        self.timer.timeout.connect(self.finished)
        self.timer.start()

    def stop(self):

        self.workingthread.stop()
        self.reject()

    def finished(self):

        if self.timer is not None:
            self.timer.stop()
            self.accept()












