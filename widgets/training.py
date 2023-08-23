from PySide2.QtCore import Qt, Signal, Slot, QThread
from PySide2.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QVBoxLayout,QMessageBox,
                               QLabel, QPushButton, QComboBox, QLineEdit)
from PySide2.QtGui import QPixmap, QMovie
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import os
import os.path as osp
import time
import itertools
import numpy as np
from utils.misc import get_working_dir
from models.cnn_model import CNNModel
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# import matplotlib.pyplot as plt

class Training(QDialog):

    def __init__(self, parent, dl_type, is_mixed):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Training")
        self.resize(600,600)
        self.setup_layout()

        self.dl_type = dl_type
        self.is_mixed = is_mixed
        self.ready_to_train = False
        # self._thread = None

    def setup_layout(self):

        self.bbox = QDialogButtonBox(QDialogButtonBox.Ok)
        
        label = QLabel('New model name:')
        self.model_name = QLineEdit()
        self.pushButton = QPushButton('Create New Model')
        self.pushButton.clicked.connect(self.create_new_model_dictionary)

        top_layout = QHBoxLayout()
        top_layout.addWidget(label)
        top_layout.addWidget(self.model_name)
        top_layout.addWidget(self.pushButton)

        self.task_type = QComboBox()
        self.task_type.insertItems(1,self.get_models())

        self.pushButton_1 = QPushButton('Start Training')
        self.pushButton_1.clicked.connect(self.start_training)

        self.training_chart = QVBoxLayout()

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.bbox)
        bottom_layout.setAlignment(Qt.AlignBottom)

        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.task_type)
        layout.addWidget(self.pushButton_1)
        layout.addLayout(self.training_chart)
        layout.addLayout(bottom_layout)
        self.setLayout(layout)

        self.bbox.accepted.connect(self.accept)
        self.bbox.rejected.connect(self.reject)

        self.insert_chart()

    def insert_chart(self):

        pg.setConfigOption('background', '#f0f0f0')
        pg.setConfigOption('foreground', '#121212')
        graphview = pg.GraphicsLayoutWidget()
        self.training_chart.addWidget(graphview)

        self.training_chart = graphview.addPlot()
        self.training_chart.setTitle('Confusion matrix for training results', size='16pt')
        self.training_chart.invertY(True)            # orient y axis to run top-to-bottom
        self.training_chart.setDefaultPadding(0.0)   # plot without padding data range
        
        # show full frame, label tick marks at top and bottom sides, with some extra space for labels:    
        self.training_chart.showAxes( True, showValues=True)

        # define major tick marks and labels:
        columns = ["Pass", "Fail"]
        self.font=QtGui.QFont()
        self.font.setPixelSize(20)
        ticks = [ (idx, label) for idx, label in enumerate( columns ) ]
        for side in ('left','top','right','bottom'):
            self.training_chart.getAxis(side).setTicks( (ticks, []) ) # add list of major ticks; no minor ticks
            self.training_chart.getAxis(side).setTickFont(self.font)

        # self.training_chart.setLabel('left', text='True label')
        # self.training_chart.setLabel('bottom', text='Predicted Ratio')
        # self.training_chart.showGrid(x=True, y=True)
        # self.training_chart.setLogMode(x=False, y=False)

    def update_cm_chart(self, confusion_array):

        self.training_chart.removeItem(self.text)

        cm = confusion_array
        self.training_chart.clear()
        corrMatrix = np.array([
            [ 1.        ,  0.5184571],
            [ 0.5184571 ,  1.       ],
        ])
        
        pg.setConfigOption('imageAxisOrder', 'row-major') # Switch default order to Row-major
        
        correlogram = pg.ImageItem()
        # create transform to center the corner element on the origin, for any assigned image:
        tr = QtGui.QTransform().translate(-0.5, -0.5) 
        correlogram.setTransform(tr)
        correlogram.setImage(cm)        

        # self.training_chart.getAxis('bottom').setHeight(5)           # include some additional space at bottom of figure

        colorMap = pg.colormap.get("CET-D1")  # choose perceptually uniform, diverging color map

        self.training_chart.addItem(correlogram)     # display correlogram

        # generate an adjustabled color bar, initially spanning -1 to 1:
        bar = pg.ColorBarItem( values=(-1,1), colorMap=colorMap)
        # link color bar and color map to correlogram, and show it in plotItem:
        bar.setImageItem(correlogram, insert_in=self.training_chart)

    def start_training(self):

        txt = self.check_source_files()
        self.training_chart.clear()

        if self.ready_to_train:

            self.text = pg.TextItem(str('Loading...'))
            self.text.setFont(self.font)
            self.training_chart.addItem(self.text)
            self.text.setPos(30,20)

            self._thread = QThread(self)
            self.cnnthread = CNNModel(self.model_path, self.cm_xlabel)
            self.cnnthread.moveToThread(self._thread)
            self.cnnthread.sig_finished.connect(self.remove_thread)
            self.cnnthread.sig_cm.connect(self.update_cm_chart)
            self._thread.started.connect(self.cnnthread.run)
            self._thread.start()
        else:
            self.show_messge_box(txt)

    def remove_thread(self):
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        print('_thread has been stopped')

    def create_new_model_dictionary(self):

        USED_DIRS = ['feature','training','result']
        CATEGORY_DIRS = ['one','zero','one_testing','zero_testing']
        unwanted_chars = set('\/:*?<>|" ')

        if any((c in unwanted_chars) for c in self.model_name.text()) or self.model_name.text().strip() == '':
            model_name = 'autodir'
        else:
            model_name = self.model_name.text()
        try:
            os.makedirs(osp.join(get_working_dir(), model_name), exist_ok=True)
            model_path = osp.join(get_working_dir(), model_name)
        except FileExistsError:
            print("That's OK.")
        else:
            for dir in USED_DIRS:
                os.makedirs(osp.join(model_path, dir), exist_ok=True)
            
        category_path = osp.join(model_path, 'training')
        for cat in CATEGORY_DIRS:
            os.makedirs(osp.join(category_path, cat), exist_ok=True)
    
        self.show_messge_box('Finished')
        self.model_name.clear()

        # renew task_type items
        self.task_type.clear()
        self.task_type.insertItems(1,self.get_models())

    def show_messge_box(self, message):

        messageBox = QMessageBox(self)
        messageBox.setWindowModality(Qt.NonModal)
        messageBox.setAttribute(Qt.WA_DeleteOnClose)
        messageBox.setIcon(QMessageBox.Warning)
        messageBox.setWindowTitle('Error')
        messageBox.setText(message)
        messageBox.setStandardButtons(QMessageBox.Ok)
        messageBox.show()

    def get_models(self):

        models = [] 
        for i in os.listdir(get_working_dir()):
            models.append(i)
        return models  

    def check_source_files(self):

        self.current_model = self.task_type.currentText()
        self.model_path = osp.join(get_working_dir(), self.current_model)

        self.source0_path = osp.join(self.model_path, 'training', 'zero')
        self.source1_path = osp.join(self.model_path, 'training', 'one')
        self.source0_nums = len(os.listdir(self.source0_path))
        self.source1_nums = len(os.listdir(self.source1_path))

        self.test0_path = osp.join(self.model_path, 'training', 'zero_testing')
        self.test1_path = osp.join(self.model_path, 'training', 'one_testing')
        self.test0_nums = len(os.listdir(self.test0_path))
        self.test1_nums = len(os.listdir(self.test1_path))

        if self.source0_nums != self.source1_nums:
            self.ready_to_train = False
            return 'The source files are not the same.'

        if self.source0_nums == 0 or self.source1_nums == 0:
            self.ready_to_train = False
            return 'The source files are empty.'

        if self.test0_nums != self.test1_nums:
            self.ready_to_train = False
            return 'The testing files are not the same.'

        if self.test0_nums == 0  or self.test1_nums == 0:
            self.ready_to_train = False
            return 'The testing files are empty.'

        self.cm_xlabel = list(itertools.repeat(0,self.test0_nums)) + list(itertools.repeat(1,self.test1_nums))

        self.ready_to_train = True
        print(self.cm_xlabel)




