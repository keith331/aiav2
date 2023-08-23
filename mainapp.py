import sys
import time
import clr
import logging
from threading import Thread
import datetime
import csv
import yaml
import os
import os.path as osp
import numpy as np
from numpy import asarray 

from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QInputDialog, QMessageBox, QComboBox, QLabel, QDialog, QVBoxLayout
from PySide2.QtCore import Slot, QTimer, Qt, QModelIndex, QThread
from PySide2.QtGui import QPixmap, QMovie
import pyqtgraph as pg

try:
    clr.AddReference(r"C:\Program Files\Audio Precision\APx500 8.0\API\AudioPrecision.API2.dll")
    clr.AddReference(r"C:\Program Files\Audio Precision\APx500 8.0\API\AudioPrecision.API.dll") 
    from AudioPrecision.API import *
except Exception:
    sys.exit()

from widgets.about import About
from widgets.training import Training
from models.heatmap import HeatMap
from ui_mainwindow import Ui_MainWindow
from utils import logger
from utils.wavhandler import WavHandler
from utils.logmodel import LogModel
from utils import misc
class MainWindow(QMainWindow):

    INITIAL_PATH = misc.get_inital_path()
    logger.use_file(INITIAL_PATH)
    APPROJX_FILE = misc.get_approjx_path('fan_lab.approjx')
    IMG_DIR = misc.get_image_dir()
    misc.create_save_dir()

    _about_widget = None
    _training_widget = None
    _progress_widget = None
    _mainthread = None

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('AI-a Noise Recognition Analysis System')
        self.setup_control()

        # self.apload()
        self.load_setting_params()
        self.set_setting_params()
        self.insert_chart()
        self.insert_heatmap_chart()
        self.result_data = []

    def setup_control(self):

        # set default values
        self.file_label = QLabel('') # for showing wavfile name on statusbar
        self.sn = 'autosn'
        self.task_type = QComboBox()
        self.ui.toolBar.addWidget(self.task_type)
        self.task_type.insertItems(1,self.get_models())
        self.recording = None

        self.ui.actionRun.triggered.connect(self.run_ap_sequence)
        self.ui.actionOpenwav.triggered.connect(self.get_wav_file)
        self.ui.actionBacktest.triggered.connect(self.run_backtest)
        self.ui.actionTraining.triggered.connect(self.show_training)
        self.ui.actionpromptsn.triggered.connect(self.prompt_sn)
        self.ui.actiontest.triggered.connect(self.show_inprogress)
        self.ui.actionAbout.triggered.connect(self.show_about)
        self.ui.pushButton.clicked.connect(self.set_autosave_folder)
        self.ui.pushButton_2.clicked.connect(self.get_recording)

    def apload(self):
        # APx = APx500_Application()
        # APx.Visible = 1
        # APx.Top = 0
        # APx.OpenProject(str(self.APPROJX_FILE))
        self.check_hardware_status()
        QApplication.processEvents()

    def ap_signal_analyzer(self):

        APx = APx500_Application()
        ''' High pass filter type 3 ( Butterworth )'''
        APx.SignalPathSetup.HighpassFilter = HighpassFilterMode.Butterworth
        ''' High pass filter frequency '''
        APx.SignalPathSetup.HighpassFilterFrequency = int(self.ui.lineEdit_3.text())
        ''' Low pass filter type 2 ( Butterworth ) '''
        APx.SignalPathSetup.LowpassFilterDigital = LowpassFilterModeDigital.Butterworth
        ''' Low pass filter frequency '''
        APx.SignalPathSetup.LowpassFilterFrequencyDigital = int(self.ui.lineEdit_4.text())            
        ''' Weighting '''
        APx.SignalPathSetup.WeightingFilter = SignalPathWeightingFilterType(self.ui.comboBox_3.currentIndex())
        ''' Set Acqquisition type as second '''
        APx.SignalAnalyzer.AcquisitionType = AcqLengthType.Seconds
        ''' Acq second '''
        APx.SignalAnalyzer.AcquisitionSeconds = self.ui.spinBox.value()
        ''' FFT length '''
        APx.SignalAnalyzer.FFTLengthSamples = 262144 #self.ui.comboBox.currentText()
        ''' FFT Window type '''
        window_types = {'Flat Top':6, 'Hanning':7, 'None':8}
        APx.SignalAnalyzer.WindowType = WindowType(window_types[self.ui.comboBox_2.currentText()])

        APx.Sequence.Run()
        # sys.exit()

    def run_ap_sequence(self):

        if self.is_model_ready():
        
            self.ax1.clear()
            self.ax2.clear()
            # if self.recording != None:

            #     sequence = Thread(target=self.ap_signal_analyzer, daemon=True)
            #     sequence.start()
            #     QApplication.processEvents()

            #     self.show_inprogress(self.recording, backtest=False)
            # else:
            #     self.show_warning_message('Select real-time recording file')

            sequence = Thread(target=self.ap_signal_analyzer, daemon=True)
            sequence.start()
            # QApplication.processEvents()

            self.show_inprogress()

        else:
            self.show_warning_message('The selected model has not been trained')

    def show_warning_message(self, message):

        messageBox = QMessageBox(self)
        messageBox.setWindowModality(Qt.NonModal)
        messageBox.setAttribute(Qt.WA_DeleteOnClose)
        messageBox.setIcon(QMessageBox.Warning)
        messageBox.setWindowTitle('Error')
        messageBox.setText(message)
        messageBox.setStandardButtons(QMessageBox.Ok)
        messageBox.show()

    def insert_chart(self):

        pg.setConfigOption('background', '#f0f0f0')
        pg.setConfigOption('foreground', '#121212')
        graphview = pg.GraphicsLayoutWidget()
                
        self.ui.verticalLayout_2.addWidget(graphview)

        self.ax1 = graphview.addPlot(title='WaveForm')
        self.ax1.setLabel('left', text='Amplitude')
        self.ax1.setLabel('bottom', text='Time', units='s')
        self.ax1.showGrid(x=True, y=True)
        self.ax1.setLogMode(x=False, y=False)
        # ax2.addLegend() 添加 legend

        graphview.nextRow()  # 垂直排列 layout 換行，不添加則為水平排列
        self.ax2 = graphview.addPlot(title='FFT')
        self.ax2.setLabel('left', text='Sound Pressure (dB)')
        self.ax2.setLabel('bottom', text='Frequency (Hz)')
        self.ax2.showGrid(x=True, y=True)
        self.ax2.setLogMode(x=True, y=False)

    def insert_heatmap_chart(self):

        pg.setConfigOption('background', '#f0f0f0')
        pg.setConfigOption('foreground', '#121212')
        # work version 
        self.heatmap_chart = pg.ImageView()
        self.heatmap_chart.clear()
        self.ui.verticalLayout.addWidget(self.heatmap_chart)
        self.heatmap_chart.ui.histogram.hide()
        self.heatmap_chart.ui.roiBtn.hide()
        self.heatmap_chart.ui.menuBtn.hide()
        
    def init_db(self): 

        # self.ui.tableView.resizeColumnsToContents()
        self.result_data.insert(0,[self.sn,'pass',90, self.get_time_stamp()])
        self.db = LogModel(self.result_data)
        self.ui.tableView.setModel(self.db)
        self.ui.tableView.setColumnWidth(3,200)
        # self.ui.tableView.selectionModel().selectionChanged.connect(self.on_change_selection)
        # self.ui.pushButton_2.clicked.connect(self.on_delete_db)

    def on_change_selection(self):
        indexes: list[QModelIndex] = self.ui.tableView.selectedIndexes()
        self.ui.pushButton_2.setEnabled(bool(indexes))

    def on_delete_db(self):
        indexes: list[QModelIndex] = self.ui.tableView.selectionModel().selectedIndexes()
        if not indexes:
            return
        self.db.removeRows(0)
        self.ui.tableView.clearSelection()

    def run_backtest(self):

        if self.is_model_ready():
        
            self.ax1.clear()
            self.ax2.clear()
            if self.file_label.text() != '':

                self.show_inprogress(self.current_wav_file, backtest=True)
                QApplication.processEvents()         
                
            else:
                self.show_warning_message('Select a wav file')
        else:
            self.show_warning_message('The selected model has not been trained')
            
    def remove_thread(self):
        self._mainthread.quit()
        self._mainthread.wait()
        self._mainthread = None
        print('Mainthread has been stopped')

    def update_charts(self, test_file):
        
        try:
            self.wav_handler = WavHandler(test_file)
            xt, yt = self.wav_handler.get_time_domain()
            if max(yt) >= -min(yt):
                ylim = max(yt)
            if max(yt) <- min(yt):
                ylim = -min(yt)
            if ylim==0:
                ylim = 1.1
            yt = yt/ylim
            self.ax1.plot(xt, yt, pen=pg.mkPen('#425a8c', width=0.5), ylim=(-1.1,1.1))

            fft_data = self.wav_handler.get_freq_domain()
            print('\n******************\n')
            xf = fft_data['freq']
            yf = fft_data['amplitude']
            print(len(yf))
            print('\n******************\n')
            self.max_freq = fft_data['max_freq']
            self.sharpness = fft_data['sharpness']
            self.ax2.plot(xf, yf, pen=pg.mkPen('#425a8c', width=0.7))           
            # self.ax2.setLimits(xMin=1000, xMax=10000)

            print(self.max_freq)
            print('===========')
            print(self.sharpness)

        except Exception as e:
            logging.info(f'Plotting chart error occured: {e}')

    def update_heatmap(self, heatmap):

        # self.ui.label_10.setPixmap(heatmap)
        # self.ui.label_10.setScaledContents(False)
        self.heatmap_chart.clear()
       
        imageitem = np.random.normal(size=(256, 256))

        import matplotlib.image as mpimg
        photo = np.array(mpimg.imread(heatmap))
        photo = photo.transpose([1,0,2])
        self.heatmap_chart.setImage(photo)

        # test version with known issue > image up side down

        # graphview = pg.GraphicsLayoutWidget()
        # self.ui.verticalLayout.addWidget(graphview)
        # self.heatmap_chart = graphview.addViewBox(row=1, col=1)
        # import matplotlib.image as mpimg
        # photo = np.array(mpimg.imread(heatmap))
        # photo = photo.transpose([1,0,2])
        # img = pg.ImageItem(photo)
        # self.heatmap_chart.addItem(img)
        # self.heatmap_chart.setAspectLocked(True)

    def update_result(self, result):

        'get return result [0] = result [1] = socre'
        if  result[0] == 'Pass':
            self.ui.label_11.setStyleSheet('''background-color: #00ff00;
                                              font-size: 26px;
                                              font-weight: bold;''')
            self.ui.label_11.setAlignment(Qt.AlignCenter)
            self.ui.label_11.setText(result[1])
            self.result = result[0]
        else:
            self.ui.label_11.setStyleSheet('''background-color: #ff0000;
                                              font-size: 26px;
                                              font-weight: bold;''')
            self.ui.label_11.setAlignment(Qt.AlignCenter)
            self.ui.label_11.setText(result[1])
            self.result = result[0]

        self.result_score = result[1]

    def is_model_ready(self):

        self.current_model = self.task_type.currentText()
        self.model_path = osp.join(misc.get_working_dir(), self.current_model)
        self.result_path = osp.join(self.model_path, 'result')

        hdf5_file = osp.join(self.result_path, 'Rnn5ep.hdf5')
        json_file = osp.join(self.result_path, 'Rnn5ep.json')

        if osp.isfile(hdf5_file) == True and osp.isfile(json_file) == True:
            return True
        else:
            return False

    def show_inprogress(self, test_file=None, backtest=False):

        def _dialog_finished():
            # self.save_log()
            self._progress_widget = None

        def finished():

            if self.timer is not None:
                self.timer.stop()
                dialog.accept()

        if self._progress_widget is None:
            fpath = osp.join(self.IMG_DIR, 'processing.gif')

            movie = QMovie(fpath)
            label = QLabel()
            label.setMovie(movie)
            movie.start()

            layout = QVBoxLayout()
            layout.addWidget(label)

            dialog = QDialog(self)
            dialog.setWindowTitle("Operation in processing")
            dialog.setModal(True)
            dialog.resize(200,100)
            dialog.setLayout(layout)

            self._progress_widget = dialog
            dialog.accepted.connect(_dialog_finished)
            dialog.rejected.connect(_dialog_finished)

            dialog.show()
            # QApplication.processEvents()
    
        save_path = osp.join(self.ui.textEdit.toPlainText())  
        if not backtest:

            time.sleep(self.ui.spinBox.value() + 2)  

                 
            files = [file for file in os.listdir(save_path) if file.endswith('.wav')]
            files.sort(key=lambda x: osp.getmtime(osp.join(save_path, x)), reverse=True)
            test_file = osp.join(save_path, files[0])

        self._mainthread = QThread()
        self.heatmapthread = HeatMap(self.model_path, test_file, save_path, self.sn, backtest)
        self.heatmapthread.moveToThread(self._mainthread)
        self._mainthread.started.connect(self.heatmapthread.run)
        self.heatmapthread.sig_finished.connect(dialog.accept)
        self.heatmapthread.sig_finished.connect(self.remove_thread)
        self.heatmapthread.sig_heatmap.connect(self.update_heatmap)
        self.heatmapthread.sig_result.connect(self.update_result)
        self._mainthread.start()
            
            # chart_thread = Thread(target=self.update_charts(test_file))
            # chart_thread.start()
            # chart_thread.join()   

        self.update_charts(test_file)
        QApplication.processEvents()

            # self.timer = QTimer(self)
            # self.timer.setInterval(3000)
            # self.timer.timeout.connect(finished)
            # self.timer.start()          

    def check_hardware_status(self):

        self.sample_rate = QLabel()
        self.sample_rate.mousePressEvent = self.refresh_hw()
        self.check_is_connected()
        APx = APx500_Application()

        input_lbl = QLabel('Input: ')

        input_ch = str(APx.SignalPathSetup.Asio.GetSelectedDevice().GetInputChannelAssignment(0).Name)
        input = APx.SignalPathSetup.InputSettings(APxInputSelection.Input1)
        mic_sens = round(input.Channels[0].Sensitivity.Value * 1000,1)
        mic_sens = str(mic_sens)
        mic_unit = str(input.Channels[0].Sensitivity.Unit)
        mic_info = QLabel(input_ch + ' ' + mic_sens + 'm' + mic_unit)
        mic_info.setStyleSheet(
                    'background: black;'
                    'color: limegreen')

        connector_type = str(APx.SignalPathSetup.InputConnector.Type)
        device = str(APx.SignalPathSetup.Asio.GetSelectedDevice().DeviceInfo.Name)
        selected_ch_nm = str(APx.SignalPathSetup.Asio.GetSelectedDevice().InputChannelCount)
        device_info = QLabel(device + ' ' + selected_ch_nm + ' Ch')
        device_info.setStyleSheet(
                    'background: black;'
                    'color: limegreen')       
        
        self.ui.statusBar.addPermanentWidget(input_lbl)
        self.ui.statusBar.addPermanentWidget(mic_info)  
        self.ui.statusBar.addPermanentWidget(device_info)
        self.ui.statusBar.addPermanentWidget(self.sample_rate)

    def check_is_connected(self):
        
        APx = APx500_Application()
        is_connected = APx.SignalPathSetup.Asio.IsConnected
        if not is_connected:
            self.sample_rate.setText('unlocked !')
            self.sample_rate.setStyleSheet(
                    'background: #8c0f0f;'
                    'color: #efefe6')
        else:       
            self.sample_rate.setText(str(APx.SignalPathSetup.Asio.GetSelectedDevice().SampleRate) + ' Hz')
            self.sample_rate.setStyleSheet(
                        'background: black;'
                        'color: limegreen')

    def refresh_hw(self):

        APx = APx500_Application()
        APx.SignalPathSetup.Asio.Reset()
        self.check_is_connected()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm', 'Are you sure?',
                QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            # APx = APx500_Application()
            # APx.Exit()
            self.save_setting_params()
            event.accept()
        else:
            event.ignore()

    def cancel_ap_operation(self):
        APx = APx500_Application()
        APx.CancelOperation()
        # APx.SuspendProcessing()
        print('stopped')
        # self.ui.actionSetting.setDisabled(True)

    def test(self):

        # self.ui.tableView.resizeColumnsToContents() 根據內容調整寬度
        # self.result_data.insert(0,[self.sn,'pass',90, self.get_time_stamp()])
        # self.logout = LogModel(self.result_data)
        # # self.ui.tableView.setModel(self.logout)
        # # self.ui.tableView.setColumnWidth(3,200)
        
        # str = self.ui.textEdit.toPlainText()
        # print('*'*30)
        # print(f'show autosave path: {str}')
        # print(f'Waveform file: {self.recording}')
        # print(f'time stamp: {self.sn}{self.get_time_stamp()}')

        # heatmap = osp.join(self.IMG_DIR, 'result.png')
        # self.ui.label_10.setPixmap(heatmap)
        # self.ui.label_10.setScaledContents(False)

        # if self.sn == 'autosn':
        #     self.ui.label_11.setStyleSheet('''background-color: #00ff00;''')
        #     self.ui.label_11.setText('Pass')
        # else:
        #     self.ui.label_11.setStyleSheet('''background-color: #ff0000;''')
        #     self.ui.label_11.setText('Fail')

        # self.save_log()

        save_path = osp.join(self.ui.textEdit.toPlainText())
        print(save_path)

    def prompt_sn(self):
        
        unwanted_chars = set('\/:*?<>|"')
        input_dialog = QInputDialog(self)
        input_dialog.setInputMode(QInputDialog.TextInput)
        input_dialog.setModal(True)
        input_dialog.setWindowTitle('Prompt SN')
        input_dialog.setLabelText('Input Serial Number')
        input_dialog.setFixedSize(400,150)
        input_dialog.show()

        if input_dialog.exec_():

            if any((c in unwanted_chars) for c in input_dialog.textValue()):
                logging.info('sn includes illegal characters')
                self.sn = 'autosn'
            else:
                self.sn = input_dialog.textValue()

    def get_models(self):

        models = [] 
        for i in os.listdir(misc.get_working_dir()):
            models.append(i)
        return models  
              
    def get_wav_file(self):
        self.current_wav_file,_ = QFileDialog.getOpenFileName(self, 'Open wav file', directory=self.INITIAL_PATH, filter='*.wav')
        self.file_label.setText(misc.get_fname_string(self.current_wav_file))        
        self.ui.statusBar.addWidget(self.file_label)

    def get_recording(self):
        self.recording,_ = QFileDialog.getOpenFileName(self, 'Set recording file', 'c:/', filter='*.wav')

    def set_autosave_folder(self):
        self.autosave_folder = QFileDialog.getExistingDirectory()
        string = self.autosave_folder.replace('/',"\\")
        self.ui.textEdit.setText(string)
        misc.create_save_dir(self.autosave_folder)

    def get_time_stamp(self):
        current_time = datetime.datetime.now()
        return current_time.strftime('%m_%d_%Y_%H_%M_%S')

    def save_setting_params(self):

        config_dict = {}
        config_dict['acquisition'] = {}
        config_dict['advanced_settings'] = {}
        config_dict['acquisition']['autosave_folder'] = self.ui.textEdit.toPlainText()
        config_dict['acquisition']['duration'] = self.ui.spinBox.value()
        config_dict['acquisition']['window'] = self.ui.comboBox_2.currentText()
        config_dict['acquisition']['high_pass'] = self.ui.lineEdit_3.text()
        config_dict['acquisition']['low_pass'] = self.ui.lineEdit_4.text()
        config_dict['acquisition']['weighting'] = self.ui.comboBox_3.currentText()
        config_dict['advanced_settings']['deep_learning'] = self.ui.comboBox_4.currentText()
        config_dict['advanced_settings']['mixed_noise'] = self.ui.checkBox.isChecked()

        with open(r'configuration.yaml', 'w') as file:
            yaml.dump(config_dict, file)

    def load_setting_params(self):

        with open('configuration.yaml', 'r') as f:

            try:
                config = yaml.load(f, Loader=yaml.FullLoader)
                self.param_autosave_folder = config['acquisition']['autosave_folder']
                self.param_duration = config['acquisition']['duration']
                self.param_window = config['acquisition']['window']
                self.param_high_pass = config['acquisition']['high_pass']
                self.param_low_pass = config['acquisition']['low_pass']
                self.param_weighting = config['acquisition']['weighting']
                self.param_dltype = config['advanced_settings']['deep_learning']
                self.param_mixed_noise = config['advanced_settings']['mixed_noise']

            except Exception as e:
                logging.error('Config file content error',e)

    def set_setting_params(self):

        self.ui.textEdit.setText(self.param_autosave_folder)
        self.ui.spinBox.setValue(self.param_duration)
        self.ui.comboBox_2.setCurrentText(self.param_window)
        self.ui.lineEdit_3.setText(self.param_high_pass)
        self.ui.lineEdit_4.setText(self.param_low_pass)
        self.ui.comboBox_3.setCurrentText(self.param_weighting)
        self.ui.comboBox_4.setCurrentText(self.param_dltype)
        self.ui.checkBox.setChecked(self.param_mixed_noise)

    @Slot()
    def show_training(self):
        
        def _dialog_finished():
            self._training_widget = None

            # update task type selection
            self.task_type.clear()
            self.task_type.insertItems(1,self.get_models())

        if self._training_widget is None:

            dl_type = self.ui.comboBox_4.currentText()
            is_mixed = self.ui.checkBox.isChecked()
            dialog = Training(self, dl_type, is_mixed)
            self._training_widget = dialog

            dialog.finished.connect(_dialog_finished)
            dialog.show()
            QApplication.processEvents()
            dialog.exec_()
        else:
            self._training_widget.show()
            self._training_widget.activateWindow()
            self._training_widget.raise_()
            self._training_widget.setFocus()

    @Slot()
    def show_about(self):
        def _dialog_finished():
            self._about_widget = None
        if self._about_widget is None:
            fpath = osp.join(self.IMG_DIR, 'logo.png')   
            pixmap = QPixmap(fpath)
            pixmap = pixmap.scaled(400,400, Qt.KeepAspectRatio)     
            dialog = About(self, pixmap)
            self._about_widget = dialog

            dialog.finished.connect(_dialog_finished)
            dialog.show()
            dialog.exec_()
        else:
            self._about_widget.show()
            self._about_widget.activateWindow()
            self._about_widget.raise_()
            self._about_widget.setFocus()

    def save_log(self):

        self.result_data = [self.sn, self.result , self.result_score, self.max_freq, self.sharpness, self.get_time_stamp()]
        csv_path = osp.join(self.ui.textEdit.toPlainText(), 'result.csv')
        try: 
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.result_data)
        except Exception as e:
            logging.info(f'Saving log file went wrong {e}')

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__': 
    mainapp = Thread(target=main)
    mainapp.start()
    mainapp.join()

    