# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import resource_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1163, 908)
        MainWindow.setStyleSheet(u"QToolBar {\n"
"    spacing: 5px; /* spacing between items in the tool bar */\n"
"}\n"
"\n"
"QToolButton:hover {\n"
"    background: #bcc7d9;\n"
"}\n"
"\n"
"QStatusBar {\n"
"    border-width: 5px;\n"
"    border-color: blue;\n"
"}")
        MainWindow.setAnimated(True)
        self.actionRun = QAction(MainWindow)
        self.actionRun.setObjectName(u"actionRun")
        icon = QIcon()
        icon.addFile(u":/icons/images/play.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.actionRun.setIcon(icon)
        self.actionAbout = QAction(MainWindow)
        self.actionAbout.setObjectName(u"actionAbout")
        icon1 = QIcon()
        icon1.addFile(u":/icons/images/info.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.actionAbout.setIcon(icon1)
        self.actionRefresh = QAction(MainWindow)
        self.actionRefresh.setObjectName(u"actionRefresh")
        icon2 = QIcon()
        icon2.addFile(u":/icons/images/hard-drive.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.actionRefresh.setIcon(icon2)
        self.actionBacktest = QAction(MainWindow)
        self.actionBacktest.setObjectName(u"actionBacktest")
        icon3 = QIcon()
        icon3.addFile(u":/icons/images/refresh-cw.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.actionBacktest.setIcon(icon3)
        self.actionOpenwav = QAction(MainWindow)
        self.actionOpenwav.setObjectName(u"actionOpenwav")
        icon4 = QIcon()
        icon4.addFile(u":/icons/images/file-plus.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.actionOpenwav.setIcon(icon4)
        self.actionTraining = QAction(MainWindow)
        self.actionTraining.setObjectName(u"actionTraining")
        icon5 = QIcon()
        icon5.addFile(u":/icons/images/cpu.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.actionTraining.setIcon(icon5)
        self.actionpromptsn = QAction(MainWindow)
        self.actionpromptsn.setObjectName(u"actionpromptsn")
        self.actiontest = QAction(MainWindow)
        self.actiontest.setObjectName(u"actiontest")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setStyleSheet(u"")
        self.verticalLayout_3 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.horizontalLayout_2 = QHBoxLayout(self.widget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.widget_2 = QWidget(self.widget)
        self.widget_2.setObjectName(u"widget_2")
        self.verticalLayoutWidget = QWidget(self.widget_2)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(0, 540, 881, 241))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.widget1 = QWidget(self.widget_2)
        self.widget1.setObjectName(u"widget1")
        self.widget1.setGeometry(QRect(0, 0, 881, 531))
        self.verticalLayout_2 = QVBoxLayout(self.widget1)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)

        self.horizontalLayout_2.addWidget(self.widget_2)

        self.widget_3 = QWidget(self.widget)
        self.widget_3.setObjectName(u"widget_3")
        self.widget_3.setMinimumSize(QSize(240, 0))
        self.widget_3.setMaximumSize(QSize(240, 16777215))
        self.groupBox = QGroupBox(self.widget_3)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(11, 11, 218, 381))
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(10, 30, 121, 16))
        self.pushButton = QPushButton(self.groupBox)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(140, 140, 71, 28))
        self.layoutWidget = QWidget(self.groupBox)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(10, 190, 201, 166))
        self.formLayout = QFormLayout(self.layoutWidget)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.label_2 = QLabel(self.layoutWidget)
        self.label_2.setObjectName(u"label_2")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label_2)

        self.comboBox = QComboBox(self.layoutWidget)
        self.comboBox.addItem("")
        self.comboBox.setObjectName(u"comboBox")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.comboBox)

        self.label_3 = QLabel(self.layoutWidget)
        self.label_3.setObjectName(u"label_3")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_3)

        self.label_4 = QLabel(self.layoutWidget)
        self.label_4.setObjectName(u"label_4")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.label_4)

        self.comboBox_2 = QComboBox(self.layoutWidget)
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.setObjectName(u"comboBox_2")

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.comboBox_2)

        self.label_5 = QLabel(self.layoutWidget)
        self.label_5.setObjectName(u"label_5")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.label_5)

        self.lineEdit_3 = QLineEdit(self.layoutWidget)
        self.lineEdit_3.setObjectName(u"lineEdit_3")

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.lineEdit_3)

        self.label_6 = QLabel(self.layoutWidget)
        self.label_6.setObjectName(u"label_6")

        self.formLayout.setWidget(4, QFormLayout.LabelRole, self.label_6)

        self.lineEdit_4 = QLineEdit(self.layoutWidget)
        self.lineEdit_4.setObjectName(u"lineEdit_4")

        self.formLayout.setWidget(4, QFormLayout.FieldRole, self.lineEdit_4)

        self.label_7 = QLabel(self.layoutWidget)
        self.label_7.setObjectName(u"label_7")

        self.formLayout.setWidget(5, QFormLayout.LabelRole, self.label_7)

        self.comboBox_3 = QComboBox(self.layoutWidget)
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.setObjectName(u"comboBox_3")

        self.formLayout.setWidget(5, QFormLayout.FieldRole, self.comboBox_3)

        self.spinBox = QSpinBox(self.layoutWidget)
        self.spinBox.setObjectName(u"spinBox")
        self.spinBox.setMinimum(6)
        self.spinBox.setMaximum(20)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.spinBox)

        self.textEdit = QTextEdit(self.groupBox)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(10, 50, 201, 81))
        self.groupBox_2 = QGroupBox(self.widget_3)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(10, 400, 221, 151))
        font1 = QFont()
        font1.setBold(True)
        font1.setWeight(75)
        self.groupBox_2.setFont(font1)
        self.checkBox = QCheckBox(self.groupBox_2)
        self.checkBox.setObjectName(u"checkBox")
        self.checkBox.setGeometry(QRect(10, 120, 101, 31))
        self.widget2 = QWidget(self.groupBox_2)
        self.widget2.setObjectName(u"widget2")
        self.widget2.setGeometry(QRect(10, 50, 201, 58))
        self.formLayout_2 = QFormLayout(self.widget2)
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.label_8 = QLabel(self.widget2)
        self.label_8.setObjectName(u"label_8")

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.label_8)

        self.comboBox_4 = QComboBox(self.widget2)
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.setObjectName(u"comboBox_4")

        self.formLayout_2.setWidget(0, QFormLayout.FieldRole, self.comboBox_4)

        self.label_9 = QLabel(self.widget2)
        self.label_9.setObjectName(u"label_9")

        self.formLayout_2.setWidget(1, QFormLayout.LabelRole, self.label_9)

        self.pushButton_2 = QPushButton(self.widget2)
        self.pushButton_2.setObjectName(u"pushButton_2")

        self.formLayout_2.setWidget(1, QFormLayout.FieldRole, self.pushButton_2)

        self.label_11 = QLabel(self.widget_3)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(10, 580, 221, 161))

        self.horizontalLayout_2.addWidget(self.widget_3)


        self.verticalLayout_3.addWidget(self.widget)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menuBar = QMenuBar(MainWindow)
        self.menuBar.setObjectName(u"menuBar")
        self.menuBar.setGeometry(QRect(0, 0, 1163, 25))
        self.menuFile = QMenu(self.menuBar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuMode = QMenu(self.menuBar)
        self.menuMode.setObjectName(u"menuMode")
        self.menuSetup = QMenu(self.menuBar)
        self.menuSetup.setObjectName(u"menuSetup")
        self.menuAbout = QMenu(self.menuBar)
        self.menuAbout.setObjectName(u"menuAbout")
        self.menuHardware = QMenu(self.menuBar)
        self.menuHardware.setObjectName(u"menuHardware")
        MainWindow.setMenuBar(self.menuBar)
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName(u"toolBar")
        self.toolBar.setMinimumSize(QSize(0, 0))
        self.toolBar.setMaximumSize(QSize(16777215, 55))
        self.toolBar.setMovable(False)
        self.toolBar.setAllowedAreas(Qt.TopToolBarArea)
        self.toolBar.setIconSize(QSize(40, 40))
        self.toolBar.setFloatable(False)
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar)
        self.statusBar = QStatusBar(MainWindow)
        self.statusBar.setObjectName(u"statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.menuBar.addAction(self.menuFile.menuAction())
        self.menuBar.addAction(self.menuMode.menuAction())
        self.menuBar.addAction(self.menuSetup.menuAction())
        self.menuBar.addAction(self.menuHardware.menuAction())
        self.menuBar.addAction(self.menuAbout.menuAction())
        self.menuFile.addAction(self.actionOpenwav)
        self.menuMode.addSeparator()
        self.menuMode.addAction(self.actionRun)
        self.menuMode.addAction(self.actionBacktest)
        self.menuSetup.addAction(self.actionTraining)
        self.menuSetup.addAction(self.actionpromptsn)
        self.menuSetup.addAction(self.actiontest)
        self.menuAbout.addAction(self.actionAbout)
        self.menuHardware.addAction(self.actionRefresh)
        self.toolBar.addAction(self.actionRun)
        self.toolBar.addAction(self.actionOpenwav)
        self.toolBar.addAction(self.actionBacktest)
        self.toolBar.addAction(self.actionTraining)
        self.toolBar.addAction(self.actionRefresh)
        self.toolBar.addAction(self.actionAbout)
        self.toolBar.addSeparator()

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionRun.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.actionAbout.setText(QCoreApplication.translate("MainWindow", u"About", None))
        self.actionRefresh.setText(QCoreApplication.translate("MainWindow", u"Refresh", None))
        self.actionBacktest.setText(QCoreApplication.translate("MainWindow", u"Backtest", None))
        self.actionOpenwav.setText(QCoreApplication.translate("MainWindow", u"Openwav", None))
        self.actionTraining.setText(QCoreApplication.translate("MainWindow", u"Training", None))
        self.actionpromptsn.setText(QCoreApplication.translate("MainWindow", u"promptsn", None))
        self.actiontest.setText(QCoreApplication.translate("MainWindow", u"test", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Acquisition", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Autosave Folder Path", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"FFT Length", None))
        self.comboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"262144", None))

        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Duration", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"FFT Window", None))
        self.comboBox_2.setItemText(0, QCoreApplication.translate("MainWindow", u"Hanning", None))
        self.comboBox_2.setItemText(1, QCoreApplication.translate("MainWindow", u"Flat Top", None))
        self.comboBox_2.setItemText(2, QCoreApplication.translate("MainWindow", u"None", None))

        self.label_5.setText(QCoreApplication.translate("MainWindow", u"High-pass", None))
        self.lineEdit_3.setText(QCoreApplication.translate("MainWindow", u"20", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Low-pass", None))
        self.lineEdit_4.setText(QCoreApplication.translate("MainWindow", u"20000", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Weighting", None))
        self.comboBox_3.setItemText(0, QCoreApplication.translate("MainWindow", u"None", None))
        self.comboBox_3.setItemText(1, QCoreApplication.translate("MainWindow", u"A-wt", None))

        self.textEdit.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'PMingLiU'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">c:\\</p></body></html>", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Advanced Settings", None))
        self.checkBox.setText(QCoreApplication.translate("MainWindow", u"Mixed Noise", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Deep Learning", None))
        self.comboBox_4.setItemText(0, QCoreApplication.translate("MainWindow", u"CNN", None))
        self.comboBox_4.setItemText(1, QCoreApplication.translate("MainWindow", u"DNN", None))
        self.comboBox_4.setItemText(2, QCoreApplication.translate("MainWindow", u"2D_CNN", None))

        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Recording wav", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"Select", None))
        self.label_11.setText("")
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuMode.setTitle(QCoreApplication.translate("MainWindow", u"Run", None))
        self.menuSetup.setTitle(QCoreApplication.translate("MainWindow", u"Setup", None))
        self.menuAbout.setTitle(QCoreApplication.translate("MainWindow", u"Help", None))
        self.menuHardware.setTitle(QCoreApplication.translate("MainWindow", u"Hardware", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
    # retranslateUi

