from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import cv2, imutils
import numpy as np
import tensorflow as tf
from keras.models import load_model


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(719, 550)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        MainWindow.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setBold(True)
        font.setWeight(75)
        MainWindow.setFont(font)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.bOpen = QtWidgets.QPushButton(self.centralwidget)
        self.bOpen.setObjectName("bOpen")
        self.horizontalLayout.addWidget(self.bOpen)
        spacerItem = QtWidgets.QSpacerItem(648, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("")
        # self.label.setPixmap(QtGui.QPixmap("../2.jpeg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.bPredict = QtWidgets.QPushButton(self.centralwidget)
        self.bPredict.setObjectName("bPredict")
        self.horizontalLayout_2.addWidget(self.bPredict)
        self.lcdNumber = QtWidgets.QLCDNumber(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lcdNumber.sizePolicy().hasHeightForWidth())
        self.lcdNumber.setSizePolicy(sizePolicy)
        self.lcdNumber.setFrameShape(QtWidgets.QFrame.Panel)
        self.lcdNumber.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.lcdNumber.setObjectName("lcdNumber")
        self.horizontalLayout_2.addWidget(self.lcdNumber)
        spacerItem1 = QtWidgets.QSpacerItem(578, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.gridLayout.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.bOpen.clicked.connect(self.loadImage)
        # self.bPredict.clicked.connect(self.lcdNumber.setBinMode)
        self.bPredict.clicked.connect(self.predict)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        

    def loadImage(self):
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        self.image = cv2.imread(self.filename)
        self.setPhoto(self.image)

    #method to display image in the GUI
    def setPhoto(self,image):
        try:
            self.tmp = image
            image = imutils.resize(image,width=640)
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(image))
        except: return

    def predict(self):
        try:
            image = self.preprocessed_img(self.image)
            matrix = np.zeros(shape=(1, 224, 1))
            matrix[0] = image
            model = load_model('covid_X-ray.h5')
            outcome = model.predict(matrix)[0][0]
            if outcome > 0.5:
                self.lcdNumber.display(1)
            else:
                self.lcdNumber.display(0)
            print(outcome)
        except: return
    # Grayscale
    def rgb2gray(self,rgb_img):
        output_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        return output_img
    # Scale to 0 to 1
    def scale(self,image):
        return image / 255
    #resize
    def resize_img(self,image, rows=224, cols=224):
        return cv2.resize(image, dsize=(rows, cols), interpolation=cv2.INTER_CUBIC)
    # resize the shape
    def reshape(self,image, axis):
        return np.expand_dims(image.mean(axis=axis), axis=1)
    # Function to call other Preprocessing Functions
    def preprocessed_img(self,input_img):
        output_img = self.rgb2gray(input_img)
        output_img = self.scale(output_img)
        output_img = self.resize_img(output_img)
        output_img = self.reshape(output_img, 1)
        return output_img





    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Net Covid Predict"))
        self.bOpen.setText(_translate("MainWindow", "OPEN"))
        self.bPredict.setText(_translate("MainWindow", "PREDICT"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
