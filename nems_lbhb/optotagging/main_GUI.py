# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './main_GUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_mainWidget(object):
    def setupUi(self, mainWidget):
        mainWidget.setObjectName("mainWidget")
        mainWidget.setEnabled(True)
        mainWidget.resize(956, 657)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(mainWidget.sizePolicy().hasHeightForWidth())
        mainWidget.setSizePolicy(sizePolicy)
        self.gridLayoutWidget = QtWidgets.QWidget(mainWidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(30, 10, 891, 601))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.activatedButton = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.activatedButton.sizePolicy().hasHeightForWidth())
        self.activatedButton.setSizePolicy(sizePolicy)
        self.activatedButton.setObjectName("activatedButton")
        self.horizontalLayout.addWidget(self.activatedButton)
        self.neutralButton = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.neutralButton.sizePolicy().hasHeightForWidth())
        self.neutralButton.setSizePolicy(sizePolicy)
        self.neutralButton.setObjectName("neutralButton")
        self.horizontalLayout.addWidget(self.neutralButton)
        self.suppresedButton = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.suppresedButton.sizePolicy().hasHeightForWidth())
        self.suppresedButton.setSizePolicy(sizePolicy)
        self.suppresedButton.setObjectName("suppresedButton")
        self.horizontalLayout.addWidget(self.suppresedButton)
        self.gridLayout.addLayout(self.horizontalLayout, 2, 1, 1, 2)
        self.neuronList = QtWidgets.QTreeWidget(self.gridLayoutWidget)
        self.neuronList.setColumnCount(3)
        self.neuronList.setObjectName("neuronList")
        self.neuronList.headerItem().setText(0, "1")
        self.neuronList.headerItem().setText(1, "2")
        self.neuronList.headerItem().setText(2, "3")
        self.gridLayout.addWidget(self.neuronList, 1, 1, 1, 2)
        self.siteLoadButton = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.siteLoadButton.sizePolicy().hasHeightForWidth())
        self.siteLoadButton.setSizePolicy(sizePolicy)
        self.siteLoadButton.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.siteLoadButton.setObjectName("siteLoadButton")
        self.gridLayout.addWidget(self.siteLoadButton, 0, 1, 1, 1)
        self.canvasLayout = QtWidgets.QHBoxLayout()
        self.canvasLayout.setObjectName("canvasLayout")
        self.gridLayout.addLayout(self.canvasLayout, 0, 0, 3, 1)
        self.siteSelecDropDown = QtWidgets.QComboBox(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.siteSelecDropDown.sizePolicy().hasHeightForWidth())
        self.siteSelecDropDown.setSizePolicy(sizePolicy)
        self.siteSelecDropDown.setObjectName("siteSelecDropDown")
        self.gridLayout.addWidget(self.siteSelecDropDown, 0, 2, 1, 1)

        self.retranslateUi(mainWidget)
        QtCore.QMetaObject.connectSlotsByName(mainWidget)

    def retranslateUi(self, mainWidget):
        _translate = QtCore.QCoreApplication.translate
        mainWidget.setWindowTitle(_translate("mainWidget", "opto_id.py"))
        self.activatedButton.setText(_translate("mainWidget", "activated"))
        self.neutralButton.setText(_translate("mainWidget", "neutral"))
        self.suppresedButton.setText(_translate("mainWidget", "suppresed"))
        self.siteLoadButton.setText(_translate("mainWidget", "load site"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWidget = QtWidgets.QWidget()
    ui = Ui_mainWidget()
    ui.setupUi(mainWidget)
    mainWidget.show()
    sys.exit(app.exec_())

