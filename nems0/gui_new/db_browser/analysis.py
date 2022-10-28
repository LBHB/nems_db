import sys
import pandas as pd
from pathlib import Path
from functools import partial

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5 import uic

from nems0 import db as nd
from nems.utils import simple_search, load_settings, save_settings

qt_creator_file = Path(r'ui') / 'analysis.ui'
Ui_Widget, QtBaseClass = uic.loadUiType(qt_creator_file)

Qt = QtCore.Qt
import db_test


class pandasModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        QtCore.QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


class ExampleApp(QtBaseClass, Ui_Widget):
    def __init__(self, parent=None):
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)

        d = load_settings('analysis')
        self.current_analysis = d.get('analysis', '')
        self.lastbatch = ''

        self.all_models = []
        self.all_cellids = []
        self.batch_data = []
        self.analysis_data = []

        self.refresh_batches()

        self.pushUpdate.clicked.connect(self.refresh_lists)
        self.comboBatch.currentIndexChanged.connect(self.reload_models)
        self.comboAnalysis.currentIndexChanged.connect(self.analysis_update)
        self.pushScatter.clicked.connect(partial(self.analysis, 'scatter'))
        self.pushBar.clicked.connect(partial(self.analysis, 'bar'))
        self.pushPareto.clicked.connect(partial(self.analysis, 'pareto'))

    def refresh_batches(self):

        sql = "SELECT * FROM Analysis order by id"
        self.analysis_data = nd.pd_query(sql)
        model = QtGui.QStandardItemModel()
        for i in self.analysis_data['name'].to_list():
            item = QtGui.QStandardItem(i)
            model.appendRow(item)
        self.comboAnalysis.setModel(model)
        index = self.comboAnalysis.findText(self.current_analysis, QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.comboAnalysis.setCurrentIndex(index)

        sql = "SELECT DISTINCT batch FROM Batches order by batch"
        self.batch_data = nd.pd_query(sql)
        model = QtGui.QStandardItemModel()
        for i in self.batch_data['batch'].to_list():
            item = QtGui.QStandardItem(i)
            model.appendRow(item)
        self.comboBatch.setModel(model)

        self.analysis_update()

    def reload_models(self):
        t = self.comboBatch.currentText()
        sql = f"SELECT DISTINCT modelname FROM Results WHERE batch={t}"
        data = nd.pd_query(sql)
        self.all_models = data['modelname'].to_list()
        sql = f"SELECT DISTINCT cellid FROM Batches WHERE batch={t}"
        data = nd.pd_query(sql)
        self.all_cellids = data['cellid'].to_list()
        self.lastbatch = t

    def analysis_update(self):
        self.current_analysis = self.comboAnalysis.currentText()
        analysis_info = self.analysis_data[
            self.analysis_data.name == self.current_analysis].reset_index()

        if analysis_info['model_search'][0] is not None:
            self.lineModelname.setText(analysis_info['model_search'][0])
        if analysis_info['cell_search'][0] is not None:
            self.lineCellid.setText(analysis_info['cell_search'][0])

        current_batch = analysis_info['batch'].values[0].split(":")[0]
        if current_batch != self.comboBatch.currentText():
            index = self.comboBatch.findText(current_batch, QtCore.Qt.MatchFixedString)
            if index >= 0:
                self.comboBatch.setCurrentIndex(index)
                self.refresh_lists()
        save_settings('analysis', {'analysis': self.current_analysis})

    def refresh_lists(self):

        t = self.comboBatch.currentText()
        if t != self.lastbatch:
            self.reload_models()

        cell_search = self.lineCellid.text()
        refined_list = simple_search(cell_search, self.all_cellids)

        model = QtGui.QStandardItemModel()
        for i in refined_list:
            item = QtGui.QStandardItem(i)
            model.appendRow(item)
        self.listCellid.setModel(model)

        model_search = self.lineModelname.text()
        refined_model_list = simple_search(model_search, self.all_models)

        model = QtGui.QStandardItemModel()
        for i in refined_model_list:
            item = QtGui.QStandardItem(i)
            model.appendRow(item)
        #d = pandasModel(data)
        #self.tableModelname.setModel(d)

        self.listModelname.setModel(model)

        # save cell and model search strings
        t = self.comboAnalysis.currentText()
        analysis_info = self.analysis_data[self.analysis_data.name == t].reset_index()
        sql = f"UPDATE Analysis set cell_search='{cell_search}',"+\
            f" model_search='{model_search}'"+\
            f" WHERE id={analysis_info['id'][0]}"
        nd.sql_command(sql)

    def get_selected(self):
        batch = self.comboBatch.currentText()

        _idxs = self.listCellid.selectedIndexes()
        if len(_idxs) == 0:
            # if none selected assume all
            self.listCellid.selectAll()
            _idxs = self.listCellid.selectedIndexes()
        selectedCellid = [self.listCellid.model().item(i.row()).text()
                          for i in _idxs]

        _idxs = self.listModelname.selectedIndexes()
        if len(_idxs) == 0:
            # if none selected assume all
            self.listModelname.selectAll()
            _idxs = self.listModelname.selectedIndexes()
        selectedModelname = [self.listModelname.model().item(i.row()).text()
                         for i in _idxs]

        return batch, selectedCellid, selectedModelname

    def analysis(self, analysis_name):
        batch, selectedCellid, selectedModelname = self.get_selected()

        if analysis_name == 'pareto':
            from nems_lbhb.plots import model_comp_pareto
            model_comp_pareto(selectedModelname, batch=int(batch),
                              goodcells=selectedCellid)


def main():
    app = QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()