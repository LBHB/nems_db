import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from PyQt5.QtWidgets import QApplication, QWidget, QTreeWidgetItem

from nems import db as nd
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.celldb import get_single_cell_data, update_single_cell_data
import nems_lbhb.baphy_io as io

from main_GUI import Ui_mainWidget


class OptoIdUi(QWidget):
    def __init__(self, *args, **kwargs):
        super(OptoIdUi, self).__init__(*args, **kwargs)
        self.ui = Ui_mainWidget()
        self.ui.setupUi(self)
        self._createCanvas()
        self._createShortcuts()  # ToDo: can this be done on qt creator?
        self.show()

    def _createCanvas(self):
        # creates MPL figure as QT compatible canvas
        self.fig = Figure(figsize=(15, 30), dpi=100)
        axes = np.empty(3, dtype=object)
        axes[0] = self.fig.add_subplot(211)
        axes[1] = self.fig.add_subplot(212, sharex=axes[0])
        axes[2] = inset_axes(axes[1], width="50%", height="50%", loc=1)

        # common formating independent of Neuron displayed
        axes[0].set_ylabel('Rep')
        axes[1].set_xlabel('time (ms)')
        axes[1].set_ylabel('Spk / sec')
        axes[2].axis('off')
        axes[2].set_xlabel('Time from light onset (sec)')

        self.axes = axes
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ui.canvasLayout.addWidget(self.canvas)

    def _createShortcuts(self):
        self.keyA = self.ui.activatedButton.setShortcut('A')
        self.keyN = self.ui.neutralButton.setShortcut('N')
        self.keyS = self.ui.suppresedButton.setShortcut('S')


class OptoIdModel():
    def __init__(self):

        self.cell_id = list()

        # rec loading parameters
        self.rasterfs = 5000
        self.recache = False
        self.options = {'resp': True, 'rasterfs': self.rasterfs, 'stim': False}

        # plotting parameters
        self.artists = list()
        self.tstart = -0.02
        self.tend = 0.1

        self.list_all_recordings()

    def list_all_recordings(self):
        print('listing all taggable sites...')
        df = nd.pd_query("SELECT * FROM gSingleCell INNER JOIN sCellFile ON gSingleCell.cellid = sCellFile.cellid" +
                         " WHERE sCellFile.RunClassid = 51 AND sCellFile.cellid LIKE %s",
                         params=("TNC%",))

        # clean up DF
        DF = pd.DataFrame()
        DF['cellid'] = df.cellid.iloc[:, 0]
        DF['siteid'] = df.siteid
        DF['recording'] = df.stimfile.apply(lambda x: x.split('.')[0])
        DF['parmfile'] = df.stimpath + df.stimfile  # full path to parameter file.
        DF['rawid'] = df.rawid.iloc[:, 1]

        self.recordings = DF.recording.unique()
        self.DF = DF
        print('done')

    def ready_recording(self, site):
        """
        load a whole recording at a time. Much more efficient than loading individual neurons.
        Uses a site name to define a list of parameters files
        :param site:
        :return:
        """
        print('loading selected site...')

        parmfile, rawid = self.DF.query('recording == @site').loc[:, ('parmfile', 'rawid')].iloc[0, :].tolist()

        self.animal = parmfile.split('/')[4]

        manager = BAPHYExperiment(parmfile=parmfile, rawid=rawid)

        rec = manager.get_recording(recache=False, **self.options)
        rec['resp'] = rec['resp'].rasterize()
        self.prestim = rec['resp'].extract_epoch('PreStimSilence').shape[-1] / self.rasterfs

        # get light on / off
        opt_data = rec['resp'].epoch_to_signal('LIGHTON')
        self.opto_mask = opt_data.extract_epoch('REFERENCE').any(axis=(1, 2))

        opt_start_stop_bins = np.argwhere(
            np.diff(opt_data.extract_epoch('REFERENCE')[self.opto_mask, :, :][0].squeeze())).squeeze() + 1
        self.opt_duration = np.diff(opt_start_stop_bins) / self.rasterfs

        self.cell_id = rec['resp'].chans

        raw_raster = rec['resp'].extract_epoch('REFERENCE').squeeze()
        start_time = self.prestim + self.tstart
        end_time = self.prestim + self.tend
        start_bin = np.floor(start_time * self.options['rasterfs']).astype(int)
        end_bin = np.floor(end_time * self.options['rasterfs']).astype(int)

        self.raster = raw_raster[:, :, start_bin:end_bin]
        self.t = np.linspace(start_time, end_time, self.raster.shape[-1], endpoint=False) - self.prestim

        print('done')

    def sort_by_dprime(self):
        return None

    def get_tag_value(self, cellid):
        tag = get_single_cell_data(cellid).phototag.iloc[0]
        if tag is None:
            tag = 'unclassified'
        elif not isinstance(tag, str):
            raise TypeError(f'unknonw tag of type {type(tag)}')
        return tag

    def set_tag_value(self, cellid, tag):
        print(f'writing neuron {cellid} tag {tag} to database')
        _ = update_single_cell_data(cellid, phototag=tag)
        print('done')

    def plot(self, cellid, axes):
        """
        plots the required data about a given neuron on the given pyqt instanced ax
        """
        print('plotting cell data...')
        self.clear_canvas()
        raster = self.raster[:, self.cell_id.index(cellid), :]
        mean_waveform = io.get_mean_spike_waveform(str(cellid), self.animal, usespkfile=True)

        if mean_waveform.size == 0:
            warnings.warn('waveform is an empty array')
            mean_waveform = np.zeros(2)

        # psth
        on = raster[self.opto_mask, :].mean(axis=0) * self.options['rasterfs']
        on_sem = raster[self.opto_mask, :].std(axis=0) / np.sqrt(self.opto_mask.sum()) * self.options['rasterfs']
        self.artists.extend(axes[1].plot(self.t, on, color='blue'))
        self.artists.append(axes[1].fill_between(self.t, on - on_sem, on + on_sem, alpha=0.3, lw=0, color='blue'))

        off = raster[~self.opto_mask, :].mean(axis=0) * self.options['rasterfs']
        off_sem = raster[~self.opto_mask, :].std(axis=0) / np.sqrt((~self.opto_mask).sum()) * self.options['rasterfs']
        self.artists.extend(axes[1].plot(self.t, off, color='grey'))
        self.artists.append(axes[1].fill_between(self.t, off - off_sem, off + off_sem, alpha=0.3, lw=0, color='grey'))

        # spike raster / light onset/offset
        st = np.where(raster[self.opto_mask, :])
        x = (st[1] / self.rasterfs) + self.tstart
        self.artists.append(axes[0].scatter(x, st[0], s=1, color='blue'))

        offset = self.opto_mask.sum() - 1
        st = np.where(raster[~self.opto_mask, :])
        x = (st[1] / self.rasterfs) + self.tstart
        self.artists.append(axes[0].scatter(x, st[0] + offset, s=1, color='grey'))
        for ss in [0, self.opt_duration]:
            self.artists.append(axes[0].axvline(ss, linestyle='--', color='lime'))
            self.artists.append(axes[1].axvline(ss, linestyle='--', color='lime'))
        axes[0].set_title(cellid)

        # plots waveform in inset
        self.artists.extend(axes[2].plot(mean_waveform, color='red'))
        print('done')

    def clear_canvas(self):
        while len(self.artists) != 0:
            art = self.artists.pop()
            art.remove()


class OptoIdCtrl():
    def __init__(self, model, view):
        self._view = view
        self._model = model
        self.populate_site_drop_down()
        self._connectSignals()

    def populate_site_drop_down(self):
        self._view.ui.siteSelecDropDown.addItems(
            self._model.recordings)  # todo instead of using specific recordigns, full site if possible

    def site_load(self):
        print('loading neurons in site')
        self._model.clear_canvas()
        self._view.canvas.draw()

        site = self._view.ui.siteSelecDropDown.currentText()
        self._model.ready_recording(site)
        self.populate_cell_table()

    def populate_cell_table(self):
        self._view.ui.neuronList.blockSignals(True)
        self._view.ui.neuronList.setSortingEnabled(False)
        self._view.ui.neuronList.clear()

        tree_items = list()
        for cell in self._model.cell_id:
            # formats neuron and tag as strigns for list item creation
            tag = self._model.get_tag_value(cell)
            cell = str(cell)
            # QTreeWidgetItem(None, QStringList(QString("item: %1").arg(i)))
            item = QTreeWidgetItem(None)
            item.setText(0, cell)
            item.setText(1, tag)
            tree_items.append(item)

        self._view.ui.neuronList.insertTopLevelItems(0, tree_items)

        self._view.ui.neuronList.setSortingEnabled(True)
        self._view.ui.neuronList.blockSignals(False)

    def update_cell_table(self, tag):
        item = self._view.ui.neuronList.currentItem()
        item.setText(1, tag)
        self._view.ui.neuronList.setCurrentItem(item)

    def display_neuron(self):
        item = self._view.ui.neuronList.currentItem()
        cellid = item.text(0)
        self._model.plot(cellid, self._view.axes)
        self._view.canvas.draw()

    def classify_activated(self):
        cell = self._view.ui.neuronList.currentItem().text(0)
        tag = 'a'
        self.update_cell_table(tag)
        self._model.set_tag_value(cell, tag)

    def classify_neutral(self):
        cell = self._view.ui.neuronList.currentItem().text(0)
        tag = 'n'
        self.update_cell_table(tag)
        self._model.set_tag_value(cell, tag)

    def classify_suppresed(self):
        cell = self._view.ui.neuronList.currentItem().text(0)
        tag = 's'
        self.update_cell_table(tag)
        self._model.set_tag_value(cell, tag)

    def _connectSignals(self):
        print('connecting signals...')
        # select site
        self._view.ui.siteLoadButton.clicked.connect(self.site_load)
        # select neuron
        self._view.ui.neuronList.itemSelectionChanged.connect(self.display_neuron)
        # classify neuron
        self._view.ui.activatedButton.clicked.connect(self.classify_activated)
        self._view.ui.neutralButton.clicked.connect(self.classify_neutral)
        self._view.ui.suppresedButton.clicked.connect(self.classify_suppresed)
        print('done')


def main():
    """Main function."""
    # Create an instance of QApplication
    optoId = QApplication(sys.argv)
    # Show the calculator's GUI
    view = OptoIdUi()
    # get model functions
    model = OptoIdModel()
    # create instance of the controller
    controller = OptoIdCtrl(model=model, view=view)
    # Execute the calculator's main loop
    sys.exit(optoId.exec_())


if __name__ == '__main__':
    main()
