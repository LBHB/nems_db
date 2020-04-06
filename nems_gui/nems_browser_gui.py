#------------------------------------------------------------------------------
# Copyright (c) 2013, Nucleic Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#------------------------------------------------------------------------------
from __future__ import unicode_literals, print_function


import enaml
from enaml.qt.qt_application import QtApplication
from atom.api import Atom, List, Typed, Unicode, Range, Bool, observe

import nems_db.xform_wrappers as nw
import nems.plots.api as nplt
import nems.xforms as xforms
import nems.epoch as ep
from nems.utils import find_module
import pandas as pd
import scipy.ndimage.filters as sf
import nems_lbhb.plots as lplt
import nems_db.db as nd
from nems_gui.recording_browser import (browse_recording)


class CellBatch(Atom):

    id = Unicode()

    def _observe_id(self, event):
        print('new id is', self.id)

    batch = Unicode()


class Cell(Atom):

    cellid = Unicode()


class Model(Atom):

    name = Unicode()


class ModelBrowser(Atom):
    """
    For a given batch, list all cellids and modelnames matching in
    Results. Clicking view will call view_model_recording for the
    currently selected model.

    """
    batch = Unicode("289")
    cell_search_string = Unicode("%")
    model_search_string = Unicode("ozgf%")
    debug = Bool(False)

    cell_list = List(Typed(Cell))
    model_list = List(Typed(Model))
    selected_cell = Unicode("")
    selected_model = Unicode("")
    last_loaded=List()
    recname=Unicode("")
    _cached_windows=List()

    def __init__(self, batch=289, cell_search_string="",
                 model_search_string="ozgf.fs100%", debug=False, *kargs):

        super().__init__(*kargs)

        self.debug = debug
        self.selected_cell = ""
        self.selected_model = ""
        self.last_loaded=['x','x',0]
        self.recname='val'
        self._cached_windows = []

    # can be deleted
    def _default_cell_list(self):
        return [
            Cell(id='TAR001', batch='269'),
            Cell(id='TAR002', batch='269'),
        ]

    def _default_model_list(self):
        return [
            Model(name='model1'),
            Model(name='model2'),
            Model(name='model3'),
        ]

#    # One pattern
#    def update(self, event):
#        # This is always triggered if you update via GUI or via code
#        pass
#
#    # Another pattern
#    def _observe_batch(self, event):
#        # This is always triggered if you update via GUI or via code
#        self.update_widgets(event)

    def print_status(self, event=None):
        print('Cell: ', self.selected_cell)
        print('Model: ', self.selected_model)

    def _observe_batch(self, event):
        # This is always triggered if you update via GUI or via code
        self.update_widgets(event)

    def _observe_cell_search_string(self, event):
        # This is always triggered if you update via GUI or via code
        self.update_widgets(event)

    def _observe_model_search_string(self, event):
        # This is always triggered if you update via GUI or via code
        self.update_widgets(event)

    def update_widgets(self, event):

        batchmask = int(self.batch)
        cellmask = self.cell_search_string
        modelmask = "%" + self.model_search_string + "%"

        d = nd.get_batch_cells(batchmask, cellid=cellmask)
        self.cell_list =  [Cell(cellid=c) for c in list(d['cellid'])]

        d = nd.pd_query("SELECT DISTINCT modelname FROM Results" +
                        " WHERE batch=%s AND modelname like %s" +
                        " ORDER BY modelname",
                        (batchmask, modelmask))
        self.model_list = [Model(name=m) for m in list(d['modelname'])]

        self.selected_cell = ""
        self.selected_model = ""

        print('Updated list widgets', event)

    def get_current_selection(self):
        aw = self

        cellid = aw.selected_cell
        modelname = aw.selected_model
        batch = int(aw.batch)

        print("Viewing {},{},{}".format(batch,cellid,modelname))

        if (aw.last_loaded[0]==cellid and aw.last_loaded[1]==modelname and
            aw.last_loaded[2]==batch):
            xf = aw.last_loaded[3]
            ctx = aw.last_loaded[4]
        else:
            xf, ctx = nw.load_model_baphy_xform(cellid, batch, modelname, eval_model=True)
            aw.last_loaded=[cellid,modelname,batch,xf,ctx]
        return xf, ctx

    def view_recording(self):
        aw = self

        cellid = aw.selected_cell
        modelname = aw.selected_model
        batch = int(aw.batch)
        xf, ctx = aw.get_current_selection()

        recname=aw.recname
        #signals = ['stim','psth','state','resp','pred','mask']
        signals = ['stim','psth','state','resp','pred']
        if type(ctx[recname]) is list:
            rec = ctx[recname][0].apply_mask()
            #rec = ctx[recname][0]
        else:
            rec = ctx[recname].copy()

        aw2 = browse_recording(rec, signals=signals,
                               cellid=cellid, modelname=modelname)

        self._cached_windows.append(aw2)

    def view_model(self):

        xf, ctx = self.get_current_selection()
        nplt.quickplot(ctx)


def main():
    with enaml.imports():
        from nems_lbhb.nems_defs import NemsForm, NEMSWindow

    try:
        app = QtApplication()
    except:
        print('Qt app already running')
        print('THERE IS A BETTER WAY TO DO THIS.')
    browser = ModelBrowser(debug=True)

    view = NEMSWindow(browser=browser)
    view.show()
    try:
        app.start()
    except:
        print('Qt app already running')
        print('THERE IS A BETTER WAY TO DO THIS.')


if __name__ == '__main__':
    main()
