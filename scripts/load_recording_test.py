# -*- coding: utf-8 -*-

import nems_db.db as nd
import nems_db.baphy as nb
import nems_db.xform_wrappers as nw

options = {}
options["stimfmt"] = "ozgf"
options["chancount"] = 18
options["rasterfs"] = 100
options['includeprestim'] = 1
#options["average_stim"]=True
#options["state_vars"]=[]

cellid = 'TAR010c-18-1'
batch=271

options={'rasterfs': 10, 'includeprestim': True, 'stimfmt': 'parm',
  'chancount': 0, 'pupil': True, 'stim': False,
  'pupil_deblink': True, 'pupil_median': 1}
options["average_stim"]=False
options["state_vars"]=['pupil']
cellid='BRT033b-12-1'
batch=301

rec=nb.baphy_load_recording(cellid,batch,options)

#rec2=nb.baphy_load_recording_nonrasterized(cellid,batch,options)


# if this complains about a missing nems_db.baphy function, look in baphy_deprecated

