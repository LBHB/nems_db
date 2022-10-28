# Hack to get plugins loaded
import os
import matplotlib.pyplot as plt
from nems import xforms
import nems_lbhb.xform_wrappers as nw
from nems0.gui.recording_browser import browse_recording, browse_context

#from nems_lbhb.rdt import plugins
#os.environ['KEYWORD_PLUGINS'] = f'["{plugins.__file__}"]'

#batch, cellid = 269, 'chn019a-a1'
#batch, cellid = 269, 'oys042c-d1'
#batch, cellid = 273, 'chn041d-b1'
#batch, cellid = 273, 'zee027b-c1'
batch, cellid = 269, 'btn144a-a1'
#modelspec = 'RDTwcg18x2-RDTfir2x15_RDTstreamgain_lvl1_dexp1'
#keywordstring = 'dlog-wc.18x1.g-fir.1x15-lvl.1'
#keywordstring = 'rdtwc.18x1.g-rdtfir.1x15-rdtgain.relative.NTARGETS-lvl.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-fir.1x15-lvl.1'

modelname = 'rdtld-rdtshf.rep-rdtsev-rdtfmt_' + keywordstring + '_init-basic'

savefile = nw.fit_model_xforms_baphy(cellid, batch, modelname, saveInDB=False)
#xf,ctx = nw.load_model_baphy_xform(cellid,batch,modelname)
xf,ctx = xforms.load_analysis(savefile)
# browse_context(ctx, 'val', signals=['stim', 'resp', 'fg_sf', 'bg_sf', 'state'])

"""
# database-free version
recording_uri = '/Users/svd/python/nems/recordings/chn019a_e3a6a2e25b582125a7a6ee98d8f8461557ae0cf7.tgz'
#recording_uri = '/Users/svd/python/nems/recordings/chn019a_16e888cad7fef05b2f51c58874bd07040ae80903.tgz'
shuff_streams=False
shuff_rep=False
xfspec = [
    ('nems0.xforms.init_context', {'batch': batch, 'cellid': cellid, 'keywordstring': keywordstring,
                                  'recording_uri': recording_uri}),
    ('nems_lbhb.rdt.io.load_recording', {}),
    ('nems_lbhb.rdt.preprocessing.rdt_shuffle', {'shuff_streams': shuff_streams, 'shuff_rep': shuff_rep}),
    ('nems_lbhb.rdt.preprocessing.split_est_val', {}),
    ('nems_lbhb.rdt.xforms.format_keywordstring', {}),
    ('nems0.xforms.init_from_keywords', {}),
    ('nems0.xforms.fit_basic_init', {}),
    ('nems0.xforms.fit_basic', {}),
    ('nems0.xforms.predict', {}),
    ('nems0.xforms.add_summary_statistics', {}),
    ('nems0.xforms.plot_summary', {}),
]

ctx = {}
for step in xfspec:
    ctx = xforms.evaluate_step(step, ctx)
"""
# browse_context(ctx, 'val', signals=['stim', 'resp', 'fg', 'bg', 'state'])
# browse_context(ctx, 'val', signals=['stim', 'resp', 'fg_sf', 'bg_sf', 'state'])
