import numpy as np
import os
import io
import logging

import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
from nems.utils import escaped_split, escaped_join
import nems.db as nd
from nems import get_setting
from nems.xform_helper import _xform_exists
from nems.registry import KeywordRegistry
from nems.plugins import (default_keywords, default_loaders,
                          default_initializers, default_fitters)
from nems.gui.recording_browser import browse_recording, browse_context
import nems.gui.editors as gui
import matplotlib.pyplot as plt

import nems_lbhb.rdt.io as rio
from nems_lbhb.baphy import baphy_data_path
from nems.recording import load_recording

log = logging.getLogger(__name__)

batch, cellid, targetid = 269, 'chn022c-a2', '10'
batch, cellid, targetid = 269, 'chn019a-a1', '04'

options = {
    'cellid': cellid,
    'batch': batch,
    'rasterfs': 100,
    'includeprestim': 1,
    'stimfmt': 'ozgf',
    'chancount': 18,
    'pupil': 0,
    'stim': 1,
    'pertrial': 1,
    'runclass': 'RDT',
    'recache': False,
}

recording_uri = baphy_data_path(**options)

rec=load_recording(recording_uri)

resp = rec['resp']
ep = resp.epochs

firsttar = (ep['name'].str.startswith('Stim , '+targetid+'+')  & ep['name'].str.endswith('Target'))
s,e = ep['start'][firsttar].values, ep['end'][firsttar].values
s = (s*resp.fs).astype(int)
e = (e*resp.fs).astype(int)
b = np.array([s,e]).T
dur = int((e-s).mean())

maxreps=4
plt.figure()
for i in range(maxreps):
    plt.subplot(2,maxreps,i+1)
    raster = resp.extract_epoch(b + dur*i)

    plt.imshow(raster[:, 0, :])
    plt.title('sample {} rep {}'.format(targetid,i))

    plt.subplot(2,maxreps,i+maxreps+1)
    plt.plot(np.mean(raster[:,0,:],axis=0))


"""
#keywordstring = 'dlog-wc.18x1.g-fir.1x15-lvl.1'
#keywordstring = 'rdtwc.18x1.g-rdtfir.1x15-rdtgain.global.NTARGETS-lvl.1'
#keywordstring = 'rdtwc.18x1.g-rdtfir.1x15-rdtgain.relative.NTARGETS-lvl.1'
keywordstring =   'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-stp.1.s-fir.1x15-lvl.1-dexp.1'
keywordstring =   'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-fir.1x15-lvl.1'

#modelname = 'rdtld-rdtshf-rdtsev-rdtfmt_' + keywordstring + '_init.t5-basic'
modelname = 'rdtld-rdtshf-rdtfmt_' + keywordstring + '_jk.nf2-init.t5-basic'
modelname = 'rdtld-rdtshf-rdtsev.j5-rdtfmt_' + keywordstring + '_init-basic'
modelname = 'rdtld-rdtshf-rdtsev.j5-rdtfmt_' + keywordstring + '_init.T4'

modelname='rdtld-rdtshf-rdtsev.j.10-rdtfmt_rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x2.g-fir.2x15-lvl.1-dexp.1_init-basic'

autoPlot = True
saveInDB = False
browse_results = False
saveFile = False

log.info('Initializing modelspec(s) for cell/batch %s/%d...', cellid, int(batch))

# Segment modelname for meta information
kws = modelname.split("_")
modelspecname = "-".join(kws[1:-1])
loadkey = kws[0]
fitkey = kws[-1]

meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
        'loader': loadkey, 'fitkey': fitkey, 'modelspecname': modelspecname,
        'username': 'nems', 'labgroup': 'lbhb', 'public': 1,
        'githash': os.environ.get('CODEHASH', ''),
        'recording': loadkey}

#recording_uri = nw.generate_recording_uri(cellid, batch, loadkey)
# code from
# xfspec = xhelp.generate_xforms_spec(recording_uri, modelname, meta)
"""
{'stim': 0, 'chancount': 0, 'pupil': 1, 'rasterfs': 20, 'rawid': None, 'cellid': 'BRT026c-15-1', 'pupil_median': 0, 'pertrial': 0, 'pupil_deblink': 1, 'stimfmt': 'parm', 'runclass': None, 'includeprestim': 1, 'batch': 307}
{'stimfmt': 'parm', 'chancount': 0, 'pupil': 1, 'rasterfs': 20, 'rawid': None, 'cellid': 'BRT026c-15-1', 'pupil_median': 0, 'pertrial': 0, 'pupil_deblink': 1, 'stim': 0, 'runclass': None, 'includeprestim': 1, 'batch': 307}
"""
#log.info('Initializing modelspec(s) for recording/model {0}/{1}...'
#         .format(recording_uri, modelname))
xforms_kwargs = {}
xforms_init_context = {'cellid': cellid, 'batch': int(batch)}
recording_uri = None
kw_kwargs ={}

# equivalent of xform_helper.generate_xforms_spec():

# parse modelname and assemble xfspecs for loader and fitter
load_keywords, model_keywords, fit_keywords = escaped_split(modelname, '_')
if recording_uri is not None:
    xforms_lib = KeywordRegistry(recording_uri=recording_uri, **xforms_kwargs)
else:
    xforms_lib = KeywordRegistry(**xforms_kwargs)

xforms_lib.register_modules([default_loaders, default_fitters,
                             default_initializers])
xforms_lib.register_plugins(get_setting('XFORMS_PLUGINS'))

keyword_lib = KeywordRegistry()
keyword_lib.register_module(default_keywords)
keyword_lib.register_plugins(get_setting('KEYWORD_PLUGINS'))

# Generate the xfspec, which defines the sequence of events
# to run through (like a packaged-up script)
xfspec = []

# 0) set up initial context
if xforms_init_context is None:
    xforms_init_context = {}
if kw_kwargs is not None:
     xforms_init_context['kw_kwargs'] = kw_kwargs
xforms_init_context['keywordstring'] = model_keywords
xforms_init_context['meta'] = meta
xfspec.append(['nems.xforms.init_context', xforms_init_context])

# 1) Load the data
xfspec.extend(xhelp._parse_kw_string(load_keywords, xforms_lib))

# 2) generate a modelspec
xfspec.append(['nems.xforms.init_from_keywords', {'registry': keyword_lib}])
#xfspec.append(['nems.xforms.init_from_keywords', {}])

# 3) fit the data
xfspec.extend(xhelp._parse_kw_string(fit_keywords, xforms_lib))

# 4) add some performance statistics
if not _xform_exists(xfspec, 'nems.xforms.predict'):
    xfspec.append(['nems.xforms.predict', {}])

# 5) add some performance statistics (optional)
if not _xform_exists(xfspec, 'nems.xforms.add_summary_statistics'):
    xfspec.append(['nems.xforms.add_summary_statistics', {}])

# 6) generate plots (optional)
if autoPlot and not _xform_exists(xfspec, 'nems.xforms.plot_summary'):
    xfspec.append(['nems.xforms.plot_summary', {}])

# equivalent of xforms.evaluate():

# Create a log stream set to the debug level; add it as a root log handler
log_stream = io.StringIO()
ch = logging.StreamHandler(log_stream)
ch.setLevel(logging.DEBUG)
fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(fmt)
ch.setFormatter(formatter)
rootlogger = logging.getLogger()
rootlogger.addHandler(ch)

ctx = {}
for xfa in xfspec:
    ctx = xforms.evaluate_step(xfa, ctx)

# Close the log, remove the handler, and add the 'log' string to context
log.info('Done (re-)evaluating xforms.')
ch.close()
rootlogger.removeFilter(ch)

log_xf = log_stream.getvalue()

# save some extra metadata
modelspec = ctx['modelspec']

if saveFile:
    # save results
    if get_setting('USE_NEMS_BAPHY_API'):
        prefix = 'http://'+get_setting('NEMS_BAPHY_API_HOST')+":"+str(get_setting('NEMS_BAPHY_API_PORT')) + '/results/'
    else:
        prefix = get_setting('NEMS_RESULTS_DIR')

    if type(cellid) is list:
        cell_name = cellid[0].split("-")[0]
    else:
        cell_name = cellid

    destination = os.path.join(prefix, str(batch), cell_name, modelspec.get_longname())

    modelspec.meta['modelpath'] = destination
    modelspec.meta['figurefile'] = os.path.join(destination, 'figure.0000.png')
    modelspec.meta.update(meta)

    log.info('Saving modelspec(s) to {0} ...'.format(destination))
    xforms.save_analysis(destination,
                         recording=ctx['rec'],
                         modelspec=modelspec,
                         xfspec=xfspec,
                         figures=ctx['figures'],
                         log=log_xf)

if saveInDB:
    # save performance and some other metadata in database Results table
    modelspec.meta['extra_results']='test'
    nd.update_results_table(modelspec)

if browse_results:
    #from nems.gui.db_browser import model_browser
    from nems.gui.recording_browser import browse_context
    aw = browse_context(ctx, rec='val', signals=['stim', 'pred', 'resp'])
    #aw = browse_context(ctx, signals=['state', 'psth', 'pred', 'resp'])
    ex = gui.browse_xform_fit(ctx, xfspec)
    #ex = EditorWidget(modelspec=ctx['modelspec'], rec=ctx['val'], xfspec=xf,
    #                  ctx=ctx, parent=self)
    #ex.show()
"""