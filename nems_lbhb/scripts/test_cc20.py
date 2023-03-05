import nems
import nems0.xform_helper as xhelp

modelname = 'ozgf.fs50.ch18.pop-loadpop.cc20-norm-pca.no-popev_wc.18x4R.g-fir.4x12xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4'
batch = 322
siteid = 'bbl086b'
xf, ctx = xhelp.fit_model_xform(siteid, batch, modelname, returnModel=True)