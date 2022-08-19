
from nems import recording
from nems_lbhb import xform_wrappers

batch=333
cellid="HOD007a"
modelname = "ozgf.fs100.ch18-ld-norm.l1-sev_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R_prefit.b322-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4"

loadey="ozgf.fs100.ch18"

uri = xform_wrappers.generate_recording_uri(cellid=cellid, batch=batch, loadkey=loadey)

rec = recording.load_recording(uri)

epochs=rec['resp'].epochs

stim_epochs = epochs.loc[epochs.name.str.startswith('STIM'),['name']].drop_duplicates()

# new data frame with split value columns
new = stim_epochs["name"].str.split("_", n = 2, expand = True)
stim_epochs["A"] = new[1]
stim_epochs["B"] = new[2]
stim_epochs["single_stream"] = ((new[1]=='null') | (new[2]=='null'))

dual_epochs = stim_epochs.loc[~stim_epochs.single_stream]
A_epochs = stim_epochs.loc[stim_epochs["B"]=='null']
B_epochs = stim_epochs.loc[stim_epochs["A"]=='null']
dual_epochs = dual_epochs.merge(A_epochs[["name","A"]], how='inner', on="A", suffixes=("","_A"))
dual_epochs = dual_epochs.merge(B_epochs[["name","B"]], how='inner', on="B", suffixes=("","_B"))
dual_epochs = dual_epochs.drop(["A","B","single_stream"], axis=1)