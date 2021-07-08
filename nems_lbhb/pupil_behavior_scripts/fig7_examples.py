from nems_lbhb.stateplots import model_per_time_wrapper, beta_comp
from nems_lbhb.stateplots import pb_model_plot, ppas_model_plot

state_list = ['st.pup0.hlf0', 'st.pup0.hlf', 'st.pup.hlf0', 'st.pup.hlf']
cellids = ["BRT009c-a1","ley046f-44-1"] # ["ley046f-41-1", "ley046g-04-1"]
batch = 309

for cellid in cellids:
    fh, stats = ppas_model_plot(cellid=cellid, batch=batch,
                  loader="psth.fs20.pup", basemodel="psthfr_stategain.S",
                  fitter="jk.nf20-basic")

for cellid in cellids:
    f = model_per_time_wrapper(cellid, batch=batch,
                               loader= "psth.fs20.pup-ld-",
                               fitter = "_jk.nf20-basic",
                               basemodel = "-ref-psthfr_stategain.S",
                               state_list=None, plot_halves=True)
"""
BRT009c: r2taskonly = 0.60, r2puponly=0.62, r2full=0.62
ley046g: r2taskonly = 0.19, r2puponly=0.22, r2full=0.22
"""

