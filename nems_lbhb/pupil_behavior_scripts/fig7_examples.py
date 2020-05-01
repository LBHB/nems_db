from nems_lbhb.stateplots import model_per_time_wrapper, beta_comp
from nems_lbhb.stateplots import pb_model_plot

state_list = ['st.pup0.hlf0', 'st.pup0.hlf', 'st.pup.hlf0', 'st.pup.hlf']
cellids = ["BRT009c-a1", "ley046g-04-1"] # ["ley046f-41-1", "ley046f-44-1"]
batch = 309

for cellid in cellids:
    f = model_per_time_wrapper(cellid, batch=batch,
                               loader= "psth.fs20.pup-ld-",
                               fitter = "_jk.nf20-basic",
                               basemodel = "-ref-psthfr_stategain.S",
                               state_list=None, plot_halves=True)
