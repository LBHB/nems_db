from nems_lbhb.stateplots import model_per_time_wrapper, beta_comp
from nems_lbhb.stateplots import pb_model_plot
import os
"""
batch = 307  # A1 SUA and MUA
batch = 309  # IC SUA and MUA

basemodels = ["-ref-psthfr.s_stategain.S",
              "-ref-psthfr.s_sdexp.S",
              "-ref.a-psthfr.s_sdexp.S"]
state_list = ['st.pup0.far0.hit0.hlf0', 'st.pup0.far0.hit0.hlf',
              'st.pup.far.hit.hlf0', 'st.pup.far.hit.hlf']
state_list = ['st.pup0.fil0', 'st.pup0.fil', 'st.pup.fil0', 'st.pup.fil']
"""
state_list = ['st.pup0.hlf0', 'st.pup0.hlf', 'st.pup.hlf0', 'st.pup.hlf']

cellids = ["TAR010c-06-1", "TAR010c-27-2"]
batch = 307

savefigs=False
save_path = '/Volumes/users/svd/docs/current/pupil_behavior/eps'

for cellid in cellids:
    f = model_per_time_wrapper(cellid, batch=307,
                               loader= "psth.fs20.pup-ld-",
                               fitter = "_jk.nf20-basic",
                               basemodel = "-ref-psthfr_stategain.S",
                               state_list=None, plot_halves=True)
    if savefigs:
        f.savefig(os.path.join(save_path, f'fig2_model_per_time_{cellid}.pdf'))


for cellid in cellids:
    fh, stats = pb_model_plot(cellid=cellid, batch=batch,
                  loader="psth.fs20.pup", basemodel="ref-psthfr_stategain.S",
                  fitter="jk.nf20-basic")
    if savefigs:
        fh.savefig(os.path.join(save_path, f'fig2_pb_psths_{cellid}.pdf'))
