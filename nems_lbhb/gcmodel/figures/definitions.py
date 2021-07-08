# Plot styling
import matplotlib.pyplot as plt

wsu_gray = '#262b2d'
wsu_gray_light = '#586469'
wsu_crimson = '#981e32'
ohsu_navy = '#0e4d8f'

#greys = plt.get_cmap('Greys')
contrast_cmap = plt.get_cmap('Reds')
spectrogram_cmap = plt.get_cmap('Blues')
#model_cmap = plt.get_cmap('viridis')
#model_color_spacing = [0.0, 0.25, 0.45, 0.65, 0.85]
#model_colors = {k: model_cmap(n) for k, n in
#                zip(['combined', 'stp', 'max', 'gc', 'LN'],
#                    model_color_spacing)}
model_cmap = plt.get_cmap('tab20c')
max_cmap = plt.get_cmap('tab10')
model_colors = {
        'LN': model_cmap(0.86),
        #'LN_trans': tuple([i + j for i, j in zip(model_cmap(0.86), (0, 0, 0, -0.25))]),
        #'max': model_cmap(0.21),
        'max': max_cmap(0.31),
        #'max_trans': tuple([i + j for i, j in zip(model_cmap(0.31), (0, 0, 0, -0.25))]),
        'gc': model_cmap(0.41),
        'stp': model_cmap(0),
        'combined': model_cmap(0.61)
        }

imp_color = model_colors['max']
notimp_color = model_colors['LN']
base_LN = model_colors['LN']
base_max = model_colors['max']
base_gc = model_colors['gc']
base_stp = model_colors['stp']
base_combined = model_colors['combined']
dark_LN = tuple([max(0, i + j) for i, j in zip(base_LN, (-0.2, -0.2, -0.2, 0))])
dark_max = tuple([max(0, i +j) for i, j in zip(base_max, (-0.2, -0.2, -0.2, 0))])
dark_gc = tuple([max(0, i +j) for i, j in zip(base_gc, (-0.2, -0.2, -0.2, 0))])
dark_stp = tuple([max(0, i +j) for i, j in zip(base_stp, (-0.2, -0.2, -0.2, 0))])
dark_combined = tuple([max(0, i +j) for i, j in zip(base_combined, (-0.2, -0.2, -0.2, 0))])
faded_LN = tuple([i if j != 3 else 0.5 for j, i in enumerate(base_LN)])
faded_max = tuple([i if j != 3 else 0.5 for j, i in enumerate(base_max)])
faded_gc = tuple([i if j != 3 else 0.5 for j, i in enumerate(base_gc)])
faded_stp = tuple([i if j != 3 else 0.5 for j, i in enumerate(base_stp)])
faded_combined = tuple([i if j != 3 else 0.5 for j, i in enumerate(base_combined)])

small_scatter = 2
big_scatter = 3

standard_fig = [2.25, 2.25]
tall_fig = [2.25, 3.00]
small_fig = [1.75, 1.75]
short_fig = [2.25, 1.75]
text_fig = [6, 6]
wide_fig = [4, 1.9]

params = {  # small version for screens
        #'font.weight': 'bold',
        #'font.size': 24,
        'font.size': 8,
        'font.family': 'Arial',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.figsize': standard_fig,
        'legend.frameon': False,
        }

dash_spacing = (2,3)


###############################################################################
######################     CURRENT MODELS     #################################
###############################################################################

# MOST CURRENT
gc_PF3_o1 = 'ozgf.fs100.ch18-ld-contrast.ms30.cont.n-sev_dlog.f-wc.18x3.g-fir.3x15-lvl.1-ctk.off1.f-dsig.d_gc2.PF'
gc_PF3_o1_ks = 'ozgf.fs100.ch18-ld-contrast.ms30.cont.n-sev_dlog.f-wc.18x3.g-fir.3x15-lvl.1-ctk.off1.f-dsig.d.k.s_gc2.PF'
gc_stp_PF3_o1 = 'ozgf.fs100.ch18-ld-contrast.ms30.cont.n-sev_dlog.f-wc.18x3.g-stp.3-fir.3x15-lvl.1-ctk.off1.f-dsig.d_gc2.PF'
gc_stp_PF3_o1_ks = 'ozgf.fs100.ch18-ld-contrast.ms30.cont.n-sev_dlog.f-wc.18x3.g-stp.3-fir.3x15-lvl.1-ctk.off1.f-dsig.d.k.s_gc2.PF'
stp_dexp3 = 'ozgf.fs100.ch18-ld-sev_dlog.f-wc.18x3.g-stp.3-fir.3x15-lvl.1-dexp.1_init-basic'
ln_dexp3 = 'ozgf.fs100.ch18-ld-sev_dlog.f-wc.18x3.g-fir.3x15-lvl.1-dexp.1_init-basic'
strf3 = 'ozgf.fs100.ch18-ld-contrast.ms30.cont.n-sev_dlog.f-wc.18x3.g-fir.3x15-lvl.1_init-basic'
summed3 = 'ozgf.fs100.ch18-ld-contrast.ms30.cont.n.off0-sev_dlog.f-wc.18x3.g-fir.3x15-lvl.1-ctk3-dsig.d_gc2.PF'
summed_stp3 = 'ozgf.fs100.ch18-ld-contrast.ms30.cont.n.off0-sev_dlog.f-wc.18x3.g-stp.3-fir.3x15-lvl.1-ctk3-dsig.d_gc2.PF'
ln_dexp1 = 'ozgf.fs100.ch18-ld-sev_dlog.f-wc.18x1.g-fir.1x15-lvl.1-dexp.1_init-basic'


# For  batch, gc, stp, LN, combined   positional args
default_args = [289, summed3, stp_dexp3, ln_dexp3, summed_stp3]
kernel_args = [289, gc_PF3_o1, stp_dexp3, ln_dexp3, gc_stp_PF3_o1]


# self-equivalence analysis
gc_h1 = 'ozgf.fs100.ch18-ld-contrast.ms30.cont.n.off0-csum-sev-esth1_dlog.f-wc.18x3.g-fir.3x15-lvl.1-dsig.d_gc4'
gc_h2 = 'ozgf.fs100.ch18-ld-contrast.ms30.cont.n.off0-csum-sev-esth2_dlog.f-wc.18x3.g-fir.3x15-lvl.1-dsig.d_gc4'
stp_h1 = 'ozgf.fs100.ch18-ld-sev-esth1_dlog.f-wc.18x3.g-stp.3-fir.3x15-lvl.1-dexp.1_init-basic'
stp_h2 = 'ozgf.fs100.ch18-ld-sev-esth2_dlog.f-wc.18x3.g-stp.3-fir.3x15-lvl.1-dexp.1_init-basic'
LN_h1 = 'ozgf.fs100.ch18-ld-sev-esth1_dlog.f-wc.18x3.g-fir.3x15-lvl.1-dexp.1_init-basic'
LN_h2 = 'ozgf.fs100.ch18-ld-sev-esth2_dlog.f-wc.18x3.g-fir.3x15-lvl.1-dexp.1_init-basic'
eq_gc = [gc_h1, gc_h2, LN_h1, LN_h2]
eq_stp = [stp_h1, stp_h2, LN_h1, LN_h2]
eq_both = [stp_h1, stp_h2, gc_h1, gc_h2, LN_h1, LN_h2]
eq_kwargs = {'batch': 289, 'stp1': stp_h1, 'stp2': stp_h2, 'gc1': gc_h1,
             'gc2': gc_h2, 'LN1': LN_h1, 'LN2': LN_h2,
             'stp_load':  '/auto/users/jacob/notes/gc_rank3/histogram_arrays/2_18_stp_seq.pkl',
             'gc_load': '/auto/users/jacob/notes/gc_rank3/histogram_arrays/2_18_gc_seq.pkl'}
cross_all = [gc_h1, stp_h2, stp_h1, gc_h2, LN_h1, LN_h2]
cross_kwargs = eq_kwargs.copy()
cross_kwargs['stp_load'] = '/auto/users/jacob/notes/gc_rank3/histogram_arrays/3_13_gc1_stp2.pkl'
cross_kwargs['gc_load'] = '/auto/users/jacob/notes/gc_rank3/histogram_arrays/3_13_gc2_stp1.pkl'

gc_ms30 = 'ozgf.fs100.ch18-ld-contrast.ms30.cont.n-sev_dlog.f-wc.18x3.g-fir.3x15-lvl.1-ctk.off1.f-dsig.d_gc2.PF'
gc_ms70 = 'ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_dlog.f-wc.18x3.g-fir.3x15-lvl.1-ctk.off1.f-dsig.d_gc2.PF'
gc_relsat = 'ozgf.fs100.ch18-ld-sev_dlog.f-wc.18x2.g-fir.2x15-ctk2.18x15.off20-dsig.relsat_gc3'

gc_r20 = 'ozgf.fs100.ch18-ld-contrast.ms30.cont.n-sev_dlog.f-wc.18x3.g-fir.3x15-lvl.1-ctk.off1.f-dsig.d_gc2.PF.rgc20'
gc_r20_ks = 'ozgf.fs100.ch18-ld-contrast.ms30.cont.n-sev_dlog.f-wc.18x3.g-fir.3x15-lvl.1-ctk.off1.f-dsig.d.k.s_gc2.PF.rgc20'


# second implementation (D=2 STRF and gc2 fitter)
gc_av2 = "ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_dlog.f-wc.18x2.g-fir.2x15-lvl.1-ctk-dsig.d_gc2"
# with bounds and normalization:
gc_av2b = "ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_dlog.f-wc.18x2.g-fir.2x15-lvl.1-ctk-dsig.d.bnd.n_gc2.PF"
ln_dexp2 = "ozgf.fs100.ch18-ld-sev_dlog.f-wc.18x2.g-fir.2x15-lvl.1-dexp.1_init-basic"
stp_dexp2 = "ozgf.fs100.ch18-ld-sev_dlog.f-wc.18x2.g-stp.2-fir.2x15-lvl.1-dexp.1_init-basic"

# combined:
gc_av_stp2 = "ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_dlog.f-wc.18x2.g-stp.2-fir.2x15-lvl.1-ctk-dsig.d_gc2"
gc_av_stp2b = "ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_dlog.f-wc.18x2.g-stp.2-fir.2x15-lvl.1-ctk-dsig.d.bnd.n_gc2.PF"
#dexp_kwargs = {'model1': gc_av2b, 'model2': stp_dexp2, 'model3': ln_dexp2,
#               'model4': gc_av_stp2b}


# _ks only for gd ratio comparison (to avoid conflation with changes to amplitude)
gc_PF_o1 = 'ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_dlog.f-wc.18x2.g-fir.2x15-lvl.1-ctk.off1.f-dsig.d_gc2.PF'
gc_PF_o1_ks = 'ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_dlog.f-wc.18x2.g-fir.2x15-lvl.1-ctk.off1.f-dsig.d.k.s_gc2.PF'


# first implementation:
gc_av = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
         "dlog.f-gcwc.18x1.g-gcfir.1x15-gclvl.1-dsig.d_gc.fx")
gc_av_stp = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
             "dlog.f-gcwc.18x1.g-stp.2-gcfir.1x15-gclvl.1-dsig.d_gc.fx")
stp_dexp = ("ozgf.fs100.ch18-ld-sev_"
            "dlog.f-wc.18x1.g-stp.2-fir.1x15-lvl.1-dexp.1_init-basic")
ln_dexp = ("ozgf.fs100.ch18-ld-sev_"
           "dlog.f-wc.18x1.g-fir.1x15-lvl.1-dexp.1_"
           "init-basic")




###############################################################################
#############################      FILE PATHS      ############################
###############################################################################

# AC data
autocorrelation = '/auto/users/jacob/notes/gc_rank3/autocorrelation/batch289_100hz_cutoff1000_run2.pkl'

# CF data
com = '/auto/users/jacob/notes/gc_rank3/characteristic_frequencies/289_com.pkl'
gauss = '/auto/users/jacob/notes/gc_rank3/characteristic_frequencies/289_gauss.pkl'
sm = '/auto/users/jacob/notes/gc_rank3/characteristic_frequencies/289_sm.pkl'
r1 = '/auto/users/jacob/notes/gc_rank3/characteristic_frequencies/289_r1.pkl'
cf_load_paths = {'com': com, 'gaussian': gauss, 'softmax': sm, 'rank1': r1}
cf_save_path = '/auto/users/jacob/notes/gc_rank3/characteristic_frequencies/comparisons/'

# equivalence histogram
eq_289 = '/auto/users/jacob/notes/gc_rank3/histogram_arrays/12_18_summed_b289.pkl'
eq_263 = '/auto/users/jacob/notes/gc_rank3/histogram_arrays/12_19_summed_b263.pkl'

# response stats
#max_289 = '/auto/users/jacob/notes/gc_rank3/response_stats/max/8_15_b289.pkl'
max_289 = '/auto/users/jacob/notes/gc_rank3/response_stats/max/9_27_b289.pkl'
#spont_289 = '/auto/users/jacob/notes/gc_rank3/response_stats/spont/9_27_b289.pkl'
#mean_289 = '/auto/users/jacob/notes/gc_rank3/response_stats/mean/9_27_b289.pkl'
mean_289 = '/auto/users/jacob/notes/gc_rank3/response_stats/mean/3_6_b289.pkl'
spont_289 = '/auto/users/jacob/notes/gc_rank3/response_stats/spont/3_6_b289.pkl'

# sigmoid ratio histogram
sigmoid_hist_o1 = '/auto/users/jacob/notes/gc_rank3/sigmoid_ratio_arrays/8_6_b289_gc_PF3_o1.npy'
# TODO: sigmoid_hist_summed = ...
# Also wanted to try breaking up by mean level

# saved sigmoid plots
sigmoid_shapes = '/auto/users/jacob/notes/gc_rank3/sigmoids/'

# saved strf_vs_resp plots
strf_vs_resp = '/auto/users/jacob/notes/gc_rank3/strf_vs_resp/'

# sound stats
ss_289 = '/auto/users/jacob/notes/gc_rank3/sound_stats/b289.pkl'
ss_263 = '/auto/users/jacob/notes/gc_rank3/sound_stats/b263.pkl'


load_paths = {'AC': autocorrelation, 'CF': cf_load_paths, 'max': max_289,
              'spont': spont_289, 'mean': mean_289,
              'sigmoid_histogram': sigmoid_hist_o1,
              'equivalence_histogram': {
                      'summed': '/auto/users/jacob/notes/gc_rank3/histogram_arrays/8_13_289_summed.npy',
                      'kernel': '/auto/users/jacob/notes/gc_rank3/histogram_arrays/8_13_289_PFo1.npy',
                      },
              'equivalence_effect_size': {
                      'summed': eq_289,
                      'kernel': '/auto/users/jacob/notes/gc_rank3/histogram_arrays/8_15_kernel_b289.pkl',
                      '263': eq_263
                      },
              'self_equivalence': {
                      'stp': '/auto/users/jacob/notes/gc_rank3/histogram_arrays/2_18_stp_seq.pkl',
                      'gc': '/auto/users/jacob/notes/gc_rank3/histogram_arrays/2_18_gc_seq.pkl'
                      },
              'sound_stats': {
                      '289': ss_289,
                      '263': ss_263
                      },
              'simulations': {
                      'stp_cell': {
                              'stp': '/auto/users/jacob/notes/gc_rank3/simulations/stp_AMT005c-20-1.pickle',
                              'gc': '/auto/users/jacob/notes/gc_rank3/simulations/gc_AMT005c-20-1.pickle',
                              },
                      'gc_cell': {
                              'gc': '/auto/users/jacob/notes/gc_rank3/simulations/gc_TAR009d-22-1.pickle',
                              'stp': '/auto/users/jacob/notes/gc_rank3/simulations/stp_TAR009d-22-1.pickle'
                              },
                      'LN_cell': {
                              'LN': '/auto/users/jacob/notes/gc_rank3/simulations/LN_TAR010c-40-1.pickle'
                              }
                      },
              'snrs': '/auto/users/jacob/notes/gc_rank3/snrs/7_17_20_289.pkl'

              }

save_paths = {'CF': cf_save_path, 'sigmoid_shapes': sigmoid_shapes,
              'strf_vs_resp': strf_vs_resp}
