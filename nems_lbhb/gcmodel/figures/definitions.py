# Plot styling
import matplotlib.pyplot as plt

wsu_gray = '#262b2d'
wsu_gray_light = '#586469'
wsu_crimson = '#981e32'
ohsu_navy = '#0e4d8f'

greys = plt.get_cmap('Greys')
contrast_cmap = plt.get_cmap('plasma')
model_cmap = plt.get_cmap('viridis')
model_color_spacing = [0.0, 0.25, 0.45, 0.65, 0.85]
model_colors = {k: model_cmap(n) for k, n in
                zip(['combined', 'stp', 'max', 'gc', 'LN'],
                    model_color_spacing)}
base_LN = model_colors['LN']
base_max = model_colors['max']
dark_LN = tuple([max(0, i + j) for i, j in zip(base_LN, (-0.2, -0.2, -0.2, 0))])
dark_max = tuple([max(0, i +j) for i, j in zip(base_max, (-0.2, -0.2, -0.2, 0))])
faded_LN = tuple([i if j != 3 else 0.5 for j, i in enumerate(base_LN)])
faded_max = tuple([i if j != 3 else 0.5 for j, i in enumerate(base_max)])

small_scatter = 2
big_scatter = 3

standard_fig = [2.25, 2.25]
small_fig = [1.75, 1.75]
short_fig = [2.25, 1.75]
text_fig = [6, 6]

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
eq_289 = '/auto/users/jacob/notes/gc_rank3/histogram_arrays/11_22_summed_b289.pkl'
eq_263 = '/auto/users/jacob/notes/gc_rank3/histogram_arrays/10_6_summed_b263.pkl'

# response stats
max_289 = '/auto/users/jacob/notes/gc_rank3/response_stats/max/8_15_b289.pkl'
spont_289 = '/auto/users/jacob/notes/gc_rank3/response_stats/spont/8_15_b289.pkl'
#max_289 = '/auto/users/jacob/notes/gc_rank3/response_stats/max/9_27_b289.pkl'
#spont_289 = '/auto/users/jacob/notes/gc_rank3/response_stats/spont/9_27_b289.pkl'
mean_289 = '/auto/users/jacob/notes/gc_rank3/response_stats/mean/9_27_b289.pkl'

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
              'sound_stats': {
                      '289': ss_289,
                      '263': ss_263
                      },

              }

save_paths = {'CF': cf_save_path, 'sigmoid_shapes': sigmoid_shapes,
              'strf_vs_resp': strf_vs_resp}
