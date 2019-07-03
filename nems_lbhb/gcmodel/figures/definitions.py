# Plot styling

wsu_gray = '#262b2d'
wsu_gray_light = '#586469'
wsu_crimson = '#981e32'

ohsu_navy = '#0e4d8f'

#params = {  # big version for posters
#        'pdf.fonttype': 42,
#        'ps.fonttype': 42,
#        'axes.linewidth': 1,
#        'font.weight': 'bold',
#        'font.size': 22,
#        #'font.family': 'Arial'
#        }

params = {  # small version for screens
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 12
        }

figsize = (12,12)
dash_spacing = (10,20)


###############################################################################
######################     CURRENT MODELS     #################################
###############################################################################

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

# MOST CURRENT
gc_PF = "ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_dlog.f-wc.18x2.g-fir.2x15-lvl.1-ctk-dsig.d.n_gc2.PF"
gc_PF_ks = "ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_dlog.f-wc.18x2.g-fir.2x15-lvl.1-ctk-dsig.d.k.s.n_gc2.PF"
gc_stp_PF = "ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_dlog.f-wc.18x2.g-stp.2-fir.2x15-lvl.1-ctk-dsig.d.n_gc2.PF"
# _ks only for gd ratio comparison (to avoid conflation with changes to amplitude)



batch = 289

# Current example cells
gc_better1 = 'TAR010c-59-1'
gc_better2 = 'TAR010c-21-3'  # much better job of bumping up FR
gc_better3 = 'TAR010c-02-1'  # does seem to track the peaks and troughs better, but very noisy
gc_better4 = 'TAR010c-38-2'  # hard to tell on this one
gc_better5 = 'TAR010c-43-2'  # seems like gc is winning for the wrong reasons maybe?
                             # first epoch, just keeps the firing rate high due to high contrast stim
                             # but loses the dynamics.
gc_better6 = 'gus028f-c2'    # maybe? still neeed to check

stp_better2 = 'AMT004b-26-1'  # yes, clear winner for STP
stp_better3 = 'AMT005c-02-2'  # pretty noisy fit, but STP definitely does a better job at keeping FR high
stp_better4 = 'bbl099g-28-1'  # good example where stp helps w/ repeated stim. but GC doesnt
stp_better5 = 'bbl104h-44-1'  # mostly the same except last epoch, then STP wins
stp_better6 = 'bbl104h-02-1'

default_args = [289, gc_PF, stp_dexp2, ln_dexp2, gc_stp_PF]

# Alternative fits
#gc_av_bnd = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
#             "dlog.f-gcwc.18x1.g-gcfir.1x15-gclvl.1.noCT-dsig.d.bnd_gc.fx")
#
#gc_av_bnd_n = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
#               "dlog.f-gcwc.18x1.g-gcfir.1x15-gclvl.1.noCT-dsig.d.bnd.n_gc.fx")
#
#gc_av_bnd_n_ks = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
#                  "dlog.f-gcwc.18x1.g-gcfir.1x15-gclvl.1.noCT-dsig.d.k.s.bnd.n_"
#                  "gc.fx")
#
#gc_av_stp_bnd_n = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
#                   "dlog.f-gcwc.18x1.g-stp.2-gcfir.1x15-gclvl.1.noCT-dsig.d.bnd.n_"
#                   "gc.fx")
#
#gc_av_ks = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
#            "dlog.f-gcwc.18x1.g-gcfir.1x15-gclvl.1-dsig.d.k.s_gc.fx")
#
#gc_av_no_fx = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
#               "dlog.f-gcwc.18x1.g-gcfir.1x15-gclvl.1-dsig.d_gc")
#
#gc_av_t5 = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
#            "dlog.f-gcwc.18x1.g-gcfir.1x15-gclvl.1-dsig.d_gc.fx.t5")
#
#gc_av_stp_t5 = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
#                "dlog.f-gcwc.18x1.g-stp.2-gcfir.1x15-gclvl.1-dsig.d_gc.fx.t5")
#
#stp_t5 = ("ozgf.fs100.ch18-ld-sev_"
#          "dlog.f-wc.18x1.g-stp.2-fir.1x15-lvl.1-dexp.1_"
#          "init-basic.t5")
#
#ln_t5 = ("ozgf.fs100.ch18-ld-sev_"
#         "dlog.f-wc.18x1.g-fir.1x15-lvl.1-dexp.1_"
#         "init-basic.t5")
#
#gc_av_t6 = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
#            "dlog.f-gcwc.18x1.g-gcfir.1x15-gclvl.1-dsig.d_gc.fx.t6")
#
#gc_av_stp_t6 = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
#                "dlog.f-gcwc.18x1.g-stp.2-gcfir.1x15-gclvl.1-dsig.d_gc.fx.t6")
#
#stp_t6 = ("ozgf.fs100.ch18-ld-sev_"
#          "dlog.f-wc.18x1.g-stp.2-fir.1x15-lvl.1-dexp.1_"
#          "init-basic.t6")
#
#ln_t6 = ("ozgf.fs100.ch18-ld-sev_"
#         "dlog.f-wc.18x1.g-fir.1x15-lvl.1-dexp.1_"
#         "init-basic.t6")



# OLD Example cells
#good_cell = 'TAR010c-13-1'
#bad_cell = 'bbl086b-02-1'
#gc_win1 = 'TAR017b-33-3'
#gc_win2 = 'TAR017b-27-2'
#gc_win3 = 'TAR010c-40-1'
#gc_win4 = 'eno052b-b1'
#gc_win5 = 'bbl104h-12-1'
#stp_win1 = 'TAR010c-58-2'
#stp_win2 = 'BRT033b-12-4'
#gc_stp_both_win = 'TAR010c-21-4'
#ln_win = 'TAR010c-15-4'
#gc_sharp_onset = 'bbl104h-10-2'
#gc_beat_stp = 'TAR009d-28-1'

###############################################################################
#################       OLD MODELS      #######################################
###############################################################################


#gc_cont_full = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
#                "dlog.f-wc.18x2.g-fir.2x15-lvl.1-"
#                "ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.l_"
#                "init.c-basic")
#
#gc_cont_dexp = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
#                "dlog.f-wc.18x2.g-fir.2x15-lvl.1-"
#                "ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.d_"
#                "init.c-basic")
#
#gc_cont_b3 = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n.b3-sev_"
#              "dlog.f-wc.18x2.g-fir.2x15-lvl.1-"
#              "ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.l_"
#              "init.c-basic")
#
#gc_stp = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n.b3-sev_"
#          "dlog.f-wc.18x2.g-stp.2-fir.2x15-lvl.1-"
#          "ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.l_"
#          "init.c-basic")
#
#gc_stp_dexp = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
#               "dlog.f-wc.18x2.g-stp.2-fir.2x15-lvl.1-"
#               "ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.d_"
#               "init.c-basic")
#
#stp_model = ("ozgf.fs100.ch18-ld-sev_"
#             "dlog.f-wc.18x2.g-stp.2-fir.2x15-lvl.1-logsig_"
#             "init-basic")
#
#ln_model = ("ozgf.fs100.ch18-ld-sev_"
#            "dlog.f-wc.18x2.g-fir.2x15-lvl.1-logsig_"
#            "init-basic")