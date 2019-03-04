###############################################################################
######################     CURRENT MODELS     #################################
###############################################################################
gc_av = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
         "dlog.f-gcwc.18x1.g-gcfir.1x15-gclvl.1-dsig.d_gc.fx")

gc_av_stp = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
             "dlog.f-gcwc.18x1.g-stp.2-gcfir.1x15-gclvl.1-dsig.d_gc.fx")

stp_dexp = ("ozgf.fs100.ch18-ld-sev_"
            "dlog.f-wc.18x1.g-stp.2-fir.1x15-lvl.1-dexp.1_init-basic")

ln_dexp = ("ozgf.fs100.ch18-ld-sev_"
           "dlog.f-wc.18x1.g-fir.1x15-lvl.1-dexp.1_"
           "init-basic")

# Alternative fits
gc_av_ks = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
            "dlog.f-gcwc.18x1.g-gcfir.1x15-gclvl.1-dsig.d.k.s_gc.fx")

gc_av_no_fx = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
               "dlog.f-gcwc.18x1.g-gcfir.1x15-gclvl.1-dsig.d_gc")

gc_av_t5 = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
            "dlog.f-gcwc.18x1.g-gcfir.1x15-gclvl.1-dsig.d_gc.fx.t5")

gc_av_stp_t5 = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
                "dlog.f-gcwc.18x1.g-stp.2-gcfir.1x15-gclvl.1-dsig.d_gc.fx.t5")

stp_t5 = ("ozgf.fs100.ch18-ld-sev_"
          "dlog.f-wc.18x1.g-stp.2-fir.1x15-lvl.1-dexp.1_"
          "init-basic.t5")

ln_t5 = ("ozgf.fs100.ch18-ld-sev_"
         "dlog.f-wc.18x1.g-fir.1x15-lvl.1-dexp.1_"
         "init-basic.t5")

gc_av_t6 = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
            "dlog.f-gcwc.18x1.g-gcfir.1x15-gclvl.1-dsig.d_gc.fx.t6")

gc_av_stp_t6 = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
                "dlog.f-gcwc.18x1.g-stp.2-gcfir.1x15-gclvl.1-dsig.d_gc.fx.t6")

stp_t6 = ("ozgf.fs100.ch18-ld-sev_"
          "dlog.f-wc.18x1.g-stp.2-fir.1x15-lvl.1-dexp.1_"
          "init-basic.t6")

ln_t6 = ("ozgf.fs100.ch18-ld-sev_"
         "dlog.f-wc.18x1.g-fir.1x15-lvl.1-dexp.1_"
         "init-basic.t6")


###############################################################################
#################       OLD MODELS      #######################################
###############################################################################


gc_cont_full = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
                "dlog.f-wc.18x2.g-fir.2x15-lvl.1-"
                "ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.l_"
                "init.c-basic")

gc_cont_dexp = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
                "dlog.f-wc.18x2.g-fir.2x15-lvl.1-"
                "ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.d_"
                "init.c-basic")

gc_cont_b3 = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n.b3-sev_"
              "dlog.f-wc.18x2.g-fir.2x15-lvl.1-"
              "ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.l_"
              "init.c-basic")

gc_stp = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n.b3-sev_"
          "dlog.f-wc.18x2.g-stp.2-fir.2x15-lvl.1-"
          "ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.l_"
          "init.c-basic")

gc_stp_dexp = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
               "dlog.f-wc.18x2.g-stp.2-fir.2x15-lvl.1-"
               "ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.d_"
               "init.c-basic")

stp_model = ("ozgf.fs100.ch18-ld-sev_"
             "dlog.f-wc.18x2.g-stp.2-fir.2x15-lvl.1-logsig_"
             "init-basic")

ln_model = ("ozgf.fs100.ch18-ld-sev_"
            "dlog.f-wc.18x2.g-fir.2x15-lvl.1-logsig_"
            "init-basic")



dexp_kwargs = {'model1': gc_av, 'model2': stp_dexp, 'model3': ln_dexp,
               'model4': gc_av_stp}

batch = 289

# Example cells
good_cell = 'TAR010c-13-1'
bad_cell = 'bbl086b-02-1'
gc_win1 = 'TAR017b-33-3'
gc_win2 = 'TAR017b-27-2'
gc_win3 = 'TAR010c-40-1'
gc_win4 = 'eno052b-b1'
gc_win5 = 'bbl104h-12-1'
stp_win1 = 'TAR010c-58-2'
stp_win2 = 'BRT033b-12-4'
gc_stp_both_win = 'TAR010c-21-4'
ln_win = 'TAR010c-15-4'
gc_sharp_onset = 'bbl104h-10-2'
gc_beat_stp = 'TAR009d-28-1'
weird = 'TAR009d-22-1'

# Interesting example cells to look at in more detail:

# STP does better, GC about equal to LN
#cellid = 'bbl104h-33-1'

# Same as previous but much bigger difference in performance
#cellid = 'BRT026c-16-2'

# Reverse again: GC better, STP about equal to LN
# GC + STP also looks like it tracks resp better, even though R is slightly lower
#cellid = 'TAR009d-22-1'

# Seems like GC and STP are improving in a similar(?) way, but
# GC + STP does even better.
#cellid = 'TAR010c-13-1'

# Similar example to previous? STRFS very different
#cellid = 'TAR010c-20-1'

# Weird failure that responds to offsets between stims
#cellid = 'TAR010c-58-2'

# Long depression, STP does well but GC and GC+STP do worse
#cellid = 'TAR017b-04-1'

# Bit noisy but similar performance boosts for all 3
#cellid = 'TAR017b-22-1'

# Another case with facilitation where STP doesn't help but
# GC and GC+STP do.
#cellid = 'gus018b-a2'

# Another case where GC and STP each help a little, but GC+STP helps a lot
# Maybe implies a case where the two individual models are capturing
# mostly independent rather than shared info?
#cellid = 'gus019c-a2'

# Both improving in a somehwat similar way?
#cellid = 'TAR009d-15-1'
