import os
import sys
import matplotlib.pyplot as plt

import logging
log = logging.getLogger(__name__)
log.disabled = True

#sys.path.append(os.path.abspath('/auto/users/svd/python/scripts/'))
import nems.db as nd
import nems_db.params
import numpy as np
import scipy.stats as ss

import nems_lbhb.stateplots as stateplots
import nems_lbhb.plots as lplt
import nems.recording as recording
import nems.epoch as ep
import nems.xforms as xforms
#import nems_lbhb.xform_wrappers as nw
import nems.plots.api as nplt
from nems.utils import find_module
import nems.metrics.api as nmet
from nems.modules.fir import da_coefficients

params = {'legend.fontsize': 6,
          'figure.figsize': (8, 6),
          'axes.labelsize': 8,
          'axes.titlesize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'pdf.fonttype': 42,
          'ps.fonttype': 42,
          'font.family': 'sans-serif',
          'font.sans-serif': ['Arial', 'Helvetica','Liberation Sans', 'DejaVu Sans',
                              'Bitstream Vera Sans', 'Computer Modern Sans Serif',
                              'Lucida Grande', 'Verdana','Geneva', 'Lucid',
                              'Avant Garde', 'sans-serif']}
plt.rcParams.update(params)

# start main code
outpath = "/auto/users/svd/docs/current/two_band_spn/eps_rev2/"
save_fig = True
#if save_fig:
plt.close('all')

if 1:
    # figure 6, SPN
    batch = 259
    #modelname="env100_dlog_stp2_fir2x15_lvl1_dexp1_basic"

    # shrinkage, normed wc
    modelname0 = "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-basic"
    modelname = "env.fs100-ld-sev_dlog.f-wc.2x3.c-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"

    # no shrinkage, wc
    modelname0 = "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-basic"
    modelname = "env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"

    # no shrinkage, wc normed
    # modelname0 = "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-basic"
    # modelname = "env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"

    modelname0 = "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-basic"
    modelname = "env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"

     # 2 channel DO
    modelname0 = "env.fs100-ld-sev_dlog-wc.2x2.c-do.2x15-lvl.1-dexp.1_init.r10-basic.b"
    modelname = "env.fs100-ld-sev_dlog-wc.2x2.c-stp.2.s-do.2x15-lvl.1-dexp.1_init.r10-basic.b"

    # 3-channel DO
    modelname0 = "env.fs100-ld-sev_dlog-wc.2x3.c-do.3x15-lvl.1-dexp.1_init.r10-basic.b"
    modelname = "env.fs100-ld-sev_dlog-wc.2x3.c-stp.3.s-do.3x15-lvl.1-dexp.1_init.r10-basic.b"

   # 4-channel DO
    modelname0 = "env.fs100-ld-sev_dlog-wc.2x4.c-do.4x15-lvl.1-dexp.1_init.r10-basic.b"
    modelname = "env.fs100-ld-sev_dlog-wc.2x4.c-stp.4.s-do.4x15-lvl.1-dexp.1_init.r10-basic.b"

    fileprefix="fig6.SPN"

elif 1:
    # figure 9, NAT

    # old
    batch = 271
    modelname0 = "ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-fir.2x15_init-basic"
    modelname = "ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-stp.2-fir.2x15_init-basic"

    # new
    #batch=289
    #modelname0 = "ozgf.fs100.ch18-ld-sev_dlog-wc.18x3-fir.3x15-lvl.1-dexp.1_init-basic"
    #modelname = "ozgf.fs100.ch18-ld-sev_dlog-wc.18x3-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"
    fileprefix="fig9.NAT"


#fh, fh0 = stp_parameter_comp(batch, modelname, modelname0=modelname0)

d = nems_db.params.fitted_params_per_batch(batch, modelname,
                                           meta=['r_test', 'r_fit', 'se_test', 'r_ceiling'],
                                           stats_keys=[], multi='first')
if modelname0 is not None:
    d0 = nems_db.params.fitted_params_per_batch(batch, modelname0,
                                                meta=['r_test', 'r_fit', 'se_test', 'r_ceiling'],
                                                stats_keys=[], multi='first')

df_r = nd.batch_comp(batch, [modelname0, modelname], stat='r_test')
df_e = nd.batch_comp(batch, [modelname0, modelname], stat='se_test')

u_bounds = np.array([-0.6, 2.1])
tau_bounds = np.array([-0.1, 1.5])
str_bounds = np.array([-0.25, 0.55])
amp_bounds = np.array([-2, 2])
amp0_bounds = np.array([-2, 2])

indices = list(d.index)

fir_index = None
do_index = None
for ind in indices:
    if '--u' in ind:
        u_index = ind
    elif ('--stp' in ind) and ('--tau' in ind):
        tau_index = ind
    elif '--fir' in ind:
        fir_index = ind
    elif ('--do' in ind) and ('gains' in ind):
        do_index = ind

u = d.loc[u_index]
tau = d.loc[tau_index]
if fir_index:
    fir = d.loc[fir_index]
elif do_index:
    fir = d.loc[do_index]
    delay_index = do_index.replace('gains','delays')
    f1s_index = do_index.replace('gains','f1s')
    taus_index = do_index.replace('gains','taus')
    for cellid in fir.index:
        print(cellid)
        c = da_coefficients(f1s=d.loc[f1s_index,cellid], taus=d.loc[taus_index,cellid],
                            delays=d.loc[delay_index,cellid], gains=d.loc[do_index,cellid],
                            n_coefs=10)
        fir[cellid]=c
else:
    raise ValueError('FIR/DO index not found')

r_test = d.loc['meta--r_test']
se_test = d.loc['meta--se_test']
r_ceiling = d.loc['meta--r_ceiling']

if modelname0 is not None:
    indices0 = list(d0.index)
    fir0_index = None
    for ind in indices0:
        if '--fir' in ind:
            fir0_index = ind
        if ('--do' in ind) and ('gains' in ind):
            do0_index = ind
    if fir0_index:
        fir0 = d0.loc[fir0_index]
    else:
        fir0 = d0.loc[do0_index]
        delay_index = do0_index.replace('gains','delays')
        f1s_index = do0_index.replace('gains','f1s')
        taus_index = do0_index.replace('gains','taus')
        for cellid in fir0.index:
            print(cellid)
            c = da_coefficients(f1s=d0.loc[f1s_index,cellid], taus=d0.loc[taus_index,cellid],
                                delays=d0.loc[delay_index,cellid], gains=d0.loc[do0_index,cellid],
                                n_coefs=10)
            fir0[cellid] = c
    r0_test = d0.loc['meta--r_test']
    se0_test = d0.loc['meta--se_test']
    r_ceiling0 = d0.loc['meta--r_ceiling']

u_mtx = np.zeros((len(u), 2))
tau_mtx = np.zeros_like(u_mtx)
m_fir = np.zeros_like(u_mtx)
m_fir0 = np.zeros_like(u_mtx)

r_test_mtx = np.zeros(len(u))
r0_test_mtx = np.zeros(len(u))
se_test_mtx = np.zeros(len(u))
se0_test_mtx = np.zeros(len(u))

r_ceiling_mtx = np.zeros(len(u))
r0_ceiling_mtx = np.zeros(len(u))
str_mtx = np.zeros_like(u_mtx)

r_test_mtx_p = np.zeros(len(u))
r0_test_mtx_p = np.zeros(len(u))
se_test_mtx_p = np.zeros(len(u))
se0_test_mtx_p = np.zeros(len(u))

i = 0
for cellid in u.index:
    r_test_mtx[i] = df_r.loc[cellid, modelname]
    se_test_mtx[i] = df_e.loc[cellid, modelname]
    r_test_mtx_p[i] = r_test[cellid]
    r_ceiling_mtx[i] = r_ceiling[cellid]
    se_test_mtx_p[i] = se_test[cellid]
    if modelname0 is not None:
        r0_test_mtx[i] = df_r.loc[cellid, modelname0]
        se0_test_mtx[i] = df_e.loc[cellid, modelname0]
        r0_test_mtx_p[i] = r0_test[cellid]
        r0_ceiling_mtx[i] = r_ceiling0[cellid]

    print("{} ln std: {:.3f} stp std: {:.3f}".format(
            cellid, np.std(fir0[cellid]), np.std(fir[cellid])))
    #if fir_index:
    fir[cellid] = fir[cellid] / np.std(fir[cellid])
    t_fir = fir[cellid]
    x = np.mean(t_fir, axis=1) # / np.std(t_fir)
    #else:
    #    x = fir[cellid][:, 0]

    mn, = np.where(x == np.min(x))
    mx, = np.where(x == np.max(x))
    xidx = np.array([mx[0], mn[0]])
    m_fir[i, :] = x[xidx]

    u_mtx[i, :] = u[cellid][xidx]
    tau_mtx[i, :] = np.abs(tau[cellid][xidx])
    str_mtx[i, :] = nmet.stp_magnitude(tau_mtx[i, :], u_mtx[i, :], fs=100, A=1.0)[0]

    #if fir0_index:
    fir0[cellid] = fir0[cellid] / np.std(fir0[cellid])
    t_fir0 = fir0[cellid]
    x = np.mean(t_fir0[:, :-3], axis=1) / np.std(t_fir0[:, :-3])
    if np.max(x) < 0:
        x = -x
    mn, = np.where(x == np.min(x))
    mx, = np.where(x == np.max(x))
    xidx = np.array([mx[0], mn[0]])
    m_fir0[i, :] = x[xidx]

    i += 1

# print(m_fir0)
two_chan = ((np.abs(m_fir0[:, 0])/np.abs(m_fir0[:, 1]) < 10) &
            (np.abs(m_fir0[:, 0])/np.abs(m_fir0[:, 1]) > 0.1))

# EI_units = (m_fir[:,0]>0) & (m_fir[:,1]<0)
EI_units = (m_fir[:,1] < 0)
#good_pred = (r_test_mtx > se_test_mtx*2)
good_pred = ((r_test_mtx > se_test_mtx*3) |
             (r0_test_mtx > se0_test_mtx*3)) # & two_chan

mod_units = (r_test_mtx-se_test_mtx) > (r0_test_mtx+se0_test_mtx)

show_units = mod_units & good_pred

u_mtx[u_mtx < u_bounds[0]] = u_bounds[0]
u_mtx[u_mtx > u_bounds[1]] = u_bounds[1]
tau_mtx[tau_mtx > tau_bounds[1]] = tau_bounds[1]
str_mtx[str_mtx < str_bounds[0]] = str_bounds[0]
str_mtx[str_mtx > str_bounds[1]] = str_bounds[1]
m_fir[m_fir < amp_bounds[0]] = amp_bounds[0]
m_fir[m_fir > amp_bounds[1]] = amp_bounds[1]
m_fir0[m_fir0 < amp0_bounds[0]] = amp0_bounds[0]
m_fir0[m_fir0 > amp0_bounds[1]] = amp0_bounds[1]

umean = np.median(u_mtx[show_units], axis=0)
uerr = np.std(u_mtx[show_units], axis=0) / np.sqrt(np.sum(show_units))
taumean = np.median(tau_mtx[show_units], axis=0)
tauerr = np.std(tau_mtx[show_units], axis=0) / np.sqrt(str_mtx.shape[0])
strmean = np.median(str_mtx[show_units], axis=0)
strerr = np.std(str_mtx[show_units], axis=0) / np.sqrt(str_mtx.shape[0])

xstr = 'E'
ystr = 'I'

dotcolor = 'black'
dotcolor_ns = 'lightgray'
thinlinecolor = 'gray'
barcolors = [(235/255, 47/255, 40/255), (115/255, 200/255, 239/255)]
barwidth = 0.5

fh0 = plt.figure(figsize=(6,8))

ax = fh0.add_subplot(3, 2, 1)
plt.plot([-0.1, amp0_bounds[1]], [0, 0], 'k--', lw=0.5)
plt.plot([0, amp0_bounds[1]], [0, amp0_bounds[1]], 'k--', lw=0.5)
plt.plot([0, 0], amp0_bounds, 'k--', lw=0.5)
plt.plot(m_fir0[~show_units, 0], m_fir0[~show_units, 1], '.', color=dotcolor_ns)
plt.plot(m_fir0[show_units, 0], m_fir0[show_units, 1], '.', color=dotcolor)
plt.title('LN STRF n={}/{} good units'.format(
        np.sum(show_units), np.sum(good_pred)))
plt.xlabel('bigger channel gain')
plt.ylabel('smaller channel gain')
ax.set_aspect('equal', 'box')
nplt.ax_remove_box(ax)

ax = fh0.add_subplot(3, 2, 2)
plt.plot(np.array([0-0.1, amp_bounds[1]]), [0, 0], 'k--', lw=0.5)
plt.plot(np.array([0, amp_bounds[1]]), np.array([0, amp_bounds[1]]), 'k--', lw=0.5)
plt.plot(np.zeros(2), np.array(amp_bounds), 'k--', lw=0.5)
plt.plot(m_fir[~show_units, 0], m_fir[~show_units, 1], '.', color=dotcolor_ns)
plt.plot(m_fir[show_units, 0], m_fir[show_units, 1], '.', color=dotcolor)
plt.title('STP STRF n={}/{} good units'.format(
        np.sum(show_units), np.sum(good_pred)))
plt.xlabel('bigger channel gain')
plt.ylabel('smaller channel gain')
ax.set_aspect('equal', 'box')
nplt.ax_remove_box(ax)

ax = fh0.add_subplot(3, 2, 3)
stateplots.beta_comp(r0_ceiling_mtx[good_pred], r_ceiling_mtx[good_pred],
                     n1='LN STRF', n2='RW3 STP STRF',
                     hist_range=[0.0, 1.0], ax=ax)
ax.set_title('good_pred: {}/{}'.format(np.sum(good_pred),len(r0_ceiling_mtx)))
ax = fh0.add_subplot(3, 2, 4)
stateplots.beta_comp(r0_test_mtx[good_pred], r_test_mtx[good_pred],
                     n1='LN STRF', n2='RW3 STP STRF',
                     hist_range=[0.0, 1.0], ax=ax)

ax = fh0.add_subplot(3,2,5)
F0=np.concatenate(fir0,axis=0)
plt.hist(F0.flatten())

ax = fh0.add_subplot(3,2,6)
F=np.concatenate(fir,axis=0)
plt.hist(F.flatten())


fh = plt.figure(figsize=(8, 5))

ax = plt.subplot(2, 3, 1)
plt.plot(np.array([0, amp_bounds[1]]), np.zeros(2), 'k--', lw=0.5)
plt.plot(np.zeros(2), np.array(amp_bounds), 'k--', lw=0.5)
plt.plot(m_fir[~show_units, 0], m_fir[~show_units, 1], '.', color=dotcolor_ns)
plt.plot(m_fir[show_units, 0], m_fir[show_units, 1], '.', color=dotcolor)
plt.title('n={}/{} good units'.format(
        np.sum(show_units), np.sum(good_pred)))
plt.xlabel('exc channel gain')
plt.ylabel('inh channel gain')
nplt.ax_remove_box(ax)

ax = plt.subplot(2, 3, 2)
plt.plot(u_bounds, u_bounds, 'k--', lw=0.5)
plt.plot(u_mtx[~show_units, 0], u_mtx[~show_units, 1], '.', color=dotcolor_ns)
plt.plot(u_mtx[show_units, 0], u_mtx[show_units, 1], '.', color=dotcolor)
plt.axis('equal')
plt.xlabel('exc channel u')
plt.ylabel('inh channel u')
plt.ylim(u_bounds)
nplt.ax_remove_box(ax)

ax = plt.subplot(2, 3, 3)
plt.plot(str_bounds, str_bounds, 'k--', lw=0.5)
plt.plot(str_mtx[~show_units, 0], str_mtx[~show_units, 1], '.', color=dotcolor_ns)
plt.plot(str_mtx[show_units, 0], str_mtx[show_units, 1], '.', color=dotcolor)
plt.axis('equal')
plt.xlabel('exc channel str')
plt.ylabel('inh channel str')
plt.ylim(str_bounds)
nplt.ax_remove_box(ax)

ax = plt.subplot(2, 3, 4)
plt.plot(np.array([-0.5, 1.5]), np.array([0, 0]), 'k--', lw=0.5)
plt.bar(np.arange(2), umean, color=barcolors, width=barwidth)
plt.errorbar(np.arange(2), umean, yerr=uerr, color='black', linewidth=2)
plt.plot(u_mtx[show_units].T, linewidth=0.5, color=thinlinecolor)
#    plt.plot(np.random.normal(0, 0.05, size=u_mtx[show_units, 0].shape),
#             u_mtx[show_units, 0], '.', color=dotcolor)
#    plt.plot(np.random.normal(1, 0.05, size=u_mtx[show_units, 0].shape),
#             u_mtx[show_units, 1], '.', color=dotcolor)

w, p = ss.wilcoxon(u_mtx[show_units, 0], u_mtx[show_units, 1])
plt.ylim(u_bounds)
plt.ylabel('u')
plt.xlabel('{} {:.3f} - {} {:.3f} - rat {:.3f} - p={:.1e}'.format(
            xstr, umean[0], ystr, umean[1], umean[1]/umean[0], p))
nplt.ax_remove_box(ax)

ax = plt.subplot(2, 3, 5)
plt.plot(np.array([-0.5, 1.5]), np.array([0, 0]), 'k--', lw=0.5)
plt.bar(np.arange(2), np.sqrt(taumean), color=barcolors, width=barwidth)
plt.errorbar(np.arange(2), np.sqrt(taumean), yerr=np.sqrt(tauerr),
             color='black', linewidth=2)
plt.plot(np.sqrt(tau_mtx[show_units].T), linewidth=0.5, color=thinlinecolor)

w, p = ss.wilcoxon(tau_mtx[show_units, 0], tau_mtx[show_units, 1])
plt.ylim((-np.sqrt(np.abs(tau_bounds[0])), np.sqrt(tau_bounds[1])))
plt.ylabel('sqrt(tau)')
plt.xlabel('E {:.3f} - I {:.3f} - rat {:.3f} - p={:.1e}'.format(
        taumean[0], taumean[1], taumean[1]/taumean[0], p))
nplt.ax_remove_box(ax)

ax = plt.subplot(2, 3, 6)
plt.plot(np.array([-0.5, 1.5]), np.array([0, 0]), 'k--', lw=0.5)
plt.bar(np.arange(2), strmean, color=barcolors, width=barwidth)
plt.errorbar(np.arange(2), strmean, yerr=strerr, color='black',
             linewidth=2)
plt.plot(str_mtx[show_units].T, linewidth=0.5, color=thinlinecolor)

w, p = ss.wilcoxon(str_mtx[show_units, 0], str_mtx[show_units, 1])
plt.ylim(str_bounds)
plt.ylabel('STP str')
plt.xlabel('E {:.3f} - I {:.3f} - rat {:.3f} - p={:.1e}'.format(
        strmean[0], strmean[1], strmean[1]/strmean[0], p))
nplt.ax_remove_box(ax)

plt.tight_layout()

if save_fig:
    fh.savefig(outpath + fileprefix + ".stp_parms_"+modelname+".pdf")
    fh0.savefig(outpath + fileprefix + ".ln_v_stp_gain_"+modelname+".pdf")

