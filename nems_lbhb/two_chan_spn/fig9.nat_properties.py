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
import nems.db as nd
import nems.plots.api as nplt
from nems.utils import find_module, ax_remove_box
from nems.metrics.stp import stp_magnitude
from nems.modules.weight_channels import gaussian_coefficients
from nems.modules.fir import da_coefficients

params = {'legend.fontsize': 6,
          'figure.figsize': (8, 6),
          'axes.labelsize': 8,
          'axes.titlesize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)

dotcolor = 'black'
dotcolor_ns = 'lightgray'
thinlinecolor = 'gray'
barcolors = [(235/255, 47/255, 40/255), (115/255, 200/255, 239/255)]
barwidth = 0.5



def stp_parameter_comp(batch, modelname, modelname0=None):

    d = nems_db.params.fitted_params_per_batch(batch, modelname,
                                               stats_keys=[], multi='first')

    u_bounds = np.array([-0.6, 2.1])
    tau_bounds = np.array([-0.1, 1.5])
    str_bounds = np.array([-0.25, 0.55])
    amp_bounds = np.array([-1, 1.5])

    indices = list(d.index)

    fir_index = None
    do_index = None
    for ind in indices:
        if '--u' in ind:
            u_index = ind
        elif '--tau' in ind:
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
        delay_index = do_index.replace('gains', 'delays')
        f1s_index = do_index.replace('gains', 'f1s')
        taus_index = do_index.replace('gains', 'taus')
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
    print(u)
    if modelname0 is not None:
        d0 = nems_db.params.fitted_params_per_batch(batch, modelname0, stats_keys=[], multi='first')
        r0_test = d0.loc['meta--r_test']
        se0_test = d0.loc['meta--se_test']

    u_mtx = np.zeros((len(u), 2))
    tau_mtx = np.zeros_like(u_mtx)
    m_fir = np.zeros_like(u_mtx)
    r_test_mtx = np.zeros(len(u))
    r0_test_mtx = np.zeros(len(u))
    se_test_mtx = np.zeros(len(u))
    se0_test_mtx = np.zeros(len(u))
    str_mtx = np.zeros_like(u_mtx)

    i = 0
    for cellid in u.index:
        r_test_mtx[i] = r_test[cellid]
        se_test_mtx[i] = se_test[cellid]
        if modelname0 is not None:
            r0_test_mtx[i] = r0_test[cellid]
            se0_test_mtx[i] = se0_test[cellid]

        t_fir = fir[cellid]
        x = np.mean(t_fir, axis=1) / np.std(t_fir)
        mn, = np.where(x == np.min(x))
        mx, = np.where(x == np.max(x))
        xidx = np.array([mx[0], mn[0]])
        m_fir[i, :] = x[xidx]
        u_mtx[i, :] = u[cellid][xidx]
        tau_mtx[i, :] = np.abs(tau[cellid][xidx])
        str_mtx[i, :] = stp_magnitude(tau_mtx[i,:], u_mtx[i,:], fs=100, A=1)[0]
        i += 1

    # EI_units = (m_fir[:,0]>0) & (m_fir[:,1]<0)
    EI_units = (m_fir[:,1] < 0)
    #good_pred = (r_test_mtx > se_test_mtx*2)
    good_pred = ((r_test_mtx > se_test_mtx*3) |
                 (r0_test_mtx > se0_test_mtx*3))

    mod_units = (r_test_mtx-se_test_mtx) > (r0_test_mtx+se0_test_mtx)

    show_units = mod_units & good_pred

    u_mtx[u_mtx < u_bounds[0]] = u_bounds[0]
    u_mtx[u_mtx > u_bounds[1]] = u_bounds[1]
    tau_mtx[tau_mtx > tau_bounds[1]] = tau_bounds[1]
    str_mtx[str_mtx < str_bounds[0]] = str_bounds[0]
    str_mtx[str_mtx > str_bounds[1]] = str_bounds[1]
    m_fir[m_fir < amp_bounds[0]] = amp_bounds[0]
    m_fir[m_fir > amp_bounds[1]] = amp_bounds[1]

    umean = np.median(u_mtx[show_units], axis=0)
    uerr = np.std(u_mtx[show_units], axis=0) / np.sqrt(np.sum(show_units))
    taumean = np.median(tau_mtx[show_units], axis=0)
    tauerr = np.std(tau_mtx[show_units], axis=0) / np.sqrt(str_mtx.shape[0])
    strmean = np.median(str_mtx[show_units], axis=0)
    strerr = np.std(str_mtx[show_units], axis=0) / np.sqrt(str_mtx.shape[0])

    xstr = 'E'
    ystr = 'I'

    fh = plt.figure(figsize=(8, 5))

    dotcolor = 'black'
    dotcolor_ns = 'lightgray'
    thinlinecolor = 'gray'
    barcolors = [(235/255, 47/255, 40/255), (115/255, 200/255, 239/255)]
    barwidth = 0.5

    ax = plt.subplot(2, 3, 1)
    plt.plot(np.array(amp_bounds), np.array(amp_bounds), 'k--')
    plt.plot(m_fir[~show_units, 0], m_fir[~show_units, 1], '.', color=dotcolor_ns)
    plt.plot(m_fir[show_units, 0], m_fir[show_units, 1], '.', color=dotcolor)
    plt.title('n={}/{} good units'.format(
            np.sum(show_units), np.sum(good_pred)))
    plt.xlabel('exc channel gain')
    plt.ylabel('inh channel gain')
    ax_remove_box(ax)

    ax = plt.subplot(2, 3, 2)
    plt.plot(u_bounds, u_bounds, 'k--')
    plt.plot(u_mtx[~show_units, 0], u_mtx[~show_units, 1], '.', color=dotcolor_ns)
    plt.plot(u_mtx[show_units, 0], u_mtx[show_units, 1], '.', color=dotcolor)
    plt.axis('equal')
    plt.xlabel('exc channel u')
    plt.ylabel('inh channel u')
    plt.ylim(u_bounds)
    ax_remove_box(ax)

    ax = plt.subplot(2, 3, 3)
    plt.plot(str_bounds, str_bounds, 'k--')
    plt.plot(str_mtx[~show_units, 0], str_mtx[~show_units, 1], '.', color=dotcolor_ns)
    plt.plot(str_mtx[show_units, 0], str_mtx[show_units, 1], '.', color=dotcolor)
    plt.axis('equal')
    plt.xlabel('exc channel str')
    plt.ylabel('inh channel str')
    plt.ylim(str_bounds)
    ax_remove_box(ax)

    ax = plt.subplot(2, 3, 4)
    plt.plot(np.array([-0.5, 1.5]), np.array([0, 0]), 'k--')
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
    ax_remove_box(ax)

    ax = plt.subplot(2, 3, 5)
    plt.plot(np.array([-0.5, 1.5]), np.array([0, 0]), 'k--')
    plt.bar(np.arange(2), np.sqrt(taumean), color=barcolors, width=barwidth)
    plt.errorbar(np.arange(2), np.sqrt(taumean), yerr=np.sqrt(tauerr),
                 color='black', linewidth=2)
    plt.plot(np.sqrt(tau_mtx[show_units].T), linewidth=0.5, color=thinlinecolor)

    w, p = ss.wilcoxon(tau_mtx[show_units, 0], tau_mtx[show_units, 1])
    plt.ylim((-np.sqrt(np.abs(tau_bounds[0])), np.sqrt(tau_bounds[1])))
    plt.ylabel('sqrt(tau)')
    plt.xlabel('E {:.3f} - I {:.3f} - rat {:.3f} - p={:.1e}'.format(
            taumean[0], taumean[1], taumean[1]/taumean[0], p))
    ax_remove_box(ax)

    ax = plt.subplot(2, 3, 6)
    plt.plot(np.array([-0.5, 1.5]), np.array([0, 0]), 'k--')
    plt.bar(np.arange(2), strmean, color=barcolors, width=barwidth)
    plt.errorbar(np.arange(2), strmean, yerr=strerr, color='black',
                 linewidth=2)
    plt.plot(str_mtx[show_units].T, linewidth=0.5, color=thinlinecolor)

    w, p = ss.wilcoxon(str_mtx[show_units, 0], str_mtx[show_units, 1])
    plt.ylim(str_bounds)
    plt.ylabel('STP str')
    plt.xlabel('E {:.3f} - I {:.3f} - rat {:.3f} - p={:.1e}'.format(
            strmean[0], strmean[1], strmean[1]/strmean[0], p))
    ax_remove_box(ax)

    plt.tight_layout()

    return fh


# start main code
outpath = "/auto/users/svd/docs/current/two_band_spn/eps_rev2/"
save_fig = True
if save_fig:
    plt.close('all')

# figure 9, NAT

# old
batch = 271
modelname0 = "ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-fir.2x15_init-basic"
modelname = "ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-stp.2-fir.2x15_init-basic"

# new
batch=289
modelname0 = "ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-fir.2x15-lvl.1-dexp.1_init-basic"
modelname = "ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-stp.2-fir.2x15-lvl.1-dexp.1_init-basic"
modelname0="ozgf.fs100.ch18-ld-sev_dlog-wc.18x3-fir.3x15-lvl.1-dexp.1_init-basic"
modelname="ozgf.fs100.ch18-ld-sev_dlog-wc.18x3-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"
#modelname0 = "ozgf.fs100.ch18-ld-sev_dlog-wc.18x2-fir.2x15-lvl.1-dexp.1_init-basic"
#modelname = "ozgf.fs100.ch18-ld-sev_dlog-wc.18x2-stp.2-fir.2x15-lvl.1-dexp.1_init-basic"
#modelname0 = "ozgf.fs100.ch18-ld-sev_dlog-wc.18x3.g-fir.3x15-lvl.1-dexp.1_init-basic"
#modelname = "ozgf.fs100.ch18-ld-sev_dlog-wc.18x3.g-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"

# DO models
modelname0="ozgf.fs100.ch18-ld-sev_dlog-wc.18x4.g-do.4x15-lvl.1-dexp.1_init.r10.b-basic"
modelname="ozgf.fs100.ch18-ld-sev_dlog-wc.18x4.g-stp.4.s-do.4x15-lvl.1-dexp.1_init.r10.b-basic"

fileprefix="fig9.NAT"

#fh = stp_parameter_comp(batch, modelname, modelname0=modelname0)

d = nems_db.params.fitted_params_per_batch(batch, modelname,
                                           stats_keys=[], multi='first')


u_bounds = np.array([-0.6, 2.1])
tau_bounds = np.array([-0.1, 1.5])
str_bounds = np.array([-0.25, 0.55])
amp_bounds = np.array([-1, 1.5])

indices = list(d.index)

fir_index = None
do_index = None
for ind in indices:
    if '--u' in ind:
        u_index = ind
    elif '--tau' in ind:
        tau_index = ind
    elif '--fir' in ind:
        fir_index = ind
    elif '--wc' in ind:
        if ind.endswith('coefficients'):
            wc_cc_index=ind
            parm_wc = False
        elif ind.endswith('mean'):
            wc_mean_index=ind
            parm_wc = True
        else:
            wc_sd_index=ind
            parm_wc = True
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

if parm_wc:
    wc_mean = d.loc[wc_mean_index]
    wc_sd = d.loc[wc_sd_index]
else:
    wc_cfs = d.loc[wc_cc_index]

r_test = d.loc['meta--r_test']
se_test = d.loc['meta--se_test']

if modelname0 is not None:
    d0 = nems_db.params.fitted_params_per_batch(batch, modelname0, stats_keys=[], multi='first')
    r0_test = d0.loc['meta--r_test']
    se0_test = d0.loc['meta--se_test']

u_mtx = np.zeros((len(u), 2))
tau_mtx = np.zeros_like(u_mtx)
m_fir = np.zeros_like(u_mtx)
mean_wc = np.zeros_like(u_mtx)
sd_wc = np.zeros_like(u_mtx)

r_test_mtx = np.zeros(len(u))
r0_test_mtx = np.zeros(len(u))
se_test_mtx = np.zeros(len(u))
se0_test_mtx = np.zeros(len(u))
str_mtx = np.zeros_like(u_mtx)
EI_cc = np.zeros(len(u))

i = 0
for cellid in u.index:
    r_test_mtx[i] = r_test[cellid]
    se_test_mtx[i] = se_test[cellid]
    if modelname0 is not None:
        r0_test_mtx[i] = r0_test[cellid]
        se0_test_mtx[i] = se0_test[cellid]

    t_fir = fir[cellid]
    x = np.mean(t_fir, axis=1) / np.std(t_fir, axis=1)
    mn, = np.where(x == np.min(x))
    mx, = np.where(x == np.max(x))
    xidx = np.array([mx[0], mn[0]])
    m_fir[i, :] = x[xidx]
    if parm_wc:
        mean_wc[i, :] = wc_mean[cellid][xidx]
        sd_wc[i, :] = wc_sd[cellid][xidx]
        W = gaussian_coefficients(mean_wc[i,:], sd_wc[i,:], 18)
        EI_cc[i] = np.corrcoef(W[0,:],W[1,:])[0,1]

    else:
        wc_c = wc_cfs[cellid][xidx]
        f = np.linspace(0,1,wc_c.shape[1])
        wc_c[wc_c < 0] = 0

        for j, w in enumerate(wc_c):
            mm = np.mean(w * f) / np.mean(w)
            mean_wc[i, j] = mm
            sd_wc[i, j] = np.mean(w >= np.max(w)/2)

        EI_cc[i] = np.corrcoef(wc_c[0,:],wc_c[1,:])[0,1]

    u_mtx[i, :] = u[cellid][xidx]
    tau_mtx[i, :] = np.abs(tau[cellid][xidx,0])
    str_mtx[i, :] = stp_magnitude(tau_mtx[i,:], u_mtx[i,:], fs=100, A=1.0)[0]

    i += 1

# EI_units = (m_fir[:,0]>0) & (m_fir[:,1]<0)
EI_units = (m_fir[:,1] < 0)
#good_pred = (r_test_mtx > se_test_mtx*2)
good_pred = ((r_test_mtx > se_test_mtx*3) |
             (r0_test_mtx > se0_test_mtx*3))

mod_units = (r_test_mtx-se_test_mtx) > (r0_test_mtx+se0_test_mtx)
show_units = mod_units & good_pred


u_mtx[u_mtx < u_bounds[0]] = u_bounds[0]
u_mtx[u_mtx > u_bounds[1]] = u_bounds[1]
tau_mtx[tau_mtx > tau_bounds[1]] = tau_bounds[1]
str_mtx[str_mtx < str_bounds[0]] = str_bounds[0]
str_mtx[str_mtx > str_bounds[1]] = str_bounds[1]
m_fir[m_fir < amp_bounds[0]] = amp_bounds[0]
m_fir[m_fir > amp_bounds[1]] = amp_bounds[1]

umean = np.median(u_mtx[show_units], axis=0)
uerr = np.std(u_mtx[show_units], axis=0) / np.sqrt(np.sum(show_units))
taumean = np.median(tau_mtx[show_units], axis=0)
tauerr = np.std(tau_mtx[show_units], axis=0) / np.sqrt(str_mtx.shape[0])
strmean = np.median(str_mtx[show_units], axis=0)
strerr = np.std(str_mtx[show_units], axis=0) / np.sqrt(str_mtx.shape[0])

xstr = 'E'
ystr = 'I'


fh=plt.figure()

ax=plt.subplot(2,3,2)
plt.plot(mean_wc[good_pred,0],mean_wc[good_pred,1],'k.')
plt.xlabel('E BF')
plt.ylabel('I BF')
ax.set_aspect('equal','box')
ax_remove_box(ax)
ax.set_title('batch {}'.format(batch))
ax = plt.subplot(2, 3, 3)
plt.plot(np.array([-0.5, 1.5]), np.array([0, 0]), 'k--')
plt.bar(np.arange(2), strmean, color=barcolors, width=barwidth)
plt.errorbar(np.arange(2), strmean, yerr=strerr, color='black',
             linewidth=2)
plt.plot(str_mtx[show_units].T, linewidth=0.5, color=thinlinecolor)

w, p = ss.wilcoxon(str_mtx[show_units, 0], str_mtx[show_units, 1])
plt.ylim(str_bounds)
plt.ylabel('STP str')
plt.xlabel('E {:.3f} - I {:.3f} - rat {:.3f} - p={:.1e}'.format(
        strmean[0], strmean[1], strmean[1]/strmean[0], p))
ax_remove_box(ax)

ax=plt.subplot(2,3,4)
plt.plot(np.array([-0.1,1.5]), np.array([-0.1,1.5]), 'k--', linewidth=0.5)
plt.plot(sd_wc[good_pred,0],sd_wc[good_pred,1],'k.')
plt.xlabel('E tuning width')
plt.ylabel('I tuning width')
ax.set_aspect('equal','box')
ax_remove_box(ax)

ax=plt.subplot(2,3,5)
good_bf = (mean_wc[:,0] >= 0) & (mean_wc[:,0] <= 1) & \
   (mean_wc[:,1] >= 0) & (mean_wc[:,1] <= 1) & good_pred
plt.plot(mean_wc[good_bf,0],str_mtx[good_bf,0],'r.')
plt.plot(mean_wc[good_bf,1],str_mtx[good_bf,1],'b.')
plt.xlabel('BF')
plt.ylabel('STP str')
#ax.set_aspect('equal','box')
ax_remove_box(ax)

ax=plt.subplot(2,3,6)
aa = good_pred & np.logical_not(mod_units)
bb = mod_units
plt.plot(EI_cc[aa],str_mtx[aa,0]-str_mtx[aa,1],'.',
         color=dotcolor_ns)
plt.plot(EI_cc[bb],str_mtx[bb,0]-str_mtx[bb,1],'.',
         color=dotcolor)
plt.plot(np.array([-1, 1]), np.array([0,0]), 'k--', linewidth=0.5)
plt.xlabel('EI_corr')
plt.ylabel('d STP str')
ax_remove_box(ax)

if save_fig:
    fh.savefig(outpath + fileprefix + ".stp_parms_"+modelname+"_batch_"+str(batch)+".pdf")

