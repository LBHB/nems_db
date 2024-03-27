import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nems0 import db
from nems0.xform_helper import load_model_xform, fit_model_xform

from nems.models import LN
from matplotlib.cm import get_cmap
from pathlib import Path

log = logging.getLogger(__name__)

# testing binaural NAT with various model architectures.
batch=338
siteids,cellids0=db.get_batch_sites(batch)

rank=8
ranks = [5, 8, 12, 16]
cell_diff = []
cell_sum = []
r_masks = []
use_layer_1 = False
strf_masked = True
data_save_path = Path('/auto/users/wingertj/data/')
df_label = '.strf_analysis.pkl'
mask_fill = [np.nan, 0]

def gen_binaural_strf_dataframes(ranks, mask_fill = 0, savepath=data_save_path):
    for rank in ranks:
        if use_layer_1:
            modelname=f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_wc.Nx1x{rank}-fir.20x1x{rank}-wc.{rank}xR-dexp.R_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4"
            dos_thresh = 0.6
            #modelname=f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_wc.Nx1x{rank}-fir.20x1x{rank}-wc.{rank}xR-dexp.R_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4.t4"
            #dos_thresh = 0.5
        else:
            modelname=f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_wc.Nx1x{rank}-fir.20x1x{rank}-wc.{rank}xR-dexp.R_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4"
            dos_thresh = 1
            # modelname=f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_wc.Nx1x{rank}-fir.20x1x{rank}-wc.{rank}xR-dexp.R_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4.t4"
            # dos_thresh = 0.7

        # nb: code to fit a model and save results so they can be loaded by load_model_xform:
        # outpath = fit_model_xform(cellid, batch, modelname, saveInDB=True)

        #generate summary plot
        # f_sum, ax_sum = plt.subplots(2, 1, sharex=True)
        # for cellid in cellids0[0:10]:
        low_good_fits = []
        load_problems = []
        site_means = []
        site_stds = []
        hdiffoverhsum = []
        bwr = get_cmap('bwr')
        rank_dataframes = []
        for cellid in cellids0:
            try:
                xf,ctx = load_model_xform(cellid, batch, modelname, eval_model=False)
            except:
                load_problems.append(cellid)
                continue
            r = ctx['modelspec'].meta['r_test'][:,0]
            cellids = ctx['modelspec'].meta['cellids']
            labels = [f"{c[8:]} {rr:.3f}" for c,rr in zip(cellids, r)]
            A_chans = [chan for chan in cellids if '-A-' in chan]
            B_chans = [chan for chan in cellids if '-B-' in chan]

            # f1=LN.LN_plot_strf(ctx['modelspec'],layer=1, plot_nl=True)
            # f2=LN.LN_plot_strf(ctx['modelspec'],labels=labels)
            if use_layer_1:
                wc2 = ctx['modelspec'].layers[2].coefficients
                wc2std = wc2.std(axis=0, keepdims=True)
                wc2std[wc2std == 0] = 1
                wc2 /= wc2std

                s = LN.LN_get_strf(ctx['modelspec'], layer=1)
            else:
                s = LN.LN_get_strf(ctx['modelspec'], layer=2)
            if B_chans:
                hcontra = np.concatenate([s[:18,:,i][:,:, np.newaxis] if '-A-' in cell else s[18:,:,i][:,:, np.newaxis] for i, cell in enumerate(cellids)], axis=2)
                hipsi = np.concatenate([s[18:,:,i][:,:, np.newaxis] if '-A-' in cell else s[:18,:,i][:,:, np.newaxis] for i, cell in enumerate(cellids)], axis=2)
                # check that B is flipped
                b_ind = [i for i, cell in enumerate(cellids) if '-B-' in cell]
                b_flip_contra = sum([np.sum(hcontra[:, :, i] == s[:18, :, i]) for i in b_ind])
                b_flip_ipsi = sum([np.sum(hipsi[:, :, i] == s[18:, :, i]) for i in b_ind])
                if (b_flip_contra == 0) and (b_flip_ipsi == 0):
                    print("Probe B ipsi/contra strfs flipped")
            else:
                hcontra = s[:18,:,:]
                hipsi = s[18:,:,:]

            stddevs = 2
            # mask_fill = np.nan
            #generate masks for pixels +- std dev threshold
            hcontra_masks = np.concatenate([np.abs(hcontra[:, :, i][:, :, np.newaxis]) > stddevs*hcontra[:, :, i].std(axis=(0,1)) for i in
                                       range(len(hcontra[0, 0, :]))], axis=2)
            hipsi_masks = np.concatenate([np.abs(hipsi[:, :, i][:, :, np.newaxis]) > stddevs*hipsi[:, :, i].std(axis=(0, 1)) for i in
                                       range(len(hipsi[0, 0, :]))], axis=2)
            econtra_masks = np.concatenate([hcontra[:, :, i][:, :, np.newaxis] > stddevs*hcontra[:, :, i].std(axis=(0,1)) for i in
                                       range(len(hcontra[0, 0, :]))], axis=2)
            icontra_masks = np.concatenate([hcontra[:, :, i][:, :, np.newaxis] < -stddevs*hcontra[:, :, i].std(axis=(0,1)) for i in
                                       range(len(hcontra[0, 0, :]))], axis=2)
            eipsi_masks = np.concatenate([hipsi[:, :, i][:, :, np.newaxis] > stddevs*hipsi[:, :, i].std(axis=(0,1)) for i in
                                       range(len(hipsi[0, 0, :]))], axis=2)
            iipsi_masks = np.concatenate([hipsi[:, :, i][:, :, np.newaxis] < -stddevs*hipsi[:, :, i].std(axis=(0,1)) for i in
                                       range(len(hcontra[0, 0, :]))], axis=2)

            # fraction of econtra pixels
            contra_e_percent = np.nansum(econtra_masks, axis=(0,1))/np.nansum(hcontra_masks, axis=(0,1))
            ipsi_e_percent = np.nansum(eipsi_masks, axis=(0,1))/np.nansum(hipsi_masks, axis=(0,1))

            # take union of both masks and invert to set non thresholded values to fill value
            union_masks = hcontra_masks|hipsi_masks
            union_masks = np.invert(union_masks)
            econtra_masks_inverted = np.invert(econtra_masks)
            icontra_masks_inverted = np.invert(icontra_masks)
            # set masked values
            masked_contra = hcontra.copy()
            masked_ipsi = hipsi.copy()
            econtra_masked_contra = hcontra.copy()
            econtra_masked_ipsi = hipsi.copy()
            icontra_masked_contra = hcontra.copy()
            icontra_masked_ipsi = hipsi.copy()
            masked_contra[union_masks] = mask_fill
            masked_ipsi[union_masks] = mask_fill
            econtra_masked_contra[econtra_masks_inverted] = mask_fill
            econtra_masked_ipsi[econtra_masks_inverted] = mask_fill
            icontra_masked_contra[icontra_masks_inverted] = mask_fill
            icontra_masked_ipsi[icontra_masks_inverted] = mask_fill
            # if strf_masked:
            #     hcontra = masked_contra
            #     hipsi = masked_ipsi

            # sim = np.sign((hcontra*hipsi).sum(axis=(0,1)))
            # new stat as of 10/27/23
            # contralateral STRF mean over mean(abs(contralateral STRF))
            mcontra_stat = np.mean(hcontra, axis=(0,1))/np.mean(np.abs(hcontra), axis=(0,1))
            ci_diff_stat = np.sum(np.abs(hcontra-hipsi), axis=(0,1))/np.sum(np.abs(hcontra), axis=(0,1))
            ci_diff_stat_2 = np.sum(np.abs(hcontra - hipsi), axis=(0, 1)) / np.sum(np.abs(hcontra + hipsi), axis=(0, 1))
            std_of_diff = np.std(hcontra - hipsi, axis=(0, 1)) / (np.std(hcontra, axis=(0, 1)) + np.std(hipsi, axis=(0, 1)))
            diffsum_ratio2 = (np.std(hcontra, axis=(0, 1))-np.std(hipsi, axis=(0, 1)))/(np.std(hcontra, axis=(0, 1))+np.std(hipsi, axis=(0, 1)))

            hsum = np.nanstd((hcontra+hipsi), axis=(0,1))
            hdiff = np.nanstd((hcontra-hipsi), axis=(0,1))
            diffsum_ratio = hdiff/hsum

            magc = np.nanstd(hcontra, axis=(0,1))
            magi = np.nanstd(hipsi, axis=(0,1))
            ic_ratio = magi/magc

            magc_masked = np.nanstd(masked_contra, axis=(0,1))
            magi_masked = np.nanstd(masked_ipsi, axis=(0,1))
            masked_ic_ratio = magi_masked/magc_masked

            # hsum = np.nanstd((hcontra+hipsi), axis=(0,1)) #/ s.std(axis=(0,1))
            # hdiff = np.nanstd((hcontra-hipsi), axis=(0,1)) #/ s.std(axis=(0,1))
            # diffsum_ratio = hdiff/hsum

            hsum_masked = np.nanstd((masked_contra+masked_ipsi), axis=(0,1)) #/ s.std(axis=(0,1))
            hdiff_masked = np.nanstd((masked_contra-masked_ipsi), axis=(0,1)) #/ s.std(axis=(0,1))
            masked_diffsum_ratio = hdiff_masked/hsum_masked

            # signed statistic for masked values
            contra_masked_signed_ratio = np.nansum(np.sign(masked_contra), axis=(0,1))/np.sum(~union_masks, axis=(0,1))
            ipsi_masked_signed_ratio = np.nansum(np.sign(masked_ipsi), axis=(0, 1))/np.sum(~union_masks, axis=(0,1))
            sum_signed_ratio = np.nansum(np.sign(masked_contra) + np.sign(masked_ipsi), axis=(0,1))/(2*np.sum(~union_masks, axis=(0,1)))
            masked_sum_sign = contra_masked_signed_ratio + ipsi_masked_signed_ratio

            # find percentage of pixels that changed sign between ipsi and contra
            # e_sign_change = np.nansum(np.sign(econtra_masked_contra) + np.sign(econtra_masked_ipsi), axis=(0,1))/2
            e_sign_change = np.sum((econtra_masked_ipsi<0), axis=(0, 1))
            # i_sign_change = np.abs(np.nansum(np.sign(icontra_masked_contra) + np.sign(icontra_masked_ipsi), axis=(0, 1)) / 2)
            i_sign_change = np.sum((icontra_masked_ipsi>0), axis=(0, 1))
            total_sign_change = (e_sign_change+i_sign_change)/np.sum(hcontra_masks, axis=(0,1))
            # f,ax=plt.subplots(1,2)
            # ax[0].scatter(hsum,hdiff)
            # mm = np.max(np.concatenate((hsum,hdiff)))
            # ax[0].plot([0,mm],[0,mm],'--')
            # ax[0].set_xlabel('std(sum)')
            # ax[0].set_ylabel('std(diff)')
            #
            # ax[1].scatter(r,magi/magc)
            # mm = np.max(r)
            # ax[1].plot([0,mm],[0,0],'--')
            # ax[1].plot([0,mm],[1,1],'--')
            # ax[1].set_xlabel('r_test')
            # ax[1].set_ylabel('std(ipsi)/std(contra)')
            #
            # f.suptitle(ctx['modelspec'].name[:35])

            # summary plot idea
            # if use_layer_1:
            #     mask_thresh = 0.95
            #     # for layer=1 hack
            #     r=wc2.std(axis=1)
            #     r_mask=wc2.std(axis=1)>mask_thresh
            # else:
            #     mask_thresh = 0.2
            #     r_mask = r > mask_thresh
            #     if sum(r_mask) < 5:
            #         low_good_fits.append(cellid)
            #         continue

            # cell_sum.append(hsum.copy())
            # cell_diff.append(hdiff.copy())
            # r_masks.append(r.copy())
            #
            # # EE/EI??
            # hdiffoverhsum.append(sum((hdiff[r_mask] / hsum[r_mask]) > dos_thresh)/sum(r_mask))
            # # take mean of strf std dev comparison
            # mean_ic_high = np.mean(magi[r_mask]/magc[r_mask])
            # std_ic_high = np.std(magi[r_mask]/magc[r_mask])
            # site_means.append(mean_ic_high)
            # site_stds.append(std_ic_high)
            # ax_sum[1].scatter(mean_ic_high, std_ic_high, c=bwr(sum((hdiff / hsum) > dos_thresh)/len(hdiff)))
            # ax_sum[1].annotate(sum(r_mask), (mean_ic_high, std_ic_high))

            # make a dataframe
            d = {
                'cell_id': [cellid for i in range(len(cellids))],
                'cell_ids': cellids,
                'rank': [rank for i in range(len(cellids))],
                'layer': ['2' if not use_layer_1 else '1' for i in range(len(cellids))],
                'model_name': [modelname for i in range(len(cellids))],
                'r_test': list(r),
                'c_strf': [hcontra[:,:,i] for i in range(len(cellids))],
                'i_strf': [hipsi[:, :, i] for i in range(len(cellids))],
                'c_strf_std': list(magc),
                'i_strf_std': list(magi),
                'ic_ratio': list(ic_ratio),
                'diff': list(hdiff),
                'sum': list(hsum),
                'diffsum_ratio': list(diffsum_ratio),
                'mask threshold': [stddevs for i in range(len(cellids))],
                'mask': [union_masks[:,:, i] for i in range(len(cellids))],
                'mask_fill': [mask_fill for i in range(len(cellids))],
                'masked c_strf': [masked_contra[:, :, i] for i in range(len(cellids))],
                'masked i_strf': [masked_ipsi[:, :, i] for i in range(len(cellids))],
                'masked c_strf_std': list(magc_masked),
                'masked i_strf_std': list(magi_masked),
                'masked ic_ratio': list(masked_ic_ratio),
                'masked diff': list(hdiff_masked),
                'masked sum': list(hsum_masked),
                'masked diffsum_ratio': list(masked_diffsum_ratio),
                'masked contra_sign_ratio': list(contra_masked_signed_ratio),
                'masked ipsi_sign_ratio': list(ipsi_masked_signed_ratio),
                'masked signed(i)/Ni + signed(c)/Nc': list(masked_sum_sign),
                'masked signed(i+c)/(Ni+Nc)': list(sum_signed_ratio),
                'econtra_masks': [econtra_masks[:, :, i] for i in range(len(cellids))],
                'icontra_masks': [icontra_masks[:, :, i] for i in range(len(cellids))],
                'contra_epercent': list(contra_e_percent),
                'total_sign_change': list(total_sign_change),
                'mean(contra)/mean(abs(contra))': mcontra_stat,
                'sum(abs(contra-ipsi))/sum(abs(contra))': ci_diff_stat,
                'sum(abs(contra-ipsi))/sum(abs(contra+hipsi))': ci_diff_stat_2,
                'std(hcontra-hipsi)/(std(hcontra)+std(hipsi))': std_of_diff,
                '(std(hcontra)-std(hipsi))/(std(hcontra)+std(hipsi))': diffsum_ratio2,

            }
            site_data = pd.DataFrame(d)
            rank_dataframes.append(site_data)
        mask_type = str(mask_fill)
        rank_dataframe = pd.concat(rank_dataframes)
        rank_dataframe.to_pickle(data_save_path/(modelname+f"_{mask_type}_"+df_label))

for fill in mask_fill:
    gen_binaural_strf_dataframes(ranks, mask_fill=fill, savepath=data_save_path)
    # compile rank data_frames into big dataframe
    rank_dfs = []
    for rank in ranks:
        if use_layer_1:
            modelname=f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_wc.Nx1x{rank}-fir.20x1x{rank}-wc.{rank}xR-dexp.R_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4"
            dos_thresh = 0.6
            #modelname=f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_wc.Nx1x{rank}-fir.20x1x{rank}-wc.{rank}xR-dexp.R_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4.t4"
            #dos_thresh = 0.5
        else:
            modelname=f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_wc.Nx1x{rank}-fir.20x1x{rank}-wc.{rank}xR-dexp.R_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4"
            dos_thresh = 1
            # modelname=f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_wc.Nx1x{rank}-fir.20x1x{rank}-wc.{rank}xR-dexp.R_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4.t4"
            # dos_thresh = 0.7
        mask_type = str(fill)
        model_df = pd.read_pickle(data_save_path/(modelname+f'_{mask_type}_'+df_label))
        rank_dfs.append(model_df)

    all_rank_dfs = pd.concat(rank_dfs)
    all_rank_dfs.to_pickle(data_save_path/f'binaural_strf_df_{mask_type}.pkl')


bp = []