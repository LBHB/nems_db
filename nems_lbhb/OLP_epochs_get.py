def get_rec_epochs(parmfile, fs=100):
    '''Takes a parmfile and outputs a dataframe containing every epoch that was played in the recording,
    with columns giving you the concurrent epoch name, BG and FG alone, and then what synthetic
    and binaural condition the epoch'''
    from nems_lbhb.baphy_experiment import BAPHYExperiment
    import nems.epoch as ep

    manager = BAPHYExperiment(parmfile)
    options = {'rasterfs': fs, 'stim': False, 'resp': True}
    rec = manager.get_recording(**options)

    stim_epochs = ep.epoch_names_matching(rec['resp'].epochs, 'STIM_')
    twostims = [epo for epo in stim_epochs if 'null' not in epo]

    epoch_df = pd.DataFrame({'BG + FG': twostims})
    epoch_df['BG'], epoch_df['FG'], epoch_df['Synth Type'], epoch_df['Binaural Type'] = \
        zip(*epoch_df['BG + FG'].apply(get_stim_type))

    return epoch_df


def get_stim_type(ep_name):
    '''Labels epochs that have two stimuli based on what kind of synthetic sound it is and what
    binaural condition it is.
    Synth: N = Normal RMS, C = Cochlear, T = Temporal, S = Spectral, U = Spectrotemporal, M = spectrotemporal
    modulation, A = Non-RMS normalized unsynethic
    Binaural: 11 = BG+FG Contra, 12 = BG Contra - FG Ipsi, 21, BG Ipsi - FG Contra, 22 = BG+FG Ipsi'''
    synth_dict = {'N': 'Unsynthetic', 'C': 'Cochlear', 'T': 'Temporal', 'S': 'Spectral',
                  'U': 'Spectrotemporal', 'M': 'Spectemp Modulation', 'A': 'Non-RMS Unsynthetic'}
    binaural_dict = {'11': 'BG Contra, FG Contra', '12': 'BG Contra, FG Ipsi',
                     '21': 'BG Ipsi, FG Contra', '22': 'BG Ipsi, FG Contra'}

    if len(ep_name.split('_')) == 3 and ep_name[:5] == 'STIM_':
        seps = (ep_name.split('_')[1], ep_name.split('_')[2])
        bg_ep, fg_ep = f"STIM_{seps[0]}_null", f"STIM_null_{seps[1]}"

        #get synth type
        if len(seps[0].split('-')) >= 5:
            synth_kind = seps[0].split('-')[4]
        else:
            synth_kind = 'A'

        #get binaural type
        if len(seps[0].split('-')) >= 4:
            bino_kind = seps[0].split('-')[3] + seps[1].split('-')[3]
        else:
            bino_kind = '11'

        synth_type = synth_dict[synth_kind]
        bino_type = binaural_dict[bino_kind]

    else:
        synth_type, bino_type = None, None
        bg_ep, fg_ep = f"STIM_{ep_name.split('_')[1]}_null", f"STIM_null_{ep_name.split('_')[2]}"

    return bg_ep, fg_ep, synth_type, bino_type