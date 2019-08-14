import re
import pickle
import pandas as pd
import numpy as np
import os.path
import logging

import nems.recording
from nems.recording import Recording, load_recording
from nems.epoch import merge_epoch, group_epochs_by_parent, add_epoch
from nems.preprocessing import average_away_epoch_occurrences
from nems_lbhb.baphy import baphy_data_path

log = logging.getLogger(__name__)

"""
from joblib import Memory

tmpdir = '/auto/data/tmp/nems'
#tmpdir = '/auto/data/tmp'
if not os.path.exists(tmpdir):
    os.makedirs(tmpdir, exist_ok=True)
memory = Memory(cachedir=tmpdir)
"""

def mark_first_reference(epochs, prestim_silence=0.5):
    def _mark_first_reference(row, stim_starts):
        if ('Reference' in row['name']) and (row['start'] in stim_starts):
            row['name'] += ', first'
        return row
    m = epochs['name'] == 'TRIAL'
    stim_starts = epochs.loc[m, 'start'].values + prestim_silence
    return epochs.apply(_mark_first_reference, axis=1, args=(stim_starts,))


def extract_stream_type(df):
    df = df.copy()
    m = df['name'] == 'TRIAL'
    df['trial'] = m.cumsum()
    result = []
    for _, group in df.groupby('trial'):
        m = group['name'].str.startswith('Stim')
        g = group.loc[m]
        has_plus = g['name'].str.contains('\+')
        if np.all(has_plus):
            stream_type = 'dual'
        elif np.all(~has_plus):
            stream_type = 'single'
        else:
            raise ValueError('stream type issue')
        m = group['name'] == 'TRIAL'
        if m.sum() != 1:
            raise ValueError('Cannot find trial epoch')
        start, end = group.loc[m].iloc[0][['start', 'end']]
        series = pd.Series({
            'name': stream_type,
            'start': start,
            'end': end,
        })
        result.append(series)
    return pd.DataFrame(result, columns=['name', 'start', 'end'])


def extract_sample_ids(df, stream):
    result = []
    for _, row in df.iterrows():
        if row['name'].startswith('Stim'):
            sample_id = row['name'].split(',')[1].strip()
            if '+' in sample_id:
                if stream == 'fg':
                    id = sample_id.split('+')[0]
                else:
                    id = sample_id.split('+')[1]
                series = pd.Series({
                    'name': int(id),
                    'start': row['start'],
                    'end': row['end'],
                })
                result.append(series)
            elif stream == 'fg':
                series = pd.Series({
                    'name': int(sample_id),
                    'start': row['start'],
                    'end': row['end'],
                })
                result.append(series)

    result = pd.DataFrame(result, columns=['name', 'start', 'end'])
    result['name'] = result['name'].apply(lambda x: f'{x:02.0f}')
    return result


def extract_repeating(df):
    m = df['name'].apply(lambda x: 'TargetRep' in x)
    epoch = df.loc[m, ['start', 'end']].values
    epoch = merge_epoch(epoch)
    return pd.DataFrame({
        'name': 'repeating',
        'start': epoch[:, 0],
        'end': epoch[:, 1],
    }, columns=['name', 'start', 'end'])


def extract_random(df):
    m = df['name'].apply(lambda x: 'Reference' in x)
    epoch = df.loc[m, ['start', 'end']].values
    epoch = merge_epoch(epoch)
    return pd.DataFrame({
        'name': 'random',
        'start': epoch[:, 0],
        'end': epoch[:, 1],
    }, columns=['name', 'start', 'end'])


def reformat_epochs(epochs, target_id):
    new_epochs = []
    new_epochs.append(epochs)
    fg_samples = extract_sample_ids(epochs, 'fg')
    bg_samples = extract_sample_ids(epochs, 'bg')
    new_epochs.append(fg_samples.copy())
    new_epochs.append(bg_samples.copy())
    fg_samples['name'] = fg_samples['name'].apply(lambda x: f'fg_{x}')
    bg_samples['name'] = bg_samples['name'].apply(lambda x: f'bg_{x}')
    new_epochs.append(fg_samples.copy())
    new_epochs.append(bg_samples.copy())
    samples = pd.concat(new_epochs, ignore_index=True)
    samples.sort_values('start', inplace=True)

    target_id.epochs = samples
    x = target_id.as_continuous()[0]
    m = np.isfinite(x) & (x != 0)
    targets = np.unique(x[m])

    # Tag the epochs as target and whether it's current or not.
    for i, target in enumerate(targets):
        target_name = '{:02d}'.format(int(target))
        m = samples['name'].str.match(target_name)
        e = samples.loc[m].copy()
        e['name'] = 'target_{}'.format(i)
        new_epochs.append(e)

        # extract all occurances of the target across all trials. find out what
        # the actual (intended) target was for each particular trial.
        x = target_id.extract_epoch(target_name)[:, 0, 0]
        m = x == target
        c = e.loc[m].copy()
        c['name'] = 'current_target'
        new_epochs.append(c)

        c = e.loc[~m].copy()
        c['name'] = 'other_target'
        new_epochs.append(c)

    repeating = extract_repeating(epochs)
    new_epochs.append(repeating)

    random = extract_random(epochs)
    new_epochs.append(random)

    stream_type = extract_stream_type(epochs)
    new_epochs.append(stream_type)

    epochs = pd.concat(new_epochs, ignore_index=True)
    epochs.drop_duplicates(inplace=True)
    epochs.sort_values('start', inplace=True)

    # This fixes a bug in BAPHY->NEMS conversion of timestamps (not all trials
    # from the same sequence are identical length and may be off by 0.001).
    # This should be fine for RDT data because all relevant epochs are
    # in units of 0.5, 0.15, 0.3, 0.25 sec.
    epochs['start'] = epochs['start'].round(2)
    epochs['end'] = epochs['end'].round(2)

    epochs = add_epoch(epochs, 'random', 'single')
    epochs = add_epoch(epochs, 'repeating', 'single')
    epochs = add_epoch(epochs, 'random', 'dual')
    epochs = add_epoch(epochs, 'repeating', 'dual')

    for i in range(len(targets)):
        target_name = f'target_{i}'
        epochs = add_epoch(epochs, target_name, 'random_single')
        epochs = add_epoch(epochs, target_name, 'repeating_single')
        epochs = add_epoch(epochs, target_name, 'random_dual')
        epochs = add_epoch(epochs, target_name, 'repeating_dual')

    return epochs


#@memory.cache
def reformat_RDT_recording(rec):
    epochs = rec['stim'].epochs
    state = rec['state']
    target_id = rec['state'].loc['target_id']

    epochs = reformat_epochs(epochs, target_id)

    # Fix NAN values
    x = state._data.copy()
    x[np.isnan(x)] = 0
    x = x.astype('i')
    x.flags['WRITEABLE'] = False
    state._data = x

    # Find out the targets
    x = target_id.as_continuous()
    targets = np.unique(x[np.isfinite(x)])
    target_map = dict((t, i) for i, t in enumerate(targets))
    target_map[0] = 0
    x_mapped = np.vectorize(target_map.get, otypes=[np.float])(x)
    m = ~np.isfinite(x_mapped)
    x_mapped[m] = 0
    x_mapped = x_mapped.astype('i')
    rec['target_id_map'] = target_id._modified_copy(x_mapped)
    rec['target_id_map'].meta = {'target_map': target_map}
    rec.meta = {'n_targets': len(targets)}

    resp = rec['resp']
    rec['resp'] = resp._modified_copy(resp._data[..., :-1])

    rec['dual_stream'] = state.loc['dual_stream']
    #rec['repeating'] = state.loc['repeating']

    for s in rec.signals.values():
        s.epochs = epochs

    rec['repeating'] = rec['stim'].epoch_to_signal('repeating')

    return rec


#@memory.cache
def load_recording(batch=None, cellid=None, reformat=True, by_sequence=True, recording_uri=None, **context):

    options = {
        'cellid': cellid,
        'batch': batch,
        'rasterfs': 100,
        'includeprestim': 1,
        'stimfmt': 'ozgf',
        'chancount': 18,
        'pupil': 0,
        'stim': 1,
        'pertrial': 1,
        'runclass': 'RDT',
        'recache': False,
    }

    if recording_uri is None:
        recording_uri = baphy_data_path(**options)

    log.info('Loading recording from %s', recording_uri)
    rec = nems.recording.load_recording(recording_uri)

    if reformat:
        rec = reformat_RDT_recording(rec)

    if by_sequence:
        rec = average_away_epoch_occurrences(rec, '^SEQUENCE')

    return {'rec': rec, 'input_name': 'fg+bg', 'output_name': 'resp'}


def remove_nan(rec):
    i = np.isnan(rec['fg'].as_continuous())
    new_signals = {}
    for name, signal in rec.items():
        new_data = signal.as_continuous()[:, i]
        new_signals[name] = signal._modified_copy(new_data)
    return Recording(new_signals)


if __name__ == '__main__':
    pass
    #rec = load_recording('269', 'btn144a-a1')['rec']

    #import itertools
    #import pylab as pl

    #epochs = rec['fg'].extract_epoch('03')
    #f, axes = pl.subplots(4, 4)
    #for i, ax in zip(range(16), itertools.chain(*axes)):
    #    ax.imshow(epochs[i])

    #epochs = rec['bg'].extract_epoch('03')
    #f, axes = pl.subplots(4, 4)
    #for i, ax in zip(range(16), itertools.chain(*axes)):
    #    ax.imshow(epochs[i])

    #epochs = rec['resp'].extract_epoch('03')
    #f, axes = pl.subplots(4, 4)
    #for i, ax in zip(range(16), itertools.chain(*axes)):
    #    ax.plot(epochs[i, 0])

    #pl.show()
