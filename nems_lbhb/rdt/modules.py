from copy import deepcopy

import numpy as np


def _get_global_gain_sf(rec, gain):
    is_repeating = rec['repeating'].as_continuous()[0].astype('bool')
    t_map = rec['target_id_map'].as_continuous()[0].astype('i')

    gain_mapped = gain[t_map]
    gain0 = np.zeros(t_map.shape, dtype=np.double)

    # Set gains for repeating portions of ss and ds
    gain0[is_repeating] = gain_mapped[is_repeating]

    # Convert to scaling factor
    sf = np.exp(gain0)

    return sf


def global_gain(rec, gain):
    sf = _get_global_gain_sf(rec, gain)
    bg = rec['bg_pred'].as_continuous()
    fg = rec['fg_pred'].as_continuous()
    pred = (bg + fg) * sf
    pred_signal = rec['bg_pred']._modified_copy(pred, name='pred')
    return [pred_signal]


def _get_relative_gain_sf(rec, fg_gain, bg_gain, single_gain):

    is_repeating = rec['repeating'].as_continuous()[0].astype('bool')
    dual_stream = rec['dual_stream'].as_continuous()[0].astype('bool')
    t_map = rec['target_id_map'].as_continuous()[0].astype('i')

    fg_gain_mapped = fg_gain[t_map]
    bg_gain_mapped = bg_gain[t_map]
    single_gain_mapped = single_gain[t_map] + single_gain[0]
    fg_gain = np.zeros(t_map.shape, dtype=np.double)
    bg_gain = np.zeros(t_map.shape, dtype=np.double)

    # Set gains for repeating portions of ss and ds
    fg_gain[is_repeating & dual_stream] = fg_gain_mapped[is_repeating & dual_stream]
    bg_gain[is_repeating & dual_stream] = bg_gain_mapped[is_repeating & dual_stream]
    fg_gain[is_repeating & ~dual_stream] = single_gain_mapped[is_repeating & ~dual_stream]

    # Convert to scaling factor
    bg_sf = np.exp(bg_gain)
    fg_sf = np.exp(fg_gain)

    return fg_sf, bg_sf


def relative_gain(rec, fg_gain, bg_gain, single_gain):
    fg_sf, bg_sf = _get_relative_gain_sf(rec, fg_gain, bg_gain, single_gain)
    bg = rec['bg_pred'].as_continuous()
    fg = rec['fg_pred'].as_continuous()

    pred = bg * bg_sf + fg * fg_sf
    pred_signal = rec['bg_pred']._modified_copy(pred, name='pred')
    return [pred_signal]


def rdt_gain(rec, fg_gain, bg_gain, single_gain):
    fg_sf, bg_sf = _get_relative_gain_sf(rec, fg_gain, bg_gain, single_gain)
    fg_sf = fg_sf[np.newaxis, :]
    bg_sf = bg_sf[np.newaxis, :]
    g_fg = rec['resp']._modified_copy(fg_sf, name='fg_sf')
    g_bg = rec['resp']._modified_copy(bg_sf, name='bg_sf')

    return [g_fg, g_bg]


def apply_gain(rec, i='fg+bg', o='pred', offset=0):
    """
    compress fg and bg streams based on dlog(.. offset) (from nems.modules.nonlinearity)
    then apply stream-specific gain and then sum. offset specifies compression
    :param rec: recording object
    :param i: string identifying of input stream signals (fg+bg or fg_pred+bg_pred)
    :param o: string name of output signal
    :param offset: compression parameter
    :return: list containing a single signal, named 'o'
    """
    # soften effects of more extreme offsets
    inflect = 2
    if isinstance(offset, int):
        offset = np.array([[offset]])

    adjoffset=offset.copy()
    adjoffset[offset > inflect] = inflect + (offset[offset > inflect]-inflect) / 50
    adjoffset[offset < -inflect] = -inflect + (offset[offset < -inflect]+inflect) / 50

    d = 10.0**adjoffset

    fg_sf = rec['fg_sf'].as_continuous()
    bg_sf = rec['bg_sf'].as_continuous()

    if i == 'fg+bg':
        bg = rec['bg'].as_continuous()
        fg = rec['fg'].as_continuous()

        bg = np.log((bg + d)/d)
        fg = np.log((fg + d)/d)
        pred = bg * bg_sf + fg * fg_sf

        bg_comp = rec['bg']._modified_copy(bg, name='bg_comp')
        fg_comp = rec['fg']._modified_copy(fg, name='fg_comp')
        pred_signal = rec['bg']._modified_copy(pred, name='pred')

        return [bg_comp, fg_comp, pred_signal]

    elif i == 'fg_pred+bg_pred':
        bg = rec['bg_pred'].as_continuous()
        fg = rec['fg_pred'].as_continuous()

        pred = bg * bg_sf + fg * fg_sf
        pred_signal = rec['bg_pred']._modified_copy(pred, name='pred')

    return [pred_signal]