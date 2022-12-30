import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import scipy.signal as snl
import scipy.stats as sst
from scipy import interpolate
from sklearn.mixture import GaussianMixture

import nems.db as nd
import nems_lbhb.baphy_io as io



def get_waveform_metrics(mwf, flip_possitive=False):
    """
    powered by Charlie: Calculates spike width in ms (sw), peak-through ratio (ptr), full width half max in ms (fwhm), end slope (es)
    time to base line in ms (bs) and trough in index, all over a smoothed and normalized waveform (wf)
    :param mwf: 1d array of waveform,
    :return: sw, prt, fwhm, bs, trough, wf
    """
    fit2 = interpolate.UnivariateSpline(np.arange(len(mwf)), mwf)
    mwf = fit2(np.linspace(0, len(mwf), 10000))
    mwf /= abs(mwf.min())
    wf = mwf

    if mwf[np.argmax(np.abs(mwf))] > 0 and flip_possitive:
        print('flipping possitive waveform to be negative... experimental')
        mwf = -mwf

    if mwf[np.argmax(np.abs(mwf))] > 0:
        sw = ptr = fwhm = es = bs = trough = np.nan

    else:
        fs = 10000 / (82 / 30000)
        valley = np.argmin(mwf)
        peak = np.argmax(mwf[valley:]) + valley
        trough = valley

        # force 0 to be the mean of the positive waveform preceding the valley
        # mi = np.argmax(mwf[:valley]) # CHR

        # more robust approach:
        # finds the positive inflexion point (gradient == 0) prior and closest to the valley, and defines all prior as baseline
        grad = np.gradient(mwf[:valley])
        zero_grad_idx = np.where(np.diff(np.sign(grad)))[0]
        pos_inflex_idx = zero_grad_idx[grad[zero_grad_idx] > 0]
        mi = pos_inflex_idx[-1]  # MLE

        baseline = np.mean(mwf[:mi])
        mwf -= baseline

        sw = (peak - valley) / fs * 1000  # ms

        # get fwhm (of valley)
        left = np.argmin(np.abs(mwf[:valley] - (mwf[valley] / 2)))
        right = np.argmin(np.abs(mwf[valley:] - (mwf[valley] / 2))) + valley
        fwhm = (right - left) / fs * 1000

        if mwf[peak] <= 0:
            ptr = 0
        else:
            ptr = abs(mwf[peak]) / abs(mwf[valley])

        # 0.5 ms ~ 1800 bins
        es = (mwf[valley + 550] - mwf[valley + 350]) # empirical values CHR.
        # save time waveform returns to baseline
        bs = np.argmin(np.abs(mwf[peak:])) / fs * 1000

    return sw, ptr, fwhm, es, bs, trough, wf


def get_optotag_DF(cellids):
    """
    given a list of cellids, returns a dataframe containing the optotags found in the celldb. To set optotags see
    nems_lbhb/optotagging/opto_id.py
    args:
        cellids: a list of cellid strings
    Returns:
        d: A pandas dataframe containing cell tags
    """

    """
    simple tool to list units and their corresponding phototagging,
    More things to be added later on like filtering by runclass
    """
    # sql = "SELECT siteid, cellid, phototag FROM gSingleCell WHERE not(isnull(phototag))"
    sql = f"SELECT cellid, phototag FROM gSingleCell WHERE " \
          f"not(isnull(phototag)) AND " \
          f"cellid IN {tuple(cellids)}"

    d = nd.pd_query(sql)

    # formats to reduce memory usage.
    for col in d.columns:
        d[col] = d[col].astype('category')

    return d


def get_waveform_DF(cellids):
    """
    given a list of cellids, returns a dataframe containing the waveforms(if found) and metrics describing them
    args:
        cellids: a list of cellid strings
    Returns:
        celltype_DF: A pandas dataframe containing cell waveform infor
    """

    if cellids is str:
        cellids = [cellids]

    failed_cells = list()
    celltype_DF = list()

    for cellid in cellids:

        try:
            mean_waveform = io.get_mean_spike_waveform(cellid, usespkfile=None)
        except:
            print(f'cant get {cellid} waveform')
            failed_cells.append(cellid)
            continue

        isolation = nd.get_cell_files(cellid).loc[:, 'isolation'].unique()
        isolation = isolation[0]
        if mean_waveform.size == 0:
            print(f'{cellid} waveform is empty')
            failed_cells.append(cellid)
            continue

        sw, ptr, fwhm, es, bs, trough, wf = get_waveform_metrics(mean_waveform)

        df = pd.DataFrame()
        df['cellid'] = (cellid,)
        df['sw'] = (sw,)
        df['ptr'] = (ptr,)
        df['fwhm'] = (fwhm,)
        df['es'] = (es,)
        df['bs'] = (bs,)
        df['trough'] = (trough,)
        df['waveform_norm'] = (wf.tolist(),)
        df['isolation'] = (isolation,)

        celltype_DF.append(df)

    celltype_DF = pd.concat(celltype_DF, ignore_index=True)

    return celltype_DF


def classify_by_spike_width(DF, margin=0.05):
    """
    spike width histograms follow a bimodal distribution. Find the valley with an error margin

    """

    if np.any(DF.sw.isnull()):
        print(f'found {np.sum(DF.sw.isnull())} cellid with no spike width in DF, ignoring...')

    # define kernel density estimate, the bandwidth is defined empirically
    kernel = sst.gaussian_kde(DF.loc[~DF.sw.isnull(), 'sw'], 0.1)
    x = np.linspace(0, 1.5, 100)
    hist = kernel(x)

    # find valley in bimodal distribution
    min_idx = snl.argrelmin(hist)[0]
    hist_threshold = x[min_idx[0]]
    # margin = 0.05  # plus or minus in ms
    print(f'waveform threshold. lower: {hist_threshold - margin:.2f}, upper {hist_threshold + margin:.2f}')

    # Classifies base on valley plus an unclasified zone of 0.1ms
    named_labels = np.empty(len(DF['sw']), dtype=object)
    named_labels[DF['sw'] < (hist_threshold - margin)] = 'narrow'
    named_labels[np.logical_and((hist_threshold - margin) <= DF['sw'],
                                (DF['sw'] < (hist_threshold + margin)))] = 'unclass'
    named_labels[(hist_threshold + margin) <= DF['sw']] = 'broad'

    DF['sw_kde'] = named_labels

    return DF, hist_threshold, margin


def cluster_by_metrics(DF):
    """
    takes dataframe of waveform metrics, and returns a copy of the DF with a cluster label column,
    this is an old approach by charlie, and has been superseeded by the simpler and more robust
    'classify_by_spikewidth'
    :param DF:
    :return:
    """

    toclust = DF.dropna(axis=0).copy()
    csw = toclust['sw'].values  # csw -= csw.mean(); csw /= csw.std(); csw*=10
    cptr = toclust['ptr'].values  # cptr -= cptr.mean(); cptr /= cptr.std()
    ces = toclust['es'].values
    # wf = toclust['waveform_norm'].values
    # trough = toclust['trough'].values

    X = np.stack((csw, cptr, ces), axis=1)
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    gmm = GaussianMixture(n_components=2).fit(X)
    labels = gmm.predict(X)

    named_labels = np.empty(len(labels), dtype=object)
    if csw[labels == 1].mean() < csw[labels == 0].mean():
        named_labels[labels == 1] = 'narrow'
        named_labels[labels == 0] = 'broad'
    else:
        named_labels[labels == 0] = 'narrow'
        named_labels[labels == 1] = 'broad'

    toclust['spike_type'] = named_labels

    merge = pd.merge(DF, toclust.loc[:, ('cellid', 'spike_type')], how='left', on='cellid', validate='1:1')

    return merge


# wrapper for classification
def get_celltype_DF(cellids):

    WF_DF = get_waveform_DF(cellids)
    OT_DF = get_optotag_DF(cellids)

    # todo add layer and or depth
    # add phototags when possible
    celltype_DF = pd.merge(WF_DF, OT_DF, on='cellid', how='left')

    return celltype_DF


def add_triple_classification_column(DF):
    # triple classification with activated, broad or narrow
    triple = np.empty(DF.shape[0], dtype=object)
    activated = DF.phototag == 'a'
    narrow = np.logical_and(~activated, DF.sw_kde == 'narrow')
    broad = np.logical_and(~activated, DF.sw_kde == 'broad')
    unclass = np.logical_and(~activated, DF.sw_kde == 'unclass')

    triple[activated] = 'activated'
    triple[narrow] = 'narrow'
    triple[broad] = 'broad'
    triple[unclass] = 'unclass'
    DF['triple'] = pd.Series(triple, dtype='category')
    return DF


### Usefull plotting functions, relate to the classification above
def plotly_celltype_histogram(DF, hist_threshold=None, margin=None):

    DF = DF.dropna(subset=['sw'])
    # plotting
    HardBlue = '#0000FF'
    fig = px.histogram(DF.dropna(subset=['triple']), nbins=100,
                       x='sw', color='triple',
                       pattern_shape='triple',
                       pattern_shape_map={'activated': '', 'narrow': "", "unclass": "x", "broad": ""},
                       barmode='stack',
                       color_discrete_map={'activated': HardBlue, 'narrow': 'darkgray', "unclass": 'lightgray',
                                           "broad": 'black'},
                       category_orders={'triple': ['activated', 'narrow', 'unclass', 'broad']},
                       )
    fig.update_traces(marker_line_width=0)
    # this scaling is a hardcoded hack defined by vidually comparing with the figure above where the historgrams are also a probability density
    kernel = sst.gaussian_kde(DF['sw'], 0.1)
    x = np.linspace(0, 1.5, 100)
    y = kernel(x) * 32
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='dimgray', dash='dot'), showlegend=False))

    if hist_threshold is not None:
        fig.add_vline(x=hist_threshold, line=dict(color='black', dash='dash', width=1), opacity=0.5)

    if margin is not None:
        fig.add_vline(x=hist_threshold - margin, line=dict(color='black', dash='dash', width=1), opacity=1)
        fig.add_vline(x=hist_threshold + margin, line=dict(color='black', dash='dash', width=1), opacity=1)

    w, h = 3, 2
    fig.update_layout(template='simple_white',
                      width=96 * w, height=96 * h,
                      margin=dict(t=10, b=10, l=10, r=10),
                      bargap=0,
                      xaxis=dict(title=dict(standoff=0,
                                            text='Peak-to-trough delay (ms)',
                                            font_size=10),
                                 tickfont_size=9),
                      yaxis=dict(title=dict(text='neuron count',
                                            standoff=0,
                                            font_size=10),
                                 tickfont_size=9),
                      legend=dict(xanchor='right', x=1,
                                  yanchor='top', y=1,
                                  font_size=9,
                                  title=dict(text='')),
                      )

    return fig


def get_aligned_waveforms(DF):

    """
    aligns disssimilar waveforms by their peak, useful for plotting all waveforms togethers
    """
    wf = DF['waveform_norm'].values
    trough = DF['trough'].values

    aligned = []
    fs = 10000 / (82 / 30000)
    for w, wave in enumerate(wf):
        wave = np.asarray(wave)
        t = int(trough[w])
        wave = wave[t - int(fs * 0.0005):t + int(fs * 0.001)]
        time = np.linspace(-.5, 1, wave.shape[0])
        aligned.append(wave)

    aligned = np.stack(aligned, axis=0)

    return aligned, time



def plotly_aligened_waveforms(DF):

    fig = go.Figure()
    HardBlue = '#0000FF'
    for clss, color in zip(['broad', 'narrow', 'activated'],['black', 'darkgray', HardBlue]):
        lines, t = get_aligned_waveforms(DF.query(f"triple == '{clss}'"))

        # Single waveform examples, decimated,
        decimate = 150
        if lines.shape[0] >= decimate:
            decimator = np.random.choice(lines.shape[0], decimate, replace=False)
            mlines = lines[decimator, :]
        else:
            mlines = lines

        for line in mlines:
            _ = fig.add_trace(
                go.Scatter(x=t, y=line, mode='lines',
                           line=dict(color=color,
                                     width=1),
                           opacity=0.5,
                           showlegend=False))


    # Average for the group, have to do it second so they are on top
    for clss, color in zip(['broad', 'narrow', 'activated'],['black', 'darkgray', HardBlue]):
        lines, t = get_aligned_waveforms(DF.query(f"triple == '{clss}'"))
        _ = fig.add_trace(
                go.Scatter(x=t, y=lines.mean(axis=0), mode='lines',
                           line=dict(color=color, width=3),
                           name=clss))

    #formating
    w, h = 2, 2
    fig.update_layout(template='simple_white',
                      width=96 * w, height=96 * h,
                      margin=dict(t=10, b=10, l=10, r=10),
                      xaxis=dict(title=dict(standoff=0,
                                            text='ms',
                                            font_size=9
                                            ),
                                 tickfont_size=9),
                      yaxis=dict(title=dict(text='',

                                            standoff=0,
                                            ),
                                 showticklabels=False,
                                 showline=False,
                                 ticks='',
                                 tickfont_size=9),
                      showlegend=True,
                      legend=dict(xanchor='right', x=1,
                                  yanchor='bottom', y=0,
                                  font_size=9,
                                  title_text='',
                                  bgcolor="rgba(0,0,0,0)"),
                      )

    return fig



if __name__ == '__main__':
    from src.utils.subsets import all_cells as goodcells

    df = get_celltype_DF(goodcells)
    df, hist_threshold, margin = classify_by_spike_width(df, margin=0.05)
    print(df.head())

    fig = plotly_celltype_histogram(df, hist_threshold, margin)
    fig.show()

    fig = plotly_aligened_waveforms(df)

    fig.show()







