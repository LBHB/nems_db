from os.path import basename, join
import logging
import os
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nems_lbhb import baphy_io

log = logging.getLogger(__name__)

def get_depth_details(siteids, sw_thresh='auto', verbose=True):
    """
    Get details on depth/area/laminar info for list of sites
    :param siteids:
    :return: dinfo - dataframe
    """
    if type(siteids) is str:
        siteids = [siteids]
    df = []
    for siteid in siteids:
        try:
            d = baphy_io.get_spike_info(siteid=siteid)

            if sw_thresh == 'auto':
                if siteid[:3] in ['AMT','ARM','BRT','DRX','TNC','bbl']: #'CLT'?
                    # UCLA probes / Intan headstage
                    sw_thresh = 0.4
                else:
                    # Neuropixels
                    sw_thresh = 0.35
            d['narrow'] = d['sw'] < sw_thresh
            df.append(d)
        except:
            log.info(f"error loading siteinfo for {siteid}")
    df = pd.concat(df)
    df = df[['siteid', 'probechannel', 'layer', 'depth', 'area', 'iso', 'sw', 'mwf', 'narrow']]

    if verbose:
        df.plot.hist(y='sw',bins=40, figsize=(3,2))
        plt.axvline(sw_thresh,color='r')
    df.loc[df['layer'] == '5', 'layer'] = '56'
    df.loc[df['layer'] == '3', 'layer'] = '13'
    df.loc[df['layer'] == 'BS', 'layer'] = '13'
    s_group = ['13']  # ,'44']
    df['superficial'] = df['layer'].isin(s_group)
    df['sorted_class'] = -1
    df.loc[df['narrow'] & df['superficial'], 'sorted_class'] = 0
    df.loc[~df['narrow'] & df['superficial'], 'sorted_class'] = 1
    df.loc[df['narrow'] & ~df['superficial'], 'sorted_class'] = 2
    df.loc[~df['narrow'] & ~df['superficial'], 'sorted_class'] = 3
    types = ['NS', 'RS', 'ND', 'RD']
    df['celltype'] = df['sorted_class'].apply(lambda x: types[x])

    return df
