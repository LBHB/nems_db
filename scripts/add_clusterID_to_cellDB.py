"""
For site, go through all cellids (might be sorted separately per file) and add the KiloSort clusterID 
to gSingleRaw for each cellid. This allows us to easily add things like sorting metrics post-hoc to celldb

Crude script:
Hardcoded for KS2 right now. Also, won't work if you've run multiple sorting jobs for the same set of 
run_nums each w/ diff KS2 params.
"""

import nems.db as nd
import os
import pandas as pd
import numpy as np

site = 'CRD002a'
animal_name = 'Cordyceps'
path = '/auto/data/daq/{0}/{1}/tmp/KiloSort/'.format(animal_name, site[:6])

# get all unique cellid/file pairs at this site
sql = "SELECT DISTINCT gSingleRaw.cellid, gSingleRaw.channum, gSingleRaw.unit, sCellFile.respfile from " \
            "gSingleRaw JOIN sCellFile ON (sCellFile.rawid=gSingleRaw.rawid) " \
            "WHERE gSingleRaw.cellid like %s"
site_regex = '%' + site + '%'
d = nd.pd_query(sql, (site_regex,))

# get rid of shitty "a1" etc. channels (WHERE ARE THEY COMING FROM???)
idx = []
for i, cid in enumerate(d.cellid):
    try:
        x=int(cid.split('-')[1])
    except ValueError:
        idx.append(i)
d = d.drop(idx, axis='index')

# Now, for each unique cellid, get the corresponding KS cluster ID and add it to the df "d"
cid_unique = d.cellid.unique()
for cid in cid_unique:
    rf = d.loc[d.cellid==cid, 'respfile']
    rn = [f[7:9] for f in rf]
    regex = '_'.join(rn)
    results_dir = os.path.join(path, [di for di in os.listdir(path) if regex == di.split(site+'_')[1].split('_KiloS')[0]][0], 'results')

    # get clusterIDs based on best channels
    best_chans = np.load(os.path.join(results_dir, 'best_channels.npy'))
    sorted_chans = pd.read_csv(os.path.join(results_dir, 'cluster_group.tsv'), sep='\t')
    cluster_IDs = sorted_chans.cluster_id[sorted_chans.group.isin(['mua', 'good'])].values
    best_chans = best_chans[sorted_chans.group.isin(['mua', 'good'])]

    this_chan_num = d[d.cellid==cid].channum.unique() - 1
    this_un_num = d[d.cellid==cid].unit.unique() - 1  
    # subtract min unit number for this channel/file pair to deal w/ appended units
    un_offset = d[(d.channum==this_chan_num[0]+1) & d.respfile.isin(rf.values)].unit.min() - 1
    this_un_num -= un_offset
    cluster_idx = np.argwhere(best_chans==this_chan_num)

    if len(cluster_idx) < this_un_num:
        raise ValueError("what's up??")

    this_clusterIDs = cluster_IDs[cluster_idx]
    this_clusterID = this_clusterIDs[this_un_num]

    d.at[d.cellid==cid, 'clusterID'] = int(this_clusterID)

update_gSingleRaw = input("Successfully ID'd KS clusterIDs for all sorted units at site {0}. \n Would you like to"
                        "update these values in the gSingleRaw table? 'Y/n' \n".format(site))

if (update_gSingleRaw == 'y') | (update_gSingleRaw == 'Y'):
    for cid in cid_unique:
        ksid = d[d.cellid==cid].clusterID.unique()
        if len(ksid) > 1:
            raise ValueError("Should have unique cluster ID per cellid")
        else:
            ksid = ksid[0]
        print("Updating gSingle Raw entry for cellid: {}".format(cid))
        sql_update = "UPDATE gSingleRaw SET kilosort_cluster_id = {0} WHERE cellid = '{1}'".format(int(ksid), cid)
        nd.sql_command(sql_update)