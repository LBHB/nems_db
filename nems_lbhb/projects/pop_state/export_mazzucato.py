import logging
from nems0 import db
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems0.recording import load_recording
from os.path import basename, join
log = logging.getLogger(__name__)

force_recache=False

# find all the cells in a batch:

batch=343
batch=294

# ask nems to figure out the relevant recording uris
loadkey = "gtgram.fs100.ch32"
loadkey = "gtgram.fs100.ch32.pup"

siteids, cellids = db.get_batch_sites(batch=batch)

print("Siteids:", siteids)
print("Cellids:", cellids)
print("Count:", len(siteids))
print("loadkey:", loadkey)

uri_list=[]

for index, cellid in enumerate(cellids):
    uri_list.append(generate_recording_uri(cellid=cellid, batch=batch, loadkey=loadkey, recache=force_recache))
    log.info(cellid +": "+ uri_list[-1])
