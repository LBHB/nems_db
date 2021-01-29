import matplotlib.pyplot as plt

import nems
import nems.db as nd


batch = 322
c2d_models = [
     'ozgf.fs100.ch18.pop-ld-cc.10-norm.l1-popev_conv2d.4.7x3.rep3-wcn.30-relu.30-wc.30xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.20-norm.l1-popev_conv2d.4.7x3.rep3-wcn.30-relu.30-wc.30xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.30-norm.l1-popev_conv2d.4.7x3.rep3-wcn.30-relu.30-wc.30xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.50-norm.l1-popev_conv2d.4.7x3.rep3-wcn.30-relu.30-wc.30xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.75-norm.l1-popev_conv2d.4.7x3.rep3-wcn.30-relu.30-wc.30xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.100-norm.l1-popev_conv2d.4.7x3.rep3-wcn.30-relu.30-wc.30xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.150-norm.l1-popev_conv2d.4.7x3.rep3-wcn.30-relu.30-wc.30xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.0-norm.l1-popev_conv2d.4.7x3.rep3-wcn.30-relu.30-wc.30xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
]
c1dx2_models = [
     'ozgf.fs100.ch18.pop-ld-cc.10-norm.l1-popev_wc.18x30.g-fir.1x12x30-relu.30-wc.30x50-fir.1x8x50-relu.50-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.20-norm.l1-popev_wc.18x30.g-fir.1x12x30-relu.30-wc.30x50-fir.1x8x50-relu.50-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.30-norm.l1-popev_wc.18x30.g-fir.1x12x30-relu.30-wc.30x50-fir.1x8x50-relu.50-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.50-norm.l1-popev_wc.18x30.g-fir.1x12x30-relu.30-wc.30x50-fir.1x8x50-relu.50-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.75-norm.l1-popev_wc.18x30.g-fir.1x12x30-relu.30-wc.30x50-fir.1x8x50-relu.50-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.100-norm.l1-popev_wc.18x30.g-fir.1x12x30-relu.30-wc.30x50-fir.1x8x50-relu.50-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.150-norm.l1-popev_wc.18x30.g-fir.1x12x30-relu.30-wc.30x50-fir.1x8x50-relu.50-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.0-norm.l1-popev_wc.18x30.g-fir.1x12x30-relu.30-wc.30x50-fir.1x8x50-relu.50-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4'
]
c1d_models = [
     'ozgf.fs100.ch18.pop-ld-cc.10-norm.l1-popev_wc.18x30.g-fir.1x20x30-relu.30.f-wc.30x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.20-norm.l1-popev_wc.18x30.g-fir.1x20x30-relu.30.f-wc.30x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.30-norm.l1-popev_wc.18x30.g-fir.1x20x30-relu.30.f-wc.30x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.50-norm.l1-popev_wc.18x30.g-fir.1x20x30-relu.30.f-wc.30x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.75-norm.l1-popev_wc.18x30.g-fir.1x20x30-relu.30.f-wc.30x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.100-norm.l1-popev_wc.18x30.g-fir.1x20x30-relu.30.f-wc.30x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.150-norm.l1-popev_wc.18x30.g-fir.1x20x30-relu.30.f-wc.30x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'ozgf.fs100.ch18.pop-ld-cc.0-norm.l1-popev_wc.18x30.g-fir.1x20x30-relu.30.f-wc.30x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',

]

# TODO:
# thoughts: the performance comparison between cell counts might not be meaningful if including all cells for each count.
# i.e. it might not be a better fit, might just be that the 10 random cells chosen for cc10 are crappy cells, while
# for cc100 the other 90 cells bring up the average. so might be better to compare average performance on a common subset
# i.e. cc100 still fit to 100 cells, but for plot use the avg r_test from the same 10 cell sused for cc10
# (should have 10 common to all cc b/c fixing random seed, so sample should start the same each time)
# -- this is what the nems_web comparison already does, conveniently, so can allready see the trend

#results = nd.batch_comp(batch, all_models, stat='r_test')  # TODO: why isn't r_ceiling being computed?
#means = results.mean()
cell_counts = [10, 20, 30, 50, 75, 100, 150, 268]  # 268 is all cells, cc0

fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(8,6), sharex=True, sharey=True)
for i, ax in enumerate([a1, a2, a3, a4]):
     # each time, truncate first i models, so that common mean is taken over a larger subset
     all_models = c2d_models[i:] + c1dx2_models[i:] + c1d_models[i:]
     results = nd.batch_comp(batch, all_models, stat='r_test')  # TODO: why isn't r_ceiling being computed?
     means = results.mean()
     c2d_means = means.loc[c2d_models[i:]].values
     c1dx2_means = means.loc[c1dx2_models[i:]].values
     c1d_means = means.loc[c1d_models[i:]].values

     ax.plot(cell_counts[i:], c2d_means, color='darkgreen', marker='s', markersize=6, label='conv2d')
     ax.plot(cell_counts[i:], c1dx2_means, color='purple', marker='^', markersize=6, label='conv1dx2')
     ax.plot(cell_counts[i:], c1d_means, color='black', marker='o', markersize=6, label='conv1d')
     if i == 0:
          ax.legend()
     ax.set_xlabel('Number of cells fit')
     ax.set_ylabel('Mean pred corr on %d cells' % cell_counts[i:][0])

fig.tight_layout()


# TODO: alternatively, get results for each cell count individually and take average over all cells used for the fit