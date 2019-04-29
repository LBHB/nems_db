import os

import matplotlib.pyplot as plt

from nems_lbhb.gcmodel.figures.summary import (performance_scatters,
                                               performance_bar, significance)
from nems_lbhb.gcmodel.figures.equivalence import (equivalence_scatter,
                                                   equivalence_histogram,
                                                   gc_vs_stp_strengths)
from nems_lbhb.gcmodel.figures.schematic import (contrast_breakdown,
                                                 contrast_vs_stp_comparison)
# Put all aliased modelnames and cellids in memory,
# e.g. test = "ozgf.fs100.ch18-ld-sev_wc.18x1.g-fir.1x15-lvl.1_init-basic"
from nems_lbhb.gcmodel.figures.definitions import *


# Plot appearance settings
params = {
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 1,
        #'font.weight': 'bold',
        'font.size': 16,
        }
plt.rcParams.update(params)


# Defaults
default_cell = 'TAR010c-13-1'
default_batch = 289
default1 = gc_av
default2 = stp_dexp
default3 = ln_dexp
default4 = gc_av_stp


def run_all(cellid=default_cell, batch=default_batch, model1=default1,
            model2=default2, model3=default3, model4=default4, se_filter=True,
            ln_filter=False, sample_every=5, save=False):
    '''
    model1: GC
    model2: STP
    model3: LN
    model4: GC+STP

    '''

    save_directory = ("/auto/users/jacob/notes/gc_figures/matplot_figs/"
                      "dexp_pdfs/")
    f1 = performance_scatters(batch, model1, model2, model3, model4,
                              se_filter=se_filter, ln_filter=ln_filter)
    f2 = equivalence_scatter(batch, model1, model2, model3, se_filter=se_filter,
                             ln_filter=ln_filter)
    f3 = performance_bar(batch, model1, model2, model3, model4,
                         se_filter=se_filter, ln_filter=ln_filter)
    f4 = significance(batch, model1, model2, model3, model4,
                      se_filter=se_filter, ln_filter=ln_filter)
    f5 = contrast_breakdown(cellid, batch, model1, model2, model3,
                            sample_every=sample_every)
    f6 = contrast_vs_stp_comparison(cellid, batch, model1, model2, model3,
                                    model4)

    if save:
        f1.savefig(save_directory + 'performance_scatters' + '.pdf')
        f2.savefig(save_directory + 'correlation_scatter' + '.pdf')
        f3.savefig(save_directory + 'summary_bar' + '.pdf')
        f4.savefig(save_directory + 'significance' + '.pdf')
        f5.savefig(save_directory + 'gc_schematic' + '.pdf')
        f6.savefig(save_directory + 'gc_vs_stp_comparison' + '.pdf')
        plt.close('all')


def example_cells(batch=default_batch, model1=default1, model2=default2,
                  model3=default3, model4=default4, run_id=0):
    example_cells = ['bbl104h-33-1', 'BRT026c-16-2', 'TAR009d-22-1',
                     'TAR010c-13-1', 'TAR010c-20-1', 'TAR010c-58-2',
                     'TAR017b-04-1', 'TAR017b-22-1', 'gus018b-a2',
                     'gus019c-a2', 'TAR009d-15-1']
    save_directory = ("/auto/users/jacob/notes/gc_figures/matplot_figs/"
                      "example_cells/run%d/" % run_id)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for c in example_cells:
        f = contrast_vs_stp_comparison(cellid=c, batch, model1, model2, model3,
                                       model4)
        f.savefig(save_directory + c + '.pdf')
        f.savefig(save_directory + c + '.png')

        plt.close('all')
