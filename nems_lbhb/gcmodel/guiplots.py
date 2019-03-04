import nems.modelspec as ms
import nems.plots.api as nplt


def contrast_kernel_output(rec, modelspec, ax=None, title=None,
                           idx=0, channels=0, xlabel='Time', ylabel='Value',
                           **options):

    output = ms.evaluate(rec, modelspec, stop=idx+1)['ctpred']
    nplt.timeseries_from_signals([output], channels=channels, xlabel=xlabel,
                                 ylabel=ylabel, ax=ax, title=title)

    return ax
