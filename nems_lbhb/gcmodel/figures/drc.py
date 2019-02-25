import matplotlib.pyplot as plt

from nems_lbhb.gcmodel.contrast import make_contrast_signal
from nems_lbhb.gcmodel.drc import rec_from_DRC

def test_DRC_with_contrast(ms=200, normalize=True, fs=100, bands=1,
                           percentile=50, n_segments=12):
    '''
    Plot a sample DRC stimulus next to assigned contrast
    and calculated contrast.
    '''
    drc = rec_from_DRC(fs=fs, n_segments=n_segments)
    rec = make_contrast_signal(drc, name='binary', continuous=False, ms=ms,
                                percentile=percentile, bands=bands)
    rec = make_contrast_signal(rec, name='continuous', continuous=True, ms=ms,
                                bands=bands, normalize=normalize)
    s = rec['stim'].as_continuous()
    c1 = rec['contrast'].as_continuous()
    c2 = rec['binary'].as_continuous()
    c3 = rec['continuous'].as_continuous()

    fig, axes = plt.subplots(4, 1)

    plt.sca(axes[0])
    plt.title('DRC stim')
    plt.imshow(s, aspect='auto', cmap=plt.get_cmap('jet'))

    plt.sca(axes[1])
    plt.title('Assigned Contrast')
    plt.imshow(c1, aspect='auto')

    plt.sca(axes[2])
    plt.title('Binary Calculated Contrast')
    plt.imshow(c2, aspect='auto')

    plt.sca(axes[3])
    plt.title('Continuous Calculated Contrast')
    plt.imshow(c3, aspect='auto')

    plt.tight_layout(h_pad=0.15)
    return fig
