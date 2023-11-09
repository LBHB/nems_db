"""
Path settings for where to store training data and model results
"""

try:
    import pupil_settings_local as psl
except ImportError:
    psl = None

if hasattr(psl, 'DEFAULT_DLC_MODEL'):
    DEFAULT_DLC_MODEL = psl.DEFAULT_DLC_MODEL
else:
    #DEFAULT_DLC_MODEL = '/auto/data/dlc/multivid-CLL-2022-01-14/config.yaml'
    # Prince LH
    #DEFAULT_DLC_MODEL = '/auto/data/dlc/free_top_2022_11/config.yaml'
    # Prince RH
    DEFAULT_DLC_MODEL = '/auto/data/dlc/free_top_RH-jereme-2023-02-17/config.yaml'

def get_dlc_model(avipath):
    # single chimney lh?
    
    if ('PRN010' in avipath) | ('PRN020' in avipath) | ('PRN030' in avipath):
        # single chimney, LH
        dlc_model = '/auto/data/dlc/free_top_2022_11/config.yaml'
    elif ('PRN04' in avipath) | ('PRN05' in avipath) | ('PRN06' in avipath) | ('PRN07' in avipath):
        # single chimney, RH
        dlc_model = '/auto/data/dlc/free_top_RH-jereme-2023-02-17/config.yaml'
    elif 'LemonDisco/training2023' in avipath:
        dlc_model = '/auto/data/dlc/free_train-svd-2023-09-26/config.yaml'
    elif 'SlipperyJack/training2023' in avipath:
        dlc_model = '/auto/data/dlc/free_train-svd-2023-09-26/config.yaml'
    elif 'SlipperyJack/SLJ' in avipath:
        # coordinates on corners of chimney
        #dlc_model = '/auto/data/dlc/two_chimney-svd-2023-09-28/config.yaml'
        # switched to tether entry points.
        dlc_model = '/auto/data/dlc/two_chimney2-svd-2023-11-03/config.yaml'
    elif 'LemonDisco/LMD' in avipath:
        dlc_model = '/auto/data/dlc/two_chimney2-svd-2023-11-03/config.yaml'
    else:
        raise ValueError('No DLC model specified for this path')

    return dlc_model
