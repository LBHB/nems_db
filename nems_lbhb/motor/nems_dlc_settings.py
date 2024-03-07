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
    
    if ('PRN00' in avipath) | ('PRN01' in avipath) | ('PRN02' in avipath) | ('PRN03' in avipath):
        # single chimney, LH
        dlc_model = '/auto/data/dlc/free_top_2022_11/config.yaml'
        # single chimney, LH, back of implant instead of rear headpost
        dlc_model = '/auto/data/dlc/left_chimney_PRNfb-svd-2024-01-11/config.yaml'

    elif ('PRN04' in avipath) | ('PRN05' in avipath) | ('PRN06' in avipath) | ('PRN07' in avipath):
        # single chimney, RH
        #dlc_model = '/auto/data/dlc/free_top_RH-jereme-2023-02-17/config.yaml'
        #dlc_model = '/auto/data/dlc/right_chimney_PRN-svd-2023-11-21/config.yaml'
        # back of implant instead of rear headpost
        dlc_model = '/auto/data/dlc/right_chimney_PRNfb-svd-2023-11-30/config.yaml'
    elif ('LemonDisco/training' in avipath) or \
            ('SlipperyJack/training' in avipath):
        # NO CHIMNEYS
        dlc_model = '/auto/data/dlc/free_train-svd-2023-09-26/config.yaml'
    elif ('SlipperyJack/SLJ' in avipath) or \
            ('LemonDisco/LMD' in avipath):
        # TWO CHIMNEYS
        # coordinates on corners of chimney
        #dlc_model = '/auto/data/dlc/two_chimney-svd-2023-09-28/config.yaml'
        # switched to tether entry points.
        #dlc_model = '/auto/data/dlc/two_chimney2-svd-2023-11-03/config.yaml'
        # back of implant instead of rear headpost
        dlc_model = '/auto/data/dlc/two_chimney3fb-svd-2024-01-10/config.yaml'
    else:
        raise ValueError('No DLC model specified for this path')

    return dlc_model
