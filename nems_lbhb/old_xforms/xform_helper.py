import nems_lbhb.baphy as nb


def get_recording_file(cellid, batch, options={}):

    # options["batch"] = batch
    # options["cellid"] = cellid
    uri = nb.baphy_data_path(cellid, batch, **options)

    return uri


def generate_recording_uri(cellid, batch, loader):
    """
    figure out filename (or eventually URI) of pre-generated
    NEMS-format recording for a given cell/batch/loader string
    very baphy-specific. Needs to be coordinated with loader processing
    in nems.xform_helper
    """

    options = {}
    if loader in ["ozgf100ch18", "ozgf100ch18n"]:
        options = {'rasterfs': 100, 'includeprestim': True,
                   'stimfmt': 'ozgf', 'chancount': 18}

    elif loader in ["ozgf100ch18pup", "ozgf100ch18npup"]:
        options = {'rasterfs': 100, 'stimfmt': 'ozgf',
                   'chancount': 18, 'pupil': True, 'stim': True,
                   'pupil_deblink': True, 'pupil_median': 2}

    elif (loader.startswith("nostim200pup") or loader.startswith("psth200pup")
          or loader.startswith("psths200pup")):
        options = {'rasterfs': 200, 'stimfmt': 'parm',
                   'chancount': 0, 'pupil': True, 'stim': False,
                   'pupil_deblink': 1, 'pupil_median': 0.5}

    elif loader.startswith("nostim10pup") or loader.startswith("psth10pup"):
        options = {'rasterfs': 10, 'stimfmt': 'parm',
                   'chancount': 0, 'pupil': True, 'stim': False,
                   'pupil_deblink': True, 'pupil_median': 2}

    elif (loader.startswith("nostim20pup") or loader.startswith("psth20pup")
          or loader.startswith("psths20pup")
          or loader.startswith("evt20pup")):
        options = {'rasterfs': 20, 'stimfmt': 'parm',
                   'chancount': 0, 'pupil': True, 'stim': False,
                   'pupil_deblink': 1, 'pupil_median': 0.5}

    elif (loader.startswith("nostim20") or loader.startswith("psth20")
          or loader.startswith("psthm20") or loader.startswith("psths20")):
        options = {'rasterfs': 20, 'stimfmt': 'parm',
                   'chancount': 0, 'pupil': False, 'stim': False}

    elif (loader.startswith("env100") or loader.startswith("envm100")):
        options = {'rasterfs': 100, 'stimfmt': 'envelope', 'chancount': 0}

    elif loader.startswith("env200"):
        options = {'rasterfs': 200, 'stimfmt': 'envelope', 'chancount': 0}

    else:
        raise ValueError('unknown loader string')

    recording_uri = get_recording_file(cellid, batch, options)

    return recording_uri