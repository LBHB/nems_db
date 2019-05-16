import scipy.io as spio
import scipy.signal as ss
import numpy as np
import scipy.ndimage.filters as sf
import os
import nems_lbhb.pup_py.keras_classes as kc
import skimage
import warnings


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def resize(im, size=(224, 224)):
    """
    Function to resize image matrix to 224 x 224. Wrote for implementation of DenseNet in keras
    """

    current_shape = im.shape
    resamp_im = im.copy()

    # resample first axis
    if current_shape[0] != size[0]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resamp_im = ss.resample(resamp_im, size[0], axis=0)

    else:
        pass

    # resample second axis
    if current_shape[1] != size[1]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resamp_im = ss.resample(resamp_im, size[1], axis=1)

    else:
        pass

    scale_factor = (size[0] / current_shape[0], size[1] / current_shape[1])

    return scale_factor, resamp_im


def augment(X, y):
    """
    Modify the input image X. Choose from list of random augmentation operations
    :param X: input image
    :param y: parameters for the image X
    :return: modified image
    """

    modX = X.copy()
    modY = y.copy()

    # choose combos of augmentation
    nops = np.random.choice([1, 2, 3], 1)[0]
    method = np.random.choice(['blur', 'flip', 'rotate', 'noise', 'none'], nops)

    ax0 = modX.shape[0]
    ax1 = modX.shape[1]

    if 'none' in method:
        method = ['none']

    if 'blur' in method:
        sigma = np.random.choice(np.arange(1, 3, 0.2), 1)[0]
        modX = sf.gaussian_filter(modX, sigma)

    if 'noise' in method:
        std = np.random.choice(np.arange(0, 3, 0.01), 1)[0]
        noise = std * (np.random.random(modX.shape) - 0.5)
        modX = modX + noise

    if 'flip' in method:

        vert_horz = np.random.choice([True, False], 1)[0]
        if vert_horz == True:
            modX = modX[:, ::-1]
            modY[0] = modX.shape[-1] - modY[0]
            modY[-1] = -modY[-1]
        else:
            modX = modX[::-1, :]
            modY[1] = modX.shape[0] - modY[1]
            modY[-1] = -modY[-1]

    if 'rotate' in method:
        # define rotation params
        deg = np.random.choice(np.arange(-8, 8, 0.5), 1)[0]
        center = (modY[0], modY[1])  # rotate around center of pupil

        # rotate image
        modX = skimage.transform.rotate(modX, deg, center=center)

        # crop image to remove the 0 padded pixels... hopefully... this is kludgy.
        modX = modX[int(ax0 - 0.85 * ax0):int(0.85 * ax0), int(ax1 - 0.85 * ax1):int(0.85 * ax1)]

        # update ellipse params
        modY[-1] = modY[-1] - (deg * np.pi / 180)
        modY[0] = modY[0] - int(ax1 - 0.85 * ax1)
        modY[1] = modY[1] - int(ax0 - 0.85 * ax0)

    if 'none' in method:
        pass

    # randomly crop all images (effectively translating them, but will also mess with scale)
    xcrop = np.random.choice(np.arange(0.8, 1, 0.01), 1)[0]
    ycrop = np.random.choice(np.arange(0.8, 1, 0.01), 1)[0]

    up = np.random.choice([False, True], 1)[0]
    right = np.random.choice([False, True], 1)[0]

    if right:
        e = int(xcrop * modX.shape[0])
        modX = modX[:e, :]
    else:
        s = modX.shape[0] - int(xcrop * modX.shape[0])
        modX = modX[s:, :]
        modY[1] = modY[1] - s

    if up:
        e = int(ycrop * modX.shape[1])
        modX = modX[:, :e]
    else:
        s = modX.shape[1] - int(ycrop * modX.shape[1])
        modX = modX[:, s:]
        modY[0] = modY[0] - s

    return modX, modY

def train(model, epochs=1):

    # data path
    path = 'training_data/data/'
    training_files = os.listdir(path)
    n_training_files = len(training_files)

    params = {
        'batch_size': 16,
        'image_dim': (224, 224),
        'n_parms': 5,
        'n_channels': 3,
        'shuffle': True,
    }

    # define train indexes
    train = [i[0] for i in np.argwhere([True if 'AMT004b' not in i else False for i in training_files])]

    # define validation indexes
    test = [i[0] for i in np.argwhere([True if 'AMT004b' in i else False for i in training_files])]

    partition = {
        'train': train,
        'validation': test
    }

    training_generator = kc.DataGenerator(partition['train'], **params)
    validation_generator = kc.DataGenerator(partition['validation'], **params)

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=6, epochs=epochs)

    return model
