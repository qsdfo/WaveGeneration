"""
RNN Vocal Generation Model

Blizzard, Music, and Huckleberry Finn data feeders.
"""

import numpy
np = numpy
#import scikits.audiolab

import random
import time
import os
import glob

__base = [
    ('OnSenBatLesCouillesDeCetteValeur', '/Users/leo/Recherche/WaveGeneration/Data/'),
    ('OnSenBatLesCouillesDeCetteValeur', '/home/aciditeam-leo/Aciditeam/WaveGeneration/Data/'),
]

__data_path = 'le_joueur_no_cond' # in float16 8secs*16000samples/sec

__train_mean_std = np.array([-2.7492260671334582e-05,
                                   0.056233098718291352],
                                   dtype='float64')

__train = lambda s: os.path.join(s, 'matrices/music_train.npy')
__valid = lambda s: os.path.join(s, 'matrices/music_valid.npy')
__test = lambda s: os.path.join(s, 'matrices/music_test.npy')

def find_dataset(path):
    for (k, v) in __base:
        tmp_path = os.path.join(v, path)
        if os.path.exists(tmp_path):
            #print "Path on {}:".format(k)
            #print tmp_path
            return tmp_path
        #print "not found on {}".format(k)
    raise Exception('{} NOT FOUND!'.format(path))

### Basic utils ###
def __round_to(x, y):
    """round x up to the nearest y"""
    return int(numpy.ceil(x / float(y))) * y

def __normalize(data):
    """To range [0., 1.]"""
#   wanna perform abs-max normalization: (x / max(abs(x:min()), x:max()) + 1) / 2
#    data = ((data/(numpy.maximum(numpy.absolute(data.min(axis=1)[:, None]), data.max(axis=1)[:, None]))) + 1.)/2.
    data -= data.min(axis=1)[:, None]
    data /= data.max(axis=1)[:, None]
    return data

def __linear_quantize(data, q_levels):
    """
    floats in (0, 1) to ints in [0, q_levels-1]
    scales normalized across axis 1
    """
    # Normalization is on mini-batch not whole file
    #eps = numpy.float64(1e-5)
    #data -= data.min(axis=1)[:, None]
    #data *= ((q_levels - eps) / data.max(axis=1)[:, None])
    #data += eps/2
    #data = data.astype('int32')

    eps = numpy.float64(1e-5)
    data *= (q_levels - eps)
    data += eps/2
    data = data.astype('int32')
    return data

def __a_law_quantize(data):
    """
    :todo:
    """
    raise NotImplementedError

def linear2mu(x, mu=255):
    """
    From Joao
    x should be normalized between -1 and 1
    Converts an array according to mu-law and discretizes it

    Note:
        mu2linear(linear2mu(x)) != x
        Because we are compressing to 8 bits here.
        They will sound pretty much the same, though.

    :usage:
        >>> bitrate, samples = scipy.io.wavfile.read('orig.wav')
        >>> norm = __normalize(samples)[None, :]  # It takes 2D as inp
        >>> mu_encoded = linear2mu(2.*norm-1.)  # From [0, 1] to [-1, 1]
        >>> print mu_encoded.min(), mu_encoded.max(), mu_encoded.dtype
        0, 255, dtype('int16')
        >>> mu_decoded = mu2linear(mu_encoded)  # Back to linear
        >>> print mu_decoded.min(), mu_decoded.max(), mu_decoded.dtype
        -1, 0.9574371, dtype('float32')
    """
    x_mu = np.sign(x) * np.log(1 + mu*np.abs(x))/np.log(1 + mu)
    return ((x_mu + 1)/2 * mu).astype('int16')

def mu2linear(x, mu=255):
    """
    From Joao with modifications
    Converts an integer array from mu to linear

    For important notes and usage see: linear2mu
    """
    mu = float(mu)
    x = x.astype('float32')
    y = 2. * (x - (mu+1.)/2.) / (mu+1.)
    return np.sign(y) * (1./mu) * ((1. + mu)**np.abs(y) - 1.)

def __mu_law_quantize(data):
    return linear2mu(data)

def __batch_quantize(data, q_levels, q_type):
    """
    One of 'linear', 'a-law', 'mu-law' for q_type.
    """
    data = data.astype('float64')
    data = __normalize(data)
    if q_type == 'linear':
        return __linear_quantize(data, q_levels)
    if q_type == 'a-law':
        return __a_law_quantize(data)
    if q_type == 'mu-law':
        # from [0, 1] to [-1, 1]
        data = 2.*data-1.
        # Automatically quantized to 256 bins.
        return __mu_law_quantize(data)
    raise NotImplementedError

__RAND_SEED = 123
def __fixed_shuffle(inp_list):
    if isinstance(inp_list, list):
        random.seed(__RAND_SEED)
        random.shuffle(inp_list)
        return
    #import collections
    #if isinstance(inp_list, (collections.Sequence)):
    if isinstance(inp_list, numpy.ndarray):
        numpy.random.seed(__RAND_SEED)
        numpy.random.shuffle(inp_list)
        return
    # destructive operations; in place; no need to return
    raise ValueError("inp_list is neither a list nor a numpy.ndarray but a "+type(inp_list))

def __make_random_batches(inp_list, batch_size):
    batches = []
    for i in xrange(len(inp_list) / batch_size):
        batches.append(inp_list[i*batch_size:(i+1)*batch_size])

    __fixed_shuffle(batches)
    return batches

### MUSIC DATASET LOADER ###
def __music_feed_epoch(files,
                       batch_size,
                       seq_len,
                       overlap,
                       q_levels,
                       q_zero,
                       q_type,
                       real_valued=False):
    """
    Helper function to load music dataset.
    Generator that yields training inputs (subbatch, reset). `subbatch` contains
    quantized audio data; `reset` is a boolean indicating the start of a new
    sequence (i.e. you should reset h0 whenever `reset` is True).

    Feeds subsequences which overlap by a specified amount, so that the model
    can always have target for every input in a given subsequence.

    Assumes all flac files have the same length.

    returns: (subbatch, reset)
    subbatch.shape: (BATCH_SIZE, SEQ_LEN + OVERLAP)
    reset: True or False
    """
    import pdb; pdb.set_trace()
    batches = __make_random_batches(files, batch_size)

    for bch in batches:
        # batch_seq_len = length of longest sequence in the batch, rounded up to
        # the nearest SEQ_LEN.
        batch_seq_len = len(bch[0])  # should be 8*16000, if 8 seconds chunks were built in preprocess.py
        batch_seq_len = __round_to(batch_seq_len, seq_len)

        batch = numpy.zeros(
            (batch_size, batch_seq_len),
            dtype='float64'
        )

        mask = numpy.ones(batch.shape, dtype='float32')

        for i, data in enumerate(bch):
            #data, fs, enc = scikits.audiolab.flacread(path)
            # data is float16 from reading the npy file
            batch[i, :len(data)] = data
            # This shouldn't change anything. All the flac files for Music
            # are the same length and the mask should be 1 every where.
            mask[i, len(data):] = numpy.float32(0)

        if not real_valued:
            batch = __batch_quantize(batch, q_levels, q_type)

            # Add very short silence at the end of the chunk
            batch = numpy.concatenate([
                numpy.full((batch_size, overlap), q_zero, dtype='int32'),
                batch
            ], axis=1)
        else:
            batch -= __train_mean_std[0]
            batch /= __train_mean_std[1]
            batch = numpy.concatenate([
                numpy.full((batch_size, overlap), 0, dtype='float32'),
                batch
            ], axis=1).astype('float32')

        mask = numpy.concatenate([
            numpy.full((batch_size, overlap), 1, dtype='float32'),
            mask
        ], axis=1)

        for i in xrange(batch_seq_len // seq_len):
            reset = numpy.int32(i==0)
            subbatch = batch[:, i*seq_len : (i+1)*seq_len + overlap]
            submask = mask[:, i*seq_len : (i+1)*seq_len + overlap]
            yield (subbatch, reset, submask)

def music_train_feed_epoch(*args):
    """
    :parameters:
        batch_size: int
        seq_len:
        overlap:
        q_levels:
        q_zero:
        q_type: One the following 'linear', 'a-law', or 'mu-law'

    4,340 (9.65 hours) in total
    With batch_size = 128:
        4,224 (9.39 hours) in total
        3,712 (88%, 8.25 hours)for training set
        256 (6%, .57 hours) for validation set
        256 (6%, .57 hours) for test set

    Note:
        32 of Beethoven's piano sonatas available on archive.org (Public Domain)

    :returns:
        A generator yielding (subbatch, reset, submask)
    """
    # Just check if valid/test sets are also available. If not, raise.
    find_dataset(__valid(__data_path))
    find_dataset(__test(__data_path))
    # Load train set
    data_path = find_dataset(__train(__data_path))
    files = numpy.load(data_path)
    generator = __music_feed_epoch(files, *args)
    return generator

def music_valid_feed_epoch(*args):
    """
    See:
        music_train_feed_epoch
    """
    data_path = find_dataset(__valid(__data_path))
    files = numpy.load(data_path)
    generator = __music_feed_epoch(files, *args)
    return generator

def music_test_feed_epoch(*args):
    """
    See:
        music_train_feed_epoch
    """
    data_path = find_dataset(__test(__data_path))
    files = numpy.load(data_path)
    generator = __music_feed_epoch(files, *args)
    return generator