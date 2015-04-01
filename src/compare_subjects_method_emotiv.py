# -*- coding: utf-8 -*-
# !/usr/bin/env python

import numpy as np
import scipy.io as sio
from scipy import signal
import os
from util import config, preprocessing, featex, offline, performance
import matplotlib.pyplot as plt


def apply_method(data, windowSize, segmenting, criterion, method, prot_iterations, prot_period):
    if segmenting == 'sliding':
        label = offline.make_label_windows(prot_iterations, prot_period, windowSize, len(FREQUENCIES))
        windows = offline.extract_windowed_segment(data, windowSize, prot_period, FS)
    elif segmenting == 'nooverlap':
        label = offline.make_label_segments(prot_iterations, prot_period, windowSize, len(FREQUENCIES))
        windows = offline.extract_segment_start(data, windowSize, prot_period, FS)
    else:
        raise AttributeError

    if criterion == 'offline':
        o = offline.offline_classify(windows, FREQUENCIES, method)
        cm = performance.get_confusion_matrix(label, o, len(FREQUENCIES))
        return performance.get_accuracy(cm), 0
    elif criterion == 'pseudoon':
        o, avg_time = offline.pseudo_online_classify(windows, FREQUENCIES, FS, method, pause=0, period=prot_period)
        cm = performance.get_confusion_matrix(label, o, len(FREQUENCIES))
        return performance.get_accuracy(cm), avg_time
    else:
        raise AttributeError


def perform(name):
    """
    Perform preprocessing and classification using one method and one window length
    Return a vector: [acc avg_time itr utility]
    """
    # read data
    data = sio.loadmat(os.path.join(config.DATA_PATH, name))
    X = data['X']

    # select channels and best CCAchannels for PSDA
    CARchannels = np.array(['F3','P7','O1','O2','P8','F4'])
    # X = X[:, np.in1d(np.array(config.SENSORS), CARchannels)]

    # Filtro passaalto
    Wcritic = np.array([0., 4., 5., 64.])
    b, a = preprocessing._get_fir_filter(Wcritic, FS, mask=[0, 1])
    X = signal.filtfilt(b, (a,), X, axis=0)

    # CAR FILTER
    X -= X.mean(axis=0)
    X = np.dot(X, preprocessing.CAR(X.shape[1]))

    myChannels = np.array(CHANNELS)
    X = X[:, np.in1d(CARchannels,myChannels)].reshape(len(X), len(myChannels))

    # Comparison parameters
    # criterion == 'offline' -> classifier just a criterion of maxima. Windows->Output 1:1
    # criterion == 'pseudoon' -> classifier with a confidence criterion. Windows->Output 1:(0|1)
    criterion = 'pseudoon'
    # segment == 'sliding' -> sliding windows with slide = 1 s
    # segment == 'nooverlap' -> windows with no overlap. Only the first and, if present, the second, are considered
    segmenting = 'sliding'
    method = METHOD(list(FREQUENCIES), (WINLENGTH - 1) * FS, FS)

    acc, avg_time = apply_method(X, WINLENGTH, segmenting, criterion, method,
                                 prot_iterations=config.RECORDING_ITERATIONS, prot_period=config.RECORDING_PERIOD)
    # avg_time = WINLENGTH + und / (1 - und)
    itr = performance.get_ITR(4, acc, avg_time) * 60
    ut = performance.get_utility(6, acc, avg_time) * 60
    records = np.array([100 * acc, avg_time, itr, ut])
    print '##'

    return records


if __name__ == '__main__':
    FREQUENCIES = [6.4, 6.9, 8.0]
    FS = config.FS
    CHANNELS = ['O1']
    METHOD = featex.CCA

    SUBJECTS = ['alan1', 'alan2', 'alex1', 'alex2', 'carlos1', 'carlos2', 'flavio1', 'flavio2']

    for WINLENGTH in [3]:
        data = np.zeros((len(SUBJECTS), 4))
        for ii, name in enumerate(SUBJECTS):
            DATA_FILE = "fullscreen_%s.mat" % name
            data[ii, :] = perform(DATA_FILE)

        filename = "mio_CCAO1_" + str(WINLENGTH) + ".txt"
        np.savetxt(filename, data, fmt="%.2f", delimiter=',')
