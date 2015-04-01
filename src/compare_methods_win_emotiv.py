 # -*- coding: utf-8 -*-
 #!/usr/bin/env python

"""
Analysis of two frequencies recording with protocol defined in config.py
"""

import numpy as np
import scipy.io as sio
from scipy import signal
import os
from util import config, preprocessing, featex, offline, performance
import matplotlib.pyplot as plt

def apply_method(data, windowSize, segmenting, criterion, method):
    if segmenting == 'sliding':
        label = offline.make_label_windows(config.RECORDING_ITERATIONS, config.RECORDING_PERIOD, windowSize, len(FREQUENCIES))
        windows = offline.extract_windowed_segment(data, windowSize, config.RECORDING_PERIOD, config.FS)
    elif segmenting == 'nooverlap':
        label = offline.make_label_segments(config.RECORDING_ITERATIONS, config.RECORDING_PERIOD, windowSize, len(FREQUENCIES))
        windows = offline.extract_segment_start(data, windowSize, config.RECORDING_PERIOD, config.FS)

    if criterion == 'offline':
        o = offline.offline_classify(windows, FREQUENCIES, method)
        cm = performance.get_confusion_matrix(label, o, len(FREQUENCIES))
        return 100 * performance.get_accuracy(cm), 0
        # return performance.get_cohen_k(cm), 0
    elif criterion == 'pseudoon':
        o, avg_time = offline.pseudo_online_classify(windows, FREQUENCIES, config.FS, method, pause=0, period=config.RECORDING_PERIOD)
        cm = performance.get_confusion_matrix(label, o, len(FREQUENCIES))
        return 100 * performance.get_accuracy(cm), avg_time


FREQUENCIES = [6.4, 6.9, 8]

def compare_methods(dataFile):
    # read data
    data = sio.loadmat(os.path.join(config.DATA_PATH, dataFile))
    X = data['X'].astype('float32', copy=False)

    # select channels and best CCAchannels for PSDA
    CARchannels = np.array(['F3','P7','O1','O2','P8','F4'])
    # X = X[:, np.in1d(np.array(config.SENSORS), CARchannels)]

    # Filtro passaalto
    Wcritic = np.array([0., 4., 5., 64.])
    b, a = preprocessing._get_fir_filter(Wcritic, config.FS, mask=[0, 1])
    X = signal.filtfilt(b, (a,), X, axis=0)
    # X = signal.fftconvolve(X, b[:, np.newaxis], mode='valid')

    # CAR FILTER
    X -= X.mean(axis=0)
    X = np.dot(X, preprocessing.CAR(X.shape[1]))

    accuracies = []
    [accuracies.append([]) for i in xrange(3)]
    undefineds = []
    [undefineds.append([]) for i in xrange(3)]
    tries = range(3, 9)

    myChannels = np.array(['O1','O2'])
    X = X[:, np.in1d(CARchannels,myChannels)].reshape(len(X), len(myChannels))

    # Comparison parameters
    # criterion == 'offline' -> classifier just a criterion of maxima. Windows->Output 1:1
    # criterion == 'pseudoon' -> classifier with a confidence criterion. Windows->Output 1:(0|1)
    criterion = 'pseudoon'
    # segment == 'sliding' -> sliding windows with slide = 1 s
    # segment == 'nooverlap' -> windows with no overlap. Only the first and, if present, the second, are considered
    segmenting = 'sliding'

    for i in tries:
        print 'Window %d' % i
        if criterion == 'offline':
            actualWindow = i
        elif criterion == 'pseudoon':
            actualWindow = i - 1

        method = featex.CCA(list(FREQUENCIES), actualWindow * config.FS, config.FS)
        acc, avg_time = apply_method(X, i, segmenting, criterion, method)
        accuracies[0].append(acc)
        # undefineds[0].append(i + und / (1 - und))

        method = featex.MSI(list(FREQUENCIES), actualWindow * config.FS, config.FS)
        acc, avg_time = apply_method(X, i, segmenting, criterion, method)
        accuracies[1].append(acc)
        # undefineds[1].append(i + und / (1 - und))

        method = featex.PSDA(list(FREQUENCIES),  actualWindow * config.FS, config.FS)
        acc, avg_time = apply_method(X, i, segmenting, criterion, method)
        accuracies[2].append(acc)
        # undefineds[2].append(i + und / (1 - und))


    fig = plt.figure(1)
    fig.suptitle(dataFile + ' ' + ":".join(CARchannels[np.in1d(CARchannels,myChannels)]))
    plt.subplot(121)
    # plt.ylim((33., 100.))
    plt.xlabel("Window length")
    plt.ylabel("Accuracy")
    plt.plot(tries, accuracies[0], '-or', label='CCA')
    plt.plot(tries, accuracies[1], '-oy', label='MSI')
    plt.plot(tries, accuracies[2], '-og', label='PSDA')
    plt.legend()
    plt.grid()
    plt.subplot(122)
    plt.xlabel("Window length")
    plt.ylabel("Time for command")
    plt.plot(tries, undefineds[0], 'r', label='CCA')
    plt.plot(tries, undefineds[1], 'y', label='MSI')
    plt.plot(tries, undefineds[2], 'g', label='PSDA')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    sample = "fullscreen_carlos1.mat"
    compare_methods(sample)