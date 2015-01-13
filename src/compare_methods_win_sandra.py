# -*- coding: utf-8 -*-
# !/usr/bin/env python

import numpy as np
import scipy.io as sio
from scipy import signal
import os
from util import config, preprocessing, featex, offline, performance
import matplotlib.pyplot as plt


def apply_method(data, windowSize, segmenting, criterion, method):
    if segmenting == 'sliding':
        label = offline.make_label_windows(ITERATIONS, PERIOD, windowSize, len(FREQUENCIES))
        windows = offline.extract_windowed_segment(data, windowSize, PERIOD, FS)
    elif segmenting == 'nooverlap':
        label = offline.make_label_segments(ITERATIONS, PERIOD, windowSize, len(FREQUENCIES))
        windows = offline.extract_segment_start(data, windowSize, PERIOD, FS)
    else:
        raise AttributeError

    if criterion == 'offline':
        o = offline.offline_classify(windows, FREQUENCIES, method)
        cm = performance.get_confusion_matrix(label, o, len(FREQUENCIES))
        return 100 * performance.get_accuracy(cm), 0
        # return performance.get_cohen_k(cm), 0
    elif criterion == 'pseudoon':
        o, c, m = offline.pseudo_online_classify(windows, FREQUENCIES, FS, method, pause=0, period=PERIOD)
        cm = performance.get_confusion_matrix(label, o, len(FREQUENCIES))
        return 100 * performance.get_accuracy(cm), (m + 0.) / (m + c)
    else:
        raise AttributeError


def process_file(dataFile, winLengths, verbose=True):
    # read data
    myData = sio.loadmat(os.path.join(config.DATA_PATH, dataFile))
    X = myData['data']
    # CAR FILTER & PASSBAND FILTER
    Wcritic = np.array([0., 4., 5., 49., 50., 300.])
    b, a = preprocessing._get_fir_filter(Wcritic, FS, 851)
    # X = signal.lfilter(b, [a,], X, axis=0)[len(b) - 1:,:]
    X = signal.fftconvolve(X, b[:, np.newaxis], mode='valid')

    X -= X.mean(axis=0)
    X = np.dot(X, preprocessing.CAR(X.shape[1]))

    # y1 = X[0:160 * FS, :]  # protocol first part, 20s x 2
    y2 = X[160 * FS: 320 * FS, :]  # protocol second part, 10s x 4
    y3 = X[320 * FS: 420 * FS, :]  # protocol first part, 5s x 5

    accuracies = np.zeros((5, len(lengths)))
    ITRs = np.zeros((5, len(lengths)))
    undefineds = np.zeros((5, len(lengths)))

    # Comparison parameters
    # criterion == 'offline' -> classifier just a criterion of maxima. Windows->Output 1:1
    # criterion == 'pseudoon' -> classifier with a confidence criterion. Windows->Output 1:(0|1)
    criterion = 'pseudoon'
    # segment == 'sliding' -> sliding windows with slide = 1 s
    # segment == 'nooverlap' -> windows with no overlap. Only the first and, if present, the second, are considered
    segmenting = 'sliding'

    # FREQUENCY REDUCTION
    # global FREQUENCIES
    # FREQUENCIES = [6.4, 6.9, 8]
    # yi = np.zeros((y3.shape[0] - ITERATIONS * PERIOD * FS, y1.shape[1]))
    # NEW_LENGTH_ITERATION = len(FREQUENCIES) * PERIOD * FS
    # for i in range(ITERATIONS):
    #     buf = y3[PERIOD * FS + i * LENGTH_ITERATION : (i+1) * LENGTH_ITERATION,:]
    #     yi[i * NEW_LENGTH_ITERATION:i * NEW_LENGTH_ITERATION + len(buf)] = buf

    yi = y2

    for il, length in enumerate(winLengths):
        if verbose:
            print 'Window %d' % length
        if criterion == 'offline':
            actualWindow = length
        elif criterion == 'pseudoon':
            actualWindow = length - 1
            # actualWindow = length

        y = yi[:, np.in1d(config.SENSORS_SANDRA, ['O2', 'Oz', 'O1'])]

        # 3 channels CCA
        method = featex.CCA(list(FREQUENCIES), actualWindow * FS, FS)
        acc, und = apply_method(y, length, segmenting, criterion, method)

        accuracies[0, il] = acc
        undefineds[0, il] = und
        # ITRs[0, il] = performance.get_ITR(4, acc, und) * (60 / actualWindow)
        ITRs[0, il] = length / (1 - und)

        # 3 channels MSI
        method = featex.MSI(list(FREQUENCIES), actualWindow * FS, FS)
        acc, und = apply_method(y, length, segmenting, criterion, method)

        accuracies[1, il] = acc
        undefineds[1, il] = und
        # ITRs[1, il] = performance.get_ITR(4, acc, und) * (60 / actualWindow)
        ITRs[1, il] = length / (1 - und)

        # 2 channels MSI
        y = yi[:, np.in1d(config.SENSORS_SANDRA, ['O2', 'Oz'])]
        method = featex.MSI(list(FREQUENCIES), actualWindow * FS, FS)
        acc, und = apply_method(y, length, segmenting, criterion, method)

        accuracies[2, il] = acc
        undefineds[2, il] = und
        # ITRs[2, il] = performance.get_ITR(4, acc, und) * (60 / actualWindow)
        ITRs[2, il] = length / (1 - und)

        # 1 channel MSI
        y = yi[:, np.in1d(config.SENSORS_SANDRA, ['Oz'])]
        method = featex.MSI(list(FREQUENCIES), actualWindow * FS, FS)
        acc, und = apply_method(y, length, segmenting, criterion, method)

        accuracies[3, il] = acc
        undefineds[3, il] = und
        # ITRs[3, il] = performance.get_ITR(4, acc, und) * (60 / actualWindow)
        ITRs[3, il] = length / (1 - und)

        # 1 channel PSDA
        y = yi[:, np.in1d(config.SENSORS_SANDRA, ['Oz'])]
        method = featex.PSDA(list(FREQUENCIES), actualWindow * FS, FS)
        acc, und = apply_method(y, length, segmenting, criterion, method)

        accuracies[4, il] = acc
        undefineds[4, il] = und
        # ITRs[4, il] = performance.get_ITR(4, acc, und) * (60 / actualWindow)
        ITRs[4, il] = length / (1 - und)

    return accuracies, undefineds, ITRs


if __name__ == '__main__':
    FS = 600
    FREQUENCIES = [5.6, 6.4, 6.9, 8]

    # protocol part 2
    ITERATIONS = 4
    PERIOD = 10

    LENGTH_ITERATION = len(FREQUENCIES) * PERIOD * FS
    lengths = [2, 3, 4, 5]

    PLOT = True
    SAVE = False

    if PLOT:
        DATA_FILE = "protocolo 7/alessandra_prot7_config1.mat"
        a, u, i = process_file(DATA_FILE, lengths)

        fig = plt.figure(1)
        fig.suptitle(DATA_FILE)
        ax = plt.subplot(131)
        plt.ylim((25., 100.))
        plt.xlabel("Window length")
        plt.ylabel("Accuracy")
        plt.plot(lengths, a[0], '-or', label='3 ch CCA')
        plt.plot(lengths, a[1], '-oy', label='3 ch MSI')
        plt.plot(lengths, a[2], '-ob', label='2 ch MSI')
        plt.plot(lengths, a[3], '-om', label='1 ch MSI')
        plt.plot(lengths, a[4], '-og', label='1 ch PSDA')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.grid()
        ax = plt.subplot(132)
        plt.xlabel("Window length")
        plt.ylabel("Time for command")
        plt.plot(lengths, u[0], '-or', label='3 ch CCA')
        plt.plot(lengths, u[1], '-oy', label='3 ch MSI')
        plt.plot(lengths, u[2], '-ob', label='2 ch MSI')
        plt.plot(lengths, u[3], '-om', label='1 ch MSI')
        plt.plot(lengths, u[4], '-og', label='1 ch PSDA')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.grid()
        ax = plt.subplot(133)
        plt.xlabel("Window length")
        plt.ylabel("Time for command")
        plt.plot(lengths, i[0], '-or', label='3 ch CCA')
        plt.plot(lengths, i[1], '-oy', label='3 ch MSI')
        plt.plot(lengths, i[2], '-ob', label='2 ch MSI')
        plt.plot(lengths, i[3], '-om', label='1 ch MSI')
        plt.plot(lengths, i[4], '-og', label='1 ch PSDA')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.grid()

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0, -0.1),
                  fancybox=True, shadow=True, ncol=5)
        plt.show()

    if SAVE:
        data = np.zeros((len(config.SUBJECTS_SANDRA), 5, len(lengths)))

        for ii, name in enumerate(config.SUBJECTS_SANDRA):
            DATA_FILE = "protocolo 7/%s_prot7_config1.mat" % name

            _, _, c = process_file(DATA_FILE, lengths, False)
            print '##'
            data[ii, :, :] = c

        filename = "aggregated_compare.txt"
        data = data.reshape(len(config.SUBJECTS_SANDRA)*5, len(lengths))
        np.savetxt(filename, data, fmt="%.2f", delimiter=',')