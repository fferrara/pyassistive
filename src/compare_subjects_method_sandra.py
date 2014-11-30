 # -*- coding: utf-8 -*-
 #!/usr/bin/env python

import numpy as np
import scipy.io as sio
from scipy import signal
import os
from util import config, preprocessing, processing, offline, performance
import matplotlib.pyplot as plt
import time

def apply_method(data, windowSize, segmenting, criterion, method):
    if segmenting == 'sliding':
        label = offline.make_label_windows(config.RECORDING_ITERATIONS, config.RECORDING_PERIOD, windowSize, len(FREQUENCIES))
        windows = offline.extract_windowed_segment(data, windowSize, config.RECORDING_ITERATIONS, config.RECORDING_PERIOD, config.FS)
    elif segmenting == 'nooverlap':
        label = offline.make_label_segments(config.RECORDING_ITERATIONS, config.RECORDING_PERIOD, windowSize, len(FREQUENCIES))
        windows = offline.extract_segment_start(data, windowSize, config.RECORDING_ITERATIONS, config.RECORDING_PERIOD, config.FS)

    if criterion == 'offline':
        o = offline.offline_classify(windows, FREQUENCIES, method)
        cm = performance.get_confusion_matrix(label, o, len(FREQUENCIES))
        return 100 * performance.get_accuracy(cm), 0
        # return performance.get_cohen_k(cm), 0
    elif criterion == 'pseudoon':
        o, c, m = offline.pseudo_online_classify(windows, FREQUENCIES, config.FS, method, pause=0) 
        cm = performance.get_confusion_matrix(label, o, len(FREQUENCIES))
        return 100 * performance.get_accuracy(cm), (m + 0.) / (m + c)   
        
def perform(name):
    """
    Perform preprocessing and classification using one method and one window length
    Return a matrix: acc = accuracy und = undefinedRate
    [20 acc20 und20
     10 acc10 und10
     5 acc5 und5]
    """
    # read data
    data = sio.loadmat(os.path.join(config.DATA_PATH, name))
    X = data['data']

    # CAR FILTER & PASSBAND FILTER
    X -= X.mean(axis=0)
    X = np.dot(X, preprocessing.CAR(X.shape[1]))

    Fs = 600
    WINLENGTH = 3

    Wcritic = np.array([0., 4., 5., 49., 50., 300.])
    b, a = preprocessing._get_fir_filter(Wcritic, Fs, 251)
    X = signal.lfilter(b, (a,), X, axis=0)

    channels = [11] # Oz
    y1 = X[0:160*Fs, channels].reshape(160*Fs, len(channels)) # protocol first part, 20s x 2
    y2 = X[160*Fs: 320*Fs, channels].reshape(160*Fs, len(channels)) # protocol second part, 10s x 4
    y3 = X[320*Fs: 420*Fs, channels].reshape(100*Fs, len(channels)) # protocol first part, 5s x 5

    # Plotting spectrum to observe power distribution
    # f,psd = preprocessing.get_psd(y1[:,0], 3, Fs)
    # plt.plot(f,psd)
    # plt.show()

    l1 = offline.make_label_matrix(y1.shape[0], period=20, window=WINLENGTH, fs=Fs, classes=4)
    l2 = offline.make_label_matrix(y2.shape[0], period=10, window=WINLENGTH, fs=Fs, classes=4)
    l3 = offline.make_label_matrix(y3.shape[0], period=5, window=WINLENGTH, fs=Fs, classes=4)

    freqs = [5.6, 6.4, 6.9, 8]
    # method = processing.FLC(freqs, (WINLENGTH -1) * Fs, order=2, mu=0.005)
    method = processing.MSI(freqs, (WINLENGTH - 1) * Fs, Fs)

    records = np.zeros((3, 3))

    windows = preprocessing.sliding_window(y1, (WINLENGTH * Fs, y1.shape[1]), (Fs, y1.shape[1]))
    o, count, missed = offline.pseudo_online_classify(windows, freqs, Fs, method, 0)
    #o = offline.offline_classify(windows, freqs, method)
    acc = performance.get_accuracy(l1, o, count)
    undefinedRate = (missed + 0.) / (missed + count)
    itr = performance.get_ITR(4, acc, undefinedRate) * 60
    records[0,:] = [20, acc, itr]
    print '##'

    windows = preprocessing.sliding_window(y2, (WINLENGTH * Fs, y2.shape[1]), (Fs, y2.shape[1]))
    o, count, missed = offline.pseudo_online_classify(windows, freqs, Fs, method, 0)
    # o = offline.offline_classify(windows, freqs, method)
    acc = performance.get_accuracy(l2, o, count)
    undefinedRate = (missed + 0.) / (missed + count)
    itr = performance.get_ITR(4, acc, undefinedRate) * 60
    records[1,:] = [10, acc, itr]
    print '##'

    windows = preprocessing.sliding_window(y3, (WINLENGTH * Fs, y3.shape[1]), (Fs, y3.shape[1]))
    o, count, missed = offline.pseudo_online_classify(windows, freqs, Fs, method, 0)
    # o = offline.offline_classify(windows, freqs, method)
    acc = performance.get_accuracy(l3, o, count)
    undefinedRate = (missed + 0.) / (missed + count)
    itr = performance.get_ITR(4, acc, undefinedRate) * 60
    records[2,:] = [5, acc, itr]
    print '##'

    return records

if __name__ == '__main__':
    data = np.zeros((len(config.SUBJECTS),3,3))

    for ii, name in enumerate(config.SUBJECTS):
        DATA_FILE = "protocolo 7/%s_prot7_config1.mat" % name

        data[ii, :, :] = perform(DATA_FILE)

    filename = "aggregated_MSI_filtfilt_Oz.txt"

    data = data.reshape(len(config.SUBJECTS)*3, 3)
    np.savetxt(filename, data, fmt="%.2f", delimiter=',')
