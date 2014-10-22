 # -*- coding: utf-8 -*-
 #!/usr/bin/env python

import numpy as np
import scipy.io as sio
from scipy import signal
import os
from util import config, preprocessing, processing
import matplotlib.pyplot as plt

DATA_FILE = "emotiv_original_newstimuli.mat"
FREQUENCIES = [8., 10.] # SX, DX

# read data
data = sio.loadmat(os.path.join(config.DATA_PATH, DATA_FILE))
X = data['X'].astype('float32', copy=False)

# select channels and best channel for PSDA
channels = np.array(['F3','P7','O1','O2','P8','F4'])
X = X[:, np.in1d(np.array(config.SENSORS), channels)]
#best = np.where(channels == 'O2')[0][0]

# CAR FILTER
X -= X.mean(axis=0)
X = np.dot(X, preprocessing.CAR(X.shape[1]))

# Filtro passabanda
Wcritic = np.array([0., 4., 5., 49., 50., 64.])
b, a = preprocessing.get_passband_filter(Wcritic, False)
X = signal.filtfilt(b, (a,), X, axis=0)

# pseudo online classification
def make_label_matrix(window):
    label = np.zeros((X.shape[0] - window * config.FS) / config.FS + 1)
    firstIndex = ((config.RECORDING_PERIOD * config.FS) - window * config.FS) / config.FS + 1
    for i in xrange(config.RECORDING_ITERATIONS):
        label[firstIndex + 2*i*config.RECORDING_PERIOD:firstIndex + (2*i + 1) * config.RECORDING_PERIOD] = 1

    return label

def pseudo_online_classify(data, window, label, frequencies, method):
    """
    Classify a recorded dataset data, according to label
    Data must be an array of sliding windows, with first dimension equal to window * FS.
    Frequencies must be a list. Values in label must represent valid indexes in the list
    Method must offer a perform(data, samples) method
    """
    errors = 0. # misclassified
    delay = 0.  # n° of windows for output
    missed = 0  # n° of windows without output
    count = 0  # n° of output provided

    STEP = 4  # 5 subwindows for each window

    # Windowing
    windows = preprocessing.sliding_window(data, (window * config.FS, data.shape[1]), (config.FS, X.shape[1]))
    iterator = enumerate(windows)
    subSamples = (window - 1) * config.FS

    for ii, win in iterator:
        subWindows = preprocessing.sliding_window(win, (subSamples, win.shape[1]), (np.floor(config.FS / STEP), win.shape[1]))

        temp = []
        for subWindow in subWindows:
            temp.append(method.perform(subWindow, window - 1))


        for i in range(len(frequencies)):
            if float(temp.count(frequencies[i])) / len(temp) >= 0.9:
                count += 1
                delay += window + missed
                out = i

                if label[ii] != out: errors += 1

                missed = 0
                for i in xrange(config.WAIT_INTER_COMMANDS + int(window)):
                    try:
                        iterator.next()
                    except StopIteration:
                        break # game over
                break # important to the else branch not be executed
        else:
            missed += 1

    return 100 * (1 - errors/count), delay/count

accuracies = []
[accuracies.append([]) for i in xrange(3)]
delays = []
[delays.append([]) for i in xrange(3)]
tries = [3, 4, 5, 6, 7, 8, 9]
channel = [3,3]

for i in tries:
    label = make_label_matrix(i)

    method = processing.CCA(list(FREQUENCIES), i - 1)
    a, b = pseudo_online_classify(X[:, channel[0]:channel[1]+1], i, label, FREQUENCIES, method)
    accuracies[0].append(a)
    delays[0].append(b)

    method = processing.MSI(list(FREQUENCIES), i - 1)
    a, b = pseudo_online_classify(X[:, channel[0]:channel[1]+1], i, label, FREQUENCIES, method)
    accuracies[1].append(a)
    delays[1].append(b)

    method = processing.PSDA(list(FREQUENCIES), i - 1)
    a, b = pseudo_online_classify(X[:, channel[0]:channel[1]+1], i, label, FREQUENCIES, method)
    accuracies[2].append(a)
    delays[2].append(b)

fig = plt.figure(1)
fig.suptitle(DATA_FILE + ' ' + ":".join(channels[channel]))
plt.subplot(121)
plt.ylim((50., 100.))
plt.xlabel("Window length")
plt.ylabel("Accuracy")
plt.plot(tries, accuracies[0], 'r', label='CCA')
plt.plot(tries, accuracies[1], 'y', label='MSI')
plt.plot(tries, accuracies[2], 'g', label='PSDA')
plt.legend()
plt.grid()
plt.subplot(122)
plt.xlabel("Window length")
plt.ylabel("Time for command")
plt.plot(tries, delays[0], 'r', label='CCA')
plt.plot(tries, delays[1], 'y', label='MSI')
plt.plot(tries, delays[2], 'g', label='PSDA')
plt.legend()
plt.grid()
plt.show()
