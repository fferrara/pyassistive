 # -*- coding: utf-8 -*-
 #!/usr/bin/env python

import numpy as np
import scipy.io as sio
from scipy import signal
import os
from util import config, preprocessing, processing, offline, performance
import matplotlib.pyplot as plt

def apply_method(data, windowSize, segmenting, criterion, method):
    if segmenting == 'sliding':
        label = offline.make_label_windows(ITERATIONS, PERIOD, windowSize, len(FREQUENCIES))
        windows = offline.extract_windowed_segment(data, windowSize, ITERATIONS, PERIOD, FS)
    elif segmenting == 'nooverlap':
        label = offline.make_label_segments(ITERATIONS, PERIOD, windowSize, len(FREQUENCIES))
        windows = offline.extract_segment_start(data, windowSize, ITERATIONS, PERIOD, FS)

    if criterion == 'offline':
        o = offline.offline_classify(windows, FREQUENCIES, method)
        cm = performance.get_confusion_matrix(label, o, len(FREQUENCIES))
        return 100 * performance.get_accuracy(cm), 0
        # return performance.get_cohen_k(cm), 0
    elif criterion == 'pseudoon':
        o, c, m = offline.pseudo_online_classify(windows, FREQUENCIES, FS, method, pause=0) 
        cm = performance.get_confusion_matrix(label, o, len(FREQUENCIES))
        return 100 * performance.get_accuracy(cm), (m + 0.) / (m + c)   

DATA_FILE = "protocolo 7/fernando_prot7_config1.mat"
FREQUENCIES = [5.6, 6.4, 6.9, 8] 
ITERATIONS = 4
PERIOD = 10
FS = 600
LENGTH_ITERATION = len(FREQUENCIES) * PERIOD * FS

# read data
data = sio.loadmat(os.path.join(config.DATA_PATH, DATA_FILE))
X = data['data']

# CAR FILTER & PASSBAND FILTER
Wcritic = np.array([0., 4., 5., 49., 50., 300.])
b, a = preprocessing._get_fir_filter(Wcritic, FS, 851)
# X = signal.lfilter(b, [a,], X, axis=0)[len(b) - 1:,:]
X = signal.fftconvolve(X, b[:, np.newaxis], mode='valid')

X -= X.mean(axis=0)
X = np.dot(X, preprocessing.CAR(X.shape[1]))


y1 = X[0:160*FS,:] # protocol first part, 20s x 2
y2 = X[160*FS: 320*FS, :] # protocol second part, 10s x 4
y3 = X[320*FS: 420*FS, :] # protocol first part, 5s x 5

accuracies = []
[accuracies.append([]) for i in xrange(5)]
ITRs = []
[ITRs.append([]) for i in xrange(5)]
undefineds = []
[undefineds.append([]) for i in xrange(5)]
lengths = [1, 2, 3, 4, 5]

# Comparison parameters
# criterion == 'offline' -> classifier just a criterion of maxima. Windows->Output 1:1 
# criterion == 'pseudoon' -> classifier with a confidence criterion. Windows->Output 1:(0|1) 
criterion = 'offline'
# segment == 'sliding' -> sliding windows with slide = 1 s
# segment == 'nooverlap' -> windows with no overlap. Only the first and, if present, the second, are considered
segmenting = 'sliding'

# FREQUENCY REDUCTION
FREQUENCIES = [6.4, 6.9, 8] 
yi = np.zeros((y2.shape[0] - ITERATIONS * PERIOD * FS, y2.shape[1]))
NEW_LENGTH_ITERATION = len(FREQUENCIES) * PERIOD * FS
for i in range(ITERATIONS):
    buf = y2[PERIOD * FS + i * LENGTH_ITERATION : (i+1) * LENGTH_ITERATION,:]
    yi[i * NEW_LENGTH_ITERATION:i * NEW_LENGTH_ITERATION + len(buf)] = buf

# yi = y2

for length in lengths:
    print 'Window %d' % length
    if criterion == 'offline':
        actualWindow = length
    elif criterion == 'pseudoon':
        actualWindow = length - 1

    y = yi[:, np.in1d(config.SENSORS_SANDRA, ['O2', 'Oz', 'O1'])]

    #### 3 channels CCA
    method = processing.CCA(list(FREQUENCIES), actualWindow * FS, FS)
    acc, und = apply_method(y, actualWindow, segmenting, criterion, method)
    
    accuracies[0].append( acc )
    undefineds[0].append( und )
    ITRs[0].append( performance.get_ITR(4,acc, und) * (60 / actualWindow) )
    #### 3 channels MSI
    method = processing.MSI(list(FREQUENCIES), actualWindow * FS, FS)
    acc, und = apply_method(y, actualWindow, segmenting, criterion, method)
    
    accuracies[1].append( acc )
    undefineds[1].append( und )
    ITRs[1].append( performance.get_ITR(4,acc, und) * (60 / actualWindow) )

    #### 2 channels MSI
    y = yi[:, np.in1d(config.SENSORS_SANDRA, ['O2', 'Oz'])]
    method = processing.MSI(list(FREQUENCIES), actualWindow * FS, FS)
    acc, und = apply_method(y, actualWindow, segmenting, criterion, method)
    
    accuracies[2].append( acc )
    undefineds[2].append( und )
    ITRs[2].append( performance.get_ITR(4,acc, und) * (60 / actualWindow) )

    #### 1 channel MSI
    y = yi[:, np.in1d(config.SENSORS_SANDRA, ['Oz'])]
    method = processing.MSI(list(FREQUENCIES), actualWindow * FS, FS)
    acc, und = apply_method(y, actualWindow, segmenting, criterion, method)
    
    accuracies[3].append( acc )
    undefineds[3].append( und )
    ITRs[3].append( performance.get_ITR(4,acc, und) * (60 / actualWindow) )

    #### 1 channel PSDA
    y = yi[:, np.in1d(config.SENSORS_SANDRA, ['Oz'])]
    method = processing.MSI(list(FREQUENCIES), actualWindow * FS, FS)
    acc, und = apply_method(y, actualWindow, segmenting, criterion, method)
    
    accuracies[4].append( acc )
    undefineds[4].append( und )
    ITRs[4].append( performance.get_ITR(4,acc, und) * (60 / actualWindow) )


fig = plt.figure(1)
fig.suptitle(DATA_FILE)
ax = plt.subplot(121)
plt.ylim((25., 100.))
plt.xlabel("Window length")
plt.ylabel("Accuracy")
plt.plot(lengths, accuracies[0], '-or', label='3 ch CCA')
plt.plot(lengths, accuracies[1], '-oy', label='3 ch MSI')
plt.plot(lengths, accuracies[2], '-ob', label='2 ch MSI')
plt.plot(lengths, accuracies[3], '-oc', label='1 ch MSI')
plt.plot(lengths, accuracies[4], '-og', label='1 ch PSDA')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax.grid()
ax = plt.subplot(122)
plt.xlabel("Window length")
plt.ylabel("Time for command")
plt.plot(lengths, undefineds[0], 'r', label='3 ch CCA')
plt.plot(lengths, undefineds[1], 'y', label='3 ch MSI')
plt.plot(lengths, undefineds[2], 'b', label='2 ch MSI')
plt.plot(lengths, undefineds[3], 'c', label='1 ch MSI')
plt.plot(lengths, undefineds[4], 'g', label='1 ch PSDA')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax.grid()

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0, -0.1),
          fancybox=True, shadow=True, ncol=5)
plt.show()
