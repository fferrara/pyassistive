 # -*- coding: utf-8 -*-
 #!/usr/bin/env python

import numpy as np
import scipy.io as sio
from scipy import signal
import os
from util import config, preprocessing, processing
from util.processing import Processing
import matplotlib.pyplot as plt

WINDOW_LENGTH = 5.
DATA_FILE = "emotiv_original_3.mat"
FREQUENCIES = [6.4, 8.] # SX, DX

# read data
data = sio.loadmat(os.path.join(config.DATA_PATH, DATA_FILE))
X = data['X'].astype('float32', copy=False)

# select channels and best channel for PSDA
channels = np.array(['F3','P7','O1','O2','P8','F4'])
X = X[:, np.in1d(np.array(config.SENSORS), channels)]
best = np.where(channels == 'O2')[0][0]

# CAR FILTER
X -= X.mean(axis=0)
X = np.dot(X, preprocessing.CAR(X.shape[1]))

# Filtro passabanda
Wcritic = np.array([0., 4., 5., 49., 50., 64.])
b, a = preprocessing.get_passband_filter(Wcritic, False)
X = signal.filtfilt(b, (a,), X, axis=0)

samples = WINDOW_LENGTH * config.FS
windows = preprocessing.sliding_window(X, (samples, X.shape[1]), (config.FS, X.shape[1]))

window = windows[0]
subs = preprocessing.sliding_window(window, (4 * config.FS, 6), (config.FS / 4, 6))
sub = subs[0]

print np.max(sub)