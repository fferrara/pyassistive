 # -*- coding: utf-8 -*-
 #!/usr/bin/env python

import numpy as np
import scipy.io as sio
from scipy import signal
import os
from util import config, preprocessing, processing
import matplotlib.pyplot as plt

DATA_FILE = "protocolo 7/celso_prot7_config1.mat"


# read data
data = sio.loadmat(os.path.join(config.DATA_PATH, DATA_FILE))
X = data['data']

Fs = 600
window = 20 * Fs

# CAR FILTER
X -= X.mean(axis=0)
X = np.dot(X, preprocessing.CAR(X.shape[1]))
# mio segnale Oz
y = X[window:2*window, 11:12]

Wcritic = np.array([0., 4., 5., 49., 50., 300.])
b, a = signal.remez(851, Wcritic, [0, 1, 0], [5.75, 1., 5.75], Hz=Fs, type='hilbert',grid_density=20), 1.
y = signal.filtfilt(b, (a,), y, axis=0)


f, p = preprocessing.get_psd(y[:,0], 4, fs=Fs, plot=True)
plt.show()

import pickle
fp = open("shared5.6.pkl","w")
pickle.dump(y, fp)
exit()

freq = 5.6
time = np.arange(window, dtype=float) / Fs
x = np.empty((4, time.shape[0]))
x[0,:] = np.sin(2 * np.pi * freq * time)
x[1,:] = np.cos(2 * np.pi * freq * time)
x[2,:] = np.sin(4 * np.pi * freq * time)
x[3,:] = np.cos(4 * np.pi * freq * time)
# X is orthogonal <=> np.corrcoef(x.T) is diagonal

w = np.zeros((1, x.shape[0])).T

# Sufficient for stability: 0 < mu < 1 / tr(R)
# R = E[x * x.T]  = np.corrcoef(x.T)
mu = 0.001
filterlen = 10

MSE = np.empty((window))
W = np.empty((window, 4, 1))

for k in xrange(1, window):

    dPrime = np.dot(x[:,k-1:k].T, w)
    e = y[k] - dPrime
    w = w + 2 * mu * e * x[:,k-1:k]

    MSE[k] = e**2
    W[k] = w

plt.subplot(211)
plt.plot(W[:,:,0])
plt.subplot(212)
plt.plot(MSE)
plt.show()