""" lmsTest.py: a simple adaptive notch filter """

from __future__ import division
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import os
from ..util import config,preprocessing, featex
import matplotlib.pyplot as pl

#...............................................................................
if __name__ == "__main__":
    import sys

    modelorder = 1
    damp = .01
    nx = 600 * 3
    freq = 6.4  # chirp
    noise = .05 * 2  # * swing
    plot = 1
    seed = 0

    DATA_FILE = "protocolo 7/andre_prot7_config1.mat"

    # read data
    data = sio.loadmat(os.path.join(config.DATA_PATH, DATA_FILE))
    X = data['data']

    # CAR FILTER & PASSBAND FILTER
    X -= X.mean(axis=0)
    X = np.dot(X, preprocessing.CAR(X.shape[1]))

    Fs = 600

    Wcritic = np.array([0., 4., 5., 49., 50., 300.])
    b, a = preprocessing._get_fir_filter(Wcritic, Fs, 251)
    X = signal.lfilter(b, (a,), X, axis=0)

    channel = [11] # Oz
    y1 = X[4*Fs:20*Fs, channel].reshape(16*Fs, len(channel)) # protocol first part, 20s x 2

    # f,psd = preprocessing.get_psd(y1[:,0], 3, Fs, config.NFFT)
    # pl.plot(f,psd)
    # pl.show()

    t = np.arange( nx + 0. ) / 600.
    x = np.empty((2 * modelorder, t.shape[0]))
    x[0,:] = np.sin(2 * np.pi * freq * t)
    x[1,:] = np.cos(2 * np.pi * freq * t)
    # x[2,:] = np.sin(4 * np.pi * freq * t)
    # x[3,:] = np.cos(4 * np.pi * freq * t)

    title = "LMS  chirp  filterlen %d  nx %d  noise %.2g  mu %.2g " % (
        modelorder, nx, noise, damp )
    print title

#...............................................................................
    ys = []
    yests = []
    lms = featex.LMS( np.zeros(2 * modelorder), mu=damp )
    for t in xrange( nx ):
        X = x[:,t]
        y = y1[t,0]  # predict
        yest = lms._est( X, y)
        ys += [y]
        yests += [yest]

    y = np.array(ys)
    yest = np.array(yests)

    print yest.shape, y.shape
    err = (yest - y)**2
    averr = "av %.2g += %.2g" % (err.mean(), err.std())
    print "LMS yest - y:", averr
    print "LMS weights:", lms.W
    if plot == 2:
        fig, ax = pl.subplots( nrows=2 )
        fig.set_size_inches( 12, 8 )
        fig.suptitle( title, fontsize=12 )
        ax[0].plot( y, color="orangered", label="y" )
        ax[0].plot( yest, label="yest" )
        ax[0].legend()
        ax[1].plot( err, label=averr )
        ax[1].legend()
        pl.show()
    if plot == 1:
        Fxx, Pxx = preprocessing.get_psd(y, 3, Fs, nfft=config.NFFT)
        Fyy, Pyy = preprocessing.get_psd(yest, 3, Fs, nfft=config.NFFT)

        fig, ax = pl.subplots( nrows=2 )
        fig.set_size_inches( 12, 8 )
        fig.suptitle( title, fontsize=12 )
        ax[0].plot( Fxx, Pxx, color="orangered", label="y" )
        ax[0].legend()
        ax[1].plot( Fyy, Pyy, label="yest" )
        ax[1].legend()
        pl.show()

